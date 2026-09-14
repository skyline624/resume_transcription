"""Registre local des personnes : empreintes uniquement, jamais l'audio source."""

import json
import math
import re
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from transcription_server.diarization.engine import SpeakerEmbedding

if TYPE_CHECKING:
    from transcription_server.pipeline import TranscriptionResult


def clean_name(name: str) -> str:
    if any(unicodedata.category(char).startswith("C") for char in name):
        raise ValueError("Le nom ne doit pas contenir de caractères de contrôle.")
    name = unicodedata.normalize("NFC", " ".join(name.split()))
    if not 1 <= len(name) <= 100 or re.fullmatch(r"SPEAKER_\d+", name, re.I):
        raise ValueError("Fournissez un nom de 1 à 100 caractères, distinct de SPEAKER_XX.")
    return name


def _normalized(embedding: SpeakerEmbedding, minimum_s: float):
    vector = np.asarray(embedding.values, dtype=np.float64)
    if (vector.ndim != 1 or not 1 <= vector.size <= 4096
            or not np.isfinite(vector).all()
            or not math.isfinite(embedding.speech_s)
            or embedding.speech_s < minimum_s):
        return None
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-8:
        return None
    return (vector / norm).tolist()


class SpeakerRegistry:
    """SQLite transactionnel ; connexions courtes utilisables depuis les threads.

    Les références nommées sont persistantes. Les empreintes à nommer expirent
    après 24 h par défaut, avec un maximum de 256 transcriptions en attente.
    Une personne peut avoir cinq références, toutes ajoutées explicitement.
    Les correspondances automatiques n'enrichissent jamais le registre.
    """

    def __init__(self, root: Path, threshold=0.75, margin=0.08,
                 minimum_s=3.0, pending_ttl_s=86400.0):
        self.root = Path(root)
        self.threshold = threshold
        self.margin = margin
        self.minimum_s = minimum_s
        self.pending_ttl_s = pending_ttl_s

    @contextmanager
    def _database(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.root / "speakers.sqlite3"
        connection = sqlite3.connect(path, timeout=30)
        try:
            path.chmod(0o600)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    transcription_id TEXT NOT NULL, label TEXT NOT NULL,
                    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, vector TEXT NOT NULL, created REAL NOT NULL,
                    PRIMARY KEY (transcription_id, label)
                );
                CREATE TABLE IF NOT EXISTS pending (
                    id TEXT PRIMARY KEY, created REAL NOT NULL, payload TEXT NOT NULL
                );
            """)
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM pending WHERE created <= ?",
                                   (time.time() - self.pending_ttl_s,))
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _person(db, person_id):
        row = db.execute("""
            SELECT p.id, p.name, COUNT(s.label) AS samples
            FROM persons p LEFT JOIN samples s ON s.person_id = p.id
            WHERE p.id = ? GROUP BY p.id
        """, (person_id,)).fetchone()
        if row is None:
            raise KeyError("Personne inconnue.")
        return dict(row)

    def list(self):
        with self._database() as db:
            return [self._person(db, row[0]) for row in
                    db.execute("SELECT id FROM persons ORDER BY name_key").fetchall()]

    def identify(self, transcription_id: str, speaker: str, name: str):
        name = clean_name(name)
        with self._database() as db:
            row = db.execute("SELECT payload FROM pending WHERE id = ?",
                             (transcription_id,)).fetchone()
            if row is None:
                raise KeyError("Transcription inconnue ou expirée. Transcrivez de nouveau l'audio.")
            payload = json.loads(row[0])
            if speaker not in payload["speakers"]:
                raise KeyError("Ce label de locuteur n'existe pas dans cette transcription.")
            reference = payload["embeddings"].get(speaker)
            if reference is None:
                raise ValueError(
                    f"Empreinte indisponible : il faut au moins {self.minimum_s:g} secondes "
                    "de parole sans chevauchement et un modèle fournissant des empreintes."
                )
            existing = db.execute("SELECT id FROM persons WHERE name_key = ?",
                                  (name.casefold(),)).fetchone()
            if existing:
                person_id = existing[0]
            else:
                if db.execute("SELECT COUNT(*) FROM persons").fetchone()[0] >= 1000:
                    raise ValueError("Le registre est limité à 1000 personnes.")
                person_id = str(uuid4())
                db.execute("INSERT INTO persons VALUES (?, ?, ?)",
                           (person_id, name, name.casefold()))
            # Nommer à nouveau le même extrait corrige son attribution.
            db.execute("""
                INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(transcription_id, label) DO UPDATE SET
                    person_id=excluded.person_id, model=excluded.model,
                    vector=excluded.vector
            """, (transcription_id, speaker, person_id, reference["model"],
                  json.dumps(reference["vector"]), time.time()))
            db.execute("""
                DELETE FROM samples WHERE person_id = ? AND rowid NOT IN (
                    SELECT rowid FROM samples WHERE person_id = ?
                    ORDER BY created DESC, rowid DESC LIMIT 5
                )
            """, (person_id, person_id))
            db.execute("DELETE FROM persons WHERE id NOT IN (SELECT person_id FROM samples)")
            return self._person(db, person_id)

    def rename(self, person_id: str, name: str):
        name = clean_name(name)
        with self._database() as db:
            self._person(db, person_id)
            try:
                db.execute("UPDATE persons SET name = ?, name_key = ? WHERE id = ?",
                           (name, name.casefold(), person_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Une autre personne porte déjà ce nom.") from exc
            return self._person(db, person_id)

    def delete(self, person_id: str):
        with self._database() as db:
            self._person(db, person_id)
            db.execute("DELETE FROM persons WHERE id = ?", (person_id,))

    def recognize(self, result: "TranscriptionResult",
                  embeddings: dict[str, SpeakerEmbedding]) -> "TranscriptionResult":
        references = {}
        for label in result.speakers[:256]:
            embedding = embeddings.get(label)
            if embedding is not None:
                vector = _normalized(embedding, self.minimum_s)
                if vector is not None:
                    references[label] = {"model": embedding.model, "vector": vector}
        if not result.speakers:
            return result
        transcription_id = str(uuid4())
        matches = {}
        with self._database() as db:
            samples = db.execute("""
                SELECT p.id, p.name, s.model, s.vector FROM samples s
                JOIN persons p ON p.id = s.person_id
            """).fetchall()
            decoded = [(row["id"], row["name"], row["model"],
                        np.asarray(json.loads(row["vector"]))) for row in samples]
            for label in result.speakers:
                reference = references.get(label)
                match = {"speaker_id": None, "name": None, "score": None,
                         "enrollable": reference is not None}
                scores = {}
                if reference is not None:
                    vector = np.asarray(reference["vector"])
                    for person_id, name, model, sample in decoded:
                        if model == reference["model"] and vector.shape == sample.shape:
                            score = float(np.clip(np.dot(vector, sample), -1, 1))
                            if person_id not in scores or score > scores[person_id][0]:
                                scores[person_id] = (score, person_id, name)
                ranked = sorted(scores.values(), reverse=True)
                if ranked:
                    best = ranked[0]
                    # Comparaison entre PERSONNES : deux références d'une même
                    # personne ne constituent pas une ambiguïté.
                    if best[0] >= self.threshold and (
                        len(ranked) == 1 or best[0] - ranked[1][0] >= self.margin
                    ):
                        match.update(speaker_id=best[1], name=best[2], score=round(best[0], 4))
                matches[label] = match
            db.execute("INSERT INTO pending VALUES (?, ?, ?)",
                       (transcription_id, time.time(), json.dumps(
                           {"speakers": result.speakers, "embeddings": references})))
            db.execute("""
                DELETE FROM pending WHERE rowid NOT IN (
                    SELECT rowid FROM pending ORDER BY created DESC, rowid DESC LIMIT 256
                )
            """)
        names = {label: match["name"] or label for label, match in matches.items()}
        return replace(
            result, transcription_id=transcription_id, speaker_matches=matches,
            speakers=list(dict.fromkeys(names[label] for label in result.speakers)),
            turns=[replace(turn, speaker=names.get(turn.speaker, turn.speaker),
                           speaker_label=turn.speaker) for turn in result.turns],
        )
