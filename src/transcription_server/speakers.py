"""Registre local des personnes : empreintes uniquement, jamais l'audio source."""

import json
import math
import re
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from transcription_server.diarization.engine import SpeakerEmbedding

if TYPE_CHECKING:
    from transcription_server.pipeline import TranscriptionResult


#: Les empreintes sont les centroides de pyannote 4 ; ce suffixe distingue
#: leur espace de celui d'un autre calcul sur le meme modele.
CENTROIDS_TAG = "pyannote4-centroids-v1"
#: Prefixe des personnes creees automatiquement pour une voix inconnue.
AUTO_PREFIX = "Voix"
MAX_PERSONS = 1000


def embedding_model_id(diarization_model: str) -> str:
    """Etiquette des empreintes produites par ce modele de diarization."""
    return f"{diarization_model}|{CENTROIDS_TAG}"


def export_embeddings(labels, embeddings: dict[str, SpeakerEmbedding], minimum_s: float):
    """Empreintes d'une transcription, pour un client qui les garde lui-meme.

    Chaque label a sa duree de parole ; le vecteur (normalise, comme le
    registre le stocke) est nul quand le registre le refuserait.
    """
    exported = {}
    for label in labels:
        embedding = embeddings.get(label)
        if embedding is None:
            continue
        vector = _normalized(embedding, minimum_s)
        speech_s = embedding.speech_s if math.isfinite(embedding.speech_s) else 0.0
        exported[label] = {"model": embedding.model, "vector": vector,
                           "speech_s": round(float(speech_s), 3)}
    return exported


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

    Un client qui garde lui-même les empreintes de ses enregistrements
    (TalkTrack) les rattache par `add_reference`, sous une clé (source,
    label) à lui, sans limite de durée ; une fusion laisse un alias pour que
    ses liens vers l'ancienne personne restent valables.
    """

    def __init__(self, root: Path, threshold=0.75, margin=0.08,
                 minimum_s=3.0, pending_ttl_s=86400.0, max_references=5):
        self.root = Path(root)
        self.threshold = threshold
        self.margin = margin
        self.minimum_s = minimum_s
        self.pending_ttl_s = pending_ttl_s
        self.max_references = max_references

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
                CREATE TABLE IF NOT EXISTS aliases (
                    old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(persons)")}
            if "auto" not in columns:
                # Registre d'avant les voix automatiques : toutes ses personnes sont nommees.
                try:
                    connection.execute(
                        "ALTER TABLE persons ADD COLUMN auto INTEGER NOT NULL DEFAULT 0")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc):  # une connexion concurrente l'a fait
                        raise
                connection.commit()
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
            SELECT p.id, p.name, COUNT(s.label) AS samples, p.auto AS auto,
                   MAX(s.created) AS last_seen
            FROM persons p LEFT JOIN samples s ON s.person_id = p.id
            WHERE p.id = ? GROUP BY p.id
        """, (person_id,)).fetchone()
        if row is None:
            raise KeyError("Personne inconnue.")
        person = dict(row)
        person["auto"] = bool(person["auto"])
        if person["last_seen"] is not None:
            person["last_seen"] = datetime.fromtimestamp(
                person["last_seen"], tz=timezone.utc).isoformat(timespec="seconds")
        return person

    def _cap(self, db, person_id):
        """Les references les plus recentes de la personne, au plus max_references."""
        db.execute("""
            DELETE FROM samples WHERE person_id = ? AND rowid NOT IN (
                SELECT rowid FROM samples WHERE person_id = ?
                ORDER BY created DESC, rowid DESC LIMIT ?
            )
        """, (person_id, person_id, self.max_references))

    @staticmethod
    def _drop_empty(db):
        """Une personne sans reference ne peut plus etre reconnue : elle disparait."""
        gone = [row[0] for row in db.execute(
            "SELECT id FROM persons WHERE id NOT IN (SELECT person_id FROM samples)")]
        for person_id in gone:
            db.execute("DELETE FROM aliases WHERE new_id = ?", (person_id,))
            db.execute("DELETE FROM persons WHERE id = ?", (person_id,))

    @staticmethod
    def _create_person(db, name, auto=False):
        if db.execute("SELECT COUNT(*) FROM persons").fetchone()[0] >= MAX_PERSONS:
            raise ValueError(f"Le registre est limité à {MAX_PERSONS} personnes.")
        person_id = str(uuid4())
        db.execute("INSERT INTO persons (id, name, name_key, auto) VALUES (?, ?, ?, ?)",
                   (person_id, name, name.casefold(), int(auto)))
        return person_id

    @staticmethod
    def _next_auto_number(db):
        row = db.execute("SELECT value FROM meta WHERE key = 'next_auto_number'").fetchone()
        return int(row[0]) if row else 1

    @staticmethod
    def _set_next_auto_number(db, number):
        db.execute("INSERT INTO meta VALUES ('next_auto_number', ?) "
                   "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (str(number),))

    def _create_auto_person(self, db):
        """Une « Voix N » au numero suivant, jamais reattribue, meme apres suppression."""
        number = self._next_auto_number(db)
        while db.execute("SELECT 1 FROM persons WHERE name_key = ?",
                         (f"{AUTO_PREFIX} {number}".casefold(),)).fetchone():
            number += 1
        person_id = self._create_person(db, f"{AUTO_PREFIX} {number}", auto=True)
        self._set_next_auto_number(db, number + 1)
        return person_id

    @staticmethod
    def _resolve(db, person_id):
        """L'id actuel d'une personne, en suivant un alias de fusion."""
        row = db.execute("SELECT new_id FROM aliases WHERE old_id = ?", (person_id,)).fetchone()
        return row[0] if row else person_id

    def _person_by_name(self, db, name):
        row = db.execute("SELECT id FROM persons WHERE name_key = ?",
                         (name.casefold(),)).fetchone()
        return row[0] if row else None

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
            person_id = self._person_by_name(db, name) or self._create_person(db, name)
            # Nommer à nouveau le même extrait corrige son attribution.
            db.execute("""
                INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(transcription_id, label) DO UPDATE SET
                    person_id=excluded.person_id, model=excluded.model,
                    vector=excluded.vector
            """, (transcription_id, speaker, person_id, reference["model"],
                  json.dumps(reference["vector"]), time.time()))
            self._cap(db, person_id)
            self._drop_empty(db)
            return self._person(db, person_id)

    def info(self):
        return {"threshold": self.threshold, "margin": self.margin,
                "min_speech_s": self.minimum_s, "max_references": self.max_references}

    def add_reference(self, source: str, speaker: str, model: str, vector,
                      speech_s=None, name=None, person_id=None, auto=False):
        """Rattache l'empreinte (source, speaker) a une personne ; la personne.

        Une seule cible : un nom (personne existante ou nouvelle), un id, ou
        une nouvelle voix automatique. Renvoyer la meme cle vers une autre
        personne deplace la reference, comme `identify`.
        """
        if sum((name is not None, person_id is not None, bool(auto))) != 1:
            raise ValueError("Indiquez une seule cible : name, person_id ou auto.")
        reference = _normalized(SpeakerEmbedding(
            model, tuple(vector), self.minimum_s if speech_s is None else speech_s),
            self.minimum_s)
        if reference is None:
            raise ValueError(
                f"Empreinte inutilisable : il faut un vecteur fini et au moins "
                f"{self.minimum_s:g} secondes de parole sans chevauchement.")
        if name is not None:
            name = clean_name(name)
        with self._database() as db:
            if auto:
                target = self._create_auto_person(db)
            elif person_id is not None:
                target = self._resolve(db, person_id)
                self._person(db, target)
            else:
                target = self._person_by_name(db, name) or self._create_person(db, name)
            db.execute("""
                INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(transcription_id, label) DO UPDATE SET
                    person_id=excluded.person_id, model=excluded.model,
                    vector=excluded.vector, created=excluded.created
            """, (source, speaker, target, model, json.dumps(reference), time.time()))
            self._cap(db, target)
            self._drop_empty(db)
            return self._person(db, target)

    def remove_reference(self, source: str, speaker: str):
        """Detache une empreinte ; sans effet si elle n'existe pas."""
        with self._database() as db:
            db.execute("DELETE FROM samples WHERE transcription_id = ? AND label = ?",
                       (source, speaker))
            self._drop_empty(db)

    def merge(self, person_id: str, into: str):
        """Les references de person_id passent a `into`, qui garde son nom."""
        with self._database() as db:
            self._person(db, person_id)
            into = self._resolve(db, into)
            self._person(db, into)
            if into == person_id:
                raise ValueError("Une personne ne peut pas être fusionnée avec elle-même.")
            db.execute("UPDATE samples SET person_id = ? WHERE person_id = ?", (into, person_id))
            db.execute("UPDATE aliases SET new_id = ? WHERE new_id = ?", (into, person_id))
            db.execute("INSERT INTO aliases VALUES (?, ?) "
                       "ON CONFLICT(old_id) DO UPDATE SET new_id = excluded.new_id",
                       (person_id, into))
            db.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            self._cap(db, into)
            return self._person(db, into)

    def aliases(self):
        with self._database() as db:
            return {row[0]: row[1] for row in db.execute("SELECT old_id, new_id FROM aliases")}

    def import_people(self, model: str, persons, next_auto_number=None):
        """Importe une bibliotheque d'empreintes ; {cle du client: personne}.

        Une seule transaction, relancable : un meme nom designe la meme
        personne, une meme cle (source, label) la meme reference. Les
        empreintes ont deja ete acceptees par leur client : pas de duree minimale.
        """
        prepared = []
        for person in persons:
            name = clean_name(person["name"])
            references = []
            for reference in person.get("references", []):
                vector = _normalized(SpeakerEmbedding(model, tuple(reference["vector"]), 0.0), 0.0)
                if vector is None:
                    raise ValueError(f"Empreinte inutilisable pour « {name} ».")
                references.append((reference["source"], reference["speaker"], vector))
            prepared.append((person["key"], name, bool(person.get("auto")), references))
        imported = {}
        with self._database() as db:
            if next_auto_number:
                self._set_next_auto_number(db, max(self._next_auto_number(db),
                                                   int(next_auto_number)))
            for key, name, auto, references in prepared:
                person_id = (self._person_by_name(db, name)
                             or self._create_person(db, name, auto=auto))
                now = time.time()
                for source, speaker, vector in references:
                    db.execute("""
                        INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(transcription_id, label) DO UPDATE SET
                            person_id=excluded.person_id, model=excluded.model,
                            vector=excluded.vector
                    """, (source, speaker, person_id, model, json.dumps(vector), now))
                self._cap(db, person_id)
                imported[key] = person_id
            self._drop_empty(db)
            return {key: self._person(db, person_id) for key, person_id in imported.items()
                    if db.execute("SELECT 1 FROM persons WHERE id = ?",
                                  (person_id,)).fetchone()}

    def rename(self, person_id: str, name: str):
        name = clean_name(name)
        with self._database() as db:
            self._person(db, person_id)
            try:
                db.execute("UPDATE persons SET name = ?, name_key = ?, auto = 0 WHERE id = ?",
                           (name, name.casefold(), person_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Une autre personne porte déjà ce nom.") from exc
            return self._person(db, person_id)

    def delete(self, person_id: str):
        with self._database() as db:
            self._person(db, person_id)
            db.execute("DELETE FROM aliases WHERE new_id = ? OR old_id = ?",
                       (person_id, person_id))
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
