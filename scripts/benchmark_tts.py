"""Benchmark reproductible de Qwen3-TTS et d'exports VoxMind."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import time
import unicodedata
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class CorpusItem:
    id: str
    category: str
    text: str


def load_corpus(path: Path) -> list[CorpusItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Le corpus TTS doit etre une liste JSON.")
    items = [CorpusItem(**item) for item in payload]
    if len({item.id for item in items}) != len(items):
        raise ValueError("Les identifiants du corpus TTS doivent etre uniques.")
    if any(not item.id or not item.category or not item.text for item in items):
        raise ValueError("Chaque entree du corpus doit etre complete.")
    return items


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_text(reference).split()
    actual = normalize_text(hypothesis).split()
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def current_commit(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if configured := os.getenv("GIT_COMMIT"):
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Commit inconnu hors depot Git : utilisez --commit ou GIT_COMMIT."
        ) from exc
    return result.stdout.strip()


def worker_vram_mib(client: httpx.Client) -> float | None:
    """Lit la VRAM dans le processus qui possède réellement le modèle Qwen."""
    response = client.get("/health")
    response.raise_for_status()
    value = response.json().get("tts", {}).get("vram_allocated_mib")
    return float(value) if isinstance(value, (int, float)) else None


def unload_worker(socket_path: Path) -> None:
    if not socket_path.exists():
        return
    from transcription_server.tts.client import UnixTtsClient

    client = UnixTtsClient(socket_path, load_timeout_s=60, generation_timeout_s=60)
    asyncio.run(client.unload(reason="benchmark-cold-start"))


def generate(
    client: httpx.Client,
    item: CorpusItem,
    model: str,
    voice: str,
    instructions: str | None,
) -> tuple[bytes, float]:
    payload: dict[str, Any] = {
        "model": model,
        "voice": voice,
        "input": item.text,
        "language": "fr",
        "response_format": "wav",
    }
    if instructions:
        payload["instructions"] = instructions
    started = time.perf_counter()
    response = client.post("/v1/audio/speech", json=payload)
    response.raise_for_status()
    return response.content, time.perf_counter() - started


def transcribe(client: httpx.Client, path: Path) -> str:
    with path.open("rb") as audio:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": (path.name, audio, "audio/wav")},
            data={"model": "parakeet", "language": "fr"},
        )
    response.raise_for_status()
    return str(response.json()["text"])


def _record(
    *, item: CorpusItem, source: str, model: str, commit: str,
    parameters: dict[str, Any], path: Path, transcript: str,
    cold_latency_s: float | None, warm_latency_s: float | None,
    vram_used_mib: float | None,
) -> dict[str, Any]:
    duration_s = wav_duration(path)
    latency = warm_latency_s if warm_latency_s is not None else cold_latency_s
    return {
        "id": item.id,
        "category": item.category,
        "source": source,
        "model": model,
        "commit": commit,
        "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
        "cold_latency_s": cold_latency_s,
        "warm_latency_s": warm_latency_s,
        "duration_s": duration_s,
        "rtf": (latency / duration_s) if latency is not None and duration_s else None,
        "vram_used_mib": vram_used_mib,
        "reference": item.text,
        "transcript": transcript,
        "wer": word_error_rate(item.text, transcript),
        "audio_path": str(path),
    }


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    corpus = load_corpus(args.corpus)
    if args.limit is not None:
        corpus = corpus[: args.limit]
    output_audio = args.output_dir / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)
    commit = current_commit(args.commit)
    rows: list[dict[str, Any]] = []
    timeout = httpx.Timeout(args.timeout_s)
    with httpx.Client(base_url=args.base_url, timeout=timeout) as client:
        for item in corpus:
            unload_worker(args.worker_socket)
            _, cold_latency = generate(
                client, item, args.model, args.voice, args.instructions
            )
            audio, warm_latency = generate(
                client, item, args.model, args.voice, args.instructions
            )
            vram_used_mib = worker_vram_mib(client)
            path = output_audio / f"qwen-{item.id}.wav"
            path.write_bytes(audio)
            transcript = transcribe(client, path)
            rows.append(
                _record(
                    item=item,
                    source="qwen",
                    model=args.model,
                    commit=commit,
                    parameters={
                        "voice": args.voice,
                        "instructions": args.instructions,
                        "language": "fr",
                        "format": "wav",
                    },
                    path=path,
                    transcript=transcript,
                    cold_latency_s=cold_latency,
                    warm_latency_s=warm_latency,
                    vram_used_mib=vram_used_mib,
                )
            )

            if args.voxmind_dir:
                candidates = [
                    candidate
                    for extension in ("wav", "mp3", "flac", "ogg")
                    if (candidate := args.voxmind_dir / f"{item.id}.{extension}").exists()
                ]
                if candidates:
                    voxmind_path = candidates[0]
                    rows.append(
                        _record(
                            item=item,
                            source="voxmind",
                            model="voxmind-export",
                            commit=commit,
                            parameters={"input": "pre-generated export"},
                            path=voxmind_path,
                            transcript=transcribe(client, voxmind_path),
                            cold_latency_s=None,
                            warm_latency_s=None,
                            vram_used_mib=None,
                        )
                    )
    return rows


def write_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not rows:
        return
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("tests/fixtures/tts_corpus_fr.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--voxmind-dir", type=Path)
    parser.add_argument("--worker-socket", type=Path, default=Path("/run/qwen-tts/worker.sock"))
    parser.add_argument("--model", default="tts-1-hd")
    parser.add_argument("--voice", default="Ryan")
    parser.add_argument("--instructions")
    parser.add_argument("--timeout-s", type=float, default=1_200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_benchmark(args)
    write_results(rows, args.output_dir)
    print(f"{len(rows)} mesure(s) ecrite(s) dans {args.output_dir}")


if __name__ == "__main__":
    main()
