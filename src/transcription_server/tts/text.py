"""Normalisation minimale et segmentation linguistique des textes TTS."""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    text: str
    pause_after_ms: int


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def segment_text(text: str, max_chars: int) -> list[TextSegment]:
    if max_chars <= 0:
        raise ValueError("max_chars doit être strictement positive.")
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Le texte à synthétiser est vide.")
    sentences = re.split(r"(?<=[.!?…])\s+", normalized)
    chunks: list[str] = []
    for sentence in sentences:
        chunks.extend(_split_long_segment(sentence, max_chars))
    return [
        TextSegment(text=chunk, pause_after_ms=_pause_after(chunk))
        for chunk in chunks
    ]


def _split_long_segment(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = _last_boundary(remaining, max_chars, ";:")
        if cut is None:
            cut = _last_boundary(remaining, max_chars, ",")
        if cut is None:
            space = remaining.rfind(" ", 0, max_chars + 1)
            cut = space if space > 0 else max_chars
            chunk = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
        else:
            chunk = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
        if chunk:
            chunks.append(chunk)
    if remaining:
        chunks.append(remaining)
    return chunks


def _last_boundary(text: str, max_chars: int, punctuation: str) -> int | None:
    for index in range(min(max_chars, len(text)) - 1, -1, -1):
        if text[index] in punctuation and (
            index + 1 == len(text) or text[index + 1].isspace()
        ):
            return index + 1
    return None


def _pause_after(text: str) -> int:
    if text.endswith(("!", "?", "…")):
        return 450
    if text.endswith("."):
        return 350
    if text.endswith((";", ":")):
        return 250
    if text.endswith(","):
        return 150
    return 100
