"""Rendu des tours de parole dans les formats de sortie supportes."""

from transcription_server.domain import Turn

UNKNOWN_SPEAKER = "INCONNU"


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """Formate des secondes en HH:MM:SS<sep>mmm.

    SRT attend une virgule avant les millisecondes, WebVTT un point.
    """
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _short_timestamp(seconds: float) -> str:
    """HH:MM:SS.cc, avec deux decimales, pour le format dialogue."""
    total_cs = max(0, round(seconds * 100))
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cents = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{cents:02d}"


def _labelled(turn: Turn) -> str:
    if turn.speaker is None:
        return turn.text
    return f"{turn.speaker}: {turn.text}"


def to_plain_text(turns: list[Turn]) -> str:
    return " ".join(t.text for t in turns if t.text)


def to_srt(turns: list[Turn]) -> str:
    blocks: list[str] = []
    for i, turn in enumerate(turns, start=1):
        start = format_timestamp(turn.start, separator=",")
        end = format_timestamp(turn.end, separator=",")
        blocks.append(f"{i}\n{start} --> {end}\n{_labelled(turn)}\n")
    return "\n".join(blocks)


def to_vtt(turns: list[Turn]) -> str:
    blocks: list[str] = []
    for turn in turns:
        start = format_timestamp(turn.start, separator=".")
        end = format_timestamp(turn.end, separator=".")
        blocks.append(f"{start} --> {end}\n{_labelled(turn)}\n")
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_dialogue(turns: list[Turn]) -> str:
    lines: list[str] = []
    for turn in turns:
        speaker = turn.speaker or UNKNOWN_SPEAKER
        lines.append(f"[{_short_timestamp(turn.start)}] {speaker}: {turn.text}")
    return "\n".join(lines)
