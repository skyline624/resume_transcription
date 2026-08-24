import pytest

from transcription_server.domain import SpeakerSegment, Turn, Word


def test_word_duration():
    w = Word(text="bonjour", start=1.0, end=1.75)
    assert w.duration == pytest.approx(0.75)


def test_word_is_immutable():
    w = Word(text="bonjour", start=1.0, end=1.75)
    with pytest.raises(Exception):
        w.text = "autre"


def test_speaker_segment_duration():
    s = SpeakerSegment(speaker="SPEAKER_00", start=2.0, end=5.5)
    assert s.duration == pytest.approx(3.5)


def test_turn_holds_words_as_tuple():
    words = (Word(text="a", start=0.0, end=0.5), Word(text="b", start=0.5, end=1.0))
    t = Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="a b", words=words)
    assert t.words == words
    assert t.speaker == "SPEAKER_00"


def test_turn_accepts_none_speaker():
    t = Turn(speaker=None, start=0.0, end=1.0, text="a", words=())
    assert t.speaker is None
