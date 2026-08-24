import shutil
import struct
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transcription_server import audio
from transcription_server.audio import (
    SAMPLE_RATE,
    AudioDecodeError,
    decode_to_pcm,
    duration_seconds,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH"
)


def _write_sine_wav(path: Path, seconds: float = 1.0, rate: int = 44100) -> None:
    """Ecrit un wav mono 16 bits contenant un sinus a 440 Hz."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        data = b"".join(
            struct.pack("<h", int(20000 * np.sin(2 * np.pi * 440 * i / rate)))
            for i in range(frames)
        )
        f.writeframes(data)


def test_decode_rend_du_float32_mono(tmp_path):
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=1.0)
    pcm = decode_to_pcm(src)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1


def test_decode_reechantillonne_a_16k(tmp_path):
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=2.0, rate=44100)
    pcm = decode_to_pcm(src)
    assert len(pcm) == pytest.approx(2 * SAMPLE_RATE, rel=0.02)


def test_decode_normalise_entre_moins_un_et_un(tmp_path):
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=0.5)
    pcm = decode_to_pcm(src)
    assert np.abs(pcm).max() <= 1.0
    assert np.abs(pcm).max() > 0.1  # le signal n'est pas nul


def test_duration_seconds():
    pcm = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    assert duration_seconds(pcm) == pytest.approx(3.0)


def test_fichier_inexistant_leve_audio_decode_error(tmp_path):
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(tmp_path / "absent.wav")


def test_fichier_non_audio_leve_audio_decode_error(tmp_path):
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"ceci n'est pas de l'audio")
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(junk)


def test_fichier_audio_vide_leve_audio_decode_error(tmp_path):
    src = tmp_path / "vide.wav"
    _write_sine_wav(src, seconds=0.0)
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(src)


def _write_float_wav(path: Path, samples, rate: int = SAMPLE_RATE) -> None:
    """Ecrit un wav IEEE float32 (wFormatTag = 3).

    Seul moyen simple de fabriquer une source dont les echantillons depassent
    1.0 : le PCM entier est borne par construction, et ffmpeg ne rabote pas le
    flottant. C'est donc la seule facon d'atteindre la branche de
    normalisation avec un vrai fichier.
    """
    data = np.asarray(samples, dtype="<f4").tobytes()
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
        + b"data"
        + struct.pack("<I", len(data))
    )
    path.write_bytes(header + data)


def _write_stereo_sine_wav(
    path: Path, seconds: float = 1.0, rate: int = SAMPLE_RATE
) -> None:
    """Ecrit un wav stereo 16 bits dont les deux voies portent le meme sinus."""
    frames = int(seconds * rate)
    voie = [int(20000 * np.sin(2 * np.pi * 440 * i / rate)) for i in range(frames)]
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"".join(struct.pack("<hh", v, v) for v in voie))


def test_decode_ramene_un_pic_superieur_a_un_dans_les_bornes(tmp_path):
    """Sans la division par le pic, ce fichier ressortirait a 4.0."""
    src = tmp_path / "sature.wav"
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    _write_float_wav(src, 4.0 * np.sin(2 * np.pi * 440 * t))
    pcm = decode_to_pcm(src)
    assert np.abs(pcm).max() == pytest.approx(1.0, rel=1e-3)
    assert pcm.dtype == np.float32


def test_decode_n_amplifie_pas_un_signal_faible(tmp_path):
    """Une normalisation inconditionnelle porterait ce sinus a 1.0.

    La source vaut 20000 sur les 32768 du 16 bits, soit un pic attendu de
    0.61 : le silence relatif d'un enregistrement doit ressortir intact.
    """
    src = tmp_path / "faible.wav"
    _write_sine_wav(src, seconds=0.5)
    pcm = decode_to_pcm(src)
    assert np.abs(pcm).max() == pytest.approx(20000 / 32768, rel=0.02)


def test_decode_du_silence_ne_produit_pas_de_nan(tmp_path):
    """Diviser par un pic nul rendrait un tableau entierement NaN."""
    src = tmp_path / "silence.wav"
    _write_float_wav(src, np.zeros(SAMPLE_RATE, dtype=np.float32))
    pcm = decode_to_pcm(src)
    assert pcm.size == SAMPLE_RATE
    assert not np.isnan(pcm).any()
    assert np.abs(pcm).max() == 0.0


def test_decode_reduit_le_stereo_en_mono(tmp_path):
    """Sans -ac 1, ffmpeg rendrait 2 x 16000 echantillons entrelaces."""
    src = tmp_path / "stereo.wav"
    _write_stereo_sine_wav(src, seconds=1.0)
    pcm = decode_to_pcm(src)
    assert pcm.ndim == 1
    assert len(pcm) == pytest.approx(SAMPLE_RATE, rel=0.02)
    assert np.abs(pcm).max() > 0.1


def test_decode_honore_le_parametre_sample_rate(tmp_path):
    """Une frequence codee en dur rendrait 16000 echantillons."""
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=1.0, rate=44100)
    pcm = decode_to_pcm(src, sample_rate=8000)
    assert len(pcm) == pytest.approx(8000, rel=0.02)


def test_decode_accepte_un_format_compresse(tmp_path):
    """La raison d'etre du module : ne pas se limiter au wav."""
    wav = tmp_path / "sine.wav"
    _write_sine_wav(wav, seconds=1.0, rate=44100)
    flac = tmp_path / "sine.flac"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(wav), str(flac)],
        check=True,
        capture_output=True,
    )
    pcm = decode_to_pcm(flac)
    assert pcm.dtype == np.float32
    assert len(pcm) == pytest.approx(SAMPLE_RATE, rel=0.02)


def test_pcm_rendu_est_inscriptible(tmp_path):
    """np.frombuffer rend une vue en lecture seule sur les octets de ffmpeg,
    et cette vue retient tout le buffer de sortie tant qu'une tranche survit.
    Les deux branches doivent rendre un tableau possede et inscriptible."""
    faible = tmp_path / "faible.wav"
    _write_sine_wav(faible, seconds=0.2)
    assert decode_to_pcm(faible).flags.writeable

    sature = tmp_path / "sature.wav"
    t = np.arange(SAMPLE_RATE // 5) / SAMPLE_RATE
    _write_float_wav(sature, 4.0 * np.sin(2 * np.pi * 440 * t))
    assert decode_to_pcm(sature).flags.writeable


def test_fichier_non_audio_rapporte_l_erreur_ffmpeg(tmp_path):
    """Sans la garde sur le code de retour, ffmpeg echoue avec une sortie
    vide et c'est le message « aucun echantillon » qui remonterait."""
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"ceci n'est pas de l'audio")
    with pytest.raises(AudioDecodeError) as exc:
        decode_to_pcm(junk)
    assert "ffmpeg n'a pas pu décoder" in str(exc.value)


def test_fichier_audio_vide_rapporte_l_absence_d_echantillon(tmp_path):
    """Un wav sans trame fait sortir ffmpeg avec le code 0 : c'est bien la
    garde sur la taille, et non celle sur le code de retour, qui doit lever."""
    src = tmp_path / "vide.wav"
    _write_sine_wav(src, seconds=0.0)
    with pytest.raises(AudioDecodeError) as exc:
        decode_to_pcm(src)
    assert "aucun échantillon" in str(exc.value)


def test_fichier_inexistant_rapporte_le_chemin(tmp_path):
    absent = tmp_path / "absent.wav"
    with pytest.raises(AudioDecodeError) as exc:
        decode_to_pcm(absent)
    assert "Fichier introuvable" in str(exc.value)
    assert str(absent) in str(exc.value)


def test_duration_seconds_honore_le_sample_rate():
    """Un SAMPLE_RATE code en dur rendrait 0.5 au lieu de 1.0."""
    pcm = np.zeros(8000, dtype=np.float32)
    assert duration_seconds(pcm, sample_rate=8000) == pytest.approx(1.0)


def test_duration_seconds_rend_une_duree_fractionnaire():
    """Une division entiere rendrait 1.0."""
    pcm = np.zeros(24000, dtype=np.float32)
    assert duration_seconds(pcm) == pytest.approx(1.5)


def test_duration_seconds_sur_un_tableau_vide():
    assert duration_seconds(np.zeros(0, dtype=np.float32)) == 0.0


def test_decode_accepte_un_chemin_texte(tmp_path):
    """La signature annonce str | Path : les deux doivent marcher."""
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=0.5, rate=44100)
    pcm = decode_to_pcm(str(src))
    assert len(pcm) == pytest.approx(0.5 * SAMPLE_RATE, rel=0.02)


def test_duration_seconds_correspond_a_la_source(tmp_path):
    """Les deux fonctions sont toujours utilisees ensemble : le PCM decode
    d'un fichier de 2 s doit se redire 2 s."""
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=2.0, rate=44100)
    assert duration_seconds(decode_to_pcm(src)) == pytest.approx(2.0, rel=0.02)


def test_ffmpeg_absent_leve_audio_decode_error(tmp_path, monkeypatch):
    """Cas d'une image mal construite : le diagnostic doit nommer ffmpeg
    plutot que de laisser remonter un FileNotFoundError du systeme."""
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=0.2)
    monkeypatch.setattr(audio, "shutil", SimpleNamespace(which=lambda _: None))
    with pytest.raises(AudioDecodeError) as exc:
        decode_to_pcm(src)
    assert "ffmpeg est introuvable" in str(exc.value)


def test_ffmpeg_est_appele_avec_nostdin(tmp_path, monkeypatch):
    """-nostdin empeche ffmpeg de consommer le stdin du processus parent.

    Aucun test de comportement ne peut l'observer -- sous pytest le stdin est
    deja neutralise -- alors que son retrait exposerait le serveur, ou ffmpeg
    heriterait du descripteur du parent. On epingle donc l'argument lui-meme.
    """
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=0.2)
    appels = []

    def faux_run(command, **kwargs):
        appels.append(command)
        return subprocess.CompletedProcess(
            command, 0, np.zeros(8, dtype=np.float32).tobytes(), b""
        )

    monkeypatch.setattr(audio.subprocess, "run", faux_run)
    decode_to_pcm(src)
    assert "-nostdin" in appels[0]
