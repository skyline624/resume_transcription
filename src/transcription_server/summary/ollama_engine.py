"""Adaptateur Ollama.

Ollama tourne hors du conteneur, sur la machine hote. Ce module ne fait donc
aucune inference lui-meme : il poste une requete HTTP et attend. C'est aussi
pourquoi il n'a besoin ni de torch ni de VRAM.
"""

import json
import logging
import urllib.error
import urllib.request

from transcription_server.summary.engine import SummaryUnavailableError

logger = logging.getLogger(__name__)


class OllamaSummaryEngine:
    """Redige via l'API de generation d'Ollama."""

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self._model

    def summarize(self, prompt: str) -> str:
        charge = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                # Ollama conserve sinon les poids en VRAM plusieurs minutes.
                # Le serveur partage un seul GPU avec ASR, diarization et TTS :
                # le redacteur doit donc liberer sa place des la reponse rendue.
                "keep_alive": 0,
                # La redaction d'un compte-rendu doit etre reproductible : une
                # temperature elevee produirait un texte different a chaque
                # appel sur la meme reunion, ce qui rend toute comparaison
                # impossible et invite le modele a broder.
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")

        requete = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=charge,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(requete, timeout=self._timeout_s) as reponse:
                corps = json.load(reponse)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise SummaryUnavailableError(
                f"Ollama a refusé la demande (HTTP {exc.code}) : {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SummaryUnavailableError(
                f"Ollama est injoignable à {self._base_url} : {exc}. "
                "Vérifiez qu'il tourne et que OLLAMA_BASE_URL est correct."
            ) from exc

        texte = (corps.get("response") or "").strip()
        if not texte:
            raise SummaryUnavailableError(
                f"Ollama a répondu sans contenu. Le modèle {self._model} "
                "est-il bien installé (ollama list) ?"
            )
        return texte


def load_ollama_engine(
    base_url: str, model: str, timeout_s: float
) -> OllamaSummaryEngine:
    """Construit le moteur sans contacter Ollama.

    La disponibilite est verifiee a la premiere demande, pas au demarrage :
    Ollama est un service externe qui peut redemarrer independamment, et le
    serveur doit continuer a transcrire meme quand la redaction est en panne.
    """
    logger.info("Rédaction de compte-rendu via Ollama : %s (%s)", model, base_url)
    return OllamaSummaryEngine(base_url=base_url, model=model, timeout_s=timeout_s)
