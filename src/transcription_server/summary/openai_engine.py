"""Adaptateur OpenAI : tout serveur qui expose /v1/chat/completions (vLLM, llama.cpp, LM Studio...).

Comme pour Ollama, ce module ne fait aucune inference lui-meme : il poste une
requete HTTP et attend. A la difference d'Ollama, le modele tourne sur une
VRAM que ce conteneur ne gere pas (vLLM garde la sienne) : inutile de
decharger la transcription et la synthese avant l'appel, d'ou
`partage_le_gpu = False`.
"""

import json
import logging
import urllib.error
import urllib.request

from transcription_server.summary.engine import SummaryUnavailableError

logger = logging.getLogger(__name__)


class OpenAISummaryEngine:
    """Redige via l'API de chat compatible OpenAI."""

    # Le serveur de chat a sa propre VRAM : la route n'a ni verrou GPU a
    # prendre ni modele a decharger.
    partage_le_gpu = False

    def __init__(
        self, base_url: str, model: str, timeout_s: float, api_key: str | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._api_key = api_key

    @property
    def name(self) -> str:
        return self._model

    def summarize(self, prompt: str) -> str:
        charge = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Meme raison que pour Ollama : un compte-rendu reproductible,
                # sans broderie.
                "temperature": 0.2,
            }
        ).encode("utf-8")
        entetes = {"Content-Type": "application/json"}
        if self._api_key:
            entetes["Authorization"] = f"Bearer {self._api_key}"

        requete = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=charge, headers=entetes
        )
        try:
            with urllib.request.urlopen(requete, timeout=self._timeout_s) as reponse:
                corps = json.load(reponse)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise SummaryUnavailableError(
                f"Le serveur de chat a refusé la demande (HTTP {exc.code}) : {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SummaryUnavailableError(
                f"Le serveur de chat est injoignable à {self._base_url} : {exc}. "
                "Vérifiez qu'il tourne et que SUMMARY_BASE_URL est correct."
            ) from exc

        try:
            texte = (corps["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            texte = ""
        if not texte:
            raise SummaryUnavailableError(
                f"Le serveur de chat a répondu sans contenu. Le modèle {self._model} "
                "est-il bien servi (GET /v1/models) ?"
            )
        return texte


def load_openai_engine(
    base_url: str, model: str, timeout_s: float, api_key: str | None = None
) -> OpenAISummaryEngine:
    """Construit le moteur sans contacter le serveur, pour la meme raison que
    load_ollama_engine : la transcription ne doit pas dependre du redacteur."""
    logger.info("Rédaction de compte-rendu via l'API OpenAI : %s (%s)", model, base_url)
    return OpenAISummaryEngine(
        base_url=base_url, model=model, timeout_s=timeout_s, api_key=api_key
    )
