"""Contrat des moteurs de redaction.

Meme motif que pour l'ASR et la diarization : la route ne connait que ce
Protocol, jamais Ollama. C'est ce qui permet de tester toute la surface HTTP
sans qu'aucun modele de langue ne tourne.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SummaryEngine(Protocol):
    """Redige un compte-rendu a partir d'une consigne complete."""

    @property
    def name(self) -> str:
        """Identifiant du modele, expose par /health."""
        ...

    def summarize(self, prompt: str) -> str:
        """Rend le texte redige. Leve SummaryUnavailableError si le service
        de redaction est injoignable ou refuse la demande."""
        ...


class SummaryUnavailableError(Exception):
    """Le service de redaction est indisponible. Correspond a un HTTP 503.

    Distinct d'une erreur de transcription : le serveur peut parfaitement
    transcrire sans savoir rediger, et l'appelant doit pouvoir faire la
    difference entre « ton fichier est mauvais » et « le modele de langue
    n'est pas la ».
    """


class StubSummaryEngine:
    """Moteur a sortie fixe, pour tester les routes sans modele de langue."""

    def __init__(self, texte: str = "Compte-rendu simulé.", name: str = "stub-summary"):
        self._texte = texte
        self._name = name
        self.prompts_recus: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def summarize(self, prompt: str) -> str:
        self.prompts_recus.append(prompt)
        return self._texte


class UnavailableSummaryEngine:
    """Moteur qui refuse toute demande.

    Utilise quand ENABLE_SUMMARY=false : la route repond alors un 503
    explicite au lieu de laisser croire que la fonction existe.
    """

    @property
    def name(self) -> str:
        return "none"

    def summarize(self, prompt: str) -> str:
        raise SummaryUnavailableError(
            "La rédaction de compte-rendu est désactivée sur ce serveur "
            "(ENABLE_SUMMARY=false)."
        )
