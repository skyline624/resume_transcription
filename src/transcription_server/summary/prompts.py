"""Gabarits de redaction du compte-rendu.

Ces textes sont la moitie du produit : c'est eux, bien plus que le choix du
modele, qui decident si le resultat est exploitable ou si c'est de la
paraphrase. Ils sont donc ici, versionnes et testables, plutot que noyes dans
le code d'appel.
"""

from transcription_server.domain import Turn

FORMATS = ("structure", "narratif")

_CONSIGNES_COMMUNES = """Tu rédiges à partir de la transcription automatique d'une conversation.

Contraintes que tu dois respecter :
- N'invente rien. Si une information est absente ou inaudible, ne la déduis pas.
- La transcription comporte des erreurs de reconnaissance : un mot qui n'a pas
  de sens dans le contexte est probablement mal transcrit. Ne le cite pas
  comme s'il était sûr.
- Les locuteurs sont anonymes (SPEAKER_00, SPEAKER_01…). N'invente pas de noms.
  Si quelqu'un se nomme dans la conversation, tu peux l'indiquer entre
  parenthèses la première fois.
- Écris en français, sans anglicismes inutiles.
- Ne commente ni ta démarche ni la qualité de la transcription."""

_GABARIT_STRUCTURE = """{communes}

Produis un compte-rendu structuré, avec exactement ces sections, dans cet ordre.
Omets une section entière si la conversation n'en dit rien — ne la remplis pas
pour la forme.

## Objet
Une phrase : de quoi cette conversation traite.

## Participants
Un par ligne : l'étiquette du locuteur, son rôle s'il ressort clairement, et
ce qu'il a principalement apporté.

## Sujets abordés
Les thèmes traités, dans l'ordre. Pour chacun, ce qui a été dit de substantiel.

## Décisions
Ce qui a été tranché. Rien d'autre. Si aucune décision n'a été prise, écris
« Aucune décision explicite ».

## Actions à mener
Une par ligne, sous la forme : qui fait quoi, et pour quand si c'est dit.

## Points en suspens
Questions soulevées et laissées sans réponse, désaccords non résolus."""

_GABARIT_NARRATIF = """{communes}

Rédige un résumé en prose continue, de trois à six paragraphes selon la
richesse de la conversation.

Restitue le déroulé : de quoi on est parti, comment la discussion a évolué, où
elle a abouti. Fais ressortir les points saillants et les éventuels
désaccords. N'emploie ni titres, ni listes à puces."""


def construire_prompt(dialogue: str, format_: str) -> str:
    """Assemble la consigne et la transcription.

    Prend le dialogue deja rendu en texte, et non des `Turn` : l'appelant peut
    ainsi fournir une transcription produite ailleurs — reprise d'un fichier,
    corrigee a la main — sans avoir a la reconstruire en objets.
    """
    if format_ not in FORMATS:
        raise ValueError(
            f"Format de compte-rendu inconnu : {format_!r}. "
            f"Attendu l'un de {', '.join(FORMATS)}."
        )
    if not dialogue.strip():
        raise ValueError(
            "La transcription fournie est vide : il n'y a rien à résumer."
        )
    gabarit = _GABARIT_STRUCTURE if format_ == "structure" else _GABARIT_NARRATIF
    consignes = gabarit.format(communes=_CONSIGNES_COMMUNES)
    return f"{consignes}\n\n---\n\nTRANSCRIPTION :\n\n{dialogue}\n"


def rendre_dialogue(turns: list[Turn]) -> str:
    """Rend la transcription sous la forme la plus lisible pour le modele.

    L'horodatage est conserve : il permet au modele de situer les propos les
    uns par rapport aux autres, et a un lecteur de retrouver le passage dans
    l'enregistrement.
    """
    lignes = []
    for tour in turns:
        if not tour.text:
            continue
        minutes, secondes = divmod(int(tour.start), 60)
        heures, minutes = divmod(minutes, 60)
        locuteur = tour.speaker or "INCONNU"
        lignes.append(f"[{heures:02d}:{minutes:02d}:{secondes:02d}] {locuteur} : {tour.text}")
    return "\n".join(lignes)
