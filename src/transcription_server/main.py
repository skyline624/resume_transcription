"""Point d'entree du serveur.

Lance par le CMD du conteneur : `python -m transcription_server.main`.
"""

import logging

import uvicorn

from transcription_server.app import build_app
from transcription_server.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    reglages = get_settings()
    # L'application est construite avant `uvicorn.run` : un defaut de
    # configuration ou un GPU absent doit faire echouer le demarrage tout de
    # suite, pas apres l'ouverture du port.
    application = build_app(reglages)
    uvicorn.run(
        application,
        host=reglages.host,
        port=reglages.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
