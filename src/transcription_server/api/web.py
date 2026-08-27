"""Distribution de l'interface web compilée, sans fallback sur les routes API."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_CSP = (
    "default-src 'self'; base-uri 'self'; form-action 'self'; "
    "connect-src 'self'; img-src 'self' data: blob:; "
    "media-src 'self' blob:; script-src 'self'; style-src 'self'; "
    "worker-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
)


def mount_web_ui(app: FastAPI, dist_path: Path) -> None:
    """Monte uniquement la racine et les assets Vite quand ils existent."""

    index = dist_path / "index.html"
    assets = dist_path / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="web-assets")

    @app.get("/", include_in_schema=False)
    async def web_index():
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Interface web non construite.")
        return FileResponse(index, headers={"Content-Security-Policy": _CSP})
