"""Workshop key release for mograder encrypted WASM notebooks.

Reuses mograder's built-in Starlette dashboard (toggle checkboxes,
release-all, auto-refresh) rather than maintaining a separate UI.

Public endpoint:
    GET /workshops/{name}/keys.json  — released keys (CORS, no auth)

SSO-protected dashboard (mounted under /live/):
    GET /live/workshops/{name}/dashboard.html  — instructor dashboard
    + JSON API endpoints for release/revoke

Setup:
    1. Place keys_all.json + dashboard.html in server/workshops/{name}/
    2. deploy-warwick.sh patches dashboard.html to use SSO token
    3. Starlette apps are auto-discovered and mounted
"""

import json
from pathlib import Path

from fastapi import APIRouter
from starlette.responses import Response

WORKSHOPS_DIR = Path(__file__).parent / "workshops"
SSO_TOKEN = "sso"  # dummy token; real auth is via Apache SSO on /live/

router = APIRouter()


# --- Public keys endpoint (CORS for GitHub Pages WASM notebooks) ---


@router.get("/workshops/{workshop}/keys.json")
async def workshop_keys(workshop: str):
    """Serve released keys. Called by WASM notebooks on GitHub Pages."""
    keys_file = WORKSHOPS_DIR / workshop / "keys.json"
    data = keys_file.read_text() if keys_file.exists() else "{}"
    return Response(
        content=data,
        media_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# --- Starlette app factory for mograder dashboards ---


def create_workshop_mounts():
    """Create mograder Starlette sub-apps for each workshop directory.

    Returns dict of {name: starlette_app} to be mounted under
    /live/workshops/{name}/ in the main FastAPI app.
    """
    from mograder.transport.workshop_server import create_workshop_starlette_routes

    apps = {}
    for ws_dir in sorted(WORKSHOPS_DIR.iterdir()):
        if not ws_dir.is_dir():
            continue
        keys_all_file = ws_dir / "keys_all.json"
        dashboard_file = ws_dir / "dashboard.html"
        if not keys_all_file.exists():
            continue
        if not dashboard_file.exists():
            continue

        keys_all = json.loads(keys_all_file.read_text())
        keys_path = ws_dir / "keys.json"

        ws_app = create_workshop_starlette_routes(
            export_dir=ws_dir,
            keys_path=keys_path,
            keys_all=keys_all,
            secret=SSO_TOKEN,
        )
        apps[ws_dir.name] = ws_app
    return apps
