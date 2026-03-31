"""Workshop key release endpoints for mograder encrypted notebooks.

Students fetch keys.json from GitHub Pages WASM notebooks (CORS required).
Instructor releases keys via SSO-protected dashboard at /live/workshops/{name}/dashboard.

Setup:
  1. Run `mograder workshop export` in CI → produces keys_all.json
  2. Copy keys_all.json to server/workshops/{name}/keys_all.json
  3. keys.json is created automatically (starts empty, grows as you release)

Usage:
  from workshops import router as workshops_router
  app.include_router(workshops_router)
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

WORKSHOPS_DIR = Path(__file__).parent / "workshops"

router = APIRouter()


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


# --- Public endpoint (CORS for GitHub Pages) ---


@router.get("/workshops/{workshop}/keys.json")
async def workshop_keys(workshop: str):
    """Serve released keys. Called by WASM notebooks on GitHub Pages."""
    keys = _load_json(WORKSHOPS_DIR / workshop / "keys.json")
    return Response(
        content=json.dumps(keys),
        media_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# --- Instructor dashboard (SSO via /live mount) ---


@router.get("/live/workshops/{workshop}/dashboard", response_class=HTMLResponse)
async def workshop_dashboard(request: Request, workshop: str):
    """Dashboard to release solutions one at a time."""
    user = request.headers.get("x-remote-user", "")
    if not user:
        raise HTTPException(status_code=403, detail="SSO login required")

    keys_all_file = WORKSHOPS_DIR / workshop / "keys_all.json"
    keys_file = WORKSHOPS_DIR / workshop / "keys.json"

    if not keys_all_file.exists():
        return HTMLResponse(
            f"<h1>Workshop '{workshop}' not configured</h1>"
            f"<p>Place keys_all.json in workshops/{workshop}/</p>",
            status_code=404,
        )

    all_keys = _load_json(keys_all_file)
    released = _load_json(keys_file)

    rows = ""
    for exercise_id in all_keys:
        is_released = exercise_id in released
        status = "✅ Released" if is_released else "🔒 Locked"
        button = "" if is_released else (
            f'<form method="POST" style="display:inline">'
            f'<input type="hidden" name="exercise" value="{exercise_id}">'
            f'<button type="submit">Release</button></form>'
        )
        rows += f"<tr><td>{exercise_id}</td><td>{status}</td><td>{button}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html><head><title>Workshop: {workshop}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }}
  h1 {{ color: #5f259f; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  button {{ background: #5f259f; color: white; border: none; padding: 6px 16px;
           border-radius: 4px; cursor: pointer; }}
  button:hover {{ background: #4a1d7a; }}
  .info {{ background: #f0f0f0; border-radius: 6px; padding: 12px; margin: 16px 0; font-size: 0.9em; }}
</style></head>
<body>
  <h1>Workshop: {workshop}</h1>
  <p>Logged in as: <strong>{user}</strong></p>
  <div class="info">Release solutions one at a time during the support session.
  Students' WASM notebooks fetch updated keys on next page load.</div>
  <table>
    <tr><th>Exercise</th><th>Status</th><th>Action</th></tr>
    {rows}
  </table>
</body></html>"""


@router.post("/live/workshops/{workshop}/dashboard")
async def workshop_release(request: Request, workshop: str):
    """Release a single exercise solution."""
    user = request.headers.get("x-remote-user", "")
    if not user:
        raise HTTPException(status_code=403, detail="SSO login required")

    form = await request.form()
    exercise_id = form.get("exercise", "")

    keys_all_file = WORKSHOPS_DIR / workshop / "keys_all.json"
    keys_file = WORKSHOPS_DIR / workshop / "keys.json"

    if not keys_all_file.exists():
        raise HTTPException(status_code=404, detail="Workshop not configured")

    all_keys = _load_json(keys_all_file)
    if exercise_id not in all_keys:
        raise HTTPException(status_code=400, detail=f"Unknown exercise: {exercise_id}")

    released = _load_json(keys_file)
    released[exercise_id] = all_keys[exercise_id]
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    keys_file.write_text(json.dumps(released, indent=2))

    return RedirectResponse(f"/live/workshops/{workshop}/dashboard", status_code=303)
