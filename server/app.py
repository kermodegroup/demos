import asyncio
import marimo
import tomllib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
import websockets
NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
PRESENTATIONS_DIR = Path(__file__).parent / "presentations"
# demos.toml lives alongside app.py on the server, but one level up in the repo
CONFIG_FILE = Path(__file__).parent / "demos.toml"
if not CONFIG_FILE.exists():
    CONFIG_FILE = Path(__file__).parent.parent / "demos.toml"
GITHUB_PAGES_BASE = "https://kermodegroup.github.io/demos"
MOLAB_BASE = "https://molab.marimo.io/github/kermodegroup/demos/blob/main"
MOLAB_PARAMS = "/wasm?include-code=false"
MORIARTY_FORMGRADER = "http://moriarty.scrtp.warwick.ac.uk:2718"
MORIARTY_HUB = "http://localhost:18080"  # SSH tunnel to sciml.warwick.cloud (RONIN)
FORMGRADER_USERS_FILE = Path(__file__).parent / "formgrader_users.txt"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create marimo server for live notebooks (mounted at /live)
server = marimo.create_asgi_app()
live_notebooks = []
for notebook in sorted(NOTEBOOKS_DIR.glob("*.py")):
    name = notebook.stem
    server = server.with_app(path=f"/{name}", root=str(notebook))
    live_notebooks.append(name)

# Create marimo server for public demos (mounted at /demos)
demo_server = marimo.create_asgi_app()
public_demos = []  # list of (name, relative_path) tuples
if PRESENTATIONS_DIR.exists():
    for notebook in sorted(PRESENTATIONS_DIR.glob("*/*.py")):
        name = notebook.parent.name
        demo_server = demo_server.with_app(path=f"/{name}", root=str(notebook))
        public_demos.append((name, f"presentations/{name}/{notebook.name}"))

# Load demo config
demo_config = []
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)
        demo_config = config.get("demos", [])

# Build config lookup
config_by_name = {d["name"]: d for d in demo_config}
config_order = [d["name"] for d in demo_config]

# Get WASM notebooks from config (those not in live_notebooks and not demos)
wasm_notebooks = [
    d["name"]
    for d in demo_config
    if d["name"] not in live_notebooks
    and d.get("type") != "demo"
    and not d.get("hidden", False)
]


# Build molab URLs for all notebooks via GitHub integration
# See https://docs.marimo.io/guides/molab/#embed-notebooks-from-github
molab_urls: dict[str, str] = {}
for name in live_notebooks:
    molab_urls[name] = f"{MOLAB_BASE}/notebooks/{name}.py"
for name in wasm_notebooks:
    molab_urls[name] = f"{MOLAB_BASE}/apps/{name}.py{MOLAB_PARAMS}"
for name, rel_path in public_demos:
    molab_urls[name] = f"{MOLAB_BASE}/{rel_path}"

# Formgrader reverse proxy access control
grader_enabled = FORMGRADER_USERS_FILE.exists()


def _formgrader_allowed_users() -> set[str]:
    """Read allowed users from file on each call (no restart needed to update)."""
    if not FORMGRADER_USERS_FILE.exists():
        return set()
    return {
        line.strip()
        for line in FORMGRADER_USERS_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _check_formgrader_access(request: Request) -> str:
    """Return username if allowed, raise 403 otherwise."""
    user = request.headers.get("x-remote-user", "")
    if not user or user not in _formgrader_allowed_users():
        raise HTTPException(status_code=403, detail="Forbidden: formgrader access required")
    return user


def get_display_title(name):
    """Get display title from config or auto-generate."""
    if name in config_by_name:
        return config_by_name[name].get(
            "title", name.replace("-", " ").replace("_", " ").title()
        )
    return name.replace("-", " ").replace("_", " ").title()


def get_sort_key(name):
    """Get sort key - config order first, then alphabetical."""
    if name in config_order:
        return (0, config_order.index(name))
    return (1, name)


@app.get("/", response_class=HTMLResponse)
def index():
    all_notebooks = []

    # Add live notebooks (served at /live, SSO protected)
    for name in live_notebooks:
        if name not in config_by_name or not config_by_name[name].get("hidden", False):
            all_notebooks.append((name, f"/live/{name}/", "live"))

    # Add WASM notebooks (redirect via /wasm/ to GitHub Pages)
    for name in wasm_notebooks:
        all_notebooks.append((name, f"/wasm/{name}/", "wasm"))

    # Add public demos (served at /demos)
    for name, _ in public_demos:
        all_notebooks.append((name, f"/demos/{name}/", "demo"))

    # Sort by config order, then alphabetically
    all_notebooks.sort(key=lambda x: get_sort_key(x[0]))

    notebook_links = "".join(
        f'<li><a href="{url}">{get_display_title(name)}</a>'
        f'<span class="badge {badge_type}">{badge_type.upper()}</span>'
        + (f'<a href="/molab/{name}/" class="molab-link" title="Open in molab (no login required)">molab</a>' if name in molab_urls else '')
        + '</li>'
        for name, url, badge_type in all_notebooks
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SciML - University of Warwick</title>
        <style>
            body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #5f259f; }}
            ul {{ list-style: none; padding: 0; }}
            li {{ margin: 10px 0; }}
            a {{ color: #0066cc; text-decoration: none; font-size: 1.1em; }}
            a:hover {{ text-decoration: underline; }}
            .badge {{ font-size: 0.7em; padding: 2px 6px; border-radius: 3px; margin-left: 8px; text-transform: uppercase; }}
            .wasm {{ background: #d4edda; color: #155724; }}
            .live {{ background: #fff3cd; color: #856404; }}
            .demo {{ background: #d1ecf1; color: #0c5460; }}
            .grader {{ background: #f8d7da; color: #721c24; }}
            .molab-link {{ font-size: 0.75em; padding: 2px 6px; border-radius: 3px; margin-left: 6px; background: #e8d5f5; color: #5f259f; text-decoration: none; }}
            .molab-link:hover {{ background: #d4b8eb; text-decoration: none; }}
            .note {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1em; margin: 1.5em 0; }}
        </style>
    </head>
    <body>
        <h1>SciML Notebooks</h1>
        <p>Interactive scientific machine learning demonstrations.
        Developed by <a href="https://warwick.ac.uk/jrkermode">James Kermode</a>
        to support teaching of Scientific Machine Learning (ES98E) and
        Predictive Modelling and Uncertainty Quantification (PX914)
        in the <a href="https://warwick.ac.uk/HetSys">HetSys CDT</a>
        and <a href="https://warwick.ac.uk/pmsc">Predictive Modelling and Scientific Computing MSc</a>.</p>
        <ul>{notebook_links}</ul>
        {"" if not grader_enabled else '<p><a href="/live/grader/">Formgrader</a> <span class="badge grader">STAFF</span></p>'}
        <div class="note">
            <p><strong>WASM</strong> notebooks run in your browser (no login required).
            <strong>LIVE</strong> notebooks require University of Warwick SSO.
            <strong>DEMO</strong> presentations are public (no login required).</p>
            <p><strong style="color: #5f259f;">molab</strong> links open notebooks in
            <a href="https://docs.marimo.io/guides/molab/">marimo's free cloud environment</a> &mdash;
            no login or installation required. Useful for external collaborators
            and students without a Warwick account.</p>
        </div>
    </body>
    </html>
    """


# Redirect /wasm/{name} to GitHub Pages
@app.get("/wasm/{name}/")
@app.get("/wasm/{name}")
def wasm_redirect(name: str):
    """Redirect WASM notebook requests to GitHub Pages."""
    return RedirectResponse(
        url=f"{GITHUB_PAGES_BASE}/{name}.html",
        status_code=302
    )


# Redirect /molab/{name} to molab.marimo.io via GitHub integration
@app.get("/molab/{name}/")
@app.get("/molab/{name}")
def molab_redirect(name: str):
    """Redirect to molab.marimo.io with the notebook loaded from GitHub."""
    if name not in molab_urls:
        raise HTTPException(status_code=404, detail=f"Notebook '{name}' not found")
    return RedirectResponse(url=molab_urls[name], status_code=302)


@app.get("/live/debug-headers")
def debug_headers(request: Request):
    """Temporary: check what headers Apache passes under /live/."""
    return {
        "headers": dict(request.headers),
        "x-remote-user": request.headers.get("x-remote-user", "(not set)"),
    }


PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
_HOP_BY_HOP = {"transfer-encoding", "connection", "keep-alive"}

_proxy_logger = logging.getLogger("uvicorn.error")


# --- Shared reverse proxy helpers ---


async def _proxy_http(
    request: Request,
    upstream_base: str,
    path: str,
    user: str,
    *,
    timeout: float = 30.0,
    service_name: str = "upstream",
    error_html: str | None = None,
) -> Response:
    """Forward an HTTP request to an upstream server and return its response."""
    target_url = f"{upstream_base}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    # Strip proxy headers so upstream sees the connection as coming from localhost
    for h in ("x-forwarded-for", "x-forwarded-host", "x-forwarded-server", "x-real-ip"):
        headers.pop(h, None)
    headers["x-remote-user"] = user

    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
    except httpx.ConnectError:
        if error_html:
            return HTMLResponse(content=error_html, status_code=502)
        raise HTTPException(
            status_code=502,
            detail=f"{service_name} server is not responding",
        )

    response_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    return Response(content=resp.content, status_code=resp.status_code, headers=response_headers)


async def _proxy_ws(
    ws: WebSocket,
    upstream_ws_url: str,
    path: str,
    user: str,
    *,
    service_name: str = "upstream",
) -> None:
    """Bidirectional WebSocket relay to an upstream server."""
    await ws.accept()

    target_url = f"{upstream_ws_url}/{path}"
    if ws.url.query:
        target_url += f"?{ws.url.query}"

    try:
        async with websockets.connect(
            target_url,
            additional_headers={"x-remote-user": user},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as upstream:

            async def client_to_upstream():
                try:
                    while True:
                        data = await ws.receive_text()
                        await upstream.send(data)
                except WebSocketDisconnect:
                    pass

            async def upstream_to_client():
                try:
                    async for message in upstream:
                        if isinstance(message, str):
                            await ws.send_text(message)
                        else:
                            await ws.send_bytes(message)
                except websockets.ConnectionClosed:
                    pass

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            _done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
    except Exception:
        _proxy_logger.exception("%s WS proxy error", service_name)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def _require_sso_user(request: Request) -> str:
    """Return username from X-Remote-User header, or raise 403."""
    user = request.headers.get("x-remote-user", "")
    if not user:
        raise HTTPException(status_code=403, detail="Forbidden: SSO login required")
    return user


# --- Formgrader reverse proxy routes (must be before app.mount("/live", ...)) ---


@app.api_route("/live/grader/{path:path}", methods=PROXY_METHODS)
async def formgrader_proxy(request: Request, path: str):
    """Reverse proxy HTTP requests to formgrader on moriarty."""
    user = _check_formgrader_access(request)
    return await _proxy_http(
        request, MORIARTY_FORMGRADER, f"live/grader/{path}", user,
        timeout=30.0, service_name="Formgrader",
    )


@app.websocket("/live/grader/{path:path}")
async def formgrader_ws_proxy(ws: WebSocket, path: str):
    """Reverse proxy WebSocket connections to formgrader on moriarty."""
    user = ws.headers.get("x-remote-user", "")
    if not user or user not in _formgrader_allowed_users():
        await ws.close(code=4003, reason="Forbidden")
        return
    await _proxy_ws(
        ws, "ws://moriarty.scrtp.warwick.ac.uk:2718", f"live/grader/{path}", user,
        service_name="formgrader",
    )


# --- Hub reverse proxy routes (under /live/hub for SSO protection) ---

_HUB_DOWN_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Notebook Server Offline</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 20px; text-align: center; }
        h1 { color: #5f259f; }
        .message { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 1.5em; margin: 2em 0; }
        a { color: #0066cc; }
        .retry { margin-top: 2em; }
        .retry a { background: #5f259f; color: white; padding: 10px 24px; border-radius: 6px; text-decoration: none; }
        .retry a:hover { background: #4a1d7a; }
    </style>
</head>
<body>
    <h1>Notebook Server Offline</h1>
    <div class="message">
        <p>The notebook editing server is currently <strong>not running</strong>.</p>
        <p>This is expected outside of active assignment periods.</p>
        <p>Please ask your instructor to start it up, then try again.</p>
    </div>
    <div class="retry">
        <a href="javascript:location.reload()">Retry</a>
    </div>
</body>
</html>"""


@app.get("/live/hub")
async def hub_redirect():
    """Redirect /live/hub to /live/hub/ so the {path:path} pattern matches."""
    return RedirectResponse("/live/hub/")


@app.api_route("/live/hub/{path:path}", methods=PROXY_METHODS)
async def hub_proxy(request: Request, path: str):
    """Reverse proxy HTTP requests to mograder hub on RONIN."""
    user = _require_sso_user(request)
    return await _proxy_http(
        request, MORIARTY_HUB, path, user,
        timeout=60.0, service_name="Hub",
        error_html=_HUB_DOWN_HTML,
    )


@app.websocket("/live/hub/{path:path}")
async def hub_ws_proxy(ws: WebSocket, path: str):
    """Reverse proxy WebSocket connections to mograder hub on RONIN."""
    user = ws.headers.get("x-remote-user", "")
    if not user:
        await ws.close(code=4003, reason="Forbidden")
        return
    await _proxy_ws(
        ws, "ws://localhost:18080", path, user,
        service_name="hub",
    )


# Mount marimo server at /live (SSO protected path)
app.mount("/live", server.build())

# Mount marimo server at /demos (public path)
app.mount("/demos", demo_server.build())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=2718)
