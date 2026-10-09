import asyncio
import marimo
import tomllib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import logging
import os
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
FORMGRADER = "http://localhost:12718"  # SSH tunnel to sciml-grader.warwick.cloud (RONIN)
MORIARTY_HUB = "http://localhost:18080"  # SSH tunnel to sciml.warwick.cloud (RONIN)
FORMGRADER_USERS_FILE = Path(__file__).parent / "formgrader_users.txt"
# Optional hub allowlist: while this file exists, /live/hub admits only the
# users listed in it (e.g. during testing, before the module opens); delete it
# to open the hub to every SSO user. Read on each request, no restart needed.
HUB_USERS_FILE = Path(__file__).parent / "hub_users.txt"
WORKSHOPS_DIR = Path(__file__).parent / "workshops"  # keys.json + keys_all.json per workshop

app = FastAPI()


class DropUntrustedRemoteUser:
    """Strip X-Remote-User from incoming requests unless Apache is known to set it.

    Users are identified only by the X-Remote-User header, which Apache must set
    from the Shibboleth session (`RequestHeader set X-Remote-User ...` in
    <Location /live>). If Apache does not set it, any value that arrives was sent
    by the client, and trusting it would let one user impersonate another (e.g.
    an instructor). Set TRUST_SSO_HEADER=0 in the service environment while the
    Apache line is missing; the raw value stays visible at /live/debug-headers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            kept = [(k, v) for k, v in scope["headers"] if k != b"x-remote-user"]
            if len(kept) != len(scope["headers"]):
                dropped = dict(scope["headers"]).get(b"x-remote-user", b"")
                scope = {**scope, "headers": kept,
                         "untrusted_remote_user": dropped.decode("latin-1")}
        await self.app(scope, receive, send)


TRUST_SSO_HEADER = os.environ.get("TRUST_SSO_HEADER", "1") != "0"
if not TRUST_SSO_HEADER:
    app.add_middleware(DropUntrustedRemoteUser)

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


def _hub_user_allowed(user: str) -> bool:
    if not HUB_USERS_FILE.exists():
        return True
    allowed = {
        line.strip()
        for line in HUB_USERS_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return user in allowed


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
        "trust_sso_header": TRUST_SSO_HEADER,
        # value received but discarded while TRUST_SSO_HEADER=0 (once Apache sets
        # the header again this shows your own username)
        "x-remote-user-discarded": request.scope.get("untrusted_remote_user", "(none)"),
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
        # a down upstream should fail fast: cap the connect phase, whatever
        # the read timeout (a dropped connection times out, not refuses)
        client_timeout = httpx.Timeout(timeout, connect=min(timeout, 5.0))
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        if error_html:
            return HTMLResponse(content=error_html, status_code=502)
        raise HTTPException(
            status_code=502,
            detail=f"{service_name} server is not responding",
        )

    # multi_items: keep repeated headers apart (a dict would merge several
    # Set-Cookie headers into one, which browsers then misread)
    response = Response(content=resp.content, status_code=resp.status_code)
    for k, v in resp.headers.multi_items():
        # content-length: Response has set it for the body actually sent
        if k.lower() not in _HOP_BY_HOP and k.lower() != "content-length":
            response.headers.append(k, v)
    return response


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
        # cookies too: the upstream app reads per-browser settings from them
        # on the WebSocket (e.g. an instructor's "view as student" switch)
        upstream_headers = {"x-remote-user": user}
        if ws.headers.get("cookie"):
            upstream_headers["cookie"] = ws.headers["cookie"]
        async with websockets.connect(
            target_url,
            additional_headers=upstream_headers,
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


_FORMGRADER_DOWN_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Formgrader Offline</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 20px; text-align: center; }
        h1 { color: #5f259f; }
        .message { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 1.5em; margin: 2em 0; }
        code { background: #f4f4f4; padding: 0 4px; }
        .retry a { background: #5f259f; color: white; padding: 10px 24px; border-radius: 6px; text-decoration: none; }
    </style>
</head>
<body>
    <h1>Formgrader Offline</h1>
    <div class="message">
        <p>The formgrader is <strong>not reachable</strong>: the grader instance
        (<code>sciml-grader.warwick.cloud</code>) may be stopped, or the
        <code>mograder-grader-tunnel</code> service on sciml is down.</p>
        <p>Start the grader in RONIN, then try again.</p>
    </div>
    <p class="retry"><a href="/live/grader/">Retry</a></p>
</body>
</html>"""


@app.get("/live/grader")
async def formgrader_redirect():
    """Redirect /live/grader to /live/grader/ so the {path:path} pattern matches."""
    return RedirectResponse("/live/grader/")


@app.api_route("/live/grader/{path:path}", methods=PROXY_METHODS)
async def formgrader_proxy(request: Request, path: str):
    """Reverse proxy HTTP requests to the formgrader on RONIN (via the tunnel)."""
    user = _check_formgrader_access(request)
    return await _proxy_http(
        request, FORMGRADER, f"live/grader/{path}", user,
        timeout=30.0, service_name="Formgrader",
        error_html=_FORMGRADER_DOWN_HTML,
    )


@app.websocket("/live/grader/{path:path}")
async def formgrader_ws_proxy(ws: WebSocket, path: str):
    """Reverse proxy WebSocket connections to the formgrader on RONIN."""
    user = ws.headers.get("x-remote-user", "")
    if not user or user not in _formgrader_allowed_users():
        await ws.close(code=4003, reason="Forbidden")
        return
    await _proxy_ws(
        ws, FORMGRADER.replace("http", "ws", 1), f"live/grader/{path}", user,
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
    if not _hub_user_allowed(user):
        raise HTTPException(status_code=403, detail="The hub is not open yet")
    return await _proxy_http(
        request, MORIARTY_HUB, path, user,
        timeout=60.0, service_name="Hub",
        error_html=_HUB_DOWN_HTML,
    )


@app.websocket("/live/hub/{path:path}")
async def hub_ws_proxy(ws: WebSocket, path: str):
    """Reverse proxy WebSocket connections to mograder hub on RONIN."""
    user = ws.headers.get("x-remote-user", "")
    if not user or not _hub_user_allowed(user):
        await ws.close(code=4003, reason="Forbidden")
        return
    await _proxy_ws(
        ws, "ws://localhost:18080", path, user,
        service_name="hub",
    )


class InstructorOnly:
    """ASGI wrapper: only users in formgrader_users.txt (by SSO header) get through.

    The workshop dashboards authenticate with a fixed token, so without this any
    Warwick SSO user could release or revoke workshop solution keys.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            user = dict(scope["headers"]).get(b"x-remote-user", b"").decode("latin-1")
            if not user or user not in _formgrader_allowed_users():
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 4003})
                    return
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body",
                            "body": b"Forbidden: instructor access required"})
                return
        await self.app(scope, receive, send)


# Workshop key release (public keys endpoint + instructor-only mograder dashboards)
from workshops import router as workshops_router, create_workshop_mounts
app.include_router(workshops_router)
for _ws_name, _ws_app in create_workshop_mounts().items():
    app.mount(f"/live/workshops/{_ws_name}", InstructorOnly(_ws_app))

# Mount the group wiki (MkDocs static build) at /live/wiki (SSO protected via /live).
# Must be registered before the /live catch-all mount below so it isn't shadowed.
# Content is deployed by the wiki repo's scripts/deploy-wiki.sh.
WIKI_DIR = Path(__file__).parent / "wiki-site"
if WIKI_DIR.exists():
    # Redirect the bare mount path to the trailing-slash form; without this the
    # StaticFiles mount (which only matches /live/wiki/...) is shadowed by the
    # /live catch-all below and 404s. Mirrors hub_redirect above.
    @app.get("/live/wiki")
    def wiki_redirect():
        return RedirectResponse("/live/wiki/")

    app.mount("/live/wiki", StaticFiles(directory=str(WIKI_DIR), html=True), name="wiki")

# Mount marimo server at /live (SSO protected path)
app.mount("/live", server.build())

# Mount marimo server at /demos (public path)
app.mount("/demos", demo_server.build())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=2718)
