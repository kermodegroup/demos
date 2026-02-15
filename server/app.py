import marimo
import tomllib
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
PRESENTATIONS_DIR = Path(__file__).parent / "presentations"
CONFIG_FILE = Path(__file__).parent / "demos.toml"
GITHUB_PAGES_BASE = "https://kermodegroup.github.io/demos"

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
public_demos = []
if PRESENTATIONS_DIR.exists():
    for notebook in sorted(PRESENTATIONS_DIR.glob("*/*.py")):
        name = notebook.parent.name
        demo_server = demo_server.with_app(path=f"/{name}", root=str(notebook))
        public_demos.append(name)

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
    for name in public_demos:
        all_notebooks.append((name, f"/demos/{name}/", "demo"))

    # Sort by config order, then alphabetically
    all_notebooks.sort(key=lambda x: get_sort_key(x[0]))

    notebook_links = "".join(
        f'<li><a href="{url}">{get_display_title(name)}</a>'
        f'<span class="badge {badge_type}">{badge_type.upper()}</span></li>'
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
        <div class="note">
            <p><strong>WASM</strong> notebooks run in your browser (no login required).
            <strong>LIVE</strong> notebooks require University of Warwick SSO.
            <strong>DEMO</strong> presentations are public (no login required).</p>
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


# Mount marimo server at /live (SSO protected path)
app.mount("/live", server.build())

# Mount marimo server at /demos (public path)
app.mount("/demos", demo_server.build())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=2718)
