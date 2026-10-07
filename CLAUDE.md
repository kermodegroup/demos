# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains interactive research demonstrations built with [marimo](https://marimo.io), a reactive Python notebook framework. Demos are deployed to two locations:
- **WASM notebooks**: https://kermodegroup.github.io/demos (public, no auth)
- **Live notebooks**: https://sciml.warwick.ac.uk (SSO protected)

## Repository Structure

```
demos/
├── apps/                    # WASM-compatible notebooks (static HTML export)
│   ├── lib/                 # Shared library modules
│   └── _mnist_data.py       # Inline MNIST data for WASM compatibility
├── notebooks/               # Live server notebooks (require native deps)
├── presentations/           # Public demo presentations (no auth)
│   └── uq-kinetics/         # UQ reaction kinetics presentation
├── scripts/
│   ├── build.py             # WASM HTML export script
│   ├── categorize_notebooks.py  # WASM vs live detection
│   ├── generate_index.py    # GitHub Pages index generator
│   ├── generate_mnist_data.py   # MNIST data generator for PCA demo
│   ├── deploy.sh            # Server-side deployment (systemd + nginx)
│   └── deploy-warwick.sh    # Local deployment script (live notebooks only)
├── server/
│   ├── app.py               # FastAPI server for live notebooks
│   ├── notebooks/           # Symlink/copy of notebooks/ for server
│   ├── presentations/       # Public presentations served at /demos
│   └── student/             # Student dashboard WASM app
├── demos.toml               # Demo ordering and display config
├── grader.toml              # Formgrader access config
├── pyproject.toml            # Project dependencies (uv managed)
└── .github/workflows/
    └── pages.yml            # GitHub Actions for WASM deployment
```

## Deployment Architecture

### Split Deployment

Notebooks are automatically categorized and deployed to different hosts:

1. **WASM (GitHub Pages)** - Notebooks in `apps/` with pure Python dependencies
   - Deployed via GitHub Actions to kermodegroup.github.io/demos
   - Exported to HTML+WASM via `marimo export html-wasm`
   - No authentication required
   - Runs entirely in browser via Pyodide

2. **Live (sciml.warwick.ac.uk)** - Notebooks in `notebooks/` with native dependencies
   - Deployed manually via `deploy-warwick.sh`
   - Served via FastAPI (`server/app.py`) using `marimo.create_asgi_app()`
   - Proxied through nginx with WebSocket support
   - Protected by University of Warwick SSO

3. **Presentations (sciml.warwick.ac.uk/demos)** - Public demo presentations
   - Served from `presentations/` directory, no auth required
   - Mounted at `/demos` path in FastAPI server

### WASM Incompatible Packages

The `scripts/categorize_notebooks.py` detects these packages and routes to live:
- ML frameworks: `jax`, `torch`, `tensorflow`
- File system: `watchdog`, `psutil`
- Database drivers: `psycopg2`, `mysqlclient`
- Native extensions: `opencv-python`, `cryptography`, `grpcio`, `pyarrow`, etc.

## Commands

### Local Development

```bash
# Run a notebook locally
marimo run apps/regression-demo.py
marimo edit notebooks/jax-test.py

# Test WASM compatibility detection
python scripts/categorize_notebooks.py --offline

# Build WASM notebooks locally
python scripts/build.py --sync-lib --output-dir _wasm_site
```

### Deployment

**WASM notebooks** deploy automatically via GitHub Actions when changes are pushed to `apps/`.

**Live notebooks** require manual deployment (2FA authentication):

```bash
# Deploy live notebooks to sciml.warwick.ac.uk
./scripts/deploy-warwick.sh
```

The deploy script:
1. Categorizes notebooks into WASM vs live
2. Syncs dependencies on server
3. Deploys live notebooks to `~/marimo-server/notebooks/`
4. Restarts marimo service on server

## Server Infrastructure

**Live server:** sciml.warwick.ac.uk

```
/home/ubuntu/marimo-server/
├── app.py           # FastAPI entry point
├── deploy.sh        # Server-side deployment script
├── notebooks/       # Live notebook .py files
├── presentations/   # Public demo presentations
├── student/         # Student dashboard WASM app
└── .venv/           # Python environment with marimo + deps
```

**Services:**
- Apache httpd + Shibboleth (`mod_shib`): SSL termination, SSO auth, reverse proxy to the app on `localhost:2718`. The host is green-walrus (`ssh sciml` = `svc_user` via green-walrus); `/etc/httpd` is managed by IT (no sudo for us).
- systemd (user units of `svc_user`): `marimo.service` (the FastAPI app), `mograder-tunnel.service` (SSH tunnel `localhost:18080` → RONIN hub `sciml.warwick.cloud:8080`)
- Let's Encrypt: SSL certificates

**SSO user header (required):** `app.py` identifies users only by the `X-Remote-User` request header, which Apache must set from the Shibboleth session inside `<Location /live>` in `/etc/httpd/conf.d/shib.conf`:

```apache
<Location /live>
  AuthType shibboleth
  ShibRequestSetting requireSession 1
  require shib-session
  RequestHeader set X-Remote-User "expr=%{REMOTE_USER}"
</Location>
```

`set` also overwrites any `X-Remote-User` a client sends, so without this line users could impersonate others. The header stopped arriving when `shib.conf` was replaced on 28 Sep 2026 (the file belongs to the shibboleth 3.6.0 package, so apparently a package update; it took effect at the 5 Oct reboot), and every `/live/hub` request then failed with `{"detail":"Forbidden: SSO login required"}`. Check with `/live/debug-headers` after any IT change or package update. IT restored the line on 7 Oct 2026 (as `RequestHeader set X-Remote-User "%{REMOTE_USER}s"`). If it goes missing again, the systemd drop-in `~/.config/systemd/user/marimo.service.d/sso-header.conf` should set `TRUST_SSO_HEADER=0` (`[Service]` / `Environment=TRUST_SSO_HEADER=0`), so `app.py` discards any client-sent `X-Remote-User` (debug-headers then reports it under `x-remote-user-discarded`). Once that shows your username coming from Apache, delete the drop-in, `systemctl --user daemon-reload && systemctl --user restart marimo`.

**Workshop dashboards:** `/live/workshops/{name}/` (dashboard and release/revoke API) is wrapped in `InstructorOnly` in `app.py`: only users listed in `formgrader_users.txt` get through (the dashboards' own token is a fixed `"sso"`). The public `/workshops/{name}/keys.json` is unaffected.

**Server routes:**
- `/` - Index page listing all notebooks with WASM/LIVE/DEMO badges
- `/live/{name}/` - Live notebooks (SSO protected)
- `/wasm/{name}/` - Redirects to GitHub Pages WASM exports
- `/demos/{name}/` - Public presentations
- `/live/student/` - Student dashboard with assignment API proxy
- `/live/grader/` - Formgrader reverse proxy to moriarty (staff only)

**Formgrader integration:**
- Reverse proxies HTTP and WebSocket to `moriarty.scrtp.warwick.ac.uk:2718`
- Access controlled via `formgrader_users.txt` on server
- Student API proxy allows any SSO user to access assignments

## GitHub Pages Configuration

After initial setup, configure GitHub Pages in repo settings:
1. Go to Settings → Pages
2. Change Source from "Deploy from a branch" to "GitHub Actions"

## Marimo Notebook Conventions

- Include dependencies in PEP 723 script metadata at top of file
- Avoid `watchdog` and other dev-only deps that break WASM
- Use `mo.ui.*` for interactive elements
- Last expression in cell is auto-displayed
- For matplotlib: use `plt.gca()` not `plt.show()`

## Library Modules (`apps/lib/`)

Shared code for regression demos:

| Module | Description |
|--------|-------------|
| `models.py` | `MyBayesianRidge`, `ConformalPrediction`, `NeuralNetworkRegression`, `QuantileRegressionUQ` |
| `kernels.py` | GP kernels (RBF, Matérn, bump, polynomial) |
| `basis.py` | Basis feature functions (RBF, Fourier, LJ) |
| `metrics.py` | Probabilistic metrics (log likelihood, CRPS) |
| `data.py` | Ground truth functions and data generation |
| `optimization.py` | GP hyperparameter optimization |

**Sync workflow:** When modifying lib/, run `--check-sync` to verify inline copies in notebooks match.

## Demo Configuration (`demos.toml`)

Controls display order and titles on the index page. Each entry has:
- `name` - matches the notebook filename (without `.py`)
- `title` - display title on the index page
- `type` - optional, e.g. `"demo"` for presentations
- `hidden` - optional, hides from index if `true`

Notebooks not in `demos.toml` still work but appear after configured ones.

## Dependencies

Managed with `uv` via `pyproject.toml`. Optional dependency groups:
- **jax**: JAX, tinygp, equinox, optax, optimistix (for GP and neural ODE demos)
- **numpyro**: NumPyro, arviz, diffrax (for BNN and probabilistic demos)
- **server**: FastAPI, uvicorn, httpx, websockets (for live server)
- **dev**: uv
