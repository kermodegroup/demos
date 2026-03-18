#!/usr/bin/env python3
"""Generate index.html for GitHub Pages WASM deployment."""

import argparse
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="_site")
    args = parser.parse_args()

    # Load demo config for titles/ordering
    config_file = Path("demos.toml")
    demo_config = []
    if config_file.exists():
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
            demo_config = config.get("demos", [])

    config_by_name = {d["name"]: d for d in demo_config}

    # Find built WASM files
    output_dir = Path(args.output_dir)
    wasm_files = sorted(output_dir.glob("*.html"))
    wasm_files = [f for f in wasm_files if f.name != "index.html"]

    # Build notebook list
    notebooks = []
    for f in wasm_files:
        name = f.stem
        cfg = config_by_name.get(name, {})
        # Skip hidden notebooks
        if cfg.get("hidden", False):
            continue
        title = cfg.get("title", name.replace("-", " ").title())
        notebooks.append((name, title))

    # Sort by config order
    config_order = [d["name"] for d in demo_config]
    notebooks.sort(
        key=lambda x: (config_order.index(x[0]) if x[0] in config_order else 999, x[0])
    )

    # Identify live notebooks from config (those not in WASM output)
    wasm_names = {n for n, _ in notebooks}
    molab_base = "https://molab.marimo.io/github/kermodegroup/demos/blob/main"
    molab_params = "/wasm?include-code=false"
    live_notebooks = []
    for d in demo_config:
        name = d["name"]
        if name in wasm_names or d.get("hidden", False) or d.get("type") == "demo":
            continue
        title = d.get("title", name.replace("-", " ").title())
        molab_url = f"{molab_base}/notebooks/{name}.py{molab_params}"
        live_notebooks.append((name, title, molab_url))

    live_notebooks.sort(
        key=lambda x: (config_order.index(x[0]) if x[0] in config_order else 999, x[0])
    )

    # Generate HTML
    wasm_links = "\n".join(
        f'        <li><a href="{name}.html">{title}</a> <span class="badge wasm">WASM</span>'
        f' <a href="{molab_base}/apps/{name}.py{molab_params}" class="molab-link" title="Open in molab (no code)">molab</a></li>'
        for name, title in notebooks
    )
    live_links = "\n".join(
        f'        <li><a href="{molab_url}" class="molab-link-primary">{title}</a>'
        f' <span class="badge live">LIVE</span>'
        f' <a href="{molab_url}" class="molab-link" title="Open in molab (no login required)">molab</a></li>'
        for _, title, molab_url in live_notebooks
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SciML Demos - University of Warwick</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #5f259f; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ margin: 10px 0; }}
        a {{ color: #0066cc; text-decoration: none; font-size: 1.1em; }}
        a:hover {{ text-decoration: underline; }}
        .badge {{ font-size: 0.7em; padding: 2px 6px; border-radius: 3px; margin-left: 8px; }}
        .wasm {{ background: #d4edda; color: #155724; }}
        .live {{ background: #fff3cd; color: #856404; }}
        .molab-link {{ font-size: 0.75em; padding: 2px 6px; border-radius: 3px; margin-left: 6px; background: #e8d5f5; color: #5f259f; text-decoration: none; }}
        .molab-link:hover {{ background: #d4b8eb; text-decoration: none; }}
        .molab-link-primary {{ color: #0066cc; text-decoration: none; font-size: 1.1em; }}
        .molab-link-primary:hover {{ text-decoration: underline; }}
        .note {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1em; margin: 1.5em 0; }}
    </style>
</head>
<body>
    <h1>SciML Notebooks</h1>
    <p>Interactive scientific machine learning demonstrations.
    Developed by <a href="https://warwick.ac.uk/jrkermode">James Kermode</a>.</p>

    <h2>Interactive Demos (WASM)</h2>
    <p>These run entirely in your browser &mdash; no server required.</p>
    <ul>
{wasm_links}
    </ul>

    <h2>Live Notebooks</h2>
    <p>These require JAX or other native dependencies.
    Available at <a href="https://sciml.warwick.ac.uk/">sciml.warwick.ac.uk</a> (Warwick SSO)
    or via <strong style="color: #5f259f;">molab</strong> links below (no login required).</p>
    <ul>
{live_links}
    </ul>

    <div class="note">
        <p><strong style="color: #5f259f;">molab</strong> links open notebooks in
        <a href="https://docs.marimo.io/guides/molab/">marimo's free cloud environment</a> &mdash;
        no login or installation required.</p>
    </div>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html)
    print(f"Generated {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
