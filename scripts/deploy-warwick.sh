#!/bin/bash
set -e

# Configuration
DEMOS_DIR=$(realpath $(dirname $0)/..)
REMOTE="sciml"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SciML Live Notebooks Deployment ===${NC}"
echo -e "WASM notebooks are deployed via GitHub Pages (kermodegroup.github.io/demos)"

# Change to demos directory
cd "$DEMOS_DIR"
echo "Working directory: $(pwd)"

# Categorize notebooks to identify live notebooks
echo -e "\n${YELLOW}Categorizing notebooks...${NC}"
uv run python scripts/categorize_notebooks.py --output-wasm wasm_notebooks.txt --output-live live_notebooks.txt

# Check what we're deploying
echo -e "\n${YELLOW}Deployment summary:${NC}"
WASM_COUNT=$(cat wasm_notebooks.txt 2>/dev/null | grep -c . || echo 0)
LIVE_COUNT=$(cat live_notebooks.txt 2>/dev/null | grep -c . || echo 0)
echo "  WASM notebooks (GitHub Pages): $WASM_COUNT"
echo "  Live notebooks (this server):  $LIVE_COUNT"

# Sync server venv with pyproject.toml
echo -e "\n${YELLOW}Syncing server dependencies...${NC}"
scp pyproject.toml ${REMOTE}:~/marimo-server/
ssh ${REMOTE} 'cd ~/marimo-server && uv sync --extra jax --extra numpyro --extra server'

# Deploy demos.toml config and app.py
echo -e "\n${YELLOW}Deploying server config...${NC}"
scp demos.toml ${REMOTE}:~/marimo-server/
scp server/app.py ${REMOTE}:~/marimo-server/
[ -f server/workshops.py ] && scp server/workshops.py ${REMOTE}:~/marimo-server/

# Fetch latest workshop keys from PX914 CI
echo -e "\n${YELLOW}Fetching workshop keys from PX914 CI...${NC}"
PX914_REPO="HetSys/PX914"
LATEST_RUN=$(gh run list --repo "$PX914_REPO" --workflow "deploy-wasm.yml" --status success --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)
if [ -n "$LATEST_RUN" ]; then
    KEYS_TMP=$(mktemp -d)
    if gh run download "$LATEST_RUN" --repo "$PX914_REPO" --name workshop-keys --dir "$KEYS_TMP" 2>/dev/null; then
        for keys_file in "$KEYS_TMP"/*/keys_all.json; do
            ws_name=$(basename "$(dirname "$keys_file")" | sed 's/-.*//') # L00a-ProbabilityFoundations -> L00a
            mkdir -p "server/workshops/$ws_name"
            cp "$keys_file" "server/workshops/$ws_name/keys_all.json"
            echo "  Updated: $ws_name/keys_all.json"
        done
        rm -rf "$KEYS_TMP"

        # Generate dashboard.html for each workshop (with SSO token)
        echo -e "  ${YELLOW}Generating workshop dashboards...${NC}"
        uv run python -c "
import json
from pathlib import Path
from mograder.transport.workshop import generate_dashboard_html

for ws_dir in sorted(Path('server/workshops').iterdir()):
    keys_all_file = ws_dir / 'keys_all.json'
    if not keys_all_file.exists():
        continue
    exercises = list(json.loads(keys_all_file.read_text()).keys())
    html = generate_dashboard_html(exercises, title=f'Workshop: {ws_dir.name}')
    # Patch token: replace hash-based extraction with fixed SSO token
    html = html.replace(
        \"let TOKEN = location.hash.replace('#token=', '');\",
        \"let TOKEN = 'sso';\"
    )
    (ws_dir / 'dashboard.html').write_text(html)
    print(f'  Generated: {ws_dir.name}/dashboard.html')
"
    else
        echo "  Warning: could not download keys artifact (may have expired)"
    fi
else
    echo "  Warning: no successful PX914 CI run found"
fi

# Deploy workshop keys (keys_all.json always; keys.json only if not already on server)
if [ -d "server/workshops" ]; then
    echo -e "\n${YELLOW}Deploying workshop keys...${NC}"
    for ws_dir in server/workshops/*/; do
        [ -d "$ws_dir" ] || continue
        ws_name=$(basename "$ws_dir")
        echo "  Workshop: $ws_name"
        ssh ${REMOTE} "mkdir -p ~/marimo-server/workshops/$ws_name"
        [ -f "$ws_dir/keys_all.json" ] && scp "$ws_dir/keys_all.json" ${REMOTE}:~/marimo-server/workshops/$ws_name/
        ssh ${REMOTE} "[ -f ~/marimo-server/workshops/$ws_name/keys.json ] || echo '{}' > ~/marimo-server/workshops/$ws_name/keys.json"
    done
fi

# Deploy live notebooks (JAX-dependent) - flatten to just basenames
echo -e "\n${YELLOW}Deploying live notebooks...${NC}"
if [ -s live_notebooks.txt ]; then
    # Clear existing notebooks first
    ssh ${REMOTE} 'rm -f ~/marimo-server/notebooks/*.py'
    # Copy each notebook with flat structure
    while IFS= read -r notebook || [ -n "$notebook" ]; do
        [ -z "$notebook" ] && continue
        echo "  Deploying: $notebook"
        scp "$notebook" ${REMOTE}:~/marimo-server/notebooks/
    done < live_notebooks.txt
else
    echo "  No live notebooks to deploy"
    # Clear remote notebooks directory if no live notebooks
    ssh ${REMOTE} 'rm -f ~/marimo-server/notebooks/*.py'
fi

# Deploy presentations (public demos)
echo -e "\n${YELLOW}Deploying presentations...${NC}"
PRESENTATIONS_DIR="${DEMOS_DIR}/presentations"
if [ -d "$PRESENTATIONS_DIR" ] && [ "$(ls -A "$PRESENTATIONS_DIR" 2>/dev/null)" ]; then
    ssh ${REMOTE} 'rm -rf ~/marimo-server/presentations && mkdir -p ~/marimo-server/presentations'
    for pres_dir in "$PRESENTATIONS_DIR"/*/; do
        [ -d "$pres_dir" ] || continue
        pres_name=$(basename "$pres_dir")
        echo "  Deploying presentation: $pres_name"
        scp -r "$pres_dir" ${REMOTE}:~/marimo-server/presentations/
    done
else
    echo "  No presentations to deploy"
fi

# Restart server
echo -e "\n${YELLOW}Restarting server...${NC}"
ssh ${REMOTE} '~/marimo-server/deploy.sh'

# Cleanup temporary files
rm -f wasm_notebooks.txt live_notebooks.txt

echo -e "\n${GREEN}=== Deployment complete ===${NC}"
echo "Live notebooks: https://sciml.warwick.ac.uk/"
echo "WASM notebooks: https://kermodegroup.github.io/demos/"
