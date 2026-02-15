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
