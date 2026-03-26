#!/usr/bin/env bash
# Deploy the mograder hub to the RONIN instance for Warwick ES98E.
#
# Usage:
#   bash scripts/deploy-ronin-hub.sh [--setup-tunnel]
#
# Options:
#   --setup-tunnel    Also set up the SSH tunnel systemd service on sciml
#
# This script provisions mograder.warwick.cloud with the hub service,
# then optionally sets up the SSH tunnel from sciml for reverse proxying.
#
# Note: RONIN instances get a new public IP on each stop/start. The SSH
# tunnel reconnects automatically (DNS name stays the same), but there
# may be a brief outage while DNS updates. Ask your RONIN admin to assign
# an Elastic IP for a stable address.
set -euo pipefail

HOST="mograder.warwick.cloud"
KEY="$HOME/.ssh/mograder.pem"
SSH_USER="ubuntu"
SCIML="sciml"
SETUP_TUNNEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-tunnel) SETUP_TUNNEL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new $SSH_USER@$HOST"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== RONIN mograder hub deployment (Warwick) ===${NC}"
echo "Host: $HOST"

# Check SSH key exists
if [ ! -f "$KEY" ]; then
    echo "ERROR: SSH key not found at $KEY"
    echo "Download the .pem from RONIN and run: mv ~/Downloads/mograder.pem ~/.ssh/mograder.pem && chmod 600 ~/.ssh/mograder.pem"
    exit 1
fi

# --- Phase 1: System setup ---
echo -e "\n${YELLOW}Phase 1: System setup${NC}"

$SSH "bash -s" << 'SETUP'
set -euo pipefail

# Mount data volume
if mountpoint -q /srv/mograder; then
    echo "  /srv/mograder already mounted"
else
    echo "  Mounting data volume at /srv/mograder..."
    # Find the extra volume (RONIN mounts it at /mnt/sdd by default)
    VOLUME=""
    for dev in /dev/nvme1n1 /dev/xvdf; do
        if [ -b "$dev" ]; then VOLUME="$dev"; break; fi
    done
    if [ -z "$VOLUME" ]; then
        echo "ERROR: No extra volume found"; exit 1
    fi
    sudo umount "$VOLUME" 2>/dev/null || true
    sudo mkdir -p /srv/mograder
    if ! sudo blkid "$VOLUME" | grep -q ext4; then
        sudo mkfs.ext4 "$VOLUME"
    fi
    sudo mount "$VOLUME" /srv/mograder
    sudo sed -i '\|/mnt/sdd|d' /etc/fstab
    if ! grep -q /srv/mograder /etc/fstab; then
        echo "$VOLUME /srv/mograder ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    fi
fi

# Install bubblewrap
if command -v bwrap &>/dev/null; then
    echo "  bubblewrap already installed"
else
    echo "  Installing bubblewrap..."
    sudo apt-get update -qq && sudo apt-get install -y -qq bubblewrap
fi

# Create service user
if id mograder &>/dev/null; then
    echo "  mograder user already exists"
else
    echo "  Creating mograder user..."
    sudo useradd -r -m -s /bin/bash mograder
fi
sudo chown -R mograder:mograder /srv/mograder

# Install uv
if sudo -u mograder bash -c 'test -x ~/.local/bin/uv'; then
    echo "  uv already installed"
else
    echo "  Installing uv..."
    sudo -u mograder bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
SETUP

# --- Phase 2: Application setup ---
echo -e "\n${YELLOW}Phase 2: Application setup${NC}"

$SSH "sudo -u mograder bash -s" << 'APP'
set -euo pipefail
cd /srv/mograder
mkdir -p course
cd course

if [ ! -f pyproject.toml ]; then
    echo "  Initialising uv project..."
    ~/.local/bin/uv init --bare
fi

echo "  Installing mograder[hub]..."
~/.local/bin/uv add "mograder[hub]>=0.2.6" --refresh-package mograder 2>&1 | tail -3
APP

# Write mograder.toml (only if it doesn't exist)
$SSH "sudo -u mograder bash -s" << 'TOML'
set -euo pipefail
CONF=/srv/mograder/course/mograder.toml
if [ -f "$CONF" ]; then
    echo "  mograder.toml already exists — skipping"
else
    echo "  Writing mograder.toml..."
    cat > "$CONF" << 'EOF'
jobs = 16
transport = "moodle"

[defaults]
headless_edit = true
timeout = 300

[rlimits]
cpu = 300
nproc = 256
nofile = 256
as = 8589934592  # 8 GB (PyTorch needs >2 GB)

[security]
use_bubblewrap = true

[hub]
port = 8080
notebooks_dir = "hub-notebooks"
release_dir = "hub-release"
session_ttl = 3600
trusted_header = "X-Remote-User"
uv_cache_dir = "/srv/mograder/.uv-cache"

[moodle]
url = "https://moodle-staging.warwick.ac.uk"
course_id = 65017
EOF
fi
TOML

# --- Phase 3: Secrets and service ---
echo -e "\n${YELLOW}Phase 3: Secrets and systemd service${NC}"

$SSH "bash -s" << 'SERVICE'
set -euo pipefail

sudo mkdir -p /etc/mograder
if [ -f /etc/mograder/hub-secret ]; then
    echo "  Hub secret already exists — keeping"
else
    echo "  Generating hub secret..."
    python3 -c "import secrets; print(secrets.token_hex(32))" | sudo tee /etc/mograder/hub-secret > /dev/null
    sudo chmod 600 /etc/mograder/hub-secret
fi

echo "  Writing /etc/mograder/env..."
SECRET=$(sudo cat /etc/mograder/hub-secret)
sudo bash -c "cat > /etc/mograder/env << EOF
MOGRADER_HUB_SECRET=$SECRET
MOGRADER_COURSE_DIR=/srv/mograder/course
EOF"
sudo chmod 600 /etc/mograder/env

echo "  Writing mograder-hub.service..."
sudo bash -c 'cat > /etc/systemd/system/mograder-hub.service << EOF
[Unit]
Description=mograder hub (student-facing)
After=network.target

[Service]
Type=simple
User=mograder
Group=mograder
WorkingDirectory=/srv/mograder/course
EnvironmentFile=/etc/mograder/env
Environment=PATH=/home/mograder/.local/bin:/usr/local/bin:/usr/bin
ExecStart=/home/mograder/.local/bin/uv run mograder hub \
    --port 8080 --host 0.0.0.0 --session-ttl 3600 --headless
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF'

sudo systemctl daemon-reload
sudo systemctl enable --now mograder-hub
sudo systemctl restart mograder-hub
SERVICE

# --- Phase 4: Verification ---
echo -e "\n${YELLOW}Phase 4: Verification${NC}"
sleep 3

STATUS=$($SSH "sudo systemctl is-active mograder-hub")
if [ "$STATUS" = "active" ]; then
    echo -e "  Service: ${GREEN}active${NC}"
else
    echo "  Service: FAILED ($STATUS)"
    $SSH "sudo journalctl -u mograder-hub --no-pager -n 10"
    exit 1
fi

HTTP_CODE=$($SSH "curl -s -o /dev/null -w '%{http_code}' -H 'X-Remote-User: test' http://localhost:8080/")
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "  Hub HTTP check: ${GREEN}200 OK${NC}"
else
    echo "  Hub HTTP check: FAILED ($HTTP_CODE)"
fi

echo -e "\n${YELLOW}Instructor token:${NC}"
$SSH "sudo -u mograder bash -c 'cd /srv/mograder/course && MOGRADER_HUB_SECRET=\$(sudo cat /etc/mograder/hub-secret) ~/.local/bin/uv run mograder hub generate-token --role instructor admin'"

# --- Phase 5: SSH tunnel on sciml (optional) ---
if [ "$SETUP_TUNNEL" = true ]; then
    echo -e "\n${YELLOW}Phase 5: Setting up SSH tunnel on sciml${NC}"

    # Copy SSH key to sciml
    echo "  Copying SSH key to sciml..."
    scp "$KEY" $SCIML:~/.ssh/mograder.pem
    ssh $SCIML "chmod 600 ~/.ssh/mograder.pem"

    # Test connectivity
    echo "  Testing sciml -> RONIN connectivity..."
    ssh $SCIML "ssh -i ~/.ssh/mograder.pem -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@$HOST hostname"

    # Create tunnel service
    echo "  Creating tunnel systemd service..."
    ssh $SCIML "bash -s" << TUNNEL
set -euo pipefail
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/mograder-tunnel.service << EOF
[Unit]
Description=SSH tunnel to mograder hub on RONIN
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 18080:localhost:8080 -i %h/.ssh/mograder.pem ubuntu@$HOST
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
loginctl enable-linger \$(whoami) 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now mograder-tunnel
TUNNEL

    sleep 3
    TUNNEL_CODE=$(ssh $SCIML "curl -s -o /dev/null -w '%{http_code}' -H 'X-Remote-User: test' http://localhost:18080/")
    if [ "$TUNNEL_CODE" = "200" ]; then
        echo -e "  Tunnel check: ${GREEN}200 OK${NC}"
    else
        echo "  Tunnel check: FAILED ($TUNNEL_CODE)"
    fi
fi

echo -e "\n${GREEN}=== Deployment complete ===${NC}"
echo "Hub: http://$HOST:8080 (via SSH tunnel: localhost:18080 on sciml)"
echo ""
echo "Next steps:"
echo "  1. Publish assignments: mograder hub publish ASSIGNMENT --url http://localhost:28080 --token TOKEN --force"
echo "     (requires local SSH tunnel: ssh -L 28080:localhost:8080 -i $KEY ubuntu@$HOST)"
echo "  2. Warm cache: mograder hub warm-cache --url http://localhost:28080 --token TOKEN"
echo "  3. Ensure sciml app.py has: MORIARTY_HUB = \"http://localhost:18080\""
echo "  4. Test: https://sciml.warwick.ac.uk/live/hub/"
