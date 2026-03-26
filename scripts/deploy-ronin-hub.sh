#!/usr/bin/env bash
# Warwick-specific post-deploy configuration for the mograder hub on RONIN.
#
# Run the generic deploy script first:
#   bash /path/to/mograder/scripts/deploy-hub-aws.sh --host sciml.warwick.cloud --key ~/.ssh/mograder.pem
#
# Then run this script:
#   bash scripts/deploy-ronin-hub.sh [--setup-tunnel]
#
# What this does (on top of the generic deploy):
#   1. Adds Moodle transport config to mograder.toml
#   2. Optionally sets up the SSH tunnel from sciml to the RONIN hub
#
# Note: RONIN instances get a new public IP on each stop/start. The SSH
# tunnel reconnects automatically (DNS name stays the same), but there
# may be a brief outage while DNS updates. Ask your RONIN admin to assign
# an Elastic IP for a stable address.
set -euo pipefail

HOST="sciml.warwick.cloud"
KEY="$HOME/.ssh/mograder.pem"
SCIML="sciml"
SETUP_TUNNEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --setup-tunnel) SETUP_TUNNEL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new ubuntu@$HOST"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Warwick RONIN hub configuration ===${NC}"
echo "Host: $HOST"

# Check the hub is already deployed
STATUS=$($SSH "sudo systemctl is-active mograder-hub 2>/dev/null || echo inactive")
if [ "$STATUS" != "active" ]; then
    echo "ERROR: mograder-hub service is not running on $HOST"
    echo "Run the generic deploy script first:"
    echo "  bash /path/to/mograder/scripts/deploy-hub-aws.sh --host $HOST --key $KEY"
    exit 1
fi
echo -e "  Hub service: ${GREEN}active${NC}"

# --- Step 1: Add Moodle config ---
echo -e "\n${YELLOW}Step 1: Moodle transport config${NC}"

$SSH "sudo -u mograder bash -s" << 'MOODLE'
set -euo pipefail
CONF=/srv/mograder/course/mograder.toml

if grep -q '\[moodle\]' "$CONF" 2>/dev/null; then
    echo "  [moodle] section already present — skipping"
else
    echo "  Adding Moodle config..."
    cat >> "$CONF" << 'EOF'

jobs = 16
transport = "moodle"

[moodle]
url = "https://moodle-staging.warwick.ac.uk"
course_id = 65017
EOF
fi
MOODLE

# --- Step 2: SSH tunnel on sciml (optional) ---
if [ "$SETUP_TUNNEL" = true ]; then
    echo -e "\n${YELLOW}Step 2: SSH tunnel on sciml${NC}"

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

echo -e "\n${GREEN}=== Warwick configuration complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Publish assignments: mograder hub publish ASSIGNMENT --url http://localhost:28080 --token TOKEN --force"
echo "     (requires local SSH tunnel: ssh -L 28080:localhost:8080 ubuntu@$HOST)"
echo "  2. Warm cache: mograder hub warm-cache --url http://localhost:28080 --token TOKEN"
echo "  3. Ensure sciml app.py has: MORIARTY_HUB = \"http://localhost:18080\""
echo "  4. Test: https://sciml.warwick.ac.uk/live/hub/"
