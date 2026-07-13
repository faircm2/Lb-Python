#!/bin/bash
# Usage: bash run.sh <script.py>
# Installs deps system-wide (no venv) and launches the sim detached
# from the terminal so it survives SSH disconnects.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash run.sh <script.py>"
    exit 1
fi

SCRIPT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pip3 install -q -r requirements.txt --break-system-packages 2>/dev/null || pip3 install -q -r requirements.txt

LOG_FILE="run_$(basename "$SCRIPT" .py)_$(date +%Y%m%d_%H%M%S).log"
nohup python3 "$SCRIPT" > "$LOG_FILE" 2>&1 &
disown

echo "Started $SCRIPT (PID $!), logging to $LOG_FILE"
echo "tail -f $LOG_FILE"
