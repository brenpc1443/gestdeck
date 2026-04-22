#!/usr/bin/env bash
# start.sh — Lanza backend Python y frontend Electron juntos (macOS/Linux).
# Uso:
#   ./start.sh                                    # sin sesión
#   ./start.sh sessions/ejemplo_demo/config.json

set -euo pipefail
cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

SESSION_CONFIG="${1:-}"
if [ -n "$SESSION_CONFIG" ]; then
  python -m core.main --config "$SESSION_CONFIG" &
else
  python -m core.main &
fi
BACKEND_PID=$!
trap "kill $BACKEND_PID" EXIT

sleep 1.5
cd app
npm run dev
