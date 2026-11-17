#!/usr/bin/env bash
# Resolve paths independently of the caller's current working directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x venv/bin/python ]]; then
  PYTHON=venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
elif [[ -x venv/Scripts/python.exe ]]; then
  PYTHON=venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi
exec "$PYTHON" scripts/run_app.py "$@"
