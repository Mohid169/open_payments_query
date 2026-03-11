#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating local virtual environment in .venv ..."
  python3 -m venv "$VENV_DIR"
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT_DIR/research_extractor.py" "$@"
