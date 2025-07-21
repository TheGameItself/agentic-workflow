#!/usr/bin/env bash
set -e

# Bootstrap a local virtual environment and install all requirements locally
VENV_DIR="venv"
REQ_FILE="requirements.txt"
GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

if [ ! -d "$VENV_DIR" ]; then
  echo "[MCP] Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

# Ensure pip is available in the venv
if ! "$PYTHON_BIN" -m pip --version &> /dev/null; then
  echo "[MCP] pip not found in venv, bootstrapping with get-pip.py..."
  curl -sS "$GET_PIP_URL" -o get-pip.py
  "$PYTHON_BIN" get-pip.py
  rm get-pip.py
fi

# Install requirements
if [ -f "$REQ_FILE" ]; then
  echo "[MCP] Installing requirements from $REQ_FILE..."
  "$PIP_BIN" install --upgrade pip
  "$PIP_BIN" install -r "$REQ_FILE"
else
  echo "[MCP] ERROR: $REQ_FILE not found!"
  exit 1
fi

echo "[MCP] Local environment ready. Activate with: source $VENV_DIR/bin/activate" 