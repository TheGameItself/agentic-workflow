#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
else
    source .venv/bin/activate
fi
python -m src.mcp.cli server
