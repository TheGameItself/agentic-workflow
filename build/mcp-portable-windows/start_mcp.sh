#!/bin/bash
# MCP Agentic Workflow Accelerator - Launcher Script
# This script starts the MCP server with proper environment setup

echo "🚀 Starting MCP Agentic Workflow Accelerator..."
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or later"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "❌ Python $PYTHON_VERSION detected. Python $REQUIRED_VERSION or later is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Install dependencies if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
    echo "📦 Checking dependencies..."
    python3 -m pip install -r requirements.txt --quiet
fi

# Check if the main CLI script exists
if [[ -f "mcp_cli.py" ]]; then
    echo "🎯 Starting MCP CLI..."
    python3 mcp_cli.py "$@"
elif [[ -f "src/mcp/cli.py" ]]; then
    echo "🎯 Starting MCP CLI from src..."
    python3 -m src.mcp.cli "$@"
else
    echo "❌ MCP CLI script not found"
    echo "Available files:"
    ls -la *.py 2>/dev/null || echo "No .py files in current directory"
    ls -la src/mcp/*.py 2>/dev/null || echo "No .py files in src/mcp/"
    exit 1
fi 