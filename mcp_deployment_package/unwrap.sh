#!/bin/bash
# MCP Agentic Workflow Accelerator - Unwrap Script
# This script extracts and sets up the MCP system from the archive

set -e  # Exit on any error

echo "📦 MCP Agentic Workflow Accelerator - Unwrap Script"
echo "=================================================="

# Check if archive exists
if [ ! -f "mcp_system.zip" ]; then
    echo "❌ Archive not found: mcp_system.zip"
    echo "   Make sure you're running this script from the deployment package directory."
    exit 1
fi

# Get target directory name
if [ -n "$1" ]; then
    TARGET_DIR="$1"
else
    TARGET_DIR="mcp_workflow_$(date +%Y%m%d_%H%M%S)"
fi

echo "📁 Target directory: $TARGET_DIR"

# Create target directory
mkdir -p "$TARGET_DIR"

# Extract archive
echo "📦 Extracting archive..."
if [[ "mcp_system.zip" == *.zip ]]; then
    unzip -q "mcp_system.zip" -d "$TARGET_DIR"
elif [[ "mcp_system.zip" == *.tar.gz ]]; then
    tar -xzf "mcp_system.zip" -C "$TARGET_DIR"
else
    echo "❌ Unsupported archive format: mcp_system.zip"
    exit 1
fi

echo "✅ Archive extracted successfully"

# Navigate to target directory
cd "$TARGET_DIR"

# Run setup
echo "🔧 Running setup..."
if [ -f "setup.sh" ]; then
    chmod +x setup.sh
    ./setup.sh
elif [ -f "setup.bat" ]; then
    echo "⚠️  Windows setup script found. Please run 'setup.bat' manually on Windows."
else
    echo "⚠️  No setup script found. Installing dependencies manually..."
    pip3 install -r requirements.txt
    python3 test_system.py
fi

echo ""
echo "🎉 MCP Agentic Workflow Accelerator is ready!"
echo "📁 Location: $TARGET_DIR"
echo ""
echo "📋 Next steps:"
echo "  cd $TARGET_DIR"
echo "  python3 mcp_cli.py --help"
echo "  python3 test_system.py"
echo ""
echo "📖 Read README.md for detailed usage instructions"
