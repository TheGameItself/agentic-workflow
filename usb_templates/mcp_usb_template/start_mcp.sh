#!/bin/bash
# MCP Agentic Workflow Accelerator - USB Launcher
# This script automatically detects the platform and launches the appropriate version

echo "MCP Agentic Workflow Accelerator - USB Edition"
echo "=============================================="

# Detect platform
PLATFORM=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Detected platform: $PLATFORM"

# Check if platform-specific version exists
PLATFORM_DIR="platforms/$PLATFORM"
if [[ ! -d "$PLATFORM_DIR" ]]; then
    echo "No version available for $PLATFORM"
    echo "Available platforms:"
    ls -1 platforms/
    exit 1
fi

# Launch the appropriate version
cd "$PLATFORM_DIR"
if [[ "$PLATFORM" == "windows" ]]; then
    if [[ -f "start_mcp.bat" ]]; then
        ./start_mcp.bat
    elif [[ -f "mcp_agentic_workflow.exe" ]]; then
        ./mcp_agentic_workflow.exe
    else
        echo "No Windows executable found"
        exit 1
    fi
else
    if [[ -f "start_mcp.sh" ]]; then
        ./start_mcp.sh
    elif [[ -f "mcp_agentic_workflow" ]]; then
        ./mcp_agentic_workflow
    else
        echo "No $PLATFORM executable found"
        exit 1
    fi
fi
