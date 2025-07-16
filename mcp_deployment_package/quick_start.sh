#!/bin/bash
# MCP Agentic Workflow Accelerator - Quick Start

echo "🚀 MCP Agentic Workflow Accelerator - Quick Start"
echo "=================================================="

# Check if unwrap script exists
if [ -f "unwrap.sh" ]; then
    echo "📦 Found unwrap script. Running deployment..."
    chmod +x unwrap.sh
    ./unwrap.sh
else
    echo "❌ Unwrap script not found."
    echo "   Make sure you're in the deployment package directory."
    exit 1
fi
