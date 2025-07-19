#!/bin/bash
# GitHub Release Script for MCP v1.0.0

set -e

echo "Creating GitHub release v1.0.0..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed"
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Not in a git repository"
    exit 1
fi

# Create the release
gh release create v1.0.0 \
    --title "MCP Agentic Workflow Accelerator v1.0.0" \
    --notes-file "releases/RELEASE_NOTES_v1.0.0.md" \
    --latest \
    releases/mcp-portable-windows-v1.0.0.zip \
    releases/mcp-portable-macos-v1.0.0.tar.gz \
    releases/mcp-portable-linux-v1.0.0.tar.gz \
    releases/mcp-docker-v1.0.0.tar.gz \
    releases/mcp-usb-v1.0.0.zip \
    releases/mcp-development-v1.0.0.tar.gz \
    releases/SHA256SUMS_v1.0.0.txt

echo "Release v1.0.0 created successfully!"
echo "View at: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name")/releases/tag/v1.0.0"
