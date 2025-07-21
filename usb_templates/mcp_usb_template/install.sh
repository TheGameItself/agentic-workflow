#!/bin/bash
# USB Installation Script for MCP Agentic Workflow Accelerator

echo "MCP Agentic Workflow Accelerator - USB Installation"
echo "=================================================="

# Check if running as root (for system-wide installation)
if [[ $EUID -eq 0 ]]; then
    echo "Installing system-wide..."
    INSTALL_DIR="/opt/mcp-agentic-workflow"
    BIN_DIR="/usr/local/bin"
else
    echo "Installing for current user..."
    INSTALL_DIR="$HOME/.local/share/mcp-agentic-workflow"
    BIN_DIR="$HOME/.local/bin"
fi

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy files
cp -r MCP_Agentic_Workflow/* "$INSTALL_DIR/"
cp -r platforms/* "$INSTALL_DIR/"

# Create launcher symlink
ln -sf "$INSTALL_DIR/start_mcp.sh" "$BIN_DIR/mcp-agentic-workflow"

# Make executable
chmod +x "$INSTALL_DIR/start_mcp.sh"
chmod +x "$BIN_DIR/mcp-agentic-workflow"

echo "Installation complete!"
echo "You can now run: mcp-agentic-workflow"
