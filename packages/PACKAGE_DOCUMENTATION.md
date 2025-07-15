# MCP Agentic Workflow Accelerator - Package Documentation

This document describes all available packages for the MCP Agentic Workflow Accelerator.

## Build Information

- Build Date: 2025-07-13 18:34:08
- Platform: linux (x86_64)
- Version: 2.0.0

## Available Packages


### Appimage
✅ Package: /home/kalxi/Documents/agentic workflow/packages/mcp-agentic-workflow-2.0.0-x86_64.AppImage
- Type: Self-contained Linux application
- Size: 1.1 MB

### Flatpak
✅ Package: /home/kalxi/Documents/agentic workflow/packages/com.mcp.agentic-workflow-2.0.0.flatpak
- Type: Sandboxed Linux application
- Size: 719.0 B

### Portable
✅ Package: /home/kalxi/Documents/agentic workflow/packages/mcp-agentic-workflow-2.0.0-linux-portable.tar.gz
- Type: Extract and run without installation
- Size: 747.7 KB

### Usb_template
✅ Package: /home/kalxi/Documents/agentic workflow/packages/mcp-usb-template-2.0.0.tar.gz
- Type: USB drive installation template
- Size: 2.2 KB

### Universal
✅ Package: /home/kalxi/Documents/agentic workflow/packages/mcp-agentic-workflow-2.0.0-universal-portable.tar.gz
- Type: Cross-platform Python package
- Size: 741.5 KB


## Installation Instructions

### Linux

#### AppImage
1. Download the AppImage file
2. Make it executable: `chmod +x mcp-agentic-workflow-*.AppImage`
3. Run: `./mcp-agentic-workflow-*.AppImage`

#### Flatpak
1. Download the Flatpak file
2. Install: `flatpak install mcp-agentic-workflow-*.flatpak`

#### Portable Archive
1. Extract the tar.gz file
2. Run: `./start_mcp.sh`

### Windows

#### Installer
1. Download the .exe installer
2. Run the installer and follow the prompts

#### Portable Package
1. Extract the .zip file
2. Run: `start_mcp.bat` or `start_mcp.ps1`

### macOS

#### App Bundle
1. Download the .app file
2. Drag to Applications or run directly

#### Portable Package
1. Extract the tar.gz file
2. Run: `./start_mcp.sh`

### USB Installation

#### Using USB Template
1. Download the USB template
2. Extract to your USB drive
3. Copy platform-specific packages to the `platforms/` directory
4. Run the launcher from the USB drive

#### Direct USB Installation
1. Download the universal portable package
2. Extract to your USB drive
3. Run the launcher for your platform

## Package Types

- **AppImage**: Self-contained Linux application
- **Flatpak**: Sandboxed Linux application
- **Portable**: Extract and run without installation
- **Universal**: Works on any platform with Python
- **USB Template**: Template for USB drive installation

## System Requirements

- **Linux**: Most modern distributions
- **Windows**: Windows 10 or later
- **macOS**: macOS 10.14 or later
- **Storage**: 500MB free space
- **Memory**: 4GB RAM minimum

## Troubleshooting

### Permission Issues
```bash
chmod +x start_mcp.sh
chmod +x *.AppImage
```

### Antivirus Warnings
Add the application directory to your antivirus exclusions.

### Python Dependencies
The universal portable package requires Python 3.8+. Dependencies will be installed automatically on first run.

## Support

For issues and documentation, see the main project repository.
