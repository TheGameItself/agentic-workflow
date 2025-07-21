# MCP Agentic Workflow Accelerator - USB Edition

This is a portable version of the MCP Agentic Workflow Accelerator designed to run from USB drives.

## Quick Start

1. Insert this USB drive into any computer
2. Run the launcher:
   - **Windows**: Double-click `start_mcp.bat`
   - **macOS/Linux**: Run `./start_mcp.sh` in Terminal

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Self-contained**: No installation required
- **Portable**: Run from any USB drive
- **Universal**: Automatically detects the platform and launches the appropriate version

## Directory Structure

```
MCP_Agentic_Workflow/
├── start_mcp.sh          # Linux/macOS launcher
├── start_mcp.bat         # Windows launcher
├── MCP_Agentic_Workflow/ # Main application files
├── docs/                 # Documentation
└── platforms/            # Platform-specific versions
    ├── linux/           # Linux executables
    ├── windows/         # Windows executables
    └── macos/           # macOS executables
```

## Installation Instructions

### For USB Drive

1. Copy the contents of this template to your USB drive
2. Copy the appropriate platform-specific files to the `platforms/` directory
3. Test the launcher on your target system

### For Local Installation

1. Extract the appropriate platform package to the `platforms/` directory
2. Run the launcher script for your platform

## System Requirements

- **Windows**: Windows 10 or later
- **macOS**: macOS 10.14 or later
- **Linux**: Most modern distributions
- **Storage**: 500MB free space
- **Memory**: 4GB RAM minimum

## Troubleshooting

### Permission Issues (Linux/macOS)
```bash
chmod +x start_mcp.sh
chmod +x platforms/*/start_mcp.sh
```

### Antivirus Warnings
Some antivirus software may flag portable executables. Add the USB drive to your antivirus exclusions.

### Platform Not Detected
If the launcher doesn't detect your platform correctly, manually navigate to the appropriate `platforms/` subdirectory and run the executable directly.

## Support

For issues and documentation, see the main project repository.
