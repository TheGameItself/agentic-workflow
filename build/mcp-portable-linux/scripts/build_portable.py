#!/usr/bin/env python3
"""
Portable Environment Builder for MCP Server
Creates a self-contained Python environment with all dependencies.
"""

import os
import sys
import shutil
import subprocess
import json
import platform
from pathlib import Path
from typing import List, Dict, Any


class PortableEnvironmentBuilder:
    """Builds a portable Python environment for the MCP server."""
    
    def __init__(self, output_dir: str = "portable_mcp"):
        self.output_dir = Path(output_dir)
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system().lower()
        self.arch = platform.machine()
        
    def build_environment(self):
        """Build the complete portable environment."""
        print(f"Building portable MCP environment for {self.platform}-{self.arch}")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Build Python environment
        self._build_python_env()
        
        # Copy project files
        self._copy_project_files()
        
        # Create launcher scripts
        self._create_launchers()
        
        # Create configuration
        self._create_config()
        
        # Create documentation
        self._create_documentation()
        
        # Create installation script
        self._create_installer()
        
        print(f"Portable environment built in: {self.output_dir}")
        
    def _build_python_env(self):
        """Build the Python environment."""
        print("Building Python environment...")
        
        # Create virtual environment
        venv_path = self.output_dir / "python_env"
        subprocess.run([
            sys.executable, "-m", "venv", str(venv_path)
        ], check=True)
        
        # Determine pip path
        if self.platform == "windows":
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
        
        # Install dependencies
        print("Installing dependencies...")
        subprocess.run([
            str(pip_path), "install", "--upgrade", "pip"
        ], check=True)
        
        # Install project dependencies
        subprocess.run([
            str(pip_path), "install", "-r", "requirements.txt"
        ], check=True)
        
        # Install additional portable dependencies
        portable_deps = [
            "pyinstaller",
            "cx_Freeze",
            "setuptools",
            "wheel"
        ]
        
        for dep in portable_deps:
            subprocess.run([
                str(pip_path), "install", dep
            ], check=True)
        
    def _copy_project_files(self):
        """Copy project source files."""
        print("Copying project files...")
        
        # Copy source code
        src_dir = self.output_dir / "src"
        if Path("src").exists():
            shutil.copytree("src", src_dir, dirs_exist_ok=True)
        
        # Copy configuration
        config_dir = self.output_dir / "config"
        if Path("config").exists():
            shutil.copytree("config", config_dir, dirs_exist_ok=True)
        
        # Copy documentation
        docs_dir = self.output_dir / "docs"
        if Path("docs").exists():
            shutil.copytree("docs", docs_dir, dirs_exist_ok=True)
        
        # Copy scripts
        scripts_dir = self.output_dir / "scripts"
        if Path("scripts").exists():
            shutil.copytree("scripts", scripts_dir, dirs_exist_ok=True)
        
        # Copy plugins
        plugins_dir = self.output_dir / "plugins"
        if Path("plugins").exists():
            shutil.copytree("plugins", plugins_dir, dirs_exist_ok=True)
        
        # Copy essential files
        essential_files = [
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "README.md",
            "idea.txt"
        ]
        
        for file in essential_files:
            if Path(file).exists():
                shutil.copy2(file, self.output_dir)
        
    def _create_launchers(self) -> str:
        """Create launcher scripts for the portable environment."""
        launcher_sh = self.output_dir / "start_mcp.sh"
        launcher_content = '''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."
cd "$(dirname "$0")"
if [ -f ./mcp_agentic_workflow ]; then
    ./mcp_agentic_workflow "$@"
else
    python3 mcp_cli.py "$@"
fi
'''
        with open(launcher_sh, 'w') as f:
            f.write(launcher_content)
        os.chmod(launcher_sh, 0o755)
        return str(launcher_sh)
    
    def _create_config(self):
        """Create default configuration."""
        print("Creating configuration...")
        
        config = {
            "server": {
                "port": 3000,
                "host": "localhost",
                "debug": False
            },
            "database": {
                "path": "data/mcp.db",
                "backup_interval": 3600
            },
            "logging": {
                "level": "INFO",
                "file": "logs/mcp.log"
            },
            "security": {
                "api_key": "",
                "rate_limit": 100
            },
            "portable": {
                "data_dir": "data",
                "logs_dir": "logs",
                "temp_dir": "temp"
            }
        }
        
        config_path = self.output_dir / "config" / "mcp-config.json"
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
    def _create_documentation(self):
        """Create portable environment documentation."""
        print("Creating documentation...")
        
        readme_content = f"""# Portable MCP Server

This is a portable version of the MCP Server that can run on any system with Python {self.platform}-{self.arch}.

## Quick Start

### Windows
```cmd
start_mcp.bat
```

### Linux/macOS
```bash
./start_mcp.sh
```

### PowerShell
```powershell
./start_mcp.ps1
```

## Directory Structure

```
portable_mcp/
├── python_env/          # Python virtual environment
├── src/                 # MCP server source code
├── config/              # Configuration files
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── plugins/             # Plugin directory
├── data/                # Data directory (created on first run)
├── logs/                # Log files (created on first run)
├── start_mcp.bat        # Windows launcher
├── start_mcp.sh         # Unix launcher
├── start_mcp.ps1        # PowerShell launcher
└── mcp_server.py        # Main server launcher
```

## Configuration

Edit `config/mcp-config.json` to customize the server settings.

## Data Storage

All data is stored in the `data/` directory and is portable with the installation.

## Logs

Log files are stored in the `logs/` directory.

## Troubleshooting

1. Ensure Python {self.python_version} is available on the target system
2. Check logs in the `logs/` directory for error messages
3. Verify configuration in `config/mcp-config.json`

## Support

For issues and questions, refer to the documentation in the `docs/` directory.
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
    def _create_installer(self):
        """Create installation script."""
        print("Creating installer...")
        
        if self.platform == "windows":
            installer_content = '''@echo off
echo Installing Portable MCP Server...
echo.

REM Create data and logs directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp

echo Installation complete!
echo.
echo To start the MCP server, run: start_mcp.bat
pause
'''
            installer_path = self.output_dir / "install.bat"
            with open(installer_path, 'w') as f:
                f.write(installer_content)
        else:
            installer_content = '''#!/bin/bash
echo "Installing Portable MCP Server..."
echo

# Create data and logs directories
mkdir -p data logs temp

echo "Installation complete!"
echo
echo "To start the MCP server, run: ./start_mcp.sh"
'''
            installer_path = self.output_dir / "install.sh"
            with open(installer_path, 'w') as f:
                f.write(installer_content)
            
            os.chmod(installer_path, 0o755)
    
    def create_archive(self, archive_name: str = None):
        """Create a compressed archive of the portable environment."""
        if archive_name is None:
            archive_name = f"mcp_server_portable_{self.platform}_{self.arch}.tar.gz"
        
        print(f"Creating archive: {archive_name}")
        
        import tarfile
        
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(self.output_dir, arcname=self.output_dir.name)
        
        print(f"Archive created: {archive_name}")


def main():
    """Main function to build portable environment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build portable MCP environment")
    parser.add_argument("--output", "-o", default="portable_mcp", 
                       help="Output directory for portable environment")
    parser.add_argument("--archive", "-a", action="store_true",
                       help="Create compressed archive")
    parser.add_argument("--archive-name", help="Archive filename")
    
    args = parser.parse_args()
    
    builder = PortableEnvironmentBuilder(args.output)
    builder.build_environment()
    
    if args.archive:
        builder.create_archive(args.archive_name)


if __name__ == "__main__":
    main() 