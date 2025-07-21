#!/usr/bin/env python3
"""
MCP Server Installation Script
Provides automated installation and setup for the MCP server.
"""

import os
import sys
import subprocess
import json
import platform
import shutil
from pathlib import Path
from typing import Dict, Any, List


class MCPInstaller:
    """Handles installation and setup of the MCP server."""
    
    def __init__(self, install_dir: str = None):
        self.install_dir = Path(install_dir) if install_dir else Path.home() / ".mcp_server"
        self.platform = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
    def install(self, options: Dict[str, Any] = None):
        """Perform complete installation."""
        options = options or {}
        
        print("MCP Server Installation")
        print("=" * 50)
        
        # Check system requirements
        self._check_requirements()
        
        # Create installation directory
        self._create_install_dir()
        
        # Install Python dependencies
        self._install_dependencies()
        
        # Setup configuration
        self._setup_configuration(options)
        
        # Setup database
        self._setup_database()
        
        # Create launcher scripts
        self._create_launchers()
        
        # Setup system integration
        if options.get("system_integration", False):
            self._setup_system_integration()
        
        # Setup IDE integration
        if options.get("ide_integration", False):
            self._setup_ide_integration()
        
        # Verify installation
        self._verify_installation()
        
        print("\nInstallation completed successfully!")
        print(f"MCP Server installed in: {self.install_dir}")
        
    def _check_requirements(self):
        """Check system requirements."""
        print("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        print(f"✓ Python {self.python_version} detected")
        
        # Check platform
        supported_platforms = ["linux", "darwin", "windows"]
        if self.platform not in supported_platforms:
            print(f"⚠ Warning: Platform {self.platform} may not be fully supported")
        else:
            print(f"✓ Platform {self.platform} supported")
        
        # Check available disk space
        free_space = shutil.disk_usage(self.install_dir.parent).free
        required_space = 500 * 1024 * 1024  # 500MB
        if free_space < required_space:
            raise RuntimeError(f"Insufficient disk space. Need at least 500MB, have {free_space // (1024*1024)}MB")
        
        print("✓ Sufficient disk space available")
        
    def _create_install_dir(self):
        """Create installation directory structure."""
        print("Creating installation directory...")
        
        # Create main directory
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "bin",
            "config",
            "data",
            "logs",
            "plugins",
            "docs",
            "scripts",
            "temp"
        ]
        
        for subdir in subdirs:
            (self.install_dir / subdir).mkdir(exist_ok=True)
        
        print(f"✓ Installation directory created: {self.install_dir}")
        
    def _install_dependencies(self):
        """Install Python dependencies."""
        print("Installing Python dependencies...")
        
        # Create virtual environment
        venv_path = self.install_dir / "venv"
        subprocess.run([
            sys.executable, "-m", "venv", str(venv_path)
        ], check=True)
        
        # Determine pip path
        if self.platform == "windows":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        # Upgrade pip
        subprocess.run([
            str(pip_path), "install", "--upgrade", "pip"
        ], check=True)
        
        # Install requirements
        if Path("requirements.txt").exists():
            subprocess.run([
                str(pip_path), "install", "-r", "requirements.txt"
            ], check=True)
        
        # Install additional dependencies
        additional_deps = [
            "setuptools",
            "wheel",
            "black",
            "flake8",
            "mypy"
        ]
        
        for dep in additional_deps:
            subprocess.run([
                str(pip_path), "install", dep
            ], check=True)
        
        print("✓ Dependencies installed successfully")
        
    def _setup_configuration(self, options: Dict[str, Any]):
        """Setup configuration files."""
        print("Setting up configuration...")
        
        # Create default configuration
        config = {
            "server": {
                "port": options.get("port", 3000),
                "host": options.get("host", "localhost"),
                "debug": options.get("debug", False)
            },
            "database": {
                "path": str(self.install_dir / "data" / "mcp.db"),
                "backup_interval": 3600
            },
            "logging": {
                "level": "INFO",
                "file": str(self.install_dir / "logs" / "mcp.log"),
                "max_size": "10MB",
                "backup_count": 5
            },
            "security": {
                "api_key": options.get("api_key", ""),
                "rate_limit": options.get("rate_limit", 100),
                "enable_auth": options.get("enable_auth", False)
            },
            "performance": {
                "max_workers": options.get("max_workers", 4),
                "memory_limit": options.get("memory_limit", "1GB"),
                "enable_monitoring": options.get("enable_monitoring", True)
            },
            "plugins": {
                "auto_load": options.get("auto_load_plugins", True),
                "plugin_dir": str(self.install_dir / "plugins")
            }
        }
        
        config_path = self.install_dir / "config" / "mcp-config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Copy source code
        if Path("src").exists():
            src_dest = self.install_dir / "src"
            shutil.copytree("src", src_dest, dirs_exist_ok=True)
        
        # Copy documentation
        if Path("docs").exists():
            docs_dest = self.install_dir / "docs"
            shutil.copytree("docs", docs_dest, dirs_exist_ok=True)
        
        print("✓ Configuration setup complete")
        
    def _setup_database(self):
        """Setup database and initialize tables."""
        print("Setting up database...")
        
        # Initialize database
        db_path = self.install_dir / "data" / "mcp.db"
        
        # Create database tables
        self._create_database_tables(db_path)
        
        print("✓ Database setup complete")
        
    def _create_database_tables(self, db_path: Path):
        """Create database tables."""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables (simplified version)
        tables = [
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                memory_type TEXT,
                priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            cursor.execute(table_sql)
        
        conn.commit()
        conn.close()
        
    def _create_launchers(self):
        """Create launcher scripts."""
        print("Creating launcher scripts...")
        
        # Create main launcher
        launcher_content = self._get_launcher_content()
        
        if self.platform == "windows":
            # Windows batch file
            launcher_path = self.install_dir / "bin" / "mcp.bat"
            with open(launcher_path, 'w') as f:
                f.write(launcher_content['windows'])
            
            # PowerShell script
            ps_path = self.install_dir / "bin" / "mcp.ps1"
            with open(ps_path, 'w') as f:
                f.write(launcher_content['powershell'])
                
        else:
            # Unix shell script
            launcher_path = self.install_dir / "bin" / "mcp"
            with open(launcher_path, 'w') as f:
                f.write(launcher_content['unix'])
            
            # Make executable
            os.chmod(launcher_path, 0o755)
        
        print("✓ Launcher scripts created")
        
    def _get_launcher_content(self) -> Dict[str, str]:
        """Get launcher script content."""
        venv_python = str(self.install_dir / "venv" / "bin" / "python")
        if self.platform == "windows":
            venv_python = str(self.install_dir / "venv" / "Scripts" / "python.exe")
        
        return {
            'windows': f'''@echo off
echo Starting MCP Server...
cd /d "{self.install_dir}"
"{venv_python}" -m src.mcp.cli server
''',
            'powershell': f'''# PowerShell launcher for MCP Server
Write-Host "Starting MCP Server..." -ForegroundColor Green
Set-Location "{self.install_dir}"
& "{venv_python}" "-m" "src.mcp.cli" "server"
''',
            'unix': f'''#!/bin/bash
echo "Starting MCP Server..."
cd "{self.install_dir}"
"{venv_python}" -m src.mcp.cli server
'''
        }
        
    def _setup_system_integration(self):
        """Setup system integration (service, PATH, etc.)."""
        print("Setting up system integration...")
        
        if self.platform == "linux":
            self._setup_linux_service()
        elif self.platform == "darwin":
            self._setup_macos_service()
        elif self.platform == "windows":
            self._setup_windows_service()
        
        # Add to PATH
        self._add_to_path()
        
        print("✓ System integration setup complete")
        
    def _setup_linux_service(self):
        """Setup Linux systemd service."""
        service_content = f"""[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
WorkingDirectory={self.install_dir}
ExecStart={self.install_dir}/venv/bin/python -m src.mcp.cli server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = Path("/etc/systemd/system/mcp-server.service")
        if os.access("/etc/systemd/system", os.W_OK):
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            # Enable and start service
            subprocess.run(["systemctl", "enable", "mcp-server"], check=True)
            subprocess.run(["systemctl", "start", "mcp-server"], check=True)
        
    def _setup_macos_service(self):
        """Setup macOS launchd service."""
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mcp.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{self.install_dir}/venv/bin/python</string>
        <string>-m</string>
        <string>src.mcp.cli</string>
        <string>server</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{self.install_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"""
        
        plist_path = Path.home() / "Library/LaunchAgents/com.mcp.server.plist"
        plist_path.parent.mkdir(exist_ok=True)
        
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        
        # Load service
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        
    def _setup_windows_service(self):
        """Setup Windows service."""
        # This would require additional dependencies like pywin32
        print("Windows service setup requires pywin32. Please install manually if needed.")
        
    def _add_to_path(self):
        """Add MCP server to system PATH."""
        bin_dir = str(self.install_dir / "bin")
        
        if self.platform == "windows":
            # Windows PATH modification
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE)
                path = winreg.QueryValueEx(key, "Path")[0]
                if bin_dir not in path:
                    new_path = path + ";" + bin_dir
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                winreg.CloseKey(key)
            except Exception as e:
                print(f"Warning: Could not update PATH: {e}")
        else:
            # Unix PATH modification
            shell_rc = Path.home() / ".bashrc"
            if not shell_rc.exists():
                shell_rc = Path.home() / ".zshrc"
            
            if shell_rc.exists():
                with open(shell_rc, 'a') as f:
                    f.write(f'\n# MCP Server\nexport PATH="$PATH:{bin_dir}"\n')
        
    def _setup_ide_integration(self):
        """Setup IDE integration."""
        print("Setting up IDE integration...")
        
        # VS Code settings
        vscode_settings = {
            "mcp.servers": {
                "mcp-server": {
                    "command": str(self.install_dir / "venv" / "bin" / "python"),
                    "args": ["-m", "src.mcp.cli", "server"],
                    "env": {
                        "MCP_CONFIG_PATH": str(self.install_dir / "config" / "mcp-config.json")
                    }
                }
            }
        }
        
        vscode_dir = Path.home() / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings_path = vscode_dir / "settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                existing_settings = json.load(f)
        else:
            existing_settings = {}
        
        existing_settings.update(vscode_settings)
        
        with open(settings_path, 'w') as f:
            json.dump(existing_settings, f, indent=2)
        
        print("✓ IDE integration setup complete")
        
    def _verify_installation(self):
        """Verify the installation."""
        print("Verifying installation...")
        
        # Check if virtual environment exists
        venv_path = self.install_dir / "venv"
        if not venv_path.exists():
            raise RuntimeError("Virtual environment not found")
        
        # Check if configuration exists
        config_path = self.install_dir / "config" / "mcp-config.json"
        if not config_path.exists():
            raise RuntimeError("Configuration file not found")
        
        # Check if database exists
        db_path = self.install_dir / "data" / "mcp.db"
        if not db_path.exists():
            raise RuntimeError("Database not found")
        
        # Test import
        try:
            sys.path.insert(0, str(self.install_dir / "src"))
            import mcp
            print("✓ MCP module imports successfully")
        except ImportError as e:
            raise RuntimeError(f"Failed to import MCP module: {e}")
        
        print("✓ Installation verification complete")
        
    def uninstall(self):
        """Uninstall the MCP server."""
        print("Uninstalling MCP Server...")
        
        # Stop services
        if self.platform == "linux":
            try:
                subprocess.run(["systemctl", "stop", "mcp-server"], check=False)
                subprocess.run(["systemctl", "disable", "mcp-server"], check=False)
            except:
                pass
        elif self.platform == "darwin":
            try:
                plist_path = Path.home() / "Library/LaunchAgents/com.mcp.server.plist"
                subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
                plist_path.unlink(missing_ok=True)
            except:
                pass
        
        # Remove installation directory
        if self.install_dir.exists():
            shutil.rmtree(self.install_dir)
        
        print("✓ MCP Server uninstalled")


def main():
    """Main installation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install MCP Server")
    parser.add_argument("--install-dir", help="Installation directory")
    parser.add_argument("--port", type=int, default=3000, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--system-integration", action="store_true", help="Setup system integration")
    parser.add_argument("--ide-integration", action="store_true", help="Setup IDE integration")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall MCP Server")
    
    args = parser.parse_args()
    
    installer = MCPInstaller(args.install_dir)
    
    if args.uninstall:
        installer.uninstall()
    else:
        options = {
            "port": args.port,
            "host": args.host,
            "api_key": args.api_key,
            "system_integration": args.system_integration,
            "ide_integration": args.ide_integration
        }
        installer.install(options)


if __name__ == "__main__":
    main() 