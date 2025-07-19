#!/usr/bin/env python3
"""
Interactive Setup Wizard for MCP Server
Provides guided setup and configuration for new users.
"""

import os
import sys
import json
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional

WELCOME = """
========================================
 MCP Server Installation Wizard
========================================
This wizard will guide you through setting up the MCP server for your system.
"""

class SetupWizard:
    """Interactive setup wizard for MCP server configuration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        
    def run(self):
        """Run the complete setup wizard."""
        print(WELCOME)
        
        # Check system requirements
        if not self.check_system_requirements():
            print("❌ System requirements not met. Please install required dependencies.")
            return False
            
        # Environment setup
        if not self.setup_environment():
            return False
            
        # Configuration setup
        if not self.setup_configuration():
            return False
            
        # Database setup
        if not self.setup_database():
            return False
            
        # Plugin setup
        if not self.setup_plugins():
            return False
            
        # Final verification
        if not self.verify_setup():
            return False
            
        print("\n✅ Setup completed successfully!")
        self.show_next_steps()
        return True
        
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        print("\n📋 Checking system requirements...")
        
        requirements = {
            "Python 3.8+": self.check_python_version(),
            "SQLite": self.check_sqlite(),
            "Required packages": self.check_packages(),
        }
        
        all_met = True
        for req, met in requirements.items():
            status = "✅" if met else "❌"
            print(f"  {status} {req}")
            if not met:
                all_met = False
                
        return all_met
        
    def check_python_version(self) -> bool:
        """Check Python version."""
        print("\n[Step 1] Checking Python version...")
        if sys.version_info < (3, 8):
            print("ERROR: Python 3.8 or higher is required.")
            sys.exit(1)
        print(f"Python version OK: {platform.python_version()}")
        return True
        
    def check_sqlite(self) -> bool:
        """Check SQLite availability."""
        try:
            import sqlite3
            return True
        except ImportError:
            return False
            
    def check_packages(self) -> bool:
        """Check if required packages are installed."""
        required = ["sqlalchemy", "numpy", "requests"]
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
                
        if missing:
            print(f"    Missing packages: {', '.join(missing)}")
            return False
        return True
        
    def setup_environment(self) -> bool:
        """Setup Python environment."""
        print("\n🐍 Setting up Python environment...")
        
        # Check if virtual environment exists
        venv_path = self.project_root / "venv"
        if venv_path.exists():
            response = input("Virtual environment already exists. Recreate? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(venv_path)
            else:
                print("  ✅ Using existing virtual environment")
                return True
                
        # Create virtual environment
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("  ✅ Virtual environment created")
            
            # Install requirements
            pip_cmd = str(venv_path / "bin" / "pip") if os.name != "nt" else str(venv_path / "Scripts" / "pip.exe")
            requirements_file = self.project_root / "requirements.txt"
            
            if requirements_file.exists():
                subprocess.run([pip_cmd, "install", "-r", str(requirements_file)], check=True)
                print("  ✅ Dependencies installed")
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to setup environment: {e}")
            return False
            
        return True
        
    def setup_configuration(self) -> bool:
        """Setup configuration files."""
        print("\n⚙️  Setting up configuration...")
        
        # Create config directory
        self.config_dir.mkdir(exist_ok=True)
        
        # Setup IDE configurations
        configs = {
            "vscode-settings.json": self.get_vscode_config(),
            "cursor-mcp.json": self.get_cursor_config(),
            "claude-mcp.json": self.get_claude_config(),
        }
        
        for filename, config in configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"  ✅ Created {filename}")
            else:
                print(f"  ✅ {filename} already exists")
                
        return True
        
    def get_vscode_config(self) -> Dict[str, Any]:
        """Get VS Code configuration."""
        return {
            "mcpServers": {
                "mcp-server": {
                    "command": "python",
                    "args": ["-m", "src.mcp.server"],
                    "env": {
                        "PYTHONPATH": str(self.project_root)
                    }
                }
            }
        }
        
    def get_cursor_config(self) -> Dict[str, Any]:
        """Get Cursor configuration."""
        return {
            "mcpServers": {
                "mcp-server": {
                    "command": "python",
                    "args": ["-m", "src.mcp.server"],
                    "env": {
                        "PYTHONPATH": str(self.project_root)
                    }
                }
            }
        }
        
    def get_claude_config(self) -> Dict[str, Any]:
        """Get Claude configuration."""
        return {
            "mcpServers": {
                "mcp-server": {
                    "command": "python",
                    "args": ["-m", "src.mcp.server"],
                    "env": {
                        "PYTHONPATH": str(self.project_root)
                    }
                }
            }
        }
        
    def setup_database(self) -> bool:
        """Setup database."""
        print("\n🗄️  Setting up database...")
        
        try:
            # Create data directory
            self.data_dir.mkdir(exist_ok=True)
            
            # Initialize database
            try:
                # Try to import and initialize database
                sys.path.insert(0, str(self.project_root))
                print("  ✅ Database initialization step (placeholder)")
            except (ImportError, Exception) as e:
                print(f"  ⚠️  Database initialization skipped: {e}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to setup database: {e}")
            return False
            
    def setup_plugins(self) -> bool:
        """Setup plugins."""
        print("\n🔌 Setting up plugins...")
        
        plugins_dir = self.project_root / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        
        # Check for example plugin
        example_plugin = plugins_dir / "example_plugin"
        if not example_plugin.exists():
            print("  ℹ️  No example plugins found")
        else:
            print("  ✅ Example plugins available")
            
        return True
        
    def verify_setup(self) -> bool:
        """Verify the setup."""
        print("\n🔍 Verifying setup...")
        
        checks = [
            ("Virtual environment", self.project_root / "venv"),
            ("Configuration directory", self.config_dir),
            ("Data directory", self.data_dir),
            ("Database file", self.data_dir / "mcp_server.db"),
        ]
        
        all_good = True
        for name, path in checks:
            if path.exists():
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}")
                all_good = False
                
        return all_good
        
    def show_next_steps(self):
        """Show next steps for the user."""
        print("\n🎯 Next Steps:")
        print("1. Activate virtual environment:")
        if os.name != "nt":
            print(f"   source {self.project_root}/venv/bin/activate")
        else:
            print(f"   {self.project_root}\\venv\\Scripts\\activate")
            
        print("2. Test the server:")
        print("   python -m src.mcp.server")
        
        print("3. Run CLI commands:")
        print("   python mcp_cli.py --help")
        
        print("4. Check documentation:")
        print("   README.md, USER_GUIDE.md, API_DOCUMENTATION.md")
        
        print("\n📚 For more information, see the documentation files.")


def main():
    """Main entry point."""
    wizard = SetupWizard()
    
    try:
        success = wizard.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 