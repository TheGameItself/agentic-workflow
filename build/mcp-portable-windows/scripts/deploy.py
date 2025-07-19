#!/usr/bin/env python3
"""
Deployment Automation Script
Creates deployment packages and sets up monitoring.
"""

import os
import sys
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class DeploymentManager:
    """Manages deployment packages and monitoring setup."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.deployment_dir = self.project_root / "deployment_packages"
        self.deployment_dir.mkdir(exist_ok=True)
        
    def create_deployment_package(self, package_type: str = "full", version: Optional[str] = None):
        """Create a deployment package."""
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        print(f"📦 Creating {package_type} deployment package v{version}")
        
        if package_type == "full":
            return self._create_full_package(version)
        elif package_type == "minimal":
            return self._create_minimal_package(version)
        elif package_type == "portable":
            return self._create_portable_package(version)
        else:
            print(f"Unknown package type: {package_type}")
            return None
            
    def _create_full_package(self, version: str) -> Optional[Path]:
        """Create full deployment package."""
        package_name = f"mcp-server-full-{version}"
        package_path = self.deployment_dir / package_name
        
        try:
            # Create package directory
            package_path.mkdir(exist_ok=True)
            
            # Copy source code
            src_dir = package_path / "src"
            shutil.copytree(self.project_root / "src", src_dir, dirs_exist_ok=True)
            
            # Copy scripts
            scripts_dir = package_path / "scripts"
            shutil.copytree(self.project_root / "scripts", scripts_dir, dirs_exist_ok=True)
            
            # Copy configuration
            config_dir = package_path / "config"
            if (self.project_root / "config").exists():
                shutil.copytree(self.project_root / "config", config_dir, dirs_exist_ok=True)
                
            # Copy documentation
            docs_dir = package_path / "docs"
            if (self.project_root / "docs").exists():
                shutil.copytree(self.project_root / "docs", docs_dir, dirs_exist_ok=True)
                
            # Copy plugins
            plugins_dir = package_path / "plugins"
            if (self.project_root / "plugins").exists():
                shutil.copytree(self.project_root / "plugins", plugins_dir, dirs_exist_ok=True)
                
            # Copy essential files
            essential_files = [
                "README.md", "requirements.txt", "pyproject.toml",
                "mcp_cli.py", "setup.py"
            ]
            
            for file_name in essential_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    shutil.copy2(file_path, package_path)
                    
            # Create deployment script
            self._create_deployment_script(package_path)
            
            # Create archive
            archive_path = self._create_archive(package_path, f"mcp-server-full-{version}")
            
            # Cleanup
            shutil.rmtree(package_path)
            
            print(f"✅ Full package created: {archive_path}")
            return archive_path
            
        except Exception as e:
            print(f"❌ Failed to create full package: {e}")
            if package_path.exists():
                shutil.rmtree(package_path)
            return None
            
    def _create_minimal_package(self, version: str) -> Optional[Path]:
        """Create minimal deployment package."""
        package_name = f"mcp-server-minimal-{version}"
        package_path = self.deployment_dir / package_name
        
        try:
            # Create package directory
            package_path.mkdir(exist_ok=True)
            
            # Copy only essential source files
            src_dir = package_path / "src"
            src_dir.mkdir()
            
            # Copy core MCP modules
            core_modules = [
                "server.py", "cli.py", "database_manager.py", "memory.py",
                "task_manager.py", "workflow.py", "context_manager.py"
            ]
            
            for module in core_modules:
                module_path = self.project_root / "src" / "mcp" / module
                if module_path.exists():
                    shutil.copy2(module_path, src_dir / "mcp")
                    
            # Copy essential scripts
            essential_scripts = [
                "setup_wizard.py", "help_system.py", "security_audit.py"
            ]
            
            scripts_dir = package_path / "scripts"
            scripts_dir.mkdir()
            
            for script in essential_scripts:
                script_path = self.project_root / "scripts" / script
                if script_path.exists():
                    shutil.copy2(script_path, scripts_dir)
                    
            # Copy minimal configuration
            config_dir = package_path / "config"
            config_dir.mkdir()
            
            # Create minimal config
            minimal_config = {
                "server": {
                    "host": "localhost",
                    "port": 8000,
                    "debug": False
                },
                "database": {
                    "type": "sqlite",
                    "path": "data/mcp_server.db"
                }
            }
            
            with open(config_dir / "config.json", 'w') as f:
                json.dump(minimal_config, f, indent=2)
                
            # Copy essential files
            essential_files = ["README.md", "requirements.txt"]
            
            for file_name in essential_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    shutil.copy2(file_path, package_path)
                    
            # Create deployment script
            self._create_deployment_script(package_path, minimal=True)
            
            # Create archive
            archive_path = self._create_archive(package_path, f"mcp-server-minimal-{version}")
            
            # Cleanup
            shutil.rmtree(package_path)
            
            print(f"✅ Minimal package created: {archive_path}")
            return archive_path
            
        except Exception as e:
            print(f"❌ Failed to create minimal package: {e}")
            if package_path.exists():
                shutil.rmtree(package_path)
            return None
            
    def _create_portable_package(self, version: str) -> Optional[Path]:
        """Create portable deployment package with embedded Python."""
        package_name = f"mcp-server-portable-{version}"
        package_path = self.deployment_dir / package_name
        
        try:
            # Create package directory
            package_path.mkdir(exist_ok=True)
            
            # Copy source code
            src_dir = package_path / "src"
            shutil.copytree(self.project_root / "src", src_dir, dirs_exist_ok=True)
            
            # Copy scripts
            scripts_dir = package_path / "scripts"
            shutil.copytree(self.project_root / "scripts", scripts_dir, dirs_exist_ok=True)
            
            # Create portable Python environment
            self._create_portable_python(package_path)
            
            # Create launcher scripts
            self._create_launcher_scripts(package_path)
            
            # Copy configuration
            config_dir = package_path / "config"
            if (self.project_root / "config").exists():
                shutil.copytree(self.project_root / "config", config_dir, dirs_exist_ok=True)
                
            # Copy essential files
            essential_files = ["README.md", "requirements.txt"]
            
            for file_name in essential_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    shutil.copy2(file_path, package_path)
                    
            # Create archive
            archive_path = self._create_archive(package_path, f"mcp-server-portable-{version}")
            
            # Cleanup
            shutil.rmtree(package_path)
            
            print(f"✅ Portable package created: {archive_path}")
            return archive_path
            
        except Exception as e:
            print(f"❌ Failed to create portable package: {e}")
            if package_path.exists():
                shutil.rmtree(package_path)
            return None
            
    def _create_portable_python(self, package_path: Path):
        """Create portable Python environment."""
        python_dir = package_path / "python"
        python_dir.mkdir()
        
        # This would require downloading and extracting a portable Python distribution
        # For now, we'll create a script to download it
        download_script = python_dir / "download_python.py"
        
        download_script_content = '''#!/usr/bin/env python3
"""
Download portable Python for the deployment package.
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_portable_python():
    """Download portable Python distribution."""
    # This is a placeholder - in a real implementation, you would:
    # 1. Download the appropriate Python distribution for the target platform
    # 2. Extract it to the python directory
    # 3. Install required packages
    
    print("Portable Python download script")
    print("This would download and setup a portable Python environment")
    
if __name__ == "__main__":
    download_portable_python()
'''
        
        with open(download_script, 'w') as f:
            f.write(download_script_content)
            
    def _create_launcher_scripts(self, package_path: Path):
        """Create launcher scripts for different platforms."""
        # Windows launcher
        windows_launcher = package_path / "run.bat"
        windows_content = '''@echo off
cd /d "%~dp0"
python\\python.exe -m src.mcp.server %*
'''
        
        with open(windows_launcher, 'w') as f:
            f.write(windows_content)
            
        # Unix launcher
        unix_launcher = package_path / "run.sh"
        unix_content = '''#!/bin/bash
cd "$(dirname "$0")"
./python/bin/python -m src.mcp.server "$@"
'''
        
        with open(unix_launcher, 'w') as f:
            f.write(unix_content)
            
        # Make Unix launcher executable
        os.chmod(unix_launcher, 0o755)
        
    def _create_deployment_script(self, package_path: Path, minimal: bool = False):
        """Create deployment script."""
        script_path = package_path / "deploy.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Deployment script for MCP Server
"""
import os
import sys
import subprocess
from pathlib import Path

def deploy():
    """Deploy the MCP server."""
    print("🚀 Deploying MCP Server...")
    
    # Setup virtual environment
    print("📦 Setting up virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Install dependencies
    print("📥 Installing dependencies...")
    pip_cmd = "venv/bin/pip" if os.name != "nt" else "venv\\\\Scripts\\\\pip.exe"
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
    
    # Initialize database
    print("🗄️ Initializing database...")
    try:
        subprocess.run([pip_cmd, "install", "sqlalchemy"], check=True)
        # Database initialization would happen here
    except Exception as e:
        print(f"⚠️ Database initialization skipped: {{e}}")
    
    print("✅ Deployment completed!")
    print("\\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name != "nt":
        print("   source venv/bin/activate")
    else:
        print("   venv\\\\Scripts\\\\activate")
    print("2. Run the server:")
    print("   python -m src.mcp.server")

if __name__ == "__main__":
    deploy()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
    def _create_archive(self, package_path: Path, archive_name: str) -> Path:
        """Create archive from package directory."""
        archive_path = self.deployment_dir / f"{archive_name}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(package_path, arcname=archive_name)
            
        return archive_path
        
    def setup_monitoring(self, config_path: Optional[Path] = None):
        """Setup monitoring and logging."""
        print("📊 Setting up monitoring...")
        
        if not config_path:
            config_path = self.project_root / "config" / "monitoring.json"
            
        config_path.parent.mkdir(exist_ok=True)
        
        monitoring_config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/mcp_server.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            "metrics": {
                "enabled": True,
                "port": 9090,
                "endpoint": "/metrics"
            },
            "health_check": {
                "enabled": True,
                "port": 8080,
                "endpoint": "/health"
            },
            "alerts": {
                "enabled": True,
                "email": {
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "from_email": "mcp-server@example.com",
                    "to_emails": ["admin@example.com"]
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
            
        print(f"✅ Monitoring configuration created: {config_path}")
        
        # Create logs directory
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create monitoring script
        monitoring_script = self.project_root / "scripts" / "monitor.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Monitoring script for MCP Server
"""
import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

def setup_monitoring():
    """Setup monitoring and logging."""
    # Load configuration
    config_path = Path("config/monitoring.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "logging": {
                "level": "INFO",
                "file": "logs/mcp_server.log"
            }
        }
    
    # Setup logging
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/mcp_server.log")
    
    # Ensure log directory exists
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("mcp_server")
    logger.info("Monitoring setup completed")
    
    return logger

if __name__ == "__main__":
    logger = setup_monitoring()
    logger.info("MCP Server monitoring started")
    
    # Keep the script running
    try:
        while True:
            time.sleep(60)
            logger.info("Monitoring heartbeat")
    except KeyboardInterrupt:
        logger.info("Monitoring stopped")
'''
        
        with open(monitoring_script, 'w') as f:
            f.write(script_content)
            
        print("✅ Monitoring script created")
        
    def list_packages(self):
        """List available deployment packages."""
        print("📦 Available Deployment Packages")
        print("=" * 40)
        
        packages = list(self.deployment_dir.glob("*.tar.gz"))
        
        if not packages:
            print("No deployment packages found.")
            return
            
        for package in sorted(packages, key=lambda x: x.stat().st_mtime, reverse=True):
            size = package.stat().st_size
            size_mb = size / (1024 * 1024)
            mtime = datetime.fromtimestamp(package.stat().st_mtime)
            
            print(f"  {package.name}")
            print(f"    Size: {size_mb:.1f} MB")
            print(f"    Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()


def main():
    """CLI interface for deployment manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deployment Manager")
    parser.add_argument("command", choices=["create", "list", "monitor"], help="Command to execute")
    parser.add_argument("--type", choices=["full", "minimal", "portable"], default="full", help="Package type")
    parser.add_argument("--version", help="Package version")
    parser.add_argument("--config", help="Monitoring config path")
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.command == "create":
        package_path = manager.create_deployment_package(args.type, args.version)
        if package_path:
            print(f"Package created successfully: {package_path}")
        else:
            print("Failed to create package")
            sys.exit(1)
    elif args.command == "list":
        manager.list_packages()
    elif args.command == "monitor":
        config_path = Path(args.config) if args.config else None
        manager.setup_monitoring(config_path)


if __name__ == "__main__":
    main() 