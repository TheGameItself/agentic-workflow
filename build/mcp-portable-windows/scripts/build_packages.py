#!/usr/bin/env python3
"""
MCP Release Package Builder
Creates various deployment packages for GitHub releases
"""

import os
import sys
import shutil
import subprocess
import tarfile
import zipfile
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPPackageBuilder:
    def __init__(self, version="1.0.0", output_dir="releases"):
        self.version = version
        self.output_dir = Path(output_dir)
        self.project_root = Path(__file__).parent.parent
        self.build_dir = Path("build")
        
        self.output_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)
        
        logger.info(f"Building MCP packages v{version}")
    
    def copy_source_files(self, target_dir):
        """Copy essential source files"""
        essential_files = [
            "src/", "config/", "docs/", "scripts/",
            "requirements.txt", "pyproject.toml", "setup.py",
            "README.md", "LICENSE", "VERSION",
            "start_mcp.sh", "start_mcp.bat", "mcp_cli.py"
        ]
        
        for item in essential_files:
            src = self.project_root / item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, target_dir / item, 
                                  ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                else:
                    shutil.copy2(src, target_dir / item)
    
    def create_portable_package(self, platform):
        """Create portable package for platform"""
        logger.info(f"Creating portable package for {platform}")
        
        package_dir = self.build_dir / f"mcp-portable-{platform}"
        package_dir.mkdir(exist_ok=True)
        
        self.copy_source_files(package_dir)
        
        # Create installer script
        if platform == "windows":
            installer = '''@echo off
echo Installing MCP...
python -m venv .venv
call .venv\\Scripts\\activate.bat
pip install -r requirements.txt
pip install -e .
echo Installation complete!
pause
'''
            with open(package_dir / "install.bat", "w") as f:
                f.write(installer)
        else:
            installer = '''#!/bin/bash
echo "Installing MCP..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
echo "Installation complete!"
'''
            with open(package_dir / "install.sh", "w") as f:
                f.write(installer)
            os.chmod(package_dir / "install.sh", 0o755)
        
        # Create archive
        if platform == "windows":
            archive_path = self.output_dir / f"mcp-portable-{platform}-v{self.version}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
        else:
            archive_path = self.output_dir / f"mcp-portable-{platform}-v{self.version}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(package_dir, arcname=package_dir.name)
        
        logger.info(f"Created: {archive_path}")
        return archive_path
    
    def create_docker_package(self):
        """Create Docker package"""
        logger.info("Creating Docker package")
        
        docker_dir = self.build_dir / "mcp-docker"
        docker_dir.mkdir(exist_ok=True)
        
        # Create Dockerfile
        dockerfile = '''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
EXPOSE 3000
CMD ["python", "-m", "src.mcp.cli", "server", "--host", "0.0.0.0"]
'''
        with open(docker_dir / "Dockerfile", "w") as f:
            f.write(dockerfile)
        
        # Create docker-compose
        compose = '''version: '3.8'
services:
  mcp-server:
    build: .
    ports:
      - "3000:3000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - MCP_LOG_LEVEL=INFO
    restart: unless-stopped
'''
        with open(docker_dir / "docker-compose.yml", "w") as f:
            f.write(compose)
        
        self.copy_source_files(docker_dir)
        
        archive_path = self.output_dir / f"mcp-docker-v{self.version}.tar.gz"
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(docker_dir, arcname=docker_dir.name)
        
        logger.info(f"Created: {archive_path}")
        return archive_path
    
    def create_usb_package(self):
        """Create USB portable package"""
        logger.info("Creating USB package")
        
        usb_dir = self.build_dir / "mcp-usb"
        usb_dir.mkdir(exist_ok=True)
        
        self.copy_source_files(usb_dir)
        
        # USB launcher
        launcher = '''#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
else
    source .venv/bin/activate
fi
python -m src.mcp.cli server
'''
        with open(usb_dir / "start_usb.sh", "w") as f:
            f.write(launcher)
        os.chmod(usb_dir / "start_usb.sh", 0o755)
        
        archive_path = self.output_dir / f"mcp-usb-v{self.version}.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in usb_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(usb_dir)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created: {archive_path}")
        return archive_path
    
    def build_all(self):
        """Build all packages"""
        packages = []
        
        # Portable packages
        for platform in ["windows", "macos", "linux"]:
            try:
                pkg = self.create_portable_package(platform)
                packages.append(pkg)
            except Exception as e:
                logger.error(f"Failed to create {platform} package: {e}")
        
        # Docker package
        try:
            pkg = self.create_docker_package()
            packages.append(pkg)
        except Exception as e:
            logger.error(f"Failed to create Docker package: {e}")
        
        # USB package
        try:
            pkg = self.create_usb_package()
            packages.append(pkg)
        except Exception as e:
            logger.error(f"Failed to create USB package: {e}")
        
        logger.info(f"Created {len(packages)} packages in {self.output_dir}")
        return packages

def main():
    parser = argparse.ArgumentParser(description="Build MCP packages")
    parser.add_argument("--version", default="1.0.0", help="Version")
    parser.add_argument("--output", default="releases", help="Output directory")
    
    args = parser.parse_args()
    
    builder = MCPPackageBuilder(args.version, args.output)
    packages = builder.build_all()
    
    print(f"Built {len(packages)} packages:")
    for pkg in packages:
        print(f"  - {pkg}")

if __name__ == "__main__":
    main()