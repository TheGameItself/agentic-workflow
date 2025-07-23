#!/usr/bin/env python3
"""
Simple Executable Builder for MCP Agentic Workflow Accelerator
Creates basic executables for the current platform.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional

class SimpleExecutableBuilder:
    """Simple executable builder for the MCP server."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the simple executable builder."""
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self.platform = platform.system().lower()
        self.arch = platform.machine()
        
        # Output directories
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.executables_dir = self.project_root / "executables"
        
        # Create directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.executables_dir.mkdir(exist_ok=True)
    
    def build_executable(self, clean: bool = True) -> str:
        """Build executable for the current platform."""
        print(f"🚀 Building MCP Agentic Workflow Accelerator Executable")
        print(f"📦 Platform: {self.platform} ({self.arch})")
        print("=" * 60)
        
        if clean:
            self._clean_build_directories()
        
        # Install PyInstaller if not available
        try:
            import PyInstaller
        except ImportError:
            print("📦 Installing PyInstaller...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        
        # Create simple spec file
        spec_file = self._create_simple_spec()
        
        # Build executable
        print("🔨 Building executable...")
        result = self._build_with_pyinstaller(spec_file)
        
        # Create launcher script
        launcher_path = self._create_launcher()
        
        # Create portable package
        self._create_portable_package()
        
        print(f"\n✅ Build completed successfully!")
        print(f"📁 Executable: {result}")
        print(f"📁 Launcher: {launcher_path}")
        print(f"📁 Portable package: {self.executables_dir}")
        
        return result
    
    def _create_simple_spec(self) -> Path:
        """Create a simple PyInstaller spec file."""
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['mcp_cli.py'],
    pathex=[str(self.project_root)],
    binaries=[],
    datas=[
        ('src/mcp', 'mcp'),
        ('config', 'config'),
        ('data', 'data'),
        ('plugins', 'plugins'),
        ('idea.txt', '.'),
        ('README.md', '.'),
        ('requirements.txt', '.'),
        ('pyproject.toml', '.'),
    ],
    hiddenimports=[
        'click',
        'rich',
        'sqlalchemy',
        'pytest',
        'typing_extensions',
        'psutil',
        'numpy',
        'scikit_learn',
        'sentence_transformers',
        'faiss',
        'python_dotenv',
        'pyyaml',
        'jinja2',
        'watchdog',
        'colorama',
        'tqdm',
        'packaging',
        'mcp',
        'mcp.memory',
        'mcp.cli',
        'mcp.workflow',
        'mcp.task_manager',
        'mcp.project_manager',
        'mcp.unified_memory',
        'mcp.context_manager',
        'mcp.reminder_engine',
        'mcp.database_manager',
        'mcp.performance_monitor',
        'mcp.regex_search',
        'mcp.rag_system',
        'mcp.server',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='mcp_agentic_workflow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
        
        spec_file = self.build_dir / "mcp_agentic_workflow.spec"
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        return spec_file
    
    def _build_with_pyinstaller(self, spec_file: Path) -> str:
        """Build executable using PyInstaller."""
        print(f"  Running PyInstaller...")
        
        # Create dist directory for this platform
        platform_dist = self.dist_dir / self.platform
        platform_dist.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            str(spec_file),
            "--distpath", str(platform_dist),
            "--workpath", str(self.build_dir / "work"),
            "--clean"
        ]
        
        print(f"  Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Find the built executable
        if self.platform == "windows":
            executable = platform_dist / "mcp_agentic_workflow.exe"
        else:
            executable = platform_dist / "mcp_agentic_workflow"
        
        if not executable.exists():
            # Check if it's in a subdirectory
            for item in platform_dist.iterdir():
                if item.is_dir():
                    potential_executable = item / executable.name
                    if potential_executable.exists():
                        executable = potential_executable
                        break
        
        if not executable.exists():
            raise FileNotFoundError(f"Executable not found. Searched in: {platform_dist}")
        
        return str(executable)
    
    def _create_launcher(self) -> str:
        """Create platform-specific launcher script."""
        platform_dist = self.dist_dir / self.platform
        
        if self.platform == "windows":
            # Create batch file
            launcher_content = f'''@echo off
echo Starting MCP Agentic Workflow Accelerator...
cd /d "%~dp0"
mcp_agentic_workflow.exe
pause
'''
            launcher_path = platform_dist / "start_mcp.bat"
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Create PowerShell script
            ps_content = f'''# PowerShell launcher for MCP Agentic Workflow Accelerator
Write-Host "Starting MCP Agentic Workflow Accelerator..." -ForegroundColor Green
Set-Location $PSScriptRoot
& "mcp_agentic_workflow.exe"
'''
            ps_path = platform_dist / "start_mcp.ps1"
            with open(ps_path, 'w') as f:
                f.write(ps_content)
            
            return str(launcher_path)
            
        else:
            # Create shell script
            launcher_content = f'''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."
cd "$(dirname "$0")"
./mcp_agentic_workflow
'''
            launcher_path = platform_dist / "start_mcp.sh"
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Make executable
            os.chmod(launcher_path, 0o755)
            
            return str(launcher_path)
    
    def _create_portable_package(self):
        """Create portable package."""
        print("  Creating portable package...")
        
        platform_dist = self.dist_dir / self.platform
        portable_dir = self.executables_dir / f"portable_{self.platform}"
        
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        
        shutil.copytree(platform_dist, portable_dir)
        
        # Create portable README
        readme_content = f'''# MCP Agentic Workflow Accelerator - Portable Package

This is a portable version of the MCP Agentic Workflow Accelerator for {self.platform}.

## Quick Start

1. Extract this package to any location
2. Run the launcher:
   - Windows: Double-click `start_mcp.bat` or run `start_mcp.ps1` in PowerShell
   - macOS/Linux: Run `./start_mcp.sh` in Terminal

## Features

- Fully portable - no installation required
- Self-contained Python environment
- All dependencies included
- Cross-platform compatibility

## System Requirements

- {self.platform.capitalize()}
- 4GB RAM minimum
- 2GB free disk space

## Support

For issues and documentation, see the main project repository.
'''
        
        with open(portable_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print(f"    Portable package created: {portable_dir}")
    
    def _clean_build_directories(self):
        """Clean build directories."""
        print("  Cleaning build directories...")
        
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        if self.executables_dir.exists():
            shutil.rmtree(self.executables_dir)
        
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.executables_dir.mkdir(exist_ok=True)

def main():
    """Main function to build executable."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build MCP Agentic Workflow Accelerator executable")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before building")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    builder = SimpleExecutableBuilder(args.project_root)
    
    try:
        result = builder.build_executable(clean=args.clean)
        print(f"\n✅ Successfully built executable: {result}")
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 