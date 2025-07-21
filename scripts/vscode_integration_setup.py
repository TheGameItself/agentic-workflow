#!/usr/bin/env python3
"""
VS Code Integration Setup Script for MCP Agentic Workflow Accelerator

This script sets up comprehensive VS Code integration similar to Cursor's MCP integration,
including configuration files, tasks, launch configurations, and keybindings.
"""

import os
import json
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class VSCodeIntegrationSetup:
    """Setup VS Code integration for MCP system."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.config_dir = self.project_root / "config"
        self.vscode_dir = self.project_root / ".vscode"
        self.system = platform.system().lower()
        
    def setup_integration(self) -> bool:
        """Setup complete VS Code integration."""
        try:
            print("🚀 Setting up VS Code integration for MCP Agentic Workflow...")
            
            # Create .vscode directory
            self.vscode_dir.mkdir(exist_ok=True)
            
            # Copy configuration files
            self._copy_config_files()
            
            # Setup workspace configuration
            self._setup_workspace()
            
            # Install recommended extensions
            self._install_extensions()
            
            # Setup MCP server configuration
            self._setup_mcp_server_config()
            
            # Create launch scripts
            self._create_launch_scripts()
            
            print("✅ VS Code integration setup complete!")
            print("\n📋 Next steps:")
            print("1. Open VS Code in this directory")
            print("2. Install recommended extensions when prompted")
            print("3. Use Ctrl+Shift+P to access MCP commands")
            print("4. Use F5 to debug MCP server")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up VS Code integration: {e}")
            return False
    
    def _copy_config_files(self):
        """Copy configuration files from config/ to .vscode/."""
        config_mappings = {
            "vscode-settings.json": "settings.json",
            "vscode-tasks.json": "tasks.json",
            "vscode-launch.json": "launch.json",
            "vscode-keybindings.json": "keybindings.json",
            "vscode-extensions.json": "extensions.json"
        }
        
        for source_file, target_file in config_mappings.items():
            source_path = self.config_dir / source_file
            target_path = self.vscode_dir / target_file
            
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                print(f"📄 Copied {source_file} → .vscode/{target_file}")
            else:
                print(f"⚠️  Warning: {source_file} not found in config/")
    
    def _setup_workspace(self):
        """Setup VS Code workspace configuration."""
        workspace_file = self.project_root / "mcp-agentic-workflow.code-workspace"
        workspace_config_source = self.config_dir / "vscode-workspace.code-workspace"
        
        if workspace_config_source.exists():
            shutil.copy2(workspace_config_source, workspace_file)
            print(f"📄 Created workspace file: {workspace_file.name}")
        
        # Update workspace paths
        self._update_workspace_paths(workspace_file)
    
    def _update_workspace_paths(self, workspace_file: Path):
        """Update workspace file with correct paths."""
        if not workspace_file.exists():
            return
            
        try:
            with open(workspace_file, 'r') as f:
                workspace_config = json.load(f)
            
            # Update Python interpreter path based on system
            python_path = self._get_python_interpreter_path()
            if python_path:
                workspace_config["settings"]["python.defaultInterpreterPath"] = python_path
            
            # Update environment variables
            workspace_folder = str(self.project_root)
            for env_key in ["terminal.integrated.env.linux", "terminal.integrated.env.osx", "terminal.integrated.env.windows"]:
                if env_key in workspace_config["settings"]:
                    workspace_config["settings"][env_key]["MCP_PROJECT_PATH"] = workspace_folder
                    workspace_config["settings"][env_key]["PYTHONPATH"] = workspace_folder
            
            with open(workspace_file, 'w') as f:
                json.dump(workspace_config, f, indent=2)
                
            print(f"✅ Updated workspace configuration with project paths")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not update workspace paths: {e}")
    
    def _get_python_interpreter_path(self) -> Optional[str]:
        """Get the appropriate Python interpreter path."""
        venv_paths = [
            self.project_root / "venv" / "bin" / "python",  # Linux/macOS
            self.project_root / "venv" / "Scripts" / "python.exe",  # Windows
            self.project_root / ".venv" / "bin" / "python",  # Linux/macOS
            self.project_root / ".venv" / "Scripts" / "python.exe",  # Windows
        ]
        
        for path in venv_paths:
            if path.exists():
                return str(path)
        
        # Fallback to system Python
        try:
            result = subprocess.run(["which", "python3"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def _install_extensions(self):
        """Install recommended VS Code extensions."""
        extensions_file = self.vscode_dir / "extensions.json"
        
        if not extensions_file.exists():
            print("⚠️  Extensions file not found, skipping extension installation")
            return
        
        try:
            with open(extensions_file, 'r') as f:
                extensions_config = json.load(f)
            
            recommendations = extensions_config.get("recommendations", [])
            
            print(f"📦 Found {len(recommendations)} recommended extensions")
            
            # Check if VS Code CLI is available
            if self._is_vscode_cli_available():
                print("🔧 Installing extensions via VS Code CLI...")
                for extension in recommendations:
                    try:
                        subprocess.run(["code", "--install-extension", extension], 
                                     check=True, capture_output=True)
                        print(f"✅ Installed: {extension}")
                    except subprocess.CalledProcessError:
                        print(f"⚠️  Failed to install: {extension}")
            else:
                print("💡 VS Code CLI not available. Extensions will be suggested when you open VS Code.")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not process extensions: {e}")
    
    def _is_vscode_cli_available(self) -> bool:
        """Check if VS Code CLI is available."""
        try:
            subprocess.run(["code", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _setup_mcp_server_config(self):
        """Setup MCP server configuration for VS Code."""
        mcp_config_file = self.vscode_dir / "mcp-config.json"
        mcp_config_source = self.config_dir / "vscode-mcp-integration.json"
        
        if mcp_config_source.exists():
            shutil.copy2(mcp_config_source, mcp_config_file)
            print(f"📄 Created MCP configuration: {mcp_config_file.name}")
            
            # Update paths in MCP config
            self._update_mcp_config_paths(mcp_config_file)
    
    def _update_mcp_config_paths(self, mcp_config_file: Path):
        """Update MCP configuration with correct paths."""
        try:
            with open(mcp_config_file, 'r') as f:
                mcp_config = json.load(f)
            
            # Update environment variables
            workspace_folder = str(self.project_root)
            for server_name, server_config in mcp_config.get("mcpServers", {}).items():
                if "env" in server_config:
                    server_config["env"]["MCP_PROJECT_PATH"] = workspace_folder
                    server_config["env"]["PYTHONPATH"] = workspace_folder
                
                if "cwd" in server_config:
                    server_config["cwd"] = workspace_folder
            
            with open(mcp_config_file, 'w') as f:
                json.dump(mcp_config, f, indent=2)
                
            print(f"✅ Updated MCP configuration with project paths")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not update MCP config paths: {e}")
    
    def _create_launch_scripts(self):
        """Create launch scripts for different platforms."""
        scripts = {
            "launch_vscode.sh": self._create_unix_launch_script(),
            "launch_vscode.bat": self._create_windows_launch_script(),
            "launch_vscode.ps1": self._create_powershell_launch_script()
        }
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        for script_name, script_content in scripts.items():
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make Unix scripts executable
            if script_name.endswith('.sh'):
                os.chmod(script_path, 0o755)
            
            print(f"📄 Created launch script: {script_name}")
    
    def _create_unix_launch_script(self) -> str:
        """Create Unix launch script."""
        return f'''#!/bin/bash
# VS Code Launch Script for MCP Agentic Workflow
# This script sets up the environment and launches VS Code with MCP integration

set -e

PROJECT_ROOT="{self.project_root}"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
fi

# Set environment variables
export MCP_PROJECT_PATH="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export MCP_LOG_LEVEL="INFO"
export MCP_AUTO_UPDATE_ENABLED="true"
export MCP_PERFORMANCE_MONITORING="true"

# Check if VS Code is installed
if ! command -v code &> /dev/null; then
    echo "❌ VS Code CLI not found. Please install VS Code and add it to PATH."
    exit 1
fi

# Launch VS Code with workspace
echo "🚀 Launching VS Code with MCP integration..."
if [ -f "mcp-agentic-workflow.code-workspace" ]; then
    code mcp-agentic-workflow.code-workspace
else
    code .
fi

echo "✅ VS Code launched successfully!"
'''
    
    def _create_windows_launch_script(self) -> str:
        """Create Windows batch launch script."""
        return f'''@echo off
REM VS Code Launch Script for MCP Agentic Workflow
REM This script sets up the environment and launches VS Code with MCP integration

set PROJECT_ROOT={self.project_root}
cd /d "%PROJECT_ROOT%"

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    echo 🐍 Activating virtual environment...
    call venv\\Scripts\\activate.bat
) else if exist ".venv\\Scripts\\activate.bat" (
    echo 🐍 Activating virtual environment...
    call .venv\\Scripts\\activate.bat
)

REM Set environment variables
set MCP_PROJECT_PATH=%PROJECT_ROOT%
set PYTHONPATH=%PROJECT_ROOT%
set MCP_LOG_LEVEL=INFO
set MCP_AUTO_UPDATE_ENABLED=true
set MCP_PERFORMANCE_MONITORING=true

REM Check if VS Code is installed
where code >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ VS Code CLI not found. Please install VS Code and add it to PATH.
    pause
    exit /b 1
)

REM Launch VS Code with workspace
echo 🚀 Launching VS Code with MCP integration...
if exist "mcp-agentic-workflow.code-workspace" (
    code mcp-agentic-workflow.code-workspace
) else (
    code .
)

echo ✅ VS Code launched successfully!
pause
'''
    
    def _create_powershell_launch_script(self) -> str:
        """Create PowerShell launch script."""
        return f'''# VS Code Launch Script for MCP Agentic Workflow
# This script sets up the environment and launches VS Code with MCP integration

$PROJECT_ROOT = "{self.project_root}"
Set-Location $PROJECT_ROOT

# Activate virtual environment if it exists
if (Test-Path "venv\\Scripts\\Activate.ps1") {{
    Write-Host "🐍 Activating virtual environment..." -ForegroundColor Green
    & .\\venv\\Scripts\\Activate.ps1
}} elseif (Test-Path ".venv\\Scripts\\Activate.ps1") {{
    Write-Host "🐍 Activating virtual environment..." -ForegroundColor Green
    & .\\.venv\\Scripts\\Activate.ps1
}}

# Set environment variables
$env:MCP_PROJECT_PATH = $PROJECT_ROOT
$env:PYTHONPATH = $PROJECT_ROOT
$env:MCP_LOG_LEVEL = "INFO"
$env:MCP_AUTO_UPDATE_ENABLED = "true"
$env:MCP_PERFORMANCE_MONITORING = "true"

# Check if VS Code is installed
try {{
    Get-Command code -ErrorAction Stop | Out-Null
}} catch {{
    Write-Host "❌ VS Code CLI not found. Please install VS Code and add it to PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}}

# Launch VS Code with workspace
Write-Host "🚀 Launching VS Code with MCP integration..." -ForegroundColor Green
if (Test-Path "mcp-agentic-workflow.code-workspace") {{
    code mcp-agentic-workflow.code-workspace
}} else {{
    code .
}}

Write-Host "✅ VS Code launched successfully!" -ForegroundColor Green
'''

    def create_portable_vscode_package(self, output_dir: str = "portable_vscode_mcp"):
        """Create a portable VS Code package with MCP integration."""
        try:
            print("📦 Creating portable VS Code package with MCP integration...")
            
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Copy project files
            self._copy_project_files(output_path)
            
            # Setup VS Code portable configuration
            self._setup_portable_vscode_config(output_path)
            
            # Create portable launcher
            self._create_portable_launcher(output_path)
            
            print(f"✅ Portable VS Code package created in: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating portable package: {e}")
            return False
    
    def _copy_project_files(self, output_path: Path):
        """Copy essential project files to portable package."""
        essential_dirs = ["src", "config", "scripts", "docs", "tests"]
        essential_files = [
            "README.md", "requirements.txt", "pyproject.toml", "setup.py",
            "mcp_cli.py", "simple_mcp_cli.py", "LICENSE"
        ]
        
        for dir_name in essential_dirs:
            src_dir = self.project_root / dir_name
            if src_dir.exists():
                dst_dir = output_path / dir_name
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print(f"📁 Copied directory: {dir_name}")
        
        for file_name in essential_files:
            src_file = self.project_root / file_name
            if src_file.exists():
                dst_file = output_path / file_name
                shutil.copy2(src_file, dst_file)
                print(f"📄 Copied file: {file_name}")
    
    def _setup_portable_vscode_config(self, output_path: Path):
        """Setup VS Code configuration for portable package."""
        vscode_dir = output_path / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Copy VS Code configuration files
        for config_file in self.vscode_dir.glob("*.json"):
            dst_file = vscode_dir / config_file.name
            shutil.copy2(config_file, dst_file)
            print(f"📄 Copied VS Code config: {config_file.name}")
        
        # Copy workspace file
        workspace_file = self.project_root / "mcp-agentic-workflow.code-workspace"
        if workspace_file.exists():
            dst_workspace = output_path / workspace_file.name
            shutil.copy2(workspace_file, dst_workspace)
            print(f"📄 Copied workspace file")
    
    def _create_portable_launcher(self, output_path: Path):
        """Create portable launcher script."""
        launcher_content = f'''#!/bin/bash
# Portable VS Code Launcher for MCP Agentic Workflow

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup environment
export MCP_PROJECT_PATH="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"
export MCP_LOG_LEVEL="INFO"
export MCP_AUTO_UPDATE_ENABLED="true"
export MCP_PERFORMANCE_MONITORING="true"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Launch VS Code
echo "🚀 Launching portable VS Code with MCP integration..."
if command -v code &> /dev/null; then
    if [ -f "mcp-agentic-workflow.code-workspace" ]; then
        code mcp-agentic-workflow.code-workspace
    else
        code .
    fi
else
    echo "❌ VS Code not found. Please install VS Code first."
    exit 1
fi
'''
        
        launcher_path = output_path / "launch_portable_vscode.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        os.chmod(launcher_path, 0o755)
        
        print(f"📄 Created portable launcher: {launcher_path.name}")


def main():
    """Main function to setup VS Code integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup VS Code integration for MCP Agentic Workflow")
    parser.add_argument("--project-root", help="Project root directory", default=None)
    parser.add_argument("--portable", help="Create portable package", action="store_true")
    parser.add_argument("--output-dir", help="Output directory for portable package", default="portable_vscode_mcp")
    
    args = parser.parse_args()
    
    setup = VSCodeIntegrationSetup(args.project_root)
    
    if args.portable:
        success = setup.create_portable_vscode_package(args.output_dir)
    else:
        success = setup.setup_integration()
    
    if success:
        print("\n🎉 VS Code integration setup completed successfully!")
    else:
        print("\n❌ VS Code integration setup failed!")
        exit(1)


if __name__ == "__main__":
    main()