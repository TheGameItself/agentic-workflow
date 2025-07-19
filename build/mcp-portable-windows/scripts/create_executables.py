#!/usr/bin/env python3
"""
Executable Builder for MCP Agentic Workflow Accelerator
Creates platform-specific executables for Windows, macOS, and Linux.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional

class ExecutableBuilder:
    """Builds platform-specific executables for the MCP server."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the executable builder."""
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self.platform = platform.system().lower()
        self.arch = platform.machine()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Output directories
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.executables_dir = self.project_root / "executables"
        
        # Create directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.executables_dir.mkdir(exist_ok=True)
        
        # Platform-specific settings
        self.platform_configs = {
            'windows': {
                'ext': '.exe',
                'icon': 'assets/icon.ico',
                'launcher': 'start_mcp.bat',
                'powershell': 'start_mcp.ps1'
            },
            'darwin': {
                'ext': '',
                'icon': 'assets/icon.icns',
                'launcher': 'start_mcp.sh',
                'app_bundle': True
            },
            'linux': {
                'ext': '',
                'icon': 'assets/icon.png',
                'launcher': 'start_mcp.sh',
                'appimage': True
            }
        }
    
    def build_all_executables(self, clean: bool = True) -> Dict[str, str]:
        """Build executables for all supported platforms."""
        print("🚀 Building MCP Agentic Workflow Accelerator Executables")
        print("=" * 60)
        
        if clean:
            self._clean_build_directories()
        
        results = {}
        
        # Build for current platform
        current_platform = self.platform
        print(f"\n📦 Building for current platform: {current_platform}")
        try:
            result = self._build_for_platform(current_platform)
            results[current_platform] = result
        except Exception as e:
            print(f"❌ Failed to build for {current_platform}: {e}")
            results[current_platform] = f"Failed: {e}"
        
        # Cross-compile for other platforms if possible
        if self._can_cross_compile():
            for platform_name in ['windows', 'darwin', 'linux']:
                if platform_name != current_platform:
                    print(f"\n📦 Cross-compiling for {platform_name}")
                    try:
                        result = self._build_for_platform(platform_name, cross_compile=True)
                        results[platform_name] = result
                    except Exception as e:
                        print(f"❌ Failed to cross-compile for {platform_name}: {e}")
                        results[platform_name] = f"Failed: {e}"
        
        # Create installers
        print("\n📦 Creating installers...")
        self._create_installers()
        
        # Create portable packages
        print("\n📦 Creating portable packages...")
        self._create_portable_packages()
        
        # Generate documentation
        print("\n📖 Generating documentation...")
        self._generate_documentation(results)
        
        return results
    
    def _build_for_platform(self, platform_name: str, cross_compile: bool = False) -> str:
        """Build executable for a specific platform."""
        config = self.platform_configs.get(platform_name, {})
        
        print(f"  Building for {platform_name}...")
        
        # Create platform-specific build directory
        platform_build_dir = self.build_dir / f"build_{platform_name}"
        platform_build_dir.mkdir(exist_ok=True)
        
        # Create spec file for PyInstaller
        spec_file = self._create_pyinstaller_spec(platform_name, platform_build_dir)
        
        # Build with PyInstaller
        executable_path = self._build_with_pyinstaller(spec_file, platform_name)
        
        # Create platform-specific launcher
        launcher_path = self._create_platform_launcher(platform_name, executable_path)
        
        # Create app bundle for macOS
        if platform_name == 'darwin' and config.get('app_bundle'):
            app_path = self._create_macos_app_bundle(executable_path)
            return str(app_path)
        
        # Create AppImage for Linux
        if platform_name == 'linux' and config.get('appimage'):
            appimage_path = self._create_linux_appimage(executable_path)
            return str(appimage_path)
        
        return str(executable_path)
    
    def _create_pyinstaller_spec(self, platform_name: str, build_dir: Path) -> Path:
        """Create PyInstaller spec file for the platform."""
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
        ('docs', 'docs'),
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
        'mcp.memory',
        'mcp.advanced_memory',
        'mcp.unified_memory',
        'mcp.task_manager',
        'mcp.context_manager',
        'mcp.reminder_engine',
        'mcp.workflow',
        'mcp.project_manager',
        'mcp.cli',
        'mcp.database_manager',
        'mcp.performance_monitor',
        'mcp.regex_search',
        'mcp.rag_system',
        'mcp.server',
        'mcp.web_interface',
        'mcp.physics_engine',
        'mcp.web_crawler',
        'mcp.research_integration',
        'mcp.plugin_system',
        'mcp.api_enhancements',
        'mcp.experimental_lobes',
        'mcp.lobes',
        'mcp.lobes.alignment_engine',
        'mcp.lobes.pattern_recognition_engine',
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
    [],
    exclude_binaries=True,
    name='mcp_agentic_workflow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mcp_agentic_workflow',
)
'''
        
        spec_file = build_dir / "mcp_agentic_workflow.spec"
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        return spec_file
    
    def _build_with_pyinstaller(self, spec_file: Path, platform_name: str) -> Path:
        """Build executable using PyInstaller."""
        # Install PyInstaller if not available
        try:
            import PyInstaller
        except ImportError:
            print("  Installing PyInstaller...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        
        # Build executable
        print(f"  Running PyInstaller for {platform_name}...")
        cmd = [
            sys.executable, "-m", "PyInstaller",
            str(spec_file),
            "--distpath", str(self.dist_dir / platform_name),
            "--workpath", str(self.build_dir / f"work_{platform_name}"),
            "--clean"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Find the built executable
        dist_platform_dir = self.dist_dir / platform_name / "mcp_agentic_workflow"
        if platform_name == "windows":
            executable = dist_platform_dir / "mcp_agentic_workflow.exe"
        else:
            executable = dist_platform_dir / "mcp_agentic_workflow"
        
        if not executable.exists():
            raise FileNotFoundError(f"Executable not found at {executable}")
        
        return executable
    
    def _create_platform_launcher(self, platform_name: str, executable_path: Path) -> Path:
        """Create platform-specific launcher script."""
        config = self.platform_configs.get(platform_name, {})
        
        if platform_name == "windows":
            # Create batch file
            launcher_content = f'''@echo off
echo Starting MCP Agentic Workflow Accelerator...
cd /d "%~dp0"
"{executable_path.name}"
pause
'''
            launcher_path = self.dist_dir / platform_name / config['launcher']
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Create PowerShell script
            ps_content = f'''# PowerShell launcher for MCP Agentic Workflow Accelerator
Write-Host "Starting MCP Agentic Workflow Accelerator..." -ForegroundColor Green
Set-Location $PSScriptRoot
& "{executable_path.name}"
'''
            ps_path = self.dist_dir / platform_name / config['powershell']
            with open(ps_path, 'w') as f:
                f.write(ps_content)
            
            return launcher_path
            
        else:
            # Create shell script
            launcher_content = f'''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."
cd "$(dirname "$0")"
./mcp_agentic_workflow/mcp_agentic_workflow
'''
            launcher_path = self.dist_dir / platform_name / config['launcher']
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Make executable
            os.chmod(launcher_path, 0o755)
            
            return launcher_path
    
    def _create_macos_app_bundle(self, executable_path: Path) -> Path:
        """Create macOS app bundle."""
        app_name = "MCP Agentic Workflow Accelerator.app"
        app_path = self.dist_dir / "darwin" / app_name
        contents_path = app_path / "Contents"
        
        # Create app bundle structure
        (contents_path / "MacOS").mkdir(parents=True, exist_ok=True)
        (contents_path / "Resources").mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        shutil.copy2(executable_path, contents_path / "MacOS" / "mcp_agentic_workflow")
        
        # Create Info.plist
        info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>mcp_agentic_workflow</string>
    <key>CFBundleIdentifier</key>
    <string>com.mcp.agentic-workflow</string>
    <key>CFBundleName</key>
    <string>MCP Agentic Workflow Accelerator</string>
    <key>CFBundleVersion</key>
    <string>2.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.14</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
'''
        with open(contents_path / "Info.plist", 'w') as f:
            f.write(info_plist)
        
        return app_path
    
    def _create_linux_appimage(self, executable_path: Path) -> Path:
        """Create Linux AppImage."""
        # This is a simplified AppImage creation
        # In a real implementation, you'd use appimagetool
        appimage_name = "mcp_agentic_workflow-x86_64.AppImage"
        appimage_path = self.dist_dir / "linux" / appimage_name
        
        # For now, just create a shell script that can be converted to AppImage
        appimage_script = f'''#!/bin/bash
# AppImage launcher for MCP Agentic Workflow Accelerator
cd "$(dirname "$0")"
./mcp_agentic_workflow/mcp_agentic_workflow
'''
        
        with open(appimage_path, 'w') as f:
            f.write(appimage_script)
        
        os.chmod(appimage_path, 0o755)
        
        return appimage_path
    
    def _create_installers(self):
        """Create platform-specific installers."""
        print("  Creating installers...")
        
        # Windows installer (using NSIS if available)
        if self.platform == "windows":
            self._create_windows_installer()
        
        # macOS installer
        if self.platform == "darwin":
            self._create_macos_installer()
        
        # Linux installer
        if self.platform == "linux":
            self._create_linux_installer()
    
    def _create_windows_installer(self):
        """Create Windows installer using NSIS."""
        nsis_script = '''
!include "MUI2.nsh"

Name "MCP Agentic Workflow Accelerator"
OutFile "mcp_agentic_workflow_setup.exe"
InstallDir "$PROGRAMFILES\\MCP Agentic Workflow Accelerator"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File /r "windows\\*"
    
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    
    CreateDirectory "$SMPROGRAMS\\MCP Agentic Workflow Accelerator"
    CreateShortCut "$SMPROGRAMS\\MCP Agentic Workflow Accelerator\\MCP Agentic Workflow Accelerator.lnk" "$INSTDIR\\mcp_agentic_workflow.exe"
    CreateShortCut "$SMPROGRAMS\\MCP Agentic Workflow Accelerator\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MCP Agentic Workflow Accelerator" "DisplayName" "MCP Agentic Workflow Accelerator"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MCP Agentic Workflow Accelerator" "UninstallString" "$INSTDIR\\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\uninstall.exe"
    RMDir /r "$INSTDIR"
    
    Delete "$SMPROGRAMS\\MCP Agentic Workflow Accelerator\\MCP Agentic Workflow Accelerator.lnk"
    Delete "$SMPROGRAMS\\MCP Agentic Workflow Accelerator\\Uninstall.lnk"
    RMDir "$SMPROGRAMS\\MCP Agentic Workflow Accelerator"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MCP Agentic Workflow Accelerator"
SectionEnd
'''
        
        nsis_file = self.build_dir / "installer.nsi"
        with open(nsis_file, 'w') as f:
            f.write(nsis_script)
        
        # Try to build installer if NSIS is available
        try:
            subprocess.run(["makensis", str(nsis_file)], check=True)
            print("  Windows installer created successfully")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  NSIS not found, skipping Windows installer")
    
    def _create_macos_installer(self):
        """Create macOS installer package."""
        # Ensure directory exists
        darwin_dist_dir = self.dist_dir / "darwin"
        darwin_dist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple installer script
        installer_script = '''#!/bin/bash
echo "Installing MCP Agentic Workflow Accelerator..."

# Create application directory
sudo mkdir -p /Applications/MCP\ Agentic\ Workflow\ Accelerator.app/Contents/MacOS
sudo mkdir -p /Applications/MCP\ Agentic\ Workflow\ Accelerator.app/Contents/Resources

# Copy files
sudo cp -r mcp_agentic_workflow/* /Applications/MCP\ Agentic\ Workflow\ Accelerator.app/Contents/MacOS/

echo "Installation complete!"
'''
        
        installer_path = darwin_dist_dir / "install.sh"
        with open(installer_path, 'w') as f:
            f.write(installer_script)
        
        os.chmod(installer_path, 0o755)
        print("  macOS installer script created")
    
    def _create_linux_installer(self):
        """Create Linux installer package."""
        # Ensure directory exists
        linux_dist_dir = self.dist_dir / "linux"
        linux_dist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple installer script
        installer_script = '''#!/bin/bash
echo "Installing MCP Agentic Workflow Accelerator..."

# Create application directory
sudo mkdir -p /opt/mcp-agentic-workflow
sudo mkdir -p /usr/local/bin

# Copy files
sudo cp -r mcp_agentic_workflow/* /opt/mcp-agentic-workflow/

# Create symlink
sudo ln -sf /opt/mcp-agentic-workflow/mcp_agentic_workflow /usr/local/bin/mcp-agentic-workflow

# Create desktop file
cat > mcp-agentic-workflow.desktop << EOF
[Desktop Entry]
Name=MCP Agentic Workflow Accelerator
Comment=Portable MCP server for agentic development workflows
Exec=/usr/local/bin/mcp-agentic-workflow
Icon=/opt/mcp-agentic-workflow/icon.png
Terminal=true
Type=Application
Categories=Development;
EOF

sudo cp mcp-agentic-workflow.desktop /usr/share/applications/

echo "Installation complete!"
'''
        
        installer_path = linux_dist_dir / "install.sh"
        with open(installer_path, 'w') as f:
            f.write(installer_script)
        
        os.chmod(installer_path, 0o755)
        print("  Linux installer script created")
    
    def _create_portable_packages(self):
        """Create portable packages for each platform."""
        print("  Creating portable packages...")
        
        for platform_name in ['windows', 'darwin', 'linux']:
            platform_dist = self.dist_dir / platform_name
            if platform_dist.exists():
                portable_dir = self.executables_dir / f"portable_{platform_name}"
                if portable_dir.exists():
                    shutil.rmtree(portable_dir)
                
                shutil.copytree(platform_dist, portable_dir)
                
                # Create portable README
                readme_content = f'''# MCP Agentic Workflow Accelerator - Portable Package

This is a portable version of the MCP Agentic Workflow Accelerator for {platform_name}.

## Quick Start

1. Extract this package to any location
2. Run the launcher:
   - Windows: Double-click `start_mcp.bat` or run `start_mcp.ps1` in PowerShell
   - macOS: Run `./start_mcp.sh` in Terminal
   - Linux: Run `./start_mcp.sh` in Terminal

## Features

- Fully portable - no installation required
- Self-contained Python environment
- All dependencies included
- Cross-platform compatibility

## System Requirements

- {platform_name.capitalize()}
- 4GB RAM minimum
- 2GB free disk space

## Support

For issues and documentation, see the main project repository.
'''
                
                with open(portable_dir / "README.md", 'w') as f:
                    f.write(readme_content)
                
                print(f"    Portable package created for {platform_name}")
    
    def _generate_documentation(self, results: Dict[str, str]):
        """Generate documentation for the built executables."""
        print("  Generating documentation...")
        
        doc_content = f'''# MCP Agentic Workflow Accelerator - Executables

This document describes the built executables for the MCP Agentic Workflow Accelerator.

## Build Information

- Build Date: {self._get_current_timestamp()}
- Python Version: {self.python_version}
- Build Platform: {self.platform} ({self.arch})

## Available Executables

'''
        
        for platform_name, result in results.items():
            if result.startswith("Failed"):
                doc_content += f'''
### {platform_name.capitalize()}
❌ Build failed: {result}
'''
            else:
                doc_content += f'''
### {platform_name.capitalize()}
✅ Executable: {result}
- Type: {'App Bundle' if platform_name == 'darwin' else 'Executable'}
- Size: {self._get_file_size(result)}
'''
        
        doc_content += f'''

## Installation Instructions

### Windows
1. Download the Windows executable package
2. Extract to desired location
3. Run `start_mcp.bat` or `start_mcp.ps1`

### macOS
1. Download the macOS app bundle
2. Drag to Applications folder or run directly
3. Or use the portable package with `./start_mcp.sh`

### Linux
1. Download the Linux executable package
2. Extract to desired location
3. Run `./start_mcp.sh`
4. Or use the AppImage if available

## Portable Packages

Portable packages are available in the `executables/portable_*` directories.
These packages contain everything needed to run the MCP server without installation.

## Troubleshooting

### Common Issues

1. **Permission Denied (Linux/macOS)**
   - Make sure the launcher script is executable: `chmod +x start_mcp.sh`

2. **Missing Dependencies**
   - The portable packages include all dependencies
   - If using the executable directly, ensure Python 3.8+ is installed

3. **Antivirus False Positive**
   - Some antivirus software may flag PyInstaller executables
   - Add the executable directory to your antivirus exclusions

### Getting Help

- Check the main project README for detailed documentation
- Review the logs in the `data/` directory
- Report issues on the project repository

## Technical Details

- Built with PyInstaller
- Self-contained Python environment
- All project dependencies included
- Cross-platform compatibility
- No external dependencies required

## License

This software is licensed under the MIT License.
'''
        
        doc_path = self.executables_dir / "EXECUTABLES_README.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        print("    Documentation generated")
    
    def _can_cross_compile(self) -> bool:
        """Check if cross-compilation is possible."""
        # This is a simplified check - in practice, you'd need proper cross-compilation setup
        return False
    
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
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size."""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except OSError:
            return "Unknown"

def main():
    """Main function to build executables."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build MCP Agentic Workflow Accelerator executables")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before building")
    parser.add_argument("--platform", choices=["windows", "darwin", "linux"], help="Build for specific platform only")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    builder = ExecutableBuilder(args.project_root)
    
    if args.platform:
        # Build for specific platform
        result = builder._build_for_platform(args.platform)
        print(f"\n✅ Built executable for {args.platform}: {result}")
    else:
        # Build for all platforms
        results = builder.build_all_executables(clean=args.clean)
        
        print("\n" + "=" * 60)
        print("🏁 Build Summary")
        print("=" * 60)
        
        for platform_name, result in results.items():
            if result.startswith("Failed"):
                print(f"❌ {platform_name}: {result}")
            else:
                print(f"✅ {platform_name}: {result}")
        
        print(f"\n📁 Executables directory: {builder.executables_dir}")
        print(f"📁 Distribution directory: {builder.dist_dir}")

if __name__ == "__main__":
    main() 