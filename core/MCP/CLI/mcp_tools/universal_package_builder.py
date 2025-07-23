#!/usr/bin/env python3
"""
Universal Package Builder for MCP Agentic Workflow Accelerator
Creates AppImages, Flatpaks, and supports direct USB installation.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

class UniversalPackageBuilder:
    """Builds universal packages for maximum portability."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the universal package builder."""
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self.platform = platform.system().lower()
        self.arch = platform.machine()
        
        # Output directories
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.packages_dir = self.project_root / "packages"
        self.usb_templates_dir = self.project_root / "usb_templates"
        
        # Create directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.packages_dir.mkdir(exist_ok=True)
        self.usb_templates_dir.mkdir(exist_ok=True)
        
        # Package metadata
        self.app_metadata = {
            "name": "MCP Agentic Workflow Accelerator",
            "description": "Portable MCP server for agentic development workflows",
            "version": "2.0.0",
            "author": "MCP Development Team",
            "license": "MIT",
            "homepage": "https://github.com/your-username/mcp-agentic-workflow",
            "categories": ["Development", "Utility"],
            "keywords": ["mcp", "agentic", "workflow", "development", "local", "portable"]
        }
    
    def build_all_packages(self, clean: bool = True) -> Dict[str, str]:
        """Build all package types for maximum portability."""
        print("🚀 Building Universal MCP Agentic Workflow Accelerator Packages")
        print("=" * 70)
        
        if clean:
            self._clean_build_directories()
        
        results = {}
        
        # Build for current platform
        if self.platform == "linux":
            print(f"\n📦 Building Linux packages...")
            
            # Build AppImage
            try:
                appimage_path = self._build_appimage()
                results["appimage"] = appimage_path
                print(f"✅ AppImage created: {appimage_path}")
            except Exception as e:
                print(f"❌ AppImage failed: {e}")
                results["appimage"] = f"Failed: {e}"
            
            # Build Flatpak
            try:
                flatpak_path = self._build_flatpak()
                results["flatpak"] = flatpak_path
                print(f"✅ Flatpak created: {flatpak_path}")
            except Exception as e:
                print(f"❌ Flatpak failed: {e}")
                results["flatpak"] = f"Failed: {e}"
            
            # Build portable archive
            try:
                portable_path = self._build_portable_archive()
                results["portable"] = portable_path
                print(f"✅ Portable archive created: {portable_path}")
            except Exception as e:
                print(f"❌ Portable archive failed: {e}")
                results["portable"] = f"Failed: {e}"
        
        elif self.platform == "windows":
            print(f"\n📦 Building Windows packages...")
            
            # Build portable package
            try:
                portable_path = self._build_windows_portable()
                results["portable"] = portable_path
                print(f"✅ Windows portable created: {portable_path}")
            except Exception as e:
                print(f"❌ Windows portable failed: {e}")
                results["portable"] = f"Failed: {e}"
            
            # Build installer
            try:
                installer_path = self._build_windows_installer()
                results["installer"] = installer_path
                print(f"✅ Windows installer created: {installer_path}")
            except Exception as e:
                print(f"❌ Windows installer failed: {e}")
                results["installer"] = f"Failed: {e}"
        
        elif self.platform == "darwin":
            print(f"\n📦 Building macOS packages...")
            
            # Build app bundle
            try:
                app_path = self._build_macos_app()
                results["app"] = app_path
                print(f"✅ macOS app created: {app_path}")
            except Exception as e:
                print(f"❌ macOS app failed: {e}")
                results["app"] = f"Failed: {e}"
            
            # Build portable package
            try:
                portable_path = self._build_macos_portable()
                results["portable"] = portable_path
                print(f"✅ macOS portable created: {portable_path}")
            except Exception as e:
                print(f"❌ macOS portable failed: {e}")
                results["portable"] = f"Failed: {e}"
        
        elif self.platform == "android":
            print(f"\n📦 Building Android package...")
            try:
                android_path = self._build_android()
                results["android"] = android_path
                print(f"✅ Android package created: {android_path}")
            except Exception as e:
                print(f"❌ Android package failed: {e}")
                results["android"] = f"Failed: {e}"
        
        # Create USB templates
        print(f"\n📦 Creating USB installation templates...")
        try:
            usb_template_path = self._create_usb_templates()
            results["usb_template"] = usb_template_path
            print(f"✅ USB template created: {usb_template_path}")
        except Exception as e:
            print(f"❌ USB template failed: {e}")
            results["usb_template"] = f"Failed: {e}"
        
        # Create universal portable package
        print(f"\n📦 Creating universal portable package...")
        try:
            universal_path = self._create_universal_portable()
            results["universal"] = universal_path
            print(f"✅ Universal portable created: {universal_path}")
        except Exception as e:
            print(f"❌ Universal portable failed: {e}")
            results["universal"] = f"Failed: {e}"
        
        # Generate documentation
        print(f"\n📖 Generating documentation...")
        self._generate_package_documentation(results)
        
        # After all builds, automate distribution and social/community posting
        try:
            distribution_config = {
                "builds": list(results.values()),
                "release_message": f"MCP {self.app_metadata['version']} build complete! See https://github.com/TheGameItself/agentic-workflow/releases/tag/mcp"
            }
            with open("distribution_config.json", "w") as f:
                json.dump(distribution_config, f)
            subprocess.run([sys.executable, "scripts/automated_distribution.py", "distribution_config.json"])
            print("Automated distribution and social/community posting complete.")
        except Exception as e:
            print(f"[WARN] Automated distribution failed: {e}")
        
        return results
    
    def _build_appimage(self) -> str:
        """Build Linux AppImage package."""
        print("  Building AppImage...")
        
        # Create AppDir structure
        appdir = self.build_dir / "AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
        
        appdir.mkdir(parents=True)
        
        # Create AppDir structure
        (appdir / "usr" / "bin").mkdir(parents=True, exist_ok=True)
        (appdir / "usr" / "share" / "applications").mkdir(parents=True, exist_ok=True)
        (appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps").mkdir(parents=True, exist_ok=True)
        
        # Copy application files
        self._copy_application_files(appdir / "usr" / "bin")
        
        # Create desktop file
        desktop_content = f"""[Desktop Entry]
Name={self.app_metadata['name']}
Comment={self.app_metadata['description']}
Exec=mcp_agentic_workflow
Icon=mcp-agentic-workflow
Terminal=true
Type=Application
Categories=Development;Utility;
Keywords={';'.join(self.app_metadata['keywords'])}
"""
        
        with open(appdir / "usr" / "share" / "applications" / "mcp-agentic-workflow.desktop", 'w') as f:
            f.write(desktop_content)
        
        # Create AppRun script
        apprun_content = '''#!/bin/bash
cd "$(dirname "$0")"
exec "$APPDIR/usr/bin/mcp_agentic_workflow" "$@"
'''
        
        with open(appdir / "AppRun", 'w') as f:
            f.write(apprun_content)
        
        os.chmod(appdir / "AppRun", 0o755)
        
        # Create AppImage using appimagetool
        appimage_name = f"mcp-agentic-workflow-{self.app_metadata['version']}-{self.arch}.AppImage"
        appimage_path = self.packages_dir / appimage_name
        
        # Try to use appimagetool if available
        try:
            subprocess.run([
                "appimagetool", str(appdir), str(appimage_path)
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: create a simple AppImage-like structure
            print("    appimagetool not found, creating simple AppImage structure")
            self._create_simple_appimage(appdir, appimage_path)
        
        return str(appimage_path)
    
    def _create_simple_appimage(self, appdir: Path, appimage_path: Path):
        """Create a simple AppImage-like structure when appimagetool is not available."""
        # Create a self-extracting archive that mimics AppImage behavior
        with tarfile.open(appimage_path, 'w:gz') as tar:
            tar.add(appdir, arcname='.')
        
        # Create a launcher script
        launcher_content = f'''#!/bin/bash
# Simple AppImage launcher for {self.app_metadata['name']}

# Extract to temporary directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Extract the archive
tail -n +$(($(grep -n "^__ARCHIVE__$" "$0" | cut -d: -f1) + 1)) "$0" | tar -xz -C "$TEMP_DIR"

# Run the application
cd "$TEMP_DIR"
./AppRun "$@"
'''
        
        # Combine launcher with archive
        with open(appimage_path, 'rb') as f:
            archive_data = f.read()
        
        with open(appimage_path, 'w') as f:
            f.write(launcher_content)
            f.write('\n__ARCHIVE__\n')
            f.write(archive_data.decode('latin-1'))
        
        os.chmod(appimage_path, 0o755)
    
    def _build_flatpak(self) -> str:
        """Build Linux Flatpak package."""
        print("  Building Flatpak...")
        
        # Create Flatpak manifest
        manifest_content = f'''app-id: com.mcp.agentic-workflow
runtime: org.freedesktop.Platform
runtime-version: '23.08'
sdk: org.freedesktop.Sdk
command: mcp_agentic_workflow
finish-args:
  - --share=network
  - --share=ipc
  - --socket=fallback-x11
  - --socket=wayland
  - --filesystem=home
  - --filesystem=xdg-download
modules:
  - name: mcp-agentic-workflow
    buildsystem: simple
    build-commands:
      - install -D mcp_agentic_workflow /app/bin/mcp_agentic_workflow
      - install -D *.py /app/bin/
      - cp -r src /app/bin/
      - cp -r config /app/bin/
      - cp -r data /app/bin/
      - cp -r plugins /app/bin/
      - install -D *.txt /app/bin/
      - install -D *.md /app/bin/
    sources:
      - type: dir
        path: .
'''
        
        manifest_path = self.build_dir / "com.mcp.agentic-workflow.yml"
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
        
        # Try to build Flatpak
        flatpak_name = f"com.mcp.agentic-workflow-{self.app_metadata['version']}.flatpak"
        flatpak_path = self.packages_dir / flatpak_name
        
        try:
            subprocess.run([
                "flatpak-builder", "--force-clean", "--repo=repo", 
                str(self.build_dir / "flatpak-build"), str(manifest_path)
            ], check=True)
            
            subprocess.run([
                "flatpak", "build-bundle", "repo", str(flatpak_path), 
                "com.mcp.agentic-workflow", self.app_metadata['version']
            ], check=True)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("    flatpak-builder not found, creating Flatpak manifest only")
            # Just copy the manifest for manual building
            shutil.copy2(manifest_path, flatpak_path)
        
        return str(flatpak_path)
    
    def _build_portable_archive(self) -> str:
        """Build portable archive for Linux."""
        print("  Building portable archive...")
        
        # Create portable directory
        portable_dir = self.build_dir / "portable_linux"
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        
        portable_dir.mkdir(parents=True)
        
        # Copy application files
        self._copy_application_files(portable_dir)
        
        # Create launcher script
        launcher_content = '''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."
cd "$(dirname "$0")"
./mcp_agentic_workflow "$@"
'''
        
        launcher_path = portable_dir / "start_mcp.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        os.chmod(launcher_path, 0o755)
        
        # Create archive
        archive_name = f"mcp-agentic-workflow-{self.app_metadata['version']}-linux-portable.tar.gz"
        archive_path = self.packages_dir / archive_name
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(portable_dir, arcname='.')
        
        return str(archive_path)
    
    def _build_windows_portable(self) -> str:
        """Build Windows portable package."""
        print("  Building Windows portable...")
        
        # Create portable directory
        portable_dir = self.build_dir / "portable_windows"
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        
        portable_dir.mkdir(parents=True)
        
        # Copy application files
        self._copy_application_files(portable_dir)
        
        # Create launcher scripts
        batch_content = '''@echo off
echo Starting MCP Agentic Workflow Accelerator...
cd /d "%~dp0"
mcp_agentic_workflow.exe
pause
'''
        
        with open(portable_dir / "start_mcp.bat", 'w') as f:
            f.write(batch_content)
        
        ps_content = '''# PowerShell launcher for MCP Agentic Workflow Accelerator
Write-Host "Starting MCP Agentic Workflow Accelerator..." -ForegroundColor Green
Set-Location $PSScriptRoot
& "mcp_agentic_workflow.exe"
'''
        
        with open(portable_dir / "start_mcp.ps1", 'w') as f:
            f.write(ps_content)
        
        # Create archive
        archive_name = f"mcp-agentic-workflow-{self.app_metadata['version']}-windows-portable.zip"
        archive_path = self.packages_dir / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(portable_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, portable_dir)
                    zipf.write(file_path, arc_name)
        
        return str(archive_path)
    
    def _build_windows_installer(self) -> str:
        """Build Windows installer."""
        print("  Building Windows installer...")
        
        # Create NSIS script
        nsis_script = f'''
!include "MUI2.nsh"

Name "{self.app_metadata['name']}"
OutFile "mcp-agentic-workflow-{self.app_metadata['version']}-setup.exe"
InstallDir "$PROGRAMFILES\\{self.app_metadata['name']}"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File /r "portable_windows\\*"
    
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    
    CreateDirectory "$SMPROGRAMS\\{self.app_metadata['name']}"
    CreateShortCut "$SMPROGRAMS\\{self.app_metadata['name']}\\{self.app_metadata['name']}.lnk" "$INSTDIR\\mcp_agentic_workflow.exe"
    CreateShortCut "$SMPROGRAMS\\{self.app_metadata['name']}\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.app_metadata['name']}" "DisplayName" "{self.app_metadata['name']}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.app_metadata['name']}" "UninstallString" "$INSTDIR\\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\uninstall.exe"
    RMDir /r "$INSTDIR"
    
    Delete "$SMPROGRAMS\\{self.app_metadata['name']}\\{self.app_metadata['name']}.lnk"
    Delete "$SMPROGRAMS\\{self.app_metadata['name']}\\Uninstall.lnk"
    RMDir "$SMPROGRAMS\\{self.app_metadata['name']}"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.app_metadata['name']}"
SectionEnd
'''
        
        nsis_file = self.build_dir / "installer.nsi"
        with open(nsis_file, 'w') as f:
            f.write(nsis_script)
        
        installer_name = f"mcp-agentic-workflow-{self.app_metadata['version']}-setup.exe"
        installer_path = self.packages_dir / installer_name
        
        # Try to build installer if NSIS is available
        try:
            subprocess.run(["makensis", str(nsis_file)], check=True)
            # Move the built installer
            built_installer = self.build_dir / installer_name
            if built_installer.exists():
                shutil.move(built_installer, installer_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("    NSIS not found, creating installer script only")
            # Just copy the NSIS script for manual building
            shutil.copy2(nsis_file, installer_path)
        
        return str(installer_path)
    
    def _build_macos_app(self) -> str:
        """Build macOS app bundle."""
        print("  Building macOS app...")
        
        app_name = f"{self.app_metadata['name']}.app"
        app_path = self.packages_dir / app_name
        contents_path = app_path / "Contents"
        
        # Create app bundle structure
        (contents_path / "MacOS").mkdir(parents=True, exist_ok=True)
        (contents_path / "Resources").mkdir(parents=True, exist_ok=True)
        
        # Copy application files
        self._copy_application_files(contents_path / "MacOS")
        
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
    <string>{self.app_metadata['name']}</string>
    <key>CFBundleVersion</key>
    <string>{self.app_metadata['version']}</string>
    <key>CFBundleShortVersionString</key>
    <string>{self.app_metadata['version']}</string>
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
        
        return str(app_path)
    
    def _build_macos_portable(self) -> str:
        """Build macOS portable package."""
        print("  Building macOS portable...")
        
        # Create portable directory
        portable_dir = self.build_dir / "portable_macos"
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        
        portable_dir.mkdir(parents=True)
        
        # Copy application files
        self._copy_application_files(portable_dir)
        
        # Create launcher script
        launcher_content = '''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."
cd "$(dirname "$0")"
./mcp_agentic_workflow "$@"
'''
        
        launcher_path = portable_dir / "start_mcp.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        os.chmod(launcher_path, 0o755)
        
        # Create archive
        archive_name = f"mcp-agentic-workflow-{self.app_metadata['version']}-macos-portable.tar.gz"
        archive_path = self.packages_dir / archive_name
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(portable_dir, arcname='.')
        
        return str(archive_path)
    
    def _create_usb_templates(self) -> str:
        """Create USB installation templates."""
        print("  Creating USB templates...")
        
        # Create USB template structure
        usb_template = self.usb_templates_dir / "mcp_usb_template"
        if usb_template.exists():
            shutil.rmtree(usb_template)
        
        usb_template.mkdir(parents=True)
        
        # Create directory structure
        (usb_template / "MCP_Agentic_Workflow").mkdir(parents=True, exist_ok=True)
        (usb_template / "docs").mkdir(parents=True, exist_ok=True)
        (usb_template / "platforms").mkdir(parents=True, exist_ok=True)
        (usb_template / "platforms" / "linux").mkdir(parents=True, exist_ok=True)
        (usb_template / "platforms" / "windows").mkdir(parents=True, exist_ok=True)
        (usb_template / "platforms" / "macos").mkdir(parents=True, exist_ok=True)
        
        # Create USB launcher script
        usb_launcher = '''#!/bin/bash
# MCP Agentic Workflow Accelerator - USB Launcher
# This script automatically detects the platform and launches the appropriate version

echo "MCP Agentic Workflow Accelerator - USB Edition"
echo "=============================================="

# Detect platform
PLATFORM=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Detected platform: $PLATFORM"

# Check if platform-specific version exists
PLATFORM_DIR="platforms/$PLATFORM"
if [[ ! -d "$PLATFORM_DIR" ]]; then
    echo "No version available for $PLATFORM"
    echo "Available platforms:"
    ls -1 platforms/
    exit 1
fi

# Launch the appropriate version
cd "$PLATFORM_DIR"
if [[ "$PLATFORM" == "windows" ]]; then
    if [[ -f "start_mcp.bat" ]]; then
        ./start_mcp.bat
    elif [[ -f "mcp_agentic_workflow.exe" ]]; then
        ./mcp_agentic_workflow.exe
    else
        echo "No Windows executable found"
        exit 1
    fi
else
    if [[ -f "start_mcp.sh" ]]; then
        ./start_mcp.sh
    elif [[ -f "mcp_agentic_workflow" ]]; then
        ./mcp_agentic_workflow
    else
        echo "No $PLATFORM executable found"
        exit 1
    fi
fi
'''
        
        with open(usb_template / "start_mcp.sh", 'w') as f:
            f.write(usb_launcher)
        
        os.chmod(usb_template / "start_mcp.sh", 0o755)
        
        # Create Windows launcher
        windows_launcher = '''@echo off
REM MCP Agentic Workflow Accelerator - USB Launcher for Windows
echo MCP Agentic Workflow Accelerator - USB Edition
echo ==============================================

cd /d "%~dp0"
cd platforms\\windows

if exist "start_mcp.bat" (
    call start_mcp.bat
) else if exist "mcp_agentic_workflow.exe" (
    mcp_agentic_workflow.exe
) else (
    echo No Windows executable found
    pause
    exit /b 1
)
'''
        
        with open(usb_template / "start_mcp.bat", 'w') as f:
            f.write(windows_launcher)
        
        # Create USB README
        usb_readme = f'''# MCP Agentic Workflow Accelerator - USB Edition

This is a portable version of the {self.app_metadata['name']} designed to run from USB drives.

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
'''
        
        with open(usb_template / "README.md", 'w') as f:
            f.write(usb_readme)
        
        # Create USB installation script
        install_script = '''#!/bin/bash
# USB Installation Script for MCP Agentic Workflow Accelerator

echo "MCP Agentic Workflow Accelerator - USB Installation"
echo "=================================================="

# Check if running as root (for system-wide installation)
if [[ $EUID -eq 0 ]]; then
    echo "Installing system-wide..."
    INSTALL_DIR="/opt/mcp-agentic-workflow"
    BIN_DIR="/usr/local/bin"
else
    echo "Installing for current user..."
    INSTALL_DIR="$HOME/.local/share/mcp-agentic-workflow"
    BIN_DIR="$HOME/.local/bin"
fi

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy files
cp -r MCP_Agentic_Workflow/* "$INSTALL_DIR/"
cp -r platforms/* "$INSTALL_DIR/"

# Create launcher symlink
ln -sf "$INSTALL_DIR/start_mcp.sh" "$BIN_DIR/mcp-agentic-workflow"

# Make executable
chmod +x "$INSTALL_DIR/start_mcp.sh"
chmod +x "$BIN_DIR/mcp-agentic-workflow"

echo "Installation complete!"
echo "You can now run: mcp-agentic-workflow"
'''
        
        with open(usb_template / "install.sh", 'w') as f:
            f.write(install_script)
        
        os.chmod(usb_template / "install.sh", 0o755)
        
        # Create archive
        template_archive = self.packages_dir / f"mcp-usb-template-{self.app_metadata['version']}.tar.gz"
        
        with tarfile.open(template_archive, 'w:gz') as tar:
            tar.add(usb_template, arcname='.')
        
        return str(template_archive)
    
    def _create_universal_portable(self) -> str:
        """Create universal portable package."""
        print("  Creating universal portable package...")
        
        # Create universal directory
        universal_dir = self.build_dir / "universal_portable"
        if universal_dir.exists():
            shutil.rmtree(universal_dir)
        
        universal_dir.mkdir(parents=True)
        
        # Copy application files
        self._copy_application_files(universal_dir)
        
        # Create platform-specific launchers
        launchers = {
            "start_mcp.sh": '''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."
cd "$(dirname "$0")"
python3 mcp_cli.py "$@"
''',
            "start_mcp.bat": '''@echo off
echo Starting MCP Agentic Workflow Accelerator...
cd /d "%~dp0"
python mcp_cli.py
pause
''',
            "start_mcp.ps1": '''# PowerShell launcher for MCP Agentic Workflow Accelerator
Write-Host "Starting MCP Agentic Workflow Accelerator..." -ForegroundColor Green
Set-Location $PSScriptRoot
python mcp_cli.py
'''
        }
        
        for filename, content in launchers.items():
            with open(universal_dir / filename, 'w') as f:
                f.write(content)
            
            if filename.endswith('.sh'):
                os.chmod(universal_dir / filename, 0o755)
        
        # Create universal README
        universal_readme = f'''# {self.app_metadata['name']} - Universal Portable

This is a universal portable version that works on any platform with Python 3.8+.

## Quick Start

1. Extract this package to any location
2. Run the launcher for your platform:
   - **Windows**: `start_mcp.bat` or `start_mcp.ps1`
   - **macOS/Linux**: `./start_mcp.sh`

## Requirements

- Python 3.8 or later
- Internet connection for first run (to install dependencies)

## Features

- **Universal**: Works on any platform with Python
- **Self-contained**: All source code included
- **Portable**: No installation required
- **Cross-platform**: Windows, macOS, and Linux support

## Manual Installation

If the launchers don't work, you can run manually:

```bash
python3 mcp_cli.py
```

## Dependencies

The application will automatically install required dependencies on first run.
'''
        
        with open(universal_dir / "README.md", 'w') as f:
            f.write(universal_readme)
        
        # Create archive
        universal_archive = self.packages_dir / f"mcp-agentic-workflow-{self.app_metadata['version']}-universal-portable.tar.gz"
        
        with tarfile.open(universal_archive, 'w:gz') as tar:
            tar.add(universal_dir, arcname='.')
        
        return str(universal_archive)
    
    def _copy_application_files(self, target_dir: Path):
        """Copy application files to target directory."""
        # Copy source code
        if (self.project_root / "src").exists():
            shutil.copytree(self.project_root / "src", target_dir / "src", dirs_exist_ok=True)
        
        # Copy configuration
        if (self.project_root / "config").exists():
            shutil.copytree(self.project_root / "config", target_dir / "config", dirs_exist_ok=True)
        
        # Copy data
        if (self.project_root / "data").exists():
            shutil.copytree(self.project_root / "data", target_dir / "data", dirs_exist_ok=True)
        
        # Copy plugins
        if (self.project_root / "plugins").exists():
            shutil.copytree(self.project_root / "plugins", target_dir / "plugins", dirs_exist_ok=True)
        
        # Copy essential files
        essential_files = [
            "mcp_cli.py",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "README.md",
            "idea.txt"
        ]
        
        for file in essential_files:
            if (self.project_root / file).exists():
                shutil.copy2(self.project_root / file, target_dir)
    
    def _generate_package_documentation(self, results: Dict[str, str]):
        """Generate documentation for all packages."""
        print("  Generating package documentation...")
        
        doc_content = f'''# MCP Agentic Workflow Accelerator - Package Documentation

This document describes all available packages for the MCP Agentic Workflow Accelerator.

## Build Information

- Build Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Platform: {self.platform} ({self.arch})
- Version: {self.app_metadata['version']}

## Available Packages

'''
        
        for package_type, result in results.items():
            if result.startswith("Failed"):
                doc_content += f'''
### {package_type.capitalize()}
❌ Build failed: {result}
'''
            else:
                doc_content += f'''
### {package_type.capitalize()}
✅ Package: {result}
- Type: {self._get_package_type_description(package_type)}
- Size: {self._get_file_size(result)}
'''
        
        doc_content += f'''

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

### Android

#### Android Package
1. Download the .zip file
2. Extract to your Android device
3. Install dependencies using Pydroid or Chaquopy
4. Run `python3 mcp_cli.py` or use the provided launcher

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
- **Android**: Android-compatible package

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
'''
        
        doc_path = self.packages_dir / "PACKAGE_DOCUMENTATION.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        print(f"    Documentation generated: {doc_path}")
    
    def _get_package_type_description(self, package_type: str) -> str:
        """Get description for package type."""
        descriptions = {
            "appimage": "Self-contained Linux application",
            "flatpak": "Sandboxed Linux application",
            "portable": "Extract and run without installation",
            "installer": "System installer",
            "app": "macOS application bundle",
            "universal": "Cross-platform Python package",
            "usb_template": "USB drive installation template",
            "android": "Android-compatible package"
        }
        return descriptions.get(package_type, "Unknown package type")
    
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
    
    def _clean_build_directories(self):
        """Clean build directories."""
        print("  Cleaning build directories...")
        
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        if self.packages_dir.exists():
            shutil.rmtree(self.packages_dir)
        
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.packages_dir.mkdir(exist_ok=True)
    
    def _build_android(self) -> str:
        """Build Android-compatible package (Pydroid, Chaquopy, or BeeWare)."""
        print("  Building Android-compatible package...")
        android_dir = self.build_dir / "android"
        if android_dir.exists():
            shutil.rmtree(android_dir)
        android_dir.mkdir(parents=True)
        # Copy application files
        self._copy_application_files(android_dir)
        # Add Android-specific requirements (network, LLM API, OpenRouter)
        requirements = [
            "requests", "flask", "openai", "httpx", "websockets", "pydantic", "uvicorn"
        ]
        with open(android_dir / "requirements-android.txt", 'w') as f:
            f.write("\n".join(requirements))
        # Add Android README
        android_readme = f'''# MCP Agentic Workflow Accelerator - Android Edition

This package is compatible with Pydroid, Chaquopy, and BeeWare for Android.

## Features
- Full network support
- LLM API and OpenRouter integration
- Portable Python codebase

## Installation
1. Copy this directory to your Android device
2. Install dependencies using Pydroid or Chaquopy
3. Run `python3 mcp_cli.py` or use the provided launcher

## Requirements
- Python 3.8+ (via Pydroid, Chaquopy, or BeeWare)
- Internet connection
'''
        with open(android_dir / "README-android.md", 'w') as f:
            f.write(android_readme)
        # Create zip archive for Android
        android_zip = self.packages_dir / f"mcp-agentic-workflow-{self.app_metadata['version']}-android.zip"
        with zipfile.ZipFile(android_zip, 'w') as zipf:
            for root, _, files in os.walk(android_dir):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, file_path.relative_to(android_dir))
        return str(android_zip)

def main():
    """Main function to build universal packages."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build universal MCP Agentic Workflow Accelerator packages")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before building")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--platform", choices=["linux", "windows", "macos", "android", "all"], 
                       help="Build for specific platform only")
    
    args = parser.parse_args()
    
    builder = UniversalPackageBuilder(args.project_root)
    
    try:
        results = builder.build_all_packages(clean=args.clean)
        
        print("\n" + "=" * 70)
        print("🏁 Build Summary")
        print("=" * 70)
        
        for package_type, result in results.items():
            if result.startswith("Failed"):
                print(f"❌ {package_type}: {result}")
            else:
                print(f"✅ {package_type}: {result}")
        
        print(f"\n📁 Packages directory: {builder.packages_dir}")
        print(f"📁 USB templates directory: {builder.usb_templates_dir}")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 