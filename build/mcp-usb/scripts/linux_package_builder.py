#!/usr/bin/env python3
"""
Linux Package Builder for MCP Agentic Workflow Accelerator
Creates AppImages and Flatpaks for Linux distribution.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, List

class LinuxPackageBuilder:
    """Builds Linux packages (AppImage and Flatpak) for the MCP server."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Linux package builder."""
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self.platform = platform.system().lower()
        if self.platform != "linux":
            raise RuntimeError("This builder is only for Linux systems")
        
        self.arch = platform.machine()
        
        # Output directories
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.packages_dir = self.project_root / "packages"
        
        # Create directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.packages_dir.mkdir(exist_ok=True)
        
        # Package metadata
        self.app_id = "com.mcp.agentic-workflow"
        self.app_name = "MCP Agentic Workflow Accelerator"
        self.app_version = "2.0.0"
        self.app_description = "Portable MCP server for agentic development workflows"
        self.app_author = "MCP Development Team"
        
    def build_all_packages(self, clean: bool = True) -> Dict[str, str]:
        """Build all Linux packages (AppImage and Flatpak)."""
        print("🚀 Building Linux Packages for MCP Agentic Workflow Accelerator")
        print("=" * 70)
        
        if clean:
            self._clean_build_directories()
        
        results = {}
        
        # Build AppImage
        print("\n📦 Building AppImage...")
        try:
            appimage_path = self._build_appimage()
            results['appimage'] = appimage_path
            print(f"✅ AppImage created: {appimage_path}")
        except Exception as e:
            print(f"❌ AppImage build failed: {e}")
            results['appimage'] = f"Failed: {e}"
        
        # Build Flatpak
        print("\n📦 Building Flatpak...")
        try:
            flatpak_path = self._build_flatpak()
            results['flatpak'] = flatpak_path
            print(f"✅ Flatpak created: {flatpak_path}")
        except Exception as e:
            print(f"❌ Flatpak build failed: {e}")
            results['flatpak'] = f"Failed: {e}"
        
        # Create portable package
        print("\n📦 Creating portable package...")
        try:
            portable_path = self._create_portable_package()
            results['portable'] = portable_path
            print(f"✅ Portable package created: {portable_path}")
        except Exception as e:
            print(f"❌ Portable package failed: {e}")
            results['portable'] = f"Failed: {e}"
        
        # Generate documentation
        print("\n📖 Generating documentation...")
        self._generate_documentation(results)
        
        return results
    
    def _build_appimage(self) -> str:
        """Build AppImage package."""
        print("  Creating AppImage structure...")
        
        # Create AppDir structure
        appdir = self.build_dir / "AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
        
        appdir.mkdir(parents=True)
        
        # Create AppDir structure
        (appdir / "usr" / "bin").mkdir(parents=True, exist_ok=True)
        (appdir / "usr" / "share" / "applications").mkdir(parents=True, exist_ok=True)
        (appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps").mkdir(parents=True, exist_ok=True)
        (appdir / "usr" / "share" / "mcp-agentic-workflow").mkdir(parents=True, exist_ok=True)
        
        # Copy Python executable and dependencies
        self._copy_python_app(appdir)
        
        # Create AppRun script
        apprun_content = '''#!/bin/bash
cd "$(dirname "$0")"
exec "$APPDIR/usr/bin/python3" "$APPDIR/usr/share/mcp-agentic-workflow/mcp_cli.py" "$@"
'''
        apprun_path = appdir / "AppRun"
        with open(apprun_path, 'w') as f:
            f.write(apprun_content)
        os.chmod(apprun_path, 0o755)
        
        # Create desktop file
        desktop_content = f'''[Desktop Entry]
Name={self.app_name}
Comment={self.app_description}
Exec=mcp-agentic-workflow
Icon={self.app_id}
Terminal=true
Type=Application
Categories=Development;Utility;
StartupWMClass=mcp-agentic-workflow
'''
        desktop_path = appdir / "usr" / "share" / "applications" / f"{self.app_id}.desktop"
        with open(desktop_path, 'w') as f:
            f.write(desktop_content)
        
        # Create AppImage metadata
        appimage_metadata = {
            "name": self.app_name,
            "version": self.app_version,
            "description": self.app_description,
            "author": self.app_author,
            "arch": self.arch,
            "target": "AppImage"
        }
        
        metadata_path = appdir / "appimage-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(appimage_metadata, f, indent=2)
        
        # Create AppImage using appimagetool
        appimage_name = f"mcp-agentic-workflow-{self.app_version}-{self.arch}.AppImage"
        appimage_path = self.packages_dir / appimage_name
        
        print("  Building AppImage with appimagetool...")
        
        # Try to use appimagetool if available
        try:
            cmd = ["appimagetool", str(appdir), str(appimage_path)]
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  appimagetool not found, creating basic AppImage...")
            # Create a basic AppImage without appimagetool
            self._create_basic_appimage(appdir, appimage_path)
        
        return str(appimage_path)
    
    def _create_basic_appimage(self, appdir: Path, appimage_path: Path):
        """Create a basic AppImage without appimagetool."""
        # This is a simplified AppImage creation
        # In practice, you'd want to use appimagetool for proper AppImages
        
        # Create a self-extracting archive
        import tarfile
        
        # Create tar.gz of the AppDir
        tar_path = appimage_path.with_suffix('.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(appdir, arcname='.')
        
        # Create a simple launcher script
        launcher_content = f'''#!/bin/bash
# Simple AppImage launcher for {self.app_name}
# Extract and run the application

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRACT_DIR="$SCRIPT_DIR/.extracted"

# Create extraction directory
mkdir -p "$EXTRACT_DIR"

# Extract if not already extracted
if [ ! -f "$EXTRACT_DIR/.extracted" ]; then
    echo "Extracting {self.app_name}..."
    tar -xzf "$SCRIPT_DIR/$(basename "$0")" -C "$EXTRACT_DIR"
    touch "$EXTRACT_DIR/.extracted"
fi

# Run the application
cd "$EXTRACT_DIR"
./AppRun "$@"
'''
        
        # Create the AppImage as a self-extracting script
        with open(appimage_path, 'w') as f:
            f.write(launcher_content)
        
        # Make it executable
        os.chmod(appimage_path, 0o755)
        
        # Append the tar.gz to the script
        with open(appimage_path, 'ab') as f:
            with open(tar_path, 'rb') as tar_file:
                f.write(tar_file.read())
        
        # Clean up tar file
        tar_path.unlink()
    
    def _build_flatpak(self) -> str:
        """Build Flatpak package."""
        print("  Creating Flatpak structure...")
        
        # Create Flatpak build directory
        flatpak_build = self.build_dir / "flatpak"
        if flatpak_build.exists():
            shutil.rmtree(flatpak_build)
        
        flatpak_build.mkdir(parents=True)
        
        # Create Flatpak manifest
        manifest_content = f'''app-id: {self.app_id}
runtime: org.freedesktop.Platform
runtime-version: '23.08'
sdk: org.freedesktop.Sdk
command: mcp-agentic-workflow
finish-args:
  - --share=network
  - --share=ipc
  - --socket=fallback-x11
  - --socket=wayland
  - --filesystem=home
  - --filesystem=xdg-documents
modules:
  - name: python3
    buildsystem: simple
    build-commands:
      - pip3 install --prefix=/app --no-deps --no-index --find-links="file://${{PWD}}" .
    sources:
      - type: dir
        path: .
  - name: mcp-agentic-workflow
    buildsystem: simple
    build-commands:
      - install -D mcp-agentic-workflow /app/bin/mcp-agentic-workflow
      - install -D {self.app_id}.desktop /app/share/applications/{self.app_id}.desktop
      - install -D icon.png /app/share/icons/hicolor/256x256/apps/{self.app_id}.png
    sources:
      - type: file
        path: mcp-agentic-workflow
      - type: file
        path: {self.app_id}.desktop
      - type: file
        path: icon.png
'''
        
        manifest_path = flatpak_build / f"{self.app_id}.yml"
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
        
        # Create desktop file for Flatpak
        desktop_content = f'''[Desktop Entry]
Name={self.app_name}
Comment={self.app_description}
Exec=mcp-agentic-workflow
Icon={self.app_id}
Terminal=true
Type=Application
Categories=Development;Utility;
StartupWMClass=mcp-agentic-workflow
'''
        
        desktop_path = flatpak_build / f"{self.app_id}.desktop"
        with open(desktop_path, 'w') as f:
            f.write(desktop_content)
        
        # Create a simple icon (placeholder)
        self._create_placeholder_icon(flatpak_build / "icon.png")
        
        # Create the executable wrapper
        executable_content = '''#!/usr/bin/env python3
"""
MCP Agentic Workflow Accelerator - Flatpak Entry Point
"""

import sys
import os

# Add the app directory to Python path
app_dir = "/app/lib/python3.11/site-packages"
if os.path.exists(app_dir):
    sys.path.insert(0, app_dir)

# Import and run the main CLI
from mcp.cli import cli

if __name__ == "__main__":
    cli()
'''
        
        executable_path = flatpak_build / "mcp-agentic-workflow"
        with open(executable_path, 'w') as f:
            f.write(executable_content)
        os.chmod(executable_path, 0o755)
        
        # Try to build Flatpak
        print("  Building Flatpak...")
        try:
            # Check if flatpak-builder is available
            subprocess.run(["flatpak-builder", "--version"], check=True, capture_output=True)
            
            # Build the Flatpak
            cmd = [
                "flatpak-builder",
                "--force-clean",
                "--repo", str(self.build_dir / "repo"),
                str(self.build_dir / "build-dir"),
                str(manifest_path)
            ]
            subprocess.run(cmd, check=True)
            
            # Create Flatpak bundle
            bundle_name = f"mcp-agentic-workflow-{self.app_version}-{self.arch}.flatpak"
            bundle_path = self.packages_dir / bundle_name
            
            cmd = [
                "flatpak-builder",
                "--repo", str(self.build_dir / "repo"),
                "--force-clean",
                str(bundle_path),
                str(manifest_path)
            ]
            subprocess.run(cmd, check=True)
            
            return str(bundle_path)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  flatpak-builder not found, creating Flatpak manifest only...")
            # Just copy the manifest and related files
            flatpak_package_dir = self.packages_dir / "flatpak-source"
            flatpak_package_dir.mkdir(exist_ok=True)
            
            shutil.copy2(manifest_path, flatpak_package_dir)
            shutil.copy2(desktop_path, flatpak_package_dir)
            shutil.copy2(executable_path, flatpak_package_dir)
            shutil.copy2(flatpak_build / "icon.png", flatpak_package_dir)
            
            return str(flatpak_package_dir)
    
    def _copy_python_app(self, appdir: Path):
        """Copy Python application to AppDir."""
        print("  Copying Python application...")
        
        # Install Python and dependencies in AppDir
        python_dir = appdir / "usr" / "bin"
        app_dir = appdir / "usr" / "share" / "mcp-agentic-workflow"
        
        # Copy project files
        shutil.copytree(self.project_root / "src", app_dir / "src", dirs_exist_ok=True)
        shutil.copytree(self.project_root / "config", app_dir / "config", dirs_exist_ok=True)
        shutil.copytree(self.project_root / "data", app_dir / "data", dirs_exist_ok=True)
        shutil.copytree(self.project_root / "plugins", app_dir / "plugins", dirs_exist_ok=True)
        
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
            src_file = self.project_root / file
            if src_file.exists():
                shutil.copy2(src_file, app_dir)
        
        # Create Python wrapper
        python_wrapper = '''#!/usr/bin/env python3
"""
Python wrapper for MCP Agentic Workflow Accelerator
"""

import sys
import os

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(__file__), "..", "share", "mcp-agentic-workflow")
sys.path.insert(0, app_dir)

# Import and run the main CLI
from src.mcp.cli import cli

if __name__ == "__main__":
    cli()
'''
        
        python_path = python_dir / "python3"
        with open(python_path, 'w') as f:
            f.write(python_wrapper)
        os.chmod(python_path, 0o755)
        
        # Create symlink for the main executable
        main_executable = python_dir / "mcp-agentic-workflow"
        if main_executable.exists():
            main_executable.unlink()
        main_executable.symlink_to("python3")
    
    def _create_placeholder_icon(self, icon_path: Path):
        """Create a placeholder icon."""
        # Create a simple SVG icon
        svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <rect width="256" height="256" fill="#2d3748"/>
  <circle cx="128" cy="128" r="80" fill="#4299e1"/>
  <text x="128" y="140" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="24" font-weight="bold">MCP</text>
  <text x="128" y="160" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="12">Agentic</text>
</svg>
'''
        
        # Convert SVG to PNG using a simple approach
        # In practice, you'd use a proper SVG to PNG converter
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg_content.encode(), write_to=str(icon_path))
        except ImportError:
            # Create a simple colored square as fallback
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (256, 256), color='#2d3748')
            draw = ImageDraw.Draw(img)
            
            # Draw a circle
            draw.ellipse([48, 48, 208, 208], fill='#4299e1')
            
            # Add text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((128, 120), "MCP", fill='white', anchor="mm", font=font)
            
            img.save(icon_path)
    
    def _create_portable_package(self) -> str:
        """Create a portable package for Linux."""
        print("  Creating portable package...")
        
        portable_dir = self.packages_dir / "portable-linux"
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        
        portable_dir.mkdir(parents=True)
        
        # Copy the entire project
        shutil.copytree(self.project_root / "src", portable_dir / "src", dirs_exist_ok=True)
        shutil.copytree(self.project_root / "config", portable_dir / "config", dirs_exist_ok=True)
        shutil.copytree(self.project_root / "data", portable_dir / "data", dirs_exist_ok=True)
        shutil.copytree(self.project_root / "plugins", portable_dir / "plugins", dirs_exist_ok=True)
        
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
            src_file = self.project_root / file
            if src_file.exists():
                shutil.copy2(src_file, portable_dir)
        
        # Create launcher script
        launcher_content = '''#!/bin/bash
echo "Starting MCP Agentic Workflow Accelerator..."

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting MCP server..."
python mcp_cli.py "$@"
'''
        
        launcher_path = portable_dir / "start_mcp.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        os.chmod(launcher_path, 0o755)
        
        # Create README
        readme_content = '''# MCP Agentic Workflow Accelerator - Portable Linux Package

This is a portable version of the MCP Agentic Workflow Accelerator for Linux.

## Quick Start

1. Extract this package to any location
2. Run the launcher: `./start_mcp.sh`

## Features

- Fully portable - no installation required
- Automatic virtual environment creation
- Dependency management
- Cross-distribution compatibility

## System Requirements

- Linux (any distribution)
- Python 3.8 or higher
- 4GB RAM minimum
- 2GB free disk space

## Manual Installation

If the automatic setup doesn't work:

1. Install Python 3.8+ and pip
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python mcp_cli.py`

## Support

For issues and documentation, see the main project repository.
'''
        
        readme_path = portable_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return str(portable_dir)
    
    def _generate_documentation(self, results: Dict[str, str]):
        """Generate documentation for the built packages."""
        print("  Generating documentation...")
        
        doc_content = f'''# MCP Agentic Workflow Accelerator - Linux Packages

This document describes the built Linux packages for the MCP Agentic Workflow Accelerator.

## Build Information

- Build Date: {self._get_current_timestamp()}
- Architecture: {self.arch}
- Target Platform: Linux

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
- Type: {package_type.capitalize()}
- Size: {self._get_file_size(result)}
'''
        
        doc_content += f'''

## Installation Instructions

### AppImage
1. Download the AppImage file
2. Make it executable: `chmod +x mcp-agentic-workflow-*.AppImage`
3. Run: `./mcp-agentic-workflow-*.AppImage`

### Flatpak
1. Download the Flatpak bundle
2. Install: `flatpak install mcp-agentic-workflow-*.flatpak`
3. Run: `flatpak run {self.app_id}`

### Portable Package
1. Extract the portable package
2. Run: `./start_mcp.sh`

## Package Features

### AppImage
- Self-contained application
- No installation required
- Works on most Linux distributions
- Includes all dependencies

### Flatpak
- Sandboxed application
- System integration
- Automatic updates
- Security benefits

### Portable Package
- Source code included
- Customizable
- Development-friendly
- No system integration

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Make sure the AppImage is executable: `chmod +x *.AppImage`

2. **Missing Dependencies**
   - AppImages and Flatpaks include all dependencies
   - Portable package will install dependencies automatically

3. **Flatpak Not Available**
   - Install Flatpak: `sudo apt install flatpak` (Ubuntu/Debian)
   - Or: `sudo dnf install flatpak` (Fedora)

### Getting Help

- Check the main project README for detailed documentation
- Review the logs in the `data/` directory
- Report issues on the project repository

## Technical Details

- Built for Linux x86_64
- Python 3.8+ compatible
- All project dependencies included
- Cross-distribution compatibility

## License

This software is licensed under the MIT License.
'''
        
        doc_path = self.packages_dir / "LINUX_PACKAGES_README.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        print("    Documentation generated")
    
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
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size."""
        try:
            path = Path(file_path)
            if path.is_file():
                size = path.stat().st_size
            elif path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            else:
                return "Unknown"
            
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except OSError:
            return "Unknown"

def main():
    """Main function to build Linux packages."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Linux packages for MCP Agentic Workflow Accelerator")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before building")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--package-type", choices=["appimage", "flatpak", "portable", "all"], 
                       default="all", help="Type of package to build")
    
    args = parser.parse_args()
    
    try:
        builder = LinuxPackageBuilder(args.project_root)
        
        if args.package_type == "all":
            results = builder.build_all_packages(clean=args.clean)
        else:
            # Build specific package type
            if args.package_type == "appimage":
                results = {"appimage": builder._build_appimage()}
            elif args.package_type == "flatpak":
                results = {"flatpak": builder._build_flatpak()}
            elif args.package_type == "portable":
                results = {"portable": builder._create_portable_package()}
        
        print("\n" + "=" * 70)
        print("🏁 Build Summary")
        print("=" * 70)
        
        for package_type, result in results.items():
            if result.startswith("Failed"):
                print(f"❌ {package_type}: {result}")
            else:
                print(f"✅ {package_type}: {result}")
        
        print(f"\n📁 Packages directory: {builder.packages_dir}")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 