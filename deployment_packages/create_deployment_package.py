#!/usr/bin/env python3
"""
Deployment Package Creator for MCP Agentic Workflow Accelerator
Creates a complete deployment package with archive, unwrap script, and README in a single folder.
"""

import os
import shutil
import zipfile
import tarfile
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import argparse

class DeploymentPackageCreator:
    """Creates complete deployment packages with archive, unwrap script, and README."""
    
    def __init__(self, project_root: str = None):
        """Initialize the deployment package creator."""
        if project_root is None:
            self.project_root = os.getcwd()
        else:
            self.project_root = project_root
        
        self.essential_files = [
            # Core source code
            'src/mcp/__init__.py',
            'src/mcp/memory.py',
            'src/mcp/advanced_memory.py',
            'src/mcp/unified_memory.py',
            'src/mcp/task_manager.py',
            'src/mcp/context_manager.py',
            'src/mcp/reminder_engine.py',
            'src/mcp/workflow.py',
            'src/mcp/project_manager.py',
            'src/mcp/cli.py',
            'src/mcp/database_manager.py',
            'src/mcp/performance_monitor.py',
            'src/mcp/regex_search.py',
            'src/mcp/rag_system.py',
            'src/mcp/server.py',
            
            # Entry points
            'mcp_cli.py',
            'setup.py',
            'pyproject.toml',
            'requirements.txt',
            
            # Documentation
            'README.md',
            'idea.txt',
            'PROJECT_STATUS_FINAL.md',
            
            # Configuration
            'config/config.cfg',
            
            # Database (essential)
            'data/unified_memory.db',
            
            # Tests
            'test_system.py',
            'tests/__init__.py',
            'tests/integration/__init__.py',
            
            # Git ignore
            '.gitignore'
        ]
        
        self.optional_files = [
            # Additional documentation
            'docs/README.md',
            'docs/research/',
            
            # Examples
            'examples/',
            
            # Scripts
            'scripts/',
            
            # Templates
            'mcp/templates/'
        ]
        
        self.excluded_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '.git',
            '.gitignore',
            '*.log',
            '*.tmp',
            '*.bak',
            '*.backup',
            '*.db.backup',
            'backup_databases',
            'env',
            '.venv',
            'venv',
            'node_modules',
            '.DS_Store',
            'Thumbs.db'
        ]
    
    def create_deployment_package(self, package_name: str = None, include_optional: bool = False, 
                                 format: str = 'zip', optimize_db: bool = True) -> str:
        """Create a complete deployment package."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if package_name is None:
            package_name = f"mcp_deployment_package_{timestamp}"
        
        print(f"📦 Creating deployment package: {package_name}")
        print("=" * 60)
        
        # Create package directory
        package_dir = package_name
        os.makedirs(package_dir, exist_ok=True)
        
        try:
            # Create temporary directory for archive contents
            temp_dir = f"temp_archive_{timestamp}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy essential files
            print("\n📁 Copying essential files...")
            copied_files = self._copy_essential_files(temp_dir)
            
            # Copy optional files if requested
            if include_optional:
                print("\n📁 Copying optional files...")
                optional_copied = self._copy_optional_files(temp_dir)
                copied_files.extend(optional_copied)
            
            # Optimize database if requested
            if optimize_db:
                print("\n🗄️  Optimizing database...")
                self._optimize_database(temp_dir)
            
            # Create deployment script
            print("\n🔧 Creating deployment script...")
            self._create_deployment_script(temp_dir)
            
            # Create archive
            print(f"\n📦 Creating {format.upper()} archive...")
            archive_name = f"mcp_system.{format}"
            if format.lower() == 'zip':
                archive_path = os.path.join(package_dir, archive_name)
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arc_name)
            elif format.lower() == 'tar':
                archive_path = os.path.join(package_dir, f"{archive_name}.gz")
                with tarfile.open(archive_path, 'w:gz') as tarf:
                    tarf.add(temp_dir, arcname='.')
            
            # Create unwrap script
            print("\n📝 Creating unwrap script...")
            self._create_unwrap_script(package_dir, archive_name)
            
            # Create package README
            print("\n📖 Creating package README...")
            self._create_package_readme(package_dir, package_name, archive_name, len(copied_files))
            
            # Create quick start script
            print("\n🚀 Creating quick start script...")
            self._create_quick_start_script(package_dir)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Get package size
            package_size = self._get_directory_size(package_dir)
            
            print(f"\n✅ Deployment package created: {package_dir}/")
            print(f"📊 Package size: {package_size}")
            print(f"📁 Files in archive: {len(copied_files)}")
            print(f"📦 Archive: {archive_name}")
            
            return package_dir
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(package_dir):
                shutil.rmtree(package_dir)
            raise e
    
    def _copy_essential_files(self, temp_dir: str) -> list:
        """Copy essential files to temporary directory."""
        copied_files = []
        
        for file_path in self.essential_files:
            source_path = os.path.join(self.project_root, file_path)
            
            if os.path.exists(source_path):
                # Create target directory if needed
                target_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy file
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, target_path)
                    copied_files.append(file_path)
                    print(f"  ✅ {file_path}")
                elif os.path.isdir(source_path):
                    shutil.copytree(source_path, target_path, ignore=shutil.ignore_patterns(*self.excluded_patterns))
                    copied_files.append(file_path)
                    print(f"  ✅ {file_path}/")
            else:
                print(f"  ⚠️  {file_path} (not found)")
        
        return copied_files
    
    def _copy_optional_files(self, temp_dir: str) -> list:
        """Copy optional files to temporary directory."""
        copied_files = []
        
        for file_path in self.optional_files:
            source_path = os.path.join(self.project_root, file_path)
            
            if os.path.exists(source_path):
                target_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, target_path)
                    copied_files.append(file_path)
                    print(f"  ✅ {file_path}")
                elif os.path.isdir(source_path):
                    shutil.copytree(source_path, target_path, ignore=shutil.ignore_patterns(*self.excluded_patterns))
                    copied_files.append(file_path)
                    print(f"  ✅ {file_path}/")
        
        return copied_files
    
    def _optimize_database(self, temp_dir: str):
        """Optimize the database for portability."""
        db_path = os.path.join(temp_dir, 'data', 'unified_memory.db')
        
        if os.path.exists(db_path):
            try:
                # Connect to database
                conn = sqlite3.connect(db_path)
                
                # Optimize database
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.execute("REINDEX")
                
                # Get database info
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memories")
                memory_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM tasks")
                task_count = cursor.fetchone()[0]
                
                conn.close()
                
                print(f"  ✅ Database optimized: {table_count} tables, {memory_count} memories, {task_count} tasks")
                
            except Exception as e:
                print(f"  ⚠️  Database optimization failed: {e}")
    
    def _create_deployment_script(self, temp_dir: str):
        """Create a deployment script for easy setup."""
        
        # Create setup script
        setup_script = """#!/bin/bash
# MCP Agentic Workflow Accelerator - Portable Setup Script

echo "🚀 Setting up MCP Agentic Workflow Accelerator..."
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "📦 Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Test the system
echo "🧪 Testing system..."
python3 test_system.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Available commands:"
echo "  python3 mcp_cli.py --help                    # Show all commands"
echo "  python3 mcp_cli.py create-task --help        # Create a new task"
echo "  python3 mcp_cli.py export-context --help     # Export context for LLM"
echo "  python3 mcp_cli.py statistics                # Show system statistics"
echo ""
echo "📖 Read README.md for detailed usage instructions"
"""
        
        setup_path = os.path.join(temp_dir, 'setup.sh')
        with open(setup_path, 'w') as f:
            f.write(setup_script)
        
        # Make executable
        os.chmod(setup_path, 0o755)
        
        # Create Windows batch file
        setup_bat = """@echo off
REM MCP Agentic Workflow Accelerator - Portable Setup Script (Windows)

echo 🚀 Setting up MCP Agentic Workflow Accelerator...
echo ==================================================

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.7+ and try again.
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Test the system
echo 🧪 Testing system...
python test_system.py

echo.
echo ✅ Setup complete!
echo.
echo 📋 Available commands:
echo   python mcp_cli.py --help                    # Show all commands
echo   python mcp_cli.py create-task --help        # Create a new task
echo   python mcp_cli.py export-context --help     # Export context for LLM
echo   python mcp_cli.py statistics                # Show system statistics
echo.
echo 📖 Read README.md for detailed usage instructions
pause
"""
        
        setup_bat_path = os.path.join(temp_dir, 'setup.bat')
        with open(setup_bat_path, 'w') as f:
            f.write(setup_bat)
        
        print("  ✅ Created setup.sh (Linux/Mac)")
        print("  ✅ Created setup.bat (Windows)")
    
    def _create_unwrap_script(self, package_dir: str, archive_name: str):
        """Create an unwrap script for easy deployment."""
        
        unwrap_script = f"""#!/bin/bash
# MCP Agentic Workflow Accelerator - Unwrap Script
# This script extracts and sets up the MCP system from the archive

set -e  # Exit on any error

echo "📦 MCP Agentic Workflow Accelerator - Unwrap Script"
echo "=================================================="

# Check if archive exists
if [ ! -f "{archive_name}" ]; then
    echo "❌ Archive not found: {archive_name}"
    echo "   Make sure you're running this script from the deployment package directory."
    exit 1
fi

# Get target directory name
if [ -n "$1" ]; then
    TARGET_DIR="$1"
else
    TARGET_DIR="mcp_workflow_$(date +%Y%m%d_%H%M%S)"
fi

echo "📁 Target directory: $TARGET_DIR"

# Create target directory
mkdir -p "$TARGET_DIR"

# Extract archive
echo "📦 Extracting archive..."
if [[ "{archive_name}" == *.zip ]]; then
    unzip -q "{archive_name}" -d "$TARGET_DIR"
elif [[ "{archive_name}" == *.tar.gz ]]; then
    tar -xzf "{archive_name}" -C "$TARGET_DIR"
else
    echo "❌ Unsupported archive format: {archive_name}"
    exit 1
fi

echo "✅ Archive extracted successfully"

# Navigate to target directory
cd "$TARGET_DIR"

# Run setup
echo "🔧 Running setup..."
if [ -f "setup.sh" ]; then
    chmod +x setup.sh
    ./setup.sh
elif [ -f "setup.bat" ]; then
    echo "⚠️  Windows setup script found. Please run 'setup.bat' manually on Windows."
else
    echo "⚠️  No setup script found. Installing dependencies manually..."
    pip3 install -r requirements.txt
    python3 test_system.py
fi

echo ""
echo "🎉 MCP Agentic Workflow Accelerator is ready!"
echo "📁 Location: $TARGET_DIR"
echo ""
echo "📋 Next steps:"
echo "  cd $TARGET_DIR"
echo "  python3 mcp_cli.py --help"
echo "  python3 test_system.py"
echo ""
echo "📖 Read README.md for detailed usage instructions"
"""
        
        unwrap_path = os.path.join(package_dir, 'unwrap.sh')
        with open(unwrap_path, 'w') as f:
            f.write(unwrap_script)
        
        # Make executable
        os.chmod(unwrap_path, 0o755)
        
        # Create Windows unwrap script
        unwrap_bat = f"""@echo off
REM MCP Agentic Workflow Accelerator - Unwrap Script (Windows)
REM This script extracts and sets up the MCP system from the archive

echo 📦 MCP Agentic Workflow Accelerator - Unwrap Script
echo ==================================================

REM Check if archive exists
if not exist "{archive_name}" (
    echo ❌ Archive not found: {archive_name}
    echo    Make sure you're running this script from the deployment package directory.
    pause
    exit /b 1
)

REM Get target directory name
if not "%1"=="" (
    set TARGET_DIR=%1
) else (
    set TARGET_DIR=mcp_workflow_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
    set TARGET_DIR=%TARGET_DIR: =0%
)

echo 📁 Target directory: %TARGET_DIR%

REM Create target directory
mkdir "%TARGET_DIR%"

REM Extract archive
echo 📦 Extracting archive...
if "{archive_name}" == *.zip (
    powershell -command "Expand-Archive -Path '{archive_name}' -DestinationPath '%TARGET_DIR%' -Force"
) else if "{archive_name}" == *.tar.gz (
    tar -xzf "{archive_name}" -C "%TARGET_DIR%"
) else (
    echo ❌ Unsupported archive format: {archive_name}
    pause
    exit /b 1
)

echo ✅ Archive extracted successfully

REM Navigate to target directory
cd "%TARGET_DIR%"

REM Run setup
echo 🔧 Running setup...
if exist "setup.bat" (
    setup.bat
) else if exist "setup.sh" (
    echo ⚠️  Linux setup script found. Please run './setup.sh' manually on Linux/Mac.
) else (
    echo ⚠️  No setup script found. Installing dependencies manually...
    pip install -r requirements.txt
    python test_system.py
)

echo.
echo 🎉 MCP Agentic Workflow Accelerator is ready!
echo 📁 Location: %TARGET_DIR%
echo.
echo 📋 Next steps:
echo   cd %TARGET_DIR%
echo   python mcp_cli.py --help
echo   python test_system.py
echo.
echo 📖 Read README.md for detailed usage instructions
pause
"""
        
        unwrap_bat_path = os.path.join(package_dir, 'unwrap.bat')
        with open(unwrap_bat_path, 'w') as f:
            f.write(unwrap_bat)
        
        print("  ✅ Created unwrap.sh (Linux/Mac)")
        print("  ✅ Created unwrap.bat (Windows)")
    
    def _create_package_readme(self, package_dir: str, package_name: str, archive_name: str, file_count: int):
        """Create a README for the deployment package."""
        
        readme_content = f"""# MCP Agentic Workflow Accelerator - Deployment Package

## 📦 Package Contents

This deployment package contains everything needed to run the MCP Agentic Workflow Accelerator:

- **{archive_name}**: The complete MCP system archive
- **unwrap.sh**: Linux/Mac deployment script
- **unwrap.bat**: Windows deployment script
- **README.md**: This file
- **quick_start.sh**: Quick start script

## 🚀 Quick Deployment

### Linux/Mac
```bash
chmod +x unwrap.sh
./unwrap.sh [target_directory]
```

### Windows
```cmd
unwrap.bat [target_directory]
```

### Manual Deployment
1. Extract `{archive_name}` to your desired directory
2. Navigate to the extracted directory
3. Run `./setup.sh` (Linux/Mac) or `setup.bat` (Windows)

## 📋 System Requirements

- **Python**: 3.7 or higher
- **Dependencies**: Automatically installed during setup
- **Storage**: ~100 MB for full deployment
- **Memory**: ~50 MB runtime

## 🎯 What's Included

The MCP Agentic Workflow Accelerator provides:

- **Memory Management**: Add, search, and manage memories with types, priorities, and tags
- **Task Management**: Create hierarchical tasks with dependencies and progress tracking
- **Context Export**: Generate minimal context packs for LLM consumption
- **RAG System**: Intelligent retrieval and search capabilities
- **Workflow Management**: Guided project phases from research to deployment
- **Performance Monitoring**: System health and optimization tools

## 📊 Package Information

- **Package Name**: {package_name}
- **Archive**: {archive_name}
- **Files in Archive**: {file_count}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔧 Available Commands

After deployment, you'll have access to 40+ CLI commands:

```bash
# Show all commands
python mcp_cli.py --help

# Add a memory
python mcp_cli.py add-memory --text "My memory" --type "general" --priority 0.8

# Create a task
python mcp_cli.py create-task --title "My Task" --description "Task description" --priority 5

# List tasks
python mcp_cli.py list-tasks --tree

# Search memories
python mcp_cli.py search-memories --query "memory"

# Export context for LLM
python mcp_cli.py export-context --types tasks,memories --max-tokens 500
```

## 🆘 Troubleshooting

### Common Issues

1. **"No module named 'mcp'"**
   - Ensure you're in the correct directory
   - Run `pip install -r requirements.txt`

2. **"Database is locked"**
   - This is a non-critical warning
   - The system continues to work normally

3. **"Permission denied" on scripts**
   - Run `chmod +x unwrap.sh` to make executable

4. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

### Verification

After deployment, verify the system works:

```bash
# Run comprehensive test
python test_system.py

# Test basic functionality
python mcp_cli.py add-memory --text "Test" --type "test"
python mcp_cli.py search-memories --query "Test"
```

## 📖 More Information

- **Project Vision**: See `idea.txt` in the deployed system
- **Full Documentation**: See `README.md` in the deployed system
- **Status Report**: See `PROJECT_STATUS_FINAL.md` in the deployed system

---

**Ready to accelerate your agentic workflows! 🚀**
"""
        
        readme_path = os.path.join(package_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print("  ✅ Created package README.md")
    
    def _create_quick_start_script(self, package_dir: str):
        """Create a quick start script."""
        
        quick_start = """#!/bin/bash
# MCP Agentic Workflow Accelerator - Quick Start

echo "🚀 MCP Agentic Workflow Accelerator - Quick Start"
echo "=================================================="

# Check if unwrap script exists
if [ -f "unwrap.sh" ]; then
    echo "📦 Found unwrap script. Running deployment..."
    chmod +x unwrap.sh
    ./unwrap.sh
else
    echo "❌ Unwrap script not found."
    echo "   Make sure you're in the deployment package directory."
    exit 1
fi
"""
        
        quick_start_path = os.path.join(package_dir, 'quick_start.sh')
        with open(quick_start_path, 'w') as f:
            f.write(quick_start)
        
        # Make executable
        os.chmod(quick_start_path, 0o755)
        
        print("  ✅ Created quick_start.sh")
    
    def _get_directory_size(self, directory: str) -> str:
        """Get human-readable directory size."""
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        
        return f"{total_size:.1f} TB"
    
    def create_minimal_package(self, package_name: str = None) -> str:
        """Create a minimal deployment package."""
        
        minimal_files = [
            # Core source code (only the most essential)
            'src/mcp/__init__.py',
            'src/mcp/memory.py',
            'src/mcp/unified_memory.py',
            'src/mcp/task_manager.py',
            'src/mcp/cli.py',
            
            # Entry points
            'mcp_cli.py',
            'requirements.txt',
            
            # Documentation
            'README.md',
            'idea.txt',
            
            # Database (essential)
            'data/unified_memory.db',
            
            # Test
            'test_system.py'
        ]
        
        # Temporarily replace essential files with minimal files
        original_essential = self.essential_files
        self.essential_files = minimal_files
        
        try:
            return self.create_deployment_package(package_name, include_optional=False, optimize_db=True)
        finally:
            # Restore original essential files
            self.essential_files = original_essential

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Create deployment package of MCP Agentic Workflow Accelerator')
    parser.add_argument('--output', '-o', help='Output package name (without extension)')
    parser.add_argument('--format', '-f', choices=['zip', 'tar'], default='zip', help='Archive format')
    parser.add_argument('--include-optional', '-i', action='store_true', help='Include optional files')
    parser.add_argument('--minimal', '-m', action='store_true', help='Create minimal package')
    parser.add_argument('--no-optimize', action='store_true', help='Skip database optimization')
    
    args = parser.parse_args()
    
    creator = DeploymentPackageCreator()
    
    try:
        if args.minimal:
            package_path = creator.create_minimal_package(args.output)
        else:
            package_path = creator.create_deployment_package(
                package_name=args.output,
                include_optional=args.include_optional,
                format=args.format,
                optimize_db=not args.no_optimize
            )
        
        print(f"\n🎉 Deployment package created successfully: {package_path}/")
        print(f"📦 Ready for distribution!")
        
    except Exception as e:
        print(f"❌ Failed to create deployment package: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 