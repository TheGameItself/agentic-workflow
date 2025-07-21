#!/usr/bin/env python3
"""
Portable Archive Creator for MCP Agentic Workflow Accelerator
Creates a condensed, portable archive with only essential files for easy deployment.
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

class PortableArchiveCreator:
    """Creates portable archives of the MCP project."""
    
    def __init__(self, project_root: str = None):
        """Initialize the archive creator."""
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
    
    def create_portable_archive(self, output_name: str = None, include_optional: bool = False, 
                               format: str = 'zip', optimize_db: bool = True) -> str:
        """Create a portable archive of the project."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_name is None:
            output_name = f"mcp_agentic_workflow_portable_{timestamp}"
        
        print(f"📦 Creating portable archive: {output_name}")
        print("=" * 50)
        
        # Create temporary directory for archive contents
        temp_dir = f"temp_archive_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
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
            archive_path = self._create_archive(temp_dir, output_name, format)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            print(f"\n✅ Portable archive created: {archive_path}")
            print(f"📊 Archive size: {self._get_file_size(archive_path)}")
            print(f"📁 Files included: {len(copied_files)}")
            
            return archive_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
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
    
    def _create_archive(self, temp_dir: str, output_name: str, format: str) -> str:
        """Create the final archive."""
        
        if format.lower() == 'zip':
            archive_path = f"{output_name}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arc_name)
        
        elif format.lower() == 'tar':
            archive_path = f"{output_name}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tarf:
                tarf.add(temp_dir, arcname='.')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return archive_path
    
    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size."""
        size = os.path.getsize(file_path)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        
        return f"{size:.1f} TB"
    
    def create_minimal_archive(self, output_name: str = None) -> str:
        """Create a minimal archive with only the most essential files."""
        
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.create_portable_archive(output_name, include_optional=False, optimize_db=True)
        finally:
            # Restore original essential files
            self.essential_files = original_essential

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Create portable archive of MCP Agentic Workflow Accelerator')
    parser.add_argument('--output', '-o', help='Output archive name (without extension)')
    parser.add_argument('--format', '-f', choices=['zip', 'tar'], default='zip', help='Archive format')
    parser.add_argument('--include-optional', '-i', action='store_true', help='Include optional files')
    parser.add_argument('--minimal', '-m', action='store_true', help='Create minimal archive')
    parser.add_argument('--no-optimize', action='store_true', help='Skip database optimization')
    
    args = parser.parse_args()
    
    creator = PortableArchiveCreator()
    
    try:
        if args.minimal:
            archive_path = creator.create_minimal_archive(args.output)
        else:
            archive_path = creator.create_portable_archive(
                output_name=args.output,
                include_optional=args.include_optional,
                format=args.format,
                optimize_db=not args.no_optimize
            )
        
        print(f"\n🎉 Archive created successfully: {archive_path}")
        print(f"📦 Ready for deployment!")
        
    except Exception as e:
        print(f"❌ Failed to create archive: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 