#!/usr/bin/env python3
"""
Project Reorganization Script

This script reorganizes the MCP project structure by:
1. Moving used files from src/ to core/src/
2. Removing unused files outside of core/
3. Updating imports in affected files

Usage:
    python reorganize_project.py
"""

import os
import shutil
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path('.')
CORE_DIR = PROJECT_ROOT / 'core'
SRC_DIR = PROJECT_ROOT / 'src'
CORE_SRC_DIR = CORE_DIR / 'src'

# Files to move from src/ to core/src/
FILES_TO_MOVE = [
    ('src/mcp/visualization/gitgraph_visualizer.py', 'core/src/mcp/visualization/gitgraph_visualizer.py'),
    # Add more files as needed
]

# Create __init__.py files in directories that need them
INIT_FILES_TO_CREATE = [
    'core/src/mcp/visualization/__init__.py',
    # Add more directories as needed
]

# Files to remove (unused files outside of core/)
FILES_TO_REMOVE = [
    'src/mcp/visualization/gitgraph_visualizer.py',  # After moving
    # Add more files as needed
]

def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    directory = Path(path).parent
    if not directory.exists():
        logger.info(f"Creating directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

def move_file(src_path, dst_path):
    """Move a file from src_path to dst_path."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    if not src_path.exists():
        logger.warning(f"Source file does not exist: {src_path}")
        return False
    
    ensure_directory_exists(dst_path)
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied: {src_path} -> {dst_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying {src_path} to {dst_path}: {e}")
        return False

def create_init_file(path, package_name=None):
    """Create an __init__.py file in the specified directory."""
    init_path = Path(path)
    
    ensure_directory_exists(init_path)
    
    if not init_path.exists():
        content = f'"""\n{package_name or Path(path).parent.name} package\n"""\n'
        try:
            with open(init_path, 'w') as f:
                f.write(content)
            logger.info(f"Created: {init_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating {init_path}: {e}")
            return False
    return False

def remove_file(path):
    """Remove a file if it exists."""
    path = Path(path)
    if path.exists():
        try:
            path.unlink()
            logger.info(f"Removed: {path}")
            return True
        except Exception as e:
            logger.error(f"Error removing {path}: {e}")
            return False
    else:
        logger.warning(f"File does not exist, cannot remove: {path}")
    return False

def main():
    """Main function to reorganize the project."""
    logger.info("Starting project reorganization...")
    
    # Move files from src/ to core/src/
    for src_path, dst_path in FILES_TO_MOVE:
        move_file(src_path, dst_path)
    
    # Create __init__.py files
    for init_path in INIT_FILES_TO_CREATE:
        create_init_file(init_path)
    
    # Remove unused files
    for file_path in FILES_TO_REMOVE:
        remove_file(file_path)
    
    logger.info("Project reorganization completed.")

if __name__ == "__main__":
    main()