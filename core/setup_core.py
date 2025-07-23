#!/usr/bin/env python3
"""
MCP Core System Setup
Setup script for the MCP Core System.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import json
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("setup")

def setup_directories():
    """Setup required directories."""
    directories = [
        "data",
        "data/logs",
        "data/backups",
        "data/config",
        "data/reports",
        "data/temp",
        "core/src/mcp/visualization"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")

def setup_config():
    """Setup default configuration."""
    config_path = Path("data/config/core_config.json")
    
    if config_path.exists():
        logger.info(f"Configuration already exists: {config_path}")
        return
    
    default_config = {
        "system": {
            "max_workers": 4,
            "enable_async": True,
            "enable_monitoring": True,
            "log_level": "INFO",
            "backup_enabled": True,
            "backup_interval": 3600,
            "performance_optimization": True,
            "experimental_features": False,
            "hormone_system_enabled": True,
            "vector_storage_enabled": True
        },
        "database": {
            "max_connections": 10,
            "enable_query_cache": True,
            "cache_size": 10000,
            "journal_mode": "wal",
            "synchronous": "normal"
        },
        "memory": {
            "working_capacity": 10,
            "short_term_capacity": 100,
            "long_term_capacity": 10000,
            "consolidation_interval": 60
        },
        "context": {
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 4096,
            "cache_enabled": True
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default configuration: {config_path}")

def install_dependencies(upgrade=False):
    """Install required dependencies."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        if upgrade:
            cmd.append("--upgrade")
        
        logger.info(f"Installing dependencies: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        logger.info("Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    
    return True

def run_health_check():
    """Run system health check."""
    try:
        cmd = [sys.executable, "core/system_health_check.py"]
        logger.info("Running system health check...")
        subprocess.check_call(cmd)
        logger.info("Health check completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Health check failed: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="MCP Core System Setup")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade dependencies")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check")
    
    args = parser.parse_args()
    
    logger.info("Starting MCP Core System setup...")
    
    # Setup directories
    setup_directories()
    
    # Setup configuration
    setup_config()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(args.upgrade):
            logger.error("Setup failed: Could not install dependencies")
            return 1
    
    # Run health check
    if not args.skip_health:
        if not run_health_check():
            logger.error("Setup failed: Health check failed")
            return 1
    
    logger.info("MCP Core System setup completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())