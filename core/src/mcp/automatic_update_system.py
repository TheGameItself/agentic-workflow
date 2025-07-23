#!/usr/bin/env python3
"""
Automatic Update System for MCP Agentic Workflow Accelerator

Provides comprehensive automatic update capabilities including:
- Version checking and comparison
- Update detection and notification
- Automatic download and installation
- Rollback capabilities
- Update verification and health checks
- Cross-platform compatibility
- Local-only operation with optional network updates
"""

import os
import sys
import json
import hashlib
import shutil
import logging
import asyncio
import threading
import time
import tempfile
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import subprocess
import platform
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import packaging.version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False


class UpdateStatus(Enum):
    """Update status enumeration."""
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    UPDATE_IN_PROGRESS = "update_in_progress"
    UPDATE_FAILED = "update_failed"
    UPDATE_COMPLETED = "update_completed"
    CHECK_FAILED = "check_failed"


@dataclass
class VersionInfo:
    """Version information structure."""
    version: str
    release_date: datetime
    changelog: str
    download_url: str
    checksum: str
    size: int
    compatibility: List[str]
    critical: bool = False
    auto_update: bool = True


@dataclass
class UpdateConfig:
    """Update configuration."""
    check_interval_hours: int = 24
    auto_update_enabled: bool = True
    auto_update_critical: bool = True
    backup_enabled: bool = True
    rollback_enabled: bool = True
    max_backup_versions: int = 3
    update_sources: List[str] = field(default_factory=lambda: [
        "https://api.github.com/repos/your-repo/agentic-workflow/releases/latest",
        "https://your-update-server.com/api/updates"
    ])
    local_update_path: Optional[str] = None
    network_timeout: int = 30
    retry_attempts: int = 3


class AutomaticUpdateSystem:
    """
    Comprehensive automatic update system for the MCP Agentic Workflow Accelerator.
    
    Features:
    - Version checking and comparison
    - Automatic update detection
    - Safe download and installation
    - Rollback capabilities
    - Health verification
    - Cross-platform support
    - Local-only operation
    """
    
    def __init__(self, project_root: Optional[str] = None, config: Optional[UpdateConfig] = None):
        """Initialize the automatic update system."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config = config or UpdateConfig()
        self.logger = logging.getLogger("automatic_update_system")
        
        # Version tracking
        self.current_version = self._get_current_version()
        self.latest_version_info: Optional[VersionInfo] = None
        self.update_status = UpdateStatus.UP_TO_DATE
        
        # Update tracking
        self.last_check_time: Optional[datetime] = None
        self.update_history: List[Dict[str, Any]] = []
        
        # Threading
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()
        
        # Directories
        self.backup_dir = self.project_root / "backups"
        self.temp_dir = self.project_root / "temp"
        self.update_dir = self.project_root / "updates"
        
        # Create directories
        self.backup_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        self.update_dir.mkdir(exist_ok=True)
        
        # Load update history
        self._load_update_history()
        
        self.logger.info(f"AutomaticUpdateSystem initialized for version {self.current_version}")
    
    def _get_current_version(self) -> str:
        """Get the current version of the MCP system."""
        # Try to read from version file
        version_file = self.project_root / "VERSION"
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                self.logger.warning(f"Error reading version file: {e}")
        
        # Try to read from package metadata
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("mcp-agentic-workflow").version
            return version
        except Exception:
            pass
        
        # Fallback to git version
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--always"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Default version
        return "2.0.0-dev"
    
    def _load_update_history(self):
        """Load update history from file."""
        history_file = self.project_root / "update_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.update_history = json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading update history: {e}")
                self.update_history = []
    
    def _save_update_history(self):
        """Save update history to file."""
        history_file = self.project_root / "update_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.update_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving update history: {e}")
    
    def start(self):
        """Start the automatic update system."""
        if self.running:
            self.logger.warning("AutomaticUpdateSystem is already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("AutomaticUpdateSystem started")
    
    def stop(self):
        """Stop the automatic update system."""
        if not self.running:
            return
        
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        
        self.logger.info("AutomaticUpdateSystem stopped")
    
    def _update_loop(self):
        """Main update loop."""
        while self.running:
            try:
                # Check for updates
                if self._should_check_for_updates():
                    self._check_for_updates()
                
                # Sleep for check interval
                time.sleep(self.config.check_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(3600)  # Sleep for 1 hour on error
    
    def _should_check_for_updates(self) -> bool:
        """Check if it's time to check for updates."""
        if not self.last_check_time:
            return True
        
        time_since_check = datetime.now() - self.last_check_time
        return time_since_check.total_seconds() >= self.config.check_interval_hours * 3600
    
    def check_for_updates(self) -> Tuple[UpdateStatus, Optional[VersionInfo]]:
        """Check for available updates."""
        with self._lock:
            return self._check_for_updates()
    
    def _check_for_updates(self) -> Tuple[UpdateStatus, Optional[VersionInfo]]:
        """Internal method to check for updates."""
        try:
            self.logger.info("Checking for updates...")
            
            # Get latest version info
            latest_info = self._fetch_latest_version_info()
            if not latest_info:
                self.update_status = UpdateStatus.CHECK_FAILED
                return self.update_status, None
            
            self.latest_version_info = latest_info
            
            # Compare versions
            if self._is_newer_version(latest_info.version):
                self.update_status = UpdateStatus.UPDATE_AVAILABLE
                self.logger.info(f"Update available: {self.current_version} -> {latest_info.version}")
                
                # Auto-update if enabled and appropriate
                if self.config.auto_update_enabled:
                    if latest_info.critical and self.config.auto_update_critical:
                        self.logger.info("Critical update detected, starting automatic update...")
                        asyncio.create_task(self._perform_automatic_update())
                    elif latest_info.auto_update:
                        self.logger.info("Non-critical update available, scheduling automatic update...")
                        # Schedule update for next idle period
                        threading.Timer(300, lambda: asyncio.create_task(self._perform_automatic_update())).start()
            else:
                self.update_status = UpdateStatus.UP_TO_DATE
                self.logger.info("System is up to date")
            
            self.last_check_time = datetime.now()
            return self.update_status, latest_info
            
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            self.update_status = UpdateStatus.CHECK_FAILED
            return self.update_status, None
    
    def _fetch_latest_version_info(self) -> Optional[VersionInfo]:
        """Fetch latest version information from update sources."""
        for source in self.config.update_sources:
            try:
                if source.startswith("http"):
                    info = self._fetch_from_remote_source(source)
                else:
                    info = self._fetch_from_local_source(source)
                
                if info:
                    return info
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source}: {e}")
                continue
        
        return None
    
    def _fetch_from_remote_source(self, url: str) -> Optional[VersionInfo]:
        """Fetch version info from remote source."""
        if not REQUESTS_AVAILABLE:
            # Fallback to urllib
            try:
                with urllib.request.urlopen(url, timeout=self.config.network_timeout) as response:
                    data = json.loads(response.read().decode())
                    return self._parse_version_info(data)
            except Exception as e:
                self.logger.error(f"Error fetching from remote source: {e}")
                return None
        
        # Use requests if available
        try:
            response = requests.get(url, timeout=self.config.network_timeout)
            response.raise_for_status()
            data = response.json()
            return self._parse_version_info(data)
        except Exception as e:
            self.logger.error(f"Error fetching from remote source: {e}")
            return None
    
    def _fetch_from_local_source(self, path: str) -> Optional[VersionInfo]:
        """Fetch version info from local source."""
        try:
            local_path = Path(path)
            if local_path.exists():
                with open(local_path, 'r') as f:
                    data = json.load(f)
                    return self._parse_version_info(data)
        except Exception as e:
            self.logger.error(f"Error fetching from local source: {e}")
        
        return None
    
    def _parse_version_info(self, data: Dict[str, Any]) -> Optional[VersionInfo]:
        """Parse version information from data."""
        try:
            return VersionInfo(
                version=data.get("version", "0.0.0"),
                release_date=datetime.fromisoformat(data.get("release_date", datetime.now().isoformat())),
                changelog=data.get("changelog", ""),
                download_url=data.get("download_url", ""),
                checksum=data.get("checksum", ""),
                size=data.get("size", 0),
                compatibility=data.get("compatibility", []),
                critical=data.get("critical", False),
                auto_update=data.get("auto_update", True)
            )
        except Exception as e:
            self.logger.error(f"Error parsing version info: {e}")
            return None
    
    def _is_newer_version(self, new_version: str) -> bool:
        """Check if the new version is newer than current version."""
        if not PACKAGING_AVAILABLE:
            # Simple string comparison fallback
            return new_version > self.current_version
        
        try:
            current = packaging.version.parse(self.current_version)
            new = packaging.version.parse(new_version)
            return new > current
        except Exception as e:
            self.logger.error(f"Error comparing versions: {e}")
            return False
    
    async def perform_update(self, version_info: Optional[VersionInfo] = None) -> bool:
        """Perform the update process."""
        if not version_info:
            version_info = self.latest_version_info
        
        if not version_info:
            self.logger.error("No version information available for update")
            return False
        
        with self._lock:
            if self.update_status == UpdateStatus.UPDATE_IN_PROGRESS:
                self.logger.warning("Update already in progress")
                return False
            
            self.update_status = UpdateStatus.UPDATE_IN_PROGRESS
        
        try:
            self.logger.info(f"Starting update to version {version_info.version}")
            
            # Create backup
            if self.config.backup_enabled:
                backup_path = await self._create_backup()
                if not backup_path:
                    self.logger.error("Failed to create backup")
                    self.update_status = UpdateStatus.UPDATE_FAILED
                    return False
            
            # Download update
            update_file = await self._download_update(version_info)
            if not update_file:
                self.logger.error("Failed to download update")
                self.update_status = UpdateStatus.UPDATE_FAILED
                return False
            
            # Verify update
            if not await self._verify_update(update_file, version_info):
                self.logger.error("Update verification failed")
                self.update_status = UpdateStatus.UPDATE_FAILED
                return False
            
            # Install update
            if not await self._install_update(update_file):
                self.logger.error("Update installation failed")
                if self.config.rollback_enabled:
                    await self._rollback_update(backup_path)
                self.update_status = UpdateStatus.UPDATE_FAILED
                return False
            
            # Verify installation
            if not await self._verify_installation():
                self.logger.error("Installation verification failed")
                if self.config.rollback_enabled:
                    await self._rollback_update(backup_path)
                self.update_status = UpdateStatus.UPDATE_FAILED
                return False
            
            # Update successful
            self._record_update_success(version_info)
            self.current_version = version_info.version
            self.update_status = UpdateStatus.UPDATE_COMPLETED
            
            self.logger.info(f"Update to version {version_info.version} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during update: {e}")
            self.update_status = UpdateStatus.UPDATE_FAILED
            return False
    
    async def _perform_automatic_update(self):
        """Perform automatic update."""
        if not self.latest_version_info:
            self.logger.warning("No version information available for automatic update")
            return
        
        success = await self.perform_update(self.latest_version_info)
        if success:
            self.logger.info("Automatic update completed successfully")
        else:
            self.logger.error("Automatic update failed")
    
    async def _create_backup(self) -> Optional[Path]:
        """Create a backup of the current installation."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{self.current_version}_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # Create backup
            shutil.copytree(self.project_root, backup_path, ignore=shutil.ignore_patterns(
                "backups", "temp", "updates", "__pycache__", "*.pyc", ".git"
            ))
            
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return None
    
    async def _download_update(self, version_info: VersionInfo) -> Optional[Path]:
        """Download the update package."""
        try:
            # Create temporary file
            temp_file = self.temp_dir / f"update_{version_info.version}.zip"
            
            if version_info.download_url.startswith("http"):
                # Download from remote URL
                if REQUESTS_AVAILABLE:
                    response = requests.get(version_info.download_url, stream=True, timeout=self.config.network_timeout)
                    response.raise_for_status()
                    
                    with open(temp_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    # Fallback to urllib
                    urllib.request.urlretrieve(version_info.download_url, temp_file)
            else:
                # Copy from local path
                shutil.copy2(version_info.download_url, temp_file)
            
            self.logger.info(f"Update downloaded: {temp_file}")
            return temp_file
            
        except Exception as e:
            self.logger.error(f"Error downloading update: {e}")
            return None
    
    async def _verify_update(self, update_file: Path, version_info: VersionInfo) -> bool:
        """Verify the downloaded update."""
        try:
            # Check file size
            if update_file.stat().st_size != version_info.size:
                self.logger.error("Update file size mismatch")
                return False
            
            # Check checksum
            if version_info.checksum:
                with open(update_file, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash != version_info.checksum:
                        self.logger.error("Update file checksum mismatch")
                        return False
            
            # Check if it's a valid archive
            try:
                if update_file.suffix == '.zip':
                    with zipfile.ZipFile(update_file, 'r') as zf:
                        zf.testzip()
                elif update_file.suffix in ['.tar', '.tar.gz', '.tgz']:
                    with tarfile.open(update_file, 'r:*') as tf:
                        tf.getmembers()
            except Exception as e:
                self.logger.error(f"Update file is not a valid archive: {e}")
                return False
            
            self.logger.info("Update verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying update: {e}")
            return False
    
    async def _install_update(self, update_file: Path) -> bool:
        """Install the update."""
        try:
            # Extract update to temporary directory
            extract_dir = self.temp_dir / "update_extract"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir()
            
            # Extract archive
            if update_file.suffix == '.zip':
                with zipfile.ZipFile(update_file, 'r') as zf:
                    zf.extractall(extract_dir)
            elif update_file.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(update_file, 'r:*') as tf:
                    tf.extractall(extract_dir)
            
            # Find the main directory in the extracted content
            extracted_items = list(extract_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                update_content_dir = extracted_items[0]
            else:
                update_content_dir = extract_dir
            
            # Install update files
            for item in update_content_dir.iterdir():
                target_path = self.project_root / item.name
                
                if item.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(item, target_path)
                else:
                    shutil.copy2(item, target_path)
            
            # Clean up
            shutil.rmtree(extract_dir)
            update_file.unlink()
            
            self.logger.info("Update installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing update: {e}")
            return False
    
    async def _verify_installation(self) -> bool:
        """Verify the installation was successful."""
        try:
            # Check if critical files exist
            critical_files = [
                "src/mcp/server.py",
                "mcp_cli.py",
                "README.md"
            ]
            
            for file_path in critical_files:
                if not (self.project_root / file_path).exists():
                    self.logger.error(f"Critical file missing after update: {file_path}")
                    return False
            
            # Try to import the main module
            try:
                sys.path.insert(0, str(self.project_root / "src"))
                import mcp.server
                sys.path.pop(0)
            except Exception as e:
                self.logger.error(f"Failed to import main module after update: {e}")
                return False
            
            self.logger.info("Installation verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying installation: {e}")
            return False
    
    async def _rollback_update(self, backup_path: Path):
        """Rollback to the previous version."""
        try:
            self.logger.info("Rolling back update...")
            
            # Remove current installation (excluding backups, temp, updates)
            for item in self.project_root.iterdir():
                if item.name not in ["backups", "temp", "updates"]:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # Restore from backup
            for item in backup_path.iterdir():
                target_path = self.project_root / item.name
                if item.is_dir():
                    shutil.copytree(item, target_path)
                else:
                    shutil.copy2(item, target_path)
            
            self.logger.info("Rollback completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
    
    def _record_update_success(self, version_info: VersionInfo):
        """Record successful update."""
        update_record = {
            "version": version_info.version,
            "timestamp": datetime.now().isoformat(),
            "previous_version": self.current_version,
            "changelog": version_info.changelog,
            "critical": version_info.critical
        }
        
        self.update_history.append(update_record)
        
        # Keep only recent history
        if len(self.update_history) > 10:
            self.update_history = self.update_history[-10:]
        
        self._save_update_history()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the update system."""
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version_info.version if self.latest_version_info else None,
            "update_status": self.update_status.value,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "auto_update_enabled": self.config.auto_update_enabled,
            "running": self.running,
            "update_history": self.update_history[-5:]  # Last 5 updates
        }
    
    def get_update_info(self) -> Optional[Dict[str, Any]]:
        """Get information about available update."""
        if not self.latest_version_info:
            return None
        
        return {
            "current_version": self.current_version,
            "new_version": self.latest_version_info.version,
            "release_date": self.latest_version_info.release_date.isoformat(),
            "changelog": self.latest_version_info.changelog,
            "size": self.latest_version_info.size,
            "critical": self.latest_version_info.critical,
            "auto_update": self.latest_version_info.auto_update
        }


# CLI interface for manual update operations
def main():
    """CLI interface for the automatic update system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Automatic Update System")
    parser.add_argument("--check", action="store_true", help="Check for updates")
    parser.add_argument("--update", action="store_true", help="Perform update")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--enable", action="store_true", help="Enable automatic updates")
    parser.add_argument("--disable", action="store_true", help="Disable automatic updates")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize update system
    update_system = AutomaticUpdateSystem()
    
    if args.check:
        status, version_info = update_system.check_for_updates()
        print(f"Update status: {status.value}")
        if version_info:
            print(f"Latest version: {version_info.version}")
            print(f"Release date: {version_info.release_date}")
            print(f"Changelog: {version_info.changelog}")
    
    elif args.update:
        asyncio.run(update_system.perform_update())
    
    elif args.status:
        status = update_system.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.enable:
        update_system.config.auto_update_enabled = True
        print("Automatic updates enabled")
    
    elif args.disable:
        update_system.config.auto_update_enabled = False
        print("Automatic updates disabled")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 