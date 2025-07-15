#!/usr/bin/env python3
"""
Plugin System for MCP Server
Provides a comprehensive plugin architecture for extending MCP server functionality.

Features:
- Dynamic plugin loading and management
- Plugin lifecycle management
- Plugin dependency resolution
- Plugin configuration management
- Plugin development utilities
- Plugin marketplace integration
"""

import os
import sys
import json
import importlib
import importlib.util
import inspect
import logging
from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod
import yaml
import zipfile
import tempfile
import shutil

@dataclass
class PluginMetadata:
    """Plugin metadata structure."""
    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: Optional[str] = None
    repository: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    api_version: str = "2.0"
    python_version: str = "3.8+"
    entry_point: str = "main"
    config_schema: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

@dataclass
class PluginInfo:
    """Plugin information including status and metadata."""
    metadata: PluginMetadata
    path: str
    enabled: bool = True
    loaded: bool = False
    error: Optional[str] = None
    load_time: Optional[datetime] = None
    last_used: Optional[datetime] = None

class PluginBase(ABC):
    """Base class for all MCP plugins."""
    
    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        self.metadata = metadata
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{metadata.name}")
        self.server = None  # Will be set by plugin manager
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the plugin. Return True if successful."""
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update plugin configuration."""
        self.config = config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema."""
        if not self.metadata.config_schema:
            return {"valid": True, "errors": []}
        
        errors = []
        schema = self.metadata.config_schema
        
        for field_name, field_config in schema.items():
            if field_config.get("required", False) and field_name not in config:
                errors.append(f"Required field missing: {field_name}")
                continue
            
            if field_name in config:
                value = config[field_name]
                expected_type = field_config.get("type")
                
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field {field_name} must be {expected_type.__name__}")
                
                if "min" in field_config and value < field_config["min"]:
                    errors.append(f"Field {field_name} too small (min {field_config['min']})")
                
                if "max" in field_config and value > field_config["max"]:
                    errors.append(f"Field {field_name} too large (max {field_config['max']})")
        
        return {"valid": len(errors) == 0, "errors": errors}

class PluginManager:
    """Manages plugin loading, lifecycle, and integration."""
    
    def __init__(self, server=None, plugin_dir: str = "plugins"):
        self.server = server
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, PluginBase] = {}
        self.logger = logging.getLogger("plugin_manager")
        self.config_file = self.plugin_dir / "plugin_config.json"
        self.load_plugin_config()
    
    def load_plugin_config(self) -> None:
        """Load plugin configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    for plugin_name, plugin_config in config.items():
                        if plugin_name in self.plugins:
                            self.plugins[plugin_name].enabled = plugin_config.get("enabled", True)
            except Exception as e:
                self.logger.error(f"Failed to load plugin config: {e}")
    
    def save_plugin_config(self) -> None:
        """Save plugin configuration to file."""
        config = {}
        for plugin_name, plugin_info in self.plugins.items():
            config[plugin_name] = {
                "enabled": plugin_info.enabled,
                "config": plugin_info.metadata.config_schema
            }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save plugin config: {e}")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugin directory."""
        discovered = []
        
        for item in self.plugin_dir.iterdir():
            if item.is_dir() and (item / "plugin.yaml").exists():
                discovered.append(item.name)
            elif item.is_file() and item.suffix == ".py" and item.name.startswith("plugin_"):
                discovered.append(item.stem)
        
        return discovered
    
    def load_plugin_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from YAML file."""
        yaml_file = plugin_path / "plugin.yaml"
        if not yaml_file.exists():
            return None
        
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            return PluginMetadata(**data)
        except Exception as e:
            self.logger.error(f"Failed to load plugin metadata from {yaml_file}: {e}")
            return None
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name."""
        plugin_path = self.plugin_dir / plugin_name
        
        if not plugin_path.exists():
            self.logger.error(f"Plugin path does not exist: {plugin_path}")
            return False
        
        # Load metadata
        metadata = self.load_plugin_metadata(plugin_path)
        if not metadata:
            self.logger.error(f"Failed to load metadata for plugin: {plugin_name}")
            return False
        
        # Check dependencies
        if not self.check_dependencies(metadata.dependencies):
            self.logger.error(f"Plugin dependencies not met: {metadata.dependencies}")
            return False
        
        # Load plugin module
        try:
            if (plugin_path / "__init__.py").exists():
                # Package plugin
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{plugin_name}",
                    plugin_path / "__init__.py"
                )
            else:
                # Single file plugin
                plugin_file = plugin_path / f"{plugin_name}.py"
                if not plugin_file.exists():
                    plugin_file = plugin_path / "main.py"
                
                if not plugin_file.exists():
                    self.logger.error(f"No plugin file found for: {plugin_name}")
                    return False
                
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{plugin_name}",
                    plugin_file
                )
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginBase) and 
                    obj != PluginBase):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                self.logger.error(f"No plugin class found in: {plugin_name}")
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class(metadata)
            if self.server:
                plugin_instance.server = self.server
            
            # Initialize plugin
            try:
                asyncio.create_task(plugin_instance.initialize())
                self.loaded_plugins[plugin_name] = plugin_instance
                
                # Update plugin info
                if plugin_name in self.plugins:
                    self.plugins[plugin_name].loaded = True
                    self.plugins[plugin_name].load_time = datetime.now()
                    self.plugins[plugin_name].error = None
                else:
                    self.plugins[plugin_name] = PluginInfo(
                        metadata=metadata,
                        path=str(plugin_path),
                        loaded=True,
                        load_time=datetime.now()
                    )
                
                self.logger.info(f"Plugin loaded successfully: {plugin_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
                if plugin_name in self.plugins:
                    self.plugins[plugin_name].error = str(e)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            if plugin_name in self.plugins:
                self.plugins[plugin_name].error = str(e)
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name."""
        if plugin_name not in self.loaded_plugins:
            return True
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            asyncio.create_task(plugin.shutdown())
            del self.loaded_plugins[plugin_name]
            
            if plugin_name in self.plugins:
                self.plugins[plugin_name].loaded = False
                self.plugins[plugin_name].last_used = datetime.now()
            
            self.logger.info(f"Plugin unloaded: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
            self.save_plugin_config()
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
            if plugin_name in self.loaded_plugins:
                self.unload_plugin(plugin_name)
            self.save_plugin_config()
            return True
        return False
    
    def check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are met."""
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                self.logger.warning(f"Dependency not available: {dep}")
                return False
        return True
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a plugin."""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginInfo]:
        """List all plugins."""
        return list(self.plugins.values())
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin names."""
        return list(self.loaded_plugins.keys())
    
    def call_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs) -> Any:
        """Call a method on a loaded plugin."""
        if plugin_name not in self.loaded_plugins:
            raise ValueError(f"Plugin not loaded: {plugin_name}")
        
        plugin = self.loaded_plugins[plugin_name]
        if not hasattr(plugin, method_name):
            raise ValueError(f"Method not found: {method_name}")
        
        method = getattr(plugin, method_name)
        if asyncio.iscoroutinefunction(method):
            return asyncio.create_task(method(*args, **kwargs))
        else:
            return method(*args, **kwargs)
    
    def install_plugin(self, plugin_path: str) -> bool:
        """Install a plugin from a path or URL."""
        try:
            # Handle different installation sources
            if plugin_path.startswith("http"):
                return self.install_from_url(plugin_path)
            elif os.path.exists(plugin_path):
                return self.install_from_path(plugin_path)
            else:
                self.logger.error(f"Invalid plugin path: {plugin_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to install plugin: {e}")
            return False
    
    def install_from_path(self, plugin_path: str) -> bool:
        """Install a plugin from a local path."""
        source_path = Path(plugin_path)
        if not source_path.exists():
            return False
        
        plugin_name = source_path.name
        target_path = self.plugin_dir / plugin_name
        
        if target_path.exists():
            shutil.rmtree(target_path)
        
        if source_path.is_file() and source_path.suffix == ".zip":
            # Extract zip file
            with zipfile.ZipFile(source_path, 'r') as zip_ref:
                zip_ref.extractall(target_path)
        else:
            # Copy directory
            shutil.copytree(source_path, target_path)
        
        # Load the newly installed plugin
        return self.load_plugin(plugin_name)
    
    def install_from_url(self, url: str) -> bool:
        """Install a plugin from a URL."""
        try:
            import requests
            
            # Download plugin
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Install from temporary file
            success = self.install_from_path(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to install from URL: {e}")
            return False
    
    def create_plugin_template(self, plugin_name: str, template_type: str = "basic") -> bool:
        """Create a new plugin template."""
        plugin_path = self.plugin_dir / plugin_name
        if plugin_path.exists():
            self.logger.error(f"Plugin directory already exists: {plugin_path}")
            return False
        
        plugin_path.mkdir(parents=True)
        
        # Create plugin.yaml
        metadata = PluginMetadata(
            name=plugin_name,
            version="0.1.0",
            description=f"A plugin for {plugin_name}",
            author="Your Name",
            license="MIT",
            tags=["example"]
        )
        
        yaml_file = plugin_path / "plugin.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(asdict(metadata), f, default_flow_style=False)
        
        # Create __init__.py
        init_content = f'''#!/usr/bin/env python3
"""
{plugin_name} Plugin for MCP Server
"""

from .plugin import {plugin_name.title()}Plugin

__all__ = ["{plugin_name.title()}Plugin"]
'''
        
        with open(plugin_path / "__init__.py", 'w') as f:
            f.write(init_content)
        
        # Create plugin.py
        plugin_content = f'''#!/usr/bin/env python3
"""
{plugin_name.title()} Plugin Implementation
"""

from ..plugin_system import PluginBase, PluginMetadata
from typing import Dict, Any

class {plugin_name.title()}Plugin(PluginBase):
    """{plugin_name.title()} plugin implementation."""
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing {plugin_name} plugin")
        # Add your initialization code here
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        self.logger.info("Shutting down {plugin_name} plugin")
        # Add your cleanup code here
        return True
    
    # Add your plugin methods here
    async def example_method(self, param: str) -> str:
        """Example plugin method."""
        return f"Hello from {self.metadata.name}: {{param}}"
'''
        
        with open(plugin_path / "plugin.py", 'w') as f:
            f.write(plugin_content)
        
        # Create README.md
        readme_content = f'''# {plugin_name.title()} Plugin

## Description

{metadata.description}

## Installation

This plugin is automatically loaded by the MCP server when placed in the plugins directory.

## Configuration

No configuration required.

## Usage

This plugin provides the following functionality:

- `example_method(param)`: Example method that returns a greeting

## Development

To modify this plugin:

1. Edit the plugin files in the `{plugin_name}` directory
2. Restart the MCP server to reload the plugin
3. Test your changes

## License

{metadata.license}
'''
        
        with open(plugin_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Plugin template created: {plugin_path}")
        return True

class PluginMarketplace:
    """Plugin marketplace for discovering and installing plugins."""
    
    def __init__(self, marketplace_url: str = "https://api.example.com/plugins"):
        self.marketplace_url = marketplace_url
        self.logger = logging.getLogger("plugin_marketplace")
    
    async def search_plugins(self, query: str = "", tags: List[str] = None) -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace."""
        try:
            import requests
            
            params = {"q": query}
            if tags:
                params["tags"] = ",".join(tags)
            
            response = requests.get(f"{self.marketplace_url}/search", params=params)
            response.raise_for_status()
            
            return response.json().get("plugins", [])
            
        except Exception as e:
            self.logger.error(f"Failed to search marketplace: {e}")
            return []
    
    async def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        try:
            import requests
            
            response = requests.get(f"{self.marketplace_url}/plugin/{plugin_id}")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get plugin info: {e}")
            return None
    
    async def download_plugin(self, plugin_id: str, target_path: str) -> bool:
        """Download a plugin from the marketplace."""
        try:
            import requests
            
            response = requests.get(f"{self.marketplace_url}/download/{plugin_id}")
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download plugin: {e}")
            return False 