# Plugin Development Guide

This guide provides comprehensive instructions for developing plugins for the MCP server.

## Overview

The MCP server includes a powerful plugin system that allows you to extend its functionality with custom features. Plugins can add new API endpoints, integrate with external services, provide additional processing capabilities, and more.

## Plugin Architecture

### Core Components

1. **PluginBase**: Abstract base class that all plugins must inherit from
2. **PluginMetadata**: Data structure containing plugin information
3. **PluginManager**: Manages plugin loading, lifecycle, and integration
4. **PluginMarketplace**: For discovering and installing plugins

### Plugin Structure

A typical plugin has the following structure:

```
plugins/
└── my_plugin/
    ├── plugin.yaml          # Plugin metadata and configuration
    ├── __init__.py          # Plugin package initialization
    ├── plugin.py            # Main plugin implementation
    ├── README.md            # Plugin documentation
    └── requirements.txt     # Plugin dependencies (optional)
```

## Creating a New Plugin

### 1. Plugin Template

Use the built-in template generator:

```bash
python -m src.mcp.cli create-plugin-template my_plugin
```

This creates a basic plugin structure with all necessary files.

### 2. Plugin Metadata (plugin.yaml)

The `plugin.yaml` file defines your plugin's metadata and configuration schema:

```yaml
name: my_plugin
version: "1.0.0"
description: "A description of what your plugin does"
author: "Your Name"
license: "MIT"
homepage: "https://github.com/your-repo/my-plugin"
repository: "https://github.com/your-repo/my-plugin"
dependencies: ["requests", "pandas"]  # Python package dependencies
tags: ["api", "data-processing", "example"]
api_version: "2.0"
python_version: "3.8+"
entry_point: "main"
config_schema:
  api_key:
    type: str
    required: true
    description: "API key for external service"
  timeout:
    type: int
    required: false
    default: 30
    min: 1
    max: 300
  enabled_features:
    type: list
    required: false
    default: ["feature1", "feature2"]
    max_items: 10
```

### 3. Plugin Implementation (plugin.py)

Your plugin must inherit from `PluginBase` and implement required methods:

```python
#!/usr/bin/env python3
"""
My Plugin Implementation
"""

import asyncio
from typing import Dict, Any, List, Optional
from src.mcp.plugin_system import PluginBase, PluginMetadata

class MyPlugin(PluginBase):
    """My custom plugin implementation."""
    
    def __init__(self, metadata: PluginMetadata, config: Optional[Dict[str, Any]] = None):
        super().__init__(metadata, config or {})
        self.data_store = {}
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing My Plugin")
        
        # Validate configuration
        validation = self.validate_config(self.config)
        if not validation["valid"]:
            self.logger.error(f"Configuration validation failed: {validation['errors']}")
            return False
        
        # Set up your plugin
        api_key = self.config.get("api_key")
        if not api_key:
            self.logger.error("API key is required")
            return False
        
        # Initialize any external connections, databases, etc.
        try:
            # Your initialization code here
            self.logger.info("My Plugin initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        self.logger.info("Shutting down My Plugin")
        
        # Clean up resources
        try:
            # Your cleanup code here
            self.data_store.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False
    
    # Add your custom methods here
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using your plugin's logic."""
        # Your processing logic here
        processed_data = data.copy()
        processed_data["processed_by"] = self.metadata.name
        processed_data["timestamp"] = asyncio.get_event_loop().time()
        
        return processed_data
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "status": "running",
            "data_count": len(self.data_store)
        }
```

### 4. Plugin Package (__init__.py)

The `__init__.py` file exports your plugin class:

```python
#!/usr/bin/env python3
"""
My Plugin for MCP Server
"""

from .plugin import MyPlugin

__all__ = ["MyPlugin"]
```

## Plugin Development Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
async def safe_method(self, param: str) -> Dict[str, Any]:
    """Example of safe error handling."""
    try:
        result = await self.process_data(param)
        return {"success": True, "data": result}
    except ValueError as e:
        self.logger.warning(f"Invalid parameter: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        return {"success": False, "error": "Internal error"}
```

### 2. Configuration Validation

Always validate your configuration:

```python
def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom configuration validation."""
    errors = []
    
    # Check required fields
    if "api_key" not in config:
        errors.append("api_key is required")
    
    # Check field types and values
    timeout = config.get("timeout", 30)
    if not isinstance(timeout, int) or timeout < 1:
        errors.append("timeout must be a positive integer")
    
    return {"valid": len(errors) == 0, "errors": errors}
```

### 3. Logging

Use the built-in logger for consistent logging:

```python
async def some_method(self):
    self.logger.debug("Starting operation")
    self.logger.info("Operation completed successfully")
    self.logger.warning("Something to watch out for")
    self.logger.error("Something went wrong")
```

### 4. Async/Await

Use async/await for I/O operations:

```python
async def fetch_data(self, url: str) -> Dict[str, Any]:
    """Fetch data from external API."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## Plugin Integration with MCP Server

### 1. Accessing Server Components

Your plugin can access server components through `self.server`:

```python
async def use_server_features(self):
    """Example of using server features."""
    if self.server:
        # Access task manager
        tasks = self.server.task_manager.get_tasks()
        
        # Access memory manager
        memories = self.server.memory_manager.search_memories("query")
        
        # Access workflow manager
        workflow_status = self.server.workflow_manager.get_workflow_status()
```

### 2. Adding API Endpoints

Plugins can add new API endpoints by implementing methods that follow the MCP pattern:

```python
async def handle_custom_endpoint(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle custom API endpoint."""
    action = params.get("action")
    
    if action == "process":
        data = params.get("data", {})
        result = await self.process_data(data)
        return {"status": "success", "result": result}
    elif action == "status":
        status = await self.get_status()
        return {"status": "success", "data": status}
    else:
        return {"status": "error", "message": "Unknown action"}
```

### 3. Plugin Lifecycle

Plugins go through the following lifecycle:

1. **Discovery**: Plugin manager discovers the plugin
2. **Loading**: Plugin module is loaded and metadata is parsed
3. **Initialization**: `initialize()` method is called
4. **Active**: Plugin is ready to handle requests
5. **Shutdown**: `shutdown()` method is called when server stops

## Testing Your Plugin

### 1. Unit Testing

Create tests for your plugin:

```python
# test_my_plugin.py
import pytest
import asyncio
from src.mcp.plugin_system import PluginMetadata
from plugins.my_plugin.plugin import MyPlugin

@pytest.fixture
def plugin_metadata():
    return PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        description="Test plugin",
        author="Test Author",
        license="MIT"
    )

@pytest.fixture
def plugin(plugin_metadata):
    return MyPlugin(plugin_metadata, {"api_key": "test_key"})

@pytest.mark.asyncio
async def test_plugin_initialization(plugin):
    result = await plugin.initialize()
    assert result is True

@pytest.mark.asyncio
async def test_plugin_shutdown(plugin):
    await plugin.initialize()
    result = await plugin.shutdown()
    assert result is True

@pytest.mark.asyncio
async def test_process_data(plugin):
    await plugin.initialize()
    data = {"test": "data"}
    result = await plugin.process_data(data)
    assert "processed_by" in result
    assert result["processed_by"] == "test_plugin"
```

### 2. Integration Testing

Test your plugin with the MCP server:

```python
# test_integration.py
import pytest
from src.mcp.plugin_system import PluginManager

@pytest.mark.asyncio
async def test_plugin_integration():
    manager = PluginManager()
    
    # Load plugin
    success = manager.load_plugin("my_plugin")
    assert success is True
    
    # Test plugin method
    result = manager.call_plugin_method("my_plugin", "get_status")
    assert result["name"] == "my_plugin"
    
    # Unload plugin
    success = manager.unload_plugin("my_plugin")
    assert success is True
```

## Plugin Distribution

### 1. Local Installation

For local development, simply place your plugin in the `plugins/` directory:

```bash
cp -r my_plugin plugins/
```

### 2. Package Distribution

Create a distributable package:

```bash
# Create plugin package
cd my_plugin
zip -r ../my_plugin.zip .

# Install from package
python -m src.mcp.cli install-plugin ../my_plugin.zip
```

### 3. Plugin Marketplace

For wider distribution, consider publishing to the plugin marketplace:

```python
# Example marketplace integration
from src.mcp.plugin_system import PluginMarketplace

marketplace = PluginMarketplace()

# Search for plugins
plugins = await marketplace.search_plugins("data processing")

# Get plugin info
info = await marketplace.get_plugin_info("plugin_id")

# Download plugin
success = await marketplace.download_plugin("plugin_id", "plugin.zip")
```

## Advanced Plugin Features

### 1. Plugin Dependencies

Handle dependencies in your plugin:

```python
# In plugin.yaml
dependencies: ["requests", "pandas", "numpy"]

# In plugin.py
def check_dependencies(self) -> bool:
    """Check if all dependencies are available."""
    required_packages = ["requests", "pandas", "numpy"]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            self.logger.error(f"Required package not found: {package}")
            return False
    
    return True
```

### 2. Plugin Configuration UI

Create a configuration interface:

```python
def get_config_schema(self) -> Dict[str, Any]:
    """Get configuration schema for UI generation."""
    return {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "title": "API Key",
                "description": "API key for external service"
            },
            "timeout": {
                "type": "integer",
                "title": "Timeout",
                "description": "Request timeout in seconds",
                "minimum": 1,
                "maximum": 300,
                "default": 30
            }
        },
        "required": ["api_key"]
    }
```

### 3. Plugin Metrics

Add metrics collection:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PluginMetrics:
    requests_processed: int = 0
    errors_encountered: int = 0
    last_request_time: Optional[datetime] = None

class MyPlugin(PluginBase):
    def __init__(self, metadata, config):
        super().__init__(metadata, config)
        self.metrics = PluginMetrics()
    
    async def process_request(self, data):
        self.metrics.requests_processed += 1
        self.metrics.last_request_time = datetime.now()
        
        try:
            result = await self.process_data(data)
            return result
        except Exception as e:
            self.metrics.errors_encountered += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "requests_processed": self.metrics.requests_processed,
            "errors_encountered": self.metrics.errors_encountered,
            "last_request_time": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None
        }
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure your plugin's dependencies are installed
2. **Configuration Errors**: Validate your `plugin.yaml` syntax
3. **Initialization Failures**: Check your `initialize()` method for errors
4. **Permission Issues**: Ensure the plugin directory is writable

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger("plugin.my_plugin").setLevel(logging.DEBUG)
```

### Getting Help

- Check the example plugin in `plugins/example_plugin/`
- Review the plugin system source code in `src/mcp/plugin_system.py`
- Use the CLI commands for plugin management
- Check the server logs for error messages

## Conclusion

The MCP plugin system provides a powerful and flexible way to extend the server's functionality. By following this guide and the best practices outlined, you can create robust, maintainable plugins that integrate seamlessly with the MCP server.

For more information, see the example plugin and the plugin system source code. 