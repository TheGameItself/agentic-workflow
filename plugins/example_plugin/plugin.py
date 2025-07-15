#!/usr/bin/env python3
"""
Example Plugin Implementation
"""

import asyncio
from typing import Dict, Any, List, Optional
from src.mcp.plugin_system import PluginBase, PluginMetadata

class ExamplePlugin(PluginBase):
    """Example plugin demonstrating MCP plugin capabilities."""
    
    def __init__(self, metadata: PluginMetadata, config: Optional[Dict[str, Any]] = None):
        super().__init__(metadata, config or {})
        self.counter = 0
        self.features = {}
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing Example Plugin")
        
        # Set up features based on configuration
        enabled_features = self.config.get("enabled_features", ["feature1", "feature2"])
        for feature in enabled_features:
            self.features[feature] = True
        
        # Set log level
        log_level = self.config.get("log_level", "INFO")
        self.logger.setLevel(getattr(self.logger, log_level))
        
        self.logger.info(f"Example Plugin initialized with features: {list(self.features.keys())}")
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        self.logger.info("Shutting down Example Plugin")
        self.features.clear()
        return True
    
    async def get_greeting(self, name: str = "World") -> str:
        """Get a personalized greeting."""
        greeting = self.config.get("greeting", "Hello from Example Plugin")
        return f"{greeting}, {name}!"
    
    async def increment_counter(self) -> int:
        """Increment and return the counter."""
        self.counter += 1
        return self.counter
    
    async def get_counter(self) -> int:
        """Get the current counter value."""
        return self.counter
    
    async def reset_counter(self) -> int:
        """Reset the counter to zero."""
        self.counter = 0
        return self.counter
    
    async def get_features(self) -> Dict[str, bool]:
        """Get the status of all features."""
        return self.features.copy()
    
    async def enable_feature(self, feature_name: str) -> bool:
        """Enable a specific feature."""
        if feature_name in ["feature1", "feature2", "feature3"]:
            self.features[feature_name] = True
            self.logger.info(f"Feature enabled: {feature_name}")
            return True
        return False
    
    async def disable_feature(self, feature_name: str) -> bool:
        """Disable a specific feature."""
        if feature_name in self.features:
            self.features[feature_name] = False
            self.logger.info(f"Feature disabled: {feature_name}")
            return True
        return False
    
    async def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of data items."""
        processed = []
        for item in data:
            # Add a timestamp and counter to each item
            processed_item = item.copy()
            processed_item["processed_at"] = asyncio.get_event_loop().time()
            processed_item["counter"] = self.counter
            processed_item["plugin_version"] = self.metadata.version
            processed.append(processed_item)
        
        self.counter += len(data)
        return processed
    
    async def get_plugin_info(self) -> Dict[str, Any]:
        """Get comprehensive plugin information."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "license": self.metadata.license,
            "features": self.features,
            "counter": self.counter,
            "config": self.config
        } 