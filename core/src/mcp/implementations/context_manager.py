"""
Context Manager Implementation for the MCP system.

This module provides a basic implementation of the IContextManager interface.

λcontext_manager(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import json
import os
from datetime import datetime
import copy

from core.src.mcp.interfaces.context_manager import IContextManager
from core.src.mcp.exceptions import MCPContextError

logger = logging.getLogger(__name__)


class BasicContextManager(IContextManager):
    """Basic implementation of the IContextManager interface."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the context manager."""
        self.contexts = {}  # Dictionary of contexts
        self.history = {}  # Dictionary of context history
        self.storage_path = storage_path
        
        if storage_path and os.path.exists(storage_path):
            self._load_from_disk()
        
        logger.info(f"BasicContextManager initialized with storage path: {storage_path}")
    
    def create_context(self, name: str, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new context."""
        try:
            context_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Create context
            self.contexts[context_id] = {
                "id": context_id,
                "name": name,
                "created_at": timestamp,
                "updated_at": timestamp,
                "data": initial_data or {}
            }
            
            # Initialize history
            self.history[context_id] = [{
                "timestamp": timestamp,
                "action": "create",
                "data": copy.deepcopy(initial_data or {})
            }]
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info(f"Created context '{name}' with ID: {context_id}")
            return context_id
        except Exception as e:
            logger.error(f"Failed to create context: {str(e)}")
            raise MCPContextError(f"Failed to create context: {str(e)}")
    
    def get_context(self, context_id: str) -> Dict[str, Any]:
        """Get a context by ID."""
        try:
            if context_id not in self.contexts:
                raise MCPContextError(f"Context not found: {context_id}")
            
            return copy.deepcopy(self.contexts[context_id])
        except Exception as e:
            logger.error(f"Failed to get context: {str(e)}")
            raise MCPContextError(f"Failed to get context: {str(e)}")
    
    def update_context(self, context_id: str, data: Dict[str, Any], merge: bool = True) -> bool:
        """Update a context with new data."""
        try:
            if context_id not in self.contexts:
                raise MCPContextError(f"Context not found: {context_id}")
            
            timestamp = datetime.now().isoformat()
            
            # Update context
            if merge:
                # Deep merge the data
                self._deep_merge(self.contexts[context_id]["data"], data)
            else:
                # Replace the data
                self.contexts[context_id]["data"] = copy.deepcopy(data)
            
            # Update timestamp
            self.contexts[context_id]["updated_at"] = timestamp
            
            # Add to history
            self.history[context_id].append({
                "timestamp": timestamp,
                "action": "update",
                "data": copy.deepcopy(data),
                "merge": merge
            })
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info(f"Updated context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update context: {str(e)}")
            raise MCPContextError(f"Failed to update context: {str(e)}")
    
    def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        try:
            if context_id not in self.contexts:
                logger.warning(f"Context not found for deletion: {context_id}")
                return False
            
            # Delete context and history
            del self.contexts[context_id]
            del self.history[context_id]
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info(f"Deleted context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete context: {str(e)}")
            raise MCPContextError(f"Failed to delete context: {str(e)}")
    
    def list_contexts(self) -> List[Dict[str, Any]]:
        """List all contexts."""
        try:
            return [
                {
                    "id": context["id"],
                    "name": context["name"],
                    "created_at": context["created_at"],
                    "updated_at": context["updated_at"],
                    "data_size": len(json.dumps(context["data"]))
                }
                for context in self.contexts.values()
            ]
        except Exception as e:
            logger.error(f"Failed to list contexts: {str(e)}")
            raise MCPContextError(f"Failed to list contexts: {str(e)}")
    
    def export_context(self, context_id: str, format_type: str = "json") -> str:
        """Export a context to a specified format."""
        try:
            if context_id not in self.contexts:
                raise MCPContextError(f"Context not found: {context_id}")
            
            context = self.contexts[context_id]
            
            if format_type.lower() == "json":
                return json.dumps(context, indent=2)
            else:
                raise MCPContextError(f"Unsupported format type: {format_type}")
        except Exception as e:
            logger.error(f"Failed to export context: {str(e)}")
            raise MCPContextError(f"Failed to export context: {str(e)}")
    
    def import_context(self, context_data: str, format_type: str = "json") -> str:
        """Import a context from a specified format."""
        try:
            if format_type.lower() == "json":
                context = json.loads(context_data)
                
                # Validate context structure
                required_fields = ["name", "data"]
                for field in required_fields:
                    if field not in context:
                        raise MCPContextError(f"Missing required field in context data: {field}")
                
                # Create a new context
                context_id = self.create_context(context["name"], context["data"])
                
                return context_id
            else:
                raise MCPContextError(f"Unsupported format type: {format_type}")
        except Exception as e:
            logger.error(f"Failed to import context: {str(e)}")
            raise MCPContextError(f"Failed to import context: {str(e)}")
    
    def merge_contexts(self, context_ids: List[str], strategy: str = "override") -> str:
        """Merge multiple contexts into a new context."""
        try:
            # Validate contexts
            for context_id in context_ids:
                if context_id not in self.contexts:
                    raise MCPContextError(f"Context not found: {context_id}")
            
            # Create a new context
            merged_data = {}
            
            # Apply merge strategy
            if strategy == "override":
                # Later contexts override earlier ones
                for context_id in context_ids:
                    self._deep_merge(merged_data, self.contexts[context_id]["data"])
            elif strategy == "append_lists":
                # Append lists, override other values
                for context_id in context_ids:
                    self._deep_merge_append_lists(merged_data, self.contexts[context_id]["data"])
            else:
                raise MCPContextError(f"Unsupported merge strategy: {strategy}")
            
            # Create a new context with merged data
            context_names = [self.contexts[context_id]["name"] for context_id in context_ids]
            new_name = f"Merged: {', '.join(context_names)}"
            new_context_id = self.create_context(new_name, merged_data)
            
            logger.info(f"Merged contexts {context_ids} into new context: {new_context_id}")
            return new_context_id
        except Exception as e:
            logger.error(f"Failed to merge contexts: {str(e)}")
            raise MCPContextError(f"Failed to merge contexts: {str(e)}")
    
    def get_context_history(self, context_id: str) -> List[Dict[str, Any]]:
        """Get the history of changes to a context."""
        try:
            if context_id not in self.history:
                raise MCPContextError(f"Context history not found: {context_id}")
            
            return copy.deepcopy(self.history[context_id])
        except Exception as e:
            logger.error(f"Failed to get context history: {str(e)}")
            raise MCPContextError(f"Failed to get context history: {str(e)}")
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    def _deep_merge_append_lists(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target dictionary, appending lists."""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge_append_lists(target[key], value)
                elif isinstance(target[key], list) and isinstance(value, list):
                    target[key].extend(copy.deepcopy(value))
                else:
                    target[key] = copy.deepcopy(value)
            else:
                target[key] = copy.deepcopy(value)
    
    def _save_to_disk(self) -> None:
        """Save the context state to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Prepare data for serialization
            serializable_data = {
                "contexts": self.contexts,
                "history": self.history
            }
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved context state to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save to disk: {str(e)}")
            raise MCPContextError(f"Failed to save to disk: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load the context state from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.contexts = data.get("contexts", {})
            self.history = data.get("history", {})
            
            logger.info(f"Loaded context state from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load from disk: {str(e)}")
            raise MCPContextError(f"Failed to load from disk: {str(e)}")