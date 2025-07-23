"""
Memory Manager Implementation for the MCP system.

This module provides a basic implementation of the IMemoryManager interface.

ℵmemory_manager(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import json
import os
from datetime import datetime

from core.src.mcp.interfaces.memory_manager import IMemoryManager
from core.src.mcp.exceptions import MCPMemoryError

logger = logging.getLogger(__name__)


class BasicMemoryManager(IMemoryManager):
    """Basic implementation of the IMemoryManager interface using in-memory storage."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the memory manager."""
        self.storage = {}  # In-memory storage
        self.metadata = {}  # Metadata storage
        self.storage_path = storage_path
        
        if storage_path and os.path.exists(storage_path):
            self._load_from_disk()
        
        logger.info(f"BasicMemoryManager initialized with storage path: {storage_path}")
    
    def store(self, data: Any, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in memory."""
        try:
            # Generate a key if not provided
            if key is None:
                key = str(uuid.uuid4())
            
            # Store the data
            self.storage[key] = data
            
            # Store metadata
            self.metadata[key] = metadata or {}
            self.metadata[key]["created_at"] = datetime.now().isoformat()
            self.metadata[key]["updated_at"] = self.metadata[key]["created_at"]
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info(f"Stored data with key: {key}")
            return key
        except Exception as e:
            logger.error(f"Failed to store data: {str(e)}")
            raise MCPMemoryError(f"Failed to store data: {str(e)}")
    
    def retrieve(self, key: str) -> Any:
        """Retrieve data from memory."""
        try:
            if key not in self.storage:
                raise MCPMemoryError(f"Key not found: {key}")
            
            logger.info(f"Retrieved data with key: {key}")
            return self.storage[key]
        except Exception as e:
            logger.error(f"Failed to retrieve data: {str(e)}")
            raise MCPMemoryError(f"Failed to retrieve data: {str(e)}")
    
    def update(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update data in memory."""
        try:
            if key not in self.storage:
                raise MCPMemoryError(f"Key not found: {key}")
            
            # Update the data
            self.storage[key] = data
            
            # Update metadata
            if metadata:
                self.metadata[key].update(metadata)
            self.metadata[key]["updated_at"] = datetime.now().isoformat()
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info(f"Updated data with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update data: {str(e)}")
            raise MCPMemoryError(f"Failed to update data: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """Delete data from memory."""
        try:
            if key not in self.storage:
                logger.warning(f"Key not found for deletion: {key}")
                return False
            
            # Delete the data and metadata
            del self.storage[key]
            del self.metadata[key]
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info(f"Deleted data with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete data: {str(e)}")
            raise MCPMemoryError(f"Failed to delete data: {str(e)}")
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for data in memory."""
        try:
            results = []
            
            # Simple string matching for now
            for key, data in self.storage.items():
                score = 0
                
                # Convert data to string for searching
                data_str = str(data)
                
                if query.lower() in data_str.lower():
                    # Calculate a simple relevance score
                    score = data_str.lower().count(query.lower()) / len(data_str)
                    results.append((key, data, score))
            
            # Sort by score and limit results
            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:limit]
            
            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to search data: {str(e)}")
            raise MCPMemoryError(f"Failed to search data: {str(e)}")
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a stored item."""
        try:
            if key not in self.metadata:
                raise MCPMemoryError(f"Key not found: {key}")
            
            return self.metadata[key]
        except Exception as e:
            logger.error(f"Failed to get metadata: {str(e)}")
            raise MCPMemoryError(f"Failed to get metadata: {str(e)}")
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys in memory."""
        try:
            keys = list(self.storage.keys())
            
            if pattern:
                # Simple pattern matching
                keys = [key for key in keys if pattern in key]
            
            logger.info(f"Listed {len(keys)} keys")
            return keys
        except Exception as e:
            logger.error(f"Failed to list keys: {str(e)}")
            raise MCPMemoryError(f"Failed to list keys: {str(e)}")
    
    def clear(self) -> bool:
        """Clear all data from memory."""
        try:
            self.storage = {}
            self.metadata = {}
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
            
            logger.info("Cleared all data from memory")
            return True
        except Exception as e:
            logger.error(f"Failed to clear data: {str(e)}")
            raise MCPMemoryError(f"Failed to clear data: {str(e)}")
    
    def _save_to_disk(self) -> None:
        """Save the memory state to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Prepare data for serialization
            serializable_data = {
                "storage": {},
                "metadata": self.metadata
            }
            
            # Convert non-serializable data to strings
            for key, value in self.storage.items():
                try:
                    # Try to serialize directly
                    json.dumps(value)
                    serializable_data["storage"][key] = value
                except (TypeError, OverflowError):
                    # Fall back to string representation
                    serializable_data["storage"][key] = str(value)
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved memory state to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save to disk: {str(e)}")
            raise MCPMemoryError(f"Failed to save to disk: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load the memory state from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.storage = data.get("storage", {})
            self.metadata = data.get("metadata", {})
            
            logger.info(f"Loaded memory state from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load from disk: {str(e)}")
            raise MCPMemoryError(f"Failed to load from disk: {str(e)}")