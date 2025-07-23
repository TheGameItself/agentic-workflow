"""
Memory Lobe Implementation for the MCP system.

This module provides an implementation of the IMemoryLobe interface.

ℵmemory_lobe(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from core.src.mcp.interfaces.memory_lobe import IMemoryLobe
from core.src.mcp.lobes.base_lobe import BaseLobe
from core.src.mcp.implementations.memory_manager import BasicMemoryManager
from core.src.mcp.exceptions import MCPLobeError, MCPMemoryError

logger = logging.getLogger(__name__)


class MemoryLobe(BaseLobe, IMemoryLobe):
    """Implementation of the IMemoryLobe interface."""
    
    def __init__(self, lobe_id: Optional[str] = None, name: Optional[str] = None, storage_path: Optional[str] = None):
        """Initialize the memory lobe."""
        super().__init__(lobe_id, name or "MemoryLobe")
        self.memory_manager = BasicMemoryManager(storage_path)
        self.memory_types = set()
        logger.info(f"MemoryLobe {self.name} initialized with storage path: {storage_path}")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the lobe with optional configuration."""
        try:
            result = super().initialize(config)
            if not result:
                return False
            
            # Initialize memory types from config
            if config and "memory_types" in config:
                self.memory_types.update(config["memory_types"])
            else:
                # Default memory types
                self.memory_types.update(["episodic", "semantic", "procedural", "working"])
            
            logger.info(f"MemoryLobe {self.name} initialized with memory types: {self.memory_types}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MemoryLobe {self.name}: {str(e)}")
            self.status = "error"
            return False
    
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        try:
            if isinstance(input_data, dict):
                action = input_data.get("action")
                
                if action == "store":
                    return self.store_memory(
                        input_data.get("data"),
                        input_data.get("memory_type", "episodic"),
                        input_data.get("metadata")
                    )
                elif action == "retrieve":
                    return self.retrieve_memory(input_data.get("memory_id"))
                elif action == "search":
                    return self.search_memories(
                        input_data.get("query"),
                        input_data.get("memory_type"),
                        input_data.get("limit", 10)
                    )
                elif action == "forget":
                    return self.forget_memory(input_data.get("memory_id"))
                else:
                    logger.warning(f"Unknown action in MemoryLobe {self.name}: {action}")
                    return None
            else:
                logger.warning(f"Invalid input data format in MemoryLobe {self.name}")
                return None
        except Exception as e:
            logger.error(f"Error processing data in MemoryLobe {self.name}: {str(e)}")
            return None
    
    def store_memory(self, memory_data: Any, memory_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory of the specified type."""
        try:
            if memory_type not in self.memory_types:
                logger.warning(f"Unknown memory type '{memory_type}' in MemoryLobe {self.name}")
                # Add the new memory type
                self.memory_types.add(memory_type)
            
            # Prepare metadata
            memory_metadata = metadata or {}
            memory_metadata["memory_type"] = memory_type
            memory_metadata["lobe_id"] = self.lobe_id
            
            # Store in memory manager
            memory_id = self.memory_manager.store(memory_data, metadata=memory_metadata)
            
            logger.info(f"MemoryLobe {self.name} stored memory {memory_id} of type {memory_type}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store memory in MemoryLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to store memory: {str(e)}")
    
    def retrieve_memory(self, memory_id: str) -> Any:
        """Retrieve a memory by ID."""
        try:
            memory_data = self.memory_manager.retrieve(memory_id)
            logger.info(f"MemoryLobe {self.name} retrieved memory {memory_id}")
            return memory_data
        except MCPMemoryError as e:
            logger.error(f"Failed to retrieve memory {memory_id} in MemoryLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to retrieve memory: {str(e)}")
    
    def search_memories(self, query: str, memory_type: Optional[str] = None, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search memories by query and optional type."""
        try:
            # Get all search results
            all_results = self.memory_manager.search(query, limit * 2)  # Get more to filter
            
            # Filter by memory type if specified
            if memory_type:
                filtered_results = []
                for memory_id, memory_data, score in all_results:
                    try:
                        metadata = self.memory_manager.get_metadata(memory_id)
                        if metadata.get("memory_type") == memory_type:
                            filtered_results.append((memory_id, memory_data, score))
                    except Exception:
                        continue
                
                results = filtered_results[:limit]
            else:
                results = all_results[:limit]
            
            logger.info(f"MemoryLobe {self.name} searched for '{query}' (type: {memory_type}), found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to search memories in MemoryLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to search memories: {str(e)}")
    
    def forget_memory(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        try:
            result = self.memory_manager.delete(memory_id)
            if result:
                logger.info(f"MemoryLobe {self.name} forgot memory {memory_id}")
            else:
                logger.warning(f"MemoryLobe {self.name} could not forget memory {memory_id} (not found)")
            return result
        except Exception as e:
            logger.error(f"Failed to forget memory {memory_id} in MemoryLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to forget memory: {str(e)}")
    
    def get_memory_types(self) -> List[str]:
        """Get all available memory types."""
        return list(self.memory_types)
    
    def receive_message(self, source_lobe_id: str, message: Any) -> bool:
        """Receive a message from another lobe."""
        try:
            logger.info(f"MemoryLobe {self.name} received message from {source_lobe_id}")
            
            # Process the message as input data
            result = self.process(message)
            
            # Send response back if needed
            if result is not None and source_lobe_id in self.connections:
                response_message = {
                    "type": "response",
                    "original_message": message,
                    "result": result
                }
                return self.send_message(source_lobe_id, response_message)
            
            return True
        except Exception as e:
            logger.error(f"Failed to receive message in MemoryLobe {self.name}: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the lobe."""
        base_status = super().get_status()
        base_status.update({
            "memory_types": list(self.memory_types),
            "total_memories": len(self.memory_manager.list_keys())
        })
        return base_status