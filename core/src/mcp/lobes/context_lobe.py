"""
Context Lobe Implementation for the MCP system.

This module provides an implementation of the IContextLobe interface.

λcontext_lobe(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from core.src.mcp.interfaces.context_lobe import IContextLobe
from core.src.mcp.lobes.base_lobe import BaseLobe
from core.src.mcp.implementations.context_manager import BasicContextManager
from core.src.mcp.exceptions import MCPLobeError, MCPContextError

logger = logging.getLogger(__name__)


class ContextLobe(BaseLobe, IContextLobe):
    """Implementation of the IContextLobe interface."""
    
    def __init__(self, lobe_id: Optional[str] = None, name: Optional[str] = None, storage_path: Optional[str] = None):
        """Initialize the context lobe."""
        super().__init__(lobe_id, name or "ContextLobe")
        self.context_manager = BasicContextManager(storage_path)
        self.context_types = set()
        logger.info(f"ContextLobe {self.name} initialized with storage path: {storage_path}")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the lobe with optional configuration."""
        try:
            result = super().initialize(config)
            if not result:
                return False
            
            # Initialize context types from config
            if config and "context_types" in config:
                self.context_types.update(config["context_types"])
            else:
                # Default context types
                self.context_types.update(["conversation", "task", "project", "session"])
            
            logger.info(f"ContextLobe {self.name} initialized with context types: {self.context_types}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ContextLobe {self.name}: {str(e)}")
            self.status = "error"
            return False
    
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        try:
            if isinstance(input_data, dict):
                action = input_data.get("action")
                
                if action == "create_context":
                    return self.create_context(
                        input_data.get("context_type", "session"),
                        input_data.get("initial_data")
                    )
                elif action == "get_context":
                    return self.get_context(input_data.get("context_id"))
                elif action == "update_context":
                    return self.update_context(
                        input_data.get("context_id"),
                        input_data.get("data", {})
                    )
                elif action == "delete_context":
                    return self.delete_context(input_data.get("context_id"))
                elif action == "list_contexts":
                    return self.list_contexts(input_data.get("context_type"))
                else:
                    logger.warning(f"Unknown action in ContextLobe {self.name}: {action}")
                    return None
            else:
                logger.warning(f"Invalid input data format in ContextLobe {self.name}")
                return None
        except Exception as e:
            logger.error(f"Error processing data in ContextLobe {self.name}: {str(e)}")
            return None
    
    def create_context(self, context_type: str, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new context of the specified type."""
        try:
            if context_type not in self.context_types:
                logger.warning(f"Unknown context type '{context_type}' in ContextLobe {self.name}")
                # Add the new context type
                self.context_types.add(context_type)
            
            # Prepare context name
            context_name = f"{context_type}-{str(uuid.uuid4())[:8]}"
            
            # Prepare initial data with type metadata
            context_data = initial_data or {}
            context_data["_context_type"] = context_type
            context_data["_lobe_id"] = self.lobe_id
            
            # Create context
            context_id = self.context_manager.create_context(context_name, context_data)
            
            logger.info(f"ContextLobe {self.name} created context {context_id} of type {context_type}")
            return context_id
        except Exception as e:
            logger.error(f"Failed to create context in ContextLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to create context: {str(e)}")
    
    def get_context(self, context_id: str) -> Dict[str, Any]:
        """Get a context by ID."""
        try:
            context = self.context_manager.get_context(context_id)
            logger.info(f"ContextLobe {self.name} retrieved context {context_id}")
            return context
        except MCPContextError as e:
            logger.error(f"Failed to get context {context_id} in ContextLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to get context: {str(e)}")
    
    def update_context(self, context_id: str, data: Dict[str, Any]) -> bool:
        """Update a context with new data."""
        try:
            result = self.context_manager.update_context(context_id, data)
            if result:
                logger.info(f"ContextLobe {self.name} updated context {context_id}")
            else:
                logger.warning(f"ContextLobe {self.name} could not update context {context_id}")
            return result
        except MCPContextError as e:
            logger.error(f"Failed to update context {context_id} in ContextLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to update context: {str(e)}")
    
    def delete_context(self, context_id: str) -> bool:
        """Delete a context by ID."""
        try:
            result = self.context_manager.delete_context(context_id)
            if result:
                logger.info(f"ContextLobe {self.name} deleted context {context_id}")
            else:
                logger.warning(f"ContextLobe {self.name} could not delete context {context_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete context {context_id} in ContextLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to delete context: {str(e)}")
    
    def list_contexts(self, context_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all contexts, optionally filtered by type."""
        try:
            all_contexts = self.context_manager.list_contexts()
            
            # Filter by context type if specified
            if context_type:
                filtered_contexts = []
                for context_summary in all_contexts:
                    try:
                        # Get full context to check type
                        full_context = self.context_manager.get_context(context_summary["id"])
                        if full_context["data"].get("_context_type") == context_type:
                            filtered_contexts.append(context_summary)
                    except Exception:
                        continue
                
                contexts = filtered_contexts
            else:
                contexts = all_contexts
            
            logger.info(f"ContextLobe {self.name} listed {len(contexts)} contexts (type: {context_type})")
            return contexts
        except Exception as e:
            logger.error(f"Failed to list contexts in ContextLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to list contexts: {str(e)}")
    
    def receive_message(self, source_lobe_id: str, message: Any) -> bool:
        """Receive a message from another lobe."""
        try:
            logger.info(f"ContextLobe {self.name} received message from {source_lobe_id}")
            
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
            logger.error(f"Failed to receive message in ContextLobe {self.name}: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the lobe."""
        base_status = super().get_status()
        try:
            contexts = self.context_manager.list_contexts()
            base_status.update({
                "context_types": list(self.context_types),
                "total_contexts": len(contexts)
            })
        except Exception as e:
            logger.error(f"Failed to get context statistics in ContextLobe {self.name}: {str(e)}")
            base_status.update({
                "context_types": list(self.context_types),
                "total_contexts": "unknown"
            })
        
        return base_status