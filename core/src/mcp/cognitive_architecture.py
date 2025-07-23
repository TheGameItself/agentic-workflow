#!/usr/bin/env python3
"""
Cognitive Architecture for MCP Core System
Implements a brain-inspired cognitive architecture for advanced reasoning and decision making.
"""

import logging
import os
import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from .memory import MemoryManager
from .workflow import WorkflowManager
from .context_manager import ContextManager
from .creative_engine import CreativeEngine
from .learning_manager import LearningManager
from .performance_monitor import PerformanceMonitor

class Thought:
    """Represents a thought in the cognitive architecture."""
    
    def __init__(self, 
                 id: str, 
                 content: str, 
                 priority: float = 0.5, 
                 source: str = "system",
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a thought."""
        self.id = id
        self.content = content
        self.priority = priority
        self.source = source
        self.metadata = metadata or {}
        self.creation_time = time.time()
        self.last_accessed = self.creation_time

class CognitiveState:
    """Represents the cognitive state of the system."""
    
    RELAXED = "relaxed"
    FOCUSED = "focused"
    CREATIVE = "creative"
    LEARNING = "learning"
    STRESSED = "stressed"
    
    def __init__(self):
        """Initialize cognitive state."""
        self.current_state = self.RELAXED
        self.previous_state = None
        self.state_duration = 0.0
        self.state_start_time = time.time()
    
    def set_state(self, new_state: str):
        """Set the cognitive state."""
        if new_state == self.current_state:
            return
        
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_duration = 0.0
        self.state_start_time = time.time()
    
    def update_duration(self):
        """Update the duration of the current state."""
        self.state_duration = time.time() - self.state_start_time

class CognitiveContext:
    """Represents the cognitive context of the system."""
    
    def __init__(self):
        """Initialize cognitive context."""
        self.active_thoughts: List[Thought] = []
        self.working_memory: Dict[str, Any] = {}
        self.attention_focus: Optional[str] = None
        self.state = CognitiveState()

class CognitiveArchitecture:
    """
    Cognitive Architecture for MCP Core System.
    
    Implements a brain-inspired cognitive architecture for advanced reasoning and decision making:
    - Thought generation and processing
    - Cognitive state management
    - Working memory management
    - Attention focus control
    - Integration with memory, workflow, and creative systems
    """
    
    def __init__(self, 
                 memory_manager: Optional[MemoryManager] = None,
                 workflow_manager: Optional[WorkflowManager] = None,
                 context_manager: Optional[ContextManager] = None,
                 creative_engine: Optional[CreativeEngine] = None,
                 learning_manager: Optional[LearningManager] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None):
        """Initialize cognitive architecture."""
        self.memory_manager = memory_manager
        self.workflow_manager = workflow_manager
        self.context_manager = context_manager
        self.creative_engine = creative_engine
        self.learning_manager = learning_manager
        self.performance_monitor = performance_monitor
        
        # Cognitive context
        self.context = CognitiveContext()
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self._shutdown_event = threading.Event()
        
        # P2P integration
        self.p2p_integration = None
        
        self.logger = logging.getLogger("cognitive_architecture")
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize cognitive architecture."""
        try:
            self.logger.info("Initializing cognitive architecture")
            
            # Initialize components if not provided
            if self.memory_manager is None and 'memory_manager' in config:
                from .memory import create_memory_manager
                self.memory_manager = create_memory_manager(config['memory_manager'])
            
            if self.workflow_manager is None and 'workflow_manager' in config:
                from .workflow import create_workflow_manager
                self.workflow_manager = create_workflow_manager(config['workflow_manager'])
            
            if self.context_manager is None and 'context_manager' in config:
                from .context_manager import create_context_manager
                self.context_manager = create_context_manager(config['context_manager'])
            
            if self.creative_engine is None and 'creative_engine' in config:
                from .creative_engine import create_creative_engine
                self.creative_engine = create_creative_engine(config['creative_engine'])
            
            if self.learning_manager is None and 'learning_manager' in config:
                from .learning_manager import create_learning_manager
                self.learning_manager = create_learning_manager(config['learning_manager'])
            
            if self.performance_monitor is None and 'performance_monitor' in config:
                from .performance_monitor import create_performance_monitor
                self.performance_monitor = create_performance_monitor(config['performance_monitor'])
            
            # Initialize P2P integration if enabled
            if config and config.get('enable_p2p', False):
                await self._initialize_p2p_integration(config.get('p2p_config', {}))
            
            self.logger.info("Cognitive architecture initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing cognitive architecture: {e}")
            return False
    
    async def _initialize_p2p_integration(self, p2p_config: Dict[str, Any]) -> bool:
        """Initialize P2P integration."""
        try:
            self.logger.info("Initializing P2P integration for cognitive architecture")
            
            # Import P2P integration
            from .core_system_p2p_integration import integrate_p2p_with_core_system
            
            # Integrate P2P with cognitive architecture
            success = await integrate_p2p_with_core_system(self, p2p_config.get('bootstrap_nodes'))
            
            if success:
                self.logger.info("P2P integration initialized successfully")
            else:
                self.logger.error("Failed to initialize P2P integration")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing P2P integration: {e}")
            return False
    
    def start_processing(self) -> bool:
        """Start cognitive processing."""
        if self.is_processing:
            self.logger.warning("Cognitive processing already running")
            return True
        
        try:
            self.logger.info("Starting cognitive processing")
            
            self.is_processing = True
            self._shutdown_event.clear()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            self.logger.info("Cognitive processing started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting cognitive processing: {e}")
            self.is_processing = False
            return False
    
    def stop_processing(self) -> bool:
        """Stop cognitive processing."""
        if not self.is_processing:
            return True
        
        try:
            self.logger.info("Stopping cognitive processing")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Wait for processing thread to terminate
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            self.is_processing = False
            self.logger.info("Cognitive processing stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping cognitive processing: {e}")
            return False
    
    def _processing_loop(self):
        """Main cognitive processing loop."""
        self.logger.info("Cognitive processing loop started")
        
        while not self._shutdown_event.is_set():
            try:
                # Update cognitive state duration
                self.context.state.update_duration()
                
                # Process thoughts based on cognitive state
                if self.context.state.current_state == CognitiveState.RELAXED:
                    self._process_relaxed()
                elif self.context.state.current_state == CognitiveState.FOCUSED:
                    self._process_focused()
                elif self.context.state.current_state == CognitiveState.CREATIVE:
                    self._process_creative()
                elif self.context.state.current_state == CognitiveState.LEARNING:
                    self._process_learning()
                elif self.context.state.current_state == CognitiveState.STRESSED:
                    self._process_stressed()
                
                # Clean up working memory
                self._cleanup_working_memory()
                
                # Sleep to avoid high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in cognitive processing loop: {e}")
                time.sleep(1.0)
        
        self.logger.info("Cognitive processing loop terminated")
    
    def add_thought(self, content: str, priority: float = 0.5, source: str = "system", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a thought to the cognitive architecture."""
        thought_id = f"thought_{int(time.time())}_{len(self.context.active_thoughts)}"
        
        thought = Thought(
            id=thought_id,
            content=content,
            priority=priority,
            source=source,
            metadata=metadata
        )
        
        self.context.active_thoughts.append(thought)
        self.logger.debug(f"Added thought: {thought_id} (priority: {priority})")
        
        return thought_id
    
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Get a thought by ID."""
        for thought in self.context.active_thoughts:
            if thought.id == thought_id:
                thought.last_accessed = time.time()
                return thought
        return None
    
    def remove_thought(self, thought_id: str) -> bool:
        """Remove a thought by ID."""
        for i, thought in enumerate(self.context.active_thoughts):
            if thought.id == thought_id:
                del self.context.active_thoughts[i]
                self.logger.debug(f"Removed thought: {thought_id}")
                return True
        return False
    
    def set_cognitive_state(self, state: str) -> bool:
        """Set the cognitive state."""
        if state not in [
            CognitiveState.RELAXED,
            CognitiveState.FOCUSED,
            CognitiveState.CREATIVE,
            CognitiveState.LEARNING,
            CognitiveState.STRESSED
        ]:
            self.logger.warning(f"Invalid cognitive state: {state}")
            return False
        
        self.context.state.set_state(state)
        self.logger.info(f"Cognitive state set to: {state}")
        return True
    
    def set_attention_focus(self, thought_id: Optional[str]) -> bool:
        """Set the attention focus."""
        if thought_id is None:
            self.context.attention_focus = None
            self.logger.debug("Attention focus cleared")
            return True
        
        # Check if thought exists
        thought = self.get_thought(thought_id)
        if thought is None:
            self.logger.warning(f"Cannot set attention focus: thought {thought_id} not found")
            return False
        
        self.context.attention_focus = thought_id
        self.logger.debug(f"Attention focus set to: {thought_id}")
        return True
    
    def get_working_memory(self, key: str) -> Any:
        """Get a value from working memory."""
        return self.context.working_memory.get(key)
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """Set a value in working memory."""
        self.context.working_memory[key] = value
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current status of the cognitive architecture."""
        return {
            'is_processing': self.is_processing,
            'cognitive_state': self.context.state.current_state,
            'state_duration': self.context.state.state_duration,
            'active_thoughts': len(self.context.active_thoughts),
            'attention_focus': self.context.attention_focus,
            'working_memory_size': len(self.context.working_memory),
            'p2p_enabled': hasattr(self, 'p2p_integration') and self.p2p_integration is not None
        }
    
    async def shutdown(self) -> bool:
        """Shutdown cognitive architecture."""
        try:
            self.logger.info("Shutting down cognitive architecture")
            
            # Stop processing
            self.stop_processing()
            
            # Shutdown P2P integration if enabled
            if self.p2p_integration:
                try:
                    await self.p2p_integration.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down P2P integration: {e}")
            
            self.logger.info("Cognitive architecture shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down cognitive architecture: {e}")
            return False
    
    # Cognitive state processing methods
    def _process_relaxed(self):
        """Process thoughts in relaxed state."""
        # In relaxed state, process all thoughts equally
        pass
    
    def _process_focused(self):
        """Process thoughts in focused state."""
        # In focused state, prioritize the attention focus
        if self.context.attention_focus:
            focused_thought = self.get_thought(self.context.attention_focus)
            if focused_thought:
                # Increase priority of focused thought
                focused_thought.priority = min(1.0, focused_thought.priority + 0.1)
    
    def _process_creative(self):
        """Process thoughts in creative state."""
        # In creative state, generate new connections between thoughts
        if self.creative_engine and len(self.context.active_thoughts) >= 2:
            # Creative processing would go here
            pass
    
    def _process_learning(self):
        """Process thoughts in learning state."""
        # In learning state, consolidate thoughts into memory
        if self.memory_manager and self.context.active_thoughts:
            # Learning processing would go here
            pass
    
    def _process_stressed(self):
        """Process thoughts in stressed state."""
        # In stressed state, focus on essential thoughts and reduce complexity
        if self.context.active_thoughts:
            # Keep only highest priority thought
            top_thought = max(self.context.active_thoughts, key=lambda t: t.priority)
            self.context.active_thoughts = [top_thought]
            self.set_attention_focus(top_thought.id)
    
    def _cleanup_working_memory(self):
        """Clean up working memory."""
        # Simple implementation: limit size
        if len(self.context.working_memory) > 20:
            # Remove oldest entries (not implemented here)
            # Would require tracking creation time for each entry
            pass

# Global cognitive architecture instance
_cognitive_architecture: Optional[CognitiveArchitecture] = None

def get_cognitive_architecture() -> CognitiveArchitecture:
    """Get the global cognitive architecture instance."""
    global _cognitive_architecture
    if _cognitive_architecture is None:
        _cognitive_architecture = CognitiveArchitecture()
    return _cognitive_architecture