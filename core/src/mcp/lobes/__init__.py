#!/usr/bin/env python3
"""
LOAB (Learning Operations and Adaptive Behaviors) Architecture
Implements brain-inspired modular components for the MCP system.
"""

import logging
import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

class LobeStatus(Enum):
    """Status of a lobe in the LOAB architecture."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class LobeType(Enum):
    """Types of lobes in the LOAB architecture."""
    MEMORY = "memory"           # Memory management and retrieval
    WORKFLOW = "workflow"       # Workflow orchestration
    CONTEXT = "context"         # Context management
    TASK = "task"               # Task management
    PROJECT = "project"         # Project management
    PERCEPTION = "perception"   # Input processing and perception
    REASONING = "reasoning"     # Reasoning and decision making
    LEARNING = "learning"       # Learning and adaptation
    CREATIVE = "creative"       # Creative generation
    EXECUTIVE = "executive"     # Executive control and coordination

class BaseLobe(ABC):
    """
    Base class for all lobes in the LOAB architecture.
    
    LOAB (Learning Operations and Adaptive Behaviors) is a brain-inspired
    modular architecture that organizes functionality into specialized lobes,
    each responsible for specific cognitive functions.
    """
    
    def __init__(self, lobe_id: str, lobe_type: LobeType):
        """Initialize base lobe."""
        self.lobe_id = lobe_id
        self.lobe_type = lobe_type
        self.status = LobeStatus.INACTIVE
        self.logger = logging.getLogger(f"loab.{lobe_id}")
        
        # Lobe state
        self.initialization_time = None
        self.last_active_time = None
        self.error_count = 0
        self.performance_metrics = {}
        
        # Inter-lobe communication
        self.input_queue = []
        self.output_queue = []
        self.connected_lobes = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the lobe."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the lobe."""
        pass
    
    def connect_to_lobe(self, lobe_id: str, lobe: 'BaseLobe') -> bool:
        """Connect to another lobe for communication."""
        if lobe_id in self.connected_lobes:
            self.logger.warning(f"Lobe {lobe_id} already connected, overwriting")
        
        self.connected_lobes[lobe_id] = lobe
        self.logger.debug(f"Connected to lobe: {lobe_id}")
        return True
    
    def send_to_lobe(self, lobe_id: str, data: Any) -> bool:
        """Send data to a connected lobe."""
        if lobe_id not in self.connected_lobes:
            self.logger.error(f"Lobe {lobe_id} not connected")
            return False
        
        try:
            self.connected_lobes[lobe_id].receive_data(self.lobe_id, data)
            return True
        except Exception as e:
            self.logger.error(f"Error sending data to lobe {lobe_id}: {e}")
            return False
    
    def receive_data(self, sender_id: str, data: Any) -> bool:
        """Receive data from another lobe."""
        try:
            self.input_queue.append({
                'sender': sender_id,
                'data': data,
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            self.logger.error(f"Error receiving data from lobe {sender_id}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current lobe status."""
        return {
            'lobe_id': self.lobe_id,
            'lobe_type': self.lobe_type.value,
            'status': self.status.value,
            'initialization_time': self.initialization_time,
            'last_active_time': self.last_active_time,
            'error_count': self.error_count,
            'performance_metrics': self.performance_metrics,
            'input_queue_size': len(self.input_queue),
            'output_queue_size': len(self.output_queue),
            'connected_lobes': list(self.connected_lobes.keys())
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the lobe."""
        healthy = self.status in [LobeStatus.ACTIVE, LobeStatus.INITIALIZING]
        
        return {
            'healthy': healthy,
            'status': self.status.value,
            'error_count': self.error_count,
            'last_active': self.last_active_time
        }

class LobeRegistry:
    """Registry for managing lobes in the LOAB architecture."""
    
    def __init__(self):
        """Initialize lobe registry."""
        self._lobes: Dict[str, BaseLobe] = {}
        self.logger = logging.getLogger("loab.registry")
    
    def register_lobe(self, lobe: BaseLobe) -> bool:
        """Register a lobe in the registry."""
        if lobe.lobe_id in self._lobes:
            self.logger.warning(f"Lobe {lobe.lobe_id} already registered, overwriting")
        
        self._lobes[lobe.lobe_id] = lobe
        self.logger.info(f"Registered lobe: {lobe.lobe_id} ({lobe.lobe_type.value})")
        return True
    
    def get_lobe(self, lobe_id: str) -> Optional[BaseLobe]:
        """Get a lobe by ID."""
        return self._lobes.get(lobe_id)
    
    def get_lobes_by_type(self, lobe_type: LobeType) -> List[BaseLobe]:
        """Get all lobes of a specific type."""
        return [lobe for lobe in self._lobes.values() if lobe.lobe_type == lobe_type]
    
    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered lobes."""
        results = {}
        
        for lobe_id, lobe in self._lobes.items():
            try:
                success = lobe.initialize()
                results[lobe_id] = success
                
                if success:
                    self.logger.info(f"Initialized lobe: {lobe_id}")
                else:
                    self.logger.error(f"Failed to initialize lobe: {lobe_id}")
                    
            except Exception as e:
                self.logger.error(f"Error initializing lobe {lobe_id}: {e}")
                results[lobe_id] = False
        
        return results
    
    def shutdown_all(self) -> Dict[str, bool]:
        """Shutdown all registered lobes."""
        results = {}
        
        for lobe_id, lobe in self._lobes.items():
            try:
                success = lobe.shutdown()
                results[lobe_id] = success
                
                if success:
                    self.logger.info(f"Shutdown lobe: {lobe_id}")
                else:
                    self.logger.error(f"Failed to shutdown lobe: {lobe_id}")
                    
            except Exception as e:
                self.logger.error(f"Error shutting down lobe {lobe_id}: {e}")
                results[lobe_id] = False
        
        return results
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all registered lobes."""
        results = {}
        
        for lobe_id, lobe in self._lobes.items():
            try:
                health = lobe.health_check()
                results[lobe_id] = health
            except Exception as e:
                self.logger.error(f"Error checking health of lobe {lobe_id}: {e}")
                results[lobe_id] = {'healthy': False, 'error': str(e)}
        
        return results
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of the lobe registry."""
        return {
            'total_lobes': len(self._lobes),
            'lobe_types': {
                lobe_type.value: len([l for l in self._lobes.values() if l.lobe_type == lobe_type])
                for lobe_type in LobeType
            },
            'active_lobes': len([l for l in self._lobes.values() if l.status == LobeStatus.ACTIVE]),
            'error_lobes': len([l for l in self._lobes.values() if l.status == LobeStatus.ERROR])
        }

# Global registry instance
_lobe_registry: Optional[LobeRegistry] = None

def get_lobe_registry() -> LobeRegistry:
    """Get the global lobe registry instance."""
    global _lobe_registry
    if _lobe_registry is None:
        _lobe_registry = LobeRegistry()
    return _lobe_registry