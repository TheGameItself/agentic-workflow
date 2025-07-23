#!/usr/bin/env python3
"""
P2P Components Initialization for MCP Core System
Initializes and manages P2P components for the core system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Import P2P components
from .p2p_core_integration import P2PCoreIntegration, create_p2p_core_integration
from .p2p_network_bus import P2PNetworkBus, MessageBusEvent, MessagePriority

async def initialize_p2p_components(core_system, bootstrap_nodes: Optional[List[Tuple[str, int]]] = None) -> bool:
    """
    Initialize P2P components for the core system.
    
    Args:
        core_system: The core system instance
        bootstrap_nodes: Optional list of bootstrap nodes for P2P network
        
    Returns:
        bool: True if initialization successful
    """
    logger = logging.getLogger("p2p_components_init")
    
    try:
        logger.info("Initializing P2P components")
        
        # Create P2P integration
        spinal_column = getattr(core_system, 'spinal_column', None)
        p2p_integration = create_p2p_core_integration(core_system, spinal_column)
        
        # Initialize P2P integration
        success = await p2p_integration.initialize(bootstrap_nodes)
        if not success:
            logger.error("Failed to initialize P2P integration")
            return False
        
        # Set P2P integration in core system
        core_system.p2p_integration = p2p_integration
        
        logger.info("P2P components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing P2P components: {e}")
        return False

async def update_p2p_components(core_system, metrics: Dict[str, Any]) -> bool:
    """
    Update P2P components with system metrics.
    
    Args:
        core_system: The core system instance
        metrics: System metrics to publish
        
    Returns:
        bool: True if update successful
    """
    if not hasattr(core_system, 'p2p_integration') or core_system.p2p_integration is None:
        return False
    
    try:
        # Publish metrics to P2P network
        await core_system.p2p_integration.network_bus.publish(
            event_type=MessageBusEvent.STATUS_UPDATE,
            content={
                'metrics': metrics,
                'timestamp': time.time()
            },
            priority=MessagePriority.LOW
        )
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating P2P components: {e}")
        return False

def get_p2p_components_status(core_system) -> Dict[str, Any]:
    """
    Get status of P2P components.
    
    Args:
        core_system: The core system instance
        
    Returns:
        dict: P2P components status
    """
    if not hasattr(core_system, 'p2p_integration') or core_system.p2p_integration is None:
        return {'enabled': False}
    
    try:
        # Get integration status
        status = core_system.p2p_integration.get_integration_status()
        status['enabled'] = True
        
        return status
        
    except Exception as e:
        logging.error(f"Error getting P2P components status: {e}")
        return {'enabled': True, 'error': str(e)}

async def shutdown_p2p_components(core_system) -> bool:
    """
    Shutdown P2P components.
    
    Args:
        core_system: The core system instance
        
    Returns:
        bool: True if shutdown successful
    """
    if not hasattr(core_system, 'p2p_integration') or core_system.p2p_integration is None:
        return True
    
    try:
        # Shutdown P2P integration
        success = await core_system.p2p_integration.shutdown()
        
        # Remove P2P integration from core system
        if success:
            core_system.p2p_integration = None
        
        return success
        
    except Exception as e:
        logging.error(f"Error shutting down P2P components: {e}")
        return False