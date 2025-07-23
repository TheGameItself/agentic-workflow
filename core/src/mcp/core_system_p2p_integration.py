#!/usr/bin/env python3
"""
Core System P2P Integration for MCP Core System
Integrates P2P network functionality into the core system.
"""

import asyncio
import logging
from typing import Optional, List, Tuple, Dict, Any

# Import P2P components
from .p2p_components_init import (
    initialize_p2p_components,
    update_p2p_components,
    get_p2p_components_status,
    shutdown_p2p_components
)

class CoreSystemP2PIntegration:
    """
    Core System P2P Integration.
    
    Provides methods for integrating P2P network functionality into the core system:
    - Initialization and shutdown of P2P components
    - P2P component status monitoring
    - P2P event handling
    - Integration with core system lifecycle
    """
    
    @staticmethod
    async def integrate_p2p_with_core(core_system, bootstrap_nodes: Optional[List[Tuple[str, int]]] = None) -> bool:
        """
        Integrate P2P functionality with the core system.
        
        Args:
            core_system: The MCPCoreSystem instance
            bootstrap_nodes: Optional list of bootstrap nodes for P2P network
            
        Returns:
            bool: True if integration successful
        """
        logger = logging.getLogger("core_system_p2p_integration")
        
        try:
            # Initialize P2P components
            success = await initialize_p2p_components(core_system, bootstrap_nodes)
            if not success:
                logger.error("Failed to initialize P2P components")
                return False
            
            # Add P2P update to core system monitoring loop
            original_monitoring_loop = core_system._monitoring_loop
            
            async def enhanced_monitoring_loop():
                """Enhanced monitoring loop with P2P updates."""
                while not core_system._shutdown_event.is_set():
                    try:
                        # Call original monitoring loop logic
                        await original_monitoring_loop()
                        
                        # Update P2P components with metrics
                        metrics = {
                            'cpu_usage': core_system.metrics.cpu_usage,
                            'memory_usage': core_system.metrics.memory_usage,
                            'active_lobes': core_system.metrics.active_lobes,
                            'uptime': core_system.metrics.uptime,
                            'timestamp': core_system.metrics.last_updated.timestamp()
                        }
                        await update_p2p_components(core_system, metrics)
                        
                        # Sleep for a while
                        await asyncio.sleep(30)  # Monitor every 30 seconds
                        
                    except Exception as e:
                        logger.error(f"Error in enhanced monitoring loop: {e}")
                        await asyncio.sleep(60)  # Wait longer on error
            
            # Replace monitoring loop with enhanced version
            core_system._monitoring_loop = enhanced_monitoring_loop
            
            # Add P2P shutdown to core system shutdown
            original_shutdown = core_system.shutdown
            
            async def enhanced_shutdown() -> bool:
                """Enhanced shutdown with P2P component shutdown."""
                try:
                    # Shutdown P2P components first
                    await shutdown_p2p_components(core_system)
                    
                    # Call original shutdown
                    return await original_shutdown()
                    
                except Exception as e:
                    logger.error(f"Error in enhanced shutdown: {e}")
                    return False
            
            # Replace shutdown with enhanced version
            core_system.shutdown = enhanced_shutdown
            
            # Add P2P status to core system status
            original_get_status = core_system.get_status
            
            def enhanced_get_status() -> Dict[str, Any]:
                """Enhanced get_status with P2P component status."""
                status = original_get_status()
                
                # Add P2P status
                status['p2p_components'] = get_p2p_components_status(core_system)
                
                return status
            
            # Replace get_status with enhanced version
            core_system.get_status = enhanced_get_status
            
            logger.info("P2P integration with core system completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate P2P with core system: {e}")
            return False
    
    @staticmethod
    def is_p2p_enabled(core_system) -> bool:
        """
        Check if P2P functionality is enabled in the core system.
        
        Args:
            core_system: The MCPCoreSystem instance
            
        Returns:
            bool: True if P2P is enabled and initialized
        """
        return (hasattr(core_system, 'p2p_integration') and 
                core_system.p2p_integration is not None and 
                getattr(core_system.p2p_integration, 'is_initialized', False))
    
    @staticmethod
    def get_p2p_status(core_system) -> Dict[str, Any]:
        """
        Get P2P status from the core system.
        
        Args:
            core_system: The MCPCoreSystem instance
            
        Returns:
            dict: P2P status information
        """
        return get_p2p_components_status(core_system)

# Convenience function
async def integrate_p2p_with_core_system(core_system, bootstrap_nodes: Optional[List[Tuple[str, int]]] = None) -> bool:
    """
    Integrate P2P functionality with the core system.
    
    Args:
        core_system: The MCPCoreSystem instance
        bootstrap_nodes: Optional list of bootstrap nodes for P2P network
        
    Returns:
        bool: True if integration successful
    """
    return await CoreSystemP2PIntegration.integrate_p2p_with_core(core_system, bootstrap_nodes)