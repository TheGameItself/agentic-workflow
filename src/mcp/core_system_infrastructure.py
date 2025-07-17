"""
Core System Infrastructure: Foundational architecture for brain-inspired modular system.

This module implements the core infrastructure for the MCP System Upgrade,
providing a brain-inspired modular architecture with hormone-based communication
between lobes. It serves as the central integration point for all brain-inspired
components.

References:
- idea.txt (brain-inspired architecture)
- cross-implementation.md (hormone system integration)
- ARCHITECTURE.md (system design principles)
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

# Import core components
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.hormone_system_controller import HormoneSystemController
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus


@dataclass
class LobeRegistration:
    """Registration information for a lobe in the system."""
    name: str
    instance: Any
    position: Tuple[float, float, float]
    connected_lobes: List[str]
    is_left_hemisphere: bool
    is_experimental: bool
    capabilities: Set[str]
    hormone_receptors: Dict[str, float]


class CoreSystemInfrastructure:
    """
    Core infrastructure for the brain-inspired modular system.
    
    This class serves as the central integration point for all brain-inspired
    components, managing lobe registration, hormone communication, and system-wide
    state aggregation.
    """
    
    def __init__(self):
        """Initialize the core system infrastructure."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CoreSystemInfrastructure")
        
        # Create event bus for communication
        self.event_bus = LobeEventBus()
        
        # Create hormone system controller
        self.hormone_controller = HormoneSystemController(event_bus=self.event_bus)
        
        # Create brain state aggregator
        self.brain_state_aggregator = BrainStateAggregator(
            hormone_engine=self.hormone_controller,
            event_bus=self.event_bus
        )
        
        # Registered lobes
        self.lobes: Dict[str, LobeRegistration] = {}
        
        # System state
        self.running = False
        self.update_thread = None
        self.update_interval = 0.5  # seconds
        
        self.logger.info("Core System Infrastructure initialized")
    
    def register_lobe(self, 
                     name: str, 
                     instance: Any, 
                     position: Tuple[float, float, float] = (0, 0, 0),
                     connected_lobes: List[str] = None,
                     is_left_hemisphere: bool = True,
                     is_experimental: bool = False,
                     capabilities: Set[str] = None,
                     hormone_receptors: Dict[str, float] = None) -> None:
        """
        Register a lobe with the core system.
        
        Args:
            name: Name of the lobe
            instance: Lobe instance
            position: 3D position for spatial hormone diffusion
            connected_lobes: List of directly connected lobes
            is_left_hemisphere: Whether the lobe is in the left hemisphere
            is_experimental: Whether the lobe is experimental
            capabilities: Set of capabilities provided by the lobe
            hormone_receptors: Initial hormone receptor sensitivities
        """
        if connected_lobes is None:
            connected_lobes = []
        
        if capabilities is None:
            capabilities = set()
            
        if hormone_receptors is None:
            hormone_receptors = {}
            
        # Register with hormone system
        self.hormone_controller.register_lobe(name, position, connected_lobes)
        
        # Store lobe registration
        self.lobes[name] = LobeRegistration(
            name=name,
            instance=instance,
            position=position,
            connected_lobes=connected_lobes,
            is_left_hemisphere=is_left_hemisphere,
            is_experimental=is_experimental,
            capabilities=capabilities,
            hormone_receptors=hormone_receptors
        )
        
        # Register with brain state aggregator
        if hasattr(self.brain_state_aggregator, 'lobes'):
            self.brain_state_aggregator.lobes[name] = instance
        
        # Set up event handlers if the lobe has them
        if hasattr(instance, 'register_event_handlers'):
            instance.register_event_handlers(self.event_bus)
            
        # Initialize hormone receptors
        for hormone, sensitivity in hormone_receptors.items():
            self.hormone_controller.adapt_receptor_sensitivity(name, hormone, sensitivity)
            
        self.logger.info(f"Registered lobe '{name}' with the core system")
    
    def start(self) -> None:
        """Start the core system."""
        if self.running:
            self.logger.warning("Core system is already running")
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Core system started")
    
    def stop(self) -> None:
        """Stop the core system."""
        if not self.running:
            self.logger.warning("Core system is not running")
            return
            
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            
        self.logger.info("Core system stopped")
    
    def _update_loop(self) -> None:
        """Main update loop for the core system."""
        while self.running:
            try:
                # Process hormone cascades
                self.hormone_controller.process_hormone_cascades()
                
                # Update brain state aggregator
                self.brain_state_aggregator.update_buffers()
                
                # Update all lobes
                self._update_lobes()
                
                # Sleep for update interval
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in core system update loop: {e}")
    
    def _update_lobes(self) -> None:
        """Update all registered lobes."""
        for name, registration in self.lobes.items():
            try:
                # Get context package for this lobe
                context_package = self.brain_state_aggregator.get_context_package(name)
                
                # Update lobe if it has an update method
                if hasattr(registration.instance, 'update'):
                    registration.instance.update(context_package)
            except Exception as e:
                self.logger.error(f"Error updating lobe '{name}': {e}")
    
    def get_hormone_levels(self) -> Dict[str, float]:
        """Get current hormone levels."""
        return self.hormone_controller.get_levels()
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state."""
        return self.brain_state_aggregator.get_context_package("global")
    
    def release_hormone(self, source_lobe: str, hormone: str, quantity: float, 
                       context: Optional[Dict] = None) -> None:
        """
        Release a hormone from a source lobe.
        
        Args:
            source_lobe: Name of the lobe releasing the hormone
            hormone: Name of the hormone to release
            quantity: Amount of hormone to release (0.0-1.0)
            context: Optional context information for learning
        """
        self.hormone_controller.release_hormone(source_lobe, hormone, quantity, context)
    
    def adapt_receptor_sensitivity(self, lobe: str, hormone: str, performance: float) -> None:
        """
        Adapt receptor sensitivity based on performance feedback.
        
        Args:
            lobe: Name of the lobe
            hormone: Name of the hormone
            performance: Performance feedback (-1.0 to 1.0)
        """
        self.hormone_controller.adapt_receptor_sensitivity(lobe, hormone, performance)
    
    def learn_optimal_hormone_profiles(self, context: Dict) -> Dict[str, float]:
        """
        Learn optimal hormone profiles for specific contexts.
        
        Args:
            context: Context information
            
        Returns:
            Dictionary of optimal hormone levels for the context
        """
        return self.hormone_controller.learn_optimal_hormone_profiles(context)
    
    def emit_event(self, event_type: str, data: Any, signal_type: str = "excitatory", 
                  context: Dict = None, priority: int = 0) -> None:
        """
        Emit an event on the event bus.
        
        Args:
            event_type: Type of event
            data: Event data
            signal_type: Signal type (excitatory, inhibitory, modulatory)
            context: Event context
            priority: Event priority
        """
        if context is None:
            context = {}
            
        self.event_bus.publish(event_type, data, signal_type, context, priority)
    
    def get_lobe_capabilities(self) -> Dict[str, Set[str]]:
        """
        Get capabilities provided by all registered lobes.
        
        Returns:
            Dictionary mapping capability names to sets of lobes providing them
        """
        capabilities: Dict[str, Set[str]] = {}
        
        for name, registration in self.lobes.items():
            for capability in registration.capabilities:
                if capability not in capabilities:
                    capabilities[capability] = set()
                capabilities[capability].add(name)
                
        return capabilities
    
    def get_lobes_by_hemisphere(self, left_hemisphere: bool = True) -> List[str]:
        """
        Get lobes in the specified hemisphere.
        
        Args:
            left_hemisphere: Whether to get lobes in the left hemisphere
            
        Returns:
            List of lobe names in the specified hemisphere
        """
        return [name for name, registration in self.lobes.items() 
                if registration.is_left_hemisphere == left_hemisphere]
    
    def get_experimental_lobes(self) -> List[str]:
        """
        Get experimental lobes.
        
        Returns:
            List of experimental lobe names
        """
        return [name for name, registration in self.lobes.items() 
                if registration.is_experimental]


# Example usage
if __name__ == "__main__":
    # Create core system
    core_system = CoreSystemInfrastructure()
    
    try:
        # Start the system
        core_system.start()
        
        # Wait for a bit
        time.sleep(5)
        
        # Get hormone levels
        hormone_levels = core_system.get_hormone_levels()
        print("Current hormone levels:")
        for hormone, level in hormone_levels.items():
            if level > 0.1:  # Only show significant levels
                print(f"  {hormone}: {level:.2f}")
                
    finally:
        # Stop the system
        core_system.stop()
"""