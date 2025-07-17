"""
Integration script for HormoneSystemController with BrainStateAggregator.

This script demonstrates how to integrate the HormoneSystemController
with the existing BrainStateAggregator for comprehensive brain-inspired
hormone communication between lobes. It now leverages the CoreSystemInfrastructure
for a more robust and complete implementation.
"""

import logging
from typing import Dict, Any, Optional
from mcp.hormone_system_controller import HormoneSystemController
from mcp.brain_state_aggregator import BrainStateAggregator
from mcp.lobes.experimental.lobe_event_bus import LobeEventBus
from mcp.core_system_infrastructure import CoreSystemInfrastructure

class HormoneSystemIntegration:
    """
    Integration class for connecting HormoneSystemController with BrainStateAggregator.
    
    This class now leverages the CoreSystemInfrastructure for a more robust
    and complete implementation of the hormone system integration.
    """
    
    def __init__(self):
        """Initialize the hormone system integration."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HormoneSystemIntegration")
        
        # Create core system infrastructure
        self.core_system = CoreSystemInfrastructure()
        
        # Get references to core components
        self.event_bus = self.core_system.event_bus
        self.hormone_controller = self.core_system.hormone_controller
        self.brain_state_aggregator = self.core_system.brain_state_aggregator
        
        # Register lobes with the hormone system
        self._register_lobes()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        self.logger.info("HormoneSystemIntegration initialized")
        
    def _register_lobes(self):
        """Register lobes with the hormone system controller."""
        # Register core lobes with 3D positions for spatial diffusion
        self.hormone_controller.register_lobe("task_management", position=(0, 0, 0))
        self.hormone_controller.register_lobe("memory", position=(1, 0, 0))
        self.hormone_controller.register_lobe("pattern_recognition", position=(0, 1, 0))
        self.hormone_controller.register_lobe("decision_making", position=(1, 1, 0))
        self.hormone_controller.register_lobe("context_management", position=(0.5, 0.5, 0))
        self.hormone_controller.register_lobe("scientific_process", position=(1, 0, 1))
        self.hormone_controller.register_lobe("error_detection", position=(0, 1, 1))
        self.hormone_controller.register_lobe("social_intelligence", position=(0.5, 1, 0.5))
        
        # Connect lobes for paracrine signaling
        self.hormone_controller.lobes["task_management"].connected_lobes = [
            "memory", "decision_making", "context_management", "social_intelligence"
        ]
        self.hormone_controller.lobes["memory"].connected_lobes = [
            "task_management", "pattern_recognition", "scientific_process", "social_intelligence"
        ]
        self.hormone_controller.lobes["pattern_recognition"].connected_lobes = [
            "memory", "decision_making", "scientific_process"
        ]
        self.hormone_controller.lobes["decision_making"].connected_lobes = [
            "task_management", "pattern_recognition", "context_management", "social_intelligence"
        ]
        self.hormone_controller.lobes["context_management"].connected_lobes = [
            "task_management", "decision_making", "social_intelligence"
        ]
        self.hormone_controller.lobes["scientific_process"].connected_lobes = [
            "memory", "pattern_recognition"
        ]
        self.hormone_controller.lobes["error_detection"].connected_lobes = [
            "task_management", "context_management"
        ]
        self.hormone_controller.lobes["social_intelligence"].connected_lobes = [
            "task_management", "decision_making", "context_management", "memory"
        ]
        
        self.logger.info("Registered lobes with hormone system")
        
    def _setup_event_handlers(self):
        """Set up event handlers for inter-lobe communication."""
        # Subscribe to events
        self.event_bus.subscribe("task_completed", self._handle_task_completed)
        self.event_bus.subscribe("error_detected", self._handle_error_detected)
        self.event_bus.subscribe("learning_event", self._handle_learning_event)
        self.event_bus.subscribe("decision_made", self._handle_decision_made)
        self.event_bus.subscribe("social_interaction", self._handle_social_interaction)
        self.event_bus.subscribe("trust_signal", self._handle_trust_signal)
        
        self.logger.info("Set up event handlers")
        
    def _handle_task_completed(self, data: Dict[str, Any]):
        """Handle task completion events by releasing dopamine."""
        task_importance = data.get("importance", 0.5)
        self.hormone_controller.release_hormone(
            "task_management", 
            "dopamine", 
            0.8 * task_importance,
            context={"event_type": "task_completed", "task_data": data}
        )
        self.logger.info(f"Released dopamine for task completion (importance: {task_importance})")
        
    def _handle_error_detected(self, data: Dict[str, Any]):
        """Handle error detection events by releasing cortisol and histamine."""
        error_severity = data.get("severity", 0.5)
        self.hormone_controller.release_hormone(
            "error_detection", 
            "cortisol", 
            0.7 * error_severity,
            context={"event_type": "error_detected", "error_data": data}
        )
        self.hormone_controller.release_hormone(
            "error_detection", 
            "histamine", 
            0.9 * error_severity,
            context={"event_type": "error_detected", "error_data": data}
        )
        self.logger.info(f"Released cortisol and histamine for error detection (severity: {error_severity})")
        
    def _handle_learning_event(self, data: Dict[str, Any]):
        """Handle learning events by releasing acetylcholine and growth hormone."""
        learning_importance = data.get("importance", 0.5)
        self.hormone_controller.release_hormone(
            "memory", 
            "acetylcholine", 
            0.7 * learning_importance,
            context={"event_type": "learning_event", "learning_data": data}
        )
        self.hormone_controller.release_hormone(
            "memory", 
            "growth_hormone", 
            0.6 * learning_importance,
            context={"event_type": "learning_event", "learning_data": data}
        )
        self.logger.info(f"Released acetylcholine and growth hormone for learning (importance: {learning_importance})")
        
    def _handle_decision_made(self, data: Dict[str, Any]):
        """Handle decision events by releasing serotonin."""
        decision_confidence = data.get("confidence", 0.5)
        self.hormone_controller.release_hormone(
            "decision_making", 
            "serotonin", 
            0.7 * decision_confidence,
            context={"event_type": "decision_made", "decision_data": data}
        )
        self.logger.info(f"Released serotonin for decision making (confidence: {decision_confidence})")
        
    def _handle_social_interaction(self, data: Dict[str, Any]):
        """Handle social interaction events by releasing oxytocin and vasopressin."""
        interaction_quality = data.get("quality", 0.5)
        interaction_type = data.get("type", "general")
        
        # Release oxytocin for positive social interactions
        if interaction_quality > 0.6:
            self.hormone_controller.release_hormone(
                "social_intelligence",
                "oxytocin",
                0.8 * interaction_quality,
                context={"event_type": "social_interaction", "interaction_data": data}
            )
            self.logger.info(f"Released oxytocin for positive social interaction (quality: {interaction_quality})")
        
        # Release vasopressin for memory-forming social interactions
        if interaction_type in ["learning", "teaching", "collaborative"]:
            self.hormone_controller.release_hormone(
                "social_intelligence",
                "vasopressin",
                0.7 * interaction_quality,
                context={"event_type": "social_interaction", "interaction_data": data}
            )
            self.logger.info(f"Released vasopressin for {interaction_type} interaction")
        
    def _handle_trust_signal(self, data: Dict[str, Any]):
        """Handle trust signal events by releasing prolactin and oxytocin."""
        trust_level = data.get("trust_level", 0.5)
        signal_type = data.get("type", "general")
        
        # Release prolactin for protective care situations
        if signal_type in ["protective", "supportive", "guidance"]:
            self.hormone_controller.release_hormone(
                "social_intelligence",
                "prolactin",
                0.8 * trust_level,
                context={"event_type": "trust_signal", "trust_data": data}
            )
            self.logger.info(f"Released prolactin for {signal_type} trust signal (level: {trust_level})")
        
        # Release oxytocin for trust-building situations
        if trust_level > 0.7:
            self.hormone_controller.release_hormone(
                "social_intelligence",
                "oxytocin",
                0.6 * trust_level,
                context={"event_type": "trust_signal", "trust_data": data}
            )
            self.logger.info(f"Released oxytocin for high trust signal (level: {trust_level})")
        
    def start(self):
        """Start the integration system."""
        self.core_system.start()
        self.logger.info("HormoneSystemIntegration started")
        
    def update(self):
        """Update the integration system."""
        # The core system handles updates automatically in its own thread
        # This method is kept for backward compatibility
        
        # Get cascade results for logging
        cascade_result = self.hormone_controller.process_hormone_cascades()
        
        # Log cascade results
        if cascade_result.triggered_cascades:
            self.logger.info(f"Triggered cascades: {cascade_result.triggered_cascades}")
        if cascade_result.emergent_effects:
            self.logger.info(f"Emergent effects: {cascade_result.emergent_effects}")
            
    def get_hormone_levels(self) -> Dict[str, float]:
        """Get current hormone levels."""
        return self.core_system.get_hormone_levels()
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state."""
        return self.core_system.get_brain_state()
    
    def adapt_receptor_sensitivity(self, lobe: str, hormone: str, performance: float):
        """Adapt receptor sensitivity based on performance feedback."""
        self.core_system.adapt_receptor_sensitivity(lobe, hormone, performance)
        
    def learn_optimal_hormone_profiles(self, context: Dict) -> Dict[str, float]:
        """Learn optimal hormone profiles for specific contexts."""
        return self.core_system.learn_optimal_hormone_profiles(context)
    
    def emit_event(self, event_type: str, data: Any, signal_type: str = "excitatory", 
                  context: Dict = None, priority: int = 0):
        """Emit an event on the event bus."""
        self.core_system.emit_event(event_type, data, signal_type, context, priority)
    
    def shutdown(self):
        """Shut down the integration system."""
        self.core_system.stop()
        self.logger.info("HormoneSystemIntegration shut down")


# Example usage
if __name__ == "__main__":
    # Create integration
    integration = HormoneSystemIntegration()
    
    try:
        # Start the system
        integration.start()
        
        # Simulate some events
        integration.event_bus.emit("task_completed", {"importance": 0.8, "name": "Important Task"})
        integration.event_bus.emit("learning_event", {"importance": 0.7, "topic": "New Concept"})
        
        # Wait for a bit to let the system process
        import time
        time.sleep(2)
        
        # Get hormone levels
        hormone_levels = integration.get_hormone_levels()
        print("Current hormone levels:")
        for hormone, level in hormone_levels.items():
            if level > 0.1:  # Only show significant levels
                print(f"  {hormone}: {level:.2f}")
                
        # Adapt receptor sensitivity based on performance
        integration.adapt_receptor_sensitivity("memory", "dopamine", 0.8)
        
        # Learn optimal hormone profile for a context
        optimal_profile = integration.learn_optimal_hormone_profiles({
            "task_type": "creative",
            "priority": "high"
        })
        print("\nOptimal hormone profile for creative high-priority task:")
        for hormone, level in optimal_profile.items():
            if level > 0.5:  # Only show significant levels
                print(f"  {hormone}: {level:.2f}")
        
        # Emit a custom event
        integration.emit_event(
            "custom_event", 
            {"data": "test data"}, 
            signal_type="excitatory", 
            context={"source": "example"}
        )
                
    finally:
        # Shut down
        integration.shutdown()