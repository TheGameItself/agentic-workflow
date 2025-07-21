"""
Example script demonstrating the HormoneSystemController in action.

This script shows how to use the HormoneSystemController to create a
brain-inspired hormone communication system between different lobes.
"""

import sys
import os
import time
import logging
from typing import Dict, Any

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp.hormone_system_controller import HormoneSystemController
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HormoneSystemExample")

class MockLobe:
    """Mock lobe for demonstration purposes"""
    
    def __init__(self, name: str, hormone_controller: HormoneSystemController, event_bus: LobeEventBus):
        self.name = name
        self.hormone_controller = hormone_controller
        self.event_bus = event_bus
        self.state = {"activity": 0.5, "stress": 0.1, "focus": 0.5}
        
        # Subscribe to hormone updates
        self.event_bus.subscribe_to_hormones(self.on_hormone_update)
        
        logger.info(f"Lobe '{name}' initialized")
        
    def on_hormone_update(self, hormone_levels: Dict[str, float]):
        """Handle hormone updates"""
        # Update lobe state based on hormones
        if "dopamine" in hormone_levels and hormone_levels["dopamine"] > 0.6:
            self.state["activity"] = min(1.0, self.state["activity"] + 0.1)
            logger.info(f"[{self.name}] Activity increased due to dopamine: {self.state['activity']:.2f}")
            
        if "cortisol" in hormone_levels and hormone_levels["cortisol"] > 0.6:
            self.state["stress"] = min(1.0, self.state["stress"] + 0.2)
            logger.info(f"[{self.name}] Stress increased due to cortisol: {self.state['stress']:.2f}")
            
        if "acetylcholine" in hormone_levels and hormone_levels["acetylcholine"] > 0.6:
            self.state["focus"] = min(1.0, self.state["focus"] + 0.15)
            logger.info(f"[{self.name}] Focus increased due to acetylcholine: {self.state['focus']:.2f}")
            
        if "gaba" in hormone_levels and hormone_levels["gaba"] > 0.5:
            self.state["stress"] = max(0.0, self.state["stress"] - 0.1)
            logger.info(f"[{self.name}] Stress decreased due to GABA: {self.state['stress']:.2f}")
            
    def perform_task(self, task_name: str, difficulty: float):
        """Perform a task and release hormones based on outcome"""
        logger.info(f"[{self.name}] Performing task: {task_name} (difficulty: {difficulty:.2f})")
        
        # Calculate success probability based on state and difficulty
        focus_factor = self.state["focus"] * 0.6
        stress_penalty = self.state["stress"] * 0.4
        activity_bonus = self.state["activity"] * 0.3
        
        success_prob = 0.7 + focus_factor + activity_bonus - stress_penalty - difficulty
        success = success_prob > 0.5
        
        if success:
            logger.info(f"[{self.name}] Successfully completed task: {task_name}")
            # Release dopamine on success
            self.hormone_controller.release_hormone(
                self.name, "dopamine", 0.8, context={"task": task_name, "outcome": "success"}
            )
            # Provide positive feedback for hormone adaptation
            self.hormone_controller.adapt_receptor_sensitivity(self.name, "acetylcholine", 0.7)
        else:
            logger.info(f"[{self.name}] Failed task: {task_name}")
            # Release cortisol on failure
            self.hormone_controller.release_hormone(
                self.name, "cortisol", 0.7, context={"task": task_name, "outcome": "failure"}
            )
            # Provide negative feedback for hormone adaptation
            self.hormone_controller.adapt_receptor_sensitivity(self.name, "dopamine", -0.5)
            
        return success
        
    def handle_stress(self):
        """Handle high stress by releasing GABA"""
        if self.state["stress"] > 0.7:
            logger.info(f"[{self.name}] Stress level critical, releasing GABA")
            self.hormone_controller.release_hormone(
                self.name, "gaba", 0.8, context={"action": "stress_reduction"}
            )
            
    def learn_new_concept(self, concept_name: str, complexity: float):
        """Learn a new concept and release learning-related hormones"""
        logger.info(f"[{self.name}] Learning new concept: {concept_name} (complexity: {complexity:.2f})")
        
        # Release acetylcholine for attention and learning
        self.hormone_controller.release_hormone(
            self.name, "acetylcholine", 0.7, context={"concept": concept_name}
        )
        
        # Release growth hormone for long-term learning
        self.hormone_controller.release_hormone(
            self.name, "growth_hormone", 0.6, context={"concept": concept_name}
        )
        
        # Calculate learning effectiveness
        focus_factor = self.state["focus"] * 0.7
        stress_penalty = self.state["stress"] * 0.3
        
        effectiveness = 0.6 + focus_factor - stress_penalty - complexity * 0.5
        effectiveness = max(0.1, min(1.0, effectiveness))
        
        logger.info(f"[{self.name}] Learning effectiveness: {effectiveness:.2f}")
        return effectiveness

def main():
    """Main function demonstrating the hormone system"""
    # Create event bus
    event_bus = LobeEventBus()
    
    # Create hormone controller
    hormone_controller = HormoneSystemController(event_bus=event_bus)
    
    try:
        # Register lobes
        hormone_controller.register_lobe("task_management", position=(0, 0, 0))
        hormone_controller.register_lobe("memory", position=(1, 0, 0))
        hormone_controller.register_lobe("decision_making", position=(0, 1, 0))
        
        # Connect lobes
        hormone_controller.lobes["task_management"].connected_lobes = ["memory", "decision_making"]
        hormone_controller.lobes["memory"].connected_lobes = ["task_management"]
        hormone_controller.lobes["decision_making"].connected_lobes = ["task_management"]
        
        # Create mock lobes
        task_lobe = MockLobe("task_management", hormone_controller, event_bus)
        memory_lobe = MockLobe("memory", hormone_controller, event_bus)
        decision_lobe = MockLobe("decision_making", hormone_controller, event_bus)
        
        # Simulation loop
        logger.info("Starting hormone system simulation")
        
        # Scenario 1: Task execution with increasing difficulty
        logger.info("\n=== Scenario 1: Task Execution ===")
        for i in range(5):
            difficulty = 0.2 + i * 0.15
            task_lobe.perform_task(f"Task {i+1}", difficulty)
            
            # Process hormone cascades
            hormone_controller.process_hormone_cascades()
            
            # Handle stress if needed
            task_lobe.handle_stress()
            
            # Print current hormone levels
            levels = hormone_controller.get_levels()
            logger.info(f"Global hormone levels: dopamine={levels['dopamine']:.2f}, " +
                       f"cortisol={levels['cortisol']:.2f}, acetylcholine={levels['acetylcholine']:.2f}")
            
            time.sleep(1)
            
        # Scenario 2: Learning new concepts
        logger.info("\n=== Scenario 2: Learning ===")
        for i in range(3):
            complexity = 0.3 + i * 0.2
            memory_lobe.learn_new_concept(f"Concept {i+1}", complexity)
            
            # Process hormone cascades
            hormone_controller.process_hormone_cascades()
            
            # Print current hormone levels
            levels = hormone_controller.get_levels()
            logger.info(f"Global hormone levels: acetylcholine={levels['acetylcholine']:.2f}, " +
                       f"growth_hormone={levels['growth_hormone']:.2f}")
            
            time.sleep(1)
            
        # Scenario 3: Decision making under stress
        logger.info("\n=== Scenario 3: Decision Making Under Stress ===")
        
        # Induce stress
        hormone_controller.release_hormone("task_management", "cortisol", 0.9)
        hormone_controller.process_hormone_cascades()
        
        # Make decisions
        for i in range(3):
            difficulty = 0.4 + i * 0.1
            decision_lobe.perform_task(f"Decision {i+1}", difficulty)
            
            # Process hormone cascades
            hormone_controller.process_hormone_cascades()
            
            # Handle stress
            decision_lobe.handle_stress()
            
            # Print current hormone levels
            levels = hormone_controller.get_levels()
            logger.info(f"Global hormone levels: cortisol={levels['cortisol']:.2f}, " +
                       f"gaba={levels['gaba']:.2f}, dopamine={levels['dopamine']:.2f}")
            
            time.sleep(1)
            
        # Scenario 4: Recovery and adaptation
        logger.info("\n=== Scenario 4: Recovery and Adaptation ===")
        
        # Release calming hormones
        hormone_controller.release_hormone("memory", "gaba", 0.8)
        hormone_controller.release_hormone("decision_making", "serotonin", 0.7)
        
        # Process hormone cascades
        hormone_controller.process_hormone_cascades()
        
        # Check receptor adaptation
        for lobe_name in ["task_management", "memory", "decision_making"]:
            lobe = hormone_controller.lobes[lobe_name]
            logger.info(f"[{lobe_name}] Receptor sensitivities: " +
                       f"dopamine={lobe.receptor_sensitivity['dopamine']:.2f}, " +
                       f"cortisol={lobe.receptor_sensitivity['cortisol']:.2f}, " +
                       f"acetylcholine={lobe.receptor_sensitivity['acetylcholine']:.2f}")
            
        # Final hormone levels
        levels = hormone_controller.get_levels()
        logger.info("\nFinal hormone levels:")
        for hormone, level in levels.items():
            if level > 0.1:
                logger.info(f"  {hormone}: {level:.2f}")
                
    finally:
        # Stop hormone controller
        hormone_controller.stop()
        logger.info("Hormone system simulation completed")

if __name__ == "__main__":
    main()