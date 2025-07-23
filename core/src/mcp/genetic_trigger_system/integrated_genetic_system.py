"""
Integrated Genetic Trigger System

This module implements the complete integration of genetic trigger system components
with environmental adaptation, optimization, and natural selection processes.

Features:
- Genetic trigger activation based on environmental conditions
- Environmental encoding and genetic memory storage
- Natural selection process for trigger optimization
- Cross-generational learning and inheritance
- Performance-based trigger selection and evolution
- Integration with hormone system and memory systems
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict

from .genetic_trigger import GeneticTrigger
from .environmental_adaptation import EnvironmentalAdaptationSystem, AdaptationPressure
from .epigenetic_memory import EpigeneticMemory
from .environmental_state import EnvironmentalState
from .hormone_system_interface import HormoneSystemInterface
from .neural_genetic_processor import NeuralGeneticProcessor
from .code_genetic_processor import CodeGeneticProcessor
from .adaptive_mutation_controller import AdaptiveMutationController


class TriggerActivationMode(Enum):
    """Modes for trigger activation"""
    ENVIRONMENTAL = "environmental"      # Based on environmental similarity
    PERFORMANCE = "performance"          # Based on performance metrics
    HORMONE = "hormone"                  # Based on hormone levels
    HYBRID = "hybrid"                    # Combined multiple factors


@dataclass
class TriggerPerformance:
    """Tracks performance metrics for genetic triggers"""
    trigger_id: str
    activation_count: int = 0
    success_count: int = 0
    average_response_time: float = 0.0
    resource_efficiency: float = 0.0
    environmental_adaptation_score: float = 0.0
    last_activation: Optional[datetime] = None
    performance_history: List[float] = field(default_factory=list)
    
    def update_performance(self, success: bool, response_time: float, 
                          resource_usage: float, adaptation_score: float):
        """Update performance metrics"""
        self.activation_count += 1
        if success:
            self.success_count += 1
        
        self.last_activation = datetime.now()
        
        # Update average response time
        if self.average_response_time == 0.0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (self.average_response_time + response_time) / 2
        
        # Update resource efficiency
        self.resource_efficiency = resource_usage
        self.environmental_adaptation_score = adaptation_score
        
        # Calculate overall performance score
        success_rate = self.success_count / self.activation_count if self.activation_count > 0 else 0.0
        performance_score = (success_rate * 0.4 + 
                           (1.0 - self.average_response_time) * 0.3 + 
                           (1.0 - self.resource_efficiency) * 0.2 + 
                           self.environmental_adaptation_score * 0.1)
        
        self.performance_history.append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_fitness_score(self) -> float:
        """Calculate fitness score based on performance"""
        if not self.performance_history:
            return 0.5
        
        # Weight recent performance more heavily
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        return sum(recent_performance) / len(recent_performance)


class IntegratedGeneticTriggerSystem:
    """
    Integrated genetic trigger system with environmental adaptation and optimization.
    
    Features:
    - Environmental condition-based trigger activation
    - Natural selection and evolution of triggers
    - Performance-based trigger selection
    - Cross-generational learning
    - Integration with hormone and memory systems
    """
    
    def __init__(self, 
                 max_triggers: int = 1000,
                 selection_pressure: float = 0.8,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.3):
        self.max_triggers = max_triggers
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.logger = logging.getLogger("IntegratedGeneticSystem")
        
        # Core components
        self.triggers: Dict[str, GeneticTrigger] = {}
        self.performance_tracker: Dict[str, TriggerPerformance] = {}
        self.environmental_adaptation = EnvironmentalAdaptationSystem()
        self.hormone_interface = HormoneSystemInterface()
        
        # System state
        self.current_environment: Optional[EnvironmentalState] = None
        self.active_triggers: Set[str] = set()
        self.trigger_population: List[str] = []
        
        # Evolution tracking
        self.generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.selection_events: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_activations = 0
        self.successful_activations = 0
        self.failed_activations = 0
        self.average_response_time = 0.0
        
        self.logger.info("Integrated Genetic Trigger System initialized")
    
    async def update_environment(self, environment: EnvironmentalState):
        """
        Update the current environmental state and evaluate triggers.
        
        Args:
            environment: New environmental state
        """
        self.current_environment = environment
        
        # Find similar environments for adaptation
        similar_memories = await self.environmental_adaptation.find_similar_environments(environment)
        
        # Evaluate all triggers for activation
        await self._evaluate_triggers_for_activation(environment, similar_memories)
        
        # Store environmental memory
        await self.environmental_adaptation.store_environmental_memory(environment, {})
        
        self.logger.debug(f"Environment updated, {len(self.active_triggers)} triggers active")
    
    async def _evaluate_triggers_for_activation(self, environment: EnvironmentalState, 
                                              similar_memories: List):
        """Evaluate all triggers for activation based on current environment"""
        self.active_triggers.clear()
        
        for trigger_id, trigger in self.triggers.items():
            # Convert environment to dict for trigger evaluation
            env_dict = self._environment_to_dict(environment)
            
            # Check if trigger should activate
            should_activate = await trigger.should_activate(env_dict)
            
            if should_activate:
                self.active_triggers.add(trigger_id)
                self.logger.debug(f"Trigger {trigger_id} activated")
        
        self.logger.info(f"Trigger evaluation complete: {len(self.active_triggers)} triggers active")
    
    def _environment_to_dict(self, environment: EnvironmentalState) -> Dict[str, Any]:
        """Convert EnvironmentalState to dictionary"""
        if hasattr(environment, '__dict__'):
            return environment.__dict__
        elif hasattr(environment, 'to_dict'):
            return environment.to_dict()
        else:
            return dict(environment)
    
    async def create_trigger_from_environment(self, environment: EnvironmentalState,
                                            genetic_sequence: str = None,
                                            activation_threshold: float = 0.7) -> GeneticTrigger:
        """
        Create a new genetic trigger based on current environment.
        
        Args:
            environment: Environmental conditions for trigger formation
            genetic_sequence: Optional genetic sequence (generated if None)
            activation_threshold: Threshold for trigger activation
            
        Returns:
            Newly created GeneticTrigger
        """
        trigger_id = f"trigger_{uuid.uuid4().hex[:8]}"
        
        # Generate genetic sequence if not provided
        if genetic_sequence is None:
            genetic_sequence = self._generate_sequence_from_environment(environment)
        
        # Convert environment to dict
        env_dict = self._environment_to_dict(environment)
        
        # Create trigger
        trigger = GeneticTrigger(
            trigger_id=trigger_id,
            formation_environment=env_dict,
            genetic_sequence=genetic_sequence,
            activation_threshold=activation_threshold
        )
        
        # Add to system
        self.triggers[trigger_id] = trigger
        self.performance_tracker[trigger_id] = TriggerPerformance(trigger_id=trigger_id)
        self.trigger_population.append(trigger_id)
        
        # Store genetic memory
        await self.environmental_adaptation.store_genetic_memory(
            trigger, environment, 0.5  # Initial neutral performance
        )
        
        self.logger.info(f"Created trigger {trigger_id} with sequence: {genetic_sequence[:20]}...")
        
        return trigger
    
    def _generate_sequence_from_environment(self, environment: EnvironmentalState) -> str:
        """Generate genetic sequence from environmental conditions"""
        # Simple sequence generation based on environment characteristics
        env_dict = self._environment_to_dict(environment)
        
        # Create a sequence based on environment features
        sequence_parts = []
        
        # Add system load encoding
        if 'system_load' in env_dict:
            load = env_dict['system_load']
            if isinstance(load, dict):
                cpu_load = load.get('cpu', 0.5)
                mem_load = load.get('memory', 0.5)
                sequence_parts.append(f"CPU{int(cpu_load*100):03d}")
                sequence_parts.append(f"MEM{int(mem_load*100):03d}")
        
        # Add performance metrics encoding
        if 'performance_metrics' in env_dict:
            perf = env_dict['performance_metrics']
            if isinstance(perf, dict):
                accuracy = perf.get('accuracy', 0.5)
                speed = perf.get('speed', 0.5)
                sequence_parts.append(f"ACC{int(accuracy*100):03d}")
                sequence_parts.append(f"SPD{int(speed*100):03d}")
        
        # Add task complexity encoding
        if 'task_complexity' in env_dict:
            complexity = env_dict['task_complexity']
            sequence_parts.append(f"CMP{int(complexity*100):03d}")
        
        # Combine into genetic sequence
        sequence = "".join(sequence_parts)
        
        # Ensure minimum length
        if len(sequence) < 20:
            sequence += "X" * (20 - len(sequence))
        
        return sequence
    
    async def select_trigger_for_environment(self, environment: EnvironmentalState,
                                           pressure: AdaptationPressure = AdaptationPressure.MEDIUM) -> Optional[GeneticTrigger]:
        """
        Select the best trigger for the current environment using natural selection.
        
        Args:
            environment: Current environmental conditions
            pressure: Adaptation pressure level
            
        Returns:
            Selected trigger or None if no suitable trigger found
        """
        if not self.triggers:
            return None
        
        # Get available triggers
        available_triggers = list(self.triggers.values())
        
        # Use natural selection to select trigger
        selected_trigger = await self.environmental_adaptation.natural_selection(
            environment, available_triggers, pressure
        )
        
        if selected_trigger:
            self.logger.info(f"Selected trigger {selected_trigger.id} for environment")
        
        return selected_trigger
    
    async def record_trigger_performance(self, trigger_id: str, success: bool,
                                       response_time: float, resource_usage: float):
        """
        Record performance metrics for a trigger.
        
        Args:
            trigger_id: ID of the trigger
            success: Whether the trigger activation was successful
            response_time: Response time in seconds
            resource_usage: Resource usage (0.0 to 1.0)
        """
        if trigger_id not in self.performance_tracker:
            self.logger.warning(f"Performance tracking not found for trigger {trigger_id}")
            return
        
        # Calculate adaptation score
        adaptation_score = 0.5  # Default neutral score
        if self.current_environment:
            trigger = self.triggers.get(trigger_id)
            if trigger:
                env_dict = self._environment_to_dict(self.current_environment)
                adaptation_score = trigger.calculate_similarity(env_dict)
        
        # Update performance tracker
        performance = self.performance_tracker[trigger_id]
        performance.update_performance(success, response_time, resource_usage, adaptation_score)
        
        # Update trigger fitness
        trigger = self.triggers.get(trigger_id)
        if trigger:
            trigger.update_fitness(performance.get_fitness_score())
        
        # Update system metrics
        self.total_activations += 1
        if success:
            self.successful_activations += 1
        else:
            self.failed_activations += 1
        
        # Update average response time
        if self.average_response_time == 0.0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (self.average_response_time + response_time) / 2
        
        self.logger.debug(f"Recorded performance for trigger {trigger_id}: success={success}, "
                         f"response_time={response_time:.3f}, resource_usage={resource_usage:.3f}")
    
    async def evolve_triggers(self, target_population_size: int = None):
        """
        Evolve the trigger population using natural selection and genetic operations.
        
        Args:
            target_population_size: Target size for population (uses max_triggers if None)
        """
        if target_population_size is None:
            target_population_size = self.max_triggers
        
        self.logger.info(f"Starting trigger evolution, population: {len(self.trigger_population)}")
        
        # Calculate fitness scores
        fitness_scores = {}
        for trigger_id in self.trigger_population:
            performance = self.performance_tracker.get(trigger_id)
            if performance:
                fitness_scores[trigger_id] = performance.get_fitness_score()
            else:
                fitness_scores[trigger_id] = 0.5  # Default neutral fitness
        
        # Sort by fitness
        sorted_triggers = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top performers for reproduction
        elite_size = max(1, int(len(sorted_triggers) * 0.2))  # Keep top 20%
        elite_triggers = [trigger_id for trigger_id, _ in sorted_triggers[:elite_size]]
        
        # Create new population
        new_population = elite_triggers.copy()
        
        # Generate offspring through crossover and mutation
        while len(new_population) < target_population_size:
            # Select parents using tournament selection
            parent1_id = self._tournament_selection(fitness_scores, tournament_size=3)
            parent2_id = self._tournament_selection(fitness_scores, tournament_size=3)
            
            if parent1_id and parent2_id:
                parent1 = self.triggers[parent1_id]
                parent2 = self.triggers[parent2_id]
                
                # Perform crossover
                if random.random() < self.crossover_rate:
                    child = GeneticTrigger.crossover(parent1, parent2)
                else:
                    # Clone parent with mutation
                    child = parent1.mutate()
                
                # Add child to population
                child_id = f"trigger_{uuid.uuid4().hex[:8]}"
                child.id = child_id
                self.triggers[child_id] = child
                self.performance_tracker[child_id] = TriggerPerformance(trigger_id=child_id)
                new_population.append(child_id)
        
        # Update population
        old_population = self.trigger_population.copy()
        self.trigger_population = new_population[:target_population_size]
        
        # Remove old triggers not in new population
        removed_triggers = set(old_population) - set(self.trigger_population)
        for trigger_id in removed_triggers:
            if trigger_id in self.triggers:
                del self.triggers[trigger_id]
            if trigger_id in self.performance_tracker:
                del self.performance_tracker[trigger_id]
        
        # Update generation
        self.generation += 1
        
        # Record evolution event
        evolution_event = {
            'generation': self.generation,
            'old_population_size': len(old_population),
            'new_population_size': len(self.trigger_population),
            'elite_size': elite_size,
            'removed_triggers': len(removed_triggers),
            'average_fitness': sum(fitness_scores.values()) / len(fitness_scores) if fitness_scores else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        self.evolution_history.append(evolution_event)
        
        self.logger.info(f"Evolution complete: generation {self.generation}, "
                        f"population {len(self.trigger_population)}, "
                        f"removed {len(removed_triggers)} triggers")
    
    def _tournament_selection(self, fitness_scores: Dict[str, float], tournament_size: int = 3) -> Optional[str]:
        """Select a trigger using tournament selection"""
        if not fitness_scores:
            return None
        
        # Randomly select tournament participants
        participants = random.sample(list(fitness_scores.keys()), 
                                   min(tournament_size, len(fitness_scores)))
        
        # Return the best participant
        return max(participants, key=lambda x: fitness_scores[x])
    
    async def optimize_triggers_for_environment(self, environment: EnvironmentalState) -> List[GeneticTrigger]:
        """
        Optimize triggers for a specific environment.
        
        Args:
            environment: Target environment for optimization
            
        Returns:
            List of optimized triggers
        """
        # Get all triggers
        triggers = list(self.triggers.values())
        
        # Use environmental adaptation system to optimize
        optimized_triggers = await self.environmental_adaptation.optimize_triggers_for_environment(
            environment, triggers
        )
        
        self.logger.info(f"Optimized {len(optimized_triggers)} triggers for environment")
        
        return optimized_triggers
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'total_triggers': len(self.triggers),
            'active_triggers': len(self.active_triggers),
            'generation': self.generation,
            'total_activations': self.total_activations,
            'successful_activations': self.successful_activations,
            'failed_activations': self.failed_activations,
            'success_rate': (self.successful_activations / self.total_activations 
                           if self.total_activations > 0 else 0.0),
            'average_response_time': self.average_response_time,
            'average_fitness': 0.0,
            'top_performers': [],
            'evolution_events': len(self.evolution_history)
        }
        
        # Calculate average fitness
        if self.performance_tracker:
            fitness_scores = [perf.get_fitness_score() for perf in self.performance_tracker.values()]
            stats['average_fitness'] = sum(fitness_scores) / len(fitness_scores)
        
        # Get top performers
        if self.performance_tracker:
            sorted_performance = sorted(
                self.performance_tracker.items(),
                key=lambda x: x[1].get_fitness_score(),
                reverse=True
            )[:5]
            stats['top_performers'] = [
                {'trigger_id': trigger_id, 'fitness': perf.get_fitness_score()}
                for trigger_id, perf in sorted_performance
            ]
        
        return stats
    
    async def cross_generational_learning(self, parent_trigger: GeneticTrigger,
                                        child_trigger: GeneticTrigger):
        """
        Transfer learning from parent to child trigger.
        
        Args:
            parent_trigger: Parent trigger with learned adaptations
            child_trigger: Child trigger to receive learning
        """
        # Use environmental adaptation system for cross-generational learning
        await self.environmental_adaptation.cross_generational_learning(
            parent_trigger, child_trigger
        )
        
        self.logger.debug(f"Cross-generational learning from {parent_trigger.id} to {child_trigger.id}")
    
    async def cleanup_old_triggers(self, max_age_days: int = 30):
        """
        Remove old triggers that haven't been used recently.
        
        Args:
            max_age_days: Maximum age in days before removal
        """
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        triggers_to_remove = []
        
        for trigger_id, performance in self.performance_tracker.items():
            if (performance.last_activation and 
                performance.last_activation < cutoff_time and
                performance.activation_count < 5):  # Only remove if rarely used
                triggers_to_remove.append(trigger_id)
        
        for trigger_id in triggers_to_remove:
            if trigger_id in self.triggers:
                del self.triggers[trigger_id]
            if trigger_id in self.performance_tracker:
                del self.performance_tracker[trigger_id]
            if trigger_id in self.trigger_population:
                self.trigger_population.remove(trigger_id)
        
        self.logger.info(f"Cleaned up {len(triggers_to_remove)} old triggers")
    
    async def save_system_state(self, filepath: str):
        """Save system state to file"""
        state = {
            'triggers': {trigger_id: trigger.to_dict() for trigger_id, trigger in self.triggers.items()},
            'performance_tracker': {
                trigger_id: {
                    'trigger_id': perf.trigger_id,
                    'activation_count': perf.activation_count,
                    'success_count': perf.success_count,
                    'average_response_time': perf.average_response_time,
                    'resource_efficiency': perf.resource_efficiency,
                    'environmental_adaptation_score': perf.environmental_adaptation_score,
                    'last_activation': perf.last_activation.isoformat() if perf.last_activation else None,
                    'performance_history': perf.performance_history
                }
                for trigger_id, perf in self.performance_tracker.items()
            },
            'trigger_population': self.trigger_population,
            'generation': self.generation,
            'evolution_history': self.evolution_history,
            'system_metrics': {
                'total_activations': self.total_activations,
                'successful_activations': self.successful_activations,
                'failed_activations': self.failed_activations,
                'average_response_time': self.average_response_time
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"System state saved to {filepath}")
    
    async def load_system_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load triggers
        self.triggers.clear()
        for trigger_id, trigger_data in state['triggers'].items():
            self.triggers[trigger_id] = GeneticTrigger.from_dict(trigger_data)
        
        # Load performance tracker
        self.performance_tracker.clear()
        for trigger_id, perf_data in state['performance_tracker'].items():
            performance = TriggerPerformance(trigger_id=perf_data['trigger_id'])
            performance.activation_count = perf_data['activation_count']
            performance.success_count = perf_data['success_count']
            performance.average_response_time = perf_data['average_response_time']
            performance.resource_efficiency = perf_data['resource_efficiency']
            performance.environmental_adaptation_score = perf_data['environmental_adaptation_score']
            performance.last_activation = (datetime.fromisoformat(perf_data['last_activation']) 
                                         if perf_data['last_activation'] else None)
            performance.performance_history = perf_data['performance_history']
            self.performance_tracker[trigger_id] = performance
        
        # Load other state
        self.trigger_population = state['trigger_population']
        self.generation = state['generation']
        self.evolution_history = state['evolution_history']
        
        # Load system metrics
        metrics = state['system_metrics']
        self.total_activations = metrics['total_activations']
        self.successful_activations = metrics['successful_activations']
        self.failed_activations = metrics['failed_activations']
        self.average_response_time = metrics['average_response_time']
        
        self.logger.info(f"System state loaded from {filepath}") 