"""
Environmental Adaptation System for Genetic Triggers

This module implements environmental adaptation mechanisms for the genetic trigger system,
including environmental encoding, genetic memory storage, and natural selection processes.

Features:
- Environmental condition encoding and similarity matching
- Genetic memory storage with epigenetic markers
- Natural selection process for trigger optimization
- Environmental pressure adaptation
- Cross-generational learning and inheritance
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
from .epigenetic_memory import EpigeneticMemory
from .environmental_state import EnvironmentalState


class AdaptationPressure(Enum):
    """Types of environmental adaptation pressure"""
    LOW = "low"           # Minimal pressure, slow adaptation
    MEDIUM = "medium"     # Moderate pressure, balanced adaptation
    HIGH = "high"         # High pressure, rapid adaptation
    CRITICAL = "critical" # Critical pressure, emergency adaptation


@dataclass
class EnvironmentalMemory:
    """Stores environmental conditions and their genetic adaptations"""
    environment_hash: str
    environmental_conditions: Dict[str, Any]
    successful_triggers: List[str] = field(default_factory=list)
    failed_triggers: List[str] = field(default_factory=list)
    adaptation_pressure: AdaptationPressure = AdaptationPressure.MEDIUM
    last_encountered: datetime = field(default_factory=datetime.now)
    encounter_count: int = 0
    success_rate: float = 0.0
    epigenetic_markers: EpigeneticMemory = field(default_factory=EpigeneticMemory)
    
    def update_success_rate(self):
        """Update success rate based on trigger performance"""
        total_triggers = len(self.successful_triggers) + len(self.failed_triggers)
        if total_triggers > 0:
            self.success_rate = len(self.successful_triggers) / total_triggers


@dataclass
class GeneticMemory:
    """Stores genetic adaptations and their performance history"""
    trigger_id: str
    environmental_conditions: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    generation: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    epigenetic_markers: EpigeneticMemory = field(default_factory=EpigeneticMemory)
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def add_performance(self, performance: float):
        """Add performance measurement to history"""
        self.performance_history.append(performance)
        self.last_used = datetime.now()
        self.usage_count += 1
    
    def get_average_performance(self) -> float:
        """Get average performance over history"""
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)
    
    def get_recent_performance(self, window: int = 10) -> float:
        """Get average performance over recent window"""
        if not self.performance_history:
            return 0.0
        recent = self.performance_history[-window:]
        return sum(recent) / len(recent)


class EnvironmentalAdaptationSystem:
    """
    Manages environmental adaptation for genetic triggers.
    
    Features:
    - Environmental condition encoding and matching
    - Genetic memory storage and retrieval
    - Natural selection process
    - Cross-generational learning
    - Adaptation pressure management
    """
    
    def __init__(self, memory_capacity: int = 1000, similarity_threshold: float = 0.7):
        self.memory_capacity = memory_capacity
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("EnvironmentalAdaptation")
        
        # Environmental memory storage
        self.environmental_memories: Dict[str, EnvironmentalMemory] = {}
        self.genetic_memories: Dict[str, GeneticMemory] = {}
        
        # Adaptation tracking
        self.adaptation_pressure_history: List[Dict[str, Any]] = []
        self.selection_events: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_adaptations = 0
        self.successful_adaptations = 0
        self.failed_adaptations = 0
        
        self.logger.info("Environmental Adaptation System initialized")
    
    def encode_environment(self, environment: EnvironmentalState) -> str:
        """
        Encode environmental conditions into a hash for storage and comparison.
        
        Args:
            environment: Environmental state to encode
            
        Returns:
            Hash string representing the environmental conditions
        """
        # Convert environment to dict for encoding
        if hasattr(environment, '__dict__'):
            env_dict = environment.__dict__
        elif hasattr(environment, 'to_dict'):
            env_dict = environment.to_dict()
        else:
            env_dict = dict(environment)
        
        # Create a normalized representation
        normalized_env = {}
        for key, value in env_dict.items():
            if isinstance(value, (int, float)):
                # Normalize numeric values to 0-1 range
                normalized_env[key] = min(1.0, max(0.0, float(value)))
            elif isinstance(value, dict):
                # Recursively normalize nested dicts
                normalized_env[key] = {
                    k: min(1.0, max(0.0, float(v))) if isinstance(v, (int, float)) else v
                    for k, v in value.items()
                }
            else:
                normalized_env[key] = value
        
        # Create hash from normalized environment
        env_str = json.dumps(normalized_env, sort_keys=True)
        return str(hash(env_str))
    
    def calculate_environmental_similarity(self, env1: EnvironmentalState, 
                                        env2: EnvironmentalState) -> float:
        """
        Calculate similarity between two environmental states.
        
        Args:
            env1: First environmental state
            env2: Second environmental state
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Convert to dicts
        def to_dict(env):
            if hasattr(env, '__dict__'):
                return env.__dict__
            elif hasattr(env, 'to_dict'):
                return env.to_dict()
            else:
                return dict(env)
        
        dict1 = to_dict(env1)
        dict2 = to_dict(env2)
        
        # Find common keys
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    diff = abs(val1 - val2) / max_val
                    similarities.append(max(0.0, 1.0 - diff))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # Nested dict similarity
                nested_sim = self.calculate_environmental_similarity(val1, val2)
                similarities.append(nested_sim)
            elif val1 == val2:
                # Exact match
                similarities.append(1.0)
            else:
                # No match
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    async def find_similar_environments(self, environment: EnvironmentalState, 
                                      threshold: float = None) -> List[EnvironmentalMemory]:
        """
        Find previously encountered environments similar to the current one.
        
        Args:
            environment: Current environmental state
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List of similar environmental memories
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        similar_memories = []
        
        for memory in self.environmental_memories.values():
            # Create EnvironmentalState from memory conditions
            memory_env = EnvironmentalState(**memory.environmental_conditions)
            similarity = self.calculate_environmental_similarity(environment, memory_env)
            
            if similarity >= threshold:
                similar_memories.append(memory)
        
        # Sort by similarity (highest first)
        similar_memories.sort(key=lambda m: self.calculate_environmental_similarity(
            environment, EnvironmentalState(**m.environmental_conditions)
        ), reverse=True)
        
        return similar_memories
    
    async def store_environmental_memory(self, environment: EnvironmentalState, 
                                       trigger_id: str, success: bool):
        """
        Store environmental memory with trigger performance.
        
        Args:
            environment: Environmental state encountered
            trigger_id: ID of the trigger that was activated
            success: Whether the trigger was successful
        """
        env_hash = self.encode_environment(environment)
        
        if env_hash not in self.environmental_memories:
            # Create new environmental memory
            memory = EnvironmentalMemory(
                environment_hash=env_hash,
                environmental_conditions=environment.__dict__ if hasattr(environment, '__dict__') else dict(environment)
            )
            self.environmental_memories[env_hash] = memory
        else:
            memory = self.environmental_memories[env_hash]
        
        # Update memory
        memory.encounter_count += 1
        memory.last_encountered = datetime.now()
        
        if success:
            memory.successful_triggers.append(trigger_id)
        else:
            memory.failed_triggers.append(trigger_id)
        
        memory.update_success_rate()
        
        # Enforce memory capacity
        if len(self.environmental_memories) > self.memory_capacity:
            await self._prune_oldest_memories()
        
        self.logger.debug(f"Stored environmental memory: {env_hash}, success: {success}")
    
    async def store_genetic_memory(self, trigger: GeneticTrigger, 
                                 environment: EnvironmentalState, 
                                 performance: float):
        """
        Store genetic memory for a trigger's performance in an environment.
        
        Args:
            trigger: Genetic trigger that was used
            environment: Environmental state where trigger was used
            performance: Performance score (0.0-1.0)
        """
        memory_id = f"{trigger.id}_{self.encode_environment(environment)}"
        
        if memory_id not in self.genetic_memories:
            # Create new genetic memory
            memory = GeneticMemory(
                trigger_id=trigger.id,
                environmental_conditions=environment.__dict__ if hasattr(environment, '__dict__') else dict(environment),
                generation=0,
                parent_id=None
            )
            self.genetic_memories[memory_id] = memory
        else:
            memory = self.genetic_memories[memory_id]
        
        # Update memory
        memory.add_performance(performance)
        memory.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance': performance,
            'environment_hash': self.encode_environment(environment)
        })
        
        # Inherit epigenetic markers from trigger
        memory.epigenetic_markers = trigger.epigenetic_markers
        
        self.logger.debug(f"Stored genetic memory: {memory_id}, performance: {performance}")
    
    async def natural_selection(self, environment: EnvironmentalState, 
                              available_triggers: List[GeneticTrigger],
                              pressure: AdaptationPressure = AdaptationPressure.MEDIUM) -> GeneticTrigger:
        """
        Perform natural selection to choose the best trigger for the environment.
        
        Args:
            environment: Current environmental state
            available_triggers: List of available genetic triggers
            pressure: Adaptation pressure level
            
        Returns:
            Selected genetic trigger
        """
        if not available_triggers:
            raise ValueError("No triggers available for selection")
        
        # Find similar environments
        similar_memories = await self.find_similar_environments(environment)
        
        # Score triggers based on historical performance
        trigger_scores = []
        for trigger in available_triggers:
            score = await self._calculate_trigger_fitness(trigger, environment, similar_memories, pressure)
            trigger_scores.append((trigger, score))
        
        # Sort by score (highest first)
        trigger_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selection based on pressure
        if pressure == AdaptationPressure.LOW:
            # Conservative selection - prefer proven triggers
            selected_trigger = trigger_scores[0][0]
        elif pressure == AdaptationPressure.MEDIUM:
            # Balanced selection - mix of proven and experimental
            if random.random() < 0.8:
                selected_trigger = trigger_scores[0][0]  # Best trigger
            else:
                # Random selection from top 50%
                top_half = trigger_scores[:max(1, len(trigger_scores) // 2)]
                selected_trigger = random.choice(top_half)[0]
        elif pressure == AdaptationPressure.HIGH:
            # Aggressive selection - prefer high-risk, high-reward
            if random.random() < 0.6:
                selected_trigger = trigger_scores[0][0]  # Best trigger
            else:
                # Random selection from all triggers
                selected_trigger = random.choice(trigger_scores)[0]
        else:  # CRITICAL
            # Emergency selection - use best available
            selected_trigger = trigger_scores[0][0]
        
        # Record selection event
        selection_event = {
            'timestamp': datetime.now().isoformat(),
            'environment_hash': self.encode_environment(environment),
            'selected_trigger_id': selected_trigger.id,
            'pressure': pressure.value,
            'available_triggers': len(available_triggers),
            'similar_environments': len(similar_memories)
        }
        self.selection_events.append(selection_event)
        
        self.logger.info(f"Natural selection: {selected_trigger.id} selected under {pressure.value} pressure")
        return selected_trigger
    
    async def _calculate_trigger_fitness(self, trigger: GeneticTrigger, 
                                       environment: EnvironmentalState,
                                       similar_memories: List[EnvironmentalMemory],
                                       pressure: AdaptationPressure) -> float:
        """
        Calculate fitness score for a trigger in the given environment.
        
        Args:
            trigger: Genetic trigger to evaluate
            environment: Current environmental state
            similar_memories: Similar environmental memories
            pressure: Adaptation pressure
            
        Returns:
            Fitness score (0.0-1.0)
        """
        base_fitness = trigger.fitness_score
        
        # Adjust based on similar environments
        if similar_memories:
            # Calculate average success rate in similar environments
            success_rates = [mem.success_rate for mem in similar_memories]
            avg_success_rate = sum(success_rates) / len(success_rates)
            
            # Boost fitness if trigger was successful in similar environments
            if trigger.id in [mem.successful_triggers for mem in similar_memories]:
                base_fitness *= 1.2
            elif trigger.id in [mem.failed_triggers for mem in similar_memories]:
                base_fitness *= 0.8
        
        # Adjust based on pressure
        pressure_multipliers = {
            AdaptationPressure.LOW: 1.0,
            AdaptationPressure.MEDIUM: 1.1,
            AdaptationPressure.HIGH: 1.3,
            AdaptationPressure.CRITICAL: 1.5
        }
        
        base_fitness *= pressure_multipliers[pressure]
        
        # Add epigenetic influence
        epigenetic_modifier = trigger.epigenetic_markers.get_expression_modifier()
        base_fitness *= (1.0 + epigenetic_modifier)
        
        return min(1.0, max(0.0, base_fitness))
    
    async def _prune_oldest_memories(self):
        """Remove oldest environmental memories to maintain capacity"""
        # Sort by last encountered time
        sorted_memories = sorted(
            self.environmental_memories.items(),
            key=lambda x: x[1].last_encountered
        )
        
        # Remove oldest 10% of memories
        remove_count = max(1, len(sorted_memories) // 10)
        for i in range(remove_count):
            env_hash, _ = sorted_memories[i]
            del self.environmental_memories[env_hash]
        
        self.logger.info(f"Pruned {remove_count} oldest environmental memories")
    
    async def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptation system"""
        return {
            'environmental_memories': len(self.environmental_memories),
            'genetic_memories': len(self.genetic_memories),
            'total_adaptations': self.total_adaptations,
            'successful_adaptations': self.successful_adaptations,
            'failed_adaptations': self.failed_adaptations,
            'success_rate': (self.successful_adaptations / self.total_adaptations 
                           if self.total_adaptations > 0 else 0.0),
            'selection_events': len(self.selection_events),
            'memory_capacity': self.memory_capacity,
            'similarity_threshold': self.similarity_threshold
        }
    
    async def optimize_triggers_for_environment(self, environment: EnvironmentalState,
                                              triggers: List[GeneticTrigger]) -> List[GeneticTrigger]:
        """
        Optimize triggers for a specific environment using genetic memory.
        
        Args:
            environment: Target environment
            triggers: List of triggers to optimize
            
        Returns:
            Optimized list of triggers
        """
        similar_memories = await self.find_similar_environments(environment)
        
        # Score and sort triggers
        trigger_scores = []
        for trigger in triggers:
            score = await self._calculate_trigger_fitness(
                trigger, environment, similar_memories, AdaptationPressure.MEDIUM
            )
            trigger_scores.append((trigger, score))
        
        # Sort by fitness score
        trigger_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return optimized list
        return [trigger for trigger, _ in trigger_scores]
    
    async def cross_generational_learning(self, parent_trigger: GeneticTrigger,
                                        child_trigger: GeneticTrigger):
        """
        Transfer learning from parent to child trigger.
        
        Args:
            parent_trigger: Parent genetic trigger
            child_trigger: Child genetic trigger
        """
        # Inherit epigenetic markers
        child_trigger.epigenetic_markers.inherit_from_parent(parent_trigger.epigenetic_markers)
        
        # Inherit genetic memory
        parent_memories = [
            memory for memory in self.genetic_memories.values()
            if memory.trigger_id == parent_trigger.id
        ]
        
        for memory in parent_memories:
            # Create child memory with inherited performance
            child_memory = GeneticMemory(
                trigger_id=child_trigger.id,
                environmental_conditions=memory.environmental_conditions,
                generation=memory.generation + 1,
                parent_id=parent_trigger.id,
                epigenetic_markers=memory.epigenetic_markers
            )
            
            # Inherit performance with some degradation
            inherited_performance = memory.get_average_performance() * 0.9
            child_memory.add_performance(inherited_performance)
            
            memory_id = f"{child_trigger.id}_{self.encode_environment(EnvironmentalState(**memory.environmental_conditions))}"
            self.genetic_memories[memory_id] = child_memory
        
        self.logger.info(f"Cross-generational learning: {child_trigger.id} inherited from {parent_trigger.id}")


# Integration with existing GeneticTriggerSystem
class EnhancedGeneticTriggerSystem:
    """
    Enhanced genetic trigger system with environmental adaptation.
    
    Integrates the environmental adaptation system with the existing genetic trigger system.
    """
    
    def __init__(self):
        self.adaptation_system = EnvironmentalAdaptationSystem()
        self.triggers: Dict[str, GeneticTrigger] = {}
        self.logger = logging.getLogger("EnhancedGeneticTriggerSystem")
    
    async def create_trigger_from_environment(self, environment: EnvironmentalState,
                                            genetic_sequence: str = None) -> GeneticTrigger:
        """Create a new genetic trigger optimized for the environment"""
        # Find similar environments to learn from
        similar_memories = await self.adaptation_system.find_similar_environments(environment)
        
        # Generate genetic sequence if not provided
        if genetic_sequence is None:
            genetic_sequence = self._generate_sequence_from_memories(similar_memories)
        
        # Create trigger
        trigger_id = f"trigger_{uuid.uuid4().hex[:8]}"
        trigger = GeneticTrigger(
            trigger_id=trigger_id,
            formation_environment=environment.__dict__ if hasattr(environment, '__dict__') else dict(environment),
            genetic_sequence=genetic_sequence
        )
        
        self.triggers[trigger_id] = trigger
        return trigger
    
    def _generate_sequence_from_memories(self, memories: List[EnvironmentalMemory]) -> str:
        """Generate genetic sequence based on environmental memories"""
        if not memories:
            return "ATGCGATCGTAGC"  # Default sequence
        
        # Analyze successful triggers from memories
        successful_triggers = []
        for memory in memories:
            successful_triggers.extend(memory.successful_triggers)
        
        # Generate sequence based on successful patterns
        # This is a simplified implementation
        return "ATGCGATCGTAGC"  # Placeholder
    
    async def select_trigger_for_environment(self, environment: EnvironmentalState,
                                           pressure: AdaptationPressure = AdaptationPressure.MEDIUM) -> GeneticTrigger:
        """Select the best trigger for the current environment"""
        available_triggers = list(self.triggers.values())
        
        if not available_triggers:
            # Create new trigger if none available
            return await self.create_trigger_from_environment(environment)
        
        # Use natural selection
        selected_trigger = await self.adaptation_system.natural_selection(
            environment, available_triggers, pressure
        )
        
        return selected_trigger
    
    async def record_trigger_performance(self, trigger: GeneticTrigger,
                                       environment: EnvironmentalState,
                                       success: bool, performance: float = None):
        """Record trigger performance for learning"""
        # Store environmental memory
        await self.adaptation_system.store_environmental_memory(
            environment, trigger.id, success
        )
        
        # Store genetic memory
        if performance is None:
            performance = 1.0 if success else 0.0
        
        await self.adaptation_system.store_genetic_memory(
            trigger, environment, performance
        )
        
        # Update trigger fitness
        trigger.update_fitness(performance)
        
        # Update statistics
        self.adaptation_system.total_adaptations += 1
        if success:
            self.adaptation_system.successful_adaptations += 1
        else:
            self.adaptation_system.failed_adaptations += 1 