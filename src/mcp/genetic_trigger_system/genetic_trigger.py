"""
Genetic Trigger System for MCP System Upgrade.

This module implements the Genetic Trigger System, which handles environmental
adaptation through genetic encoding and selection, activating optimized pathways
when similar conditions recur.

Enhanced with dual implementation strategy, hormone integration, and epigenetic memory.
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import enhanced genetic components (will be created)
try:
    from .neural_genetic_processor import NeuralGeneticProcessor
    from .code_genetic_processor import CodeGeneticProcessor
    from .hormone_system_interface import HormoneSystemInterface
    from .adaptive_mutation_controller import AdaptiveMutationController
    from .epigenetic_memory import EpigeneticMemory
    from ..neural_network_models.performance_tracker import PerformanceTracker
except ImportError:
    # Fallback for missing components
    NeuralGeneticProcessor = None
    CodeGeneticProcessor = None
    HormoneSystemInterface = None
    AdaptiveMutationController = None
    EpigeneticMemory = None
    PerformanceTracker = None

class GeneticTrigger:
    """
    Represents a genetic trigger that activates when specific environmental conditions are met.
    
    A genetic trigger encodes environmental conditions and associated neural pathway
    configurations, allowing the system to automatically optimize for different
    operating conditions through a simulated natural selection process.
    """
    
    def __init__(self, 
                 formation_environment: Dict[str, Any],
                 pathway_config: Dict[str, Any],
                 parent_ids: List[str] = None):
        """
        Initialize a genetic trigger.
        
        Args:
            formation_environment: Environmental conditions that led to the trigger's creation.
            pathway_config: Neural pathway configuration associated with the trigger.
            parent_ids: IDs of parent triggers if created through crossover.
        """
        self.id = str(uuid.uuid4())
        self.dna_signature = self._encode_environment(formation_environment)
        self.pathway_config = pathway_config
        self.creation_time = datetime.now()
        self.activation_count = 0
        self.success_rate = 0.0
        self.fitness_score = 0.5  # Initial neutral fitness
        self.mutation_rate = 0.05  # 5% mutation rate
        self.parent_ids = parent_ids or []
        self.formation_environment = formation_environment
        
        self.logger = logging.getLogger("GeneticTrigger")
        self.logger.info(f"Created genetic trigger {self.id} with DNA signature: {self.dna_signature[:20]}...")
    
    def _encode_environment(self, environment: Dict[str, Any]) -> str:
        """
        Convert environmental state to a DNA-like sequence.
        
        Args:
            environment: Environmental conditions to encode.
            
        Returns:
            DNA-like sequence representing the environmental conditions.
        """
        # Simple encoding: convert environment dict to a string representation
        # In a real implementation, this would use a more sophisticated encoding
        
        # Start with an empty DNA string
        dna = ""
        
        # Define our genetic alphabet (A, T, G, C)
        bases = ["A", "T", "G", "C"]
        
        # Convert each environment key-value pair to a DNA sequence
        for key, value in sorted(environment.items()):
            # Convert key to a sequence of bases
            key_hash = hash(key) % 1000
            key_dna = ""
            for i in range(10):  # Use 10 bases for each key
                key_dna += bases[(key_hash + i) % 4]
            
            # Convert value to a sequence of bases
            if isinstance(value, (int, float)):
                # Numeric values: encode magnitude in sequence length
                magnitude = min(int(abs(value) * 10), 100)
                value_dna = bases[0] * magnitude
            elif isinstance(value, str):
                # String values: hash to a sequence
                value_hash = hash(value) % 1000
                value_dna = ""
                for i in range(min(len(value) * 2, 50)):  # Use up to 50 bases
                    value_dna += bases[(value_hash + i) % 4]
            elif isinstance(value, bool):
                # Boolean values: simple encoding
                value_dna = "ATGC" if value else "CGTA"
            else:
                # Other types: fixed sequence
                value_dna = "NNNN"
            
            # Add separator between key and value
            separator = "TATA"  # Common promoter sequence in real DNA
            
            # Add terminator
            terminator = "AATAAA"  # Common terminator sequence in real DNA
            
            # Combine into a gene-like sequence
            dna += key_dna + separator + value_dna + terminator
        
        return dna
    
    def calculate_similarity(self, environment: Dict[str, Any]) -> float:
        """
        Calculate similarity between current environment and formation environment.
        
        Args:
            environment: Current environmental conditions.
            
        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not environment or not self.formation_environment:
            return 0.0
        
        # Count matching keys
        matching_keys = set(environment.keys()) & set(self.formation_environment.keys())
        if not matching_keys:
            return 0.0
        
        # Calculate similarity for each matching key
        similarities = []
        for key in matching_keys:
            env_value = environment[key]
            formation_value = self.formation_environment[key]
            
            # Calculate similarity based on value type
            if isinstance(env_value, (int, float)) and isinstance(formation_value, (int, float)):
                # Numeric similarity: 1.0 - normalized absolute difference
                max_val = max(abs(env_value), abs(formation_value))
                if max_val == 0:
                    similarities.append(1.0)  # Both values are 0
                else:
                    diff = abs(env_value - formation_value) / max_val
                    similarities.append(max(0.0, 1.0 - diff))
            elif isinstance(env_value, str) and isinstance(formation_value, str):
                # String similarity: ratio of matching characters
                if not env_value and not formation_value:
                    similarities.append(1.0)  # Both strings are empty
                elif not env_value or not formation_value:
                    similarities.append(0.0)  # One string is empty
                else:
                    # Simple string similarity
                    max_len = max(len(env_value), len(formation_value))
                    min_len = min(len(env_value), len(formation_value))
                    prefix_match = 0
                    for i in range(min_len):
                        if env_value[i] == formation_value[i]:
                            prefix_match += 1
                        else:
                            break
                    similarities.append(prefix_match / max_len)
            elif isinstance(env_value, bool) and isinstance(formation_value, bool):
                # Boolean similarity: 1.0 if equal, 0.0 if not
                similarities.append(1.0 if env_value == formation_value else 0.0)
            else:
                # Different types: 0.0 similarity
                similarities.append(0.0)
        
        # Overall similarity is weighted average of key similarities
        if not similarities:
            return 0.0
        
        return sum(similarities) / len(similarities)
    
    def should_activate(self, environment: Dict[str, Any], threshold: float = 0.7) -> bool:
        """
        Determine if the trigger should activate based on environmental similarity.
        
        Args:
            environment: Current environmental conditions.
            threshold: Similarity threshold for activation.
            
        Returns:
            True if the trigger should activate, False otherwise.
        """
        similarity = self.calculate_similarity(environment)
        should_activate = similarity >= threshold
        
        if should_activate:
            self.activation_count += 1
            self.logger.info(f"Activating trigger {self.id} (similarity: {similarity:.2f})")
        
        return should_activate
    
    def update_fitness(self, performance: float) -> None:
        """
        Update the trigger's fitness score based on performance feedback.
        
        Args:
            performance: Performance score between 0.0 and 1.0.
        """
        # Update success rate using exponential moving average
        alpha = 0.3  # Weight for new performance
        self.success_rate = alpha * performance + (1 - alpha) * self.success_rate
        
        # Update fitness score based on success rate and activation count
        activation_factor = min(self.activation_count / 10, 1.0)  # Cap at 10 activations
        self.fitness_score = self.success_rate * (0.5 + 0.5 * activation_factor)
        
        self.logger.info(f"Updated trigger {self.id} fitness to {self.fitness_score:.2f}")
    
    def mutate(self) -> 'GeneticTrigger':
        """
        Create a mutated copy of this trigger.
        
        Returns:
            A new GeneticTrigger with mutations.
        """
        # Create a copy of the formation environment
        mutated_environment = dict(self.formation_environment)
        
        # Mutate some values
        for key in mutated_environment:
            if random.random() < self.mutation_rate:
                value = mutated_environment[key]
                
                # Mutate based on value type
                if isinstance(value, float):
                    # Add Gaussian noise
                    mutated_environment[key] = value + random.gauss(0, 0.1)
                elif isinstance(value, int):
                    # Add or subtract a small integer
                    mutated_environment[key] = value + random.randint(-2, 2)
                elif isinstance(value, str):
                    # No string mutation for now
                    pass
                elif isinstance(value, bool):
                    # Flip boolean with low probability
                    if random.random() < 0.1:
                        mutated_environment[key] = not value
        
        # Create a copy of the pathway config
        mutated_pathway = dict(self.pathway_config)
        
        # Mutate some pathway parameters
        for key in mutated_pathway:
            if random.random() < self.mutation_rate:
                value = mutated_pathway[key]
                
                # Mutate based on value type
                if isinstance(value, float):
                    # Add Gaussian noise
                    mutated_pathway[key] = value + random.gauss(0, 0.05)
                elif isinstance(value, int):
                    # Add or subtract a small integer
                    mutated_pathway[key] = value + random.randint(-1, 1)
                elif isinstance(value, str):
                    # No string mutation for now
                    pass
                elif isinstance(value, bool):
                    # Flip boolean with low probability
                    if random.random() < 0.1:
                        mutated_pathway[key] = not value
        
        # Create a new trigger with the mutated values
        mutated_trigger = GeneticTrigger(
            formation_environment=mutated_environment,
            pathway_config=mutated_pathway,
            parent_ids=[self.id]
        )
        
        # Inherit some properties from parent
        mutated_trigger.mutation_rate = self.mutation_rate * random.uniform(0.9, 1.1)
        
        self.logger.info(f"Created mutated trigger {mutated_trigger.id} from {self.id}")
        return mutated_trigger
    
    @classmethod
    def crossover(cls, trigger1: 'GeneticTrigger', trigger2: 'GeneticTrigger') -> 'GeneticTrigger':
        """
        Create a new trigger by crossing over two parent triggers.
        
        Args:
            trigger1: First parent trigger.
            trigger2: Second parent trigger.
            
        Returns:
            A new GeneticTrigger created through crossover.
        """
        # Combine environment dictionaries
        combined_env = {}
        all_keys = set(trigger1.formation_environment.keys()) | set(trigger2.formation_environment.keys())
        
        for key in all_keys:
            # Choose value from either parent
            if key in trigger1.formation_environment and key in trigger2.formation_environment:
                # Both parents have this key, randomly choose or blend
                if random.random() < 0.5:
                    combined_env[key] = trigger1.formation_environment[key]
                else:
                    combined_env[key] = trigger2.formation_environment[key]
            elif key in trigger1.formation_environment:
                # Only first parent has this key
                combined_env[key] = trigger1.formation_environment[key]
            else:
                # Only second parent has this key
                combined_env[key] = trigger2.formation_environment[key]
        
        # Combine pathway configurations
        combined_pathway = {}
        all_keys = set(trigger1.pathway_config.keys()) | set(trigger2.pathway_config.keys())
        
        for key in all_keys:
            # Choose value from either parent
            if key in trigger1.pathway_config and key in trigger2.pathway_config:
                # Both parents have this key, randomly choose or blend
                if random.random() < 0.5:
                    combined_pathway[key] = trigger1.pathway_config[key]
                else:
                    combined_pathway[key] = trigger2.pathway_config[key]
            elif key in trigger1.pathway_config:
                # Only first parent has this key
                combined_pathway[key] = trigger1.pathway_config[key]
            else:
                # Only second parent has this key
                combined_pathway[key] = trigger2.pathway_config[key]
        
        # Create a new trigger with the combined values
        child_trigger = cls(
            formation_environment=combined_env,
            pathway_config=combined_pathway,
            parent_ids=[trigger1.id, trigger2.id]
        )
        
        # Inherit some properties from parents
        child_trigger.mutation_rate = (trigger1.mutation_rate + trigger2.mutation_rate) / 2
        
        logging.getLogger("GeneticTrigger").info(
            f"Created crossover trigger {child_trigger.id} from {trigger1.id} and {trigger2.id}"
        )
        return child_trigger
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the trigger to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the trigger.
        """
        return {
            "id": self.id,
            "dna_signature": self.dna_signature,
            "pathway_config": self.pathway_config,
            "creation_time": self.creation_time.isoformat(),
            "activation_count": self.activation_count,
            "success_rate": self.success_rate,
            "fitness_score": self.fitness_score,
            "mutation_rate": self.mutation_rate,
            "parent_ids": self.parent_ids,
            "formation_environment": self.formation_environment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneticTrigger':
        """
        Create a trigger from a dictionary representation.
        
        Args:
            data: Dictionary representation of a trigger.
            
        Returns:
            A new GeneticTrigger instance.
        """
        trigger = cls(
            formation_environment=data["formation_environment"],
            pathway_config=data["pathway_config"],
            parent_ids=data["parent_ids"]
        )
        
        trigger.id = data["id"]
        trigger.dna_signature = data["dna_signature"]
        trigger.creation_time = datetime.fromisoformat(data["creation_time"])
        trigger.activation_count = data["activation_count"]
        trigger.success_rate = data["success_rate"]
        trigger.fitness_score = data["fitness_score"]
        trigger.mutation_rate = data["mutation_rate"]
        
        return trigger