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
import itertools

# Import enhanced genetic components (now implemented)
from .neural_genetic_processor import NeuralGeneticProcessor
from .code_genetic_processor import CodeGeneticProcessor
from .hormone_system_interface import HormoneSystemInterface
from .adaptive_mutation_controller import AdaptiveMutationController
from .epigenetic_memory import EpigeneticMemory
from ..neural_network_models.performance_tracker import PerformanceTracker

class GeneticTrigger:
    """
    Represents a genetic trigger that activates when specific environmental conditions are met.
    
    Implements dual code/neural activation, hormone feedback, and epigenetic memory.
    """
    
    # 256 unique codons (quadruplets of A/T/G/C)
    CODON_TABLE = [''.join(codon) for codon in itertools.product('ATGC', repeat=4)]  # 4^4 = 256
    CODON_TO_BYTE = {codon: i for i, codon in enumerate(CODON_TABLE)}

    @staticmethod
    def encode_dict_to_codon(data: Dict[str, Any]) -> str:
        """
        Encode a dictionary to a 256-codon DNA string using 4-letter codons.
        """
        import json
        b = json.dumps(data, sort_keys=True).encode('utf-8')
        return ''.join(GeneticTrigger.CODON_TABLE[byte] for byte in b)

    @staticmethod
    def decode_codon_to_dict(codon_str: str) -> Dict[str, Any]:
        """
        Decode a 256-codon DNA string (4-letter codons) back to a dictionary.
        """
        import json
        codons = [codon_str[i:i+4] for i in range(0, len(codon_str), 4)]
        b = bytes([GeneticTrigger.CODON_TO_BYTE[c] for c in codons])
        return json.loads(b.decode('utf-8'))

    def __init__(self, 
                 formation_environment: Dict[str, Any],
                 pathway_config: Dict[str, Any],
                 parent_ids: Optional[List[str]] = None):
        """
        Initialize a genetic trigger.
        
        Args:
            formation_environment: Environmental conditions that led to the trigger's creation.
            pathway_config: Neural pathway configuration associated with the trigger.
            parent_ids: IDs of parent triggers if created through crossover.
        """
        self.id = str(uuid.uuid4())
        # Use 256-codon encoding for dna_signature
        self.dna_signature = self.encode_dict_to_codon(formation_environment)
        self.pathway_config = pathway_config
        self.creation_time = datetime.now()
        self.activation_count = 0
        self.success_rate = 0.0
        self.fitness_score = 0.5  # Initial neutral fitness
        self.mutation_rate = 0.05  # 5% mutation rate
        self.parent_ids: List[str] = parent_ids if parent_ids is not None else []
        self.formation_environment = formation_environment
        
        self.logger = logging.getLogger("GeneticTrigger")
        self.logger.info(f"Created genetic trigger {self.id} with DNA signature: {self.dna_signature[:20]}...")
        
        # Dual implementation and integration
        self.code_impl = CodeGeneticProcessor()
        self.neural_impl = NeuralGeneticProcessor()
        self.performance_tracker = PerformanceTracker() if PerformanceTracker else None
        self.hormone_interface = HormoneSystemInterface()
        self.mutation_controller = AdaptiveMutationController()
        self.epigenetic_markers = EpigeneticMemory()
        # Split-brain A/B testing registration
        self.ab_test_group = None  # 'left' or 'right' for split-brain testing
    
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
    
    def _get_recommended_impl(self) -> str:
        """
        Get the recommended implementation ('neural' or 'algorithmic') from the performance tracker.
        Returns 'algorithmic' if no recommendation is available.
        """
        if self.performance_tracker:
            summary = self.performance_tracker.get_performance_summary("genetic_trigger_activation")
            rec = summary["functions"].get("genetic_trigger_activation", {}).get("recommended_implementation")
            if rec in ("neural", "algorithmic"):
                return rec
        return "algorithmic"

    async def should_activate(self, environment: Dict[str, Any], threshold: float = 0.7, resource_metrics: Dict[str, float] = None) -> bool:
        """
        Determine if the trigger should activate based on environmental similarity, using dual code/neural implementation.
        Triggers hormone feedback on activation and stores epigenetic marker.
        Adapts to resource metrics (CPU, memory, activity).
        """
        # Resource-aware adaptation
        if resource_metrics:
            cpu = resource_metrics.get('cpu', 0.0)
            mem = resource_metrics.get('memory', 0.0)
            activity = resource_metrics.get('activity', 0.0)
            # Always update mutation rate and threshold if high resource
            if cpu > 0.8 or mem > 0.8:
                old_rate = self.mutation_rate
                self.mutation_rate = min(0.2, max(0.05, self.mutation_rate * 1.5))
                threshold = min(0.95, threshold + 0.1)
                self.logger.info(f"[TEST] Resource high: cpu={cpu}, mem={mem}. Increased mutation_rate from {old_rate} to {self.mutation_rate}, threshold to {threshold}")
            if mem > 0.95:
                self.logger.info("[TEST] Memory very high: triggering memory consolidation/pruning.")
        impl = self._get_recommended_impl()
        if impl == "neural" and self.neural_impl:
            result = await self.neural_impl.evaluate_activation(environment)
        elif impl == "algorithmic" and self.code_impl:
            result = self.code_impl.evaluate_activation(environment, self.formation_environment, threshold)
        else:
            result = self.calculate_similarity(environment) >= threshold
        if self.hormone_interface:
            await self.hormone_interface.release_genetic_hormones(result)
        if result and self.epigenetic_markers:
            marker_key = f"activation_{self.id}"
            encoded_env = self.encode_dict_to_codon(environment)
            self.epigenetic_markers.set_marker(marker_key, encoded_env)
            self.logger.info(f"Epigenetic marker set: {marker_key}")
        return result
    
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
    
    def receive_memory_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Receive feedback from memory or hormone systems to adapt trigger parameters and epigenetic markers.
        Args:
            feedback: Dictionary with feedback information (e.g., performance, tier transitions, hormone levels).
        """
        self.logger.info(f"[GeneticTrigger] Received memory/hormone feedback: {feedback}")
        # Example: update fitness or mutation rate based on feedback
        if 'performance' in feedback:
            self.update_fitness(feedback['performance'])
        if 'mutation_rate' in feedback:
            self.mutation_rate = feedback['mutation_rate']
        # Store hormone feedback as epigenetic marker
        if 'hormone_levels' in feedback and self.epigenetic_markers:
            marker_key = f"hormone_feedback_{self.id}_{datetime.now().isoformat()}"
            encoded_hormone = self.encode_dict_to_codon(feedback['hormone_levels'])
            self.epigenetic_markers.set_marker(marker_key, encoded_hormone)
            self.logger.info(f"Epigenetic marker set for hormone feedback: {marker_key}")
        # Adapt mutation rate based on hormone feedback (e.g., high cortisol -> higher mutation)
        if 'hormone_levels' in feedback and self.mutation_controller:
            cortisol = feedback['hormone_levels'].get('cortisol', 0.0)
            # Example: if cortisol > 0.7, increase mutation rate
            if cortisol > 0.7:
                self.mutation_rate = min(0.2, self.mutation_rate * 1.2)
                self.logger.info(f"Increased mutation rate due to high cortisol: {self.mutation_rate}")
        # Extend with more adaptation logic as needed

    def register_ab_test_group(self, group: str) -> None:
        """
        Register this trigger for split-brain A/B testing ('left' or 'right').
        """
        assert group in ('left', 'right')
        self.ab_test_group = group

    def force_resource_adaptation(self, resource_metrics: Dict[str, float], threshold: float = 0.7) -> float:
        """
        For testing: directly apply resource adaptation logic and return new mutation rate.
        """
        cpu = resource_metrics.get('cpu', 0.0)
        mem = resource_metrics.get('memory', 0.0)
        activity = resource_metrics.get('activity', 0.0)
        if cpu > 0.8 or mem > 0.8:
            old_rate = self.mutation_rate
            self.mutation_rate = min(0.2, max(0.05, self.mutation_rate * 1.5))
            threshold = min(0.95, threshold + 0.1)
            self.logger.info(f"[TEST] Resource high: cpu={cpu}, mem={mem}. Increased mutation_rate from {old_rate} to {self.mutation_rate}, threshold to {threshold}")
        if mem > 0.95:
            self.logger.info("[TEST] Memory very high: triggering memory consolidation/pruning.")
        return self.mutation_rate

    @staticmethod
    def multi_point_crossover(parent1: 'GeneticTrigger', parent2: 'GeneticTrigger', points: int = 2) -> 'GeneticTrigger':
        """
        Perform multi-point crossover between two parent triggers to produce a child.
        Handles empty/mismatched environments and avoids index errors.
        Args:
            parent1, parent2: Parent GeneticTrigger instances.
            points: Number of crossover points.
        Returns:
            A new GeneticTrigger child.
        """
        import random, logging
        env1 = list(parent1.formation_environment.items())
        env2 = list(parent2.formation_environment.items())
        if not env1 and not env2:
            logging.warning("Both parents have empty environments; returning parent1 copy.")
            return GeneticTrigger({}, {}, parent_ids=[parent1.id, parent2.id])
        length = min(len(env1), len(env2))
        if length < 2:
            logging.warning("Environments too short for crossover; merging environments.")
            child_env = dict(env1 + env2)
        else:
            crossover_points = sorted(random.sample(range(1, length), min(points, length-1)))
            child_env = {}
            toggle = False
            i = 0
            for idx in range(length):
                if i < len(crossover_points) and idx == crossover_points[i]:
                    toggle = not toggle
                    i += 1
                if toggle:
                    child_env[env2[idx][0]] = env2[idx][1]
                else:
                    child_env[env1[idx][0]] = env1[idx][1]
            for k, v in (env1 + env2):
                if k not in child_env:
                    child_env[k] = v
        child_pathway = {**parent1.pathway_config, **parent2.pathway_config}
        return GeneticTrigger(child_env, child_pathway, parent_ids=[parent1.id, parent2.id])

    def evaluate_fitness_landscape(self, test_environments: list[dict]) -> float:
        """
        Evaluate fitness over a landscape of test environments.
        Args:
            test_environments: List of environment dicts.
        Returns:
            Average activation rate (float).
        """
        activations = 0
        for env in test_environments:
            if self.code_impl.evaluate_activation(env, self.formation_environment):
                activations += 1
        return activations / len(test_environments) if test_environments else 0.0

    def cross_pollinate(self, chrom1, chrom2):
        """
        Cross-pollinate two chromosomes based on compatibility and diversity metrics.
        Returns a new child chromosome.
        """
        ids1 = {e.element_id for e in chrom1.elements}
        ids2 = {e.element_id for e in chrom2.elements}
        compatibility = len(ids1 & ids2) / max(1, len(ids1 | ids2))
        diversity = 1.0 - compatibility
        child_elements = []
        for e1, e2 in zip(chrom1.elements, chrom2.elements):
            chosen = e1 if random.random() < diversity else e2
            child_elements.append(chosen)
        longer = chrom1.elements if len(chrom1.elements) > len(chrom2.elements) else chrom2.elements
        child_elements += longer[len(child_elements):]
        lineage = chrom1.chromosome_id + "|" + chrom2.chromosome_id
        return type(chrom1)(
            chromosome_id=f"child_{hash(lineage)}",
            elements=child_elements,
            telomere_length=min(chrom1.telomere_length, chrom2.telomere_length),
            crossover_hotspots=[len(child_elements)//2],
            structural_variants=[{"lineage": lineage, "compatibility": compatibility, "diversity": diversity}]
        )

    def evaluate_fitness_multiobjective(self, chromosome, objectives):
        score = 0.0
        for k, v in objectives.items():
            score += v
        return min(1.0, max(0.0, score / (len(objectives) or 1)))

    def calculate_diversity(self, population):
        if len(population) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                ids1 = {e.element_id for e in population[i].elements}
                ids2 = {e.element_id for e in population[j].elements}
                total += 1.0 - (len(ids1 & ids2) / max(1, len(ids1 | ids2)))
                count += 1
        return total / count if count else 0.0

    def track_lineage(self, chromosome):
        return [sv.get("lineage") for sv in chromosome.structural_variants if "lineage" in sv]

    def rollback_to_ancestor(self, chromosome, ancestor_id):
        if any(ancestor_id in (sv.get("lineage") or "") for sv in chromosome.structural_variants):
            return chromosome
        else:
            return chromosome

    def neighborhood_analysis(self, population, target):
        return [c for c in population if c != target][:3]