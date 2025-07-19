"""
Genetic Trigger System component for MCP System Upgrade.

This component handles environmental adaptation through genetic encoding
and selection, activating optimized pathways when similar conditions recur.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import time
import hashlib
import json

# Import from monitoring system
from ..monitoring_system import EnvironmentalState

# Import local components
from .genetic_trigger import GeneticTrigger
from .neural_genetic_processor import NeuralGeneticProcessor
from .code_genetic_processor import CodeGeneticProcessor
from .hormone_system_interface import HormoneSystemInterface
from .adaptive_mutation_controller import AdaptiveMutationController
from .epigenetic_memory import EpigeneticMemory


class GeneticCodon(Enum):
    """256-codon system for genetic encoding"""
    # Core codons (0-63)
    START = "ATG"
    STOP = "TAA"
    PAUSE = "TAG"
    SKIP = "TGA"
    
    # Expression codons (64-127)
    HIGH_EXPRESSION = "AAA"
    MED_EXPRESSION = "AAC"
    LOW_EXPRESSION = "AAG"
    NO_EXPRESSION = "AAT"
    
    # Regulatory codons (128-191)
    PROMOTER = "ACA"
    ENHANCER = "ACC"
    SILENCER = "ACG"
    TERMINATOR = "ACT"
    
    # Specialized codons (192-255)
    STRESS_RESPONSE = "AGA"
    LEARNING_MARKER = "AGC"
    ADAPTATION_SIGNAL = "AGG"
    EVOLUTION_TRIGGER = "AGT"


@dataclass
class EpigeneticMarker:
    """Epigenetic marker for gene expression modification"""
    methylation_pattern: Dict[str, float] = field(default_factory=dict)
    histone_modifications: Dict[str, float] = field(default_factory=dict)
    stress_markers: Dict[str, float] = field(default_factory=dict)
    imprinting_markers: Dict[str, bool] = field(default_factory=dict)
    chromatin_state: Dict[str, str] = field(default_factory=dict)
    
    def get_expression_modifier(self, gene_id: str) -> float:
        """Calculate expression modifier for a gene"""
        modifier = 1.0
        
        # Methylation effects (typically repressive)
        if gene_id in self.methylation_pattern:
            modifier *= (1.0 - self.methylation_pattern[gene_id] * 0.5)
        
        # Histone modification effects
        if gene_id in self.histone_modifications:
            modifier *= (1.0 + self.histone_modifications[gene_id] * 0.3)
        
        # Stress marker effects
        if gene_id in self.stress_markers:
            modifier *= (1.0 + self.stress_markers[gene_id] * 0.2)
        
        return max(0.1, min(3.0, modifier))


@dataclass
class GeneticSequence:
    """Genetic sequence with codon-based encoding"""
    sequence: str
    codons: List[str] = field(default_factory=list)
    promoter_regions: Dict[int, float] = field(default_factory=dict)
    enhancer_regions: Dict[int, float] = field(default_factory=dict)
    silencer_regions: Dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract codons from sequence"""
        if not self.codons:
            self.codons = [self.sequence[i:i+3] for i in range(0, len(self.sequence), 3)]
    
    def add_promoter(self, position: int, strength: float):
        """Add promoter region"""
        self.promoter_regions[position] = strength
    
    def add_enhancer(self, position: int, strength: float):
        """Add enhancer region"""
        self.enhancer_regions[position] = strength
    
    def get_expression_level(self, position: int) -> float:
        """Calculate expression level at position"""
        expression = 1.0
        
        # Promoter effects
        for pos, strength in self.promoter_regions.items():
            distance = abs(position - pos)
            if distance < 10:
                expression *= (1.0 + strength * (1.0 - distance / 10))
        
        # Enhancer effects
        for pos, strength in self.enhancer_regions.items():
            distance = abs(position - pos)
            if distance < 20:
                expression *= (1.0 + strength * 0.5 * (1.0 - distance / 20))
        
        return max(0.1, expression)
    
    def mutate(self, mutation_rate: float = 0.01) -> 'GeneticSequence':
        """Create mutated version of sequence"""
        import random
        
        new_sequence = ""
        for i, char in enumerate(self.sequence):
            if random.random() < mutation_rate:
                # Simple mutation: replace with different base
                bases = ['A', 'T', 'G', 'C']
                bases.remove(char)
                new_sequence += random.choice(bases)
            else:
                new_sequence += char
        
        return GeneticSequence(new_sequence)
    
    def crossover(self, other: 'GeneticSequence') -> tuple['GeneticSequence', 'GeneticSequence']:
        """Perform genetic crossover with another sequence"""
        import random
        
        # Single-point crossover
        crossover_point = random.randint(1, min(len(self.sequence), len(other.sequence)) - 1)
        
        child1_seq = self.sequence[:crossover_point] + other.sequence[crossover_point:]
        child2_seq = other.sequence[:crossover_point] + self.sequence[crossover_point:]
        
        return GeneticSequence(child1_seq), GeneticSequence(child2_seq)


class GeneticTriggerSystem:
    """Main genetic trigger system for environmental adaptation"""
    
    def __init__(self):
        self.triggers: Dict[str, GeneticTrigger] = {}
        self.environment_history: List[EnvironmentalState] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.active_triggers: List[str] = []
        
    async def update_environment(self, environment: EnvironmentalState):
        """Update current environment and evaluate triggers"""
        self.environment_history.append(environment)
        
        # Keep only recent history
        if len(self.environment_history) > 1000:
            self.environment_history = self.environment_history[-1000:]
        
        # Evaluate all triggers
        self.active_triggers = []
        for trigger_id, trigger in self.triggers.items():
            if await trigger.should_activate(environment):
                self.active_triggers.append(trigger_id)
    
    async def create_trigger(self, environment: EnvironmentalState, 
                           genetic_sequence: str, activation_threshold: float = 0.7) -> str:
        """Create new genetic trigger"""
        trigger_id = f"trigger_{len(self.triggers)}_{int(time.time())}"
        trigger = GeneticTrigger(trigger_id, environment, genetic_sequence, activation_threshold)
        self.triggers[trigger_id] = trigger
        return trigger_id
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_triggers': len(self.triggers),
            'active_triggers': len(self.active_triggers),
            'environment_history_size': len(self.environment_history),
            'average_performance': sum(self.performance_metrics.get('overall', [0.5])) / max(1, len(self.performance_metrics.get('overall', [0.5])))
        }


# Export all necessary classes
__all__ = [
    'EnvironmentalState',
    'EpigeneticMarker',
    'EpigeneticMemory',
    'GeneticSequence',
    'GeneticTrigger',
    'GeneticTriggerSystem',
    'GeneticCodon',
    'NeuralGeneticProcessor',
    'CodeGeneticProcessor',
    'HormoneSystemInterface',
    'AdaptiveMutationController'
]