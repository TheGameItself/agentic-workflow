"""
Genetic Trigger System - Brain-inspired environmental adaptation using genetic algorithms

This module implements a sophisticated genetic trigger system that uses biological
concepts like DNA encoding, codon-based activation, epigenetic memory, and evolutionary
adaptation to create neural network triggers that activate based on environmental
conditions.

Key Features:
- Environmental state encoding as DNA-like sequences
- Codon-based activation patterns (64 genetic codons mapped to behaviors)
- Epigenetic memory with methylation and histone modifications
- Evolutionary adaptation with natural selection, mutation, and crossover
- Horizontal gene transfer between system components
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

# numpy not available, using built-in math functions


class GeneticCodon(Enum):
    """Standard genetic codons mapped to system behaviors"""
    # Start/Stop codons
    ATG = "start_pathway"  # Start codon - Initialize new learning pathways
    TAA = "stop_process"   # Stop codon - Terminate processes
    TAG = "pause_process"  # Stop codon - Pause processes  
    TGA = "halt_process"   # Stop codon - Halt processes
    
    # Amino acid codons mapped to behaviors
    GCN = "structural_stability"    # Alanine - Structural stability
    CGN = "activation_signal"       # Arginine - Positive charge/activation
    AAR = "basic_adaptation"        # Lysine - Basic environment adaptation
    GAR = "acidic_response"         # Glutamic Acid - Acidic environment
    CAR = "resource_management"     # Histidine - Resource management
    UUR = "memory_consolidation"    # Leucine - Memory consolidation
    CUN = "pattern_recognition"     # Leucine - Pattern recognition
    AUG = "learning_initiation"     # Methionine - Learning initiation
    UUY = "attention_focus"         # Phenylalanine - Attention focus
    CCN = "coordination"            # Proline - Coordination
    UCN = "communication"           # Serine - Communication
    ACN = "adaptation"              # Threonine - Adaptation
    UGG = "creative_synthesis"      # Tryptophan - Creative synthesis
    UAY = "error_detection"         # Tyrosine - Error detection
    GUN = "decision_making"         # Valine - Decision making


@dataclass
class EnvironmentalState:
    """Represents the current environmental conditions"""
    task_complexity: float = 0.0
    user_satisfaction: float = 0.0
    system_load: float = 0.0
    error_rate: float = 0.0
    learning_rate: float = 0.0
    collaboration_level: float = 0.0
    creativity_demand: float = 0.0
    time_pressure: float = 0.0
    resource_availability: float = 1.0
    context_switches: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_complexity': self.task_complexity,
            'user_satisfaction': self.user_satisfaction,
            'system_load': self.system_load,
            'error_rate': self.error_rate,
            'learning_rate': self.learning_rate,
            'collaboration_level': self.collaboration_level,
            'creativity_demand': self.creativity_demand,
            'time_pressure': self.time_pressure,
            'resource_availability': self.resource_availability,
            'context_switches': self.context_switches,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentalState':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class EpigeneticMarker:
    """Represents epigenetic modifications that affect gene expression"""
    methylation_pattern: Dict[str, float] = field(default_factory=dict)
    histone_modifications: Dict[str, float] = field(default_factory=dict)
    non_coding_rna: Dict[str, float] = field(default_factory=dict)
    imprinting_markers: Dict[str, bool] = field(default_factory=dict)
    stress_markers: Dict[str, float] = field(default_factory=dict)
    
    def get_expression_modifier(self, gene_region: str) -> float:
        """Calculate expression modification factor for a gene region"""
        modifier = 1.0
        
        # Methylation typically suppresses expression
        if gene_region in self.methylation_pattern:
            modifier *= (1.0 - self.methylation_pattern[gene_region] * 0.5)
        
        # Histone modifications can enhance or suppress
        if gene_region in self.histone_modifications:
            modifier *= (1.0 + self.histone_modifications[gene_region])
        
        # Non-coding RNA fine-tunes expression
        if gene_region in self.non_coding_rna:
            modifier *= (1.0 + self.non_coding_rna[gene_region] * 0.3)
        
        # Stress markers can dramatically alter expression
        if gene_region in self.stress_markers:
            stress_factor = self.stress_markers[gene_region]
            if stress_factor > 0.7:  # High stress
                modifier *= 1.5  # Upregulate stress response genes
            elif stress_factor < 0.3:  # Low stress
                modifier *= 0.8  # Downregulate stress response genes
        
        return max(0.1, min(3.0, modifier))  # Constrain bounds


class EpigeneticMemory:
    """Manages epigenetic memory and inheritance mechanisms"""
    
    def __init__(self):
        self.markers = EpigeneticMarker()
        self.inheritance_history = []
        self.adaptation_memory = {}
        
    def add_stress_marker(self, gene_region: str, stress_level: float):
        """Add stress-induced epigenetic marker"""
        self.markers.stress_markers[gene_region] = stress_level
        
        # Stress can cause methylation changes
        if stress_level > 0.8:
            self.markers.methylation_pattern[gene_region] = min(
                1.0, self.markers.methylation_pattern.get(gene_region, 0) + 0.2
            )
    
    def add_learning_marker(self, gene_region: str, learning_success: float):
        """Add learning-induced histone modifications"""
        if learning_success > 0.6:
            # Successful learning enhances expression
            self.markers.histone_modifications[gene_region] = min(
                1.0, self.markers.histone_modifications.get(gene_region, 0) + 0.1
            )
    
    def inherit_from_parent(self, parent_memory: 'EpigeneticMemory'):
        """Inherit epigenetic markers from parent system"""
        # Inherit some methylation patterns
        for region, value in parent_memory.markers.methylation_pattern.items():
            if random.random() < 0.7:  # 70% inheritance rate
                self.markers.methylation_pattern[region] = value * 0.8
        
        # Inherit imprinting markers
        self.markers.imprinting_markers.update(
            parent_memory.markers.imprinting_markers
        )
        
        self.inheritance_history.append({
            'timestamp': time.time(),
            'parent_id': id(parent_memory),
            'inherited_regions': list(parent_memory.markers.methylation_pattern.keys())
        })
    
    def get_expression_modifier(self, gene_region: str = "default") -> float:
        """Get expression modification factor"""
        return self.markers.get_expression_modifier(gene_region)


class GeneticSequence:
    """Represents a genetic sequence with codon-based encoding"""
    
    def __init__(self, sequence: str = ""):
        self.sequence = sequence.upper().replace('U', 'T')  # Convert RNA to DNA
        self.codons = self._extract_codons()
        self.promoter_regions = {}
        self.enhancer_regions = {}
        
    def _extract_codons(self) -> List[str]:
        """Extract codons from sequence"""
        codons = []
        for i in range(0, len(self.sequence) - 2, 3):
            codon = self.sequence[i:i+3]
            if len(codon) == 3:
                codons.append(codon)
        return codons
    
    def add_promoter(self, position: int, strength: float):
        """Add promoter region that regulates expression"""
        self.promoter_regions[position] = strength
    
    def add_enhancer(self, position: int, strength: float):
        """Add enhancer region that boosts expression"""
        self.enhancer_regions[position] = strength
    
    def get_expression_level(self, position: int) -> float:
        """Calculate expression level at given position"""
        base_level = 0.5
        
        # Find nearest promoter
        promoter_effect = 0.0
        for pos, strength in self.promoter_regions.items():
            distance = abs(position - pos)
            if distance < 100:  # Promoter effective range
                promoter_effect += strength * (1.0 - distance / 100.0)
        
        # Find enhancer effects
        enhancer_effect = 0.0
        for pos, strength in self.enhancer_regions.items():
            distance = abs(position - pos)
            if distance < 200:  # Enhancer effective range
                enhancer_effect += strength * (1.0 - distance / 200.0)
        
        return min(2.0, base_level + promoter_effect + enhancer_effect)
    
    def mutate(self, mutation_rate: float = 0.01) -> 'GeneticSequence':
        """Create mutated copy of sequence"""
        bases = ['A', 'T', 'G', 'C']
        mutated_sequence = ""
        
        for base in self.sequence:
            if random.random() < mutation_rate:
                # Point mutation
                mutated_sequence += random.choice(bases)
            else:
                mutated_sequence += base
        
        # Occasional insertions/deletions
        if random.random() < mutation_rate * 0.1:
            # Insertion
            pos = random.randint(0, len(mutated_sequence))
            mutated_sequence = (mutated_sequence[:pos] + 
                              random.choice(bases) + 
                              mutated_sequence[pos:])
        
        if random.random() < mutation_rate * 0.1 and len(mutated_sequence) > 3:
            # Deletion
            pos = random.randint(0, len(mutated_sequence) - 1)
            mutated_sequence = mutated_sequence[:pos] + mutated_sequence[pos+1:]
        
        new_sequence = GeneticSequence(mutated_sequence)
        new_sequence.promoter_regions = self.promoter_regions.copy()
        new_sequence.enhancer_regions = self.enhancer_regions.copy()
        
        return new_sequence
    
    def crossover(self, other: 'GeneticSequence') -> Tuple['GeneticSequence', 'GeneticSequence']:
        """Perform genetic crossover with another sequence"""
        min_len = min(len(self.sequence), len(other.sequence))
        if min_len < 6:
            return self, other
        
        crossover_point = random.randint(3, min_len - 3)
        
        child1_seq = self.sequence[:crossover_point] + other.sequence[crossover_point:]
        child2_seq = other.sequence[:crossover_point] + self.sequence[crossover_point:]
        
        child1 = GeneticSequence(child1_seq)
        child2 = GeneticSequence(child2_seq)
        
        # Inherit regulatory regions
        child1.promoter_regions.update(self.promoter_regions)
        child1.enhancer_regions.update(other.enhancer_regions)
        child2.promoter_regions.update(other.promoter_regions)
        child2.enhancer_regions.update(self.enhancer_regions)
        
        return child1, child2


class GeneticTrigger:
    """
    Main genetic trigger class that activates based on environmental conditions
    using genetic algorithms and biological inspiration
    """
    
    def __init__(self, formation_environment: EnvironmentalState, 
                 genetic_sequence: Optional[GeneticSequence] = None,
                 trigger_id: Optional[str] = None):
        self.trigger_id = trigger_id or self._generate_id()
        self.formation_environment = formation_environment
        self.dna_signature = self._encode_environment(formation_environment)
        self.genetic_sequence = genetic_sequence or self._generate_sequence()
        self.codon_map = self._build_codon_activation_map()
        self.epigenetic_markers = EpigeneticMemory()
        self.expression_level = 0.0
        self.activation_threshold = 0.6
        self.fitness_score = 0.5
        self.activation_history = []
        self.performance_history = []
        self.generation = 0
        self.parent_ids = []
        
    def _generate_id(self) -> str:
        """Generate unique trigger ID"""
        return hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
    
    def _encode_environment(self, env: EnvironmentalState) -> str:
        """Convert environmental state to DNA-like sequence"""
        # Normalize environmental values to 0-3 range (4 DNA bases)
        base_map = ['A', 'T', 'G', 'C']
        
        def value_to_base(value: float) -> str:
            # Convert float to base index
            normalized = max(0.0, min(1.0, value))
            index = int(normalized * 3.99)  # 0-3 range
            return base_map[index]
        
        # Encode each environmental parameter
        sequence = ""
        sequence += value_to_base(env.task_complexity) * 3
        sequence += value_to_base(env.user_satisfaction) * 3
        sequence += value_to_base(env.system_load) * 3
        sequence += value_to_base(env.error_rate) * 3
        sequence += value_to_base(env.learning_rate) * 3
        sequence += value_to_base(env.collaboration_level) * 3
        sequence += value_to_base(env.creativity_demand) * 3
        sequence += value_to_base(env.time_pressure) * 3
        sequence += value_to_base(env.resource_availability) * 3
        
        # Add context switches as special pattern
        switches_normalized = min(1.0, env.context_switches / 10.0)
        sequence += value_to_base(switches_normalized) * 3
        
        return sequence
    
    def _generate_sequence(self) -> GeneticSequence:
        """Generate initial genetic sequence"""
        # Create sequence based on formation environment
        base_sequence = self.dna_signature
        
        # Add some random variation
        bases = ['A', 'T', 'G', 'C']
        for _ in range(random.randint(10, 30)):
            base_sequence += random.choice(bases)
        
        sequence = GeneticSequence(base_sequence)
        
        # Add regulatory regions
        sequence.add_promoter(0, 0.8)  # Strong promoter at start
        sequence.add_enhancer(len(base_sequence) // 2, 0.6)  # Mid-sequence enhancer
        
        return sequence
    
    def _build_codon_activation_map(self) -> Dict[str, float]:
        """Build mapping from codons to activation strengths"""
        codon_map = {}
        
        for codon in self.genetic_sequence.codons:
            # Map codon to activation strength based on formation environment
            if codon == "ATG":  # Start codon
                codon_map[codon] = 1.0
            elif codon in ["TAA", "TAG", "TGA"]:  # Stop codons
                codon_map[codon] = 0.0
            else:
                # Calculate activation based on environmental context
                base_activation = 0.5
                
                # Modify based on formation environment characteristics
                if self.formation_environment.task_complexity > 0.7:
                    if codon.startswith('G'):  # G-rich codons for complexity
                        base_activation += 0.3
                
                if self.formation_environment.creativity_demand > 0.6:
                    if 'TGG' in codon:  # Tryptophan codon for creativity
                        base_activation += 0.4
                
                if self.formation_environment.error_rate > 0.5:
                    if codon.startswith('TA'):  # Error detection codons
                        base_activation += 0.3
                
                codon_map[codon] = min(1.0, base_activation)
        
        return codon_map
    
    def should_activate(self, current_environment: EnvironmentalState) -> bool:
        """Determine if trigger should activate based on current environment"""
        # Calculate environmental similarity
        similarity = self._calculate_environmental_similarity(current_environment)
        
        # Calculate codon match score
        codon_match = self._match_genetic_sequence(current_environment)
        
        # Get epigenetic influence
        epigenetic_influence = self.epigenetic_markers.get_expression_modifier()
        
        # Calculate final activation score
        activation_score = similarity * codon_match * epigenetic_influence
        
        # Record activation attempt
        self.activation_history.append({
            'timestamp': time.time(),
            'environment': current_environment.to_dict(),
            'similarity': similarity,
            'codon_match': codon_match,
            'epigenetic_influence': epigenetic_influence,
            'activation_score': activation_score,
            'activated': activation_score > self.activation_threshold
        })
        
        # Update expression level
        self.expression_level = activation_score
        
        return activation_score > self.activation_threshold
    
    def _calculate_environmental_similarity(self, current_env: EnvironmentalState) -> float:
        """Calculate similarity between current and formation environments"""
        formation = self.formation_environment
        
        # Calculate weighted similarity across all parameters
        similarities = []
        weights = []
        
        # Task complexity similarity
        similarities.append(1.0 - abs(formation.task_complexity - current_env.task_complexity))
        weights.append(0.15)
        
        # User satisfaction similarity
        similarities.append(1.0 - abs(formation.user_satisfaction - current_env.user_satisfaction))
        weights.append(0.12)
        
        # System load similarity
        similarities.append(1.0 - abs(formation.system_load - current_env.system_load))
        weights.append(0.10)
        
        # Error rate similarity
        similarities.append(1.0 - abs(formation.error_rate - current_env.error_rate))
        weights.append(0.15)
        
        # Learning rate similarity
        similarities.append(1.0 - abs(formation.learning_rate - current_env.learning_rate))
        weights.append(0.12)
        
        # Collaboration level similarity
        similarities.append(1.0 - abs(formation.collaboration_level - current_env.collaboration_level))
        weights.append(0.10)
        
        # Creativity demand similarity
        similarities.append(1.0 - abs(formation.creativity_demand - current_env.creativity_demand))
        weights.append(0.10)
        
        # Time pressure similarity
        similarities.append(1.0 - abs(formation.time_pressure - current_env.time_pressure))
        weights.append(0.08)
        
        # Resource availability similarity
        similarities.append(1.0 - abs(formation.resource_availability - current_env.resource_availability))
        weights.append(0.08)
        
        # Weighted average
        weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
        
        return max(0.0, min(1.0, weighted_similarity))
    
    def _match_genetic_sequence(self, current_env: EnvironmentalState) -> float:
        """Match current environment against genetic sequence patterns"""
        current_signature = self._encode_environment(current_env)
        
        # Calculate sequence similarity using longest common subsequence
        lcs_length = self._longest_common_subsequence(self.dna_signature, current_signature)
        max_length = max(len(self.dna_signature), len(current_signature))
        
        if max_length == 0:
            return 0.0
        
        sequence_similarity = lcs_length / max_length
        
        # Calculate codon activation score
        current_sequence = GeneticSequence(current_signature)
        codon_activation = 0.0
        total_codons = len(current_sequence.codons)
        
        if total_codons > 0:
            for codon in current_sequence.codons:
                if codon in self.codon_map:
                    codon_activation += self.codon_map[codon]
            codon_activation /= total_codons
        
        # Combine sequence similarity and codon activation
        return (sequence_similarity * 0.6 + codon_activation * 0.4)
    
    def _longest_common_subsequence(self, seq1: str, seq2: str) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def update_fitness(self, performance_score: float):
        """Update fitness based on performance"""
        self.performance_history.append({
            'timestamp': time.time(),
            'performance': performance_score
        })
        
        # Calculate fitness as weighted average of recent performance
        recent_performances = [p['performance'] for p in self.performance_history[-10:]]
        self.fitness_score = sum(recent_performances) / len(recent_performances)
        
        # Update epigenetic markers based on performance
        if performance_score > 0.7:
            self.epigenetic_markers.add_learning_marker("performance", performance_score)
        elif performance_score < 0.3:
            self.epigenetic_markers.add_stress_marker("performance", 1.0 - performance_score)
    
    def mutate(self, mutation_rate: float = 0.01) -> 'GeneticTrigger':
        """Create mutated copy of this trigger"""
        mutated_sequence = self.genetic_sequence.mutate(mutation_rate)
        
        # Create new trigger with mutated sequence
        new_trigger = GeneticTrigger(
            self.formation_environment,
            mutated_sequence,
            f"{self.trigger_id}_mut_{random.randint(1000, 9999)}"
        )
        
        # Inherit some characteristics
        new_trigger.activation_threshold = self.activation_threshold + random.gauss(0, 0.05)
        new_trigger.activation_threshold = max(0.1, min(0.9, new_trigger.activation_threshold))
        new_trigger.generation = self.generation + 1
        new_trigger.parent_ids = [self.trigger_id]
        
        # Inherit epigenetic markers with some variation
        new_trigger.epigenetic_markers.inherit_from_parent(self.epigenetic_markers)
        
        return new_trigger
    
    def crossover(self, other: 'GeneticTrigger') -> Tuple['GeneticTrigger', 'GeneticTrigger']:
        """Perform genetic crossover with another trigger"""
        child1_seq, child2_seq = self.genetic_sequence.crossover(other.genetic_sequence)
        
        # Create child triggers
        child1 = GeneticTrigger(
            self.formation_environment,
            child1_seq,
            f"{self.trigger_id}_{other.trigger_id}_c1_{random.randint(1000, 9999)}"
        )
        
        child2 = GeneticTrigger(
            other.formation_environment,
            child2_seq,
            f"{self.trigger_id}_{other.trigger_id}_c2_{random.randint(1000, 9999)}"
        )
        
        # Inherit characteristics
        child1.activation_threshold = (self.activation_threshold + other.activation_threshold) / 2
        child2.activation_threshold = (self.activation_threshold + other.activation_threshold) / 2
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        child1.parent_ids = [self.trigger_id, other.trigger_id]
        child2.parent_ids = [self.trigger_id, other.trigger_id]
        
        # Inherit epigenetic markers
        child1.epigenetic_markers.inherit_from_parent(self.epigenetic_markers)
        child2.epigenetic_markers.inherit_from_parent(other.epigenetic_markers)
        
        return child1, child2
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize trigger to dictionary"""
        return {
            'trigger_id': self.trigger_id,
            'formation_environment': self.formation_environment.to_dict(),
            'dna_signature': self.dna_signature,
            'genetic_sequence': self.genetic_sequence.sequence,
            'activation_threshold': self.activation_threshold,
            'fitness_score': self.fitness_score,
            'expression_level': self.expression_level,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'codon_map': self.codon_map
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneticTrigger':
        """Deserialize trigger from dictionary"""
        formation_env = EnvironmentalState.from_dict(data['formation_environment'])
        genetic_seq = GeneticSequence(data['genetic_sequence'])
        
        trigger = cls(formation_env, genetic_seq, data['trigger_id'])
        trigger.activation_threshold = data['activation_threshold']
        trigger.fitness_score = data['fitness_score']
        trigger.expression_level = data['expression_level']
        trigger.generation = data['generation']
        trigger.parent_ids = data['parent_ids']
        trigger.codon_map = data['codon_map']
        
        return trigger


class GeneticTriggerPopulation:
    """Manages a population of genetic triggers with evolutionary algorithms"""
    
    def __init__(self, population_size: int = 50, database_path: str = "data/genetic_triggers.db"):
        self.population_size = population_size
        self.database_path = Path(database_path)
        self.triggers: List[GeneticTrigger] = []
        self.generation = 0
        self.selection_pressure = 0.3  # Top 30% survive
        self.mutation_rate = 0.02
        self.crossover_rate = 0.7
        self.elite_size = 5  # Number of elite individuals to preserve
        
        self._initialize_database()
        self._load_population()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS genetic_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    generation INTEGER,
                    fitness_score REAL,
                    trigger_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id TEXT,
                    environment_data TEXT,
                    activation_score REAL,
                    activated BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers (trigger_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id TEXT,
                    performance_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers (trigger_id)
                )
            """)
    
    def _load_population(self):
        """Load existing population from database"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT trigger_data FROM genetic_triggers 
                ORDER BY fitness_score DESC 
                LIMIT ?
            """, (self.population_size,))
            
            for (trigger_data,) in cursor.fetchall():
                try:
                    trigger_dict = json.loads(trigger_data)
                    trigger = GeneticTrigger.from_dict(trigger_dict)
                    self.triggers.append(trigger)
                except Exception as e:
                    logging.warning(f"Failed to load trigger: {e}")
        
        # Initialize population if empty
        if len(self.triggers) < self.population_size // 2:
            self._initialize_random_population()
    
    def _initialize_random_population(self):
        """Initialize population with random triggers"""
        while len(self.triggers) < self.population_size:
            # Create random environmental state
            env = EnvironmentalState(
                task_complexity=random.random(),
                user_satisfaction=random.random(),
                system_load=random.random(),
                error_rate=random.random(),
                learning_rate=random.random(),
                collaboration_level=random.random(),
                creativity_demand=random.random(),
                time_pressure=random.random(),
                resource_availability=random.random(),
                context_switches=random.randint(0, 10)
            )
            
            trigger = GeneticTrigger(env)
            self.triggers.append(trigger)
    
    def add_trigger(self, trigger: GeneticTrigger):
        """Add new trigger to population"""
        self.triggers.append(trigger)
        self._save_trigger(trigger)
        
        # Maintain population size
        if len(self.triggers) > self.population_size * 1.2:
            self._cull_population()
    
    def _save_trigger(self, trigger: GeneticTrigger):
        """Save trigger to database"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO genetic_triggers 
                (trigger_id, generation, fitness_score, trigger_data, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                trigger.trigger_id,
                trigger.generation,
                trigger.fitness_score,
                json.dumps(trigger.to_dict())
            ))
    
    def evaluate_triggers(self, environment: EnvironmentalState) -> List[GeneticTrigger]:
        """Evaluate all triggers against current environment"""
        activated_triggers = []
        
        for trigger in self.triggers:
            if trigger.should_activate(environment):
                activated_triggers.append(trigger)
        
        return activated_triggers
    
    def update_fitness(self, trigger_id: str, performance_score: float):
        """Update fitness for specific trigger"""
        for trigger in self.triggers:
            if trigger.trigger_id == trigger_id:
                trigger.update_fitness(performance_score)
                self._save_trigger(trigger)
                
                # Save performance history
                with sqlite3.connect(self.database_path) as conn:
                    conn.execute("""
                        INSERT INTO performance_history (trigger_id, performance_score)
                        VALUES (?, ?)
                    """, (trigger_id, performance_score))
                break
    
    def evolve_population(self):
        """Perform one generation of evolution"""
        if len(self.triggers) < 4:
            return
        
        # Sort by fitness
        self.triggers.sort(key=lambda t: t.fitness_score, reverse=True)
        
        # Select survivors (natural selection)
        survivors_count = max(4, int(len(self.triggers) * self.selection_pressure))
        survivors = self.triggers[:survivors_count]
        
        # Preserve elite
        elite = self.triggers[:self.elite_size]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(survivors) >= 2:
                # Crossover
                parent1, parent2 = random.sample(survivors, 2)
                child1, child2 = parent1.crossover(parent2)
                new_population.extend([child1, child2])
            else:
                # Mutation
                parent = random.choice(survivors)
                mutant = parent.mutate(self.mutation_rate)
                new_population.append(mutant)
        
        # Trim to population size
        new_population = new_population[:self.population_size]
        
        # Update population
        self.triggers = new_population
        self.generation += 1
        
        # Save new triggers
        for trigger in new_population:
            if trigger not in elite:  # Don't re-save elite
                self._save_trigger(trigger)
        
        logging.info(f"Evolution complete. Generation {self.generation}, "
                    f"Population size: {len(self.triggers)}")
    
    def _cull_population(self):
        """Remove underperforming triggers"""
        self.triggers.sort(key=lambda t: t.fitness_score, reverse=True)
        self.triggers = self.triggers[:self.population_size]
    
    def get_best_triggers(self, n: int = 10) -> List[GeneticTrigger]:
        """Get top N performing triggers"""
        sorted_triggers = sorted(self.triggers, key=lambda t: t.fitness_score, reverse=True)
        return sorted_triggers[:n]
    
    def horizontal_gene_transfer(self, source_trigger: GeneticTrigger, 
                               target_trigger: GeneticTrigger, 
                               transfer_rate: float = 0.1):
        """Transfer genetic material between triggers"""
        if random.random() > transfer_rate:
            return
        
        # Transfer successful codon patterns
        for codon, activation in source_trigger.codon_map.items():
            if activation > 0.8:  # High-performing codon
                if codon in target_trigger.codon_map:
                    # Blend activation strengths
                    target_trigger.codon_map[codon] = (
                        target_trigger.codon_map[codon] * 0.7 + activation * 0.3
                    )
        
        # Transfer epigenetic markers
        for region, value in source_trigger.epigenetic_markers.markers.histone_modifications.items():
            if value > 0.5:  # Beneficial modification
                target_trigger.epigenetic_markers.markers.histone_modifications[region] = (
                    target_trigger.epigenetic_markers.markers.histone_modifications.get(region, 0) * 0.8 + 
                    value * 0.2
                )
        
        logging.info(f"Horizontal gene transfer: {source_trigger.trigger_id} -> {target_trigger.trigger_id}")
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics"""
        if not self.triggers:
            return {}
        
        fitness_scores = [t.fitness_score for t in self.triggers]
        generations = [t.generation for t in self.triggers]
        
        return {
            'population_size': len(self.triggers),
            'generation': self.generation,
            'avg_fitness': sum(fitness_scores) / len(fitness_scores),
            'max_fitness': max(fitness_scores),
            'min_fitness': min(fitness_scores),
            'avg_generation': sum(generations) / len(generations),
            'max_generation': max(generations),
            'genetic_diversity': self._calculate_genetic_diversity()
        }
    
    def _calculate_genetic_diversity(self) -> float:
        """Calculate genetic diversity of population"""
        if len(self.triggers) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.triggers)):
            for j in range(i + 1, len(self.triggers)):
                # Calculate genetic distance between triggers
                seq1 = self.triggers[i].genetic_sequence.sequence
                seq2 = self.triggers[j].genetic_sequence.sequence
                
                # Hamming distance for sequences of different lengths
                min_len = min(len(seq1), len(seq2))
                max_len = max(len(seq1), len(seq2))
                
                if max_len == 0:
                    distance = 0.0
                else:
                    hamming_dist = sum(c1 != c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
                    length_diff = abs(len(seq1) - len(seq2))
                    distance = (hamming_dist + length_diff) / max_len
                
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0


class GeneticTriggerSystem:
    """
    Main system that manages genetic triggers and their evolutionary adaptation
    """
    
    def __init__(self, database_path: str = "data/genetic_triggers.db"):
        self.population = GeneticTriggerPopulation(database_path=database_path)
        self.current_environment = EnvironmentalState()
        self.active_triggers: List[GeneticTrigger] = []
        self.evolution_interval = 100  # Evolve every N evaluations
        self.evaluation_count = 0
        self.logger = logging.getLogger(__name__)
        
    async def update_environment(self, **kwargs):
        """Update current environmental state"""
        for key, value in kwargs.items():
            if hasattr(self.current_environment, key):
                setattr(self.current_environment, key, value)
        
        self.current_environment.timestamp = time.time()
        
        # Evaluate triggers against new environment
        await self.evaluate_current_environment()
    
    async def evaluate_current_environment(self) -> List[GeneticTrigger]:
        """Evaluate all triggers against current environment"""
        self.active_triggers = self.population.evaluate_triggers(self.current_environment)
        self.evaluation_count += 1
        
        # Periodic evolution
        if self.evaluation_count % self.evolution_interval == 0:
            await self.evolve_population()
        
        # Log activation
        if self.active_triggers:
            self.logger.info(f"Activated {len(self.active_triggers)} genetic triggers")
            for trigger in self.active_triggers:
                self.logger.debug(f"Trigger {trigger.trigger_id} activated with score {trigger.expression_level:.3f}")
        
        return self.active_triggers
    
    async def evolve_population(self):
        """Perform evolutionary step"""
        self.logger.info("Starting genetic trigger evolution...")
        
        # Perform horizontal gene transfer between high-performing triggers
        best_triggers = self.population.get_best_triggers(5)
        if len(best_triggers) >= 2:
            for _ in range(3):  # Multiple transfer events
                source, target = random.sample(best_triggers, 2)
                self.population.horizontal_gene_transfer(source, target)
        
        # Evolve population
        self.population.evolve_population()
        
        stats = self.population.get_population_stats()
        self.logger.info(f"Evolution complete: {stats}")
    
    def report_performance(self, trigger_id: str, performance_score: float):
        """Report performance outcome for a trigger"""
        self.population.update_fitness(trigger_id, performance_score)
        
        # Immediate evolution if performance is exceptional (good or bad)
        if performance_score > 0.9 or performance_score < 0.1:
            asyncio.create_task(self.evolve_population())
    
    def create_trigger_from_environment(self, environment: EnvironmentalState) -> GeneticTrigger:
        """Create new trigger based on current environment"""
        trigger = GeneticTrigger(environment)
        self.population.add_trigger(trigger)
        return trigger
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        pop_stats = self.population.get_population_stats()
        
        return {
            'population_stats': pop_stats,
            'current_environment': self.current_environment.to_dict(),
            'active_triggers': len(self.active_triggers),
            'evaluation_count': self.evaluation_count,
            'evolution_interval': self.evolution_interval
        }
    
    def get_active_trigger_behaviors(self) -> Dict[str, List[str]]:
        """Get behaviors associated with currently active triggers"""
        behaviors = {}
        
        for trigger in self.active_triggers:
            trigger_behaviors = []
            
            for codon in trigger.genetic_sequence.codons:
                try:
                    codon_enum = GeneticCodon(codon)
                    trigger_behaviors.append(codon_enum.value)
                except ValueError:
                    # Unknown codon, map to generic behavior
                    if codon.startswith('A'):
                        trigger_behaviors.append("attention_enhancement")
                    elif codon.startswith('T'):
                        trigger_behaviors.append("task_focus")
                    elif codon.startswith('G'):
                        trigger_behaviors.append("growth_oriented")
                    elif codon.startswith('C'):
                        trigger_behaviors.append("coordination_boost")
            
            behaviors[trigger.trigger_id] = trigger_behaviors
        
        return behaviors


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test_genetic_trigger_system():
        """Test the genetic trigger system"""
        system = GeneticTriggerSystem()
        
        # Simulate different environmental conditions
        environments = [
            {'task_complexity': 0.8, 'user_satisfaction': 0.6, 'error_rate': 0.2},
            {'task_complexity': 0.3, 'creativity_demand': 0.9, 'collaboration_level': 0.7},
            {'system_load': 0.9, 'time_pressure': 0.8, 'resource_availability': 0.3},
            {'learning_rate': 0.7, 'user_satisfaction': 0.9, 'error_rate': 0.1}
        ]
        
        # Test environment updates and trigger activation
        for i, env_update in enumerate(environments):
            print(f"\n--- Test Environment {i+1} ---")
            await system.update_environment(**env_update)
            
            active_triggers = system.active_triggers
            print(f"Active triggers: {len(active_triggers)}")
            
            # Simulate performance feedback
            for trigger in active_triggers:
                performance = random.uniform(0.3, 0.9)
                system.report_performance(trigger.trigger_id, performance)
                print(f"Trigger {trigger.trigger_id}: performance {performance:.3f}")
            
            # Show active behaviors
            behaviors = system.get_active_trigger_behaviors()
            for trigger_id, behavior_list in behaviors.items():
                print(f"Trigger {trigger_id} behaviors: {behavior_list[:3]}")  # Show first 3
        
        # Show final system stats
        print(f"\n--- Final System Stats ---")
        stats = system.get_system_stats()
        print(f"Population size: {stats['population_stats']['population_size']}")
        print(f"Average fitness: {stats['population_stats']['avg_fitness']:.3f}")
        print(f"Genetic diversity: {stats['population_stats']['genetic_diversity']:.3f}")
        print(f"Total evaluations: {stats['evaluation_count']}")
    
    # Run test
    asyncio.run(test_genetic_trigger_system())