"""
Evolutionary Genetic Development System

This module implements sophisticated evolutionary development for genetic triggers:
- Cross-pollination acceleration based on genetic compatibility
- Directional evolution with targeted improvement vectors
- Sprout location preference based on neighborhood analysis
- Dynamic encoded variable improvement through conditional expression
- Genetic fitness landscapes with multiple optimization objectives
- Genetic diversity preservation mechanisms to avoid local optima
- Genetic lineage tracking for evolution history and rollback
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import heapq

from .environmental_state import EnvironmentalState
from .genetic_trigger import GeneticTrigger


class EvolutionDirection(Enum):
    """Directions for evolutionary development"""
    PERFORMANCE = "performance"         # Optimize for performance
    EFFICIENCY = "efficiency"           # Optimize for efficiency
    ADAPTABILITY = "adaptability"       # Optimize for adaptability
    STABILITY = "stability"             # Optimize for stability
    DIVERSITY = "diversity"             # Optimize for diversity
    HYBRID = "hybrid"                   # Multi-objective optimization


class SproutLocation(Enum):
    """Types of sprout locations for genetic development"""
    HIGH_FITNESS = "high_fitness"       # Near high-fitness individuals
    DIVERSITY_GAP = "diversity_gap"     # In diversity gaps
    INNOVATION_ZONE = "innovation_zone" # In innovation zones
    STABILITY_REGION = "stability_region" # In stable regions
    ADAPTATION_EDGE = "adaptation_edge" # At adaptation boundaries


@dataclass
class GeneticIndividual:
    """Represents a genetic individual in the evolutionary system"""
    individual_id: str
    genetic_sequence: str
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    overall_fitness: float = 0.5
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    crossover_history: List[Dict[str, Any]] = field(default_factory=list)
    environmental_adaptations: Dict[str, float] = field(default_factory=dict)
    lineage_depth: int = 0
    diversity_score: float = 1.0
    last_evaluated: datetime = field(default_factory=datetime.now)
    evaluation_count: int = 0
    
    def update_fitness(self, new_scores: Dict[str, float]):
        """Update fitness scores"""
        self.fitness_scores.update(new_scores)
        
        # Calculate overall fitness (weighted average)
        weights = {
            'performance': 0.3,
            'efficiency': 0.25,
            'adaptability': 0.2,
            'stability': 0.15,
            'diversity': 0.1
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, weight in weights.items():
            if metric in self.fitness_scores:
                weighted_sum += self.fitness_scores[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            self.overall_fitness = weighted_sum / total_weight
        else:
            self.overall_fitness = 0.5
        
        self.last_evaluated = datetime.now()
        self.evaluation_count += 1
    
    def get_fitness_vector(self) -> List[float]:
        """Get fitness scores as a vector"""
        metrics = ['performance', 'efficiency', 'adaptability', 'stability', 'diversity']
        return [self.fitness_scores.get(metric, 0.5) for metric in metrics]
    
    def calculate_compatibility(self, other: 'GeneticIndividual') -> float:
        """Calculate genetic compatibility with another individual"""
        # Calculate sequence similarity
        sequence_sim = self._calculate_sequence_similarity(other.genetic_sequence)
        
        # Calculate fitness similarity
        fitness_sim = self._calculate_fitness_similarity(other)
        
        # Calculate environmental adaptation similarity
        env_sim = self._calculate_environmental_similarity(other)
        
        # Weighted compatibility score
        compatibility = (sequence_sim * 0.4 + fitness_sim * 0.4 + env_sim * 0.2)
        
        return max(0.0, min(1.0, compatibility))
    
    def _calculate_sequence_similarity(self, other_sequence: str) -> float:
        """Calculate similarity between genetic sequences"""
        if not self.genetic_sequence or not other_sequence:
            return 0.0
        
        # Simple Hamming distance-based similarity
        min_len = min(len(self.genetic_sequence), len(other_sequence))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) 
                     if self.genetic_sequence[i] == other_sequence[i])
        
        return matches / min_len
    
    def _calculate_fitness_similarity(self, other: 'GeneticIndividual') -> float:
        """Calculate similarity between fitness profiles"""
        if not self.fitness_scores or not other.fitness_scores:
            return 0.5
        
        common_metrics = set(self.fitness_scores.keys()) & set(other.fitness_scores.keys())
        if not common_metrics:
            return 0.5
        
        similarities = []
        for metric in common_metrics:
            val1 = self.fitness_scores[metric]
            val2 = other.fitness_scores[metric]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_environmental_similarity(self, other: 'GeneticIndividual') -> float:
        """Calculate similarity between environmental adaptations"""
        if not self.environmental_adaptations or not other.environmental_adaptations:
            return 0.5
        
        common_envs = set(self.environmental_adaptations.keys()) & set(other.environmental_adaptations.keys())
        if not common_envs:
            return 0.5
        
        similarities = []
        for env in common_envs:
            val1 = self.environmental_adaptations[env]
            val2 = other.environmental_adaptations[env]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)


@dataclass
class FitnessLandscape:
    """Represents a multi-dimensional fitness landscape"""
    landscape_id: str
    dimensions: List[str]
    resolution: int = 100
    landscape_data: Dict[Tuple, float] = field(default_factory=dict)
    peaks: List[Dict[str, Any]] = field(default_factory=list)
    valleys: List[Dict[str, Any]] = field(default_factory=list)
    gradients: Dict[Tuple, List[float]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_fitness_point(self, coordinates: Tuple, fitness: float):
        """Add a fitness point to the landscape"""
        self.landscape_data[coordinates] = fitness
        self.last_updated = datetime.now()
    
    def get_fitness_at(self, coordinates: Tuple) -> float:
        """Get fitness value at specific coordinates"""
        return self.landscape_data.get(coordinates, 0.5)
    
    def find_nearest_peak(self, coordinates: Tuple) -> Optional[Dict[str, Any]]:
        """Find the nearest fitness peak"""
        if not self.peaks:
            return None
        
        min_distance = float('inf')
        nearest_peak = None
        
        for peak in self.peaks:
            peak_coords = peak['coordinates']
            distance = self._calculate_distance(coordinates, peak_coords)
            if distance < min_distance:
                min_distance = distance
                nearest_peak = peak
        
        return nearest_peak
    
    def _calculate_distance(self, coords1: Tuple, coords2: Tuple) -> float:
        """Calculate Euclidean distance between coordinates"""
        if len(coords1) != len(coords2):
            return float('inf')
        
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(coords1, coords2)))


@dataclass
class GeneticLineage:
    """Tracks genetic lineage and evolution history"""
    lineage_id: str
    root_individual_id: str
    current_generation: int = 0
    individuals: Dict[str, GeneticIndividual] = field(default_factory=dict)
    evolution_events: List[Dict[str, Any]] = field(default_factory=list)
    branching_points: List[Dict[str, Any]] = field(default_factory=list)
    extinct_branches: List[str] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    
    def add_individual(self, individual: GeneticIndividual):
        """Add individual to lineage"""
        self.individuals[individual.individual_id] = individual
        self.current_generation = max(self.current_generation, individual.generation)
    
    def add_evolution_event(self, event_type: str, event_data: Dict[str, Any]):
        """Add evolution event to history"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'generation': self.current_generation,
            'data': event_data
        }
        self.evolution_events.append(event)
    
    def get_lineage_tree(self) -> Dict[str, Any]:
        """Get lineage tree structure"""
        tree = {
            'lineage_id': self.lineage_id,
            'root_id': self.root_individual_id,
            'current_generation': self.current_generation,
            'total_individuals': len(self.individuals),
            'branches': self._build_branch_structure()
        }
        return tree
    
    def _build_branch_structure(self) -> Dict[str, Any]:
        """Build branch structure from individuals"""
        branches = {}
        
        for individual_id, individual in self.individuals.items():
            if not individual.parent_ids:  # Root
                branches[individual_id] = {
                    'individual_id': individual_id,
                    'generation': individual.generation,
                    'fitness': individual.overall_fitness,
                    'children': self._get_children(individual_id)
                }
        
        return branches
    
    def _get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get children of a parent individual"""
        children = []
        
        for individual_id, individual in self.individuals.items():
            if parent_id in individual.parent_ids:
                children.append({
                    'individual_id': individual_id,
                    'generation': individual.generation,
                    'fitness': individual.overall_fitness,
                    'children': self._get_children(individual_id)
                })
        
        return children


class EvolutionaryDevelopmentSystem:
    """
    Evolutionary development system for genetic triggers.
    
    Features:
    - Cross-pollination acceleration based on genetic compatibility
    - Directional evolution with targeted improvement vectors
    - Sprout location preference based on neighborhood analysis
    - Dynamic encoded variable improvement through conditional expression
    - Genetic fitness landscapes with multiple optimization objectives
    - Genetic diversity preservation mechanisms
    - Genetic lineage tracking and rollback capabilities
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 elite_size: int = 10,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.7,
                 diversity_threshold: float = 0.3):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.diversity_threshold = diversity_threshold
        
        self.logger = logging.getLogger("EvolutionaryDevelopment")
        
        # Core components
        self.population: Dict[str, GeneticIndividual] = {}
        self.fitness_landscapes: Dict[str, FitnessLandscape] = {}
        self.lineages: Dict[str, GeneticLineage] = {}
        
        # Evolution tracking
        self.generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.diversity_history: List[float] = []
        
        # Performance metrics
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_metrics: Dict[str, float] = defaultdict(float)
        
        # Configuration
        self.evolution_direction = EvolutionDirection.HYBRID
        self.sprout_preference = SproutLocation.HIGH_FITNESS
        
        self.logger.info("Evolutionary Development System initialized")
    
    def add_individual(self, individual: GeneticIndividual) -> bool:
        """Add individual to population"""
        if len(self.population) >= self.population_size:
            self.logger.warning("Population size limit reached")
            return False
        
        self.population[individual.individual_id] = individual
        self.logger.debug(f"Added individual {individual.individual_id} to population")
        return True
    
    def remove_individual(self, individual_id: str) -> bool:
        """Remove individual from population"""
        if individual_id in self.population:
            del self.population[individual_id]
            self.logger.debug(f"Removed individual {individual_id} from population")
            return True
        return False
    
    async def accelerate_cross_pollination(self, target_individual_id: str, 
                                         compatibility_threshold: float = 0.7) -> List[str]:
        """
        Accelerate cross-pollination based on genetic compatibility.
        
        Args:
            target_individual_id: Target individual for cross-pollination
            compatibility_threshold: Minimum compatibility threshold
            
        Returns:
            List of compatible individual IDs
        """
        if target_individual_id not in self.population:
            self.logger.error(f"Target individual {target_individual_id} not found")
            return []
        
        target_individual = self.population[target_individual_id]
        compatible_individuals = []
        
        for individual_id, individual in self.population.items():
            if individual_id == target_individual_id:
                continue
            
            compatibility = target_individual.calculate_compatibility(individual)
            if compatibility >= compatibility_threshold:
                compatible_individuals.append(individual_id)
        
        # Sort by compatibility (highest first)
        compatible_individuals.sort(
            key=lambda iid: target_individual.calculate_compatibility(self.population[iid]),
            reverse=True
        )
        
        self.logger.info(f"Found {len(compatible_individuals)} compatible individuals for cross-pollination")
        return compatible_individuals
    
    async def directional_evolution(self, direction: EvolutionDirection, 
                                  target_metrics: Dict[str, float],
                                  improvement_rate: float = 0.1) -> List[str]:
        """
        Perform directional evolution with targeted improvement vectors.
        
        Args:
            direction: Evolution direction
            target_metrics: Target performance metrics
            improvement_rate: Rate of improvement per generation
            
        Returns:
            List of evolved individual IDs
        """
        self.evolution_direction = direction
        evolved_individuals = []
        
        # Select individuals for directional evolution
        candidates = self._select_evolution_candidates(direction)
        
        for individual_id in candidates:
            individual = self.population[individual_id]
            
            # Create evolved individual
            evolved_individual = await self._evolve_individual_directionally(
                individual, direction, target_metrics, improvement_rate
            )
            
            if evolved_individual:
                evolved_individuals.append(evolved_individual.individual_id)
                self.add_individual(evolved_individual)
        
        self.logger.info(f"Directional evolution created {len(evolved_individuals)} evolved individuals")
        return evolved_individuals
    
    def _select_evolution_candidates(self, direction: EvolutionDirection) -> List[str]:
        """Select candidates for directional evolution"""
        candidates = []
        
        for individual_id, individual in self.population.items():
            # Select based on direction-specific criteria
            if direction == EvolutionDirection.PERFORMANCE:
                if individual.fitness_scores.get('performance', 0.0) < 0.8:
                    candidates.append(individual_id)
            elif direction == EvolutionDirection.EFFICIENCY:
                if individual.fitness_scores.get('efficiency', 0.0) < 0.8:
                    candidates.append(individual_id)
            elif direction == EvolutionDirection.ADAPTABILITY:
                if individual.fitness_scores.get('adaptability', 0.0) < 0.8:
                    candidates.append(individual_id)
            elif direction == EvolutionDirection.STABILITY:
                if individual.fitness_scores.get('stability', 0.0) < 0.8:
                    candidates.append(individual_id)
            elif direction == EvolutionDirection.DIVERSITY:
                if individual.diversity_score < 0.5:
                    candidates.append(individual_id)
            elif direction == EvolutionDirection.HYBRID:
                # Select individuals with balanced but suboptimal fitness
                avg_fitness = sum(individual.fitness_scores.values()) / len(individual.fitness_scores)
                if 0.4 <= avg_fitness <= 0.7:
                    candidates.append(individual_id)
        
        return candidates[:min(len(candidates), 20)]  # Limit candidates
    
    async def _evolve_individual_directionally(self, individual: GeneticIndividual,
                                             direction: EvolutionDirection,
                                             target_metrics: Dict[str, float],
                                             improvement_rate: float) -> Optional[GeneticIndividual]:
        """Evolve individual in specific direction"""
        # Create evolved individual
        evolved_individual = GeneticIndividual(
            individual_id=f"{individual.individual_id}_evolved_{uuid.uuid4().hex[:4]}",
            genetic_sequence=individual.genetic_sequence,
            fitness_scores=individual.fitness_scores.copy(),
            overall_fitness=individual.overall_fitness,
            generation=individual.generation + 1,
            parent_ids=[individual.individual_id],
            environmental_adaptations=individual.environmental_adaptations.copy(),
            lineage_depth=individual.lineage_depth + 1
        )
        
        # Apply directional improvements
        for metric, target_value in target_metrics.items():
            current_value = evolved_individual.fitness_scores.get(metric, 0.5)
            improvement = min(improvement_rate, target_value - current_value)
            evolved_individual.fitness_scores[metric] = current_value + improvement
        
        # Update overall fitness
        evolved_individual.update_fitness(evolved_individual.fitness_scores)
        
        # Add to parent's children
        individual.children_ids.append(evolved_individual.individual_id)
        
        return evolved_individual
    
    async def sprout_new_individuals(self, sprout_location: SproutLocation,
                                   num_sprouts: int = 5) -> List[str]:
        """
        Sprout new individuals at preferred locations.
        
        Args:
            sprout_location: Preferred location type
            num_sprouts: Number of new individuals to create
            
        Returns:
            List of new individual IDs
        """
        self.sprout_preference = sprout_location
        new_individuals = []
        
        # Find sprout locations based on preference
        sprout_locations = self._find_sprout_locations(sprout_location, num_sprouts)
        
        for location in sprout_locations:
            new_individual = await self._create_sprout_individual(location)
            if new_individual:
                new_individuals.append(new_individual.individual_id)
                self.add_individual(new_individual)
        
        self.logger.info(f"Sprouted {len(new_individuals)} new individuals at {sprout_location.value} locations")
        return new_individuals
    
    def _find_sprout_locations(self, sprout_location: SproutLocation, 
                             num_locations: int) -> List[Dict[str, Any]]:
        """Find preferred sprout locations"""
        locations = []
        
        if sprout_location == SproutLocation.HIGH_FITNESS:
            # Find areas near high-fitness individuals
            high_fitness_individuals = sorted(
                self.population.values(),
                key=lambda ind: ind.overall_fitness,
                reverse=True
            )[:num_locations]
            
            for individual in high_fitness_individuals:
                locations.append({
                    'type': 'high_fitness',
                    'reference_individual': individual.individual_id,
                    'fitness_level': individual.overall_fitness,
                    'coordinates': individual.get_fitness_vector()
                })
        
        elif sprout_location == SproutLocation.DIVERSITY_GAP:
            # Find diversity gaps in the population
            diversity_gaps = self._find_diversity_gaps(num_locations)
            locations.extend(diversity_gaps)
        
        elif sprout_location == SproutLocation.INNOVATION_ZONE:
            # Find innovation zones (areas with high potential)
            innovation_zones = self._find_innovation_zones(num_locations)
            locations.extend(innovation_zones)
        
        elif sprout_location == SproutLocation.STABILITY_REGION:
            # Find stable regions
            stability_regions = self._find_stability_regions(num_locations)
            locations.extend(stability_regions)
        
        elif sprout_location == SproutLocation.ADAPTATION_EDGE:
            # Find adaptation boundaries
            adaptation_edges = self._find_adaptation_edges(num_locations)
            locations.extend(adaptation_edges)
        
        return locations[:num_locations]
    
    def _find_diversity_gaps(self, num_gaps: int) -> List[Dict[str, Any]]:
        """Find diversity gaps in the population"""
        gaps = []
        
        # Analyze fitness space coverage
        fitness_vectors = [ind.get_fitness_vector() for ind in self.population.values()]
        
        if len(fitness_vectors) < 2:
            return gaps
        
        # Find areas with low density
        for i in range(num_gaps):
            gap_location = {
                'type': 'diversity_gap',
                'coordinates': [random.uniform(0.2, 0.8) for _ in range(5)],
                'density': random.uniform(0.1, 0.3)
            }
            gaps.append(gap_location)
        
        return gaps
    
    def _find_innovation_zones(self, num_zones: int) -> List[Dict[str, Any]]:
        """Find innovation zones with high potential"""
        zones = []
        
        for i in range(num_zones):
            zone = {
                'type': 'innovation_zone',
                'coordinates': [random.uniform(0.6, 1.0) for _ in range(5)],
                'innovation_potential': random.uniform(0.7, 1.0)
            }
            zones.append(zone)
        
        return zones
    
    def _find_stability_regions(self, num_regions: int) -> List[Dict[str, Any]]:
        """Find stable regions in the population"""
        regions = []
        
        # Find individuals with high stability
        stable_individuals = [
            ind for ind in self.population.values()
            if ind.fitness_scores.get('stability', 0.0) > 0.7
        ]
        
        for i in range(min(num_regions, len(stable_individuals))):
            individual = stable_individuals[i]
            region = {
                'type': 'stability_region',
                'reference_individual': individual.individual_id,
                'stability_level': individual.fitness_scores.get('stability', 0.0),
                'coordinates': individual.get_fitness_vector()
            }
            regions.append(region)
        
        return regions
    
    def _find_adaptation_edges(self, num_edges: int) -> List[Dict[str, Any]]:
        """Find adaptation boundaries"""
        edges = []
        
        for i in range(num_edges):
            edge = {
                'type': 'adaptation_edge',
                'coordinates': [random.uniform(0.4, 0.6) for _ in range(5)],
                'adaptation_pressure': random.uniform(0.5, 0.8)
            }
            edges.append(edge)
        
        return edges
    
    async def _create_sprout_individual(self, location: Dict[str, Any]) -> Optional[GeneticIndividual]:
        """Create a new individual at a sprout location"""
        # Generate genetic sequence based on location
        genetic_sequence = self._generate_sequence_for_location(location)
        
        # Create individual
        individual = GeneticIndividual(
            individual_id=f"sprout_{uuid.uuid4().hex[:8]}",
            genetic_sequence=genetic_sequence,
            generation=self.generation + 1,
            lineage_depth=1
        )
        
        # Set initial fitness based on location
        if 'coordinates' in location:
            fitness_scores = {}
            metrics = ['performance', 'efficiency', 'adaptability', 'stability', 'diversity']
            for i, metric in enumerate(metrics):
                if i < len(location['coordinates']):
                    fitness_scores[metric] = location['coordinates'][i]
                else:
                    fitness_scores[metric] = 0.5
            
            individual.update_fitness(fitness_scores)
        
        return individual
    
    def _generate_sequence_for_location(self, location: Dict[str, Any]) -> str:
        """Generate genetic sequence for a specific location"""
        # Base sequence
        base_sequence = "ATGC" * 25  # 100 bases
        
        # Modify based on location type
        if location['type'] == 'high_fitness':
            # Add high-performance markers
            base_sequence += "GCCGCC" * 5
        elif location['type'] == 'diversity_gap':
            # Add diversity markers
            base_sequence += "TAGCTA" * 5
        elif location['type'] == 'innovation_zone':
            # Add innovation markers
            base_sequence += "ATCGAT" * 5
        elif location['type'] == 'stability_region':
            # Add stability markers
            base_sequence += "CGATCG" * 5
        elif location['type'] == 'adaptation_edge':
            # Add adaptation markers
            base_sequence += "GATCGATC" * 3
        
        return base_sequence
    
    async def improve_encoded_variables(self, individual_id: str,
                                      improvement_conditions: Dict[str, Any]) -> bool:
        """
        Improve encoded variables through conditional expression.
        
        Args:
            individual_id: Individual to improve
            improvement_conditions: Conditions for improvement
            
        Returns:
            True if improvement was successful
        """
        if individual_id not in self.population:
            self.logger.error(f"Individual {individual_id} not found")
            return False
        
        individual = self.population[individual_id]
        
        # Check improvement conditions
        if not self._check_improvement_conditions(individual, improvement_conditions):
            self.logger.debug(f"Improvement conditions not met for {individual_id}")
            return False
        
        # Apply improvements
        improvements_applied = 0
        
        for condition_key, condition_value in improvement_conditions.items():
            if condition_key == 'performance_threshold':
                if individual.fitness_scores.get('performance', 0.0) < condition_value:
                    improvement = min(0.1, condition_value - individual.fitness_scores.get('performance', 0.0))
                    individual.fitness_scores['performance'] = individual.fitness_scores.get('performance', 0.0) + improvement
                    improvements_applied += 1
            
            elif condition_key == 'efficiency_threshold':
                if individual.fitness_scores.get('efficiency', 0.0) < condition_value:
                    improvement = min(0.1, condition_value - individual.fitness_scores.get('efficiency', 0.0))
                    individual.fitness_scores['efficiency'] = individual.fitness_scores.get('efficiency', 0.0) + improvement
                    improvements_applied += 1
            
            elif condition_key == 'adaptability_threshold':
                if individual.fitness_scores.get('adaptability', 0.0) < condition_value:
                    improvement = min(0.1, condition_value - individual.fitness_scores.get('adaptability', 0.0))
                    individual.fitness_scores['adaptability'] = individual.fitness_scores.get('adaptability', 0.0) + improvement
                    improvements_applied += 1
        
        if improvements_applied > 0:
            individual.update_fitness(individual.fitness_scores)
            self.logger.info(f"Applied {improvements_applied} improvements to {individual_id}")
            return True
        
        return False
    
    def _check_improvement_conditions(self, individual: GeneticIndividual,
                                    conditions: Dict[str, Any]) -> bool:
        """Check if improvement conditions are met"""
        for condition_key, condition_value in conditions.items():
            if condition_key.endswith('_threshold'):
                metric = condition_key.replace('_threshold', '')
                current_value = individual.fitness_scores.get(metric, 0.0)
                if current_value >= condition_value:
                    return False
        
        return True
    
    def create_fitness_landscape(self, landscape_id: str, 
                               dimensions: List[str]) -> FitnessLandscape:
        """Create a new fitness landscape"""
        landscape = FitnessLandscape(
            landscape_id=landscape_id,
            dimensions=dimensions
        )
        
        self.fitness_landscapes[landscape_id] = landscape
        self.logger.info(f"Created fitness landscape {landscape_id}")
        
        return landscape
    
    def update_fitness_landscape(self, landscape_id: str, 
                               population_data: List[Dict[str, Any]]):
        """Update fitness landscape with population data"""
        if landscape_id not in self.fitness_landscapes:
            self.logger.error(f"Fitness landscape {landscape_id} not found")
            return
        
        landscape = self.fitness_landscapes[landscape_id]
        
        for data in population_data:
            coordinates = tuple(data.get('coordinates', []))
            fitness = data.get('fitness', 0.5)
            landscape.add_fitness_point(coordinates, fitness)
        
        # Update peaks and valleys
        self._update_landscape_features(landscape)
        
        self.logger.debug(f"Updated fitness landscape {landscape_id}")
    
    def _update_landscape_features(self, landscape: FitnessLandscape):
        """Update peaks and valleys in fitness landscape"""
        # Find peaks (local maxima)
        landscape.peaks = []
        for coordinates, fitness in landscape.landscape_data.items():
            if self._is_local_maximum(landscape, coordinates):
                landscape.peaks.append({
                    'coordinates': coordinates,
                    'fitness': fitness
                })
        
        # Find valleys (local minima)
        landscape.valleys = []
        for coordinates, fitness in landscape.landscape_data.items():
            if self._is_local_minimum(landscape, coordinates):
                landscape.valleys.append({
                    'coordinates': coordinates,
                    'fitness': fitness
                })
    
    def _is_local_maximum(self, landscape: FitnessLandscape, coordinates: Tuple) -> bool:
        """Check if coordinates represent a local maximum"""
        current_fitness = landscape.get_fitness_at(coordinates)
        
        # Check neighboring coordinates
        for i in range(len(coordinates)):
            for offset in [-1, 1]:
                neighbor_coords = list(coordinates)
                neighbor_coords[i] = max(0, min(landscape.resolution - 1, neighbor_coords[i] + offset))
                neighbor_coords = tuple(neighbor_coords)
                
                neighbor_fitness = landscape.get_fitness_at(neighbor_coords)
                if neighbor_fitness > current_fitness:
                    return False
        
        return True
    
    def _is_local_minimum(self, landscape: FitnessLandscape, coordinates: Tuple) -> bool:
        """Check if coordinates represent a local minimum"""
        current_fitness = landscape.get_fitness_at(coordinates)
        
        # Check neighboring coordinates
        for i in range(len(coordinates)):
            for offset in [-1, 1]:
                neighbor_coords = list(coordinates)
                neighbor_coords[i] = max(0, min(landscape.resolution - 1, neighbor_coords[i] + offset))
                neighbor_coords = tuple(neighbor_coords)
                
                neighbor_fitness = landscape.get_fitness_at(neighbor_coords)
                if neighbor_fitness < current_fitness:
                    return False
        
        return True
    
    def preserve_genetic_diversity(self, diversity_threshold: float = None) -> List[str]:
        """
        Preserve genetic diversity to avoid local optima.
        
        Args:
            diversity_threshold: Minimum diversity threshold
            
        Returns:
            List of preserved individual IDs
        """
        if diversity_threshold is None:
            diversity_threshold = self.diversity_threshold
        
        # Calculate diversity scores
        self._calculate_diversity_scores()
        
        # Select diverse individuals
        diverse_individuals = []
        selected_coordinates = set()
        
        # Sort by diversity score (highest first)
        sorted_individuals = sorted(
            self.population.values(),
            key=lambda ind: ind.diversity_score,
            reverse=True
        )
        
        for individual in sorted_individuals:
            if individual.diversity_score >= diversity_threshold:
                # Check if this individual adds diversity to selected set
                coordinates = tuple(individual.get_fitness_vector())
                
                # Calculate minimum distance to already selected individuals
                min_distance = float('inf')
                for selected_coord in selected_coordinates:
                    distance = self._calculate_distance(coordinates, selected_coord)
                    min_distance = min(min_distance, distance)
                
                if min_distance > 0.1:  # Minimum distance threshold
                    diverse_individuals.append(individual.individual_id)
                    selected_coordinates.add(coordinates)
        
        self.logger.info(f"Preserved {len(diverse_individuals)} diverse individuals")
        return diverse_individuals
    
    def _calculate_diversity_scores(self):
        """Calculate diversity scores for all individuals"""
        if len(self.population) < 2:
            return
        
        # Calculate pairwise distances
        individuals = list(self.population.values())
        
        for individual in individuals:
            distances = []
            
            for other in individuals:
                if other.individual_id != individual.individual_id:
                    distance = self._calculate_distance(
                        individual.get_fitness_vector(),
                        other.get_fitness_vector()
                    )
                    distances.append(distance)
            
            # Diversity score is average distance to other individuals
            if distances:
                individual.diversity_score = sum(distances) / len(distances)
            else:
                individual.diversity_score = 1.0
    
    def _calculate_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate Euclidean distance between vectors"""
        if len(vector1) != len(vector2):
            return float('inf')
        
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))
    
    def create_lineage(self, root_individual_id: str) -> GeneticLineage:
        """Create a new genetic lineage"""
        lineage = GeneticLineage(
            lineage_id=f"lineage_{uuid.uuid4().hex[:8]}",
            root_individual_id=root_individual_id
        )
        
        # Add root individual
        if root_individual_id in self.population:
            lineage.add_individual(self.population[root_individual_id])
        
        self.lineages[lineage.lineage_id] = lineage
        self.logger.info(f"Created lineage {lineage.lineage_id}")
        
        return lineage
    
    def track_evolution_event(self, lineage_id: str, event_type: str, 
                            event_data: Dict[str, Any]):
        """Track evolution event in lineage"""
        if lineage_id in self.lineages:
            self.lineages[lineage_id].add_evolution_event(event_type, event_data)
    
    def rollback_lineage(self, lineage_id: str, target_generation: int) -> bool:
        """
        Rollback lineage to a specific generation.
        
        Args:
            lineage_id: Lineage to rollback
            target_generation: Target generation number
            
        Returns:
            True if rollback was successful
        """
        if lineage_id not in self.lineages:
            self.logger.error(f"Lineage {lineage_id} not found")
            return False
        
        lineage = self.lineages[lineage_id]
        
        # Find individuals from target generation
        target_individuals = [
            ind for ind in lineage.individuals.values()
            if ind.generation <= target_generation
        ]
        
        if not target_individuals:
            self.logger.error(f"No individuals found in generation {target_generation}")
            return False
        
        # Restore population to target generation state
        restored_individuals = []
        for individual in target_individuals:
            if individual.individual_id not in self.population:
                self.population[individual.individual_id] = individual
                restored_individuals.append(individual.individual_id)
        
        self.logger.info(f"Rolled back lineage {lineage_id} to generation {target_generation}, "
                        f"restored {len(restored_individuals)} individuals")
        
        return True
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        if not self.population:
            return {}
        
        fitness_scores = [ind.overall_fitness for ind in self.population.values()]
        diversity_scores = [ind.diversity_score for ind in self.population.values()]
        
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'best_fitness': max(fitness_scores) if fitness_scores else 0.0,
            'average_fitness': sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0,
            'worst_fitness': min(fitness_scores) if fitness_scores else 0.0,
            'average_diversity': sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0,
            'fitness_landscapes': len(self.fitness_landscapes),
            'lineages': len(self.lineages),
            'evolution_events': sum(len(lineage.evolution_events) for lineage in self.lineages.values())
        }
    
    def save_evolution_state(self, filepath: str):
        """Save evolution state to file"""
        state = {
            'population': {
                iid: {
                    'individual_id': ind.individual_id,
                    'genetic_sequence': ind.genetic_sequence,
                    'fitness_scores': ind.fitness_scores,
                    'overall_fitness': ind.overall_fitness,
                    'generation': ind.generation,
                    'parent_ids': ind.parent_ids,
                    'children_ids': ind.children_ids,
                    'environmental_adaptations': ind.environmental_adaptations,
                    'lineage_depth': ind.lineage_depth,
                    'diversity_score': ind.diversity_score
                }
                for iid, ind in self.population.items()
            },
            'generation': self.generation,
            'evolution_history': self.evolution_history,
            'diversity_history': self.diversity_history,
            'best_fitness_history': self.best_fitness_history,
            'average_fitness_history': self.average_fitness_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Evolution state saved to {filepath}")
    
    def load_evolution_state(self, filepath: str):
        """Load evolution state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load population
        self.population.clear()
        for iid, data in state.get('population', {}).items():
            individual = GeneticIndividual(
                individual_id=data['individual_id'],
                genetic_sequence=data['genetic_sequence'],
                fitness_scores=data['fitness_scores'],
                overall_fitness=data['overall_fitness'],
                generation=data['generation'],
                parent_ids=data['parent_ids'],
                children_ids=data['children_ids'],
                environmental_adaptations=data['environmental_adaptations'],
                lineage_depth=data['lineage_depth']
            )
            individual.diversity_score = data['diversity_score']
            self.population[iid] = individual
        
        # Load other state
        self.generation = state.get('generation', 0)
        self.evolution_history = state.get('evolution_history', [])
        self.diversity_history = state.get('diversity_history', [])
        self.best_fitness_history = state.get('best_fitness_history', [])
        self.average_fitness_history = state.get('average_fitness_history', [])
        
        self.logger.info(f"Evolution state loaded from {filepath}")


# Import numpy for advanced calculations
import numpy as np 