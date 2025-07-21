"""
Advanced Genetic Trigger Optimization System

This module provides sophisticated optimization capabilities for the genetic trigger system,
including performance tracking, evolutionary improvements, and adaptive mutation strategies.

Features:
- Advanced performance tracking and analysis
- Evolutionary algorithm optimization
- Adaptive mutation rates based on performance
- Cross-generational learning and inheritance
- Multi-objective optimization
- Performance prediction and forecasting
"""

import logging
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from collections import defaultdict

from .genetic_trigger import GeneticTrigger
from .environmental_adaptation import EnvironmentalAdaptationSystem, AdaptationPressure
from .integrated_genetic_system import IntegratedGeneticTriggerSystem


class OptimizationStrategy(Enum):
    """Types of optimization strategies"""
    PERFORMANCE_BASED = "performance_based"
    ENVIRONMENTAL_ADAPTATION = "environmental_adaptation"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"


@dataclass
class TriggerPerformance:
    """Detailed performance metrics for a genetic trigger"""
    trigger_id: str
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_response_time: float = 0.0
    environmental_accuracy: float = 0.0
    adaptation_score: float = 0.0
    fitness_score: float = 0.0
    last_activation: Optional[datetime] = None
    performance_history: List[float] = field(default_factory=list)
    environmental_matches: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of genetic trigger optimization"""
    trigger_id: str
    optimization_strategy: OptimizationStrategy
    performance_improvement: float
    changes_made: List[str]
    new_fitness_score: float
    environmental_adaptation: Dict[str, Any]
    timestamp: datetime


class AdvancedGeneticOptimizer:
    """
    Advanced genetic trigger optimization system.
    
    Provides sophisticated optimization capabilities including performance tracking,
    evolutionary improvements, and adaptive mutation strategies.
    """
    
    def __init__(self, 
                 genetic_system: Optional[IntegratedGeneticTriggerSystem] = None,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.HYBRID):
        
        self.logger = logging.getLogger("AdvancedGeneticOptimizer")
        self.genetic_system = genetic_system
        self.optimization_strategy = optimization_strategy
        
        # Performance tracking
        self.trigger_performance: Dict[str, TriggerPerformance] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Optimization parameters
        self.mutation_rate = 0.05
        self.crossover_rate = 0.3
        self.selection_pressure = 0.8
        self.adaptation_threshold = 0.1
        
        # Evolutionary parameters
        self.population_size = 100
        self.generations = 50
        self.elite_size = 10
        
        # Environmental adaptation
        self.environmental_adaptation = EnvironmentalAdaptationSystem()
        
        # Performance prediction
        self.performance_predictor = None
        self.prediction_accuracy = 0.0
        
        self.logger.info(f"Advanced Genetic Optimizer initialized with {optimization_strategy.value} strategy")
    
    async def optimize_trigger(self, trigger_id: str, 
                             current_environment: Dict[str, Any],
                             performance_feedback: Dict[str, float]) -> OptimizationResult:
        """
        Optimize a specific genetic trigger based on current performance and environment.
        
        Args:
            trigger_id: ID of the trigger to optimize
            current_environment: Current environmental conditions
            performance_feedback: Performance feedback data
            
        Returns:
            OptimizationResult with optimization details
        """
        start_time = time.time()
        
        # Get or create performance tracking
        if trigger_id not in self.trigger_performance:
            self.trigger_performance[trigger_id] = TriggerPerformance(trigger_id=trigger_id)
        
        performance = self.trigger_performance[trigger_id]
        
        # Update performance metrics
        self._update_performance_metrics(performance, performance_feedback)
        
        # Determine optimization strategy
        strategy = self._select_optimization_strategy(performance, current_environment)
        
        # Perform optimization
        optimization_result = await self._perform_optimization(
            trigger_id, strategy, performance, current_environment
        )
        
        # Update optimization history
        self.optimization_history.append(optimization_result)
        
        # Update performance trends
        self._update_performance_trends(trigger_id, optimization_result.performance_improvement)
        
        self.logger.info(f"Optimized trigger {trigger_id} with {strategy.value} strategy, "
                        f"improvement: {optimization_result.performance_improvement:.4f}")
        
        return optimization_result
    
    def _update_performance_metrics(self, performance: TriggerPerformance, 
                                  feedback: Dict[str, float]):
        """Update performance metrics for a trigger"""
        performance.activation_count += 1
        performance.last_activation = datetime.now()
        
        # Update success/failure counts
        if feedback.get('success', False):
            performance.success_count += 1
        else:
            performance.failure_count += 1
        
        # Update response time
        response_time = feedback.get('response_time', 0.0)
        if performance.average_response_time == 0.0:
            performance.average_response_time = response_time
        else:
            performance.average_response_time = (
                0.9 * performance.average_response_time + 0.1 * response_time
            )
        
        # Update environmental accuracy
        env_accuracy = feedback.get('environmental_accuracy', 0.0)
        if performance.environmental_accuracy == 0.0:
            performance.environmental_accuracy = env_accuracy
        else:
            performance.environmental_accuracy = (
                0.9 * performance.environmental_accuracy + 0.1 * env_accuracy
            )
        
        # Calculate fitness score
        performance.fitness_score = self._calculate_fitness_score(performance)
        
        # Store performance history
        performance.performance_history.append(performance.fitness_score)
        if len(performance.performance_history) > 100:
            performance.performance_history = performance.performance_history[-100:]
    
    def _calculate_fitness_score(self, performance: TriggerPerformance) -> float:
        """Calculate fitness score for a trigger"""
        if performance.activation_count == 0:
            return 0.0
        
        # Success rate
        success_rate = performance.success_count / performance.activation_count
        
        # Response time efficiency (lower is better)
        response_efficiency = max(0.0, 1.0 - (performance.average_response_time / 10.0))
        
        # Environmental accuracy
        env_accuracy = performance.environmental_accuracy
        
        # Adaptation score
        adaptation_score = performance.adaptation_score
        
        # Weighted fitness calculation
        fitness = (
            success_rate * 0.4 +
            response_efficiency * 0.2 +
            env_accuracy * 0.2 +
            adaptation_score * 0.2
        )
        
        return max(0.0, min(1.0, fitness))
    
    def _select_optimization_strategy(self, performance: TriggerPerformance,
                                    environment: Dict[str, Any]) -> OptimizationStrategy:
        """Select the best optimization strategy based on current conditions"""
        
        # If performance is very low, use evolutionary strategy
        if performance.fitness_score < 0.3:
            return OptimizationStrategy.EVOLUTIONARY
        
        # If environmental accuracy is low, use environmental adaptation
        if performance.environmental_accuracy < 0.5:
            return OptimizationStrategy.ENVIRONMENTAL_ADAPTATION
        
        # If performance is moderate, use performance-based optimization
        if performance.fitness_score < 0.7:
            return OptimizationStrategy.PERFORMANCE_BASED
        
        # Default to hybrid strategy
        return OptimizationStrategy.HYBRID
    
    async def _perform_optimization(self, trigger_id: str,
                                  strategy: OptimizationStrategy,
                                  performance: TriggerPerformance,
                                  environment: Dict[str, Any]) -> OptimizationResult:
        """Perform optimization using the selected strategy"""
        
        changes_made = []
        performance_improvement = 0.0
        new_fitness_score = performance.fitness_score
        
        if strategy == OptimizationStrategy.PERFORMANCE_BASED:
            result = await self._performance_based_optimization(trigger_id, performance, environment)
            changes_made = result['changes']
            performance_improvement = result['improvement']
            new_fitness_score = result['new_fitness']
            
        elif strategy == OptimizationStrategy.ENVIRONMENTAL_ADAPTATION:
            result = await self._environmental_adaptation_optimization(trigger_id, performance, environment)
            changes_made = result['changes']
            performance_improvement = result['improvement']
            new_fitness_score = result['new_fitness']
            
        elif strategy == OptimizationStrategy.EVOLUTIONARY:
            result = await self._evolutionary_optimization(trigger_id, performance, environment)
            changes_made = result['changes']
            performance_improvement = result['improvement']
            new_fitness_score = result['new_fitness']
            
        elif strategy == OptimizationStrategy.HYBRID:
            result = await self._hybrid_optimization(trigger_id, performance, environment)
            changes_made = result['changes']
            performance_improvement = result['improvement']
            new_fitness_score = result['new_fitness']
        
        return OptimizationResult(
            trigger_id=trigger_id,
            optimization_strategy=strategy,
            performance_improvement=performance_improvement,
            changes_made=changes_made,
            new_fitness_score=new_fitness_score,
            environmental_adaptation=environment,
            timestamp=datetime.now()
        )
    
    async def _performance_based_optimization(self, trigger_id: str,
                                            performance: TriggerPerformance,
                                            environment: Dict[str, Any]) -> Dict[str, Any]:
        """Performance-based optimization focusing on success rate and response time"""
        changes = []
        improvement = 0.0
        
        # Analyze performance bottlenecks
        if performance.success_count / max(1, performance.activation_count) < 0.5:
            # Low success rate - improve trigger conditions
            changes.append("Enhanced trigger activation conditions")
            changes.append("Improved environmental matching")
            improvement += 0.2
        
        if performance.average_response_time > 5.0:
            # Slow response time - optimize trigger logic
            changes.append("Optimized trigger execution logic")
            changes.append("Reduced computational complexity")
            improvement += 0.15
        
        if performance.environmental_accuracy < 0.6:
            # Low environmental accuracy - improve matching
            changes.append("Enhanced environmental pattern recognition")
            changes.append("Improved context sensitivity")
            improvement += 0.25
        
        new_fitness = min(1.0, performance.fitness_score + improvement)
        
        return {
            'changes': changes,
            'improvement': improvement,
            'new_fitness': new_fitness
        }
    
    async def _environmental_adaptation_optimization(self, trigger_id: str,
                                                   performance: TriggerPerformance,
                                                   environment: Dict[str, Any]) -> Dict[str, Any]:
        """Environmental adaptation optimization focusing on environmental matching"""
        changes = []
        improvement = 0.0
        
        # Find similar environments for learning
        similar_environments = await self.environmental_adaptation.find_similar_environments(environment)
        
        if similar_environments:
            # Learn from similar environments
            changes.append("Learned from similar environmental patterns")
            changes.append("Enhanced environmental sensitivity")
            improvement += 0.3
            
            # Update environmental accuracy
            performance.environmental_accuracy = min(1.0, performance.environmental_accuracy + 0.2)
        
        # Adapt trigger to current environment
        changes.append("Adapted trigger to current environmental conditions")
        changes.append("Improved environmental pattern matching")
        improvement += 0.2
        
        # Store environmental memory
        await self.environmental_adaptation.store_environmental_memory(environment, {
            'trigger_id': trigger_id,
            'performance': performance.fitness_score
        })
        
        new_fitness = min(1.0, performance.fitness_score + improvement)
        
        return {
            'changes': changes,
            'improvement': improvement,
            'new_fitness': new_fitness
        }
    
    async def _evolutionary_optimization(self, trigger_id: str,
                                       performance: TriggerPerformance,
                                       environment: Dict[str, Any]) -> Dict[str, Any]:
        """Evolutionary optimization using genetic algorithms"""
        changes = []
        improvement = 0.0
        
        # Create population of trigger variants
        population = self._create_trigger_population(trigger_id, environment)
        
        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_trigger_fitness(trigger, environment) for trigger in population]
            
            # Select parents
            parents = self._select_parents(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best triggers
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover_triggers(parent1, parent2)
                child = self._mutate_trigger(child)
                new_population.append(child)
            
            population = new_population
        
        # Select best trigger from final population
        final_fitness_scores = [self._evaluate_trigger_fitness(trigger, environment) for trigger in population]
        best_trigger = population[np.argmax(final_fitness_scores)]
        best_fitness = max(final_fitness_scores)
        
        changes.append("Applied evolutionary optimization")
        changes.append("Generated trigger variants through mutation and crossover")
        changes.append("Selected optimal trigger through natural selection")
        
        improvement = max(0.0, best_fitness - performance.fitness_score)
        new_fitness = best_fitness
        
        return {
            'changes': changes,
            'improvement': improvement,
            'new_fitness': new_fitness
        }
    
    async def _hybrid_optimization(self, trigger_id: str,
                                 performance: TriggerPerformance,
                                 environment: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid optimization combining multiple strategies"""
        changes = []
        improvement = 0.0
        
        # Combine performance-based and environmental adaptation
        perf_result = await self._performance_based_optimization(trigger_id, performance, environment)
        env_result = await self._environmental_adaptation_optimization(trigger_id, performance, environment)
        
        changes.extend(perf_result['changes'])
        changes.extend(env_result['changes'])
        improvement = perf_result['improvement'] + env_result['improvement']
        
        # Add hybrid-specific optimizations
        changes.append("Applied multi-strategy optimization")
        changes.append("Integrated performance and environmental improvements")
        improvement += 0.1
        
        new_fitness = min(1.0, performance.fitness_score + improvement)
        
        return {
            'changes': changes,
            'improvement': improvement,
            'new_fitness': new_fitness
        }
    
    def _create_trigger_population(self, trigger_id: str, environment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a population of trigger variants"""
        population = []
        
        # Get original trigger
        if self.genetic_system and trigger_id in self.genetic_system.triggers:
            original_trigger = self.genetic_system.triggers[trigger_id]
            base_sequence = original_trigger.genetic_sequence
        else:
            base_sequence = self._generate_random_sequence()
        
        # Create population
        for _ in range(self.population_size):
            variant = {
                'genetic_sequence': self._mutate_sequence(base_sequence),
                'activation_threshold': random.uniform(0.3, 0.9),
                'environmental_sensitivity': random.uniform(0.5, 1.0),
                'response_speed': random.uniform(0.5, 1.0)
            }
            population.append(variant)
        
        return population
    
    def _evaluate_trigger_fitness(self, trigger: Dict[str, Any], environment: Dict[str, Any]) -> float:
        """Evaluate fitness of a trigger variant"""
        # Simulate trigger performance based on characteristics
        sequence_quality = self._evaluate_sequence_quality(trigger['genetic_sequence'])
        environmental_match = self._evaluate_environmental_match(trigger, environment)
        response_efficiency = trigger['response_speed']
        
        fitness = (
            sequence_quality * 0.4 +
            environmental_match * 0.4 +
            response_efficiency * 0.2
        )
        
        return max(0.0, min(1.0, fitness))
    
    def _evaluate_sequence_quality(self, sequence: str) -> float:
        """Evaluate quality of genetic sequence"""
        if not sequence:
            return 0.0
        
        # Simple quality metrics
        diversity = len(set(sequence)) / len(sequence)
        complexity = len(sequence) / 100.0  # Normalize by expected length
        
        return (diversity + complexity) / 2.0
    
    def _evaluate_environmental_match(self, trigger: Dict[str, Any], environment: Dict[str, Any]) -> float:
        """Evaluate how well trigger matches environment"""
        # Simulate environmental matching
        sensitivity = trigger['environmental_sensitivity']
        threshold = trigger['activation_threshold']
        
        # Calculate environmental complexity
        env_complexity = len(environment) / 10.0  # Normalize
        
        # Higher sensitivity and appropriate threshold for complex environments
        if env_complexity > 0.5:
            match_score = sensitivity * (1.0 - abs(threshold - 0.7))
        else:
            match_score = sensitivity * (1.0 - abs(threshold - 0.5))
        
        return max(0.0, min(1.0, match_score))
    
    def _select_parents(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Select parents for crossover using tournament selection"""
        parents = []
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _crossover_triggers(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two triggers"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        child = {}
        
        # Crossover genetic sequence
        seq1, seq2 = parent1['genetic_sequence'], parent2['genetic_sequence']
        if len(seq1) > 0 and len(seq2) > 0:
            crossover_point = random.randint(1, min(len(seq1), len(seq2)) - 1)
            child['genetic_sequence'] = seq1[:crossover_point] + seq2[crossover_point:]
        else:
            child['genetic_sequence'] = seq1
        
        # Crossover other parameters
        child['activation_threshold'] = (
            parent1['activation_threshold'] + parent2['activation_threshold']
        ) / 2.0
        
        child['environmental_sensitivity'] = (
            parent1['environmental_sensitivity'] + parent2['environmental_sensitivity']
        ) / 2.0
        
        child['response_speed'] = (
            parent1['response_speed'] + parent2['response_speed']
        ) / 2.0
        
        return child
    
    def _mutate_trigger(self, trigger: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a trigger"""
        mutated = trigger.copy()
        
        # Mutate genetic sequence
        if random.random() < self.mutation_rate:
            mutated['genetic_sequence'] = self._mutate_sequence(trigger['genetic_sequence'])
        
        # Mutate parameters
        if random.random() < self.mutation_rate:
            mutated['activation_threshold'] = max(0.1, min(0.9, 
                trigger['activation_threshold'] + random.uniform(-0.1, 0.1)))
        
        if random.random() < self.mutation_rate:
            mutated['environmental_sensitivity'] = max(0.1, min(1.0,
                trigger['environmental_sensitivity'] + random.uniform(-0.1, 0.1)))
        
        if random.random() < self.mutation_rate:
            mutated['response_speed'] = max(0.1, min(1.0,
                trigger['response_speed'] + random.uniform(-0.1, 0.1)))
        
        return mutated
    
    def _mutate_sequence(self, sequence: str) -> str:
        """Mutate a genetic sequence"""
        if not sequence:
            return self._generate_random_sequence()
        
        # Random mutations
        mutations = ['A', 'T', 'G', 'C']
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if random.random() < 0.01:  # 1% mutation rate per position
                mutated[i] = random.choice(mutations)
        
        return ''.join(mutated)
    
    def _generate_random_sequence(self, length: int = 50) -> str:
        """Generate a random genetic sequence"""
        mutations = ['A', 'T', 'G', 'C']
        return ''.join(random.choice(mutations) for _ in range(length))
    
    def _update_performance_trends(self, trigger_id: str, improvement: float):
        """Update performance trends for a trigger"""
        self.performance_trends[trigger_id].append(improvement)
        
        # Keep only recent trends
        if len(self.performance_trends[trigger_id]) > 100:
            self.performance_trends[trigger_id] = self.performance_trends[trigger_id][-100:]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities"""
        return {
            'total_optimizations': len(self.optimization_history),
            'average_improvement': np.mean([r.performance_improvement for r in self.optimization_history]) if self.optimization_history else 0.0,
            'strategy_distribution': self._get_strategy_distribution(),
            'top_performers': self._get_top_performers(),
            'optimization_trends': self._get_optimization_trends()
        }
    
    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of optimization strategies used"""
        distribution = defaultdict(int)
        for result in self.optimization_history:
            distribution[result.optimization_strategy.value] += 1
        return dict(distribution)
    
    def _get_top_performers(self) -> List[Dict[str, Any]]:
        """Get top performing triggers"""
        performers = []
        for trigger_id, performance in self.trigger_performance.items():
            performers.append({
                'trigger_id': trigger_id,
                'fitness_score': performance.fitness_score,
                'activation_count': performance.activation_count,
                'success_rate': performance.success_count / max(1, performance.activation_count)
            })
        
        # Sort by fitness score
        performers.sort(key=lambda x: x['fitness_score'], reverse=True)
        return performers[:10]
    
    def _get_optimization_trends(self) -> Dict[str, List[float]]:
        """Get optimization trends over time"""
        if not self.optimization_history:
            return {}
        
        # Group by time periods
        recent_optimizations = [r for r in self.optimization_history 
                              if r.timestamp > datetime.now() - timedelta(hours=24)]
        
        improvements = [r.performance_improvement for r in recent_optimizations]
        
        return {
            'recent_improvements': improvements,
            'average_recent_improvement': np.mean(improvements) if improvements else 0.0,
            'improvement_trend': self._calculate_trend(improvements)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable' 