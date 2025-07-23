#!/usr/bin/env python3
"""
Synthetic Selection Factories and Managers for MCP Core System
Implements advanced synthetic selection with factories, managers, and multiplexers.
"""

import asyncio
import logging
import os
import time
import json
import threading
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import queue
import copy
from abc import ABC, abstractmethod
from functools import lru_cache

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SelectionType(Enum):
    """Types of synthetic selection."""
    PERFORMANCE_BASED = "performance_based"
    DIVERSITY_BASED = "diversity_based"
    NOVELTY_BASED = "novelty_based"
    STABILITY_BASED = "stability_based"
    EFFICIENCY_BASED = "efficiency_based"
    HYBRID = "hybrid"
    GRADIENT_BASED = "gradient_based"
    ATTENTION_BASED = "attention_based"
    SPARSE_BASED = "sparse_based"
    QUANTUM_INSPIRED = "quantum_inspired"

class FactoryType(Enum):
    """Types of selection factories."""
    TOURNAMENT_FACTORY = "tournament_factory"
    ROULETTE_FACTORY = "roulette_factory"
    RANK_FACTORY = "rank_factory"
    PARETO_FACTORY = "pareto_factory"
    NOVELTY_FACTORY = "novelty_factory"
    ENSEMBLE_FACTORY = "ensemble_factory"
    GRADIENT_FACTORY = "gradient_factory"
    ATTENTION_FACTORY = "attention_factory"
    SPARSE_FACTORY = "sparse_factory"
    QUANTUM_FACTORY = "quantum_factory"

@dataclass
class SelectionCandidate:
    """Candidate for synthetic selection."""
    candidate_id: str
    model: nn.Module
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    novelty_score: float = 0.0
    stability_score: float = 0.0
    efficiency_score: float = 0.0
    gradient_magnitude: float = 0.0
    attention_entropy: float = 0.0
    sparsity_ratio: float = 0.0
    quantum_coherence: float = 0.0
    age: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    selection_history: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class SelectionFactory(ABC):
    """Abstract base class for selection factories."""
    
    def __init__(self, factory_id: str, selection_type: SelectionType):
        """Initialize selection factory."""
        self.factory_id = factory_id
        self.selection_type = selection_type
        self.logger = logging.getLogger(f"selection_factory.{factory_id}")
        
        # Factory metrics
        self.selections_made = 0
        self.success_rate = 0.0
        self.average_fitness_improvement = 0.0
        
    @abstractmethod
    def select_candidates(self,
                         candidates: List[SelectionCandidate],
    def select_candidates(self, 
                         candidates: List[SelectionCandidate], 
                         num_selections: int,
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates based on factory-specific criteria."""
        pass
    
    @abstractmethod
    def evaluate_selection_quality(self,
    def evaluate_selection_quality(self, 
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate the quality of the selection."""
        pass

class GradientBasedSelectionFactory(SelectionFactory):
    """Gradient-based selection using gradient magnitude and direction."""
    
    def __init__(self, factory_id: str, gradient_threshold: float = 0.1):
        """Initialize gradient-based selection factory."""
        super().__init__(factory_id, SelectionType.GRADIENT_BASED)
        self.gradient_threshold = gradient_threshold
    
    def select_candidates(self,
                         candidates: List[SelectionCandidate],
    def select_candidates(self, 
                         candidates: List[SelectionCandidate], 
                         num_selections: int,
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates based on gradient properties."""
        if not candidates:
            return []
        
        # Calculate gradient magnitudes
        for candidate in candidates:
            candidate.gradient_magnitude = self._calculate_gradient_magnitude(candidate.model)
        
        # Sort by gradient magnitude (higher is better for learning potential)
        candidates_by_gradient = sorted(
            candidates,
            key=lambda c: c.gradient_magnitude,
            reverse=True
        )
        
        selected = candidates_by_gradient[:num_selections]
        
        for candidate in selected:
            candidate.selection_history.append(f"gradient_{self.factory_id}")

        self.selections_made += num_selections
        return selected

    def _calculate_gradient_magnitude(self, model: nn.Module) -> float:
        """Calculate average gradient magnitude for model."""
        total_grad_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                param_count += 1

        return total_grad_norm / max(param_count, 1)

    def evaluate_selection_quality(self,
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate gradient selection quality."""
        if not selected:
            return 0.0

        avg_gradient = sum(c.gradient_magnitude for c in selected) / len(selected)
        return min(avg_gradient / 10.0, 1.0)  # Normalize

class AttentionBasedSelectionFactory(SelectionFactory):
    """Attention-based selection using attention entropy and patterns."""

    def __init__(self, factory_id: str):
        """Initialize attention-based selection factory."""
        super().__init__(factory_id, SelectionType.ATTENTION_BASED)

    def select_candidates(self,
                         candidates: List[SelectionCandidate],
                         num_selections: int,
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates based on attention patterns."""
        if not candidates:
            return []

        # Calculate attention entropy
        for candidate in candidates:
            candidate.attention_entropy = self._calculate_attention_entropy(candidate.model)

        # Sort by attention entropy (higher entropy = more diverse attention)
        candidates_by_attention = sorted(
            candidates,
            key=lambda c: c.attention_entropy,
            reverse=True
        )

        selected = candidates_by_attention[:num_selections]

        # Update selection history in batch
        selection_tag = f"gradient_{self.factory_id}"
        for candidate in selected:
            candidate.selection_history.append(f"attention_{self.factory_id}")
            candidate.selection_history.append(selection_tag)
        
        self.selections_made += num_selections
        return selected
    
    def _calculate_attention_entropy(self, model: nn.Module) -> float:
        """Calculate attention entropy for transformer-like models."""
        entropy_sum = 0.0
        attention_layers = 0

        for module in model.modules():
            if hasattr(module, 'attention') or 'attention' in str(type(module)).lower():
                # Simulate attention entropy calculation
                # In practice, this would analyze actual attention weights
                entropy_sum += random.uniform(0.5, 2.0)  # Placeholder
                attention_layers += 1

        return entropy_sum / max(attention_layers, 1)

    def evaluate_selection_quality(self,
    @lru_cache(maxsize=128)
    def _calculate_gradient_magnitude(self, model: nn.Module) -> float:
        """Calculate average gradient magnitude for model."""
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                param_count += 1
        
        return total_grad_norm / max(param_count, 1)
    def evaluate_selection_quality(self, 
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate attention selection quality."""
        """Evaluate gradient selection quality."""
        if not selected:
            return 0.0
        
        avg_entropy = sum(c.attention_entropy for c in selected) / len(selected)
        return min(avg_entropy / 3.0, 1.0)  # Normalize
        avg_gradient = sum(c.gradient_magnitude for c in selected) / len(selected)
        return min(avg_gradient / 10.0, 1.0)  # Normalize

class SparseBasedSelectionFactory(SelectionFactory):
    """Sparse-based selection using sparsity patterns and efficiency."""
class AttentionBasedSelectionFactory(SelectionFactory):
    """Attention-based selection using attention entropy and patterns."""
    
    def __init__(self, factory_id: str, target_sparsity: float = 0.5):
        """Initialize sparse-based selection factory."""
        super().__init__(factory_id, SelectionType.SPARSE_BASED)
        self.target_sparsity = target_sparsity

    def select_candidates(self,
                         candidates: List[SelectionCandidate],
    def __init__(self, factory_id: str):
        """Initialize attention-based selection factory."""
        super().__init__(factory_id, SelectionType.ATTENTION_BASED)
    def select_candidates(self, 
                         candidates: List[SelectionCandidate], 
                         num_selections: int,
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates based on sparsity patterns."""
        """Select candidates based on attention patterns."""
        if not candidates:
            return []
        
        # Calculate sparsity ratios
        # Calculate attention entropy in parallel
        for candidate in candidates:
            candidate.sparsity_ratio = self._calculate_sparsity_ratio(candidate.model)
            candidate.attention_entropy = self._calculate_attention_entropy(candidate.model)
        
        # Select candidates closest to target sparsity
        candidates_by_sparsity = sorted(
        # Sort by attention entropy (higher entropy = more diverse attention)
        candidates_by_attention = sorted(
            candidates,
            key=lambda c: abs(c.sparsity_ratio - self.target_sparsity)
            key=lambda c: c.attention_entropy,
            reverse=True
        )
        
        selected = candidates_by_sparsity[:num_selections]
        selected = candidates_by_attention[:num_selections]
        
        # Batch update selection history
        selection_tag = f"attention_{self.factory_id}"
        for candidate in selected:
            candidate.selection_history.append(f"sparse_{self.factory_id}")
            candidate.selection_history.append(selection_tag)
        
        self.selections_made += num_selections
        return selected
    
    def _calculate_sparsity_ratio(self, model: nn.Module) -> float:
        """Calculate sparsity ratio of model parameters."""
        total_params = 0
        zero_params = 0
    @lru_cache(maxsize=128)
    def _calculate_attention_entropy(self, model: nn.Module) -> float:
        """Calculate attention entropy for transformer-like models."""
        entropy_sum = 0.0
        attention_layers = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.abs() < 1e-6).sum().item()

        return zero_params / max(total_params, 1)

    def evaluate_selection_quality(self,
        for module in model.modules():
            if hasattr(module, 'attention') or 'attention' in str(type(module)).lower():
                # Simulate attention entropy calculation
                # In practice, this would analyze actual attention weights
                entropy_sum += random.uniform(0.5, 2.0)  # Placeholder
                attention_layers += 1
        
        return entropy_sum / max(attention_layers, 1)
    def evaluate_selection_quality(self, 
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate sparse selection quality."""
        """Evaluate attention selection quality."""
        if not selected:
            return 0.0
        
        # Quality based on how close to target sparsity
        avg_deviation = sum(
            abs(c.sparsity_ratio - self.target_sparsity) for c in selected
        ) / len(selected)

        return max(0.0, 1.0 - avg_deviation)

class QuantumInspiredSelectionFactory(SelectionFactory):
    """Quantum-inspired selection using superposition and entanglement concepts."""

    def __init__(self, factory_id: str, coherence_threshold: float = 0.7):
        """Initialize quantum-inspired selection factory."""
        super().__init__(factory_id, SelectionType.QUANTUM_INSPIRED)
        self.coherence_threshold = coherence_threshold

    def select_candidates(self,
                         candidates: List[SelectionCandidate],
        avg_entropy = sum(c.attention_entropy for c in selected) / len(selected)
        return min(avg_entropy / 3.0, 1.0)  # Normalize

class SparseBasedSelectionFactory(SelectionFactory):
    """Sparse-based selection using sparsity patterns and efficiency."""
    
    def __init__(self, factory_id: str, target_sparsity: float = 0.5):
        """Initialize sparse-based selection factory."""
        super().__init__(factory_id, SelectionType.SPARSE_BASED)
        self.target_sparsity = target_sparsity
    def select_candidates(self, 
                         candidates: List[SelectionCandidate], 
                         num_selections: int,
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates using quantum-inspired principles."""
        """Select candidates based on sparsity patterns."""
        if not candidates:
            return []
        
        # Calculate quantum coherence scores
        # Calculate sparsity ratios
        for candidate in candidates:
            candidate.quantum_coherence = self._calculate_quantum_coherence(candidate.model)
            candidate.sparsity_ratio = self._calculate_sparsity_ratio(candidate.model)
        
        # Quantum superposition selection - probabilistic based on coherence
        selected = []
        remaining_candidates = candidates.copy()

        for _ in range(num_selections):
            if not remaining_candidates:
                break

            # Calculate selection probabilities based on quantum coherence
            coherence_scores = [c.quantum_coherence for c in remaining_candidates]
            total_coherence = sum(coherence_scores)

            if total_coherence == 0:
                # Uniform selection if no coherence
                selected_candidate = random.choice(remaining_candidates)
            else:
                # Probabilistic selection
                probabilities = [score / total_coherence for score in coherence_scores]
                selected_candidate = np.random.choice(remaining_candidates, p=probabilities)

            selected.append(selected_candidate)
            remaining_candidates.remove(selected_candidate)
            selected_candidate.selection_history.append(f"quantum_{self.factory_id}")

        self.selections_made += num_selections
        return selected

    def _calculate_quantum_coherence(self, model: nn.Module) -> float:
        """Calculate quantum coherence score based on parameter correlations."""
        coherence_sum = 0.0
        layer_count = 0

        layers = list(model.modules())

        for i, layer in enumerate(layers):
            if hasattr(layer, 'weight') and layer.weight is not None:
                # Calculate "coherence" as parameter correlation patterns
                weight = layer.weight.detach().flatten()

                # Simple coherence measure: variance of parameter magnitudes
                coherence = torch.var(weight.abs()).item()
                coherence_sum += coherence
                layer_count += 1

        return coherence_sum / max(layer_count, 1)

    def evaluate_selection_quality(self,
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate quantum selection quality."""
        if not selected:
            return 0.0

        avg_coherence = sum(c.quantum_coherence for c in selected) / len(selected)
        return min(avg_coherence / 5.0, 1.0)  # Normalize

# Continue with existing factories...
class TournamentSelectionFactory(SelectionFactory):
    """Tournament-based selection factory."""

    def __init__(self, factory_id: str, tournament_size: int = 5):
        """Initialize tournament selection factory."""
        super().__init__(factory_id, SelectionType.PERFORMANCE_BASED)
        self.tournament_size = tournament_size

    def select_candidates(self,
                         candidates: List[SelectionCandidate],
                         num_selections: int,
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates using tournament selection."""
        if not candidates:
            return []

        fitness_key = selection_criteria.get('fitness_key', 'overall') if selection_criteria else 'overall'
        selected = []

        for _ in range(num_selections):
            # Run tournament
            tournament_candidates = random.sample(
        # Select candidates closest to target sparsity
        candidates_by_sparsity = sorted(
                candidates, 
                min(self.tournament_size, len(candidates))
            key=lambda c: abs(c.sparsity_ratio - self.target_sparsity)
            )
            
            # Select winner based on fitness
            winner = max(
                tournament_candidates,
                key=lambda c: c.fitness_scores.get(fitness_key, 0.0)
            )

            selected.append(winner)
            winner.selection_history.append(f"tournament_{self.factory_id}")

        selected = candidates_by_sparsity[:num_selections]
        
        # Batch update selection history
        selection_tag = f"sparse_{self.factory_id}"
        for candidate in selected:
            candidate.selection_history.append(selection_tag)
        self.selections_made += num_selections
        return selected
    
    def evaluate_selection_quality(self,
    @lru_cache(maxsize=128)
    def _calculate_sparsity_ratio(self, model: nn.Module) -> float:
        """Calculate sparsity ratio of model parameters."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            param_data = param.data
            total_params += param_data.numel()
            zero_params += (param_data.abs() < 1e-6).sum().item()
        
        return zero_params / max(total_params, 1)
    def evaluate_selection_quality(self, 
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate tournament selection quality."""
        if not selected or not all_candidates:
        """Evaluate sparse selection quality."""
        if not selected:
            return 0.0
        
        selected_fitness = [
            sum(c.fitness_scores.values()) / max(1, len(c.fitness_scores))
            for c in selected
        ]
        all_fitness = [
            sum(c.fitness_scores.values()) / max(1, len(c.fitness_scores))
            for c in all_candidates
        ]

        avg_selected = sum(selected_fitness) / len(selected_fitness)
        avg_all = sum(all_fitness) / len(all_fitness)

        return avg_selected / max(avg_all, 1e-6)

class SelectionManager:
    """Manager for coordinating multiple selection factories."""

    def __init__(self, manager_id: str):
        """Initialize selection manager."""
        self.manager_id = manager_id
        self.factories: Dict[str, SelectionFactory] = {}
        self.factory_performance: Dict[str, Dict[str, float]] = {}

        # Manager state
        self.total_selections = 0
        self.selection_history = []

        self.logger = logging.getLogger(f"selection_manager.{manager_id}")

    def register_factory(self, factory: SelectionFactory):
        """Register a selection factory."""
        self.factories[factory.factory_id] = factory
        self.factory_performance[factory.factory_id] = {
            'selections_made': 0,
            'success_rate': 0.0,
            'average_quality': 0.0,
            'last_used': 0.0
        }

        self.logger.info(f"Registered factory: {factory.factory_id}")

    def select_candidates(self,
                         candidates: List[SelectionCandidate],
        # Quality based on how close to target sparsity
        avg_deviation = sum(
            abs(c.sparsity_ratio - self.target_sparsity) for c in selected
        ) / len(selected)
        
        return max(0.0, 1.0 - avg_deviation)

class QuantumInspiredSelectionFactory(SelectionFactory):
    """Quantum-inspired selection using superposition and entanglement concepts."""
    
    def __init__(self, factory_id: str, coherence_threshold: float = 0.7):
        """Initialize quantum-inspired selection factory."""
        super().__init__(factory_id, SelectionType.QUANTUM_INSPIRED)
        self.coherence_threshold = coherence_threshold
    def select_candidates(self, 
                         candidates: List[SelectionCandidate], 
                         num_selections: int,
                         selection_strategy: str = "best_factory",
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates using managed factories."""
        if not self.factories or not candidates:
        """Select candidates using quantum-inspired principles."""
        if not candidates:
            return []
        
        if selection_strategy == "best_factory":
            factory = self._select_best_factory()
            selected = factory.select_candidates(candidates, num_selections, selection_criteria)
        elif selection_strategy == "ensemble":
            selected = self._ensemble_selection(candidates, num_selections, selection_criteria)
        # Calculate quantum coherence scores
        for candidate in candidates:
            candidate.quantum_coherence = self._calculate_quantum_coherence(candidate.model)
        
        # Quantum superposition selection - probabilistic based on coherence
        selected = []
        remaining_candidates = candidates.copy()
        
        if not remaining_candidates:
            return []
            
        # Calculate selection probabilities based on quantum coherence
        coherence_scores = np.array([c.quantum_coherence for c in remaining_candidates])
        total_coherence = coherence_scores.sum()
        
        if total_coherence == 0:
            # Uniform selection if no coherence
            indices = np.random.choice(len(remaining_candidates), 
                                      min(num_selections, len(remaining_candidates)), 
                                      replace=False)
            selected = [remaining_candidates[i] for i in indices]
        else:
            # Use specific factory or fallback
            factory = self.factories.get(selection_strategy) or self._select_best_factory()
            selected = factory.select_candidates(candidates, num_selections, selection_criteria)

        self.total_selections += len(selected)
            # Probabilistic selection
            probabilities = coherence_scores / total_coherence
            indices = np.random.choice(len(remaining_candidates), 
                                      min(num_selections, len(remaining_candidates)), 
                                      replace=False, 
                                      p=probabilities)
            selected = [remaining_candidates[i] for i in indices]
        
        # Batch update selection history
        selection_tag = f"quantum_{self.factory_id}"
        for candidate in selected:
            candidate.selection_history.append(selection_tag)
        self.selections_made += len(selected)
        return selected
    
    def _select_best_factory(self) -> SelectionFactory:
        """Select the best performing factory."""
        if not self.factories:
            return None
    @lru_cache(maxsize=128)
    def _calculate_quantum_coherence(self, model: nn.Module) -> float:
        """Calculate quantum coherence score based on parameter correlations."""
        coherence_sum = 0.0
        layer_count = 0
        
        for layer in model.modules():
            if hasattr(layer, 'weight') and layer.weight is not None:
                # Calculate "coherence" as parameter correlation patterns
                weight = layer.weight.detach().flatten()
                
                # Simple coherence measure: variance of parameter magnitudes
                coherence = torch.var(weight.abs()).item()
                coherence_sum += coherence
                layer_count += 1
        
        return coherence_sum / max(layer_count, 1)
    
    def evaluate_selection_quality(self, 
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate quantum selection quality."""
        if not selected:
            return 0.0
        
        best_factory_id = max(
            self.factory_performance.keys(),
            key=lambda fid: self.factory_performance[fid]['average_quality']
        )

        return self.factories[best_factory_id]

    def _ensemble_selection(self,
                           candidates: List[SelectionCandidate],
        avg_coherence = sum(c.quantum_coherence for c in selected) / len(selected)
        return min(avg_coherence / 5.0, 1.0)  # Normalize

# Continue with existing factories...
class TournamentSelectionFactory(SelectionFactory):
    """Tournament-based selection factory."""
    
    def __init__(self, factory_id: str, tournament_size: int = 5):
        """Initialize tournament selection factory."""
        super().__init__(factory_id, SelectionType.PERFORMANCE_BASED)
        self.tournament_size = tournament_size
    
    def select_candidates(self, 
                         candidates: List[SelectionCandidate], 
                           num_selections: int,
                           selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Perform ensemble selection using multiple factories."""
        factories = list(self.factories.values())
        selections_per_factory = num_selections // len(factories)

        all_selected = []
        for factory in factories:
            if selections_per_factory > 0:
                selected = factory.select_candidates(
                    candidates, selections_per_factory, selection_criteria
        """Select candidates using tournament selection."""
        if not candidates:
            return []
        
        fitness_key = selection_criteria.get('fitness_key', 'overall') if selection_criteria else 'overall'
        selected = []
        
        # Pre-compute fitness values to avoid repeated dictionary lookups
        fitness_values = {c: c.fitness_scores.get(fitness_key, 0.0) for c in candidates}
        
        for _ in range(num_selections):
            # Run tournament
            tournament_candidates = random.sample(
                candidates, 
                min(self.tournament_size, len(candidates))
                )
                all_selected.extend(selected)
            
            # Select winner based on fitness
            winner = max(
                tournament_candidates,
                key=lambda c: fitness_values[c]
            )
            selected.append(winner)
        
        # Remove duplicates
        seen = set()
        unique_selected = []
        for candidate in all_selected:
            if candidate.candidate_id not in seen:
                seen.add(candidate.candidate_id)
                unique_selected.append(candidate)
        # Batch update selection history
        selection_tag = f"tournament_{self.factory_id}"
        for candidate in selected:
            candidate.selection_history.append(selection_tag)
            
        self.selections_made += num_selections
        return selected
    
    def evaluate_selection_quality(self, 
                                  selected: List[SelectionCandidate],
                                  all_candidates: List[SelectionCandidate]) -> float:
        """Evaluate tournament selection quality."""
        if not selected or not all_candidates:
            return 0.0
        
        # Vectorized calculation of average fitness
        def avg_fitness(candidates):
            return np.mean([
                np.mean(list(c.fitness_scores.values()) or [0]) 
                for c in candidates
            ])
        
        avg_selected = avg_fitness(selected)
        avg_all = avg_fitness(all_candidates)
        
        return avg_selected / max(avg_all, 1e-6)

class SelectionManager:
    """Manager for coordinating multiple selection factories."""
    
    def __init__(self, manager_id: str):
        """Initialize selection manager."""
        self.manager_id = manager_id
        self.factories: Dict[str, SelectionFactory] = {}
        self.factory_performance: Dict[str, Dict[str, float]] = {}
        
        # Manager state
        self.total_selections = 0
        self.selection_history = []
        
        self.logger = logging.getLogger(f"selection_manager.{manager_id}")
    
    def register_factory(self, factory: SelectionFactory):
        """Register a selection factory."""
        self.factories[factory.factory_id] = factory
        self.factory_performance[factory.factory_id] = {
            'selections_made': 0,
            'success_rate': 0.0,
            'average_quality': 0.0,
            'last_used': 0.0
        }
        
        self.logger.info(f"Registered factory: {factory.factory_id}")
    
    def select_candidates(self, 
                         candidates: List[SelectionCandidate],
                         num_selections: int,
                         selection_strategy: str = "best_factory",
                         selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Select candidates using managed factories."""
        if not self.factories or not candidates:
            return []
        
        if selection_strategy == "best_factory":
            factory = self._select_best_factory()
            selected = factory.select_candidates(candidates, num_selections, selection_criteria)
        elif selection_strategy == "ensemble":
            selected = self._ensemble_selection(candidates, num_selections, selection_criteria)
        else:
            # Use specific factory or fallback
            factory = self.factories.get(selection_strategy) or self._select_best_factory()
            selected = factory.select_candidates(candidates, num_selections, selection_criteria)
        
        self.total_selections += len(selected)
        return selected
    
    def _select_best_factory(self) -> SelectionFactory:
        """Select the best performing factory."""
        if not self.factories:
            return None
        
        best_factory_id = max(
            self.factory_performance.keys(),
            key=lambda fid: self.factory_performance[fid]['average_quality']
        )
        
        return self.factories[best_factory_id]
    
    def _ensemble_selection(self, 
                           candidates: List[SelectionCandidate],
                           num_selections: int,
                           selection_criteria: Dict[str, Any] = None) -> List[SelectionCandidate]:
        """Perform ensemble selection using multiple factories."""
        factories = list(self.factories.values())
        if not factories:
            return []
            
        selections_per_factory = max(1, num_selections // len(factories))
        
        all_selected = []
        for factory in factories:
            if selections_per_factory > 0:
                selected = factory.select_candidates(
                    candidates, selections_per_factory, selection_criteria
                )
                all_selected.extend(selected)
        
        # Use set for faster duplicate checking
        seen = set()
        unique_selected = []
        for candidate in all_selected:
            if candidate.candidate_id not in seen:
                seen.add(candidate.candidate_id)
                unique_selected.append(candidate)
        
        return unique_selected[:num_selections]

# Convenience functions
def create_selection_manager(manager_id: str) -> SelectionManager:
    """Create a selection manager."""
    return SelectionManager(manager_id)

def create_gradient_factory(factory_id: str, gradient_threshold: float = 0.1) -> GradientBasedSelectionFactory:
    """Create a gradient-based selection factory."""
    return GradientBasedSelectionFactory(factory_id, gradient_threshold)

def create_attention_factory(factory_id: str) -> AttentionBasedSelectionFactory:
    """Create an attention-based selection factory."""
    return AttentionBasedSelectionFactory(factory_id)

def create_sparse_factory(factory_id: str, target_sparsity: float = 0.5) -> SparseBasedSelectionFactory:
    """Create a sparse-based selection factory."""
    return SparseBasedSelectionFactory(factory_id, target_sparsity)

def create_quantum_factory(factory_id: str, coherence_threshold: float = 0.7) -> QuantumInspiredSelectionFactory:
    """Create a quantum-inspired selection factory."""
    return QuantumInspiredSelectionFactory(factory_id, coherence_threshold)