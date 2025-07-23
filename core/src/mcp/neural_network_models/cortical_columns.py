#!/usr/bin/env python3
"""
Cortical Columns for MCP Core System
Implements dynamic swarming and stacking of pattern recognition cortical columns.
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

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ColumnState(Enum):
    """States of cortical columns."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    LEARNING = "learning"
    PREDICTING = "predicting"
    SWARMING = "swarming"
    STACKED = "stacked"

@dataclass
class ColumnMetrics:
    """Metrics for cortical column performance."""
    activation_count: int = 0
    prediction_accuracy: float = 0.0
    learning_rate: float = 0.001
    energy_consumption: float = 0.0
    pattern_diversity: float = 0.0
    collaboration_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class CorticalColumn(nn.Module):
    """
    Individual cortical column with pattern recognition capabilities.
    
    Features:
    - Hierarchical temporal memory (HTM) inspired architecture
    - Sparse distributed representations
    - Predictive coding
    - Lateral inhibition
    - Synaptic plasticity
    """
    
    def __init__(self, 
                 column_id: str,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 64,
                 sparsity: float = 0.02,
                 learning_rate: float = 0.001,
                 device: str = None):
        """Initialize cortical column."""
        super(CorticalColumn, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CorticalColumn")
        
        self.column_id = column_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Column state
        self.state = ColumnState.INACTIVE
        self.metrics = ColumnMetrics()
        self.logger = logging.getLogger(f"cortical_column.{column_id}")
        
        # Neural architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Lateral connections for inhibition
        self.lateral_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        
        # Memory for temporal patterns
        self.memory_buffer = []
        self.max_memory_size = 1000
        
        # Pattern recognition state
        self.active_patterns = set()
        self.predicted_patterns = set()
        
        # Move to device
        self.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x: torch.Tensor, apply_sparsity: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through cortical column."""
        # Encode input
        encoded = self.encoder(x)
        
        # Apply lateral inhibition
        if apply_sparsity:
            encoded = self._apply_lateral_inhibition(encoded)
        
        # Predict next state
        prediction = self.predictor(encoded)
        
        # Decode for reconstruction
        reconstruction = self.decoder(prediction)
        
        return {
            'encoded': encoded,
            'prediction': prediction,
            'reconstruction': reconstruction
        }
    
    def _apply_lateral_inhibition(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply lateral inhibition to maintain sparsity."""
        batch_size = activations.shape[0]
        
        # Calculate inhibition
        inhibition = torch.matmul(activations, self.lateral_weights)
        
        # Apply inhibition
        inhibited = activations - 0.1 * inhibition
        
        # Apply sparsity constraint (top-k activation)
        k = max(1, int(self.hidden_dim * self.sparsity))
        
        # Get top-k activations
        topk_values, topk_indices = torch.topk(inhibited, k, dim=1)
        
        # Create sparse representation
        sparse_activations = torch.zeros_like(inhibited)
        sparse_activations.scatter_(1, topk_indices, topk_values)
        
        return sparse_activations
    
    def learn_pattern(self, input_data: torch.Tensor, target: torch.Tensor = None) -> float:
        """Learn from input pattern."""
        self.state = ColumnState.LEARNING
        self.train()
        
        # Forward pass
        outputs = self.forward(input_data)
        
        # Calculate losses
        reconstruction_loss = F.mse_loss(outputs['reconstruction'], input_data)
        
        if target is not None:
            prediction_loss = F.mse_loss(outputs['prediction'], target)
            total_loss = reconstruction_loss + prediction_loss
        else:
            # Self-supervised learning
            total_loss = reconstruction_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.metrics.activation_count += 1
        self.metrics.energy_consumption += total_loss.item()
        
        # Store pattern in memory
        self._store_pattern(input_data, outputs['encoded'])
        
        return total_loss.item()
    
    def predict_pattern(self, input_data: torch.Tensor) -> torch.Tensor:
        """Predict next pattern given current input."""
        self.state = ColumnState.PREDICTING
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_data, apply_sparsity=False)
            return outputs['prediction']
    
    def _store_pattern(self, input_data: torch.Tensor, encoded: torch.Tensor):
        """Store pattern in memory buffer."""
        pattern = {
            'input': input_data.cpu(),
            'encoded': encoded.cpu(),
            'timestamp': time.time()
        }
        
        self.memory_buffer.append(pattern)
        
        # Limit memory size
        if len(self.memory_buffer) > self.max_memory_size:
            self.memory_buffer.pop(0)
    
    def get_similarity(self, other_column: 'CorticalColumn', input_data: torch.Tensor) -> float:
        """Calculate similarity with another column."""
        self.eval()
        other_column.eval()
        
        with torch.no_grad():
            self_output = self.forward(input_data)
            other_output = other_column.forward(input_data)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                self_output['encoded'].flatten(),
                other_output['encoded'].flatten(),
                dim=0
            ).item()
            
            return similarity
    
    def update_metrics(self, accuracy: float = None):
        """Update column metrics."""
        if accuracy is not None:
            self.metrics.prediction_accuracy = accuracy
        
        # Calculate pattern diversity
        if len(self.memory_buffer) > 1:
            patterns = torch.stack([p['encoded'] for p in self.memory_buffer[-10:]])
            pairwise_distances = torch.pdist(patterns.flatten(1))
            self.metrics.pattern_diversity = pairwise_distances.mean().item()
        
        self.metrics.last_updated = datetime.now()

class ColumnSwarm:
    """
    Dynamic swarm of cortical columns with collaborative processing.
    
    Features:
    - Dynamic column allocation
    - Collaborative pattern recognition
    - Load balancing
    - Emergent specialization
    """
    
    def __init__(self,
                 max_columns: int = 100,
                 min_columns: int = 10,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 64,
                 device: str = None):
        """Initialize column swarm."""
        self.max_columns = max_columns
        self.min_columns = min_columns
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Swarm state
        self.columns: Dict[str, CorticalColumn] = {}
        self.active_columns: List[str] = []
        self.column_assignments: Dict[str, List[str]] = {}  # Task -> Column IDs
        
        # Swarm metrics
        self.swarm_metrics = {
            'total_columns': 0,
            'active_columns': 0,
            'average_accuracy': 0.0,
            'load_balance': 0.0,
            'collaboration_efficiency': 0.0
        }
        
        self.logger = logging.getLogger("column_swarm")
        
        # Initialize minimum columns
        self._initialize_columns()
    
    def _initialize_columns(self):
        """Initialize minimum number of columns."""
        for i in range(self.min_columns):
            column_id = f"column_{i:04d}"
            column = CorticalColumn(
                column_id=column_id,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                device=self.device
            )
            self.columns[column_id] = column
            self.active_columns.append(column_id)
        
        self.swarm_metrics['total_columns'] = len(self.columns)
        self.swarm_metrics['active_columns'] = len(self.active_columns)
        
        self.logger.info(f"Initialized swarm with {self.min_columns} columns")
    
    def add_column(self, specialized_for: str = None) -> str:
        """Add a new column to the swarm."""
        if len(self.columns) >= self.max_columns:
            self.logger.warning("Maximum columns reached, cannot add more")
            return None
        
        column_id = f"column_{len(self.columns):04d}"
        
        # Create specialized column if requested
        if specialized_for and specialized_for in self.column_assignments:
            # Use similar parameters to existing columns for this task
            existing_columns = self.column_assignments[specialized_for]
            if existing_columns:
                reference_column = self.columns[existing_columns[0]]
                column = CorticalColumn(
                    column_id=column_id,
                    input_dim=reference_column.input_dim,
                    hidden_dim=reference_column.hidden_dim,
                    output_dim=reference_column.output_dim,
                    sparsity=reference_column.sparsity,
                    learning_rate=reference_column.learning_rate,
                    device=self.device
                )
            else:
                column = CorticalColumn(
                    column_id=column_id,
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    device=self.device
                )
        else:
            column = CorticalColumn(
                column_id=column_id,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                device=self.device
            )
        
        self.columns[column_id] = column
        self.active_columns.append(column_id)
        
        # Assign to task if specified
        if specialized_for:
            if specialized_for not in self.column_assignments:
                self.column_assignments[specialized_for] = []
            self.column_assignments[specialized_for].append(column_id)
        
        self.swarm_metrics['total_columns'] = len(self.columns)
        self.swarm_metrics['active_columns'] = len(self.active_columns)
        
        self.logger.info(f"Added column {column_id}" + 
                        (f" specialized for {specialized_for}" if specialized_for else ""))
        
        return column_id
    
    def remove_column(self, column_id: str) -> bool:
        """Remove a column from the swarm."""
        if column_id not in self.columns:
            return False
        
        if len(self.active_columns) <= self.min_columns:
            self.logger.warning("Cannot remove column, minimum columns required")
            return False
        
        # Remove from assignments
        for task, assigned_columns in self.column_assignments.items():
            if column_id in assigned_columns:
                assigned_columns.remove(column_id)
        
        # Remove column
        del self.columns[column_id]
        if column_id in self.active_columns:
            self.active_columns.remove(column_id)
        
        self.swarm_metrics['total_columns'] = len(self.columns)
        self.swarm_metrics['active_columns'] = len(self.active_columns)
        
        self.logger.info(f"Removed column {column_id}")
        return True
    
    def process_batch(self, 
                     input_batch: torch.Tensor, 
                     task_id: str = "default",
                     collaborative: bool = True) -> Dict[str, torch.Tensor]:
        """Process a batch of inputs using the swarm."""
        # Select columns for this task
        if task_id in self.column_assignments and self.column_assignments[task_id]:
            selected_columns = self.column_assignments[task_id]
        else:
            # Use all active columns
            selected_columns = self.active_columns.copy()
        
        # Distribute batch across columns
        batch_size = input_batch.shape[0]
        columns_per_sample = min(len(selected_columns), max(1, len(selected_columns) // 2))
        
        results = {
            'predictions': [],
            'encodings': [],
            'reconstructions': [],
            'column_assignments': []
        }
        
        for i in range(batch_size):
            sample = input_batch[i:i+1]
            
            if collaborative:
                # Use multiple columns collaboratively
                sample_columns = random.sample(selected_columns, columns_per_sample)
                
                sample_predictions = []
                sample_encodings = []
                sample_reconstructions = []
                
                for col_id in sample_columns:
                    column = self.columns[col_id]
                    column.state = ColumnState.SWARMING
                    
                    with torch.no_grad():
                        outputs = column.forward(sample)
                        sample_predictions.append(outputs['prediction'])
                        sample_encodings.append(outputs['encoded'])
                        sample_reconstructions.append(outputs['reconstruction'])
                
                # Aggregate results
                if sample_predictions:
                    avg_prediction = torch.stack(sample_predictions).mean(dim=0)
                    avg_encoding = torch.stack(sample_encodings).mean(dim=0)
                    avg_reconstruction = torch.stack(sample_reconstructions).mean(dim=0)
                    
                    results['predictions'].append(avg_prediction)
                    results['encodings'].append(avg_encoding)
                    results['reconstructions'].append(avg_reconstruction)
                    results['column_assignments'].append(sample_columns)
            else:
                # Use single best column
                best_column_id = self._select_best_column(sample, selected_columns)
                column = self.columns[best_column_id]
                
                with torch.no_grad():
                    outputs = column.forward(sample)
                    results['predictions'].append(outputs['prediction'])
                    results['encodings'].append(outputs['encoded'])
                    results['reconstructions'].append(outputs['reconstruction'])
                    results['column_assignments'].append([best_column_id])
        
        # Stack results
        if results['predictions']:
            results['predictions'] = torch.cat(results['predictions'], dim=0)
            results['encodings'] = torch.cat(results['encodings'], dim=0)
            results['reconstructions'] = torch.cat(results['reconstructions'], dim=0)
        
        return results
    
    def _select_best_column(self, sample: torch.Tensor, candidate_columns: List[str]) -> str:
        """Select the best column for processing a sample."""
        best_column = candidate_columns[0]
        best_score = -float('inf')
        
        for col_id in candidate_columns:
            column = self.columns[col_id]
            
            # Score based on metrics and recent performance
            score = (column.metrics.prediction_accuracy * 0.4 +
                    column.metrics.pattern_diversity * 0.3 +
                    (1.0 - column.metrics.energy_consumption / 100.0) * 0.3)
            
            if score > best_score:
                best_score = score
                best_column = col_id
        
        return best_column
    
    def train_swarm(self, 
                   input_batch: torch.Tensor, 
                   target_batch: torch.Tensor = None,
                   task_id: str = "default") -> Dict[str, float]:
        """Train the swarm on a batch of data."""
        # Ensure task assignment exists
        if task_id not in self.column_assignments:
            self.column_assignments[task_id] = self.active_columns.copy()
        
        selected_columns = self.column_assignments[task_id]
        batch_size = input_batch.shape[0]
        
        # Distribute training across columns
        losses = {}
        
        for i, col_id in enumerate(selected_columns):
            column = self.columns[col_id]
            
            # Assign samples to this column
            start_idx = (i * batch_size) // len(selected_columns)
            end_idx = ((i + 1) * batch_size) // len(selected_columns)
            
            if start_idx < end_idx:
                sample_batch = input_batch[start_idx:end_idx]
                target_sample = target_batch[start_idx:end_idx] if target_batch is not None else None
                
                # Train column
                total_loss = 0.0
                for j in range(sample_batch.shape[0]):
                    sample = sample_batch[j:j+1]
                    target = target_sample[j:j+1] if target_sample is not None else None
                    
                    loss = column.learn_pattern(sample, target)
                    total_loss += loss
                
                losses[col_id] = total_loss / sample_batch.shape[0]
        
        # Update swarm metrics
        self._update_swarm_metrics()
        
        return losses
    
    def _update_swarm_metrics(self):
        """Update swarm-level metrics."""
        if not self.active_columns:
            return
        
        # Calculate average accuracy
        accuracies = [self.columns[col_id].metrics.prediction_accuracy 
                     for col_id in self.active_columns]
        self.swarm_metrics['average_accuracy'] = sum(accuracies) / len(accuracies)
        
        # Calculate load balance (how evenly distributed the work is)
        activations = [self.columns[col_id].metrics.activation_count 
                      for col_id in self.active_columns]
        if activations:
            mean_activation = sum(activations) / len(activations)
            variance = sum((a - mean_activation) ** 2 for a in activations) / len(activations)
            self.swarm_metrics['load_balance'] = 1.0 / (1.0 + variance / (mean_activation + 1e-6))
        
        # Calculate collaboration efficiency
        collaboration_scores = [self.columns[col_id].metrics.collaboration_score 
                               for col_id in self.active_columns]
        self.swarm_metrics['collaboration_efficiency'] = sum(collaboration_scores) / len(collaboration_scores)
    
    def optimize_swarm(self):
        """Optimize swarm configuration based on performance."""
        # Remove underperforming columns
        if len(self.active_columns) > self.min_columns:
            worst_columns = sorted(
                self.active_columns,
                key=lambda col_id: self.columns[col_id].metrics.prediction_accuracy
            )
            
            # Remove bottom 10% if they're significantly underperforming
            num_to_remove = max(0, min(
                len(self.active_columns) - self.min_columns,
                len(self.active_columns) // 10
            ))
            
            for col_id in worst_columns[:num_to_remove]:
                if self.columns[col_id].metrics.prediction_accuracy < 0.3:
                    self.remove_column(col_id)
        
        # Add columns for overloaded tasks
        for task_id, assigned_columns in self.column_assignments.items():
            if assigned_columns:
                avg_load = sum(self.columns[col_id].metrics.activation_count 
                              for col_id in assigned_columns) / len(assigned_columns)
                
                if avg_load > 1000 and len(self.columns) < self.max_columns:
                    self.add_column(specialized_for=task_id)
    
    def get_swarm_state(self) -> Dict[str, Any]:
        """Get current swarm state."""
        return {
            'metrics': self.swarm_metrics,
            'total_columns': len(self.columns),
            'active_columns': len(self.active_columns),
            'column_states': {
                col_id: {
                    'state': column.state.value,
                    'metrics': {
                        'activation_count': column.metrics.activation_count,
                        'prediction_accuracy': column.metrics.prediction_accuracy,
                        'energy_consumption': column.metrics.energy_consumption,
                        'pattern_diversity': column.metrics.pattern_diversity
                    }
                }
                for col_id, column in self.columns.items()
            },
            'task_assignments': self.column_assignments
        }

# Convenience functions
def create_column_swarm(max_columns: int = 100, **kwargs) -> ColumnSwarm:
    """Create a cortical column swarm."""
    return ColumnSwarm(max_columns=max_columns, **kwargs)

def create_cortical_column(column_id: str, **kwargs) -> CorticalColumn:
    """Create a single cortical column."""
    return CorticalColumn(column_id=column_id, **kwargs)