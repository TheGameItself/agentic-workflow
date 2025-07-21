"""
Advanced Training Engine with Genetic Evolution

Implements sophisticated training algorithms that integrate with the genetic
data exchange system for continuous evolution and P2P learning optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import asyncio
import time
import json
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import math

from .genetic_diffusion_model import GeneticDiffusionModel, NetworkGene
from ..genetic_data_exchange import GeneticDataExchange
from ..hormone_system_integration import HormoneSystem


@dataclass
class TrainingConfiguration:
    """Configuration for genetic training process"""
    # Basic training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-4
    
    # Genetic evolution parameters
    evolution_frequency: int = 10  # Evolve every N epochs
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    selection_pressure: float = 0.7
    
    # Population parameters
    population_size: int = 5
    elite_ratio: float = 0.2
    diversity_threshold: float = 0.1
    
    # Hormone-influenced parameters
    hormone_influence: bool = True
    stress_adaptation: bool = True
    reward_modulation: bool = True
    
    # P2P learning parameters
    p2p_sharing: bool = True
    sharing_frequency: int = 20  # Share every N epochs
    fitness_threshold: float = 0.8
    
    # Advanced optimization
    adaptive_learning_rate: bool = True
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 20
    
    # Diffusion training parameters
    diffusion_training: bool = False
    diffusion_weight: float = 0.1


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    
    # Genetic metrics
    generation: int
    population_diversity: float
    best_fitness: float
    mutation_count: int
    crossover_count: int
    
    # Hormone metrics
    dopamine_level: float = 0.0
    serotonin_level: float = 0.0
    cortisol_level: float = 0.0
    
    # Performance metrics
    training_time: float = 0.0
    memory_usage: float = 0.0
    gradient_norm: float = 0.0
    
    # P2P metrics
    shared_improvements: int = 0
    received_improvements: int = 0
    network_fitness: float = 0.0


class GeneticPopulation:
    """Manages population of genetic neural networks"""
    
    def __init__(self, config: TrainingConfiguration, input_dim: int, output_dim: int,
                 genetic_exchange: Optional[GeneticDataExchange] = None):
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.genetic_exchange = genetic_exchange
        
        # Initialize population
        self.population: List[GeneticDiffusionModel] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        
        # Performance tracking
        self.best_individual = None
        self.best_fitness = 0.0
        self.diversity_history = []
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with diverse individuals"""
        for i in range(self.config.population_size):
            model = GeneticDiffusionModel(self.input_dim, self.output_dim, self.genetic_exchange)
            
            # Add some initial diversity
            if i > 0:
                # Mutate from base architecture
                asyncio.run(model.evolve_architecture())
            
            self.population.append(model)
            self.fitness_scores.append(0.0)
    
    def evaluate_fitness(self, model: GeneticDiffusionModel, 
                        train_loader: DataLoader, val_loader: DataLoader,
                        criterion: nn.Module) -> float:
        """Evaluate fitness of an individual model"""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        # Fitness combines accuracy and efficiency
        parameter_count = sum(p.numel() for p in model.parameters())
        efficiency_factor = 1.0 / (1.0 + parameter_count / 1000000)  # Prefer smaller models
        
        fitness = accuracy * 0.8 + efficiency_factor * 0.2
        
        # Add genetic diversity bonus
        diversity_bonus = self._calculate_diversity_bonus(model)
        fitness += diversity_bonus * 0.1
        
        return max(0.0, min(1.0, fitness))
    
    def _calculate_diversity_bonus(self, model: GeneticDiffusionModel) -> float:
        """Calculate diversity bonus for maintaining population diversity"""
        if len(self.population) <= 1:
            return 0.0
        
        # Compare genetic similarity with other individuals
        similarities = []
        model_genes = [gene.gene_id for gene in model.genes]
        
        for other_model in self.population:
            if other_model is model:
                continue
            
            other_genes = [gene.gene_id for gene in other_model.genes]
            
            # Calculate Jaccard similarity
            intersection = len(set(model_genes) & set(other_genes))
            union = len(set(model_genes) | set(other_genes))
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        # Diversity bonus is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities)
        diversity_bonus = 1.0 - avg_similarity
        
        return diversity_bonus
    
    def selection(self) -> List[GeneticDiffusionModel]:
        """Select parents for reproduction using tournament selection"""
        parents = []
        tournament_size = max(2, int(self.config.population_size * 0.3))
        
        for _ in range(self.config.population_size):
            # Tournament selection
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            
            # Select best from tournament
            best_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[best_idx])
        
        return parents
    
    def crossover(self, parent1: GeneticDiffusionModel, 
                  parent2: GeneticDiffusionModel) -> GeneticDiffusionModel:
        """Perform genetic crossover between two parents"""
        if random.random() < self.config.crossover_rate:
            offspring = parent1.crossover_with_model(parent2)
            return offspring
        else:
            # Return copy of better parent
            if self.fitness_scores[self.population.index(parent1)] > \
               self.fitness_scores[self.population.index(parent2)]:
                return parent1
            else:
                return parent2
    
    def mutation(self, individual: GeneticDiffusionModel) -> GeneticDiffusionModel:
        """Apply mutations to an individual"""
        if random.random() < self.config.mutation_rate:
            # Evolve architecture
            asyncio.run(individual.evolve_architecture())
        
        return individual
    
    async def evolve_generation(self, train_loader: DataLoader, val_loader: DataLoader,
                              criterion: nn.Module) -> Dict[str, float]:
        """Evolve population for one generation"""
        # Evaluate current population
        for i, model in enumerate(self.population):
            fitness = self.evaluate_fitness(model, train_loader, val_loader, criterion)
            self.fitness_scores[i] = fitness
        
        # Track best individual
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_individual = self.population[best_idx]
        
        # Calculate population diversity
        diversity = self._calculate_population_diversity()
        self.diversity_history.append(diversity)
        
        # Selection
        parents = self.selection()
        
        # Create new generation
        new_population = []
        crossover_count = 0
        mutation_count = 0
        
        # Keep elite individuals
        elite_count = max(1, int(self.config.population_size * self.config.elite_ratio))
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            offspring = self.crossover(parent1, parent2)
            if offspring != parent1 and offspring != parent2:
                crossover_count += 1
            
            offspring = self.mutation(offspring)
            if random.random() < self.config.mutation_rate:
                mutation_count += 1
            
            new_population.append(offspring)
        
        # Replace population
        self.population = new_population[:self.config.population_size]
        self.fitness_scores = [0.0] * len(self.population)
        self.generation += 1
        
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean([self.evaluate_fitness(model, train_loader, val_loader, criterion) 
                                  for model in self.population]),
            'diversity': diversity,
            'crossover_count': crossover_count,
            'mutation_count': mutation_count
        }
    
    def _calculate_population_diversity(self) -> float:
        """Calculate overall population diversity"""
        if len(self.population) <= 1:
            return 0.0
        
        # Calculate pairwise genetic distances
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._calculate_genetic_distance(self.population[i], self.population[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_genetic_distance(self, model1: GeneticDiffusionModel, 
                                  model2: GeneticDiffusionModel) -> float:
        """Calculate genetic distance between two models"""
        genes1 = [gene.gene_id for gene in model1.genes]
        genes2 = [gene.gene_id for gene in model2.genes]
        
        # Hamming distance normalized by maximum possible distance
        max_len = max(len(genes1), len(genes2))
        if max_len == 0:
            return 0.0
        
        # Pad shorter list
        genes1.extend([''] * (max_len - len(genes1)))
        genes2.extend([''] * (max_len - len(genes2)))
        
        differences = sum(g1 != g2 for g1, g2 in zip(genes1, genes2))
        return differences / max_len
    
    def get_best_model(self) -> GeneticDiffusionModel:
        """Get the best model from current population"""
        if self.best_individual is not None:
            return self.best_individual
        
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]


class AdvancedTrainingEngine:
    """Advanced training engine with genetic evolution and P2P learning"""
    
    def __init__(self, config: TrainingConfiguration, 
                 genetic_exchange: Optional[GeneticDataExchange] = None,
                 hormone_system: Optional[HormoneSystem] = None):
        self.config = config
        self.genetic_exchange = genetic_exchange
        self.hormone_system = hormone_system
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history: List[TrainingMetrics] = []
        
        # Genetic population
        self.population: Optional[GeneticPopulation] = None
        
        # Hormone levels
        self.hormone_levels = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'cortisol': 0.3,
            'adrenaline': 0.2
        }
        
        # P2P learning state
        self.shared_improvements = 0
        self.received_improvements = 0
        
    def initialize_population(self, input_dim: int, output_dim: int):
        """Initialize genetic population"""
        self.population = GeneticPopulation(
            self.config, input_dim, output_dim, self.genetic_exchange
        )
    
    async def train(self, train_loader: DataLoader, val_loader: DataLoader,
                   criterion: nn.Module, device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """Main training loop with genetic evolution"""
        if self.population is None:
            # Infer dimensions from data
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch[0].shape[-1]
            output_dim = len(torch.unique(sample_batch[1]))
            self.initialize_population(input_dim, output_dim)
        
        training_start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Get current best model
            best_model = self.population.get_best_model()
            
            # Train current best model
            train_metrics = await self._train_epoch(best_model, train_loader, criterion, device)
            val_metrics = await self._validate_epoch(best_model, val_loader, criterion, device)
            
            # Update hormone levels based on performance
            if self.config.hormone_influence:
                self._update_hormone_levels(train_metrics, val_metrics)
            
            # Genetic evolution
            if epoch % self.config.evolution_frequency == 0 and epoch > 0:
                evolution_metrics = await self.population.evolve_generation(
                    train_loader, val_loader, criterion
                )
                print(f"Evolution - Gen {evolution_metrics['generation']}: "
                      f"Best fitness: {evolution_metrics['best_fitness']:.4f}, "
                      f"Diversity: {evolution_metrics['diversity']:.4f}")
            
            # P2P sharing
            if (self.config.p2p_sharing and epoch % self.config.sharing_frequency == 0 
                and epoch > 0 and self.genetic_exchange):
                await self._share_improvements(best_model)
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_accuracy=train_metrics['accuracy'],
                val_accuracy=val_metrics['accuracy'],
                learning_rate=train_metrics['learning_rate'],
                generation=self.population.generation,
                population_diversity=self.population._calculate_population_diversity(),
                best_fitness=self.population.best_fitness,
                mutation_count=0,  # Would be tracked in evolution
                crossover_count=0,  # Would be tracked in evolution
                dopamine_level=self.hormone_levels['dopamine'],
                serotonin_level=self.hormone_levels['serotonin'],
                cortisol_level=self.hormone_levels['cortisol'],
                training_time=epoch_time,
                memory_usage=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                gradient_norm=train_metrics.get('gradient_norm', 0.0),
                shared_improvements=self.shared_improvements,
                received_improvements=self.received_improvements
            )
            
            self.training_history.append(metrics)
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress reporting
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Dopamine: {self.hormone_levels['dopamine']:.3f}")
        
        total_training_time = time.time() - training_start_time
        
        return {
            'best_model': self.population.get_best_model(),
            'training_history': self.training_history,
            'total_training_time': total_training_time,
            'final_generation': self.population.generation,
            'best_fitness': self.population.best_fitness,
            'population_diversity': self.population._calculate_population_diversity()
        }
    
    async def _train_epoch(self, model: GeneticDiffusionModel, train_loader: DataLoader,
                          criterion: nn.Module, device: torch.device) -> Dict[str, float]:
        """Train model for one epoch"""
        model.train()
        
        # Hormone-influenced learning rate
        base_lr = self.config.learning_rate
        if self.config.hormone_influence:
            lr_modifier = 1.0 + (self.hormone_levels['dopamine'] - 0.5) * 0.5
            lr_modifier *= 1.0 + (self.hormone_levels['adrenaline'] - 0.2) * 0.3
            learning_rate = base_lr * lr_modifier
        else:
            learning_rate = base_lr
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                              weight_decay=self.config.weight_decay)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_gradient_norm = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Add diffusion loss if enabled
            if self.config.diffusion_training:
                diffusion_loss = model.diffusion_loss(batch_x)
                loss = loss + self.config.diffusion_weight * diffusion_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clipping
                )
                total_gradient_norm += gradient_norm.item()
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_y.size(0)
            total_correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples,
            'learning_rate': learning_rate,
            'gradient_norm': total_gradient_norm / len(train_loader)
        }
    
    async def _validate_epoch(self, model: GeneticDiffusionModel, val_loader: DataLoader,
                             criterion: nn.Module, device: torch.device) -> Dict[str, float]:
        """Validate model for one epoch"""
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _update_hormone_levels(self, train_metrics: Dict[str, float], 
                              val_metrics: Dict[str, float]):
        """Update hormone levels based on training performance"""
        # Dopamine: reward signal based on improvement
        if hasattr(self, 'prev_val_loss'):
            if val_metrics['loss'] < self.prev_val_loss:
                self.hormone_levels['dopamine'] = min(1.0, self.hormone_levels['dopamine'] + 0.1)
            else:
                self.hormone_levels['dopamine'] = max(0.0, self.hormone_levels['dopamine'] - 0.05)
        self.prev_val_loss = val_metrics['loss']
        
        # Serotonin: confidence based on validation accuracy
        self.hormone_levels['serotonin'] = 0.3 + 0.7 * val_metrics['accuracy']
        
        # Cortisol: stress based on loss magnitude and gradient norm
        stress_factor = min(1.0, train_metrics['loss'] + train_metrics.get('gradient_norm', 0.0) / 10.0)
        self.hormone_levels['cortisol'] = 0.1 + 0.8 * stress_factor
        
        # Adrenaline: urgency based on learning rate and performance change
        if hasattr(self, 'prev_train_loss'):
            performance_change = abs(train_metrics['loss'] - self.prev_train_loss)
            self.hormone_levels['adrenaline'] = min(1.0, 0.2 + performance_change * 5.0)
        self.prev_train_loss = train_metrics['loss']
        
        # Apply hormone system integration if available
        if self.hormone_system:
            self.hormone_system.update_hormone_levels(self.hormone_levels)
    
    async def _share_improvements(self, model: GeneticDiffusionModel):
        """Share model improvements with P2P network"""
        if not self.genetic_exchange:
            return
        
        # Check if model is good enough to share
        current_fitness = self.population.evaluate_fitness(
            model, None, None, nn.CrossEntropyLoss()  # Simplified for sharing check
        )
        
        if current_fitness >= self.config.fitness_threshold:
            success = await model.share_genetic_improvements()
            if success:
                self.shared_improvements += 1
                print(f"Shared improvements: fitness {current_fitness:.3f}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.training_history:
            return {}
        
        final_metrics = self.training_history[-1]
        
        return {
            'total_epochs': len(self.training_history),
            'final_train_loss': final_metrics.train_loss,
            'final_val_loss': final_metrics.val_loss,
            'final_train_accuracy': final_metrics.train_accuracy,
            'final_val_accuracy': final_metrics.val_accuracy,
            'best_val_loss': self.best_val_loss,
            'final_generation': final_metrics.generation,
            'final_population_diversity': final_metrics.population_diversity,
            'best_fitness': final_metrics.best_fitness,
            'total_shared_improvements': self.shared_improvements,
            'total_received_improvements': self.received_improvements,
            'final_hormone_levels': {
                'dopamine': final_metrics.dopamine_level,
                'serotonin': final_metrics.serotonin_level,
                'cortisol': final_metrics.cortisol_level
            },
            'average_training_time_per_epoch': np.mean([m.training_time for m in self.training_history]),
            'total_training_time': sum(m.training_time for m in self.training_history)
        }


# Example usage and testing
class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, size: int = 1000, input_dim: int = 784, num_classes: int = 10):
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


async def test_training_engine():
    """Test the advanced training engine"""
    # Create configuration
    config = TrainingConfiguration(
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        evolution_frequency=5,
        population_size=3,
        p2p_sharing=True,
        sharing_frequency=10
    )
    
    # Create genetic exchange system
    genetic_exchange = GeneticDataExchange("training_test_organism")
    
    # Create training engine
    engine = AdvancedTrainingEngine(config, genetic_exchange)
    
    # Create datasets
    train_dataset = SimpleDataset(1000, 784, 10)
    val_dataset = SimpleDataset(200, 784, 10)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("Starting genetic training...")
    results = await engine.train(train_loader, val_loader, criterion)
    
    # Print results
    summary = engine.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nBest model generation: {results['final_generation']}")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Population diversity: {results['population_diversity']:.4f}")


if __name__ == "__main__":
    asyncio.run(test_training_engine())