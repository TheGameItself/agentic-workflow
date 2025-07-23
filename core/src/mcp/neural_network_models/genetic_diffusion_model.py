#!/usr/bin/env python3
"""
Genetic Diffusion Model for MCP Core System
Combines genetic algorithms with diffusion models for enhanced generative capabilities.
"""

import logging
import os
import time
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import diffusion model
from .diffusion_model import DiffusionModel, DiffusionConfig

class GeneticConfig:
    """Configuration for genetic algorithm."""
    
    def __init__(self,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 5,
                 elitism: int = 2,
                 generations: int = 100,
                 fitness_threshold: float = 0.95,
                 log_level: str = "INFO"):
        """Initialize genetic configuration."""
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.generations = generations
        self.fitness_threshold = fitness_threshold
        self.log_level = log_level

class GeneticDiffusionModel:
    """
    Genetic Diffusion Model for MCP Core System.
    
    Combines genetic algorithms with diffusion models to create
    a powerful generative system that can evolve and optimize
    embeddings and other continuous representations.
    """
    
    def __init__(self, diffusion_config: Optional[DiffusionConfig] = None,
                genetic_config: Optional[GeneticConfig] = None):
        """Initialize genetic diffusion model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GeneticDiffusionModel")
        
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.genetic_config = genetic_config or GeneticConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.genetic_config.log_level))
        
        # Create output directory
        os.makedirs(self.diffusion_config.output_dir, exist_ok=True)
        
        # Initialize diffusion model
        self.diffusion_model = DiffusionModel(self.diffusion_config)
        
        # Setup device
        self.device = torch.device(self.diffusion_config.device)
        
        # Genetic algorithm state
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation = 0
        
        # Evolution progress
        self.evolution_progress = {
            'status': 'idle',
            'current_generation': 0,
            'total_generations': self.genetic_config.generations,
            'best_fitness': -float('inf'),
            'average_fitness': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def initialize_population(self, seed_embeddings: Optional[torch.Tensor] = None):
        """
        Initialize the genetic population.
        
        Args:
            seed_embeddings: Optional tensor of embeddings to seed the population
        """
        population_size = self.genetic_config.population_size
        input_dim = self.diffusion_config.input_dim
        
        # Create population
        population = []
        
        # Use seed embeddings if provided
        if seed_embeddings is not None:
            if not isinstance(seed_embeddings, torch.Tensor):
                seed_embeddings = torch.tensor(seed_embeddings, dtype=torch.float32)
            
            seed_embeddings = seed_embeddings.to(self.device)
            
            # Use seed embeddings for part of the population
            num_seeds = min(seed_embeddings.shape[0], population_size // 2)
            for i in range(num_seeds):
                population.append(seed_embeddings[i].clone())
            
            # Generate variations of seed embeddings
            for i in range(num_seeds, population_size):
                # Select a random seed embedding
                seed_idx = random.randint(0, seed_embeddings.shape[0] - 1)
                seed = seed_embeddings[seed_idx].clone()
                
                # Add random noise
                noise = torch.randn_like(seed) * 0.1
                population.append(seed + noise)
        
        # Fill remaining population with random embeddings
        while len(population) < population_size:
            population.append(torch.randn(input_dim, device=self.device))
        
        self.population = population
        self.generation = 0
        
        self.logger.info(f"Initialized population with {len(population)} individuals")
    
    def evolve(self, fitness_function: Callable[[torch.Tensor], float], 
              generations: Optional[int] = None, 
              fitness_threshold: Optional[float] = None):
        """
        Evolve the population using genetic algorithms.
        
        Args:
            fitness_function: Function that evaluates fitness of an individual
            generations: Number of generations to evolve (overrides config)
            fitness_threshold: Fitness threshold to stop evolution (overrides config)
        """
        if generations is None:
            generations = self.genetic_config.generations
        
        if fitness_threshold is None:
            fitness_threshold = self.genetic_config.fitness_threshold
        
        if not self.population:
            self.logger.warning("Population not initialized, initializing with random embeddings")
            self.initialize_population()
        
        # Update evolution progress
        self.evolution_progress = {
            'status': 'evolving',
            'current_generation': self.generation,
            'total_generations': self.generation + generations,
            'best_fitness': self.best_fitness,
            'average_fitness': 0.0,
            'start_time': time.time(),
            'end_time': None
        }
        
        # Evolution loop
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in self.population:
                fitness = fitness_function(individual)
                fitness_scores.append(fitness)
            
            # Find best individual
            best_idx = np.argmax(fitness_scores)
            current_best = self.population[best_idx]
            current_best_fitness = fitness_scores[best_idx]
            
            # Update best overall
            if current_best_fitness > self.best_fitness:
                self.best_individual = current_best.clone()
                self.best_fitness = current_best_fitness
            
            # Calculate average fitness
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            
            # Update progress
            self.generation += 1
            self.evolution_progress['current_generation'] = self.generation
            self.evolution_progress['best_fitness'] = self.best_fitness
            self.evolution_progress['average_fitness'] = average_fitness
            
            # Log progress
            self.logger.info(f"Generation {self.generation}/{self.evolution_progress['total_generations']}, "
                           f"Best Fitness: {self.best_fitness:.4f}, "
                           f"Average Fitness: {average_fitness:.4f}")
            
            # Check termination condition
            if self.best_fitness >= fitness_threshold:
                self.logger.info(f"Reached fitness threshold {fitness_threshold}")
                break
            
            # Create next generation
            next_population = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            for i in range(self.genetic_config.elitism):
                if i < len(sorted_indices):
                    next_population.append(self.population[sorted_indices[i]].clone())
            
            # Fill rest of population with crossover and mutation
            while len(next_population) < len(self.population):
                # Selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Crossover
                if random.random() < self.genetic_config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add to next generation
                next_population.append(child1)
                if len(next_population) < len(self.population):
                    next_population.append(child2)
            
            # Update population
            self.population = next_population
        
        # Update evolution progress
        self.evolution_progress['status'] = 'completed'
        self.evolution_progress['end_time'] = time.time()
        
        self.logger.info(f"Evolution completed after {self.generation} generations")
        self.logger.info(f"Best fitness: {self.best_fitness:.4f}")
        
        return self.best_individual, self.best_fitness
    
    def _tournament_selection(self, fitness_scores: List[float]) -> torch.Tensor:
        """Tournament selection for genetic algorithm."""
        tournament_size = min(self.genetic_config.tournament_size, len(self.population))
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]
    
    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform crossover between two parents."""
        # Interpolation crossover
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def _mutate(self, individual: torch.Tensor) -> torch.Tensor:
        """Mutate an individual."""
        mutation_mask = torch.rand_like(individual) < self.genetic_config.mutation_rate
        mutation = torch.randn_like(individual) * 0.1
        return individual + mutation_mask * mutation
    
    def train_diffusion_model(self, embeddings: Optional[torch.Tensor] = None):
        """
        Train the diffusion model on the current population or provided embeddings.
        
        Args:
            embeddings: Optional tensor of embeddings to train on
        """
        if embeddings is None:
            if not self.population:
                self.logger.warning("Population not initialized, cannot train diffusion model")
                return False
            
            # Use population as training data
            embeddings = torch.stack(self.population)
        
        # Train diffusion model
        self.logger.info("Training diffusion model on population")
        model_path = self.diffusion_model.train(embeddings)
        
        return model_path
    
    def generate_with_diffusion(self, batch_size: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate samples using the diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of generated samples
        """
        if not self.diffusion_model.is_trained:
            self.logger.warning("Diffusion model is not trained, training on current population")
            self.train_diffusion_model()
        
        return self.diffusion_model.sample(batch_size, seed)
    
    def enhance_population_with_diffusion(self, ratio: float = 0.2):
        """
        Enhance the genetic population with diffusion model samples.
        
        Args:
            ratio: Ratio of population to replace with diffusion samples
        """
        if not self.population:
            self.logger.warning("Population not initialized, cannot enhance")
            return False
        
        if not self.diffusion_model.is_trained:
            self.logger.info("Training diffusion model on current population")
            self.train_diffusion_model()
        
        # Number of individuals to replace
        num_replace = int(len(self.population) * ratio)
        if num_replace < 1:
            return False
        
        # Generate new individuals
        new_individuals = self.diffusion_model.sample(num_replace)
        
        # Replace worst individuals
        if hasattr(self, 'last_fitness_scores') and len(self.last_fitness_scores) == len(self.population):
            # Replace worst individuals based on fitness
            sorted_indices = np.argsort(self.last_fitness_scores)
            for i in range(num_replace):
                if i < len(sorted_indices):
                    self.population[sorted_indices[i]] = new_individuals[i]
        else:
            # Replace random individuals
            replace_indices = random.sample(range(len(self.population)), num_replace)
            for i, idx in enumerate(replace_indices):
                self.population[idx] = new_individuals[i]
        
        self.logger.info(f"Enhanced population with {num_replace} diffusion-generated individuals")
        return True
    
    def save_model(self, name: Optional[str] = None) -> str:
        """
        Save the genetic diffusion model.
        
        Args:
            name: Optional name for the model directory
            
        Returns:
            Path to the saved model directory
        """
        if name is None:
            name = f"genetic-{int(time.time())}"
        
        model_dir = os.path.join(self.diffusion_config.output_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save diffusion model
        diffusion_dir = os.path.join(model_dir, "diffusion")
        os.makedirs(diffusion_dir, exist_ok=True)
        
        if self.diffusion_model.is_trained:
            self.diffusion_model._save_checkpoint(diffusion_dir)
        
        # Save genetic config
        genetic_config_path = os.path.join(model_dir, "genetic_config.json")
        with open(genetic_config_path, "w") as f:
            json.dump(vars(self.genetic_config), f, indent=2)
        
        # Save best individual if available
        if self.best_individual is not None:
            best_path = os.path.join(model_dir, "best_individual.pt")
            torch.save(self.best_individual, best_path)
        
        # Save population
        if self.population:
            population_path = os.path.join(model_dir, "population.pt")
            population_tensor = torch.stack(self.population)
            torch.save(population_tensor, population_path)
        
        self.logger.info(f"Saved genetic diffusion model to {model_dir}")
        return model_dir
    
    def load_model(self, path: str) -> bool:
        """
        Load a saved genetic diffusion model.
        
        Args:
            path: Path to the model directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load diffusion model
            diffusion_dir = os.path.join(path, "diffusion")
            if os.path.exists(diffusion_dir):
                self.diffusion_model.load_checkpoint(diffusion_dir)
            
            # Load genetic config
            genetic_config_path = os.path.join(path, "genetic_config.json")
            if os.path.exists(genetic_config_path):
                with open(genetic_config_path, "r") as f:
                    genetic_config_dict = json.load(f)
                    for key, value in genetic_config_dict.items():
                        setattr(self.genetic_config, key, value)
            
            # Load best individual
            best_path = os.path.join(path, "best_individual.pt")
            if os.path.exists(best_path):
                self.best_individual = torch.load(best_path, map_location=self.device)
                # Estimate fitness as we don't have the fitness function
                self.best_fitness = 0.0
            
            # Load population
            population_path = os.path.join(path, "population.pt")
            if os.path.exists(population_path):
                population_tensor = torch.load(population_path, map_location=self.device)
                self.population = [population_tensor[i] for i in range(population_tensor.shape[0])]
                self.generation = 1  # Assume at least one generation
            
            self.logger.info(f"Loaded genetic diffusion model from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading genetic diffusion model: {e}")
            return False
    
    def get_evolution_progress(self) -> Dict[str, Any]:
        """Get current evolution progress."""
        progress = self.evolution_progress.copy()
        
        # Add elapsed time if evolution is in progress
        if progress['status'] == 'evolving' and progress['start_time']:
            progress['elapsed_time'] = time.time() - progress['start_time']
        elif progress['end_time'] and progress['start_time']:
            progress['elapsed_time'] = progress['end_time'] - progress['start_time']
        
        return progress
    
    def get_diffusion_progress(self) -> Dict[str, Any]:
        """Get current diffusion model training progress."""
        return self.diffusion_model.get_training_progress()

# Convenience functions

def create_genetic_diffusion_model(diffusion_config: Optional[DiffusionConfig] = None,
                                 genetic_config: Optional[GeneticConfig] = None) -> GeneticDiffusionModel:
    """Create a genetic diffusion model with custom configuration."""
    return GeneticDiffusionModel(
        diffusion_config or DiffusionConfig(),
        genetic_config or GeneticConfig()
    )

def get_default_genetic_diffusion_model() -> GeneticDiffusionModel:
    """Get a default genetic diffusion model."""
    return GeneticDiffusionModel()

def load_genetic_diffusion_model(path: str) -> Optional[GeneticDiffusionModel]:
    """Load a pretrained genetic diffusion model."""
    try:
        # Load diffusion config if available
        diffusion_config_path = os.path.join(path, "diffusion", "config.json")
        if os.path.exists(diffusion_config_path):
            with open(diffusion_config_path, "r") as f:
                diffusion_config_dict = json.load(f)
                diffusion_config = DiffusionConfig(**diffusion_config_dict)
        else:
            diffusion_config = DiffusionConfig()
        
        # Load genetic config if available
        genetic_config_path = os.path.join(path, "genetic_config.json")
        if os.path.exists(genetic_config_path):
            with open(genetic_config_path, "r") as f:
                genetic_config_dict = json.load(f)
                genetic_config = GeneticConfig(**genetic_config_dict)
        else:
            genetic_config = GeneticConfig()
        
        # Create model and load checkpoint
        model = GeneticDiffusionModel(diffusion_config, genetic_config)
        success = model.load_model(path)
        
        if success:
            return model
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error loading genetic diffusion model: {e}")
        return None