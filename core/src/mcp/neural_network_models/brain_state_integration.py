#!/usr/bin/env python3
"""
Brain State Integration for MCP Core System
Integrates various neural models to create a unified brain state.
"""

import logging
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import neural models
from .hormone_neural_integration import HormoneNeuralIntegration, HormoneConfig
from .diffusion_model import DiffusionModel, DiffusionConfig
from .genetic_diffusion_model import GeneticDiffusionModel, GeneticConfig

class BrainStateConfig:
    """Configuration for brain state integration."""
    
    def __init__(self,
                 embedding_dim: int = 128,
                 state_dim: int = 256,
                 hormone_integration: bool = True,
                 diffusion_model: bool = True,
                 genetic_model: bool = False,
                 device: str = None,
                 output_dir: str = "data/models",
                 log_level: str = "INFO"):
        """Initialize brain state configuration."""
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.hormone_integration = hormone_integration
        self.diffusion_model = diffusion_model
        self.genetic_model = genetic_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.log_level = log_level

class BrainStateNetwork(nn.Module):
    """Neural network for brain state integration."""
    
    def __init__(self, embedding_dim: int, state_dim: int, hormone_dim: int = 4):
        """Initialize brain state network."""
        super(BrainStateNetwork, self).__init__()
        
        # Embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, state_dim),
            nn.ReLU()
        )
        
        # Hormone projection
        self.hormone_projection = nn.Sequential(
            nn.Linear(hormone_dim, state_dim),
            nn.ReLU()
        )
        
        # State integration
        self.state_integration = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.Tanh()  # State values between -1 and 1
        )
    
    def forward(self, embedding, hormones):
        """Forward pass."""
        embedding_proj = self.embedding_projection(embedding)
        hormone_proj = self.hormone_projection(hormones)
        
        # Concatenate projections
        combined = torch.cat([embedding_proj, hormone_proj], dim=1)
        
        # Integrate into unified state
        state = self.state_integration(combined)
        
        return state

class BrainStateIntegration:
    """
    Brain State Integration for MCP Core System.
    
    Integrates various neural models (hormone system, diffusion model,
    genetic model) to create a unified brain state that represents
    the current cognitive state of the system.
    
    This brain state can be used to:
    - Influence decision making
    - Adapt system behavior
    - Generate creative solutions
    - Maintain system coherence
    """
    
    def __init__(self, config: BrainStateConfig = None):
        """Initialize brain state integration."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BrainStateIntegration")
        
        self.config = config or BrainStateConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.hormone_integration = None
        self.diffusion_model = None
        self.genetic_model = None
        
        if self.config.hormone_integration:
            hormone_config = HormoneConfig(
                output_dim=4,
                device=self.config.device,
                output_dir=self.config.output_dir
            )
            self.hormone_integration = HormoneNeuralIntegration(hormone_config)
        
        if self.config.diffusion_model:
            diffusion_config = DiffusionConfig(
                input_dim=self.config.embedding_dim,
                device=self.config.device,
                output_dir=self.config.output_dir
            )
            self.diffusion_model = DiffusionModel(diffusion_config)
        
        if self.config.genetic_model:
            genetic_config = GeneticConfig()
            diffusion_config = DiffusionConfig(
                input_dim=self.config.embedding_dim,
                device=self.config.device,
                output_dir=self.config.output_dir
            )
            self.genetic_model = GeneticDiffusionModel(diffusion_config, genetic_config)
        
        # Initialize brain state network
        self.brain_state_network = BrainStateNetwork(
            embedding_dim=self.config.embedding_dim,
            state_dim=self.config.state_dim,
            hormone_dim=4  # Four hormone types
        )
        self.brain_state_network.to(self.device)
        
        # Current brain state
        self.current_state = torch.zeros(self.config.state_dim, device=self.device)
        
        # Brain state history
        self.state_history = []
        
        # Training state
        self.is_trained = False
    
    def update_brain_state(self, embedding: Optional[torch.Tensor] = None, 
                         metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Update the brain state based on current embedding and metrics.
        
        Args:
            embedding: Current embedding tensor
            metrics: Current system metrics
            
        Returns:
            Updated brain state tensor
        """
        # Default embedding if not provided
        if embedding is None:
            embedding = torch.zeros(self.config.embedding_dim, device=self.device)
        elif not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32, device=self.device)
        
        # Ensure correct shape
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
        
        # Update hormone levels if integration is enabled
        hormone_levels = {
            'stress': 0.0,
            'efficiency': 1.0,
            'adaptation': 0.5,
            'stability': 1.0
        }
        
        if self.hormone_integration and metrics:
            hormone_levels = self.hormone_integration.update_hormone_levels(metrics)
        
        # Convert hormone levels to tensor
        hormone_tensor = torch.tensor([
            hormone_levels['stress'],
            hormone_levels['efficiency'],
            hormone_levels['adaptation'],
            hormone_levels['stability']
        ], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Generate brain state
        with torch.no_grad():
            self.brain_state_network.eval()
            state = self.brain_state_network(embedding, hormone_tensor)
            self.current_state = state.squeeze(0)
        
        # Record state history
        self.state_history.append({
            'timestamp': time.time(),
            'state': self.current_state.cpu().numpy(),
            'hormone_levels': hormone_levels
        })
        
        # Keep history limited
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        return self.current_state
    
    def get_current_state(self) -> torch.Tensor:
        """Get current brain state."""
        return self.current_state
    
    def get_state_embedding(self) -> torch.Tensor:
        """
        Get embedding representation of current brain state.
        
        This can be used as input to other models or for storage.
        """
        # Project brain state to embedding space
        with torch.no_grad():
            # Simple projection (can be replaced with more sophisticated method)
            if self.current_state.shape[0] > self.config.embedding_dim:
                # Reduce dimensionality
                embedding = self.current_state[:self.config.embedding_dim]
            else:
                # Pad with zeros
                embedding = torch.zeros(self.config.embedding_dim, device=self.device)
                embedding[:self.current_state.shape[0]] = self.current_state
        
        return embedding
    
    def generate_creative_embedding(self, seed_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a creative embedding based on current brain state.
        
        Args:
            seed_embedding: Optional seed embedding to start from
            
        Returns:
            Generated embedding tensor
        """
        if self.diffusion_model and self.diffusion_model.is_trained:
            # Use diffusion model to generate creative embedding
            if seed_embedding is not None:
                # Add noise based on adaptation hormone level
                adaptation_level = 0.5
                if self.hormone_integration:
                    adaptation_level = self.hormone_integration.hormone_levels['adaptation']
                
                # Add noise proportional to adaptation level
                noise_scale = adaptation_level * 0.5
                noise = torch.randn_like(seed_embedding) * noise_scale
                noisy_embedding = seed_embedding + noise
                
                # Denoise with diffusion model (simplified)
                with torch.no_grad():
                    # Use diffusion model to sample
                    generated = self.diffusion_model.sample(1)[0]
                    
                    # Interpolate between seed and generated based on adaptation level
                    creative_embedding = (1 - adaptation_level) * noisy_embedding + adaptation_level * generated
            else:
                # Generate completely new embedding
                with torch.no_grad():
                    creative_embedding = self.diffusion_model.sample(1)[0]
            
            return creative_embedding
            
        elif self.genetic_model and len(self.genetic_model.population) > 0:
            # Use genetic model to generate creative embedding
            if seed_embedding is not None:
                # Find closest individual in population
                distances = []
                for individual in self.genetic_model.population:
                    distance = torch.norm(individual - seed_embedding)
                    distances.append(distance.item())
                
                closest_idx = np.argmin(distances)
                closest = self.genetic_model.population[closest_idx]
                
                # Create variation
                adaptation_level = 0.5
                if self.hormone_integration:
                    adaptation_level = self.hormone_integration.hormone_levels['adaptation']
                
                # Add noise proportional to adaptation level
                noise_scale = adaptation_level * 0.5
                noise = torch.randn_like(seed_embedding) * noise_scale
                creative_embedding = closest + noise
            else:
                # Select random individual from population
                idx = random.randint(0, len(self.genetic_model.population) - 1)
                creative_embedding = self.genetic_model.population[idx]
            
            return creative_embedding
        
        else:
            # Fallback: generate random embedding influenced by brain state
            if seed_embedding is not None:
                # Add noise based on current brain state
                state_influence = torch.tanh(self.current_state[:10])  # Use first 10 dimensions
                noise_scale = torch.mean(torch.abs(state_influence)).item() * 0.5
                noise = torch.randn_like(seed_embedding) * noise_scale
                creative_embedding = seed_embedding + noise
            else:
                # Generate random embedding
                creative_embedding = torch.randn(self.config.embedding_dim, device=self.device)
                
                # Scale based on stability hormone
                stability = 1.0
                if self.hormone_integration:
                    stability = self.hormone_integration.hormone_levels['stability']
                
                # More stable = less random
                creative_embedding = creative_embedding * (1.0 - stability * 0.5)
            
            return creative_embedding
    
    def train_brain_state_network(self, embeddings: torch.Tensor, hormone_levels: torch.Tensor, 
                                epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3):
        """
        Train the brain state network.
        
        Args:
            embeddings: Embedding tensors
            hormone_levels: Hormone level tensors
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        if not isinstance(hormone_levels, torch.Tensor):
            hormone_levels = torch.tensor(hormone_levels, dtype=torch.float32)
        
        # Move to device
        embeddings = embeddings.to(self.device)
        hormone_levels = hormone_levels.to(self.device)
        
        # Create dataset
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(embeddings, hormone_levels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.brain_state_network.parameters(), lr=learning_rate)
        
        # Training loop
        self.brain_state_network.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_embeddings, batch_hormones in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                state = self.brain_state_network(batch_embeddings, batch_hormones)
                
                # Compute loss (self-supervised)
                # We want the state to be influenced by both embeddings and hormones
                # but also to be stable and coherent
                
                # Reconstruction loss
                embedding_pred = state @ self.brain_state_network.embedding_projection[0].weight
                hormone_pred = state @ self.brain_state_network.hormone_projection[0].weight
                
                recon_loss = F.mse_loss(embedding_pred, batch_embeddings) + F.mse_loss(hormone_pred, batch_hormones)
                
                # Coherence loss (encourage smoothness)
                coherence_loss = torch.mean(torch.abs(torch.diff(state, dim=1)))
                
                # Total loss
                loss = recon_loss + 0.1 * coherence_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Log progress
            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        self._save_model(f"brain-state-{int(time.time())}")
        
        self.is_trained = True
        self.logger.info("Brain state network training completed")
    
    def _save_model(self, name: str) -> str:
        """Save model checkpoint."""
        model_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.brain_state_network.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        return model_dir
    
    def load_model(self, path: str) -> bool:
        """Load model checkpoint."""
        try:
            # Load model
            model_path = os.path.join(path, "model.pt")
            self.brain_state_network.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Load config
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        setattr(self.config, key, value)
            
            self.is_trained = True
            self.logger.info(f"Loaded brain state network from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading brain state network: {e}")
            return False
    
    def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get brain state history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of brain state history entries
        """
        return self.state_history[-limit:]
    
    def pretrain_all_components(self, embeddings: Optional[torch.Tensor] = None,
                              metrics_history: Optional[List[Dict[str, float]]] = None):
        """
        Pretrain all neural components.
        
        Args:
            embeddings: Embedding tensors for training diffusion and genetic models
            metrics_history: Metrics history for training hormone integration
        """
        # Train hormone integration if available
        if self.hormone_integration and metrics_history:
            self.logger.info("Pretraining hormone integration...")
            inputs, outputs = self.hormone_integration.generate_training_data(metrics_history)
            self.hormone_integration.train(inputs, outputs)
        
        # Train diffusion model if available
        if self.diffusion_model and embeddings is not None:
            self.logger.info("Pretraining diffusion model...")
            self.diffusion_model.train(embeddings)
        
        # Initialize genetic model if available
        if self.genetic_model and embeddings is not None:
            self.logger.info("Initializing genetic model...")
            self.genetic_model.initialize_population(embeddings)
        
        # Train brain state network if other components are trained
        if ((self.hormone_integration and self.hormone_integration.is_trained) or
            (self.diffusion_model and self.diffusion_model.is_trained)):
            
            self.logger.info("Pretraining brain state network...")
            
            # Generate training data
            train_embeddings = []
            train_hormones = []
            
            # Use diffusion model to generate embeddings if available
            if self.diffusion_model and self.diffusion_model.is_trained:
                generated_embeddings = self.diffusion_model.sample(100)
                train_embeddings.append(generated_embeddings)
            
            # Use genetic model population if available
            if self.genetic_model and len(self.genetic_model.population) > 0:
                population_tensor = torch.stack(self.genetic_model.population)
                train_embeddings.append(population_tensor)
            
            # Use provided embeddings
            if embeddings is not None:
                train_embeddings.append(embeddings)
            
            # Combine embeddings
            if train_embeddings:
                combined_embeddings = torch.cat(train_embeddings)
                
                # Generate hormone levels
                hormone_levels = []
                for _ in range(combined_embeddings.shape[0]):
                    # Generate random hormone levels
                    hormone = torch.rand(4, device=self.device)
                    hormone_levels.append(hormone)
                
                combined_hormones = torch.stack(hormone_levels)
                
                # Train brain state network
                self.train_brain_state_network(
                    combined_embeddings, 
                    combined_hormones,
                    epochs=50,
                    batch_size=32
                )
        
        self.logger.info("Pretraining of all components completed")

# Convenience functions

def create_brain_state_integration(config: Optional[BrainStateConfig] = None) -> BrainStateIntegration:
    """Create a brain state integration with custom configuration."""
    return BrainStateIntegration(config or BrainStateConfig())

def get_default_brain_state_integration() -> BrainStateIntegration:
    """Get a default brain state integration."""
    return BrainStateIntegration()

def load_brain_state_integration(path: str) -> Optional[BrainStateIntegration]:
    """Load a pretrained brain state integration."""
    try:
        # Load config if available
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                config = BrainStateConfig(**config_dict)
        else:
            config = BrainStateConfig()
        
        # Create model and load checkpoint
        model = BrainStateIntegration(config)
        success = model.load_model(path)
        
        if success:
            return model
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error loading brain state integration: {e}")
        return None