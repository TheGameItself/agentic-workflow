#!/usr/bin/env python3
"""
Diffusion Model for MCP Core System
Implements diffusion-based generative models for the MCP architecture.
"""

import logging
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class DiffusionConfig:
    """Configuration for diffusion model."""
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 batch_size: int = 32,
                 learning_rate: float = 2e-4,
                 epochs: int = 100,
                 device: str = None,
                 output_dir: str = "data/models",
                 save_steps: int = 500,
                 log_level: str = "INFO"):
        """Initialize diffusion configuration."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.log_level = log_level

class SimpleUNet(nn.Module):
    """Simple U-Net model for diffusion."""
    
    def __init__(self, input_dim: int, hidden_dim: int, timesteps: int):
        """Initialize U-Net model."""
        super(SimpleUNet, self).__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU()
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU()
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        """Forward pass."""
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Encoder
        e1 = self.encoder1(x)
        e1 = e1 + t_emb  # Add time embedding
        e2 = self.encoder2(e1)
        
        # Middle
        m = self.middle(e2)
        
        # Decoder with skip connections
        d1 = self.decoder1(torch.cat([m, e2], dim=1))
        d2 = self.decoder2(torch.cat([d1, e1], dim=1))
        
        return d2

class DiffusionModel:
    """
    Diffusion Model for MCP Core System.
    
    Implements a diffusion-based generative model for creating
    and manipulating embeddings and other continuous representations.
    """
    
    def __init__(self, config: DiffusionConfig = None):
        """Initialize diffusion model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DiffusionModel")
        
        self.config = config or DiffusionConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = SimpleUNet(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            timesteps=self.config.timesteps
        )
        
        # Setup device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Setup diffusion parameters
        self.betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.timesteps,
            device=self.device
        )
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Training state
        self.is_trained = False
        self.training_progress = {
            'status': 'idle',
            'current_epoch': 0,
            'total_epochs': self.config.epochs,
            'current_step': 0,
            'total_steps': 0,
            'loss': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """Compute loss for denoising process."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def train(self, embeddings: torch.Tensor, epochs: int = None, batch_size: int = None):
        """
        Train the diffusion model on embeddings.
        
        Args:
            embeddings: Tensor of embeddings to train on
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size for training (overrides config)
        """
        if epochs is None:
            epochs = self.config.epochs
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Convert embeddings to tensor if needed
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        # Move to device
        embeddings = embeddings.to(self.device)
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Update training state
        self.training_progress = {
            'status': 'training',
            'current_epoch': 0,
            'total_epochs': epochs,
            'current_step': 0,
            'total_steps': len(dataloader) * epochs,
            'loss': 0.0,
            'start_time': time.time(),
            'end_time': None
        }
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, (batch,) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Sample random timesteps
                t = torch.randint(0, self.config.timesteps, (batch.shape[0],), device=self.device).long()
                
                # Compute loss
                loss = self.p_losses(batch, t)
                
                # Update
                loss.backward()
                optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                self.training_progress['current_step'] = epoch * len(dataloader) + step
                self.training_progress['loss'] = loss.item()
                
                # Log progress
                if step % 10 == 0:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(dataloader)}, Loss: {loss.item():.6f}")
                
                # Save checkpoint
                if (epoch * len(dataloader) + step) % self.config.save_steps == 0:
                    self._save_checkpoint(f"diffusion-{int(time.time())}")
            
            # Update epoch progress
            self.training_progress['current_epoch'] = epoch + 1
            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save final model
        model_path = self._save_checkpoint(f"diffusion-{int(time.time())}")
        
        # Update training state
        self.training_progress['status'] = 'completed'
        self.training_progress['end_time'] = time.time()
        self.is_trained = True
        
        self.logger.info(f"Diffusion model training completed and saved to {model_path}")
        return model_path
    
    def _save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        model_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        return model_dir
    
    def load_checkpoint(self, path: str) -> bool:
        """Load model checkpoint."""
        try:
            # Load model
            model_path = os.path.join(path, "model.pt")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Load config
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        setattr(self.config, key, value)
            
            self.is_trained = True
            self.logger.info(f"Loaded diffusion model from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading diffusion model: {e}")
            return False
    
    @torch.no_grad()
    def sample(self, batch_size: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate samples from the diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of generated samples
        """
        if not self.is_trained:
            self.logger.warning("Model is not trained, samples may be random noise")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(batch_size, self.config.input_dim, device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.config.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t_tensor)
            
            # Get alpha and beta for this timestep
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # No noise for t=0
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Update x
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        return x
    
    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                   num_steps: int = 10, seed: Optional[int] = None) -> List[torch.Tensor]:
        """
        Interpolate between two embeddings using the diffusion model.
        
        Args:
            x1: First embedding
            x2: Second embedding
            num_steps: Number of interpolation steps
            seed: Random seed for reproducibility
            
        Returns:
            List of interpolated embeddings
        """
        if not self.is_trained:
            self.logger.warning("Model is not trained, interpolation may be linear")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Convert to tensors if needed
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2, dtype=torch.float32)
        
        # Move to device
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        
        # Ensure correct shape
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        if len(x2.shape) == 1:
            x2 = x2.unsqueeze(0)
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        interpolations = []
        
        for alpha in alphas:
            # Linear interpolation
            x_interp = (1 - alpha) * x1 + alpha * x2
            
            # Add noise and denoise
            t = self.config.timesteps // 2  # Mid-level noise
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
            
            # Add noise
            noise = torch.randn_like(x_interp)
            x_noisy = self.q_sample(x_start=x_interp, t=t_tensor, noise=noise)
            
            # Denoise
            for i in reversed(range(t + 1)):
                t_i = torch.full((1,), i, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x_noisy, t_i)
                
                # Get alpha and beta for this timestep
                alpha_i = self.alphas[i]
                alpha_cumprod_i = self.alphas_cumprod[i]
                beta_i = self.betas[i]
                
                # No noise for i=0
                if i > 0:
                    noise_i = torch.randn_like(x_noisy)
                else:
                    noise_i = torch.zeros_like(x_noisy)
                
                # Update x
                x_noisy = (1 / torch.sqrt(alpha_i)) * (
                    x_noisy - ((1 - alpha_i) / torch.sqrt(1 - alpha_cumprod_i)) * predicted_noise
                ) + torch.sqrt(beta_i) * noise_i
            
            interpolations.append(x_noisy.cpu())
        
        return interpolations
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        progress = self.training_progress.copy()
        
        # Add elapsed time if training is in progress
        if progress['status'] == 'training' and progress['start_time']:
            progress['elapsed_time'] = time.time() - progress['start_time']
        elif progress['end_time'] and progress['start_time']:
            progress['elapsed_time'] = progress['end_time'] - progress['start_time']
        
        return progress

# Convenience functions

def create_diffusion_model(config: Optional[DiffusionConfig] = None) -> DiffusionModel:
    """Create a diffusion model with custom configuration."""
    return DiffusionModel(config or DiffusionConfig())

def get_default_diffusion_model() -> DiffusionModel:
    """Get a default diffusion model."""
    return DiffusionModel()

def load_diffusion_model(path: str) -> Optional[DiffusionModel]:
    """Load a pretrained diffusion model."""
    try:
        # Load config if available
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                config = DiffusionConfig(**config_dict)
        else:
            config = DiffusionConfig()
        
        # Create model and load checkpoint
        model = DiffusionModel(config)
        success = model.load_checkpoint(path)
        
        if success:
            return model
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error loading diffusion model: {e}")
        return None