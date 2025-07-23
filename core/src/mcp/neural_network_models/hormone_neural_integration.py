#!/usr/bin/env python3
"""
Hormone Neural Integration for MCP Core System
Integrates hormone system with neural networks for adaptive behavior.
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
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class HormoneConfig:
    """Configuration for hormone neural integration."""
    
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dim: int = 64,
                 output_dim: int = 4,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: str = None,
                 output_dir: str = "data/models",
                 log_level: str = "INFO"):
        """Initialize hormone configuration."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.log_level = log_level

class HormoneDataset(Dataset):
    """Dataset for hormone system training."""
    
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """Initialize hormone dataset."""
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class HormoneNetwork(nn.Module):
    """Neural network for hormone system."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize hormone network."""
        super(HormoneNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Hormone levels are between 0 and 1
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.layers(x)

class HormoneNeuralIntegration:
    """
    Hormone Neural Integration for MCP Core System.
    
    Integrates the brain-inspired hormone system with neural networks
    to create an adaptive regulatory system that can modulate system
    behavior based on internal and external conditions.
    
    Hormones:
    - Stress: Increases under high resource usage or errors
    - Efficiency: Reflects system performance and optimization
    - Adaptation: Controls learning rate and flexibility
    - Stability: Reflects system reliability and consistency
    """
    
    def __init__(self, config: HormoneConfig = None):
        """Initialize hormone neural integration."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HormoneNeuralIntegration")
        
        self.config = config or HormoneConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = HormoneNetwork(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim
        )
        
        # Setup device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Hormone names
        self.hormone_names = ['stress', 'efficiency', 'adaptation', 'stability']
        
        # Feature names (default)
        self.feature_names = [f'feature_{i}' for i in range(self.config.input_dim)]
        
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
        
        # Hormone state
        self.hormone_levels = {
            'stress': 0.0,
            'efficiency': 1.0,
            'adaptation': 0.5,
            'stability': 1.0
        }
        
        # Hormone history
        self.hormone_history = []
        
        # Feedback system
        self.feedback_weights = {
            'stress': 0.0,
            'efficiency': 0.0,
            'adaptation': 0.0,
            'stability': 0.0
        }
    
    def train(self, inputs: Union[torch.Tensor, np.ndarray], 
             outputs: Union[torch.Tensor, np.ndarray],
             feature_names: Optional[List[str]] = None,
             epochs: Optional[int] = None,
             batch_size: Optional[int] = None):
        """
        Train the hormone neural network.
        
        Args:
            inputs: Input features tensor (system metrics)
            outputs: Output hormone levels tensor
            feature_names: Names of input features
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size for training (overrides config)
        """
        if epochs is None:
            epochs = self.config.epochs
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Convert inputs and outputs to tensors if needed
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs, dtype=torch.float32)
        
        # Move to device
        inputs = inputs.to(self.device)
        outputs = outputs.to(self.device)
        
        # Store feature names if provided
        if feature_names and len(feature_names) == inputs.shape[1]:
            self.feature_names = feature_names
        
        # Create dataset and dataloader
        dataset = HormoneDataset(inputs, outputs)
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
            for step, (batch_inputs, batch_outputs) in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_inputs)
                
                # Compute loss
                loss = F.mse_loss(predictions, batch_outputs)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                self.training_progress['current_step'] = epoch * len(dataloader) + step
                self.training_progress['loss'] = loss.item()
                
                # Log progress
                if step % 10 == 0:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(dataloader)}, Loss: {loss.item():.6f}")
            
            # Update epoch progress
            self.training_progress['current_epoch'] = epoch + 1
            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save model
        model_path = self._save_model(f"hormone-{int(time.time())}")
        
        # Update training state
        self.training_progress['status'] = 'completed'
        self.training_progress['end_time'] = time.time()
        self.is_trained = True
        
        self.logger.info(f"Hormone neural network training completed and saved to {model_path}")
        return model_path
    
    def _save_model(self, name: str) -> str:
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
        
        # Save feature names
        feature_path = os.path.join(model_dir, "features.json")
        with open(feature_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        
        return model_dir
    
    def load_model(self, path: str) -> bool:
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
            
            # Load feature names
            feature_path = os.path.join(path, "features.json")
            if os.path.exists(feature_path):
                with open(feature_path, "r") as f:
                    self.feature_names = json.load(f)
            
            self.is_trained = True
            self.logger.info(f"Loaded hormone neural network from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading hormone neural network: {e}")
            return False
    
    @torch.no_grad()
    def predict_hormone_levels(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Predict hormone levels from system metrics.
        
        Args:
            metrics: Dictionary of system metrics
            
        Returns:
            Dictionary of predicted hormone levels
        """
        # Convert metrics to tensor
        input_features = []
        for feature_name in self.feature_names:
            if feature_name in metrics:
                input_features.append(metrics[feature_name])
            else:
                # Use default value if metric not available
                input_features.append(0.0)
        
        input_tensor = torch.tensor([input_features], dtype=torch.float32, device=self.device)
        
        # Predict hormone levels
        self.model.eval()
        predictions = self.model(input_tensor).cpu().numpy()[0]
        
        # Convert to dictionary
        hormone_levels = {}
        for i, hormone_name in enumerate(self.hormone_names):
            if i < len(predictions):
                hormone_levels[hormone_name] = float(predictions[i])
            else:
                hormone_levels[hormone_name] = 0.0
        
        return hormone_levels
    
    def update_hormone_levels(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update hormone levels based on system metrics.
        
        Args:
            metrics: Dictionary of system metrics
            
        Returns:
            Dictionary of updated hormone levels
        """
        if self.is_trained:
            # Use neural network to predict hormone levels
            predicted_levels = self.predict_hormone_levels(metrics)
            
            # Apply feedback weights
            for hormone, weight in self.feedback_weights.items():
                if hormone in predicted_levels:
                    predicted_levels[hormone] += weight
                    # Clamp to valid range
                    predicted_levels[hormone] = max(0.0, min(1.0, predicted_levels[hormone]))
            
            # Update hormone levels
            self.hormone_levels = predicted_levels
            
        else:
            # Fallback to rule-based update
            self._rule_based_update(metrics)
        
        # Record hormone history
        self.hormone_history.append({
            'timestamp': time.time(),
            'levels': self.hormone_levels.copy(),
            'metrics': metrics.copy()
        })
        
        # Keep history limited
        if len(self.hormone_history) > 1000:
            self.hormone_history = self.hormone_history[-1000:]
        
        return self.hormone_levels
    
    def _rule_based_update(self, metrics: Dict[str, float]):
        """
        Update hormone levels using rule-based approach.
        
        Args:
            metrics: Dictionary of system metrics
        """
        # Stress hormone - increases with high resource usage
        if 'cpu_usage' in metrics or 'memory_usage' in metrics:
            cpu = metrics.get('cpu_usage', 0.0) / 100.0
            memory = metrics.get('memory_usage', 0.0) / 100.0
            resource_stress = (cpu + memory) / 2.0
            
            self.hormone_levels['stress'] = (
                self.hormone_levels['stress'] * 0.9 + resource_stress * 0.1
            )
        
        # Efficiency hormone - decreases with errors and high response times
        if 'error_count' in metrics or 'response_time' in metrics:
            error_factor = 0.0
            if 'error_count' in metrics:
                error_factor = max(0.0, min(1.0, metrics['error_count'] / 100.0))
            
            time_factor = 0.0
            if 'response_time' in metrics:
                time_factor = max(0.0, min(1.0, metrics['response_time'] / 10.0))
            
            efficiency_impact = (error_factor + time_factor) / 2.0
            
            self.hormone_levels['efficiency'] = (
                self.hormone_levels['efficiency'] * 0.95 - efficiency_impact * 0.05
            )
        
        # Adaptation hormone - increases with changing conditions
        if 'throughput' in metrics or 'active_lobes' in metrics:
            change_factor = 0.0
            if len(self.hormone_history) > 0:
                last_metrics = self.hormone_history[-1]['metrics']
                
                if 'throughput' in metrics and 'throughput' in last_metrics:
                    throughput_change = abs(metrics['throughput'] - last_metrics['throughput'])
                    change_factor += min(1.0, throughput_change / 10.0)
                
                if 'active_lobes' in metrics and 'active_lobes' in last_metrics:
                    lobe_change = abs(metrics['active_lobes'] - last_metrics['active_lobes'])
                    change_factor += min(1.0, lobe_change / 2.0)
            
            self.hormone_levels['adaptation'] = (
                self.hormone_levels['adaptation'] * 0.9 + change_factor * 0.1
            )
        
        # Stability hormone - decreases with errors and increases with uptime
        if 'error_count' in metrics or 'uptime' in metrics:
            stability_factor = 0.0
            
            if 'error_count' in metrics:
                error_impact = max(0.0, min(1.0, metrics['error_count'] / 100.0))
                stability_factor -= error_impact * 0.1
            
            if 'uptime' in metrics:
                # Uptime in hours
                uptime_hours = metrics['uptime'] / 3600.0
                uptime_factor = min(1.0, uptime_hours / 24.0)  # Max effect after 24 hours
                stability_factor += uptime_factor * 0.05
            
            self.hormone_levels['stability'] = (
                self.hormone_levels['stability'] + stability_factor
            )
        
        # Apply feedback weights
        for hormone, weight in self.feedback_weights.items():
            self.hormone_levels[hormone] += weight
        
        # Clamp all hormone levels to valid range
        for hormone in self.hormone_levels:
            self.hormone_levels[hormone] = max(0.0, min(1.0, self.hormone_levels[hormone]))
    
    def provide_feedback(self, hormone: str, direction: float):
        """
        Provide feedback to adjust hormone levels.
        
        Args:
            hormone: Name of hormone to adjust
            direction: Direction and magnitude of adjustment (-1.0 to 1.0)
        """
        if hormone in self.feedback_weights:
            # Apply feedback with decay
            self.feedback_weights[hormone] = direction * 0.1
            
            self.logger.debug(f"Feedback provided for {hormone}: {direction}")
            return True
        else:
            self.logger.warning(f"Unknown hormone: {hormone}")
            return False
    
    def decay_feedback(self):
        """Decay feedback weights over time."""
        for hormone in self.feedback_weights:
            self.feedback_weights[hormone] *= 0.9
    
    def get_hormone_levels(self) -> Dict[str, float]:
        """Get current hormone levels."""
        return self.hormone_levels.copy()
    
    def get_hormone_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get hormone level history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of hormone history entries
        """
        return self.hormone_history[-limit:]
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        progress = self.training_progress.copy()
        
        # Add elapsed time if training is in progress
        if progress['status'] == 'training' and progress['start_time']:
            progress['elapsed_time'] = time.time() - progress['start_time']
        elif progress['end_time'] and progress['start_time']:
            progress['elapsed_time'] = progress['end_time'] - progress['start_time']
        
        return progress
    
    def generate_training_data(self, metrics_history: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from metrics history.
        
        Args:
            metrics_history: List of metric dictionaries
            
        Returns:
            Tuple of (inputs, outputs) arrays for training
        """
        if not metrics_history:
            self.logger.error("No metrics history provided")
            return np.array([]), np.array([])
        
        # Extract feature names from first metrics entry
        feature_names = list(metrics_history[0].keys())
        self.feature_names = feature_names
        
        # Create input and output arrays
        inputs = []
        outputs = []
        
        for metrics in metrics_history:
            # Create input features
            input_features = [metrics.get(feature, 0.0) for feature in feature_names]
            inputs.append(input_features)
            
            # Generate target hormone levels using rule-based approach
            self._rule_based_update(metrics)
            hormone_levels = [
                self.hormone_levels['stress'],
                self.hormone_levels['efficiency'],
                self.hormone_levels['adaptation'],
                self.hormone_levels['stability']
            ]
            outputs.append(hormone_levels)
        
        return np.array(inputs), np.array(outputs)

# Convenience functions

def create_hormone_integration(config: Optional[HormoneConfig] = None) -> HormoneNeuralIntegration:
    """Create a hormone neural integration with custom configuration."""
    return HormoneNeuralIntegration(config or HormoneConfig())

def get_default_hormone_integration() -> HormoneNeuralIntegration:
    """Get a default hormone neural integration."""
    return HormoneNeuralIntegration()

def load_hormone_integration(path: str) -> Optional[HormoneNeuralIntegration]:
    """Load a pretrained hormone neural integration."""
    try:
        # Load config if available
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                config = HormoneConfig(**config_dict)
        else:
            config = HormoneConfig()
        
        # Create model and load checkpoint
        model = HormoneNeuralIntegration(config)
        success = model.load_model(path)
        
        if success:
            return model
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error loading hormone neural integration: {e}")
        return None