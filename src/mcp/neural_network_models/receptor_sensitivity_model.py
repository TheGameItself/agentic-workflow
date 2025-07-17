"""
Receptor Sensitivity Adaptation Model for Hormone System.

This module implements a neural network model for receptor sensitivity adaptation
based on performance feedback. It provides both algorithmic and neural network
implementations that can adapt receptor sensitivities to optimize responses.

References:
- Requirements 2.3, 2.6 from MCP System Upgrade specification
- Neural network alternatives for computational optimization
"""

import logging
import math
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Try to import torch, but provide fallbacks if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Neural receptor sensitivity model will use fallback implementation.")


@dataclass
class ReceptorPerformanceData:
    """Data structure for receptor performance tracking."""
    lobe_name: str
    hormone_name: str
    receptor_subtype: str
    current_sensitivity: float
    performance_score: float
    context: Dict[str, Any]
    timestamp: datetime


class ReceptorSensitivityNetwork:
    """
    Neural network model for receptor sensitivity adaptation.
    
    This class implements a neural network that can learn and predict
    optimal receptor sensitivity levels based on performance feedback.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 24, output_dim: int = 1):
        """
        Initialize the receptor sensitivity neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (typically 1 for sensitivity)
        """
        self.logger = logging.getLogger("ReceptorSensitivityNetwork")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Model architecture
        if TORCH_AVAILABLE:
            self.model = self._create_torch_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = nn.MSELoss()
        else:
            self.model = self._create_fallback_model()
        
        # Training state
        self.is_trained = False
        self.training_iterations = 0
        self.training_loss_history = []
        self.validation_loss_history = []
        
        self.logger.info(f"Initialized receptor sensitivity neural network (PyTorch available: {TORCH_AVAILABLE})")
    
    def _create_torch_model(self) -> nn.Module:
        """
        Create PyTorch neural network model.
        
        Returns:
            PyTorch neural network model
        """
        class SensitivityModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                # Use sigmoid and scale to 0.1-2.0 range for sensitivity
                x = 0.1 + 1.9 * torch.sigmoid(self.fc3(x))
                return x
        
        return SensitivityModel(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_fallback_model(self) -> Dict:
        """
        Create a fallback model when PyTorch is not available.
        
        Returns:
            Simple dictionary-based model
        """
        return {
            "weights1": np.random.randn(self.input_dim, self.hidden_dim) * 0.1,
            "bias1": np.zeros(self.hidden_dim),
            "weights2": np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1,
            "bias2": np.zeros(self.hidden_dim),
            "weights3": np.random.randn(self.hidden_dim, self.output_dim) * 0.1,
            "bias3": np.zeros(self.output_dim)
        }
    
    def _fallback_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for fallback model.
        
        Args:
            x: Input features
            
        Returns:
            Model predictions
        """
        # First layer
        h1 = np.maximum(0, np.dot(x, self.model["weights1"]) + self.model["bias1"])
        
        # Apply dropout (randomly zero out some activations)
        dropout_mask = np.random.binomial(1, 0.8, size=h1.shape)
        h1 *= dropout_mask / 0.8  # Scale by dropout probability
        
        # Second layer
        h2 = np.maximum(0, np.dot(h1, self.model["weights2"]) + self.model["bias2"])
        
        # Apply dropout
        dropout_mask = np.random.binomial(1, 0.8, size=h2.shape)
        h2 *= dropout_mask / 0.8
        
        # Output layer with sigmoid activation scaled to 0.1-2.0 range
        sigmoid_output = 1 / (1 + np.exp(-np.dot(h2, self.model["weights3"]) - self.model["bias3"]))
        output = 0.1 + 1.9 * sigmoid_output  # Scale to 0.1-2.0 range
        
        return output    

    def predict(self, features: Union[np.ndarray, List[float]]) -> float:
        """
        Predict optimal receptor sensitivity based on input features.
        
        Args:
            features: Input features for sensitivity prediction
            
        Returns:
            Predicted optimal receptor sensitivity (0.1-2.0 range)
        """
        # Convert input to appropriate format
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        
        # Ensure correct shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        if TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                output = self.model(x)
                return output.item()
        else:
            return self._fallback_forward(features).item()
    
    def train(self, 
             training_data: List[Tuple[List[float], float]], 
             validation_data: Optional[List[Tuple[List[float], float]]] = None,
             epochs: int = 100,
             batch_size: int = 32,
             learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train the neural network on receptor sensitivity data.
        
        Args:
            training_data: List of (features, target) pairs
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
        
        if not training_data:
            self.logger.warning("No training data provided")
            return {
                "success": False,
                "error": "No training data provided",
                "training_time": 0.0
            }
        
        # Prepare data
        X_train = np.array([features for features, _ in training_data], dtype=np.float32)
        y_train = np.array([target for _, target in training_data], dtype=np.float32).reshape(-1, 1)
        
        if validation_data:
            X_val = np.array([features for features, _ in validation_data], dtype=np.float32)
            y_val = np.array([target for _, target in validation_data], dtype=np.float32).reshape(-1, 1)
        
        # Train the model
        if TORCH_AVAILABLE:
            train_results = self._train_torch_model(
                X_train, y_train, 
                X_val, y_val if validation_data else None,
                epochs, batch_size, learning_rate
            )
        else:
            train_results = self._train_fallback_model(
                X_train, y_train, 
                X_val, y_val if validation_data else None,
                epochs, batch_size, learning_rate
            )
        
        # Update training state
        self.is_trained = True
        self.training_iterations += epochs
        
        # Calculate training time
        training_time = time.time() - start_time
        
        self.logger.info(f"Trained receptor sensitivity neural network for {epochs} epochs in {training_time:.2f}s")
        
        return {
            "success": True,
            "training_time": training_time,
            "final_loss": train_results["final_loss"],
            "validation_loss": train_results.get("validation_loss"),
            "epochs": epochs,
            "training_iterations": self.training_iterations
        }    
  
  def _train_torch_model(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          epochs: int = 100,
                          batch_size: int = 32,
                          learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train PyTorch model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training results
        """
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / len(train_loader)
            self.training_loss_history.append(avg_loss)
            
            # Validate if validation data is provided
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.loss_fn(val_outputs, y_val_tensor).item()
                    self.validation_loss_history.append(val_loss)
                self.model.train()
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                if X_val is not None and y_val is not None:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Final validation
        val_loss = None
        if X_val is not None and y_val is not None:
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.loss_fn(val_outputs, y_val_tensor).item()
        
        return {
            "final_loss": self.training_loss_history[-1],
            "validation_loss": val_loss,
            "loss_history": self.training_loss_history,
            "val_loss_history": self.validation_loss_history if val_loss is not None else None
        }    
    
    def _train_fallback_model(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None,
                             y_val: Optional[np.ndarray] = None,
                             epochs: int = 100,
                             batch_size: int = 32,
                             learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train fallback model using simple gradient descent.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training results
        """
        # Simple SGD implementation
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                h1 = np.maximum(0, np.dot(X_batch, self.model["weights1"]) + self.model["bias1"])
                h2 = np.maximum(0, np.dot(h1, self.model["weights2"]) + self.model["bias2"])
                sigmoid_output = 1 / (1 + np.exp(-np.dot(h2, self.model["weights3"]) - self.model["bias3"]))
                output = 0.1 + 1.9 * sigmoid_output  # Scale to 0.1-2.0 range
                
                # Calculate loss (MSE)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                
                # Backward pass (simplified)
                # Output layer gradients
                d_output = 2 * (output - y_batch) / len(X_batch)
                d_sigmoid = 1.9 * sigmoid_output * (1 - sigmoid_output)  # Chain rule with scaling factor
                d_h2 = np.dot(d_output * d_sigmoid, self.model["weights3"].T)
                
                # Hidden layer gradients
                d_relu2 = d_h2 * (h2 > 0)
                d_h1 = np.dot(d_relu2, self.model["weights2"].T)
                d_relu1 = d_h1 * (h1 > 0)
                
                # Update weights and biases
                self.model["weights3"] -= learning_rate * np.dot(h2.T, d_output * d_sigmoid)
                self.model["bias3"] -= learning_rate * np.sum(d_output * d_sigmoid, axis=0)
                self.model["weights2"] -= learning_rate * np.dot(h1.T, d_relu2)
                self.model["bias2"] -= learning_rate * np.sum(d_relu2, axis=0)
                self.model["weights1"] -= learning_rate * np.dot(X_batch.T, d_relu1)
                self.model["bias1"] -= learning_rate * np.sum(d_relu1, axis=0)
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / n_batches
            self.training_loss_history.append(avg_loss)
            
            # Validate if validation data is provided
            if X_val is not None and y_val is not None:
                # Forward pass for validation
                h1_val = np.maximum(0, np.dot(X_val, self.model["weights1"]) + self.model["bias1"])
                h2_val = np.maximum(0, np.dot(h1_val, self.model["weights2"]) + self.model["bias2"])
                sigmoid_output_val = 1 / (1 + np.exp(-np.dot(h2_val, self.model["weights3"]) - self.model["bias3"]))
                output_val = 0.1 + 1.9 * sigmoid_output_val  # Scale to 0.1-2.0 range
                
                # Calculate validation loss
                val_loss = np.mean((output_val - y_val) ** 2)
                self.validation_loss_history.append(val_loss)
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                if X_val is not None and y_val is not None:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Final validation
        val_loss = None
        if X_val is not None and y_val is not None:
            h1_val = np.maximum(0, np.dot(X_val, self.model["weights1"]) + self.model["bias1"])
            h2_val = np.maximum(0, np.dot(h1_val, self.model["weights2"]) + self.model["bias2"])
            sigmoid_output_val = 1 / (1 + np.exp(-np.dot(h2_val, self.model["weights3"]) - self.model["bias3"]))
            output_val = 0.1 + 1.9 * sigmoid_output_val  # Scale to 0.1-2.0 range
            val_loss = np.mean((output_val - y_val) ** 2)
        
        return {
            "final_loss": self.training_loss_history[-1],
            "validation_loss": val_loss,
            "loss_history": self.training_loss_history,
            "val_loss_history": self.validation_loss_history if val_loss is not None else None
        }  
  
    def save(self, filepath: str) -> bool:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if TORCH_AVAILABLE:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_iterations': self.training_iterations,
                    'is_trained': self.is_trained,
                    'training_loss_history': self.training_loss_history,
                    'validation_loss_history': self.validation_loss_history,
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'output_dim': self.output_dim
                }, filepath)
            else:
                # Save fallback model
                np.savez(filepath, 
                    weights1=self.model["weights1"],
                    bias1=self.model["bias1"],
                    weights2=self.model["weights2"],
                    bias2=self.model["bias2"],
                    weights3=self.model["weights3"],
                    bias3=self.model["bias3"],
                    training_iterations=self.training_iterations,
                    is_trained=self.is_trained,
                    training_loss_history=np.array(self.training_loss_history),
                    validation_loss_history=np.array(self.validation_loss_history),
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim
                )
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file not found: {filepath}")
                return False
            
            if TORCH_AVAILABLE and filepath.endswith('.pt'):
                checkpoint = torch.load(filepath)
                
                # Recreate model if dimensions don't match
                if (self.input_dim != checkpoint['input_dim'] or 
                    self.hidden_dim != checkpoint['hidden_dim'] or 
                    self.output_dim != checkpoint['output_dim']):
                    
                    self.input_dim = checkpoint['input_dim']
                    self.hidden_dim = checkpoint['hidden_dim']
                    self.output_dim = checkpoint['output_dim']
                    self.model = self._create_torch_model()
                    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_iterations = checkpoint['training_iterations']
                self.is_trained = checkpoint['is_trained']
                self.training_loss_history = checkpoint['training_loss_history']
                self.validation_loss_history = checkpoint['validation_loss_history']
                
            elif filepath.endswith('.npz'):
                # Load fallback model
                data = np.load(filepath)
                
                # Update dimensions if needed
                if (self.input_dim != data['input_dim'] or 
                    self.hidden_dim != data['hidden_dim'] or 
                    self.output_dim != data['output_dim']):
                    
                    self.input_dim = int(data['input_dim'])
                    self.hidden_dim = int(data['hidden_dim'])
                    self.output_dim = int(data['output_dim'])
                
                self.model = {
                    "weights1": data['weights1'],
                    "bias1": data['bias1'],
                    "weights2": data['weights2'],
                    "bias2": data['bias2'],
                    "weights3": data['weights3'],
                    "bias3": data['bias3']
                }
                
                self.training_iterations = int(data['training_iterations'])
                self.is_trained = bool(data['is_trained'])
                self.training_loss_history = data['training_loss_history'].tolist()
                self.validation_loss_history = data['validation_loss_history'].tolist()
            
            else:
                self.logger.error(f"Unsupported model file format: {filepath}")
                return False
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False


class ReceptorSensitivityAdapter:
    """
    Adapter for receptor sensitivity based on performance feedback.
    
    This class provides both algorithmic and neural implementations for
    adapting receptor sensitivities based on performance feedback.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the receptor sensitivity adapter.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.logger = logging.getLogger("ReceptorSensitivityAdapter")
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "models", "receptor_sensitivity")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Neural networks for different hormone-lobe combinations
        self.networks: Dict[str, ReceptorSensitivityNetwork] = {}
        
        # Performance history for algorithmic adaptation
        self.performance_history: Dict[str, List[ReceptorPerformanceData]] = {}
        
        # Maximum history length per receptor
        self.max_history_length = 100
        
        # Implementation selection
        self.active_implementation = "algorithmic"  # Start with algorithmic implementation
        self.implementation_performance = {
            "algorithmic": {"accuracy": 0.0, "latency": 0.0, "calls": 0},
            "neural": {"accuracy": 0.0, "latency": 0.0, "calls": 0}
        }
        
        self.logger.info(f"Initialized receptor sensitivity adapter (model directory: {self.model_dir})")
    
    def _get_receptor_key(self, lobe_name: str, hormone_name: str, receptor_subtype: str = None) -> str:
        """
        Get a unique key for a receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            
        Returns:
            Unique receptor key
        """
        if receptor_subtype:
            return f"{lobe_name}:{hormone_name}:{receptor_subtype}"
        else:
            return f"{lobe_name}:{hormone_name}"
    
    def get_or_create_network(self, lobe_name: str, hormone_name: str, receptor_subtype: str = None) -> ReceptorSensitivityNetwork:
        """
        Get an existing network or create a new one for a receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            
        Returns:
            Neural network for the receptor
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        if receptor_key not in self.networks:
            self.networks[receptor_key] = ReceptorSensitivityNetwork()
            
            # Try to load pre-trained model
            model_path = os.path.join(self.model_dir, f"{receptor_key.replace(':', '_')}_model")
            if TORCH_AVAILABLE:
                model_path += ".pt"
            else:
                model_path += ".npz"
                
            if os.path.exists(model_path):
                self.networks[receptor_key].load(model_path)
                self.logger.info(f"Loaded pre-trained model for {receptor_key}")
        
        return self.networks[receptor_key]
    
    def adapt_sensitivity(self, 
                         lobe_name: str, 
                         hormone_name: str, 
                         performance: float,
                         current_sensitivity: Optional[float] = None,
                         receptor_subtype: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt receptor sensitivity based on performance feedback.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score (0.0-1.0)
            current_sensitivity: Current sensitivity level (0.1-2.0)
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New receptor sensitivity level (0.1-2.0)
        """
        # Ensure performance is within valid range
        performance = max(0.0, min(1.0, performance))
        
        # Use default sensitivity if not provided
        if current_sensitivity is None:
            current_sensitivity = 0.5
        else:
            current_sensitivity = max(0.1, min(2.0, current_sensitivity))
        
        # Use empty context if not provided
        if context is None:
            context = {}
        
        # Record performance data
        self._record_performance(lobe_name, hormone_name, receptor_subtype, current_sensitivity, performance, context)
        
        # Choose implementation
        if self.active_implementation == "neural" and self._can_use_neural(lobe_name, hormone_name, receptor_subtype):
            start_time = time.time()
            new_sensitivity = self._adapt_neural(lobe_name, hormone_name, performance, current_sensitivity, receptor_subtype, context)
            latency = time.time() - start_time
            
            # Update performance metrics
            self._update_implementation_performance("neural", latency)
            
            return new_sensitivity
        else:
            start_time = time.time()
            new_sensitivity = self._adapt_algorithmic(lobe_name, hormone_name, performance, current_sensitivity, receptor_subtype, context)
            latency = time.time() - start_time
            
            # Update performance metrics
            self._update_implementation_performance("algorithmic", latency)
            
            return new_sensitivity
    
    def _record_performance(self, 
                           lobe_name: str, 
                           hormone_name: str, 
                           receptor_subtype: Optional[str],
                           current_sensitivity: float,
                           performance: float,
                           context: Dict[str, Any]) -> None:
        """
        Record performance data for a receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            current_sensitivity: Current sensitivity level
            performance: Performance score
            context: Context information
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Initialize history if needed
        if receptor_key not in self.performance_history:
            self.performance_history[receptor_key] = []
        
        # Create performance data point
        data_point = ReceptorPerformanceData(
            lobe_name=lobe_name,
            hormone_name=hormone_name,
            receptor_subtype=receptor_subtype or "default",
            current_sensitivity=current_sensitivity,
            performance_score=performance,
            context=context,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.performance_history[receptor_key].append(data_point)
        
        # Limit history size
        if len(self.performance_history[receptor_key]) > self.max_history_length:
            self.performance_history[receptor_key].pop(0)
    
    def _can_use_neural(self, lobe_name: str, hormone_name: str, receptor_subtype: Optional[str] = None) -> bool:
        """
        Check if neural implementation can be used for a receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            
        Returns:
            True if neural implementation can be used
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Check if we have enough training data
        if receptor_key in self.performance_history:
            return len(self.performance_history[receptor_key]) >= 10
        
        return False
    
    def _adapt_neural(self, lobe_name: str, hormone_name: str, performance: float,
                     current_sensitivity: float, receptor_subtype: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt sensitivity using neural network implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New sensitivity level
        """
        network = self.get_or_create_network(lobe_name, hormone_name, receptor_subtype)
        
        # Prepare features for neural network
        features = self._extract_features(lobe_name, hormone_name, performance, current_sensitivity, context)
        
        # Get prediction from neural network
        predicted_sensitivity = network.predict(features)
        
        # Train network with new data if we have enough history
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        if receptor_key in self.performance_history and len(self.performance_history[receptor_key]) >= 20:
            self._train_network_incrementally(network, receptor_key)
        
        return predicted_sensitivity
    
    def _adapt_algorithmic(self, lobe_name: str, hormone_name: str, performance: float,
                          current_sensitivity: float, receptor_subtype: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt sensitivity using algorithmic implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New sensitivity level
        """
        # Simple algorithmic adaptation based on performance
        if performance > 0.8:
            # Good performance, slightly increase sensitivity
            adjustment = 0.05 * (performance - 0.8)
        elif performance < 0.5:
            # Poor performance, decrease sensitivity
            adjustment = -0.1 * (0.5 - performance)
        else:
            # Moderate performance, small adjustment
            adjustment = 0.02 * (performance - 0.65)
        
        # Apply context-based modulation
        if context:
            stress_level = context.get('stress_level', 0.0)
            urgency = context.get('urgency', 0.0)
            
            # Increase sensitivity under stress or urgency
            if stress_level > 0.7 or urgency > 0.8:
                adjustment += 0.03
        
        # Calculate new sensitivity
        new_sensitivity = current_sensitivity + adjustment
        
        # Ensure within valid range
        new_sensitivity = max(0.1, min(2.0, new_sensitivity))
        
        return new_sensitivity
    
    def _extract_features(self, lobe_name: str, hormone_name: str, performance: float,
                         current_sensitivity: float, context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Extract features for neural network input.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            context: Optional context information
            
        Returns:
            Feature vector for neural network
        """
        features = [
            performance,
            current_sensitivity,
            hash(lobe_name) % 100 / 100.0,  # Normalized lobe hash
            hash(hormone_name) % 100 / 100.0,  # Normalized hormone hash
        ]
        
        # Add context features if available
        if context:
            features.extend([
                context.get('stress_level', 0.0),
                context.get('urgency', 0.0),
                context.get('confidence', 0.5),
                context.get('workload', 0.5),
                context.get('time_of_day', 0.5),
                context.get('system_load', 0.5)
            ])
        else:
            # Default context values
            features.extend([0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        
        return features
    
    def _train_network_incrementally(self, network: ReceptorSensitivityNetwork, receptor_key: str):
        """
        Train network incrementally with recent performance data.
        
        Args:
            network: Neural network to train
            receptor_key: Receptor key for performance history
        """
        if receptor_key not in self.performance_history:
            return
        
        # Get recent performance data
        recent_data = self.performance_history[receptor_key][-20:]  # Last 20 data points
        
        # Prepare training data
        training_data = []
        for data_point in recent_data:
            features = self._extract_features(
                data_point.lobe_name,
                data_point.hormone_name,
                data_point.performance_score,
                data_point.current_sensitivity,
                data_point.context
            )
            
            # Target is optimal sensitivity based on performance
            target_sensitivity = self._calculate_optimal_sensitivity(data_point)
            
            training_data.append((features, target_sensitivity))
        
        # Train network with small number of epochs for incremental learning
        if training_data:
            network.train(training_data, epochs=5, batch_size=min(8, len(training_data)))
    
    def _calculate_optimal_sensitivity(self, data_point: ReceptorPerformanceData) -> float:
        """
        Calculate optimal sensitivity based on performance data.
        
        Args:
            data_point: Performance data point
            
        Returns:
            Optimal sensitivity level
        """
        # Simple heuristic for optimal sensitivity
        performance = data_point.performance_score
        current_sensitivity = data_point.current_sensitivity
        
        if performance > 0.8:
            # Good performance, maintain or slightly increase
            return min(2.0, current_sensitivity * 1.05)
        elif performance < 0.5:
            # Poor performance, decrease sensitivity
            return max(0.1, current_sensitivity * 0.9)
        else:
            # Moderate performance, adjust based on performance level
            adjustment_factor = 0.95 + 0.1 * performance
            return max(0.1, min(2.0, current_sensitivity * adjustment_factor))
    
    def _update_implementation_performance(self, implementation: str, latency: float):
        """
        Update performance metrics for an implementation.
        
        Args:
            implementation: Implementation type ('neural' or 'algorithmic')
            latency: Processing latency
        """
        if implementation in self.implementation_performance:
            metrics = self.implementation_performance[implementation]
            metrics['calls'] += 1
            metrics['latency'] = (metrics['latency'] * (metrics['calls'] - 1) + latency) / metrics['calls']
    
    def switch_implementation(self, implementation: str):
        """
        Switch to a different implementation.
        
        Args:
            implementation: Implementation to switch to ('neural' or 'algorithmic')
        """
        if implementation in ['neural', 'algorithmic']:
            self.active_implementation = implementation
            self.logger.info(f"Switched to {implementation} implementation")
        else:
            self.logger.warning(f"Unknown implementation: {implementation}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for both implementations.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'active_implementation': self.active_implementation,
            'implementation_performance': self.implementation_performance.copy(),
            'total_receptors': len(self.networks),
            'total_performance_records': sum(len(history) for history in self.performance_history.values())
        }ance: Performance score
            context: Context information
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Initialize history if needed
        if receptor_key not in self.performance_history:
            self.performance_history[receptor_key] = []
        
        # Create performance data point
        data_point = ReceptorPerformanceData(
            lobe_name=lobe_name,
            hormone_name=hormone_name,
            receptor_subtype=receptor_subtype or "default",
            current_sensitivity=current_sensitivity,
            performance_score=performance,
            context=context,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.performance_history[receptor_key].append(data_point)
        
        # Limit history size
        if len(self.performance_history[receptor_key]) > self.max_history_length:
            self.performance_history[receptor_key].pop(0)
    
    def _can_use_neural(self, lobe_name: str, hormone_name: str, receptor_subtype: Optional[str] = None) -> bool:
        """
        Check if neural implementation can be used for a receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            
        Returns:
            True if neural implementation can be used
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Check if we have enough training data
        if receptor_key in self.performance_history:
            return len(self.performance_history[receptor_key]) >= 10
        
        return False
    
    def _adapt_neural(self, lobe_name: str, hormone_name: str, performance: float,
                     current_sensitivity: float, receptor_subtype: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt sensitivity using neural network implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New sensitivity level
        """
        network = self.get_or_create_network(lobe_name, hormone_name, receptor_subtype)
        
        # Prepare features for neural network
        features = self._extract_features(lobe_name, hormone_name, performance, current_sensitivity, context)
        
        # Get prediction from neural network
        predicted_sensitivity = network.predict(features)
        
        # Train network with new data if we have enough history
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        if receptor_key in self.performance_history and len(self.performance_history[receptor_key]) >= 20:
            self._train_network_incrementally(network, receptor_key)
        
        return predicted_sensitivity
    
    def _adapt_algorithmic(self, lobe_name: str, hormone_name: str, performance: float,
                          current_sensitivity: float, receptor_subtype: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt sensitivity using algorithmic implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New sensitivity level
        """
        # Simple algorithmic adaptation based on performance
        if performance > 0.8:
            # Good performance, slightly increase sensitivity
            adjustment = 0.05 * (performance - 0.8)
        elif performance < 0.5:
            # Poor performance, decrease sensitivity
            adjustment = -0.1 * (0.5 - performance)
        else:
            # Moderate performance, small adjustment
            adjustment = 0.02 * (performance - 0.65)
        
        # Apply context-based modulation
        if context:
            stress_level = context.get('stress_level', 0.0)
            urgency = context.get('urgency', 0.0)
            
            # Increase sensitivity under stress or urgency
            if stress_level > 0.7 or urgency > 0.8:
                adjustment += 0.03
        
        # Calculate new sensitivity
        new_sensitivity = current_sensitivity + adjustment
        
        # Ensure within valid range
        new_sensitivity = max(0.1, min(2.0, new_sensitivity))
        
        return new_sensitivity
    
    def _extract_features(self, lobe_name: str, hormone_name: str, performance: float,
                         current_sensitivity: float, context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Extract features for neural network input.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            context: Optional context information
            
        Returns:
            Feature vector for neural network
        """
        features = [
            performance,
            current_sensitivity,
            hash(lobe_name) % 100 / 100.0,  # Normalized lobe hash
            hash(hormone_name) % 100 / 100.0,  # Normalized hormone hash
        ]
        
        # Add context features if available
        if context:
            features.extend([
                context.get('stress_level', 0.0),
                context.get('urgency', 0.0),
                context.get('confidence', 0.5),
                context.get('workload', 0.5),
                context.get('time_of_day', 0.5),
                context.get('system_load', 0.5)
            ])
        else:
            # Default context values
            features.extend([0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        
        return features
    
    def _train_network_incrementally(self, network: ReceptorSensitivityNetwork, receptor_key: str):
        """
        Train network incrementally with recent performance data.
        
        Args:
            network: Neural network to train
            receptor_key: Receptor key for performance history
        """
        if receptor_key not in self.performance_history:
            return
        
        # Get recent performance data
        recent_data = self.performance_history[receptor_key][-20:]  # Last 20 data points
        
        # Prepare training data
        training_data = []
        for data_point in recent_data:
            features = self._extract_features(
                data_point.lobe_name,
                data_point.hormone_name,
                data_point.performance_score,
                data_point.current_sensitivity,
                data_point.context
            )
            
            # Target is optimal sensitivity based on performance
            target_sensitivity = self._calculate_optimal_sensitivity(data_point)
            
            training_data.append((features, target_sensitivity))
        
        # Train network with small number of epochs for incremental learning
        if training_data:
            network.train(training_data, epochs=5, batch_size=min(8, len(training_data)))
    
    def _calculate_optimal_sensitivity(self, data_point: ReceptorPerformanceData) -> float:
        """
        Calculate optimal sensitivity based on performance data.
        
        Args:
            data_point: Performance data point
            
        Returns:
            Optimal sensitivity level
        """
        # Simple heuristic for optimal sensitivity
        performance = data_point.performance_score
        current_sensitivity = data_point.current_sensitivity
        
        if performance > 0.8:
            # Good performance, maintain or slightly increase
            return min(2.0, current_sensitivity * 1.05)
        elif performance < 0.5:
            # Poor performance, decrease sensitivity
            return max(0.1, current_sensitivity * 0.9)
        else:
            # Moderate performance, adjust based on performance level
            adjustment_factor = 0.95 + 0.1 * performance
            return max(0.1, min(2.0, current_sensitivity * adjustment_factor))
    
    def _update_implementation_performance(self, implementation: str, latency: float):
        """
        Update performance metrics for an implementation.
        
        Args:
            implementation: Implementation type ('neural' or 'algorithmic')
            latency: Processing latency
        """
        if implementation in self.implementation_performance:
            metrics = self.implementation_performance[implementation]
            metrics['calls'] += 1
            metrics['latency'] = (metrics['latency'] * (metrics['calls'] - 1) + latency) / metrics['calls']
    
    def switch_implementation(self, implementation: str):
        """
        Switch to a different implementation.
        
        Args:
            implementation: Implementation to switch to ('neural' or 'algorithmic')
        """
        if implementation in ['neural', 'algorithmic']:
            self.active_implementation = implementation
            self.logger.info(f"Switched to {implementation} implementation")
        else:
            self.logger.warning(f"Unknown implementation: {implementation}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for both implementations.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'active_implementation': self.active_implementation,
            'implementation_performance': self.implementation_performance.copy(),
            'total_receptors': len(self.networks),
            'total_performance_records': sum(len(history) for history in self.performance_history.values())
        }ance: Performance score
            context: Context information
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Initialize history if needed
        if receptor_key not in self.performance_history:
            self.performance_history[receptor_key] = []
        
        # Create performance data point
        data_point = ReceptorPerformanceData(
            lobe_name=lobe_name,
            hormone_name=hormone_name,
            receptor_subtype=receptor_subtype or "default",
            current_sensitivity=current_sensitivity,
            performance_score=performance,
            context=context,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.performance_history[receptor_key].append(data_point)
        
        # Limit history size
        if len(self.performance_history[receptor_key]) > self.max_history_length:
            self.performance_history[receptor_key].pop(0) 
   
    def _adapt_algorithmic(self, 
                          lobe_name: str, 
                          hormone_name: str, 
                          performance: float,
                          current_sensitivity: float,
                          receptor_subtype: Optional[str],
                          context: Dict[str, Any]) -> float:
        """
        Adapt receptor sensitivity using algorithmic implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Context information
            
        Returns:
            New receptor sensitivity level
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Get performance history
        history = self.performance_history.get(receptor_key, [])
        
        # If no history, make a small adjustment based on performance
        if len(history) < 2:
            # If performance is good, slightly increase sensitivity
            if performance > 0.7:
                adjustment = 0.05
            # If performance is poor, slightly decrease sensitivity
            elif performance < 0.3:
                adjustment = -0.05
            # Otherwise, make a very small adjustment
            else:
                adjustment = (performance - 0.5) * 0.1
                
            new_sensitivity = current_sensitivity + adjustment
            
        else:
            # Calculate performance trend
            recent_history = history[-min(10, len(history)):]
            avg_performance = sum(point.performance_score for point in recent_history) / len(recent_history)
            
            # Calculate performance derivative (rate of change)
            if len(recent_history) >= 2:
                performance_change = recent_history[-1].performance_score - recent_history[0].performance_score
                sensitivity_change = recent_history[-1].current_sensitivity - recent_history[0].current_sensitivity
            else:
                performance_change = 0.0
                sensitivity_change = 0.0
            
            # Determine adjustment direction and magnitude
            if abs(sensitivity_change) < 0.001:
                # If sensitivity hasn't changed, try a small adjustment based on performance
                adjustment = (performance - 0.5) * 0.1
            else:
                # If sensitivity has changed, adjust based on whether performance improved
                performance_derivative = performance_change / sensitivity_change if sensitivity_change != 0 else 0.0
                
                # If performance is improving with sensitivity changes, continue in same direction
                if performance_derivative > 0.1:
                    adjustment = 0.05 * (sensitivity_change / abs(sensitivity_change))
                # If performance is declining with sensitivity changes, reverse direction
                elif performance_derivative < -0.1:
                    adjustment = -0.05 * (sensitivity_change / abs(sensitivity_change))
                # If performance is stable, make a small adjustment based on current performance
                else:
                    adjustment = (performance - 0.5) * 0.05
            
            # Apply adjustment with adaptive learning rate based on performance volatility
            performance_values = [point.performance_score for point in recent_history]
            if len(performance_values) >= 2:
                variance = sum((p - avg_performance) ** 2 for p in performance_values) / len(performance_values)
                volatility = math.sqrt(variance)
                
                # Reduce learning rate if performance is volatile
                learning_rate = 1.0 / (1.0 + 10.0 * volatility)
            else:
                learning_rate = 0.5
                
            new_sensitivity = current_sensitivity + adjustment * learning_rate
        
        # Ensure sensitivity stays within valid range
        new_sensitivity = max(0.1, min(2.0, new_sensitivity))
        
        self.logger.debug(f"Algorithmic adaptation for {receptor_key}: {current_sensitivity:.2f} -> {new_sensitivity:.2f} "
                         f"(performance: {performance:.2f})")
        
        return new_sensitivity    
   
 def _adapt_neural(self, 
                     lobe_name: str, 
                     hormone_name: str, 
                     performance: float,
                     current_sensitivity: float,
                     receptor_subtype: Optional[str],
                     context: Dict[str, Any]) -> float:
        """
        Adapt receptor sensitivity using neural implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Context information
            
        Returns:
            New receptor sensitivity level
        """
        # Get neural network for this receptor
        network = self.get_or_create_network(lobe_name, hormone_name, receptor_subtype)
        
        # Extract features for prediction
        features = self._extract_features(lobe_name, hormone_name, performance, current_sensitivity, receptor_subtype, context)
        
        # Predict new sensitivity
        new_sensitivity = network.predict(features)
        
        # Ensure sensitivity stays within valid range
        new_sensitivity = max(0.1, min(2.0, new_sensitivity))
        
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        self.logger.debug(f"Neural adaptation for {receptor_key}: {current_sensitivity:.2f} -> {new_sensitivity:.2f} "
                         f"(performance: {performance:.2f})")
        
        return new_sensitivity
    
    def _extract_features(self, 
                         lobe_name: str, 
                         hormone_name: str, 
                         performance: float,
                         current_sensitivity: float,
                         receptor_subtype: Optional[str],
                         context: Dict[str, Any]) -> List[float]:
        """
        Extract features for neural network input.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Context information
            
        Returns:
            List of features for neural network input
        """
        # Basic features
        features = [
            current_sensitivity,
            performance,
            # Encode lobe name as a numeric feature (hash-based)
            hash(lobe_name) % 1000 / 1000.0,
            # Encode hormone name as a numeric feature (hash-based)
            hash(hormone_name) % 1000 / 1000.0
        ]
        
        # Add receptor subtype feature if available
        if receptor_subtype:
            features.append(hash(receptor_subtype) % 1000 / 1000.0)
        else:
            features.append(0.0)
        
        # Extract relevant context features
        context_features = [
            context.get("hormone_level", 0.0),
            context.get("brain_state_arousal", 0.5),
            context.get("brain_state_focus", 0.5),
            context.get("task_priority", 0.5),
            context.get("system_load", 0.5)
        ]
        
        # Combine all features
        return features + context_features  
  
    def set_active_implementation(self, implementation: str) -> bool:
        """
        Set the active implementation for receptor sensitivity adaptation.
        
        Args:
            implementation: Implementation to use ('neural' or 'algorithmic')
            
        Returns:
            True if successful, False otherwise
        """
        if implementation not in ["neural", "algorithmic"]:
            self.logger.error(f"Invalid implementation: {implementation}")
            return False
        
        # If switching to neural, check if PyTorch is available
        if implementation == "neural" and not TORCH_AVAILABLE:
            self.logger.warning("Cannot switch to neural implementation: PyTorch not available")
            return False
        
        self.active_implementation = implementation
        self.logger.info(f"Switched to {implementation} implementation for receptor sensitivity adaptation")
        return True
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for both implementations.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.implementation_performance.copy()
    
    def get_receptor_performance_history(self, lobe_name: str, hormone_name: str, receptor_subtype: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            
        Returns:
            List of performance data points
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        history = self.performance_history.get(receptor_key, [])
        
        # Convert to dictionaries for easier serialization
        return [
            {
                "lobe_name": point.lobe_name,
                "hormone_name": point.hormone_name,
                "receptor_subtype": point.receptor_subtype,
                "current_sensitivity": point.current_sensitivity,
                "performance_score": point.performance_score,
                "timestamp": point.timestamp.isoformat(),
                "context": point.context
            }
            for point in history
        ]


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create adapter
    adapter = ReceptorSensitivityAdapter()
    
    # Simulate adaptation
    print("Simulating receptor sensitivity adaptation...")
    
    # Initial sensitivity
    sensitivity = 0.5
    
    # Simulate performance feedback
    for i in range(10):
        # Simulate performance based on sensitivity
        # (in a real system, this would be actual performance feedback)
        optimal_sensitivity = 0.8  # Pretend this is the unknown optimal value
        distance_from_optimal = abs(sensitivity - optimal_sensitivity)
        performance = max(0.0, 1.0 - distance_from_optimal)
        
        print(f"Iteration {i+1}: Sensitivity = {sensitivity:.2f}, Performance = {performance:.2f}")
        
        # Adapt sensitivity
        sensitivity = adapter.adapt_sensitivity(
            "memory", 
            "dopamine", 
            performance, 
            sensitivity,
            context={"task_priority": 0.7}
        )
    
    print(f"Final sensitivity: {sensitivity:.2f}")
    
    # Train neural model
    if TORCH_AVAILABLE:
        print("\nTraining neural model...")
        results = adapter.train_neural_models(epochs=50)
        print(f"Training results: {results}")
        
        # Switch to neural implementation
        adapter.set_active_implementation("neural")
        
        # Test neural adaptation
        print("\nTesting neural adaptation...")
        sensitivity = 0.5
        for i in range(5):
            distance_from_optimal = abs(sensitivity - optimal_sensitivity)
            performance = max(0.0, 1.0 - distance_from_optimal)
            
            print(f"Neural iteration {i+1}: Sensitivity = {sensitivity:.2f}, Performance = {performance:.2f}")
            
            sensitivity = adapter.adapt_sensitivity(
                "memory", 
                "dopamine", 
                performance, 
                sensitivity,
                context={"task_priority": 0.7}
            )
        
        print(f"Final neural sensitivity: {sensitivity:.2f}")
    
    # Get performance metrics
    metrics = adapter.get_performance_metrics()
    print(f"\nPerformance metrics: {metrics}")f
ormance: Performance score
            context: Context information
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Initialize history if needed
        if receptor_key not in self.performance_history:
            self.performance_history[receptor_key] = []
        
        # Create performance data point
        data_point = ReceptorPerformanceData(
            lobe_name=lobe_name,
            hormone_name=hormone_name,
            receptor_subtype=receptor_subtype or "default",
            current_sensitivity=current_sensitivity,
            performance_score=performance,
            context=context,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.performance_history[receptor_key].append(data_point)
        
        # Limit history size
        if len(self.performance_history[receptor_key]) > self.max_history_length:
            self.performance_history[receptor_key].pop(0)
    
    def _can_use_neural(self, lobe_name: str, hormone_name: str, receptor_subtype: Optional[str] = None) -> bool:
        """
        Check if neural implementation can be used for a receptor.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            receptor_subtype: Optional receptor subtype
            
        Returns:
            True if neural implementation can be used
        """
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        
        # Check if we have enough training data
        if receptor_key in self.performance_history:
            return len(self.performance_history[receptor_key]) >= 10
        
        return False
    
    def _adapt_neural(self, lobe_name: str, hormone_name: str, performance: float,
                     current_sensitivity: float, receptor_subtype: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt sensitivity using neural network implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New sensitivity level
        """
        network = self.get_or_create_network(lobe_name, hormone_name, receptor_subtype)
        
        # Prepare features for neural network
        features = self._extract_features(lobe_name, hormone_name, performance, current_sensitivity, context)
        
        # Get prediction from neural network
        predicted_sensitivity = network.predict(features)
        
        # Train network with new data if we have enough history
        receptor_key = self._get_receptor_key(lobe_name, hormone_name, receptor_subtype)
        if receptor_key in self.performance_history and len(self.performance_history[receptor_key]) >= 20:
            self._train_network_incrementally(network, receptor_key)
        
        return predicted_sensitivity
    
    def _adapt_algorithmic(self, lobe_name: str, hormone_name: str, performance: float,
                          current_sensitivity: float, receptor_subtype: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adapt sensitivity using algorithmic implementation.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            receptor_subtype: Optional receptor subtype
            context: Optional context information
            
        Returns:
            New sensitivity level
        """
        # Simple algorithmic adaptation based on performance
        if performance > 0.8:
            # Good performance, slightly increase sensitivity
            adjustment = 0.05 * (performance - 0.8)
        elif performance < 0.5:
            # Poor performance, decrease sensitivity
            adjustment = -0.1 * (0.5 - performance)
        else:
            # Moderate performance, small adjustment
            adjustment = 0.02 * (performance - 0.65)
        
        # Apply context-based modulation
        if context:
            stress_level = context.get('stress_level', 0.0)
            urgency = context.get('urgency', 0.0)
            
            # Increase sensitivity under stress or urgency
            if stress_level > 0.7 or urgency > 0.8:
                adjustment += 0.03
        
        # Calculate new sensitivity
        new_sensitivity = current_sensitivity + adjustment
        
        # Ensure within valid range
        new_sensitivity = max(0.1, min(2.0, new_sensitivity))
        
        return new_sensitivity
    
    def _extract_features(self, lobe_name: str, hormone_name: str, performance: float,
                         current_sensitivity: float, context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Extract features for neural network input.
        
        Args:
            lobe_name: Name of the lobe
            hormone_name: Name of the hormone
            performance: Performance score
            current_sensitivity: Current sensitivity level
            context: Optional context information
            
        Returns:
            Feature vector for neural network
        """
        features = [
            performance,
            current_sensitivity,
            hash(lobe_name) % 100 / 100.0,  # Normalized lobe hash
            hash(hormone_name) % 100 / 100.0,  # Normalized hormone hash
        ]
        
        # Add context features if available
        if context:
            features.extend([
                context.get('stress_level', 0.0),
                context.get('urgency', 0.0),
                context.get('confidence', 0.5),
                context.get('workload', 0.5),
                context.get('time_of_day', 0.5),
                context.get('system_load', 0.5)
            ])
        else:
            # Default context values
            features.extend([0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        
        return features
    
    def _train_network_incrementally(self, network: ReceptorSensitivityNetwork, receptor_key: str):
        """
        Train network incrementally with recent performance data.
        
        Args:
            network: Neural network to train
            receptor_key: Receptor key for performance history
        """
        if receptor_key not in self.performance_history:
            return
        
        # Get recent performance data
        recent_data = self.performance_history[receptor_key][-20:]  # Last 20 data points
        
        # Prepare training data
        training_data = []
        for data_point in recent_data:
            features = self._extract_features(
                data_point.lobe_name,
                data_point.hormone_name,
                data_point.performance_score,
                data_point.current_sensitivity,
                data_point.context
            )
            
            # Target is optimal sensitivity based on performance
            target_sensitivity = self._calculate_optimal_sensitivity(data_point)
            
            training_data.append((features, target_sensitivity))
        
        # Train network with small number of epochs for incremental learning
        if training_data:
            network.train(training_data, epochs=5, batch_size=min(8, len(training_data)))
    
    def _calculate_optimal_sensitivity(self, data_point: ReceptorPerformanceData) -> float:
        """
        Calculate optimal sensitivity based on performance data.
        
        Args:
            data_point: Performance data point
            
        Returns:
            Optimal sensitivity level
        """
        # Simple heuristic for optimal sensitivity
        performance = data_point.performance_score
        current_sensitivity = data_point.current_sensitivity
        
        if performance > 0.8:
            # Good performance, maintain or slightly increase
            return min(2.0, current_sensitivity * 1.05)
        elif performance < 0.5:
            # Poor performance, decrease sensitivity
            return max(0.1, current_sensitivity * 0.9)
        else:
            # Moderate performance, adjust based on performance level
            adjustment_factor = 0.95 + 0.1 * performance
            return max(0.1, min(2.0, current_sensitivity * adjustment_factor))
    
    def _update_implementation_performance(self, implementation: str, latency: float):
        """
        Update performance metrics for an implementation.
        
        Args:
            implementation: Implementation type ('neural' or 'algorithmic')
            latency: Processing latency
        """
        if implementation in self.implementation_performance:
            metrics = self.implementation_performance[implementation]
            metrics['calls'] += 1
            metrics['latency'] = (metrics['latency'] * (metrics['calls'] - 1) + latency) / metrics['calls']
    
    def switch_implementation(self, implementation: str):
        """
        Switch to a different implementation.
        
        Args:
            implementation: Implementation to switch to ('neural' or 'algorithmic')
        """
        if implementation in ['neural', 'algorithmic']:
            self.active_implementation = implementation
            self.logger.info(f"Switched to {implementation} implementation")
        else:
            self.logger.warning(f"Unknown implementation: {implementation}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for both implementations.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'active_implementation': self.active_implementation,
            'implementation_performance': self.implementation_performance.copy(),
            'total_receptors': len(self.networks),
            'total_performance_records': sum(len(history) for history in self.performance_history.values())
        }