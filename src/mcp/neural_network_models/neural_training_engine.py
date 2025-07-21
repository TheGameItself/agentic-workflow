"""
Neural Training Engine: Specialized training engine for hormone calculations.

This module provides a specialized training engine for hormone calculations,
with automatic performance comparison and model persistence.

References:
- Requirements 1.6, 1.7, 1.8 from MCP System Upgrade specification
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..dual_implementation import ImplementationType, PerformanceMetrics
from ..fallback_manager import FallbackManager
from ..implementation_switching_monitor import ImplementationSwitchingMonitor
from .training_engine import AdvancedTrainingEngine, TrainingConfiguration


class ModelStatus(Enum):
    """Status of a neural model."""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


@dataclass
class TrainingExample:
    """Training example for hormone calculations."""
    inputs: Dict[str, float]
    output: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingResult:
    """Result of training a neural model."""
    model_id: str
    hormone_type: str
    success: bool
    accuracy: float
    training_time: float
    epochs_completed: int
    early_stopping: bool
    loss_history: List[float]
    val_loss_history: List[float]
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing algorithmic and neural implementations."""
    hormone_type: str
    algorithmic_metrics: PerformanceMetrics
    neural_metrics: PerformanceMetrics
    better_implementation: ImplementationType
    improvement_factor: float
    timestamp: float = field(default_factory=time.time)


class HormoneNeuralModel(nn.Module):
    """Neural network model for hormone calculations."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None):
        """
        Initialize the hormone neural model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())  # Hormone levels are between 0 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)


class HormoneDataset(Dataset):
    """Dataset for hormone calculation training."""
    
    def __init__(self, examples: List[TrainingExample]):
        """
        Initialize the hormone dataset.
        
        Args:
            examples: List of training examples
        """
        self.examples = examples
        
        # Extract features and targets
        all_keys = set()
        for example in examples:
            all_keys.update(example.inputs.keys())
        
        self.feature_keys = sorted(list(all_keys))
        
        # Convert to tensors
        self.features = torch.zeros((len(examples), len(self.feature_keys)))
        self.targets = torch.zeros((len(examples), 1))
        
        for i, example in enumerate(examples):
            for j, key in enumerate(self.feature_keys):
                self.features[i, j] = example.inputs.get(key, 0.0)
            self.targets[i, 0] = example.output
    
    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Example index
            
        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]
    
    def get_feature_keys(self) -> List[str]:
        """
        Get the feature keys.
        
        Returns:
            List of feature keys
        """
        return self.feature_keys.copy()


class NeuralTrainingEngine:
    """
    Specialized training engine for hormone calculations.
    
    This class provides a specialized training engine for hormone calculations,
    with automatic performance comparison and model persistence.
    """
    
    def __init__(self, model_dir: str = "data/neural_models"):
        """
        Initialize the neural training engine.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = model_dir
        self.logger = logging.getLogger("NeuralTrainingEngine")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Models
        self.models: Dict[str, HormoneNeuralModel] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.feature_keys: Dict[str, List[str]] = {}
        
        # Training data
        self.training_data: Dict[str, List[TrainingExample]] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, Dict[str, PerformanceMetrics]] = {}
        self.comparison_results: Dict[str, List[ComparisonResult]] = {}
        
        # Fallback manager
        self.fallback_manager = FallbackManager()
        
        # Implementation switching monitor
        self.switching_monitor = ImplementationSwitchingMonitor()
        
        # Load existing models
        self._load_existing_models()
        
        self.logger.info("Neural training engine initialized")
    
    def _load_existing_models(self) -> None:
        """Load existing models from disk."""
        if not os.path.exists(self.model_dir):
            return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pt"):
                hormone_type = filename.split("_")[0]
                model_path = os.path.join(self.model_dir, filename)
                
                try:
                    # Load model
                    checkpoint = torch.load(model_path)
                    
                    # Create model
                    model = HormoneNeuralModel(
                        input_size=checkpoint["input_size"],
                        hidden_sizes=checkpoint["hidden_sizes"]
                    )
                    
                    # Load state
                    model.load_state_dict(checkpoint["model_state"])
                    
                    # Store model
                    self.models[hormone_type] = model
                    self.model_status[hormone_type] = ModelStatus.TRAINED
                    self.feature_keys[hormone_type] = checkpoint["feature_keys"]
                    
                    self.logger.info(f"Loaded model for hormone type: {hormone_type}")
                except Exception as e:
                    self.logger.error(f"Error loading model for hormone type {hormone_type}: {e}")
    
    def add_training_example(self, hormone_type: str, example: TrainingExample) -> None:
        """
        Add a training example for a hormone type.
        
        Args:
            hormone_type: Type of hormone
            example: Training example
        """
        if hormone_type not in self.training_data:
            self.training_data[hormone_type] = []
        
        self.training_data[hormone_type].append(example)
    
    def add_training_examples(self, hormone_type: str, examples: List[TrainingExample]) -> None:
        """
        Add multiple training examples for a hormone type.
        
        Args:
            hormone_type: Type of hormone
            examples: List of training examples
        """
        if hormone_type not in self.training_data:
            self.training_data[hormone_type] = []
        
        self.training_data[hormone_type].extend(examples)
    
    def get_training_examples(self, hormone_type: str) -> List[TrainingExample]:
        """
        Get training examples for a hormone type.
        
        Args:
            hormone_type: Type of hormone
            
        Returns:
            List of training examples
        """
        return self.training_data.get(hormone_type, []).copy()
    
    def clear_training_examples(self, hormone_type: str) -> None:
        """
        Clear training examples for a hormone type.
        
        Args:
            hormone_type: Type of hormone
        """
        if hormone_type in self.training_data:
            self.training_data[hormone_type] = []    

    async def train_hormone_calculator(self, 
                                     hormone_type: str, 
                                     config: Optional[TrainingConfiguration] = None) -> TrainingResult:
        """
        Train a neural model for hormone calculations.
        
        Args:
            hormone_type: Type of hormone
            config: Training configuration
            
        Returns:
            Training result
        """
        # Check if we have training data
        if hormone_type not in self.training_data or not self.training_data[hormone_type]:
            return TrainingResult(
                model_id=f"{hormone_type}_model",
                hormone_type=hormone_type,
                success=False,
                accuracy=0.0,
                training_time=0.0,
                epochs_completed=0,
                early_stopping=False,
                loss_history=[],
                val_loss_history=[],
                error_message="No training data available"
            )
        
        # Use default configuration if not provided
        if config is None:
            config = TrainingConfiguration()
        
        # Create dataset
        examples = self.training_data[hormone_type]
        dataset = HormoneDataset(examples)
        
        # Split into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )
        
        # Create model
        input_size = len(dataset.get_feature_keys())
        model = HormoneNeuralModel(input_size=input_size)
        
        # Set model status
        self.model_status[hormone_type] = ModelStatus.TRAINING
        
        # Train model
        start_time = time.time()
        
        try:
            # Create optimizer and loss function
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
            
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            loss_history = []
            val_loss_history = []
            
            for epoch in range(config.epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                loss_history.append(train_loss)
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_loss_history.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self._save_model(
                        hormone_type=hormone_type,
                        model=model,
                        feature_keys=dataset.get_feature_keys(),
                        input_size=input_size
                    )
                else:
                    patience_counter += 1
                    
                    if patience_counter >= 10:  # Early stopping patience
                        self.logger.info(
                            f"Early stopping at epoch {epoch} for hormone type {hormone_type}"
                        )
                        break
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                    )
            
            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                total_error = 0.0
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    total_error += torch.sum((outputs - batch_y) ** 2).item()
                
                mse = total_error / (len(val_dataset) * 1)  # 1 output dimension
                accuracy = 1.0 - min(1.0, mse)  # Convert MSE to accuracy (0-1)
            
            # Store model
            self.models[hormone_type] = model
            self.model_status[hormone_type] = ModelStatus.TRAINED
            self.feature_keys[hormone_type] = dataset.get_feature_keys()
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create result
            result = TrainingResult(
                model_id=f"{hormone_type}_model",
                hormone_type=hormone_type,
                success=True,
                accuracy=accuracy,
                training_time=training_time,
                epochs_completed=epoch + 1,
                early_stopping=patience_counter >= 10,
                loss_history=loss_history,
                val_loss_history=val_loss_history
            )
            
            self.logger.info(
                f"Successfully trained model for hormone type {hormone_type} "
                f"(accuracy: {accuracy:.4f}, time: {training_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.model_status[hormone_type] = ModelStatus.FAILED
            
            self.logger.error(f"Error training model for hormone type {hormone_type}: {e}")
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create result
            result = TrainingResult(
                model_id=f"{hormone_type}_model",
                hormone_type=hormone_type,
                success=False,
                accuracy=0.0,
                training_time=training_time,
                epochs_completed=0,
                early_stopping=False,
                loss_history=[],
                val_loss_history=[],
                error_message=str(e)
            )
            
            return result
    
    def _save_model(self, 
                   hormone_type: str, 
                   model: HormoneNeuralModel, 
                   feature_keys: List[str],
                   input_size: int) -> None:
        """
        Save a model to disk.
        
        Args:
            hormone_type: Type of hormone
            model: Neural model
            feature_keys: List of feature keys
            input_size: Input size
        """
        # Create checkpoint
        checkpoint = {
            "model_state": model.state_dict(),
            "feature_keys": feature_keys,
            "input_size": input_size,
            "hidden_sizes": [layer.out_features for layer in model.model if isinstance(layer, nn.Linear)][:-1],
            "timestamp": time.time()
        }
        
        # Save checkpoint
        model_path = os.path.join(self.model_dir, f"{hormone_type}_model.pt")
        torch.save(checkpoint, model_path)
        
        self.logger.info(f"Saved model for hormone type {hormone_type} to {model_path}")
    
    def load_model(self, hormone_type: str) -> Optional[HormoneNeuralModel]:
        """
        Load a model for a hormone type.
        
        Args:
            hormone_type: Type of hormone
            
        Returns:
            Neural model, or None if not found
        """
        if hormone_type in self.models:
            return self.models[hormone_type]
        
        # Try to load from disk
        model_path = os.path.join(self.model_dir, f"{hormone_type}_model.pt")
        
        if os.path.exists(model_path):
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path)
                
                # Create model
                model = HormoneNeuralModel(
                    input_size=checkpoint["input_size"],
                    hidden_sizes=checkpoint["hidden_sizes"]
                )
                
                # Load state
                model.load_state_dict(checkpoint["model_state"])
                
                # Store model
                self.models[hormone_type] = model
                self.model_status[hormone_type] = ModelStatus.TRAINED
                self.feature_keys[hormone_type] = checkpoint["feature_keys"]
                
                self.logger.info(f"Loaded model for hormone type {hormone_type}")
                
                return model
            except Exception as e:
                self.logger.error(f"Error loading model for hormone type {hormone_type}: {e}")
                return None
        
        return None 
   
    def get_model_status(self, hormone_type: str) -> Optional[ModelStatus]:
        """
        Get the status of a model.
        
        Args:
            hormone_type: Type of hormone
            
        Returns:
            Model status, or None if not found
        """
        return self.model_status.get(hormone_type)
    
    def get_all_model_statuses(self) -> Dict[str, ModelStatus]:
        """
        Get the status of all models.
        
        Returns:
            Dictionary mapping hormone types to model statuses
        """
        return self.model_status.copy()
    
    def predict(self, hormone_type: str, inputs: Dict[str, float]) -> Optional[float]:
        """
        Make a prediction using a trained model.
        
        Args:
            hormone_type: Type of hormone
            inputs: Input features
            
        Returns:
            Predicted hormone level, or None if model not found
        """
        # Check if model exists
        model = self.load_model(hormone_type)
        
        if model is None:
            return None
        
        # Check if we can use neural implementation
        if not self.fallback_manager.can_use_neural(f"hormone_{hormone_type}"):
            return None
        
        # Get feature keys
        feature_keys = self.feature_keys.get(hormone_type, [])
        
        if not feature_keys:
            return None
        
        try:
            # Create input tensor
            input_tensor = torch.zeros(len(feature_keys))
            
            for i, key in enumerate(feature_keys):
                input_tensor[i] = inputs.get(key, 0.0)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                
                # Record success
                self.fallback_manager.record_success(f"hormone_{hormone_type}")
                
                return output.item()
        except Exception as e:
            # Record failure
            self.fallback_manager.record_failure(f"hormone_{hormone_type}", e)
            
            self.logger.error(f"Error making prediction for hormone type {hormone_type}: {e}")
            return None
    
    def compare_implementations(self, 
                              hormone_type: str, 
                              algorithmic_impl: Callable[[Dict[str, float]], float],
                              test_data: List[Dict[str, float]]) -> ComparisonResult:
        """
        Compare algorithmic and neural implementations.
        
        Args:
            hormone_type: Type of hormone
            algorithmic_impl: Algorithmic implementation function
            test_data: List of input data for testing
            
        Returns:
            Comparison result
        """
        # Check if model exists
        model = self.load_model(hormone_type)
        
        if model is None:
            # Neural implementation not available
            return ComparisonResult(
                hormone_type=hormone_type,
                algorithmic_metrics=PerformanceMetrics(
                    accuracy=1.0,
                    speed=1.0,
                    resource_usage=0.1,
                    error_rate=0.0,
                    confidence_score=1.0
                ),
                neural_metrics=PerformanceMetrics(
                    accuracy=0.0,
                    speed=0.0,
                    resource_usage=0.0,
                    error_rate=1.0,
                    confidence_score=0.0
                ),
                better_implementation=ImplementationType.ALGORITHMIC,
                improvement_factor=float('inf')
            )
        
        # Test algorithmic implementation
        algo_start_time = time.time()
        algo_results = []
        algo_errors = 0
        
        for inputs in test_data:
            try:
                result = algorithmic_impl(inputs)
                algo_results.append(result)
            except Exception:
                algo_errors += 1
                algo_results.append(None)
        
        algo_time = time.time() - algo_start_time
        
        # Test neural implementation
        neural_start_time = time.time()
        neural_results = []
        neural_errors = 0
        
        for inputs in test_data:
            try:
                result = self.predict(hormone_type, inputs)
                neural_results.append(result)
            except Exception:
                neural_errors += 1
                neural_results.append(None)
        
        neural_time = time.time() - neural_start_time
        
        # Calculate metrics
        algo_speed = len(test_data) / algo_time if algo_time > 0 else 0.0
        neural_speed = len(test_data) / neural_time if neural_time > 0 else 0.0
        
        algo_error_rate = algo_errors / len(test_data) if test_data else 0.0
        neural_error_rate = neural_errors / len(test_data) if test_data else 0.0
        
        # Calculate accuracy (how close neural results are to algorithmic results)
        accuracy_sum = 0.0
        accuracy_count = 0
        
        for algo_result, neural_result in zip(algo_results, neural_results):
            if algo_result is not None and neural_result is not None:
                # Calculate relative error
                if abs(algo_result) > 1e-6:
                    relative_error = abs(neural_result - algo_result) / abs(algo_result)
                    accuracy = 1.0 - min(1.0, relative_error)
                else:
                    accuracy = 1.0 - min(1.0, abs(neural_result - algo_result))
                
                accuracy_sum += accuracy
                accuracy_count += 1
        
        neural_accuracy = accuracy_sum / accuracy_count if accuracy_count > 0 else 0.0
        
        # Create metrics
        algo_metrics = PerformanceMetrics(
            accuracy=1.0,  # Algorithmic implementation is the reference
            speed=algo_speed,
            resource_usage=0.1,  # Assume low resource usage
            error_rate=algo_error_rate,
            confidence_score=1.0
        )
        
        neural_metrics = PerformanceMetrics(
            accuracy=neural_accuracy,
            speed=neural_speed,
            resource_usage=0.3,  # Assume higher resource usage
            error_rate=neural_error_rate,
            confidence_score=0.9
        )
        
        # Calculate weighted scores
        weights = {
            "accuracy": 0.4,
            "speed": 0.3,
            "resource_usage": 0.1,
            "error_rate": 0.1,
            "confidence_score": 0.1
        }
        
        algo_score = algo_metrics.weighted_score(weights)
        neural_score = neural_metrics.weighted_score(weights)
        
        # Determine better implementation
        if neural_score > algo_score:
            better_impl = ImplementationType.NEURAL
            improvement_factor = neural_score / algo_score if algo_score > 0 else 1.5
        else:
            better_impl = ImplementationType.ALGORITHMIC
            improvement_factor = algo_score / neural_score if neural_score > 0 else 1.5
        
        # Create comparison result
        result = ComparisonResult(
            hormone_type=hormone_type,
            algorithmic_metrics=algo_metrics,
            neural_metrics=neural_metrics,
            better_implementation=better_impl,
            improvement_factor=improvement_factor
        )
        
        # Store comparison result
        if hormone_type not in self.comparison_results:
            self.comparison_results[hormone_type] = []
        
        self.comparison_results[hormone_type].append(result)
        
        # Store performance metrics
        if hormone_type not in self.performance_metrics:
            self.performance_metrics[hormone_type] = {}
        
        self.performance_metrics[hormone_type]["algorithmic"] = algo_metrics
        self.performance_metrics[hormone_type]["neural"] = neural_metrics
        
        # Record switch event if needed
        component_name = f"hormone_{hormone_type}"
        
        self.switching_monitor.record_switch_event({
            "component": component_name,
            "from_implementation": ImplementationType.ALGORITHMIC,
            "to_implementation": better_impl,
            "reason": "Performance comparison",
            "performance_improvement": improvement_factor
        })
        
        return result    

    def should_switch_implementation(self, hormone_type: str) -> bool:
        """
        Determine if the implementation should be switched.
        
        Args:
            hormone_type: Type of hormone
            
        Returns:
            True if the neural implementation should be used, False otherwise
        """
        # Check if we have comparison results
        if hormone_type not in self.comparison_results or not self.comparison_results[hormone_type]:
            return False
        
        # Get latest comparison result
        latest_result = self.comparison_results[hormone_type][-1]
        
        # Check if neural implementation is better
        if latest_result.better_implementation == ImplementationType.NEURAL:
            # Check if improvement is significant
            if latest_result.improvement_factor >= 1.1:  # 10% improvement
                # Check if we can use neural implementation
                if self.fallback_manager.can_use_neural(f"hormone_{hormone_type}"):
                    return True
        
        return False
    
    def get_comparison_results(self, hormone_type: str) -> List[ComparisonResult]:
        """
        Get comparison results for a hormone type.
        
        Args:
            hormone_type: Type of hormone
            
        Returns:
            List of comparison results
        """
        return self.comparison_results.get(hormone_type, []).copy()
    
    def get_performance_metrics(self, hormone_type: str) -> Dict[str, PerformanceMetrics]:
        """
        Get performance metrics for a hormone type.
        
        Args:
            hormone_type: Type of hormone
            
        Returns:
            Dictionary mapping implementation types to performance metrics
        """
        return self.performance_metrics.get(hormone_type, {}).copy()
    
    def save_improved_model(self, hormone_type: str, model: HormoneNeuralModel) -> bool:
        """
        Save an improved model.
        
        Args:
            hormone_type: Type of hormone
            model: Neural model
            
        Returns:
            True if the model was saved, False otherwise
        """
        # Check if we have feature keys
        if hormone_type not in self.feature_keys:
            return False
        
        feature_keys = self.feature_keys[hormone_type]
        input_size = len(feature_keys)
        
        try:
            # Save model
            self._save_model(
                hormone_type=hormone_type,
                model=model,
                feature_keys=feature_keys,
                input_size=input_size
            )
            
            # Update model
            self.models[hormone_type] = model
            self.model_status[hormone_type] = ModelStatus.TRAINED
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving improved model for hormone type {hormone_type}: {e}")
            return False
    
    def load_pretrained_models(self) -> Dict[str, HormoneNeuralModel]:
        """
        Load all pretrained models.
        
        Returns:
            Dictionary mapping hormone types to neural models
        """
        # Check model directory
        if not os.path.exists(self.model_dir):
            return {}
        
        # Load all models
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pt"):
                hormone_type = filename.split("_")[0]
                self.load_model(hormone_type)
        
        return self.models.copy()
    
    async def train_incrementally(self, 
                                hormone_type: str, 
                                new_examples: List[TrainingExample],
                                config: Optional[TrainingConfiguration] = None) -> TrainingResult:
        """
        Train a model incrementally with new examples.
        
        Args:
            hormone_type: Type of hormone
            new_examples: New training examples
            config: Training configuration
            
        Returns:
            Training result
        """
        # Add new examples to training data
        self.add_training_examples(hormone_type, new_examples)
        
        # Train model
        return await self.train_hormone_calculator(hormone_type, config)
    
    async def background_training_loop(self, interval: float = 3600.0) -> None:
        """
        Run a background training loop that periodically trains models.
        
        Args:
            interval: Training interval in seconds
        """
        self.logger.info(f"Starting background training loop with interval {interval}s")
        
        while True:
            try:
                # Get hormone types with training data
                hormone_types = [
                    hormone_type for hormone_type, examples in self.training_data.items()
                    if examples
                ]
                
                for hormone_type in hormone_types:
                    # Check if model needs training
                    status = self.get_model_status(hormone_type)
                    
                    if status in [None, ModelStatus.NOT_TRAINED, ModelStatus.FAILED]:
                        self.logger.info(f"Training model for hormone type {hormone_type} in background")
                        
                        # Train model
                        await self.train_hormone_calculator(hormone_type)
                
                # Sleep for the specified interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in background training loop: {e}")
                await asyncio.sleep(interval)


# Example usage
async def test_neural_training_engine():
    """Test the neural training engine."""
    import random
    
    # Create engine
    engine = NeuralTrainingEngine(model_dir="data/test_neural_models")
    
    # Create training examples
    examples = []
    
    for _ in range(100):
        # Generate random inputs
        inputs = {
            "input1": random.random(),
            "input2": random.random(),
            "input3": random.random()
        }
        
        # Calculate output (simple function for testing)
        output = 0.3 * inputs["input1"] + 0.5 * inputs["input2"] + 0.2 * inputs["input3"]
        
        # Create example
        example = TrainingExample(inputs=inputs, output=output)
        examples.append(example)
    
    # Add training examples
    engine.add_training_examples("test_hormone", examples)
    
    # Train model
    result = await engine.train_hormone_calculator("test_hormone")
    
    print("\nTraining Result:")
    print(f"Success: {result.success}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Epochs completed: {result.epochs_completed}")
    
    # Test prediction
    test_inputs = {
        "input1": 0.7,
        "input2": 0.3,
        "input3": 0.5
    }
    
    expected_output = 0.3 * test_inputs["input1"] + 0.5 * test_inputs["input2"] + 0.2 * test_inputs["input3"]
    predicted_output = engine.predict("test_hormone", test_inputs)
    
    print("\nPrediction Test:")
    print(f"Inputs: {test_inputs}")
    print(f"Expected output: {expected_output:.4f}")
    print(f"Predicted output: {predicted_output:.4f}")
    print(f"Error: {abs(predicted_output - expected_output):.4f}")
    
    # Compare implementations
    def algorithmic_impl(inputs):
        return 0.3 * inputs["input1"] + 0.5 * inputs["input2"] + 0.2 * inputs["input3"]
    
    test_data = []
    for _ in range(10):
        test_data.append({
            "input1": random.random(),
            "input2": random.random(),
            "input3": random.random()
        })
    
    comparison = engine.compare_implementations("test_hormone", algorithmic_impl, test_data)
    
    print("\nComparison Result:")
    print(f"Better implementation: {comparison.better_implementation.value}")
    print(f"Improvement factor: {comparison.improvement_factor:.2f}")
    print(f"Algorithmic metrics: {comparison.algorithmic_metrics}")
    print(f"Neural metrics: {comparison.neural_metrics}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_neural_training_engine())