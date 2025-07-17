"""
Training Engine for Neural Network Models component.

This module provides incremental training functionality for neural network models,
including training data collection, background training, and performance monitoring.
It implements requirements 1.7 and 6.3 for incremental and background training.
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np


class TrainingStatus(Enum):
    """Training status enumeration."""
    IDLE = "idle"
    COLLECTING = "collecting"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingData:
    """Training data point for neural network models."""
    function_name: str
    input_data: Any
    expected_output: Any
    algorithmic_output: Any
    timestamp: datetime
    context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "algorithmic_output": self.algorithmic_output,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingData':
        """Create from dictionary."""
        return cls(
            function_name=data["function_name"],
            input_data=data["input_data"],
            expected_output=data["expected_output"],
            algorithmic_output=data["algorithmic_output"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data["context"],
            performance_metrics=data["performance_metrics"]
        )


@dataclass
class TrainingResult:
    """Result of a training session."""
    function_name: str
    status: TrainingStatus
    training_time: float
    samples_processed: int
    performance_improvement: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class TrainingDataCollector:
    """
    Collects training data from algorithmic implementations.
    
    This class monitors algorithmic implementations and collects input/output
    pairs along with performance metrics for training neural network alternatives.
    """
    
    def __init__(self, max_samples_per_function: int = 10000):
        """
        Initialize the training data collector.
        
        Args:
            max_samples_per_function: Maximum number of samples to keep per function.
        """
        self.logger = logging.getLogger("TrainingDataCollector")
        self.max_samples_per_function = max_samples_per_function
        self.training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_function))
        self.collection_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()
        
        # Initialize collection statistics
        self._initialize_stats()
    
    def _initialize_stats(self):
        """Initialize collection statistics."""
        for function_name in self.training_data.keys():
            self.collection_stats[function_name] = {
                "total_samples": 0,
                "samples_today": 0,
                "last_collection": None,
                "average_performance": 0.0,
                "collection_rate": 0.0
            }
    
    def collect_sample(self, 
                      function_name: str,
                      input_data: Any,
                      expected_output: Any,
                      algorithmic_output: Any,
                      performance_metrics: Dict[str, float],
                      context: Dict[str, Any] = None) -> bool:
        """
        Collect a training sample from an algorithmic implementation.
        
        Args:
            function_name: Name of the function being sampled.
            input_data: Input data for the function.
            expected_output: Expected output (ground truth if available).
            algorithmic_output: Output from the algorithmic implementation.
            performance_metrics: Performance metrics for this sample.
            context: Additional context information.
            
        Returns:
            True if the sample was collected successfully, False otherwise.
        """
        try:
            with self._lock:
                # Create training data point
                training_point = TrainingData(
                    function_name=function_name,
                    input_data=input_data,
                    expected_output=expected_output,
                    algorithmic_output=algorithmic_output,
                    timestamp=datetime.now(),
                    context=context or {},
                    performance_metrics=performance_metrics
                )
                
                # Add to collection
                self.training_data[function_name].append(training_point)
                
                # Update statistics
                stats = self.collection_stats[function_name]
                stats["total_samples"] += 1
                stats["last_collection"] = datetime.now()
                
                # Update daily count
                today = datetime.now().date()
                if stats.get("last_date") != today:
                    stats["samples_today"] = 1
                    stats["last_date"] = today
                else:
                    stats["samples_today"] += 1
                
                # Update average performance
                if "accuracy" in performance_metrics:
                    current_avg = stats.get("average_performance", 0.0)
                    total_samples = stats["total_samples"]
                    new_avg = ((current_avg * (total_samples - 1)) + performance_metrics["accuracy"]) / total_samples
                    stats["average_performance"] = new_avg
                
                # Calculate collection rate (samples per hour)
                if stats["total_samples"] > 1:
                    first_sample_time = self.training_data[function_name][0].timestamp
                    time_diff = (datetime.now() - first_sample_time).total_seconds() / 3600  # hours
                    stats["collection_rate"] = stats["total_samples"] / max(time_diff, 0.001)
                
                self.logger.debug(f"Collected training sample for {function_name}. "
                                f"Total samples: {stats['total_samples']}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error collecting training sample for {function_name}: {e}")
            return False
    
    def get_training_data(self, function_name: str, max_samples: int = None) -> List[TrainingData]:
        """
        Get training data for a specific function.
        
        Args:
            function_name: Name of the function.
            max_samples: Maximum number of samples to return.
            
        Returns:
            List of training data points.
        """
        with self._lock:
            if function_name not in self.training_data:
                return []
            
            data = list(self.training_data[function_name])
            
            if max_samples and len(data) > max_samples:
                # Return most recent samples
                data = data[-max_samples:]
            
            return data
    
    def get_collection_stats(self, function_name: str = None) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            function_name: Specific function name, or None for all functions.
            
        Returns:
            Dictionary of collection statistics.
        """
        with self._lock:
            if function_name:
                return self.collection_stats.get(function_name, {})
            else:
                return dict(self.collection_stats)
    
    def clear_training_data(self, function_name: str) -> bool:
        """
        Clear training data for a specific function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            True if data was cleared successfully, False otherwise.
        """
        try:
            with self._lock:
                if function_name in self.training_data:
                    self.training_data[function_name].clear()
                    self.collection_stats[function_name] = {
                        "total_samples": 0,
                        "samples_today": 0,
                        "last_collection": None,
                        "average_performance": 0.0,
                        "collection_rate": 0.0
                    }
                    self.logger.info(f"Cleared training data for {function_name}")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error clearing training data for {function_name}: {e}")
            return False
    
    def save_training_data(self, function_name: str, filepath: str) -> bool:
        """
        Save training data to a file.
        
        Args:
            function_name: Name of the function.
            filepath: Path to save the data.
            
        Returns:
            True if data was saved successfully, False otherwise.
        """
        try:
            with self._lock:
                data = self.get_training_data(function_name)
                
                # Convert to serializable format
                serializable_data = [point.to_dict() for point in data]
                
                # Save to file
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    json.dump({
                        "function_name": function_name,
                        "saved_at": datetime.now().isoformat(),
                        "sample_count": len(serializable_data),
                        "data": serializable_data
                    }, f, indent=2)
                
                self.logger.info(f"Saved {len(serializable_data)} training samples for {function_name} to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving training data for {function_name}: {e}")
            return False
    
    def load_training_data(self, function_name: str, filepath: str) -> bool:
        """
        Load training data from a file.
        
        Args:
            function_name: Name of the function.
            filepath: Path to load the data from.
            
        Returns:
            True if data was loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Training data file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            # Validate data format
            if saved_data.get("function_name") != function_name:
                self.logger.error(f"Function name mismatch in saved data: "
                                f"expected {function_name}, got {saved_data.get('function_name')}")
                return False
            
            with self._lock:
                # Clear existing data
                self.training_data[function_name].clear()
                
                # Load data points
                for point_data in saved_data.get("data", []):
                    training_point = TrainingData.from_dict(point_data)
                    self.training_data[function_name].append(training_point)
                
                # Update statistics
                self.collection_stats[function_name]["total_samples"] = len(saved_data.get("data", []))
                
                self.logger.info(f"Loaded {len(saved_data.get('data', []))} training samples for {function_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading training data for {function_name}: {e}")
            return False


class BackgroundTrainer:
    """
    Handles background training of neural network models.
    
    This class implements background training functionality that runs during
    idle periods to train neural network alternatives using collected data.
    """
    
    def __init__(self, 
                 data_collector: TrainingDataCollector,
                 min_samples_for_training: int = 100,
                 training_interval: int = 300,  # 5 minutes
                 max_training_time: int = 1800):  # 30 minutes
        """
        Initialize the background trainer.
        
        Args:
            data_collector: Training data collector instance.
            min_samples_for_training: Minimum samples needed before training.
            training_interval: Interval between training sessions (seconds).
            max_training_time: Maximum time for a single training session (seconds).
        """
        self.logger = logging.getLogger("BackgroundTrainer")
        self.data_collector = data_collector
        self.min_samples_for_training = min_samples_for_training
        self.training_interval = training_interval
        self.max_training_time = max_training_time
        
        # Training state
        self.training_status: Dict[str, TrainingStatus] = defaultdict(lambda: TrainingStatus.IDLE)
        self.training_results: Dict[str, List[TrainingResult]] = defaultdict(list)
        self.last_training_time: Dict[str, datetime] = {}
        
        # Background training control
        self.is_running = False
        self.training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Resource monitoring
        self.system_idle_threshold = 0.3  # CPU usage below 30% considered idle
        self.memory_usage_threshold = 0.8  # Don't train if memory usage above 80%
    
    def start_background_training(self) -> bool:
        """
        Start background training thread.
        
        Returns:
            True if background training was started successfully, False otherwise.
        """
        if self.is_running:
            self.logger.warning("Background training is already running")
            return False
        
        try:
            self.is_running = True
            self._stop_event.clear()
            self.training_thread = threading.Thread(target=self._background_training_loop, daemon=True)
            self.training_thread.start()
            
            self.logger.info("Started background training thread")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting background training: {e}")
            self.is_running = False
            return False
    
    def stop_background_training(self) -> bool:
        """
        Stop background training thread.
        
        Returns:
            True if background training was stopped successfully, False otherwise.
        """
        if not self.is_running:
            return True
        
        try:
            self._stop_event.set()
            self.is_running = False
            
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=10)
            
            self.logger.info("Stopped background training thread")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping background training: {e}")
            return False
    
    def _background_training_loop(self):
        """Main background training loop."""
        self.logger.info("Background training loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check if system is idle and resources are available
                if self._should_train():
                    # Get functions that need training
                    functions_to_train = self._get_functions_needing_training()
                    
                    for function_name in functions_to_train:
                        if self._stop_event.is_set():
                            break
                        
                        # Train the function
                        self._train_function(function_name)
                
                # Wait for next training interval
                self._stop_event.wait(self.training_interval)
                
            except Exception as e:
                self.logger.error(f"Error in background training loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        self.logger.info("Background training loop stopped")
    
    def _should_train(self) -> bool:
        """
        Check if the system should perform background training.
        
        Returns:
            True if training should proceed, False otherwise.
        """
        try:
            # Check system resource usage
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            
            # Only train if system is relatively idle
            if cpu_usage > self.system_idle_threshold:
                self.logger.debug(f"System not idle (CPU: {cpu_usage:.2f}), skipping training")
                return False
            
            if memory_usage > self.memory_usage_threshold:
                self.logger.debug(f"Memory usage too high ({memory_usage:.2f}), skipping training")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return False
    
    def _get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage as a float between 0.0 and 1.0.
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=1) / 100.0
        except ImportError:
            # If psutil is not available, assume moderate usage
            self.logger.debug("psutil not available, assuming moderate CPU usage")
            return 0.5
        except Exception as e:
            self.logger.warning(f"Error getting CPU usage: {e}")
            return 0.5
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage percentage.
        
        Returns:
            Memory usage as a float between 0.0 and 1.0.
        """
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # If psutil is not available, assume moderate usage
            self.logger.debug("psutil not available, assuming moderate memory usage")
            return 0.5
        except Exception as e:
            self.logger.warning(f"Error getting memory usage: {e}")
            return 0.5
    
    def _get_functions_needing_training(self) -> List[str]:
        """
        Get list of functions that need training.
        
        Returns:
            List of function names that need training.
        """
        functions_to_train = []
        
        # Get all functions with collected data
        stats = self.data_collector.get_collection_stats()
        
        for function_name, function_stats in stats.items():
            # Check if function has enough samples
            if function_stats.get("total_samples", 0) < self.min_samples_for_training:
                continue
            
            # Check if function hasn't been trained recently
            last_training = self.last_training_time.get(function_name)
            if last_training:
                time_since_training = datetime.now() - last_training
                if time_since_training.total_seconds() < self.training_interval:
                    continue
            
            # Check if function is not currently being trained
            if self.training_status[function_name] in [TrainingStatus.TRAINING, TrainingStatus.EVALUATING]:
                continue
            
            functions_to_train.append(function_name)
        
        return functions_to_train    
def _train_function(self, function_name: str) -> TrainingResult:
        """
        Train a neural model for a specific function.
        
        Args:
            function_name: Name of the function to train.
            
        Returns:
            Training result object.
        """
        self.logger.info(f"Starting training for {function_name}")
        
        # Update training status
        self.training_status[function_name] = TrainingStatus.TRAINING
        start_time = time.time()
        
        try:
            # Get training data
            training_data = self.data_collector.get_training_data(function_name)
            if not training_data:
                self.logger.warning(f"No training data available for {function_name}")
                self.training_status[function_name] = TrainingStatus.IDLE
                return TrainingResult(
                    function_name=function_name,
                    status=TrainingStatus.FAILED,
                    training_time=0.0,
                    samples_processed=0,
                    performance_improvement=0.0,
                    error_message="No training data available"
                )
            
            # Prepare training data
            inputs = [td.input_data for td in training_data]
            expected_outputs = [td.expected_output for td in training_data]
            
            # Get model from model manager (would be injected in a real implementation)
            model = self._get_model(function_name)
            if not model:
                self.logger.error(f"Could not get model for {function_name}")
                self.training_status[function_name] = TrainingStatus.IDLE
                return TrainingResult(
                    function_name=function_name,
                    status=TrainingStatus.FAILED,
                    training_time=0.0,
                    samples_processed=0,
                    performance_improvement=0.0,
                    error_message="Could not get model"
                )
            
            # Get baseline performance before training
            baseline_performance = self._evaluate_model(model, function_name, inputs, expected_outputs)
            
            # Train the model
            self.logger.info(f"Training model for {function_name} with {len(training_data)} samples")
            training_metrics = self._perform_training(model, function_name, inputs, expected_outputs)
            
            # Check if we've exceeded max training time
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_training_time:
                self.logger.warning(f"Training for {function_name} exceeded max time ({elapsed_time:.1f}s)")
            
            # Evaluate model after training
            self.training_status[function_name] = TrainingStatus.EVALUATING
            post_performance = self._evaluate_model(model, function_name, inputs, expected_outputs)
            
            # Calculate performance improvement
            performance_improvement = post_performance.get("accuracy", 0.0) - baseline_performance.get("accuracy", 0.0)
            
            # Save model if improved
            if performance_improvement > 0.01:  # 1% improvement threshold
                self._save_model(model, function_name)
                self.logger.info(f"Model for {function_name} improved by {performance_improvement:.4f}, saved")
            else:
                self.logger.info(f"Model for {function_name} did not improve significantly ({performance_improvement:.4f})")
            
            # Update last training time
            self.last_training_time[function_name] = datetime.now()
            
            # Create training result
            training_time = time.time() - start_time
            result = TrainingResult(
                function_name=function_name,
                status=TrainingStatus.COMPLETED,
                training_time=training_time,
                samples_processed=len(training_data),
                performance_improvement=performance_improvement,
                metrics=post_performance
            )
            
            # Add to training results history
            self.training_results[function_name].append(result)
            
            # Limit history size
            if len(self.training_results[function_name]) > 10:
                self.training_results[function_name].pop(0)
            
            self.logger.info(f"Completed training for {function_name} in {training_time:.1f}s with "
                           f"{performance_improvement:.4f} improvement")
            
            # Reset status
            self.training_status[function_name] = TrainingStatus.IDLE
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error training model for {function_name}: {e}")
            
            # Reset status
            self.training_status[function_name] = TrainingStatus.IDLE
            
            # Create failure result
            result = TrainingResult(
                function_name=function_name,
                status=TrainingStatus.FAILED,
                training_time=elapsed_time,
                samples_processed=0,
                performance_improvement=0.0,
                error_message=str(e)
            )
            
            # Add to training results history
            self.training_results[function_name].append(result)
            
            return result
    
    def _get_model(self, function_name: str) -> Any:
        """
        Get a model for a specific function.
        
        In a real implementation, this would use the ModelManager.
        For now, we create a dummy model for demonstration.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Model object, or None if not available.
        """
        try:
            # This is a placeholder for demonstration
            # In a real implementation, this would get the model from ModelManager
            
            # Create a dummy model
            model = {
                "function_name": function_name,
                "type": "neural",
                "version": "1.0.0",
                "weights": [0.5, 0.5, 0.5],  # Dummy weights
                "bias": 0.1  # Dummy bias
            }
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error getting model for {function_name}: {e}")
            return None
    
    def _evaluate_model(self, model: Any, function_name: str, inputs: List[Any], expected_outputs: List[Any]) -> Dict[str, float]:
        """
        Evaluate a model's performance.
        
        Args:
            model: Model to evaluate.
            function_name: Name of the function.
            inputs: List of input data.
            expected_outputs: List of expected outputs.
            
        Returns:
            Dictionary of performance metrics.
        """
        try:
            # This is a placeholder for demonstration
            # In a real implementation, this would use actual model evaluation
            
            # Simulate evaluation with random metrics
            import random
            
            # Generate a somewhat realistic accuracy based on model "version"
            base_accuracy = 0.7  # Start with a reasonable base accuracy
            version_bonus = 0.0
            if hasattr(model, "version"):
                version_str = str(model.get("version", "1.0.0"))
                version_parts = version_str.split(".")
                if len(version_parts) >= 2:
                    try:
                        minor_version = int(version_parts[1])
                        version_bonus = min(0.2, minor_version * 0.02)  # Up to 0.2 bonus for version
                    except ValueError:
                        pass
            
            # Add some randomness
            accuracy = min(0.99, base_accuracy + version_bonus + random.uniform(-0.05, 0.05))
            
            # Generate other metrics
            latency = random.uniform(10, 100)  # milliseconds
            resource_usage = random.uniform(0.1, 0.5)  # 0.0-1.0 scale
            
            return {
                "accuracy": accuracy,
                "latency": latency,
                "resource_usage": resource_usage,
                "samples_evaluated": len(inputs)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model for {function_name}: {e}")
            return {"accuracy": 0.0, "error": str(e)}
    
    def _perform_training(self, model: Any, function_name: str, inputs: List[Any], expected_outputs: List[Any]) -> Dict[str, float]:
        """
        Perform model training.
        
        Args:
            model: Model to train.
            function_name: Name of the function.
            inputs: List of input data.
            expected_outputs: List of expected outputs.
            
        Returns:
            Dictionary of training metrics.
        """
        try:
            # This is a placeholder for demonstration
            # In a real implementation, this would use actual model training
            
            # Simulate training time based on data size
            training_time = 0.01 * len(inputs)
            time.sleep(min(training_time, 1.0))  # Sleep for a bit to simulate training, max 1 second
            
            # Simulate training by updating model weights
            if "weights" in model:
                import random
                # Update weights with small random changes
                model["weights"] = [w + random.uniform(-0.1, 0.1) for w in model["weights"]]
                
            # Simulate version update
            if "version" in model:
                version_parts = model["version"].split(".")
                if len(version_parts) >= 3:
                    patch = int(version_parts[2]) + 1
                    model["version"] = f"{version_parts[0]}.{version_parts[1]}.{patch}"
            
            # Return metrics
            return {
                "epochs": 10,
                "training_time": training_time,
                "samples_trained": len(inputs),
                "loss": 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Error training model for {function_name}: {e}")
            return {"error": str(e)}
    
    def _save_model(self, model: Any, function_name: str) -> bool:
        """
        Save a trained model.
        
        Args:
            model: Model to save.
            function_name: Name of the function.
            
        Returns:
            True if the model was saved successfully, False otherwise.
        """
        try:
            # This is a placeholder for demonstration
            # In a real implementation, this would use ModelManager.save_model
            
            self.logger.info(f"Saved model for {function_name} (simulated)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model for {function_name}: {e}")
            return False
    
    def get_training_status(self, function_name: str) -> TrainingStatus:
        """
        Get the current training status for a function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Current training status.
        """
        return self.training_status.get(function_name, TrainingStatus.IDLE)
    
    def get_training_results(self, function_name: str) -> List[TrainingResult]:
        """
        Get training results for a function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            List of training results, most recent first.
        """
        return list(reversed(self.training_results.get(function_name, [])))
    
    def force_train_function(self, function_name: str) -> TrainingResult:
        """
        Force training for a specific function, regardless of timing or status.
        
        Args:
            function_name: Name of the function to train.
            
        Returns:
            Training result.
        """
        # Check if we have data for this function
        data = self.data_collector.get_training_data(function_name)
        if not data:
            self.logger.warning(f"No training data available for {function_name}")
            return TrainingResult(
                function_name=function_name,
                status=TrainingStatus.FAILED,
                training_time=0.0,
                samples_processed=0,
                performance_improvement=0.0,
                error_message="No training data available"
            )
        
        # Check if already training
        if self.training_status.get(function_name) == TrainingStatus.TRAINING:
            self.logger.warning(f"Already training {function_name}")
            return TrainingResult(
                function_name=function_name,
                status=TrainingStatus.FAILED,
                training_time=0.0,
                samples_processed=0,
                performance_improvement=0.0,
                error_message="Already training"
            )
        
        # Train the function
        return self._train_function(function_name)

cl
ass IncrementalTrainingSystem:
    """
    Integrates training data collection and background training for neural network models.
    
    This class provides a complete incremental training system that collects training data
    from algorithmic implementations and uses it to train neural network alternatives
    during idle periods.
    """
    
    def __init__(self, model_manager=None, data_dir: str = None):
        """
        Initialize the incremental training system.
        
        Args:
            model_manager: ModelManager instance, or None to create a new one.
            data_dir: Directory for storing training data, or None for default.
        """
        self.logger = logging.getLogger("IncrementalTrainingSystem")
        
        # Set up data directory
        self.data_dir = data_dir or os.path.join(os.getcwd(), "training_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create components
        self.data_collector = TrainingDataCollector(max_samples_per_function=10000)
        self.background_trainer = BackgroundTrainer(
            data_collector=self.data_collector,
            min_samples_for_training=100,
            training_interval=300,  # 5 minutes
            max_training_time=1800  # 30 minutes
        )
        
        # Store model manager reference
        self.model_manager = model_manager
        
        # Load any existing training data
        self._load_saved_training_data()
        
        self.logger.info("IncrementalTrainingSystem initialized")
    
    def _load_saved_training_data(self):
        """Load any saved training data from disk."""
        try:
            if not os.path.exists(self.data_dir):
                return
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json"):
                    function_name = filename.split(".")[0]
                    filepath = os.path.join(self.data_dir, filename)
                    
                    if self.data_collector.load_training_data(function_name, filepath):
                        self.logger.info(f"Loaded training data for {function_name} from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading saved training data: {e}")
    
    def start(self):
        """Start the incremental training system."""
        # Start background training
        self.background_trainer.start_background_training()
        self.logger.info("Started incremental training system")
    
    def stop(self):
        """Stop the incremental training system."""
        # Stop background training
        self.background_trainer.stop_background_training()
        
        # Save all training data
        self._save_all_training_data()
        
        self.logger.info("Stopped incremental training system")
    
    def _save_all_training_data(self):
        """Save all training data to disk."""
        try:
            stats = self.data_collector.get_collection_stats()
            
            for function_name in stats.keys():
                filepath = os.path.join(self.data_dir, f"{function_name}.json")
                if self.data_collector.save_training_data(function_name, filepath):
                    self.logger.info(f"Saved training data for {function_name} to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
    
    def collect_sample(self, 
                      function_name: str,
                      input_data: Any,
                      expected_output: Any,
                      algorithmic_output: Any,
                      performance_metrics: Dict[str, float],
                      context: Dict[str, Any] = None) -> bool:
        """
        Collect a training sample from an algorithmic implementation.
        
        Args:
            function_name: Name of the function being sampled.
            input_data: Input data for the function.
            expected_output: Expected output (ground truth if available).
            algorithmic_output: Output from the algorithmic implementation.
            performance_metrics: Performance metrics for this sample.
            context: Additional context information.
            
        Returns:
            True if the sample was collected successfully, False otherwise.
        """
        return self.data_collector.collect_sample(
            function_name=function_name,
            input_data=input_data,
            expected_output=expected_output,
            algorithmic_output=algorithmic_output,
            performance_metrics=performance_metrics,
            context=context
        )
    
    def force_train_function(self, function_name: str) -> TrainingResult:
        """
        Force training for a specific function, regardless of timing or status.
        
        Args:
            function_name: Name of the function to train.
            
        Returns:
            Training result.
        """
        return self.background_trainer.force_train_function(function_name)
    
    def get_training_status(self, function_name: str) -> TrainingStatus:
        """
        Get the current training status for a function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Current training status.
        """
        return self.background_trainer.get_training_status(function_name)
    
    def get_training_results(self, function_name: str) -> List[TrainingResult]:
        """
        Get training results for a function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            List of training results, most recent first.
        """
        return self.background_trainer.get_training_results(function_name)
    
    def get_collection_stats(self, function_name: str = None) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            function_name: Specific function name, or None for all functions.
            
        Returns:
            Dictionary of collection statistics.
        """
        return self.data_collector.get_collection_stats(function_name)
    
    def clear_training_data(self, function_name: str) -> bool:
        """
        Clear training data for a specific function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            True if data was cleared successfully, False otherwise.
        """
        return self.data_collector.clear_training_data(function_name)
    
    def save_training_data(self, function_name: str) -> bool:
        """
        Save training data for a specific function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            True if data was saved successfully, False otherwise.
        """
        filepath = os.path.join(self.data_dir, f"{function_name}.json")
        return self.data_collector.save_training_data(function_name, filepath)
    
    def get_functions_with_data(self) -> List[str]:
        """
        Get list of functions that have training data.
        
        Returns:
            List of function names.
        """
        stats = self.data_collector.get_collection_stats()
        return [name for name, stat in stats.items() if stat.get("total_samples", 0) > 0]
    
    def get_trainable_functions(self) -> List[Dict[str, Any]]:
        """
        Get list of functions that have enough data for training.
        
        Returns:
            List of dictionaries with function information.
        """
        stats = self.data_collector.get_collection_stats()
        min_samples = self.background_trainer.min_samples_for_training
        
        trainable = []
        for name, stat in stats.items():
            if stat.get("total_samples", 0) >= min_samples:
                status = self.background_trainer.get_training_status(name)
                last_training = self.background_trainer.last_training_time.get(name)
                
                trainable.append({
                    "function_name": name,
                    "samples": stat.get("total_samples", 0),
                    "status": status.value,
                    "last_trained": last_training.isoformat() if last_training else None,
                    "collection_rate": stat.get("collection_rate", 0.0)
                })
        
        return trainable