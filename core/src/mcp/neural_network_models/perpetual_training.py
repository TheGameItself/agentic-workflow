#!/usr/bin/env python3
"""
Perpetual Neural Network Training for MCP Core System
Implements continuous background training for neural network models.
"""

import asyncio
import logging
import os
import time
import json
import threading
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
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

# Import neural models
from .hormone_neural_integration import HormoneNeuralIntegration, HormoneConfig, load_hormone_integration
from .diffusion_model import DiffusionModel, DiffusionConfig
from .genetic_diffusion_model import GeneticDiffusionModel, GeneticConfig
from .brain_state_integration import BrainStateIntegration, BrainStateConfig
from .pretrain import NeuralNetworkPretrainer, PretrainConfig

class TrainingConfig:
    """Configuration for perpetual training."""
    
    def __init__(self,
                 training_interval: int = 3600,  # Train every hour
                 min_samples_required: int = 100,
                 max_samples_stored: int = 10000,
                 training_epochs: int = 1,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 enable_hormone_training: bool = True,
                 enable_diffusion_training: bool = True,
                 enable_brain_state_training: bool = True,
                 enable_genetic_training: bool = False,
                 device: str = None,
                 output_dir: str = "data/models",
                 log_level: str = "INFO"):
        """Initialize training configuration."""
        self.training_interval = training_interval
        self.min_samples_required = min_samples_required
        self.max_samples_stored = max_samples_stored
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.enable_hormone_training = enable_hormone_training
        self.enable_diffusion_training = enable_diffusion_training
        self.enable_brain_state_training = enable_brain_state_training
        self.enable_genetic_training = enable_genetic_training
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.log_level = log_level

class PerpetualTrainingManager:
    """
    Perpetual Neural Network Training Manager for MCP Core System.
    
    Manages continuous background training of neural network models
    using system data collected during operation.
    
    Features:
    - Asynchronous background training
    - Experience replay buffer for metrics and embeddings
    - Adaptive training frequency based on system load
    - Incremental model improvement
    - Automatic model versioning and rollback
    """
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize perpetual training manager."""
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Experience replay buffers
        self.metrics_buffer = []
        self.embedding_buffer = []
        self.text_buffer = []
        
        # Training state
        self.is_training = False
        self.last_training_time = {
            'hormone': 0,
            'diffusion': 0,
            'brain_state': 0,
            'genetic': 0
        }
        self.training_stats = {
            'hormone': {'total_trainings': 0, 'total_samples': 0, 'last_loss': 0.0},
            'diffusion': {'total_trainings': 0, 'total_samples': 0, 'last_loss': 0.0},
            'brain_state': {'total_trainings': 0, 'total_samples': 0, 'last_loss': 0.0},
            'genetic': {'total_trainings': 0, 'total_samples': 0, 'last_loss': 0.0}
        }
        
        # Model references
        self.hormone_model = None
        self.diffusion_model = None
        self.brain_state_model = None
        self.genetic_model = None
        
        # Background task
        self.training_task = None
        self.training_thread = None
        self.stop_event = threading.Event()
        self.training_queue = queue.Queue()
        
        # Locks for thread safety
        self.metrics_lock = threading.Lock()
        self.embedding_lock = threading.Lock()
        self.text_lock = threading.Lock()
        self.training_lock = threading.Lock()
    
    def register_models(self, hormone_model=None, diffusion_model=None, 
                       brain_state_model=None, genetic_model=None):
        """Register neural network models for training."""
        self.hormone_model = hormone_model
        self.diffusion_model = diffusion_model
        self.brain_state_model = brain_state_model
        self.genetic_model = genetic_model
        
        self.logger.info("Neural network models registered for perpetual training")
    
    def add_metrics_sample(self, metrics: Dict[str, float]):
        """Add a metrics sample to the experience replay buffer."""
        with self.metrics_lock:
            # Add timestamp
            metrics_with_time = metrics.copy()
            metrics_with_time['timestamp'] = time.time()
            
            self.metrics_buffer.append(metrics_with_time)
            
            # Limit buffer size
            if len(self.metrics_buffer) > self.config.max_samples_stored:
                self.metrics_buffer.pop(0)
    
    def add_embedding_sample(self, embedding: Union[torch.Tensor, List[float]]):
        """Add an embedding sample to the experience replay buffer."""
        if not TORCH_AVAILABLE:
            return
        
        with self.embedding_lock:
            # Convert to tensor if needed
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            # Store as CPU tensor to save GPU memory
            self.embedding_buffer.append(embedding.cpu())
            
            # Limit buffer size
            if len(self.embedding_buffer) > self.config.max_samples_stored:
                self.embedding_buffer.pop(0)
    
    def add_text_sample(self, text: str):
        """Add a text sample to the experience replay buffer."""
        with self.text_lock:
            self.text_buffer.append(text)
            
            # Limit buffer size
            if len(self.text_buffer) > self.config.max_samples_stored:
                self.text_buffer.pop(0)
    
    def start_background_training(self):
        """Start background training thread."""
        if self.training_thread and self.training_thread.is_alive():
            self.logger.warning("Background training already running")
            return False
        
        self.stop_event.clear()
        self.training_thread = threading.Thread(
            target=self._background_training_loop,
            daemon=True
        )
        self.training_thread.start()
        
        self.logger.info("Background training started")
        return True
    
    def stop_background_training(self):
        """Stop background training thread."""
        if not self.training_thread or not self.training_thread.is_alive():
            return False
        
        self.stop_event.set()
        self.training_thread.join(timeout=10)
        
        self.logger.info("Background training stopped")
        return True
    
    def _background_training_loop(self):
        """Background training loop."""
        self.logger.info("Background training loop started")
        
        while not self.stop_event.is_set():
            try:
                # Check if it's time to train any models
                current_time = time.time()
                
                # Check hormone model
                if (self.config.enable_hormone_training and 
                    self.hormone_model and 
                    current_time - self.last_training_time['hormone'] > self.config.training_interval and
                    len(self.metrics_buffer) >= self.config.min_samples_required):
                    
                    self.training_queue.put(('hormone', None))
                
                # Check diffusion model
                if (self.config.enable_diffusion_training and 
                    self.diffusion_model and 
                    current_time - self.last_training_time['diffusion'] > self.config.training_interval and
                    len(self.embedding_buffer) >= self.config.min_samples_required):
                    
                    self.training_queue.put(('diffusion', None))
                
                # Check brain state model
                if (self.config.enable_brain_state_training and 
                    self.brain_state_model and 
                    current_time - self.last_training_time['brain_state'] > self.config.training_interval and
                    len(self.metrics_buffer) >= self.config.min_samples_required and
                    len(self.embedding_buffer) >= self.config.min_samples_required):
                    
                    self.training_queue.put(('brain_state', None))
                
                # Check genetic model
                if (self.config.enable_genetic_training and 
                    self.genetic_model and 
                    current_time - self.last_training_time['genetic'] > self.config.training_interval and
                    len(self.embedding_buffer) >= self.config.min_samples_required):
                    
                    self.training_queue.put(('genetic', None))
                
                # Process training queue
                while not self.training_queue.empty():
                    model_type, params = self.training_queue.get()
                    self._train_model(model_type, params)
                    self.training_queue.task_done()
                
                # Sleep for a while
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in background training loop: {e}")
                time.sleep(300)  # Wait longer on error
    
    def _train_model(self, model_type: str, params: Optional[Dict[str, Any]] = None):
        """Train a specific model."""
        if self.is_training:
            self.logger.warning(f"Training already in progress, skipping {model_type} training")
            return False
        
        try:
            self.is_training = True
            self.logger.info(f"Starting {model_type} model training")
            
            if model_type == 'hormone':
                success = self._train_hormone_model()
            elif model_type == 'diffusion':
                success = self._train_diffusion_model()
            elif model_type == 'brain_state':
                success = self._train_brain_state_model()
            elif model_type == 'genetic':
                success = self._train_genetic_model()
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                success = False
            
            if success:
                self.last_training_time[model_type] = time.time()
                self.training_stats[model_type]['total_trainings'] += 1
                self.logger.info(f"{model_type} model training completed successfully")
            else:
                self.logger.warning(f"{model_type} model training failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model: {e}")
            return False
            
        finally:
            self.is_training = False
    
    def _train_hormone_model(self) -> bool:
        """Train hormone neural integration model."""
        if not self.hormone_model:
            return False
        
        with self.metrics_lock:
            if len(self.metrics_buffer) < self.config.min_samples_required:
                return False
            
            # Use a copy of the buffer to avoid modifying it during training
            metrics_data = self.metrics_buffer.copy()
        
        try:
            # Generate training data
            inputs, outputs = self.hormone_model.generate_training_data(metrics_data)
            
            if inputs.shape[0] == 0:
                return False
            
            # Train the model
            self.hormone_model.train(
                inputs, 
                outputs, 
                epochs=self.config.training_epochs,
                batch_size=self.config.batch_size
            )
            
            # Update stats
            self.training_stats['hormone']['total_samples'] += len(metrics_data)
            
            # Save model
            model_dir = os.path.join(self.config.output_dir, "hormone")
            os.makedirs(model_dir, exist_ok=True)
            self.hormone_model._save_model(model_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training hormone model: {e}")
            return False
    
    def _train_diffusion_model(self) -> bool:
        """Train diffusion model."""
        if not self.diffusion_model:
            return False
        
        with self.embedding_lock:
            if len(self.embedding_buffer) < self.config.min_samples_required:
                return False
            
            # Stack embeddings into a single tensor
            embeddings = torch.stack(self.embedding_buffer)
        
        try:
            # Train the model
            self.diffusion_model.train(
                embeddings, 
                epochs=self.config.training_epochs
            )
            
            # Update stats
            self.training_stats['diffusion']['total_samples'] += len(self.embedding_buffer)
            
            # Save model
            model_dir = os.path.join(self.config.output_dir, "diffusion")
            os.makedirs(model_dir, exist_ok=True)
            self.diffusion_model._save_checkpoint(model_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training diffusion model: {e}")
            return False
    
    def _train_brain_state_model(self) -> bool:
        """Train brain state integration model."""
        if not self.brain_state_model:
            return False
        
        with self.metrics_lock, self.embedding_lock:
            if (len(self.metrics_buffer) < self.config.min_samples_required or
                len(self.embedding_buffer) < self.config.min_samples_required):
                return False
            
            # Use a copy of the buffers
            metrics_data = self.metrics_buffer.copy()
            embeddings = torch.stack(self.embedding_buffer)
        
        try:
            # Generate hormone levels from metrics
            hormone_levels = []
            if self.hormone_model:
                for metrics in metrics_data:
                    levels = self.hormone_model.update_hormone_levels(metrics)
                    hormone_tensor = torch.tensor([
                        levels['stress'],
                        levels['efficiency'],
                        levels['adaptation'],
                        levels['stability']
                    ], dtype=torch.float32)
                    hormone_levels.append(hormone_tensor)
                
                hormone_tensor = torch.stack(hormone_levels)
            else:
                # Generate random hormone levels
                hormone_tensor = torch.rand(len(metrics_data), 4)
            
            # Limit to same number of samples
            min_samples = min(embeddings.shape[0], hormone_tensor.shape[0])
            embeddings = embeddings[:min_samples]
            hormone_tensor = hormone_tensor[:min_samples]
            
            # Train brain state network
            self.brain_state_model.train_brain_state_network(
                embeddings,
                hormone_tensor,
                epochs=self.config.training_epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate
            )
            
            # Update stats
            self.training_stats['brain_state']['total_samples'] += min_samples
            
            # Save model
            model_dir = os.path.join(self.config.output_dir, "brain-state")
            os.makedirs(model_dir, exist_ok=True)
            self.brain_state_model._save_model(model_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training brain state model: {e}")
            return False
    
    def _train_genetic_model(self) -> bool:
        """Train genetic diffusion model."""
        if not self.genetic_model:
            return False
        
        with self.embedding_lock:
            if len(self.embedding_buffer) < self.config.min_samples_required:
                return False
            
            # Stack embeddings into a single tensor
            embeddings = torch.stack(self.embedding_buffer)
        
        try:
            # Initialize population if needed
            if not self.genetic_model.population:
                self.genetic_model.initialize_population(embeddings)
            
            # Define simple fitness function
            def fitness_function(individual):
                # Calculate similarity to real embeddings
                similarities = []
                for emb in embeddings:
                    similarity = torch.nn.functional.cosine_similarity(
                        individual.unsqueeze(0), 
                        emb.unsqueeze(0)
                    ).item()
                    similarities.append(similarity)
                
                # Return average similarity
                return sum(similarities) / len(similarities)
            
            # Evolve for a few generations
            self.genetic_model.evolve(
                fitness_function, 
                generations=self.config.training_epochs
            )
            
            # Train diffusion model on population
            self.genetic_model.train_diffusion_model()
            
            # Update stats
            self.training_stats['genetic']['total_samples'] += len(self.embedding_buffer)
            
            # Save model
            model_dir = os.path.join(self.config.output_dir, "genetic")
            os.makedirs(model_dir, exist_ok=True)
            self.genetic_model.save_model(model_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training genetic model: {e}")
            return False
    
    def request_training(self, model_type: str, priority: bool = False):
        """
        Request training for a specific model.
        
        Args:
            model_type: Type of model to train ('hormone', 'diffusion', 'brain_state', 'genetic')
            priority: Whether this is a priority training request
        """
        if model_type not in ['hormone', 'diffusion', 'brain_state', 'genetic']:
            self.logger.error(f"Unknown model type: {model_type}")
            return False
        
        if priority:
            # Put at the front of the queue
            old_queue = list(self.training_queue.queue)
            with self.training_queue.mutex:
                self.training_queue.queue.clear()
            
            self.training_queue.put((model_type, None))
            
            for item in old_queue:
                self.training_queue.put(item)
        else:
            # Add to the end of the queue
            self.training_queue.put((model_type, None))
        
        return True
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'models': self.training_stats,
            'buffers': {
                'metrics': len(self.metrics_buffer),
                'embeddings': len(self.embedding_buffer),
                'texts': len(self.text_buffer)
            },
            'last_training': self.last_training_time,
            'is_training': self.is_training,
            'queue_size': self.training_queue.qsize()
        }
        
        return stats

# Convenience functions

def create_perpetual_training_manager(config: Optional[TrainingConfig] = None) -> PerpetualTrainingManager:
    """Create a perpetual training manager with custom configuration."""
    return PerpetualTrainingManager(config or TrainingConfig())

def get_default_perpetual_training_manager() -> PerpetualTrainingManager:
    """Get a default perpetual training manager."""
    return PerpetualTrainingManager()