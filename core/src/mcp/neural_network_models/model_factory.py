#!/usr/bin/env python3
"""
Neural Network Model Factory for MCP Core System
Provides factory methods for creating and managing neural network models.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type

# Import model types
try:
    from .hormone_neural_integration import HormoneNeuralIntegration, HormoneConfig
    from .brain_state_integration import BrainStateIntegration, BrainStateConfig
    from .diffusion_model import DiffusionModel, DiffusionConfig
    from .genetic_diffusion_model import GeneticDiffusionModel, GeneticConfig
    from .pretrain import NeuralNetworkPretrainer, PretrainConfig
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

class ModelRegistry:
    """Registry for neural network models."""
    
    def __init__(self):
        """Initialize model registry."""
        self.models = {}
        self.configs = {}
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, model_name: str, model: Any, config: Any = None) -> bool:
        """Register a model in the registry."""
        if model_name in self.models:
            self.logger.warning(f"Model {model_name} already registered, overwriting")
        
        self.models[model_name] = model
        if config:
            self.configs[model_name] = config
        
        self.logger.info(f"Registered model: {model_name}")
        return True
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from the registry."""
        return self.models.get(model_name)
    
    def get_config(self, model_name: str) -> Optional[Any]:
        """Get a model's configuration from the registry."""
        return self.configs.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())

class ModelFactory:
    """
    Factory for creating neural network models.
    
    Provides methods for creating various types of neural network models
    with appropriate configurations and managing their lifecycle.
    """
    
    def __init__(self, models_dir: str = "data/models"):
        """Initialize model factory."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
        self.logger = logging.getLogger(__name__)
    
    def create_hormone_integration(self, config: Optional[Dict[str, Any]] = None) -> Optional[HormoneNeuralIntegration]:
        """Create a hormone neural integration model."""
        if not MODELS_AVAILABLE:
            self.logger.error("Neural network models not available")
            return None
        
        try:
            # Create configuration
            hormone_config = HormoneConfig()
            if config:
                for key, value in config.items():
                    if hasattr(hormone_config, key):
                        setattr(hormone_config, key, value)
            
            # Create model
            model = HormoneNeuralIntegration(hormone_config)
            
            # Register model
            self.registry.register_model("hormone_integration", model, hormone_config)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating hormone integration: {e}")
            return None
    
    def create_brain_state_integration(self, config: Optional[Dict[str, Any]] = None) -> Optional[BrainStateIntegration]:
        """Create a brain state integration model."""
        if not MODELS_AVAILABLE:
            self.logger.error("Neural network models not available")
            return None
        
        try:
            # Create configuration
            brain_config = BrainStateConfig()
            if config:
                for key, value in config.items():
                    if hasattr(brain_config, key):
                        setattr(brain_config, key, value)
            
            # Create model
            model = BrainStateIntegration(brain_config)
            
            # Register model
            self.registry.register_model("brain_state_integration", model, brain_config)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating brain state integration: {e}")
            return None
    
    def create_diffusion_model(self, config: Optional[Dict[str, Any]] = None) -> Optional[DiffusionModel]:
        """Create a diffusion model."""
        if not MODELS_AVAILABLE:
            self.logger.error("Neural network models not available")
            return None
        
        try:
            # Create configuration
            diffusion_config = DiffusionConfig()
            if config:
                for key, value in config.items():
                    if hasattr(diffusion_config, key):
                        setattr(diffusion_config, key, value)
            
            # Create model
            model = DiffusionModel(diffusion_config)
            
            # Register model
            self.registry.register_model("diffusion_model", model, diffusion_config)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating diffusion model: {e}")
            return None
    
    def create_genetic_diffusion_model(self, config: Optional[Dict[str, Any]] = None) -> Optional[GeneticDiffusionModel]:
        """Create a genetic diffusion model."""
        if not MODELS_AVAILABLE:
            self.logger.error("Neural network models not available")
            return None
        
        try:
            # Create diffusion configuration
            diffusion_config = DiffusionConfig()
            if config and 'diffusion' in config:
                for key, value in config['diffusion'].items():
                    if hasattr(diffusion_config, key):
                        setattr(diffusion_config, key, value)
            
            # Create genetic configuration
            genetic_config = GeneticConfig()
            if config and 'genetic' in config:
                for key, value in config['genetic'].items():
                    if hasattr(genetic_config, key):
                        setattr(genetic_config, key, value)
            
            # Create model
            model = GeneticDiffusionModel(diffusion_config, genetic_config)
            
            # Register model
            self.registry.register_model("genetic_diffusion_model", model, {
                'diffusion': diffusion_config,
                'genetic': genetic_config
            })
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating genetic diffusion model: {e}")
            return None
    
    def create_pretrainer(self, config: Optional[Dict[str, Any]] = None) -> Optional[NeuralNetworkPretrainer]:
        """Create a neural network pretrainer."""
        if not MODELS_AVAILABLE:
            self.logger.error("Neural network models not available")
            return None
        
        try:
            # Create configuration
            pretrain_config = PretrainConfig()
            if config:
                for key, value in config.items():
                    if hasattr(pretrain_config, key):
                        setattr(pretrain_config, key, value)
            
            # Create model
            model = NeuralNetworkPretrainer(pretrain_config)
            
            # Register model
            self.registry.register_model("pretrainer", model, pretrain_config)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating pretrainer: {e}")
            return None
    
    def load_model(self, model_type: str, model_path: str) -> Optional[Any]:
        """Load a model from disk."""
        if not MODELS_AVAILABLE:
            self.logger.error("Neural network models not available")
            return None
        
        try:
            full_path = self.models_dir / model_path
            
            if model_type == "hormone_integration":
                from .hormone_neural_integration import load_hormone_integration
                model = load_hormone_integration(str(full_path))
                if model:
                    self.registry.register_model(f"hormone_integration_{model_path}", model)
                return model
                
            elif model_type == "brain_state_integration":
                from .brain_state_integration import load_brain_state_integration
                model = load_brain_state_integration(str(full_path))
                if model:
                    self.registry.register_model(f"brain_state_integration_{model_path}", model)
                return model
                
            elif model_type == "diffusion_model":
                from .diffusion_model import load_diffusion_model
                model = load_diffusion_model(str(full_path))
                if model:
                    self.registry.register_model(f"diffusion_model_{model_path}", model)
                return model
                
            elif model_type == "genetic_diffusion_model":
                from .genetic_diffusion_model import load_genetic_diffusion_model
                model = load_genetic_diffusion_model(str(full_path))
                if model:
                    self.registry.register_model(f"genetic_diffusion_model_{model_path}", model)
                return model
                
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def save_model_config(self, model_name: str, path: str) -> bool:
        """Save a model's configuration to disk."""
        try:
            model = self.registry.get_model(model_name)
            config = self.registry.get_config(model_name)
            
            if not model or not config:
                self.logger.error(f"Model {model_name} not found in registry")
                return False
            
            # Create directory if it doesn't exist
            save_path = self.models_dir / path
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = save_path / "config.json"
            
            # Convert config to dictionary
            if isinstance(config, dict):
                config_dict = config
            else:
                config_dict = vars(config)
            
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Saved model configuration to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model configuration: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available pretrained models."""
        models = {
            'hormone_integration': [],
            'brain_state_integration': [],
            'diffusion_model': [],
            'genetic_diffusion_model': []
        }
        
        try:
            if not self.models_dir.exists():
                return models
                
            # Check for hormone integration models
            hormone_dir = self.models_dir / "hormone"
            if hormone_dir.exists() and hormone_dir.is_dir():
                for item in hormone_dir.iterdir():
                    if item.is_dir() and (item / "model.pt").exists():
                        models['hormone_integration'].append(item.name)
            
            # Check for brain state integration models
            brain_dir = self.models_dir / "brain-state"
            if brain_dir.exists() and brain_dir.is_dir():
                for item in brain_dir.iterdir():
                    if item.is_dir() and (item / "model.pt").exists():
                        models['brain_state_integration'].append(item.name)
            
            # Check for diffusion models
            diffusion_dir = self.models_dir / "diffusion"
            if diffusion_dir.exists() and diffusion_dir.is_dir():
                for item in diffusion_dir.iterdir():
                    if item.is_dir() and (item / "model.pt").exists():
                        models['diffusion_model'].append(item.name)
            
            # Check for genetic diffusion models
            genetic_dir = self.models_dir / "genetic"
            if genetic_dir.exists() and genetic_dir.is_dir():
                for item in genetic_dir.iterdir():
                    if item.is_dir() and (item / "diffusion/model.pt").exists():
                        models['genetic_diffusion_model'].append(item.name)
            
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
        
        return models

# Global factory instance
_model_factory: Optional[ModelFactory] = None

def get_model_factory() -> ModelFactory:
    """Get the global model factory instance."""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory