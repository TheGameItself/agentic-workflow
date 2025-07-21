"""
Model Manager for Neural Network Models component.

This module provides functionality for loading, saving, and managing neural models
for hormone calculations. It implements a model registry for different hormone functions
and supports both algorithmic and neural implementations.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

class ModelRegistry:
    """
    Registry for neural network models used in hormone calculations.
    
    This class maintains a registry of available models for different hormone functions,
    tracks their performance, and manages model versioning.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.registered_functions: Set[str] = set()
        self.function_models: Dict[str, List[str]] = {}
        self.default_models: Dict[str, str] = {}
        self.model_dependencies: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("ModelRegistry")
        
    def register_function(self, function_name: str, model_types: List[str] = None, 
                         default_model: str = None) -> bool:
        """
        Register a hormone function with the registry.
        
        Args:
            function_name: Name of the hormone function to register.
            model_types: List of model types that can implement this function.
            default_model: Default model type to use for this function.
            
        Returns:
            True if the function was registered successfully, False otherwise.
        """
        if function_name in self.registered_functions:
            self.logger.warning(f"Function already registered: {function_name}")
            return False
        
        model_types = model_types or ["default"]
        default_model = default_model or model_types[0]
        
        if default_model not in model_types:
            self.logger.error(f"Default model {default_model} not in model types {model_types}")
            return False
        
        self.registered_functions.add(function_name)
        self.function_models[function_name] = model_types
        self.default_models[function_name] = default_model
        
        self.logger.info(f"Registered function {function_name} with model types {model_types}")
        return True
    
    def register_model_dependency(self, function_name: str, dependency: str) -> bool:
        """
        Register a dependency between hormone functions.
        
        Args:
            function_name: Name of the dependent function.
            dependency: Name of the function it depends on.
            
        Returns:
            True if the dependency was registered successfully, False otherwise.
        """
        if function_name not in self.registered_functions:
            self.logger.error(f"Cannot register dependency for unregistered function: {function_name}")
            return False
        
        if dependency not in self.registered_functions:
            self.logger.error(f"Cannot register unregistered dependency: {dependency}")
            return False
        
        if function_name not in self.model_dependencies:
            self.model_dependencies[function_name] = []
        
        if dependency not in self.model_dependencies[function_name]:
            self.model_dependencies[function_name].append(dependency)
            self.logger.info(f"Registered dependency {dependency} for function {function_name}")
        
        return True
    
    def get_model_types(self, function_name: str) -> List[str]:
        """
        Get available model types for a function.
        
        Args:
            function_name: Name of the hormone function.
            
        Returns:
            List of available model types, or empty list if function not registered.
        """
        return self.function_models.get(function_name, [])
    
    def get_default_model(self, function_name: str) -> Optional[str]:
        """
        Get the default model type for a function.
        
        Args:
            function_name: Name of the hormone function.
            
        Returns:
            Default model type, or None if function not registered.
        """
        return self.default_models.get(function_name)
    
    def set_default_model(self, function_name: str, model_type: str) -> bool:
        """
        Set the default model type for a function.
        
        Args:
            function_name: Name of the hormone function.
            model_type: Model type to set as default.
            
        Returns:
            True if the default was set successfully, False otherwise.
        """
        if function_name not in self.registered_functions:
            self.logger.error(f"Cannot set default for unregistered function: {function_name}")
            return False
        
        if model_type not in self.function_models.get(function_name, []):
            self.logger.error(f"Model type {model_type} not registered for function {function_name}")
            return False
        
        self.default_models[function_name] = model_type
        self.logger.info(f"Set default model for {function_name} to {model_type}")
        return True
    
    def get_dependencies(self, function_name: str) -> List[str]:
        """
        Get dependencies for a function.
        
        Args:
            function_name: Name of the hormone function.
            
        Returns:
            List of dependencies, or empty list if none.
        """
        return self.model_dependencies.get(function_name, [])
    
    def is_registered(self, function_name: str) -> bool:
        """
        Check if a function is registered.
        
        Args:
            function_name: Name of the hormone function.
            
        Returns:
            True if the function is registered, False otherwise.
        """
        return function_name in self.registered_functions
    
    def get_all_registered_functions(self) -> List[str]:
        """
        Get all registered functions.
        
        Returns:
            List of all registered function names.
        """
        return list(self.registered_functions)


class ModelManager:
    """
    Manages neural network models for hormone calculations.
    
    This class handles loading, saving, and managing neural models for
    different hormone functions. It provides a unified interface for
    accessing neural implementations of hormone calculations.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory where models are stored. If None, defaults to
                        a 'models' directory in the current working directory.
        """
        self.logger = logging.getLogger("ModelManager")
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models")
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.registry = ModelRegistry()
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create subdirectories for different model types
        os.makedirs(os.path.join(self.models_dir, "hormone"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "diffusion"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "cascade"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "receptor"), exist_ok=True)
        
        # Initialize registry with common hormone functions
        self._initialize_registry()
        
        # Load available models
        self._discover_available_models()
        
        self.logger.info(f"ModelManager initialized with models directory: {self.models_dir}")
    
    def _initialize_registry(self):
        """Initialize the model registry with common hormone functions."""
        # Register hormone production functions
        self.registry.register_function("dopamine_production", ["mlp", "lstm", "algorithmic"], "algorithmic")
        self.registry.register_function("serotonin_production", ["mlp", "lstm", "algorithmic"], "algorithmic")
        self.registry.register_function("cortisol_production", ["mlp", "lstm", "algorithmic"], "algorithmic")
        self.registry.register_function("oxytocin_production", ["mlp", "lstm", "algorithmic"], "algorithmic")
        self.registry.register_function("adrenaline_production", ["mlp", "lstm", "algorithmic"], "algorithmic")
        
        # Register hormone diffusion functions
        self.registry.register_function("spatial_diffusion", ["cnn", "graph_nn", "algorithmic"], "algorithmic")
        self.registry.register_function("temporal_diffusion", ["lstm", "transformer", "algorithmic"], "algorithmic")
        
        # Register hormone cascade functions
        self.registry.register_function("cascade_trigger", ["mlp", "decision_tree", "algorithmic"], "algorithmic")
        self.registry.register_function("cascade_inhibition", ["mlp", "decision_tree", "algorithmic"], "algorithmic")
        
        # Register receptor functions
        self.registry.register_function("receptor_sensitivity", ["mlp", "algorithmic"], "algorithmic")
        self.registry.register_function("receptor_adaptation", ["lstm", "algorithmic"], "algorithmic")
        
        # Register dependencies
        self.registry.register_model_dependency("cascade_trigger", "dopamine_production")
        self.registry.register_model_dependency("cascade_trigger", "serotonin_production")
        self.registry.register_model_dependency("cascade_inhibition", "cortisol_production")
        self.registry.register_model_dependency("receptor_adaptation", "receptor_sensitivity")
    
    def _discover_available_models(self):
        """Discover available models in the models directory."""
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith(".model"):
                    function_name = file.split(".")[0]
                    # Don't load the model yet, just note that it's available
                    if function_name not in self.model_metadata:
                        self.model_metadata[function_name] = {
                            "function_name": function_name,
                            "available": True,
                            "path": os.path.join(root, file)
                        }
                        
                        # Check for metadata file
                        metadata_path = os.path.join(root, f"{function_name}.metadata")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                self.model_metadata[function_name].update(metadata)
                            except Exception as e:
                                self.logger.warning(f"Error loading metadata for {function_name}: {e}")
    
    def get_model(self, function_name: str) -> Any:
        """
        Get a neural model for a specific function.
        
        Args:
            function_name: Name of the function for which to get a model.
            
        Returns:
            The neural model, or None if no model is available.
        """
        if function_name in self.models:
            return self.models[function_name]
        
        # Try to load the model if it's not already loaded
        if self.load_model(function_name):
            return self.models[function_name]
        
        # If the function is registered but no model is available, create a new one
        if self.registry.is_registered(function_name):
            model_type = self.registry.get_default_model(function_name)
            if self.create_new_model(function_name, model_type):
                return self.models[function_name]
        
        self.logger.warning(f"No model available for function: {function_name}")
        return None
    
    def save_model(self, function_name: str) -> bool:
        """
        Save a trained model to disk.
        
        Args:
            function_name: Name of the function for which to save the model.
            
        Returns:
            True if the model was saved successfully, False otherwise.
        """
        if function_name not in self.models:
            self.logger.warning(f"No model to save for function: {function_name}")
            return False
        
        try:
            model = self.models[function_name]
            
            # Determine appropriate subdirectory based on function name
            if "production" in function_name:
                subdir = "hormone"
            elif "diffusion" in function_name:
                subdir = "diffusion"
            elif "cascade" in function_name:
                subdir = "cascade"
            elif "receptor" in function_name:
                subdir = "receptor"
            else:
                subdir = ""
                
            # Create full path
            model_dir = os.path.join(self.models_dir, subdir)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{function_name}.model")
            
            # Get current metadata or initialize new metadata
            metadata = self.model_metadata.get(function_name, {}).copy()
            
            # Update metadata
            metadata.update({
                "function_name": function_name,
                "saved_at": datetime.now().isoformat(),
                "version": metadata.get("version", "1.0.0"),
                "model_type": getattr(model, "type", "unknown"),
                "path": model_path
            })
            
            # Add performance metrics if available
            if "performance_metrics" in metadata:
                # Keep existing metrics
                pass
            else:
                metadata["performance_metrics"] = {}
            
            # Placeholder for actual model saving logic
            # In a real implementation, this would use a framework like PyTorch or TensorFlow
            # to save the model to disk
            
            # For demonstration, we'll create a dummy model file
            with open(model_path, 'w') as f:
                f.write(f"Model for {function_name}\n")
                f.write(f"Type: {metadata.get('model_type', 'unknown')}\n")
                f.write(f"Version: {metadata.get('version', '1.0.0')}\n")
                f.write(f"Saved at: {metadata.get('saved_at')}\n")
            
            # Save metadata to a separate file
            metadata_path = os.path.join(model_dir, f"{function_name}.metadata")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update internal metadata
            self.model_metadata[function_name] = metadata
            
            self.logger.info(f"Saved model for {function_name} to {model_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving model for {function_name}: {e}")
            return False
    
    def load_model(self, function_name: str) -> bool:
        """
        Load a pre-trained model from disk.
        
        Args:
            function_name: Name of the function for which to load the model.
            
        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            # Check if we have metadata for this model
            if function_name in self.model_metadata and "path" in self.model_metadata[function_name]:
                model_path = self.model_metadata[function_name]["path"]
            else:
                # Try to find the model in subdirectories
                model_path = None
                for subdir in ["hormone", "diffusion", "cascade", "receptor", ""]:
                    path = os.path.join(self.models_dir, subdir, f"{function_name}.model")
                    if os.path.exists(path):
                        model_path = path
                        break
            
            if not model_path or not os.path.exists(model_path):
                self.logger.warning(f"No saved model found for function: {function_name}")
                return False
            
            # Placeholder for actual model loading logic
            # In a real implementation, this would use a framework like PyTorch or TensorFlow
            # to load the model from disk
            
            # For demonstration, create a dummy model with information from the model file
            model_type = "unknown"
            model_version = "1.0.0"
            
            try:
                with open(model_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Type:"):
                            model_type = line.split(":", 1)[1].strip()
                        elif line.startswith("Version:"):
                            model_version = line.split(":", 1)[1].strip()
            except Exception as e:
                self.logger.warning(f"Error reading model file for {function_name}: {e}")
            
            # Create a dummy model object
            self.models[function_name] = {
                "type": model_type,
                "function": function_name,
                "version": model_version,
                "loaded_at": datetime.now().isoformat()
            }
            
            # Load metadata if available
            metadata_path = os.path.join(os.path.dirname(model_path), f"{function_name}.metadata")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update with loaded_at timestamp
                    metadata["loaded_at"] = datetime.now().isoformat()
                    self.model_metadata[function_name] = metadata
                except Exception as e:
                    self.logger.warning(f"Error loading metadata for {function_name}: {e}")
                    # Create basic metadata
                    self.model_metadata[function_name] = {
                        "function_name": function_name,
                        "loaded_at": datetime.now().isoformat(),
                        "version": model_version,
                        "model_type": model_type,
                        "path": model_path
                    }
            else:
                # Create basic metadata
                self.model_metadata[function_name] = {
                    "function_name": function_name,
                    "loaded_at": datetime.now().isoformat(),
                    "version": model_version,
                    "model_type": model_type,
                    "path": model_path
                }
            
            self.logger.info(f"Loaded model for {function_name} from {model_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading model for {function_name}: {e}")
            return False
    
    def get_performance_metrics(self, function_name: str) -> Dict[str, float]:
        """
        Get performance metrics for a neural model.
        
        Args:
            function_name: Name of the function for which to get performance metrics.
            
        Returns:
            Dictionary of performance metrics, or an empty dictionary if no metrics are available.
        """
        if function_name not in self.model_metadata:
            return {}
        
        return self.model_metadata[function_name].get("performance_metrics", {})
    
    def update_performance_metrics(self, function_name: str, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a neural model.
        
        Args:
            function_name: Name of the function for which to update metrics.
            metrics: Dictionary of performance metrics to update.
        """
        if function_name not in self.model_metadata:
            self.model_metadata[function_name] = {}
        
        if "performance_metrics" not in self.model_metadata[function_name]:
            self.model_metadata[function_name]["performance_metrics"] = {}
        
        # Update metrics with new values
        self.model_metadata[function_name]["performance_metrics"].update(metrics)
        
        # Add timestamp
        self.model_metadata[function_name]["last_updated"] = datetime.now().isoformat()
        
        # Calculate and update aggregate metrics
        perf_metrics = self.model_metadata[function_name]["performance_metrics"]
        
        # Calculate overall score if accuracy and latency are available
        if "accuracy" in perf_metrics and "latency" in perf_metrics:
            # Normalize latency (lower is better)
            norm_latency = max(0, 1 - min(1, perf_metrics["latency"] / 1000))
            
            # Calculate weighted score (70% accuracy, 30% latency)
            overall_score = 0.7 * perf_metrics["accuracy"] + 0.3 * norm_latency
            perf_metrics["overall_score"] = overall_score
        
        # Save updated metadata to disk if we have a path
        if "path" in self.model_metadata[function_name]:
            model_dir = os.path.dirname(self.model_metadata[function_name]["path"])
            metadata_path = os.path.join(model_dir, f"{function_name}.metadata")
            
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(self.model_metadata[function_name], f, indent=2)
            except Exception as e:
                self.logger.warning(f"Error saving metadata for {function_name}: {e}")
        
        self.logger.info(f"Updated performance metrics for {function_name}: {metrics}")
    
    def list_available_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of function names for which models are available.
        """
        return list(self.model_metadata.keys())
    
    def create_new_model(self, function_name: str, model_type: str = "default") -> bool:
        """
        Create a new model for a function.
        
        Args:
            function_name: Name of the function for which to create a model.
            model_type: Type of model to create.
            
        Returns:
            True if the model was created successfully, False otherwise.
        """
        if function_name in self.models:
            self.logger.warning(f"Model already exists for function: {function_name}")
            return False
        
        try:
            # Check if the function is registered
            if self.registry.is_registered(function_name):
                # Validate model type
                valid_types = self.registry.get_model_types(function_name)
                if model_type not in valid_types:
                    self.logger.warning(f"Invalid model type {model_type} for function {function_name}. "
                                      f"Valid types: {valid_types}")
                    # Use default model type
                    model_type = self.registry.get_default_model(function_name)
            
            # Placeholder for actual model creation logic
            # In a real implementation, this would create a new neural network model
            # using a framework like PyTorch or TensorFlow
            
            # For demonstration, create a dummy model
            self.models[function_name] = {
                "type": model_type,
                "function": function_name,
                "version": "1.0.0",
                "created_at": datetime.now().isoformat()
            }
            
            # Initialize metadata
            self.model_metadata[function_name] = {
                "function_name": function_name,
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "model_type": model_type,
                "performance_metrics": {}
            }
            
            # Save the new model
            self.save_model(function_name)
            
            self.logger.info(f"Created new {model_type} model for {function_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating model for {function_name}: {e}")
            return False
    
    def delete_model(self, function_name: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            function_name: Name of the function for which to delete the model.
            
        Returns:
            True if the model was deleted successfully, False otherwise.
        """
        if function_name not in self.model_metadata:
            self.logger.warning(f"No model to delete for function: {function_name}")
            return False
        
        try:
            # Remove from memory
            if function_name in self.models:
                del self.models[function_name]
            
            # Get file paths
            model_path = self.model_metadata[function_name].get("path")
            if model_path:
                metadata_path = os.path.join(os.path.dirname(model_path), f"{function_name}.metadata")
                
                # Delete files
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
            
            # Remove from metadata
            del self.model_metadata[function_name]
            
            self.logger.info(f"Deleted model for {function_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error deleting model for {function_name}: {e}")
            return False
    
    def get_model_info(self, function_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            function_name: Name of the function for which to get model information.
            
        Returns:
            Dictionary with model information, or empty dictionary if model not found.
        """
        if function_name not in self.model_metadata:
            return {}
        
        info = self.model_metadata[function_name].copy()
        
        # Add registry information if available
        if self.registry.is_registered(function_name):
            info["registered"] = True
            info["valid_model_types"] = self.registry.get_model_types(function_name)
            info["default_model_type"] = self.registry.get_default_model(function_name)
            info["dependencies"] = self.registry.get_dependencies(function_name)
        else:
            info["registered"] = False
        
        # Add loaded status
        info["loaded"] = function_name in self.models
        
        return info
    
    def compare_implementations(self, function_name: str) -> Dict[str, Any]:
        """
        Compare neural and algorithmic implementations for a function.
        
        Args:
            function_name: Name of the function to compare.
            
        Returns:
            Dictionary with comparison results.
        """
        results = {
            "function_name": function_name,
            "timestamp": datetime.now().isoformat(),
            "implementations": {},
            "recommendation": None
        }
        
        # Check if we have performance metrics for this function
        if function_name not in self.model_metadata:
            return results
        
        # Get neural implementation metrics
        neural_metrics = self.get_performance_metrics(function_name)
        
        # For demonstration, generate algorithmic metrics
        # In a real implementation, these would come from the algorithmic implementation
        algorithmic_metrics = {
            "accuracy": 0.85,
            "latency": 5.0,
            "resource_usage": 0.2
        }
        
        # Add to results
        results["implementations"]["neural"] = neural_metrics
        results["implementations"]["algorithmic"] = algorithmic_metrics
        
        # Compare and make recommendation
        if "accuracy" in neural_metrics and "latency" in neural_metrics:
            neural_score = 0.7 * neural_metrics["accuracy"] - 0.3 * (neural_metrics["latency"] / 100)
            algo_score = 0.7 * algorithmic_metrics["accuracy"] - 0.3 * (algorithmic_metrics["latency"] / 100)
            
            results["scores"] = {
                "neural": neural_score,
                "algorithmic": algo_score
            }
            
            # Recommend the better implementation with a small hysteresis
            # to prevent frequent switching
            if neural_score > algo_score * 1.1:
                results["recommendation"] = "neural"
            elif algo_score > neural_score * 1.1:
                results["recommendation"] = "algorithmic"
            else:
                # If scores are close, stick with algorithmic for stability
                results["recommendation"] = "algorithmic"
        
        return results
    
    def export_models(self, export_dir: str) -> bool:
        """
        Export all models to a directory.
        
        Args:
            export_dir: Directory to export models to.
            
        Returns:
            True if models were exported successfully, False otherwise.
        """
        try:
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # Export each model
            for function_name, metadata in self.model_metadata.items():
                if "path" in metadata and os.path.exists(metadata["path"]):
                    # Copy model file
                    dest_path = os.path.join(export_dir, os.path.basename(metadata["path"]))
                    shutil.copy2(metadata["path"], dest_path)
                    
                    # Copy metadata file
                    metadata_path = os.path.join(os.path.dirname(metadata["path"]), 
                                               f"{function_name}.metadata")
                    if os.path.exists(metadata_path):
                        dest_metadata = os.path.join(export_dir, f"{function_name}.metadata")
                        shutil.copy2(metadata_path, dest_metadata)
            
            self.logger.info(f"Exported models to {export_dir}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting models: {e}")
            return False
    
    def import_models(self, import_dir: str) -> bool:
        """
        Import models from a directory.
        
        Args:
            import_dir: Directory to import models from.
            
        Returns:
            True if models were imported successfully, False otherwise.
        """
        try:
            # Check if import directory exists
            if not os.path.exists(import_dir) or not os.path.isdir(import_dir):
                self.logger.error(f"Import directory does not exist: {import_dir}")
                return False
            
            # Import each model file
            imported_count = 0
            for file in os.listdir(import_dir):
                if file.endswith(".model"):
                    function_name = file.split(".")[0]
                    
                    # Copy model file
                    src_path = os.path.join(import_dir, file)
                    
                    # Determine appropriate subdirectory
                    if "production" in function_name:
                        subdir = "hormone"
                    elif "diffusion" in function_name:
                        subdir = "diffusion"
                    elif "cascade" in function_name:
                        subdir = "cascade"
                    elif "receptor" in function_name:
                        subdir = "receptor"
                    else:
                        subdir = ""
                    
                    # Create destination directory
                    dest_dir = os.path.join(self.models_dir, subdir)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Copy model file
                    dest_path = os.path.join(dest_dir, file)
                    shutil.copy2(src_path, dest_path)
                    
                    # Copy metadata file if available
                    metadata_file = f"{function_name}.metadata"
                    src_metadata = os.path.join(import_dir, metadata_file)
                    if os.path.exists(src_metadata):
                        dest_metadata = os.path.join(dest_dir, metadata_file)
                        shutil.copy2(src_metadata, dest_metadata)
                    
                    imported_count += 1
            
            # Rediscover available models
            self._discover_available_models()
            
            self.logger.info(f"Imported {imported_count} models from {import_dir}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error importing models: {e}")
            return False
    
    def get_model_version(self, function_name: str) -> str:
        """
        Get the version of a model.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Version string, or "unknown" if not available.
        """
        if function_name in self.model_metadata:
            return self.model_metadata[function_name].get("version", "unknown")
        return "unknown"
    
    def update_model_version(self, function_name: str, version: str) -> bool:
        """
        Update the version of a model.
        
        Args:
            function_name: Name of the function.
            version: New version string.
            
        Returns:
            True if the version was updated successfully, False otherwise.
        """
        if function_name not in self.model_metadata:
            self.logger.warning(f"No model found for function: {function_name}")
            return False
        
        try:
            # Update version in metadata
            self.model_metadata[function_name]["version"] = version
            
            # Update version in model if loaded
            if function_name in self.models:
                self.models[function_name]["version"] = version
            
            # Save updated metadata to disk if we have a path
            if "path" in self.model_metadata[function_name]:
                model_dir = os.path.dirname(self.model_metadata[function_name]["path"])
                metadata_path = os.path.join(model_dir, f"{function_name}.metadata")
                
                try:
                    with open(metadata_path, 'w') as f:
                        json.dump(self.model_metadata[function_name], f, indent=2)
                except Exception as e:
                    self.logger.warning(f"Error saving metadata for {function_name}: {e}")
            
            self.logger.info(f"Updated version for {function_name} to {version}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating version for {function_name}: {e}")
            return False