#!/usr/bin/env python3
"""
Neural Network Models Package for MCP Core System
Provides neural network models for various MCP components.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Check for required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    return {
        'torch': TORCH_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE
    }

def get_available_models(models_dir: str = "data/models") -> Dict[str, List[str]]:
    """Get available pretrained models."""
    models = {
        'embedding': [],
        'classification': [],
        'similarity': [],
        'diffusion': [],
        'genetic': []
    }
    
    try:
        if not os.path.exists(models_dir):
            return models
            
        for model_type in models.keys():
            model_prefix = f"{model_type}-"
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path) and item.startswith(model_prefix):
                    models[model_type].append(item)
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
    
    return models

def get_default_model_path(model_type: str) -> Optional[str]:
    """Get the path to the default model of the specified type."""
    models_dir = "data/models"
    models = get_available_models(models_dir)
    
    if model_type in models and models[model_type]:
        # Sort by timestamp (assuming format model_type-timestamp)
        sorted_models = sorted(models[model_type], reverse=True)
        return os.path.join(models_dir, sorted_models[0])
    
    return None

def load_embedding_model(model_path: Optional[str] = None) -> Optional[Any]:
    """Load an embedding model for semantic search."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("Sentence Transformers not available. Please install with: pip install sentence-transformers")
        return None
    
    try:
        if model_path is None:
            model_path = get_default_model_path('embedding')
        
        if model_path is None:
            # Use default pretrained model
            model_path = 'all-MiniLM-L6-v2'
        
        model = SentenceTransformer(model_path)
        logger.info(f"Loaded embedding model from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

def load_classification_model(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a classification model."""
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers not available. Please install with: pip install transformers")
        return None
    
    try:
        if model_path is None:
            model_path = get_default_model_path('classification')
        
        if model_path is None:
            logger.error("No classification model available")
            return None
        
        from transformers import AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load label mapping if available
        label_mapping = {}
        label_mapping_path = os.path.join(model_path, "label_mapping.json")
        if os.path.exists(label_mapping_path):
            import json
            with open(label_mapping_path, "r") as f:
                label_mapping = json.load(f)
        
        logger.info(f"Loaded classification model from {model_path}")
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_mapping': label_mapping
        }
        
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        return None

def load_similarity_model(model_path: Optional[str] = None) -> Optional[Any]:
    """Load a similarity model for text comparison."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("Sentence Transformers not available. Please install with: pip install sentence-transformers")
        return None
    
    try:
        if model_path is None:
            model_path = get_default_model_path('similarity')
        
        if model_path is None:
            # Use default pretrained model
            model_path = 'all-MiniLM-L6-v2'
        
        model = SentenceTransformer(model_path)
        logger.info(f"Loaded similarity model from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading similarity model: {e}")
        return None