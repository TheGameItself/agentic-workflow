#!/usr/bin/env python3
"""
Reference Models for MCP Core System
Provides access to reference models for natural language and semantic tasks.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Reference model paths
REFERENCE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
GPT2_MODEL_PATH = os.path.join(REFERENCE_MODELS_DIR, "gpt2")
GPT2_SLOPE_MAP_PATH = os.path.join(REFERENCE_MODELS_DIR, "gpt2_slope_maps")

class ReferenceModelManager:
    """Manager for reference models used in MCP."""
    
    def __init__(self, models_dir: str = REFERENCE_MODELS_DIR):
        """Initialize reference model manager."""
        self.models_dir = models_dir
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.slope_maps = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        self.logger = logging.getLogger("reference_models")
    
    def load_gpt2(self, device: str = None) -> Tuple[Any, Any]:
        """Load GPT-2 model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available")
            return None, None
        
        if "gpt2" in self.loaded_models and "gpt2" in self.loaded_tokenizers:
            return self.loaded_models["gpt2"], self.loaded_tokenizers["gpt2"]
        
        try:
            # Determine device
            if device is None:
                device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            
            # Check if model is downloaded
            if not os.path.exists(GPT2_MODEL_PATH):
                self.logger.info("GPT-2 model not found locally, downloading...")
                
                # Download model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                model = AutoModelForCausalLM.from_pretrained("gpt2")
                
                # Save locally
                tokenizer.save_pretrained(GPT2_MODEL_PATH)
                model.save_pretrained(GPT2_MODEL_PATH)
                
                self.logger.info(f"GPT-2 model downloaded and saved to {GPT2_MODEL_PATH}")
            else:
                # Load from local path
                self.logger.info(f"Loading GPT-2 model from {GPT2_MODEL_PATH}")
                tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
                model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_PATH)
            
            # Move model to device
            model = model.to(device)
            
            # Store in cache
            self.loaded_models["gpt2"] = model
            self.loaded_tokenizers["gpt2"] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading GPT-2 model: {e}")
            return None, None
    
    def load_gpt2_slope_maps(self) -> Dict[str, Any]:
        """Load GPT-2 slope maps for weight compression reference."""
        if "gpt2_slope_maps" in self.slope_maps:
            return self.slope_maps["gpt2_slope_maps"]
        
        try:
            # Check if slope maps exist
            if not os.path.exists(GPT2_SLOPE_MAP_PATH):
                self.logger.warning(f"GPT-2 slope maps not found at {GPT2_SLOPE_MAP_PATH}")
                return {}
            
            # Load slope maps
            slope_maps = {}
            for filename in os.listdir(GPT2_SLOPE_MAP_PATH):
                if filename.endswith(".json"):
                    with open(os.path.join(GPT2_SLOPE_MAP_PATH, filename), 'r') as f:
                        slope_data = json.load(f)
                        param_name = filename.replace(".json", "").replace("_", ".")
                        slope_maps[param_name] = slope_data
            
            self.logger.info(f"Loaded {len(slope_maps)} GPT-2 slope maps")
            
            # Store in cache
            self.slope_maps["gpt2_slope_maps"] = slope_maps
            
            return slope_maps
            
        except Exception as e:
            self.logger.error(f"Error loading GPT-2 slope maps: {e}")
            return {}
    
    def generate_text(self, 
                     prompt: str, 
                     model_name: str = "gpt2",
                     max_length: int = 100,
                     temperature: float = 0.7,
                     top_p: float = 0.9) -> str:
        """Generate text using a reference model."""
        if not TRANSFORMERS_AVAILABLE:
            return "Transformers library not available"
        
        try:
            # Load model and tokenizer
            if model_name == "gpt2":
                model, tokenizer = self.load_gpt2()
            else:
                self.logger.error(f"Unknown model: {model_name}")
                return f"Unknown model: {model_name}"
            
            if model is None or tokenizer is None:
                return "Failed to load model or tokenizer"
            
            # Encode prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    def get_embeddings(self, 
                      text: Union[str, List[str]], 
                      model_name: str = "gpt2") -> torch.Tensor:
        """Get embeddings from a reference model."""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        try:
            # Load model and tokenizer
            if model_name == "gpt2":
                model, tokenizer = self.load_gpt2()
            else:
                self.logger.error(f"Unknown model: {model_name}")
                return None
            
            if model is None or tokenizer is None:
                return None
            
            # Prepare input
            if isinstance(text, str):
                text = [text]
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use last hidden state as embeddings
                embeddings = outputs.hidden_states[-1]
                
                # Average over sequence length
                embeddings = embeddings.mean(dim=1)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {e}")
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
            if model_name in self.loaded_tokenizers:
                del self.loaded_tokenizers[model_name]
            
            return True
        
        return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available reference models."""
        available_models = []
        
        # Check for GPT-2
        if os.path.exists(GPT2_MODEL_PATH):
            available_models.append("gpt2")
        
        # Check for slope maps
        if os.path.exists(GPT2_SLOPE_MAP_PATH):
            available_models.append("gpt2_slope_maps")
        
        return available_models

# Global instance
_reference_model_manager: Optional[ReferenceModelManager] = None

def get_reference_model_manager() -> ReferenceModelManager:
    """Get the global reference model manager instance."""
    global _reference_model_manager
    if _reference_model_manager is None:
        _reference_model_manager = ReferenceModelManager()
    return _reference_model_manager