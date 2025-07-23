#!/usr/bin/env python3
"""
Neural Network Pretraining Script for MCP Core System
Pretrains all neural network models used in the system.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
import json
import numpy as np
import random

# Add core/src to path
core_src = Path(__file__).parent / "src"
sys.path.insert(0, str(core_src))

# Check for required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/pretrain.log")
    ]
)

logger = logging.getLogger("pretrain")

# Import neural models
from mcp.neural_network_models.pretrain import NeuralNetworkPretrainer, PretrainConfig
from mcp.neural_network_models.diffusion_model import DiffusionModel, DiffusionConfig
from mcp.neural_network_models.genetic_diffusion_model import GeneticDiffusionModel, GeneticConfig
from mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration, HormoneConfig
from mcp.neural_network_models.brain_state_integration import BrainStateIntegration, BrainStateConfig

# Import core system for data access
from mcp.core_system import initialize_core_system, shutdown_core_system, SystemConfiguration

async def generate_training_data(args):
    """Generate training data from system memory and metrics."""
    logger.info("Initializing core system to extract training data...")
    
    # Initialize core system
    config = SystemConfiguration(
        max_workers=2,
        enable_async=True,
        enable_monitoring=True,
        log_level="INFO",
        data_directory="data",
        backup_enabled=False,
        performance_optimization=False,
        experimental_features=False,
        hormone_system_enabled=True
    )
    
    system = await initialize_core_system(config)
    
    if not system:
        logger.error("Failed to initialize core system")
        return None, None
    
    try:
        # Extract memory data
        logger.info("Extracting memory data...")
        memory_request = {
            'method': 'memory/search',
            'params': {
                'query': '',
                'limit': 10000
            }
        }
        
        memory_response = await system.execute_request(memory_request)
        
        if not memory_response.get('success'):
            logger.warning("Failed to retrieve memories")
            memories = []
        else:
            memories = memory_response.get('results', [])
        
        logger.info(f"Retrieved {len(memories)} memories")
        
        # Extract metrics history
        logger.info("Extracting metrics history...")
        metrics_history = []
        
        # Get current metrics
        current_metrics = system.get_metrics().__dict__
        metrics_history.append(current_metrics)
        
        # Get hormone levels
        hormone_levels = system.hormone_levels
        
        # Get performance monitor metrics if available
        if system.performance_monitor:
            all_metrics = system.performance_monitor.get_all_metrics()
            for metric_name, stats in all_metrics.items():
                if 'current' in stats:
                    current_metrics[metric_name] = stats['current']
        
        # Generate synthetic metrics history
        logger.info("Generating synthetic metrics history...")
        base_metrics = current_metrics.copy()
        
        for i in range(100):
            synthetic_metrics = base_metrics.copy()
            
            # Add random variations
            for key in synthetic_metrics:
                if isinstance(synthetic_metrics[key], (int, float)) and key != 'last_updated':
                    # Add random variation (±20%)
                    variation = synthetic_metrics[key] * 0.2
                    synthetic_metrics[key] += random.uniform(-variation, variation)
                    
                    # Ensure non-negative values
                    if key in ['cpu_usage', 'memory_usage']:
                        synthetic_metrics[key] = max(0.0, min(100.0, synthetic_metrics[key]))
                    else:
                        synthetic_metrics[key] = max(0.0, synthetic_metrics[key])
            
            metrics_history.append(synthetic_metrics)
        
        logger.info(f"Generated {len(metrics_history)} metric entries")
        
        # Process memories into embeddings
        embeddings = []
        texts = []
        
        for memory in memories:
            text = memory.get('text', '')
            if text and len(text.strip()) > 10:  # Filter out very short texts
                texts.append(text)
        
        # Generate synthetic embeddings if no real data available
        if not texts or len(texts) < 100:
            logger.info("Generating synthetic embeddings...")
            
            # Generate random embeddings
            embedding_dim = args.embedding_dim
            num_embeddings = max(100, len(texts))
            
            synthetic_embeddings = torch.randn(num_embeddings, embedding_dim)
            embeddings = synthetic_embeddings
            
            # Generate synthetic texts if needed
            if len(texts) < 100:
                logger.info("Generating synthetic texts...")
                
                # Simple synthetic texts
                words = ["system", "memory", "workflow", "task", "context", "performance", 
                         "optimization", "neural", "network", "brain", "hormone", "diffusion",
                         "genetic", "integration", "model", "training", "data", "embedding",
                         "vector", "tensor", "matrix", "algorithm", "function", "process"]
                
                for i in range(100 - len(texts)):
                    # Generate random text
                    text_length = random.randint(5, 15)
                    text = " ".join(random.choices(words, k=text_length))
                    texts.append(text)
        
        # Return data
        return {
            'texts': texts,
            'embeddings': embeddings,
            'metrics_history': metrics_history,
            'hormone_levels': hormone_levels
        }
        
    finally:
        # Shutdown core system
        await shutdown_core_system()

def pretrain_embedding_model(data, args):
    """Pretrain embedding model."""
    logger.info("Pretraining embedding model...")
    
    config = PretrainConfig(
        model_type="embedding",
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    pretrainer = NeuralNetworkPretrainer(config)
    
    # Train on texts
    if data['texts']:
        pretrainer.pretrain_embedding_model(texts=data['texts'])
        logger.info("Embedding model pretraining completed")
        return True
    else:
        logger.warning("No text data available for embedding model pretraining")
        return False

def pretrain_classification_model(data, args):
    """Pretrain classification model."""
    logger.info("Pretraining classification model...")
    
    config = PretrainConfig(
        model_type="classification",
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    pretrainer = NeuralNetworkPretrainer(config)
    
    # Generate labels (use memory types or random labels)
    if data['texts']:
        # Generate random labels for demonstration
        labels = [random.randint(0, 3) for _ in range(len(data['texts']))]
        
        pretrainer.pretrain_classification_model(texts=data['texts'], labels=labels)
        logger.info("Classification model pretraining completed")
        return True
    else:
        logger.warning("No text data available for classification model pretraining")
        return False

def pretrain_similarity_model(data, args):
    """Pretrain similarity model."""
    logger.info("Pretraining similarity model...")
    
    config = PretrainConfig(
        model_type="similarity",
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    pretrainer = NeuralNetworkPretrainer(config)
    
    # Train on texts
    if data['texts']:
        pretrainer.pretrain_similarity_model(texts=data['texts'])
        logger.info("Similarity model pretraining completed")
        return True
    else:
        logger.warning("No text data available for similarity model pretraining")
        return False

def pretrain_diffusion_model(data, args):
    """Pretrain diffusion model."""
    logger.info("Pretraining diffusion model...")
    
    config = DiffusionConfig(
        input_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        timesteps=100,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    diffusion_model = DiffusionModel(config)
    
    # Train on embeddings
    if isinstance(data['embeddings'], torch.Tensor) and data['embeddings'].shape[0] > 0:
        diffusion_model.train(data['embeddings'])
        logger.info("Diffusion model pretraining completed")
        return True
    else:
        # Generate random embeddings
        logger.info("Generating random embeddings for diffusion model...")
        embeddings = torch.randn(100, args.embedding_dim)
        diffusion_model.train(embeddings)
        logger.info("Diffusion model pretraining completed with synthetic data")
        return True

def pretrain_genetic_model(data, args):
    """Pretrain genetic diffusion model."""
    logger.info("Pretraining genetic diffusion model...")
    
    diffusion_config = DiffusionConfig(
        input_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        timesteps=100,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    genetic_config = GeneticConfig(
        population_size=50,
        generations=10
    )
    
    genetic_model = GeneticDiffusionModel(diffusion_config, genetic_config)
    
    # Initialize population with embeddings
    if isinstance(data['embeddings'], torch.Tensor) and data['embeddings'].shape[0] > 0:
        genetic_model.initialize_population(data['embeddings'])
    else:
        # Generate random embeddings
        logger.info("Generating random embeddings for genetic model...")
        embeddings = torch.randn(50, args.embedding_dim)
        genetic_model.initialize_population(embeddings)
    
    # Define simple fitness function
    def fitness_function(individual):
        # Simple fitness function based on L2 norm
        return 1.0 / (1.0 + torch.norm(individual).item())
    
    # Evolve for a few generations
    genetic_model.evolve(fitness_function, generations=5)
    
    # Train diffusion model on population
    genetic_model.train_diffusion_model()
    
    # Save model
    genetic_model.save_model(f"genetic-{int(time.time())}")
    
    logger.info("Genetic diffusion model pretraining completed")
    return True

def pretrain_hormone_integration(data, args):
    """Pretrain hormone neural integration."""
    logger.info("Pretraining hormone neural integration...")
    
    config = HormoneConfig(
        input_dim=len(data['metrics_history'][0]) if data['metrics_history'] else 10,
        hidden_dim=args.hidden_dim,
        output_dim=4,  # Four hormone types
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    hormone_integration = HormoneNeuralIntegration(config)
    
    # Generate training data
    if data['metrics_history']:
        inputs, outputs = hormone_integration.generate_training_data(data['metrics_history'])
        
        if inputs.shape[0] > 0:
            hormone_integration.train(inputs, outputs)
            logger.info("Hormone neural integration pretraining completed")
            return True
    
    logger.warning("No metrics history available for hormone integration pretraining")
    return False

def pretrain_brain_state_integration(data, args):
    """Pretrain brain state integration."""
    logger.info("Pretraining brain state integration...")
    
    config = BrainStateConfig(
        embedding_dim=args.embedding_dim,
        state_dim=args.hidden_dim,
        hormone_integration=True,
        diffusion_model=True,
        genetic_model=False,
        output_dir=args.output_dir
    )
    
    brain_state = BrainStateIntegration(config)
    
    # Pretrain all components
    brain_state.pretrain_all_components(
        embeddings=data['embeddings'] if isinstance(data['embeddings'], torch.Tensor) else None,
        metrics_history=data['metrics_history']
    )
    
    logger.info("Brain state integration pretraining completed")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Neural Network Pretraining for MCP Core System")
    
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", "embedding", "classification", "similarity", 
                                "diffusion", "genetic", "hormone", "brain_state"],
                        help="Models to pretrain")
    
    parser.add_argument("--base-model", default="all-MiniLM-L6-v2",
                        help="Base model for transformer-based models")
    
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension")
    
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension")
    
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    
    parser.add_argument("--output-dir", default="data/models",
                        help="Output directory for models")
    
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data only")
    
    args = parser.parse_args()
    
    # Check for PyTorch
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for neural network pretraining")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate or load training data
    if args.synthetic:
        logger.info("Using synthetic data for pretraining")
        
        # Generate synthetic data
        data = {
            'texts': [f"Synthetic text {i}" for i in range(100)],
            'embeddings': torch.randn(100, args.embedding_dim),
            'metrics_history': [
                {f'metric_{j}': random.random() for j in range(10)}
                for i in range(100)
            ],
            'hormone_levels': {
                'stress': 0.0,
                'efficiency': 1.0,
                'adaptation': 0.5,
                'stability': 1.0
            }
        }
    else:
        # Extract data from system
        data = asyncio.run(generate_training_data(args))
        
        if not data:
            logger.error("Failed to generate training data")
            return 1
    
    # Determine which models to pretrain
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["embedding", "classification", "similarity", 
                          "diffusion", "genetic", "hormone", "brain_state"]
    
    # Pretrain models
    results = {}
    
    if "embedding" in models_to_train:
        results["embedding"] = pretrain_embedding_model(data, args)
    
    if "classification" in models_to_train:
        results["classification"] = pretrain_classification_model(data, args)
    
    if "similarity" in models_to_train:
        results["similarity"] = pretrain_similarity_model(data, args)
    
    if "diffusion" in models_to_train:
        results["diffusion"] = pretrain_diffusion_model(data, args)
    
    if "genetic" in models_to_train:
        results["genetic"] = pretrain_genetic_model(data, args)
    
    if "hormone" in models_to_train:
        results["hormone"] = pretrain_hormone_integration(data, args)
    
    if "brain_state" in models_to_train:
        results["brain_state"] = pretrain_brain_state_integration(data, args)
    
    # Print results
    logger.info("Pretraining results:")
    for model, success in results.items():
        logger.info(f"  {model}: {'Success' if success else 'Failed'}")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"pretrain_results_{int(time.time())}.json")
    with open(results_path, "w") as f:
        json.dump({
            'timestamp': time.time(),
            'args': vars(args),
            'results': {k: bool(v) for k, v in results.items()}
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Return success if all requested models were trained successfully
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())