"""
Genetic Diffusion Model

Advanced neural network that uses genetic algorithms for architecture evolution
and diffusion processes for parameter optimization. Integrates with the genetic
data exchange system for P2P evolution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
import asyncio
from dataclasses import dataclass
import json
import time
from collections import defaultdict

from ..genetic_data_exchange import (
    GeneticDataExchange, GeneticChromosome, GeneticElement, 
    GeneticElementType, ChromatinState, EpigeneticMarker
)


@dataclass
class NetworkGene:
    """Genetic representation of neural network components"""
    gene_id: str
    gene_type: str  # 'layer', 'connection', 'activation', 'optimizer'
    parameters: Dict[str, Any]
    expression_strength: float
    mutation_rate: float
    crossover_probability: float
    environmental_sensitivity: Dict[str, float]


@dataclass
class DiffusionParameters:
    """Parameters for diffusion-based optimization"""
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # "linear", "cosine", "exponential"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    guidance_scale: float = 7.5


class GeneticNeuralLayer(nn.Module):
    """Neural layer with genetic encoding capabilities"""
    
    def __init__(self, gene: NetworkGene, input_dim: int, output_dim: int):
        super().__init__()
        self.gene = gene
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize layer based on genetic parameters
        self.layer_type = gene.parameters.get('type', 'linear')
        self.activation_type = gene.parameters.get('activation', 'relu')
        self.dropout_rate = gene.parameters.get('dropout', 0.0)
        
        # Create layer components
        if self.layer_type == 'linear':
            self.linear = nn.Linear(input_dim, output_dim)
        elif self.layer_type == 'conv1d':
            kernel_size = gene.parameters.get('kernel_size', 3)
            self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size//2)
        elif self.layer_type == 'attention':
            self.attention = nn.MultiheadAttention(input_dim, 
                                                 gene.parameters.get('num_heads', 8))
            self.linear = nn.Linear(input_dim, output_dim)
        
        # Dropout layer
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        # Normalization
        if gene.parameters.get('batch_norm', False):
            self.batch_norm = nn.BatchNorm1d(output_dim)
        
        # Initialize weights based on genetic encoding
        self._initialize_genetic_weights()
    
    def _initialize_genetic_weights(self):
        """Initialize weights based on genetic encoding"""
        # Use genetic sequence to influence weight initialization
        genetic_seed = hash(self.gene.gene_id) % 2**32
        torch.manual_seed(genetic_seed)
        
        # Apply expression strength to initialization scale
        init_scale = self.gene.expression_strength
        
        if hasattr(self, 'linear'):
            nn.init.xavier_normal_(self.linear.weight, gain=init_scale)
            if self.linear.bias is not None:
                nn.init.constant_(self.linear.bias, 0)
        
        if hasattr(self, 'conv'):
            nn.init.xavier_normal_(self.conv.weight, gain=init_scale)
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with genetic-influenced computation"""
        # Apply environmental modulation
        environmental_factor = self._calculate_environmental_factor()
        
        if self.layer_type == 'linear':
            x = self.linear(x)
        elif self.layer_type == 'conv1d':
            x = self.conv(x)
        elif self.layer_type == 'attention':
            x, _ = self.attention(x, x, x)
            x = self.linear(x)
        
        # Apply batch normalization if present
        if hasattr(self, 'batch_norm') and x.dim() >= 2:
            if x.dim() == 3:  # Sequence data
                x = x.transpose(1, 2)
                x = self.batch_norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.batch_norm(x)
        
        # Apply activation
        x = self._apply_activation(x)
        
        # Apply dropout
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        # Apply environmental modulation
        x = x * environmental_factor
        
        return x
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function based on genetic encoding"""
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'gelu':
            return F.gelu(x)
        elif self.activation_type == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation_type == 'mish':
            return x * torch.tanh(F.softplus(x))
        elif self.activation_type == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.01)
        else:
            return x
    
    def _calculate_environmental_factor(self) -> float:
        """Calculate environmental modulation factor"""
        # This would be connected to the hormone system in practice
        base_factor = 1.0
        
        # Simulate environmental influence based on gene sensitivity
        for env_var, sensitivity in self.gene.environmental_sensitivity.items():
            # In practice, this would read from actual environmental state
            env_value = random.uniform(0.8, 1.2)  # Simulated environmental value
            factor_change = (env_value - 1.0) * sensitivity
            base_factor += factor_change
        
        return max(0.1, min(2.0, base_factor))  # Clamp to reasonable range
    
    def mutate(self, mutation_strength: float = 0.1) -> 'GeneticNeuralLayer':
        """Create mutated version of this layer"""
        # Mutate gene parameters
        new_gene = NetworkGene(
            gene_id=f"mut_{self.gene.gene_id}",
            gene_type=self.gene.gene_type,
            parameters=self.gene.parameters.copy(),
            expression_strength=max(0.1, min(2.0, 
                self.gene.expression_strength + random.gauss(0, mutation_strength))),
            mutation_rate=self.gene.mutation_rate,
            crossover_probability=self.gene.crossover_probability,
            environmental_sensitivity=self.gene.environmental_sensitivity.copy()
        )
        
        # Mutate specific parameters
        if random.random() < self.gene.mutation_rate:
            if 'dropout' in new_gene.parameters:
                new_gene.parameters['dropout'] = max(0.0, min(0.8,
                    new_gene.parameters['dropout'] + random.gauss(0, 0.1)))
        
        # Create new layer with mutated gene
        new_layer = GeneticNeuralLayer(new_gene, self.input_dim, self.output_dim)
        
        # Copy and mutate weights
        if hasattr(self, 'linear') and hasattr(new_layer, 'linear'):
            with torch.no_grad():
                new_layer.linear.weight.copy_(self.linear.weight)
                new_layer.linear.bias.copy_(self.linear.bias)
                
                # Apply mutation to weights
                weight_mutation = torch.randn_like(new_layer.linear.weight) * mutation_strength
                new_layer.linear.weight.add_(weight_mutation)
                
                bias_mutation = torch.randn_like(new_layer.linear.bias) * mutation_strength
                new_layer.linear.bias.add_(bias_mutation)
        
        return new_layer


class DiffusionScheduler:
    """Noise scheduler for diffusion-based optimization"""
    
    def __init__(self, params: DiffusionParameters):
        self.params = params
        self.betas = self._create_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule for diffusion process"""
        if self.params.noise_schedule == "linear":
            return torch.linspace(self.params.beta_start, self.params.beta_end, 
                                self.params.diffusion_steps)
        elif self.params.noise_schedule == "cosine":
            timesteps = torch.arange(self.params.diffusion_steps)
            s = 0.008
            x = (timesteps / self.params.diffusion_steps + s) / (1 + s)
            alphas_cumprod = torch.cos(x * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, 0, 0.999)
        elif self.params.noise_schedule == "exponential":
            return torch.exp(torch.linspace(np.log(self.params.beta_start), 
                                          np.log(self.params.beta_end),
                                          self.params.diffusion_steps))
        else:
            raise ValueError(f"Unknown noise schedule: {self.params.noise_schedule}")
    
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to diffusion schedule"""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[timesteps])
        
        # Reshape for broadcasting
        while sqrt_alphas_cumprod.dim() < x0.dim():
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
        
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    
    def step(self, model_output: torch.Tensor, timestep: int, 
             sample: torch.Tensor) -> torch.Tensor:
        """Perform one denoising step"""
        t = timestep
        
        # Get schedule values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        
        if self.params.prediction_type == "epsilon":
            # Predict original sample from noise
            pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.params.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.params.prediction_type == "v_prediction":
            pred_original_sample = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(beta_prod_t) * model_output
        
        # Compute coefficients for pred_original_sample and current sample
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        return prev_sample


class GeneticDiffusionModel(nn.Module):
    """Neural network model that combines genetic evolution with diffusion optimization"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 genetic_exchange: Optional[GeneticDataExchange] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.genetic_exchange = genetic_exchange
        
        # Diffusion parameters
        self.diffusion_params = DiffusionParameters()
        self.scheduler = DiffusionScheduler(self.diffusion_params)
        
        # Genetic components
        self.genes: List[NetworkGene] = []
        self.layers: nn.ModuleList = nn.ModuleList()
        
        # Performance tracking
        self.performance_history = []
        self.generation = 0
        
        # Initialize with basic architecture
        self._initialize_genetic_architecture()
        
        # Time embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
    def _initialize_genetic_architecture(self):
        """Initialize neural architecture from genetic encoding"""
        # Create basic genes for initial architecture
        layer_configs = [
            {'type': 'linear', 'activation': 'relu', 'dropout': 0.1},
            {'type': 'linear', 'activation': 'gelu', 'dropout': 0.2},
            {'type': 'linear', 'activation': 'swish', 'dropout': 0.1}
        ]
        
        current_dim = self.input_dim
        for i, config in enumerate(layer_configs):
            next_dim = self.output_dim if i == len(layer_configs) - 1 else max(64, current_dim // 2)
            
            gene = NetworkGene(
                gene_id=f"layer_{i}",
                gene_type="layer",
                parameters=config,
                expression_strength=1.0,
                mutation_rate=0.1,
                crossover_probability=0.3,
                environmental_sensitivity={'performance': 0.5, 'complexity': 0.3}
            )
            
            layer = GeneticNeuralLayer(gene, current_dim, next_dim)
            
            self.genes.append(gene)
            self.layers.append(layer)
            current_dim = next_dim
    
    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get time embedding for diffusion process"""
        # Sinusoidal position embedding
        half_dim = 64
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)
    
    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through genetic neural network"""
        # Add time conditioning for diffusion
        if timesteps is not None:
            time_emb = self.get_time_embedding(timesteps)
            # Add time embedding to input (simplified approach)
            if x.dim() == 2 and time_emb.dim() == 2:
                x = x + time_emb[:, :x.size(1)]
        
        # Pass through genetic layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def diffusion_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Calculate diffusion training loss"""
        batch_size = x0.size(0)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.diffusion_params.diffusion_steps, 
                                (batch_size,), device=x0.device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Add noise to clean data
        noisy_x = self.scheduler.add_noise(x0, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.forward(noisy_x, timesteps)
        
        # Calculate loss
        if self.diffusion_params.prediction_type == "epsilon":
            target = noise
        elif self.diffusion_params.prediction_type == "sample":
            target = x0
        elif self.diffusion_params.prediction_type == "v_prediction":
            sqrt_alphas_cumprod = torch.sqrt(self.scheduler.alphas_cumprod[timesteps])
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.scheduler.alphas_cumprod[timesteps])
            target = sqrt_alphas_cumprod * noise - sqrt_one_minus_alphas_cumprod * x0
        
        loss = F.mse_loss(predicted_noise, target)
        return loss
    
    def generate_sample(self, shape: Tuple[int, ...], num_inference_steps: int = 50) -> torch.Tensor:
        """Generate sample using diffusion process"""
        device = next(self.parameters()).device
        
        # Start with random noise
        sample = torch.randn(shape, device=device)
        
        # Denoising loop
        timesteps = torch.linspace(self.diffusion_params.diffusion_steps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, device=device)
        
        for t in timesteps:
            with torch.no_grad():
                # Predict noise
                timestep_batch = t.repeat(shape[0])
                model_output = self.forward(sample, timestep_batch)
                
                # Perform denoising step
                sample = self.scheduler.step(model_output, t.item(), sample)
        
        return sample
    
    async def evolve_architecture(self, performance_threshold: float = 0.8) -> bool:
        """Evolve neural architecture based on genetic algorithms"""
        current_performance = self._calculate_current_performance()
        
        if current_performance < performance_threshold:
            # Apply mutations
            mutated_layers = []
            for layer in self.layers:
                if random.random() < layer.gene.mutation_rate:
                    mutated_layer = layer.mutate()
                    mutated_layers.append(mutated_layer)
                else:
                    mutated_layers.append(layer)
            
            # Test mutated architecture
            old_layers = self.layers
            self.layers = nn.ModuleList(mutated_layers)
            
            # Evaluate performance (simplified)
            new_performance = self._calculate_current_performance()
            
            if new_performance > current_performance:
                # Keep mutations
                self.generation += 1
                print(f"Evolution successful: Gen {self.generation}, "
                      f"Performance: {current_performance:.3f} -> {new_performance:.3f}")
                return True
            else:
                # Revert mutations
                self.layers = old_layers
                return False
        
        return False
    
    def _calculate_current_performance(self) -> float:
        """Calculate current model performance"""
        # Simplified performance calculation
        # In practice, this would use validation data
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        param_efficiency = 1.0 / (1.0 + total_params / 1000000)  # Prefer smaller models
        
        # Expression strength average
        avg_expression = sum(gene.expression_strength for gene in self.genes) / len(self.genes)
        
        # Combine metrics
        performance = (param_efficiency + avg_expression) / 2.0
        
        # Add some noise to simulate real performance variation
        performance += random.gauss(0, 0.05)
        
        return max(0.0, min(1.0, performance))
    
    def encode_to_genetic_data(self) -> Dict[str, Any]:
        """Encode current model state to genetic data format"""
        # Extract model weights
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        
        # Extract architecture information
        architecture = {}
        for i, gene in enumerate(self.genes):
            architecture[f"layer_{i}"] = gene.parameters
        
        # Performance metrics
        performance_metrics = {
            'current_performance': self._calculate_current_performance(),
            'generation': self.generation,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'avg_expression_strength': sum(g.expression_strength for g in self.genes) / len(self.genes)
        }
        
        return {
            'model_id': f"genetic_diffusion_{id(self)}",
            'architecture': architecture,
            'weights': weights,
            'diffusion_params': {
                'diffusion_steps': self.diffusion_params.diffusion_steps,
                'noise_schedule': self.diffusion_params.noise_schedule,
                'beta_start': self.diffusion_params.beta_start,
                'beta_end': self.diffusion_params.beta_end
            },
            'performance_metrics': performance_metrics,
            'genes': [
                {
                    'gene_id': gene.gene_id,
                    'gene_type': gene.gene_type,
                    'parameters': gene.parameters,
                    'expression_strength': gene.expression_strength,
                    'mutation_rate': gene.mutation_rate,
                    'crossover_probability': gene.crossover_probability,
                    'environmental_sensitivity': gene.environmental_sensitivity
                }
                for gene in self.genes
            ]
        }
    
    def load_from_genetic_data(self, genetic_data: Dict[str, Any]) -> bool:
        """Load model state from genetic data"""
        try:
            # Load genes
            self.genes = []
            for gene_data in genetic_data.get('genes', []):
                gene = NetworkGene(
                    gene_id=gene_data['gene_id'],
                    gene_type=gene_data['gene_type'],
                    parameters=gene_data['parameters'],
                    expression_strength=gene_data['expression_strength'],
                    mutation_rate=gene_data['mutation_rate'],
                    crossover_probability=gene_data['crossover_probability'],
                    environmental_sensitivity=gene_data['environmental_sensitivity']
                )
                self.genes.append(gene)
            
            # Rebuild architecture
            self.layers = nn.ModuleList()
            current_dim = self.input_dim
            
            for i, gene in enumerate(self.genes):
                next_dim = self.output_dim if i == len(self.genes) - 1 else max(64, current_dim // 2)
                layer = GeneticNeuralLayer(gene, current_dim, next_dim)
                self.layers.append(layer)
                current_dim = next_dim
            
            # Load weights if available
            weights = genetic_data.get('weights', {})
            if weights:
                state_dict = {}
                for name, weight_array in weights.items():
                    state_dict[name] = torch.from_numpy(weight_array)
                
                # Load compatible weights
                current_state = self.state_dict()
                for name, param in state_dict.items():
                    if name in current_state and current_state[name].shape == param.shape:
                        current_state[name].copy_(param)
            
            # Load diffusion parameters
            diffusion_params = genetic_data.get('diffusion_params', {})
            if diffusion_params:
                self.diffusion_params.diffusion_steps = diffusion_params.get('diffusion_steps', 1000)
                self.diffusion_params.noise_schedule = diffusion_params.get('noise_schedule', 'cosine')
                self.diffusion_params.beta_start = diffusion_params.get('beta_start', 0.0001)
                self.diffusion_params.beta_end = diffusion_params.get('beta_end', 0.02)
                
                # Recreate scheduler
                self.scheduler = DiffusionScheduler(self.diffusion_params)
            
            # Load performance metrics
            performance_metrics = genetic_data.get('performance_metrics', {})
            self.generation = performance_metrics.get('generation', 0)
            
            return True
        except Exception as e:
            print(f"Error loading genetic data: {e}")
            return False
    
    async def share_genetic_improvements(self) -> bool:
        """Share genetic improvements with P2P network"""
        if not self.genetic_exchange:
            return False
        
        # Encode current state
        genetic_data = self.encode_to_genetic_data()
        
        # Create genetic packet
        packet = self.genetic_exchange.create_genetic_packet(
            "neural_network", genetic_data, privacy_level="public"
        )
        
        # Share with network
        success = await self.genetic_exchange.share_genetic_data(packet)
        
        if success:
            print(f"Shared genetic improvements: Gen {self.generation}, "
                  f"Performance: {genetic_data['performance_metrics']['current_performance']:.3f}")
        
        return success
    
    def crossover_with_model(self, other_model: 'GeneticDiffusionModel') -> 'GeneticDiffusionModel':
        """Perform genetic crossover with another model"""
        # Create offspring model
        offspring = GeneticDiffusionModel(self.input_dim, self.output_dim, self.genetic_exchange)
        
        # Crossover genes
        offspring_genes = []
        min_genes = min(len(self.genes), len(other_model.genes))
        
        for i in range(min_genes):
            if random.random() < self.genes[i].crossover_probability:
                # Take gene from other model
                parent_gene = other_model.genes[i]
            else:
                # Take gene from self
                parent_gene = self.genes[i]
            
            # Create offspring gene with potential mutations
            offspring_gene = NetworkGene(
                gene_id=f"cross_{parent_gene.gene_id}_{i}",
                gene_type=parent_gene.gene_type,
                parameters=parent_gene.parameters.copy(),
                expression_strength=parent_gene.expression_strength,
                mutation_rate=parent_gene.mutation_rate,
                crossover_probability=parent_gene.crossover_probability,
                environmental_sensitivity=parent_gene.environmental_sensitivity.copy()
            )
            
            offspring_genes.append(offspring_gene)
        
        # Build offspring architecture
        offspring.genes = offspring_genes
        offspring.layers = nn.ModuleList()
        
        current_dim = offspring.input_dim
        for i, gene in enumerate(offspring_genes):
            next_dim = offspring.output_dim if i == len(offspring_genes) - 1 else max(64, current_dim // 2)
            layer = GeneticNeuralLayer(gene, current_dim, next_dim)
            offspring.layers.append(layer)
            current_dim = next_dim
        
        return offspring


# Example usage and testing
async def test_genetic_diffusion_model():
    """Test the genetic diffusion model"""
    # Create genetic exchange system
    genetic_exchange = GeneticDataExchange("test_organism")
    
    # Create model
    model = GeneticDiffusionModel(784, 10, genetic_exchange)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    # Standard forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Diffusion forward pass
    timesteps = torch.randint(0, 1000, (batch_size,))
    diffusion_output = model(x, timesteps)
    print(f"Diffusion output shape: {diffusion_output.shape}")
    
    # Test diffusion loss
    loss = model.diffusion_loss(x)
    print(f"Diffusion loss: {loss.item():.4f}")
    
    # Test evolution
    print("Testing evolution...")
    for i in range(5):
        evolved = await model.evolve_architecture()
        print(f"Evolution step {i+1}: {'Success' if evolved else 'No change'}")
    
    # Test genetic encoding/decoding
    genetic_data = model.encode_to_genetic_data()
    print(f"Encoded genetic data keys: {list(genetic_data.keys())}")
    
    # Create new model and load genetic data
    new_model = GeneticDiffusionModel(784, 10)
    success = new_model.load_from_genetic_data(genetic_data)
    print(f"Genetic data loading: {'Success' if success else 'Failed'}")
    
    # Test crossover
    offspring = model.crossover_with_model(new_model)
    print(f"Crossover offspring genes: {len(offspring.genes)}")
    
    # Test P2P sharing
    shared = await model.share_genetic_improvements()
    print(f"P2P sharing: {'Success' if shared else 'Failed'}")


if __name__ == "__main__":
    asyncio.run(test_genetic_diffusion_model())