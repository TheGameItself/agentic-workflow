"""
Advanced Diffusion Model with Genetic Integration

Implements state-of-the-art diffusion models that can evolve through genetic
algorithms and share improvements via P2P networks. Supports multiple diffusion
types including DDPM, DDIM, and custom genetic diffusion processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import asyncio
from dataclasses import dataclass
from enum import Enum
import random

from .genetic_diffusion_model import GeneticDiffusionModel, NetworkGene, DiffusionParameters
from ..genetic_data_exchange import GeneticDataExchange, GeneticChromosome, GeneticElement


class DiffusionType(Enum):
    """Types of diffusion processes"""
    DDPM = "ddpm"  # Denoising Diffusion Probabilistic Models
    DDIM = "ddim"  # Denoising Diffusion Implicit Models
    GENETIC = "genetic"  # Genetic-inspired diffusion
    SCORE_BASED = "score_based"  # Score-based generative models
    FLOW_MATCHING = "flow_matching"  # Flow matching models


@dataclass
class GeneticDiffusionConfig:
    """Configuration for genetic diffusion processes"""
    # Standard diffusion parameters
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"  # "linear", "cosine", "sigmoid"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Genetic parameters
    genetic_mutation_rate: float = 0.1
    genetic_crossover_rate: float = 0.3
    population_diversity_weight: float = 0.2
    
    # Advanced features
    self_conditioning: bool = True
    classifier_free_guidance: bool = True
    guidance_scale: float = 7.5
    
    # Genetic-specific features
    chromosome_diffusion: bool = True
    epigenetic_noise: bool = True
    hormone_modulation: bool = True


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time steps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GeneticAttentionBlock(nn.Module):
    """Attention block with genetic encoding"""
    
    def __init__(self, channels: int, gene: NetworkGene):
        super().__init__()
        self.channels = channels
        self.gene = gene
        
        # Attention parameters from gene
        num_heads = gene.parameters.get('num_heads', 8)
        dropout_rate = gene.parameters.get('dropout', 0.1)
        
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Genetic modulation
        self.genetic_scale = nn.Parameter(torch.ones(1) * gene.expression_strength)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        
        # Reshape for attention
        x_norm = self.norm(x)
        x_flat = x_norm.view(batch, channels, -1).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).view(batch, channels, height, width)
        
        # Apply genetic modulation
        attn_out = attn_out * self.genetic_scale
        
        return x + attn_out


class GeneticResBlock(nn.Module):
    """Residual block with genetic encoding"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, gene: NetworkGene):
        super().__init__()
        self.gene = gene
        
        # Block parameters from gene
        dropout_rate = gene.parameters.get('dropout', 0.1)
        activation_type = gene.parameters.get('activation', 'swish')
        use_scale_shift = gene.parameters.get('scale_shift', True)
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            self._get_activation(activation_type),
            nn.Linear(time_emb_dim, out_channels * 2 if use_scale_shift else out_channels)
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        # Genetic modulation
        self.genetic_scale = nn.Parameter(torch.ones(1) * gene.expression_strength)
        self.use_scale_shift = use_scale_shift
        
    def _get_activation(self, activation_type: str) -> nn.Module:
        """Get activation function based on genetic encoding"""
        if activation_type == 'swish':
            return nn.SiLU()
        elif activation_type == 'gelu':
            return nn.GELU()
        elif activation_type == 'relu':
            return nn.ReLU()
        elif activation_type == 'mish':
            return nn.Mish()
        else:
            return nn.SiLU()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First convolution
        h = self.norm1(x)
        h = self._get_activation(self.gene.parameters.get('activation', 'swish'))(h)
        h = self.conv1(h)
        
        # Time embedding
        time_out = self.time_mlp(time_emb)
        
        if self.use_scale_shift:
            # Scale and shift modulation
            scale, shift = time_out.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = h + time_out[:, :, None, None]
        
        # Second convolution
        h = self.norm2(h)
        h = self._get_activation(self.gene.parameters.get('activation', 'swish'))(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Apply genetic modulation and residual connection
        h = h * self.genetic_scale
        return h + residual


class GeneticUNet(nn.Module):
    """U-Net architecture with genetic encoding"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 genetic_exchange: Optional[GeneticDataExchange] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.genetic_exchange = genetic_exchange
        
        # Time embedding
        time_emb_dim = 256
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(128),
            nn.Linear(128, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Genetic components
        self.genes: List[NetworkGene] = []
        self._initialize_genetic_architecture()
        
        # Build U-Net with genetic blocks
        self._build_unet(time_emb_dim)
        
    def _initialize_genetic_architecture(self):
        """Initialize genetic architecture for U-Net"""
        # Encoder genes
        encoder_configs = [
            {'type': 'conv', 'activation': 'swish', 'dropout': 0.1, 'num_heads': 8},
            {'type': 'conv', 'activation': 'gelu', 'dropout': 0.15, 'num_heads': 8},
            {'type': 'conv', 'activation': 'swish', 'dropout': 0.2, 'num_heads': 16},
            {'type': 'conv', 'activation': 'mish', 'dropout': 0.1, 'num_heads': 16}
        ]
        
        # Decoder genes
        decoder_configs = [
            {'type': 'conv', 'activation': 'swish', 'dropout': 0.2, 'num_heads': 16},
            {'type': 'conv', 'activation': 'gelu', 'dropout': 0.15, 'num_heads': 16},
            {'type': 'conv', 'activation': 'swish', 'dropout': 0.1, 'num_heads': 8},
            {'type': 'conv', 'activation': 'swish', 'dropout': 0.05, 'num_heads': 8}
        ]
        
        # Create genes
        for i, config in enumerate(encoder_configs):
            gene = NetworkGene(
                gene_id=f"encoder_{i}",
                gene_type="encoder_block",
                parameters=config,
                expression_strength=random.uniform(0.8, 1.2),
                mutation_rate=0.1,
                crossover_probability=0.3,
                environmental_sensitivity={'performance': 0.5, 'diversity': 0.3}
            )
            self.genes.append(gene)
        
        for i, config in enumerate(decoder_configs):
            gene = NetworkGene(
                gene_id=f"decoder_{i}",
                gene_type="decoder_block",
                parameters=config,
                expression_strength=random.uniform(0.8, 1.2),
                mutation_rate=0.1,
                crossover_probability=0.3,
                environmental_sensitivity={'performance': 0.5, 'diversity': 0.3}
            )
            self.genes.append(gene)
    
    def _build_unet(self, time_emb_dim: int):
        """Build U-Net architecture with genetic blocks"""
        # Channel dimensions
        channels = [64, 128, 256, 512]
        
        # Input projection
        self.input_conv = nn.Conv2d(self.in_channels, channels[0], 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            # Residual blocks
            blocks = nn.ModuleList([
                GeneticResBlock(in_ch if j == 0 else out_ch, out_ch, time_emb_dim, self.genes[i])
                for j in range(2)
            ])
            self.encoder_blocks.append(blocks)
            
            # Attention
            if i >= 2:  # Add attention in deeper layers
                attention = GeneticAttentionBlock(out_ch, self.genes[i])
                self.encoder_attentions.append(attention)
            else:
                self.encoder_attentions.append(nn.Identity())
            
            # Downsample
            if i < len(channels) - 1:
                self.encoder_downsample.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.encoder_downsample.append(nn.Identity())
            
            in_ch = out_ch
        
        # Middle block
        mid_gene = NetworkGene(
            gene_id="middle_block",
            gene_type="middle_block",
            parameters={'activation': 'swish', 'dropout': 0.1, 'num_heads': 16},
            expression_strength=1.0,
            mutation_rate=0.05,
            crossover_probability=0.2,
            environmental_sensitivity={'performance': 0.7}
        )
        self.genes.append(mid_gene)
        
        self.middle_block = nn.ModuleList([
            GeneticResBlock(channels[-1], channels[-1], time_emb_dim, mid_gene),
            GeneticAttentionBlock(channels[-1], mid_gene),
            GeneticResBlock(channels[-1], channels[-1], time_emb_dim, mid_gene)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        decoder_gene_idx = len(encoder_configs)
        for i, out_ch in enumerate(reversed(channels)):
            in_ch = channels[-1-i] + (channels[-2-i] if i > 0 else 0)  # Skip connection
            
            # Residual blocks
            blocks = nn.ModuleList([
                GeneticResBlock(in_ch if j == 0 else out_ch, out_ch, time_emb_dim, 
                              self.genes[decoder_gene_idx + i])
                for j in range(2)
            ])
            self.decoder_blocks.append(blocks)
            
            # Attention
            if i <= 1:  # Add attention in deeper layers
                attention = GeneticAttentionBlock(out_ch, self.genes[decoder_gene_idx + i])
                self.decoder_attentions.append(attention)
            else:
                self.decoder_attentions.append(nn.Identity())
            
            # Upsample
            if i < len(channels) - 1:
                self.decoder_upsample.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.decoder_upsample.append(nn.Identity())
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], self.out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass through genetic U-Net"""
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Input projection
        h = self.input_conv(x)
        
        # Encoder
        encoder_features = []
        for blocks, attention, downsample in zip(
            self.encoder_blocks, self.encoder_attentions, self.encoder_downsample
        ):
            for block in blocks:
                h = block(h, time_emb)
            h = attention(h)
            encoder_features.append(h)
            h = downsample(h)
        
        # Middle block
        for block in self.middle_block:
            if isinstance(block, GeneticResBlock):
                h = block(h, time_emb)
            else:
                h = block(h)
        
        # Decoder
        for i, (blocks, attention, upsample) in enumerate(zip(
            self.decoder_blocks, self.decoder_attentions, self.decoder_upsample
        )):
            # Skip connection
            if i > 0:
                skip_feat = encoder_features[-(i+1)]
                h = torch.cat([h, skip_feat], dim=1)
            
            for block in blocks:
                h = block(h, time_emb)
            h = attention(h)
            h = upsample(h)
        
        # Output projection
        return self.output_conv(h)


class AdvancedDiffusionModel(nn.Module):
    """Advanced diffusion model with genetic evolution capabilities"""
    
    def __init__(self, config: GeneticDiffusionConfig, 
                 in_channels: int = 3, out_channels: int = 3,
                 genetic_exchange: Optional[GeneticDataExchange] = None):
        super().__init__()
        self.config = config
        self.genetic_exchange = genetic_exchange
        
        # Core U-Net model
        self.unet = GeneticUNet(in_channels, out_channels, genetic_exchange)
        
        # Noise schedule
        self.register_buffer('betas', self._create_noise_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                           F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Genetic diffusion components
        if config.chromosome_diffusion:
            self.chromosome_encoder = self._build_chromosome_encoder()
        
        # Self-conditioning
        if config.self_conditioning:
            self.self_cond_prob = 0.5
        
        # Performance tracking
        self.generation = 0
        self.fitness_history = []
        
    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule based on configuration"""
        if self.config.beta_schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, 
                                self.config.num_timesteps)
        elif self.config.beta_schedule == "cosine":
            timesteps = torch.arange(self.config.num_timesteps)
            s = 0.008
            x = (timesteps / self.config.num_timesteps + s) / (1 + s)
            alphas_cumprod = torch.cos(x * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, 0, 0.999)
        elif self.config.beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, self.config.num_timesteps)
            return torch.sigmoid(betas) * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
    
    def _build_chromosome_encoder(self) -> nn.Module:
        """Build encoder for genetic chromosome data"""
        return nn.Sequential(
            nn.Linear(256, 512),  # Assume 256-dim chromosome encoding
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process (add noise)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        # Reshape for broadcasting
        while sqrt_alphas_cumprod_t.dim() < x_start.dim():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None,
                 chromosome_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate training losses"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Self-conditioning
        x_self_cond = None
        if self.config.self_conditioning and random.random() < self.self_cond_prob:
            with torch.no_grad():
                x_self_cond = self.unet(x_noisy, t)
                x_self_cond.detach_()
        
        # Predict noise
        if x_self_cond is not None:
            x_input = torch.cat([x_noisy, x_self_cond], dim=1)
        else:
            x_input = x_noisy
        
        predicted_noise = self.unet(x_input, t)
        
        # Basic denoising loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Genetic diversity loss
        if self.config.population_diversity_weight > 0:
            diversity_loss = self._calculate_diversity_loss()
            loss = loss + self.config.population_diversity_weight * diversity_loss
        
        # Chromosome consistency loss
        if chromosome_data is not None and self.config.chromosome_diffusion:
            chromosome_loss = self._calculate_chromosome_loss(predicted_noise, chromosome_data)
            loss = loss + 0.1 * chromosome_loss
        
        return loss
    
    def _calculate_diversity_loss(self) -> torch.Tensor:
        """Calculate loss to maintain genetic diversity"""
        # Encourage diversity in gene expression
        gene_expressions = []
        for gene in self.unet.genes:
            gene_expressions.append(gene.expression_strength)
        
        if len(gene_expressions) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        gene_tensor = torch.tensor(gene_expressions, device=next(self.parameters()).device)
        
        # Penalize low variance (encourage diversity)
        diversity_loss = 1.0 / (gene_tensor.var() + 1e-8)
        
        return diversity_loss
    
    def _calculate_chromosome_loss(self, predicted_noise: torch.Tensor, 
                                 chromosome_data: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on genetic chromosome data"""
        # Encode chromosome data
        chromosome_features = self.chromosome_encoder(chromosome_data)
        
        # Calculate consistency between predicted noise and chromosome features
        # This is a simplified approach - in practice, this would be more sophisticated
        noise_features = predicted_noise.mean(dim=[2, 3])  # Global average pooling
        
        # Ensure same dimensionality
        if noise_features.size(-1) != chromosome_features.size(-1):
            noise_features = F.adaptive_avg_pool1d(
                noise_features.unsqueeze(1), chromosome_features.size(-1)
            ).squeeze(1)
        
        consistency_loss = F.mse_loss(noise_features, chromosome_features)
        
        return consistency_loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Single denoising step"""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        sqrt_recip_alphas_t = (1.0 / self.alphas[t]).sqrt()
        
        # Reshape for broadcasting
        while betas_t.dim() < x.dim():
            betas_t = betas_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.unsqueeze(-1)
        
        # Predict noise
        predicted_noise = self.unet(x, t)
        
        # Calculate mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance = betas_t
            noise = torch.randn_like(x)
            return model_mean + posterior_variance.sqrt() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, ...], 
                      chromosome_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full denoising loop"""
        device = next(self.parameters()).device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Denoising loop
        for i in reversed(range(0, self.config.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        return img
    
    @torch.no_grad()
    def sample(self, batch_size: int = 1, 
               chromosome_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate samples"""
        shape = (batch_size, self.unet.in_channels, 32, 32)  # Assume 32x32 images
        return self.p_sample_loop(shape, chromosome_data)
    
    async def evolve_architecture(self) -> bool:
        """Evolve the diffusion model architecture"""
        # Mutate genes
        mutated = False
        for gene in self.unet.genes:
            if random.random() < self.config.genetic_mutation_rate:
                # Mutate expression strength
                gene.expression_strength = max(0.1, min(2.0,
                    gene.expression_strength + random.gauss(0, 0.1)))
                
                # Mutate parameters
                if 'dropout' in gene.parameters:
                    gene.parameters['dropout'] = max(0.0, min(0.5,
                        gene.parameters['dropout'] + random.gauss(0, 0.05)))
                
                mutated = True
        
        if mutated:
            self.generation += 1
            print(f"Evolved diffusion model to generation {self.generation}")
        
        return mutated
    
    def encode_to_genetic_data(self) -> Dict[str, Any]:
        """Encode model to genetic data format"""
        # Extract model state
        state_dict = {k: v.cpu().numpy() for k, v in self.state_dict().items()}
        
        # Extract genetic information
        genes_data = []
        for gene in self.unet.genes:
            genes_data.append({
                'gene_id': gene.gene_id,
                'gene_type': gene.gene_type,
                'parameters': gene.parameters,
                'expression_strength': gene.expression_strength,
                'mutation_rate': gene.mutation_rate,
                'crossover_probability': gene.crossover_probability,
                'environmental_sensitivity': gene.environmental_sensitivity
            })
        
        return {
            'model_type': 'genetic_diffusion',
            'config': {
                'num_timesteps': self.config.num_timesteps,
                'beta_schedule': self.config.beta_schedule,
                'beta_start': self.config.beta_start,
                'beta_end': self.config.beta_end,
                'genetic_mutation_rate': self.config.genetic_mutation_rate,
                'genetic_crossover_rate': self.config.genetic_crossover_rate
            },
            'state_dict': state_dict,
            'genes': genes_data,
            'generation': self.generation,
            'fitness_history': self.fitness_history
        }
    
    async def share_genetic_improvements(self) -> bool:
        """Share improvements with P2P network"""
        if not self.genetic_exchange:
            return False
        
        genetic_data = self.encode_to_genetic_data()
        
        packet = self.genetic_exchange.create_genetic_packet(
            "diffusion_model", genetic_data, privacy_level="public"
        )
        
        success = await self.genetic_exchange.share_genetic_data(packet)
        
        if success:
            print(f"Shared diffusion model improvements: Gen {self.generation}")
        
        return success


# Example usage and testing
async def test_genetic_diffusion():
    """Test the genetic diffusion model"""
    # Create configuration
    config = GeneticDiffusionConfig(
        num_timesteps=100,  # Reduced for testing
        beta_schedule="cosine",
        genetic_mutation_rate=0.1,
        chromosome_diffusion=True,
        self_conditioning=True
    )
    
    # Create genetic exchange
    genetic_exchange = GeneticDataExchange("diffusion_test_organism")
    
    # Create model
    model = AdvancedDiffusionModel(config, 3, 3, genetic_exchange)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, config.num_timesteps, (batch_size,))
    
    # Training loss
    loss = model.p_losses(x, t)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    samples = model.sample(batch_size=2)
    print(f"Generated samples shape: {samples.shape}")
    
    # Test evolution
    evolved = await model.evolve_architecture()
    print(f"Evolution: {'Success' if evolved else 'No change'}")
    
    # Test genetic encoding
    genetic_data = model.encode_to_genetic_data()
    print(f"Genetic data keys: {list(genetic_data.keys())}")
    
    # Test P2P sharing
    shared = await model.share_genetic_improvements()
    print(f"P2P sharing: {'Success' if shared else 'Failed'}")


if __name__ == "__main__":
    asyncio.run(test_genetic_diffusion())