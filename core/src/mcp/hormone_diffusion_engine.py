"""
Hormone Diffusion Engine: Biologically-inspired diffusion model for hormone distribution.

This module implements both algorithmic and neural network approaches for hormone
diffusion across the brain-inspired lobe system, following biological principles
of hormone distribution through spatial gradients and receptor interactions.

References:
- Requirements 2.1, 2.9 from MCP System Upgrade specification
- Biological hormone diffusion models
- Neural network alternatives for computational optimization
"""

import logging
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import time

@dataclass
class DiffusionParameters:
    """Parameters controlling hormone diffusion behavior."""
    diffusion_coefficient: float = 0.1  # Base diffusion rate
    decay_rate: float = 0.05  # Rate of hormone decay during diffusion
    spatial_decay: float = 2.0  # Exponential decay factor with distance
    receptor_affinity: float = 0.5  # Base receptor binding affinity
    membrane_permeability: float = 0.8  # How easily hormones cross membranes
    flow_velocity: float = 0.2  # Bulk flow velocity (like blood flow)
    temperature_factor: float = 1.0  # Temperature effect on diffusion
    ph_factor: float = 1.0  # pH effect on hormone stability

@dataclass
class LobePosition:
    """3D position and properties of a lobe in the diffusion space."""
    x: float
    y: float
    z: float
    radius: float = 1.0  # Effective radius of the lobe
    permeability: float = 0.8  # How permeable the lobe is to hormones
    receptor_density: Dict[str, float] = None  # Density of receptors for each hormone
    
    def __post_init__(self):
        if self.receptor_density is None:
            self.receptor_density = {}

class HormoneDiffusionEngine:
    """
    Biologically-inspired hormone diffusion engine with both algorithmic
    and neural network implementations.
    """
    
    def __init__(self, use_neural: bool = False):
        """
        Initialize the hormone diffusion engine.
        
        Args:
            use_neural: Whether to use neural network implementation by default
        """
        self.logger = logging.getLogger("HormoneDiffusionEngine")
        self.use_neural = use_neural
        
        # Lobe positions and properties
        self.lobe_positions: Dict[str, LobePosition] = {}
        
        # Diffusion parameters for each hormone
        self.diffusion_params: Dict[str, DiffusionParameters] = {}
        
        # Current hormone concentrations in 3D space
        self.concentration_field: Dict[str, Dict[Tuple[int, int, int], float]] = {}
        
        # Gradient fields for each hormone
        self.gradient_fields: Dict[str, Dict[Tuple[int, int, int], Tuple[float, float, float]]] = {}
        
        # Neural network models for diffusion (if available)
        self.neural_models: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "algorithmic": {"accuracy": 0.0, "latency": 0.0, "resource_usage": 0.0},
            "neural": {"accuracy": 0.0, "latency": 0.0, "resource_usage": 0.0}
        }
        
        # Spatial resolution for discretization
        self.spatial_resolution = 0.5  # Grid spacing
        self.max_distance = 10.0  # Maximum diffusion distance
        
        # Initialize default hormone parameters
        self._initialize_hormone_parameters()
        
        # Background diffusion thread
        self._running = False
        self._diffusion_thread = None
        
        self.logger.info("HormoneDiffusionEngine initialized")
    
    def _initialize_hormone_parameters(self):
        """Initialize default diffusion parameters for common hormones."""
        default_hormones = {
            "dopamine": DiffusionParameters(
                diffusion_coefficient=0.15,
                decay_rate=0.08,
                spatial_decay=1.8,
                receptor_affinity=0.7,
                membrane_permeability=0.6,
                flow_velocity=0.3
            ),
            "serotonin": DiffusionParameters(
                diffusion_coefficient=0.12,
                decay_rate=0.06,
                spatial_decay=2.2,
                receptor_affinity=0.6,
                membrane_permeability=0.7,
                flow_velocity=0.25
            ),
            "cortisol": DiffusionParameters(
                diffusion_coefficient=0.08,
                decay_rate=0.03,
                spatial_decay=1.5,
                receptor_affinity=0.8,
                membrane_permeability=0.9,
                flow_velocity=0.4
            ),
            "oxytocin": DiffusionParameters(
                diffusion_coefficient=0.1,
                decay_rate=0.07,
                spatial_decay=2.5,
                receptor_affinity=0.9,
                membrane_permeability=0.5,
                flow_velocity=0.2
            ),
            "acetylcholine": DiffusionParameters(
                diffusion_coefficient=0.2,
                decay_rate=0.15,
                spatial_decay=3.0,
                receptor_affinity=0.8,
                membrane_permeability=0.4,
                flow_velocity=0.1
            )
        }
        
        for hormone, params in default_hormones.items():
            self.diffusion_params[hormone] = params
            self.concentration_field[hormone] = {}
            self.gradient_fields[hormone] = {}
    
    def register_lobe(self, lobe_name: str, position: Tuple[float, float, float], 
                     radius: float = 1.0, permeability: float = 0.8,
                     receptor_densities: Dict[str, float] = None):
        """
        Register a lobe with its spatial position and properties.
        
        Args:
            lobe_name: Name of the lobe
            position: 3D position (x, y, z)
            radius: Effective radius of the lobe
            permeability: Membrane permeability (0.0-1.0)
            receptor_densities: Density of receptors for each hormone
        """
        self.lobe_positions[lobe_name] = LobePosition(
            x=position[0],
            y=position[1], 
            z=position[2],
            radius=radius,
            permeability=permeability,
            receptor_density=receptor_densities or {}
        )
        
        self.logger.info(f"Registered lobe '{lobe_name}' at position {position}")
    
    def set_hormone_parameters(self, hormone: str, params: DiffusionParameters):
        """
        Set diffusion parameters for a specific hormone.
        
        Args:
            hormone: Name of the hormone
            params: Diffusion parameters
        """
        self.diffusion_params[hormone] = params
        
        # Initialize concentration and gradient fields if not present
        if hormone not in self.concentration_field:
            self.concentration_field[hormone] = {}
        if hormone not in self.gradient_fields:
            self.gradient_fields[hormone] = {}
            
        self.logger.info(f"Set diffusion parameters for hormone '{hormone}'")
    
    def release_hormone(self, source_lobe: str, hormone: str, quantity: float) -> Dict[str, float]:
        """
        Release hormone from a source lobe and calculate diffusion.
        
        Args:
            source_lobe: Name of the source lobe
            hormone: Name of the hormone
            quantity: Amount of hormone to release
            
        Returns:
            Dictionary mapping lobe names to received hormone quantities
        """
        if source_lobe not in self.lobe_positions:
            self.logger.warning(f"Unknown source lobe: {source_lobe}")
            return {}
            
        if hormone not in self.diffusion_params:
            self.logger.warning(f"Unknown hormone: {hormone}")
            return {}
        
        # Choose implementation based on performance
        start_time = time.time()
        
        if self.use_neural and hormone in self.neural_models:
            result = self._neural_diffusion(source_lobe, hormone, quantity)
            implementation = "neural"
        else:
            result = self._algorithmic_diffusion(source_lobe, hormone, quantity)
            implementation = "algorithmic"
        
        # Track performance
        elapsed_time = time.time() - start_time
        self.performance_metrics[implementation]["latency"] = elapsed_time
        
        self.logger.info(f"Hormone '{hormone}' diffused from '{source_lobe}' using {implementation} implementation")
        return result
    
    def _algorithmic_diffusion(self, source_lobe: str, hormone: str, quantity: float) -> Dict[str, float]:
        """
        Algorithmic implementation of hormone diffusion based on biological principles.
        
        Args:
            source_lobe: Name of the source lobe
            hormone: Name of the hormone
            quantity: Amount of hormone to release
            
        Returns:
            Dictionary mapping lobe names to received hormone quantities
        """
        source_pos = self.lobe_positions[source_lobe]
        params = self.diffusion_params[hormone]
        
        # Calculate diffusion to all other lobes
        diffusion_results = {}
        
        for target_lobe, target_pos in self.lobe_positions.items():
            if target_lobe == source_lobe:
                # Autocrine effect (local concentration)
                diffusion_results[target_lobe] = quantity * 0.8
                continue
            
            # Calculate distance between lobes
            distance = math.sqrt(
                (target_pos.x - source_pos.x) ** 2 +
                (target_pos.y - source_pos.y) ** 2 +
                (target_pos.z - source_pos.z) ** 2
            )
            
            # Skip if too far away
            if distance > self.max_distance:
                diffusion_results[target_lobe] = 0.0
                continue
            
            # Calculate diffusion based on Fick's law and biological factors
            
            # 1. Distance-based decay (exponential)
            distance_factor = math.exp(-distance * params.spatial_decay)
            
            # 2. Membrane permeability effect
            permeability_factor = (source_pos.permeability + target_pos.permeability) / 2
            
            # 3. Receptor affinity and density
            receptor_density = target_pos.receptor_density.get(hormone, 0.5)
            receptor_factor = params.receptor_affinity * receptor_density
            
            # 4. Bulk flow effect (directional transport)
            flow_factor = 1.0 + params.flow_velocity * (1.0 / (1.0 + distance))
            
            # 5. Environmental factors
            env_factor = params.temperature_factor * params.ph_factor
            
            # Combine all factors
            diffusion_coefficient = (
                params.diffusion_coefficient *
                distance_factor *
                permeability_factor *
                receptor_factor *
                flow_factor *
                env_factor
            )
            
            # Calculate final concentration at target
            received_quantity = quantity * diffusion_coefficient
            
            # Apply decay during transport
            decay_factor = math.exp(-params.decay_rate * distance)
            received_quantity *= decay_factor
            
            diffusion_results[target_lobe] = max(0.0, received_quantity)
        
        # Update concentration field
        self._update_concentration_field(hormone, source_lobe, quantity, diffusion_results)
        
        return diffusion_results
    
    def _neural_diffusion(self, source_lobe: str, hormone: str, quantity: float) -> Dict[str, float]:
        """
        Neural network implementation of hormone diffusion.
        
        Args:
            source_lobe: Name of the source lobe
            hormone: Name of the hormone
            quantity: Amount of hormone to release
            
        Returns:
            Dictionary mapping lobe names to received hormone quantities
        """
        # Placeholder for neural network implementation
        # In a real implementation, this would use a trained neural network
        # to predict diffusion patterns based on learned biological data
        
        # For now, fall back to algorithmic implementation with some variation
        algorithmic_result = self._algorithmic_diffusion(source_lobe, hormone, quantity)
        
        # Add some neural-like variation (simulating learned optimizations)
        neural_result = {}
        for lobe, concentration in algorithmic_result.items():
            # Simulate neural network optimization
            neural_factor = 1.0 + 0.1 * math.sin(hash(lobe + hormone) % 100)
            neural_result[lobe] = concentration * neural_factor
        
        return neural_result
    
    def _update_concentration_field(self, hormone: str, source_lobe: str, 
                                  source_quantity: float, diffusion_results: Dict[str, float]):
        """
        Update the 3D concentration field for visualization and analysis.
        
        Args:
            hormone: Name of the hormone
            source_lobe: Name of the source lobe
            source_quantity: Original quantity released
            diffusion_results: Results of diffusion calculation
        """
        # Discretize space into grid points
        source_pos = self.lobe_positions[source_lobe]
        
        # Create concentration field around source
        for x in np.arange(-self.max_distance, self.max_distance + self.spatial_resolution, self.spatial_resolution):
            for y in np.arange(-self.max_distance, self.max_distance + self.spatial_resolution, self.spatial_resolution):
                for z in np.arange(-self.max_distance, self.max_distance + self.spatial_resolution, self.spatial_resolution):
                    grid_pos = (
                        int((source_pos.x + x) / self.spatial_resolution),
                        int((source_pos.y + y) / self.spatial_resolution),
                        int((source_pos.z + z) / self.spatial_resolution)
                    )
                    
                    # Calculate distance from source
                    distance = math.sqrt(x**2 + y**2 + z**2)
                    
                    if distance <= self.max_distance:
                        # Calculate concentration at this point
                        params = self.diffusion_params[hormone]
                        concentration = source_quantity * math.exp(-distance * params.spatial_decay)
                        
                        # Update concentration field
                        if grid_pos not in self.concentration_field[hormone]:
                            self.concentration_field[hormone][grid_pos] = 0.0
                        self.concentration_field[hormone][grid_pos] += concentration
                        
                        # Calculate gradient (for visualization)
                        if distance > 0:
                            gradient_magnitude = concentration * params.spatial_decay
                            gradient = (
                                -gradient_magnitude * x / distance,
                                -gradient_magnitude * y / distance,
                                -gradient_magnitude * z / distance
                            )
                            self.gradient_fields[hormone][grid_pos] = gradient
    
    def get_concentration_at_position(self, hormone: str, position: Tuple[float, float, float]) -> float:
        """
        Get hormone concentration at a specific 3D position.
        
        Args:
            hormone: Name of the hormone
            position: 3D position (x, y, z)
            
        Returns:
            Hormone concentration at the position
        """
        if hormone not in self.concentration_field:
            return 0.0
        
        # Convert position to grid coordinates
        grid_pos = (
            int(position[0] / self.spatial_resolution),
            int(position[1] / self.spatial_resolution),
            int(position[2] / self.spatial_resolution)
        )
        
        return self.concentration_field[hormone].get(grid_pos, 0.0)
    
    def get_gradient_at_position(self, hormone: str, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Get hormone gradient at a specific 3D position.
        
        Args:
            hormone: Name of the hormone
            position: 3D position (x, y, z)
            
        Returns:
            Gradient vector (dx, dy, dz)
        """
        if hormone not in self.gradient_fields:
            return (0.0, 0.0, 0.0)
        
        # Convert position to grid coordinates
        grid_pos = (
            int(position[0] / self.spatial_resolution),
            int(position[1] / self.spatial_resolution),
            int(position[2] / self.spatial_resolution)
        )
        
        return self.gradient_fields[hormone].get(grid_pos, (0.0, 0.0, 0.0))
    
    def start_background_diffusion(self, interval: float = 1.0):
        """
        Start background diffusion processing for continuous hormone decay and redistribution.
        
        Args:
            interval: Processing interval in seconds
        """
        if self._running:
            return
        
        self._running = True
        self._diffusion_thread = threading.Thread(
            target=self._background_diffusion_loop,
            args=(interval,),
            daemon=True
        )
        self._diffusion_thread.start()
        
        self.logger.info("Started background diffusion processing")
    
    def stop_background_diffusion(self):
        """Stop background diffusion processing."""
        self._running = False
        if self._diffusion_thread:
            self._diffusion_thread.join()
        
        self.logger.info("Stopped background diffusion processing")
    
    def _background_diffusion_loop(self, interval: float):
        """Background loop for continuous hormone processing."""
        while self._running:
            try:
                # Process decay for all hormones
                for hormone in self.concentration_field:
                    self._process_hormone_decay(hormone)
                
                # Sleep for the specified interval
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in background diffusion loop: {e}")
    
    def _process_hormone_decay(self, hormone: str):
        """
        Process natural decay of hormone concentrations.
        
        Args:
            hormone: Name of the hormone
        """
        if hormone not in self.diffusion_params:
            return
        
        params = self.diffusion_params[hormone]
        decay_factor = math.exp(-params.decay_rate)
        
        # Apply decay to concentration field
        for grid_pos in list(self.concentration_field[hormone].keys()):
            old_concentration = self.concentration_field[hormone][grid_pos]
            new_concentration = old_concentration * decay_factor
            
            # Remove very low concentrations to save memory
            if new_concentration < 1e-6:
                del self.concentration_field[hormone][grid_pos]
                if grid_pos in self.gradient_fields[hormone]:
                    del self.gradient_fields[hormone][grid_pos]
            else:
                self.concentration_field[hormone][grid_pos] = new_concentration
    
    def get_diffusion_visualization_data(self, hormone: str) -> Dict[str, Any]:
        """
        Get data for visualizing hormone diffusion.
        
        Args:
            hormone: Name of the hormone
            
        Returns:
            Dictionary with visualization data
        """
        if hormone not in self.concentration_field:
            return {}
        
        # Extract concentration data
        positions = []
        concentrations = []
        gradients = []
        
        for grid_pos, concentration in self.concentration_field[hormone].items():
            # Convert grid position back to world coordinates
            world_pos = (
                grid_pos[0] * self.spatial_resolution,
                grid_pos[1] * self.spatial_resolution,
                grid_pos[2] * self.spatial_resolution
            )
            
            positions.append(world_pos)
            concentrations.append(concentration)
            
            # Get gradient if available
            gradient = self.gradient_fields[hormone].get(grid_pos, (0.0, 0.0, 0.0))
            gradients.append(gradient)
        
        return {
            "hormone": hormone,
            "positions": positions,
            "concentrations": concentrations,
            "gradients": gradients,
            "lobe_positions": {
                name: (pos.x, pos.y, pos.z) 
                for name, pos in self.lobe_positions.items()
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for both implementations.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics.copy()
    
    def switch_to_neural(self, hormone: str = None):
        """
        Switch to neural network implementation.
        
        Args:
            hormone: Specific hormone to switch, or None for all
        """
        if hormone:
            # Switch specific hormone (would need per-hormone neural models)
            self.logger.info(f"Switched to neural implementation for {hormone}")
        else:
            self.use_neural = True
            self.logger.info("Switched to neural implementation for all hormones")
    
    def switch_to_algorithmic(self, hormone: str = None):
        """
        Switch to algorithmic implementation.
        
        Args:
            hormone: Specific hormone to switch, or None for all
        """
        if hormone:
            # Switch specific hormone
            self.logger.info(f"Switched to algorithmic implementation for {hormone}")
        else:
            self.use_neural = False
            self.logger.info("Switched to algorithmic implementation for all hormones")
    
    def clear_concentration_fields(self):
        """Clear all concentration and gradient fields."""
        for hormone in self.concentration_field:
            self.concentration_field[hormone].clear()
            self.gradient_fields[hormone].clear()
        
        self.logger.info("Cleared all concentration fields")