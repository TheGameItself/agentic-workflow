"""
Unit tests for the Hormone Diffusion Engine.

Tests both algorithmic and neural network implementations of biologically-inspired
hormone diffusion across the brain-inspired lobe system.
"""

import pytest
import math
import time
from src.mcp.hormone_diffusion_engine import (
    HormoneDiffusionEngine, 
    DiffusionParameters, 
    LobePosition
)

class TestHormoneDiffusionEngine:
    """Test suite for HormoneDiffusionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = HormoneDiffusionEngine(use_neural=False)
        
        # Register test lobes
        self.engine.register_lobe("source_lobe", (0.0, 0.0, 0.0), radius=1.0)
        self.engine.register_lobe("target_lobe_1", (2.0, 0.0, 0.0), radius=1.0)
        self.engine.register_lobe("target_lobe_2", (0.0, 3.0, 0.0), radius=1.0)
        self.engine.register_lobe("distant_lobe", (15.0, 15.0, 15.0), radius=1.0)
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine is not None
        assert not self.engine.use_neural
        assert len(self.engine.lobe_positions) == 4
        assert "dopamine" in self.engine.diffusion_params
        assert "serotonin" in self.engine.diffusion_params
    
    def test_lobe_registration(self):
        """Test lobe registration functionality."""
        # Test basic registration
        assert "source_lobe" in self.engine.lobe_positions
        source_pos = self.engine.lobe_positions["source_lobe"]
        assert source_pos.x == 0.0
        assert source_pos.y == 0.0
        assert source_pos.z == 0.0
        assert source_pos.radius == 1.0
        
        # Test registration with receptor densities
        receptor_densities = {"dopamine": 0.8, "serotonin": 0.6}
        self.engine.register_lobe("test_lobe", (1.0, 1.0, 1.0), 
                                 receptor_densities=receptor_densities)
        
        test_pos = self.engine.lobe_positions["test_lobe"]
        assert test_pos.receptor_density["dopamine"] == 0.8
        assert test_pos.receptor_density["serotonin"] == 0.6
    
    def test_hormone_parameter_setting(self):
        """Test setting custom hormone parameters."""
        custom_params = DiffusionParameters(
            diffusion_coefficient=0.2,
            decay_rate=0.1,
            spatial_decay=1.5,
            receptor_affinity=0.9
        )
        
        self.engine.set_hormone_parameters("test_hormone", custom_params)
        
        assert "test_hormone" in self.engine.diffusion_params
        assert self.engine.diffusion_params["test_hormone"].diffusion_coefficient == 0.2
        assert self.engine.diffusion_params["test_hormone"].decay_rate == 0.1
        assert "test_hormone" in self.engine.concentration_field
        assert "test_hormone" in self.engine.gradient_fields
    
    def test_algorithmic_diffusion_basic(self):
        """Test basic algorithmic diffusion functionality."""
        # Release dopamine from source lobe
        result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # Check that all lobes received some hormone
        assert "source_lobe" in result
        assert "target_lobe_1" in result
        assert "target_lobe_2" in result
        assert "distant_lobe" in result
        
        # Source lobe should have highest concentration (autocrine effect)
        assert result["source_lobe"] > result["target_lobe_1"]
        assert result["source_lobe"] > result["target_lobe_2"]
        
        # Closer lobes should receive more hormone than distant ones
        assert result["target_lobe_1"] > result["distant_lobe"]
        assert result["target_lobe_2"] > result["distant_lobe"]
        
        # All concentrations should be non-negative
        for concentration in result.values():
            assert concentration >= 0.0
    
    def test_distance_based_decay(self):
        """Test that hormone concentration decreases with distance."""
        result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # Calculate distances
        source_pos = self.engine.lobe_positions["source_lobe"]
        target1_pos = self.engine.lobe_positions["target_lobe_1"]
        target2_pos = self.engine.lobe_positions["target_lobe_2"]
        distant_pos = self.engine.lobe_positions["distant_lobe"]
        
        dist1 = math.sqrt((target1_pos.x - source_pos.x)**2)  # Distance = 2.0
        dist2 = math.sqrt((target2_pos.y - source_pos.y)**2)  # Distance = 3.0
        dist_distant = math.sqrt(
            (distant_pos.x - source_pos.x)**2 + 
            (distant_pos.y - source_pos.y)**2 + 
            (distant_pos.z - source_pos.z)**2
        )  # Distance ≈ 26.0
        
        # Closer lobe should receive more hormone
        assert result["target_lobe_1"] > result["target_lobe_2"]
        
        # Very distant lobe should receive minimal hormone
        assert result["distant_lobe"] < 0.01
    
    def test_hormone_specific_diffusion(self):
        """Test that different hormones have different diffusion patterns."""
        # Release dopamine and serotonin from the same source
        dopamine_result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        serotonin_result = self.engine.release_hormone("source_lobe", "serotonin", 1.0)
        
        # Results should be different due to different diffusion parameters
        for lobe in dopamine_result:
            if lobe != "source_lobe":  # Skip autocrine effect which is the same
                assert dopamine_result[lobe] != serotonin_result[lobe]
    
    def test_concentration_field_update(self):
        """Test that concentration fields are properly updated."""
        # Initially, concentration field should be empty or have minimal values
        initial_concentration = self.engine.get_concentration_at_position("dopamine", (0.0, 0.0, 0.0))
        
        # Release hormone
        self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # Concentration at source position should increase
        final_concentration = self.engine.get_concentration_at_position("dopamine", (0.0, 0.0, 0.0))
        assert final_concentration > initial_concentration
        
        # Concentration should decrease with distance from source
        nearby_concentration = self.engine.get_concentration_at_position("dopamine", (1.0, 0.0, 0.0))
        distant_concentration = self.engine.get_concentration_at_position("dopamine", (5.0, 0.0, 0.0))
        assert final_concentration > nearby_concentration > distant_concentration
    
    def test_gradient_calculation(self):
        """Test hormone gradient calculation."""
        # Release hormone to create gradients
        self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # Get gradient at a position near the source
        gradient = self.engine.get_gradient_at_position("dopamine", (1.0, 0.0, 0.0))
        
        # Gradient should point toward the source (negative x direction)
        assert len(gradient) == 3
        assert gradient[0] < 0  # Negative x component (pointing toward source)
        
        # Gradient magnitude should be reasonable
        gradient_magnitude = math.sqrt(sum(g**2 for g in gradient))
        assert gradient_magnitude > 0
    
    def test_neural_vs_algorithmic_comparison(self):
        """Test comparison between neural and algorithmic implementations."""
        # Test algorithmic implementation
        self.engine.use_neural = False
        alg_result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        alg_metrics = self.engine.get_performance_metrics()
        
        # Test neural implementation (currently falls back to algorithmic with variation)
        self.engine.use_neural = True
        neural_result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        neural_metrics = self.engine.get_performance_metrics()
        
        # Both should return valid results
        assert len(alg_result) == len(neural_result)
        assert all(conc >= 0 for conc in alg_result.values())
        assert all(conc >= 0 for conc in neural_result.values())
        
        # Performance metrics should be tracked
        assert "algorithmic" in alg_metrics
        assert "neural" in neural_metrics
        assert "latency" in alg_metrics["algorithmic"]
        assert "latency" in neural_metrics["neural"]
    
    def test_background_diffusion_processing(self):
        """Test background diffusion processing for hormone decay."""
        # Release hormone to create initial concentration
        self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        initial_concentration = self.engine.get_concentration_at_position("dopamine", (0.0, 0.0, 0.0))
        
        # Start background processing
        self.engine.start_background_diffusion(interval=0.1)
        
        # Wait for some decay to occur
        time.sleep(0.3)
        
        # Stop background processing
        self.engine.stop_background_diffusion()
        
        # Concentration should have decreased due to decay
        final_concentration = self.engine.get_concentration_at_position("dopamine", (0.0, 0.0, 0.0))
        assert final_concentration < initial_concentration
    
    def test_visualization_data_generation(self):
        """Test generation of visualization data."""
        # Release hormone to create data
        self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # Get visualization data
        viz_data = self.engine.get_diffusion_visualization_data("dopamine")
        
        # Check data structure
        assert "hormone" in viz_data
        assert "positions" in viz_data
        assert "concentrations" in viz_data
        assert "gradients" in viz_data
        assert "lobe_positions" in viz_data
        
        assert viz_data["hormone"] == "dopamine"
        assert len(viz_data["positions"]) == len(viz_data["concentrations"])
        assert len(viz_data["positions"]) == len(viz_data["gradients"])
        
        # Check that lobe positions are included
        assert "source_lobe" in viz_data["lobe_positions"]
        assert len(viz_data["lobe_positions"]["source_lobe"]) == 3  # x, y, z coordinates
    
    def test_implementation_switching(self):
        """Test switching between neural and algorithmic implementations."""
        # Start with algorithmic
        assert not self.engine.use_neural
        
        # Switch to neural
        self.engine.switch_to_neural()
        assert self.engine.use_neural
        
        # Switch back to algorithmic
        self.engine.switch_to_algorithmic()
        assert not self.engine.use_neural
    
    def test_concentration_field_clearing(self):
        """Test clearing of concentration fields."""
        # Release hormone to populate fields
        self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # Verify fields have data
        assert len(self.engine.concentration_field["dopamine"]) > 0
        assert len(self.engine.gradient_fields["dopamine"]) > 0
        
        # Clear fields
        self.engine.clear_concentration_fields()
        
        # Verify fields are empty
        assert len(self.engine.concentration_field["dopamine"]) == 0
        assert len(self.engine.gradient_fields["dopamine"]) == 0
    
    def test_receptor_density_effects(self):
        """Test that receptor density affects hormone reception."""
        # Create lobes with different receptor densities
        self.engine.register_lobe("high_receptor_lobe", (2.0, 2.0, 0.0), 
                                 receptor_densities={"dopamine": 1.0})
        self.engine.register_lobe("low_receptor_lobe", (2.0, -2.0, 0.0), 
                                 receptor_densities={"dopamine": 0.1})
        
        # Release hormone
        result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # High receptor density lobe should receive more hormone
        # (assuming similar distances)
        assert result["high_receptor_lobe"] > result["low_receptor_lobe"]
    
    def test_membrane_permeability_effects(self):
        """Test that membrane permeability affects diffusion."""
        # Create lobes with different permeabilities
        self.engine.register_lobe("permeable_lobe", (2.0, 2.0, 0.0), permeability=1.0)
        self.engine.register_lobe("impermeable_lobe", (2.0, -2.0, 0.0), permeability=0.1)
        
        # Release hormone
        result = self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        # More permeable lobe should receive more hormone
        assert result["permeable_lobe"] > result["impermeable_lobe"]
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test unknown source lobe
        result = self.engine.release_hormone("unknown_lobe", "dopamine", 1.0)
        assert result == {}
        
        # Test unknown hormone
        result = self.engine.release_hormone("source_lobe", "unknown_hormone", 1.0)
        assert result == {}
        
        # Test negative quantity (should be handled gracefully)
        result = self.engine.release_hormone("source_lobe", "dopamine", -1.0)
        assert all(conc >= 0 for conc in result.values())
    
    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Release hormone to generate metrics
        self.engine.release_hormone("source_lobe", "dopamine", 1.0)
        
        metrics = self.engine.get_performance_metrics()
        
        # Check structure
        assert "algorithmic" in metrics
        assert "neural" in metrics
        
        # Check that latency was recorded
        assert "latency" in metrics["algorithmic"]
        assert metrics["algorithmic"]["latency"] >= 0
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.engine, '_running') and self.engine._running:
            self.engine.stop_background_diffusion()


class TestDiffusionParameters:
    """Test suite for DiffusionParameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = DiffusionParameters()
        
        assert params.diffusion_coefficient == 0.1
        assert params.decay_rate == 0.05
        assert params.spatial_decay == 2.0
        assert params.receptor_affinity == 0.5
        assert params.membrane_permeability == 0.8
        assert params.flow_velocity == 0.2
        assert params.temperature_factor == 1.0
        assert params.ph_factor == 1.0
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = DiffusionParameters(
            diffusion_coefficient=0.2,
            decay_rate=0.1,
            spatial_decay=1.5,
            receptor_affinity=0.9
        )
        
        assert params.diffusion_coefficient == 0.2
        assert params.decay_rate == 0.1
        assert params.spatial_decay == 1.5
        assert params.receptor_affinity == 0.9
        # Other parameters should have default values
        assert params.membrane_permeability == 0.8
        assert params.flow_velocity == 0.2


class TestLobePosition:
    """Test suite for LobePosition dataclass."""
    
    def test_default_position(self):
        """Test default position values."""
        pos = LobePosition(x=1.0, y=2.0, z=3.0)
        
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0
        assert pos.radius == 1.0
        assert pos.permeability == 0.8
        assert pos.receptor_density == {}
    
    def test_custom_position(self):
        """Test custom position values."""
        receptor_densities = {"dopamine": 0.8, "serotonin": 0.6}
        pos = LobePosition(
            x=5.0, y=6.0, z=7.0,
            radius=2.0,
            permeability=0.9,
            receptor_density=receptor_densities
        )
        
        assert pos.x == 5.0
        assert pos.y == 6.0
        assert pos.z == 7.0
        assert pos.radius == 2.0
        assert pos.permeability == 0.9
        assert pos.receptor_density == receptor_densities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])