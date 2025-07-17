"""
Unit tests for receptor sensitivity adaptation functionality.

This module tests the receptor sensitivity adaptation model including
both algorithmic and neural implementations.
"""

import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any

# Import the modules to test
from src.mcp.neural_network_models.receptor_sensitivity_model import (
    ReceptorSensitivityNetwork,
    ReceptorSensitivityAdapter,
    ReceptorPerformanceData
)


class TestReceptorSensitivityNetwork:
    """Test cases for ReceptorSensitivityNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = ReceptorSensitivityNetwork()
        
        assert network.input_dim == 10
        assert network.hidden_dim == 24
        assert network.output_dim == 1
        assert not network.is_trained
        assert network.training_iterations == 0
        assert len(network.training_loss_history) == 0
    
    def test_prediction_without_training(self):
        """Test prediction with untrained network."""
        network = ReceptorSensitivityNetwork()
        
        # Create dummy features
        features = [0.5, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.5]
        
        # Make prediction
        prediction = network.predict(features)
        
        # Should return a value in valid range
        assert 0.1 <= prediction <= 2.0
        assert isinstance(prediction, float)
    
    def test_training_with_data(self):
        """Test network training with sample data."""
        network = ReceptorSensitivityNetwork()
        
        # Create training data
        training_data = []
        for i in range(50):
            features = [np.random.random() for _ in range(10)]
            target = 0.5 + 0.5 * np.random.random()  # Target between 0.5 and 1.0
            training_data.append((features, target))
        
        # Train network
        result = network.train(training_data, epochs=10, batch_size=16)
        
        # Check training results
        assert result["success"] is True
        assert result["training_time"] > 0
        assert result["epochs"] == 10
        assert network.is_trained is True
        assert network.training_iterations == 10
        assert len(network.training_loss_history) == 10
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        network = ReceptorSensitivityNetwork()
        
        # Train with some data
        training_data = [(np.random.random(10).tolist(), 0.8) for _ in range(20)]
        network.train(training_data, epochs=5)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.npz")
            
            # Save model
            save_success = network.save(model_path)
            assert save_success is True
            assert os.path.exists(model_path)
            
            # Create new network and load model
            new_network = ReceptorSensitivityNetwork()
            load_success = new_network.load(model_path)
            assert load_success is True
            assert new_network.is_trained is True
            assert new_network.training_iterations == 5


class TestReceptorSensitivityAdapter:
    """Test cases for ReceptorSensitivityAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = ReceptorSensitivityAdapter(model_dir=temp_dir)
            
            assert adapter.model_dir == temp_dir
            assert adapter.active_implementation == "algorithmic"
            assert len(adapter.networks) == 0
            assert len(adapter.performance_history) == 0
    
    def test_receptor_key_generation(self):
        """Test receptor key generation."""
        adapter = ReceptorSensitivityAdapter()
        
        # Test without receptor subtype
        key1 = adapter._get_receptor_key("memory", "dopamine")
        assert key1 == "memory:dopamine"
        
        # Test with receptor subtype
        key2 = adapter._get_receptor_key("memory", "dopamine", "D1")
        assert key2 == "memory:dopamine:D1"
    
    def test_algorithmic_adaptation_basic(self):
        """Test basic algorithmic adaptation."""
        adapter = ReceptorSensitivityAdapter()
        
        # Test adaptation with good performance
        new_sensitivity = adapter.adapt_sensitivity(
            "memory", "dopamine", 0.8, 0.5
        )
        
        # Should increase sensitivity for good performance
        assert new_sensitivity > 0.5
        assert 0.1 <= new_sensitivity <= 2.0
        
        # Test adaptation with poor performance
        new_sensitivity = adapter.adapt_sensitivity(
            "memory", "dopamine", 0.2, 0.5
        )
        
        # Should decrease sensitivity for poor performance
        assert new_sensitivity < 0.5
        assert 0.1 <= new_sensitivity <= 2.0
    
    def test_algorithmic_adaptation_with_history(self):
        """Test algorithmic adaptation with performance history."""
        adapter = ReceptorSensitivityAdapter()
        
        # Build up some history
        sensitivity = 0.5
        for i in range(10):
            performance = 0.6 + 0.1 * i / 10  # Gradually improving performance
            sensitivity = adapter.adapt_sensitivity(
                "memory", "dopamine", performance, sensitivity
            )
        
        # Should have adapted based on history
        assert 0.1 <= sensitivity <= 2.0
        
        # Check that history was recorded
        history = adapter.get_receptor_performance_history("memory", "dopamine")
        assert len(history) == 10
    
    def test_performance_recording(self):
        """Test performance data recording."""
        adapter = ReceptorSensitivityAdapter()
        
        # Record some performance data
        adapter.adapt_sensitivity(
            "memory", "dopamine", 0.7, 0.5, 
            context={"task_priority": 0.8}
        )
        
        # Check that data was recorded
        history = adapter.get_receptor_performance_history("memory", "dopamine")
        assert len(history) == 1
        
        data_point = history[0]
        assert data_point["lobe_name"] == "memory"
        assert data_point["hormone_name"] == "dopamine"
        assert data_point["performance_score"] == 0.7
        assert data_point["current_sensitivity"] == 0.5
        assert data_point["context"]["task_priority"] == 0.8
    
    def test_feature_extraction(self):
        """Test feature extraction for neural network."""
        adapter = ReceptorSensitivityAdapter()
        
        features = adapter._extract_features(
            "memory", "dopamine", 0.7, 0.5, "D1",
            {"hormone_level": 0.6, "task_priority": 0.8}
        )
        
        # Should have 10 features
        assert len(features) == 10
        
        # All features should be numeric
        for feature in features:
            assert isinstance(feature, float)
            assert 0.0 <= feature <= 1.0
    
    def test_implementation_switching(self):
        """Test switching between implementations."""
        adapter = ReceptorSensitivityAdapter()
        
        # Should start with algorithmic
        assert adapter.active_implementation == "algorithmic"
        
        # Switch to neural (may fail if PyTorch not available)
        success = adapter.set_active_implementation("neural")
        if success:
            assert adapter.active_implementation == "neural"
        
        # Switch back to algorithmic
        success = adapter.set_active_implementation("algorithmic")
        assert success is True
        assert adapter.active_implementation == "algorithmic"
        
        # Test invalid implementation
        success = adapter.set_active_implementation("invalid")
        assert success is False
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        adapter = ReceptorSensitivityAdapter()
        
        # Perform some adaptations
        for i in range(5):
            adapter.adapt_sensitivity("memory", "dopamine", 0.7, 0.5)
        
        # Check performance metrics
        metrics = adapter.get_performance_metrics()
        assert "algorithmic" in metrics
        assert metrics["algorithmic"]["calls"] == 5
        assert metrics["algorithmic"]["latency"] > 0
    
    def test_neural_model_training(self):
        """Test neural model training with collected data."""
        adapter = ReceptorSensitivityAdapter()
        
        # Collect enough data for training
        for i in range(60):
            performance = 0.5 + 0.3 * np.random.random()
            sensitivity = 0.3 + 0.4 * np.random.random()
            adapter.adapt_sensitivity("memory", "dopamine", performance, sensitivity)
        
        # Train neural models
        results = adapter.train_neural_models(epochs=5, batch_size=16)
        
        # Check training results
        if "memory:dopamine" in results:
            result = results["memory:dopamine"]
            assert result["success"] is True
            assert result["training_time"] > 0
    
    def test_context_handling(self):
        """Test handling of context information."""
        adapter = ReceptorSensitivityAdapter()
        
        # Test with context
        context = {
            "hormone_level": 0.7,
            "brain_state_arousal": 0.8,
            "task_priority": 0.9
        }
        
        sensitivity = adapter.adapt_sensitivity(
            "memory", "dopamine", 0.7, 0.5, context=context
        )
        
        assert 0.1 <= sensitivity <= 2.0
        
        # Check that context was recorded
        history = adapter.get_receptor_performance_history("memory", "dopamine")
        assert len(history) == 1
        assert history[0]["context"]["hormone_level"] == 0.7
    
    def test_sensitivity_bounds(self):
        """Test that sensitivity stays within valid bounds."""
        adapter = ReceptorSensitivityAdapter()
        
        # Test with extreme values
        sensitivity = adapter.adapt_sensitivity("memory", "dopamine", 1.0, 1.9)
        assert 0.1 <= sensitivity <= 2.0
        
        sensitivity = adapter.adapt_sensitivity("memory", "dopamine", 0.0, 0.2)
        assert 0.1 <= sensitivity <= 2.0
    
    def test_multiple_receptors(self):
        """Test handling multiple receptors simultaneously."""
        adapter = ReceptorSensitivityAdapter()
        
        # Test different hormone-lobe combinations
        combinations = [
            ("memory", "dopamine"),
            ("memory", "serotonin"),
            ("task_management", "dopamine"),
            ("decision_making", "oxytocin")
        ]
        
        for lobe, hormone in combinations:
            sensitivity = adapter.adapt_sensitivity(lobe, hormone, 0.7, 0.5)
            assert 0.1 <= sensitivity <= 2.0
        
        # Check that separate histories were maintained
        for lobe, hormone in combinations:
            history = adapter.get_receptor_performance_history(lobe, hormone)
            assert len(history) == 1


class TestReceptorPerformanceData:
    """Test cases for ReceptorPerformanceData."""
    
    def test_data_structure(self):
        """Test ReceptorPerformanceData structure."""
        data = ReceptorPerformanceData(
            lobe_name="memory",
            hormone_name="dopamine",
            receptor_subtype="D1",
            current_sensitivity=0.7,
            performance_score=0.8,
            context={"task_priority": 0.9},
            timestamp=datetime.now()
        )
        
        assert data.lobe_name == "memory"
        assert data.hormone_name == "dopamine"
        assert data.receptor_subtype == "D1"
        assert data.current_sensitivity == 0.7
        assert data.performance_score == 0.8
        assert data.context["task_priority"] == 0.9
        assert isinstance(data.timestamp, datetime)


def test_integration_with_hormone_system():
    """Test integration with hormone system components."""
    adapter = ReceptorSensitivityAdapter()
    
    # Simulate hormone system integration
    hormone_levels = {
        "dopamine": 0.7,
        "serotonin": 0.6,
        "cortisol": 0.3
    }
    
    brain_state = {
        "arousal": 0.8,
        "focus": 0.7,
        "stress": 0.3
    }
    
    # Test adaptation with hormone system context
    for hormone, level in hormone_levels.items():
        context = {
            "hormone_level": level,
            "brain_state_arousal": brain_state["arousal"],
            "brain_state_focus": brain_state["focus"]
        }
        
        sensitivity = adapter.adapt_sensitivity(
            "memory", hormone, 0.7, 0.5, context=context
        )
        
        assert 0.1 <= sensitivity <= 2.0


if __name__ == "__main__":
    # Run basic tests
    print("Running receptor sensitivity adaptation tests...")
    
    # Test network
    print("Testing ReceptorSensitivityNetwork...")
    network_test = TestReceptorSensitivityNetwork()
    network_test.test_initialization()
    network_test.test_prediction_without_training()
    network_test.test_training_with_data()
    network_test.test_save_and_load()
    print("✓ ReceptorSensitivityNetwork tests passed")
    
    # Test adapter
    print("Testing ReceptorSensitivityAdapter...")
    adapter_test = TestReceptorSensitivityAdapter()
    adapter_test.test_initialization()
    adapter_test.test_receptor_key_generation()
    adapter_test.test_algorithmic_adaptation_basic()
    adapter_test.test_algorithmic_adaptation_with_history()
    adapter_test.test_performance_recording()
    adapter_test.test_feature_extraction()
    adapter_test.test_implementation_switching()
    adapter_test.test_performance_metrics_tracking()
    adapter_test.test_neural_model_training()
    adapter_test.test_context_handling()
    adapter_test.test_sensitivity_bounds()
    adapter_test.test_multiple_receptors()
    print("✓ ReceptorSensitivityAdapter tests passed")
    
    # Test data structure
    print("Testing ReceptorPerformanceData...")
    data_test = TestReceptorPerformanceData()
    data_test.test_data_structure()
    print("✓ ReceptorPerformanceData tests passed")
    
    # Test integration
    print("Testing integration...")
    test_integration_with_hormone_system()
    print("✓ Integration tests passed")
    
    print("\nAll receptor sensitivity adaptation tests passed! ✓")