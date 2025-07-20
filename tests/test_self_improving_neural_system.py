"""
Tests for Self-Improving Neural Network System

This module contains comprehensive tests for the self-improving neural network
system, ensuring it can properly optimize, improve, and pretrain neural networks
through recursive self-improvement.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mcp.neural_network_models.self_improving_neural_system import (
    SelfImprovingNeuralSystem,
    ModelImprovementTask,
    SelfImprovementMetrics
)
from mcp.neural_network_models.self_improvement_integration import (
    SelfImprovementIntegration,
    SelfImprovementConfig
)
from mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
from mcp.hormone_system_controller import HormoneSystemController
from mcp.brain_state_aggregator import BrainStateAggregator
from mcp.genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem


class TestSelfImprovingNeuralSystem:
    """Test cases for SelfImprovingNeuralSystem"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing"""
        hormone_integration = Mock(spec=HormoneNeuralIntegration)
        genetic_system = Mock(spec=IntegratedGeneticTriggerSystem)
        hormone_system = Mock(spec=HormoneSystemController)
        brain_state = Mock(spec=BrainStateAggregator)
        
        # Setup mock hormone system
        hormone_system.get_hormone_levels.return_value = {
            'dopamine': 0.6,
            'serotonin': 0.5,
            'cortisol': 0.3,
            'adrenaline': 0.2
        }
        
        # Setup mock brain state
        brain_state.get_system_state.return_value = {
            'memory_usage': 0.4,
            'system_load': 0.3,
            'error_rate': 0.01
        }
        
        return {
            'hormone_integration': hormone_integration,
            'genetic_system': genetic_system,
            'hormone_system': hormone_system,
            'brain_state': brain_state
        }
    
    @pytest.fixture
    def self_improving_system(self, mock_components, temp_model_dir):
        """Create SelfImprovingNeuralSystem instance for testing"""
        system = SelfImprovingNeuralSystem(
            hormone_integration=mock_components['hormone_integration'],
            genetic_system=mock_components['genetic_system'],
            hormone_system=mock_components['hormone_system'],
            brain_state=mock_components['brain_state']
        )
        
        # Update config to use temp directory
        system.config['model_save_path'] = temp_model_dir
        system.config['improvement_interval'] = 1.0  # 1 second for testing
        
        return system
    
    @pytest.mark.asyncio
    async def test_initialization(self, self_improving_system):
        """Test system initialization"""
        assert self_improving_system.iteration == 0
        assert self_improving_system.total_improvements == 0
        assert self_improving_system.best_improvement == 0.0
        assert not self_improving_system.improvement_active
    
    @pytest.mark.asyncio
    async def test_start_stop_improvement_loop(self, self_improving_system):
        """Test starting and stopping the improvement loop"""
        # Start improvement loop
        await self_improving_system.start_self_improvement_loop()
        assert self_improving_system.improvement_active
        
        # Wait a bit for loop to start
        await asyncio.sleep(0.1)
        
        # Stop improvement loop
        await self_improving_system.stop_self_improvement_loop()
        assert not self_improving_system.improvement_active
    
    @pytest.mark.asyncio
    async def test_generate_improvement_tasks(self, self_improving_system, mock_components):
        """Test generation of improvement tasks"""
        # Mock hormone integration to return some models
        mock_components['hormone_integration'].neural_models = {
            'dopamine': Mock(),
            'serotonin': Mock(),
            'cortisol': Mock()
        }
        
        # Mock performance tracker
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.7
        mock_metrics.latency = 50.0
        
        self_improving_system.performance_tracker.get_latest_metrics = Mock(return_value=mock_metrics)
        
        # Generate tasks
        tasks = await self_improving_system._generate_improvement_tasks()
        
        # Should generate tasks for hormone models
        assert len(tasks) > 0
        assert all(isinstance(task, ModelImprovementTask) for task in tasks)
        assert all('hormone_' in task.model_id for task in tasks)
    
    @pytest.mark.asyncio
    async def test_determine_improvement_type(self, self_improving_system):
        """Test improvement type determination"""
        # Test low accuracy -> training
        performance = {'current_accuracy': 0.6, 'current_latency': 100}
        improvement_type = self_improving_system._determine_improvement_type(performance)
        assert improvement_type == 'training'
        
        # Test high latency -> optimization
        performance = {'current_accuracy': 0.8, 'current_latency': 300}
        improvement_type = self_improving_system._determine_improvement_type(performance)
        assert improvement_type == 'optimization'
        
        # Test medium accuracy -> architecture
        performance = {'current_accuracy': 0.75, 'current_latency': 150}
        improvement_type = self_improving_system._determine_improvement_type(performance)
        assert improvement_type == 'architecture'
        
        # Test high accuracy -> distillation
        performance = {'current_accuracy': 0.92, 'current_latency': 80}
        improvement_type = self_improving_system._determine_improvement_type(performance)
        assert improvement_type == 'distillation'
    
    @pytest.mark.asyncio
    async def test_calculate_task_priority(self, self_improving_system, mock_components):
        """Test task priority calculation"""
        performance = {
            'current_accuracy': 0.7,
            'target_metrics': {'accuracy': 0.9}
        }
        
        # Mock genetic system
        mock_components['genetic_system'].get_active_triggers = AsyncMock(return_value=[
            {'targets': ['neural_optimization']}
        ])
        
        priority = await self_improving_system._calculate_task_priority('test_model', performance)
        
        # Priority should be positive and reasonable
        assert priority > 0
        assert priority <= 10.0
    
    @pytest.mark.asyncio
    async def test_generate_hormone_training_data(self, self_improving_system):
        """Test generation of hormone training data"""
        training_data = await self_improving_system._generate_hormone_training_data('dopamine')
        
        # Should generate training data
        assert len(training_data) > 0
        assert all(isinstance(item, tuple) for item in training_data)
        assert all(len(item) == 2 for item in training_data)  # (features, target)
        
        # Check feature dimensions
        features, target = training_data[0]
        assert len(features) == 10  # 10 context features
        assert isinstance(target, float)
    
    @pytest.mark.asyncio
    async def test_extract_features_from_context(self, self_improving_system):
        """Test feature extraction from context"""
        context = {
            'stress_level': 0.5,
            'error_frequency': 0.1,
            'resource_constraints': 0.3,
            'task_complexity': 0.7,
            'user_interaction_level': 0.4,
            'system_load': 0.6,
            'memory_usage': 0.8,
            'network_activity': 0.2,
            'learning_rate': 0.9,
            'confidence_level': 0.6
        }
        
        features = self_improving_system._extract_features_from_context(context)
        
        assert len(features) == 10
        assert all(isinstance(f, float) for f in features)
        assert features == list(context.values())
    
    @pytest.mark.asyncio
    async def test_get_current_environment(self, self_improving_system, mock_components):
        """Test getting current environment state"""
        environment = await self_improving_system._get_current_environment()
        
        assert isinstance(environment, dict)
        assert 'system_load' in environment
        assert 'memory_usage' in environment
        assert 'error_rate' in environment
        assert 'timestamp' in environment
    
    @pytest.mark.asyncio
    async def test_save_improved_model(self, self_improving_system, temp_model_dir):
        """Test saving improved models"""
        model_id = 'test_model'
        mock_model = Mock()
        
        # Mock model size
        self_improving_system._get_model_size = Mock(return_value=1000)
        
        # Mock evaluation
        self_improving_system._evaluate_model = AsyncMock(return_value={
            'accuracy': 0.85,
            'latency': 45.0
        })
        
        # Mock test data generation
        self_improving_system._generate_test_data = AsyncMock(return_value=[])
        
        await self_improving_system._save_improved_model(model_id, mock_model)
        
        # Check that model files were created
        model_files = list(Path(temp_model_dir).glob(f"{model_id}_*.pkl"))
        metadata_files = list(Path(temp_model_dir).glob(f"{model_id}_*_metadata.json"))
        
        assert len(model_files) > 0
        assert len(metadata_files) > 0
        
        # Check registry update
        assert model_id in self_improving_system.model_registry
        assert model_id in self_improving_system.model_versions
    
    @pytest.mark.asyncio
    async def test_get_model_size(self, self_improving_system):
        """Test model size calculation"""
        # Test with PyTorch model
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=Mock(return_value=100)) for _ in range(3)]
        
        size = self_improving_system._get_model_size(mock_model)
        assert size == 300  # 3 * 100 parameters
    
    @pytest.mark.asyncio
    async def test_record_model_improvement(self, self_improving_system):
        """Test recording model improvements"""
        task = ModelImprovementTask(
            model_id='test_model',
            model_type='hormone',
            improvement_type='training'
        )
        
        result = {
            'performance_gain': 0.15,
            'new_metrics': {'accuracy': 0.85}
        }
        
        execution_time = 5.0
        
        self_improving_system._record_model_improvement(task, result, execution_time)
        
        assert self_improving_system.total_improvements == 1
        assert self_improving_system.best_improvement == 0.15
        assert 'test_model' in self_improving_system.model_performance
    
    @pytest.mark.asyncio
    async def test_get_improvement_status(self, self_improving_system):
        """Test getting improvement status"""
        status = await self_improving_system.get_improvement_status()
        
        assert isinstance(status, dict)
        assert 'iteration' in status
        assert 'total_improvements' in status
        assert 'best_improvement' in status
        assert 'improvement_active' in status
        assert 'models_registered' in status
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, self_improving_system, mock_components):
        """Test getting available models"""
        # Mock hormone integration
        mock_components['hormone_integration'].neural_models = {
            'dopamine': Mock(),
            'serotonin': Mock()
        }
        
        models = await self_improving_system.get_available_models()
        
        assert len(models) == 2
        assert 'hormone_dopamine' in models
        assert 'hormone_serotonin' in models


class TestSelfImprovementIntegration:
    """Test cases for SelfImprovementIntegration"""
    
    @pytest.fixture
    def mock_enhanced_mcp(self):
        """Create mock enhanced MCP"""
        enhanced_mcp = Mock()
        enhanced_mcp.hormone_integration = Mock(spec=HormoneNeuralIntegration)
        enhanced_mcp.genetic_optimizer = Mock(spec=IntegratedGeneticTriggerSystem)
        enhanced_mcp.hormone_system = Mock(spec=HormoneSystemController)
        enhanced_mcp.brain_state = Mock(spec=BrainStateAggregator)
        enhanced_mcp.performance_tracker = Mock()
        
        return enhanced_mcp
    
    @pytest.fixture
    def integration_config(self):
        """Create integration configuration"""
        return SelfImprovementConfig(
            enabled=True,
            auto_start=False,  # Don't auto-start for testing
            improvement_interval=1.0,
            max_concurrent_improvements=2,
            performance_threshold=0.1,
            hormone_driven=True,
            genetic_enhanced=True
        )
    
    @pytest.fixture
    async def self_improvement_integration(self, mock_enhanced_mcp, integration_config):
        """Create SelfImprovementIntegration instance"""
        integration = SelfImprovementIntegration(
            enhanced_mcp=mock_enhanced_mcp,
            config=integration_config
        )
        
        await integration.initialize()
        return integration
    
    @pytest.mark.asyncio
    async def test_initialization(self, self_improvement_integration):
        """Test integration initialization"""
        assert self_improvement_integration.initialized
        assert self_improvement_integration.self_improving_system is not None
        assert not self_improvement_integration.improvement_active
    
    @pytest.mark.asyncio
    async def test_start_stop_self_improvement(self, self_improvement_integration):
        """Test starting and stopping self-improvement"""
        # Start self-improvement
        await self_improvement_integration.start_self_improvement()
        assert self_improvement_integration.improvement_active
        
        # Stop self-improvement
        await self_improvement_integration.stop_self_improvement()
        assert not self_improvement_integration.improvement_active
    
    @pytest.mark.asyncio
    async def test_force_improvement_cycle(self, self_improvement_integration):
        """Test forcing an improvement cycle"""
        # Mock the self-improving system
        self_improvement_integration.self_improving_system.force_improvement_cycle = AsyncMock()
        
        await self_improvement_integration.force_improvement_cycle()
        
        # Check that force_improvement_cycle was called
        self_improvement_integration.self_improving_system.force_improvement_cycle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_custom_improvement_task(self, self_improvement_integration):
        """Test adding custom improvement tasks"""
        # Mock the self-improving system
        self_improvement_integration.self_improving_system.add_improvement_task = AsyncMock()
        
        await self_improvement_integration.add_custom_improvement_task(
            model_id='test_model',
            improvement_type='training',
            priority=1.5
        )
        
        # Check that add_improvement_task was called
        self_improvement_integration.self_improving_system.add_improvement_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_infer_model_type(self, self_improvement_integration):
        """Test model type inference"""
        assert self_improvement_integration._infer_model_type('hormone_dopamine') == 'hormone'
        assert self_improvement_integration._infer_model_type('pattern_recognition') == 'pattern'
        assert self_improvement_integration._infer_model_type('memory_consolidation') == 'memory'
        assert self_improvement_integration._infer_model_type('genetic_trigger') == 'genetic'
        assert self_improvement_integration._infer_model_type('unknown_model') == 'generic'
    
    @pytest.mark.asyncio
    async def test_get_improvement_status(self, self_improvement_integration):
        """Test getting improvement status"""
        # Mock the self-improving system
        mock_status = {
            'iteration': 5,
            'total_improvements': 10,
            'improvement_active': True
        }
        self_improvement_integration.self_improving_system.get_improvement_status = AsyncMock(return_value=mock_status)
        
        # Mock hormone system
        self_improvement_integration.hormone_system.get_hormone_levels.return_value = {
            'dopamine': 0.6,
            'serotonin': 0.5
        }
        
        # Mock brain state
        self_improvement_integration.brain_state.get_system_state.return_value = {
            'memory_usage': 0.4
        }
        
        status = await self_improvement_integration.get_improvement_status()
        
        assert 'integration_runtime' in status
        assert 'hormone_levels' in status
        assert 'brain_state' in status
        assert 'config' in status
    
    @pytest.mark.asyncio
    async def test_optimize_specific_model(self, self_improvement_integration):
        """Test optimizing a specific model"""
        # Mock the self-improving system
        self_improvement_integration.self_improving_system.get_model_performance = AsyncMock(return_value=[0.7, 0.75, 0.8])
        self_improvement_integration.self_improving_system.add_improvement_task = AsyncMock()
        
        await self_improvement_integration.optimize_specific_model('test_model', 'auto')
        
        # Check that add_improvement_task was called
        self_improvement_integration.self_improving_system.add_improvement_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_model_performance(self, self_improvement_integration):
        """Test model performance analysis"""
        # Mock performance history
        self_improvement_integration.self_improving_system.get_model_performance = AsyncMock(return_value=[0.6, 0.7, 0.8])
        
        performance = await self_improvement_integration._analyze_model_performance('test_model')
        
        assert 'accuracy' in performance
        assert 'needs_improvement' in performance
        assert 'trend' in performance
        assert performance['accuracy'] == 0.7  # Average of last 10
        assert performance['trend'] == 'improving'  # 0.8 > 0.6
    
    @pytest.mark.asyncio
    async def test_determine_optimization_type(self, self_improvement_integration):
        """Test optimization type determination"""
        # Test low accuracy -> training
        performance = {'accuracy': 0.5}
        opt_type = self_improvement_integration._determine_optimization_type(performance)
        assert opt_type == 'training'
        
        # Test medium accuracy -> architecture
        performance = {'accuracy': 0.7}
        opt_type = self_improvement_integration._determine_optimization_type(performance)
        assert opt_type == 'architecture'
        
        # Test high accuracy -> optimization
        performance = {'accuracy': 0.85}
        opt_type = self_improvement_integration._determine_optimization_type(performance)
        assert opt_type == 'optimization'
    
    @pytest.mark.asyncio
    async def test_get_system_health_report(self, self_improvement_integration):
        """Test system health report generation"""
        # Mock status and history
        mock_status = {'iteration': 10, 'total_improvements': 5}
        mock_history = [
            {'total_improvement': 0.1, 'best_performance_gain': 0.05},
            {'total_improvement': 0.15, 'best_performance_gain': 0.08}
        ]
        
        self_improvement_integration.self_improving_system.get_improvement_status = AsyncMock(return_value=mock_status)
        self_improvement_integration.self_improving_system.get_improvement_history = AsyncMock(return_value=mock_history)
        
        # Mock hormone system
        self_improvement_integration.hormone_system.get_hormone_levels.return_value = {
            'dopamine': 0.6,
            'serotonin': 0.5,
            'cortisol': 0.3
        }
        
        health_report = await self_improvement_integration.get_system_health_report()
        
        assert 'status' in health_report
        assert 'health_metrics' in health_report
        assert 'hormone_balance' in health_report
        assert 'improvement_status' in health_report
        assert 'recent_activity' in health_report
    
    @pytest.mark.asyncio
    async def test_calculate_health_metrics(self, self_improvement_integration):
        """Test health metrics calculation"""
        status = {'iteration': 10, 'total_improvements': 5}
        history = [
            {'total_improvement': 0.1, 'best_performance_gain': 0.05},
            {'total_improvement': 0.15, 'best_performance_gain': 0.08}
        ]
        
        metrics = self_improvement_integration._calculate_health_metrics(status, history)
        
        assert 'overall_health' in metrics
        assert 'improvement_efficiency' in metrics
        assert 'system_stability' in metrics
        assert 'learning_rate' in metrics
        
        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_hormone_balance(self, self_improvement_integration):
        """Test hormone balance analysis"""
        hormone_levels = {
            'dopamine': 0.6,
            'serotonin': 0.5,
            'cortisol': 0.3,
            'growth_hormone': 0.7
        }
        
        balance = self_improvement_integration._analyze_hormone_balance(hormone_levels)
        
        assert 'overall_balance' in balance
        assert 'stress_level' in balance
        assert 'learning_capacity' in balance
        assert 'stability' in balance
    
    @pytest.mark.asyncio
    async def test_configure_improvement(self, self_improvement_integration):
        """Test improvement configuration"""
        config_updates = {
            'improvement_interval': 120.0,
            'max_concurrent_improvements': 5,
            'performance_threshold': 0.15
        }
        
        await self_improvement_integration.configure_improvement(config_updates)
        
        # Check that config was updated
        assert self_improvement_integration.config.improvement_interval == 120.0
        assert self_improvement_integration.config.max_concurrent_improvements == 5
        assert self_improvement_integration.config.performance_threshold == 0.15
    
    @pytest.mark.asyncio
    async def test_export_improvement_data(self, self_improvement_integration):
        """Test exporting improvement data"""
        # Mock various methods
        mock_status = {'iteration': 5, 'total_improvements': 10}
        mock_history = [{'iteration': 1, 'models_improved': 2}]
        mock_health = {'status': 'healthy'}
        mock_models = ['model1', 'model2']
        
        self_improvement_integration.self_improving_system.get_improvement_status = AsyncMock(return_value=mock_status)
        self_improvement_integration.self_improving_system.get_improvement_history = AsyncMock(return_value=mock_history)
        self_improvement_integration.get_system_health_report = AsyncMock(return_value=mock_health)
        self_improvement_integration.self_improving_system.get_available_models = AsyncMock(return_value=mock_models)
        
        export_data = await self_improvement_integration.export_improvement_data()
        
        assert 'export_timestamp' in export_data
        assert 'status' in export_data
        assert 'history' in export_data
        assert 'health_report' in export_data
        assert 'available_models' in export_data
        assert 'configuration' in export_data
    
    @pytest.mark.asyncio
    async def test_cleanup(self, self_improvement_integration):
        """Test cleanup"""
        # Start improvement to test cleanup
        await self_improvement_integration.start_self_improvement()
        assert self_improvement_integration.improvement_active
        
        # Cleanup
        await self_improvement_integration.cleanup()
        
        assert not self_improvement_integration.improvement_active
        assert not self_improvement_integration.initialized


class TestIntegrationScenarios:
    """Integration test scenarios for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_improvement_cycle(self, temp_model_dir):
        """Test a complete improvement cycle"""
        # Create real components
        hormone_system = HormoneSystemController()
        brain_state = BrainStateAggregator()
        genetic_system = IntegratedGeneticTriggerSystem()
        hormone_integration = HormoneNeuralIntegration(hormone_system_controller=hormone_system)
        
        # Create self-improving system
        system = SelfImprovingNeuralSystem(
            hormone_integration=hormone_integration,
            genetic_system=genetic_system,
            hormone_system=hormone_system,
            brain_state=brain_state
        )
        
        # Update config
        system.config['model_save_path'] = temp_model_dir
        system.config['improvement_interval'] = 1.0
        
        # Run a single improvement cycle
        await system._run_improvement_cycle()
        
        # Check that iteration increased
        assert system.iteration == 1
        
        # Check that improvement history was updated
        assert len(system.improvement_history) > 0
    
    @pytest.mark.asyncio
    async def test_model_improvement_workflow(self, temp_model_dir):
        """Test complete model improvement workflow"""
        # Create components
        hormone_system = HormoneSystemController()
        hormone_integration = HormoneNeuralIntegration(hormone_system_controller=hormone_system)
        
        # Create system
        system = SelfImprovingNeuralSystem(
            hormone_integration=hormone_integration,
            hormone_system=hormone_system
        )
        system.config['model_save_path'] = temp_model_dir
        
        # Create improvement task
        task = ModelImprovementTask(
            model_id='test_model',
            model_type='hormone',
            improvement_type='training',
            priority=1.0
        )
        
        # Mock model and training data
        mock_model = Mock()
        system._get_model = AsyncMock(return_value=mock_model)
        system._generate_training_data = AsyncMock(return_value=[([0.1, 0.2, 0.3], 0.5)])
        system._train_with_improved_strategies = AsyncMock(return_value=mock_model)
        system._evaluate_model = AsyncMock(return_value={'accuracy': 0.85, 'latency': 45.0})
        system._generate_test_data = AsyncMock(return_value=[([0.1, 0.2, 0.3], 0.5)])
        
        # Execute task
        result = await system._execute_improvement_task(task)
        
        # Check result
        assert result is not None
        assert 'performance_gain' in result
        assert 'new_metrics' in result
        assert task.status == 'completed'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 