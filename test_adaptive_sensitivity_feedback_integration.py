#!/usr/bin/env python3
"""
Comprehensive tests for adaptive column sensitivity and feedback integration
in the Enhanced Pattern Recognition Engine.

Tests cover:
- Column sensitivity adaptation based on feedback
- Pattern feedback processing and learning integration
- Dynamic sensitivity adjustment for optimal performance
- Cross-lobe sensory data sharing
- Hormone-based feedback modulation
- Cross-modal learning and adaptation
"""

import tempfile
import os
import time
import json
from unittest.mock import Mock, patch

# Import the enhanced pattern recognition engine
import sys
sys.path.append('src')

from mcp.lobes.pattern_recognition_engine_enhanced import (
    EnhancedPatternRecognitionEngine,
    EnhancedNeuralColumn,
    SensoryDataPropagator,
    AdaptiveSensitivityManager
)


class TestEnhancedNeuralColumnAdaptiveSensitivity:
    """Test adaptive sensitivity features of enhanced neural columns."""
    
    def setup_method(self):
        """Setup test environment."""
        self.column = EnhancedNeuralColumn(
            'test_column',
            ['visual', 'pattern', 'test'],
            position=(0, 0, 0)
        )
    
    def test_hormone_based_sensitivity_adaptation(self):
        """Test sensitivity adaptation based on hormone levels."""
        # Test dopamine enhancement
        feedback_data = {
            'performance': 0.8,
            'accuracy': 0.9,
            'response_time': 0.5,
            'user_satisfaction': 0.8,
            'hormone_levels': {
                'dopamine': 0.8,
                'serotonin': 0.6,
                'cortisol': 0.2,
                'norepinephrine': 0.7
            }
        }
        
        initial_sensitivity = self.column.sensitivity
        self.column.adapt_sensitivity_enhanced(feedback_data)
        
        # High dopamine with good performance should increase sensitivity
        assert self.column.sensitivity > initial_sensitivity
        assert len(self.column.feedback_history) == 1
        assert 'hormone_modulation' in self.column.feedback_history[0]
    
    def test_cortisol_stress_response_adaptation(self):
        """Test sensitivity reduction under high cortisol (stress)."""
        feedback_data = {
            'performance': 0.3,
            'accuracy': 0.4,
            'response_time': 2.0,
            'user_satisfaction': 0.2,
            'hormone_levels': {
                'dopamine': 0.3,
                'serotonin': 0.4,
                'cortisol': 0.8,
                'norepinephrine': 0.5
            }
        }
        
        initial_sensitivity = self.column.sensitivity
        self.column.adapt_sensitivity_enhanced(feedback_data)
        
        # High cortisol with poor performance should decrease sensitivity
        assert self.column.sensitivity < initial_sensitivity
    
    def test_receptor_sensitivity_adaptation(self):
        """Test individual receptor sensitivity adaptation."""
        # Setup receptors
        self.column.process_sensory_input("test visual data", "visual")
        self.column.process_sensory_input("test audio data", "auditory")
        
        feedback_data = {
            'performance': 0.7,
            'accuracy': 0.8,
            'response_time': 0.6,
            'user_satisfaction': 0.7,
            'modality_performance': {
                'visual': 0.9,
                'auditory': 0.3
            },
            'hormone_levels': {
                'dopamine': 0.7,
                'serotonin': 0.6,
                'cortisol': 0.2
            }
        }
        
        initial_visual_sensitivity = self.column.sensory_receptors['visual']['sensitivity']
        initial_auditory_sensitivity = self.column.sensory_receptors['auditory']['sensitivity']
        
        self.column.adapt_sensitivity_enhanced(feedback_data)
        
        # Visual should increase (good performance), auditory should decrease (poor performance)
        assert self.column.sensory_receptors['visual']['sensitivity'] > initial_visual_sensitivity
        assert self.column.sensory_receptors['auditory']['sensitivity'] < initial_auditory_sensitivity
    
    def test_feedback_integration_processing(self):
        """Test comprehensive feedback integration processing."""
        feedback_data = {
            'type': 'pattern_completion_validation',
            'source': 'test_system',
            'pattern_type': 'visual',
            'correct_prediction': True,
            'actual_completion': {'type': 'completed_visual', 'confidence': 0.9}
        }
        
        # Initialize completion accuracy tracking
        self.column.completion_accuracy['visual'] = {
            'predictions': 5,
            'correct_predictions': 3,
            'average_accuracy': 0.6,
            'last_updated': time.time()
        }
        
        result = self.column.process_feedback_integration(feedback_data)
        
        assert result['feedback_type'] == 'pattern_completion_validation'
        assert result['integration_success'] == True
        assert self.column.completion_accuracy['visual']['correct_predictions'] == 4
        assert self.column.completion_accuracy['visual']['average_accuracy'] == 0.8  # 4/5
    
    def test_cross_modal_feedback_processing(self):
        """Test cross-modal feedback processing."""
        feedback_data = {
            'type': 'cross_modal_feedback',
            'source': 'cross_modal_system',
            'primary_modality': 'visual',
            'secondary_modality': 'auditory',
            'success': True,
            'performance': 0.8,
            'accuracy': 0.7,
            'response_time': 0.5,
            'user_satisfaction': 0.8
        }
        
        # Setup receptors
        self.column.process_sensory_input("visual data", "visual")
        self.column.process_sensory_input("audio data", "auditory")
        
        initial_visual_sensitivity = self.column.sensory_receptors['visual']['sensitivity']
        initial_auditory_sensitivity = self.column.sensory_receptors['auditory']['sensitivity']
        
        result = self.column.process_feedback_integration(feedback_data)
        
        # Successful cross-modal feedback should increase both modality sensitivities
        assert self.column.sensory_receptors['visual']['sensitivity'] >= initial_visual_sensitivity
        assert self.column.sensory_receptors['auditory']['sensitivity'] >= initial_auditory_sensitivity
        assert result['integration_success'] == True
    
    def test_hormone_triggered_feedback_processing(self):
        """Test hormone-triggered feedback processing."""
        feedback_data = {
            'type': 'hormone_triggered_feedback',
            'source': 'hormone_system',
            'hormone_trigger': 'dopamine',
            'trigger_strength': 0.8,
            'behavioral_change': {'increased_learning': True},
            'performance': 0.7,
            'accuracy': 0.8,
            'response_time': 0.4,
            'user_satisfaction': 0.9
        }
        
        initial_sensitivity = self.column.sensitivity
        initial_learning_rate = self.column.learning_rate
        
        result = self.column.process_feedback_integration(feedback_data)
        
        # High dopamine trigger should increase sensitivity and learning rate
        assert self.column.sensitivity > initial_sensitivity
        assert self.column.learning_rate > initial_learning_rate
        assert result['integration_success'] == True
    
    def test_adaptive_performance_metrics(self):
        """Test comprehensive adaptive performance metrics."""
        # Add some feedback history
        for i in range(5):
            feedback = {
                'composite_score': 0.7 + (i * 0.05),
                'hormone_modulation': 1.1,
                'sensitivity_after': 1.0 + (i * 0.1),
                'learning_rate_after': 0.1 + (i * 0.01),
                'timestamp': time.time() + i
            }
            self.column.feedback_history.append(feedback)
        
        metrics = self.column.get_adaptive_performance_metrics()
        
        assert 'total_feedback_received' in metrics
        assert 'recent_feedback_count' in metrics
        assert 'average_composite_score' in metrics
        assert 'hormone_modulation_average' in metrics
        assert 'sensitivity_adaptation_range' in metrics
        assert 'learning_rate_adaptation_range' in metrics
        
        assert metrics['total_feedback_received'] == 5
        assert metrics['recent_feedback_count'] == 5
        assert metrics['average_composite_score'] > 0.7
        assert metrics['hormone_modulation_average'] == 1.1


class TestSensoryDataPropagator:
    """Test cross-lobe sensory data sharing functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        from mcp.lobes.experimental.lobe_event_bus import LobeEventBus
        self.event_bus = LobeEventBus()
        self.propagator = SensoryDataPropagator(self.event_bus)
    
    def test_propagation_rule_registration(self):
        """Test registration of propagation rules."""
        self.propagator.register_propagation_rule(
            source_lobe='pattern_recognition',
            target_lobes=['alignment_engine', 'hormone_engine'],
            data_types=['success', 'error'],
            priority=0.8
        )
        
        rule_id = 'pattern_recognition_to_alignment_engine_hormone_engine'
        assert rule_id in self.propagator.propagation_rules
        
        rule = self.propagator.propagation_rules[rule_id]
        assert rule['source_lobe'] == 'pattern_recognition'
        assert 'alignment_engine' in rule['target_lobes']
        assert 'success' in rule['data_types']
        assert rule['priority'] == 0.8
    
    def test_hormone_based_priority_adjustment(self):
        """Test hormone-based priority adjustment."""
        base_priority = 0.5
        hormone_levels = {
            'dopamine': 0.8,
            'cortisol': 0.7,
            'norepinephrine': 0.9,
            'serotonin': 0.6
        }
        
        # Test reward-related data with high dopamine
        adjusted_priority = self.propagator._adjust_priority_by_hormones(
            base_priority, hormone_levels, 'success'
        )
        assert adjusted_priority > base_priority
        
        # Test threat-related data with high cortisol
        adjusted_priority = self.propagator._adjust_priority_by_hormones(
            base_priority, hormone_levels, 'error'
        )
        assert adjusted_priority > base_priority
        
        # Test attention-demanding data with high norepinephrine
        adjusted_priority = self.propagator._adjust_priority_by_hormones(
            base_priority, hormone_levels, 'urgent'
        )
        assert adjusted_priority > base_priority
    
    def test_sensory_data_propagation(self):
        """Test complete sensory data propagation."""
        # Register propagation rule
        self.propagator.register_propagation_rule(
            source_lobe='pattern_recognition',
            target_lobes=['alignment_engine'],
            data_types=['test_data'],
            priority=0.5
        )
        
        sensory_data = {
            'source_lobe': 'pattern_recognition',
            'data_type': 'test_data',
            'modality': 'visual',
            'content': {'pattern': 'test_pattern'},
            'priority': 0.6,
            'confidence': 0.8
        }
        
        hormone_levels = {
            'dopamine': 0.6,
            'serotonin': 0.7,
            'cortisol': 0.2
        }
        
        result = self.propagator.propagate_sensory_data(sensory_data, hormone_levels)
        
        assert result['source_lobe'] == 'pattern_recognition'
        assert result['data_type'] == 'test_data'
        assert result['rules_applied'] == 1
        assert 'alignment_engine' in result['target_lobes']
        assert result['propagation_success'] == True
    
    def test_standardized_data_format_creation(self):
        """Test creation of standardized sensory data format."""
        sensory_data = {
            'data_type': 'pattern_recognition',
            'modality': 'visual',
            'content': {'pattern': 'test'},
            'confidence': 0.7
        }
        
        standardized = self.propagator._create_standardized_format(
            sensory_data, 'source_lobe', 'target_lobe', 0.8
        )
        
        assert standardized['type'] == 'cross_lobe_sensory_data'
        assert standardized['source_lobe'] == 'source_lobe'
        assert standardized['target_lobe'] == 'target_lobe'
        assert standardized['data_type'] == 'pattern_recognition'
        assert standardized['priority'] == 0.8
        assert 'propagation_id' in standardized


class TestAdaptiveSensitivityManager:
    """Test adaptive sensitivity management across columns."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = AdaptiveSensitivityManager()
    
    def test_column_registration_and_updates(self):
        """Test column registration and sensitivity updates."""
        self.manager.register_column('test_column', 1.0)
        
        assert 'test_column' in self.manager.column_sensitivities
        assert self.manager.column_sensitivities['test_column']['current_sensitivity'] == 1.0
        
        # Update sensitivity
        self.manager.update_column_sensitivity('test_column', 1.2, 0.8)
        
        assert self.manager.column_sensitivities['test_column']['current_sensitivity'] == 1.2
        assert len(self.manager.column_sensitivities['test_column']['sensitivity_history']) == 2
    
    def test_cross_column_learning(self):
        """Test cross-column learning application."""
        # Register columns
        self.manager.register_column('source_column', 1.0)
        self.manager.register_column('target_column', 1.0)
        
        # Set good performance correlation for source
        self.manager.column_sensitivities['source_column']['performance_correlation'] = 0.8
        self.manager.column_sensitivities['source_column']['current_sensitivity'] = 1.5
        
        initial_target_sensitivity = self.manager.column_sensitivities['target_column']['current_sensitivity']
        
        self.manager.apply_cross_column_learning('source_column', ['target_column'], 0.2)
        
        # Target column sensitivity should be adjusted based on source
        final_target_sensitivity = self.manager.column_sensitivities['target_column']['current_sensitivity']
        assert final_target_sensitivity != initial_target_sensitivity
        
        # Check learning tracking
        learning_key = 'source_column_to_target_column'
        assert learning_key in self.manager.cross_column_learning
        assert self.manager.cross_column_learning[learning_key]['applications'] == 1
    
    def test_hormone_modulation_application(self):
        """Test hormone-based modulation of column sensitivities."""
        # Register test columns
        self.manager.register_column('test_column_1', 1.0)
        self.manager.register_column('test_column_2', 1.0)
        
        hormone_levels = {
            'dopamine': 0.8,
            'cortisol': 0.3,
            'serotonin': 0.7
        }
        
        initial_sensitivities = {
            col_id: data['current_sensitivity'] 
            for col_id, data in self.manager.column_sensitivities.items()
        }
        
        self.manager.apply_hormone_modulation(hormone_levels)
        
        # Check that sensitivities were modulated
        for col_id, initial_sensitivity in initial_sensitivities.items():
            current_sensitivity = self.manager.column_sensitivities[col_id]['current_sensitivity']
            # With high dopamine and moderate serotonin, sensitivity should generally increase
            assert current_sensitivity >= initial_sensitivity * 0.9  # Allow for some variation
        
        # Check hormone modulation history
        assert len(self.manager.hormone_sensitivity_modulation) == 1
    
    def test_sensitivity_analytics(self):
        """Test comprehensive sensitivity analytics."""
        # Register multiple columns with different sensitivities
        self.manager.register_column('high_perf_column', 1.5)
        self.manager.register_column('low_perf_column', 0.8)
        self.manager.register_column('avg_perf_column', 1.0)
        
        # Set performance correlations
        self.manager.column_sensitivities['high_perf_column']['performance_correlation'] = 0.8
        self.manager.column_sensitivities['low_perf_column']['performance_correlation'] = 0.2
        self.manager.column_sensitivities['avg_perf_column']['performance_correlation'] = 0.5
        
        # Add some sensitivity history
        for col_id in self.manager.column_sensitivities:
            for i in range(10):
                sensitivity = 1.0 + (i * 0.1)
                self.manager.column_sensitivities[col_id]['sensitivity_history'].append(sensitivity)
        
        analytics = self.manager.get_sensitivity_analytics()
        
        assert analytics['total_columns'] == 3
        assert 'average_sensitivity' in analytics
        assert 'sensitivity_variance' in analytics
        assert len(analytics['high_performing_columns']) >= 1
        assert 'adaptation_trends' in analytics
        
        # Check high-performing columns identification
        high_perf_columns = [col['column_id'] for col in analytics['high_performing_columns']]
        assert 'high_perf_column' in high_perf_columns


class TestEnhancedPatternRecognitionEngineIntegration:
    """Test complete integration of adaptive sensitivity and feedback features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.engine = EnhancedPatternRecognitionEngine(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_cross_lobe_communication_setup(self):
        """Test cross-lobe communication setup."""
        assert self.engine.adaptive_sensitivity_manager is not None
        assert self.engine.sensory_data_propagator is not None
        assert len(self.engine.sensory_data_propagator.propagation_rules) > 0
    
    def test_cross_lobe_sensory_data_sharing_implementation(self):
        """Test complete cross-lobe sensory data sharing implementation."""
        sensory_data = {
            'data_type': 'success',
            'modality': 'visual',
            'content': {'pattern_recognized': True, 'confidence': 0.9},
            'priority': 0.7,
            'confidence': 0.9
        }
        
        hormone_levels = {
            'dopamine': 0.8,
            'serotonin': 0.6,
            'cortisol': 0.2
        }
        
        result = self.engine.implement_cross_lobe_sensory_data_sharing(
            sensory_data, hormone_levels
        )
        
        assert result['source_lobe'] == 'pattern_recognition'
        assert result['propagation_success'] == True
        assert len(result['target_lobes']) > 0
        
        # Check that sharing activity was stored
        sharing_stats = self.engine.get_cross_lobe_sharing_statistics()
        assert 'sharing_activity' in sharing_stats
        assert sharing_stats['total_columns'] > 0
    
    def test_adaptive_feedback_integration_processing(self):
        """Test comprehensive adaptive feedback integration."""
        feedback_data = {
            'type': 'comprehensive_feedback',
            'target_columns': ['text_processor', 'visual_processor'],
            'performance': 0.8,
            'accuracy': 0.9,
            'response_time': 0.5,
            'user_satisfaction': 0.8,
            'modality_performance': {
                'visual': 0.9,
                'text': 0.7
            },
            'hormone_levels': {
                'dopamine': 0.8,
                'serotonin': 0.7,
                'cortisol': 0.2,
                'norepinephrine': 0.6
            },
            'enable_cross_modal_learning': True,
            'primary_modality': 'visual',
            'secondary_modalities': ['text'],
            'cross_modal_strength': 0.15
        }
        
        result = self.engine.process_adaptive_feedback_integration(feedback_data)
        
        assert len(result['processed_columns']) >= 2
        assert result['cross_modal_learning_applied'] == True
        assert result['hormone_modulation_applied'] == True
        assert 'sensitivity_adjustments' in result
        
        # Check that columns were actually processed
        for column_result in result['processed_columns']:
            assert 'column_id' in column_result
            assert 'integration_result' in column_result
    
    def test_hormone_update_handling(self):
        """Test hormone update handling for sensitivity adjustment."""
        # Simulate hormone update
        hormone_data = {
            'dopamine': 0.9,
            'serotonin': 0.8,
            'cortisol': 0.1,
            'norepinephrine': 0.7
        }
        
        # Get initial sensitivities
        initial_sensitivities = {}
        for column_id, column in self.engine.neural_columns.items():
            initial_sensitivities[column_id] = column.sensitivity
        
        # Trigger hormone update
        self.engine._on_hormone_update(hormone_data)
        
        # Check that sensitivities were adjusted
        sensitivity_changes = 0
        for column_id, column in self.engine.neural_columns.items():
            if abs(column.sensitivity - initial_sensitivities[column_id]) > 0.01:
                sensitivity_changes += 1
        
        # At least some columns should have sensitivity changes
        assert sensitivity_changes > 0
    
    def test_cross_lobe_feedback_handling(self):
        """Test cross-lobe feedback handling."""
        feedback_data = {
            'source_lobe': 'alignment_engine',
            'type': 'pattern_recognition_feedback',
            'column_id': 'text_processor',
            'performance_data': {
                'performance_score': 0.8,
                'accuracy': 0.9,
                'response_time': 0.4
            },
            'feedback_type': 'association_strength_feedback',
            'association_key': 'text_visual',
            'helpful': True,
            'strength_adjustment': 0.1
        }
        
        # Trigger cross-lobe feedback
        self.engine._on_cross_lobe_feedback(feedback_data)
        
        # Check that the target column processed the feedback
        target_column = self.engine.neural_columns['text_processor']
        assert len(target_column.feedback_history) > 0
    
    def test_comprehensive_sharing_statistics(self):
        """Test comprehensive cross-lobe sharing statistics."""
        # Perform some sharing activities
        for i in range(3):
            sensory_data = {
                'data_type': f'test_type_{i}',
                'modality': 'visual',
                'content': {'test': f'data_{i}'},
                'priority': 0.5 + (i * 0.1),
                'confidence': 0.7 + (i * 0.1)
            }
            
            self.engine.implement_cross_lobe_sensory_data_sharing(sensory_data)
        
        stats = self.engine.get_cross_lobe_sharing_statistics()
        
        assert 'sharing_activity' in stats
        assert 'propagation_rules' in stats
        assert 'sensitivity_management' in stats
        assert stats['total_columns'] > 0
        
        # Check that sharing activities were recorded
        sharing_activity = stats['sharing_activity']
        assert len(sharing_activity) > 0


def run_comprehensive_tests():
    """Run all comprehensive tests for adaptive sensitivity and feedback integration."""
    print("Running comprehensive tests for adaptive sensitivity and feedback integration...")
    
    # Test Enhanced Neural Column Adaptive Sensitivity
    print("\n=== Testing Enhanced Neural Column Adaptive Sensitivity ===")
    test_class = TestEnhancedNeuralColumnAdaptiveSensitivity()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_class.setup_method()
        method = getattr(test_class, method_name)
        try:
            method()
            print(f"✓ {method_name} passed")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
    
    # Test Sensory Data Propagator
    print("\n=== Testing Sensory Data Propagator ===")
    test_class = TestSensoryDataPropagator()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_class.setup_method()
        method = getattr(test_class, method_name)
        try:
            method()
            print(f"✓ {method_name} passed")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
    
    # Test Adaptive Sensitivity Manager
    print("\n=== Testing Adaptive Sensitivity Manager ===")
    test_class = TestAdaptiveSensitivityManager()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_class.setup_method()
        method = getattr(test_class, method_name)
        try:
            method()
            print(f"✓ {method_name} passed")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
    
    # Test Enhanced Pattern Recognition Engine Integration
    print("\n=== Testing Enhanced Pattern Recognition Engine Integration ===")
    test_class = TestEnhancedPatternRecognitionEngineIntegration()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_class.setup_method()
        method = getattr(test_class, method_name)
        try:
            method()
            print(f"✓ {method_name} passed")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
        finally:
            test_class.teardown_method()
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    run_comprehensive_tests()