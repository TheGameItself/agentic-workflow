#!/usr/bin/env python3
"""
Comprehensive tests for Enhanced Pattern Recognition Engine with Neural Column Architecture.

Tests pattern recognition accuracy, learning effectiveness, sensory input processing,
pattern response generation, and completion prediction capabilities.
"""

import unittest
import tempfile
import os
import json
import time
from typing import Dict, List, Any

# Import the enhanced pattern recognition engine
try:
    from src.mcp.lobes.pattern_recognition_engine_enhanced import (
        EnhancedPatternRecognitionEngine,
        EnhancedNeuralColumn
    )
except ImportError:
    # Fallback for testing
    import sys
    sys.path.append('src')
    from mcp.lobes.pattern_recognition_engine_enhanced import (
        EnhancedPatternRecognitionEngine,
        EnhancedNeuralColumn
    )


class TestEnhancedNeuralColumn(unittest.TestCase):
    """Test cases for EnhancedNeuralColumn functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.column = EnhancedNeuralColumn(
            'test_column',
            ['text', 'string', 'word'],
            position=(0, 0, 0)
        )
    
    def test_column_initialization(self):
        """Test neural column initialization."""
        self.assertEqual(self.column.column_id, 'test_column')
        self.assertEqual(self.column.pattern_types, ['text', 'string', 'word'])
        self.assertEqual(self.column.position, (0, 0, 0))
        self.assertEqual(self.column.sensitivity, 1.0)
        self.assertEqual(self.column.learning_rate, 0.1)
        self.assertFalse(self.column.use_neural)
    
    def test_pattern_processing(self):
        """Test basic pattern processing through neural column."""
        test_pattern = {
            'type': 'text',
            'data': 'hello world',
            'confidence': 0.8
        }
        
        result = self.column.process_pattern(test_pattern)
        
        self.assertIn('type', result)
        self.assertIn('data', result)
        self.assertIn('confidence', result)
        self.assertIn('column_id', result)
        self.assertEqual(result['column_id'], 'test_column')
        self.assertEqual(result['processing_method'], 'algorithmic')
        self.assertGreater(result['confidence'], 0.0)
    
    def test_sensory_input_processing(self):
        """Test sensory input processing capabilities."""
        # Test text input
        text_result = self.column.process_sensory_input("Hello world", "textual")
        self.assertEqual(text_result['modality'], 'textual')
        self.assertEqual(text_result['column_id'], 'test_column')
        self.assertIn('processed_patterns', text_result)
        self.assertGreater(len(text_result['processed_patterns']), 0)
        
        # Test structured input
        dict_result = self.column.process_sensory_input(
            {"key": "value", "number": 42}, "visual"
        )
        self.assertEqual(dict_result['modality'], 'visual')
        self.assertIn('processed_patterns', dict_result)
        
        # Test sequence input
        list_result = self.column.process_sensory_input([1, 2, 3, 4], "auditory")
        self.assertEqual(list_result['modality'], 'auditory')
        self.assertIn('processed_patterns', list_result)
    
    def test_pattern_response_generation(self):
        """Test pattern response generation."""
        test_pattern = {
            'type': 'text',
            'data': 'test message',
            'confidence': 0.7
        }
        
        response = self.column.generate_pattern_response(test_pattern)
        
        self.assertIn('response_type', response)
        self.assertIn('response_data', response)
        self.assertIn('confidence', response)
        self.assertIn('generating_column', response)
        self.assertEqual(response['generating_column'], 'test_column')
        self.assertFalse(response['learned_response'])  # First time, should be default
        
        # Test learned response after multiple uses
        for _ in range(3):
            self.column.generate_pattern_response(test_pattern)
        
        # Should now have learned response pattern
        self.assertIn('text', self.column.response_patterns)
    
    def test_pattern_association_learning(self):
        """Test pattern association learning with strength tracking."""
        pattern1 = {'type': 'text', 'data': 'hello', 'confidence': 0.8}
        pattern2 = {'type': 'word', 'data': 'greeting', 'confidence': 0.7}
        
        # Learn association
        self.column.learn_pattern_association_with_strength(
            pattern1, pattern2, strength=0.2, context="greeting_context"
        )
        
        # Check associations were learned
        self.assertIn('text', self.column.pattern_associations)
        self.assertIn('word', self.column.pattern_associations)
        self.assertGreater(self.column.pattern_associations['text'], 0.0)
        self.assertGreater(self.column.pattern_associations['word'], 0.0)
        
        # Check association strengths
        association_key = "text_word"
        self.assertIn(association_key, self.column.association_strengths)
        self.assertGreater(self.column.association_strengths[association_key]['strength'], 0.0)
        self.assertEqual(self.column.association_strengths[association_key]['reinforcements'], 1)
        self.assertIn("greeting_context", self.column.association_strengths[association_key]['contexts'])
    
    def test_pattern_completion_prediction(self):
        """Test pattern completion prediction with accuracy tracking."""
        # First, learn some associations
        pattern1 = {'type': 'text', 'data': 'hello world', 'confidence': 0.8}
        pattern2 = {'type': 'text', 'data': 'hello universe', 'confidence': 0.7}
        
        self.column.learn_pattern_association_with_strength(pattern1, pattern2, 0.3)
        
        # Test completion prediction
        partial_pattern = {'type': 'text', 'data': 'hello', 'confidence': 0.6}
        completion = self.column.predict_pattern_completion_with_accuracy(partial_pattern)
        
        self.assertIn('type', completion)
        self.assertIn('confidence', completion)
        self.assertIn('completion_source', completion)
        self.assertIn('column_id', completion)
        self.assertEqual(completion['column_id'], 'test_column')
        self.assertEqual(completion['completion_source'], 'enhanced_association_learning')
        
        # Check accuracy tracking was initialized
        self.assertIn('text', self.column.completion_accuracy)
        self.assertEqual(self.column.completion_accuracy['text']['predictions'], 1)
    
    def test_sensitivity_adaptation(self):
        """Test enhanced sensitivity adaptation based on feedback."""
        initial_sensitivity = self.column.sensitivity
        initial_learning_rate = self.column.learning_rate
        
        # Test positive feedback
        positive_feedback = {
            'performance': 0.9,
            'accuracy': 0.85,
            'response_time': 0.5,
            'user_satisfaction': 0.8
        }
        
        self.column.adapt_sensitivity_enhanced(positive_feedback)
        
        # Sensitivity should increase with good feedback
        self.assertGreaterEqual(self.column.sensitivity, initial_sensitivity)
        
        # Test negative feedback
        negative_feedback = {
            'performance': 0.2,
            'accuracy': 0.3,
            'response_time': 2.0,
            'user_satisfaction': 0.1
        }
        
        self.column.adapt_sensitivity_enhanced(negative_feedback)
        
        # Check feedback history
        self.assertEqual(len(self.column.feedback_history), 2)
        self.assertIn('composite_score', self.column.feedback_history[-1])
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking and comparison."""
        test_pattern = {'type': 'text', 'data': 'test', 'confidence': 0.7}
        
        # Process pattern to generate metrics
        result = self.column.process_pattern(test_pattern)
        
        # Check metrics were recorded
        self.assertIn('algorithmic', self.column.performance_metrics)
        algo_metrics = self.column.performance_metrics['algorithmic']
        
        self.assertIn('processing_time', algo_metrics)
        self.assertIn('confidence', algo_metrics)
        self.assertIn('avg_processing_time', algo_metrics)
        self.assertIn('avg_confidence', algo_metrics)
        
        # Test performance summary
        summary = self.column.get_column_performance_summary()
        self.assertIn('column_id', summary)
        self.assertIn('performance_metrics', summary)
        self.assertIn('association_count', summary)
        self.assertIn('feedback_count', summary)


class TestEnhancedPatternRecognitionEngine(unittest.TestCase):
    """Test cases for EnhancedPatternRecognitionEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.engine = EnhancedPatternRecognitionEngine(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_engine_initialization(self):
        """Test pattern recognition engine initialization."""
        self.assertIsNotNone(self.engine.neural_columns)
        self.assertGreater(len(self.engine.neural_columns), 0)
        
        # Check that default columns were created
        expected_columns = [
            'text_processor', 'visual_processor', 'sequence_processor',
            'structure_processor', 'auditory_processor', 'tactile_processor',
            'multimodal_processor'
        ]
        
        for column_id in expected_columns:
            self.assertIn(column_id, self.engine.neural_columns)
    
    def test_neural_column_creation(self):
        """Test creation of new neural columns."""
        column = self.engine.create_neural_column(
            'custom_column',
            ['custom', 'special'],
            position=(2, 2, 2)
        )
        
        self.assertIsInstance(column, EnhancedNeuralColumn)
        self.assertEqual(column.column_id, 'custom_column')
        self.assertEqual(column.pattern_types, ['custom', 'special'])
        self.assertEqual(column.position, (2, 2, 2))
        self.assertIn('custom_column', self.engine.neural_columns)
    
    def test_sensory_input_processing(self):
        """Test sensory input processing through engine."""
        # Test text processing
        text_result = self.engine.process_sensory_input("Hello world", "textual")
        self.assertEqual(text_result['modality'], 'textual')
        self.assertIn('processed_patterns', text_result)
        
        # Test visual processing
        visual_result = self.engine.process_sensory_input(
            {"shape": "circle", "color": "red"}, "visual"
        )
        self.assertEqual(visual_result['modality'], 'visual')
        
        # Test sequence processing
        sequence_result = self.engine.process_sensory_input([1, 2, 3, 4, 5], "temporal")
        self.assertEqual(sequence_result['modality'], 'temporal')
        
        # Check cross-lobe sensory data sharing
        self.assertIn('textual', self.engine.sensory_data_sharing)
        self.assertIn('visual', self.engine.sensory_data_sharing)
    
    def test_pattern_response_generation(self):
        """Test pattern response generation through engine."""
        test_pattern = {
            'type': 'text',
            'data': 'test message',
            'confidence': 0.8
        }
        
        response = self.engine.generate_pattern_response(test_pattern)
        
        self.assertIn('response_type', response)
        self.assertIn('response_data', response)
        self.assertIn('confidence', response)
        self.assertIn('generating_column', response)
    
    def test_pattern_association_learning(self):
        """Test pattern association learning across columns."""
        patterns = [
            {'type': 'text', 'data': 'hello', 'confidence': 0.8},
            {'type': 'word', 'data': 'greeting', 'confidence': 0.7},
            {'type': 'textual', 'data': 'salutation', 'confidence': 0.6}
        ]
        
        result = self.engine.learn_pattern_associations(patterns, strength=0.2)
        
        self.assertIn('associations_learned', result)
        self.assertIn('patterns_processed', result)
        self.assertEqual(result['patterns_processed'], 3)
        self.assertGreater(result['associations_learned'], 0)
    
    def test_pattern_completion_prediction(self):
        """Test pattern completion prediction through engine."""
        # First learn some associations
        training_patterns = [
            {'type': 'text', 'data': 'hello world', 'confidence': 0.8},
            {'type': 'text', 'data': 'hello universe', 'confidence': 0.7}
        ]
        
        self.engine.learn_pattern_associations(training_patterns, strength=0.3)
        
        # Test completion
        partial_pattern = {'type': 'text', 'data': 'hello', 'confidence': 0.6}
        completion = self.engine.predict_pattern_completion(partial_pattern)
        
        self.assertIn('type', completion)
        self.assertIn('confidence', completion)
        self.assertIn('completion_source', completion)
    
    def test_column_sensitivity_adaptation(self):
        """Test column sensitivity adaptation."""
        feedback = {
            'performance': 0.9,
            'accuracy': 0.85,
            'response_time': 0.3,
            'user_satisfaction': 0.8
        }
        
        # Test adaptation for existing column
        self.engine.adapt_column_sensitivity('text_processor', feedback)
        
        # Check that column was adapted
        text_column = self.engine.neural_columns['text_processor']
        self.assertGreater(len(text_column.feedback_history), 0)
    
    def test_batch_pattern_processing(self):
        """Test batch processing of multiple patterns."""
        patterns = [
            {'type': 'text', 'data': 'hello', 'confidence': 0.8},
            {'type': 'visual', 'data': 'image_data', 'confidence': 0.7},
            {'type': 'list', 'data': [1, 2, 3], 'confidence': 0.6},
            {'type': 'dict', 'data': {'key': 'value'}, 'confidence': 0.9}
        ]
        
        results = self.engine.batch_process_patterns(patterns)
        
        self.assertEqual(len(results), len(patterns))
        for result in results:
            self.assertIn('type', result)
            self.assertIn('confidence', result)
            self.assertIn('column_id', result)
    
    def test_cross_lobe_sensory_data_sharing(self):
        """Test cross-lobe sensory data sharing functionality."""
        # Process different types of sensory data
        self.engine.process_sensory_input("text data", "textual")
        self.engine.process_sensory_input([1, 2, 3], "auditory")
        self.engine.process_sensory_input({"visual": "data"}, "visual")
        
        # Test getting all sensory data
        all_data = self.engine.get_cross_lobe_sensory_data()
        self.assertIn('textual', all_data)
        self.assertIn('auditory', all_data)
        self.assertIn('visual', all_data)
        
        # Test getting specific modality data
        textual_data = self.engine.get_cross_lobe_sensory_data('textual')
        self.assertIn('textual', textual_data)
        self.assertGreater(len(textual_data['textual']), 0)
    
    def test_column_states_tracking(self):
        """Test neural column states tracking."""
        # Process some data to activate columns
        self.engine.process_sensory_input("test text", "textual")
        self.engine.process_sensory_input([1, 2, 3], "temporal")
        
        # Get column states
        states = self.engine.get_column_states()
        
        self.assertIsInstance(states, dict)
        self.assertIn('text_processor', states)
        self.assertIn('sequence_processor', states)
        
        # Check state structure
        for column_id, state in states.items():
            self.assertIn('column_id', state)
            self.assertIn('sensitivity', state)
            self.assertIn('learning_rate', state)
            self.assertIn('performance_metrics', state)


class TestPatternRecognitionAccuracy(unittest.TestCase):
    """Test cases for pattern recognition accuracy evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.engine = EnhancedPatternRecognitionEngine(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_pattern_recognition_accuracy_evaluation(self):
        """Test pattern recognition accuracy evaluation."""
        # Create test patterns and ground truth
        test_patterns = [
            {'type': 'text', 'data': 'hello', 'confidence': 0.8},
            {'type': 'visual', 'data': 'image', 'confidence': 0.7},
            {'type': 'list', 'data': [1, 2, 3], 'confidence': 0.9}
        ]
        
        ground_truth = [
            {'type': 'text', 'data': 'hello'},
            {'type': 'visual', 'data': 'image'},
            {'type': 'list', 'data': [1, 2, 3]}
        ]
        
        accuracy_results = self.engine.evaluate_pattern_recognition_accuracy(
            test_patterns, ground_truth
        )
        
        self.assertIn('accuracy', accuracy_results)
        self.assertIn('average_confidence', accuracy_results)
        self.assertIn('correct_count', accuracy_results)
        self.assertIn('total_count', accuracy_results)
        
        self.assertGreaterEqual(accuracy_results['accuracy'], 0.0)
        self.assertLessEqual(accuracy_results['accuracy'], 1.0)
        self.assertEqual(accuracy_results['total_count'], 3)
    
    def test_learning_effectiveness_evaluation(self):
        """Test learning effectiveness evaluation."""
        # Create training patterns
        training_patterns = [
            {'type': 'text', 'data': 'hello world', 'confidence': 0.8},
            {'type': 'text', 'data': 'hello universe', 'confidence': 0.7},
            {'type': 'word', 'data': 'greeting', 'confidence': 0.9},
            {'type': 'word', 'data': 'salutation', 'confidence': 0.6}
        ]
        
        # Create test patterns for completion
        test_patterns = [
            {'type': 'text', 'data': 'hello complete sentence', 'confidence': 0.8},
            {'type': 'word', 'data': 'greeting phrase', 'confidence': 0.7}
        ]
        
        learning_results = self.engine.evaluate_learning_effectiveness(
            training_patterns, test_patterns, learning_iterations=3
        )
        
        self.assertIn('average_confidence', learning_results)
        self.assertIn('association_completions', learning_results)
        self.assertIn('total_completions', learning_results)
        self.assertIn('association_ratio', learning_results)
        
        self.assertGreaterEqual(learning_results['average_confidence'], 0.0)
        self.assertLessEqual(learning_results['average_confidence'], 1.0)
        self.assertEqual(learning_results['total_completions'], 2)
    
    def test_sensory_processing_accuracy(self):
        """Test accuracy of sensory input processing."""
        # Test different modalities
        modalities_data = [
            ("textual", "This is a text message"),
            ("visual", {"shape": "circle", "color": "red"}),
            ("auditory", [440, 880, 1320]),  # Frequencies
            ("tactile", {"pressure": 0.5, "texture": "smooth"}),
            ("multimodal", {"text": "hello", "image": "data", "sound": [440]})
        ]
        
        for modality, data in modalities_data:
            result = self.engine.process_sensory_input(data, modality)
            
            # Check basic structure
            self.assertEqual(result['modality'], modality)
            self.assertIn('processed_patterns', result)
            self.assertIn('column_id', result)
            self.assertGreater(len(result['processed_patterns']), 0)
            
            # Check pattern processing quality
            for pattern in result['processed_patterns']:
                self.assertIn('type', pattern)
                self.assertIn('confidence', pattern)
                self.assertGreater(pattern['confidence'], 0.0)
                self.assertLessEqual(pattern['confidence'], 1.0)
    
    def test_response_generation_quality(self):
        """Test quality of pattern response generation."""
        test_patterns = [
            {'type': 'text', 'data': 'hello', 'confidence': 0.8},
            {'type': 'visual_pattern', 'data': 'circle', 'confidence': 0.7},
            {'type': 'sequence', 'data': [1, 2, 3], 'confidence': 0.9}
        ]
        
        for pattern in test_patterns:
            response = self.engine.generate_pattern_response(pattern)
            
            # Check response structure
            self.assertIn('response_type', response)
            self.assertIn('response_data', response)
            self.assertIn('confidence', response)
            self.assertIn('generating_column', response)
            
            # Check response quality
            self.assertGreater(response['confidence'], 0.0)
            self.assertLessEqual(response['confidence'], 1.0)
            self.assertIsNotNone(response['response_data'])


class TestPatternRecognitionPerformance(unittest.TestCase):
    """Test cases for pattern recognition performance and optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.engine = EnhancedPatternRecognitionEngine(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_processing_speed(self):
        """Test processing speed of pattern recognition."""
        patterns = [
            {'type': 'text', 'data': f'message_{i}', 'confidence': 0.8}
            for i in range(100)
        ]
        
        start_time = time.time()
        results = self.engine.batch_process_patterns(patterns)
        end_time = time.time()
        
        processing_time = end_time - start_time
        patterns_per_second = len(patterns) / processing_time
        
        # Should process at least 10 patterns per second
        self.assertGreater(patterns_per_second, 10)
        self.assertEqual(len(results), len(patterns))
    
    def test_memory_efficiency(self):
        """Test memory efficiency of pattern storage."""
        # Process many patterns to test memory management
        for i in range(200):
            self.engine.process_sensory_input(f"data_{i}", "textual")
        
        # Check that sensory data sharing doesn't grow unbounded
        textual_data = self.engine.get_cross_lobe_sensory_data('textual')
        self.assertLessEqual(len(textual_data['textual']), 50)  # Should be capped
    
    def test_learning_convergence(self):
        """Test that learning converges to better performance."""
        # Create consistent training patterns
        training_patterns = [
            {'type': 'text', 'data': 'hello world', 'confidence': 0.8},
            {'type': 'text', 'data': 'hello universe', 'confidence': 0.8}
        ] * 10  # Repeat for consistency
        
        # Measure initial performance
        initial_completion = self.engine.predict_pattern_completion(
            {'type': 'text', 'data': 'hello', 'confidence': 0.6}
        )
        initial_confidence = initial_completion['confidence']
        
        # Train multiple iterations
        for _ in range(5):
            self.engine.learn_pattern_associations(training_patterns, strength=0.1)
        
        # Measure final performance
        final_completion = self.engine.predict_pattern_completion(
            {'type': 'text', 'data': 'hello', 'confidence': 0.6}
        )
        final_confidence = final_completion['confidence']
        
        # Performance should improve with learning
        self.assertGreaterEqual(final_confidence, initial_confidence)


def run_comprehensive_tests():
    """Run all comprehensive tests for enhanced pattern recognition."""
    test_suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestEnhancedNeuralColumn),
        unittest.TestLoader().loadTestsFromTestCase(TestEnhancedPatternRecognitionEngine),
        unittest.TestLoader().loadTestsFromTestCase(TestPatternRecognitionAccuracy),
        unittest.TestLoader().loadTestsFromTestCase(TestPatternRecognitionPerformance)
    ]
    
    combined_suite = unittest.TestSuite(test_suites)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running Enhanced Pattern Recognition Engine Tests...")
    print("=" * 60)
    
    success = run_comprehensive_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Enhanced Pattern Recognition Engine is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above for details.")
    
    print("\nTest Coverage:")
    print("- Neural column initialization and configuration")
    print("- Pattern processing with algorithmic and neural alternatives")
    print("- Sensory input processing across multiple modalities")
    print("- Pattern response generation with learned responses")
    print("- Pattern association learning with strength tracking")
    print("- Pattern completion prediction with accuracy tracking")
    print("- Adaptive sensitivity based on multi-dimensional feedback")
    print("- Cross-lobe sensory data sharing")
    print("- Batch processing efficiency")
    print("- Performance metrics tracking and comparison")
    print("- Learning effectiveness evaluation")
    print("- Memory efficiency and resource management")