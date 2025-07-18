#!/usr/bin/env python3
"""
Simple demonstration of Enhanced Pattern Recognition Engine capabilities.

This script demonstrates the key features of the enhanced pattern recognition system:
- Neural column architecture with specialized processing
- Sensory input processing across multiple modalities
- Pattern association learning and completion prediction
- Adaptive sensitivity based on feedback
"""

import tempfile
import os
import json

# Import the enhanced pattern recognition engine
try:
    from src.mcp.lobes.pattern_recognition_engine_enhanced import EnhancedPatternRecognitionEngine
except ImportError:
    import sys
    sys.path.append('src')
    from mcp.lobes.pattern_recognition_engine_enhanced import EnhancedPatternRecognitionEngine


def demonstrate_enhanced_pattern_recognition():
    """Demonstrate enhanced pattern recognition capabilities."""
    print("🧠 Enhanced Pattern Recognition Engine Demonstration")
    print("=" * 60)
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        # Initialize the enhanced engine
        engine = EnhancedPatternRecognitionEngine(db_path=temp_db.name)
        print(f"✅ Initialized engine with {len(engine.neural_columns)} neural columns")
        
        # Demonstrate sensory input processing
        print("\n🔍 Sensory Input Processing:")
        print("-" * 30)
        
        # Process different types of sensory input
        sensory_inputs = [
            ("textual", "Hello, this is a text message about artificial intelligence"),
            ("visual", {"shape": "circle", "color": "red", "size": "large"}),
            ("auditory", [440, 880, 1320, 1760]),  # Musical frequencies
            ("tactile", {"pressure": 0.7, "texture": "rough", "temperature": "warm"}),
            ("multimodal", {"text": "red circle", "image_data": "circle_red.jpg", "sound": [440]})
        ]
        
        processed_results = []
        for modality, data in sensory_inputs:
            result = engine.process_sensory_input(data, modality)
            processed_results.append(result)
            
            print(f"  {modality.upper()}: Processed through column '{result['column_id']}'")
            print(f"    Patterns detected: {len(result['processed_patterns'])}")
            if result['processed_patterns']:
                confidence = result['processed_patterns'][0].get('confidence', 0.0)
                print(f"    Confidence: {confidence:.2f}")
        
        # Demonstrate pattern association learning
        print("\n🎓 Pattern Association Learning:")
        print("-" * 30)
        
        # Create training patterns that should be associated
        training_patterns = [
            {'type': 'text', 'data': 'hello world', 'confidence': 0.8},
            {'type': 'textual', 'data': 'greeting message', 'confidence': 0.7},
            {'type': 'visual', 'data': 'red circle', 'confidence': 0.9},
            {'type': 'visual_pattern', 'data': 'circular shape', 'confidence': 0.8},
            {'type': 'auditory', 'data': 'musical note', 'confidence': 0.6},
            {'type': 'sound', 'data': 'frequency pattern', 'confidence': 0.7}
        ]
        
        # Learn associations
        learning_result = engine.learn_pattern_associations(training_patterns, strength=0.2)
        print(f"  Learned {learning_result['associations_learned']} associations")
        print(f"  From {learning_result['patterns_processed']} patterns")
        
        # Demonstrate pattern completion prediction
        print("\n🔮 Pattern Completion Prediction:")
        print("-" * 30)
        
        # Test completion predictions
        partial_patterns = [
            {'type': 'text', 'data': 'hello', 'confidence': 0.6},
            {'type': 'visual', 'data': 'red', 'confidence': 0.5},
            {'type': 'auditory', 'data': 'musical', 'confidence': 0.7}
        ]
        
        for partial in partial_patterns:
            completion = engine.predict_pattern_completion(partial)
            print(f"  Partial: '{partial['data']}' ({partial['type']})")
            print(f"    Completion confidence: {completion['confidence']:.2f}")
            print(f"    Source: {completion['completion_source']}")
        
        # Demonstrate pattern response generation
        print("\n💬 Pattern Response Generation:")
        print("-" * 30)
        
        test_patterns = [
            {'type': 'text', 'data': 'urgent message', 'confidence': 0.9},
            {'type': 'visual_pattern', 'data': 'warning sign', 'confidence': 0.8},
            {'type': 'sequence', 'data': [1, 2, 3, 4], 'confidence': 0.7}
        ]
        
        for pattern in test_patterns:
            response = engine.generate_pattern_response(pattern)
            print(f"  Pattern: '{pattern['data']}' ({pattern['type']})")
            print(f"    Response: {response['response_type']}")
            print(f"    Confidence: {response['confidence']:.2f}")
            print(f"    Generated by: {response['generating_column']}")
        
        # Demonstrate adaptive sensitivity
        print("\n⚡ Adaptive Sensitivity:")
        print("-" * 30)
        
        # Simulate feedback for different performance scenarios
        feedback_scenarios = [
            {
                'name': 'Excellent Performance',
                'feedback': {
                    'performance': 0.95,
                    'accuracy': 0.92,
                    'response_time': 0.3,
                    'user_satisfaction': 0.9
                }
            },
            {
                'name': 'Poor Performance',
                'feedback': {
                    'performance': 0.2,
                    'accuracy': 0.3,
                    'response_time': 2.5,
                    'user_satisfaction': 0.1
                }
            }
        ]
        
        for scenario in feedback_scenarios:
            # Get initial sensitivity
            text_column = engine.neural_columns['text_processor']
            initial_sensitivity = text_column.sensitivity
            
            # Apply feedback
            engine.adapt_column_sensitivity('text_processor', scenario['feedback'])
            
            # Show adaptation
            final_sensitivity = text_column.sensitivity
            change = final_sensitivity - initial_sensitivity
            
            print(f"  {scenario['name']}:")
            print(f"    Sensitivity change: {change:+.3f} ({initial_sensitivity:.3f} → {final_sensitivity:.3f})")
        
        # Demonstrate cross-lobe sensory data sharing
        print("\n🔗 Cross-Lobe Sensory Data Sharing:")
        print("-" * 30)
        
        # Get shared sensory data
        shared_data = engine.get_cross_lobe_sensory_data(limit=3)
        
        for modality, data_list in shared_data.items():
            if data_list:
                print(f"  {modality.upper()}: {len(data_list)} shared data points")
                latest = data_list[-1]
                print(f"    Latest: Column '{latest['column_id']}' processed data")
        
        # Show column performance summary
        print("\n📊 Neural Column Performance Summary:")
        print("-" * 30)
        
        column_states = engine.get_column_states()
        for column_id, state in column_states.items():
            if state['feedback_count'] > 0 or state['association_count'] > 0:
                print(f"  {column_id}:")
                print(f"    Sensitivity: {state['sensitivity']:.3f}")
                print(f"    Associations: {state['association_count']}")
                print(f"    Feedback events: {state['feedback_count']}")
                if state['recent_feedback_score'] != 0.5:  # Not default
                    print(f"    Recent feedback score: {state['recent_feedback_score']:.3f}")
        
        # Demonstrate batch processing efficiency
        print("\n⚡ Batch Processing Efficiency:")
        print("-" * 30)
        
        import time
        
        # Create a batch of patterns
        batch_patterns = [
            {'type': 'text', 'data': f'message_{i}', 'confidence': 0.7}
            for i in range(50)
        ]
        
        start_time = time.time()
        batch_results = engine.batch_process_patterns(batch_patterns)
        end_time = time.time()
        
        processing_time = end_time - start_time
        patterns_per_second = len(batch_patterns) / processing_time
        
        print(f"  Processed {len(batch_patterns)} patterns in {processing_time:.3f} seconds")
        print(f"  Rate: {patterns_per_second:.1f} patterns/second")
        
        print("\n✅ Enhanced Pattern Recognition Engine demonstration completed!")
        print("\nKey Features Demonstrated:")
        print("- ✅ Neural column architecture with specialized processing")
        print("- ✅ Multi-modal sensory input processing")
        print("- ✅ Pattern association learning with strength tracking")
        print("- ✅ Pattern completion prediction with accuracy tracking")
        print("- ✅ Adaptive sensitivity based on multi-dimensional feedback")
        print("- ✅ Pattern response generation with learned responses")
        print("- ✅ Cross-lobe sensory data sharing")
        print("- ✅ Efficient batch processing")
        print("- ✅ Performance metrics tracking and optimization")
        
    finally:
        # Clean up
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)


if __name__ == '__main__':
    demonstrate_enhanced_pattern_recognition()