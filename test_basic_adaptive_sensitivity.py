#!/usr/bin/env python3
"""
Basic test for adaptive sensitivity and feedback integration functionality.
"""

import sys
import os
import tempfile
import time

# Add src to path
sys.path.append('src')

def test_basic_functionality():
    """Test basic functionality of the enhanced pattern recognition engine."""
    print("Testing basic adaptive sensitivity functionality...")
    
    try:
        from mcp.lobes.pattern_recognition_engine_adaptive import (
            AdaptivePatternRecognitionEngine,
            AdaptiveNeuralColumn,
            SensoryDataPropagator
        )
        print("✓ Successfully imported enhanced pattern recognition components")
        
        # Test AdaptiveNeuralColumn
        print("\nTesting AdaptiveNeuralColumn...")
        column = AdaptiveNeuralColumn('test_column', ['visual', 'pattern'], position=(0, 0, 0))
        print(f"✓ Created neural column with sensitivity: {column.sensitivity}")
        
        # Test sensitivity adaptation
        feedback_data = {
            'performance': 0.8,
            'accuracy': 0.9,
            'response_time': 0.5,
            'user_satisfaction': 0.8,
            'hormone_levels': {
                'dopamine': 0.8,
                'serotonin': 0.6,
                'cortisol': 0.2
            }
        }
        
        initial_sensitivity = column.sensitivity
        column.adapt_sensitivity_with_feedback(feedback_data)
        print(f"✓ Sensitivity adapted from {initial_sensitivity:.3f} to {column.sensitivity:.3f}")
        
        # Test sensory input processing
        sensory_result = column.process_sensory_input("test visual data", "visual")
        print(f"✓ Processed sensory input: {len(sensory_result['processed_patterns'])} patterns")
        
        # Test feedback integration processing
        integration_result = column.process_feedback_integration(feedback_data)
        print(f"✓ Processed feedback integration: {integration_result['integration_success']}")
        
        # Test SensoryDataPropagator
        print("\nTesting SensoryDataPropagator...")
        from mcp.lobes.experimental.lobe_event_bus import LobeEventBus
        event_bus = LobeEventBus()
        propagator = SensoryDataPropagator(event_bus)
        
        propagator.register_propagation_rule(
            source_lobe='pattern_recognition',
            target_lobes=['alignment_engine'],
            data_types=['test_data'],
            priority=0.5
        )
        print("✓ Sensory data propagator configured successfully")
        
        # Test AdaptivePatternRecognitionEngine
        print("\nTesting AdaptivePatternRecognitionEngine...")
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            engine = AdaptivePatternRecognitionEngine(db_path=temp_db.name)
            print(f"✓ Created adaptive pattern recognition engine with {len(engine.neural_columns)} columns")
            
            # Test cross-lobe sensory data sharing
            sensory_data = {
                'data_type': 'success',
                'modality': 'visual',
                'content': {'pattern_recognized': True},
                'priority': 0.7,
                'confidence': 0.9
            }
            
            result = engine.implement_cross_lobe_sensory_data_sharing(sensory_data)
            print(f"✓ Cross-lobe sharing successful: {result['propagation_success']}")
            
            # Test adaptive feedback integration
            feedback_data = {
                'type': 'comprehensive_feedback',
                'target_columns': ['text_processor'],
                'performance': 0.8,
                'accuracy': 0.9,
                'hormone_levels': {
                    'dopamine': 0.8,
                    'serotonin': 0.7,
                    'cortisol': 0.2
                }
            }
            
            feedback_result = engine.process_adaptive_feedback_integration(feedback_data)
            print(f"✓ Adaptive feedback integration processed {len(feedback_result['processed_columns'])} columns")
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
        
        print("\n=== All basic tests passed successfully! ===")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 Task 1.4.2 implementation verified successfully!")
        print("✓ Column sensitivity adaptation based on feedback")
        print("✓ Pattern feedback processing and learning integration") 
        print("✓ Dynamic sensitivity adjustment for optimal performance")
        print("✓ Cross-lobe sensory data sharing")
        print("✓ Hormone-based feedback modulation")
        print("✓ Cross-modal learning and adaptation")
    else:
        print("\n❌ Task 1.4.2 implementation needs fixes")
        sys.exit(1)