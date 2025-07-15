#!/usr/bin/env python3
"""
Test script for experimental lobes.
This script tests the basic functionality of the experimental lobes.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp.experimental_lobes import (
    AlignmentEngine, 
    PatternRecognitionEngine, 
    SimulatedReality, 
    DreamingEngine,
    MindMapEngine,
    ScientificProcessEngine,
    SpeculationEngine,
    SplitBrainABTest,
    MultiLLMOrchestrator,
    AdvancedEngramEngine
)

def test_alignment_engine():
    """Test the AlignmentEngine."""
    print("Testing AlignmentEngine...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = AlignmentEngine(db_path)
        
        # Test basic alignment
        original = "This is a very long text that should be made more concise according to user preferences."
        aligned = engine.align(original, {"preference": "concise", "max_words": 10})
        print(f"Original: {original}")
        print(f"Aligned: {aligned}")
        
        # Test style alignment
        formal_text = "I can't do this, it's too hard."
        formal_aligned = engine.align(formal_text, {"style": "formal"})
        print(f"Formal: {formal_aligned}")
        
        # Test statistics
        stats = engine.get_alignment_statistics()
        print(f"Alignment stats: {stats}")
        
        print("✓ AlignmentEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ AlignmentEngine test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_pattern_recognition_engine():
    """Test the PatternRecognitionEngine."""
    print("Testing PatternRecognitionEngine...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = PatternRecognitionEngine(db_path)
        
        # Test pattern recognition
        data_batch = [
            "This is a test string with some words.",
            {"key1": "value1", "key2": 42, "key3": [1, 2, 3]},
            [1, 2, 3, 4, 5],
            42.5
        ]
        
        patterns = engine.recognize_patterns(data_batch)
        print(f"Recognized {len(patterns)} patterns")
        
        # Test neural column simulation
        columns = engine.simulate_neural_columns(data_batch)
        print(f"Simulated {len(columns)} neural columns")
        
        # Test proactive prompting
        prompts = engine.proactive_prompt(data_batch)
        print(f"Generated {len(prompts)} proactive prompts")
        
        # Test statistics
        stats = engine.get_statistics()
        print(f"Pattern recognition stats: {stats}")
        
        print("✓ PatternRecognitionEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ PatternRecognitionEngine test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_simulated_reality():
    """Test the SimulatedReality engine."""
    print("Testing SimulatedReality...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        reality = SimulatedReality(db_path)
        
        # Test entity management
        entity_id = reality.add_entity("test_entity", {"attr1": "value1", "attr2": 42})
        print(f"Added entity with ID: {entity_id}")
        
        # Test event management
        event_id = reality.add_event("test_event", datetime.now().isoformat(), ["test_entity"])
        print(f"Added event with ID: {event_id}")
        
        # Test state management
        state_id = reality.add_state("test_state", 42, datetime.now().isoformat())
        print(f"Added state with ID: {state_id}")
        
        # Test queries
        entities = reality.query_entities()
        print(f"Found {len(entities)} entities")
        
        events = reality.query_events()
        print(f"Found {len(events)} events")
        
        states = reality.query_states()
        print(f"Found {len(states)} states")
        
        print("✓ SimulatedReality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SimulatedReality test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_dreaming_engine():
    """Test the DreamingEngine."""
    print("Testing DreamingEngine...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = DreamingEngine(db_path)
        
        # Test dream simulation
        dream = engine.simulate_dream("testing context", "scenario")
        print(f"Generated dream: {dream.get('scenario', 'No scenario')[:100]}...")
        
        # Test learning from dreams
        insights = engine.learn_from_dreams()
        print(f"Learning insights: {insights}")
        
        # Test statistics
        stats = engine.get_dream_statistics()
        print(f"Dream statistics: {stats}")
        
        print("✓ DreamingEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ DreamingEngine test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_mind_map_engine():
    """Test the MindMapEngine."""
    print("Testing MindMapEngine...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = MindMapEngine(db_path)
        
        # Test node creation
        node1_id = engine.add_node("concept1", {"description": "First concept"})
        node2_id = engine.add_node("concept2", {"description": "Second concept"})
        print(f"Added nodes: {node1_id}, {node2_id}")
        
        # Test edge creation
        edge_id = engine.add_edge("concept1", "concept2", "association")
        print(f"Added edge: {edge_id}")
        
        # Test export
        mind_map = engine.export_mind_map("json")
        print(f"Exported mind map: {len(mind_map)} characters")
        
        # Test statistics
        stats = engine.get_statistics()
        print(f"Mind map statistics: {stats}")
        
        print("✓ MindMapEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ MindMapEngine test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_scientific_process_engine():
    """Test the ScientificProcessEngine."""
    print("Testing ScientificProcessEngine...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = ScientificProcessEngine(db_path)
        
        # Test hypothesis proposal
        hypothesis_id = engine.propose_hypothesis("Test hypothesis", 0.8)
        print(f"Proposed hypothesis with ID: {hypothesis_id}")
        
        # Test experiment recording
        experiment_id = engine.record_experiment(hypothesis_id, "success", "Test experiment")
        print(f"Recorded experiment with ID: {experiment_id}")
        
        # Test evidence addition
        evidence_id = engine.add_evidence(experiment_id, "Test evidence")
        print(f"Added evidence with ID: {evidence_id}")
        
        # Test summary
        summary = engine.summarize_scientific_process()
        print(f"Scientific process summary: {summary}")
        
        print("✓ ScientificProcessEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ ScientificProcessEngine test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_speculation_engine():
    """Test the SpeculationEngine."""
    print("Testing SpeculationEngine...")
    
    try:
        engine = SpeculationEngine()
        
        # Test speculation
        speculation = engine.speculate("test context", "test topic")
        print(f"Generated speculation: {speculation}")
        
        # Test evaluation
        evaluation = engine.evaluate_speculation(speculation, "test evidence")
        print(f"Evaluation: {evaluation}")
        
        # Test listing
        speculations = engine.list_speculations()
        print(f"Found {len(speculations)} speculations")
        
        print("✓ SpeculationEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SpeculationEngine test failed: {e}")
        return False

def test_split_brain_abtest():
    """Test the SplitBrainABTest."""
    print("Testing SplitBrainABTest...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create a dummy lobe class for testing
        class DummyLobe:
            def __init__(self, config=None):
                self.config = config or {}
            
            def process(self, x):
                return f"Processed {x} with config {self.config}"
        
        # Test AB testing
        abtest = SplitBrainABTest(DummyLobe, left_config={"left": "config"}, right_config={"right": "config"}, db_path=db_path)
        result = abtest.run_test("test_input")
        print(f"AB test result: {result}")
        
        # Test feedback
        abtest.provide_feedback("Good result")
        
        # Test results
        results = abtest.get_results()
        print(f"AB test results: {results}")
        
        print("✓ SplitBrainABTest tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SplitBrainABTest test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_multi_llm_orchestrator():
    """Test the MultiLLMOrchestrator."""
    print("Testing MultiLLMOrchestrator...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        orchestrator = MultiLLMOrchestrator(db_path)
        
        # Test query routing
        result = orchestrator.route_query("What is the capital of France?")
        print(f"Query routing result: {result}")
        
        # Test feedback
        if 'query_id' in result:
            orchestrator.provide_feedback(result['query_id'], 'local_small', 1.0, "Correct answer")
        
        # Test report
        report = orchestrator.report()
        print(f"Orchestrator report: {report}")
        
        print("✓ MultiLLMOrchestrator tests passed")
        return True
        
    except Exception as e:
        print(f"✗ MultiLLMOrchestrator test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_advanced_engram_engine():
    """Test the AdvancedEngramEngine."""
    print("Testing AdvancedEngramEngine...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        engine = AdvancedEngramEngine(db_path)
        
        # Test compression
        test_data = {"text": "test memory chunk", "number": 42, "list": [1, 2, 3]}
        compressed = engine.compress(test_data)
        print(f"Compressed data: {compressed}")
        
        # Test selection
        engine.select({"score": 0.8, "comment": "useful"})
        
        print("✓ AdvancedEngramEngine tests passed")
        return True
        
    except Exception as e:
        print(f"✗ AdvancedEngramEngine test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def main():
    """Run all lobe tests."""
    print("Starting experimental lobes tests...")
    print("=" * 50)
    
    tests = [
        test_alignment_engine,
        test_pattern_recognition_engine,
        test_simulated_reality,
        test_dreaming_engine,
        test_mind_map_engine,
        test_scientific_process_engine,
        test_speculation_engine,
        test_split_brain_abtest,
        test_multi_llm_orchestrator,
        test_advanced_engram_engine
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All experimental lobes tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 