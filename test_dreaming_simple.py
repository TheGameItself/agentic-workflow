#!/usr/bin/env python3
"""
Simple test for DreamingEngine functionality without pytest dependency.
"""

import asyncio
import tempfile
import os
import sys
from unittest.mock import Mock

# Import the DreamingEngine
sys.path.append('src')
from mcp.dreaming_engine import DreamingEngine


def test_dreaming_engine():
    """Test basic DreamingEngine functionality."""
    print("Testing DreamingEngine comprehensive upgrade...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Create mock memory manager
        memory_manager = Mock()
        memory_manager.search_memories.return_value = [
            {'content': 'test memory 1', 'relevance': 0.8},
            {'content': 'test memory 2', 'relevance': 0.6}
        ]
        memory_manager.add_memory.return_value = True
        
        # Create DreamingEngine
        engine = DreamingEngine(db_path=db_path, memory_manager=memory_manager)
        engine.hormone_integration = Mock()
        engine.hormone_integration.emit_event = Mock()
        
        print("✓ DreamingEngine initialized successfully")
        
        # Test 1: Dream cycle initiation
        print("\nTest 1: Dream cycle initiation")
        trigger_context = {
            'recent_activity': 'problem_solving',
            'stress_level': 0.6,
            'creativity_need': 0.8
        }
        
        result = engine.initiate_dream_cycle(
            trigger_context=trigger_context,
            cycle_type="natural"
        )
        
        assert result['status'] == 'completed'
        assert result['cycle_type'] == 'natural'
        assert 'cycle_id' in result
        assert 'total_insights' in result
        assert 'contamination_filtered' in result
        assert result['cycle_quality'] >= 0.0
        print("✓ Dream cycle initiation works correctly")
        
        # Test 2: Scenario simulation
        print("\nTest 2: Scenario simulation")
        async def test_simulation():
            context = "Solving a complex technical problem with limited resources"
            
            result = await engine.simulate_dream(
                context=context,
                dream_type="problem_solving",
                simulation_data={'complexity': 'high', 'urgency': 'medium'}
            )
            
            assert result['scenario_id'] is not None
            assert result['dream_type'] == 'problem_solving'
            assert result['context'] == context
            assert 'simulation_result' in result
            assert 'insights' in result
            assert 'quality_score' in result
            assert 'learning_value' in result
            assert 'recommendations' in result
            return result
        
        sim_result = asyncio.run(test_simulation())
        print("✓ Scenario simulation works correctly")
        
        # Test 3: Contamination filtering
        print("\nTest 3: Contamination filtering")
        consolidation_result = {
            'insights': [
                {
                    'content': 'Use systematic approach to problem solving',
                    'type': 'problem_solution',
                    'confidence': 0.8
                },
                {
                    'content': 'Flying unicorns can solve all problems',
                    'type': 'creative_idea',
                    'confidence': 0.9
                }
            ],
            'meta_insights': []
        }
        
        filtered_result = engine._filter_dream_contamination(consolidation_result)
        
        assert 'safe_insights' in filtered_result
        assert 'contaminated_insights' in filtered_result
        assert 'safety_score' in filtered_result
        
        # Should filter out fantastical content
        contaminated_insights = filtered_result['contaminated_insights']
        contaminated_contents = [insight['content'] for insight in contaminated_insights]
        assert any('unicorn' in content.lower() for content in contaminated_contents)
        print("✓ Contamination filtering works correctly")
        
        # Test 4: Memory protection
        print("\nTest 4: Memory protection mechanisms")
        safe_insights = [
            {
                'content': 'Regular code reviews improve quality',
                'type': 'optimization',
                'confidence': 0.8,
                'applicability': ['development', 'quality_assurance']
            }
        ]
        
        filtered_result_safe = {
            'safe_insights': safe_insights,
            'safe_meta_insights': [],
            'safety_score': 1.0
        }
        
        integration_result = engine._integrate_safe_insights(filtered_result_safe)
        
        assert 'total_integrated' in integration_result
        assert 'integration_success_rate' in integration_result
        assert 'memory_updates' in integration_result
        print("✓ Memory protection mechanisms work correctly")
        
        # Test 5: Insight safety classification
        print("\nTest 5: Insight safety classification")
        safe_insight = {
            'content': 'Implementing automated testing reduces bugs',
            'type': 'optimization',
            'confidence': 0.8
        }
        assert engine._is_insight_safe(safe_insight) == True
        
        contaminated_insight = {'content': 'Flying carpets provide fast transportation'}
        assert engine._is_insight_safe(contaminated_insight) == False
        print("✓ Insight safety classification works correctly")
        
        # Test 6: Dream statistics
        print("\nTest 6: Dream statistics")
        stats = engine.get_dream_statistics()
        assert 'total_scenarios' in stats
        assert 'total_insights' in stats
        assert 'average_quality_score' in stats
        assert 'insights_by_type' in stats
        print("✓ Dream statistics work correctly")
        
        print("\n🎉 All DreamingEngine tests passed successfully!")
        print(f"   - Dream cycles initiated: {stats['total_scenarios']}")
        print(f"   - Insights generated: {stats['total_insights']}")
        print(f"   - Average quality score: {stats['average_quality_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    success = test_dreaming_engine()
    if success:
        print("\n✅ DreamingEngine comprehensive upgrade implementation is complete and working!")
    else:
        print("\n❌ DreamingEngine tests failed!")
        sys.exit(1)