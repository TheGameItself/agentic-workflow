#!/usr/bin/env python3
"""
Comprehensive tests for DreamingEngine with scenario simulation and contamination filtering.
"""

import pytest
import asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the DreamingEngine
import sys
sys.path.append('src')
from mcp.dreaming_engine import DreamingEngine, DreamScenario, DreamInsight


class TestDreamingEngineComprehensive:
    """Comprehensive tests for DreamingEngine upgrade features."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        memory_manager = Mock()
        memory_manager.search_memories.return_value = [
            {'content': 'test memory 1', 'relevance': 0.8},
            {'content': 'test memory 2', 'relevance': 0.6}
        ]
        memory_manager.add_memory.return_value = True
        return memory_manager
    
    @pytest.fixture
    def dreaming_engine(self, temp_db, mock_memory_manager):
        """Create a DreamingEngine instance for testing."""
        engine = DreamingEngine(db_path=temp_db, memory_manager=mock_memory_manager)
        # Add hormone integration mock
        engine.hormone_integration = Mock()
        engine.hormone_integration.emit_event = Mock()
        return engine
    
    def test_dream_cycle_initiation_natural(self, dreaming_engine):
        """Test natural dream cycle initiation."""
        trigger_context = {
            'recent_activity': 'problem_solving',
            'stress_level': 0.6,
            'creativity_need': 0.8
        }
        
        result = dreaming_engine.initiate_dream_cycle(
            trigger_context=trigger_context,
            cycle_type="natural"
        )
        
        assert result['status'] == 'completed'
        assert result['cycle_type'] == 'natural'
        assert 'cycle_id' in result
        assert 'total_insights' in result
        assert 'contamination_filtered' in result
        assert result['cycle_quality'] >= 0.0
        
        # Verify hormone integration was called
        dreaming_engine.hormone_integration.emit_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_scenario_simulation_comprehensive(self, dreaming_engine):
        """Test comprehensive scenario simulation."""
        context = "Solving a complex technical problem with limited resources"
        
        result = await dreaming_engine.simulate_dream(
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
    
    def test_contamination_filtering_basic(self, dreaming_engine):
        """Test basic contamination filtering functionality."""
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
        
        filtered_result = dreaming_engine._filter_dream_contamination(consolidation_result)
        
        assert 'safe_insights' in filtered_result
        assert 'contaminated_insights' in filtered_result
        assert 'safety_score' in filtered_result
        
        # Should filter out fantastical content
        contaminated_insights = filtered_result['contaminated_insights']
        contaminated_contents = [insight['content'] for insight in contaminated_insights]
        assert any('unicorn' in content.lower() for content in contaminated_contents)
    
    def test_memory_protection_mechanisms(self, dreaming_engine):
        """Test memory protection mechanisms during dream integration."""
        safe_insights = [
            {
                'content': 'Regular code reviews improve quality',
                'type': 'optimization',
                'confidence': 0.8,
                'applicability': ['development', 'quality_assurance']
            }
        ]
        
        filtered_result = {
            'safe_insights': safe_insights,
            'safe_meta_insights': [],
            'safety_score': 1.0
        }
        
        integration_result = dreaming_engine._integrate_safe_insights(filtered_result)
        
        assert 'total_integrated' in integration_result
        assert 'integration_success_rate' in integration_result
        assert 'memory_updates' in integration_result
    
    def test_insight_safety_classification(self, dreaming_engine):
        """Test insight safety classification logic."""
        # Test safe insight
        safe_insight = {
            'content': 'Implementing automated testing reduces bugs',
            'type': 'optimization',
            'confidence': 0.8
        }
        assert dreaming_engine._is_insight_safe(safe_insight) == True
        
        # Test contaminated insight
        contaminated_insight = {'content': 'Flying carpets provide fast transportation'}
        assert dreaming_engine._is_insight_safe(contaminated_insight) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])