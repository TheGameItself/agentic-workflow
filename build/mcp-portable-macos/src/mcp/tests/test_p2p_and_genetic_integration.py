"""
Integration tests for P2P status visualization and genetic expression architecture.
"""

import pytest
import asyncio
import sys
import os
import time
from unittest.mock import Mock, AsyncMock

# Add src root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mcp.p2p_status_visualization import (
    P2PStatusVisualizer, 
    ReputationScorer, 
    UserStatus, 
    UserSegment, 
    StatusBarData,
    P2PStatusBarRenderer
)
from mcp.genetic_expression_architecture import (
    GeneticExpressionArchitecture,
    InterruptionType,
    CircuitLayout,
    InterruptionPoint,
    AlignmentHook,
    UniversalHook,
    ReputationScore,
    BStarNode,
    ConditionChain
)


class TestP2PStatusVisualization:
    """Test P2P status visualization system"""
    
    @pytest.fixture
    def mock_p2p_system(self):
        """Create mock P2P system for testing"""
        mock_system = Mock()
        mock_system.get_system_status.return_value = {
            'connected_peers': 10,
            'peer_data': {
                f'peer_{i}': {
                    'uptime': 3600 + i * 100,
                    'successful_transfers': 50 + i * 10,
                    'total_transfers': 60 + i * 10,
                    'avg_response_time': 1000 + i * 50,
                    'data_quality_score': 0.8 + (i % 3) * 0.1,
                    'network_contribution_score': 0.7 + (i % 2) * 0.2,
                    'last_seen': time.time() - (i * 30)
                }
                for i in range(10)
            },
            'uptime': 7200,
            'genetic_diversity': 0.75,
            'system_metrics': {'network_fitness': 0.8}
        }
        return mock_system
    
    @pytest.fixture
    def visualizer(self, mock_p2p_system):
        """Create P2P status visualizer instance"""
        return P2PStatusVisualizer(mock_p2p_system)
    
    def test_reputation_scorer(self):
        """Test reputation scoring functionality"""
        scorer = ReputationScorer()
        
        # Test with valid user data
        user_data = {
            'uptime': 86400,  # 24 hours
            'successful_transfers': 100,
            'total_transfers': 110,
            'avg_response_time': 1000,
            'data_quality_score': 0.9,
            'network_contribution_score': 0.8
        }
        
        reputation = scorer.calculate_user_reputation(user_data)
        assert 0.0 <= reputation <= 1.0
        assert reputation > 0.5  # Should be high for good data
        
        # Test capabilities assessment
        capabilities = scorer.assess_user_capabilities(user_data)
        assert isinstance(capabilities, list)
        assert 'high_reputation_server' in capabilities
        assert 'fast_response' in capabilities
    
    @pytest.mark.asyncio
    async def test_status_update(self, visualizer):
        """Test status update functionality"""
        await visualizer.update_status()
        
        assert visualizer.current_status is not None
        assert visualizer.current_status.total_users > 0
        assert len(visualizer.current_status.segments) > 0
        
        # Check that segments have valid data
        for segment in visualizer.current_status.segments:
            assert segment.count >= 0
            assert 0.0 <= segment.percentage <= 100.0
            assert isinstance(segment.tooltip, str)
    
    def test_status_bar_rendering(self, visualizer):
        """Test status bar rendering"""
        # Create test segments
        segments = [
            UserSegment(
                status=UserStatus.IDLE,
                count=5,
                percentage=50.0,
                tooltip="5 idle users",
                reputation_score=0.5
            ),
            UserSegment(
                status=UserStatus.HIGH_REPUTATION,
                count=3,
                percentage=30.0,
                tooltip="3 high-reputation servers",
                reputation_score=0.9
            ),
            UserSegment(
                status=UserStatus.ACTIVE,
                count=2,
                percentage=20.0,
                tooltip="2 active users",
                reputation_score=0.6
            )
        ]
        
        # Test ASCII rendering
        ascii_bar = P2PStatusBarRenderer.render_ascii_bar(segments, width=20)
        assert isinstance(ascii_bar, str)
        assert len(ascii_bar) > 0
        
        # Test HTML rendering
        html_bar = P2PStatusBarRenderer.render_html_bar(segments)
        assert isinstance(html_bar, str)
        assert "p2p-status-bar" in html_bar
        
        # Test JSON rendering
        status_data = StatusBarData(
            segments=segments,
            total_users=10,
            last_update=time.time(),
            network_health=0.8,
            average_reputation=0.7
        )
        json_status = P2PStatusBarRenderer.render_json_status(status_data)
        assert isinstance(json_status, dict)
        assert "segments" in json_status
        assert "total_users" in json_status
    
    def test_network_health_calculation(self, visualizer):
        """Test network health calculation"""
        system_status = {
            'uptime': 86400,  # 24 hours
            'connected_peers': 15,
            'genetic_diversity': 0.8,
            'system_metrics': {'network_fitness': 0.9}
        }
        
        health = visualizer._calculate_network_health(system_status)
        assert 0.0 <= health <= 1.0
        assert health > 0.5  # Should be high for good metrics
    
    @pytest.mark.asyncio
    async def test_real_time_updates(self, visualizer):
        """Test real-time status updates"""
        updates = await visualizer.get_real_time_updates()
        
        assert isinstance(updates, dict)
        assert "status_bar" in updates
        assert "segments" in updates
        assert "network_health" in updates


class TestGeneticExpressionArchitecture:
    """Test genetic expression architecture system"""
    
    @pytest.fixture
    def architecture(self):
        """Create genetic expression architecture instance"""
        return GeneticExpressionArchitecture()
    
    def test_interruption_points(self, architecture):
        """Test interruption point functionality"""
        def test_condition(context):
            return context.get('test_value', 0) > 5
        
        def test_handler(context):
            return {"action": "test_action"}
        
        # Add interruption point
        success = architecture.add_interruption_point(
            "test_point",
            InterruptionType.CONDITIONAL,
            test_condition,
            test_handler
        )
        
        assert success
        assert "test_point" in architecture.interruption_points
        
        # Test duplicate addition
        success = architecture.add_interruption_point(
            "test_point",
            InterruptionType.CONDITIONAL,
            test_condition,
            test_handler
        )
        assert not success  # Should fail for duplicate
    
    def test_alignment_hooks(self, architecture):
        """Test alignment hook functionality"""
        def test_trigger(context):
            return context.get('trigger_value', False)
        
        # Add alignment hook
        success = architecture.add_alignment_hook(
            "test_hook",
            "source_lobe",
            ["target_lobe1", "target_lobe2"],
            test_trigger,
            {"sync_data": "test"}
        )
        
        assert success
        assert "test_hook" in architecture.alignment_hooks
        
        hook = architecture.alignment_hooks["test_hook"]
        assert hook.source_lobe == "source_lobe"
        assert len(hook.target_lobes) == 2
    
    def test_universal_hooks(self, architecture):
        """Test universal hook functionality"""
        def test_callback(context):
            return {"result": "test"}
        
        def test_validation(context):
            return context.get('valid', False)
        
        # Add universal hook
        success = architecture.add_universal_hook(
            "test_universal",
            "test_type",
            "test_integration",
            test_callback,
            test_validation
        )
        
        assert success
        assert "test_universal" in architecture.universal_hooks
        
        hook = architecture.universal_hooks["test_universal"]
        assert hook.hook_type == "test_type"
        assert hook.integration_point == "test_integration"
    
    def test_b_star_search(self, architecture):
        """Test B* search functionality"""
        test_circuit = {
            "nodes": {
                "start": {"type": "input"},
                "process1": {"type": "computation"},
                "process2": {"type": "computation"},
                "end": {"type": "output"}
            },
            "edges": [
                {"source": "start", "target": "process1"},
                {"source": "process1", "target": "process2"},
                {"source": "process2", "target": "end"}
            ]
        }
        
        # Test B* search
        path = architecture.scaffold_b_star_search(test_circuit, "start", "end")
        
        assert isinstance(path, list)
        assert len(path) >= 2
        assert path[0] == "start"
        assert path[-1] == "end"
        
        # Test with invalid circuit
        invalid_circuit = {"nodes": {}, "edges": []}
        path = architecture.scaffold_b_star_search(invalid_circuit, "start", "end")
        assert path == ["start", "end"]
    
    def test_condition_chains(self, architecture):
        """Test condition chain functionality"""
        def condition1(context):
            return context.get('value1', 0) > 10
        
        def condition2(context):
            return context.get('value2', '') == 'test'
        
        def fallback_condition(context):
            return context.get('fallback', False)
        
        # Create condition chain
        success = architecture.scaffold_condition_chain(
            "test_chain",
            [condition1, condition2],
            {"1": [], "2": ["1"]},
            ["1", "2"],
            [fallback_condition]
        )
        
        assert success
        assert "test_chain" in architecture.condition_chains
    
    @pytest.mark.asyncio
    async def test_condition_evaluation(self, architecture):
        """Test condition chain evaluation"""
        def condition1(context):
            return context.get('value1', 0) > 10
        
        def condition2(context):
            return context.get('value2', '') == 'test'
        
        # Create condition chain
        architecture.scaffold_condition_chain(
            "test_eval_chain",
            [condition1, condition2],
            {"1": [], "2": ["1"]},
            ["1", "2"]
        )
        
        # Test successful evaluation
        result = await architecture.evaluate_condition_chain("test_eval_chain", {
            "value1": 15,
            "value2": "test"
        })
        
        assert result["success"] is True
        assert "results" in result
        assert "1" in result["results"]
        assert "2" in result["results"]
        
        # Test failed evaluation
        result = await architecture.evaluate_condition_chain("test_eval_chain", {
            "value1": 5,
            "value2": "wrong"
        })
        
        assert result["success"] is False
    
    def test_reputation_scoring(self, architecture):
        """Test reputation scoring functionality"""
        # Update reputation score
        success = architecture.update_reputation_score(
            "test_sequence",
            performance_score=0.9,
            reliability_score=0.8,
            efficiency_score=0.7,
            adaptability_score=0.6
        )
        
        assert success
        
        # Get reputation score
        score = architecture.get_reputation_score("test_sequence")
        assert score is not None
        assert score.sequence_id == "test_sequence"
        assert score.overall_score > 0.7  # Should be high for good scores
        assert score.evaluation_count == 1
        
        # Update again to test evaluation count
        architecture.update_reputation_score(
            "test_sequence",
            performance_score=0.8,
            reliability_score=0.7,
            efficiency_score=0.6,
            adaptability_score=0.5
        )
        
        score = architecture.get_reputation_score("test_sequence")
        assert score.evaluation_count == 2
    
    def test_top_performing_sequences(self, architecture):
        """Test top performing sequences retrieval"""
        # Add multiple sequences
        sequences = [
            ("seq1", 0.9, 0.8, 0.7, 0.6),
            ("seq2", 0.8, 0.9, 0.8, 0.7),
            ("seq3", 0.7, 0.6, 0.9, 0.8)
        ]
        
        for seq_id, perf, rel, eff, adapt in sequences:
            architecture.update_reputation_score(seq_id, perf, rel, eff, adapt)
        
        # Get top performing sequences
        top_sequences = architecture.get_top_performing_sequences(limit=2)
        
        assert len(top_sequences) == 2
        assert top_sequences[0].overall_score >= top_sequences[1].overall_score
    
    @pytest.mark.asyncio
    async def test_execution_with_interruptions(self, architecture):
        """Test execution with interruption points"""
        def test_condition(context):
            return context.get('interrupt', False)
        
        def test_handler(context):
            return {"interrupted": True}
        
        def test_function(context):
            return {"result": "success"}
        
        # Add interruption point
        architecture.add_interruption_point(
            "test_interrupt",
            InterruptionType.CONDITIONAL,
            test_condition,
            test_handler
        )
        
        # Test execution without interruption
        result = await architecture.execute_with_interruptions(
            test_function,
            {"interrupt": False}
        )
        
        assert result["success"] is True
        assert result["result"]["result"] == "success"
        assert len(result["interruptions_triggered"]) == 0
        
        # Test execution with interruption
        result = await architecture.execute_with_interruptions(
            test_function,
            {"interrupt": True}
        )
        
        assert result["success"] is True
        assert len(result["interruptions_triggered"]) == 1
        assert result["interruptions_triggered"][0]["point_id"] == "test_interrupt"
    
    def test_performance_metrics(self, architecture):
        """Test performance metrics collection"""
        # Add some test data
        architecture.add_interruption_point(
            "test_point",
            InterruptionType.CONDITIONAL,
            lambda x: True,
            lambda x: x
        )
        
        architecture.add_alignment_hook(
            "test_hook",
            "source",
            ["target"],
            lambda x: True,
            {}
        )
        
        architecture.update_reputation_score(
            "test_seq",
            0.8, 0.7, 0.6, 0.5
        )
        
        # Get performance metrics
        metrics = architecture.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "interruption_points_count" in metrics
        assert "alignment_hooks_count" in metrics
        assert "reputation_scores_count" in metrics
        assert metrics["interruption_points_count"] == 1
        assert metrics["alignment_hooks_count"] == 1
        assert metrics["reputation_scores_count"] == 1


class TestP2PGeneticIntegration:
    """Test integration between P2P and genetic systems"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated P2P and genetic system"""
        mock_p2p = Mock()
        mock_p2p.get_system_status.return_value = {
            'connected_peers': 5,
            'peer_data': {
                f'peer_{i}': {
                    'uptime': 3600,
                    'successful_transfers': 100,
                    'total_transfers': 110,
                    'avg_response_time': 1000,
                    'data_quality_score': 0.9,
                    'network_contribution_score': 0.8,
                    'last_seen': time.time()
                }
                for i in range(5)
            },
            'uptime': 7200,
            'genetic_diversity': 0.8,
            'system_metrics': {'network_fitness': 0.9}
        }
        
        visualizer = P2PStatusVisualizer(mock_p2p)
        architecture = GeneticExpressionArchitecture()
        
        return {
            'visualizer': visualizer,
            'architecture': architecture,
            'p2p_system': mock_p2p
        }
    
    @pytest.mark.asyncio
    async def test_integrated_status_and_genetics(self, integrated_system):
        """Test integrated status visualization and genetic architecture"""
        visualizer = integrated_system['visualizer']
        architecture = integrated_system['architecture']
        
        # Update P2P status
        await visualizer.update_status()
        
        # Add genetic interruption point based on P2P status
        def p2p_condition(context):
            if visualizer.current_status:
                return visualizer.current_status.network_health < 0.5
            return False
        
        def p2p_handler(context):
            return {"action": "optimize_network", "reason": "low_health"}
        
        architecture.add_interruption_point(
            "p2p_health_check",
            InterruptionType.PERFORMANCE,
            p2p_condition,
            p2p_handler
        )
        
        # Test execution with P2P-aware interruptions
        def test_function(context):
            return {"status": "executed"}
        
        result = await architecture.execute_with_interruptions(
            test_function,
            {"test": "data"}
        )
        
        assert result["success"] is True
        assert "interruptions_triggered" in result
    
    def test_genetic_reputation_in_p2p_context(self, integrated_system):
        """Test genetic reputation scoring in P2P context"""
        architecture = integrated_system['architecture']
        
        # Simulate genetic sequences from P2P peers
        peer_sequences = [
            ("peer_0_sequence", 0.9, 0.8, 0.7, 0.6),
            ("peer_1_sequence", 0.8, 0.9, 0.8, 0.7),
            ("peer_2_sequence", 0.7, 0.6, 0.9, 0.8)
        ]
        
        for seq_id, perf, rel, eff, adapt in peer_sequences:
            architecture.update_reputation_score(seq_id, perf, rel, eff, adapt)
        
        # Get top performing sequences (which could be from P2P peers)
        top_sequences = architecture.get_top_performing_sequences(limit=3)
        
        assert len(top_sequences) == 3
        assert all("peer" in seq.sequence_id for seq in top_sequences)
        
        # Verify reputation scores are properly calculated
        for seq in top_sequences:
            assert 0.0 <= seq.overall_score <= 1.0
            assert seq.evaluation_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 