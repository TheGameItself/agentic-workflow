#!/usr/bin/env python3
"""
Comprehensive test suite for new engines and functionality.

Tests the dreaming engine, engram engine, and scientific process engine
as well as their integration with the main MCP server.
"""

import unittest
import tempfile
import os
import json
import asyncio
from datetime import datetime
from src.mcp.dreaming_engine import DreamingEngine
from src.mcp.engram_engine import EngramEngine
from src.mcp.scientific_engine import ScientificProcessEngine
from src.mcp.memory import MemoryManager
from src.mcp.server import MCPServer


class TestNewEngines(unittest.TestCase):
    """Test suite for new engines."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_engines.db")
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(db_path=self.db_path)
        
        # Initialize engines
        self.dreaming_engine = DreamingEngine(memory_manager=self.memory_manager)
        self.engram_engine = EngramEngine(memory_manager=self.memory_manager)
        self.scientific_engine = ScientificProcessEngine(memory_manager=self.memory_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary files
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_dreaming_engine_initialization(self):
        """Test dreaming engine initialization."""
        self.assertIsNotNone(self.dreaming_engine)
        self.assertIsNotNone(self.dreaming_engine.db_path)
        self.assertIsNotNone(self.dreaming_engine.dream_types)
        self.assertIsNotNone(self.dreaming_engine.insight_types)
    
    def test_engram_engine_initialization(self):
        """Test engram engine initialization."""
        self.assertIsNotNone(self.engram_engine)
        self.assertIsNotNone(self.engram_engine.db_path)
        self.assertIsNotNone(self.engram_engine.compression_methods)
        self.assertIsNotNone(self.engram_engine.mutation_rate)
    
    def test_scientific_engine_initialization(self):
        """Test scientific process engine initialization."""
        self.assertIsNotNone(self.scientific_engine)
        self.assertIsNotNone(self.scientific_engine.db_path)
        self.assertIsNotNone(self.scientific_engine.hypothesis_categories)
        self.assertIsNotNone(self.scientific_engine.methodologies)
    
    async def test_dream_simulation(self):
        """Test dream simulation functionality."""
        context = "How to optimize a machine learning pipeline"
        dream_type = "problem_solving"
        
        result = await self.dreaming_engine.simulate_dream(context, dream_type)
        
        self.assertIsInstance(result, dict)
        self.assertIn('scenario_id', result)
        self.assertIn('dream_type', result)
        self.assertIn('context', result)
        self.assertIn('simulation_result', result)
        self.assertIn('insights', result)
        self.assertIn('quality_score', result)
        self.assertIn('learning_value', result)
        self.assertIn('recommendations', result)
        
        # Check that insights were generated
        self.assertGreater(len(result['insights']), 0)
        
        # Check quality score is reasonable
        self.assertGreaterEqual(result['quality_score'], 0.0)
        self.assertLessEqual(result['quality_score'], 1.0)
    
    def test_engram_creation(self):
        """Test engram creation functionality."""
        content = "Machine learning models require careful hyperparameter tuning"
        content_type = "text"
        tags = ["machine_learning", "optimization"]
        associations = []
        
        engram_id = self.engram_engine.create_engram(content, content_type, tags, associations)
        
        self.assertIsInstance(engram_id, str)
        self.assertGreater(len(engram_id), 0)
        
        # Test engram search
        results = self.engram_engine.search_engrams("machine learning", "semantic", 5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
    
    def test_engram_merging(self):
        """Test engram merging functionality."""
        # Create multiple engrams
        engram1_id = self.engram_engine.create_engram(
            "Neural networks are powerful for pattern recognition",
            "text",
            ["neural_networks", "pattern_recognition"]
        )
        
        engram2_id = self.engram_engine.create_engram(
            "Deep learning requires large datasets",
            "text",
            ["deep_learning", "datasets"]
        )
        
        # Merge engrams
        merged_id = self.engram_engine.merge_engrams([engram1_id, engram2_id], "diffusion")
        
        self.assertIsInstance(merged_id, str)
        self.assertGreater(len(merged_id), 0)
        self.assertNotEqual(merged_id, engram1_id)
        self.assertNotEqual(merged_id, engram2_id)
    
    def test_hypothesis_proposal(self):
        """Test hypothesis proposal functionality."""
        statement = "Using ensemble methods improves model performance by 15%"
        category = "causal"
        variables = ["ensemble_method", "model_performance"]
        assumptions = ["sufficient data available", "proper evaluation metrics"]
        confidence = 0.7
        
        hypothesis_id = self.scientific_engine.propose_hypothesis(
            statement, category, variables, assumptions, confidence
        )
        
        self.assertIsInstance(hypothesis_id, str)
        self.assertGreater(len(hypothesis_id), 0)
    
    def test_experiment_design(self):
        """Test experiment design functionality."""
        # First create a hypothesis
        hypothesis_id = self.scientific_engine.propose_hypothesis(
            "Feature engineering improves model accuracy",
            "causal",
            ["feature_engineering", "model_accuracy"]
        )
        
        # Design experiment
        experiment_id = self.scientific_engine.design_experiment(
            hypothesis_id,
            "randomized_control",
            sample_size=100,
            duration_days=14
        )
        
        self.assertIsInstance(experiment_id, str)
        self.assertGreater(len(experiment_id), 0)
    
    def test_experiment_execution(self):
        """Test experiment execution functionality."""
        # Create hypothesis and experiment
        hypothesis_id = self.scientific_engine.propose_hypothesis(
            "Data augmentation reduces overfitting",
            "causal",
            ["data_augmentation", "overfitting"]
        )
        
        experiment_id = self.scientific_engine.design_experiment(
            hypothesis_id,
            "randomized_control"
        )
        
        # Run experiment
        results = self.scientific_engine.run_experiment(experiment_id)
        
        self.assertIsInstance(results, dict)
        self.assertIn('experiment_id', results)
        self.assertIn('results', results)
        self.assertIn('statistical_significance', results)
        self.assertIn('conclusion', results)
    
    def test_hypothesis_analysis(self):
        """Test hypothesis analysis functionality."""
        # Create hypothesis
        hypothesis_id = self.scientific_engine.propose_hypothesis(
            "Regularization improves generalization",
            "causal",
            ["regularization", "generalization"]
        )
        
        # Design and run experiment
        experiment_id = self.scientific_engine.design_experiment(hypothesis_id)
        self.scientific_engine.run_experiment(experiment_id)
        
        # Analyze hypothesis
        analysis = self.scientific_engine.analyze_hypothesis(hypothesis_id)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('hypothesis_id', analysis)
        self.assertIn('evidence_strength', analysis)
        self.assertIn('status', analysis)
        self.assertIn('meta_analysis', analysis)
        self.assertIn('conclusion', analysis)
    
    def test_dreaming_statistics(self):
        """Test dreaming engine statistics."""
        # Simulate some dreams first
        asyncio.run(self.dreaming_engine.simulate_dream("Test context 1", "problem_solving"))
        asyncio.run(self.dreaming_engine.simulate_dream("Test context 2", "creative_exploration"))
        
        # Get statistics
        stats = self.dreaming_engine.get_dream_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_scenarios', stats)
        self.assertIn('total_insights', stats)
        self.assertIn('average_quality_score', stats)
        self.assertIn('average_learning_value', stats)
        self.assertIn('insights_by_type', stats)
        self.assertIn('dreams_by_type', stats)
    
    def test_engram_statistics(self):
        """Test engram engine statistics."""
        # Create some engrams first
        self.engram_engine.create_engram("Test content 1", "text", ["test"])
        self.engram_engine.create_engram("Test content 2", "text", ["test"])
        
        # Get statistics
        stats = self.engram_engine.get_engram_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_engrams', stats)
        self.assertIn('average_quality_score', stats)
        self.assertIn('average_compression_ratio', stats)
        self.assertIn('total_accesses', stats)
        self.assertIn('content_type_distribution', stats)
        self.assertIn('quality_distribution', stats)
    
    def test_scientific_summary(self):
        """Test scientific process engine summary."""
        # Create hypothesis and run experiment
        hypothesis_id = self.scientific_engine.propose_hypothesis(
            "Cross-validation improves model selection",
            "causal",
            ["cross_validation", "model_selection"]
        )
        
        experiment_id = self.scientific_engine.design_experiment(hypothesis_id)
        self.scientific_engine.run_experiment(experiment_id)
        
        # Get summary
        summary = self.scientific_engine.get_scientific_summary(hypothesis_id)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('hypothesis', summary)
        self.assertIn('experiments', summary)
        self.assertIn('evidence', summary)
        self.assertIn('conclusions', summary)
        self.assertIn('recommendations', summary)
    
    def test_engram_evolution(self):
        """Test engram evolution functionality."""
        # Create initial engram
        engram_id = self.engram_engine.create_engram(
            "Initial content for evolution",
            "text",
            ["evolution_test"]
        )
        
        # Evolve engram
        evolved_id = self.engram_engine.evolve_engram(engram_id, "mutation")
        
        self.assertIsInstance(evolved_id, str)
        self.assertGreater(len(evolved_id), 0)
        self.assertNotEqual(evolved_id, engram_id)
    
    def test_engram_feedback(self):
        """Test engram feedback functionality."""
        # Create engram
        engram_id = self.engram_engine.create_engram(
            "Content for feedback testing",
            "text",
            ["feedback_test"]
        )
        
        # Provide feedback
        success = self.engram_engine.provide_feedback(
            engram_id,
            feedback_score=0.8,
            feedback_text="Good quality content",
            feedback_type="user"
        )
        
        self.assertTrue(success)
    
    def test_dreaming_feedback(self):
        """Test dreaming engine feedback functionality."""
        # Simulate dream
        result = asyncio.run(self.dreaming_engine.simulate_dream("Feedback test context"))
        scenario_id = result['scenario_id']
        
        # Provide feedback
        success = self.dreaming_engine.provide_feedback(
            scenario_id=scenario_id,
            feedback_score=0.9,
            feedback_text="Very insightful dream"
        )
        
        self.assertTrue(success)
    
    def test_learning_insights(self):
        """Test learning insights from dreaming."""
        # Simulate multiple dreams
        asyncio.run(self.dreaming_engine.simulate_dream("Learning context 1"))
        asyncio.run(self.dreaming_engine.simulate_dream("Learning context 2"))
        
        # Get learning insights
        insights = self.dreaming_engine.get_learning_insights(limit=10)
        
        self.assertIsInstance(insights, list)
        # May be empty if no high-confidence insights were generated
        if insights:
            self.assertIn('id', insights[0])
            self.assertIn('content', insights[0])
            self.assertIn('insight_type', insights[0])
            self.assertIn('confidence', insights[0])


class TestMCPServerIntegration(unittest.TestCase):
    """Test MCP server integration with new engines."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.server = MCPServer(project_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    async def test_dream_simulation_endpoint(self):
        """Test dream simulation endpoint."""
        params = {
            "context": "How to improve code quality",
            "dream_type": "problem_solving",
            "simulation_data": {"complexity": "high"}
        }
        
        result = await self.server._handle_simulate_dream(params)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("dream_result", result)
    
    async def test_engram_creation_endpoint(self):
        """Test engram creation endpoint."""
        params = {
            "content": "Test engram content",
            "content_type": "text",
            "tags": ["test", "engram"],
            "associations": []
        }
        
        result = await self.server._handle_create_engram(params)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("engram_id", result)
    
    async def test_hypothesis_proposal_endpoint(self):
        """Test hypothesis proposal endpoint."""
        params = {
            "statement": "Test hypothesis statement",
            "category": "causal",
            "variables": ["var1", "var2"],
            "assumptions": ["assumption1"],
            "confidence": 0.6
        }
        
        result = await self.server._handle_propose_hypothesis(params)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("hypothesis_id", result)
    
    async def test_experiment_design_endpoint(self):
        """Test experiment design endpoint."""
        # First create a hypothesis
        hypothesis_params = {
            "statement": "Test experiment hypothesis",
            "category": "causal",
            "variables": ["var1"],
            "confidence": 0.5
        }
        hypothesis_result = await self.server._handle_propose_hypothesis(hypothesis_params)
        hypothesis_id = hypothesis_result["hypothesis_id"]
        
        # Design experiment
        params = {
            "hypothesis_id": hypothesis_id,
            "methodology": "randomized_control",
            "sample_size": 50,
            "duration_days": 7
        }
        
        result = await self.server._handle_design_experiment(params)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("experiment_id", result)


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestNewEngines))
    suite.addTest(unittest.makeSuite(TestMCPServerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!") 