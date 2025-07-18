#!/usr/bin/env python3
"""
Standalone test for the ScientificProcessEngine.

This test file is designed to run independently of the main test suite
to verify the functionality of the enhanced ScientificProcessEngine.
"""

import os
import sys
import unittest
import tempfile
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the WorkingMemory classes to avoid dependencies
class MockWorkingMemory:
    def __init__(self):
        self.data = []
    
    def add(self, item):
        self.data.append(item)
        return True

class MockShortTermMemory(MockWorkingMemory):
    pass

class MockLongTermMemory(MockWorkingMemory):
    pass

# Mock the module imports
sys.modules['src.mcp.lobes.shared_lobes.working_memory'] = MagicMock()
sys.modules['src.mcp.lobes.shared_lobes.working_memory'].WorkingMemory = MockWorkingMemory
sys.modules['src.mcp.lobes.shared_lobes.working_memory'].ShortTermMemory = MockShortTermMemory
sys.modules['src.mcp.lobes.shared_lobes.working_memory'].LongTermMemory = MockLongTermMemory

# Import the ScientificProcessEngine class
from src.mcp.lobes.experimental.scientific_process.scientific_process_engine import (
    ScientificProcessEngine, Observation, Hypothesis, ExperimentDesign, 
    Experiment, ExperimentResult, Analysis, Evidence, ValidationResult, Finding
)


class TestScientificProcessEngine(unittest.TestCase):
    """Test cases for the enhanced ScientificProcessEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_scientific_process.db")
        self.engine = ScientificProcessEngine(db_path=self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_propose_hypothesis(self):
        """Test proposing a hypothesis."""
        hypothesis_id = self.engine.propose_hypothesis("The sky is blue.")
        self.assertIsNotNone(hypothesis_id)
        
        # Verify hypothesis was stored
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT statement FROM hypotheses WHERE id = ?", (hypothesis_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "The sky is blue.")

    def test_design_experiment(self):
        """Test designing an experiment."""
        hypothesis_id = self.engine.propose_hypothesis("The sky is blue.")
        experiment_id = self.engine.design_experiment(hypothesis_id, "Observe sky color at noon.")
        
        self.assertIsNotNone(experiment_id)
        
        # Verify experiment was stored
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT design FROM experiments WHERE id = ?", (experiment_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "Observe sky color at noon.")

    def test_add_evidence(self):
        """Test adding evidence to an experiment."""
        hypothesis_id = self.engine.propose_hypothesis("The sky is blue.")
        experiment_id = self.engine.design_experiment(hypothesis_id, "Observe sky color at noon.")
        evidence_id = self.engine.add_evidence(experiment_id, "Observed blue sky.", 0.9)
        
        self.assertIsNotNone(evidence_id)
        
        # Verify evidence was stored
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data, quality_score FROM evidence WHERE id = ?", (evidence_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "Observed blue sky.")
        self.assertEqual(result[1], 0.9)

    def test_analyze_hypothesis(self):
        """Test analyzing a hypothesis."""
        hypothesis_id = self.engine.propose_hypothesis("The sky is blue.")
        experiment_id = self.engine.design_experiment(hypothesis_id, "Observe sky color at noon.")
        self.engine.add_evidence(experiment_id, "Observed blue sky.", 0.9)
        
        analysis = self.engine.analyze_hypothesis(hypothesis_id)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["hypothesis_id"], hypothesis_id)
        self.assertEqual(analysis["evidence_count"], 1)
        self.assertAlmostEqual(analysis["avg_quality"], 0.9)

    def test_formulate_hypothesis(self):
        """Test formulating a hypothesis from an observation."""
        # This is a new method we're adding
        observation = Observation(
            id="obs1",
            description="Sky color observation",
            pattern="blue color during daytime",
            variables=["time_of_day", "weather_conditions"],
            confidence=0.8
        )
        
        hypothesis_id = self.engine.formulate_hypothesis(observation)
        
        self.assertIsNotNone(hypothesis_id)
        
        # Verify hypothesis was stored
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT statement FROM hypotheses WHERE id = ?", (hypothesis_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertIn("blue color", result[0])

    def test_execute_experiment(self):
        """Test executing an experiment."""
        # This is a new method we're adding
        hypothesis_id = self.engine.propose_hypothesis("The sky is blue.")
        experiment_design = self.engine.design_experiment_plan(
            hypothesis_id=hypothesis_id,
            methodology="observation",
            sample_size=10,
            duration_days=1
        )
        
        result = self.engine.execute_experiment(experiment_design)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.experiment_id, experiment_design.id)
        self.assertIsNotNone(result.raw_data)
        self.assertIsNotNone(result.processed_data)
        self.assertIsNotNone(result.statistical_analysis)

    def test_analyze_results(self):
        """Test analyzing experiment results."""
        # This is a new method we're adding
        hypothesis_id = self.engine.propose_hypothesis("The sky is blue.")
        experiment_design = self.engine.design_experiment_plan(
            hypothesis_id=hypothesis_id,
            methodology="observation",
            sample_size=10,
            duration_days=1
        )
        
        result = self.engine.execute_experiment(experiment_design)
        analysis = self.engine.analyze_results(result)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.hypothesis_id, hypothesis_id)
        self.assertEqual(analysis.experiment_id, experiment_design.id)
        self.assertIsNotNone(analysis.result_data)

    def test_validate_against_research(self):
        """Test validating findings against research."""
        # This is a new method we're adding
        finding = Finding(
            id="finding1",
            hypothesis_id="hyp1",
            statement="The sky appears blue due to Rayleigh scattering",
            effect_size=0.8,
            confidence=0.9,
            domain="atmospheric_science",
            supporting_experiments=["exp1", "exp2"],
            created_at=datetime.now()
        )
        
        validation = self.engine.validate_against_research(finding)
        
        self.assertIsNotNone(validation)
        self.assertEqual(validation.finding_id, finding.id)
        self.assertIn(validation.validation_status, ["validated", "partially_validated", "unvalidated"])

    def test_update_knowledge_base(self):
        """Test updating the knowledge base with validated findings."""
        # This is a new method we're adding
        finding = Finding(
            id="finding1",
            hypothesis_id="hyp1",
            statement="The sky appears blue due to Rayleigh scattering",
            effect_size=0.8,
            confidence=0.9,
            domain="atmospheric_science",
            supporting_experiments=["exp1", "exp2"],
            created_at=datetime.now()
        )
        
        validation = ValidationResult(
            finding_id=finding.id,
            validation_status="validated",
            research_support={"support_level": "high", "studies": 15},
            contradictory_evidence={"level": "low", "studies": 2},
            novelty_score=0.3,
            replication_recommendations=["Repeat with different atmospheric conditions"]
        )
        
        # Should not raise any exceptions
        self.engine.update_knowledge_base(finding, validation)
        
        # Verify knowledge entry was created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_base WHERE finding_id = ?", (finding.id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()