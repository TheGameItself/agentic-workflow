#!/usr/bin/env python3
"""
Unit Tests for Lambda Self-Repair Agent
@{CORE.SELF_REPAIR.TESTS.001} Unit tests for the Lambda Self-Repair Agent.
#{testing,unit_tests,self_repair,lambda,agent}
τ(β(testing_implementation))
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

# Import the agent
from core.SELF_REPAIR.implementations.agents.lambda_self_repair_agent import (
    LambdaSelfRepairAgent, Issue, RepairAction, RepairResult
)

class TestLambdaSelfRepairAgent(unittest.TestCase):
    """Unit tests for the Lambda Self-Repair Agent."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock core system
        self.mock_core_system = MagicMock()
        self.mock_core_system.memory_manager = MagicMock()
        self.mock_core_system.workflow_manager = MagicMock()
        self.mock_core_system._shutdown_event = MagicMock()
        self.mock_core_system._shutdown_event.is_set.return_value = False
        
        # Create agent with mock core system
        self.agent = LambdaSelfRepairAgent(self.mock_core_system)
        
        # Mock neural models
        self.agent.anomaly_detector = MagicMock()
        self.agent.repair_predictor = MagicMock()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.core_system, self.mock_core_system)
        self.assertEqual(self.agent.memory_manager, self.mock_core_system.memory_manager)
        self.assertEqual(self.agent.workflow_manager, self.mock_core_system.workflow_manager)
        self.assertEqual(len(self.agent.repair_history), 0)
        self.assertEqual(len(self.agent.active_repairs), 0)
        self.assertIsNotNone(self.agent.diagnostic_fn)
        self.assertIsNotNone(self.agent.repair_fn)
        self.assertIsNotNone(self.agent.optimize_fn)
        self.assertIsNotNone(self.agent.learn_fn)
    
    def test_create_diagnostic_function(self):
        """Test creation of diagnostic function."""
        diagnostic_fn = self.agent._create_diagnostic_function()
        self.assertIsNotNone(diagnostic_fn)
        self.assertTrue(callable(diagnostic_fn))
        
        # Mock _detect_anomalies method
        self.agent._detect_anomalies = MagicMock(return_value=[])
        
        # Call diagnostic function
        system_state = {"components": {}}
        diagnostic_fn(system_state)
        
        # Verify _detect_anomalies was called
        self.agent._detect_anomalies.assert_called_once_with(system_state)
    
    def test_create_repair_function(self):
        """Test creation of repair function."""
        repair_fn = self.agent._create_repair_function()
        self.assertIsNotNone(repair_fn)
        self.assertTrue(callable(repair_fn))
        
        # Mock _fix_issue method
        self.agent._fix_issue = MagicMock(return_value={})
        
        # Create test issue
        issue = Issue(
            id="test_issue",
            component="test_component",
            severity=0.8,
            description="Test issue",
            timestamp=datetime.now(),
            metrics={"cpu": 0.9}
        )
        
        # Call repair function
        system_state = {"components": {}}
        repair_fn(issue)(system_state)
        
        # Verify _fix_issue was called
        self.agent._fix_issue.assert_called_once_with(issue, system_state)
    
    @patch('core.SELF_REPAIR.implementations.agents.lambda_self_repair_agent.create_model')
    def test_agent_with_mocked_models(self, mock_create_model):
        """Test agent with mocked neural models."""
        # Setup mock models
        mock_anomaly_detector = MagicMock()
        mock_repair_predictor = MagicMock()
        
        mock_create_model.side_effect = lambda model_name: {
            "anomaly_detector": mock_anomaly_detector,
            "repair_predictor": mock_repair_predictor
        }.get(model_name)
        
        # Create agent with mocked models
        agent = LambdaSelfRepairAgent(self.mock_core_system)
        
        # Verify models were created
        self.assertEqual(agent.anomaly_detector, mock_anomaly_detector)
        self.assertEqual(agent.repair_predictor, mock_repair_predictor)
        
        # Verify create_model was called correctly
        mock_create_model.assert_any_call("anomaly_detector")
        mock_create_model.assert_any_call("repair_predictor")
    
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_run_method(self, mock_sleep):
        """Test the run method."""
        # Setup mocks
        self.agent._get_system_state = MagicMock(return_value={"components": {}})
        self.agent.diagnostic_fn = MagicMock(return_value=[])
        self.agent._plan_repair = MagicMock()
        self.agent._should_execute_repair = MagicMock(return_value=False)
        self.agent._should_optimize = MagicMock(return_value=False)
        self.agent._should_learn = MagicMock(return_value=False)
        
        # Set shutdown after one iteration
        self.mock_core_system._shutdown_event.is_set.side_effect = [False, True]
        
        # Run the agent
        await self.agent.run()
        
        # Verify methods were called
        self.agent._get_system_state.assert_called_once()
        self.agent.diagnostic_fn.assert_called_once()
        self.agent._should_optimize.assert_called_once()
        self.agent._should_learn.assert_called_once()
        mock_sleep.assert_called_once_with(5)
    
    def test_functional_composition(self):
        """Test functional composition of repair operations."""
        # Create test issue
        issue = Issue(
            id="test_issue",
            component="test_component",
            severity=0.8,
            description="Test issue",
            timestamp=datetime.now(),
            metrics={"cpu": 0.9}
        )
        
        # Create test system state
        system_state = {"components": {"test_component": {"status": "error"}}}
        
        # Mock fix_issue to return a new system state
        self.agent._fix_issue = MagicMock(return_value={
            "components": {"test_component": {"status": "ok"}}
        })
        
        # Create repair function using lambda calculus principles
        repair_fn = self.agent._create_repair_function()
        
        # Apply repair function to issue and system state
        new_system_state = repair_fn(issue)(system_state)
        
        # Verify fix_issue was called correctly
        self.agent._fix_issue.assert_called_once_with(issue, system_state)
        
        # Verify new system state
        self.assertEqual(new_system_state["components"]["test_component"]["status"], "ok")

if __name__ == '__main__':
    unittest.main()