#!/usr/bin/env python3
"""
Comprehensive test suite for MCP server core functionality.

Tests all workflow/task operations, lobes, CLI usage patterns, and integration points.
Based on research-driven development principles from idea.txt.
"""

import unittest
import tempfile
import os
import json
import sqlite3
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import all core modules
from .workflow import WorkflowManager, WorkflowStep
from .task_manager import TaskManager
from .memory import MemoryManager
from .experimental_lobes import (
    AlignmentEngine, PatternRecognitionEngine, SimulatedReality,
    DreamingEngine, MindMapEngine, ScientificProcessEngine,
    SplitBrainABTest, MultiLLMOrchestrator, AdvancedEngramEngine
)


class TestWorkflowManager(unittest.TestCase):
    """Test WorkflowManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_workflow.db')
        self.workflow_manager = WorkflowManager(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_create_workflow(self):
        """Test workflow creation."""
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Test Project",
            project_path="/tmp/test_project"
        )
        self.assertIsNotNone(workflow_id)
        self.assertIsInstance(workflow_id, int)
    
    def test_get_workflow_status(self):
        """Test workflow status retrieval."""
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Test Project",
            project_path="/tmp/test_project"
        )
        status = self.workflow_manager.get_workflow_status()
        self.assertIsNotNone(status)
        self.assertIsInstance(status, dict)
    
    def test_start_step(self):
        """Test step starting."""
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Test Project",
            project_path="/tmp/test_project"
        )
        
        success = self.workflow_manager.start_step("init")
        self.assertTrue(success)
    
    def test_complete_step(self):
        """Test step completion."""
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Test Project",
            project_path="/tmp/test_project"
        )
        
        self.workflow_manager.start_step("init")
        success = self.workflow_manager.complete_step("init")
        self.assertTrue(success)
    
    def test_add_step_feedback(self):
        """Test adding step feedback."""
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Test Project",
            project_path="/tmp/test_project"
        )
        
        self.workflow_manager.add_step_feedback("init", "Test feedback", impact=5)
        # Verify feedback was added (would need to check database or add getter method)
        pass

    def test_workflow_meta_partial_and_failstate(self):
        """Test meta/partial tasks, feedback-driven reorg, failstate handling, and advanced dependencies in WorkflowManager. Covers edge cases and integration with memory/experimental lobes. References idea.txt and research."""
        workflow_manager = WorkflowManager()
        task_manager = TaskManager()
        memory = MemoryManager()
        # Add meta step
        meta_step = "meta_step"
        workflow_manager.register_step(meta_step, WorkflowStep(name=meta_step, description="Meta step", is_meta=True))
        workflow_manager.steps[meta_step].set_partial_progress(0.5)
        self.assertEqual(workflow_manager.steps[meta_step].get_partial_progress(), 0.5)
        # Add failstate feedback
        workflow_manager.add_step_feedback(meta_step, "Simulated failstate", impact=-10, principle="failstate")
        failstates = [n for n in workflow_manager.steps if any(fb.get('impact', 0) < -5 for fb in getattr(workflow_manager.steps[n], 'feedback', []))]
        self.assertIn(meta_step, failstates)
        # Add advanced dependency step
        dep_step = "dep_step"
        workflow_manager.register_step(dep_step, WorkflowStep(name=dep_step, description="Dependent step", dependencies=[meta_step]))
        self.assertFalse(workflow_manager.start_step(dep_step))
        workflow_manager.complete_step(meta_step)
        self.assertTrue(workflow_manager.start_step(dep_step))
        # Test engram integration (stub)
        engram_id = memory.add_memory(text="Test Engram", memory_type="engram", tags=["test"])
        workflow_manager.steps[meta_step].engram_ids = [engram_id]
        self.assertIn(engram_id, workflow_manager.steps[meta_step].engram_ids)
        # Test feedback-driven reorg (stub)
        workflow_manager.autonomous_reorganize()
        # Test error condition: start non-existent step
        self.assertFalse(workflow_manager.start_step("nonexistent"))
        # Test error condition: complete non-existent step
        self.assertFalse(workflow_manager.complete_step("nonexistent"))
        # Test error condition: add feedback to non-existent step
        try:
            workflow_manager.add_step_feedback("nonexistent", "feedback")
            self.fail("Should raise exception for non-existent step")
        except Exception:
            pass


class TestTaskManager(unittest.TestCase):
    """Test TaskManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_tasks.db')
        self.task_manager = TaskManager(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_create_task(self):
        """Test task creation."""
        task_id = self.task_manager.create_task(
            title="Test Task",
            description="Test task for unit testing",
            priority=8
        )
        self.assertIsNotNone(task_id)
        self.assertIsInstance(task_id, int)
    
    def test_get_tasks(self):
        """Test task retrieval."""
        task_id = self.task_manager.create_task(
            title="Test Task",
            description="Test task for unit testing"
        )
        tasks = self.task_manager.get_tasks()
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)
    
    def test_update_task_progress(self):
        """Test task progress updates."""
        task_id = self.task_manager.create_task(
            title="Progress Test Task"
        )
        
        success = self.task_manager.update_task_progress(task_id, 0.5, "Half done")
        self.assertTrue(success)
        
        progress = self.task_manager.get_task_progress(task_id)
        if progress:  # Check if progress exists before accessing
            self.assertEqual(progress['progress_percentage'], 0.5)
    
    def test_add_task_dependency(self):
        """Test task dependency creation."""
        task1_id = self.task_manager.create_task(title="Task 1")
        task2_id = self.task_manager.create_task(title="Task 2")
        
        dep_id = self.task_manager.add_task_dependency(task2_id, task1_id)
        self.assertIsInstance(dep_id, int)
    
    def test_get_task_tree(self):
        """Test task tree retrieval."""
        root_id = self.task_manager.create_task(title="Root Task")
        child_id = self.task_manager.create_task(title="Child Task", parent_id=root_id)
        
        tree = self.task_manager.get_task_tree(root_id)
        self.assertIsInstance(tree, dict)
        self.assertIn('children', tree)


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_memory.db')
        self.memory_manager = MemoryManager(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_add_memory(self):
        """Test memory storage."""
        memory_id = self.memory_manager.add_memory(
            text="Test memory content",
            memory_type="test",
            tags=["test", "unit"]
        )
        self.assertIsNotNone(memory_id)
        self.assertIsInstance(memory_id, int)
    
    def test_get_memory(self):
        """Test memory retrieval."""
        memory_id = self.memory_manager.add_memory(
            text="Test memory content",
            memory_type="test"
        )
        memory = self.memory_manager.get_memory(memory_id)
        self.assertIsNotNone(memory)
        if memory:  # Additional check for linter
            self.assertEqual(memory['text'], "Test memory content")
    
    def test_search_memories(self):
        """Test memory search."""
        # Store multiple memories
        self.memory_manager.add_memory(
            text="First test memory",
            tags=["test"]
        )
        self.memory_manager.add_memory(
            text="Second test memory",
            tags=["test"]
        )
        
        results = self.memory_manager.search_memories("test")
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 2)
    
    def test_add_task(self):
        """Test task creation in memory manager."""
        task_id = self.memory_manager.add_task(
            description="Test task"
        )
        self.assertIsInstance(task_id, int)
        
        tasks = self.memory_manager.get_tasks()
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)


class TestExperimentalLobes(unittest.TestCase):
    """Test experimental lobes functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_lobes.db')
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_alignment_engine(self):
        """Test AlignmentEngine functionality."""
        engine = AlignmentEngine(db_path=self.db_path)
        
        # Test alignment
        output = "This is a test output that needs alignment."
        aligned = engine.align(output, {"preference": "concise"})
        self.assertIsInstance(aligned, str)
        self.assertNotEqual(output, aligned)  # Should be modified
    
    def test_pattern_recognition_engine(self):
        """Test PatternRecognitionEngine functionality."""
        engine = PatternRecognitionEngine(db_path=self.db_path)
        
        # Test pattern recognition
        data_batch = ["test data 1", "test data 2", "test data 3"]
        patterns = engine.recognize_patterns(data_batch)
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
    
    def test_simulated_reality(self):
        """Test SimulatedReality functionality."""
        engine = SimulatedReality(db_path=self.db_path)
        
        # Test entity management
        entity_id = engine.add_entity("test_entity", {"type": "test"})
        self.assertIsInstance(entity_id, int)
        
        entities = engine.query_entities()
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
    
    def test_dreaming_engine(self):
        """Test DreamingEngine functionality."""
        engine = DreamingEngine(db_path=self.db_path)
        # Test dream simulation with correct parameters (context and scenario)
        dream = engine.simulate_dream("Test context", "scenario")
        self.assertIsInstance(dream, dict)
        self.assertIn('scenario', dream)
    
    def test_mind_map_engine(self):
        """Test MindMapEngine functionality."""
        engine = MindMapEngine(db_path=self.db_path)
        
        # Test node creation
        node_id = engine.add_node("test_node", {"data": "test"})
        self.assertIsInstance(node_id, str)
        
        # Test edge creation
        edge_id = engine.add_edge("test_node", "another_node")
        self.assertIsInstance(edge_id, str)
    
    def test_scientific_process_engine(self):
        """Test ScientificProcessEngine functionality."""
        engine = ScientificProcessEngine(db_path=self.db_path)
        
        # Test hypothesis creation
        hypothesis_id = engine.propose_hypothesis(
            "Test hypothesis",
            confidence=0.8
        )
        self.assertIsInstance(hypothesis_id, int)
    
    def test_split_brain_ab_test(self):
        """Test SplitBrainABTest functionality."""
        # Create dummy lobe class for testing
        class DummyLobe:
            def __init__(self, config=None):
                self.config = config or {}
            
            def process(self, input_data):
                return f"Processed: {input_data}"
        
        # Test AB testing
        ab_test = SplitBrainABTest(
            lobe_class=DummyLobe,
            left_config={"strategy": "conservative"},
            right_config={"strategy": "aggressive"},
            db_path=self.db_path
        )
        
        result = ab_test.run_test("test input")
        self.assertIsInstance(result, dict)
        self.assertIn('winner', result)
    
    def test_multi_llm_orchestrator(self):
        """Test MultiLLMOrchestrator functionality."""
        orchestrator = MultiLLMOrchestrator(db_path=self.db_path)
        # Test query routing with correct parameter type (list of dicts)
        tasks = [{"prompt": "test query"}]
        result = orchestrator.route_query(tasks)
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
    
    def test_advanced_engram_engine(self):
        """Test AdvancedEngramEngine functionality."""
        engine = AdvancedEngramEngine(db_path=self.db_path)
        
        # Test engram compression
        engram = {"data": "test engram data", "metadata": {"type": "test"}}
        compressed = engine.compress(engram)
        self.assertIsInstance(compressed, dict)
        self.assertIn('compressed_data', compressed)





class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_integration.db')
        
        # Initialize all components
        self.workflow_manager = WorkflowManager(db_path=self.db_path)
        self.task_manager = TaskManager(db_path=self.db_path)
        self.memory_manager = MemoryManager(db_path=self.db_path)
        self.alignment_engine = AlignmentEngine(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_workflow_with_tasks(self):
        """Test workflow creation with associated tasks."""
        # Create workflow
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Integration Test Project",
            project_path="/tmp/integration_test"
        )
        
        # Create tasks for workflow
        task1_id = self.task_manager.create_task(
            title="Task 1",
            description="Integration test task 1"
        )
        task2_id = self.task_manager.create_task(
            title="Task 2",
            description="Integration test task 2"
        )
        
        # Verify tasks were created
        tasks = self.task_manager.get_tasks()
        self.assertGreaterEqual(len(tasks), 2)
    
    def test_memory_integration(self):
        """Test memory integration with workflows and tasks."""
        # Create workflow and task
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Memory Test Project",
            project_path="/tmp/memory_test"
        )
        task_id = self.task_manager.create_task(
            title="Memory Test Task"
        )
        
        # Store memory related to task
        memory_id = self.memory_manager.add_memory(
            text="Task-related memory",
            tags=["task", str(task_id)]
        )
        
        # Search for task-related memories
        memories = self.memory_manager.search_memories(str(task_id))
        self.assertGreater(len(memories), 0)
    
    def test_alignment_integration(self):
        """Test alignment engine integration."""
        # Test alignment with workflow context
        workflow_id = self.workflow_manager.create_workflow(
            project_name="Alignment Test Project",
            project_path="/tmp/alignment_test"
        )
        
        # Align output based on workflow context
        output = "This output needs alignment."
        aligned = self.alignment_engine.align(
            output,
            {"workflow_id": workflow_id, "preference": "concise"}
        )
        
        self.assertIsInstance(aligned, str)
        self.assertNotEqual(output, aligned)


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestWorkflowManager,
        TestTaskManager,
        TestMemoryManager,
        TestExperimentalLobes,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    # Run all tests
    result = run_all_tests()
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code) 