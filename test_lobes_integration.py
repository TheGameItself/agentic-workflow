#!/usr/bin/env python3
"""
Integration tests for all experimental lobes in the MCP server.
Tests the interaction between different lobes and their combined functionality.
"""

import unittest
import tempfile
import os
import json
from datetime import datetime
from src.mcp.experimental_lobes import (
    AlignmentEngine,
    PatternRecognitionEngine,
    SimulatedReality,
    DreamingEngine,
    MindMapEngine,
    ScientificProcessEngine,
    SplitBrainABTest,
    MultiLLMOrchestrator,
    AdvancedEngramEngine
)
from src.mcp.memory import MemoryManager
from src.mcp.task_manager import TaskManager
from src.mcp.workflow import WorkflowManager


class TestLobesIntegration(unittest.TestCase):
    """Integration tests for all experimental lobes."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        
        # Initialize core components
        self.memory_manager = MemoryManager(db_path=self.db_path)
        self.task_manager = TaskManager(db_path=self.db_path)
        self.workflow_manager = WorkflowManager()
        
        # Initialize all lobes
        self.alignment_engine = AlignmentEngine(self.memory_manager)
        self.pattern_engine = PatternRecognitionEngine(self.memory_manager)
        self.reality_engine = SimulatedReality(self.memory_manager)
        self.dreaming_engine = DreamingEngine(self.memory_manager)
        self.mindmap_engine = MindMapEngine(self.memory_manager)
        self.scientific_engine = ScientificProcessEngine(self.memory_manager)
        self.split_brain = SplitBrainABTest(self.memory_manager)
        self.multi_llm = MultiLLMOrchestrator(self.memory_manager)
        self.engram_engine = AdvancedEngramEngine(self.memory_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_alignment_engine_integration(self):
        """Test alignment engine integration with memory and tasks."""
        # Create test task
        task_id = self.task_manager.create_task(
            title="Test Alignment Task",
            description="Test task for alignment engine",
            priority=5
        )
        
        # Test alignment analysis
        alignment_result = self.alignment_engine.analyze_alignment(
            task_id=task_id,
            user_preferences={"efficiency": 0.8, "quality": 0.9},
            context="High priority task requiring careful attention"
        )
        
        self.assertIsInstance(alignment_result, dict)
        self.assertIn('alignment_score', alignment_result)
        self.assertIn('suggestions', alignment_result)
        self.assertGreaterEqual(alignment_result['alignment_score'], 0.0)
        self.assertLessEqual(alignment_result['alignment_score'], 1.0)
        
        # Test feedback integration
        feedback_id = self.task_manager.add_task_feedback(
            task_id, "Good alignment with user preferences", impact_score=2
        )
        
        updated_alignment = self.alignment_engine.update_alignment_model(
            task_id=task_id,
            feedback_id=feedback_id
        )
        
        self.assertIsInstance(updated_alignment, dict)
    
    def test_pattern_recognition_integration(self):
        """Test pattern recognition engine integration."""
        # Add test memories with patterns
        memories = [
            "Task completed successfully with good documentation",
            "Task failed due to missing requirements",
            "Task completed successfully with thorough testing",
            "Task failed due to poor planning",
            "Task completed successfully with clear communication"
        ]
        
        memory_ids = []
        for memory in memories:
            mem_id = self.memory_manager.add_memory(
                text=memory,
                memory_type="task_outcome",
                priority=3
            )
            memory_ids.append(mem_id)
        
        # Test pattern recognition
        patterns = self.pattern_engine.recognize_patterns(
            memory_ids=memory_ids,
            pattern_type="success_failure"
        )
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Test pattern application to new task
        new_task_id = self.task_manager.create_task(
            title="New Task",
            description="Apply learned patterns",
            priority=4
        )
        
        applied_patterns = self.pattern_engine.apply_patterns(
            task_id=new_task_id,
            patterns=patterns
        )
        
        self.assertIsInstance(applied_patterns, dict)
        self.assertIn('applied_patterns', applied_patterns)
    
    def test_simulated_reality_integration(self):
        """Test simulated reality engine integration."""
        # Create simulated entities
        entity_id = self.reality_engine.create_entity(
            entity_type="user",
            properties={"name": "Test User", "role": "developer"}
        )
        
        self.assertIsInstance(entity_id, str)
        
        # Create simulated event
        event_id = self.reality_engine.create_event(
            event_type="task_completion",
            entities=[entity_id],
            properties={"task_name": "Test Task", "duration": 120}
        )
        
        self.assertIsInstance(event_id, str)
        
        # Test state tracking
        state_id = self.reality_engine.update_state(
            entity_id=entity_id,
            state_type="workload",
            properties={"active_tasks": 3, "completed_tasks": 15}
        )
        
        self.assertIsInstance(state_id, str)
        
        # Test reality query
        reality_info = self.reality_engine.query_reality(
            query_type="entity_state",
            entity_id=entity_id
        )
        
        self.assertIsInstance(reality_info, dict)
        self.assertIn('entity_id', reality_info)
    
    def test_dreaming_engine_integration(self):
        """Test dreaming engine integration."""
        # Create scenario for dreaming
        scenario = {
            "context": "Software development project",
            "entities": ["developer", "project_manager", "client"],
            "constraints": ["budget", "timeline", "quality"],
            "goals": ["deliver_on_time", "meet_requirements", "maintain_quality"]
        }
        
        # Test scenario simulation
        simulation_result = self.dreaming_engine.simulate_scenario(
            scenario=scenario,
            iterations=5,
            duration_minutes=10
        )
        
        self.assertIsInstance(simulation_result, dict)
        self.assertIn('outcomes', simulation_result)
        self.assertIn('insights', simulation_result)
        
        # Test insight extraction
        insights = self.dreaming_engine.extract_insights(
            simulation_id=simulation_result.get('simulation_id')
        )
        
        self.assertIsInstance(insights, list)
        
        # Test insight application
        if insights:
            applied_insight = self.dreaming_engine.apply_insight(
                insight=insights[0],
                target_context="current_project"
            )
            
            self.assertIsInstance(applied_insight, dict)
    
    def test_mindmap_engine_integration(self):
        """Test mind map engine integration."""
        # Create nodes
        node1_id = self.mindmap_engine.create_node(
            content="Project Planning",
            node_type="concept"
        )
        
        node2_id = self.mindmap_engine.create_node(
            content="Requirements Analysis",
            node_type="concept"
        )
        
        node3_id = self.mindmap_engine.create_node(
            content="Implementation",
            node_type="concept"
        )
        
        # Create associations
        assoc1_id = self.mindmap_engine.create_association(
            source_node=node1_id,
            target_node=node2_id,
            relationship_type="leads_to",
            strength=0.8
        )
        
        assoc2_id = self.mindmap_engine.create_association(
            source_node=node2_id,
            target_node=node3_id,
            relationship_type="enables",
            strength=0.9
        )
        
        # Test graph traversal
        path = self.mindmap_engine.find_path(
            start_node=node1_id,
            end_node=node3_id
        )
        
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        
        # Test association discovery
        associations = self.mindmap_engine.discover_associations(
            node_id=node2_id,
            max_distance=2
        )
        
        self.assertIsInstance(associations, list)
    
    def test_scientific_process_integration(self):
        """Test scientific process engine integration."""
        # Create hypothesis
        hypothesis_id = self.scientific_engine.create_hypothesis(
            statement="Using TDD improves code quality",
            context="Software development practices",
            variables=["test_coverage", "bug_count", "development_time"]
        )
        
        self.assertIsInstance(hypothesis_id, str)
        
        # Design experiment
        experiment_id = self.scientific_engine.design_experiment(
            hypothesis_id=hypothesis_id,
            methodology="A/B testing",
            sample_size=100,
            duration_days=30
        )
        
        self.assertIsInstance(experiment_id, str)
        
        # Add test data
        test_data = {
            "test_coverage": [85, 90, 88, 92, 87],
            "bug_count": [5, 3, 4, 2, 3],
            "development_time": [120, 110, 115, 105, 112]
        }
        
        analysis_result = self.scientific_engine.analyze_results(
            experiment_id=experiment_id,
            data=test_data
        )
        
        self.assertIsInstance(analysis_result, dict)
        self.assertIn('statistical_significance', analysis_result)
        self.assertIn('conclusion', analysis_result)
        
        # Test hypothesis validation
        validation = self.scientific_engine.validate_hypothesis(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id
        )
        
        self.assertIsInstance(validation, dict)
        self.assertIn('validated', validation)
    
    def test_split_brain_integration(self):
        """Test split brain AB testing integration."""
        # Create test scenario
        scenario = {
            "task": "Code review process optimization",
            "left_approach": "Automated review tools",
            "right_approach": "Manual peer review",
            "metrics": ["review_time", "bug_catch_rate", "developer_satisfaction"]
        }
        
        # Start AB test
        test_id = self.split_brain.start_ab_test(
            scenario=scenario,
            duration_days=7,
            sample_size=50
        )
        
        self.assertIsInstance(test_id, str)
        
        # Add test data
        left_data = {
            "review_time": [15, 12, 18, 14, 16],
            "bug_catch_rate": [0.85, 0.88, 0.82, 0.87, 0.84],
            "developer_satisfaction": [7, 8, 6, 7, 7]
        }
        
        right_data = {
            "review_time": [45, 52, 38, 48, 42],
            "bug_catch_rate": [0.92, 0.89, 0.94, 0.91, 0.93],
            "developer_satisfaction": [8, 7, 9, 8, 8]
        }
        
        self.split_brain.add_test_data(
            test_id=test_id,
            side="left",
            data=left_data
        )
        
        self.split_brain.add_test_data(
            test_id=test_id,
            side="right",
            data=right_data
        )
        
        # Analyze results
        analysis = self.split_brain.analyze_results(test_id=test_id)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('winner', analysis)
        self.assertIn('confidence', analysis)
        self.assertIn('recommendations', analysis)
    
    def test_multi_llm_orchestrator_integration(self):
        """Test multi-LLM orchestrator integration."""
        # Define task types
        task_types = {
            "code_review": "Requires deep understanding of code quality and best practices",
            "documentation": "Requires clear writing and technical communication",
            "debugging": "Requires analytical thinking and problem-solving",
            "planning": "Requires strategic thinking and project management"
        }
        
        # Configure orchestrator
        self.multi_llm.configure_routing(
            task_types=task_types,
            routing_strategy="specialized"
        )
        
        # Test task routing
        routing_result = self.multi_llm.route_task(
            task_description="Review this Python code for security vulnerabilities",
            task_type="code_review"
        )
        
        self.assertIsInstance(routing_result, dict)
        self.assertIn('assigned_llm', routing_result)
        self.assertIn('reasoning', routing_result)
        
        # Test parallel processing
        tasks = [
            {"description": "Write API documentation", "type": "documentation"},
            {"description": "Debug authentication issue", "type": "debugging"},
            {"description": "Plan sprint roadmap", "type": "planning"}
        ]
        
        parallel_results = self.multi_llm.process_parallel(
            tasks=tasks,
            max_concurrent=3
        )
        
        self.assertIsInstance(parallel_results, list)
        self.assertEqual(len(parallel_results), len(tasks))
    
    def test_engram_engine_integration(self):
        """Test advanced engram engine integration."""
        # Create coding model
        model_id = self.engram_engine.create_coding_model(
            model_type="task_completion",
            parameters={
                "complexity_threshold": 0.7,
                "time_estimation_factor": 1.2,
                "quality_weight": 0.8
            }
        )
        
        self.assertIsInstance(model_id, str)
        
        # Train model with data
        training_data = [
            {
                "task_complexity": 0.6,
                "estimated_time": 120,
                "actual_time": 135,
                "quality_score": 0.85
            },
            {
                "task_complexity": 0.8,
                "estimated_time": 240,
                "actual_time": 280,
                "quality_score": 0.92
            }
        ]
        
        training_result = self.engram_engine.train_model(
            model_id=model_id,
            data=training_data
        )
        
        self.assertIsInstance(training_result, dict)
        self.assertIn('accuracy', training_result)
        
        # Test model prediction
        prediction = self.engram_engine.predict(
            model_id=model_id,
            input_data={
                "task_complexity": 0.7,
                "estimated_time": 180
            }
        )
        
        self.assertIsInstance(prediction, dict)
        self.assertIn('predicted_time', prediction)
        self.assertIn('predicted_quality', prediction)
        
        # Test engram compression
        compression_result = self.engram_engine.compress_engrams(
            model_ids=[model_id],
            compression_ratio=0.5
        )
        
        self.assertIsInstance(compression_result, dict)
        self.assertIn('compressed_size', compression_result)
    
    def test_cross_lobe_integration(self):
        """Test integration between multiple lobes."""
        # Create a complex scenario involving multiple lobes
        
        # 1. Create task with alignment analysis
        task_id = self.task_manager.create_task(
            title="Optimize Database Queries",
            description="Improve database performance through query optimization",
            priority=5
        )
        
        alignment_result = self.alignment_engine.analyze_alignment(
            task_id=task_id,
            user_preferences={"performance": 0.9, "maintainability": 0.7},
            context="Production database with high load"
        )
        
        # 2. Use pattern recognition to find similar tasks
        similar_patterns = self.pattern_engine.recognize_patterns(
            memory_ids=self.memory_manager.search_memories("database optimization", limit=5),
            pattern_type="performance_improvement"
        )
        
        # 3. Create mind map of the optimization process
        root_node = self.mindmap_engine.create_node(
            content="Database Optimization",
            node_type="concept"
        )
        
        sub_nodes = [
            self.mindmap_engine.create_node("Query Analysis", "step"),
            self.mindmap_engine.create_node("Index Optimization", "step"),
            self.mindmap_engine.create_node("Performance Testing", "step")
        ]
        
        for sub_node in sub_nodes:
            self.mindmap_engine.create_association(
                source_node=root_node,
                target_node=sub_node,
                relationship_type="includes",
                strength=0.9
            )
        
        # 4. Use scientific process to test optimization hypothesis
        hypothesis_id = self.scientific_engine.create_hypothesis(
            statement="Adding composite indexes improves query performance by 30%",
            context="Database optimization",
            variables=["query_time", "index_count", "data_size"]
        )
        
        # 5. Use split brain to test different optimization approaches
        ab_test_id = self.split_brain.start_ab_test(
            scenario={
                "task": "Database optimization approach",
                "left_approach": "Index-based optimization",
                "right_approach": "Query rewriting",
                "metrics": ["query_time", "memory_usage", "maintenance_overhead"]
            },
            duration_days=3,
            sample_size=20
        )
        
        # 6. Use engram engine to predict outcomes
        prediction_model = self.engram_engine.create_coding_model(
            model_type="optimization_impact",
            parameters={"baseline_performance": 100, "improvement_threshold": 0.2}
        )
        
        prediction = self.engram_engine.predict(
            model_id=prediction_model,
            input_data={
                "current_query_time": 500,
                "optimization_complexity": 0.6,
                "data_size": 1000000
            }
        )
        
        # Verify all components worked together
        self.assertIsInstance(alignment_result, dict)
        self.assertIsInstance(similar_patterns, list)
        self.assertIsInstance(root_node, str)
        self.assertIsInstance(hypothesis_id, str)
        self.assertIsInstance(ab_test_id, str)
        self.assertIsInstance(prediction, dict)
        
        # Test that the integration provides coherent results
        self.assertGreater(alignment_result.get('alignment_score', 0), 0)
        self.assertGreater(len(similar_patterns), 0)
        self.assertIn('predicted_time', prediction)


class TestLobesPerformance(unittest.TestCase):
    """Performance tests for lobes integration."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_performance.db")
        self.memory_manager = MemoryManager(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up performance test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_large_scale_pattern_recognition(self):
        """Test pattern recognition with large dataset."""
        pattern_engine = PatternRecognitionEngine(self.memory_manager)
        
        # Create large dataset
        memory_ids = []
        for i in range(1000):
            mem_id = self.memory_manager.add_memory(
                text=f"Task {i} completed with {'success' if i % 3 == 0 else 'failure'}",
                memory_type="task_outcome",
                priority=2
            )
            memory_ids.append(mem_id)
        
        # Test pattern recognition performance
        import time
        start_time = time.time()
        
        patterns = pattern_engine.recognize_patterns(
            memory_ids=memory_ids,
            pattern_type="success_failure"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(processing_time, 10.0)  # 10 seconds max
        self.assertIsInstance(patterns, list)
    
    def test_concurrent_lobe_operations(self):
        """Test concurrent operations across multiple lobes."""
        import threading
        import time
        
        alignment_engine = AlignmentEngine(self.memory_manager)
        pattern_engine = PatternRecognitionEngine(self.memory_manager)
        mindmap_engine = MindMapEngine(self.memory_manager)
        
        results = {}
        errors = []
        
        def run_alignment():
            try:
                result = alignment_engine.analyze_alignment(
                    task_id=1,
                    user_preferences={"efficiency": 0.8},
                    context="Test context"
                )
                results['alignment'] = result
            except Exception as e:
                errors.append(f"Alignment error: {e}")
        
        def run_patterns():
            try:
                memory_ids = [self.memory_manager.add_memory("test", "test", 1)]
                result = pattern_engine.recognize_patterns(memory_ids, "test")
                results['patterns'] = result
            except Exception as e:
                errors.append(f"Pattern error: {e}")
        
        def run_mindmap():
            try:
                node_id = mindmap_engine.create_node("test", "concept")
                results['mindmap'] = node_id
            except Exception as e:
                errors.append(f"Mindmap error: {e}")
        
        # Run operations concurrently
        threads = [
            threading.Thread(target=run_alignment),
            threading.Thread(target=run_patterns),
            threading.Thread(target=run_mindmap)
        ]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete without errors and within reasonable time
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertLess(processing_time, 5.0)  # 5 seconds max
        self.assertIn('alignment', results)
        self.assertIn('patterns', results)
        self.assertIn('mindmap', results)


if __name__ == "__main__":
    # Run integration tests
    unittest.main(verbosity=2) 