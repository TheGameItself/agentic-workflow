"""
Performance tests for the MCP system.

This module tests system performance under various conditions.

βperformance_tests(system_efficiency)
"""

import pytest
import time
import threading
import concurrent.futures
from typing import Any, Dict, List
import statistics

from core.src.mcp.implementations.memory_manager import BasicMemoryManager
from core.src.mcp.implementations.workflow_engine import BasicWorkflowEngine
from core.src.mcp.implementations.context_manager import BasicContextManager
from core.src.mcp.lobes.memory_lobe import MemoryLobe


class TestMemoryPerformance:
    """Test memory manager performance."""
    
    def test_memory_store_performance(self, memory_manager: BasicMemoryManager):
        """Test memory storage performance."""
        data_sizes = [100, 1000, 10000]  # Different data sizes
        results = {}
        
        for size in data_sizes:
            # Create test data
            test_data = {"data": "x" * size, "size": size}
            
            # Measure storage time
            start_time = time.time()
            key = memory_manager.store(test_data)
            end_time = time.time()
            
            storage_time = end_time - start_time
            results[size] = storage_time
            
            # Verify data was stored
            assert key is not None
            retrieved_data = memory_manager.retrieve(key)
            assert retrieved_data["size"] == size
        
        # Verify performance scales reasonably
        assert results[100] <= results[1000] <= results[10000]
        
        # Storage should be fast (under 1 second for all sizes)
        for size, time_taken in results.items():
            assert time_taken < 1.0, f"Storage took too long for size {size}: {time_taken}s"
    
    def test_memory_search_performance(self, memory_manager: BasicMemoryManager):
        """Test memory search performance."""
        # Store multiple items
        num_items = 100
        keys = []
        
        for i in range(num_items):
            data = {
                "id": i,
                "content": f"This is test content number {i}",
                "category": f"category_{i % 10}"
            }
            key = memory_manager.store(data)
            keys.append(key)
        
        # Test search performance
        search_queries = ["test", "content", "category_5"]
        
        for query in search_queries:
            start_time = time.time()
            results = memory_manager.search(query, limit=20)
            end_time = time.time()
            
            search_time = end_time - start_time
            
            # Search should be fast
            assert search_time < 0.5, f"Search took too long for query '{query}': {search_time}s"
            
            # Should return relevant results
            assert len(results) > 0
            assert all(query.lower() in str(result[1]).lower() for result in results)
    
    def test_concurrent_memory_access(self, memory_manager: BasicMemoryManager):
        """Test concurrent access to memory manager."""
        num_threads = 10
        operations_per_thread = 50
        
        def worker_function(thread_id: int) -> List[float]:
            """Worker function for concurrent testing."""
            times = []
            
            for i in range(operations_per_thread):
                # Store operation
                start_time = time.time()
                data = {"thread": thread_id, "operation": i, "data": f"data_{thread_id}_{i}"}
                key = memory_manager.store(data)
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Retrieve operation
                start_time = time.time()
                retrieved_data = memory_manager.retrieve(key)
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Verify data integrity
                assert retrieved_data["thread"] == thread_id
                assert retrieved_data["operation"] == i
            
            return times
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            all_times = []
            
            for future in concurrent.futures.as_completed(futures):
                thread_times = future.result()
                all_times.extend(thread_times)
        
        # Analyze performance
        avg_time = statistics.mean(all_times)
        max_time = max(all_times)
        
        # Performance should be reasonable under concurrent load
        assert avg_time < 0.1, f"Average operation time too high: {avg_time}s"
        assert max_time < 1.0, f"Maximum operation time too high: {max_time}s"


class TestWorkflowPerformance:
    """Test workflow engine performance."""
    
    def test_workflow_creation_performance(self, workflow_engine: BasicWorkflowEngine):
        """Test workflow creation performance."""
        num_workflows = 100
        creation_times = []
        
        for i in range(num_workflows):
            workflow_data = {
                "name": f"Test Workflow {i}",
                "description": f"Performance test workflow {i}",
                "steps": [
                    {"name": f"Step {j}", "action": f"action_{j}", "parameters": {"param": j}}
                    for j in range(5)  # 5 steps per workflow
                ]
            }
            
            start_time = time.time()
            workflow_id = workflow_engine.create_workflow(workflow_data)
            end_time = time.time()
            
            creation_time = end_time - start_time
            creation_times.append(creation_time)
            
            # Verify workflow was created
            assert workflow_id is not None
            workflow = workflow_engine.get_workflow(workflow_id)
            assert workflow["name"] == f"Test Workflow {i}"
        
        # Analyze performance
        avg_time = statistics.mean(creation_times)
        max_time = max(creation_times)
        
        assert avg_time < 0.05, f"Average workflow creation time too high: {avg_time}s"
        assert max_time < 0.2, f"Maximum workflow creation time too high: {max_time}s"
    
    def test_workflow_execution_performance(self, workflow_engine: BasicWorkflowEngine):
        """Test workflow execution performance."""
        # Create a test workflow
        workflow_data = {
            "name": "Performance Test Workflow",
            "description": "Workflow for performance testing",
            "steps": [
                {"name": "Step 1", "action": "test_action", "parameters": {"delay": 0.01}},
                {"name": "Step 2", "action": "test_action", "parameters": {"delay": 0.01}},
                {"name": "Step 3", "action": "test_action", "parameters": {"delay": 0.01}}
            ]
        }
        
        workflow_id = workflow_engine.create_workflow(workflow_data)
        
        # Test execution performance
        num_executions = 10
        execution_times = []
        
        for i in range(num_executions):
            start_time = time.time()
            
            # Simulate workflow execution by updating status
            workflow_engine.update_workflow_status(workflow_id, "running")
            workflow_engine.update_workflow_status(workflow_id, "completed")
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
        
        # Analyze performance
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        
        assert avg_time < 0.1, f"Average workflow execution time too high: {avg_time}s"
        assert max_time < 0.5, f"Maximum workflow execution time too high: {max_time}s"


class TestLobePerformance:
    """Test lobe performance."""
    
    def test_lobe_initialization_performance(self, temp_dir: str):
        """Test lobe initialization performance."""
        num_lobes = 50
        initialization_times = []
        lobes = []
        
        for i in range(num_lobes):
            lobe = MemoryLobe()
            config = {"storage_path": f"{temp_dir}/lobe_{i}.json"}
            
            start_time = time.time()
            success = lobe.initialize(config)
            end_time = time.time()
            
            initialization_time = end_time - start_time
            initialization_times.append(initialization_time)
            
            assert success
            lobes.append(lobe)
        
        # Analyze performance
        avg_time = statistics.mean(initialization_times)
        max_time = max(initialization_times)
        
        assert avg_time < 0.1, f"Average lobe initialization time too high: {avg_time}s"
        assert max_time < 0.5, f"Maximum lobe initialization time too high: {max_time}s"
        
        # Cleanup
        for lobe in lobes:
            lobe.shutdown()
    
    def test_lobe_message_passing_performance(self, temp_dir: str):
        """Test message passing performance between lobes."""
        # Create two lobes
        lobe1 = MemoryLobe()
        lobe2 = MemoryLobe()
        
        config1 = {"storage_path": f"{temp_dir}/lobe1.json"}
        config2 = {"storage_path": f"{temp_dir}/lobe2.json"}
        
        assert lobe1.initialize(config1)
        assert lobe2.initialize(config2)
        
        # Connect lobes
        assert lobe1.connect(lobe2, "test_connection")
        assert lobe2.connect(lobe1, "test_connection")
        
        # Test message passing performance
        num_messages = 100
        message_times = []
        
        for i in range(num_messages):
            message = {"id": i, "content": f"Test message {i}"}
            
            start_time = time.time()
            success = lobe1.send_message(lobe2.lobe_id, message)
            end_time = time.time()
            
            message_time = end_time - start_time
            message_times.append(message_time)
            
            assert success
        
        # Analyze performance
        avg_time = statistics.mean(message_times)
        max_time = max(message_times)
        
        assert avg_time < 0.05, f"Average message passing time too high: {avg_time}s"
        assert max_time < 0.2, f"Maximum message passing time too high: {max_time}s"
        
        # Cleanup
        lobe1.shutdown()
        lobe2.shutdown()


class TestSystemLoadPerformance:
    """Test system performance under load."""
    
    def test_system_under_load(self, temp_dir: str):
        """Test system performance under heavy load."""
        # Create multiple components
        memory_lobe = MemoryLobe()
        workflow_lobe = WorkflowLobe()
        context_lobe = ContextLobe()
        
        # Initialize components
        memory_config = {"storage_path": f"{temp_dir}/memory_load.json"}
        workflow_config = {"storage_path": f"{temp_dir}/workflow_load.json"}
        context_config = {"storage_path": f"{temp_dir}/context_load.json"}
        
        assert memory_lobe.initialize(memory_config)
        assert workflow_lobe.initialize(workflow_config)
        assert context_lobe.initialize(context_config)
        
        # Connect components
        assert memory_lobe.connect(workflow_lobe, "workflow")
        assert workflow_lobe.connect(memory_lobe, "memory")
        
        def load_worker(worker_id: int, num_operations: int) -> List[float]:
            """Worker function for load testing."""
            operation_times = []
            
            for i in range(num_operations):
                # Store data
                start_time = time.time()
                result = memory_lobe.process({
                    "action": "store",
                    "data": {"worker": worker_id, "operation": i, "data": f"load_test_{worker_id}_{i}"}
                })
                end_time = time.time()
                operation_times.append(end_time - start_time)
                
                # Create workflow
                start_time = time.time()
                workflow_result = workflow_lobe.process({
                    "action": "create_workflow",
                    "workflow": {
                        "name": f"Load Test Workflow {worker_id}_{i}",
                        "description": "Load test workflow",
                        "steps": [{"name": "Step 1", "action": "test", "parameters": {}}]
                    }
                })
                end_time = time.time()
                operation_times.append(end_time - start_time)
            
            return operation_times
        
        # Run load test
        num_workers = 5
        operations_per_worker = 20
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(load_worker, i, operations_per_worker)
                for i in range(num_workers)
            ]
            
            all_times = []
            for future in concurrent.futures.as_completed(futures):
                worker_times = future.result()
                all_times.extend(worker_times)
        
        # Analyze performance under load
        avg_time = statistics.mean(all_times)
        max_time = max(all_times)
        p95_time = statistics.quantiles(all_times, n=20)[18]  # 95th percentile
        
        # Performance should degrade gracefully under load
        assert avg_time < 0.2, f"Average operation time under load too high: {avg_time}s"
        assert max_time < 2.0, f"Maximum operation time under load too high: {max_time}s"
        assert p95_time < 0.5, f"95th percentile operation time under load too high: {p95_time}s"
        
        # Cleanup
        memory_lobe.shutdown()
        workflow_lobe.shutdown()
        context_lobe.shutdown()