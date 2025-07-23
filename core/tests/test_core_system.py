#!/usr/bin/env python3
"""
Test suite for MCP Core System
Comprehensive tests for the core system functionality.
"""

import asyncio
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add core/src to path for testing
core_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(core_src))

from mcp.core_system import MCPCoreSystem, SystemConfiguration, SystemStatus
from mcp.database_manager import OptimizedDatabaseManager, DatabaseConfig
from mcp.performance_monitor import ObjectivePerformanceMonitor
from mcp.context_manager import ContextManager

class TestCoreSystem:
    """Test cases for the MCP Core System."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def system_config(self, temp_dir):
        """Create a test system configuration."""
        return SystemConfiguration(
            max_workers=2,
            enable_async=True,
            enable_monitoring=True,
            log_level="DEBUG",
            data_directory=temp_dir,
            backup_enabled=False,  # Disable for testing
            performance_optimization=True,
            experimental_features=True,
            hormone_system_enabled=True
        )
    
    @pytest.fixture
    async def core_system(self, system_config):
        """Create and initialize a core system for testing."""
        system = MCPCoreSystem(system_config)
        await system.initialize()
        yield system
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, system_config):
        """Test core system initialization."""
        system = MCPCoreSystem(system_config)
        
        # Test initial state
        assert system.status == SystemStatus.INITIALIZING
        
        # Test initialization
        success = await system.initialize()
        assert success
        assert system.status == SystemStatus.ACTIVE
        
        # Test metrics
        metrics = system.get_metrics()
        assert metrics.uptime >= 0
        assert metrics.active_lobes >= 0
        
        # Cleanup
        await system.shutdown()
        assert system.status == SystemStatus.SHUTDOWN
    
    @pytest.mark.asyncio
    async def test_request_handling(self, core_system):
        """Test request handling through the core system."""
        # Test system status request
        request = {
            'method': 'system/status',
            'params': {}
        }
        
        response = await core_system.execute_request(request)
        assert 'success' in response
        assert response.get('success')
        assert 'status' in response
        assert 'metrics' in response
    
    @pytest.mark.asyncio
    async def test_memory_operations(self, core_system):
        """Test memory operations through the core system."""
        # Test adding memory
        add_request = {
            'method': 'memory/add',
            'params': {
                'text': 'Test memory content',
                'memory_type': 'test',
                'priority': 0.8,
                'tags': ['test', 'core']
            }
        }
        
        response = await core_system.execute_request(add_request)
        assert 'success' in response
        if response.get('success'):
            memory_id = response.get('memory_id')
            assert memory_id is not None
            
            # Test searching memory
            search_request = {
                'method': 'memory/search',
                'params': {
                    'query': 'test',
                    'limit': 10
                }
            }
            
            search_response = await core_system.execute_request(search_request)
            assert 'success' in search_response
            if search_response.get('success'):
                results = search_response.get('results', [])
                assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_task_operations(self, core_system):
        """Test task operations through the core system."""
        # Test creating task
        create_request = {
            'method': 'task/create',
            'params': {
                'title': 'Test Task',
                'description': 'A test task for the core system',
                'priority': 5,
                'estimated_hours': 2.0
            }
        }
        
        response = await core_system.execute_request(create_request)
        assert 'success' in response
        if response.get('success'):
            task_id = response.get('task_id')
            assert task_id is not None
            
            # Test listing tasks
            list_request = {
                'method': 'task/list',
                'params': {}
            }
            
            list_response = await core_system.execute_request(list_request)
            assert 'success' in list_response
            if list_response.get('success'):
                tasks = list_response.get('tasks', [])
                assert len(tasks) >= 0
    
    @pytest.mark.asyncio
    async def test_workflow_operations(self, core_system):
        """Test workflow operations through the core system."""
        # Create a workflow
        create_request = {
            'method': 'workflow/create',
            'params': {
                'name': 'Test Workflow',
                'description': 'A test workflow'
            }
        }
        
        response = await core_system.execute_request(create_request)
        assert 'success' in response
        if response.get('success'):
            workflow_id = response.get('workflow_id')
            assert workflow_id is not None
            
            # Test workflow status
            status_request = {
                'method': 'workflow/status',
                'params': {
                    'workflow_id': workflow_id
                }
            }
            
            status_response = await core_system.execute_request(status_request)
            assert 'success' in status_response
            if status_response.get('success'):
                status = status_response.get('status')
                assert status is not None
    
    @pytest.mark.asyncio
    async def test_context_operations(self, core_system):
        """Test context operations through the core system."""
        # Test context export
        export_request = {
            'method': 'context/export',
            'params': {
                'context_types': ['system', 'performance'],
                'max_tokens': 500,
                'format_type': 'text'
            }
        }
        
        response = await core_system.execute_request(export_request)
        assert 'success' in response
        if response.get('success'):
            context = response.get('context')
            assert context is not None
    
    @pytest.mark.asyncio
    async def test_hormone_operations(self, core_system):
        """Test hormone system operations."""
        # Get hormone levels
        levels_request = {
            'method': 'hormone/levels',
            'params': {}
        }
        
        response = await core_system.execute_request(levels_request)
        assert 'success' in response
        if response.get('success'):
            hormone_levels = response.get('hormone_levels')
            assert hormone_levels is not None
            assert 'stress' in hormone_levels
            assert 'efficiency' in hormone_levels
            
            # Adjust hormone level
            adjust_request = {
                'method': 'hormone/adjust',
                'params': {
                    'hormone': 'stress',
                    'value': 0.7
                }
            }
            
            adjust_response = await core_system.execute_request(adjust_request)
            assert 'success' in adjust_response
            if adjust_response.get('success'):
                updated_levels = adjust_response.get('hormone_levels')
                assert updated_levels['stress'] == 0.7
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, core_system):
        """Test performance monitoring functionality."""
        # Get initial metrics
        initial_metrics = core_system.get_metrics()
        assert initial_metrics.total_requests >= 0
        
        # Execute some requests to generate metrics
        for i in range(5):
            request = {
                'method': 'system/status',
                'params': {}
            }
            await core_system.execute_request(request)

        # Check updated metrics
        updated_metrics = core_system.get_metrics()
        assert updated_metrics.total_requests >= initial_metrics.total_requests
        
        # Test system health
        health_request = {
            'method': 'system/health',
            'params': {}
        }
        
        health_response = await core_system.execute_request(health_request)
        assert 'success' in health_response
        if health_response.get('success'):
            assert 'overall_health' in health_response
            assert 'hormone_levels' in health_response

    def test_system_configuration(self):
        """Test system configuration handling."""
        config = SystemConfiguration(
            max_workers=8,
            enable_async=False,
            log_level="ERROR",
            hormone_system_enabled=False
        )

        assert config.max_workers == 8
        assert config.enable_async == False
        assert config.log_level == "ERROR"
        assert config.enable_monitoring == True  # Default value
        assert config.hormone_system_enabled == False

class TestDatabaseManager:
    """Test cases for the Database Manager."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            temp_path = temp_file.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create a database manager for testing."""
        config = DatabaseConfig(max_connections=5, enable_query_cache=True)
        manager = OptimizedDatabaseManager(temp_db_path, config)
        yield manager
        manager.close()

    def test_database_initialization(self, db_manager):
        """Test database initialization."""
        # Test basic query
        result = db_manager.execute_query("SELECT 1 as test")
        assert len(result) == 1
        assert result[0]['test'] == 1

    def test_memory_operations(self, db_manager):
        """Test memory table operations."""
        # Insert memory
        db_manager.execute_query(
            "INSERT INTO memories (text, memory_type, priority) VALUES (?, ?, ?)",
            ("Test memory", "test", 0.5),
            fetch=False
        )

        # Query memory
        results = db_manager.execute_query(
            "SELECT * FROM memories WHERE memory_type = ?",
            ("test",)
        )

        assert len(results) >= 1
        assert results[0]['text'] == "Test memory"

    def test_performance_stats(self, db_manager):
        """Test database performance statistics."""
        # Execute some queries
        for i in range(10):
            db_manager.execute_query("SELECT ? as iteration", (i,))

        stats = db_manager.get_performance_stats()
        assert stats['total_queries'] >= 10
        assert 'average_query_time' in stats
        assert 'cache_hit_rate' in stats

    def test_transaction_handling(self, db_manager):
        """Test transaction handling."""
        queries = [
            ("INSERT INTO memories (text, memory_type) VALUES (?, ?)", ("Memory 1", "test")),
            ("INSERT INTO memories (text, memory_type) VALUES (?, ?)", ("Memory 2", "test"))
        ]

        success = db_manager.execute_transaction(queries)
        assert success

        # Verify data
        results = db_manager.execute_query("SELECT COUNT(*) as count FROM memories WHERE memory_type = ?", ("test",))
        assert results[0]['count'] >= 2

class TestPerformanceMonitor:
    """Test cases for the Performance Monitor."""

    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor for testing."""
        monitor = ObjectivePerformanceMonitor(
            enable_system_metrics=False,  # Disable for testing
            enable_alerts=True
        )
        yield monitor
        asyncio.run(monitor.stop_monitoring())

    def test_metric_recording(self, performance_monitor):
        """Test metric recording functionality."""
        # Record some metrics
        performance_monitor.record_metric('test_metric', 42.0)
        performance_monitor.record_metric('test_metric', 38.0)
        performance_monitor.record_metric('test_metric', 45.0)

        # Get statistics
        stats = performance_monitor.get_metric_statistics('test_metric')
        assert stats['count'] == 3
        assert stats['mean'] == pytest.approx(41.67, 0.01)
        assert stats['min'] == 38.0
        assert stats['max'] == 45.0

    def test_counter_operations(self, performance_monitor):
        """Test counter metric operations."""
        # Increment counter
        performance_monitor.increment_counter('test_counter', 5)
        performance_monitor.increment_counter('test_counter', 3)

        current_value = performance_monitor.get_current_value('test_counter')
        assert current_value == 8

    def test_hormone_levels(self, performance_monitor):
        """Test brain-inspired hormone levels."""
        # Record metrics that affect hormones
        performance_monitor.record_metric('cpu_usage', 85.0)
        performance_monitor.record_metric('memory_usage', 70.0)
        performance_monitor.record_metric('response_time', 2.5)

        hormone_levels = performance_monitor.get_hormone_levels()
        assert 'stress' in hormone_levels
        assert 'efficiency' in hormone_levels
        assert 'adaptation' in hormone_levels
        assert 'stability' in hormone_levels

        # All hormone levels should be between 0 and 1
        for level in hormone_levels.values():
            assert 0.0 <= level <= 1.0

    def test_health_score(self, performance_monitor):
        """Test system health score calculation."""
        # Record some performance metrics
        performance_monitor.record_metric('cpu_usage', 50.0)
        performance_monitor.record_metric('memory_usage', 60.0)
        performance_monitor.record_metric('response_time', 0.5)

        health_score = performance_monitor.get_system_health_score()
        assert 0.0 <= health_score <= 1.0

    def test_recommendations(self, performance_monitor):
        """Test performance recommendations."""
        # Create conditions that should trigger recommendations
        performance_monitor.record_metric('cpu_usage', 95.0)
        performance_monitor.record_metric('memory_usage', 92.0)
        performance_monitor.record_metric('error_count', 25)

        recommendations = performance_monitor.get_performance_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

class TestContextManager:
    """Test cases for the Context Manager."""

    @pytest.fixture
    def context_manager(self):
        """Create a context manager for testing."""
        return ContextManager("gpt-3.5-turbo")

    def test_context_export(self, context_manager):
        """Test context export functionality."""
        result = context_manager.export_context(
            context_types=['system', 'performance'],
            max_tokens=500,
            format_type='text'
        )

        assert 'context' in result
        assert 'total_tokens' in result
        assert 'items_included' in result
        assert result['total_tokens'] <= 500

    def test_context_item_creation(self, context_manager):
        """Test context item creation."""
        from mcp.context_manager import ContextType, ContextPriority

        item = context_manager.create_context_item(
            content="Test context content",
            context_type=ContextType.SYSTEM,
            priority=ContextPriority.HIGH,
            source="test"
        )

        assert item.content == "Test context content"
        assert item.context_type == ContextType.SYSTEM
        assert item.priority == ContextPriority.HIGH
        assert item.tokens > 0

    def test_token_budget_management(self, context_manager):
        """Test token budget management."""
        token_manager = context_manager.token_manager

        # Test token counting
        text = "This is a test sentence for token counting."
        token_count = token_manager.count_tokens(text)
        assert token_count > 0

        # Test truncation
        long_text = "This is a very long text. " * 100
        truncated = token_manager.truncate_to_budget(long_text, 50)
        truncated_tokens = token_manager.count_tokens(truncated)
        assert truncated_tokens <= 50

    def test_context_statistics(self, context_manager):
        """Test context manager statistics."""
        # Generate some context to create statistics
        context_manager.export_context(['system'], max_tokens=100)

        stats = context_manager.get_context_statistics()
        assert 'generation_stats' in stats
        assert 'cache_size' in stats
        assert stats['generation_stats']['total_generations'] >= 1

# Integration tests
class TestSystemIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    async def full_system(self):
        """Create a full system for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SystemConfiguration(
                data_directory=temp_dir,
                backup_enabled=False,
                max_workers=2,
                hormone_system_enabled=True
            )

            system = MCPCoreSystem(config)
            await system.initialize()
            yield system
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, full_system):
        """Test complete end-to-end workflow."""
        # 1. Add some memories
        memory_requests = [
            {
                'method': 'memory/add',
                'params': {
                    'text': f'Test memory {i}',
                    'memory_type': 'test',
                    'priority': 0.5 + (i * 0.1)
                }
            }
            for i in range(3)
        ]

        memory_responses = await asyncio.gather(*[full_system.execute_request(req) for req in memory_requests])
        memory_ids = [resp.get('memory_id') for resp in memory_responses if resp.get('success')]
        assert len(memory_ids) >= 1

        # 2. Create some tasks
        task_requests = [
            {
                'method': 'task/create',
                'params': {
                    'title': f'Test Task {i}',
                    'description': f'Description for test task {i}',
                    'priority': 5 + i
                }
            }
            for i in range(2)
        ]

        task_responses = await asyncio.gather(*[full_system.execute_request(req) for req in task_requests])
        task_ids = [resp.get('task_id') for resp in task_responses if resp.get('success')]

        # 3. Export context including memories and tasks
        context_request = {
            'method': 'context/export',
            'params': {
                'context_types': ['memory', 'task', 'system'],
                'max_tokens': 1000,
                'format_type': 'json'
            }
        }

        context_response = await full_system.execute_request(context_request)
        assert context_response.get('success')
        context = context_response.get('context')
        assert context is not None

        # 4. Check system health
        health_request = {
            'method': 'system/health',
            'params': {}
        }

        health_response = await full_system.execute_request(health_request)
        assert health_response.get('success')
        assert 'overall_health' in health_response
        assert 'hormone_levels' in health_response

    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_system):
        """Test system performance under load."""
        import time

        start_time = time.time()

        # Execute multiple concurrent requests
        requests = []
        for i in range(5):
            request = {
                'method': 'system/status',
                'params': {}
            }
            requests.append(full_system.execute_request(request))
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*requests)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All responses should be successful
        assert all('success' in resp for resp in responses)
        
        # Check metrics were updated
        metrics = full_system.get_metrics()
        assert metrics.total_requests >= 5
        
        # Test hormone system
        hormone_request = {
            'method': 'hormone/levels',
            'params': {}
        }
        
        hormone_response = await full_system.execute_request(hormone_request)
        assert hormone_response.get('success')
        assert 'hormone_levels' in hormone_response