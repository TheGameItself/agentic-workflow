#!/usr/bin/env python3
"""
Test suite for MCP Database Manager
Comprehensive tests for the database manager functionality.
"""

import pytest
import tempfile
import os
import sys
import json
import time
import threading
from pathlib import Path
import sqlite3

# Add core/src to path for testing
core_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(core_src))

from mcp.database_manager import OptimizedDatabaseManager, DatabaseConfig, ConnectionPool, QueryCache


class TestDatabaseManager:
    """Test cases for the OptimizedDatabaseManager."""
    
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
    def db_config(self):
        """Create a test database configuration."""
        return DatabaseConfig(
            max_connections=5,
            connection_timeout=5,
            enable_wal_mode=True,
            enable_foreign_keys=True,
            cache_size=2000,
            temp_store="memory",
            synchronous="normal",
            journal_mode="wal",
            auto_vacuum="incremental",
            enable_query_cache=True,
            query_cache_size=500
        )
    
    @pytest.fixture
    def db_manager(self, temp_db_path, db_config):
        """Create a database manager for testing."""
        manager = OptimizedDatabaseManager(temp_db_path, db_config)
        yield manager
        # Cleanup
        manager.connection_pool.close_all()
    
    def test_database_initialization(self, db_manager):
        """Test database initialization."""
        # Test basic query
        result = db_manager.execute_query("SELECT 1 as test")
        assert len(result) == 1
        assert result[0]['test'] == 1
        
        # Check if tables were created
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = db_manager.execute_query(tables_query)
        table_names = [table['name'] for table in tables]
        
        # Check for core tables
        assert 'memories' in table_names
        assert 'tasks' in table_names
        assert 'workflows' in table_names
        assert 'projects' in table_names
        assert 'context_packs' in table_names
        assert 'performance_metrics' in table_names
    
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
        
        assert len(results) == 1
        assert results[0]['text'] == "Test memory"
        assert results[0]['memory_type'] == "test"
        assert results[0]['priority'] == 0.5
    
    def test_task_operations(self, db_manager):
        """Test task table operations."""
        # Insert task
        db_manager.execute_query(
            "INSERT INTO tasks (title, description, priority, estimated_hours) VALUES (?, ?, ?, ?)",
            ("Test task", "Task description", 3, 2.5),
            fetch=False
        )
        
        # Query task
        results = db_manager.execute_query(
            "SELECT * FROM tasks WHERE title = ?",
            ("Test task",)
        )
        
        assert len(results) == 1
        assert results[0]['title'] == "Test task"
        assert results[0]['description'] == "Task description"
        assert results[0]['priority'] == 3
        assert results[0]['estimated_hours'] == 2.5
    
    def test_workflow_operations(self, db_manager):
        """Test workflow table operations."""
        # Insert workflow
        db_manager.execute_query(
            "INSERT INTO workflows (name, description) VALUES (?, ?)",
            ("Test workflow", "Workflow description"),
            fetch=False
        )
        
        # Get the workflow ID
        workflow_results = db_manager.execute_query(
            "SELECT id FROM workflows WHERE name = ?",
            ("Test workflow",)
        )
        workflow_id = workflow_results[0]['id']
        
        # Insert workflow step
        db_manager.execute_query(
            "INSERT INTO workflow_steps (workflow_id, name, description, order_index) VALUES (?, ?, ?, ?)",
            (workflow_id, "Step 1", "First step", 0),
            fetch=False
        )
        
        # Query workflow steps
        step_results = db_manager.execute_query(
            "SELECT * FROM workflow_steps WHERE workflow_id = ?",
            (workflow_id,)
        )
        
        assert len(step_results) == 1
        assert step_results[0]['name'] == "Step 1"
        assert step_results[0]['order_index'] == 0
    
    def test_project_operations(self, db_manager):
        """Test project table operations."""
        # Insert project with JSON config
        config = {"language": "python", "version": "3.9", "settings": {"debug": True}}
        db_manager.execute_query(
            "INSERT INTO projects (name, description, path, config) VALUES (?, ?, ?, ?)",
            ("Test project", "Project description", "/path/to/project", json.dumps(config)),
            fetch=False
        )
        
        # Query project
        results = db_manager.execute_query(
            "SELECT * FROM projects WHERE name = ?",
            ("Test project",)
        )
        
        assert len(results) == 1
        assert results[0]['name'] == "Test project"
        assert results[0]['path'] == "/path/to/project"
        
        # Parse JSON config
        stored_config = json.loads(results[0]['config'])
        assert stored_config['language'] == "python"
        assert stored_config['settings']['debug'] is True
    
    def test_context_pack_operations(self, db_manager):
        """Test context pack table operations."""
        # Create context data
        context_data = {
            "system_info": {"version": "1.0", "status": "active"},
            "memory_items": [{"id": 1, "text": "Memory item 1"}, {"id": 2, "text": "Memory item 2"}]
        }
        
        # Insert context pack
        db_manager.execute_query(
            "INSERT INTO context_packs (name, description, context_data) VALUES (?, ?, ?)",
            ("Test pack", "Context pack description", json.dumps(context_data)),
            fetch=False
        )
        
        # Query context pack
        results = db_manager.execute_query(
            "SELECT * FROM context_packs WHERE name = ?",
            ("Test pack",)
        )
        
        assert len(results) == 1
        assert results[0]['name'] == "Test pack"
        
        # Parse JSON context data
        stored_context = json.loads(results[0]['context_data'])
        assert stored_context['system_info']['version'] == "1.0"
        assert len(stored_context['memory_items']) == 2
    
    def test_performance_metrics(self, db_manager):
        """Test performance metrics table operations."""
        # Insert metrics
        db_manager.execute_query(
            "INSERT INTO performance_metrics (metric_name, metric_value, component) VALUES (?, ?, ?)",
            ("cpu_usage", 45.2, "core_system"),
            fetch=False
        )
        
        db_manager.execute_query(
            "INSERT INTO performance_metrics (metric_name, metric_value, component) VALUES (?, ?, ?)",
            ("memory_usage", 128.5, "memory_lobe"),
            fetch=False
        )
        
        # Query metrics
        results = db_manager.execute_query(
            "SELECT * FROM performance_metrics ORDER BY id"
        )
        
        assert len(results) == 2
        assert results[0]['metric_name'] == "cpu_usage"
        assert results[0]['metric_value'] == 45.2
        assert results[1]['metric_name'] == "memory_usage"
        assert results[1]['component'] == "memory_lobe"
    
    def test_query_cache(self, db_manager):
        """Test query caching functionality."""
        # Execute a query that should be cached
        query = "SELECT * FROM sqlite_master WHERE type='table'"
        
        # First execution (cache miss)
        start_time = time.time()
        first_result = db_manager.execute_query(query)
        first_execution_time = time.time() - start_time
        
        # Second execution (should be cache hit)
        start_time = time.time()
        second_result = db_manager.execute_query(query)
        second_execution_time = time.time() - start_time
        
        # Results should be the same
        assert len(first_result) == len(second_result)
        
        # Get performance stats
        stats = db_manager.get_performance_stats()
        assert stats['cache_hits'] >= 1
        
        # Clear cache and try again
        if db_manager.query_cache:
            db_manager.query_cache.clear()
            
            # Execute again (should be cache miss)
            third_result = db_manager.execute_query(query)
            assert len(third_result) == len(first_result)
            
            # Check updated stats
            updated_stats = db_manager.get_performance_stats()
            assert updated_stats['cache_misses'] > stats['cache_misses']
    
    def test_execute_many(self, db_manager):
        """Test executemany functionality."""
        # Prepare multiple inserts
        data = [
            ("Memory 1", "test", 0.1),
            ("Memory 2", "test", 0.2),
            ("Memory 3", "test", 0.3),
            ("Memory 4", "test", 0.4),
            ("Memory 5", "test", 0.5)
        ]
        
        # Execute many
        rows_affected = db_manager.execute_many(
            "INSERT INTO memories (text, memory_type, priority) VALUES (?, ?, ?)",
            data
        )
        
        assert rows_affected == 5
        
        # Query to verify
        results = db_manager.execute_query(
            "SELECT * FROM memories WHERE memory_type = ? ORDER BY priority",
            ("test",)
        )
        
        assert len(results) == 5
        assert results[0]['text'] == "Memory 1"
        assert results[0]['priority'] == 0.1
        assert results[4]['text'] == "Memory 5"
        assert results[4]['priority'] == 0.5
    
    def test_transaction(self, db_manager):
        """Test transaction functionality."""
        # Prepare transaction queries
        queries = [
            ("INSERT INTO memories (text, memory_type, priority) VALUES (?, ?, ?)", 
             ("Transaction memory 1", "transaction", 0.7)),
            ("INSERT INTO tasks (title, description, priority) VALUES (?, ?, ?)",
             ("Transaction task", "Created in transaction", 2)),
            ("INSERT INTO workflows (name, description) VALUES (?, ?)",
             ("Transaction workflow", "Created in transaction"))
        ]
        
        # Execute transaction
        success = db_manager.execute_transaction(queries)
        assert success
        
        # Verify results
        memory_results = db_manager.execute_query(
            "SELECT * FROM memories WHERE memory_type = ?",
            ("transaction",)
        )
        assert len(memory_results) == 1
        
        task_results = db_manager.execute_query(
            "SELECT * FROM tasks WHERE title = ?",
            ("Transaction task",)
        )
        assert len(task_results) == 1
        
        workflow_results = db_manager.execute_query(
            "SELECT * FROM workflows WHERE name = ?",
            ("Transaction workflow",)
        )
        assert len(workflow_results) == 1
    
    def test_backup_database(self, db_manager, temp_db_path):
        """Test database backup functionality."""
        # Insert some data
        db_manager.execute_query(
            "INSERT INTO memories (text, memory_type) VALUES (?, ?)",
            ("Backup test memory", "backup_test"),
            fetch=False
        )
        
        # Create backup path
        backup_path = temp_db_path + ".backup"
        
        # Perform backup
        success = db_manager.backup_database(backup_path)
        assert success
        assert os.path.exists(backup_path)
        
        # Verify backup by connecting to it directly
        conn = sqlite3.connect(backup_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE memory_type = ?", ("backup_test",))
        results = cursor.fetchall()
        conn.close()
        
        assert len(results) == 1
        assert results[0]['text'] == "Backup test memory"
        
        # Clean up backup file
        if os.path.exists(backup_path):
            os.unlink(backup_path)
    
    def test_optimize_database(self, db_manager):
        """Test database optimization."""
        # Insert and delete some data to create fragmentation
        for i in range(100):
            db_manager.execute_query(
                "INSERT INTO memories (text, memory_type) VALUES (?, ?)",
                (f"Temp memory {i}", "temp"),
                fetch=False
            )
        
        # Delete half the records
        db_manager.execute_query(
            "DELETE FROM memories WHERE text LIKE ? AND CAST(substr(text, 13) AS INTEGER) < 50",
            ("Temp memory %",),
            fetch=False
        )
        
        # Run optimization
        db_manager.optimize_database()
        
        # Verify remaining records
        results = db_manager.execute_query(
            "SELECT COUNT(*) as count FROM memories WHERE memory_type = ?",
            ("temp",)
        )
        assert results[0]['count'] == 50
    
    def test_performance_stats(self, db_manager):
        """Test performance statistics collection."""
        # Execute some queries to generate stats
        for i in range(10):
            db_manager.execute_query("SELECT 1")
            db_manager.execute_query("SELECT * FROM sqlite_master")
        
        # Get performance stats
        stats = db_manager.get_performance_stats()
        
        assert stats['total_queries'] >= 20
        assert 'average_query_time' in stats
        assert 'cache_hit_rate' in stats
        assert 'database_size_bytes' in stats
        assert stats['database_size_bytes'] > 0
    
    def test_get_table_info(self, db_manager):
        """Test getting table information."""
        # Get info for memories table
        schema_result = db_manager.execute_query("PRAGMA table_info(memories)")
        
        # Verify schema information
        column_names = [col['name'] for col in schema_result]
        assert 'id' in column_names
        assert 'text' in column_names
        assert 'memory_type' in column_names
        assert 'priority' in column_names
        assert 'created_at' in column_names
    
    def test_concurrent_access(self, db_manager):
        """Test concurrent database access."""
        # Number of concurrent operations
        num_threads = 10
        operations_per_thread = 10
        
        # Function to run in each thread
        def worker(thread_id):
            for i in range(operations_per_thread):
                try:
                    # Insert a memory
                    db_manager.execute_query(
                        "INSERT INTO memories (text, memory_type, priority) VALUES (?, ?, ?)",
                        (f"Thread {thread_id} Memory {i}", f"thread_{thread_id}", 0.5),
                        fetch=False
                    )
                    
                    # Read memories
                    db_manager.execute_query(
                        "SELECT * FROM memories WHERE memory_type = ?",
                        (f"thread_{thread_id}",)
                    )
                except Exception as e:
                    print(f"Thread {thread_id} error: {e}")
                    return False
            return True
        
        # Create and start threads
        threads = []
        results = [None] * num_threads
        
        for i in range(num_threads):
            thread = threading.Thread(target=lambda idx=i: results.__setitem__(idx, worker(idx)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check if all threads completed successfully
        assert all(results)
        
        # Verify data
        for i in range(num_threads):
            count_result = db_manager.execute_query(
                "SELECT COUNT(*) as count FROM memories WHERE memory_type = ?",
                (f"thread_{i}",)
            )
            assert count_result[0]['count'] == operations_per_thread


class TestConnectionPool:
    """Test cases for the ConnectionPool."""
    
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
    def db_config(self):
        """Create a test database configuration."""
        return DatabaseConfig(
            max_connections=3,
            connection_timeout=5
        )
    
    @pytest.fixture
    def connection_pool(self, temp_db_path, db_config):
        """Create a connection pool for testing."""
        # Create database file
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.close()
        
        # Create pool
        pool = ConnectionPool(temp_db_path, db_config)
        yield pool
        pool.close_all()
    
    def test_pool_initialization(self, connection_pool, db_config):
        """Test connection pool initialization."""
        # Check if pool was initialized with correct number of connections
        assert connection_pool.pool.qsize() == db_config.max_connections
    
    def test_get_connection(self, connection_pool):
        """Test getting a connection from the pool."""
        with connection_pool.get_connection() as conn:
            # Test if connection is valid
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        # Connection should be returned to pool
        assert connection_pool.pool.qsize() > 0
    
    def test_multiple_connections(self, connection_pool, db_config):
        """Test getting multiple connections from the pool."""
        connections = []
        
        # Get all connections from pool
        for _ in range(db_config.max_connections):
            with connection_pool.get_connection() as conn:
                connections.append(conn)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1
        
        # Pool should be empty now
        assert connection_pool.pool.qsize() == 0
    
    def test_connection_reuse(self, connection_pool):
        """Test connection reuse."""
        # Get and release a connection
        with connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
        
        # Get another connection (should be the same one)
        with connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM test WHERE value = ?", ("test_value",))
            result = cursor.fetchone()
            assert result[0] == "test_value"
    
    def test_close_all(self, connection_pool, db_config):
        """Test closing all connections."""
        # Get some connections first
        connections = []
        for _ in range(2):
            with connection_pool.get_connection() as conn:
                connections.append(conn)
        
        # Close all connections
        connection_pool.close_all()
        
        # Pool should be empty
        assert connection_pool.pool.qsize() == 0
        assert connection_pool.active_connections == 0


class TestQueryCache:
    """Test cases for the QueryCache."""
    
    @pytest.fixture
    def query_cache(self):
        """Create a query cache for testing."""
        return QueryCache(max_size=5)
    
    def test_cache_operations(self, query_cache):
        """Test basic cache operations."""
        # Create test data
        query = "SELECT * FROM test WHERE id = ?"
        params = (1,)
        result = [{"id": 1, "name": "Test"}]
        
        # Initially cache should be empty
        assert query_cache.get(query, params) is None
        
        # Add to cache
        query_cache.put(query, params, result)
        
        # Get from cache
        cached_result = query_cache.get(query, params)
        assert cached_result == result
        
        # Different params should be a cache miss
        assert query_cache.get(query, (2,)) is None
        
        # Different query should be a cache miss
        assert query_cache.get("SELECT * FROM other_table", params) is None
    
    def test_cache_eviction(self, query_cache):
        """Test cache eviction when full."""
        # Fill cache to capacity
        for i in range(5):
            query = f"SELECT * FROM test WHERE id = {i}"
            params = (i,)
            result = [{"id": i, "name": f"Test {i}"}]
            query_cache.put(query, params, result)
        
        # Add one more item to trigger eviction
        query_cache.put(
            "SELECT * FROM test WHERE id = 10",
            (10,),
            [{"id": 10, "name": "Test 10"}]
        )
        
        # First item should be evicted
        assert query_cache.get("SELECT * FROM test WHERE id = 0", (0,)) is None
        
        # Last item should be in cache
        assert query_cache.get("SELECT * FROM test WHERE id = 10", (10,)) is not None
    
    def test_cache_clear(self, query_cache):
        """Test clearing the cache."""
        # Add some items to cache
        for i in range(3):
            query = f"SELECT * FROM test WHERE id = {i}"
            params = (i,)
            result = [{"id": i, "name": f"Test {i}"}]
            query_cache.put(query, params, result)
        
        # Clear cache
        query_cache.clear()
        
        # Cache should be empty
        for i in range(3):
            query = f"SELECT * FROM test WHERE id = {i}"
            params = (i,)
            assert query_cache.get(query, params) is None


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])