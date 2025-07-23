#!/usr/bin/env python3
"""
Optimized Database Manager for MCP Core System
High-performance database operations with connection pooling, caching, and optimization.
"""

import sqlite3
import threading
import time
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Iterator
from queue import Queue, Empty
import hashlib
from functools import lru_cache

class DatabaseConfig:
    """Database configuration settings."""
    max_connections: int = 10
    connection_timeout: int = 30
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    cache_size: int = 10000  # pages
    temp_store: str = "memory"
    synchronous: str = "normal"
    journal_mode: str = "wal"
    auto_vacuum: str = "incremental"
    enable_query_cache: bool = True
    query_cache_size: int = 1000

class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, config: DatabaseConfig):
        self.db_path = db_path
        self.config = config
        self.pool = Queue(maxsize=config.max_connections)
        self.active_connections = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize pool with connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        for _ in range(self.config.max_connections):
            conn = self._create_connection()
            self.pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new optimized SQLite connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.config.connection_timeout,
            check_same_thread=False
        )

        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row

        # Apply optimizations
        cursor = conn.cursor()

        if self.config.enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys = ON")

        cursor.execute(f"PRAGMA cache_size = -{self.config.cache_size}")
        cursor.execute(f"PRAGMA temp_store = {self.config.temp_store}")
        cursor.execute(f"PRAGMA synchronous = {self.config.synchronous}")
        cursor.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
        cursor.execute(f"PRAGMA auto_vacuum = {self.config.auto_vacuum}")

        # Enable memory-mapped I/O for better performance
        cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB

        conn.commit()
        return conn

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            # Try to get connection from pool
            try:
                conn = self.pool.get(timeout=5)
            except Empty:
                # Create new connection if pool is empty
                with self.lock:
                    if self.active_connections < self.config.max_connections:
                        for _ in range(min(3, self.config.max_connections)):  # Start with fewer connections
                            conn = self._create_connection()
                            self.pool.put(conn)
                            self.active_connections += 1
                    else:
                        # Wait for connection to become available
                        conn = self.pool.get(timeout=self.config.connection_timeout)

            yield conn

        finally:
            if conn:
                # Return connection to pool
                try:
                    self.pool.put(conn, timeout=1)
                except:
                    # Pool is full, close connection
                    conn.close()
                    with self.lock:
                        self.active_connections -= 1

    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break

        with self.lock:
            self.active_connections = 0

class QueryCache:
    """LRU cache for database queries."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, params: tuple) -> str:
        """Create cache key from query and parameters."""
        key_data = f"{query}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, params: tuple) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result."""
        key = self._make_key(query, params)

        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1

        return None

    def put(self, query: str, params: tuple, result: List[Dict[str, Any]]):
        """Cache query result."""
        # Don't cache empty results or very large result sets
        if not result or len(result) > 1000:
            return
            
        key = self._make_key(query, params)

        with self.lock:
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]

            # Add/update cache
            if key in self.cache:
                self.access_order.remove(key)

            self.cache[key] = result
            self.access_order.append(key)

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0

class OptimizedDatabaseManager:
    """
    High-performance database manager with advanced features.

    Features:
    - Connection pooling for concurrent access
    - Query result caching
    - Automatic schema management
    - Performance monitoring
    - Backup and recovery
    - Query optimization
    """

    # SQL statements for schema initialization
    SCHEMA_SQL = {
        'memories': '''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                memory_type TEXT DEFAULT 'general',
                priority REAL DEFAULT 0.5,
                context TEXT,
                tags TEXT,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                memory_order INTEGER DEFAULT 1,
                vector_embedding BLOB,  -- For vector storage
                metadata TEXT  -- JSON metadata
            )
        ''',
        'tasks': '''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                parent_id INTEGER,
                estimated_hours REAL DEFAULT 0.0,
                actual_hours REAL DEFAULT 0.0,
                progress REAL DEFAULT 0.0,
                accuracy_critical BOOLEAN DEFAULT FALSE,
                due_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                tags TEXT,  -- JSON array
                metadata TEXT,  -- JSON metadata
                FOREIGN KEY (parent_id) REFERENCES tasks (id)
            )
        ''',
        'workflows': '''
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'not_started',
                current_step TEXT,
                progress REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                metadata TEXT  -- JSON metadata
            )
        ''',
        'workflow_steps': '''
            CREATE TABLE IF NOT EXISTS workflow_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'not_started',
                order_index INTEGER DEFAULT 0,
                dependencies TEXT,  -- JSON array of step names
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                metadata TEXT,  -- JSON metadata
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        ''',
        'projects': '''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                path TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config TEXT,  -- JSON configuration
                metadata TEXT  -- JSON metadata
            )
        ''',
        'context_packs': '''
            CREATE TABLE IF NOT EXISTS context_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                context_data TEXT NOT NULL,  -- JSON context data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                metadata TEXT  -- JSON metadata
            )
        ''',
        'performance_metrics': '''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                component TEXT,
                metadata TEXT  -- JSON metadata
            )
        '''
    }
    
    # Indexes for performance
    INDEXES_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories (priority)",
        "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (created_at)",
        "CREATE INDEX IF NOT EXISTS idx_memories_order ON memories (memory_order)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks (priority)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks (parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks (due_date)",
        "CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows (status)",
        "CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow ON workflow_steps (workflow_id)",
        "CREATE INDEX IF NOT EXISTS idx_projects_status ON projects (status)",
        "CREATE INDEX IF NOT EXISTS idx_context_packs_created ON context_packs (created_at)",
        "CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics (metric_name)",
        "CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics (timestamp)"
    ]

    def __init__(self, db_path: str, config: Optional[DatabaseConfig] = None):
        self.db_path = Path(db_path)
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.connection_pool = ConnectionPool(str(self.db_path), self.config)
        self.query_cache = QueryCache(self.config.query_cache_size) if self.config.enable_query_cache else None

        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'slow_queries': []
        }

        # Initialize database schema
        self._initialize_schema()

        self.logger.info(f"Optimized database manager initialized: {self.db_path}")

    def _initialize_schema(self):
        """Initialize database schema with all required tables."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Create tables from schema definitions
            for table_name, schema_sql in self.SCHEMA_SQL.items():
                cursor.execute(schema_sql)

            # Create indexes for performance
            for index_sql in self.INDEXES_SQL:
                cursor.execute(index_sql)

            conn.commit()

    def execute_query(self, query: str, params: tuple = (), fetch: bool = True,
                     use_cache: bool = True) -> Union[List[Dict[str, Any]], int]:
        """
        Execute a database query with caching and performance tracking.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            use_cache: Whether to use query cache

        Returns:
            Query results or affected row count
        """
        start_time = time.time()

        # Check cache for SELECT queries
        if fetch and use_cache and self.query_cache and query.strip().upper().startswith('SELECT'):
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                self.query_stats['cache_hits'] += 1
                return cached_result
            else:
                self.query_stats['cache_misses'] += 1

        # Execute query
        with self.connection_pool.get_connection() as conn:
            # Apply optimizations
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if fetch:
                # Fetch results and convert to list of dicts
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]

                # Cache SELECT query results
                if use_cache and self.query_cache and query.strip().upper().startswith('SELECT'):
                    self.query_cache.put(query, params, result)

                # Update performance stats
                execution_time = time.time() - start_time
                self.query_stats['total_queries'] += 1
                self.query_stats['total_time'] += execution_time

                # Track slow queries
                if execution_time > 1.0:  # Queries taking more than 1 second
                    self.query_stats['slow_queries'].append({
                        'query': query,
                        'params': params,
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Keep only last 100 slow queries
                    if len(self.query_stats['slow_queries']) > 100:
                        self.query_stats['slow_queries'].pop(0)

                return result
            else:
                # Return affected row count
                conn.commit()
                
                # Update performance stats
                execution_time = time.time() - start_time
                self.query_stats['total_queries'] += 1
                self.query_stats['total_time'] += execution_time
                
                return cursor.rowcount

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute a query with multiple parameter sets."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def execute_transaction(self, queries: List[Tuple[str, tuple]]) -> bool:
        """Execute multiple queries in a transaction."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                for query, params in queries:
                    cursor.execute(query, params)

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            return False

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            with self.connection_pool.get_connection() as source_conn:
                backup_conn = sqlite3.connect(str(backup_path))
                source_conn.backup(backup_conn)
                backup_conn.close()

            self.logger.info(f"Database backup created: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False

    def optimize_database(self):
        """Optimize database performance."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                # Vacuum to reclaim space and defragment
                cursor.execute("VACUUM")

                # Update statistics for query optimizer
                cursor.execute("ANALYZE")

                # Incremental vacuum if enabled
                if self.config.auto_vacuum == "incremental":
                    cursor.execute("PRAGMA incremental_vacuum")

                conn.commit()

            # Clear query cache to ensure fresh results
            if self.query_cache:
                self.query_cache.clear()

            self.logger.info("Database optimization completed")

        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        stats = self.query_stats.copy()

        # Calculate additional metrics
        if stats['total_queries'] > 0:
            stats['average_query_time'] = stats['total_time'] / stats['total_queries']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        else:
            stats['average_query_time'] = 0
            stats['cache_hit_rate'] = 0

        # Add database size info
        try:
            db_size = self.db_path.stat().st_size
            stats['database_size_bytes'] = db_size
            stats['database_size_mb'] = db_size / (1024 * 1024)
        except:
            stats['database_size_bytes'] = 0
            stats['database_size_mb'] = 0

        return stats

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table."""
        schema_result = self.execute_query(f"PRAGMA table_info({table_name})")
        index_result = self.execute_query(f"PRAGMA index_list({table_name})")
        
        return {
            'columns': schema_result,
            'indexes': index_result,
            'row_count': self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")[0]['count']
        }
    
    def close(self):
        """Close the database manager and all connections."""
        if self.connection_pool:
            self.connection_pool.close_all()
        
        self.logger.info(f"Database manager closed: {self.db_path}")