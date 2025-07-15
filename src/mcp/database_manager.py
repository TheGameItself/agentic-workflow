#!/usr/bin/env python3
"""
Optimized Database Manager
Provides connection pooling, query optimization, and better error handling.
"""

import sqlite3
import threading
import time
import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from queue import Queue, Empty
import logging

def optimize_sqlite_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Pure function to optimize SQLite connection settings for performance and concurrency."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    return conn

class DatabaseConnectionPool:
    """Thread-safe database connection pool with automatic retry logic."""
    
    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._active_connections = 0
        self._logger = logging.getLogger(__name__)
        
        # Initialize pool with connections
        for _ in range(min(3, max_connections)):
            self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level=None  # Enable autocommit mode
        )
        conn = optimize_sqlite_connection(conn)
        return conn
    
    @contextmanager
    def get_connection(self, max_retries: int = 3, retry_delay: float = 0.1):
        """Get a database connection from the pool with retry logic."""
        conn = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Try to get connection from pool
                try:
                    conn = self._pool.get_nowait()
                except Empty:
                    # Pool is empty, create new connection if under limit
                    with self._lock:
                        if self._active_connections < self.max_connections:
                            conn = self._create_connection()
                            self._active_connections += 1
                        else:
                            # Wait for a connection to become available
                            conn = self._pool.get(timeout=1.0)
                
                # Test connection
                conn.execute("SELECT 1")
                
                try:
                    yield conn
                finally:
                    # Return connection to pool
                    if conn:
                        try:
                            conn.execute("SELECT 1")  # Test if connection is still valid
                            self._pool.put(conn)
                        except sqlite3.Error:
                            # Connection is broken, create new one
                            self._active_connections -= 1
                            try:
                                conn.close()
                            except:
                                pass
                            self._create_connection()
                            self._active_connections += 1
                return
                
            except sqlite3.Error as e:
                last_error = e
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                    with self._lock:
                        self._active_connections -= 1
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    break
        
        raise last_error or Exception("Failed to get database connection")

class OptimizedDatabaseManager:
    """Optimized database manager with connection pooling and query optimization."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = DatabaseConnectionPool(db_path)
        self._logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database with optimized schema
        self._init_database()
    
    def _init_database(self):
        """Initialize database with optimized schema and indexes."""
        with self.pool.get_connection() as conn:
            # Create optimized indexes for common queries
            self._create_indexes(conn)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create optimized indexes for better query performance."""
        indexes = [
            # Memory indexes
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
            
            # Task indexes
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_critical ON tasks(accuracy_critical)",
            
            # Task dependencies
            "CREATE INDEX IF NOT EXISTS idx_task_deps_task ON task_dependencies(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_task_deps_depends ON task_dependencies(depends_on_task_id)",
            
            # Task notes
            "CREATE INDEX IF NOT EXISTS idx_task_notes_task ON task_notes(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_task_notes_type ON task_notes(note_type)",
            
            # Advanced memory indexes
            "CREATE INDEX IF NOT EXISTS idx_adv_memories_type ON advanced_memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_adv_memories_quality ON advanced_memories(quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_adv_memories_category ON advanced_memories(category)",
            
            # Context packs
            "CREATE INDEX IF NOT EXISTS idx_context_packs_type ON context_packs(context_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_packs_access ON context_packs(last_accessed)",
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError:
                # Index might already exist
                pass
    
    def execute_query(self, sql: str, params: Optional[tuple] = None, fetch: bool = True) -> List[Dict[str, Any]]:
        """Execute a query with optimized error handling and retry logic."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                if params is not None:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                if fetch:
                    # Get column names
                    columns = [description[0] for description in cursor.description]
                    # Fetch all results
                    rows = cursor.fetchall()
                    # Convert to list of dictionaries
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    conn.commit()
                    return []
                    
            except sqlite3.Error as e:
                self._logger.error(f"Database error: {e}")
                raise
    
    def execute_many(self, sql: str, params_list: List[tuple]) -> None:
        """Execute multiple queries in a batch for better performance."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params_list)
                conn.commit()
            except sqlite3.Error as e:
                self._logger.error(f"Batch execution error: {e}")
                raise
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information."""
        return self.execute_query(f"PRAGMA table_info({table_name})")
    
    def optimize_database(self):
        """Run database optimization commands."""
        with self.pool.get_connection() as conn:
            try:
                conn.execute("VACUUM")  # Rebuild database file
                conn.execute("ANALYZE")  # Update statistics
                conn.execute("REINDEX")  # Rebuild indexes
            except sqlite3.Error as e:
                self._logger.error(f"Database optimization error: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        stats = {}
        
        # Get table sizes
        tables = self.execute_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        
        for table in tables:
            table_name = table['name']
            count_result = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            stats[table_name] = count_result[0]['count'] if count_result else 0
        
        # Get database file size
        try:
            stats['file_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
        except OSError:
            stats['file_size_mb'] = 0
        
        return stats 