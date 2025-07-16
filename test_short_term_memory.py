#!/usr/bin/env python3
"""
Test script for ShortTermMemory class to verify functionality
"""

import sys
import os
import logging
import time
import threading
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Callable, Union
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)

class ShortTermMemory:
    """
    Enhanced short-term memory for recent, high-priority, or volatile information.
    Features:
    - SQLite backend with time-based indexing
    - Automatic cleanup after 30 days with priority-based retention
    - Medium-frequency access patterns with < 1GB capacity management
    - Priority-based retention system
    - Thread-safe operations
    """
    
    def __init__(self, db_path: Optional[str] = None, capacity_gb: float = 1.0, 
                 retention_days: int = 30, fallback: Optional[Callable] = None):
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self.retention_days = retention_days
        self.fallback = fallback
        self.logger = logging.getLogger("ShortTermMemory")
        self._lock = threading.RLock()
        
        # Setup database path
        if db_path:
            self.db_path = db_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, 'short_term_memory.db')
        
        self._init_database()
        self.logger.info(f"[ShortTermMemory] Initialized with {capacity_gb}GB capacity, {retention_days} day retention")
    
    def _init_database(self):
        """Initialize SQLite database with time-based indexing."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main table with time-based indexing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS short_term_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    data TEXT NOT NULL,
                    data_type TEXT DEFAULT 'json',
                    context TEXT DEFAULT 'default',
                    priority REAL DEFAULT 0.5,
                    memory_type TEXT DEFAULT 'general',
                    tags TEXT DEFAULT '[]',
                    size_bytes INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    UNIQUE(key, context)
                )
            """)
            
            # Create time-based indices for efficient queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON short_term_items(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON short_term_items(last_accessed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON short_term_items(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON short_term_items(priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context ON short_term_items(context)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON short_term_items(memory_type)")
            
            # Create composite indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_priority ON short_term_items(context, priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type_created ON short_term_items(memory_type, created_at)")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error initializing database: {ex}")
            raise
    
    def add(self, key: str, item: Any, context: str = "default", priority: float = 0.5,
            memory_type: str = "general", tags: Optional[List[str]] = None, 
            ttl_seconds: Optional[int] = None) -> bool:
        """Add item with priority and automatic capacity management."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                # Serialize data
                if isinstance(item, (dict, list)):
                    data_str = json.dumps(item)
                    data_type = 'json'
                else:
                    data_str = str(item)
                    data_type = 'string'
                
                # Calculate size
                size_bytes = len(data_str.encode('utf-8'))
                
                # Check capacity and cleanup if needed
                self._manage_capacity(size_bytes)
                
                # Calculate expiry
                expires_at = None
                if ttl_seconds:
                    expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
                
                # Prepare tags
                tags_json = json.dumps(tags or [])
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert or replace item
                cursor.execute("""
                    INSERT OR REPLACE INTO short_term_items 
                    (key, data, data_type, context, priority, memory_type, tags, 
                     size_bytes, access_count, created_at, updated_at, last_accessed, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (key, data_str, data_type, context, priority, memory_type, 
                      tags_json, size_bytes, expires_at))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"[ShortTermMemory] Added item '{key}' in context '{context}' ({size_bytes} bytes)")
                return True
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error adding item '{key}': {ex}")
            if self.fallback:
                return self.fallback(key, item, context)
            return False
    
    def get(self, key: str, context: str = "default") -> Optional[Any]:
        """Get item and update access tracking."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get item and check expiry
                cursor.execute("""
                    SELECT data, data_type, expires_at, access_count 
                    FROM short_term_items 
                    WHERE key = ? AND context = ?
                """, (key, context))
                
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                
                data_str, data_type, expires_at, access_count = row
                
                # Check if expired
                if expires_at:
                    expires_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_dt:
                        cursor.execute("DELETE FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                        conn.commit()
                        conn.close()
                        return None
                
                # Update access tracking
                cursor.execute("""
                    UPDATE short_term_items 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE key = ? AND context = ?
                """, (key, context))
                
                conn.commit()
                conn.close()
                
                # Deserialize data
                if data_type == 'json':
                    return json.loads(data_str)
                else:
                    return data_str
                    
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting item '{key}': {ex}")
            return None
    
    def _manage_capacity(self, new_item_size: int):
        """Manage capacity by removing low-priority old items."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current size
            cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM short_term_items")
            current_size = cursor.fetchone()[0]
            
            # If adding new item would exceed capacity, remove items
            while current_size + new_item_size > self.capacity_bytes:
                # Remove lowest priority, least accessed items first
                cursor.execute("""
                    SELECT key, context, size_bytes FROM short_term_items 
                    ORDER BY priority ASC, access_count ASC, last_accessed ASC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if not row:
                    break  # No more items to remove
                
                key, context, size_bytes = row
                cursor.execute("DELETE FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                current_size -= size_bytes
                
                self.logger.debug(f"[ShortTermMemory] Removed item '{key}' for capacity management")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error managing capacity: {ex}")
    
    def cleanup_expired(self) -> int:
        """Remove expired items and old items based on retention policy."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Remove explicitly expired items
                cursor.execute("DELETE FROM short_term_items WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP")
                expired_count = cursor.rowcount
                
                # Remove items older than retention period (but keep high priority items)
                cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
                cursor.execute("""
                    DELETE FROM short_term_items 
                    WHERE created_at < ? AND priority < 0.8
                """, (cutoff_date,))
                old_count = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                total_removed = expired_count + old_count
                if total_removed > 0:
                    self.logger.info(f"[ShortTermMemory] Cleaned up {total_removed} items ({expired_count} expired, {old_count} old)")
                
                return total_removed
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error during cleanup: {ex}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM short_term_items")
                total_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT context) FROM short_term_items")
                total_contexts = cursor.fetchone()[0]
                
                # Size information
                cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM short_term_items")
                current_size = cursor.fetchone()[0]
                
                # Access statistics
                cursor.execute("SELECT AVG(access_count) FROM short_term_items")
                avg_access = cursor.fetchone()[0] or 0
                
                conn.close()
                
                return {
                    'total_items': total_items,
                    'total_contexts': total_contexts,
                    'current_size_bytes': current_size,
                    'current_size_mb': round(current_size / (1024 * 1024), 2),
                    'capacity_gb': round(self.capacity_bytes / (1024 * 1024 * 1024), 2),
                    'utilization_percent': round((current_size / self.capacity_bytes) * 100, 2),
                    'average_access_count': round(avg_access, 2),
                    'retention_days': self.retention_days
                }
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting stats: {ex}")
            return {'error': str(ex)}


def test_short_term_memory():
    """Test ShortTermMemory functionality"""
    print("Testing ShortTermMemory...")
    
    # Initialize
    stm = ShortTermMemory(capacity_gb=0.001)  # 1MB for testing
    
    # Test adding items
    print("Adding test items...")
    stm.add("test1", {"data": "test data 1"}, priority=0.8)
    stm.add("test2", "simple string data", context="testing", priority=0.6)
    stm.add("test3", [1, 2, 3, 4, 5], memory_type="list", tags=["numbers"])
    
    # Test retrieval
    print("Testing retrieval...")
    item1 = stm.get("test1")
    print(f"Retrieved item1: {item1}")
    
    item2 = stm.get("test2", "testing")
    print(f"Retrieved item2: {item2}")
    
    # Test stats
    print("Getting stats...")
    stats = stm.get_stats()
    print(f"Stats: {stats}")
    
    # Test cleanup
    print("Testing cleanup...")
    removed = stm.cleanup_expired()
    print(f"Removed {removed} expired items")
    
    print("ShortTermMemory test completed successfully!")


if __name__ == "__main__":
    test_short_term_memory()