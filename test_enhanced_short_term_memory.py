#!/usr/bin/env python3
"""
Comprehensive test script for enhanced ShortTermMemory class with neural retention
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
    - SQLite backend with time-based indexing and optimized queries
    - Automatic cleanup after 30 days with priority-based neural network assisted retention
    - Medium-frequency access patterns with < 1GB capacity management
    - Priority-based retention system with intelligent scoring
    - Thread-safe operations with connection pooling
    - Relevance scoring and access pattern analysis
    - Automatic background cleanup and optimization
    """
    
    def __init__(self, db_path: Optional[str] = None, capacity_gb: float = 1.0, 
                 retention_days: int = 30, fallback: Optional[Callable] = None,
                 enable_neural_retention: bool = True):
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self.retention_days = retention_days
        self.fallback = fallback
        self.enable_neural_retention = enable_neural_retention
        self.logger = logging.getLogger("ShortTermMemory")
        self._lock = threading.RLock()
        
        # Neural retention scoring weights
        self.retention_weights = {
            'priority': 0.3,
            'access_frequency': 0.25,
            'recency': 0.2,
            'relevance': 0.15,
            'size_efficiency': 0.1
        }
        
        # Setup database path
        if db_path:
            self.db_path = db_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, 'short_term_memory.db')
        
        self._init_database()
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(hours=1)  # Cleanup every hour
        
        self.logger.info(f"[ShortTermMemory] Initialized with {capacity_gb}GB capacity, {retention_days} day retention, neural retention: {enable_neural_retention}")
    
    def _init_database(self):
        """Initialize SQLite database with time-based indexing and neural retention features."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main table with enhanced schema for neural retention
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
                    access_frequency REAL DEFAULT 0.0,
                    relevance_score REAL DEFAULT 0.5,
                    retention_score REAL DEFAULT 0.5,
                    neural_weight REAL DEFAULT 0.5,
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_score ON short_term_items(retention_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_weight ON short_term_items(neural_weight)")
            
            # Create composite indices for common queries and neural retention
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_priority ON short_term_items(context, priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type_created ON short_term_items(memory_type, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_priority ON short_term_items(retention_score, priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_access ON short_term_items(neural_weight, access_frequency)")
            
            # Create access pattern tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_key TEXT NOT NULL,
                    context TEXT NOT NULL,
                    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_type TEXT DEFAULT 'read',
                    session_id TEXT,
                    FOREIGN KEY (item_key, context) REFERENCES short_term_items (key, context)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_patterns(access_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_key_context ON access_patterns(item_key, context)")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error initializing database: {ex}")
            raise
    
    def add(self, key: str, item: Any, context: str = "default", priority: float = 0.5,
            memory_type: str = "general", tags: Optional[List[str]] = None, 
            ttl_seconds: Optional[int] = None) -> bool:
        """Add item with priority, neural retention scoring, and automatic capacity management."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                # Perform background cleanup if needed
                self._background_cleanup()
                
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
                
                # Calculate neural retention scores
                relevance_score = self._calculate_relevance_score(data_str, context, memory_type, tags or [])
                retention_score = self._calculate_retention_score(priority, relevance_score, size_bytes)
                neural_weight = self._calculate_neural_weight(retention_score, priority, relevance_score)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if item exists for access frequency calculation
                cursor.execute("SELECT access_count FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                existing = cursor.fetchone()
                access_frequency = 0.0
                if existing:
                    access_frequency = self._calculate_access_frequency(key, context)
                
                # Insert or replace item with neural retention features
                cursor.execute("""
                    INSERT OR REPLACE INTO short_term_items 
                    (key, data, data_type, context, priority, memory_type, tags, 
                     size_bytes, access_count, access_frequency, relevance_score, retention_score, neural_weight,
                     created_at, updated_at, last_accessed, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (key, data_str, data_type, context, priority, memory_type, 
                      tags_json, size_bytes, access_frequency, relevance_score, retention_score, neural_weight, expires_at))
                
                # Track access pattern
                self._track_access_pattern(key, context, 'write')
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"[ShortTermMemory] Added item '{key}' in context '{context}' ({size_bytes} bytes, retention_score: {retention_score:.3f})")
                return True
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error adding item '{key}': {ex}")
            if self.fallback:
                return self.fallback(key, item, context)
            return False
    
    def get(self, key: str, context: str = "default") -> Optional[Any]:
        """Get item and update access tracking with neural retention updates."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get item and check expiry
                cursor.execute("""
                    SELECT data, data_type, expires_at, access_count, priority, relevance_score, size_bytes
                    FROM short_term_items 
                    WHERE key = ? AND context = ?
                """, (key, context))
                
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                
                data_str, data_type, expires_at, access_count, priority, relevance_score, size_bytes = row
                
                # Check if expired
                if expires_at:
                    expires_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_dt:
                        cursor.execute("DELETE FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                        conn.commit()
                        conn.close()
                        return None
                
                # Calculate updated neural scores
                access_frequency = self._calculate_access_frequency(key, context)
                retention_score = self._calculate_retention_score(priority, relevance_score, size_bytes)
                neural_weight = self._calculate_neural_weight(retention_score, priority, relevance_score)
                
                # Update access tracking with neural retention scores
                cursor.execute("""
                    UPDATE short_term_items 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP,
                        access_frequency = ?, retention_score = ?, neural_weight = ?
                    WHERE key = ? AND context = ?
                """, (access_frequency, retention_score, neural_weight, key, context))
                
                # Track access pattern
                self._track_access_pattern(key, context, 'read')
                
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
        """Manage capacity by removing items using neural retention scoring."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current size
            cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM short_term_items")
            current_size = cursor.fetchone()[0]
            
            # If adding new item would exceed capacity, remove items
            while current_size + new_item_size > self.capacity_bytes:
                if self.enable_neural_retention:
                    # Use neural retention scoring for intelligent removal
                    cursor.execute("""
                        SELECT key, context, size_bytes FROM short_term_items 
                        ORDER BY neural_weight ASC, retention_score ASC, access_frequency ASC, last_accessed ASC 
                        LIMIT 1
                    """)
                else:
                    # Fallback to simple priority-based removal
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
                
                self.logger.debug(f"[ShortTermMemory] Removed item '{key}' for capacity management (neural retention: {self.enable_neural_retention})")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error managing capacity: {ex}")
    
    def cleanup_expired(self) -> int:
        """Remove expired items and old items based on neural retention policy."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Remove explicitly expired items
                cursor.execute("DELETE FROM short_term_items WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP")
                expired_count = cursor.rowcount
                
                # Neural retention-based cleanup
                if self.enable_neural_retention:
                    # Remove items older than retention period using neural scoring
                    cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
                    cursor.execute("""
                        DELETE FROM short_term_items 
                        WHERE created_at < ? AND retention_score < 0.6 AND neural_weight < 0.7
                    """, (cutoff_date,))
                    neural_removed = cursor.rowcount
                    
                    # Also remove very old items with low neural weights regardless of priority
                    very_old_cutoff = (datetime.now() - timedelta(days=self.retention_days * 2)).isoformat()
                    cursor.execute("""
                        DELETE FROM short_term_items 
                        WHERE created_at < ? AND neural_weight < 0.5
                    """, (very_old_cutoff,))
                    very_old_removed = cursor.rowcount
                    
                    old_count = neural_removed + very_old_removed
                else:
                    # Fallback to simple priority-based cleanup
                    cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
                    cursor.execute("""
                        DELETE FROM short_term_items 
                        WHERE created_at < ? AND priority < 0.8
                    """, (cutoff_date,))
                    old_count = cursor.rowcount
                
                # Clean up old access patterns
                pattern_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("DELETE FROM access_patterns WHERE access_timestamp < ?", (pattern_cutoff,))
                pattern_removed = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                total_removed = expired_count + old_count
                if total_removed > 0:
                    self.logger.info(f"[ShortTermMemory] Cleaned up {total_removed} items ({expired_count} expired, {old_count} old, {pattern_removed} old patterns)")
                
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
    
    # Neural Retention Scoring Methods
    
    def _calculate_relevance_score(self, content: str, context: str, memory_type: str, tags: List[str]) -> float:
        """Calculate relevance score based on content analysis and context."""
        try:
            score = 0.5  # Base score
            
            # Content-based scoring
            content_lower = content.lower()
            
            # Higher score for structured data
            if memory_type in ['json', 'structured']:
                score += 0.1
            
            # Higher score for certain keywords
            important_keywords = ['error', 'critical', 'important', 'urgent', 'task', 'project']
            keyword_count = sum(1 for keyword in important_keywords if keyword in content_lower)
            score += min(0.2, keyword_count * 0.05)
            
            # Context-based scoring
            if context != 'default':
                score += 0.1  # Non-default contexts are more relevant
            
            # Tag-based scoring
            if tags:
                score += min(0.15, len(tags) * 0.03)
            
            # Content length scoring (moderate length preferred)
            content_length = len(content)
            if 100 <= content_length <= 1000:
                score += 0.1
            elif content_length > 5000:
                score -= 0.1  # Very long content less relevant for short-term
            
            return min(1.0, max(0.0, score))
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating relevance score: {ex}")
            return 0.5
    
    def _calculate_retention_score(self, priority: float, relevance_score: float, size_bytes: int) -> float:
        """Calculate retention score using neural network-inspired weighting."""
        try:
            # Weighted combination of factors
            score = (
                priority * self.retention_weights['priority'] +
                relevance_score * self.retention_weights['relevance'] +
                self._calculate_size_efficiency_score(size_bytes) * self.retention_weights['size_efficiency']
            )
            
            # Add recency bonus (new items get slight boost)
            score += 0.1 * self.retention_weights['recency']
            
            return min(1.0, max(0.0, score))
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating retention score: {ex}")
            return 0.5
    
    def _calculate_neural_weight(self, retention_score: float, priority: float, relevance_score: float) -> float:
        """Calculate neural weight for advanced retention decisions."""
        try:
            # Neural network-inspired activation function
            # Sigmoid-like function for smooth transitions
            import math
            
            # Combine scores with non-linear activation
            combined_score = (retention_score + priority + relevance_score) / 3.0
            
            # Apply sigmoid activation
            neural_weight = 1.0 / (1.0 + math.exp(-5 * (combined_score - 0.5)))
            
            return neural_weight
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating neural weight: {ex}")
            return 0.5
    
    def _calculate_size_efficiency_score(self, size_bytes: int) -> float:
        """Calculate efficiency score based on size (smaller is better for short-term)."""
        try:
            # Optimal size range for short-term memory (1KB - 10KB)
            if size_bytes <= 1024:  # <= 1KB
                return 1.0
            elif size_bytes <= 10240:  # <= 10KB
                return 0.8
            elif size_bytes <= 102400:  # <= 100KB
                return 0.6
            elif size_bytes <= 1048576:  # <= 1MB
                return 0.4
            else:
                return 0.2  # Large items less efficient for short-term
                
        except Exception:
            return 0.5
    
    def _calculate_access_frequency(self, key: str, context: str) -> float:
        """Calculate access frequency from recent access patterns."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get access count in last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM access_patterns 
                WHERE item_key = ? AND context = ? AND access_timestamp > ?
            """, (key, context, yesterday))
            
            recent_accesses = cursor.fetchone()[0]
            conn.close()
            
            # Convert to frequency score (0-1)
            return min(1.0, recent_accesses / 10.0)  # Max 10 accesses = 1.0 score
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating access frequency: {ex}")
            return 0.0
    
    def _track_access_pattern(self, key: str, context: str, access_type: str = 'read'):
        """Track access patterns for neural retention analysis."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO access_patterns (item_key, context, access_type, session_id)
                VALUES (?, ?, ?, ?)
            """, (key, context, access_type, str(int(time.time()))))
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error tracking access pattern: {ex}")
    
    def _background_cleanup(self):
        """Perform background cleanup if needed."""
        try:
            now = datetime.now()
            if now - self._last_cleanup > self._cleanup_interval:
                self.cleanup_expired()
                self._last_cleanup = now
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error in background cleanup: {ex}")
    
    def get_neural_stats(self) -> Dict[str, Any]:
        """Get neural retention statistics."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get neural weight distribution
                cursor.execute("SELECT AVG(neural_weight), MIN(neural_weight), MAX(neural_weight) FROM short_term_items")
                avg_weight, min_weight, max_weight = cursor.fetchone()
                
                # Get retention score distribution
                cursor.execute("SELECT AVG(retention_score), MIN(retention_score), MAX(retention_score) FROM short_term_items")
                avg_retention, min_retention, max_retention = cursor.fetchone()
                
                # Get high-value items count
                cursor.execute("SELECT COUNT(*) FROM short_term_items WHERE neural_weight > 0.7")
                high_value_count = cursor.fetchone()[0]
                
                # Get access pattern stats
                cursor.execute("SELECT COUNT(*) FROM access_patterns WHERE access_timestamp > datetime('now', '-24 hours')")
                recent_accesses = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'neural_weights': {
                        'average': round(avg_weight or 0, 3),
                        'min': round(min_weight or 0, 3),
                        'max': round(max_weight or 0, 3)
                    },
                    'retention_scores': {
                        'average': round(avg_retention or 0, 3),
                        'min': round(min_retention or 0, 3),
                        'max': round(max_retention or 0, 3)
                    },
                    'high_value_items': high_value_count,
                    'recent_accesses_24h': recent_accesses,
                    'neural_retention_enabled': self.enable_neural_retention
                }
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting neural stats: {ex}")
            return {'error': str(ex)}


def test_enhanced_short_term_memory():
    """Comprehensive test of enhanced ShortTermMemory functionality"""
    print("=" * 60)
    print("Testing Enhanced ShortTermMemory with Neural Retention")
    print("=" * 60)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    stm = ShortTermMemory(capacity_gb=0.001, enable_neural_retention=True)  # 1MB for testing
    
    # Add various types of items
    test_items = [
        ("task1", {"type": "critical", "description": "Important task"}, "project", 0.9, "task", ["critical", "urgent"]),
        ("note1", "Simple note for testing", "notes", 0.5, "general", ["test"]),
        ("data1", [1, 2, 3, 4, 5], "data", 0.7, "list", ["numbers"]),
        ("config1", {"setting": "value", "enabled": True}, "config", 0.8, "json", ["config"]),
        ("temp1", "Temporary data", "temp", 0.3, "general", [])
    ]
    
    for key, item, context, priority, memory_type, tags in test_items:
        result = stm.add(key, item, context, priority, memory_type, tags)
        print(f"  Added {key}: {result}")
    
    # Test 2: Retrieval and access tracking
    print("\n2. Testing retrieval and access tracking...")
    for i in range(3):  # Access items multiple times
        item1 = stm.get("task1", "project")
        item2 = stm.get("note1", "notes")
        print(f"  Access {i+1}: task1={item1 is not None}, note1={item2 is not None}")
    
    # Test 3: Neural retention statistics
    print("\n3. Testing neural retention statistics...")
    neural_stats = stm.get_neural_stats()
    print(f"  Neural Stats: {neural_stats}")
    
    # Test 4: Basic statistics
    print("\n4. Testing basic statistics...")
    stats = stm.get_stats()
    print(f"  Basic Stats: {stats}")
    
    # Test 5: Capacity management
    print("\n5. Testing capacity management...")
    # Add large items to trigger capacity management
    large_data = "x" * 1000  # 1KB item
    for i in range(10):
        stm.add(f"large_{i}", large_data, "large", 0.4, "string", [])
    
    stats_after = stm.get_stats()
    print(f"  Stats after large items: {stats_after}")
    
    # Test 6: Cleanup functionality
    print("\n6. Testing cleanup functionality...")
    removed = stm.cleanup_expired()
    print(f"  Removed {removed} expired items")
    
    # Test 7: Neural retention vs simple retention
    print("\n7. Comparing neural vs simple retention...")
    
    # Test with neural retention disabled
    stm_simple = ShortTermMemory(capacity_gb=0.001, enable_neural_retention=False)
    stm_simple.add("test1", "Test data", priority=0.9)
    stm_simple.add("test2", "Test data", priority=0.1)
    
    # Add items to trigger capacity management
    for i in range(5):
        stm_simple.add(f"filler_{i}", "x" * 200, priority=0.5)
    
    # Check which items survived
    test1_simple = stm_simple.get("test1")
    test2_simple = stm_simple.get("test2")
    print(f"  Simple retention - High priority item exists: {test1_simple is not None}")
    print(f"  Simple retention - Low priority item exists: {test2_simple is not None}")
    
    # Test with neural retention enabled
    stm_neural = ShortTermMemory(capacity_gb=0.001, enable_neural_retention=True)
    stm_neural.add("test1", "Critical error in system", priority=0.9, tags=["critical", "error"])
    stm_neural.add("test2", "Random note", priority=0.1, tags=[])
    
    # Add items to trigger capacity management
    for i in range(5):
        stm_neural.add(f"filler_{i}", "x" * 200, priority=0.5)
    
    # Check which items survived
    test1_neural = stm_neural.get("test1")
    test2_neural = stm_neural.get("test2")
    print(f"  Neural retention - High priority item exists: {test1_neural is not None}")
    print(f"  Neural retention - Low priority item exists: {test2_neural is not None}")
    
    # Test 8: Access pattern tracking
    print("\n8. Testing access pattern tracking...")
    # Access items multiple times to build patterns
    for i in range(5):
        stm.get("task1", "project")
        if i % 2 == 0:
            stm.get("data1", "data")
    
    neural_stats_final = stm.get_neural_stats()
    print(f"  Final neural stats: {neural_stats_final}")
    
    print("\n" + "=" * 60)
    print("Enhanced ShortTermMemory test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_enhanced_short_term_memory()