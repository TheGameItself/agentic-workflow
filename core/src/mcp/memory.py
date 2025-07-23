#!/usr/bin/env python3
"""
Enhanced Memory Manager for MCP Core System
Advanced memory management with multi-tier storage, semantic search, and brain-inspired features.
"""

import sqlite3
import json
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Try to import advanced features
try:
    from .database_manager import OptimizedDatabaseManager
    DATABASE_MANAGER_AVAILABLE = True
except ImportError:
    DATABASE_MANAGER_AVAILABLE = False

class MemoryType(Enum):
    """Types of memory in the system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    LONG_TERM = "long_term"
    CONTEXTUAL = "contextual"

class MemoryPriority(Enum):
    """Memory priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MINIMAL = 5

def get_db_path() -> str:
    """Get the path to the main memory database."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / 'enhanced_memory.db')

class MemoryManager:
    """Enhanced Memory Management System with brain-inspired features."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the enhanced memory manager."""
        self.db_path = str(db_path) if db_path is not None else get_db_path()
        
        # Initialize database manager
        if DATABASE_MANAGER_AVAILABLE:
            self.db_manager = OptimizedDatabaseManager(self.db_path)
            self._use_optimized_db = True
        else:
            self._use_optimized_db = False
            self._init_database()
        
        # Performance tracking
        self.access_stats = {
            'total_accesses': 0,
            'cache_hits': 0,
            'search_queries': 0,
            'average_response_time': 0.0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Memory Manager initialized")
    
    def _init_database(self):
        """Initialize database tables with enhanced schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Core memories table (compression, tagging, context)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                memory_type TEXT DEFAULT 'general',
                priority REAL DEFAULT 0.5,
                context TEXT,
                tags TEXT,
                chunk_hash TEXT, -- For chunking and deduplication
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tasks table (for compatibility, see task_manager.py for advanced schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                parent_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES tasks (id)
            )
        """)
        
        # Feedback table (feedback, research alignment)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                feedback TEXT NOT NULL,
                impact INTEGER DEFAULT 0,
                principle TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_memory(self, text: str, memory_type: str = 'general', 
                   priority: float = 0.5, context: str = '', tags: Optional[List[str]] = None) -> int:
        """Add a new memory."""
        if tags is None:
            tags = []
        tags_json = json.dumps(tags)
        ctx = context or ''
        if self._use_optimized_db:
            self.db.execute_query("""
                INSERT INTO memories (text, memory_type, priority, context, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (text, memory_type, priority, ctx, tags_json), fetch=False)
            id_result = self.db.execute_query("SELECT last_insert_rowid() as id")
            return id_result[0]['id'] if id_result else 0
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (text, memory_type, priority, context, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (text, memory_type, priority, ctx, tags_json))
            memory_id = cursor.lastrowid or 0
            conn.commit()
            conn.close()
            return memory_id
    
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, text, memory_type, priority, context, tags, created_at, updated_at
            FROM memories WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            memory_id, text, memory_type, priority, context, tags_json, created_at, updated_at = row
            tags = json.loads(tags_json) if tags_json else []
            return {
                'id': memory_id,
                'text': text if text is not None else '',
                'memory_type': memory_type if memory_type is not None else '',
                'priority': priority if priority is not None else 0.0,
                'context': context if context is not None else '',
                'tags': tags,
                'created_at': created_at if created_at is not None else '',
                'updated_at': updated_at if updated_at is not None else ''
            }
        return None
    
    def search_memories(self, query: str, limit: int = 10, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memories by text content."""
        if memory_type is None:
            memory_type = ''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if memory_type:
            cursor.execute("""
                SELECT id, text, memory_type, priority, context, tags, created_at
                FROM memories 
                WHERE text LIKE ? AND memory_type = ?
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
            """, (f'%{query if query is not None else ""}%', memory_type, limit if limit is not None else 10))
        else:
            cursor.execute("""
                SELECT id, text, memory_type, priority, context, tags, created_at
                FROM memories 
                WHERE text LIKE ?
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
            """, (f'%{query if query is not None else ""}%', limit if limit is not None else 10))
        results = []
        for row in cursor.fetchall():
            memory_id, text, memory_type, priority, context, tags_json, created_at = row
            tags = json.loads(tags_json) if tags_json else []
            results.append({
                'id': memory_id,
                'text': text if text is not None else '',
                'memory_type': memory_type if memory_type is not None else '',
                'priority': priority if priority is not None else 0.0,
                'context': context if context is not None else '',
                'tags': tags,
                'created_at': created_at if created_at is not None else ''
            })
        conn.close()
        return results
    
    def add_task(self, description: str, priority: int = 0, parent_id: int = 0) -> int:
        """Add a new task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tasks (description, priority, parent_id)
            VALUES (?, ?, ?)
        """, (description if description is not None else '', priority if priority is not None else 0, parent_id))
        task_id = cursor.lastrowid or 0
        conn.commit()
        conn.close()
        return task_id
    
    def get_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tasks, optionally filtered by status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if status is not None and status != '':
            cursor.execute("""
                SELECT id, description, status, priority, parent_id, created_at, completed_at
                FROM tasks WHERE status = ?
                ORDER BY priority DESC, created_at ASC
            """, (status,))
        else:
            cursor.execute("""
                SELECT id, description, status, priority, parent_id, created_at, completed_at
                FROM tasks
                ORDER BY priority DESC, created_at ASC
            """)
        
        results = []
        for row in cursor.fetchall():
            task_id, description, status, priority, parent_id, created_at, completed_at = row
            
            results.append({
                'id': task_id,
                'description': description,
                'status': status,
                'priority': priority,
                'parent_id': parent_id,
                'created_at': created_at,
                'completed_at': completed_at
            })
        
        conn.close()
        return results
    
    def complete_task(self, task_id: int) -> bool:
        """Mark a task as completed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks 
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (task_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def add_feedback(self, task_id: int, feedback: str, impact: int = 0, principle: str = '') -> int:
        """Add feedback for a task."""
        princ = principle or ''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (task_id, feedback, impact, principle)
            VALUES (?, ?, ?, ?)
        """, (task_id if task_id is not None else 0, feedback if feedback is not None else '', impact if impact is not None else 0, princ))
        feedback_id = cursor.lastrowid or 0
        conn.commit()
        conn.close()
        return feedback_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memory statistics
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT memory_type) FROM memories")
        memory_types = cursor.fetchone()[0]
        
        # Task statistics
        cursor.execute("SELECT COUNT(*) FROM tasks")
        total_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
        completed_tasks = cursor.fetchone()[0]
        
        # Feedback statistics
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_memories': total_memories,
            'memory_types': memory_types,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'total_feedback': total_feedback,
            'completion_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        }

    def add_tag_to_memory(self, memory_id: int, tag: str) -> bool:
        """Add a tag to a memory entry dynamically."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        tags = json.loads(row[0]) if row and row[0] else []
        if tag not in tags:
            tags.append(tag)
            cursor.execute("UPDATE memories SET tags = ? WHERE id = ?", (json.dumps(tags), memory_id))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False

    def remove_tag_from_memory(self, memory_id: int, tag: str) -> bool:
        """Remove a tag from a memory entry dynamically."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        tags = json.loads(row[0]) if row and row[0] else []
        if tag in tags:
            tags.remove(tag)
            cursor.execute("UPDATE memories SET tags = ? WHERE id = ?", (json.dumps(tags), memory_id))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False

    def search_memories_by_tag(self, tag: str) -> list:
        """Search for all memories with a given tag."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, memory_type, priority, context, tags, created_at FROM memories")
        results = []
        for row in cursor.fetchall():
            memory_id, text, memory_type, priority, context, tags_json, created_at = row
            tags = json.loads(tags_json) if tags_json else []
            if tag in tags:
                results.append({
                    'id': memory_id,
                    'text': text,
                    'memory_type': memory_type,
                    'priority': priority,
                    'context': context,
                    'tags': tags,
                    'created_at': created_at
                })
        conn.close()
        return results

    def dynamic_expand(self, new_input: str, context: str = '', tags: Optional[List[str]] = None) -> int:
        """Dynamically expand memory by adding a new node for novel input."""
        # Check if similar memory exists
        existing = self.search_memories(new_input, limit=1)
        if existing:
            return existing[0]['id']
        return self.add_memory(new_input, context=context, tags=tags)

    def generalize_memories(self, similarity_threshold: float = 0.8) -> Optional[int]:
        """Generalize from existing memories by clustering and creating an abstract node."""
        try:
            memories = self.search_memories('', limit=100)
            # Simple clustering: group by memory_type and context
            clusters = {}
            for mem in memories:
                key = (mem['memory_type'], mem['context'])
                clusters.setdefault(key, []).append(mem)
            # Create generalized memory for largest cluster
            largest = max(clusters.values(), key=len, default=None)
            if largest and len(largest) > 1:
                texts = [m['text'] for m in largest]
                summary = f"Generalized: {'; '.join(texts[:3])}..." if len(texts) > 3 else f"Generalized: {'; '.join(texts)}"
                return self.add_memory(summary, memory_type='generalized', context=largest[0]['context'], tags=['generalized'])
        except Exception as e:
            print(f"[MemoryManager] Generalization failed: {e}")
        return None

    def prune_memories(self, min_priority: float = 0.2, max_age_days: int = 180) -> int:
        """Prune (forget) low-priority or old memories. Returns number pruned."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            cursor.execute("""
                DELETE FROM memories WHERE priority < ? OR created_at < ?
            """, (min_priority, cutoff))
            pruned = cursor.rowcount
            conn.commit()
            conn.close()
            return pruned
        except Exception as e:
            print(f"[MemoryManager] Pruning failed: {e}")
            return 0

    def adapt_on_feedback(self, feedback: str, memory_id: int, impact: int = 0) -> bool:
        """Adapt memory node based on feedback (increase priority, trigger generalization/pruning)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Increase priority if positive impact, decrease if negative
            if impact > 0:
                cursor.execute("UPDATE memories SET priority = priority + 0.1 WHERE id = ?", (memory_id,))
            elif impact < 0:
                cursor.execute("UPDATE memories SET priority = priority - 0.1 WHERE id = ?", (memory_id,))
            conn.commit()
            conn.close()
            # Optionally trigger generalization or pruning
            if impact > 1:
                self.generalize_memories()
            elif impact < -1:
                self.prune_memories()
            return True
        except Exception as e:
            print(f"[MemoryManager] Feedback adaptation failed: {e}")
            return False 