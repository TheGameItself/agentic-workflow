#!/usr/bin/env python3
"""
TaskLobe: Task Management Engine for MCP

This module implements the TaskLobe, responsible for hierarchical task management, dependencies, and progress tracking.
See src/mcp/lobes.py for the lobe registry and architecture overview.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from enum import Enum
from .unified_memory import UnifiedMemoryManager
from datetime import datetime, timedelta

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    CRITICAL = 10
    HIGH = 8
    MEDIUM = 5
    LOW = 2
    MINIMAL = 1

class TaskManager:
    """Unified task management with priority trees and dependencies."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the task manager."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'unified_memory.db')
        self.db_path = str(db_path)
        self.task_types = {}
        self._init_task_database()
        # Register default task types
        self.register_task_type('general', {'description': 'General task'})
        self.register_task_type('accuracy_critical', {'description': 'Accuracy-critical task'})
    
    def _init_task_database(self):
        """
        Initialize the task database with advanced schema.
        - Ensures all advanced features: meta-tasks, tagging, feedback, chunking, crosslinks, partial completion, accuracy-critical, etc.
        - Research-aligned: see README.md, idea.txt, Zolfagharinejad et al., 2024 (EPJ B), Ren & Xia, 2024 (arXiv).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Enhanced tasks table (meta-tasks, accuracy-critical, chunking, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                parent_id INTEGER,
                estimated_hours REAL DEFAULT 0.0,
                actual_hours REAL DEFAULT 0.0,
                accuracy_critical BOOLEAN DEFAULT FALSE,
                is_meta BOOLEAN DEFAULT FALSE,
                meta_type TEXT,
                chunk_hash TEXT, -- For chunking and deduplication
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                due_date TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES tasks (id)
            )
        """)
        # Task dependencies table (crosslinks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                depends_on_task_id INTEGER,
                dependency_type TEXT DEFAULT 'blocks',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks (id),
                FOREIGN KEY (depends_on_task_id) REFERENCES tasks (id)
            )
        """)
        # Task notes with line numbers (partial completion, chunking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                note_text TEXT NOT NULL,
                line_number INTEGER,
                file_path TEXT,
                note_type TEXT DEFAULT 'general',
                chunk_hash TEXT, -- For chunking and deduplication
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        # Task progress tracking (partial completion)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                progress_percentage REAL DEFAULT 0.0,
                current_step TEXT,
                partial_completion_notes TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        # Task feedback and lessons (feedback, research alignment)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                feedback_text TEXT NOT NULL,
                lesson_learned TEXT,
                principle TEXT,
                impact_score INTEGER DEFAULT 0,
                feedback_type TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)
        # Task tags and categories (tagging, chunking, crosslinks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                tag_name TEXT NOT NULL,
                tag_type TEXT DEFAULT 'general',
                parent_tag_id INTEGER,
                color TEXT,
                icon TEXT,
                description TEXT,
                chunk_hash TEXT, -- For chunking and deduplication
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks (id),
                FOREIGN KEY (parent_tag_id) REFERENCES task_tags (id)
            )
        """)
        conn.commit()
        conn.close()
    
    def create_task(self, title: str, description: Optional[str] = None, priority: int = 5,
                   parent_id: Optional[int] = None, estimated_hours: float = 0.0,
                   accuracy_critical: bool = False, due_date: Optional[datetime] = None,
                   tags: Optional[List[str]] = None, is_meta: bool = False, meta_type: Optional[str] = None) -> int:
        """Create a new task with full metadata, including meta-task support."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Ensure is_meta is boolean and meta_type is string
        is_meta_val = bool(is_meta) if is_meta is not None else False
        meta_type_val = str(meta_type) if meta_type is not None else ""
        # Ensure due_date is string or None
        if due_date is not None and hasattr(due_date, 'isoformat'):
            due_date_val = due_date.isoformat()
        else:
            due_date_val = due_date if due_date is None or isinstance(due_date, str) else str(due_date)
        parent_id_val = int(parent_id) if parent_id is not None else None
        due_date_sql_val = str(due_date_val) if due_date_val is not None else None
        cursor.execute("""
            INSERT INTO tasks (title, description, priority, parent_id, 
                              estimated_hours, accuracy_critical, due_date, is_meta, meta_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, description or '', priority, parent_id_val, estimated_hours, 
              accuracy_critical, due_date_sql_val, is_meta_val, meta_type_val))
        
        task_id = cursor.lastrowid
        
        # Add tags if provided
        if tags:
            for tag in tags:
                cursor.execute("""
                    INSERT INTO task_tags (task_id, tag_name)
                    VALUES (?, ?)
                """, (task_id, tag))
        
        # Initialize progress tracking
        cursor.execute("""
            INSERT INTO task_progress (task_id, progress_percentage, current_step)
            VALUES (?, ?, ?)
        """, (task_id, 0.0, "Not started"))
        
        conn.commit()
        conn.close()
        
        return int(task_id) if task_id is not None else -1
    
    def add_task_dependency(self, task_id: int, depends_on_task_id: int,
                           dependency_type: str = 'blocks') -> int:
        """Add a dependency between tasks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO task_dependencies (task_id, depends_on_task_id, dependency_type)
            VALUES (?, ?, ?)
        """, (task_id, depends_on_task_id, dependency_type))
        
        dependency_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return int(dependency_id) if dependency_id is not None else -1
    
    def add_task_note(self, task_id: int, note_text: str, line_number: Optional[int] = None,
                     file_path: Optional[str] = None, note_type: str = 'general') -> int:
        """Add a note to a task with optional line number and file path."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO task_notes (task_id, note_text, line_number, file_path, note_type)
            VALUES (?, ?, ?, ?, ?)
        """, (task_id, note_text, line_number if line_number is not None else None, file_path or '', note_type))
        
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return int(note_id) if note_id is not None else -1
    
    def update_task_progress(self, task_id: int, progress_percentage: float,
                           current_step: Optional[str] = None, partial_completion_notes: Optional[str] = None) -> bool:
        """Update task progress with partial completion support."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update progress
        cursor.execute("""
            UPDATE task_progress 
            SET progress_percentage = ?, current_step = ?, partial_completion_notes = ?,
                last_updated = ?
            WHERE task_id = ?
        """, (progress_percentage, current_step, partial_completion_notes, 
              datetime.now(), task_id))
        
        # Update task status based on progress
        if progress_percentage >= 100.0:
            status = TaskStatus.COMPLETED.value
            completed_at = datetime.now()
        elif progress_percentage > 0:
            status = TaskStatus.IN_PROGRESS.value
            completed_at = None
        else:
            status = TaskStatus.PENDING.value
            completed_at = None
        
        cursor.execute("""
            UPDATE tasks 
            SET status = ?, completed_at = ?, updated_at = ?
            WHERE id = ?
        """, (status, completed_at, datetime.now(), task_id))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_task_tree(self, root_task_id: Optional[int] = None, include_completed: bool = False) -> Dict[str, Any]:
        """Get the complete task tree structure."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if root_task_id:
            # Get specific subtree
            cursor.execute("""
                WITH RECURSIVE task_tree AS (
                    SELECT id, title, description, status, priority, parent_id,
                           estimated_hours, actual_hours, accuracy_critical, is_meta, meta_type,
                           created_at, updated_at, started_at, completed_at, due_date,
                           0 as level
                    FROM tasks WHERE id = ?
                    UNION ALL
                    SELECT t.id, t.title, t.description, t.status, t.priority, t.parent_id,
                           t.estimated_hours, t.actual_hours, t.accuracy_critical, t.is_meta, t.meta_type,
                           t.created_at, t.updated_at, t.started_at, t.completed_at, t.due_date,
                           tt.level + 1
                    FROM tasks t
                    JOIN task_tree tt ON t.parent_id = tt.id
                )
                SELECT id, title, description, status, priority, parent_id,
                       estimated_hours, actual_hours, accuracy_critical, is_meta, meta_type,
                       created_at, updated_at, started_at, completed_at, due_date
                FROM task_tree
                ORDER BY level, priority DESC, created_at ASC
            """, (root_task_id,))
        else:
            # Get all tasks
            status_filter = "" if include_completed else "WHERE status != 'completed'"
            cursor.execute(f"""
                SELECT id, title, description, status, priority, parent_id,
                       estimated_hours, actual_hours, accuracy_critical, is_meta, meta_type,
                       created_at, updated_at, started_at, completed_at, due_date
                FROM tasks {status_filter}
                ORDER BY priority DESC, created_at ASC
            """)
        
        tasks = cursor.fetchall()
        conn.close()
        
        # Build tree structure
        task_dict = {}
        root_tasks = []
        
        for task in tasks:
            task_id, title, description, status, priority, parent_id, \
            estimated_hours, actual_hours, accuracy_critical, is_meta, meta_type, created_at, \
            updated_at, started_at, completed_at, due_date = task
            
            task_data = {
                'id': task_id,
                'title': title,
                'description': description,
                'status': status,
                'priority': priority,
                'parent_id': parent_id,
                'estimated_hours': estimated_hours,
                'actual_hours': actual_hours,
                'accuracy_critical': accuracy_critical,
                'is_meta': is_meta,
                'meta_type': meta_type,
                'created_at': created_at,
                'updated_at': updated_at,
                'started_at': started_at,
                'completed_at': completed_at,
                'due_date': due_date,
                'children': [],
                'dependencies': self.get_task_dependencies(task_id),
                'notes': self.get_task_notes(task_id),
                'progress': self.get_task_progress(task_id)
            }
            
            task_dict[task_id] = task_data
            
            if parent_id is None:
                root_tasks.append(task_data)
            else:
                if parent_id in task_dict:
                    task_dict[parent_id]['children'].append(task_data)
        
        return {
            'root_tasks': root_tasks,
            'all_tasks': task_dict
        }
    
    def get_task_dependencies(self, task_id: int) -> List[Dict[str, Any]]:
        """Get dependencies for a specific task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT td.id, td.dependency_type, t.id, t.title, t.status, t.priority
            FROM task_dependencies td
            JOIN tasks t ON td.depends_on_task_id = t.id
            WHERE td.task_id = ?
        """, (task_id,))
        
        dependencies = []
        for row in cursor.fetchall():
            dep_id, dep_type, dep_task_id, dep_title, dep_status, dep_priority = row
            dependencies.append({
                'dependency_id': dep_id,
                'dependency_type': dep_type,
                'task_id': dep_task_id,
                'title': dep_title,
                'status': dep_status,
                'priority': dep_priority
            })
        
        conn.close()
        return dependencies
    
    def get_task_notes(self, task_id: int) -> List[Dict[str, Any]]:
        """Get notes for a specific task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, note_text, line_number, file_path, note_type, created_at
            FROM task_notes WHERE task_id = ?
            ORDER BY created_at ASC
        """, (task_id,))
        
        notes = []
        for row in cursor.fetchall():
            note_id, note_text, line_number, file_path, note_type, created_at = row
            notes.append({
                'id': note_id,
                'note_text': note_text,
                'line_number': line_number,
                'file_path': file_path,
                'note_type': note_type,
                'created_at': created_at
            })
        
        conn.close()
        return notes
    
    def get_task_progress(self, task_id: int) -> Dict[str, Any]:
        """Get progress information for a specific task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT progress_percentage, current_step, partial_completion_notes, last_updated
            FROM task_progress WHERE task_id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            progress_percentage, current_step, partial_completion_notes, last_updated = row
            return {
                'progress_percentage': progress_percentage,
                'current_step': current_step,
                'partial_completion_notes': partial_completion_notes,
                'last_updated': last_updated
            }
        
        return {
            'progress_percentage': 0.0,
            'current_step': 'Not started',
            'partial_completion_notes': None,
            'last_updated': None
        }
    
    def get_blocked_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks that are blocked by incomplete dependencies."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT t.id, t.title, t.status, t.priority,
                   GROUP_CONCAT(dep.title) as blocking_tasks
            FROM tasks t
            JOIN task_dependencies td ON t.id = td.task_id
            JOIN tasks dep ON td.depends_on_task_id = dep.id
            WHERE dep.status != 'completed' AND t.status != 'completed'
            GROUP BY t.id
            ORDER BY t.priority DESC
        """)
        
        blocked_tasks = []
        for row in cursor.fetchall():
            task_id, title, status, priority, blocking_tasks = row
            blocked_tasks.append({
                'id': task_id,
                'title': title,
                'status': status,
                'priority': priority,
                'blocking_tasks': blocking_tasks.split(',') if blocking_tasks else []
            })
        
        conn.close()
        return blocked_tasks
    
    def get_accuracy_critical_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks marked as accuracy-critical."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, description, status, priority, estimated_hours,
                   created_at, due_date
            FROM tasks 
            WHERE accuracy_critical = TRUE
            ORDER BY priority DESC, created_at ASC
        """)
        
        critical_tasks = []
        for row in cursor.fetchall():
            task_id, title, description, status, priority, estimated_hours, created_at, due_date = row
            critical_tasks.append({
                'id': task_id,
                'title': title,
                'description': description,
                'status': status,
                'priority': priority,
                'estimated_hours': estimated_hours,
                'created_at': created_at,
                'due_date': due_date
            })
        
        conn.close()
        return critical_tasks
    
    def add_task_feedback(self, task_id: int, feedback_text: str, lesson_learned: Optional[str] = None,
                         principle: Optional[str] = None, impact_score: int = 0,
                         feedback_type: str = 'general') -> int:
        """Add feedback and lessons learned to a task. Triggers auto-suggestions and flags issues based on feedback."""
        task_id_val = int(task_id) if task_id is not None else 0
        feedback_text_val = feedback_text or ''
        lesson_learned_val = lesson_learned or ''
        principle_val = principle or ''
        impact_score_val = int(impact_score) if impact_score is not None else 0
        feedback_type_val = feedback_type or 'general'
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO task_feedback 
            (task_id, feedback_text, lesson_learned, principle, impact_score, feedback_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (task_id_val, feedback_text_val, lesson_learned_val, principle_val, impact_score_val, feedback_type_val))
        feedback_id = cursor.lastrowid
        # --- Auto-flagging and suggestions logic ---
        # Flag task if negative impact or feedback_type is 'issue'
        flagged = False
        critical = False
        if impact_score_val < 0 or feedback_type_val == 'issue':
            cursor.execute("""
                INSERT INTO task_tags (task_id, tag_name, tag_type)
                VALUES (?, ?, ?)
            """, (task_id_val, 'flagged', 'system'))
            flagged = True
        # Escalate if repeated negative feedback
        cursor.execute("""
            SELECT COUNT(*) FROM task_feedback WHERE task_id = ? AND impact_score < 0
        """, (task_id_val,))
        neg_count = cursor.fetchone()[0]
        if neg_count >= 3:
            cursor.execute("""
                INSERT INTO task_tags (task_id, tag_name, tag_type)
                VALUES (?, ?, ?)
            """, (task_id_val, 'critical', 'system'))
            critical = True
        # Suggest improvement if positive feedback
        if impact_score_val > 2 and feedback_type_val == 'general':
            cursor.execute("""
                INSERT INTO task_tags (task_id, tag_name, tag_type)
                VALUES (?, ?, ?)
            """, (task_id_val, 'success', 'system'))
        conn.commit()
        conn.close()
        # --- Feedback-driven reminder adaptation ---
        if flagged or critical:
            unified = UnifiedMemoryManager()
            status = 'critical' if critical else 'flagged'
            interval = 6 if critical else 12
            unified.adapt_task_reminder_on_feedback(task_id_val, status=status, interval_hours=interval)
        return int(feedback_id) if feedback_id is not None else 0

    def get_task_suggestions_and_flags(self, task_id: int) -> dict:
        """Get suggestions and flags for a task based on feedback and tags."""
        task_id_val = int(task_id) if task_id is not None else 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tag_name FROM task_tags WHERE task_id = ? AND tag_type = 'system'
        """, (task_id_val,))
        tags = [row[0] for row in cursor.fetchall()]
        suggestions = []
        flags = []
        if 'flagged' in tags:
            flags.append('flagged')
        if 'critical' in tags:
            flags.append('critical')
        if 'success' in tags:
            suggestions.append('This task has positive feedback. Consider using as a template or best practice.')
        conn.close()
        return {'suggestions': suggestions, 'flags': flags}
    
    def get_tasks(self, status: str = None, priority_min: int = None, include_completed: bool = True) -> List[Dict[str, Any]]:
        """Get tasks with optional filtering, including meta-task fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        status = status or ''
        priority_min = int(priority_min) if priority_min is not None else 0
        include_completed = include_completed if include_completed is not None else True
        query = """
            SELECT t.id, t.title, t.description, t.status, t.priority, 
                   t.parent_id, t.estimated_hours, t.actual_hours, 
                   t.accuracy_critical, t.is_meta, t.meta_type, t.created_at, t.updated_at, 
                   t.started_at, t.completed_at, t.due_date
            FROM tasks t
            WHERE 1=1
        """
        params = []
        if status:
            query += " AND t.status = ?"
            params.append(status)
        if priority_min is not None:
            query += " AND t.priority >= ?"
            params.append(priority_min)
        if not include_completed:
            query += " AND t.status != 'completed'"
        query += " ORDER BY t.priority DESC, t.created_at ASC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        tasks = []
        for row in rows:
            task = {
                'id': row[0],
                'title': row[1] or '',
                'description': row[2] or '',
                'status': row[3] or '',
                'priority': row[4] if row[4] is not None else 0,
                'parent_id': row[5] if row[5] is not None else 0,
                'estimated_hours': row[6] if row[6] is not None else 0.0,
                'actual_hours': row[7] if row[7] is not None else 0.0,
                'accuracy_critical': bool(row[8]),
                'is_meta': bool(row[9]),
                'meta_type': row[10] or '',
                'created_at': row[11] or '',
                'updated_at': row[12] or '',
                'started_at': row[13] or '',
                'completed_at': row[14] or '',
                'due_date': row[15] or ''
            }
            tasks.append(task)
        conn.close()
        return tasks

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM tasks")
        total_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
        completed_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE accuracy_critical = TRUE")
        critical_tasks = cursor.fetchone()[0]
        
        # Status distribution
        cursor.execute("""
            SELECT status, COUNT(*) FROM tasks GROUP BY status
        """)
        status_distribution = dict(cursor.fetchall())
        
        # Priority distribution
        cursor.execute("""
            SELECT priority, COUNT(*) FROM tasks GROUP BY priority ORDER BY priority DESC
        """)
        priority_distribution = dict(cursor.fetchall())
        
        # Time tracking
        cursor.execute("""
            SELECT SUM(estimated_hours), SUM(actual_hours) FROM tasks
        """)
        time_data = cursor.fetchone()
        total_estimated = time_data[0] or 0
        total_actual = time_data[1] or 0
        
        conn.close()
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'critical_tasks': critical_tasks,
            'completion_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'status_distribution': status_distribution,
            'priority_distribution': priority_distribution,
            'time_tracking': {
                'total_estimated_hours': total_estimated,
                'total_actual_hours': total_actual,
                'accuracy': (total_actual / total_estimated * 100) if total_estimated > 0 else 0
            }
        }
    
    def register_task_type(self, name: str, metadata: dict):
        """Register a new task type at runtime."""
        self.task_types[name] = metadata
        self._persist_task_type_metadata(name, metadata)

    def _persist_task_type_metadata(self, name: str, metadata: dict):
        """Persist task type metadata in the DB for dynamic types."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_types (
                name TEXT PRIMARY KEY,
                metadata TEXT
            )
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO task_types (name, metadata)
            VALUES (?, ?)
        """, (name, json.dumps(metadata)))
        conn.commit()
        conn.close()

    def create_tag(self, tag_name: str, tag_type: str = 'general', parent_tag_id: int = None, color: str = None, icon: str = None, description: str = None) -> int:
        """Create a tag for tasks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO task_tags (tag_name, tag_type, parent_tag_id, color, icon, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            tag_name or '',
            tag_type or 'general',
            int(parent_tag_id) if parent_tag_id is not None else 0,
            color or '',
            icon or '',
            description or ''
        ))
        tag_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return int(tag_id) if tag_id is not None else -1

    def update_tag(self, tag_id: int, tag_name: str = None, tag_type: str = None, parent_tag_id: int = None, color: str = None, icon: str = None, description: str = None) -> bool:
        """Update a tag for tasks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE task_tags SET tag_name = ?, tag_type = ?, parent_tag_id = ?, color = ?, icon = ?, description = ? WHERE id = ?
        """, (
            tag_name or '',
            tag_type or 'general',
            int(parent_tag_id) if parent_tag_id is not None else 0,
            color or '',
            icon or '',
            description or '',
            int(tag_id)
        ))
        conn.commit()
        conn.close()
        return True

    def get_tag(self, tag_id: int) -> dict:
        """Retrieve a tag and its metadata."""
        tag_id_val = int(tag_id) if tag_id is not None else 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, tag_name, tag_type, parent_tag_id, color, icon, description, created_at FROM task_tags WHERE id = ?", (tag_id_val,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {}
        return {
            'id': row[0],
            'tag_name': row[1],
            'tag_type': row[2],
            'parent_tag_id': row[3],
            'color': row[4],
            'icon': row[5],
            'description': row[6],
            'created_at': row[7]
        }

    def get_tags(self, task_id: int = None, parent_tag_id: int = None, tag_type: str = None) -> list:
        """Retrieve tags, optionally filtered by task, parent, or type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT id, tag_name, tag_type, parent_tag_id, color, icon, description, created_at FROM task_tags WHERE 1=1"
        params = []
        if task_id is not None:
            query += " AND task_id = ?"
            params.append(int(task_id))
        if parent_tag_id is not None:
            query += " AND parent_tag_id = ?"
            params.append(int(parent_tag_id))
        if tag_type is not None:
            query += " AND tag_type = ?"
            params.append(tag_type or 'general')
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        tags = []
        for row in rows:
            tags.append({
                'id': row[0],
                'tag_name': row[1] or '',
                'tag_type': row[2] or 'general',
                'parent_tag_id': row[3] if row[3] is not None else 0,
                'color': row[4] or '',
                'icon': row[5] or '',
                'description': row[6] or '',
                'created_at': row[7] or ''
            })
        return tags

    def dynamic_expand_task(self, title: str, description: str = '', priority: int = 5, parent_id: Optional[int] = None, tags: Optional[List[str]] = None) -> int:
        """Dynamically expand by adding a new task if not already present."""
        try:
            existing = self.get_tasks()
            for t in existing:
                if t.get('title') == title and t.get('description') == description:
                    return t['id']
            return self.create_task(title, description, priority, parent_id, tags=tags)
        except Exception as e:
            print(f"[TaskManager] Dynamic expand failed: {e}")
            return -1

    def generalize_tasks(self) -> Optional[int]:
        """Generalize from existing tasks by clustering and creating an abstract task."""
        try:
            tasks = self.get_tasks()
            clusters = {}
            for t in tasks:
                key = (t.get('priority'), t.get('parent_id'))
                clusters.setdefault(key, []).append(t)
            largest = max(clusters.values(), key=len, default=None)
            if largest and len(largest) > 1:
                titles = [t['title'] for t in largest]
                summary = f"Generalized: {'; '.join(titles[:3])}..." if len(titles) > 3 else f"Generalized: {'; '.join(titles)}"
                return self.create_task(summary, description='Generalized task', priority=largest[0].get('priority', 5), parent_id=largest[0].get('parent_id'))
        except Exception as e:
            print(f"[TaskManager] Generalization failed: {e}")
        return None

    def prune_tasks(self, min_priority: int = 2, max_age_days: int = 180) -> int:
        """Prune (remove) low-priority or old tasks. Returns number pruned."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            cursor.execute("""
                DELETE FROM tasks WHERE priority < ? OR created_at < ?
            """, (min_priority, cutoff))
            pruned = cursor.rowcount
            conn.commit()
            conn.close()
            return pruned
        except Exception as e:
            print(f"[TaskManager] Pruning failed: {e}")
            return 0

    def adapt_on_feedback(self, feedback: str, task_id: int, impact: int = 0) -> bool:
        """Adapt task node based on feedback (increase priority, trigger generalization/pruning)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if impact > 0:
                cursor.execute("UPDATE tasks SET priority = priority + 1 WHERE id = ?", (task_id,))
            elif impact < 0:
                cursor.execute("UPDATE tasks SET priority = priority - 1 WHERE id = ?", (task_id,))
            conn.commit()
            conn.close()
            if impact > 1:
                self.generalize_tasks()
            elif impact < -1:
                self.prune_tasks()
            return True
        except Exception as e:
            print(f"[TaskManager] Feedback adaptation failed: {e}")
            return False 