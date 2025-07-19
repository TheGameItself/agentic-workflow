#!/usr/bin/env python3
"""
Unified Memory Management System
Combines basic memory, advanced memory, and reminder features into a cohesive interface.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import hashlib
import time

from .memory import MemoryManager
from .advanced_memory import AdvancedMemoryManager, TFIDFEncoder, VectorEncoder
from .reminder_engine import EnhancedReminderEngine

class UnifiedMemoryManager:
    """
    Three-Tier Memory Manager: Integrates WorkingMemory, ShortTermMemory, and LongTermMemory into a unified interface.
    - WorkingMemory: Immediate, fast-access, limited capacity
    - ShortTermMemory: Efficient encoding, FAISS/SQLite hybrid
    - LongTermMemory: Advanced engram compression and association
    Provides automatic tier transitions, cross-tier search, and consolidation workflows.
    """
    
    def __init__(self, db_path: Optional[str] = None, encoder: Optional[VectorEncoder] = None):
        """Initialize the unified memory manager. Optionally specify a vector encoder (TFIDFEncoder, RaBitQEncoder, etc.)."""
        if not db_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path_str = os.path.join(data_dir, 'unified_memory.db')
        else:
            db_path_str = str(db_path)
        self.db_path: str = db_path_str
        self.encoder: VectorEncoder = encoder or TFIDFEncoder()
        self.basic_memory = MemoryManager(self.db_path)
        self.advanced_memory = AdvancedMemoryManager(self.db_path, encoder=self.encoder)
        self.reminder_engine = EnhancedReminderEngine(self.db_path)
        # --- Three-Tier Memory Integration ---
        from .lobes.shared_lobes.working_memory import WorkingMemory, ShortTermMemory, LongTermMemory
        self.working_memory = WorkingMemory(capacity_mb=10.0, rolling_window_size=128)
        self.short_term_memory = ShortTermMemory(db_path=self.db_path, capacity_gb=1.0)
        self.long_term_memory = LongTermMemory(db_path=self.db_path, capacity_gb=9.0)
        self._init_unified_database()
    
    def _init_unified_database(self):
        """
        Initialize unified database with cross-references and engram support.
        - Ensures all advanced features: compression, tagging, chunking, crosslinks, feedback fields, memory order, merged provenance, etc.
        - Research-aligned: see README.md, idea.txt, Zolfagharinejad et al., 2024 (EPJ B), Ren & Xia, 2024 (arXiv).
        """
        assert isinstance(self.db_path, str)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Unified memory mapping table (crosslinks basic/advanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS unified_memory_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                basic_memory_id INTEGER,
                advanced_memory_id INTEGER,
                mapping_type TEXT DEFAULT 'auto',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (basic_memory_id) REFERENCES memories (id),
                FOREIGN KEY (advanced_memory_id) REFERENCES advanced_memories (id)
            )
        """)
        
        # Memory access tracking (for feedback, provenance, chunking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                access_type TEXT,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Engram table (supports chunking, tagging, merged provenance, memory order)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                memory_ids TEXT, -- JSON list of memory IDs
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                merged_from TEXT, -- JSON list of engram IDs
                memory_order INTEGER DEFAULT 1,
                feedback TEXT, -- Feedback field for research alignment
                chunk_hash TEXT -- For chunking and deduplication
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_memory(self, text: str, memory_type: Optional[str] = 'general', 
                   priority: float = 0.5, context: Optional[str] = None, 
                   tags: Optional[List[str]] = None, use_advanced: bool = True,
                   create_reminder: bool = False, memory_order: int = 1) -> Dict[str, Any]:
        """
        Add a memory using the three-tier system:
        - Stage 1: Add to WorkingMemory (immediate access)
        - Stage 2: Promote to ShortTermMemory (encoded, vectorized)
        - Stage 3: Consolidate to LongTermMemory (compressed, associated)
        """
        key = hashlib.sha256(f"{text}-{time.time()}".encode()).hexdigest()[:16]
        result = {'working_id': None, 'short_term_id': None, 'long_term_id': None, 'success': False}
        try:
            # 1. Add to WorkingMemory
            wm_success = self.working_memory.add(key, text, context or "default", priority)
            result['working_id'] = key if wm_success else None
            # 2. Promote to ShortTermMemory if needed
            if use_advanced:
                stm_success = self.short_term_memory.add(key, text, context or "default", priority, memory_type or "general", tags)
                result['short_term_id'] = key if stm_success else None
                # 3. Consolidate to LongTermMemory if high priority or by policy
                if priority > 0.8 or memory_order == 1:
                    ltm_success = self.long_term_memory.add(key, text, category=memory_type or "general", tags=tags, importance_score=priority)
                    result['long_term_id'] = key if ltm_success else None
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
        return result
    
    def _create_memory_mapping(self, basic_id: int, advanced_id: int):
        """Create mapping between basic and advanced memories."""
        try:
            # self.db_path is always a string, guaranteed by __init__ (linter false positive)
            conn = sqlite3.connect(str(self.db_path), timeout=60.0)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO unified_memory_mapping (basic_memory_id, advanced_memory_id)
                VALUES (?, ?)
            """, (basic_id, advanced_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not create memory mapping: {e}")
    
    def search_memories(self, query: str, limit: Optional[int] = 10, memory_type: Optional[str] = '', memory_order: Optional[int] = None, use_vector_search: bool = True) -> List[Dict[str, Any]]:
        """Search memories using the unified system, with memory order filter."""
        results = []
        # Ensure types
        if memory_type is None:
            memory_type = ''
        if limit is None or not isinstance(limit, int):
            try:
                limit = int(limit) if limit is not None else 10
            except Exception:
                limit = 10
        if memory_order is not None:
            adv_results = self.advanced_memory.vector_search(query, limit, memory_type)
            adv_results = [r for r in adv_results if r.get('memory_order') == memory_order]
            for result in adv_results:
                result['search_type'] = 'advanced'
                result['source'] = 'advanced_memory'
                results.append(result)
            return results[:limit]
        if use_vector_search:
            advanced_results = self.advanced_memory.vector_search(query, limit, memory_type)
        else:
            advanced_results = self.basic_memory.search_memories(query, limit, memory_type)
        for result in advanced_results:
            result['search_type'] = 'advanced'
            result['source'] = 'advanced_memory'
            results.append(result)
        results.sort(key=lambda x: x.get('priority', 0) + x.get('similarity', 0), reverse=True)
        return results[:limit]
    
    def get_memory_details(self, memory_id: int, include_relationships: bool = True,
                          include_quality_report: bool = True) -> Dict[str, Any]:
        """Get comprehensive memory details."""
        details = {}
        
        # Try to get from basic memory first
        basic_memory = self.basic_memory.get_memory(memory_id)
        if basic_memory:
            details['basic_memory'] = basic_memory
            details['source'] = 'basic'
        
        # Try to get from advanced memory
        advanced_memory = self.advanced_memory.get_memory(memory_id)
        if advanced_memory:
            details['advanced_memory'] = advanced_memory
            details['source'] = 'advanced'
        
        # Get relationships if requested
        if include_relationships and 'advanced_memory' in details:
            relationships = self.advanced_memory.get_memory_relationships(memory_id)
            details['relationships'] = relationships
        
        # Get quality report if requested
        if include_quality_report and 'advanced_memory' in details:
            quality_report = self.advanced_memory.get_memory_quality_report(memory_id)
            details['quality_report'] = quality_report
        
        # Get related reminders
        reminders = self.reminder_engine.check_due_reminders()
        memory_reminders = [r for r in reminders if r['memory_id'] == memory_id]
        details['reminders'] = memory_reminders
        
        return details
    
    def add_task(self, description: str, priority: int = 0, parent_id: Optional[int] = None,
                 create_reminder: bool = False) -> Dict[str, Any]:
        """Add a task with optional reminder."""
        result = {
            'task_id': None,
            'reminder_id': None,
            'success': False
        }
        try:
            safe_parent_id = parent_id if parent_id is not None else 0
            task_id = self.basic_memory.add_task(description, priority, safe_parent_id)
            result['task_id'] = task_id
            if create_reminder:
                # Create a reminder directly linked to the task
                reminder_id = self.reminder_engine.create_task_reminder(
                    task_id=task_id, interval_hours=24, reminder_type='task_feedback', trigger_conditions=None
                )
                result['reminder_id'] = reminder_id
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
        return result
    
    def get_tasks(self, status: Optional[str] = None, include_reminders: bool = True) -> List[Dict[str, Any]]:
        """Get tasks with optional reminder information."""
        safe_status = status if status is not None else ''
        tasks = self.basic_memory.get_tasks(safe_status)
        
        if include_reminders:
            reminders = self.reminder_engine.check_due_reminders()
            reminder_map = {r['memory_id']: r for r in reminders}
            
            for task in tasks:
                task['reminders'] = reminder_map.get(task['id'], None)
        
        return tasks
    
    def complete_task(self, task_id: int, feedback: Optional[str] = None, 
                     impact: int = 0, principle: Optional[str] = None) -> Dict[str, Any]:
        """Complete a task with optional feedback."""
        result = {
            'success': False,
            'feedback_id': None
        }
        
        try:
            # Complete the task
            success = self.basic_memory.complete_task(task_id)
            result['success'] = success
            
            # Add feedback if provided
            if feedback and success:
                safe_principle = principle if principle is not None else ''
                feedback_id = self.basic_memory.add_feedback(task_id, feedback, impact, safe_principle)
                result['feedback_id'] = feedback_id
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_due_reminders(self, current_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all due reminders with enhanced context."""
        safe_context = current_context if current_context is not None else ''
        reminders = self.reminder_engine.check_due_reminders(safe_context)
        
        # Enhance reminders with memory details
        enhanced_reminders = []
        for reminder in reminders:
            memory_details = self.get_memory_details(reminder['memory_id'], 
                                                   include_relationships=False,
                                                   include_quality_report=False)
            
            enhanced_reminder = {
                **reminder,
                'memory_details': memory_details.get('basic_memory') or memory_details.get('advanced_memory')
            }
            enhanced_reminders.append(enhanced_reminder)
        
        return enhanced_reminders
    
    def process_reminder_feedback(self, reminder_id: int, feedback_score: int,
                                response_time_seconds: Optional[int] = None) -> bool:
        """Process feedback for a reminder."""
        safe_response_time = response_time_seconds if response_time_seconds is not None else 0
        return self.reminder_engine.process_reminder_feedback(
            reminder_id, feedback_score, safe_response_time
        )
    
    def get_memory_suggestions(self, context: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get personalized memory suggestions."""
        safe_context = context if context is not None else ''
        suggestions = []
        
        # Get reminder suggestions
        reminder_suggestions = self.reminder_engine.get_personalized_reminder_suggestions(safe_context)
        suggestions.extend(reminder_suggestions)
        
        # Get related memories based on context
        if safe_context:
            related_memories = self.search_memories(safe_context, limit=limit, memory_type='advanced', use_vector_search=True)
            for memory in related_memories:
                suggestions.append({
                    'memory_id': memory['id'],
                    'text': memory['text'][:100] + '...' if len(memory['text']) > 100 else memory['text'],
                    'memory_type': memory['memory_type'],
                    'category': memory.get('category'),
                    'suggestion_score': memory.get('similarity', 0.5),
                    'suggested_reminder_type': 'context_aware',
                    'reason': f"Related to current context: {safe_context}"
                })
        
        # Sort by suggestion score and return top results
        suggestions.sort(key=lambda x: x['suggestion_score'], reverse=True)
        return suggestions[:limit]
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {}
        
        # Basic memory statistics
        basic_stats = self.basic_memory.get_statistics()
        stats['basic_memory'] = basic_stats
        
        # Advanced memory statistics
        advanced_stats = self.advanced_memory.get_statistics()
        stats['advanced_memory'] = advanced_stats
        
        # Reminder statistics
        reminder_stats = self.reminder_engine.get_reminder_statistics()
        stats['reminders'] = reminder_stats
        
        # Unified statistics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM unified_memory_mapping")
        total_mappings = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM memory_access_log 
            WHERE timestamp >= datetime('now', '-7 days')
        """)
        weekly_accesses = cursor.fetchone()[0]
        
        conn.close()
        
        stats['unified'] = {
            'total_memory_mappings': total_mappings,
            'weekly_memory_accesses': weekly_accesses,
            'system_health': self._calculate_system_health(basic_stats, advanced_stats, reminder_stats)
        }
        
        return stats
    
    def _calculate_system_health(self, basic_stats: Dict, advanced_stats: Dict, 
                               reminder_stats: Dict) -> str:
        """Calculate overall system health."""
        # Simple health calculation based on various metrics
        total_memories = basic_stats.get('total_memories', 0)
        total_relationships = advanced_stats.get('total_relationships', 0)
        avg_effectiveness = reminder_stats.get('average_effectiveness', 0.0)
        
        if total_memories == 0:
            return 'empty'
        elif total_memories < 10:
            return 'growing'
        elif avg_effectiveness > 0.7 and total_relationships > total_memories * 0.5:
            return 'excellent'
        elif avg_effectiveness > 0.5:
            return 'good'
        else:
            return 'needs_attention'
    
    def export_memory_data(self, format: str = 'json') -> str:
        """Export memory data in specified format."""
        if format.lower() == 'json':
            return self._export_to_json()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_json(self) -> str:
        """Export all memory data to JSON format."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'basic_memories': [],
            'advanced_memories': [],
            'tasks': [],
            'feedback': [],
            'reminders': []
        }
        
        # Export basic memories
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories")
        basic_memories = cursor.fetchall()
        for memory in basic_memories:
            export_data['basic_memories'].append({
                'id': memory[0],
                'text': memory[1],
                'memory_type': memory[2],
                'priority': memory[3],
                'context': memory[4],
                'tags': memory[5],
                'created_at': memory[6],
                'updated_at': memory[7]
            })
        
        # Export tasks
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()
        for task in tasks:
            export_data['tasks'].append({
                'id': task[0],
                'description': task[1],
                'status': task[2],
                'priority': task[3],
                'parent_id': task[4],
                'created_at': task[5],
                'completed_at': task[6]
            })
        
        conn.close()
        
        return json.dumps(export_data, indent=2)
    
    def import_memory_data(self, data: str, format: str = 'json') -> Dict[str, Any]:
        """Import memory data from specified format."""
        if format.lower() == 'json':
            return self._import_from_json(data)
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def _import_from_json(self, data: str) -> Dict[str, Any]:
        """Import memory data from JSON format."""
        try:
            import_data = json.loads(data)
            result = {
                'imported_memories': 0,
                'imported_tasks': 0,
                'errors': []
            }
            
            # Import basic memories
            for memory_data in import_data.get('basic_memories', []):
                try:
                    self.basic_memory.add_memory(
                        memory_data['text'],
                        memory_data.get('memory_type', 'general'),
                        memory_data.get('priority', 0.5),
                        memory_data.get('context'),
                        memory_data.get('tags')
                    )
                    result['imported_memories'] += 1
                except Exception as e:
                    result['errors'].append(f"Memory import error: {str(e)}")
            
            # Import tasks
            for task_data in import_data.get('tasks', []):
                try:
                    self.basic_memory.add_task(
                        task_data['description'],
                        task_data.get('priority', 0),
                        task_data.get('parent_id')
                    )
                    result['imported_tasks'] += 1
                except Exception as e:
                    result['errors'].append(f"Task import error: {str(e)}")
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {str(e)}")

    def vector_search(self, query: str, limit: int = 10, memory_type: Optional[str] = None, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Expose advanced vector search via the unified interface."""
        try:
            safe_memory_type = memory_type if memory_type is not None else ''
            return self.advanced_memory.vector_search(query, limit, safe_memory_type, min_similarity)
        except Exception as e:
            print(f"Error in vector_search: {e}")
            return []

    def get_memory_relationships(self, memory_id: int) -> List[Dict[str, Any]]:
        """Expose memory relationships via the unified interface."""
        try:
            return self.advanced_memory.get_memory_relationships(memory_id)
        except Exception as e:
            print(f"Error in get_memory_relationships: {e}")
            return []

    def get_memory_quality_report(self, memory_id: int) -> Dict[str, Any]:
        """
        Expose enhanced memory quality/confidence scoring via the unified interface.
        
        This method uses the comprehensive MemoryQualityAssessment system if available,
        otherwise falls back to the basic quality report from advanced_memory.
        """
        try:
            # Try to use the enhanced quality assessment system
            from .memory_quality_assessment import MemoryQualityAssessment
            quality_assessment = MemoryQualityAssessment(self.db_path)
            return quality_assessment.get_memory_quality_report(memory_id)
        except ImportError:
            # Fall back to advanced memory's quality report
            try:
                return self.advanced_memory.get_memory_quality_report(memory_id)
            except Exception as e:
                print(f"Error in get_memory_quality_report: {e}")
                return {}

    def create_engram(self, title: str, description: str = "", memory_ids: Optional[list] = None, tags: Optional[list] = None) -> int:
        """Create a new engram and return its ID."""
        assert isinstance(self.db_path, str)
        memory_ids_json = json.dumps(memory_ids or [])
        tags_json = json.dumps(tags or [])
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO engrams (title, description, memory_ids, tags)
            VALUES (?, ?, ?, ?)
            """,
            (title, description, memory_ids_json, tags_json)
        )
        engram_id = cursor.lastrowid
        conn.commit()
        conn.close()
        if engram_id is None:
            return -1
        return engram_id

    def get_engram(self, engram_id: int) -> dict:
        """Retrieve an engram by ID."""
        assert isinstance(self.db_path, str)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM engrams WHERE id = ?", (engram_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {}
        return {
            'id': row[0],
            'title': row[1],
            'description': row[2],
            'memory_ids': json.loads(row[3] or '[]'),
            'tags': json.loads(row[4] or '[]'),
            'created_at': row[5],
            'updated_at': row[6],
            'merged_from': json.loads(row[7] or '[]'),
            'memory_order': row[8],
            'feedback': row[9],
            'chunk_hash': row[10]
        }

    def list_engrams(self) -> List[dict]:
        """List all engrams."""
        assert isinstance(self.db_path, str)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description, memory_ids, tags, created_at, updated_at, merged_from, memory_order, feedback, chunk_hash FROM engrams")
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'memory_ids': json.loads(row[3] or '[]'),
                'tags': json.loads(row[4] or '[]'),
                'created_at': row[5],
                'updated_at': row[6],
                'merged_from': json.loads(row[7] or '[]'),
                'memory_order': row[8],
                'feedback': row[9],
                'chunk_hash': row[10]
            }
            for row in rows
        ]

    def update_engram(self, engram_id: int, title: Optional[str] = None, description: Optional[str] = None, memory_ids: Optional[list] = None, tags: Optional[list] = None) -> bool:
        """Update an engram's fields."""
        assert isinstance(self.db_path, str)
        engram = self.get_engram(engram_id)
        if not engram:
            return False
        title = title if title is not None else engram['title']
        description = description if description is not None else engram['description']
        memory_ids_json = json.dumps(memory_ids if memory_ids is not None else engram['memory_ids'])
        tags_json = json.dumps(tags if tags is not None else engram['tags'])
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE engrams SET title = ?, description = ?, memory_ids = ?, tags = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """,
            (title, description, memory_ids_json, tags_json, engram_id)
        )
        conn.commit()
        conn.close()
        return True

    def merge_engrams(self, engram_ids: list, new_title: str, new_description: str = "", tags: Optional[list] = None) -> int:
        """Merge multiple engrams into a new one, tracking provenance."""
        assert isinstance(self.db_path, str)
        all_memory_ids = []
        for eid in engram_ids:
            engram = self.get_engram(eid)
            all_memory_ids.extend(engram.get('memory_ids', []))
        merged_from_json = json.dumps(engram_ids)
        tags_json = json.dumps(tags or [])
        memory_ids_json = json.dumps(list(set(all_memory_ids)))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO engrams (title, description, memory_ids, tags, merged_from)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_title, new_description, memory_ids_json, tags_json, merged_from_json)
        )
        new_engram_id = cursor.lastrowid
        conn.commit()
        conn.close()
        if new_engram_id is None:
            return -1
        return new_engram_id

    def advanced_engram_features(self, engrams: list, feedback: Optional[list] = None, coding_model=None, diffusion_model=None, selection_strategy=None, backend=None) -> dict:
        """
        Advanced engram features:
        - Dynamic coding models for engram compression, recall, and mutation.
        - Diffusion models for engram merging, synthesis, and probabilistic recall.
        - Feedback-driven selection of engram representations.
        - Pluggable model backends for experimentation and AB testing.
        See idea.txt for requirements and TODO_DEVELOPMENT_PLAN.md for roadmap.
        """
        if not engrams:
            return {"status": "no_engrams", "result": None}
        # Use provided or default models
        coding = coding_model or (lambda x: x)
        diffusion = diffusion_model or (lambda x, y: x + y if isinstance(x, list) and isinstance(y, list) else [x, y])
        selection = selection_strategy or (lambda engrams: engrams[0] if engrams else None)
        # Step 1: Encode engrams
        encoded_engrams = [coding(e) for e in engrams]
        # Step 2: Optionally merge engrams (pairwise for demo)
        merged = encoded_engrams[0]
        for e in encoded_engrams[1:]:
            merged = diffusion(merged, e)
        # Step 3: Select best engram (feedback-driven or default)
        selected = selection(encoded_engrams)
        # Step 4: Store merged engram in unified memory
        engram_id = self.create_engram(
            title="Merged Engram",
            description="Auto-generated by advanced_engram_features",
            memory_ids=[e.get('id', 0) for e in engrams if 'id' in e],
            tags=["auto", "merged"]
        )
        return {
            "status": "processed",
            "merged": merged,
            "selected": selected,
            "engram_id": engram_id
        }

    def set_coding_model(self, coding_model):
        """Set the dynamic coding model for engram operations."""
        self.coding_model = coding_model

    def set_diffusion_model(self, diffusion_model):
        """Set the diffusion model for engram merging/synthesis."""
        self.diffusion_model = diffusion_model

    def set_selection_strategy(self, selection_strategy):
        """Set the feedback-driven selection strategy for engrams."""
        self.selection_strategy = selection_strategy

    def set_engram_backend(self, backend):
        """Set the pluggable backend for engram experimentation and AB testing."""
        self.engram_backend = backend

    # --- TESTS FOR UNIFIED MEMORY OPERATIONS ---
    def _test_add_and_get_memory(self):
        """Test adding and retrieving a memory."""
        # Pylint false positive: add_memory always returns a dict, never None
        # See: https://pylint.readthedocs.io/en/stable/user_guide/messages/error/assignment-from-none.html
        result = self.add_memory(text="Test memory", memory_type="test", priority=1.0, context="", tags=[])  # pylint: disable=assignment-from-none
        mem_id = result.get("basic_memory_id")
        assert mem_id is not None, "add_memory did not return a valid basic_memory_id"
        mem = self.basic_memory.get_memory(mem_id)
        assert mem is not None and mem.get('text') == "Test memory"
        return True

    def _test_vector_search(self):
        """Test vector search returns results and correct structure."""
        results = self.vector_search("Test", limit=1)
        assert isinstance(results, list)
        return True

    def _test_reminder_engine(self):
        """Test reminder engine basic functionality."""
        # get_all_reminders is a dynamic plugin method and may not be present in all EnhancedReminderEngine implementations.
        # type: ignore is justified here per project standards and linter docs.
        if hasattr(self.reminder_engine, 'get_all_reminders'):
            reminders = self.reminder_engine.get_all_reminders()  # type: ignore[attr-defined]
            assert isinstance(reminders, list)
            return True
        return False

    # --- END TESTS ---

    # --- DOCUMENTATION ---
    """
    UnifiedMemoryManager API and CLI Usage
    ======================================
    
    - The UnifiedMemoryManager class provides a unified interface for managing all memory types (basic, advanced/vector, reminders).
    - All memory operations are persisted to the database for full recovery and auditability.
    - CLI commands are available for adding, searching, and managing memories, reminders, and encoders.
    - See README.md and CLI help for usage examples.
    
    CLI Usage Examples:
    -------------------
    - Add a memory:
        $ mcp add-memory --text "Remember to refactor module X" --type "reminder" --priority 0.8 --tags "refactor,urgent" --encoder tfidf
    - Search memories:
        $ mcp search-memories --query "refactor" --limit 5
    - Vector search:
        $ mcp vector-search --query "optimize performance" --encoder rabitq
    - Add a reminder:
        $ mcp add-reminder --text "Check logs" --interval 7d
    - Get memory statistics:
        $ mcp memory-stats
    
    Best Practices:
    ---------------
    - Use advanced encoders (TFIDF, RaBitQ) for high-quality vector search and memory compression.
    - Regularly review and update reminders to maintain project alignment and avoid context loss.
    - Use tags and memory order to organize and prioritize information for LLM and user workflows.
    - Refer to idea.txt for the vision and requirements that guide memory system design and usage.
    
    References:
    -----------
    - idea.txt: Project vision, requirements, and best practices
    - README.md: General usage and integration notes
    - CLI --help: Command-specific options and examples
    """
    # --- END DOCUMENTATION --- 

    def adapt_task_reminder_on_feedback(self, task_id: int, status: str = 'flagged', interval_hours: int = 12):
        """Create or adapt a reminder for a flagged/critical task. If a reminder exists, update its interval/type; else, create one. Log actions for traceability."""
        reminders = self.reminder_engine.get_reminders_by_task_id(int(task_id))
        if reminders:
            for reminder in reminders:
                self.reminder_engine.update_reminder_interval(int(reminder['id']), int(interval_hours))
                self.reminder_engine.update_reminder_type(int(reminder['id']), str(f'task_{status}'))
                print(f"[UnifiedMemoryManager] Updated reminder {str(reminder['id'])} for task {str(task_id)} to type 'task_{status}' and interval {str(interval_hours)}h.")
        else:
            reminder_id = self.reminder_engine.create_task_reminder(
                task_id=int(task_id), interval_hours=int(interval_hours), reminder_type=str(f'task_{status}'), trigger_conditions=None
            )
            print(f"[UnifiedMemoryManager] Created new reminder {str(reminder_id)} for task {str(task_id)} with type 'task_{status}' and interval {str(interval_hours)}h.")

    def promote_memory_order(self, memory_id: int) -> bool:
        """Promote a memory to a higher order (e.g., 2->1, 3->2)."""
        memory = self.advanced_memory.get_memory(memory_id)
        if not memory:
            return False
        current_order = memory.get('memory_order', 1)
        if current_order <= 1:
            return False  # Already highest
        return self.advanced_memory.update_memory_order(memory_id, current_order - 1)

    def demote_memory_order(self, memory_id: int) -> bool:
        """Demote a memory to a lower order (e.g., 1->2, 2->3)."""
        memory = self.advanced_memory.get_memory(memory_id)
        if not memory:
            return False
        current_order = memory.get('memory_order', 1)
        if current_order >= 3:
            return False  # Already lowest
        return self.advanced_memory.update_memory_order(memory_id, current_order + 1)

    def dynamic_expand_memory(self, new_input: str, context: str = '', tags: Optional[List[str]] = None) -> int:
        """Expose dynamic expansion for unified memory."""
        try:
            return self.basic_memory.dynamic_expand(new_input, context, tags)
        except Exception as e:
            print(f"[UnifiedMemoryManager] Dynamic expand failed: {e}")
            return -1

    def generalize_memories(self, similarity_threshold: float = 0.8) -> Optional[int]:
        """Expose generalization for unified memory."""
        try:
            return self.basic_memory.generalize_memories(similarity_threshold)
        except Exception as e:
            print(f"[UnifiedMemoryManager] Generalization failed: {e}")
            return None

    def prune_memories(self, min_priority: float = 0.2, max_age_days: int = 180) -> int:
        """Expose pruning for unified memory."""
        try:
            return self.basic_memory.prune_memories(min_priority, max_age_days)
        except Exception as e:
            print(f"[UnifiedMemoryManager] Pruning failed: {e}")
            return 0    

    def detect_memory_relationships(self, memory_id: int, max_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Detect relationships between a memory and other memories in the system.
        
        Args:
            memory_id: ID of the memory to analyze
            max_candidates: Maximum number of candidate memories to check
            
        Returns:
            List of detected relationships with scores
        """
        try:
            # Try to use the enhanced quality assessment system
            from .memory_quality_assessment import MemoryQualityAssessment
            quality_assessment = MemoryQualityAssessment(self.db_path)
            return quality_assessment.detect_memory_relationships(memory_id, max_candidates)
        except ImportError:
            # Fall back to basic relationship detection
            try:
                return self.advanced_memory.get_memory_relationships(memory_id)
            except Exception as e:
                print(f"Error in detect_memory_relationships: {e}")
                return []
    
    def store_memory_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        Store detected relationships in the database.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Number of relationships stored
        """
        try:
            # Try to use the enhanced quality assessment system
            from .memory_quality_assessment import MemoryQualityAssessment
            quality_assessment = MemoryQualityAssessment(self.db_path)
            return quality_assessment.store_memory_relationships(relationships)
        except ImportError:
            # No fallback for storing enhanced relationships
            print("Enhanced memory relationship storage not available")
            return 0
    
    def consolidate_memories(self, memory_ids: List[int], consolidation_type: str = 'merge') -> Dict[str, Any]:
        """
        Consolidate multiple memories into a single memory.
        
        Args:
            memory_ids: List of memory IDs to consolidate
            consolidation_type: Type of consolidation ('merge', 'summarize', 'compress')
            
        Returns:
            Dictionary with consolidation results
        """
        try:
            # Try to use the enhanced quality assessment system
            from .memory_quality_assessment import MemoryQualityAssessment
            quality_assessment = MemoryQualityAssessment(self.db_path)
            return quality_assessment.consolidate_memories(memory_ids, consolidation_type)
        except ImportError:
            # No fallback for memory consolidation
            return {'error': 'Memory consolidation not available'}
    
    def optimize_memory_storage(self, max_candidates: int = 100, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Optimize memory storage by identifying and consolidating similar memories.
        
        Args:
            max_candidates: Maximum number of memories to analyze
            similarity_threshold: Minimum similarity threshold for consolidation
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Try to use the enhanced quality assessment system
            from .memory_quality_assessment import MemoryQualityAssessment
            quality_assessment = MemoryQualityAssessment(self.db_path)
            return quality_assessment.optimize_memory_storage(max_candidates, similarity_threshold)
        except ImportError:
            # No fallback for memory optimization
            return {'error': 'Memory optimization not available'}

    def cross_tier_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across all three tiers and merge results by relevance.
        """
        results = []
        # WorkingMemory search
        wm_results = self.working_memory.get_all()
        for r in wm_results:
            if query.lower() in str(r.get('data', '')).lower():
                results.append({'id': r.get('key'), 'data': r.get('data'), 'priority': r.get('priority', 0.5), 'source': 'working'})
        # ShortTermMemory search
        stm_results = self.short_term_memory.search(query, limit=limit)
        for r in stm_results:
            results.append({'id': r.get('key'), 'data': r.get('data'), 'priority': r.get('priority', 0.5), 'source': 'short_term'})
        # LongTermMemory semantic search
        ltm_results = self.long_term_memory.semantic_search(query, limit=limit)
        for r in ltm_results:
            results.append({'id': r.get('key'), 'data': r.get('content'), 'priority': r.get('importance_score', 0.5), 'source': 'long_term'})
        # Sort and deduplicate by relevance/priority
        results = sorted(results, key=lambda x: x.get('priority', 0), reverse=True)
        seen = set()
        deduped = []
        for r in results:
            mid = r.get('id')
            if mid and mid not in seen:
                deduped.append(r)
                seen.add(mid)
        return deduped[:limit]

    def consolidate_workflow(self):
        """
        Automatic memory consolidation workflow:
        - Move items from WorkingMemory to ShortTermMemory based on access/priority
        - Consolidate ShortTermMemory to LongTermMemory periodically or by policy
        """
        # Promote from WorkingMemory to ShortTermMemory
        for item in self.working_memory.get_all():
            key = item['key']
            text = item['data']
            context = item.get('context', 'default')
            priority = item.get('priority', 0.5)
            self.short_term_memory.add(key, text, context, priority)
            self.working_memory.remove(key, context)
        # Consolidate ShortTermMemory to LongTermMemory
        for item in self.short_term_memory.get_all():
            key = item['key']
            text = item['data']
            context = item.get('context', 'default')
            priority = item.get('priority', 0.5)
            self.long_term_memory.add(key, text, category=context, importance_score=priority)
            self.short_term_memory.remove(key, context)
            self.long_term_memory.remove(key)