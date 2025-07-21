#!/usr/bin/env python3
"""
Enhanced Reminder Engine
Implements spaced repetition, adaptive spacing, and personalized forgetting curves.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("reminder_engine")

class EnhancedReminderEngine:
    """Enhanced reminder engine with advanced scheduling algorithms."""
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the reminder engine."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'unified_memory.db')
        self.db_path: str = str(db_path)
        self._init_reminder_database()
    
    def _init_reminder_database(self) -> None:
        """Initialize reminder-specific database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced reminders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER,
                    task_id INTEGER,
                    reminder_type TEXT,
                    trigger_conditions TEXT,
                    next_reminder TIMESTAMP,
                    reminder_count INTEGER DEFAULT 0,
                    last_triggered TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    effectiveness_score REAL DEFAULT 0.0,
                    user_feedback_history TEXT,
                    easiness_factor REAL DEFAULT 2.5,
                    context_history TEXT,
                    FOREIGN KEY (memory_id) REFERENCES advanced_memories (id)
                )
            """)
            
            # Context patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User retention rates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_retention_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    content_type TEXT,
                    retention_rate REAL DEFAULT 0.5,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize reminder database: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def create_spaced_repetition_reminder(self, memory_id: int, initial_interval_hours: int = 24) -> int:
        """Create a spaced repetition reminder for a memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            next_reminder = datetime.now() + timedelta(hours=initial_interval_hours)
            
            cursor.execute("""
                INSERT INTO enhanced_reminders 
                (memory_id, reminder_type, next_reminder, easiness_factor)
                VALUES (?, ?, ?, ?)
            """, (memory_id, 'spaced_repetition', next_reminder, 2.5))
            
            reminder_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created spaced repetition reminder for memory_id={memory_id}, id={reminder_id}")
            return int(reminder_id) if reminder_id is not None else -1
        except Exception as e:
            logger.error(f"Failed to create spaced repetition reminder: {e}")
            return -1
        finally:
            if 'conn' in locals():
                conn.close()
    
    def create_context_aware_reminder(self, memory_id: int, context_keywords: List[str], 
                                    trigger_context: Optional[str] = None) -> int:
        """Create a context-aware reminder."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            trigger_conditions = {
                'context_keywords': context_keywords,
                'trigger_context': trigger_context,
                'type': 'context_aware'
            }
            
            cursor.execute("""
                INSERT INTO enhanced_reminders 
                (memory_id, reminder_type, trigger_conditions, next_reminder)
                VALUES (?, ?, ?, ?)
            """, (memory_id, 'context_aware', json.dumps(trigger_conditions), datetime.now()))
            
            reminder_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created context-aware reminder for memory_id={memory_id}, id={reminder_id}")
            return int(reminder_id) if reminder_id is not None else -1
        except Exception as e:
            logger.error(f"Failed to create context-aware reminder: {e}")
            return -1
        finally:
            if 'conn' in locals():
                conn.close()
    
    def create_adaptive_reminder(self, memory_id: int, base_interval_hours: int = 24,
                               adaptation_factor: float = 1.0) -> int:
        """Create an adaptive reminder that adjusts based on user feedback."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate initial interval based on adaptation factor
            initial_interval = base_interval_hours * adaptation_factor
            next_reminder = datetime.now() + timedelta(hours=initial_interval)
            
            trigger_conditions = {
                'base_interval_hours': base_interval_hours,
                'adaptation_factor': adaptation_factor,
                'type': 'adaptive'
            }
            
            cursor.execute("""
                INSERT INTO enhanced_reminders 
                (memory_id, reminder_type, trigger_conditions, next_reminder, easiness_factor)
                VALUES (?, ?, ?, ?, ?)
            """, (memory_id, 'adaptive', json.dumps(trigger_conditions), next_reminder, 2.5))
            
            reminder_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created adaptive reminder for memory_id={memory_id}, id={reminder_id}")
            return int(reminder_id) if reminder_id is not None else -1
        except Exception as e:
            logger.error(f"Failed to create adaptive reminder: {e}")
            return -1
        finally:
            if 'conn' in locals():
                conn.close()
    
    def create_task_reminder(self, task_id: int, interval_hours: int = 24, reminder_type: str = 'task_feedback', trigger_conditions: Optional[dict] = None) -> int:
        """Create a reminder directly linked to a task (not just memory)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            next_reminder = datetime.now() + timedelta(hours=interval_hours)
            trigger_conditions_json = json.dumps(trigger_conditions) if trigger_conditions is not None else '{}'
            cursor.execute(
                """
                INSERT INTO enhanced_reminders 
                (task_id, reminder_type, trigger_conditions, next_reminder, easiness_factor)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(task_id), str(reminder_type), trigger_conditions_json, next_reminder, 2.5)
            )
            reminder_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created task reminder for task_id={task_id}, id={reminder_id}")
            return int(reminder_id) if reminder_id is not None else -1
        except Exception as e:
            logger.error(f"Failed to create task reminder: {e}")
            return -1
        finally:
            if 'conn' in locals():
                conn.close()
    
    def check_due_reminders(self, current_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Check for due reminders, including context-aware ones."""
        due_reminders: List[Dict[str, Any]] = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check time-based reminders
            cursor.execute("""
                SELECT r.id, r.memory_id, r.reminder_type, r.reminder_count,
                       r.easiness_factor, r.trigger_conditions, r.next_reminder,
                       m.text, m.memory_type, m.category
                FROM enhanced_reminders r
                JOIN advanced_memories m ON r.memory_id = m.id
                WHERE r.next_reminder <= ? AND r.reminder_type IN ('spaced_repetition', 'adaptive')
            """, (datetime.now(),))
            
            time_based_reminders = cursor.fetchall()
            
            for reminder in time_based_reminders:
                reminder_id, memory_id, reminder_type, reminder_count, easiness_factor, \
                trigger_conditions, next_reminder, text, memory_type, category = reminder
                
                due_reminders.append({
                    'reminder_id': reminder_id,
                    'memory_id': memory_id,
                    'reminder_type': reminder_type,
                    'reminder_count': reminder_count,
                    'easiness_factor': easiness_factor,
                    'text': text,
                    'memory_type': memory_type,
                    'category': category,
                    'trigger_reason': 'time_based'
                })
            
            # Check context-aware reminders if context is provided
            if current_context:
                cursor.execute("""
                    SELECT r.id, r.memory_id, r.reminder_type, r.reminder_count,
                           r.easiness_factor, r.trigger_conditions, r.next_reminder,
                           m.text, m.memory_type, m.category
                    FROM enhanced_reminders r
                    JOIN advanced_memories m ON r.memory_id = m.id
                    WHERE r.reminder_type = 'context_aware'
                """)
                
                context_reminders = cursor.fetchall()
                
                for reminder in context_reminders:
                    reminder_id, memory_id, reminder_type, reminder_count, easiness_factor, \
                    trigger_conditions, next_reminder, text, memory_type, category = reminder
                    
                    try:
                        conditions = json.loads(trigger_conditions)
                        context_keywords = conditions.get('context_keywords', [])
                        
                        # Check if current context matches any keywords
                        context_lower = current_context.lower()
                        if any(keyword.lower() in context_lower for keyword in context_keywords):
                            due_reminders.append({
                                'reminder_id': reminder_id,
                                'memory_id': memory_id,
                                'reminder_type': reminder_type,
                                'reminder_count': reminder_count,
                                'easiness_factor': easiness_factor,
                                'text': text,
                                'memory_type': memory_type,
                                'category': category,
                                'trigger_reason': 'context_match',
                                'matched_keywords': [kw for kw in context_keywords if kw.lower() in context_lower]
                            })
                    
                    except (json.JSONDecodeError, KeyError):
                        continue
            return due_reminders
        except Exception as e:
            logger.error(f"Failed to check due reminders: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def process_reminder_feedback(self, reminder_id: int, feedback_score: int, response_time_seconds: int = 0) -> bool:
        """
        Process feedback for a reminder and update scheduling.
        Implements dynamic self-tuning of reminder intervals and effectiveness scores based on user feedback and performance metrics.
        See idea.txt line 185: all non-user-editable settings should be dynamically adjusting with all useful metrics.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current reminder data
            cursor.execute("""
                SELECT memory_id, reminder_type, reminder_count, easiness_factor, 
                       trigger_conditions, effectiveness_score, user_feedback_history
                FROM enhanced_reminders WHERE id = ?
            """, (reminder_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False
            
            memory_id, reminder_type, reminder_count, easiness_factor, \
            trigger_conditions, effectiveness_score, feedback_history_json = row
            
            # Update feedback history
            feedback_history = json.loads(feedback_history_json) if feedback_history_json else []
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'score': feedback_score,
                'response_time': response_time_seconds or 0
            }
            feedback_history.append(feedback_entry)
            
            # Update effectiveness score
            new_effectiveness = self._calculate_effectiveness_score(feedback_history)
            
            # Calculate next reminder interval based on feedback
            if reminder_type == 'spaced_repetition':
                new_interval = self._calculate_spaced_repetition_interval(
                    feedback_score, reminder_count, easiness_factor
                )
            elif reminder_type == 'adaptive':
                new_interval = self._calculate_adaptive_interval(
                    feedback_score, reminder_count, easiness_factor, trigger_conditions, reminder_id
                )
            else:
                new_interval = 24  # Default 24 hours
            
            # Update easiness factor (SuperMemo algorithm)
            new_easiness = self._update_easiness_factor(easiness_factor, feedback_score)
            
            # Schedule next reminder
            next_reminder = datetime.now() + timedelta(hours=new_interval)
            
            # Update reminder
            cursor.execute("""
                UPDATE enhanced_reminders 
                SET reminder_count = ?, easiness_factor = ?, next_reminder = ?,
                    effectiveness_score = ?, user_feedback_history = ?, last_triggered = ?
                WHERE id = ?
            """, (reminder_count + 1, new_easiness, next_reminder, 
                  new_effectiveness, json.dumps(feedback_history), datetime.now(), reminder_id))
            
            conn.commit()
            logger.info(f"Processed feedback for reminder_id={reminder_id}, score={feedback_score}")
            return True
        except Exception as e:
            logger.error(f"Failed to process feedback for reminder_id={reminder_id}: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _calculate_effectiveness_score(self, feedback_history: List[Dict[str, Any]]) -> float:
        """Calculate effectiveness score based on feedback history."""
        if not feedback_history:
            return 0.0
        
        # Calculate weighted average of recent feedback
        recent_feedback = feedback_history[-10:]  # Last 10 feedback entries
        total_weight = 0
        weighted_sum = 0
        
        for i, feedback in enumerate(recent_feedback):
            weight = i + 1  # More recent feedback has higher weight
            total_weight += weight
            weighted_sum += feedback['score'] * weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_spaced_repetition_interval(self, feedback_score: int, 
                                            reminder_count: int, easiness_factor: float) -> int:
        """Calculate next interval using spaced repetition algorithm."""
        if reminder_count == 0:
            return 24  # First reminder after 24 hours
        
        if reminder_count == 1:
            return 24 * 6  # Second reminder after 6 days
        
        # SuperMemo algorithm
        if feedback_score >= 4:  # Good response
            interval = int(24 * easiness_factor * (reminder_count - 1))
        elif feedback_score == 3:  # Medium response
            interval = int(24 * easiness_factor * (reminder_count - 1) * 0.5)
        else:  # Poor response
            interval = 24  # Reset to 24 hours
        
        return max(interval, 1)  # Minimum 1 hour
    
    def _calculate_adaptive_interval(self, feedback_score: int, reminder_count: int,
                                   easiness_factor: float, trigger_conditions: str, reminder_id: Optional[int] = None) -> int:
        """Calculate adaptive interval based on user performance."""
        try:
            conditions = json.loads(trigger_conditions) if trigger_conditions else {}
            base_interval = conditions.get('base_interval_hours', 24)
            adaptation_factor = conditions.get('adaptation_factor', 1.0)
        except (json.JSONDecodeError, KeyError, TypeError):
            base_interval = 24
            adaptation_factor = 1.0
        
        # Adjust adaptation factor based on feedback
        if feedback_score >= 4:
            adaptation_factor *= 1.2  # Increase interval
        elif feedback_score <= 2:
            adaptation_factor *= 0.8  # Decrease interval
        
        # Calculate new interval
        interval = int(base_interval * adaptation_factor * easiness_factor)
        
        # Update adaptation factor in database
        if reminder_id is not None:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                conditions['adaptation_factor'] = adaptation_factor
                cursor.execute("""
                    UPDATE enhanced_reminders 
                    SET trigger_conditions = ? WHERE id = ?
                """, (json.dumps(conditions), reminder_id))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to update adaptation factor for reminder_id={reminder_id}: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()
        
        return max(interval, 1)
    
    def _update_easiness_factor(self, current_easiness: float, feedback_score: int) -> float:
        """Update easiness factor using SuperMemo algorithm."""
        if feedback_score >= 4:
            # Good response - increase easiness
            new_easiness = current_easiness + 0.1
        elif feedback_score == 3:
            # Medium response - slight decrease
            new_easiness = current_easiness - 0.05
        else:
            # Poor response - significant decrease
            new_easiness = current_easiness - 0.2
        
        return max(new_easiness, 1.3)  # Minimum easiness factor
    
    def get_reminder_statistics(self) -> Dict[str, Any]:
        """Get reminder system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total reminders
        cursor.execute("SELECT COUNT(*) FROM enhanced_reminders")
        total_reminders = cursor.fetchone()[0]
        
        # Reminders by type
        cursor.execute("""
            SELECT reminder_type, COUNT(*) FROM enhanced_reminders 
            GROUP BY reminder_type
        """)
        reminders_by_type = dict(cursor.fetchall())
        
        # Average effectiveness
        cursor.execute("SELECT AVG(effectiveness_score) FROM enhanced_reminders")
        avg_effectiveness = cursor.fetchone()[0] or 0.0
        
        # Due reminders
        cursor.execute("""
            SELECT COUNT(*) FROM enhanced_reminders 
            WHERE next_reminder <= ?
        """, (datetime.now(),))
        due_reminders = cursor.fetchone()[0]
        
        # Reminder frequency distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN reminder_count = 0 THEN 'new'
                    WHEN reminder_count <= 3 THEN 'frequent'
                    WHEN reminder_count <= 10 THEN 'regular'
                    ELSE 'established'
                END as frequency_level,
                COUNT(*)
            FROM enhanced_reminders 
            GROUP BY frequency_level
        """)
        frequency_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_reminders': total_reminders,
            'reminders_by_type': reminders_by_type,
            'average_effectiveness': avg_effectiveness,
            'due_reminders': due_reminders,
            'frequency_distribution': frequency_distribution
        }
    
    def get_personalized_reminder_suggestions(self, user_context: str = None) -> List[Dict[str, Any]]:
        """Get personalized reminder suggestions based on context and history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        suggestions = []
        
        # Get memories that might need reminders
        cursor.execute("""
            SELECT m.id, m.text, m.memory_type, m.category, m.last_accessed,
                   m.access_count, m.quality_score
            FROM advanced_memories m
            LEFT JOIN enhanced_reminders r ON m.id = r.memory_id
            WHERE r.id IS NULL  -- No existing reminder
            ORDER BY m.last_accessed ASC, m.quality_score DESC
            LIMIT 10
        """)
        
        potential_reminders = cursor.fetchall()
        
        for memory_id, text, memory_type, category, last_accessed, \
            access_count, quality_score in potential_reminders:
            
            # Calculate suggestion score
            days_since_access = (datetime.now() - datetime.fromisoformat(last_accessed)).days if last_accessed else 30
            suggestion_score = (days_since_access * 0.3 + quality_score * 0.7) if last_accessed else 0.8
            
            if suggestion_score > 0.5:  # Only suggest if score is high enough
                reminder_type = self._suggest_reminder_type(memory_type, category, access_count)
                
                suggestions.append({
                    'memory_id': memory_id,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'memory_type': memory_type,
                    'category': category,
                    'suggestion_score': suggestion_score,
                    'suggested_reminder_type': reminder_type,
                    'reason': f"Last accessed {days_since_access} days ago, quality score: {quality_score:.2f}"
                })
        
        conn.close()
        return sorted(suggestions, key=lambda x: x['suggestion_score'], reverse=True)
    
    def _suggest_reminder_type(self, memory_type: str, category: str, access_count: int) -> str:
        """Suggest the best reminder type for a memory."""
        if access_count == 0:
            return 'spaced_repetition'  # New memory
        elif access_count <= 3:
            return 'adaptive'  # Learning memory
        elif category in ['code', 'documentation']:
            return 'context_aware'  # Context-dependent memory
        else:
            return 'spaced_repetition'  # General memory

    def get_reminders_by_task_id(self, task_id: int) -> list:
        """Return all reminders for a given task_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, task_id, reminder_type, next_reminder FROM enhanced_reminders WHERE task_id = ?", (task_id,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {'id': row[0], 'task_id': row[1], 'reminder_type': row[2], 'next_reminder': row[3]} for row in rows
        ]

    def update_reminder_interval(self, reminder_id: int, interval_hours: int):
        """Update the next_reminder time for a reminder."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        next_reminder = datetime.now() + timedelta(hours=interval_hours)
        cursor.execute("UPDATE enhanced_reminders SET next_reminder = ? WHERE id = ?", (next_reminder, reminder_id))
        conn.commit()
        conn.close()

    def update_reminder_type(self, reminder_id: int, reminder_type: str) -> None:
        """Update the reminder_type for a reminder."""
        if reminder_type is None:
            logger.error("reminder_type must not be None")
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE enhanced_reminders SET reminder_type = ? WHERE id = ?", (reminder_type, reminder_id))
        conn.commit()
        conn.close() 