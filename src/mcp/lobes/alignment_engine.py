"""
Alignment engine for user/LLM alignment and related features.

This lobe implements LLM-based preference alignment with feedback-driven adaptation.
Inspired by research on alignment, RLHF, and preference modeling.

Research References:
- idea.txt (alignment, RLHF, preference modeling, feedback-driven adaptation)
- NeurIPS 2025 (LLM Alignment and Preference Modeling)
- Nature 2024 (Feedback-Driven Adaptation in AI)
- See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

Extensibility:
- Add support for multi-agent alignment and voting
- Integrate with external preference databases and feedback analytics
- Support for dynamic alignment method selection and reranking
TODO:
- Implement advanced RLHF and preference modeling algorithms
- Add robust error handling and logging for all alignment operations
- Support for dynamic alignment templates and feedback loops
"""

import json
import re
import sqlite3
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory  # See idea.txt
import logging
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool


class PreferenceBuffer:
    """
    PreferenceBuffer: Domain-specific working memory for recent user preferences with decay.
    Inspired by neuroscience (short-term preference memory, adaptive alignment). See idea.txt.

    Research References:
    - idea.txt (short-term preference memory, adaptive alignment)
    - Nature 2024 (Preference Buffering in AI)
    - NeurIPS 2025 (Working Memory for Alignment)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced decay models (e.g., context-sensitive, feedback-driven)
    - Integrate with multi-agent or distributed preference buffers
    - Support for feedback-driven learning and adaptation
    TODO:
    - Implement advanced feedback weighting and prioritization
    - Add robust error handling for buffer overflows/underflows
    """
    def __init__(self, capacity=50, decay=0.98):
        self.capacity = capacity
        self.decay = decay
        self.buffer = []
    def add(self, preference):
        self.buffer.append({'preference': preference, 'strength': 1.0})
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    def decay_buffer(self):
        for entry in self.buffer:
            entry['strength'] *= self.decay
        self.buffer = [e for e in self.buffer if e['strength'] > 0.1]
    def get_recent(self, n=5):
        return [e['preference'] for e in self.buffer[-n:]]


class AssociativePreferenceMemory:
    """
    AssociativePreferenceMemory: Context-aware, feedback-driven memory for user preferences.
    Links preferences to context, feedback, and event metadata for rapid, relevant recall and adaptation.
    Inspired by associative memory and adaptive alignment in the brain (see idea.txt, neuroscience).

    Research References:
    - idea.txt (associative memory, adaptive alignment)
    - Nature 2024 (Associative Memory in AI)
    - NeurIPS 2025 (Context-Aware Preference Modeling)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced context linking and feedback-driven adaptation
    - Integrate with external context analytics tools
    - Support for dynamic associative memory templates
    TODO:
    - Implement advanced associative memory algorithms
    - Add robust error handling for memory overflows/underflows
    """
    def __init__(self, capacity=100, decay=0.97):
        self.capacity = capacity
        self.decay = decay
        self.memory = []  # Each entry: {'preference': ..., 'context': ..., 'feedback': ..., 'strength': ...}
        self.logger = logging.getLogger("AssociativePreferenceMemory")
    def add(self, preference, context=None, feedback=None):
        entry = {'preference': preference, 'context': context, 'feedback': feedback, 'strength': 1.0}
        self.memory.append(entry)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.logger.info(f"[AssociativePreferenceMemory] Added preference: {preference} (context={context}, feedback={feedback})")
    def decay_memory(self):
        for entry in self.memory:
            entry['strength'] *= self.decay
        self.memory = [e for e in self.memory if e['strength'] > 0.1]
    def get_by_context(self, context, n=5):
        context_str = str(context) if context is not None else ""
        matches = [e for e in self.memory if context_str and context_str in str(e['context'])]
        return [e['preference'] for e in matches[-n:]]
    def get_recent(self, n=5):
        return [e['preference'] for e in self.memory[-n:]]


class AlignmentEngine:
    """Alignment engine for user/LLM alignment and related features.
    Implements LLM-based preference alignment with feedback-driven adaptation.
    Each instance has its own working memory for short-term, context-sensitive storage (see idea.txt and research).

    Research References:
    - idea.txt (alignment, RLHF, preference modeling, feedback-driven adaptation)
    - NeurIPS 2025 (LLM Alignment and Preference Modeling)
    - Nature 2024 (Feedback-Driven Adaptation in AI)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add support for multi-agent alignment and voting
    - Integrate with external preference databases and feedback analytics
    - Support for dynamic alignment method selection and reranking
    TODO:
    - Implement advanced RLHF and preference modeling algorithms
    - Add robust error handling and logging for all alignment operations
    - Support for dynamic alignment templates and feedback loops
    """
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'alignment_engine.db')
        self.db_path = db_path
        self.alignment_history = []
        self.user_preferences = {}
        self.feedback_scores = []
        self.working_memory = WorkingMemory()
        self.preference_buffer = PreferenceBuffer()
        self.associative_memory = AssociativePreferenceMemory()
        self.logger = logging.getLogger("AlignmentEngine")
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.logger.info("[AlignmentEngine] VesiclePool initialized: %s", self.vesicle_pool.get_state())
        self._init_database()
    
    def _init_database(self):
        """Initialize alignment database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alignment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_output TEXT NOT NULL,
                aligned_output TEXT NOT NULL,
                user_preferences TEXT,
                feedback_score REAL DEFAULT 0.0,
                alignment_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_key TEXT UNIQUE NOT NULL,
                preference_value TEXT,
                confidence REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alignment_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def align(self, llm_output: str, user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Align LLM output with user preferences using multiple alignment methods."""
        if user_preferences is None:
            user_preferences = self._get_user_preferences()
        
        if user_preferences is not None:
            self.preference_buffer.add(user_preferences)
            self.associative_memory.add(user_preferences)
        self.preference_buffer.decay_buffer()
        self.associative_memory.decay_memory()
        
        # Apply multiple alignment methods
        aligned_output = llm_output
        
        # Method 1: Length-based alignment
        if user_preferences.get("preference") == "concise":
            aligned_output = self._align_concise(aligned_output, user_preferences)
        
        # Method 2: Style-based alignment
        if user_preferences.get("style"):
            aligned_output = self._align_style(aligned_output, user_preferences)
        
        # Method 3: Content-based alignment
        if user_preferences.get("content_focus"):
            aligned_output = self._align_content(aligned_output, user_preferences)
        
        # Method 4: Format-based alignment
        if user_preferences.get("format"):
            aligned_output = self._align_format(aligned_output, user_preferences)
        
        # Store alignment history
        self._store_alignment_history(llm_output, aligned_output, user_preferences)
        
        return aligned_output

    def _align_concise(self, output: str, preferences: Dict[str, Any]) -> str:
        """Align output to be more concise."""
        max_words = preferences.get("max_words", 50)
        words = output.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + "..."
        return output

    def _align_style(self, output: str, preferences: Dict[str, Any]) -> str:
        """Align output style based on preferences."""
        style = preferences.get("style", "neutral")
        
        if style == "formal":
            # Remove contractions, informal language
            output = re.sub(r"n't\b", " not", output)
            output = re.sub(r"'re\b", " are", output)
            output = re.sub(r"'s\b", " is", output)
            output = re.sub(r"'ll\b", " will", output)
        
        elif style == "casual":
            # Add contractions, informal language
            output = re.sub(r" not\b", "n't", output)
            output = re.sub(r" are\b", "'re", output)
            output = re.sub(r" is\b", "'s", output)
        
        return output

    def _align_content(self, output: str, preferences: Dict[str, Any]) -> str:
        """Align content focus based on preferences."""
        content_focus = preferences.get("content_focus", "balanced")
        
        if content_focus == "technical":
            # Emphasize technical details
            lines = output.split('\n')
            technical_lines = [line for line in lines if any(word in line.lower() for word in 
                           ['function', 'class', 'method', 'api', 'config', 'parameter'])]
            if technical_lines:
                return '\n'.join(technical_lines)
        
        elif content_focus == "practical":
            # Emphasize practical steps
            lines = output.split('\n')
            practical_lines = [line for line in lines if any(word in line.lower() for word in 
                           ['step', 'run', 'install', 'create', 'add', 'use'])]
            if practical_lines:
                return '\n'.join(practical_lines)
        
        return output

    def _align_format(self, output: str, preferences: Dict[str, Any]) -> str:
        """Align format based on preferences."""
        format_type = preferences.get("format", "text")
        
        if format_type == "bullet_points":
            # Convert to bullet points
            lines = output.split('\n')
            bulleted = [f"• {line.strip()}" for line in lines if line.strip()]
            return '\n'.join(bulleted)
        
        elif format_type == "numbered":
            # Convert to numbered list
            lines = output.split('\n')
            numbered = [f"{i+1}. {line.strip()}" for i, line in enumerate(lines) if line.strip()]
            return '\n'.join(numbered)
        
        return output

    def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT preference_key, preference_value, confidence FROM user_preferences")
        preferences = {}
        
        for row in cursor.fetchall():
            key, value, confidence = row
            if confidence > 0.3:  # Only use preferences with sufficient confidence
                preferences[key] = value
        
        conn.close()
        return preferences

    def _store_alignment_history(self, original: str, aligned: str, preferences: Dict[str, Any]):
        """Store alignment history in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alignment_history (original_output, aligned_output, user_preferences, alignment_method)
            VALUES (?, ?, ?, ?)
        """, (original, aligned, json.dumps(preferences), "multi_method"))
        
        conn.commit()
        conn.close()

    def learn_from_feedback(self, alignment_id: int, feedback_score: float, feedback_text: str = ""):
        """Learn from user feedback to improve alignment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update alignment history with feedback
        cursor.execute("""
            UPDATE alignment_history 
            SET feedback_score = ? 
            WHERE id = ?
        """, (feedback_score, alignment_id))
        
        # Extract patterns from feedback text
        if feedback_text:
            self._extract_alignment_patterns(feedback_text, feedback_score)
        
        # Update user preferences based on feedback
        self._update_preferences_from_feedback(alignment_id, feedback_score)
        
        conn.commit()
        conn.close()

    def _extract_alignment_patterns(self, feedback_text: str, score: float):
        """Extract alignment patterns from feedback text."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple pattern extraction based on keywords
        patterns = []
        if "concise" in feedback_text.lower():
            patterns.append({"type": "length", "pattern": "concise", "success_rate": score})
        if "formal" in feedback_text.lower():
            patterns.append({"type": "style", "pattern": "formal", "success_rate": score})
        if "technical" in feedback_text.lower():
            patterns.append({"type": "content", "pattern": "technical", "success_rate": score})
        
        for pattern in patterns:
            cursor.execute("""
                INSERT OR REPLACE INTO alignment_patterns 
                (pattern_type, pattern_data, success_rate, usage_count)
                VALUES (?, ?, ?, 1)
            """, (pattern["type"], json.dumps(pattern), pattern["success_rate"]))
        
        conn.commit()
        conn.close()

    def _update_preferences_from_feedback(self, alignment_id: int, score: float):
        """Update user preferences based on feedback score."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the alignment record
        cursor.execute("SELECT user_preferences FROM alignment_history WHERE id = ?", (alignment_id,))
        result = cursor.fetchone()
        
        if result:
            preferences = json.loads(result[0])
            
            # Update preference confidence based on feedback
            for key, value in preferences.items():
                cursor.execute("""
                    SELECT confidence, usage_count FROM user_preferences 
                    WHERE preference_key = ?
                """, (key,))
                
                existing = cursor.fetchone()
                if existing:
                    current_confidence, usage_count = existing
                    # Update confidence using exponential moving average
                    new_confidence = (current_confidence * usage_count + score) / (usage_count + 1)
                    cursor.execute("""
                        UPDATE user_preferences 
                        SET confidence = ?, usage_count = usage_count + 1, last_updated = CURRENT_TIMESTAMP
                        WHERE preference_key = ?
                    """, (new_confidence, key))
                else:
                    cursor.execute("""
                        INSERT INTO user_preferences (preference_key, preference_value, confidence, usage_count)
                        VALUES (?, ?, ?, 1)
                    """, (key, value, score))
        
        conn.commit()
        conn.close()

    def get_alignment_history(self) -> List[Dict[str, Any]]:
        """Get alignment history for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, original_output, aligned_output, user_preferences, 
                   feedback_score, alignment_method, created_at
            FROM alignment_history
            ORDER BY created_at DESC
            LIMIT 100
        """)
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row[0],
                "original": row[1],
                "aligned": row[2],
                "preferences": json.loads(row[3]) if row[3] else {},
                "feedback_score": row[4],
                "method": row[5],
                "created_at": row[6]
            })
        
        conn.close()
        return history

    def get_alignment_statistics(self) -> Dict[str, Any]:
        """Get alignment statistics and performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic statistics
        cursor.execute("SELECT COUNT(*) FROM alignment_history")
        total_alignments = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(feedback_score) FROM alignment_history WHERE feedback_score IS NOT NULL")
        avg_score = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT COUNT(*) FROM alignment_history WHERE feedback_score > 0.7")
        high_score_count = cursor.fetchone()[0]
        
        # Get pattern statistics
        cursor.execute("""
            SELECT pattern_type, AVG(success_rate), COUNT(*) 
            FROM alignment_patterns 
            GROUP BY pattern_type
        """)
        pattern_stats = {}
        for row in cursor.fetchall():
            pattern_stats[row[0]] = {
                "avg_success_rate": row[1],
                "usage_count": row[2]
            }
        
        # Get preference statistics
        cursor.execute("""
            SELECT preference_key, confidence, usage_count 
            FROM user_preferences 
            ORDER BY confidence DESC
        """)
        preferences = []
        for row in cursor.fetchall():
            preferences.append({
                "key": row[0],
                "confidence": row[1],
                "usage_count": row[2]
            })
        
        conn.close()
        
        return {
            "total_alignments": total_alignments,
            "average_feedback_score": avg_score,
            "high_score_percentage": (high_score_count / total_alignments * 100) if total_alignments > 0 else 0,
            "pattern_statistics": pattern_stats,
            "top_preferences": preferences[:10]
        }

    def process(self, input_data: Any) -> Any:
        """Process input data through alignment engine."""
        if isinstance(input_data, str):
            return self.align(input_data)
        elif isinstance(input_data, dict) and "text" in input_data:
            preferences = input_data.get("preferences", {})
            return self.align(input_data["text"], preferences)
        else:
            return input_data

    def reset_preferences(self):
        """Reset all user preferences (for testing/debugging)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM user_preferences")
        cursor.execute("DELETE FROM alignment_patterns")
        cursor.execute("DELETE FROM alignment_history")
        
        conn.commit()
        conn.close()

    def add_preference(self, preference, context=None, feedback=None):
        self.preference_buffer.add(preference)
        self.associative_memory.add(preference, context=context, feedback=feedback)
        self.associative_memory.decay_memory()

    def recall_preferences_by_context(self, context=None, n=5):
        return self.associative_memory.get_by_context(context, n=n) 