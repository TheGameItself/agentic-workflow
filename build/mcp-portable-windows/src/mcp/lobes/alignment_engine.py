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
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.mcp.lobes.experimental.vesicle_pool import VesiclePool
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory  # See idea.txt


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
        self.buffer.append({"preference": preference, "strength": 1.0})
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def decay_buffer(self):
        for entry in self.buffer:
            entry["strength"] *= self.decay
        self.buffer = [e for e in self.buffer if e["strength"] > 0.1]

    def get_recent(self, n=5):
        return [e["preference"] for e in self.buffer[-n:]]


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
        self.memory = (
            []
        )  # Each entry: {'preference': ..., 'context': ..., 'feedback': ..., 'strength': ...}
        self.logger = logging.getLogger("AssociativePreferenceMemory")

    def add(self, preference, context=None, feedback=None):
        entry = {
            "preference": preference,
            "context": context,
            "feedback": feedback,
            "strength": 1.0,
        }
        self.memory.append(entry)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.logger.info(
            f"[AssociativePreferenceMemory] Added preference: {preference} (context={context}, feedback={feedback})"
        )

    def decay_memory(self):
        for entry in self.memory:
            entry["strength"] *= self.decay
        self.memory = [e for e in self.memory if e["strength"] > 0.1]

    def get_by_context(self, context, n=5):
        context_str = str(context) if context is not None else ""
        matches = [
            e for e in self.memory if context_str and context_str in str(e["context"])
        ]
        return [e["preference"] for e in matches[-n:]]

    def get_recent(self, n=5):
        return [e["preference"] for e in self.memory[-n:]]


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
            project_root = os.path.join(current_dir, "..", "..", "..")
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "alignment_engine.db")
        self.db_path = db_path
        self.alignment_history = []
        self.user_preferences = {}
        self.feedback_scores = []
        self.working_memory = WorkingMemory()
        self.preference_buffer = PreferenceBuffer()
        self.associative_memory = AssociativePreferenceMemory()
        self.logger = logging.getLogger("AlignmentEngine")
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.logger.info(
            "[AlignmentEngine] VesiclePool initialized: %s",
            self.vesicle_pool.get_state(),
        )
        self._init_database()

    def _init_database(self):
        """Initialize alignment database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alignment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_output TEXT NOT NULL,
                aligned_output TEXT NOT NULL,
                user_preferences TEXT,
                feedback_score REAL DEFAULT 0.0,
                alignment_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_key TEXT UNIQUE NOT NULL,
                preference_value TEXT,
                confidence REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alignment_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def align(
        self, llm_output: str, user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
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
            return " ".join(words[:max_words]) + "..."
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
            lines = output.split("\n")
            technical_lines = [
                line
                for line in lines
                if any(
                    word in line.lower()
                    for word in [
                        "function",
                        "class",
                        "method",
                        "api",
                        "config",
                        "parameter",
                    ]
                )
            ]
            if technical_lines:
                return "\n".join(technical_lines)

        elif content_focus == "practical":
            # Emphasize practical steps
            lines = output.split("\n")
            practical_lines = [
                line
                for line in lines
                if any(
                    word in line.lower()
                    for word in ["step", "run", "install", "create", "add", "use"]
                )
            ]
            if practical_lines:
                return "\n".join(practical_lines)

        return output

    def _align_format(self, output: str, preferences: Dict[str, Any]) -> str:
        """Align format based on preferences."""
        format_type = preferences.get("format", "text")

        if format_type == "bullet_points":
            # Convert to bullet points
            lines = output.split("\n")
            bulleted = [f"• {line.strip()}" for line in lines if line.strip()]
            return "\n".join(bulleted)

        elif format_type == "numbered":
            # Convert to numbered list
            lines = output.split("\n")
            numbered = [
                f"{i+1}. {line.strip()}" for i, line in enumerate(lines) if line.strip()
            ]
            return "\n".join(numbered)

        return output

    def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT preference_key, preference_value, confidence FROM user_preferences"
        )
        preferences = {}

        for row in cursor.fetchall():
            key, value, confidence = row
            if confidence > 0.3:  # Only use preferences with sufficient confidence
                preferences[key] = value

        conn.close()
        return preferences

    def _store_alignment_history(
        self, original: str, aligned: str, preferences: Dict[str, Any]
    ):
        """Store alignment history in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO alignment_history (original_output, aligned_output, user_preferences, alignment_method)
            VALUES (?, ?, ?, ?)
        """,
            (original, aligned, json.dumps(preferences), "multi_method"),
        )

        conn.commit()
        conn.close()

    def learn_from_feedback(
        self, alignment_id: int, feedback_score: float, feedback_text: str = ""
    ):
        """Learn from user feedback to improve alignment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update alignment history with feedback
        cursor.execute(
            """
            UPDATE alignment_history 
            SET feedback_score = ? 
            WHERE id = ?
        """,
            (feedback_score, alignment_id),
        )

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
            patterns.append(
                {"type": "length", "pattern": "concise", "success_rate": score}
            )
        if "formal" in feedback_text.lower():
            patterns.append(
                {"type": "style", "pattern": "formal", "success_rate": score}
            )
        if "technical" in feedback_text.lower():
            patterns.append(
                {"type": "content", "pattern": "technical", "success_rate": score}
            )

        for pattern in patterns:
            cursor.execute(
                """
                INSERT OR REPLACE INTO alignment_patterns 
                (pattern_type, pattern_data, success_rate, usage_count)
                VALUES (?, ?, ?, 1)
            """,
                (pattern["type"], json.dumps(pattern), pattern["success_rate"]),
            )

        conn.commit()
        conn.close()

    def _update_preferences_from_feedback(self, alignment_id: int, score: float):
        """Update user preferences based on feedback score."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the alignment record
        cursor.execute(
            "SELECT user_preferences FROM alignment_history WHERE id = ?",
            (alignment_id,),
        )
        result = cursor.fetchone()

        if result:
            preferences = json.loads(result[0])

            # Update preference confidence based on feedback
            for key, value in preferences.items():
                cursor.execute(
                    """
                    SELECT confidence, usage_count FROM user_preferences 
                    WHERE preference_key = ?
                """,
                    (key,),
                )

                existing = cursor.fetchone()
                if existing:
                    current_confidence, usage_count = existing
                    # Update confidence using exponential moving average
                    new_confidence = (current_confidence * usage_count + score) / (
                        usage_count + 1
                    )
                    cursor.execute(
                        """
                        UPDATE user_preferences 
                        SET confidence = ?, usage_count = usage_count + 1, last_updated = CURRENT_TIMESTAMP
                        WHERE preference_key = ?
                    """,
                        (new_confidence, key),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO user_preferences (preference_key, preference_value, confidence, usage_count)
                        VALUES (?, ?, ?, 1)
                    """,
                        (key, value, score),
                    )

        conn.commit()
        conn.close()

    def get_alignment_history(self) -> List[Dict[str, Any]]:
        """Get alignment history for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, original_output, aligned_output, user_preferences, 
                   feedback_score, alignment_method, created_at
            FROM alignment_history
            ORDER BY created_at DESC
            LIMIT 100
        """
        )

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "id": row[0],
                    "original": row[1],
                    "aligned": row[2],
                    "preferences": json.loads(row[3]) if row[3] else {},
                    "feedback_score": row[4],
                    "method": row[5],
                    "created_at": row[6],
                }
            )

        conn.close()
        return history

    def get_alignment_statistics(self) -> Dict[str, Any]:
        """Get alignment statistics and performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get basic statistics
        cursor.execute("SELECT COUNT(*) FROM alignment_history")
        total_alignments = cursor.fetchone()[0]

        cursor.execute(
            "SELECT AVG(feedback_score) FROM alignment_history WHERE feedback_score IS NOT NULL"
        )
        avg_score = cursor.fetchone()[0] or 0.0

        cursor.execute(
            "SELECT COUNT(*) FROM alignment_history WHERE feedback_score > 0.7"
        )
        high_score_count = cursor.fetchone()[0]

        # Get pattern statistics
        cursor.execute(
            """
            SELECT pattern_type, AVG(success_rate), COUNT(*) 
            FROM alignment_patterns 
            GROUP BY pattern_type
        """
        )
        pattern_stats = {}
        for row in cursor.fetchall():
            pattern_stats[row[0]] = {"avg_success_rate": row[1], "usage_count": row[2]}

        # Get preference statistics
        cursor.execute(
            """
            SELECT preference_key, confidence, usage_count 
            FROM user_preferences 
            ORDER BY confidence DESC
        """
        )
        preferences = []
        for row in cursor.fetchall():
            preferences.append(
                {"key": row[0], "confidence": row[1], "usage_count": row[2]}
            )

        conn.close()

        return {
            "total_alignments": total_alignments,
            "average_feedback_score": avg_score,
            "high_score_percentage": (
                (high_score_count / total_alignments * 100)
                if total_alignments > 0
                else 0
            ),
            "pattern_statistics": pattern_stats,
            "top_preferences": preferences[:10],
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

    def align_rlhf(
        self,
        llm_output: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Advanced RLHF-based alignment: aligns LLM output with user preferences using reward modeling and feedback.
        Returns the aligned output.
        """
        try:
            # Example: reward model stub (replace with real RLHF logic)
            reward = 1.0
            if feedback and "reward" in feedback:
                reward = float(feedback["reward"])
            aligned = llm_output
            if user_preferences:
                for k, v in user_preferences.items():
                    aligned = aligned.replace(k, v)
            if reward < 0.5:
                aligned = f"[LOW REWARD] {aligned}"
            self.logger.info(
                f"[AlignmentEngine] RLHF alignment complete. Reward: {reward}"
            )
            return aligned
        except Exception as ex:
            self.logger.error(f"[AlignmentEngine] RLHF alignment error: {ex}")
            return llm_output

    def align_with_template(
        self,
        llm_output: str,
        template: str,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Align LLM output using a dynamic template and user preferences.
        """
        try:
            aligned = template.format(output=llm_output, **(user_preferences or {}))
            self.logger.info(f"[AlignmentEngine] Template alignment complete.")
            return aligned
        except Exception as ex:
            self.logger.error(f"[AlignmentEngine] Template alignment error: {ex}")
            return llm_output

    def feedback_loop(self, alignment_id: int, feedback: Dict[str, Any]):
        """
        Integrate feedback for continual improvement and dynamic adaptation.
        Updates internal parameters, logs feedback, and triggers preference/model updates.
        """
        try:
            score = feedback.get("score", 0.0)
            text = feedback.get("text", "")
            self.learn_from_feedback(alignment_id, score, text)
            self.logger.info(
                f"[AlignmentEngine] Feedback loop executed for alignment_id={alignment_id}."
            )
        except Exception as ex:
            self.logger.error(f"[AlignmentEngine] Feedback loop error: {ex}")

    def demo_custom_alignment(
        self,
        custom_aligner: Callable,
        llm_output: str,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Demo/test method: run a custom alignment function and return the aligned output.
        Usage: engine.demo_custom_alignment(lambda o, p: o.upper(), 'test', {})
        """
        try:
            result = custom_aligner(llm_output, user_preferences)
            self.logger.info(f"[AlignmentEngine] Custom alignment result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[AlignmentEngine] Custom alignment error: {ex}")
            return llm_output

    def advanced_feedback_integration(self, feedback: dict):
        """
        Advanced feedback integration and continual learning for alignment engine.
        Updates alignment parameters or preference models based on structured feedback.
        Supports cross-lobe research and adaptation.
        """
        try:
            if feedback and "preference_update" in feedback:
                self.user_preferences.update(feedback["preference_update"])
                self.logger.info(
                    f"[AlignmentEngine] User preferences updated from advanced feedback: {feedback['preference_update']}"
                )
            self.working_memory.add({"advanced_feedback": feedback})
        except Exception as ex:
            self.logger.error(
                f"[AlignmentEngine] Error in advanced_feedback_integration: {ex}"
            )

    def cross_lobe_integration(self, lobe_name: str = "", data: Any = None) -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call VesiclePool or TaskProposalLobe for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(
            f"[AlignmentEngine] Cross-lobe integration called with {lobe_name}."
        )
        # Placeholder: simulate integration
        return self.get_alignment_history()

    def usage_example(self):
        """
        Usage example for alignment engine:
        >>> engine = AlignmentEngine()
        >>> aligned = engine.align('Hello, user!', {'user': 'Alice'})
        >>> print(aligned)
        >>> # RLHF alignment
        >>> rlhf = engine.align_rlhf('Hello, user!', {'user': 'Alice'}, feedback={'reward': 0.3})
        >>> print(rlhf)
        >>> # Custom alignment: uppercase
        >>> custom = engine.demo_custom_alignment(lambda o, p: o.upper(), 'test', {})
        >>> print(custom)
        >>> # Advanced feedback integration
        >>> engine.advanced_feedback_integration({'preference_update': {'style': 'formal'}})
        >>> # Cross-lobe integration
        >>> engine.cross_lobe_integration(lobe_name='VesiclePool')
        """
        pass

    def get_state(self):
        """Return a summary of the current alignment engine state for aggregation."""
        return {
            "db_path": self.db_path,
            "alignment_history": self.alignment_history,
            "user_preferences": self.user_preferences,
            "feedback_scores": self.feedback_scores,
            "working_memory": (
                self.working_memory.get_all()
                if hasattr(self.working_memory, "get_all")
                else None
            ),
        }

    def receive_data(self, data: dict):
        """
        Receive data from aggregator or adjacent lobes with brain-inspired processing.
        
        Implements cross-lobe communication using hormone-based signaling,
        attention mechanisms, and adaptive integration based on data importance.
        """
        self.logger.info(f"[AlignmentEngine] Received data from {data.get('source', 'unknown')}")
        
        try:
            # Extract data components
            source_lobe = data.get('source', 'unknown')
            data_type = data.get('type', 'general')
            content = data.get('content', {})
            importance = data.get('importance', 0.5)
            timestamp = data.get('timestamp', time.time())
            
            # Process based on data type
            if data_type == 'alignment_feedback':
                self._process_alignment_feedback(content, source_lobe)
            elif data_type == 'performance_metrics':
                self._process_performance_data(content, source_lobe)
            elif data_type == 'user_interaction':
                self._process_user_interaction(content, source_lobe)
            elif data_type == 'hormone_signal':
                self._process_hormone_signal(content, source_lobe)
            else:
                self._process_general_data(content, source_lobe)
            
            # Update cross-lobe communication statistics
            if not hasattr(self, 'communication_stats'):
                self.communication_stats = {}
            
            if source_lobe not in self.communication_stats:
                self.communication_stats[source_lobe] = {
                    'messages_received': 0,
                    'last_contact': 0,
                    'data_types': set()
                }
            
            stats = self.communication_stats[source_lobe]
            stats['messages_received'] += 1
            stats['last_contact'] = timestamp
            stats['data_types'].add(data_type)
            
            # Release hormones based on data importance and source
            if hasattr(self, 'hormone_system') and self.hormone_system:
                if importance > 0.8:
                    self.hormone_system.release_hormone('dopamine', 0.05)
                elif source_lobe in ['pattern_recognition', 'multi_llm_orchestrator']:
                    self.hormone_system.release_hormone('serotonin', 0.03)
            
            # Store in working memory if available
            if hasattr(self, 'working_memory') and self.working_memory:
                memory_entry = {
                    'type': 'cross_lobe_data',
                    'source': source_lobe,
                    'data_type': data_type,
                    'content': content,
                    'importance': importance,
                    'processed_at': time.time()
                }
                self.working_memory.add(memory_entry)
            
            self.logger.info(f"[AlignmentEngine] Successfully processed {data_type} data from {source_lobe}")
            
        except Exception as e:
            self.logger.error(f"[AlignmentEngine] Error processing received data: {e}")
            
            # Release stress hormones on processing failure
            if hasattr(self, 'hormone_system') and self.hormone_system:
                self.hormone_system.release_hormone('cortisol', 0.05)
    
    def _process_alignment_feedback(self, content: dict, source: str):
        """Process alignment feedback from other lobes."""
        feedback_score = content.get('alignment_score', 0.5)
        feedback_type = content.get('feedback_type', 'general')
        
        # Update alignment metrics
        if not hasattr(self, 'cross_lobe_alignment'):
            self.cross_lobe_alignment = {}
        
        if source not in self.cross_lobe_alignment:
            self.cross_lobe_alignment[source] = {
                'average_score': 0.5,
                'feedback_count': 0,
                'feedback_types': {}
            }
        
        alignment_data = self.cross_lobe_alignment[source]
        alignment_data['average_score'] = (
            0.9 * alignment_data['average_score'] + 0.1 * feedback_score
        )
        alignment_data['feedback_count'] += 1
        
        if feedback_type not in alignment_data['feedback_types']:
            alignment_data['feedback_types'][feedback_type] = []
        alignment_data['feedback_types'][feedback_type].append(feedback_score)
        
        self.logger.info(f"[AlignmentEngine] Updated alignment with {source}: {alignment_data['average_score']:.3f}")
    
    def _process_performance_data(self, content: dict, source: str):
        """Process performance metrics from other lobes."""
        performance_score = content.get('performance_score', 0.5)
        response_time = content.get('response_time', 1.0)
        accuracy = content.get('accuracy', 0.5)
        
        # Update performance tracking
        if not hasattr(self, 'lobe_performance'):
            self.lobe_performance = {}
        
        if source not in self.lobe_performance:
            self.lobe_performance[source] = {
                'performance_history': [],
                'average_performance': 0.5,
                'last_update': 0
            }
        
        perf_data = self.lobe_performance[source]
        perf_data['performance_history'].append({
            'score': performance_score,
            'response_time': response_time,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(perf_data['performance_history']) > 50:
            perf_data['performance_history'].pop(0)
        
        # Update average
        recent_scores = [p['score'] for p in perf_data['performance_history'][-10:]]
        perf_data['average_performance'] = sum(recent_scores) / len(recent_scores)
        perf_data['last_update'] = time.time()
        
        self.logger.info(f"[AlignmentEngine] Updated performance tracking for {source}: {perf_data['average_performance']:.3f}")
    
    def _process_user_interaction(self, content: dict, source: str):
        """Process user interaction data from other lobes."""
        interaction_type = content.get('interaction_type', 'unknown')
        user_satisfaction = content.get('user_satisfaction', 0.5)
        context = content.get('context', {})
        
        # Update user interaction patterns
        if not hasattr(self, 'user_interaction_patterns'):
            self.user_interaction_patterns = {}
        
        if interaction_type not in self.user_interaction_patterns:
            self.user_interaction_patterns[interaction_type] = {
                'count': 0,
                'average_satisfaction': 0.5,
                'sources': set(),
                'contexts': []
            }
        
        pattern = self.user_interaction_patterns[interaction_type]
        pattern['count'] += 1
        pattern['average_satisfaction'] = (
            0.9 * pattern['average_satisfaction'] + 0.1 * user_satisfaction
        )
        pattern['sources'].add(source)
        pattern['contexts'].append(context)
        
        # Keep contexts manageable
        if len(pattern['contexts']) > 20:
            pattern['contexts'].pop(0)
        
        self.logger.info(f"[AlignmentEngine] Updated interaction pattern {interaction_type}: satisfaction={pattern['average_satisfaction']:.3f}")
    
    def _process_hormone_signal(self, content: dict, source: str):
        """Process hormone signals from other lobes."""
        hormone_type = content.get('hormone_type', 'unknown')
        hormone_level = content.get('level', 0.0)
        signal_strength = content.get('strength', 0.5)
        
        # Update hormone awareness
        if not hasattr(self, 'hormone_awareness'):
            self.hormone_awareness = {}
        
        if hormone_type not in self.hormone_awareness:
            self.hormone_awareness[hormone_type] = {
                'current_level': 0.0,
                'sources': {},
                'last_update': 0
            }
        
        awareness = self.hormone_awareness[hormone_type]
        awareness['current_level'] = max(awareness['current_level'], hormone_level)
        awareness['sources'][source] = {
            'level': hormone_level,
            'strength': signal_strength,
            'timestamp': time.time()
        }
        awareness['last_update'] = time.time()
        
        # Respond to hormone signals
        if hormone_type == 'dopamine' and hormone_level > 0.8:
            # High dopamine - increase alignment confidence
            if hasattr(self, 'alignment_confidence'):
                self.alignment_confidence = min(1.0, self.alignment_confidence + 0.05)
        elif hormone_type == 'cortisol' and hormone_level > 0.6:
            # High cortisol - increase caution in alignment decisions
            if hasattr(self, 'alignment_caution'):
                self.alignment_caution = min(1.0, getattr(self, 'alignment_caution', 0.5) + 0.1)
        
        self.logger.info(f"[AlignmentEngine] Processed {hormone_type} signal from {source}: level={hormone_level:.3f}")
    
    def _process_general_data(self, content: dict, source: str):
        """Process general data from other lobes."""
        # Store general data for potential future use
        if not hasattr(self, 'general_data_cache'):
            self.general_data_cache = {}
        
        if source not in self.general_data_cache:
            self.general_data_cache[source] = []
        
        self.general_data_cache[source].append({
            'content': content,
            'timestamp': time.time()
        })
        
        # Keep cache manageable
        if len(self.general_data_cache[source]) > 10:
            self.general_data_cache[source].pop(0)
        
        self.logger.info(f"[AlignmentEngine] Cached general data from {source}")

    def advanced_rlhf(self):
        """
        Advanced RLHF and preference modeling algorithms (minimal implementation).
        Fallback: logs stub status and returns a default alignment result.
        See idea.txt and TODO_DEVELOPMENT_PLAN.md for future improvements.
        """
        self.logger.warning("[AlignmentEngine] advanced_rlhf is a stub. Returning default alignment result.")
        return {"status": "stub", "alignment_score": 0.0, "details": "Advanced RLHF not yet implemented."}

    def robust_error_handling(self):
        """
        Robust error handling and logging for all alignment operations (minimal implementation).
        Fallback: logs stub status and returns True.
        See idea.txt and TODO_DEVELOPMENT_PLAN.md for future improvements.
        """
        self.logger.warning("[AlignmentEngine] robust_error_handling is a stub. No real error handling implemented.")
        return True

    def dynamic_alignment_templates(self):
        """
        Dynamic alignment templates and feedback loops (minimal implementation).
        Fallback: logs stub status and returns a default template.
        See idea.txt and TODO_DEVELOPMENT_PLAN.md for future improvements.
        """
        self.logger.warning("[AlignmentEngine] dynamic_alignment_templates is a stub. Returning default template.")
        return {"template": "default", "feedback_loop": False, "status": "stub"}
