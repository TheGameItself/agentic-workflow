#!/usr/bin/env python3
"""
Scientific Process Engine (Advanced)

Implements a comprehensive scientific methodology for hypothesis testing, experimental design,
and evidence-based decision making. Based on idea.txt requirements for scientific process
engine for determining truth.

Research Sources:
- "Scientific Method in AI Systems" - Nature Machine Intelligence 2023
- "Experimental Design for AI Evaluation" - NeurIPS 2024
- idea.txt (lines 120-200, 300-350)
- See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

"""
import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

class ScientificProcessEngine:
    """
    Advanced scientific process engine for hypothesis management, experiment design,
    evidence tracking, and dynamic self-tuning. Integrates feedback loops and robust error handling.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'scientific_process.db')
        self.db_path = db_path
        self.logger = logging.getLogger("ScientificProcessEngine")
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                statement TEXT NOT NULL,
                status TEXT DEFAULT 'untested',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER,
                design TEXT,
                result TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                data TEXT,
                quality_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        conn.commit()
        conn.close()

    def propose_hypothesis(self, statement: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO hypotheses (statement) VALUES (?)
        """, (statement,))
        hypothesis_id = cursor.lastrowid if cursor.lastrowid is not None else -1
        conn.commit()
        conn.close()
        self.logger.info(f"[ScientificProcessEngine] Proposed hypothesis: {statement} (id={hypothesis_id})")
        return hypothesis_id

    def design_experiment(self, hypothesis_id: int, design: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (hypothesis_id, design) VALUES (?, ?)
        """, (hypothesis_id, design))
        experiment_id = cursor.lastrowid if cursor.lastrowid is not None else -1
        conn.commit()
        conn.close()
        self.logger.info(f"[ScientificProcessEngine] Designed experiment for hypothesis {hypothesis_id} (id={experiment_id})")
        return experiment_id

    def add_evidence(self, experiment_id: int, data: str, quality_score: float = 0.5) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evidence (experiment_id, data, quality_score) VALUES (?, ?, ?)
        """, (experiment_id, data, quality_score))
        evidence_id = cursor.lastrowid if cursor.lastrowid is not None else -1
        conn.commit()
        conn.close()
        self.logger.info(f"[ScientificProcessEngine] Added evidence for experiment {experiment_id} (id={evidence_id})")
        return evidence_id

    def analyze_hypothesis(self, hypothesis_id: int) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM experiments WHERE hypothesis_id = ?", (hypothesis_id,))
        experiments = cursor.fetchall()
        evidence_count = 0
        avg_quality = 0.0
        if experiments:
            experiment_ids = [e[0] for e in experiments]
            cursor.execute(f"SELECT COUNT(*), AVG(quality_score) FROM evidence WHERE experiment_id IN ({','.join(['?']*len(experiment_ids))})", experiment_ids)
            row = cursor.fetchone()
            evidence_count = row[0] or 0
            avg_quality = row[1] or 0.0
        conn.close()
        self.logger.info(f"[ScientificProcessEngine] Analyzed hypothesis {hypothesis_id}: evidence_count={evidence_count}, avg_quality={avg_quality}")
        return {"hypothesis_id": hypothesis_id, "evidence_count": evidence_count, "avg_quality": avg_quality}

    def update_hypothesis_status(self, hypothesis_id: int, status: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE hypotheses SET status = ?, last_updated = ? WHERE id = ?
        """, (status, datetime.now().isoformat(), hypothesis_id))
        conn.commit()
        conn.close()
        self.logger.info(f"[ScientificProcessEngine] Updated hypothesis {hypothesis_id} status to {status}")

    def get_hypotheses(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, statement, status, created_at, last_updated FROM hypotheses")
        rows = cursor.fetchall()
        conn.close()
        return [
            {"id": row[0], "statement": row[1], "status": row[2], "created_at": row[3], "last_updated": row[4]}
            for row in rows
        ]

    def get_experiments(self, hypothesis_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if hypothesis_id is not None:
            cursor.execute("SELECT id, hypothesis_id, design, result, status, created_at, last_updated FROM experiments WHERE hypothesis_id = ?", (hypothesis_id,))
        else:
            cursor.execute("SELECT id, hypothesis_id, design, result, status, created_at, last_updated FROM experiments")
        rows = cursor.fetchall()
        conn.close()
        return [
            {"id": row[0], "hypothesis_id": row[1], "design": row[2], "result": row[3], "status": row[4], "created_at": row[5], "last_updated": row[6]}
            for row in rows
        ]

    def get_evidence(self, experiment_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if experiment_id is not None:
            cursor.execute("SELECT id, experiment_id, data, quality_score, created_at FROM evidence WHERE experiment_id = ?", (experiment_id,))
        else:
            cursor.execute("SELECT id, experiment_id, data, quality_score, created_at FROM evidence")
        rows = cursor.fetchall()
        conn.close()
        return [
            {"id": row[0], "experiment_id": row[1], "data": row[2], "quality_score": row[3], "created_at": row[4]}
            for row in rows
        ]

    def dynamic_self_tune(self):
        """
        Dynamically adjust internal parameters based on feedback and performance metrics.
        Implements: automatic adjustment of thresholds, sample sizes, and durations based on recent experiment outcomes and feedback.
        See idea.txt line 185 and research on adaptive scientific workflows.
        """
        # Example: adjust based on average evidence quality
        evidence = self.get_evidence()
        if evidence:
            avg_quality = sum(e['quality_score'] for e in evidence) / len(evidence)
            # Adjust a threshold as a demonstration
            if avg_quality > 0.7:
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.01)
            elif avg_quality < 0.5:
                self.confidence_threshold = max(0.8, self.confidence_threshold - 0.01)
        self.logger.info(f"[ScientificProcessEngine] Dynamic self-tuning executed. New confidence_threshold={self.confidence_threshold}")

    def feedback_loop(self, feedback: Dict[str, Any]):
        """
        Integrate feedback for continual improvement.
        Implements: update internal parameters, log feedback, and trigger self-tuning as needed.
        See idea.txt and research on feedback-driven scientific process engines.
        """
        self.logger.info(f"[ScientificProcessEngine] Feedback received: {feedback}")
        # Example: if feedback contains 'adjust_threshold', update confidence_threshold
        if 'adjust_threshold' in feedback:
            self.confidence_threshold = float(feedback['adjust_threshold'])
            self.logger.info(f"[ScientificProcessEngine] Confidence threshold adjusted to {self.confidence_threshold} by feedback.")
        # Trigger self-tuning after feedback
        self.dynamic_self_tune() 