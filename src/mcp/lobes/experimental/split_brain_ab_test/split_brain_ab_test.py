from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
from typing import Optional, Any, Dict
import logging
import random

class SplitBrainABTest:
    """
    Split Brain AB Test Engine
    Parallel agent teams for AB testing and feedback-driven selection. See idea.txt for requirements.
    Each instance has its own working memory for short-term, context-sensitive storage (see idea.txt and research).
    Implements comparison and selection mechanisms for split-brain architectures and AB testing.
    """
    def __init__(self, lobe_class=None, left_config=None, right_config=None, db_path: Optional[str] = None, **kwargs):
        self.lobe_class = lobe_class
        self.left_config = left_config
        self.right_config = right_config
        self.db_path = db_path
        self.working_memory = WorkingMemory()

    def run_ab_test(self, lobe_a, lobe_b, input_data: Any = None, feedback: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Run an AB test between two lobes (A and B) on the same input data.
        - Each lobe processes the input independently.
        - Feedback (if provided) is used to select the better result.
        - Returns both results and the selected winner.
        """
        if not lobe_a or not lobe_b:
            logging.warning("[SplitBrainABTest] Both lobes must be provided for AB testing.")
            return {"status": "error", "message": "Both lobes required"}
        result_a = lobe_a.process(input_data) if hasattr(lobe_a, 'process') else lobe_a(input_data)
        result_b = lobe_b.process(input_data) if hasattr(lobe_b, 'process') else lobe_b(input_data)
        # Feedback-driven selection (score-based or random for demo)
        winner = None
        if feedback:
            score_a = feedback.get('a', 0)
            score_b = feedback.get('b', 0)
            if score_a > score_b:
                winner = 'A'
            elif score_b > score_a:
                winner = 'B'
            else:
                winner = random.choice(['A', 'B'])
        else:
            winner = random.choice(['A', 'B'])
        return {
            "status": "completed",
            "result_a": result_a,
            "result_b": result_b,
            "winner": winner
        }

    def run_test(self, *args, **kwargs):
        """Alias for run_ab_test for test compatibility."""
        return self.run_ab_test(*args, **kwargs) 