from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory  # See idea.txt
from typing import Optional, Any, Dict
import logging
import random

class SplitBrainABTest:
    """
    Split Brain AB Test Engine
    Parallel agent teams for AB testing and feedback-driven selection.
    Each instance has its own working memory for short-term, context-sensitive storage (see idea.txt and research).
    Implements comparison and selection mechanisms for split-brain architectures and AB testing.

    Research References:
    - idea.txt (split-brain architectures, AB testing, feedback-driven selection)
    - Nature 2024 (Split-Brain Architectures for AI)
    - NeurIPS 2025 (Parallel Agent Teams and AB Testing)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add support for multi-arm bandit and advanced selection algorithms
    - Integrate with distributed agent orchestration
    - Support for dynamic feedback and reward models
    TODO:
    - Implement advanced feedback-driven selection and reranking
    - Add robust error handling and logging for all test cases
    - Support for dynamic agent registration and removal
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
        
        Extensibility:
        - Add support for multi-arm bandit and advanced selection algorithms
        - Integrate with distributed agent orchestration
        - Support for dynamic feedback and reward models
        TODO:
        - Implement advanced feedback-driven selection and reranking
        - Add robust error handling and logging for all test cases
        - Support for dynamic agent registration and removal
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

    def advanced_feedback_selection(self, lobe_a, lobe_b, input_data: Any = None, feedback: Optional[Dict[str, float]] = None, selection_fn: Optional[Any] = None) -> Dict[str, Any]:
        """
        Advanced feedback-driven selection and reranking for AB testing.
        Allows custom selection functions and dynamic feedback models.
        Returns both results and the selected winner.
        """
        try:
            result_a = lobe_a.process(input_data) if hasattr(lobe_a, 'process') else lobe_a(input_data)
            result_b = lobe_b.process(input_data) if hasattr(lobe_b, 'process') else lobe_b(input_data)
            if selection_fn and callable(selection_fn):
                winner = selection_fn(result_a, result_b, feedback)
            elif feedback:
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
        except Exception as ex:
            logging.error(f"[SplitBrainABTest] Error in advanced_feedback_selection: {ex}")
            return {"status": "error", "message": str(ex)}

    def register_agent(self, agent, side: str = 'A'):
        """
        Dynamically register an agent (lobe) to the left or right side.
        """
        if side == 'A':
            self.left_config = agent
        elif side == 'B':
            self.right_config = agent
        else:
            logging.warning(f"[SplitBrainABTest] Unknown side: {side}")
        logging.info(f"[SplitBrainABTest] Registered agent to side {side}.")

    def remove_agent(self, side: str = 'A'):
        """
        Dynamically remove an agent (lobe) from the left or right side.
        """
        if side == 'A':
            self.left_config = None
        elif side == 'B':
            self.right_config = None
        else:
            logging.warning(f"[SplitBrainABTest] Unknown side: {side}")
        logging.info(f"[SplitBrainABTest] Removed agent from side {side}.")

    def cross_lobe_integration(self, lobe_a, lobe_b, input_data: Any = None, lobe_name: str = "") -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call DecisionMakingLobe or PatternRecognitionEngine for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        logging.info(f"[SplitBrainABTest] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        return self.advanced_feedback_selection(lobe_a, lobe_b, input_data=input_data)

    def usage_example(self):
        """
        Usage example for split brain AB test:
        >>> ab = SplitBrainABTest()
        >>> ab.register_agent(lambda x: x + 1, side='A')
        >>> ab.register_agent(lambda x: x * 2, side='B')
        >>> result = ab.advanced_feedback_selection(ab.left_config, ab.right_config, input_data=3, feedback={'a': 2, 'b': 1})
        >>> print(result)
        >>> # Cross-lobe integration
        >>> ab.cross_lobe_integration(ab.left_config, ab.right_config, input_data=3, lobe_name='DecisionMakingLobe')
        """
        pass 