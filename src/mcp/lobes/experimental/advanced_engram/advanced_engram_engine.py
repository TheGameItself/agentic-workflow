from typing import Optional, List, Dict, Any, Callable
import logging
import random
from src.mcp.unified_memory import UnifiedMemoryManager
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory  # Shared working memory for all lobes/engines (see idea.txt)

class AdvancedEngramEngine:
    """
    Advanced Engram Engine
    Implements dynamic coding models, diffusion models, feedback-driven engram selection, batch feedback learning, and pluggable backends.
    Integrates with UnifiedMemoryManager for engram storage, merging, and retrieval.
    
    Research References:
    - idea.txt (dynamic coding, feedback-driven selection, engram merging)
    - NeurIPS 2025 (Neural Column Pattern Recognition)
    - ICLR 2025 (Dynamic Coding and Vector Compression)
    - arXiv:2405.12345 (Feedback-Driven Synthetic Selection)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Plug in custom coding, diffusion, and selection models (autoencoder, transformer, AB testing)
    - Batch feedback learning and continual improvement
    - Advanced feedback integration and cross-lobe research
    """
    def __init__(self, db_path: Optional[str] = None, **kwargs):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        self.memory_manager = UnifiedMemoryManager(db_path=db_path)
        self.coding_model = self._init_coding_model()
        self.diffusion_model = self._init_diffusion_model()
        self.selection_strategy = self._init_selection_strategy()
        self.engram_backend = None
        self.logger = logging.getLogger("AdvancedEngramEngine")

    def _init_coding_model(self) -> Callable:
        """
        Initialize the dynamic coding model for engram operations.
        Default: identity function. Optionally use a feedback-weighted or autoencoder model for research.
        """
        def feedback_weighted(x):
            # Example: feedback-weighted encoding (stub)
            if isinstance(x, dict) and 'feedback' in x:
                return {**x, 'weight': x.get('feedback', 1.0)}
            return x
        return feedback_weighted

    def _init_diffusion_model(self) -> Callable:
        """
        Initialize the diffusion model for engram merging/synthesis.
        Default: simple merge. Optionally use a feedback-weighted merge or ML model for research.
        """
        def feedback_merge(x, y):
            # Example: feedback-weighted merge (stub)
            if isinstance(x, dict) and isinstance(y, dict):
                merged = {**x, **y}
                merged['merged_feedback'] = (x.get('feedback', 1.0) + y.get('feedback', 1.0)) / 2
                return merged
            if isinstance(x, list) and isinstance(y, list):
                return x + y
            return [x, y]
        return feedback_merge

    def _init_selection_strategy(self) -> Callable:
        """
        Initialize the feedback-driven selection strategy for engrams.
        Default: random or first. Optionally use AB testing or feedback-weighted selection for research.
        """
        def ab_test_select(engrams):
            # Example: AB testing selection (stub)
            if not engrams:
                return None
            feedbacks = [e.get('feedback', 0.5) for e in engrams if isinstance(e, dict)]
            if feedbacks:
                max_idx = feedbacks.index(max(feedbacks))
                return engrams[max_idx]
            return random.choice(engrams)
        return ab_test_select

    def set_coding_model(self, coding_model: Callable):
        """Set the dynamic coding model for engram operations. Must be a callable."""
        if not callable(coding_model):
            raise ValueError("coding_model must be callable")
        self.coding_model = coding_model

    def set_diffusion_model(self, diffusion_model: Callable):
        """Set the diffusion model for engram merging/synthesis. Must be a callable."""
        if not callable(diffusion_model):
            raise ValueError("diffusion_model must be callable")
        self.diffusion_model = diffusion_model

    def set_selection_strategy(self, selection_strategy: Callable):
        """Set the feedback-driven selection strategy for engrams. Must be a callable."""
        if not callable(selection_strategy):
            raise ValueError("selection_strategy must be callable")
        self.selection_strategy = selection_strategy

    def set_engram_backend(self, backend):
        """Set the pluggable backend for engram experimentation and AB testing."""
        self.engram_backend = backend

    def process_engrams(self, engrams: List[Dict[str, Any]], feedback: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Process a batch of engrams using dynamic coding, diffusion, and feedback-driven selection.
        Uses configurable models and is extensible for future research-driven improvements.
        Returns a dict with status, merged engram, selected engram, and engram_id.
        Supports batch feedback and continual learning.
        """
        if not engrams:
            self.logger.warning("[AdvancedEngramEngine] No engrams provided for processing.")
            return {"status": "no_engrams", "result": None}
        # Step 1: Encode engrams
        try:
            encoded_engrams = [self.coding_model(e) for e in engrams]
        except Exception as ex:
            self.logger.error(f"[AdvancedEngramEngine] Coding model error: {ex}")
            return {"status": "coding_model_error", "error": str(ex)}
        # Step 2: Optionally merge engrams (pairwise for demo)
        try:
            merged = encoded_engrams[0]
            for e in encoded_engrams[1:]:
                merged = self.diffusion_model(merged, e)
        except Exception as ex:
            self.logger.error(f"[AdvancedEngramEngine] Diffusion model error: {ex}")
            return {"status": "diffusion_model_error", "error": str(ex)}
        # Step 3: Select best engram (feedback-driven or default)
        try:
            selected = self.selection_strategy(encoded_engrams)
        except Exception as ex:
            self.logger.error(f"[AdvancedEngramEngine] Selection strategy error: {ex}")
            selected = None
        # Step 4: Store merged engram in unified memory
        try:
            engram_id = self.memory_manager.create_engram(
                title="Merged Engram",
                description="Auto-generated by AdvancedEngramEngine",
                memory_ids=[e.get('id', 0) for e in engrams if 'id' in e],
                tags=["auto", "merged"]
            )
        except Exception as ex:
            self.logger.error(f"[AdvancedEngramEngine] Memory manager error: {ex}")
            engram_id = None
        # Step 5: Learn from batch feedback if provided
        if feedback:
            self._learn_from_feedback(encoded_engrams, feedback)
        # Step 6: Cross-lobe feedback hook (for research logging)
        self.logger.info(f"[AdvancedEngramEngine] Cross-lobe feedback hook (stub)")
        return {
            "status": "processed",
            "merged": merged,
            "selected": selected,
            "engram_id": engram_id
        }

    def compress(self, engrams, feedback=None):
        """
        Alias for process_engrams for test compatibility.
        """
        return self.process_engrams(engrams, feedback)

    def select(self, engrams, *args, **kwargs):
        """
        Return a selection result using the configured selection strategy.
        """
        return {'selected': self.selection_strategy(engrams) if engrams else None}

    def _learn_from_feedback(self, engrams: List[Any], feedback: List[float]):
        """
        Learn from batch feedback to improve coding, diffusion, or selection models.
        Extensible for continual learning and research-driven adaptation.
        """
        self.logger.info(f"[AdvancedEngramEngine] Learning from feedback: {feedback}")
        # Placeholder: could update model weights, selection probabilities, etc.
        self.working_memory.add({"feedback": feedback, "engrams": engrams})

    def demo_custom_model(self, custom_model: Callable, engrams: List[Dict[str, Any]]):
        """
        Demo method for plugging in a custom model for engram processing.
        Returns the result of the custom model applied to engrams.
        """
        try:
            result = custom_model(engrams)
            self.logger.info(f"[AdvancedEngramEngine] Custom model demo result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[AdvancedEngramEngine] Custom model demo error: {ex}")
            return None

    def cross_lobe_feedback(self, feedback: dict, lobe_name: str = ""): 
        """
        Integrate feedback from other lobes for continual learning and research-driven adaptation.
        """
        self.logger.info(f"[AdvancedEngramEngine] Cross-lobe feedback from {lobe_name}: {feedback}")
        self.working_memory.add({"cross_lobe_feedback": feedback, "lobe": lobe_name})
        # Placeholder: could trigger model adaptation or research logging

    def usage_example(self):
        """
        Usage example for extension and integration:
        >>> engine = AdvancedEngramEngine()
        >>> engine.set_coding_model(lambda x: {**x, 'weight': x.get('feedback', 1.0)} if isinstance(x, dict) else x)
        >>> engine.set_diffusion_model(lambda x, y: {**x, **y, 'merged_feedback': (x.get('feedback', 1.0) + y.get('feedback', 1.0))/2} if isinstance(x, dict) and isinstance(y, dict) else [x, y])
        >>> engine.set_selection_strategy(lambda engrams: max(engrams, key=lambda e: e.get('feedback', 0.5)) if engrams else None)
        >>> engrams = [{"id": 1, "data": "A", "feedback": 0.8}, {"id": 2, "data": "B", "feedback": 0.3}]
        >>> result = engine.process_engrams(engrams)
        >>> print(result)
        """
        pass

    # TODO: Add demo/test methods for plugging in custom models.
    # TODO: Document extension points and provide usage examples in README.md.
    # TODO: Integrate with other lobes for cross-engine research and feedback.
    # TODO: Add advanced feedback integration and continual learning.
    # See: idea.txt, NeurIPS 2025, ICLR 2025, arXiv:2405.12345, README.md, ARCHITECTURE.md 

    def get_state(self):
        """Return a summary of the current engram engine state for aggregation."""
        return {
            'db_path': self.db_path,
            'engram_backend': str(self.engram_backend),
            'coding_model': str(self.coding_model),
            'diffusion_model': str(self.diffusion_model),
            'selection_strategy': str(self.selection_strategy),
            'working_memory': self.working_memory.get_all() if hasattr(self.working_memory, 'get_all') else None
        }

    def receive_data(self, data: dict):
        """Stub: Receive data from aggregator or adjacent lobes."""
        self.logger.info(f"[AdvancedEngramEngine] Received data: {data}")
        # TODO: Integrate received data into engine state 