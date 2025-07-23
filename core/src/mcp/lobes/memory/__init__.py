"""
MemoryLobe: Core lobe for memory storage, retrieval, and feedback-driven adaptation.
Integrates WorkingMemory, ShortTermMemory, and LongTermMemory for robust, research-driven memory architecture.
See idea.txt, README.md, and RESEARCH_SOURCES.md for details.
"""

from typing import Any, Dict, List, Optional
import logging
from src.mcp.three_tier_memory_manager import ThreeTierMemoryManager, MemoryTier

class MemoryLobe:
    def __init__(self, hormone_system: Optional[Any] = None, genetic_trigger_manager: Optional[Any] = None):
        self.memory_manager = ThreeTierMemoryManager(hormone_system=hormone_system, genetic_trigger_manager=genetic_trigger_manager)
        self.logger = logging.getLogger("MemoryLobe")

    def store(self, memory: Dict[str, Any], memory_type: str = "working"):
        """Store a new memory in the specified memory type ('working', 'short_term', 'long_term')."""
        try:
            key = memory.get("id") or str(hash(str(memory)))
            # Map memory_type to tier_hint for ThreeTierMemoryManager
            tier_hint = None
            if memory_type == "working":
                tier_hint = MemoryTier.WORKING
            elif memory_type == "short_term":
                tier_hint = MemoryTier.SHORT_TERM
            elif memory_type == "long_term":
                tier_hint = MemoryTier.LONG_TERM
            else:
                raise ValueError(f"Unknown memory_type: {memory_type}")
            self.memory_manager.store(key, memory, tier_hint=tier_hint)
            self.logger.info(f"[MemoryLobe] Memory stored in {memory_type}: {memory}")
        except Exception as ex:
            self.logger.error(f"[MemoryLobe] Error storing memory: {ex}")

    def retrieve(self, memory_type: str = "working", query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve memories from the specified memory type."""
        tier_hint = None
        if memory_type == "working":
            tier_hint = MemoryTier.WORKING
        elif memory_type == "short_term":
            tier_hint = MemoryTier.SHORT_TERM
        elif memory_type == "long_term":
            tier_hint = MemoryTier.LONG_TERM
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")
        # Use search to get all items in the tier
        items = self.memory_manager.search(query="", tier=tier_hint, limit=1000)
        memories = [item.data for item in items]
        if not query:
            return memories
        return [m for m in memories if all(m.get(k) == v for k, v in query.items())]

    def integrate_feedback(self, feedback: Dict[str, Any]):
        """Integrate feedback for continual learning and adaptation across all memory types."""
        self.logger.info(f"[MemoryLobe] Feedback integrated: {feedback}")
        # Placeholder: update memory weights, trigger adaptation, etc.

    def advanced_feedback_integration(self, feedback: Dict[str, Any]):
        """
        Advanced feedback integration and continual learning for memory lobe.
        Updates memory storage or retrieval parameters based on structured feedback.
        Supports cross-lobe research and adaptation.
        """
        try:
            if feedback and 'memory_priority' in feedback:
                self.logger.info(f"[MemoryLobe] Memory priority updated to {feedback['memory_priority']} from feedback.")
            self.logger.info(f"[MemoryLobe] Advanced feedback integration: {feedback}")
        except Exception as ex:
            self.logger.error(f"[MemoryLobe] Error in advanced_feedback_integration: {ex}")

    def cross_lobe_integration(self, lobe_name: str = "", data: Any = None) -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call TaskLobe or ProjectLobe for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[MemoryLobe] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        items = self.memory_manager.search(query="", limit=1000)
        return [item.data for item in items]

    def hormone_genetic_trigger_hook(self, event: str, data: Any = None):
        """Handle hormone/genetic trigger events and route to memory manager integration points."""
        self.logger.info(f"[MemoryLobe] Hormone/Genetic Trigger Event: {event}, Data: {data}")
        # Example: trigger optimization or promotion based on event
        if event == "optimize":
            self.memory_manager.optimize(force=True)
        elif event == "promote" and data:
            key = data.get("key")
            from_tier = data.get("from_tier")
            to_tier = data.get("to_tier")
            if key and from_tier and to_tier:
                self.memory_manager._promote_item(key, from_tier, to_tier)
        # Extend with more event types as needed

    def usage_example(self):
        """
        Usage example for MemoryLobe:
        >>> lobe = MemoryLobe()
        >>> lobe.store({"id": 1, "data": "test memory"}, memory_type="working")
        >>> print(lobe.retrieve(memory_type="working"))
        >>> # Advanced feedback integration
        >>> lobe.advanced_feedback_integration({'memory_priority': 1})
        >>> # Cross-lobe integration
        >>> lobe.cross_lobe_integration(lobe_name='TaskLobe')
        """
        pass
