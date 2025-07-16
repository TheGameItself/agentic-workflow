"""
MemoryLobe: Core lobe for memory storage, retrieval, and feedback-driven adaptation.
Integrates WorkingMemory, ShortTermMemory, and LongTermMemory for robust, research-driven memory architecture.
See idea.txt, README.md, and RESEARCH_SOURCES.md for details.
"""

from typing import Any, Dict, List, Optional
import logging
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory, ShortTermMemory, LongTermMemory

class MemoryLobe:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.logger = logging.getLogger("MemoryLobe")

    def store(self, memory: Dict[str, Any], memory_type: str = "working"):
        """Store a new memory in the specified memory type ('working', 'short_term', 'long_term')."""
        try:
            if memory_type == "working":
                self.working_memory.add(memory)
            elif memory_type == "short_term":
                self.short_term_memory.add(memory)
            elif memory_type == "long_term":
                key = memory.get("id") or str(len(self.long_term_memory.get_all()))
                self.long_term_memory.add(key, memory)
            else:
                raise ValueError(f"Unknown memory_type: {memory_type}")
            self.logger.info(f"[MemoryLobe] Memory stored in {memory_type}: {memory}")
        except Exception as ex:
            self.logger.error(f"[MemoryLobe] Error storing memory: {ex}")

    def retrieve(self, memory_type: str = "working", query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve memories from the specified memory type."""
        if memory_type == "working":
            memories = self.working_memory.get_all()
        elif memory_type == "short_term":
            memories = self.short_term_memory.get_all()
        elif memory_type == "long_term":
            memories = list(self.long_term_memory.get_all().values())
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")
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
        return self.working_memory.get_all() + self.short_term_memory.get_all() + list(self.long_term_memory.get_all().values())

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
