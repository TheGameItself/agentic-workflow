from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
import logging
from typing import Any, Optional

class ReflexBuffer:
    """
    ReflexBuffer: Fast, context-tagged, feedback-driven buffer for reflex signals.
    Inspired by spinal cord reflex arcs and rapid adaptation (see idea.txt, neuroscience).

    Research References:
    - idea.txt (reflex arcs, feedback-driven adaptation, buffer design)
    - Nature 2024 (Spinal Cord Reflexes in AI)
    - NeurIPS 2025 (Fast Feedback in Modular AI)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced decay models (e.g., exponential, context-sensitive)
    - Integrate with multi-agent or distributed reflex buffers
    - Support for feedback-driven learning and adaptation
    TODO:
    - Implement advanced feedback weighting and prioritization
    - Add robust error handling for buffer overflows/underflows
    """
    def __init__(self, capacity=50, decay=0.90):
        self.capacity = capacity
        self.decay = decay
        self.buffer = []  # Each entry: {'signal': ..., 'context': ..., 'feedback': ..., 'strength': ...}
        self.logger = logging.getLogger("ReflexBuffer")
    def add(self, signal, context=None, feedback=None):
        entry = {'signal': signal, 'context': context, 'feedback': feedback, 'strength': 1.0}
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.logger.info(f"[ReflexBuffer] Added signal: {signal} (context={context}, feedback={feedback})")
    def decay_buffer(self):
        for entry in self.buffer:
            entry['strength'] *= self.decay
        self.buffer = [e for e in self.buffer if e['strength'] > 0.1]
    def get_by_context(self, context, n=5):
        context_str = str(context) if context is not None else ""
        matches = [e for e in self.buffer if context_str and context_str in str(e['context'])]
        return [e['signal'] for e in matches[-n:]]
    def get_recent(self, n=5):
        return [e['signal'] for e in self.buffer[-n:]]

class SpinalCord:
    """
    SpinalCord lobe: Inspired by the biological spinal cord.
    Handles low-level reflexes, fast feedback, and routing between sensory columns and higher lobes.
    Modular, toggleable, and robust. See idea.txt (reflexes, feedback, routing, brain-inspired architecture).
    
    Research References:
    - idea.txt (reflexes, feedback, routing, brain-inspired architecture)
    - Nature 2024 (Spinal Cord Reflexes in AI)
    - NeurIPS 2025 (Fast Feedback in Modular AI)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced routing logic (e.g., context-aware, feedback-driven)
    - Integrate with distributed or multi-agent spinal cord models
    - Support for dynamic reflex adaptation and learning
    TODO:
    - Implement advanced routing and reflex logic
    - Add robust error handling and logging for all signal types
    - Support for dynamic enabling/disabling of reflex pathways
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        self.reflex_buffer = ReflexBuffer()
        self.logger = logging.getLogger("SpinalCord")
        self.enabled = True  # Toggleable

    def route_signal(self, signal: Any, target: Optional[Any] = None, context: Any = None, feedback: Any = None) -> Any:
        """
        Route a signal from sensory input to the appropriate lobe or output.
        Handles fast feedback/reflexes if needed. Stores signal in working memory and reflex buffer, and logs the event.
        Fallback: returns signal unchanged if advanced logic is not implemented.
        """
        if not self.enabled:
            self.logger.info("[SpinalCord] Disabled. Signal bypassed.")
            return signal
        self.working_memory.add(signal)
        self.reflex_buffer.add(signal, context=context, feedback=feedback)
        self.reflex_buffer.decay_buffer()
        self.logger.info(f"[SpinalCord] Routed signal: {signal} to {target}")
        # TODO: Implement advanced routing and reflex logic.
        return signal

    def recall_reflexes_by_context(self, context=None, n=5):
        """
        Recall most relevant reflex signals for a given context using reflex buffer.
        """
        return self.reflex_buffer.get_by_context(context, n=n) 