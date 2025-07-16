import logging
from typing import Any, List, Dict, Optional, Callable

class WorkingMemory:
    """
    Context-sensitive, temporary working memory for lobes/engines.
    Used for feedback, adaptation, and short-lived storage.
    See idea.txt for requirements and research references.
    Extensible for custom fallback and error handling.
    """
    def __init__(self, fallback: Optional[Callable] = None):
        self.memory: List[Any] = []
        self.logger = logging.getLogger("WorkingMemory")
        self.fallback = fallback
    def add(self, item: Any):
        try:
            self.memory.append(item)
            self.logger.info(f"[WorkingMemory] Added item: {item}")
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error adding item: {ex}")
            if self.fallback:
                self.fallback(item)
    def clear(self):
        self.memory.clear()
        self.logger.info("[WorkingMemory] Cleared working memory.")
    def get_all(self) -> List[Any]:
        return self.memory.copy()
    def fallback_hook(self, *args, **kwargs):
        if self.fallback:
            return self.fallback(*args, **kwargs)
        self.logger.warning("[WorkingMemory] No fallback defined.")

class ShortTermMemory:
    """
    Short-term memory for recent, high-priority, or volatile information.
    Stores recent tasks, user interactions, session data, etc.
    See idea.txt for requirements and research references.
    Extensible for custom fallback and error handling.
    """
    def __init__(self, capacity: int = 100, fallback: Optional[Callable] = None):
        self.capacity = capacity
        self.memory: List[Any] = []
        self.logger = logging.getLogger("ShortTermMemory")
        self.fallback = fallback
    def add(self, item: Any):
        try:
            self.memory.append(item)
            if len(self.memory) > self.capacity:
                self.memory.pop(0)
            self.logger.info(f"[ShortTermMemory] Added item: {item}")
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error adding item: {ex}")
            if self.fallback:
                self.fallback(item)
    def get_recent(self, n: int = 5) -> List[Any]:
        return self.memory[-n:]
    def clear(self):
        self.memory.clear()
        self.logger.info("[ShortTermMemory] Cleared short-term memory.")
    def get_all(self) -> List[Any]:
        return self.memory.copy()
    def fallback_hook(self, *args, **kwargs):
        if self.fallback:
            return self.fallback(*args, **kwargs)
        self.logger.warning("[ShortTermMemory] No fallback defined.")

class LongTermMemory:
    """
    Long-term memory for persistent, structured, and research-driven storage.
    Stores vector DBs, engrams, knowledge, historical logs, etc.
    See idea.txt for requirements and research references.
    Extensible for custom fallback and error handling.
    """
    def __init__(self, fallback: Optional[Callable] = None):
        self.memory: Dict[str, Any] = {}
        self.logger = logging.getLogger("LongTermMemory")
        self.fallback = fallback
    def add(self, key: str, value: Any):
        try:
            self.memory[key] = value
            self.logger.info(f"[LongTermMemory] Added/updated key: {key}")
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error adding key {key}: {ex}")
            if self.fallback:
                self.fallback(key, value)
    def get(self, key: str) -> Optional[Any]:
        return self.memory.get(key)
    def remove(self, key: str):
        if key in self.memory:
            del self.memory[key]
            self.logger.info(f"[LongTermMemory] Removed key: {key}")
    def get_all(self) -> Dict[str, Any]:
        return self.memory.copy()
    def clear(self):
        self.memory.clear()
        self.logger.info("[LongTermMemory] Cleared long-term memory.")
    def fallback_hook(self, *args, **kwargs):
        if self.fallback:
            return self.fallback(*args, **kwargs)
        self.logger.warning("[LongTermMemory] No fallback defined.") 