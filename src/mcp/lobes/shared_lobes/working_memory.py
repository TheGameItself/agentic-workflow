import logging
from typing import Any, List, Dict, Optional

class WorkingMemory:
    """
    Context-sensitive, temporary working memory for lobes/engines.
    Used for feedback, adaptation, and short-lived storage. See idea.txt.
    """
    def __init__(self):
        self.memory: List[Any] = []
        self.logger = logging.getLogger("WorkingMemory")
    def add(self, item: Any):
        self.memory.append(item)
        self.logger.info(f"[WorkingMemory] Added item: {item}")
    def clear(self):
        self.memory.clear()
        self.logger.info("[WorkingMemory] Cleared working memory.")
    def get_all(self) -> List[Any]:
        return self.memory.copy()

class ShortTermMemory:
    """
    Short-term memory for recent, high-priority, or volatile information.
    Stores recent tasks, user interactions, session data, etc. See idea.txt.
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory: List[Any] = []
        self.logger = logging.getLogger("ShortTermMemory")
    def add(self, item: Any):
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.logger.info(f"[ShortTermMemory] Added item: {item}")
    def get_recent(self, n: int = 5) -> List[Any]:
        return self.memory[-n:]
    def clear(self):
        self.memory.clear()
        self.logger.info("[ShortTermMemory] Cleared short-term memory.")
    def get_all(self) -> List[Any]:
        return self.memory.copy()

class LongTermMemory:
    """
    Long-term memory for persistent, structured, and research-driven storage.
    Stores vector DBs, engrams, knowledge, historical logs, etc. See idea.txt.
    """
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.logger = logging.getLogger("LongTermMemory")
    def add(self, key: str, value: Any):
        self.memory[key] = value
        self.logger.info(f"[LongTermMemory] Added/updated key: {key}")
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