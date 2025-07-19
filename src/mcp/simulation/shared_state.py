import threading
from typing import Dict, Any, List, Optional

class SharedSimulationState:
    """
    Thread-safe global simulation state registry for all simulation engines.
    Provides methods for registering, updating, and querying entities, events, and states.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.states: Dict[str, Dict[str, Any]] = {}

    def register_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        with self._lock:
            self.entities[entity_id] = entity_data

    def update_entity(self, entity_id: str, updates: Dict[str, Any]):
        with self._lock:
            if entity_id in self.entities:
                self.entities[entity_id].update(updates)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.entities.get(entity_id)

    def register_event(self, event: Dict[str, Any]):
        with self._lock:
            self.events.append(event)

    def get_events(self, filter_fn=None) -> List[Dict[str, Any]]:
        with self._lock:
            if filter_fn:
                return [e for e in self.events if filter_fn(e)]
            return list(self.events)

    def register_state(self, state_id: str, state_data: Dict[str, Any]):
        with self._lock:
            self.states[state_id] = state_data

    def update_state(self, state_id: str, updates: Dict[str, Any]):
        with self._lock:
            if state_id in self.states:
                self.states[state_id].update(updates)

    def get_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.states.get(state_id)

    def get_all_entities(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self.entities)

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self.states)

    def get_all_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.events) 