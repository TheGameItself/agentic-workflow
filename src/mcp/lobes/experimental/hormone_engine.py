"""
HormoneEngine: Biologically inspired neuromodulator simulation for MCP
Tracks hormone levels (dopamine, serotonin, acetylcholine, noradrenaline, cortisol), updates based on events, applies decay, and broadcasts state via event bus.
References: See idea.txt, Inworld AI agent research, and arXiv:2406.06237 for quantization/compression inspiration.
"""

from typing import Dict, Callable
import logging
import threading
import time

class HormoneEngine:
    def __init__(self, event_bus, decay_rate: float = 0.01, tick_interval: float = 1.0):
        self.levels: Dict[str, float] = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "acetylcholine": 0.5,
            "noradrenaline": 0.5,
            "cortisol": 0.1,
        }
        self.event_bus = event_bus
        self.decay_rate = decay_rate
        self.tick_interval = tick_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._register_event_handlers()
        self._thread.start()
        logging.info("HormoneEngine initialized with levels: %s", self.levels)

    def _register_event_handlers(self):
        self.event_bus.subscribe("reward", self.on_reward)
        self.event_bus.subscribe("error", self.on_error)
        self.event_bus.subscribe("novelty", self.on_novelty)
        self.event_bus.subscribe("task_completed", self.on_task_completed)
        self.event_bus.subscribe("goal_failed", self.on_goal_failed)
        # Extend with more event types as needed

    def on_reward(self, event):
        self._adjust("dopamine", 0.2)
        self._broadcast()

    def on_error(self, event):
        self._adjust("cortisol", 0.2)
        self._broadcast()

    def on_novelty(self, event):
        self._adjust("noradrenaline", 0.1)
        self._broadcast()

    def on_task_completed(self, event):
        self._adjust("serotonin", 0.1)
        self._broadcast()

    def on_goal_failed(self, event):
        self._adjust("cortisol", 0.1)
        self._broadcast()

    def _adjust(self, hormone: str, delta: float):
        old = self.levels[hormone]
        self.levels[hormone] = min(1.0, max(0.0, self.levels[hormone] + delta))
        logging.debug(f"Hormone '{hormone}' adjusted from {old} to {self.levels[hormone]}")

    def _decay_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.tick_interval)
            for k in self.levels:
                old = self.levels[k]
                self.levels[k] = max(0.0, self.levels[k] - self.decay_rate)
                if self.levels[k] != old:
                    logging.debug(f"Hormone '{k}' decayed from {old} to {self.levels[k]}")
            self._broadcast()

    def _broadcast(self):
        self.event_bus.emit("hormone_update", dict(self.levels))
        logging.info(f"HormoneEngine broadcast: {self.levels}")

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def get_levels(self):
        """Return current hormone levels for aggregation."""
        return dict(self.levels)

    def receive_data(self, data: dict):
        """Stub: Receive data from aggregator or adjacent lobes."""
        logging.info(f"[HormoneEngine] Received data: {data}")
        # TODO: Integrate received data into hormone state

# Usage: Instantiate HormoneEngine with the event bus in the MCP system. 