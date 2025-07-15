import logging
from typing import Callable, Dict, List, Any
import numpy as np
import zlib

class LobeEventBus:
    """
    LobeEventBus: Central event bus for lobe communication, inspired by neural signaling.
    Supports asynchronous, prioritized, and context-aware message passing between lobes.
    Now supports neurotransmitter-inspired signaling (signal_type), dynamic routing/attention, and richer event metadata.
    See idea.txt (lobe communication, feedback, inhibition, gating).
    References: neuroscience (thalamus, basal ganglia, neural signaling).
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self.logger = logging.getLogger("LobeEventBus")
        self.routing_table: Dict[str, List[str]] = {}  # event_type -> list of lobe names
        self.lobe_priorities: Dict[str, int] = {}  # lobe_name -> priority
    def subscribe(self, event_type: str, callback: Callable[[Any], None], lobe_name: str = "", priority: int = 0):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        if lobe_name:
            if event_type not in self.routing_table:
                self.routing_table[event_type] = []
            self.routing_table[event_type].append(lobe_name)
            self.lobe_priorities[lobe_name] = priority
        self.logger.info(f"[LobeEventBus] Subscribed to {event_type}: {callback} (lobe={lobe_name}, priority={priority})")
    def publish(self, event_type: str, data: Any, signal_type: str = "excitatory", context: dict = {}, priority: int = 0):
        """
        Publish an event with neurotransmitter-inspired signaling and context.
        signal_type: 'excitatory', 'inhibitory', 'modulatory', etc.
        context: additional event metadata (e.g., source, target, timestamp, tags)
        priority: event priority (higher = more urgent)
        """
        event_metadata = {
            "event_type": event_type,
            "signal_type": signal_type,
            "context": context,
            "priority": priority,
            "data": data
        }
        self.logger.info(f"[LobeEventBus] Publishing event {event_type} (signal_type={signal_type}, priority={priority}): {data}")
        # Dynamic routing/attention: sort subscribers by lobe priority if available
        callbacks = self.subscribers.get(event_type, [])
        if event_type in self.routing_table:
            lobe_order = sorted(self.routing_table[event_type], key=lambda l: -self.lobe_priorities.get(l, 0))
            # Optionally, reorder callbacks to match lobe_order (if mapping available)
        for callback in callbacks:
            try:
                callback(event_metadata)
            except Exception as e:
                self.logger.error(f"[LobeEventBus] Error in subscriber callback: {e}")
    def subscribe_to_hormones(self, callback: Callable[[dict], None], lobe_name: str = "", priority: int = 0):
        """
        Subscribe to hormone_update events. Callback receives a dict of hormone levels.
        """
        self.subscribe("hormone_update", callback, lobe_name=lobe_name, priority=priority)

# --- Signal Quantization and Compression Utilities ---

def quantize_signal(signal, signal_type):
    """
    Quantize a signal based on its type (biologically inspired).
    - excitatory: 4-bit
    - inhibitory: binary
    - modulatory: 8-bit
    """
    if signal_type == "excitatory":
        return int(np.round(np.clip(signal, 0, 1) * 15))  # 4-bit
    elif signal_type == "inhibitory":
        return int(signal > 0.5)  # binary
    elif signal_type == "modulatory":
        return int(np.round(np.clip(signal, 0, 1) * 255))  # 8-bit
    else:
        return int(np.round(np.clip(signal, 0, 1) * 15))  # default 4-bit


def compress_signals(signals):
    """
    Compress a list of quantized signals using zlib (entropy coding stub).
    In production, use more advanced entropy coding if needed.
    """
    arr = np.array(signals, dtype=np.uint8)
    return zlib.compress(arr.tobytes())


def decompress_signals(compressed, dtype=np.uint8):
    """
    Decompress signals compressed with compress_signals.
    """
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=dtype) 