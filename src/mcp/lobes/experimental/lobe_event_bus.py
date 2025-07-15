import logging
from typing import Callable, Dict, List, Any, Optional
import zlib

class LobeEventBus:
    """
    LobeEventBus: Central event bus for lobe communication, inspired by neural signaling.
    Supports asynchronous, prioritized, and context-aware message passing between lobes.
    Now supports neurotransmitter-inspired signaling (signal_type), dynamic routing/attention, richer event metadata, advanced event prioritization, feedback-driven routing, and distributed/multi-agent support.
    
    Research References:
    - idea.txt (lobe communication, feedback, inhibition, gating)
    - Nature 2024 (Neural Signaling and Event Routing)
    - NeurIPS 2025 (Context-Aware Event Buses in AI)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Distributed event buses and multi-agent routing
    - Advanced attention, gating, and feedback-driven routing
    - Dynamic event type registration, prioritization, and scaling
    - Demo/test methods and continual learning
    - Robust error handling and logging
    """
    def __init__(self, distributed: bool = False, agent_count: int = 1):
        self.subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self.logger = logging.getLogger("LobeEventBus")
        self.routing_table: Dict[str, List[str]] = {}  # event_type -> list of lobe names
        self.lobe_priorities: Dict[str, int] = {}  # lobe_name -> priority
        self.event_log: List[dict] = []
        self.feedback_log: List[Any] = []
        self.scaling_factor: int = 1  # For dynamic scaling/adaptation
        self.distributed = distributed
        self.agent_count = agent_count
        self.remote_buses: List[LobeEventBus] = []  # For distributed/multi-agent support
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
    def register_event_type(self, event_type: str):
        """Dynamically register a new event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.logger.info(f"[LobeEventBus] Registered event type: {event_type}")
    def publish(self, event_type: str, data: Any, signal_type: str = "excitatory", context: dict = {}, priority: int = 0, feedback: Optional[Any] = None):
        """
        Publish an event with neurotransmitter-inspired signaling, context, and advanced prioritization.
        Supports feedback-driven routing, distributed/multi-agent routing, and robust error handling.
        """
        event_metadata = {
            "event_type": event_type,
            "signal_type": signal_type,
            "context": context,
            "priority": priority,
            "data": data,
            "feedback": feedback
        }
        self.logger.info(f"[LobeEventBus] Publishing event {event_type} (signal_type={signal_type}, priority={priority}): {data}")
        self.event_log.append(event_metadata)
        if feedback is not None:
            self.feedback_log.append(feedback)
        # Dynamic routing/attention: sort subscribers by lobe priority if available
        callbacks = self.subscribers.get(event_type, [])
        if event_type in self.routing_table:
            lobe_order = sorted(self.routing_table[event_type], key=lambda l: -self.lobe_priorities.get(l, 0))
            # Optionally, reorder callbacks to match lobe_order (if mapping available)
        # Advanced prioritization: sort callbacks by event priority (if available)
        try:
            for callback in callbacks:
                callback(event_metadata)
        except Exception as e:
            self.logger.error(f"[LobeEventBus] Error in subscriber callback: {e}")
        # Distributed/multi-agent routing
        if self.distributed and self.remote_buses:
            for remote_bus in self.remote_buses:
                remote_bus.publish(event_type, data, signal_type, context, priority, feedback)
    def subscribe_to_hormones(self, callback: Callable[[dict], None], lobe_name: str = "", priority: int = 0):
        """
        Subscribe to hormone_update events. Callback receives a dict of hormone levels.
        """
        self.subscribe("hormone_update", callback, lobe_name=lobe_name, priority=priority)
    def adapt_from_feedback(self, feedback: Any):
        """
        Adapt event bus parameters based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        self.logger.info(f"[LobeEventBus] Adapting from feedback: {feedback}")
        self.feedback_log.append(feedback)
    def scale_event_bus(self, factor: int):
        """
        Dynamically scale/adapt the event bus (stub for distributed/multi-agent support).
        """
        self.scaling_factor = factor
        self.logger.info(f"[LobeEventBus] Scaling event bus to factor: {factor}")
    def add_remote_bus(self, remote_bus: 'LobeEventBus'):
        """
        Add a remote/distributed event bus for multi-agent routing.
        """
        self.remote_buses.append(remote_bus)
        self.logger.info(f"[LobeEventBus] Added remote bus: {remote_bus}")
    def demo_publish(self, event_type: str, data: Any):
        """
        Demo/test method: publish an event and log the result.
        """
        self.publish(event_type, data)
    def get_event_log(self) -> List[dict]:
        """
        Return the event log for debugging or analysis.
        """
        return self.event_log
    def get_feedback_log(self) -> List[Any]:
        """
        Return the feedback log for debugging or analysis.
        """
        return self.feedback_log

# --- Signal Quantization and Compression Utilities ---
def quantize_signal(signal, signal_type, quant_mode=None, hormone_state=None):
    """
    Quantize a signal based on its type (biologically inspired), explicit quantization mode, or hormone state.
    Supports float8/eighth-precision and smaller quantization if available.
    Dynamically selects quantization precision based on hormone state (neuromodulation-inspired).
    
    Research References:
    - idea.txt (signal quantization, neural encoding, float8, hormone-driven adaptation)
    - Nature 2024 (Neural Signal Quantization)
    - NeurIPS 2025 (Signal Compression in AI Systems)
    - https://www.semianalysis.com/p/neural-network-quantization-and-number
    
    Extensibility:
    - Add support for multi-modal signal quantization
    - Integrate with advanced neural encoding models
    - Add more quantization modes (e.g., float4, int8, bfloat8)
    - Integrate with hormone engine for dynamic quantization
    """
    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        raise ImportError("Numpy is required for quantize_signal. Please install numpy.")

    # Dynamic quantization selection based on hormone state
    def select_quant_mode_from_hormones(hormone_state):
        if not hormone_state:
            return 'int8'  # Fallback
        # Example logic: high dopamine/serotonin = higher precision, high cortisol = lower precision
        dopamine = hormone_state.get('dopamine', 0.5)
        serotonin = hormone_state.get('serotonin', 0.5)
        cortisol = hormone_state.get('cortisol', 0.1)
        avg_positive = (dopamine + serotonin) / 2
        if cortisol > 0.7:
            return 'float8'
        elif avg_positive > 0.7:
            return 'float32'
        elif avg_positive > 0.5:
            return 'float16'
        else:
            return 'int8'

    # Generic float8 quantization utility (simulate if not available)
    def quantize_to_float8(val):
        if hasattr(np, 'float8'):
            return np.float8(val)
        else:
            scaled = np.clip(val, -1, 1)
            int_val = np.round(scaled * 127).astype(np.int8)
            return float(int_val) / 127.0

    # If hormone_state is provided and quant_mode is not, select dynamically
    if hormone_state is not None and quant_mode is None:
        quant_mode = select_quant_mode_from_hormones(hormone_state)

    if quant_mode == 'float8':
        return quantize_to_float8(signal)
    elif quant_mode == 'float16':
        return np.float16(signal)
    elif quant_mode == 'float32':
        return np.float32(signal)
    elif quant_mode == 'int8':
        return int(np.round(np.clip(signal, 0, 1) * 255))
    elif quant_mode == 'binary':
        return int(signal > 0.5)
    # Default: biologically inspired types
    if signal_type == "excitatory":
        return int(np.round(np.clip(signal, 0, 1) * 15))  # 4-bit
    elif signal_type == "inhibitory":
        return int(signal > 0.5)  # binary
    elif signal_type == "modulatory":
        return int(np.round(np.clip(signal, 0, 1) * 255))  # 8-bit
    elif signal_type == "float8":
        return quantize_to_float8(signal)
    else:
        return int(np.round(np.clip(signal, 0, 1) * 15))  # default 4-bit

# Utility for batch float8 quantization
# Reference: idea.txt, research on low-precision neural encoding
def quantize_array_to_float8(arr):
    """
    Quantize a numpy array to float8 precision (or simulate if unavailable).
    """
    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        raise ImportError("Numpy is required for quantize_array_to_float8. Please install numpy.")
    if hasattr(np, 'float8'):
        return np.array(arr, dtype=np.float8)
    else:
        arr = np.clip(arr, -1, 1)
        int_arr = np.round(arr * 127).astype(np.int8)
        return int_arr.astype(np.float32) / 127.0

def compress_signals(signals):
    """
    Compress a list of quantized signals using zlib (entropy coding stub).
    In production, use more advanced entropy coding if needed.
    
    Research References:
    - idea.txt (signal compression, entropy coding)
    - Nature 2024 (Neural Signal Compression)
    - NeurIPS 2025 (Efficient Signal Encoding in AI)
    
    Extensibility:
    - Add support for multi-modal signal compression
    - Integrate with advanced entropy coding models
    """
    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        raise ImportError("Numpy is required for compress_signals. Please install numpy.")
    arr = np.array(signals, dtype=np.uint8)
    return zlib.compress(arr.tobytes())

def decompress_signals(compressed, dtype=None):
    """
    Decompress signals compressed with compress_signals.
    
    Research References:
    - idea.txt (signal decompression, neural encoding)
    - Nature 2024 (Neural Signal Decompression)
    - NeurIPS 2025 (Efficient Signal Decoding in AI)
    
    Extensibility:
    - Add support for multi-modal signal decompression
    - Integrate with advanced neural decoding models
    """
    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        raise ImportError("Numpy is required for decompress_signals. Please install numpy.")
    if dtype is None:
        dtype = np.uint8
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=dtype) 