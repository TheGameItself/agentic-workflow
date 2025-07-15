from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
import logging
from typing import Any, Optional, Dict
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
import random
import time

class QuantalVesicle:
    """
    Represents a discrete 'quantum' of sensory input, inspired by synaptic vesicle release.
    Supports both spontaneous (noise/baseline) and evoked (event-driven) release.
    """
    def __init__(self, data, channel, evoked=True, timestamp=None):
        self.data = data
        self.channel = channel
        self.evoked = evoked
        self.timestamp = timestamp or time.time()
    def __repr__(self):
        mode = 'evoked' if self.evoked else 'spontaneous'
        return f"<QuantalVesicle channel={self.channel} mode={mode} data={self.data}>"

class VisualChannel:
    """Visual sensory channel with its own working memory and preprocessing."""
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("VisualChannel")
    def process_input(self, data: Any) -> Any:
        self.working_memory.add(data)
        self.logger.info(f"[VisualChannel] Processed visual input: {data}")
        # TODO: Add visual preprocessing (normalization, feature extraction)
        return data
    def quantal_release(self, data: Any, evoked=True) -> QuantalVesicle:
        vesicle = QuantalVesicle(data, channel="visual", evoked=evoked)
        self.logger.info(f"[VisualChannel] Quantal release: {vesicle}")
        return vesicle
    def spontaneous_release(self) -> QuantalVesicle:
        # Simulate spontaneous (noise) release
        noise = random.gauss(0, 0.1)
        vesicle = QuantalVesicle(noise, channel="visual", evoked=False)
        self.logger.info(f"[VisualChannel] Spontaneous quantal release: {vesicle}")
        return vesicle

class AuditoryChannel:
    """Auditory sensory channel with its own working memory and preprocessing."""
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("AuditoryChannel")
    def process_input(self, data: Any) -> Any:
        self.working_memory.add(data)
        self.logger.info(f"[AuditoryChannel] Processed auditory input: {data}")
        # TODO: Add auditory preprocessing (feature extraction, echoic memory)
        return data
    def quantal_release(self, data: Any, evoked=True) -> QuantalVesicle:
        vesicle = QuantalVesicle(data, channel="auditory", evoked=evoked)
        self.logger.info(f"[AuditoryChannel] Quantal release: {vesicle}")
        return vesicle
    def spontaneous_release(self) -> QuantalVesicle:
        # Simulate spontaneous (noise) release
        noise = random.gauss(0, 0.1)
        vesicle = QuantalVesicle(noise, channel="auditory", evoked=False)
        self.logger.info(f"[AuditoryChannel] Spontaneous quantal release: {vesicle}")
        return vesicle

class SensoryColumn:
    """
    SensoryColumn lobe: Inspired by neural columns and quantal neurotransmitter release.
    Models discrete, vesicle-like 'quanta' for inter-lobe signaling, supports both spontaneous and evoked release modes.
    See idea.txt and Wikipedia - Quantal neurotransmitter release.
    """
    def __init__(self, db_path: Optional[str] = None, event_bus: Optional[LobeEventBus] = None):
        self.db_path = db_path
        self.logger = logging.getLogger("SensoryColumn")
        self.enabled = True  # Toggleable
        # Register channels
        self.channels: Dict[str, Any] = {
            "visual": VisualChannel(),
            "auditory": AuditoryChannel(),
        }
        self.event_bus = event_bus or LobeEventBus()
        self.spontaneous_rate = 0.01  # Probability per call for spontaneous release
    def process_input(self, data: Any, channel: str = "visual") -> Any:
        """
        Route input to the appropriate sensory channel.
        """
        if not self.enabled:
            self.logger.info("[SensoryColumn] Disabled. Input bypassed.")
            return data
        if channel in self.channels:
            # Evoked quantal release
            vesicle = self.channels[channel].quantal_release(data, evoked=True)
            self.event_bus.publish('sensory_quantal_release', {'vesicle': vesicle.__dict__})
            # Occasionally emit spontaneous quantal release
            if random.random() < self.spontaneous_rate:
                spont_vesicle = self.channels[channel].spontaneous_release()
                self.event_bus.publish('sensory_quantal_release', {'vesicle': spont_vesicle.__dict__})
            return vesicle.data
        else:
            self.logger.warning(f"[SensoryColumn] Unknown channel: {channel}. Input bypassed.")
            return data 