from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
import logging
from typing import Any, Optional, Dict, Callable, List
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
import random
import time

class QuantalVesicle:
    """
    Represents a discrete 'quantum' of sensory input, inspired by synaptic vesicle release.
    Supports both spontaneous (noise/baseline) and evoked (event-driven) release.

    Research References:
    - idea.txt (quantal signaling, neural columns, sensory input modeling)
    - Nature 2024 (Quantal Neurotransmitter Release)
    - NeurIPS 2025 (Neural Column Pattern Recognition)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add new vesicle types (e.g., chemical, electrical, synthetic)
    - Integrate with advanced event bus or signal routing
    - Support for multi-modal quantal encoding
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
    """Visual sensory channel with its own working memory and preprocessing.
    Extensible for advanced visual preprocessing, LLM-based vision, and multi-modal input.
    """
    def __init__(self, feature_extractor: Optional[Callable] = None):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("VisualChannel")
        self.feature_extractor = feature_extractor
    def process_input(self, data: Any) -> Any:
        self.working_memory.add(data)
        self.logger.info(f"[VisualChannel] Processed visual input: {data}")
        if self.feature_extractor and callable(self.feature_extractor):
            try:
                features = self.feature_extractor(data)
                self.logger.info(f"[VisualChannel] Extracted features: {features}")
                return features
            except Exception as ex:
                self.logger.error(f"[VisualChannel] Feature extraction error: {ex}")
        # TODO: Add visual preprocessing (normalization, feature extraction)
        return data
    def quantal_release(self, data: Any, evoked=True) -> QuantalVesicle:
        vesicle = QuantalVesicle(data, channel="visual", evoked=evoked)
        self.logger.info(f"[VisualChannel] Quantal release: {vesicle}")
        return vesicle
    def spontaneous_release(self) -> QuantalVesicle:
        noise = random.gauss(0, 0.1)
        vesicle = QuantalVesicle(noise, channel="visual", evoked=False)
        self.logger.info(f"[VisualChannel] Spontaneous quantal release: {vesicle}")
        return vesicle

class AuditoryChannel:
    """Auditory sensory channel with its own working memory and preprocessing.
    Extensible for advanced auditory preprocessing, LLM-based audio, and multi-modal input.
    """
    def __init__(self, feature_extractor: Optional[Callable] = None):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("AuditoryChannel")
        self.feature_extractor = feature_extractor
    def process_input(self, data: Any) -> Any:
        self.working_memory.add(data)
        self.logger.info(f"[AuditoryChannel] Processed auditory input: {data}")
        if self.feature_extractor and callable(self.feature_extractor):
            try:
                features = self.feature_extractor(data)
                self.logger.info(f"[AuditoryChannel] Extracted features: {features}")
                return features
            except Exception as ex:
                self.logger.error(f"[AuditoryChannel] Feature extraction error: {ex}")
        # TODO: Add auditory preprocessing (feature extraction, echoic memory)
        return data
    def quantal_release(self, data: Any, evoked=True) -> QuantalVesicle:
        vesicle = QuantalVesicle(data, channel="auditory", evoked=evoked)
        self.logger.info(f"[AuditoryChannel] Quantal release: {vesicle}")
        return vesicle
    def spontaneous_release(self) -> QuantalVesicle:
        noise = random.gauss(0, 0.1)
        vesicle = QuantalVesicle(noise, channel="auditory", evoked=False)
        self.logger.info(f"[AuditoryChannel] Spontaneous quantal release: {vesicle}")
        return vesicle

class SensoryColumn:
    """
    SensoryColumn lobe: Inspired by neural columns and quantal neurotransmitter release.
    Models discrete, vesicle-like 'quanta' for inter-lobe signaling, supports both spontaneous and evoked release modes.
    Supports dynamic channel registration/removal, multi-modal/cross-modal integration, feedback-driven adaptation, and robust error handling.
    Prepares for LLM-based sensory processing and continual learning.
    
    Research References:
    - idea.txt (neural columns, quantal signaling, sensory input modeling)
    - Nature 2024 (Quantal Neurotransmitter Release)
    - NeurIPS 2025 (Neural Column Pattern Recognition)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add new sensory channels (e.g., tactile, olfactory, synthetic)
    - Integrate with advanced event bus or signal routing
    - Support for multi-modal and cross-modal sensory integration
    - Add feedback-driven adaptation and learning
    - Dynamic channel registration/removal
    - LLM-based sensory processing
    """
    def __init__(self, db_path: Optional[str] = None, event_bus: Optional[LobeEventBus] = None):
        self.db_path = db_path
        self.logger = logging.getLogger("SensoryColumn")
        self.enabled = True  # Toggleable
        self.channels: Dict[str, Any] = {
            "visual": VisualChannel(),
            "auditory": AuditoryChannel(),
        }
        self.event_bus = event_bus or LobeEventBus()
        self.spontaneous_rate = 0.01  # Probability per call for spontaneous release
        self.feedback_log: List[Any] = []
    def process_input(self, data: Any, channel: str = "visual") -> Any:
        """
        Route input to the appropriate sensory channel. Supports robust error handling and feedback-driven adaptation.
        """
        if not self.enabled:
            self.logger.info("[SensoryColumn] Disabled. Input bypassed.")
            return data
        if channel in self.channels:
            try:
                vesicle = self.channels[channel].quantal_release(data, evoked=True)
                self.event_bus.publish('sensory_quantal_release', {'vesicle': vesicle.__dict__})
                if random.random() < self.spontaneous_rate:
                    spont_vesicle = self.channels[channel].spontaneous_release()
                    self.event_bus.publish('sensory_quantal_release', {'vesicle': spont_vesicle.__dict__})
                return vesicle.data
            except Exception as ex:
                self.logger.error(f"[SensoryColumn] Error processing input for channel '{channel}': {ex}")
                return data
        else:
            self.logger.warning(f"[SensoryColumn] Unknown channel: {channel}. Input bypassed.")
            return data
    def register_channel(self, name: str, channel_obj: Any):
        """Dynamically register a new sensory channel."""
        self.channels[name] = channel_obj
        self.logger.info(f"[SensoryColumn] Registered new channel: {name}")
    def remove_channel(self, name: str):
        """Dynamically remove a sensory channel."""
        if name in self.channels:
            del self.channels[name]
            self.logger.info(f"[SensoryColumn] Removed channel: {name}")
    def adapt_from_feedback(self, feedback: Any):
        """
        Adapt sensory processing parameters based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        self.logger.info(f"[SensoryColumn] Adapting from feedback: {feedback}")
        self.feedback_log.append(feedback) 