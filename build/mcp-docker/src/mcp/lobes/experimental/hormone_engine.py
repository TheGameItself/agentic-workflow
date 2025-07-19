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
        """
        Receive data from aggregator or adjacent lobes with hormone-based processing.
        
        Implements cross-lobe communication by translating received data into
        appropriate hormone releases and level adjustments based on biological
        hormone system patterns.
        """
        logging.info(f"[HormoneEngine] Received data from {data.get('source', 'unknown')}")
        
        try:
            # Extract data components
            source_lobe = data.get('source', 'unknown')
            data_type = data.get('type', 'general')
            content = data.get('content', {})
            importance = data.get('importance', 0.5)
            timestamp = data.get('timestamp', time.time())
            
            # Process different types of data and trigger appropriate hormone responses
            if data_type == 'success_signal':
                self._process_success_signal(content, source_lobe, importance)
            elif data_type == 'error_signal':
                self._process_error_signal(content, source_lobe, importance)
            elif data_type == 'performance_data':
                self._process_performance_data(content, source_lobe, importance)
            elif data_type == 'user_feedback':
                self._process_user_feedback(content, source_lobe, importance)
            elif data_type == 'task_completion':
                self._process_task_completion(content, source_lobe, importance)
            elif data_type == 'learning_progress':
                self._process_learning_progress(content, source_lobe, importance)
            elif data_type == 'resource_status':
                self._process_resource_status(content, source_lobe, importance)
            else:
                self._process_general_signal(content, source_lobe, importance)
            
            # Update cross-lobe communication tracking
            if not hasattr(self, 'lobe_interactions'):
                self.lobe_interactions = {}
            
            if source_lobe not in self.lobe_interactions:
                self.lobe_interactions[source_lobe] = {
                    'message_count': 0,
                    'last_interaction': 0,
                    'hormone_triggers': {},
                    'average_importance': 0.5
                }
            
            interaction = self.lobe_interactions[source_lobe]
            interaction['message_count'] += 1
            interaction['last_interaction'] = timestamp
            interaction['average_importance'] = (
                0.9 * interaction['average_importance'] + 0.1 * importance
            )
            
            # Track which hormones this lobe tends to trigger
            if data_type not in interaction['hormone_triggers']:
                interaction['hormone_triggers'][data_type] = 0
            interaction['hormone_triggers'][data_type] += 1
            
            logging.info(f"[HormoneEngine] Successfully processed {data_type} from {source_lobe}")
            
        except Exception as e:
            logging.error(f"[HormoneEngine] Error processing received data: {e}")
            # Release cortisol on processing errors
            self._adjust('cortisol', 0.1)
    
    def _process_success_signal(self, content: dict, source: str, importance: float):
        """Process success signals and release appropriate reward hormones."""
        success_level = content.get('success_level', 0.5)
        task_type = content.get('task_type', 'general')
        
        # Release dopamine for success
        dopamine_amount = min(0.3, success_level * importance * 0.2)
        self._adjust('dopamine', dopamine_amount)
        
        # Release serotonin for sustained well-being
        if success_level > 0.8:
            serotonin_amount = min(0.2, success_level * importance * 0.15)
            self._adjust('serotonin', serotonin_amount)
        
        logging.info(f"[HormoneEngine] Success signal from {source}: dopamine +{dopamine_amount:.3f}")
    
    def _process_error_signal(self, content: dict, source: str, importance: float):
        """Process error signals and release stress hormones."""
        error_severity = content.get('severity', 0.5)
        error_type = content.get('error_type', 'general')
        recoverable = content.get('recoverable', True)
        
        # Release cortisol for stress response
        cortisol_amount = min(0.4, error_severity * importance * 0.3)
        self._adjust('cortisol', cortisol_amount)
        
        # If error is severe and non-recoverable, release more stress hormones
        if error_severity > 0.8 and not recoverable:
            additional_cortisol = min(0.2, error_severity * 0.2)
            self._adjust('cortisol', additional_cortisol)
        
        logging.info(f"[HormoneEngine] Error signal from {source}: cortisol +{cortisol_amount:.3f}")
    
    def _process_performance_data(self, content: dict, source: str, importance: float):
        """Process performance data and adjust hormone levels accordingly."""
        performance_score = content.get('performance_score', 0.5)
        response_time = content.get('response_time', 1.0)
        accuracy = content.get('accuracy', 0.5)
        
        # Good performance triggers reward hormones
        if performance_score > 0.7:
            dopamine_amount = min(0.15, (performance_score - 0.5) * importance * 0.2)
            self._adjust('dopamine', dopamine_amount)
        
        # Poor performance triggers stress response
        elif performance_score < 0.3:
            cortisol_amount = min(0.2, (0.5 - performance_score) * importance * 0.3)
            self._adjust('cortisol', cortisol_amount)
        
        # Fast response times trigger small dopamine release
        if response_time < 0.5:
            speed_bonus = min(0.05, (0.5 - response_time) * importance * 0.1)
            self._adjust('dopamine', speed_bonus)
        
        logging.info(f"[HormoneEngine] Performance data from {source}: score={performance_score:.3f}")
    
    def _process_user_feedback(self, content: dict, source: str, importance: float):
        """Process user feedback and release appropriate hormones."""
        satisfaction = content.get('satisfaction', 0.5)
        feedback_type = content.get('feedback_type', 'general')
        positive_feedback = content.get('positive_feedback', True)
        
        if positive_feedback and satisfaction > 0.6:
            # Positive feedback releases dopamine and serotonin
            dopamine_amount = min(0.25, satisfaction * importance * 0.3)
            serotonin_amount = min(0.15, satisfaction * importance * 0.2)
            
            self._adjust('dopamine', dopamine_amount)
            self._adjust('serotonin', serotonin_amount)
            
        elif not positive_feedback or satisfaction < 0.4:
            # Negative feedback releases cortisol
            cortisol_amount = min(0.3, (0.6 - satisfaction) * importance * 0.4)
            self._adjust('cortisol', cortisol_amount)
        
        logging.info(f"[HormoneEngine] User feedback from {source}: satisfaction={satisfaction:.3f}")
    
    def _process_task_completion(self, content: dict, source: str, importance: float):
        """Process task completion signals."""
        completion_rate = content.get('completion_rate', 0.5)
        task_difficulty = content.get('difficulty', 0.5)
        time_taken = content.get('time_taken', 1.0)
        
        # Task completion releases dopamine, more for difficult tasks
        base_dopamine = completion_rate * 0.1
        difficulty_bonus = task_difficulty * 0.1
        dopamine_amount = min(0.3, (base_dopamine + difficulty_bonus) * importance)
        
        self._adjust('dopamine', dopamine_amount)
        
        # Quick completion of difficult tasks gets extra reward
        if completion_rate > 0.8 and task_difficulty > 0.7 and time_taken < 1.0:
            bonus_dopamine = min(0.1, task_difficulty * importance * 0.1)
            self._adjust('dopamine', bonus_dopamine)
        
        logging.info(f"[HormoneEngine] Task completion from {source}: rate={completion_rate:.3f}")
    
    def _process_learning_progress(self, content: dict, source: str, importance: float):
        """Process learning progress signals."""
        learning_rate = content.get('learning_rate', 0.5)
        accuracy_improvement = content.get('accuracy_improvement', 0.0)
        knowledge_gained = content.get('knowledge_gained', 0.5)
        
        # Learning progress releases growth hormone and dopamine
        if learning_rate > 0.5:
            # Note: growth_hormone not in default levels, using serotonin as substitute
            serotonin_amount = min(0.2, learning_rate * importance * 0.25)
            self._adjust('serotonin', serotonin_amount)
        
        # Significant improvement releases dopamine
        if accuracy_improvement > 0.1:
            dopamine_amount = min(0.15, accuracy_improvement * importance * 1.5)
            self._adjust('dopamine', dopamine_amount)
        
        logging.info(f"[HormoneEngine] Learning progress from {source}: rate={learning_rate:.3f}")
    
    def _process_resource_status(self, content: dict, source: str, importance: float):
        """Process resource status updates."""
        memory_usage = content.get('memory_usage', 0.5)
        cpu_usage = content.get('cpu_usage', 0.5)
        storage_usage = content.get('storage_usage', 0.5)
        
        # High resource usage triggers stress response
        max_usage = max(memory_usage, cpu_usage, storage_usage)
        if max_usage > 0.8:
            cortisol_amount = min(0.25, (max_usage - 0.5) * importance * 0.5)
            self._adjust('cortisol', cortisol_amount)
        
        # Low resource usage allows for serotonin release (substitute for growth hormone)
        elif max_usage < 0.3:
            serotonin_amount = min(0.1, (0.5 - max_usage) * importance * 0.2)
            self._adjust('serotonin', serotonin_amount)
        
        logging.info(f"[HormoneEngine] Resource status from {source}: max_usage={max_usage:.3f}")
    
    def _process_general_signal(self, content: dict, source: str, importance: float):
        """Process general signals with default hormone response."""
        signal_strength = content.get('signal_strength', 0.5)
        signal_type = content.get('signal_type', 'neutral')
        
        # General positive signals release small amounts of serotonin
        if signal_type == 'positive' or signal_strength > 0.7:
            serotonin_amount = min(0.05, signal_strength * importance * 0.1)
            self._adjust('serotonin', serotonin_amount)
        
        # General negative signals release small amounts of cortisol
        elif signal_type == 'negative' or signal_strength < 0.3:
            cortisol_amount = min(0.05, (0.5 - signal_strength) * importance * 0.1)
            self._adjust('cortisol', cortisol_amount)
        
        logging.info(f"[HormoneEngine] General signal from {source}: type={signal_type}, strength={signal_strength:.3f}")

# Usage: Instantiate HormoneEngine with the event bus in the MCP system. 