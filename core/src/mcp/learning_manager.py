#!/usr/bin/env python3
"""
Learning Manager for MCP Core System
Implements continuous learning and adaptation mechanisms.
"""

import logging
import json
import time
import threading
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

class LearningType(Enum):
    """Types of learning."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    META = "meta"
    EXPERIENTIAL = "experiential"

class LearningPhase(Enum):
    """Learning phases."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONSOLIDATION = "consolidation"
    GENERALIZATION = "generalization"

@dataclass
class LearningExperience:
    """Representation of a learning experience."""
    id: str
    experience_type: LearningType
    input_data: Any
    expected_output: Any = None
    actual_output: Any = None
    reward: float = 0.0
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningPattern:
    """Learned pattern or rule."""
    id: str
    pattern: str
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)

class LearningManager:
    """
    Learning Manager for MCP Core System.
    
    Implements various learning mechanisms to enable continuous
    improvement and adaptation of the system.
    
    Features:
    - Multiple learning types (supervised, unsupervised, reinforcement, etc.)
    - Experience replay and pattern recognition
    - Adaptive learning rates based on performance
    - Transfer learning between domains
    - Meta-learning for learning how to learn
    - Continuous model updating
    """
    
    def __init__(self):
        """Initialize learning manager."""
        self.logger = logging.getLogger(__name__)
        
        # Learning state
        self.current_phase = LearningPhase.EXPLORATION
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        
        # Experience storage
        self.experiences = deque(maxlen=10000)
        self.patterns = {}
        self.learned_rules = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.success_rates = defaultdict(list)
        
        # Learning statistics
        self.stats = {
            'total_experiences': 0,
            'patterns_learned': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'learning_sessions': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks for different learning events
        self.learning_callbacks = {
            'pattern_discovered': [],
            'performance_improved': [],
            'learning_phase_changed': []
        }
    
    def add_experience(self, experience_type: LearningType, input_data: Any,
                      expected_output: Any = None, actual_output: Any = None,
                      reward: float = 0.0, context: Optional[Dict[str, Any]] = None) -> str:
        """Add a learning experience."""
        with self.lock:
            experience_id = f"exp_{int(time.time())}_{len(self.experiences)}"
            
            experience = LearningExperience(
                id=experience_id,
                experience_type=experience_type,
                input_data=input_data,
                expected_output=expected_output,
                actual_output=actual_output,
                reward=reward,
                context=context or {}
            )
            
            self.experiences.append(experience)
            self.stats['total_experiences'] += 1
            
            # Trigger learning if enough experiences
            if len(self.experiences) % 100 == 0:
                self._trigger_learning_session()
            
            return experience_id
    
    def add_supervised_experience(self, input_data: Any, expected_output: Any,
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """Add a supervised learning experience."""
        return self.add_experience(
            LearningType.SUPERVISED,
            input_data,
            expected_output=expected_output,
            context=context
        )
    
    def add_reinforcement_experience(self, input_data: Any, action: Any, reward: float,
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """Add a reinforcement learning experience."""
        return self.add_experience(
            LearningType.REINFORCEMENT,
            input_data,
            actual_output=action,
            reward=reward,
            context=context
        )
    
    def learn_from_feedback(self, experience_id: str, feedback: Dict[str, Any]) -> bool:
        """Learn from feedback on a previous experience."""
        with self.lock:
            # Find the experience
            experience = None
            for exp in self.experiences:
                if exp.id == experience_id:
                    experience = exp
                    break
            
            if not experience:
                return False
            
            # Update experience with feedback
            if 'reward' in feedback:
                experience.reward = feedback['reward']
            
            if 'correct_output' in feedback:
                experience.expected_output = feedback['correct_output']
            
            # Calculate success
            success = feedback.get('success', experience.reward > 0)
            
            # Update statistics
            if success:
                self.stats['successful_predictions'] += 1
            else:
                self.stats['failed_predictions'] += 1
            
            # Update success rate for this type of experience
            self.success_rates[experience.experience_type].append(success)
            
            # Trigger pattern learning
            self._learn_patterns_from_experience(experience)
            
            return True
    
    def _learn_patterns_from_experience(self, experience: LearningExperience):
        """Learn patterns from a single experience."""
        # Simple pattern learning based on input-output relationships
        if experience.expected_output is not None:
            # Create a simple pattern representation
            pattern_key = f"{type(experience.input_data).__name__}_to_{type(experience.expected_output).__name__}"
            
            if pattern_key not in self.patterns:
                pattern_id = f"pattern_{len(self.patterns)}"
                self.patterns[pattern_key] = LearningPattern(
                    id=pattern_id,
                    pattern=pattern_key,
                    confidence=0.5
                )
                self.stats['patterns_learned'] += 1
                
                # Notify callbacks
                self._notify_callbacks('pattern_discovered', self.patterns[pattern_key])
            
            # Update pattern confidence based on success
            pattern = self.patterns[pattern_key]
            pattern.usage_count += 1
            
            if experience.reward > 0 or (experience.actual_output == experience.expected_output):
                pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1.0) / pattern.usage_count
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 0.0) / pattern.usage_count
            
            pattern.confidence = pattern.success_rate
            pattern.last_used = datetime.now()
    
    def _trigger_learning_session(self):
        """Trigger a learning session to update models and patterns."""
        with self.lock:
            self.stats['learning_sessions'] += 1
            
            # Analyze recent performance
            recent_experiences = list(self.experiences)[-100:]
            
            if recent_experiences:
                # Calculate recent success rate
                recent_successes = sum(1 for exp in recent_experiences if exp.reward > 0)
                recent_success_rate = recent_successes / len(recent_experiences)
                
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'success_rate': recent_success_rate,
                    'num_experiences': len(recent_experiences)
                })
                
                # Adapt learning parameters based on performance
                self._adapt_learning_parameters(recent_success_rate)
                
                # Update learning phase
                self._update_learning_phase(recent_success_rate)
    
    def _adapt_learning_parameters(self, success_rate: float):
        """Adapt learning parameters based on performance."""
        # Adjust learning rate
        if success_rate > 0.8:
            # High success rate - reduce learning rate for stability
            self.learning_rate *= 0.95
        elif success_rate < 0.4:
            # Low success rate - increase learning rate for faster adaptation
            self.learning_rate *= 1.05
        
        # Clamp learning rate
        self.learning_rate = max(0.01, min(0.5, self.learning_rate))
        
        # Adjust exploration rate
        if success_rate > 0.7:
            # High success rate - reduce exploration
            self.exploration_rate *= 0.98
        else:
            # Low success rate - increase exploration
            self.exploration_rate *= 1.02
        
        # Clamp exploration rate
        self.exploration_rate = max(0.1, min(0.5, self.exploration_rate))
    
    def _update_learning_phase(self, success_rate: float):
        """Update the current learning phase."""
        old_phase = self.current_phase
        
        if success_rate > 0.8:
            self.current_phase = LearningPhase.EXPLOITATION
        elif success_rate > 0.6:
            self.current_phase = LearningPhase.CONSOLIDATION
        elif success_rate > 0.4:
            self.current_phase = LearningPhase.GENERALIZATION
        else:
            self.current_phase = LearningPhase.EXPLORATION
        
        if old_phase != self.current_phase:
            self.logger.info(f"Learning phase changed from {old_phase.value} to {self.current_phase.value}")
            self._notify_callbacks('learning_phase_changed', {
                'old_phase': old_phase,
                'new_phase': self.current_phase,
                'success_rate': success_rate
            })
    
    def predict(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[Any, float]:
        """Make a prediction based on learned patterns."""
        with self.lock:
            # Simple prediction based on patterns
            input_type = type(input_data).__name__
            
            # Find matching patterns
            matching_patterns = []
            for pattern_key, pattern in self.patterns.items():
                if input_type in pattern_key:
                    matching_patterns.append(pattern)
            
            if not matching_patterns:
                # No patterns found - return default prediction
                return None, 0.0
            
            # Select best pattern based on confidence
            best_pattern = max(matching_patterns, key=lambda p: p.confidence)
            
            # Simple prediction (would be more sophisticated in practice)
            prediction = f"prediction_based_on_{best_pattern.pattern}"
            confidence = best_pattern.confidence
            
            return prediction, confidence
    
    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations for improving learning."""
        recommendations = []
        
        with self.lock:
            # Check success rates
            overall_success_rate = (
                self.stats['successful_predictions'] / 
                max(1, self.stats['successful_predictions'] + self.stats['failed_predictions'])
            )
            
            if overall_success_rate < 0.5:
                recommendations.append("Consider increasing exploration rate to discover new patterns")
                recommendations.append("Review recent experiences for common failure modes")
            
            if overall_success_rate > 0.9:
                recommendations.append("High success rate achieved - consider more challenging tasks")
                recommendations.append("Reduce exploration rate to exploit learned knowledge")
            
            # Check learning phase
            if self.current_phase == LearningPhase.EXPLORATION:
                recommendations.append("In exploration phase - focus on gathering diverse experiences")
            elif self.current_phase == LearningPhase.EXPLOITATION:
                recommendations.append("In exploitation phase - leverage learned patterns")
            
            # Check pattern diversity
            if len(self.patterns) < 5:
                recommendations.append("Limited patterns learned - increase experience diversity")
            
            # Check recent performance trend
            if len(self.performance_history) >= 2:
                recent_trend = self.performance_history[-1]['success_rate'] - self.performance_history[-2]['success_rate']
                if recent_trend < -0.1:
                    recommendations.append("Performance declining - review recent changes")
                elif recent_trend > 0.1:
                    recommendations.append("Performance improving - continue current approach")
        
        return recommendations
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for learning events."""
        if event_type in self.learning_callbacks:
            self.learning_callbacks[event_type].append(callback)
            return True
        return False
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """Notify registered callbacks of an event."""
        if event_type in self.learning_callbacks:
            for callback in self.learning_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in learning callback: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        with self.lock:
            # Calculate success rates by type
            type_success_rates = {}
            for exp_type, successes in self.success_rates.items():
                if successes:
                    type_success_rates[exp_type.value] = sum(successes) / len(successes)
            
            # Calculate overall success rate
            total_predictions = self.stats['successful_predictions'] + self.stats['failed_predictions']
            overall_success_rate = (
                self.stats['successful_predictions'] / max(1, total_predictions)
            )
            
            # Get pattern statistics
            pattern_stats = {}
            if self.patterns:
                pattern_confidences = [p.confidence for p in self.patterns.values()]
                pattern_stats = {
                    'total_patterns': len(self.patterns),
                    'average_confidence': sum(pattern_confidences) / len(pattern_confidences),
                    'max_confidence': max(pattern_confidences),
                    'min_confidence': min(pattern_confidences)
                }
            
            return {
                'basic_stats': self.stats,
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'current_phase': self.current_phase.value
                },
                'performance': {
                    'overall_success_rate': overall_success_rate,
                    'success_rates_by_type': type_success_rates,
                    'recent_performance_trend': len(self.performance_history)
                },
                'patterns': pattern_stats,
                'experience_buffer_size': len(self.experiences)
            }
    
    def export_learned_knowledge(self) -> Dict[str, Any]:
        """Export learned knowledge for persistence or transfer."""
        with self.lock:
            return {
                'patterns': {
                    pattern_key: {
                        'id': pattern.id,
                        'pattern': pattern.pattern,
                        'confidence': pattern.confidence,
                        'usage_count': pattern.usage_count,
                        'success_rate': pattern.success_rate,
                        'created_at': pattern.created_at.isoformat(),
                        'last_used': pattern.last_used.isoformat() if pattern.last_used else None
                    }
                    for pattern_key, pattern in self.patterns.items()
                },
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'current_phase': self.current_phase.value
                },
                'statistics': self.stats,
                'export_timestamp': datetime.now().isoformat()
            }
    
    def import_learned_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """Import previously learned knowledge."""
        try:
            with self.lock:
                # Import patterns
                if 'patterns' in knowledge:
                    for pattern_key, pattern_data in knowledge['patterns'].items():
                        pattern = LearningPattern(
                            id=pattern_data['id'],
                            pattern=pattern_data['pattern'],
                            confidence=pattern_data['confidence'],
                            usage_count=pattern_data['usage_count'],
                            success_rate=pattern_data['success_rate'],
                            created_at=datetime.fromisoformat(pattern_data['created_at'])
                        )
                        
                        if pattern_data['last_used']:
                            pattern.last_used = datetime.fromisoformat(pattern_data['last_used'])
                        
                        self.patterns[pattern_key] = pattern
                
                # Import learning parameters
                if 'learning_parameters' in knowledge:
                    params = knowledge['learning_parameters']
                    self.learning_rate = params.get('learning_rate', self.learning_rate)
                    self.exploration_rate = params.get('exploration_rate', self.exploration_rate)
                    
                    phase_value = params.get('current_phase')
                    if phase_value:
                        for phase in LearningPhase:
                            if phase.value == phase_value:
                                self.current_phase = phase
                                break
                
                # Import statistics
                if 'statistics' in knowledge:
                    self.stats.update(knowledge['statistics'])
                
                self.logger.info("Successfully imported learned knowledge")
                return True
                
        except Exception as e:
            self.logger.error(f"Error importing learned knowledge: {e}")
            return False

# Global learning manager instance
_learning_manager: Optional[LearningManager] = None

def get_learning_manager() -> LearningManager:
    """Get the global learning manager instance."""
    global _learning_manager
    if _learning_manager is None:
        _learning_manager = LearningManager()
    return _learning_manager