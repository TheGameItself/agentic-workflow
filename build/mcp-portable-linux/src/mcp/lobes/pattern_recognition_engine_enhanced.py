"""
Enhanced Pattern Recognition Engine with Neural Column Architecture.

This module implements a comprehensive neural column-inspired pattern recognition system
with sensory input processing, pattern association learning, completion prediction,
and adaptive sensitivity based on feedback.
"""

import json
import sqlite3
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from datetime import datetime

# Import dependencies with fallbacks for testing
try:
    from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
    from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
    from src.mcp.lobes.experimental.vesicle_pool import VesiclePool
    from src.mcp.brain_state_aggregator import BrainStateAggregator
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # Create functional fallback classes for testing
    class WorkingMemory:
        """Fallback working memory implementation for pattern recognition."""
        def __init__(self, capacity: int = 100):
            self.capacity = capacity
            self.items = []
            self.access_count = {}
        
        def add(self, data: Any) -> None:
            """Add data to working memory with LRU eviction."""
            item_id = id(data)
            if len(self.items) >= self.capacity:
                # Remove least recently used item
                lru_item = min(self.items, key=lambda x: self.access_count.get(id(x), 0))
                self.items.remove(lru_item)
                self.access_count.pop(id(lru_item), None)
            
            self.items.append(data)
            self.access_count[item_id] = self.access_count.get(item_id, 0) + 1
        
        def get_all(self) -> List[Any]:
            """Get all items in working memory."""
            return self.items.copy()
    
    class LobeEventBus:
        """Fallback event bus implementation for cross-lobe communication."""
        def __init__(self):
            self.subscribers = {}
            self.event_history = []
        
        def subscribe(self, event: str, callback: Callable) -> None:
            """Subscribe to an event with callback."""
            if event not in self.subscribers:
                self.subscribers[event] = []
            self.subscribers[event].append(callback)
        
        def publish(self, event: str, data: Any) -> None:
            """Publish event to all subscribers."""
            self.event_history.append({'event': event, 'data': data, 'timestamp': time.time()})
            if len(self.event_history) > 1000:  # Keep history manageable
                self.event_history.pop(0)
            
            if event in self.subscribers:
                for callback in self.subscribers[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        logging.warning(f"Event callback failed for {event}: {e}")
    
    class VesiclePool:
        """Fallback vesicle pool implementation for neurotransmitter-like signaling."""
        def __init__(self):
            self.vesicles = {}
            self.release_history = []
        
        def get_state(self) -> Dict[str, Any]:
            """Get current vesicle pool state."""
            return {
                'vesicle_count': len(self.vesicles),
                'recent_releases': len([r for r in self.release_history if time.time() - r['timestamp'] < 60]),
                'total_releases': len(self.release_history)
            }
    
    class BrainStateAggregator:
        """Fallback brain state aggregator for system-wide monitoring."""
        def __init__(self):
            self.state_history = []
            self.current_state = {'timestamp': time.time(), 'active_lobes': []}
        
        def update_state(self, lobe_id: str, state: Dict[str, Any]) -> None:
            """Update state for a specific lobe."""
            self.current_state[lobe_id] = state
            self.current_state['timestamp'] = time.time()
            
        def get_current_state(self) -> Dict[str, Any]:
            """Get current aggregated brain state."""
            return self.current_state.copy()

# Stub for test compatibility
class AdaptiveSensitivityManager:
    pass

# Stub for test compatibility
class SensoryDataPropagator:
    pass

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None


class EnhancedNeuralColumn:
    """
    Enhanced Neural column implementation inspired by cortical columns.
    Each column specializes in processing specific pattern types with adaptive sensitivity,
    sensory input processing, pattern response generation, and completion prediction.
    """
    
    def __init__(self, column_id: str, pattern_types: List[str], position: Tuple[float, float, float] = (0, 0, 0)):
        self.column_id = column_id
        self.pattern_types = pattern_types
        self.position = position
        self.activation_state = 0.0
        self.sensitivity = 1.0
        self.learning_rate = 0.1
        self.pattern_associations = {}
        self.completion_predictions = {}
        self.feedback_history = []
        self.logger = logging.getLogger(f"EnhancedNeuralColumn_{column_id}")
        
        # Enhanced neural network alternative for pattern processing
        self.neural_processor = None
        self.algorithmic_processor = self._default_algorithmic_processor
        self.use_neural = False
        self.performance_metrics = {"neural": {}, "algorithmic": {}}
        
        # Enhanced sensory processing capabilities
        self.sensory_receptors = {}  # Different receptor types for different modalities
        self.response_patterns = {}  # Learned response patterns
        self.association_strengths = {}  # Track association strengths over time
        self.completion_accuracy = {}  # Track completion prediction accuracy
        
        # Adaptive learning parameters
        self.adaptation_rate = 0.05
        self.sensitivity_bounds = (0.1, 2.0)
        self.learning_rate_bounds = (0.01, 0.3)
        
        self.logger.info(f"Enhanced NeuralColumn {column_id} initialized with pattern types: {pattern_types}")
    
    def _default_algorithmic_processor(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Default algorithmic pattern processor."""
        confidence = pattern.get('confidence', 0.5)
        pattern_type = pattern.get('type', 'unknown')
        
        # Simple algorithmic processing
        if pattern_type in self.pattern_types:
            confidence *= self.sensitivity
            activation = min(1.0, confidence * 1.2)
        else:
            activation = confidence * 0.3
            
        return {
            'type': f"processed_{pattern_type}",
            'data': pattern.get('data'),
            'confidence': confidence,
            'activation': activation,
            'column_id': self.column_id,
            'processing_method': 'algorithmic'
        }
    
    def process_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Process pattern through neural column with implementation switching."""
        start_time = time.time()
        
        # Choose implementation based on performance
        if self.use_neural and self.neural_processor:
            try:
                result = self._neural_process(pattern)
                processing_time = time.time() - start_time
                self._update_performance_metrics('neural', processing_time, result)
                return result
            except Exception as e:
                self.logger.warning(f"Neural processing failed: {e}, falling back to algorithmic")
                self.use_neural = False
        
        # Algorithmic processing
        result = self.algorithmic_processor(pattern)
        processing_time = time.time() - start_time
        self._update_performance_metrics('algorithmic', processing_time, result)
        
        # Update activation state
        self.activation_state = result.get('activation', 0.0)
        
        return result
    
    def _neural_process(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Neural network pattern processing."""
        if not self.neural_processor:
            raise ValueError("Neural processor not initialized")
            
        # Simulate neural processing with enhanced capabilities
        confidence = pattern.get('confidence', 0.5)
        pattern_type = pattern.get('type', 'unknown')
        
        # Enhanced neural processing with learned associations
        if pattern_type in self.pattern_associations:
            association_boost = self.pattern_associations[pattern_type] * 0.2
            confidence += association_boost
            
        # Apply sensitivity adaptation
        confidence *= self.sensitivity
        activation = min(1.0, confidence * 1.3)  # Neural processing slightly better
        
        return {
            'type': f"neural_processed_{pattern_type}",
            'data': pattern.get('data'),
            'confidence': min(1.0, confidence),
            'activation': activation,
            'column_id': self.column_id,
            'processing_method': 'neural',
            'associations': self.pattern_associations.get(pattern_type, 0.0)
        }
    
    def process_sensory_input(self, sensory_data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through specialized receptors for this column."""
        # Initialize receptor for this modality if not exists
        if modality not in self.sensory_receptors:
            self.sensory_receptors[modality] = {
                'sensitivity': 1.0,
                'adaptation_rate': 0.05,
                'response_history': [],
                'pattern_count': 0
            }
        
        receptor = self.sensory_receptors[modality]
        
        # Extract patterns from sensory data
        if isinstance(sensory_data, str):
            patterns = [{"type": f"{modality}_text", "data": sensory_data, "confidence": 0.6}]
        elif isinstance(sensory_data, dict):
            patterns = [{"type": f"{modality}_structure", "data": sensory_data, "confidence": 0.7}]
        elif isinstance(sensory_data, list):
            patterns = [{"type": f"{modality}_sequence", "data": sensory_data, "confidence": 0.65}]
        else:
            patterns = [{"type": f"{modality}_raw", "data": str(sensory_data), "confidence": 0.5}]
        
        # Process patterns through column
        processed_patterns = []
        for pattern in patterns:
            # Apply receptor sensitivity
            pattern['confidence'] *= receptor['sensitivity']
            
            # Process through column
            result = self.process_pattern(pattern)
            processed_patterns.append(result)
            
            # Update receptor statistics
            receptor['pattern_count'] += 1
            receptor['response_history'].append({
                'confidence': result.get('confidence', 0.0),
                'activation': result.get('activation', 0.0),
                'timestamp': time.time()
            })
            
            # Keep history manageable
            if len(receptor['response_history']) > 50:
                receptor['response_history'].pop(0)
        
        return {
            'modality': modality,
            'column_id': self.column_id,
            'processed_patterns': processed_patterns,
            'receptor_sensitivity': receptor['sensitivity'],
            'pattern_count': receptor['pattern_count'],
            'timestamp': time.time()
        }
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate appropriate response to recognized pattern based on learned response patterns."""
        pattern_type = pattern.get('type', 'unknown')
        confidence = pattern.get('confidence', 0.5)
        
        # Check if we have learned response patterns for this type
        if pattern_type in self.response_patterns:
            response_template = self.response_patterns[pattern_type]
            response_confidence = confidence * response_template.get('effectiveness', 1.0)
            
            response = {
                'response_type': response_template.get('response_type', f"learned_response_to_{pattern_type}"),
                'response_data': response_template.get('response_data'),
                'confidence': min(1.0, response_confidence),
                'activation_level': self.activation_state,
                'generating_column': self.column_id,
                'context': context,
                'learned_response': True,
                'timestamp': time.time()
            }
        else:
            # Generate default response
            response = {
                'response_type': f"default_response_to_{pattern_type}",
                'response_data': f"Processed {pattern_type} with confidence {confidence:.2f}",
                'confidence': confidence * self.sensitivity,
                'activation_level': self.activation_state,
                'generating_column': self.column_id,
                'context': context,
                'learned_response': False,
                'timestamp': time.time()
            }
        
        # Store response pattern for learning
        if pattern_type not in self.response_patterns:
            self.response_patterns[pattern_type] = {
                'response_type': response['response_type'],
                'response_data': response['response_data'],
                'effectiveness': 1.0,
                'usage_count': 1
            }
        else:
            self.response_patterns[pattern_type]['usage_count'] += 1
        
        return response   
 
    def learn_pattern_association_with_strength(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any], 
                                              strength: float = 0.1, context: str = None):
        """Enhanced pattern association learning with strength tracking and context."""
        type1 = pattern1.get('type', 'unknown')
        type2 = pattern2.get('type', 'unknown')
        
        # Create association key
        association_key = f"{type1}_{type2}"
        
        # Initialize association strengths if not exists
        if association_key not in self.association_strengths:
            self.association_strengths[association_key] = {
                'strength': 0.0,
                'reinforcements': 0,
                'contexts': [],
                'last_updated': time.time()
            }
        
        # Update association strength
        association = self.association_strengths[association_key]
        association['strength'] += strength * self.learning_rate
        association['reinforcements'] += 1
        association['last_updated'] = time.time()
        
        # Add context if provided
        if context and context not in association['contexts']:
            association['contexts'].append(context)
            if len(association['contexts']) > 10:  # Keep context list manageable
                association['contexts'].pop(0)
        
        # Normalize strength to prevent unbounded growth
        association['strength'] = min(1.0, association['strength'])
        
        # Also update individual pattern associations
        if type1 not in self.pattern_associations:
            self.pattern_associations[type1] = 0.0
        if type2 not in self.pattern_associations:
            self.pattern_associations[type2] = 0.0
            
        self.pattern_associations[type1] += strength * self.learning_rate * 0.5
        self.pattern_associations[type2] += strength * self.learning_rate * 0.5
        
        # Normalize individual associations
        self.pattern_associations[type1] = min(1.0, self.pattern_associations[type1])
        self.pattern_associations[type2] = min(1.0, self.pattern_associations[type2])
        
        self.logger.info(f"Enhanced association learning: {type1} <-> {type2}, strength: {association['strength']:.3f}")
    
    def predict_pattern_completion_with_accuracy(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pattern completion prediction with accuracy tracking."""
        pattern_type = partial_pattern.get('type', 'unknown')
        
        # Check if we have completion accuracy data for this pattern type
        if pattern_type in self.completion_accuracy:
            accuracy_data = self.completion_accuracy[pattern_type]
            base_confidence = accuracy_data.get('average_accuracy', 0.5)
        else:
            base_confidence = 0.3
            self.completion_accuracy[pattern_type] = {
                'predictions': 0,
                'correct_predictions': 0,
                'average_accuracy': 0.3,
                'last_updated': time.time()
            }
        
        # Use association strength to improve prediction
        if pattern_type in self.pattern_associations:
            association_boost = self.pattern_associations[pattern_type] * 0.3
            completion_confidence = min(1.0, base_confidence + association_boost)
        else:
            completion_confidence = base_confidence
        
        # Generate completion prediction
        prediction = {
            'type': f"completed_{pattern_type}",
            'data': partial_pattern.get('data'),
            'confidence': completion_confidence,
            'completion_source': 'enhanced_association_learning',
            'column_id': self.column_id,
            'base_accuracy': base_confidence,
            'association_boost': self.pattern_associations.get(pattern_type, 0.0),
            'prediction_id': f"{pattern_type}_{int(time.time())}",
            'timestamp': time.time()
        }
        
        # Store prediction for validation
        self.completion_predictions[pattern_type] = prediction
        
        # Update prediction count
        self.completion_accuracy[pattern_type]['predictions'] += 1
        
        return prediction    

    def adapt_sensitivity_enhanced(self, feedback: Dict[str, Any]):
        """Enhanced sensitivity adaptation with multiple feedback dimensions and hormone integration."""
        performance = feedback.get('performance', 0.5)
        accuracy = feedback.get('accuracy', 0.5)
        response_time = feedback.get('response_time', 1.0)
        user_satisfaction = feedback.get('user_satisfaction', 0.5)
        hormone_levels = feedback.get('hormone_levels', {})
        
        # Calculate composite feedback score with hormone influence
        base_composite_score = (
            performance * 0.25 + 
            accuracy * 0.25 + 
            (1.0 / max(response_time, 0.1)) * 0.2 +  # Inverse of response time
            user_satisfaction * 0.2
        )
        
        # Apply hormone modulation to feedback processing
        hormone_modulation = self._calculate_hormone_modulation(hormone_levels)
        composite_score = base_composite_score * hormone_modulation
        
        # Adaptive sensitivity adjustment with hormone-based learning rate
        dopamine_level = hormone_levels.get('dopamine', 0.5)
        serotonin_level = hormone_levels.get('serotonin', 0.5)
        cortisol_level = hormone_levels.get('cortisol', 0.1)
        
        # Dopamine enhances learning from positive feedback
        if composite_score > 0.7 and dopamine_level > 0.6:
            sensitivity_boost = self.adaptation_rate * (1.0 + dopamine_level * 0.5)
            self.sensitivity = min(self.sensitivity_bounds[1], 
                                 self.sensitivity * (1.0 + sensitivity_boost))
        elif composite_score < 0.3 or cortisol_level > 0.7:
            # Cortisol triggers conservative adaptation
            sensitivity_reduction = self.adaptation_rate * (1.0 + cortisol_level * 0.3)
            self.sensitivity = max(self.sensitivity_bounds[0], 
                                 self.sensitivity * (1.0 - sensitivity_reduction))
        
        # Serotonin stabilizes learning rate adjustments
        serotonin_stability = max(0.5, serotonin_level)
        if accuracy > 0.8:
            learning_boost = 1.02 * serotonin_stability
            self.learning_rate = min(self.learning_rate_bounds[1], 
                                   self.learning_rate * learning_boost)
        elif accuracy < 0.4:
            learning_reduction = 0.98 / serotonin_stability
            self.learning_rate = max(self.learning_rate_bounds[0], 
                                   self.learning_rate * learning_reduction)
        
        # Advanced receptor sensitivity adaptation for each modality
        self._adapt_receptor_sensitivities(feedback, hormone_levels)
        
        # Store enhanced feedback with hormone context
        enhanced_feedback = feedback.copy()
        enhanced_feedback.update({
            'composite_score': composite_score,
            'base_composite_score': base_composite_score,
            'hormone_modulation': hormone_modulation,
            'sensitivity_after': self.sensitivity,
            'learning_rate_after': self.learning_rate,
            'hormone_levels': hormone_levels,
            'timestamp': time.time()
        })
        
        self.feedback_history.append(enhanced_feedback)
        if len(self.feedback_history) > 100:
            self.feedback_history.pop(0)
            
        self.logger.info(f"Enhanced sensitivity adaptation: composite_score={composite_score:.3f}, "
                        f"sensitivity={self.sensitivity:.3f}, learning_rate={self.learning_rate:.3f}, "
                        f"hormone_modulation={hormone_modulation:.3f}")
    
    def _calculate_hormone_modulation(self, hormone_levels: Dict[str, float]) -> float:
        """Calculate hormone-based modulation factor for feedback processing."""
        dopamine = hormone_levels.get('dopamine', 0.5)
        serotonin = hormone_levels.get('serotonin', 0.5)
        cortisol = hormone_levels.get('cortisol', 0.1)
        norepinephrine = hormone_levels.get('norepinephrine', 0.5)
        
        # Dopamine enhances positive feedback processing
        dopamine_factor = 0.8 + (dopamine * 0.4)
        
        # Serotonin provides stability and confidence
        serotonin_factor = 0.9 + (serotonin * 0.2)
        
        # Cortisol reduces sensitivity to feedback (stress response)
        cortisol_factor = 1.0 - (cortisol * 0.3)
        
        # Norepinephrine enhances attention to feedback
        attention_factor = 0.9 + (norepinephrine * 0.2)
        
        # Combine factors with biological weighting
        modulation = (
            dopamine_factor * 0.3 +
            serotonin_factor * 0.25 +
            cortisol_factor * 0.25 +
            attention_factor * 0.2
        )
        
        return max(0.3, min(1.7, modulation))  # Bounded modulation
    
    def _adapt_receptor_sensitivities(self, feedback: Dict[str, Any], hormone_levels: Dict[str, float]):
        """Adapt individual receptor sensitivities based on feedback and hormone levels."""
        modality_performance = feedback.get('modality_performance', {})
        
        for modality, receptor in self.sensory_receptors.items():
            if modality in modality_performance:
                modality_score = modality_performance[modality]
                
                # Calculate hormone-influenced adaptation rate
                base_adaptation = receptor['adaptation_rate']
                hormone_influence = self._calculate_hormone_modulation(hormone_levels)
                adapted_rate = base_adaptation * hormone_influence
                
                # Adapt receptor sensitivity
                if modality_score > 0.7:
                    receptor['sensitivity'] = min(2.0, 
                        receptor['sensitivity'] * (1.0 + adapted_rate))
                elif modality_score < 0.3:
                    receptor['sensitivity'] = max(0.1, 
                        receptor['sensitivity'] * (1.0 - adapted_rate))
                
                # Update adaptation rate based on learning success
                if modality_score > 0.8:
                    receptor['adaptation_rate'] = min(0.1, 
                        receptor['adaptation_rate'] * 1.01)
                elif modality_score < 0.2:
                    receptor['adaptation_rate'] = max(0.01, 
                        receptor['adaptation_rate'] * 0.99)
                
                self.logger.debug(f"Adapted {modality} receptor: sensitivity={receptor['sensitivity']:.3f}, "
                                f"adaptation_rate={receptor['adaptation_rate']:.3f}")
    
    def process_feedback_integration(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process comprehensive feedback integration with cross-modal learning."""
        feedback_type = feedback_data.get('type', 'general')
        source = feedback_data.get('source', 'unknown')
        
        # Process different types of feedback
        if feedback_type == 'pattern_completion_validation':
            self._process_completion_feedback(feedback_data)
        elif feedback_type == 'association_strength_feedback':
            self._process_association_feedback(feedback_data)
        elif feedback_type == 'cross_modal_feedback':
            self._process_cross_modal_feedback(feedback_data)
        elif feedback_type == 'hormone_triggered_feedback':
            self._process_hormone_feedback(feedback_data)
        
        # Apply sensitivity adaptation
        self.adapt_sensitivity_enhanced(feedback_data)
        
        # Generate feedback integration report
        integration_report = {
            'feedback_type': feedback_type,
            'source': source,
            'processing_timestamp': time.time(),
            'sensitivity_after': self.sensitivity,
            'learning_rate_after': self.learning_rate,
            'receptor_sensitivities': {
                modality: receptor['sensitivity'] 
                for modality, receptor in self.sensory_receptors.items()
            },
            'integration_success': True
        }
        
        return integration_report
    
    def _process_completion_feedback(self, feedback_data: Dict[str, Any]):
        """Process feedback about pattern completion predictions."""
        pattern_type = feedback_data.get('pattern_type', 'unknown')
        predicted_correctly = feedback_data.get('correct_prediction', False)
        actual_completion = feedback_data.get('actual_completion', {})
        
        if pattern_type in self.completion_accuracy:
            accuracy_data = self.completion_accuracy[pattern_type]
            
            if predicted_correctly:
                accuracy_data['correct_predictions'] += 1
            
            # Update average accuracy
            total_predictions = accuracy_data['predictions']
            if total_predictions > 0:
                accuracy_data['average_accuracy'] = (
                    accuracy_data['correct_predictions'] / total_predictions
                )
            
            accuracy_data['last_updated'] = time.time()
            
            self.logger.info(f"Updated completion accuracy for {pattern_type}: "
                           f"{accuracy_data['average_accuracy']:.3f}")
    
    def _process_association_feedback(self, feedback_data: Dict[str, Any]):
        """Process feedback about pattern associations."""
        association_key = feedback_data.get('association_key', '')
        association_helpful = feedback_data.get('helpful', True)
        strength_adjustment = feedback_data.get('strength_adjustment', 0.0)
        
        if association_key in self.association_strengths:
            association = self.association_strengths[association_key]
            
            if association_helpful:
                association['strength'] += abs(strength_adjustment) * 0.1
            else:
                association['strength'] -= abs(strength_adjustment) * 0.1
            
            # Normalize strength
            association['strength'] = max(0.0, min(1.0, association['strength']))
            association['last_updated'] = time.time()
            
            self.logger.info(f"Updated association {association_key}: "
                           f"strength={association['strength']:.3f}")
    
    def _process_cross_modal_feedback(self, feedback_data: Dict[str, Any]):
        """Process feedback about cross-modal pattern recognition."""
        primary_modality = feedback_data.get('primary_modality', 'visual')
        secondary_modality = feedback_data.get('secondary_modality', 'auditory')
        cross_modal_success = feedback_data.get('success', False)
        
        # Adjust cross-modal sensitivity based on feedback
        if cross_modal_success:
            # Increase sensitivity for both modalities
            for modality in [primary_modality, secondary_modality]:
                if modality in self.sensory_receptors:
                    receptor = self.sensory_receptors[modality]
                    receptor['sensitivity'] = min(2.0, receptor['sensitivity'] * 1.05)
        else:
            # Slightly decrease sensitivity to reduce false cross-modal associations
            for modality in [primary_modality, secondary_modality]:
                if modality in self.sensory_receptors:
                    receptor = self.sensory_receptors[modality]
                    receptor['sensitivity'] = max(0.1, receptor['sensitivity'] * 0.98)
        
        self.logger.info(f"Cross-modal feedback processed: {primary_modality} <-> {secondary_modality}, "
                        f"success={cross_modal_success}")
    
    def _process_hormone_feedback(self, feedback_data: Dict[str, Any]):
        """Process hormone-triggered feedback for dynamic sensitivity adjustment."""
        hormone_trigger = feedback_data.get('hormone_trigger', 'dopamine')
        trigger_strength = feedback_data.get('trigger_strength', 0.5)
        behavioral_change = feedback_data.get('behavioral_change', {})
        
        # Apply hormone-specific sensitivity adjustments
        if hormone_trigger == 'dopamine' and trigger_strength > 0.7:
            # High dopamine: increase overall sensitivity and learning rate
            self.sensitivity = min(self.sensitivity_bounds[1], 
                                 self.sensitivity * (1.0 + trigger_strength * 0.1))
            self.learning_rate = min(self.learning_rate_bounds[1], 
                                   self.learning_rate * (1.0 + trigger_strength * 0.05))
            
        elif hormone_trigger == 'cortisol' and trigger_strength > 0.6:
            # High cortisol: decrease sensitivity, increase caution
            self.sensitivity = max(self.sensitivity_bounds[0], 
                                 self.sensitivity * (1.0 - trigger_strength * 0.1))
            
        elif hormone_trigger == 'serotonin' and trigger_strength > 0.6:
            # High serotonin: stabilize learning parameters
            target_sensitivity = (self.sensitivity_bounds[0] + self.sensitivity_bounds[1]) / 2
            self.sensitivity = 0.9 * self.sensitivity + 0.1 * target_sensitivity
        
        self.logger.info(f"Hormone feedback processed: {hormone_trigger} "
                        f"(strength={trigger_strength:.3f})")
    
    def get_adaptive_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive adaptive performance metrics including feedback integration."""
        base_metrics = self.get_column_performance_summary()
        
        # Add feedback integration metrics
        recent_feedback = self.feedback_history[-10:] if self.feedback_history else []
        
        feedback_metrics = {
            'total_feedback_received': len(self.feedback_history),
            'recent_feedback_count': len(recent_feedback),
            'average_composite_score': (
                sum(f.get('composite_score', 0.5) for f in recent_feedback) / 
                len(recent_feedback)
            ) if recent_feedback else 0.5,
            'hormone_modulation_average': (
                sum(f.get('hormone_modulation', 1.0) for f in recent_feedback) / 
                len(recent_feedback)
            ) if recent_feedback else 1.0,
            'sensitivity_adaptation_range': {
                'min': min((f.get('sensitivity_after', self.sensitivity) for f in self.feedback_history), 
                          default=self.sensitivity),
                'max': max((f.get('sensitivity_after', self.sensitivity) for f in self.feedback_history), 
                          default=self.sensitivity),
                'current': self.sensitivity
            },
            'learning_rate_adaptation_range': {
                'min': min((f.get('learning_rate_after', self.learning_rate) for f in self.feedback_history), 
                          default=self.learning_rate),
                'max': max((f.get('learning_rate_after', self.learning_rate) for f in self.feedback_history), 
                          default=self.learning_rate),
                'current': self.learning_rate
            }
        }
        
        # Combine with base metrics
        base_metrics.update(feedback_metrics)
        return base_metrics
    
    def _update_performance_metrics(self, method: str, processing_time: float, result: Dict[str, Any]):
        """Update performance metrics for implementation comparison."""
        if method not in self.performance_metrics:
            self.performance_metrics[method] = {}
            
        metrics = self.performance_metrics[method]
        
        # Update metrics
        metrics['processing_time'] = processing_time
        metrics['confidence'] = result.get('confidence', 0.0)
        metrics['activation'] = result.get('activation', 0.0)
        metrics['last_updated'] = time.time()
        
        # Calculate running averages
        if 'avg_processing_time' not in metrics:
            metrics['avg_processing_time'] = processing_time
        else:
            metrics['avg_processing_time'] = 0.9 * metrics['avg_processing_time'] + 0.1 * processing_time
            
        if 'avg_confidence' not in metrics:
            metrics['avg_confidence'] = result.get('confidence', 0.0)
        else:
            metrics['avg_confidence'] = 0.9 * metrics['avg_confidence'] + 0.1 * result.get('confidence', 0.0)
    
    def should_switch_to_neural(self) -> bool:
        """Determine if should switch to neural implementation based on performance."""
        if not self.neural_processor:
            return False
            
        neural_metrics = self.performance_metrics.get('neural', {})
        algo_metrics = self.performance_metrics.get('algorithmic', {})
        
        if not neural_metrics or not algo_metrics:
            return False
            
        # Compare average performance
        neural_score = neural_metrics.get('avg_confidence', 0.0) / max(neural_metrics.get('avg_processing_time', 1.0), 0.001)
        algo_score = algo_metrics.get('avg_confidence', 0.0) / max(algo_metrics.get('avg_processing_time', 1.0), 0.001)
        
        return neural_score > algo_score * 1.1  # 10% improvement threshold
    
    def get_column_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for this column."""
        return {
            'column_id': self.column_id,
            'pattern_types': self.pattern_types,
            'position': self.position,
            'activation_state': self.activation_state,
            'sensitivity': self.sensitivity,
            'learning_rate': self.learning_rate,
            'use_neural': self.use_neural,
            'performance_metrics': self.performance_metrics,
            'sensory_receptors': {
                modality: {
                    'sensitivity': receptor['sensitivity'],
                    'pattern_count': receptor['pattern_count'],
                    'avg_confidence': sum(h['confidence'] for h in receptor['response_history'][-10:]) / min(len(receptor['response_history']), 10) if receptor['response_history'] else 0.0
                }
                for modality, receptor in self.sensory_receptors.items()
            },
            'association_count': len(self.pattern_associations),
            'response_patterns_count': len(self.response_patterns),
            'completion_accuracy': {
                pattern_type: data['average_accuracy']
                for pattern_type, data in self.completion_accuracy.items()
            },
            'feedback_count': len(self.feedback_history),
            'recent_feedback_score': (
                sum(f.get('composite_score', 0.5) for f in self.feedback_history[-5:]) / 
                min(len(self.feedback_history), 5)
            ) if self.feedback_history else 0.5
        }


class EnhancedPatternRecognitionEngine:
    """
    Enhanced Pattern Recognition Engine with Neural Column Architecture.
    
    Implements neural column-inspired processing with sensory input handling,
    pattern association learning, completion prediction, and adaptive sensitivity.
    Supports both neural network and algorithmic implementations with automatic switching.
    """
    
    def __init__(self, db_path: Optional[str] = None, event_bus: Optional[LobeEventBus] = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'pattern_recognition_enhanced.db')
        
        self.db_path = db_path
        self.neural_columns = {}  # Dictionary of EnhancedNeuralColumn objects
        self.column_states = {}  # Track column activation states
        self.pattern_confidence = {}  # Track pattern confidence scores
        self.working_memory = WorkingMemory()
        self.event_bus = event_bus or LobeEventBus()
        self.logger = logging.getLogger("EnhancedPatternRecognitionEngine")
        
        # Enhanced processing components
        self.sensory_data_sharing = {}  # Cross-lobe sensory data sharing
        self.pattern_associations = {}  # Global pattern associations
        self.completion_predictions = {}  # Global completion predictions
        
        # Cross-lobe communication and feedback integration
        self.cross_lobe_connections = {}  # Track connections to other lobes
        self.hormone_feedback_processor = None  # Will be initialized in setup
        self.adaptive_sensitivity_manager = None  # Will be initialized in setup
        self.sensory_data_propagator = None  # Will be initialized in setup
        
        self.logger.info("Enhanced PatternRecognitionEngine initialized with neural column architecture")
        self._init_database()
        self._initialize_enhanced_neural_columns()
        self._setup_cross_lobe_communication()
    
    def _init_database(self):
        """Initialize enhanced pattern recognition database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognized_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                frequency INTEGER DEFAULT 1,
                context TEXT,
                modality TEXT,
                column_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS neural_columns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_id TEXT UNIQUE NOT NULL,
                column_type TEXT NOT NULL,
                pattern_types TEXT,
                position TEXT,
                sensitivity REAL DEFAULT 1.0,
                learning_rate REAL DEFAULT 0.1,
                activation_state REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern1_type TEXT NOT NULL,
                pattern2_type TEXT NOT NULL,
                association_strength REAL DEFAULT 0.0,
                reinforcements INTEGER DEFAULT 0,
                context TEXT,
                column_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensory_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                modality TEXT NOT NULL,
                data_type TEXT NOT NULL,
                processed_data TEXT,
                confidence REAL DEFAULT 0.5,
                column_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _initialize_enhanced_neural_columns(self):
        """Initialize enhanced neural columns for common pattern types with specialized capabilities."""
        # Text processing column with enhanced sensory capabilities
        self.neural_columns['text_processor'] = EnhancedNeuralColumn(
            'text_processor',
            ['text', 'string', 'word', 'textual', 'linguistic'],
            position=(0, 0, 0)
        )
        
        # Visual processing column with spatial awareness
        self.neural_columns['visual_processor'] = EnhancedNeuralColumn(
            'visual_processor',
            ['visual', 'image', 'spatial', 'geometric', 'visual_pattern'],
            position=(1, 0, 0)
        )
        
        # Sequence processing column with temporal patterns
        self.neural_columns['sequence_processor'] = EnhancedNeuralColumn(
            'sequence_processor',
            ['list', 'sequence', 'array', 'temporal', 'ordered'],
            position=(0, 1, 0)
        )
        
        # Structure processing column with hierarchical patterns
        self.neural_columns['structure_processor'] = EnhancedNeuralColumn(
            'structure_processor',
            ['dict', 'object', 'structure', 'hierarchical', 'nested'],
            position=(1, 1, 0)
        )
        
        # Auditory processing column for sound patterns
        self.neural_columns['auditory_processor'] = EnhancedNeuralColumn(
            'auditory_processor',
            ['audio', 'sound', 'frequency', 'acoustic', 'auditory'],
            position=(0, 0, 1)
        )
        
        # Tactile processing column for touch patterns
        self.neural_columns['tactile_processor'] = EnhancedNeuralColumn(
            'tactile_processor',
            ['tactile', 'touch', 'pressure', 'texture', 'haptic'],
            position=(1, 0, 1)
        )
        
        # Multi-modal processing column for complex patterns
        self.neural_columns['multimodal_processor'] = EnhancedNeuralColumn(
            'multimodal_processor',
            ['multimodal', 'complex', 'combined', 'integrated', 'fusion'],
            position=(0.5, 0.5, 0.5)
        )
        
        self.logger.info(f"Initialized {len(self.neural_columns)} enhanced neural columns with specialized capabilities")  
  
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> EnhancedNeuralColumn:
        """Create a new enhanced neural column for specific pattern types."""
        column = EnhancedNeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, pattern_types, position, sensitivity, learning_rate, activation_state)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (column_id, 'enhanced', json.dumps(pattern_types), json.dumps(position), 
              column.sensitivity, column.learning_rate, column.activation_state))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created enhanced neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Enhanced sensory input processing through neural column architecture."""
        # Find the best column for this modality
        best_column = None
        best_match_score = 0.0
        
        for column in self.neural_columns.values():
            # Check if column handles this modality
            modality_match = any(modality.lower() in pattern_type.lower() for pattern_type in column.pattern_types)
            if modality_match:
                # Calculate match score based on column specialization
                match_score = column.sensitivity * (column.activation_state + 0.1)  # Add small base to avoid zero
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_column = column
        
        # Use multimodal processor if no specific column found
        if not best_column:
            best_column = self.neural_columns.get('multimodal_processor')
        
        # Process through selected column
        if best_column:
            result = best_column.process_sensory_input(data, modality)
            
            # Store in cross-lobe sensory data sharing
            if modality not in self.sensory_data_sharing:
                self.sensory_data_sharing[modality] = []
            
            self.sensory_data_sharing[modality].append({
                'data': data,
                'processed_patterns': result.get('processed_patterns', []),
                'column_id': best_column.column_id,
                'timestamp': time.time()
            })
            
            # Keep sensory data sharing manageable
            if len(self.sensory_data_sharing[modality]) > 50:
                self.sensory_data_sharing[modality].pop(0)
            
            # Update column states
            self.column_states[best_column.column_id] = {
                'last_activation': time.time(),
                'modality': modality,
                'pattern_count': result.get('pattern_count', 0)
            }
            
            # Store in database
            self._store_sensory_data(data, modality, result, best_column.column_id)
            
            self.logger.info(f"Processed {modality} sensory input through column {best_column.column_id}")
            return result
        else:
            # Fallback processing
            return {
                'modality': modality,
                'processed_patterns': [{'type': f"{modality}_fallback", 'data': data, 'confidence': 0.3}],
                'column_id': 'fallback',
                'timestamp': time.time()
            }
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern through appropriate neural column."""
        pattern_type = pattern.get('type', 'unknown')
        
        # Find best column for response generation
        best_column = None
        best_activation = 0.0
        
        for column in self.neural_columns.values():
            if any(pt in pattern_type for pt in column.pattern_types):
                if column.activation_state > best_activation:
                    best_activation = column.activation_state
                    best_column = column
        
        if best_column:
            return best_column.generate_pattern_response(pattern, context)
        else:
            # Default response
            return {
                'response_type': f"default_response_to_{pattern_type}",
                'response_data': f"Processed {pattern_type}",
                'confidence': pattern.get('confidence', 0.5) * 0.5,
                'generating_column': 'default',
                'context': context,
                'learned_response': False,
                'timestamp': time.time()
            }    

    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns across all relevant neural columns."""
        associations_learned = 0
        
        # Learn pairwise associations
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                pattern1 = patterns[i]
                pattern2 = patterns[j]
                
                # Find relevant neural columns
                for column in self.neural_columns.values():
                    if (any(pt in pattern1.get('type', '') for pt in column.pattern_types) or 
                        any(pt in pattern2.get('type', '') for pt in column.pattern_types)):
                        column.learn_pattern_association_with_strength(pattern1, pattern2, strength)
                        associations_learned += 1
                
                # Store global association
                self._store_pattern_association(pattern1, pattern2, strength)
        
        return {
            'associations_learned': associations_learned,
            'patterns_processed': len(patterns),
            'learning_strength': strength,
            'timestamp': time.time()
        }
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern using best available neural column."""
        pattern_type = partial_pattern.get('type', 'unknown')
        
        # Find best neural column for prediction
        best_prediction = None
        best_confidence = 0.0
        
        for column in self.neural_columns.values():
            if any(pt in pattern_type for pt in column.pattern_types):
                prediction = column.predict_pattern_completion_with_accuracy(partial_pattern)
                if prediction.get('confidence', 0.0) > best_confidence:
                    best_confidence = prediction.get('confidence', 0.0)
                    best_prediction = prediction
        
        if not best_prediction:
            # Default prediction
            best_prediction = {
                'type': f"completed_{pattern_type}",
                'data': partial_pattern.get('data'),
                'confidence': 0.2,
                'completion_source': 'default',
                'column_id': 'default',
                'timestamp': time.time()
            }
        
        return best_prediction
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity_enhanced(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = column.get_column_performance_summary()
        return states
    
    def get_cross_lobe_sensory_data(self, modality: str = None, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get cross-lobe sensory data for sharing between columns."""
        if modality:
            return {modality: self.sensory_data_sharing.get(modality, [])[-limit:]}
        else:
            return {
                mod: data[-limit:] for mod, data in self.sensory_data_sharing.items()
            }
    
    def batch_process_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple patterns in batch mode for efficiency."""
        results = []
        
        # Group patterns by type for efficient processing
        pattern_groups = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)
        
        # Process each group through appropriate columns
        for pattern_type, group_patterns in pattern_groups.items():
            # Find best column for this pattern type
            best_column = None
            for column in self.neural_columns.values():
                if any(pt in pattern_type for pt in column.pattern_types):
                    best_column = column
                    break
            
            if not best_column:
                # Use multimodal processor as fallback
                best_column = self.neural_columns.get('multimodal_processor')
            
            # Process patterns through column
            if best_column:
                for pattern in group_patterns:
                    result = best_column.process_pattern(pattern)
                    results.append(result)
            else:
                # Fallback processing
                for pattern in group_patterns:
                    results.append({
                        'type': f"fallback_{pattern.get('type', 'unknown')}",
                        'data': pattern.get('data'),
                        'confidence': pattern.get('confidence', 0.5) * 0.5,
                        'column_id': 'fallback'
                    })
        
        return results    
    
    def evaluate_pattern_recognition_accuracy(self, test_patterns: List[Dict[str, Any]], 
                                           ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate pattern recognition accuracy against ground truth."""
        if len(test_patterns) != len(ground_truth):
            self.logger.warning("Test patterns and ground truth must have the same length")
            return {"accuracy": 0.0}
        
        correct = 0
        confidence_sum = 0.0
        
        for test, truth in zip(test_patterns, ground_truth):
            # Process test pattern
            processed = None
            for column in self.neural_columns.values():
                if any(pt in test.get('type', '') for pt in column.pattern_types):
                    processed = column.process_pattern(test)
                    break
            
            if not processed:
                continue
            
            # Compare with ground truth (simplified comparison)
            if (processed.get('type', '').replace('processed_', '').replace('neural_processed_', '') == 
                truth.get('type', '') and processed.get('data') == truth.get('data')):
                correct += 1
            
            confidence_sum += processed.get('confidence', 0.0)
        
        accuracy = correct / len(test_patterns) if test_patterns else 0.0
        avg_confidence = confidence_sum / len(test_patterns) if test_patterns else 0.0
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "correct_count": correct,
            "total_count": len(test_patterns)
        }
    
    def evaluate_learning_effectiveness(self, training_patterns: List[Dict[str, Any]], 
                                     test_patterns: List[Dict[str, Any]],
                                     learning_iterations: int = 5) -> Dict[str, Any]:
        """Evaluate learning effectiveness by training on patterns and testing completion."""
        # Train associations
        for _ in range(learning_iterations):
            self.learn_pattern_associations(training_patterns)
        
        # Test pattern completion
        completion_results = []
        for pattern in test_patterns:
            # Create partial pattern (simulate incomplete data)
            partial_data = pattern.get('data')
            if isinstance(partial_data, str) and len(partial_data) > 2:
                partial_data = partial_data[:len(partial_data)//2]
            elif isinstance(partial_data, list) and len(partial_data) > 1:
                partial_data = partial_data[:len(partial_data)//2]
            
            partial_pattern = {
                'type': pattern.get('type'),
                'data': partial_data,
                'confidence': 0.5
            }
            
            # Predict completion
            completion = self.predict_pattern_completion(partial_pattern)
            completion_results.append(completion)
        
        # Calculate metrics
        confidence_sum = sum(result.get('confidence', 0.0) for result in completion_results)
        avg_confidence = confidence_sum / len(completion_results) if completion_results else 0.0
        
        # Count completions from association learning vs default
        association_completions = sum(1 for result in completion_results 
                                   if result.get('completion_source') == 'enhanced_association_learning')
        
        return {
            "average_confidence": avg_confidence,
            "association_completions": association_completions,
            "total_completions": len(completion_results),
            "association_ratio": association_completions / len(completion_results) if completion_results else 0.0
        }
    
    def _store_sensory_data(self, data: Any, modality: str, result: Dict[str, Any], column_id: str):
        """Store sensory data in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sensory_data (modality, data_type, processed_data, confidence, column_id)
            VALUES (?, ?, ?, ?, ?)
        """, (modality, type(data).__name__, json.dumps(result), 
              result.get('processed_patterns', [{}])[0].get('confidence', 0.5), column_id))
        
        conn.commit()
        conn.close()
    
    def _store_pattern_association(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any], strength: float):
        """Store pattern association in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        type1 = pattern1.get('type', 'unknown')
        type2 = pattern2.get('type', 'unknown')
        
        cursor.execute("""
            INSERT OR REPLACE INTO pattern_associations 
            (pattern1_type, pattern2_type, association_strength, reinforcements)
            VALUES (?, ?, ?, 1)
        """, (type1, type2, strength))
        
        conn.commit()
        conn.close()

    def get_cross_lobe_sharing_statistics(self) -> Dict[str, Any]:
        """Get statistics about cross-lobe sensory data sharing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get sharing activity statistics
        cursor.execute("""
            SELECT data_type, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM sensory_data 
            WHERE column_id = 'cross_lobe_sharing'
            GROUP BY data_type
        """)
        
        sharing_stats = {}
        for row in cursor.fetchall():
            sharing_stats[row[0]] = {
                'count': row[1],
                'average_confidence': row[2]
            }
        
        # Get propagation rule statistics
        propagation_stats = {}
        if self.sensory_data_propagator:
            for rule_id, rule in self.sensory_data_propagator.propagation_rules.items():
                propagation_stats[rule_id] = {
                    'usage_count': rule['usage_count'],
                    'success_rate': rule['success_rate'],
                    'priority': rule['priority']
                }
        
        # Get sensitivity management statistics
        sensitivity_stats = {}
        if self.adaptive_sensitivity_manager:
            sensitivity_stats = self.adaptive_sensitivity_manager.get_sensitivity_analytics()
        
        conn.close()
        
        return {
            'sharing_activity': sharing_stats,
            'propagation_rules': propagation_stats,
            'sensitivity_management': sensitivity_stats,
            'total_columns': len(self.neural_columns),
            'cross_lobe_connections': len(self.cross_lobe_connections)
        }


# Backward compatibility aliases
PatternRecognitionEngine = EnhancedPatternRecognitionEngine
NeuralColumn = EnhancedNeuralColumn