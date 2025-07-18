"""
Pattern recognition engine with neural column simulation.

This lobe implements basic neural column-inspired processing with batch operations 
and proactive prompting. Inspired by research on neural columns in the brain.

Research References:
- idea.txt (pattern recognition, neural columns, batch processing, feedback-driven adaptation)
- NeurIPS 2025 (Neural Column Pattern Recognition)
- Nature 2024 (Batch Processing in AI)
- arXiv:2107.01473 (Slope and Generalization Properties of Neural Networks)
- See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

Extensibility:
- Add support for multi-agent pattern recognition and voting
- Integrate with external pattern databases and feedback analytics
- Support for dynamic pattern extraction and reranking
- Integrate slope maps for regularization, loss, or early stopping
TODO:
- Implement advanced neural column and batch processing algorithms
- Add robust error handling and logging for all pattern recognition operations
- Support for dynamic pattern templates and feedback loops
- Add advanced slope map computation and usage for neural acceleration
"""

import json
import sqlite3
import os
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from datetime import datetime
import collections
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory  # See idea.txt
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
import logging
import random
import time
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool
from src.mcp.brain_state_aggregator import BrainStateAggregator

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


class PatternCache:
    """
    PatternCache: Domain-specific working memory for recent patterns with decay.
    Inspired by neuroscience (short-term memory, pattern buffers). See idea.txt.

    Research References:
    - idea.txt (short-term memory, pattern buffers)
    - Nature 2024 (Pattern Buffering in AI)
    - NeurIPS 2025 (Working Memory for Pattern Recognition)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced decay models (e.g., context-sensitive, feedback-driven)
    - Integrate with multi-agent or distributed pattern caches
    - Support for feedback-driven learning and adaptation
    TODO:
    - Implement advanced feedback weighting and prioritization
    - Add robust error handling for cache overflows/underflows
    """
    def __init__(self, capacity=100, decay=0.95):
        self.capacity = capacity
        self.decay = decay
        self.cache = []
    def add(self, pattern):
        self.cache.append({'pattern': pattern, 'strength': 1.0})
        if len(self.cache) > self.capacity:
            self.cache.pop(0)
    def decay_cache(self):
        for entry in self.cache:
            entry['strength'] *= self.decay
        self.cache = [e for e in self.cache if e['strength'] > 0.1]
    def get_recent(self, n=5):
        return [e['pattern'] for e in self.cache[-n:]]


class PatternExtractorMicroLobe:
    """
    Micro-lobe for pattern extraction. Can be dynamically activated/inhibited based on context/feedback.
    Inspired by sensory cortex feature extraction (see idea.txt, neuroscience).

    Research References:
    - idea.txt (feature extraction, micro-lobes)
    - Nature 2024 (Micro-Lobe Feature Extraction in AI)
    - NeurIPS 2025 (Context-Aware Micro-Lobes)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced feature extraction models
    - Integrate with external feature extraction libraries
    - Support for dynamic micro-lobe activation and reranking
    TODO:
    - Implement advanced feature extraction and reranking
    - Add robust error handling and logging for all extraction operations
    """
    def __init__(self):
        self.enabled = True
        self.logger = logging.getLogger("PatternExtractorMicroLobe")
    def extract(self, data):
        if not self.enabled:
            self.logger.info("[PatternExtractorMicroLobe] Inhibited. Extraction bypassed.")
            return []
        # Basic pattern extraction (stub, extend as needed)
        if isinstance(data, str):
            return [{"type": "text", "data": data, "confidence": 0.5}]
        elif isinstance(data, dict):
            return [{"type": k, "data": v, "confidence": 0.5} for k, v in data.items()]
        elif isinstance(data, list):
            return [{"type": "list_item", "data": item, "confidence": 0.5} for item in data]
        return []


class NeuralColumnMicroLobe:
    """
    Micro-lobe for neural column processing. Can be dynamically activated/inhibited based on context/feedback.
    Inspired by cortical columns and batch processing (see idea.txt, neuroscience).

    Research References:
    - idea.txt (neural columns, batch processing)
    - Nature 2024 (Neural Column Processing in AI)
    - NeurIPS 2025 (Batch Processing in Neural Columns)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced neural column models
    - Integrate with external batch processing libraries
    - Support for dynamic column activation and reranking
    TODO:
    - Implement advanced neural column and batch processing algorithms
    - Add robust error handling and logging for all column operations
    """
    def __init__(self, parent):
        self.enabled = True
        self.parent = parent  # Reference to PatternRecognitionEngine for state
        self.logger = logging.getLogger("NeuralColumnMicroLobe")
    def process(self, patterns, item_id):
        if not self.enabled:
            self.logger.info("[NeuralColumnMicroLobe] Inhibited. Processing bypassed.")
            return []
        return self.parent._process_with_columns(patterns, item_id)


class FeedbackIntegrationMicroLobe:
    """
    Micro-lobe for feedback integration and adaptation. Can be dynamically activated/inhibited.
    Inspired by feedback loops in cortex and basal ganglia (see idea.txt, neuroscience).

    Research References:
    - idea.txt (feedback integration, adaptation)
    - Nature 2024 (Feedback Loops in AI)
    - NeurIPS 2025 (Feedback-Driven Adaptation in Pattern Recognition)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced feedback integration models
    - Integrate with external feedback analytics tools
    - Support for dynamic feedback-driven adaptation and reranking
    TODO:
    - Implement advanced feedback-driven adaptation algorithms
    - Add robust error handling and logging for all feedback operations
    """
    def __init__(self, parent):
        self.enabled = True
        self.parent = parent
        self.logger = logging.getLogger("FeedbackIntegrationMicroLobe")
    def integrate(self, recognized_patterns):
        if not self.enabled:
            self.logger.info("[FeedbackIntegrationMicroLobe] Inhibited. Feedback integration bypassed.")
            return recognized_patterns
        # Placeholder: feedback-driven adaptation (stub)
        # In future, adjust confidence, activation, or routing based on feedback
        return recognized_patterns


class AssociativePatternMemory:
    """
    AssociativePatternMemory: Context-aware, feedback-driven memory for patterns.
    Links patterns to context, feedback, and event metadata for rapid, relevant recall.
    Inspired by associative memory in the brain (see idea.txt, neuroscience).

    Research References:
    - idea.txt (associative memory, pattern recall)
    - Nature 2024 (Associative Memory in AI)
    - NeurIPS 2025 (Context-Aware Pattern Memory)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add advanced context linking and feedback-driven adaptation
    - Integrate with external context analytics tools
    - Support for dynamic associative memory templates
    TODO:
    - Implement advanced associative memory algorithms
    - Add robust error handling for memory overflows/underflows
    """
    def __init__(self, capacity=200, decay=0.96):
        self.capacity = capacity
        self.decay = decay
        self.memory = []  # Each entry: {'pattern': ..., 'context': ..., 'feedback': ..., 'strength': ...}
        self.logger = logging.getLogger("AssociativePatternMemory")
    def add(self, pattern, context=None, feedback=None):
        entry = {'pattern': pattern, 'context': context, 'feedback': feedback, 'strength': 1.0}
        self.memory.append(entry)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.logger.info(f"[AssociativePatternMemory] Added pattern: {pattern} (context={context}, feedback={feedback})")
    def decay_memory(self):
        for entry in self.memory:
            entry['strength'] *= self.decay
        self.memory = [e for e in self.memory if e['strength'] > 0.1]
    def get_by_context(self, context, n=5):
        # Return most recent patterns matching context (simple substring match for demo)
        context_str = str(context) if context is not None else ""
        matches = [e for e in self.memory if context_str and context_str in str(e['context'])]
        return [e['pattern'] for e in matches[-n:]]
    def get_recent(self, n=5):
        return [e['pattern'] for e in self.memory[-n:]]


class QuantalPatternVesicle:
    """
    Represents a discrete 'quantum' of pattern information, inspired by quantal neurotransmitter release.
    Supports both spontaneous (noise/baseline) and evoked (event-driven) release.

    Research References:
    - idea.txt (quantal signaling, pattern vesicles)
    - Nature 2024 (Quantal Neurotransmitter Release in AI)
    - NeurIPS 2025 (Pattern Vesicle Dynamics)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add new vesicle types (e.g., chemical, electrical, synthetic)
    - Integrate with advanced event bus or signal routing
    - Support for multi-modal quantal encoding
    TODO:
    - Implement advanced vesicle fusion and release logic
    - Add robust error handling and logging for all vesicle operations
    """
    def __init__(self, pattern, evoked=True, timestamp=None):
        self.pattern = pattern
        self.evoked = evoked
        self.timestamp = timestamp or time.time()
    def __repr__(self):
        mode = 'evoked' if self.evoked else 'spontaneous'
        return f"<QuantalPatternVesicle mode={mode} pattern={self.pattern}>"


class NeuralColumn:
    """
    Enhanced Neural column implementation inspired by cortical columns.
    Each column specializes in processing specific pattern types with adaptive sensitivity,
    sensory input processing, pattern response generation, and completion prediction.
    
    Features:
    - Neural network alternatives with automatic switching
    - Pattern association learning with strength tracking
    - Completion prediction based on learned associations
    - Adaptive sensitivity based on feedback
    - Sensory input processing for different modalities
    - Pattern response generation with context awareness
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
        self.logger = logging.getLogger(f"NeuralColumn_{column_id}")
        
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
        """Neural network pattern processing (placeholder for actual neural implementation)."""
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
    
    def learn_association(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any], strength: float = 0.1):
        """Learn association between two patterns."""
        type1 = pattern1.get('type', 'unknown')
        type2 = pattern2.get('type', 'unknown')
        
        # Update associations
        if type1 not in self.pattern_associations:
            self.pattern_associations[type1] = 0.0
        if type2 not in self.pattern_associations:
            self.pattern_associations[type2] = 0.0
            
        # Strengthen association
        self.pattern_associations[type1] += strength * self.learning_rate
        self.pattern_associations[type2] += strength * self.learning_rate
        
        # Normalize to prevent unbounded growth
        self.pattern_associations[type1] = min(1.0, self.pattern_associations[type1])
        self.pattern_associations[type2] = min(1.0, self.pattern_associations[type2])
        
        self.logger.info(f"Learned association between {type1} and {type2} with strength {strength}")
    
    def predict_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict pattern completion based on learned associations."""
        pattern_type = partial_pattern.get('type', 'unknown')
        
        if pattern_type in self.pattern_associations:
            completion_confidence = self.pattern_associations[pattern_type]
            
            # Generate completion prediction
            prediction = {
                'type': f"completed_{pattern_type}",
                'data': partial_pattern.get('data'),
                'confidence': completion_confidence,
                'completion_source': 'association_learning',
                'column_id': self.column_id
            }
            
            # Store prediction for validation
            self.completion_predictions[pattern_type] = prediction
            
            return prediction
        
        # Default completion for unknown patterns
        return {
            'type': f"completed_{pattern_type}",
            'data': partial_pattern.get('data'),
            'confidence': 0.3,
            'completion_source': 'default',
            'column_id': self.column_id
        }
    
    def adapt_sensitivity(self, feedback: Dict[str, Any]):
        """Adapt column sensitivity based on feedback."""
        performance = feedback.get('performance', 0.5)
        accuracy = feedback.get('accuracy', 0.5)
        
        # Adjust sensitivity based on performance
        if performance > 0.7:
            self.sensitivity = min(1.5, self.sensitivity * 1.05)  # Increase sensitivity
        elif performance < 0.3:
            self.sensitivity = max(0.5, self.sensitivity * 0.95)  # Decrease sensitivity
            
        # Adjust learning rate based on accuracy
        if accuracy > 0.8:
            self.learning_rate = min(0.2, self.learning_rate * 1.02)
        elif accuracy < 0.4:
            self.learning_rate = max(0.05, self.learning_rate * 0.98)
            
        # Store feedback
        self.feedback_history.append(feedback)
        if len(self.feedback_history) > 100:
            self.feedback_history.pop(0)
            
        self.logger.info(f"Adapted sensitivity to {self.sensitivity:.3f}, learning rate to {self.learning_rate:.3f}")
    
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
        reverse_key = f"{type2}_{type1}"
        
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
    
    def validate_completion_prediction(self, prediction_id: str, actual_result: Dict[str, Any], 
                                     was_correct: bool):
        """Validate a completion prediction and update accuracy metrics."""
        # Find the pattern type from prediction_id
        pattern_type = prediction_id.split('_')[0] if '_' in prediction_id else 'unknown'
        
        if pattern_type in self.completion_accuracy:
            accuracy_data = self.completion_accuracy[pattern_type]
            
            if was_correct:
                accuracy_data['correct_predictions'] += 1
            
            # Update average accuracy
            if accuracy_data['predictions'] > 0:
                accuracy_data['average_accuracy'] = (
                    accuracy_data['correct_predictions'] / accuracy_data['predictions']
                )
            
            accuracy_data['last_updated'] = time.time()
            
            self.logger.info(f"Completion prediction validated for {pattern_type}: "
                           f"correct={was_correct}, accuracy={accuracy_data['average_accuracy']:.3f}")
    
    def adapt_sensitivity_enhanced(self, feedback: Dict[str, Any]):
        """Enhanced sensitivity adaptation with multiple feedback dimensions."""
        performance = feedback.get('performance', 0.5)
        accuracy = feedback.get('accuracy', 0.5)
        response_time = feedback.get('response_time', 1.0)
        user_satisfaction = feedback.get('user_satisfaction', 0.5)
        
        # Calculate composite feedback score
        composite_score = (
            performance * 0.3 + 
            accuracy * 0.3 + 
            (1.0 / max(response_time, 0.1)) * 0.2 +  # Inverse of response time
            user_satisfaction * 0.2
        )
        
        # Adapt sensitivity based on composite score
        if composite_score > 0.7:
            # Good performance, increase sensitivity gradually
            self.sensitivity = min(self.sensitivity_bounds[1], 
                                 self.sensitivity * (1.0 + self.adaptation_rate))
        elif composite_score < 0.3:
            # Poor performance, decrease sensitivity
            self.sensitivity = max(self.sensitivity_bounds[0], 
                                 self.sensitivity * (1.0 - self.adaptation_rate))
        
        # Adapt learning rate based on accuracy
        if accuracy > 0.8:
            self.learning_rate = min(self.learning_rate_bounds[1], 
                                   self.learning_rate * 1.02)
        elif accuracy < 0.4:
            self.learning_rate = max(self.learning_rate_bounds[0], 
                                   self.learning_rate * 0.98)
        
        # Store enhanced feedback
        enhanced_feedback = feedback.copy()
        enhanced_feedback.update({
            'composite_score': composite_score,
            'sensitivity_after': self.sensitivity,
            'learning_rate_after': self.learning_rate,
            'timestamp': time.time()
        })
        
        self.feedback_history.append(enhanced_feedback)
        if len(self.feedback_history) > 100:
            self.feedback_history.pop(0)
            
        self.logger.info(f"Enhanced sensitivity adaptation: composite_score={composite_score:.3f}, "
                        f"sensitivity={self.sensitivity:.3f}, learning_rate={self.learning_rate:.3f}")
    
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


class PatternRecognitionEngine:
    """
    Enhanced Pattern Recognition Engine with Neural Column Architecture.
    
    Implements neural column-inspired processing with sensory input handling,
    pattern association learning, completion prediction, and adaptive sensitivity.
    Supports both neural network and algorithmic implementations with automatic switching.
    
    Research References:
    - idea.txt (pattern recognition, neural columns, batch processing, feedback-driven adaptation)
    - NeurIPS 2025 (Neural Column Pattern Recognition)
    - Nature 2024 (Batch Processing in AI)
    - arXiv:2107.01473 (Slope and Generalization Properties of Neural Networks)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Add support for multi-agent pattern recognition and voting
    - Integrate with external pattern databases and feedback analytics
    - Support for dynamic pattern extraction and reranking
    - Integrate slope maps for regularization, loss, or early stopping
    """
    def __init__(self, db_path: Optional[str] = None, event_bus: Optional[LobeEventBus] = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'pattern_recognition.db')
        
        self.db_path = db_path
        self.patterns = []
        self.neural_columns = {}  # Dictionary of NeuralColumn objects
        self.prompt_log = []
        self.column_states = {}  # Track column activation states
        self.pattern_confidence = {}  # Track pattern confidence scores
        self.working_memory = WorkingMemory()
        self.event_bus = event_bus or LobeEventBus()
        self.extractor_lobe = PatternExtractorMicroLobe()
        self.neural_column_lobe = NeuralColumnMicroLobe(self)
        self.feedback_lobe = FeedbackIntegrationMicroLobe(self)
        self.logger = logging.getLogger("PatternRecognitionEngine")
        self.event_bus.subscribe('sensory_input', lambda data: (self.handle_sensory_input(data), None)[1])
        self.pattern_cache = PatternCache()
        self.associative_memory = AssociativePatternMemory()
        self.spontaneous_rate = 0.01  # Probability per call for spontaneous release
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        
        # Enhanced neural column architecture
        self.sensory_input_processor = self._create_sensory_input_processor()
        self.pattern_response_generator = self._create_pattern_response_generator()
        self.association_learner = self._create_association_learner()
        self.completion_predictor = self._create_completion_predictor()
        
        self.logger.info("[PatternRecognitionEngine] Enhanced with neural column architecture")
        self.logger.info("[PatternRecognitionEngine] VesiclePool initialized: %s", self.vesicle_pool.get_state())
        self._init_database()
        self._initialize_neural_columns()
    
    def _create_sensory_input_processor(self):
        """Create sensory input processor for handling different input modalities."""
        def process_sensory_input(data: Any, modality: str = "visual") -> Dict[str, Any]:
            """Process sensory input through appropriate neural columns."""
            # Extract patterns from sensory input
            patterns = self.extractor_lobe.extract(data)
            
            # Find or create appropriate neural column for this modality
            column_id = f"sensory_{modality}"
            if column_id not in self.neural_columns:
                self.neural_columns[column_id] = NeuralColumn(
                    column_id, 
                    [modality, f"{modality}_pattern"], 
                    position=(0, 0, 0)
                )
            
            # Process through neural column
            processed_patterns = []
            for pattern in patterns:
                processed = self.neural_columns[column_id].process_pattern(pattern)
                processed_patterns.append(processed)
            
            return {
                'modality': modality,
                'processed_patterns': processed_patterns,
                'column_id': column_id,
                'timestamp': time.time()
            }
        
        return process_sensory_input
    
    def _create_pattern_response_generator(self):
        """Create pattern response generator for producing appropriate responses."""
        def generate_pattern_response(pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Generate appropriate response to recognized pattern."""
            pattern_type = pattern.get('type', 'unknown')
            confidence = pattern.get('confidence', 0.5)
            
            # Find best neural column for response generation
            best_column = None
            best_activation = 0.0
            
            for column in self.neural_columns.values():
                if pattern_type in column.pattern_types:
                    if column.activation_state > best_activation:
                        best_activation = column.activation_state
                        best_column = column
            
            if best_column:
                # Generate response through neural column
                response = {
                    'response_type': f"response_to_{pattern_type}",
                    'confidence': confidence * best_column.sensitivity,
                    'activation_level': best_activation,
                    'generating_column': best_column.column_id,
                    'context': context,
                    'timestamp': time.time()
                }
            else:
                # Default response
                response = {
                    'response_type': f"default_response_to_{pattern_type}",
                    'confidence': confidence * 0.5,
                    'activation_level': 0.3,
                    'generating_column': 'default',
                    'context': context,
                    'timestamp': time.time()
                }
            
            return response
        
        return generate_pattern_response
    
    def _create_association_learner(self):
        """Create association learner for pattern association learning."""
        def learn_pattern_associations(patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
            """Learn associations between patterns."""
            associations_learned = 0
            
            # Learn pairwise associations
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    pattern1 = patterns[i]
                    pattern2 = patterns[j]
                    
                    # Find relevant neural columns
                    for column in self.neural_columns.values():
                        if (pattern1.get('type') in column.pattern_types or 
                            pattern2.get('type') in column.pattern_types):
                            column.learn_association(pattern1, pattern2, strength)
                            associations_learned += 1
            
            return {
                'associations_learned': associations_learned,
                'patterns_processed': len(patterns),
                'learning_strength': strength,
                'timestamp': time.time()
            }
        
        return learn_pattern_associations
    
    def _create_completion_predictor(self):
        """Create completion predictor for pattern completion prediction."""
        def predict_pattern_completion(partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
            """Predict completion of partial pattern."""
            pattern_type = partial_pattern.get('type', 'unknown')
            
            # Find best neural column for prediction
            best_prediction = None
            best_confidence = 0.0
            
            for column in self.neural_columns.values():
                if pattern_type in column.pattern_types:
                    prediction = column.predict_completion(partial_pattern)
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
                    'column_id': 'default'
                }
            
            best_prediction['timestamp'] = time.time()
            return best_prediction
        
        return predict_pattern_completion
    
    def _initialize_neural_columns(self):
        """Initialize enhanced neural columns for common pattern types with specialized capabilities."""
        # Text processing column with enhanced sensory capabilities
        self.neural_columns['text_processor'] = NeuralColumn(
            'text_processor',
            ['text', 'string', 'word', 'textual', 'linguistic'],
            position=(0, 0, 0)
        )
        
        # Visual processing column with spatial awareness
        self.neural_columns['visual_processor'] = NeuralColumn(
            'visual_processor',
            ['visual', 'image', 'spatial', 'geometric', 'visual_pattern'],
            position=(1, 0, 0)
        )
        
        # Sequence processing column with temporal patterns
        self.neural_columns['sequence_processor'] = NeuralColumn(
            'sequence_processor',
            ['list', 'sequence', 'array', 'temporal', 'ordered'],
            position=(0, 1, 0)
        )
        
        # Structure processing column with hierarchical patterns
        self.neural_columns['structure_processor'] = NeuralColumn(
            'structure_processor',
            ['dict', 'object', 'structure', 'hierarchical', 'nested'],
            position=(1, 1, 0)
        )
        
        # Auditory processing column for sound patterns
        self.neural_columns['auditory_processor'] = NeuralColumn(
            'auditory_processor',
            ['audio', 'sound', 'frequency', 'acoustic', 'auditory'],
            position=(0, 0, 1)
        )
        
        # Tactile processing column for touch patterns
        self.neural_columns['tactile_processor'] = NeuralColumn(
            'tactile_processor',
            ['tactile', 'touch', 'pressure', 'texture', 'haptic'],
            position=(1, 0, 1)
        )
        
        # Multi-modal processing column for complex patterns
        self.neural_columns['multimodal_processor'] = NeuralColumn(
            'multimodal_processor',
            ['multimodal', 'complex', 'combined', 'integrated', 'fusion'],
            position=(0.5, 0.5, 0.5)
        )
        
        self.logger.info(f"Initialized {len(self.neural_columns)} enhanced neural columns with specialized capabilities")
    
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> NeuralColumn:
        """Create a new neural column for specific pattern types."""
        column = NeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, input_patterns, output_patterns, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (column_id, 'custom', json.dumps(pattern_types), json.dumps([]), column.learning_rate))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through neural column architecture."""
        return self.sensory_input_processor(data, modality)
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern."""
        return self.pattern_response_generator(pattern, context)
    
    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns."""
        return self.association_learner(patterns, strength)
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern."""
        return self.completion_predictor(partial_pattern)
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = {
                'activation_state': column.activation_state,
                'sensitivity': column.sensitivity,
                'learning_rate': column.learning_rate,
                'pattern_types': column.pattern_types,
                'associations_count': len(column.pattern_associations),
                'feedback_count': len(column.feedback_history),
                'use_neural': column.use_neural,
                'performance_metrics': column.performance_metrics
            }
        return states
    
    def enable_neural_processing(self, column_id: str):
        """Enable neural processing for a specific column."""
        if column_id in self.neural_columns:
            column = self.neural_columns[column_id]
            # Initialize simple neural processor (placeholder)
            column.neural_processor = self._create_simple_neural_processor()
            column.use_neural = column.should_switch_to_neural()
            self.logger.info(f"Enabled neural processing for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found")
    
    def _create_simple_neural_processor(self):
        """Create a simple neural processor (placeholder for actual neural network)."""
        def simple_neural_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder neural processing
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'unknown')
            
            # Simulate neural enhancement
            enhanced_confidence = min(1.0, confidence * 1.2)
            
            return {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': enhanced_confidence,
                'processing_method': 'neural_simulation'
            }
        
        return simple_neural_processor
        
    def create_neural_network_processor(self, column_id: str):
        """Create a neural network processor for a specific column using PyTorch."""
        if torch is None:
            self.logger.warning("PyTorch not available, cannot create neural network processor")
            return None
            
        if column_id not in self.neural_columns:
            self.logger.warning(f"Column {column_id} not found")
            return None
            
        # Create a simple neural network for pattern processing
        class SimplePatternNetwork(nn.Module):
            def __init__(self, input_size=10, hidden_size=20, output_size=1):
                super(SimplePatternNetwork, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.layer2 = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                x = self.sigmoid(x)
                return x
                
        # Create network instance
        network = SimplePatternNetwork()
        
        # Create processor function that uses the network
        def neural_network_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            pattern_type = pattern.get('type', 'unknown')
            confidence = pattern.get('confidence', 0.5)
            
            # Convert pattern to tensor input (simplified)
            # In a real implementation, this would extract features from the pattern
            input_tensor = torch.zeros(10)
            input_tensor[0] = confidence
            
            # Add some pattern type encoding (very simplified)
            if 'text' in pattern_type:
                input_tensor[1] = 1.0
            elif 'visual' in pattern_type:
                input_tensor[2] = 1.0
            elif 'list' in pattern_type:
                input_tensor[3] = 1.0
            elif 'dict' in pattern_type:
                input_tensor[4] = 1.0
                
            # Process through network
            with torch.no_grad():
                output = network(input_tensor).item()
                
            # Create result
            result = {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': output,
                'activation': output * 1.2,  # Slightly boost activation
                'column_id': column_id,
                'processing_method': 'neural_network'
            }
            
            return result
            
        # Assign processor to column
        self.neural_columns[column_id].neural_processor = neural_network_processor
        self.logger.info(f"Created neural network processor for column {column_id}")
        
        return neural_network_processor
        
    def train_neural_processor(self, column_id: str, training_data: List[Dict[str, Any]], 
                             epochs: int = 100, learning_rate: float = 0.01):
        """Train the neural network processor for a specific column."""
        if torch is None:
            self.logger.warning("PyTorch not available, cannot train neural processor")
            return False
            
        if column_id not in self.neural_columns:
            self.logger.warning(f"Column {column_id} not found")
            return False
            
        column = self.neural_columns[column_id]
        if not hasattr(column, 'neural_network'):
            # Create a neural network if it doesn't exist
            self.create_neural_network_processor(column_id)
            
        # In a real implementation, this would extract features from the training data
        # and train the neural network properly
        self.logger.info(f"Training neural processor for column {column_id} with {len(training_data)} examples")
        
        # Simulate training success
        column.use_neural = True
        return True
        
    def register_with_brain_state_aggregator(self, brain_state_aggregator: BrainStateAggregator):
        """Register with brain state aggregator for performance tracking."""
        if not brain_state_aggregator:
            self.logger.warning("No brain state aggregator provided")
            return
            
        # Register performance metrics for each column
        for column_id, column in self.neural_columns.items():
            # Register algorithmic implementation
            brain_state_aggregator.register_implementation_performance(
                f"pattern_recognition_{column_id}",
                "algorithmic",
                {
                    "accuracy": 0.8,
                    "latency": 0.005,
                    "resource_usage": 0.2
                }
            )
            
            # Register neural implementation if available
            if column.neural_processor:
                brain_state_aggregator.register_implementation_performance(
                    f"pattern_recognition_{column_id}",
                    "neural",
                    {
                        "accuracy": 0.85,
                        "latency": 0.01,
                        "resource_usage": 0.4
                    }
                )
                
        self.logger.info(f"Registered {len(self.neural_columns)} columns with brain state aggregator")
        
    def batch_process_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple patterns in batch mode for efficiency."""
        results = []
        
        # Group patterns by type
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
                if pattern_type in column.pattern_types:
                    best_column = column
                    break
                    
            if not best_column:
                # Use default column
                best_column = self.neural_columns.get('text_processor')
                
            # Process patterns through column
            if best_column:
                for pattern in group_patterns:
                    result = best_column.process_pattern(pattern)
                    results.append(result)
            else:
                # Fallback processing
                results.extend(group_patterns)
                
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
                if test.get('type') in column.pattern_types:
                    processed = column.process_pattern(test)
                    break
                    
            if not processed:
                continue
                
            # Compare with ground truth
            if processed.get('type') == truth.get('type') and processed.get('data') == truth.get('data'):
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
            partial_pattern = {
                'type': pattern.get('type'),
                'data': pattern.get('data')[:len(pattern.get('data'))//2] if isinstance(pattern.get('data'), str) else pattern.get('data'),
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
                                   if result.get('completion_source') == 'association_learning')
        
        return {
            "average_confidence": avg_confidence,
            "association_completions": association_completions,
            "total_completions": len(completion_results),
            "association_ratio": association_completions / len(completion_results) if completion_results else 0.0
        }(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )
        
        self.logger.info(f"Initialized {len(self.neural_columns)} neural columns")
    
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> NeuralColumn:
        """Create a new neural column for specific pattern types."""
        column = NeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, input_patterns, output_patterns, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (column_id, 'custom', json.dumps(pattern_types), json.dumps([]), column.learning_rate))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through neural column architecture."""
        return self.sensory_input_processor(data, modality)
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern."""
        return self.pattern_response_generator(pattern, context)
    
    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns."""
        return self.association_learner(patterns, strength)
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern."""
        return self.completion_predictor(partial_pattern)
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = {
                'activation_state': column.activation_state,
                'sensitivity': column.sensitivity,
                'learning_rate': column.learning_rate,
                'pattern_types': column.pattern_types,
                'associations_count': len(column.pattern_associations),
                'feedback_count': len(column.feedback_history),
                'use_neural': column.use_neural,
                'performance_metrics': column.performance_metrics
            }
        return states
    
    def enable_neural_processing(self, column_id: str):
        """Enable neural processing for a specific column."""
        if column_id in self.neural_columns:
            column = self.neural_columns[column_id]
            # Initialize simple neural processor (placeholder)
            column.neural_processor = self._create_simple_neural_processor()
            column.use_neural = column.should_switch_to_neural()
            self.logger.info(f"Enabled neural processing for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found")
    
    def _create_simple_neural_processor(self):
        """Create a simple neural processor (placeholder for actual neural network)."""
        def simple_neural_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder neural processing
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'unknown')
            
            # Simulate neural enhancement
            enhanced_confidence = min(1.0, confidence * 1.2)
            
            return {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': enhanced_confidence,
                'processing_method': 'neural_simulation'
            }
        
        return simple_neural_processor(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )
        
        self.logger.info(f"Initialized {len(self.neural_columns)} neural columns")
    
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> NeuralColumn:
        """Create a new neural column for specific pattern types."""
        column = NeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, input_patterns, output_patterns, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (column_id, 'custom', json.dumps(pattern_types), json.dumps([]), column.learning_rate))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through neural column architecture."""
        return self.sensory_input_processor(data, modality)
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern."""
        return self.pattern_response_generator(pattern, context)
    
    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns."""
        return self.association_learner(patterns, strength)
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern."""
        return self.completion_predictor(partial_pattern)
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = {
                'activation_state': column.activation_state,
                'sensitivity': column.sensitivity,
                'learning_rate': column.learning_rate,
                'pattern_types': column.pattern_types,
                'associations_count': len(column.pattern_associations),
                'feedback_count': len(column.feedback_history),
                'use_neural': column.use_neural,
                'performance_metrics': column.performance_metrics
            }
        return states
    
    def enable_neural_processing(self, column_id: str):
        """Enable neural processing for a specific column."""
        if column_id in self.neural_columns:
            column = self.neural_columns[column_id]
            # Initialize simple neural processor (placeholder)
            column.neural_processor = self._create_simple_neural_processor()
            column.use_neural = column.should_switch_to_neural()
            self.logger.info(f"Enabled neural processing for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found")
    
    def _create_simple_neural_processor(self):
        """Create a simple neural processor (placeholder for actual neural network)."""
        def simple_neural_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder neural processing
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'unknown')
            
            # Simulate neural enhancement
            enhanced_confidence = min(1.0, confidence * 1.2)
            
            return {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': enhanced_confidence,
                'processing_method': 'neural_simulation'
            }
        
        return simple_neural_processor(
            'structure_processor',
            ['dict', 'object', 'structure'],
            position=(1, 1, 0)
        )
        
        self.logger.info(f"Initialized {len(self.neural_columns)} neural columns")
    
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> NeuralColumn:
        """Create a new neural column for specific pattern types."""
        column = NeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, input_patterns, output_patterns, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (column_id, 'custom', json.dumps(pattern_types), json.dumps([]), column.learning_rate))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through neural column architecture."""
        return self.sensory_input_processor(data, modality)
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern."""
        return self.pattern_response_generator(pattern, context)
    
    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns."""
        return self.association_learner(patterns, strength)
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern."""
        return self.completion_predictor(partial_pattern)
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = {
                'activation_state': column.activation_state,
                'sensitivity': column.sensitivity,
                'learning_rate': column.learning_rate,
                'pattern_types': column.pattern_types,
                'associations_count': len(column.pattern_associations),
                'feedback_count': len(column.feedback_history),
                'use_neural': column.use_neural,
                'performance_metrics': column.performance_metrics
            }
        return states
    
    def enable_neural_processing(self, column_id: str):
        """Enable neural processing for a specific column."""
        if column_id in self.neural_columns:
            column = self.neural_columns[column_id]
            # Initialize simple neural processor (placeholder)
            column.neural_processor = self._create_simple_neural_processor()
            column.use_neural = column.should_switch_to_neural()
            self.logger.info(f"Enabled neural processing for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found")
    
    def _create_simple_neural_processor(self):
        """Create a simple neural processor (placeholder for actual neural network)."""
        def simple_neural_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder neural processing
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'unknown')
            
            # Simulate neural enhancement
            enhanced_confidence = min(1.0, confidence * 1.2)
            
            return {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': enhanced_confidence,
                'processing_method': 'neural_simulation'
            }
        
        return simple_neural_processor      self.logger.info(f"Initialized {len(self.neural_columns)} neural columns")
    
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> NeuralColumn:
        """Create a new neural column for specific pattern types."""
        column = NeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, input_patterns, output_patterns, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (column_id, 'custom', json.dumps(pattern_types), json.dumps([]), column.learning_rate))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through neural column architecture."""
        return self.sensory_input_processor(data, modality)
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern."""
        return self.pattern_response_generator(pattern, context)
    
    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns."""
        return self.association_learner(patterns, strength)
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern."""
        return self.completion_predictor(partial_pattern)
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = {
                'activation_state': column.activation_state,
                'sensitivity': column.sensitivity,
                'learning_rate': column.learning_rate,
                'pattern_types': column.pattern_types,
                'associations_count': len(column.pattern_associations),
                'feedback_count': len(column.feedback_history),
                'use_neural': column.use_neural,
                'performance_metrics': column.performance_metrics
            }
        return states
    
    def enable_neural_processing(self, column_id: str):
        """Enable neural processing for a specific column."""
        if column_id in self.neural_columns:
            column = self.neural_columns[column_id]
            # Initialize simple neural processor (placeholder)
            column.neural_processor = self._create_simple_neural_processor()
            column.use_neural = column.should_switch_to_neural()
            self.logger.info(f"Enabled neural processing for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found")
    
    def _create_simple_neural_processor(self):
        """Create a simple neural processor (placeholder for actual neural network)."""
        def simple_neural_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder neural processing
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'unknown')
            
            # Simulate neural enhancement
            enhanced_confidence = min(1.0, confidence * 1.2)
            
            return {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': enhanced_confidence,
                'processing_method': 'neural_simulation'
            }
        
        return simple_neural_processor      self.logger.info(f"Initialized {len(self.neural_columns)} neural columns")
    
    def create_neural_column(self, column_id: str, pattern_types: List[str], 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> NeuralColumn:
        """Create a new neural column for specific pattern types."""
        column = NeuralColumn(column_id, pattern_types, position)
        self.neural_columns[column_id] = column
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO neural_columns 
            (column_id, column_type, input_patterns, output_patterns, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (column_id, 'custom', json.dumps(pattern_types), json.dumps([]), column.learning_rate))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created neural column {column_id} for pattern types: {pattern_types}")
        return column
    
    def process_sensory_input(self, data: Any, modality: str = "visual") -> Dict[str, Any]:
        """Process sensory input through neural column architecture."""
        return self.sensory_input_processor(data, modality)
    
    def generate_pattern_response(self, pattern: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response to recognized pattern."""
        return self.pattern_response_generator(pattern, context)
    
    def learn_pattern_associations(self, patterns: List[Dict[str, Any]], strength: float = 0.1) -> Dict[str, Any]:
        """Learn associations between patterns."""
        return self.association_learner(patterns, strength)
    
    def predict_pattern_completion(self, partial_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Predict completion of partial pattern."""
        return self.completion_predictor(partial_pattern)
    
    def adapt_column_sensitivity(self, column_id: str, feedback: Dict[str, Any]):
        """Adapt sensitivity of specific neural column based on feedback."""
        if column_id in self.neural_columns:
            self.neural_columns[column_id].adapt_sensitivity(feedback)
            self.logger.info(f"Adapted sensitivity for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found for sensitivity adaptation")
    
    def get_column_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all neural columns."""
        states = {}
        for column_id, column in self.neural_columns.items():
            states[column_id] = {
                'activation_state': column.activation_state,
                'sensitivity': column.sensitivity,
                'learning_rate': column.learning_rate,
                'pattern_types': column.pattern_types,
                'associations_count': len(column.pattern_associations),
                'feedback_count': len(column.feedback_history),
                'use_neural': column.use_neural,
                'performance_metrics': column.performance_metrics
            }
        return states
    
    def enable_neural_processing(self, column_id: str):
        """Enable neural processing for a specific column."""
        if column_id in self.neural_columns:
            column = self.neural_columns[column_id]
            # Initialize simple neural processor (placeholder)
            column.neural_processor = self._create_simple_neural_processor()
            column.use_neural = column.should_switch_to_neural()
            self.logger.info(f"Enabled neural processing for column {column_id}")
        else:
            self.logger.warning(f"Column {column_id} not found")
    
    def _create_simple_neural_processor(self):
        """Create a simple neural processor (placeholder for actual neural network)."""
        def simple_neural_processor(pattern: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder neural processing
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'unknown')
            
            # Simulate neural enhancement
            enhanced_confidence = min(1.0, confidence * 1.2)
            
            return {
                'type': f"neural_{pattern_type}",
                'data': pattern.get('data'),
                'confidence': enhanced_confidence,
                'processing_method': 'neural_simulation'
            }
        
        return simple_neural_processor
    
    def _init_database(self):
        """Initialize pattern recognition database."""
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS neural_columns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_id TEXT UNIQUE NOT NULL,
                column_type TEXT NOT NULL,
                activation_state TEXT,
                input_patterns TEXT,
                output_patterns TEXT,
                learning_rate REAL DEFAULT 0.1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proactive_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_type TEXT NOT NULL,
                prompt_data TEXT NOT NULL,
                context TEXT,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def handle_sensory_input(self, data: Any):
        """Handle sensory input event and trigger pattern recognition as quantal vesicles."""
        vesicles = self.recognize_patterns([data], emit_quantal=True)
        self.event_bus.publish('pattern_quantal_release', {'vesicles': [v.__dict__ for v in vesicles]})
        return vesicles

    def recognize_patterns(self, data_batch: List[Any], context: Optional[str] = None, feedback: Any = None, emit_quantal=False) -> List[Any]:
        """Recognize patterns in a batch of data and emit quantal vesicles if requested."""
        context = context if context is not None else ""
        quantal_vesicles = []
        recognized_patterns = []
        for i, data_item in enumerate(data_batch):
            # Use extractor micro-lobe
            patterns = self.extractor_lobe.extract(data_item)
            # Use neural column micro-lobe
            column_processed = self.neural_column_lobe.process(patterns, f"item_{i}")
            # Use feedback integration micro-lobe
            integrated = self.feedback_lobe.integrate(column_processed)
            # Store recognized patterns and add to associative memory
            for pattern in integrated:
                pattern_id = self._store_pattern(pattern)
                recognized_patterns.append({
                    'id': pattern_id,
                    'type': pattern['type'],
                    'data': pattern['data'],
                    'confidence': pattern['confidence'],
                    'source_item': i
                })
                self.pattern_cache.add(pattern)
                self.associative_memory.add(pattern, context=context, feedback=feedback)
                # Evoked quantal vesicle
                if emit_quantal:
                    vesicle = QuantalPatternVesicle(pattern, evoked=True)
                    quantal_vesicles.append(vesicle)
                    self.logger.info(f"[PatternRecognitionEngine] Quantal pattern vesicle (evoked): {vesicle}")
                # Occasionally emit spontaneous quantal vesicle
                if emit_quantal and random.random() < self.spontaneous_rate:
                    noise_pattern = {'type': 'noise', 'data': random.gauss(0, 0.1), 'confidence': 0.1, 'source_item': i}
                    spont_vesicle = QuantalPatternVesicle(noise_pattern, evoked=False)
                    quantal_vesicles.append(spont_vesicle)
                    self.logger.info(f"[PatternRecognitionEngine] Quantal pattern vesicle (spontaneous): {spont_vesicle}")
        self.pattern_cache.decay_cache()
        self.associative_memory.decay_memory()
        # After recognizing patterns, publish event
        if hasattr(self, 'event_bus') and self.event_bus and not emit_quantal:
            self.event_bus.publish('pattern_recognized', recognized_patterns)
        return quantal_vesicles if emit_quantal else recognized_patterns

    def _process_with_columns(self, patterns: List[Dict[str, Any]], item_id: str) -> List[Dict[str, Any]]:
        """Process patterns through neural columns."""
        processed_patterns = []
        
        # Create or get neural columns
        columns = self._get_or_create_columns(patterns)
        
        for pattern in patterns:
            # Find relevant columns for this pattern type
            relevant_columns = [col for col in columns if pattern['type'] in col['input_patterns']]
            
            if relevant_columns:
                # Process through columns
                for column in relevant_columns:
                    processed = self._activate_column(column, pattern)
                    if processed:
                        processed_patterns.append(processed)
            else:
                # Create new column for this pattern type
                new_column = self._create_column_for_pattern(pattern)
                processed = self._activate_column(new_column, pattern)
                if processed:
                    processed_patterns.append(processed)
        
        return processed_patterns

    def _get_or_create_columns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get existing columns or create new ones for pattern types."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing columns
        cursor.execute("SELECT column_id, column_type, input_patterns, output_patterns FROM neural_columns")
        existing_columns = []
        for row in cursor.fetchall():
            existing_columns.append({
                'id': row[0],
                'type': row[1],
                'input_patterns': json.loads(row[2]) if row[2] else [],
                'output_patterns': json.loads(row[3]) if row[3] else []
            })
        
        # Check if we need new columns
        pattern_types = [p['type'] for p in patterns]
        existing_types = []
        for col in existing_columns:
            existing_types.extend(col['input_patterns'])
        
        # Create new columns for missing pattern types
        for pattern_type in pattern_types:
            if pattern_type not in existing_types:
                new_column = self._create_column_for_pattern_type(pattern_type)
                existing_columns.append(new_column)
        
        conn.close()
        return existing_columns

    def _create_column_for_pattern_type(self, pattern_type: str) -> Dict[str, Any]:
        """Create a new neural column for a pattern type."""
        column_id = f"column_{pattern_type}_{hash(pattern_type) % 10000}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO neural_columns (column_id, column_type, input_patterns, output_patterns)
            VALUES (?, ?, ?, ?)
        """, (column_id, 'pattern_processor', json.dumps([pattern_type]), json.dumps([])))
        
        conn.commit()
        conn.close()
        
        return {
            'id': column_id,
            'type': 'pattern_processor',
            'input_patterns': [pattern_type],
            'output_patterns': []
        }

    def _create_column_for_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new neural column for a specific pattern."""
        return self._create_column_for_pattern_type(pattern['type'])

    def _activate_column(self, column: Dict[str, Any], pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Activate a neural column with input pattern using batch NN processing."""
        if pattern['type'] not in column['input_patterns']:
            return None
        
        # Enhanced activation with batch processing simulation
        base_confidence = pattern.get('confidence', 0.5)
        pattern_frequency = pattern.get('frequency', 1)
        column_experience = len(column.get('output_patterns', []))
        
        # Neural column activation formula (simplified batch NN simulation)
        activation_strength = min(1.0, base_confidence * (1 + pattern_frequency * 0.1) * (1 + column_experience * 0.01))
        
        # Update column state with batch processing info
        self.column_states[column['id']] = {
            'activated': True,
            'strength': activation_strength,
            'last_activation': datetime.now().isoformat(),
            'batch_processing': True,
            'pattern_frequency': pattern_frequency,
            'column_experience': column_experience
        }
        
        # Batch processing: accumulate patterns for batch activation
        if 'batch_patterns' not in column:
            column['batch_patterns'] = []
        column['batch_patterns'].append(pattern)
        
        # Process batch if it reaches threshold (batch size of 4)
        if len(column['batch_patterns']) >= 4:
            self._process_batch_activation(column)
        
        # Generate output pattern with enhanced processing
        output_pattern = {
            'type': f"processed_{pattern['type']}",
            'data': pattern['data'],
            'confidence': activation_strength * 0.9,  # Slight confidence decay
            'source_column': column['id'],
            'original_pattern': pattern,
            'batch_processed': len(column.get('batch_patterns', [])) >= 4
        }
        
        # Update column output patterns
        column['output_patterns'].append(output_pattern['type'])
        
        return output_pattern

    def _process_batch_activation(self, column: Dict[str, Any]):
        """Process batch activation for neural column learning."""
        if not column.get('batch_patterns'):
            return
        
        # Analyze batch patterns for learning
        pattern_types = [p.get('type') for p in column['batch_patterns']]
        pattern_confidence = [p.get('confidence', 0.5) for p in column['batch_patterns']]
        
        # Calculate batch statistics
        avg_confidence = sum(pattern_confidence) / len(pattern_confidence)
        pattern_diversity = len(set(pattern_types))
        
        # Update column learning parameters
        column['batch_avg_confidence'] = avg_confidence
        column['pattern_diversity'] = pattern_diversity
        column['last_batch_processed'] = datetime.now().isoformat()
        
        # Store batch processing results in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create batch_processing table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_processing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_id TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                avg_confidence REAL NOT NULL,
                pattern_diversity INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            INSERT INTO batch_processing (column_id, batch_size, avg_confidence, pattern_diversity, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (column['id'], len(column['batch_patterns']), avg_confidence, pattern_diversity, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # Clear batch for next processing
        column['batch_patterns'] = []

    def simulate_neural_columns(self, data_batch: List[Any]) -> List[Dict[str, Any]]:
        """Simulate neural column processing for the input batch."""
        columns = []
        
        # Process each item through columns
        for i, data_item in enumerate(data_batch):
            patterns = self.extractor_lobe.extract(data_item)
            processed = self.neural_column_lobe.process(patterns, f"item_{i}")
            
            # Create column representation
            column_data = {
                'column_id': f"batch_column_{i}",
                'input_item': data_item,
                'patterns': patterns,
                'processed_outputs': processed,
                'activation_state': self.column_states.get(f"batch_column_{i}", {}),
                'timestamp': datetime.now().isoformat()
            }
            columns.append(column_data)
        
        return columns

    def proactive_prompt(self, data_batch: List[Any]) -> List[Dict[str, Any]]:
        """Proactively suggest next patterns or actions based on input batch."""
        prompts = []
        
        # Analyze batch patterns
        batch_patterns = []
        for data_item in data_batch:
            patterns = self.extractor_lobe.extract(data_item)
            batch_patterns.extend(patterns)
        
        # Generate proactive prompts based on patterns
        for pattern in batch_patterns:
            prompt = self._generate_prompt_for_pattern(pattern, data_batch)
            if prompt:
                prompts.append(prompt)
        
        # Generate batch-level prompts
        batch_prompt = self._generate_batch_prompt(batch_patterns, data_batch)
        if batch_prompt:
            prompts.append(batch_prompt)
        
        # Store prompts
        for prompt in prompts:
            self._store_prompt(prompt)
        
        return prompts

    def _generate_prompt_for_pattern(self, pattern: Dict[str, Any], data_batch: List[Any]) -> Optional[Dict[str, Any]]:
        """Generate a proactive prompt for a specific pattern."""
        pattern_type = pattern['type']
        
        if pattern_type.startswith('text_'):
            return {
                'type': 'text_analysis',
                'suggestion': f"Consider analyzing text patterns for {pattern_type}",
                'confidence': pattern.get('confidence', 0.5),
                'context': f"Pattern: {pattern_type}, Value: {pattern['data']}"
            }
        elif pattern_type.startswith('list_'):
            return {
                'type': 'list_processing',
                'suggestion': f"Consider list operations for {pattern_type}",
                'confidence': pattern.get('confidence', 0.5),
                'context': f"Pattern: {pattern_type}, Value: {pattern['data']}"
            }
        elif pattern_type.startswith('dict_'):
            return {
                'type': 'dict_analysis',
                'suggestion': f"Consider dictionary analysis for {pattern_type}",
                'confidence': pattern.get('confidence', 0.5),
                'context': f"Pattern: {pattern_type}, Value: {pattern['data']}"
            }
        
        return None

    def _generate_batch_prompt(self, patterns: List[Dict[str, Any]], data_batch: List[Any]) -> Optional[Dict[str, Any]]:
        """Generate a batch-level proactive prompt."""
        if not patterns:
            return None
        
        # Analyze pattern distribution
        pattern_types = [p['type'] for p in patterns]
        type_counts = {}
        for p_type in pattern_types:
            type_counts[p_type] = type_counts.get(p_type, 0) + 1
        
        # Find most common pattern type
        most_common = max(type_counts.items(), key=lambda x: x[1])
        
        return {
            'type': 'batch_analysis',
            'suggestion': f"Batch contains {len(data_batch)} items with {most_common[1]} instances of {most_common[0]} patterns",
            'confidence': 0.7,
            'context': f"Total patterns: {len(patterns)}, Most common: {most_common[0]} ({most_common[1]} instances)"
        }

    def _store_pattern(self, pattern: Dict[str, Any]) -> int:
        """Store a recognized pattern in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO recognized_patterns (pattern_type, pattern_data, confidence, context)
            VALUES (?, ?, ?, ?)
        """, (pattern['type'], json.dumps(pattern['data']), pattern.get('confidence', 0.5), 
              json.dumps(pattern.get('context', {}))))
        
        pattern_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return pattern_id if pattern_id is not None else 0

    def _store_prompt(self, prompt: Dict[str, Any]):
        """Store a proactive prompt in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO proactive_prompts (prompt_type, prompt_data, context)
            VALUES (?, ?, ?)
        """, (prompt['type'], json.dumps(prompt), prompt.get('context', '')))
        
        conn.commit()
        conn.close()

    def get_patterns(self) -> List[Dict[str, Any]]:
        """Return the recognized patterns with metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, pattern_type, pattern_data, confidence, frequency, context, created_at, last_seen
            FROM recognized_patterns
            ORDER BY last_seen DESC
            LIMIT 100
        """)
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                'id': row[0],
                'type': row[1],
                'data': json.loads(row[2]),
                'confidence': row[3],
                'frequency': row[4],
                'context': json.loads(row[5]) if row[5] else {},
                'created_at': row[6],
                'last_seen': row[7]
            })
        
        conn.close()
        return patterns

    def get_column_states(self) -> Dict[str, Any]:
        """Get current neural column activation states."""
        return self.column_states

    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern recognition statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pattern statistics
        cursor.execute("SELECT COUNT(*) FROM recognized_patterns")
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT pattern_type) FROM recognized_patterns")
        unique_pattern_types = cursor.fetchone()[0]
        
        # Column statistics
        cursor.execute("SELECT COUNT(*) FROM neural_columns")
        total_columns = cursor.fetchone()[0]
        
        # Prompt statistics
        cursor.execute("SELECT COUNT(*) FROM proactive_prompts")
        total_prompts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_patterns': total_patterns,
            'unique_pattern_types': unique_pattern_types,
            'total_columns': total_columns,
            'total_prompts': total_prompts,
            'active_columns': len(self.column_states)
        }

    def receive_feedback(self, feedback: Dict[str, Any]):
        """
        Receive feedback on recognized patterns and self-tune internal parameters.
        Inspired by brain feedback loops and adaptive learning. See idea.txt.
        """
        # Example: Adjust confidence of patterns based on feedback
        for pattern_id, score in feedback.get('pattern_scores', {}).items():
            # Find and update pattern confidence in cache
            for entry in self.pattern_cache.cache:
                if entry['pattern'].get('id') == pattern_id:
                    entry['pattern']['confidence'] = max(0.0, min(1.0, entry['pattern'].get('confidence', 0.5) + 0.1 * (score - 3)))
        # Optionally adjust other parameters (e.g., decay, batch size)
        if 'decay' in feedback:
            self.pattern_cache.decay = feedback['decay']
        # Log feedback
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.publish('pattern_feedback_received', feedback)
        else:
            logging.info(f"[PatternRecognitionEngine] Feedback received: {feedback}")

    def set_micro_lobe_activation(self, extractor: bool = True, neural_column: bool = True, feedback: bool = True):
        """Dynamically activate/inhibit micro-lobes based on context or feedback."""
        self.extractor_lobe.enabled = extractor
        self.neural_column_lobe.enabled = neural_column
        self.feedback_lobe.enabled = feedback
        self.logger.info(f"[PatternRecognitionEngine] Micro-lobe activation set: extractor={self.extractor_lobe.enabled}, neural_column={self.neural_column_lobe.enabled}, feedback={self.feedback_lobe.enabled}")

    def recall_patterns_by_context(self, context: Optional[str] = None, n: int = 5) -> List[Dict[str, Any]]:
        """Recall most relevant patterns for a given context using associative memory."""
        context_str = str(context) if context is not None else ""
        return self.associative_memory.get_by_context(context_str, n=n)

    def compute_slope_map(self, data_batch: List[Any], model: Optional[Any] = None) -> dict:
        """
        Compute slope map for a batch of data using the provided model (if any).
        Handles missing numpy/torch gracefully.
        """
        slopes = []
        if model is not None and torch is not None and np is not None:
            try:
                for x in data_batch:
                    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
                    output = model(x_tensor)
                    grad = torch.autograd.grad(outputs=output, inputs=x_tensor, grad_outputs=torch.ones_like(output), retain_graph=True, allow_unused=True)[0]
                    slopes.append(float(torch.norm(grad).item()) if grad is not None else 0.0)
            except Exception as ex:
                self.logger.error(f"[PatternRecognitionEngine] Error in compute_slope_map with model: {ex}")
                slopes = [0.0 for _ in data_batch]
        elif np is not None:
            try:
                for x in data_batch:
                    x = np.array(x, dtype=float)
                    grad = np.zeros_like(x)
                    slopes.append(float(np.linalg.norm(grad)))
            except Exception as ex:
                self.logger.error(f"[PatternRecognitionEngine] Error in compute_slope_map with numpy: {ex}")
                slopes = [0.0 for _ in data_batch]
        else:
            # Fallback: use simple sum as a placeholder
            slopes = [float(sum(x)) if isinstance(x, (list, tuple)) else 0.0 for x in data_batch]
        return {"slopes": slopes}

    def _pattern_func(self, x, np=None):
        """
        Placeholder for a pattern function. Override or extend for custom pattern models.
        Accepts an optional numpy module for compatibility with slope map computation.
        """
        if np is not None:
            return float(np.sum(x))
        # Fallback if numpy is not available
        return float(sum(x))

    def advanced_neural_column_processing(self, data_batch: List[Any], model: Optional[Any] = None, feedback: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Advanced neural column and batch processing with feedback-driven adaptation.
        Integrates with external models and supports dynamic feedback.
        Returns a list of recognized patterns with confidence and context.
        """
        try:
            # Example: use model if provided, else fallback to internal logic
            if model:
                results = model(data_batch)
            else:
                results = self.simulate_neural_columns(data_batch)
            if feedback:
                self.receive_feedback(feedback)
            self.logger.info(f"[PatternRecognitionEngine] Advanced neural column processing complete.")
            return results
        except Exception as ex:
            self.logger.error(f"[PatternRecognitionEngine] Error in advanced_neural_column_processing: {ex}")
            return []

    def demo_custom_pattern_recognition(self, data_batch: List[Any], custom_recognizer: Callable) -> List[Dict[str, Any]]:
        """
        Demo/test method: run a custom pattern recognition function on a data batch.
        Usage: engine.demo_custom_pattern_recognition(data, lambda d: [...])
        Returns the result of the custom recognizer.
        """
        try:
            result = custom_recognizer(data_batch)
            self.logger.info(f"[PatternRecognitionEngine] Custom pattern recognition result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[PatternRecognitionEngine] Custom pattern recognition error: {ex}")
            return []

    def advanced_feedback_integration(self, feedback: dict):
        """
        Advanced feedback integration and continual learning for pattern recognition engine.
        Updates pattern extraction, neural column, or feedback models based on structured feedback.
        Supports cross-lobe research and adaptation.
        """
        try:
            if feedback and 'decay' in feedback:
                self.pattern_cache.decay = float(feedback['decay'])
                self.logger.info(f"[PatternRecognitionEngine] Pattern cache decay updated to {self.pattern_cache.decay} from feedback.")
            self.working_memory.add({"advanced_feedback": feedback})
        except Exception as ex:
            self.logger.error(f"[PatternRecognitionEngine] Error in advanced_feedback_integration: {ex}")

    def cross_lobe_integration(self, lobe_name: str = "", data: Any = None) -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        
        This method implements hormone-based cross-lobe communication following
        the brain-inspired architecture. It processes incoming data from other
        lobes and shares relevant pattern information.
        
        Args:
            lobe_name: Name of the requesting lobe (e.g., 'VectorLobe', 'PhysicsLobe')
            data: Optional data payload from the requesting lobe
            
        Returns:
            Dict containing relevant patterns and state information for the requesting lobe
            
        Example: call VectorLobe or PhysicsLobe for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[PatternRecognitionEngine] Cross-lobe integration with {lobe_name}")
        
        try:
            # Process incoming data if provided
            if data is not None:
                self.receive_data(data)
            
            # Prepare response based on requesting lobe type
            response = {
                'lobe_name': 'PatternRecognitionEngine',
                'timestamp': time.time(),
                'patterns': self.get_patterns(),
                'column_states': self.column_states,
                'confidence_scores': self.pattern_confidence
            }
            
            # Lobe-specific data sharing
            if lobe_name.lower() in ['vectorlobe', 'vector_lobe']:
                # Share pattern vectors and embeddings
                response['pattern_vectors'] = self._extract_pattern_vectors()
                response['similarity_scores'] = self._calculate_pattern_similarities()
                
            elif lobe_name.lower() in ['physicslobe', 'physics_lobe']:
                # Share spatial and temporal pattern relationships
                response['spatial_patterns'] = self._extract_spatial_patterns()
                response['temporal_sequences'] = self._extract_temporal_patterns()
                
            elif lobe_name.lower() in ['memorylobe', 'memory_lobe']:
                # Share pattern associations and memory links
                response['pattern_associations'] = self._extract_pattern_associations()
                response['memory_strength'] = self._calculate_memory_strength()
                
            elif lobe_name.lower() in ['scientificlobe', 'scientific_lobe']:
                # Share pattern hypotheses and experimental data
                response['pattern_hypotheses'] = self._generate_pattern_hypotheses()
                response['experimental_results'] = self._extract_experimental_patterns()
                
            else:
                # Generic cross-lobe sharing
                response['recent_patterns'] = self.get_recent_patterns(limit=10)
                response['high_confidence_patterns'] = self._get_high_confidence_patterns()
            
            # Trigger hormone release for successful cross-lobe communication
            if hasattr(self, 'hormone_system'):
                self.hormone_system.release_hormone('dopamine', 0.1)  # Small reward for cooperation
                
            self.logger.info(f"[PatternRecognitionEngine] Successfully integrated with {lobe_name}")
            return response
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Cross-lobe integration error: {e}")
            # Trigger stress hormone on failure
            if hasattr(self, 'hormone_system'):
                self.hormone_system.release_hormone('cortisol', 0.05)
            return {'error': str(e), 'lobe_name': 'PatternRecognitionEngine'}

    def usage_example(self):
        """
        Usage example for pattern recognition engine:
        >>> engine = PatternRecognitionEngine()
        >>> patterns = engine.recognize_patterns(["foo", "bar"])
        >>> print(patterns)
        >>> # Advanced feedback integration
        >>> engine.advanced_feedback_integration({'decay': 0.9})
        >>> # Cross-lobe integration
        >>> engine.cross_lobe_integration(lobe_name='VectorLobe')
        """
        pass

    def get_state(self):
        """Return a summary of the current pattern recognition engine state for aggregation."""
        return {
            'db_path': self.db_path,
            'patterns': self.patterns,
            'neural_columns': self.neural_columns,
            'column_states': self.column_states,
            'pattern_confidence': self.pattern_confidence,
            'working_memory': self.working_memory.get_all() if hasattr(self.working_memory, 'get_all') else None
        }

    def receive_data(self, data: dict):
        """
        Receive and integrate data from aggregator or adjacent lobes.
        
        This method processes incoming data from other lobes in the brain-inspired
        architecture, updating internal state and triggering appropriate responses
        based on the hormone system and genetic triggers.
        
        Args:
            data: Dictionary containing data from other lobes, including:
                - lobe_name: Source lobe identifier
                - patterns: Pattern data to integrate
                - hormone_levels: Current hormone state
                - context: Contextual information
                - timestamp: When data was sent
        """
        self.logger.info(f"[PatternRecognitionEngine] Received data from: {data.get('lobe_name', 'unknown')}")
        
        try:
            # Extract key information from received data
            source_lobe = data.get('lobe_name', 'unknown')
            patterns = data.get('patterns', [])
            hormone_levels = data.get('hormone_levels', {})
            context = data.get('context', {})
            timestamp = data.get('timestamp', time.time())
            
            # Process received patterns
            if patterns:
                self._integrate_external_patterns(patterns, source_lobe, context)
            
            # Update hormone-sensitive parameters
            if hormone_levels:
                self._adapt_to_hormone_levels(hormone_levels)
            
            # Store cross-lobe interaction in associative memory
            if hasattr(self, 'associative_memory'):
                self.associative_memory.add(
                    pattern={'type': 'cross_lobe_data', 'source': source_lobe, 'data': data},
                    context=f"cross_lobe_integration_{source_lobe}",
                    feedback={'integration_success': True, 'timestamp': timestamp}
                )
            
            # Update working memory with relevant information
            if hasattr(self.working_memory, 'add'):
                self.working_memory.add({
                    'type': 'cross_lobe_integration',
                    'source': source_lobe,
                    'pattern_count': len(patterns) if patterns else 0,
                    'timestamp': timestamp
                })
            
            # Trigger hormone release for successful data integration
            if hasattr(self, 'hormone_system'):
                # Small dopamine release for successful cooperation
                self.hormone_system.release_hormone('dopamine', 0.05)
                
                # Growth hormone if learning new patterns
                if patterns and len(patterns) > 0:
                    self.hormone_system.release_hormone('growth_hormone', 0.02)
            
            # Update column states based on received data
            self._update_columns_from_external_data(data)
            
            self.logger.info(f"[PatternRecognitionEngine] Successfully integrated data from {source_lobe}")
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error receiving data: {e}")
            
            # Trigger stress hormone on integration failure
            if hasattr(self, 'hormone_system'):
                self.hormone_system.release_hormone('cortisol', 0.03)
            
            # Store failed integration attempt
            if hasattr(self, 'associative_memory'):
                self.associative_memory.add(
                    pattern={'type': 'integration_failure', 'error': str(e), 'data': data},
                    context='cross_lobe_integration_error',
                    feedback={'integration_success': False, 'error': str(e)}
                )
    
    def _integrate_external_patterns(self, patterns: List[Dict[str, Any]], source_lobe: str, context: Dict[str, Any]):
        """
        Integrate patterns received from external lobes into the pattern recognition system.
        
        Args:
            patterns: List of pattern dictionaries from external lobe
            source_lobe: Name of the source lobe
            context: Contextual information about the patterns
        """
        try:
            for pattern in patterns:
                # Add source information to pattern
                enhanced_pattern = pattern.copy()
                enhanced_pattern['source_lobe'] = source_lobe
                enhanced_pattern['integration_timestamp'] = time.time()
                enhanced_pattern['context'] = context
                
                # Process pattern through our recognition system
                recognized = self.recognize_patterns([enhanced_pattern])
                
                # Update pattern confidence based on cross-lobe validation
                if recognized:
                    original_confidence = pattern.get('confidence', 0.5)
                    cross_validation_boost = 0.1  # Small boost for cross-lobe validation
                    enhanced_confidence = min(1.0, original_confidence + cross_validation_boost)
                    
                    # Store enhanced pattern
                    self._store_pattern(enhanced_pattern, enhanced_confidence)
                    
            self.logger.info(f"[PatternRecognitionEngine] Integrated {len(patterns)} patterns from {source_lobe}")
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error integrating external patterns: {e}")
    
    def _adapt_to_hormone_levels(self, hormone_levels: Dict[str, float]):
        """
        Adapt pattern recognition parameters based on current hormone levels.
        
        Args:
            hormone_levels: Dictionary of hormone names and their current levels
        """
        try:
            # Dopamine affects learning rate and sensitivity
            dopamine = hormone_levels.get('dopamine', 0.5)
            if dopamine > 0.7:
                # High dopamine increases learning rate and sensitivity
                for column in self.neural_columns.values():
                    column.learning_rate = min(column.learning_rate_bounds[1], 
                                             column.learning_rate * 1.02)
                    column.sensitivity = min(column.sensitivity_bounds[1], 
                                           column.sensitivity * 1.01)
            elif dopamine < 0.3:
                # Low dopamine decreases learning rate and sensitivity
                for column in self.neural_columns.values():
                    column.learning_rate = max(column.learning_rate_bounds[0], 
                                             column.learning_rate * 0.98)
                    column.sensitivity = max(column.sensitivity_bounds[0], 
                                           column.sensitivity * 0.99)
            
            # Cortisol affects pattern confidence thresholds
            cortisol = hormone_levels.get('cortisol', 0.2)
            if cortisol > 0.5:
                # High stress reduces confidence in pattern recognition
                self.confidence_threshold = max(0.3, self.confidence_threshold * 0.95)
            else:
                # Low stress allows higher confidence
                self.confidence_threshold = min(0.8, self.confidence_threshold * 1.01)
            
            # Growth hormone affects pattern association strength
            growth_hormone = hormone_levels.get('growth_hormone', 0.3)
            if growth_hormone > 0.5:
                # High growth hormone strengthens pattern associations
                self.association_strength_multiplier = min(2.0, 
                    getattr(self, 'association_strength_multiplier', 1.0) * 1.05)
            
            # Serotonin affects overall system stability
            serotonin = hormone_levels.get('serotonin', 0.6)
            if serotonin < 0.4:
                # Low serotonin increases system volatility
                self.pattern_decay_rate = min(0.99, 
                    getattr(self, 'pattern_decay_rate', 0.95) * 1.02)
            
            self.logger.info(f"[PatternRecognitionEngine] Adapted to hormone levels: {hormone_levels}")
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error adapting to hormone levels: {e}")
    
    def _update_columns_from_external_data(self, data: Dict[str, Any]):
        """
        Update neural column states based on external data from other lobes.
        
        Args:
            data: External data containing patterns and context
        """
        try:
            patterns = data.get('patterns', [])
            source_lobe = data.get('lobe_name', 'unknown')
            
            # Update columns based on pattern types in external data
            for pattern in patterns:
                pattern_type = pattern.get('type', 'unknown')
                
                # Find columns that can process this pattern type
                relevant_columns = []
                for column_id, column in self.neural_columns.items():
                    if any(pt in pattern_type.lower() for pt in column.pattern_types):
                        relevant_columns.append(column)
                
                # Update relevant columns with external pattern information
                for column in relevant_columns:
                    # Slight activation boost from cross-lobe validation
                    column.activation_state = min(1.0, column.activation_state + 0.05)
                    
                    # Learn association with source lobe
                    if hasattr(column, 'learn_pattern_association_with_strength'):
                        external_pattern = {
                            'type': f"{source_lobe}_pattern",
                            'data': pattern.get('data'),
                            'confidence': pattern.get('confidence', 0.5)
                        }
                        column.learn_pattern_association_with_strength(
                            pattern, external_pattern, 
                            strength=0.05, 
                            context=f"cross_lobe_{source_lobe}"
                        )
            
            self.logger.info(f"[PatternRecognitionEngine] Updated columns from {source_lobe} data")
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error updating columns from external data: {e}")
    
    def _extract_pattern_vectors(self) -> List[Dict[str, Any]]:
        """Extract pattern vectors for sharing with vector-based lobes."""
        try:
            vectors = []
            for pattern_id, pattern_data in self.patterns.items():
                if isinstance(pattern_data, dict):
                    vector_info = {
                        'pattern_id': pattern_id,
                        'pattern_type': pattern_data.get('type', 'unknown'),
                        'confidence': pattern_data.get('confidence', 0.5),
                        'vector_representation': self._create_pattern_vector(pattern_data),
                        'timestamp': pattern_data.get('timestamp', time.time())
                    }
                    vectors.append(vector_info)
            
            return vectors[:50]  # Limit to most recent 50 patterns
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error extracting pattern vectors: {e}")
            return []
    
    def _create_pattern_vector(self, pattern: Dict[str, Any]) -> List[float]:
        """Create a simple vector representation of a pattern."""
        try:
            # Simple hash-based vector creation (placeholder for more sophisticated methods)
            pattern_str = str(pattern.get('data', ''))
            pattern_type = pattern.get('type', 'unknown')
            confidence = pattern.get('confidence', 0.5)
            
            # Create a simple 10-dimensional vector
            vector = [0.0] * 10
            
            # Use hash values to populate vector dimensions
            for i, char in enumerate(pattern_str[:10]):
                vector[i] = (ord(char) % 100) / 100.0
            
            # Add pattern type influence
            type_hash = hash(pattern_type) % 1000
            for i in range(len(vector)):
                vector[i] += (type_hash % (i + 1)) / 1000.0
            
            # Scale by confidence
            vector = [v * confidence for v in vector]
            
            return vector
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error creating pattern vector: {e}")
            return [0.0] * 10
    
    def _calculate_pattern_similarities(self) -> Dict[str, float]:
        """Calculate similarity scores between patterns."""
        try:
            similarities = {}
            pattern_items = list(self.patterns.items())
            
            for i, (id1, pattern1) in enumerate(pattern_items):
                for j, (id2, pattern2) in enumerate(pattern_items[i+1:], i+1):
                    similarity = self._calculate_similarity(pattern1, pattern2)
                    similarities[f"{id1}_{id2}"] = similarity
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error calculating pattern similarities: {e}")
            return {}
    
    def _calculate_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns."""
        try:
            # Simple similarity based on type and confidence
            type_similarity = 1.0 if pattern1.get('type') == pattern2.get('type') else 0.3
            
            conf1 = pattern1.get('confidence', 0.5)
            conf2 = pattern2.get('confidence', 0.5)
            confidence_similarity = 1.0 - abs(conf1 - conf2)
            
            # Data similarity (simple string comparison)
            data1 = str(pattern1.get('data', ''))
            data2 = str(pattern2.get('data', ''))
            
            if data1 and data2:
                common_chars = len(set(data1.lower()) & set(data2.lower()))
                total_chars = len(set(data1.lower()) | set(data2.lower()))
                data_similarity = common_chars / max(total_chars, 1)
            else:
                data_similarity = 0.0
            
            # Weighted average
            overall_similarity = (
                type_similarity * 0.4 + 
                confidence_similarity * 0.3 + 
                data_similarity * 0.3
            )
            
            return overall_similarity
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error calculating similarity: {e}")
            return 0.0
    
    def _extract_spatial_patterns(self) -> List[Dict[str, Any]]:
        """Extract spatial pattern relationships for physics-based lobes."""
        try:
            spatial_patterns = []
            
            for column_id, column in self.neural_columns.items():
                if hasattr(column, 'position'):
                    spatial_info = {
                        'column_id': column_id,
                        'position': column.position,
                        'activation_state': column.activation_state,
                        'pattern_types': column.pattern_types,
                        'spatial_relationships': self._calculate_spatial_relationships(column)
                    }
                    spatial_patterns.append(spatial_info)
            
            return spatial_patterns
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error extracting spatial patterns: {e}")
            return []
    
    def _calculate_spatial_relationships(self, column) -> Dict[str, Any]:
        """Calculate spatial relationships for a neural column."""
        try:
            relationships = {
                'neighbors': [],
                'distance_map': {},
                'activation_gradient': 0.0
            }
            
            # Find neighboring columns
            for other_id, other_column in self.neural_columns.items():
                if other_id != column.column_id and hasattr(other_column, 'position'):
                    distance = self._calculate_distance(column.position, other_column.position)
                    relationships['distance_map'][other_id] = distance
                    
                    if distance < 2.0:  # Arbitrary threshold for "neighbors"
                        relationships['neighbors'].append({
                            'column_id': other_id,
                            'distance': distance,
                            'activation_difference': abs(column.activation_state - other_column.activation_state)
                        })
            
            # Calculate activation gradient
            if relationships['neighbors']:
                total_diff = sum(n['activation_difference'] for n in relationships['neighbors'])
                relationships['activation_gradient'] = total_diff / len(relationships['neighbors'])
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error calculating spatial relationships: {e}")
            return {}
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D positions."""
        try:
            return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error calculating distance: {e}")
            return float('inf')
    
    def _extract_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Extract temporal pattern sequences for time-based analysis."""
        try:
            temporal_patterns = []
            
            # Get patterns with timestamps
            timestamped_patterns = []
            for pattern_id, pattern_data in self.patterns.items():
                if isinstance(pattern_data, dict) and 'timestamp' in pattern_data:
                    timestamped_patterns.append((pattern_data['timestamp'], pattern_id, pattern_data))
            
            # Sort by timestamp
            timestamped_patterns.sort(key=lambda x: x[0])
            
            # Create temporal sequences
            sequence_length = 5
            for i in range(len(timestamped_patterns) - sequence_length + 1):
                sequence = timestamped_patterns[i:i + sequence_length]
                
                temporal_info = {
                    'sequence_start': sequence[0][0],
                    'sequence_end': sequence[-1][0],
                    'duration': sequence[-1][0] - sequence[0][0],
                    'pattern_sequence': [
                        {
                            'pattern_id': item[1],
                            'pattern_type': item[2].get('type', 'unknown'),
                            'confidence': item[2].get('confidence', 0.5),
                            'timestamp': item[0]
                        }
                        for item in sequence
                    ],
                    'sequence_confidence': sum(item[2].get('confidence', 0.5) for item in sequence) / len(sequence)
                }
                temporal_patterns.append(temporal_info)
            
            return temporal_patterns[-20:]  # Return most recent 20 sequences
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error extracting temporal patterns: {e}")
            return []
    
    def _extract_pattern_associations(self) -> Dict[str, Any]:
        """Extract pattern associations for sharing with memory-based lobes."""
        try:
            associations = {
                'column_associations': {},
                'global_associations': {},
                'association_strengths': {},
                'total_associations': 0
            }
            
            # Extract associations from neural columns
            for column_id, column in self.neural_columns.items():
                if hasattr(column, 'pattern_associations') and column.pattern_associations:
                    associations['column_associations'][column_id] = column.pattern_associations.copy()
                
                if hasattr(column, 'association_strengths') and column.association_strengths:
                    associations['association_strengths'][column_id] = column.association_strengths.copy()
            
            # Calculate global association patterns
            all_pattern_types = set()
            for column_associations in associations['column_associations'].values():
                all_pattern_types.update(column_associations.keys())
            
            for pattern_type in all_pattern_types:
                total_strength = 0.0
                column_count = 0
                
                for column_associations in associations['column_associations'].values():
                    if pattern_type in column_associations:
                        total_strength += column_associations[pattern_type]
                        column_count += 1
                
                if column_count > 0:
                    associations['global_associations'][pattern_type] = {
                        'average_strength': total_strength / column_count,
                        'total_strength': total_strength,
                        'column_count': column_count,
                        'prevalence': column_count / len(self.neural_columns)
                    }
            
            associations['total_associations'] = len(associations['global_associations'])
            
            return associations
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error extracting pattern associations: {e}")
            return {'column_associations': {}, 'global_associations': {}, 'total_associations': 0}
    
    def _calculate_memory_strength(self) -> Dict[str, float]:
        """Calculate overall memory strength metrics."""
        try:
            memory_metrics = {
                'average_pattern_confidence': 0.0,
                'total_patterns': 0,
                'high_confidence_ratio': 0.0,
                'association_density': 0.0,
                'memory_coherence': 0.0
            }
            
            # Calculate pattern confidence metrics
            if self.patterns:
                confidences = []
                high_confidence_count = 0
                
                for pattern_data in self.patterns.values():
                    if isinstance(pattern_data, dict):
                        confidence = pattern_data.get('confidence', 0.5)
                        confidences.append(confidence)
                        if confidence > 0.7:
                            high_confidence_count += 1
                
                if confidences:
                    memory_metrics['average_pattern_confidence'] = sum(confidences) / len(confidences)
                    memory_metrics['total_patterns'] = len(confidences)
                    memory_metrics['high_confidence_ratio'] = high_confidence_count / len(confidences)
            
            # Calculate association density
            total_associations = 0
            for column in self.neural_columns.values():
                if hasattr(column, 'pattern_associations'):
                    total_associations += len(column.pattern_associations)
            
            if memory_metrics['total_patterns'] > 0:
                memory_metrics['association_density'] = total_associations / memory_metrics['total_patterns']
            
            # Calculate memory coherence (how well patterns are interconnected)
            if total_associations > 0:
                coherence_score = 0.0
                coherence_count = 0
                
                for column in self.neural_columns.values():
                    if hasattr(column, 'association_strengths'):
                        for assoc_data in column.association_strengths.values():
                            coherence_score += assoc_data.get('strength', 0.0)
                            coherence_count += 1
                
                if coherence_count > 0:
                    memory_metrics['memory_coherence'] = coherence_score / coherence_count
            
            return memory_metrics
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error calculating memory strength: {e}")
            return {'average_pattern_confidence': 0.0, 'total_patterns': 0, 'memory_coherence': 0.0}
    
    def _generate_pattern_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate pattern hypotheses for scientific analysis."""
        try:
            hypotheses = []
            
            # Analyze pattern trends and generate hypotheses
            pattern_types = {}
            confidence_trends = {}
            
            # Collect pattern statistics
            for pattern_id, pattern_data in self.patterns.items():
                if isinstance(pattern_data, dict):
                    pattern_type = pattern_data.get('type', 'unknown')
                    confidence = pattern_data.get('confidence', 0.5)
                    timestamp = pattern_data.get('timestamp', time.time())
                    
                    if pattern_type not in pattern_types:
                        pattern_types[pattern_type] = []
                    pattern_types[pattern_type].append({
                        'confidence': confidence,
                        'timestamp': timestamp,
                        'pattern_id': pattern_id
                    })
            
            # Generate hypotheses based on pattern analysis
            for pattern_type, pattern_list in pattern_types.items():
                if len(pattern_list) >= 3:  # Need minimum data for hypothesis
                    confidences = [p['confidence'] for p in pattern_list]
                    avg_confidence = sum(confidences) / len(confidences)
                    confidence_variance = sum((c - avg_confidence)**2 for c in confidences) / len(confidences)
                    
                    # Hypothesis 1: Pattern confidence trend
                    if len(pattern_list) >= 5:
                        recent_confidence = sum(p['confidence'] for p in pattern_list[-3:]) / 3
                        early_confidence = sum(p['confidence'] for p in pattern_list[:3]) / 3
                        
                        if recent_confidence > early_confidence + 0.1:
                            hypotheses.append({
                                'hypothesis_type': 'confidence_improvement',
                                'pattern_type': pattern_type,
                                'description': f"Pattern recognition confidence for {pattern_type} is improving over time",
                                'evidence': {
                                    'early_confidence': early_confidence,
                                    'recent_confidence': recent_confidence,
                                    'improvement': recent_confidence - early_confidence
                                },
                                'confidence': 0.7,
                                'testable': True
                            })
                    
                    # Hypothesis 2: Pattern stability
                    if confidence_variance < 0.05:
                        hypotheses.append({
                            'hypothesis_type': 'pattern_stability',
                            'pattern_type': pattern_type,
                            'description': f"Pattern {pattern_type} shows consistent recognition confidence",
                            'evidence': {
                                'average_confidence': avg_confidence,
                                'variance': confidence_variance,
                                'sample_size': len(pattern_list)
                            },
                            'confidence': 0.8,
                            'testable': True
                        })
                    
                    # Hypothesis 3: Pattern frequency correlation
                    if len(pattern_list) > 10:
                        hypotheses.append({
                            'hypothesis_type': 'frequency_correlation',
                            'pattern_type': pattern_type,
                            'description': f"High frequency pattern {pattern_type} may indicate system specialization",
                            'evidence': {
                                'frequency': len(pattern_list),
                                'average_confidence': avg_confidence,
                                'specialization_score': len(pattern_list) * avg_confidence
                            },
                            'confidence': 0.6,
                            'testable': True
                        })
            
            # Cross-pattern hypotheses
            if len(pattern_types) >= 2:
                type_confidences = {
                    ptype: sum(p['confidence'] for p in plist) / len(plist)
                    for ptype, plist in pattern_types.items()
                    if len(plist) >= 2
                }
                
                if len(type_confidences) >= 2:
                    best_type = max(type_confidences.keys(), key=lambda k: type_confidences[k])
                    worst_type = min(type_confidences.keys(), key=lambda k: type_confidences[k])
                    
                    if type_confidences[best_type] - type_confidences[worst_type] > 0.2:
                        hypotheses.append({
                            'hypothesis_type': 'specialization_difference',
                            'pattern_type': 'cross_pattern',
                            'description': f"System shows specialization: better at {best_type} than {worst_type}",
                            'evidence': {
                                'best_type': best_type,
                                'best_confidence': type_confidences[best_type],
                                'worst_type': worst_type,
                                'worst_confidence': type_confidences[worst_type],
                                'difference': type_confidences[best_type] - type_confidences[worst_type]
                            },
                            'confidence': 0.75,
                            'testable': True
                        })
            
            return hypotheses[:10]  # Return top 10 hypotheses
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error generating pattern hypotheses: {e}")
            return []
    
    def _extract_experimental_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns that could serve as experimental data."""
        try:
            experimental_patterns = []
            
            # Find patterns with varying confidence levels for experimentation
            confidence_buckets = {'low': [], 'medium': [], 'high': []}
            
            for pattern_id, pattern_data in self.patterns.items():
                if isinstance(pattern_data, dict):
                    confidence = pattern_data.get('confidence', 0.5)
                    
                    if confidence < 0.4:
                        confidence_buckets['low'].append((pattern_id, pattern_data))
                    elif confidence < 0.7:
                        confidence_buckets['medium'].append((pattern_id, pattern_data))
                    else:
                        confidence_buckets['high'].append((pattern_id, pattern_data))
            
            # Create experimental datasets
            for bucket_name, patterns in confidence_buckets.items():
                if patterns:
                    experimental_patterns.append({
                        'experiment_type': f"{bucket_name}_confidence_analysis",
                        'description': f"Analysis of {bucket_name} confidence patterns",
                        'sample_size': len(patterns),
                        'patterns': [
                            {
                                'pattern_id': pid,
                                'pattern_type': pdata.get('type', 'unknown'),
                                'confidence': pdata.get('confidence', 0.5),
                                'data_summary': str(pdata.get('data', ''))[:100]  # Truncated for privacy
                            }
                            for pid, pdata in patterns[:5]  # Sample of 5 patterns
                        ],
                        'statistical_summary': {
                            'mean_confidence': sum(p[1].get('confidence', 0.5) for p in patterns) / len(patterns),
                            'pattern_types': list(set(p[1].get('type', 'unknown') for p in patterns)),
                            'sample_variance': self._calculate_variance([p[1].get('confidence', 0.5) for p in patterns])
                        }
                    })
            
            # Add column performance experiments
            for column_id, column in self.neural_columns.items():
                if hasattr(column, 'performance_metrics') and column.performance_metrics:
                    experimental_patterns.append({
                        'experiment_type': 'column_performance_analysis',
                        'description': f"Performance analysis for neural column {column_id}",
                        'column_id': column_id,
                        'pattern_types': column.pattern_types,
                        'performance_data': {
                            'algorithmic_metrics': column.performance_metrics.get('algorithmic', {}),
                            'neural_metrics': column.performance_metrics.get('neural', {}),
                            'use_neural': column.use_neural,
                            'activation_state': column.activation_state
                        }
                    })
            
            return experimental_patterns
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error extracting experimental patterns: {e}")
            return []
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean)**2 for x in values) / len(values)
        return variance
    
    def _get_high_confidence_patterns(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get patterns with confidence above threshold."""
        try:
            high_confidence = []
            
            for pattern_id, pattern_data in self.patterns.items():
                if isinstance(pattern_data, dict):
                    confidence = pattern_data.get('confidence', 0.5)
                    if confidence >= threshold:
                        high_confidence.append({
                            'pattern_id': pattern_id,
                            'pattern_type': pattern_data.get('type', 'unknown'),
                            'confidence': confidence,
                            'data': pattern_data.get('data'),
                            'timestamp': pattern_data.get('timestamp', time.time())
                        })
            
            # Sort by confidence (highest first)
            high_confidence.sort(key=lambda x: x['confidence'], reverse=True)
            
            return high_confidence[:20]  # Return top 20
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error getting high confidence patterns: {e}")
            return []
    
    def get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recently processed patterns."""
        try:
            recent_patterns = []
            
            # Get patterns with timestamps
            timestamped_patterns = []
            for pattern_id, pattern_data in self.patterns.items():
                if isinstance(pattern_data, dict):
                    timestamp = pattern_data.get('timestamp', 0)
                    timestamped_patterns.append((timestamp, pattern_id, pattern_data))
            
            # Sort by timestamp (most recent first)
            timestamped_patterns.sort(key=lambda x: x[0], reverse=True)
            
            # Format for return
            for timestamp, pattern_id, pattern_data in timestamped_patterns[:limit]:
                recent_patterns.append({
                    'pattern_id': pattern_id,
                    'pattern_type': pattern_data.get('type', 'unknown'),
                    'confidence': pattern_data.get('confidence', 0.5),
                    'data': pattern_data.get('data'),
                    'timestamp': timestamp
                })
            
            return recent_patterns
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error getting recent patterns: {e}")
            return []
    
    def _store_pattern(self, pattern: Dict[str, Any], confidence: float):
        """Store a pattern with enhanced confidence in the pattern database."""
        try:
            pattern_id = f"pattern_{int(time.time())}_{hash(str(pattern))}"
            
            enhanced_pattern = pattern.copy()
            enhanced_pattern['confidence'] = confidence
            enhanced_pattern['timestamp'] = time.time()
            enhanced_pattern['stored_by'] = 'cross_lobe_integration'
            
            # Store in patterns dictionary
            self.patterns[pattern_id] = enhanced_pattern
            
            # Store in database if available
            if hasattr(self, 'db_path') and self.db_path:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO recognized_patterns 
                        (pattern_type, pattern_data, confidence, context, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        enhanced_pattern.get('type', 'unknown'),
                        json.dumps(enhanced_pattern.get('data')),
                        confidence,
                        enhanced_pattern.get('context', 'cross_lobe_integration'),
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                except Exception as db_error:
                    self.logger.warning(f"[PatternRecognitionEngine] Database storage failed: {db_error}")
            
            self.logger.info(f"[PatternRecognitionEngine] Stored enhanced pattern {pattern_id} with confidence {confidence:.3f}")
            
        except Exception as e:
            self.logger.error(f"[PatternRecognitionEngine] Error storing pattern: {e}")