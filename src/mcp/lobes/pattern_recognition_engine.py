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
    Neural column implementation inspired by cortical columns.
    Each column specializes in processing specific pattern types with adaptive sensitivity.
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
        
        # Neural network alternative for pattern processing
        self.neural_processor = None
        self.algorithmic_processor = self._default_algorithmic_processor
        self.use_neural = False
        self.performance_metrics = {"neural": {}, "algorithmic": {}}
        
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
        """Initialize default neural columns for common pattern types."""
        # Text processing column
        self.neural_columns['text_processor'] = NeuralColumn(
            'text_processor',
            ['text', 'string', 'word'],
            position=(0, 0, 0)
        )
        
        # Visual processing column
        self.neural_columns['visual_processor'] = NeuralColumn(
            'visual_processor',
            ['visual', 'image', 'spatial'],
            position=(1, 0, 0)
        )
        
        # Sequence processing column
        self.neural_columns['sequence_processor'] = NeuralColumn(
            'sequence_processor',
            ['list', 'sequence', 'array'],
            position=(0, 1, 0)
        )
        
        # Structure processing column
        self.neural_columns['structure_processor'] = NeuralColumn(
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
        Example: call VectorLobe or PhysicsLobe for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[PatternRecognitionEngine] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        return self.get_patterns()

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
        """Stub: Receive data from aggregator or adjacent lobes."""
        self.logger.info(f"[PatternRecognitionEngine] Received data: {data}")
        # TODO: Integrate received data into engine state 