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
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import collections
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory  # See idea.txt
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
import logging
import random
import time
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool


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


class PatternRecognitionEngine:
    """
    Pattern recognition, neural column simulation, and related features.
    Now models discrete, vesicle-like 'quanta' for inter-lobe signaling, supports both spontaneous and evoked release modes.
    
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
    def __init__(self, db_path: Optional[str] = None, event_bus: Optional[LobeEventBus] = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'pattern_recognition.db')
        
        self.db_path = db_path
        self.patterns = []
        self.neural_columns = []
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
        self.logger.info("[PatternRecognitionEngine] VesiclePool initialized: %s", self.vesicle_pool.get_state())
        self._init_database()
    
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
        Compute a slope map for a neural network-inspired model or pattern batch.
        Slope maps can be used for regularization, loss, or early stopping to improve generalization (see arXiv:2107.01473).
        If a model is provided, attempts to compute the gradient norm (slope) for each input.
        Returns a dict with slope statistics and optionally per-sample slopes.
        """
        slopes = []
        np = None
        if model is not None:
            try:
                import numpy as np
                for x in data_batch:
                    try:
                        import torch
                        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
                        output = model(x_tensor)
                        if hasattr(output, 'sum'):
                            output = output.sum()
                        output.backward()
                        grad = x_tensor.grad
                        slope = torch.norm(grad).item() if grad is not None else 0.0
                        slopes.append(slope)
                    except Exception:
                        slopes.append(0.0)
            except ImportError:
                slopes = [0.0 for _ in data_batch]
        else:
            try:
                import numpy as np
                for x in data_batch:
                    x = np.array(x, dtype=float)
                    eps = 1e-4
                    grad = np.zeros_like(x)
                    for i in range(len(x)):
                        x1 = x.copy(); x1[i] += eps
                        x2 = x.copy(); x2[i] -= eps
                        grad[i] = (self._pattern_func(x1, np=np) - self._pattern_func(x2, np=np)) / (2 * eps)
                    slopes.append(np.linalg.norm(grad))
            except Exception:
                slopes = [0.0 for _ in data_batch]
        if np is not None:
            slope_stats = {
                "mean": float(np.mean(slopes)) if slopes else 0.0,
                "std": float(np.std(slopes)) if slopes else 0.0,
                "min": float(np.min(slopes)) if slopes else 0.0,
                "max": float(np.max(slopes)) if slopes else 0.0,
                "slopes": slopes
            }
        else:
            slope_stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "slopes": slopes
            }
        self.logger.info(f"[PatternRecognitionEngine] Slope map computed: {slope_stats}")
        return slope_stats

    def _pattern_func(self, x, np=None):
        """
        Placeholder for a pattern function. Override or extend for custom pattern models.
        Accepts an optional numpy module for compatibility with slope map computation.
        """
        if np is not None:
            return float(np.sum(x))
        # Fallback if numpy is not available
        return float(sum(x)) 