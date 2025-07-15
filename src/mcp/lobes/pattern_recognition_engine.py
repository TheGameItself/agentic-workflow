"""
Pattern recognition engine with neural column simulation.

This lobe implements basic neural column-inspired processing with batch operations 
and proactive prompting. Inspired by research on neural columns in the brain.
"""

import json
import sqlite3
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import collections


class PatternRecognitionEngine:
    """Pattern recognition, neural column simulation, and related features.
    Implements basic neural column-inspired processing with batch operations and proactive prompting.
    """
    def __init__(self, db_path: Optional[str] = None):
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

    def recognize_patterns(self, data_batch: List[Any]) -> List[Dict[str, Any]]:
        """Recognize patterns in a batch of data using neural column-inspired approach."""
        recognized_patterns = []
        
        for i, data_item in enumerate(data_batch):
            # Extract basic patterns from data
            patterns = self._extract_basic_patterns(data_item)
            
            # Apply neural column processing
            column_processed = self._process_with_columns(patterns, f"item_{i}")
            
            # Store recognized patterns
            for pattern in column_processed:
                pattern_id = self._store_pattern(pattern)
                recognized_patterns.append({
                    'id': pattern_id,
                    'type': pattern['type'],
                    'data': pattern['data'],
                    'confidence': pattern['confidence'],
                    'source_item': i
                })
        
        return recognized_patterns

    def _extract_basic_patterns(self, data_item: Any) -> List[Dict[str, Any]]:
        """Extract basic patterns from a data item."""
        patterns = []
        
        if isinstance(data_item, str):
            # Text patterns
            patterns.extend(self._extract_text_patterns(data_item))
        elif isinstance(data_item, dict):
            # Dictionary patterns
            patterns.extend(self._extract_dict_patterns(data_item))
        elif isinstance(data_item, list):
            # List patterns
            patterns.extend(self._extract_list_patterns(data_item))
        elif isinstance(data_item, (int, float)):
            # Numeric patterns
            patterns.extend(self._extract_numeric_patterns(data_item))
        
        return patterns

    def _extract_text_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract patterns from text data."""
        patterns = []
        
        # Length pattern
        patterns.append({
            'type': 'text_length',
            'data': len(text),
            'confidence': 0.8
        })
        
        # Word count pattern
        word_count = len(text.split())
        patterns.append({
            'type': 'word_count',
            'data': word_count,
            'confidence': 0.9
        })
        
        # Common word patterns
        words = text.lower().split()
        if words:
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']
            common_count = sum(1 for word in words if word in common_words)
            patterns.append({
                'type': 'common_word_ratio',
                'data': common_count / len(words),
                'confidence': 0.7
            })
        
        # Special character patterns
        special_chars = sum(1 for char in text if not char.isalnum() and char != ' ')
        patterns.append({
            'type': 'special_char_ratio',
            'data': special_chars / len(text) if text else 0,
            'confidence': 0.6
        })
        
        return patterns

    def _extract_dict_patterns(self, data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from dictionary data."""
        patterns = []
        
        # Key count pattern
        patterns.append({
            'type': 'key_count',
            'data': len(data_dict),
            'confidence': 0.9
        })
        
        # Value type patterns
        type_counts = {}
        for value in data_dict.values():
            value_type = type(value).__name__
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        
        for value_type, count in type_counts.items():
            patterns.append({
                'type': f'value_type_{value_type}',
                'data': count,
                'confidence': 0.8
            })
        
        # Nested structure pattern
        nested_count = sum(1 for value in data_dict.values() 
                          if isinstance(value, (dict, list)))
        patterns.append({
            'type': 'nested_structure_ratio',
            'data': nested_count / len(data_dict) if data_dict else 0,
            'confidence': 0.7
        })
        
        return patterns

    def _extract_list_patterns(self, data_list: List[Any]) -> List[Dict[str, Any]]:
        """Extract patterns from list data."""
        patterns = []
        
        # Length pattern
        patterns.append({
            'type': 'list_length',
            'data': len(data_list),
            'confidence': 0.9
        })
        
        # Element type patterns
        if data_list:
            type_counts = {}
            for item in data_list:
                item_type = type(item).__name__
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            for item_type, count in type_counts.items():
                patterns.append({
                    'type': f'element_type_{item_type}',
                    'data': count,
                    'confidence': 0.8
                })
        
        return patterns

    def _extract_numeric_patterns(self, number: Union[int, float]) -> List[Dict[str, Any]]:
        """Extract patterns from numeric data."""
        patterns = []
        
        # Magnitude pattern
        magnitude = abs(number)
        if magnitude > 0:
            magnitude_order = len(str(int(magnitude)))
            patterns.append({
                'type': 'magnitude_order',
                'data': magnitude_order,
                'confidence': 0.9
            })
        
        # Sign pattern
        patterns.append({
            'type': 'sign',
            'data': 'positive' if number >= 0 else 'negative',
            'confidence': 1.0
        })
        
        # Integer vs float pattern
        patterns.append({
            'type': 'number_type',
            'data': 'integer' if isinstance(number, int) else 'float',
            'confidence': 1.0
        })
        
        return patterns

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
            patterns = self._extract_basic_patterns(data_item)
            processed = self._process_with_columns(patterns, f"item_{i}")
            
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
            patterns = self._extract_basic_patterns(data_item)
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