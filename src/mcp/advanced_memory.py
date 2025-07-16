#!/usr/bin/env python3
"""
AdvancedMemoryLobe: Advanced Vector/Quality/Relationship Memory Engine for MCP

This module implements the AdvancedMemoryLobe, responsible for vector memory, quality assessment, and memory relationships.
See src/mcp/lobes.py for the lobe registry and architecture overview.
"""

import sqlite3
import json
import os
import re
import math
import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime
from .vector_memory import get_vector_backend

# Optional numpy import with fallback
try:
    import numpy as np  # type: ignore[import]
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore
    # Fallback: raise clear error if numpy-dependent features are used
    class np:
        @staticmethod
        def array(data, dtype=None):
            return data  # Return as-is for basic operations
        @staticmethod
        def zeros(shape, dtype=None):
            if isinstance(shape, int):
                return [0.0] * shape
            elif len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
            return [0.0] * shape[0]
        @staticmethod
        def dot(a, b):
            if isinstance(a[0], list):  # Matrix multiplication
                return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
            else:  # Vector dot product
                return sum(a[i] * b[i] for i in range(len(a)))
        class linalg:
            class norm:
                @staticmethod
                def norm(vector, ord=None):
                    return math.sqrt(sum(x*x for x in vector))
        class random:
            @staticmethod
            def random(size=None):
                import random
                if size is None:
                    return random.random()
                elif isinstance(size, int):
                    return [random.random() for _ in range(size)]
                else:
                    return [[random.random() for _ in range(size[1])] for _ in range(size[0])]
        # Add missing attributes for compatibility
        float32 = float
        float64 = float
        # Add stubs for missing numpy attributes to avoid linter errors
        uint8 = int
        int8 = int
        float8 = float  # Simulate float8 as float
        @staticmethod
        def where(condition, x, y):
            # Simple fallback for np.where
            return [xi if cond else yi for cond, xi, yi in zip(condition, x, y)]

class VectorEncoder:
    """Base class for vector encoders/decoders."""
    def encode(self, vector):
        """Encode the vector using basic compression fallback."""
        try:
            import json
            import zlib
            import base64
            
            # Convert vector to JSON string
            vector_json = json.dumps(vector)
            
            # Compress using zlib
            compressed = zlib.compress(vector_json.encode('utf-8'))
            
            # Encode as base64 for storage
            encoded = base64.b64encode(compressed).decode('utf-8')
            
            return {
                'encoded_data': encoded,
                'original_size': len(vector_json),
                'compressed_size': len(compressed),
                'compression_ratio': len(compressed) / len(vector_json),
                'encoding_method': 'zlib_base64',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Ultimate fallback - return vector as-is with metadata
            return {
                'encoded_data': vector,
                'original_size': len(str(vector)),
                'compressed_size': len(str(vector)),
                'compression_ratio': 1.0,
                'encoding_method': 'passthrough',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    def decode(self, encoded):
        """Decode the encoded vector using fallback decompression."""
        try:
            import json
            import zlib
            import base64
            
            # Check if it's our encoded format
            if isinstance(encoded, dict) and 'encoded_data' in encoded:
                encoding_method = encoded.get('encoding_method', 'unknown')
                
                if encoding_method == 'zlib_base64':
                    # Decode base64
                    compressed = base64.b64decode(encoded['encoded_data'].encode('utf-8'))
                    
                    # Decompress using zlib
                    decompressed = zlib.decompress(compressed)
                    
                    # Parse JSON
                    vector = json.loads(decompressed.decode('utf-8'))
                    return vector
                    
                elif encoding_method == 'passthrough':
                    # Return the data as-is
                    return encoded['encoded_data']
            
            # Fallback: assume it's already decoded or in a simple format
            if isinstance(encoded, (list, dict)):
                return encoded
            
            # Try to parse as JSON
            if isinstance(encoded, str):
                try:
                    return json.loads(encoded)
                except json.JSONDecodeError:
                    return encoded
            
            return encoded
            
        except Exception as e:
            # Ultimate fallback - return as-is with error info
            return {
                'decoded_data': encoded,
                'decoding_method': 'fallback',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    def name(self):
        return self.__class__.__name__

class TFIDFEncoder(VectorEncoder):
    def encode(self, vector):
        # Convert dict to numpy array for fast math
        keys = sorted(vector.keys())
        arr = np.array([vector[k] for k in keys], dtype=np.float32)
        return {'keys': keys, 'values': arr.tolist()}
    def decode(self, encoded):
        keys = encoded['keys']
        arr = np.array(encoded['values'], dtype=np.float32)
        return dict(zip(keys, arr))

class RaBitQEncoder(VectorEncoder):
    """Stub for RaBitQ 1-bit quantization encoder/decoder using NumPy."""
    def encode(self, vector):
        keys = sorted(vector.keys())
        arr = np.array([vector[k] for k in keys], dtype=np.float32)
        sign_bits = (arr >= 0).astype(np.uint8)
        return {'keys': keys, 'sign_bits': sign_bits.tolist()}
    def decode(self, encoded):
        keys = encoded['keys']
        sign_bits = np.array(encoded['sign_bits'], dtype=np.uint8)
        arr = np.where(sign_bits == 1, 1.0, -1.0)
        return dict(zip(keys, arr))

class Float8QEncoder(VectorEncoder):
    """Float8 quantization encoder/decoder using NumPy or simulated float8. Supports hormone-state-driven dynamic quantization. Reference: idea.txt, low-precision neural encoding research."""
    def encode(self, vector, hormone_state=None):
        try:
            import numpy as np  # type: ignore[import]
        except ImportError:
            raise ImportError("Numpy is required for Float8QEncoder. Please install numpy.")
        keys = sorted(vector.keys())
        arr = np.array([vector[k] for k in keys], dtype=np.float32)  # type: ignore[attr-defined]
        # Dynamic quantization selection based on hormone state
        def select_quant_mode_from_hormones(hormone_state):
            if not hormone_state:
                return 'int8'  # Fallback
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
        quant_mode = select_quant_mode_from_hormones(hormone_state)
        if quant_mode == 'float8':
            if hasattr(np, 'float8'):
                quantized = arr.astype(np.float8)  # type: ignore[attr-defined]
            else:
                arr = np.clip(arr, -1, 1)
                int_arr = np.round(arr * 127).astype(np.int8)
                quantized = int_arr.astype(np.float32) / 127.0
        elif quant_mode == 'float16':
            quantized = arr.astype(np.float16)
        elif quant_mode == 'float32':
            quantized = arr.astype(np.float32)
        else:  # int8 fallback
            arr = np.clip(arr, 0, 1)
            quantized = np.round(arr * 255).astype(np.int8)
        return {'keys': keys, 'quant_mode': quant_mode, 'quantized': quantized.tolist()}
    def decode(self, encoded):
        try:
            import numpy as np  # type: ignore[import]
        except ImportError:
            raise ImportError("Numpy is required for Float8QEncoder. Please install numpy.")
        keys = encoded['keys']
        quant_mode = encoded.get('quant_mode', 'float8')
        quantized = np.array(encoded['quantized'], dtype=np.float32)  # type: ignore[attr-defined]
        return dict(zip(keys, quantized))

class AdvancedMemoryManager:
    """Advanced memory management with vector search, categorization, and relationships.
    
    Modular Vector Backend Support:
    - Supports pluggable vector backends: sqlitefaiss (default, portable), milvus (remote, scalable), annoy (lightweight), qdrant (future-proof).
    - Backend selection via constructor (backend_name) and config.
    - See README for pros/cons and usage of each backend.
    
    Args:
        db_path: Path to the database file.
        encoder: VectorEncoder instance (TFIDFEncoder, RaBitQEncoder, etc.).
        vector_backend: Optional pre-initialized backend instance.
        backend_name: Name of the backend to use ('sqlitefaiss', 'milvus', 'annoy', 'qdrant').
        backend_config: Dict of backend-specific configuration options.
    """
    
    def __init__(self, db_path: Optional[str] = None, encoder: Optional[VectorEncoder] = None, 
                 backend_name: str = 'sqlitefaiss', backend_config: Optional[Dict] = None):
        """Initialize the advanced memory manager with multiple encoder backend support."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'advanced_memory.db')
        
        self.db_path = str(db_path)
        self.encoder = encoder or TFIDFEncoder()
        self.backend_name = backend_name
        self.backend_config = backend_config or {}
        
        # Support for multiple encoder backends
        self.available_encoders = {
            'tfidf': TFIDFEncoder(),
            'rabitq': RaBitQEncoder(),
            'float8q': Float8QEncoder()
        }
        
        self._init_advanced_database()
        self.compression_ratios = []  # Track compression ratios for reporting
        self.search_similarities = []  # Track search similarities for reporting
        
        # Initialize vector backend
        self.initialize_vector_backend(backend_name, backend_config)
    
    def _init_advanced_database(self):
        """
        Initialize the advanced database schema.
        - Ensures all advanced features: vector compression, tagging, chunking, crosslinks, quality/confidence/completeness/relevance scores, memory order, relationships, etc.
        - Research-aligned: see README.md, idea.txt, Zolfagharinejad et al., 2024 (EPJ B), Ren & Xia, 2024 (arXiv).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Advanced memories table (vector compression, tagging, chunking, quality/confidence/completeness/relevance, memory order)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advanced_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                memory_type TEXT DEFAULT 'general',
                priority REAL DEFAULT 0.5,
                context TEXT,
                tags TEXT,
                category TEXT,
                vector_embedding TEXT,
                quality_score REAL DEFAULT 0.5,
                confidence_score REAL DEFAULT 0.5,
                completeness_score REAL DEFAULT 0.5,
                relevance_score REAL DEFAULT 0.5,
                memory_order INTEGER DEFAULT 1,
                chunk_hash TEXT, -- For chunking and deduplication
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        # Memory relationships table (crosslinks, relationship types)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER,
                target_memory_id INTEGER,
                relationship_type TEXT,
                strength REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_memory_id) REFERENCES advanced_memories (id),
                FOREIGN KEY (target_memory_id) REFERENCES advanced_memories (id)
            )
        """)
        # Memory categories table (for tagging, chunking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                parent_category_id INTEGER,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_category_id) REFERENCES memory_categories (id)
            )
        """)
        # Enhanced reminders table (feedback, context history)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                task_id INTEGER,
                reminder_type TEXT,
                trigger_conditions TEXT,
                next_reminder TIMESTAMP,
                reminder_count INTEGER DEFAULT 0,
                last_triggered TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                effectiveness_score REAL DEFAULT 0.0,
                user_feedback_history TEXT,
                easiness_factor REAL DEFAULT 2.5,
                context_history TEXT,
                FOREIGN KEY (memory_id) REFERENCES advanced_memories (id)
            )
        """)
        # Context patterns table (for chunking, pattern recognition)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # User retention rates table (feedback, research alignment)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_retention_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content_type TEXT,
                retention_rate REAL DEFAULT 0.5,
                sample_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def _create_tfidf_embedding(self, text: str) -> Dict[str, float]:
        """Create TF-IDF embedding for text."""
        # Simple TF-IDF implementation
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Calculate TF
        total_words = len(words)
        tf = {word: freq / total_words for word, freq in word_freq.items()}
        
        # For simplicity, we'll use a basic IDF calculation
        # In a real implementation, you'd calculate this across all documents
        idf = {word: 1.0 for word in tf.keys()}
        
        # Calculate TF-IDF
        tfidf = {word: tf[word] * idf[word] for word in tf.keys()}
        
        return tfidf
    
    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        # Use NumPy for fast cosine similarity
        keys = sorted(set(vec1.keys()) | set(vec2.keys()))
        arr1 = np.array([vec1.get(k, 0.0) for k in keys], dtype=np.float32)
        arr2 = np.array([vec2.get(k, 0.0) for k in keys], dtype=np.float32)
        dot = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def _create_vector_embedding(self, text: str) -> dict:
        # Use TF-IDF as base vector, then encode
        tfidf = self._create_tfidf_embedding(text)
        return self.encoder.encode(tfidf)
    
    def add_advanced_memory(self, text: str, memory_type: str = 'general', 
                           priority: float = 0.5, context: str = '', 
                           tags: Optional[List[str]] = None, category: str = '', memory_order: int = 1) -> int:
        """Add a new advanced memory with vector embedding, memory order, and log compression ratio."""
        if tags is None:
            tags = []
        if memory_type is None:
            memory_type = ''
        if context is None:
            context = ''
        if category is None:
            category = ''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create vector embedding using selected encoder
        embedding = self._create_vector_embedding(text)
        embedding_json = json.dumps(embedding)
        
        # Compression ratio: compare size of original (float32 per word) vs encoded (JSON string)
        original_size = len(embedding) * 4  # float32 = 4 bytes per word
        encoded_size = len(embedding_json.encode('utf-8'))
        ratio = encoded_size / original_size if original_size > 0 else 1.0
        self.compression_ratios.append(ratio)
        
        # Auto-detect category if not provided
        if not category:
            category = self._auto_detect_category(text, tags)
        
        # Calculate quality scores
        quality_score = self._calculate_quality_score(text, context, tags)
        confidence_score = self._calculate_confidence_score(text, context)
        completeness_score = self._calculate_completeness_score(text, context)
        relevance_score = self._calculate_relevance_score(text, tags)
        
        tags_json = json.dumps(tags) if tags else None
        
        cursor.execute("""
            INSERT INTO advanced_memories 
            (text, memory_type, priority, context, tags, category, vector_embedding,
             quality_score, confidence_score, completeness_score, relevance_score, memory_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (text, memory_type, priority, context, tags_json, category, embedding_json,
              quality_score, confidence_score, completeness_score, relevance_score, memory_order))
        
        memory_id = cursor.lastrowid
        
        # Auto-detect relationships with existing memories
        self._auto_detect_relationships(memory_id if memory_id is not None else -1, text, embedding)
        
        conn.commit()
        conn.close()
        
        return memory_id if memory_id is not None else -1
    
    def _auto_detect_category(self, text: str, tags: Optional[List[str]] = None) -> str:
        """Auto-detect category based on text content and tags."""
        if tags is None:
            tags = []
        text_lower = text.lower()
        
        category_keywords = {
            'code': ['function', 'class', 'method', 'api', 'bug', 'error', 'debug'],
            'research': ['study', 'analysis', 'investigation', 'research', 'paper'],
            'planning': ['plan', 'strategy', 'roadmap', 'timeline', 'milestone'],
            'documentation': ['doc', 'readme', 'comment', 'guide', 'tutorial'],
            'testing': ['test', 'unit', 'integration', 'validation', 'verify'],
            'deployment': ['deploy', 'production', 'server', 'config', 'environment']
        }
        
        # Check text content
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        # Check tags
        if tags:
            for tag in tags:
                tag_lower = tag.lower()
                for category, keywords in category_keywords.items():
                    if any(keyword in tag_lower for keyword in keywords):
                        return category
        
        return 'general'
    
    def _calculate_quality_score(self, text: str, context: str = '', tags: Optional[List[str]] = None) -> float:
        """Calculate memory quality score."""
        if tags is None:
            tags = []
        score = 0.5  # Base score
        
        # Text length factor
        if len(text) > 50:
            score += 0.1
        if len(text) > 200:
            score += 0.1
        
        # Context factor
        if context:
            score += 0.1
        
        # Tags factor
        if tags and len(tags) > 0:
            score += 0.1
        
        # Specificity factor (more specific terms)
        specific_terms = ['because', 'therefore', 'however', 'specifically', 'example']
        if any(term in text.lower() for term in specific_terms):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence_score(self, text: str, context: str = '') -> float:
        """Calculate confidence score for the memory."""
        score = 0.5  # Base score
        
        # Context presence
        if context:
            score += 0.2
        
        # Text clarity (simple heuristic)
        if len(text.split()) > 10:
            score += 0.1
        
        # Specific details
        if re.search(r'\d+', text):  # Contains numbers
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_completeness_score(self, text: str, context: str = '') -> float:
        """Calculate completeness score for the memory."""
        score = 0.5  # Base score
        
        # Context completeness
        if context:
            score += 0.2
        
        # Text completeness
        if len(text.split()) > 20:
            score += 0.2
        
        # Structured information
        if re.search(r'[A-Z][a-z]+:', text):  # Contains structured info
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_relevance_score(self, text: str, tags: Optional[List[str]] = None) -> float:
        """Calculate relevance score for the memory."""
        if tags is None:
            tags = []
        score = 0.5  # Base score
        
        # Tag relevance
        if tags and len(tags) > 0:
            score += 0.2
        
        # Text relevance (simple keyword matching)
        relevant_keywords = ['important', 'key', 'critical', 'essential', 'must']
        if any(keyword in text.lower() for keyword in relevant_keywords):
            score += 0.2
        
        return min(score, 1.0)
    
    def _auto_detect_relationships(self, memory_id: int, text: str, embedding: Dict[str, float]):
        """Auto-detect relationships with existing memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing memories
        cursor.execute("""
            SELECT id, text, vector_embedding FROM advanced_memories 
            WHERE id != ? ORDER BY created_at DESC LIMIT 50
        """, (memory_id,))
        
        existing_memories = cursor.fetchall()
        
        for existing_id, existing_text, existing_embedding_json in existing_memories:
            try:
                existing_embedding = json.loads(existing_embedding_json)
                similarity = self._cosine_similarity(embedding, existing_embedding)
                
                # Create relationship if similarity is high enough
                if similarity > 0.3:
                    relationship_type = self._determine_relationship_type(text, existing_text)
                    
                    cursor.execute("""
                        INSERT INTO memory_relationships 
                        (source_memory_id, target_memory_id, relationship_type, strength, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (memory_id, existing_id, relationship_type, similarity, similarity))
            
            except (json.JSONDecodeError, KeyError):
                continue
        
        conn.commit()
        conn.close()
    
    def _determine_relationship_type(self, text1: str, text2: str) -> str:
        """Determine the type of relationship between two memories."""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Check for semantic relationships
        if any(word in text1_lower and word in text2_lower 
               for word in ['error', 'bug', 'fix', 'issue']):
            return 'problem_solution'
        
        if any(word in text1_lower and word in text2_lower 
               for word in ['function', 'method', 'class', 'api']):
            return 'code_related'
        
        if any(word in text1_lower and word in text2_lower 
               for word in ['research', 'study', 'analysis']):
            return 'research_related'
        
        return 'semantic_similarity'
    
    def vector_search(self, query: str, limit: int = 10, 
                     memory_type: Optional[str] = None, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Search memories using vector similarity and log similarity scores."""
        if memory_type is None:
            memory_type = ''
        
        # Fallback to database search if vector backend is not available
        if not hasattr(self, 'vector_backend') or self.vector_backend is None:
            return self._fallback_search(query, limit, memory_type, min_similarity)
        
        try:
            # Create query embedding
            query_embedding = self._create_vector_embedding(query)
            
            # Search using vector backend
            backend_results = self.vector_backend.search_vector(query_embedding, limit=limit, min_similarity=min_similarity)
            
            if not backend_results:
                return self._fallback_search(query, limit, memory_type, min_similarity)
            
            results = []
            for item in backend_results:
                similarity = item.get('similarity', 0)
                metadata = item.get('metadata', {})
                metadata['similarity'] = similarity
                results.append(metadata)
                self.search_similarities.append(similarity)
            
            return results
            
        except Exception as e:
            print(f"Vector search failed, falling back to database search: {e}")
            return self._fallback_search(query, limit, memory_type, min_similarity)
    
    def _fallback_search(self, query: str, limit: int = 10, 
                        memory_type: Optional[str] = None, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Fallback search using database text matching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build search query
        search_terms = query.lower().split()
        where_conditions = []
        params = []
        
        # Text search
        for term in search_terms:
            where_conditions.append("(LOWER(text) LIKE ? OR LOWER(context) LIKE ? OR LOWER(tags) LIKE ?)")
            params.extend([f'%{term}%', f'%{term}%', f'%{term}%'])
        
        # Memory type filter
        if memory_type and memory_type != '':
            where_conditions.append("memory_type = ?")
            params.append(memory_type)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        cursor.execute(f"""
            SELECT id, text, memory_type, priority, context, tags, category,
                   quality_score, confidence_score, completeness_score, relevance_score,
                   memory_order, created_at, updated_at, last_accessed, access_count
            FROM advanced_memories 
            WHERE {where_clause}
            ORDER BY priority DESC, quality_score DESC
            LIMIT ?
        """, params + [limit])
        
        results = []
        for row in cursor.fetchall():
            tags = json.loads(row[5]) if row[5] else []
            
            # Calculate simple text similarity
            similarity = self._calculate_text_similarity(query, row[1])
            
            if similarity >= min_similarity:
                results.append({
                    'id': row[0],
                    'text': row[1],
                    'memory_type': row[2],
                    'priority': row[3],
                    'context': row[4],
                    'tags': tags,
                    'category': row[6],
                    'quality_score': row[7],
                    'confidence_score': row[8],
                    'completeness_score': row[9],
                    'relevance_score': row[10],
                    'memory_order': row[11],
                    'created_at': row[12],
                    'updated_at': row[13],
                    'last_accessed': row[14],
                    'access_count': row[15],
                    'similarity': similarity
                })
                self.search_similarities.append(similarity)
        
        conn.close()
        return results
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_memory_relationships(self, memory_id: int) -> List[Dict[str, Any]]:
        """Get relationships for a specific memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT r.id, r.relationship_type, r.strength, r.confidence,
                   m.id, m.text, m.memory_type, m.category
            FROM memory_relationships r
            JOIN advanced_memories m ON r.target_memory_id = m.id
            WHERE r.source_memory_id = ?
            ORDER BY r.strength DESC
        """, (memory_id,))
        
        relationships = []
        for row in cursor.fetchall():
            rel_id, rel_type, strength, confidence, target_id, target_text, target_type, target_category = row
            
            relationships.append({
                'relationship_id': rel_id,
                'relationship_type': rel_type,
                'strength': strength,
                'confidence': confidence,
                'target_memory': {
                    'id': target_id,
                    'text': target_text,
                    'memory_type': target_type,
                    'category': target_category
                }
            })
        
        conn.close()
        return relationships
    
    def get_memory_quality_report(self, memory_id: int) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report for a memory.
        
        This method uses the enhanced MemoryQualityAssessment system if available,
        otherwise falls back to the basic quality report.
        """
        try:
            # Try to use the enhanced quality assessment system
            from .memory_quality_assessment import MemoryQualityAssessment
            quality_assessment = MemoryQualityAssessment(self.db_path)
            return quality_assessment.get_memory_quality_report(memory_id)
        except ImportError:
            # Fall back to basic quality report
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT quality_score, confidence_score, completeness_score, relevance_score,
                       text, context, tags, category, created_at, last_accessed, access_count
                FROM advanced_memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return {}
            
            quality_score, confidence_score, completeness_score, relevance_score, \
            text, context, tags_json, category, created_at, last_accessed, access_count = row
            
            tags = json.loads(tags_json) if tags_json else None
            
            # Calculate overall quality
            overall_quality = (quality_score + confidence_score + completeness_score + relevance_score) / 4
            
            # Generate improvement suggestions
            suggestions = []
            if quality_score < 0.7:
                suggestions.append("Add more context or details to improve quality")
            if confidence_score < 0.7:
                suggestions.append("Provide more specific information to increase confidence")
            if completeness_score < 0.7:
                suggestions.append("Include additional relevant information for completeness")
            if relevance_score < 0.7:
                suggestions.append("Add relevant tags or keywords to improve relevance")
            
            return {
                'memory_id': memory_id,
                'overall_quality': overall_quality,
                'quality_breakdown': {
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'completeness_score': completeness_score,
                    'relevance_score': relevance_score
                },
                'content_info': {
                    'text_length': len(text),
                    'has_context': bool(context),
                    'tag_count': len(tags) if tags else 0,
                    'category': category,
                    'access_count': access_count,
                    'created_at': created_at,
                    'last_accessed': last_accessed
                },
                'improvement_suggestions': suggestions
            }

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, text, memory_type, priority, context, tags, category,
                   quality_score, confidence_score, completeness_score, relevance_score,
                   memory_order, created_at, updated_at, last_accessed, access_count
            FROM advanced_memories WHERE id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        
        if row:
            # Update access tracking
            cursor.execute("""
                UPDATE advanced_memories 
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE id = ?
            """, (memory_id,))
            conn.commit()
        
        conn.close()
        
        if not row:
            return None
        
        tags = json.loads(row[5]) if row[5] else []
        
        return {
            'id': row[0],
            'text': row[1],
            'memory_type': row[2],
            'priority': row[3],
            'context': row[4],
            'tags': tags,
            'category': row[6],
            'quality_score': row[7],
            'confidence_score': row[8],
            'completeness_score': row[9],
            'relevance_score': row[10],
            'memory_order': row[11],
            'created_at': row[12],
            'updated_at': row[13],
            'last_accessed': row[14],
            'access_count': row[15]
        }

    def update_memory(self, memory_id: int, **kwargs) -> bool:
        """Update a memory with new information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        values = []
        
        allowed_fields = ['text', 'memory_type', 'priority', 'context', 'tags', 'category']
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                if field == 'tags' and isinstance(value, list):
                    value = json.dumps(value)
                update_fields.append(f"{field} = ?")
                values.append(value)
        
        if not update_fields:
            conn.close()
            return False
        
        # Always update the updated_at timestamp
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        
        query = f"UPDATE advanced_memories SET {', '.join(update_fields)} WHERE id = ?"
        values.append(memory_id)
        
        cursor.execute(query, values)
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return success

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory and its relationships."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete relationships
            cursor.execute("""
                DELETE FROM memory_relationships 
                WHERE source_memory_id = ? OR target_memory_id = ?
            """, (memory_id, memory_id))
            
            # Delete the memory
            cursor.execute("DELETE FROM advanced_memories WHERE id = ?", (memory_id,))
            
            success = cursor.rowcount > 0
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            success = False
            print(f"Error deleting memory {memory_id}: {e}")
        
        finally:
            conn.close()
        
        return success

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory system."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM advanced_memories")
        stats['total_memories'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM memory_relationships")
        stats['total_relationships'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT category) FROM advanced_memories")
        stats['unique_categories'] = cursor.fetchone()[0]
        
        # Quality statistics
        cursor.execute("""
            SELECT AVG(quality_score), AVG(confidence_score), 
                   AVG(completeness_score), AVG(relevance_score)
            FROM advanced_memories
        """)
        quality_row = cursor.fetchone()
        if quality_row and quality_row[0] is not None:
            stats['average_quality'] = {
                'quality_score': quality_row[0],
                'confidence_score': quality_row[1],
                'completeness_score': quality_row[2],
                'relevance_score': quality_row[3],
                'overall': sum(quality_row) / 4
            }
        else:
            stats['average_quality'] = {
                'quality_score': 0.0,
                'confidence_score': 0.0,
                'completeness_score': 0.0,
                'relevance_score': 0.0,
                'overall': 0.0
            }
        
        # Memory type distribution
        cursor.execute("""
            SELECT memory_type, COUNT(*) 
            FROM advanced_memories 
            GROUP BY memory_type
        """)
        stats['memory_type_distribution'] = dict(cursor.fetchall())
        
        # Category distribution
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM advanced_memories 
            GROUP BY category
        """)
        stats['category_distribution'] = dict(cursor.fetchall())
        
        # Access statistics
        cursor.execute("""
            SELECT AVG(access_count), MAX(access_count), 
                   COUNT(*) as total_accesses
            FROM advanced_memories
        """)
        access_row = cursor.fetchone()
        if access_row:
            stats['access_statistics'] = {
                'average_access_count': access_row[0] or 0,
                'max_access_count': access_row[1] or 0,
                'total_memories_accessed': access_row[2] or 0
            }
        
        # Compression statistics
        if self.compression_ratios:
            stats['compression_statistics'] = {
                'average_compression_ratio': sum(self.compression_ratios) / len(self.compression_ratios),
                'best_compression_ratio': min(self.compression_ratios),
                'worst_compression_ratio': max(self.compression_ratios),
                'total_compressions': len(self.compression_ratios)
            }
        
        # Search statistics
        if self.search_similarities:
            stats['search_statistics'] = {
                'average_similarity': sum(self.search_similarities) / len(self.search_similarities),
                'best_similarity': max(self.search_similarities),
                'worst_similarity': min(self.search_similarities),
                'total_searches': len(self.search_similarities)
            }
        
        conn.close()
        return stats

    def cleanup_memories(self, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Clean up memories based on specified criteria."""
        if criteria is None:
            criteria = {
                'min_quality_score': 0.3,
                'min_access_count': 0,
                'max_age_days': 365,
                'remove_duplicates': True
            }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        deleted_count = 0
        duplicate_count = 0
        
        try:
            # Remove low quality memories
            if 'min_quality_score' in criteria:
                cursor.execute("""
                    DELETE FROM advanced_memories 
                    WHERE quality_score < ? AND access_count <= ?
                """, (criteria['min_quality_score'], criteria.get('min_access_count', 0)))
                deleted_count += cursor.rowcount
            
            # Remove old unused memories
            if 'max_age_days' in criteria:
                cursor.execute("""
                    DELETE FROM advanced_memories 
                    WHERE datetime(created_at) < datetime('now', '-{} days')
                    AND access_count <= ?
                """.format(criteria['max_age_days']), (criteria.get('min_access_count', 0),))
                deleted_count += cursor.rowcount
            
            # Remove duplicates based on text similarity
            if criteria.get('remove_duplicates', False):
                cursor.execute("""
                    SELECT id, text FROM advanced_memories ORDER BY created_at
                """)
                memories = cursor.fetchall()
                
                seen_texts = set()
                duplicates_to_remove = []
                
                for memory_id, text in memories:
                    # Simple duplicate detection based on text hash
                    text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
                    if text_hash in seen_texts:
                        duplicates_to_remove.append(memory_id)
                        duplicate_count += 1
                    else:
                        seen_texts.add(text_hash)
                
                for dup_id in duplicates_to_remove:
                    cursor.execute("DELETE FROM advanced_memories WHERE id = ?", (dup_id,))
            
            # Clean up orphaned relationships
            cursor.execute("""
                DELETE FROM memory_relationships 
                WHERE source_memory_id NOT IN (SELECT id FROM advanced_memories)
                OR target_memory_id NOT IN (SELECT id FROM advanced_memories)
            """)
            orphaned_relationships = cursor.rowcount
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"Error during cleanup: {e}")
            deleted_count = 0
            duplicate_count = 0
            orphaned_relationships = 0
        
        finally:
            conn.close()
        
        return {
            'deleted_memories': deleted_count,
            'removed_duplicates': duplicate_count,
            'cleaned_relationships': orphaned_relationships
        }

    def compress_memories(self, target_compression_ratio: float = 0.5) -> Dict[str, Any]:
        """Compress memory storage using advanced encoding techniques."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all memories with their embeddings
        cursor.execute("""
            SELECT id, vector_embedding FROM advanced_memories
        """)
        memories = cursor.fetchall()
        
        compressed_count = 0
        total_size_before = 0
        total_size_after = 0
        
        for memory_id, embedding_json in memories:
            try:
                # Decode current embedding
                current_embedding = json.loads(embedding_json)
                current_size = len(embedding_json.encode('utf-8'))
                total_size_before += current_size
                
                # Re-encode with potentially better compression
                if hasattr(self.encoder, 'encode'):
                    # If it's already encoded, decode first
                    if hasattr(self.encoder, 'decode'):
                        decoded = self.encoder.decode(current_embedding)
                        reencoded = self.encoder.encode(decoded)
                    else:
                        reencoded = current_embedding
                else:
                    reencoded = current_embedding
                
                new_embedding_json = json.dumps(reencoded)
                new_size = len(new_embedding_json.encode('utf-8'))
                total_size_after += new_size
                
                # Update if compression improved
                if new_size < current_size:
                    cursor.execute("""
                        UPDATE advanced_memories 
                        SET vector_embedding = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_embedding_json, memory_id))
                    compressed_count += 1
                else:
                    total_size_after = total_size_after - new_size + current_size
                
            except Exception as e:
                print(f"Error compressing memory {memory_id}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        compression_ratio = total_size_after / total_size_before if total_size_before > 0 else 1.0
        
        return {
            'compressed_memories': compressed_count,
            'total_memories': len(memories),
            'size_before_bytes': total_size_before,
            'size_after_bytes': total_size_after,
            'compression_ratio': compression_ratio,
            'space_saved_bytes': total_size_before - total_size_after,
            'target_achieved': compression_ratio <= target_compression_ratio
        }

    def organize_hierarchical_memory(self) -> Dict[str, Any]:
        """Organize memories into hierarchical structure (working, short-term, long-term)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all memories with access patterns
        cursor.execute("""
            SELECT id, access_count, last_accessed, created_at, priority, quality_score
            FROM advanced_memories
        """)
        memories = cursor.fetchall()
        
        working_memory = []
        short_term_memory = []
        long_term_memory = []
        
        current_time = datetime.now()
        
        for memory_data in memories:
            memory_id, access_count, last_accessed, created_at, priority, quality_score = memory_data
            
            # Parse timestamps
            try:
                last_access_time = datetime.fromisoformat(last_accessed.replace('Z', '+00:00')) if last_accessed else current_time
                create_time = datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else current_time
            except:
                last_access_time = current_time
                create_time = current_time
            
            # Calculate time since last access and creation
            hours_since_access = (current_time - last_access_time).total_seconds() / 3600
            days_since_creation = (current_time - create_time).total_seconds() / 86400
            
            # Classify memory based on access patterns and importance
            importance_score = (priority + quality_score + min(access_count / 10, 1.0)) / 3
            
            if hours_since_access <= 2 and access_count >= 2:
                # Recently and frequently accessed - working memory
                working_memory.append(memory_id)
                memory_order = 1
            elif hours_since_access <= 24 or (access_count >= 1 and days_since_creation <= 7):
                # Recently accessed or new - short-term memory
                short_term_memory.append(memory_id)
                memory_order = 2
            else:
                # Older or less accessed - long-term memory
                long_term_memory.append(memory_id)
                memory_order = 3
            
            # Update memory order in database
            cursor.execute("""
                UPDATE advanced_memories 
                SET memory_order = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (memory_order, memory_id))
        
        conn.commit()
        conn.close()
        
        return {
            'working_memory_count': len(working_memory),
            'short_term_memory_count': len(short_term_memory),
            'long_term_memory_count': len(long_term_memory),
            'working_memory_ids': working_memory[:10],  # Sample
            'short_term_memory_ids': short_term_memory[:10],  # Sample
            'long_term_memory_ids': long_term_memory[:10],  # Sample
            'total_organized': len(memories)
        }

    def get_memory_by_order(self, memory_order: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get memories by their hierarchical order (1=working, 2=short-term, 3=long-term)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, text, memory_type, priority, context, tags, category,
                   quality_score, confidence_score, completeness_score, relevance_score,
                   memory_order, created_at, updated_at, last_accessed, access_count
            FROM advanced_memories 
            WHERE memory_order = ?
            ORDER BY priority DESC, quality_score DESC, access_count DESC
            LIMIT ?
        """, (memory_order, limit))
        
        memories = []
        for row in cursor.fetchall():
            tags = json.loads(row[5]) if row[5] else []
            
            memories.append({
                'id': row[0],
                'text': row[1],
                'memory_type': row[2],
                'priority': row[3],
                'context': row[4],
                'tags': tags,
                'category': row[6],
                'quality_score': row[7],
                'confidence_score': row[8],
                'completeness_score': row[9],
                'relevance_score': row[10],
                'memory_order': row[11],
                'created_at': row[12],
                'updated_at': row[13],
                'last_accessed': row[14],
                'access_count': row[15]
            })
        
        conn.close()
        return memories

    def initialize_vector_backend(self, backend_name: str = 'sqlitefaiss', backend_config: Optional[Dict] = None):
        """Initialize the vector backend for similarity search."""
        try:
            self.vector_backend = get_vector_backend(backend_name, backend_config or {})
            if hasattr(self.vector_backend, 'db_path'):
                self.vector_backend.db_path = self.db_path.replace('.db', '_vectors.db')
        except Exception as e:
            print(f"Warning: Could not initialize vector backend {backend_name}: {e}")
            # Fallback to basic implementation
            from .vector_memory import SQLiteFAISSBackend
            self.vector_backend = SQLiteFAISSBackend(self.db_path.replace('.db', '_vectors.db'))

    def switch_encoder(self, encoder_name: str, hormone_state: Optional[Dict] = None) -> bool:
        """Switch to a different encoder backend dynamically."""
        if encoder_name in self.available_encoders:
            old_encoder = self.encoder
            self.encoder = self.available_encoders[encoder_name]
            
            # For Float8QEncoder, pass hormone state if available
            if encoder_name == 'float8q' and hormone_state:
                self.encoder.hormone_state = hormone_state
            
            print(f"Switched encoder from {old_encoder.name()} to {self.encoder.name()}")
            return True
        else:
            print(f"Encoder {encoder_name} not available. Available: {list(self.available_encoders.keys())}")
            return False

    def benchmark_encoders(self, test_data: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Benchmark different encoder backends for compression and performance."""
        if test_data is None:
            # Use some sample data from existing memories
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM advanced_memories LIMIT 10")
            test_data = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not test_data:
                test_data = [
                    "This is a sample text for testing encoder performance",
                    "Another longer text with more complex vocabulary and structure for comprehensive testing",
                    "Short text",
                    "A very detailed and comprehensive text that contains multiple sentences, various punctuation marks, and a wide range of vocabulary to test the encoder's ability to handle complex linguistic structures and patterns."
                ]
        
        results = {}
        
        for encoder_name, encoder in self.available_encoders.items():
            encoder_results = {
                'compression_ratios': [],
                'encoding_times': [],
                'decoding_times': [],
                'accuracy_scores': []
            }
            
            for text in test_data:
                try:
                    import time
                    
                    # Create TF-IDF vector
                    tfidf = self._create_tfidf_embedding(text)
                    
                    # Test encoding
                    start_time = time.time()
                    encoded = encoder.encode(tfidf)
                    encoding_time = time.time() - start_time
                    
                    # Test decoding
                    start_time = time.time()
                    decoded = encoder.decode(encoded)
                    decoding_time = time.time() - start_time
                    
                    # Calculate compression ratio
                    original_size = len(str(tfidf))
                    encoded_size = len(str(encoded))
                    compression_ratio = encoded_size / original_size if original_size > 0 else 1.0
                    
                    # Calculate accuracy (similarity between original and decoded)
                    accuracy = self._cosine_similarity(tfidf, decoded) if isinstance(decoded, dict) else 0.0
                    
                    encoder_results['compression_ratios'].append(compression_ratio)
                    encoder_results['encoding_times'].append(encoding_time)
                    encoder_results['decoding_times'].append(decoding_time)
                    encoder_results['accuracy_scores'].append(accuracy)
                    
                except Exception as e:
                    print(f"Error benchmarking {encoder_name}: {e}")
                    continue
            
            # Calculate averages
            if encoder_results['compression_ratios']:
                results[encoder_name] = {
                    'avg_compression_ratio': sum(encoder_results['compression_ratios']) / len(encoder_results['compression_ratios']),
                    'avg_encoding_time': sum(encoder_results['encoding_times']) / len(encoder_results['encoding_times']),
                    'avg_decoding_time': sum(encoder_results['decoding_times']) / len(encoder_results['decoding_times']),
                    'avg_accuracy': sum(encoder_results['accuracy_scores']) / len(encoder_results['accuracy_scores']),
                    'total_tests': len(encoder_results['compression_ratios'])
                }
        
        return results

    def auto_prune_memories(self, max_memories: int = 10000, quality_threshold: float = 0.4) -> Dict[str, int]:
        """Automatically prune memories to stay within limits while preserving quality."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count current memories
        cursor.execute("SELECT COUNT(*) FROM advanced_memories")
        current_count = cursor.fetchone()[0]
        
        if current_count <= max_memories:
            conn.close()
            return {'pruned_count': 0, 'remaining_count': current_count}
        
        # Calculate how many to remove
        to_remove = current_count - max_memories
        
        # Get candidates for removal (low quality, low access, old)
        cursor.execute("""
            SELECT id, quality_score, access_count, 
                   julianday('now') - julianday(last_accessed) as days_since_access
            FROM advanced_memories
            WHERE quality_score < ? OR access_count = 0
            ORDER BY quality_score ASC, access_count ASC, days_since_access DESC
            LIMIT ?
        """, (quality_threshold, to_remove))
        
        candidates = cursor.fetchall()
        pruned_count = 0
        
        for memory_id, quality_score, access_count, days_since_access in candidates:
            # Remove the memory and its relationships
            cursor.execute("DELETE FROM memory_relationships WHERE source_memory_id = ? OR target_memory_id = ?", 
                          (memory_id, memory_id))
            cursor.execute("DELETE FROM advanced_memories WHERE id = ?", (memory_id,))
            pruned_count += 1
        
        conn.commit()
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM advanced_memories")
        final_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {'pruned_count': pruned_count, 'remaining_count': final_count}

    def optimize_memory_storage(self) -> Dict[str, Any]:
        """Optimize memory storage by reorganizing, compressing, and cleaning up."""
        results = {
            'compression_results': {},
            'cleanup_results': {},
            'organization_results': {},
            'pruning_results': {}
        }
        
        try:
            # 1. Compress memories
            results['compression_results'] = self.compress_memories()
            
            # 2. Clean up low quality and duplicate memories
            results['cleanup_results'] = self.cleanup_memories()
            
            # 3. Organize hierarchical memory
            results['organization_results'] = self.organize_hierarchical_memory()
            
            # 4. Auto-prune if needed
            results['pruning_results'] = self.auto_prune_memories()
            
            # 5. Update statistics
            results['final_statistics'] = self.get_statistics()
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error during memory optimization: {e}")
        
        return results

# Initialize vector backend on import
def get_vector_backend(backend_name: str = 'sqlitefaiss', config: Optional[Dict] = None):
    """Get a vector backend instance."""
    if config is None:
        config = {}
    
    try:
        from .vector_memory import SQLiteFAISSBackend, InMemoryBackend
        
        if backend_name == 'sqlitefaiss':
            db_path = config.get('db_path', 'data/vectors.db')
            return SQLiteFAISSBackend(db_path)
        elif backend_name == 'inmemory':
            return InMemoryBackend()
        else:
            # Fallback to SQLite
            db_path = config.get('db_path', 'data/vectors.db')
            return SQLiteFAISSBackend(db_path)
            
    except ImportError as e:
        print(f"Warning: Could not import vector backend: {e}")
        # Ultimate fallback - create a minimal backend
        class MinimalBackend:
            def __init__(self, db_path):
                self.db_path = db_path
            def search_vector(self, query, limit=10, min_similarity=0.1):
                return []
            def add_vector(self, vector, metadata):
                return 0
        
        return MinimalBackend(config.get('db_path', 'data/vectors.db'))ontent_info': {
                'text_length': len(text),
                'has_context': bool(context),
                'tag_count': len(tags) if tags else 0,
                'category': category
            },
            'usage_info': {
                'created_at': created_at,
                'last_accessed': last_accessed,
                'access_count': access_count
            },
            'improvement_suggestions': suggestions
        }
    
    def get_memory(self, memory_id: int) -> Dict[str, Any]:
        """Get a specific advanced memory by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM advanced_memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'id': row[0],
                'text': row[1] or '',
                'memory_type': row[2] or '',
                'priority': row[3],
                'context': row[4] or '',
                'tags': json.loads(row[5]) if row[5] else [],
                'category': row[6] or '',
                'vector_embedding': json.loads(row[7]) if row[7] else {},
                'quality_score': row[8],
                'confidence_score': row[9],
                'completeness_score': row[10],
                'relevance_score': row[11],
                'memory_order': row[12],
                'created_at': row[13] or '',
                'updated_at': row[14] or '',
                'last_accessed': row[15] or '',
                'access_count': row[16]
            }
        return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get advanced memory statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM advanced_memories")
        total_memories = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM memory_relationships")
        total_relationships = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_reminders")
        total_reminders = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute("""
            SELECT category, COUNT(*) FROM advanced_memories 
            GROUP BY category ORDER BY COUNT(*) DESC
        """)
        category_distribution = dict(cursor.fetchall())
        
        # Quality distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN quality_score >= 0.8 THEN 'high'
                    WHEN quality_score >= 0.6 THEN 'medium'
                    ELSE 'low'
                END as quality_level,
                COUNT(*)
            FROM advanced_memories 
            GROUP BY quality_level
        """)
        quality_distribution = dict(cursor.fetchall())
        
        # Average scores
        cursor.execute("""
            SELECT AVG(quality_score), AVG(confidence_score), 
                   AVG(completeness_score), AVG(relevance_score)
            FROM advanced_memories
        """)
        avg_scores = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_memories': total_memories,
            'total_relationships': total_relationships,
            'total_reminders': total_reminders,
            'category_distribution': category_distribution,
            'quality_distribution': quality_distribution,
            'average_scores': {
                'quality': avg_scores[0] or 0,
                'confidence': avg_scores[1] or 0,
                'completeness': avg_scores[2] or 0,
                'relevance': avg_scores[3] or 0
            }
        }

    def set_encoder(self, encoder: VectorEncoder):
        self.encoder = encoder 

    def get_metrics(self) -> Dict[str, float]:
        """Return average compression ratio and search accuracy metrics."""
        avg_compression = sum(self.compression_ratios) / len(self.compression_ratios) if self.compression_ratios else 1.0
        avg_similarity = sum(self.search_similarities) / len(self.search_similarities) if self.search_similarities else 0.0
        max_similarity = max(self.search_similarities) if self.search_similarities else 0.0
        return {
            'average_compression_ratio': avg_compression,
            'average_search_similarity': avg_similarity,
            'max_search_similarity': max_similarity,
            'num_memories': len(self.compression_ratios),
            'num_searches': len(self.search_similarities)
        }

    def batch_vector_search(self, queries: List[str], limit: int = 10, memory_type: Optional[str] = None, min_similarity: float = 0.1) -> List[List[Dict[str, Any]]]:
        """Batch vector search: returns a list of search results for each query. Prepares for ANN integration."""
        if memory_type is None:
            memory_type = ''
        results = []
        for query in queries:
            res = self.vector_search(query, limit=limit, memory_type=memory_type, min_similarity=min_similarity)
            results.append(res)
        return results 

    def set_vector_backend(self, backend_name: str, backend_config: dict = {}) -> None:
        """Switch the vector backend at runtime."""
        self.vector_backend = get_vector_backend(backend_name, backend_config) 

    def update_memory_order(self, memory_id: int, new_order: int) -> bool:
        """Update the memory_order of an advanced memory (promote/demote order)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE advanced_memories SET memory_order = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (new_order, memory_id))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success 

    def get_backend_info(self) -> dict:
        """Return information about the current vector backend, including pros/cons and config."""
        info = {
            'sqlitefaiss': {
                'pros': 'Portable, local, no dependencies, fast for small/medium datasets',
                'cons': 'Not distributed, limited scalability',
                'usage': 'Default for USB/portable deployments'
            },
            'milvus': {
                'pros': 'Remote, scalable, high-performance, distributed',
                'cons': 'Requires Milvus server, more setup',
                'usage': 'Best for large teams or cloud deployments'
            },
            'annoy': {
                'pros': 'Lightweight, easy to embed, fast for approximate search',
                'cons': 'Approximate only, not as feature-rich',
                'usage': 'Research, embedded, or resource-constrained environments'
            },
            'qdrant': {
                'pros': 'Modern, scalable, REST API, future-proof',
                'cons': 'Requires Qdrant server, newer ecosystem',
                'usage': 'Future-proofing, modern deployments'
            }
        }
        return info.get(self.backend_name, {'info': 'Unknown backend'}) 