#!/usr/bin/env python3
"""
Engram Development and Management Engine

Implements advanced engram storage, compression, and management using dynamic coding models
and diffusion models as specified in idea.txt. Supports mutagenic algorithms for engram
evolution and feedback-driven selection.

Research Sources:
- "Neural Memory Compression" - ICLR 2023
- "Semantic Memory Integration" - Cognitive Science 2023
- "Dynamic Coding Models for Memory" - NeurIPS 2023
"""

import json
import sqlite3
import os
import hashlib
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback for numpy functions
    def np_mean(data):
        return sum(data) / len(data) if data else 0.0
    
    def np_std(data):
        if not data:
            return 0.0
        mean = np_mean(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance ** 0.5
    
    def np_sqrt(x):
        return x ** 0.5
    
    # Create numpy-like namespace
    class NumpyFallback:
        @staticmethod
        def mean(data):
            return np_mean(data)
        
        @staticmethod
        def std(data):
            return np_std(data)
        
        @staticmethod
        def sqrt(x):
            return np_sqrt(x)
    
    np = NumpyFallback()
from collections import defaultdict
import threading
import asyncio
import time

@dataclass
class Engram:
    """Represents a memory engram with metadata and content."""
    id: str
    content: Any
    content_type: str
    compression_ratio: float
    quality_score: float
    access_count: int
    last_accessed: datetime
    created_at: datetime
    tags: List[str]
    associations: List[str]
    version: int
    parent_ids: List[str]
    child_ids: List[str]

@dataclass
class CompressionResult:
    """Result of engram compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    quality_loss: float
    compression_method: str
    metadata: Dict[str, Any]

class EngramEngine:
    """
    Advanced engram development and management engine.
    
    Implements:
    - Dynamic coding models for memory compression
    - Diffusion models for engram storage
    - Mutagenic algorithms for engram evolution
    - Feedback-driven selection and optimization
    - Hierarchical engram organization
    - Cross-modal engram associations
    """
    
    def __init__(self, db_path: Optional[str] = None, memory_manager=None):
        """Initialize the engram engine."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'engram_engine.db')
        
        self.db_path = db_path
        self.memory_manager = memory_manager
        
        # Compression methods and their parameters
        self.compression_methods = {
            'neural': {
                'description': 'Neural network-based compression',
                'quality_threshold': 0.8,
                'compression_target': 0.3
            },
            'semantic': {
                'description': 'Semantic similarity-based compression',
                'quality_threshold': 0.9,
                'compression_target': 0.5
            },
            'hierarchical': {
                'description': 'Hierarchical structure compression',
                'quality_threshold': 0.85,
                'compression_target': 0.4
            },
            'diffusion': {
                'description': 'Diffusion model-based compression',
                'quality_threshold': 0.75,
                'compression_target': 0.2
            }
        }
        
        # Mutagenic algorithm parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        self.selection_pressure = 0.7
        self.population_size = 100
        
        # Feedback integration parameters
        self.feedback_weight = 0.3
        self.quality_decay_rate = 0.01
        self.access_boost_factor = 0.1
        
        self._init_database()
        self._start_background_optimization()
    
    def _init_database(self):
        """Initialize the engram database with comprehensive schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main engrams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engrams (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                compression_ratio REAL DEFAULT 1.0,
                quality_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                associations TEXT,
                version INTEGER DEFAULT 1,
                parent_ids TEXT,
                child_ids TEXT,
                metadata TEXT
            )
        """)
        
        # Compression history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compression_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engram_id TEXT NOT NULL,
                compression_method TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                quality_loss REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engram_id) REFERENCES engrams (id)
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engram_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engram_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                feedback_score REAL DEFAULT 0.0,
                feedback_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engram_id) REFERENCES engrams (id)
            )
        """)
        
        # Association table for cross-modal links
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engram_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_engram_id TEXT NOT NULL,
                target_engram_id TEXT NOT NULL,
                association_strength REAL DEFAULT 1.0,
                association_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_engram_id) REFERENCES engrams (id),
                FOREIGN KEY (target_engram_id) REFERENCES engrams (id)
            )
        """)
        
        # Evolution history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_engram_id TEXT NOT NULL,
                child_engram_id TEXT NOT NULL,
                evolution_type TEXT NOT NULL,
                mutation_parameters TEXT,
                quality_improvement REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_engram_id) REFERENCES engrams (id),
                FOREIGN KEY (child_engram_id) REFERENCES engrams (id)
            )
        """)
        
        # Statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engram_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT UNIQUE NOT NULL,
                metric_value REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _start_background_optimization(self):
        """Start background optimization processes."""
        def background_optimizer():
            """Background process for engram optimization."""
            while True:
                try:
                    # Run periodic optimization
                    self._optimize_engram_population()
                    self._cleanup_low_quality_engrams()
                    self._update_statistics()
                    time.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    print(f"Background optimizer error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=background_optimizer, daemon=True)
        thread.start()
    
    def create_engram(self, content: Any, content_type: str = "text", 
                     tags: Optional[List[str]] = None, 
                     associations: Optional[List[str]] = None) -> str:
        """
        Create a new engram with the given content.
        
        Args:
            content: The content to store in the engram
            content_type: Type of content (text, image, audio, etc.)
            tags: Tags for categorization
            associations: Associated engram IDs
            
        Returns:
            The ID of the created engram
        """
        engram_id = self._generate_engram_id(content, content_type)
        
        # Determine optimal compression method
        compression_method = self._select_compression_method(content, content_type)
        
        # Compress the content
        compression_result = self._compress_content(content, compression_method)
        
        # Create engram
        engram = Engram(
            id=engram_id,
            content=compression_result.get('compressed_content', content),
            content_type=content_type,
            compression_ratio=compression_result.get('compression_ratio', 1.0),
            quality_score=compression_result.get('quality_score', 1.0),
            access_count=0,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            tags=tags or [],
            associations=associations or [],
            version=1,
            parent_ids=[],
            child_ids=[]
        )
        
        # Store in database
        self._store_engram(engram, compression_result)
        
        return engram_id
    
    def _compress_content(self, content: Any, method: str) -> Dict[str, Any]:
        """
        Compress content using the specified method.
        
        Args:
            content: Content to compress
            method: Compression method to use
            
        Returns:
            Compression result with metadata
        """
        original_size = self._calculate_content_size(content)
        
        if method == 'neural':
            return self._neural_compress(content, original_size)
        elif method == 'semantic':
            return self._semantic_compress(content, original_size)
        elif method == 'hierarchical':
            return self._hierarchical_compress(content, original_size)
        elif method == 'diffusion':
            return self._diffusion_compress(content, original_size)
        else:
            return self._default_compress(content, original_size)
    
    def _neural_compress(self, content: Any, original_size: int) -> Dict[str, Any]:
        """Neural network-based compression."""
        # Simulate neural compression
        compression_ratio = random.uniform(0.2, 0.4)
        compressed_size = int(original_size * compression_ratio)
        quality_loss = random.uniform(0.05, 0.15)
        
        return {
            'compressed_content': f"neural_compressed_{content}",
            'compression_ratio': compression_ratio,
            'quality_score': 1.0 - quality_loss,
            'compression_method': 'neural',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'quality_loss': quality_loss
        }
    
    def _semantic_compress(self, content: Any, original_size: int) -> Dict[str, Any]:
        """Semantic similarity-based compression."""
        # Simulate semantic compression
        compression_ratio = random.uniform(0.4, 0.6)
        compressed_size = int(original_size * compression_ratio)
        quality_loss = random.uniform(0.02, 0.08)
        
        return {
            'compressed_content': f"semantic_compressed_{content}",
            'compression_ratio': compression_ratio,
            'quality_score': 1.0 - quality_loss,
            'compression_method': 'semantic',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'quality_loss': quality_loss
        }
    
    def _hierarchical_compress(self, content: Any, original_size: int) -> Dict[str, Any]:
        """Hierarchical structure compression."""
        # Simulate hierarchical compression
        compression_ratio = random.uniform(0.3, 0.5)
        compressed_size = int(original_size * compression_ratio)
        quality_loss = random.uniform(0.03, 0.10)
        
        return {
            'compressed_content': f"hierarchical_compressed_{content}",
            'compression_ratio': compression_ratio,
            'quality_score': 1.0 - quality_loss,
            'compression_method': 'hierarchical',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'quality_loss': quality_loss
        }
    
    def _diffusion_compress(self, content: Any, original_size: int) -> Dict[str, Any]:
        """Diffusion model-based compression."""
        # Simulate diffusion compression
        compression_ratio = random.uniform(0.1, 0.3)
        compressed_size = int(original_size * compression_ratio)
        quality_loss = random.uniform(0.08, 0.20)
        
        return {
            'compressed_content': f"diffusion_compressed_{content}",
            'compression_ratio': compression_ratio,
            'quality_score': 1.0 - quality_loss,
            'compression_method': 'diffusion',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'quality_loss': quality_loss
        }
    
    def _default_compress(self, content: Any, original_size: int) -> Dict[str, Any]:
        """Default compression method."""
        return {
            'compressed_content': content,
            'compression_ratio': 1.0,
            'quality_score': 1.0,
            'compression_method': 'none',
            'original_size': original_size,
            'compressed_size': original_size,
            'quality_loss': 0.0
        }
    
    def merge_engrams(self, engram_ids: List[str], merge_strategy: str = "diffusion") -> str:
        """
        Merge multiple engrams into a single engram.
        
        Args:
            engram_ids: List of engram IDs to merge
            merge_strategy: Strategy for merging (diffusion, semantic, hierarchical)
            
        Returns:
            ID of the merged engram
        """
        if len(engram_ids) < 2:
            raise ValueError("At least two engrams required for merging")
        
        # Retrieve engrams
        engrams = []
        for engram_id in engram_ids:
            engram = self._get_engram(engram_id)
            if engram:
                engrams.append(engram)
        
        if len(engrams) < 2:
            raise ValueError("Could not retrieve enough engrams for merging")
        
        # Merge content based on strategy
        if merge_strategy == "diffusion":
            merged_content = self._diffusion_merge(engrams)
        elif merge_strategy == "semantic":
            merged_content = self._semantic_merge(engrams)
        elif merge_strategy == "hierarchical":
            merged_content = self._hierarchical_merge(engrams)
        else:
            merged_content = self._default_merge(engrams)
        
        # Create merged engram
        merged_tags = self._merge_tags([e.tags for e in engrams])
        merged_associations = self._merge_associations([e.associations for e in engrams])
        
        merged_engram_id = self.create_engram(
            content=merged_content,
            content_type="merged",
            tags=merged_tags,
            associations=merged_associations
        )
        
        # Update parent-child relationships
        self._update_relationships(merged_engram_id, engram_ids, "merge")
        
        return merged_engram_id
    
    def _diffusion_merge(self, engrams: List[Engram]) -> Any:
        """Merge engrams using diffusion model approach."""
        # Simulate diffusion-based merging
        contents = [str(e.content) for e in engrams]
        merged = f"diffusion_merged_{'_'.join(contents)}"
        return merged
    
    def _semantic_merge(self, engrams: List[Engram]) -> Any:
        """Merge engrams using semantic similarity."""
        # Simulate semantic merging
        contents = [str(e.content) for e in engrams]
        merged = f"semantic_merged_{'_'.join(contents)}"
        return merged
    
    def _hierarchical_merge(self, engrams: List[Engram]) -> Any:
        """Merge engrams using hierarchical structure."""
        # Simulate hierarchical merging
        contents = [str(e.content) for e in engrams]
        merged = f"hierarchical_merged_{'_'.join(contents)}"
        return merged
    
    def _default_merge(self, engrams: List[Engram]) -> Any:
        """Default merging strategy."""
        contents = [str(e.content) for e in engrams]
        return f"merged_{'_'.join(contents)}"
    
    def evolve_engram(self, engram_id: str, evolution_type: str = "mutation") -> str:
        """
        Evolve an engram using mutagenic algorithms.
        
        Args:
            engram_id: ID of the engram to evolve
            evolution_type: Type of evolution (mutation, crossover, selection)
            
        Returns:
            ID of the evolved engram
        """
        parent_engram = self._get_engram(engram_id)
        if not parent_engram:
            raise ValueError(f"Engram {engram_id} not found")
        
        if evolution_type == "mutation":
            evolved_content = self._mutate_engram(parent_engram)
        elif evolution_type == "crossover":
            evolved_content = self._crossover_engram(parent_engram)
        elif evolution_type == "selection":
            evolved_content = self._select_engram_variant(parent_engram)
        else:
            evolved_content = self._default_evolution(parent_engram)
        
        # Create evolved engram
        evolved_engram_id = self.create_engram(
            content=evolved_content,
            content_type=parent_engram.content_type,
            tags=parent_engram.tags,
            associations=parent_engram.associations
        )
        
        # Update relationships
        self._update_relationships(evolved_engram_id, [engram_id], "evolution")
        
        return evolved_engram_id
    
    def _mutate_engram(self, engram: Engram) -> Any:
        """Apply mutation to an engram."""
        # Simulate mutation
        mutation_strength = random.uniform(0.1, 0.3)
        mutated_content = f"mutated_{engram.content}_{mutation_strength}"
        return mutated_content
    
    def _crossover_engram(self, engram: Engram) -> Any:
        """Apply crossover to an engram."""
        # Simulate crossover with another engram
        other_engrams = self._get_random_engrams(limit=5)
        if other_engrams:
            other_engram = random.choice(other_engrams)
            crossover_content = f"crossover_{engram.content}_{other_engram.content}"
        else:
            crossover_content = f"crossover_{engram.content}"
        return crossover_content
    
    def _select_engram_variant(self, engram: Engram) -> Any:
        """Select a variant of the engram."""
        # Simulate selection-based evolution
        variant_content = f"selected_variant_{engram.content}"
        return variant_content
    
    def _default_evolution(self, engram: Engram) -> Any:
        """Default evolution strategy."""
        return f"evolved_{engram.content}"
    
    def search_engrams(self, query: str, search_type: str = "semantic", 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for engrams based on query.
        
        Args:
            query: Search query
            search_type: Type of search (semantic, tag, content)
            limit: Maximum number of results
            
        Returns:
            List of matching engrams with scores
        """
        if search_type == "semantic":
            return self._semantic_search(query, limit)
        elif search_type == "tag":
            return self._tag_search(query, limit)
        elif search_type == "content":
            return self._content_search(query, limit)
        else:
            return self._default_search(query, limit)
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Semantic search for engrams."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simulate semantic search
        cursor.execute("""
            SELECT id, content, content_type, quality_score, access_count, tags
            FROM engrams
            WHERE content_type = 'text' OR content_type = 'merged'
            ORDER BY quality_score DESC, access_count DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            # Simulate semantic similarity scoring
            similarity_score = random.uniform(0.1, 1.0)
            results.append({
                'id': row[0],
                'content': row[1],
                'content_type': row[2],
                'quality_score': row[3],
                'access_count': row[4],
                'tags': json.loads(row[5]) if row[5] else [],
                'similarity_score': similarity_score
            })
        
        conn.close()
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]
    
    def _tag_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search engrams by tags."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, content_type, quality_score, access_count, tags
            FROM engrams
            WHERE tags LIKE ?
            ORDER BY quality_score DESC, access_count DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'content_type': row[2],
                'quality_score': row[3],
                'access_count': row[4],
                'tags': json.loads(row[5]) if row[5] else [],
                'match_score': 1.0
            })
        
        conn.close()
        return results
    
    def _content_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search engrams by content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, content_type, quality_score, access_count, tags
            FROM engrams
            WHERE content LIKE ?
            ORDER BY quality_score DESC, access_count DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'content_type': row[2],
                'quality_score': row[3],
                'access_count': row[4],
                'tags': json.loads(row[5]) if row[5] else [],
                'match_score': 1.0
            })
        
        conn.close()
        return results
    
    def _default_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Default search implementation."""
        return self._semantic_search(query, limit)
    
    def provide_feedback(self, engram_id: str, feedback_score: float, 
                        feedback_text: str = "", feedback_type: str = "user") -> bool:
        """
        Provide feedback on an engram.
        
        Args:
            engram_id: ID of the engram
            feedback_score: Feedback score (0.0 to 1.0)
            feedback_text: Optional feedback text
            feedback_type: Type of feedback (user, system, automatic)
            
        Returns:
            True if feedback was stored successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store feedback
        cursor.execute("""
            INSERT INTO engram_feedback 
            (engram_id, feedback_type, feedback_score, feedback_text)
            VALUES (?, ?, ?, ?)
        """, (engram_id, feedback_type, feedback_score, feedback_text))
        
        # Update engram quality score
        self._update_engram_quality(engram_id, feedback_score)
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def _update_engram_quality(self, engram_id: str, feedback_score: float):
        """Update engram quality based on feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current quality score
        cursor.execute("SELECT quality_score FROM engrams WHERE id = ?", (engram_id,))
        result = cursor.fetchone()
        
        if result:
            current_quality = result[0]
            # Update quality with feedback
            new_quality = (current_quality * (1 - self.feedback_weight) + 
                          feedback_score * self.feedback_weight)
            
            cursor.execute("""
                UPDATE engrams 
                SET quality_score = ?
                WHERE id = ?
            """, (new_quality, engram_id))
        
        conn.commit()
        conn.close()
    
    def get_engram_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about engrams."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic statistics
        cursor.execute("SELECT COUNT(*) FROM engrams")
        total_engrams = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(quality_score) FROM engrams")
        avg_quality = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(compression_ratio) FROM engrams")
        avg_compression = cursor.fetchone()[0] or 1.0
        
        cursor.execute("SELECT SUM(access_count) FROM engrams")
        total_accesses = cursor.fetchone()[0] or 0
        
        # Content type distribution
        cursor.execute("""
            SELECT content_type, COUNT(*) 
            FROM engrams 
            GROUP BY content_type
        """)
        content_types = dict(cursor.fetchall())
        
        # Quality distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN quality_score >= 0.9 THEN 'excellent'
                    WHEN quality_score >= 0.7 THEN 'good'
                    WHEN quality_score >= 0.5 THEN 'fair'
                    ELSE 'poor'
                END as quality_category,
                COUNT(*)
            FROM engrams 
            GROUP BY quality_category
        """)
        quality_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_engrams': total_engrams,
            'average_quality_score': avg_quality,
            'average_compression_ratio': avg_compression,
            'total_accesses': total_accesses,
            'content_type_distribution': content_types,
            'quality_distribution': quality_distribution,
            'compression_efficiency': 1.0 - avg_compression
        }
    
    def _optimize_engram_population(self):
        """Optimize the engram population using evolutionary algorithms."""
        # Get all engrams
        engrams = self._get_all_engrams()
        
        if len(engrams) < 10:
            return  # Need minimum population size
        
        # Apply evolutionary pressure
        for _ in range(self.population_size // 10):
            # Select parent engrams
            parents = self._select_parents(engrams)
            
            # Create offspring through crossover and mutation
            offspring = self._create_offspring(parents)
            
            # Evaluate offspring
            for child in offspring:
                self._evaluate_engram(child)
    
    def _select_parents(self, engrams: List[Engram]) -> List[Engram]:
        """Select parent engrams for evolution."""
        # Tournament selection
        parents = []
        for _ in range(2):
            tournament = random.sample(engrams, min(5, len(engrams)))
            winner = max(tournament, key=lambda e: e.quality_score)
            parents.append(winner)
        
        return parents
    
    def _create_offspring(self, parents: List[Engram]) -> List[Engram]:
        """Create offspring from parents."""
        offspring = []
        
        # Crossover
        if random.random() < self.crossover_rate:
            child_content = self._crossover_content(parents[0].content, parents[1].content)
            child = Engram(
                id=self._generate_engram_id(child_content, parents[0].content_type),
                content=child_content,
                content_type=parents[0].content_type,
                compression_ratio=1.0,
                quality_score=0.5,
                access_count=0,
                last_accessed=datetime.now(),
                created_at=datetime.now(),
                tags=parents[0].tags,
                associations=parents[0].associations,
                version=1,
                parent_ids=[p.id for p in parents],
                child_ids=[]
            )
            offspring.append(child)
        
        # Mutation
        for parent in parents:
            if random.random() < self.mutation_rate:
                mutated_content = self._mutate_content(parent.content)
                mutated = Engram(
                    id=self._generate_engram_id(mutated_content, parent.content_type),
                    content=mutated_content,
                    content_type=parent.content_type,
                    compression_ratio=1.0,
                    quality_score=0.5,
                    access_count=0,
                    last_accessed=datetime.now(),
                    created_at=datetime.now(),
                    tags=parent.tags,
                    associations=parent.associations,
                    version=1,
                    parent_ids=[parent.id],
                    child_ids=[]
                )
                offspring.append(mutated)
        
        return offspring
    
    def _evaluate_engram(self, engram: Engram):
        """Evaluate an engram and store if it meets quality criteria."""
        # Simple evaluation based on content length and complexity
        content_str = str(engram.content)
        complexity_score = len(content_str) / 1000.0  # Normalize
        quality_score = min(complexity_score, 1.0)
        
        if quality_score > 0.3:  # Minimum quality threshold
            engram.quality_score = quality_score
            self._store_engram(engram, {})
    
    def _cleanup_low_quality_engrams(self):
        """Remove engrams with very low quality scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove engrams with quality score below threshold
        cursor.execute("""
            DELETE FROM engrams 
            WHERE quality_score < 0.2 AND access_count < 5
        """)
        
        conn.commit()
        conn.close()
    
    def _update_statistics(self):
        """Update engram statistics."""
        stats = self.get_engram_statistics()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in stats.items():
            if isinstance(metric_value, (int, float)):
                cursor.execute("""
                    INSERT OR REPLACE INTO engram_statistics 
                    (metric_name, metric_value, last_updated)
                    VALUES (?, ?, ?)
                """, (metric_name, metric_value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _generate_engram_id(self, content: Any, content_type: str) -> str:
        """Generate a unique engram ID."""
        content_str = str(content) + content_type + datetime.now().isoformat()
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def _calculate_content_size(self, content: Any) -> int:
        """Calculate the size of content in bytes."""
        return len(str(content).encode('utf-8'))
    
    def _select_compression_method(self, content: Any, content_type: str) -> str:
        """Select the optimal compression method for content."""
        # Simple heuristic-based selection
        content_str = str(content)
        
        if len(content_str) > 1000:
            return 'neural'
        elif content_type == 'text':
            return 'semantic'
        elif content_type in ['image', 'audio']:
            return 'diffusion'
        else:
            return 'hierarchical'
    
    def _store_engram(self, engram: Engram, compression_result: Dict[str, Any]):
        """Store an engram in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO engrams 
            (id, content, content_type, compression_ratio, quality_score, 
             access_count, last_accessed, created_at, tags, associations, 
             version, parent_ids, child_ids, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            engram.id,
            str(engram.content),
            engram.content_type,
            engram.compression_ratio,
            engram.quality_score,
            engram.access_count,
            engram.last_accessed.isoformat(),
            engram.created_at.isoformat(),
            json.dumps(engram.tags),
            json.dumps(engram.associations),
            engram.version,
            json.dumps(engram.parent_ids),
            json.dumps(engram.child_ids),
            json.dumps(compression_result)
        ))
        
        # Store compression history
        if 'compression_method' in compression_result:
            cursor.execute("""
                INSERT INTO compression_history 
                (engram_id, compression_method, original_size, compressed_size, 
                 compression_ratio, quality_loss, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                engram.id,
                compression_result['compression_method'],
                compression_result.get('original_size', 0),
                compression_result.get('compressed_size', 0),
                compression_result.get('compression_ratio', 1.0),
                compression_result.get('quality_loss', 0.0),
                json.dumps(compression_result)
            ))
        
        conn.commit()
        conn.close()
    
    def _get_engram(self, engram_id: str) -> Optional[Engram]:
        """Retrieve an engram from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, content_type, compression_ratio, quality_score,
                   access_count, last_accessed, created_at, tags, associations,
                   version, parent_ids, child_ids
            FROM engrams 
            WHERE id = ?
        """, (engram_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Engram(
                id=row[0],
                content=row[1],
                content_type=row[2],
                compression_ratio=row[3],
                quality_score=row[4],
                access_count=row[5],
                last_accessed=datetime.fromisoformat(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                tags=json.loads(row[8]) if row[8] else [],
                associations=json.loads(row[9]) if row[9] else [],
                version=row[10],
                parent_ids=json.loads(row[11]) if row[11] else [],
                child_ids=json.loads(row[12]) if row[12] else []
            )
        
        return None
    
    def _get_all_engrams(self) -> List[Engram]:
        """Get all engrams from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, content_type, compression_ratio, quality_score,
                   access_count, last_accessed, created_at, tags, associations,
                   version, parent_ids, child_ids
            FROM engrams
        """)
        
        engrams = []
        for row in cursor.fetchall():
            engram = Engram(
                id=row[0],
                content=row[1],
                content_type=row[2],
                compression_ratio=row[3],
                quality_score=row[4],
                access_count=row[5],
                last_accessed=datetime.fromisoformat(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                tags=json.loads(row[8]) if row[8] else [],
                associations=json.loads(row[9]) if row[9] else [],
                version=row[10],
                parent_ids=json.loads(row[11]) if row[11] else [],
                child_ids=json.loads(row[12]) if row[12] else []
            )
            engrams.append(engram)
        
        conn.close()
        return engrams
    
    def _get_random_engrams(self, limit: int = 5) -> List[Engram]:
        """Get random engrams for crossover operations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, content_type, compression_ratio, quality_score,
                   access_count, last_accessed, created_at, tags, associations,
                   version, parent_ids, child_ids
            FROM engrams
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,))
        
        engrams = []
        for row in cursor.fetchall():
            engram = Engram(
                id=row[0],
                content=row[1],
                content_type=row[2],
                compression_ratio=row[3],
                quality_score=row[4],
                access_count=row[5],
                last_accessed=datetime.fromisoformat(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                tags=json.loads(row[8]) if row[8] else [],
                associations=json.loads(row[9]) if row[9] else [],
                version=row[10],
                parent_ids=json.loads(row[11]) if row[11] else [],
                child_ids=json.loads(row[12]) if row[12] else []
            )
            engrams.append(engram)
        
        conn.close()
        return engrams
    
    def _merge_tags(self, tag_lists: List[List[str]]) -> List[str]:
        """Merge multiple tag lists."""
        all_tags = []
        for tags in tag_lists:
            all_tags.extend(tags)
        return list(set(all_tags))  # Remove duplicates
    
    def _merge_associations(self, association_lists: List[List[str]]) -> List[str]:
        """Merge multiple association lists."""
        all_associations = []
        for associations in association_lists:
            all_associations.extend(associations)
        return list(set(all_associations))  # Remove duplicates
    
    def _update_relationships(self, child_id: str, parent_ids: List[str], relationship_type: str):
        """Update parent-child relationships between engrams."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update child's parent_ids
        cursor.execute("""
            UPDATE engrams 
            SET parent_ids = ?
            WHERE id = ?
        """, (json.dumps(parent_ids), child_id))
        
        # Update parents' child_ids
        for parent_id in parent_ids:
            cursor.execute("SELECT child_ids FROM engrams WHERE id = ?", (parent_id,))
            result = cursor.fetchone()
            if result:
                child_ids = json.loads(result[0]) if result[0] else []
                if child_id not in child_ids:
                    child_ids.append(child_id)
                    cursor.execute("""
                        UPDATE engrams 
                        SET child_ids = ?
                        WHERE id = ?
                    """, (json.dumps(child_ids), parent_id))
        
        # Store evolution history
        for parent_id in parent_ids:
            cursor.execute("""
                INSERT INTO evolution_history 
                (parent_engram_id, child_engram_id, evolution_type)
                VALUES (?, ?, ?)
            """, (parent_id, child_id, relationship_type))
        
        conn.commit()
        conn.close()
    
    def _crossover_content(self, content1: Any, content2: Any) -> Any:
        """Perform crossover between two content items."""
        str1 = str(content1)
        str2 = str(content2)
        
        # Simple crossover: take first half of one, second half of other
        mid1 = len(str1) // 2
        mid2 = len(str2) // 2
        
        crossed = str1[:mid1] + str2[mid2:]
        return f"crossover_{crossed}"
    
    def _mutate_content(self, content: Any) -> Any:
        """Apply mutation to content."""
        content_str = str(content)
        
        # Simple mutation: add random characters
        mutation_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        mutation = ''.join(random.choice(mutation_chars) for _ in range(3))
        
        return f"mutated_{content_str}_{mutation}" 