#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System
Advanced context retrieval and generation system for optimal LLM interaction.
"""

import sqlite3
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import math

@dataclass(frozen=True)
class RAGChunk:
    """Represents a chunk of information for RAG retrieval."""
    id: int
    content: str
    source_type: str  # 'memory', 'task', 'code', 'document', 'feedback'
    source_id: int
    project_id: Optional[str]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: str  # Use str for ISO format, or datetime if always present
    last_accessed: str
    access_count: int
    relevance_score: float

@dataclass
class RAGQuery:
    """Represents a RAG query with context."""
    query: str
    context: Dict[str, Any]
    max_tokens: int
    chunk_types: List[str]
    project_id: Optional[str]
    user_id: Optional[str]

@dataclass
class RAGResult:
    """Represents a RAG retrieval result."""
    chunks: List[RAGChunk]
    total_tokens: int
    relevance_scores: List[float]
    sources: List[str]
    summary: str
    confidence: float

class RAGSystem:
    """Advanced RAG system for intelligent context retrieval."""
    
    def __init__(self, db_path: str = "data/rag_system.db"):
        """Initialize the RAG system."""
        self.db_path = db_path or "data/rag_system.db"
        self._init_rag_database()
    
    def _init_rag_database(self):
        """Initialize the RAG database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # RAG chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                project_id TEXT,
                embedding TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0.0,
                chunk_hash TEXT UNIQUE
            )
        """)
        
        # Query history for learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                context TEXT,
                retrieved_chunks TEXT,
                user_feedback INTEGER,
                tokens_used INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Chunk relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                related_chunk_id INTEGER,
                relationship_type TEXT,
                strength REAL DEFAULT 1.0,
                FOREIGN KEY (chunk_id) REFERENCES rag_chunks (id),
                FOREIGN KEY (related_chunk_id) REFERENCES rag_chunks (id)
            )
        """)
        
        # Relevance patterns for learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relevance_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                source_type TEXT,
                relevance_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_chunk(self, content: str, source_type: str, source_id: int, project_id: str = "", metadata: dict = None) -> int:
        """Add a chunk to the RAG database."""
        if metadata is None:
            metadata = {}
        
        # Generate chunk hash for deduplication
        chunk_hash = hashlib.md5(f"{content}:{source_type}:{source_id}".encode()).hexdigest()
        
        # Create embedding (simplified TF-IDF for now, can be enhanced with better embeddings)
        embedding = self._create_embedding(content)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO rag_chunks 
                (content, source_type, source_id, project_id, embedding, metadata, chunk_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                content,
                source_type,
                source_id,
                project_id,
                json.dumps(embedding),
                json.dumps(metadata),
                chunk_hash
            ))
            
            chunk_id = cursor.lastrowid
            conn.commit()
            return chunk_id if chunk_id is not None else 0
            
        except sqlite3.IntegrityError:
            # Chunk already exists, return existing ID
            cursor.execute("SELECT id FROM rag_chunks WHERE chunk_hash = ?", (chunk_hash,))
            return cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        finally:
            conn.close()
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create a simple TF-IDF embedding for the text."""
        # Simple word frequency embedding (can be enhanced with better models)
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Normalize by text length
        text_length = len(words)
        if text_length > 0:
            embedding = [word_freq.get(word, 0) / text_length for word in sorted(word_freq.keys())]
        else:
            embedding = []
        
        return embedding
    
    def retrieve_context(self, query: RAGQuery) -> RAGResult:
        """Retrieve relevant context based on the query."""
        # Get relevant chunks
        chunks = self._find_relevant_chunks(query)
        
        # Rank and filter chunks
        ranked_chunks = self._rank_chunks(chunks, query)
        
        # Generate optimized context
        result = self._generate_context(ranked_chunks, query)
        
        # Log query for learning
        self._log_query(query, result)
        
        return result
    
    def _find_relevant_chunks(self, query: RAGQuery) -> List[RAGChunk]:
        """Find chunks relevant to the query."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query conditions
        conditions = []
        params = []
        
        if query.chunk_types:
            placeholders = ','.join(['?' for _ in query.chunk_types])
            conditions.append(f"source_type IN ({placeholders})")
            params.extend(query.chunk_types)
        
        if query.project_id:
            conditions.append("project_id = ?")
            params.append(query.project_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor.execute(f"""
            SELECT id, content, source_type, source_id, project_id, embedding, metadata,
                   created_at, last_accessed, access_count, relevance_score
            FROM rag_chunks
            WHERE {where_clause}
            ORDER BY relevance_score DESC, last_accessed DESC
            LIMIT 100
        """, params)
        
        chunks = []
        for row in cursor.fetchall():
            embedding = json.loads(row[5]) if row[5] else []
            chunk = RAGChunk(
                id=row[0],
                content=row[1],
                source_type=row[2],
                source_id=row[3],
                project_id=row[4] if row[4] is not None else "",
                embedding=embedding,
                metadata=json.loads(row[6]) if row[6] else {},
                created_at=row[7],
                last_accessed=row[8],
                access_count=row[9] if row[9] is not None else 0,
                relevance_score=row[10] if row[10] is not None else 0.0
            )
            chunks.append(chunk)
        
        conn.close()
        return chunks
    
    def _rank_chunks(self, chunks: List[RAGChunk], query: RAGQuery) -> List[Tuple[RAGChunk, float]]:
        """Rank chunks by relevance to the query."""
        query_embedding = self._create_embedding(query.query)
        ranked_chunks = []
        
        for chunk in chunks:
            # Calculate semantic similarity
            semantic_score = self._calculate_similarity(query_embedding, chunk.embedding)
            
            # Calculate recency score
            recency_score = self._calculate_recency_score(chunk.last_accessed)
            
            # Calculate usage score
            usage_score = min(chunk.access_count / 10.0, 1.0)  # Cap at 1.0
            
            # Calculate context relevance
            context_score = self._calculate_context_relevance(chunk, query.context)
            
            # Combine scores
            total_score = (
                semantic_score * 0.4 +
                recency_score * 0.2 +
                usage_score * 0.2 +
                context_score * 0.2
            )
            
            ranked_chunks.append((chunk, total_score))
        
        # Sort by score and return top chunks
        ranked_chunks.sort(key=lambda x: x[1], reverse=True)
        return ranked_chunks
    
    def _calculate_similarity(self, embedding1: list, embedding2: list) -> float:
        """Calculate cosine similarity between embeddings."""
        if embedding1 is None:
            embedding1 = []
        if embedding2 is None:
            embedding2 = []
        if not embedding1 or not embedding2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(a * a for a in embedding2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _calculate_recency_score(self, last_accessed: str) -> float:
        """Calculate recency score based on last access time."""
        if not last_accessed:
            return 0.1
        
        days_since = (datetime.now() - datetime.fromisoformat(last_accessed)).days
        return max(0.1, 1.0 - (days_since / 30.0))  # Decay over 30 days
    
    def _calculate_context_relevance(self, chunk: RAGChunk, context: Dict[str, Any]) -> float:
        """Calculate relevance based on current context."""
        if not context:
            return 0.5  # Neutral score
        
        score = 0.5
        
        # Check if chunk is related to current project
        if context.get('project_id') and chunk.project_id == context.get('project_id'):
            score += 0.3
        
        # Check if chunk is related to current task
        if context.get('current_task') and 'task' in chunk.source_type:
            score += 0.2
        
        # Check if chunk is related to current phase
        if context.get('current_phase') and chunk.metadata.get('phase') == context.get('current_phase'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_context(self, ranked_chunks: List[Tuple[RAGChunk, float]], 
                         query: RAGQuery) -> RAGResult:
        """Generate optimized context from ranked chunks."""
        selected_chunks = []
        total_tokens = 0
        relevance_scores = []
        sources = []
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        for chunk, score in ranked_chunks:
            chunk_tokens = len(chunk.content) // 4
            
            if total_tokens + chunk_tokens <= query.max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                relevance_scores.append(score)
                sources.append(f"{chunk.source_type}:{chunk.source_id}")
            else:
                break
        
        # Generate summary
        summary = self._generate_summary(selected_chunks, query)
        
        # Calculate confidence based on relevance scores
        confidence = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return RAGResult(
            chunks=selected_chunks,
            total_tokens=total_tokens,
            relevance_scores=relevance_scores,
            sources=sources,
            summary=summary,
            confidence=confidence
        )
    
    def _generate_summary(self, chunks: List[RAGChunk], query: RAGQuery) -> str:
        """Generate a summary of the retrieved chunks."""
        if not chunks:
            return "No relevant context found."
        
        # Group chunks by source type
        grouped = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.source_type].append(chunk)
        
        summary_parts = []
        
        for source_type, source_chunks in grouped.items():
            if source_type == 'memory':
                summary_parts.append(f"📝 {len(source_chunks)} relevant memories")
            elif source_type == 'task':
                summary_parts.append(f"📋 {len(source_chunks)} related tasks")
            elif source_type == 'code':
                summary_parts.append(f"💻 {len(source_chunks)} code references")
            elif source_type == 'document':
                summary_parts.append(f"📄 {len(source_chunks)} documents")
            elif source_type == 'feedback':
                summary_parts.append(f"💡 {len(source_chunks)} feedback items")
        
        summary = f"Retrieved: {' | '.join(summary_parts)}"
        
        # Add key insights
        key_insights = []
        for chunk in chunks[:3]:  # Top 3 chunks
            if chunk.metadata.get('key_insight'):
                key_insights.append(chunk.metadata['key_insight'])
        
        if key_insights:
            summary += f"\nKey insights: {'; '.join(key_insights)}"
        
        return summary
    
    def _log_query(self, query: RAGQuery, result: RAGResult):
        """Log query for learning and improvement."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rag_queries (query, context, retrieved_chunks, tokens_used)
            VALUES (?, ?, ?, ?)
        """, (
            query.query,
            json.dumps(query.context),
            json.dumps([c.id for c in result.chunks]),
            result.total_tokens
        ))
        
        # Update chunk access counts
        for chunk in result.chunks:
            cursor.execute("""
                UPDATE rag_chunks 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chunk.id,))
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, query_id: int, feedback_score: int, feedback_text: str = None):
        """Add user feedback to improve the system."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE rag_queries 
            SET user_feedback = ?
            WHERE id = ?
        """, (feedback_score, query_id))
        
        # Update relevance patterns based on feedback
        if feedback_text:
            # Extract patterns from feedback
            patterns = self._extract_patterns(feedback_text)
            for pattern in patterns:
                cursor.execute("""
                    INSERT OR REPLACE INTO relevance_patterns (pattern, relevance_score, usage_count)
                    VALUES (?, ?, COALESCE((SELECT usage_count + 1 FROM relevance_patterns WHERE pattern = ?), 1))
                """, (pattern, feedback_score / 5.0, pattern))
        
        conn.commit()
        conn.close()
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from feedback text."""
        # Simple pattern extraction (can be enhanced)
        words = re.findall(r'\b\w+\b', text.lower())
        patterns = []
        
        # Extract 2-3 word phrases
        for i in range(len(words) - 1):
            patterns.append(f"{words[i]} {words[i+1]}")
        
        return patterns[:10]  # Limit to top 10 patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chunk statistics
        cursor.execute("SELECT COUNT(*), source_type FROM rag_chunks GROUP BY source_type")
        chunk_stats = dict(cursor.fetchall())
        
        # Query statistics
        cursor.execute("SELECT COUNT(*), AVG(tokens_used) FROM rag_queries")
        query_count, avg_tokens = cursor.fetchone()
        
        # Top patterns
        cursor.execute("""
            SELECT pattern, relevance_score, usage_count 
            FROM relevance_patterns 
            ORDER BY usage_count DESC 
            LIMIT 5
        """)
        top_patterns = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_chunks': sum(chunk_stats.values()),
            'chunks_by_type': chunk_stats,
            'total_queries': query_count,
            'average_tokens_per_query': avg_tokens or 0,
            'top_patterns': top_patterns
        }
    
    def cleanup_old_chunks(self, days_old: int = 90):
        """Clean up old, rarely accessed chunks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cursor.execute("""
            DELETE FROM rag_chunks 
            WHERE last_accessed < ? AND access_count < 3
        """, (cutoff_date.isoformat(),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def index_all_docstrings(self, project_root: str):
        """Extract and index all docstrings from Python files as RAG chunks (source_type='docstring')."""
        from .regex_search import extract_all_docstrings
        docstrings = extract_all_docstrings(project_root)
        for file_path, lineno, doc in docstrings:
            metadata = {"file": file_path, "line": lineno}
            self.add_chunk(
                content=doc,
                source_type="docstring",
                source_id=0,
                project_id="",
                metadata=metadata
            ) 