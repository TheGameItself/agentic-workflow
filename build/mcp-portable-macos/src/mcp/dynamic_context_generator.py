#!/usr/bin/env python3
"""
Dynamic Context Generator with Speculation
Advanced context generation system that incorporates speculation, attention mechanisms, 
and intelligent context switching for optimal LLM performance.

Based on research from:
- Scaled ReAttention (SRA) for long-context comprehension
- Dynamic context switching for smarter, faster inference
- Context engineering principles for production agents
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ContextType(Enum):
    """Types of context that can be generated."""
    MEMORY = "memory"
    TASK = "task"
    CODE = "code"
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    SPECULATION = "speculation"
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    SHORT_TERM_MEMORY = "short_term_memory"


class SpeculationLevel(Enum):
    """Levels of speculation for context generation."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AGGRESSIVE = "aggressive"


@dataclass
class ContextChunk:
    """A chunk of context with metadata."""
    content: str
    context_type: ContextType
    relevance_score: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_weights: Optional[List[float]] = None
    speculation_level: SpeculationLevel = SpeculationLevel.NONE
    token_count: int = 0
    embedding: Optional[List[float]] = None
    chunk_id: str = ""


@dataclass
class DynamicContextConfig:
    """Configuration for dynamic context generation."""
    max_context_tokens: int = 4000
    speculation_enabled: bool = True
    attention_mechanism_enabled: bool = True
    dynamic_switching_enabled: bool = True
    compression_enabled: bool = True
    cache_enabled: bool = True
    max_speculation_chunks: int = 5
    attention_window_size: int = 512
    speculation_threshold: float = 0.7
    context_compression_ratio: float = 0.3
    update_interval_seconds: int = 30
    max_cache_size: int = 1000


class ScaledReAttention:
    """
    Implementation of Scaled ReAttention (SRA) for enhanced attention mechanisms.
    Based on research from arXiv:2307.13365
    """
    
    def __init__(self, attention_window_size: int = 512):
        self.attention_window_size = attention_window_size
        self.logger = logging.getLogger("scaled_reattention")
    
    def calculate_attention_scores(self, query: List[float], keys: List[List[float]], 
                                 values: List[List[float]]) -> Tuple[List[float], List[float]]:
        """
        Calculate attention scores using scaled reattention mechanism.
        
        Args:
            query: Query vector
            keys: Key vectors
            values: Value vectors
            
        Returns:
            Tuple of (attention_weights, attention_output)
        """
        if not TORCH_AVAILABLE:
            return self._fallback_attention(query, keys, values)
        
        try:
            # Convert to tensors
            query_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
            keys_tensor = torch.tensor(keys, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)
            
            # Calculate scaled attention scores
            attention_scores = torch.matmul(query_tensor, keys_tensor.transpose(-2, -1))
            attention_scores = attention_scores / np.sqrt(keys_tensor.size(-1))
            
            # Apply softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            attention_output = torch.matmul(attention_weights, values_tensor)
            
            return attention_weights.squeeze().tolist(), attention_output.squeeze().tolist()
            
        except Exception as e:
            self.logger.warning(f"Error in scaled attention calculation: {e}")
            return self._fallback_attention(query, keys, values)
    
    def _fallback_attention(self, query: List[float], keys: List[List[float]], 
                          values: List[List[float]]) -> Tuple[List[float], List[float]]:
        """Fallback attention calculation without torch."""
        # Simple dot product attention
        attention_scores = []
        for key in keys:
            score = sum(q * k for q, k in zip(query, key))
            attention_scores.append(score)
        
        # Softmax approximation
        max_score = max(attention_scores)
        exp_scores = [np.exp(score - max_score) for score in attention_scores]
        sum_exp = sum(exp_scores)
        attention_weights = [score / sum_exp for score in exp_scores]
        
        # Apply attention to values
        attention_output = [0.0] * len(values[0]) if values else []
        for weight, value in zip(attention_weights, values):
            for i, v in enumerate(value):
                attention_output[i] += weight * v
        
        return attention_weights, attention_output


class ContextSpeculator:
    """
    Advanced context speculation system that predicts what context will be needed.
    Based on dynamic context switching research.
    """
    
    def __init__(self, config: DynamicContextConfig):
        self.config = config
        self.logger = logging.getLogger("context_speculator")
        self.speculation_history = deque(maxlen=1000)
        self.pattern_database = defaultdict(list)
        self.speculation_models = {}
        
    def speculate_context(self, current_context: List[ContextChunk], 
                         user_query: str, conversation_history: List[Dict[str, Any]]) -> List[ContextChunk]:
        """
        Speculate what context will be needed based on current state.
        
        Args:
            current_context: Current context chunks
            user_query: Current user query
            conversation_history: History of conversation
            
        Returns:
            List of speculated context chunks
        """
        speculated_chunks = []
        
        # Analyze current context patterns
        context_patterns = self._analyze_context_patterns(current_context, conversation_history)
        
        # Predict needed context based on patterns
        predicted_needs = self._predict_context_needs(user_query, context_patterns)
        
        # Generate speculative context chunks
        for need in predicted_needs:
            if need['confidence'] >= self.config.speculation_threshold:
                chunk = self._generate_speculative_chunk(need, current_context)
                if chunk:
                    speculated_chunks.append(chunk)
        
        # Limit speculation chunks
        speculated_chunks = speculated_chunks[:self.config.max_speculation_chunks]
        
        self.logger.info(f"Generated {len(speculated_chunks)} speculative context chunks")
        return speculated_chunks
    
    def _analyze_context_patterns(self, current_context: List[ContextChunk], 
                                conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in current context and conversation history."""
        patterns = {
            'context_types': defaultdict(int),
            'topic_transitions': [],
            'attention_focus': [],
            'memory_access_patterns': []
        }
        
        # Analyze context type distribution
        for chunk in current_context:
            patterns['context_types'][chunk.context_type.value] += 1
        
        # Analyze topic transitions
        if len(conversation_history) > 1:
            for i in range(1, len(conversation_history)):
                prev_topic = self._extract_topic(conversation_history[i-1])
                curr_topic = self._extract_topic(conversation_history[i])
                if prev_topic != curr_topic:
                    patterns['topic_transitions'].append({
                        'from': prev_topic,
                        'to': curr_topic,
                        'index': i
                    })
        
        # Analyze attention focus
        for chunk in current_context:
            if chunk.attention_weights:
                focus_index = np.argmax(chunk.attention_weights)
                patterns['attention_focus'].append({
                    'chunk_type': chunk.context_type.value,
                    'focus_position': focus_index,
                    'max_attention': max(chunk.attention_weights)
                })
        
        return patterns
    
    def _predict_context_needs(self, user_query: str, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict what context will be needed based on patterns."""
        predictions = []
        
        # Predict based on query content
        query_keywords = self._extract_keywords(user_query.lower())
        
        # Predict memory needs
        if any(keyword in ['remember', 'recall', 'previous', 'before'] for keyword in query_keywords):
            predictions.append({
                'context_type': ContextType.MEMORY,
                'confidence': 0.9,
                'reason': 'Query contains memory-related keywords'
            })
        
        # Predict task context needs
        if any(keyword in ['task', 'todo', 'project', 'work'] for keyword in query_keywords):
            predictions.append({
                'context_type': ContextType.TASK,
                'confidence': 0.8,
                'reason': 'Query contains task-related keywords'
            })
        
        # Predict code context needs
        if any(keyword in ['code', 'function', 'class', 'import', 'def'] for keyword in query_keywords):
            predictions.append({
                'context_type': ContextType.CODE,
                'confidence': 0.85,
                'reason': 'Query contains code-related keywords'
            })
        
        # Predict based on conversation patterns
        if patterns['topic_transitions']:
            last_transition = patterns['topic_transitions'][-1]
            predictions.append({
                'context_type': ContextType.CONVERSATION,
                'confidence': 0.7,
                'reason': f'Recent topic transition from {last_transition["from"]} to {last_transition["to"]}'
            })
        
        # Predict based on attention patterns
        if patterns['attention_focus']:
            most_focused = max(patterns['attention_focus'], key=lambda x: x['max_attention'])
            predictions.append({
                'context_type': ContextType(most_focused['chunk_type']),
                'confidence': 0.75,
                'reason': f'High attention focus on {most_focused["chunk_type"]} context'
            })
        
        return predictions
    
    def _generate_speculative_chunk(self, need: Dict[str, Any], 
                                  current_context: List[ContextChunk]) -> Optional[ContextChunk]:
        """Generate a speculative context chunk based on predicted need."""
        try:
            # Create speculative content based on context type
            if need['context_type'] == ContextType.MEMORY:
                content = self._generate_memory_speculation(current_context)
            elif need['context_type'] == ContextType.TASK:
                content = self._generate_task_speculation(current_context)
            elif need['context_type'] == ContextType.CODE:
                content = self._generate_code_speculation(current_context)
            else:
                content = self._generate_general_speculation(need, current_context)
            
            if not content:
                return None
            
            # Create speculative chunk
            chunk = ContextChunk(
                content=content,
                context_type=need['context_type'],
                relevance_score=need['confidence'],
                timestamp=datetime.now(),
                source="speculation",
                metadata={
                    'speculation_reason': need['reason'],
                    'confidence': need['confidence'],
                    'speculative': True
                },
                speculation_level=SpeculationLevel.MEDIUM,
                token_count=len(content.split()),
                chunk_id=f"spec_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            )
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error generating speculative chunk: {e}")
            return None
    
    def _generate_memory_speculation(self, current_context: List[ContextChunk]) -> str:
        """Generate speculative memory context."""
        memory_chunks = [c for c in current_context if c.context_type == ContextType.MEMORY]
        
        if not memory_chunks:
            return "No previous memories found. Consider creating new memories for future reference."
        
        # Find most relevant memories
        relevant_memories = sorted(memory_chunks, key=lambda x: x.relevance_score, reverse=True)[:3]
        
        speculation = "Based on current context, you might need to recall:\n"
        for i, memory in enumerate(relevant_memories, 1):
            speculation += f"{i}. {memory.content[:100]}...\n"
        
        return speculation
    
    def _generate_task_speculation(self, current_context: List[ContextChunk]) -> str:
        """Generate speculative task context."""
        task_chunks = [c for c in current_context if c.context_type == ContextType.TASK]
        
        if not task_chunks:
            return "No active tasks found. Consider creating tasks to track progress."
        
        # Find active tasks
        active_tasks = [t for t in task_chunks if 'status' in t.metadata and t.metadata['status'] == 'active']
        
        if not active_tasks:
            return "No active tasks found. Consider reviewing completed tasks or creating new ones."
        
        speculation = "Current active tasks that might be relevant:\n"
        for i, task in enumerate(active_tasks[:3], 1):
            speculation += f"{i}. {task.content[:100]}...\n"
        
        return speculation
    
    def _generate_code_speculation(self, current_context: List[ContextChunk]) -> str:
        """Generate speculative code context."""
        code_chunks = [c for c in current_context if c.context_type == ContextType.CODE]
        
        if not code_chunks:
            return "No code context found. Consider loading relevant code files or functions."
        
        # Find most recent code
        recent_code = sorted(code_chunks, key=lambda x: x.timestamp, reverse=True)[:2]
        
        speculation = "Recent code context that might be relevant:\n"
        for i, code in enumerate(recent_code, 1):
            speculation += f"{i}. {code.content[:150]}...\n"
        
        return speculation
    
    def _generate_general_speculation(self, need: Dict[str, Any], 
                                    current_context: List[ContextChunk]) -> str:
        """Generate general speculative context."""
        return f"Speculative context for {need['context_type'].value}: {need['reason']}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        keywords = text.split()
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in keywords if word not in common_words and len(word) > 2]
    
    def _extract_topic(self, message: Dict[str, Any]) -> str:
        """Extract topic from a message."""
        content = message.get('content', '')
        if not content:
            return 'unknown'
        
        # Simple topic extraction based on first few words
        words = content.split()[:5]
        return ' '.join(words).lower()


class DynamicContextGenerator:
    """
    Advanced dynamic context generator that incorporates speculation, attention mechanisms,
    and intelligent context switching for optimal LLM performance.
    """
    
    def __init__(self, config: Optional[DynamicContextConfig] = None):
        self.config = config or DynamicContextConfig()
        self.logger = logging.getLogger("dynamic_context_generator")
        
        # Core components
        self.sra = ScaledReAttention(self.config.attention_window_size)
        self.speculator = ContextSpeculator(self.config)
        
        # State management
        self.context_cache = {}
        self.attention_history = deque(maxlen=1000)
        self.context_history = deque(maxlen=1000)
        self.speculation_history = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize embedding model if available
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Initialized sentence transformer for embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedding model: {e}")
    
    async def generate_dynamic_context(self, 
                                     user_query: str,
                                     conversation_history: List[Dict[str, Any]],
                                     available_context: List[ContextChunk],
                                     speculation_level: SpeculationLevel = SpeculationLevel.MEDIUM) -> Dict[str, Any]:
        """
        Generate dynamic context with speculation and attention mechanisms.
        
        Args:
            user_query: Current user query
            conversation_history: History of conversation
            available_context: Available context chunks
            speculation_level: Level of speculation to apply
            
        Returns:
            Dictionary containing generated context and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze current context and generate embeddings
            context_embeddings = await self._generate_context_embeddings(available_context)
            
            # Step 2: Apply attention mechanism
            attention_context = await self._apply_attention_mechanism(
                user_query, context_embeddings, available_context
            )
            
            # Step 3: Generate speculative context
            speculative_context = []
            if self.config.speculation_enabled and speculation_level != SpeculationLevel.NONE:
                speculative_context = self.speculator.speculate_context(
                    available_context, user_query, conversation_history
                )
            
            # Step 4: Compress and optimize context
            optimized_context = await self._compress_context(
                attention_context + speculative_context
            )
            
            # Step 5: Apply dynamic context switching
            final_context = await self._apply_dynamic_switching(
                optimized_context, user_query, conversation_history
            )
            
            # Step 6: Generate final context
            result = {
                'context_chunks': final_context,
                'total_tokens': sum(chunk.token_count for chunk in final_context),
                'speculation_chunks': len(speculative_context),
                'attention_applied': len(attention_context),
                'compression_ratio': len(final_context) / max(len(available_context), 1),
                'generation_time': time.time() - start_time,
                'metadata': {
                    'speculation_level': speculation_level.value,
                    'attention_enabled': self.config.attention_mechanism_enabled,
                    'dynamic_switching_enabled': self.config.dynamic_switching_enabled,
                    'compression_enabled': self.config.compression_enabled
                }
            }
            
            # Update history
            self._update_history(result)
            
            self.logger.info(f"Generated dynamic context with {len(final_context)} chunks, "
                           f"{result['total_tokens']} tokens in {result['generation_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating dynamic context: {e}")
            return {
                'context_chunks': available_context[:10],  # Fallback to first 10 chunks
                'total_tokens': 0,
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    
    async def _generate_context_embeddings(self, context_chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Generate embeddings for context chunks."""
        if not self.embedding_model:
            return context_chunks
        
        try:
            for chunk in context_chunks:
                if not chunk.embedding:
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk.content)
                    chunk.embedding = embedding.tolist()
            
            return context_chunks
        except Exception as e:
            self.logger.warning(f"Error generating embeddings: {e}")
            return context_chunks
    
    async def _apply_attention_mechanism(self, user_query: str, 
                                       context_chunks: List[ContextChunk],
                                       original_chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Apply attention mechanism to context chunks."""
        if not self.config.attention_mechanism_enabled or not context_chunks:
            return context_chunks
        
        try:
            # Generate query embedding
            if self.embedding_model:
                query_embedding = self.embedding_model.encode(user_query).tolist()
            else:
                # Simple fallback
                query_embedding = [1.0] * 384  # Default embedding size
            
            # Prepare keys and values
            keys = []
            values = []
            for chunk in context_chunks:
                if chunk.embedding:
                    keys.append(chunk.embedding)
                    values.append(chunk.embedding)
                else:
                    # Fallback to simple representation
                    keys.append([1.0] * 384)
                    values.append([1.0] * 384)
            
            # Calculate attention scores
            attention_weights, attention_output = self.sra.calculate_attention_scores(
                query_embedding, keys, values
            )
            
            # Apply attention weights to chunks
            attended_chunks = []
            for i, chunk in enumerate(context_chunks):
                if i < len(attention_weights):
                    # Create new chunk with attention weights
                    attended_chunk = ContextChunk(
                        content=chunk.content,
                        context_type=chunk.context_type,
                        relevance_score=chunk.relevance_score * attention_weights[i],
                        timestamp=chunk.timestamp,
                        source=chunk.source,
                        metadata=chunk.metadata.copy(),
                        attention_weights=[attention_weights[i]],
                        speculation_level=chunk.speculation_level,
                        token_count=chunk.token_count,
                        embedding=chunk.embedding,
                        chunk_id=chunk.chunk_id
                    )
                    attended_chunks.append(attended_chunk)
            
            # Sort by attention-weighted relevance
            attended_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return attended_chunks
            
        except Exception as e:
            self.logger.warning(f"Error applying attention mechanism: {e}")
            return context_chunks
    
    async def _compress_context(self, context_chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Compress context to fit within token limits."""
        if not self.config.compression_enabled:
            return context_chunks
        
        try:
            total_tokens = sum(chunk.token_count for chunk in context_chunks)
            
            if total_tokens <= self.config.max_context_tokens:
                return context_chunks
            
            # Calculate compression target
            target_tokens = int(self.config.max_context_tokens * self.config.context_compression_ratio)
            
            # Sort by relevance and compress
            sorted_chunks = sorted(context_chunks, key=lambda x: x.relevance_score, reverse=True)
            
            compressed_chunks = []
            current_tokens = 0
            
            for chunk in sorted_chunks:
                if current_tokens + chunk.token_count <= target_tokens:
                    compressed_chunks.append(chunk)
                    current_tokens += chunk.token_count
                else:
                    # Try to compress this chunk
                    compressed_chunk = self._compress_chunk(chunk, target_tokens - current_tokens)
                    if compressed_chunk:
                        compressed_chunks.append(compressed_chunk)
                        current_tokens += compressed_chunk.token_count
                    break
            
            self.logger.info(f"Compressed context from {total_tokens} to {current_tokens} tokens")
            return compressed_chunks
            
        except Exception as e:
            self.logger.warning(f"Error compressing context: {e}")
            return context_chunks[:10]  # Fallback to first 10 chunks
    
    def _compress_chunk(self, chunk: ContextChunk, max_tokens: int) -> Optional[ContextChunk]:
        """Compress a single chunk to fit within token limit."""
        if chunk.token_count <= max_tokens:
            return chunk
        
        # Simple compression: take first part of content
        words = chunk.content.split()
        compressed_words = words[:max_tokens]
        compressed_content = ' '.join(compressed_words) + '...'
        
        return ContextChunk(
            content=compressed_content,
            context_type=chunk.context_type,
            relevance_score=chunk.relevance_score,
            timestamp=chunk.timestamp,
            source=chunk.source,
            metadata={**chunk.metadata, 'compressed': True},
            attention_weights=chunk.attention_weights,
            speculation_level=chunk.speculation_level,
            token_count=len(compressed_words),
            embedding=chunk.embedding,
            chunk_id=chunk.chunk_id + "_compressed"
        )
    
    async def _apply_dynamic_switching(self, context_chunks: List[ContextChunk],
                                     user_query: str,
                                     conversation_history: List[Dict[str, Any]]) -> List[ContextChunk]:
        """Apply dynamic context switching based on current needs."""
        if not self.config.dynamic_switching_enabled:
            return context_chunks
        
        try:
            # Analyze current needs
            current_needs = self._analyze_current_needs(user_query, conversation_history)
            
            # Reorder context based on needs
            reordered_chunks = []
            
            # Prioritize chunks that match current needs
            for need in current_needs:
                matching_chunks = [c for c in context_chunks if c.context_type == need['type']]
                reordered_chunks.extend(matching_chunks)
            
            # Add remaining chunks
            used_chunks = set(chunk.chunk_id for chunk in reordered_chunks)
            remaining_chunks = [c for c in context_chunks if c.chunk_id not in used_chunks]
            reordered_chunks.extend(remaining_chunks)
            
            return reordered_chunks
            
        except Exception as e:
            self.logger.warning(f"Error applying dynamic switching: {e}")
            return context_chunks
    
    def _analyze_current_needs(self, user_query: str, 
                             conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze current context needs based on query and history."""
        needs = []
        
        # Analyze query for context needs
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ['memory', 'remember', 'recall']):
            needs.append({'type': ContextType.MEMORY, 'priority': 1})
        
        if any(word in query_lower for word in ['task', 'todo', 'project']):
            needs.append({'type': ContextType.TASK, 'priority': 1})
        
        if any(word in query_lower for word in ['code', 'function', 'class']):
            needs.append({'type': ContextType.CODE, 'priority': 1})
        
        if any(word in query_lower for word in ['document', 'file', 'read']):
            needs.append({'type': ContextType.DOCUMENT, 'priority': 1})
        
        # Analyze conversation history for patterns
        if conversation_history:
            recent_messages = conversation_history[-5:]  # Last 5 messages
            for message in recent_messages:
                content = message.get('content', '').lower()
                if any(word in content for word in ['memory', 'remember']):
                    needs.append({'type': ContextType.MEMORY, 'priority': 2})
                if any(word in content for word in ['task', 'todo']):
                    needs.append({'type': ContextType.TASK, 'priority': 2})
        
        return needs
    
    def _update_history(self, result: Dict[str, Any]):
        """Update context generation history."""
        with self._lock:
            self.context_history.append({
                'timestamp': datetime.now(),
                'total_tokens': result['total_tokens'],
                'speculation_chunks': result['speculation_chunks'],
                'generation_time': result['generation_time']
            })
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context generation."""
        with self._lock:
            if not self.context_history:
                return {'error': 'No context history available'}
            
            recent_contexts = list(self.context_history)[-100:]  # Last 100 generations
            
            return {
                'total_generations': len(self.context_history),
                'average_tokens': np.mean([c['total_tokens'] for c in recent_contexts]),
                'average_generation_time': np.mean([c['generation_time'] for c in recent_contexts]),
                'average_speculation_chunks': np.mean([c['speculation_chunks'] for c in recent_contexts]),
                'total_speculation_chunks': sum(c['speculation_chunks'] for c in recent_contexts),
                'compression_ratio': np.mean([c.get('compression_ratio', 1.0) for c in recent_contexts])
            }


# CLI integration
def create_dynamic_context_generator(config: Optional[DynamicContextConfig] = None) -> DynamicContextGenerator:
    """Create a dynamic context generator instance."""
    return DynamicContextGenerator(config)


if __name__ == "__main__":
    # Example usage
    config = DynamicContextConfig(
        max_context_tokens=4000,
        speculation_enabled=True,
        attention_mechanism_enabled=True,
        dynamic_switching_enabled=True
    )
    
    generator = DynamicContextGenerator(config)
    
    # Example context chunks
    chunks = [
        ContextChunk(
            content="This is a memory about the project structure",
            context_type=ContextType.MEMORY,
            relevance_score=0.8,
            timestamp=datetime.now(),
            source="memory_system"
        ),
        ContextChunk(
            content="Current task: Implement user authentication",
            context_type=ContextType.TASK,
            relevance_score=0.9,
            timestamp=datetime.now(),
            source="task_system"
        )
    ]
    
    # Example usage
    async def example():
        result = await generator.generate_dynamic_context(
            user_query="What was the last task I was working on?",
            conversation_history=[],
            available_context=chunks,
            speculation_level=SpeculationLevel.MEDIUM
        )
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(example()) 