#!/usr/bin/env python3
"""
Enhanced Context Manager for MCP Core System
Advanced context generation, management, and optimization for LLM consumption.
"""

import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import hashlib
import tiktoken
from functools import lru_cache

# Try to import advanced features
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class ContextType(Enum):
    """Types of context that can be generated."""
    MEMORY = "memory"
    TASK = "task"
    WORKFLOW = "workflow"
    PROJECT = "project"
    CODE = "code"
    DOCUMENTATION = "documentation"
    ERROR = "error"
    PERFORMANCE = "performance"
    SYSTEM = "system"

class ContextPriority(Enum):
    """Priority levels for context items."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MINIMAL = 5

@dataclass
class ContextItem:
    """Individual context item with metadata."""
    content: str
    context_type: ContextType
    priority: ContextPriority
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    relevance_score: float = 0.0
    embedding: Optional[List[float]] = None

@dataclass
class ContextPack:
    """Collection of context items with optimization metadata."""
    name: str
    description: str
    items: List[ContextItem]
    total_tokens: int
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    optimization_level: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)

class TokenBudgetManager:
    """Manages token budgets for context generation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.encoding = None
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except:
                self.encoding = None
        
        # Model-specific token limits
        self.model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "claude-3-haiku": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-opus": 200000,
        }
        
        self.max_tokens = self.model_limits.get(model_name, 4096)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: rough estimation
            return len(text.split()) * 1.3
    
    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate and decode
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        else:
            # Fallback: word-based truncation
            words = text.split()
            estimated_tokens = len(words) * 1.3
            
            if estimated_tokens <= max_tokens:
                return text
            
            # Truncate words
            target_words = int(max_tokens / 1.3)
            return " ".join(words[:target_words])
    
    def optimize_for_budget(self, items: List[ContextItem], max_tokens: int) -> List[ContextItem]:
        """Optimize context items to fit within token budget."""
        # Sort by priority and relevance
        sorted_items = sorted(
            items,
            key=lambda x: (x.priority.value, -x.relevance_score)
        )
        
        optimized_items = []
        current_tokens = 0
        
        for item in sorted_items:
            item_tokens = self.count_tokens(item.content)
            
            if current_tokens + item_tokens <= max_tokens:
                # Item fits completely
                item.tokens = item_tokens
                optimized_items.append(item)
                current_tokens += item_tokens
            elif current_tokens < max_tokens:
                # Truncate item to fit remaining budget
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # Only truncate if meaningful space left
                    truncated_content = self.truncate_to_budget(item.content, remaining_tokens)
                    item.content = truncated_content
                    item.tokens = self.count_tokens(truncated_content)
                    optimized_items.append(item)
                    break
            else:
                # No more space
                break
        
        return optimized_items

class SemanticAnalyzer:
    """Analyzes semantic similarity and relevance."""
    
    def __init__(self):
        self.model = None
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.model = None
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get semantic embedding for text."""
        if not self.model:
            return None
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        with self.cache_lock:
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        try:
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Cache result
            with self.cache_lock:
                self.cache[text_hash] = embedding
                
                # Limit cache size
                if len(self.cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(self.cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.cache[key]
            
            return embedding
            
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not NUMPY_AVAILABLE or not embedding1 or not embedding2:
            return 0.0
        
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logging.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_most_relevant(self, query: str, items: List[ContextItem], top_k: int = 10) -> List[ContextItem]:
        """Find most relevant context items for a query."""
        if not self.model:
            # Fallback to keyword matching
            return self._keyword_relevance(query, items, top_k)
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return self._keyword_relevance(query, items, top_k)
            
            # Calculate relevance scores
            for item in items:
                if not item.embedding:
                    item.embedding = self.get_embedding(item.content)
                
                if item.embedding:
                    item.relevance_score = self.calculate_similarity(query_embedding, item.embedding)
                else:
                    item.relevance_score = self._keyword_similarity(query, item.content)
            
            # Sort by relevance and return top k
            sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
            return sorted_items[:top_k]
            
        except Exception as e:
            logging.error(f"Semantic relevance calculation failed: {e}")
            return self._keyword_relevance(query, items, top_k)
    
    def _keyword_relevance(self, query: str, items: List[ContextItem], top_k: int) -> List[ContextItem]:
        """Fallback keyword-based relevance calculation."""
        query_words = set(query.lower().split())
        
        for item in items:
            item.relevance_score = self._keyword_similarity(query, item.content)
        
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
        return sorted_items[:top_k]
    
    def _keyword_similarity(self, query: str, text: str) -> float:
        """Calculate keyword-based similarity score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0

class ContextManager:
    """
    Enhanced Context Manager for MCP Core System.
    
    Features:
    - Multi-source context aggregation
    - Intelligent token budget management
    - Semantic relevance scoring
    - Context optimization and caching
    - Dynamic context adaptation
    - Performance monitoring
    """
    
    def __init__(self, default_model: str = "gpt-3.5-turbo"):
        self.default_model = default_model
        self.token_manager = TokenBudgetManager(default_model)
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Context storage
        self.context_packs: Dict[str, ContextPack] = {}
        self.context_cache: Dict[str, Tuple[str, datetime]] = {}
        
        # Performance tracking
        self.generation_stats = {
            'total_generations': 0,
            'cache_hits': 0,
            'average_generation_time': 0.0,
            'total_tokens_generated': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Context Manager initialized")
    
    def create_context_item(self, content: str, context_type: ContextType, 
                          priority: ContextPriority = ContextPriority.MEDIUM,
                          source: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        """Create a new context item."""
        item = ContextItem(
            content=content,
            context_type=context_type,
            priority=priority,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {},
            tokens=self.token_manager.count_tokens(content)
        )
        
        # Generate embedding if semantic analyzer is available
        if self.semantic_analyzer.model:
            item.embedding = self.semantic_analyzer.get_embedding(content)
        
        return item
    
    def export_context(self, context_types: List[str], max_tokens: int = 1000,
                      format_type: str = "text", query: Optional[str] = None,
                      use_semantic_search: bool = True, optimization_level: str = "standard") -> Dict[str, Any]:
        """
        Export optimized context for LLM consumption.
        
        Args:
            context_types: Types of context to include
            max_tokens: Maximum token budget
            format_type: Output format (text, json, markdown)
            query: Optional query for relevance scoring
            use_semantic_search: Whether to use semantic similarity
            optimization_level: Level of optimization (minimal, standard, aggressive)
            
        Returns:
            Dictionary containing optimized context and metadata
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Generate cache key
                cache_key = self._generate_cache_key(context_types, max_tokens, query, optimization_level)
                
                # Check cache
                if cache_key in self.context_cache:
                    cached_context, cache_time = self.context_cache[cache_key]
                    if datetime.now() - cache_time < timedelta(minutes=5):  # 5-minute cache
                        self.generation_stats['cache_hits'] += 1
                        return json.loads(cached_context)
                
                # Collect context items
                context_items = self._collect_context_items(context_types)
                
                # Apply semantic filtering if query provided
                if query and use_semantic_search and context_items:
                    context_items = self.semantic_analyzer.find_most_relevant(
                        query, context_items, top_k=min(50, len(context_items))
                    )
                
                # Optimize for token budget
                optimized_items = self.token_manager.optimize_for_budget(context_items, max_tokens)
                
                # Apply optimization level
                if optimization_level == "aggressive":
                    optimized_items = self._apply_aggressive_optimization(optimized_items, max_tokens)
                elif optimization_level == "minimal":
                    optimized_items = self._apply_minimal_optimization(optimized_items)
                
                # Format output
                result = self._format_context_output(optimized_items, format_type, max_tokens)
                
                # Update statistics
                generation_time = time.time() - start_time
                self._update_generation_stats(generation_time, result.get('total_tokens', 0))
                
                # Cache result
                self.context_cache[cache_key] = (json.dumps(result), datetime.now())
                
                # Limit cache size
                if len(self.context_cache) > 100:
                    self._cleanup_cache()
                
                return result
                
        except Exception as e:
            self.logger.error(f"Context export failed: {e}")
            return {
                'error': str(e),
                'context': '',
                'total_tokens': 0,
                'items_included': 0,
                'generation_time': time.time() - start_time
            }
    
    def _collect_context_items(self, context_types: List[str]) -> List[ContextItem]:
        """Collect context items from various sources."""
        items = []
        
        try:
            # Import managers (with fallback handling)
            try:
                from .memory import MemoryManager
                from .task_manager import TaskManager
                from .workflow import WorkflowManager
                from .project_manager import ProjectManager
                
                memory_manager = MemoryManager()
                task_manager = TaskManager()
                workflow_manager = WorkflowManager()
                project_manager = ProjectManager()
                
            except ImportError:
                self.logger.warning("Some managers not available for context collection")
                return items
            
            for context_type in context_types:
                if context_type == "memory" or context_type == "memories":
                    items.extend(self._collect_memory_context(memory_manager))
                elif context_type == "task" or context_type == "tasks":
                    items.extend(self._collect_task_context(task_manager))
                elif context_type == "workflow":
                    items.extend(self._collect_workflow_context(workflow_manager))
                elif context_type == "project":
                    items.extend(self._collect_project_context(project_manager))
                elif context_type == "system":
                    items.extend(self._collect_system_context())
                elif context_type == "performance":
                    items.extend(self._collect_performance_context())
            
        except Exception as e:
            self.logger.error(f"Error collecting context items: {e}")
        
        return items
    
    def _collect_memory_context(self, memory_manager) -> List[ContextItem]:
        """Collect context from memory system."""
        items = []
        
        try:
            # Get recent high-priority memories
            memories = memory_manager.search_memories("", limit=20)  # Get recent memories
            
            for memory in memories:
                priority = ContextPriority.HIGH if memory.get('priority', 0) > 0.7 else ContextPriority.MEDIUM
                
                item = self.create_context_item(
                    content=f"Memory: {memory.get('text', '')}",
                    context_type=ContextType.MEMORY,
                    priority=priority,
                    source="memory_manager",
                    metadata={
                        'memory_id': memory.get('id'),
                        'memory_type': memory.get('memory_type'),
                        'created_at': memory.get('created_at'),
                        'tags': memory.get('tags', [])
                    }
                )
                items.append(item)
                
        except Exception as e:
            self.logger.error(f"Error collecting memory context: {e}")
        
        return items
    
    def _collect_task_context(self, task_manager) -> List[ContextItem]:
        """Collect context from task system."""
        items = []
        
        try:
            # Get active and recent tasks
            tasks = task_manager.get_tasks()
            
            for task in tasks[:15]:  # Limit to recent tasks
                priority = ContextPriority.CRITICAL if task.get('accuracy_critical') else ContextPriority.MEDIUM
                
                content = f"Task: {task.get('title', '')}"
                if task.get('description'):
                    content += f"\nDescription: {task['description']}"
                content += f"\nStatus: {task.get('status', 'unknown')}"
                content += f"\nPriority: {task.get('priority', 0)}"
                
                item = self.create_context_item(
                    content=content,
                    context_type=ContextType.TASK,
                    priority=priority,
                    source="task_manager",
                    metadata={
                        'task_id': task.get('id'),
                        'status': task.get('status'),
                        'priority': task.get('priority'),
                        'created_at': task.get('created_at')
                    }
                )
                items.append(item)
                
        except Exception as e:
            self.logger.error(f"Error collecting task context: {e}")
        
        return items
    
    def _collect_workflow_context(self, workflow_manager) -> List[ContextItem]:
        """Collect context from workflow system."""
        items = []
        
        try:
            status = workflow_manager.get_workflow_status()
            
            content = f"Workflow Status:\n"
            content += f"Progress: {status.get('progress', 0):.1%}\n"
            content += f"Current Step: {status.get('current_step', 'None')}\n"
            
            if status.get('completed_steps'):
                content += f"Completed Steps: {', '.join(status['completed_steps'])}\n"
            
            item = self.create_context_item(
                content=content,
                context_type=ContextType.WORKFLOW,
                priority=ContextPriority.HIGH,
                source="workflow_manager",
                metadata=status
            )
            items.append(item)
            
        except Exception as e:
            self.logger.error(f"Error collecting workflow context: {e}")
        
        return items
    
    def _collect_project_context(self, project_manager) -> List[ContextItem]:
        """Collect context from project system."""
        items = []
        
        try:
            project_info = project_manager.get_project_info()
            
            if project_info:
                content = f"Project: {project_info.get('name', 'Unknown')}\n"
                content += f"Path: {project_info.get('path', 'Unknown')}\n"
                
                if project_info.get('description'):
                    content += f"Description: {project_info['description']}\n"
                
                item = self.create_context_item(
                    content=content,
                    context_type=ContextType.PROJECT,
                    priority=ContextPriority.HIGH,
                    source="project_manager",
                    metadata=project_info
                )
                items.append(item)
                
        except Exception as e:
            self.logger.error(f"Error collecting project context: {e}")
        
        return items
    
    def _collect_system_context(self) -> List[ContextItem]:
        """Collect system context information."""
        items = []
        
        try:
            import psutil
            
            content = f"System Status:\n"
            content += f"CPU Usage: {psutil.cpu_percent()}%\n"
            content += f"Memory Usage: {psutil.virtual_memory().percent}%\n"
            content += f"Disk Usage: {psutil.disk_usage('/').percent}%\n"
            
            item = self.create_context_item(
                content=content,
                context_type=ContextType.SYSTEM,
                priority=ContextPriority.LOW,
                source="system_monitor"
            )
            items.append(item)
            
        except Exception as e:
            self.logger.error(f"Error collecting system context: {e}")
        
        return items
    
    def _collect_performance_context(self) -> List[ContextItem]:
        """Collect performance context information."""
        items = []
        
        try:
            content = f"Performance Metrics:\n"
            content += f"Context Generations: {self.generation_stats['total_generations']}\n"
            content += f"Cache Hit Rate: {self.generation_stats['cache_hits'] / max(self.generation_stats['total_generations'], 1):.2%}\n"
            content += f"Average Generation Time: {self.generation_stats['average_generation_time']:.3f}s\n"
            
            item = self.create_context_item(
                content=content,
                context_type=ContextType.PERFORMANCE,
                priority=ContextPriority.LOW,
                source="context_manager"
            )
            items.append(item)
            
        except Exception as e:
            self.logger.error(f"Error collecting performance context: {e}")
        
        return items
    
    def _apply_aggressive_optimization(self, items: List[ContextItem], max_tokens: int) -> List[ContextItem]:
        """Apply aggressive optimization to reduce token usage."""
        optimized_items = []
        
        for item in items:
            # Summarize long content
            if item.tokens > 200:
                # Extract key sentences
                sentences = item.content.split('.')
                key_sentences = sentences[:3]  # Take first 3 sentences
                item.content = '. '.join(key_sentences) + '.'
                item.tokens = self.token_manager.count_tokens(item.content)
            
            # Remove low-priority items if over budget
            if sum(i.tokens for i in optimized_items) + item.tokens <= max_tokens:
                optimized_items.append(item)
            elif item.priority.value <= 2:  # Only include high priority items
                optimized_items.append(item)
                break
        
        return optimized_items
    
    def _apply_minimal_optimization(self, items: List[ContextItem]) -> List[ContextItem]:
        """Apply minimal optimization, preserving most content."""
        # Just sort by priority, minimal changes
        return sorted(items, key=lambda x: x.priority.value)
    
    def _format_context_output(self, items: List[ContextItem], format_type: str, max_tokens: int) -> Dict[str, Any]:
        """Format context items into the requested output format."""
        total_tokens = sum(item.tokens for item in items)
        
        if format_type.lower() == "json":
            context_data = {
                'items': [
                    {
                        'content': item.content,
                        'type': item.context_type.value,
                        'priority': item.priority.value,
                        'source': item.source,
                        'tokens': item.tokens,
                        'relevance_score': item.relevance_score,
                        'timestamp': item.timestamp.isoformat(),
                        'metadata': item.metadata
                    }
                    for item in items
                ]
            }
            context_str = json.dumps(context_data, indent=2)
            
        elif format_type.lower() == "markdown":
            context_parts = []
            for item in items:
                context_parts.append(f"## {item.context_type.value.title()} Context")
                context_parts.append(f"**Source:** {item.source}")
                context_parts.append(f"**Priority:** {item.priority.name}")
                context_parts.append(f"**Relevance:** {item.relevance_score:.3f}")
                context_parts.append("")
                context_parts.append(item.content)
                context_parts.append("")
            
            context_str = "\n".join(context_parts)
            
        else:  # Default to text format
            context_parts = []
            for item in items:
                context_parts.append(f"[{item.context_type.value.upper()}] {item.content}")
            
            context_str = "\n\n".join(context_parts)
        
        return {
            'context': context_str,
            'total_tokens': total_tokens,
            'items_included': len(items),
            'format': format_type,
            'optimization_applied': total_tokens <= max_tokens,
            'token_budget_used': min(total_tokens / max_tokens, 1.0) if max_tokens > 0 else 0,
            'generation_timestamp': datetime.now().isoformat(),
            'context_types': list(set(item.context_type.value for item in items))
        }
    
    def _generate_cache_key(self, context_types: List[str], max_tokens: int, 
                          query: Optional[str], optimization_level: str) -> str:
        """Generate cache key for context request."""
        key_data = f"{sorted(context_types)}:{max_tokens}:{query}:{optimization_level}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_generation_stats(self, generation_time: float, tokens_generated: int):
        """Update context generation statistics."""
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_tokens_generated'] += tokens_generated
        
        # Update average generation time
        total_gens = self.generation_stats['total_generations']
        current_avg = self.generation_stats['average_generation_time']
        self.generation_stats['average_generation_time'] = (
            (current_avg * (total_gens - 1) + generation_time) / total_gens
        )
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        keys_to_remove = []
        for key, (_, cache_time) in self.context_cache.items():
            if cache_time < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.context_cache[key]
    
    def save_context_pack(self, name: str, context_data: Dict[str, Any], 
                         description: str = "") -> str:
        """Save a context pack for reuse."""
        pack_id = hashlib.md5(f"{name}:{datetime.now()}".encode()).hexdigest()[:8]
        
        pack = ContextPack(
            name=name,
            description=description,
            items=[],  # Would need to reconstruct from context_data
            total_tokens=context_data.get('total_tokens', 0),
            created_at=datetime.now(),
            last_used=datetime.now(),
            metadata=context_data
        )
        
        with self.lock:
            self.context_packs[pack_id] = pack
        
        self.logger.info(f"Saved context pack: {name} (ID: {pack_id})")
        return pack_id
    
    def load_context_pack(self, pack_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved context pack."""
        with self.lock:
            if pack_id in self.context_packs:
                pack = self.context_packs[pack_id]
                pack.last_used = datetime.now()
                pack.usage_count += 1
                return pack.metadata
        
        return None
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        with self.lock:
            return {
                'generation_stats': self.generation_stats.copy(),
                'cache_size': len(self.context_cache),
                'context_packs': len(self.context_packs),
                'semantic_analyzer_available': self.semantic_analyzer.model is not None,
                'token_manager_model': self.token_manager.model_name,
                'max_tokens': self.token_manager.max_tokens
            }

# Convenience functions

def create_context_manager(model_name: str = "gpt-3.5-turbo") -> ContextManager:
    """Create a context manager with specified model."""
    return ContextManager(model_name)

def export_quick_context(context_types: List[str], max_tokens: int = 1000) -> str:
    """Quick context export with default settings."""
    manager = ContextManager()
    result = manager.export_context(context_types, max_tokens)
    return result.get('context', '')