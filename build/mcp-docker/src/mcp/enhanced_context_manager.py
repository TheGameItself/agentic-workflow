#!/usr/bin/env python3
"""
Enhanced Context Manager with Dynamic Context Generation
Integrates dynamic context generation with speculation and attention mechanisms
into the existing MCP context management system.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .dynamic_context_generator import (
    DynamicContextGenerator, DynamicContextConfig, ContextChunk, 
    ContextType, SpeculationLevel
)
from .memory import MemoryManager
from .task_manager import TaskManager
from .unified_memory import UnifiedMemoryManager


@dataclass
class EnhancedContextConfig:
    """Configuration for enhanced context management."""
    enable_dynamic_generation: bool = True
    enable_speculation: bool = True
    enable_attention: bool = True
    enable_compression: bool = True
    max_context_tokens: int = 4000
    speculation_level: SpeculationLevel = SpeculationLevel.MEDIUM
    cache_context: bool = True
    cache_ttl_seconds: int = 3600
    auto_optimize: bool = True


class EnhancedContextManager:
    """
    Enhanced context manager that integrates dynamic context generation
    with the existing MCP context management system.
    """
    
    def __init__(self, config: Optional[EnhancedContextConfig] = None):
        self.config = config or EnhancedContextConfig()
        self.logger = logging.getLogger("enhanced_context_manager")
        
        # Core components
        self.dynamic_generator = DynamicContextGenerator(
            DynamicContextConfig(
                max_context_tokens=self.config.max_context_tokens,
                speculation_enabled=self.config.enable_speculation,
                attention_mechanism_enabled=self.config.enable_attention,
                compression_enabled=self.config.enable_compression
            )
        )
        
        # Context cache
        self.context_cache = {}
        self.cache_timestamps = {}
        
        # Statistics
        self.generation_stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'average_generation_time': 0,
            'speculation_hits': 0,
            'cache_hits': 0
        }
    
    async def get_enhanced_context(self, 
                                 user_query: str,
                                 conversation_history: List[Dict[str, Any]],
                                 memory_manager: Optional[MemoryManager] = None,
                                 task_manager: Optional[TaskManager] = None,
                                 unified_memory: Optional[UnifiedMemoryManager] = None,
                                 context_types: Optional[List[str]] = None,
                                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Get enhanced context with dynamic generation and speculation.
        
        Args:
            user_query: Current user query
            conversation_history: History of conversation
            memory_manager: Memory manager instance
            task_manager: Task manager instance
            unified_memory: Unified memory manager instance
            context_types: Types of context to include
            max_tokens: Maximum tokens for context
            
        Returns:
            Dictionary containing enhanced context and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(user_query, context_types)
            if self.config.cache_context and cache_key in self.context_cache:
                if self._is_cache_valid(cache_key):
                    self.generation_stats['cache_hits'] += 1
                    cached_result = self.context_cache[cache_key].copy()
                    cached_result['cached'] = True
                    return cached_result
            
            # Collect available context
            available_context = await self._collect_available_context(
                memory_manager, task_manager, unified_memory, context_types
            )
            
            # Generate dynamic context
            if self.config.enable_dynamic_generation:
                result = await self.dynamic_generator.generate_dynamic_context(
                    user_query=user_query,
                    conversation_history=conversation_history,
                    available_context=available_context,
                    speculation_level=self.config.speculation_level
                )
            else:
                # Fallback to simple context
                result = {
                    'context_chunks': available_context[:10],
                    'total_tokens': sum(chunk.token_count for chunk in available_context[:10]),
                    'speculation_chunks': 0,
                    'attention_applied': 0,
                    'compression_ratio': 1.0,
                    'generation_time': time.time() - start_time,
                    'metadata': {
                        'speculation_level': 'none',
                        'attention_enabled': False,
                        'dynamic_switching_enabled': False,
                        'compression_enabled': False
                    }
                }
            
            # Apply token limit if specified
            if max_tokens and result['total_tokens'] > max_tokens:
                result = await self._apply_token_limit(result, max_tokens)
            
            # Update statistics
            self._update_statistics(result, time.time() - start_time)
            
            # Cache result
            if self.config.cache_context:
                self.context_cache[cache_key] = result.copy()
                self.cache_timestamps[cache_key] = datetime.now()
            
            # Add metadata
            result['enhanced'] = True
            result['config'] = {
                'dynamic_generation': self.config.enable_dynamic_generation,
                'speculation': self.config.enable_speculation,
                'attention': self.config.enable_attention,
                'compression': self.config.enable_compression
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced context: {e}")
            return {
                'context_chunks': [],
                'total_tokens': 0,
                'error': str(e),
                'generation_time': time.time() - start_time,
                'enhanced': False
            }
    
    async def _collect_available_context(self, 
                                       memory_manager: Optional[MemoryManager],
                                       task_manager: Optional[TaskManager],
                                       unified_memory: Optional[UnifiedMemoryManager],
                                       context_types: Optional[List[str]]) -> List[ContextChunk]:
        """Collect available context from various sources."""
        context_chunks = []
        
        # Collect memory context
        if memory_manager and (not context_types or 'memory' in context_types):
            memory_chunks = await self._collect_memory_context(memory_manager)
            context_chunks.extend(memory_chunks)
        
        # Collect task context
        if task_manager and (not context_types or 'task' in context_types):
            task_chunks = await self._collect_task_context(task_manager)
            context_chunks.extend(task_chunks)
        
        # Collect unified memory context
        if unified_memory and (not context_types or 'unified_memory' in context_types):
            unified_chunks = await self._collect_unified_memory_context(unified_memory)
            context_chunks.extend(unified_chunks)
        
        # Collect conversation context
        if not context_types or 'conversation' in context_types:
            conversation_chunks = await self._collect_conversation_context()
            context_chunks.extend(conversation_chunks)
        
        return context_chunks
    
    async def _collect_memory_context(self, memory_manager: MemoryManager) -> List[ContextChunk]:
        """Collect context from memory manager."""
        try:
            # Get recent memories
            memories = memory_manager.get_recent_memories(limit=20)
            
            chunks = []
            for memory in memories:
                chunk = ContextChunk(
                    content=memory.get('content', ''),
                    context_type=ContextType.MEMORY,
                    relevance_score=memory.get('relevance', 0.5),
                    timestamp=datetime.fromisoformat(memory.get('timestamp', datetime.now().isoformat())),
                    source='memory_manager',
                    metadata={
                        'memory_id': memory.get('id'),
                        'memory_type': memory.get('type', 'general'),
                        'tags': memory.get('tags', [])
                    },
                    token_count=len(memory.get('content', '').split()),
                    chunk_id=f"memory_{memory.get('id', 'unknown')}"
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            self.logger.warning(f"Error collecting memory context: {e}")
            return []
    
    async def _collect_task_context(self, task_manager: TaskManager) -> List[ContextChunk]:
        """Collect context from task manager."""
        try:
            # Get active and recent tasks
            tasks = task_manager.get_active_tasks() + task_manager.get_recent_tasks(limit=10)
            
            chunks = []
            for task in tasks:
                chunk = ContextChunk(
                    content=f"Task: {task.get('title', '')} - {task.get('description', '')}",
                    context_type=ContextType.TASK,
                    relevance_score=0.8 if task.get('status') == 'active' else 0.5,
                    timestamp=datetime.fromisoformat(task.get('created_at', datetime.now().isoformat())),
                    source='task_manager',
                    metadata={
                        'task_id': task.get('id'),
                        'status': task.get('status'),
                        'priority': task.get('priority'),
                        'progress': task.get('progress', 0)
                    },
                    token_count=len(f"{task.get('title', '')} {task.get('description', '')}".split()),
                    chunk_id=f"task_{task.get('id', 'unknown')}"
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            self.logger.warning(f"Error collecting task context: {e}")
            return []
    
    async def _collect_unified_memory_context(self, unified_memory: UnifiedMemoryManager) -> List[ContextChunk]:
        """Collect context from unified memory manager."""
        try:
            # Get recent unified memory entries
            entries = unified_memory.get_recent_entries(limit=15)
            
            chunks = []
            for entry in entries:
                chunk = ContextChunk(
                    content=entry.get('content', ''),
                    context_type=ContextType.WORKING_MEMORY,
                    relevance_score=entry.get('relevance', 0.6),
                    timestamp=datetime.fromisoformat(entry.get('timestamp', datetime.now().isoformat())),
                    source='unified_memory',
                    metadata={
                        'entry_id': entry.get('id'),
                        'memory_type': entry.get('type', 'working'),
                        'access_count': entry.get('access_count', 0)
                    },
                    token_count=len(entry.get('content', '').split()),
                    chunk_id=f"unified_{entry.get('id', 'unknown')}"
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            self.logger.warning(f"Error collecting unified memory context: {e}")
            return []
    
    async def _collect_conversation_context(self) -> List[ContextChunk]:
        """Collect context from conversation history."""
        try:
            # This would be implemented based on conversation storage
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.warning(f"Error collecting conversation context: {e}")
            return []
    
    async def _apply_token_limit(self, result: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Apply token limit to context result."""
        if result['total_tokens'] <= max_tokens:
            return result
        
        # Sort chunks by relevance and take top ones
        chunks = result['context_chunks']
        sorted_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)
        
        limited_chunks = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            if current_tokens + chunk.token_count <= max_tokens:
                limited_chunks.append(chunk)
                current_tokens += chunk.token_count
            else:
                break
        
        result['context_chunks'] = limited_chunks
        result['total_tokens'] = current_tokens
        result['token_limit_applied'] = True
        
        return result
    
    def _generate_cache_key(self, user_query: str, context_types: Optional[List[str]]) -> str:
        """Generate cache key for context."""
        import hashlib
        
        key_data = {
            'query': user_query.lower(),
            'types': sorted(context_types) if context_types else [],
            'config': {
                'speculation': self.config.enable_speculation,
                'attention': self.config.enable_attention,
                'compression': self.config.enable_compression
            }
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached context is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        timestamp = self.cache_timestamps[cache_key]
        age = (datetime.now() - timestamp).total_seconds()
        
        return age < self.config.cache_ttl_seconds
    
    def _update_statistics(self, result: Dict[str, Any], generation_time: float):
        """Update generation statistics."""
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_tokens_generated'] += result.get('total_tokens', 0)
        
        # Update average generation time
        current_avg = self.generation_stats['average_generation_time']
        total_gens = self.generation_stats['total_generations']
        self.generation_stats['average_generation_time'] = (
            (current_avg * (total_gens - 1) + generation_time) / total_gens
        )
        
        # Update speculation hits
        if result.get('speculation_chunks', 0) > 0:
            self.generation_stats['speculation_hits'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context generation statistics."""
        stats = self.generation_stats.copy()
        
        # Add dynamic generator statistics
        dynamic_stats = self.dynamic_generator.get_context_statistics()
        stats['dynamic_generator'] = dynamic_stats
        
        # Add cache statistics
        stats['cache'] = {
            'total_cached': len(self.context_cache),
            'cache_hit_rate': (
                stats['cache_hits'] / max(stats['total_generations'], 1)
            )
        }
        
        return stats
    
    def clear_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Context cache cleared")
    
    def optimize_config(self, performance_data: Dict[str, Any]):
        """Optimize configuration based on performance data."""
        if not self.config.auto_optimize:
            return
        
        try:
            # Analyze performance and adjust configuration
            avg_tokens = performance_data.get('average_tokens', 0)
            avg_time = performance_data.get('average_generation_time', 0)
            cache_hit_rate = performance_data.get('cache_hit_rate', 0)
            
            # Adjust speculation level based on performance
            if avg_time > 2.0:  # If generation takes too long
                if self.config.speculation_level == SpeculationLevel.AGGRESSIVE:
                    self.config.speculation_level = SpeculationLevel.HIGH
                elif self.config.speculation_level == SpeculationLevel.HIGH:
                    self.config.speculation_level = SpeculationLevel.MEDIUM
            
            # Adjust compression based on token usage
            if avg_tokens > self.config.max_context_tokens * 0.8:
                self.config.enable_compression = True
            
            # Adjust cache TTL based on hit rate
            if cache_hit_rate < 0.3:
                self.config.cache_ttl_seconds = min(self.config.cache_ttl_seconds * 2, 7200)
            elif cache_hit_rate > 0.7:
                self.config.cache_ttl_seconds = max(self.config.cache_ttl_seconds // 2, 1800)
            
            self.logger.info(f"Configuration optimized based on performance data")
            
        except Exception as e:
            self.logger.warning(f"Error optimizing configuration: {e}")


# CLI integration
def create_enhanced_context_manager(config: Optional[EnhancedContextConfig] = None) -> EnhancedContextManager:
    """Create an enhanced context manager instance."""
    return EnhancedContextManager(config)


if __name__ == "__main__":
    # Example usage
    config = EnhancedContextConfig(
        enable_dynamic_generation=True,
        enable_speculation=True,
        enable_attention=True,
        max_context_tokens=4000
    )
    
    manager = EnhancedContextManager(config)
    
    # Example usage
    async def example():
        result = await manager.get_enhanced_context(
            user_query="What was the last task I was working on?",
            conversation_history=[],
            context_types=['memory', 'task']
        )
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(example()) 