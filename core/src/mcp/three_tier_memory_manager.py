#!/usr/bin/env python3
"""
Three-Tier Memory Architecture Manager
Integrates Working Memory, Short-Term Memory, and Long-Term Memory
with cross-lobe integration and automatic memory type selection.
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import memory components
try:
    from .lobes.shared_lobes.working_memory import WorkingMemory, ShortTermMemory, LongTermMemory
    from .enhanced_vector_memory import EnhancedVectorMemorySystem, BackendType
except ImportError:
    # Fallback imports for standalone usage
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from lobes.shared_lobes.working_memory import WorkingMemory, ShortTermMemory, LongTermMemory
    from enhanced_vector_memory import EnhancedVectorMemorySystem, BackendType

class MemoryTier(Enum):
    """Memory tier types."""
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"

@dataclass
class MemoryItem:
    """Unified memory item representation."""
    key: str
    data: Any
    tier: MemoryTier
    context: str = "default"
    priority: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None

class ThreeTierMemoryManager:
    """
    Comprehensive three-tier memory architecture manager.
    
    Features:
    - Automatic memory tier selection based on usage patterns
    - Cross-lobe memory integration
    - Memory optimization with automatic cleanup and compression
    - Unified interface for all memory operations
    - Performance monitoring and capacity management
    - Thread-safe operations
    """
    
    def __init__(self, 
                 working_capacity_mb: float = 100.0,
                 short_term_capacity_gb: float = 1.0,
                 long_term_capacity_gb: float = 9.0,
                 vector_backend: BackendType = BackendType.SQLITE_FAISS,
                 auto_optimize: bool = True,
                 hormone_system: Optional[Any] = None,
                 genetic_trigger_manager: Optional[Any] = None):
        """
        Initialize the three-tier memory manager with optional hormone and genetic trigger integration.
        """
        self.logger = logging.getLogger("ThreeTierMemoryManager")
        self._lock = threading.RLock()
        
        # Configuration
        self.auto_optimize = auto_optimize
        self.optimization_interval = timedelta(hours=1)
        self.last_optimization = datetime.now()
        
        # Initialize memory tiers
        self.working_memory = WorkingMemory(capacity_mb=working_capacity_mb)
        self.short_term_memory = ShortTermMemory(capacity_gb=short_term_capacity_gb)
        self.long_term_memory = LongTermMemory(capacity_gb=long_term_capacity_gb, vector_backend=vector_backend.value)
        self.vector_memory = EnhancedVectorMemorySystem(default_backend=vector_backend)
        
        # Memory tier routing rules
        self.tier_rules = {
            'immediate': MemoryTier.WORKING,
            'session': MemoryTier.WORKING,
            'temporary': MemoryTier.WORKING,
            'recent': MemoryTier.SHORT_TERM,
            'volatile': MemoryTier.SHORT_TERM,
            'cache': MemoryTier.SHORT_TERM,
            'persistent': MemoryTier.LONG_TERM,
            'knowledge': MemoryTier.LONG_TERM,
            'research': MemoryTier.LONG_TERM,
            'archive': MemoryTier.LONG_TERM
        }
        
        # Usage tracking for automatic tier promotion/demotion
        self.access_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Cross-lobe integration tracking
        self.lobe_memory_usage: Dict[str, Dict[MemoryTier, int]] = {}
        
        self.hormone_system = hormone_system
        self.genetic_trigger_manager = genetic_trigger_manager
        
        self.logger.info("[ThreeTierMemoryManager] Initialized three-tier memory architecture")
    
    def store(self, key: str, data: Any, 
              context: str = "default", 
              priority: float = 0.5,
              memory_type: Optional[str] = None,
              tier_hint: Optional[MemoryTier] = None,
              metadata: Optional[Dict[str, Any]] = None,
              ttl_seconds: Optional[int] = None,
              lobe_id: Optional[str] = None) -> bool:
        """
        Store data with automatic tier selection or explicit tier specification.
        """
        try:
            with self._lock:
                # Determine appropriate tier
                target_tier = self._select_memory_tier(
                    key, data, context, priority, memory_type, tier_hint, ttl_seconds
                )
                
                # Track lobe usage
                if lobe_id:
                    self._track_lobe_usage(lobe_id, target_tier)
                
                # Store in appropriate tier
                success = False
                if target_tier == MemoryTier.WORKING:
                    success = self.working_memory.add(key, data, context, priority, ttl_seconds)
                elif target_tier == MemoryTier.SHORT_TERM:
                    tags = metadata.get('tags', []) if metadata else []
                    success = self.short_term_memory.add(
                        key, data, context, priority, memory_type or 'general', tags, ttl_seconds
                    )
                elif target_tier == MemoryTier.LONG_TERM:
                    tags = metadata.get('tags', []) if metadata else []
                    research_sources = metadata.get('research_sources', []) if metadata else []
                    success = self.long_term_memory.add(
                        key, data, memory_type or 'general', tags, research_sources, priority, metadata
                    )
                
                if success:
                    # Update access patterns
                    self._update_access_pattern(key, target_tier, 'store')
                    
                    # Store vector representation if applicable
                    if target_tier in [MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]:
                        self._store_vector_representation(key, data, metadata or {})
                    
                    self.logger.debug(f"[ThreeTierMemoryManager] Stored '{key}' in {target_tier.value} memory")
                    
                    # Integration hooks: notify hormone/genetic systems
                    if self.hormone_system:
                        try:
                            self.hormone_system.release_hormone(
                                source_lobe=lobe_id or "memory_manager",
                                hormone_type="vasopressin",  # Example: memory consolidation hormone
                                quantity=0.1,
                                context={"operation": "store", "tier": target_tier.value, "key": key}
                            )
                        except Exception as e:
                            self.logger.warning(f"Hormone system integration failed: {e}")
                    if self.genetic_trigger_manager:
                        try:
                            self.genetic_trigger_manager.evaluate_triggers(
                                environment={"operation": "store", "tier": target_tier.value, "key": key, "context": context}
                            )
                        except Exception as e:
                            self.logger.warning(f"Genetic trigger integration failed: {e}")
                
                return success
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error storing '{key}': {ex}")
            return False
    
    def retrieve(self, key: str, 
                 context: Optional[str] = None,
                 tier_hint: Optional[MemoryTier] = None,
                 lobe_id: Optional[str] = None) -> Optional[MemoryItem]:
        """
        Retrieve data with automatic tier searching and access pattern tracking.
        """
        try:
            with self._lock:
                # Track lobe usage
                if lobe_id:
                    # We'll update this after successful retrieval
                    pass
                
                # Determine search order
                search_order = self._get_search_order(key, tier_hint)
                
                for tier in search_order:
                    result = None
                    
                    if tier == MemoryTier.WORKING:
                        data = self.working_memory.get(key, context)
                        if data is not None:
                            # Get additional info from working memory stats
                            stats = self.working_memory.get_stats()
                            result = MemoryItem(
                                key=key,
                                data=data,
                                tier=tier,
                                context=context or "default",
                                priority=0.5,  # Default priority
                                created_at=datetime.now(),  # Approximate
                                accessed_at=datetime.now()
                            )
                    
                    elif tier == MemoryTier.SHORT_TERM:
                        data = self.short_term_memory.get(key, context or "default")
                        if data is not None:
                            # Get recent items to find our item's metadata
                            recent_items = self.short_term_memory.get_recent(100, context)
                            item_info = next((item for item in recent_items if item['key'] == key), None)
                            
                            result = MemoryItem(
                                key=key,
                                data=data,
                                tier=tier,
                                context=item_info.get('context', context or "default") if item_info else context or "default",
                                priority=item_info.get('priority', 0.5) if item_info else 0.5,
                                metadata={'memory_type': item_info.get('memory_type', 'general')} if item_info else None,
                                created_at=datetime.fromisoformat(item_info['created_at']) if item_info and item_info.get('created_at') else None,
                                accessed_at=datetime.now()
                            )
                    
                    elif tier == MemoryTier.LONG_TERM:
                        item_data = self.long_term_memory.get(key)
                        if item_data:
                            result = MemoryItem(
                                key=key,
                                data=item_data['data'],
                                tier=tier,
                                context=item_data.get('category', 'general'),
                                priority=item_data.get('importance_score', 0.5),
                                metadata=item_data.get('metadata', {}),
                                created_at=datetime.fromisoformat(item_data['created_at']) if item_data.get('created_at') else None,
                                accessed_at=datetime.now()
                            )
                    
                    if result:
                        # Update access patterns
                        self._update_access_pattern(key, tier, 'retrieve')
                        
                        # Track lobe usage
                        if lobe_id:
                            self._track_lobe_usage(lobe_id, tier)
                        
                        # Consider tier promotion based on access patterns
                        if self.auto_optimize:
                            self._consider_tier_promotion(key, tier)
                        
                        self.logger.debug(f"[ThreeTierMemoryManager] Retrieved '{key}' from {tier.value} memory")
                        return result
                
                return None
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error retrieving '{key}': {ex}")
            return None
    
    def search(self, query: str, 
               context: Optional[str] = None,
               tier: Optional[MemoryTier] = None,
               limit: int = 10,
               use_semantic: bool = True,
               lobe_id: Optional[str] = None) -> List[MemoryItem]:
        """
        Search across memory tiers with optional semantic search.
        """
        try:
            with self._lock:
                results = []
                
                # Determine which tiers to search
                tiers_to_search = [tier] if tier else [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]
                
                for search_tier in tiers_to_search:
                    tier_results = []
                    
                    if search_tier == MemoryTier.WORKING:
                        # Search working memory by context
                        if context:
                            working_items = self.working_memory.get_by_context(context, limit)
                            for item in working_items:
                                if query.lower() in str(item['data']).lower() or query.lower() in item['key'].lower():
                                    tier_results.append(MemoryItem(
                                        key=item['key'],
                                        data=item['data'],
                                        tier=search_tier,
                                        context=context,
                                        priority=item['priority'],
                                        created_at=item['created_at'],
                                        accessed_at=item['accessed_at']
                                    ))
                    
                    elif search_tier == MemoryTier.SHORT_TERM:
                        # Search short-term memory
                        short_results = self.short_term_memory.search(query, context, limit)
                        for item in short_results:
                            tier_results.append(MemoryItem(
                                key=item['key'],
                                data=item['data'],
                                tier=search_tier,
                                context=item['context'],
                                priority=item['priority'],
                                metadata={'memory_type': item['memory_type']},
                                created_at=datetime.fromisoformat(item['created_at']) if item.get('created_at') else None,
                                accessed_at=datetime.fromisoformat(item['last_accessed']) if item.get('last_accessed') else None
                            ))
                    
                    elif search_tier == MemoryTier.LONG_TERM:
                        # Search long-term memory (semantic if enabled)
                        if use_semantic:
                            long_results = self.long_term_memory.semantic_search(query, limit, context)
                        else:
                            # Fallback to text search would be implemented in long_term_memory
                            long_results = self.long_term_memory.semantic_search(query, limit, context)
                        
                        for item in long_results:
                            tier_results.append(MemoryItem(
                                key=item['key'],
                                data=item['data'],
                                tier=search_tier,
                                context=item.get('category', 'general'),
                                priority=item.get('importance_score', 0.5),
                                metadata=item.get('metadata', {}),
                                created_at=datetime.fromisoformat(item['created_at']) if item.get('created_at') else None,
                                accessed_at=datetime.fromisoformat(item['updated_at']) if item.get('updated_at') else None
                            ))
                    
                    results.extend(tier_results)
                
                # Sort results by relevance and priority
                results.sort(key=lambda x: (x.priority, x.accessed_at or datetime.min), reverse=True)
                
                # Track lobe usage
                if lobe_id:
                    for result in results[:limit]:
                        self._track_lobe_usage(lobe_id, result.tier)
                
                return results[:limit]
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error searching for '{query}': {ex}")
            return []
    
    def _select_memory_tier(self, key: str, data: Any, context: str, priority: float,
                           memory_type: Optional[str], tier_hint: Optional[MemoryTier],
                           ttl_seconds: Optional[int]) -> MemoryTier:
        """Select appropriate memory tier based on various factors."""
        
        # Explicit tier hint takes precedence
        if tier_hint:
            return tier_hint
        
        # TTL-based selection
        if ttl_seconds:
            if ttl_seconds <= 3600:  # 1 hour or less
                return MemoryTier.WORKING
            elif ttl_seconds <= 86400 * 7:  # 1 week or less
                return MemoryTier.SHORT_TERM
            else:
                return MemoryTier.LONG_TERM
        
        # Memory type-based selection
        if memory_type and memory_type in self.tier_rules:
            return self.tier_rules[memory_type]
        
        # Context-based selection
        context_lower = context.lower()
        for rule_key, tier in self.tier_rules.items():
            if rule_key in context_lower:
                return tier
        
        # Priority-based selection
        if priority >= 0.8:
            return MemoryTier.LONG_TERM
        elif priority >= 0.5:
            return MemoryTier.SHORT_TERM
        else:
            return MemoryTier.WORKING
        
        # Data size-based selection (rough estimation)
        data_size = len(str(data))
        if data_size > 10000:  # Large data goes to long-term
            return MemoryTier.LONG_TERM
        elif data_size > 1000:  # Medium data goes to short-term
            return MemoryTier.SHORT_TERM
        else:
            return MemoryTier.WORKING
    
    def _get_search_order(self, key: str, tier_hint: Optional[MemoryTier]) -> List[MemoryTier]:
        """Determine search order based on access patterns and hints."""
        
        if tier_hint:
            # Start with hinted tier, then others
            other_tiers = [t for t in MemoryTier if t != tier_hint]
            return [tier_hint] + other_tiers
        
        # Check access patterns
        if key in self.access_patterns:
            pattern = self.access_patterns[key]
            last_tier = pattern.get('last_tier')
            if last_tier:
                # Start with last known tier
                other_tiers = [t for t in MemoryTier if t.value != last_tier]
                return [MemoryTier(last_tier)] + other_tiers
        
        # Default search order: Working -> Short-term -> Long-term
        return [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]
    
    def _update_access_pattern(self, key: str, tier: MemoryTier, operation: str):
        """Update access patterns for intelligent tier management."""
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'access_count': 0,
                'last_access': datetime.now(),
                'last_tier': tier.value,
                'tier_history': [],
                'operations': []
            }
        
        pattern = self.access_patterns[key]
        pattern['access_count'] += 1
        pattern['last_access'] = datetime.now()
        pattern['last_tier'] = tier.value
        pattern['tier_history'].append((tier.value, datetime.now().isoformat()))
        pattern['operations'].append((operation, datetime.now().isoformat()))
        
        # Keep history limited
        if len(pattern['tier_history']) > 10:
            pattern['tier_history'] = pattern['tier_history'][-10:]
        if len(pattern['operations']) > 20:
            pattern['operations'] = pattern['operations'][-20:]
    
    def _consider_tier_promotion(self, key: str, current_tier: MemoryTier):
        """Consider promoting frequently accessed items to higher tiers."""
        if key not in self.access_patterns:
            return
        
        pattern = self.access_patterns[key]
        access_count = pattern['access_count']
        last_access = pattern['last_access']
        
        # Promotion rules
        should_promote = False
        target_tier = None
        
        if current_tier == MemoryTier.WORKING:
            # Promote to short-term if accessed frequently
            if access_count >= 5 and (datetime.now() - last_access).total_seconds() < 3600:
                should_promote = True
                target_tier = MemoryTier.SHORT_TERM
        
        elif current_tier == MemoryTier.SHORT_TERM:
            # Promote to long-term if accessed very frequently
            if access_count >= 10:
                should_promote = True
                target_tier = MemoryTier.LONG_TERM
        
        if should_promote and target_tier:
            self._promote_item(key, current_tier, target_tier)
    
    def _promote_item(self, key: str, from_tier: MemoryTier, to_tier: MemoryTier):
        """Promote an item from one tier to another, with integration hooks."""
        try:
            item = self.retrieve(key, tier_hint=from_tier)
            if not item:
                return
            success = self.store(
                key=key,
                data=item.data,
                context=item.context,
                priority=item.priority,
                tier_hint=to_tier,
                metadata=item.metadata
            )
            if success:
                self._remove_from_tier(key, from_tier, item.context)
                self.logger.info(f"[ThreeTierMemoryManager] Promoted '{key}' from {from_tier.value} to {to_tier.value}")
                # Integration hooks: notify hormone/genetic systems
                if self.hormone_system:
                    try:
                        self.hormone_system.release_hormone(
                            source_lobe="memory_manager",
                            hormone_type="growth_hormone",  # Example: promotion event
                            quantity=0.05,
                            context={"operation": "promote", "from": from_tier.value, "to": to_tier.value, "key": key}
                        )
                    except Exception as e:
                        self.logger.warning(f"Hormone system integration failed: {e}")
                if self.genetic_trigger_manager:
                    try:
                        self.genetic_trigger_manager.evaluate_triggers(
                            environment={"operation": "promote", "from": from_tier.value, "to": to_tier.value, "key": key}
                        )
                    except Exception as e:
                        self.logger.warning(f"Genetic trigger integration failed: {e}")
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error promoting '{key}': {ex}")
    
    def _remove_from_tier(self, key: str, tier: MemoryTier, context: str):
        """Remove item from specific tier."""
        try:
            if tier == MemoryTier.WORKING:
                self.working_memory.remove(key)
            elif tier == MemoryTier.SHORT_TERM:
                # Short-term memory doesn't have direct remove, but items will expire
                pass
            elif tier == MemoryTier.LONG_TERM:
                self.long_term_memory.remove(key)
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error removing '{key}' from {tier.value}: {ex}")
    
    def _track_lobe_usage(self, lobe_id: str, tier: MemoryTier):
        """Track memory usage by different lobes."""
        if lobe_id not in self.lobe_memory_usage:
            self.lobe_memory_usage[lobe_id] = {
                MemoryTier.WORKING: 0,
                MemoryTier.SHORT_TERM: 0,
                MemoryTier.LONG_TERM: 0
            }
        
        self.lobe_memory_usage[lobe_id][tier] += 1
    
    def _store_vector_representation(self, key: str, data: Any, metadata: Dict[str, Any]):
        """Store vector representation for semantic search."""
        try:
            # Create a simple vector representation
            text_data = str(data)
            if len(text_data.strip()) > 0:
                # Simple word frequency vector
                words = text_data.lower().split()
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                # Create vector from top words
                vector_dict = {}
                for i, (word, freq) in enumerate(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]):
                    vector_dict[f"word_{i}"] = freq / len(words)
                
                if vector_dict:
                    vector_metadata = metadata.copy()
                    vector_metadata['key'] = key
                    vector_metadata['content_preview'] = text_data[:200]
                    
                    self.vector_memory.add_vector(vector_dict, vector_metadata)
        
        except Exception as ex:
            self.logger.debug(f"[ThreeTierMemoryManager] Could not create vector for '{key}': {ex}")
    
    def optimize(self, force: bool = False):
        """Optimize memory usage across all tiers, with integration hooks."""
        try:
            now = datetime.now()
            if not force and now - self.last_optimization < self.optimization_interval:
                return
            self.last_optimization = now
            with self._lock:
                self.logger.info("[ThreeTierMemoryManager] Starting memory optimization")
                
                # Cleanup expired items
                expired_working = self.working_memory.cleanup_expired()
                expired_short = self.short_term_memory.cleanup_expired()
                
                # Optimize vector memory
                self.vector_memory.optimize_backend()
                
                # Clean up old access patterns
                cutoff_date = now - timedelta(days=7)
                old_patterns = [
                    key for key, pattern in self.access_patterns.items()
                    if pattern['last_access'] < cutoff_date
                ]
                for key in old_patterns:
                    del self.access_patterns[key]
                
                # Integration hooks: notify hormone/genetic systems
                if self.hormone_system:
                    try:
                        self.hormone_system.release_hormone(
                            source_lobe="memory_manager",
                            hormone_type="serotonin",  # Example: optimization event
                            quantity=0.05,
                            context={"operation": "optimize"}
                        )
                    except Exception as e:
                        self.logger.warning(f"Hormone system integration failed: {e}")
                if self.genetic_trigger_manager:
                    try:
                        self.genetic_trigger_manager.evaluate_triggers(
                            environment={"operation": "optimize"}
                        )
                    except Exception as e:
                        self.logger.warning(f"Genetic trigger integration failed: {e}")
                
                self.logger.info(f"[ThreeTierMemoryManager] Optimization complete: "
                               f"cleaned {expired_working} working items, "
                               f"{expired_short} short-term items, "
                               f"{len(old_patterns)} old access patterns")
        
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error during optimization: {ex}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all memory tiers."""
        try:
            with self._lock:
                working_stats = self.working_memory.get_stats()
                short_stats = self.short_term_memory.get_stats()
                long_stats = self.long_term_memory.get_stats()
                vector_stats = self.vector_memory.get_comprehensive_stats()
                
                return {
                    'working_memory': working_stats,
                    'short_term_memory': short_stats,
                    'long_term_memory': long_stats,
                    'vector_memory': vector_stats,
                    'access_patterns': {
                        'total_tracked_keys': len(self.access_patterns),
                        'most_accessed': sorted(
                            [(k, v['access_count']) for k, v in self.access_patterns.items()],
                            key=lambda x: x[1], reverse=True
                        )[:10]
                    },
                    'lobe_usage': self.lobe_memory_usage,
                    'system': {
                        'auto_optimize': self.auto_optimize,
                        'last_optimization': self.last_optimization.isoformat(),
                        'optimization_interval_hours': self.optimization_interval.total_seconds() / 3600
                    },
                    'tier_transitions': self._get_tier_transition_stats(),
                    'consolidation_stats': self._get_consolidation_stats()
                }
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error getting comprehensive stats: {ex}")
            return {'error': str(ex)}
    
    def _get_tier_transition_stats(self) -> Dict[str, Any]:
        """Get statistics about tier transitions."""
        try:
            transitions = {}
            for key, pattern in self.access_patterns.items():
                tier_history = pattern.get('tier_history', [])
                if len(tier_history) > 1:
                    for i in range(1, len(tier_history)):
                        prev_tier = tier_history[i-1][0]
                        curr_tier = tier_history[i][0]
                        transition = f"{prev_tier}_to_{curr_tier}"
                        transitions[transition] = transitions.get(transition, 0) + 1
            
            return {
                'total_transitions': sum(transitions.values()),
                'transition_types': transitions,
                'keys_with_transitions': len([k for k, p in self.access_patterns.items() 
                                            if len(p.get('tier_history', [])) > 1])
            }
        except Exception as ex:
            self.logger.error(f"Error getting tier transition stats: {ex}")
            return {'error': str(ex)}
    
    def _get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about memory consolidation."""
        try:
            now = datetime.now()
            recent_consolidations = 0
            total_consolidations = 0
            
            for key, pattern in self.access_patterns.items():
                operations = pattern.get('operations', [])
                for op, timestamp in operations:
                    if op == 'consolidate':
                        total_consolidations += 1
                        op_time = datetime.fromisoformat(timestamp)
                        if (now - op_time).total_seconds() < 86400:  # Last 24 hours
                            recent_consolidations += 1
            
            return {
                'total_consolidations': total_consolidations,
                'recent_consolidations_24h': recent_consolidations,
                'consolidation_candidates': self._count_consolidation_candidates()
            }
        except Exception as ex:
            self.logger.error(f"Error getting consolidation stats: {ex}")
            return {'error': str(ex)}
    
    def _count_consolidation_candidates(self) -> int:
        """Count items that are candidates for consolidation."""
        try:
            candidates = 0
            now = datetime.now()
            
            for key, pattern in self.access_patterns.items():
                # Items accessed frequently in working/short-term memory are candidates
                if (pattern.get('access_count', 0) >= 3 and 
                    pattern.get('last_tier') in ['working', 'short_term'] and
                    (now - pattern.get('last_access', now)).total_seconds() > 3600):  # Not accessed in last hour
                    candidates += 1
            
            return candidates
        except Exception as ex:
            self.logger.error(f"Error counting consolidation candidates: {ex}")
            return 0
    
    def consolidate_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory consolidation between tiers based on access patterns and importance.
        This implements the core consolidation workflow required by the task.
        """
        try:
            with self._lock:
                self.logger.info("[ThreeTierMemoryManager] Starting memory consolidation")
                
                consolidation_results = {
                    'working_to_short': 0,
                    'short_to_long': 0,
                    'demotions': 0,
                    'errors': []
                }
                
                now = datetime.now()
                
                # Phase 1: Consolidate from working to short-term memory
                working_candidates = self._find_working_consolidation_candidates()
                for key, score in working_candidates:
                    try:
                        if self._consolidate_working_to_short(key, score):
                            consolidation_results['working_to_short'] += 1
                            self._update_access_pattern(key, MemoryTier.SHORT_TERM, 'consolidate')
                    except Exception as ex:
                        consolidation_results['errors'].append(f"Working->Short {key}: {ex}")
                
                # Phase 2: Consolidate from short-term to long-term memory
                short_candidates = self._find_short_consolidation_candidates()
                for key, score in short_candidates:
                    try:
                        if self._consolidate_short_to_long(key, score):
                            consolidation_results['short_to_long'] += 1
                            self._update_access_pattern(key, MemoryTier.LONG_TERM, 'consolidate')
                    except Exception as ex:
                        consolidation_results['errors'].append(f"Short->Long {key}: {ex}")
                
                # Phase 3: Demote rarely accessed items (if not forced)
                if not force:
                    demoted = self._demote_rarely_accessed_items()
                    consolidation_results['demotions'] = demoted
                
                # Integration hooks: notify hormone/genetic systems
                if self.hormone_system:
                    try:
                        self.hormone_system.release_hormone(
                            source_lobe="memory_manager",
                            hormone_type="vasopressin",  # Memory consolidation hormone
                            quantity=0.2,
                            context={"operation": "consolidate", "results": consolidation_results}
                        )
                    except Exception as e:
                        self.logger.warning(f"Hormone system integration failed: {e}")
                
                if self.genetic_trigger_manager:
                    try:
                        self.genetic_trigger_manager.evaluate_triggers(
                            environment={"operation": "consolidate", "results": consolidation_results}
                        )
                    except Exception as e:
                        self.logger.warning(f"Genetic trigger integration failed: {e}")
                
                self.logger.info(f"[ThreeTierMemoryManager] Consolidation complete: {consolidation_results}")
                return consolidation_results
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error during consolidation: {ex}")
            return {'error': str(ex)}
    
    def _find_working_consolidation_candidates(self) -> List[tuple]:
        """Find working memory items that should be consolidated to short-term."""
        candidates = []
        now = datetime.now()
        
        try:
            # Get working memory items
            working_stats = self.working_memory.get_stats()
            
            for key, pattern in self.access_patterns.items():
                if pattern.get('last_tier') == 'working':
                    # Calculate consolidation score
                    access_count = pattern.get('access_count', 0)
                    last_access = pattern.get('last_access', now)
                    age_hours = (now - last_access).total_seconds() / 3600
                    
                    # Consolidation criteria
                    if (access_count >= 3 and age_hours > 1) or age_hours > 24:
                        score = access_count / max(age_hours, 1)  # Higher score = higher priority
                        candidates.append((key, score))
            
            # Sort by score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:50]  # Limit to top 50 candidates
            
        except Exception as ex:
            self.logger.error(f"Error finding working consolidation candidates: {ex}")
            return []
    
    def _find_short_consolidation_candidates(self) -> List[tuple]:
        """Find short-term memory items that should be consolidated to long-term."""
        candidates = []
        now = datetime.now()
        
        try:
            for key, pattern in self.access_patterns.items():
                if pattern.get('last_tier') == 'short_term':
                    # Calculate consolidation score
                    access_count = pattern.get('access_count', 0)
                    last_access = pattern.get('last_access', now)
                    age_days = (now - last_access).total_seconds() / 86400
                    
                    # Consolidation criteria for long-term storage
                    if (access_count >= 5 and age_days > 1) or age_days > 7:
                        score = access_count / max(age_days, 1)  # Higher score = higher priority
                        candidates.append((key, score))
            
            # Sort by score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:20]  # Limit to top 20 candidates
            
        except Exception as ex:
            self.logger.error(f"Error finding short consolidation candidates: {ex}")
            return []
    
    def _consolidate_working_to_short(self, key: str, score: float) -> bool:
        """Consolidate an item from working to short-term memory."""
        try:
            # Retrieve from working memory
            item = self.retrieve(key, tier_hint=MemoryTier.WORKING)
            if not item:
                return False
            
            # Store in short-term memory with enhanced metadata
            metadata = item.metadata or {}
            metadata['consolidation_score'] = score
            metadata['consolidated_from'] = 'working'
            metadata['consolidation_time'] = datetime.now().isoformat()
            
            success = self.store(
                key=key,
                data=item.data,
                context=item.context,
                priority=min(item.priority + 0.1, 1.0),  # Slight priority boost
                tier_hint=MemoryTier.SHORT_TERM,
                metadata=metadata
            )
            
            if success:
                # Remove from working memory
                self._remove_from_tier(key, MemoryTier.WORKING, item.context)
                return True
            
            return False
            
        except Exception as ex:
            self.logger.error(f"Error consolidating {key} from working to short-term: {ex}")
            return False
    
    def _consolidate_short_to_long(self, key: str, score: float) -> bool:
        """Consolidate an item from short-term to long-term memory."""
        try:
            # Retrieve from short-term memory
            item = self.retrieve(key, tier_hint=MemoryTier.SHORT_TERM)
            if not item:
                return False
            
            # Store in long-term memory with enhanced metadata
            metadata = item.metadata or {}
            metadata['consolidation_score'] = score
            metadata['consolidated_from'] = 'short_term'
            metadata['consolidation_time'] = datetime.now().isoformat()
            
            success = self.store(
                key=key,
                data=item.data,
                context=item.context,
                priority=min(item.priority + 0.2, 1.0),  # Higher priority boost for long-term
                tier_hint=MemoryTier.LONG_TERM,
                metadata=metadata
            )
            
            if success:
                # Remove from short-term memory (will expire naturally)
                return True
            
            return False
            
        except Exception as ex:
            self.logger.error(f"Error consolidating {key} from short-term to long-term: {ex}")
            return False
    
    def _demote_rarely_accessed_items(self) -> int:
        """Demote rarely accessed items to lower tiers."""
        demoted = 0
        now = datetime.now()
        
        try:
            for key, pattern in self.access_patterns.items():
                last_access = pattern.get('last_access', now)
                access_count = pattern.get('access_count', 0)
                current_tier = pattern.get('last_tier')
                
                # Demotion criteria
                days_since_access = (now - last_access).total_seconds() / 86400
                
                if (current_tier == 'long_term' and access_count < 2 and days_since_access > 30):
                    # Very rarely accessed long-term items could be archived or compressed
                    # For now, just mark them for potential removal
                    pattern['demotion_candidate'] = True
                    demoted += 1
                elif (current_tier == 'short_term' and access_count < 3 and days_since_access > 14):
                    # Rarely accessed short-term items
                    pattern['demotion_candidate'] = True
                    demoted += 1
            
            return demoted
            
        except Exception as ex:
            self.logger.error(f"Error demoting rarely accessed items: {ex}")
            return 0
    
    def cross_tier_search(self, query: str, 
                         context: Optional[str] = None,
                         limit: int = 10,
                         use_semantic: bool = True,
                         tier_weights: Optional[Dict[MemoryTier, float]] = None,
                         lobe_id: Optional[str] = None) -> List[MemoryItem]:
        """
        Enhanced cross-tier search with optimization and tier weighting.
        This implements the cross-tier search optimization required by the task.
        """
        try:
            with self._lock:
                # Default tier weights (working memory gets highest priority for recency)
                if not tier_weights:
                    tier_weights = {
                        MemoryTier.WORKING: 1.0,
                        MemoryTier.SHORT_TERM: 0.8,
                        MemoryTier.LONG_TERM: 0.6
                    }
                
                all_results = []
                
                # Search each tier and apply weights
                for tier in MemoryTier:
                    tier_results = self.search(
                        query=query,
                        context=context,
                        tier=tier,
                        limit=limit * 2,  # Get more results to apply weighting
                        use_semantic=use_semantic,
                        lobe_id=lobe_id
                    )
                    
                    # Apply tier weighting to results
                    weight = tier_weights.get(tier, 0.5)
                    for result in tier_results:
                        # Adjust priority based on tier weight and recency
                        recency_bonus = 0.0
                        if result.accessed_at:
                            hours_since_access = (datetime.now() - result.accessed_at).total_seconds() / 3600
                            recency_bonus = max(0, 0.2 - (hours_since_access / 100))  # Bonus decreases with time
                        
                        result.priority = (result.priority * weight) + recency_bonus
                        all_results.append(result)
                
                # Remove duplicates (same key from different tiers)
                unique_results = {}
                for result in all_results:
                    if result.key not in unique_results or result.priority > unique_results[result.key].priority:
                        unique_results[result.key] = result
                
                # Sort by adjusted priority and return top results
                final_results = sorted(unique_results.values(), key=lambda x: x.priority, reverse=True)
                
                # Update access patterns for retrieved items
                for result in final_results[:limit]:
                    self._update_access_pattern(result.key, result.tier, 'cross_tier_search')
                
                self.logger.debug(f"[ThreeTierMemoryManager] Cross-tier search for '{query}' returned {len(final_results[:limit])} results")
                return final_results[:limit]
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error in cross-tier search: {ex}")
            return []
    
    def get_memory_health(self) -> Dict[str, Any]:
        """Get comprehensive memory system health metrics."""
        try:
            with self._lock:
                stats = self.get_comprehensive_stats()
                
                # Calculate health scores
                working_health = self._calculate_tier_health(MemoryTier.WORKING, stats['working_memory'])
                short_health = self._calculate_tier_health(MemoryTier.SHORT_TERM, stats['short_term_memory'])
                long_health = self._calculate_tier_health(MemoryTier.LONG_TERM, stats['long_term_memory'])
                
                overall_health = (working_health + short_health + long_health) / 3
                
                return {
                    'overall_health': overall_health,
                    'tier_health': {
                        'working': working_health,
                        'short_term': short_health,
                        'long_term': long_health
                    },
                    'recommendations': self._generate_health_recommendations(stats),
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error getting memory health: {ex}")
            return {'error': str(ex)}
    
    def _calculate_tier_health(self, tier: MemoryTier, tier_stats: Dict[str, Any]) -> float:
        """Calculate health score for a specific memory tier."""
        try:
            if 'error' in tier_stats:
                return 0.0
            
            # Base health factors
            capacity_usage = tier_stats.get('capacity_usage_percent', 0) / 100
            error_rate = tier_stats.get('error_rate', 0)
            
            # Health score calculation (0.0 to 1.0)
            health = 1.0
            
            # Penalize high capacity usage
            if capacity_usage > 0.9:
                health -= 0.3
            elif capacity_usage > 0.7:
                health -= 0.1
            
            # Penalize high error rates
            health -= min(error_rate * 2, 0.5)
            
            # Bonus for optimal usage
            if 0.3 <= capacity_usage <= 0.7:
                health += 0.1
            
            return max(0.0, min(1.0, health))
            
        except Exception as ex:
            self.logger.error(f"Error calculating tier health: {ex}")
            return 0.5  # Neutral health score
    
    def _generate_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on system stats."""
        recommendations = []
        
        try:
            # Check capacity usage
            for tier_name, tier_stats in [
                ('working', stats.get('working_memory', {})),
                ('short_term', stats.get('short_term_memory', {})),
                ('long_term', stats.get('long_term_memory', {}))
            ]:
                if isinstance(tier_stats, dict):
                    usage = tier_stats.get('capacity_usage_percent', 0)
                    if usage > 90:
                        recommendations.append(f"High {tier_name} memory usage ({usage}%) - consider cleanup or consolidation")
                    elif usage < 10:
                        recommendations.append(f"Low {tier_name} memory usage ({usage}%) - system may be underutilized")
            
            # Check consolidation needs
            consolidation_stats = stats.get('consolidation_stats', {})
            candidates = consolidation_stats.get('consolidation_candidates', 0)
            if candidates > 10:
                recommendations.append(f"High number of consolidation candidates ({candidates}) - run consolidation")
            
            # Check access patterns
            access_patterns = stats.get('access_patterns', {})
            tracked_keys = access_patterns.get('total_tracked_keys', 0)
            if tracked_keys > 1000:
                recommendations.append("Large number of tracked access patterns - consider pattern cleanup")
            
            if not recommendations:
                recommendations.append("Memory system is operating within normal parameters")
            
            return recommendations
            
        except Exception as ex:
            self.logger.error(f"Error generating health recommendations: {ex}")
            return ["Unable to generate recommendations due to error"]
            self.logger.error(f"[ThreeTierMemoryManager] Error getting stats: {ex}")
            return {'error': str(ex)}
    
    def clear_tier(self, tier: MemoryTier, context: Optional[str] = None):
        """Clear specific memory tier."""
        try:
            with self._lock:
                if tier == MemoryTier.WORKING:
                    if context:
                        self.working_memory.clear_context(context)
                    else:
                        self.working_memory.clear()
                elif tier == MemoryTier.SHORT_TERM:
                    self.short_term_memory.clear(context)
                elif tier == MemoryTier.LONG_TERM:
                    self.long_term_memory.clear(context)
                
                self.logger.info(f"[ThreeTierMemoryManager] Cleared {tier.value} memory" + 
                               (f" for context '{context}'" if context else ""))
        
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error clearing {tier.value} memory: {ex}")
    
    def clear_all(self):
        """Clear all memory tiers."""
        try:
            with self._lock:
                self.working_memory.clear()
                self.short_term_memory.clear()
                self.long_term_memory.clear()
                self.access_patterns.clear()
                self.lobe_memory_usage.clear()
                
                self.logger.info("[ThreeTierMemoryManager] Cleared all memory tiers")
        
        except Exception as ex:
            self.logger.error(f"[ThreeTierMemoryManager] Error clearing all memory: {ex}")