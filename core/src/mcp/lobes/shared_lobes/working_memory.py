import logging
import time
import threading
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Callable, Union
from datetime import datetime, timedelta

class WorkingMemory:
    """
    Rolling and adaptive working memory that focuses on differences and changes.
    Features:
    - Rolling window of recent changes and context diffs
    - Adaptive storage based on change patterns and frequency
    - Diff-based storage to track what has changed rather than static data
    - Context-sensitive rolling updates for immediate operations
    - High-frequency read/write optimization with change tracking
    - Thread-safe operations with rolling eviction
    """
    
    def __init__(self, capacity_mb: float = 100.0, rolling_window_size: int = 1000, 
                 diff_threshold: float = 0.1, fallback: Optional[Callable] = None):
        self.capacity_bytes = int(capacity_mb * 1024 * 1024)  # Convert MB to bytes
        self.rolling_window_size = rolling_window_size
        self.diff_threshold = diff_threshold  # Minimum change to trigger storage
        
        # Rolling memory structures
        self.memory: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.change_history: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()  # Track changes over time
        self.context_diffs: Dict[str, Dict[str, Any]] = {}  # Context -> recent changes
        self.adaptive_weights: Dict[str, float] = {}  # Key -> adaptive importance weight
        
        # Rolling window tracking
        self.rolling_changes: List[Dict[str, Any]] = []  # Recent changes in order
        self.context_patterns: Dict[str, Dict[str, Any]] = {}  # Context usage patterns
        
        # Traditional indices
        self.context_index: Dict[str, List[str]] = {}  # Context -> list of keys
        self.access_count: Dict[str, int] = {}
        self.change_frequency: Dict[str, int] = {}  # How often items change
        
        self.current_size = 0
        self.session_id = str(int(time.time()))
        self.logger = logging.getLogger("RollingWorkingMemory")
        self.fallback = fallback
        self._lock = threading.RLock()
        
        self.logger.info(f"[RollingWorkingMemory] Initialized with {capacity_mb}MB capacity, "
                        f"rolling window: {rolling_window_size}, diff threshold: {diff_threshold}")
    
    def add(self, key: str, item: Any, context: str = "default", priority: float = 0.5, ttl_seconds: Optional[int] = None) -> bool:
        """Add item with rolling diff-based storage and adaptive management."""
        try:
            with self._lock:
                # Create context-sensitive key
                context_key = f"{context}::{key}"
                
                # Check if this is an update and calculate diff
                is_update = context_key in self.memory
                old_item = self.memory.get(context_key) if is_update else None
                diff_data = self._calculate_diff(old_item['data'] if old_item else None, item)
                
                # Only store if diff is significant enough
                if is_update and not self._is_significant_change(diff_data):
                    # Update access patterns but don't store redundant data
                    self._update_access_patterns(context_key, context)
                    return True
                
                # Calculate item size (rough estimation)
                item_size = self._estimate_size(item)
                
                # Check if item would exceed capacity
                if item_size > self.capacity_bytes:
                    self.logger.warning(f"[RollingWorkingMemory] Item too large: {item_size} bytes > {self.capacity_bytes} bytes")
                    if self.fallback:
                        return self.fallback(key, item, context)
                    return False
                
                # Rolling eviction based on adaptive weights
                while self.current_size + item_size > self.capacity_bytes and self.memory:
                    self._evict_adaptive()
                
                # Calculate expiry time
                expires_at = None
                if ttl_seconds:
                    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                
                # Store the item with diff information
                memory_item = {
                    'data': item,
                    'context': context,
                    'key': key,
                    'priority': priority,
                    'created_at': datetime.now(),
                    'accessed_at': datetime.now(),
                    'expires_at': expires_at,
                    'size': item_size,
                    'session_id': self.session_id,
                    'diff_data': diff_data,  # Store what changed
                    'change_magnitude': self._calculate_change_magnitude(diff_data),
                    'adaptive_weight': self._calculate_adaptive_weight(context_key, context, priority)
                }
                
                # Track change in rolling history
                self._track_rolling_change(context_key, old_item, memory_item)
                
                # Remove existing item if updating
                if is_update:
                    self._remove_item_by_context_key(context_key)
                
                # Add new item
                self.memory[context_key] = memory_item
                self.current_size += item_size
                self.access_count[context_key] = self.access_count.get(context_key, 0) + 1
                self.change_frequency[context_key] = self.change_frequency.get(context_key, 0) + 1
                
                # Update adaptive weights based on usage patterns
                self._update_adaptive_weights(context_key, context)
                
                # Update context index and patterns
                if context not in self.context_index:
                    self.context_index[context] = []
                if key not in self.context_index[context]:
                    self.context_index[context].append(key)
                
                # Update context patterns for adaptive behavior
                self._update_context_patterns(context, key, memory_item)
                
                self.logger.debug(f"[RollingWorkingMemory] Added item '{key}' in context '{context}' "
                                f"({item_size} bytes, change_magnitude: {memory_item['change_magnitude']:.3f})")
                return True
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error adding item '{key}': {ex}")
            if self.fallback:
                return self.fallback(key, item, context)
            return False
    
    def get(self, key: str, context: Optional[str] = None) -> Optional[Any]:
        """Get item with LRU update and context filtering."""
        try:
            with self._lock:
                # Use default context if none specified
                if context is None:
                    context = "default"
                
                # Create context-sensitive key
                context_key = f"{context}::{key}"
                
                if context_key not in self.memory:
                    return None
                
                item = self.memory[context_key]
                
                # Check if expired
                if item['expires_at'] and datetime.now() > item['expires_at']:
                    self._remove_item_by_context_key(context_key)
                    return None
                
                # Update LRU order and access tracking
                self.memory.move_to_end(context_key)
                item['accessed_at'] = datetime.now()
                self.access_count[context_key] = self.access_count.get(context_key, 0) + 1
                
                return item['data']
                
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error getting item '{key}': {ex}")
            return None
    
    def get_by_context(self, context: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all items for a specific context."""
        try:
            with self._lock:
                results = []
                keys = self.context_index.get(context, [])
                
                for key in keys:
                    context_key = f"{context}::{key}"
                    if context_key in self.memory:
                        item = self.memory[context_key]
                        # Check if expired
                        if item['expires_at'] and datetime.now() > item['expires_at']:
                            self._remove_item_by_context_key(context_key)
                            continue
                        
                        results.append({
                            'key': key,  # Return original key, not context_key
                            'data': item['data'],
                            'priority': item['priority'],
                            'created_at': item['created_at'],
                            'accessed_at': item['accessed_at']
                        })
                
                # Sort by priority and access time
                results.sort(key=lambda x: (x['priority'], x['accessed_at']), reverse=True)
                
                if limit:
                    results = results[:limit]
                
                return results
                
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error getting context '{context}': {ex}")
            return []
    
    def update_priority(self, key: str, priority: float, context: str = "default") -> bool:
        """Update item priority for better LRU management."""
        try:
            with self._lock:
                context_key = f"{context}::{key}"
                if context_key in self.memory:
                    self.memory[context_key]['priority'] = priority
                    return True
                return False
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error updating priority for '{key}': {ex}")
            return False
    
    def remove(self, key: str, context: str = "default") -> bool:
        """Remove specific item."""
        try:
            with self._lock:
                return self._remove_item(key, context)
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error removing item '{key}': {ex}")
            return False
    
    def clear_context(self, context: str) -> int:
        """Clear all items in a specific context."""
        try:
            with self._lock:
                keys = self.context_index.get(context, []).copy()
                removed_count = 0
                
                for key in keys:
                    if self._remove_item(key, context):
                        removed_count += 1
                
                if context in self.context_index:
                    del self.context_index[context]
                
                self.logger.info(f"[WorkingMemory] Cleared {removed_count} items from context '{context}'")
                return removed_count
                
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error clearing context '{context}': {ex}")
            return 0
    
    def clear(self) -> None:
        """Clear all working memory (session reset)."""
        try:
            with self._lock:
                self.memory.clear()
                self.context_index.clear()
                self.access_count.clear()
                self.current_size = 0
                
                # Clear rolling and adaptive structures
                self.change_history.clear()
                self.context_diffs.clear()
                self.adaptive_weights.clear()
                self.rolling_changes.clear()
                self.context_patterns.clear()
                self.change_frequency.clear()
                
                # Generate new session ID with microsecond precision to ensure uniqueness
                import time
                self.session_id = f"{int(time.time())}_{int(time.time() * 1000000) % 1000000}"
                self.logger.info("[RollingWorkingMemory] Cleared all working memory and reset session")
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error clearing memory: {ex}")
    
    def cleanup_expired(self) -> int:
        """Remove expired items."""
        try:
            with self._lock:
                expired_keys = []
                now = datetime.now()
                
                for context_key, item in self.memory.items():
                    if item['expires_at'] and now > item['expires_at']:
                        expired_keys.append(context_key)
                
                for context_key in expired_keys:
                    self._remove_item_by_context_key(context_key)
                
                if expired_keys:
                    self.logger.info(f"[WorkingMemory] Cleaned up {len(expired_keys)} expired items")
                
                return len(expired_keys)
                
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error during cleanup: {ex}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            with self._lock:
                total_items = len(self.memory)
                contexts = len(self.context_index)
                avg_access = sum(self.access_count.values()) / len(self.access_count) if self.access_count else 0
                
                return {
                    'total_items': total_items,
                    'total_contexts': contexts,
                    'current_size_bytes': self.current_size,
                    'current_size_mb': round(self.current_size / (1024 * 1024), 2),
                    'capacity_mb': round(self.capacity_bytes / (1024 * 1024), 2),
                    'utilization_percent': round((self.current_size / self.capacity_bytes) * 100, 2),
                    'average_access_count': round(avg_access, 2),
                    'session_id': self.session_id
                }
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error getting stats: {ex}")
            return {'error': str(ex)}
    
    def _remove_item_by_context_key(self, context_key: str) -> bool:
        """Internal method to remove an item by context key and update indices."""
        if context_key not in self.memory:
            return False
        
        item = self.memory[context_key]
        context = item['context']
        original_key = item['key']
        
        # Update size
        self.current_size -= item['size']
        
        # Remove from memory
        del self.memory[context_key]
        
        # Remove from access count
        if context_key in self.access_count:
            del self.access_count[context_key]
        
        # Update context index
        if context in self.context_index and original_key in self.context_index[context]:
            self.context_index[context].remove(original_key)
            if not self.context_index[context]:
                del self.context_index[context]
        
        return True
    
    def _remove_item(self, key: str, context: str = "default") -> bool:
        """Internal method to remove an item and update indices."""
        context_key = f"{context}::{key}"
        return self._remove_item_by_context_key(context_key)
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.memory:
            return False
        
        # Get LRU item (first in OrderedDict)
        lru_context_key = next(iter(self.memory))
        self.logger.debug(f"[WorkingMemory] Evicting LRU item: {lru_context_key}")
        return self._remove_item_by_context_key(lru_context_key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Rough estimation of object size in bytes."""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, bool):
                return 1
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj) + 64
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items()) + 64
            else:
                # Fallback estimation
                return len(str(obj)) * 2 + 64
        except Exception:
            return 1024  # Default 1KB for unknown objects
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items (for compatibility)."""
        try:
            with self._lock:
                results = []
                for key, item in self.memory.items():
                    results.append({
                        'key': key,
                        'data': item['data'],
                        'context': item['context'],
                        'priority': item['priority'],
                        'created_at': item['created_at'],
                        'accessed_at': item['accessed_at']
                    })
                return results
        except Exception as ex:
            self.logger.error(f"[WorkingMemory] Error getting all items: {ex}")
            return []
    
    def fallback_hook(self, *args, **kwargs):
        """Fallback hook for error handling."""
        if self.fallback:
            return self.fallback(*args, **kwargs)
        self.logger.warning("[WorkingMemory] No fallback defined.")
    
    # Rolling and Adaptive Methods
    
    def _calculate_diff(self, old_data: Any, new_data: Any) -> Dict[str, Any]:
        """Calculate difference between old and new data."""
        try:
            if old_data is None:
                return {'type': 'new', 'magnitude': 1.0, 'changes': {'added': new_data}}
            
            if isinstance(old_data, dict) and isinstance(new_data, dict):
                changes = {}
                added = {k: v for k, v in new_data.items() if k not in old_data}
                removed = {k: v for k, v in old_data.items() if k not in new_data}
                modified = {k: {'old': old_data[k], 'new': new_data[k]} 
                           for k in old_data.keys() & new_data.keys() 
                           if old_data[k] != new_data[k]}
                
                if added: changes['added'] = added
                if removed: changes['removed'] = removed
                if modified: changes['modified'] = modified
                
                magnitude = (len(added) + len(removed) + len(modified)) / max(len(old_data), len(new_data), 1)
                return {'type': 'dict_diff', 'magnitude': magnitude, 'changes': changes}
            
            elif isinstance(old_data, (list, tuple)) and isinstance(new_data, (list, tuple)):
                old_set = set(str(x) for x in old_data)
                new_set = set(str(x) for x in new_data)
                added = new_set - old_set
                removed = old_set - new_set
                
                magnitude = (len(added) + len(removed)) / max(len(old_data), len(new_data), 1)
                return {'type': 'list_diff', 'magnitude': magnitude, 
                       'changes': {'added': list(added), 'removed': list(removed)}}
            
            elif isinstance(old_data, str) and isinstance(new_data, str):
                # Simple string diff
                if old_data == new_data:
                    magnitude = 0.0
                else:
                    # Calculate edit distance ratio
                    import difflib
                    ratio = difflib.SequenceMatcher(None, old_data, new_data).ratio()
                    magnitude = 1.0 - ratio
                
                return {'type': 'string_diff', 'magnitude': magnitude, 
                       'changes': {'old': old_data, 'new': new_data}}
            
            else:
                # Generic comparison
                magnitude = 0.0 if old_data == new_data else 1.0
                return {'type': 'value_diff', 'magnitude': magnitude, 
                       'changes': {'old': old_data, 'new': new_data}}
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error calculating diff: {ex}")
            return {'type': 'error', 'magnitude': 1.0, 'changes': {'error': str(ex)}}
    
    def _is_significant_change(self, diff_data: Dict[str, Any]) -> bool:
        """Determine if a change is significant enough to store."""
        return diff_data.get('magnitude', 0.0) >= self.diff_threshold
    
    def _calculate_change_magnitude(self, diff_data: Dict[str, Any]) -> float:
        """Calculate the magnitude of change for adaptive weighting."""
        return diff_data.get('magnitude', 0.0)
    
    def _calculate_adaptive_weight(self, context_key: str, context: str, priority: float) -> float:
        """Calculate adaptive weight based on usage patterns and context."""
        try:
            # Base weight from priority
            weight = priority
            
            # Adjust based on access frequency
            access_freq = self.access_count.get(context_key, 0)
            change_freq = self.change_frequency.get(context_key, 0)
            
            # Higher access frequency increases weight
            if access_freq > 0:
                weight += min(0.3, access_freq / 100.0)
            
            # Higher change frequency increases weight (indicates dynamic data)
            if change_freq > 0:
                weight += min(0.2, change_freq / 50.0)
            
            # Context-based adjustments
            context_pattern = self.context_patterns.get(context, {})
            if context_pattern.get('high_activity', False):
                weight += 0.1
            
            return min(1.0, weight)  # Cap at 1.0
            
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error calculating adaptive weight: {ex}")
            return priority
    
    def _track_rolling_change(self, context_key: str, old_item: Optional[Dict], new_item: Dict):
        """Track changes in rolling history."""
        try:
            change_record = {
                'timestamp': datetime.now(),
                'context_key': context_key,
                'context': new_item['context'],
                'key': new_item['key'],
                'change_type': 'update' if old_item else 'create',
                'magnitude': new_item.get('change_magnitude', 0.0),
                'diff_data': new_item.get('diff_data', {})
            }
            
            # Add to rolling changes
            self.rolling_changes.append(change_record)
            
            # Maintain rolling window size
            if len(self.rolling_changes) > self.rolling_window_size:
                self.rolling_changes.pop(0)
            
            # Update change history for this key
            if context_key not in self.change_history:
                self.change_history[context_key] = []
            
            self.change_history[context_key].append(change_record)
            
            # Maintain change history size per key
            if len(self.change_history[context_key]) > 50:  # Keep last 50 changes per key
                self.change_history[context_key].pop(0)
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error tracking rolling change: {ex}")
    
    def _update_access_patterns(self, context_key: str, context: str):
        """Update access patterns without storing redundant data."""
        try:
            if context_key in self.memory:
                item = self.memory[context_key]
                item['accessed_at'] = datetime.now()
                self.access_count[context_key] = self.access_count.get(context_key, 0) + 1
                
                # Move to end for LRU
                self.memory.move_to_end(context_key)
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error updating access patterns: {ex}")
    
    def _update_adaptive_weights(self, context_key: str, context: str):
        """Update adaptive weights based on usage patterns."""
        try:
            if context_key in self.memory:
                item = self.memory[context_key]
                new_weight = self._calculate_adaptive_weight(context_key, context, item['priority'])
                item['adaptive_weight'] = new_weight
                self.adaptive_weights[context_key] = new_weight
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error updating adaptive weights: {ex}")
    
    def _update_context_patterns(self, context: str, key: str, memory_item: Dict):
        """Update context usage patterns for adaptive behavior."""
        try:
            if context not in self.context_patterns:
                self.context_patterns[context] = {
                    'total_items': 0,
                    'total_changes': 0,
                    'avg_change_magnitude': 0.0,
                    'high_activity': False,
                    'last_activity': datetime.now()
                }
            
            pattern = self.context_patterns[context]
            pattern['total_items'] = len(self.context_index.get(context, []))
            pattern['total_changes'] += 1
            pattern['last_activity'] = datetime.now()
            
            # Update average change magnitude
            current_magnitude = memory_item.get('change_magnitude', 0.0)
            pattern['avg_change_magnitude'] = (
                (pattern['avg_change_magnitude'] * (pattern['total_changes'] - 1) + current_magnitude) 
                / pattern['total_changes']
            )
            
            # Determine if this is a high-activity context
            pattern['high_activity'] = (
                pattern['total_changes'] > 10 and 
                pattern['avg_change_magnitude'] > 0.3
            )
            
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error updating context patterns: {ex}")
    
    def _evict_adaptive(self) -> bool:
        """Evict items based on adaptive weights and rolling patterns."""
        try:
            if not self.memory:
                return False
            
            # Find item with lowest adaptive weight and least recent access
            min_score = float('inf')
            evict_key = None
            
            for context_key, item in self.memory.items():
                # Calculate eviction score (lower = more likely to evict)
                adaptive_weight = item.get('adaptive_weight', 0.5)
                access_count = self.access_count.get(context_key, 0)
                change_freq = self.change_frequency.get(context_key, 0)
                
                # Time since last access (in hours)
                time_since_access = (datetime.now() - item['accessed_at']).total_seconds() / 3600
                
                # Lower score = higher eviction priority
                score = (adaptive_weight * 0.4 + 
                        min(access_count / 100.0, 0.3) * 0.3 + 
                        min(change_freq / 50.0, 0.2) * 0.2 - 
                        min(time_since_access / 24.0, 0.1) * 0.1)
                
                if score < min_score:
                    min_score = score
                    evict_key = context_key
            
            if evict_key:
                self.logger.debug(f"[RollingWorkingMemory] Adaptive eviction of '{evict_key}' (score: {min_score:.3f})")
                return self._remove_item_by_context_key(evict_key)
            
            return False
            
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error in adaptive eviction: {ex}")
            return self._evict_lru()  # Fallback to LRU
    
    def get_rolling_changes(self, limit: Optional[int] = None, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent rolling changes, optionally filtered by context."""
        try:
            with self._lock:
                changes = self.rolling_changes
                
                if context:
                    changes = [c for c in changes if c['context'] == context]
                
                if limit:
                    changes = changes[-limit:]
                
                return changes
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error getting rolling changes: {ex}")
            return []
    
    def get_context_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get context usage patterns for analysis."""
        try:
            with self._lock:
                return dict(self.context_patterns)
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error getting context patterns: {ex}")
            return {}
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive memory statistics."""
        try:
            with self._lock:
                total_changes = len(self.rolling_changes)
                avg_change_magnitude = (
                    sum(c.get('magnitude', 0.0) for c in self.rolling_changes) / total_changes
                    if total_changes > 0 else 0.0
                )
                
                high_activity_contexts = [
                    ctx for ctx, pattern in self.context_patterns.items() 
                    if pattern.get('high_activity', False)
                ]
                
                return {
                    'total_rolling_changes': total_changes,
                    'avg_change_magnitude': round(avg_change_magnitude, 3),
                    'rolling_window_size': self.rolling_window_size,
                    'diff_threshold': self.diff_threshold,
                    'high_activity_contexts': high_activity_contexts,
                    'total_context_patterns': len(self.context_patterns),
                    'adaptive_weights_count': len(self.adaptive_weights)
                }
                
        except Exception as ex:
            self.logger.error(f"[RollingWorkingMemory] Error getting adaptive stats: {ex}")
            return {'error': str(ex)}

class ShortTermMemory:
    """
    Enhanced short-term memory for recent, high-priority, or volatile information.
    Features:
    - SQLite backend with time-based indexing and optimized queries
    - Automatic cleanup after 30 days with priority-based neural network assisted retention
    - Medium-frequency access patterns with < 1GB capacity management
    - Priority-based retention system with intelligent scoring
    - Thread-safe operations with connection pooling
    - Relevance scoring and access pattern analysis
    - Automatic background cleanup and optimization
    """
    
    def __init__(self, db_path: Optional[str] = None, capacity_gb: float = 1.0, 
                 retention_days: int = 30, fallback: Optional[Callable] = None,
                 enable_neural_retention: bool = True):
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self.retention_days = retention_days
        self.fallback = fallback
        self.enable_neural_retention = enable_neural_retention
        self.logger = logging.getLogger("ShortTermMemory")
        self._lock = threading.RLock()
        
        # Neural retention scoring weights
        self.retention_weights = {
            'priority': 0.3,
            'access_frequency': 0.25,
            'recency': 0.2,
            'relevance': 0.15,
            'size_efficiency': 0.1
        }
        
        # Setup database path
        if db_path:
            self.db_path = db_path
        else:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, 'short_term_memory.db')
        
        self._init_database()
        self._migrate_database()  # Handle existing databases
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(hours=1)  # Cleanup every hour
        
        self.logger.info(f"[ShortTermMemory] Initialized with {capacity_gb}GB capacity, {retention_days} day retention, neural retention: {enable_neural_retention}")
    
    def _init_database(self):
        """Initialize SQLite database with time-based indexing and neural retention features."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main table with enhanced schema for neural retention
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS short_term_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    data TEXT NOT NULL,
                    data_type TEXT DEFAULT 'json',
                    context TEXT DEFAULT 'default',
                    priority REAL DEFAULT 0.5,
                    memory_type TEXT DEFAULT 'general',
                    tags TEXT DEFAULT '[]',
                    size_bytes INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    access_frequency REAL DEFAULT 0.0,
                    relevance_score REAL DEFAULT 0.5,
                    retention_score REAL DEFAULT 0.5,
                    neural_weight REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    UNIQUE(key, context)
                )
            """)
            
            # Create time-based indices for efficient queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON short_term_items(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON short_term_items(last_accessed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON short_term_items(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON short_term_items(priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context ON short_term_items(context)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON short_term_items(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_score ON short_term_items(retention_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_weight ON short_term_items(neural_weight)")
            
            # Create composite indices for common queries and neural retention
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_priority ON short_term_items(context, priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type_created ON short_term_items(memory_type, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_priority ON short_term_items(retention_score, priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_access ON short_term_items(neural_weight, access_frequency)")
            
            # Create access pattern tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_key TEXT NOT NULL,
                    context TEXT NOT NULL,
                    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_type TEXT DEFAULT 'read',
                    session_id TEXT,
                    FOREIGN KEY (item_key, context) REFERENCES short_term_items (key, context)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_patterns(access_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_key_context ON access_patterns(item_key, context)")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error initializing database: {ex}")
            raise
    
    def _migrate_database(self):
        """Migrate existing database to support neural retention features."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if neural retention columns exist
            cursor.execute("PRAGMA table_info(short_term_items)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add missing columns for neural retention
            if 'access_frequency' not in columns:
                cursor.execute("ALTER TABLE short_term_items ADD COLUMN access_frequency REAL DEFAULT 0.0")
                self.logger.info("[ShortTermMemory] Added access_frequency column")
            
            if 'relevance_score' not in columns:
                cursor.execute("ALTER TABLE short_term_items ADD COLUMN relevance_score REAL DEFAULT 0.5")
                self.logger.info("[ShortTermMemory] Added relevance_score column")
            
            if 'retention_score' not in columns:
                cursor.execute("ALTER TABLE short_term_items ADD COLUMN retention_score REAL DEFAULT 0.5")
                self.logger.info("[ShortTermMemory] Added retention_score column")
            
            if 'neural_weight' not in columns:
                cursor.execute("ALTER TABLE short_term_items ADD COLUMN neural_weight REAL DEFAULT 0.5")
                self.logger.info("[ShortTermMemory] Added neural_weight column")
            
            # Create new indices if they don't exist
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_score ON short_term_items(retention_score)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_weight ON short_term_items(neural_weight)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_priority ON short_term_items(retention_score, priority)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_neural_access ON short_term_items(neural_weight, access_frequency)")
            except Exception as idx_ex:
                self.logger.warning(f"[ShortTermMemory] Could not create some indices: {idx_ex}")
            
            # Create access patterns table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_key TEXT NOT NULL,
                    context TEXT NOT NULL,
                    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_type TEXT DEFAULT 'read',
                    session_id TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_patterns(access_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_key_context ON access_patterns(item_key, context)")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error migrating database: {ex}")
            # Don't raise here, continue with existing functionality
    
    def add(self, key: str, item: Any, context: str = "default", priority: float = 0.5,
            memory_type: str = "general", tags: Optional[List[str]] = None, 
            ttl_seconds: Optional[int] = None) -> bool:
        """Add item with priority, neural retention scoring, and automatic capacity management."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                # Perform background cleanup if needed
                self._background_cleanup()
                
                # Serialize data
                if isinstance(item, (dict, list)):
                    data_str = json.dumps(item)
                    data_type = 'json'
                else:
                    data_str = str(item)
                    data_type = 'string'
                
                # Calculate size
                size_bytes = len(data_str.encode('utf-8'))
                
                # Check capacity and cleanup if needed
                self._manage_capacity(size_bytes)
                
                # Calculate expiry
                expires_at = None
                if ttl_seconds:
                    expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
                
                # Prepare tags
                tags_json = json.dumps(tags or [])
                
                # Calculate neural retention scores
                relevance_score = self._calculate_relevance_score(data_str, context, memory_type, tags or [])
                retention_score = self._calculate_retention_score(priority, relevance_score, size_bytes)
                neural_weight = self._calculate_neural_weight(retention_score, priority, relevance_score)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if item exists for access frequency calculation
                cursor.execute("SELECT access_count FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                existing = cursor.fetchone()
                access_frequency = 0.0
                if existing:
                    access_frequency = self._calculate_access_frequency(key, context)
                
                # Insert or replace item with neural retention features
                cursor.execute("""
                    INSERT OR REPLACE INTO short_term_items 
                    (key, data, data_type, context, priority, memory_type, tags, 
                     size_bytes, access_count, access_frequency, relevance_score, retention_score, neural_weight,
                     created_at, updated_at, last_accessed, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (key, data_str, data_type, context, priority, memory_type, 
                      tags_json, size_bytes, access_frequency, relevance_score, retention_score, neural_weight, expires_at))
                
                # Track access pattern
                self._track_access_pattern(key, context, 'write')
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"[ShortTermMemory] Added item '{key}' in context '{context}' ({size_bytes} bytes, retention_score: {retention_score:.3f})")
                return True
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error adding item '{key}': {ex}")
            if self.fallback:
                return self.fallback(key, item, context)
            return False
    
    def get(self, key: str, context: str = "default") -> Optional[Any]:
        """Get item and update access tracking with neural retention updates."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get item and check expiry
                cursor.execute("""
                    SELECT data, data_type, expires_at, access_count, priority, relevance_score, size_bytes
                    FROM short_term_items 
                    WHERE key = ? AND context = ?
                """, (key, context))
                
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                
                data_str, data_type, expires_at, access_count, priority, relevance_score, size_bytes = row
                
                # Check if expired
                if expires_at:
                    expires_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_dt:
                        cursor.execute("DELETE FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                        conn.commit()
                        conn.close()
                        return None
                
                # Calculate updated neural scores
                access_frequency = self._calculate_access_frequency(key, context)
                retention_score = self._calculate_retention_score(priority, relevance_score, size_bytes)
                neural_weight = self._calculate_neural_weight(retention_score, priority, relevance_score)
                
                # Update access tracking with neural retention scores
                cursor.execute("""
                    UPDATE short_term_items 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP,
                        access_frequency = ?, retention_score = ?, neural_weight = ?
                    WHERE key = ? AND context = ?
                """, (access_frequency, retention_score, neural_weight, key, context))
                
                # Track access pattern
                self._track_access_pattern(key, context, 'read')
                
                conn.commit()
                conn.close()
                
                # Deserialize data
                if data_type == 'json':
                    return json.loads(data_str)
                else:
                    return data_str
                    
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting item '{key}': {ex}")
            return None
    
    def get_recent(self, n: int = 5, context: Optional[str] = None, 
                   memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent items with optional filtering."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build query with optional filters
                query = """
                    SELECT key, data, data_type, context, priority, memory_type, 
                           tags, created_at, last_accessed, access_count
                    FROM short_term_items 
                    WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """
                params = []
                
                if context:
                    query += " AND context = ?"
                    params.append(context)
                
                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(n)
                
                cursor.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    key, data_str, data_type, ctx, priority, mem_type, tags_json, created_at, last_accessed, access_count = row
                    
                    # Deserialize data and tags
                    if data_type == 'json':
                        data = json.loads(data_str)
                    else:
                        data = data_str
                    
                    tags = json.loads(tags_json) if tags_json else []
                    
                    results.append({
                        'key': key,
                        'data': data,
                        'context': ctx,
                        'priority': priority,
                        'memory_type': mem_type,
                        'tags': tags,
                        'created_at': created_at,
                        'last_accessed': last_accessed,
                        'access_count': access_count
                    })
                
                conn.close()
                return results
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting recent items: {ex}")
            return []
    
    def search(self, query: str, context: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search items by content."""
        try:
            with self._lock:
                import sqlite3
                import json
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build search query
                sql_query = """
                    SELECT key, data, data_type, context, priority, memory_type, 
                           tags, created_at, last_accessed, access_count
                    FROM short_term_items 
                    WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    AND (data LIKE ? OR key LIKE ?)
                """
                params = [f'%{query}%', f'%{query}%']
                
                if context:
                    sql_query += " AND context = ?"
                    params.append(context)
                
                sql_query += " ORDER BY priority DESC, last_accessed DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql_query, params)
                results = []
                
                for row in cursor.fetchall():
                    key, data_str, data_type, ctx, priority, mem_type, tags_json, created_at, last_accessed, access_count = row
                    
                    # Deserialize data and tags
                    if data_type == 'json':
                        data = json.loads(data_str)
                    else:
                        data = data_str
                    
                    tags = json.loads(tags_json) if tags_json else []
                    
                    results.append({
                        'key': key,
                        'data': data,
                        'context': ctx,
                        'priority': priority,
                        'memory_type': mem_type,
                        'tags': tags,
                        'created_at': created_at,
                        'last_accessed': last_accessed,
                        'access_count': access_count
                    })
                
                conn.close()
                return results
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error searching items: {ex}")
            return []
    
    def cleanup_expired(self) -> int:
        """Remove expired items and old items based on neural retention policy."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Remove explicitly expired items
                cursor.execute("DELETE FROM short_term_items WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP")
                expired_count = cursor.rowcount
                
                # Neural retention-based cleanup
                if self.enable_neural_retention:
                    # Remove items older than retention period using neural scoring
                    cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
                    cursor.execute("""
                        DELETE FROM short_term_items 
                        WHERE created_at < ? AND retention_score < 0.6 AND neural_weight < 0.7
                    """, (cutoff_date,))
                    neural_removed = cursor.rowcount
                    
                    # Also remove very old items with low neural weights regardless of priority
                    very_old_cutoff = (datetime.now() - timedelta(days=self.retention_days * 2)).isoformat()
                    cursor.execute("""
                        DELETE FROM short_term_items 
                        WHERE created_at < ? AND neural_weight < 0.5
                    """, (very_old_cutoff,))
                    very_old_removed = cursor.rowcount
                    
                    old_count = neural_removed + very_old_removed
                else:
                    # Fallback to simple priority-based cleanup
                    cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
                    cursor.execute("""
                        DELETE FROM short_term_items 
                        WHERE created_at < ? AND priority < 0.8
                    """, (cutoff_date,))
                    old_count = cursor.rowcount
                
                # Clean up old access patterns
                pattern_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("DELETE FROM access_patterns WHERE access_timestamp < ?", (pattern_cutoff,))
                pattern_removed = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                total_removed = expired_count + old_count
                if total_removed > 0:
                    self.logger.info(f"[ShortTermMemory] Cleaned up {total_removed} items ({expired_count} expired, {old_count} old, {pattern_removed} old patterns)")
                
                return total_removed
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error during cleanup: {ex}")
            return 0
    
    def _manage_capacity(self, new_item_size: int):
        """Manage capacity by removing items using neural retention scoring."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current size
            cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM short_term_items")
            current_size = cursor.fetchone()[0]
            
            # If adding new item would exceed capacity, remove items
            while current_size + new_item_size > self.capacity_bytes:
                if self.enable_neural_retention:
                    # Use neural retention scoring for intelligent removal
                    cursor.execute("""
                        SELECT key, context, size_bytes FROM short_term_items 
                        ORDER BY neural_weight ASC, retention_score ASC, access_frequency ASC, last_accessed ASC 
                        LIMIT 1
                    """)
                else:
                    # Fallback to simple priority-based removal
                    cursor.execute("""
                        SELECT key, context, size_bytes FROM short_term_items 
                        ORDER BY priority ASC, access_count ASC, last_accessed ASC 
                        LIMIT 1
                    """)
                
                row = cursor.fetchone()
                if not row:
                    break  # No more items to remove
                
                key, context, size_bytes = row
                cursor.execute("DELETE FROM short_term_items WHERE key = ? AND context = ?", (key, context))
                current_size -= size_bytes
                
                self.logger.debug(f"[ShortTermMemory] Removed item '{key}' for capacity management (neural retention: {self.enable_neural_retention})")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error managing capacity: {ex}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM short_term_items")
                total_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT context) FROM short_term_items")
                total_contexts = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT memory_type) FROM short_term_items")
                total_types = cursor.fetchone()[0]
                
                # Size information
                cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM short_term_items")
                current_size = cursor.fetchone()[0]
                
                # Access statistics
                cursor.execute("SELECT AVG(access_count) FROM short_term_items")
                avg_access = cursor.fetchone()[0] or 0
                
                # Age statistics
                cursor.execute("""
                    SELECT COUNT(*) FROM short_term_items 
                    WHERE created_at > datetime('now', '-1 day')
                """)
                recent_items = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'total_items': total_items,
                    'total_contexts': total_contexts,
                    'total_memory_types': total_types,
                    'current_size_bytes': current_size,
                    'current_size_mb': round(current_size / (1024 * 1024), 2),
                    'capacity_gb': round(self.capacity_bytes / (1024 * 1024 * 1024), 2),
                    'utilization_percent': round((current_size / self.capacity_bytes) * 100, 2),
                    'average_access_count': round(avg_access, 2),
                    'recent_items_24h': recent_items,
                    'retention_days': self.retention_days
                }
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting stats: {ex}")
            return {'error': str(ex)}
    
    def clear(self, context: Optional[str] = None):
        """Clear items, optionally filtered by context."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if context:
                    cursor.execute("DELETE FROM short_term_items WHERE context = ?", (context,))
                    self.logger.info(f"[ShortTermMemory] Cleared context '{context}'")
                else:
                    cursor.execute("DELETE FROM short_term_items")
                    self.logger.info("[ShortTermMemory] Cleared all short-term memory")
                
                conn.commit()
                conn.close()
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error clearing memory: {ex}")
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items (for compatibility)."""
        return self.get_recent(n=1000)  # Return up to 1000 recent items
    
    def fallback_hook(self, *args, **kwargs):
        """Fallback hook for error handling."""
        if self.fallback:
            return self.fallback(*args, **kwargs)
        self.logger.warning("[ShortTermMemory] No fallback defined.")
    
    # Neural Retention Scoring Methods
    
    def _calculate_relevance_score(self, content: str, context: str, memory_type: str, tags: List[str]) -> float:
        """Calculate relevance score based on content analysis and context."""
        try:
            score = 0.5  # Base score
            
            # Content-based scoring
            content_lower = content.lower()
            
            # Higher score for structured data
            if memory_type in ['json', 'structured']:
                score += 0.1
            
            # Higher score for certain keywords
            important_keywords = ['error', 'critical', 'important', 'urgent', 'task', 'project']
            keyword_count = sum(1 for keyword in important_keywords if keyword in content_lower)
            score += min(0.2, keyword_count * 0.05)
            
            # Context-based scoring
            if context != 'default':
                score += 0.1  # Non-default contexts are more relevant
            
            # Tag-based scoring
            if tags:
                score += min(0.15, len(tags) * 0.03)
            
            # Content length scoring (moderate length preferred)
            content_length = len(content)
            if 100 <= content_length <= 1000:
                score += 0.1
            elif content_length > 5000:
                score -= 0.1  # Very long content less relevant for short-term
            
            return min(1.0, max(0.0, score))
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating relevance score: {ex}")
            return 0.5
    
    def _calculate_retention_score(self, priority: float, relevance_score: float, size_bytes: int) -> float:
        """Calculate retention score using neural network-inspired weighting."""
        try:
            # Weighted combination of factors
            score = (
                priority * self.retention_weights['priority'] +
                relevance_score * self.retention_weights['relevance'] +
                self._calculate_size_efficiency_score(size_bytes) * self.retention_weights['size_efficiency']
            )
            
            # Add recency bonus (new items get slight boost)
            score += 0.1 * self.retention_weights['recency']
            
            return min(1.0, max(0.0, score))
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating retention score: {ex}")
            return 0.5
    
    def _calculate_neural_weight(self, retention_score: float, priority: float, relevance_score: float) -> float:
        """Calculate neural weight for advanced retention decisions."""
        try:
            # Neural network-inspired activation function
            # Sigmoid-like function for smooth transitions
            import math
            
            # Combine scores with non-linear activation
            combined_score = (retention_score + priority + relevance_score) / 3.0
            
            # Apply sigmoid activation
            neural_weight = 1.0 / (1.0 + math.exp(-5 * (combined_score - 0.5)))
            
            return neural_weight
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating neural weight: {ex}")
            return 0.5
    
    def _calculate_size_efficiency_score(self, size_bytes: int) -> float:
        """Calculate efficiency score based on size (smaller is better for short-term)."""
        try:
            # Optimal size range for short-term memory (1KB - 10KB)
            if size_bytes <= 1024:  # <= 1KB
                return 1.0
            elif size_bytes <= 10240:  # <= 10KB
                return 0.8
            elif size_bytes <= 102400:  # <= 100KB
                return 0.6
            elif size_bytes <= 1048576:  # <= 1MB
                return 0.4
            else:
                return 0.2  # Large items less efficient for short-term
                
        except Exception:
            return 0.5
    
    def _calculate_access_frequency(self, key: str, context: str) -> float:
        """Calculate access frequency from recent access patterns."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get access count in last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM access_patterns 
                WHERE item_key = ? AND context = ? AND access_timestamp > ?
            """, (key, context, yesterday))
            
            recent_accesses = cursor.fetchone()[0]
            conn.close()
            
            # Convert to frequency score (0-1)
            return min(1.0, recent_accesses / 10.0)  # Max 10 accesses = 1.0 score
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error calculating access frequency: {ex}")
            return 0.0
    
    def _track_access_pattern(self, key: str, context: str, access_type: str = 'read'):
        """Track access patterns for neural retention analysis."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO access_patterns (item_key, context, access_type, session_id)
                VALUES (?, ?, ?, ?)
            """, (key, context, access_type, str(int(time.time()))))
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error tracking access pattern: {ex}")
    
    def _background_cleanup(self):
        """Perform background cleanup if needed."""
        try:
            now = datetime.now()
            if now - self._last_cleanup > self._cleanup_interval:
                self.cleanup_expired()
                self._last_cleanup = now
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error in background cleanup: {ex}")
    
    def update_neural_weights(self) -> int:
        """Update neural weights for all items based on current usage patterns."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get all items for neural weight update
                cursor.execute("""
                    SELECT key, context, priority, relevance_score, size_bytes, access_count
                    FROM short_term_items
                """)
                
                updated_count = 0
                for row in cursor.fetchall():
                    key, context, priority, relevance_score, size_bytes, access_count = row
                    
                    # Recalculate scores
                    access_frequency = self._calculate_access_frequency(key, context)
                    retention_score = self._calculate_retention_score(priority, relevance_score, size_bytes)
                    neural_weight = self._calculate_neural_weight(retention_score, priority, relevance_score)
                    
                    # Update in database
                    cursor.execute("""
                        UPDATE short_term_items 
                        SET access_frequency = ?, retention_score = ?, neural_weight = ?
                        WHERE key = ? AND context = ?
                    """, (access_frequency, retention_score, neural_weight, key, context))
                    
                    updated_count += 1
                
                conn.commit()
                conn.close()
                
                if updated_count > 0:
                    self.logger.info(f"[ShortTermMemory] Updated neural weights for {updated_count} items")
                
                return updated_count
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error updating neural weights: {ex}")
            return 0
    
    def get_neural_stats(self) -> Dict[str, Any]:
        """Get neural retention statistics."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get neural weight distribution
                cursor.execute("SELECT AVG(neural_weight), MIN(neural_weight), MAX(neural_weight) FROM short_term_items")
                avg_weight, min_weight, max_weight = cursor.fetchone()
                
                # Get retention score distribution
                cursor.execute("SELECT AVG(retention_score), MIN(retention_score), MAX(retention_score) FROM short_term_items")
                avg_retention, min_retention, max_retention = cursor.fetchone()
                
                # Get high-value items count
                cursor.execute("SELECT COUNT(*) FROM short_term_items WHERE neural_weight > 0.7")
                high_value_count = cursor.fetchone()[0]
                
                # Get access pattern stats
                cursor.execute("SELECT COUNT(*) FROM access_patterns WHERE access_timestamp > datetime('now', '-24 hours')")
                recent_accesses = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'neural_weights': {
                        'average': round(avg_weight or 0, 3),
                        'min': round(min_weight or 0, 3),
                        'max': round(max_weight or 0, 3)
                    },
                    'retention_scores': {
                        'average': round(avg_retention or 0, 3),
                        'min': round(min_retention or 0, 3),
                        'max': round(max_retention or 0, 3)
                    },
                    'high_value_items': high_value_count,
                    'recent_accesses_24h': recent_accesses,
                    'neural_retention_enabled': self.enable_neural_retention
                }
                
        except Exception as ex:
            self.logger.error(f"[ShortTermMemory] Error getting neural stats: {ex}")
            return {'error': str(ex)}

class LongTermMemory:
    """
    Enhanced long-term memory for persistent, structured, and research-driven storage.
    Features:
    - SQLite backend with vector search integration
    - Semantic search capabilities with research-driven storage
    - Backup and versioning system with < 9GB total capacity management
    - Compression and deduplication
    - Thread-safe operations
    """
    
    def __init__(self, db_path: Optional[str] = None, capacity_gb: float = 9.0, 
                 vector_backend: Optional[str] = "sqlite_faiss", fallback: Optional[Callable] = None):
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self.fallback = fallback
        self.logger = logging.getLogger("LongTermMemory")
        self._lock = threading.RLock()
        
        # Setup database path
        if db_path:
            self.db_path = db_path
        else:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, 'long_term_memory.db')
        
        # Initialize vector backend
        self.vector_backend_name = vector_backend
        self._init_vector_backend()
        self._init_database()
        
        self.logger.info(f"[LongTermMemory] Initialized with {capacity_gb}GB capacity, vector backend: {vector_backend}")
    
    def _init_vector_backend(self):
        """Initialize vector backend for semantic search."""
        try:
            # Import vector backend classes
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            from vector_memory import SQLiteFAISSBackend, InMemoryBackend
            
            if self.vector_backend_name == "sqlite_faiss":
                vector_db_path = self.db_path.replace('.db', '_vectors.db')
                self.vector_backend = SQLiteFAISSBackend(vector_db_path)
            else:
                # Fallback to in-memory backend
                self.vector_backend = InMemoryBackend()
                
        except Exception as ex:
            self.logger.warning(f"[LongTermMemory] Could not initialize vector backend: {ex}, using fallback")
            # Simple fallback vector backend
            self.vector_backend = None
    
    def _init_database(self):
        """Initialize SQLite database with comprehensive schema."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main long-term storage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT DEFAULT 'text',
                    compressed_content BLOB,
                    compression_ratio REAL DEFAULT 1.0,
                    vector_id INTEGER,
                    metadata TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]',
                    category TEXT DEFAULT 'general',
                    research_sources TEXT DEFAULT '[]',
                    quality_score REAL DEFAULT 0.5,
                    importance_score REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    parent_version_id INTEGER,
                    UNIQUE(key),
                    FOREIGN KEY (parent_version_id) REFERENCES long_term_items (id)
                )
            """)
            
            # Backup and versioning table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    version_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    change_description TEXT,
                    FOREIGN KEY (memory_id) REFERENCES long_term_items (id),
                    UNIQUE(memory_id, version_number)
                )
            """)
            
            # Research sources tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_url TEXT,
                    source_title TEXT,
                    source_type TEXT DEFAULT 'web',
                    credibility_score REAL DEFAULT 0.5,
                    last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Memory relationships for semantic connections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_memory_id INTEGER NOT NULL,
                    target_memory_id INTEGER NOT NULL,
                    relationship_type TEXT DEFAULT 'related',
                    strength REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_memory_id) REFERENCES long_term_items (id),
                    FOREIGN KEY (target_memory_id) REFERENCES long_term_items (id),
                    UNIQUE(source_memory_id, target_memory_id, relationship_type)
                )
            """)
            
            # Create indices for efficient queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_key ON long_term_items(key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_items(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_created ON long_term_items(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_accessed ON long_term_items(last_accessed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_importance ON long_term_items(importance_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_quality ON long_term_items(quality_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions(memory_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON memory_relationships(source_memory_id)")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error initializing database: {ex}")
            raise
    
    def add(self, key: str, content: Any, category: str = "general", 
            tags: Optional[List[str]] = None, research_sources: Optional[List[str]] = None,
            importance_score: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add content with compression, vectorization, and versioning."""
        try:
            with self._lock:
                import sqlite3
                import json
                import zlib
                
                # Serialize content
                if isinstance(content, (dict, list)):
                    content_str = json.dumps(content)
                    content_type = 'json'
                else:
                    content_str = str(content)
                    content_type = 'text'
                
                # Compress content if it's large
                original_size = len(content_str.encode('utf-8'))
                compressed_content = None
                compression_ratio = 1.0
                
                if original_size > 1024:  # Compress if > 1KB
                    compressed_content = zlib.compress(content_str.encode('utf-8'))
                    compression_ratio = len(compressed_content) / original_size
                
                # Manage capacity
                self._manage_capacity(original_size)
                
                # Generate vector for semantic search
                vector_id = None
                if self.vector_backend and content_str.strip():
                    try:
                        # Simple TF-IDF-like vector generation
                        words = content_str.lower().split()
                        word_freq = {}
                        for word in words:
                            word_freq[word] = word_freq.get(word, 0) + 1
                        
                        # Create vector from top words
                        vector_dict = {}
                        for i, (word, freq) in enumerate(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:100]):
                            vector_dict[f"word_{i}"] = freq / len(words)
                        
                        if vector_dict:
                            vector_metadata = {
                                'key': key,
                                'category': category,
                                'content_preview': content_str[:200]
                            }
                            vector_id = self.vector_backend.add_vector(vector_dict, vector_metadata)
                    except Exception as ex:
                        self.logger.warning(f"[LongTermMemory] Could not create vector for '{key}': {ex}")
                
                # Prepare data for database
                tags_json = json.dumps(tags or [])
                sources_json = json.dumps(research_sources or [])
                metadata_json = json.dumps(metadata or {})
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if key exists for versioning
                cursor.execute("SELECT id, version FROM long_term_items WHERE key = ?", (key,))
                existing = cursor.fetchone()
                
                if existing:
                    # Create version backup
                    existing_id, current_version = existing
                    cursor.execute("""
                        INSERT INTO memory_versions (memory_id, version_number, content, metadata, change_description)
                        SELECT id, version, content, metadata, 'Updated content' 
                        FROM long_term_items WHERE id = ?
                    """, (existing_id,))
                    
                    # Update existing item
                    cursor.execute("""
                        UPDATE long_term_items 
                        SET content = ?, content_type = ?, compressed_content = ?, compression_ratio = ?,
                            vector_id = ?, metadata = ?, tags = ?, category = ?, research_sources = ?,
                            importance_score = ?, size_bytes = ?, updated_at = CURRENT_TIMESTAMP,
                            version = version + 1
                        WHERE key = ?
                    """, (content_str, content_type, compressed_content, compression_ratio,
                          vector_id, metadata_json, tags_json, category, sources_json,
                          importance_score, original_size, key))
                else:
                    # Insert new item
                    cursor.execute("""
                        INSERT INTO long_term_items 
                        (key, content, content_type, compressed_content, compression_ratio,
                         vector_id, metadata, tags, category, research_sources, importance_score, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (key, content_str, content_type, compressed_content, compression_ratio,
                          vector_id, metadata_json, tags_json, category, sources_json,
                          importance_score, original_size))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"[LongTermMemory] Added/updated '{key}' in category '{category}' ({original_size} bytes, {compression_ratio:.2f} compression)")
                return True
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error adding item '{key}': {ex}")
            if self.fallback:
                return self.fallback(key, content)
            return False
    
    def get(self, key: str, include_versions: bool = False) -> Optional[Dict[str, Any]]:
        """Get item with decompression and access tracking."""
        try:
            with self._lock:
                import sqlite3
                import json
                import zlib
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, content, content_type, compressed_content, compression_ratio,
                           metadata, tags, category, research_sources, quality_score, 
                           importance_score, access_count, created_at, updated_at, version
                    FROM long_term_items WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                
                (item_id, content, content_type, compressed_content, compression_ratio,
                 metadata_json, tags_json, category, sources_json, quality_score,
                 importance_score, access_count, created_at, updated_at, version) = row
                
                # Decompress content if needed
                if compressed_content:
                    content = zlib.decompress(compressed_content).decode('utf-8')
                
                # Update access tracking
                cursor.execute("""
                    UPDATE long_term_items 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE key = ?
                """, (key,))
                
                # Get versions if requested
                versions = []
                if include_versions:
                    cursor.execute("""
                        SELECT version_number, content, created_at, change_description
                        FROM memory_versions WHERE memory_id = ?
                        ORDER BY version_number DESC
                    """, (item_id,))
                    
                    for v_row in cursor.fetchall():
                        versions.append({
                            'version': v_row[0],
                            'content': v_row[1],
                            'created_at': v_row[2],
                            'description': v_row[3]
                        })
                
                conn.commit()
                conn.close()
                
                # Deserialize data
                if content_type == 'json':
                    data = json.loads(content)
                else:
                    data = content
                
                result = {
                    'key': key,
                    'data': data,
                    'metadata': json.loads(metadata_json) if metadata_json else {},
                    'tags': json.loads(tags_json) if tags_json else [],
                    'category': category,
                    'research_sources': json.loads(sources_json) if sources_json else [],
                    'quality_score': quality_score,
                    'importance_score': importance_score,
                    'access_count': access_count + 1,
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'version': version,
                    'compression_ratio': compression_ratio
                }
                
                if include_versions:
                    result['versions'] = versions
                
                return result
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error getting item '{key}': {ex}")
            return None
    
    def semantic_search(self, query: str, limit: int = 10, category: Optional[str] = None,
                       min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Perform semantic search using vector backend."""
        try:
            with self._lock:
                if not self.vector_backend:
                    return self._fallback_text_search(query, limit, category)
                
                # Generate query vector
                words = query.lower().split()
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                query_vector = {}
                for i, (word, freq) in enumerate(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:100]):
                    query_vector[f"word_{i}"] = freq / len(words)
                
                if not query_vector:
                    return []
                
                # Search vectors
                vector_results = self.vector_backend.search_vector(query_vector, limit * 2, min_similarity)
                
                # Get full items from database
                results = []
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for v_result in vector_results:
                    metadata = v_result.get('metadata', {})
                    key = metadata.get('key')
                    if not key:
                        continue
                    
                    # Apply category filter
                    if category:
                        cursor.execute("SELECT category FROM long_term_items WHERE key = ?", (key,))
                        row = cursor.fetchone()
                        if not row or row[0] != category:
                            continue
                    
                    # Get full item
                    item = self.get(key)
                    if item:
                        item['similarity'] = v_result.get('similarity', 0.0)
                        results.append(item)
                
                conn.close()
                
                # Sort by similarity and importance
                results.sort(key=lambda x: (x.get('similarity', 0) + x.get('importance_score', 0)) / 2, reverse=True)
                return results[:limit]
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error in semantic search: {ex}")
            return self._fallback_text_search(query, limit, category)
    
    def _fallback_text_search(self, query: str, limit: int = 10, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback text search when vector backend is unavailable."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql_query = """
                SELECT key FROM long_term_items 
                WHERE (content LIKE ? OR key LIKE ?)
            """
            params = [f'%{query}%', f'%{query}%']
            
            if category:
                sql_query += " AND category = ?"
                params.append(category)
            
            sql_query += " ORDER BY importance_score DESC, last_accessed DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql_query, params)
            results = []
            
            for row in cursor.fetchall():
                key = row[0]
                item = self.get(key)
                if item:
                    results.append(item)
            
            conn.close()
            return results
            
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error in fallback search: {ex}")
            return []
    
    def add_relationship(self, source_key: str, target_key: str, 
                        relationship_type: str = "related", strength: float = 0.5) -> bool:
        """Add semantic relationship between memories."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get memory IDs
                cursor.execute("SELECT id FROM long_term_items WHERE key = ?", (source_key,))
                source_row = cursor.fetchone()
                if not source_row:
                    conn.close()
                    return False
                
                cursor.execute("SELECT id FROM long_term_items WHERE key = ?", (target_key,))
                target_row = cursor.fetchone()
                if not target_row:
                    conn.close()
                    return False
                
                source_id, target_id = source_row[0], target_row[0]
                
                # Insert relationship
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_relationships 
                    (source_memory_id, target_memory_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                """, (source_id, target_id, relationship_type, strength))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"[LongTermMemory] Added relationship: {source_key} -> {target_key} ({relationship_type})")
                return True
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error adding relationship: {ex}")
            return False
    
    def get_related(self, key: str, relationship_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get related memories."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get source memory ID
                cursor.execute("SELECT id FROM long_term_items WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return []
                
                source_id = row[0]
                
                # Build query for related items
                sql_query = """
                    SELECT ltm.key, mr.relationship_type, mr.strength
                    FROM memory_relationships mr
                    JOIN long_term_items ltm ON mr.target_memory_id = ltm.id
                    WHERE mr.source_memory_id = ?
                """
                params = [source_id]
                
                if relationship_type:
                    sql_query += " AND mr.relationship_type = ?"
                    params.append(relationship_type)
                
                sql_query += " ORDER BY mr.strength DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql_query, params)
                results = []
                
                for row in cursor.fetchall():
                    related_key, rel_type, strength = row
                    item = self.get(related_key)
                    if item:
                        item['relationship_type'] = rel_type
                        item['relationship_strength'] = strength
                        results.append(item)
                
                conn.close()
                return results
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error getting related items: {ex}")
            return []
    
    def _manage_capacity(self, new_item_size: int):
        """Manage capacity by removing low-importance old items."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current size
            cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM long_term_items")
            current_size = cursor.fetchone()[0]
            
            # If adding new item would exceed capacity, remove items
            while current_size + new_item_size > self.capacity_bytes:
                # Remove lowest importance, least accessed items first
                cursor.execute("""
                    SELECT key, size_bytes FROM long_term_items 
                    ORDER BY importance_score ASC, access_count ASC, last_accessed ASC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if not row:
                    break  # No more items to remove
                
                key, size_bytes = row
                
                # Remove from vector backend if exists
                if self.vector_backend:
                    try:
                        cursor.execute("SELECT vector_id FROM long_term_items WHERE key = ?", (key,))
                        vector_row = cursor.fetchone()
                        if vector_row and vector_row[0]:
                            # Note: Most vector backends don't support deletion by ID
                            # This would need backend-specific implementation
                            pass
                    except Exception:
                        pass
                
                # Remove from database
                cursor.execute("DELETE FROM long_term_items WHERE key = ?", (key,))
                cursor.execute("DELETE FROM memory_versions WHERE memory_id IN (SELECT id FROM long_term_items WHERE key = ?)", (key,))
                
                current_size -= size_bytes
                self.logger.debug(f"[LongTermMemory] Removed item '{key}' for capacity management")
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error managing capacity: {ex}")
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """Create backup of long-term memory."""
        try:
            import shutil
            import datetime
            
            if not backup_path:
                backup_dir = os.path.dirname(self.db_path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"ltm_backup_{timestamp}.db")
            
            shutil.copy2(self.db_path, backup_path)
            
            # Also backup vector database if exists
            vector_db_path = self.db_path.replace('.db', '_vectors.db')
            if os.path.exists(vector_db_path):
                vector_backup_path = backup_path.replace('.db', '_vectors.db')
                shutil.copy2(vector_db_path, vector_backup_path)
            
            self.logger.info(f"[LongTermMemory] Created backup: {backup_path}")
            return backup_path
            
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error creating backup: {ex}")
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM long_term_items")
                total_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT category) FROM long_term_items")
                total_categories = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memory_relationships")
                total_relationships = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memory_versions")
                total_versions = cursor.fetchone()[0]
                
                # Size information
                cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM long_term_items")
                current_size = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(compression_ratio) FROM long_term_items WHERE compression_ratio < 1.0")
                avg_compression = cursor.fetchone()[0] or 1.0
                
                # Quality metrics
                cursor.execute("SELECT AVG(quality_score) FROM long_term_items")
                avg_quality = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT AVG(importance_score) FROM long_term_items")
                avg_importance = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT AVG(access_count) FROM long_term_items")
                avg_access = cursor.fetchone()[0] or 0.0
                
                # Vector backend stats
                vector_stats = {}
                if self.vector_backend:
                    try:
                        vector_stats = self.vector_backend.get_stats()
                    except Exception:
                        vector_stats = {'error': 'unavailable'}
                
                conn.close()
                
                return {
                    'total_items': total_items,
                    'total_categories': total_categories,
                    'total_relationships': total_relationships,
                    'total_versions': total_versions,
                    'current_size_bytes': current_size,
                    'current_size_gb': round(current_size / (1024 * 1024 * 1024), 3),
                    'capacity_gb': round(self.capacity_bytes / (1024 * 1024 * 1024), 2),
                    'utilization_percent': round((current_size / self.capacity_bytes) * 100, 2),
                    'average_compression_ratio': round(avg_compression, 3),
                    'average_quality_score': round(avg_quality, 2),
                    'average_importance_score': round(avg_importance, 2),
                    'average_access_count': round(avg_access, 2),
                    'vector_backend': self.vector_backend_name,
                    'vector_stats': vector_stats
                }
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error getting stats: {ex}")
            return {'error': str(ex)}
    
    def remove(self, key: str) -> bool:
        """Remove item and its relationships."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get item ID for cleanup
                cursor.execute("SELECT id FROM long_term_items WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return False
                
                item_id = row[0]
                
                # Remove relationships
                cursor.execute("DELETE FROM memory_relationships WHERE source_memory_id = ? OR target_memory_id = ?", (item_id, item_id))
                
                # Remove versions
                cursor.execute("DELETE FROM memory_versions WHERE memory_id = ?", (item_id,))
                
                # Remove main item
                cursor.execute("DELETE FROM long_term_items WHERE key = ?", (key,))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"[LongTermMemory] Removed item: {key}")
                return True
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error removing item '{key}': {ex}")
            return False
    
    def clear(self, category: Optional[str] = None):
        """Clear items, optionally filtered by category."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if category:
                    # Get item IDs for cleanup
                    cursor.execute("SELECT id FROM long_term_items WHERE category = ?", (category,))
                    item_ids = [row[0] for row in cursor.fetchall()]
                    
                    if item_ids:
                        placeholders = ','.join('?' * len(item_ids))
                        cursor.execute(f"DELETE FROM memory_relationships WHERE source_memory_id IN ({placeholders}) OR target_memory_id IN ({placeholders})", item_ids + item_ids)
                        cursor.execute(f"DELETE FROM memory_versions WHERE memory_id IN ({placeholders})", item_ids)
                        cursor.execute("DELETE FROM long_term_items WHERE category = ?", (category,))
                    
                    self.logger.info(f"[LongTermMemory] Cleared category '{category}'")
                else:
                    cursor.execute("DELETE FROM memory_relationships")
                    cursor.execute("DELETE FROM memory_versions")
                    cursor.execute("DELETE FROM long_term_items")
                    self.logger.info("[LongTermMemory] Cleared all long-term memory")
                
                conn.commit()
                conn.close()
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error clearing memory: {ex}")
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items (for compatibility)."""
        try:
            with self._lock:
                import sqlite3
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT key FROM long_term_items ORDER BY importance_score DESC, last_accessed DESC")
                results = []
                
                for row in cursor.fetchall():
                    key = row[0]
                    item = self.get(key)
                    if item:
                        results.append(item)
                
                conn.close()
                return results
                
        except Exception as ex:
            self.logger.error(f"[LongTermMemory] Error getting all items: {ex}")
            return []
    
    def fallback_hook(self, *args, **kwargs):
        """Fallback hook for error handling."""
        if self.fallback:
            return self.fallback(*args, **kwargs)
        self.logger.warning("[LongTermMemory] No fallback defined.")