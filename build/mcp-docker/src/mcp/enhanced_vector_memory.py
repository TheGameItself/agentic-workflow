#!/usr/bin/env python3
"""
Enhanced Vector Memory System with Multiple Backends
Provides automatic backend selection, fallback mechanisms, and optimization.
Supports SQLite/FAISS, Milvus, Qdrant, Annoy, and research-driven backends.
"""

import logging
import threading
import time
import json
import os
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import vector backends
try:
    from .vector_memory import (
        VectorBackend, SQLiteFAISSBackend, MilvusBackend, 
        AnnoyBackend, QdrantBackend, InMemoryBackend,
        NeuromorphicBackend, ReservoirBackend, HyperdimensionalBackend
    )
except ImportError:
    # Fallback imports for standalone usage
    import sys
    sys.path.append(os.path.dirname(__file__))
    from vector_memory import (
        VectorBackend, SQLiteFAISSBackend, MilvusBackend, 
        AnnoyBackend, QdrantBackend, InMemoryBackend,
        NeuromorphicBackend, ReservoirBackend, HyperdimensionalBackend
    )

class BackendType(Enum):
    """Supported vector backend types."""
    SQLITE_FAISS = "sqlite_faiss"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    ANNOY = "annoy"
    IN_MEMORY = "in_memory"
    NEUROMORPHIC = "neuromorphic"
    RESERVOIR = "reservoir"
    HYPERDIMENSIONAL = "hyperdimensional"

@dataclass
class BackendConfig:
    """Configuration for vector backends."""
    backend_type: BackendType
    config: Dict[str, Any]
    priority: int = 1
    enabled: bool = True
    fallback_for: Optional[List[BackendType]] = None

@dataclass
class VectorSearchResult:
    """Standardized vector search result."""
    id: Union[int, str]
    similarity: float
    metadata: Dict[str, Any]
    backend_used: str
    search_time_ms: float

class EnhancedVectorMemorySystem:
    """
    Enhanced vector memory system with multiple backends, automatic selection,
    and comprehensive fallback mechanisms.
    
    Features:
    - Multiple backend support with automatic selection
    - Fallback mechanisms for reliability
    - Performance monitoring and optimization
    - Similarity scoring and metadata handling
    - Thread-safe operations
    - Capacity management and cleanup
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 default_backend: BackendType = BackendType.SQLITE_FAISS,
                 capacity_gb: float = 2.0):
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)
        self.logger = logging.getLogger("EnhancedVectorMemorySystem")
        self._lock = threading.RLock()
        
        # Backend management
        self.backends: Dict[BackendType, VectorBackend] = {}
        self.backend_configs: Dict[BackendType, BackendConfig] = {}
        self.backend_health: Dict[BackendType, Dict[str, Any]] = {}
        self.active_backend: Optional[BackendType] = None
        self.fallback_chain: List[BackendType] = []
        
        # Performance tracking
        self.performance_stats: Dict[BackendType, Dict[str, float]] = {}
        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(minutes=5)
        
        # Load configuration
        self._load_config(config_path)
        
        # Initialize backends
        self._initialize_backends()
        
        # Select optimal backend
        self._select_optimal_backend()
        
        self.logger.info(f"[EnhancedVectorMemorySystem] Initialized with {len(self.backends)} backends, active: {self.active_backend}")
    
    def _load_config(self, config_path: Optional[str]):
        """Load backend configurations."""
        # Default configurations
        default_configs = {
            BackendType.SQLITE_FAISS: BackendConfig(
                backend_type=BackendType.SQLITE_FAISS,
                config={
                    "db_path": self._get_data_path("vector_memory.db")
                },
                priority=1,
                enabled=True
            ),
            BackendType.IN_MEMORY: BackendConfig(
                backend_type=BackendType.IN_MEMORY,
                config={},
                priority=2,
                enabled=True,
                fallback_for=[BackendType.SQLITE_FAISS]
            ),
            BackendType.MILVUS: BackendConfig(
                backend_type=BackendType.MILVUS,
                config={
                    "host": "localhost",
                    "port": "19530",
                    "collection_name": "mcp_vectors",
                    "dim": 128
                },
                priority=0,  # Highest priority if available
                enabled=False  # Disabled by default
            ),
            BackendType.QDRANT: BackendConfig(
                backend_type=BackendType.QDRANT,
                config={
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "mcp_vectors",
                    "dim": 128
                },
                priority=0,
                enabled=False
            ),
            BackendType.ANNOY: BackendConfig(
                backend_type=BackendType.ANNOY,
                config={
                    "dim": 128,
                    "index_path": self._get_data_path("annoy_index.ann")
                },
                priority=3,
                enabled=True
            ),
            BackendType.NEUROMORPHIC: BackendConfig(
                backend_type=BackendType.NEUROMORPHIC,
                config={
                    "dim": 128
                },
                priority=4,
                enabled=True,
                fallback_for=[BackendType.MILVUS, BackendType.QDRANT]
            ),
            BackendType.RESERVOIR: BackendConfig(
                backend_type=BackendType.RESERVOIR,
                config={
                    "dim": 128,
                    "reservoir_size": 256
                },
                priority=5,
                enabled=True
            ),
            BackendType.HYPERDIMENSIONAL: BackendConfig(
                backend_type=BackendType.HYPERDIMENSIONAL,
                config={
                    "dim": 10000
                },
                priority=6,
                enabled=True
            )
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Update default configs with file configs
                for backend_name, config_data in file_config.items():
                    try:
                        backend_type = BackendType(backend_name)
                        if backend_type in default_configs:
                            default_configs[backend_type].config.update(config_data.get('config', {}))
                            default_configs[backend_type].enabled = config_data.get('enabled', True)
                            default_configs[backend_type].priority = config_data.get('priority', default_configs[backend_type].priority)
                    except ValueError:
                        self.logger.warning(f"Unknown backend type in config: {backend_name}")
                        
            except Exception as ex:
                self.logger.error(f"Error loading config from {config_path}: {ex}")
        
        self.backend_configs = default_configs
    
    def _get_data_path(self, filename: str) -> str:
        """Get path for data files."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, filename)
    
    def _initialize_backends(self):
        """Initialize all enabled backends."""
        for backend_type, config in self.backend_configs.items():
            if not config.enabled:
                continue
            
            try:
                backend = self._create_backend(backend_type, config.config)
                if backend:
                    self.backends[backend_type] = backend
                    self.performance_stats[backend_type] = {
                        'total_operations': 0,
                        'total_time_ms': 0,
                        'avg_time_ms': 0,
                        'error_count': 0,
                        'last_error': None
                    }
                    self.backend_health[backend_type] = {
                        'status': 'healthy',
                        'last_check': datetime.now(),
                        'error_count': 0,
                        'last_error': None
                    }
                    self.logger.info(f"[EnhancedVectorMemorySystem] Initialized backend: {backend_type.value}")
                
            except Exception as ex:
                self.logger.error(f"[EnhancedVectorMemorySystem] Failed to initialize {backend_type.value}: {ex}")
                self.backend_health[backend_type] = {
                    'status': 'failed',
                    'last_check': datetime.now(),
                    'error_count': 1,
                    'last_error': str(ex)
                }
    
    def _create_backend(self, backend_type: BackendType, config: Dict[str, Any]) -> Optional[VectorBackend]:
        """Create a specific backend instance."""
        try:
            if backend_type == BackendType.SQLITE_FAISS:
                return SQLiteFAISSBackend(config['db_path'])
            elif backend_type == BackendType.MILVUS:
                return MilvusBackend(config)
            elif backend_type == BackendType.QDRANT:
                return QdrantBackend(config)
            elif backend_type == BackendType.ANNOY:
                return AnnoyBackend(config)
            elif backend_type == BackendType.IN_MEMORY:
                return InMemoryBackend()
            elif backend_type == BackendType.NEUROMORPHIC:
                return NeuromorphicBackend(config.get('dim', 128))
            elif backend_type == BackendType.RESERVOIR:
                return ReservoirBackend(config.get('dim', 128), config.get('reservoir_size', 256))
            elif backend_type == BackendType.HYPERDIMENSIONAL:
                return HyperdimensionalBackend(config.get('dim', 10000))
            else:
                self.logger.warning(f"Unknown backend type: {backend_type}")
                return None
                
        except Exception as ex:
            self.logger.error(f"Error creating {backend_type.value} backend: {ex}")
            return None
    
    def _select_optimal_backend(self):
        """Select the optimal backend based on priority and health."""
        if not self.backends:
            self.logger.error("[EnhancedVectorMemorySystem] No backends available!")
            return
        
        # Sort backends by priority (lower number = higher priority) and health
        available_backends = []
        for backend_type, backend in self.backends.items():
            config = self.backend_configs[backend_type]
            health = self.backend_health[backend_type]
            
            if health['status'] == 'healthy':
                available_backends.append((backend_type, config.priority))
        
        if not available_backends:
            # All backends are unhealthy, use any available
            available_backends = [(bt, self.backend_configs[bt].priority) for bt in self.backends.keys()]
        
        # Sort by priority
        available_backends.sort(key=lambda x: x[1])
        
        # Select the highest priority backend
        self.active_backend = available_backends[0][0]
        
        # Build fallback chain
        self.fallback_chain = [bt for bt, _ in available_backends[1:]]
        
        self.logger.info(f"[EnhancedVectorMemorySystem] Selected active backend: {self.active_backend.value}")
        self.logger.debug(f"[EnhancedVectorMemorySystem] Fallback chain: {[bt.value for bt in self.fallback_chain]}")
    
    def add_vector(self, vector: Union[Dict[str, float], List[float]], 
                   metadata: Dict[str, Any], 
                   backend_preference: Optional[BackendType] = None) -> Optional[int]:
        """Add vector with automatic backend selection and fallback."""
        with self._lock:
            # Normalize vector format
            if isinstance(vector, list):
                vector_dict = {f"dim_{i}": float(v) for i, v in enumerate(vector)}
            else:
                vector_dict = {k: float(v) for k, v in vector.items()}
            
            # Try preferred backend first
            if backend_preference and backend_preference in self.backends:
                result = self._try_add_vector(backend_preference, vector_dict, metadata)
                if result is not None:
                    return result
            
            # Try active backend
            if self.active_backend:
                result = self._try_add_vector(self.active_backend, vector_dict, metadata)
                if result is not None:
                    return result
            
            # Try fallback backends
            for backend_type in self.fallback_chain:
                result = self._try_add_vector(backend_type, vector_dict, metadata)
                if result is not None:
                    return result
            
            self.logger.error("[EnhancedVectorMemorySystem] All backends failed to add vector")
            return None
    
    def _try_add_vector(self, backend_type: BackendType, vector: Dict[str, float], 
                       metadata: Dict[str, Any]) -> Optional[int]:
        """Try to add vector to specific backend."""
        try:
            start_time = time.time()
            backend = self.backends[backend_type]
            
            # Add backend info to metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata['backend'] = backend_type.value
            enhanced_metadata['added_at'] = datetime.now().isoformat()
            
            result = backend.add_vector(vector, enhanced_metadata)
            
            # Update performance stats
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(backend_type, elapsed_ms, success=True)
            
            return result
            
        except Exception as ex:
            self.logger.error(f"[EnhancedVectorMemorySystem] Error adding vector to {backend_type.value}: {ex}")
            self._update_performance_stats(backend_type, 0, success=False, error=str(ex))
            self._update_backend_health(backend_type, error=str(ex))
            return None
    
    def search_similar(self, query_vector: Union[Dict[str, float], List[float]], 
                      limit: int = 10, min_similarity: float = 0.1,
                      backend_preference: Optional[BackendType] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors with comprehensive fallback."""
        with self._lock:
            # Normalize vector format
            if isinstance(query_vector, list):
                query_dict = {f"dim_{i}": float(v) for i, v in enumerate(query_vector)}
            else:
                query_dict = {k: float(v) for k, v in query_vector.items()}
            
            # Try preferred backend first
            if backend_preference and backend_preference in self.backends:
                results = self._try_search_similar(backend_preference, query_dict, limit, min_similarity, metadata_filter)
                if results:
                    return results
            
            # Try active backend
            if self.active_backend:
                results = self._try_search_similar(self.active_backend, query_dict, limit, min_similarity, metadata_filter)
                if results:
                    return results
            
            # Try fallback backends
            for backend_type in self.fallback_chain:
                results = self._try_search_similar(backend_type, query_dict, limit, min_similarity, metadata_filter)
                if results:
                    return results
            
            self.logger.error("[EnhancedVectorMemorySystem] All backends failed to search vectors")
            return []
    
    def _try_search_similar(self, backend_type: BackendType, query_vector: Dict[str, float],
                           limit: int, min_similarity: float, 
                           metadata_filter: Optional[Dict[str, Any]]) -> List[VectorSearchResult]:
        """Try to search vectors in specific backend."""
        try:
            start_time = time.time()
            backend = self.backends[backend_type]
            
            raw_results = backend.search_vector(query_vector, limit, min_similarity)
            
            # Convert to standardized format
            results = []
            for raw_result in raw_results:
                # Apply metadata filter if specified
                if metadata_filter:
                    metadata = raw_result.get('metadata', {})
                    if not all(metadata.get(k) == v for k, v in metadata_filter.items()):
                        continue
                
                result = VectorSearchResult(
                    id=raw_result.get('id', 0),
                    similarity=raw_result.get('similarity', 0.0),
                    metadata=raw_result.get('metadata', {}),
                    backend_used=backend_type.value,
                    search_time_ms=(time.time() - start_time) * 1000
                )
                results.append(result)
            
            # Update performance stats
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(backend_type, elapsed_ms, success=True)
            
            return results
            
        except Exception as ex:
            self.logger.error(f"[EnhancedVectorMemorySystem] Error searching in {backend_type.value}: {ex}")
            self._update_performance_stats(backend_type, 0, success=False, error=str(ex))
            self._update_backend_health(backend_type, error=str(ex))
            return []
    
    def batch_search(self, query_vectors: List[Union[Dict[str, float], List[float]]], 
                    limit: int = 10, min_similarity: float = 0.1) -> List[List[VectorSearchResult]]:
        """Batch search for multiple query vectors."""
        results = []
        for query_vector in query_vectors:
            result = self.search_similar(query_vector, limit, min_similarity)
            results.append(result)
        return results
    
    def optimize_backend(self, backend_type: Optional[BackendType] = None):
        """Optimize backend indices and performance."""
        backends_to_optimize = [backend_type] if backend_type else list(self.backends.keys())
        
        for bt in backends_to_optimize:
            if bt not in self.backends:
                continue
            
            try:
                backend = self.backends[bt]
                self.logger.info(f"[EnhancedVectorMemorySystem] Optimizing backend: {bt.value}")
                
                # Create/rebuild index
                backend.create_index()
                
                # Load index into memory if needed
                backend.load_index()
                
                self.logger.info(f"[EnhancedVectorMemorySystem] Optimized backend: {bt.value}")
                
            except Exception as ex:
                self.logger.error(f"[EnhancedVectorMemorySystem] Error optimizing {bt.value}: {ex}")
    
    def health_check(self, force: bool = False):
        """Perform health check on all backends."""
        now = datetime.now()
        if not force and now - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = now
        
        for backend_type, backend in self.backends.items():
            try:
                # Get backend stats as health indicator
                stats = backend.get_stats()
                
                if 'error' in stats:
                    self._update_backend_health(backend_type, error=stats['error'])
                else:
                    self._update_backend_health(backend_type, healthy=True)
                    
            except Exception as ex:
                self._update_backend_health(backend_type, error=str(ex))
        
        # Reselect optimal backend if needed
        current_health = self.backend_health.get(self.active_backend, {})
        if current_health.get('status') != 'healthy':
            self.logger.warning(f"[EnhancedVectorMemorySystem] Active backend {self.active_backend.value} is unhealthy, reselecting...")
            self._select_optimal_backend()
    
    def _update_performance_stats(self, backend_type: BackendType, elapsed_ms: float, 
                                 success: bool, error: Optional[str] = None):
        """Update performance statistics for a backend."""
        if backend_type not in self.performance_stats:
            return
        
        stats = self.performance_stats[backend_type]
        stats['total_operations'] += 1
        
        if success:
            stats['total_time_ms'] += elapsed_ms
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_operations']
        else:
            stats['error_count'] += 1
            stats['last_error'] = error
    
    def _update_backend_health(self, backend_type: BackendType, 
                              healthy: bool = False, error: Optional[str] = None):
        """Update backend health status."""
        if backend_type not in self.backend_health:
            return
        
        health = self.backend_health[backend_type]
        health['last_check'] = datetime.now()
        
        if healthy:
            health['status'] = 'healthy'
            health['error_count'] = 0
            health['last_error'] = None
        else:
            health['status'] = 'unhealthy'
            health['error_count'] += 1
            health['last_error'] = error
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self._lock:
            stats = {
                'active_backend': self.active_backend.value if self.active_backend else None,
                'total_backends': len(self.backends),
                'healthy_backends': sum(1 for h in self.backend_health.values() if h['status'] == 'healthy'),
                'backend_stats': {},
                'performance_stats': self.performance_stats.copy(),
                'health_stats': self.backend_health.copy(),
                'capacity_gb': round(self.capacity_bytes / (1024 * 1024 * 1024), 2),
                'last_health_check': self.last_health_check.isoformat()
            }
            
            # Get individual backend stats
            for backend_type, backend in self.backends.items():
                try:
                    backend_stats = backend.get_stats()
                    stats['backend_stats'][backend_type.value] = backend_stats
                except Exception as ex:
                    stats['backend_stats'][backend_type.value] = {'error': str(ex)}
            
            return stats
    
    def switch_backend(self, backend_type: BackendType) -> bool:
        """Manually switch to a specific backend."""
        if backend_type not in self.backends:
            self.logger.error(f"[EnhancedVectorMemorySystem] Backend {backend_type.value} not available")
            return False
        
        with self._lock:
            self.active_backend = backend_type
            # Rebuild fallback chain excluding the active backend
            self.fallback_chain = [bt for bt in self.backends.keys() if bt != backend_type]
            
            self.logger.info(f"[EnhancedVectorMemorySystem] Switched to backend: {backend_type.value}")
            return True
    
    def enable_backend(self, backend_type: BackendType) -> bool:
        """Enable a specific backend."""
        if backend_type not in self.backend_configs:
            return False
        
        config = self.backend_configs[backend_type]
        config.enabled = True
        
        # Initialize if not already done
        if backend_type not in self.backends:
            backend = self._create_backend(backend_type, config.config)
            if backend:
                self.backends[backend_type] = backend
                self.performance_stats[backend_type] = {
                    'total_operations': 0,
                    'total_time_ms': 0,
                    'avg_time_ms': 0,
                    'error_count': 0,
                    'last_error': None
                }
                self.backend_health[backend_type] = {
                    'status': 'healthy',
                    'last_check': datetime.now(),
                    'error_count': 0,
                    'last_error': None
                }
                self.logger.info(f"[EnhancedVectorMemorySystem] Enabled backend: {backend_type.value}")
                return True
        
        return False
    
    def disable_backend(self, backend_type: BackendType) -> bool:
        """Disable a specific backend."""
        if backend_type not in self.backend_configs:
            return False
        
        with self._lock:
            self.backend_configs[backend_type].enabled = False
            
            # Remove from active backends
            if backend_type in self.backends:
                del self.backends[backend_type]
            
            if backend_type in self.performance_stats:
                del self.performance_stats[backend_type]
            
            if backend_type in self.backend_health:
                del self.backend_health[backend_type]
            
            # Reselect backend if this was the active one
            if self.active_backend == backend_type:
                self._select_optimal_backend()
            
            self.logger.info(f"[EnhancedVectorMemorySystem] Disabled backend: {backend_type.value}")
            return True
    
    def cleanup(self):
        """Cleanup resources and optimize storage."""
        with self._lock:
            for backend_type, backend in self.backends.items():
                try:
                    # Optimize each backend
                    backend.create_index()
                    self.logger.debug(f"[EnhancedVectorMemorySystem] Cleaned up backend: {backend_type.value}")
                except Exception as ex:
                    self.logger.error(f"[EnhancedVectorMemorySystem] Error cleaning up {backend_type.value}: {ex}")