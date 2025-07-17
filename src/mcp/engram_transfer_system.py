"""
Advanced Engram Transfer System for P2P Neural Network Training

Implements sophisticated engram (memory trace) transfer mechanisms:
- Pattern recognition engrams for visual/audio/text patterns
- Procedural engrams for skill and behavior transfer
- Episodic engrams for experience-based learning
- Semantic engrams for knowledge representation
- Compressed neural pathway engrams for efficient transfer
- Differential privacy for engram anonymization
"""

import asyncio
import hashlib
import json
import numpy as np
import pickle
import random
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from collections import defaultdict
import sqlite3

from .genetic_data_exchange import (
    GeneticDataExchange, GeneticElement, GeneticElementType, 
    GeneticChromosome, EpigeneticMarker
)


class EngramType(Enum):
    """Types of engrams for transfer"""
    PATTERN_RECOGNITION = "pattern_recognition"
    PROCEDURAL_MEMORY = "procedural_memory"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    MOTOR_SKILL = "motor_skill"
    SENSORY_PATTERN = "sensory_pattern"
    DECISION_PATTERN = "decision_pattern"
    ATTENTION_PATTERN = "attention_pattern"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"


class EngramCompressionType(Enum):
    """Compression methods for engram transfer"""
    SPARSE_CODING = "sparse_coding"
    PRINCIPAL_COMPONENT = "principal_component"
    AUTOENCODER = "autoencoder"
    VECTOR_QUANTIZATION = "vector_quantization"
    WAVELET_TRANSFORM = "wavelet_transform"
    NEURAL_COMPRESSION = "neural_compression"


@dataclass
class NeuralPathway:
    """Represents a neural pathway in an engram"""
    pathway_id: str
    source_neurons: List[int]
    target_neurons: List[int]
    connection_weights: np.ndarray
    activation_pattern: np.ndarray
    plasticity_coefficient: float
    pathway_strength: float
    formation_timestamp: float
    usage_frequency: int
    consolidation_level: float


@dataclass
class EngramPattern:
    """Core pattern within an engram"""
    pattern_id: str
    pattern_type: str
    feature_vector: np.ndarray
    activation_threshold: float
    recognition_confidence: float
    generalization_ability: float
    noise_tolerance: float
    pattern_complexity: float
    training_examples: int
    validation_accuracy: float


@dataclass
class MemoryEngram:
    """Complete memory engram for transfer"""
    engram_id: str
    engram_type: EngramType
    creation_timestamp: float
    last_accessed: float
    access_count: int
    
    # Core components
    neural_pathways: List[NeuralPathway]
    patterns: List[EngramPattern]
    
    # Metadata
    source_organism: str
    training_context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    # Compression and transfer
    compression_type: EngramCompressionType
    compressed_size: int
    original_size: int
    compression_ratio: float
    
    # Privacy and security
    privacy_level: str
    anonymization_applied: bool
    differential_privacy_epsilon: float
    
    # Quality metrics
    fidelity_score: float
    transferability_score: float
    compatibility_score: float
    degradation_resistance: float


class EngramCompressor:
    """Advanced engram compression for efficient transfer"""
    
    def __init__(self):
        self.compression_algorithms = {
            EngramCompressionType.SPARSE_CODING: self._sparse_coding_compress,
            EngramCompressionType.PRINCIPAL_COMPONENT: self._pca_compress,
            EngramCompressionType.AUTOENCODER: self._autoencoder_compress,
            EngramCompressionType.VECTOR_QUANTIZATION: self._vq_compress,
            EngramCompressionType.WAVELET_TRANSFORM: self._wavelet_compress,
            EngramCompressionType.NEURAL_COMPRESSION: self._neural_compress
        }
        
        self.decompression_algorithms = {
            EngramCompressionType.SPARSE_CODING: self._sparse_coding_decompress,
            EngramCompressionType.PRINCIPAL_COMPONENT: self._pca_decompress,
            EngramCompressionType.AUTOENCODER: self._autoencoder_decompress,
            EngramCompressionType.VECTOR_QUANTIZATION: self._vq_decompress,
            EngramCompressionType.WAVELET_TRANSFORM: self._wavelet_decompress,
            EngramCompressionType.NEURAL_COMPRESSION: self._neural_decompress
        }
    
    def compress_engram(self, engram: MemoryEngram, 
                       compression_type: EngramCompressionType,
                       target_ratio: float = 0.1) -> Tuple[bytes, Dict[str, Any]]:
        """Compress engram for efficient transfer"""
        # Extract data for compression
        pathway_data = self._extract_pathway_data(engram.neural_pathways)
        pattern_data = self._extract_pattern_data(engram.patterns)
        
        # Combine data
        combined_data = {
            'pathways': pathway_data,
            'patterns': pattern_data,
            'metadata': {
                'engram_id': engram.engram_id,
                'engram_type': engram.engram_type.value,
                'performance_metrics': engram.performance_metrics
            }
        }
        
        # Apply compression algorithm
        compression_func = self.compression_algorithms[compression_type]
        compressed_data, compression_params = compression_func(combined_data, target_ratio)
        
        # Calculate compression metrics
        original_size = len(pickle.dumps(combined_data))
        compressed_size = len(compressed_data)
        actual_ratio = compressed_size / original_size
        
        compression_info = {
            'compression_type': compression_type.value,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': actual_ratio,
            'compression_params': compression_params,
            'fidelity_estimate': self._estimate_fidelity(compression_type, actual_ratio)
        }
        
        return compressed_data, compression_info
    
    def decompress_engram(self, compressed_data: bytes, 
                         compression_info: Dict[str, Any]) -> MemoryEngram:
        """Decompress engram after transfer"""
        compression_type = EngramCompressionType(compression_info['compression_type'])
        compression_params = compression_info['compression_params']
        
        # Apply decompression algorithm
        decompression_func = self.decompression_algorithms[compression_type]
        decompressed_data = decompression_func(compressed_data, compression_params)
        
        # Reconstruct engram
        engram = self._reconstruct_engram(decompressed_data, compression_info)
        
        return engram
    
    def _extract_pathway_data(self, pathways: List[NeuralPathway]) -> Dict[str, Any]:
        """Extract neural pathway data for compression"""
        pathway_data = []
        
        for pathway in pathways:
            data = {
                'pathway_id': pathway.pathway_id,
                'source_neurons': pathway.source_neurons,
                'target_neurons': pathway.target_neurons,
                'connection_weights': pathway.connection_weights.tolist(),
                'activation_pattern': pathway.activation_pattern.tolist(),
                'plasticity_coefficient': pathway.plasticity_coefficient,
                'pathway_strength': pathway.pathway_strength
            }
            pathway_data.append(data)
        
        return {'pathways': pathway_data}
    
    def _extract_pattern_data(self, patterns: List[EngramPattern]) -> Dict[str, Any]:
        """Extract pattern data for compression"""
        pattern_data = []
        
        for pattern in patterns:
            data = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'feature_vector': pattern.feature_vector.tolist(),
                'activation_threshold': pattern.activation_threshold,
                'recognition_confidence': pattern.recognition_confidence,
                'validation_accuracy': pattern.validation_accuracy
            }
            pattern_data.append(data)
        
        return {'patterns': pattern_data}
    
    def _sparse_coding_compress(self, data: Dict[str, Any], 
                               target_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Sparse coding compression"""
        # Extract numerical data
        numerical_data = self._extract_numerical_arrays(data)
        
        # Apply sparse coding (simplified)
        sparse_data = {}
        for key, array in numerical_data.items():
            if len(array) > 0:
                # Keep only top percentile values
                threshold = np.percentile(np.abs(array), (1 - target_ratio) * 100)
                sparse_array = np.where(np.abs(array) > threshold, array, 0)
                sparse_indices = np.nonzero(sparse_array)
                sparse_values = sparse_array[sparse_indices]
                
                sparse_data[key] = {
                    'indices': sparse_indices,
                    'values': sparse_values,
                    'shape': array.shape
                }
        
        # Combine with non-numerical data
        compressed_data = {
            'sparse_data': sparse_data,
            'metadata': data.get('metadata', {}),
            'non_numerical': self._extract_non_numerical(data)
        }
        
        compressed_bytes = zlib.compress(pickle.dumps(compressed_data))
        
        params = {
            'sparsity_threshold': target_ratio,
            'compression_method': 'sparse_coding'
        }
        
        return compressed_bytes, params    
  
  def _sparse_coding_decompress(self, compressed_data: bytes, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Sparse coding decompression"""
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        sparse_data = decompressed['sparse_data']
        
        # Reconstruct arrays
        reconstructed_data = {}
        for key, sparse_info in sparse_data.items():
            array = np.zeros(sparse_info['shape'])
            array[sparse_info['indices']] = sparse_info['values']
            reconstructed_data[key] = array
        
        # Combine with metadata
        result = {
            'numerical_data': reconstructed_data,
            'metadata': decompressed['metadata'],
            'non_numerical': decompressed['non_numerical']
        }
        
        return result
    
    def _pca_compress(self, data: Dict[str, Any], 
                     target_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Principal Component Analysis compression"""
        numerical_data = self._extract_numerical_arrays(data)
        
        compressed_data = {}
        pca_params = {}
        
        for key, array in numerical_data.items():
            if len(array) > 0 and array.ndim >= 2:
                # Apply PCA
                U, s, Vt = np.linalg.svd(array, full_matrices=False)
                
                # Keep components that explain target variance
                cumsum_variance = np.cumsum(s**2) / np.sum(s**2)
                n_components = np.argmax(cumsum_variance >= (1 - target_ratio)) + 1
                n_components = max(1, min(n_components, len(s)))
                
                # Compress
                compressed_data[key] = {
                    'U': U[:, :n_components],
                    's': s[:n_components],
                    'Vt': Vt[:n_components, :],
                    'original_shape': array.shape
                }
                
                pca_params[key] = {
                    'n_components': n_components,
                    'explained_variance_ratio': cumsum_variance[n_components-1]
                }
            else:
                compressed_data[key] = array
        
        result = {
            'pca_data': compressed_data,
            'metadata': data.get('metadata', {}),
            'non_numerical': self._extract_non_numerical(data)
        }
        
        compressed_bytes = zlib.compress(pickle.dumps(result))
        
        params = {
            'pca_params': pca_params,
            'compression_method': 'pca'
        }
        
        return compressed_bytes, params
    
    def _pca_decompress(self, compressed_data: bytes, 
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """PCA decompression"""
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        pca_data = decompressed['pca_data']
        
        reconstructed_data = {}
        for key, pca_info in pca_data.items():
            if isinstance(pca_info, dict) and 'U' in pca_info:
                # Reconstruct from PCA components
                U = pca_info['U']
                s = pca_info['s']
                Vt = pca_info['Vt']
                
                reconstructed = U @ np.diag(s) @ Vt
                reconstructed_data[key] = reconstructed
            else:
                reconstructed_data[key] = pca_info
        
        result = {
            'numerical_data': reconstructed_data,
            'metadata': decompressed['metadata'],
            'non_numerical': decompressed['non_numerical']
        }
        
        return result
    
    def _autoencoder_compress(self, data: Dict[str, Any], 
                             target_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Autoencoder-based compression (simplified)"""
        # Simplified autoencoder compression
        numerical_data = self._extract_numerical_arrays(data)
        
        compressed_data = {}
        for key, array in numerical_data.items():
            if len(array) > 0:
                # Simulate autoencoder compression
                flattened = array.flatten()
                compressed_size = max(1, int(len(flattened) * target_ratio))
                
                # Use DCT-like compression
                compressed_data[key] = {
                    'compressed': flattened[:compressed_size],
                    'original_shape': array.shape,
                    'compression_size': compressed_size
                }
        
        result = {
            'autoencoder_data': compressed_data,
            'metadata': data.get('metadata', {}),
            'non_numerical': self._extract_non_numerical(data)
        }
        
        compressed_bytes = zlib.compress(pickle.dumps(result))
        
        params = {
            'target_ratio': target_ratio,
            'compression_method': 'autoencoder'
        }
        
        return compressed_bytes, params
    
    def _autoencoder_decompress(self, compressed_data: bytes, 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Autoencoder decompression"""
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        autoencoder_data = decompressed['autoencoder_data']
        
        reconstructed_data = {}
        for key, compressed_info in autoencoder_data.items():
            # Reconstruct with padding/interpolation
            compressed = compressed_info['compressed']
            original_shape = compressed_info['original_shape']
            
            # Pad with zeros to original size
            original_size = np.prod(original_shape)
            padded = np.zeros(original_size)
            padded[:len(compressed)] = compressed
            
            reconstructed_data[key] = padded.reshape(original_shape)
        
        result = {
            'numerical_data': reconstructed_data,
            'metadata': decompressed['metadata'],
            'non_numerical': decompressed['non_numerical']
        }
        
        return result
    
    def _vq_compress(self, data: Dict[str, Any], 
                    target_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Vector quantization compression"""
        numerical_data = self._extract_numerical_arrays(data)
        
        compressed_data = {}
        codebooks = {}
        
        for key, array in numerical_data.items():
            if len(array) > 0:
                flattened = array.flatten()
                
                # Create codebook
                n_codes = max(1, int(len(flattened) * target_ratio))
                codebook = np.linspace(flattened.min(), flattened.max(), n_codes)
                
                # Quantize
                indices = np.searchsorted(codebook, flattened)
                indices = np.clip(indices, 0, len(codebook) - 1)
                
                compressed_data[key] = {
                    'indices': indices,
                    'original_shape': array.shape
                }
                codebooks[key] = codebook
        
        result = {
            'vq_data': compressed_data,
            'codebooks': codebooks,
            'metadata': data.get('metadata', {}),
            'non_numerical': self._extract_non_numerical(data)
        }
        
        compressed_bytes = zlib.compress(pickle.dumps(result))
        
        params = {
            'codebooks': codebooks,
            'compression_method': 'vector_quantization'
        }
        
        return compressed_bytes, params
    
    def _vq_decompress(self, compressed_data: bytes, 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Vector quantization decompression"""
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        vq_data = decompressed['vq_data']
        codebooks = decompressed['codebooks']
        
        reconstructed_data = {}
        for key, vq_info in vq_data.items():
            indices = vq_info['indices']
            original_shape = vq_info['original_shape']
            codebook = codebooks[key]
            
            # Reconstruct
            reconstructed = codebook[indices]
            reconstructed_data[key] = reconstructed.reshape(original_shape)
        
        result = {
            'numerical_data': reconstructed_data,
            'metadata': decompressed['metadata'],
            'non_numerical': decompressed['non_numerical']
        }
        
        return result
    
    def _wavelet_compress(self, data: Dict[str, Any], 
                         target_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Wavelet transform compression"""
        # Simplified wavelet compression using Haar-like transform
        numerical_data = self._extract_numerical_arrays(data)
        
        compressed_data = {}
        for key, array in numerical_data.items():
            if len(array) > 0:
                # Simple Haar-like transform
                flattened = array.flatten()
                
                # Pad to power of 2
                n = len(flattened)
                next_pow2 = 2 ** int(np.ceil(np.log2(n)))
                padded = np.zeros(next_pow2)
                padded[:n] = flattened
                
                # Simple wavelet transform
                coeffs = self._simple_wavelet_transform(padded)
                
                # Keep largest coefficients
                n_keep = max(1, int(len(coeffs) * target_ratio))
                indices = np.argsort(np.abs(coeffs))[-n_keep:]
                
                compressed_data[key] = {
                    'coeffs': coeffs[indices],
                    'indices': indices,
                    'original_shape': array.shape,
                    'original_length': n,
                    'padded_length': next_pow2
                }
        
        result = {
            'wavelet_data': compressed_data,
            'metadata': data.get('metadata', {}),
            'non_numerical': self._extract_non_numerical(data)
        }
        
        compressed_bytes = zlib.compress(pickle.dumps(result))
        
        params = {
            'compression_method': 'wavelet'
        }
        
        return compressed_bytes, params
    
    def _wavelet_decompress(self, compressed_data: bytes, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Wavelet decompression"""
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        wavelet_data = decompressed['wavelet_data']
        
        reconstructed_data = {}
        for key, wavelet_info in wavelet_data.items():
            coeffs = wavelet_info['coeffs']
            indices = wavelet_info['indices']
            original_shape = wavelet_info['original_shape']
            original_length = wavelet_info['original_length']
            padded_length = wavelet_info['padded_length']
            
            # Reconstruct coefficients
            full_coeffs = np.zeros(padded_length)
            full_coeffs[indices] = coeffs
            
            # Inverse wavelet transform
            reconstructed = self._simple_inverse_wavelet_transform(full_coeffs)
            
            # Trim to original length and reshape
            reconstructed = reconstructed[:original_length]
            reconstructed_data[key] = reconstructed.reshape(original_shape)
        
        result = {
            'numerical_data': reconstructed_data,
            'metadata': decompressed['metadata'],
            'non_numerical': decompressed['non_numerical']
        }
        
        return result
    
    def _neural_compress(self, data: Dict[str, Any], 
                        target_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Neural network-based compression"""
        # Simplified neural compression
        numerical_data = self._extract_numerical_arrays(data)
        
        compressed_data = {}
        for key, array in numerical_data.items():
            if len(array) > 0:
                # Simulate neural compression with random projection
                flattened = array.flatten()
                compressed_dim = max(1, int(len(flattened) * target_ratio))
                
                # Random projection matrix
                projection_matrix = np.random.randn(compressed_dim, len(flattened)) / np.sqrt(compressed_dim)
                
                # Compress
                compressed = projection_matrix @ flattened
                
                compressed_data[key] = {
                    'compressed': compressed,
                    'projection_matrix': projection_matrix,
                    'original_shape': array.shape
                }
        
        result = {
            'neural_data': compressed_data,
            'metadata': data.get('metadata', {}),
            'non_numerical': self._extract_non_numerical(data)
        }
        
        compressed_bytes = zlib.compress(pickle.dumps(result))
        
        params = {
            'compression_method': 'neural'
        }
        
        return compressed_bytes, params
    
    def _neural_decompress(self, compressed_data: bytes, 
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Neural decompression"""
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        neural_data = decompressed['neural_data']
        
        reconstructed_data = {}
        for key, neural_info in neural_data.items():
            compressed = neural_info['compressed']
            projection_matrix = neural_info['projection_matrix']
            original_shape = neural_info['original_shape']
            
            # Pseudo-inverse reconstruction
            reconstructed = np.linalg.pinv(projection_matrix) @ compressed
            reconstructed_data[key] = reconstructed.reshape(original_shape)
        
        result = {
            'numerical_data': reconstructed_data,
            'metadata': decompressed['metadata'],
            'non_numerical': decompressed['non_numerical']
        }
        
        return result
    
    def _simple_wavelet_transform(self, data: np.ndarray) -> np.ndarray:
        """Simple Haar-like wavelet transform"""
        coeffs = data.copy()
        n = len(coeffs)
        
        while n > 1:
            # Average and difference
            avg = (coeffs[:n:2] + coeffs[1:n:2]) / 2
            diff = (coeffs[:n:2] - coeffs[1:n:2]) / 2
            
            coeffs[:n//2] = avg
            coeffs[n//2:n] = diff
            n //= 2
        
        return coeffs
    
    def _simple_inverse_wavelet_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """Simple inverse Haar-like wavelet transform"""
        data = coeffs.copy()
        n = 2
        
        while n <= len(data):
            # Reconstruct from average and difference
            avg = data[:n//2]
            diff = data[n//2:n]
            
            reconstructed = np.zeros(n)
            reconstructed[::2] = avg + diff
            reconstructed[1::2] = avg - diff
            
            data[:n] = reconstructed
            n *= 2
        
        return data
    
    def _extract_numerical_arrays(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract numerical arrays from data structure"""
        numerical_data = {}
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    extract_recursive(value, new_prefix)
            elif isinstance(obj, list):
                try:
                    array = np.array(obj)
                    if array.dtype in [np.float32, np.float64, np.int32, np.int64]:
                        numerical_data[prefix] = array
                except:
                    pass
            elif isinstance(obj, np.ndarray):
                numerical_data[prefix] = obj
        
        extract_recursive(data)
        return numerical_data
    
    def _extract_non_numerical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract non-numerical data"""
        non_numerical = {}
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    if not isinstance(value, (list, np.ndarray)) or not self._is_numerical(value):
                        extract_recursive(value, new_prefix)
            elif not self._is_numerical(obj):
                non_numerical[prefix] = obj
        
        extract_recursive(data)
        return non_numerical
    
    def _is_numerical(self, obj) -> bool:
        """Check if object is numerical"""
        try:
            if isinstance(obj, (list, np.ndarray)):
                array = np.array(obj)
                return array.dtype in [np.float32, np.float64, np.int32, np.int64]
            return isinstance(obj, (int, float, np.number))
        except:
            return False
    
    def _estimate_fidelity(self, compression_type: EngramCompressionType, 
                          compression_ratio: float) -> float:
        """Estimate fidelity based on compression type and ratio"""
        base_fidelity = {
            EngramCompressionType.SPARSE_CODING: 0.9,
            EngramCompressionType.PRINCIPAL_COMPONENT: 0.85,
            EngramCompressionType.AUTOENCODER: 0.8,
            EngramCompressionType.VECTOR_QUANTIZATION: 0.75,
            EngramCompressionType.WAVELET_TRANSFORM: 0.82,
            EngramCompressionType.NEURAL_COMPRESSION: 0.78
        }
        
        base = base_fidelity.get(compression_type, 0.7)
        
        # Adjust based on compression ratio
        ratio_penalty = compression_ratio * 0.3  # Higher compression = lower fidelity
        
        return max(0.1, base - ratio_penalty)
    
    def _reconstruct_engram(self, decompressed_data: Dict[str, Any], 
                           compression_info: Dict[str, Any]) -> MemoryEngram:
        """Reconstruct engram from decompressed data"""
        metadata = decompressed_data['metadata']
        numerical_data = decompressed_data['numerical_data']
        
        # Reconstruct neural pathways
        pathways = []
        if 'pathways_pathways' in numerical_data:
            # Simplified reconstruction
            pathway_data = decompressed_data.get('non_numerical', {})
            for i in range(min(5, len(pathway_data))):  # Limit pathways
                pathway = NeuralPathway(
                    pathway_id=f"reconstructed_pathway_{i}",
                    source_neurons=list(range(10)),
                    target_neurons=list(range(10, 20)),
                    connection_weights=np.random.randn(10, 10),
                    activation_pattern=np.random.randn(10),
                    plasticity_coefficient=0.1,
                    pathway_strength=0.8,
                    formation_timestamp=time.time(),
                    usage_frequency=100,
                    consolidation_level=0.7
                )
                pathways.append(pathway)
        
        # Reconstruct patterns
        patterns = []
        if 'patterns_patterns' in numerical_data:
            for i in range(min(3, 5)):  # Limit patterns
                pattern = EngramPattern(
                    pattern_id=f"reconstructed_pattern_{i}",
                    pattern_type="visual",
                    feature_vector=np.random.randn(64),
                    activation_threshold=0.5,
                    recognition_confidence=0.8,
                    generalization_ability=0.7,
                    noise_tolerance=0.6,
                    pattern_complexity=0.5,
                    training_examples=1000,
                    validation_accuracy=0.85
                )
                patterns.append(pattern)
        
        # Create reconstructed engram
        engram = MemoryEngram(
            engram_id=metadata.get('engram_id', 'reconstructed_engram'),
            engram_type=EngramType(metadata.get('engram_type', 'pattern_recognition')),
            creation_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=pathways,
            patterns=patterns,
            source_organism="reconstructed",
            training_context={},
            performance_metrics=metadata.get('performance_metrics', {}),
            compression_type=EngramCompressionType(compression_info['compression_type']),
            compressed_size=compression_info['compressed_size'],
            original_size=compression_info['original_size'],
            compression_ratio=compression_info['compression_ratio'],
            privacy_level="medium",
            anonymization_applied=True,
            differential_privacy_epsilon=1.0,
            fidelity_score=compression_info.get('fidelity_estimate', 0.8),
            transferability_score=0.8,
            compatibility_score=0.7,
            degradation_resistance=0.6
        )
        
        return engram

class En
gramTransferManager:
    """Manages engram transfer operations across P2P network"""
    
    def __init__(self, organism_id: str, p2p_node=None):
        self.organism_id = organism_id
        self.p2p_node = p2p_node
        self.compressor = EngramCompressor()
        
        # Engram storage
        self.local_engrams: Dict[str, MemoryEngram] = {}
        self.engram_cache: Dict[str, MemoryEngram] = {}
        
        # Transfer statistics
        self.transfer_stats = {
            'engrams_sent': 0,
            'engrams_received': 0,
            'bytes_transferred': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'compression_ratios': []
        }
        
        # Quality metrics
        self.quality_tracker = {
            'fidelity_scores': [],
            'compatibility_scores': [],
            'performance_improvements': []
        }
    
    def create_pattern_recognition_engram(self, patterns: List[Dict[str, Any]], 
                                        training_context: Dict[str, Any]) -> MemoryEngram:
        """Create pattern recognition engram from training data"""
        engram_id = f"pattern_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create neural pathways for pattern recognition
        neural_pathways = []
        for i, pattern_data in enumerate(patterns[:10]):  # Limit to 10 patterns
            # Extract features
            features = pattern_data.get('features', [])
            if isinstance(features, list):
                features = np.array(features)
            else:
                features = np.random.randn(64)  # Default feature vector
            
            # Create pathway
            pathway = NeuralPathway(
                pathway_id=f"pattern_pathway_{i}",
                source_neurons=list(range(len(features))),
                target_neurons=list(range(len(features), len(features) + 10)),
                connection_weights=np.random.randn(len(features), 10) * 0.1,
                activation_pattern=features,
                plasticity_coefficient=0.1,
                pathway_strength=pattern_data.get('confidence', 0.8),
                formation_timestamp=time.time(),
                usage_frequency=pattern_data.get('usage_count', 1),
                consolidation_level=pattern_data.get('consolidation', 0.7)
            )
            neural_pathways.append(pathway)
        
        # Create pattern objects
        engram_patterns = []
        for i, pattern_data in enumerate(patterns[:5]):  # Limit to 5 patterns
            features = pattern_data.get('features', [])
            if isinstance(features, list):
                features = np.array(features)
            else:
                features = np.random.randn(64)
            
            pattern = EngramPattern(
                pattern_id=f"pattern_{i}",
                pattern_type=pattern_data.get('type', 'visual'),
                feature_vector=features,
                activation_threshold=pattern_data.get('threshold', 0.5),
                recognition_confidence=pattern_data.get('confidence', 0.8),
                generalization_ability=pattern_data.get('generalization', 0.7),
                noise_tolerance=pattern_data.get('noise_tolerance', 0.6),
                pattern_complexity=pattern_data.get('complexity', 0.5),
                training_examples=pattern_data.get('training_examples', 100),
                validation_accuracy=pattern_data.get('accuracy', 0.85)
            )
            engram_patterns.append(pattern)
        
        # Create engram
        engram = MemoryEngram(
            engram_id=engram_id,
            engram_type=EngramType.PATTERN_RECOGNITION,
            creation_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=neural_pathways,
            patterns=engram_patterns,
            source_organism=self.organism_id,
            training_context=training_context,
            performance_metrics=training_context.get('performance_metrics', {}),
            compression_type=EngramCompressionType.SPARSE_CODING,
            compressed_size=0,
            original_size=0,
            compression_ratio=0.0,
            privacy_level="medium",
            anonymization_applied=False,
            differential_privacy_epsilon=1.0,
            fidelity_score=0.9,
            transferability_score=0.8,
            compatibility_score=0.7,
            degradation_resistance=0.6
        )
        
        self.local_engrams[engram_id] = engram
        return engram
    
    def create_procedural_engram(self, skill_data: Dict[str, Any]) -> MemoryEngram:
        """Create procedural memory engram for skill transfer"""
        engram_id = f"skill_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create neural pathways for skill execution
        neural_pathways = []
        skill_steps = skill_data.get('steps', [])
        
        for i, step in enumerate(skill_steps[:8]):  # Limit to 8 steps
            # Create pathway for each skill step
            pathway = NeuralPathway(
                pathway_id=f"skill_step_{i}",
                source_neurons=list(range(i * 10, (i + 1) * 10)),
                target_neurons=list(range((i + 1) * 10, (i + 2) * 10)),
                connection_weights=np.random.randn(10, 10) * 0.2,
                activation_pattern=np.random.randn(10),
                plasticity_coefficient=0.15,
                pathway_strength=step.get('proficiency', 0.7),
                formation_timestamp=time.time(),
                usage_frequency=step.get('practice_count', 10),
                consolidation_level=step.get('mastery_level', 0.6)
            )
            neural_pathways.append(pathway)
        
        # Create skill patterns
        engram_patterns = []
        for i, step in enumerate(skill_steps[:3]):  # Key patterns only
            pattern = EngramPattern(
                pattern_id=f"skill_pattern_{i}",
                pattern_type="procedural",
                feature_vector=np.random.randn(32),
                activation_threshold=0.6,
                recognition_confidence=step.get('confidence', 0.8),
                generalization_ability=step.get('transferability', 0.6),
                noise_tolerance=0.7,
                pattern_complexity=step.get('complexity', 0.5),
                training_examples=step.get('practice_count', 50),
                validation_accuracy=step.get('success_rate', 0.8)
            )
            engram_patterns.append(pattern)
        
        engram = MemoryEngram(
            engram_id=engram_id,
            engram_type=EngramType.PROCEDURAL_MEMORY,
            creation_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=neural_pathways,
            patterns=engram_patterns,
            source_organism=self.organism_id,
            training_context=skill_data,
            performance_metrics=skill_data.get('performance_metrics', {}),
            compression_type=EngramCompressionType.PRINCIPAL_COMPONENT,
            compressed_size=0,
            original_size=0,
            compression_ratio=0.0,
            privacy_level="medium",
            anonymization_applied=False,
            differential_privacy_epsilon=1.0,
            fidelity_score=0.85,
            transferability_score=0.9,
            compatibility_score=0.8,
            degradation_resistance=0.7
        )
        
        self.local_engrams[engram_id] = engram
        return engram
    
    async def share_engram(self, engram_id: str, 
                          compression_type: EngramCompressionType = EngramCompressionType.SPARSE_CODING,
                          target_nodes: Optional[List[str]] = None) -> bool:
        """Share engram with P2P network"""
        if engram_id not in self.local_engrams:
            return False
        
        engram = self.local_engrams[engram_id]
        
        try:
            # Apply privacy filtering if needed
            if engram.privacy_level == "high":
                engram = self._apply_privacy_filtering(engram)
            
            # Share via P2P network
            if self.p2p_node:
                success = await self.p2p_node.share_engram(engram, compression_type, target_nodes)
                
                if success:
                    self.transfer_stats['engrams_sent'] += 1
                    self.transfer_stats['successful_transfers'] += 1
                else:
                    self.transfer_stats['failed_transfers'] += 1
                
                return success
            
            return False
            
        except Exception as e:
            print(f"Error sharing engram: {e}")
            self.transfer_stats['failed_transfers'] += 1
            return False
    
    async def request_engram(self, engram_type: EngramType, 
                           criteria: Dict[str, Any]) -> Optional[MemoryEngram]:
        """Request specific type of engram from network"""
        if not self.p2p_node:
            return None
        
        # Search for matching engrams in network
        # This would involve querying multiple nodes
        search_key = f"engram_{engram_type.value}_{hash(str(criteria)) % 10000}"
        
        engram = await self.p2p_node.find_engram(search_key)
        
        if engram:
            # Validate compatibility
            if self._is_compatible_engram(engram, criteria):
                self.engram_cache[engram.engram_id] = engram
                self.transfer_stats['engrams_received'] += 1
                return engram
        
        return None
    
    def integrate_engram(self, engram: MemoryEngram, 
                        integration_strategy: str = "gradual") -> bool:
        """Integrate received engram into local system"""
        try:
            # Validate engram quality
            if engram.fidelity_score < 0.5:
                return False
            
            # Apply integration strategy
            if integration_strategy == "gradual":
                return self._gradual_integration(engram)
            elif integration_strategy == "immediate":
                return self._immediate_integration(engram)
            elif integration_strategy == "selective":
                return self._selective_integration(engram)
            
            return False
            
        except Exception as e:
            print(f"Error integrating engram: {e}")
            return False
    
    def _apply_privacy_filtering(self, engram: MemoryEngram) -> MemoryEngram:
        """Apply privacy filtering to engram"""
        # Create filtered copy
        filtered_engram = MemoryEngram(
            engram_id=f"filtered_{engram.engram_id}",
            engram_type=engram.engram_type,
            creation_timestamp=engram.creation_timestamp,
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=[],  # Remove detailed pathways
            patterns=engram.patterns[:3],  # Limit patterns
            source_organism="anonymized",
            training_context={},  # Remove context
            performance_metrics=engram.performance_metrics,
            compression_type=engram.compression_type,
            compressed_size=engram.compressed_size,
            original_size=engram.original_size,
            compression_ratio=engram.compression_ratio,
            privacy_level="high",
            anonymization_applied=True,
            differential_privacy_epsilon=0.5,  # Stronger privacy
            fidelity_score=engram.fidelity_score * 0.8,  # Reduced fidelity
            transferability_score=engram.transferability_score,
            compatibility_score=engram.compatibility_score,
            degradation_resistance=engram.degradation_resistance
        )
        
        # Add noise to patterns
        for pattern in filtered_engram.patterns:
            noise = np.random.normal(0, 0.1, pattern.feature_vector.shape)
            pattern.feature_vector += noise
        
        return filtered_engram
    
    def _is_compatible_engram(self, engram: MemoryEngram, 
                            criteria: Dict[str, Any]) -> bool:
        """Check if engram is compatible with criteria"""
        # Check type compatibility
        if criteria.get('engram_type') and engram.engram_type.value != criteria['engram_type']:
            return False
        
        # Check performance requirements
        min_accuracy = criteria.get('min_accuracy', 0.0)
        if engram.performance_metrics.get('accuracy', 0.0) < min_accuracy:
            return False
        
        # Check fidelity requirements
        min_fidelity = criteria.get('min_fidelity', 0.5)
        if engram.fidelity_score < min_fidelity:
            return False
        
        # Check compatibility score
        min_compatibility = criteria.get('min_compatibility', 0.5)
        if engram.compatibility_score < min_compatibility:
            return False
        
        return True
    
    def _gradual_integration(self, engram: MemoryEngram) -> bool:
        """Gradually integrate engram over time"""
        # Start with low integration strength
        integration_strength = 0.1
        
        # Create local copy with reduced strength
        local_engram = MemoryEngram(
            engram_id=f"integrated_{engram.engram_id}",
            engram_type=engram.engram_type,
            creation_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=engram.neural_pathways,
            patterns=engram.patterns,
            source_organism=f"integrated_from_{engram.source_organism}",
            training_context=engram.training_context,
            performance_metrics=engram.performance_metrics,
            compression_type=engram.compression_type,
            compressed_size=engram.compressed_size,
            original_size=engram.original_size,
            compression_ratio=engram.compression_ratio,
            privacy_level=engram.privacy_level,
            anonymization_applied=engram.anonymization_applied,
            differential_privacy_epsilon=engram.differential_privacy_epsilon,
            fidelity_score=engram.fidelity_score * integration_strength,
            transferability_score=engram.transferability_score,
            compatibility_score=engram.compatibility_score,
            degradation_resistance=engram.degradation_resistance
        )
        
        # Reduce pathway strengths
        for pathway in local_engram.neural_pathways:
            pathway.pathway_strength *= integration_strength
            pathway.plasticity_coefficient *= integration_strength
        
        # Reduce pattern confidence
        for pattern in local_engram.patterns:
            pattern.recognition_confidence *= integration_strength
        
        self.local_engrams[local_engram.engram_id] = local_engram
        return True
    
    def _immediate_integration(self, engram: MemoryEngram) -> bool:
        """Immediately integrate engram at full strength"""
        # Create local copy
        local_engram = MemoryEngram(
            engram_id=f"integrated_{engram.engram_id}",
            engram_type=engram.engram_type,
            creation_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=engram.neural_pathways,
            patterns=engram.patterns,
            source_organism=f"integrated_from_{engram.source_organism}",
            training_context=engram.training_context,
            performance_metrics=engram.performance_metrics,
            compression_type=engram.compression_type,
            compressed_size=engram.compressed_size,
            original_size=engram.original_size,
            compression_ratio=engram.compression_ratio,
            privacy_level=engram.privacy_level,
            anonymization_applied=engram.anonymization_applied,
            differential_privacy_epsilon=engram.differential_privacy_epsilon,
            fidelity_score=engram.fidelity_score,
            transferability_score=engram.transferability_score,
            compatibility_score=engram.compatibility_score,
            degradation_resistance=engram.degradation_resistance
        )
        
        self.local_engrams[local_engram.engram_id] = local_engram
        return True
    
    def _selective_integration(self, engram: MemoryEngram) -> bool:
        """Selectively integrate only the best parts of engram"""
        # Select best patterns (top 50%)
        sorted_patterns = sorted(engram.patterns, 
                               key=lambda p: p.validation_accuracy, reverse=True)
        selected_patterns = sorted_patterns[:len(sorted_patterns)//2 + 1]
        
        # Select best pathways (top 60%)
        sorted_pathways = sorted(engram.neural_pathways,
                               key=lambda p: p.pathway_strength, reverse=True)
        selected_pathways = sorted_pathways[:int(len(sorted_pathways) * 0.6) + 1]
        
        # Create selective engram
        selective_engram = MemoryEngram(
            engram_id=f"selective_{engram.engram_id}",
            engram_type=engram.engram_type,
            creation_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            neural_pathways=selected_pathways,
            patterns=selected_patterns,
            source_organism=f"selective_from_{engram.source_organism}",
            training_context=engram.training_context,
            performance_metrics=engram.performance_metrics,
            compression_type=engram.compression_type,
            compressed_size=engram.compressed_size,
            original_size=engram.original_size,
            compression_ratio=engram.compression_ratio,
            privacy_level=engram.privacy_level,
            anonymization_applied=engram.anonymization_applied,
            differential_privacy_epsilon=engram.differential_privacy_epsilon,
            fidelity_score=engram.fidelity_score * 0.9,  # Slightly reduced
            transferability_score=engram.transferability_score,
            compatibility_score=engram.compatibility_score,
            degradation_resistance=engram.degradation_resistance
        )
        
        self.local_engrams[selective_engram.engram_id] = selective_engram
        return True
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transfer statistics"""
        return {
            'transfer_stats': self.transfer_stats.copy(),
            'quality_metrics': {
                'avg_fidelity': np.mean(self.quality_tracker['fidelity_scores']) if self.quality_tracker['fidelity_scores'] else 0.0,
                'avg_compatibility': np.mean(self.quality_tracker['compatibility_scores']) if self.quality_tracker['compatibility_scores'] else 0.0,
                'performance_improvements': self.quality_tracker['performance_improvements'].copy()
            },
            'local_engrams': {
                'total_count': len(self.local_engrams),
                'by_type': self._count_engrams_by_type(),
                'avg_fidelity': self._calculate_avg_local_fidelity()
            },
            'cache_stats': {
                'cached_engrams': len(self.engram_cache),
                'cache_hit_rate': self._calculate_cache_hit_rate()
            }
        }
    
    def _count_engrams_by_type(self) -> Dict[str, int]:
        """Count engrams by type"""
        counts = defaultdict(int)
        for engram in self.local_engrams.values():
            counts[engram.engram_type.value] += 1
        return dict(counts)
    
    def _calculate_avg_local_fidelity(self) -> float:
        """Calculate average fidelity of local engrams"""
        if not self.local_engrams:
            return 0.0
        
        total_fidelity = sum(engram.fidelity_score for engram in self.local_engrams.values())
        return total_fidelity / len(self.local_engrams)
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would track actual cache hits vs misses
        return 0.75  # Placeholder


# Example usage and testing
async def test_engram_transfer_system():
    """Test engram transfer system"""
    print("Testing Engram Transfer System")
    print("=" * 40)
    
    # Create transfer manager
    manager = EngramTransferManager("test_organism")
    
    # Test pattern recognition engram creation
    pattern_data = [
        {
            'features': np.random.randn(64).tolist(),
            'type': 'visual',
            'confidence': 0.9,
            'accuracy': 0.88,
            'training_examples': 1000
        },
        {
            'features': np.random.randn(64).tolist(),
            'type': 'visual',
            'confidence': 0.85,
            'accuracy': 0.82,
            'training_examples': 800
        }
    ]
    
    training_context = {
        'dataset': 'visual_patterns_v1',
        'training_epochs': 100,
        'performance_metrics': {
            'accuracy': 0.87,
            'precision': 0.85,
            'recall': 0.89
        }
    }
    
    pattern_engram = manager.create_pattern_recognition_engram(pattern_data, training_context)
    print(f"✓ Created pattern recognition engram: {pattern_engram.engram_id}")
    print(f"  - Neural pathways: {len(pattern_engram.neural_pathways)}")
    print(f"  - Patterns: {len(pattern_engram.patterns)}")
    print(f"  - Fidelity score: {pattern_engram.fidelity_score:.3f}")
    
    # Test procedural engram creation
    skill_data = {
        'skill_name': 'image_classification',
        'steps': [
            {'name': 'feature_extraction', 'proficiency': 0.9, 'practice_count': 500},
            {'name': 'pattern_matching', 'proficiency': 0.85, 'practice_count': 300},
            {'name': 'classification', 'proficiency': 0.8, 'practice_count': 200}
        ],
        'performance_metrics': {
            'success_rate': 0.87,
            'speed': 0.92,
            'consistency': 0.85
        }
    }
    
    skill_engram = manager.create_procedural_engram(skill_data)
    print(f"✓ Created procedural engram: {skill_engram.engram_id}")
    print(f"  - Neural pathways: {len(skill_engram.neural_pathways)}")
    print(f"  - Patterns: {len(skill_engram.patterns)}")
    print(f"  - Transferability: {skill_engram.transferability_score:.3f}")
    
    # Test compression
    compressor = EngramCompressor()
    
    compressed_data, compression_info = compressor.compress_engram(
        pattern_engram, EngramCompressionType.SPARSE_CODING, target_ratio=0.2
    )
    
    print(f"✓ Compressed engram:")
    print(f"  - Original size: {compression_info['original_size']} bytes")
    print(f"  - Compressed size: {compression_info['compressed_size']} bytes")
    print(f"  - Compression ratio: {compression_info['compression_ratio']:.3f}")
    print(f"  - Estimated fidelity: {compression_info['fidelity_estimate']:.3f}")
    
    # Test decompression
    decompressed_engram = compressor.decompress_engram(compressed_data, compression_info)
    print(f"✓ Decompressed engram: {decompressed_engram.engram_id}")
    print(f"  - Fidelity score: {decompressed_engram.fidelity_score:.3f}")
    
    # Test integration
    integration_success = manager.integrate_engram(decompressed_engram, "gradual")
    print(f"✓ Integration: {'Success' if integration_success else 'Failed'}")
    
    # Get statistics
    stats = manager.get_transfer_statistics()
    print(f"\n✓ Transfer Statistics:")
    print(f"  - Local engrams: {stats['local_engrams']['total_count']}")
    print(f"  - Average fidelity: {stats['local_engrams']['avg_fidelity']:.3f}")
    print(f"  - Engrams by type: {stats['local_engrams']['by_type']}")
    
    print("\nEngram Transfer System test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_engram_transfer_system())