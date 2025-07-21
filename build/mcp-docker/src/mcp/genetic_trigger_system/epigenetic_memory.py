"""
EpigeneticMemory: Stores epigenetic markers for genetic triggers.

Implements comprehensive epigenetic memory with methylation patterns,
histone modifications, stress markers, and inheritance mechanisms.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EpigeneticMarkers:
    """Collection of epigenetic markers"""
    methylation_pattern: Dict[str, float] = field(default_factory=dict)
    histone_modifications: Dict[str, float] = field(default_factory=dict)
    stress_markers: Dict[str, float] = field(default_factory=dict)
    imprinting_markers: Dict[str, bool] = field(default_factory=dict)
    chromatin_state: Dict[str, str] = field(default_factory=dict)


class EpigeneticMemory:
    """
    Comprehensive epigenetic memory system for genetic triggers.
    
    Stores and manages epigenetic markers including methylation patterns,
    histone modifications, stress markers, and inheritance mechanisms.
    """
    
    def __init__(self):
        """Initialize epigenetic memory."""
        self.logger = logging.getLogger("EpigeneticMemory")
        self.markers = EpigeneticMarkers()
        self.inheritance_history: List[Dict[str, Any]] = []
        self.environmental_history: List[Dict[str, Any]] = []
        self.adaptation_memory: Dict[str, float] = {}
        
    def add_stress_marker(self, gene_id: str, stress_level: float) -> None:
        """Add a stress marker for a specific gene."""
        self.markers.stress_markers[gene_id] = stress_level
        self.logger.info(f"Added stress marker for {gene_id}: {stress_level}")
        
    def add_learning_marker(self, gene_id: str, learning_level: float) -> None:
        """Add a learning marker (histone modification) for a specific gene."""
        self.markers.histone_modifications[gene_id] = learning_level
        self.logger.info(f"Added learning marker for {gene_id}: {learning_level}")
        
    def add_methylation_marker(self, gene_id: str, methylation_level: float) -> None:
        """Add a methylation marker for a specific gene."""
        self.markers.methylation_pattern[gene_id] = methylation_level
        self.logger.info(f"Added methylation marker for {gene_id}: {methylation_level}")
        
    def get_expression_modifier(self, gene_id: str) -> float:
        """Calculate expression modifier for a gene based on epigenetic markers."""
        modifier = 1.0
        
        # Methylation effects (typically repressive)
        if gene_id in self.markers.methylation_pattern:
            modifier *= (1.0 - self.markers.methylation_pattern[gene_id] * 0.5)
        
        # Histone modification effects
        if gene_id in self.markers.histone_modifications:
            modifier *= (1.0 + self.markers.histone_modifications[gene_id] * 0.3)
        
        # Stress marker effects
        if gene_id in self.markers.stress_markers:
            modifier *= (1.0 + self.markers.stress_markers[gene_id] * 0.2)
        
        return max(0.1, min(3.0, modifier))
        
    def inherit_from_parent(self, parent_memory: 'EpigeneticMemory') -> None:
        """Inherit epigenetic markers from parent with some reduction."""
        # Record inheritance
        inheritance_record = {
            'timestamp': datetime.now().isoformat(),
            'parent_markers_count': len(parent_memory.markers.methylation_pattern),
            'inherited_markers': []
        }
        
        # Inherit methylation patterns with reduction
        for gene_id, level in parent_memory.markers.methylation_pattern.items():
            inherited_level = level * 0.7  # 30% reduction
            self.markers.methylation_pattern[gene_id] = inherited_level
            inheritance_record['inherited_markers'].append({
                'gene_id': gene_id,
                'original_level': level,
                'inherited_level': inherited_level,
                'type': 'methylation'
            })
        
        # Inherit histone modifications with reduction
        for gene_id, level in parent_memory.markers.histone_modifications.items():
            inherited_level = level * 0.8  # 20% reduction
            self.markers.histone_modifications[gene_id] = inherited_level
            inheritance_record['inherited_markers'].append({
                'gene_id': gene_id,
                'original_level': level,
                'inherited_level': inherited_level,
                'type': 'histone'
            })
        
        # Inherit imprinting markers
        for gene_id, imprinted in parent_memory.markers.imprinting_markers.items():
            if imprinted:
                self.markers.imprinting_markers[gene_id] = True
                inheritance_record['inherited_markers'].append({
                    'gene_id': gene_id,
                    'type': 'imprinting'
                })
        
        self.inheritance_history.append(inheritance_record)
        self.logger.info(f"Inherited {len(inheritance_record['inherited_markers'])} markers from parent")
        
    def set_marker(self, key: str, value: Any) -> None:
        """Set a generic marker."""
        self.logger.info(f"Setting marker {key}={value}")
        # Store in adaptation memory
        self.adaptation_memory[key] = value
        
    def get_marker(self, key: str) -> Any:
        """Get a generic marker."""
        value = self.adaptation_memory.get(key)
        self.logger.info(f"Getting marker {key}={value}")
        return value
        
    def add_environmental_context(self, context: Dict[str, Any]) -> None:
        """Add environmental context to memory."""
        self.environmental_history.append({
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        # Keep only recent history
        if len(self.environmental_history) > 1000:
            self.environmental_history = self.environmental_history[-1000:]
            
    def calculate_adaptation_score(self, target_environment: Dict[str, Any]) -> float:
        """Calculate adaptation score based on environmental similarity."""
        if not self.environmental_history:
            return 0.5
            
        # Find similar environments
        similarities = []
        for record in self.environmental_history:
            similarity = self._calculate_similarity(record['context'], target_environment)
            similarities.append(similarity)
            
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.5
        
    def _calculate_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two environmental contexts."""
        if not context1 or not context2:
            return 0.0
            
        # Simple similarity calculation
        all_keys = set(context1.keys()) | set(context2.keys())
        if not all_keys:
            return 0.0
            
        similarities = []
        for key in all_keys:
            val1 = context1.get(key, 0.0)
            val2 = context2.get(key, 0.0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    diff = abs(val1 - val2) / max_val
                    similarities.append(max(0.0, 1.0 - diff))
            else:
                # For non-numeric values, simple equality check
                similarities.append(1.0 if val1 == val2 else 0.0)
                
        return sum(similarities) / len(similarities) if similarities else 0.0
