"""
Environmental State for Genetic Trigger System

This module defines the EnvironmentalState class that represents the current
environmental conditions for genetic trigger activation and adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class EnvironmentalState:
    """
    Represents the current environmental state for genetic trigger evaluation.
    
    This class encapsulates all environmental factors that can influence
    genetic trigger activation and adaptation, including system metrics,
    performance indicators, resource usage, and hormone levels.
    """
    
    timestamp: str
    system_load: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    hormone_levels: Dict[str, float] = field(default_factory=dict)
    task_complexity: float = 0.5
    adaptation_pressure: float = 0.5
    network_conditions: Dict[str, float] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    external_factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert environmental state to dictionary"""
        return {
            'timestamp': self.timestamp,
            'system_load': self.system_load,
            'performance_metrics': self.performance_metrics,
            'resource_usage': self.resource_usage,
            'hormone_levels': self.hormone_levels,
            'task_complexity': self.task_complexity,
            'adaptation_pressure': self.adaptation_pressure,
            'network_conditions': self.network_conditions,
            'user_context': self.user_context,
            'external_factors': self.external_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentalState':
        """Create environmental state from dictionary"""
        return cls(
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            system_load=data.get('system_load', {}),
            performance_metrics=data.get('performance_metrics', {}),
            resource_usage=data.get('resource_usage', {}),
            hormone_levels=data.get('hormone_levels', {}),
            task_complexity=data.get('task_complexity', 0.5),
            adaptation_pressure=data.get('adaptation_pressure', 0.5),
            network_conditions=data.get('network_conditions', {}),
            user_context=data.get('user_context', {}),
            external_factors=data.get('external_factors', {})
        )
    
    def get_overall_load(self) -> float:
        """Calculate overall system load"""
        if not self.system_load:
            return 0.0
        
        cpu_load = self.system_load.get('cpu', 0.0)
        memory_load = self.system_load.get('memory', 0.0)
        return (cpu_load + memory_load) / 2
    
    def get_overall_performance(self) -> float:
        """Calculate overall performance score"""
        if not self.performance_metrics:
            return 0.5
        
        values = list(self.performance_metrics.values())
        return sum(values) / len(values)
    
    def get_overall_hormone_level(self) -> float:
        """Calculate overall hormone level"""
        if not self.hormone_levels:
            return 0.5
        
        values = list(self.hormone_levels.values())
        return sum(values) / len(values)
    
    def is_high_stress(self) -> bool:
        """Check if environment indicates high stress"""
        return (self.get_overall_load() > 0.8 or 
                self.get_overall_performance() < 0.3 or
                self.adaptation_pressure > 0.8)
    
    def is_optimal(self) -> bool:
        """Check if environment is optimal for performance"""
        return (0.3 <= self.get_overall_load() <= 0.7 and
                0.6 <= self.get_overall_performance() <= 0.9 and
                0.2 <= self.adaptation_pressure <= 0.6)
    
    def get_environmental_signature(self) -> str:
        """Generate a signature for the environmental state"""
        signature_parts = [
            f"L{self.get_overall_load():.2f}",
            f"P{self.get_overall_performance():.2f}",
            f"H{self.get_overall_hormone_level():.2f}",
            f"C{self.task_complexity:.2f}",
            f"A{self.adaptation_pressure:.2f}"
        ]
        return "_".join(signature_parts)
    
    def calculate_similarity(self, other: 'EnvironmentalState') -> float:
        """Calculate similarity with another environmental state"""
        if not isinstance(other, EnvironmentalState):
            return 0.0
        
        similarities = []
        
        # System load similarity
        if self.system_load and other.system_load:
            load_sim = self._calculate_dict_similarity(self.system_load, other.system_load)
            similarities.append(load_sim)
        
        # Performance metrics similarity
        if self.performance_metrics and other.performance_metrics:
            perf_sim = self._calculate_dict_similarity(self.performance_metrics, other.performance_metrics)
            similarities.append(perf_sim)
        
        # Resource usage similarity
        if self.resource_usage and other.resource_usage:
            res_sim = self._calculate_dict_similarity(self.resource_usage, other.resource_usage)
            similarities.append(res_sim)
        
        # Hormone levels similarity
        if self.hormone_levels and other.hormone_levels:
            horm_sim = self._calculate_dict_similarity(self.hormone_levels, other.hormone_levels)
            similarities.append(horm_sim)
        
        # Scalar similarities
        task_sim = 1.0 - abs(self.task_complexity - other.task_complexity)
        pressure_sim = 1.0 - abs(self.adaptation_pressure - other.adaptation_pressure)
        similarities.extend([task_sim, pressure_sim])
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_dict_similarity(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
        """Calculate similarity between two dictionaries of float values"""
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    def __str__(self) -> str:
        """String representation of environmental state"""
        return (f"EnvironmentalState(load={self.get_overall_load():.2f}, "
                f"perf={self.get_overall_performance():.2f}, "
                f"complexity={self.task_complexity:.2f}, "
                f"pressure={self.adaptation_pressure:.2f})")
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"EnvironmentalState(timestamp='{self.timestamp}', "
                f"system_load={self.system_load}, "
                f"performance_metrics={self.performance_metrics}, "
                f"resource_usage={self.resource_usage}, "
                f"hormone_levels={self.hormone_levels}, "
                f"task_complexity={self.task_complexity}, "
                f"adaptation_pressure={self.adaptation_pressure})") 