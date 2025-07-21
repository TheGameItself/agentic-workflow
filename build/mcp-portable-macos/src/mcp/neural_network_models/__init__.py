"""
Neural Network Models component for MCP System Upgrade.

This component provides neural implementations of hormone calculations
that can take over when they outperform algorithmic methods. It includes
a model registry for different hormone functions, a model manager for
loading, saving, and managing neural models, a performance tracker for
monitoring and analyzing neural model performance, and integration with
the Brain State Aggregator.
"""

from .model_manager import ModelManager, ModelRegistry
from .performance_tracker import PerformanceTracker, PerformanceMetrics, PerformanceComparison
from .brain_state_integration import NeuralPerformanceIntegration

__all__ = [
    "ModelManager", 
    "ModelRegistry", 
    "PerformanceTracker", 
    "PerformanceMetrics", 
    "PerformanceComparison",
    "NeuralPerformanceIntegration"
]