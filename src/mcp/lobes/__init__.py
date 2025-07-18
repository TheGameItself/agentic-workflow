"""
MCP Lobes package.

This package contains brain-inspired lobe implementations for the MCP system,
including pattern recognition, alignment, multi-LLM orchestration, and advanced
engram processing with hormone-based cross-lobe communication.
"""

# Import specific lobe implementations from their respective modules
from .pattern_recognition_engine_adaptive import AdaptivePatternRecognitionEngine as EnhancedPatternRecognitionEngine
from .alignment_engine import AlignmentEngine as AlignmentEngineLobe

# For backward compatibility, create aliases
PatternRecognitionEngine = EnhancedPatternRecognitionEngine

__all__ = [
    'PatternRecognitionEngine',
    'EnhancedPatternRecognitionEngine',
    'AlignmentEngineLobe'
]