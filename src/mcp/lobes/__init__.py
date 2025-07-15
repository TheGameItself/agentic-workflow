"""
Lobes package for the MCP server.

This package contains all the experimental lobes organized by functionality.
Each lobe represents a specialized cognitive function inspired by brain research.

Note: Experimental lobes should be imported directly from experimental_lobes.py to avoid circular imports.
"""

from .alignment_engine import AlignmentEngine
from .pattern_recognition_engine import PatternRecognitionEngine

# Experimental lobes (SimulatedReality, DreamingEngine, MindMapEngine, ScientificProcessEngine, SpeculationEngine, SplitBrainABTest, MultiLLMOrchestrator, AdvancedEngramEngine, DecisionMakingLobe, EmotionContextLobe, CreativityLobe, ErrorDetectionLobe)
# should be imported directly from experimental_lobes.py as needed to avoid circular import issues.

__all__ = [
    'AlignmentEngine',
    'PatternRecognitionEngine',
    # Experimental lobes are not included here to prevent circular imports
] 