# Experimental lobes package for MCP
# Each lobe should be in its own module for modularity and clarity

__all__ = [
    "AlignmentEngine",
    "AdvancedEngramEngine",
    "DecisionMakingLobe",
    "ErrorDetectionLobe",
    "MindMapEngine",
    "MultiLLMOrchestrator",
    "PatternRecognitionEngine",
    "ScientificProcessEngine",
    "SimulatedReality",
    "SplitBrainABTest",
    "TaskProposalLobe",
    "SensoryColumn",
    "SpinalCord",
    "DreamingEngine",
]

from .advanced_engram.advanced_engram_engine import AdvancedEngramEngine
from .mind_map.mind_map_engine import MindMapEngine
from .scientific_process.scientific_process_engine import ScientificProcessEngine
from .split_brain_ab_test.split_brain_ab_test import SplitBrainABTest
from .decision_making.decision_making_lobe import DecisionMakingLobe
from .error_detection.error_detection_lobe import ErrorDetectionLobe
from .task_proposal.task_proposal_lobe import TaskProposalLobe
from .multi_llm_orchestrator.multi_llm_orchestrator import MultiLLMOrchestrator
from .sensory_column.sensory_column import SensoryColumn
from .spinal_cord.spinal_cord import SpinalCord
from .dreaming_engine import DreamingEngine

# Stubs for missing experimental lobes and utilities
class AlignmentEngine:
    """Stub for AlignmentEngine. TODO: Implement per idea.txt."""
    pass

class SimulatedReality:
    """Stub for SimulatedReality lobe. Implements simulated external reality for deep reasoning. TODO: Expand per idea.txt requirements."""
    def __init__(self, *args, **kwargs):
        pass
    def simulate(self, input_data=None):
        """Minimal simulation method (stub)."""
        return {"status": "stub", "result": None}

class PatternRecognitionEngine:
    """Stub for PatternRecognitionEngine. See src/mcp/lobes/pattern_recognition_engine.py for full implementation. Refer to idea.txt for requirements."""
    pass

class DreamingEngine:
    """Stub for DreamingEngine. Implements alternative scenario simulation and learning from dreams. TODO: Expand per idea.txt requirements."""
    def __init__(self, *args, **kwargs):
        pass
    def dream(self, input_data=None):
        """Minimal dream simulation method (stub)."""
        return {"status": "stub", "result": None}

# Remove imports of missing modules to resolve circular import errors
