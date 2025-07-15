# Experimental lobes package for MCP
# Modular plug-and-play architecture: Each lobe is a fully independent module with clear interfaces and research references.
# See idea.txt and latest research for requirements and design principles.

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
from .simulated_reality import SimulatedReality
from src.mcp.lobes.alignment_engine import AlignmentEngine
from src.mcp.lobes.pattern_recognition_engine import PatternRecognitionEngine

# In-place stubs for missing lobes (should be modularized in the future if not already present as modules)
# class AlignmentEngine:
#     """Stub for AlignmentEngine. See idea.txt and [Deep Research Agents, arXiv:2506.18096]."""
#     pass
#
# class PatternRecognitionEngine:
#     """Stub for PatternRecognitionEngine. See idea.txt and [Python Agentic Frameworks, Medium 2025]."""
#     pass
