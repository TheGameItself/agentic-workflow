# DEPRECATED: This file is preserved for migration history and test compatibility only.
# All experimental lobes are now modularized in src/mcp/lobes/experimental/ as individual modules.
# Do not add new logic here. See modularized files for current implementations.

"""
This module previously consolidated stub/experimental lobes for easier maintenance and future development.
All lobes are now modularized. See src/mcp/lobes/experimental/ for current code.
This file is preserved for test compatibility and migration history only.
"""

# --- Legacy Re-Exports for Test Compatibility Only ---
from .lobes.alignment_engine import AlignmentEngine
from .lobes.pattern_recognition_engine import PatternRecognitionEngine
from .lobes.experimental.advanced_engram.advanced_engram_engine import AdvancedEngramEngine
from .lobes.experimental.mind_map.mind_map_engine import MindMapEngine
from .lobes.experimental.scientific_process.scientific_process_engine import ScientificProcessEngine
from .lobes.experimental.split_brain_ab_test.split_brain_ab_test import SplitBrainABTest
from .lobes.experimental.decision_making.decision_making_lobe import DecisionMakingLobe
from .lobes.experimental.error_detection.error_detection_lobe import ErrorDetectionLobe
from .lobes.experimental.task_proposal.task_proposal_lobe import TaskProposalLobe
from .lobes.experimental.multi_llm_orchestrator.multi_llm_orchestrator import MultiLLMOrchestrator

# No active class definitions or logic should be added here.
# All stub classes (SimulatedReality, SpeculationEngine, etc.) have been removed as they are now modularized.

"""
This module re-exports experimental lobes for test and integration compatibility.
All classes are stubs or minimal implementations unless otherwise noted.
See idea.txt for requirements and future expansion.
"""