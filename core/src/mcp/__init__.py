"""
MCP Core System - Brain-Inspired Agentic Workflow Accelerator

This package provides the core implementation of the MCP (Model Context Protocol) system,
a comprehensive framework for AI agent workflows, memory management, and context orchestration.

The system follows a brain-inspired modular architecture with specialized "lobes" for
different cognitive functions, enabling sophisticated AI agent capabilities.
"""

__version__ = "1.1.0"
__author__ = "MCP Development Team"
__description__ = "Brain-inspired agentic workflow accelerator"

# Core imports for easy access
from .memory import MemoryManager
from .workflow import WorkflowManager
from .project_manager import ProjectManager
from .task_manager import TaskManager
from .context_manager import ContextManager
from .server import MCPServer

# Advanced imports (with fallback handling)
try:
    from .unified_memory import UnifiedMemoryManager
    from .rag_system import RAGSystem
    from .performance_monitor import ObjectivePerformanceMonitor
    from .advanced_memory import TFIDFEncoder, RaBitQEncoder
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

# Experimental imports (with fallback handling)
try:
    from .experimental_lobes import (
        SimulatedReality,
        DreamingEngine,
        PatternRecognitionEngine,
        AlignmentEngine,
        ScientificProcessEngine
    )
    EXPERIMENTAL_FEATURES_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_FEATURES_AVAILABLE = False

# Core exports
__all__ = [
    # Core managers
    'MemoryManager',
    'WorkflowManager', 
    'ProjectManager',
    'TaskManager',
    'ContextManager',
    'MCPServer',
    
    # Feature availability flags
    'ADVANCED_FEATURES_AVAILABLE',
    'EXPERIMENTAL_FEATURES_AVAILABLE',
    
    # Version info
    '__version__',
    '__author__',
    '__description__',
]

# Conditionally add advanced features to exports
if ADVANCED_FEATURES_AVAILABLE:
    __all__.extend([
        'UnifiedMemoryManager',
        'RAGSystem', 
        'ObjectivePerformanceMonitor',
        'TFIDFEncoder',
        'RaBitQEncoder',
    ])

if EXPERIMENTAL_FEATURES_AVAILABLE:
    __all__.extend([
        'SimulatedReality',
        'DreamingEngine',
        'PatternRecognitionEngine', 
        'AlignmentEngine',
        'ScientificProcessEngine',
    ])

def get_system_info():
    """Get comprehensive system information."""
    return {
        'version': __version__,
        'description': __description__,
        'advanced_features': ADVANCED_FEATURES_AVAILABLE,
        'experimental_features': EXPERIMENTAL_FEATURES_AVAILABLE,
        'core_modules': [
            'MemoryManager',
            'WorkflowManager',
            'ProjectManager', 
            'TaskManager',
            'ContextManager',
            'MCPServer'
        ]
    }

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    # Check core dependencies
    try:
        import click
        import sqlite3
        import json
        import pathlib
    except ImportError as e:
        missing_deps.append(f"Core dependency: {e}")
    
    # Check advanced dependencies
    if ADVANCED_FEATURES_AVAILABLE:
        try:
            import numpy
            import faiss
            import sklearn
        except ImportError as e:
            missing_deps.append(f"Advanced dependency: {e}")
    
    # Check experimental dependencies
    if EXPERIMENTAL_FEATURES_AVAILABLE:
        try:
            import torch
            import transformers
        except ImportError as e:
            missing_deps.append(f"Experimental dependency: {e}")
    
    return {
        'all_available': len(missing_deps) == 0,
        'missing_dependencies': missing_deps
    }

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())