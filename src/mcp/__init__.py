#!/usr/bin/env python3
"""
MCP Agentic Workflow Accelerator
A portable, local-only Python MCP server for accelerating agentic development workflows.
"""

# This file marks the MCP package as supporting PEP 561 type checking.
# See https://peps.python.org/pep-0561/ and https://docs.basedpyright.com/v1.26.0/usage/typed-libraries/

__version__ = "2.0.0"
__author__ = "MCP Development Team"
__description__ = "Portable, local-only Python MCP server for agentic development workflows"

from .memory import MemoryManager
from .regex_search import RegexSearchEngine, SearchQuery, SearchType, SearchScope, RegexSearchFormatter
from .hormone_system_controller import HormoneSystemController

__all__ = [
    'MemoryManager',
    'RegexSearchEngine',
    'SearchQuery', 
    'SearchType',
    'SearchScope',
    'RegexSearchFormatter',
    'HormoneSystemController',
    '__version__',
    '__author__',
    '__description__'
]

def get_version():
    """Get the current version of the MCP package."""
    return __version__

def create_mcp_instance():
    """Create a new MCP instance with memory management."""
    return MemoryManager() 