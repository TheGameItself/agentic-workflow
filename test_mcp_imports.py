#!/usr/bin/env python3
"""
Test script to check MCP imports and identify issues.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing MCP imports...")
    
    # Test basic imports
    print("✓ Basic imports working")
    
    # Test MCP module imports
    from mcp import cli
    print("✓ MCP CLI module imported successfully")
    
    # Test specific components
    from mcp.memory import MemoryManager
    print("✓ MemoryManager imported successfully")
    
    from mcp.workflow import WorkflowManager
    print("✓ WorkflowManager imported successfully")
    
    from mcp.project_manager import ProjectManager
    print("✓ ProjectManager imported successfully")
    
    from mcp.task_manager import TaskManager
    print("✓ TaskManager imported successfully")
    
    print("\n✅ All MCP imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc() 