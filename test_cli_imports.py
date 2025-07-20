#!/usr/bin/env python3
"""
Test script to check CLI imports
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    from mcp.memory import MemoryManager
    print("✓ MemoryManager imported")
except ImportError as e:
    print(f"✗ MemoryManager import failed: {e}")

try:
    from mcp.workflow import WorkflowManager
    print("✓ WorkflowManager imported")
except ImportError as e:
    print(f"✗ WorkflowManager import failed: {e}")

try:
    from mcp.project_manager import ProjectManager
    print("✓ ProjectManager imported")
except ImportError as e:
    print(f"✗ ProjectManager import failed: {e}")

try:
    from mcp.task_manager import TaskManager
    print("✓ TaskManager imported")
except ImportError as e:
    print(f"✗ TaskManager import failed: {e}")

try:
    from mcp.unified_memory import UnifiedMemoryManager
    print("✓ UnifiedMemoryManager imported")
except ImportError as e:
    print(f"✗ UnifiedMemoryManager import failed: {e}")

try:
    from mcp.performance_monitor import ObjectivePerformanceMonitor
    print("✓ ObjectivePerformanceMonitor imported")
except ImportError as e:
    print(f"✗ ObjectivePerformanceMonitor import failed: {e}")

try:
    from mcp.rag_system import RAGSystem
    print("✓ RAGSystem imported")
except ImportError as e:
    print(f"✗ RAGSystem import failed: {e}")

try:
    from mcp.reminder_engine import EnhancedReminderEngine
    print("✓ EnhancedReminderEngine imported")
except ImportError as e:
    print(f"✗ EnhancedReminderEngine import failed: {e}")

try:
    from mcp.advanced_memory import TFIDFEncoder, RaBitQEncoder
    print("✓ TFIDFEncoder, RaBitQEncoder imported")
except ImportError as e:
    print(f"✗ advanced_memory import failed: {e}")

try:
    from mcp.server import MCPServer
    print("✓ MCPServer imported")
except ImportError as e:
    print(f"✗ MCPServer import failed: {e}")

try:
    from mcp.automatic_update_system import AutomaticUpdateSystem, UpdateStatus
    print("✓ AutomaticUpdateSystem, UpdateStatus imported")
except ImportError as e:
    print(f"✗ automatic_update_system import failed: {e}")

print("\nTesting CLI import...")
try:
    from mcp.cli import cli
    print("✓ CLI imported successfully")
except Exception as e:
    print(f"✗ CLI import failed: {e}")
    import traceback
    traceback.print_exc() 