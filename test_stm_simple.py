#!/usr/bin/env python3
"""
Simple test for enhanced ShortTermMemory
"""

import sys
import os
import logging
import time
import threading
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Callable, Union
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_short_term_memory():
    """Test the enhanced ShortTermMemory functionality"""
    print("Testing Enhanced ShortTermMemory...")
    
    # Import the class directly from the file
    import importlib.util
    spec = importlib.util.spec_from_file_location("working_memory", "src/mcp/lobes/shared_lobes/working_memory.py")
    working_memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(working_memory)
    
    ShortTermMemory = working_memory.ShortTermMemory
    
    # Initialize with neural retention
    stm = ShortTermMemory(capacity_gb=0.001, enable_neural_retention=True)
    print("✓ Initialized successfully")
    
    # Test adding items with different priorities and characteristics
    test_cases = [
        ("critical_task", {"type": "error", "message": "Critical system error"}, "system", 0.9, "task", ["critical", "error"]),
        ("user_note", "Remember to check email", "personal", 0.5, "note", ["reminder"]),
        ("temp_data", [1, 2, 3, 4, 5], "temp", 0.3, "data", []),
        ("config", {"debug": True, "timeout": 30}, "settings", 0.7, "config", ["system"])
    ]
    
    for key, item, context, priority, memory_type, tags in test_cases:
        result = stm.add(key, item, context, priority, memory_type, tags)
        print(f"✓ Added {key}: {result}")
    
    # Test retrieval and access tracking
    print("\nTesting retrieval...")
    for i in range(3):
        item = stm.get("critical_task", "system")
        print(f"  Access {i+1}: Retrieved critical_task: {item is not None}")
    
    # Test neural statistics
    print("\nTesting neural statistics...")
    neural_stats = stm.get_neural_stats()
    print(f"Neural Stats: {neural_stats}")
    
    # Test basic statistics
    print("\nTesting basic statistics...")
    stats = stm.get_stats()
    print(f"Basic Stats: {stats}")
    
    # Test capacity management by adding many items
    print("\nTesting capacity management...")
    for i in range(10):
        stm.add(f"filler_{i}", "x" * 100, "filler", 0.2)
    
    stats_after = stm.get_stats()
    print(f"Stats after adding filler: {stats_after}")
    
    # Check if high-priority items survived
    critical_item = stm.get("critical_task", "system")
    temp_item = stm.get("temp_data", "temp")
    print(f"Critical item survived: {critical_item is not None}")
    print(f"Temp item survived: {temp_item is not None}")
    
    # Test cleanup
    print("\nTesting cleanup...")
    removed = stm.cleanup_expired()
    print(f"Removed {removed} expired items")
    
    print("\n✓ Enhanced ShortTermMemory test completed successfully!")

if __name__ == "__main__":
    test_short_term_memory()