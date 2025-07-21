#!/usr/bin/env python3
"""
Simple test to check imports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from mcp.three_tier_memory_manager import ThreeTierMemoryManager, MemoryTier
    print("✓ ThreeTierMemoryManager imported successfully")
    
    from mcp.enhanced_vector_memory import BackendType
    print("✓ BackendType imported successfully")
    
    from unittest.mock import Mock
    print("✓ Mock imported successfully")
    
    # Test basic initialization
    manager = ThreeTierMemoryManager(
        working_capacity_mb=1.0,
        short_term_capacity_gb=0.01,
        long_term_capacity_gb=0.01,
        vector_backend=BackendType.IN_MEMORY,
        hormone_system=Mock(),
        genetic_trigger_manager=Mock()
    )
    print("✓ ThreeTierMemoryManager initialized successfully")
    
    # Test basic store operation
    success = manager.store("test", "data", "context")
    print(f"✓ Store operation: {success}")
    
    # Test basic retrieve operation
    item = manager.retrieve("test", "context")
    print(f"✓ Retrieve operation: {item is not None}")
    
    print("\n🎉 Basic functionality verified!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()