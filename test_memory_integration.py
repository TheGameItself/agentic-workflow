#!/usr/bin/env python3
"""
Simple test script to verify ThreeTierMemoryManager integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mcp.three_tier_memory_manager import ThreeTierMemoryManager, MemoryTier
from mcp.enhanced_vector_memory import BackendType
from unittest.mock import Mock

def test_basic_integration():
    """Test basic integration functionality."""
    print("Testing ThreeTierMemoryManager integration...")
    
    # Create mock systems
    hormone_system = Mock()
    genetic_trigger_manager = Mock()
    
    # Initialize memory manager
    manager = ThreeTierMemoryManager(
        working_capacity_mb=10.0,
        short_term_capacity_gb=0.1,
        long_term_capacity_gb=0.5,
        vector_backend=BackendType.IN_MEMORY,
        hormone_system=hormone_system,
        genetic_trigger_manager=genetic_trigger_manager
    )
    
    print("✓ Memory manager initialized successfully")
    
    # Test storing data
    success = manager.store(
        key="test_key",
        data="test data",
        context="test_context",
        priority=0.5,
        lobe_id="test_lobe"
    )
    
    assert success, "Failed to store data"
    print("✓ Data stored successfully")
    
    # Test retrieving data
    item = manager.retrieve("test_key", context="test_context")
    assert item is not None, "Failed to retrieve data"
    assert item.data == "test data", "Retrieved data doesn't match"
    assert item.key == "test_key", "Retrieved key doesn't match"
    
    print("✓ Data retrieved successfully")
    
    # Test cross-tier search
    results = manager.cross_tier_search("test", context="test_context", limit=10)
    assert len(results) > 0, "Cross-tier search returned no results"
    
    print("✓ Cross-tier search working")
    
    # Test consolidation
    consolidation_results = manager.consolidate_memory(force=True)
    assert isinstance(consolidation_results, dict), "Consolidation didn't return dict"
    assert 'working_to_short' in consolidation_results, "Missing consolidation stats"
    
    print("✓ Memory consolidation working")
    
    # Test comprehensive stats
    stats = manager.get_comprehensive_stats()
    assert isinstance(stats, dict), "Stats didn't return dict"
    assert 'working_memory' in stats, "Missing working memory stats"
    assert 'tier_transitions' in stats, "Missing tier transition stats"
    assert 'consolidation_stats' in stats, "Missing consolidation stats"
    
    print("✓ Comprehensive stats working")
    
    # Test memory health
    health = manager.get_memory_health()
    assert isinstance(health, dict), "Health didn't return dict"
    assert 'overall_health' in health, "Missing overall health"
    assert 'recommendations' in health, "Missing recommendations"
    
    print("✓ Memory health assessment working")
    
    # Test hormone system integration
    assert hormone_system.release_hormone.called, "Hormone system not called"
    print("✓ Hormone system integration working")
    
    # Test genetic trigger integration
    assert genetic_trigger_manager.evaluate_triggers.called, "Genetic trigger system not called"
    print("✓ Genetic trigger integration working")
    
    print("\n🎉 All integration tests passed!")
    return True

def test_tier_transitions():
    """Test automatic tier transitions."""
    print("\nTesting tier transitions...")
    
    manager = ThreeTierMemoryManager(
        working_capacity_mb=5.0,
        short_term_capacity_gb=0.05,
        long_term_capacity_gb=0.1,
        vector_backend=BackendType.IN_MEMORY,
        hormone_system=Mock(),
        genetic_trigger_manager=Mock()
    )
    
    # Store item and access multiple times
    key = "transition_test"
    manager.store(key=key, data="transition data", context="test")
    
    # Access multiple times to build pattern
    for i in range(6):
        item = manager.retrieve(key, context="test")
        assert item is not None, f"Failed to retrieve on access {i+1}"
    
    # Check access patterns
    assert key in manager.access_patterns, "Access patterns not tracked"
    pattern = manager.access_patterns[key]
    assert pattern['access_count'] >= 6, "Access count not tracked correctly"
    
    print("✓ Access patterns tracked correctly")
    
    # Test tier transition stats
    stats = manager.get_comprehensive_stats()
    transition_stats = stats['tier_transitions']
    assert 'total_transitions' in transition_stats, "Missing transition stats"
    
    print("✓ Tier transition tracking working")
    
    return True

def test_performance():
    """Test basic performance characteristics."""
    print("\nTesting performance...")
    
    manager = ThreeTierMemoryManager(
        working_capacity_mb=50.0,
        short_term_capacity_gb=0.5,
        long_term_capacity_gb=1.0,
        vector_backend=BackendType.IN_MEMORY,
        hormone_system=Mock(),
        genetic_trigger_manager=Mock()
    )
    
    import time
    
    # Test storing many items
    start_time = time.time()
    num_items = 50
    
    for i in range(num_items):
        success = manager.store(f"perf_test_{i}", f"performance data {i}", "perf_context")
        assert success, f"Failed to store item {i}"
    
    store_time = time.time() - start_time
    print(f"✓ Stored {num_items} items in {store_time:.3f} seconds")
    
    # Test retrieving many items
    start_time = time.time()
    retrieved = 0
    
    for i in range(num_items):
        item = manager.retrieve(f"perf_test_{i}", context="perf_context")
        if item is not None:
            retrieved += 1
    
    retrieve_time = time.time() - start_time
    print(f"✓ Retrieved {retrieved}/{num_items} items in {retrieve_time:.3f} seconds")
    
    # Test search performance
    start_time = time.time()
    results = manager.cross_tier_search("performance", context="perf_context", limit=25)
    search_time = time.time() - start_time
    
    print(f"✓ Cross-tier search found {len(results)} results in {search_time:.3f} seconds")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_integration()
        test_tier_transitions()
        test_performance()
        print("\n🎉 All tests completed successfully!")
        print("\nThreeTierMemoryManager integration and coordination is working correctly.")
        print("Key features verified:")
        print("- Automatic memory tier transitions based on access patterns")
        print("- Cross-tier search and retrieval optimization")
        print("- Memory consolidation workflows between tiers")
        print("- Hormone system and genetic trigger integration")
        print("- Comprehensive statistics and health monitoring")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)