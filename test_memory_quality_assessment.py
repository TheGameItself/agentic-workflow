#!/usr/bin/env python3
"""
Test script for the Memory Quality Assessment system.
"""

import os
import sys
import json
from datetime import datetime

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from src.mcp.memory_quality_assessment import MemoryQualityAssessment
from src.mcp.advanced_memory import AdvancedMemoryManager
from src.mcp.unified_memory import UnifiedMemoryManager

def test_memory_quality_assessment():
    """Test the memory quality assessment system."""
    print("Testing Memory Quality Assessment System...")
    
    # Create a test database
    test_db_path = os.path.join(current_dir, 'data', 'test_memory_quality.db')
    
    # Remove existing test database if it exists
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    
    # Initialize memory systems
    advanced_memory = AdvancedMemoryManager(test_db_path)
    quality_assessment = MemoryQualityAssessment(test_db_path)
    
    # Add test memories
    print("Adding test memories...")
    
    # High quality memory
    high_quality_id = advanced_memory.add_advanced_memory(
        text="This is a detailed and specific memory about Python programming. "
             "It contains specific details like version 3.9.5 and references "
             "to important concepts. The code example is: `def calculate_sum(a, b): return a + b`. "
             "According to the official documentation, this is the recommended approach.",
        memory_type="code",
        priority=0.8,
        context="Python programming best practices",
        tags=["python", "programming", "best-practices"],
        category="code"
    )
    
    # Medium quality memory
    medium_quality_id = advanced_memory.add_advanced_memory(
        text="Python programming is fun. You can do many things with it.",
        memory_type="general",
        priority=0.5,
        context="",
        tags=["python"],
        category="general"
    )
    
    # Low quality memory
    low_quality_id = advanced_memory.add_advanced_memory(
        text="Note to self: look into this later",
        memory_type="general",
        priority=0.3,
        context="",
        tags=[],
        category=""
    )
    
    print(f"Added memories with IDs: {high_quality_id}, {medium_quality_id}, {low_quality_id}")
    
    # Test quality assessment
    print("\nTesting quality assessment...")
    
    high_quality_report = quality_assessment.assess_memory_quality(high_quality_id)
    medium_quality_report = quality_assessment.assess_memory_quality(medium_quality_id)
    low_quality_report = quality_assessment.assess_memory_quality(low_quality_id)
    
    print(f"High quality memory score: {high_quality_report['overall_score']:.2f}")
    print(f"Medium quality memory score: {medium_quality_report['overall_score']:.2f}")
    print(f"Low quality memory score: {low_quality_report['overall_score']:.2f}")
    
    # Test relationship detection
    print("\nTesting relationship detection...")
    
    # Add related memories
    related_memory_id = advanced_memory.add_advanced_memory(
        text="Python version 3.9.5 introduced several new features including improved "
             "type hints and dictionary merging with the | operator.",
        memory_type="code",
        priority=0.7,
        context="Python programming features",
        tags=["python", "programming", "features"],
        category="code"
    )
    
    relationships = quality_assessment.detect_memory_relationships(high_quality_id)
    stored_count = quality_assessment.store_memory_relationships(relationships)
    
    print(f"Detected {len(relationships)} relationships")
    print(f"Stored {stored_count} relationships")
    
    # Test memory consolidation
    print("\nTesting memory consolidation...")
    
    # Add more related memories for consolidation
    memory_id1 = advanced_memory.add_advanced_memory(
        text="Python dictionaries can be merged using the | operator in Python 3.9+",
        memory_type="code",
        priority=0.6,
        context="Python features",
        tags=["python", "dictionaries"],
        category="code"
    )
    
    memory_id2 = advanced_memory.add_advanced_memory(
        text="The | operator for dictionaries was introduced in Python 3.9 and allows "
             "for easy dictionary merging without modifying the originals.",
        memory_type="code",
        priority=0.6,
        context="Python features",
        tags=["python", "dictionaries", "operators"],
        category="code"
    )
    
    # Test different consolidation types
    merge_result = quality_assessment.consolidate_memories([memory_id1, memory_id2], 'merge')
    print(f"Merge result: new memory ID = {merge_result.get('new_memory_id')}")
    print(f"Compression ratio: {merge_result.get('compression_ratio'):.2f}")
    print(f"Information retention: {merge_result.get('information_retention'):.2f}")
    
    # Test memory optimization
    print("\nTesting memory optimization...")
    optimization_result = quality_assessment.optimize_memory_storage(similarity_threshold=0.3)
    print(f"Optimization result: {optimization_result.get('clusters_found')} clusters found")
    print(f"Memories consolidated: {optimization_result.get('memories_consolidated')}")
    print(f"New memories created: {optimization_result.get('new_memories_created')}")
    
    print("\nMemory Quality Assessment tests completed successfully!")

if __name__ == "__main__":
    test_memory_quality_assessment()