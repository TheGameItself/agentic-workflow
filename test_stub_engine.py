#!/usr/bin/env python3
"""Test script for stub elimination engine."""

import sys
sys.path.append('src/mcp')

from stub_elimination_engine import StubEliminationEngine

def test_stub_engine():
    """Test the stub elimination engine."""
    print("Creating StubEliminationEngine...")
    engine = StubEliminationEngine()
    
    print("Testing with a specific file...")
    # Test with just one file to avoid long scan times
    test_paths = ["src/mcp"]
    stubs = engine.scan_for_stubs(test_paths)
    
    print(f"Found {len(stubs)} stubs in MCP source")
    
    if stubs:
        print("\nFirst few stubs:")
        for i, stub in enumerate(stubs[:3]):
            print(f"  {i+1}. {stub.file_path}:{stub.line_number} - {stub.function_name} ({stub.stub_type}, {stub.severity})")
    
    # Test replacement generation
    if stubs:
        print("\nGenerating replacements...")
        replacements = engine.generate_replacements(stubs[:2])  # Just first 2
        print(f"Generated {len(replacements)} replacements")
        
        if replacements:
            print("\nFirst replacement:")
            replacement = replacements[0]
            print(f"  For: {replacement.stub_info.function_name}")
            print(f"  Type: {replacement.replacement_type}")
            print(f"  Confidence: {replacement.confidence}")
            print(f"  Code preview: {replacement.replacement_code[:100]}...")
    
    print("\nStub engine test completed successfully!")

if __name__ == "__main__":
    test_stub_engine()