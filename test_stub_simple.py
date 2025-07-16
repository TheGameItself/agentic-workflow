#!/usr/bin/env python3
"""Simple test for stub detection."""

import sys
import os
sys.path.append('src/mcp')

from stub_elimination_engine import StubEliminationEngine
from pathlib import Path

def test_single_file():
    """Test stub detection on a single file."""
    print("Testing stub detection on dreaming_engine.py...")
    
    engine = StubEliminationEngine()
    
    # Read the file directly and check for NotImplementedError
    file_path = Path("src/mcp/dreaming_engine.py")
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Count NotImplementedError occurrences
        nie_count = content.count('NotImplementedError')
        print(f"Found {nie_count} NotImplementedError instances in the file")
        
        # Test the engine's scan on this specific file
        engine._scan_file(file_path)
        
        print(f"Engine detected {len(engine.detected_stubs)} stubs")
        
        for stub in engine.detected_stubs[:3]:
            print(f"  - {stub.function_name} at line {stub.line_number} ({stub.stub_type}, {stub.severity})")
        
        # Test replacement generation
        if engine.detected_stubs:
            replacements = engine.generate_replacements(engine.detected_stubs[:1])
            if replacements:
                print(f"\nGenerated replacement for {replacements[0].stub_info.function_name}:")
                print(replacements[0].replacement_code[:200] + "...")
    else:
        print("File not found!")

if __name__ == "__main__":
    test_single_file()