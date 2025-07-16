#!/usr/bin/env python3
"""Simple test of the stub elimination engine."""

import sys
import os
sys.path.append('src/mcp')

from stub_elimination_engine import StubEliminationEngine
from pathlib import Path

def main():
    print("=== SIMPLE STUB ELIMINATION TEST ===")
    
    engine = StubEliminationEngine()
    
    # Test scanning a specific file that we know has stubs
    test_files = [
        "src/mcp/lobes/alignment_engine.py",
        "src/mcp/lobes/pattern_recognition_engine.py"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\nScanning {file_path}...")
            engine._scan_file(Path(file_path))
    
    print(f"\nTotal stubs detected: {len(engine.detected_stubs)}")
    
    if engine.detected_stubs:
        # Show first few stubs
        print("\nFirst 5 stubs found:")
        for i, stub in enumerate(engine.detected_stubs[:5]):
            print(f"{i+1}. {Path(stub.file_path).name}:{stub.line_number}")
            print(f"   Function: {stub.function_name}")
            print(f"   Type: {stub.stub_type}")
            print(f"   Severity: {stub.severity}")
            print(f"   Context: {stub.context[:60]}...")
        
        # Generate replacements for first few stubs
        print(f"\nGenerating replacements for first 3 stubs...")
        test_stubs = engine.detected_stubs[:3]
        replacements = engine.generate_replacements(test_stubs)
        
        print(f"Generated {len(replacements)} replacements")
        
        if replacements:
            print("\nFirst replacement:")
            replacement = replacements[0]
            print(f"Function: {replacement.stub_info.function_name}")
            print(f"Confidence: {replacement.confidence}")
            print("Code:")
            print(replacement.replacement_code)
    
    print("\n=== Test completed! ===")

if __name__ == "__main__":
    main()