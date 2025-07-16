#!/usr/bin/env python3
"""Full test of stub elimination functionality."""

import sys
import os
sys.path.append('src/mcp')

from stub_elimination_engine import StubEliminationEngine, eliminate_stubs_in_project
from pathlib import Path

def test_full_elimination():
    """Test the complete stub elimination process."""
    print("=== FULL STUB ELIMINATION TEST ===")
    
    # Test with a few key files that have NotImplementedError
    test_files = [
        "src/mcp/dreaming_engine.py",
        "src/mcp/advanced_memory.py", 
        "src/mcp/brain_state_aggregator.py"
    ]
    
    engine = StubEliminationEngine()
    
    # Scan specific files
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\n--- Scanning {file_path} ---")
            engine._scan_file(Path(file_path))
    
    print(f"\nTotal stubs detected: {len(engine.detected_stubs)}")
    
    # Generate report
    report = engine.get_stub_report()
    print(f"Severity breakdown: {report['severity_breakdown']}")
    print(f"Type breakdown: {report['type_breakdown']}")
    
    # Show critical stubs
    critical_stubs = report.get('critical_stubs', [])
    print(f"\nCritical stubs ({len(critical_stubs)}):")
    for stub in critical_stubs[:5]:
        print(f"  - {Path(stub.file_path).name}:{stub.line_number} - {stub.function_name}")
    
    # Generate replacements for critical stubs
    if critical_stubs:
        print(f"\n--- Generating replacements for {len(critical_stubs)} critical stubs ---")
        replacements = engine.generate_replacements(critical_stubs)
        print(f"Generated {len(replacements)} replacements")
        
        # Show first replacement
        if replacements:
            replacement = replacements[0]
            print(f"\nSample replacement for {replacement.stub_info.function_name}:")
            print(f"File: {Path(replacement.stub_info.file_path).name}")
            print(f"Line: {replacement.stub_info.line_number}")
            print(f"Confidence: {replacement.confidence}")
            print("Code:")
            print(replacement.replacement_code)
        
        # Test dry run application
        print("\n--- Testing dry run application ---")
        results = engine.apply_replacements(replacements[:3], dry_run=True)
        print(f"Would modify {len(results['files_modified'])} files")
        print(f"Success rate: {results['successful']}/{results['total_replacements']}")
        
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
    
    print("\n=== Test completed successfully! ===")

def test_convenience_function():
    """Test the convenience function."""
    print("\n=== TESTING CONVENIENCE FUNCTION ===")
    
    # This would scan the entire project, so let's limit it
    # results = eliminate_stubs_in_project(dry_run=True)
    # print(f"Total stubs found: {results['scan_results']['total_stubs']}")
    print("Convenience function test skipped (would scan entire project)")

if __name__ == "__main__":
    test_full_elimination()
    test_convenience_function()