#!/usr/bin/env python3
"""Enhanced test of the improved stub elimination functionality."""

import sys
import os
sys.path.append('src/mcp')

from stub_elimination_engine import StubEliminationEngine
from pathlib import Path

def test_comprehensive_scan():
    """Test comprehensive scanning of the entire source directory."""
    print("=== COMPREHENSIVE STUB ELIMINATION TEST ===")
    
    engine = StubEliminationEngine()
    
    # Scan the entire src directory
    print("Scanning entire src/ directory for stubs...")
    stubs = engine.scan_for_stubs(["src/"])
    
    print(f"\nTotal stubs detected: {len(stubs)}")
    
    # Generate comprehensive report
    report = engine.get_stub_report()
    print(f"Severity breakdown: {report['severity_breakdown']}")
    print(f"Type breakdown: {report['type_breakdown']}")
    
    # Show some examples of each type
    for stub_type, count in report['type_breakdown'].items():
        if count > 0:
            print(f"\n--- {stub_type.upper()} Examples ({count} total) ---")
            examples = [s for s in stubs if s.stub_type == stub_type][:3]
            for stub in examples:
                print(f"  {Path(stub.file_path).name}:{stub.line_number} - {stub.function_name}")
                print(f"    Context: {stub.context[:80]}...")
    
    return engine, stubs

def test_replacement_generation():
    """Test the enhanced replacement generation."""
    print("\n=== TESTING REPLACEMENT GENERATION ===")
    
    engine, stubs = test_comprehensive_scan()
    
    if not stubs:
        print("No stubs found to generate replacements for.")
        return engine, []
    
    # Generate replacements for high-priority stubs
    high_priority_stubs = [s for s in stubs if s.severity in ['critical', 'high']]
    print(f"\nGenerating replacements for {len(high_priority_stubs)} high-priority stubs...")
    
    replacements = engine.generate_replacements(high_priority_stubs)
    print(f"Generated {len(replacements)} replacements")
    
    # Show sample replacements
    if replacements:
        print("\n--- Sample Replacements ---")
        for i, replacement in enumerate(replacements[:3]):
            print(f"\n{i+1}. {replacement.stub_info.function_name} in {Path(replacement.stub_info.file_path).name}")
            print(f"   Type: {replacement.replacement_type}")
            print(f"   Confidence: {replacement.confidence}")
            print("   Code:")
            print("   " + "\n   ".join(replacement.replacement_code.strip().split('\n')))
    
    return engine, replacements

def test_validation():
    """Test the validation functionality."""
    print("\n=== TESTING VALIDATION ===")
    
    engine, replacements = test_replacement_generation()
    
    if not replacements:
        print("No replacements to validate.")
        return
    
    # Validate replacements
    validation_results = engine.validate_replacements(replacements)
    
    print(f"Validation Results:")
    print(f"  Total: {validation_results['total_replacements']}")
    print(f"  Valid: {validation_results['valid']}")
    print(f"  Invalid: {validation_results['invalid']}")
    print(f"  Warnings: {validation_results['warnings']}")
    
    if validation_results['syntax_errors']:
        print(f"\nSyntax Errors ({len(validation_results['syntax_errors'])}):")
        for error in validation_results['syntax_errors'][:3]:
            print(f"  - {Path(error['file']).name}:{error['line']} - {error['function']}")
            print(f"    Error: {error['error']}")
    
    if validation_results['issues']:
        print(f"\nIssues Found ({len(validation_results['issues'])}):")
        for issue in validation_results['issues'][:5]:
            print(f"  - {issue}")

def test_comprehensive_report():
    """Test the comprehensive reporting functionality."""
    print("\n=== TESTING COMPREHENSIVE REPORT ===")
    
    engine = StubEliminationEngine()
    stubs = engine.scan_for_stubs(["src/"])
    
    if stubs:
        replacements = engine.generate_replacements(stubs[:10])  # Limit for testing
        comprehensive_report = engine.generate_comprehensive_report()
        
        print("Comprehensive Report Generated:")
        print(f"  Scan Results: {comprehensive_report['scan_results']['total_stubs']} stubs found")
        
        if 'validation_results' in comprehensive_report:
            val_results = comprehensive_report['validation_results']
            if isinstance(val_results, dict) and 'total_replacements' in val_results:
                print(f"  Validation: {val_results['valid']}/{val_results['total_replacements']} valid")
        
        print(f"  Recommendations: {len(comprehensive_report['recommendations'])}")
        for rec in comprehensive_report['recommendations'][:3]:
            print(f"    - {rec}")
    else:
        print("No stubs found for comprehensive report.")

def test_dry_run_application():
    """Test dry run application of replacements."""
    print("\n=== TESTING DRY RUN APPLICATION ===")
    
    engine = StubEliminationEngine()
    stubs = engine.scan_for_stubs(["src/"])
    
    if not stubs:
        print("No stubs found for dry run test.")
        return
    
    # Generate replacements for a few stubs
    test_stubs = stubs[:5]  # Limit for testing
    replacements = engine.generate_replacements(test_stubs)
    
    if replacements:
        print(f"Testing dry run application of {len(replacements)} replacements...")
        
        # Apply in dry run mode
        results = engine.apply_replacements(replacements, dry_run=True)
        
        print(f"Dry Run Results:")
        print(f"  Total replacements: {results['total_replacements']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Files that would be modified: {len(results['files_modified'])}")
        
        if results['files_modified']:
            print("  Files:")
            for file_path in results['files_modified'][:3]:
                print(f"    - {Path(file_path).name}")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:
                print(f"    - {error}")
    else:
        print("No replacements generated for dry run test.")

if __name__ == "__main__":
    try:
        test_comprehensive_scan()
        test_replacement_generation()
        test_validation()
        test_comprehensive_report()
        test_dry_run_application()
        print("\n=== ALL TESTS COMPLETED SUCCESSFULLY! ===")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()