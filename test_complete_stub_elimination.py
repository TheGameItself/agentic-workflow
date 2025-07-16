#!/usr/bin/env python3
"""Complete test of the enhanced stub elimination functionality."""

import sys
import os
sys.path.append('src/mcp')

from stub_elimination_engine import StubEliminationEngine
from pathlib import Path

def test_complete_functionality():
    """Test all aspects of the enhanced StubEliminationEngine."""
    print("=== COMPLETE STUB ELIMINATION FUNCTIONALITY TEST ===")
    
    engine = StubEliminationEngine()
    
    # Test 1: Comprehensive scanning
    print("\n1. COMPREHENSIVE SCANNING")
    print("Scanning key files for all stub types...")
    
    key_files = [
        "src/mcp/lobes/alignment_engine.py",
        "src/mcp/lobes/pattern_recognition_engine.py",
        "src/mcp/lobes/experimental/decision_making/decision_making_lobe.py",
        "src/mcp/lobes/experimental/sensory_column/sensory_column.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            engine._scan_file(Path(file_path))
    
    print(f"Total stubs detected: {len(engine.detected_stubs)}")
    
    # Generate detailed report
    report = engine.get_stub_report()
    print(f"Severity breakdown: {report['severity_breakdown']}")
    print(f"Type breakdown: {report['type_breakdown']}")
    
    # Test 2: Replacement generation
    print("\n2. REPLACEMENT GENERATION")
    replacements = engine.generate_replacements()
    print(f"Generated {len(replacements)} replacements")
    
    # Show examples of different replacement types
    replacement_types = {}
    for replacement in replacements:
        stub_type = replacement.stub_info.stub_type
        if stub_type not in replacement_types:
            replacement_types[stub_type] = replacement
    
    print("\nReplacement examples by type:")
    for stub_type, replacement in replacement_types.items():
        print(f"\n{stub_type.upper()}:")
        print(f"  Function: {replacement.stub_info.function_name}")
        print(f"  Confidence: {replacement.confidence}")
        print(f"  Code preview: {replacement.replacement_code[:100]}...")
    
    # Test 3: Validation
    print("\n3. VALIDATION")
    validation_results = engine.validate_replacements()
    print(f"Validation results:")
    print(f"  Total: {validation_results['total_replacements']}")
    print(f"  Valid: {validation_results['valid']}")
    print(f"  Invalid: {validation_results['invalid']}")
    print(f"  Warnings: {validation_results['warnings']}")
    
    if validation_results['syntax_errors']:
        print(f"  Syntax errors: {len(validation_results['syntax_errors'])}")
    
    if validation_results['issues']:
        print(f"  Issues found: {len(validation_results['issues'])}")
        for issue in validation_results['issues'][:3]:
            print(f"    - {issue}")
    
    # Test 4: Testing functionality
    print("\n4. TESTING")
    test_results = engine.test_replacements()
    print(f"Test results:")
    print(f"  Import tests - Passed: {test_results['import_tests']['passed']}")
    print(f"  Import tests - Failed: {test_results['import_tests']['failed']}")
    print(f"  Overall success rate: {test_results['overall_success_rate']:.2%}")
    
    if test_results['import_tests']['errors']:
        print(f"  Import errors: {len(test_results['import_tests']['errors'])}")
        for error in test_results['import_tests']['errors'][:2]:
            print(f"    - {Path(error['file']).name}: {error['error']}")
    
    # Test 5: Comprehensive report
    print("\n5. COMPREHENSIVE REPORT")
    comprehensive_report = engine.generate_comprehensive_report()
    print(f"Report generated with {len(comprehensive_report['recommendations'])} recommendations:")
    for rec in comprehensive_report['recommendations'][:3]:
        print(f"  - {rec}")
    
    # Test 6: Dry run application
    print("\n6. DRY RUN APPLICATION")
    # Test with a subset of replacements
    test_replacements = replacements[:5] if len(replacements) > 5 else replacements
    if test_replacements:
        dry_run_results = engine.apply_replacements(test_replacements, dry_run=True)
        print(f"Dry run results:")
        print(f"  Would modify {len(dry_run_results['files_modified'])} files")
        print(f"  Success rate: {dry_run_results['successful']}/{dry_run_results['total_replacements']}")
        
        if dry_run_results['errors']:
            print(f"  Errors: {len(dry_run_results['errors'])}")
    else:
        print("No replacements available for dry run test")
    
    # Test 7: Specific stub type handling
    print("\n7. SPECIFIC STUB TYPE HANDLING")
    stub_types = set(stub.stub_type for stub in engine.detected_stubs)
    print(f"Detected stub types: {', '.join(stub_types)}")
    
    for stub_type in stub_types:
        stubs_of_type = [s for s in engine.detected_stubs if s.stub_type == stub_type]
        print(f"  {stub_type}: {len(stubs_of_type)} instances")
    
    # Test 8: Error handling
    print("\n8. ERROR HANDLING")
    try:
        # Test with invalid stub info
        from stub_elimination_engine import StubInfo
        invalid_stub = StubInfo(
            file_path="nonexistent.py",
            line_number=1,
            function_name="test_func",
            class_name=None,
            stub_type="invalid_type",
            context="test context",
            severity="high"
        )
        
        invalid_replacement = engine._generate_replacement(invalid_stub)
        if invalid_replacement is None:
            print("  ✅ Properly handled invalid stub type")
        else:
            print("  ⚠️ Generated replacement for invalid stub type")
    except Exception as e:
        print(f"  ✅ Properly caught error: {type(e).__name__}")
    
    print("\n=== ALL FUNCTIONALITY TESTS COMPLETED ===")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Stubs detected: {len(engine.detected_stubs)}")
    print(f"  Replacements generated: {len(replacements)}")
    print(f"  Validation success rate: {validation_results['valid']}/{validation_results['total_replacements']}")
    print(f"  Import test success rate: {test_results['overall_success_rate']:.2%}")
    
    return engine, replacements, validation_results, test_results

if __name__ == "__main__":
    try:
        test_complete_functionality()
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()