#!/usr/bin/env python3
"""
Test script for Core Infrastructure and Stub Elimination components.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.stub_elimination_engine import StubEliminationEngine, scan_project_for_stubs
from mcp.implementation_validator import ImplementationValidator, validate_mcp_system
from mcp.fallback_manager import FallbackManager, handle_error_with_fallback


def test_stub_elimination_engine():
    """Test the StubEliminationEngine."""
    print("=" * 50)
    print("Testing StubEliminationEngine")
    print("=" * 50)
    
    try:
        # Test basic initialization
        engine = StubEliminationEngine()
        print("✓ StubEliminationEngine initialized successfully")
        
        # Test stub scanning (limited scope to avoid long execution)
        stubs = engine.scan_for_stubs([str(Path("src/mcp").resolve())])
        print(f"✓ Scanned for stubs, found {len(stubs)} potential issues")
        
        # Test report generation
        report = engine.get_stub_report()
        print(f"✓ Generated stub report: {report.get('total_stubs', 0)} total stubs")
        
        # Test convenience function
        project_stubs = scan_project_for_stubs(str(Path(".").resolve()))
        print(f"✓ Convenience function works: found {len(project_stubs)} stubs")
        
        return True
        
    except Exception as e:
        print(f"✗ StubEliminationEngine test failed: {e}")
        return False


def test_implementation_validator():
    """Test the ImplementationValidator."""
    print("\n" + "=" * 50)
    print("Testing ImplementationValidator")
    print("=" * 50)
    
    try:
        # Test basic initialization
        validator = ImplementationValidator()
        print("✓ ImplementationValidator initialized successfully")
        
        # Test module discovery
        modules = validator._discover_modules()
        print(f"✓ Discovered {len(modules)} modules")
        
        # Test validation config loading
        config = validator.validation_config
        print(f"✓ Loaded validation config with {len(config.get('validation_checks', []))} checks")
        
        # Test stub detection in production
        no_stubs = validator.ensure_no_stubs_in_production()
        print(f"✓ Production stub check: {'No critical stubs' if no_stubs else 'Found critical stubs'}")
        
        # Test convenience function
        system_validation = validate_mcp_system(str(Path(".").resolve()))
        print(f"✓ System validation completed: {system_validation.overall_completion:.1f}% complete")
        
        return True
        
    except Exception as e:
        print(f"✗ ImplementationValidator test failed: {e}")
        return False


async def test_fallback_manager():
    """Test the FallbackManager."""
    print("\n" + "=" * 50)
    print("Testing FallbackManager")
    print("=" * 50)
    
    try:
        # Test basic initialization
        manager = FallbackManager()
        print("✓ FallbackManager initialized successfully")
        
        # Test error statistics
        stats = manager.get_error_statistics()
        print(f"✓ Error statistics retrieved: {stats.get('total_errors', 0)} total errors")
        
        # Test fallback registry
        rules_count = len(manager.fallback_registry.rules)
        print(f"✓ Fallback registry has {rules_count} rules")
        
        # Test error handling with a simple error
        test_error = ValueError("Test error for fallback")
        test_context = {
            'function_name': 'test_function',
            'module_name': 'test_module',
            'args': (),
            'kwargs': {}
        }
        
        result = await manager.handle_error(test_error, test_context)
        print(f"✓ Error handling test completed: {type(result)}")
        
        # Test convenience function
        result2 = await handle_error_with_fallback(test_error, test_context)
        print(f"✓ Convenience function works: {type(result2)}")
        
        # Test health assessment
        health = manager._assess_system_health()
        print(f"✓ System health: {health['status']} (score: {health['score']})")
        
        return True
        
    except Exception as e:
        print(f"✗ FallbackManager test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("\n" + "=" * 50)
    print("Testing Component Integration")
    print("=" * 50)
    
    try:
        # Test that validator can use stub engine
        validator = ImplementationValidator()
        stub_engine = validator.stub_engine
        print("✓ Validator integrates with StubEliminationEngine")
        
        # Test that components can work together
        stubs = stub_engine.scan_for_stubs([str(Path("src/mcp").resolve())])
        if stubs:
            print(f"✓ Found {len(stubs)} stubs that could be handled by fallback system")
        else:
            print("✓ No stubs found - system appears well-implemented")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Core Infrastructure and Stub Elimination Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(test_stub_elimination_engine())
    results.append(test_implementation_validator())
    results.append(await test_fallback_manager())
    results.append(test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Core infrastructure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)