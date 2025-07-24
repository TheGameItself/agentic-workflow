#!/usr/bin/env python3
"""
Test runner for the MCP system.

This script runs all tests and generates coverage reports.

βtest_runner(comprehensive_execution)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run MCP system tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel processes")
    
    args = parser.parse_args()
    
    # Change to the test directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Determine which tests to run
    test_patterns = []
    
    if args.unit:
        test_patterns.extend([
            "test_memory_manager.py",
            "test_workflow_engine.py", 
            "test_context_manager.py",
            "test_database_manager.py",
            "test_memory_lobe.py",
            "test_base_lobe.py"
        ])
    elif args.integration:
        test_patterns.append("test_integration.py")
    elif args.performance:
        test_patterns.append("test_performance.py")
    else:
        # Run all tests by default
        test_patterns = ["."]
    
    success = True
    
    # Run tests
    for pattern in test_patterns:
        cmd = pytest_cmd + [pattern]
        if not run_command(cmd, f"Tests: {pattern}"):
            success = False
    
    # Generate coverage report if requested
    if args.coverage:
        coverage_cmd = [
            "python", "-m", "pytest",
            "--cov=core.src.mcp",
            "--cov-report=html",
            "--cov-report=term",
            "--cov-report=xml",
            "."
        ]
        if not run_command(coverage_cmd, "Coverage Report"):
            success = False
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("💥 Some tests failed. Check the output above.")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())