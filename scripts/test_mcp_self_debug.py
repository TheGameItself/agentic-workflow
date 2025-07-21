#!/usr/bin/env python3
"""
MCP Self-Debugging Test Script
Comprehensive test script to demonstrate the MCP server's self-debugging capabilities.

This script runs various self-tests and demonstrates how the MCP server can:
1. Test its own components
2. Validate its configuration
3. Check database integrity
4. Monitor performance
5. Generate self-documentation
6. Update its own documentation

Usage:
    python scripts/test_mcp_self_debug.py [--full] [--save-reports] [--update-docs]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mcp.self_debug import MCPSelfDebugger
from mcp.server import MCPServer

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test MCP self-debugging capabilities')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive test')
    parser.add_argument('--save-reports', action='store_true', help='Save test reports to files')
    parser.add_argument('--update-docs', action='store_true', help='Update documentation')
    parser.add_argument('--component', help='Test specific component only')
    parser.add_argument('--format', choices=['text', 'json', 'detailed'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    print("🔍 MCP Self-Debugging Test Suite")
    print("=" * 50)
    
    # Initialize MCP server
    print("🚀 Initializing MCP server...")
    mcp_server = MCPServer()
    debugger = MCPSelfDebugger(mcp_server)
    
    if args.component:
        # Test specific component
        test_specific_component(debugger, mcp_server, args.component, args.format)
    elif args.full:
        # Run full comprehensive test
        run_full_test(debugger, args.save_reports, args.update_docs, args.format)
    else:
        # Run basic test
        run_basic_test(debugger, args.save_reports, args.format)

def test_specific_component(debugger, mcp_server, component_name, format):
    """Test a specific component."""
    print(f"🔧 Testing component: {component_name}")
    
    if not hasattr(mcp_server, component_name):
        print(f"❌ Component '{component_name}' not found")
        return
    
    component_obj = getattr(mcp_server, component_name)
    start_time = time.time()
    
    try:
        # Test component functionality
        if hasattr(component_obj, 'get_status'):
            status = component_obj.get_status()
        elif hasattr(component_obj, 'health_check'):
            status = component_obj.health_check()
        else:
            status = {"status": "available"}
        
        duration = time.time() - start_time
        health_status = "HEALTHY" if status.get("status") != "error" else "UNHEALTHY"
        
        result = {
            "component": component_name,
            "status": health_status,
            "response_time": duration,
            "details": status
        }
        
        print(f"✅ Component {component_name}: {health_status} ({duration:.3f}s)")
        
        if format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"   Details: {status}")
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Component {component_name} failed: {e}")
        print(f"   Duration: {duration:.3f}s")

def run_basic_test(debugger, save_reports, format):
    """Run basic self-test."""
    print("🧪 Running basic self-test...")
    
    # Test core components
    print("  Testing core components...")
    debugger._test_core_components()
    
    # Test database integrity
    print("  Testing database integrity...")
    debugger._test_database_integrity()
    
    # Test configuration
    print("  Testing configuration...")
    debugger._test_configuration()
    
    # Generate summary
    summary = {
        "timestamp": time.time(),
        "test_type": "basic",
        "components_tested": len(debugger.health_status),
        "healthy_components": len([h for h in debugger.health_status.values() if h.status == "HEALTHY"]),
        "test_results": len(debugger.test_results),
        "passed_tests": len([r for r in debugger.test_results if r.status == "PASS"]),
        "failed_tests": len([r for r in debugger.test_results if r.status == "FAIL"])
    }
    
    print(f"\n📊 Basic Test Summary:")
    print(f"  Components tested: {summary['components_tested']}")
    print(f"  Healthy components: {summary['healthy_components']}")
    print(f"  Tests passed: {summary['passed_tests']}")
    print(f"  Tests failed: {summary['failed_tests']}")
    
    if save_reports:
        report_file = debugger.save_test_report(summary)
        print(f"  Report saved to: {report_file}")
    
    if format == 'json':
        print(json.dumps(summary, indent=2, default=str))

def run_full_test(debugger, save_reports, update_docs, format):
    """Run full comprehensive test."""
    print("🧪 Running comprehensive self-test...")
    
    start_time = time.time()
    
    # Run all test categories
    test_categories = [
        ("Core Components", debugger._test_core_components),
        ("Database Integrity", debugger._test_database_integrity),
        ("Memory Systems", debugger._test_memory_systems),
        ("Performance Metrics", debugger._test_performance_metrics),
        ("API Endpoints", debugger._test_api_endpoints),
        ("Configuration", debugger._test_configuration),
        ("Security Features", debugger._test_security_features),
        ("Integration Points", debugger._test_integration_points),
        ("Documentation Sync", debugger._test_documentation_sync),
        ("Error Handling", debugger._test_error_handling)
    ]
    
    for category_name, test_func in test_categories:
        print(f"  Testing {category_name}...")
        try:
            test_func()
        except Exception as e:
            print(f"    ❌ {category_name} test failed: {e}")
    
    total_duration = time.time() - start_time
    
    # Generate comprehensive report
    report = debugger._generate_test_report(total_duration)
    
    # Update documentation if requested
    if update_docs:
        print("📚 Updating documentation...")
        if debugger.documentation_updates:
            debugger._apply_documentation_updates()
            print(f"  Applied {len(debugger.documentation_updates)} documentation updates")
        else:
            print("  No documentation updates needed")
    
    # Save reports if requested
    if save_reports:
        report_file = debugger.save_test_report(report)
        print(f"  Test report saved to: {report_file}")
        
        # Generate and save documentation
        print("  Generating self-documentation...")
        documentation = debugger.generate_self_documentation()
        docs_file = debugger.save_documentation(documentation)
        print(f"  Documentation saved to: {docs_file}")
    
    # Display results
    if format == 'json':
        print(json.dumps(report, indent=2, default=str))
    elif format == 'detailed':
        display_detailed_results(report)
    else:
        display_summary_results(report)

def display_summary_results(report):
    """Display summary of test results."""
    summary = report['summary']
    
    print(f"\n📊 Comprehensive Test Summary")
    print("=" * 50)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⚠️  Warnings: {summary['warnings']}")
    print(f"⏭️  Skipped: {summary['skipped']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Component Health: {summary['component_health_rate']:.1f}%")
    print(f"Average Response Time: {summary['avg_response_time']:.3f}s")
    print(f"Total Duration: {report['total_duration']:.2f}s")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

def display_detailed_results(report):
    """Display detailed test results."""
    display_summary_results(report)
    
    print(f"\n📋 Detailed Results:")
    print("=" * 50)
    
    for result in report['test_results']:
        status_icon = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARNING': '⚠️',
            'SKIP': '⏭️'
        }.get(result['status'], '❓')
        
        print(f"\n{status_icon} {result['test_name']}")
        print(f"   Status: {result['status']}")
        print(f"   Duration: {result['duration']:.3f}s")
        print(f"   Message: {result['message']}")
        
        if result.get('details'):
            print(f"   Details: {result['details']}")

def demonstrate_cli_commands():
    """Demonstrate CLI commands for self-debugging."""
    print("\n🔧 CLI Commands for Self-Debugging:")
    print("=" * 50)
    
    commands = [
        ("python mcp_cli.py self-test", "Run comprehensive self-test"),
        ("python mcp_cli.py self-test --save-report", "Run test and save report"),
        ("python mcp_cli.py self-test --include-docs", "Run test with documentation generation"),
        ("python mcp_cli.py health-check", "Check health of all components"),
        ("python mcp_cli.py health-check --component memory_manager", "Check specific component"),
        ("python mcp_cli.py generate-docs", "Generate self-documentation"),
        ("python mcp_cli.py generate-docs --save", "Generate and save documentation"),
        ("python mcp_cli.py check-docs", "Check for outdated documentation"),
        ("python mcp_cli.py check-docs --update", "Check and update outdated docs"),
    ]
    
    for command, description in commands:
        print(f"  {command}")
        print(f"    {description}")
        print()

if __name__ == '__main__':
    try:
        main()
        demonstrate_cli_commands()
        print("\n✅ Self-debugging test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 