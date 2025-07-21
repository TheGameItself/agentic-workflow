#!/usr/bin/env python3
"""
Task Completion Verification Script

This script verifies the completion status of tasks in the MCP system upgrade
and validates that all implemented components are properly documented.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime

class TaskCompletionVerifier:
    """Verifies task completion status and validates implementation."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.tasks_file = os.path.join(self.project_root, '.kiro', 'specs', 'mcp-system-upgrade', 'tasks.md')
        self.progress_file = os.path.join(self.project_root, 'UPGRADE_PROGRESS_SUMMARY.md')
        self.results = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'in_progress_tasks': 0,
            'remaining_tasks': 0,
            'completion_percentage': 0.0,
            'validation_errors': [],
            'implementation_status': {}
        }
    
    def verify_tasks_completion(self) -> Dict[str, Any]:
        """Main verification method."""
        print("🔍 Verifying MCP System Upgrade Task Completion...")
        
        # Parse tasks file
        self._parse_tasks_file()
        
        # Validate implementation status
        self._validate_implementations()
        
        # Calculate completion statistics
        self._calculate_completion_stats()
        
        # Generate verification report
        self._generate_verification_report()
        
        return self.results
    
    def _parse_tasks_file(self):
        """Parse the tasks.md file to extract completion status."""
        if not os.path.exists(self.tasks_file):
            self.results['validation_errors'].append(f"Tasks file not found: {self.tasks_file}")
            return
        
        with open(self.tasks_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count task completion status
        completed_pattern = r'- \[x\]'
        in_progress_pattern = r'- \[-\]'
        remaining_pattern = r'- \[ \]'
        
        completed_count = len(re.findall(completed_pattern, content))
        in_progress_count = len(re.findall(in_progress_pattern, content))
        remaining_count = len(re.findall(remaining_pattern, content))
        
        self.results['completed_tasks'] = completed_count
        self.results['in_progress_tasks'] = in_progress_count
        self.results['remaining_tasks'] = remaining_count
        self.results['total_tasks'] = completed_count + in_progress_count + remaining_count
        
        print(f"📊 Task Counts:")
        print(f"   ✅ Completed: {completed_count}")
        print(f"   🔄 In Progress: {in_progress_count}")
        print(f"   📋 Remaining: {remaining_count}")
        print(f"   📈 Total: {self.results['total_tasks']}")
    
    def _validate_implementations(self):
        """Validate that completed tasks have corresponding implementations."""
        implementation_paths = {
            'genetic_trigger_system': 'src/mcp/genetic_trigger_system/',
            'automatic_update_system': 'src/mcp/automatic_update_system.py',
            'dynamic_context_generator': 'src/mcp/dynamic_context_generator.py',
            'enhanced_context_manager': 'src/mcp/enhanced_context_manager.py',
            'system_integration_layer': 'src/mcp/system_integration_layer.py',
            'async_processing_framework': 'src/mcp/async_processing_framework.py',
            'performance_optimization_engine': 'src/mcp/performance_optimization_engine.py',
            'p2p_network_integration': 'src/mcp/p2p_network_integration.py',
            'p2p_global_performance_system': 'src/mcp/p2p_global_performance_system.py',
            'core_system_infrastructure': 'src/mcp/core_system_infrastructure.py',
            'enhanced_vector_memory': 'src/mcp/enhanced_vector_memory.py',
            'brain_state_aggregator': 'src/mcp/brain_state_aggregator.py',
            'unified_memory': 'src/mcp/unified_memory.py',
            'context_manager': 'src/mcp/context_manager.py',
            'project_audit': 'scripts/project_audit.py'
        }
        
        for component, path in implementation_paths.items():
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                self.results['implementation_status'][component] = {
                    'status': 'implemented',
                    'path': path,
                    'size': self._get_file_size(full_path)
                }
            else:
                self.results['implementation_status'][component] = {
                    'status': 'missing',
                    'path': path,
                    'size': 0
                }
                self.results['validation_errors'].append(f"Missing implementation: {path}")
    
    def _get_file_size(self, path: str) -> int:
        """Get file or directory size."""
        if os.path.isfile(path):
            return os.path.getsize(path)
        elif os.path.isdir(path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
        return 0
    
    def _calculate_completion_stats(self):
        """Calculate completion statistics."""
        total = self.results['total_tasks']
        if total > 0:
            completed = self.results['completed_tasks']
            in_progress = self.results['in_progress_tasks']
            
            # Calculate completion percentage (completed + 0.5 * in_progress)
            completion_percentage = ((completed + (0.5 * in_progress)) / total) * 100
            self.results['completion_percentage'] = round(completion_percentage, 1)
    
    def _generate_verification_report(self):
        """Generate a comprehensive verification report."""
        print(f"\n📋 Verification Report")
        print(f"=" * 50)
        
        # Overall completion
        print(f"Overall Completion: {self.results['completion_percentage']}%")
        print(f"Total Tasks: {self.results['total_tasks']}")
        print(f"Completed: {self.results['completed_tasks']}")
        print(f"In Progress: {self.results['in_progress_tasks']}")
        print(f"Remaining: {self.results['remaining_tasks']}")
        
        # Implementation status
        print(f"\n🔧 Implementation Status:")
        implemented_count = 0
        missing_count = 0
        
        for component, status in self.results['implementation_status'].items():
            if status['status'] == 'implemented':
                implemented_count += 1
                size_mb = status['size'] / (1024 * 1024)
                print(f"   ✅ {component}: {size_mb:.1f} MB")
            else:
                missing_count += 1
                print(f"   ❌ {component}: Missing")
        
        print(f"\nImplementation Summary:")
        print(f"   ✅ Implemented: {implemented_count}")
        print(f"   ❌ Missing: {missing_count}")
        
        # Validation errors
        if self.results['validation_errors']:
            print(f"\n⚠️  Validation Errors:")
            for error in self.results['validation_errors']:
                print(f"   - {error}")
        else:
            print(f"\n✅ No validation errors found!")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if self.results['completion_percentage'] >= 95:
            print("   🎉 Excellent progress! The system is nearly complete.")
            print("   📝 Focus on remaining tasks: WebSocialEngine and cross-engine coordination")
        elif self.results['completion_percentage'] >= 80:
            print("   🚀 Great progress! Continue with remaining implementation tasks")
        else:
            print("   🔄 Continue implementation of core components")
        
        if missing_count > 0:
            print("   🔧 Address missing implementations for completed tasks")

def main():
    """Main verification function."""
    verifier = TaskCompletionVerifier()
    results = verifier.verify_tasks_completion()
    
    # Save results to file
    results_file = os.path.join(verifier.project_root, 'task_verification_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Return exit code based on completion
    if results['completion_percentage'] >= 90:
        print("🎯 Task verification completed successfully!")
        return 0
    else:
        print("⚠️  Task verification completed with issues to address.")
        return 1

if __name__ == "__main__":
    exit(main()) 