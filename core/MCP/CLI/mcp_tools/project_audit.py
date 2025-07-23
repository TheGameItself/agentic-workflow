#!/usr/bin/env python3
"""
Project Audit and Cleanup Script for MCP System

This script performs a comprehensive audit of the project to identify:
- Junk files (logs, temp files, backups)
- Obsolete or redundant files
- Stub implementations that need attention
- Code quality issues
- Project organization improvements

Based on the TODO list requirements and idea.txt specifications.
"""

import os
import re
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
import json

class ProjectAuditor:
    """Comprehensive project audit and cleanup system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.logger = self._setup_logging()
        self.audit_results = {
            'junk_files': [],
            'obsolete_files': [],
            'stub_implementations': [],
            'code_quality_issues': [],
            'organization_issues': [],
            'recommendations': []
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the audit system."""
        logger = logging.getLogger("project_auditor")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def audit_project(self) -> Dict[str, Any]:
        """Perform comprehensive project audit."""
        self.logger.info("Starting comprehensive project audit...")
        
        # Audit different aspects
        self._audit_junk_files()
        self._audit_obsolete_files()
        self._audit_stub_implementations()
        self._audit_code_quality()
        self._audit_project_organization()
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.audit_results
    
    def _audit_junk_files(self):
        """Identify junk files (logs, temp files, backups)."""
        self.logger.info("Auditing junk files...")
        
        junk_patterns = [
            '*.log', '*.tmp', '*.bak', '*.old', '*.swp', '*.swo',
            '*~', '.#*', '.DS_Store', 'Thumbs.db', '__pycache__',
            '*.pyc', '*.pyo', '.pytest_cache', '.coverage'
        ]
        
        for pattern in junk_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    self.audit_results['junk_files'].append({
                        'path': str(file_path),
                        'type': 'junk_file',
                        'size': file_path.stat().st_size,
                        'reason': f'Matches pattern: {pattern}'
                    })
                elif file_path.is_dir():
                    self.audit_results['junk_files'].append({
                        'path': str(file_path),
                        'type': 'junk_directory',
                        'size': 0,
                        'reason': f'Junk directory: {pattern}'
                    })
    
    def _audit_obsolete_files(self):
        """Identify obsolete or redundant files."""
        self.logger.info("Auditing obsolete files...")
        
        # Check for duplicate files
        file_hashes = {}
        for file_path in self.project_root.rglob('*.py'):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content_hash = hash(content)
                    if content_hash in file_hashes:
                        self.audit_results['obsolete_files'].append({
                            'path': str(file_path),
                            'type': 'duplicate',
                            'duplicate_of': file_hashes[content_hash],
                            'reason': 'Duplicate content'
                        })
                    else:
                        file_hashes[content_hash] = str(file_path)
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
        
        # Check for test files that might be obsolete
        for file_path in self.project_root.rglob('test_*.py'):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if 'TODO' in content or 'NotImplementedError' in content:
                        self.audit_results['obsolete_files'].append({
                            'path': str(file_path),
                            'type': 'incomplete_test',
                            'reason': 'Contains TODO or NotImplementedError'
                        })
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
    
    def _audit_stub_implementations(self):
        """Identify stub implementations that need attention."""
        self.logger.info("Auditing stub implementations...")
        
        stub_patterns = [
            (r'raise\s+NotImplementedError', 'NotImplementedError'),
            (r'def\s+\w+\([^)]*\):\s*\n\s*pass\s*$', 'pass_only_function'),
            (r'#\s*(TODO|FIXME|XXX|HACK)\b', 'todo_comment'),
            (r'"""Stub:.*?"""', 'stub_docstring'),
            (r'#\s*stub[:\s]|#.*stub.*implementation', 'stub_comment')
        ]
        
        for file_path in self.project_root.rglob('*.py'):
            if file_path.is_file() and 'test_venv' not in str(file_path):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    for pattern, stub_type in stub_patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            self.audit_results['stub_implementations'].append({
                                'path': str(file_path),
                                'line': line_num,
                                'type': stub_type,
                                'context': match.group(0)[:100],
                                'severity': 'high' if stub_type == 'NotImplementedError' else 'medium'
                            })
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
    
    def _audit_code_quality(self):
        """Audit code quality issues."""
        self.logger.info("Auditing code quality...")
        
        quality_patterns = [
            (r'def\s+\w+\([^)]*\):\s*\n\s*"""[^"]*"""\s*\n\s*$', 'empty_method_body'),
            (r'return\s+(None|{}|\[\]|""|\'\')?\s*#.*(placeholder|stub)', 'placeholder_return'),
            (r'print\s*\(', 'print_statement'),
            (r'import\s+\*', 'wildcard_import')
        ]
        
        for file_path in self.project_root.rglob('*.py'):
            if file_path.is_file() and 'test_venv' not in str(file_path):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    for pattern, issue_type in quality_patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            self.audit_results['code_quality_issues'].append({
                                'path': str(file_path),
                                'line': line_num,
                                'type': issue_type,
                                'context': match.group(0)[:100],
                                'severity': 'medium'
                            })
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")
    
    def _audit_project_organization(self):
        """Audit project organization and structure."""
        self.logger.info("Auditing project organization...")
        
        # Check for files in wrong locations
        misplaced_files = []
        for file_path in self.project_root.rglob('*.py'):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.project_root)
                path_parts = relative_path.parts
                
                # Check for Python files in root that should be in src/
                if len(path_parts) == 1 and path_parts[0].endswith('.py'):
                    if path_parts[0] not in ['setup.py', 'quick_test.py', 'verify_stub_elimination.py']:
                        misplaced_files.append(str(file_path))
                
                # Check for test files outside tests/ directory
                if 'test_' in path_parts[-1] and 'tests' not in path_parts:
                    misplaced_files.append(str(file_path))
        
        for file_path in misplaced_files:
            self.audit_results['organization_issues'].append({
                'path': file_path,
                'type': 'misplaced_file',
                'reason': 'File in unexpected location',
                'recommendation': 'Move to appropriate directory'
            })
    
    def _generate_recommendations(self):
        """Generate recommendations based on audit results."""
        self.logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Junk file recommendations
        if self.audit_results['junk_files']:
            recommendations.append({
                'priority': 'high',
                'category': 'cleanup',
                'action': 'Remove junk files',
                'description': f"Found {len(self.audit_results['junk_files'])} junk files to clean up",
                'files': [f['path'] for f in self.audit_results['junk_files']]
            })
        
        # Stub implementation recommendations
        critical_stubs = [s for s in self.audit_results['stub_implementations'] if s['severity'] == 'high']
        if critical_stubs:
            recommendations.append({
                'priority': 'critical',
                'category': 'implementation',
                'action': 'Fix critical stub implementations',
                'description': f"Found {len(critical_stubs)} critical stub implementations",
                'files': list(set([s['path'] for s in critical_stubs]))
            })
        
        # Code quality recommendations
        if self.audit_results['code_quality_issues']:
            recommendations.append({
                'priority': 'medium',
                'category': 'quality',
                'action': 'Improve code quality',
                'description': f"Found {len(self.audit_results['code_quality_issues'])} code quality issues",
                'files': list(set([i['path'] for i in self.audit_results['code_quality_issues']]))
            })
        
        # Organization recommendations
        if self.audit_results['organization_issues']:
            recommendations.append({
                'priority': 'medium',
                'category': 'organization',
                'action': 'Reorganize project structure',
                'description': f"Found {len(self.audit_results['organization_issues'])} organization issues",
                'files': [i['path'] for i in self.audit_results['organization_issues']]
            })
        
        self.audit_results['recommendations'] = recommendations
    
    def generate_report(self) -> str:
        """Generate a comprehensive audit report."""
        report = []
        report.append("=" * 80)
        report.append("MCP PROJECT AUDIT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Project Root: {self.project_root}")
        report.append("")
        
        # Summary
        total_issues = (
            len(self.audit_results['junk_files']) +
            len(self.audit_results['obsolete_files']) +
            len(self.audit_results['stub_implementations']) +
            len(self.audit_results['code_quality_issues']) +
            len(self.audit_results['organization_issues'])
        )
        
        report.append("SUMMARY:")
        report.append(f"  Total Issues Found: {total_issues}")
        report.append(f"  Junk Files: {len(self.audit_results['junk_files'])}")
        report.append(f"  Obsolete Files: {len(self.audit_results['obsolete_files'])}")
        report.append(f"  Stub Implementations: {len(self.audit_results['stub_implementations'])}")
        report.append(f"  Code Quality Issues: {len(self.audit_results['code_quality_issues'])}")
        report.append(f"  Organization Issues: {len(self.audit_results['organization_issues'])}")
        report.append("")
        
        # Detailed findings
        for category, items in self.audit_results.items():
            if items and category != 'recommendations':
                report.append(f"{category.upper().replace('_', ' ')}:")
                for item in items[:10]:  # Show first 10 items
                    if isinstance(item, dict):
                        report.append(f"  - {item.get('path', str(item))}")
                if len(items) > 10:
                    report.append(f"  ... and {len(items) - 10} more")
                report.append("")
        
        # Recommendations
        if self.audit_results['recommendations']:
            report.append("RECOMMENDATIONS:")
            for rec in self.audit_results['recommendations']:
                report.append(f"  [{rec['priority'].upper()}] {rec['action']}")
                report.append(f"      {rec['description']}")
                report.append("")
        
        return "\n".join(report)
    
    def cleanup_junk_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up junk files identified in the audit."""
        self.logger.info(f"Cleaning up junk files (dry_run={dry_run})...")
        
        cleanup_results = {
            'removed_files': [],
            'removed_dirs': [],
            'errors': [],
            'total_size_freed': 0
        }
        
        for junk_item in self.audit_results['junk_files']:
            try:
                path = Path(junk_item['path'])
                if path.exists():
                    if path.is_file():
                        size = path.stat().st_size
                        if not dry_run:
                            path.unlink()
                        cleanup_results['removed_files'].append(str(path))
                        cleanup_results['total_size_freed'] += size
                    elif path.is_dir():
                        if not dry_run:
                            shutil.rmtree(path)
                        cleanup_results['removed_dirs'].append(str(path))
            except Exception as e:
                cleanup_results['errors'].append({
                    'path': junk_item['path'],
                    'error': str(e)
                })
        
        return cleanup_results

def main():
    """Main function to run the project audit."""
    auditor = ProjectAuditor()
    
    # Perform audit
    audit_results = auditor.audit_project()
    
    # Generate and print report
    report = auditor.generate_report()
    print(report)
    
    # Save report to file
    report_file = auditor.project_root / "project_audit_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Ask user if they want to clean up junk files
    if audit_results['junk_files']:
        print(f"\nFound {len(audit_results['junk_files'])} junk files.")
        response = input("Would you like to clean up junk files? (y/N): ")
        if response.lower() == 'y':
            cleanup_results = auditor.cleanup_junk_files(dry_run=False)
            print(f"Cleanup completed:")
            print(f"  Removed files: {len(cleanup_results['removed_files'])}")
            print(f"  Removed directories: {len(cleanup_results['removed_dirs'])}")
            print(f"  Total size freed: {cleanup_results['total_size_freed']} bytes")
            if cleanup_results['errors']:
                print(f"  Errors: {len(cleanup_results['errors'])}")

if __name__ == "__main__":
    main() 