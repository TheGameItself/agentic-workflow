#!/usr/bin/env python3
"""
Enhanced Search Tool with Standards Enforcement
@{CORE.PFSUS.CLI.SEARCH.001} Advanced search with automatic PFSUS standards enforcement.
#{search,pfsus,standards,enforcement,automation,lambda_operators}
λ(ℵ(Δ(β(enhanced_search_enforcement))))
"""

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSearchEnforcer:
    """
    Enhanced search tool with automatic PFSUS standards enforcement.
    Combines powerful search capabilities with real-time standards validation.
    """
    
    # PFSUS compliance patterns
    COMPLIANCE_PATTERNS = {
        'lambda_operators': r'[λℵΔτiβΩ]:[\w_]+\(',
        'self_reference': r'τ:self_reference\(',
        'visual_meta': r'@\{visual-meta-start\}',
        'mmcp_footer': r'%% MMCP-FOOTER:',
        'pfsus_header': r'@\{.*\}.*PFSUS',
        'dependency_refs': r'# Dependencies: @\{',
        'file_format_compliance': r'file_format.*v\d+\.\d+\.\d+',
        'canonical_address': r'canonical_address:"[\w-]+"'
    }
    
    # Search enhancement patterns
    SEARCH_ENHANCEMENTS = {
        'function_definitions': r'def\s+(\w+)\s*\(',
        'class_definitions': r'class\s+(\w+)\s*[\(:]',
        'import_statements': r'(?:from\s+[\w.]+\s+)?import\s+([\w.,\s*]+)',
        'todo_comments': r'(?i)(?:todo|fixme|hack|xxx)[:]\s*(.+)',
        'lambda_usage': r'([λℵΔτiβΩ]):(\w+)\(([^)]*)\)',
        'pfsus_violations': r'(?!.*[λℵΔτiβΩ]:)^(#{1,6})\s+(.+)$',
        'missing_docstrings': r'^(def|class)\s+\w+.*:\s*$',
        'long_lines': r'^.{120,}$'
    }
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root)
        self.search_results = []
        self.enforcement_results = []
        
    async def enhanced_search(self, 
                            query: str, 
                            file_pattern: str = "*",
                            enforce_standards: bool = True,
                            auto_fix: bool = False) -> Dict[str, Any]:
        """
        Enhanced search with optional standards enforcement.
        
        Args:
            query: Search query (regex pattern)
            file_pattern: File pattern to search in
            enforce_standards: Whether to check PFSUS compliance
            auto_fix: Whether to automatically fix violations
        """
        logger.info(f"🔍 Enhanced search: '{query}' in '{file_pattern}'")
        
        results = {
            'matches': [],
            'compliance_issues': [],
            'fixes_applied': [],
            'statistics': {}
        }
        
        # Perform search
        search_matches = await self._perform_search(query, file_pattern)
        results['matches'] = search_matches
        
        # Enforce standards if requested
        if enforce_standards:
            compliance_issues = await self._check_compliance(search_matches)
            results['compliance_issues'] = compliance_issues
            
            if auto_fix:
                fixes = await self._apply_fixes(compliance_issues)
                results['fixes_applied'] = fixes
        
        # Generate statistics
        results['statistics'] = self._generate_statistics(results)
        
        return results
    
    async def _perform_search(self, query: str, file_pattern: str) -> List[Dict]:
        """Perform the actual search operation."""
        matches = []
        
        # Use ripgrep if available, otherwise fallback to Python regex
        try:
            matches = await self._ripgrep_search(query, file_pattern)
        except (FileNotFoundError, subprocess.CalledProcessError):
            matches = await self._python_regex_search(query, file_pattern)
        
        return matches
    
    async def _ripgrep_search(self, query: str, file_pattern: str) -> List[Dict]:
        """Use ripgrep for fast searching."""
        cmd = [
            'rg', 
            '--json',
            '--context', '2',
            '--glob', file_pattern,
            query,
            str(self.workspace_root)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0 and process.returncode != 1:  # 1 means no matches
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        matches = []
        for line in stdout.decode().strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if data['type'] == 'match':
                        matches.append({
                            'file': data['data']['path']['text'],
                            'line_number': data['data']['line_number'],
                            'line_text': data['data']['lines']['text'],
                            'match_text': data['data']['submatches'][0]['match']['text'] if data['data']['submatches'] else '',
                            'context_before': [],
                            'context_after': []
                        })
                except json.JSONDecodeError:
                    continue
        
        return matches
    
    async def _python_regex_search(self, query: str, file_pattern: str) -> List[Dict]:
        """Fallback Python regex search."""
        matches = []
        pattern = re.compile(query, re.MULTILINE | re.IGNORECASE)
        
        for file_path in self.workspace_root.rglob(file_pattern):
            if file_path.is_file() and not self._should_skip_file(file_path):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        match = pattern.search(line)
                        if match:
                            matches.append({
                                'file': str(file_path.relative_to(self.workspace_root)),
                                'line_number': line_num,
                                'line_text': line,
                                'match_text': match.group(0),
                                'context_before': lines[max(0, line_num-3):line_num-1],
                                'context_after': lines[line_num:min(len(lines), line_num+2)]
                            })
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        return matches
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if file should be skipped."""
        skip_patterns = [
            r'\.git/',
            r'__pycache__/',
            r'\.vscode/',
            r'\.idea/',
            r'node_modules/',
            r'build/',
            r'dist/',
            r'\.pyc$',
            r'\.log$'
        ]
        
        file_str = str(file_path)
        return any(re.search(pattern, file_str) for pattern in skip_patterns)
    
    async def _check_compliance(self, matches: List[Dict]) -> List[Dict]:
        """Check PFSUS compliance for matched files."""
        compliance_issues = []
        
        # Get unique files from matches
        files_to_check = set(match['file'] for match in matches)
        
        for file_path in files_to_check:
            full_path = self.workspace_root / file_path
            if full_path.exists():
                issues = await self._analyze_file_compliance(full_path)
                compliance_issues.extend(issues)
        
        return compliance_issues
    
    async def _analyze_file_compliance(self, file_path: Path) -> List[Dict]:
        """Analyze individual file for PFSUS compliance."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            return issues
        
        # Check for missing lambda operators in headers
        if file_path.suffix == '.md':
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            for level, header_text in headers:
                if not re.search(r'[λℵΔτiβΩ]:', header_text):
                    issues.append({
                        'file': str(file_path.relative_to(self.workspace_root)),
                        'type': 'missing_lambda_operator',
                        'line': header_text,
                        'suggestion': self._suggest_lambda_operator(header_text),
                        'severity': 'medium'
                    })
        
        # Check for missing PFSUS elements
        for pattern_name, pattern in self.COMPLIANCE_PATTERNS.items():
            if not re.search(pattern, content, re.MULTILINE | re.DOTALL):
                if self._should_have_pattern(file_path, pattern_name):
                    issues.append({
                        'file': str(file_path.relative_to(self.workspace_root)),
                        'type': 'missing_pfsus_element',
                        'element': pattern_name,
                        'severity': 'high' if pattern_name in ['self_reference', 'mmcp_footer'] else 'medium'
                    })
        
        # Check for code quality issues
        if file_path.suffix == '.py':
            code_issues = self._check_python_quality(content, file_path)
            issues.extend(code_issues)
        
        return issues
    
    def _suggest_lambda_operator(self, header_text: str) -> str:
        """Suggest appropriate lambda operator for header."""
        header_lower = header_text.lower()
        
        # Memory/storage operations
        if any(word in header_lower for word in ['memory', 'storage', 'data', 'cache']):
            return f'ℵ:{header_text.lower().replace(" ", "_")}'
        
        # Workflow/orchestration
        if any(word in header_lower for word in ['workflow', 'process', 'step', 'phase']):
            return f'Δ:{header_text.lower().replace(" ", "_")}'
        
        # Time/runtime
        if any(word in header_lower for word in ['time', 'runtime', 'execution', 'schedule']):
            return f'τ:{header_text.lower().replace(" ", "_")}'
        
        # Improvement/optimization
        if any(word in header_lower for word in ['improve', 'optimize', 'enhance', 'upgrade']):
            return f'i:{header_text.lower().replace(" ", "_")}'
        
        # Monitoring/validation
        if any(word in header_lower for word in ['monitor', 'validate', 'check', 'test']):
            return f'β:{header_text.lower().replace(" ", "_")}'
        
        # System/foundational
        if any(word in header_lower for word in ['system', 'core', 'foundation', 'setup']):
            return f'Ω:{header_text.lower().replace(" ", "_")}'
        
        # Default to lambda
        return f'λ:{header_text.lower().replace(" ", "_")}'
    
    def _should_have_pattern(self, file_path: Path, pattern_name: str) -> bool:
        """Determine if file should have specific PFSUS pattern."""
        file_type_requirements = {
            '.md': ['self_reference', 'mmcp_footer', 'visual_meta'],
            '.py': ['pfsus_header', 'dependency_refs'],
            '.mmd': ['lambda_operators', 'canonical_address'],
            '.json': ['file_format_compliance']
        }
        
        required_patterns = file_type_requirements.get(file_path.suffix, [])
        return pattern_name in required_patterns
    
    def _check_python_quality(self, content: str, file_path: Path) -> List[Dict]:
        """Check Python code quality issues."""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                issues.append({
                    'file': str(file_path.relative_to(self.workspace_root)),
                    'type': 'code_quality',
                    'issue': 'line_too_long',
                    'line_number': line_num,
                    'line': line[:100] + '...' if len(line) > 100 else line,
                    'severity': 'low'
                })
            
            # Check for missing docstrings
            if re.match(r'^\s*(def|class)\s+\w+.*:\s*$', line):
                next_line = lines[line_num] if line_num < len(lines) else ''
                if not next_line.strip().startswith('"""') and not next_line.strip().startswith("'''"):
                    issues.append({
                        'file': str(file_path.relative_to(self.workspace_root)),
                        'type': 'code_quality',
                        'issue': 'missing_docstring',
                        'line_number': line_num,
                        'line': line.strip(),
                        'severity': 'medium'
                    })
        
        return issues
    
    async def _apply_fixes(self, compliance_issues: List[Dict]) -> List[Dict]:
        """Apply automatic fixes for compliance issues."""
        fixes_applied = []
        
        # Group issues by file
        issues_by_file = {}
        for issue in compliance_issues:
            file_path = issue['file']
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)
        
        # Apply fixes file by file
        for file_path, issues in issues_by_file.items():
            full_path = self.workspace_root / file_path
            fixes = await self._fix_file_issues(full_path, issues)
            fixes_applied.extend(fixes)
        
        return fixes_applied
    
    async def _fix_file_issues(self, file_path: Path, issues: List[Dict]) -> List[Dict]:
        """Fix issues in a specific file."""
        fixes = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
        except (UnicodeDecodeError, PermissionError):
            return fixes
        
        # Apply fixes based on issue type
        for issue in issues:
            if issue['type'] == 'missing_lambda_operator':
                # Fix missing lambda operators in headers
                old_header = issue['line']
                new_header = f"## {issue['suggestion']}"
                content = content.replace(f"## {old_header}", new_header)
                
                fixes.append({
                    'file': str(file_path.relative_to(self.workspace_root)),
                    'type': 'lambda_operator_added',
                    'old': old_header,
                    'new': issue['suggestion']
                })
            
            elif issue['type'] == 'missing_pfsus_element':
                # Add missing PFSUS elements
                if issue['element'] == 'self_reference':
                    self_ref = self._generate_self_reference(file_path, content)
                    content += f'\n{self_ref}'
                    
                    fixes.append({
                        'file': str(file_path.relative_to(self.workspace_root)),
                        'type': 'self_reference_added',
                        'content': self_ref[:100] + '...'
                    })
                
                elif issue['element'] == 'mmcp_footer':
                    footer = self._generate_mmcp_footer(file_path)
                    content += f'\n{footer}'
                    
                    fixes.append({
                        'file': str(file_path.relative_to(self.workspace_root)),
                        'type': 'mmcp_footer_added',
                        'content': footer
                    })
        
        # Write changes if content was modified
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
        
        return fixes
    
    def _generate_self_reference(self, file_path: Path, content: str) -> str:
        """Generate self-reference block for file."""
        import hashlib
        
        file_name = file_path.name
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        self_ref = f'## τ:self_reference({file_name.replace(".", "_")}_metadata)\n'
        self_ref += f'{{type:Documentation, file:"{file_name}", version:"1.0.0", '
        self_ref += f'checksum:"sha256:{checksum}", '
        self_ref += f'canonical_address:"{file_name.replace(".", "-")}", '
        self_ref += f'pfsus_compliant:true, lambda_operators:true}}'
        
        return self_ref
    
    def _generate_mmcp_footer(self, file_path: Path) -> str:
        """Generate MMCP footer for file."""
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        footer = f'%% MMCP-FOOTER: version=1.0.0; timestamp={timestamp}; '
        footer += f'author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; '
        footer += f'file_format={file_path.suffix[1:]}.{file_path.stem.split(".")[-1]}.v1.0.0.md'
        
        return footer
    
    def _generate_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate search and compliance statistics."""
        stats = {
            'total_matches': len(results['matches']),
            'files_with_matches': len(set(match['file'] for match in results['matches'])),
            'compliance_issues': len(results['compliance_issues']),
            'fixes_applied': len(results['fixes_applied']),
            'issue_types': {},
            'severity_breakdown': {}
        }
        
        # Analyze issue types
        for issue in results['compliance_issues']:
            issue_type = issue['type']
            stats['issue_types'][issue_type] = stats['issue_types'].get(issue_type, 0) + 1
            
            severity = issue.get('severity', 'medium')
            stats['severity_breakdown'][severity] = stats['severity_breakdown'].get(severity, 0) + 1
        
        return stats
    
    def format_results(self, results: Dict[str, Any], format_type: str = 'text') -> str:
        """Format search results for display."""
        if format_type == 'json':
            return json.dumps(results, indent=2)
        
        # Text format
        output = []
        output.append("# Enhanced Search Results with Standards Enforcement")
        output.append(f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        # Statistics
        stats = results['statistics']
        output.append("### λ:search_statistics(comprehensive_metrics)")
        output.append(f"- **Total Matches**: {stats['total_matches']}")
        output.append(f"- **Files with Matches**: {stats['files_with_matches']}")
        output.append(f"- **Compliance Issues**: {stats['compliance_issues']}")
        output.append(f"- **Fixes Applied**: {stats['fixes_applied']}")
        output.append("")
        
        # Search matches
        if results['matches']:
            output.append("### ℵ:search_matches(detailed_results)")
            for match in results['matches']:
                output.append(f"#### {match['file']}:{match['line_number']}")
                output.append(f"```")
                output.append(match['line_text'])
                output.append(f"```")
                output.append("")
        
        # Compliance issues
        if results['compliance_issues']:
            output.append("### β:compliance_issues(standards_violations)")
            for issue in results['compliance_issues']:
                output.append(f"#### {issue['file']}")
                output.append(f"- **Type**: {issue['type']}")
                output.append(f"- **Severity**: {issue.get('severity', 'medium')}")
                if 'suggestion' in issue:
                    output.append(f"- **Suggestion**: {issue['suggestion']}")
                output.append("")
        
        # Fixes applied
        if results['fixes_applied']:
            output.append("### i:fixes_applied(automatic_corrections)")
            for fix in results['fixes_applied']:
                output.append(f"#### {fix['file']}")
                output.append(f"- **Type**: {fix['type']}")
                if 'old' in fix and 'new' in fix:
                    output.append(f"- **Changed**: `{fix['old']}` → `{fix['new']}`")
                output.append("")
        
        return '\n'.join(output)

async def main():
    """Main CLI interface for enhanced search with standards enforcement."""
    parser = argparse.ArgumentParser(description='Enhanced Search with Standards Enforcement')
    parser.add_argument('query', help='Search query (regex pattern)')
    parser.add_argument('--workspace', '-w', default='.', help='Workspace root directory')
    parser.add_argument('--pattern', '-p', default='*', help='File pattern to search')
    parser.add_argument('--enforce', '-e', action='store_true', help='Enforce PFSUS standards')
    parser.add_argument('--fix', '-f', action='store_true', help='Auto-fix violations')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    parser.add_argument('--output', '-o', help='Output file')
    
    args = parser.parse_args()
    
    searcher = EnhancedSearchEnforcer(args.workspace)
    
    # Perform enhanced search
    results = await searcher.enhanced_search(
        query=args.query,
        file_pattern=args.pattern,
        enforce_standards=args.enforce,
        auto_fix=args.fix
    )
    
    # Format and output results
    formatted_results = searcher.format_results(results, args.format)
    
    if args.output:
        Path(args.output).write_text(formatted_results, encoding='utf-8')
        print(f"Results saved to: {args.output}")
    else:
        print(formatted_results)

if __name__ == "__main__":
    asyncio.run(main())

# λ(ℵ(Δ(β(enhanced_search_enforcement_complete)))) Processing complete
# Version: 1.0.0 | Last Modified: 2025-07-22T00:00:00Z
# Dependencies: @{CORE.PFSUS.STANDARD.001, CORE.SEARCH.001, CORE.UTILS.001}
# Related: @{CORE.TESTS.SEARCH.001, CORE.DOCS.SEARCH.001}