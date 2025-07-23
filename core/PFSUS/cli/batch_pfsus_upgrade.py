#!/usr/bin/env python3
"""
PFSUS Batch Upgrade Tool
@{CORE.PFSUS.CLI.BATCH_UPGRADE.001} Automated batch upgrading of files to PFSUS v2.0.0 standards.
#{pfsus,batch,upgrade,standards,enforcement,automation,english_shorthand}
λbatch_upgrade(pfsus_standards_v2)
"""

import os
import re
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PFSUSBatchUpgrade:
    """
    Batch upgrade tool for PFSUS standards v2.0.0.
    Automatically applies English language shorthand and other improvements.
    """
    
    # Operator patterns for v1.4.0 to v2.0.0 migration
    OPERATOR_PATTERNS = {
        r'λ:(\w+)\(([^)]*)\)': r'λ\1(\2)',
        r'ℵ:(\w+)\(([^)]*)\)': r'ℵ\1(\2)',
        r'Δ:(\w+)\(([^)]*)\)': r'Δ\1(\2)',
        r'τ:(\w+)\(([^)]*)\)': r'τ\1(\2)',
        r'i:(\w+)\(([^)]*)\)': r'i\1(\2)',
        r'β:(\w+)\(([^)]*)\)': r'β\1(\2)',
        r'Ω:(\w+)\(([^)]*)\)': r'Ω\1(\2)'
    }
    
    # Common words for each operator category
    OPERATOR_WORDS = {
        'λ': ['function', 'process', 'transform', 'compute', 'calculate', 'execute', 'api', 'method', 
              'operation', 'algorithm', 'procedure', 'routine', 'handler', 'callback', 'interface'],
        'ℵ': ['memory', 'storage', 'data', 'cache', 'store', 'collection', 'database', 'repository', 
              'record', 'document', 'entity', 'object', 'instance', 'variable', 'container'],
        'Δ': ['workflow', 'process', 'transition', 'change', 'transform', 'orchestrate', 'coordinate', 
              'sequence', 'pipeline', 'flow', 'stream', 'step', 'phase', 'stage', 'operation'],
        'τ': ['time', 'runtime', 'schedule', 'execution', 'timing', 'duration', 'interval', 'period', 
              'frequency', 'cycle', 'timestamp', 'deadline', 'timeout', 'delay', 'latency'],
        'i': ['improve', 'optimize', 'enhance', 'upgrade', 'refine', 'boost', 'accelerate', 'streamline', 
              'efficiency', 'performance', 'quality', 'effectiveness', 'productivity', 'capability'],
        'β': ['validate', 'test', 'verify', 'check', 'monitor', 'inspect', 'examine', 'analyze', 
              'evaluate', 'assess', 'review', 'audit', 'probe', 'diagnose', 'scrutinize'],
        'Ω': ['system', 'core', 'foundation', 'base', 'framework', 'infrastructure', 'platform', 
              'architecture', 'structure', 'backbone', 'kernel', 'engine', 'root', 'initialize']
    }
    
    # File types to process
    FILE_PATTERNS = {
        r'.*\.md$': 'markdown',
        r'.*\.py$': 'python',
        r'.*\.mmcp\.mmd$': 'pfsus_standard',
        r'.*\.mmcp\.mdd$': 'pfsus_template',
        r'.*\.json$': 'json'
    }
    
    def __init__(self, workspace_root: str = ".", backup: bool = True):
        self.workspace_root = Path(workspace_root)
        self.backup = backup
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'patterns_applied': 0,
            'backups_created': 0,
            'errors': 0
        }
    
    async def batch_upgrade(self, file_pattern: str = "*", dry_run: bool = False) -> Dict[str, Any]:
        """Run batch upgrade on all matching files."""
        logger.info(f"🔍 Starting batch upgrade to PFSUS v2.0.0 standards...")
        
        results = {
            'processed_files': [],
            'modified_files': [],
            'errors': [],
            'stats': {}
        }
        
        # Find all files matching the pattern
        files_to_process = []
        for file_path in self.workspace_root.rglob(file_pattern):
            if file_path.is_file() and not self._should_skip_file(file_path):
                file_type = self._determine_file_type(file_path)
                if file_type:
                    files_to_process.append((file_path, file_type))
        
        logger.info(f"Found {len(files_to_process)} files to process.")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_file,
                    file_path,
                    file_type,
                    dry_run
                )
                for file_path, file_type in files_to_process
            ]
            
            completed_results = await asyncio.gather(*tasks)
            
            for result in completed_results:
                results['processed_files'].append(result['file'])
                if result['modified']:
                    results['modified_files'].append(result['file'])
                if result['error']:
                    results['errors'].append({
                        'file': result['file'],
                        'error': result['error']
                    })
                
                # Update stats
                self.stats['files_processed'] += 1
                if result['modified']:
                    self.stats['files_modified'] += 1
                if result['patterns_applied']:
                    self.stats['patterns_applied'] += result['patterns_applied']
                if result['backup_created']:
                    self.stats['backups_created'] += 1
                if result['error']:
                    self.stats['errors'] += 1
        
        results['stats'] = self.stats
        return results
    
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
            r'\.log$',
            r'\.bak$'
        ]
        
        file_str = str(file_path)
        return any(re.search(pattern, file_str) for pattern in skip_patterns)
    
    def _determine_file_type(self, file_path: Path) -> Optional[str]:
        """Determine file type based on path."""
        file_str = str(file_path)
        
        for pattern, file_type in self.FILE_PATTERNS.items():
            if re.match(pattern, file_str):
                return file_type
        
        return None
    
    def _process_file(self, file_path: Path, file_type: str, dry_run: bool) -> Dict[str, Any]:
        """Process a single file for upgrade."""
        result = {
            'file': str(file_path),
            'type': file_type,
            'modified': False,
            'patterns_applied': 0,
            'backup_created': False,
            'error': None
        }
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Apply upgrades based on file type
            if file_type == 'markdown':
                content, patterns_applied = self._upgrade_markdown(content)
            elif file_type == 'python':
                content, patterns_applied = self._upgrade_python(content)
            elif file_type == 'pfsus_standard' or file_type == 'pfsus_template':
                content, patterns_applied = self._upgrade_pfsus(content)
            elif file_type == 'json':
                content, patterns_applied = self._upgrade_json(content)
            
            result['patterns_applied'] = patterns_applied
            
            # Check if content was modified
            if content != original_content:
                result['modified'] = True
                
                if not dry_run:
                    # Create backup if requested
                    if self.backup:
                        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                        shutil.copy2(file_path, backup_path)
                        result['backup_created'] = True
                    
                    # Write updated content
                    file_path.write_text(content, encoding='utf-8')
                    logger.info(f"✅ Updated {file_path} with {patterns_applied} pattern applications.")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error processing {file_path}: {e}")
        
        return result
    
    def _upgrade_markdown(self, content: str) -> Tuple[str, int]:
        """Upgrade markdown content to PFSUS v2.0.0."""
        patterns_applied = 0
        
        # 1. Apply operator pattern migration
        for pattern, replacement in self.OPERATOR_PATTERNS.items():
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                patterns_applied += len(re.findall(pattern, content))
                content = new_content
        
        # 2. Add missing PFSUS elements if needed
        if '## τ:self_reference' in content or '## τself_reference' in content:
            # Already has self-reference, update to v2.0.0 format
            content = re.sub(r'## τ:self_reference\(([^)]*)\)', r'## τself_reference(\1)', content)
        elif not re.search(r'## τself_reference\(', content):
            # Add self-reference block
            file_name = Path(content[:100].splitlines()[0].strip('# ')).stem if content else "document"
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            self_ref = f'\n## τself_reference({file_name}_metadata)\n'
            self_ref += f'{{type:Documentation, file:"{file_name}", version:"1.0.0", '
            self_ref += f'checksum:"sha256:{checksum}", '
            self_ref += f'canonical_address:"{file_name.replace(".", "-")}", '
            self_ref += f'pfsus_compliant:true, english_shorthand:true}}\n'
            
            # Add before footer if it exists, otherwise at end
            if '%% MMCP-FOOTER:' in content:
                content = content.replace('%% MMCP-FOOTER:', self_ref + '\n%% MMCP-FOOTER:')
            else:
                content += self_ref
            
            patterns_applied += 1
        
        # 3. Update or add MMCP footer
        if '%% MMCP-FOOTER:' in content:
            # Update existing footer
            footer_pattern = r'%% MMCP-FOOTER: (.*?)$'
            footer_match = re.search(footer_pattern, content, re.MULTILINE)
            if footer_match:
                footer_content = footer_match.group(1)
                if 'english_shorthand' not in footer_content:
                    new_footer = footer_content.rstrip() + '; english_shorthand=true'
                    content = content.replace(footer_match.group(0), f'%% MMCP-FOOTER: {new_footer}')
                    patterns_applied += 1
        else:
            # Add new footer
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            footer = f'\n%% MMCP-FOOTER: version=1.0.0; timestamp={timestamp}; '
            footer += f'author=MCP_Core_Team; pfsus_compliant=true; english_shorthand=true\n'
            content += footer
            patterns_applied += 1
        
        # 4. Apply English language shorthand to headers
        header_pattern = r'^(#+)\s+([^λℵΔτiβΩ].+)$'
        
        def header_replacement(match):
            nonlocal patterns_applied
            level, header_text = match.groups()
            
            # Determine appropriate operator based on header text
            operator = self._determine_operator_for_text(header_text)
            
            # Convert to camelCase and remove special characters
            header_word = re.sub(r'[^a-zA-Z0-9_\s]', '', header_text.lower())
            header_word = re.sub(r'\s+', '_', header_word.strip())
            
            patterns_applied += 1
            return f'{level} {operator}{header_word}'
        
        content = re.sub(header_pattern, header_replacement, content, flags=re.MULTILINE)
        
        return content, patterns_applied
    
    def _upgrade_python(self, content: str) -> Tuple[str, int]:
        """Upgrade Python content to PFSUS v2.0.0."""
        patterns_applied = 0
        
        # 1. Update imports and module docstrings
        if '"""' in content[:500]:  # Check if there's a docstring at the top
            docstring_pattern = r'"""(.*?)"""'
            docstring_match = re.search(docstring_pattern, content[:500], re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1)
                new_docstring = docstring
                
                # Apply operator patterns to docstring
                for pattern, replacement in self.OPERATOR_PATTERNS.items():
                    new_docstring = re.sub(pattern, replacement, new_docstring)
                
                if new_docstring != docstring:
                    content = content.replace(f'"""{docstring}"""', f'"""{new_docstring}"""')
                    patterns_applied += 1
        
        # 2. Update function and class docstrings
        def_pattern = r'def\s+(\w+).*?:\s*"""(.*?)"""'
        
        def def_replacement(match):
            nonlocal patterns_applied
            func_name, docstring = match.groups()
            new_docstring = docstring
            
            # Apply operator patterns to docstring
            for pattern, replacement in self.OPERATOR_PATTERNS.items():
                new_docstring = re.sub(pattern, replacement, new_docstring)
            
            # Add operator annotation if missing
            if not any(op in new_docstring for op in ['λ', 'ℵ', 'Δ', 'τ', 'i', 'β', 'Ω']):
                operator = self._determine_operator_for_text(func_name)
                new_docstring = f"{new_docstring}\n    \n    {operator}{func_name}({func_name.split('_')[0]})"
                patterns_applied += 1
            
            return f'def {func_name}:"""{new_docstring}"""'
        
        content = re.sub(def_pattern, def_replacement, content, flags=re.DOTALL)
        
        # 3. Update comments
        comment_pattern = r'#\s+([^λℵΔτiβΩ].+)'
        
        def comment_replacement(match):
            nonlocal patterns_applied
            comment_text = match.group(1)
            
            # Skip certain comments
            if any(skip in comment_text.lower() for skip in ['todo', 'fixme', 'hack', 'note']):
                return f'# {comment_text}'
            
            # Determine appropriate operator based on comment text
            operator = self._determine_operator_for_text(comment_text)
            
            # Create a camelCase word from the first few words
            words = comment_text.split()
            if words:
                first_word = re.sub(r'[^a-zA-Z0-9_]', '', words[0].lower())
                patterns_applied += 1
                return f'# {operator}{first_word} {" ".join(words[1:])}'
            
            return f'# {comment_text}'
        
        content = re.sub(comment_pattern, comment_replacement, content, flags=re.MULTILINE)
        
        return content, patterns_applied
    
    def _upgrade_pfsus(self, content: str) -> Tuple[str, int]:
        """Upgrade PFSUS standard content to v2.0.0."""
        patterns_applied = 0
        
        # 1. Update version reference if it's a v1.x.x file
        if 'version:"1.' in content:
            content = re.sub(r'version:"1\.\d+\.\d+"', 'version:"2.0.0"', content)
            patterns_applied += 1
        
        # 2. Update standard reference
        if 'standard:"PFSUS+EARS+LambdaJSON+MathOps"' in content:
            content = content.replace(
                'standard:"PFSUS+EARS+LambdaJSON+MathOps"', 
                'standard:"PFSUS+EARS+LambdaJSON+MathOps+EnglishShorthand"'
            )
            patterns_applied += 1
        
        # 3. Apply operator pattern migration
        for pattern, replacement in self.OPERATOR_PATTERNS.items():
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                patterns_applied += len(re.findall(pattern, content))
                content = new_content
        
        # 4. Update self-reference block
        if '{type:SelfReference' in content:
            if 'english_shorthand:' not in content:
                content = re.sub(
                    r'(\{type:SelfReference.*?)(\})',
                    r'\1, english_shorthand:true\2',
                    content
                )
                patterns_applied += 1
        
        # 5. Update footer
        if '%% MMCP-FOOTER:' in content:
            if 'english_shorthand=' not in content:
                content = re.sub(
                    r'(%% MMCP-FOOTER:.*?)(;|\n)',
                    r'\1; english_shorthand=true\2',
                    content
                )
                patterns_applied += 1
        
        return content, patterns_applied
    
    def _upgrade_json(self, content: str) -> Tuple[str, int]:
        """Upgrade JSON content to PFSUS v2.0.0."""
        patterns_applied = 0
        
        try:
            # Parse JSON
            data = json.loads(content)
            
            # Check if this is a configuration file
            if isinstance(data, dict):
                # Add operator prefixes to top-level keys if appropriate
                new_data = {}
                for key, value in data.items():
                    if not any(op in key for op in ['λ', 'ℵ', 'Δ', 'τ', 'i', 'β', 'Ω']):
                        operator = self._determine_operator_for_text(key)
                        new_key = f"{operator}{key}"
                        new_data[new_key] = value
                        patterns_applied += 1
                    else:
                        new_data[key] = value
                
                # Add PFSUS metadata if missing
                if 'meta' not in new_data and 'metadata' not in new_data:
                    new_data['τmetadata'] = {
                        'pfsus_compliant': True,
                        'english_shorthand': True,
                        'version': '2.0.0',
                        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                    }
                    patterns_applied += 1
                
                # Convert back to JSON with pretty formatting
                return json.dumps(new_data, indent=2), patterns_applied
        except json.JSONDecodeError:
            # Not valid JSON, return unchanged
            pass
        
        return content, patterns_applied
    
    def _determine_operator_for_text(self, text: str) -> str:
        """Determine the most appropriate operator for a text string."""
        text_lower = text.lower()
        
        # Count occurrences of words from each operator category
        scores = {op: 0 for op in self.OPERATOR_WORDS.keys()}
        
        for op, words in self.OPERATOR_WORDS.items():
            for word in words:
                if word in text_lower:
                    scores[op] += 1
        
        # Find operator with highest score
        max_score = 0
        best_operator = 'λ'  # Default to lambda
        
        for op, score in scores.items():
            if score > max_score:
                max_score = score
                best_operator = op
        
        return best_operator
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive upgrade report."""
        report = []
        report.append("# PFSUS v2.0.0 Batch Upgrade Report")
        report.append(f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        stats = results['stats']
        report.append(f"### λsummary(upgrade_statistics)")
        report.append(f"- **Files Processed**: {stats['files_processed']}")
        report.append(f"- **Files Modified**: {stats['files_modified']}")
        report.append(f"- **Patterns Applied**: {stats['patterns_applied']}")
        report.append(f"- **Backups Created**: {stats['backups_created']}")
        report.append(f"- **Errors**: {stats['errors']}")
        report.append(f"- **Success Rate**: {(stats['files_modified'] / stats['files_processed'] * 100) if stats['files_processed'] > 0 else 0:.1f}%")
        report.append("")
        
        # Modified files
        if results['modified_files']:
            report.append(f"### ℵmodified_files(upgraded_to_v2)")
            for file_path in sorted(results['modified_files']):
                report.append(f"- {file_path}")
            report.append("")
        
        # Errors
        if results['errors']:
            report.append(f"### βerrors(upgrade_failures)")
            for error in results['errors']:
                report.append(f"#### {error['file']}")
                report.append(f"- **Error**: {error['error']}")
                report.append("")
        
        # Next steps
        report.append(f"### Δnext_steps(post_upgrade_actions)")
        report.append("1. **Review Modified Files**: Check the upgraded files for any issues.")
        report.append("2. **Run Tests**: Ensure all tests pass with the upgraded files.")
        report.append("3. **Update Documentation**: Update any documentation that references the old format.")
        report.append("4. **Commit Changes**: Commit the upgraded files to version control.")
        report.append("")
        
        # Footer
        report.append(f"## τself_reference(upgrade_report_metadata)")
        report.append(f"{{type:Report, file:\"pfsus_upgrade_report.md\", version:\"2.0.0\", timestamp:\"{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}\", pfsus_compliant:true, english_shorthand:true}}")
        report.append("")
        report.append(f"%% MMCP-FOOTER: version=2.0.0; timestamp={datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}; author=MCP_Core_Team; pfsus_compliant=true; english_shorthand=true")
        
        return '\n'.join(report)

async def main():
    """Main CLI interface for PFSUS batch upgrade."""
    parser = argparse.ArgumentParser(description='PFSUS Batch Upgrade Tool')
    parser.add_argument('--workspace', '-w', default='.', help='Workspace root directory')
    parser.add_argument('--pattern', '-p', default='*', help='File pattern to process')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run mode (no changes)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup files')
    parser.add_argument('--report', '-r', help='Generate report file')
    
    args = parser.parse_args()
    
    upgrader = PFSUSBatchUpgrade(args.workspace, not args.no_backup)
    
    # Run batch upgrade
    results = await upgrader.batch_upgrade(args.pattern, args.dry_run)
    
    # Generate and output report
    report = upgrader.generate_report(results)
    
    if args.report:
        Path(args.report).write_text(report, encoding='utf-8')
        print(f"Report saved to: {args.report}")
    else:
        print(report)

if __name__ == "__main__":
    asyncio.run(main())

# λbatch_upgrade_complete(pfsus_v2_migration) Processing complete
# Version: 1.0.0 | Last Modified: 2025-07-22T19:30:00Z
# Dependencies: @{CORE.PFSUS.STANDARD.002, CORE.UTILS.001}
# Related: @{CORE.PFSUS.CLI.ENFORCER.001, CORE.DOCS.UPGRADE.001}