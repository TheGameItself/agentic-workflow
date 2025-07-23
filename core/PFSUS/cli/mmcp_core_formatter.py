#!/usr/bin/env python3
"""
MMCP Core Formatter
@{CORE.PFSUS.CLI.FORMATTER.001} Core formatter for MMCP files with regex/anti-regex processing.
#{formatter,mmcp,core,cli,regex,wrapper}
λ(ℵ(Δ(β(core_formatting))))
"""

import os
import re
import sys
import json
import base64
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CalculusNotation:
    """Calculus notation wrapper system with order-agnostic nesting."""
    
    PRECEDENCE = {
        'τ': 7,  # Turing - highest precedence
        'Ω': 6,  # Omega
        'β': 5,  # Beta
        'Δ': 4,  # Delta
        'ℵ': 3,  # Alef
        'λ': 2,  # Lambda
        'i': 1   # Imaginary - lowest precedence
    }
    
    SYMBOLS = {
        'lambda': 'λ',
        'alef': 'ℵ',
        'delta': 'Δ',
        'beta': 'β',
        'omega': 'Ω',
        'imaginary': 'i',
        'turing': 'τ'
    }
    
    @classmethod
    def normalize_wrappers(cls, wrappers: List[str]) -> List[str]:
        """Normalize wrapper order based on precedence rules."""
        # Convert names to symbols
        symbols = [cls.SYMBOLS.get(w.lower(), w) for w in wrappers]
        
        # Sort by precedence (highest first)
        sorted_symbols = sorted(symbols, key=lambda x: cls.PRECEDENCE.get(x, 0), reverse=True)
        
        return sorted_symbols
    
    @classmethod
    def detect_content_type(cls, content: str) -> str:
        """Auto-detect appropriate calculus notation based on content analysis."""
        content_lower = content.lower()
        
        # State machine patterns
        if any(pattern in content_lower for pattern in ['state', 'machine', 'transition', 'class ']):
            return 'τ'  # Turing
        
        # Set operations
        if any(pattern in content_lower for pattern in ['{', '}', 'set', 'collection', 'dict']):
            return 'ℵ'  # Alef
        
        # Meta operations
        if any(pattern in content_lower for pattern in ['@', 'decorator', 'meta', 'change']):
            return 'Δ'  # Delta
        
        # Reduction operations
        if any(pattern in content_lower for pattern in ['reduce', 'fold', 'substitute']):
            return 'β'  # Beta
        
        # Terminal operations
        if any(pattern in content_lower for pattern in ['return', 'final', 'end', 'terminal']):
            return 'Ω'  # Omega
        
        # Complex/speculative
        if any(pattern in content_lower for pattern in ['complex', 'imaginary', 'speculative', 'maybe']):
            return 'i'  # Imaginary
        
        # Default to lambda for functional content
        return 'λ'
    
    @classmethod
    def wrap_content(cls, content: str, wrappers: List[str]) -> str:
        """Wrap content with calculus notation in proper order."""
        normalized_wrappers = cls.normalize_wrappers(wrappers)
        
        wrapped_content = content
        for wrapper in reversed(normalized_wrappers):  # Innermost first
            wrapped_content = f"{wrapper}:mmcp_wrapper(\n{wrapped_content}\n)"
        
        return wrapped_content

class MMCPRegexValidator:
    """Megalithic regex validator for MMCP content."""
    
    # Megalithic regex for valid MMCP content
    MEGALITHIC_REGEX = re.compile(
        r"^(\[ \] #\"\.root\"#.*|"
        r"#+\s.*|"
        r"---.*|"
        r"##\s*\{.*\}.*|"
        r"@\{.*\}.*|"
        r"%%.*|"
        r"<mmcp:nested.*>.*</mmcp:nested>|"
        r"\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.*|"
        r"\s*\"[a-zA-Z_][a-zA-Z0-9_]*\"\s*:\s*.*|"
        r"\s*\[.*\]\s*|"
        r"\s*\{.*\}\s*|"
        r"\s*[0-9]+\s*|"
        r"\s*\".*\"\s*|"
        r"\s*'.*'\s*|"
        r"\s*(true|false|null|undefined|NaN|Infinity|-Infinity)\s*|"
        r"\s*[λℵΔβΩiτ]:.*|"
        r"\s*base64:.*|"
        r"\s*$)",
        re.MULTILINE
    )
    
    # Megalithic counter-regex for invalid MMCP content
    MEGALITHIC_COUNTERREGEX = re.compile(
        r"^(?!(\[ \] #\"\.root\"#.*|"
        r"#+\s.*|"
        r"---.*|"
        r"##\s*\{.*\}.*|"
        r"@\{.*\}.*|"
        r"%%.*|"
        r"<mmcp:nested.*>.*</mmcp:nested>|"
        r"\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.*|"
        r"\s*\"[a-zA-Z_][a-zA-Z0-9_]*\"\s*:\s*.*|"
        r"\s*\[.*\]\s*|"
        r"\s*\{.*\}\s*|"
        r"\s*[0-9]+\s*|"
        r"\s*\".*\"\s*|"
        r"\s*'.*'\s*|"
        r"\s*(true|false|null|undefined|NaN|Infinity|-Infinity)\s*|"
        r"\s*[λℵΔβΩiτ]:.*|"
        r"\s*base64:.*|"
        r"\s*$)).*",
        re.MULTILINE
    )
    
    @classmethod
    def validate_content(cls, content: str) -> Tuple[bool, List[str]]:
        """Validate content using megalithic regex patterns."""
        invalid_lines = []
        valid = True
        
        for i, line in enumerate(content.splitlines(), 1):
            if line.strip() and not cls.MEGALITHIC_REGEX.match(line):
                if cls.MEGALITHIC_COUNTERREGEX.match(line):
                    invalid_lines.append(f"Line {i}: {line}")
                    valid = False
        
        return valid, invalid_lines

class MMCPWrapper:
    """MMCP content wrapper for various file formats."""
    
    WRAPPER_FORMATS = {
        '.py': {
            'start': "'''\n# MMCP-START",
            'end': "# MMCP-END\n'''",
            'comment': '#'
        },
        '.js': {
            'start': "/*\n * MMCP-START",
            'end': " * MMCP-END\n */",
            'comment': '//'
        },
        '.md': {
            'start': "```mmcp\n<!-- MMCP-START -->",
            'end': "<!-- MMCP-END -->\n```",
            'comment': '<!--'
        },
        '.json': {
            'start': '{\n  "__mmcp": {\n    "content": "',
            'end': '"\n  }\n}',
            'comment': '//'
        },
        '.sql': {
            'start': "/*\n * MMCP-START",
            'end': " * MMCP-END\n */",
            'comment': '--'
        },
        '.toml': {
            'start': "# MMCP-START\n[mmcp]\ncontent = '''",
            'end': "'''\n# MMCP-END",
            'comment': '#'
        }
    }
    
    @classmethod
    def wrap_content(cls, content: str, file_extension: str, calculus_wrappers: List[str] = None) -> str:
        """Wrap MMCP content for specific file format."""
        if file_extension not in cls.WRAPPER_FORMATS:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        wrapper_format = cls.WRAPPER_FORMATS[file_extension]
        
        # Apply calculus notation wrappers if specified
        if calculus_wrappers:
            content = CalculusNotation.wrap_content(content, calculus_wrappers)
        
        # Handle JSON special case (escape quotes)
        if file_extension == '.json':
            content = content.replace('"', '\\"').replace('\n', '\\n')
        
        wrapped_content = f"{wrapper_format['start']}\n{content}\n{wrapper_format['end']}"
        
        return wrapped_content
    
    @classmethod
    def unwrap_content(cls, wrapped_content: str, file_extension: str) -> str:
        """Extract MMCP content from wrapped format."""
        if file_extension not in cls.WRAPPER_FORMATS:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        wrapper_format = cls.WRAPPER_FORMATS[file_extension]
        
        # Find start and end markers
        start_marker = wrapper_format['start'].split('\n')[-1]  # Last line of start
        end_marker = wrapper_format['end'].split('\n')[0]       # First line of end
        
        lines = wrapped_content.splitlines()
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if start_marker in line and start_idx is None:
                start_idx = i + 1
            elif end_marker in line and start_idx is not None:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None:
            raise ValueError("Could not find MMCP content markers")
        
        content = '\n'.join(lines[start_idx:end_idx])
        
        # Handle JSON special case (unescape)
        if file_extension == '.json':
            content = content.replace('\\"', '"').replace('\\n', '\n')
        
        return content

class MMCPCoreFormatter:
    """Main MMCP core formatter class."""
    
    def __init__(self, core_path: Path):
        """Initialize the core formatter."""
        self.core_path = Path(core_path)
        self.stats = {
            'files_processed': 0,
            'files_wrapped': 0,
            'files_validated': 0,
            'validation_errors': 0,
            'start_time': datetime.now()
        }
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def find_mmcp_files(self) -> List[Path]:
        """Find all MMCP files in the core directory."""
        mmcp_files = []
        
        # Find files with .mmcp. in the name
        for pattern in ['**/*.mmcp.*', '**/*.mmcp']:
            mmcp_files.extend(self.core_path.glob(pattern))
        
        # Find files with MMCP content markers
        for ext in ['.py', '.js', '.md', '.json', '.sql', '.toml']:
            for file_path in self.core_path.glob(f'**/*{ext}'):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if 'MMCP-START' in content or 'mmcp_wrapper' in content:
                        mmcp_files.append(file_path)
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        return list(set(mmcp_files))  # Remove duplicates
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single MMCP file."""
        logger.info(f"Processing file: {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_size = len(content)
            
            # Validate content
            is_valid, invalid_lines = MMCPRegexValidator.validate_content(content)
            
            # Auto-detect calculus notation if not already wrapped
            if not any(symbol in content for symbol in CalculusNotation.SYMBOLS.values()):
                detected_notation = CalculusNotation.detect_content_type(content)
                logger.info(f"Auto-detected calculus notation: {detected_notation}")
            
            # Generate file statistics
            result = {
                'file_path': str(file_path),
                'original_size': original_size,
                'is_valid': is_valid,
                'invalid_lines': invalid_lines,
                'processed_at': datetime.now().isoformat(),
                'checksum': hashlib.sha256(content.encode()).hexdigest()
            }
            
            self.stats['files_processed'] += 1
            if is_valid:
                self.stats['files_validated'] += 1
            else:
                self.stats['validation_errors'] += len(invalid_lines)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def wrap_file_safely(self, file_path: Path, calculus_wrappers: List[str] = None) -> bool:
        """Safely wrap a file with MMCP format."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check if already wrapped
            if 'MMCP-START' in content or 'mmcp_wrapper' in content:
                logger.info(f"File {file_path} already wrapped, skipping")
                return False
            
            # Determine file extension
            if '.mmcp.' in file_path.name:
                # Extract the wrapper extension
                parts = file_path.name.split('.')
                wrapper_ext = f".{parts[-1]}"
            else:
                wrapper_ext = file_path.suffix
            
            # Auto-detect calculus notation if not specified
            if not calculus_wrappers:
                detected_notation = CalculusNotation.detect_content_type(content)
                calculus_wrappers = [detected_notation]
            
            # Wrap the content
            wrapped_content = MMCPWrapper.wrap_content(content, wrapper_ext, calculus_wrappers)
            
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            file_path.rename(backup_path)
            
            # Write wrapped content
            file_path.write_text(wrapped_content, encoding='utf-8')
            
            logger.info(f"Successfully wrapped file: {file_path}")
            self.stats['files_wrapped'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error wrapping file {file_path}: {e}")
            return False
    
    async def process_core_async(self, wrap_files: bool = False, validate_only: bool = False) -> Dict[str, Any]:
        """Process all core files asynchronously."""
        logger.info(f"Starting core processing: wrap_files={wrap_files}, validate_only={validate_only}")
        
        # Find all MMCP files
        mmcp_files = self.find_mmcp_files()
        logger.info(f"Found {len(mmcp_files)} MMCP files")
        
        # Process files in parallel
        loop = asyncio.get_event_loop()
        tasks = []
        
        for file_path in mmcp_files:
            if validate_only:
                task = loop.run_in_executor(self.executor, self.process_file, file_path)
            else:
                task = loop.run_in_executor(self.executor, self.process_file, file_path)
                if wrap_files:
                    wrap_task = loop.run_in_executor(self.executor, self.wrap_file_safely, file_path)
                    tasks.append(wrap_task)
            
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out wrap results (boolean) from process results (dict)
        process_results = [r for r in results if isinstance(r, dict)]
        
        # Generate summary report
        end_time = datetime.now()
        processing_time = (end_time - self.stats['start_time']).total_seconds()
        
        report = {
            'summary': {
                'total_files_found': len(mmcp_files),
                'files_processed': self.stats['files_processed'],
                'files_wrapped': self.stats['files_wrapped'],
                'files_validated': self.stats['files_validated'],
                'validation_errors': self.stats['validation_errors'],
                'processing_time_seconds': processing_time,
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': end_time.isoformat()
            },
            'file_results': process_results,
            'validation_summary': {
                'valid_files': len([r for r in process_results if r.get('is_valid', False)]),
                'invalid_files': len([r for r in process_results if not r.get('is_valid', True)]),
                'total_invalid_lines': sum(len(r.get('invalid_lines', [])) for r in process_results)
            }
        }
        
        return report
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate a manifest of all MMCP files in the core."""
        mmcp_files = self.find_mmcp_files()
        
        manifest = {
            'version': '1.3.0',
            'generated_at': datetime.now().isoformat(),
            'core_path': str(self.core_path),
            'total_files': len(mmcp_files),
            'files': []
        }
        
        for file_path in mmcp_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                file_info = {
                    'path': str(file_path.relative_to(self.core_path)),
                    'size': len(content),
                    'checksum': hashlib.sha256(content.encode()).hexdigest(),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'extension': file_path.suffix,
                    'is_wrapped': 'MMCP-START' in content or 'mmcp_wrapper' in content
                }
                manifest['files'].append(file_info)
            except Exception as e:
                logger.error(f"Error processing file {file_path} for manifest: {e}")
        
        return manifest

def main():
    """Main entry point for the MMCP core formatter."""
    parser = argparse.ArgumentParser(description='MMCP Core Formatter')
    parser.add_argument('core_path', help='Path to the core directory')
    parser.add_argument('--wrap', action='store_true', help='Wrap files with MMCP format')
    parser.add_argument('--validate-only', action='store_true', help='Only validate files, do not wrap')
    parser.add_argument('--output', '-o', help='Output file for the report')
    parser.add_argument('--manifest', action='store_true', help='Generate manifest file')
    parser.add_argument('--calculus', nargs='+', help='Calculus notation wrappers to apply')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize formatter
    formatter = MMCPCoreFormatter(args.core_path)
    
    try:
        if args.manifest:
            # Generate manifest
            manifest = formatter.generate_manifest()
            manifest_path = Path(args.core_path) / 'MMCP_MANIFEST.json'
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            logger.info(f"Manifest generated: {manifest_path}")
        else:
            # Process core files
            report = asyncio.run(formatter.process_core_async(
                wrap_files=args.wrap,
                validate_only=args.validate_only
            ))
            
            # Output report
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
                logger.info(f"Report written to: {output_path}")
            else:
                print(json.dumps(report, indent=2))
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# @{CORE.PFSUS.CLI.FORMATTER.001} End of core formatter implementation
# #{formatter,mmcp,core,cli,regex,wrapper,complete} Final tags
# λ(ℵ(Δ(β(formatting_complete)))) Processing complete
# Version: 1.0.0 | Last Modified: 2025-07-21T12:00:00Z
# Dependencies: @{CORE.PFSUS.STANDARD.001, CORE.UTILS.001}
# Related: @{CORE.TESTS.FORMATTER.001, CORE.DOCS.FORMATTER.001}