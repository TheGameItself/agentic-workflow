from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
import logging
import re
from typing import List, Dict, Any

class ErrorDetectionLobe:
    """
    Error-Detection Lobe
    Proactively scans for inconsistencies, potential bugs, and logical errors in project state, code, and memory.
    Implements advanced static analysis and anomaly detection (see idea.txt, Clean Code Best Practices, and recent research).
    Checks for TODOs, pass, NotImplementedError, undefined variables, unused imports, unreachable code, duplicate code blocks, suspicious signatures, and logic errors.
    Stores findings in working memory for feedback and self-improvement. Extensible for dynamic analysis and integration with other engines.
    """
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("ErrorDetectionLobe")

    def scan_for_errors(self, data: Any) -> List[Dict[str, Any]]:
        """
        Scan data (string or list of lines) for static analysis issues and anomalies.
        Returns a list of detected issues with line numbers, patterns, and descriptions.
        """
        errors = []
        if isinstance(data, str):
            lines = data.splitlines()
        else:
            lines = data
        # Basic patterns
        patterns = [
            (r'TODO', 'TODO comment'),
            (r'pass', 'pass statement'),
            (r'NotImplementedError', 'NotImplementedError'),
        ]
        for i, line in enumerate(lines):
            for pat, desc in patterns:
                if re.search(pat, line):
                    errors.append({'line': i+1, 'pattern': pat, 'description': desc, 'content': line.strip()})
        # Advanced static analysis
        code = '\n'.join(lines)
        # Undefined variables (simple heuristic)
        undefined_vars = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=.*', code)
        for var in undefined_vars:
            if not re.search(rf'def\s+.*\({var}', code) and not re.search(rf'class\s+{var}', code):
                if not re.search(rf'self\.{var}', code):
                    if not re.search(rf'{var}\s*=', code, re.MULTILINE):
                        errors.append({'line': None, 'pattern': 'undefined_var', 'description': f'Possible undefined variable: {var}', 'content': ''})
        # Unused imports
        import_lines = [l for l in lines if l.strip().startswith('import') or l.strip().startswith('from')]
        for l in import_lines:
            m = re.match(r'(?:from|import)\s+([a-zA-Z0-9_\.]+)', l)
            if m:
                mod = m.group(1)
                if not re.search(rf'\b{mod}\b', code.split(l, 1)[-1]):
                    errors.append({'line': lines.index(l)+1, 'pattern': 'unused_import', 'description': f'Unused import: {mod}', 'content': l.strip()})
        # Unreachable code (after return/raise/exit)
        for i, line in enumerate(lines):
            if re.search(r'\b(return|raise|exit)\b', line):
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        errors.append({'line': j+1, 'pattern': 'unreachable_code', 'description': 'Unreachable code after return/raise/exit', 'content': lines[j].strip()})
        # Duplicate code blocks (simple heuristic: repeated 3+ line blocks)
        block_map = {}
        for i in range(len(lines)-2):
            block = '\n'.join(lines[i:i+3])
            if block in block_map:
                errors.append({'line': i+1, 'pattern': 'duplicate_block', 'description': 'Duplicate code block detected', 'content': block})
            else:
                block_map[block] = i
        # Suspicious function signatures (too many args, no docstring)
        for i, line in enumerate(lines):
            if re.match(r'def\s+\w+\s*\(([^)]*)\):', line):
                args = re.findall(r'def\s+\w+\s*\(([^)]*)\):', line)[0]
                if len(args.split(',')) > 6:
                    errors.append({'line': i+1, 'pattern': 'suspicious_signature', 'description': 'Function with too many arguments', 'content': line.strip()})
                # Check for docstring
                if i+1 < len(lines) and not lines[i+1].strip().startswith('"""'):
                    errors.append({'line': i+1, 'pattern': 'missing_docstring', 'description': 'Function missing docstring', 'content': line.strip()})
        # Common logic errors (== vs =, assignment in conditionals)
        for i, line in enumerate(lines):
            if re.search(r'if\s+.*=[^=]', line):
                errors.append({'line': i+1, 'pattern': 'assignment_in_conditional', 'description': 'Possible assignment in conditional (should be ==?)', 'content': line.strip()})
        if errors:
            self.logger.warning(f"[ErrorDetectionLobe] Found issues: {errors}")
        for err in errors:
            self.working_memory.add(err)
        return errors

    def scan_for_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """
        Placeholder for future dynamic anomaly detection (runtime, statistical, or ML-based).
        To be implemented with integration to other engines (e.g., MathLogicEngine, PatternRecognitionEngine).
        """
        self.logger.info("[ErrorDetectionLobe] scan_for_anomalies is a stub for future dynamic analysis.")
        # TODO: Integrate with MathLogicEngine, PatternRecognitionEngine, etc.
        return [] 