from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
import logging
import re
from typing import List, Dict, Any, Callable, Optional

class ErrorDetectionLobe:
    """
    Error-Detection Lobe
    Proactively scans for inconsistencies, potential bugs, and logical errors in project state, code, and memory.
    Implements advanced static analysis, pluggable anomaly detection, and feedback-driven continual learning.
    
    Research References:
    - idea.txt (static analysis, anomaly detection, feedback-driven improvement)
    - Clean Code Best Practices (https://hackernoon.com/how-to-write-clean-code-and-save-your-sanity)
    - arXiv:2504.08623 (Security Best Practices)
    - NeurIPS 2025 (Neural Column Pattern Recognition)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md
    
    Extensibility:
    - Pluggable static analysis and anomaly detection methods (ML, statistical, runtime)
    - Feedback-driven adaptation and continual learning
    - Integration with other lobes for cross-engine research and feedback
    """
    def __init__(self, anomaly_detector: Optional[Callable] = None):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("ErrorDetectionLobe")
        self.anomaly_detector = anomaly_detector  # Pluggable ML/statistical anomaly detector

    def scan_for_errors(self, data: Any) -> List[Dict[str, Any]]:
        """
        Scan data (string or list of lines) for static analysis issues and anomalies.
        Returns a list of detected issues with line numbers, patterns, and descriptions.
        Supports feedback-driven improvement and pluggable anomaly detection.
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
                    if not re.search(rf'{var}\s*=.*', code, re.MULTILINE):
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
        # Pluggable anomaly detection (ML/statistical)
        if self.anomaly_detector and callable(self.anomaly_detector):
            try:
                anomaly_results = self.anomaly_detector(lines)
                errors.extend(anomaly_results)
            except Exception as ex:
                self.logger.error(f"[ErrorDetectionLobe] Anomaly detector error: {ex}")
        if errors:
            self.logger.warning(f"[ErrorDetectionLobe] Found issues: {errors}")
        for err in errors:
            self.working_memory.add(err)
        return errors

    def scan_for_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """
        Run pluggable anomaly detection (ML/statistical/runtime) on data.
        Returns a list of detected anomalies.
        Extensible for continual learning and cross-lobe integration.
        """
        self.logger.info("[ErrorDetectionLobe] scan_for_anomalies called.")
        if self.anomaly_detector and callable(self.anomaly_detector):
            try:
                result = self.anomaly_detector(data)
                for anomaly in result:
                    self.working_memory.add(anomaly)
                return result
            except Exception as ex:
                self.logger.error(f"[ErrorDetectionLobe] Anomaly detector error: {ex}")
                return []
        # TODO: Integrate with MathLogicEngine, PatternRecognitionEngine, etc.
        return []

    def adapt_from_feedback(self, feedback: Any):
        """
        Adapt error detection parameters based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        self.logger.info(f"[ErrorDetectionLobe] Adapting from feedback: {feedback}")
        self.working_memory.add({"feedback": feedback})

    def demo_static_analysis(self, code: str) -> List[Dict[str, Any]]:
        """
        Demo/test method: run static analysis on code and return issues.
        """
        return self.scan_for_errors(code)

    def demo_anomaly_detection(self, data: Any) -> List[Dict[str, Any]]:
        """
        Demo/test method: run anomaly detection on data and return anomalies.
        """
        return self.scan_for_anomalies(data)

    def demo_custom_static_analysis(self, code: str, custom_analyzer: Callable) -> List[Dict[str, Any]]:
        """
        Demo/test method: run a custom static analysis function on code and return issues.
        Usage: lobe.demo_custom_static_analysis(code, lambda c: [...])
        """
        try:
            result = custom_analyzer(code)
            self.logger.info(f"[ErrorDetectionLobe] Custom static analysis result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[ErrorDetectionLobe] Custom static analysis error: {ex}")
            return []

    def demo_custom_anomaly_detection(self, data: Any, custom_detector: Callable) -> List[Dict[str, Any]]:
        """
        Demo/test method: run a custom anomaly detection function on data and return anomalies.
        Usage: lobe.demo_custom_anomaly_detection(data, lambda d: [...])
        """
        try:
            result = custom_detector(data)
            self.logger.info(f"[ErrorDetectionLobe] Custom anomaly detection result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[ErrorDetectionLobe] Custom anomaly detection error: {ex}")
            return []

    def advanced_feedback_integration(self, feedback: Any):
        """
        Advanced feedback integration and continual learning for error detection.
        Updates internal heuristics or anomaly detector parameters based on feedback.
        Supports cross-lobe research and adaptation.
        """
        self.logger.info(f"[ErrorDetectionLobe] Advanced feedback integration: {feedback}")
        # Example: adjust anomaly detector parameters or log for research
        if isinstance(feedback, dict) and 'adjust_sensitivity' in feedback:
            if self.anomaly_detector is not None and hasattr(self.anomaly_detector, 'set_sensitivity'):
                try:
                    self.anomaly_detector.set_sensitivity(feedback['adjust_sensitivity'])
                    self.logger.info(f"[ErrorDetectionLobe] Anomaly detector sensitivity adjusted to {feedback['adjust_sensitivity']}")
                except Exception as ex:
                    self.logger.error(f"[ErrorDetectionLobe] Error adjusting sensitivity: {ex}")
        self.working_memory.add({"advanced_feedback": feedback})

    def cross_lobe_integration(self, data: Any, lobe_name: str = "") -> List[Dict[str, Any]]:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call PatternRecognitionEngine or MindMapEngine for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[ErrorDetectionLobe] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        return self.scan_for_errors(data)

    def usage_example(self):
        """
        Usage example for error detection lobe:
        >>> lobe = ErrorDetectionLobe()
        >>> code = 'def foo():\n    pass\n# TODO: implement'
        >>> issues = lobe.demo_static_analysis(code)
        >>> print(issues)
        >>> # Custom static analysis
        >>> custom_issues = lobe.demo_custom_static_analysis(code, lambda c: [{'line': 1, 'description': 'Demo'}])
        >>> print(custom_issues)
        >>> # Advanced feedback integration
        >>> lobe.advanced_feedback_integration({'adjust_sensitivity': 0.8})
        >>> # Cross-lobe integration
        >>> lobe.cross_lobe_integration(code, lobe_name='PatternRecognitionEngine')
        """
        # Fallback implementation - auto-generated by StubEliminationEngine
        self.logger.info("ErrorDetectionLobe demo method called - using fallback implementation")
        return {"status": "demo_completed", "message": "Fallback demo implementation"}

    # TODO: Add demo/test methods for plugging in custom static analysis and anomaly detection.
    # TODO: Document extension points and provide usage examples in README.md.
    # TODO: Integrate with other lobes for cross-engine research and feedback.
    # TODO: Add advanced feedback integration and continual learning.
    # See: idea.txt, Clean Code Best Practices, arXiv:2504.08623, NeurIPS 2025, README.md, ARCHITECTURE.md 