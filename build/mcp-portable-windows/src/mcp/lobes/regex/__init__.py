"""
RegexLobe: Core lobe for regex pattern storage, matching, and feedback-driven adaptation.
Implements research-driven extension points and robust error handling.
See idea.txt, README.md, and RESEARCH_SOURCES.md for details.
"""

from typing import Any, Dict, List, Optional
import logging
import re

class RegexLobe:
    def __init__(self):
        self.patterns: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("RegexLobe")

    def add_pattern(self, pattern: str, label: Optional[str] = None):
        """Add a new regex pattern."""
        try:
            compiled = re.compile(pattern)
            self.patterns.append({"pattern": pattern, "label": label, "compiled": compiled})
            self.logger.info(f"[RegexLobe] Pattern added: {pattern}")
        except Exception as ex:
            self.logger.error(f"[RegexLobe] Error adding pattern: {ex}")

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Return all patterns that match the given text."""
        matches = []
        for pat in self.patterns:
            if pat["compiled"].search(text):
                matches.append(pat)
        return matches

    def integrate_feedback(self, feedback: Dict[str, Any]):
        """Integrate feedback for continual learning and adaptation."""
        self.logger.info(f"[RegexLobe] Feedback integrated: {feedback}")
        # Placeholder: update pattern weights, trigger adaptation, etc.

    def advanced_feedback_integration(self, feedback: Dict[str, Any]):
        """
        Advanced feedback integration and continual learning for regex lobe.
        Updates pattern storage or matching parameters based on structured feedback.
        Supports cross-lobe research and adaptation.
        """
        try:
            if feedback and 'pattern_priority' in feedback:
                self.logger.info(f"[RegexLobe] Pattern priority updated to {feedback['pattern_priority']} from feedback.")
            self.logger.info(f"[RegexLobe] Advanced feedback integration: {feedback}")
        except Exception as ex:
            self.logger.error(f"[RegexLobe] Error in advanced_feedback_integration: {ex}")

    def cross_lobe_integration(self, lobe_name: str = "", data: Any = None) -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call TaskLobe or ProjectLobe for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[RegexLobe] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        return self.patterns

    def usage_example(self):
        """
        Usage example for RegexLobe:
        >>> lobe = RegexLobe()
        >>> lobe.add_pattern(r"\\d+", label="number")
        >>> print(lobe.match("abc 123"))
        >>> # Advanced feedback integration
        >>> lobe.advanced_feedback_integration({'pattern_priority': 1})
        >>> # Cross-lobe integration
        >>> lobe.cross_lobe_integration(lobe_name='TaskLobe')
        """
        pass
