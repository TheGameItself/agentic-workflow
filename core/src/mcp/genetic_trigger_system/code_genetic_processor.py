"""
CodeGeneticProcessor: Code-based implementation for genetic trigger activation.
Implements direct evaluation of environmental activation using algorithmic logic.
"""
from typing import Dict, Any
import logging

class CodeGeneticProcessor:
    def __init__(self):
        self.logger = logging.getLogger("CodeGeneticProcessor")

    def evaluate_activation(self, environment: Dict[str, Any], formation_environment: Dict[str, Any], threshold: float = 0.7) -> bool:
        """
        Evaluate whether the genetic trigger should activate using code logic.
        Args:
            environment: Current environmental conditions.
            formation_environment: The environment the trigger was formed in (required).
            threshold: Similarity threshold for activation.
        Returns:
            True if activation is recommended, False otherwise.
        """
        # Calculate similarity (simple ratio of matching keys/values)
        matching_keys = set(environment.keys()) & set(formation_environment.keys())
        if not matching_keys:
            return False
        similarities = []
        for key in matching_keys:
            v1, v2 = environment[key], formation_environment[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                max_val = max(abs(v1), abs(v2))
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(max(0.0, 1.0 - abs(v1 - v2) / max_val))
            elif isinstance(v1, str) and isinstance(v2, str):
                similarities.append(1.0 if v1 == v2 else 0.0)
            elif isinstance(v1, bool) and isinstance(v2, bool):
                similarities.append(1.0 if v1 == v2 else 0.0)
            else:
                similarities.append(0.0)
        score = sum(similarities) / len(similarities)
        self.logger.info(f"CodeGeneticProcessor similarity score: {score:.2f}")
        return score >= threshold
