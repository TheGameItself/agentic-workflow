"""
AdaptiveMutationController: Calculates adaptive mutation rates for genetic triggers.
"""
import logging
from typing import Any

class AdaptiveMutationController:
    def __init__(self):
        self.logger = logging.getLogger("AdaptiveMutationController")

    def calculate_adaptive_rate(self, fitness_score: float) -> float:
        """
        Calculate adaptive mutation rate based on fitness score.
        Args:
            fitness_score: Fitness score of the genetic trigger.
        Returns:
            Adaptive mutation rate (float).
        """
        # Lower fitness -> higher mutation, higher fitness -> lower mutation
        rate = max(0.01, 0.1 * (1.0 - fitness_score))
        self.logger.info(f"AdaptiveMutationController: fitness={fitness_score:.2f}, mutation_rate={rate:.3f}")
        return rate
