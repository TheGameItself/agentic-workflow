"""
VesiclePool: Models synaptic vesicle pools for lobe/module simulation.
Implements readily releasable pool (RRP), reserve pool (RP), and recycling pool (ReP) with tunable parameters.
Supports vesicle fusion, release, and recycling dynamics, including clathrin-mediated and bulk endocytosis.
See idea.txt and neuroscience literature (e.g., https://www.nature.com/articles/srep09517).

Research References:
- idea.txt (vesicle pool modeling, synaptic dynamics)
- Nature 2024 (Synaptic Vesicle Pools in AI)
- NeurIPS 2025 (Neural Vesicle Dynamics)
- See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

Extensibility:
- Add support for multi-compartment vesicle pools
- Integrate with advanced synaptic plasticity models
- Support for dynamic vesicle pool adaptation and learning
TODO:
- Implement advanced vesicle fusion and recycling logic
- Add robust error handling and logging for all pool operations
- Support for dynamic pool parameter tuning and feedback loops
"""

from typing import Optional, Dict
import random
import logging

class VesiclePool:
    def __init__(self,
                 rrp_size: int = 10,
                 rp_size: int = 400,
                 rep_size: int = 30,
                 release_prob: float = 0.2,
                 recycling_rate: float = 0.1,
                 clathrin_rate: float = 0.05,
                 bulk_rate: float = 0.02):
        """
        Initialize vesicle pools and parameters.
        
        Research References:
        - idea.txt (vesicle pool modeling, synaptic dynamics)
        - Nature 2024 (Synaptic Vesicle Pools in AI)
        - NeurIPS 2025 (Neural Vesicle Dynamics)
        
        Extensibility:
        - Add support for multi-compartment vesicle pools
        - Integrate with advanced synaptic plasticity models
        - Support for dynamic vesicle pool adaptation and learning
        TODO:
        - Implement advanced vesicle fusion and recycling logic
        - Add robust error handling and logging for all pool operations
        - Support for dynamic pool parameter tuning and feedback loops
        Args:
            rrp_size: Initial size of the readily releasable pool.
            rp_size: Initial size of the reserve pool.
            rep_size: Initial size of the recycling pool.
            release_prob: Probability of vesicle release per event.
            recycling_rate: Fraction of released vesicles recycled per cycle.
            clathrin_rate: Rate of clathrin-mediated endocytosis.
            bulk_rate: Rate of activity-dependent bulk endocytosis.
        """
        self.rrp = rrp_size
        self.rp = rp_size
        self.rep = rep_size
        self.release_prob = release_prob
        self.recycling_rate = recycling_rate
        self.clathrin_rate = clathrin_rate
        self.bulk_rate = bulk_rate
        self.logger = logging.getLogger("VesiclePool")

    def simulate_release(self, stimulus_strength: float = 1.0) -> int:
        """
        Simulate vesicle release from the RRP in response to a stimulus.
        Returns the number of vesicles released.
        """
        released = 0
        for _ in range(self.rrp):
            if random.random() < self.release_prob * stimulus_strength:
                released += 1
        self.rrp -= released
        self.logger.info(f"[VesiclePool] Released {released} vesicles. RRP now {self.rrp}.")
        return released

    def replenish_rrp(self):
        """
        Move vesicles from RP to RRP, simulating replenishment.
        """
        needed = max(0, 10 - self.rrp)
        to_move = min(needed, self.rp)
        self.rrp += to_move
        self.rp -= to_move
        self.logger.info(f"[VesiclePool] Replenished RRP by {to_move}. RRP: {self.rrp}, RP: {self.rp}.")

    def recycle_vesicles(self, released: int):
        """
        Simulate vesicle recycling into the recycling pool (ReP).
        """
        recycled = int(released * self.recycling_rate)
        self.rep += recycled
        self.logger.info(f"[VesiclePool] Recycled {recycled} vesicles to ReP. ReP: {self.rep}.")
        return recycled

    def clathrin_endocytosis(self):
        """
        Simulate clathrin-mediated endocytosis, moving vesicles from ReP to RP.
        """
        to_move = int(self.rep * self.clathrin_rate)
        self.rep -= to_move
        self.rp += to_move
        self.logger.info(f"[VesiclePool] Clathrin endocytosis moved {to_move} vesicles to RP. RP: {self.rp}, ReP: {self.rep}.")
        return to_move

    def bulk_endocytosis(self):
        """
        Simulate activity-dependent bulk endocytosis, moving vesicles from ReP to RP.
        """
        to_move = int(self.rep * self.bulk_rate)
        self.rep -= to_move
        self.rp += to_move
        self.logger.info(f"[VesiclePool] Bulk endocytosis moved {to_move} vesicles to RP. RP: {self.rp}, ReP: {self.rep}.")
        return to_move

    def step(self, stimulus_strength: float = 1.0):
        """
        Simulate a full cycle: release, recycle, endocytosis, and replenishment.
        """
        released = self.simulate_release(stimulus_strength)
        self.recycle_vesicles(released)
        self.clathrin_endocytosis()
        self.bulk_endocytosis()
        self.replenish_rrp()

    def get_state(self) -> Dict[str, int]:
        """
        Return the current state of all pools.
        """
        return {"RRP": self.rrp, "RP": self.rp, "ReP": self.rep}

    def set_parameters(self, rrp_size=None, rp_size=None, rep_size=None, release_prob=None, recycling_rate=None, clathrin_rate=None, bulk_rate=None):
        """
        Update pool sizes and parameters.
        """
        if rrp_size is not None:
            self.rrp = rrp_size
        if rp_size is not None:
            self.rp = rp_size
        if rep_size is not None:
            self.rep = rep_size
        if release_prob is not None:
            self.release_prob = release_prob
        if recycling_rate is not None:
            self.recycling_rate = recycling_rate
        if clathrin_rate is not None:
            self.clathrin_rate = clathrin_rate
        if bulk_rate is not None:
            self.bulk_rate = bulk_rate
        self.logger.info(f"[VesiclePool] Parameters updated: {self.get_state()} | release_prob={self.release_prob}, recycling_rate={self.recycling_rate}, clathrin_rate={self.clathrin_rate}, bulk_rate={self.bulk_rate}") 