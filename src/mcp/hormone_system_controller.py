"""
HormoneSystemController: Comprehensive hormone system for inter-lobe communication

This module implements a biologically-inspired hormone system that enables
complex communication between brain-like lobes in the MCP system. It handles
hormone production, circulation, receptor adaptation, and cascade effects.

References:
- idea.txt (brain-inspired architecture)
- cross-implementation.md (hormone system integration)
- arXiv:2406.06237 (neuromodulation in AI systems)
"""

import logging
import math
import random
import time
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import threading
from dataclasses import dataclass, field

# Define hormone-related data structures
@dataclass
class Hormone:
    """Represents a hormone with its properties and effects"""
    name: str
    base_decay_rate: float  # How quickly it naturally decays
    diffusion_rate: float   # How quickly it spreads to other lobes
    half_life: float        # Half-life in seconds
    description: str        # What this hormone does
    opposing_hormones: List[str] = field(default_factory=list)  # Hormones that counteract this one
    synergistic_hormones: List[str] = field(default_factory=list)  # Hormones that enhance this one

@dataclass
class LobeState:
    """Represents the state of a lobe in the hormone system"""
    name: str
    position: Tuple[float, float, float]  # 3D position for spatial diffusion
    local_hormone_levels: Dict[str, float] = field(default_factory=dict)
    receptor_sensitivity: Dict[str, float] = field(default_factory=dict)
    receptor_subtypes: Dict[str, List[str]] = field(default_factory=dict)
    vesicle_storage: Dict[str, List[Dict]] = field(default_factory=dict)
    connected_lobes: List[str] = field(default_factory=list)

@dataclass
class CascadeResult:
    """Results of processing hormone cascades"""
    triggered_cascades: List[str]
    affected_lobes: Dict[str, Dict[str, float]]
    feedback_loops: List[Dict]
    emergent_effects: List[str]

class HormoneSystemController:
    """
    Comprehensive hormone system controller that manages hormone production,
    circulation, receptor adaptation, and cascade effects between lobes.
    """
    
    def __init__(self, event_bus=None, decay_interval: float = 0.5):
        """
        Initialize the hormone system controller.
        
        Args:
            event_bus: Event bus for broadcasting hormone updates
            decay_interval: How often (in seconds) to process hormone decay
        """
        self.event_bus = event_bus
        self.decay_interval = decay_interval
        
        # Initialize hormone definitions with biological properties
        self.hormones = self._initialize_hormones()
        
        # Global hormone levels (bloodstream)
        self.global_hormone_levels: Dict[str, float] = {
            hormone: 0.0 for hormone in self.hormones
        }
        
        # Lobe states for hormone processing
        self.lobes: Dict[str, LobeState] = {}
        
        # Hormone cascade definitions
        self.cascades = self._initialize_cascades()
        
        # Receptor learning history
        self.receptor_performance_history: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        
        # Hormone gradient field for spatial diffusion
        self.gradient_fields: Dict[str, Dict[Tuple[float, float, float], float]] = {}
        
        # Start decay and diffusion thread
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
        logging.info("HormoneSystemController initialized with %d hormones", len(self.hormones))

    def _initialize_hormones(self) -> Dict[str, Hormone]:
        """
        Initialize all hormone definitions with their biological properties.
        
        Returns:
            Dictionary of hormone definitions
        """
        return {
            "dopamine": Hormone(
                name="dopamine",
                base_decay_rate=0.4,
                diffusion_rate=0.3,
                half_life=2.5,
                description="Reward signaling and motivation enhancement",
                opposing_hormones=["gaba"],
                synergistic_hormones=["norepinephrine"]
            ),
            "serotonin": Hormone(
                name="serotonin",
                base_decay_rate=0.2,
                diffusion_rate=0.25,
                half_life=4.6,
                description="Confidence and decision stability",
                opposing_hormones=["cortisol"],
                synergistic_hormones=["oxytocin", "dopamine"]
            ),
            "cortisol": Hormone(
                name="cortisol",
                base_decay_rate=0.12,
                diffusion_rate=0.4,
                half_life=8.2,
                description="Stress response and priority adjustment",
                opposing_hormones=["serotonin", "gaba"],
                synergistic_hormones=["adrenaline", "norepinephrine"]
            ),
            "oxytocin": Hormone(
                name="oxytocin",
                base_decay_rate=0.3,
                diffusion_rate=0.2,
                half_life=3.1,
                description="Collaboration and trust metrics",
                opposing_hormones=["testosterone"],
                synergistic_hormones=["vasopressin", "serotonin"]
            ),
            "vasopressin": Hormone(
                name="vasopressin",
                base_decay_rate=0.17,
                diffusion_rate=0.15,
                half_life=5.8,
                description="Memory formation and social bonding",
                opposing_hormones=[],
                synergistic_hormones=["oxytocin", "growth_hormone"]
            ),
            "growth_hormone": Hormone(
                name="growth_hormone",
                base_decay_rate=0.07,
                diffusion_rate=0.1,
                half_life=15.3,
                description="Learning rate and capability expansion",
                opposing_hormones=[],
                synergistic_hormones=["acetylcholine"]
            ),
            "acetylcholine": Hormone(
                name="acetylcholine",
                base_decay_rate=0.5,
                diffusion_rate=0.35,
                half_life=0.8,
                description="Attention focus and learning enhancement",
                opposing_hormones=[],
                synergistic_hormones=["norepinephrine"]
            ),
            "norepinephrine": Hormone(
                name="norepinephrine",
                base_decay_rate=0.4,
                diffusion_rate=0.3,
                half_life=2.1,
                description="Alertness and arousal modulation",
                opposing_hormones=["gaba"],
                synergistic_hormones=["adrenaline", "dopamine"]
            ),
            "adrenaline": Hormone(
                name="adrenaline",
                base_decay_rate=0.5,
                diffusion_rate=0.4,
                half_life=1.8,
                description="Urgency detection and response acceleration",
                opposing_hormones=["gaba"],
                synergistic_hormones=["cortisol", "norepinephrine"]
            ),
            "testosterone": Hormone(
                name="testosterone",
                base_decay_rate=0.01,
                diffusion_rate=0.05,
                half_life=96.0,
                description="Competitive drive and risk-taking behavior",
                opposing_hormones=["oxytocin"],
                synergistic_hormones=["dopamine"]
            ),
            "estrogen": Hormone(
                name="estrogen",
                base_decay_rate=0.014,
                diffusion_rate=0.06,
                half_life=72.0,
                description="Pattern recognition and memory consolidation",
                opposing_hormones=[],
                synergistic_hormones=["vasopressin"]
            ),
            "gaba": Hormone(
                name="gaba",
                base_decay_rate=0.6,
                diffusion_rate=0.25,
                half_life=1.2,
                description="Inhibitory control and noise reduction",
                opposing_hormones=["glutamate"],
                synergistic_hormones=[]
            ),
            "insulin": Hormone(
                name="insulin",
                base_decay_rate=0.15,
                diffusion_rate=0.2,
                half_life=6.4,
                description="Resource allocation and energy management",
                opposing_hormones=["glucagon"],
                synergistic_hormones=[]
            ),
            "glucagon": Hormone(
                name="glucagon",
                base_decay_rate=0.3,
                diffusion_rate=0.25,
                half_life=3.6,
                description="Emergency resource mobilization",
                opposing_hormones=["insulin"],
                synergistic_hormones=["adrenaline"]
            ),
            "leptin": Hormone(
                name="leptin",
                base_decay_rate=0.04,
                diffusion_rate=0.1,
                half_life=24.0,
                description="Satiety signaling and resource conservation",
                opposing_hormones=["ghrelin"],
                synergistic_hormones=[]
            ),
            "ghrelin": Hormone(
                name="ghrelin",
                base_decay_rate=0.3,
                diffusion_rate=0.2,
                half_life=3.6,
                description="Hunger signaling and resource seeking",
                opposing_hormones=["leptin"],
                synergistic_hormones=[]
            ),
            "prolactin": Hormone(
                name="prolactin",
                base_decay_rate=0.05,
                diffusion_rate=0.1,
                half_life=18.7,
                description="Nurturing behavior and protective responses",
                opposing_hormones=[],
                synergistic_hormones=["oxytocin"]
            ),
            "histamine": Hormone(
                name="histamine",
                base_decay_rate=0.25,
                diffusion_rate=0.3,
                half_life=4.2,
                description="Inflammatory response and threat detection",
                opposing_hormones=[],
                synergistic_hormones=["cortisol"]
            ),
            "aldosterone": Hormone(
                name="aldosterone",
                base_decay_rate=0.1,
                diffusion_rate=0.15,
                half_life=12.0,
                description="Fluid balance and system stability",
                opposing_hormones=[],
                synergistic_hormones=[]
            ),
            "calcitonin": Hormone(
                name="calcitonin",
                base_decay_rate=0.08,
                diffusion_rate=0.1,
                half_life=10.0,
                description="Bone remodeling and structural adaptation",
                opposing_hormones=[],
                synergistic_hormones=[]
            ),
            "secretin": Hormone(
                name="secretin",
                base_decay_rate=0.2,
                diffusion_rate=0.15,
                half_life=5.0,
                description="pH balance and environmental adaptation",
                opposing_hormones=[],
                synergistic_hormones=[]
            ),
            "angiotensin_ii": Hormone(
                name="angiotensin_ii",
                base_decay_rate=0.3,
                diffusion_rate=0.2,
                half_life=4.0,
                description="Pressure regulation and system maintenance",
                opposing_hormones=[],
                synergistic_hormones=["cortisol"]
            ),
            "endorphins": Hormone(
                name="endorphins",
                base_decay_rate=0.15,
                diffusion_rate=0.2,
                half_life=7.3,
                description="Pain tolerance and persistence mechanisms",
                opposing_hormones=[],
                synergistic_hormones=["dopamine"]
            ),
            "melatonin": Hormone(
                name="melatonin",
                base_decay_rate=0.08,
                diffusion_rate=0.1,
                half_life=12.0,
                description="Rest/activity cycle optimization",
                opposing_hormones=["norepinephrine"],
                synergistic_hormones=["gaba"]
            ),
        }

    def _initialize_cascades(self) -> Dict[str, Dict]:
        """
        Initialize hormone cascade definitions.
        
        Returns:
            Dictionary of cascade definitions
        """
        return {
            "stress_cascade": {
                "trigger_conditions": {
                    "cortisol": 0.7,
                    "adrenaline": 0.5
                },
                "sequence": [
                    {"hormone": "cortisol", "target_lobes": ["task_management", "error_detection"], "multiplier": 1.2},
                    {"hormone": "norepinephrine", "target_lobes": ["all"], "multiplier": 0.8},
                    {"hormone": "adrenaline", "target_lobes": ["resource_management"], "multiplier": 1.5},
                    {"hormone": "histamine", "target_lobes": ["error_detection"], "multiplier": 1.3}
                ],
                "feedback_loops": [
                    {"hormone": "gaba", "threshold": 0.6, "effect": "inhibit", "target": "cortisol"}
                ]
            },
            "reward_cascade": {
                "trigger_conditions": {
                    "dopamine": 0.8
                },
                "sequence": [
                    {"hormone": "dopamine", "target_lobes": ["task_management", "decision_making"], "multiplier": 1.2},
                    {"hormone": "serotonin", "target_lobes": ["decision_making"], "multiplier": 0.9},
                    {"hormone": "oxytocin", "target_lobes": ["social_intelligence"], "multiplier": 1.1},
                    {"hormone": "endorphins", "target_lobes": ["creativity_engine"], "multiplier": 1.0}
                ],
                "feedback_loops": [
                    {"hormone": "leptin", "threshold": 0.7, "effect": "inhibit", "target": "dopamine"}
                ]
            },
            "learning_cascade": {
                "trigger_conditions": {
                    "acetylcholine": 0.7,
                    "growth_hormone": 0.5
                },
                "sequence": [
                    {"hormone": "acetylcholine", "target_lobes": ["memory", "pattern_recognition"], "multiplier": 1.3},
                    {"hormone": "growth_hormone", "target_lobes": ["memory"], "multiplier": 1.2},
                    {"hormone": "vasopressin", "target_lobes": ["memory"], "multiplier": 1.1},
                    {"hormone": "estrogen", "target_lobes": ["pattern_recognition"], "multiplier": 0.9}
                ],
                "feedback_loops": [
                    {"hormone": "cortisol", "threshold": 0.7, "effect": "inhibit", "target": "growth_hormone"}
                ]
            },
            "focus_cascade": {
                "trigger_conditions": {
                    "norepinephrine": 0.7
                },
                "sequence": [
                    {"hormone": "norepinephrine", "target_lobes": ["task_management", "context_management"], "multiplier": 1.2},
                    {"hormone": "acetylcholine", "target_lobes": ["memory", "pattern_recognition"], "multiplier": 1.1},
                    {"hormone": "gaba", "target_lobes": ["context_management"], "multiplier": 0.8},
                    {"hormone": "insulin", "target_lobes": ["resource_management"], "multiplier": 1.0}
                ],
                "feedback_loops": [
                    {"hormone": "melatonin", "threshold": 0.6, "effect": "inhibit", "target": "norepinephrine"}
                ]
            },
            "emergency_response": {
                "trigger_conditions": {
                    "histamine": 0.9,
                    "cortisol": 0.8
                },
                "sequence": [
                    {"hormone": "histamine", "target_lobes": ["error_detection"], "multiplier": 1.5},
                    {"hormone": "adrenaline", "target_lobes": ["all"], "multiplier": 1.3},
                    {"hormone": "cortisol", "target_lobes": ["task_management"], "multiplier": 1.2},
                    {"hormone": "glucagon", "target_lobes": ["resource_management"], "multiplier": 1.4},
                    {"hormone": "norepinephrine", "target_lobes": ["all"], "multiplier": 1.1}
                ],
                "feedback_loops": [
                    {"hormone": "gaba", "threshold": 0.5, "effect": "inhibit", "target": "histamine"},
                    {"hormone": "serotonin", "threshold": 0.6, "effect": "inhibit", "target": "cortisol"}
                ]
            }
        }

    def register_lobe(self, lobe_name: str, position: Tuple[float, float, float] = (0, 0, 0), 
                     connected_lobes: List[str] = None) -> None:
        """
        Register a lobe with the hormone system.
        
        Args:
            lobe_name: Name of the lobe
            position: 3D position for spatial diffusion
            connected_lobes: List of directly connected lobes
        """
        if connected_lobes is None:
            connected_lobes = []
            
        # Initialize lobe state
        self.lobes[lobe_name] = LobeState(
            name=lobe_name,
            position=position,
            local_hormone_levels={hormone: 0.0 for hormone in self.hormones},
            receptor_sensitivity={hormone: 0.5 for hormone in self.hormones},
            receptor_subtypes=self._initialize_receptor_subtypes(lobe_name),
            vesicle_storage={},
            connected_lobes=connected_lobes
        )
        
        # Initialize receptor performance history
        self.receptor_performance_history[lobe_name] = {
            hormone: [] for hormone in self.hormones
        }
        
        logging.info(f"Registered lobe '{lobe_name}' with the hormone system")

    def _initialize_receptor_subtypes(self, lobe_name: str) -> Dict[str, List[str]]:
        """
        Initialize receptor subtypes based on lobe type.
        
        Args:
            lobe_name: Name of the lobe
            
        Returns:
            Dictionary mapping hormones to their receptor subtypes
        """
        # Default receptor subtypes
        default_subtypes = {
            "dopamine": ["D1", "D2"],
            "serotonin": ["5HT1", "5HT2"],
            "cortisol": ["GR", "MR"],
            "oxytocin": ["OTR"],
            "vasopressin": ["V1a", "V1b"],
            "growth_hormone": ["GHR"],
            "acetylcholine": ["nicotinic", "muscarinic"],
            "norepinephrine": ["alpha1", "alpha2", "beta1"],
            "adrenaline": ["alpha1", "beta1", "beta2"],
            "testosterone": ["AR"],
            "estrogen": ["ERalpha", "ERbeta"],
            "gaba": ["GABA_A", "GABA_B"],
            "insulin": ["IR"],
            "glucagon": ["GR"],
            "leptin": ["LepR"],
            "ghrelin": ["GHSR"],
            "prolactin": ["PRLR"],
            "histamine": ["H1", "H2"],
            "aldosterone": ["MR"],
            "calcitonin": ["CTR"],
            "secretin": ["SCTR"],
            "angiotensin_ii": ["AT1", "AT2"],
            "endorphins": ["MOR", "DOR", "KOR"],
            "melatonin": ["MT1", "MT2"]
        }
        
        # Lobe-specific receptor subtypes
        lobe_specific_subtypes = {
            "task_management": {
                "dopamine": ["D2", "D3"],
                "norepinephrine": ["alpha1", "alpha2", "beta1"],
                "cortisol": ["GR"],
                "insulin": ["IR"],
                "gaba": ["GABA_A", "GABA_B"]
            },
            "memory": {
                "dopamine": ["D1", "D5"],
                "acetylcholine": ["nicotinic", "muscarinic_M1"],
                "vasopressin": ["V1a", "V1b"],
                "cortisol": ["GR", "MR"],
                "growth_hormone": ["GHR"]
            },
            "pattern_recognition": {
                "acetylcholine": ["muscarinic_M1", "muscarinic_M4"],
                "estrogen": ["ERalpha", "ERbeta"],
                "histamine": ["H1"],
                "dopamine": ["D1", "D4"]
            },
            "decision_making": {
                "serotonin": ["5HT1A", "5HT2A"],
                "dopamine": ["D1", "D2"],
                "oxytocin": ["OTR"],
                "testosterone": ["AR"]
            },
            "context_management": {
                "gaba": ["GABA_A", "GABA_B"],
                "acetylcholine": ["muscarinic_M1", "nicotinic"],
                "aldosterone": ["MR"],
                "calcitonin": ["CTR"]
            },
            "scientific_process": {
                "growth_hormone": ["GHR"],
                "acetylcholine": ["muscarinic_M1", "nicotinic"],
                "prolactin": ["PRLR"],
                "secretin": ["SCTR"]
            },
            "error_detection": {
                "histamine": ["H1", "H2"],
                "cortisol": ["GR"],
                "angiotensin_ii": ["AT1", "AT2"]
            },
            "social_intelligence": {
                "oxytocin": ["OTR"],
                "vasopressin": ["V1a"],
                "prolactin": ["PRLR"]
            },
            "resource_management": {
                "insulin": ["IR"],
                "glucagon": ["GR"],
                "leptin": ["LepR"]
            },
            "creativity_engine": {
                "dopamine": ["D2", "D4"],
                "testosterone": ["AR"],
                "endorphins": ["MOR", "DOR"],
                "ghrelin": ["GHSR"]
            }
        }
        
        # Return lobe-specific subtypes if available, otherwise default
        if lobe_name in lobe_specific_subtypes:
            # Merge default with lobe-specific, prioritizing lobe-specific
            result = default_subtypes.copy()
            result.update(lobe_specific_subtypes[lobe_name])
            return result
        else:
            return default_subtypes

    def release_hormone(self, source_lobe: str, hormone: str, quantity: float, 
                       context: Optional[Dict] = None) -> None:
        """
        Release a hormone from a source lobe.
        
        Args:
            source_lobe: Name of the lobe releasing the hormone
            hormone: Name of the hormone to release
            quantity: Amount of hormone to release (0.0-1.0)
            context: Optional context information for learning
        """
        if source_lobe not in self.lobes:
            logging.warning(f"Unknown lobe '{source_lobe}' attempted to release hormone")
            return
            
        if hormone not in self.hormones:
            logging.warning(f"Unknown hormone '{hormone}' attempted to be released")
            return
            
        quantity = min(1.0, max(0.0, quantity))
        
        # Store hormone in vesicles for controlled release
        self._store_in_vesicles(source_lobe, hormone, quantity)
        
        # Immediate local effect (autocrine signaling)
        self._apply_autocrine_effect(source_lobe, hormone, quantity)
        
        # Add to global circulation (endocrine signaling)
        self._add_to_circulation(source_lobe, hormone, quantity)
        
        # Apply to nearby lobes (paracrine signaling)
        self._apply_paracrine_effect(source_lobe, hormone, quantity)
        
        # Log the hormone release
        logging.info(f"Lobe '{source_lobe}' released {quantity:.2f} of '{hormone}'")
        
        # Check for cascade triggers
        self._check_cascade_triggers()
        
        # Broadcast hormone update if event bus is available
        if self.event_bus:
            self.event_bus.emit("hormone_update", self.get_levels())

    def _store_in_vesicles(self, lobe_name: str, hormone: str, quantity: float) -> None:
        """
        Store hormone in vesicles for controlled release.
        
        Args:
            lobe_name: Name of the lobe
            hormone: Name of the hormone
            quantity: Amount of hormone to store
        """
        lobe = self.lobes[lobe_name]
        
        if hormone not in lobe.vesicle_storage:
            lobe.vesicle_storage[hormone] = []
            
        # Create vesicles based on quantity
        vesicle_size = 0.1  # Standard vesicle capacity
        num_vesicles = int(quantity / vesicle_size) + 1
        
        for _ in range(num_vesicles):
            actual_size = min(vesicle_size, quantity)
            quantity -= actual_size
            
            if actual_size <= 0:
                break
                
            vesicle = {
                "content": actual_size,
                "age": 0,
                "release_threshold": random.uniform(0.3, 0.8)
            }
            lobe.vesicle_storage[hormone].append(vesicle)

    def _apply_autocrine_effect(self, lobe_name: str, hormone: str, quantity: float) -> None:
        """
        Apply immediate local effect of hormone on the source lobe.
        
        Args:
            lobe_name: Name of the lobe
            hormone: Name of the hormone
            quantity: Amount of hormone
        """
        lobe = self.lobes[lobe_name]
        
        # Apply receptor sensitivity
        effective_quantity = quantity * lobe.receptor_sensitivity.get(hormone, 0.5) * 0.8
        
        # Update local hormone level
        lobe.local_hormone_levels[hormone] += effective_quantity
        lobe.local_hormone_levels[hormone] = min(1.0, lobe.local_hormone_levels[hormone])

    def _add_to_circulation(self, source_lobe: str, hormone: str, quantity: float) -> None:
        """
        Add hormone to global circulation.
        
        Args:
            source_lobe: Name of the source lobe
            hormone: Name of the hormone
            quantity: Amount of hormone
        """
        # Add to global circulation with reduced effect (systemic dilution)
        self.global_hormone_levels[hormone] += quantity * 0.6
        self.global_hormone_levels[hormone] = min(1.0, self.global_hormone_levels[hormone])
        
        # Update gradient field for spatial diffusion
        source_position = self.lobes[source_lobe].position
        
        if hormone not in self.gradient_fields:
            self.gradient_fields[hormone] = {}
            
        self.gradient_fields[hormone][source_position] = self.gradient_fields[hormone].get(source_position, 0) + quantity

    def _apply_paracrine_effect(self, source_lobe: str, hormone: str, quantity: float) -> None:
        """
        Apply hormone effect to nearby lobes (paracrine signaling).
        
        Args:
            source_lobe: Name of the source lobe
            hormone: Name of the hormone
            quantity: Amount of hormone
        """
        lobe = self.lobes[source_lobe]
        
        # Apply to directly connected lobes
        for neighbor_name in lobe.connected_lobes:
            if neighbor_name in self.lobes:
                neighbor = self.lobes[neighbor_name]
                
                # Calculate distance-based effect
                distance = self._calculate_distance(lobe.position, neighbor.position)
                diffusion_factor = math.exp(-distance * 2)  # Exponential decay with distance
                
                # Apply receptor sensitivity
                effective_quantity = quantity * neighbor.receptor_sensitivity.get(hormone, 0.5) * 0.4 * diffusion_factor
                
                # Update neighbor's local hormone level
                neighbor.local_hormone_levels[hormone] += effective_quantity
                neighbor.local_hormone_levels[hormone] = min(1.0, neighbor.local_hormone_levels[hormone])

    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def _process_loop(self) -> None:
        """
        Main processing loop for hormone decay and diffusion.
        """
        while self._running:
            try:
                # Process hormone decay
                self._process_decay()
                
                # Process hormone diffusion
                self._process_diffusion()
                
                # Process vesicle release
                self._process_vesicle_release()
                
                # Sleep for decay interval
                time.sleep(self.decay_interval)
            except Exception as e:
                logging.error(f"Error in hormone processing loop: {e}")

    def _process_decay(self) -> None:
        """
        Process natural decay of hormones.
        """
        # Decay global hormone levels
        for hormone_name, hormone in self.hormones.items():
            old_level = self.global_hormone_levels[hormone_name]
            
            # Calculate decay based on half-life
            decay_constant = 0.693 / hormone.half_life
            decay_amount = old_level * (1 - math.exp(-decay_constant * self.decay_interval))
            
            self.global_hormone_levels[hormone_name] -= decay_amount
            self.global_hormone_levels[hormone_name] = max(0.0, self.global_hormone_levels[hormone_name])
            
        # Decay local hormone levels in each lobe
        for lobe_name, lobe in self.lobes.items():
            for hormone_name, hormone in self.hormones.items():
                old_level = lobe.local_hormone_levels[hormone_name]
                
                # Calculate decay based on half-life
                decay_constant = 0.693 / hormone.half_life
                decay_amount = old_level * (1 - math.exp(-decay_constant * self.decay_interval))
                
                lobe.local_hormone_levels[hormone_name] -= decay_amount
                lobe.local_hormone_levels[hormone_name] = max(0.0, lobe.local_hormone_levels[hormone_name])

    def _process_diffusion(self) -> None:
        """
        Process diffusion of hormones between lobes.
        """
        for hormone_name, hormone in self.hormones.items():
            # Skip if no gradient field for this hormone
            if hormone_name not in self.gradient_fields:
                continue
                
            # Calculate diffusion for each source point
            for source_pos, initial_quantity in list(self.gradient_fields[hormone_name].items()):
                # Skip if quantity is too small
                if initial_quantity < 0.01:
                    del self.gradient_fields[hormone_name][source_pos]
                    continue
                    
                # Reduce source quantity due to diffusion
                self.gradient_fields[hormone_name][source_pos] *= (1 - hormone.diffusion_rate * self.decay_interval)
                
                # Apply diffusion to all lobes
                for lobe_name, lobe in self.lobes.items():
                    distance = self._calculate_distance(source_pos, lobe.position)
                    
                    # Skip if too far away
                    if distance > 5.0:
                        continue
                        
                    # Calculate diffusion using Gaussian diffusion model
                    diffusion_coeff = hormone.diffusion_rate
                    time_step = self.decay_interval
                    
                    concentration = initial_quantity * math.exp(
                        -(distance**2) / (4 * diffusion_coeff * time_step)
                    ) / math.sqrt(4 * math.pi * diffusion_coeff * time_step)
                    
                    # Apply receptor sensitivity
                    effective_concentration = concentration * lobe.receptor_sensitivity.get(hormone_name, 0.5)
                    
                    # Update local hormone level
                    lobe.local_hormone_levels[hormone_name] += effective_concentration
                    lobe.local_hormone_levels[hormone_name] = min(1.0, lobe.local_hormone_levels[hormone_name])

    def _process_vesicle_release(self) -> None:
        """
        Process controlled release of hormones from vesicles.
        """
        for lobe_name, lobe in self.lobes.items():
            for hormone_name in list(lobe.vesicle_storage.keys()):
                # Skip if no vesicles
                if not lobe.vesicle_storage[hormone_name]:
                    continue
                    
                vesicles_to_remove = []
                total_released = 0.0
                
                for i, vesicle in enumerate(lobe.vesicle_storage[hormone_name]):
                    # Age the vesicle
                    vesicle["age"] += self.decay_interval
                    
                    # Calculate release probability based on age and threshold
                    release_prob = min(1.0, vesicle["age"] / 10.0) * (1 - vesicle["release_threshold"])
                    
                    if random.random() < release_prob:
                        total_released += vesicle["content"]
                        vesicles_to_remove.append(i)
                        
                # Remove released vesicles
                for i in sorted(vesicles_to_remove, reverse=True):
                    del lobe.vesicle_storage[hormone_name][i]
                    
                # Apply released hormones to local level
                if total_released > 0:
                    lobe.local_hormone_levels[hormone_name] += total_released
                    lobe.local_hormone_levels[hormone_name] = min(1.0, lobe.local_hormone_levels[hormone_name])

    def _check_cascade_triggers(self) -> None:
        """
        Check if any hormone cascades should be triggered.
        """
        for cascade_name, cascade in self.cascades.items():
            # Check if trigger conditions are met
            triggered = True
            for hormone, threshold in cascade["trigger_conditions"].items():
                if self.global_hormone_levels.get(hormone, 0) < threshold:
                    triggered = False
                    break
                    
            if triggered:
                self._trigger_cascade(cascade_name, cascade)

    def _trigger_cascade(self, cascade_name: str, cascade: Dict) -> None:
        """
        Trigger a hormone cascade.
        
        Args:
            cascade_name: Name of the cascade
            cascade: Cascade definition
        """
        logging.info(f"Triggering hormone cascade: {cascade_name}")
        
        # Process cascade sequence
        for step in cascade["sequence"]:
            hormone = step["hormone"]
            target_lobes = step["target_lobes"]
            multiplier = step["multiplier"]
            
            # Apply to all lobes if specified
            if "all" in target_lobes:
                target_lobes = list(self.lobes.keys())
                
            # Apply hormone effect to target lobes
            for lobe_name in target_lobes:
                if lobe_name in self.lobes:
                    lobe = self.lobes[lobe_name]
                    
                    # Calculate cascade effect
                    base_level = lobe.local_hormone_levels.get(hormone, 0)
                    cascade_effect = 0.2 * multiplier  # Base cascade effect
                    
                    # Apply to local hormone level
                    lobe.local_hormone_levels[hormone] += cascade_effect
                    lobe.local_hormone_levels[hormone] = min(1.0, lobe.local_hormone_levels[hormone])
                    
                    logging.debug(f"Cascade {cascade_name}: {hormone} in {lobe_name} increased by {cascade_effect:.2f}")
        
        # Process feedback loops
        for feedback in cascade["feedback_loops"]:
            hormone = feedback["hormone"]
            threshold = feedback["threshold"]
            effect = feedback["effect"]
            target = feedback["target"]
            
            # Check if feedback should activate
            if self.global_hormone_levels.get(hormone, 0) >= threshold:
                if effect == "inhibit":
                    # Inhibit target hormone
                    self.global_hormone_levels[target] *= 0.7
                    logging.debug(f"Feedback loop: {hormone} inhibited {target}")
                elif effect == "enhance":
                    # Enhance target hormone
                    self.global_hormone_levels[target] *= 1.3
                    self.global_hormone_levels[target] = min(1.0, self.global_hormone_levels[target])
                    logging.debug(f"Feedback loop: {hormone} enhanced {target}")

    def process_hormone_cascades(self) -> CascadeResult:
        """
        Process all hormone cascades and return the results.
        
        Returns:
            CascadeResult object with cascade processing results
        """
        triggered_cascades = []
        affected_lobes = {}
        feedback_loops = []
        emergent_effects = []
        
        # Check each cascade
        for cascade_name, cascade in self.cascades.items():
            # Check if trigger conditions are met
            triggered = True
            for hormone, threshold in cascade["trigger_conditions"].items():
                if self.global_hormone_levels.get(hormone, 0) < threshold:
                    triggered = False
                    break
                    
            if triggered:
                triggered_cascades.append(cascade_name)
                
                # Track affected lobes
                for step in cascade["sequence"]:
                    hormone = step["hormone"]
                    target_lobes = step["target_lobes"]
                    multiplier = step["multiplier"]
                    
                    # Apply to all lobes if specified
                    if "all" in target_lobes:
                        target_lobes = list(self.lobes.keys())
                        
                    # Record affected lobes
                    for lobe_name in target_lobes:
                        if lobe_name not in affected_lobes:
                            affected_lobes[lobe_name] = {}
                            
                        affected_lobes[lobe_name][hormone] = multiplier
                
                # Track feedback loops
                for feedback in cascade["feedback_loops"]:
                    feedback_loops.append({
                        "cascade": cascade_name,
                        "hormone": feedback["hormone"],
                        "target": feedback["target"],
                        "effect": feedback["effect"]
                    })
                    
        # Check for emergent effects (combinations of hormones)
        if self.global_hormone_levels.get("dopamine", 0) > 0.7 and self.global_hormone_levels.get("oxytocin", 0) > 0.6:
            emergent_effects.append("enhanced_collaboration")
            
        if self.global_hormone_levels.get("cortisol", 0) > 0.8 and self.global_hormone_levels.get("norepinephrine", 0) > 0.7:
            emergent_effects.append("focused_stress_response")
            
        if self.global_hormone_levels.get("acetylcholine", 0) > 0.7 and self.global_hormone_levels.get("growth_hormone", 0) > 0.5:
            emergent_effects.append("accelerated_learning")
            
        return CascadeResult(
            triggered_cascades=triggered_cascades,
            affected_lobes=affected_lobes,
            feedback_loops=feedback_loops,
            emergent_effects=emergent_effects
        )

    def adapt_receptor_sensitivity(self, lobe: str, hormone: str, performance: float,
                                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        Adapt receptor sensitivity based on performance feedback.
        
        Args:
            lobe: Name of the lobe
            hormone: Name of the hormone
            performance: Performance feedback (0.0 to 1.0)
            context: Optional context information
        """
        if lobe not in self.lobes:
            logging.warning(f"Unknown lobe '{lobe}' in adapt_receptor_sensitivity")
            return
            
        if hormone not in self.hormones:
            logging.warning(f"Unknown hormone '{hormone}' in adapt_receptor_sensitivity")
            return
            
        lobe_state = self.lobes[lobe]
        current_sensitivity = lobe_state.receptor_sensitivity.get(hormone, 0.5)
        
        # Normalize performance to 0.0-1.0 range if it's in -1.0 to 1.0 range
        if performance < 0:
            performance = (performance + 1.0) / 2.0
        
        # Use advanced receptor sensitivity model if available
        if hasattr(self, 'receptor_model') and self.receptor_model:
            try:
                new_sensitivity = self.receptor_model.adapt_sensitivity(
                    lobe, hormone, performance, current_sensitivity, context=context
                )
                
                # Update sensitivity
                lobe_state.receptor_sensitivity[hormone] = new_sensitivity
                
                logging.debug(f"Advanced adaptation {hormone} receptor sensitivity in {lobe} from {current_sensitivity:.3f} to {new_sensitivity:.3f}")
                return
                
            except Exception as e:
                logging.warning(f"Advanced receptor adaptation failed, falling back to simple method: {e}")
        
        # Fallback to simple adaptation
        # Record performance for learning
        self.receptor_performance_history[lobe][hormone].append((current_sensitivity, performance))
        
        # Trim history if too long
        if len(self.receptor_performance_history[lobe][hormone]) > 100:
            self.receptor_performance_history[lobe][hormone] = self.receptor_performance_history[lobe][hormone][-100:]
        
        # Simple adaptation based on performance
        if performance > 0.8:
            adjustment = 0.05 * (performance - 0.8)
        elif performance < 0.5:
            adjustment = -0.1 * (0.5 - performance)
        else:
            adjustment = 0.02 * (performance - 0.65)
        
        # Apply context-based modulation
        if context:
            stress_level = context.get('stress_level', 0.0)
            urgency = context.get('urgency', 0.0)
            
            # Increase sensitivity under stress or urgency
            if stress_level > 0.7 or urgency > 0.8:
                adjustment += 0.03
        
        new_sensitivity = current_sensitivity + adjustment
        
        # Constrain sensitivity bounds
        new_sensitivity = max(0.1, min(2.0, new_sensitivity))
        
        # Update sensitivity
        lobe_state.receptor_sensitivity[hormone] = new_sensitivity
        
        logging.debug(f"Simple adaptation {hormone} receptor sensitivity in {lobe} from {current_sensitivity:.3f} to {new_sensitivity:.3f}")

    def learn_optimal_hormone_profiles(self, context: Dict) -> Dict[str, float]:
        """
        Learn optimal hormone profiles for specific contexts.
        
        Args:
            context: Context information
            
        Returns:
            Dictionary of optimal hormone levels for the context
        """
        # Convert context to a string key
        context_key = str(sorted(context.items()))
        
        # Simple context-based hormone profile learning
        # In a real implementation, this would use more sophisticated machine learning
        optimal_profile = {}
        
        # Default baseline profile
        for hormone in self.hormones:
            optimal_profile[hormone] = 0.5
            
        # Adjust based on context type
        if "task_type" in context:
            task_type = context["task_type"]
            
            if task_type == "creative":
                optimal_profile.update({
                    "dopamine": 0.7,
                    "testosterone": 0.6,
                    "endorphins": 0.6,
                    "ghrelin": 0.5,
                    "cortisol": 0.3
                })
            elif task_type == "analytical":
                optimal_profile.update({
                    "acetylcholine": 0.8,
                    "norepinephrine": 0.7,
                    "cortisol": 0.5,
                    "estrogen": 0.6,
                    "dopamine": 0.4
                })
            elif task_type == "social":
                optimal_profile.update({
                    "oxytocin": 0.8,
                    "vasopressin": 0.7,
                    "serotonin": 0.7,
                    "prolactin": 0.6,
                    "dopamine": 0.6
                })
            elif task_type == "emergency":
                optimal_profile.update({
                    "adrenaline": 0.9,
                    "cortisol": 0.8,
                    "norepinephrine": 0.8,
                    "histamine": 0.7,
                    "glucagon": 0.7
                })
                
        # Adjust based on priority
        if "priority" in context:
            priority = context["priority"]
            
            if priority == "high":
                optimal_profile["cortisol"] = min(1.0, optimal_profile.get("cortisol", 0.5) + 0.2)
                optimal_profile["norepinephrine"] = min(1.0, optimal_profile.get("norepinephrine", 0.5) + 0.2)
            elif priority == "low":
                optimal_profile["cortisol"] = max(0.0, optimal_profile.get("cortisol", 0.5) - 0.2)
                optimal_profile["serotonin"] = min(1.0, optimal_profile.get("serotonin", 0.5) + 0.1)
                
        # Adjust based on cognitive load
        if "cognitive_load" in context:
            cognitive_load = context["cognitive_load"]
            
            if cognitive_load == "high":
                optimal_profile["acetylcholine"] = min(1.0, optimal_profile.get("acetylcholine", 0.5) + 0.3)
                optimal_profile["insulin"] = min(1.0, optimal_profile.get("insulin", 0.5) + 0.2)
            elif cognitive_load == "low":
                optimal_profile["gaba"] = min(1.0, optimal_profile.get("gaba", 0.5) + 0.2)
                optimal_profile["melatonin"] = min(1.0, optimal_profile.get("melatonin", 0.5) + 0.1)
                
        return optimal_profile

    def get_levels(self) -> Dict[str, float]:
        """
        Get current global hormone levels.
        
        Returns:
            Dictionary of current hormone levels
        """
        return dict(self.global_hormone_levels)

    def get_lobe_levels(self, lobe_name: str) -> Dict[str, float]:
        """
        Get current hormone levels for a specific lobe.
        
        Args:
            lobe_name: Name of the lobe
            
        Returns:
            Dictionary of current hormone levels for the lobe
        """
        if lobe_name not in self.lobes:
            return {}
            
        return dict(self.lobes[lobe_name].local_hormone_levels)

    def stop(self) -> None:
        """
        Stop the hormone system controller.
        """
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logging.info("HormoneSystemController stopped")