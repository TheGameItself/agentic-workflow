"""
ResourceOptimizationEngine: Dynamic resource optimization based on workload patterns.

This module implements a biologically-inspired resource optimization system that
dynamically adjusts hormone production rates, triggers memory consolidation,
schedules background training, and adapts to resource constraints based on
workload patterns and system state.
"""

import logging
import time
import asyncio
import psutil
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
from collections import deque

from src.mcp.hormone_system_controller import HormoneSystemController
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
from src.mcp.workload_pattern_analyzer import WorkloadPatternAnalyzer, WorkloadPattern, ResourcePrediction
from src.mcp.workload_pattern_analyzer import WorkloadPatternAnalyzer, WorkloadPattern, ResourcePrediction


@dataclass
class ResourceMetrics:
    """Resource usage metrics for the system."""
    cpu_usage: float = 0.0  # CPU usage percentage (0-100)
    memory_usage: float = 0.0  # Memory usage percentage (0-100)
    memory_available: int = 0  # Available memory in bytes
    disk_usage: float = 0.0  # Disk usage percentage (0-100)
    network_usage: float = 0.0  # Network usage in bytes/sec
    io_wait: float = 0.0  # IO wait percentage (0-100)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResourceConstraints:
    """Resource constraints for the system."""
    max_cpu_usage: float = 80.0  # Maximum CPU usage percentage
    max_memory_usage: float = 80.0  # Maximum memory usage percentage
    max_disk_usage: float = 90.0  # Maximum disk usage percentage
    max_network_usage: float = 1000000  # Maximum network usage in bytes/sec
    priority_lobes: List[str] = field(default_factory=list)  # Lobes to prioritize


@dataclass
class TrainingSchedule:
    """Schedule for background training tasks."""
    tasks: List[Dict[str, Any]] = field(default_factory=list)  # Training tasks
    priority: Dict[str, int] = field(default_factory=dict)  # Task priorities
    resource_allocation: Dict[str, float] = field(default_factory=dict)  # Resource alloon
    start_times: Dict[str, str] = field(default_factory=dict)  # Task start times
s


@dataclass
class AdaptationPlan:
    """Plan for adapting to resource constraints."""
    hormone_adjustments: Dict[str, float] = field(default_factory=dict)  # Hormone level adjustments
    lobe_priority_changes: Dict[str, int] = field(default_factory=dict)  # Lobe priority chages
    memory_consolidation_targets: List[str] = field(default_fgets
    background_task_adjustments: Dict[str, str] = field(default_factory=dict)  # Task a



@dataclass
class RecoveryState:
    """State information for recovery from resource constraints."""
    previous_hormone_levels: Dict[str, float] = field(default_factory=dict)   levels
    optimal_hormone_levels: Dict[str, float] = field(default_factory=dict)  # Optimal hormone 
    recovery_start_time: str = field(default_factory=lambda: datetime.now().isofo
    recovery_duration: float = 300.0  # Recovery duration in seconds
)


class ResourceOptimizgine:
    """
    Engine for optimizing resource usage based on workload patterns and system constraints.
    
    This engine dynamically adjusts hormone production rates, triggers memory consolidation,
    schedules background training, and adapts to resource constraints based on workload
    patterns and system state.
 """
 
    def __
                 horNone,
                 brain_state_aggregator: Optional[BrainStateAggregaone,
                 event_bus: Optional[LobeEventBus] = None):
        """
        Initialize the resource optimization engine.
        
        Args:
      n
            brain_state_agg
       ts
        """
    gging
        self.logger = logging.getLogger("ResourceOptimizationEngine")
        
        # Store dependencies
       ler
    tor
        self.event_bus 
        
        # Resource metrics history
        self.metrics_history: deque = deque(maxlen=1000)  #
        sel = {}
        self.current_constraints = ResourceConstrains()
        
        # Hor
        self.hormone_adjustment_active = False
        self.hormone_baseline_levels: Dict[str, float] = {}
        self.hormone_adjustment_factors: Dict[str, float] = {}
        
        # Memory consoliate
        self.memory_consolidation_active = False
        n
        self.last_consolidato
        self.consolidation_cooldown = 300  # 5 minutns
        
        # Recovery state
         = False
        self.recovery_state: Optioe
        
        # Background training state
        self.training_schedule: Optional[TrainingSchedulne
        et()
        
        # Initialize metrics
        self.current_metrics = self._collect_resource_metris()
        
        
        self.pattern_analyzer = Worker()
        
        # Resource prediction state
        self.current_prediction: Optional[ResourcePrediction] = None
        self.last_prediction_time = datetime.now() - timedelta(minutes=5)
        conds
        
        # Constraint adaptation state
        self.adaptation_plans: Dict[str, AdaptationPlan] = {}
         = set()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        self.logger.info("Re")
    
    def f):
        """Set up event handler"
        if self.event_bus:
        t)
            self.event_bus.subscribe("memory_pressure_detected", spressure)
    
            self.event_bus.subscribe)
            self.event_bus.subscribe("resource_recovery_detectedry)
            
            self.logger.info("Resource event handlers registered")
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """
        Collect current resource usage metrics.
        
        Returns:
    n
        """
        try
            # Collect CPU usage
        =0.1)
            
            # Collect memory usage
           
            cent
            memory_available = ble
            
            sk usage
            disk = psutil.disk_usa'/')
            disk_usage = disk.percent
            
            # Collect network usage (simplified)
             0.0
            try:
                net_io = psutil.net_io_co()
                network_usage = net_iecv
            except:
                pass  # Network metrics might noe
            
            # Coed)
            io_wait = 0.0
            try:
                io_wait
            except:
            rms
            
            metrics = Ress(
                sage,
                memory_usage=memory_usage,
                meme,
                disk_usage=disk_usage,
            ge,
                io_wait=io_wait
            )
            
            # Add to history
            self.metrics_history.appen
            
            # Update pattern an
            s(
            ge,
                memory_usagee,
                disk_usage=disk_usage,
            w()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecti}")
           
            return ResourceMetrics()
    
    def update(self):
        ""
        # Collect current metrics
        self.current_metrics = self._collect_rmetrics()
        
        ad
        if self._is_high_computatio
            self._handle_high_computatd()
        
        re
        if self._is_memory_pressure():
            self._handle_memory_pr
        
        
        if self._is_system_idle():
            self._handle_system_idle_internal()
        
        ive
        if self.in_recovery_mode ae:
            self._update_recovery()
        
        # Update workload patterns and predictions
        self._update_workload_patterns()
        
    ediction
        if (datetime.now() - self.last_prediction_
           ()
            self.last_prediction_time = datetime.now()
        
        # Log cupdates)
        if len(self.metrics_history) % 50 == 0:
           )
    
    def _is_high_computational_load(self) -> bool:
        """
    l load.
        
        Ret
            True if high computational load detected,erwise
        
        # Check d
        cpu_threshold = self.current_constraints.max_cpu_usag of max
        retthreshold
    
    def _is_memory_pressure(self) -> bool:
    
        Check if the system is under msure.
        
        Returns:
        herwise
        """
        # Check if memory usage is above threshold
        ret00
    
    def _is_system_idle(self) -> bool:
    
        Check if the system is idle.
        
        Returns:
            True if system is idle, False otherwise
        """
        # Check if CPU usage is below idle threshold
        retue
    
    def _handle_high_computational_load(self):
        """H"""
        if not self.hormone_adjustment_active:
            # Store baseline hormone levels if not already stored
            ler:
                self.hormone_baseline_levels = self.hormone_controller.get_els()
            
            # Calculate adjustment factors based on load
            load_factor = min(1.0, self.current_metrics.cpu_usage / 100.0)
            
            # Higher load = more reduction in non-essential hormones
            reduction_factor = 0.5 + (0.4 * load_factor)  # 0.5 to 0.9 reduction
            
            # Adtion
            self.hormone_adjustment_factors = {
                # Reduce these hormones during high load
                "dopamine": 1.0 - (0.3 * load_factor),  # Reduce reward sign
                "serotonin": 1.0 - (0.2 * load_factor),  # Reduce confidence s
             
            
                "growth_hormone": 1.0 -antly
                
            
                "cortisol": 1.0 + (0.3 *e
                "adrenaline": 1.0 + (0.4 * load_furgency
            
            }
            
            nts
            self._apply_ts()
            
            # Set adjustment active 
            self.hormone_adjustment_active = True
            
            self.logger.info(f"Activated hormone adjustmend "
                           f"(CPU: {self.current_metrics.cpu_usage:.")
            
            # Emit event
            if selfs:
                self.event_bus.emit(
                    "resource_optimization_activated",
                    {
                        "type": "highoad",
                        "cpu_usage": self.current_metrics.cpu_usage,
        rs
                    }
                )
    
    def _handle_memory_pressure_internal(self):
        """Internal handler for memory pressure."""
        # Chedown)
        curr
        time_since_last = (current_time - seonds()
        
    :
            # Trigger memory consolidation
            self.trigger_memory_consolidation(
                self.current_metrics.memory_usage / 100.0,
                self.memory_consolidation_d
            )
            
            # Update last consolidation time
            self.last_consolidation_time = current_time
    
    def _handle_system_idle_internal(self):
        """Internal handler for system idle state."""
        # Sch
        if n
            available_resources = Resource
                cpu_usage=self.current_metrics.cpu_usage,
    
                memory_available=self.current_metrics.memory_ava
           ge,
                network_usage=self.currenage,
        wait
            )
            
           
            self.schedule_background_tras)
    
    def _handle_resource_constraint(self, da Any]):
        "
        Handle resource constraintnt.
        
        Args:
            data: Event data containing constraint information
        """
        # Extract constraint information
        constraint_type = data.get("constraint_type", "unknown")
        s", 0.5)
        
        # Create constraint object
        constraints = ResourceConstraints(
        age),
            max_memory_usage=d),
            max_disk_usage=data.get("max_disk_usage", self.current_constrge),
        e),
            priority_lobes=data.get("priority_lobes", self.current_constraints.priority_lobes)
        )
        
        # Update current constraints
        self.current_constraints = constraints
        
    s
        adaptation_plan = self.adapt_to_resource_constraintsnts)
        
        self.logger.info(f"Adapting t
        
        # Logion plan
        self.logger.info(f"Adaptation plan: {len(adaptation_plan.hoents, "
           ges, "
                       f"{len(adaptation_plan)
    
    def _handle_memory_pressure(self, data: Dict[str, Any]):
        ""
        Handle memory pressure event.
        
        Args:
           
        """
        tion
        memor
        threshold = data.get("threshold", self.memory_cod)
        
        # Trigger memory consolidation
        self.trigger_memory_consolidation(memod)
    
    def _handle_system_idle(self, data: Dict[str, Any]):
        """
        Handle system idle event.
        
        Args:
         ation
        "
        # Extract available resources
        available_resources = ResourceMetrics(
    
            memory_usage=data.get("memory_usage", self.cur
           ,
            disk_usage=data.get("dik_usage),
        age),
            i)
        )
        
        # Schedule background training
        self.schedule_background_training(available
    
    def _handle_high_activity(self, data: Dict[
        """
        Handle high activity event.
        
        Args:
            data: Event data contaiation
        """
        # Exels
        active_lobes = data.get("active_lobes", {})
        
        # Prioritize resources for active l
        if active_lobes:
            resource_allocation = da{})
            updated_allocation = self.prioritize_lces(
                list()),
                resource_allocation
            )
            
            self.
           
            # Emit updated allocation event
           bus:
                self.event_bus.emit(
        pdated",
             {
                        "allocation": updated_allocation,
           e_lobes
                    }
                )
    
    def tr, Any]):
        """
        Handle resource recovery event.
        
        Args:
            data: Event data containing recovery information
        """
        tate
        previous_hormone", {})
        optimal_hormone_levels = data.get("optimal_
     
        # Create recovery state
        recovery_state = RecoveryState(
            previous_hormone_levels=previous_hormone_levels,
            optima
        
        )
        
        ecovery
        self.restore_optimary_state)
    
    def _apply_hormone_adjustments(self):
        """Apply hormone adjustments based on current adjustment fact""
        if not self.hormone_controller or not self.hormone_baseline_level
            return
        
        # Get current hormone levels
        current_levels = self.hormone_controller.get_hormone_le
        
        # Apply adjustments
        for hormone, factor in self.hormone_adjustment_fac
            if hormone in current_levels:
                tor
                baseline = self.hormone_baseline_levels.get(hormon.5)
                target_level = baseline
                
                nge
                target_level = max(0.0
                
                # Calculate difference from current
                curr.0)
                diff = target_level - current
                
                # Applyence)
                adjustment = diff * 0.2
                new_level = current + adjustment
                
        vel
                if abs(adjustmen
                    self.hormone_controller.set_hormone_level(hormone, new_level)
                    
                    self.logger.debug(f"Adjusted {hormone} frof} "
        })")
    
    def _update_recovery(self):
        """Update recovery process if active."""
        
            return
        
        me
        start_time = datetime.fromisoforma
        elapsed = (datetime.now() - start_time).total_seconds()
        total_duration = self.recovery_sturation
        
        # Calculate progress
        progress = min(1.0, elapsed / total_duration)
        self.recs
        
        # Get current hormone levels
        current_levels = self.hormone_controller.get_hormone_levels()
        
        mal levels
        for hormone, optimal in self.retems():
            if hormone in clevels:
                # Calculate target level ress
                previous = self.recovee, 0.0)
            rogress
                
            
                current e, 0.0)
                if abs(target_t
                    self.hormone_con_level)
        
        # Check if re
        if progress >= 1.0:
            self.in_recovery_mode = False
            self.reco= None
            
    e")
            
            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                  ,
        {
                        "timestamp
                        "duration": elapsed
                    }
        
    
    def _update_workload_patterns(self):
        ""
        # Get current pattern from analyzer
        current_pattern = self.pattern_analyzer.get_currern()
        if current_pattern:
        
            
            # Log pattern updates periodically
            if len(self.metrics_history) % 200 == 0:
        "
                if current_patteicity:
                    periodicity_str = f" w"
                
        "
                               f"(con2f})")
    
    def _generate_resource_prediction(sel
        """Generate predi
        # Get prediction from analyzer
        prediction = self.patt()
        if prediction:
            self.current_prediction = prediction
            
        
            self.logger.info(f"Generated re")
            
            # Emit prediction event
            if self.event_bus:
                self.event_bus.emit(
                    "resource_pr
                    {
                        "prediction": {
             ce,
         n_horizon,
        
                            "cpu_trend": "5 
                                                ble",
                            "memory_trend": "increasing" if any(v > self.cu 
                                                             for v in prediction.predicted_memory.stable"
                        },
           )
                    }
           )
            
            # Take proactive actions
            
    
    def _apply_predictive_optimizations(sen):
        """
        Apply optimizations basetion.
        
        :
            prediction: iction
        """
        is too low
        if prediction.confidence < 0.5:
            return
        
        # Check for predicted high CPU usage
        max_predicted_cpu = max(prediction.predicted_cpu.values()) ife 0
        
            # Preemptively reduction
    tive:
                self.logger.info(f"Preemptively reducing hormone production due to p usage "
           
                self._handle_hi)
        
        # Chee
        max_predicted_memory = else 0
        if max_predicted_memory > self.memory_consoold
            lidation
            curr)
            time_since_last = (cseconds()
            
            if timldown:
        "
                               f"({max_predicted_memory)
                self.trigger_memory_consolion(
                    max_predicted_memory / 100.0,
                    self.memoryction
                
                self
    
    def _autocorrelation(self, series: List[float]) -> List[float]:
        """
        Calculate autocorrelation ofes.
        
        Args:
            series: Time seriesta
            
        Returns:
            Lists
        """
        # C
        data = np.arra
        
        # Noata
        data = data - np.mean(data)
        
        # Calculate autocorr
        'full')
        
        # Normalize and take only the posiags
        
        
        return result.tolist()
    
    def _find_peaks(self, series: List[float], thres
        """
        Find peaks in a series.
        
        Args:
            series: Series data
           tection
            
        Returns:
            List of peak indices
        """
        peaks = []
        
        # Skip first few elem
        for i in range(5, len(series) - 1):
            if series[i] > threshold and series[i] > series[i-1] and series[i] > s
                peaks.append(i)
                
        retueaks
    
    def 
        """
        Check if 
    
        Args:
            series: Series data
            threshold: Threshold for burstiness detection
            
        Returns:
        
        """
        if not series:
        lse
            
        # Calculate mean and standard deviation
        ies)
        std = np.std(series)
        
        on
        cv = std / mean if mean > 0 el
        
        # Check for burstiness
        retld
    
    def t]:
        """
        Get currently active lobes and their activity levels.
        
        Returns:
            Dictionary mapping lobe names to activity levels
        """
        or
        if self.brain_state_aggregator:
            # This is a simplified approach 
            brain_state = self.brain_state_aggregator.get_environment_state()
        
            # Extract active lobes (simplified)
            active_lobes = {}
        tate:
                for lobe_name, lobe_state in brain_state.get("lo
                    if isinstance(lobe_state, dict) and "activity" in lobe_s
        vity"]
            
            return active_lobes
        
        # Return empty dict if no brain state aggregator
        return {}
    
    def _log_current_state(self):
        """Log current resource optimization state."""
        self1f}%, "
                       f"Memory: {self.current_metrics}%, "
                       f"Disk: {self.current_metrics.disk_usage:.1f}%")
        
        if self.hormone_adjustment_active:
         )
        
        if self.memory_consolidatioive:
            self.logger.info("Memory cons
        
        if self.in_recovery_mode andate:
            self.logger.info(f"Recovery activ")
    
    def trigger_memory_consolidation(self, memory_usage: float, threshold: float):
        "
        Trigger memo
        
        Args:
            memory_usage: Current memory usage0-1)
            thresion
        """
        if self.memory_consolidation_active:
            return
        
    rue
        
        # Cshold
        severity = min(1.0, (memory_usage - threshold) / (1.0 - threshold 0.2)
        
        # Adjage
        if self.hormone_controller:
            # Increase vasopressin to enhance memory consolidation
           
                "memory", "vasopressin", 0.8 * se, 
                context={"event_typey}
            )
        
            # Reduce growth hormone to slow do
            current_levels = self.hormone_contrevels()
        els:
                new_level = max(0.1, current_levels["growth_h
                self.hormone_contro_level)
        
        self.logger.info(2f}, "
                       f"sever)")
        
        # Emit event
        if sent_bus:
        
                "memory_consolidation_triggered",
          {
                    ,
                    "thresshold,
                    "severity": ty,
                    "timestamp": datetime.now().imat()
                }
            )
        
        # Reset flag after a delay
        def reset):
            salse
        ")
        
        # Schedule reset after 30 seconds
        timer = threading.Timer(30.0, reset_flag)
        timer.daemon = True
        )
    
    def schedule_background_training(self, available_resources: ResourceMetricule:
        """
        Schedule background training tasks based on available resources.
        
        Args:
            available_resources: Available system resources
            
        Returns:
            TrainingSchedule os
        """
        # Calculate available capacity
        available_cpu CPU
        available_memory = max(0.0, 70.0 - available_resources.mry
        
        # Skip if not enough resources available
        if available_:
            reture()
        
        # Create schedule
        schedule = TrainingSchedule()
        
        # Amit
        if available_cpu > 30.0 and available_memory > 30.0:
        training
            s
                "task_id": "neural_model_training",
             {
                25.0,
                    "memory": 20.0
           },
                "estimated_duration": 300.0,  # 5 minutes
                "priority": 1
            })
            
        1
            schedule.resource_allocation["neural_mod
            schedule.start_times["neural_model_training"] = datetime.now().iso
            schedule.duration_estimates["neural_model_training"] = 300.0
        
            self.active_training_tasks.add("neural_model_tra")
        elif avail5.0:
            # Medium re
            schedule.tasks.append({
                "task_iddate",
                "resource_requients": {
        0.0,
                    "memory": 10.0
                },
             s
                "priority": 2
            })
            
            schedule.priority[ = 2
            schedule.resource_allocatiosources
            schedule.start_times["incremen()
            schedule.duration_estimates["incremental_120.0
            
            s
        
        # Add pattern analysis task (ligh
        schedule.tasks.append({
            "task_id": "patter",
            "resource_requirements": {
                "cpu": 5.0,
                "memory": 5.0
            },
            "
            "priority": 3
        })
        
        schedule.priority["pat
        schedule.resource_allocation["p
        schedule.start_times["pattern_anal)
        schedule.duration_estimates["pattern_analysis.0
        
        s
        
        # Store schedule
        self.training_schedule = schedule
        
        self.logger.info(f"Scheduled 
                       f"(available C)")
        
        # Emit event
        :
            self.event_bus.emit(
                "background_training_scheduled",
                {
                le": {
                        "t
                        "tasks": [s],
                
                    },
                    "available_resources": {
                ,
                        "memory": avory
                    },
                    "timestamp": datetime.now().isomat()
                }
            )
        
        return schedule
    
    def predict_workload_pattern(self) -> WorkloadPattern:
        """
        Predict workload pattern based data.
        
        Returns:
            WorkloadPattern object with pred pattern
        """
        # Get current pattern from analyzer
        return s")
    
    def predict_resource_needs(self, workload_pattern: Wor:
        "
        Predict future re.
        
        Args:
            workload_pattern: Wo
            
        Returns:
            ResourcePrediction object with predicds
        """
        alyzer
        return self.pattction()
    
    def nPlan:
        """
        Adapt system behavior to resource constraints.
        
        Args:
            constraints: Rnts
            
        Returns:
            Adaptons
        """
        # Create adaptation plan
        plan = AdaptationPlan()
        
        # Calcularity
        cpu_s/ 
        ))
        memory_severity = max(0.0, min(1.0, (self.
                                     (constrai
        disk_severity = max(0.0, mi/ 
                                   (const0.2)))
        
        # Overall severity
        severity = max(cpuseverity)
        
        # Skip if no constraints are violated
        if s
            return plan
        
        # Adjust hormones based on constraints
        if self.r:
            # Get current hormone levels
            current_levels = self.hormone_controller.get_)
            
            # Ad
            if cpu_severity > 0.0:
                ones
                plan.hormone_adjustment)
                plan.hormone_adjus
                plan.hormone_adjustmentsty))
                
                # Estimatngs
                plan.estimated_resource_savingsrity
            
            if memory_severity > 0.0:
                # Memory 
                plan.
            ty))
                
                # Add memory consolidation targets
        
                
                # Estimate resource savings
                plan.estimated_resource_savings["memory"] = 10.0 + 20.0 * memory_severity
        
        # Adjust lobe priorities
        es:
            #
            for lobe in constraints.priority_lobes:
                plan.lobe_priority_changes[lobe] = 10  # Inc 10
            
            # De lobes
            non_essential_lobes = ["dre"]
           _lobes:
                if lobe not bes:
                    plan.lobe_priority by 5
        
        # Adjust background tasks
        if self.training_schedule:
        s:
                if seve
                    # High severity -l tasks
                    plan.background_taskncel"
                elif severity >
        use tasks
                    plan.background
                else:
                    # Low severity - reduce resource allocation
                    plan.background_task_adjustments[task_id] "
        
        # Apply adaptation plan
        self._apply_adaptation_plan(plan)
        
        return plan
    
    def _apply_adaptation_plan(self, planPlan):
        """
        Apply adaptatsystem.
        
        Args:
            plan: AdaptationPlan object
        """
        # Apply hormone adjustment
        if self.hormone_controller:
        ):
                self.hormone_controller.set_hormone_level)
                self.logger.debug(fnts")
        
        # Apply lobe priority changes
        if self.brain_state_aggregator:
            # In a real istate
            pass
        
        # Apply background task adjustments
        for task_):
        sks:
                self.active_training_tasks.remove(task_id)
        ints")
            elif action == "pause":
    task
                self.logger.info(f"Paused background task {task_id} due to resource constraint")
           :
                # Reduce resource allocation
        
             
                    self.logger.info(f"Reduced resources f
        
        # Triggeneeded
        if plan.memory_consolidation_targets:
           ion(
                self.current_me
                self.memory_consolidation
            )
    
    def restore_optimal_levels(self, recovery_state: RecoveryState):
        """
        Start recovery process to restoe levels.
        
        Args:
            recovery_state: Recovery state information
        """
        self.in_recovery_mode = True
        self.recovery_state = recovery_state
        
        self.logger.info(f"Starting hormone level recoveryf}s)")
        
        # Emit event
        if self.nt_bus:
            self.event_bus.emit(
                "resource_recovery_started",
                {
                    "recovery_state": {
                    tion,
                        "hormone_count": len(recovery_vels)
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def prioritize_lobe_resources(self, active_lobes: List[str], 
            
        """
        Prioritize resourcelobes.
        
        Args:
            active_lobes: List of active lobe names
            ocation
            
        Returns:
            Updated resource allocation
        """
        # Start with current allocation or empty dict
        updated_allocation = resource_allocation.copy() if resour{}
        
        # Calculate total available resources
        total_resources = 100.0
        allocated_resoures())
        available_resources = max(0.0, total_resrces)
        
        # Calculate resources per active lobe
        if a
            resources_per_lobe = available_resources / les)
            
            
            for lobe in active_
                if lobe in updated_allocation:
                    # Increase existing allocation
                5)
                else:
                    # New allocation
                    updated_allocation[lobe] = resources_per_lobe
        
        # Normalize allocations to ensure total <= 100%
        total_updated = sum(updated_allocation.values())
        if total_updated > total_resources:
            scale_factor = total_resources / total_updated
            for lobe in updated_allocation:
                updated_allocation[lobe] *= scale_factor
        
        return updated_allocation
    
    def get_currn]:
        """
        Get 
        
        Returns:
            ResourcePrediction object or None if no prediction avaable
        """
        return self.current_prediction
    
    def get_a:
        """
        Get the adaptation plan for a specific constraint type.
        
        Args:
            constraint_type: Tyraint
            
        Returns:
            ble
        """
        return self.adaptation_plans.get(


# For threading support
import threading