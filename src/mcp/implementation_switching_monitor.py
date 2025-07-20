"""
Implementation Switching Monitor: Tracks and manages implementation switching decisions.

This module provides a centralized system for monitoring and managing implementation
switching decisions across the MCP system, with visualization and reporting capabilities.

References:
- Requirements 1.2, 1.3, 1.4, 1.5 from MCP System Upgrade specification
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .dual_implementation import DualImplementation, DualImplementationRegistry, ImplementationType


@dataclass
class SwitchEvent:
    """Record of an implementation switch event."""
    component: str
    from_implementation: ImplementationType
    to_implementation: ImplementationType
    reason: str
    performance_improvement: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStatus:
    """Current status of a component with dual implementation."""
    component: str
    current_implementation: ImplementationType
    algorithmic_performance: float
    neural_performance: float
    last_switch_time: float
    switch_count: int
    fallback_count: int
    in_fallback_mode: bool
    last_updated: float = field(default_factory=time.time)


class ImplementationSwitchingMonitor:
    """
    Monitors and manages implementation switching decisions across the MCP system.
    
    This class provides a centralized system for tracking implementation switching
    decisions, with visualization and reporting capabilities.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImplementationSwitchingMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the implementation switching monitor."""
        if self._initialized:
            return
            
        self.logger = logging.getLogger("ImplementationSwitchingMonitor")
        self.registry = DualImplementationRegistry()
        
        # Tracking data
        self.switch_events: List[SwitchEvent] = []
        self.component_status: Dict[str, ComponentStatus] = {}
        self.performance_history: Dict[str, List[Tuple[float, float, float]]] = {}  # timestamp, algo_perf, neural_perf
        
        # Configuration
        self.max_history_length = 1000
        self.report_interval = 3600  # seconds
        self.last_report_time = 0.0
        
        # Initialize status for all registered implementations
        self._update_all_statuses()
        
        self.logger.info("Implementation switching monitor initialized")
        self._initialized = True
    
    def _update_all_statuses(self) -> None:
        """Update status for all registered implementations."""
        for name, impl in self.registry.get_all().items():
            self._update_component_status(name, impl)
    
    def _update_component_status(self, name: str, impl: DualImplementation) -> None:
        """
        Update status for a specific component.
        
        Args:
            name: Component name
            impl: Dual implementation instance
        """
        summary = impl.get_performance_summary()
        
        # Calculate performance scores
        algo_perf = 0.0
        if summary["algorithmic_metrics"]["count"] > 0:
            algo_metrics = summary["algorithmic_metrics"]
            algo_perf = (
                algo_metrics["accuracy"] * 0.4 +
                min(1.0, algo_metrics["speed"] / 1000.0) * 0.2 +
                (1.0 - algo_metrics["resource_usage"]) * 0.2 +
                (1.0 - algo_metrics["error_rate"]) * 0.1 +
                algo_metrics["confidence_score"] * 0.1
            )
        
        neural_perf = 0.0
        if summary["neural_metrics"]["count"] > 0:
            neural_metrics = summary["neural_metrics"]
            neural_perf = (
                neural_metrics["accuracy"] * 0.4 +
                min(1.0, neural_metrics["speed"] / 1000.0) * 0.2 +
                (1.0 - neural_metrics["resource_usage"]) * 0.2 +
                (1.0 - neural_metrics["error_rate"]) * 0.1 +
                neural_metrics["confidence_score"] * 0.1
            )
        
        # Find last switch time
        last_switch_time = 0.0
        for event in reversed(self.switch_events):
            if event.component == name:
                last_switch_time = event.timestamp
                break
        
        # Update status
        self.component_status[name] = ComponentStatus(
            component=name,
            current_implementation=ImplementationType(summary["current_implementation"]),
            algorithmic_performance=algo_perf,
            neural_performance=neural_perf,
            last_switch_time=last_switch_time,
            switch_count=summary["switch_count"],
            fallback_count=summary["fallback_count"],
            in_fallback_mode=summary["in_fallback_mode"],
            last_updated=time.time()
        )
        
        # Update performance history
        if name not in self.performance_history:
            self.performance_history[name] = []
        
        self.performance_history[name].append((time.time(), algo_perf, neural_perf))
        
        # Trim history if needed
        if len(self.performance_history[name]) > self.max_history_length:
            self.performance_history[name] = self.performance_history[name][-self.max_history_length:]
    
    def record_switch_event(self, event: SwitchEvent) -> None:
        """
        Record an implementation switch event.
        
        Args:
            event: Switch event to record
        """
        self.switch_events.append(event)
        self.logger.info(
            f"Implementation switch: {event.component} from {event.from_implementation.value} "
            f"to {event.to_implementation.value} (improvement: {event.performance_improvement:.2f})"
        )
        
        # Trim history if needed
        if len(self.switch_events) > self.max_history_length:
            self.switch_events = self.switch_events[-self.max_history_length:]
        
        # Update component status
        impl = self.registry.get(event.component)
        if impl:
            self._update_component_status(event.component, impl)
        
        # Generate report if needed
        current_time = time.time()
        if current_time - self.last_report_time > self.report_interval:
            self.generate_report()
            self.last_report_time = current_time
    
    def update_all(self) -> None:
        """Update status for all registered implementations."""
        self._update_all_statuses()
    
    def get_component_status(self, component: str) -> Optional[ComponentStatus]:
        """
        Get current status for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Component status, or None if not found
        """
        return self.component_status.get(component)
    
    def get_all_component_statuses(self) -> Dict[str, ComponentStatus]:
        """
        Get current status for all components.
        
        Returns:
            Dictionary mapping component names to statuses
        """
        return self.component_status.copy()
    
    def get_performance_history(self, component: str) -> List[Tuple[float, float, float]]:
        """
        Get performance history for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            List of (timestamp, algorithmic_performance, neural_performance) tuples
        """
        return self.performance_history.get(component, [])
    
    def get_switch_events(self, component: Optional[str] = None) -> List[SwitchEvent]:
        """
        Get switch events, optionally filtered by component.
        
        Args:
            component: Component name to filter by, or None for all events
            
        Returns:
            List of switch events
        """
        if component is None:
            return self.switch_events.copy()
        else:
            return [event for event in self.switch_events if event.component == component]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of implementation switching.
        
        Returns:
            Report data
        """
        # Update all statuses first
        self.update_all()
        
        # Calculate statistics
        total_components = len(self.component_status)
        neural_components = sum(1 for status in self.component_status.values() 
                              if status.current_implementation == ImplementationType.NEURAL)
        algorithmic_components = total_components - neural_components
        
        total_switches = sum(status.switch_count for status in self.component_status.values())
        total_fallbacks = sum(status.fallback_count for status in self.component_status.values())
        
        # Calculate average performance
        avg_algo_perf = 0.0
        avg_neural_perf = 0.0
        if total_components > 0:
            avg_algo_perf = sum(status.algorithmic_performance for status in self.component_status.values()) / total_components
            avg_neural_perf = sum(status.neural_performance for status in self.component_status.values()) / total_components
        
        # Identify components with significant performance differences
        significant_diff_threshold = 0.2
        significantly_better_neural = [
            name for name, status in self.component_status.items()
            if status.neural_performance - status.algorithmic_performance > significant_diff_threshold
        ]
        
        significantly_better_algo = [
            name for name, status in self.component_status.items()
            if status.algorithmic_performance - status.neural_performance > significant_diff_threshold
        ]
        
        # Create report
        report = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "total_components": total_components,
            "neural_components": neural_components,
            "algorithmic_components": algorithmic_components,
            "total_switches": total_switches,
            "total_fallbacks": total_fallbacks,
            "average_algorithmic_performance": avg_algo_perf,
            "average_neural_performance": avg_neural_perf,
            "significantly_better_neural": significantly_better_neural,
            "significantly_better_algorithmic": significantly_better_algo,
            "components": {
                name: {
                    "current_implementation": status.current_implementation.value,
                    "algorithmic_performance": status.algorithmic_performance,
                    "neural_performance": status.neural_performance,
                    "switch_count": status.switch_count,
                    "fallback_count": status.fallback_count,
                    "in_fallback_mode": status.in_fallback_mode,
                    "last_switch_time": status.last_switch_time,
                    "last_updated": status.last_updated
                }
                for name, status in self.component_status.items()
            }
        }
        
        # Log report summary
        self.logger.info(
            f"Implementation report: {neural_components}/{total_components} neural, "
            f"{total_switches} switches, {total_fallbacks} fallbacks, "
            f"avg perf: algo={avg_algo_perf:.2f}, neural={avg_neural_perf:.2f}"
        )
        
        return report
    
    def save_report(self, filename: str) -> None:
        """
        Generate a report and save it to a file.
        
        Args:
            filename: File to save the report to
        """
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Saved implementation report to {filename}")
    
    async def monitor_loop(self, interval: float = 60.0) -> None:
        """
        Run a monitoring loop that periodically updates status and generates reports.
        
        Args:
            interval: Update interval in seconds
        """
        self.logger.info(f"Starting implementation monitoring loop with interval {interval}s")
        
        while True:
            try:
                self.update_all()
                
                # Generate report if needed
                current_time = time.time()
                if current_time - self.last_report_time > self.report_interval:
                    self.generate_report()
                    self.last_report_time = current_time
                
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)


# Example usage
async def test_implementation_monitor():
    """Test the implementation switching monitor."""
    from .dual_implementation import ExampleDualImplementation
    
    # Create and register implementations
    impl1 = ExampleDualImplementation("example_doubler")
    impl2 = ExampleDualImplementation("example_tripler")
    
    registry = DualImplementationRegistry()
    registry.register(impl1)
    registry.register(impl2)
    
    # Create monitor
    monitor = ImplementationSwitchingMonitor()
    
    # Process some data
    for i in range(100):
        await impl1.process(i)
        await impl2.process(i)
    
    # Record a switch event
    event = SwitchEvent(
        component="example_doubler",
        from_implementation=ImplementationType.ALGORITHMIC,
        to_implementation=ImplementationType.NEURAL,
        reason="Performance improvement",
        performance_improvement=1.2
    )
    monitor.record_switch_event(event)
    
    # Generate and print report
    report = monitor.generate_report()
    print("\nImplementation Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_implementation_monitor())