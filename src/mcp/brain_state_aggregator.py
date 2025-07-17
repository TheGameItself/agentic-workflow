"""
BrainStateAggregator: Central aggregator for brain state, lobe, hormone, sensory, and vector memory data.
Implements predictive and ratio-based buffering for memory/CPU optimization.
Stub for integration with all major lobes and engines.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus


class BrainStateAggregator:
    def __init__(
        self,
        lobes: Optional[Dict[str, Any]] = None,
        hormone_engine: Any = None,
        sensory_column: Any = None,
        vector_memory: Any = None,
        event_bus: Optional[LobeEventBus] = None,
    ):
        self.lobes = lobes or {}
        self.hormone_engine = hormone_engine
        self.sensory_column = sensory_column
        self.vector_memory = vector_memory
        self.buffers: Dict[str, Any] = {}
        self.logger = logging.getLogger("BrainStateAggregator")
        self.event_bus = event_bus or LobeEventBus()
        self.prefetch_history = []  # Track buffer access patterns
        
        # Hormone level tracking
        self.hormone_levels: Dict[str, float] = {}  # Current hormone levels
        self.hormone_history: Dict[str, List[Dict[str, Any]]] = {}  # Historical hormone data
        self.hormone_thresholds: Dict[str, Dict[str, float]] = {}  # Thresholds for hormone effects
        self.hormone_cascade_history: List[Dict[str, Any]] = []  # History of hormone cascades
        self.hormone_timestamps: Dict[str, str] = {}  # Timestamps of last hormone updates
        self.hormone_trends: Dict[str, List[float]] = {}  # Trend data for hormone levels
        self.hormone_source_tracking: Dict[str, Dict[str, float]] = {}  # Track hormone sources by lobe
        self.max_history_length = 100  # Maximum number of historical entries to keep per hormone
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}  # Component -> Implementation -> Metrics
        self.implementation_history: Dict[str, List[Dict[str, Any]]] = {}  # History of implementation changes
        self.active_implementations: Dict[str, str] = {}  # Currently active implementation for each component
        self.performance_thresholds: Dict[str, Dict[str, float]] = {}  # Thresholds for implementation switching
        self.performance_trends: Dict[str, Dict[str, List[float]]] = {}  # Trend data for performance metrics
        self.last_comparison_time: Dict[str, str] = {}  # Last time implementations were compared
        self.comparison_frequency: Dict[str, int] = {}  # How often to compare implementations (in seconds)
        
        # Initialize comprehensive system
        self.initialize_comprehensive_system()
        
    def update_hormone_levels(self, hormone_levels: Dict[str, float], source_lobe: Optional[str] = None) -> None:
        """
        Update hormone levels in the brain state.
        
        Args:
            hormone_levels: Dictionary mapping hormone names to their levels (0.0-1.0)
            source_lobe: Optional name of the lobe that produced the hormones
        """
        current_time = datetime.now().isoformat()
        
        for hormone, level in hormone_levels.items():
            # Ensure level is within valid range
            level = max(0.0, min(1.0, level))
            
            # Store previous level for trend calculation
            previous_level = self.hormone_levels.get(hormone, 0.0)
            
            # Update current level
            self.hormone_levels[hormone] = level
            
            # Update timestamp
            self.hormone_timestamps[hormone] = current_time
            
            # Track source lobe if provided
            if source_lobe:
                if hormone not in self.hormone_source_tracking:
                    self.hormone_source_tracking[hormone] = {}
                self.hormone_source_tracking[hormone][source_lobe] = level
            
            # Update history
            if hormone not in self.hormone_history:
                self.hormone_history[hormone] = []
                
            history_entry = {
                "level": level,
                "timestamp": current_time,
                "source_lobe": source_lobe,
                "delta": level - previous_level
            }
            
            self.hormone_history[hormone].append(history_entry)
            
            # Limit history size
            if len(self.hormone_history[hormone]) > self.max_history_length:
                self.hormone_history[hormone].pop(0)
                
            # Update trend data
            if hormone not in self.hormone_trends:
                self.hormone_trends[hormone] = []
                
            self.hormone_trends[hormone].append(level)
            
            # Limit trend data size
            if len(self.hormone_trends[hormone]) > self.max_history_length:
                self.hormone_trends[hormone].pop(0)
        
        # Update buffer for hormone data
        self.buffers["hormone"] = self.hormone_levels
        
        # Emit event if event bus is available
        if self.event_bus:
            self.event_bus.emit(
                "hormone_levels_updated",
                {
                    "hormone_levels": self.hormone_levels,
                    "timestamp": current_time,
                    "source_lobe": source_lobe
                }
            )
            
        self.logger.info(f"Updated hormone levels: {', '.join([f'{h}={v:.2f}' for h, v in hormone_levels.items()])}")
    
    def register_hormone_cascade(self, cascade_data: Dict[str, Any]) -> None:
        """
        Register a hormone cascade event in the brain state.
        
        Args:
            cascade_data: Dictionary containing cascade information
        """
        # Add timestamp if not present
        if "timestamp" not in cascade_data:
            cascade_data["timestamp"] = datetime.now().isoformat()
            
        # Add to cascade history
        self.hormone_cascade_history.append(cascade_data)
        
        # Limit cascade history size
        if len(self.hormone_cascade_history) > self.max_history_length:
            self.hormone_cascade_history.pop(0)
            
        # Log cascade
        self.logger.info(f"Registered hormone cascade: {cascade_data.get('name', 'unnamed')} "
                        f"with {len(cascade_data.get('affected_hormones', []))} affected hormones")
    
    def set_hormone_threshold(self, hormone: str, threshold_type: str, value: float) -> None:
        """
        Set a threshold value for a hormone effect.
        
        Args:
            hormone: Name of the hormone
            threshold_type: Type of threshold (e.g., 'activation', 'inhibition')
            value: Threshold value (0.0-1.0)
        """
        # Ensure value is within valid range
        value = max(0.0, min(1.0, value))
        
        # Initialize hormone thresholds if needed
        if hormone not in self.hormone_thresholds:
            self.hormone_thresholds[hormone] = {}
            
        # Set threshold
        self.hormone_thresholds[hormone][threshold_type] = value
        
        self.logger.info(f"Set {threshold_type} threshold for {hormone} to {value:.2f}")
    
    def get_hormone_level(self, hormone: str) -> float:
        """
        Get the current level of a specific hormone.
        
        Args:
            hormone: Name of the hormone
            
        Returns:
            Current hormone level (0.0-1.0), or 0.0 if hormone not found
        """
        return self.hormone_levels.get(hormone, 0.0)
    
    def get_hormone_levels(self) -> Dict[str, float]:
        """
        Get all current hormone levels.
        
        Returns:
            Dictionary mapping hormone names to their current levels
        """
        return self.hormone_levels.copy()
    
    def get_hormone_history(self, hormone: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific hormone.
        
        Args:
            hormone: Name of the hormone
            limit: Optional limit on number of history entries to return
            
        Returns:
            List of historical hormone data entries, or empty list if hormone not found
        """
        history = self.hormone_history.get(hormone, [])
        
        if limit is not None:
            return history[-limit:]
        
        return history
    
    def get_hormone_trend(self, hormone: str, window_size: int = 10) -> Dict[str, Any]:
        """
        Calculate trend information for a hormone over a specified window.
        
        Args:
            hormone: Name of the hormone
            window_size: Number of recent entries to consider
            
        Returns:
            Dictionary with trend information
        """
        trend_data = self.hormone_trends.get(hormone, [])
        
        if not trend_data or len(trend_data) < 2:
            return {
                "direction": "stable",
                "magnitude": 0.0,
                "volatility": 0.0
            }
            
        # Get recent data points
        recent_data = trend_data[-min(window_size, len(trend_data)):]
        
        # Calculate direction and magnitude
        start_value = recent_data[0]
        end_value = recent_data[-1]
        delta = end_value - start_value
        
        # Calculate volatility (standard deviation)
        mean = sum(recent_data) / len(recent_data)
        variance = sum((x - mean) ** 2 for x in recent_data) / len(recent_data)
        volatility = variance ** 0.5
        
        # Determine direction
        if abs(delta) < 0.05:
            direction = "stable"
        elif delta > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
            
        return {
            "direction": direction,
            "magnitude": abs(delta),
            "volatility": volatility,
            "start_value": start_value,
            "end_value": end_value,
            "mean": mean
        }
    
    def get_hormone_context_package(self, lobe_name: str) -> Dict[str, Any]:
        """
        Get hormone-specific context for a lobe.
        
        Args:
            lobe_name: Name of the lobe
            
        Returns:
            Dictionary with hormone context information
        """
        # Get current hormone levels
        hormone_levels = self.get_hormone_levels()
        
        # Get hormone trends
        hormone_trends = {
            hormone: self.get_hormone_trend(hormone)
            for hormone in hormone_levels
        }
        
        # Get hormone thresholds
        hormone_thresholds = self.hormone_thresholds.copy()
        
        # Get recent cascades
        recent_cascades = self.hormone_cascade_history[-10:] if self.hormone_cascade_history else []
        
        # Get lobe-specific hormone sources
        lobe_hormone_sources = {}
        for hormone, sources in self.hormone_source_tracking.items():
            if lobe_name in sources:
                lobe_hormone_sources[hormone] = sources[lobe_name]
        
        return {
            "current_levels": hormone_levels,
            "trends": hormone_trends,
            "thresholds": hormone_thresholds,
            "recent_cascades": recent_cascades,
            "lobe_hormone_sources": lobe_hormone_sources
        }
    
    def update_buffers(self):
        """Predictively prefetch and buffer data from all sources, with ratio-based optimization."""
        # Predictive: Prefetch based on last access pattern
        lobe_names = list(self.lobes.keys())
        if self.prefetch_history:
            # Simple prediction: prefetch the most recently accessed lobe first
            lobe_names = [self.prefetch_history[-1]] + [n for n in lobe_names if n != self.prefetch_history[-1]]
        for lobe_name in lobe_names:
            lobe = self.lobes[lobe_name]
            if hasattr(lobe, "get_state"):
                self.buffers[lobe_name] = lobe.get_state()
                self.prefetch_history.append(lobe_name)
                if len(self.prefetch_history) > 100:
                    self.prefetch_history.pop(0)
        
        # Get hormone levels from engine and update internal state
        if self.hormone_engine and hasattr(self.hormone_engine, "get_levels"):
            hormone_levels = self.hormone_engine.get_levels()
            # Update internal hormone tracking
            self.update_hormone_levels(hormone_levels)
            
            # Get cascade information if available
            if hasattr(self.hormone_engine, "get_recent_cascades"):
                recent_cascades = self.hormone_engine.get_recent_cascades()
                for cascade in recent_cascades:
                    self.register_hormone_cascade(cascade)
        else:
            # Ensure hormone buffer exists even if no hormone engine is available
            self.buffers["hormone"] = self.hormone_levels
            
        if self.sensory_column and hasattr(self.sensory_column, "get_latest"):
            self.buffers["sensory"] = self.sensory_column.get_latest()
        if self.vector_memory and hasattr(self.vector_memory, "get_relevant_vectors"):
            self.buffers["vector_memory"] = self.vector_memory.get_relevant_vectors()
            
        # Ratio-based: Only keep most-accessed buffers (simulate optimization)
        access_counts = {n: self.prefetch_history.count(n) for n in self.buffers}
        sorted_lobes = sorted(access_counts, key=lambda n: access_counts[n], reverse=True)
        
        # Keep top N buffers (simulate memory/CPU optimization)
        # Always keep hormone buffer regardless of access count
        N = min(5, len(sorted_lobes))
        preserved_buffers = {k: self.buffers[k] for k in sorted_lobes[:N]}
        if "hormone" not in preserved_buffers:
            preserved_buffers["hormone"] = self.buffers.get("hormone", {})
        self.buffers = preserved_buffers
        
        self.logger.info("[BrainStateAggregator] Buffers updated (predictive/ratio logic).")
        
        # Event bus: Predictively broadcast buffer update
        self.event_bus.predictive_broadcast(
            event_type="brain_state_update",
            data={
                "buffers": self.buffers, 
                "access_counts": access_counts,
                "hormone_levels": self.hormone_levels,
                "hormone_trends": {h: self.get_hormone_trend(h) for h in self.hormone_levels}
            },
            context={"recent_lobe": self.prefetch_history[-1] if self.prefetch_history else None}
        )
    
    def get_context_package(self, lobe_name: str) -> Dict[str, Any]:
        """Return a context package for a lobe, including all relevant data."""
        return {
            "internal": self.buffers.get(lobe_name),
            "adjacent": {n: v for n, v in self.buffers.items() if n != lobe_name},
            "brain_state": self._get_brain_state(),
            "vector_memory": self.buffers.get("vector_memory"),
            "sensory": self.buffers.get("sensory"),
            "hormone": self.get_hormone_context_package(lobe_name),
            "performance": self.get_performance_context_package(),
            "environment": self.get_environment_state(),
        }
    
    def get_performance_context_package(self) -> Dict[str, Any]:
        """
        Get performance-specific context information.
        
        Returns:
            Dictionary with performance context information
        """
        return {
            "active_implementations": self.active_implementations.copy(),
            "recent_switches": self._get_recent_implementation_switches(),
            "performance_summary": self._get_performance_summary(),
            "neural_availability": getattr(self, 'neural_available', False),
            "fallback_status": getattr(self, 'fallback_triggers', {}).get('fallback_active', False)
        }
    
    def _get_recent_implementation_switches(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent implementation switches across all components."""
        all_switches = []
        
        for component, history in self.implementation_history.items():
            for switch in history[-limit:]:
                switch_info = switch.copy()
                switch_info["component"] = component
                all_switches.append(switch_info)
                
        # Sort by timestamp (most recent first)
        all_switches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return all_switches[:limit]
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics across all components."""
        summary = {}
        
        for component, implementations in self.performance_metrics.items():
            component_summary = {
                "active_implementation": self.active_implementations.get(component, "unknown"),
                "implementations": {}
            }
            
            for impl_type, metrics in implementations.items():
                if metrics and isinstance(metrics, dict):
                    # Extract numeric metrics only
                    numeric_metrics = {k: v for k, v in metrics.items() 
                                     if isinstance(v, (int, float)) and k != "last_updated"}
                    if numeric_metrics:
                        component_summary["implementations"][impl_type] = numeric_metrics
                        
            if component_summary["implementations"]:
                summary[component] = component_summary
                
        return summary

    def _get_brain_state(self) -> Dict[str, Any]:
        """Aggregate global brain state from all sources."""
        # TODO: Implement aggregation logic
        return {k: v for k, v in self.buffers.items()}
    
    def register_implementation_performance(self, 
                                           component: str, 
                                           implementation_type: str, 
                                           metrics: Dict[str, float]) -> None:
        """
        Register performance metrics for neural vs. algorithmic implementations.
        
        Args:
            component: Name of the component being measured
            implementation_type: Type of implementation ('neural' or 'algorithmic')
            metrics: Dictionary of performance metrics
        """
        current_time = datetime.now().isoformat()
        
        # Initialize component data structures if they don't exist
        if component not in self.performance_metrics:
            self.performance_metrics[component] = {}
            self.performance_trends[component] = {}
            self.implementation_history[component] = []
            self.comparison_frequency[component] = 60  # Default to comparing every 60 seconds
            
        # Initialize implementation data structures if they don't exist
        if implementation_type not in self.performance_metrics[component]:
            self.performance_metrics[component][implementation_type] = {}
            self.performance_trends[component][implementation_type] = {}
            
        # Update metrics
        for metric_name, value in metrics.items():
            # Store current value
            self.performance_metrics[component][implementation_type][metric_name] = value
            
            # Initialize trend data if it doesn't exist
            if metric_name not in self.performance_trends[component][implementation_type]:
                self.performance_trends[component][implementation_type][metric_name] = []
                
            # Add to trend data
            self.performance_trends[component][implementation_type][metric_name].append(value)
            
            # Limit trend data size
            if len(self.performance_trends[component][implementation_type][metric_name]) > self.max_history_length:
                self.performance_trends[component][implementation_type][metric_name].pop(0)
        
        # Add timestamp to metrics
        self.performance_metrics[component][implementation_type]["last_updated"] = current_time
        
        # Set active implementation if not already set
        if component not in self.active_implementations:
            self.active_implementations[component] = implementation_type
            
        # Record last comparison time if not already set
        if component not in self.last_comparison_time:
            self.last_comparison_time[component] = current_time
            
        # Check if it's time to compare implementations
        self._check_implementation_comparison(component, current_time)
        
        # Log registration
        self.logger.info(f"Registered {implementation_type} performance for {component}: "
                        f"{', '.join([f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float))])}")
        
        # Emit event if event bus is available
        if self.event_bus:
            self.event_bus.emit(
                "implementation_performance_updated",
                {
                    "component": component,
                    "implementation_type": implementation_type,
                    "metrics": metrics,
                    "timestamp": current_time
                }
            )
    
    def _check_implementation_comparison(self, component: str, current_time: str) -> None:
        """
        Check if it's time to compare implementations and select the best one.
        
        Args:
            component: Name of the component
            current_time: Current timestamp as ISO format string
        """
        # Skip if component doesn't have multiple implementations
        if component not in self.performance_metrics or len(self.performance_metrics[component]) < 2:
            return
            
        # Skip if comparison frequency hasn't been reached
        if component in self.last_comparison_time:
            last_time = datetime.fromisoformat(self.last_comparison_time[component])
            current = datetime.fromisoformat(current_time)
            seconds_elapsed = (current - last_time).total_seconds()
            
            if seconds_elapsed < self.comparison_frequency[component]:
                return
                
        # Compare implementations and select the best one
        best_implementation = self._select_best_implementation(component)
        
        # Update last comparison time
        self.last_comparison_time[component] = current_time
        
        # If the best implementation is different from the current one, switch
        if best_implementation != self.active_implementations.get(component):
            self._switch_implementation(component, best_implementation, current_time)
    
    def _select_best_implementation(self, component: str) -> str:
        """
        Select the best implementation based on performance metrics.
        
        Args:
            component: Name of the component
            
        Returns:
            Name of the best implementation
        """
        if component not in self.performance_metrics:
            return "algorithmic"  # Default to algorithmic if no metrics
            
        implementations = list(self.performance_metrics[component].keys())
        if not implementations:
            return "algorithmic"  # Default to algorithmic if no implementations
            
        if len(implementations) == 1:
            return implementations[0]  # Only one implementation available
            
        # Calculate composite scores for each implementation
        scores = {}
        for impl in implementations:
            metrics = self.performance_metrics[component][impl]
            
            # Skip implementations with no metrics
            if not metrics:
                continue
                
            # Calculate composite score based on common metrics
            # Higher is better for accuracy, lower is better for latency and resource_usage
            score = 0.0
            weight_sum = 0.0
            
            if "accuracy" in metrics and isinstance(metrics["accuracy"], (int, float)):
                score += 0.5 * metrics["accuracy"]  # Weight accuracy more heavily
                weight_sum += 0.5
                
            if "latency" in metrics and isinstance(metrics["latency"], (int, float)):
                # Lower latency is better, so invert
                score += 0.3 * (1.0 - min(1.0, metrics["latency"] / 1000.0))
                weight_sum += 0.3
                
            if "resource_usage" in metrics and isinstance(metrics["resource_usage"], (int, float)):
                # Lower resource usage is better, so invert
                score += 0.2 * (1.0 - min(1.0, metrics["resource_usage"]))
                weight_sum += 0.2
                
            # Normalize score
            if weight_sum > 0:
                scores[impl] = score / weight_sum
            else:
                scores[impl] = 0.0
                
        # If no valid scores, return current implementation or default
        if not scores:
            return self.active_implementations.get(component, "algorithmic")
            
        # Find implementation with highest score
        best_impl = max(scores.items(), key=lambda x: x[1])[0]
        
        # Apply hysteresis to prevent frequent switching
        # Only switch if new implementation is significantly better
        current_impl = self.active_implementations.get(component)
        if current_impl in scores:
            # Require 10% improvement to switch
            if scores[best_impl] < scores[current_impl] * 1.1:
                return current_impl
                
        return best_impl
    
    def _switch_implementation(self, component: str, new_implementation: str, timestamp: str) -> None:
        """
        Switch the active implementation for a component.
        
        Args:
            component: Name of the component
            new_implementation: Name of the new implementation
            timestamp: Timestamp of the switch
        """
        old_implementation = self.active_implementations.get(component)
        
        # Update active implementation
        self.active_implementations[component] = new_implementation
        
        # Initialize component history if needed
        if component not in self.implementation_history:
            self.implementation_history[component] = []
            
        # Prepare metrics data if available
        metrics_data = {}
        if component in self.performance_metrics:
            metrics_data = {
                impl: {k: v for k, v in metrics.items() if k != "last_updated"}
                for impl, metrics in self.performance_metrics[component].items()
            }
        
        # Record in history
        history_entry = {
            "timestamp": timestamp,
            "old_implementation": old_implementation,
            "new_implementation": new_implementation,
            "reason": "performance_based",
            "metrics": metrics_data
        }
        
        self.implementation_history[component].append(history_entry)
        
        # Limit history size
        if len(self.implementation_history[component]) > self.max_history_length:
            self.implementation_history[component].pop(0)
            
        # Log switch
        self.logger.info(f"Switched {component} implementation from {old_implementation} to {new_implementation} "
                        f"based on performance metrics")
        
        # Emit event if event bus is available
        if self.event_bus:
            self.event_bus.emit(
                "implementation_switched",
                {
                    "component": component,
                    "old_implementation": old_implementation,
                    "new_implementation": new_implementation,
                    "timestamp": timestamp,
                    "metrics": history_entry["metrics"]
                }
            )
    
    def set_implementation_threshold(self, component: str, metric: str, value: float) -> None:
        """
        Set a threshold for implementation switching.
        
        Args:
            component: Name of the component
            metric: Name of the metric
            value: Threshold value
        """
        # Initialize component thresholds if needed
        if component not in self.performance_thresholds:
            self.performance_thresholds[component] = {}
            
        # Set threshold
        self.performance_thresholds[component][metric] = value
        
        self.logger.info(f"Set {metric} threshold for {component} to {value:.4f}")
    
    def set_comparison_frequency(self, component: str, seconds: int) -> None:
        """
        Set how often implementations should be compared.
        
        Args:
            component: Name of the component
            seconds: Number of seconds between comparisons
        """
        self.comparison_frequency[component] = max(1, seconds)  # Minimum 1 second
        
        self.logger.info(f"Set comparison frequency for {component} to {seconds} seconds")
    
    def get_active_implementation(self, component: str) -> str:
        """
        Get the currently active implementation for a component.
        
        Args:
            component: Name of the component
            
        Returns:
            Name of the active implementation, or None if component not found
        """
        return self.active_implementations.get(component)
    
    def get_implementation_metrics(self, component: str, implementation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a component implementation.
        
        Args:
            component: Name of the component
            implementation_type: Optional type of implementation, if None returns all implementations
            
        Returns:
            Dictionary of performance metrics
        """
        if component not in self.performance_metrics:
            return {}
            
        if implementation_type is not None:
            return self.performance_metrics[component].get(implementation_type, {}).copy()
            
        return {impl: metrics.copy() for impl, metrics in self.performance_metrics[component].items()}
    
    def get_implementation_history(self, component: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of implementation switches for a component.
        
        Args:
            component: Name of the component
            limit: Optional limit on number of history entries to return
            
        Returns:
            List of implementation switch history entries
        """
        history = self.implementation_history.get(component, [])
        
        if limit is not None:
            return history[-limit:]
            
        return history.copy()
    
    def get_implementation_trend(self, component: str, implementation_type: str, 
                               metric: str, window_size: int = 10) -> Dict[str, Any]:
        """
        Calculate trend information for a performance metric over a specified window.
        
        Args:
            component: Name of the component
            implementation_type: Type of implementation
            metric: Name of the metric
            window_size: Number of recent entries to consider
            
        Returns:
            Dictionary with trend information
        """
        if (component not in self.performance_trends or 
            implementation_type not in self.performance_trends[component] or
            metric not in self.performance_trends[component][implementation_type]):
            return {
                "direction": "stable",
                "magnitude": 0.0,
                "volatility": 0.0
            }
            
        trend_data = self.performance_trends[component][implementation_type][metric]
        
        if not trend_data or len(trend_data) < 2:
            return {
                "direction": "stable",
                "magnitude": 0.0,
                "volatility": 0.0
            }
            
        # Get recent data points
        recent_data = trend_data[-min(window_size, len(trend_data)):]
        
        # Calculate direction and magnitude
        start_value = recent_data[0]
        end_value = recent_data[-1]
        delta = end_value - start_value
        
        # Calculate volatility (standard deviation)
        mean = sum(recent_data) / len(recent_data)
        variance = sum((x - mean) ** 2 for x in recent_data) / len(recent_data)
        volatility = variance ** 0.5
        
        # Determine direction
        if abs(delta) < 0.05:
            direction = "stable"
        elif delta > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
            
        return {
            "direction": direction,
            "magnitude": abs(delta),
            "volatility": volatility,
            "start_value": start_value,
            "end_value": end_value,
            "mean": mean
        }
    
    def get_performance_context_package(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance monitoring context package.
        
        Args:
            component: Optional component name to filter results
            
        Returns:
            Dictionary with performance context information
        """
        # Get active implementations
        active_impls = self.active_implementations.copy()
        
        # Get performance metrics
        if component is not None:
            metrics = {component: self.get_implementation_metrics(component)}
        else:
            metrics = {comp: self.get_implementation_metrics(comp) for comp in self.performance_metrics}
            
        # Get implementation trends
        trends = {}
        for comp in metrics:
            trends[comp] = {}
            for impl in metrics[comp]:
                trends[comp][impl] = {}
                for metric_name in metrics[comp][impl]:
                    if metric_name != "last_updated" and isinstance(metrics[comp][impl][metric_name], (int, float)):
                        trends[comp][impl][metric_name] = self.get_implementation_trend(comp, impl, metric_name)
        
        # Get recent implementation switches
        recent_switches = {}
        for comp in self.implementation_history:
            history = self.implementation_history[comp]
            if history:
                recent_switches[comp] = history[-min(5, len(history)):]
                
        return {
            "active_implementations": active_impls,
            "metrics": metrics,
            "trends": trends,
            "recent_switches": recent_switches,
            "thresholds": self.performance_thresholds.copy()
        }
    
    def get_environment_state(self) -> Dict[str, Any]:
        """
        Get the current environment state for genetic triggers.
        
        Returns:
            Dictionary with environment state information
        """
        return self._collect_environment_state()
        
    def initialize_comprehensive_system(self):
        """Initialize the comprehensive brain state system."""
        # Initialize predictive models
        self._initialize_predictive_models()
        
        # Initialize default hormone thresholds
        self._initialize_default_hormone_thresholds()
        
        self.logger.info("Comprehensive brain state system initialized")
        
    def _initialize_default_hormone_thresholds(self):
        """Initialize default thresholds for hormone effects."""
        default_thresholds = {
            "dopamine": {
                "activation": 0.3,
                "reward": 0.5,
                "learning": 0.6,
                "inhibition": 0.8
            },
            "serotonin": {
                "activation": 0.4,
                "stability": 0.6,
                "inhibition": 0.8
            },
            "cortisol": {
                "activation": 0.3,
                "stress": 0.6,
                "emergency": 0.8
            },
            "oxytocin": {
                "activation": 0.4,
                "trust": 0.7
            },
            "adrenaline": {
                "activation": 0.3,
                "urgency": 0.6,
                "emergency": 0.8
            },
            "gaba": {
                "activation": 0.3,
                "inhibition": 0.5
            }
        }
        
        self.hormone_thresholds = default_thresholds
        
    def _initialize_predictive_models(self):
        """Initialize predictive models and ratio logic with comprehensive neural network fallbacks."""
        self.logger.info("Initializing predictive models with comprehensive implementation")
        
        try:
            # Initialize advanced predictive models with neural network fallbacks
            self.predictive_models = {
                'activity_predictor': {
                    'type': 'hybrid_neural_statistical',
                    'neural_model': None,  # Placeholder for neural network
                    'statistical_fallback': {
                        'window_size': 20,
                        'weights': [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22],  # Exponential weighting
                        'seasonal_adjustment': True,
                        'trend_detection': True
                    },
                    'accuracy': 0.0,
                    'predictions': [],
                    'confidence_intervals': [],
                    'last_update': datetime.now().isoformat(),
                    'performance_metrics': {
                        'mae': 0.0,  # Mean Absolute Error
                        'rmse': 0.0,  # Root Mean Square Error
                        'mape': 0.0   # Mean Absolute Percentage Error
                    }
                },
                'hormone_predictor': {
                    'type': 'time_series_neural',
                    'neural_model': None,
                    'statistical_fallback': {
                        'arima_order': (2, 1, 2),  # ARIMA parameters
                        'seasonal_order': (1, 1, 1, 24),  # Seasonal ARIMA
                        'trend_components': ['linear', 'quadratic'],
                        'volatility_model': 'garch'
                    },
                    'hormone_correlations': {},
                    'cascade_predictions': {},
                    'feedback_loops': []
                }
            }
            
            # Initialize neural network placeholders with fallback detection
            self._setup_neural_fallback_detection()
            
            self.logger.info("Comprehensive predictive models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing predictive models: {e}")
            # Robust fallback with minimal functionality
            self._initialize_minimal_fallback_models()
            
    def _setup_neural_fallback_detection(self):
        """Setup neural network availability detection and fallback mechanisms."""
        try:
            # Check for neural network libraries
            self.neural_available = False
            self.neural_libraries = {}
            
            try:
                import numpy as np
                self.neural_libraries['numpy'] = np
                self.neural_available = True
            except ImportError:
                self.logger.warning("NumPy not available - using basic fallbacks")
            
            # Setup fallback triggers
            self.fallback_triggers = {
                'neural_failure_count': 0,
                'max_failures': 3,
                'performance_threshold': 0.6,
                'fallback_active': not self.neural_available
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up neural fallback detection: {e}")
            self.neural_available = False
            self.fallback_triggers = {'fallback_active': True}
            
    def _initialize_minimal_fallback_models(self):
        """Initialize minimal fallback models when full initialization fails."""
        self.predictive_models = {
            'activity_predictor': {'type': 'simple_average', 'predictions': []},
            'hormone_predictor': {'type': 'last_value', 'last_values': {}}
        }
        self.performance_tracker = {'fallback_mode': True}
    
    def _collect_environment_state(self) -> Dict[str, Any]:
        """
        Collect environment state from various system components.
        
        Returns:
            Dictionary with consolidated environment state information
        """
        current_time = datetime.now().isoformat()
        environment_state = {
            "timestamp": current_time,
            "system_state": {},
            "lobe_states": {},
            "hormone_state": {},
            "performance_state": {},
            "resource_state": {},
            "sensory_state": {},
            "memory_state": {}
        }
        
        try:
            # Collect system-level state
            environment_state["system_state"] = self._collect_system_state()
            
            # Collect lobe states
            environment_state["lobe_states"] = self._collect_lobe_states()
            
            # Collect hormone state
            environment_state["hormone_state"] = self._collect_hormone_state()
            
            # Collect performance state
            environment_state["performance_state"] = self._collect_performance_state()
            
            # Collect resource state
            environment_state["resource_state"] = self._collect_resource_state()
            
            # Collect sensory state
            environment_state["sensory_state"] = self._collect_sensory_state()
            
            # Collect memory state
            environment_state["memory_state"] = self._collect_memory_state()
            
        except Exception as e:
            self.logger.error(f"Error collecting environment state: {e}")
            environment_state["error"] = str(e)
            
        return environment_state
    
    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect system-level state information."""
        import psutil
        import os
        
        try:
            system_state = {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
                "process_count": len(psutil.pids()),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0],
                "uptime": psutil.boot_time(),
                "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
                "active_implementations": self.active_implementations.copy(),
                "buffer_count": len(self.buffers),
                "event_bus_active": self.event_bus is not None
            }
        except ImportError:
            # Fallback if psutil is not available
            system_state = {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "process_count": 0,
                "load_average": [0.0, 0.0, 0.0],
                "uptime": 0,
                "network_io": {},
                "active_implementations": self.active_implementations.copy(),
                "buffer_count": len(self.buffers),
                "event_bus_active": self.event_bus is not None,
                "psutil_unavailable": True
            }
        except Exception as e:
            system_state = {
                "error": str(e),
                "active_implementations": self.active_implementations.copy(),
                "buffer_count": len(self.buffers),
                "event_bus_active": self.event_bus is not None
            }
            
        return system_state
    
    def _collect_lobe_states(self) -> Dict[str, Any]:
        """Collect state information from all registered lobes."""
        lobe_states = {}
        
        for lobe_name, lobe in self.lobes.items():
            try:
                lobe_state = {
                    "name": lobe_name,
                    "active": True,
                    "last_activity": None,
                    "state_data": None,
                    "hormone_production": {},
                    "hormone_consumption": {},
                    "performance_metrics": {}
                }
                
                # Get lobe state if available
                if hasattr(lobe, "get_state"):
                    lobe_state["state_data"] = lobe.get_state()
                    
                # Get lobe activity information if available
                if hasattr(lobe, "get_last_activity"):
                    lobe_state["last_activity"] = lobe.get_last_activity()
                    
                # Get hormone production information
                if lobe_name in self.hormone_source_tracking:
                    lobe_state["hormone_production"] = self.hormone_source_tracking[lobe_name].copy()
                    
                # Get performance metrics for this lobe
                if lobe_name in self.performance_metrics:
                    lobe_state["performance_metrics"] = self.performance_metrics[lobe_name].copy()
                    
                # Check if lobe is in buffers (indicates recent activity)
                lobe_state["in_buffer"] = lobe_name in self.buffers
                lobe_state["access_count"] = self.prefetch_history.count(lobe_name)
                
                lobe_states[lobe_name] = lobe_state
                
            except Exception as e:
                lobe_states[lobe_name] = {
                    "name": lobe_name,
                    "active": False,
                    "error": str(e)
                }
                
        return lobe_states
    
    def _collect_hormone_state(self) -> Dict[str, Any]:
        """Collect current hormone system state."""
        hormone_state = {
            "current_levels": self.hormone_levels.copy(),
            "recent_changes": {},
            "cascade_activity": len(self.hormone_cascade_history),
            "threshold_violations": [],
            "trend_analysis": {},
            "source_distribution": self.hormone_source_tracking.copy(),
            "total_hormones": len(self.hormone_levels)
        }
        
        # Analyze recent changes
        for hormone, history in self.hormone_history.items():
            if history:
                recent_entries = history[-5:]  # Last 5 entries
                if len(recent_entries) >= 2:
                    change = recent_entries[-1]["level"] - recent_entries[0]["level"]
                    hormone_state["recent_changes"][hormone] = {
                        "change": change,
                        "direction": "increasing" if change > 0 else "decreasing" if change < 0 else "stable",
                        "volatility": self._calculate_volatility(recent_entries)
                    }
                    
        # Check for threshold violations
        for hormone, level in self.hormone_levels.items():
            if hormone in self.hormone_thresholds:
                for threshold_type, threshold_value in self.hormone_thresholds[hormone].items():
                    if level > threshold_value:
                        hormone_state["threshold_violations"].append({
                            "hormone": hormone,
                            "threshold_type": threshold_type,
                            "current_level": level,
                            "threshold_value": threshold_value,
                            "excess": level - threshold_value
                        })
                        
        # Get trend analysis for each hormone
        for hormone in self.hormone_levels:
            hormone_state["trend_analysis"][hormone] = self.get_hormone_trend(hormone)
            
        return hormone_state
    
    def _collect_performance_state(self) -> Dict[str, Any]:
        """Collect performance metrics and implementation state."""
        performance_state = {
            "active_implementations": self.active_implementations.copy(),
            "recent_switches": [],
            "performance_summary": {},
            "comparison_due": {},
            "neural_availability": getattr(self, 'neural_available', False),
            "fallback_status": getattr(self, 'fallback_triggers', {}).get('fallback_active', False)
        }
        
        # Get recent implementation switches
        for component, history in self.implementation_history.items():
            if history:
                recent_switches = history[-3:]  # Last 3 switches
                performance_state["recent_switches"].extend([
                    {
                        "component": component,
                        "timestamp": switch["timestamp"],
                        "from": switch["old_implementation"],
                        "to": switch["new_implementation"],
                        "reason": switch["reason"]
                    }
                    for switch in recent_switches
                ])
                
        # Summarize performance metrics
        for component, implementations in self.performance_metrics.items():
            component_summary = {}
            for impl_type, metrics in implementations.items():
                if metrics and isinstance(metrics, dict):
                    # Extract numeric metrics only
                    numeric_metrics = {k: v for k, v in metrics.items() 
                                     if isinstance(v, (int, float)) and k != "last_updated"}
                    if numeric_metrics:
                        component_summary[impl_type] = numeric_metrics
                        
            if component_summary:
                performance_state["performance_summary"][component] = component_summary
                
        # Check which components are due for comparison
        current_time = datetime.now()
        for component, last_comparison in self.last_comparison_time.items():
            if component in self.comparison_frequency:
                last_time = datetime.fromisoformat(last_comparison)
                seconds_since = (current_time - last_time).total_seconds()
                frequency = self.comparison_frequency[component]
                
                performance_state["comparison_due"][component] = {
                    "seconds_since_last": seconds_since,
                    "frequency": frequency,
                    "due": seconds_since >= frequency
                }
                
        return performance_state
    
    def _collect_resource_state(self) -> Dict[str, Any]:
        """Collect resource usage and availability information."""
        resource_state = {
            "memory_usage": {},
            "computational_load": {},
            "storage_usage": {},
            "buffer_efficiency": {},
            "predictive_model_status": {}
        }
        
        try:
            # Memory usage analysis
            resource_state["memory_usage"] = {
                "hormone_history_size": sum(len(history) for history in self.hormone_history.values()),
                "performance_history_size": sum(len(history) for history in self.implementation_history.values()),
                "buffer_size": len(self.buffers),
                "trend_data_size": sum(len(trends) for component_trends in self.performance_trends.values() 
                                     for trends in component_trends.values() 
                                     for trends in trends.values()),
                "cascade_history_size": len(self.hormone_cascade_history)
            }
            
            # Computational load indicators
            resource_state["computational_load"] = {
                "active_lobes": len([lobe for lobe in self.lobes.values() if hasattr(lobe, "get_state")]),
                "hormone_calculations": len(self.hormone_levels),
                "performance_comparisons": len(self.performance_metrics),
                "predictive_models": len(getattr(self, 'predictive_models', {}))
            }
            
            # Buffer efficiency metrics
            if self.prefetch_history:
                unique_accesses = len(set(self.prefetch_history))
                total_accesses = len(self.prefetch_history)
                resource_state["buffer_efficiency"] = {
                    "hit_ratio": unique_accesses / total_accesses if total_accesses > 0 else 0,
                    "total_accesses": total_accesses,
                    "unique_accesses": unique_accesses,
                    "buffer_count": len(self.buffers)
                }
                
            # Predictive model status
            if hasattr(self, 'predictive_models'):
                for model_name, model_data in self.predictive_models.items():
                    resource_state["predictive_model_status"][model_name] = {
                        "type": model_data.get("type", "unknown"),
                        "accuracy": model_data.get("accuracy", 0.0),
                        "last_update": model_data.get("last_update", "never"),
                        "neural_available": model_data.get("neural_model") is not None
                    }
                    
        except Exception as e:
            resource_state["error"] = str(e)
            
        return resource_state
    
    def _collect_sensory_state(self) -> Dict[str, Any]:
        """Collect sensory input and processing state."""
        sensory_state = {
            "sensory_column_active": self.sensory_column is not None,
            "latest_sensory_data": None,
            "sensory_processing_status": "unknown"
        }
        
        try:
            if self.sensory_column:
                if hasattr(self.sensory_column, "get_latest"):
                    sensory_state["latest_sensory_data"] = self.sensory_column.get_latest()
                    sensory_state["sensory_processing_status"] = "active"
                    
                if hasattr(self.sensory_column, "get_status"):
                    sensory_state["detailed_status"] = self.sensory_column.get_status()
                    
                if hasattr(self.sensory_column, "get_processing_metrics"):
                    sensory_state["processing_metrics"] = self.sensory_column.get_processing_metrics()
            else:
                sensory_state["sensory_processing_status"] = "inactive"
                
        except Exception as e:
            sensory_state["error"] = str(e)
            sensory_state["sensory_processing_status"] = "error"
            
        return sensory_state
    
    def _collect_memory_state(self) -> Dict[str, Any]:
        """Collect memory system state and vector storage information."""
        memory_state = {
            "vector_memory_active": self.vector_memory is not None,
            "memory_capacity": "unknown",
            "memory_usage": "unknown",
            "vector_count": 0,
            "memory_quality": "unknown"
        }
        
        try:
            if self.vector_memory:
                if hasattr(self.vector_memory, "get_relevant_vectors"):
                    vectors = self.vector_memory.get_relevant_vectors()
                    if vectors:
                        memory_state["vector_count"] = len(vectors) if isinstance(vectors, (list, tuple)) else 1
                        
                if hasattr(self.vector_memory, "get_capacity"):
                    memory_state["memory_capacity"] = self.vector_memory.get_capacity()
                    
                if hasattr(self.vector_memory, "get_usage"):
                    memory_state["memory_usage"] = self.vector_memory.get_usage()
                    
                if hasattr(self.vector_memory, "get_quality_metrics"):
                    memory_state["memory_quality"] = self.vector_memory.get_quality_metrics()
                    
                if hasattr(self.vector_memory, "get_status"):
                    memory_state["detailed_status"] = self.vector_memory.get_status()
                    
        except Exception as e:
            memory_state["error"] = str(e)
            
        return memory_state
    
    def _calculate_volatility(self, entries: List[Dict[str, Any]]) -> float:
        """Calculate volatility from a list of hormone history entries."""
        if len(entries) < 2:
            return 0.0
            
        levels = [entry["level"] for entry in entries]
        mean = sum(levels) / len(levels)
        variance = sum((level - mean) ** 2 for level in levels) / len(levels)
        return variance ** 0.5