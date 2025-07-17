"""
Performance Tracker for Neural Network Models component.

This module provides functionality for tracking and analyzing performance metrics
for neural network models, including metrics collection, history storage, and
comparative analysis against algorithmic implementations.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum


@dataclass
class PerformanceMetrics:
    """Performance metrics for a neural model."""
    function_name: str
    implementation_type: str  # "neural" or "algorithmic"
    accuracy: float
    latency: float  # in milliseconds
    resource_usage: float  # 0.0-1.0 scale
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "function_name": self.function_name,
            "implementation_type": self.implementation_type,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "resource_usage": self.resource_usage,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
        
        # Add additional metrics
        for key, value in self.additional_metrics.items():
            result[key] = value
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        # Extract standard fields
        standard_fields = {
            "function_name": data["function_name"],
            "implementation_type": data["implementation_type"],
            "accuracy": data["accuracy"],
            "latency": data["latency"],
            "resource_usage": data["resource_usage"],
            "timestamp": datetime.fromisoformat(data["timestamp"]),
            "context": data.get("context", {})
        }
        
        # Extract additional metrics
        additional_metrics = {}
        for key, value in data.items():
            if key not in standard_fields and isinstance(value, (int, float)):
                additional_metrics[key] = value
                
        return cls(
            **standard_fields,
            additional_metrics=additional_metrics
        )


@dataclass
class PerformanceComparison:
    """Comparison between neural and algorithmic implementations."""
    function_name: str
    timestamp: datetime
    neural_metrics: Optional[Dict[str, float]] = None
    algorithmic_metrics: Optional[Dict[str, float]] = None
    neural_score: float = 0.0
    algorithmic_score: float = 0.0
    recommended_implementation: str = "algorithmic"
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "timestamp": self.timestamp.isoformat(),
            "neural_metrics": self.neural_metrics,
            "algorithmic_metrics": self.algorithmic_metrics,
            "neural_score": self.neural_score,
            "algorithmic_score": self.algorithmic_score,
            "recommended_implementation": self.recommended_implementation,
            "reason": self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceComparison':
        """Create from dictionary."""
        return cls(
            function_name=data["function_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            neural_metrics=data.get("neural_metrics"),
            algorithmic_metrics=data.get("algorithmic_metrics"),
            neural_score=data.get("neural_score", 0.0),
            algorithmic_score=data.get("algorithmic_score", 0.0),
            recommended_implementation=data.get("recommended_implementation", "algorithmic"),
            reason=data.get("reason", "")
        )


class PerformanceTracker:
    """
    Tracks and analyzes performance metrics for neural network models.
    
    This class collects performance metrics for neural models, maintains historical
    data, and provides comparative analysis against algorithmic implementations.
    """
    
    def __init__(self, metrics_dir: str = None, max_history_length: int = 100):
        """
        Initialize the performance tracker.
        
        Args:
            metrics_dir: Directory where metrics are stored. If None, defaults to
                        a 'metrics' directory in the current working directory.
            max_history_length: Maximum number of historical entries to keep per function.
        """
        self.logger = logging.getLogger("PerformanceTracker")
        self.metrics_dir = metrics_dir or os.path.join(os.getcwd(), "metrics")
        self.max_history_length = max_history_length
        
        # Performance metrics storage
        self.current_metrics: Dict[str, Dict[str, PerformanceMetrics]] = defaultdict(dict)
        self.metrics_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_history_length)))
        self.comparison_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_length))
        
        # Trend analysis
        self.trends: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Anomaly detection
        self.baseline_metrics: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        self.anomaly_thresholds: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Load existing metrics if available
        self._load_existing_metrics()
        
        self.logger.info(f"PerformanceTracker initialized with metrics directory: {self.metrics_dir}")
    
    def _load_existing_metrics(self):
        """Load existing metrics from disk."""
        try:
            # Check if metrics directory exists
            if not os.path.exists(self.metrics_dir):
                return
                
            # Load metrics for each function
            for filename in os.listdir(self.metrics_dir):
                if filename.endswith(".metrics"):
                    function_name = filename.split(".")[0]
                    filepath = os.path.join(self.metrics_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            metrics_data = json.load(f)
                            
                        # Process metrics data
                        if "current_metrics" in metrics_data:
                            for impl_type, metrics in metrics_data["current_metrics"].items():
                                self.current_metrics[function_name][impl_type] = PerformanceMetrics.from_dict(metrics)
                                
                        if "metrics_history" in metrics_data:
                            for impl_type, history in metrics_data["metrics_history"].items():
                                for entry in history:
                                    self.metrics_history[function_name][impl_type].append(PerformanceMetrics.from_dict(entry))
                                    
                        if "comparison_history" in metrics_data:
                            for entry in metrics_data["comparison_history"]:
                                self.comparison_history[function_name].append(PerformanceComparison.from_dict(entry))
                                
                        if "baseline_metrics" in metrics_data:
                            self.baseline_metrics[function_name] = metrics_data["baseline_metrics"]
                                
                        if "anomaly_thresholds" in metrics_data:
                            self.anomaly_thresholds[function_name] = metrics_data["anomaly_thresholds"]
                                
                        self.logger.info(f"Loaded metrics for {function_name} from {filepath}")
                        
                    except Exception as e:
                        self.logger.error(f"Error loading metrics for {function_name}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error loading existing metrics: {e}")
        

    def record_metrics(self, 
                      function_name: str, 
                      implementation_type: str, 
                      accuracy: float, 
                      latency: float, 
                      resource_usage: float, 
                      context: Dict[str, Any] = None, 
                      additional_metrics: Dict[str, float] = None) -> bool:
        """
        Record performance metrics for a model.
        
        Args:
            function_name: Name of the function being measured.
            implementation_type: Type of implementation ('neural' or 'algorithmic').
            accuracy: Accuracy metric (0.0-1.0).
            latency: Processing time in milliseconds.
            resource_usage: Resource usage (0.0-1.0).
            context: Additional context information.
            additional_metrics: Additional performance metrics.
            
        Returns:
            True if metrics were recorded successfully, False otherwise.
        """
        try:
            with self._lock:
                # Create metrics object
                metrics = PerformanceMetrics(
                    function_name=function_name,
                    implementation_type=implementation_type,
                    accuracy=accuracy,
                    latency=latency,
                    resource_usage=resource_usage,
                    timestamp=datetime.now(),
                    context=context or {},
                    additional_metrics=additional_metrics or {}
                )
                
                # Update current metrics
                self.current_metrics[function_name][implementation_type] = metrics
                
                # Add to history
                self.metrics_history[function_name][implementation_type].append(metrics)
                
                # Update trends
                self._update_trends(function_name, implementation_type, metrics)
                
                # Check for anomalies
                anomalies = self._check_anomalies(function_name, implementation_type, metrics)
                if anomalies:
                    self.logger.warning(f"Anomalies detected for {function_name} ({implementation_type}): {anomalies}")
                
                # Compare implementations if both neural and algorithmic metrics are available
                if "neural" in self.current_metrics[function_name] and "algorithmic" in self.current_metrics[function_name]:
                    self._compare_implementations(function_name)
                
                self.logger.info(f"Recorded {implementation_type} metrics for {function_name}: "
                               f"accuracy={accuracy:.4f}, latency={latency:.2f}ms, "
                               f"resource_usage={resource_usage:.4f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error recording metrics for {function_name}: {e}")
            return False
    
    def _update_trends(self, function_name: str, implementation_type: str, metrics: PerformanceMetrics):
        """
        Update trend data for a function.
        
        Args:
            function_name: Name of the function.
            implementation_type: Type of implementation.
            metrics: Performance metrics.
        """
        # Update standard metrics trends
        self.trends[function_name][implementation_type]["accuracy"].append(metrics.accuracy)
        self.trends[function_name][implementation_type]["latency"].append(metrics.latency)
        self.trends[function_name][implementation_type]["resource_usage"].append(metrics.resource_usage)
        
        # Update additional metrics trends
        for metric_name, value in metrics.additional_metrics.items():
            self.trends[function_name][implementation_type][metric_name].append(value)
            
        # Limit trend data size
        for metric_name, values in self.trends[function_name][implementation_type].items():
            if len(values) > self.max_history_length:
                self.trends[function_name][implementation_type][metric_name] = values[-self.max_history_length:]    

    def _check_anomalies(self, function_name: str, implementation_type: str, metrics: PerformanceMetrics) -> Dict[str, float]:
        """
        Check for anomalies in performance metrics.
        
        Args:
            function_name: Name of the function.
            implementation_type: Type of implementation.
            metrics: Performance metrics.
            
        Returns:
            Dictionary of detected anomalies and their severity.
        """
        anomalies = {}
        
        # Skip if no baseline metrics are available
        if function_name not in self.baseline_metrics or implementation_type not in self.baseline_metrics[function_name]:
            return anomalies
            
        baseline = self.baseline_metrics[function_name][implementation_type]
        thresholds = self.anomaly_thresholds.get(function_name, {}).get(implementation_type, {})
        
        # Check standard metrics
        for metric_name in ["accuracy", "latency", "resource_usage"]:
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                current_value = getattr(metrics, metric_name)
                threshold = thresholds.get(metric_name, 0.2)  # Default 20% threshold
                
                # For accuracy, lower is worse; for latency and resource_usage, higher is worse
                if metric_name == "accuracy":
                    if baseline_value - current_value > threshold * baseline_value:
                        anomalies[metric_name] = (baseline_value - current_value) / baseline_value
                else:
                    if current_value - baseline_value > threshold * baseline_value:
                        anomalies[metric_name] = (current_value - baseline_value) / baseline_value
        
        # Check additional metrics
        for metric_name, current_value in metrics.additional_metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                threshold = thresholds.get(metric_name, 0.2)  # Default 20% threshold
                
                # Assume higher deviation is worse for additional metrics
                if abs(current_value - baseline_value) > threshold * abs(baseline_value):
                    anomalies[metric_name] = abs(current_value - baseline_value) / abs(baseline_value)
                    
        return anomalies
    
    def _compare_implementations(self, function_name: str) -> PerformanceComparison:
        """
        Compare neural and algorithmic implementations.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Performance comparison result.
        """
        neural_metrics = self.current_metrics[function_name].get("neural")
        algo_metrics = self.current_metrics[function_name].get("algorithmic")
        
        if not neural_metrics or not algo_metrics:
            return None
            
        # Extract metrics for comparison
        neural_data = {
            "accuracy": neural_metrics.accuracy,
            "latency": neural_metrics.latency,
            "resource_usage": neural_metrics.resource_usage
        }
        
        algo_data = {
            "accuracy": algo_metrics.accuracy,
            "latency": algo_metrics.latency,
            "resource_usage": algo_metrics.resource_usage
        }
        
        # Calculate composite scores
        # Higher accuracy is better, lower latency and resource usage are better
        neural_score = (0.6 * neural_data["accuracy"] + 
                       0.3 * (1.0 - min(1.0, neural_data["latency"] / 1000)) + 
                       0.1 * (1.0 - neural_data["resource_usage"]))
                       
        algo_score = (0.6 * algo_data["accuracy"] + 
                     0.3 * (1.0 - min(1.0, algo_data["latency"] / 1000)) + 
                     0.1 * (1.0 - algo_data["resource_usage"]))
        
        # Determine recommended implementation with hysteresis
        # Require 10% improvement to switch from algorithmic to neural
        if neural_score > algo_score * 1.1:
            recommended = "neural"
            reason = f"Neural implementation outperforms algorithmic by {(neural_score/algo_score - 1)*100:.1f}%"
        elif algo_score > neural_score:
            recommended = "algorithmic"
            reason = f"Algorithmic implementation outperforms neural by {(algo_score/neural_score - 1)*100:.1f}%"
        else:
            # If scores are close, stick with algorithmic for stability
            recommended = "algorithmic"
            reason = "Scores are similar, using algorithmic implementation for stability"
        
        # Create comparison result
        comparison = PerformanceComparison(
            function_name=function_name,
            timestamp=datetime.now(),
            neural_metrics=neural_data,
            algorithmic_metrics=algo_data,
            neural_score=neural_score,
            algorithmic_score=algo_score,
            recommended_implementation=recommended,
            reason=reason
        )
        
        # Add to history
        self.comparison_history[function_name].append(comparison)
        
        self.logger.info(f"Compared implementations for {function_name}: "
                       f"neural={neural_score:.4f}, algorithmic={algo_score:.4f}, "
                       f"recommended={recommended}")
                       
        return comparison    
  
  def get_current_metrics(self, function_name: str, implementation_type: str = None) -> Union[Dict[str, PerformanceMetrics], PerformanceMetrics, None]:
        """
        Get current performance metrics.
        
        Args:
            function_name: Name of the function.
            implementation_type: Optional type of implementation.
            
        Returns:
            Current metrics for the specified function and implementation,
            or dictionary of all implementation metrics if implementation_type is None,
            or None if no metrics are available.
        """
        with self._lock:
            if function_name not in self.current_metrics:
                return None
                
            if implementation_type:
                return self.current_metrics[function_name].get(implementation_type)
            else:
                return self.current_metrics[function_name]
    
    def get_metrics_history(self, function_name: str, implementation_type: str = None, limit: int = None) -> Union[Dict[str, List[PerformanceMetrics]], List[PerformanceMetrics], None]:
        """
        Get historical performance metrics.
        
        Args:
            function_name: Name of the function.
            implementation_type: Optional type of implementation.
            limit: Optional limit on number of history entries to return.
            
        Returns:
            Historical metrics for the specified function and implementation,
            or dictionary of all implementation histories if implementation_type is None,
            or None if no metrics are available.
        """
        with self._lock:
            if function_name not in self.metrics_history:
                return None
                
            if implementation_type:
                history = list(self.metrics_history[function_name].get(implementation_type, []))
                if limit:
                    return history[-limit:]
                return history
            else:
                result = {}
                for impl_type, history in self.metrics_history[function_name].items():
                    result[impl_type] = list(history)
                    if limit:
                        result[impl_type] = result[impl_type][-limit:]
                return result
    
    def get_comparison_history(self, function_name: str, limit: int = None) -> List[PerformanceComparison]:
        """
        Get historical implementation comparisons.
        
        Args:
            function_name: Name of the function.
            limit: Optional limit on number of history entries to return.
            
        Returns:
            List of historical comparisons, or empty list if none available.
        """
        with self._lock:
            history = list(self.comparison_history.get(function_name, []))
            if limit:
                return history[-limit:]
            return history
    
    def get_trend_analysis(self, function_name: str, implementation_type: str, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """
        Get trend analysis for a specific metric.
        
        Args:
            function_name: Name of the function.
            implementation_type: Type of implementation.
            metric_name: Name of the metric.
            window_size: Number of recent entries to consider.
            
        Returns:
            Dictionary with trend information.
        """
        with self._lock:
            if (function_name not in self.trends or 
                implementation_type not in self.trends[function_name] or
                metric_name not in self.trends[function_name][implementation_type]):
                return {
                    "direction": "unknown",
                    "magnitude": 0.0,
                    "volatility": 0.0
                }
                
            trend_data = self.trends[function_name][implementation_type][metric_name]
            
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
            if abs(delta) < 0.05 * abs(start_value) if start_value != 0 else 0.01:
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
                "mean": mean,
                "data_points": len(recent_data)
            } 
   
    def set_baseline_metrics(self, function_name: str, implementation_type: str, metrics: Dict[str, float]) -> bool:
        """
        Set baseline metrics for anomaly detection.
        
        Args:
            function_name: Name of the function.
            implementation_type: Type of implementation.
            metrics: Dictionary of baseline metric values.
            
        Returns:
            True if baseline was set successfully, False otherwise.
        """
        try:
            with self._lock:
                self.baseline_metrics[function_name][implementation_type] = metrics.copy()
                self.logger.info(f"Set baseline metrics for {function_name} ({implementation_type})")
                return True
        except Exception as e:
            self.logger.error(f"Error setting baseline metrics for {function_name}: {e}")
            return False
    
    def set_anomaly_threshold(self, function_name: str, implementation_type: str, metric_name: str, threshold: float) -> bool:
        """
        Set anomaly detection threshold for a metric.
        
        Args:
            function_name: Name of the function.
            implementation_type: Type of implementation.
            metric_name: Name of the metric.
            threshold: Threshold value (as a fraction of baseline).
            
        Returns:
            True if threshold was set successfully, False otherwise.
        """
        try:
            with self._lock:
                self.anomaly_thresholds[function_name][implementation_type][metric_name] = threshold
                self.logger.info(f"Set anomaly threshold for {function_name} ({implementation_type}.{metric_name}) to {threshold}")
                return True
        except Exception as e:
            self.logger.error(f"Error setting anomaly threshold for {function_name}: {e}")
            return False
    
    def calculate_baseline_from_history(self, function_name: str, implementation_type: str, window_size: int = None) -> bool:
        """
        Calculate baseline metrics from historical data.
        
        Args:
            function_name: Name of the function.
            implementation_type: Type of implementation.
            window_size: Number of recent entries to consider, or None for all.
            
        Returns:
            True if baseline was calculated successfully, False otherwise.
        """
        try:
            with self._lock:
                if (function_name not in self.metrics_history or 
                    implementation_type not in self.metrics_history[function_name]):
                    self.logger.warning(f"No history available for {function_name} ({implementation_type})")
                    return False
                    
                history = list(self.metrics_history[function_name][implementation_type])
                if not history:
                    self.logger.warning(f"Empty history for {function_name} ({implementation_type})")
                    return False
                    
                # Use specified window or all history
                if window_size:
                    history = history[-min(window_size, len(history)):]
                
                # Calculate average metrics
                baseline = {}
                
                # Standard metrics
                for metric_name in ["accuracy", "latency", "resource_usage"]:
                    values = [getattr(metrics, metric_name) for metrics in history]
                    baseline[metric_name] = sum(values) / len(values)
                
                # Additional metrics (use first entry as reference for available metrics)
                if history:
                    for metric_name in history[0].additional_metrics:
                        values = [metrics.additional_metrics.get(metric_name, 0) for metrics in history]
                        if values:
                            baseline[metric_name] = sum(values) / len(values)
                
                # Set as baseline
                self.baseline_metrics[function_name][implementation_type] = baseline
                
                self.logger.info(f"Calculated baseline metrics for {function_name} ({implementation_type}) "
                               f"from {len(history)} historical entries")
                return True
                
        except Exception as e:
            self.logger.error(f"Error calculating baseline for {function_name}: {e}")
            return False    

    def get_recommended_implementation(self, function_name: str) -> Tuple[str, str]:
        """
        Get the recommended implementation for a function.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            Tuple of (implementation_type, reason).
        """
        with self._lock:
            # Get most recent comparison
            comparisons = self.get_comparison_history(function_name, limit=1)
            if comparisons:
                comparison = comparisons[0]
                return comparison.recommended_implementation, comparison.reason
            
            # Default to algorithmic if no comparison available
            return "algorithmic", "No comparison data available"
    
    def save_metrics(self, function_name: str) -> bool:
        """
        Save metrics to disk.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            True if metrics were saved successfully, False otherwise.
        """
        try:
            with self._lock:
                # Skip if no metrics for this function
                if function_name not in self.current_metrics:
                    return False
                    
                # Prepare data for serialization
                metrics_data = {
                    "function_name": function_name,
                    "saved_at": datetime.now().isoformat(),
                    "current_metrics": {},
                    "metrics_history": {},
                    "comparison_history": [],
                    "baseline_metrics": self.baseline_metrics.get(function_name, {}),
                    "anomaly_thresholds": self.anomaly_thresholds.get(function_name, {})
                }
                
                # Serialize current metrics
                for impl_type, metrics in self.current_metrics[function_name].items():
                    metrics_data["current_metrics"][impl_type] = metrics.to_dict()
                    
                # Serialize metrics history
                for impl_type, history in self.metrics_history[function_name].items():
                    metrics_data["metrics_history"][impl_type] = [metrics.to_dict() for metrics in history]
                    
                # Serialize comparison history
                metrics_data["comparison_history"] = [comparison.to_dict() for comparison in self.comparison_history.get(function_name, [])]
                
                # Save to file
                filepath = os.path.join(self.metrics_dir, f"{function_name}.metrics")
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                    
                self.logger.info(f"Saved metrics for {function_name} to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving metrics for {function_name}: {e}")
            return False
    
    def load_metrics(self, function_name: str) -> bool:
        """
        Load metrics from disk.
        
        Args:
            function_name: Name of the function.
            
        Returns:
            True if metrics were loaded successfully, False otherwise.
        """
        try:
            filepath = os.path.join(self.metrics_dir, f"{function_name}.metrics")
            
            if not os.path.exists(filepath):
                self.logger.warning(f"No metrics file found for {function_name}")
                return False
                
            with open(filepath, 'r') as f:
                metrics_data = json.load(f)
                
            with self._lock:
                # Load current metrics
                if "current_metrics" in metrics_data:
                    for impl_type, metrics in metrics_data["current_metrics"].items():
                        self.current_metrics[function_name][impl_type] = PerformanceMetrics.from_dict(metrics)
                        
                # Load metrics history
                if "metrics_history" in metrics_data:
                    for impl_type, history in metrics_data["metrics_history"].items():
                        self.metrics_history[function_name][impl_type] = deque(maxlen=self.max_history_length)
                        for entry in history:
                            self.metrics_history[function_name][impl_type].append(PerformanceMetrics.from_dict(entry))
                            
                # Load comparison history
                if "comparison_history" in metrics_data:
                    self.comparison_history[function_name] = deque(maxlen=self.max_history_length)
                    for entry in metrics_data["comparison_history"]:
                        self.comparison_history[function_name].append(PerformanceComparison.from_dict(entry))
                        
                # Load baseline metrics
                if "baseline_metrics" in metrics_data:
                    self.baseline_metrics[function_name] = metrics_data["baseline_metrics"]
                    
                # Load anomaly thresholds
                if "anomaly_thresholds" in metrics_data:
                    self.anomaly_thresholds[function_name] = metrics_data["anomaly_thresholds"]
                    
                # Rebuild trends from history
                for impl_type, history in self.metrics_history[function_name].items():
                    for metrics in history:
                        self._update_trends(function_name, impl_type, metrics)
                        
                self.logger.info(f"Loaded metrics for {function_name} from {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading metrics for {function_name}: {e}")
            return False    

    def get_performance_summary(self, function_name: str = None) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Args:
            function_name: Optional name of the function, or None for all functions.
            
        Returns:
            Dictionary with performance summary.
        """
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "functions": {}
            }
            
            # Get functions to summarize
            functions = [function_name] if function_name else list(self.current_metrics.keys())
            
            for func in functions:
                if func not in self.current_metrics:
                    continue
                    
                function_summary = {
                    "implementations": {},
                    "recommended_implementation": None,
                    "recommendation_reason": None
                }
                
                # Add current metrics for each implementation
                for impl_type, metrics in self.current_metrics[func].items():
                    function_summary["implementations"][impl_type] = {
                        "accuracy": metrics.accuracy,
                        "latency": metrics.latency,
                        "resource_usage": metrics.resource_usage,
                        "last_updated": metrics.timestamp.isoformat()
                    }
                    
                    # Add additional metrics
                    for metric_name, value in metrics.additional_metrics.items():
                        function_summary["implementations"][impl_type][metric_name] = value
                
                # Add recommendation if available
                if func in self.comparison_history and self.comparison_history[func]:
                    latest_comparison = self.comparison_history[func][-1]
                    function_summary["recommended_implementation"] = latest_comparison.recommended_implementation
                    function_summary["recommendation_reason"] = latest_comparison.reason
                    function_summary["neural_score"] = latest_comparison.neural_score
                    function_summary["algorithmic_score"] = latest_comparison.algorithmic_score
                
                summary["functions"][func] = function_summary
            
            return summary