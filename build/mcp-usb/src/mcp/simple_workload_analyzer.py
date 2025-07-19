"""
SimpleWorkloadAnalyzer: Basic pattern recognition for resource optimization.

This module implements basic time-series analysis to detect patterns in
system workload, enabling predictive resource allocation and optimization.
This is a simplified version that doesn't require external dependencies.
"""

import math
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging


@dataclass
class WorkloadPattern:
    """Workload pattern information."""
    pattern_type: str  # Type of pattern (e.g., "cyclic", "bursty", "steady")
    periodicity: Optional[float] = None  # Time period in seconds for cyclic patterns
    peak_times: List[str] = None  # Peak times for recurring patterns
    active_lobes: Dict[str, float] = None  # Lobe activity levels
    resource_intensity: Dict[str, float] = None  # Resource intensity by type
    confidence: float = 0.0  # Confidence in pattern detection (0-1)
    
    def __post_init__(self):
        if self.peak_times is None:
            self.peak_times = []
        if self.active_lobes is None:
            self.active_lobes = {}
        if self.resource_intensity is None:
            self.resource_intensity = {}


@dataclass
class ResourcePrediction:
    """Prediction of future resource needs."""
    predicted_cpu: Dict[str, float] = None  # Predicted CPU by time
    predicted_memory: Dict[str, float] = None  # Predicted memory by time
    predicted_disk: Dict[str, float] = None  # Predicted disk by time
    confidence: float = 0.0  # Confidence in prediction (0-1)
    recommended_actions: List[str] = None  # Recommended actions
    prediction_horizon: int = 60  # Prediction horizon in seconds
    
    def __post_init__(self):
        if self.predicted_cpu is None:
            self.predicted_cpu = {}
        if self.predicted_memory is None:
            self.predicted_memory = {}
        if self.predicted_disk is None:
            self.predicted_disk = {}
        if self.recommended_actions is None:
            self.recommended_actions = []


class SimpleWorkloadAnalyzer:
    """
    Basic pattern recognition for resource optimization.
    
    This class analyzes time-series data of resource usage to detect patterns,
    predict future resource needs, and recommend optimization strategies.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the workload pattern analyzer.
        
        Args:
            history_size: Maximum number of data points to keep in history
        """
        self.logger = logging.getLogger("SimpleWorkloadAnalyzer")
        self.history_size = history_size
        
        # Time series data
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # Pattern detection state
        self.detected_patterns = {}
        self.pattern_confidence = {}
        self.last_analysis_time = datetime.now() - timedelta(hours=1)
        self.analysis_interval = 60  # Seconds between full pattern analysis
        
        # Prediction state
        self.current_prediction = None
        self.prediction_accuracy_history = deque(maxlen=100)
        
        self.logger.info("SimpleWorkloadAnalyzer initialized")
    
    def add_data_point(self, cpu_usage: float, memory_usage: float, 
                      disk_usage: float, timestamp: Optional[datetime] = None):
        """
        Add a new data point to the time series.
        
        Args:
            cpu_usage: CPU usage percentage (0-100)
            memory_usage: Memory usage percentage (0-100)
            disk_usage: Disk usage percentage (0-100)
            timestamp: Timestamp of the data point (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.cpu_history.append(cpu_usage)
        self.memory_history.append(memory_usage)
        self.disk_history.append(disk_usage)
        self.timestamp_history.append(timestamp)
        
        # Check if it's time for a full pattern analysis
        if (datetime.now() - self.last_analysis_time).total_seconds() > self.analysis_interval:
            self._perform_full_analysis()
            self.last_analysis_time = datetime.now()
    
    def _perform_full_analysis(self):
        """Perform a full analysis of the time series data."""
        if len(self.cpu_history) < 20:
            self.logger.info("Not enough data for pattern analysis")
            return
        
        # Analyze each resource type
        cpu_pattern = self._analyze_pattern(list(self.cpu_history), "cpu")
        memory_pattern = self._analyze_pattern(list(self.memory_history), "memory")
        disk_pattern = self._analyze_pattern(list(self.disk_history), "disk")
        
        # Combine patterns
        combined_pattern = self._combine_patterns([cpu_pattern, memory_pattern, disk_pattern])
        
        # Update detected patterns
        self.detected_patterns["main"] = combined_pattern
        
        # Generate prediction based on pattern
        self.current_prediction = self._generate_prediction(combined_pattern)
        
        self.logger.info(f"Pattern analysis complete: {combined_pattern.pattern_type} "
                       f"with {combined_pattern.confidence:.2f} confidence")
    
    def _analyze_pattern(self, data: List[float], resource_type: str) -> WorkloadPattern:
        """
        Analyze a single resource time series for patterns.
        
        Args:
            data: List of resource usage values
            resource_type: Type of resource ("cpu", "memory", "disk")
            
        Returns:
            WorkloadPattern object with detected pattern information
        """
        if len(data) < 20:
            return WorkloadPattern(pattern_type="unknown", confidence=0.0)
        
        # Simple pattern detection
        pattern_type = "steady"
        periodicity = None
        confidence = 0.5  # Default confidence
        
        # Calculate basic statistics
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)
        
        # Check for cyclic patterns (simplified)
        if len(data) >= 40:
            # Look for repeating patterns
            for period in range(10, min(30, len(data) // 2)):
                correlation = self._simple_autocorrelation(data, period)
                if correlation > 0.7:  # Strong correlation
                    pattern_type = "cyclic"
                    periodicity = period
                    confidence = min(0.9, 0.5 + correlation * 0.5)
                    break
        
        # Check for burstiness
        if pattern_type != "cyclic" and std_dev > mean * 0.5:
            # High variance relative to mean suggests burstiness
            pattern_type = "bursty"
            confidence = 0.7
        
        # Calculate resource intensity
        resource_intensity = {
            "mean": mean,
            "std_dev": std_dev,
            "max": max(data),
            "min": min(data)
        }
        
        # Detect peak times
        peak_times = self._detect_peak_times(data)
        
        return WorkloadPattern(
            pattern_type=pattern_type,
            periodicity=periodicity,
            peak_times=peak_times,
            resource_intensity={resource_type: resource_intensity},
            confidence=confidence
        )
    
    def _simple_autocorrelation(self, data: List[float], lag: int) -> float:
        """
        Calculate simple autocorrelation for a specific lag.
        
        Args:
            data: Time series data
            lag: Lag value
            
        Returns:
            Autocorrelation value (-1 to 1)
        """
        if len(data) <= lag:
            return 0.0
        
        # Calculate mean
        mean = sum(data) / len(data)
        
        # Calculate autocorrelation
        numerator = 0.0
        denominator1 = 0.0
        denominator2 = 0.0
        
        for i in range(len(data) - lag):
            x1 = data[i] - mean
            x2 = data[i + lag] - mean
            numerator += x1 * x2
            denominator1 += x1 * x1
            denominator2 += x2 * x2
        
        if denominator1 == 0.0 or denominator2 == 0.0:
            return 0.0
        
        return numerator / math.sqrt(denominator1 * denominator2)
    
    def _detect_peak_times(self, data: List[float]) -> List[str]:
        """
        Detect peak times in a time series.
        
        Args:
            data: Time series data
            
        Returns:
            List of peak times as strings (HH:MM format)
        """
        if len(data) < 10 or len(self.timestamp_history) < 10:
            return []
        
        # Calculate mean and standard deviation
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)
        
        # Find peaks (values above mean + std_dev)
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > mean + std_dev and 
                data[i] > data[i-1] and 
                data[i] > data[i+1]):
                peaks.append(i)
        
        # Convert peak indices to timestamps
        peak_times = []
        timestamps = list(self.timestamp_history)
        
        for peak in peaks:
            if peak < len(timestamps):
                peak_time = timestamps[peak]
                peak_times.append(peak_time.strftime("%H:%M"))
        
        return peak_times
    
    def _combine_patterns(self, patterns: List[WorkloadPattern]) -> WorkloadPattern:
        """
        Combine multiple resource patterns into a single pattern.
        
        Args:
            patterns: List of WorkloadPattern objects
            
        Returns:
            Combined WorkloadPattern
        """
        # Count pattern types
        pattern_counts = {}
        for pattern in patterns:
            if pattern.pattern_type not in pattern_counts:
                pattern_counts[pattern.pattern_type] = 0
            pattern_counts[pattern.pattern_type] += 1
        
        # Select most common pattern type
        if not pattern_counts:
            return WorkloadPattern(pattern_type="steady", confidence=0.0)
            
        most_common_type = max(pattern_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average periodicity for cyclic patterns
        periodicity = None
        if most_common_type == "cyclic":
            periodicities = [p.periodicity for p in patterns 
                           if p.pattern_type == "cyclic" and p.periodicity is not None]
            if periodicities:
                periodicity = sum(periodicities) / len(periodicities)
        
        # Combine resource intensities
        resource_intensity = {}
        for pattern in patterns:
            if pattern.resource_intensity:
                resource_intensity.update(pattern.resource_intensity)
        
        # Combine peak times
        all_peak_times = []
        for pattern in patterns:
            if pattern.peak_times:
                all_peak_times.extend(pattern.peak_times)
        
        # Calculate average confidence
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns) if patterns else 0.0
        
        return WorkloadPattern(
            pattern_type=most_common_type,
            periodicity=periodicity,
            peak_times=list(set(all_peak_times)),  # Remove duplicates
            resource_intensity=resource_intensity,
            confidence=avg_confidence
        )
    
    def _generate_prediction(self, pattern: WorkloadPattern) -> ResourcePrediction:
        """
        Generate resource prediction based on detected pattern.
        
        Args:
            pattern: Detected workload pattern
            
        Returns:
            ResourcePrediction object with future resource needs
        """
        # Create prediction with 60-second horizon
        prediction = ResourcePrediction(
            confidence=pattern.confidence,
            prediction_horizon=60
        )
        
        # Generate time points for prediction (one per second)
        current_time = datetime.now()
        time_points = [(current_time + timedelta(seconds=i)).strftime("%H:%M:%S") 
                      for i in range(prediction.prediction_horizon)]
        
        # Get current values as baseline
        current_cpu = self.cpu_history[-1] if self.cpu_history else 0.0
        current_memory = self.memory_history[-1] if self.memory_history else 0.0
        current_disk = self.disk_history[-1] if self.disk_history else 0.0
        
        # Generate predictions based on pattern type
        if pattern.pattern_type == "steady":
            # Steady pattern: predict constant values
            for time_point in time_points:
                prediction.predicted_cpu[time_point] = current_cpu
                prediction.predicted_memory[time_point] = current_memory
                prediction.predicted_disk[time_point] = current_disk
                
            prediction.recommended_actions = ["maintain_current_resources"]
            
        elif pattern.pattern_type == "cyclic" and pattern.periodicity:
            # Cyclic pattern: predict based on periodicity
            for i, time_point in enumerate(time_points):
                # Calculate phase in the cycle
                phase = i % int(pattern.periodicity)
                phase_ratio = phase / pattern.periodicity if pattern.periodicity > 0 else 0
                
                # Use historical data at similar phase if available
                if len(self.cpu_history) > pattern.periodicity:
                    historical_index = int(len(self.cpu_history) - pattern.periodicity + phase)
                    if 0 <= historical_index < len(self.cpu_history):
                        prediction.predicted_cpu[time_point] = self.cpu_history[historical_index]
                        prediction.predicted_memory[time_point] = self.memory_history[historical_index]
                        prediction.predicted_disk[time_point] = self.disk_history[historical_index]
                    else:
                        # Fallback to sinusoidal approximation
                        cpu_min = min(self.cpu_history)
                        cpu_max = max(self.cpu_history)
                        cpu_amplitude = (cpu_max - cpu_min) / 2
                        cpu_mean = (cpu_max + cpu_min) / 2
                        prediction.predicted_cpu[time_point] = cpu_mean + cpu_amplitude * math.sin(phase_ratio * 2 * math.pi)
                        
                        mem_min = min(self.memory_history)
                        mem_max = max(self.memory_history)
                        mem_amplitude = (mem_max - mem_min) / 2
                        mem_mean = (mem_max + mem_min) / 2
                        prediction.predicted_memory[time_point] = mem_mean + mem_amplitude * math.sin(phase_ratio * 2 * math.pi)
                        
                        prediction.predicted_disk[time_point] = current_disk  # Disk usually changes more slowly
                else:
                    # Not enough history, use current values
                    prediction.predicted_cpu[time_point] = current_cpu
                    prediction.predicted_memory[time_point] = current_memory
                    prediction.predicted_disk[time_point] = current_disk
            
            prediction.recommended_actions = [
                "allocate_resources_cyclically",
                f"prepare_for_peak_in_{pattern.periodicity:.0f}_seconds"
            ]
            
        elif pattern.pattern_type == "bursty":
            # Bursty pattern: predict potential spikes
            for time_point in time_points:
                # Base prediction on current values
                prediction.predicted_cpu[time_point] = current_cpu
                prediction.predicted_memory[time_point] = current_memory
                prediction.predicted_disk[time_point] = current_disk
            
            # Add spike prediction for a random point in the future
            spike_time = time_points[len(time_points) // 2]  # Middle of prediction horizon
            prediction.predicted_cpu[spike_time] = min(100.0, current_cpu * 2.0)
            prediction.predicted_memory[spike_time] = min(100.0, current_memory * 1.5)
            
            prediction.recommended_actions = [
                "maintain_resource_buffer",
                "prepare_for_unpredictable_spikes"
            ]
        
        return prediction
    
    def get_current_pattern(self) -> Optional[WorkloadPattern]:
        """
        Get the currently detected workload pattern.
        
        Returns:
            WorkloadPattern object or None if no pattern detected
        """
        return self.detected_patterns.get("main")
    
    def get_resource_prediction(self) -> Optional[ResourcePrediction]:
        """
        Get the current resource prediction.
        
        Returns:
            ResourcePrediction object or None if no prediction available
        """
        return self.current_prediction
    
    def evaluate_prediction_accuracy(self, actual_cpu: float, actual_memory: float, 
                                    actual_disk: float, timestamp: datetime):
        """
        Evaluate the accuracy of previous predictions.
        
        Args:
            actual_cpu: Actual CPU usage
            actual_memory: Actual memory usage
            actual_disk: Actual disk usage
            timestamp: Timestamp of the measurements
        """
        if not self.current_prediction:
            return
        
        # Find the closest time point in the prediction
        time_str = timestamp.strftime("%H:%M:%S")
        if time_str in self.current_prediction.predicted_cpu:
            # Calculate prediction error
            cpu_error = abs(self.current_prediction.predicted_cpu[time_str] - actual_cpu)
            memory_error = abs(self.current_prediction.predicted_memory[time_str] - actual_memory)
            disk_error = abs(self.current_prediction.predicted_disk[time_str] - actual_disk)
            
            # Calculate accuracy (inverse of normalized error)
            cpu_accuracy = max(0.0, 1.0 - cpu_error / 100.0)
            memory_accuracy = max(0.0, 1.0 - memory_error / 100.0)
            disk_accuracy = max(0.0, 1.0 - disk_error / 100.0)
            
            # Average accuracy
            avg_accuracy = (cpu_accuracy + memory_accuracy + disk_accuracy) / 3.0
            
            # Store accuracy
            self.prediction_accuracy_history.append(avg_accuracy)
            
            # Update prediction confidence based on historical accuracy
            if self.prediction_accuracy_history:
                avg_historical_accuracy = sum(self.prediction_accuracy_history) / len(self.prediction_accuracy_history)
                if self.current_prediction:
                    self.current_prediction.confidence = avg_historical_accuracy