"""
MonitoringSystem Visualization Integration: Integrates visualization capabilities with MonitoringSystem.

This module extends the MonitoringSystem with the visualization capabilities from
MonitoringSystemVisualization, providing a unified interface for monitoring and visualization.

References:
- Requirements 4.4, 4.9 from mcp-system-upgrade spec
- idea.txt (brain-inspired architecture)
- cross-implementation.md (hormone system integration)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

from src.mcp.monitoring_system import MonitoringSystem
from src.mcp.monitoring_system_visualization import (
    MonitoringSystemVisualization,
    VisualizationOptions,
    PerformanceComparisonReport
)


class MonitoringSystemWithVisualization(MonitoringSystem):
    """
    Extended MonitoringSystem with integrated visualization capabilities.
    
    This class combines the real-time monitoring capabilities of MonitoringSystem
    with the advanced visualization and reporting features of MonitoringSystemVisualization.
    """
    
    def __init__(self, 
                 database_path: str = "data/monitoring_system.db",
                 update_interval: float = 1.0,
                 max_history_entries: int = 10000):
        """
        Initialize the monitoring system with visualization capabilities.
        
        Args:
            database_path: Path to SQLite database for persistent storage
            update_interval: How often to update monitoring data (seconds)
            max_history_entries: Maximum number of historical entries to keep
        """
        super().__init__(database_path, update_interval, max_history_entries)
        self.visualization = MonitoringSystemVisualization(database_path)
        self.logger = logging.getLogger("MonitoringSystemWithVisualization")
        
        self.logger.info("MonitoringSystemWithVisualization initialized")
    
    def generate_multi_hormone_visualization(self, 
                                           hormone_names: List[str],
                                           start_time: Optional[str] = None,
                                           end_time: Optional[str] = None,
                                           resolution: str = "minute",
                                           options: Optional[VisualizationOptions] = None) -> Dict[str, Any]:
        """
        Generate visualization data for multiple hormones for comparison.
        
        Args:
            hormone_names: List of hormone names to visualize
            start_time: Start time (ISO format), defaults to 1 hour ago
            end_time: End time (ISO format), defaults to now
            resolution: Time resolution ("second", "minute", "hour", "day")
            options: Visualization options for customization
            
        Returns:
            Dictionary with visualization data for multiple hormones
        """
        return self.visualization.generate_multi_hormone_visualization(
            hormone_names=hormone_names,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
            options=options
        )
    
    def generate_cascade_visualization(self,
                                     cascade_name: Optional[str] = None,
                                     start_time: Optional[str] = None,
                                     end_time: Optional[str] = None,
                                     limit: int = 10) -> Dict[str, Any]:
        """
        Generate visualization data for hormone cascades.
        
        Args:
            cascade_name: Optional cascade name to filter by
            start_time: Start time (ISO format), defaults to 24 hours ago
            end_time: End time (ISO format), defaults to now
            limit: Maximum number of cascades to include
            
        Returns:
            Dictionary with cascade visualization data
        """
        return self.visualization.generate_cascade_visualization(
            cascade_name=cascade_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def generate_performance_comparison_report(self,
                                             component: str,
                                             start_time: Optional[str] = None,
                                             end_time: Optional[str] = None) -> PerformanceComparisonReport:
        """
        Generate a comprehensive performance comparison report between implementations.
        
        Args:
            component: Component name to generate report for
            start_time: Start time (ISO format), defaults to 7 days ago
            end_time: End time (ISO format), defaults to now
            
        Returns:
            PerformanceComparisonReport object with detailed comparison
        """
        return self.visualization.generate_performance_comparison_report(
            component=component,
            start_time=start_time,
            end_time=end_time
        )
    
    def generate_system_performance_report(self,
                                         start_time: Optional[str] = None,
                                         end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive system-wide performance report.
        
        Args:
            start_time: Start time (ISO format), defaults to 7 days ago
            end_time: End time (ISO format), defaults to now
            
        Returns:
            Dictionary with system-wide performance report
        """
        return self.visualization.generate_system_performance_report(
            start_time=start_time,
            end_time=end_time
        )
    
    def generate_hormone_trend_report(self,
                                    hormone_names: Optional[List[str]] = None,
                                    start_time: Optional[str] = None,
                                    end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a trend analysis report for hormone levels.
        
        Args:
            hormone_names: Optional list of hormone names to include, defaults to all
            start_time: Start time (ISO format), defaults to 7 days ago
            end_time: End time (ISO format), defaults to now
            
        Returns:
            Dictionary with hormone trend analysis
        """
        return self.visualization.generate_hormone_trend_report(
            hormone_names=hormone_names,
            start_time=start_time,
            end_time=end_time
        )
    
    def record_performance_metrics(self,
                                 component: str,
                                 implementation: str,
                                 metrics: Dict[str, float],
                                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record performance metrics for a component implementation.
        
        Args:
            component: Component name (e.g., "hormone_calculator")
            implementation: Implementation name (e.g., "algorithmic", "neural")
            metrics: Dictionary of metric names to values
            context: Optional context information
        """
        timestamp = datetime.now().isoformat()
        
        with self._lock:
            # Store in database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                for metric_name, metric_value in metrics.items():
                    cursor.execute("""
                        INSERT INTO performance_metrics 
                        (component, implementation, timestamp, metric_name, metric_value, context)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        component,
                        implementation,
                        timestamp,
                        metric_name,
                        metric_value,
                        self._serialize_json(context) if context else None
                    ))
                
                conn.commit()
        
        self.logger.debug(f"Recorded performance metrics for {component} ({implementation})")
    
    def record_implementation_switch(self,
                                   component: str,
                                   old_implementation: str,
                                   new_implementation: str,
                                   reason: str,
                                   performance_comparison: Dict[str, Dict[str, float]],
                                   switch_trigger: str = "performance",
                                   confidence_score: float = 0.0,
                                   expected_improvement: Optional[Dict[str, float]] = None) -> None:
        """
        Record an implementation switch event.
        
        Args:
            component: Component name (e.g., "hormone_calculator")
            old_implementation: Previous implementation name
            new_implementation: New implementation name
            reason: Reason for the switch
            performance_comparison: Performance comparison data
            switch_trigger: Trigger for the switch ("performance", "failure", "manual", "scheduled")
            confidence_score: Confidence in the switch decision (0.0 to 1.0)
            expected_improvement: Expected improvement in metrics
        """
        timestamp = datetime.now().isoformat()
        
        with self._lock:
            # Store in database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO implementation_switches 
                    (component, old_implementation, new_implementation, switch_timestamp,
                     reason, performance_comparison, switch_trigger, confidence_score, expected_improvement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    component,
                    old_implementation,
                    new_implementation,
                    timestamp,
                    reason,
                    self._serialize_json(performance_comparison),
                    switch_trigger,
                    confidence_score,
                    self._serialize_json(expected_improvement) if expected_improvement else None
                ))
                
                conn.commit()
        
        self.logger.info(f"Recorded implementation switch for {component}: {old_implementation} -> {new_implementation}")


# Example usage
if __name__ == "__main__":
    monitoring = MonitoringSystemWithVisualization()
    
    try:
        # Start monitoring
        monitoring.start_monitoring()
        
        # Record performance metrics
        monitoring.record_performance_metrics(
            component="hormone_calculator",
            implementation="algorithmic",
            metrics={
                "accuracy": 0.85,
                "latency": 0.15,
                "resource_usage": 0.3
            }
        )
        
        monitoring.record_performance_metrics(
            component="hormone_calculator",
            implementation="neural",
            metrics={
                "accuracy": 0.95,
                "latency": 0.1,
                "resource_usage": 0.5
            }
        )
        
        # Record implementation switch
        monitoring.record_implementation_switch(
            component="hormone_calculator",
            old_implementation="algorithmic",
            new_implementation="neural",
            reason="Neural implementation showed 15% better accuracy",
            performance_comparison={
                "algorithmic": {
                    "accuracy": 0.85,
                    "latency": 0.15,
                    "resource_usage": 0.3
                },
                "neural": {
                    "accuracy": 0.95,
                    "latency": 0.1,
                    "resource_usage": 0.5
                }
            },
            switch_trigger="performance",
            confidence_score=0.8,
            expected_improvement={
                "accuracy": 0.1,
                "latency": 0.05
            }
        )
        
        # Generate performance report
        report = monitoring.generate_performance_comparison_report("hormone_calculator")
        if report:
            print(f"Generated performance report for {report.component}")
            print(f"Active implementation: {report.winner}")
            print(f"Recommendations: {report.recommendations}")
        
    finally:
        # Stop monitoring
        monitoring.stop_monitoring()
"""