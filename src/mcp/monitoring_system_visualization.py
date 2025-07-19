"""
MonitoringSystem Visualization: Enhanced visualization and reporting capabilities.

This module extends the MonitoringSystem with advanced visualization data generation,
performance comparison reporting, and automated trend analysis capabilities.

References:
- Requirements 4.4, 4.9 from mcp-system-upgrade spec
- idea.txt (brain-inspired architecture)
- cross-implementation.md (hormone system integration)
"""

import logging
import sqlite3
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from pathlib import Path

from dataclasses import dataclass, field


@dataclass
class PerformanceComparisonReport:
    """Comprehensive performance comparison between implementations."""
    component: str
    report_timestamp: str
    time_range: Tuple[str, str]
    implementations: Dict[str, Dict[str, Any]]
    comparison_metrics: Dict[str, Dict[str, float]]
    winner: str
    improvement_percentage: Dict[str, float]
    confidence_score: float
    recommendations: List[str]
    trend_analysis: Dict[str, Any]


@dataclass
class TrendAnalysis:
    """Analysis of performance trends over time."""
    metric_name: str
    time_points: List[str]
    values: List[float]
    trend_direction: str  # "increasing", "decreasing", "stable", "fluctuating"
    trend_strength: float  # 0.0 to 1.0
    anomalies: List[Dict[str, Any]]
    forecast: List[float]
    correlation_factors: Dict[str, float]


@dataclass
class VisualizationOptions:
    """Options for customizing visualizations."""
    chart_type: str = "line"  # "line", "bar", "scatter", "area", "heatmap"
    color_scheme: str = "default"
    include_anomalies: bool = True
    smoothing_factor: float = 0.0  # 0.0 to 1.0
    include_forecast: bool = False
    forecast_periods: int = 5
    confidence_interval: bool = False
    stack_series: bool = False
    normalize_values: bool = False
    logarithmic_scale: bool = False
    annotations: List[Dict[str, Any]] = field(default_factory=list)

class MonitoringSystemVisualization:
    """
    Enhanced visualization and reporting capabilities for the MonitoringSystem.
    
    This class extends the MonitoringSystem with advanced visualization data generation,
    performance comparison reporting, and automated trend analysis capabilities.
    """
    
    def __init__(self, database_path: str = "data/monitoring_system.db"):
        """
        Initialize the visualization system.
        
        Args:
            database_path: Path to SQLite database for persistent storage
        """
        self.database_path = database_path
        self.logger = logging.getLogger("MonitoringSystemVisualization")
        
        # Ensure database path exists
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
    
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
        if not options:
            options = VisualizationOptions()
            
        if not end_time:
            end_time = datetime.now().isoformat()
        if not start_time:
            start_time = (datetime.now() - timedelta(hours=1)).isoformat()
        
        # Query database for hormone data
        hormone_data = {}
        time_points = set()
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            for hormone_name in hormone_names:
                cursor.execute("""
                    SELECT timestamp, level, source_lobe
                    FROM hormone_levels
                    WHERE hormone_name = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (hormone_name, start_time, end_time))
                
                rows = cursor.fetchall()
                
                # Process data points
                data_points = []
                for timestamp, level, source_lobe in rows:
                    data_points.append({
                        "timestamp": timestamp,
                        "value": level,
                        "source_lobe": source_lobe
                    })
                    time_points.add(timestamp)
                
                hormone_data[hormone_name] = data_points
        
        # Create unified time series with all time points
        sorted_time_points = sorted(time_points)
        
        # Apply smoothing if requested
        if options.smoothing_factor > 0:
            for hormone_name in hormone_names:
                if hormone_name in hormone_data and hormone_data[hormone_name]:
                    values = [point["value"] for point in hormone_data[hormone_name]]
                    smoothed_values = self._apply_smoothing(values, options.smoothing_factor)
                    
                    for i, point in enumerate(hormone_data[hormone_name]):
                        if i < len(smoothed_values):
                            point["value"] = smoothed_values[i]
        
        # Generate forecast if requested
        forecasts = {}
        if options.include_forecast and len(sorted_time_points) > 10:
            for hormone_name in hormone_names:
                if hormone_name in hormone_data and len(hormone_data[hormone_name]) > 10:
                    values = [point["value"] for point in hormone_data[hormone_name]]
                    forecast = self._generate_forecast(values, options.forecast_periods)
                    forecasts[hormone_name] = forecast
        
        # Detect anomalies if requested
        anomalies = {}
        if options.include_anomalies:
            for hormone_name in hormone_names:
                if hormone_name in hormone_data and len(hormone_data[hormone_name]) > 10:
                    values = [point["value"] for point in hormone_data[hormone_name]]
                    detected_anomalies = self._detect_value_anomalies(values)
                    
                    if detected_anomalies:
                        anomalies[hormone_name] = [
                            {
                                "index": idx,
                                "timestamp": hormone_data[hormone_name][idx]["timestamp"],
                                "value": hormone_data[hormone_name][idx]["value"],
                                "deviation": deviation
                            }
                            for idx, deviation in detected_anomalies
                        ]
        
        return {
            "chart_type": options.chart_type,
            "time_points": sorted_time_points,
            "hormone_data": hormone_data,
            "forecasts": forecasts if options.include_forecast else {},
            "anomalies": anomalies if options.include_anomalies else {},
            "metadata": {
                "start_time": start_time,
                "end_time": end_time,
                "resolution": resolution,
                "hormone_count": len(hormone_names),
                "data_point_count": sum(len(data) for data in hormone_data.values())
            },
            "options": {
                "color_scheme": options.color_scheme,
                "smoothing_factor": options.smoothing_factor,
                "logarithmic_scale": options.logarithmic_scale,
                "stack_series": options.stack_series,
                "normalize_values": options.normalize_values
            }
        } 
   
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
        if not end_time:
            end_time = datetime.now().isoformat()
        if not start_time:
            start_time = (datetime.now() - timedelta(hours=24)).isoformat()
        
        # Query database for cascade data
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT cascade_name, timestamp, trigger_hormone, trigger_level,
                       affected_hormones, affected_lobes, duration, effects, feedback_loops
                FROM hormone_cascades
                WHERE timestamp BETWEEN ? AND ?
            """
            params = [start_time, end_time]
            
            if cascade_name:
                query += " AND cascade_name = ?"
                params.append(cascade_name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        # Process cascade data
        cascades = []
        for row in rows:
            cascade = {
                "cascade_name": row[0],
                "timestamp": row[1],
                "trigger_hormone": row[2],
                "trigger_level": row[3],
                "affected_hormones": json.loads(row[4]) if row[4] else [],
                "affected_lobes": json.loads(row[5]) if row[5] else [],
                "duration": row[6],
                "effects": json.loads(row[7]) if row[7] else {},
                "feedback_loops": json.loads(row[8]) if row[8] else []
            }
            cascades.append(cascade)
        
        # Generate nodes and links for network visualization
        nodes = []
        links = []
        node_ids = set()
        
        for cascade in cascades:
            # Add trigger hormone node
            trigger_id = f"hormone_{cascade['trigger_hormone']}"
            if trigger_id not in node_ids:
                nodes.append({
                    "id": trigger_id,
                    "type": "hormone",
                    "name": cascade['trigger_hormone'],
                    "level": cascade['trigger_level']
                })
                node_ids.add(trigger_id)
            
            # Add affected hormone nodes and links
            for hormone in cascade['affected_hormones']:
                hormone_id = f"hormone_{hormone}"
                if hormone_id not in node_ids:
                    nodes.append({
                        "id": hormone_id,
                        "type": "hormone",
                        "name": hormone
                    })
                    node_ids.add(hormone_id)
                
                links.append({
                    "source": trigger_id,
                    "target": hormone_id,
                    "cascade": cascade['cascade_name'],
                    "timestamp": cascade['timestamp']
                })
            
            # Add affected lobe nodes and links
            for lobe in cascade['affected_lobes']:
                lobe_id = f"lobe_{lobe}"
                if lobe_id not in node_ids:
                    nodes.append({
                        "id": lobe_id,
                        "type": "lobe",
                        "name": lobe
                    })
                    node_ids.add(lobe_id)
                
                # Link from trigger hormone to lobe
                links.append({
                    "source": trigger_id,
                    "target": lobe_id,
                    "cascade": cascade['cascade_name'],
                    "timestamp": cascade['timestamp']
                })
                
                # Link from affected hormones to lobe
                for hormone in cascade['affected_hormones']:
                    hormone_id = f"hormone_{hormone}"
                    links.append({
                        "source": hormone_id,
                        "target": lobe_id,
                        "cascade": cascade['cascade_name'],
                        "timestamp": cascade['timestamp']
                    })
        
        return {
            "cascades": cascades,
            "network": {
                "nodes": nodes,
                "links": links
            },
            "metadata": {
                "start_time": start_time,
                "end_time": end_time,
                "cascade_count": len(cascades),
                "unique_hormones": len([n for n in nodes if n["type"] == "hormone"]),
                "unique_lobes": len([n for n in nodes if n["type"] == "lobe"])
            }
        }    

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
        if not end_time:
            end_time = datetime.now().isoformat()
        if not start_time:
            start_time = (datetime.now() - timedelta(days=7)).isoformat()
        
        # Query database for performance metrics
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Get all implementations for this component
            cursor.execute("""
                SELECT DISTINCT implementation
                FROM performance_metrics
                WHERE component = ? AND timestamp BETWEEN ? AND ?
            """, (component, start_time, end_time))
            
            implementations = [row[0] for row in cursor.fetchall()]
            
            if not implementations:
                self.logger.warning(f"No performance data found for component {component}")
                return None
            
            # Get metrics for each implementation
            implementation_metrics = {}
            for impl in implementations:
                cursor.execute("""
                    SELECT metric_name, AVG(metric_value) as avg_value, 
                           MIN(metric_value) as min_value,
                           MAX(metric_value) as max_value,
                           COUNT(metric_value) as count
                    FROM performance_metrics
                    WHERE component = ? AND implementation = ? AND timestamp BETWEEN ? AND ?
                    GROUP BY metric_name
                """, (component, impl, start_time, end_time))
                
                metrics = {}
                for row in cursor.fetchall():
                    metrics[row[0]] = {
                        "avg": row[1],
                        "min": row[2],
                        "max": row[3],
                        "count": row[4]
                    }
                
                # Get time series data for trend analysis
                cursor.execute("""
                    SELECT metric_name, timestamp, metric_value
                    FROM performance_metrics
                    WHERE component = ? AND implementation = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (component, impl, start_time, end_time))
                
                time_series = {}
                for row in cursor.fetchall():
                    metric_name = row[0]
                    if metric_name not in time_series:
                        time_series[metric_name] = []
                    
                    time_series[metric_name].append({
                        "timestamp": row[1],
                        "value": row[2]
                    })
                
                implementation_metrics[impl] = {
                    "metrics": metrics,
                    "time_series": time_series
                }
            
            # Get implementation switch history
            cursor.execute("""
                SELECT old_implementation, new_implementation, switch_timestamp, 
                       reason, performance_comparison, confidence_score
                FROM implementation_switches
                WHERE component = ? AND switch_timestamp BETWEEN ? AND ?
                ORDER BY switch_timestamp
            """, (component, start_time, end_time))
            
            switch_history = []
            for row in cursor.fetchall():
                switch_history.append({
                    "old_implementation": row[0],
                    "new_implementation": row[1],
                    "timestamp": row[2],
                    "reason": row[3],
                    "performance_comparison": json.loads(row[4]) if row[4] else {},
                    "confidence_score": row[5]
                })
        
        # Compare implementations
        comparison_metrics = {}
        for metric_name in set().union(*[impl_data["metrics"].keys() for impl_data in implementation_metrics.values()]):
            comparison_metrics[metric_name] = {}
            
            # Get values for each implementation
            impl_values = {}
            for impl, impl_data in implementation_metrics.items():
                if metric_name in impl_data["metrics"]:
                    impl_values[impl] = impl_data["metrics"][metric_name]["avg"]
            
            # Determine best implementation for this metric
            if metric_name in ["accuracy", "confidence"]:
                # Higher is better
                best_impl = max(impl_values.items(), key=lambda x: x[1])[0] if impl_values else None
            elif metric_name in ["latency", "resource_usage", "error_rate"]:
                # Lower is better
                best_impl = min(impl_values.items(), key=lambda x: x[1])[0] if impl_values else None
            else:
                # Default to higher is better
                best_impl = max(impl_values.items(), key=lambda x: x[1])[0] if impl_values else None
            
            # Calculate relative performance
            if best_impl and impl_values:
                best_value = impl_values[best_impl]
                for impl, value in impl_values.items():
                    if metric_name in ["accuracy", "confidence"]:
                        # Higher is better, calculate percentage of best
                        relative_perf = (value / best_value) * 100 if best_value else 100
                    else:
                        # Lower is better, calculate percentage improvement over worst
                        worst_value = max(impl_values.values())
                        relative_perf = ((worst_value - value) / worst_value) * 100 if worst_value else 0
                    
                    comparison_metrics[metric_name][impl] = {
                        "value": value,
                        "relative_performance": relative_perf,
                        "is_best": impl == best_impl
                    }
        
        # Determine overall winner
        implementation_scores = {}
        for impl in implementations:
            score = 0
            for metric_name, metric_data in comparison_metrics.items():
                if impl in metric_data:
                    # Weight by importance of metric
                    weight = 1.0
                    if metric_name == "accuracy":
                        weight = 2.0
                    elif metric_name == "latency":
                        weight = 1.5
                    
                    score += metric_data[impl]["relative_performance"] * weight
            
            implementation_scores[impl] = score
        
        winner = max(implementation_scores.items(), key=lambda x: x[1])[0] if implementation_scores else implementations[0]
        
        # Calculate improvement percentages
        improvement_percentage = {}
        for metric_name, metric_data in comparison_metrics.items():
            if len(metric_data) > 1:  # Need at least 2 implementations to compare
                best_impl = next((impl for impl, data in metric_data.items() if data["is_best"]), None)
                other_impls = [impl for impl in metric_data.keys() if impl != best_impl]
                
                if best_impl and other_impls:
                    # Calculate average improvement over other implementations
                    avg_other_value = statistics.mean([metric_data[impl]["value"] for impl in other_impls])
                    best_value = metric_data[best_impl]["value"]
                    
                    if metric_name in ["accuracy", "confidence"]:
                        # Higher is better
                        if avg_other_value > 0:
                            improvement = ((best_value - avg_other_value) / avg_other_value) * 100
                        else:
                            improvement = 100.0
                    else:
                        # Lower is better
                        if best_value > 0:
                            improvement = ((avg_other_value - best_value) / avg_other_value) * 100
                        else:
                            improvement = 100.0
                    
                    improvement_percentage[metric_name] = improvement   
     
        # Generate trend analysis
        trend_analysis = {}
        for impl, impl_data in implementation_metrics.items():
            impl_trends = {}
            
            for metric_name, time_series in impl_data["time_series"].items():
                if len(time_series) > 5:  # Need enough data points for trend analysis
                    values = [point["value"] for point in time_series]
                    timestamps = [point["timestamp"] for point in time_series]
                    
                    trend_direction, trend_strength = self._analyze_trend(values)
                    anomalies = self._detect_value_anomalies(values)
                    forecast = self._generate_forecast(values, 5)
                    
                    impl_trends[metric_name] = {
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "anomalies": [
                            {
                                "index": idx,
                                "timestamp": timestamps[idx],
                                "value": values[idx],
                                "deviation": deviation
                            }
                            for idx, deviation in anomalies
                        ],
                        "forecast": forecast
                    }
            
            trend_analysis[impl] = impl_trends
        
        # Generate recommendations
        recommendations = []
        
        # Recommend best implementation
        if winner:
            recommendations.append(f"Use {winner} implementation for {component} for best overall performance")
        
        # Specific metric recommendations
        for metric_name, metric_data in comparison_metrics.items():
            best_impl = next((impl for impl, data in metric_data.items() if data["is_best"]), None)
            if best_impl and best_impl != winner:
                recommendations.append(f"Consider {best_impl} implementation for {component} if {metric_name} is critical")
        
        # Trend-based recommendations
        if winner in trend_analysis:
            for metric_name, trend_data in trend_analysis[winner].items():
                if trend_data["direction"] == "decreasing" and metric_name in ["accuracy", "confidence"]:
                    recommendations.append(f"Monitor decreasing {metric_name} trend in {winner} implementation")
                elif trend_data["direction"] == "increasing" and metric_name in ["latency", "resource_usage", "error_rate"]:
                    recommendations.append(f"Monitor increasing {metric_name} trend in {winner} implementation")
        
        # Calculate confidence score
        confidence_factors = []
        
        # More data points = higher confidence
        total_data_points = sum(impl_data["metrics"].get("accuracy", {}).get("count", 0) 
                               for impl_data in implementation_metrics.values())
        data_confidence = min(1.0, total_data_points / 1000)
        confidence_factors.append(data_confidence)
        
        # Larger performance gap = higher confidence
        perf_gaps = []
        for metric_name, metric_data in comparison_metrics.items():
            if len(metric_data) > 1:
                values = [data["value"] for data in metric_data.values()]
                if max(values) > 0:
                    gap = (max(values) - min(values)) / max(values)
                    perf_gaps.append(gap)
        
        gap_confidence = statistics.mean(perf_gaps) if perf_gaps else 0.5
        confidence_factors.append(gap_confidence)
        
        # Consistent winner across metrics = higher confidence
        metric_winners = [
            next((impl for impl, data in metric_data.items() if data["is_best"]), None)
            for metric_data in comparison_metrics.values()
        ]
        winner_counts = {}
        for metric_winner in metric_winners:
            if metric_winner:
                winner_counts[metric_winner] = winner_counts.get(metric_winner, 0) + 1
        
        consistency_confidence = max(winner_counts.values()) / len(metric_winners) if metric_winners else 0.5
        confidence_factors.append(consistency_confidence)
        
        # Calculate overall confidence score
        confidence_score = statistics.mean(confidence_factors)
        
        return PerformanceComparisonReport(
            component=component,
            report_timestamp=datetime.now().isoformat(),
            time_range=(start_time, end_time),
            implementations=implementation_metrics,
            comparison_metrics=comparison_metrics,
            winner=winner,
            improvement_percentage=improvement_percentage,
            confidence_score=confidence_score,
            recommendations=recommendations,
            trend_analysis=trend_analysis
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
        if not end_time:
            end_time = datetime.now().isoformat()
        if not start_time:
            start_time = (datetime.now() - timedelta(days=7)).isoformat()
        
        # Query database for all components with performance data
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT component
                FROM performance_metrics
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time))
            
            components = [row[0] for row in cursor.fetchall()]
        
        # Generate reports for each component
        component_reports = {}
        for component in components:
            report = self.generate_performance_comparison_report(component, start_time, end_time)
            if report:
                component_reports[component] = report
        
        # Aggregate system-wide metrics
        system_metrics = {
            "accuracy": [],
            "latency": [],
            "resource_usage": [],
            "error_rate": [],
            "confidence": []
        }
        
        for component, report in component_reports.items():
            for metric_name in system_metrics.keys():
                for impl, impl_data in report.implementations.items():
                    if metric_name in impl_data["metrics"]:
                        system_metrics[metric_name].append(impl_data["metrics"][metric_name]["avg"])
        
        # Calculate system-wide averages
        system_averages = {}
        for metric_name, values in system_metrics.items():
            if values:
                system_averages[metric_name] = statistics.mean(values)
        
        # Generate system-wide recommendations
        system_recommendations = []
        
        # Identify components with significant performance gaps
        significant_improvements = []
        for component, report in component_reports.items():
            for metric_name, improvement in report.improvement_percentage.items():
                if improvement > 20.0:  # 20% improvement threshold
                    significant_improvements.append({
                        "component": component,
                        "metric": metric_name,
                        "improvement": improvement,
                        "winner": report.winner
                    })
        
        # Sort by improvement percentage
        significant_improvements.sort(key=lambda x: x["improvement"], reverse=True)
        
        # Generate recommendations for top improvements
        for improvement in significant_improvements[:5]:
            system_recommendations.append(
                f"Switch {improvement['component']} to {improvement['winner']} implementation "
                f"for {improvement['improvement']:.1f}% improvement in {improvement['metric']}"
            )
        
        # Identify components with concerning trends
        concerning_trends = []
        for component, report in component_reports.items():
            for impl, trends in report.trend_analysis.items():
                if impl == report.winner:  # Only care about trends in the winning implementation
                    for metric_name, trend_data in trends.items():
                        is_concerning = (
                            (trend_data["direction"] == "decreasing" and metric_name in ["accuracy", "confidence"]) or
                            (trend_data["direction"] == "increasing" and metric_name in ["latency", "resource_usage", "error_rate"])
                        )
                        
                        if is_concerning and trend_data["strength"] > 0.5:  # Strong trend
                            concerning_trends.append({
                                "component": component,
                                "implementation": impl,
                                "metric": metric_name,
                                "direction": trend_data["direction"],
                                "strength": trend_data["strength"]
                            })
        
        # Generate recommendations for concerning trends
        for trend in concerning_trends:
            system_recommendations.append(
                f"Investigate {trend['direction']} {trend['metric']} trend in {trend['component']} "
                f"({trend['implementation']} implementation)"
            )
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "time_range": (start_time, end_time),
            "component_count": len(component_reports),
            "system_averages": system_averages,
            "component_reports": component_reports,
            "significant_improvements": significant_improvements,
            "concerning_trends": concerning_trends,
            "system_recommendations": system_recommendations
        }    
   
    def _apply_smoothing(self, values: List[float], smoothing_factor: float) -> List[float]:
        """Apply exponential smoothing to a time series."""
        if not values:
            return []
        alpha = min(1.0, max(0.0, smoothing_factor))
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed_val = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_val)
        return smoothed
    
    def _analyze_trend(self, values: List[float]) -> Tuple[str, float]:
        """
        Analyze trend direction and strength.
        
        Returns:
            Tuple of (direction, strength) where direction is one of
            "increasing", "decreasing", "stable", "fluctuating" and
            strength is a value between 0.0 and 1.0
        """
        if len(values) < 3:
            return "stable", 0.0
        
        # Calculate linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate R-squared to determine trend strength
        y_pred = [x_mean + slope * (i - x_mean) for i in range(n)]
        ss_total = sum((y - y_mean) ** 2 for y in values)
        ss_residual = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        
        if ss_total == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_residual / ss_total)
        
        # Determine trend direction
        if abs(slope) < 0.001 or r_squared < 0.1:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # If R-squared is low but variance is high, it's fluctuating
        if r_squared < 0.3:
            variance = statistics.variance(values) if len(values) > 1 else 0
            mean = statistics.mean(values)
            coefficient_of_variation = (math.sqrt(variance) / mean) if mean != 0 else 0
            
            if coefficient_of_variation > 0.2:
                direction = "fluctuating"
        
        # Strength is based on R-squared
        strength = min(1.0, max(0.0, r_squared))
        
        return direction, strength
    
    def _detect_value_anomalies(self, values: List[float]) -> List[Tuple[int, float]]:
        """
        Detect anomalies in a time series using Z-score method.
        
        Returns:
            List of tuples (index, deviation) for anomalous points
        """
        if len(values) < 5:
            return []
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        if stdev == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / stdev)
            if z_score > 2.5:  # Threshold for anomaly
                anomalies.append((i, z_score))
        
        return anomalies
    
    def _generate_forecast(self, values: List[float], periods: int) -> List[float]:
        """
        Generate a simple forecast for future values.
        
        Args:
            values: Historical values
            periods: Number of periods to forecast
            
        Returns:
            List of forecasted values
        """
        if len(values) < 5:
            return []
        
        # Use simple exponential smoothing for forecasting
        alpha = 0.3  # Smoothing factor
        
        # Initialize with first value
        level = values[0]
        forecast = []
        
        # Process historical values
        for value in values[1:]:
            last_level = level
            level = alpha * value + (1 - alpha) * last_level
        
        # Generate forecast
        for _ in range(periods):
            forecast.append(level)
        
        return forecast
    
    def _calculate_correlation(self, values1: List[float], values2: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two series."""
        if len(values1) != len(values2) or len(values1) < 3:
            return 0.0
        
        n = len(values1)
        
        # Calculate means
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n
        
        # Calculate covariance and variances
        covariance = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))
        variance1 = sum((x - mean1) ** 2 for x in values1)
        variance2 = sum((x - mean2) ** 2 for x in values2)
        
        # Calculate correlation
        if variance1 == 0 or variance2 == 0:
            return 0.0
        
        correlation = covariance / (math.sqrt(variance1) * math.sqrt(variance2))
        
        return correlation


# Example usage
if __name__ == "__main__":
    visualization = MonitoringSystemVisualization()
    
    # Generate multi-hormone visualization
    multi_viz = visualization.generate_multi_hormone_visualization(
        hormone_names=["dopamine", "serotonin", "cortisol"],
        options=VisualizationOptions(
            chart_type="line",
            smoothing_factor=0.3,
            include_anomalies=True,
            include_forecast=True
        )
    )
    print(f"Generated multi-hormone visualization with {len(multi_viz['hormone_data'])} hormones")
    
    # Generate cascade visualization
    cascade_viz = visualization.generate_cascade_visualization()
    print(f"Generated cascade visualization with {len(cascade_viz['cascades'])} cascades")
    
    # Generate performance comparison report
    perf_report = visualization.generate_performance_comparison_report("hormone_calculator")
    if perf_report:
        print(f"Generated performance report for {perf_report.component}")
        print(f"Winner: {perf_report.winner}")
        print(f"Recommendations: {perf_report.recommendations}")
    
    # Generate system performance report
    sys_report = visualization.generate_system_performance_report()
    print(f"Generated system performance report with {sys_report['component_count']} components")
    print(f"System recommendations: {sys_report['system_recommendations']}")