"""
P2P Global Performance Projection and Benchmarking System

This module provides comprehensive P2P network performance analysis, projection,
and benchmarking capabilities for the MCP system.

Features:
- Global P2P network performance monitoring
- Performance projection and prediction
- Benchmarking against historical data
- Network topology analysis
- Latency and throughput optimization
- Geographic performance mapping
- Load balancing recommendations
- Performance trend analysis
- Real-time performance alerts
- Cross-region performance comparison
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import random
from pathlib import Path

# System imports
from .system_integration_layer import SystemIntegrationLayer
from .async_processing_framework import AsyncProcessingFramework


class PerformanceMetric(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    BANDWIDTH = "bandwidth"
    CONNECTIONS = "connections"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


class BenchmarkType(Enum):
    """Types of benchmarks"""
    LATENCY_BENCHMARK = "latency"
    THROUGHPUT_BENCHMARK = "throughput"
    AVAILABILITY_BENCHMARK = "availability"
    RELIABILITY_BENCHMARK = "reliability"
    LOAD_TEST = "load_test"
    STRESS_TEST = "stress_test"
    ENDURANCE_TEST = "endurance_test"


@dataclass
class P2PNodeMetrics:
    """P2P node performance metrics"""
    node_id: str
    region: str
    country: str
    latency_ms: float
    throughput_mbps: float
    availability_percent: float
    reliability_score: float
    bandwidth_mbps: float
    active_connections: int
    response_time_ms: float
    error_rate_percent: float
    cpu_usage: float
    memory_usage: float
    network_usage: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkTopology:
    """P2P network topology information"""
    total_nodes: int
    regions: Dict[str, int]
    countries: Dict[str, int]
    connection_matrix: Dict[str, List[str]]
    average_connections_per_node: float
    network_diameter: int
    clustering_coefficient: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceProjection:
    """Performance projection data"""
    metric_type: PerformanceMetric
    current_value: float
    projected_value: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0.0 to 1.0
    factors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    benchmark_type: BenchmarkType
    test_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    error_rate_percent: float
    performance_score: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceConfig:
    """Configuration for P2P performance system"""
    # Monitoring settings
    monitoring_interval: float = 30.0  # seconds
    metrics_history_size: int = 1000
    enable_real_time_monitoring: bool = True
    
    # Projection settings
    projection_horizon_hours: int = 24
    confidence_level: float = 0.95
    enable_machine_learning: bool = True
    
    # Benchmarking settings
    benchmark_duration: float = 300.0  # 5 minutes
    benchmark_concurrent_requests: int = 100
    benchmark_timeout: float = 30.0
    enable_automated_benchmarking: bool = True
    
    # Alerting settings
    latency_threshold_ms: float = 200.0
    throughput_threshold_mbps: float = 10.0
    availability_threshold_percent: float = 95.0
    error_rate_threshold_percent: float = 5.0
    
    # Geographic settings
    enable_geographic_analysis: bool = True
    region_grouping: bool = True
    country_grouping: bool = True
    
    # Optimization settings
    enable_load_balancing: bool = True
    enable_route_optimization: bool = True
    enable_connection_pooling: bool = True


class P2PGlobalPerformanceSystem:
    """
    Comprehensive P2P global performance projection and benchmarking system.
    
    This system provides advanced performance monitoring, projection, and
    benchmarking capabilities for P2P networks.
    """
    
    def __init__(self, 
                 system_integration: Optional[SystemIntegrationLayer] = None,
                 async_framework: Optional[AsyncProcessingFramework] = None,
                 config: Optional[PerformanceConfig] = None):
        """Initialize the P2P global performance system"""
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger("P2PGlobalPerformanceSystem")
        
        # System integration
        self.system_integration = system_integration
        self.async_framework = async_framework
        
        # State management
        self.running = False
        self.node_metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metrics_history_size)
        )
        self.network_topology_history: deque = deque(maxlen=100)
        self.performance_projections: Dict[PerformanceMetric, List[PerformanceProjection]] = defaultdict(list)
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Performance tracking
        self.global_metrics: Dict[str, float] = {}
        self.regional_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.country_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Alerting
        self.performance_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Machine learning models (placeholder for future implementation)
        self.ml_models: Dict[str, Any] = {}
        
        self.logger.info("P2PGlobalPerformanceSystem initialized")
    
    async def start(self):
        """Start the P2P global performance system"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting P2PGlobalPerformanceSystem")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._projection_loop()),
            asyncio.create_task(self._benchmarking_loop()),
            asyncio.create_task(self._alerting_loop()),
            asyncio.create_task(self._optimization_loop())
        ]
        
        self.logger.info("P2PGlobalPerformanceSystem started successfully")
    
    async def stop(self):
        """Stop the P2P global performance system"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping P2PGlobalPerformanceSystem")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("P2PGlobalPerformanceSystem stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect P2P network metrics
                await self._collect_p2p_metrics()
                
                # Update global metrics
                await self._update_global_metrics()
                
                # Update regional and country metrics
                await self._update_geographic_metrics()
                
                # Check for performance alerts
                await self._check_performance_alerts()
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _projection_loop(self):
        """Performance projection loop"""
        while self.running:
            try:
                if self.config.enable_machine_learning:
                    await self._generate_performance_projections()
                
                await asyncio.sleep(3600)  # Generate projections every hour
                
            except Exception as e:
                self.logger.error(f"Error in projection loop: {e}")
                await asyncio.sleep(3600)
    
    async def _benchmarking_loop(self):
        """Automated benchmarking loop"""
        while self.running:
            try:
                if self.config.enable_automated_benchmarking:
                    await self._run_automated_benchmarks()
                
                await asyncio.sleep(7200)  # Run benchmarks every 2 hours
                
            except Exception as e:
                self.logger.error(f"Error in benchmarking loop: {e}")
                await asyncio.sleep(7200)
    
    async def _alerting_loop(self):
        """Performance alerting loop"""
        while self.running:
            try:
                await self._process_performance_alerts()
                
                await asyncio.sleep(300)  # Process alerts every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(300)
    
    async def _optimization_loop(self):
        """Performance optimization loop"""
        while self.running:
            try:
                await self._optimize_network_performance()
                
                await asyncio.sleep(1800)  # Optimize every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(1800)
    
    async def _collect_p2p_metrics(self):
        """Collect P2P network metrics from all nodes"""
        try:
            if not self.system_integration or not hasattr(self.system_integration, 'p2p_network'):
                return
            
            p2p_network = self.system_integration.p2p_network
            nodes = p2p_network.get_all_nodes()
            
            for node in nodes:
                try:
                    metrics = await self._collect_node_metrics(node)
                    if metrics:
                        self.node_metrics_history[node['id']].append(metrics)
                        
                        # Update topology if needed
                        await self._update_network_topology()
                        
                except Exception as e:
                    self.logger.error(f"Error collecting metrics for node {node.get('id', 'unknown')}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error collecting P2P metrics: {e}")
    
    async def _collect_node_metrics(self, node: Dict[str, Any]) -> Optional[P2PNodeMetrics]:
        """Collect metrics for a specific node"""
        try:
            node_id = node.get('id', 'unknown')
            
            # Simulate metric collection (replace with actual implementation)
            metrics = P2PNodeMetrics(
                node_id=node_id,
                region=node.get('region', 'unknown'),
                country=node.get('country', 'unknown'),
                latency_ms=random.uniform(10, 200),
                throughput_mbps=random.uniform(1, 100),
                availability_percent=random.uniform(90, 99.9),
                reliability_score=random.uniform(0.8, 1.0),
                bandwidth_mbps=random.uniform(10, 1000),
                active_connections=random.randint(1, 100),
                response_time_ms=random.uniform(5, 50),
                error_rate_percent=random.uniform(0, 5),
                cpu_usage=random.uniform(10, 80),
                memory_usage=random.uniform(20, 90),
                network_usage=random.uniform(5, 60)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting node metrics: {e}")
            return None
    
    async def _update_network_topology(self):
        """Update network topology information"""
        try:
            if not self.system_integration or not hasattr(self.system_integration, 'p2p_network'):
                return
            
            p2p_network = self.system_integration.p2p_network
            nodes = p2p_network.get_all_nodes()
            
            # Calculate topology metrics
            regions = defaultdict(int)
            countries = defaultdict(int)
            connection_matrix = defaultdict(list)
            
            for node in nodes:
                region = node.get('region', 'unknown')
                country = node.get('country', 'unknown')
                node_id = node.get('id', 'unknown')
                
                regions[region] += 1
                countries[country] += 1
                
                # Get connections (simulated)
                connections = random.sample([n['id'] for n in nodes if n['id'] != node_id], 
                                         min(5, len(nodes) - 1))
                connection_matrix[node_id] = connections
            
            # Calculate network metrics
            total_nodes = len(nodes)
            total_connections = sum(len(conns) for conns in connection_matrix.values())
            avg_connections = total_connections / total_nodes if total_nodes > 0 else 0
            
            # Calculate network diameter (simplified)
            network_diameter = self._calculate_network_diameter(connection_matrix)
            
            # Calculate clustering coefficient (simplified)
            clustering_coefficient = self._calculate_clustering_coefficient(connection_matrix)
            
            topology = NetworkTopology(
                total_nodes=total_nodes,
                regions=dict(regions),
                countries=dict(countries),
                connection_matrix=dict(connection_matrix),
                average_connections_per_node=avg_connections,
                network_diameter=network_diameter,
                clustering_coefficient=clustering_coefficient
            )
            
            self.network_topology_history.append(topology)
            
        except Exception as e:
            self.logger.error(f"Error updating network topology: {e}")
    
    def _calculate_network_diameter(self, connection_matrix: Dict[str, List[str]]) -> int:
        """Calculate network diameter (simplified implementation)"""
        try:
            if not connection_matrix:
                return 0
            
            # Simplified diameter calculation
            max_path_length = 0
            nodes = list(connection_matrix.keys())
            
            for start_node in nodes[:10]:  # Limit for performance
                for end_node in nodes[:10]:
                    if start_node != end_node:
                        path_length = self._find_shortest_path(connection_matrix, start_node, end_node)
                        max_path_length = max(max_path_length, path_length)
            
            return max_path_length
            
        except Exception as e:
            self.logger.error(f"Error calculating network diameter: {e}")
            return 0
    
    def _find_shortest_path(self, connection_matrix: Dict[str, List[str]], 
                           start: str, end: str) -> int:
        """Find shortest path between two nodes (BFS)"""
        try:
            if start == end:
                return 0
            
            visited = set()
            queue = [(start, 0)]
            
            while queue:
                node, distance = queue.pop(0)
                
                if node == end:
                    return distance
                
                if node in visited:
                    continue
                
                visited.add(node)
                
                for neighbor in connection_matrix.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
            
            return float('inf')  # No path found
            
        except Exception as e:
            self.logger.error(f"Error finding shortest path: {e}")
            return 0
    
    def _calculate_clustering_coefficient(self, connection_matrix: Dict[str, List[str]]) -> float:
        """Calculate clustering coefficient (simplified implementation)"""
        try:
            if not connection_matrix:
                return 0.0
            
            total_coefficient = 0.0
            node_count = 0
            
            for node, neighbors in connection_matrix.items():
                if len(neighbors) < 2:
                    continue
                
                # Count triangles
                triangles = 0
                for i, neighbor1 in enumerate(neighbors):
                    for neighbor2 in neighbors[i+1:]:
                        if neighbor2 in connection_matrix.get(neighbor1, []):
                            triangles += 1
                
                # Calculate local clustering coefficient
                max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
                if max_triangles > 0:
                    local_coefficient = triangles / max_triangles
                    total_coefficient += local_coefficient
                    node_count += 1
            
            return total_coefficient / node_count if node_count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating clustering coefficient: {e}")
            return 0.0
    
    async def _update_global_metrics(self):
        """Update global performance metrics"""
        try:
            all_metrics = []
            for node_metrics in self.node_metrics_history.values():
                if node_metrics:
                    all_metrics.extend(node_metrics)
            
            if not all_metrics:
                return
            
            # Calculate global averages
            self.global_metrics = {
                'average_latency_ms': statistics.mean([m.latency_ms for m in all_metrics]),
                'average_throughput_mbps': statistics.mean([m.throughput_mbps for m in all_metrics]),
                'average_availability_percent': statistics.mean([m.availability_percent for m in all_metrics]),
                'average_reliability_score': statistics.mean([m.reliability_score for m in all_metrics]),
                'average_bandwidth_mbps': statistics.mean([m.bandwidth_mbps for m in all_metrics]),
                'total_active_connections': sum([m.active_connections for m in all_metrics]),
                'average_response_time_ms': statistics.mean([m.response_time_ms for m in all_metrics]),
                'average_error_rate_percent': statistics.mean([m.error_rate_percent for m in all_metrics]),
                'total_nodes': len(self.node_metrics_history),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating global metrics: {e}")
    
    async def _update_geographic_metrics(self):
        """Update regional and country performance metrics"""
        try:
            # Group metrics by region and country
            regional_data = defaultdict(list)
            country_data = defaultdict(list)
            
            for node_metrics in self.node_metrics_history.values():
                if node_metrics:
                    latest_metrics = node_metrics[-1]
                    regional_data[latest_metrics.region].append(latest_metrics)
                    country_data[latest_metrics.country].append(latest_metrics)
            
            # Calculate regional metrics
            for region, metrics in regional_data.items():
                self.regional_metrics[region] = {
                    'average_latency_ms': statistics.mean([m.latency_ms for m in metrics]),
                    'average_throughput_mbps': statistics.mean([m.throughput_mbps for m in metrics]),
                    'average_availability_percent': statistics.mean([m.availability_percent for m in metrics]),
                    'node_count': len(metrics),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate country metrics
            for country, metrics in country_data.items():
                self.country_metrics[country] = {
                    'average_latency_ms': statistics.mean([m.latency_ms for m in metrics]),
                    'average_throughput_mbps': statistics.mean([m.throughput_mbps for m in metrics]),
                    'average_availability_percent': statistics.mean([m.availability_percent for m in metrics]),
                    'node_count': len(metrics),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error updating geographic metrics: {e}")
    
    async def _generate_performance_projections(self):
        """Generate performance projections using historical data"""
        try:
            for metric_type in PerformanceMetric:
                projection = await self._project_metric(metric_type)
                if projection:
                    self.performance_projections[metric_type].append(projection)
                    
                    # Keep only recent projections
                    if len(self.performance_projections[metric_type]) > 100:
                        self.performance_projections[metric_type] = \
                            self.performance_projections[metric_type][-100:]
            
        except Exception as e:
            self.logger.error(f"Error generating performance projections: {e}")
    
    async def _project_metric(self, metric_type: PerformanceMetric) -> Optional[PerformanceProjection]:
        """Project a specific metric"""
        try:
            # Collect historical data for the metric
            historical_data = []
            
            for node_metrics in self.node_metrics_history.values():
                for metrics in node_metrics:
                    if metric_type == PerformanceMetric.LATENCY:
                        historical_data.append(metrics.latency_ms)
                    elif metric_type == PerformanceMetric.THROUGHPUT:
                        historical_data.append(metrics.throughput_mbps)
                    elif metric_type == PerformanceMetric.AVAILABILITY:
                        historical_data.append(metrics.availability_percent)
                    elif metric_type == PerformanceMetric.RELIABILITY:
                        historical_data.append(metrics.reliability_score)
                    elif metric_type == PerformanceMetric.BANDWIDTH:
                        historical_data.append(metrics.bandwidth_mbps)
                    elif metric_type == PerformanceMetric.CONNECTIONS:
                        historical_data.append(metrics.active_connections)
                    elif metric_type == PerformanceMetric.RESPONSE_TIME:
                        historical_data.append(metrics.response_time_ms)
                    elif metric_type == PerformanceMetric.ERROR_RATE:
                        historical_data.append(metrics.error_rate_percent)
            
            if len(historical_data) < 10:
                return None
            
            # Calculate current value (average of recent data)
            current_value = statistics.mean(historical_data[-10:])
            
            # Simple linear projection (replace with ML model)
            if len(historical_data) >= 20:
                recent_trend = statistics.mean(historical_data[-10:]) - statistics.mean(historical_data[-20:-10])
                projected_value = current_value + recent_trend * 24  # 24 hours ahead
                
                # Calculate confidence interval
                std_dev = statistics.stdev(historical_data[-20:])
                confidence_margin = 1.96 * std_dev / math.sqrt(20)  # 95% confidence
                confidence_interval = (projected_value - confidence_margin, 
                                     projected_value + confidence_margin)
                
                # Determine trend direction
                if recent_trend > 0.01:
                    trend_direction = "improving"
                    trend_strength = min(abs(recent_trend) / current_value, 1.0)
                elif recent_trend < -0.01:
                    trend_direction = "declining"
                    trend_strength = min(abs(recent_trend) / current_value, 1.0)
                else:
                    trend_direction = "stable"
                    trend_strength = 0.0
                
                return PerformanceProjection(
                    metric_type=metric_type,
                    current_value=current_value,
                    projected_value=projected_value,
                    confidence_interval=confidence_interval,
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    factors=["historical_trend", "network_load", "geographic_distribution"]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error projecting metric {metric_type}: {e}")
            return None
    
    async def _run_automated_benchmarks(self):
        """Run automated performance benchmarks"""
        try:
            # Run different types of benchmarks
            benchmark_types = [
                BenchmarkType.LATENCY_BENCHMARK,
                BenchmarkType.THROUGHPUT_BENCHMARK,
                BenchmarkType.AVAILABILITY_BENCHMARK,
                BenchmarkType.LOAD_TEST
            ]
            
            for benchmark_type in benchmark_types:
                result = await self._run_benchmark(benchmark_type)
                if result:
                    self.benchmark_results.append(result)
                    
                    # Keep only recent results
                    if len(self.benchmark_results) > 50:
                        self.benchmark_results = self.benchmark_results[-50:]
            
        except Exception as e:
            self.logger.error(f"Error running automated benchmarks: {e}")
    
    async def _run_benchmark(self, benchmark_type: BenchmarkType) -> Optional[BenchmarkResult]:
        """Run a specific benchmark test"""
        try:
            start_time = time.time()
            
            # Simulate benchmark test
            total_requests = self.config.benchmark_concurrent_requests * 10
            successful_requests = int(total_requests * 0.95)  # 95% success rate
            failed_requests = total_requests - successful_requests
            
            # Simulate latency distribution
            latencies = [random.uniform(10, 200) for _ in range(successful_requests)]
            average_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Calculate throughput
            test_duration = time.time() - start_time
            throughput_rps = successful_requests / test_duration if test_duration > 0 else 0
            
            # Calculate error rate
            error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
            
            # Calculate performance score
            performance_score = self._calculate_benchmark_score(
                average_latency, throughput_rps, error_rate, benchmark_type
            )
            
            # Generate recommendations
            recommendations = self._generate_benchmark_recommendations(
                average_latency, throughput_rps, error_rate, benchmark_type
            )
            
            return BenchmarkResult(
                benchmark_type=benchmark_type,
                test_duration=test_duration,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_latency_ms=average_latency,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                throughput_rps=throughput_rps,
                error_rate_percent=error_rate,
                performance_score=performance_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error running benchmark {benchmark_type}: {e}")
            return None
    
    def _calculate_benchmark_score(self, latency: float, throughput: float, 
                                  error_rate: float, benchmark_type: BenchmarkType) -> float:
        """Calculate performance score for benchmark result"""
        try:
            # Base score starts at 100
            score = 100.0
            
            # Penalize high latency
            if latency > 100:
                score -= (latency - 100) * 0.5
            
            # Penalize low throughput
            if throughput < 10:
                score -= (10 - throughput) * 2
            
            # Penalize high error rate
            score -= error_rate * 2
            
            # Bonus for good performance
            if latency < 50:
                score += 10
            if throughput > 50:
                score += 10
            if error_rate < 1:
                score += 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark score: {e}")
            return 0.0
    
    def _generate_benchmark_recommendations(self, latency: float, throughput: float,
                                          error_rate: float, benchmark_type: BenchmarkType) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        if latency > 150:
            recommendations.append("Consider optimizing network routing to reduce latency")
        
        if throughput < 20:
            recommendations.append("Increase network bandwidth or optimize data transfer")
        
        if error_rate > 5:
            recommendations.append("Investigate and fix network reliability issues")
        
        if benchmark_type == BenchmarkType.LOAD_TEST and throughput < 50:
            recommendations.append("Consider scaling up network capacity for high load")
        
        if benchmark_type == BenchmarkType.AVAILABILITY_BENCHMARK and error_rate > 2:
            recommendations.append("Implement redundancy and failover mechanisms")
        
        return recommendations
    
    async def _check_performance_alerts(self):
        """Check for performance alerts"""
        try:
            alerts = []
            
            # Check global metrics
            if self.global_metrics:
                if self.global_metrics.get('average_latency_ms', 0) > self.config.latency_threshold_ms:
                    alerts.append({
                        'type': 'high_latency',
                        'severity': 'warning',
                        'message': f"Global average latency ({self.global_metrics['average_latency_ms']:.1f}ms) exceeds threshold",
                        'value': self.global_metrics['average_latency_ms'],
                        'threshold': self.config.latency_threshold_ms
                    })
                
                if self.global_metrics.get('average_throughput_mbps', 0) < self.config.throughput_threshold_mbps:
                    alerts.append({
                        'type': 'low_throughput',
                        'severity': 'warning',
                        'message': f"Global average throughput ({self.global_metrics['average_throughput_mbps']:.1f}Mbps) below threshold",
                        'value': self.global_metrics['average_throughput_mbps'],
                        'threshold': self.config.throughput_threshold_mbps
                    })
                
                if self.global_metrics.get('average_availability_percent', 100) < self.config.availability_threshold_percent:
                    alerts.append({
                        'type': 'low_availability',
                        'severity': 'critical',
                        'message': f"Global availability ({self.global_metrics['average_availability_percent']:.1f}%) below threshold",
                        'value': self.global_metrics['average_availability_percent'],
                        'threshold': self.config.availability_threshold_percent
                    })
                
                if self.global_metrics.get('average_error_rate_percent', 0) > self.config.error_rate_threshold_percent:
                    alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'critical',
                        'message': f"Global error rate ({self.global_metrics['average_error_rate_percent']:.1f}%) exceeds threshold",
                        'value': self.global_metrics['average_error_rate_percent'],
                        'threshold': self.config.error_rate_threshold_percent
                    })
            
            # Add alerts to history
            self.performance_alerts.extend(alerts)
            self.alert_history.extend(alerts)
            
            # Keep only recent alerts
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-100:]
            
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    async def _process_performance_alerts(self):
        """Process and handle performance alerts"""
        try:
            # Process critical alerts
            critical_alerts = [a for a in self.performance_alerts if a['severity'] == 'critical']
            
            for alert in critical_alerts:
                self.logger.critical(f"Critical performance alert: {alert['message']}")
                
                # Trigger immediate optimization
                await self._trigger_emergency_optimization(alert)
            
            # Process warning alerts
            warning_alerts = [a for a in self.performance_alerts if a['severity'] == 'warning']
            
            for alert in warning_alerts:
                self.logger.warning(f"Performance warning: {alert['message']}")
                
                # Schedule optimization
                await self._schedule_optimization(alert)
            
            # Clear processed alerts
            self.performance_alerts.clear()
            
        except Exception as e:
            self.logger.error(f"Error processing performance alerts: {e}")
    
    async def _trigger_emergency_optimization(self, alert: Dict[str, Any]):
        """Trigger emergency optimization for critical alerts"""
        try:
            alert_type = alert['type']
            
            if alert_type == 'high_latency':
                await self._optimize_latency()
            elif alert_type == 'low_throughput':
                await self._optimize_throughput()
            elif alert_type == 'low_availability':
                await self._optimize_availability()
            elif alert_type == 'high_error_rate':
                await self._optimize_reliability()
            
        except Exception as e:
            self.logger.error(f"Error triggering emergency optimization: {e}")
    
    async def _schedule_optimization(self, alert: Dict[str, Any]):
        """Schedule optimization for warning alerts"""
        try:
            # Schedule optimization in background
            if self.async_framework:
                await self.async_framework.submit_task(
                    self._run_scheduled_optimization(alert),
                    priority="HIGH"
                )
            
        except Exception as e:
            self.logger.error(f"Error scheduling optimization: {e}")
    
    async def _run_scheduled_optimization(self, alert: Dict[str, Any]):
        """Run scheduled optimization"""
        try:
            alert_type = alert['type']
            
            if alert_type == 'high_latency':
                await self._optimize_latency()
            elif alert_type == 'low_throughput':
                await self._optimize_throughput()
            
        except Exception as e:
            self.logger.error(f"Error running scheduled optimization: {e}")
    
    async def _optimize_network_performance(self):
        """Optimize network performance based on current metrics"""
        try:
            if not self.global_metrics:
                return
            
            # Check if optimization is needed
            latency = self.global_metrics.get('average_latency_ms', 0)
            throughput = self.global_metrics.get('average_throughput_mbps', 0)
            availability = self.global_metrics.get('average_availability_percent', 100)
            error_rate = self.global_metrics.get('average_error_rate_percent', 0)
            
            # Optimize based on metrics
            if latency > 100:
                await self._optimize_latency()
            
            if throughput < 20:
                await self._optimize_throughput()
            
            if availability < 98:
                await self._optimize_availability()
            
            if error_rate > 2:
                await self._optimize_reliability()
            
        except Exception as e:
            self.logger.error(f"Error optimizing network performance: {e}")
    
    async def _optimize_latency(self):
        """Optimize network latency"""
        try:
            self.logger.info("Optimizing network latency")
            
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                # Implement latency optimization
                p2p_network = self.system_integration.p2p_network
                
                # Optimize routing
                p2p_network.optimize_routing()
                
                # Optimize connection pooling
                p2p_network.optimize_connections()
            
        except Exception as e:
            self.logger.error(f"Error optimizing latency: {e}")
    
    async def _optimize_throughput(self):
        """Optimize network throughput"""
        try:
            self.logger.info("Optimizing network throughput")
            
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                # Implement throughput optimization
                p2p_network = self.system_integration.p2p_network
                
                # Optimize bandwidth allocation
                p2p_network.optimize_bandwidth()
                
                # Optimize data transfer protocols
                p2p_network.optimize_transfer_protocols()
            
        except Exception as e:
            self.logger.error(f"Error optimizing throughput: {e}")
    
    async def _optimize_availability(self):
        """Optimize network availability"""
        try:
            self.logger.info("Optimizing network availability")
            
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                # Implement availability optimization
                p2p_network = self.system_integration.p2p_network
                
                # Implement redundancy
                p2p_network.implement_redundancy()
                
                # Optimize failover mechanisms
                p2p_network.optimize_failover()
            
        except Exception as e:
            self.logger.error(f"Error optimizing availability: {e}")
    
    async def _optimize_reliability(self):
        """Optimize network reliability"""
        try:
            self.logger.info("Optimizing network reliability")
            
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                # Implement reliability optimization
                p2p_network = self.system_integration.p2p_network
                
                # Implement error correction
                p2p_network.implement_error_correction()
                
                # Optimize retry mechanisms
                p2p_network.optimize_retry_mechanisms()
            
        except Exception as e:
            self.logger.error(f"Error optimizing reliability: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            return {
                'global_metrics': self.global_metrics,
                'regional_metrics': dict(self.regional_metrics),
                'country_metrics': dict(self.country_metrics),
                'network_topology': {
                    'current': self.network_topology_history[-1].__dict__ if self.network_topology_history else None,
                    'total_nodes': len(self.node_metrics_history),
                    'regions': len(self.regional_metrics),
                    'countries': len(self.country_metrics)
                },
                'performance_projections': {
                    metric.value: [p.__dict__ for p in projections[-5:]]  # Last 5 projections
                    for metric, projections in self.performance_projections.items()
                },
                'recent_benchmarks': [
                    result.__dict__ for result in self.benchmark_results[-5:]  # Last 5 benchmarks
                ],
                'active_alerts': len(self.performance_alerts),
                'total_alerts': len(self.alert_history),
                'system_status': 'running' if self.running else 'stopped',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def run_custom_benchmark(self, benchmark_type: BenchmarkType, 
                                 duration: float = None, 
                                 concurrent_requests: int = None) -> Optional[BenchmarkResult]:
        """Run a custom benchmark test"""
        try:
            # Use provided parameters or defaults
            test_duration = duration or self.config.benchmark_duration
            requests = concurrent_requests or self.config.benchmark_concurrent_requests
            
            # Temporarily update config
            original_duration = self.config.benchmark_duration
            original_requests = self.config.benchmark_concurrent_requests
            
            self.config.benchmark_duration = test_duration
            self.config.benchmark_concurrent_requests = requests
            
            # Run benchmark
            result = await self._run_benchmark(benchmark_type)
            
            # Restore original config
            self.config.benchmark_duration = original_duration
            self.config.benchmark_concurrent_requests = original_requests
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running custom benchmark: {e}")
            return None
    
    async def get_performance_projections(self, metric_type: PerformanceMetric = None) -> List[PerformanceProjection]:
        """Get performance projections"""
        try:
            if metric_type:
                return self.performance_projections.get(metric_type, [])
            else:
                # Return all projections
                all_projections = []
                for projections in self.performance_projections.values():
                    all_projections.extend(projections)
                return all_projections
            
        except Exception as e:
            self.logger.error(f"Error getting performance projections: {e}")
            return []
    
    async def get_geographic_performance(self, region: str = None, country: str = None) -> Dict[str, Any]:
        """Get geographic performance data"""
        try:
            if region:
                return self.regional_metrics.get(region, {})
            elif country:
                return self.country_metrics.get(country, {})
            else:
                return {
                    'regions': dict(self.regional_metrics),
                    'countries': dict(self.country_metrics)
                }
            
        except Exception as e:
            self.logger.error(f"Error getting geographic performance: {e}")
            return {}


# Global instance management
_p2p_performance_system_instance: Optional[P2PGlobalPerformanceSystem] = None


def get_p2p_performance_system(system_integration: Optional[SystemIntegrationLayer] = None,
                              async_framework: Optional[AsyncProcessingFramework] = None,
                              config: Optional[PerformanceConfig] = None) -> P2PGlobalPerformanceSystem:
    """Get the global P2P performance system instance"""
    global _p2p_performance_system_instance
    
    if _p2p_performance_system_instance is None:
        _p2p_performance_system_instance = P2PGlobalPerformanceSystem(
            system_integration=system_integration,
            async_framework=async_framework,
            config=config
        )
    
    return _p2p_performance_system_instance


async def initialize_p2p_performance_system(system_integration: Optional[SystemIntegrationLayer] = None,
                                           async_framework: Optional[AsyncProcessingFramework] = None,
                                           config: Optional[PerformanceConfig] = None) -> P2PGlobalPerformanceSystem:
    """Initialize the global P2P performance system"""
    global _p2p_performance_system_instance
    
    if _p2p_performance_system_instance is not None:
        await _p2p_performance_system_instance.stop()
    
    _p2p_performance_system_instance = P2PGlobalPerformanceSystem(
        system_integration=system_integration,
        async_framework=async_framework,
        config=config
    )
    
    return _p2p_performance_system_instance 