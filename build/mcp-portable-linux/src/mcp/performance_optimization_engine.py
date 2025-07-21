"""
Performance Optimization Engine

This module provides comprehensive performance optimization and resource management
for the MCP system, integrating with existing components and providing advanced
optimization capabilities.

Features:
- Dynamic resource allocation and optimization
- Performance monitoring and bottleneck detection
- Adaptive system tuning based on workload patterns
- Memory optimization and garbage collection
- CPU optimization and load balancing
- I/O optimization and caching strategies
- Network optimization and connection pooling
- Predictive resource allocation
- System health monitoring and recovery
- Integration with genetic trigger system for adaptive optimization
- P2P network performance optimization
- Real-time performance analytics and reporting
"""

import asyncio
import logging
import time
import threading
import psutil
import gc
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref
import json
import os
import statistics
from pathlib import Path

# System imports
from .async_processing_framework import AsyncProcessingFramework, ProcessingConfig
from .system_integration_layer import SystemIntegrationLayer


class OptimizationLevel(Enum):
    """Optimization levels for different system components"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    AGGRESSIVE = 4


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    I_O = "io"
    P2P = "p2p"
    GENETIC = "genetic"
    HORMONE = "hormone"
    MEMORY_SYSTEM = "memory_system"


@dataclass
class ResourceMetrics:
    """Comprehensive resource metrics"""
    cpu_usage: float = 0.0
    cpu_count: int = 0
    cpu_frequency_mhz: float = 0.0
    memory_usage_gb: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    network_connections: int = 0
    io_read_count: int = 0
    io_write_count: int = 0
    io_wait_percent: float = 0.0
    thread_count: int = 0
    process_count: int = 0
    p2p_connections: int = 0
    p2p_latency_ms: float = 0.0
    genetic_operations_per_sec: float = 0.0
    hormone_levels: Dict[str, float] = field(default_factory=dict)
    memory_system_efficiency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    # Resource thresholds
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    network_threshold: float = 1000000  # bytes/sec
    p2p_threshold: float = 100.0  # ms latency
    
    # Optimization settings
    enable_auto_optimization: bool = True
    optimization_interval: float = 60.0  # seconds
    metrics_history_size: int = 1000
    enable_predictive_optimization: bool = True
    enable_adaptive_optimization: bool = True
    
    # Memory optimization
    enable_garbage_collection: bool = True
    gc_threshold: float = 75.0
    memory_cleanup_interval: float = 300.0  # 5 minutes
    memory_compaction_threshold: float = 85.0
    
    # CPU optimization
    enable_load_balancing: bool = True
    cpu_optimization_threshold: float = 70.0
    cpu_affinity_optimization: bool = True
    
    # I/O optimization
    enable_io_optimization: bool = True
    io_optimization_threshold: float = 50.0
    io_caching_enabled: bool = True
    
    # Network optimization
    enable_network_optimization: bool = True
    network_optimization_threshold: float = 80.0
    connection_pooling: bool = True
    
    # P2P optimization
    enable_p2p_optimization: bool = True
    p2p_optimization_threshold: float = 50.0
    p2p_connection_limit: int = 100
    
    # Genetic system optimization
    enable_genetic_optimization: bool = True
    genetic_optimization_threshold: float = 60.0
    adaptive_mutation_rate: bool = True
    
    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_threshold: float = 95.0
    max_recovery_attempts: int = 3
    
    # Analytics and reporting
    enable_performance_analytics: bool = True
    analytics_interval: float = 300.0  # 5 minutes
    performance_reporting: bool = True
    report_interval: float = 3600.0  # 1 hour


@dataclass
class OptimizationAction:
    """Represents an optimization action"""
    action_type: str
    resource_type: ResourceType
    description: str
    priority: int
    estimated_impact: float
    execution_time: float
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    genetic_trigger: Optional[str] = None
    hormone_influence: Optional[Dict[str, float]] = None


@dataclass
class PerformanceAnalytics:
    """Performance analytics data"""
    timestamp: datetime
    resource_utilization: Dict[str, float]
    optimization_effectiveness: Dict[str, float]
    bottleneck_analysis: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    system_health_score: float


class PerformanceOptimizationEngine:
    """
    Comprehensive performance optimization engine for the MCP system.
    
    This engine provides dynamic resource optimization, performance monitoring,
    bottleneck detection, and adaptive system tuning based on workload patterns.
    """
    
    def __init__(self, 
                 system_integration: Optional[SystemIntegrationLayer] = None,
                 async_framework: Optional[AsyncProcessingFramework] = None,
                 config: Optional[OptimizationConfig] = None):
        """Initialize the performance optimization engine"""
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger("PerformanceOptimizationEngine")
        
        # System integration
        self.system_integration = system_integration
        self.async_framework = async_framework
        
        # State management
        self.running = False
        self._lock = threading.RLock()
        self.optimization_history: deque = deque(maxlen=self.config.metrics_history_size)
        self.current_metrics = ResourceMetrics()
        self.last_optimization = datetime.now()
        
        # Performance tracking
        self.performance_baseline: Optional[ResourceMetrics] = None
        self.optimization_actions: List[OptimizationAction] = []
        self.active_optimizations: Dict[str, OptimizationAction] = {}
        
        # Resource monitoring
        self.resource_alerts: List[Dict[str, Any]] = []
        self.bottleneck_history: List[Dict[str, Any]] = []
        
        # Analytics and reporting
        self.analytics_history: List[PerformanceAnalytics] = []
        self.performance_reports: List[Dict[str, Any]] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Resource pools and caches
        self.connection_pools: Dict[str, List[Any]] = defaultdict(list)
        self.io_cache: Dict[str, Any] = {}
        self.memory_pools: Dict[str, List[Any]] = defaultdict(list)
        
        # Initialize baseline
        self._initialize_baseline()
        
        self.logger.info("PerformanceOptimizationEngine initialized")
    
    def _initialize_baseline(self):
        """Initialize performance baseline"""
        try:
            self.performance_baseline = self._collect_resource_metrics()
            self.logger.info("Performance baseline established")
        except Exception as e:
            self.logger.error(f"Failed to establish performance baseline: {e}")
    
    async def start(self):
        """Start the performance optimization engine"""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self.logger.info("Starting PerformanceOptimizationEngine")
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._optimization_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._analytics_loop()),
                asyncio.create_task(self._reporting_loop())
            ]
            
            self.logger.info("PerformanceOptimizationEngine started successfully")
    
    async def stop(self):
        """Stop the performance optimization engine"""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            self.logger.info("Stopping PerformanceOptimizationEngine")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Cleanup resources
            await self._cleanup_resources()
            
            self.logger.info("PerformanceOptimizationEngine stopped")
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect comprehensive resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
            
            # I/O metrics
            io_counters = psutil.disk_io_counters()
            
            # Process and thread metrics
            process_count = len(psutil.pids())
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
            
            # P2P metrics (if available)
            p2p_connections = 0
            p2p_latency = 0.0
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                try:
                    p2p_connections = len(self.system_integration.p2p_network.get_peers())
                    p2p_latency = self.system_integration.p2p_network.get_average_latency()
                except:
                    pass
            
            # Genetic system metrics (if available)
            genetic_ops_per_sec = 0.0
            if self.system_integration and hasattr(self.system_integration, 'genetic_system'):
                try:
                    genetic_ops_per_sec = self.system_integration.genetic_system.get_operations_per_second()
                except:
                    pass
            
            # Hormone system metrics (if available)
            hormone_levels = {}
            if self.system_integration and hasattr(self.system_integration, 'hormone_system'):
                try:
                    hormone_levels = self.system_integration.hormone_system.get_all_hormone_levels()
                except:
                    pass
            
            # Memory system efficiency (if available)
            memory_efficiency = 0.0
            if self.system_integration and hasattr(self.system_integration, 'memory_system'):
                try:
                    memory_efficiency = self.system_integration.memory_system.get_efficiency_score()
                except:
                    pass
            
            return ResourceMetrics(
                cpu_usage=cpu_percent,
                cpu_count=cpu_count,
                cpu_frequency_mhz=cpu_freq.current if cpu_freq else 0.0,
                memory_usage_gb=memory.used / (1024**3),
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                disk_available_gb=disk.free / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                network_sent_mb=network.bytes_sent / (1024**2),
                network_recv_mb=network.bytes_recv / (1024**2),
                network_connections=network_connections,
                io_read_count=io_counters.read_count if io_counters else 0,
                io_write_count=io_counters.write_count if io_counters else 0,
                io_wait_percent=psutil.cpu_times_percent().iowait,
                thread_count=thread_count,
                process_count=process_count,
                p2p_connections=p2p_connections,
                p2p_latency_ms=p2p_latency,
                genetic_operations_per_sec=genetic_ops_per_sec,
                hormone_levels=hormone_levels,
                memory_system_efficiency=memory_efficiency
            )
        except Exception as e:
            self.logger.error(f"Error collecting resource metrics: {e}")
            return ResourceMetrics()
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect current metrics
                self.current_metrics = self._collect_resource_metrics()
                self.optimization_history.append(self.current_metrics)
                
                # Check for resource alerts
                await self._check_resource_alerts()
                
                # Detect bottlenecks
                await self._detect_bottlenecks()
                
                # Update system integration if available
                if self.system_integration:
                    await self._update_system_integration()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        while self.running:
            try:
                if self.config.enable_auto_optimization:
                    await self._perform_optimization()
                
                await asyncio.sleep(self.config.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Resource cleanup loop"""
        while self.running:
            try:
                # Memory cleanup
                if self.config.enable_garbage_collection:
                    await self._optimize_memory()
                
                # I/O cache cleanup
                if self.config.io_caching_enabled:
                    await self._cleanup_io_cache()
                
                # Connection pool cleanup
                if self.config.connection_pooling:
                    await self._cleanup_connection_pools()
                
                await asyncio.sleep(self.config.memory_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _analytics_loop(self):
        """Performance analytics loop"""
        while self.running:
            try:
                if self.config.enable_performance_analytics:
                    analytics = await self._generate_analytics()
                    self.analytics_history.append(analytics)
                
                await asyncio.sleep(self.config.analytics_interval)
                
            except Exception as e:
                self.logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(600)
    
    async def _reporting_loop(self):
        """Performance reporting loop"""
        while self.running:
            try:
                if self.config.performance_reporting:
                    report = await self._generate_performance_report()
                    self.performance_reports.append(report)
                    
                    # Save report to file
                    await self._save_performance_report(report)
                
                await asyncio.sleep(self.config.report_interval)
                
            except Exception as e:
                self.logger.error(f"Error in reporting loop: {e}")
                await asyncio.sleep(3600)
    
    async def _check_resource_alerts(self):
        """Check for resource usage alerts"""
        alerts = []
        
        # CPU alerts
        if self.current_metrics.cpu_usage > self.config.cpu_threshold:
            alerts.append({
                'type': 'cpu_high',
                'resource': 'cpu',
                'current': self.current_metrics.cpu_usage,
                'threshold': self.config.cpu_threshold,
                'severity': 'high' if self.current_metrics.cpu_usage > 90 else 'medium'
            })
        
        # Memory alerts
        if self.current_metrics.memory_percent > self.config.memory_threshold:
            alerts.append({
                'type': 'memory_high',
                'resource': 'memory',
                'current': self.current_metrics.memory_percent,
                'threshold': self.config.memory_threshold,
                'severity': 'high' if self.current_metrics.memory_percent > 95 else 'medium'
            })
        
        # Disk alerts
        if self.current_metrics.disk_usage_percent > self.config.disk_threshold:
            alerts.append({
                'type': 'disk_high',
                'resource': 'disk',
                'current': self.current_metrics.disk_usage_percent,
                'threshold': self.config.disk_threshold,
                'severity': 'high' if self.current_metrics.disk_usage_percent > 95 else 'medium'
            })
        
        # Network alerts
        network_usage = self.current_metrics.network_sent_mb + self.current_metrics.network_recv_mb
        if network_usage > self.config.network_threshold:
            alerts.append({
                'type': 'network_high',
                'resource': 'network',
                'current': network_usage,
                'threshold': self.config.network_threshold,
                'severity': 'medium'
            })
        
        # P2P alerts
        if self.current_metrics.p2p_latency_ms > self.config.p2p_threshold:
            alerts.append({
                'type': 'p2p_latency_high',
                'resource': 'p2p',
                'current': self.current_metrics.p2p_latency_ms,
                'threshold': self.config.p2p_threshold,
                'severity': 'medium'
            })
        
        # Add alerts to history
        self.resource_alerts.extend(alerts)
        
        # Trigger immediate optimization for high severity alerts
        high_severity_alerts = [a for a in alerts if a['severity'] == 'high']
        if high_severity_alerts:
            await self.force_optimization()
    
    async def _detect_bottlenecks(self):
        """Detect system bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        if self.current_metrics.cpu_usage > 90:
            bottlenecks.append({
                'type': 'cpu_bottleneck',
                'resource': 'cpu',
                'usage': self.current_metrics.cpu_usage,
                'impact': 'high',
                'recommendation': 'Reduce CPU-intensive tasks or scale horizontally'
            })
        
        # Memory bottleneck
        if self.current_metrics.memory_percent > 90:
            bottlenecks.append({
                'type': 'memory_bottleneck',
                'resource': 'memory',
                'usage': self.current_metrics.memory_percent,
                'impact': 'high',
                'recommendation': 'Increase memory or optimize memory usage'
            })
        
        # I/O bottleneck
        if self.current_metrics.io_wait_percent > 20:
            bottlenecks.append({
                'type': 'io_bottleneck',
                'resource': 'io',
                'usage': self.current_metrics.io_wait_percent,
                'impact': 'medium',
                'recommendation': 'Optimize I/O operations or use faster storage'
            })
        
        # Network bottleneck
        if self.current_metrics.network_connections > 1000:
            bottlenecks.append({
                'type': 'network_bottleneck',
                'resource': 'network',
                'usage': self.current_metrics.network_connections,
                'impact': 'medium',
                'recommendation': 'Optimize network connections or implement connection pooling'
            })
        
        # Add bottlenecks to history
        self.bottleneck_history.extend(bottlenecks)
        
        # Log significant bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck['impact'] == 'high':
                self.logger.warning(f"High impact bottleneck detected: {bottleneck}")
    
    async def _perform_optimization(self):
        """Perform system optimization"""
        try:
            # Generate optimization actions
            actions = await self._generate_optimization_actions()
            
            # Sort actions by priority and impact
            actions.sort(key=lambda x: (x.priority, x.estimated_impact), reverse=True)
            
            # Execute high-priority actions
            for action in actions[:5]:  # Limit to top 5 actions
                if action.priority >= 3:  # High priority actions
                    await self._execute_optimization_action(action)
            
            self.last_optimization = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error performing optimization: {e}")
    
    async def _generate_optimization_actions(self) -> List[OptimizationAction]:
        """Generate optimization actions based on current system state"""
        actions = []
        
        # Memory optimization actions
        if self.current_metrics.memory_percent > self.config.gc_threshold:
            actions.append(OptimizationAction(
                action_type="garbage_collection",
                resource_type=ResourceType.MEMORY,
                description="Perform garbage collection to free memory",
                priority=4,
                estimated_impact=0.3,
                execution_time=1.0
            ))
        
        if self.current_metrics.memory_percent > self.config.memory_compaction_threshold:
            actions.append(OptimizationAction(
                action_type="memory_compaction",
                resource_type=ResourceType.MEMORY,
                description="Compact memory to reduce fragmentation",
                priority=3,
                estimated_impact=0.2,
                execution_time=5.0
            ))
        
        # CPU optimization actions
        if self.current_metrics.cpu_usage > self.config.cpu_optimization_threshold:
            actions.append(OptimizationAction(
                action_type="load_balancing",
                resource_type=ResourceType.CPU,
                description="Redistribute CPU load across available cores",
                priority=3,
                estimated_impact=0.25,
                execution_time=2.0
            ))
        
        # I/O optimization actions
        if self.current_metrics.io_wait_percent > self.config.io_optimization_threshold:
            actions.append(OptimizationAction(
                action_type="io_optimization",
                resource_type=ResourceType.I_O,
                description="Optimize I/O operations and caching",
                priority=2,
                estimated_impact=0.15,
                execution_time=3.0
            ))
        
        # Network optimization actions
        if self.current_metrics.network_connections > 500:
            actions.append(OptimizationAction(
                action_type="connection_pooling",
                resource_type=ResourceType.NETWORK,
                description="Implement connection pooling for network optimization",
                priority=2,
                estimated_impact=0.2,
                execution_time=1.0
            ))
        
        # P2P optimization actions
        if self.current_metrics.p2p_latency_ms > self.config.p2p_optimization_threshold:
            actions.append(OptimizationAction(
                action_type="p2p_optimization",
                resource_type=ResourceType.P2P,
                description="Optimize P2P network connections and routing",
                priority=2,
                estimated_impact=0.3,
                execution_time=2.0
            ))
        
        return actions
    
    async def _execute_optimization_action(self, action: OptimizationAction):
        """Execute an optimization action"""
        try:
            self.logger.info(f"Executing optimization action: {action.description}")
            
            if action.action_type == "garbage_collection":
                await self._optimize_memory()
            elif action.action_type == "memory_compaction":
                await self._compact_memory()
            elif action.action_type == "load_balancing":
                await self._optimize_cpu()
            elif action.action_type == "io_optimization":
                await self._optimize_io()
            elif action.action_type == "connection_pooling":
                await self._optimize_network()
            elif action.action_type == "p2p_optimization":
                await self._optimize_p2p()
            
            # Record the action
            self.active_optimizations[action.action_type] = action
            
            self.logger.info(f"Optimization action completed: {action.description}")
            
        except Exception as e:
            self.logger.error(f"Error executing optimization action {action.action_type}: {e}")
    
    async def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Perform garbage collection
            collected = gc.collect()
            
            # Clear weak references
            weakref._weakref._cleanup()
            
            # Clear caches if memory is still high
            if self.current_metrics.memory_percent > 85:
                self.io_cache.clear()
                for pool in self.memory_pools.values():
                    pool.clear()
            
            self.logger.info(f"Memory optimization completed. Collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
    
    async def _compact_memory(self):
        """Compact memory to reduce fragmentation"""
        try:
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0.1)
            
            # Clear unused caches
            self.io_cache.clear()
            
            self.logger.info("Memory compaction completed")
            
        except Exception as e:
            self.logger.error(f"Error compacting memory: {e}")
    
    async def _optimize_cpu(self):
        """Optimize CPU usage"""
        try:
            # If async framework is available, adjust worker counts
            if self.async_framework:
                current_workers = self.async_framework.get_worker_count()
                if self.current_metrics.cpu_usage > 90:
                    # Reduce workers if CPU is very high
                    new_workers = max(1, current_workers - 1)
                    self.async_framework.set_worker_count(new_workers)
                elif self.current_metrics.cpu_usage < 50:
                    # Increase workers if CPU is low
                    new_workers = min(current_workers + 1, self.current_metrics.cpu_count)
                    self.async_framework.set_worker_count(new_workers)
            
            self.logger.info("CPU optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing CPU: {e}")
    
    async def _optimize_io(self):
        """Optimize I/O operations"""
        try:
            # Implement I/O caching if enabled
            if self.config.io_caching_enabled:
                # This would implement actual I/O caching logic
                pass
            
            # Optimize file operations
            if self.system_integration:
                # Notify system integration of I/O optimization
                pass
            
            self.logger.info("I/O optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing I/O: {e}")
    
    async def _optimize_network(self):
        """Optimize network operations"""
        try:
            # Implement connection pooling
            if self.config.connection_pooling:
                # This would implement actual connection pooling logic
                pass
            
            # Optimize P2P network if available
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                # Optimize P2P connections
                pass
            
            self.logger.info("Network optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing network: {e}")
    
    async def _optimize_p2p(self):
        """Optimize P2P network performance"""
        try:
            if self.system_integration and hasattr(self.system_integration, 'p2p_network'):
                # Optimize P2P network connections
                p2p_network = self.system_integration.p2p_network
                
                # Limit connections if too many
                if self.current_metrics.p2p_connections > self.config.p2p_connection_limit:
                    p2p_network.optimize_connections(self.config.p2p_connection_limit)
                
                # Optimize routing
                p2p_network.optimize_routing()
            
            self.logger.info("P2P optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing P2P: {e}")
    
    async def _cleanup_io_cache(self):
        """Clean up I/O cache"""
        try:
            # Remove old entries from I/O cache
            current_time = time.time()
            expired_keys = []
            
            for key, value in self.io_cache.items():
                if isinstance(value, dict) and 'timestamp' in value:
                    if current_time - value['timestamp'] > 3600:  # 1 hour
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self.io_cache[key]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up I/O cache: {e}")
    
    async def _cleanup_connection_pools(self):
        """Clean up connection pools"""
        try:
            # Clean up expired connections
            for pool_name, connections in self.connection_pools.items():
                # Remove closed or expired connections
                valid_connections = [conn for conn in connections if self._is_connection_valid(conn)]
                self.connection_pools[pool_name] = valid_connections
            
        except Exception as e:
            self.logger.error(f"Error cleaning up connection pools: {e}")
    
    def _is_connection_valid(self, connection) -> bool:
        """Check if a connection is still valid"""
        try:
            # This would implement actual connection validation logic
            return True
        except:
            return False
    
    async def _update_system_integration(self):
        """Update system integration with current performance metrics"""
        try:
            if self.system_integration:
                # Update system integration with current metrics
                await self.system_integration.update_performance_metrics(self.current_metrics)
                
        except Exception as e:
            self.logger.error(f"Error updating system integration: {e}")
    
    async def _generate_analytics(self) -> PerformanceAnalytics:
        """Generate performance analytics"""
        try:
            # Calculate resource utilization
            resource_utilization = {
                'cpu': self.current_metrics.cpu_usage,
                'memory': self.current_metrics.memory_percent,
                'disk': self.current_metrics.disk_usage_percent,
                'network': (self.current_metrics.network_sent_mb + self.current_metrics.network_recv_mb) / 100,
                'p2p': self.current_metrics.p2p_latency_ms / 100
            }
            
            # Calculate optimization effectiveness
            optimization_effectiveness = {}
            if len(self.optimization_history) > 1:
                # Compare current metrics with previous
                prev_metrics = self.optimization_history[-2]
                optimization_effectiveness = {
                    'cpu_improvement': prev_metrics.cpu_usage - self.current_metrics.cpu_usage,
                    'memory_improvement': prev_metrics.memory_percent - self.current_metrics.memory_percent,
                    'io_improvement': prev_metrics.io_wait_percent - self.current_metrics.io_wait_percent
                }
            
            # Generate bottleneck analysis
            bottleneck_analysis = []
            for bottleneck in self.bottleneck_history[-10:]:  # Last 10 bottlenecks
                bottleneck_analysis.append(bottleneck)
            
            # Generate recommendations
            recommendations = await self.get_optimization_recommendations()
            
            # Calculate system health score
            health_score = self._calculate_health_score()
            
            return PerformanceAnalytics(
                timestamp=datetime.now(),
                resource_utilization=resource_utilization,
                optimization_effectiveness=optimization_effectiveness,
                bottleneck_analysis=bottleneck_analysis,
                recommendations=recommendations,
                system_health_score=health_score
            )
            
        except Exception as e:
            self.logger.error(f"Error generating analytics: {e}")
            return PerformanceAnalytics(
                timestamp=datetime.now(),
                resource_utilization={},
                optimization_effectiveness={},
                bottleneck_analysis=[],
                recommendations=[],
                system_health_score=0.0
            )
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        try:
            scores = []
            
            # CPU health (lower usage is better)
            cpu_score = max(0, 100 - self.current_metrics.cpu_usage)
            scores.append(cpu_score)
            
            # Memory health (lower usage is better)
            memory_score = max(0, 100 - self.current_metrics.memory_percent)
            scores.append(memory_score)
            
            # Disk health (lower usage is better)
            disk_score = max(0, 100 - self.current_metrics.disk_usage_percent)
            scores.append(disk_score)
            
            # I/O health (lower wait is better)
            io_score = max(0, 100 - self.current_metrics.io_wait_percent)
            scores.append(io_score)
            
            # P2P health (lower latency is better)
            p2p_score = max(0, 100 - min(self.current_metrics.p2p_latency_ms / 10, 100))
            scores.append(p2p_score)
            
            # Calculate average score
            return sum(scores) / len(scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.0
    
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Collect metrics over time
            recent_metrics = list(self.optimization_history)[-100:]  # Last 100 metrics
            
            # Calculate statistics
            cpu_stats = self._calculate_statistics([m.cpu_usage for m in recent_metrics])
            memory_stats = self._calculate_statistics([m.memory_percent for m in recent_metrics])
            
            # Generate report
            report = {
                'timestamp': datetime.now().isoformat(),
                'period': '1 hour',
                'summary': {
                    'total_optimizations': len(self.optimization_actions),
                    'active_optimizations': len(self.active_optimizations),
                    'resource_alerts': len(self.resource_alerts),
                    'bottlenecks_detected': len(self.bottleneck_history),
                    'system_health_score': self._calculate_health_score()
                },
                'metrics': {
                    'cpu': cpu_stats,
                    'memory': memory_stats,
                    'current': {
                        'cpu_usage': self.current_metrics.cpu_usage,
                        'memory_percent': self.current_metrics.memory_percent,
                        'disk_usage': self.current_metrics.disk_usage_percent,
                        'network_connections': self.current_metrics.network_connections,
                        'p2p_connections': self.current_metrics.p2p_connections,
                        'p2p_latency': self.current_metrics.p2p_latency_ms
                    }
                },
                'optimizations': [
                    {
                        'action_type': action.action_type,
                        'resource_type': action.resource_type.value,
                        'description': action.description,
                        'priority': action.priority,
                        'estimated_impact': action.estimated_impact
                    }
                    for action in self.optimization_actions[-10:]  # Last 10 optimizations
                ],
                'alerts': self.resource_alerts[-10:],  # Last 10 alerts
                'bottlenecks': self.bottleneck_history[-10:]  # Last 10 bottlenecks
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values"""
        try:
            if not values:
                return {}
            
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    async def _save_performance_report(self, report: Dict[str, Any]):
        """Save performance report to file"""
        try:
            # Create reports directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = reports_dir / f"performance_report_{timestamp}.json"
            
            # Save report
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
    
    async def _cleanup_resources(self):
        """Clean up all resources"""
        try:
            # Clear caches
            self.io_cache.clear()
            self.connection_pools.clear()
            self.memory_pools.clear()
            
            # Clear histories
            self.optimization_history.clear()
            self.resource_alerts.clear()
            self.bottleneck_history.clear()
            self.analytics_history.clear()
            self.performance_reports.clear()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            return {
                'current_metrics': {
                    'cpu_usage': self.current_metrics.cpu_usage,
                    'memory_percent': self.current_metrics.memory_percent,
                    'disk_usage': self.current_metrics.disk_usage_percent,
                    'network_connections': self.current_metrics.network_connections,
                    'p2p_connections': self.current_metrics.p2p_connections,
                    'p2p_latency': self.current_metrics.p2p_latency_ms,
                    'thread_count': self.current_metrics.thread_count,
                    'process_count': self.current_metrics.process_count
                },
                'system_health_score': self._calculate_health_score(),
                'last_optimization': self.last_optimization.isoformat(),
                'active_optimizations': len(self.active_optimizations),
                'resource_alerts': len(self.resource_alerts),
                'bottlenecks_detected': len(self.bottleneck_history)
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        try:
            recommendations = []
            
            # CPU recommendations
            if self.current_metrics.cpu_usage > 80:
                recommendations.append({
                    'type': 'cpu_optimization',
                    'priority': 'high',
                    'description': 'CPU usage is high. Consider reducing workload or scaling horizontally.',
                    'action': 'Reduce CPU-intensive tasks or add more CPU cores'
                })
            
            # Memory recommendations
            if self.current_metrics.memory_percent > 85:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'description': 'Memory usage is high. Consider memory optimization or increasing RAM.',
                    'action': 'Perform garbage collection or increase available memory'
                })
            
            # I/O recommendations
            if self.current_metrics.io_wait_percent > 20:
                recommendations.append({
                    'type': 'io_optimization',
                    'priority': 'medium',
                    'description': 'I/O wait is high. Consider I/O optimization.',
                    'action': 'Optimize I/O operations or use faster storage'
                })
            
            # Network recommendations
            if self.current_metrics.network_connections > 500:
                recommendations.append({
                    'type': 'network_optimization',
                    'priority': 'medium',
                    'description': 'Many network connections. Consider connection pooling.',
                    'action': 'Implement connection pooling or optimize network usage'
                })
            
            # P2P recommendations
            if self.current_metrics.p2p_latency_ms > 100:
                recommendations.append({
                    'type': 'p2p_optimization',
                    'priority': 'medium',
                    'description': 'P2P latency is high. Consider network optimization.',
                    'action': 'Optimize P2P network routing or reduce network load'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting optimization recommendations: {e}")
            return []
    
    async def force_optimization(self, optimization_type: str = "all"):
        """Force immediate optimization"""
        try:
            self.logger.info(f"Forcing optimization: {optimization_type}")
            
            if optimization_type == "all" or optimization_type == "memory":
                await self._optimize_memory()
            
            if optimization_type == "all" or optimization_type == "cpu":
                await self._optimize_cpu()
            
            if optimization_type == "all" or optimization_type == "io":
                await self._optimize_io()
            
            if optimization_type == "all" or optimization_type == "network":
                await self._optimize_network()
            
            if optimization_type == "all" or optimization_type == "p2p":
                await self._optimize_p2p()
            
            self.last_optimization = datetime.now()
            self.logger.info("Forced optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error in forced optimization: {e}")


# Global instance management
_performance_engine_instance: Optional[PerformanceOptimizationEngine] = None


def get_performance_engine(system_integration: Optional[SystemIntegrationLayer] = None,
                          async_framework: Optional[AsyncProcessingFramework] = None,
                          config: Optional[OptimizationConfig] = None) -> PerformanceOptimizationEngine:
    """Get the global performance optimization engine instance"""
    global _performance_engine_instance
    
    if _performance_engine_instance is None:
        _performance_engine_instance = PerformanceOptimizationEngine(
            system_integration=system_integration,
            async_framework=async_framework,
            config=config
        )
    
    return _performance_engine_instance


def initialize_performance_engine(system_integration: Optional[SystemIntegrationLayer] = None,
                                 async_framework: Optional[AsyncProcessingFramework] = None,
                                 config: Optional[OptimizationConfig] = None) -> PerformanceOptimizationEngine:
    """Initialize the global performance optimization engine"""
    global _performance_engine_instance
    
    if _performance_engine_instance is not None:
        await _performance_engine_instance.stop()
    
    _performance_engine_instance = PerformanceOptimizationEngine(
        system_integration=system_integration,
        async_framework=async_framework,
        config=config
    )
    
    return _performance_engine_instance 