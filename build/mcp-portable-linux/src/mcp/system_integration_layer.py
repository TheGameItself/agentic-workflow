"""
System Integration Layer with Backward Compatibility

This module provides a comprehensive integration layer that coordinates all MCP systems
while maintaining backward compatibility with existing implementations.

Features:
- Unified interface for all system components
- Backward compatibility with legacy systems
- Genetic trigger system integration
- P2P network coordination
- Hormone system management
- Memory system coordination
- Performance monitoring and optimization
- Async processing framework
- Resource management and allocation
"""

import asyncio
import logging
import time
import threading
import os
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
import uuid
from datetime import datetime, timedelta

# Core system imports
from .memory import MemoryManager
from .workflow import WorkflowManager
from .project_manager import ProjectManager
from .task_manager import TaskManager
from .context_manager import ContextManager
from .unified_memory import UnifiedMemoryManager
from .rag_system import RAGSystem
from .performance_monitor import ObjectivePerformanceMonitor
from .reminder_engine import EnhancedReminderEngine
from .hypothetical_engine import HypotheticalEngine
from .dreaming_engine import DreamingEngine
from .engram_engine import EngramEngine
from .auto_management_daemon import AutoManagementDaemon

# Genetic system imports
try:
    from .genetic_trigger_system.integrated_genetic_system import IntegratedGeneticSystem
    from .genetic_trigger_system.environmental_state import EnvironmentalState
    from .genetic_data_exchange import GeneticDataExchange
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    IntegratedGeneticSystem = None
    EnvironmentalState = None
    GeneticDataExchange = None

# P2P system imports
try:
    from .p2p_network import P2PNetwork
    from .integrated_p2p_genetic_system import IntegratedP2PGeneticSystem
    from .p2p_papal_integration import P2PPapalIntegration
    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False
    P2PNetwork = None
    IntegratedP2PGeneticSystem = None
    P2PPapalIntegration = None

# Hormone system imports
try:
    from .hormone_system_integration import HormoneSystemIntegration
    from .hormone_system_controller import HormoneSystemController
    HORMONE_AVAILABLE = True
except ImportError:
    HORMONE_AVAILABLE = False
    HormoneSystemIntegration = None
    HormoneSystemController = None

# Vector memory imports
try:
    from .enhanced_vector_memory import EnhancedVectorMemorySystem
    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    VECTOR_MEMORY_AVAILABLE = False
    EnhancedVectorMemorySystem = None

# Monitoring imports
try:
    from .monitoring_system_visualization_integration import MonitoringSystemWithVisualization
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    MonitoringSystemWithVisualization = None


class SystemComponent(Enum):
    """Available system components"""
    MEMORY = "memory"
    WORKFLOW = "workflow"
    PROJECT = "project"
    TASK = "task"
    CONTEXT = "context"
    UNIFIED_MEMORY = "unified_memory"
    RAG = "rag"
    PERFORMANCE = "performance"
    REMINDER = "reminder"
    HYPOTHETICAL = "hypothetical"
    DREAMING = "dreaming"
    ENGRAM = "engram"
    AUTO_MANAGEMENT = "auto_management"
    GENETIC = "genetic"
    P2P = "p2p"
    HORMONE = "hormone"
    VECTOR_MEMORY = "vector_memory"
    MONITORING = "monitoring"


@dataclass
class IntegrationConfig:
    """Configuration for system integration"""
    # Core system settings
    enable_genetic_system: bool = True
    enable_p2p_network: bool = True
    enable_hormone_system: bool = True
    enable_vector_memory: bool = True
    enable_monitoring: bool = True
    
    # Performance settings
    max_async_workers: int = 8
    async_timeout: float = 30.0
    health_check_interval: float = 60.0
    
    # Resource limits
    max_memory_gb: float = 10.0
    max_cpu_percent: float = 80.0
    
    # Backward compatibility
    legacy_mode: bool = False
    compatibility_layer: bool = True
    
    # Integration settings
    cross_system_communication: bool = True
    unified_state_management: bool = True
    performance_optimization: bool = True


@dataclass
class SystemState:
    """Current state of all integrated systems"""
    timestamp: datetime = field(default_factory=datetime.now)
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    health_status: Dict[str, str] = field(default_factory=dict)
    integration_status: Dict[str, bool] = field(default_factory=dict)


class SystemIntegrationLayer:
    """
    Comprehensive system integration layer with backward compatibility.
    
    This class coordinates all MCP systems while maintaining compatibility
    with existing implementations and providing a unified interface.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None, project_path: Optional[str] = None):
        """Initialize the system integration layer"""
        self.config = config or IntegrationConfig()
        self.project_path = project_path or os.getcwd()
        self.logger = logging.getLogger("SystemIntegrationLayer")
        
        # Core system components
        self.core_components: Dict[str, Any] = {}
        self.legacy_components: Dict[str, Any] = {}
        
        # Integration state
        self.state = SystemState()
        self.running = False
        self._lock = threading.RLock()
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_async_workers)
        self.async_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance monitoring
        self.performance_history: List[Dict[str, Any]] = []
        self.last_optimization = datetime.now()
        
        # Initialize components
        self._initialize_core_components()
        self._initialize_advanced_components()
        self._setup_integration_hooks()
        
        self.logger.info("SystemIntegrationLayer initialized successfully")
    
    def _initialize_core_components(self):
        """Initialize core system components"""
        try:
            # Core memory and workflow systems
            self.core_components[SystemComponent.MEMORY.value] = MemoryManager()
            self.core_components[SystemComponent.WORKFLOW.value] = WorkflowManager()
            self.core_components[SystemComponent.PROJECT.value] = ProjectManager()
            self.core_components[SystemComponent.TASK.value] = TaskManager()
            self.core_components[SystemComponent.CONTEXT.value] = ContextManager()
            self.core_components[SystemComponent.UNIFIED_MEMORY.value] = UnifiedMemoryManager()
            self.core_components[SystemComponent.RAG.value] = RAGSystem()
            self.core_components[SystemComponent.PERFORMANCE.value] = ObjectivePerformanceMonitor(self.project_path)
            self.core_components[SystemComponent.REMINDER.value] = EnhancedReminderEngine()
            self.core_components[SystemComponent.HYPOTHETICAL.value] = HypotheticalEngine(
                self.core_components[SystemComponent.MEMORY.value],
                self.core_components[SystemComponent.UNIFIED_MEMORY.value]
            )
            self.core_components[SystemComponent.DREAMING.value] = DreamingEngine()
            self.core_components[SystemComponent.ENGRAM.value] = EngramEngine(
                memory_manager=self.core_components[SystemComponent.MEMORY.value]
            )
            self.core_components[SystemComponent.AUTO_MANAGEMENT.value] = AutoManagementDaemon(
                self.core_components[SystemComponent.WORKFLOW.value],
                self.core_components[SystemComponent.TASK.value],
                self.core_components[SystemComponent.PERFORMANCE.value],
                self.logger
            )
            
            self.logger.info("Core components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing core components: {e}")
            raise
    
    def _initialize_advanced_components(self):
        """Initialize advanced system components with fallbacks"""
        # Genetic system
        if self.config.enable_genetic_system and GENETIC_AVAILABLE:
            try:
                self.core_components[SystemComponent.GENETIC.value] = IntegratedGeneticSystem()
                self.state.integration_status[SystemComponent.GENETIC.value] = True
                self.logger.info("Genetic system integrated successfully")
            except Exception as e:
                self.logger.warning(f"Genetic system integration failed: {e}")
                self.state.integration_status[SystemComponent.GENETIC.value] = False
        
        # P2P network
        if self.config.enable_p2p_network and P2P_AVAILABLE:
            try:
                organism_id = str(uuid.uuid4())
                self.core_components[SystemComponent.P2P.value] = P2PNetwork(organism_id)
                self.state.integration_status[SystemComponent.P2P.value] = True
                self.logger.info("P2P network integrated successfully")
            except Exception as e:
                self.logger.warning(f"P2P network integration failed: {e}")
                self.state.integration_status[SystemComponent.P2P.value] = False
        
        # Hormone system
        if self.config.enable_hormone_system and HORMONE_AVAILABLE:
            try:
                self.core_components[SystemComponent.HORMONE.value] = HormoneSystemIntegration()
                self.state.integration_status[SystemComponent.HORMONE.value] = True
                self.logger.info("Hormone system integrated successfully")
            except Exception as e:
                self.logger.warning(f"Hormone system integration failed: {e}")
                self.state.integration_status[SystemComponent.HORMONE.value] = False
        
        # Vector memory
        if self.config.enable_vector_memory and VECTOR_MEMORY_AVAILABLE:
            try:
                self.core_components[SystemComponent.VECTOR_MEMORY.value] = EnhancedVectorMemorySystem()
                self.state.integration_status[SystemComponent.VECTOR_MEMORY.value] = True
                self.logger.info("Vector memory system integrated successfully")
            except Exception as e:
                self.logger.warning(f"Vector memory integration failed: {e}")
                self.state.integration_status[SystemComponent.VECTOR_MEMORY.value] = False
        
        # Monitoring system
        if self.config.enable_monitoring and MONITORING_AVAILABLE:
            try:
                self.core_components[SystemComponent.MONITORING.value] = MonitoringSystemWithVisualization()
                self.state.integration_status[SystemComponent.MONITORING.value] = True
                self.logger.info("Monitoring system integrated successfully")
            except Exception as e:
                self.logger.warning(f"Monitoring system integration failed: {e}")
                self.state.integration_status[SystemComponent.MONITORING.value] = False
    
    def _setup_integration_hooks(self):
        """Set up cross-system integration hooks"""
        if not self.config.cross_system_communication:
            return
        
        # Set up hormone system integration if available
        if SystemComponent.HORMONE.value in self.core_components:
            hormone_system = self.core_components[SystemComponent.HORMONE.value]
            
            # Connect hormone system to other components
            for component_name, component in self.core_components.items():
                if hasattr(component, 'hormone_system'):
                    component.hormone_system = hormone_system
                if hasattr(component, 'set_hormone_system'):
                    component.set_hormone_system(hormone_system)
        
        # Set up genetic system integration if available
        if SystemComponent.GENETIC.value in self.core_components:
            genetic_system = self.core_components[SystemComponent.GENETIC.value]
            
            # Connect genetic system to memory and performance components
            if SystemComponent.MEMORY.value in self.core_components:
                memory_manager = self.core_components[SystemComponent.MEMORY.value]
                if hasattr(memory_manager, 'genetic_system'):
                    memory_manager.genetic_system = genetic_system
            
            if SystemComponent.PERFORMANCE.value in self.core_components:
                performance_monitor = self.core_components[SystemComponent.PERFORMANCE.value]
                if hasattr(performance_monitor, 'genetic_system'):
                    performance_monitor.genetic_system = genetic_system
        
        # Set up P2P network integration if available
        if SystemComponent.P2P.value in self.core_components:
            p2p_network = self.core_components[SystemComponent.P2P.value]
            
            # Connect P2P network to genetic and memory systems
            if SystemComponent.GENETIC.value in self.core_components:
                genetic_system = self.core_components[SystemComponent.GENETIC.value]
                if hasattr(genetic_system, 'p2p_network'):
                    genetic_system.p2p_network = p2p_network
        
        self.logger.info("Integration hooks configured successfully")
    
    async def start(self):
        """Start the system integration layer"""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self.logger.info("Starting SystemIntegrationLayer")
            
            # Start core components
            for component_name, component in self.core_components.items():
                try:
                    if hasattr(component, 'start') and callable(getattr(component, 'start')):
                        if asyncio.iscoroutinefunction(component.start):
                            await component.start()
                        else:
                            component.start()
                    self.state.health_status[component_name] = "running"
                except Exception as e:
                    self.logger.error(f"Error starting {component_name}: {e}")
                    self.state.health_status[component_name] = "error"
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("SystemIntegrationLayer started successfully")
    
    async def stop(self):
        """Stop the system integration layer"""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            self.logger.info("Stopping SystemIntegrationLayer")
            
            # Stop all async tasks
            for task_name, task in self.async_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Stop core components
            for component_name, component in self.core_components.items():
                try:
                    if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                        if asyncio.iscoroutinefunction(component.stop):
                            await component.stop()
                        else:
                            component.stop()
                    self.state.health_status[component_name] = "stopped"
                except Exception as e:
                    self.logger.error(f"Error stopping {component_name}: {e}")
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("SystemIntegrationLayer stopped successfully")
    
    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        self.async_tasks['health_monitor'] = health_task
        
        # Performance optimization task
        if self.config.performance_optimization:
            optimization_task = asyncio.create_task(self._performance_optimization_loop())
            self.async_tasks['performance_optimization'] = optimization_task
        
        # State synchronization task
        if self.config.unified_state_management:
            sync_task = asyncio.create_task(self._state_synchronization_loop())
            self.async_tasks['state_sync'] = sync_task
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.running:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    async def _performance_optimization_loop(self):
        """Background performance optimization loop"""
        while self.running:
            try:
                await self._optimize_system_performance()
                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _state_synchronization_loop(self):
        """Background state synchronization loop"""
        while self.running:
            try:
                await self._synchronize_system_state()
                await asyncio.sleep(30)  # Every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in state synchronization loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_system_health(self):
        """Check health of all system components"""
        current_time = datetime.now()
        
        for component_name, component in self.core_components.items():
            try:
                # Check if component is responsive
                if hasattr(component, 'get_status'):
                    status = component.get_status()
                    self.state.health_status[component_name] = status.get('status', 'unknown')
                elif hasattr(component, 'is_healthy'):
                    is_healthy = component.is_healthy()
                    self.state.health_status[component_name] = "healthy" if is_healthy else "unhealthy"
                else:
                    self.state.health_status[component_name] = "unknown"
                
                # Update component state
                self.state.components[component_name] = {
                    'last_check': current_time,
                    'status': self.state.health_status[component_name]
                }
                
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                self.state.health_status[component_name] = "error"
        
        # Update overall state
        self.state.timestamp = current_time
    
    async def _optimize_system_performance(self):
        """Optimize system performance based on current metrics"""
        try:
            # Get current performance metrics
            metrics = await self.get_performance_metrics()
            
            # Check if optimization is needed
            if metrics.get('cpu_usage', 0) > self.config.max_cpu_percent:
                await self._optimize_cpu_usage()
            
            if metrics.get('memory_usage_gb', 0) > self.config.max_memory_gb:
                await self._optimize_memory_usage()
            
            # Update optimization history
            self.last_optimization = datetime.now()
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics,
                'optimization_applied': True
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
    
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        # Reduce async workers if CPU usage is high
        if self.executor._max_workers > 2:
            new_workers = max(2, self.executor._max_workers - 1)
            self.executor._max_workers = new_workers
            self.logger.info(f"Reduced async workers to {new_workers} due to high CPU usage")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # Trigger memory cleanup in components
        for component_name, component in self.core_components.items():
            try:
                if hasattr(component, 'optimize_memory'):
                    if asyncio.iscoroutinefunction(component.optimize_memory):
                        await component.optimize_memory()
                    else:
                        component.optimize_memory()
                elif hasattr(component, 'cleanup'):
                    if asyncio.iscoroutinefunction(component.cleanup):
                        await component.cleanup()
                    else:
                        component.cleanup()
            except Exception as e:
                self.logger.error(f"Memory optimization failed for {component_name}: {e}")
    
    async def _synchronize_system_state(self):
        """Synchronize state across all system components"""
        current_time = datetime.now()
        
        # Update resource usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.state.resource_usage = {
                'cpu_percent': process.cpu_percent(),
                'memory_usage_gb': memory_info.rss / (1024**3),
                'memory_percent': process.memory_percent(),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
        
        # Update performance metrics
        if SystemComponent.PERFORMANCE.value in self.core_components:
            try:
                performance_monitor = self.core_components[SystemComponent.PERFORMANCE.value]
                if hasattr(performance_monitor, 'get_performance_summary'):
                    summary = performance_monitor.get_performance_summary()
                    self.state.performance_metrics.update(summary)
            except Exception as e:
                self.logger.error(f"Failed to get performance metrics: {e}")
        
        # Update state timestamp
        self.state.timestamp = current_time
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a specific system component"""
        return self.core_components.get(component_name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all system components"""
        return self.core_components.copy()
    
    async def get_system_state(self) -> SystemState:
        """Get current system state"""
        await self._synchronize_system_state()
        return self.state
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        metrics = {}
        
        # Get resource usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            metrics.update({
                'cpu_usage': process.cpu_percent(),
                'memory_usage_gb': memory_info.rss / (1024**3),
                'memory_percent': process.memory_percent(),
                'threads': process.num_threads()
            })
        except Exception as e:
            self.logger.error(f"Failed to get resource metrics: {e}")
        
        # Get component-specific metrics
        if SystemComponent.PERFORMANCE.value in self.core_components:
            try:
                performance_monitor = self.core_components[SystemComponent.PERFORMANCE.value]
                if hasattr(performance_monitor, 'get_performance_summary'):
                    summary = performance_monitor.get_performance_summary()
                    metrics.update(summary)
            except Exception as e:
                self.logger.error(f"Failed to get performance metrics: {e}")
        
        return metrics
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function synchronously"""
        return func(*args, **kwargs)
    
    async def call_component_method(self, component_name: str, method_name: str, *args, **kwargs) -> Any:
        """Call a method on a specific component"""
        component = self.get_component(component_name)
        if not component:
            raise ValueError(f"Component {component_name} not found")
        
        if not hasattr(component, method_name):
            raise ValueError(f"Method {method_name} not found on component {component_name}")
        
        method = getattr(component, method_name)
        
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return await self.execute_async(method, *args, **kwargs)
    
    # Backward compatibility methods
    def get_memory_manager(self) -> MemoryManager:
        """Get memory manager (backward compatibility)"""
        return self.core_components[SystemComponent.MEMORY.value]
    
    def get_workflow_manager(self) -> WorkflowManager:
        """Get workflow manager (backward compatibility)"""
        return self.core_components[SystemComponent.WORKFLOW.value]
    
    def get_project_manager(self) -> ProjectManager:
        """Get project manager (backward compatibility)"""
        return self.core_components[SystemComponent.PROJECT.value]
    
    def get_task_manager(self) -> TaskManager:
        """Get task manager (backward compatibility)"""
        return self.core_components[SystemComponent.TASK.value]
    
    def get_context_manager(self) -> ContextManager:
        """Get context manager (backward compatibility)"""
        return self.core_components[SystemComponent.CONTEXT.value]
    
    def get_unified_memory(self) -> UnifiedMemoryManager:
        """Get unified memory manager (backward compatibility)"""
        return self.core_components[SystemComponent.UNIFIED_MEMORY.value]
    
    def get_rag_system(self) -> RAGSystem:
        """Get RAG system (backward compatibility)"""
        return self.core_components[SystemComponent.RAG.value]
    
    def get_performance_monitor(self) -> ObjectivePerformanceMonitor:
        """Get performance monitor (backward compatibility)"""
        return self.core_components[SystemComponent.PERFORMANCE.value]
    
    def get_reminder_engine(self) -> EnhancedReminderEngine:
        """Get reminder engine (backward compatibility)"""
        return self.core_components[SystemComponent.REMINDER.value]
    
    def get_hypothetical_engine(self) -> HypotheticalEngine:
        """Get hypothetical engine (backward compatibility)"""
        return self.core_components[SystemComponent.HYPOTHETICAL.value]
    
    def get_dreaming_engine(self) -> DreamingEngine:
        """Get dreaming engine (backward compatibility)"""
        return self.core_components[SystemComponent.DREAMING.value]
    
    def get_engram_engine(self) -> EngramEngine:
        """Get engram engine (backward compatibility)"""
        return self.core_components[SystemComponent.ENGRAM.value]
    
    def get_auto_management_daemon(self) -> AutoManagementDaemon:
        """Get auto management daemon (backward compatibility)"""
        return self.core_components[SystemComponent.AUTO_MANAGEMENT.value]
    
    # Advanced component getters
    def get_genetic_system(self) -> Optional[Any]:
        """Get genetic system if available"""
        return self.core_components.get(SystemComponent.GENETIC.value)
    
    def get_p2p_network(self) -> Optional[Any]:
        """Get P2P network if available"""
        return self.core_components.get(SystemComponent.P2P.value)
    
    def get_hormone_system(self) -> Optional[Any]:
        """Get hormone system if available"""
        return self.core_components.get(SystemComponent.HORMONE.value)
    
    def get_vector_memory(self) -> Optional[Any]:
        """Get vector memory system if available"""
        return self.core_components.get(SystemComponent.VECTOR_MEMORY.value)
    
    def get_monitoring_system(self) -> Optional[Any]:
        """Get monitoring system if available"""
        return self.core_components.get(SystemComponent.MONITORING.value)
    
    def is_component_available(self, component_name: str) -> bool:
        """Check if a component is available and healthy"""
        if component_name not in self.core_components:
            return False
        
        health_status = self.state.health_status.get(component_name, "unknown")
        return health_status in ["running", "healthy"]
    
    def get_integration_status(self) -> Dict[str, bool]:
        """Get integration status of all components"""
        return self.state.integration_status.copy()
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status of all components"""
        return self.state.health_status.copy()
    
    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component"""
        try:
            component = self.get_component(component_name)
            if not component:
                return False
            
            # Stop component
            if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                if asyncio.iscoroutinefunction(component.stop):
                    await component.stop()
                else:
                    component.stop()
            
            # Reinitialize component
            self._initialize_single_component(component_name)
            
            # Start component
            if hasattr(component, 'start') and callable(getattr(component, 'start')):
                if asyncio.iscoroutinefunction(component.start):
                    await component.start()
                else:
                    component.start()
            
            self.state.health_status[component_name] = "running"
            self.logger.info(f"Component {component_name} restarted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart component {component_name}: {e}")
            self.state.health_status[component_name] = "error"
            return False
    
    def _initialize_single_component(self, component_name: str):
        """Initialize a single component"""
        # This would need to be implemented based on the specific component
        # For now, we'll just log that it's not implemented
        self.logger.warning(f"Single component initialization not implemented for {component_name}")


# Global instance for easy access
_system_integration_layer: Optional[SystemIntegrationLayer] = None


def get_system_integration_layer(config: IntegrationConfig = None) -> SystemIntegrationLayer:
    """Get the global system integration layer instance"""
    global _system_integration_layer
    if _system_integration_layer is None:
        _system_integration_layer = SystemIntegrationLayer(config)
    return _system_integration_layer


def initialize_system_integration(config: IntegrationConfig = None) -> SystemIntegrationLayer:
    """Initialize the global system integration layer"""
    global _system_integration_layer
    if _system_integration_layer is not None:
        # Stop existing instance
        asyncio.create_task(_system_integration_layer.stop())
    
    _system_integration_layer = SystemIntegrationLayer(config)
    return _system_integration_layer 