#!/usr/bin/env python3
"""
MCP Core System - Unified Brain-Inspired Architecture
Comprehensive core system that orchestrates all MCP components with optimized performance.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
import json
import sqlite3
from contextlib import asynccontextmanager

# Core imports
from .lobes import LobeRegistry, BaseLobe, LobeStatus
from .memory import MemoryManager
from .workflow import WorkflowManager
from .project_manager import ProjectManager
from .task_manager import TaskManager
from .context_manager import ContextManager
from .performance_monitor import ObjectivePerformanceMonitor
from .database_manager import OptimizedDatabaseManager, DatabaseConfig

# Import neural network models if available
try:
    from .neural_network_models import check_dependencies
    from .neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
    from .neural_network_models.brain_state_integration import BrainStateIntegration
    from .neural_network_models.diffusion_model import DiffusionModel
    from .neural_network_models.genetic_diffusion_model import GeneticDiffusionModel
    from .neural_network_models.perpetual_training import PerpetualTrainingManager, TrainingConfig
    from .neural_network_models.model_factory import ModelFactory
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False

# Import advanced components
try:
    from .cognitive_architecture import CognitiveArchitecture
    from .creative_engine import CreativeEngine
    from .learning_manager import LearningManager
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ADVANCED_COMPONENTS_AVAILABLE = False

class SystemStatus(Enum):
    """System status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"
    ERROR = "error"

@dataclass
class SystemMetrics:
    """System performance metrics."""
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_lobes: int = 0
    total_requests: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    throughput: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SystemConfiguration:
    """System configuration settings."""
    max_workers: int = 4
    enable_async: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"
    data_directory: str = "data"
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    performance_optimization: bool = True
    experimental_features: bool = False
    hormone_system_enabled: bool = True
    vector_storage_enabled: bool = True
    
class MCPCoreSystem:
    """
    MCP Core System - Unified orchestration of all MCP components.
    
    Features:
    - Brain-inspired modular architecture with lobes
    - High-performance async/await support
    - Comprehensive monitoring and metrics
    - Automatic optimization and self-healing
    - Robust error handling and recovery
    - Thread-safe operations
    - Resource management and cleanup
    - Hormone system integration
    - Vector storage for advanced memory
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        """Initialize the MCP Core System."""
        self.config = config or SystemConfiguration()
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        self.metrics = SystemMetrics()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Core components
        self.lobe_registry = LobeRegistry()
        self.memory_manager = None
        self.workflow_manager = None
        self.project_manager = None
        self.task_manager = None
        self.context_manager = None
        self.performance_monitor = None
        self.db_manager = None
        
        # Neural network components
        self.hormone_integration = None
        self.brain_state_integration = None
        self.neural_dependencies = None
        self.perpetual_training_manager = None
        
        # Advanced components
        self.cognitive_architecture = None
        self.creative_engine = None
        self.learning_manager = None
        self.perpetual_training_manager = None
        
        # Advanced components
        self.cognitive_architecture = None
        self.creative_engine = None
        self.learning_manager = None
        self.model_factory = None
        
        # Async components
        self.event_loop = None
        self.executor = None
        self.process_executor = None
        
        # Monitoring and optimization
        self._monitoring_task = None
        self._optimization_task = None
        self._backup_task = None
        self._hormone_task = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Hormone system
        self.hormone_levels = {
            'stress': 0.0,      # High when system under load
            'efficiency': 1.0,   # High when system performing well
            'adaptation': 0.5,   # Drives system changes
            'stability': 1.0     # High when system is stable
        }
        
        self.logger.info("MCP Core System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger("mcp.core")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path(self.config.data_directory) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "mcp_core.log")
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """
        Initialize all core components asynchronously.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Starting MCP Core System initialization...")
            
            # Setup data directory
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize executors
            if self.config.enable_async:
                self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
                self.process_executor = ProcessPoolExecutor(max_workers=2)
            
            # Initialize database manager
            db_path = data_dir / "mcp_core.db"
            db_config = DatabaseConfig(
                max_connections=self.config.max_workers * 2,
                enable_query_cache=True,
                enable_wal_mode=True
            )
            self.db_manager = OptimizedDatabaseManager(str(db_path), db_config)
            
            # Initialize core managers
            await self._initialize_core_managers()
            
            # Initialize and register lobes
            await self._initialize_lobes()
            
            # Start monitoring and optimization
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            if self.config.performance_optimization:
                await self._start_optimization()
            
            if self.config.backup_enabled:
                await self._start_backup_system()
                
            if self.config.hormone_system_enabled:
                await self._start_hormone_system()
            
            self.status = SystemStatus.ACTIVE
            self.logger.info("MCP Core System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Core System: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def _initialize_core_managers(self):
        """Initialize core management components."""
        try:
            # Initialize managers with optimized settings
            db_path = Path(self.config.data_directory) / "mcp_core.db"
            
            self.memory_manager = MemoryManager(str(db_path))
            self.workflow_manager = WorkflowManager(str(db_path))
            self.project_manager = ProjectManager()
            self.task_manager = TaskManager(str(db_path))
            self.context_manager = ContextManager()
            
            if self.config.enable_monitoring:
                self.performance_monitor = ObjectivePerformanceMonitor(
                    enable_system_metrics=True,
                    enable_alerts=True,
                    alert_callback=self._handle_performance_alert
                )
                await self.performance_monitor.start_monitoring()
            
            # Initialize neural network components if available
            if NEURAL_MODELS_AVAILABLE and self.config.hormone_system_enabled:
                await self._initialize_neural_components()
            
            self.logger.info("Core managers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core managers: {e}")
            raise
    
    async def _initialize_neural_components(self):
        """Initialize neural network components."""
        try:
            # Check neural network dependencies
            self.neural_dependencies = check_dependencies()
            
            if not all(self.neural_dependencies.values()):
                missing = [name for name, available in self.neural_dependencies.items() if not available]
                self.logger.warning(f"Some neural network dependencies are missing: {missing}")
            
            # Initialize hormone integration
            models_dir = Path(self.config.data_directory) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load pretrained hormone integration
            from .neural_network_models.hormone_neural_integration import load_hormone_integration
            hormone_model = load_hormone_integration(str(models_dir / "hormone"))
            
            if hormone_model:
                self.hormone_integration = hormone_model
                self.logger.info("Loaded pretrained hormone integration model")
            else:
                # Create new hormone integration
                from .neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
                self.hormone_integration = HormoneNeuralIntegration()
                self.logger.info("Created new hormone integration model")
            
            # Initialize brain state integration if dependencies available
            if self.neural_dependencies.get('torch', False):
                # Try to load pretrained brain state integration
                from .neural_network_models.brain_state_integration import load_brain_state_integration
                brain_state_model = load_brain_state_integration(str(models_dir / "brain-state"))
                
                if brain_state_model:
                    self.brain_state_integration = brain_state_model
                    self.logger.info("Loaded pretrained brain state integration model")
                else:
                    # Create new brain state integration
                    from .neural_network_models.brain_state_integration import BrainStateIntegration
                    self.brain_state_integration = BrainStateIntegration()
                    self.logger.info("Created new brain state integration model")
            
            # Initialize model factory
            if NEURAL_MODELS_AVAILABLE:
                self.model_factory = ModelFactory(str(models_dir))
                self.logger.info("Neural network model factory initialized")
            
            # Initialize perpetual training manager
            if NEURAL_MODELS_AVAILABLE and self.neural_dependencies.get('torch', False):
                training_config = TrainingConfig(
                    training_interval=3600,  # Train every hour
                    enable_hormone_training=True,
                    enable_diffusion_training=True,
                    enable_brain_state_training=True,
                    output_dir=str(models_dir)
                )
                self.perpetual_training_manager = PerpetualTrainingManager(training_config)
                
                # Register models for training
                self.perpetual_training_manager.register_models(
                    hormone_model=self.hormone_integration,
                    brain_state_model=self.brain_state_integration
                )
                
                # Start background training
                self.perpetual_training_manager.start_background_training()
                self.logger.info("Perpetual training manager initialized and started")
            
            # Initialize advanced components if available
            if ADVANCED_COMPONENTS_AVAILABLE:
                await self._initialize_advanced_components()
            
            self.logger.info("Neural network components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize neural components: {e}")
            # Non-critical, continue without neural components
    
    async def _initialize_lobes(self):
        """Initialize and register all available lobes."""
        try:
            # Import and register core lobes
            from .lobes.memory_lobe import MemoryLobe
            
            # Create and register memory lobe
            memory_lobe = MemoryLobe()
            self.lobe_registry.register_lobe(memory_lobe)
            
            # Register additional lobes if available
            try:
                from .lobes.workflow_lobe import WorkflowLobe
                workflow_lobe = WorkflowLobe()
                self.lobe_registry.register_lobe(workflow_lobe)
            except ImportError:
                self.logger.debug("WorkflowLobe not available")
                
            try:
                from .lobes.context_lobe import ContextLobe
                context_lobe = ContextLobe()
                self.lobe_registry.register_lobe(context_lobe)
            except ImportError:
                self.logger.debug("ContextLobe not available")
                
            try:
                from .lobes.task_lobe import TaskLobe
                task_lobe = TaskLobe()
                self.lobe_registry.register_lobe(task_lobe)
            except ImportError:
                self.logger.debug("TaskLobe not available")
                
            # Initialize all registered lobes
            results = self.lobe_registry.initialize_all()
            
            successful_lobes = sum(1 for success in results.values() if success)
            total_lobes = len(results)
            
            self.logger.info(f"Initialized {successful_lobes}/{total_lobes} lobes successfully")
            
            if successful_lobes < total_lobes:
                failed_lobes = [name for name, success in results.items() if not success]
                self.logger.warning(f"Failed to initialize lobes: {failed_lobes}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize lobes: {e}")
            raise
    
    async def _start_monitoring(self):
        """Start system monitoring tasks."""
        try:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("System monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    async def _start_optimization(self):
        """Start performance optimization tasks."""
        try:
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            self.logger.info("Performance optimization started")
        except Exception as e:
            self.logger.error(f"Failed to start optimization: {e}")
    
    async def _start_backup_system(self):
        """Start automated backup system."""
        try:
            self._backup_task = asyncio.create_task(self._backup_loop())
            self.logger.info("Backup system started")
        except Exception as e:
            self.logger.error(f"Failed to start backup system: {e}")
            
    async def _start_hormone_system(self):
        """Start hormone system for brain-inspired regulation."""
        try:
            self._hormone_task = asyncio.create_task(self._hormone_loop())
            self.logger.info("Hormone system started")
        except Exception as e:
            self.logger.error(f"Failed to start hormone system: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_metrics()
                await self._check_system_health()
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._optimize_performance()
                await asyncio.sleep(300)  # Optimize every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _backup_loop(self):
        """Main backup loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_backup()
                await asyncio.sleep(self.config.backup_interval)
            except Exception as e:
                self.logger.error(f"Error in backup loop: {e}")
                await asyncio.sleep(self.config.backup_interval * 2)
                
    async def _hormone_loop(self):
        """Main hormone regulation loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._regulate_hormone_levels()
                await asyncio.sleep(60)  # Update hormones every minute
            except Exception as e:
                self.logger.error(f"Error in hormone loop: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _update_metrics(self):
        """Update system performance metrics."""
        try:
            import psutil
            
            # Update basic metrics
            self.metrics.uptime = (datetime.now() - self.start_time).total_seconds()
            self.metrics.memory_usage = psutil.virtual_memory().percent
            self.metrics.cpu_usage = psutil.cpu_percent()
            self.metrics.active_lobes = len([
                lobe for lobe in self.lobe_registry._lobes.values()
                if lobe.status == LobeStatus.ACTIVE
            ])
            self.metrics.last_updated = datetime.now()
            
            # Update performance monitor if available
            if self.performance_monitor:
                self.performance_monitor.record_metric("memory_usage", self.metrics.memory_usage)
                self.performance_monitor.record_metric("cpu_usage", self.metrics.cpu_usage)
                self.performance_monitor.record_metric("active_lobes", self.metrics.active_lobes)
                self.performance_monitor.record_metric("uptime", self.metrics.uptime)
                self.performance_monitor.record_metric("total_requests", self.metrics.total_requests)
                self.performance_monitor.record_metric("error_count", self.metrics.error_count)
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    async def _check_system_health(self):
        """Check overall system health and take corrective actions."""
        try:
            # Check lobe health
            lobe_health = self.lobe_registry.health_check_all()
            unhealthy_lobes = [
                name for name, health in lobe_health.items()
                if not health.get('healthy', False)
            ]
            
            if unhealthy_lobes:
                self.logger.warning(f"Unhealthy lobes detected: {unhealthy_lobes}")
                # Attempt to restart unhealthy lobes
                for lobe_name in unhealthy_lobes:
                    await self._restart_lobe(lobe_name)
            
            # Check resource usage
            if self.metrics.memory_usage > 90:
                self.logger.warning("High memory usage detected, triggering cleanup")
                await self._perform_cleanup()
            
            if self.metrics.cpu_usage > 95:
                self.logger.warning("High CPU usage detected, reducing load")
                await self._reduce_system_load()
                
            # Check database health
            if self.db_manager:
                db_stats = self.db_manager.get_performance_stats()
                if db_stats.get('slow_queries', []) and len(db_stats['slow_queries']) > 5:
                    self.logger.warning(f"Detected {len(db_stats['slow_queries'])} slow database queries")
                    # Trigger database optimization
                    self.db_manager.optimize_database()
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
    
    async def _optimize_performance(self):
        """Perform system performance optimization."""
        try:
            # Database optimization
            await self._optimize_databases()
            
            # Memory optimization
            await self._optimize_memory()
            
            # Lobe optimization
            await self._optimize_lobes()
            
            # Get recommendations from performance monitor
            if self.performance_monitor:
                recommendations = self.performance_monitor.get_performance_recommendations()
                if recommendations:
                    self.logger.info(f"Performance recommendations: {recommendations}")
                    # Apply automatic optimizations based on recommendations
                    await self._apply_performance_recommendations(recommendations)
            
            self.logger.debug("Performance optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error in performance optimization: {e}")
            
    async def _regulate_hormone_levels(self):
        """Regulate hormone levels based on system state."""
        try:
            # Get current metrics
            metrics = {
                'cpu_usage': self.metrics.cpu_usage,
                'memory_usage': self.metrics.memory_usage,
                'uptime': self.metrics.uptime,
                'total_requests': self.metrics.total_requests,
                'error_count': self.metrics.error_count,
                'active_lobes': self.metrics.active_lobes
            }
            
            # Add performance monitor metrics if available
            if self.performance_monitor:
                # Get additional metrics from performance monitor
                for metric_name in ['response_time', 'throughput', 'queue_size']:
                    value = self.performance_monitor.get_current_value(metric_name)
                    if value is not None:
                        metrics[metric_name] = value
                
                # Get hormone levels from performance monitor as fallback
                performance_hormone_levels = self.performance_monitor.get_hormone_levels()
                
                # Use as fallback if neural integration not available
                if not self.hormone_integration:
                    self.hormone_levels = performance_hormone_levels
            
            # Use neural hormone integration if available
            if self.hormone_integration:
                # Update hormone levels using neural network
                self.hormone_levels = self.hormone_integration.update_hormone_levels(metrics)
                
                # Decay feedback weights
                self.hormone_integration.decay_feedback()
            
            # Update brain state if available
            if self.brain_state_integration and NEURAL_MODELS_AVAILABLE:
                # Generate embedding from current state (simplified)
                try:
                    import torch
                    embedding_dim = self.brain_state_integration.config.embedding_dim
                    # Simple embedding from metrics
                    embedding_values = []
                    for key in sorted(metrics.keys()):
                        if isinstance(metrics[key], (int, float)):
                            embedding_values.append(float(metrics[key]))
                    
                    # Pad or truncate to embedding_dim
                    if len(embedding_values) > embedding_dim:
                        embedding_values = embedding_values[:embedding_dim]
                    else:
                        embedding_values.extend([0.0] * (embedding_dim - len(embedding_values)))
                    
                    # Create embedding tensor
                    embedding = torch.tensor(embedding_values, dtype=torch.float32)
                    
                    # Update brain state
                    self.brain_state_integration.update_brain_state(embedding, metrics)
                except Exception as e:
                    self.logger.error(f"Error updating brain state: {e}")
            
            # Apply hormone effects to system behavior
            
            # Stress hormone affects resource allocation
            if self.hormone_levels['stress'] > 0.7:
                # High stress - reduce non-essential operations
                self.logger.info("High stress detected, reducing non-essential operations")
                await self._reduce_system_load()
                
            # Efficiency hormone affects optimization frequency
            if self.hormone_levels['efficiency'] < 0.4:
                # Low efficiency - trigger immediate optimization
                self.logger.info("Low efficiency detected, triggering optimization")
                await self._optimize_performance()
                
            # Adaptation hormone affects learning rate
            adaptation_level = self.hormone_levels['adaptation']
            # Apply adaptation level to learning parameters
            if adaptation_level > 0.7:
                # High adaptation - increase learning rate
                if self.hormone_integration:
                    self.hormone_integration.provide_feedback('adaptation', 0.1)
            
            # Stability hormone affects risk tolerance
            stability_level = self.hormone_levels['stability']
            # Apply stability level to risk parameters
            if stability_level < 0.3:
                # Low stability - reduce risk
                if self.hormone_integration:
                    self.hormone_integration.provide_feedback('stability', 0.1)
            
            # Record hormone levels in database for analysis
            if self.db_manager:
                timestamp = datetime.now().isoformat()
                self.db_manager.execute_query(
                    """INSERT INTO performance_metrics 
                       (metric_name, metric_value, timestamp, component, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("hormone_levels", 0.0, timestamp, "hormone_system", 
                     json.dumps(self.hormone_levels)),
                    fetch=False
                )
                
        except Exception as e:
            self.logger.error(f"Error regulating hormone levels: {e}")
    
    async def _perform_backup(self):
        """Perform system backup."""
        try:
            backup_dir = Path(self.config.data_directory) / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"mcp_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup databases
            data_dir = Path(self.config.data_directory)
            for db_file in data_dir.glob("*.db"):
                backup_file = backup_path / db_file.name
                
                # Use SQLite backup API for safe backup
                if db_file.suffix == '.db':
                    await self._backup_sqlite_db(str(db_file), str(backup_file))
            
            # Backup configuration files
            config_dir = Path(self.config.data_directory) / "config"
            if config_dir.exists():
                import shutil
                backup_config_dir = backup_path / "config"
                backup_config_dir.mkdir(exist_ok=True)
                for config_file in config_dir.glob("*.json"):
                    shutil.copy(config_file, backup_config_dir / config_file.name)
            
            # Cleanup old backups (keep last 10)
            backups = sorted(list(backup_dir.glob("mcp_backup_*")), key=lambda x: x.stat().st_mtime)
            for old_backup in backups[:-10]:
                import shutil
                shutil.rmtree(old_backup, ignore_errors=True)
            
            self.logger.info(f"Backup completed: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error in backup: {e}")
    
    async def _backup_sqlite_db(self, source_path: str, backup_path: str):
        """Safely backup SQLite database."""
        def backup_db():
            source_conn = sqlite3.connect(source_path)
            backup_conn = sqlite3.connect(backup_path)
            source_conn.backup(backup_conn)
            source_conn.close()
            backup_conn.close()
        
        if self.executor:
            await asyncio.get_event_loop().run_in_executor(self.executor, backup_db)
        else:
            backup_db()
    
    async def _restart_lobe(self, lobe_name: str):
        """Restart a specific lobe."""
        try:
            lobe = self.lobe_registry.get_lobe(lobe_name)
            if lobe:
                self.logger.info(f"Restarting lobe: {lobe_name}")
                lobe.shutdown()
                await asyncio.sleep(1)  # Brief pause
                success = lobe.initialize()
                if success:
                    self.logger.info(f"Successfully restarted lobe: {lobe_name}")
                else:
                    self.logger.error(f"Failed to restart lobe: {lobe_name}")
        except Exception as e:
            self.logger.error(f"Error restarting lobe {lobe_name}: {e}")
    
    async def _perform_cleanup(self):
        """Perform system cleanup to free resources."""
        try:
            # Cleanup temporary files
            temp_dir = Path(self.config.data_directory) / "temp"
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir.mkdir(exist_ok=True)
            
            # Cleanup old logs
            log_dir = Path(self.config.data_directory) / "logs"
            if log_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=7)
                for log_file in log_dir.glob("*.log"):
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                        log_file.unlink(missing_ok=True)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear caches
            if self.db_manager and hasattr(self.db_manager, 'query_cache'):
                if self.db_manager.query_cache:
                    self.db_manager.query_cache.clear()
            
            # Clear context cache
            if self.context_manager and hasattr(self.context_manager, 'context_cache'):
                self.context_manager.context_cache.clear()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    async def _reduce_system_load(self):
        """Reduce system load during high CPU usage."""
        try:
            # Reduce monitoring frequency temporarily
            if self._monitoring_task:
                await asyncio.sleep(5)
            
            # Pause non-critical operations
            if self._optimization_task:
                await asyncio.sleep(10)
                
            # Increase hormone stress level
            self.hormone_levels['stress'] = min(self.hormone_levels['stress'] + 0.2, 1.0)
            
            # Reduce database cache size temporarily
            if self.db_manager and hasattr(self.db_manager, 'query_cache'):
                if self.db_manager.query_cache:
                    self.db_manager.query_cache.max_size = self.db_manager.query_cache.max_size // 2
            
            self.logger.info("System load reduction measures applied")
            
        except Exception as e:
            self.logger.error(f"Error reducing system load: {e}")
    
    async def _optimize_databases(self):
        """Optimize database performance."""
        try:
            # Use optimized database manager if available
            if self.db_manager:
                self.db_manager.optimize_database()
                return
                
            # Fallback to direct optimization
            data_dir = Path(self.config.data_directory)
            for db_file in data_dir.glob("*.db"):
                def optimize_db():
                    conn = sqlite3.connect(str(db_file))
                    conn.execute("VACUUM")
                    conn.execute("ANALYZE")
                    conn.close()
                
                if self.executor:
                    await asyncio.get_event_loop().run_in_executor(self.executor, optimize_db)
                else:
                    optimize_db()
            
        except Exception as e:
            self.logger.error(f"Error optimizing databases: {e}")
    
    async def _optimize_memory(self):
        """Optimize memory usage."""
        try:
            # Trigger garbage collection
            import gc
            gc.collect()
            
            # Optimize memory manager if available
            if self.memory_manager and hasattr(self.memory_manager, 'optimize'):
                self.memory_manager.optimize()
                
            # Prune old memories
            if self.memory_manager and hasattr(self.memory_manager, 'prune_memories'):
                pruned = self.memory_manager.prune_memories(min_priority=0.2, max_age_days=180)
                if pruned > 0:
                    self.logger.info(f"Pruned {pruned} old memories")
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
    
    async def _optimize_lobes(self):
        """Optimize lobe performance."""
        try:
            for lobe_name, lobe in self.lobe_registry._lobes.items():
                if hasattr(lobe, 'optimize'):
                    try:
                        lobe.optimize()
                    except Exception as e:
                        self.logger.error(f"Error optimizing lobe {lobe_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing lobes: {e}")
            
    async def _apply_performance_recommendations(self, recommendations: List[str]):
        """Apply automatic optimizations based on performance recommendations."""
        try:
            for recommendation in recommendations:
                if "CPU usage is very high" in recommendation:
                    await self._reduce_system_load()
                    
                elif "Memory usage is very high" in recommendation:
                    await self._perform_cleanup()
                    
                elif "Response times are high" in recommendation:
                    # Optimize database indexes
                    if self.db_manager:
                        self.db_manager.optimize_database()
                        
                elif "High error count detected" in recommendation:
                    # Reset error counters and check system health
                    self.metrics.error_count = 0
                    await self._check_system_health()
                    
        except Exception as e:
            self.logger.error(f"Error applying performance recommendations: {e}")
            
    def _handle_performance_alert(self, alert):
        """Handle performance alerts from the monitoring system."""
        try:
            self.logger.warning(f"Performance alert: {alert.message}")
            
            # Take action based on alert level and metric
            if alert.level == "critical":
                if alert.metric_name == "cpu_usage":
                    asyncio.create_task(self._reduce_system_load())
                elif alert.metric_name == "memory_usage":
                    asyncio.create_task(self._perform_cleanup())
                elif alert.metric_name == "error_count":
                    # Log detailed error information
                    self.logger.error(f"Critical error count: {alert.value}")
                    
            # Record alert in database
            if self.db_manager:
                self.db_manager.execute_query(
                    """INSERT INTO performance_metrics 
                       (metric_name, metric_value, timestamp, component, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"alert_{alert.metric_name}", alert.value, 
                     alert.timestamp.isoformat(), "performance_monitor", 
                     json.dumps({"level": alert.level.value, "message": alert.message})),
                    fetch=False
                )
                
        except Exception as e:
            self.logger.error(f"Error handling performance alert: {e}")
    
    # Public API methods
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a request through the appropriate system component.
        
        Args:
            request: Request dictionary with method and parameters
            
        Returns:
            Response dictionary
        """
        try:
            start_time = time.time()
            method = request.get('method')
            params = request.get('params', {})
            
            # Route request to appropriate component
            if method.startswith('memory/'):
                result = await self._handle_memory_request(method, params)
            elif method.startswith('workflow/'):
                result = await self._handle_workflow_request(method, params)
            elif method.startswith('task/'):
                result = await self._handle_task_request(method, params)
            elif method.startswith('project/'):
                result = await self._handle_project_request(method, params)
            elif method.startswith('context/'):
                result = await self._handle_context_request(method, params)
            elif method.startswith('system/'):
                result = await self._handle_system_request(method, params)
            elif method.startswith('hormone/'):
                result = await self._handle_hormone_request(method, params)
            else:
                result = {'error': f'Unknown method: {method}'}
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.total_requests += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time)
                / self.metrics.total_requests
            )
            
            if 'error' in result:
                self.metrics.error_count += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing request: {e}")
            self.metrics.error_count += 1
            return {'error': str(e)}
    
    async def _handle_memory_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory-related requests."""
        if not self.memory_manager:
            return {'error': 'Memory manager not available'}
        
        try:
            if method == 'memory/add':
                memory_id = self.memory_manager.add_memory(**params)
                return {'success': True, 'memory_id': memory_id}
            elif method == 'memory/search':
                results = self.memory_manager.search_memories(**params)
                return {'success': True, 'results': results}
            elif method == 'memory/get':
                memory = self.memory_manager.get_memory(params.get('memory_id'))
                return {'success': True, 'memory': memory}
            elif method == 'memory/tag':
                success = self.memory_manager.add_tag_to_memory(
                    params.get('memory_id'), params.get('tag')
                )
                return {'success': success}
            elif method == 'memory/untag':
                success = self.memory_manager.remove_tag_from_memory(
                    params.get('memory_id'), params.get('tag')
                )
                return {'success': success}
            elif method == 'memory/search_by_tag':
                results = self.memory_manager.search_memories_by_tag(params.get('tag'))
                return {'success': True, 'results': results}
            elif method == 'memory/stats':
                stats = self.memory_manager.get_statistics()
                return {'success': True, 'stats': stats}
            else:
                return {'error': f'Unknown memory method: {method}'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _handle_workflow_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow-related requests."""
        if not self.workflow_manager:
            return {'error': 'Workflow manager not available'}
        
        try:
            if method == 'workflow/status':
                status = self.workflow_manager.get_workflow_status(params.get('workflow_id'))
                return {'success': True, 'status': status}
            elif method == 'workflow/create':
                workflow_id = self.workflow_manager.create_workflow(
                    params.get('name'), 
                    params.get('description', '')
                )
                return {'success': True, 'workflow_id': workflow_id}
            elif method == 'workflow/add_step':
                success = self.workflow_manager.add_step(
                    params.get('workflow_id'),
                    params.get('step_name'),
                    params.get('description', ''),
                    params.get('dependencies', [])
                )
                return {'success': success}
            elif method == 'workflow/start':
                success = self.workflow_manager.start_workflow(params.get('workflow_id'))
                return {'success': success}
            elif method == 'workflow/pause':
                success = self.workflow_manager.pause_workflow(params.get('workflow_id'))
                return {'success': success}
            elif method == 'workflow/resume':
                success = self.workflow_manager.resume_workflow(params.get('workflow_id'))
                return {'success': success}
            else:
                return {'error': f'Unknown workflow method: {method}'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _handle_task_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task-related requests."""
        if not self.task_manager:
            return {'error': 'Task manager not available'}
        
        try:
            if method == 'task/create':
                task_id = self.task_manager.create_task(**params)
                return {'success': True, 'task_id': task_id}
            elif method == 'task/list':
                tasks = self.task_manager.get_tasks(params.get('status'))
                return {'success': True, 'tasks': tasks}
            elif method == 'task/update':
                success = self.task_manager.update_task_progress(**params)
                return {'success': success}
            elif method == 'task/complete':
                success = self.task_manager.complete_task(params.get('task_id'))
                return {'success': success}
            else:
                return {'error': f'Unknown task method: {method}'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _handle_project_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project-related requests."""
        if not self.project_manager:
            return {'error': 'Project manager not available'}
        
        try:
            if method == 'project/status':
                info = self.project_manager.get_project_info()
                return {'success': True, 'project': info}
            elif method == 'project/init':
                result = self.project_manager.init_project(**params)
                return {'success': True, 'result': result}
            else:
                return {'error': f'Unknown project method: {method}'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _handle_context_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context-related requests."""
        if not self.context_manager:
            return {'error': 'Context manager not available'}
        
        try:
            if method == 'context/export':
                result = self.context_manager.export_context(**params)
                return {'success': True, 'context': result}
            elif method == 'context/save_pack':
                pack_id = self.context_manager.save_context_pack(
                    params.get('name'),
                    params.get('context_data'),
                    params.get('description', '')
                )
                return {'success': True, 'pack_id': pack_id}
            elif method == 'context/load_pack':
                pack = self.context_manager.load_context_pack(params.get('pack_id'))
                return {'success': True, 'pack': pack}
            elif method == 'context/stats':
                stats = self.context_manager.get_context_statistics()
                return {'success': True, 'stats': stats}
            else:
                return {'error': f'Unknown context method: {method}'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _handle_system_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system-related requests."""
        try:
            if method == 'system/status':
                return {
                    'success': True,
                    'status': self.status.value,
                    'metrics': {
                        'uptime': self.metrics.uptime,
                        'memory_usage': self.metrics.memory_usage,
                        'cpu_usage': self.metrics.cpu_usage,
                        'active_lobes': self.metrics.active_lobes,
                        'total_requests': self.metrics.total_requests,
                        'error_count': self.metrics.error_count,
                        'average_response_time': self.metrics.average_response_time
                    }
                }
            elif method == 'system/health':
                lobe_health = self.lobe_registry.health_check_all()
                return {
                    'success': True,
                    'overall_health': self.status.value,
                    'lobe_health': lobe_health,
                    'metrics': self.metrics.__dict__,
                    'hormone_levels': self.hormone_levels
                }
            elif method == 'system/optimize':
                asyncio.create_task(self._optimize_performance())
                return {'success': True, 'message': 'Optimization started'}
            elif method == 'system/backup':
                asyncio.create_task(self._perform_backup())
                return {'success': True, 'message': 'Backup started'}
            elif method == 'system/cleanup':
                asyncio.create_task(self._perform_cleanup())
                return {'success': True, 'message': 'Cleanup started'}
            else:
                return {'error': f'Unknown system method: {method}'}
        except Exception as e:
            return {'error': str(e)}
            
    async def _handle_hormone_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hormone-related requests."""
        try:
            if method == 'hormone/levels':
                return {
                    'success': True,
                    'hormone_levels': self.hormone_levels
                }
            elif method == 'hormone/adjust':
                hormone_name = params.get('hormone')
                value = params.get('value')
                if hormone_name in self.hormone_levels and isinstance(value, (int, float)):
                    self.hormone_levels[hormone_name] = max(0.0, min(1.0, value))
                    return {'success': True, 'hormone_levels': self.hormone_levels}
                else:
                    return {'error': 'Invalid hormone name or value'}
            elif method == 'hormone/reset':
                self.hormone_levels = {
                    'stress': 0.0,
                    'efficiency': 1.0,
                    'adaptation': 0.5,
                    'stability': 1.0
                }
                return {'success': True, 'hormone_levels': self.hormone_levels}
            else:
                return {'error': f'Unknown hormone method: {method}'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.metrics
    
    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return self.status
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        try:
            self.logger.info("Initiating MCP Core System shutdown...")
            self.status = SystemStatus.SHUTDOWN
            
            # Signal shutdown to all tasks
            self._shutdown_event.set()
            
            # Cancel monitoring tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._optimization_task:
                self._optimization_task.cancel()
            if self._backup_task:
                self._backup_task.cancel()
            if self._hormone_task:
                self._hormone_task.cancel()
            
            # Stop performance monitor
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()
            
            # Shutdown all lobes
            self.lobe_registry.shutdown_all()
            
            # Close database manager
            if self.db_manager:
                self.db_manager.close()
            
            # Shutdown executors
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.process_executor:
                self.process_executor.shutdown(wait=True)
            
            self.logger.info("MCP Core System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global system instance
_core_system: Optional[MCPCoreSystem] = None

def get_core_system() -> Optional[MCPCoreSystem]:
    """Get the global core system instance."""
    return _core_system

async def initialize_core_system(config: Optional[SystemConfiguration] = None) -> MCPCoreSystem:
    """Initialize and return the global core system."""
    global _core_system
    if _core_system is None:
        _core_system = MCPCoreSystem(config)
        await _core_system.initialize()
    return _core_system

async def shutdown_core_system():
    """Shutdown the global core system."""
    global _core_system
    if _core_system:
        await _core_system.shutdown()
        _core_system = None