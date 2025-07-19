"""
Asynchronous Processing Framework

This module provides a comprehensive asynchronous processing framework for the MCP system,
including task scheduling, priority queues, resource management, and distributed processing.

Features:
- Advanced task scheduling with priority queues
- Resource-aware task execution
- Distributed processing across multiple workers
- Task dependency management
- Progress tracking and monitoring
- Error handling and recovery
- Performance optimization
- Load balancing
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import uuid
from datetime import datetime, timedelta
import weakref
from collections import defaultdict
import traceback


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskType(Enum):
    """Task types for different processing requirements"""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    MIXED = "mixed"


@dataclass
class Task:
    """Represents an asynchronous task"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    task_type: TaskType = TaskType.MIXED
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Configuration for the async processing framework"""
    # Worker configuration
    max_workers: int = 8
    max_process_workers: int = 4
    max_thread_workers: int = 16
    
    # Queue configuration
    max_queue_size: int = 1000
    priority_queue_enabled: bool = True
    
    # Resource limits
    max_memory_gb: float = 10.0
    max_cpu_percent: float = 80.0
    
    # Timeout and retry settings
    default_timeout: float = 300.0  # 5 minutes
    default_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance settings
    enable_load_balancing: bool = True
    enable_resource_monitoring: bool = True
    enable_progress_tracking: bool = True
    
    # Monitoring settings
    health_check_interval: float = 30.0
    metrics_collection_interval: float = 60.0


class PriorityTaskQueue:
    """Priority queue for task scheduling"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {
            priority: queue.PriorityQueue(maxsize=max_size)
            for priority in TaskPriority
        }
        self._lock = threading.RLock()
        self._size = 0
    
    def put(self, task: Task) -> bool:
        """Add a task to the appropriate priority queue"""
        with self._lock:
            if self._size >= self.max_size:
                return False
            
            # Use negative priority for correct ordering (lower number = higher priority)
            priority_value = -task.priority.value
            self.queues[task.priority].put((priority_value, task.created_at.timestamp(), task))
            self._size += 1
            return True
    
    def get(self) -> Optional[Task]:
        """Get the highest priority task"""
        with self._lock:
            for priority in TaskPriority:
                try:
                    _, _, task = self.queues[priority].get_nowait()
                    self._size -= 1
                    return task
                except queue.Empty:
                    continue
            return None
    
    def size(self) -> int:
        """Get current queue size"""
        return self._size
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._size == 0


class ResourceMonitor:
    """Monitor system resources for load balancing"""
    
    def __init__(self):
        self.logger = logging.getLogger("ResourceMonitor")
        self.last_check = datetime.now()
        self.resource_history = []
        
    async def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            import psutil
            process = psutil.Process()
            
            resources = {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_usage_gb': process.memory_info().rss / (1024**3),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
            
            # Add to history
            self.resource_history.append({
                'timestamp': datetime.now(),
                'resources': resources
            })
            
            # Keep only recent history
            if len(self.resource_history) > 100:
                self.resource_history.pop(0)
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Error getting system resources: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_usage_gb': 0.0,
                'threads': 0,
                'open_files': 0,
                'connections': 0
            }
    
    def is_system_overloaded(self, config: ProcessingConfig) -> bool:
        """Check if system is overloaded"""
        if not self.resource_history:
            return False
        
        latest = self.resource_history[-1]['resources']
        return (
            latest['cpu_percent'] > config.max_cpu_percent or
            latest['memory_usage_gb'] > config.max_memory_gb
        )
    
    def get_resource_trend(self) -> Dict[str, float]:
        """Get resource usage trend over time"""
        if len(self.resource_history) < 2:
            return {}
        
        recent = self.resource_history[-10:]  # Last 10 measurements
        if not recent:
            return {}
        
        trends = {}
        for key in ['cpu_percent', 'memory_percent', 'memory_usage_gb']:
            values = [entry['resources'][key] for entry in recent]
            if values:
                trends[key] = sum(values) / len(values)
        
        return trends


class AsyncProcessingFramework:
    """
    Comprehensive asynchronous processing framework for MCP system.
    
    Features:
    - Priority-based task scheduling
    - Resource-aware execution
    - Distributed processing
    - Task dependency management
    - Progress tracking
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the async processing framework"""
        self.config = config if config is not None else ProcessingConfig()
        self.logger = logging.getLogger("AsyncProcessingFramework")
        
        # Task management
        self.task_queue = PriorityTaskQueue(self.config.max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_thread_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_process_workers)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Performance tracking
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0,
            'queue_wait_time': 0.0
        }
        
        # Framework state
        self.running = False
        self._lock = threading.RLock()
        self._task_counter = 0
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info("AsyncProcessingFramework initialized")
    
    async def start(self):
        """Start the async processing framework"""
        with self._lock:
            if self.running:
                return
            
            self.running = True
            self.logger.info("Starting AsyncProcessingFramework")
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._task_scheduler_loop()),
                asyncio.create_task(self._resource_monitor_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            self.logger.info("AsyncProcessingFramework started successfully")
    
    async def stop(self):
        """Stop the async processing framework"""
        with self._lock:
            if not self.running:
                return
            
            self.running = False
            self.logger.info("Stopping AsyncProcessingFramework")
            
            # Cancel all running tasks
            for task_id, task in self.running_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown executors
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            self.logger.info("AsyncProcessingFramework stopped successfully")
    
    async def submit_task(self, 
                         func: Callable, 
                         *args, 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         task_type: TaskType = TaskType.MIXED,
                         timeout: Optional[float] = None,
                         retries: int = None,
                         dependencies: List[str] = None,
                         metadata: Dict[str, Any] = None,
                         **kwargs) -> str:
        """Submit a task for asynchronous execution"""
        # Check if framework is running
        if not self.running:
            raise RuntimeError("AsyncProcessingFramework is not running. Call start() first.")
        
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            task_type=task_type,
            timeout=timeout if timeout is not None else self.config.default_timeout,
            retries=retries if retries is not None else self.config.default_retries,
            dependencies=dependencies if dependencies is not None else [],
            metadata=metadata if metadata is not None else {}
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to queue if no dependencies
        if not task.dependencies:
            if not self.task_queue.put(task):
                task.status = TaskStatus.FAILED
                task.error = Exception("Task queue is full")
                self.logger.error(f"Task queue is full, cannot submit task {task_id}")
                return task_id
        
        self.logger.info(f"Submitted task {task_id} with priority {priority.value}")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a completed task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Wait for task completion
        start_time = time.time()
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.FAILED:
            raise task.error or Exception(f"Task {task_id} failed")
        
        return task.result
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.RUNNING and task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            task.status = TaskStatus.CANCELLED
            self.logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task"""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].status
    
    def get_task_progress(self, task_id: str) -> Optional[float]:
        """Get the progress of a task (0.0 to 1.0)"""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].progress
    
    async def _task_scheduler_loop(self):
        """Main task scheduling loop"""
        while self.running:
            try:
                # Check for available tasks
                task = self.task_queue.get()
                if task:
                    # Check dependencies
                    if await self._check_dependencies(task):
                        # Execute task
                        asyncio.create_task(self._execute_task(task))
                    else:
                        # Re-queue task if dependencies not met
                        self.task_queue.put(task)
                
                # Check for resource constraints
                if self.config.enable_load_balancing:
                    if self.resource_monitor.is_system_overloaded(self.config):
                        await asyncio.sleep(1.0)  # Slow down if overloaded
                        continue
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in task scheduler loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if all dependencies for a task are completed"""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _execute_task(self, task: Task):
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Create asyncio task
            asyncio_task = asyncio.create_task(self._run_task_with_timeout(task))
            self.running_tasks[task.task_id] = asyncio_task
            
            # Wait for completion
            await asyncio_task
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = e
            
            # Retry if possible
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.error = None
                await asyncio.sleep(self.config.retry_delay)
                self.task_queue.put(task)
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retries})")
            else:
                self.performance_metrics['tasks_failed'] += 1
        
        finally:
            # Clean up
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    async def _run_task_with_timeout(self, task: Task):
        """Run a task with timeout handling"""
        try:
            # Choose executor based on task type
            if task.task_type == TaskType.CPU_INTENSIVE:
                # Use process executor for CPU-intensive tasks
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_executor, 
                    task.func, 
                    *task.args, 
                    **task.kwargs
                )
            else:
                # Use thread executor for other tasks
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_executor, 
                    task.func, 
                    *task.args, 
                    **task.kwargs
                )
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            task.progress = 1.0
            
            # Update metrics
            if task.started_at is not None and task.completed_at is not None:
                processing_time = (task.completed_at - task.started_at).total_seconds()
            else:
                processing_time = 0.0
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['total_processing_time'] += processing_time
            self.performance_metrics['average_task_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['tasks_completed']
            )
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout} seconds")
            self.logger.error(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            self.logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    async def _resource_monitor_loop(self):
        """Resource monitoring loop"""
        while self.running:
            try:
                await self.resource_monitor.get_system_resources()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _cleanup_loop(self):
        """Cleanup completed tasks"""
        while self.running:
            try:
                # Remove old completed tasks
                current_time = datetime.now()
                tasks_to_remove = []
                
                for task_id, task in self.tasks.items():
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        # Keep tasks for 1 hour
                        task_time = task.completed_at if task.completed_at is not None else task.created_at
                        if (current_time - task_time) > timedelta(hours=1):
                            tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                
                if tasks_to_remove:
                    self.logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        metrics.update({
            'queue_size': self.task_queue.size,
            'running_tasks': len(self.running_tasks),
            'total_tasks': len(self.tasks),
            'resource_trend': self.resource_monitor.get_resource_trend()
        })
        return metrics
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks"""
        summary = {
            'total': len(self.tasks),
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'timeout': 0
        }
        
        for task in self.tasks.values():
            summary[task.status.value] += 1
        
        return summary
    
    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending and running tasks to complete"""
        start_time = time.time()
        
        while True:
            # Check if all tasks are done
            pending_or_running = sum(
                1 for task in self.tasks.values()
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
            )
            
            if pending_or_running == 0:
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            await asyncio.sleep(0.1)


# Global instance for easy access
_async_framework: Optional[AsyncProcessingFramework] = None


def get_async_framework(config: Optional[ProcessingConfig] = None) -> AsyncProcessingFramework:
    """Get the global async processing framework instance"""
    global _async_framework
    if _async_framework is None:
        _async_framework = AsyncProcessingFramework(config)
    return _async_framework


def initialize_async_framework(config: Optional[ProcessingConfig] = None) -> AsyncProcessingFramework:
    """Initialize the global async processing framework"""
    global _async_framework
    if _async_framework is not None:
        # Stop existing instance
        asyncio.create_task(_async_framework.stop())
    
    _async_framework = AsyncProcessingFramework(config)
    return _async_framework 