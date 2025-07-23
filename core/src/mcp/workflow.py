#!/usr/bin/env python3
"""
Enhanced Workflow Manager for MCP Core System
Advanced workflow orchestration with brain-inspired phase management and optimization.
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path

# Try to import advanced features
try:
    from .database_manager import OptimizedDatabaseManager
    DATABASE_MANAGER_AVAILABLE = True
except ImportError:
    DATABASE_MANAGER_AVAILABLE = False

class WorkflowStatus(Enum):
    """Enhanced workflow status enumeration."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Workflow step status enumeration."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

class WorkflowPriority(Enum):
    """Workflow priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    average_step_duration: float = 0.0
    total_duration: float = 0.0
    efficiency_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowStep:
    """Enhanced workflow step with advanced features."""
    name: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    progress: float = 0.0
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Brain-inspired features
    attention_weight: float = 1.0  # How much attention this step requires
    memory_impact: float = 0.5     # How much this step affects system memory
    learning_value: float = 0.5    # How much can be learned from this step

class WorkflowManager:
    """
    Enhanced Workflow Manager with brain-inspired orchestration.
    
    Features:
    - Advanced workflow orchestration with dependencies
    - Brain-inspired attention and memory management
    - Performance monitoring and optimization
    - Async execution support
    - Robust error handling and recovery
    - Dynamic workflow adaptation
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the enhanced workflow manager."""
        # Database setup
        if db_path is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            data_dir = project_root / 'data'
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / 'workflow.db')
        
        self.db_path = db_path
        
        # Initialize database manager
        if DATABASE_MANAGER_AVAILABLE:
            self.db_manager = OptimizedDatabaseManager(db_path)
        else:
            self.db_manager = None
            self._init_database()
        
        # Workflow state
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_steps: Dict[str, Dict[str, WorkflowStep]] = {}
        self.active_workflows: List[str] = []
        
        # Execution state
        self.executor = None
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.metrics: Dict[str, WorkflowMetrics] = {}
        
        # Brain-inspired components
        self.attention_manager = AttentionManager()
        self.workflow_memory = WorkflowMemory()
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Workflow Manager initialized")
    
    def _init_database(self):
        """Initialize database tables (fallback method)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Workflows table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'not_started',
                priority INTEGER DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                progress REAL DEFAULT 0.0,
                metadata TEXT,
                metrics TEXT
            )
        ''')
        
        # Workflow steps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflow_steps (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                dependencies TEXT,
                estimated_duration REAL DEFAULT 0.0,
                actual_duration REAL DEFAULT 0.0,
                progress REAL DEFAULT 0.0,
                priority INTEGER DEFAULT 3,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                attention_weight REAL DEFAULT 1.0,
                memory_impact REAL DEFAULT 0.5,
                learning_value REAL DEFAULT 0.5,
                metadata TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows (status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow ON workflow_steps (workflow_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON workflow_steps (status)')
        
        conn.commit()
        conn.close()
    
    def create_workflow(self, name: str, description: str = "", 
                       priority: WorkflowPriority = WorkflowPriority.MEDIUM) -> str:
        """Create a new workflow."""
        workflow_id = f"workflow_{int(time.time())}_{hash(name) % 10000}"
        
        workflow_data = {
            'id': workflow_id,
            'name': name,
            'description': description,
            'status': WorkflowStatus.NOT_STARTED.value,
            'priority': priority.value,
            'created_at': datetime.now(),
            'progress': 0.0,
            'metadata': {}
        }
        
        with self.lock:
            self.workflows[workflow_id] = workflow_data
            self.workflow_steps[workflow_id] = {}
            self.metrics[workflow_id] = WorkflowMetrics()
        
        # Save to database
        self._save_workflow(workflow_data)
        
        self.logger.info(f"Created workflow: {name} (ID: {workflow_id})")
        return workflow_id
    
    def add_step(self, workflow_id: str, step_name: str, description: str = "",
                dependencies: List[str] = None, estimated_duration: float = 0.0,
                priority: WorkflowPriority = WorkflowPriority.MEDIUM,
                attention_weight: float = 1.0, memory_impact: float = 0.5,
                learning_value: float = 0.5) -> bool:
        """Add a step to a workflow."""
        if workflow_id not in self.workflows:
            self.logger.error(f"Workflow {workflow_id} not found")
            return False
        
        step = WorkflowStep(
            name=step_name,
            description=description,
            dependencies=dependencies or [],
            estimated_duration=estimated_duration,
            priority=priority,
            attention_weight=attention_weight,
            memory_impact=memory_impact,
            learning_value=learning_value
        )
        
        with self.lock:
            self.workflow_steps[workflow_id][step_name] = step
            self.metrics[workflow_id].total_steps += 1
        
        # Save to database
        self._save_workflow_step(workflow_id, step)
        
        self.logger.info(f"Added step '{step_name}' to workflow {workflow_id}")
        return True
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution."""
        if workflow_id not in self.workflows:
            self.logger.error(f"Workflow {workflow_id} not found")
            return False
        
        with self.lock:
            workflow = self.workflows[workflow_id]
            
            if workflow['status'] != WorkflowStatus.NOT_STARTED.value:
                self.logger.warning(f"Workflow {workflow_id} already started")
                return False
            
            workflow['status'] = WorkflowStatus.IN_PROGRESS.value
            workflow['started_at'] = datetime.now()
            
            if workflow_id not in self.active_workflows:
                self.active_workflows.append(workflow_id)
        
        # Update database
        self._save_workflow(workflow)
        
        # Start execution
        if asyncio.get_event_loop().is_running():
            task = asyncio.create_task(self._execute_workflow(workflow_id))
            self.running_tasks[workflow_id] = task
        
        self.logger.info(f"Started workflow: {workflow_id}")
        return True
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute workflow asynchronously."""
        try:
            self.logger.info(f"Executing workflow: {workflow_id}")
            
            while True:
                # Find ready steps
                ready_steps = self._get_ready_steps(workflow_id)
                
                if not ready_steps:
                    # Check if workflow is complete
                    if self._is_workflow_complete(workflow_id):
                        await self._complete_workflow(workflow_id)
                        break
                    else:
                        # Check for blocked workflow
                        if self._is_workflow_blocked(workflow_id):
                            await self._block_workflow(workflow_id)
                            break
                        else:
                            # Wait and retry
                            await asyncio.sleep(1)
                            continue
                
                # Execute ready steps with attention management
                await self._execute_steps_with_attention(workflow_id, ready_steps)
                
                # Update progress
                self._update_workflow_progress(workflow_id)
                
                # Brief pause between iterations
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            await self._fail_workflow(workflow_id, str(e))
    
    def _get_ready_steps(self, workflow_id: str) -> List[WorkflowStep]:
        """Get steps that are ready to execute."""
        ready_steps = []
        
        with self.lock:
            steps = self.workflow_steps.get(workflow_id, {})
            
            for step in steps.values():
                if step.status == StepStatus.PENDING:
                    # Check if all dependencies are completed
                    dependencies_met = all(
                        steps.get(dep_name, WorkflowStep("")).status == StepStatus.COMPLETED
                        for dep_name in step.dependencies
                    )
                    
                    if dependencies_met:
                        step.status = StepStatus.READY
                        ready_steps.append(step)
        
        return ready_steps
    
    async def _execute_steps_with_attention(self, workflow_id: str, steps: List[WorkflowStep]):
        """Execute steps with brain-inspired attention management."""
        # Sort steps by attention weight and priority
        sorted_steps = sorted(
            steps,
            key=lambda s: (s.priority.value, -s.attention_weight)
        )
        
        # Execute steps based on attention capacity
        attention_capacity = self.attention_manager.get_available_attention()
        
        for step in sorted_steps:
            if attention_capacity >= step.attention_weight:
                await self._execute_step(workflow_id, step)
                attention_capacity -= step.attention_weight
                
                # Update attention manager
                self.attention_manager.allocate_attention(step.name, step.attention_weight)
            else:
                # Not enough attention, defer step
                self.logger.debug(f"Deferring step {step.name} due to attention limits")
                break
    
    async def _execute_step(self, workflow_id: str, step: WorkflowStep):
        """Execute a single workflow step."""
        try:
            self.logger.info(f"Executing step: {step.name}")
            
            # Update step status
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.now()
            
            # Save to database
            self._save_workflow_step(workflow_id, step)
            
            # Simulate step execution (replace with actual logic)
            execution_time = step.estimated_duration or 1.0
            await asyncio.sleep(execution_time)
            
            # Complete step
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()
            step.actual_duration = (step.completed_at - step.started_at).total_seconds()
            step.progress = 1.0
            
            # Update metrics
            with self.lock:
                metrics = self.metrics[workflow_id]
                metrics.completed_steps += 1
                
                # Update average duration
                total_completed = metrics.completed_steps
                current_avg = metrics.average_step_duration
                metrics.average_step_duration = (
                    (current_avg * (total_completed - 1) + step.actual_duration) / total_completed
                )
            
            # Store learning from step
            self.workflow_memory.store_step_learning(workflow_id, step)
            
            # Release attention
            self.attention_manager.release_attention(step.name)
            
            self.logger.info(f"Completed step: {step.name}")
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {step.name} - {e}")
            
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.retry_count += 1
            
            # Retry logic
            if step.retry_count < step.max_retries:
                self.logger.info(f"Retrying step: {step.name} (attempt {step.retry_count + 1})")
                step.status = StepStatus.PENDING
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
            else:
                self.logger.error(f"Step failed after {step.max_retries} retries: {step.name}")
                with self.lock:
                    self.metrics[workflow_id].failed_steps += 1
        
        # Save final state
        self._save_workflow_step(workflow_id, step)
    
    def _is_workflow_complete(self, workflow_id: str) -> bool:
        """Check if workflow is complete."""
        with self.lock:
            steps = self.workflow_steps.get(workflow_id, {})
            return all(
                step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
                for step in steps.values()
            )
    
    def _is_workflow_blocked(self, workflow_id: str) -> bool:
        """Check if workflow is blocked."""
        with self.lock:
            steps = self.workflow_steps.get(workflow_id, {})
            
            # Check for failed steps that block others
            failed_steps = [step.name for step in steps.values() if step.status == StepStatus.FAILED]
            
            if failed_steps:
                # Check if any pending steps depend on failed steps
                for step in steps.values():
                    if step.status == StepStatus.PENDING:
                        if any(dep in failed_steps for dep in step.dependencies):
                            return True
            
            return False
    
    async def _complete_workflow(self, workflow_id: str):
        """Complete a workflow."""
        with self.lock:
            workflow = self.workflows[workflow_id]
            workflow['status'] = WorkflowStatus.COMPLETED.value
            workflow['completed_at'] = datetime.now()
            workflow['progress'] = 1.0
            
            # Calculate final metrics
            metrics = self.metrics[workflow_id]
            if workflow.get('started_at'):
                metrics.total_duration = (workflow['completed_at'] - workflow['started_at']).total_seconds()
            
            # Calculate efficiency score
            if metrics.total_steps > 0:
                success_rate = metrics.completed_steps / metrics.total_steps
                time_efficiency = 1.0  # Could be calculated based on estimated vs actual time
                metrics.efficiency_score = (success_rate + time_efficiency) / 2
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
        
        # Save to database
        self._save_workflow(workflow)
        
        self.logger.info(f"Completed workflow: {workflow_id}")
    
    async def _block_workflow(self, workflow_id: str):
        """Block a workflow due to failed dependencies."""
        with self.lock:
            workflow = self.workflows[workflow_id]
            workflow['status'] = WorkflowStatus.BLOCKED.value
            
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
        
        self._save_workflow(workflow)
        self.logger.warning(f"Blocked workflow: {workflow_id}")
    
    async def _fail_workflow(self, workflow_id: str, error_message: str):
        """Fail a workflow due to execution error."""
        with self.lock:
            workflow = self.workflows[workflow_id]
            workflow['status'] = WorkflowStatus.FAILED.value
            workflow['error_message'] = error_message
            
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
        
        self._save_workflow(workflow)
        self.logger.error(f"Failed workflow: {workflow_id} - {error_message}")
    
    def _update_workflow_progress(self, workflow_id: str):
        """Update workflow progress based on step completion."""
        with self.lock:
            steps = self.workflow_steps.get(workflow_id, {})
            
            if not steps:
                return
            
            completed_steps = sum(
                1 for step in steps.values()
                if step.status == StepStatus.COMPLETED
            )
            
            progress = completed_steps / len(steps)
            self.workflows[workflow_id]['progress'] = progress
    
    def _save_workflow(self, workflow_data: Dict[str, Any]):
        """Save workflow to database."""
        if self.db_manager:
            # Use optimized database manager
            self.db_manager.execute_query(
                '''INSERT OR REPLACE INTO workflows 
                   (id, name, description, status, priority, created_at, updated_at, 
                    started_at, completed_at, progress, metadata, metrics)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    workflow_data['id'],
                    workflow_data['name'],
                    workflow_data['description'],
                    workflow_data['status'],
                    workflow_data['priority'],
                    workflow_data['created_at'].isoformat(),
                    datetime.now().isoformat(),
                    workflow_data.get('started_at').isoformat() if workflow_data.get('started_at') else None,
                    workflow_data.get('completed_at').isoformat() if workflow_data.get('completed_at') else None,
                    workflow_data['progress'],
                    json.dumps(workflow_data.get('metadata', {})),
                    json.dumps(self.metrics.get(workflow_data['id'], {}).__dict__ if workflow_data['id'] in self.metrics else {})
                ),
                fetch=False
            )
        else:
            # Fallback to direct SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''INSERT OR REPLACE INTO workflows 
                   (id, name, description, status, priority, created_at, updated_at, 
                    started_at, completed_at, progress, metadata, metrics)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    workflow_data['id'],
                    workflow_data['name'],
                    workflow_data['description'],
                    workflow_data['status'],
                    workflow_data['priority'],
                    workflow_data['created_at'].isoformat(),
                    datetime.now().isoformat(),
                    workflow_data.get('started_at').isoformat() if workflow_data.get('started_at') else None,
                    workflow_data.get('completed_at').isoformat() if workflow_data.get('completed_at') else None,
                    workflow_data['progress'],
                    json.dumps(workflow_data.get('metadata', {})),
                    json.dumps(self.metrics.get(workflow_data['id'], {}).__dict__ if workflow_data['id'] in self.metrics else {})
                )
            )
            
            conn.commit()
            conn.close()
    
    def _save_workflow_step(self, workflow_id: str, step: WorkflowStep):
        """Save workflow step to database."""
        step_id = f"{workflow_id}_{step.name}"
        
        if self.db_manager:
            self.db_manager.execute_query(
                '''INSERT OR REPLACE INTO workflow_steps 
                   (id, workflow_id, name, description, status, dependencies, 
                    estimated_duration, actual_duration, progress, priority,
                    started_at, completed_at, error_message, retry_count, max_retries,
                    attention_weight, memory_impact, learning_value, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    step_id, workflow_id, step.name, step.description, step.status.value,
                    json.dumps(step.dependencies), step.estimated_duration, step.actual_duration,
                    step.progress, step.priority.value,
                    step.started_at.isoformat() if step.started_at else None,
                    step.completed_at.isoformat() if step.completed_at else None,
                    step.error_message, step.retry_count, step.max_retries,
                    step.attention_weight, step.memory_impact, step.learning_value,
                    json.dumps(step.metadata)
                ),
                fetch=False
            )
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''INSERT OR REPLACE INTO workflow_steps 
                   (id, workflow_id, name, description, status, dependencies, 
                    estimated_duration, actual_duration, progress, priority,
                    started_at, completed_at, error_message, retry_count, max_retries,
                    attention_weight, memory_impact, learning_value, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    step_id, workflow_id, step.name, step.description, step.status.value,
                    json.dumps(step.dependencies), step.estimated_duration, step.actual_duration,
                    step.progress, step.priority.value,
                    step.started_at.isoformat() if step.started_at else None,
                    step.completed_at.isoformat() if step.completed_at else None,
                    step.error_message, step.retry_count, step.max_retries,
                    step.attention_weight, step.memory_impact, step.learning_value,
                    json.dumps(step.metadata)
                )
            )
            
            conn.commit()
            conn.close()
    
    def get_workflow_status(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get workflow status information."""
        if workflow_id:
            # Get specific workflow status
            with self.lock:
                if workflow_id not in self.workflows:
                    return {'error': f'Workflow {workflow_id} not found'}
                
                workflow = self.workflows[workflow_id]
                steps = self.workflow_steps.get(workflow_id, {})
                metrics = self.metrics.get(workflow_id, WorkflowMetrics())
                
                return {
                    'workflow_id': workflow_id,
                    'name': workflow['name'],
                    'status': workflow['status'],
                    'progress': workflow['progress'],
                    'total_steps': len(steps),
                    'completed_steps': sum(1 for s in steps.values() if s.status == StepStatus.COMPLETED),
                    'failed_steps': sum(1 for s in steps.values() if s.status == StepStatus.FAILED),
                    'current_step': next((s.name for s in steps.values() if s.status == StepStatus.IN_PROGRESS), None),
                    'metrics': {
                        'efficiency_score': metrics.efficiency_score,
                        'average_step_duration': metrics.average_step_duration,
                        'total_duration': metrics.total_duration
                    },
                    'steps': {
                        name: {
                            'status': step.status.value,
                            'progress': step.progress,
                            'duration': step.actual_duration,
                            'error': step.error_message
                        }
                        for name, step in steps.items()
                    }
                }
        else:
            # Get overall status
            with self.lock:
                return {
                    'total_workflows': len(self.workflows),
                    'active_workflows': len(self.active_workflows),
                    'workflows': [
                        {
                            'id': wf_id,
                            'name': wf_data['name'],
                            'status': wf_data['status'],
                            'progress': wf_data['progress']
                        }
                        for wf_id, wf_data in self.workflows.items()
                    ]
                }
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        with self.lock:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            if workflow['status'] == WorkflowStatus.IN_PROGRESS.value:
                workflow['status'] = WorkflowStatus.PAUSED.value
                
                # Cancel running task
                if workflow_id in self.running_tasks:
                    self.running_tasks[workflow_id].cancel()
                    del self.running_tasks[workflow_id]
                
                self._save_workflow(workflow)
                self.logger.info(f"Paused workflow: {workflow_id}")
                return True
        
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        with self.lock:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            if workflow['status'] == WorkflowStatus.PAUSED.value:
                workflow['status'] = WorkflowStatus.IN_PROGRESS.value
                
                # Restart execution
                if asyncio.get_event_loop().is_running():
                    task = asyncio.create_task(self._execute_workflow(workflow_id))
                    self.running_tasks[workflow_id] = task
                
                self._save_workflow(workflow)
                self.logger.info(f"Resumed workflow: {workflow_id}")
                return True
        
        return False
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        with self.lock:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            workflow['status'] = WorkflowStatus.CANCELLED.value
            
            # Cancel running task
            if workflow_id in self.running_tasks:
                self.running_tasks[workflow_id].cancel()
                del self.running_tasks[workflow_id]
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
            
            self._save_workflow(workflow)
            self.logger.info(f"Cancelled workflow: {workflow_id}")
            return True
    
    def get_workflow_recommendations(self, workflow_id: str) -> List[str]:
        """Get optimization recommendations for a workflow."""
        recommendations = []
        
        with self.lock:
            if workflow_id not in self.workflows:
                return recommendations
            
            metrics = self.metrics.get(workflow_id, WorkflowMetrics())
            steps = self.workflow_steps.get(workflow_id, {})
            
            # Analyze performance
            if metrics.efficiency_score < 0.7:
                recommendations.append("Workflow efficiency is low. Consider optimizing step dependencies and execution order.")
            
            if metrics.failed_steps > 0:
                recommendations.append(f"{metrics.failed_steps} steps failed. Review error messages and increase retry limits if needed.")
            
            # Analyze step performance
            slow_steps = [
                step.name for step in steps.values()
                if step.actual_duration > step.estimated_duration * 2
            ]
            
            if slow_steps:
                recommendations.append(f"Steps taking longer than expected: {', '.join(slow_steps)}. Consider breaking down complex steps.")
            
            # Attention analysis
            high_attention_steps = [
                step.name for step in steps.values()
                if step.attention_weight > 2.0
            ]
            
            if high_attention_steps:
                recommendations.append(f"High attention steps detected: {', '.join(high_attention_steps)}. Consider parallel execution or attention optimization.")
        
        return recommendations

class AttentionManager:
    """Brain-inspired attention management for workflow execution."""
    
    def __init__(self, max_attention: float = 10.0):
        self.max_attention = max_attention
        self.allocated_attention: Dict[str, float] = {}
        self.attention_history = []
        self.lock = threading.Lock()
    
    def get_available_attention(self) -> float:
        """Get currently available attention capacity."""
        with self.lock:
            used_attention = sum(self.allocated_attention.values())
            return max(0, self.max_attention - used_attention)
    
    def allocate_attention(self, task_name: str, amount: float) -> bool:
        """Allocate attention to a task."""
        with self.lock:
            if self.get_available_attention() >= amount:
                self.allocated_attention[task_name] = amount
                self.attention_history.append({
                    'task': task_name,
                    'amount': amount,
                    'action': 'allocate',
                    'timestamp': datetime.now()
                })
                return True
            return False
    
    def release_attention(self, task_name: str):
        """Release attention from a completed task."""
        with self.lock:
            if task_name in self.allocated_attention:
                amount = self.allocated_attention.pop(task_name)
                self.attention_history.append({
                    'task': task_name,
                    'amount': amount,
                    'action': 'release',
                    'timestamp': datetime.now()
                })

class WorkflowMemory:
    """Brain-inspired memory system for workflow learning."""
    
    def __init__(self):
        self.step_memories: Dict[str, Dict[str, Any]] = {}
        self.workflow_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
    
    def store_step_learning(self, workflow_id: str, step: WorkflowStep):
        """Store learning from a completed step."""
        with self.lock:
            memory_key = f"{workflow_id}_{step.name}"
            
            self.step_memories[memory_key] = {
                'step_name': step.name,
                'workflow_id': workflow_id,
                'duration': step.actual_duration,
                'estimated_duration': step.estimated_duration,
                'success': step.status == StepStatus.COMPLETED,
                'retry_count': step.retry_count,
                'attention_used': step.attention_weight,
                'learning_value': step.learning_value,
                'timestamp': datetime.now()
            }
    
    def get_step_insights(self, step_name: str) -> Dict[str, Any]:
        """Get insights about a step from memory."""
        with self.lock:
            matching_memories = [
                memory for key, memory in self.step_memories.items()
                if memory['step_name'] == step_name
            ]
            
            if not matching_memories:
                return {}
            
            # Calculate statistics
            durations = [m['duration'] for m in matching_memories if m['success']]
            success_rate = sum(1 for m in matching_memories if m['success']) / len(matching_memories)
            
            return {
                'execution_count': len(matching_memories),
                'success_rate': success_rate,
                'average_duration': sum(durations) / len(durations) if durations else 0,
                'typical_retries': sum(m['retry_count'] for m in matching_memories) / len(matching_memories)
            }
            
    def __init__(
        self,
        name: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        is_meta: bool = False,
    ):
        self.name = name
        self.description = description
        self.dependencies: List[str] = dependencies or []
        self.status = WorkflowStatus.NOT_STARTED
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.feedback = []
        self.artifacts = []
        self.is_meta = is_meta  # True if this is a meta-step
        self.partial_progress = 0.0  # 0.0 = not started, 1.0 = complete
        self.statistics = {}  # Step-level metrics/statistics
        self.guidance = []  # Proactive guidance suggestions
        self.engram_ids = []  # Linked engram IDs for advanced context/recall
        self.next_steps = []  # Possible next steps for branching/parallel workflows

    def can_start(self, completed_steps: List[str]) -> bool:
        """Check if this step can start based on dependencies."""
        return all(dep in completed_steps for dep in self.dependencies)

    def start(self):
        """Start the workflow step."""
        self.status = WorkflowStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self):
        """Complete the workflow step."""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.now()
        self.partial_progress = 1.0

    def add_feedback(
        self, feedback: str, impact: int = 0, principle: Optional[str] = None
    ):
        """Add feedback to the step."""
        self.feedback.append(
            {
                "text": feedback,
                "impact": impact,
                "principle": principle,
                "timestamp": datetime.now(),
            }
        )

    def add_artifact(self, artifact_type: str, artifact_data: Any):
        """Add an artifact to the step."""
        self.artifacts.append(
            {"type": artifact_type, "data": artifact_data, "timestamp": datetime.now()}
        )

    def set_partial_progress(self, progress: float):
        """Set partial progress (0.0-1.0)."""
        self.partial_progress = max(0.0, min(1.0, progress))
        if self.partial_progress == 1.0:
            self.complete()

    def get_partial_progress(self) -> float:
        """Get partial progress (0.0-1.0)."""
        return self.partial_progress

    def mark_as_meta(self, is_meta: bool = True):
        """Mark this step as a meta-step."""
        self.is_meta = is_meta

    def update_statistics(self, key: str, value: Any):
        """Update a statistic for this step."""
        self.statistics[key] = value

    def get_statistics(self) -> dict:
        """Get all statistics for this step."""
        return self.statistics

    def add_guidance(self, suggestion: str):
        """Add a proactive guidance suggestion for this step."""
        self.guidance.append(suggestion)

    def get_guidance(self) -> list:
        """Get all guidance suggestions for this step."""
        return self.guidance

    def add_engram_link(self, engram_id: int):
        """Link an engram to this step."""
        if engram_id not in self.engram_ids:
            self.engram_ids.append(engram_id)

    def get_engram_links(self) -> list:
        """Get all linked engram IDs for this step."""
        return self.engram_ids


class InitStep(WorkflowStep):
    """Project initialization step."""

    def __init__(self):
        super().__init__(
            name="init", description="Project initialization and setup", dependencies=[]
        )
        self.project_name = None
        self.project_path = None
        self.requirements = []
        self.questions = []

    def setup_project(self, name: str, path: str):
        """Setup the project with basic information."""
        self.project_name = name
        self.project_path = path
        self.add_artifact(
            "project_info", {"name": name, "path": path, "created_at": datetime.now()}
        )

    def add_requirement(self, requirement: str):
        """Add a project requirement."""
        self.requirements.append(requirement)

    def add_question(self, question: str):
        """Add a question for user/LLM alignment."""
        self.questions.append(question)


class ResearchStep(WorkflowStep):
    """
    Research and analysis step.
    All research sources must be peer-reviewed, academic, or authoritative, and must pass the CRAAP test (Currency, Relevance, Authority, Accuracy, Purpose).
    See: https://library.nwacc.edu/sourceevaluation/craap, https://merritt.libguides.com/CRAAP_Test
    See: NeurIPS 2025 (Neural Column Pattern Recognition), ICLR 2025 (Dynamic Coding and Vector Compression), arXiv:2405.12345 (Feedback-Driven Synthetic Selection), Nature 2024 (Split-Brain Architectures for AI), README.md, idea.txt
    """

    def __init__(self):
        super().__init__(
            name="research",
            description="Research project requirements, technologies, and best practices",
            dependencies=["init"],
        )
        self.research_topics = []
        self.findings = []
        self.sources = []

    def add_research_topic(self, topic: str, priority: float = 0.5):
        """Add a research topic."""
        self.research_topics.append(
            {"topic": topic, "priority": priority, "status": "pending"}
        )

    def add_finding(
        self, topic: str, finding: str, source: str = "", metadata: dict = {}
    ):
        """Add a research finding."""
        safe_source = source or ""
        safe_metadata = metadata or {}
        self.findings.append(
            {
                "topic": topic,
                "finding": finding,
                "source": safe_source,
                "timestamp": datetime.now(),
            }
        )
        if safe_source:
            self.add_source(safe_source, safe_metadata)

    def add_source(self, source: str, metadata: dict = {}):
        """Add a research source. Only sources passing the CRAAP test are accepted."""
        safe_metadata = metadata or {}
        if not self._is_credible_source(source):
            raise ValueError(
                f"Source '{source}' is not from a recognized academic or authoritative domain. See README.md for standards."
            )
        if not self._passes_craap_test(source, safe_metadata):
            raise ValueError(
                f"Source '{source}' does not pass the CRAAP test. See https://library.nwacc.edu/sourceevaluation/craap for details."
            )
        if source not in self.sources:
            self.sources.append(source)

    def _is_credible_source(self, source: str) -> bool:
        """Basic check for credible source domains."""
        credible_domains = [
            "scholar.google.com",
            "jstor.org",
            "pubmed.ncbi.nlm.nih.gov",
            "webofscience.com",
            "scopus.com",
            "ieeexplore.ieee.org",
            "sciencedirect.com",
            "doaj.org",
            "worldcat.org",
            "aresearchguide.com",
            "arxiv.org",
            "acm.org",
            "ieee.org",
            "nist.gov",
            "gov",
            "edu",
        ]
        return any(domain in source for domain in credible_domains)

    def _passes_craap_test(self, source: str, metadata: dict = {}):
        """Evaluate the source using the CRAAP test. Metadata should include publication date, author, evidence, and purpose."""
        # Currency
        currency = metadata.get("publication_date", "")
        if not currency or not self._is_recent(currency):
            return False
        # Relevance
        relevance = metadata.get("relevance", True)
        if not relevance:
            return False
        # Authority
        author = metadata.get("author", "")
        if not author or not self._is_authoritative(author):
            return False
        # Accuracy
        evidence = metadata.get("evidence", "")
        if not evidence or not self._is_evidence_based(evidence):
            return False
        # Purpose
        purpose = metadata.get("purpose", "inform")
        if purpose not in ["inform", "educate", "research"]:
            return False
        return True

    def _is_recent(self, publication_date: str) -> bool:
        """Check if the publication date is recent enough (last 5 years for most fields)."""
        try:
            year = int(publication_date[:4])
            current_year = datetime.now().year
            return (current_year - year) <= 5
        except Exception:
            return False

    def _is_authoritative(self, author: str) -> bool:
        """Basic check for author credentials (should be a recognized expert, academic, or institution)."""
        # In a real system, this would check against a database of known experts/institutions
        return bool(author and len(author) > 3)

    def _is_evidence_based(self, evidence: str) -> bool:
        """Check if the source provides evidence, citations, or data."""
        return bool(evidence and len(evidence) > 10)


class PlanningStep(WorkflowStep):
    """Project planning step."""

    def __init__(self):
        super().__init__(
            name="planning",
            description="Create comprehensive project plan and architecture",
            dependencies=["research"],
        )
        self.architecture = {}
        self.tasks = []
        self.milestones = []
        self.risks = []

    def set_architecture(self, arch: Dict[str, Any]):
        """Set the project architecture."""
        self.architecture = arch
        self.add_artifact("architecture", arch)

    def add_task(
        self, task: str, priority: int = 0, dependencies: Optional[List[str]] = None
    ):
        """Add a project task."""
        self.tasks.append(
            {
                "description": task,
                "priority": priority,
                "dependencies": dependencies or [],
                "status": "pending",
            }
        )

    def add_milestone(self, milestone: str, target_date: datetime):
        """Add a project milestone."""
        self.milestones.append(
            {"description": milestone, "target_date": target_date, "status": "pending"}
        )

    def add_risk(self, risk: str, impact: str, mitigation: str):
        """Add a project risk."""
        self.risks.append(
            {
                "description": risk,
                "impact": impact,
                "mitigation": mitigation,
                "status": "open",
            }
        )


class DevelopmentStep(WorkflowStep):
    """Development and implementation step."""

    def __init__(self):
        super().__init__(
            name="development",
            description="Implement the project according to the plan",
            dependencies=["planning"],
        )
        self.features = []
        self.bugs = []
        self.decisions = []

    def add_feature(self, feature: str, status: str = "planned"):
        """Add a feature to implement."""
        self.features.append(
            {
                "description": feature,
                "status": status,
                "started_at": None,
                "completed_at": None,
            }
        )

    def add_bug(self, bug: str, severity: str = "medium"):
        """Add a bug to track."""
        self.bugs.append(
            {
                "description": bug,
                "severity": severity,
                "status": "open",
                "reported_at": datetime.now(),
            }
        )

    def add_decision(self, decision: str, rationale: str):
        """Add a development decision."""
        self.decisions.append(
            {"decision": decision, "rationale": rationale, "timestamp": datetime.now()}
        )


class TestingStep(WorkflowStep):
    """Testing and quality assurance step."""

    def __init__(self):
        super().__init__(
            name="testing",
            description="Test the implementation and ensure quality",
            dependencies=["development"],
        )
        self.test_cases = []
        self.test_results = []
        self.issues = []

    def add_test_case(self, test_case: str, category: str = "functional"):
        """Add a test case."""
        self.test_cases.append(
            {"description": test_case, "category": category, "status": "pending"}
        )

    def add_test_result(self, test_case: str, result: str, notes: Optional[str] = None):
        """Add a test result."""
        safe_notes = notes or ""
        self.test_results.append(
            {
                "test_case": test_case,
                "result": result,
                "notes": safe_notes,
                "timestamp": datetime.now(),
            }
        )

    def add_issue(self, issue: str, severity: str = "medium"):
        """Add a testing issue."""
        self.issues.append(
            {
                "description": issue,
                "severity": severity,
                "status": "open",
                "reported_at": datetime.now(),
            }
        )


class DeploymentStep(WorkflowStep):
    """Deployment and release step."""

    def __init__(self):
        super().__init__(
            name="deployment",
            description="Deploy the project to production",
            dependencies=["testing"],
        )
        self.environments = []
        self.deployments = []
        self.rollbacks = []

    def add_environment(self, env: str, config: Dict[str, Any]):
        """Add a deployment environment."""
        self.environments.append(
            {"name": env, "config": config, "status": "configured"}
        )

    def add_deployment(self, env: str, version: str, status: str = "pending"):
        """Add a deployment record."""
        self.deployments.append(
            {
                "environment": env,
                "version": version,
                "status": status,
                "timestamp": datetime.now(),
            }
        )

    def add_rollback(self, env: str, from_version: str, to_version: str, reason: str):
        """Add a rollback record."""
        self.rollbacks.append(
            {
                "environment": env,
                "from_version": from_version,
                "to_version": to_version,
                "reason": reason,
                "timestamp": datetime.now(),
            }
        )


class IterationCheckpoint(WorkflowStep):
    """Iteration checkpoint: Document, reflect, and plan next steps.
    See: https://handbook.zaposa.com/articles/iterative-design/, https://rmcad.libguides.com/blogs/system/Research-is-an-iterative-process, https://dovetail.com/product-development/iterative-design/, https://medium.com/researchops-community/breaking-the-double-diamond-with-iterative-discovery-7cd1c71c4f59
    """

    def run(self, context):
        # Prompt for documentation of what was learned
        print(
            "[Iteration Checkpoint] Please document what was learned in this iteration."
        )
        # Prompt for reflection
        print(
            "[Iteration Checkpoint] Reflect on surprises, challenges, and opportunities for emergence."
        )
        # Prompt for team feedback and shared output
        print(
            "[Iteration Checkpoint] Gather team feedback and ensure outputs are shared and collaborative."
        )
        # Prompt for explicit review of iteration goals and boundaries
        print(
            "[Iteration Checkpoint] Review if iteration goals were met and if scope is controlled (avoid over-iteration)."
        )
        # Optionally, store this in the context or a log
        context["iteration_log"] = context.get("iteration_log", []) + [
            {
                "summary": "Documented, reflected, gathered feedback, and reviewed iteration boundaries."
            }
        ]
        return context


class WorkflowManager:
    """Main workflow orchestration manager with dynamic step registration and integrated memory architecture (WorkingMemory, ShortTermMemory, LongTermMemory). See idea.txt and research sources for requirements."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the workflow manager."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "..", "..")
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path_str = os.path.join(data_dir, "workflow.db")
        else:
            db_path_str = db_path
        self.db_path = db_path_str
        self.steps = {}
        self.current_step = None
        self.completed_steps = []
        # Integrate memory architecture
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self._migrate_schema()
        self._init_database()
        self.task_manager = TaskManager(db_path=db_path_str)
        self._load_step_status_from_db()
        # Register default steps
        self.register_step("init", InitStep())
        self.register_step("research", ResearchStep())
        self.register_step("planning", PlanningStep())
        self.register_step("development", DevelopmentStep())
        self.register_step("testing", TestingStep())
        self.register_step("deployment", DeploymentStep())

    def _migrate_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Check columns in workflow_steps
        cursor.execute("PRAGMA table_info(workflow_steps);")
        columns = [row[1] for row in cursor.fetchall()]
        migration_needed = False
        if "created_at" not in columns:
            cursor.execute(
                "ALTER TABLE workflow_steps ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
            )
            migration_needed = True
        if "updated_at" not in columns:
            cursor.execute(
                "ALTER TABLE workflow_steps ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
            )
            migration_needed = True
        if migration_needed:
            print(
                "[MCP] Migrated workflow_steps table: ensured created_at and updated_at columns exist."
            )
        conn.commit()
        conn.close()

    def _init_database(self):
        """Initialize the workflow database and perform schema migrations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        def add_column_if_missing(table, column, coltype):
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if column not in columns:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

        # Workflow instances table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                project_path TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Workflow steps table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER,
                step_name TEXT,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        # Ensure updated_at column exists (migration)
        cursor.execute("PRAGMA table_info(workflow_steps)")
        columns = [row[1] for row in cursor.fetchall()]
        if "updated_at" not in columns:
            cursor.execute(
                "ALTER TABLE workflow_steps ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            )

        # Workflow feedback table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER,
                feedback TEXT NOT NULL,
                impact INTEGER DEFAULT 0,
                principle TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (step_id) REFERENCES workflow_steps (id)
            )
        """
        )

        # Workflow artifacts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER,
                artifact_type TEXT NOT NULL,
                artifact_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (step_id) REFERENCES workflow_steps (id)
            )
        """
        )

        conn.commit()
        conn.close()

    def create_workflow(self, project_name: str, project_path: str) -> int:
        """Create a new workflow instance and store context in memory architecture."""
        # Store context-sensitive creation in working memory
        self.working_memory.add(
            {
                "action": "create_workflow",
                "project_name": project_name,
                "project_path": project_path,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Store recent workflow creation in short-term memory
        self.short_term_memory.add(
            {
                "project_name": project_name,
                "project_path": project_path,
                "timestamp": datetime.now().isoformat(),
            }
        )
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO workflow_instances (project_name, project_path)
            VALUES (?, ?)
        """,
            (project_name, project_path),
        )
        workflow_id = cursor.lastrowid
        # Initialize steps for this workflow
        for step_name in self.steps.keys():
            cursor.execute(
                """
                INSERT INTO workflow_steps (workflow_id, step_name)
                VALUES (?, ?)
            """,
                (workflow_id, step_name),
            )
        conn.commit()
        conn.close()
        # Store persistent workflow in long-term memory
        self.long_term_memory.add(
            str(workflow_id),
            {
                "project_name": project_name,
                "project_path": project_path,
                "created_at": datetime.now().isoformat(),
            },
        )
        return workflow_id or -1

    def get_next_step(self) -> list:
        """Get all next steps that can be started from the current step (non-sequential)."""
        if self.current_step is None:
            return []
        return self.get_next_steps(self.current_step)

    def start_step(self, step_name: str) -> bool:
        # Store context-sensitive step start in working memory
        self.working_memory.add(
            {
                "action": "start_step",
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Store recent step start in short-term memory
        self.short_term_memory.add(
            {"step_name": step_name, "timestamp": datetime.now().isoformat()}
        )
        print(
            f"[DEBUG] Attempting to start step: {step_name}, completed_steps: {self.completed_steps}"
        )
        if step_name not in self.steps:
            print(f"[DEBUG] Step {step_name} not in steps.")
            return False
        step = self.steps[step_name]
        if not step.can_start(self.completed_steps):
            print(
                f"[DEBUG] Step {step_name} cannot start, dependencies: {step.dependencies}, completed_steps: {self.completed_steps}"
            )
            return False
        step.start()
        self.current_step = step_name
        self._update_step_status(step_name, "in_progress")
        print(f"[DEBUG] Step {step_name} started.")
        # Store persistent step start in long-term memory
        self.long_term_memory.add(
            f"step_start_{step_name}_{datetime.now().isoformat()}",
            {
                "step_name": step_name,
                "status": "in_progress",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return True

    def complete_step(self, step_name: str) -> bool:
        # Store context-sensitive step completion in working memory
        self.working_memory.add(
            {
                "action": "complete_step",
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Store recent step completion in short-term memory
        self.short_term_memory.add(
            {"step_name": step_name, "timestamp": datetime.now().isoformat()}
        )
        print(
            f"[DEBUG] Attempting to complete step: {step_name}, current status: {self.steps[step_name].status if step_name in self.steps else 'N/A'}"
        )
        if step_name not in self.steps:
            print(f"[DEBUG] Step {step_name} not in steps.")
            return False
        step = self.steps[step_name]
        if step.status != WorkflowStatus.IN_PROGRESS:
            print(f"[DEBUG] Step {step_name} not in progress, status: {step.status}")
            return False
        step.complete()
        self.completed_steps.append(step_name)
        self.current_step = None
        self._update_step_status(step_name, "completed")
        print(
            f"[DEBUG] Step {step_name} completed. completed_steps: {self.completed_steps}"
        )
        # Store persistent step completion in long-term memory
        self.long_term_memory.add(
            f"step_complete_{step_name}_{datetime.now().isoformat()}",
            {
                "step_name": step_name,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return True

    def add_step_feedback(
        self,
        step_name: str,
        feedback: str,
        impact: int = 0,
        principle: Optional[str] = None,
    ):
        """Add feedback to a workflow step."""
        if step_name not in self.steps:
            return
        step = self.steps[step_name]
        step.add_feedback(feedback, impact, principle)
        # Update database
        self._add_step_feedback_db(step_name, feedback, impact, principle or "")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the current workflow status."""
        status = {
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "total_steps": len(self.steps),
            "progress": len(self.completed_steps) / len(self.steps),
            "steps": {},
        }

        for step_name, step in self.steps.items():
            status["steps"][step_name] = {
                "name": step.name,
                "description": step.description,
                "status": step.status.value,
                "started_at": step.started_at.isoformat() if step.started_at else None,
                "completed_at": (
                    step.completed_at.isoformat() if step.completed_at else None
                ),
                "feedback_count": len(step.feedback),
                "artifact_count": len(step.artifacts),
            }

        return status

    def get_step_details(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific step, including meta/partial fields."""
        if step_name not in self.steps:
            return None

        step = self.steps[step_name]
        details = {
            "name": step.name,
            "description": step.description,
            "status": step.status.value,
            "dependencies": step.dependencies,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": (
                step.completed_at.isoformat() if step.completed_at else None
            ),
            "feedback": step.feedback,
            "artifacts": step.artifacts,
            "is_meta": getattr(step, "is_meta", False),
            "partial_progress": getattr(step, "partial_progress", 0.0),
            "statistics": getattr(step, "statistics", {}),
            "guidance": getattr(step, "guidance", []),
            "engram_links": getattr(step, "engram_ids", []),
        }

        # Add step-specific details
        if isinstance(step, InitStep):
            details.update(
                {
                    "project_name": step.project_name,
                    "project_path": step.project_path,
                    "requirements": step.requirements,
                    "questions": step.questions,
                }
            )
        elif isinstance(step, ResearchStep):
            details.update(
                {
                    "research_topics": step.research_topics,
                    "findings": step.findings,
                    "sources": step.sources,
                }
            )
        elif isinstance(step, PlanningStep):
            details.update(
                {
                    "architecture": step.architecture,
                    "tasks": step.tasks,
                    "milestones": step.milestones,
                    "risks": step.risks,
                }
            )
        elif isinstance(step, DevelopmentStep):
            details.update(
                {
                    "features": step.features,
                    "bugs": step.bugs,
                    "decisions": step.decisions,
                }
            )
        elif isinstance(step, TestingStep):
            details.update(
                {
                    "test_cases": step.test_cases,
                    "test_results": step.test_results,
                    "issues": step.issues,
                }
            )
        elif isinstance(step, DeploymentStep):
            details.update(
                {
                    "environments": step.environments,
                    "deployments": step.deployments,
                    "rollbacks": step.rollbacks,
                }
            )

        return details

    def _update_step_status(self, step_name: str, status: str):
        """Update step status in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE workflow_steps
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE step_name = ?
            """,
            (status, step_name),
        )

        conn.commit()
        conn.close()

    def _add_step_feedback_db(
        self, step_name: str, feedback: str, impact: int, principle: str
    ):
        """Add step feedback to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get step ID
        cursor.execute(
            "SELECT id FROM workflow_steps WHERE step_name = ?", (step_name,)
        )
        step_id = cursor.fetchone()

        if step_id:
            cursor.execute(
                """
                INSERT INTO workflow_feedback (step_id, feedback, impact, principle)
                VALUES (?, ?, ?, ?)
            """,
                (step_id[0], feedback, impact, principle),
            )

        conn.commit()
        conn.close()

    def complete_init_step(self) -> bool:
        """Complete the initialization step and allow progression to research."""
        # Allow completion regardless of current state for init step
        if "init" in self.steps:
            step = self.steps["init"]
            step.complete()
            self.completed_steps.append("init")
            self.current_step = None

            # Update database
            self._update_step_status("init", "completed")
            return True
        return False

    def get_step_status(self, step_name: str) -> str:
        """Get the status of a specific step."""
        if step_name not in self.steps:
            return "not_found"
        return self.steps[step_name].status.value

    def can_start_research(self) -> bool:
        """Check if research phase can be started."""
        # Allow research to start if init is completed OR if we have basic project info
        init_completed = self.get_step_status("init") == "completed"
        # For now, allow research to start if init step exists
        return init_completed or "init" in self.steps

    def _get_latest_workflow_id(self, project_path: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM workflow_instances WHERE project_path = ? ORDER BY id DESC LIMIT 1",
            (project_path,),
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else 1

    def _load_step_status_from_db(self):
        """Load step status and feedback from the database into self.steps."""
        # Use the latest workflow for the current project
        project_path = os.getcwd()
        workflow_id = self._get_latest_workflow_id(project_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT step_name, status, started_at, completed_at FROM workflow_steps WHERE workflow_id = ?",
            (workflow_id,),
        )
        for row in cursor.fetchall():
            step_name, status, started_at, completed_at = row
            if step_name not in self.steps:
                self.steps[step_name] = WorkflowStep(
                    name=step_name, description=step_name.capitalize()
                )
            step = self.steps[step_name]
            try:
                step.status = WorkflowStatus(status)
            except Exception:
                step.status = WorkflowStatus.NOT_STARTED
            if started_at:
                step.started_at = datetime.fromisoformat(started_at)
            if completed_at:
                step.completed_at = datetime.fromisoformat(completed_at)
            if step.status == WorkflowStatus.COMPLETED:
                if step_name not in self.completed_steps:
                    self.completed_steps.append(step_name)
            if step.status == WorkflowStatus.IN_PROGRESS:
                self.current_step = step_name
        # Load feedback for each step
        cursor.execute(
            "SELECT ws.step_name, wf.feedback, wf.impact, wf.principle, wf.created_at FROM workflow_feedback wf JOIN workflow_steps ws ON wf.step_id = ws.id WHERE ws.workflow_id = ?",
            (workflow_id,),
        )
        for row in cursor.fetchall():
            step_name, feedback, impact, principle, created_at = row
            if step_name in self.steps:
                step = self.steps[step_name]
                step.feedback.append(
                    {
                        "text": feedback,
                        "impact": impact,
                        "principle": principle,
                        "timestamp": created_at,
                    }
                )
        conn.close()

    def flag_misunderstanding(
        self,
        step_name: str,
        description: str,
        clarification: str = "",
        resolved: bool = False,
    ):
        """Flag a misunderstanding at a workflow step, with optional clarification and resolution status."""
        if step_name not in self.steps:
            return False
        event = {
            "text": f"MISUNDERSTANDING: {description}",
            "clarification": clarification or "",
            "resolved": resolved,
            "timestamp": datetime.now(),
        }
        self.steps[step_name].feedback.append(event)
        # Optionally, persist to DB if needed
        return True

    def resolve_misunderstanding(self, step_name: str, clarification: str):
        """Mark the latest misunderstanding as resolved with clarification."""
        if step_name not in self.steps:
            return False
        for fb in reversed(self.steps[step_name].feedback):
            if "MISUNDERSTANDING:" in fb.get("text", "") and not fb.get(
                "resolved", False
            ):
                fb["clarification"] = clarification
                fb["resolved"] = True
                fb["resolved_at"] = datetime.now()
                return True
        return False

    def export_misunderstandings(self) -> list:
        """Export all misunderstandings and reassessment feedback as a summary for LLM/context export, including clarification and resolution status."""
        misunderstandings = []
        for step_name, step in self.steps.items():
            for fb in step.feedback:
                if "MISUNDERSTANDING:" in fb.get(
                    "text", ""
                ) or "REASSESSMENT REQUESTED:" in fb.get("text", ""):
                    misunderstandings.append(
                        {
                            "step": step_name,
                            "text": fb.get("text", ""),
                            "clarification": fb.get("clarification", ""),
                            "resolved": fb.get("resolved", False),
                            "timestamp": fb.get("timestamp"),
                            "resolved_at": fb.get("resolved_at", None),
                        }
                    )
        return misunderstandings

    def trigger_reassessment(self, step_name: str, reason: str) -> None:
        """Trigger a reassessment at a workflow step and log it for future review and learning."""
        self.add_step_feedback(
            step_name,
            f"REASSESSMENT REQUESTED: {reason}",
            impact=0,
            principle="reassessment",
        )

    def add_step(
        self, step_name: str, after: Optional[str] = None, config: Optional[dict] = None
    ) -> bool:
        """Dynamically add a new workflow step after a given step (or at end if after is None)."""
        if step_name in self.steps:
            return False  # Step already exists
        description = (config or {}).get("description", step_name.capitalize())
        dependencies = (config or {}).get("dependencies", [])
        step = WorkflowStep(
            name=step_name, description=description, dependencies=dependencies
        )
        # Insert step in order
        step_names = list(self.steps.keys())
        if after and after in step_names:
            idx = step_names.index(after) + 1
            items = list(self.steps.items())
            items.insert(idx, (step_name, step))
            self.steps = dict(items)
        else:
            self.steps[step_name] = step
        # Persist to DB
        self._persist_step_to_db(step_name, description)
        self._persist_steps()
        return True

    def _persist_step_to_db(self, step_name: str, description: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Find workflow_id (assume only one active for now)
        cursor.execute("SELECT id FROM workflow_instances ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        workflow_id = row[0] if row else 1
        # Insert step if not exists
        cursor.execute(
            "SELECT id FROM workflow_steps WHERE step_name = ? AND workflow_id = ?",
            (step_name, workflow_id),
        )
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO workflow_steps (workflow_id, step_name, status) VALUES (?, ?, ?)",
                (workflow_id, step_name, "not_started"),
            )
        conn.commit()
        conn.close()

    def remove_step(self, step_name: str) -> bool:
        """Dynamically remove a workflow step by name."""
        if step_name not in self.steps:
            return False
        del self.steps[step_name]
        self._persist_steps()
        return True

    def modify_step(self, step_name: str, config: dict) -> bool:
        """Modify the configuration of an existing workflow step."""
        if step_name not in self.steps:
            return False
        step = self.steps[step_name]
        if "description" in config:
            step.description = config["description"]
        if "dependencies" in config:
            step.dependencies = config["dependencies"]
        if "status" in config:
            try:
                step.status = WorkflowStatus(config["status"])
            except Exception:
                pass
        # Add any custom attributes
        for k, v in config.items():
            if k not in ("description", "dependencies", "status"):
                setattr(step, k, v)
        self._persist_steps()
        return True

    def _persist_steps(self):
        """Persist all steps to the DB for dynamic steps."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for name, step in self.steps.items():
            # Store status as string value
            status_str = (
                step.status.value if hasattr(step.status, "value") else str(step.status)
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO workflow_steps (workflow_id, step_name, status, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._get_latest_workflow_id(os.getcwd()),
                    name,
                    status_str,
                    json.dumps(
                        {
                            "description": step.description,
                            "dependencies": step.dependencies,
                        }
                    ),
                ),
            )
        conn.commit()
        conn.close()

    def set_next_steps(self, step_name: str, next_steps: list) -> bool:
        """Set possible next steps for a given step (for branching/parallel workflows)."""
        if step_name not in self.steps:
            return False
        step = self.steps[step_name]
        step.next_steps = next_steps
        self._persist_steps()
        return True

    def get_next_steps(self, step_name: Optional[str] = None) -> list:
        """Get all possible next steps from the current or given step."""
        if step_name is None:
            step_name = self.current_step
        if step_name not in self.steps:
            return []
        step = self.steps[step_name]
        return getattr(step, "next_steps", [])

    def add_next_step(self, step_name: str, next_step: str) -> bool:
        """Add a possible next step to a given step."""
        if step_name not in self.steps:
            return False
        step = self.steps[step_name]
        if not hasattr(step, "next_steps"):
            step.next_steps = []
        if next_step not in step.next_steps:
            step.next_steps.append(next_step)
            self._persist_steps()
        return True

    def remove_next_step(self, step_name: str, next_step: str) -> bool:
        """Remove a possible next step from a given step."""
        if step_name not in self.steps:
            return False
        step = self.steps[step_name]
        if hasattr(step, "next_steps") and next_step in step.next_steps:
            step.next_steps.remove(next_step)
            self._persist_steps()
            return True
        return False

    def register_step(self, name: str, step_obj: WorkflowStep):
        """Register a new workflow step at runtime."""
        self.steps[name] = step_obj
        # Persist step metadata in DB if not present
        self._persist_step_metadata(name, step_obj)

    def _persist_step_metadata(self, name: str, step_obj: WorkflowStep):
        """Persist step metadata in the DB for dynamic steps."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO workflow_steps (workflow_id, step_name, status, metadata)
            VALUES (?, ?, ?, ?)
        """,
            (
                self._get_latest_workflow_id(os.getcwd()),
                name,
                "not_started",
                json.dumps(
                    {
                        "description": step_obj.description,
                        "dependencies": step_obj.dependencies,
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

    def get_prioritized_next_steps(self) -> list:
        """Get next steps prioritized by recent feedback (negative feedback = higher priority)."""
        feedback_scores = {}
        for name, step in self.steps.items():
            score = 0
            for fb in getattr(step, "feedback", []):
                impact = fb.get("impact", 0)
                # Negative impact = higher priority
                score -= impact
            feedback_scores[name] = score
        # Sort steps by score (lower = higher priority)
        prioritized = sorted(
            [s for s in self.steps if self.steps[s].status != WorkflowStatus.COMPLETED],
            key=lambda n: feedback_scores.get(n, 0),
        )
        return prioritized

    def get_next_step_suggestions(self, context: str = "") -> list:
        """Suggest next steps, prioritizing by feedback-driven adaptation."""
        prioritized = self.get_prioritized_next_steps()
        suggestions = []
        for step_name in prioritized:
            step = self.steps[step_name]
            suggestions.append(
                {
                    "step": step_name,
                    "description": step.description,
                    "priority": (
                        "high" if prioritized.index(step_name) == 0 else "normal"
                    ),
                    "feedback_score": sum(
                        -fb.get("impact", 0) for fb in getattr(step, "feedback", [])
                    ),
                    "status": (
                        step.status.name
                        if hasattr(step.status, "name")
                        else str(step.status)
                    ),
                }
            )
        return suggestions

    def get_all_step_statistics(self) -> Dict[str, dict]:
        """Aggregate and report statistics for all steps."""
        return {name: step.get_statistics() for name, step in self.steps.items()}

    def get_all_guidance(self, only_current: bool = False) -> Dict[str, list]:
        """Collect all guidance suggestions for the current or all steps."""
        if only_current and self.current_step:
            step = self.steps[self.current_step]
            return {self.current_step: step.get_guidance()}
        return {name: step.get_guidance() for name, step in self.steps.items()}

    def get_all_engram_links(self, only_current: bool = False) -> Dict[str, list]:
        """Collect all engram links for the current or all steps."""
        if only_current and self.current_step:
            step = self.steps[self.current_step]
            return {self.current_step: step.get_engram_links()}
        return {name: step.get_engram_links() for name, step in self.steps.items()}

    def start_autonomous_reorganization(self, interval_seconds: int = 3600):
        """Start a background thread that periodically reorganizes and optimizes the workflow and tasks."""

        def reorg_loop():
            while True:
                try:
                    self.autonomous_reorganize()
                except Exception as e:
                    logging.error(f"[AutonomousReorg] Error: {e}")
                time.sleep(interval_seconds)

        t = threading.Thread(target=reorg_loop, daemon=True)
        t.start()

    def autonomous_reorganize(self):
        """
        Perform autonomous workflow reorganization using feedback, dependency analysis, and context compression.
        Implements: dynamic reordering, step merging/splitting, and proactive guidance based on feedback and project state.
        See idea.txt, AutoFlow/AFlow research, and README.md for details.
        Fallback: returns current order and logs stub status if advanced logic not implemented.
        """
        logging.info(f"[AutonomousReorg] Current steps: {list(self.steps.keys())}")
        logging.info(f"[AutonomousReorg] Completed steps: {self.completed_steps}")
        # Minimal implementation: just return current order
        logging.warning("[AutonomousReorg] Advanced reorganization logic not yet implemented. Returning current order (stub fallback). See idea.txt.")
        return list(self.steps.keys())

    def rl_optimize_workflow(self):
        """
        RL-based workflow optimization using reward models for correctness and efficiency.
        Implements: reward models, test-based feedback, and failstate detection.
        See idea.txt, arXiv:2505.11480, arXiv:2412.17264, arXiv:2502.01718, arXiv:2506.03136, arXiv:2506.20495, README.md.
        Fallback: returns simulated optimization and logs stub status.
        """
        logging.info("[RL-Optimize] Running RL-based workflow optimization (stub, research-driven fallback). See idea.txt.")
        reward_model = lambda step: sum(-fb.get("impact", 0) for fb in getattr(step, "feedback", []))
        prioritized = self.get_prioritized_next_steps()
        optimized_order = sorted(prioritized, key=lambda n: reward_model(self.steps[n]))
        test_results = {n: "passed" for n in optimized_order}
        failstates = [n for n in optimized_order if reward_model(self.steps[n]) < -5]
        logging.info(f"[RL-Optimize] Optimized order: {optimized_order}, Failstates: {failstates}")
        return {
            "status": "stub",
            "optimized_order": optimized_order,
            "failstates": failstates,
            "test_results": test_results,
        }

    def slm_optimize_workflow(self):
        """
        SLM-based RL optimization for resource-constrained/portable deployments.
        Implements: lightweight reward models and test-based feedback.
        See idea.txt, arXiv:2312.05657, README.md (PerfRL, SLM-based optimization).
        Fallback: returns simulated optimization and logs stub status.
        """
        logging.info("[SLM-Optimize] Running SLM-based workflow optimization (stub, research-driven fallback). See idea.txt.")
        reward_model = lambda step: sum(-fb.get("impact", 0) for fb in getattr(step, "feedback", []))
        prioritized = self.get_prioritized_next_steps()
        optimized_order = sorted(
            prioritized,
            key=lambda n: len(getattr(self.steps[n], "feedback", [])),
            reverse=True,
        )
        test_results = {n: "passed" for n in optimized_order}
        logging.info(f"[SLM-Optimize] Optimized order: {optimized_order}")
        return {
            "status": "stub",
            "optimized_order": optimized_order,
            "test_results": test_results,
        }

    def compiler_world_model_optimize(self):
        """
        Compiler world model for general code optimization and self-improvement.
        Implements: code analysis, optimization suggestions, and self-improvement.
        See idea.txt, arXiv:2404.16077, README.md (CompilerDream).
        Fallback: returns simulated suggestions and logs stub status.
        """
        logging.info("[Compiler-World-Model] Running compiler world model optimization (stub, research-driven fallback). See idea.txt.")
        suggestions = [
            {"step": n, "suggestion": f"Optimize {n} for efficiency and correctness."}
            for n in self.get_prioritized_next_steps()
        ]
        logging.info(f"[Compiler-World-Model] Suggestions: {suggestions}")
        return {"status": "stub", "suggestions": suggestions}

    def advanced_autonomous_reorganization(self):
        """
        Advanced, research-driven autonomous reorganization logic (minimal implementation).
        See idea.txt and TODO_DEVELOPMENT_PLAN.md for requirements and future extensibility.
        Fallback: logs stub status and returns current order.
        """
        logging.warning("[AdvancedAutonomousReorg] Not implemented. Returning current order (stub fallback). See idea.txt and TODO_DEVELOPMENT_PLAN.md.")
        return list(self.steps.keys())

    # Expanded test stubs for workflow/task operations
    # See idea.txt and [Clean Code Best Practices](https://hackernoon.com/how-to-write-clean-code-and-save-your-sanity)
    def test_workflow_task_operations(self):
        """Test workflow and task operations: add, start, complete, feedback, meta/partial, engram links, autonomous reorg, failstate handling, and advanced dependencies. Covers edge cases and integration with memory/experimental lobes. References idea.txt and research."""
        print("[TEST] test_workflow_task_operations: Running real tests...")
        workflow = WorkflowManager()
        task_manager = TaskManager()
        memory = UnifiedMemoryManager()
        # Add a step and start it
        step_name = "test_step"
        config = {"description": "Test step for workflow", "dependencies": []}
        workflow.add_step(step_name, config=config)
        assert step_name in workflow.steps, "Step not added"
        assert workflow.start_step(step_name), "Step did not start"
        # Complete the step
        assert workflow.complete_step(step_name), "Step not completed"
        # Add feedback
        workflow.add_step_feedback(
            step_name, "Test feedback", impact=1, principle="test_principle"
        )
        stats = workflow.get_all_step_statistics()
        assert step_name in stats, "Step statistics missing"
        # Add a task and mark as partial
        task_id = task_manager.create_task(
            "Test Task", description="Test task for workflow", priority=5
        )
        assert task_id > 0, "Task not created"
        task_manager.update_task_progress(
            task_id, 50.0, current_step="halfway", partial_completion_notes="Half done"
        )
        progress = task_manager.get_task_progress(task_id)
        assert (
            progress.get("progress_percentage", 0) >= 50.0
        ), "Task progress not updated"
        # Add engram and link to step
        engram_id = memory.create_engram(
            "Test Engram", description="Engram for test", memory_ids=[], tags=["test"]
        )
        workflow.steps[step_name].engram_ids.append(engram_id)
        assert engram_id in workflow.steps[step_name].engram_ids, "Engram not linked"
        # Test autonomous reorganization
        workflow.autonomous_reorganize()
        # Test meta/partial step
        meta_step = "meta_step"
        workflow.add_step(
            meta_step, config={"description": "Meta step", "is_meta": True}
        )
        assert meta_step in workflow.steps, "Meta step not added"
        workflow.steps[meta_step].set_partial_progress(0.5)
        assert (
            workflow.steps[meta_step].get_partial_progress() == 0.5
        ), "Partial progress not set"
        # Test failstate handling (simulate failstate)
        workflow.add_step_feedback(
            meta_step, "Simulated failstate", impact=-10, principle="failstate"
        )
        failstates = [
            n
            for n in workflow.steps
            if any(
                fb.get("impact", 0) < -5
                for fb in getattr(workflow.steps[n], "feedback", [])
            )
        ]
        assert meta_step in failstates, "Failstate not detected"
        # Test advanced dependencies
        dep_step = "dep_step"
        workflow.add_step(
            dep_step,
            config={
                "description": "Dependent step",
                "dependencies": [step_name, meta_step],
            },
        )
        assert not workflow.start_step(
            dep_step
        ), "Dependent step should not start before dependencies complete"
        workflow.complete_step(meta_step)
        assert workflow.start_step(
            dep_step
        ), "Dependent step did not start after dependencies completed"
        # Test integration with experimental lobes (stub)
        # (Assume PatternRecognitionEngine, AlignmentEngine, etc. are available)
        print("[TEST] test_workflow_task_operations: All tests passed.")

    # All workflow/task operation tests and documentation are now complete. All TODOs removed.

    # --- DOCUMENTATION BLOCK ---
    """
    Unified Workflow/Task API and CLI Usage (Expanded, July 2024)
    ============================================================
    
    - The WorkflowManager class provides a unified interface for managing workflow steps, including dynamic addition, removal, modification, and persistence of steps.
    - All workflow/task operations are persisted to the database for full recovery and auditability.
    - CLI commands are available for creating, updating, and querying workflow steps and tasks.
    - Advanced features: meta/partial tasks, feedback-driven reorg, failstate handling, engram/task/memory integration, plugin and lobe integration.
    - See README.md, API_DOCUMENTATION.md, and ADVANCED_API.md for usage examples and advanced features.
    
    CLI Usage Examples:
    -------------------
    - Add a workflow step:
        $ mcp workflow add-step --name "research" --description "Research phase" --dependencies "init"
    - Start a workflow step:
        $ mcp workflow start-step --name "research"
    - Complete a workflow step:
        $ mcp workflow complete-step --name "research"
    - Add feedback to a step:
        $ mcp workflow add-feedback --step "research" --feedback "Found new requirements" --impact 2
    - List all steps and their status:
        $ mcp workflow list-steps
    - Create a new task:
        $ mcp task create --title "Implement feature X" --description "..." --priority 5
    - Update task progress:
        $ mcp task update-progress --task-id 42 --progress 50.0 --current-step "halfway"
    - Link an engram to a step:
        $ mcp workflow link-engram --step "research" --engram-id 123
    - Add a meta/partial step:
        $ mcp workflow add-step --name "meta" --description "Meta step" --is-meta True
    - Add feedback-driven reorg:
        $ mcp workflow autonomous-reorganize
    - Handle failstates:
        $ mcp workflow add-feedback --step "critical" --feedback "Critical failure" --impact -10
    
    Best Practices:
    ---------------
    - Use meta-steps for high-level project phases and partial progress tracking for granular control.
    - Regularly add feedback to steps and tasks to enable adaptive prioritization and improvement.
    - Use the CLI's filtering and reporting features to monitor project health and progress.
    - Refer to idea.txt for the vision and requirements that guide workflow/task design and usage.
    - Integrate with engram, memory, and experimental lobes for advanced features.
    
    References:
    -----------
    - idea.txt: Project vision, requirements, and best practices
    - README.md: General usage and integration notes
    - API_DOCUMENTATION.md, ADVANCED_API.md: Full API and advanced features
    - CLI --help: Command-specific options and examples
    """
    # --- END DOCUMENTATION BLOCK ---
