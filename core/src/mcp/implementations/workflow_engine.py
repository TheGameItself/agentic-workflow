"""
Workflow Engine Implementation for the MCP system.

This module provides a basic implementation of the IWorkflowEngine interface.

Δworkflow_engine(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging
import jsont datetime
import asyncio
from enum import Enum

from core.src.mcp.interfaces.workflow_engine import IWorkflowEngine
from core.src.mcp.exceptions import MCPWorkflowError

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Enum for workflow status values."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(Enum):
    """Enum for task status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BasicWorkflowEngine(IWorkflowEngine):
    """Basic implementation of the IWorkflowEngine interface."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the workflow engine."""
        self.workflows = {}  # In-memory storage for workflows
        self.tasks = {}  # In-memory storage for tasks
        self.storage_path = storage_path
        logger.info(f"BasicWorkflowEngine initialized with storage path: {storage_path}")
    
    def create_workflow(self, name: str, description: Optional[str] = None) -> str:
        """Create a new workflow."""
        try:
            workflow_id = str(uuid.uuid4())
            
            self.workflows[workflow_id] = {
                "id": workflow_id,
                "name": name,
                "description": description or "",
                "status": WorkflowStatus.CREATED.value,
                "tasks": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "execution_history": []
            }
            
            logger.info(f"Created workflow '{name}' with ID: {workflow_id}")
            return workflow_id
        except Exception as e:
            logger.error(f"Failed to create workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to create workflow: {str(e)}")
    
    def add_task(self, workflow_id: str, task_name: str, task_function: Callable, 
                dependencies: Optional[List[str]] = None) -> str:
        """Add a task to a workflow."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            task_id = str(uuid.uuid4())
            
            task = {
                "id": task_id,
                "name": task_name,
                "function": task_function,
                "dependencies": dependencies or [],
                "status": TaskStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "result": None,
                "error": None
            }
            
            self.tasks[task_id] = task
            self.workflows[workflow_id]["tasks"].append(task_id)
            self.workflows[workflow_id]["updated_at"] = datetime.now().isoformat()
            
            logger.info(f"Added task '{task_name}' to workflow {workflow_id}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to add task: {str(e)}")
            raise MCPWorkflowError(f"Failed to add task: {str(e)}")
    
    def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow with optional input data."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Update workflow status
            workflow["status"] = WorkflowStatus.RUNNING.value
            workflow["updated_at"] = datetime.now().isoformat()
            
            # Create execution record
            execution_id = str(uuid.uuid4())
            execution_record = {
                "id": execution_id,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "status": WorkflowStatus.RUNNING.value,
                "input_data": input_data or {},
                "output_data": {},
                "task_results": {}
            }
            
            workflow["execution_history"].append(execution_record)
            
            # Execute tasks in dependency order
            task_results = self._execute_tasks_in_order(workflow_id, input_data or {})
            
            # Update execution record
            execution_record["completed_at"] = datetime.now().isoformat()
            execution_record["status"] = WorkflowStatus.COMPLETED.value
            execution_record["output_data"] = task_results
            execution_record["task_results"] = {task_id: self.tasks[task_id]["result"] 
                                              for task_id in workflow["tasks"] 
                                              if self.tasks[task_id]["result"] is not None}
            
            # Update workflow status
            workflow["status"] = WorkflowStatus.COMPLETED.value
            workflow["updated_at"] = datetime.now().isoformat()
            
            logger.info(f"Executed workflow {workflow_id} successfully")
            return execution_record
        except Exception as e:
            logger.error(f"Failed to execute workflow: {str(e)}")
            
            # Update workflow status to failed
            if workflow_id in self.workflows:
                self.workflows[workflow_id]["status"] = WorkflowStatus.FAILED.value
                self.workflows[workflow_id]["updated_at"] = datetime.now().isoformat()
                
                # Update execution record if it exists
                if "execution_history" in self.workflows[workflow_id] and self.workflows[workflow_id]["execution_history"]:
                    latest_execution = self.workflows[workflow_id]["execution_history"][-1]
                    latest_execution["completed_at"] = datetime.now().isoformat()
                    latest_execution["status"] = WorkflowStatus.FAILED.value
            
            raise MCPWorkflowError(f"Failed to execute workflow: {str(e)}")
    
    def _execute_tasks_in_order(self, workflow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks in dependency order."""
        workflow = self.workflows[workflow_id]
        task_ids = workflow["tasks"]
        
        # Build dependency graph
        dependency_graph = {task_id: set(self.tasks[task_id]["dependencies"]) for task_id in task_ids}
        
        # Find tasks with no dependencies
        no_deps = [task_id for task_id, deps in dependency_graph.items() if not deps]
        
        # Results dictionary
        results = {}
        
        # Process tasks in order
        while no_deps:
            current_task_id = no_deps.pop(0)
            current_task = self.tasks[current_task_id]
            
            try:
                # Update task status
                current_task["status"] = TaskStatus.RUNNING.value
                current_task["updated_at"] = datetime.now().isoformat()
                
                # Execute task function
                task_input = {**input_data, **results}
                current_task["result"] = current_task["function"](task_input)
                
                # Update task status
                current_task["status"] = TaskStatus.COMPLETED.value
                current_task["updated_at"] = datetime.now().isoformat()
                
                # Add result to results dictionary
                results[current_task_id] = current_task["result"]
                
                # Update dependent tasks
                for task_id, deps in dependency_graph.items():
                    if current_task_id in deps:
                        deps.remove(current_task_id)
                        if not deps:
                            no_deps.append(task_id)
            
            except Exception as e:
                # Update task status to failed
                current_task["status"] = TaskStatus.FAILED.value
                current_task["error"] = str(e)
                current_task["updated_at"] = datetime.now().isoformat()
                
                logger.error(f"Task {current_task_id} failed: {str(e)}")
                raise MCPWorkflowError(f"Task {current_task_id} failed: {str(e)}")
        
        return results
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Calculate task statistics
            task_stats = {status.value: 0 for status in TaskStatus}
            for task_id in workflow["tasks"]:
                task_status = self.tasks[task_id]["status"]
                task_stats[task_status] = task_stats.get(task_status, 0) + 1
            
            return {
                "id": workflow["id"],
                "name": workflow["name"],
                "status": workflow["status"],
                "task_count": len(workflow["tasks"]),
                "task_stats": task_stats,
                "created_at": workflow["created_at"],
                "updated_at": workflow["updated_at"],
                "last_execution": workflow["execution_history"][-1] if workflow["execution_history"] else None
            }
        except Exception as e:
            logger.error(f"Failed to get workflow status: {str(e)}")
            raise MCPWorkflowError(f"Failed to get workflow status: {str(e)}")
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        try:
            return [
                {
                    "id": workflow["id"],
                    "name": workflow["name"],
                    "description": workflow["description"],
                    "status": workflow["status"],
                    "task_count": len(workflow["tasks"]),
                    "created_at": workflow["created_at"],
                    "updated_at": workflow["updated_at"]
                }
                for workflow in self.workflows.values()
            ]
        except Exception as e:
            logger.error(f"Failed to list workflows: {str(e)}")
            raise MCPWorkflowError(f"Failed to list workflows: {str(e)}")
    
    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get details of a specific workflow."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Get task details
            tasks = [
                {
                    "id": task_id,
                    "name": self.tasks[task_id]["name"],
                    "status": self.tasks[task_id]["status"],
                    "dependencies": self.tasks[task_id]["dependencies"],
                    "created_at": self.tasks[task_id]["created_at"],
                    "updated_at": self.tasks[task_id]["updated_at"]
                }
                for task_id in workflow["tasks"]
            ]
            
            return {
                "id": workflow["id"],
                "name": workflow["name"],
                "description": workflow["description"],
                "status": workflow["status"],
                "tasks": tasks,
                "created_at": workflow["created_at"],
                "updated_at": workflow["updated_at"],
                "execution_history": workflow["execution_history"]
            }
        except Exception as e:
            logger.error(f"Failed to get workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to get workflow: {str(e)}")
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            # Get task IDs to delete
            task_ids = self.workflows[workflow_id]["tasks"]
            
            # Delete tasks
            for task_id in task_ids:
                if task_id in self.tasks:
                    del self.tasks[task_id]
            
            # Delete workflow
            del self.workflows[workflow_id]
            
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to delete workflow: {str(e)}")
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            if workflow["status"] != WorkflowStatus.RUNNING.value:
                raise MCPWorkflowError(f"Workflow {workflow_id} is not running")
            
            workflow["status"] = WorkflowStatus.PAUSED.value
            workflow["updated_at"] = datetime.now().isoformat()
            
            logger.info(f"Paused workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to pause workflow: {str(e)}")
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        try:
            if workflow_id not in self.workflows:
                raise MCPWorkflowError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            if workflow["status"] != WorkflowStatus.PAUSED.value:
                raise MCPWorkflowError(f"Workflow {workflow_id} is not paused")
            
            workflow["status"] = WorkflowStatus.RUNNING.value
            workflow["updated_at"] = datetime.now().isoformat()
            
            # TODO: Implement actual resumption of execution
            # This would require storing execution state
            
            logger.info(f"Resumed workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to resume workflow: {str(e)}")

# τ:self_reference(workflow_engine_implementation_metadata)
{type:Implementation, file:"core/src/mcp/implementations/workflow_engine.py", version:"1.0.0", checksum:"sha256:workflow_engine_implementation_checksum", canonical_address:"workflow-engine-implementation", pfsus_compliant:true, lambda_operators:true, file_format:"implementation.workflow.v1.0.0.py"}

%% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-22T00:00:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=implementation.workflow.v1.0.0.pyow: {workflow_id}")
                return False
            
            # Pause each execution
            for execution_id in running_executions:
                self.executions[execution_id]["status"] = "paused"
            
            logger.info(f"Paused workflow: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to pause workflow: {str(e)}")
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        try:
            # Find paused executions for this workflow
            paused_executions = [
                execution_id for execution_id, execution in self.executions.items()
                if execution["workflow_id"] == workflow_id and execution["status"] == "paused"
            ]
            
            if not paused_executions:
                logger.warning(f"No paused executions found for workflow: {workflow_id}")
                return False
            
            # Resume each execution
            for execution_id in paused_executions:
                self.executions[execution_id]["status"] = "running"
                
                # Start execution in a separate thread
                thread = threading.Thread(
                    target=self._execute_workflow_thread,
                    args=(workflow_id, execution_id, self.executions[execution_id]["input_data"])
                )
                thread.start()
            
            logger.info(f"Resumed workflow: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume workflow: {str(e)}")
            raise MCPWorkflowError(f"Failed to resume workflow: {str(e)}")
    
    def _execute_workflow_thread(self, workflow_id: str, execution_id: str, input_data: Dict[str, Any]) -> None:
        """Execute a workflow in a separate thread."""
        try:
            workflow = self.workflows[workflow_id]
            execution = self.executions[execution_id]
            
            # Get tasks
            tasks = []
            for task_id in workflow["tasks"]:
                if task_id in self.tasks:
                    tasks.append(self.tasks[task_id])
            
            # Sort tasks by dependencies
            sorted_tasks = self._topological_sort(tasks)
            
            # Execute tasks
            results = input_data.copy()
            for task in sorted_tasks:
                # Check if execution is paused or cancelled
                if self.executions[execution_id]["status"] != "running":
                    logger.info(f"Execution {execution_id} is {self.executions[execution_id]['status']}")
                    return
                
                try:
                    # Execute task
                    task_result = task["function"](results)
                    
                    # Store result
                    self.executions[execution_id]["task_results"][task["id"]] = {
                        "status": "completed",
                        "result": task_result
                    }
                    
                    # Update results for next task
                    if isinstance(task_result, dict):
                        results.update(task_result)
                except Exception as e:
                    error_message = f"Task '{task['name']}' failed: {str(e)}"
                    logger.error(error_message)
                    
                    # Store error
                    self.executions[execution_id]["task_results"][task["id"]] = {
                        "status": "failed",
                        "error": error_message
                    }
                    self.executions[execution_id]["errors"].append(error_message)
                    
                    # Mark execution as failed
                    self.executions[execution_id]["status"] = "failed"
                    self.executions[execution_id]["end_time"] = datetime.now().isoformat()
                    return
            
            # Mark execution as completed
            self.executions[execution_id]["status"] = "completed"
            self.executions[execution_id]["end_time"] = datetime.now().isoformat()
            self.executions[execution_id]["output_data"] = results
            
            logger.info(f"Execution {execution_id} completed successfully")
        except Exception as e:
            error_message = f"Workflow execution failed: {str(e)}"
            logger.error(error_message)
            
            # Mark execution as failed
            self.executions[execution_id]["status"] = "failed"
            self.executions[execution_id]["end_time"] = datetime.now().isoformat()
            self.executions[execution_id]["errors"].append(error_message)
    
    def _topological_sort(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort tasks by dependencies using topological sort."""
        # Create a dictionary of tasks by ID
        task_dict = {task["id"]: task for task in tasks}
        
        # Create a dictionary of dependencies
        dependencies = {task["id"]: set(task["dependencies"]) for task in tasks}
        
        # Create a list of tasks with no dependencies
        no_dependencies = [task for task in tasks if not task["dependencies"]]
        
        # Create a list of sorted tasks
        sorted_tasks = []
        
        # Process tasks with no dependencies
        while no_dependencies:
            # Get the next task
            task = no_dependencies.pop(0)
            sorted_tasks.append(task)
            
            # Update dependencies
            for task_id, deps in dependencies.items():
                if task["id"] in deps:
                    deps.remove(task["id"])
                    if not deps and task_id not in [t["id"] for t in sorted_tasks + no_dependencies]:
                        no_dependencies.append(task_dict[task_id])
        
        # Check for circular dependencies
        if len(sorted_tasks) != len(tasks):
            raise MCPWorkflowError("Circular dependencies detected in workflow")
        
        return sorted_tasks
    
    def _save_to_disk(self) -> None:
        """Save the workflow state to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Prepare data for serialization
            serializable_data = {
                "workflows": self.workflows,
                "tasks": {},
                "executions": self.executions
            }
            
            # Convert non-serializable task functions to strings
            for task_id, task in self.tasks.items():
                serializable_task = task.copy()
                serializable_task["function"] = str(task["function"])
                serializable_data["tasks"][task_id] = serializable_task
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved workflow state to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save to disk: {str(e)}")
            raise MCPWorkflowError(f"Failed to save to disk: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load the workflow state from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.workflows = data.get("workflows", {})
            self.executions = data.get("executions", {})
            
            # Note: Task functions cannot be loaded from disk
            # They need to be re-registered after loading
            
            logger.info(f"Loaded workflow state from {self.storage_path}")
            logger.warning("Task functions could not be loaded and need to be re-registered")
        except Exception as e:
            logger.error(f"Failed to load from disk: {str(e)}")
            raise MCPWorkflowError(f"Failed to load from disk: {str(e)}")