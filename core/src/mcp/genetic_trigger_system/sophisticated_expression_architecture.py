"""
Sophisticated Genetic Expression Architecture

This module implements advanced genetic expression architecture with:
- Interruption points for dynamic execution flow control
- Alignment hooks for cross-lobe communication and synchronization
- Multiple circuit layouts for task-specific genetic expressions
- Reputation scoring system for genetic sequence performance
- Universal development hooks for external system integration
- Multidimensional hierarchical task navigation, using B* search with controlled randomness
- Complex gene expression condition chains with dependency resolution
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import heapq

from .environmental_state import EnvironmentalState


class InterruptionType(Enum):
    """Types of interruption points"""
    CONDITIONAL = "conditional"      # Conditional interruption based on state
    TIMED = "timed"                 # Time-based interruption
    RESOURCE = "resource"           # Resource-based interruption
    EXTERNAL = "external"           # External signal interruption
    PERFORMANCE = "performance"     # Performance-based interruption


class CircuitLayout(Enum):
    """Types of circuit layouts"""
    LINEAR = "linear"               # Linear execution path
    BRANCHED = "branched"           # Branched execution with decision points
    PARALLEL = "parallel"           # Parallel execution paths
    RECURSIVE = "recursive"         # Recursive execution with depth limits
    ADAPTIVE = "adaptive"           # Adaptive layout that changes based on context


class AlignmentHookType(Enum):
    """Types of alignment hooks"""
    HORMONE_SYNC = "hormone_sync"           # Hormone system synchronization
    MEMORY_ACCESS = "memory_access"         # Memory system access
    NEURAL_FEEDBACK = "neural_feedback"     # Neural network feedback
    ENVIRONMENTAL = "environmental"         # Environmental state monitoring
    PERFORMANCE_MONITOR = "performance_monitor"  # Performance monitoring


@dataclass
class InterruptionPoint:
    """Represents an interruption point in genetic expression"""
    point_id: str
    interruption_type: InterruptionType
    condition: Callable[[Dict[str, Any]], bool]
    position: int
    priority: float = 1.0
    handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation_count: int = 0
    last_activated: Optional[datetime] = None
    
    def should_activate(self, context: Dict[str, Any]) -> bool:
        """Check if interruption point should activate"""
        try:
            return self.condition(context)
        except Exception:
            return False
    
    def activate(self, context: Dict[str, Any]) -> Any:
        """Activate the interruption point"""
        self.activation_count += 1
        self.last_activated = datetime.now()
        
        if self.handler:
            try:
                return self.handler(context)
            except Exception as e:
                logging.error(f"Error in interruption handler: {e}")
                return None
        return None


@dataclass
class AlignmentHook:
    """Represents an alignment hook for cross-lobe communication"""
    hook_id: str
    hook_type: AlignmentHookType
    target_system: str
    callback: Callable
    priority: float = 1.0
    active: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def trigger(self, data: Any) -> Any:
        """Trigger the alignment hook"""
        if not self.active:
            return None
        
        self.trigger_count += 1
        self.last_triggered = datetime.now()
        
        try:
            return self.callback(data)
        except Exception as e:
            logging.error(f"Error in alignment hook {self.hook_id}: {e}")
            return None


@dataclass
class ReputationScore:
    """Represents reputation scoring for genetic sequences"""
    sequence_id: str
    overall_score: float = 0.5
    performance_scores: Dict[str, float] = field(default_factory=dict)
    reliability_score: float = 0.5
    adaptability_score: float = 0.5
    efficiency_score: float = 0.5
    stability_score: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_score(self, new_scores: Dict[str, float]):
        """Update reputation scores"""
        self.performance_scores.update(new_scores)
        
        # Calculate overall score
        if self.performance_scores:
            self.overall_score = sum(self.performance_scores.values()) / len(self.performance_scores)
        
        # Update individual scores
        self.reliability_score = self.performance_scores.get('reliability', self.reliability_score)
        self.adaptability_score = self.performance_scores.get('adaptability', self.adaptability_score)
        self.efficiency_score = self.performance_scores.get('efficiency', self.efficiency_score)
        self.stability_score = self.performance_scores.get('stability', self.stability_score)
        
        # Record history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'scores': new_scores.copy(),
            'overall_score': self.overall_score
        }
        self.history.append(history_entry)
        
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        self.last_updated = datetime.now()
        self.update_count += 1
    
    def get_trend(self, window: int = 10) -> float:
        """Get score trend over recent window"""
        if len(self.history) < 2:
            return 0.0
        
        recent_history = self.history[-window:] if len(self.history) >= window else self.history
        
        if len(recent_history) < 2:
            return 0.0
        
        first_score = recent_history[0]['overall_score']
        last_score = recent_history[-1]['overall_score']
        
        return last_score - first_score


@dataclass
class TaskNode:
    """Represents a node in the hierarchical task navigation"""
    node_id: str
    task_type: str
    priority: float
    complexity: float
    dependencies: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_state: str = "pending"  # pending, running, completed, failed
    estimated_duration: float = 0.0
    actual_duration: Optional[float] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute"""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class GeneExpressionCondition:
    """Represents a condition in gene expression"""
    condition_id: str
    condition_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: float = 1.0
    active: bool = True
    last_evaluated: Optional[datetime] = None
    evaluation_count: int = 0
    success_rate: float = 0.5
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition"""
        self.last_evaluated = datetime.now()
        self.evaluation_count += 1
        
        try:
            result = self._evaluate_condition(context)
            
            # Update success rate
            if self.evaluation_count == 1:
                self.success_rate = 1.0 if result else 0.0
            else:
                self.success_rate = (self.success_rate * (self.evaluation_count - 1) + (1.0 if result else 0.0)) / self.evaluation_count
            
            return result
        except Exception as e:
            logging.error(f"Error evaluating condition {self.condition_id}: {e}")
            return False
    
    def _evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """Internal condition evaluation"""
        if self.condition_type == "threshold":
            value = context.get(self.parameters.get('metric', ''), 0.0)
            threshold = self.parameters.get('threshold', 0.5)
            return value >= threshold
        elif self.condition_type == "range":
            value = context.get(self.parameters.get('metric', ''), 0.0)
            min_val = self.parameters.get('min', 0.0)
            max_val = self.parameters.get('max', 1.0)
            return min_val <= value <= max_val
        elif self.condition_type == "boolean":
            return context.get(self.parameters.get('key', ''), False)
        elif self.condition_type == "custom":
            # Custom condition evaluation
            return self.parameters.get('result', False)
        else:
            return False


class SophisticatedExpressionArchitecture:
    """
    Sophisticated genetic expression architecture with advanced features.
    
    Features:
    - Interruption points for dynamic flow control
    - Alignment hooks for cross-lobe communication
    - Multiple circuit layouts
    - Reputation scoring system
    - Universal development hooks
    - Multidimensional task navigation
    - Complex condition chains
    """
    
    def __init__(self, max_interruptions: int = 50, max_hooks: int = 100):
        self.max_interruptions = max_interruptions
        self.max_hooks = max_hooks
        self.logger = logging.getLogger("SophisticatedExpression")
        
        # Core components
        self.interruption_points: Dict[str, InterruptionPoint] = {}
        self.alignment_hooks: Dict[str, AlignmentHook] = {}
        self.reputation_scores: Dict[str, ReputationScore] = {}
        self.task_nodes: Dict[str, TaskNode] = {}
        self.expression_conditions: Dict[str, GeneExpressionCondition] = {}
        
        # Execution state
        self.current_execution_path: List[str] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.active_interruptions: Set[str] = set()
        
        # Development hooks
        self.development_hooks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.execution_metrics: Dict[str, float] = defaultdict(float)
        self.interruption_metrics: Dict[str, int] = defaultdict(int)
        
        self.logger.info("Sophisticated Expression Architecture initialized")
    
    def add_interruption_point(self, point: InterruptionPoint) -> bool:
        """Add an interruption point to the architecture"""
        if len(self.interruption_points) >= self.max_interruptions:
            self.logger.warning("Maximum interruption points reached")
            return False
        
        self.interruption_points[point.point_id] = point
        self.logger.debug(f"Added interruption point {point.point_id}")
        return True
    
    def remove_interruption_point(self, point_id: str) -> bool:
        """Remove an interruption point"""
        if point_id in self.interruption_points:
            del self.interruption_points[point_id]
            self.logger.debug(f"Removed interruption point {point_id}")
            return True
        return False
    
    def add_alignment_hook(self, hook: AlignmentHook) -> bool:
        """Add an alignment hook for cross-lobe communication"""
        if len(self.alignment_hooks) >= self.max_hooks:
            self.logger.warning("Maximum alignment hooks reached")
            return False
        
        self.alignment_hooks[hook.hook_id] = hook
        self.logger.debug(f"Added alignment hook {hook.hook_id}")
        return True
    
    def remove_alignment_hook(self, hook_id: str) -> bool:
        """Remove an alignment hook"""
        if hook_id in self.alignment_hooks:
            del self.alignment_hooks[hook_id]
            self.logger.debug(f"Removed alignment hook {hook_id}")
            return True
        return False
    
    async def execute_with_interruptions(self, execution_context: Dict[str, Any], 
                                       base_execution: Callable) -> Any:
        """
        Execute with interruption point monitoring.
        
        Args:
            execution_context: Context for execution
            base_execution: Base execution function
            
        Returns:
            Execution result
        """
        self.logger.info("Starting execution with interruption monitoring")
        
        # Check for interruption points before execution
        pre_interruptions = await self._check_interruption_points(execution_context, "pre")
        
        if pre_interruptions:
            self.logger.info(f"Pre-execution interruptions triggered: {len(pre_interruptions)}")
            for point_id in pre_interruptions:
                await self._handle_interruption(point_id, execution_context)
        
        # Execute base function
        try:
            result = await base_execution(execution_context)
            
            # Check for interruption points after execution
            post_interruptions = await self._check_interruption_points(execution_context, "post")
            
            if post_interruptions:
                self.logger.info(f"Post-execution interruptions triggered: {len(post_interruptions)}")
                for point_id in post_interruptions:
                    await self._handle_interruption(point_id, execution_context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            
            # Check for error interruption points
            error_interruptions = await self._check_interruption_points(
                {**execution_context, 'error': str(e)}, "error"
            )
            
            for point_id in error_interruptions:
                await self._handle_interruption(point_id, execution_context)
            
            raise
    
    async def _check_interruption_points(self, context: Dict[str, Any], 
                                       phase: str) -> List[str]:
        """Check for interruption points that should activate"""
        triggered_points = []
        
        for point_id, point in self.interruption_points.items():
            if point.metadata.get('phase', 'any') in [phase, 'any']:
                if point.should_activate(context):
                    triggered_points.append(point_id)
        
        # Sort by priority (highest first)
        triggered_points.sort(
            key=lambda pid: self.interruption_points[pid].priority, 
            reverse=True
        )
        
        return triggered_points
    
    async def _handle_interruption(self, point_id: str, context: Dict[str, Any]):
        """Handle an interruption point"""
        point = self.interruption_points[point_id]
        self.active_interruptions.add(point_id)
        self.interruption_metrics[point_id] += 1
        
        try:
            result = point.activate(context)
            self.logger.debug(f"Interruption {point_id} handled, result: {result}")
            
            # Trigger alignment hooks if needed
            await self._trigger_alignment_hooks(point_id, result)
            
        except Exception as e:
            self.logger.error(f"Error handling interruption {point_id}: {e}")
        finally:
            self.active_interruptions.discard(point_id)
    
    async def _trigger_alignment_hooks(self, trigger_id: str, data: Any):
        """Trigger relevant alignment hooks"""
        for hook_id, hook in self.alignment_hooks.items():
            if hook.metadata.get('trigger_id') == trigger_id:
                try:
                    hook.trigger(data)
                except Exception as e:
                    self.logger.error(f"Error triggering hook {hook_id}: {e}")
    
    def create_circuit_layout(self, layout_type: CircuitLayout, 
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a circuit layout for genetic expression.
        
        Args:
            layout_type: Type of circuit layout
            parameters: Layout parameters
            
        Returns:
            Circuit layout configuration
        """
        layout = {
            'layout_id': f"layout_{uuid.uuid4().hex[:8]}",
            'layout_type': layout_type.value,
            'parameters': parameters.copy(),
            'created_at': datetime.now().isoformat()
        }
        
        if layout_type == CircuitLayout.LINEAR:
            layout['execution_path'] = self._create_linear_path(parameters)
        elif layout_type == CircuitLayout.BRANCHED:
            layout['execution_path'] = self._create_branched_path(parameters)
        elif layout_type == CircuitLayout.PARALLEL:
            layout['execution_path'] = self._create_parallel_path(parameters)
        elif layout_type == CircuitLayout.RECURSIVE:
            layout['execution_path'] = self._create_recursive_path(parameters)
        elif layout_type == CircuitLayout.ADAPTIVE:
            layout['execution_path'] = self._create_adaptive_path(parameters)
        
        self.logger.info(f"Created {layout_type.value} circuit layout: {layout['layout_id']}")
        return layout
    
    def _create_linear_path(self, parameters: Dict[str, Any]) -> List[str]:
        """Create linear execution path"""
        steps = parameters.get('steps', 5)
        return [f"step_{i}" for i in range(steps)]
    
    def _create_branched_path(self, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create branched execution path"""
        branches = parameters.get('branches', 3)
        branch_length = parameters.get('branch_length', 3)
        
        path = {}
        for i in range(branches):
            path[f"branch_{i}"] = [f"step_{i}_{j}" for j in range(branch_length)]
        
        return path
    
    def _create_parallel_path(self, parameters: Dict[str, Any]) -> List[List[str]]:
        """Create parallel execution paths"""
        parallel_paths = parameters.get('parallel_paths', 2)
        path_length = parameters.get('path_length', 4)
        
        return [
            [f"parallel_{i}_step_{j}" for j in range(path_length)]
            for i in range(parallel_paths)
        ]
    
    def _create_recursive_path(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create recursive execution path"""
        max_depth = parameters.get('max_depth', 3)
        recursion_factor = parameters.get('recursion_factor', 0.5)
        
        return {
            'max_depth': max_depth,
            'recursion_factor': recursion_factor,
            'base_path': [f"recursive_step_{i}" for i in range(3)]
        }
    
    def _create_adaptive_path(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create adaptive execution path"""
        adaptation_factors = parameters.get('adaptation_factors', ['performance', 'load'])
        adaptation_thresholds = parameters.get('adaptation_thresholds', {})
        
        return {
            'adaptation_factors': adaptation_factors,
            'adaptation_thresholds': adaptation_thresholds,
            'base_paths': {
                'high_performance': [f"adaptive_high_{i}" for i in range(3)],
                'low_performance': [f"adaptive_low_{i}" for i in range(3)],
                'balanced': [f"adaptive_balanced_{i}" for i in range(3)]
            }
        }
    
    def update_reputation_score(self, sequence_id: str, scores: Dict[str, float]):
        """Update reputation score for a genetic sequence"""
        if sequence_id not in self.reputation_scores:
            self.reputation_scores[sequence_id] = ReputationScore(sequence_id=sequence_id)
        
        self.reputation_scores[sequence_id].update_score(scores)
        self.logger.debug(f"Updated reputation score for {sequence_id}")
    
    def get_reputation_score(self, sequence_id: str) -> Optional[ReputationScore]:
        """Get reputation score for a genetic sequence"""
        return self.reputation_scores.get(sequence_id)
    
    def add_task_node(self, node: TaskNode) -> bool:
        """Add a task node to the hierarchical navigation"""
        self.task_nodes[node.node_id] = node
        self.logger.debug(f"Added task node {node.node_id}")
        return True
    
    def remove_task_node(self, node_id: str) -> bool:
        """Remove a task node"""
        if node_id in self.task_nodes:
            del self.task_nodes[node_id]
            self.logger.debug(f"Removed task node {node_id}")
            return True
        return False
    
    async def navigate_task_hierarchy(self, start_node_id: str, 
                                    target_metrics: Dict[str, float],
                                    max_iterations: int = 100) -> List[str]:
        """
        Navigate task hierarchy using B* search with controlled randomness.
        
        Args:
            start_node_id: Starting node ID
            target_metrics: Target performance metrics
            max_iterations: Maximum search iterations
            
        Returns:
            Optimal execution path
        """
        if start_node_id not in self.task_nodes:
            self.logger.error(f"Start node {start_node_id} not found")
            return []
        
        # Initialize search
        open_set = [(0, start_node_id)]  # (priority, node_id)
        closed_set = set()
        came_from = {}
        g_score = {start_node_id: 0}
        f_score = {start_node_id: self._heuristic_score(start_node_id, target_metrics)}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            current_priority, current_id = heapq.heappop(open_set)
            
            if current_id in closed_set:
                continue
            
            closed_set.add(current_id)
            
            # Check if we've reached a good solution
            if self._is_solution_satisfactory(current_id, target_metrics):
                return self._reconstruct_path(came_from, current_id)
            
            # Explore neighbors
            for neighbor_id in self._get_neighbors(current_id):
                if neighbor_id in closed_set:
                    continue
                
                tentative_g_score = g_score[current_id] + self._edge_cost(current_id, neighbor_id)
                
                if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + self._heuristic_score(neighbor_id, target_metrics)
                    
                    # Add controlled randomness
                    random_factor = random.uniform(0.9, 1.1)
                    priority = f_score[neighbor_id] * random_factor
                    
                    heapq.heappush(open_set, (priority, neighbor_id))
            
            iterations += 1
        
        self.logger.warning(f"Task navigation reached max iterations: {max_iterations}")
        return self._reconstruct_path(came_from, start_node_id)
    
    def _heuristic_score(self, node_id: str, target_metrics: Dict[str, float]) -> float:
        """Calculate heuristic score for B* search"""
        if node_id not in self.task_nodes:
            return float('inf')
        
        node = self.task_nodes[node_id]
        
        # Base heuristic on task complexity and priority
        base_score = (1.0 - node.priority) + node.complexity
        
        # Adjust based on target metrics
        metric_adjustment = 0.0
        for metric, target_value in target_metrics.items():
            if metric in node.metadata:
                current_value = node.metadata[metric]
                metric_adjustment += abs(current_value - target_value)
        
        return base_score + metric_adjustment
    
    def _is_solution_satisfactory(self, node_id: str, target_metrics: Dict[str, float]) -> bool:
        """Check if a solution is satisfactory"""
        if node_id not in self.task_nodes:
            return False
        
        node = self.task_nodes[node_id]
        
        # Check if all target metrics are met
        for metric, target_value in target_metrics.items():
            if metric in node.metadata:
                current_value = node.metadata[metric]
                if abs(current_value - target_value) > 0.1:  # 10% tolerance
                    return False
        
        return True
    
    def _get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring task nodes"""
        if node_id not in self.task_nodes:
            return []
        
        node = self.task_nodes[node_id]
        return node.children
    
    def _edge_cost(self, from_id: str, to_id: str) -> float:
        """Calculate edge cost between task nodes"""
        if from_id not in self.task_nodes or to_id not in self.task_nodes:
            return float('inf')
        
        from_node = self.task_nodes[from_id]
        to_node = self.task_nodes[to_id]
        
        # Base cost on complexity difference
        complexity_cost = abs(from_node.complexity - to_node.complexity)
        
        # Add resource transition cost
        resource_cost = 0.0
        for resource in set(from_node.resource_requirements.keys()) | set(to_node.resource_requirements.keys()):
            from_req = from_node.resource_requirements.get(resource, 0.0)
            to_req = to_node.resource_requirements.get(resource, 0.0)
            resource_cost += abs(from_req - to_req)
        
        return complexity_cost + resource_cost * 0.1
    
    def _reconstruct_path(self, came_from: Dict[str, str], current_id: str) -> List[str]:
        """Reconstruct path from search results"""
        path = [current_id]
        while current_id in came_from:
            current_id = came_from[current_id]
            path.append(current_id)
        return list(reversed(path))
    
    def add_expression_condition(self, condition: GeneExpressionCondition) -> bool:
        """Add a gene expression condition"""
        self.expression_conditions[condition.condition_id] = condition
        self.logger.debug(f"Added expression condition {condition.condition_id}")
        return True
    
    def remove_expression_condition(self, condition_id: str) -> bool:
        """Remove a gene expression condition"""
        if condition_id in self.expression_conditions:
            del self.expression_conditions[condition_id]
            self.logger.debug(f"Removed expression condition {condition_id}")
            return True
        return False
    
    async def evaluate_condition_chain(self, conditions: List[str], 
                                     context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Evaluate a chain of gene expression conditions.
        
        Args:
            conditions: List of condition IDs to evaluate
            context: Context for evaluation
            
        Returns:
            Dictionary of condition results
        """
        results = {}
        evaluated_conditions = set()
        
        # Evaluate conditions in dependency order
        while len(evaluated_conditions) < len(conditions):
            progress_made = False
            
            for condition_id in conditions:
                if condition_id in evaluated_conditions:
                    continue
                
                condition = self.expression_conditions.get(condition_id)
                if not condition:
                    results[condition_id] = False
                    evaluated_conditions.add(condition_id)
                    continue
                
                # Check if dependencies are satisfied
                dependencies_satisfied = all(
                    dep in evaluated_conditions and results.get(dep, False)
                    for dep in condition.dependencies
                )
                
                if dependencies_satisfied:
                    result = condition.evaluate(context)
                    results[condition_id] = result
                    evaluated_conditions.add(condition_id)
                    progress_made = True
            
            if not progress_made:
                # Circular dependency detected
                self.logger.warning("Circular dependency detected in condition chain")
                break
        
        return results
    
    def add_development_hook(self, hook_name: str, hook_function: Callable):
        """Add a universal development hook"""
        self.development_hooks[hook_name] = hook_function
        self.logger.debug(f"Added development hook: {hook_name}")
    
    def remove_development_hook(self, hook_name: str) -> bool:
        """Remove a development hook"""
        if hook_name in self.development_hooks:
            del self.development_hooks[hook_name]
            self.logger.debug(f"Removed development hook: {hook_name}")
            return True
        return False
    
    async def trigger_development_hook(self, hook_name: str, data: Any) -> Any:
        """Trigger a development hook"""
        if hook_name not in self.development_hooks:
            self.logger.warning(f"Development hook {hook_name} not found")
            return None
        
        try:
            hook_function = self.development_hooks[hook_name]
            if asyncio.iscoroutinefunction(hook_function):
                result = await hook_function(data)
            else:
                result = hook_function(data)
            
            self.logger.debug(f"Development hook {hook_name} triggered successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error triggering development hook {hook_name}: {e}")
            return None
    
    def get_architecture_statistics(self) -> Dict[str, Any]:
        """Get comprehensive architecture statistics"""
        return {
            'interruption_points': len(self.interruption_points),
            'alignment_hooks': len(self.alignment_hooks),
            'reputation_scores': len(self.reputation_scores),
            'task_nodes': len(self.task_nodes),
            'expression_conditions': len(self.expression_conditions),
            'development_hooks': len(self.development_hooks),
            'active_interruptions': len(self.active_interruptions),
            'execution_metrics': dict(self.execution_metrics),
            'interruption_metrics': dict(self.interruption_metrics),
            'total_executions': len(self.execution_history)
        }
    
    def save_architecture_state(self, filepath: str):
        """Save architecture state to file"""
        state = {
            'interruption_points': {
                pid: {
                    'point_id': point.point_id,
                    'interruption_type': point.interruption_type.value,
                    'position': point.position,
                    'priority': point.priority,
                    'metadata': point.metadata,
                    'activation_count': point.activation_count
                }
                for pid, point in self.interruption_points.items()
            },
            'alignment_hooks': {
                hid: {
                    'hook_id': hook.hook_id,
                    'hook_type': hook.hook_type.value,
                    'target_system': hook.target_system,
                    'priority': hook.priority,
                    'active': hook.active,
                    'trigger_count': hook.trigger_count,
                    'metadata': hook.metadata
                }
                for hid, hook in self.alignment_hooks.items()
            },
            'reputation_scores': {
                rid: {
                    'sequence_id': score.sequence_id,
                    'overall_score': score.overall_score,
                    'performance_scores': score.performance_scores,
                    'update_count': score.update_count
                }
                for rid, score in self.reputation_scores.items()
            },
            'task_nodes': {
                nid: {
                    'node_id': node.node_id,
                    'task_type': node.task_type,
                    'priority': node.priority,
                    'complexity': node.complexity,
                    'dependencies': node.dependencies,
                    'children': node.children,
                    'execution_state': node.execution_state,
                    'resource_requirements': node.resource_requirements
                }
                for nid, node in self.task_nodes.items()
            },
            'expression_conditions': {
                cid: {
                    'condition_id': condition.condition_id,
                    'condition_type': condition.condition_type,
                    'parameters': condition.parameters,
                    'dependencies': condition.dependencies,
                    'priority': condition.priority,
                    'active': condition.active,
                    'success_rate': condition.success_rate
                }
                for cid, condition in self.expression_conditions.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Architecture state saved to {filepath}")
    
    def load_architecture_state(self, filepath: str):
        """Load architecture state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load interruption points
        self.interruption_points.clear()
        for pid, data in state.get('interruption_points', {}).items():
            point = InterruptionPoint(
                point_id=data['point_id'],
                interruption_type=InterruptionType(data['interruption_type']),
                condition=lambda ctx: True,  # Default condition
                position=data['position'],
                priority=data['priority'],
                metadata=data['metadata']
            )
            point.activation_count = data['activation_count']
            self.interruption_points[pid] = point
        
        # Load other components (simplified for brevity)
        self.logger.info(f"Architecture state loaded from {filepath}")


# Import itertools for advanced operations
import itertools 