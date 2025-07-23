#!/usr/bin/env python3
"""
System Healer for Self-Repair
@{CORE.SELF_REPAIR.HEALER.001} System healing component for self-repair.
#{healing,repair,recovery,functional}
β(λ(healing_implementation))
"""

import asyncio
import logging
import time
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass
from functools import reduce
from datetime import datetime

# Import core components
from core.src.mcp.core_system import MCPCoreSystem
from core.src.mcp.workflow import WorkflowManager

logger = logging.getLogger(__name__)

@dataclass
class RepairAction:
    """Immutable representation of a repair action."""
    id: str
    component: str
    action_type: str
    parameters: Dict[str, Any]
    priority: float  # 0.0 to 1.0
    estimated_impact: float  # 0.0 to 1.0
    estimated_success: float  # 0.0 to 1.0

@dataclass
class RepairResult:
    """Immutable representation of a repair result."""
    action_id: str
    success: bool
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    timestamp: datetime
    duration: float  # seconds
    side_effects: List[str]

class SystemHealer:
    """
    System healing component for self-repair.
    
    Implements repair strategies using functional programming principles.
    Each repair is a pure function: system_state -> system_state
    """
    
    def __init__(self, core_system: MCPCoreSystem):
        """Initialize the system healer."""
        self.core_system = core_system
        self.workflow_manager = core_system.workflow_manager
        
        # Repair strategies (functions)
        self.repair_strategies: Dict[str, Callable] = {
            "restart_component": self._restart_component_strategy,
            "reallocate_resources": self._reallocate_resources_strategy,
            "clear_cache": self._clear_cache_strategy,
            "reconfigure_component": self._reconfigure_component_strategy,
            "rollback_component": self._rollback_component_strategy
        }
        
        # Repair history
        self.repair_history: List[RepairResult] = []
        
        logger.info("System Healer initialized")
    
    async def execute_repair(self, repair_action: RepairAction) -> RepairResult:
        """Execute a repair action and return the result."""
        logger.info(f"Executing repair: {repair_action.action_type} on {repair_action.component}")
        
        # Get repair strategy
        repair_strategy = self.repair_strategies.get(repair_action.action_type)
        if not repair_strategy:
            logger.error(f"Unknown repair strategy: {repair_action.action_type}")
            return RepairResult(
                action_id=repair_action.id,
                success=False,
                metrics_before={},
                metrics_after={},
                timestamp=datetime.now(),
                duration=0.0,
                side_effects=["Unknown repair strategy"]
            )
        
        # Get metrics before repair
        metrics_before = self._get_component_metrics(repair_action.component)
        
        # Execute repair strategy
        start_time = time.time()
        try:
            success, side_effects = await repair_strategy(
                repair_action.component, 
                repair_action.parameters
            )
        except Exception as e:
            logger.error(f"Error executing repair: {e}")
            success = False
            side_effects = [f"Error: {str(e)}"]
        
        duration = time.time() - start_time
        
        # Get metrics after repair
        metrics_after = self._get_component_metrics(repair_action.component)
        
        # Create result
        result = RepairResult(
            action_id=repair_action.id,
            success=success,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            timestamp=datetime.now(),
            duration=duration,
            side_effects=side_effects
        )
        
        # Add to history
        self.repair_history.append(result)
        
        logger.info(f"Repair completed: success={success}, duration={duration:.2f}s")
        return result
    
    def _get_component_metrics(self, component_id: str) -> Dict[str, float]:
        """Get metrics for a specific component."""
        component = self.core_system.components.get(component_id)
        if not component:
            return {}
        
        return {
            "cpu_usage": component.get_cpu_usage(),
            "memory_usage": component.get_memory_usage(),
            "error_rate": component.get_error_rate(),
            "response_time": component.get_response_time()
        }
    
    async def _restart_component_strategy(
        self, component_id: str, parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Strategy to restart a component."""
        component = self.core_system.components.get(component_id)
        if not component:
            return False, ["Component not found"]
        
        try:
            # Graceful shutdown
            await component.shutdown(timeout=parameters.get("timeout", 30))
            
            # Wait for shutdown
            await asyncio.sleep(parameters.get("wait_time", 2))
            
            # Start component
            await component.start()
            
            # Verify component is running
            if component.is_active:
                return True, ["Component restarted successfully"]
            else:
                return False, ["Component failed to start"]
                
        except Exception as e:
            return False, [f"Restart failed: {str(e)}"]
    
    async def _reallocate_resources_strategy(
        self, component_id: str, parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Strategy to reallocate resources for a component."""
        component = self.core_system.components.get(component_id)
        if not component:
            return False, ["Component not found"]
        
        try:
            # Get current resource allocation
            current_allocation = component.get_resource_allocation()
            
            # Calculate new allocation
            new_allocation = {
                "cpu": parameters.get("cpu", current_allocation.get("cpu")),
                "memory": parameters.get("memory", current_allocation.get("memory")),
                "disk": parameters.get("disk", current_allocation.get("disk")),
                "network": parameters.get("network", current_allocation.get("network"))
            }
            
            # Apply new allocation
            success = await component.set_resource_allocation(new_allocation)
            
            if success:
                return True, ["Resources reallocated successfully"]
            else:
                return False, ["Failed to reallocate resources"]
                
        except Exception as e:
            return False, [f"Resource reallocation failed: {str(e)}"]
    
    # Additional repair strategies would be implemented here...