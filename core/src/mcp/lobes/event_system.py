"""
Event System for MCP Lobe Communication
@{CORE.LOBES.EVENT.001} Δ:event_orchestration(inter_lobe_communication_system)
#{event_system,lobe_communication,message_passing,async_events,lambda_operators}
Δ(β(τ(event_system_orchestration)))
"""

import asyncio
import uuid
import logging
from typing import Any, Dict, List, Optional, Callable, Set, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

from core.src.mcp.interfaces.lobe import ILobe
from core.src.mcp.exceptions import MCPLobeError

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """λ:priority_levels(event_importance_classification)"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """β:event_categories(system_event_classification)"""
    LOBE_MESSAGE = "lobe_message"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR_EVENT = "error_event"
    STATUS_UPDATE = "status_update"
    WORKFLOW_EVENT = "workflow_event"
    MEMORY_EVENT = "memory_event"
    CONTEXT_EVENT = "context_event"


@dataclass
class Event:
    """ℵ:event_structure(message_data_container)"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.SYSTEM_NOTIFICATION
    source_lobe_id: str = ""
    target_lobe_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    retry_count: int = 0
    max_retries: int = 3


class EventSystem:
    """
    Δ:event_orchestration(centralized_lobe_communication_system)
    
    Manages inter-lobe communication through an event-driven architecture.
    Provides asynchronous message passing, event queuing, and delivery guarantees.
    """
    
    def __init__(self, max_workers: int = 4):
        """τ:initialization(event_system_setup)"""
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[EventType, Set[Callable]] = {}
        self.lobe_registry: Dict[str, ILobe] = {}
        self.running = False
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # β:metrics_tracking(system_performance_monitoring)
        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "events_retried": 0,
            "average_processing_time": 0.0
        }
        
        logger.info(f"EventSystem initialized with {max_workers} workers")
    
    async def start(self) -> bool:
        """Ω:system_startup(event_processing_activation)"""
        try:
            if self.running:
                logger.warning("EventSystem already running")
                return True
            
            self.running = True
            
            # λ:worker_creation(async_event_processors)
            for i in range(self.max_workers):
                task = asyncio.create_task(self._event_processor(f"worker-{i}"))
                self.processing_tasks.add(task)
            
            logger.info("EventSystem started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start EventSystem: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Ω:system_shutdown(graceful_termination)"""
        try:
            self.running = False
            
            # i:graceful_shutdown(task_completion_wait)
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            self.executor.shutdown(wait=True)
            logger.info("EventSystem stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to stop EventSystem: {str(e)}")
            return False
    
    def register_lobe(self, lobe: ILobe) -> bool:
        """ℵ:lobe_registration(communication_participant_enrollment)"""
        try:
            lobe_status = lobe.get_status()
            lobe_id = lobe_status["lobe_id"]
            
            if lobe_id in self.lobe_registry:
                logger.warning(f"Lobe {lobe_id} already registered")
                return True
            
            self.lobe_registry[lobe_id] = lobe
            logger.info(f"Registered lobe: {lobe_status['name']} ({lobe_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to register lobe: {str(e)}")
            return False
    
    def unregister_lobe(self, lobe_id: str) -> bool:
        """ℵ:lobe_deregistration(communication_participant_removal)"""
        try:
            if lobe_id not in self.lobe_registry:
                logger.warning(f"Lobe {lobe_id} not registered")
                return True
            
            del self.lobe_registry[lobe_id]
            logger.info(f"Unregistered lobe: {lobe_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister lobe: {str(e)}")
            return False
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> bool:
        """β:subscription_management(event_listener_registration)"""
        try:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = set()
            
            self.subscribers[event_type].add(callback)
            logger.info(f"Subscribed callback to {event_type.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {event_type.value}: {str(e)}")
            return False
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> bool:
        """β:unsubscription_management(event_listener_removal)"""
        try:
            if event_type in self.subscribers:
                self.subscribers[event_type].discard(callback)
                logger.info(f"Unsubscribed callback from {event_type.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {event_type.value}: {str(e)}")
            return False
    
    async def publish_event(self, event: Event) -> bool:
        """Δ:event_publication(message_distribution)"""
        try:
            if not self.running:
                logger.error("EventSystem not running")
                return False
            
            await self.event_queue.put(event)
            logger.debug(f"Published event {event.event_id} of type {event.event_type.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {str(e)}")
            return False
    
    async def send_lobe_message(self, source_lobe_id: str, target_lobe_id: str, 
                               message: Any, priority: EventPriority = EventPriority.NORMAL) -> bool:
        """τ:lobe_messaging(direct_inter_lobe_communication)"""
        event = Event(
            event_type=EventType.LOBE_MESSAGE,
            source_lobe_id=source_lobe_id,
            target_lobe_id=target_lobe_id,
            data={"message": message},
            priority=priority
        )
        return await self.publish_event(event)
    
    async def broadcast_event(self, source_lobe_id: str, event_type: EventType, 
                             data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL) -> bool:
        """Δ:event_broadcasting(system_wide_notification)"""
        event = Event(
            event_type=event_type,
            source_lobe_id=source_lobe_id,
            data=data,
            priority=priority
        )
        return await self.publish_event(event)
    
    async def _event_processor(self, worker_id: str) -> None:
        """λ:event_processing(async_message_handler)"""
        logger.info(f"Event processor {worker_id} started")
        
        while self.running:
            try:
                # τ:event_retrieval(queue_message_extraction)
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                start_time = datetime.now()
                success = await self._process_event(event)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # β:metrics_update(performance_tracking)
                self._update_metrics(success, processing_time)
                
                if not success and event.retry_count < event.max_retries:
                    event.retry_count += 1
                    await self.event_queue.put(event)
                    self.metrics["events_retried"] += 1
                    logger.warning(f"Retrying event {event.event_id} (attempt {event.retry_count})")
                
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processor {worker_id} error: {str(e)}")
        
        logger.info(f"Event processor {worker_id} stopped")
    
    async def _process_event(self, event: Event) -> bool:
        """i:event_processing(individual_message_handling)"""
        try:
            # Δ:targeted_delivery(specific_lobe_messaging)
            if event.target_lobe_id and event.target_lobe_id in self.lobe_registry:
                target_lobe = self.lobe_registry[event.target_lobe_id]
                success = target_lobe.receive_message(event.source_lobe_id, event.data.get("message"))
                if not success:
                    logger.error(f"Failed to deliver message to lobe {event.target_lobe_id}")
                    return False
            
            # β:subscriber_notification(callback_execution)
            if event.event_type in self.subscribers:
                for callback in self.subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Subscriber callback failed: {str(e)}")
            
            event.processed = True
            logger.debug(f"Successfully processed event {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id}: {str(e)}")
            return False
    
    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """β:metrics_management(performance_statistics_update)"""
        if success:
            self.metrics["events_processed"] += 1
        else:
            self.metrics["events_failed"] += 1
        
        # i:average_calculation(rolling_performance_metric)
        total_events = self.metrics["events_processed"] + self.metrics["events_failed"]
        if total_events > 0:
            current_avg = self.metrics["average_processing_time"]
            self.metrics["average_processing_time"] = (
                (current_avg * (total_events - 1) + processing_time) / total_events
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """β:metrics_retrieval(system_performance_statistics)"""
        return {
            **self.metrics,
            "queue_size": self.event_queue.qsize(),
            "registered_lobes": len(self.lobe_registry),
            "active_subscriptions": sum(len(subs) for subs in self.subscribers.values()),
            "running": self.running
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Ω:system_status(comprehensive_state_information)"""
        return {
            "running": self.running,
            "workers": self.max_workers,
            "metrics": self.get_metrics(),
            "registered_lobes": list(self.lobe_registry.keys()),
            "subscribed_events": list(self.subscribers.keys())
        }


# λ:module_initialization(event_system_singleton)
_event_system_instance: Optional[EventSystem] = None


def get_event_system() -> EventSystem:
    """Ω:singleton_access(global_event_system_instance)"""
    global _event_system_instance
    if _event_system_instance is None:
        _event_system_instance = EventSystem()
    return _event_system_instance


# τ:self_reference(event_system_metadata)
# {type:Implementation, file:"event_system.py", version:"1.0.0", 
#  checksum:"sha256:event_system_checksum", canonical_address:"event-system-lobe", 
#  pfsus_compliant:true, lambda_operators:true, 
#  dependencies:["ILobe", "MCPLobeError", "asyncio", "threading"]}

# Dependencies: @{CORE.INTERFACES.LOBE.001, CORE.EXCEPTIONS.001, CORE.ASYNC.001}
# Related: @{CORE.LOBES.MEMORY.001, CORE.LOBES.WORKFLOW.001, CORE.LOBES.CONTEXT.001}

# λ(Δ(β(τ(event_system_implementation_complete)))) Processing complete
# Version: 1.0.0 | Last Modified: 2025-07-22T00:00:00Z