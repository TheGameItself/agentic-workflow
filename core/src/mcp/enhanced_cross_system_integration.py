"""
Enhanced Cross-System Integration

This module provides advanced coordination and integration between all
brain-inspired components of the MCP system, including hormone system,
neural networks, genetic triggers, memory systems, and cognitive engines.

Features:
- Advanced cross-lobe communication and coordination
- Hormone-triggered system-wide optimizations
- Neural network performance integration
- Genetic trigger system coordination
- Memory system integration
- Cognitive engine coordination
- Real-time system state synchronization
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import defaultdict

from .hormone_system_controller import HormoneSystemController
from .neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
from .genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem
from .brain_state_aggregator import BrainStateAggregator
from .enhanced_monitoring_system import EnhancedMonitoringSystem
from .genetic_trigger_system.advanced_optimization import AdvancedGeneticOptimizer


class IntegrationEvent(Enum):
    """Types of integration events"""
    HORMONE_CASCADE = "hormone_cascade"
    NEURAL_SWITCH = "neural_switch"
    GENETIC_ACTIVATION = "genetic_activation"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    COGNITIVE_ENGINE_ACTIVATION = "cognitive_engine_activation"
    SYSTEM_OPTIMIZATION = "system_optimization"
    PERFORMANCE_ANOMALY = "performance_anomaly"


@dataclass
class SystemEvent:
    """System integration event"""
    event_type: IntegrationEvent
    source_component: str
    target_components: List[str]
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical


@dataclass
class CrossSystemState:
    """Comprehensive cross-system state"""
    timestamp: datetime
    hormone_levels: Dict[str, float]
    neural_performance: Dict[str, Dict[str, float]]
    genetic_triggers: Dict[str, Any]
    memory_state: Dict[str, Any]
    cognitive_engines: Dict[str, Any]
    lobe_states: Dict[str, Dict[str, Any]]
    system_health: float
    optimization_recommendations: List[str]


class EnhancedCrossSystemIntegration:
    """
    Enhanced cross-system integration coordinator.
    
    Provides advanced coordination between all brain-inspired components
    with real-time synchronization and optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EnhancedCrossSystemIntegration")
        
        # Core system components
        self.hormone_system: Optional[HormoneSystemController] = None
        self.neural_integration: Optional[HormoneNeuralIntegration] = None
        self.genetic_system: Optional[IntegratedGeneticTriggerSystem] = None
        self.brain_state: Optional[BrainStateAggregator] = None
        self.monitoring_system: Optional[EnhancedMonitoringSystem] = None
        self.genetic_optimizer: Optional[AdvancedGeneticOptimizer] = None
        
        # Event management
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[IntegrationEvent, List[Callable]] = defaultdict(list)
        self.event_history: List[SystemEvent] = []
        
        # State management
        self.current_state: Optional[CrossSystemState] = None
        self.state_history: List[CrossSystemState] = []
        self.state_update_interval = 1.0  # seconds
        
        # Coordination parameters
        self.coordination_active = False
        self.optimization_interval = 30.0  # seconds
        self.synchronization_interval = 5.0  # seconds
        
        # Performance tracking
        self.integration_metrics: Dict[str, List[float]] = defaultdict(list)
        self.coordination_latency: List[float] = []
        
        # Callbacks for external systems
        self.state_callbacks: List[Callable] = []
        self.optimization_callbacks: List[Callable] = []
        
        self.logger.info("Enhanced Cross-System Integration initialized")
    
    def register_components(self,
                          hormone_system: HormoneSystemController,
                          neural_integration: HormoneNeuralIntegration,
                          genetic_system: IntegratedGeneticTriggerSystem,
                          brain_state: BrainStateAggregator,
                          monitoring_system: EnhancedMonitoringSystem,
                          genetic_optimizer: AdvancedGeneticOptimizer):
        """Register all system components"""
        self.hormone_system = hormone_system
        self.neural_integration = neural_integration
        self.genetic_system = genetic_system
        self.brain_state = brain_state
        self.monitoring_system = monitoring_system
        self.genetic_optimizer = genetic_optimizer
        
        self.logger.info("All system components registered")
    
    async def start_coordination(self):
        """Start cross-system coordination"""
        if self.coordination_active:
            return
        
        self.coordination_active = True
        
        # Start coordination tasks
        asyncio.create_task(self._coordination_loop())
        asyncio.create_task(self._state_synchronization_loop())
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._event_processing_loop())
        
        self.logger.info("Cross-system coordination started")
    
    async def stop_coordination(self):
        """Stop cross-system coordination"""
        self.coordination_active = False
        self.logger.info("Cross-system coordination stopped")
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while self.coordination_active:
            try:
                start_time = time.time()
                
                # Collect current system state
                state = await self._collect_cross_system_state()
                self.current_state = state
                
                # Store in history
                self.state_history.append(state)
                if len(self.state_history) > 1000:
                    self.state_history = self.state_history[-1000:]
                
                # Analyze system interactions
                await self._analyze_system_interactions(state)
                
                # Generate optimization recommendations
                recommendations = await self._generate_optimization_recommendations(state)
                
                # Trigger state callbacks
                await self._trigger_state_callbacks(state, recommendations)
                
                # Record coordination latency
                latency = time.time() - start_time
                self.coordination_latency.append(latency)
                if len(self.coordination_latency) > 100:
                    self.coordination_latency = self.coordination_latency[-100:]
                
                await asyncio.sleep(self.state_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(self.state_update_interval)
    
    async def _state_synchronization_loop(self):
        """State synchronization loop"""
        while self.coordination_active:
            try:
                # Synchronize state across all components
                await self._synchronize_component_states()
                
                await asyncio.sleep(self.synchronization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in state synchronization: {e}")
                await asyncio.sleep(self.synchronization_interval)
    
    async def _optimization_loop(self):
        """System optimization loop"""
        while self.coordination_active:
            try:
                # Perform system-wide optimizations
                await self._perform_system_optimizations()
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _event_processing_loop(self):
        """Event processing loop"""
        while self.coordination_active:
            try:
                # Process events from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in event processing: {e}")
    
    async def _collect_cross_system_state(self) -> CrossSystemState:
        """Collect comprehensive cross-system state"""
        timestamp = datetime.now()
        
        # Collect hormone levels
        hormone_levels = {}
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_levels()
        
        # Collect neural performance
        neural_performance = {}
        if self.neural_integration:
            status = self.neural_integration.get_system_status()
            neural_performance = {
                'implementations': status.get('current_implementations', {}),
                'performance_summary': status.get('performance_summary', {}),
                'models_available': status.get('neural_models_available', 0)
            }
        
        # Collect genetic trigger state
        genetic_triggers = {}
        if self.genetic_system:
            genetic_triggers = {
                'active_triggers': list(self.genetic_system.active_triggers),
                'total_triggers': len(self.genetic_system.triggers),
                'generation': self.genetic_system.generation,
                'success_rate': self.genetic_system.successful_activations / max(1, self.genetic_system.total_activations)
            }
        
        # Collect memory state
        memory_state = {}
        if self.brain_state:
            memory_state = {
                'working_memory_size': len(getattr(self.brain_state, 'working_memory', {}).get('items', [])),
                'short_term_memory_size': len(getattr(self.brain_state, 'short_term_memory', {}).get('items', [])),
                'long_term_memory_size': len(getattr(self.brain_state, 'long_term_memory', {}).get('items', []))
            }
        
        # Collect cognitive engine states
        cognitive_engines = {}
        if self.brain_state:
            cognitive_engines = {
                'dreaming_engine': getattr(self.brain_state, 'dreaming_engine', {}).get('active', False),
                'scientific_engine': getattr(self.brain_state, 'scientific_engine', {}).get('active', False),
                'hypothetical_engine': getattr(self.brain_state, 'hypothetical_engine', {}).get('active', False)
            }
        
        # Collect lobe states
        lobe_states = {}
        if self.brain_state and hasattr(self.brain_state, 'lobes'):
            for lobe_name, lobe in self.brain_state.lobes.items():
                lobe_states[lobe_name] = {
                    'active': getattr(lobe, 'active', False),
                    'hormone_levels': getattr(lobe, 'local_hormone_levels', {}),
                    'performance': getattr(lobe, 'performance_metrics', {})
                }
        
        # Calculate system health
        system_health = self._calculate_system_health(
            hormone_levels, neural_performance, genetic_triggers, memory_state
        )
        
        return CrossSystemState(
            timestamp=timestamp,
            hormone_levels=hormone_levels,
            neural_performance=neural_performance,
            genetic_triggers=genetic_triggers,
            memory_state=memory_state,
            cognitive_engines=cognitive_engines,
            lobe_states=lobe_states,
            system_health=system_health,
            optimization_recommendations=[]
        )
    
    def _calculate_system_health(self, hormone_levels: Dict[str, float],
                               neural_performance: Dict[str, Any],
                               genetic_triggers: Dict[str, Any],
                               memory_state: Dict[str, Any]) -> float:
        """Calculate overall system health"""
        health_components = []
        
        # Hormone system health
        if hormone_levels:
            hormone_balance = 1.0 - np.std(list(hormone_levels.values()))
            health_components.append(max(0.0, hormone_balance))
        
        # Neural system health
        if neural_performance:
            neural_score = neural_performance.get('performance_summary', {}).get('overall_score', 0.5)
            health_components.append(neural_score)
        
        # Genetic system health
        if genetic_triggers:
            success_rate = genetic_triggers.get('success_rate', 0.5)
            health_components.append(success_rate)
        
        # Memory system health
        if memory_state:
            memory_efficiency = 1.0 - (memory_state.get('working_memory_size', 0) / 1000.0)
            health_components.append(max(0.0, memory_efficiency))
        
        return np.mean(health_components) if health_components else 0.5
    
    async def _analyze_system_interactions(self, state: CrossSystemState):
        """Analyze interactions between system components"""
        
        # Check for hormone-triggered neural switches
        await self._check_hormone_neural_interactions(state)
        
        # Check for genetic trigger activations
        await self._check_genetic_trigger_interactions(state)
        
        # Check for memory consolidation triggers
        await self._check_memory_interactions(state)
        
        # Check for cognitive engine activations
        await self._check_cognitive_engine_interactions(state)
    
    async def _check_hormone_neural_interactions(self, state: CrossSystemState):
        """Check for hormone-triggered neural network switches"""
        if not self.hormone_system or not self.neural_integration:
            return
        
        # Check for high cortisol levels (stress response)
        cortisol_level = state.hormone_levels.get('cortisol', 0.0)
        if cortisol_level > 0.7:
            # High stress - consider switching to algorithmic implementations
            event = SystemEvent(
                event_type=IntegrationEvent.NEURAL_SWITCH,
                source_component='hormone_system',
                target_components=['neural_integration'],
                data={'reason': 'high_cortisol', 'level': cortisol_level},
                timestamp=datetime.now(),
                priority=3
            )
            await self.event_queue.put(event)
        
        # Check for high dopamine levels (reward response)
        dopamine_level = state.hormone_levels.get('dopamine', 0.0)
        if dopamine_level > 0.8:
            # High reward - consider switching to neural implementations
            event = SystemEvent(
                event_type=IntegrationEvent.NEURAL_SWITCH,
                source_component='hormone_system',
                target_components=['neural_integration'],
                data={'reason': 'high_dopamine', 'level': dopamine_level},
                timestamp=datetime.now(),
                priority=2
            )
            await self.event_queue.put(event)
    
    async def _check_genetic_trigger_interactions(self, state: CrossSystemState):
        """Check for genetic trigger interactions"""
        if not self.genetic_system:
            return
        
        # Check for genetic trigger activations
        active_triggers = state.genetic_triggers.get('active_triggers', [])
        if active_triggers:
            event = SystemEvent(
                event_type=IntegrationEvent.GENETIC_ACTIVATION,
                source_component='genetic_system',
                target_components=['hormone_system', 'neural_integration'],
                data={'active_triggers': active_triggers},
                timestamp=datetime.now(),
                priority=2
            )
            await self.event_queue.put(event)
    
    async def _check_memory_interactions(self, state: CrossSystemState):
        """Check for memory system interactions"""
        if not self.brain_state:
            return
        
        # Check for memory consolidation needs
        working_memory_size = state.memory_state.get('working_memory_size', 0)
        if working_memory_size > 800:  # High working memory usage
            event = SystemEvent(
                event_type=IntegrationEvent.MEMORY_CONSOLIDATION,
                source_component='brain_state',
                target_components=['memory_system'],
                data={'working_memory_size': working_memory_size},
                timestamp=datetime.now(),
                priority=2
            )
            await self.event_queue.put(event)
    
    async def _check_cognitive_engine_interactions(self, state: CrossSystemState):
        """Check for cognitive engine interactions"""
        if not self.brain_state:
            return
        
        # Check for cognitive engine activations
        active_engines = []
        for engine, active in state.cognitive_engines.items():
            if active:
                active_engines.append(engine)
        
        if active_engines:
            event = SystemEvent(
                event_type=IntegrationEvent.COGNITIVE_ENGINE_ACTIVATION,
                source_component='brain_state',
                target_components=['hormone_system', 'memory_system'],
                data={'active_engines': active_engines},
                timestamp=datetime.now(),
                priority=2
            )
            await self.event_queue.put(event)
    
    async def _process_event(self, event: SystemEvent):
        """Process a system integration event"""
        self.event_history.append(event)
        
        # Limit event history
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        # Handle event based on type
        if event.event_type == IntegrationEvent.NEURAL_SWITCH:
            await self._handle_neural_switch_event(event)
        elif event.event_type == IntegrationEvent.GENETIC_ACTIVATION:
            await self._handle_genetic_activation_event(event)
        elif event.event_type == IntegrationEvent.MEMORY_CONSOLIDATION:
            await self._handle_memory_consolidation_event(event)
        elif event.event_type == IntegrationEvent.COGNITIVE_ENGINE_ACTIVATION:
            await self._handle_cognitive_engine_event(event)
        
        # Trigger event handlers
        for handler in self.event_handlers[event.event_type]:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    async def _handle_neural_switch_event(self, event: SystemEvent):
        """Handle neural network switch events"""
        reason = event.data.get('reason', 'unknown')
        
        if reason == 'high_cortisol':
            # Switch to algorithmic implementations for stability
            if self.neural_integration:
                # Force algorithmic implementations
                for hormone in self.neural_integration.current_implementations:
                    self.neural_integration.current_implementations[hormone] = 'algorithmic'
                
                self.logger.info("Switched to algorithmic implementations due to high cortisol")
        
        elif reason == 'high_dopamine':
            # Consider switching to neural implementations
            if self.neural_integration:
                # Allow neural implementations where beneficial
                for hormone in self.neural_integration.current_implementations:
                    # Check if neural implementation is available and performing well
                    if hormone in self.neural_integration.neural_models:
                        self.neural_integration.current_implementations[hormone] = 'neural'
                
                self.logger.info("Switched to neural implementations due to high dopamine")
    
    async def _handle_genetic_activation_event(self, event: SystemEvent):
        """Handle genetic trigger activation events"""
        active_triggers = event.data.get('active_triggers', [])
        
        # Trigger hormone cascades for genetic activations
        if self.hormone_system:
            for trigger_id in active_triggers:
                # Release growth hormone for genetic adaptation
                self.hormone_system.release_hormone('growth_hormone', 0.1)
                
                # Release vasopressin for memory consolidation
                self.hormone_system.release_hormone('vasopressin', 0.05)
        
        self.logger.info(f"Handled genetic trigger activation for {len(active_triggers)} triggers")
    
    async def _handle_memory_consolidation_event(self, event: SystemEvent):
        """Handle memory consolidation events"""
        working_memory_size = event.data.get('working_memory_size', 0)
        
        # Trigger memory consolidation
        if self.brain_state and hasattr(self.brain_state, 'consolidate_memory'):
            await self.brain_state.consolidate_memory()
        
        # Release vasopressin for memory consolidation
        if self.hormone_system:
            self.hormone_system.release_hormone('vasopressin', 0.1)
        
        self.logger.info(f"Triggered memory consolidation for {working_memory_size} items")
    
    async def _handle_cognitive_engine_event(self, event: SystemEvent):
        """Handle cognitive engine activation events"""
        active_engines = event.data.get('active_engines', [])
        
        # Release acetylcholine for cognitive activity
        if self.hormone_system:
            self.hormone_system.release_hormone('acetylcholine', 0.1)
        
        # Release norepinephrine for attention
        if self.hormone_system:
            self.hormone_system.release_hormone('norepinephrine', 0.05)
        
        self.logger.info(f"Handled cognitive engine activation for {active_engines}")
    
    async def _synchronize_component_states(self):
        """Synchronize state across all components"""
        if not self.current_state:
            return
        
        # Synchronize hormone levels across components
        if self.hormone_system and self.brain_state:
            hormone_levels = self.current_state.hormone_levels
            for lobe_name, lobe_state in self.current_state.lobe_states.items():
                if lobe_name in self.brain_state.lobes:
                    lobe = self.brain_state.lobes[lobe_name]
                    if hasattr(lobe, 'local_hormone_levels'):
                        lobe.local_hormone_levels.update(hormone_levels)
        
        # Synchronize neural performance data
        if self.neural_integration and self.monitoring_system:
            neural_performance = self.current_state.neural_performance
            # Update monitoring system with neural performance data
            if hasattr(self.monitoring_system, 'update_neural_performance'):
                self.monitoring_system.update_neural_performance(neural_performance)
    
    async def _perform_system_optimizations(self):
        """Perform system-wide optimizations"""
        if not self.current_state:
            return
        
        # Generate optimization recommendations
        recommendations = await self._generate_optimization_recommendations(self.current_state)
        
        # Apply optimizations
        for recommendation in recommendations:
            await self._apply_optimization_recommendation(recommendation)
    
    async def _generate_optimization_recommendations(self, state: CrossSystemState) -> List[str]:
        """Generate optimization recommendations based on current state"""
        recommendations = []
        
        # Check system health
        if state.system_health < 0.6:
            recommendations.append("System health is low - consider performance optimization")
        
        # Check hormone balance
        hormone_levels = state.hormone_levels
        if hormone_levels:
            cortisol_level = hormone_levels.get('cortisol', 0.0)
            if cortisol_level > 0.8:
                recommendations.append("High cortisol levels - consider stress reduction")
            
            dopamine_level = hormone_levels.get('dopamine', 0.0)
            if dopamine_level < 0.3:
                recommendations.append("Low dopamine levels - consider reward optimization")
        
        # Check neural performance
        neural_performance = state.neural_performance
        if neural_performance:
            overall_score = neural_performance.get('performance_summary', {}).get('overall_score', 0.5)
            if overall_score < 0.5:
                recommendations.append("Neural performance is low - consider model retraining")
        
        # Check genetic trigger performance
        genetic_triggers = state.genetic_triggers
        if genetic_triggers:
            success_rate = genetic_triggers.get('success_rate', 0.5)
            if success_rate < 0.4:
                recommendations.append("Genetic trigger success rate is low - consider optimization")
        
        # Check memory usage
        memory_state = state.memory_state
        if memory_state:
            working_memory_size = memory_state.get('working_memory_size', 0)
            if working_memory_size > 900:
                recommendations.append("High working memory usage - consider consolidation")
        
        return recommendations
    
    async def _apply_optimization_recommendation(self, recommendation: str):
        """Apply an optimization recommendation"""
        if "stress reduction" in recommendation.lower():
            # Reduce cortisol levels
            if self.hormone_system:
                self.hormone_system.release_hormone('serotonin', 0.1)
                self.hormone_system.release_hormone('gaba', 0.05)
        
        elif "reward optimization" in recommendation.lower():
            # Increase dopamine levels
            if self.hormone_system:
                self.hormone_system.release_hormone('dopamine', 0.1)
        
        elif "model retraining" in recommendation.lower():
            # Trigger neural model retraining
            if self.neural_integration:
                # This would trigger background retraining
                pass
        
        elif "genetic optimization" in recommendation.lower():
            # Trigger genetic optimization
            if self.genetic_optimizer:
                # This would trigger genetic trigger optimization
                pass
        
        elif "memory consolidation" in recommendation.lower():
            # Trigger memory consolidation
            if self.brain_state and hasattr(self.brain_state, 'consolidate_memory'):
                await self.brain_state.consolidate_memory()
        
        self.logger.info(f"Applied optimization: {recommendation}")
    
    async def _trigger_state_callbacks(self, state: CrossSystemState, recommendations: List[str]):
        """Trigger callbacks for state updates"""
        for callback in self.state_callbacks:
            try:
                await callback(state, recommendations)
            except Exception as e:
                self.logger.error(f"Error in state callback: {e}")
    
    def add_event_handler(self, event_type: IntegrationEvent, handler: Callable):
        """Add event handler for specific event type"""
        self.event_handlers[event_type].append(handler)
    
    def add_state_callback(self, callback: Callable):
        """Add callback for state updates"""
        self.state_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable):
        """Add callback for optimization events"""
        self.optimization_callbacks.append(callback)
    
    def get_current_state(self) -> Optional[CrossSystemState]:
        """Get current cross-system state"""
        return self.current_state
    
    def get_event_history(self, event_type: Optional[IntegrationEvent] = None) -> List[SystemEvent]:
        """Get event history, optionally filtered by type"""
        if event_type:
            return [e for e in self.event_history if e.event_type == event_type]
        return self.event_history
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        return {
            'coordination_latency': {
                'average': np.mean(self.coordination_latency) if self.coordination_latency else 0.0,
                'max': max(self.coordination_latency) if self.coordination_latency else 0.0,
                'min': min(self.coordination_latency) if self.coordination_latency else 0.0
            },
            'total_events_processed': len(self.event_history),
            'event_type_distribution': self._get_event_type_distribution(),
            'system_health_trend': self._get_system_health_trend()
        }
    
    def _get_event_type_distribution(self) -> Dict[str, int]:
        """Get distribution of event types"""
        distribution = defaultdict(int)
        for event in self.event_history:
            distribution[event.event_type.value] += 1
        return dict(distribution)
    
    def _get_system_health_trend(self) -> List[float]:
        """Get system health trend over time"""
        return [state.system_health for state in self.state_history[-50:]]
    
    def export_integration_data(self, format: str = 'json') -> str:
        """Export integration data"""
        if format == 'json':
            data = {
                'current_state': self.current_state.__dict__ if self.current_state else None,
                'event_history': [e.__dict__ for e in self.event_history[-100:]],
                'integration_metrics': self.get_integration_metrics(),
                'system_health_trend': self._get_system_health_trend()
            }
            return json.dumps(data, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}") 