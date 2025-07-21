"""
Self-Improvement Integration for MCP Neural Networks

This module integrates the self-improving neural system with the existing MCP
components, providing a unified interface for managing recursive self-improvement
of neural networks throughout the system.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .self_improving_neural_system import SelfImprovingNeuralSystem, ModelImprovementTask
from .hormone_neural_integration import HormoneNeuralIntegration
from .performance_tracker import PerformanceTracker
from ..genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem
from ..hormone_system_controller import HormoneSystemController
from ..brain_state_aggregator import BrainStateAggregator
from ..enhanced_mcp_integration import EnhancedMCPIntegration


@dataclass
class SelfImprovementConfig:
    """Configuration for self-improvement integration"""
    enabled: bool = True
    auto_start: bool = True
    improvement_interval: float = 300.0  # 5 minutes
    max_concurrent_improvements: int = 3
    performance_threshold: float = 0.1  # 10% improvement required
    hormone_driven: bool = True
    genetic_enhanced: bool = True
    cross_model_distillation: bool = True
    model_backup_enabled: bool = True
    improvement_history_size: int = 1000


class SelfImprovementIntegration:
    """
    Integration layer for self-improving neural networks in the MCP system.
    
    This class provides a unified interface for managing the recursive self-improvement
    of neural networks across all MCP components.
    """
    
    def __init__(self, 
                 enhanced_mcp: Optional[EnhancedMCPIntegration] = None,
                 config: Optional[SelfImprovementConfig] = None):
        
        self.logger = logging.getLogger("SelfImprovementIntegration")
        
        # Configuration
        self.config = config or SelfImprovementConfig()
        
        # Core MCP integration
        self.enhanced_mcp = enhanced_mcp
        
        # Component references (will be set during initialization)
        self.hormone_integration: Optional[HormoneNeuralIntegration] = None
        self.genetic_system: Optional[IntegratedGeneticTriggerSystem] = None
        self.hormone_system: Optional[HormoneSystemController] = None
        self.brain_state: Optional[BrainStateAggregator] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # Self-improving system
        self.self_improving_system: Optional[SelfImprovingNeuralSystem] = None
        
        # Integration state
        self.initialized = False
        self.improvement_active = False
        self.integration_start_time: Optional[datetime] = None
        
        # Performance monitoring
        self.total_improvements = 0
        self.total_performance_gain = 0.0
        self.best_single_improvement = 0.0
        self.improvement_success_rate = 0.0
        
        self.logger.info("Self-Improvement Integration initialized")
    
    async def initialize(self):
        """Initialize the self-improvement integration with MCP components"""
        if self.initialized:
            self.logger.warning("Self-improvement integration already initialized")
            return
        
        try:
            # Get component references from enhanced MCP
            if self.enhanced_mcp:
                self.hormone_integration = getattr(self.enhanced_mcp, 'hormone_integration', None)
                self.genetic_system = getattr(self.enhanced_mcp, 'genetic_optimizer', None)
                self.hormone_system = getattr(self.enhanced_mcp, 'hormone_system', None)
                self.brain_state = getattr(self.enhanced_mcp, 'brain_state', None)
                self.performance_tracker = getattr(self.enhanced_mcp, 'performance_tracker', None)
            
            # Initialize self-improving system
            self.self_improving_system = SelfImprovingNeuralSystem(
                hormone_integration=self.hormone_integration,
                genetic_system=self.genetic_system,
                hormone_system=self.hormone_system,
                brain_state=self.brain_state
            )
            
            # Start self-improvement if auto-start is enabled
            if self.config.auto_start and self.config.enabled:
                await self.start_self_improvement()
            
            self.initialized = True
            self.integration_start_time = datetime.now()
            
            self.logger.info("Self-improvement integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing self-improvement integration: {e}")
            raise
    
    async def start_self_improvement(self):
        """Start the self-improvement process"""
        if not self.initialized:
            await self.initialize()
        
        if self.improvement_active:
            self.logger.warning("Self-improvement already active")
            return
        
        try:
            # Start the self-improving system
            await self.self_improving_system.start_self_improvement_loop()
            
            self.improvement_active = True
            self.logger.info("Self-improvement process started")
            
            # Trigger initial hormone response
            if self.hormone_system:
                self.hormone_system.release_hormone(
                    'self_improvement', 'dopamine', 0.2,
                    context={'action': 'start_improvement', 'timestamp': datetime.now().isoformat()}
                )
            
        except Exception as e:
            self.logger.error(f"Error starting self-improvement: {e}")
            raise
    
    async def stop_self_improvement(self):
        """Stop the self-improvement process"""
        if not self.improvement_active:
            self.logger.warning("Self-improvement not active")
            return
        
        try:
            # Stop the self-improving system
            await self.self_improving_system.stop_self_improvement_loop()
            
            self.improvement_active = False
            self.logger.info("Self-improvement process stopped")
            
            # Trigger hormone response for stopping
            if self.hormone_system:
                self.hormone_system.release_hormone(
                    'self_improvement', 'serotonin', 0.1,
                    context={'action': 'stop_improvement', 'timestamp': datetime.now().isoformat()}
                )
            
        except Exception as e:
            self.logger.error(f"Error stopping self-improvement: {e}")
            raise
    
    async def force_improvement_cycle(self):
        """Force an immediate improvement cycle"""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.self_improving_system.force_improvement_cycle()
            self.logger.info("Forced improvement cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in forced improvement cycle: {e}")
            raise
    
    async def add_custom_improvement_task(self, 
                                        model_id: str, 
                                        improvement_type: str,
                                        priority: float = 1.0,
                                        target_metrics: Optional[Dict[str, float]] = None,
                                        constraints: Optional[Dict[str, Any]] = None):
        """Add a custom improvement task"""
        if not self.initialized:
            await self.initialize()
        
        task = ModelImprovementTask(
            model_id=model_id,
            model_type=self._infer_model_type(model_id),
            improvement_type=improvement_type,
            priority=priority,
            target_metrics=target_metrics or {},
            constraints=constraints or {}
        )
        
        await self.self_improving_system.add_improvement_task(task)
        self.logger.info(f"Added custom improvement task: {model_id} ({improvement_type})")
    
    def _infer_model_type(self, model_id: str) -> str:
        """Infer model type from model ID"""
        if 'hormone_' in model_id:
            return 'hormone'
        elif 'pattern_' in model_id:
            return 'pattern'
        elif 'memory_' in model_id:
            return 'memory'
        elif 'genetic_' in model_id:
            return 'genetic'
        else:
            return 'generic'
    
    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get comprehensive improvement status"""
        if not self.initialized:
            return {'status': 'not_initialized'}
        
        # Get basic status
        basic_status = await self.self_improving_system.get_improvement_status()
        
        # Calculate additional metrics
        runtime = None
        if self.integration_start_time:
            runtime = (datetime.now() - self.integration_start_time).total_seconds()
        
        # Get hormone levels
        hormone_levels = {}
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_hormone_levels()
        
        # Get brain state
        brain_state = {}
        if self.brain_state:
            brain_state = self.brain_state.get_system_state()
        
        return {
            **basic_status,
            'integration_runtime': runtime,
            'hormone_levels': hormone_levels,
            'brain_state': brain_state,
            'config': {
                'enabled': self.config.enabled,
                'auto_start': self.config.auto_start,
                'improvement_interval': self.config.improvement_interval,
                'max_concurrent_improvements': self.config.max_concurrent_improvements,
                'performance_threshold': self.config.performance_threshold,
                'hormone_driven': self.config.hormone_driven,
                'genetic_enhanced': self.config.genetic_enhanced,
                'cross_model_distillation': self.config.cross_model_distillation
            }
        }
    
    async def get_improvement_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get improvement history"""
        if not self.initialized:
            return []
        
        history = await self.self_improving_system.get_improvement_history(limit)
        return [self._convert_metrics_to_dict(metrics) for metrics in history]
    
    def _convert_metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert metrics object to dictionary"""
        return {
            'iteration': metrics.iteration,
            'models_improved': metrics.models_improved,
            'total_improvement': metrics.total_improvement,
            'best_performance_gain': metrics.best_performance_gain,
            'average_training_time': metrics.average_training_time,
            'hormone_levels': metrics.hormone_levels,
            'genetic_activations': metrics.genetic_activations,
            'memory_usage': metrics.memory_usage,
            'timestamp': metrics.timestamp.isoformat()
        }
    
    async def get_model_performance(self, model_id: str) -> List[float]:
        """Get performance history for a specific model"""
        if not self.initialized:
            return []
        
        return await self.self_improving_system.get_model_performance(model_id)
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models for improvement"""
        if not self.initialized:
            return []
        
        return await self.self_improving_system.get_available_models()
    
    async def optimize_specific_model(self, model_id: str, optimization_type: str = 'auto'):
        """Optimize a specific model"""
        if not self.initialized:
            await self.initialize()
        
        # Determine optimization type if auto
        if optimization_type == 'auto':
            performance = await self._analyze_model_performance(model_id)
            optimization_type = self._determine_optimization_type(performance)
        
        # Create optimization task
        await self.add_custom_improvement_task(
            model_id=model_id,
            improvement_type=optimization_type,
            priority=2.0  # Higher priority for manual optimization
        )
        
        self.logger.info(f"Started optimization of model {model_id} with type {optimization_type}")
    
    async def _analyze_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Analyze performance of a specific model"""
        performance_history = await self.get_model_performance(model_id)
        
        if not performance_history:
            return {'accuracy': 0.5, 'needs_improvement': True}
        
        recent_performance = performance_history[-10:]  # Last 10 measurements
        avg_accuracy = sum(recent_performance) / len(recent_performance)
        
        return {
            'accuracy': avg_accuracy,
            'needs_improvement': avg_accuracy < 0.8,
            'trend': 'improving' if len(recent_performance) > 1 and recent_performance[-1] > recent_performance[0] else 'stable'
        }
    
    def _determine_optimization_type(self, performance: Dict[str, Any]) -> str:
        """Determine optimization type based on performance"""
        accuracy = performance.get('accuracy', 0.5)
        
        if accuracy < 0.6:
            return 'training'  # Need better training
        elif accuracy < 0.8:
            return 'architecture'  # Need architecture improvements
        else:
            return 'optimization'  # Need optimization
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        if not self.initialized:
            return {'status': 'not_initialized', 'health': 'unknown'}
        
        try:
            # Get improvement status
            status = await self.get_improvement_status()
            
            # Get recent history
            recent_history = await self.get_improvement_history(10)
            
            # Calculate health metrics
            health_metrics = self._calculate_health_metrics(status, recent_history)
            
            # Get hormone balance
            hormone_balance = {}
            if self.hormone_system:
                hormone_levels = self.hormone_system.get_hormone_levels()
                hormone_balance = self._analyze_hormone_balance(hormone_levels)
            
            return {
                'status': 'healthy' if health_metrics['overall_health'] > 0.7 else 'degraded',
                'health_metrics': health_metrics,
                'hormone_balance': hormone_balance,
                'improvement_status': status,
                'recent_activity': recent_history,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_health_metrics(self, status: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate system health metrics"""
        metrics = {
            'overall_health': 0.5,
            'improvement_efficiency': 0.5,
            'system_stability': 0.5,
            'learning_rate': 0.5
        }
        
        # Calculate improvement efficiency
        if history:
            recent_improvements = [h['total_improvement'] for h in history if h['total_improvement'] > 0]
            if recent_improvements:
                metrics['improvement_efficiency'] = min(1.0, sum(recent_improvements) / len(recent_improvements))
        
        # Calculate system stability
        if status.get('iteration', 0) > 0:
            metrics['system_stability'] = min(1.0, status.get('total_improvements', 0) / status.get('iteration', 1))
        
        # Calculate learning rate
        if history and len(history) > 1:
            recent_gains = [h['best_performance_gain'] for h in history[-5:] if h['best_performance_gain'] > 0]
            if recent_gains:
                metrics['learning_rate'] = min(1.0, sum(recent_gains) / len(recent_gains))
        
        # Calculate overall health
        metrics['overall_health'] = (
            metrics['improvement_efficiency'] * 0.4 +
            metrics['system_stability'] * 0.3 +
            metrics['learning_rate'] * 0.3
        )
        
        return metrics
    
    def _analyze_hormone_balance(self, hormone_levels: Dict[str, float]) -> Dict[str, Any]:
        """Analyze hormone balance for system health"""
        balance = {
            'overall_balance': 'normal',
            'stress_level': 'low',
            'learning_capacity': 'high',
            'stability': 'good'
        }
        
        # Analyze stress level (cortisol)
        cortisol = hormone_levels.get('cortisol', 0.3)
        if cortisol > 0.7:
            balance['stress_level'] = 'high'
        elif cortisol > 0.5:
            balance['stress_level'] = 'moderate'
        
        # Analyze learning capacity (dopamine + growth hormone)
        dopamine = hormone_levels.get('dopamine', 0.5)
        growth_hormone = hormone_levels.get('growth_hormone', 0.5)
        learning_score = (dopamine + growth_hormone) / 2
        
        if learning_score > 0.7:
            balance['learning_capacity'] = 'high'
        elif learning_score > 0.5:
            balance['learning_capacity'] = 'moderate'
        else:
            balance['learning_capacity'] = 'low'
        
        # Analyze stability (serotonin)
        serotonin = hormone_levels.get('serotonin', 0.5)
        if serotonin > 0.7:
            balance['stability'] = 'excellent'
        elif serotonin > 0.5:
            balance['stability'] = 'good'
        else:
            balance['stability'] = 'poor'
        
        # Overall balance
        if balance['stress_level'] == 'high' or balance['stability'] == 'poor':
            balance['overall_balance'] = 'poor'
        elif balance['learning_capacity'] == 'high' and balance['stability'] == 'excellent':
            balance['overall_balance'] = 'excellent'
        else:
            balance['overall_balance'] = 'normal'
        
        return balance
    
    async def configure_improvement(self, config_updates: Dict[str, Any]):
        """Update improvement configuration"""
        # Update config
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update self-improving system config if it exists
        if self.self_improving_system:
            for key, value in config_updates.items():
                if key in self.self_improving_system.config:
                    self.self_improving_system.config[key] = value
        
        self.logger.info(f"Updated improvement configuration: {config_updates}")
    
    async def export_improvement_data(self) -> Dict[str, Any]:
        """Export improvement data for analysis"""
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        try:
            status = await self.get_improvement_status()
            history = await self.get_improvement_history(100)
            health_report = await self.get_system_health_report()
            
            return {
                'export_timestamp': datetime.now().isoformat(),
                'status': status,
                'history': history,
                'health_report': health_report,
                'available_models': await self.get_available_models(),
                'configuration': {
                    'enabled': self.config.enabled,
                    'auto_start': self.config.auto_start,
                    'improvement_interval': self.config.improvement_interval,
                    'max_concurrent_improvements': self.config.max_concurrent_improvements,
                    'performance_threshold': self.config.performance_threshold,
                    'hormone_driven': self.config.hormone_driven,
                    'genetic_enhanced': self.config.genetic_enhanced,
                    'cross_model_distillation': self.config.cross_model_distillation
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting improvement data: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.improvement_active:
                await self.stop_self_improvement()
            
            self.initialized = False
            self.logger.info("Self-improvement integration cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Convenience functions for easy integration
async def create_self_improvement_integration(enhanced_mcp: EnhancedMCPIntegration,
                                            config: Optional[SelfImprovementConfig] = None) -> SelfImprovementIntegration:
    """Create and initialize a self-improvement integration"""
    integration = SelfImprovementIntegration(enhanced_mcp, config)
    await integration.initialize()
    return integration


async def start_self_improvement_for_mcp(enhanced_mcp: EnhancedMCPIntegration):
    """Quick start function for self-improvement in MCP"""
    config = SelfImprovementConfig(
        enabled=True,
        auto_start=True,
        improvement_interval=300.0,
        hormone_driven=True,
        genetic_enhanced=True
    )
    
    integration = await create_self_improvement_integration(enhanced_mcp, config)
    return integration 