#!/usr/bin/env python3
"""
Simple Demo: Self-Improving Neural Network System

This script demonstrates the core concepts of the self-improving neural network
system without complex dependencies.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MockNeuralModel:
    """Mock neural model for demonstration"""
    model_id: str
    accuracy: float = 0.7
    latency: float = 100.0
    parameters: int = 1000
    
    def train(self, training_data: List[tuple]) -> Dict[str, float]:
        """Mock training process"""
        # Simulate training improvement
        improvement = np.random.uniform(0.01, 0.05)
        self.accuracy = min(0.95, self.accuracy + improvement)
        self.latency = max(10.0, self.latency * 0.95)
        
        return {
            'accuracy': self.accuracy,
            'latency': self.latency,
            'improvement': improvement
        }
    
    def optimize(self) -> Dict[str, float]:
        """Mock optimization process"""
        # Simulate optimization improvement
        self.latency = max(5.0, self.latency * 0.9)
        return {
            'accuracy': self.accuracy,
            'latency': self.latency,
            'optimization_gain': 0.1
        }


@dataclass
class MockHormoneSystem:
    """Mock hormone system for demonstration"""
    hormone_levels: Dict[str, float] = field(default_factory=lambda: {
        'dopamine': 0.5,
        'serotonin': 0.5,
        'cortisol': 0.3,
        'adrenaline': 0.2
    })
    
    def get_hormone_levels(self) -> Dict[str, float]:
        return self.hormone_levels.copy()
    
    def release_hormone(self, source: str, hormone: str, quantity: float, context: Dict):
        """Mock hormone release"""
        self.hormone_levels[hormone] = min(1.0, self.hormone_levels.get(hormone, 0.0) + quantity)
        logger.info(f"Hormone released: {hormone} = {self.hormone_levels[hormone]:.3f}")


@dataclass
class MockGeneticSystem:
    """Mock genetic system for demonstration"""
    active_triggers: List[Dict] = field(default_factory=list)
    
    def get_active_triggers(self, environment: Dict) -> List[Dict]:
        """Mock genetic trigger activation"""
        if environment.get('system_load', 0.0) > 0.5:
            return [{'targets': ['neural_optimization'], 'priority': 0.8}]
        return []


class SimpleSelfImprovingSystem:
    """Simplified self-improving neural system for demonstration"""
    
    def __init__(self):
        self.logger = logging.getLogger("SimpleSelfImprovingSystem")
        
        # Mock components
        self.hormone_system = MockHormoneSystem()
        self.genetic_system = MockGeneticSystem()
        
        # Neural models
        self.neural_models: Dict[str, MockNeuralModel] = {
            'hormone_dopamine': MockNeuralModel('hormone_dopamine', 0.75, 80.0),
            'hormone_serotonin': MockNeuralModel('hormone_serotonin', 0.72, 85.0),
            'hormone_cortisol': MockNeuralModel('hormone_cortisol', 0.68, 90.0),
            'pattern_recognition': MockNeuralModel('pattern_recognition', 0.80, 60.0),
            'memory_consolidation': MockNeuralModel('memory_consolidation', 0.78, 70.0)
        }
        
        # Improvement tracking
        self.iteration = 0
        self.total_improvements = 0
        self.improvement_history: List[Dict] = []
        self.improvement_active = False
        
        self.logger.info("Simple Self-Improving System initialized")
    
    async def start_self_improvement(self):
        """Start the self-improvement process"""
        self.improvement_active = True
        self.logger.info("Self-improvement process started")
        
        # Trigger initial hormone response
        self.hormone_system.release_hormone(
            'self_improvement', 'dopamine', 0.2,
            context={'action': 'start_improvement'}
        )
    
    async def stop_self_improvement(self):
        """Stop the self-improvement process"""
        self.improvement_active = False
        self.logger.info("Self-improvement process stopped")
        
        # Trigger hormone response for stopping
        self.hormone_system.release_hormone(
            'self_improvement', 'serotonin', 0.1,
            context={'action': 'stop_improvement'}
        )
    
    async def run_improvement_cycle(self):
        """Run a single improvement cycle"""
        self.iteration += 1
        self.logger.info(f"Starting improvement cycle {self.iteration}")
        
        # Generate improvement tasks
        tasks = self._generate_improvement_tasks()
        
        # Execute tasks
        completed_tasks = []
        for task in tasks:
            try:
                result = await self._execute_improvement_task(task)
                if result:
                    completed_tasks.append(result)
            except Exception as e:
                self.logger.error(f"Error executing task {task['model_id']}: {e}")
        
        # Update metrics
        self._update_improvement_metrics(completed_tasks)
        
        # Trigger hormone responses
        if completed_tasks:
            self._trigger_improvement_hormones(len(completed_tasks))
        
        self.logger.info(f"Completed improvement cycle {self.iteration} with {len(completed_tasks)} improvements")
        return completed_tasks
    
    def _generate_improvement_tasks(self) -> List[Dict]:
        """Generate improvement tasks based on current model performance"""
        tasks = []
        
        for model_id, model in self.neural_models.items():
            # Determine if model needs improvement
            needs_improvement = model.accuracy < 0.85 or model.latency > 50.0
            
            if needs_improvement:
                # Determine improvement type
                if model.accuracy < 0.75:
                    improvement_type = 'training'
                elif model.latency > 80.0:
                    improvement_type = 'optimization'
                else:
                    improvement_type = 'architecture'
                
                # Calculate priority based on hormone levels
                hormone_levels = self.hormone_system.get_hormone_levels()
                base_priority = 1.0
                
                # High dopamine increases priority
                if hormone_levels.get('dopamine', 0.5) > 0.6:
                    base_priority *= 1.5
                
                # High cortisol increases priority for optimization
                if hormone_levels.get('cortisol', 0.3) > 0.5 and improvement_type == 'optimization':
                    base_priority *= 1.3
                
                task = {
                    'model_id': model_id,
                    'improvement_type': improvement_type,
                    'priority': base_priority,
                    'current_accuracy': model.accuracy,
                    'current_latency': model.latency
                }
                
                tasks.append(task)
        
        # Sort by priority
        tasks.sort(key=lambda x: x['priority'], reverse=True)
        return tasks[:3]  # Limit to 3 tasks per cycle
    
    async def _execute_improvement_task(self, task: Dict) -> Optional[Dict]:
        """Execute a specific improvement task"""
        model_id = task['model_id']
        improvement_type = task['improvement_type']
        
        self.logger.info(f"Executing improvement task: {model_id} ({improvement_type})")
        
        model = self.neural_models[model_id]
        start_time = time.time()
        
        try:
            if improvement_type == 'training':
                # Generate mock training data
                training_data = self._generate_training_data(model_id)
                result = model.train(training_data)
                performance_gain = result['improvement']
                
            elif improvement_type == 'optimization':
                result = model.optimize()
                performance_gain = result['optimization_gain']
                
            else:  # architecture
                # Mock architecture improvement
                model.accuracy = min(0.95, model.accuracy + 0.02)
                model.latency = max(10.0, model.latency * 0.95)
                result = {'accuracy': model.accuracy, 'latency': model.latency}
                performance_gain = 0.02
            
            execution_time = time.time() - start_time
            
            return {
                'task_id': model_id,
                'improvement_type': improvement_type,
                'performance_gain': performance_gain,
                'execution_time': execution_time,
                'new_metrics': result
            }
            
        except Exception as e:
            self.logger.error(f"Error executing task {model_id}: {e}")
            return None
    
    def _generate_training_data(self, model_id: str) -> List[tuple]:
        """Generate mock training data"""
        # Generate 100 training samples
        training_data = []
        for _ in range(100):
            # Mock features (10-dimensional)
            features = np.random.random(10).tolist()
            # Mock target (0-1 range)
            target = np.random.random()
            training_data.append((features, target))
        
        return training_data
    
    def _update_improvement_metrics(self, completed_tasks: List[Dict]):
        """Update improvement metrics"""
        if not completed_tasks:
            return
        
        total_improvement = sum(task.get('performance_gain', 0) for task in completed_tasks)
        best_gain = max(task.get('performance_gain', 0) for task in completed_tasks)
        avg_execution_time = np.mean([task.get('execution_time', 0) for task in completed_tasks])
        
        # Get hormone levels
        hormone_levels = self.hormone_system.get_hormone_levels()
        
        metrics = {
            'iteration': self.iteration,
            'models_improved': len(completed_tasks),
            'total_improvement': total_improvement,
            'best_performance_gain': best_gain,
            'average_execution_time': avg_execution_time,
            'hormone_levels': hormone_levels,
            'timestamp': datetime.now().isoformat()
        }
        
        self.improvement_history.append(metrics)
        self.total_improvements += len(completed_tasks)
        
        self.logger.info(f"Improvement metrics: {len(completed_tasks)} models improved, "
                        f"total gain: {total_improvement:.4f}, best gain: {best_gain:.4f}")
    
    def _trigger_improvement_hormones(self, improvements_count: int):
        """Trigger hormone responses based on improvement success"""
        if improvements_count > 0:
            # Release dopamine for successful improvements
            self.hormone_system.release_hormone(
                'system_optimization', 'dopamine', 0.1 * improvements_count,
                context={'improvements': improvements_count, 'type': 'neural_optimization'}
            )
            
            # Release growth hormone for learning
            if improvements_count > 2:
                self.hormone_system.release_hormone(
                    'system_optimization', 'growth_hormone', 0.05 * improvements_count,
                    context={'improvements': improvements_count, 'type': 'neural_optimization'}
                )
    
    def get_improvement_status(self) -> Dict[str, Any]:
        """Get improvement status"""
        return {
            'iteration': self.iteration,
            'total_improvements': self.total_improvements,
            'improvement_active': self.improvement_active,
            'models_count': len(self.neural_models),
            'hormone_levels': self.hormone_system.get_hormone_levels()
        }
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance of all models"""
        return {
            model_id: {
                'accuracy': model.accuracy,
                'latency': model.latency,
                'parameters': model.parameters
            }
            for model_id, model in self.neural_models.items()
        }


class SimpleSelfImprovementDemo:
    """Demonstration class for the simple self-improving system"""
    
    def __init__(self):
        self.system = SimpleSelfImprovingSystem()
        self.demo_active = False
    
    async def run_demo(self):
        """Run the complete demonstration"""
        logger.info("Starting Simple Self-Improving Neural Network System Demo")
        logger.info("=" * 60)
        
        try:
            # Start self-improvement
            await self.system.start_self_improvement()
            
            # Run multiple improvement cycles
            for cycle in range(5):
                logger.info(f"\n=== Improvement Cycle {cycle + 1} ===")
                
                # Show initial state
                await self._show_system_state()
                
                # Run improvement cycle
                results = await self.system.run_improvement_cycle()
                
                # Show results
                await self._show_improvement_results(results)
                
                # Wait between cycles
                await asyncio.sleep(2)
            
            # Show final state
            await self._show_final_state()
            
            # Stop self-improvement
            await self.system.stop_self_improvement()
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
    
    async def _show_system_state(self):
        """Show current system state"""
        status = self.system.get_improvement_status()
        performance = self.system.get_model_performance()
        
        logger.info("Current System State:")
        logger.info(f"  Iteration: {status['iteration']}")
        logger.info(f"  Total Improvements: {status['total_improvements']}")
        logger.info(f"  Active: {status['improvement_active']}")
        logger.info(f"  Models: {status['models_count']}")
        
        logger.info("Hormone Levels:")
        for hormone, level in status['hormone_levels'].items():
            logger.info(f"  {hormone}: {level:.3f}")
        
        logger.info("Model Performance:")
        for model_id, metrics in performance.items():
            logger.info(f"  {model_id}: accuracy={metrics['accuracy']:.3f}, "
                       f"latency={metrics['latency']:.1f}ms")
    
    async def _show_improvement_results(self, results: List[Dict]):
        """Show improvement results"""
        if not results:
            logger.info("No improvements in this cycle")
            return
        
        logger.info("Improvement Results:")
        for result in results:
            logger.info(f"  {result['task_id']} ({result['improvement_type']}): "
                       f"gain={result['performance_gain']:.4f}, "
                       f"time={result['execution_time']:.2f}s")
    
    async def _show_final_state(self):
        """Show final system state"""
        logger.info("\n=== Final System State ===")
        
        status = self.system.get_improvement_status()
        performance = self.system.get_model_performance()
        
        logger.info(f"Total Iterations: {status['iteration']}")
        logger.info(f"Total Improvements: {status['total_improvements']}")
        
        logger.info("Final Model Performance:")
        for model_id, metrics in performance.items():
            logger.info(f"  {model_id}: accuracy={metrics['accuracy']:.3f}, "
                       f"latency={metrics['latency']:.1f}ms")
        
        logger.info("Final Hormone Levels:")
        for hormone, level in status['hormone_levels'].items():
            logger.info(f"  {hormone}: {level:.3f}")
        
        # Show improvement history
        logger.info(f"\nImprovement History: {len(self.system.improvement_history)} cycles")
        for i, metrics in enumerate(self.system.improvement_history[-3:]):  # Last 3 cycles
            logger.info(f"  Cycle {metrics['iteration']}: {metrics['models_improved']} models improved, "
                       f"total gain: {metrics['total_improvement']:.4f}")


async def main():
    """Main demo function"""
    demo = SimpleSelfImprovementDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 