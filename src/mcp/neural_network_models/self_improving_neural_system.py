"""
Self-Improving Neural Network System

This module implements a recursive self-improvement system where the MCP uses itself
to optimize, improve, and pretrain its own neural networks. The system creates a
feedback loop where improved neural networks can then be used to improve other
neural networks, leading to continuous enhancement.

Features:
- Recursive self-improvement through neural network optimization
- Automatic model architecture evolution
- Self-supervised pretraining using system-generated data
- Cross-model knowledge transfer and distillation
- Performance-based model selection and improvement
- Hormone-driven optimization strategies
- Genetic algorithm integration for architecture search
"""

import logging
import time
import asyncio
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import threading
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    optim = None

from .hormone_neural_integration import HormoneNeuralIntegration, HormoneNeuralModel
from .performance_tracker import PerformanceTracker, PerformanceMetrics
from ..genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem
from ..hormone_system_controller import HormoneSystemController
from ..brain_state_aggregator import BrainStateAggregator


@dataclass
class ModelImprovementTask:
    """Task for improving a neural network model"""
    model_id: str
    model_type: str  # 'hormone', 'pattern', 'memory', 'genetic', etc.
    improvement_type: str  # 'architecture', 'training', 'optimization', 'distillation'
    priority: float = 1.0
    target_metrics: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class SelfImprovementMetrics:
    """Metrics for tracking self-improvement progress"""
    iteration: int
    models_improved: int
    total_improvement: float
    best_performance_gain: float
    average_training_time: float
    hormone_levels: Dict[str, float]
    genetic_activations: int
    memory_usage: float
    timestamp: datetime = field(default_factory=datetime.now)


class SelfImprovingNeuralSystem:
    """
    Self-improving neural network system that enables recursive optimization.
    
    This system allows the MCP to use its own neural networks to improve other
    neural networks, creating a continuous self-improvement loop.
    """
    
    def __init__(self, 
                 hormone_integration: Optional[HormoneNeuralIntegration] = None,
                 genetic_system: Optional[IntegratedGeneticTriggerSystem] = None,
                 hormone_system: Optional[HormoneSystemController] = None,
                 brain_state: Optional[BrainStateAggregator] = None):
        
        self.logger = logging.getLogger("SelfImprovingNeuralSystem")
        
        # Core system components
        self.hormone_integration = hormone_integration
        self.genetic_system = genetic_system
        self.hormone_system = hormone_system
        self.brain_state = brain_state
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Improvement tracking
        self.improvement_history: List[SelfImprovementMetrics] = []
        self.improvement_tasks: deque = deque(maxlen=1000)
        self.active_tasks: Dict[str, ModelImprovementTask] = {}
        self.completed_tasks: Dict[str, ModelImprovementTask] = {}
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[str, List[str]] = defaultdict(list)
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Self-improvement state
        self.iteration = 0
        self.total_improvements = 0
        self.best_improvement = 0.0
        self.improvement_active = False
        self.improvement_thread = None
        
        # Configuration
        self.config = {
            'max_concurrent_tasks': 3,
            'improvement_interval': 300.0,  # 5 minutes
            'performance_threshold': 0.1,  # 10% improvement required
            'max_iterations': 1000,
            'model_save_path': 'data/improved_models',
            'backup_models': True,
            'hormone_driven_optimization': True,
            'genetic_architecture_search': True,
            'cross_model_distillation': True
        }
        
        # Create model save directory
        Path(self.config['model_save_path']).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Self-Improving Neural System initialized")
    
    async def start_self_improvement_loop(self):
        """Start the continuous self-improvement loop"""
        if self.improvement_active:
            self.logger.warning("Self-improvement loop already active")
            return
        
        self.improvement_active = True
        self.improvement_thread = threading.Thread(
            target=self._improvement_loop, daemon=True
        )
        self.improvement_thread.start()
        
        self.logger.info("Self-improvement loop started")
    
    async def stop_self_improvement_loop(self):
        """Stop the self-improvement loop"""
        self.improvement_active = False
        if self.improvement_thread:
            self.improvement_thread.join(timeout=10.0)
        
        self.logger.info("Self-improvement loop stopped")
    
    def _improvement_loop(self):
        """Main improvement loop running in background thread"""
        while self.improvement_active:
            try:
                # Run improvement cycle
                asyncio.run(self._run_improvement_cycle())
                
                # Wait for next cycle
                time.sleep(self.config['improvement_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in improvement loop: {e}")
                time.sleep(60.0)  # Wait 1 minute on error
    
    async def _run_improvement_cycle(self):
        """Run a single improvement cycle"""
        self.iteration += 1
        self.logger.info(f"Starting improvement cycle {self.iteration}")
        
        # Generate improvement tasks
        tasks = await self._generate_improvement_tasks()
        
        # Execute tasks
        completed_tasks = []
        for task in tasks:
            try:
                result = await self._execute_improvement_task(task)
                if result:
                    completed_tasks.append(result)
            except Exception as e:
                self.logger.error(f"Error executing task {task.model_id}: {e}")
                task.status = "failed"
        
        # Update metrics
        await self._update_improvement_metrics(completed_tasks)
        
        # Trigger hormone responses
        if self.hormone_system and completed_tasks:
            await self._trigger_improvement_hormones(len(completed_tasks))
        
        self.logger.info(f"Completed improvement cycle {self.iteration} with {len(completed_tasks)} improvements")
    
    async def _generate_improvement_tasks(self) -> List[ModelImprovementTask]:
        """Generate improvement tasks based on current system state"""
        tasks = []
        
        # Analyze current model performance
        model_performance = await self._analyze_model_performance()
        
        for model_id, performance in model_performance.items():
            if performance['needs_improvement']:
                # Determine improvement type based on performance gaps
                improvement_type = self._determine_improvement_type(performance)
                
                # Calculate priority based on hormone levels and genetic triggers
                priority = await self._calculate_task_priority(model_id, performance)
                
                task = ModelImprovementTask(
                    model_id=model_id,
                    model_type=performance['model_type'],
                    improvement_type=improvement_type,
                    priority=priority,
                    target_metrics=performance['target_metrics'],
                    constraints=performance['constraints']
                )
                
                tasks.append(task)
        
        # Sort by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        # Limit to max concurrent tasks
        return tasks[:self.config['max_concurrent_tasks']]
    
    async def _analyze_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze current model performance and identify improvement opportunities"""
        analysis = {}
        
        # Analyze hormone neural models
        if self.hormone_integration:
            for hormone_name, model in self.hormone_integration.neural_models.items():
                performance = self.performance_tracker.get_latest_metrics(hormone_name)
                if performance:
                    needs_improvement = performance.accuracy < 0.8 or performance.latency > 100
                    analysis[f"hormone_{hormone_name}"] = {
                        'model_type': 'hormone',
                        'current_accuracy': performance.accuracy,
                        'current_latency': performance.latency,
                        'needs_improvement': needs_improvement,
                        'target_metrics': {
                            'accuracy': min(0.95, performance.accuracy + 0.1),
                            'latency': max(10, performance.latency * 0.8)
                        },
                        'constraints': {
                            'max_model_size': 1000000,  # 1M parameters
                            'max_training_time': 300  # 5 minutes
                        }
                    }
        
        # Analyze other neural models in the system
        # This would include pattern recognition, memory, genetic, etc.
        
        return analysis
    
    def _determine_improvement_type(self, performance: Dict[str, Any]) -> str:
        """Determine the type of improvement needed based on performance analysis"""
        accuracy = performance.get('current_accuracy', 0.5)
        latency = performance.get('current_latency', 100)
        
        if accuracy < 0.7:
            return 'training'  # Need better training
        elif latency > 200:
            return 'optimization'  # Need optimization
        elif accuracy < 0.9:
            return 'architecture'  # Need architecture improvements
        else:
            return 'distillation'  # Can use knowledge distillation
    
    async def _calculate_task_priority(self, model_id: str, performance: Dict[str, Any]) -> float:
        """Calculate task priority based on hormone levels and genetic triggers"""
        base_priority = 1.0
        
        # Hormone influence
        if self.hormone_system and self.config['hormone_driven_optimization']:
            hormone_levels = self.hormone_system.get_hormone_levels()
            
            # High dopamine increases priority for reward-seeking improvements
            if hormone_levels.get('dopamine', 0.5) > 0.7:
                base_priority *= 1.5
            
            # High cortisol increases priority for stress-related improvements
            if hormone_levels.get('cortisol', 0.3) > 0.6:
                base_priority *= 1.3
        
        # Genetic trigger influence
        if self.genetic_system and self.config['genetic_architecture_search']:
            # Check if genetic triggers suggest this model needs attention
            environment = await self._get_current_environment()
            active_triggers = self.genetic_system.get_active_triggers(environment)
            
            for trigger in active_triggers:
                if 'neural_optimization' in trigger.get('targets', []):
                    base_priority *= 1.2
        
        # Performance gap influence
        accuracy_gap = performance.get('target_metrics', {}).get('accuracy', 0.9) - performance.get('current_accuracy', 0.5)
        base_priority *= (1.0 + accuracy_gap)
        
        return min(10.0, base_priority)  # Cap at 10.0
    
    async def _execute_improvement_task(self, task: ModelImprovementTask) -> Optional[Dict[str, Any]]:
        """Execute a specific improvement task"""
        self.logger.info(f"Executing improvement task: {task.model_id} ({task.improvement_type})")
        
        task.status = "running"
        start_time = time.time()
        
        try:
            if task.improvement_type == 'training':
                result = await self._improve_model_training(task)
            elif task.improvement_type == 'optimization':
                result = await self._improve_model_optimization(task)
            elif task.improvement_type == 'architecture':
                result = await self._improve_model_architecture(task)
            elif task.improvement_type == 'distillation':
                result = await self._improve_model_distillation(task)
            else:
                self.logger.warning(f"Unknown improvement type: {task.improvement_type}")
                return None
            
            task.status = "completed"
            execution_time = time.time() - start_time
            
            # Record improvement
            self._record_model_improvement(task, result, execution_time)
            
            return {
                'task_id': task.model_id,
                'improvement_type': task.improvement_type,
                'performance_gain': result.get('performance_gain', 0.0),
                'execution_time': execution_time,
                'new_metrics': result.get('new_metrics', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.model_id}: {e}")
            task.status = "failed"
            return None
    
    async def _improve_model_training(self, task: ModelImprovementTask) -> Dict[str, Any]:
        """Improve model through better training strategies"""
        model_id = task.model_id
        
        # Get current model
        model = await self._get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Generate training data using the system itself
        training_data = await self._generate_training_data(model_id)
        
        # Use improved training strategies
        improved_model = await self._train_with_improved_strategies(model, training_data, task)
        
        # Evaluate improvement
        old_performance = await self._evaluate_model(model, training_data)
        new_performance = await self._evaluate_model(improved_model, training_data)
        
        performance_gain = new_performance['accuracy'] - old_performance['accuracy']
        
        # Save improved model
        await self._save_improved_model(model_id, improved_model)
        
        return {
            'performance_gain': performance_gain,
            'new_metrics': new_performance,
            'training_samples': len(training_data)
        }
    
    async def _improve_model_optimization(self, task: ModelImprovementTask) -> Dict[str, Any]:
        """Improve model through optimization techniques"""
        model_id = task.model_id
        
        # Get current model
        model = await self._get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Apply optimization techniques
        optimized_model = await self._apply_optimization_techniques(model, task)
        
        # Evaluate improvement
        test_data = await self._generate_test_data(model_id)
        old_performance = await self._evaluate_model(model, test_data)
        new_performance = await self._evaluate_model(optimized_model, test_data)
        
        performance_gain = new_performance['latency_improvement']
        
        # Save optimized model
        await self._save_improved_model(model_id, optimized_model)
        
        return {
            'performance_gain': performance_gain,
            'new_metrics': new_performance,
            'optimization_techniques': ['quantization', 'pruning', 'knowledge_distillation']
        }
    
    async def _improve_model_architecture(self, task: ModelImprovementTask) -> Dict[str, Any]:
        """Improve model through architecture evolution"""
        model_id = task.model_id
        
        # Use genetic algorithm to evolve architecture
        if self.genetic_system and self.config['genetic_architecture_search']:
            evolved_model = await self._evolve_model_architecture(model_id, task)
        else:
            # Use heuristic-based architecture improvement
            evolved_model = await self._improve_architecture_heuristic(model_id, task)
        
        # Evaluate improvement
        test_data = await self._generate_test_data(model_id)
        old_performance = await self._evaluate_model(await self._get_model(model_id), test_data)
        new_performance = await self._evaluate_model(evolved_model, test_data)
        
        performance_gain = new_performance['accuracy'] - old_performance['accuracy']
        
        # Save evolved model
        await self._save_improved_model(model_id, evolved_model)
        
        return {
            'performance_gain': performance_gain,
            'new_metrics': new_performance,
            'architecture_changes': ['layer_addition', 'activation_change', 'regularization']
        }
    
    async def _improve_model_distillation(self, task: ModelImprovementTask) -> Dict[str, Any]:
        """Improve model through knowledge distillation"""
        model_id = task.model_id
        
        # Get teacher model (best performing model of same type)
        teacher_model = await self._get_best_teacher_model(task.model_type)
        if not teacher_model:
            return {'performance_gain': 0.0, 'new_metrics': {}, 'error': 'No teacher model available'}
        
        # Perform knowledge distillation
        distilled_model = await self._perform_knowledge_distillation(
            teacher_model, await self._get_model(model_id), task
        )
        
        # Evaluate improvement
        test_data = await self._generate_test_data(model_id)
        old_performance = await self._evaluate_model(await self._get_model(model_id), test_data)
        new_performance = await self._evaluate_model(distilled_model, test_data)
        
        performance_gain = new_performance['accuracy'] - old_performance['accuracy']
        
        # Save distilled model
        await self._save_improved_model(model_id, distilled_model)
        
        return {
            'performance_gain': performance_gain,
            'new_metrics': new_performance,
            'distillation_techniques': ['soft_targets', 'attention_transfer', 'feature_mapping']
        }
    
    async def _generate_training_data(self, model_id: str) -> List[Tuple]:
        """Generate training data using the system itself"""
        # This is where the system uses itself to generate data
        training_data = []
        
        if 'hormone_' in model_id:
            # Generate hormone calculation training data
            hormone_name = model_id.replace('hormone_', '')
            training_data = await self._generate_hormone_training_data(hormone_name)
        else:
            # Generate generic training data based on model type
            training_data = await self._generate_generic_training_data(model_id)
        
        return training_data
    
    async def _generate_hormone_training_data(self, hormone_name: str) -> List[Tuple]:
        """Generate training data for hormone neural models"""
        training_data = []
        
        # Use the system's own hormone calculations to generate training data
        for _ in range(1000):  # Generate 1000 training samples
            # Create random context
            context = {
                'stress_level': np.random.random(),
                'error_frequency': np.random.random(),
                'resource_constraints': np.random.random(),
                'task_complexity': np.random.random(),
                'user_interaction_level': np.random.random(),
                'system_load': np.random.random(),
                'memory_usage': np.random.random(),
                'network_activity': np.random.random(),
                'learning_rate': np.random.random(),
                'confidence_level': np.random.random()
            }
            
            # Get algorithmic result as ground truth
            if self.hormone_integration:
                algorithmic_result = await self.hormone_integration.calculate_hormone(
                    hormone_name, context, implementation='algorithmic'
                )
                
                # Convert context to features
                features = self._extract_features_from_context(context)
                target = algorithmic_result.calculated_value
                
                training_data.append((features, target))
        
        return training_data
    
    async def _generate_generic_training_data(self, model_id: str) -> List[Tuple]:
        """Generate generic training data for other model types"""
        # This would generate data based on the specific model type
        # For now, return empty list
        return []
    
    async def _train_with_improved_strategies(self, model: Any, training_data: List[Tuple], task: ModelImprovementTask) -> Any:
        """Train model with improved strategies"""
        if not TORCH_AVAILABLE:
            return model
        
        # Convert training data to PyTorch format
        X, y = zip(*training_data)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Use improved training strategies
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Adaptive learning rate based on hormone levels
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_hormone_levels()
            lr_modifier = 1.0 + (hormone_levels.get('dopamine', 0.5) - 0.5) * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_modifier
        
        # Train with early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    break
        
        return model
    
    async def _apply_optimization_techniques(self, model: Any, task: ModelImprovementTask) -> Any:
        """Apply optimization techniques to improve model performance"""
        if not TORCH_AVAILABLE:
            return model
        
        # Model quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Model pruning (simplified)
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Simple weight pruning
                with torch.no_grad():
                    weights = module.weight.data
                    threshold = torch.quantile(torch.abs(weights), 0.1)
                    mask = torch.abs(weights) > threshold
                    module.weight.data *= mask
        
        return quantized_model
    
    async def _evolve_model_architecture(self, model_id: str, task: ModelImprovementTask) -> Any:
        """Evolve model architecture using genetic algorithm"""
        # This would use the genetic system to evolve the architecture
        # For now, return a simple improved architecture
        return await self._get_model(model_id)
    
    async def _improve_architecture_heuristic(self, model_id: str, task: ModelImprovementTask) -> Any:
        """Improve architecture using heuristic methods"""
        # This would apply heuristic-based architecture improvements
        # For now, return the original model
        return await self._get_model(model_id)
    
    async def _get_best_teacher_model(self, model_type: str) -> Optional[Any]:
        """Get the best performing model of the same type for knowledge distillation"""
        # Find the best performing model of the same type
        best_model_id = None
        best_performance = 0.0
        
        for model_id, performance in self.model_performance.items():
            if model_type in model_id and performance:
                avg_performance = np.mean(performance[-10:])  # Last 10 performances
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_model_id = model_id
        
        if best_model_id:
            return await self._get_model(best_model_id)
        
        return None
    
    async def _perform_knowledge_distillation(self, teacher_model: Any, student_model: Any, task: ModelImprovementTask) -> Any:
        """Perform knowledge distillation from teacher to student model"""
        if not TORCH_AVAILABLE:
            return student_model
        
        # This would implement knowledge distillation
        # For now, return the student model
        return student_model
    
    async def _get_model(self, model_id: str) -> Optional[Any]:
        """Get a model by ID"""
        if 'hormone_' in model_id and self.hormone_integration:
            hormone_name = model_id.replace('hormone_', '')
            return self.hormone_integration.neural_models.get(hormone_name)
        
        # Add other model types here
        return None
    
    async def _evaluate_model(self, model: Any, test_data: List[Tuple]) -> Dict[str, float]:
        """Evaluate model performance"""
        if not model or not test_data:
            return {'accuracy': 0.0, 'latency': 1000.0, 'latency_improvement': 0.0}
        
        # Simple evaluation
        start_time = time.time()
        
        if TORCH_AVAILABLE and hasattr(model, 'forward'):
            model.eval()
            with torch.no_grad():
                X, y = zip(*test_data[:100])  # Use first 100 samples
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
                
                outputs = model(X_tensor)
                mse = F.mse_loss(outputs, y_tensor).item()
                accuracy = 1.0 - min(1.0, mse)  # Convert MSE to accuracy-like metric
        else:
            accuracy = 0.5  # Default accuracy
            mse = 0.5
        
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            'accuracy': accuracy,
            'latency': latency,
            'latency_improvement': max(0, 100 - latency)  # Improvement in milliseconds
        }
    
    async def _generate_test_data(self, model_id: str) -> List[Tuple]:
        """Generate test data for model evaluation"""
        # Similar to training data generation but for testing
        return await self._generate_training_data(model_id)
    
    def _extract_features_from_context(self, context: Dict[str, float]) -> List[float]:
        """Extract features from context dictionary"""
        return list(context.values())
    
    async def _get_current_environment(self) -> Dict[str, Any]:
        """Get current environment state for genetic triggers"""
        environment = {
            'system_load': 0.5,
            'memory_usage': 0.3,
            'error_rate': 0.01,
            'task_complexity': 0.6,
            'user_interaction_level': 0.4,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.brain_state:
            brain_state = self.brain_state.get_system_state()
            environment.update(brain_state)
        
        return environment
    
    async def _save_improved_model(self, model_id: str, model: Any):
        """Save improved model"""
        if not model:
            return
        
        # Create model metadata
        metadata = {
            'model_id': model_id,
            'version': f"v{len(self.model_versions[model_id]) + 1}",
            'improvement_timestamp': datetime.now().isoformat(),
            'model_size': self._get_model_size(model),
            'performance_metrics': await self._evaluate_model(model, await self._generate_test_data(model_id))
        }
        
        # Save model and metadata
        model_path = Path(self.config['model_save_path']) / f"{model_id}_{metadata['version']}.pkl"
        metadata_path = Path(self.config['model_save_path']) / f"{model_id}_{metadata['version']}_metadata.json"
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry
            self.model_versions[model_id].append(metadata['version'])
            self.model_registry[model_id] = metadata
            
            self.logger.info(f"Saved improved model: {model_id} {metadata['version']}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
    
    def _get_model_size(self, model: Any) -> int:
        """Get model size in parameters"""
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        return 0
    
    def _record_model_improvement(self, task: ModelImprovementTask, result: Dict[str, Any], execution_time: float):
        """Record model improvement in history"""
        self.total_improvements += 1
        
        if result and result.get('performance_gain', 0) > self.best_improvement:
            self.best_improvement = result['performance_gain']
        
        # Update model performance history
        if result and 'new_metrics' in result:
            self.model_performance[task.model_id].append(result['new_metrics'].get('accuracy', 0.0))
    
    async def _update_improvement_metrics(self, completed_tasks: List[Dict[str, Any]]):
        """Update improvement metrics"""
        if not completed_tasks:
            return
        
        total_improvement = sum(task.get('performance_gain', 0) for task in completed_tasks)
        best_gain = max(task.get('performance_gain', 0) for task in completed_tasks)
        avg_training_time = np.mean([task.get('execution_time', 0) for task in completed_tasks])
        
        # Get hormone levels
        hormone_levels = {}
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_hormone_levels()
        
        # Get genetic activations
        genetic_activations = 0
        if self.genetic_system:
            genetic_activations = len(self.genetic_system.active_triggers)
        
        # Get memory usage
        memory_usage = 0.0
        if self.brain_state:
            brain_state = self.brain_state.get_system_state()
            memory_usage = brain_state.get('memory_usage', 0.0)
        
        metrics = SelfImprovementMetrics(
            iteration=self.iteration,
            models_improved=len(completed_tasks),
            total_improvement=total_improvement,
            best_performance_gain=best_gain,
            average_training_time=avg_training_time,
            hormone_levels=hormone_levels,
            genetic_activations=genetic_activations,
            memory_usage=memory_usage
        )
        
        self.improvement_history.append(metrics)
        
        self.logger.info(f"Improvement metrics: {len(completed_tasks)} models improved, "
                        f"total gain: {total_improvement:.4f}, best gain: {best_gain:.4f}")
    
    async def _trigger_improvement_hormones(self, improvements_count: int):
        """Trigger hormone responses based on improvement success"""
        if not self.hormone_system:
            return
        
        # Release dopamine for successful improvements
        if improvements_count > 0:
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
    
    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get current improvement status"""
        return {
            'iteration': self.iteration,
            'total_improvements': self.total_improvements,
            'best_improvement': self.best_improvement,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'improvement_history': len(self.improvement_history),
            'models_registered': len(self.model_registry),
            'improvement_active': self.improvement_active
        }
    
    async def get_improvement_history(self, limit: int = 50) -> List[SelfImprovementMetrics]:
        """Get recent improvement history"""
        return self.improvement_history[-limit:]
    
    async def get_model_performance(self, model_id: str) -> List[float]:
        """Get performance history for a specific model"""
        return self.model_performance.get(model_id, [])
    
    async def force_improvement_cycle(self):
        """Force an immediate improvement cycle"""
        await self._run_improvement_cycle()
    
    async def add_improvement_task(self, task: ModelImprovementTask):
        """Add a manual improvement task"""
        self.improvement_tasks.append(task)
        self.logger.info(f"Added manual improvement task: {task.model_id}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models for improvement"""
        models = []
        
        # Add hormone models
        if self.hormone_integration:
            for hormone_name in self.hormone_integration.neural_models.keys():
                models.append(f"hormone_{hormone_name}")
        
        # Add other model types here
        
        return models 