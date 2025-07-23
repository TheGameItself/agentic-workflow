"""
Split-Brain A/B Testing for Genetic System

This module implements split-brain A/B testing for genetic trigger systems:
- Left/right lobe folder structure for parallel implementations
- Performance comparison framework for genetic variants
- Automatic selection of superior genetic implementations
- Comprehensive testing and validation
"""

import asyncio
import logging
import random
import uuid
import os
import shutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

from .genetic_trigger import GeneticTrigger
from .environmental_state import EnvironmentalState


class BrainLobe(Enum):
    """Brain lobe types for split-brain testing"""
    LEFT = "left"
    RIGHT = "right"


class TestPhase(Enum):
    """Phases of A/B testing"""
    INITIALIZATION = "initialization"
    BASELINE = "baseline"
    VARIANT_TESTING = "variant_testing"
    PERFORMANCE_COMPARISON = "performance_comparison"
    SELECTION = "selection"
    DEPLOYMENT = "deployment"


class SelectionCriteria(Enum):
    """Criteria for selecting superior implementations"""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    ADAPTABILITY = "adaptability"
    HYBRID = "hybrid"


@dataclass
class ABTestConfiguration:
    """Configuration for A/B testing"""
    test_id: str
    test_name: str
    description: str
    left_lobe_config: Dict[str, Any]
    right_lobe_config: Dict[str, Any]
    test_duration: timedelta
    sample_size: int
    selection_criteria: SelectionCriteria
    confidence_level: float = 0.95
    minimum_improvement: float = 0.1
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True


@dataclass
class TestResult:
    """Results from A/B testing"""
    test_id: str
    lobe: BrainLobe
    performance_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    adaptability_metrics: Dict[str, float]
    overall_score: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    test_duration: timedelta
    completion_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison results between lobes"""
    test_id: str
    left_lobe_result: TestResult
    right_lobe_result: TestResult
    winner: Optional[BrainLobe]
    improvement_magnitude: float
    statistical_significance: bool
    confidence_level: float
    comparison_metrics: Dict[str, float]
    recommendation: str
    comparison_time: datetime = field(default_factory=datetime.now)


class SplitBrainABTesting:
    """
    Split-brain A/B testing system for genetic trigger implementations.
    
    Features:
    - Left/right lobe parallel implementations
    - Performance comparison framework
    - Automatic selection of superior implementations
    - Statistical significance testing
    - Comprehensive result analysis
    """
    
    def __init__(self, 
                 base_directory: str = "split_brain_tests",
                 max_concurrent_tests: int = 5,
                 result_retention_days: int = 30):
        self.base_directory = Path(base_directory)
        self.max_concurrent_tests = max_concurrent_tests
        self.result_retention_days = result_retention_days
        
        self.logger = logging.getLogger("SplitBrainABTesting")
        
        # Core components
        self.test_configurations: Dict[str, ABTestConfiguration] = {}
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[BrainLobe, TestResult]] = {}
        self.comparison_results: Dict[str, ComparisonResult] = {}
        
        # Performance tracking
        self.overall_performance_history: List[Dict[str, Any]] = []
        self.lobe_performance_comparisons: Dict[str, List[ComparisonResult]] = defaultdict(list)
        
        # File structure
        self.left_lobe_dir = self.base_directory / "left_lobe"
        self.right_lobe_dir = self.base_directory / "right_lobe"
        self.results_dir = self.base_directory / "results"
        self.configs_dir = self.base_directory / "configs"
        
        # Initialize directory structure
        self._initialize_directory_structure()
        
        self.logger.info("Split-Brain A/B Testing System initialized")
    
    def _initialize_directory_structure(self):
        """Initialize the directory structure for split-brain testing"""
        # Create base directories
        self.base_directory.mkdir(exist_ok=True)
        self.left_lobe_dir.mkdir(exist_ok=True)
        self.right_lobe_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different test types
        for lobe_dir in [self.left_lobe_dir, self.right_lobe_dir]:
            (lobe_dir / "genetic_triggers").mkdir(exist_ok=True)
            (lobe_dir / "environmental_adaptation").mkdir(exist_ok=True)
            (lobe_dir / "expression_architecture").mkdir(exist_ok=True)
            (lobe_dir / "evolutionary_development").mkdir(exist_ok=True)
            (lobe_dir / "performance_monitoring").mkdir(exist_ok=True)
        
        self.logger.info("Directory structure initialized")
    
    def create_test_configuration(self, test_name: str, description: str,
                                left_config: Dict[str, Any], right_config: Dict[str, Any],
                                test_duration: timedelta, sample_size: int,
                                selection_criteria: SelectionCriteria) -> str:
        """
        Create a new A/B test configuration.
        
        Args:
            test_name: Name of the test
            description: Test description
            left_config: Configuration for left lobe
            right_config: Configuration for right lobe
            test_duration: Duration of the test
            sample_size: Number of samples for testing
            selection_criteria: Criteria for selecting winner
            
        Returns:
            Test configuration ID
        """
        test_id = f"ab_test_{uuid.uuid4().hex[:8]}"
        
        config = ABTestConfiguration(
            test_id=test_id,
            test_name=test_name,
            description=description,
            left_lobe_config=left_config,
            right_lobe_config=right_config,
            test_duration=test_duration,
            sample_size=sample_size,
            selection_criteria=selection_criteria
        )
        
        self.test_configurations[test_id] = config
        
        # Save configuration to file
        config_file = self.configs_dir / f"{test_id}.json"
        with open(config_file, 'w') as f:
            json.dump({
                'test_id': config.test_id,
                'test_name': config.test_name,
                'description': config.description,
                'left_lobe_config': config.left_lobe_config,
                'right_lobe_config': config.right_lobe_config,
                'test_duration': config.test_duration.total_seconds(),
                'sample_size': config.sample_size,
                'selection_criteria': config.selection_criteria.value,
                'confidence_level': config.confidence_level,
                'minimum_improvement': config.minimum_improvement,
                'created_at': config.created_at.isoformat(),
                'active': config.active
            }, f, indent=2)
        
        self.logger.info(f"Created A/B test configuration: {test_id}")
        return test_id
    
    async def deploy_lobe_implementations(self, test_id: str) -> bool:
        """
        Deploy implementations to left and right lobes.
        
        Args:
            test_id: Test configuration ID
            
        Returns:
            True if deployment was successful
        """
        if test_id not in self.test_configurations:
            self.logger.error(f"Test configuration {test_id} not found")
            return False
        
        config = self.test_configurations[test_id]
        
        try:
            # Deploy left lobe implementation
            left_success = await self._deploy_lobe_implementation(
                BrainLobe.LEFT, config.left_lobe_config, test_id
            )
            
            # Deploy right lobe implementation
            right_success = await self._deploy_lobe_implementation(
                BrainLobe.RIGHT, config.right_lobe_config, test_id
            )
            
            if left_success and right_success:
                self.logger.info(f"Successfully deployed implementations for test {test_id}")
                return True
            else:
                self.logger.error(f"Failed to deploy implementations for test {test_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying implementations for test {test_id}: {e}")
            return False
    
    async def _deploy_lobe_implementation(self, lobe: BrainLobe, config: Dict[str, Any], 
                                        test_id: str) -> bool:
        """Deploy implementation to specific lobe"""
        try:
            lobe_dir = self.left_lobe_dir if lobe == BrainLobe.LEFT else self.right_lobe_dir
            test_dir = lobe_dir / test_id
            test_dir.mkdir(exist_ok=True)
            
            # Create implementation files based on configuration
            await self._create_implementation_files(test_dir, config, lobe)
            
            # Create performance monitoring setup
            await self._setup_performance_monitoring(test_dir, config)
            
            self.logger.debug(f"Deployed {lobe.value} lobe implementation for test {test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying {lobe.value} lobe implementation: {e}")
            return False
    
    async def _create_implementation_files(self, test_dir: Path, config: Dict[str, Any], 
                                         lobe: BrainLobe):
        """Create implementation files for the lobe"""
        # Create genetic trigger implementation
        genetic_trigger_file = test_dir / "genetic_trigger.py"
        await self._create_genetic_trigger_implementation(genetic_trigger_file, config, lobe)
        
        # Create environmental adaptation implementation
        env_adaptation_file = test_dir / "environmental_adaptation.py"
        await self._create_environmental_adaptation_implementation(env_adaptation_file, config, lobe)
        
        # Create expression architecture implementation
        expression_arch_file = test_dir / "expression_architecture.py"
        await self._create_expression_architecture_implementation(expression_arch_file, config, lobe)
        
        # Create evolutionary development implementation
        evolutionary_dev_file = test_dir / "evolutionary_development.py"
        await self._create_evolutionary_development_implementation(evolutionary_dev_file, config, lobe)
    
    async def _create_genetic_trigger_implementation(self, file_path: Path, 
                                                   config: Dict[str, Any], lobe: BrainLobe):
        """Create genetic trigger implementation file"""
        implementation_code = f'''
"""
Genetic Trigger Implementation - {lobe.value.upper()} Lobe
Test Configuration: {config.get('test_name', 'Unknown')}
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class {lobe.value.capitalize()}LobeGeneticTrigger:
    """Genetic trigger implementation for {lobe.value} lobe"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{lobe.value.capitalize()}LobeGeneticTrigger")
        self.performance_metrics = {{}}
        self.activation_count = 0
        
    async def should_activate(self, environment: Dict[str, Any]) -> bool:
        """Determine if trigger should activate"""
        self.activation_count += 1
        
        # {lobe.value} lobe specific logic
        if lobe == BrainLobe.LEFT:
            # Left lobe: More conservative activation
            threshold = self.config.get('activation_threshold', 0.7)
            return environment.get('confidence', 0.0) > threshold
        else:
            # Right lobe: More aggressive activation
            threshold = self.config.get('activation_threshold', 0.5)
            return environment.get('confidence', 0.0) > threshold
    
    async def execute_trigger(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the genetic trigger"""
        start_time = datetime.now()
        
        # {lobe.value} lobe specific execution
        if lobe == BrainLobe.LEFT:
            # Left lobe: Sequential processing
            result = await self._sequential_processing(environment)
        else:
            # Right lobe: Parallel processing
            result = await self._parallel_processing(environment)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record performance metrics
        self.performance_metrics[f"execution_{self.activation_count}"] = {{
            'execution_time': execution_time,
            'success': result.get('success', False),
            'efficiency': result.get('efficiency', 0.0)
        }}
        
        return result
    
    async def _sequential_processing(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Sequential processing for left lobe"""
        await asyncio.sleep(0.01)  # Simulate processing time
        return {{
            'success': True,
            'efficiency': 0.8,
            'method': 'sequential'
        }}
    
    async def _parallel_processing(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processing for right lobe"""
        await asyncio.sleep(0.005)  # Simulate faster processing
        return {{
            'success': True,
            'efficiency': 0.9,
            'method': 'parallel'
        }}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {{
            'activation_count': self.activation_count,
            'performance_metrics': self.performance_metrics,
            'average_execution_time': self._calculate_average_execution_time(),
            'success_rate': self._calculate_success_rate()
        }}
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time"""
        if not self.performance_metrics:
            return 0.0
        
        times = [metrics['execution_time'] for metrics in self.performance_metrics.values()]
        return sum(times) / len(times)
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate"""
        if not self.performance_metrics:
            return 0.0
        
        successes = sum(1 for metrics in self.performance_metrics.values() 
                       if metrics.get('success', False))
        return successes / len(self.performance_metrics)
'''
        
        with open(file_path, 'w') as f:
            f.write(implementation_code)
    
    async def _create_environmental_adaptation_implementation(self, file_path: Path, 
                                                            config: Dict[str, Any], lobe: BrainLobe):
        """Create environmental adaptation implementation file"""
        implementation_code = f'''
"""
Environmental Adaptation Implementation - {lobe.value.upper()} Lobe
"""

import asyncio
import logging
from typing import Dict, Any

class {lobe.value.capitalize()}LobeEnvironmentalAdaptation:
    """Environmental adaptation for {lobe.value} lobe"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{lobe.value.capitalize()}LobeEnvironmentalAdaptation")
        self.adaptation_history = []
        
    async def adapt_to_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt to environmental conditions"""
        # {lobe.value} lobe specific adaptation
        adaptation_result = {{
            'adaptation_success': True,
            'adaptation_time': 0.01,
            'method': '{lobe.value}_lobe_adaptation'
        }}
        
        self.adaptation_history.append(adaptation_result)
        return adaptation_result
'''
        
        with open(file_path, 'w') as f:
            f.write(implementation_code)
    
    async def _create_expression_architecture_implementation(self, file_path: Path, 
                                                           config: Dict[str, Any], lobe: BrainLobe):
        """Create expression architecture implementation file"""
        implementation_code = f'''
"""
Expression Architecture Implementation - {lobe.value.upper()} Lobe
"""

import asyncio
import logging
from typing import Dict, Any

class {lobe.value.capitalize()}LobeExpressionArchitecture:
    """Expression architecture for {lobe.value} lobe"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{lobe.value.capitalize()}LobeExpressionArchitecture")
        
    async def execute_expression(self, expression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute genetic expression"""
        # {lobe.value} lobe specific expression
        return {{
            'expression_success': True,
            'expression_time': 0.02,
            'method': '{lobe.value}_lobe_expression'
        }}
'''
        
        with open(file_path, 'w') as f:
            f.write(implementation_code)
    
    async def _create_evolutionary_development_implementation(self, file_path: Path, 
                                                            config: Dict[str, Any], lobe: BrainLobe):
        """Create evolutionary development implementation file"""
        implementation_code = f'''
"""
Evolutionary Development Implementation - {lobe.value.upper()} Lobe
"""

import asyncio
import logging
from typing import Dict, Any

class {lobe.value.capitalize()}LobeEvolutionaryDevelopment:
    """Evolutionary development for {lobe.value} lobe"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{lobe.value.capitalize()}LobeEvolutionaryDevelopment")
        
    async def evolve_population(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve genetic population"""
        # {lobe.value} lobe specific evolution
        return {{
            'evolution_success': True,
            'evolution_time': 0.03,
            'method': '{lobe.value}_lobe_evolution'
        }}
'''
        
        with open(file_path, 'w') as f:
            f.write(implementation_code)
    
    async def _setup_performance_monitoring(self, test_dir: Path, config: Dict[str, Any]):
        """Setup performance monitoring for the test"""
        monitoring_file = test_dir / "performance_monitor.py"
        
        monitoring_code = '''
"""
Performance Monitoring for A/B Test
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self, test_id: str, lobe: str):
        self.test_id = test_id
        self.lobe = lobe
        self.metrics = []
        
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'metadata': metadata or {}
        }
        self.metrics.append(metric)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return {
            'test_id': self.test_id,
            'lobe': self.lobe,
            'metrics': self.metrics,
            'total_metrics': len(self.metrics)
        }
        
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_metrics(), f, indent=2)
'''
        
        with open(monitoring_file, 'w') as f:
            f.write(monitoring_code)
    
    async def run_ab_test(self, test_id: str, test_environment: Dict[str, Any]) -> bool:
        """
        Run A/B test with both lobe implementations.
        
        Args:
            test_id: Test configuration ID
            test_environment: Environment for testing
            
        Returns:
            True if test completed successfully
        """
        if test_id not in self.test_configurations:
            self.logger.error(f"Test configuration {test_id} not found")
            return False
        
        if len(self.active_tests) >= self.max_concurrent_tests:
            self.logger.error("Maximum concurrent tests reached")
            return False
        
        config = self.test_configurations[test_id]
        
        try:
            # Initialize test
            self.active_tests[test_id] = {
                'start_time': datetime.now(),
                'phase': TestPhase.INITIALIZATION,
                'left_lobe_metrics': [],
                'right_lobe_metrics': [],
                'test_environment': test_environment
            }
            
            # Deploy implementations
            await self.deploy_lobe_implementations(test_id)
            
            # Run baseline testing
            await self._run_baseline_testing(test_id, test_environment)
            
            # Run variant testing
            await self._run_variant_testing(test_id, test_environment)
            
            # Compare performance
            comparison_result = await self._compare_performance(test_id)
            
            # Select winner
            winner = await self._select_winner(test_id, comparison_result)
            
            # Record results
            await self._record_test_results(test_id, comparison_result, winner)
            
            # Cleanup
            del self.active_tests[test_id]
            
            self.logger.info(f"A/B test {test_id} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running A/B test {test_id}: {e}")
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            return False
    
    async def _run_baseline_testing(self, test_id: str, test_environment: Dict[str, Any]):
        """Run baseline testing phase"""
        self.active_tests[test_id]['phase'] = TestPhase.BASELINE
        
        # Run baseline tests for both lobes
        left_baseline = await self._test_lobe_implementation(
            BrainLobe.LEFT, test_id, test_environment, baseline=True
        )
        right_baseline = await self._test_lobe_implementation(
            BrainLobe.RIGHT, test_id, test_environment, baseline=True
        )
        
        self.active_tests[test_id]['left_lobe_baseline'] = left_baseline
        self.active_tests[test_id]['right_lobe_baseline'] = right_baseline
    
    async def _run_variant_testing(self, test_id: str, test_environment: Dict[str, Any]):
        """Run variant testing phase"""
        self.active_tests[test_id]['phase'] = TestPhase.VARIANT_TESTING
        
        config = self.test_configurations[test_id]
        
        # Run variant tests for both lobes
        left_variant = await self._test_lobe_implementation(
            BrainLobe.LEFT, test_id, test_environment, baseline=False
        )
        right_variant = await self._test_lobe_implementation(
            BrainLobe.RIGHT, test_id, test_environment, baseline=False
        )
        
        self.active_tests[test_id]['left_lobe_variant'] = left_variant
        self.active_tests[test_id]['right_lobe_variant'] = right_variant
    
    async def _test_lobe_implementation(self, lobe: BrainLobe, test_id: str, 
                                      test_environment: Dict[str, Any], 
                                      baseline: bool = False) -> Dict[str, Any]:
        """Test a specific lobe implementation"""
        config = self.test_configurations[test_id]
        lobe_dir = self.left_lobe_dir if lobe == BrainLobe.LEFT else self.right_lobe_dir
        test_dir = lobe_dir / test_id
        
        # Simulate testing (in real implementation, this would load and test actual code)
        test_results = {
            'lobe': lobe.value,
            'test_id': test_id,
            'baseline': baseline,
            'performance_metrics': {},
            'efficiency_metrics': {},
            'stability_metrics': {},
            'adaptability_metrics': {},
            'overall_score': 0.0,
            'sample_size': config.sample_size,
            'test_duration': config.test_duration.total_seconds()
        }
        
        # Generate simulated metrics
        if baseline:
            # Baseline metrics
            test_results['performance_metrics'] = {
                'accuracy': random.uniform(0.6, 0.8),
                'speed': random.uniform(0.5, 0.7),
                'throughput': random.uniform(0.4, 0.6)
            }
            test_results['efficiency_metrics'] = {
                'resource_usage': random.uniform(0.3, 0.5),
                'energy_consumption': random.uniform(0.4, 0.6),
                'memory_usage': random.uniform(0.3, 0.5)
            }
        else:
            # Variant metrics (potentially improved)
            improvement_factor = 1.0 + random.uniform(-0.1, 0.2)  # -10% to +20%
            
            test_results['performance_metrics'] = {
                'accuracy': min(1.0, random.uniform(0.6, 0.8) * improvement_factor),
                'speed': min(1.0, random.uniform(0.5, 0.7) * improvement_factor),
                'throughput': min(1.0, random.uniform(0.4, 0.6) * improvement_factor)
            }
            test_results['efficiency_metrics'] = {
                'resource_usage': max(0.0, random.uniform(0.3, 0.5) / improvement_factor),
                'energy_consumption': max(0.0, random.uniform(0.4, 0.6) / improvement_factor),
                'memory_usage': max(0.0, random.uniform(0.3, 0.5) / improvement_factor)
            }
        
        # Calculate overall score
        performance_avg = sum(test_results['performance_metrics'].values()) / len(test_results['performance_metrics'])
        efficiency_avg = sum(test_results['efficiency_metrics'].values()) / len(test_results['efficiency_metrics'])
        test_results['overall_score'] = (performance_avg + (1.0 - efficiency_avg)) / 2
        
        return test_results
    
    async def _compare_performance(self, test_id: str) -> ComparisonResult:
        """Compare performance between left and right lobes"""
        config = self.test_configurations[test_id]
        test_data = self.active_tests[test_id]
        
        # Get variant results
        left_result = test_data['left_lobe_variant']
        right_result = test_data['right_lobe_variant']
        
        # Calculate improvement magnitude
        left_score = left_result['overall_score']
        right_score = right_result['overall_score']
        improvement_magnitude = abs(right_score - left_score)
        
        # Determine winner
        winner = None
        if right_score > left_score + config.minimum_improvement:
            winner = BrainLobe.RIGHT
        elif left_score > right_score + config.minimum_improvement:
            winner = BrainLobe.LEFT
        
        # Calculate statistical significance (simplified)
        statistical_significance = improvement_magnitude > 0.05  # 5% threshold
        
        # Create comparison result
        comparison_result = ComparisonResult(
            test_id=test_id,
            left_lobe_result=TestResult(
                test_id=test_id,
                lobe=BrainLobe.LEFT,
                performance_metrics=left_result['performance_metrics'],
                efficiency_metrics=left_result['efficiency_metrics'],
                stability_metrics=left_result.get('stability_metrics', {}),
                adaptability_metrics=left_result.get('adaptability_metrics', {}),
                overall_score=left_score,
                confidence_interval=(left_score - 0.05, left_score + 0.05),
                sample_size=config.sample_size,
                test_duration=config.test_duration
            ),
            right_lobe_result=TestResult(
                test_id=test_id,
                lobe=BrainLobe.RIGHT,
                performance_metrics=right_result['performance_metrics'],
                efficiency_metrics=right_result['efficiency_metrics'],
                stability_metrics=right_result.get('stability_metrics', {}),
                adaptability_metrics=right_result.get('adaptability_metrics', {}),
                overall_score=right_score,
                confidence_interval=(right_score - 0.05, right_score + 0.05),
                sample_size=config.sample_size,
                test_duration=config.test_duration
            ),
            winner=winner,
            improvement_magnitude=improvement_magnitude,
            statistical_significance=statistical_significance,
            confidence_level=config.confidence_level,
            comparison_metrics={
                'performance_difference': right_result['performance_metrics'].get('accuracy', 0.0) - 
                                        left_result['performance_metrics'].get('accuracy', 0.0),
                'efficiency_difference': left_result['efficiency_metrics'].get('resource_usage', 0.0) - 
                                       right_result['efficiency_metrics'].get('resource_usage', 0.0)
            },
            recommendation=self._generate_recommendation(winner, improvement_magnitude, statistical_significance)
        )
        
        return comparison_result
    
    def _generate_recommendation(self, winner: Optional[BrainLobe], 
                               improvement_magnitude: float, 
                               statistical_significance: bool) -> str:
        """Generate recommendation based on comparison results"""
        if not winner:
            return "No clear winner - both implementations perform similarly"
        
        if not statistical_significance:
            return f"{winner.value.capitalize()} lobe shows slight improvement but not statistically significant"
        
        if improvement_magnitude > 0.1:
            return f"Strong recommendation for {winner.value} lobe implementation (significant improvement)"
        else:
            return f"Recommend {winner.value} lobe implementation (moderate improvement)"
    
    async def _select_winner(self, test_id: str, comparison_result: ComparisonResult) -> Optional[BrainLobe]:
        """Select the winning implementation"""
        config = self.test_configurations[test_id]
        
        if not comparison_result.statistical_significance:
            self.logger.info(f"Test {test_id}: No statistically significant difference")
            return None
        
        if comparison_result.improvement_magnitude < config.minimum_improvement:
            self.logger.info(f"Test {test_id}: Improvement below minimum threshold")
            return None
        
        winner = comparison_result.winner
        if winner:
            self.logger.info(f"Test {test_id}: Selected {winner.value} lobe as winner")
        
        return winner
    
    async def _record_test_results(self, test_id: str, comparison_result: ComparisonResult, 
                                 winner: Optional[BrainLobe]):
        """Record test results"""
        # Store results
        self.test_results[test_id] = {
            BrainLobe.LEFT: comparison_result.left_lobe_result,
            BrainLobe.RIGHT: comparison_result.right_lobe_result
        }
        
        self.comparison_results[test_id] = comparison_result
        
        # Save to file
        results_file = self.results_dir / f"{test_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'test_id': test_id,
                'winner': winner.value if winner else None,
                'comparison_result': {
                    'improvement_magnitude': comparison_result.improvement_magnitude,
                    'statistical_significance': comparison_result.statistical_significance,
                    'recommendation': comparison_result.recommendation,
                    'comparison_time': comparison_result.comparison_time.isoformat()
                },
                'left_lobe_results': {
                    'overall_score': comparison_result.left_lobe_result.overall_score,
                    'performance_metrics': comparison_result.left_lobe_result.performance_metrics,
                    'efficiency_metrics': comparison_result.left_lobe_result.efficiency_metrics
                },
                'right_lobe_results': {
                    'overall_score': comparison_result.right_lobe_result.overall_score,
                    'performance_metrics': comparison_result.right_lobe_result.performance_metrics,
                    'efficiency_metrics': comparison_result.right_lobe_result.efficiency_metrics
                }
            }, f, indent=2)
        
        # Update performance history
        self.overall_performance_history.append({
            'test_id': test_id,
            'timestamp': datetime.now().isoformat(),
            'winner': winner.value if winner else None,
            'improvement_magnitude': comparison_result.improvement_magnitude,
            'statistical_significance': comparison_result.statistical_significance
        })
        
        self.lobe_performance_comparisons[test_id].append(comparison_result)
    
    def get_test_results(self, test_id: str) -> Optional[ComparisonResult]:
        """Get results for a specific test"""
        return self.comparison_results.get(test_id)
    
    def get_all_test_results(self) -> Dict[str, ComparisonResult]:
        """Get all test results"""
        return self.comparison_results.copy()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        if not self.comparison_results:
            return {}
        
        total_tests = len(self.comparison_results)
        successful_tests = sum(1 for result in self.comparison_results.values() 
                             if result.winner is not None)
        
        left_wins = sum(1 for result in self.comparison_results.values() 
                       if result.winner == BrainLobe.LEFT)
        right_wins = sum(1 for result in self.comparison_results.values() 
                        if result.winner == BrainLobe.RIGHT)
        
        significant_tests = sum(1 for result in self.comparison_results.values() 
                              if result.statistical_significance)
        
        avg_improvement = sum(result.improvement_magnitude 
                            for result in self.comparison_results.values()) / total_tests
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'left_lobe_wins': left_wins,
            'right_lobe_wins': right_wins,
            'statistically_significant': significant_tests,
            'significance_rate': significant_tests / total_tests if total_tests > 0 else 0.0,
            'average_improvement': avg_improvement
        }
    
    def cleanup_old_results(self):
        """Clean up old test results"""
        cutoff_time = datetime.now() - timedelta(days=self.result_retention_days)
        
        # Remove old result files
        for result_file in self.results_dir.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                if file_time < cutoff_time:
                    result_file.unlink()
                    self.logger.debug(f"Removed old result file: {result_file}")
            except Exception as e:
                self.logger.error(f"Error removing old result file {result_file}: {e}")
        
        # Remove old test directories
        for lobe_dir in [self.left_lobe_dir, self.right_lobe_dir]:
            for test_dir in lobe_dir.iterdir():
                if test_dir.is_dir():
                    try:
                        dir_time = datetime.fromtimestamp(test_dir.stat().st_mtime)
                        if dir_time < cutoff_time:
                            shutil.rmtree(test_dir)
                            self.logger.debug(f"Removed old test directory: {test_dir}")
                    except Exception as e:
                        self.logger.error(f"Error removing old test directory {test_dir}: {e}")
    
    def save_testing_state(self, filepath: str):
        """Save testing state to file"""
        state = {
            'test_configurations': {
                tid: {
                    'test_id': config.test_id,
                    'test_name': config.test_name,
                    'description': config.description,
                    'selection_criteria': config.selection_criteria.value,
                    'created_at': config.created_at.isoformat(),
                    'active': config.active
                }
                for tid, config in self.test_configurations.items()
            },
            'comparison_results': {
                tid: {
                    'winner': result.winner.value if result.winner else None,
                    'improvement_magnitude': result.improvement_magnitude,
                    'statistical_significance': result.statistical_significance,
                    'recommendation': result.recommendation
                }
                for tid, result in self.comparison_results.items()
            },
            'performance_history': self.overall_performance_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Testing state saved to {filepath}")
    
    def load_testing_state(self, filepath: str):
        """Load testing state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load test configurations (simplified)
        self.logger.info(f"Testing state loaded from {filepath}")


# Import BrainLobe enum for the implementation files
from enum import Enum
BrainLobe = Enum('BrainLobe', ['LEFT', 'RIGHT']) 