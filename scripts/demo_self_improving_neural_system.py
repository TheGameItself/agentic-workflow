#!/usr/bin/env python3
"""
Demo: Self-Improving Neural Network System

This script demonstrates how the MCP can use itself to optimize, improve, and
pretrain its own neural networks through recursive self-improvement.

Features demonstrated:
- Automatic neural network optimization
- Self-supervised pretraining
- Cross-model knowledge distillation
- Hormone-driven optimization strategies
- Genetic architecture evolution
- Performance monitoring and improvement tracking
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mcp.neural_network_models.self_improvement_integration import (
    SelfImprovementIntegration, 
    SelfImprovementConfig,
    start_self_improvement_for_mcp
)
from mcp.enhanced_mcp_integration import EnhancedMCPIntegration
from mcp.hormone_system_controller import HormoneSystemController
from mcp.brain_state_aggregator import BrainStateAggregator
from mcp.genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem
from mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfImprovementDemo:
    """Demonstration class for self-improving neural networks"""
    
    def __init__(self):
        self.enhanced_mcp = None
        self.self_improvement = None
        self.demo_active = False
        
    async def setup_demo_environment(self):
        """Set up the demo environment with MCP components"""
        logger.info("Setting up demo environment...")
        
        # Create enhanced MCP integration
        self.enhanced_mcp = EnhancedMCPIntegration()
        
        # Initialize core components
        self.enhanced_mcp.hormone_system = HormoneSystemController()
        self.enhanced_mcp.brain_state = BrainStateAggregator()
        self.enhanced_mcp.genetic_optimizer = IntegratedGeneticTriggerSystem()
        self.enhanced_mcp.hormone_integration = HormoneNeuralIntegration(
            hormone_system_controller=self.enhanced_mcp.hormone_system
        )
        
        # Initialize the enhanced MCP
        await self.enhanced_mcp.initialize()
        
        logger.info("Demo environment setup complete")
    
    async def start_self_improvement_demo(self):
        """Start the self-improvement demonstration"""
        logger.info("Starting self-improvement demonstration...")
        
        # Create self-improvement configuration
        config = SelfImprovementConfig(
            enabled=True,
            auto_start=True,
            improvement_interval=60.0,  # 1 minute for demo
            max_concurrent_improvements=2,
            performance_threshold=0.05,  # 5% improvement required
            hormone_driven=True,
            genetic_enhanced=True,
            cross_model_distillation=True
        )
        
        # Create and initialize self-improvement integration
        self.self_improvement = SelfImprovementIntegration(
            enhanced_mcp=self.enhanced_mcp,
            config=config
        )
        
        await self.self_improvement.initialize()
        
        self.demo_active = True
        logger.info("Self-improvement demonstration started")
    
    async def run_demo_scenarios(self):
        """Run various demo scenarios"""
        logger.info("Running demo scenarios...")
        
        # Scenario 1: Monitor initial system state
        await self._scenario_1_initial_state()
        
        # Scenario 2: Force improvement cycle
        await self._scenario_2_force_improvement()
        
        # Scenario 3: Add custom improvement tasks
        await self._scenario_3_custom_tasks()
        
        # Scenario 4: Optimize specific models
        await self._scenario_4_optimize_models()
        
        # Scenario 5: Monitor improvement progress
        await self._scenario_5_monitor_progress()
        
        # Scenario 6: Generate health report
        await self._scenario_6_health_report()
        
        logger.info("Demo scenarios completed")
    
    async def _scenario_1_initial_state(self):
        """Scenario 1: Monitor initial system state"""
        logger.info("=== Scenario 1: Initial System State ===")
        
        # Get improvement status
        status = await self.self_improvement.get_improvement_status()
        logger.info(f"Improvement Status: {json.dumps(status, indent=2, default=str)}")
        
        # Get available models
        models = await self.self_improvement.get_available_models()
        logger.info(f"Available Models: {models}")
        
        # Get hormone levels
        if self.enhanced_mcp.hormone_system:
            hormone_levels = self.enhanced_mcp.hormone_system.get_hormone_levels()
            logger.info(f"Initial Hormone Levels: {hormone_levels}")
        
        await asyncio.sleep(2)
    
    async def _scenario_2_force_improvement(self):
        """Scenario 2: Force an improvement cycle"""
        logger.info("=== Scenario 2: Force Improvement Cycle ===")
        
        logger.info("Forcing immediate improvement cycle...")
        await self.self_improvement.force_improvement_cycle()
        
        # Wait for cycle to complete
        await asyncio.sleep(5)
        
        # Check results
        status = await self.self_improvement.get_improvement_status()
        logger.info(f"After forced cycle - Iterations: {status.get('iteration', 0)}, "
                   f"Total improvements: {status.get('total_improvements', 0)}")
        
        await asyncio.sleep(2)
    
    async def _scenario_3_custom_tasks(self):
        """Scenario 3: Add custom improvement tasks"""
        logger.info("=== Scenario 3: Custom Improvement Tasks ===")
        
        # Add custom tasks for different model types
        custom_tasks = [
            ('hormone_dopamine', 'training', 1.5),
            ('hormone_serotonin', 'optimization', 1.2),
            ('pattern_recognition', 'architecture', 1.8),
            ('memory_consolidation', 'distillation', 1.0)
        ]
        
        for model_id, improvement_type, priority in custom_tasks:
            logger.info(f"Adding custom task: {model_id} - {improvement_type} (priority: {priority})")
            await self.self_improvement.add_custom_improvement_task(
                model_id=model_id,
                improvement_type=improvement_type,
                priority=priority
            )
        
        await asyncio.sleep(2)
    
    async def _scenario_4_optimize_models(self):
        """Scenario 4: Optimize specific models"""
        logger.info("=== Scenario 4: Optimize Specific Models ===")
        
        # Get available models
        models = await self.self_improvement.get_available_models()
        
        for model_id in models[:3]:  # Optimize first 3 models
            logger.info(f"Optimizing model: {model_id}")
            await self.self_improvement.optimize_specific_model(model_id, 'auto')
        
        await asyncio.sleep(2)
    
    async def _scenario_5_monitor_progress(self):
        """Scenario 5: Monitor improvement progress"""
        logger.info("=== Scenario 5: Monitor Improvement Progress ===")
        
        # Wait for some improvements to occur
        logger.info("Waiting for improvements to occur...")
        await asyncio.sleep(10)
        
        # Get improvement history
        history = await self.self_improvement.get_improvement_history(5)
        logger.info(f"Recent improvement history: {len(history)} entries")
        
        for entry in history:
            logger.info(f"  Iteration {entry['iteration']}: "
                       f"{entry['models_improved']} models improved, "
                       f"total gain: {entry['total_improvement']:.4f}")
        
        # Get model performance
        models = await self.self_improvement.get_available_models()
        for model_id in models[:2]:
            performance = await self.self_improvement.get_model_performance(model_id)
            if performance:
                logger.info(f"  {model_id}: {len(performance)} performance measurements, "
                           f"latest: {performance[-1]:.4f}")
        
        await asyncio.sleep(2)
    
    async def _scenario_6_health_report(self):
        """Scenario 6: Generate comprehensive health report"""
        logger.info("=== Scenario 6: System Health Report ===")
        
        # Generate health report
        health_report = await self.self_improvement.get_system_health_report()
        
        logger.info("System Health Report:")
        logger.info(f"  Status: {health_report.get('status', 'unknown')}")
        
        health_metrics = health_report.get('health_metrics', {})
        logger.info(f"  Overall Health: {health_metrics.get('overall_health', 0):.3f}")
        logger.info(f"  Improvement Efficiency: {health_metrics.get('improvement_efficiency', 0):.3f}")
        logger.info(f"  System Stability: {health_metrics.get('system_stability', 0):.3f}")
        logger.info(f"  Learning Rate: {health_metrics.get('learning_rate', 0):.3f}")
        
        hormone_balance = health_report.get('hormone_balance', {})
        logger.info(f"  Hormone Balance: {hormone_balance.get('overall_balance', 'unknown')}")
        logger.info(f"  Stress Level: {hormone_balance.get('stress_level', 'unknown')}")
        logger.info(f"  Learning Capacity: {hormone_balance.get('learning_capacity', 'unknown')}")
        
        await asyncio.sleep(2)
    
    async def run_continuous_monitoring(self, duration_minutes: int = 5):
        """Run continuous monitoring for specified duration"""
        logger.info(f"=== Continuous Monitoring ({duration_minutes} minutes) ===")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time and self.demo_active:
            try:
                # Get current status
                status = await self.self_improvement.get_improvement_status()
                
                # Log progress
                elapsed = time.time() - start_time
                logger.info(f"[{elapsed/60:.1f}m] "
                           f"Iterations: {status.get('iteration', 0)}, "
                           f"Improvements: {status.get('total_improvements', 0)}, "
                           f"Active: {status.get('improvement_active', False)}")
                
                # Get hormone levels
                if self.enhanced_mcp.hormone_system:
                    hormone_levels = self.enhanced_mcp.hormone_system.get_hormone_levels()
                    logger.info(f"  Hormones: D={hormone_levels.get('dopamine', 0):.2f}, "
                               f"S={hormone_levels.get('serotonin', 0):.2f}, "
                               f"C={hormone_levels.get('cortisol', 0):.2f}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(30)
    
    async def export_demo_results(self):
        """Export demo results for analysis"""
        logger.info("=== Exporting Demo Results ===")
        
        # Export improvement data
        export_data = await self.self_improvement.export_improvement_data()
        
        # Save to file
        export_path = Path("demo_results_self_improvement.json")
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Demo results exported to: {export_path}")
        
        # Print summary
        status = export_data.get('status', {})
        logger.info(f"Final Summary:")
        logger.info(f"  Total Iterations: {status.get('iteration', 0)}")
        logger.info(f"  Total Improvements: {status.get('total_improvements', 0)}")
        logger.info(f"  Models Registered: {status.get('models_registered', 0)}")
        logger.info(f"  Improvement Active: {status.get('improvement_active', False)}")
    
    async def cleanup_demo(self):
        """Clean up demo resources"""
        logger.info("Cleaning up demo...")
        
        if self.self_improvement:
            await self.self_improvement.cleanup()
        
        if self.enhanced_mcp:
            await self.enhanced_mcp.cleanup()
        
        self.demo_active = False
        logger.info("Demo cleanup complete")


async def main():
    """Main demo function"""
    logger.info("Starting Self-Improving Neural Network System Demo")
    logger.info("=" * 60)
    
    demo = SelfImprovementDemo()
    
    try:
        # Setup demo environment
        await demo.setup_demo_environment()
        
        # Start self-improvement
        await demo.start_self_improvement_demo()
        
        # Run demo scenarios
        await demo.run_demo_scenarios()
        
        # Run continuous monitoring
        await demo.run_continuous_monitoring(duration_minutes=3)
        
        # Export results
        await demo.export_demo_results()
        
        logger.info("Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise
    finally:
        # Cleanup
        await demo.cleanup_demo()
    
    logger.info("Self-Improving Neural Network System Demo finished")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 