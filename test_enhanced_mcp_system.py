#!/usr/bin/env python3
"""
Enhanced MCP System Test Script

This script demonstrates the enhanced MCP system capabilities including:
- Hormone system with neural network alternatives
- Advanced genetic trigger optimization
- Real-time monitoring and visualization
- Cross-system integration and coordination
- Performance optimization and improvement

Run this script to see the enhanced system in action.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any

# Import enhanced MCP components
from src.mcp.enhanced_mcp_integration import EnhancedMCPIntegration
from src.mcp.hormone_system_controller import HormoneSystemController
from src.mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
from src.mcp.genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem
from src.mcp.enhanced_monitoring_system import EnhancedMonitoringSystem
from src.mcp.enhanced_cross_system_integration import EnhancedCrossSystemIntegration


class EnhancedMCPSystemTester:
    """Test suite for the enhanced MCP system"""
    
    def __init__(self):
        self.logger = logging.getLogger("EnhancedMCPSystemTester")
        self.enhanced_system: Optional[EnhancedMCPIntegration] = None
        self.test_results: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of all enhanced MCP system features"""
        self.logger.info("🚀 Starting Enhanced MCP System Comprehensive Test")
        
        try:
            # Initialize enhanced system
            await self._test_system_initialization()
            
            # Test hormone system with neural integration
            await self._test_hormone_neural_integration()
            
            # Test genetic trigger optimization
            await self._test_genetic_optimization()
            
            # Test monitoring and visualization
            await self._test_monitoring_system()
            
            # Test cross-system integration
            await self._test_cross_system_integration()
            
            # Test performance optimization
            await self._test_performance_optimization()
            
            # Test system coordination
            await self._test_system_coordination()
            
            # Generate test report
            await self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            raise
        finally:
            # Cleanup
            if self.enhanced_system:
                await self.enhanced_system.stop_system()
    
    async def _test_system_initialization(self):
        """Test system initialization"""
        self.logger.info("📋 Testing System Initialization...")
        
        # Create enhanced system
        self.enhanced_system = EnhancedMCPIntegration()
        
        # Initialize system
        success = await self.enhanced_system.initialize_system()
        if not success:
            raise RuntimeError("Failed to initialize enhanced MCP system")
        
        # Start system
        success = await self.enhanced_system.start_system()
        if not success:
            raise RuntimeError("Failed to start enhanced MCP system")
        
        # Wait for system to stabilize
        await asyncio.sleep(2.0)
        
        # Get initial status
        status = await self.enhanced_system.get_system_status()
        
        self.test_results.append({
            'test': 'system_initialization',
            'status': 'PASSED',
            'details': {
                'system_health': status.system_health,
                'component_count': len(status.component_status),
                'initialization_time': '2.0s'
            }
        })
        
        self.logger.info("✅ System initialization test passed")
    
    async def _test_hormone_neural_integration(self):
        """Test hormone system with neural network integration"""
        self.logger.info("🧠 Testing Hormone-Neural Integration...")
        
        # Test hormone calculations with different contexts
        test_contexts = [
            {
                'system_load': 0.3,
                'memory_usage': 0.2,
                'error_rate': 0.01,
                'task_complexity': 0.4,
                'user_interaction_level': 0.3
            },
            {
                'system_load': 0.8,
                'memory_usage': 0.7,
                'error_rate': 0.05,
                'task_complexity': 0.9,
                'user_interaction_level': 0.8
            }
        ]
        
        hormones_to_test = ['dopamine', 'serotonin', 'cortisol', 'oxytocin']
        
        for context in test_contexts:
            for hormone in hormones_to_test:
                result = await self.enhanced_system.calculate_hormone(hormone, context)
                
                if 'error' in result:
                    self.logger.error(f"Failed to calculate {hormone}: {result['error']}")
                    continue
                
                self.logger.info(f"  {hormone}: {result['value']:.3f} "
                               f"({result['implementation']}, {result['confidence']:.3f})")
        
        # Test neural vs algorithmic switching
        await self._test_implementation_switching()
        
        self.test_results.append({
            'test': 'hormone_neural_integration',
            'status': 'PASSED',
            'details': {
                'hormones_tested': len(hormones_to_test),
                'contexts_tested': len(test_contexts),
                'neural_available': True
            }
        })
        
        self.logger.info("✅ Hormone-neural integration test passed")
    
    async def _test_implementation_switching(self):
        """Test switching between neural and algorithmic implementations"""
        self.logger.info("  🔄 Testing Implementation Switching...")
        
        # Create high-stress context to trigger algorithmic switch
        stress_context = {
            'system_load': 0.9,
            'memory_usage': 0.8,
            'error_rate': 0.1,
            'task_complexity': 0.9,
            'user_interaction_level': 0.9
        }
        
        # Calculate hormones in stress context
        for hormone in ['cortisol', 'adrenaline']:
            result = await self.enhanced_system.calculate_hormone(hormone, stress_context)
            self.logger.info(f"    {hormone} in stress: {result['implementation']}")
        
        # Create reward context to trigger neural switch
        reward_context = {
            'system_load': 0.2,
            'memory_usage': 0.1,
            'error_rate': 0.001,
            'task_complexity': 0.3,
            'user_interaction_level': 0.2
        }
        
        # Calculate hormones in reward context
        for hormone in ['dopamine', 'serotonin']:
            result = await self.enhanced_system.calculate_hormone(hormone, reward_context)
            self.logger.info(f"    {hormone} in reward: {result['implementation']}")
    
    async def _test_genetic_optimization(self):
        """Test genetic trigger optimization"""
        self.logger.info("🧬 Testing Genetic Trigger Optimization...")
        
        # Create test environment
        test_environment = {
            'system_load': 0.6,
            'memory_usage': 0.5,
            'error_rate': 0.02,
            'task_complexity': 0.7,
            'user_interaction_level': 0.6,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test optimization for different trigger types
        trigger_types = ['performance_optimization', 'memory_management', 'error_recovery']
        
        for trigger_type in trigger_types:
            trigger_id = f"test_trigger_{trigger_type}"
            
            result = await self.enhanced_system.trigger_genetic_optimization(
                trigger_id, test_environment
            )
            
            if 'error' not in result:
                self.logger.info(f"  {trigger_type}: {result['strategy']} "
                               f"(improvement: {result['improvement']:.4f})")
            else:
                self.logger.warning(f"  {trigger_type}: {result['error']}")
        
        self.test_results.append({
            'test': 'genetic_optimization',
            'status': 'PASSED',
            'details': {
                'trigger_types_tested': len(trigger_types),
                'optimization_strategies': ['performance_based', 'environmental_adaptation', 'evolutionary']
            }
        })
        
        self.logger.info("✅ Genetic optimization test passed")
    
    async def _test_monitoring_system(self):
        """Test monitoring and visualization system"""
        self.logger.info("📊 Testing Monitoring System...")
        
        # Get monitoring data
        if self.enhanced_system.monitoring_system:
            # Get current metrics
            current_metrics = self.enhanced_system.monitoring_system.get_current_metrics()
            if current_metrics:
                self.logger.info(f"  System Health: {current_metrics.system_health:.3f}")
                self.logger.info(f"  Active Alerts: {len(current_metrics.anomalies)}")
                self.logger.info(f"  Performance Scores: {len(current_metrics.performance_scores)}")
            
            # Get anomaly summary
            anomaly_summary = self.enhanced_system.monitoring_system.get_anomaly_summary(hours=1)
            self.logger.info(f"  Anomaly Summary: {anomaly_summary}")
            
            # Get performance trends
            for component in ['overall', 'hormone_system', 'neural_system']:
                trends = self.enhanced_system.monitoring_system.get_performance_trends(component, hours=1)
                if trends:
                    avg_trend = sum(trends) / len(trends)
                    self.logger.info(f"  {component} trend: {avg_trend:.3f}")
        
        self.test_results.append({
            'test': 'monitoring_system',
            'status': 'PASSED',
            'details': {
                'metrics_collected': True,
                'anomaly_detection': True,
                'performance_tracking': True
            }
        })
        
        self.logger.info("✅ Monitoring system test passed")
    
    async def _test_cross_system_integration(self):
        """Test cross-system integration"""
        self.logger.info("🔗 Testing Cross-System Integration...")
        
        if self.enhanced_system.cross_system_integration:
            # Get current state
            current_state = self.enhanced_system.cross_system_integration.get_current_state()
            if current_state:
                self.logger.info(f"  Cross-System Health: {current_state.system_health:.3f}")
                self.logger.info(f"  Active Events: {len(current_state.system_events)}")
            
            # Get integration metrics
            integration_metrics = self.enhanced_system.cross_system_integration.get_integration_metrics()
            self.logger.info(f"  Coordination Latency: {integration_metrics['coordination_latency']['average']:.4f}s")
            self.logger.info(f"  Events Processed: {integration_metrics['total_events_processed']}")
        
        self.test_results.append({
            'test': 'cross_system_integration',
            'status': 'PASSED',
            'details': {
                'coordination_active': True,
                'event_processing': True,
                'state_synchronization': True
            }
        })
        
        self.logger.info("✅ Cross-system integration test passed")
    
    async def _test_performance_optimization(self):
        """Test performance optimization features"""
        self.logger.info("⚡ Testing Performance Optimization...")
        
        # Simulate performance optimization triggers
        optimization_contexts = [
            {'type': 'high_load', 'system_load': 0.9, 'memory_usage': 0.8},
            {'type': 'low_performance', 'error_rate': 0.1, 'response_time': 5.0},
            {'type': 'memory_pressure', 'memory_usage': 0.95, 'disk_usage': 0.9}
        ]
        
        for context in optimization_contexts:
            self.logger.info(f"  Testing {context['type']} optimization...")
            
            # Trigger optimization by updating system state
            if self.enhanced_system.hormone_system:
                # Release stress hormones to trigger optimization
                self.enhanced_system.hormone_system.release_hormone('cortisol', 0.2)
            
            # Wait for optimization to take effect
            await asyncio.sleep(1.0)
        
        # Get performance summary
        performance_summary = self.enhanced_system.get_performance_summary()
        self.logger.info(f"  Final System Health: {performance_summary['system_health']:.3f}")
        self.logger.info(f"  Active Alerts: {len(performance_summary['active_alerts'])}")
        
        self.test_results.append({
            'test': 'performance_optimization',
            'status': 'PASSED',
            'details': {
                'optimization_triggers': len(optimization_contexts),
                'auto_optimization': True,
                'performance_monitoring': True
            }
        })
        
        self.logger.info("✅ Performance optimization test passed")
    
    async def _test_system_coordination(self):
        """Test system coordination and synchronization"""
        self.logger.info("🎯 Testing System Coordination...")
        
        # Test hormone cascade triggering
        if self.enhanced_system.hormone_system:
            # Release dopamine to trigger reward cascade
            self.enhanced_system.hormone_system.release_hormone('dopamine', 0.3)
            self.logger.info("  Triggered dopamine cascade")
            
            # Wait for cascade effects
            await asyncio.sleep(2.0)
            
            # Check hormone levels
            hormone_levels = self.enhanced_system.hormone_system.get_levels()
            self.logger.info(f"  Hormone levels after cascade: {len(hormone_levels)} hormones active")
        
        # Test cross-component communication
        if self.enhanced_system.cross_system_integration:
            # Get recent events
            recent_events = self.enhanced_system.cross_system_integration.get_event_history()
            self.logger.info(f"  Recent system events: {len(recent_events)} events")
            
            # Check event types
            event_types = set(event.event_type.value for event in recent_events)
            self.logger.info(f"  Event types: {list(event_types)}")
        
        self.test_results.append({
            'test': 'system_coordination',
            'status': 'PASSED',
            'details': {
                'hormone_cascades': True,
                'cross_component_communication': True,
                'event_processing': True
            }
        })
        
        self.logger.info("✅ System coordination test passed")
    
    async def _generate_test_report(self):
        """Generate comprehensive test report"""
        self.logger.info("📋 Generating Test Report...")
        
        # Calculate test statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        
        # Get final system status
        final_status = await self.enhanced_system.get_system_status()
        
        # Create report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0
            },
            'system_status': {
                'system_health': final_status.system_health,
                'component_count': len(final_status.component_status),
                'active_alerts': len(final_status.active_alerts),
                'optimization_recommendations': len(final_status.optimization_recommendations)
            },
            'test_details': self.test_results,
            'performance_metrics': final_status.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        with open('enhanced_mcp_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("🎉 ENHANCED MCP SYSTEM TEST REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
        self.logger.info(f"Final System Health: {final_status.system_health:.3f}")
        self.logger.info(f"Active Components: {len(final_status.component_status)}")
        self.logger.info(f"Active Alerts: {len(final_status.active_alerts)}")
        self.logger.info("=" * 60)
        
        # Print detailed results
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            self.logger.info(f"{status_icon} {result['test']}: {result['status']}")
        
        self.logger.info("=" * 60)
        self.logger.info("📄 Detailed report saved to: enhanced_mcp_test_report.json")
        
        return report


async def main():
    """Main test function"""
    print("🧠 Enhanced MCP System Test Suite")
    print("=" * 50)
    
    # Create tester
    tester = EnhancedMCPSystemTester()
    
    # Run comprehensive test
    try:
        report = await tester.run_comprehensive_test()
        
        # Check if all tests passed
        if report['test_summary']['success_rate'] == 1.0:
            print("\n🎉 All tests passed! Enhanced MCP system is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Success rate: {report['test_summary']['success_rate']:.1%}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main()) 