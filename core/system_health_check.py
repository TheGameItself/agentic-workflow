#!/usr/bin/env python3
"""
MCP Core System Health Check
Comprehensive health check for the MCP Core System.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add core/src to path
core_src = Path(__file__).parent / "src"
sys.path.insert(0, str(core_src))

from mcp.core_system import initialize_core_system, shutdown_core_system, SystemConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("health_check")

async def run_health_check():
    """Run comprehensive health check on the MCP Core System."""
    logger.info("Starting MCP Core System health check...")
    
    # Initialize system with test configuration
    config = SystemConfiguration(
        max_workers=2,
        enable_async=True,
        enable_monitoring=True,
        log_level="INFO",
        data_directory="data",
        backup_enabled=False,  # Disable for health check
        performance_optimization=True,
        experimental_features=False,
        hormone_system_enabled=True
    )
    
    start_time = time.time()
    system = await initialize_core_system(config)
    initialization_time = time.time() - start_time
    
    if not system:
        logger.error("Failed to initialize MCP Core System")
        return False
    
    logger.info(f"System initialized in {initialization_time:.2f} seconds")
    
    # Check system status
    status = system.get_status()
    logger.info(f"System status: {status.value}")
    
    # Check metrics
    metrics = system.get_metrics()
    logger.info(f"System metrics: CPU {metrics.cpu_usage:.1f}%, Memory {metrics.memory_usage:.1f}%")
    
    # Check lobes
    lobe_health = system.lobe_registry.health_check_all()
    healthy_lobes = sum(1 for health in lobe_health.values() if health.get('healthy', False))
    total_lobes = len(lobe_health)
    
    logger.info(f"Lobe health: {healthy_lobes}/{total_lobes} lobes healthy")
    
    for lobe_name, health in lobe_health.items():
        if not health.get('healthy', False):
            logger.warning(f"Unhealthy lobe: {lobe_name} - Status: {health.get('status')}")
    
    # Check hormone system
    if system.config.hormone_system_enabled:
        logger.info(f"Hormone levels: {json.dumps(system.hormone_levels, indent=2)}")
    
    # Test basic operations
    logger.info("Testing basic operations...")
    
    # Test memory operation
    memory_request = {
        'method': 'memory/add',
        'params': {
            'text': 'Health check test memory',
            'memory_type': 'test',
            'priority': 0.8
        }
    }
    
    memory_response = await system.execute_request(memory_request)
    if memory_response.get('success'):
        logger.info("Memory operation successful")
    else:
        logger.error(f"Memory operation failed: {memory_response.get('error')}")
    
    # Test system status operation
    status_request = {
        'method': 'system/status',
        'params': {}
    }
    
    status_response = await system.execute_request(status_request)
    if status_response.get('success'):
        logger.info("System status operation successful")
    else:
        logger.error(f"System status operation failed: {status_response.get('error')}")
    
    # Generate health report
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': status.value,
        'metrics': {
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'uptime': metrics.uptime,
            'total_requests': metrics.total_requests,
            'error_count': metrics.error_count
        },
        'lobes': {
            'total': total_lobes,
            'healthy': healthy_lobes,
            'details': lobe_health
        },
        'hormone_levels': system.hormone_levels,
        'operations_test': {
            'memory': memory_response.get('success', False),
            'status': status_response.get('success', False)
        }
    }
    
    # Save health report
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(health_report, f, indent=2)
    
    logger.info(f"Health report saved to {report_path}")
    
    # Shutdown system
    await shutdown_core_system()
    logger.info("MCP Core System health check completed")
    
    # Return overall health status
    return healthy_lobes == total_lobes and all(
        health_report['operations_test'].values()
    )

if __name__ == "__main__":
    success = asyncio.run(run_health_check())
    sys.exit(0 if success else 1)