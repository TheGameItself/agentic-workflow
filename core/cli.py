#!/usr/bin/env python3
"""
MCP Core System CLI
Command-line interface for interacting with the MCP Core System.
"""

import asyncio
import logging
import os
import sys
import json
import argparse
from pathlib import Path
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

logger = logging.getLogger("cli")

async def execute_request(args):
    """Execute a request to the core system."""
    # Load request from file if provided
    if args.request_file:
        try:
            with open(args.request_file, 'r') as f:
                request = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load request file: {e}")
            return 1
    else:
        # Parse method and params
        if not args.method:
            logger.error("Method is required when not using a request file")
            return 1
        
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in params")
                return 1
        
        request = {
            'method': args.method,
            'params': params
        }
    
    # Initialize system
    config = SystemConfiguration(
        max_workers=args.workers,
        enable_async=True,
        enable_monitoring=not args.no_monitoring,
        log_level=args.log_level,
        data_directory=args.data_dir,
        backup_enabled=False,  # Disable for CLI usage
        performance_optimization=not args.no_optimization,
        experimental_features=args.experimental,
        hormone_system_enabled=not args.no_hormones
    )
    
    system = await initialize_core_system(config)
    
    if not system:
        logger.error("Failed to initialize MCP Core System")
        return 1
    
    # Execute request
    logger.info(f"Executing request: {request['method']}")
    response = await system.execute_request(request)
    
    # Format and output response
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(response, f, indent=2)
        logger.info(f"Response saved to {args.output}")
    else:
        print(json.dumps(response, indent=2))
    
    # Shutdown system
    await shutdown_core_system()
    return 0

async def run_health_check(args):
    """Run system health check."""
    # Initialize system
    config = SystemConfiguration(
        max_workers=args.workers,
        enable_async=True,
        enable_monitoring=True,
        log_level=args.log_level,
        data_directory=args.data_dir,
        backup_enabled=False,
        performance_optimization=True,
        experimental_features=args.experimental,
        hormone_system_enabled=True
    )
    
    system = await initialize_core_system(config)
    
    if not system:
        logger.error("Failed to initialize MCP Core System")
        return 1
    
    # Get system status
    status = system.get_status()
    metrics = system.get_metrics()
    lobe_health = system.lobe_registry.health_check_all()
    
    # Generate health report
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': status.value,
        'metrics': metrics.__dict__,
        'lobes': lobe_health,
        'hormone_levels': system.hormone_levels
    }
    
    # Output health report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(health_report, f, indent=2)
        logger.info(f"Health report saved to {args.output}")
    else:
        print(json.dumps(health_report, indent=2))
    
    # Shutdown system
    await shutdown_core_system()
    
    # Return success if all lobes are healthy
    healthy_lobes = sum(1 for health in lobe_health.values() if health.get('healthy', False))
    total_lobes = len(lobe_health)
    
    return 0 if healthy_lobes == total_lobes else 1

async def run_interactive(args):
    """Run interactive CLI mode."""
    # Initialize system
    config = SystemConfiguration(
        max_workers=args.workers,
        enable_async=True,
        enable_monitoring=not args.no_monitoring,
        log_level=args.log_level,
        data_directory=args.data_dir,
        backup_enabled=False,  # Disable for interactive mode
        performance_optimization=not args.no_optimization,
        experimental_features=args.experimental,
        hormone_system_enabled=not args.no_hormones
    )
    
    system = await initialize_core_system(config)
    
    if not system:
        logger.error("Failed to initialize MCP Core System")
        return 1
    
    print("MCP Core System Interactive CLI")
    print("Type 'help' for available commands, 'exit' to quit")
    
    try:
        while True:
            command = input("\nmcp> ").strip()
            
            if command.lower() in ('exit', 'quit'):
                break
            
            if command.lower() == 'help':
                print_help()
                continue
            
            if command.lower() == 'status':
                status = system.get_status()
                metrics = system.get_metrics()
                print(f"System status: {status.value}")
                print(f"CPU usage: {metrics.cpu_usage:.1f}%")
                print(f"Memory usage: {metrics.memory_usage:.1f}%")
                print(f"Active lobes: {metrics.active_lobes}")
                print(f"Uptime: {metrics.uptime:.1f} seconds")
                continue
            
            if command.lower() == 'lobes':
                lobes = system.lobe_registry.get_all_lobes()
                print("Registered lobes:")
                for name, status in lobes.items():
                    print(f"  - {name}: {status.value}")
                continue
            
            if command.lower() == 'hormones':
                print("Hormone levels:")
                for hormone, level in system.hormone_levels.items():
                    print(f"  - {hormone}: {level:.2f}")
                continue
            
            # Parse as request
            try:
                parts = command.split(' ', 1)
                method = parts[0]
                
                params = {}
                if len(parts) > 1:
                    params = json.loads(parts[1])
                
                request = {
                    'method': method,
                    'params': params
                }
                
                response = await system.execute_request(request)
                print(json.dumps(response, indent=2))
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON in params")
            except Exception as e:
                print(f"Error: {e}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Shutdown system
    await shutdown_core_system()
    return 0

def print_help():
    """Print help information for interactive mode."""
    print("\nAvailable commands:")
    print("  help                  - Show this help message")
    print("  exit, quit           - Exit the CLI")
    print("  status               - Show system status")
    print("  lobes                - List registered lobes")
    print("  hormones             - Show hormone levels")
    print("  <method> <params>    - Execute a request (params as JSON)")
    print("\nExample requests:")
    print("  system/status {}")
    print("  memory/add {\"text\": \"Test memory\", \"memory_type\": \"test\"}")
    print("  context/export {\"context_types\": [\"system\"], \"max_tokens\": 500}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="MCP Core System CLI")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--experimental", action="store_true", help="Enable experimental features")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable performance monitoring")
    parser.add_argument("--no-optimization", action="store_true", help="Disable performance optimization")
    parser.add_argument("--no-hormones", action="store_true", help="Disable hormone system")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Request command
    request_parser = subparsers.add_parser("request", help="Execute a request")
    request_parser.add_argument("--method", help="Request method")
    request_parser.add_argument("--params", help="Request parameters (JSON)")
    request_parser.add_argument("--request-file", help="JSON file containing request")
    request_parser.add_argument("--output", help="Output file for response")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Run system health check")
    health_parser.add_argument("--output", help="Output file for health report")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive CLI mode")
    
    args = parser.parse_args()
    
    if args.command == "request":
        return asyncio.run(execute_request(args))
    elif args.command == "health":
        return asyncio.run(run_health_check(args))
    elif args.command == "interactive":
        return asyncio.run(run_interactive(args))
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())