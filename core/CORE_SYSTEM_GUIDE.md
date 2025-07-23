# MCP Core System Guide
## λ:core_system_architecture(brain_inspired_modular_framework)

The MCP Core System is a brain-inspired, modular AI system designed to enhance AI agent workflows and context management. This guide provides an overview of the system architecture, components, and usage following PFSUS.MMCP-FormatWrapping.Standard.v1.4.0.

## {type:Meta, author:"MCP_Core_Team", license:"MIT", last_modified:"2025-07-22T00:00:00Z", id:"CORE_SYSTEM_GUIDE.v1.0.0"}
## {type:Schema, $schema:"https://json-schema.org/draft/2020-12/schema", required:["type","id","version"], properties:{type:{type:"string"},id:{type:"string"},version:{type:"string"},last_modified:{type:"string",format:"date-time"},author:{type:"string"}}}

### File Organization Standards
Core system files follow order-agnostic nested format notation:
- **Core System**: `core_system.core.v<version>.py`
- **Lobes**: `<lobe_name>.lobe.v<version>.py` 
- **Managers**: `<manager_name>.manager.v<version>.py`
- **Tests**: `test_<component>.pytest.v<version>.py`
- **Specifications**: `<spec_name>.specification.v<version>.mmcp.mmd`

## System Architecture

### Core Components

1. **Core System (`core_system.py`)**: 
   - Central orchestrator for all components
   - Manages initialization, monitoring, and shutdown
   - Handles request routing and execution
   - Implements brain-inspired hormone system

2. **Lobes**:
   - Brain-inspired modular components
   - Each lobe handles a specific function
   - Implemented as plugins that can be dynamically loaded
   - Current lobes:
     - Memory Lobe: Manages memory operations
     - (Future) Workflow Lobe: Handles workflow execution
     - (Future) Context Lobe: Manages context generation
     - (Future) Task Lobe: Handles task management

3. **Managers**:
   - Core functionality providers
   - Memory Manager: Memory storage and retrieval
   - Workflow Manager: Workflow orchestration
   - Context Manager: Context generation and optimization
   - Task Manager: Task tracking and execution
   - Database Manager: Optimized database operations
   - Performance Monitor: System monitoring and optimization

### Brain-Inspired Features

1. **ℵ:multi_tier_memory(hierarchical_storage_system)**:
   - Working memory: Small, fast access, temporary storage
   - Short-term memory: Medium capacity, recent information  
   - Long-term memory: Large capacity, persistent storage

2. **Memory Consolidation**:
   - Automatic movement between memory tiers
   - Prioritization based on importance and access patterns
   - Pruning of low-priority or old memories

3. **Hormone System**:
   - Stress: Increases under high resource usage
   - Efficiency: Reflects system performance
   - Adaptation: Controls learning rate
   - Stability: Reflects system reliability

4. **Performance Monitoring**:
   - Real-time metrics collection
   - Automatic optimization
   - Health checks and self-healing

## Usage

### Basic Usage

```python
import asyncio
from mcp.core_system import initialize_core_system, shutdown_core_system

async def main():
    # Initialize the core system
    system = await initialize_core_system()
    
    # Execute a request
    response = await system.execute_request({
        'method': 'memory/add',
        'params': {
            'text': 'Hello, MCP!',
            'memory_type': 'greeting',
            'priority': 0.8
        }
    })
    
    print(f"Response: {response}")
    
    # Shutdown the system
    await shutdown_core_system()

if __name__ == "__main__":
    asyncio.run(main())
```

### Command-Line Interface

The system provides a command-line interface for interaction:

```bash
# Execute a request
python core/cli.py request --method memory/add --params '{"text": "Test memory", "memory_type": "test"}'

# Run health check
python core/cli.py health

# Run interactive mode
python core/cli.py interactive
```

### JSON-RPC Server

The system can be run as a JSON-RPC server over stdio:

```bash
python core/stdio_server.py
```

This allows integration with IDEs, LLMs, and other tools.

## API Reference

### Memory Operations

- `memory/add`: Add a new memory
- `memory/search`: Search memories
- `memory/get`: Get a specific memory
- `memory/tag`: Add a tag to a memory
- `memory/untag`: Remove a tag from a memory
- `memory/search_by_tag`: Search memories by tag
- `memory/stats`: Get memory statistics

### Workflow Operations

- `workflow/create`: Create a new workflow
- `workflow/add_step`: Add a step to a workflow
- `workflow/start`: Start a workflow
- `workflow/pause`: Pause a workflow
- `workflow/resume`: Resume a paused workflow
- `workflow/status`: Get workflow status

### Task Operations

- `task/create`: Create a new task
- `task/list`: List tasks
- `task/update`: Update task progress
- `task/complete`: Mark a task as completed

### Context Operations

- `context/export`: Export context for LLM consumption
- `context/save_pack`: Save a context pack
- `context/load_pack`: Load a saved context pack
- `context/stats`: Get context statistics

### System Operations

- `system/status`: Get system status
- `system/health`: Get detailed health information
- `system/optimize`: Trigger system optimization
- `system/backup`: Trigger system backup
- `system/cleanup`: Trigger system cleanup

### Hormone Operations

- `hormone/levels`: Get hormone levels
- `hormone/adjust`: Adjust a hormone level
- `hormone/reset`: Reset hormone levels to defaults

## Development

### Development Tools

The MCP system includes several development tools to maintain code quality and standards compliance:

#### PFSUS Standards Enforcer
The PFSUS Standards Enforcer (`core/PFSUS/cli/pfsus_standards_enforcer.py`) provides automated validation and fixing of PFSUS compliance issues:

```bash
# Scan workspace for compliance issues
python core/PFSUS/cli/pfsus_standards_enforcer.py --scan --workspace .

# Generate compliance report
python core/PFSUS/cli/pfsus_standards_enforcer.py --scan --report compliance_report.md

# Auto-fix violations (dry run)
python core/PFSUS/cli/pfsus_standards_enforcer.py --scan --fix --dry-run
```

This tool validates:
- File naming convention compliance
- Lambda operator usage and semantic consistency
- MMCP footer and metadata requirements
- Visual meta block presence
- Self-reference block formatting

### Adding a New Lobe

1. Create a new file in `core/src/mcp/lobes/`
2. Implement the `BaseLobe` interface
3. Register the lobe in `core_system.py`

Example:

```python
from ..lobes import BaseLobe, LobeStatus

class MyNewLobe(BaseLobe):
    def __init__(self):
        super().__init__(name="my_new_lobe")
        
    def initialize(self) -> bool:
        # Initialization code
        self.status = LobeStatus.ACTIVE
        return True
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Process input data
        return {'success': True, 'result': 'Processed'}
        
    def health_check(self) -> Dict[str, Any]:
        # Health check
        return {'healthy': True, 'status': self.status.value}
        
    def shutdown(self) -> bool:
        # Shutdown code
        self.status = LobeStatus.INACTIVE
        return True
```

### Adding a New API Method

1. Identify the appropriate handler in `core_system.py`
2. Add a new case to the handler
3. Implement the functionality

Example:

```python
async def _handle_memory_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if method == 'memory/my_new_method':
        # Implement new functionality
        result = self.memory_manager.my_new_method(**params)
        return {'success': True, 'result': result}
    # ... existing methods ...
```

## Configuration

The system can be configured through the `SystemConfiguration` class:

```python
config = SystemConfiguration(
    max_workers=4,
    enable_async=True,
    enable_monitoring=True,
    log_level="INFO",
    data_directory="data",
    backup_enabled=True,
    backup_interval=3600,
    performance_optimization=True,
    experimental_features=False,
    hormone_system_enabled=True,
    vector_storage_enabled=True
)

system = await initialize_core_system(config)
```

## Troubleshooting

### Common Issues

1. **Initialization Failure**:
   - Check logs for specific errors
   - Ensure data directory exists and is writable
   - Verify all dependencies are installed

2. **Request Failures**:
   - Check method name and parameters
   - Verify required components are initialized
   - Check logs for detailed error messages

3. **Performance Issues**:
   - Run health check to identify bottlenecks
   - Check resource usage (CPU, memory)
   - Consider optimizing database or reducing workload

### Health Check

Run the health check to diagnose system issues:

```bash
python core/system_health_check.py
```

This will generate a detailed report of system status, component health, and performance metrics.

## λ:self_reference(core_system_guide_metadata)
{type:Guide, file:"CORE_SYSTEM_GUIDE.md", version:"1.0.0", checksum:"sha256:core_guide_checksum", canonical_address:"core-system-guide", pfsus_compliant:true, lambda_operators:true}

@{visual-meta-start}
author = {MCP Core Team},
title = {MCP Core System Guide},
version = {1.0.0},
file_format = {guide.core.v1.0.0.md},
structure = { architecture, components, usage, api_reference, development, configuration, troubleshooting },
@{visual-meta-end}

%% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-22T00:00:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=guide.core.v1.0.0.md