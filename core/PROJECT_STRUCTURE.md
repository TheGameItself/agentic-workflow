# MCP Project Structure

## λ:project_architecture(core_system_organization)

This document outlines the reorganized project structure for the MCP (Model Context Protocol) system, following PFSUS.MMCP-FormatWrapping.Standard.v1.4.0 file naming conventions.

### File Naming Convention Standards

Files follow order-agnostic nested format notation with mathematical operator precedence:

- **Core Components**: `*.core.system.v<version>.py`
- **Lambda Sequences**: `*.lambda.sequence.v<version>.mmd` (functional transformations)
- **Agent Specifications**: `*.agent.specification.v<version>.mmcp.mmd` (MMCP-compliant agent docs)
- **Test Files**: `test_*.pytest.v<version>.py` (testing framework integration)

## Core Directory Structure

The core directory contains all essential components of the MCP system:

```
core/
├── src/
│   └── mcp/
│       ├── lobes/                  # Brain-inspired modular components
│       ├── neural_network_models/  # Neural network implementations
│       ├── visualization/          # Visualization tools
│       ├── async_processing/       # Asynchronous processing framework
│       ├── core_system.py          # Core system implementation
│       ├── database_manager.py     # Database management
│       ├── memory.py               # Memory management
│       ├── workflow.py             # Workflow orchestration
│       ├── context_manager.py      # Context management
│       ├── p2p_network_bus.py      # P2P networking
│       ├── spinal_column.py        # Neural translation layer
│       └── ...                     # Other core components
├── tests/                          # Unit and integration tests
├── CORE_SYSTEM_GUIDE.md            # Core system documentation
├── setup_core.py                   # Core setup script
├── system_health_check.py          # System health check utility
└── README.md                       # Core documentation
```

## Key Components

### Core System

The `core_system.py` module serves as the central orchestrator for all MCP components, providing:

- System initialization and shutdown
- Component management
- Request handling
- Error recovery
- Performance monitoring

### Database Manager

The `database_manager.py` module provides optimized database operations with:

- Connection pooling
- Query caching
- Performance monitoring
- Automatic schema management
- Backup and recovery

### Memory Management

The `memory.py` module implements a multi-tier memory system with:

- Working memory
- Short-term memory
- Long-term memory
- Vector-based semantic search
- Memory consolidation

### Workflow Orchestration

The `workflow.py` module manages structured workflows with:

- Phases and steps
- Dependencies
- Progress tracking
- Error handling

### Neural Network Models

The `neural_network_models` package contains implementations of:

- Hormone neural integration
- Diffusion models
- Brain state integration
- Genetic diffusion models
- Cortical columns

### P2P Network

The P2P networking components enable distributed operation:

- Message bus
- Core integration
- Research tracking
- Status visualization

### Visualization

The `visualization` package provides tools for visualizing:

- Workflows
- Strategy patterns
- System state
- Performance metrics

## Development Guidelines

1. All new components should be added to the appropriate directory in `core/src/mcp/`
2. Tests should be added to `core/tests/`
3. Documentation should be updated to reflect changes
4. Follow the brain-inspired modular architecture
5. Use the strategy pattern for interchangeable components
6. Implement proper error handling and logging
7. **File Naming**: Follow PFSUS order-agnostic nested format notation
8. **Lambda Operators**: Use mathematical operator wrappers (λ, ℵ, Δ, β, Ω, τ, i) for enhanced semantic representation

## ℵ:self_reference(project_structure_metadata)

{type:Structure, file:"PROJECT_STRUCTURE.md", version:"1.0.0", checksum:"sha256:project_structure_checksum", canonical_address:"project-structure", pfsus_compliant:true, lambda_operators:true}

@{visual-meta-start}
author = {MCP Core Team},
title = {MCP Project Structure},
version = {1.0.0},
file_format = {structure.project.v1.0.0.md},
structure = { directory_organization, components, file_naming_standards, development_guidelines },
@{visual-meta-end}

%% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-22T00:00:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=structure.project.v1.0.0.md
