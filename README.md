# MCP: Model Context Protocol System
## λ:system_overview(brain_inspired_modular_ai_system)

MCP (Model Context Protocol) is a brain-inspired, modular AI system designed to enhance AI agent workflows and context management. It provides a sophisticated framework for managing AI interactions, memory, and task execution with PFSUS-compliant lambda operator integration and order-agnostic file format wrapping.

### File Naming Standards
Following PFSUS.MMCP-FormatWrapping.Standard.v1.4.0, files use order-agnostic nested format notation:
- **Standard Format**: `<Name>.<Type>.<Version>.<Wrapper>.<Extension>`
- **Lambda Sequence**: `*.lambda.alef.md.sequence.mmd.py` (functional → set-theoretic → markdown → sequence → mermaid → python)
- **Calculus Notation**: Files may include mathematical operator wrappers (λ, ℵ, Δ, β, Ω, τ, i) with automatic precedence resolution

## Core Capabilities
### λ:capabilities_matrix(modular_brain_architecture)

- **λ:architecture(brain_inspired_lobes)**: Modular design with specialized "lobes" for different functions
- **ℵ:memory_management(multi_tier_vector_storage)**: Multi-tier memory system with vector storage and semantic search
- **Δ:workflow_orchestration(structured_task_dependencies)**: Structured workflow management with phases, tasks, and dependencies
- **τ:context_management(dynamic_llm_export)**: Dynamic context generation and export for LLM consumption
- **i:self_improvement(optimization_enhancement)**: Built-in mechanisms for system optimization and enhancement
- **β:visualization(system_state_workflows)**: Tools for visualizing system state and workflows
- **Ω:p2p_network(distributed_operation)**: Peer-to-peer capabilities for distributed operation

## Project Structure
### λ:project_structure(consolidated_core_architecture)

The project has been reorganized to consolidate all core components in the `core/` directory:

```
λ:directory_tree(core_consolidation) {
  core/
  ├── src/mcp/           # τ:core_system_code(modular_components)
  ├── tests/             # β:testing_framework(unit_integration)
  ├── setup_core.py      # Ω:setup_script(initialization)
  └── README.md          # ℵ:documentation(core_guide)
}
```

For detailed structure information, see [[core/PROJECT_STRUCTURE.md]] @{STRUCTURE.CORE.001}.

## Getting Started
### λ:installation_sequence(dependency_setup_health_check)

#### Prerequisites
- **τ:runtime_environment(python_3_9_plus)**: Python 3.9+
- **ℵ:dependency_management(requirements_txt)**: Required packages (see `requirements.txt`)

#### Installation
```bash
# λ:step_1(dependency_installation)
pip install -r requirements.txt

# Ω:step_2(core_system_initialization)  
python core/setup_core.py

# β:step_3(system_health_validation)
python core/system_health_check.py
```

## Usage
### λ:usage_patterns(server_cli_testing)

#### τ:server_execution(mcp_runtime)
```bash
# Ω:server_startup(core_mcp_server)
python -m core.src.mcp.server
```

#### β:cli_interface(command_execution)
```bash
# λ:cli_invocation(parameterized_commands)
python -m core.src.mcp.cli [command]
```

#### Δ:testing_framework(pytest_execution)
```bash
# ℵ:test_suite(comprehensive_validation)
pytest core/tests/
```

## Development
### i:development_guide(architecture_guidelines)

See [[core/CORE_SYSTEM_GUIDE.md]] @{DEV.GUIDE.001} for development guidelines and architecture details.

### β:code_quality_tools(standards_enforcement)

The project includes automated tools for maintaining code quality and standards compliance:

#### PFSUS Standards Enforcer
```bash
# λ:standards_scan(workspace_compliance_check)
python core/PFSUS/cli/pfsus_standards_enforcer.py --scan --workspace .

# ℵ:auto_fix(compliance_corrections)
python core/PFSUS/cli/pfsus_standards_enforcer.py --scan --fix --dry-run

# Δ:report_generation(compliance_analysis)
python core/PFSUS/cli/pfsus_standards_enforcer.py --scan --report compliance_report.md
```

This tool automatically:
- **τ:file_validation(naming_convention_compliance)**: Validates PFSUS file naming conventions
- **i:lambda_optimization(operator_usage_analysis)**: Analyzes and suggests lambda operator improvements
- **β:compliance_monitoring(standards_adherence_checking)**: Monitors adherence to MMCP standards
- **Ω:automated_fixing(violation_correction)**: Provides automated fixes for common violations

## License
### Ω:legal_framework(mit_license)

This project is licensed under the MIT License - see the LICENSE file for details.

## τ:self_reference(readme_metadata)
{type:Documentation, file:"README.md", version:"1.0.0", checksum:"sha256:readme_checksum", canonical_address:"mcp-readme", pfsus_compliant:true, lambda_operators:true, file_format:"readme.project.v1.0.0.md"}

@{visual-meta-start}
author = {MCP Core Team},
title = {MCP: Model Context Protocol System},
version = {1.0.0},
file_format = {readme.project.v1.0.0.md},
structure = { overview, capabilities, project_structure, installation, usage, development, license },
file_naming_standards = {pfsus_compliant, order_agnostic, lambda_operators},
@{visual-meta-end}

%% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-22T00:00:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=readme.project.v1.0.0.md