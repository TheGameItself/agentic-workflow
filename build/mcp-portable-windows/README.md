# 🚀 MCP Agentic Workflow Accelerator - AI Development Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Compatible-brightgreen.svg)](https://modelcontextprotocol.io/)
[![Local](https://img.shields.io/badge/Local-Only-orange.svg)](https://github.com/search?q=local+only+python)
[![Portable](https://img.shields.io/badge/Portable-Yes-purple.svg)](https://github.com/search?q=portable+python+application)
[![Auto-Update](https://img.shields.io/badge/Auto--Update-Enabled-brightgreen.svg)](https://github.com/search?q=automatic+update+system)

> **Accelerate AI-powered development workflows with a portable, local MCP server for LLMs**

- **Python 3.8+** | **MIT License** | **MCP-Compatible** | **Local-Only** | **Portable** | **Auto-Update** | **"Advanced" API**

---

## Overview

The MCP Agentic Workflow Accelerator transforms single prompts into complete applications using intelligent project management, memory systems, and workflow orchestration. It is fully portable, requires no network, and is designed for seamless IDE integration with **automatic self-updating capabilities**.

---

## Key Features

- **Project Initialization**: Instantly structure new projects from ideas.
- **Intelligent Memory**: Vector-based, cross-project, and quality-assessed.
- **Task Management**: Hierarchical, priority-based, with partial completion.
- **Research Automation**: Guided research and findings tracking.
- **Context Optimization**: Minimal, relevant context for LLMs.
- **Safety**: Accuracy-critical task protection.
- **IDE Integration**: Works with Cursor, VS Code, and more.
- **Automatic Updates**: Self-updating system with rollback capabilities.
- **Performance Monitoring**: Real-time optimization and resource management.
- **P2P Network**: Decentralized genetic data exchange and collaboration.

---

## Automatic Update System

The MCP system includes a comprehensive automatic update system that ensures you always have the latest features and improvements:

### Update Features
- **Automatic Version Checking**: Regularly checks for updates without user intervention
- **Smart Update Detection**: Identifies critical vs. optional updates
- **Safe Installation**: Creates backups before updating with rollback capabilities
- **Verification System**: Validates downloads and installations for security
- **Cross-Platform Support**: Works on Linux, macOS, and Windows
- **Local-Only Operation**: Can update from local sources without internet
- **Health Monitoring**: Verifies system integrity after updates

### Update Commands
```bash
# Check for updates
python -m src.mcp.automatic_update_system --check

# Perform update
python -m src.mcp.automatic_update_system --update

# Show update status
python -m src.mcp.automatic_update_system --status

# Enable/disable automatic updates
python -m src.mcp.automatic_update_system --enable
python -m src.mcp.automatic_update_system --disable
```

### Update Configuration
The system can be configured via environment variables or config files:
- `MCP_AUTO_UPDATE_ENABLED`: Enable/disable automatic updates
- `MCP_UPDATE_CHECK_INTERVAL`: Hours between update checks (default: 24)
- `MCP_UPDATE_SOURCES`: Comma-separated list of update sources
- `MCP_BACKUP_ENABLED`: Enable backup before updates (default: true)

---

## Brain-Inspired Architecture & Neural Networks

### Modular "Lobe" System ✅

The MCP server is inspired by the human brain, with each "lobe" or engine responsible for a specialized cognitive function. **🎉 IMPLEMENTATION COMPLETE - 100% of components fully implemented**:

- ✅ **Three-Tier Memory System**: Working, short-term, and long-term memory with automatic consolidation
- ✅ **Pattern Recognition Engine**: Neural column simulation with adaptive sensitivity and cross-lobe sensory data sharing
- ✅ **Genetic Trigger System**: Environmental adaptation with dual code/neural implementations and A/B testing
- ✅ **Hormone System**: Cross-lobe communication using biologically-inspired hormone signaling
- ✅ **P2P Network**: Decentralized genetic data exchange with global performance benchmarking
- ✅ **Simulated Reality Engine**: Entity-relationship-state tracking for world modeling
- ✅ **Physics Math Engine**: Advanced mathematical computations and differential equation solving
- ✅ **Dreaming Engine**: Multi-agent scenario simulation for creative insight and risk assessment
- ✅ **Scientific Process Engine**: Structured hypothesis-experiment-analysis workflow
- ✅ **Multi-LLM Orchestrator**: Task routing and load balancing across multiple LLMs
- ✅ **Advanced Engram Engine**: Neural network-based memory compression and predictive modeling
- ✅ **Performance Optimization Engine**: Real-time resource monitoring and adaptive optimization
- ✅ **WebSocialEngine**: Comprehensive web interaction and social intelligence with CAPTCHA handling
- ✅ **Cross-Engine Coordination**: Multi-engine simulation coordination with shared state management

### Cross-Lobe Communication (Implemented)

The system implements comprehensive brain-inspired cross-lobe communication:

- **Sensory Data Sharing**: Standardized `cross_lobe_sensory_data` format with propagation rules
- **Hormone-Triggered Propagation**: Dynamic priority adjustment based on dopamine, cortisol, norepinephrine, and serotonin levels
- **Adaptive Sensitivity Management**: Cross-column learning and hormone-based sensitivity modulation
- **Real-Time Synchronization**: Immediate cross-lobe data availability with comprehensive statistics
- **Performance Monitoring**: Detailed tracking of sharing activity, rule usage, and hormone influence

### P2P Genetic Data Exchange (Implemented)

Secure, decentralized sharing of optimizations using genetic-inspired encoding:

- **256-Codon Genetic Encoding**: Extended genetic alphabet for rich metadata encoding with integration instructions (when, where, how, why, what, order)
- **Privacy-Preserving Pipeline**: Multi-stage data sanitization and cryptographic security
- **DHT Routing**: Distributed hash table for efficient peer discovery and data routing
- **Engram Transfer**: Compressed memory structure sharing with multiple compression algorithms
- **Network Orchestration**: Coordinated genetic operations across network nodes
- **Status Visualization**: Real-time P2P network status with reputation scoring

### Data Flow

- **Request Flow**:  
  `Client → API Layer → Core Service → Experimental Lobe → Data Layer → Response`
- **Memory Flow**:  
  `Input → Preprocessing → Vectorization → Storage → Retrieval → Relevance Scoring → Context Assembly`
- **Learning Flow**:  
  `Feedback → Pattern Recognition → Model Update → Optimization → Deployment`
- **Update Flow**:  
  `Version Check → Download → Verify → Backup → Install → Verify → Rollback (if needed)`

### Memory System

- **WorkingMemory**: Context-sensitive, temporary storage for immediate feedback and adaptation.
- **ShortTermMemory**: Recent, high-priority, or volatile information for rapid recall.
- **LongTermMemory**: Persistent, structured, and research-driven storage (vector DBs, engrams, knowledge).

All lobes/engines use these memory types for robust, brain-inspired information flow.

---

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python mcp_cli.py --help
```

Or, for a simplified CLI:

```bash
python simple_mcp_cli.py help
```

For automatic setup and configuration:

```bash
python scripts/setup_wizard.py
```

---

## CLI Commands (Summary)

- **Project**: `init-project`, `show-questions`, `answer-question`, `project-status`
- **Workflow**: `start-research`, `add-research-topic`, `add-finding`, `start-planning`, `workflow-status`
- **Memory**: `add-memory`, `search-memories`, `get-memory`
- **Task**: `create-task`, `list-tasks`, `update-task-progress`, `add-task-note`, `add-task-dependency`, `task-tree`
- **Context**: `export-context`, `get-context-pack`
- **Advanced**: `bulk-update-task-status`, `add-task-feedback`, `statistics`
- **Updates**: `check-updates`, `perform-update`, `update-status`, `enable-auto-update`, `disable-auto-update`
- **Performance**: `optimize-system`, `performance-report`, `resource-status`

---

## IDE & Service Integration

**Example (VS Code):**

Add to `.vscode/tasks.json`:
```json
{
  "label": "MCP: Export Context",
  "type": "shell",
  "command": "python",
  "args": ["mcp_cli.py", "export-context", "--types", "tasks,memories,progress", "--max-tokens", "1000"]
}
```

**For other IDEs and services:**  
See the `docs/` vault for ready-to-use configuration snippets for Cursor, JetBrains, Neovim, Emacs, Sublime, Copilot, and more.

---

## Performance & Monitoring

The system includes comprehensive performance monitoring and optimization:

### Performance Features
- **Real-time Resource Monitoring**: CPU, memory, disk, and network usage tracking
- **Adaptive Optimization**: Automatic resource allocation and constraint adaptation
- **Predictive Resource Allocation**: Machine learning-based resource prediction
- **Performance Analytics**: Detailed performance metrics and trend analysis
- **Bottleneck Detection**: Automatic identification and resolution of performance issues
- **P2P Benchmarking**: Global performance comparison and projection

### Monitoring Commands
```bash
# Get performance report
python mcp_cli.py performance-report

# Optimize system
python mcp_cli.py optimize-system

# Check resource status
python mcp_cli.py resource-status

# Get P2P network status
python mcp_cli.py p2p-status
```

---

## Project Structure

- `src/` — Main source code (modular, lobe-based)
- `frontend/` — Web UI
- `docs/` — Documentation vault (Obsidian-compatible)
- `config/` — Configuration files
- `scripts/` — Build and utility scripts
- `plugins/` — Plugin system
- `data/` — Databases
- `packages/` — Distribution packages
- `deployment_packages/` — Portable deployment scripts
- `usb_templates/` — USB deployment templates
- `backups/` — Automatic update backups
- `temp/` — Temporary files
- `updates/` — Update packages
- `README.md` — This file
- `requirements.txt`, `pyproject.toml`, `setup.py` — Packaging
- `start_mcp.sh`, `start_mcp.bat`, `start_mcp.ps1` — Launch scripts
- `LICENSE`

> For the full, up-to-date file tree, see [`docs/FILETREE.txt`](docs/FILETREE.txt).

---

## Installation Options

### Standard Installation
```bash
git clone <repository>
cd agentic-workflow
python scripts/setup_wizard.py
```

### Portable Installation
```bash
# Download portable package
python scripts/build_portable.py

# Extract and run
./mcp_agentic_workflow
```

### USB Installation
```bash
# Create USB deployment
python scripts/universal_package_builder.py --usb

# Copy to USB drive and run
./start_mcp.sh
```

### Automatic Installation
```bash
# Run automatic installer
python scripts/install.py --auto
```

---

## Configuration

The system uses a centralized configuration system:

### Main Configuration
- `config/config.cfg` — Central configuration file
- `config/cursor-mcp.json` — Cursor IDE integration
- `config/vscode-settings.json` — VS Code integration
- `config/claude-mcp.json` — Claude integration

### Environment Variables
- `MCP_API_KEY` — API authentication key
- `MCP_PROJECT_PATH` — Project root path
- `MCP_VECTOR_BACKEND` — Vector database backend
- `MCP_LOG_LEVEL` — Logging level
- `MCP_DEBUG_MODE` — Debug mode toggle
- `MCP_AUTO_UPDATE_ENABLED` — Automatic updates
- `MCP_PERFORMANCE_MONITORING` — Performance monitoring

---

## Research & Vision

- Built on the latest research in memory, context, and agentic workflows.
- All features and architecture are documented and cross-referenced in `docs/` and `idea.txt`.
- See `docs/RESEARCH_SOURCES.md` for references and standards.
- Implements cutting-edge research in neural networks, genetic algorithms, and brain-inspired computing.

---

## Contributing

- See `CONTRIBUTING.md` and `idea.txt` for guidelines and vision.
- All contributions must be local, portable, and research-aligned.
- The system automatically updates and improves based on research and feedback.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Support

For support and documentation:
- **User Guide**: `docs/USER_GUIDE.md`
- **Developer Guide**: `docs/DEVELOPER_GUIDE.md`
- **API Documentation**: `docs/API_DOCUMENTATION.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Research Sources**: `docs/RESEARCH_SOURCES.md`

The system includes comprehensive help and documentation accessible via:
```bash
python scripts/help_system.py
```
