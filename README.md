# 🚀 MCP Agentic Workflow Accelerator - AI Development Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Compatible-brightgreen.svg)](https://modelcontextprotocol.io/)
[![Local](https://img.shields.io/badge/Local-Only-orange.svg)](https://github.com/search?q=local+only+python)
[![Portable](https://img.shields.io/badge/Portable-Yes-purple.svg)](https://github.com/search?q=portable+python+application)

> **Accelerate AI-powered development workflows with a portable, local MCP server for LLMs**

- **Python 3.8+** | **MIT License** | **MCP-Compatible** | **Local-Only** | **Portable** | "Advanced" API

---

## Overview

The MCP Agentic Workflow Accelerator transforms single prompts into complete applications using intelligent project management, memory systems, and workflow orchestration. It is fully portable, requires no network, and is designed for seamless IDE integration.

---

## Key Features

- **Project Initialization**: Instantly structure new projects from ideas.
- **Intelligent Memory**: Vector-based, cross-project, and quality-assessed.
- **Task Management**: Hierarchical, priority-based, with partial completion.
- **Research Automation**: Guided research and findings tracking.
- **Context Optimization**: Minimal, relevant context for LLMs.
- **Safety**: Accuracy-critical task protection.
- **IDE Integration**: Works with Cursor, VS Code, and more.

---

## Brain-Inspired Architecture & Neural Networks

### Modular "Lobe" System

The MCP server is inspired by the human brain, with each "lobe" or engine responsible for a specialized cognitive function:

- **Alignment Engine**: LLM-based preference modeling and feedback learning.
- **Pattern Recognition Engine**: Neural column simulation for pattern and anomaly detection.
- **Simulated Reality Engine**: Entity-relationship-state tracking for world modeling.
- **Dreaming Engine**: Multi-agent scenario simulation for creative insight and risk assessment.
- **Mind Map Engine**: Graph-based conceptual association and knowledge mapping.
- **Scientific Process Engine**: Structured hypothesis-experiment-analysis workflow.
- **Split Brain AB Test Engine**: Parallel A/B testing for continuous self-improvement.
- **Multi-LLM Orchestrator**: Task routing and load balancing across multiple LLMs.
- **Advanced Engram Engine**: Neural network-based memory compression, predictive modeling, and adaptive learning.

### Data Flow

- **Request Flow**:  
  `Client → API Layer → Core Service → Experimental Lobe → Data Layer → Response`
- **Memory Flow**:  
  `Input → Preprocessing → Vectorization → Storage → Retrieval → Relevance Scoring → Context Assembly`
- **Learning Flow**:  
  `Feedback → Pattern Recognition → Model Update → Optimization → Deployment`

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

---

## CLI Commands (Summary)

- **Project**: `init-project`, `show-questions`, `answer-question`, `project-status`
- **Workflow**: `start-research`, `add-research-topic`, `add-finding`, `start-planning`, `workflow-status`
- **Memory**: `add-memory`, `search-memories`, `get-memory`
- **Task**: `create-task`, `list-tasks`, `update-task-progress`, `add-task-note`, `add-task-dependency`, `task-tree`
- **Context**: `export-context`, `get-context-pack`
- **Advanced**: `bulk-update-task-status`, `add-task-feedback`, `statistics`

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
- `README.md` — This file
- `requirements.txt`, `pyproject.toml`, `setup.py` — Packaging
- `start_mcp.sh`, `start_mcp.bat`, `start_mcp.ps1` — Launch scripts
- `idea.txt` — **Never remove or modify**
- `.gitignore`, `LICENSE`

> For the full, up-to-date file tree, see [`docs/FILETREE.txt`](docs/FILETREE.txt).

---

## Research & Vision

- Built on the latest research in memory, context, and agentic workflows.
- All features and architecture are documented and cross-referenced in `docs/` and `idea.txt`.
- See `docs/RESEARCH_SOURCES.md` for references and standards.

---

## Contributing

- See `CONTRIBUTING.md` and `idea.txt` for guidelines and vision.
- All contributions must be local, portable, and research-aligned.

---

## License

MIT License. See [LICENSE](LICENSE).

---
