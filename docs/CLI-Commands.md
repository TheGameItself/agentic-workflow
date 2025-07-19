# 🖥️ MCP CLI Commands Reference

## Overview

The MCP Command Line Interface provides comprehensive access to all MCP functionality. This reference is organized by functional area for maximum usability.

**Quick Start**: See **[[cli/CLI-Quick-Reference]]** for the most commonly used commands.

## CLI Documentation Structure

### Essential Commands
- **[[cli/CLI-Quick-Reference]]** - Most commonly used commands with examples
- **[[cli/CLI-Project-Management]]** - Project initialization and management
- **[[cli/CLI-Research-Workflow]]** - Research and knowledge management
- **[[cli/CLI-Task-Management]]** - Task creation and tracking

### Advanced Commands
- **[[cli/CLI-Memory-System]]** - Memory operations and optimization
- **[[cli/CLI-Context-Export]]** - Context generation for LLMs
- **[[cli/CLI-Performance]]** - System monitoring and optimization
- **[[cli/CLI-P2P-Network]]** - Peer-to-peer collaboration

### System Commands
- **[[cli/CLI-Configuration]]** - System configuration and setup
- **[[cli/CLI-Troubleshooting]]** - Diagnostic and repair commands
- **[[cli/CLI-Integration]]** - IDE and service integration
- **[[cli/CLI-Advanced-Features]]** - Advanced system features

## Quick Navigation by Use Case

### New Users - Get Started Fast
1. **[[cli/CLI-Quick-Reference]]** - Essential commands
2. **[[cli/CLI-Project-Management]]** - Create your first project
3. **[[INSTALLATION_GUIDE]]** - Installation help
4. **[[USER_GUIDE]]** - Complete user guide

### Daily Development Workflow
1. **[[cli/CLI-Project-Management]]** - Project operations
2. **[[cli/CLI-Task-Management]]** - Task tracking
3. **[[cli/CLI-Context-Export]]** - LLM context generation
4. **[[cli/CLI-Memory-System]]** - Knowledge management

### Research & Learning
1. **[[cli/CLI-Research-Workflow]]** - Research automation
2. **[[cli/CLI-Memory-System]]** - Knowledge storage
3. **[[cli/CLI-Context-Export]]** - Research context export
4. **[[cli/CLI-Advanced-Features]]** - Advanced research tools

### System Administration
1. **[[cli/CLI-Configuration]]** - System setup
2. **[[cli/CLI-Performance]]** - Monitoring and optimization
3. **[[cli/CLI-Troubleshooting]]** - Diagnostic tools
4. **[[cli/CLI-Integration]]** - IDE and service setup

### Collaboration & Networking
1. **[[cli/CLI-P2P-Network]]** - Network participation
2. **[[cli/CLI-Advanced-Features]]** - Collaboration features
3. **[[cli/CLI-Integration]]** - Team integration
4. **[[P2P-Network]]** - P2P system documentation

## Command Categories Overview

### Project & Workflow Commands
- `init-project` - Initialize new projects
- `project-status` - Check project status
- `start-research` - Begin research workflows
- `workflow-status` - Monitor workflow progress

### Task Management Commands
- `create-task` - Create new tasks
- `list-tasks` - List and filter tasks
- `update-task-progress` - Update task progress
- `task-tree` - View task hierarchy

### Memory & Knowledge Commands
- `add-memory` - Store information
- `search-memories` - Find stored information
- `consolidate-memories` - Optimize memory storage
- `memory-stats` - View memory statistics

### Context & Export Commands
- `export-context` - Generate LLM context
- `get-context-pack` - Get specialized context
- `export-research` - Export research findings
- `export-tasks` - Export task information

### System & Performance Commands
- `health-check` - System health verification
- `performance-report` - Performance analysis
- `optimize-system` - System optimization
- `diagnose-performance` - Performance diagnostics

### Network & Collaboration Commands
- `p2p-connect` - Connect to P2P network
- `p2p-status` - Network status
- `p2p-share-optimization` - Share improvements
- `p2p-sync-optimizations` - Sync network improvements

## Basic Usage Patterns

### Command Syntax
```bash
mcp <command> [arguments] [options]
```

### Common Options
- `--help` - Show command help
- `--verbose` - Detailed output
- `--format json|markdown|yaml` - Output format
- `--limit N` - Limit results
- `--priority N` - Set priority (0.0-1.0)

### Global Options
- `--config PATH` - Custom config file
- `--debug` - Enable debug mode
- `--quiet` - Suppress output
- `--version` - Show version information

## Quick Examples

### Essential Daily Commands
```bash
# Check system and project status
mcp health-check
mcp project-status

# Export context for AI development
mcp export-context --format json --max-tokens 2000

# Add important information
mcp add-memory "Key insight" --type "learning"

# Update task progress
mcp update-task-progress "current-task" 75
```

### Complete Project Workflow
```bash
# 1. Initialize project
mcp init-project "My AI App" --type "web-application"

# 2. Start research
mcp start-research "Modern web development"

# 3. Create tasks
mcp create-task "Frontend Development" --priority 5

# 4. Export context for AI assistant
mcp export-context --types "tasks,research" --format json
```

## Getting Help

### Command Help
```bash
# General help
mcp --help

# Command-specific help
mcp <command> --help

# Examples for specific commands
mcp examples --command "export-context"
```

### Documentation Resources
- **[[USER_GUIDE]]** - Complete user functionality guide
- **[[API_DOCUMENTATION]]** - Programmatic API reference
- **[[Troubleshooting]]** - Common issues and solutions
- **[[INSTALLATION_GUIDE]]** - Installation and setup help

### Interactive Help
```bash
# Interactive command builder
mcp interactive

# Show available commands
mcp list-commands

# Show command examples
mcp examples --category "project-management"
```

## Integration Examples

### Shell Integration
```bash
# Add to ~/.bashrc or ~/.zshrc
alias mcps="mcp project-status"
alias mcpt="mcp list-tasks --status active"
alias mcpe="mcp export-context --format json"
```

### IDE Integration
```json
// VS Code tasks.json example
{
  "label": "MCP: Export Context",
  "type": "shell",
  "command": "mcp",
  "args": ["export-context", "--format", "json"]
}
```

## Related Documentation

### Core Documentation
- **[[USER_GUIDE]]** - User-facing functionality
- **[[API_DOCUMENTATION]]** - Programmatic access
- **[[DEVELOPER_GUIDE]]** - Development patterns
- **[[ARCHITECTURE]]** - System architecture

### Specialized Guides
- **[[INSTALLATION_GUIDE]]** - Installation and setup
- **[[Performance-Optimization]]** - System optimization
- **[[Troubleshooting]]** - Problem resolution
- **[[IDE_INTEGRATION]]** - IDE setup and configuration

---

*This CLI reference is designed for maximum usability. For detailed command information, follow the cross-links to focused command documentation.*