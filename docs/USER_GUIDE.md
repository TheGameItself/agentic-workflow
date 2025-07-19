# MCP User Guide (Updated)

## Overview
This guide explains how to use the upgraded MCP system.

- See [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
- See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for method details

## New Features
- **Three-Tier Memory:** All memory operations are now managed by a unified system (see UnifiedMemoryManager)
- **WebSocialEngine:** Enables web crawling, social media interaction, and digital identity management
- **GeneticTrigger:** Advanced adaptation, dual code/neural, and A/B testing for environmental triggers
- **P2P Benchmarking:** Secure, async benchmarking and global performance projection

## Usage Notes
- Memory, web/social, and genetic features are accessible via the main MCP interface
- P2P benchmarking and status visualization are available in the integration layer

## Cross-References
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

## Introduction

The MCP (Model Context Protocol) Server is a powerful, portable development accelerator that provides intelligent assistance for LLM-driven development workflows. This guide will help you get started and make the most of the MCP server's capabilities.

## Quick Start

### Installation

1. **Download the portable package** or use the installation script:
   ```bash
   python scripts/install.py --install-dir ~/.mcp_server
   ```

2. **Start the server**:
   ```bash
   # Using the launcher script
   ./start_mcp.sh  # Linux/macOS
   start_mcp.bat   # Windows
   
   # Or directly
   python -m src.mcp.cli server
   ```

3. **Verify the server is running**:
   ```bash
   python -m src.mcp.cli status
   ```

### First Project

1. **Create a new project**:
   ```bash
   python -m src.mcp.cli init-project "My Project" /path/to/project
   ```

2. **Add your project idea**:
   ```bash
   python -m src.mcp.cli add-memory "Build a web application for task management" --type "project_idea" --priority 5
   ```

3. **Start the workflow**:
   ```bash
   python -m src.mcp.cli start-step "init"
   ```

## Core Concepts

### 1. Workflows

Workflows are the main organizational structure for your projects. Each workflow consists of steps that guide the development process.

**Default Workflow Steps:**
- `init`: Project initialization and setup
- `research`: Research and requirements gathering
- `planning`: Project planning and architecture
- `development`: Active development
- `testing`: Testing and quality assurance
- `deployment`: Deployment and delivery

**Managing Workflows:**
```bash
# List workflow steps
python -m src.mcp.cli list-workflow-steps

# Start a step
python -m src.mcp.cli start-step "research"

# Complete a step
python -m src.mcp.cli complete-step "research"

# Add feedback to a step
python -m src.mcp.cli add-step-feedback "research" "Found excellent resources on React patterns" --impact 2
```

### 2. Tasks

Tasks are individual work items within your project. They can be organized hierarchically and have dependencies.

**Creating Tasks:**
```bash
# Create a simple task
python -m src.mcp.cli create-task "Setup development environment" "Install Node.js and dependencies" --priority 5

# Create a task with parent
python -m src.mcp.cli create-task "Install Node.js" "Download and install Node.js" --parent "Setup development environment"

# Create a meta task
python -m src.mcp.cli create-task "Code Review" "Review all code changes" --meta --meta-type "review"
```

**Managing Tasks:**
```bash
# List all tasks
python -m src.mcp.cli list-tasks

# Update task progress
python -m src.mcp.cli update-task-progress <task-id> 75 "Almost done with implementation"

# Add task notes
python -m src.mcp.cli add-task-note <task-id> "Important: Remember to test edge cases" --line 42 --file src/main.py

# Add task feedback
python -m src.mcp.cli add-task-feedback <task-id> "Good progress, but need more error handling" --impact 1
```

### 3. Memory

The memory system stores and retrieves information intelligently, helping you maintain context across sessions.

**Adding Memories:**
```bash
# Add project requirements
python -m src.mcp.cli add-memory "Must support real-time collaboration" --type "requirement" --priority 5

# Add research findings
python -m src.mcp.cli add-memory "WebSocket is best for real-time features" --type "research" --priority 4

# Add lessons learned
python -m src.mcp.cli add-memory "Always validate user input" --type "lesson" --priority 3
```

**Searching Memories:**
```bash
# Search for relevant memories
python -m src.mcp.cli search-memories "real-time collaboration" --limit 10

# Search by type
python -m src.mcp.cli search-memories "validation" --type "lesson"

# Search by tags
python -m src.mcp.cli search-memories "security" --tags "important" "critical"
```

### 4. Context Management

The context manager provides intelligent context retrieval and optimization.

**Exporting Context:**
```bash
# Export current context
python -m src.mcp.cli export-context --output context.json

# Get relevant context for a query
python -m src.mcp.cli get-relevant-context "database optimization" --limit 10
```

## Advanced Features

### 1. Experimental Lobes

The MCP server includes advanced cognitive engines inspired by human brain functions.

#### Alignment Engine
Ensures your work stays aligned with your goals and preferences.

```bash
# Analyze alignment of current work
python -m src.mcp.cli analyze-alignment --task-id <task-id> --preferences '{"quality": 0.9, "speed": 0.7}'
```

#### Pattern Recognition Engine
Identifies patterns in your work to improve efficiency with neural column architecture.

```bash
# Recognize patterns in recent tasks
python -m src.mcp.cli recognize-patterns --pattern-type "success_failure" --limit 20

# Process sensory input through pattern recognition
python -m src.mcp.cli process-sensory-input "user feedback data" --modality "textual"

# Get cross-lobe sensory data (In Progress)
python -m src.mcp.cli get-cross-lobe-data --modality "visual" --limit 10
```

#### Cross-Lobe Communication (Implemented)
The system supports comprehensive brain-inspired communication between different cognitive lobes:

```bash
# View cross-lobe sharing statistics
python -m src.mcp.cli cross-lobe-stats

# Monitor hormone-triggered data propagation
python -m src.mcp.cli monitor-hormone-propagation --duration 60

# Test cross-lobe sensory data sharing
python -m src.mcp.cli test-cross-lobe-sharing --modality visual --priority 0.8

# Configure propagation rules
python -m src.mcp.cli configure-propagation-rule \
  --source pattern_recognition \
  --targets alignment_engine,hormone_engine \
  --data-types success,error \
  --priority 0.7
```

#### P2P Genetic Data Exchange (Implemented)
Secure, decentralized sharing of optimizations using genetic-inspired encoding:

```bash
# Start P2P network node
python -m src.mcp.cli start-p2p-node --node-id organism_1 --port 10000

# Share genetic optimization data
python -m src.mcp.cli share-genetic-data \
  --data-type neural_network \
  --data-file model_weights.pkl \
  --target-peers peer1,peer2

# Monitor genetic data exchange
python -m src.mcp.cli monitor-genetic-exchange --duration 300

# View genetic network statistics
python -m src.mcp.cli genetic-network-stats
```

#### Mind Map Engine
Creates visual associations between concepts and ideas.

```bash
# Create a mind map node
python -m src.mcp.cli create-mindmap-node "Database Design" --type "concept"

# Create associations
python -m src.mcp.cli create-mindmap-association "Database Design" "User Authentication" --relationship "enables"
```

### 2. Performance Monitoring

Monitor your development performance and get insights.

```bash
# Get performance report
python -m src.mcp.cli performance-report --format json

# Get feedback analytics
python -m src.mcp.cli feedback-analytics --export analytics.json

# Run self-assessment
python -m src.mcp.cli self-assessment --detailed
```

### 3. Research Integration

Automate research and information gathering.

```bash
# Start research session
python -m src.mcp.cli start-research "React performance optimization" --sources 10

# Get research results
python -m src.mcp.cli get-research-results --topic "React performance"

# Validate research sources
python -m src.mcp.cli validate-sources --input sources.json
```

## IDE Integration

### VS Code

1. **Install the MCP extension** (if available)
2. **Configure the server** in VS Code settings:
   ```json
   {
     "mcp.servers": {
       "mcp-server": {
         "command": "python",
         "args": ["-m", "src.mcp.cli", "server"],
         "env": {
           "MCP_CONFIG_PATH": "/path/to/config/mcp-config.json"
         }
       }
     }
   }
   ```

### Cursor

1. **Add MCP server to Cursor configuration**
2. **Use MCP commands** through the command palette

### Claude Desktop

1. **Configure Claude to use the MCP server**
2. **Access MCP functions** through Claude's interface

## Configuration

### Server Configuration

Edit `config/mcp-config.json` to customize server settings:

```json
{
  "server": {
    "port": 3000,
    "host": "localhost",
    "debug": false
  },
  "database": {
    "path": "data/mcp.db",
    "backup_interval": 3600
  },
  "logging": {
    "level": "INFO",
    "file": "logs/mcp.log"
  },
  "security": {
    "api_key": "your-api-key",
    "rate_limit": 100
  }
}
```

### Environment Variables

Set these environment variables for customization:

```bash
export MCP_SERVER_PORT=3000
export MCP_DB_PATH=/path/to/database
export MCP_LOG_LEVEL=INFO
export MCP_API_KEY=your-api-key
```

## Best Practices

### 1. Project Organization

- **Use descriptive task names** that clearly indicate what needs to be done
- **Break large tasks** into smaller, manageable subtasks
- **Add context to memories** with relevant tags and metadata
- **Regular progress updates** help the system learn and improve

### 2. Memory Management

- **Tag memories appropriately** for better searchability
- **Use different memory types** (requirement, research, lesson, etc.)
- **Set appropriate priorities** to help with context selection
- **Regular memory cleanup** to maintain relevance

### 3. Workflow Optimization

- **Complete workflow steps** before moving to the next
- **Add feedback to steps** to improve future recommendations
- **Use meta tasks** for high-level project management
- **Leverage dependencies** to ensure proper task ordering

### 4. Performance Monitoring

- **Regular self-assessments** help identify improvement areas
- **Monitor performance metrics** to track progress
- **Review feedback analytics** to understand patterns
- **Use insights** to optimize your workflow

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if port is already in use
   - Verify Python environment is activated
   - Check log files for errors

2. **Database errors**
   - Ensure database directory exists and is writable
   - Check database file permissions
   - Try recreating the database

3. **Memory search not working**
   - Verify memory was added successfully
   - Check search query syntax
   - Ensure vector database is properly initialized

4. **Performance issues**
   - Monitor system resources
   - Check configuration settings
   - Review log files for bottlenecks

### Getting Help

1. **Check the logs** in the `logs/` directory
2. **Review configuration** in `config/mcp-config.json`
3. **Run diagnostics**:
   ```bash
   python -m src.mcp.cli diagnose
   ```
4. **Export debug information**:
   ```bash
   python -m src.mcp.cli export-debug --output debug_info.json
   ```

## Advanced Usage

### Custom Plugins

Create custom plugins to extend functionality:

1. **Create plugin directory**:
   ```bash
   mkdir plugins/my_plugin
   ```

2. **Create plugin files**:
   ```python
   # plugins/my_plugin/plugin.py
   class MyPlugin:
       def process(self, data):
           # Plugin logic here
           return {"result": "processed"}
   ```

3. **Load plugin**:
   ```bash
   python -m src.mcp.cli load-plugin plugins/my_plugin
   ```

### API Integration

Use the MCP server programmatically:

```python
from src.mcp.server import MCPServer

# Initialize server
server = MCPServer()

# Create task
task_id = server.task_manager.create_task("API Task", "Description")

# Add memory
memory_id = server.memory_manager.add_memory("API memory", "type")

# Get context
context = server.context_manager.export_context()
```

### Batch Operations

Perform operations on multiple items:

```bash
# Batch create tasks
python -m src.mcp.cli batch-create-tasks tasks.json

# Batch update progress
python -m src.mcp.cli batch-update-progress progress.json

# Batch add memories
python -m src.mcp.cli batch-add-memories memories.json
```

## Conclusion

The MCP server is designed to be your intelligent development companion. By following this guide and exploring the advanced features, you can significantly enhance your development workflow and productivity.

Remember to:
- Start with simple workflows and gradually explore advanced features
- Provide regular feedback to help the system learn
- Use the monitoring tools to track your progress
- Customize the configuration to match your needs

For more information, refer to the API documentation and advanced guides in the `docs/` directory. 