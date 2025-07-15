# MCP Server CLI Usage Guide

## Overview

The MCP Server provides a comprehensive command-line interface for managing projects, workflows, tasks, and system operations. This guide covers all available CLI commands and their usage patterns.

## Installation and Setup

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd agentic-workflow

# Install dependencies
pip install -r requirements.txt

# Initialize the project
python -m src.mcp.cli init
```

### Environment Configuration

```bash
# Set up environment variables
export MCP_SERVER_PORT=3000
export MCP_DB_PATH=/path/to/database
export MCP_LOG_LEVEL=INFO

# Or use configuration file
cp config/mcp-config.example.json config/mcp-config.json
# Edit config/mcp-config.json with your settings
```

## Core Commands

### Project Management

#### Initialize Project
```bash
# Basic project initialization
python -m src.mcp.cli init-project "My Project" /path/to/project

# With custom configuration
python -m src.mcp.cli init-project "My Project" /path/to/project --config custom-config.json

# Initialize with specific template
python -m src.mcp.cli init-project "My Project" /path/to/project --template web-app
```

#### Project Status
```bash
# Get current project status
python -m src.mcp.cli project-status

# Get detailed status with metrics
python -m src.mcp.cli project-status --detailed

# Export project status to file
python -m src.mcp.cli project-status --export status.json
```

#### Project Operations
```bash
# Create new project
python -m src.mcp.cli create "Project Name" /path/to/project

# List all projects
python -m src.mcp.cli list-projects

# Switch to different project
python -m src.mcp.cli switch-project /path/to/project

# Export project data
python -m src.mcp.cli export-project --output project-backup.json

# Import project data
python -m src.mcp.cli import-project project-backup.json
```

### Task Management

#### Task Operations
```bash
# Create a new task
python -m src.mcp.cli create-task "Task Title" "Task description" --priority 5

# Create task with parent
python -m src.mcp.cli create-task "Subtask" "Description" --parent "Parent Task" --priority 3

# Create meta task
python -m src.mcp.cli create-task "Review Tasks" "Review all tasks" --meta --meta-type review

# List all tasks
python -m src.mcp.cli list-tasks

# List tasks with filters
python -m src.mcp.cli list-tasks --status active --priority high

# Get task details
python -m src.mcp.cli get-task <task-id>

# Update task
python -m src.mcp.cli update-task <task-id> --title "New Title" --priority 4

# Delete task
python -m src.mcp.cli delete-task <task-id>
```

#### Task Progress
```bash
# Update task progress
python -m src.mcp.cli update-task-progress <task-id> 75 "Almost done"

# Add task note
python -m src.mcp.cli add-task-note <task-id> "Important note" --line 42 --file src/main.py

# Add task feedback
python -m src.mcp.cli add-task-feedback <task-id> "Good progress" --impact 2 --principle "efficiency"

# Get task tree
python -m src.mcp.cli task-tree --root <task-id>
```

#### Task Dependencies
```bash
# Add task dependency
python -m src.mcp.cli add-task-dependency <task-id> <depends-on-task-id>

# List task dependencies
python -m src.mcp.cli list-task-dependencies <task-id>

# Remove task dependency
python -m src.mcp.cli remove-task-dependency <dependency-id>
```

### Workflow Management

#### Workflow Operations
```bash
# Create workflow
python -m src.mcp.cli create-workflow "My Workflow" /path/to/project

# List workflow steps
python -m src.mcp.cli list-workflow-steps

# Start workflow step
python -m src.mcp.cli start-step "init"

# Complete workflow step
python -m src.mcp.cli complete-step "init"

# Get workflow status
python -m src.mcp.cli workflow-status

# Add step feedback
python -m src.mcp.cli add-step-feedback "init" "Step completed successfully" --impact 1
```

#### Workflow Configuration
```bash
# Register custom step
python -m src.mcp.cli register-step "custom" "Custom step description"

# Modify step
python -m src.mcp.cli modify-step "custom" --description "Updated description"

# Set step dependencies
python -m src.mcp.cli set-step-dependencies "step2" "step1"

# Set next steps
python -m src.mcp.cli set-next-steps "step1" "step2" "step3"
```

### Memory Management

#### Memory Operations
```bash
# Add memory
python -m src.mcp.cli add-memory "Important information" --type "requirement" --priority 5

# Search memories
python -m src.mcp.cli search-memories "search query" --limit 10

# Get memory by ID
python -m src.mcp.cli get-memory <memory-id>

# Update memory
python -m src.mcp.cli update-memory <memory-id> --text "Updated text" --priority 3

# Delete memory
python -m src.mcp.cli delete-memory <memory-id>

# Export memories
python -m src.mcp.cli export-memories --output memories.json

# Import memories
python -m src.mcp.cli import-memories memories.json
```

#### Memory Search
```bash
# Search by type
python -m src.mcp.cli search-memories "query" --type "requirement"

# Search by tags
python -m src.mcp.cli search-memories "query" --tags "important" "urgent"

# Find similar memories
python -m src.mcp.cli find-similar-memories <memory-id> --limit 5
```

### Context Management

#### Context Operations
```bash
# Export context
python -m src.mcp.cli export-context --output context.json

# Import context
python -m src.mcp.cli import-context context.json

# Get relevant context
python -m src.mcp.cli get-relevant-context "query" --limit 10

# Optimize context
python -m src.mcp.cli optimize-context --max-size 1000
```

### System Management

#### Server Operations
```bash
# Start MCP server
python -m src.mcp.cli start-server --port 3000

# Stop server
python -m src.mcp.cli stop-server

# Get server status
python -m src.mcp.cli server-status

# Restart server
python -m src.mcp.cli restart-server
```

#### System Monitoring
```bash
# Get system metrics
python -m src.mcp.cli system-metrics

# Get performance report
python -m src.mcp.cli performance-report --format json

# Get feedback analytics
python -m src.mcp.cli feedback-analytics --export analytics.json

# Run self-assessment
python -m src.mcp.cli self-assessment --detailed
```

#### Database Operations
```bash
# Backup database
python -m src.mcp.cli backup-db --output backup.sqlite

# Restore database
python -m src.mcp.cli restore-db backup.sqlite

# Optimize database
python -m src.mcp.cli optimize-db

# Reset database
python -m src.mcp.cli reset-db --confirm
```

### Plugin Management

#### Plugin Operations
```bash
# List installed plugins
python -m src.mcp.cli list-plugins

# Install plugin
python -m src.mcp.cli install-plugin /path/to/plugin

# Enable plugin
python -m src.mcp.cli enable-plugin plugin-name

# Disable plugin
python -m src.mcp.cli disable-plugin plugin-name

# Remove plugin
python -m src.mcp.cli remove-plugin plugin-name

# Update plugin
python -m src.mcp.cli update-plugin plugin-name
```

### Research and Development

#### Research Operations
```bash
# Start research session
python -m src.mcp.cli start-research "Research topic" --sources 5

# Get research results
python -m src.mcp.cli get-research-results --topic "topic"

# Export research data
python -m src.mcp.cli export-research --output research.json

# Validate research sources
python -m src.mcp.cli validate-sources --input sources.json
```

#### Development Tools
```bash
# Generate code
python -m src.mcp.cli generate-code "function description" --language python

# Analyze code
python -m src.mcp.cli analyze-code /path/to/file.py

# Run tests
python -m src.mcp.cli run-tests --path /path/to/tests

# Generate documentation
python -m src.mcp.cli generate-docs --input /path/to/code --output docs/
```

## Advanced Usage

### Batch Operations

```bash
# Batch create tasks
python -m src.mcp.cli batch-create-tasks tasks.json

# Batch update progress
python -m src.mcp.cli batch-update-progress progress.json

# Batch add memories
python -m src.mcp.cli batch-add-memories memories.json
```

### Automation Scripts

```bash
# Run automated workflow
python -m src.mcp.cli run-workflow workflow.json

# Schedule periodic tasks
python -m src.mcp.cli schedule-task "Daily backup" --cron "0 2 * * *"

# Monitor and alert
python -m src.mcp.cli monitor --alert-on-failure --email admin@example.com
```

### Integration Commands

```bash
# Export for IDE integration
python -m src.mcp.cli export-ide-config --ide vscode --output .vscode/settings.json

# Generate MCP configuration
python -m src.mcp.cli generate-mcp-config --output mcp-config.json

# Setup git hooks
python -m src.mcp.cli setup-git-hooks --hooks pre-commit post-commit
```

## Configuration

### Configuration File Format

```json
{
  "server": {
    "port": 3000,
    "host": "localhost",
    "debug": false
  },
  "database": {
    "path": "/path/to/database.sqlite",
    "backup_interval": 3600
  },
  "logging": {
    "level": "INFO",
    "file": "/path/to/logs/mcp.log"
  },
  "security": {
    "api_key": "your-api-key",
    "rate_limit": 100
  }
}
```

### Environment Variables

```bash
# Server configuration
export MCP_SERVER_PORT=3000
export MCP_SERVER_HOST=localhost
export MCP_DEBUG=false

# Database configuration
export MCP_DB_PATH=/path/to/database.sqlite
export MCP_DB_BACKUP_INTERVAL=3600

# Logging configuration
export MCP_LOG_LEVEL=INFO
export MCP_LOG_FILE=/path/to/logs/mcp.log

# Security configuration
export MCP_API_KEY=your-api-key
export MCP_RATE_LIMIT=100
```

## Error Handling

### Common Error Codes

- `E001`: Database connection error
- `E002`: Invalid task ID
- `E003`: Workflow step not found
- `E004`: Memory not found
- `E005`: Permission denied
- `E006`: Invalid configuration
- `E007`: Plugin not found
- `E008`: Network error

### Debug Mode

```bash
# Enable debug mode
python -m src.mcp.cli --debug <command>

# Get detailed error information
python -m src.mcp.cli --verbose <command>

# Log to file
python -m src.mcp.cli --log-file debug.log <command>
```

## Best Practices

1. **Use Configuration Files**: Store settings in configuration files rather than command line arguments
2. **Batch Operations**: Use batch commands for multiple operations
3. **Regular Backups**: Schedule regular database backups
4. **Monitor Performance**: Use performance monitoring commands regularly
5. **Error Handling**: Always check return codes and error messages
6. **Security**: Use API keys and rate limiting in production
7. **Logging**: Enable appropriate logging levels for debugging

## Examples

### Complete Project Setup

```bash
# Initialize project
python -m src.mcp.cli init-project "Web Application" /path/to/webapp

# Create initial tasks
python -m src.mcp.cli create-task "Requirements Analysis" "Analyze project requirements" --priority 5
python -m src.mcp.cli create-task "Design Architecture" "Design system architecture" --priority 4
python -m src.mcp.cli create-task "Implement Core" "Implement core functionality" --priority 3

# Start workflow
python -m src.mcp.cli start-step "init"

# Add project memories
python -m src.mcp.cli add-memory "Must be scalable and secure" --type "requirement" --priority 5

# Get project status
python -m src.mcp.cli project-status --detailed
```

### Daily Workflow

```bash
# Check project status
python -m src.mcp.cli project-status

# List active tasks
python -m src.mcp.cli list-tasks --status active

# Update task progress
python -m src.mcp.cli update-task-progress <task-id> 50 "Halfway done"

# Add daily notes
python -m src.mcp.cli add-task-note <task-id> "Daily progress update"

# Run self-assessment
python -m src.mcp.cli self-assessment

# Backup project
python -m src.mcp.cli backup-db --output daily-backup.sqlite
```

This CLI usage guide provides comprehensive coverage of all available commands and their usage patterns. 