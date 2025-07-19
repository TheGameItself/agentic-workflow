# ⚡ CLI Quick Reference

## Most Used Commands

### Project Management
```bash
# Initialize new project
mcp init-project "Project Name" --type "web-application"

# Check project status
mcp project-status

# Show project questions
mcp show-questions
```

### Research & Knowledge
```bash
# Start research
mcp start-research "Research Topic"

# Add research finding
mcp add-finding "Key insight" --source "reference" --confidence 0.9

# Search memories
mcp search-memories "query" --limit 10
```

### Task Management
```bash
# Create task
mcp create-task "Task Name" --priority 5

# Update progress
mcp update-task-progress "task-name" 75

# List tasks
mcp list-tasks --status active
```

### Context Export
```bash
# Export for LLM
mcp export-context --format json --max-tokens 2000

# Export specific types
mcp export-context --types "tasks,memories,research" --format json

# Get context pack
mcp get-context-pack "development" --include-code-examples
```

### Memory Operations
```bash
# Add memory
mcp add-memory "Important information" --type "learning" --priority 0.8

# Consolidate memories
mcp consolidate-memories

# Memory statistics
mcp memory-stats
```

### System Operations
```bash
# Health check
mcp health-check

# Performance report
mcp performance-report

# System optimization
mcp optimize-system
```

## Command Syntax Patterns

### Basic Syntax
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
- `--version` - Show version

## Quick Examples

### Complete Workflow Example
```bash
# 1. Initialize project
mcp init-project "AI Chat Bot" --type "python-package"

# 2. Start research
mcp start-research "Conversational AI patterns"
mcp add-research-topic "Natural language processing"

# 3. Create tasks
mcp create-task "Design conversation flow" --priority 5
mcp create-task "Implement NLP pipeline" --priority 4
mcp create-task "Create user interface" --priority 3

# 4. Add knowledge
mcp add-memory "Use transformer models for better understanding" --type "best-practice"

# 5. Export context for AI assistant
mcp export-context --types "tasks,research,memories" --max-tokens 1500 --format json
```

### Research Workflow Example
```bash
# Start comprehensive research
mcp start-research "Modern web development"
mcp add-research-topic "React performance optimization"
mcp add-research-topic "API security best practices"
mcp add-research-topic "Database design patterns"

# Add findings as you research
mcp add-finding "React.memo prevents unnecessary re-renders" \
  --source "React documentation" \
  --confidence 0.95 \
  --tags "react,performance"

# Export research for AI coding session
mcp export-context --types "research,findings" --format markdown
```

### Daily Development Example
```bash
# Morning routine
mcp project-status
mcp list-tasks --status active --limit 5
mcp export-context --types "tasks,progress" --format json

# During development
mcp add-memory "Fixed authentication bug by updating JWT validation" --type "solution"
mcp update-task-progress "implement-auth" 90 --note "Almost complete, testing remaining"

# End of day
mcp consolidate-memories
mcp performance-report
mcp task-tree --show-progress
```

## Error Handling

### Common Error Patterns
```bash
# Check if command succeeded
if mcp health-check; then
    echo "System healthy"
else
    echo "System issues detected"
    mcp diagnose-performance
fi
```

### Debugging Commands
```bash
# Enable debug mode
mcp --debug command-name

# Check system logs
mcp logs --level error --tail 50

# Diagnose issues
mcp diagnose-performance
mcp system-status --detailed
```

## Integration Examples

### VS Code Integration
```json
// .vscode/tasks.json
{
  "label": "MCP: Export Context",
  "type": "shell",
  "command": "mcp",
  "args": ["export-context", "--format", "json", "--max-tokens", "2000"]
}
```

### Shell Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
alias mcps="mcp project-status"
alias mcpt="mcp list-tasks --status active"
alias mcpe="mcp export-context --format json --max-tokens 2000"
alias mcph="mcp health-check"
```

### Automation Scripts
```bash
#!/bin/bash
# daily-mcp-routine.sh

echo "🌅 Starting daily MCP routine..."

# Check system health
if ! mcp health-check; then
    echo "⚠️  System health issues detected"
    mcp diagnose-performance
fi

# Show today's tasks
echo "📋 Today's active tasks:"
mcp list-tasks --status active --limit 10

# Consolidate memories from yesterday
echo "🧠 Consolidating memories..."
mcp consolidate-memories

# Export context for morning development session
echo "📤 Exporting context..."
mcp export-context --types "tasks,memories,progress" --format json > morning-context.json

echo "✅ Daily routine complete!"
```

## Related Documentation

- **[[CLI-Project-Management]]** - Detailed project commands
- **[[CLI-Research-Workflow]]** - Research command details
- **[[CLI-Task-Management]]** - Task management commands
- **[[CLI-Memory-System]]** - Memory operation commands
- **[[../USER_GUIDE]]** - Complete user guide
- **[[../Troubleshooting]]** - Troubleshooting help

---

*This quick reference covers the most commonly used MCP CLI commands. For detailed information on specific command categories, follow the cross-links above.*