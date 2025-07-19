# MCP CLI Commands Reference

## Quick Command Reference

### Project Management
```bash
mcp init-project "Project Name" --type "web-app"
mcp project-status
mcp switch-project "Project Name"
```

### Research & Knowledge
```bash
mcp start-research "Research Topic"
mcp add-research-topic "Subtopic"
mcp add-finding "Finding" --source "URL" --confidence 0.9
mcp search-research "query"
```

### Task Management
```bash
mcp create-task "Task Name" --priority 5
mcp list-tasks --status pending
mcp update-task-progress "Task" 75 --note "Progress update"
mcp task-tree --show-progress
```

### Memory System
```bash
mcp add-memory "Information" --type "best-practice"
mcp search-memories "query" --limit 10
mcp consolidate-memories
mcp memory-stats
```

### Context Generation
```bash
mcp export-context --format json --max-tokens 2000
mcp get-context-pack "development" --include-code
```

### Performance & Monitoring
```bash
mcp performance-report
mcp resource-status
mcp optimize-system --target memory
mcp health-check
```

### P2P Network
```bash
mcp p2p-connect --network "development"
mcp p2p-share-optimization "optimization-name"
mcp p2p-sync-optimizations
mcp p2p-status
```

### System Management
```bash
mcp system-status
mcp clear-cache --type all
mcp reset-config --backup
mcp help [command]
```

## Common Workflows

### New Project Setup
```bash
mcp init-project "My App" --type "web-app"
mcp start-research "Modern web development"
mcp create-task "Frontend" --priority 5
mcp export-context --format json
```

### Daily Development
```bash
mcp task-tree --show-progress
mcp export-context --types "tasks,progress"
mcp performance-report --period day
```

### Research Session
```bash
mcp start-research "Topic"
mcp add-finding "Key insight" --confidence 0.8
mcp add-memory "Best practice" --type "best-practice"
```

For detailed command documentation, see the full CLI reference.