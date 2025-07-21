# MCP CLI Commands Reference

## Overview

This document provides a comprehensive reference for all MCP command-line interface commands. The CLI provides access to all major system functionality including project management, memory operations, performance monitoring, and troubleshooting.

## Command Categories

### Project Management Commands

#### Project Initialization
```bash
# Initialize new project
mcp init-project "Project Name" --type "web-app"
mcp init-project "E-commerce Site" --type "full-stack"
mcp init-project "Data Analysis Tool" --type "python-package"

# Project status and configuration
mcp project-status
mcp update-project --description "Updated description"
mcp set-preference "coding_style" "functional"
```

#### Project Questions and Answers
```bash
# Show project questions
mcp show-questions

# Answer specific questions
mcp answer-question "What framework should we use?" "React with TypeScript"

# Project configuration
mcp update-project --add-tag "ai" --add-tag "ecommerce"
```

### Research and Discovery Commands

#### Research Management
```bash
# Start research phase
mcp start-research "Machine Learning in Web Development"

# Add research topics
mcp add-research-topic "TensorFlow.js integration"
mcp add-research-topic "Model deployment strategies"

# Add research findings
mcp add-finding "TensorFlow.js can run models directly in browser" \
  --source "https://tensorflow.org/js" \
  --confidence 0.9 \
  --tags "ml,browser,performance"

# Search and export research
mcp search-research "tensorflow" --limit 10
mcp export-research --format markdown
```

#### Workflow Status
```bash
# Check workflow status
mcp workflow-status
mcp start-planning
```

### Memory System Commands

#### Memory Operations
```bash
# Add different types of memories
mcp add-memory "Use React hooks for state management" --type "best-practice"
mcp add-memory "API endpoint: /api/v1/users" --type "reference"
mcp add-memory "Bug: Login fails on Safari" --type "issue" --priority 5

# Search memories
mcp search-memories "react hooks" --limit 5
mcp search-memories --type "issue" --priority-min 4

# Memory management
mcp consolidate-memories  # Optimize memory storage
mcp memory-stats          # View memory usage statistics
mcp get-memory "specific-key"
```

### Task Management Commands

#### Task Creation and Organization
```bash
# Create hierarchical tasks
mcp create-task "User Authentication System" --priority 5
mcp create-task "Login Component" --parent "User Authentication System"
mcp create-task "Registration Form" --parent "User Authentication System"

# Task dependencies and progress
mcp add-task-dependency "Login Component" "Database Schema"
mcp update-task-progress "Login Component" 75 \
  --note "Basic authentication working, need to add validation"

# Task management
mcp list-tasks
mcp task-tree
mcp task-tree --show-progress --show-notes
mcp complete-task "Database Schema"
```

#### Advanced Task Operations
```bash
# Bulk operations
mcp bulk-update-task-status --status "in-progress" \
  --filter "priority>=4" --filter "status=pending"

# Task feedback and analytics
mcp add-task-feedback "Login Component" \
  --feedback "Consider using JWT tokens for better security" \
  --type "suggestion"

mcp task-analytics --show-velocity --show-bottlenecks
```

### Context Management Commands

#### Context Export
```bash
# Basic context export
mcp export-context --format json

# Filtered context export
mcp export-context \
  --types "tasks,memories,research" \
  --max-tokens 2000 \
  --priority-min 3 \
  --format json

# Context packs for specific purposes
mcp get-context-pack "development" --include-code-examples
mcp get-context-pack "debugging" --include-error-logs
mcp get-context-pack "planning" --include-architecture
```

### Performance and Optimization Commands

#### System Performance
```bash
# Monitor system performance
mcp performance-report
mcp resource-status

# Optimize system
mcp optimize-system --target "memory"
mcp optimize-system --target "speed"
mcp optimize-system --aggressive

# Performance analytics
mcp performance-history --days 7
mcp performance-compare --baseline "last-week"
```

#### Workflow Optimization
```bash
# Analyze workflow efficiency
mcp workflow-analytics
mcp identify-bottlenecks

# Optimize task scheduling
mcp optimize-task-schedule
mcp suggest-task-priorities
```

### P2P Network Commands

#### Network Participation
```bash
# Join P2P network
mcp p2p-connect --network "development"

# Share optimizations
mcp p2p-share-optimization "task-prioritization" \
  --description "Improved task priority algorithm"

# Receive optimizations
mcp p2p-sync-optimizations
mcp p2p-list-available-optimizations

# Network status
mcp p2p-status
mcp p2p-network-health
```

#### Collaboration Features
```bash
# Share project templates
mcp p2p-share-template "react-typescript-starter"

# Collaborate on research
mcp p2p-share-research "AI development patterns"
```

### Genetic Evolution Commands

#### System Learning
```bash
# View learning progress
mcp genetic-status
mcp learning-analytics

# Manual optimization triggers
mcp trigger-evolution --focus "task-management"
mcp trigger-evolution --focus "context-generation"

# Learning preferences
mcp set-learning-preference "exploration_rate" 0.3
mcp set-learning-preference "adaptation_speed" "moderate"

# Evolution analytics
mcp optimization-impact
mcp genetic-evolution-stats
mcp adaptation-history
```

### Troubleshooting Commands

#### System Diagnostics
```bash
# Basic health checks
mcp --version
mcp health-check

# System diagnostics
mcp diagnose-performance
mcp identify-bottlenecks
mcp system-diagnostics
```

#### Cache and Configuration Management
```bash
# Clear caches
mcp clear-cache --all
mcp clear-cache --type memory
mcp clear-cache --type search

# Configuration management
mcp reset-config --backup
mcp config validate
mcp config reset
```

#### Component Management
```bash
# Reinstall components
mcp reinstall --component memory
mcp reinstall --component genetic-system

# Component status
mcp component-status memory
mcp component-status genetic-system
mcp component-status p2p-network
```

### Advanced Commands

#### Statistics and Analytics
```bash
# System statistics
mcp statistics
mcp productivity-report --period week
mcp productivity-trends --metric "tasks-completed"

# Workflow efficiency
mcp workflow-efficiency
mcp time-tracking-summary
```

#### Integration Commands
```bash
# IDE integration
mcp configure-ide vscode
mcp configure-ide cursor
mcp ide-status

# Git integration
mcp git-link --repo "https://github.com/user/project"
mcp git-sync-tasks
mcp git-sync-progress
```

## Command Options and Flags

### Global Options
- `--verbose, -v`: Increase output verbosity
- `--quiet, -q`: Suppress non-essential output
- `--format`: Output format (json, yaml, table, markdown)
- `--config`: Specify custom configuration file
- `--debug`: Enable debug mode

### Common Filters
- `--priority-min N`: Filter by minimum priority
- `--priority-max N`: Filter by maximum priority
- `--type TYPE`: Filter by type
- `--status STATUS`: Filter by status
- `--limit N`: Limit number of results
- `--since DATE`: Filter by date range

## Environment Variables

### Configuration Variables
- `MCP_PROJECT_PATH`: Default project path
- `MCP_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MCP_DEBUG_MODE`: Enable debug mode (true/false)
- `MCP_CONFIG_FILE`: Custom configuration file path

### Performance Variables
- `MCP_MEMORY_LIMIT`: Memory usage limit
- `MCP_WORKER_THREADS`: Number of worker threads
- `MCP_CACHE_SIZE`: Cache size limit

## Related Documentation

- [[API_DOCUMENTATION]] - Programmatic API reference
- [[USER_GUIDE]] - User-facing functionality guide
- [[DEVELOPER_GUIDE]] - Development and integration guide
- [[Troubleshooting]] - Detailed troubleshooting guide
- [[Performance-Optimization]] - Performance tuning guide
- [[Memory-System]] - Memory system commands and APIs
- [[P2P-Network]] - P2P networking commands
- [[Genetic-System]] - Genetic evolution commands

## Examples and Use Cases

### Complete Project Workflow
```bash
# 1. Initialize project
mcp init-project "E-commerce Platform" --type "full-stack"

# 2. Start research phase
mcp start-research "Modern e-commerce architecture"
mcp add-research-topic "Payment processing security"
mcp add-research-topic "Scalable database design"

# 3. Create task hierarchy
mcp create-task "Backend Development" --priority 5
mcp create-task "API Design" --parent "Backend Development"
mcp create-task "Database Schema" --parent "Backend Development"

# 4. Export context for development
mcp export-context --types "tasks,research,memories" --format json
```

### Performance Monitoring Workflow
```bash
# 1. Check system health
mcp health-check
mcp performance-report

# 2. Identify issues
mcp diagnose-performance
mcp identify-bottlenecks

# 3. Optimize system
mcp optimize-system --aggressive
mcp clear-cache --all

# 4. Monitor improvements
mcp performance-compare --baseline "pre-optimization"
```

This CLI reference provides comprehensive coverage of all MCP commands with practical examples and usage patterns.