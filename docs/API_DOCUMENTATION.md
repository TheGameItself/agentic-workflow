# MCP Server API Documentation

## Overview

The MCP Server provides a unified API for managing workflows, tasks, memory, and context. This document describes the complete API interface for all core components.

## Core Components

### 1. Workflow Manager API

The WorkflowManager handles project workflows with steps, dependencies, and progress tracking.

#### Workflow Creation and Management

```python
# Create a new workflow
workflow_id = workflow_manager.create_workflow(project_name, project_path)

# Register custom steps
workflow_manager.register_step(step_name, WorkflowStep)

# Modify existing steps
workflow_manager.modify_step(step_name, modifications_dict)

# Remove steps
workflow_manager.remove_step(step_name)
```

#### Step Lifecycle Management

```python
# Start a step
workflow_manager.start_step(step_name)

# Complete a step
workflow_manager.complete_step(step_name)

# Check step status
status = workflow_manager.steps[step_name].status

# Get step progress
progress = workflow_manager.steps[step_name].get_partial_progress()
```

#### Dependencies and Flow Control

```python
# Set step dependencies
step.dependencies = ["step1", "step2"]

# Set next steps
workflow_manager.set_next_steps(step_name, ["next_step1", "next_step2"])

# Get next steps
next_steps = workflow_manager.get_next_steps(step_name)

# Remove next step
workflow_manager.remove_next_step(step_name, "step_to_remove")
```

#### Feedback and Learning

```python
# Add step feedback
workflow_manager.add_step_feedback(
    step_name, 
    feedback_text, 
    impact=1, 
    principle="principle_name"
)

# Get step feedback
feedbacks = workflow_manager.steps[step_name].feedback
```

### 2. Task Manager API

The TaskManager handles individual tasks with hierarchical relationships and progress tracking.

#### Task Creation and Management

```python
# Create a task
task_id = task_manager.create_task(
    title="Task Title",
    description="Task description",
    priority=5,
    parent_id=None,
    is_meta=False,
    meta_type=None,
    tags=[]
)

# Get task information
task = task_manager.get_task(task_id)

# Update task
task_manager.update_task(task_id, updates_dict)

# Delete task
task_manager.delete_task(task_id)
```

#### Task Dependencies

```python
# Add dependency
dep_id = task_manager.add_task_dependency(task_id, depends_on_task_id)

# Get dependencies
dependencies = task_manager.get_task_dependencies(task_id)

# Remove dependency
task_manager.remove_task_dependency(dep_id)
```

#### Progress Tracking

```python
# Update progress
task_manager.update_task_progress(
    task_id, 
    progress_percentage, 
    current_step="description",
    partial_completion_notes="notes"
)

# Get progress
progress = task_manager.get_task_progress(task_id)
```

#### Notes and Feedback

```python
# Add note
note_id = task_manager.add_task_note(
    task_id, 
    note_text, 
    line_number=None, 
    file_path=None
)

# Get notes
notes = task_manager.get_task_notes(task_id)

# Add feedback
feedback_id = task_manager.add_task_feedback(
    task_id,
    feedback_text,
    lesson_learned="lesson",
    principle="principle",
    impact_score=1
)
```

#### Task Hierarchy

```python
# Get task tree
tree = task_manager.get_task_tree(root_task_id=task_id)

# Get subtasks
subtasks = task_manager.get_subtasks(task_id)

# Get parent task
parent = task_manager.get_parent_task(task_id)
```

#### Tags and Organization

```python
# Create tag
tag_id = task_manager.create_tag(tag_name, tag_type="general")

# Update tag
task_manager.update_tag(tag_id, tag_name="new_name")

# Get tag
tag = task_manager.get_tag(tag_id)

# Get task tags
tags = task_manager.get_tags(task_id=task_id)

# Get all tags
all_tags = task_manager.get_all_tags()
```

### 3. Memory Manager API

The MemoryManager handles persistent memory storage with vector search capabilities.

#### Memory Operations

```python
# Add memory
memory_id = memory_manager.add_memory(
    text="memory content",
    memory_type="type",
    priority=1,
    tags=[],
    metadata={}
)

# Get memory
memory = memory_manager.get_memory(memory_id)

# Update memory
memory_manager.update_memory(memory_id, updates_dict)

# Delete memory
memory_manager.delete_memory(memory_id)
```

#### Search and Retrieval

```python
# Search memories
results = memory_manager.search_memories(
    query="search query",
    limit=10,
    memory_type=None,
    tags=[]
)

# Get memories by type
memories = memory_manager.get_memories_by_type(memory_type)

# Get memories by tags
memories = memory_manager.get_memories_by_tags(tags)
```

#### Vector Operations

```python
# Get memory vector
vector = memory_manager.get_memory_vector(memory_id)

# Find similar memories
similar = memory_manager.find_similar_memories(memory_id, limit=5)

# Batch operations
memory_manager.batch_add_memories(memories_list)
```

### 4. Context Manager API

The ContextManager provides intelligent context management and retrieval.

#### Context Operations

```python
# Export context
context = context_manager.export_context()

# Import context
context_manager.import_context(context_data)

# Get relevant context
relevant = context_manager.get_relevant_context(query, limit=10)
```

#### Context Optimization

```python
# Compress context
compressed = context_manager.compress_context(context_data)

# Decompress context
decompressed = context_manager.decompress_context(compressed_data)

# Optimize context size
optimized = context_manager.optimize_context_size(context_data, max_size=1000)
```

### 5. MCP Server API

The MCPServer provides the main interface for all MCP operations.

#### Server Management

```python
# Initialize server
server = MCPServer()

# Get server status
status = server.get_status()

# Start server
server.start()

# Stop server
server.stop()
```

#### Unified Operations

```python
# Create project
project_info = server.create_project(project_name, project_path)

# Get project status
project_status = server.get_project_status()

# Export project data
project_data = server.export_project_data()

# Import project data
server.import_project_data(project_data)
```

## Error Handling

All API methods return structured responses with error information:

```python
# Success response
{
    "success": True,
    "data": result_data,
    "message": "Operation completed successfully"
}

# Error response
{
    "success": False,
    "error": "Error description",
    "error_code": "ERROR_CODE",
    "details": additional_error_info
}
```

## Authentication and Security

The API supports multiple authentication methods:

```python
# API key authentication
server.set_api_key("your_api_key")

# Rate limiting
server.set_rate_limit(requests_per_minute=100)

# Input validation
server.enable_input_validation(True)
```

## Performance Considerations

- Use batch operations for multiple items
- Implement proper error handling
- Monitor memory usage with large datasets
- Use appropriate search limits
- Cache frequently accessed data

## Best Practices

1. **Task Management**: Break large tasks into smaller subtasks
2. **Memory Usage**: Use appropriate memory types and tags
3. **Workflow Design**: Plan dependencies carefully
4. **Error Handling**: Always check return values
5. **Performance**: Use batch operations when possible
6. **Security**: Validate all inputs and use authentication

## Examples

### Complete Project Workflow

```python
# Initialize server
server = MCPServer()

# Create project
project = server.create_project("My Project", "/path/to/project")

# Create workflow
workflow_id = server.workflow_manager.create_workflow("My Project", "/path/to/project")

# Add tasks
task1 = server.task_manager.create_task("Research", "Research requirements", priority=5)
task2 = server.task_manager.create_task("Design", "Design architecture", priority=4, parent_id=task1)
task3 = server.task_manager.create_task("Implement", "Implement solution", priority=3, parent_id=task2)

# Start workflow
server.workflow_manager.start_step("init")

# Track progress
server.task_manager.update_task_progress(task1, 50, "Halfway through research")

# Add feedback
server.task_manager.add_task_feedback(task1, "Good progress", impact_score=2)

# Complete workflow
server.workflow_manager.complete_step("init")
```

### Memory Management

```python
# Add project memories
memory_id = server.memory_manager.add_memory(
    text="Important project requirement: Must be scalable",
    memory_type="requirement",
    priority=5,
    tags=["scalability", "architecture"]
)

# Search for relevant memories
relevant = server.memory_manager.search_memories("scalability", limit=5)

# Get context for decision making
context = server.context_manager.get_relevant_context("scalability decision")
```

# Memory API (Updated)

## Overview
The MCP API now supports a three-tier memory system:
- **WorkingMemory**: For context-sensitive, temporary storage.
- **ShortTermMemory**: For recent, high-priority, or volatile information.
- **LongTermMemory**: For persistent, structured, and research-driven storage.

## Usage in Lobes/Engines
All lobes/engines should use the appropriate memory type for their needs:
- Use `WorkingMemory.add(item)` for context-sensitive, feedback, and temporary storage.
- Use `ShortTermMemory.add(item)` for recent, high-priority, or volatile information.
- Use `LongTermMemory.add(key, value)` for persistent, structured, and research-driven storage.

## Example
```
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory, ShortTermMemory, LongTermMemory

wm = WorkingMemory()
wm.add({"context": "session", "data": "temp"})

stm = ShortTermMemory()
stm.add({"task": "recent", "priority": 1})

ltm = LongTermMemory()
ltm.add("knowledge_1", {"fact": "persistent"})
```

See `ARCHITECTURE.md` and `idea.txt` for more details and research references.

This API documentation provides a comprehensive guide to using all MCP Server features effectively. 