# MCP Server Developer Guide

## Introduction

Welcome to the MCP Server development team! This guide will help you understand the codebase, set up your development environment, and contribute effectively to the project.

## Development Environment Setup

### Prerequisites

- **Python 3.8+**: The project requires Python 3.8 or higher
- **IDE**: VS Code, PyCharm, or your preferred IDE
- **Docker**: Optional, for containerized development

### Initial Setup

1. **Download or copy the project files** to your working directory.

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks** (optional, for code quality):
   ```bash
   pre-commit install
   ```

5. **Run initial tests**:
   ```bash
   python -m pytest tests/ -v
   ```

### Development Tools

#### Code Quality Tools
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Code quality hooks

#### Testing Tools
- **Pytest**: Test framework
- **Coverage**: Test coverage reporting
- **Pytest-cov**: Coverage integration

#### Documentation Tools
- **Sphinx**: Documentation generation
- **MkDocs**: Alternative documentation

## Project Structure

```
agentic-workflow/
├── src/mcp/                    # Main source code
│   ├── __init__.py
│   ├── server.py               # Main server implementation
│   ├── cli.py                  # Command-line interface
│   ├── memory.py               # Memory management
│   ├── task_manager.py         # Task management
│   ├── workflow.py             # Workflow management
│   ├── context_manager.py      # Context management
│   ├── experimental_lobes.py   # Experimental cognitive engines
│   ├── lobes/                  # Individual lobe implementations
│   │   ├── alignment_engine.py
│   │   └── pattern_recognition_engine.py
│   ├── performance_monitor.py  # Performance monitoring
│   ├── plugin_system.py        # Plugin system
│   └── api_enhancements.py     # API enhancements
├── tests/                      # Test suite
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── config/                     # Configuration files
├── plugins/                    # Plugin examples
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # Project overview
```

## Core Architecture

### 1. Server Architecture

The MCP server follows a modular architecture with clear separation of concerns:

```python
# Main server class
class MCPServer:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.task_manager = TaskManager()
        self.workflow_manager = WorkflowManager()
        self.context_manager = ContextManager()
        # ... other components
```

### 2. Component Interaction

Components interact through well-defined interfaces:

```python
# Example: Task creation with memory integration
def create_task_with_context(self, title, description, context_query):
    # Get relevant context
    context = self.context_manager.get_relevant_context(context_query)
    
    # Create task
    task_id = self.task_manager.create_task(title, description)
    
    # Store context as memory
    self.memory_manager.add_memory(
        f"Task context for {title}: {context}",
        memory_type="task_context"
    )
    
    return task_id
```

### 3. Experimental Lobes

The experimental lobes implement advanced cognitive functions:

```python
# Example: Using the alignment engine
class AlignmentEngine:
    def analyze_alignment(self, task_id, user_preferences, context):
        # Analyze task alignment with user preferences
        alignment_score = self._calculate_alignment(task_id, user_preferences)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(alignment_score, context)
        
        return {
            "alignment_score": alignment_score,
            "suggestions": suggestions,
            "confidence": self._calculate_confidence()
        }
```

## Development Workflow

### 1. Feature Development

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature**:
   - Follow coding standards
   - Add comprehensive tests
   - Update documentation

3. **Run tests**:
   ```bash
   python -m pytest tests/ -v --cov=src
   ```

4. **Code quality checks**:
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```

5. **Create pull request**:
   - Include detailed description
   - Reference related issues
   - Request code review

### 2. Testing Strategy

#### Unit Tests
```python
# Example unit test
def test_task_creation():
    task_manager = TaskManager()
    task_id = task_manager.create_task("Test Task", "Description")
    
    assert task_id is not None
    assert isinstance(task_id, int)
    
    task = task_manager.get_task(task_id)
    assert task["title"] == "Test Task"
```

#### Integration Tests
```python
# Example integration test
def test_workflow_with_memory():
    server = MCPServer()
    
    # Create workflow
    workflow_id = server.workflow_manager.create_workflow("Test Project", "/tmp/test")
    
    # Add memory
    memory_id = server.memory_manager.add_memory("Important requirement", "requirement")
    
    # Verify integration
    context = server.context_manager.export_context()
    assert "workflows" in context
    assert "memories" in context
```

#### Performance Tests
```python
# Example performance test
def test_memory_search_performance():
    memory_manager = MemoryManager()
    
    # Add test data
    for i in range(1000):
        memory_manager.add_memory(f"Test memory {i}", "test")
    
    # Measure search performance
    start_time = time.time()
    results = memory_manager.search_memories("test", limit=100)
    end_time = time.time()
    
    assert end_time - start_time < 1.0  # Should complete within 1 second
```

### 3. Code Quality Standards

#### Code Style
- Follow PEP 8 guidelines
- Use Black for formatting
- Maximum line length: 88 characters
- Use type hints for all functions

#### Documentation
- Docstrings for all public functions
- Inline comments for complex logic
- Update README.md for user-facing changes
- Update API documentation for interface changes

#### Error Handling
```python
# Example error handling
def safe_operation(self, operation_func, *args, **kwargs):
    try:
        return operation_func(*args, **kwargs)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise MCPError("Database operation failed") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise MCPError("Operation failed") from e
```

## Plugin Development

### 1. Plugin Structure

```python
# Example plugin structure
class ExamplePlugin:
    def __init__(self, metadata, config=None):
        self.metadata = metadata
        self.config = config or {}
    
    def process(self, data):
        # Plugin logic here
        return {"result": "processed_data"}
    
    def cleanup(self):
        # Cleanup resources
        pass
```

### 2. Plugin Metadata

```python
# Plugin metadata
metadata = PluginMetadata(
    name="example_plugin",
    version="1.0.0",
    description="Example plugin for MCP server",
    author="Developer Name",
    dependencies=["requests", "numpy"]
)
```

### 3. Plugin Testing

```python
# Plugin test
def test_example_plugin():
    plugin = ExamplePlugin(metadata)
    
    result = plugin.process({"input": "test"})
    assert "result" in result
    assert result["result"] == "processed_data"
```

## Database Schema

### 1. Core Tables

```sql
-- Tasks table
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    parent_id INTEGER,
    is_meta BOOLEAN DEFAULT FALSE,
    meta_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES tasks (id)
);

-- Memories table
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    memory_type TEXT,
    priority INTEGER DEFAULT 1,
    vector_data BLOB,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflows table
CREATE TABLE workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    project_path TEXT,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Migration Strategy

```python
# Example migration
def migrate_database(self, from_version, to_version):
    if from_version < "1.1.0":
        self._add_task_metadata_column()
    
    if from_version < "1.2.0":
        self._add_memory_vector_index()
```

## Performance Optimization

### 1. Database Optimization

```python
# Connection pooling
class DatabaseManager:
    def __init__(self, db_path, max_connections=10):
        self.pool = QueuePool(
            creator=lambda: sqlite3.connect(db_path),
            max_overflow=0,
            pool_size=max_connections
        )
```

### 2. Memory Management

```python
# Efficient memory usage
class MemoryManager:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
        self.vector_cache = LRUCache(maxsize=500)
```

### 3. Async Processing

```python
# Async task processing
async def process_tasks_async(self, tasks):
    semaphore = asyncio.Semaphore(10)  # Limit concurrent tasks
    
    async def process_task(task):
        async with semaphore:
            return await self._process_single_task(task)
    
    return await asyncio.gather(*[process_task(task) for task in tasks])
```

## Debugging and Troubleshooting

### 1. Logging

```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Debug Mode

```python
# Enable debug mode
server = MCPServer(debug=True)
server.start()
```

### 3. Performance Profiling

```python
# Profile performance
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

## Contributing Guidelines

### 1. Code Review Process

1. **Self-review**: Review your own code before submitting
2. **Peer review**: Request review from team members
3. **Automated checks**: Ensure all CI checks pass
4. **Documentation**: Update relevant documentation

### 2. Commit Messages

Follow conventional commit format:
```
feat: add new task creation endpoint
fix: resolve memory leak in vector search
docs: update API documentation
test: add integration tests for workflow manager
```

### 3. Issue Reporting

When reporting issues, include:
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment information
- Relevant logs

### 4. Feature Requests

When requesting features, include:
- Clear description of the feature
- Use cases and benefits
- Implementation suggestions
- Priority level

## Deployment

### 1. Development Deployment

```bash
# Run in development mode
python -m src.mcp.cli server --debug --port 3000
```

### 2. Production Deployment

```bash
# Build portable environment
python scripts/build_portable.py --output portable_mcp

# Create archive
python scripts/build_portable.py --archive --archive-name mcp_server.tar.gz
```

### 3. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 3000
CMD ["python", "-m", "src.mcp.cli", "server"]
```

## Resources

### 1. Documentation
- [API Documentation](API_DOCUMENTATION.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [User Guide](USER_GUIDE.md)

### 2. External Resources
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Python Documentation](https://docs.python.org/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)

### 3. Development Tools
- [VS Code](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Docker](https://www.docker.com/)

## Getting Help

### 1. Internal Resources
- Code comments and docstrings
- Test examples
- Architecture documentation

### 2. Team Communication
- Code review discussions
- Team meetings
- Documentation updates

### 3. External Support
- Stack Overflow
- Python community forums
- MCP community discussions

This developer guide provides comprehensive information for contributing to the MCP server project. Follow these guidelines to ensure high-quality contributions and maintain code consistency. 