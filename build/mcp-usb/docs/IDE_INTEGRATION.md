# IDE Integration Guide

This guide provides comprehensive instructions for integrating the MCP server with various IDEs and development environments.

## Overview

The MCP server provides a stdio JSON-RPC interface that can be integrated with any IDE or development environment that supports JSON-RPC communication. The server implements the full MCP protocol and provides advanced features for agentic development workflows.

## Supported IDEs and Environments

### 1. Cursor IDE

Cursor has built-in support for MCP servers. To integrate:

1. **Install the MCP server**:
   ```bash
   # Clone and setup the project
   git clone <repository>
   cd agentic-workflow
   python -m pip install -e .
   ```

2. **Configure Cursor**:
   - Open Cursor settings
   - Navigate to "AI" → "Model Context Protocol"
   - Add a new server configuration:
   ```json
   {
     "name": "Agentic Workflow MCP",
     "command": "python",
     "args": ["-m", "src.mcp.mcp_stdio"],
     "env": {
       "PYTHONPATH": "/path/to/agentic-workflow/src"
     }
   }
   ```

3. **Environment Variables**:
   ```bash
   export MCP_API_KEY="your_api_key_here"
   export MCP_PROJECT_PATH="/path/to/your/project"
   export MCP_VECTOR_BACKEND="sqlitefaiss"
   ```

### 2. VS Code

VS Code can integrate with MCP servers through extensions:

1. **Install MCP Extension**:
   - Search for "Model Context Protocol" in VS Code extensions
   - Install the official MCP extension

2. **Configure Extension**:
   ```json
   // .vscode/settings.json
   {
     "mcp.servers": {
       "agentic-workflow": {
         "command": "python",
         "args": ["-m", "src.mcp.mcp_stdio"],
         "env": {
           "PYTHONPATH": "${workspaceFolder}/src"
         }
       }
     }
   }
   ```

3. **Usage**:
   - Use Command Palette: `MCP: Connect to Server`
   - Select "agentic-workflow" server
   - Access MCP features through the MCP panel

### 3. Claude Desktop

Claude Desktop supports MCP servers natively:

1. **Server Configuration**:
   ```json
   // ~/.config/claude/mcp-servers.json
   {
     "agentic-workflow": {
       "command": "python",
       "args": ["-m", "src.mcp.mcp_stdio"],
       "env": {
         "PYTHONPATH": "/path/to/agentic-workflow/src"
       }
     }
   }
   ```

2. **Usage**:
   - Claude will automatically detect and connect to the server
   - Use MCP commands in chat: `/mcp agentic-workflow create_project`

### 4. LMStudio

LMStudio supports MCP through its API:

1. **Configure LMStudio**:
   ```json
   // lmstudio_config.json
   {
     "mcp_servers": {
       "agentic-workflow": {
         "command": "python",
         "args": ["-m", "src.mcp.mcp_stdio"],
         "env": {
           "PYTHONPATH": "/path/to/agentic-workflow/src"
         }
       }
     }
   }
   ```

2. **API Integration**:
   ```python
   import requests
   
   # LMStudio MCP endpoint
   response = requests.post("http://localhost:1234/v1/chat/completions", {
     "model": "local-model",
     "messages": [
       {"role": "user", "content": "Create a new project using MCP"}
     ],
     "tools": [{"type": "mcp", "server": "agentic-workflow"}]
   })
   ```

### 5. Local AI Services (Ollama, etc.)

For local AI services that support MCP:

1. **Ollama Configuration**:
   ```yaml
   # ollama_config.yaml
   mcp_servers:
     agentic-workflow:
       command: python
       args: ["-m", "src.mcp.mcp_stdio"]
       env:
         PYTHONPATH: "/path/to/agentic-workflow/src"
   ```

2. **Usage with Ollama**:
   ```bash
   ollama run llama2 --mcp-server agentic-workflow
   ```

## MCP Protocol Methods

The server implements the following MCP methods:

### Project Management
- `create_project` - Create a new project
- `get_project_status` - Get current project status
- `update_configuration` - Update project configuration
- `export_project` - Export project data
- `import_project` - Import project data

### Workflow Management
- `start_workflow_step` - Start a workflow step
- `get_workflow_status` - Get workflow status
- `update_workflow` - Update workflow configuration

### Task Management
- `create_task` - Create a new task
- `update_task` - Update task status
- `get_tasks` - List all tasks
- `delete_task` - Delete a task

### Memory Management
- `add_memory` - Add a memory entry
- `search_memories` - Search memories
- `get_memories` - Get all memories
- `update_memory` - Update memory entry

### Context Management
- `get_context` - Get current context
- `update_context` - Update context
- `optimize_context` - Optimize context for tokens

### RAG System
- `rag_query` - Query the RAG system
- `add_document` - Add document to RAG
- `search_documents` - Search documents

### Performance Monitoring
- `get_performance` - Get performance metrics
- `optimize_system` - Optimize system performance
- `get_prometheus_metrics` - Get Prometheus metrics

### Experimental Lobes
- `run_lobe` - Run an experimental lobe
- `get_lobe_status` - Get lobe status
- `configure_lobe` - Configure lobe parameters

## Example Usage

### Creating a Project
```json
{
  "jsonrpc": "2.0",
  "method": "create_project",
  "params": {
    "name": "my_project",
    "description": "A new project",
    "idea_file": "idea.txt"
  },
  "id": 1
}
```

### Creating a Task
```json
{
  "jsonrpc": "2.0",
  "method": "create_task",
  "params": {
    "title": "Implement feature X",
    "description": "Add new functionality",
    "priority": "high",
    "tags": ["feature", "frontend"]
  },
  "id": 2
}
```

### Querying RAG System
```json
{
  "jsonrpc": "2.0",
  "method": "rag_query",
  "params": {
    "query": "How to implement authentication?",
    "max_results": 5
  },
  "id": 3
}
```

## Environment Configuration

### Required Environment Variables
```bash
# API Authentication
export MCP_API_KEY="your_api_key_here"

# Project Configuration
export MCP_PROJECT_PATH="/path/to/project"
export MCP_VECTOR_BACKEND="sqlitefaiss"

# Performance Monitoring
export MCP_PROMETHEUS_ENABLED="true"
export MCP_NETDATA_ENABLED="true"

# Database Configuration
export MCP_DB_PATH="/path/to/database"
export MCP_DB_POOL_SIZE="10"

# Logging
export MCP_LOG_LEVEL="INFO"
export MCP_LOG_FILE="/path/to/logs/mcp.log"
```

### Optional Configuration
```bash
# Vector Backend Options
export MCP_VECTOR_BACKEND="milvus"  # or "annoy", "qdrant"
export MCP_MILVUS_HOST="localhost"
export MCP_MILVUS_PORT="19530"

# Rate Limiting
export MCP_RATE_LIMIT_ENABLED="true"
export MCP_REQUESTS_PER_MINUTE="60"

# Security
export MCP_AUTH_ENABLED="true"
export MCP_ALLOWED_ORIGINS="*"
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure PYTHONPATH is set correctly
   export PYTHONPATH="/path/to/agentic-workflow/src:$PYTHONPATH"
   ```

2. **Permission Errors**:
   ```bash
   # Make the stdio server executable
   chmod +x src/mcp/mcp_stdio.py
   ```

3. **Database Errors**:
   ```bash
   # Initialize database
   python -m src.mcp.cli init-db
   ```

4. **Vector Backend Issues**:
   ```bash
   # Check vector backend status
   python -m src.mcp.cli vector-status
   ```

### Debug Mode
```bash
# Enable debug logging
export MCP_LOG_LEVEL="DEBUG"
export MCP_DEBUG_MODE="true"

# Run with verbose output
python -m src.mcp.mcp_stdio --verbose
```

### Testing Integration
```bash
# Test stdio server
echo '{"jsonrpc": "2.0", "method": "list_endpoints", "params": {}, "id": 1}' | python -m src.mcp.mcp_stdio

# Test with curl (if HTTP server is enabled)
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "list_endpoints", "params": {}, "id": 1}'
```

## Advanced Configuration

### Custom Server Configuration
```python
# custom_mcp_server.py
from src.mcp.mcp_stdio import MCPStdioServer

class CustomMCPServer(MCPStdioServer):
    def __init__(self):
        super().__init__()
        # Add custom initialization
        
    async def handle_request(self, request_data: str) -> str:
        # Add custom request handling
        return await super().handle_request(request_data)

if __name__ == "__main__":
    import asyncio
    server = CustomMCPServer()
    asyncio.run(server.run())
```

### Plugin Development
```python
# my_plugin.py
from src.mcp.server import MCPServer

class MyPlugin:
    def __init__(self, server: MCPServer):
        self.server = server
        
    async def my_custom_method(self, params: dict) -> dict:
        # Implement custom functionality
        return {"result": "success"}

# Register plugin
server = MCPServer()
plugin = MyPlugin(server)
```

## Security Considerations

1. **API Key Management**:
   - Use secure API keys
   - Rotate keys regularly
   - Store keys securely (not in code)

2. **Rate Limiting**:
   - Enable rate limiting in production
   - Monitor usage patterns
   - Set appropriate limits

3. **Input Validation**:
   - Validate all inputs
   - Sanitize user data
   - Use parameterized queries

4. **Network Security**:
   - Use HTTPS in production
   - Implement proper CORS
   - Restrict allowed origins

## Performance Optimization

1. **Database Optimization**:
   - Use connection pooling
   - Implement query caching
   - Optimize indexes

2. **Vector Backend**:
   - Choose appropriate backend
   - Optimize batch operations
   - Monitor memory usage

3. **Caching**:
   - Cache frequent queries
   - Use Redis for distributed caching
   - Implement cache invalidation

## Monitoring and Observability

1. **Prometheus Metrics**:
   - Enable Prometheus monitoring
   - Set up dashboards
   - Configure alerts

2. **Logging**:
   - Use structured logging
   - Implement log rotation
   - Monitor error rates

3. **Health Checks**:
   - Implement health endpoints
   - Monitor system resources
   - Set up automated testing

## Support and Community

- **Documentation**: See `docs/` directory
- **Issues**: Report on GitHub
- **Discussions**: Use GitHub Discussions
- **Contributing**: See CONTRIBUTING.md

## License

This project is licensed under the MIT License. See LICENSE file for details. 