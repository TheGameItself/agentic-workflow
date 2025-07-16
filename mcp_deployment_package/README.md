# MCP Agentic Workflow Accelerator - Deployment Package

## 📦 Package Contents

This deployment package contains everything needed to run the MCP Agentic Workflow Accelerator:

- **mcp_system.zip**: The complete MCP system archive
- **unwrap.sh**: Linux/Mac deployment script
- **unwrap.bat**: Windows deployment script
- **README.md**: This file
- **quick_start.sh**: Quick start script

## 🚀 Quick Deployment

### Linux/Mac
```bash
chmod +x unwrap.sh
./unwrap.sh [target_directory]
```

### Windows
```cmd
unwrap.bat [target_directory]
```

### Manual Deployment
1. Extract `mcp_system.zip` to your desired directory
2. Navigate to the extracted directory
3. Run `./setup.sh` (Linux/Mac) or `setup.bat` (Windows)

## 📋 System Requirements

- **Python**: 3.7 or higher
- **Dependencies**: Automatically installed during setup
- **Storage**: ~100 MB for full deployment
- **Memory**: ~50 MB runtime

## 🎯 What's Included

The MCP Agentic Workflow Accelerator provides:

- **Memory Management**: Add, search, and manage memories with types, priorities, and tags
- **Task Management**: Create hierarchical tasks with dependencies and progress tracking
- **Context Export**: Generate minimal context packs for LLM consumption
- **RAG System**: Intelligent retrieval and search capabilities
- **Workflow Management**: Guided project phases from research to deployment
- **Performance Monitoring**: System health and optimization tools

## 📊 Package Information

- **Package Name**: mcp_deployment_package_20250715_172709
- **Archive**: mcp_system.zip
- **Files in Archive**: 25
- **Created**: 2025-07-15 17:27:09

## 🔧 Available Commands

After deployment, you'll have access to 40+ CLI commands:

```bash
# Show all commands
python mcp_cli.py --help

# Add a memory
python mcp_cli.py add-memory --text "My memory" --type "general" --priority 0.8

# Create a task
python mcp_cli.py create-task --title "My Task" --description "Task description" --priority 5

# List tasks
python mcp_cli.py list-tasks --tree

# Search memories
python mcp_cli.py search-memories --query "memory"

# Export context for LLM
python mcp_cli.py export-context --types tasks,memories --max-tokens 500
```

## 🆘 Troubleshooting

### Common Issues

1. **"No module named 'mcp'"**
   - Ensure you're in the correct directory
   - Run `pip install -r requirements.txt`

2. **"Database is locked"**
   - This is a non-critical warning
   - The system continues to work normally

3. **"Permission denied" on scripts**
   - Run `chmod +x unwrap.sh` to make executable

4. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

### Verification

After deployment, verify the system works:

```bash
# Run comprehensive test
python test_system.py

# Test basic functionality
python mcp_cli.py add-memory --text "Test" --type "test"
python mcp_cli.py search-memories --query "Test"
```

## 📖 More Information

- **Project Vision**: See `idea.txt` in the deployed system
- **Full Documentation**: See `README.md` in the deployed system
- **Status Report**: See `PROJECT_STATUS_FINAL.md` in the deployed system

---

**Ready to accelerate your agentic workflows! 🚀**
