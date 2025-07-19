# ⚡ MCP Quick Start Guide

## Get Up and Running in 5 Minutes

This guide will get you from zero to your first MCP-powered AI development project in just 5 minutes.

---

## 🚀 Step 1: Installation (2 minutes)

### Option A: Portable Package (Recommended)

1. **Download** the latest release for your platform:
   - **Windows**: `mcp-portable-windows-v1.0.0.zip`
   - **macOS**: `mcp-portable-macos-v1.0.0.tar.gz`
   - **Linux**: `mcp-portable-linux-v1.0.0.tar.gz`

2. **Extract** to your desired location:
   ```bash
   # Windows: Right-click and "Extract All"
   
   # macOS/Linux:
   tar -xzf mcp-portable-*.tar.gz
   cd mcp-portable-*
   ```

3. **Install** by running the installer:
   ```bash
   # Windows
   install.bat
   
   # macOS/Linux
   ./install.sh
   ```

### Option B: Docker (Alternative)

```bash
# Download and run
wget https://github.com/your-repo/mcp/releases/latest/download/mcp-docker-v1.0.0.tar.gz
tar -xzf mcp-docker-v1.0.0.tar.gz
cd mcp-docker
docker-compose up -d
```

---

## 🎯 Step 2: First Project (2 minutes)

### Start MCP Server

```bash
# Windows
start_mcp.bat

# macOS/Linux
./start_mcp.sh
```

You should see:
```
🚀 MCP Server starting...
🧠 Brain-inspired architecture initialized
🌐 Server running at http://localhost:3000
✅ Ready for AI development acceleration!
```

### Create Your First Project

Open a new terminal and run:

```bash
# Initialize a web application project
mcp init-project "My First AI App" --type "web-application"

# Start research on your topic
mcp start-research "Modern web development with AI"

# Add specific research areas
mcp add-research-topic "React with AI integration"
mcp add-research-topic "API design best practices"
```

### Create Task Structure

```bash
# Create main development tasks
mcp create-task "Frontend Development" --priority 5
mcp create-task "Backend API" --priority 5
mcp create-task "AI Integration" --priority 4

# Add subtasks
mcp create-task "User Interface" --parent "Frontend Development"
mcp create-task "Authentication System" --parent "Backend API"
```

---

## 🤖 Step 3: AI Integration (1 minute)

### Export Context for Your LLM

```bash
# Generate optimized context for your AI assistant
mcp export-context --format json --max-tokens 2000
```

This creates a JSON file with:
- Your project structure and goals
- Research findings and insights
- Current task priorities and dependencies
- Relevant memories and best practices

### Use with Your Favorite AI Tool

**VS Code + Cursor:**
1. Copy the exported context
2. Paste into your AI chat
3. Ask: "Help me implement the user interface task based on this context"

**Claude/ChatGPT:**
1. Upload the context file
2. Ask: "Based on this MCP context, help me build the authentication system"

**Command Line:**
```bash
# Direct integration with various AI tools
mcp export-context | your-ai-tool --context-file -
```

---

## 🎉 You're Ready!

### What You Just Accomplished

✅ **Installed MCP** with brain-inspired AI development acceleration  
✅ **Created your first project** with intelligent structure  
✅ **Set up research automation** for continuous learning  
✅ **Organized tasks hierarchically** with smart prioritization  
✅ **Generated AI-optimized context** for seamless LLM integration  

### Your MCP System Now Provides

🧠 **Intelligent Memory**: Three-tier memory system learning from your workflow  
🔄 **Adaptive Optimization**: Genetic algorithms improving your development patterns  
🌐 **P2P Collaboration**: Sharing optimizations with the global MCP network  
📊 **Performance Monitoring**: Real-time insights into your development efficiency  

---

## 🚀 Next Steps (Choose Your Path)

### For Immediate Productivity

```bash
# Add your first memory
mcp add-memory "Use TypeScript for better type safety" --type "best-practice"

# Update task progress
mcp update-task-progress "User Interface" 25 --note "Created basic component structure"

# Get project status
mcp project-status
mcp task-tree
```

### For Advanced Features

```bash
# Join P2P network for collaboration
mcp p2p-connect --network "development"

# Configure IDE integration
mcp configure-ide vscode  # or cursor, claude

# Set up automated workflows
mcp create-workflow "daily-standup" --trigger "time:09:00"
```

### For Customization

```bash
# Set your preferences
mcp set-preference "language" "python"
mcp set-preference "framework" "fastapi"
mcp set-preference "workflow_style" "agile"

# Configure memory system
mcp configure-memory --working-memory-size 100
```

---

## 💡 Pro Tips

### Maximize AI Integration

1. **Export context frequently** during development
2. **Use specific, searchable memory descriptions**
3. **Update task progress with meaningful notes**
4. **Tag everything** for easy filtering and search

### Optimize Performance

1. **Let MCP learn** from your patterns (genetic optimization)
2. **Join the P2P network** to benefit from community optimizations
3. **Use hierarchical tasks** for better organization
4. **Regular memory consolidation** improves search performance

### Best Practices

1. **Start with research** before coding
2. **Break large tasks** into smaller, manageable pieces
3. **Use dependencies** to maintain proper task ordering
4. **Regular context exports** keep your AI assistant informed

---

## 🔧 Common Commands Reference

### Project Management
```bash
mcp init-project "Name" --type "web-app|api|mobile|data-science"
mcp project-status
mcp switch-project "Project Name"
```

### Research & Memory
```bash
mcp start-research "Topic"
mcp add-finding "Key insight" --source "URL" --confidence 0.9
mcp add-memory "Important info" --type "reference|best-practice|issue"
mcp search-memories "query"
```

### Task Management
```bash
mcp create-task "Task Name" --priority 1-5
mcp update-task-progress "Task" 50 --note "Progress update"
mcp task-tree
mcp complete-task "Task Name"
```

### Context & Integration
```bash
mcp export-context --format json --max-tokens 2000
mcp get-context-pack "development|debugging|planning"
mcp configure-ide vscode|cursor|claude
```

### System Management
```bash
mcp health-check
mcp system-status
mcp performance-report
mcp optimize-system
```

---

## 🆘 Need Help?

### Quick Diagnostics
```bash
mcp health-check          # Verify system health
mcp --version            # Check version
mcp help                 # List all commands
mcp help <command>       # Get command help
```

### Documentation
- **[[USER_GUIDE]]** - Complete user manual
- **[[INSTALLATION_GUIDE]]** - Detailed setup instructions
- **[[TROUBLESHOOTING]]** - Common issues and solutions
- **[[API_DOCUMENTATION]]** - Technical reference

### Support Resources
- **GitHub Issues** - Bug reports and feature requests
- **Community Forum** - Discussions and support
- **Documentation Vault** - Comprehensive guides and references

---

## 🎊 Welcome to AI-Accelerated Development!

You now have a powerful AI development acceleration platform that:

- **Learns from your workflow** and continuously optimizes
- **Integrates seamlessly** with your favorite AI tools
- **Manages complexity** through intelligent organization
- **Accelerates development** through context optimization
- **Connects you** to a global network of developers

**Start building amazing AI-powered applications with MCP!** 🚀

---

*Total setup time: ~5 minutes | Next: Explore the [[USER_GUIDE]] for advanced features*