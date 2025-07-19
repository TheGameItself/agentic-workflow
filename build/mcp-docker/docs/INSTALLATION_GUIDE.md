# 🚀 MCP Installation Guide

## Overview

The MCP Agentic Workflow Accelerator offers multiple installation methods to suit different use cases and environments. Choose the method that best fits your needs.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 2GB free space (10GB recommended)
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.11 or higher
- **Memory**: 16GB RAM
- **Storage**: 20GB free space (SSD recommended)
- **OS**: Latest stable versions
- **Network**: Internet connection for initial setup (optional for offline use)

## 🎯 Quick Start (Recommended)

### Option 1: Portable Release Package (Easiest)

1. **Download the latest release** from [GitHub Releases](https://github.com/your-repo/mcp-agentic-workflow/releases)
2. **Extract the package** to your desired location
3. **Run the installer**:
   ```bash
   # Windows
   ./install.bat
   
   # macOS/Linux
   ./install.sh
   ```
4. **Start MCP**:
   ```bash
   ./start_mcp.sh    # macOS/Linux
   ./start_mcp.bat   # Windows
   ```

### Option 2: Python Package Installation

```bash
# Install from PyPI (when available)
pip install mcp-agentic-workflow

# Or install from source
git clone https://github.com/your-repo/mcp-agentic-workflow.git
cd mcp-agentic-workflow
pip install -e .
```

## 📦 Installation Methods

### 1. Portable Standalone Package

**Best for**: Users who want a complete, self-contained installation

```bash
# Download and extract
wget https://github.com/your-repo/mcp-agentic-workflow/releases/latest/download/mcp-portable-linux.tar.gz
tar -xzf mcp-portable-linux.tar.gz
cd mcp-portable

# Run setup
./setup.sh

# Start MCP
./mcp-server
```

**Features**:
- ✅ Self-contained Python environment
- ✅ All dependencies included
- ✅ No system Python required
- ✅ Portable across machines
- ✅ USB deployment ready

### 2. Docker Installation

**Best for**: Containerized deployments and development

```bash
# Pull the official image
docker pull mcpai/agentic-workflow:latest

# Run with default configuration
docker run -d -p 3000:3000 --name mcp-server mcpai/agentic-workflow:latest

# Run with custom configuration
docker run -d -p 3000:3000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  --name mcp-server mcpai/agentic-workflow:latest
```

**Docker Compose**:
```yaml
version: '3.8'
services:
  mcp-server:
    image: mcpai/agentic-workflow:latest
    ports:
      - "3000:3000"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - MCP_LOG_LEVEL=INFO
      - MCP_DEBUG_MODE=false
    restart: unless-stopped
```

### 3. Development Installation

**Best for**: Developers and contributors

```bash
# Clone repository
git clone https://github.com/your-repo/mcp-agentic-workflow.git
cd mcp-agentic-workflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest

# Start development server
python -m src.mcp.cli server --debug
```

### 4. USB Portable Installation

**Best for**: Portable use across different machines

```bash
# Download USB package
wget https://github.com/your-repo/mcp-agentic-workflow/releases/latest/download/mcp-usb-portable.zip

# Extract to USB drive
unzip mcp-usb-portable.zip -d /path/to/usb/drive/

# Run from USB (no installation required)
cd /path/to/usb/drive/mcp-portable
./start_mcp.sh
```

### 5. System Package Installation

**Best for**: System-wide installation on Linux

```bash
# Ubuntu/Debian
wget https://github.com/your-repo/mcp-agentic-workflow/releases/latest/download/mcp-agentic-workflow.deb
sudo dpkg -i mcp-agentic-workflow.deb
sudo apt-get install -f  # Fix dependencies if needed

# CentOS/RHEL/Fedora
wget https://github.com/your-repo/mcp-agentic-workflow/releases/latest/download/mcp-agentic-workflow.rpm
sudo rpm -i mcp-agentic-workflow.rpm

# Arch Linux
yay -S mcp-agentic-workflow-git
```

## 🔧 Configuration

### Initial Setup

1. **Run the setup wizard**:
   ```bash
   mcp setup
   ```

2. **Configure your IDE integration**:
   ```bash
   # VS Code
   mcp configure vscode
   
   # Cursor
   mcp configure cursor
   
   # Custom IDE
   mcp configure custom
   ```

3. **Verify installation**:
   ```bash
   mcp --version
   mcp health-check
   ```

### Configuration Files

The MCP system uses several configuration files:

- **Main Config**: `config/config.cfg`
- **IDE Integration**: `config/ide-integration.json`
- **Performance**: `config/performance.json`
- **Security**: `config/security.json`

### Environment Variables

```bash
# Core settings
export MCP_PROJECT_PATH="/path/to/your/projects"
export MCP_LOG_LEVEL="INFO"
export MCP_DEBUG_MODE="false"

# Performance settings
export MCP_MEMORY_LIMIT="8GB"
export MCP_WORKER_THREADS="4"

# Integration settings
export MCP_IDE="vscode"  # or cursor, claude, etc.
export MCP_API_PORT="3000"
```

## 🔌 IDE Integration

### VS Code Integration

1. **Install the MCP extension** (when available):
   ```bash
   code --install-extension mcp-ai.mcp-agentic-workflow
   ```

2. **Configure workspace settings**:
   ```json
   {
     "mcp.server.enabled": true,
     "mcp.server.port": 3000,
     "mcp.server.autoStart": true,
     "mcp.features.contextOptimization": true,
     "mcp.features.taskManagement": true
   }
   ```

3. **Add tasks** to `.vscode/tasks.json`:
   ```json
   {
     "version": "2.0.0",
     "tasks": [
       {
         "label": "MCP: Export Context",
         "type": "shell",
         "command": "mcp",
         "args": ["export-context", "--format", "json"],
         "group": "build"
       }
     ]
   }
   ```

### Cursor Integration

1. **Configure MCP server** in Cursor settings:
   ```json
   {
     "mcp": {
       "servers": {
         "agentic-workflow": {
           "command": "mcp",
           "args": ["server"],
           "env": {
             "MCP_LOG_LEVEL": "INFO"
           }
         }
       }
     }
   }
   ```

2. **Enable MCP features**:
   - Context optimization
   - Task management
   - Memory integration
   - Research automation

### Claude Integration

Configure Claude to use MCP server:
```json
{
  "mcpServers": {
    "agentic-workflow": {
      "command": "mcp",
      "args": ["server", "--port", "3000"],
      "env": {
        "MCP_PROJECT_PATH": "/path/to/projects"
      }
    }
  }
}
```

## 🚀 Getting Started

### First Steps

1. **Initialize a project**:
   ```bash
   mcp init-project "My AI Project" --type "web-app"
   ```

2. **Start research**:
   ```bash
   mcp start-research "AI development best practices"
   ```

3. **Create tasks**:
   ```bash
   mcp create-task "Setup development environment" --priority 5
   ```

4. **Export context for your LLM**:
   ```bash
   mcp export-context --format json --max-tokens 2000
   ```

### Example Workflow

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

## 🔍 Troubleshooting

### Common Issues

#### Installation Issues

**Problem**: Python version compatibility
```bash
# Check Python version
python --version

# Install specific Python version if needed
# Use pyenv or conda for version management
```

**Problem**: Permission errors
```bash
# Linux/macOS
sudo chown -R $USER:$USER ~/.mcp
chmod +x ./install.sh

# Windows (run as administrator)
icacls . /grant %USERNAME%:F /T
```

#### Runtime Issues

**Problem**: Server won't start
```bash
# Check logs
mcp logs --tail 50

# Verify configuration
mcp config validate

# Reset to defaults
mcp config reset
```

**Problem**: IDE integration not working
```bash
# Verify MCP server is running
mcp status

# Check IDE configuration
mcp configure --verify

# Restart IDE and MCP server
mcp restart
```

### Getting Help

1. **Check the logs**:
   ```bash
   mcp logs --level debug
   ```

2. **Run diagnostics**:
   ```bash
   mcp diagnose
   ```

3. **Community support**:
   - GitHub Issues: [Report bugs and feature requests](https://github.com/your-repo/mcp-agentic-workflow/issues)
   - Documentation: [[TROUBLESHOOTING]]
   - Developer Guide: [[DEVELOPER_GUIDE]]

## 📚 Next Steps

After installation, explore these resources:

1. **[[USER_GUIDE]]** - Learn how to use MCP effectively
2. **[[API_DOCUMENTATION]]** - Integrate MCP with your tools
3. **[[DEVELOPER_GUIDE]]** - Contribute to MCP development
4. **[[ARCHITECTURE]]** - Understand the system design
5. **[[EXAMPLES]]** - See real-world usage examples

## 🔄 Updates and Maintenance

### Automatic Updates

MCP includes an automatic update system:

```bash
# Check for updates
mcp update check

# Install updates
mcp update install

# Configure automatic updates
mcp update configure --auto-check --interval 24h
```

### Manual Updates

```bash
# Update from package manager
pip install --upgrade mcp-agentic-workflow

# Update from source
git pull origin main
pip install -e .

# Update Docker image
docker pull mcpai/agentic-workflow:latest
```

### Backup and Restore

```bash
# Backup configuration and data
mcp backup create --include-data

# Restore from backup
mcp backup restore backup-2024-01-15.tar.gz

# List available backups
mcp backup list
```

---

## 🎉 Welcome to MCP!

You're now ready to accelerate your AI development workflow with the MCP Agentic Workflow Accelerator. The system provides intelligent project management, memory systems, and workflow orchestration to transform single prompts into complete applications.

**Happy coding!** 🚀