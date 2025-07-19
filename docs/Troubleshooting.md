# 🔧 MCP Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Python Version Compatibility
**Problem**: "Python version not supported" error
**Solution**:
```bash
# Check Python version
python --version
python3 --version

# Install Python 3.8+ if needed
# Windows: Download from python.org
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11
```

#### Permission Errors
**Problem**: Permission denied during installation
**Solution**:
```bash
# Linux/macOS
sudo chown -R $USER:$USER ~/.mcp
chmod +x ./install.sh

# Windows (run as administrator)
icacls . /grant %USERNAME%:F /T
```

#### Virtual Environment Issues
**Problem**: Virtual environment creation fails
**Solution**:
```bash
# Ensure venv module is available
python -m pip install --upgrade pip
python -m pip install virtualenv

# Create environment manually
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Runtime Issues

#### Server Won't Start
**Problem**: MCP server fails to start
**Diagnosis**:
```bash
mcp health-check
mcp logs --tail 50
```

**Solutions**:
```bash
# Check port availability
netstat -an | grep 3000

# Start with different port
mcp server --port 3001

# Reset configuration
mcp reset-config --backup
```

#### Memory Issues
**Problem**: High memory usage or out of memory errors
**Solutions**:
```bash
# Check memory usage
mcp resource-status

# Optimize memory
mcp optimize-system --target memory

# Clear caches
mcp clear-cache --type all

# Consolidate memories
mcp consolidate-memories --aggressive
```

#### Performance Issues
**Problem**: Slow response times
**Diagnosis**:
```bash
mcp diagnose-performance
mcp performance-report --detailed
```

**Solutions**:
```bash
# System optimization
mcp optimize-system

# Clear caches
mcp clear-cache --type all

# Check resource limits
mcp configure-resources --max-memory 8GB
```

### IDE Integration Issues

#### VS Code Integration
**Problem**: MCP not working with VS Code
**Solutions**:
```bash
# Verify MCP server is running
mcp status

# Check VS Code configuration
mcp configure-ide vscode --verify

# Restart both MCP and VS Code
mcp restart
```

#### Context Export Issues
**Problem**: Context export fails or produces empty results
**Solutions**:
```bash
# Check project status
mcp project-status

# Verify memory content
mcp memory-stats

# Export with different parameters
mcp export-context --format json --max-tokens 1000
```

### P2P Network Issues

#### Connection Problems
**Problem**: Cannot connect to P2P network
**Solutions**:
```bash
# Check network status
mcp p2p-status

# Try different network
mcp p2p-connect --network "backup"

# Check firewall settings
# Ensure port 10000 is open
```

#### Sync Issues
**Problem**: P2P synchronization fails
**Solutions**:
```bash
# Force sync
mcp p2p-sync-optimizations --force

# Check peer connections
mcp p2p-status --peers

# Reconnect to network
mcp p2p-disconnect
mcp p2p-connect --network "development"
```

### Data Issues

#### Corrupted Database
**Problem**: Database corruption errors
**Solutions**:
```bash
# Check database integrity
mcp diagnose-database

# Repair database
mcp repair-database --backup

# Restore from backup
mcp restore-backup --latest
```

#### Memory System Issues
**Problem**: Memory search not working
**Solutions**:
```bash
# Rebuild search index
mcp rebuild-search-index

# Check memory integrity
mcp memory-diagnostics

# Consolidate memories
mcp consolidate-memories --force
```

## Diagnostic Commands

### System Health
```bash
# Comprehensive health check
mcp health-check --detailed

# Component-specific checks
mcp health-check --component memory
mcp health-check --component genetic-system
mcp health-check --component p2p-network
```

### Log Analysis
```bash
# View recent logs
mcp logs --tail 100

# Filter by level
mcp logs --level error --since 1h

# Component-specific logs
mcp logs --component memory --detailed
```

### Performance Diagnostics
```bash
# Performance analysis
mcp diagnose-performance --detailed

# Resource usage
mcp resource-status --live

# Bottleneck identification
mcp identify-bottlenecks
```

## Recovery Procedures

### Configuration Reset
```bash
# Backup current config
mcp backup-config

# Reset to defaults
mcp reset-config --backup

# Restore specific component
mcp restore-config --component memory
```

### Data Recovery
```bash
# Create backup
mcp backup-data --include-all

# Restore from backup
mcp restore-data --backup-file backup-2024-01-15.tar.gz

# Partial recovery
mcp restore-data --component memory --backup-file backup.tar.gz
```

### Complete System Reset
```bash
# Full system backup
mcp backup-system --complete

# Reset everything
mcp reset-system --confirm

# Restore from backup
mcp restore-system --backup-file complete-backup.tar.gz
```

## Getting Help

### Built-in Help
```bash
# Command help
mcp help
mcp help <command>

# Documentation
mcp docs --topic troubleshooting
mcp docs --search "error message"
```

### Diagnostic Information
```bash
# System information
mcp system-info
mcp environment-info

# Export diagnostic data
mcp export-diagnostics --include-logs --include-config
```

### Community Support
- GitHub Issues: Report bugs and get help
- Documentation: Comprehensive guides and references
- Community Forum: Discussions and solutions

## Prevention Tips

### Regular Maintenance
```bash
# Weekly health check
mcp health-check --detailed

# Monthly optimization
mcp optimize-system --aggressive

# Regular backups
mcp backup-system --schedule weekly
```

### Best Practices
- Keep MCP updated to latest version
- Regular memory consolidation
- Monitor resource usage
- Backup before major changes
- Use appropriate token limits for context export

### Monitoring Setup
```bash
# Enable monitoring
mcp configure-monitoring --enable

# Set up alerts
mcp configure-alerts --memory-threshold 80% --disk-threshold 90%

# Regular reports
mcp schedule-report --type performance --frequency daily
```

For additional help, see the complete documentation or contact support.