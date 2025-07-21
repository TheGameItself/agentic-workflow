# MCP Self-Debugging System

## Overview

The MCP Self-Debugging System is a comprehensive testing and diagnostic framework that allows the MCP server to test itself, validate its configuration, monitor performance, and update its own documentation. This system enables the MCP server to be self-aware and self-improving.

## Features

### 🔍 Comprehensive Self-Testing
- **Component Health Checks**: Tests all MCP components for functionality and performance
- **Database Integrity Validation**: Verifies database connections and data integrity
- **Configuration Validation**: Ensures all configuration settings are valid
- **Memory System Testing**: Validates memory operations and vector search capabilities
- **Performance Monitoring**: Tracks response times and resource usage
- **API Endpoint Testing**: Verifies CLI commands and API functionality
- **Security Feature Validation**: Checks authentication and rate limiting
- **Integration Point Testing**: Validates external system connections
- **Error Handling Verification**: Tests graceful error recovery
- **Documentation Synchronization**: Identifies and updates outdated documentation

### 📊 Detailed Reporting
- **Test Results**: Comprehensive test results with pass/fail status
- **Performance Metrics**: Response times and resource usage statistics
- **Health Status**: Component health with detailed diagnostics
- **Recommendations**: Actionable recommendations for improvements
- **Documentation Updates**: Tracks documentation changes and updates

### 🔧 CLI Integration
- **Self-Test Command**: Run comprehensive self-tests via CLI
- **Health Check Command**: Check specific component health
- **Documentation Generation**: Generate self-documentation
- **Documentation Validation**: Check for outdated documentation

## Architecture

### Core Components

#### MCPSelfDebugger
The main self-debugging orchestrator that coordinates all testing activities.

```python
class MCPSelfDebugger:
    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.test_results: List[TestResult] = []
        self.health_status: Dict[str, ComponentHealth] = {}
        self.documentation_updates: List[DocumentationUpdate] = []
```

#### TestResult
Represents the result of an individual test.

```python
@dataclass
class TestResult:
    test_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING', 'SKIP'
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
```

#### ComponentHealth
Represents the health status of a component.

```python
@dataclass
class ComponentHealth:
    component_name: str
    status: str  # 'HEALTHY', 'DEGRADED', 'UNHEALTHY'
    response_time: float
    error_count: int
    last_error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
```

#### DocumentationUpdate
Tracks documentation changes and updates.

```python
@dataclass
class DocumentationUpdate:
    file_path: str
    update_type: str  # 'CREATE', 'UPDATE', 'DELETE'
    changes: List[str]
    timestamp: datetime = None
```

## Usage

### CLI Commands

#### Comprehensive Self-Test
```bash
# Run basic self-test
python mcp_cli.py self-test

# Run with report saving
python mcp_cli.py self-test --save-report

# Run with documentation generation
python mcp_cli.py self-test --include-docs

# Run with detailed output
python mcp_cli.py self-test --format detailed
```

#### Health Checks
```bash
# Check all components
python mcp_cli.py health-check

# Check specific component
python mcp_cli.py health-check --component memory_manager

# JSON output
python mcp_cli.py health-check --format json
```

#### Documentation Management
```bash
# Generate self-documentation
python mcp_cli.py generate-docs

# Generate and save documentation
python mcp_cli.py generate-docs --save

# Check for outdated documentation
python mcp_cli.py check-docs

# Check and update outdated docs
python mcp_cli.py check-docs --update
```

#### Fully Automated Self-Upgrade
```bash
python mcp_cli.py self-upgrade
```

### Programmatic Usage

#### Basic Self-Test
```python
from mcp.self_debug import MCPSelfDebugger
from mcp.server import MCPServer

# Initialize
mcp_server = MCPServer()
debugger = MCPSelfDebugger(mcp_server)

# Run comprehensive test
report = debugger.run_comprehensive_self_test()

# Save report
debugger.save_test_report(report, "test_report.json")
```

#### Component-Specific Testing
```python
# Test specific component
debugger._test_core_components()
debugger._test_database_integrity()
debugger._test_memory_systems()
debugger._test_performance_metrics()
```

#### Documentation Generation
```python
# Generate self-documentation
documentation = debugger.generate_self_documentation()

# Save documentation
debugger.save_documentation(documentation, "self_docs.json")
```

## Test Categories

### 1. Core Components Test
Tests all major MCP components:
- MemoryManager
- WorkflowManager
- ProjectManager
- TaskManager
- UnifiedMemory
- RAGSystem
- PerformanceMonitor

**What it tests:**
- Component availability
- Basic functionality
- Response times
- Error handling

### 2. Database Integrity Test
Validates database connections and operations:
- SQLite database connections
- Table structure validation
- Basic query operations
- Data integrity checks

**What it tests:**
- Database file existence
- Connection establishment
- Table accessibility
- Query execution

### 3. Memory Systems Test
Tests memory operations and vector search:
- Memory creation and retrieval
- Vector search functionality
- Memory encoding systems
- Search result validation

**What it tests:**
- Memory CRUD operations
- Vector similarity search
- TFIDF and RaBitQ encoders
- Search result accuracy

### 4. Performance Metrics Test
Monitors system performance:
- Response time tracking
- Resource usage monitoring
- Performance alerts
- Benchmark comparisons

**What it tests:**
- Component response times
- Memory usage
- CPU utilization
- Performance thresholds

### 5. API Endpoints Test
Validates CLI and API functionality:
- Command availability
- Parameter validation
- Response formatting
- Error handling

**What it tests:**
- CLI command registration
- Command parameter parsing
- Response generation
- Error message formatting

### 6. Configuration Test
Validates system configuration:
- Project configuration
- Environment variables
- Feature flags
- Security settings

**What it tests:**
- Configuration file validity
- Required settings presence
- Setting value validation
- Configuration consistency

### 7. Security Features Test
Validates security mechanisms:
- Authentication settings
- Rate limiting configuration
- API key validation
- Access control

**What it tests:**
- Authentication enablement
- Rate limiting setup
- Security policy enforcement
- Access control mechanisms

### 8. Integration Points Test
Tests external system connections:
- Vector backend connections
- Monitoring system integration
- External API connections
- Plugin system

**What it tests:**
- Vector database connectivity
- Prometheus/NetData integration
- External service availability
- Plugin loading mechanisms

### 9. Documentation Sync Test
Manages documentation updates:
- Documentation age analysis
- Update requirement identification
- Documentation generation
- Change tracking

**What it tests:**
- Documentation file timestamps
- Content freshness
- Update requirements
- Documentation consistency

### 10. Error Handling Test
Validates error recovery mechanisms:
- Graceful error handling
- Error message generation
- Recovery procedures
- Logging mechanisms

**What it tests:**
- Exception handling
- Error message clarity
- Recovery procedures
- Logging completeness

## Output Formats

### Text Format (Default)
Human-readable output with emojis and formatting:
```
🧪 Self-Test Summary
==================================================
Total Tests: 25
✅ Passed: 23
❌ Failed: 1
⚠️  Warnings: 1
⏭️  Skipped: 0
Success Rate: 92.0%
Component Health: 95.0%
Average Response Time: 0.045s
Total Duration: 2.34s

💡 Recommendations:
  - Address 1 failed tests
  - Review 1 warning
```

### JSON Format
Structured data for programmatic processing:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total_tests": 25,
    "passed": 23,
    "failed": 1,
    "warnings": 1,
    "skipped": 0,
    "success_rate": 92.0,
    "component_health_rate": 95.0,
    "avg_response_time": 0.045
  },
  "test_results": [...],
  "health_status": {...},
  "recommendations": [...]
}
```

### Detailed Format
Comprehensive output with full test details:
```
🧪 Self-Test Summary
==================================================
Total Tests: 25
✅ Passed: 23
❌ Failed: 1
⚠️  Warnings: 1
⏭️  Skipped: 0
Success Rate: 92.0%
Component Health: 95.0%
Average Response Time: 0.045s
Total Duration: 2.34s

📋 Detailed Results:
==================================================

✅ core_component_MemoryManager
   Status: PASS
   Duration: 0.023s
   Message: Component MemoryManager is healthy
   Details: {'status': 'healthy', 'memory_count': 150}

❌ core_component_WorkflowManager
   Status: FAIL
   Duration: 0.045s
   Message: Component WorkflowManager failed: Database connection error
   Details: {'error': 'Connection refused'}
```

## File Structure

### Test Reports
Test reports are saved to the `logs/` directory:
```
logs/
├── mcp_self_test_report_20240115_103000.json
├── mcp_self_test_report_20240115_143000.json
└── mcp_self_test_report_20240115_183000.json
```

### Documentation
Self-documentation is saved to the `docs/canonical/development/` directory:
```
docs/canonical/development/
├── mcp_self_documentation_20240115_103000.json
├── mcp_self_documentation_20240115_143000.json
└── mcp_self_documentation_20240115_183000.json
```

## Integration with MCP Server

The self-debugging system is fully integrated with the MCP server and can be accessed through:

1. **CLI Commands**: Direct command-line interface
2. **Programmatic API**: Python API for custom testing
3. **Automated Testing**: Scheduled or triggered testing
4. **Monitoring Integration**: Continuous health monitoring

## Best Practices

### Regular Testing
- Run comprehensive tests weekly
- Run health checks daily
- Monitor performance continuously
- Update documentation monthly

### Test Configuration
- Configure appropriate timeouts
- Set realistic performance thresholds
- Enable detailed logging for debugging
- Save reports for historical analysis

### Documentation Management
- Review outdated documentation regularly
- Update documentation with code changes
- Maintain documentation consistency
- Track documentation changes

### Performance Monitoring
- Monitor response times
- Track resource usage
- Set up performance alerts
- Analyze performance trends

## Troubleshooting

### Common Issues

#### Test Failures
- Check component availability
- Verify database connections
- Review configuration settings
- Check error logs

#### Performance Issues
- Monitor resource usage
- Check for memory leaks
- Review database queries
- Analyze response times

#### Documentation Issues
- Verify file permissions
- Check file paths
- Review update procedures
- Validate documentation format

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.getLogger("mcp_self_debug").setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Automated Testing**: Scheduled self-tests
- **Performance Benchmarking**: Historical performance tracking
- **Predictive Analytics**: Failure prediction
- **Integration Testing**: Cross-component testing
- **Load Testing**: Stress testing capabilities
- **Security Testing**: Vulnerability scanning
- **Compliance Testing**: Regulatory compliance validation

### Extensibility
The self-debugging system is designed to be extensible:
- Custom test categories
- Plugin-based testing
- Custom reporting formats
- Integration with external tools

## Conclusion

The MCP Self-Debugging System provides comprehensive testing and diagnostic capabilities that enable the MCP server to be self-aware, self-testing, and self-improving. This system ensures reliability, performance, and maintainability of the MCP server while providing detailed insights into its operation and health.

For more information, see:
- [MCP Server Architecture](architecture.md)
- [Performance Monitoring](performance-optimization/Performance-Monitoring.md)
- [Testing Guidelines](testing/Testing-Guidelines.md)
- [Documentation Standards](development/Documentation-Standards.md)

## Fully Automated Self-Upgrade

MCP now supports a fully non-interactive self-upgrade process. This means:
- All update checks, downloads, and installations are performed without any user prompts.
- If an update is available, it is applied automatically.
- MCP will restart itself after upgrading, with no human intervention required.
- All actions and errors are logged for later review.

### Usage

To trigger a fully automated self-upgrade:

```bash
python mcp_cli.py self-upgrade
```

This command will:
- Check for updates
- Apply updates if available
- Restart MCP automatically
- Never ask for user input at any stage

This can be run manually, on a schedule, or as part of the self-debugging or health-check workflow.

**Note:** If MCP is running as a service, the restart will be handled by the service manager. If running as a script, MCP will re-execute itself after upgrading. 