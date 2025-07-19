# 🛠️ Development Environment Setup

## Prerequisites

### Required Software
- **Python 3.8+**: Primary development language
- **Git**: Version control system
- **SQLite**: Database engine (usually included with Python)
- **Node.js 16+**: For frontend development (optional)

### Recommended Tools
- **VS Code**: Primary IDE with Python extension
- **Docker**: For containerized development
- **Postman**: API testing
- **SQLite Browser**: Database inspection

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/mcp-agentic-workflow.git
cd mcp-agentic-workflow
```

### 2. Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### 3. Development Dependencies
```bash
# Code quality tools
pip install black flake8 mypy isort

# Testing tools
pip install pytest pytest-cov pytest-asyncio

# Documentation tools
pip install mkdocs mkdocs-material

# Optional: Frontend development
npm install  # If working on frontend
```

### 4. Database Setup
```bash
# Initialize databases
python scripts/setup_databases.py

# Run migrations (if any)
python scripts/migrate_databases.py
```

### 5. Configuration
```bash
# Copy example configuration
cp config/config.example.cfg config/config.cfg

# Edit configuration as needed
# Set development-specific settings
```

## IDE Configuration

### VS Code Setup
1. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - GitLens
   - SQLite Viewer

2. Configure workspace settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

### PyCharm Setup
1. Open project in PyCharm
2. Configure Python interpreter to use virtual environment
3. Enable code inspections for Python
4. Configure code style to use Black formatter

## Verification

### 1. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/unit/    # Unit tests only
```

### 2. Code Quality Checks
```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Import sorting
isort src/ tests/
```

### 3. Start Development Server
```bash
# Start MCP server in development mode
python -m src.mcp.server --debug

# Or use the CLI
python mcp_cli.py --help
```

### 4. Verify Installation
```bash
# Check system health
python mcp_cli.py health-check

# Run simple functionality test
python simple_test.py
```

## Development Scripts

### Useful Scripts
- `scripts/setup_wizard.py` - Interactive setup
- `scripts/verify_environment.py` - Environment verification
- `scripts/build_portable.py` - Build portable package
- `scripts/run_tests.py` - Comprehensive test runner

### Daily Development Commands
```bash
# Start development session
source venv/bin/activate
python -m src.mcp.server --debug

# Run tests before committing
pytest --cov=src
black src/ tests/
flake8 src/ tests/

# Build and test package
python scripts/build_portable.py
```

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version

# Use specific Python version
python3.11 -m venv venv
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Database Issues
```bash
# Reset databases
rm data/*.db
python scripts/setup_databases.py
```

#### Permission Issues
```bash
# Fix permissions (Linux/macOS)
chmod +x scripts/*.py
chmod +x *.sh
```

## Related Documentation

- **[[Development-Workflow]]** - Development process and contribution workflow
- **[[Code-Standards]]** - Coding standards and quality requirements
- **[[Testing-Guide]]** - Testing strategies and frameworks
- **[[../INSTALLATION_GUIDE]]** - User installation guide

---

*For additional help, see [[../Troubleshooting]] or [[Debugging-Guide]]*