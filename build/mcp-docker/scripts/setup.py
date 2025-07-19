#!/usr/bin/env python3
"""
MCP Agentic Workflow Accelerator Setup Script
Guided setup and initialization for the MCP system.
"""

import os
import sys
import subprocess
from pathlib import Path
import platform
import sqlite3
from scripts.security_audit import SecurityAuditor

def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("🚀 MCP Agentic Workflow Accelerator Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("📦 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def initialize_databases():
    """Initialize the database systems."""
    print("\n🗄️  Initializing databases...")
    try:
        # Import and initialize each system
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        from mcp.memory import MemoryManager
        from mcp.unified_memory import UnifiedMemoryManager
        from mcp.task_manager import TaskManager
        
        # Initialize each system
        memory = MemoryManager()
        unified = UnifiedMemoryManager()
        tasks = TaskManager()
        
        print("✅ All databases initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize databases: {e}")
        return False

def run_system_test():
    """Run the comprehensive system test."""
    print("\n🧪 Running system tests...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All system tests passed")
            return True
        else:
            print("❌ System tests failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run system tests: {e}")
        return False

def create_sample_project():
    """Create a sample project for demonstration."""
    print("\n📁 Creating sample project...")
    try:
        from mcp.project_manager import ProjectManager
        
        manager = ProjectManager()
        result = manager.init_project("sample_project", "examples/sample_project")
        
        print(f"✅ Sample project created: {result['project_path']}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample project: {e}")
        return False

def build_embedded_python_env():
    """Build a fully embedded Python environment for MCP (cross-platform)."""
    print("\n\U0001F512 Building embedded Python environment...")
    system = platform.system().lower()
    project_root = Path(__file__).resolve().parent.parent
    embedded_env_path = project_root / "embedded_env"
    if system in ["linux", "darwin"]:
        try:
            import shutil
            # Remove existing embedded_env if present
            if embedded_env_path.exists():
                shutil.rmtree(embedded_env_path)
            # Create new conda environment
            print("- Creating conda environment at ./embedded_env ...")
            subprocess.check_call([
                "conda", "create", "-y", "-p", str(embedded_env_path), "python=3.10"
            ])
            # Install dependencies
            print("- Installing dependencies into embedded environment ...")
            subprocess.check_call([
                "conda", "install", "-y", "-p", str(embedded_env_path), "--file", "requirements.txt"])
            # Pack the environment
            print("- Packing environment with conda-pack ...")
            subprocess.check_call([
                "conda", "install", "-y", "-c", "conda-forge", "conda-pack"])
            subprocess.check_call([
                "conda-pack", "-p", str(embedded_env_path), "-o", str(project_root / "embedded_env.tar.gz")
            ])
            print("\u2705 Embedded Python environment built and packed at ./embedded_env.tar.gz")
            return True
        except Exception as e:
            print(f"\u274c Failed to build embedded environment: {e}")
            print("- Please ensure conda and conda-pack are installed and available in your PATH.")
            return False
    elif system == "windows":
        try:
            import urllib.request, zipfile
            # Download embeddable Python zip
            py_zip_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
            py_zip_path = project_root / "python_embed.zip"
            print(f"- Downloading embeddable Python from {py_zip_url} ...")
            urllib.request.urlretrieve(py_zip_url, py_zip_path)
            # Extract zip
            with zipfile.ZipFile(py_zip_path, 'r') as zip_ref:
                zip_ref.extractall(embedded_env_path)
            print("- Embeddable Python extracted to ./embedded_env")
            # Install dependencies into embeddable Python
            requirements_path = project_root / "requirements.txt"
            embedded_python = embedded_env_path / "python.exe"
            pip_path = embedded_env_path / "python.exe"
            if embedded_python.exists() and requirements_path.exists():
                print("- Installing dependencies into embeddable Python environment ...")
                try:
                    subprocess.check_call([
                        str(embedded_python), "-m", "pip", "install", "--upgrade", "pip"
                    ])
                    subprocess.check_call([
                        str(embedded_python), "-m", "pip", "install", "-r", str(requirements_path)
                    ])
                    print("\u2705 Dependencies installed into embeddable Python environment.")
                except Exception as dep_e:
                    print(f"\u274c Failed to install dependencies: {dep_e}")
                    print("- Please run the following command manually in the embedded_env directory:")
                    print(f"  {embedded_python} -m pip install -r {requirements_path}")
            else:
                print("\u274c Could not find embedded Python or requirements.txt for dependency installation.")
            print("\u2705 Embeddable Python environment prepared at ./embedded_env")
            return True
        except Exception as e:
            print(f"\u274c Failed to build embeddable Python environment: {e}")
            return False
    else:
        print(f"\u274c Unsupported platform: {system}")
        print("- [Manual fallback] Use conda-pack or PyInstaller to bundle Python, dependencies, and MCP code into a portable directory or executable.")
        print("- See: https://conda.github.io/conda-pack/ and https://pyinstaller.org/")
        return False

def show_next_steps():
    """Show next steps for using the system."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("1. Initialize a new project:")
    print("   python mcp.py init-project --name 'my_project'")
    print()
    print("2. View available commands:")
    print("   python mcp.py --help")
    print()
    print("3. Start with a sample project:")
    print("   cd examples/sample_project")
    print("   python ../../mcp.py show-questions")
    print()
    print("4. Export context for LLM:")
    print("   python mcp.py export-context --types 'tasks,memories' --max-tokens 1000")
    print()
    print("📚 Documentation:")
    print("- README.md: Comprehensive usage guide")
    print("- idea.txt: Original project vision")
    print("- PROJECT_STATUS_FINAL.md: Complete feature overview")
    print()
    print("🔧 Key Features:")
    print("- Dynamic Q&A system for project alignment")
    print("- Vector memory search with quality assessment")
    print("- Priority tree task management")
    print("- Token-efficient context export for LLMs")
    print("- Accuracy-critical task protection")
    print("- Cross-project learning and memory recall")
    print()
    print("🚀 Ready to accelerate your agentic development workflow!")

def run_lint():
    """Run flake8 linting for code quality enforcement."""
    print("\n🔍 Running flake8 linting...")
    try:
        result = subprocess.run([sys.executable, '-m', 'flake8', 'src/mcp/'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ flake8 linting passed")
            return True
        else:
            print("❌ flake8 linting failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run flake8: {e}")
        return False

def run_static_analysis():
    """Run mypy static type analysis."""
    print("\n🔍 Running mypy static analysis...")
    try:
        result = subprocess.run([sys.executable, '-m', 'mypy', 'src/mcp/'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ mypy static analysis passed")
            return True
        else:
            print("❌ mypy static analysis failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run mypy: {e}")
        return False

def run_test_coverage():
    """Run pytest with coverage reporting."""
    print("\n🧪 Running pytest with coverage...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', '--cov=src/mcp', '--cov-report=term-missing'], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode == 0:
            print("✅ pytest coverage passed")
            return True
        else:
            print("❌ pytest coverage failed")
            return False
    except Exception as e:
        print(f"❌ Failed to run pytest coverage: {e}")
        return False

def migrate_add_memory_order_field(db_path):
    """Add memory_order field to advanced_memories if it does not exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if memory_order column exists
    cursor.execute("PRAGMA table_info(advanced_memories)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'memory_order' not in columns:
        cursor.execute("ALTER TABLE advanced_memories ADD COLUMN memory_order INTEGER DEFAULT 1")
        print("Added memory_order column to advanced_memories.")
    else:
        print("memory_order column already exists.")
    conn.commit()
    conn.close()

def main():
    """Main setup function."""
    print_banner()
    
    # Check if we're in the right directory
    if not os.path.exists("mcp.py"):
        print("❌ Please run this script from the project root directory")
        print("   (where mcp.py is located)")
        return False
    
    # Run setup steps
    steps = [
        ("Build Embedded Python Environment", build_embedded_python_env),
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Initialize Databases", initialize_databases),
        ("Run System Tests", run_system_test),
        ("Run Linting (flake8)", run_lint),
        ("Run Static Analysis (mypy)", run_static_analysis),
        ("Run Test Coverage (pytest)", run_test_coverage),
        ("Create Sample Project", create_sample_project),
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please check the error messages above and try again.")
            return False
    
    # Run security audit after tests
    print("\n🔒 Running Security Audit...")
    auditor = SecurityAuditor()
    audit_results = auditor.run_full_audit()
    print(f"\nSecurity Audit Score: {audit_results.get('score', 'N/A')}")
    print(f"Summary: {audit_results.get('summary', {})}")
    if audit_results.get('issues'):
        print(f"Issues found: {len(audit_results['issues'])}")
        for issue in audit_results['issues']:
            print(f"  - {issue}")
    if audit_results.get('warnings'):
        print(f"Warnings: {len(audit_results['warnings'])}")
        for warning in audit_results['warnings']:
            print(f"  - {warning}")
    if audit_results.get('recommendations'):
        print(f"Recommendations: {len(audit_results['recommendations'])}")
        for rec in audit_results['recommendations']:
            print(f"  - {rec}")
    print("\nSecurity audit complete.")
    
    show_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# Code quality gates: flake8, mypy, and pytest coverage are now enforced in setup. See https://dovetail.com/product-development/iterative-design/ for best practices. 

# If run as a script, perform migration on the default DB
if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'unified_memory.db')
    migrate_add_memory_order_field(db_path) 