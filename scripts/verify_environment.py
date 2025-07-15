#!/usr/bin/env python3
"""
Environment Verification Script
Verifies Python version, dependencies, and portability for the MCP server.
This script is designed to be portable and make no system changes.
"""

import sys
import os
import platform
import subprocess
import importlib
import json
from pathlib import Path

# Minimum Python version
MIN_PYTHON_VERSION = (3, 8)

# Required dependencies
REQUIRED_PACKAGES = {
    'click': '8.0.0',
    'sqlite3': None,  # Built-in
    'json': None,     # Built-in
    'configparser': None,  # Built-in
    're': None,       # Built-in
    'base64': None,   # Built-in
    'inspect': None,  # Built-in
    'shutil': None,   # Built-in
    'pathlib': None,  # Built-in
    'packaging': None,  # For version checks
}

# Optional dependencies (for enhanced features)
OPTIONAL_PACKAGES = {
    'sentence_transformers': '2.0.0',
    'numpy': '1.20.0',
    'flask': '2.0.0',
}

def get_project_root():
    """Get the project root directory."""
    current = Path(__file__).resolve().parent.parent
    return current

def check_python_version():
    """Check if Python version meets requirements."""
    current_version = sys.version_info[:2]
    if current_version < MIN_PYTHON_VERSION:
        return False, f"Python {'.'.join(map(str, MIN_PYTHON_VERSION))}+ required, found {'.'.join(map(str, current_version))}"
    return True, f"Python {'.'.join(map(str, current_version))} ✓"

def check_platform():
    """Check platform compatibility."""
    system = platform.system().lower()
    if system in ['linux', 'darwin', 'windows']:
        return True, f"Platform: {platform.system()} {platform.release()} ✓"
    return False, f"Unsupported platform: {platform.system()}"

def check_package(package_name, min_version=None):
    """Check if a package is available and meets version requirements."""
    try:
        module = importlib.import_module(package_name)
        if min_version:
            # Try to get version info
            version = getattr(module, '__version__', None)
            if version:
                try:
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) < pkg_version.parse(min_version):
                        return False, f"{package_name} {min_version}+ required, found {version}"
                except ImportError:
                    # packaging not available, skip version check
                    pass
        return True, f"{package_name} ✓"
    except ImportError:
        return False, f"{package_name} not found"

def check_embedded_python():
    """Check if we're running in an embedded Python environment."""
    # Check for embedded Python indicators
    embedded_indicators = [
        'pythonw.exe' in sys.executable.lower(),
        'embedded' in sys.executable.lower(),
        hasattr(sys, 'frozen'),
        'site-packages' not in sys.path
    ]
    
    is_embedded = any(embedded_indicators)
    return is_embedded, f"Embedded Python: {'Yes' if is_embedded else 'No'}"

def check_portability():
    """Check portability aspects."""
    issues = []
    
    # Check for hardcoded paths
    project_root = get_project_root()
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(('.py', '.cfg', '.md')):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Check for hardcoded separators
                        if '\\' in content and platform.system() != 'Windows':
                            issues.append(f"Windows separator in {file_path.relative_to(project_root)}")
                        if '/' in content and platform.system() == 'Windows':
                            # This is actually fine for Python, but note it
                            print('[verify_environment] Fallback: check not implemented yet. See idea.txt for future improvements.')
                except Exception:
                    pass
    
    status = '✓' if len(issues) == 0 else f'{len(issues)} issues found'
    return len(issues) == 0, f"Portability: {status}"

def check_file_structure():
    """Check if required files and directories exist (now checks src/mcp/)."""
    project_root = get_project_root()
    # Updated to reflect new structure: source files are in src/mcp/
    required_structure = [
        'src/mcp/__init__.py',
        'src/mcp/memory.py',
        'src/mcp/workflow.py',
        'src/mcp/context_manager.py',
        'src/mcp/vector_memory.py',
        'src/mcp/advanced_memory.py',
        'src/mcp/task_manager.py',
        'src/mcp/project_manager.py',
        'src/mcp/unified_memory.py',
        'src/mcp/cli.py',
        'requirements.txt',
        'idea.txt'
    ]
    missing = []
    for item in required_structure:
        if not (project_root / item).exists():
            missing.append(item)
    status = '✓' if len(missing) == 0 else f'Missing: {", ".join(missing)}'
    return len(missing) == 0, f"File structure: {status}"

def check_dependencies():
    """Check all required and optional dependencies."""
    results = {}
    
    # Check required packages
    for package, min_version in REQUIRED_PACKAGES.items():
        success, message = check_package(package, min_version)
        results[package] = {'required': True, 'success': success, 'message': message}
    
    # Check optional packages
    for package, min_version in OPTIONAL_PACKAGES.items():
        success, message = check_package(package, min_version)
        results[package] = {'required': False, 'success': success, 'message': message}
    
    return results

def check_embedded_env_presence():
    """Check for presence of embedded Python environment (cross-platform)."""
    project_root = get_project_root()
    system = platform.system().lower()
    found = False
    details = []
    if system in ["linux", "darwin"]:
        # Look for conda-packed or PyInstaller bundle
        if (project_root / "embedded_env").exists():
            found = True
            details.append("conda-packed environment found: ./embedded_env")
        elif any(f.suffix in [".tar.gz", ".tar", ".zip"] and "embedded_env" in f.name for f in project_root.iterdir()):
            found = True
            details.append("conda-packed archive found in project root")
        elif (project_root / "dist").exists():
            found = True
            details.append("PyInstaller bundle found: ./dist/")
    elif system == "windows":
        # Look for embeddable zip or PythonEmbed4Win
        if any(f.suffix == ".zip" and "python" in f.name.lower() for f in project_root.iterdir()):
            found = True
            details.append("Python embeddable zip found in project root")
        elif (project_root / "PythonEmbed4Win.ps1").exists():
            found = True
            details.append("PythonEmbed4Win script found")
    status = '✓' if found else 'Not found'
    msg = f"Embedded Python Environment: {status}"
    if details:
        msg += " (" + "; ".join(details) + ")"
    if not found:
        msg += "\n  [Action Required] Please build the embedded environment using setup.py or follow the platform-specific instructions."
    return found, msg

def run_verification():
    """Run all verification checks."""
    print("🔍 MCP Server Environment Verification")
    print("=" * 50)
    
    checks = []
    
    # Python version
    success, message = check_python_version()
    checks.append(('Python Version', success, message))
    print(f"Python: {message}")
    
    # Platform
    success, message = check_platform()
    checks.append(('Platform', success, message))
    print(f"Platform: {message}")
    
    # Embedded Python
    success, message = check_embedded_python()
    checks.append(('Embedded Python', success, message))
    print(f"Embedded: {message}")
    
    # Embedded Python Environment presence
    success, message = check_embedded_env_presence()
    checks.append(('Embedded Python Environment', success, message))
    print(f"Embedded Env: {message}")
    
    # File structure
    success, message = check_file_structure()
    checks.append(('File Structure', success, message))
    print(f"Files: {message}")
    
    # Portability
    success, message = check_portability()
    checks.append(('Portability', success, message))
    print(f"Portability: {message}")
    
    # Dependencies
    print("\n📦 Dependencies:")
    dep_results = check_dependencies()
    for package, result in dep_results.items():
        status = "✓" if result['success'] else "✗"
        req_marker = "[REQUIRED]" if result['required'] else "[OPTIONAL]"
        print(f"  {status} {package} {req_marker}: {result['message']}")
        checks.append((f"Dependency: {package}", result['success'], result['message']))
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, success, _ in checks if success)
    total = len(checks)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed! Environment is ready for MCP server.")
        return True
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        return False

def generate_report():
    """Generate a detailed verification report."""
    report = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'platform': platform.platform(),
        'project_root': str(get_project_root()),
        'checks': {}
    }
    
    # Run all checks and collect results
    checks = [
        ('python_version', check_python_version),
        ('platform', check_platform),
        ('embedded_python', check_embedded_python),
        ('embedded_env_presence', check_embedded_env_presence),
        ('file_structure', check_file_structure),
        ('portability', check_portability),
        ('dependencies', check_dependencies)
    ]
    
    for name, check_func in checks:
        if name == 'dependencies':
            report['checks'][name] = check_func()
        else:
            success, message = check_func()
            report['checks'][name] = {'success': success, 'message': message}
    
    return report

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--report':
        # Generate JSON report
        report = generate_report()
        print(json.dumps(report, indent=2))
    else:
        # Run interactive verification
        success = run_verification()
        sys.exit(0 if success else 1) 