#!/usr/bin/env python3
"""
Implementation Validation Framework for MCP System

This module implements the ImplementationValidator to scan the codebase for completion status,
validate implementations, and ensure no stubs remain in production code.
"""

import ast
import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import subprocess
from collections import defaultdict

from .stub_elimination_engine import StubEliminationEngine, StubInfo


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    status: str  # 'pass', 'fail', 'warning', 'skip'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = 'medium'  # 'critical', 'high', 'medium', 'low'


@dataclass
class ModuleValidation:
    """Validation results for a module."""
    module_name: str
    file_path: str
    total_methods: int
    implemented_methods: int
    stub_methods: int
    test_coverage: float
    validation_results: List[ValidationResult] = field(default_factory=list)
    last_validated: datetime = field(default_factory=datetime.now)


@dataclass
class SystemValidation:
    """Overall system validation results."""
    total_modules: int
    validated_modules: int
    overall_completion: float
    overall_coverage: float
    critical_issues: List[ValidationResult]
    module_validations: List[ModuleValidation] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)


class ImplementationValidator:
    """
    Comprehensive implementation validation framework.
    
    This validator ensures:
    - No stub implementations remain in production code
    - All critical methods have complete implementations
    - Code coverage meets minimum requirements
    - Integration points are properly implemented
    - Error handling is comprehensive
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the implementation validator."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.logger = self._setup_logging()
        self.stub_engine = StubEliminationEngine(str(self.project_root))
        self.validation_config = self._load_validation_config()
        self.module_validations: Dict[str, ModuleValidation] = {}
        self.system_validation: Optional[SystemValidation] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the implementation validator."""
        logger = logging.getLogger("implementation_validator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger    

    def _load_validation_config(self) -> Dict[str, Any]:
        """Load validation configuration."""
        config_path = self.project_root / "config" / "validation_config.json"
        
        default_config = {
            "minimum_coverage": 80.0,
            "critical_modules": [
                "server.py",
                "memory.py", 
                "workflow.py",
                "task_manager.py",
                "context_manager.py"
            ],
            "required_methods": {
                "MCPServer": [
                    "handle_request",
                    "_route_request",
                    "initialize"
                ],
                "MemoryManager": [
                    "add_memory",
                    "search_memories",
                    "get_memory"
                ],
                "TaskManager": [
                    "create_task",
                    "update_task",
                    "get_tasks"
                ]
            },
            "validation_checks": [
                "stub_detection",
                "method_implementation",
                "error_handling",
                "integration_points",
                "test_coverage",
                "documentation"
            ],
            "severity_thresholds": {
                "critical": 0,  # No critical issues allowed
                "high": 5,      # Max 5 high severity issues
                "medium": 20,   # Max 20 medium severity issues
                "low": 50       # Max 50 low severity issues
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load validation config: {e}")
        
        return default_config
    
    def validate_system(self, target_modules: Optional[List[str]] = None) -> SystemValidation:
        """
        Perform comprehensive system validation.
        
        Args:
            target_modules: Specific modules to validate. If None, validates all modules.
            
        Returns:
            SystemValidation results.
        """
        self.logger.info("Starting comprehensive system validation")
        
        # Get modules to validate
        if target_modules is None:
            target_modules = self._discover_modules()
        
        # Validate each module
        module_validations = []
        for module_name in target_modules:
            try:
                validation = self._validate_module(module_name)
                if validation:
                    module_validations.append(validation)
                    self.module_validations[module_name] = validation
            except Exception as e:
                self.logger.error(f"Error validating module {module_name}: {e}")
        
        # Calculate overall metrics
        total_modules = len(target_modules)
        validated_modules = len(module_validations)
        
        if module_validations:
            overall_completion = sum(v.implemented_methods / max(v.total_methods, 1) 
                                   for v in module_validations) / len(module_validations) * 100
            overall_coverage = sum(v.test_coverage for v in module_validations) / len(module_validations)
        else:
            overall_completion = 0.0
            overall_coverage = 0.0
        
        # Collect critical issues
        critical_issues = []
        for validation in module_validations:
            critical_issues.extend([r for r in validation.validation_results 
                                  if r.severity == 'critical'])
        
        self.system_validation = SystemValidation(
            total_modules=total_modules,
            validated_modules=validated_modules,
            overall_completion=overall_completion,
            overall_coverage=overall_coverage,
            critical_issues=critical_issues,
            module_validations=module_validations
        )
        
        self.logger.info(f"System validation complete: {overall_completion:.1f}% implementation, "
                        f"{overall_coverage:.1f}% coverage, {len(critical_issues)} critical issues")
        
        return self.system_validation
    
    def _discover_modules(self) -> List[str]:
        """Discover all Python modules in the project."""
        modules = []
        src_path = self.project_root / "src" / "mcp"
        
        if src_path.exists():
            for py_file in src_path.rglob("*.py"):
                if py_file.name != "__init__.py" and not py_file.name.startswith("test_"):
                    relative_path = py_file.relative_to(src_path)
                    module_name = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]
                    modules.append(module_name)
        
        return modules    

    def _validate_module(self, module_name: str) -> Optional[ModuleValidation]:
        """Validate a specific module."""
        self.logger.debug(f"Validating module: {module_name}")
        
        # Find the module file
        module_path = self._find_module_path(module_name)
        if not module_path or not module_path.exists():
            self.logger.warning(f"Module file not found: {module_name}")
            return None
        
        # Initialize validation result
        validation = ModuleValidation(
            module_name=module_name,
            file_path=str(module_path),
            total_methods=0,
            implemented_methods=0,
            stub_methods=0,
            test_coverage=0.0
        )
        
        # Run validation checks
        validation.validation_results.extend(self._check_stub_detection(module_path))
        validation.validation_results.extend(self._check_method_implementation(module_path))
        validation.validation_results.extend(self._check_error_handling(module_path))
        validation.validation_results.extend(self._check_integration_points(module_path))
        validation.validation_results.extend(self._check_documentation(module_path))
        
        # Calculate method statistics
        method_stats = self._analyze_methods(module_path)
        validation.total_methods = method_stats['total']
        validation.implemented_methods = method_stats['implemented']
        validation.stub_methods = method_stats['stubs']
        
        # Calculate test coverage
        validation.test_coverage = self._calculate_test_coverage(module_name)
        
        return validation
    
    def _find_module_path(self, module_name: str) -> Optional[Path]:
        """Find the file path for a module."""
        src_path = self.project_root / "src" / "mcp"
        module_file = module_name.replace(".", "/") + ".py"
        module_path = src_path / module_file
        
        if module_path.exists():
            return module_path
        
        # Try alternative locations
        alt_path = self.project_root / "src" / module_file
        if alt_path.exists():
            return alt_path
        
        return None
    
    def _check_stub_detection(self, module_path: Path) -> List[ValidationResult]:
        """Check for stub implementations in a module."""
        results = []
        
        try:
            # Use stub engine to detect stubs
            stubs = self.stub_engine.scan_for_stubs([str(module_path.parent)])
            module_stubs = [s for s in stubs if s.file_path == str(module_path)]
            
            if module_stubs:
                critical_stubs = [s for s in module_stubs if s.severity == 'critical']
                high_stubs = [s for s in module_stubs if s.severity == 'high']
                
                if critical_stubs:
                    results.append(ValidationResult(
                        check_name="stub_detection",
                        status="fail",
                        message=f"Found {len(critical_stubs)} critical stubs",
                        details={"stubs": [s.__dict__ for s in critical_stubs]},
                        severity="critical"
                    ))
                
                if high_stubs:
                    results.append(ValidationResult(
                        check_name="stub_detection",
                        status="warning",
                        message=f"Found {len(high_stubs)} high-priority stubs",
                        details={"stubs": [s.__dict__ for s in high_stubs]},
                        severity="high"
                    ))
            else:
                results.append(ValidationResult(
                    check_name="stub_detection",
                    status="pass",
                    message="No stubs detected",
                    severity="low"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_name="stub_detection",
                status="fail",
                message=f"Error during stub detection: {e}",
                severity="high"
            ))
        
        return results    

    def _check_method_implementation(self, module_path: Path) -> List[ValidationResult]:
        """Check method implementations in a module."""
        results = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find all classes and their methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    required_methods = self.validation_config.get("required_methods", {}).get(class_name, [])
                    
                    if required_methods:
                        implemented_methods = []
                        for method_node in node.body:
                            if isinstance(method_node, ast.FunctionDef):
                                implemented_methods.append(method_node.name)
                        
                        missing_methods = set(required_methods) - set(implemented_methods)
                        
                        if missing_methods:
                            results.append(ValidationResult(
                                check_name="method_implementation",
                                status="fail",
                                message=f"Class {class_name} missing required methods: {list(missing_methods)}",
                                details={"class": class_name, "missing_methods": list(missing_methods)},
                                severity="critical"
                            ))
                        else:
                            results.append(ValidationResult(
                                check_name="method_implementation",
                                status="pass",
                                message=f"Class {class_name} has all required methods",
                                details={"class": class_name},
                                severity="low"
                            ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_name="method_implementation",
                status="fail",
                message=f"Error checking method implementation: {e}",
                severity="high"
            ))
        
        return results
    
    def _check_error_handling(self, module_path: Path) -> List[ValidationResult]:
        """Check error handling implementation."""
        results = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count try/except blocks and functions
            try_blocks = 0
            functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_blocks += 1
                elif isinstance(node, ast.FunctionDef):
                    functions += 1
            
            if functions > 0:
                error_handling_ratio = try_blocks / functions
                
                if error_handling_ratio < 0.3:  # Less than 30% of functions have error handling
                    results.append(ValidationResult(
                        check_name="error_handling",
                        status="warning",
                        message=f"Low error handling coverage: {error_handling_ratio:.1%}",
                        details={"ratio": error_handling_ratio, "try_blocks": try_blocks, "functions": functions},
                        severity="medium"
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="error_handling",
                        status="pass",
                        message=f"Good error handling coverage: {error_handling_ratio:.1%}",
                        details={"ratio": error_handling_ratio},
                        severity="low"
                    ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_name="error_handling",
                status="fail",
                message=f"Error checking error handling: {e}",
                severity="medium"
            ))
        
        return results    

    def _check_integration_points(self, module_path: Path) -> List[ValidationResult]:
        """Check integration points and dependencies."""
        results = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common integration patterns
            integration_patterns = [
                ("database", r"sqlite3|SQLAlchemy|\.db"),
                ("logging", r"logging\.|logger\."),
                ("async", r"async\s+def|await\s+"),
                ("error_handling", r"try:|except:|finally:"),
                ("imports", r"from\s+\.|import\s+")
            ]
            
            found_patterns = {}
            for pattern_name, pattern in integration_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                found_patterns[pattern_name] = len(matches)
            
            # Validate integration completeness
            if found_patterns.get("database", 0) > 0 and found_patterns.get("error_handling", 0) == 0:
                results.append(ValidationResult(
                    check_name="integration_points",
                    status="warning",
                    message="Database integration without error handling",
                    details=found_patterns,
                    severity="medium"
                ))
            
            if found_patterns.get("async", 0) > 0 and found_patterns.get("error_handling", 0) == 0:
                results.append(ValidationResult(
                    check_name="integration_points",
                    status="warning",
                    message="Async code without error handling",
                    details=found_patterns,
                    severity="medium"
                ))
            
            results.append(ValidationResult(
                check_name="integration_points",
                status="pass",
                message="Integration points analyzed",
                details=found_patterns,
                severity="low"
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_name="integration_points",
                status="fail",
                message=f"Error checking integration points: {e}",
                severity="medium"
            ))
        
        return results
    
    def _check_documentation(self, module_path: Path) -> List[ValidationResult]:
        """Check documentation completeness."""
        results = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count functions and docstrings
            functions_with_docstrings = 0
            total_functions = 0
            classes_with_docstrings = 0
            total_classes = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        functions_with_docstrings += 1
                elif isinstance(node, ast.ClassDef):
                    total_classes += 1
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        classes_with_docstrings += 1
            
            # Calculate documentation coverage
            if total_functions > 0:
                function_doc_ratio = functions_with_docstrings / total_functions
            else:
                function_doc_ratio = 1.0
            
            if total_classes > 0:
                class_doc_ratio = classes_with_docstrings / total_classes
            else:
                class_doc_ratio = 1.0
            
            overall_doc_ratio = (function_doc_ratio + class_doc_ratio) / 2
            
            if overall_doc_ratio < 0.5:
                results.append(ValidationResult(
                    check_name="documentation",
                    status="warning",
                    message=f"Low documentation coverage: {overall_doc_ratio:.1%}",
                    details={
                        "function_coverage": function_doc_ratio,
                        "class_coverage": class_doc_ratio,
                        "overall_coverage": overall_doc_ratio
                    },
                    severity="medium"
                ))
            else:
                results.append(ValidationResult(
                    check_name="documentation",
                    status="pass",
                    message=f"Good documentation coverage: {overall_doc_ratio:.1%}",
                    details={"overall_coverage": overall_doc_ratio},
                    severity="low"
                ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_name="documentation",
                status="fail",
                message=f"Error checking documentation: {e}",
                severity="low"
            ))
        
        return results    

    def _analyze_methods(self, module_path: Path) -> Dict[str, int]:
        """Analyze method implementation status."""
        stats = {"total": 0, "implemented": 0, "stubs": 0}
        
        try:
            # Use stub engine to get detailed analysis
            stubs = self.stub_engine.scan_for_stubs([str(module_path.parent)])
            module_stubs = [s for s in stubs if s.file_path == str(module_path)]
            
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count all methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    stats["total"] += 1
            
            # Count stubs
            stats["stubs"] = len(module_stubs)
            stats["implemented"] = stats["total"] - stats["stubs"]
            
        except Exception as e:
            self.logger.error(f"Error analyzing methods in {module_path}: {e}")
        
        return stats
    
    def _calculate_test_coverage(self, module_name: str) -> float:
        """Calculate test coverage for a module."""
        try:
            # Try to run coverage analysis
            # This is a simplified implementation - in practice, you'd integrate with coverage.py
            test_file = f"test_{module_name.replace('.', '_')}.py"
            test_paths = [
                self.project_root / "tests" / test_file,
                self.project_root / "src" / "tests" / test_file,
                self.project_root / f"test_{module_name.split('.')[-1]}.py"
            ]
            
            # Check if test file exists
            has_tests = any(path.exists() for path in test_paths)
            
            if has_tests:
                # Simplified coverage calculation
                # In a real implementation, you'd run coverage.py
                return 75.0  # Placeholder
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Error calculating coverage for {module_name}: {e}")
            return 0.0
    
    def ensure_no_stubs_in_production(self) -> bool:
        """
        Ensure no stubs remain in production code.
        
        Returns:
            True if no stubs found, False otherwise.
        """
        self.logger.info("Checking for stubs in production code")
        
        # Scan for stubs
        stubs = self.stub_engine.scan_for_stubs()
        critical_stubs = [s for s in stubs if s.severity in ['critical', 'high']]
        
        if critical_stubs:
            self.logger.error(f"Found {len(critical_stubs)} critical/high-priority stubs in production code")
            for stub in critical_stubs[:5]:  # Show first 5
                self.logger.error(f"  {stub.file_path}:{stub.line_number} - {stub.function_name} ({stub.stub_type})")
            return False
        
        self.logger.info("No critical stubs found in production code")
        return True
    
    def run_automated_tests(self) -> Dict[str, Any]:
        """
        Run automated tests to validate implementations.
        
        Returns:
            Test results summary.
        """
        self.logger.info("Running automated validation tests")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage": 0.0,
            "details": []
        }
        
        try:
            # Try to run pytest if available
            test_command = ["python", "-m", "pytest", "--tb=short", "-v"]
            test_dir = self.project_root / "tests"
            
            if test_dir.exists():
                result = subprocess.run(
                    test_command + [str(test_dir)],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                
                # Parse pytest output (simplified)
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "passed" in line and "failed" in line:
                        # Extract test counts
                        import re
                        match = re.search(r'(\d+) passed.*?(\d+) failed', line)
                        if match:
                            results["tests_passed"] = int(match.group(1))
                            results["tests_failed"] = int(match.group(2))
                            results["tests_run"] = results["tests_passed"] + results["tests_failed"]
                
                results["details"].append({
                    "test_type": "pytest",
                    "status": "completed" if result.returncode == 0 else "failed",
                    "output": result.stdout[:1000]  # Truncate output
                })
            
            # Run core functionality tests
            core_test_results = self._run_core_tests()
            results["details"].extend(core_test_results)
            
        except Exception as e:
            self.logger.error(f"Error running automated tests: {e}")
            results["details"].append({
                "test_type": "error",
                "status": "failed",
                "message": str(e)
            })
        
        return results
    
    def _run_core_tests(self) -> List[Dict[str, Any]]:
        """Run core functionality tests."""
        tests = []
        
        # Test stub engine
        try:
            stubs = self.stub_engine.scan_for_stubs()
            tests.append({
                "test_type": "stub_engine",
                "status": "passed",
                "message": f"Scanned successfully, found {len(stubs)} stubs"
            })
        except Exception as e:
            tests.append({
                "test_type": "stub_engine",
                "status": "failed",
                "message": str(e)
            })
        
        # Test module discovery
        try:
            modules = self._discover_modules()
            tests.append({
                "test_type": "module_discovery",
                "status": "passed",
                "message": f"Discovered {len(modules)} modules"
            })
        except Exception as e:
            tests.append({
                "test_type": "module_discovery",
                "status": "failed",
                "message": str(e)
            })
        
        return tests


# Convenience functions
def validate_mcp_system(project_root: Optional[str] = None) -> SystemValidation:
    """Convenience function to validate the MCP system."""
    validator = ImplementationValidator(project_root)
    return validator.validate_system()


def ensure_production_ready(project_root: Optional[str] = None) -> bool:
    """Convenience function to ensure system is production ready."""
    validator = ImplementationValidator(project_root)
    return validator.ensure_no_stubs_in_production()