#!/usr/bin/env python3
"""
Security Audit Script for MCP Server
Performs comprehensive security analysis and validation.
"""

import os
import sys
import json
import sqlite3
import hashlib
import secrets
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import ast
import importlib.util


class SecurityAuditor:
    """Comprehensive security auditor for the MCP server.
    Now includes AST-based static analysis for dangerous patterns (eval, exec, etc.),
    and generates a summary of findings with actionable next steps.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.ast_issues = []  # New: AST-based issues
        
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete security audit, including AST-based static analysis."""
        print("Starting comprehensive security audit...")
        
        results = {
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "score": 100,
            "summary": {}
        }
        
        # Run all audit checks
        checks = [
            self._audit_input_validation,
            self._audit_authentication,
            self._audit_authorization,
            self._audit_data_validation,
            self._audit_sql_injection,
            self._audit_file_operations,
            self._audit_network_security,
            self._audit_cryptography,
            self._audit_configuration,
            self._audit_dependencies,
            self._audit_logging,
            self._audit_error_handling,
            self._audit_ast_dangerous_patterns  # New: AST-based static analysis
        ]
        
        for check in checks:
            try:
                check_result = check()
                results["issues"].extend(check_result.get("issues", []))
                results["warnings"].extend(check_result.get("warnings", []))
                results["recommendations"].extend(check_result.get("recommendations", []))
            except Exception as e:
                results["issues"].append(f"Audit check failed: {e}")
        
        # Calculate security score
        results["score"] = self._calculate_security_score(results)
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        # Add actionable next steps
        results["next_steps"] = self._generate_next_steps(results)
        
        return results
    
    def _audit_input_validation(self) -> Dict[str, List[str]]:
        """Audit input validation and sanitization."""
        print("Auditing input validation...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for input validation in API endpoints
        api_files = list(self.project_root.rglob("*.py"))
        
        for file_path in api_files:
            if "api" in str(file_path).lower() or "server" in str(file_path).lower():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for missing input validation
                if "def" in content and "request" in content:
                    if not re.search(r'validate|sanitize|clean', content, re.IGNORECASE):
                        warnings.append(f"Missing input validation in {file_path}")
                        recommendations.append(f"Add input validation to {file_path}")
                
                # Check for dangerous input handling
                dangerous_patterns = [
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'__import__\s*\(',
                    r'getattr\s*\(.*,\s*[\'"]__.*[\'"]',
                    r'setattr\s*\(.*,\s*[\'"]__.*[\'"]'
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, content):
                        issues.append(f"Dangerous code pattern found in {file_path}: {pattern}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_authentication(self) -> Dict[str, List[str]]:
        """Audit authentication mechanisms."""
        print("Auditing authentication...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for hardcoded credentials
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for hardcoded passwords/keys
            hardcoded_patterns = [
                r'password\s*=\s*[\'"][^\'"]+[\'"]',
                r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
                r'secret\s*=\s*[\'"][^\'"]+[\'"]',
                r'token\s*=\s*[\'"][^\'"]+[\'"]'
            ]
            
            for pattern in hardcoded_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) > 5:  # Likely not a placeholder
                        issues.append(f"Hardcoded credential found in {file_path}: {match}")
        
        # Check for weak authentication
        auth_files = list(self.project_root.rglob("*auth*.py"))
        if not auth_files:
            warnings.append("No dedicated authentication module found")
            recommendations.append("Implement dedicated authentication module")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_authorization(self) -> Dict[str, List[str]]:
        """Audit authorization and access control."""
        print("Auditing authorization...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for missing access controls
        for file_path in self.project_root.rglob("*.py"):
            if "api" in str(file_path).lower() or "server" in str(file_path).lower():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for missing authorization checks
                if "def" in content and any(method in content for method in ["get", "post", "put", "delete"]):
                    if not re.search(r'authorize|permission|role|access', content, re.IGNORECASE):
                        warnings.append(f"Missing authorization checks in {file_path}")
                        recommendations.append(f"Add authorization checks to {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_data_validation(self) -> Dict[str, List[str]]:
        """Audit data validation and sanitization."""
        print("Auditing data validation...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for SQL injection vulnerabilities
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for string concatenation in SQL
            sql_patterns = [
                r'execute\s*\(\s*[\'"][^\'"]*\+',
                r'execute\s*\(\s*f[\'"][^\'"]*\{',
                r'execute\s*\(\s*[\'"][^\'"]*%s[^\'"]*[\'"]\s*%',
                r'execute\s*\(\s*[\'"][^\'"]*\{[^\'"]*[\'"]\s*\.format'
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, content):
                    issues.append(f"Potential SQL injection vulnerability in {file_path}")
                    recommendations.append(f"Use parameterized queries in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_sql_injection(self) -> Dict[str, List[str]]:
        """Audit SQL injection vulnerabilities."""
        print("Auditing SQL injection vulnerabilities...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check database operations
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for database operations
            if "sqlite" in content.lower() or "database" in content.lower():
                # Check for unsafe SQL construction
                unsafe_patterns = [
                    r'cursor\.execute\s*\(\s*[\'"][^\'"]*\+',
                    r'cursor\.execute\s*\(\s*f[\'"][^\'"]*\{',
                    r'cursor\.execute\s*\(\s*[\'"][^\'"]*%s[^\'"]*[\'"]\s*%',
                    r'cursor\.execute\s*\(\s*[\'"][^\'"]*\{[^\'"]*[\'"]\s*\.format'
                ]
                
                for pattern in unsafe_patterns:
                    if re.search(pattern, content):
                        issues.append(f"SQL injection vulnerability in {file_path}")
                        recommendations.append(f"Use parameterized queries in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_file_operations(self) -> Dict[str, List[str]]:
        """Audit file operation security."""
        print("Auditing file operations...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for unsafe file operations
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for path traversal vulnerabilities
            dangerous_patterns = [
                r'open\s*\(\s*[^\'"]*\+',
                r'open\s*\(\s*f[\'"][^\'"]*\{',
                r'Path\s*\(\s*[^\'"]*\+',
                r'Path\s*\(\s*f[\'"][^\'"]*\{'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, content):
                    warnings.append(f"Potential path traversal vulnerability in {file_path}")
                    recommendations.append(f"Validate file paths in {file_path}")
            
            # Check for unsafe file permissions
            if "chmod" in content and "0o777" in content:
                issues.append(f"Unsafe file permissions in {file_path}")
                recommendations.append(f"Use restrictive file permissions in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_network_security(self) -> Dict[str, List[str]]:
        """Audit network security."""
        print("Auditing network security...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for unsafe network operations
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for HTTP without TLS
            if "http://" in content and not "https://" in content:
                warnings.append(f"HTTP without TLS in {file_path}")
                recommendations.append(f"Use HTTPS in {file_path}")
            
            # Check for hardcoded URLs
            if re.search(r'url\s*=\s*[\'"][^\'"]*http://[^\'"]*[\'"]', content):
                issues.append(f"Hardcoded HTTP URL in {file_path}")
                recommendations.append(f"Use HTTPS URLs in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_cryptography(self) -> Dict[str, List[str]]:
        """Audit cryptographic implementations."""
        print("Auditing cryptography...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for weak cryptographic algorithms
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for weak hashing algorithms
            weak_algorithms = ["md5", "sha1"]
            for algorithm in weak_algorithms:
                if algorithm in content.lower():
                    warnings.append(f"Weak cryptographic algorithm {algorithm} in {file_path}")
                    recommendations.append(f"Use SHA-256 or better in {file_path}")
            
            # Check for hardcoded encryption keys
            if re.search(r'key\s*=\s*[\'"][^\'"]{8,}[\'"]', content):
                issues.append(f"Hardcoded encryption key in {file_path}")
                recommendations.append(f"Use environment variables for keys in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_configuration(self) -> Dict[str, List[str]]:
        """Audit configuration security."""
        print("Auditing configuration security...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check configuration files
        config_files = list(self.project_root.rglob("*.json")) + list(self.project_root.rglob("*.yaml"))
        
        for config_file in config_files:
            if "config" in str(config_file).lower():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix == '.json':
                            config = json.load(f)
                        else:
                            import yaml
                            config = yaml.safe_load(f)
                    
                    # Check for sensitive data in config
                    sensitive_keys = ["password", "secret", "key", "token", "api_key"]
                    for key in sensitive_keys:
                        if key in str(config).lower():
                            warnings.append(f"Sensitive data in configuration file {config_file}")
                            recommendations.append(f"Use environment variables for sensitive data in {config_file}")
                
                except Exception as e:
                    warnings.append(f"Could not parse configuration file {config_file}: {e}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_dependencies(self) -> Dict[str, List[str]]:
        """Audit dependency security."""
        print("Auditing dependencies...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for known vulnerabilities
        try:
            # Try to use safety if available
            result = subprocess.run(["safety", "check"], capture_output=True, text=True)
            if result.returncode != 0:
                warnings.append("Dependency vulnerability check failed")
                recommendations.append("Install and run 'safety check' for dependency audit")
        except FileNotFoundError:
            warnings.append("Safety tool not found")
            recommendations.append("Install safety: pip install safety")
        
        # Check requirements files
        req_files = ["requirements.txt", "requirements-dev.txt"]
        for req_file in req_files:
            if Path(req_file).exists():
                with open(req_file, 'r') as f:
                    content = f.read()
                
                # Check for pinned versions
                unpinned = re.findall(r'^[a-zA-Z0-9_-]+$', content, re.MULTILINE)
                if unpinned:
                    warnings.append(f"Unpinned dependencies in {req_file}")
                    recommendations.append(f"Pin dependency versions in {req_file}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_logging(self) -> Dict[str, List[str]]:
        """Audit logging security."""
        print("Auditing logging security...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for sensitive data in logs
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for logging sensitive data
            sensitive_patterns = [
                r'log.*password',
                r'log.*secret',
                r'log.*token',
                r'log.*api_key'
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"Sensitive data logging in {file_path}")
                    recommendations.append(f"Remove sensitive data from logs in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_error_handling(self) -> Dict[str, List[str]]:
        """Audit error handling security."""
        print("Auditing error handling...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for information disclosure in error messages
        for file_path in self.project_root.rglob("*.py"):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for detailed error messages
            if "traceback" in content.lower() or "stack trace" in content.lower():
                warnings.append(f"Potential information disclosure in {file_path}")
                recommendations.append(f"Sanitize error messages in {file_path}")
            
            # Check for exception handling
            if "raise" in content and "except" not in content:
                warnings.append(f"Unhandled exceptions in {file_path}")
                recommendations.append(f"Add proper exception handling in {file_path}")
        
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _audit_ast_dangerous_patterns(self) -> Dict[str, List[str]]:
        """AST-based static analysis for dangerous patterns (eval, exec, etc.)."""
        print("Running AST-based static analysis for dangerous patterns...")
        issues = []
        warnings = []
        recommendations = []
        for file_path in self.project_root.rglob("*.py"):
            try:
                with open(file_path, 'r') as f:
                    source = f.read()
                tree = ast.parse(source, filename=str(file_path))
                for node in ast.walk(tree):
                    func_name = None
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            func_name = node.func.attr
                        if func_name in ('eval', 'exec', 'compile', 'execfile'):
                            issues.append(f"Dangerous function '{func_name}' used in {file_path} at line {node.lineno}")
                            recommendations.append(f"Remove or refactor use of '{func_name}' in {file_path}")
            except Exception as e:
                warnings.append(f"AST analysis failed for {file_path}: {e}")
        return {"issues": issues, "warnings": warnings, "recommendations": recommendations}
    
    def _calculate_security_score(self, results: Dict[str, Any]) -> int:
        """Calculate security score based on audit results."""
        score = 100
        
        # Deduct points for issues
        score -= len(results["issues"]) * 10
        
        # Deduct points for warnings
        score -= len(results["warnings"]) * 5
        
        # Ensure score doesn't go below 0
        return max(0, score)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit summary."""
        return {
            "total_issues": len(results["issues"]),
            "total_warnings": len(results["warnings"]),
            "total_recommendations": len(results["recommendations"]),
            "security_score": results["score"],
            "risk_level": self._get_risk_level(results["score"])
        }
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level based on security score."""
        if score >= 90:
            return "LOW"
        elif score >= 70:
            return "MEDIUM"
        elif score >= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps based on audit results."""
        next_steps = []
        if results["issues"]:
            next_steps.append("Review and remediate all critical issues found in the audit.")
        if results["warnings"]:
            next_steps.append("Address warnings to improve security posture.")
        if results["recommendations"]:
            next_steps.append("Implement recommendations for best practices and compliance.")
        if not (results["issues"] or results["warnings"] or results["recommendations"]):
            next_steps.append("No critical issues found. Maintain regular audits and code reviews.")
        return next_steps
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "security_audit_report.json"):
        """Generate detailed security audit report."""
        report = {
            "timestamp": str(Path().cwd()),
            "project": str(self.project_root),
            "audit_results": results,
            "recommendations": {
                "immediate": [issue for issue in results["issues"]],
                "short_term": [warning for warning in results["warnings"]],
                "long_term": [rec for rec in results["recommendations"]]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Security audit report saved to: {output_file}")
        return report


def main():
    """Main function to run security audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run security audit on MCP server")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default="security_audit_report.json", help="Output report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor(args.project_root)
    results = auditor.run_full_audit()
    
    # Print summary
    print("\n" + "="*50)
    print("SECURITY AUDIT SUMMARY")
    print("="*50)
    print(f"Security Score: {results['score']}/100")
    print(f"Risk Level: {results['summary']['risk_level']}")
    print(f"Total Issues: {results['summary']['total_issues']}")
    print(f"Total Warnings: {results['summary']['total_warnings']}")
    print(f"Total Recommendations: {results['summary']['total_recommendations']}")
    
    if results["issues"]:
        print("\nCRITICAL ISSUES:")
        for issue in results["issues"]:
            print(f"  ❌ {issue}")
    
    if results["warnings"]:
        print("\nWARNINGS:")
        for warning in results["warnings"]:
            print(f"  ⚠️  {warning}")
    
    if results["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  💡 {rec}")
    
    # Generate report
    auditor.generate_report(results, args.output)
    
    # Exit with appropriate code
    if results["issues"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 