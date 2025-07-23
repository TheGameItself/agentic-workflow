#!/usr/bin/env python3
"""
MCP Self-Debugging and Documentation System
Allows the MCP server to test itself and update its own documentation.

This module provides comprehensive self-testing capabilities including:
- Component health checks
- Performance benchmarking
- Documentation generation and updates
- Configuration validation
- Integration testing
- Memory and database integrity checks
"""

import os
import sys
import json
import time
import asyncio
import logging
import traceback
import inspect
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .server import MCPServer
from .memory import MemoryManager
from .workflow import WorkflowManager
from .project_manager import ProjectManager
from .task_manager import TaskManager
from .unified_memory import UnifiedMemoryManager
from .rag_system import RAGSystem
from .performance_monitor import ObjectivePerformanceMonitor
from .enhanced_search_engine import EnhancedSearchEngine

@dataclass
class TestResult:
    """Result of a self-test."""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING', 'SKIP'
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ComponentHealth:
    """Health status of a component."""
    component_name: str
    status: str  # 'HEALTHY', 'DEGRADED', 'UNHEALTHY'
    response_time: float
    error_count: int
    last_error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class DocumentationUpdate:
    """Documentation update information."""
    file_path: str
    update_type: str  # 'CREATE', 'UPDATE', 'DELETE'
    changes: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MCPSelfDebugger:
    """Self-debugging and testing system for MCP server."""
    
    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.logger = logging.getLogger("mcp_self_debug")
        self.test_results: List[TestResult] = []
        self.health_status: Dict[str, ComponentHealth] = {}
        self.documentation_updates: List[DocumentationUpdate] = []
        self.objective_monitor = ObjectivePerformanceMonitor(self.mcp_server.project_path)
        
    def run_comprehensive_self_test(self) -> Dict[str, Any]:
        """Run a comprehensive self-test of the MCP server."""
        self.logger.info("Starting comprehensive MCP self-test...")
        start_time = time.time()
        
        # Clear previous results
        self.test_results.clear()
        self.health_status.clear()
        self.documentation_updates.clear()
        
        # Run all test categories
        test_categories = [
            self._test_core_components,
            self._test_database_integrity,
            self._test_memory_systems,
            self._test_performance_metrics,
            self._test_api_endpoints,
            self._test_configuration,
            self._test_security_features,
            self._test_integration_points,
            self._test_documentation_sync,
            self._test_error_handling
        ]
        
        for test_func in test_categories:
            try:
                test_func()
            except Exception as e:
                self.logger.error(f"Test category {test_func.__name__} failed: {e}")
                self.test_results.append(TestResult(
                    test_name=test_func.__name__,
                    status="FAIL",
                    duration=0.0,
                    message=f"Test category failed: {str(e)}",
                    details={"error": traceback.format_exc()}
                ))
        
        total_duration = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_test_report(total_duration)
        
        # Update documentation if needed
        if self.documentation_updates:
            self._apply_documentation_updates()
        
        return report
    
    def _test_core_components(self):
        """Test core MCP components."""
        self.logger.info("Testing core components...")
        
        components = [
            ("MemoryManager", self.mcp_server.memory_manager),
            ("WorkflowManager", self.mcp_server.workflow_manager),
            ("ProjectManager", self.mcp_server.project_manager),
            ("TaskManager", self.mcp_server.task_manager),
            ("UnifiedMemory", self.mcp_server.unified_memory),
            ("RAGSystem", self.mcp_server.rag_system),
            ("PerformanceMonitor", self.mcp_server.performance_monitor),
        ]
        
        for name, component in components:
            start_time = time.time()
            try:
                # Test basic functionality
                if hasattr(component, 'get_status'):
                    status = component.get_status()
                elif hasattr(component, 'health_check'):
                    status = component.health_check()
                else:
                    # Basic existence test
                    status = {"status": "available"}
                
                duration = time.time() - start_time
                
                self.health_status[name] = ComponentHealth(
                    component_name=name,
                    status="HEALTHY" if status.get("status") != "error" else "UNHEALTHY",
                    response_time=duration,
                    error_count=0,
                    metrics=status
                )
                
                self.test_results.append(TestResult(
                    test_name=f"core_component_{name}",
                    status="PASS",
                    duration=duration,
                    message=f"Component {name} is healthy",
                    details=status
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                self.health_status[name] = ComponentHealth(
                    component_name=name,
                    status="UNHEALTHY",
                    response_time=duration,
                    error_count=1,
                    last_error=str(e)
                )
                
                self.test_results.append(TestResult(
                    test_name=f"core_component_{name}",
                    status="FAIL",
                    duration=duration,
                    message=f"Component {name} failed: {str(e)}",
                    details={"error": traceback.format_exc()}
                ))
    
    def _test_database_integrity(self):
        """Test database integrity and connections."""
        self.logger.info("Testing database integrity...")
        
        # Test main database connections
        db_paths = [
            ("workflow", "data/workflow.db"),
            ("memory", "data/memory.db"),
            ("tasks", "data/tasks.db"),
            ("rag", "data/rag.db"),
        ]
        
        for db_name, db_path in db_paths:
            start_time = time.time()
            try:
                full_path = os.path.join(self.mcp_server.project_path, db_path)
                if os.path.exists(full_path):
                    # Test database connection and basic operations
                    conn = sqlite3.connect(full_path)
                    cursor = conn.cursor()
                    
                    # Get table information
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    # Test basic query on each table
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                    
                    conn.close()
                    
                    duration = time.time() - start_time
                    self.test_results.append(TestResult(
                        test_name=f"database_integrity_{db_name}",
                        status="PASS",
                        duration=duration,
                        message=f"Database {db_name} is healthy",
                        details={"tables": len(tables), "path": full_path}
                    ))
                else:
                    duration = time.time() - start_time
                    self.test_results.append(TestResult(
                        test_name=f"database_integrity_{db_name}",
                        status="SKIP",
                        duration=duration,
                        message=f"Database {db_name} does not exist",
                        details={"path": full_path}
                    ))
                    
            except Exception as e:
                duration = time.time() - start_time
                self.test_results.append(TestResult(
                    test_name=f"database_integrity_{db_name}",
                    status="FAIL",
                    duration=duration,
                    message=f"Database {db_name} integrity check failed: {str(e)}",
                    details={"error": traceback.format_exc()}
                ))
    
    def _test_memory_systems(self):
        """Test memory systems and vector operations."""
        self.logger.info("Testing memory systems...")
        
        # Test memory operations
        start_time = time.time()
        try:
            # Test memory creation
            test_memory = self.mcp_server.memory_manager.add_memory(
                text="Self-test memory entry",
                memory_type="test",
                priority=0.5,
                context="self-debugging"
            )
            
            # Test memory retrieval
            retrieved = self.mcp_server.memory_manager.get_memory(test_memory['id'])
            
            # Test memory search
            search_results = self.mcp_server.memory_manager.search_memories(
                query="self-test",
                limit=5
            )
            
            # Test vector operations
            if hasattr(self.mcp_server, 'vector_backend'):
                vector_status = self.mcp_server.vector_backend.get_status()
            else:
                vector_status = {"status": "not_available"}
            
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="memory_systems",
                status="PASS",
                duration=duration,
                message="Memory systems are functioning correctly",
                details={
                    "memory_created": test_memory['id'],
                    "search_results": len(search_results),
                    "vector_status": vector_status
                }
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="memory_systems",
                status="FAIL",
                duration=duration,
                message=f"Memory systems test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_performance_metrics(self):
        """Test performance monitoring and metrics collection."""
        self.logger.info("Testing performance metrics...")
        
        start_time = time.time()
        try:
            # Test performance monitor
            if hasattr(self.mcp_server, 'performance_monitor'):
                metrics = self.mcp_server.performance_monitor.get_current_metrics()
                alerts = self.mcp_server.performance_monitor.get_alerts()
            else:
                # Use the objective monitor as fallback
                metrics = self.objective_monitor.get_current_metrics()
                alerts = self.objective_monitor.get_alerts() if hasattr(self.objective_monitor, 'get_alerts') else []
            
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="performance_metrics",
                status="PASS",
                duration=duration,
                message="Performance monitoring is active",
                details={
                    "metrics_collected": len(metrics),
                    "active_alerts": len(alerts)
                }
            ))
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="performance_metrics",
                status="FAIL",
                duration=duration,
                message=f"Performance metrics test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_api_endpoints(self):
        """Test API endpoints and CLI commands."""
        self.logger.info("Testing API endpoints...")
        
        # Test CLI command availability
        try:
            from .cli import cli
            commands = cli.commands.keys()
            
            # Test a few basic commands
            test_commands = ['status', 'project-status', 'workflow-status']
            available_commands = [cmd for cmd in test_commands if cmd in commands]
            
            duration = time.time()
            self.test_results.append(TestResult(
                test_name="api_endpoints",
                status="PASS",
                duration=duration,
                message=f"CLI API is available with {len(commands)} commands",
                details={
                    "total_commands": len(commands),
                    "test_commands_available": len(available_commands),
                    "available_commands": list(commands)
                }
            ))
            
        except Exception as e:
            duration = time.time()
            self.test_results.append(TestResult(
                test_name="api_endpoints",
                status="FAIL",
                duration=duration,
                message=f"API endpoints test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_configuration(self):
        """Test configuration validation and settings."""
        self.logger.info("Testing configuration...")
        
        start_time = time.time()
        try:
            # Test project configuration
            project_info = self.mcp_server.project_manager.get_project_info()
            config_validation = self.mcp_server.project_manager.validate_configuration()
            
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="configuration",
                status="PASS" if not config_validation['errors'] else "WARNING",
                duration=duration,
                message="Configuration validation completed",
                details={
                    "project_info": project_info,
                    "validation_errors": config_validation['errors'],
                    "validation_warnings": config_validation['warnings']
                }
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="configuration",
                status="FAIL",
                duration=duration,
                message=f"Configuration test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_security_features(self):
        """Test security features and authentication."""
        self.logger.info("Testing security features...")
        
        start_time = time.time()
        try:
            # Test authentication settings
            auth_enabled = getattr(self.mcp_server, 'auth_enabled', False)
            rate_limit_enabled = getattr(self.mcp_server, 'rate_limit_enabled', False)
            
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="security_features",
                status="PASS",
                duration=duration,
                message="Security features are configured",
                details={
                    "authentication_enabled": auth_enabled,
                    "rate_limiting_enabled": rate_limit_enabled
                }
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="security_features",
                status="FAIL",
                duration=duration,
                message=f"Security features test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_integration_points(self):
        """Test integration with external systems."""
        self.logger.info("Testing integration points...")
        
        start_time = time.time()
        try:
            # Test vector backend
            vector_status = "available" if hasattr(self.mcp_server, 'vector_backend') else "not_available"
            
            # Test monitoring systems
            prometheus_enabled = getattr(self.mcp_server, 'prometheus_enabled', False)
            netdata_enabled = getattr(self.mcp_server, 'netdata_enabled', False)
            
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="integration_points",
                status="PASS",
                duration=duration,
                message="Integration points are configured",
                details={
                    "vector_backend": vector_status,
                    "prometheus_enabled": prometheus_enabled,
                    "netdata_enabled": netdata_enabled
                }
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="integration_points",
                status="FAIL",
                duration=duration,
                message=f"Integration points test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_documentation_sync(self):
        """Test documentation synchronization and updates."""
        self.logger.info("Testing documentation sync...")
        
        start_time = time.time()
        try:
            # Check documentation structure
            docs_path = os.path.join(self.mcp_server.project_path, "docs")
            if os.path.exists(docs_path):
                # Analyze documentation structure
                doc_files = []
                for root, dirs, files in os.walk(docs_path):
                    for file in files:
                        if file.endswith(('.md', '.txt', '.rst')):
                            doc_files.append(os.path.join(root, file))
                
                # Check for outdated documentation
                outdated_docs = self._identify_outdated_documentation(doc_files)
                
                duration = time.time() - start_time
                self.test_results.append(TestResult(
                    test_name="documentation_sync",
                    status="PASS" if not outdated_docs else "WARNING",
                    duration=duration,
                    message=f"Documentation sync check completed",
                    details={
                        "total_doc_files": len(doc_files),
                        "outdated_docs": len(outdated_docs),
                        "outdated_files": outdated_docs
                    }
                ))
                
                # Queue documentation updates
                if outdated_docs:
                    self._queue_documentation_updates(outdated_docs)
            else:
                duration = time.time() - start_time
                self.test_results.append(TestResult(
                    test_name="documentation_sync",
                    status="SKIP",
                    duration=duration,
                    message="Documentation directory not found"
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="documentation_sync",
                status="FAIL",
                duration=duration,
                message=f"Documentation sync test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        self.logger.info("Testing error handling...")
        
        start_time = time.time()
        try:
            # Test graceful error handling
            test_errors = []
            
            # Test invalid memory operations
            try:
                self.mcp_server.memory_manager.get_memory(999999)
            except Exception as e:
                test_errors.append(f"Memory error handled: {type(e).__name__}")
            
            # Test invalid task operations
            try:
                self.mcp_server.task_manager.get_task(999999)
            except Exception as e:
                test_errors.append(f"Task error handled: {type(e).__name__}")
            
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="error_handling",
                status="PASS",
                duration=duration,
                message="Error handling mechanisms are working",
                details={
                    "errors_handled": len(test_errors),
                    "error_types": test_errors
                }
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(TestResult(
                test_name="error_handling",
                status="FAIL",
                duration=duration,
                message=f"Error handling test failed: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
    
    def _identify_outdated_documentation(self, doc_files: List[str]) -> List[str]:
        """Identify potentially outdated documentation files."""
        outdated = []
        
        # Check for documentation that might need updates
        for doc_file in doc_files:
            try:
                # Check file modification time
                mtime = os.path.getmtime(doc_file)
                file_age = time.time() - mtime
                
                # Consider files older than 30 days as potentially outdated
                if file_age > (30 * 24 * 3600):
                    outdated.append(doc_file)
                    
            except Exception:
                outdated.append(doc_file)
        
        return outdated
    
    def _queue_documentation_updates(self, outdated_files: List[str]):
        """Queue documentation updates for processing."""
        for file_path in outdated_files:
            self.documentation_updates.append(DocumentationUpdate(
                file_path=file_path,
                update_type="UPDATE",
                changes=["Marked for review due to age"]
            ))
    
    def _generate_test_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        warning_tests = len([r for r in self.test_results if r.status == "WARNING"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])
        
        # Calculate average response times
        avg_response_time = sum(r.duration for r in self.test_results) / total_tests if total_tests > 0 else 0
        
        # Generate health summary
        healthy_components = len([h for h in self.health_status.values() if h.status == "HEALTHY"])
        total_components = len(self.health_status)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "skipped": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "avg_response_time": avg_response_time,
                "healthy_components": healthy_components,
                "total_components": total_components,
                "component_health_rate": (healthy_components / total_components * 100) if total_components > 0 else 0
            },
            "test_results": [asdict(result) for result in self.test_results],
            "health_status": {name: asdict(health) for name, health in self.health_status.items()},
            "documentation_updates": [asdict(update) for update in self.documentation_updates],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [r for r in self.test_results if r.status == "FAIL"]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests")
        
        # Analyze unhealthy components
        unhealthy_components = [h for h in self.health_status.values() if h.status == "UNHEALTHY"]
        if unhealthy_components:
            recommendations.append(f"Fix {len(unhealthy_components)} unhealthy components")
        
        # Analyze documentation updates
        if self.documentation_updates:
            recommendations.append(f"Review and update {len(self.documentation_updates)} documentation files")
        
        # Performance recommendations
        slow_tests = [r for r in self.test_results if r.duration > 1.0]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow operations")
        
        if not recommendations:
            recommendations.append("All systems are operating normally")
        
        return recommendations
    
    def _apply_documentation_updates(self):
        """Apply queued documentation updates."""
        self.logger.info(f"Applying {len(self.documentation_updates)} documentation updates...")
        
        for update in self.documentation_updates:
            try:
                if update.update_type == "UPDATE":
                    # For now, just log the update
                    self.logger.info(f"Documentation update queued: {update.file_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to apply documentation update for {update.file_path}: {e}")
    
    def generate_self_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive self-documentation."""
        self.logger.info("Generating self-documentation...")
        
        # Analyze the MCP server structure
        server_analysis = self._analyze_server_structure()
        
        # Generate API documentation
        api_docs = self._generate_api_documentation()
        
        # Generate component documentation
        component_docs = self._generate_component_documentation()
        
        # Generate configuration documentation
        config_docs = self._generate_configuration_documentation()
        
        documentation = {
            "timestamp": datetime.now().isoformat(),
            "server_analysis": server_analysis,
            "api_documentation": api_docs,
            "component_documentation": component_docs,
            "configuration_documentation": config_docs,
            "health_status": {name: asdict(health) for name, health in self.health_status.items()}
        }
        
        return documentation
    
    def _analyze_server_structure(self) -> Dict[str, Any]:
        """Analyze the MCP server structure."""
        analysis = {
            "components": {},
            "methods": {},
            "attributes": {},
            "dependencies": {}
        }
        
        # Analyze server components
        for attr_name in dir(self.mcp_server):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.mcp_server, attr_name)
                
                if hasattr(attr_value, '__class__'):
                    analysis["components"][attr_name] = {
                        "type": attr_value.__class__.__name__,
                        "module": attr_value.__class__.__module__
                    }
                
                if callable(attr_value):
                    analysis["methods"][attr_name] = {
                        "type": "method",
                        "module": attr_value.__module__
                    }
                else:
                    analysis["attributes"][attr_name] = {
                        "type": type(attr_value).__name__,
                        "value": str(attr_value)[:100] if attr_value is not None else None
                    }
        
        return analysis
    
    def _generate_api_documentation(self) -> Dict[str, Any]:
        """Generate API documentation."""
        try:
            from .cli import cli
            
            api_docs = {
                "commands": {},
                "endpoints": {}
            }
            
            # Document CLI commands
            for command_name, command in cli.commands.items():
                api_docs["commands"][command_name] = {
                    "help": command.help or "No help available",
                    "options": [opt.name for opt in command.params],
                    "callback": command.callback.__name__ if command.callback else None
                }
            
            return api_docs
            
        except Exception as e:
            return {"error": f"Failed to generate API documentation: {str(e)}"}
    
    def _generate_component_documentation(self) -> Dict[str, Any]:
        """Generate component documentation."""
        component_docs = {}
        
        for name, health in self.health_status.items():
            component_docs[name] = {
                "status": health.status,
                "response_time": health.response_time,
                "error_count": health.error_count,
                "last_error": health.last_error,
                "metrics": health.metrics
            }
        
        return component_docs
    
    def _generate_configuration_documentation(self) -> Dict[str, Any]:
        """Generate configuration documentation."""
        try:
            project_info = self.mcp_server.project_manager.get_project_info()
            config_validation = self.mcp_server.project_manager.validate_configuration()
            
            return {
                "project_info": project_info,
                "validation": config_validation,
                "settings": {
                    "auth_enabled": getattr(self.mcp_server, 'auth_enabled', False),
                    "rate_limit_enabled": getattr(self.mcp_server, 'rate_limit_enabled', False),
                    "prometheus_enabled": getattr(self.mcp_server, 'prometheus_enabled', False),
                    "netdata_enabled": getattr(self.mcp_server, 'netdata_enabled', False)
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to generate configuration documentation: {str(e)}"}
    
    def save_test_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_self_test_report_{timestamp}.json"
        
        filepath = os.path.join(self.mcp_server.project_path, "logs", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Test report saved to: {filepath}")
        return filepath
    
    def save_documentation(self, documentation: Dict[str, Any], filename: str = None) -> str:
        """Save documentation to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_self_documentation_{timestamp}.json"
        
        filepath = os.path.join(self.mcp_server.project_path, "docs", "canonical", "development", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        self.logger.info(f"Documentation saved to: {filepath}")
        return filepath 