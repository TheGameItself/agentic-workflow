#!/usr/bin/env python3
"""
Performance Monitoring and Optimization System
Tracks system performance and provides optimization recommendations.
Enhanced with Prometheus/Netdata integration and real-time monitoring.
"""

import time
import os
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import threading
from collections import defaultdict, deque
import sympy
import numpy as np
from scipy import stats
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
import subprocess
import psutil
import requests
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    """Represents a single metric with metadata."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    description: str = ""

class PrometheusExporter:
    """Exports metrics in Prometheus format."""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics: List[Metric] = []
        self._start_http_server()
    
    def _start_http_server(self):
        """Start HTTP server for Prometheus metrics endpoint."""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            
            class MetricsHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/metrics':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(self.server.exporter.format_metrics().encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    # Suppress access logs
                    pass
            
            server = HTTPServer(('localhost', self.port), MetricsHandler)
            server.exporter = self
            
            def run_server():
                server.serve_forever()
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"Warning: Could not start Prometheus exporter: {e}")
    
    def add_metric(self, metric: Metric):
        """Add a metric to the exporter."""
        self.metrics.append(metric)
    
    def format_metrics(self) -> str:
        """Format metrics in Prometheus text format."""
        lines = []
        
        # Group metrics by name
        by_name = defaultdict(list)
        for metric in self.metrics:
            by_name[metric.name].append(metric)
        
        for name, metrics in by_name.items():
            # Add metric type and help
            metric_type = metrics[0].metric_type.value
            description = metrics[0].description or f"Metric {name}"
            
            lines.append(f"# HELP {name} {description}")
            lines.append(f"# TYPE {name} {metric_type}")
            
            # Add metric values
            for metric in metrics:
                labels_str = ""
                if metric.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                timestamp_ms = int(metric.timestamp.timestamp() * 1000)
                lines.append(f"{name}{labels_str} {metric.value} {timestamp_ms}")
        
        return "\n".join(lines)

class NetdataIntegration:
    """Integration with Netdata for system monitoring."""
    
    def __init__(self, netdata_url: str = "http://localhost:19999"):
        self.netdata_url = netdata_url
        self.session = requests.Session()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics from Netdata."""
        try:
            response = self.session.get(f"{self.netdata_url}/api/v1/info")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Warning: Could not connect to Netdata: {e}")
        return {}
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            response = self.session.get(f"{self.netdata_url}/api/v1/data?chart=system.cpu")
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    # Get the latest CPU usage value
                    latest = data['data'][-1]
                    return float(latest[1])  # CPU usage is typically the second value
        except Exception as e:
            print(f"Warning: Could not get CPU usage from Netdata: {e}")
        return 0.0
    
    def get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            response = self.session.get(f"{self.netdata_url}/api/v1/data?chart=system.ram")
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    latest = data['data'][-1]
                    return float(latest[1])  # Memory usage percentage
        except Exception as e:
            print(f"Warning: Could not get memory usage from Netdata: {e}")
        return 0.0
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage for all mounted filesystems."""
        try:
            response = self.session.get(f"{self.netdata_url}/api/v1/data?chart=system.io")
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    latest = data['data'][-1]
                    return {
                        'read_bytes': float(latest[1]),
                        'write_bytes': float(latest[2])
                    }
        except Exception as e:
            print(f"Warning: Could not get disk usage from Netdata: {e}")
        return {'read_bytes': 0.0, 'write_bytes': 0.0}

class AlertingSystem:
    """Alerting system for performance monitoring."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.notification_channels: List[Dict[str, Any]] = []
    
    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set thresholds for a metric."""
        self.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
    
    def check_alert(self, metric_name: str, value: float) -> Optional[str]:
        """Check if a metric value triggers an alert."""
        if metric_name not in self.thresholds:
            return None
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['warning']:
            return 'warning'
        
        return None
    
    def add_alert(self, metric_name: str, value: float, severity: str, message: str):
        """Add an alert."""
        alert = {
            'metric_name': metric_name,
            'value': value,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)
        self._send_notification(alert)
    
    def _send_notification(self, alert: Dict[str, Any]):
        """Send notification for an alert."""
        # Simple console notification for now
        print(f"ALERT [{alert['severity'].upper()}] {alert['metric_name']}: {alert['value']} - {alert['message']}")
        
        # Could be extended to send emails, Slack messages, etc.
        for channel in self.notification_channels:
            if channel['type'] == 'console':
                print(f"Alert sent to console: {alert}")
            elif channel['type'] == 'email':
                self._send_email_alert(channel, alert)
            elif channel['type'] == 'slack':
                self._send_slack_alert(channel, alert)
    
    def _send_email_alert(self, channel: Dict[str, Any], alert: Dict[str, Any]):
        """Send email alert (placeholder)."""
        # Implementation would use smtplib or similar
        pass
    
    def _send_slack_alert(self, channel: Dict[str, Any], alert: Dict[str, Any]):
        """Send Slack alert (placeholder)."""
        # Implementation would use Slack API
        pass
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.alerts if alert['severity'] == severity]
        return self.alerts
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()

class RealTimeMonitor:
    """Real-time performance monitoring with continuous data collection."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.prometheus_exporter = PrometheusExporter()
        self.netdata_integration = NetdataIntegration()
        self.alerting_system = AlertingSystem()
        
        # Set up default thresholds
        self.alerting_system.set_threshold('cpu_usage', warning=80.0, critical=95.0)
        self.alerting_system.set_threshold('memory_usage', warning=85.0, critical=95.0)
        self.alerting_system.set_threshold('disk_usage', warning=90.0, critical=95.0)
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self._record_metric('cpu_usage', cpu_percent, MetricType.GAUGE, 
                          {'component': 'system'}, timestamp, "CPU usage percentage")
        
        # Memory usage
        memory = psutil.virtual_memory()
        self._record_metric('memory_usage', memory.percent, MetricType.GAUGE,
                          {'component': 'system'}, timestamp, "Memory usage percentage")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._record_metric('disk_usage', disk_percent, MetricType.GAUGE,
                          {'component': 'system', 'mount': '/'}, timestamp, "Disk usage percentage")
        
        # Network I/O
        network = psutil.net_io_counters()
        self._record_metric('network_bytes_sent', network.bytes_sent, MetricType.COUNTER,
                          {'component': 'network'}, timestamp, "Network bytes sent")
        self._record_metric('network_bytes_recv', network.bytes_recv, MetricType.COUNTER,
                          {'component': 'network'}, timestamp, "Network bytes received")
        
        # Process count
        process_count = len(psutil.pids())
        self._record_metric('process_count', process_count, MetricType.GAUGE,
                          {'component': 'system'}, timestamp, "Number of running processes")
        
        # Check for alerts
        self._check_alerts()
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, 
                      labels: Dict[str, str], timestamp: datetime, description: str = ""):
        """Record a metric."""
        metric = Metric(name, value, metric_type, labels, timestamp, description)
        
        # Store in history
        self.metrics_history[name].append(metric)
        
        # Export to Prometheus
        self.prometheus_exporter.add_metric(metric)
    
    def _check_alerts(self):
        """Check if any metrics trigger alerts."""
        for name, history in self.metrics_history.items():
            if not history:
                continue
            
            latest_metric = history[-1]
            alert_severity = self.alerting_system.check_alert(name, latest_metric.value)
            
            if alert_severity:
                self.alerting_system.add_alert(
                    name, latest_metric.value, alert_severity,
                    f"{name} is {latest_metric.value:.2f}"
                )
    
    def get_metric_history(self, metric_name: str, minutes: int = 60) -> List[Metric]:
        """Get metric history for the last N minutes."""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff_time]
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        current_metrics = {}
        for name, history in self.metrics_history.items():
            if history:
                current_metrics[name] = history[-1].value
        return current_metrics
    
    def get_metric_statistics(self, metric_name: str, minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a metric over the last N minutes."""
        history = self.get_metric_history(metric_name, minutes)
        if not history:
            return {}
        
        values = [m.value for m in history]
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'count': len(values)
        }

class ObjectivePerformanceMonitor:
    """Collects and reports objective performance metrics for the MCP project."""
    def __init__(self, project_path: str):
        if not isinstance(project_path, str) or not project_path:
            raise ValueError("project_path must be a non-empty string")
        self.project_path = project_path
        self.metrics = {}
        self.reports = []
        self.feedback_analytics = FeedbackAnalyticsEngine()
        self.real_time_monitor = RealTimeMonitor()
        self.real_time_monitor.start_monitoring()

    def collect_metrics(self, workflow_manager, task_manager, feedback_model=None):
        """Collect objective metrics from workflow, tasks, and feedback."""
        metrics = {}
        # Task completion rate
        steps = workflow_manager.get_workflow_status().get('steps', {})
        completed = sum(1 for s in steps.values() if s.get('status') == 'completed')
        total = len(steps)
        metrics['task_completion_rate'] = (completed / total * 100) if total else 0
        # Code quality (placeholder: count .py files, could integrate linter)
        code_files = self._count_code_files()
        metrics['code_file_count'] = code_files
        # Test coverage (placeholder: count test files)
        test_files = self._count_test_files()
        metrics['test_file_count'] = test_files
        # Feedback scores (if available)
        if feedback_model and 'feedback_history' in feedback_model:
            scores = [f.get('rating', 0) for f in feedback_model['feedback_history'] if 'rating' in f]
            metrics['avg_feedback_score'] = sum(scores) / len(scores) if scores else None
        # Resource usage (placeholder: disk usage)
        metrics['disk_usage_mb'] = self._get_disk_usage_mb()
        
        # Add real-time system metrics
        current_metrics = self.real_time_monitor.get_current_metrics()
        metrics.update(current_metrics)
        
        self.metrics = metrics
        return metrics

    def generate_report(self, workflow_manager, task_manager, feedback_model=None, milestone=None):
        """Generate a performance report and save it to the project directory."""
        metrics = self.collect_metrics(workflow_manager, task_manager, feedback_model)
        report = {
            'timestamp': datetime.now().isoformat(),
            'milestone': milestone,
            'metrics': metrics,
            'alerts': self.real_time_monitor.alerting_system.get_alerts()
        }
        self.reports.append(report)
        self._save_report(report)
        return report

    def _save_report(self, report: Dict[str, Any]):
        reports_dir = os.path.join(self.project_path, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        fname = f"performance_report_{report['timestamp'].replace(':', '-')}.json"
        with open(os.path.join(reports_dir, fname), 'w') as f:
            json.dump(report, f, indent=2)

    def _count_code_files(self):
        count = 0
        for root, dirs, files in os.walk(os.path.join(self.project_path, 'src')):
            count += sum(1 for f in files if f.endswith('.py'))
        return count

    def _count_test_files(self):
        count = 0
        for root, dirs, files in os.walk(os.path.join(self.project_path, 'tests')):
            count += sum(1 for f in files if f.startswith('test_') and f.endswith('.py'))
        return count

    def _get_disk_usage_mb(self):
        total = 0
        for dirpath, dirnames, filenames in os.walk(self.project_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 2)

    def compute_statistics(self, workflow_manager, task_manager, feedback_model=None) -> dict:
        """Compute statistical summaries: task completion trends, feedback distributions, resource usage."""
        stats = {}
        # Task completion trends
        steps = workflow_manager.get_workflow_status().get('steps', {})
        completed = [s for s in steps.values() if s.get('status') == 'completed']
        in_progress = [s for s in steps.values() if s.get('status') == 'in_progress']
        not_started = [s for s in steps.values() if s.get('status') == 'not_started']
        stats['task_completion'] = {
            'completed': len(completed),
            'in_progress': len(in_progress),
            'not_started': len(not_started),
            'total': len(steps)
        }
        # Feedback distribution
        feedback = (feedback_model or {}).get('feedback_history', [])
        stats['feedback_count'] = len(feedback)
        # Resource usage (disk, files)
        stats['resource_usage'] = self.metrics.get('resource_usage', {})
        
        # Add real-time statistics
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
            stats[f'{metric_name}_stats'] = self.real_time_monitor.get_metric_statistics(metric_name, 60)
        
        return stats

    def generate_statistical_report(self, workflow_manager, task_manager, feedback_model=None, format='json') -> str:
        """Generate a statistical summary report (JSON or Markdown)."""
        stats = self.compute_statistics(workflow_manager, task_manager, feedback_model)
        if format == 'markdown':
            report = ["# MCP Statistical Summary\n"]
            report.append(f"- **Tasks Completed:** {stats['task_completion']['completed']} / {stats['task_completion']['total']}")
            report.append(f"- **Tasks In Progress:** {stats['task_completion']['in_progress']}")
            report.append(f"- **Tasks Not Started:** {stats['task_completion']['not_started']}")
            report.append(f"- **Feedback Entries:** {stats['feedback_count']}")
            report.append(f"- **Resource Usage:** {stats['resource_usage']}")
            
            # Add real-time metrics
            for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
                if f'{metric_name}_stats' in stats:
                    metric_stats = stats[f'{metric_name}_stats']
                    report.append(f"- **{metric_name.replace('_', ' ').title()}:** {metric_stats.get('mean', 0):.2f}% (avg)")
            
            return '\n'.join(report)
        return json.dumps(stats, indent=2)

    def fetch_all_feedback(self, task_manager, workflow_manager, rag_system, reminder_engine, system_feedback=None):
        """Fetch feedback from all sources and return as lists."""
        # Task feedback
        task_feedback = []
        try:
            conn = sqlite3.connect(task_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT task_id, feedback_text, lesson_learned, principle, impact_score, feedback_type, created_at FROM task_feedback")
            for row in cursor.fetchall():
                task_feedback.append({
                    'type': 'task',
                    'task_id': row[0],
                    'feedback': row[1],
                    'lesson_learned': row[2],
                    'principle': row[3],
                    'impact': row[4],
                    'feedback_type': row[5],
                    'timestamp': row[6]
                })
            conn.close()
        except Exception:
            pass
        # Workflow feedback
        workflow_feedback = []
        for step in getattr(workflow_manager, 'steps', {}).values():
            for fb in getattr(step, 'feedback', []):
                entry = dict(fb)
                entry['type'] = 'workflow'
                workflow_feedback.append(entry)
        # RAG feedback
        rag_feedback = []
        try:
            conn = sqlite3.connect(rag_system.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, query, user_feedback, created_at FROM rag_queries WHERE user_feedback IS NOT NULL")
            for row in cursor.fetchall():
                rag_feedback.append({
                    'type': 'rag',
                    'query_id': row[0],
                    'query': row[1],
                    'feedback_score': row[2],
                    'timestamp': row[3]
                })
            conn.close()
        except Exception:
            pass
        # Reminder feedback
        reminder_feedback = []
        try:
            conn = sqlite3.connect(reminder_engine.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_feedback_history FROM enhanced_reminders WHERE user_feedback_history IS NOT NULL")
            for row in cursor.fetchall():
                reminder_id = row[0]
                feedback_history = json.loads(row[1]) if row[1] else []
                for fb in feedback_history:
                    entry = dict(fb)
                    entry['type'] = 'reminder'
                    entry['reminder_id'] = reminder_id
                    reminder_feedback.append(entry)
            conn.close()
        except Exception:
            pass
        # System feedback
        system_fb = system_feedback or []
        for fb in system_fb:
            fb['type'] = 'system'
        return task_feedback, workflow_feedback, rag_feedback, reminder_feedback, system_fb

    def generate_feedback_analytics_report(self, task_manager, workflow_manager, rag_system, reminder_engine, system_feedback=None, format='json'):
        """Aggregate, analyze, and return feedback analytics report in JSON or Markdown."""
        task_fb, workflow_fb, rag_fb, reminder_fb, system_fb = self.fetch_all_feedback(
            task_manager, workflow_manager, rag_system, reminder_engine, system_feedback
        )
        self.feedback_analytics.collect_feedback(
            task_feedback=task_fb,
            workflow_feedback=workflow_fb,
            rag_feedback=rag_fb,
            reminder_feedback=reminder_fb,
            system_feedback=system_fb
        )
        self.feedback_analytics.analyze()
        if format == 'markdown':
            return self.feedback_analytics.to_markdown()
        return self.feedback_analytics.to_json()

    def schedule_periodic_reports(self, workflow_manager, task_manager, rag_system, reminder_engine, interval_minutes: int = 60, system_feedback=None):
        """
        Schedule periodic generation of feedback analytics and performance reports.
        Can be run in a background thread or async task.
        Args:
            workflow_manager: WorkflowManager instance
            task_manager: TaskManager instance
            rag_system: RAGSystem instance
            reminder_engine: EnhancedReminderEngine instance
            interval_minutes: How often to generate reports (default: 60 minutes)
            system_feedback: Optional system feedback list
        """
        def periodic_report_loop():
            while True:
                try:
                    # Generate feedback analytics report
                    feedback_report = self.generate_feedback_analytics_report(
                        task_manager, workflow_manager, rag_system, reminder_engine, system_feedback, format='json'
                    )
                    # Generate performance report
                    perf_report = self.generate_report(
                        workflow_manager, task_manager, feedback_model=None
                    )
                    # Optionally, save or log reports
                    print(f"[PeriodicReport] Feedback Analytics: {feedback_report}")
                    print(f"[PeriodicReport] Performance Report: {perf_report}")
                except Exception as e:
                    print(f"[PeriodicReport] Error generating reports: {e}")
                time.sleep(interval_minutes * 60)
        # To use: run in a background thread or async task
        thread = threading.Thread(target=periodic_report_loop, daemon=True)
        thread.start()

class MathLogicEngine:
    """Advanced calculus, math, logic, tensor, and CUDA engine for the MCP server."""
    def evaluate_expression(self, expr: str) -> dict:
        """Evaluate a symbolic or numeric expression (safe)."""
        try:
            result = sympy.sympify(expr).evalf()
            return {"result": float(result), "symbolic": str(result)}
        except Exception as e:
            return {"error": str(e)}

    def solve_equation(self, equation: str, variable: str = 'x') -> dict:
        """Solve a symbolic equation for a variable."""
        try:
            x = sympy.symbols(variable)
            sol = sympy.solve(equation, x)
            return {"solutions": [str(s) for s in sol]}
        except Exception as e:
            return {"error": str(e)}

    def run_statistical_test(self, data1, data2=None, test_type='t-test') -> dict:
        """Run a statistical test (t-test, chi2, etc.) on provided data."""
        try:
            if test_type == 't-test' and data2 is not None:
                t, p = stats.ttest_ind(data1, data2)
                return {"t_stat": t, "p_value": p}
            elif test_type == 'chi2' and data2 is not None:
                chi2, p, dof, expected = stats.chi2_contingency([data1, data2])
                return {"chi2": chi2, "p_value": p, "dof": dof, "expected": expected.tolist()}
            elif test_type == 'mean':
                return {"mean": float(np.mean(data1)), "std": float(np.std(data1))}
            else:
                return {"error": "Unsupported test type or missing data."}
        except Exception as e:
            return {"error": str(e)}

    def tensor_operation(self, a, b, op='add', use_cuda=False) -> dict:
        """Perform tensor operations (add, multiply, dot, etc.) with optional CUDA acceleration."""
        try:
            if TORCH_AVAILABLE and use_cuda and torch.cuda.is_available():
                ta = torch.tensor(a).cuda()
                tb = torch.tensor(b).cuda()
                if op == 'add':
                    result = (ta + tb).cpu().numpy().tolist()
                elif op == 'multiply':
                    result = (ta * tb).cpu().numpy().tolist()
                elif op == 'dot':
                    result = torch.dot(ta.flatten(), tb.flatten()).item()
                else:
                    return {"error": "Unsupported tensor op."}
                return {"result": result, "device": "cuda"}
            else:
                na = np.array(a)
                nb = np.array(b)
                if op == 'add':
                    result = (na + nb).tolist()
                elif op == 'multiply':
                    result = (na * nb).tolist()
                elif op == 'dot':
                    result = float(np.dot(na.flatten(), nb.flatten()))
                else:
                    return {"error": "Unsupported tensor op."}
                return {"result": result, "device": "cpu"}
        except Exception as e:
            return {"error": str(e)}

class SentimentPhilosophyEngine:
    """Local sentiment and philosophy/argument analysis engine for the MCP server."""
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.analyzer:
            return {"error": "VADER sentiment analyzer not available. Please install vaderSentiment."}
        scores = self.analyzer.polarity_scores(text)
        return {"sentiment": scores}

    def analyze_argument(self, text: str) -> Dict[str, Any]:
        # Simple rule-based argument structure detection
        stance = "neutral"
        if any(w in text.lower() for w in ["should", "must", "need to", "ought"]):
            stance = "prescriptive"
        elif any(w in text.lower() for w in ["is", "are", "was", "were"]):
            stance = "descriptive"
        # Detect presence of premise/conclusion markers
        has_premise = any(w in text.lower() for w in ["because", "since", "as", "due to"])
        has_conclusion = any(w in text.lower() for w in ["therefore", "thus", "so", "hence"])
        return {
            "stance": stance,
            "has_premise": has_premise,
            "has_conclusion": has_conclusion
        }

class GitIntegrationEngine:
    """Portable, local git integration for the MCP server."""
    def _run_git(self, args, cwd=None):
        try:
            result = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, check=True)
            return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "returncode": result.returncode}
        except subprocess.CalledProcessError as e:
            return {"error": e.stderr.strip(), "returncode": e.returncode}
        except Exception as e:
            return {"error": str(e)}

    def status(self, repo_path=None):
        return self._run_git(['status'], cwd=repo_path)

    def commit(self, message, repo_path=None):
        return self._run_git(['commit', '-am', message], cwd=repo_path)

    def diff(self, repo_path=None):
        return self._run_git(['diff'], cwd=repo_path)

    def branch(self, repo_path=None):
        return self._run_git(['branch'], cwd=repo_path)

    def log(self, repo_path=None, n=10):
        return self._run_git(['log', f'-n{n}'], cwd=repo_path)

class FeedbackAnalyticsEngine:
    """Aggregates and analyzes feedback from all MCP sources."""
    def __init__(self):
        self.sources = {}
        self.analytics = {}

    def collect_feedback(self, task_feedback=None, workflow_feedback=None, rag_feedback=None, reminder_feedback=None, system_feedback=None):
        """Aggregate feedback from all sources into a unified list."""
        feedback = []
        if task_feedback:
            feedback.extend(task_feedback)
        if workflow_feedback:
            feedback.extend(workflow_feedback)
        if rag_feedback:
            feedback.extend(rag_feedback)
        if reminder_feedback:
            feedback.extend(reminder_feedback)
        if system_feedback:
            feedback.extend(system_feedback)
        self.sources = {
            'task': task_feedback or [],
            'workflow': workflow_feedback or [],
            'rag': rag_feedback or [],
            'reminder': reminder_feedback or [],
            'system': system_feedback or []
        }
        self.analytics['all_feedback'] = feedback
        return feedback

    def analyze(self):
        """Compute analytics: counts, trends, breakdowns by type/principle/impact, etc."""
        feedback = self.analytics.get('all_feedback', [])
        by_type = defaultdict(list)
        by_principle = defaultdict(list)
        by_impact = defaultdict(list)
        timeline = defaultdict(list)
        for f in feedback:
            ftype = f.get('type') or f.get('feedback_type') or 'general'
            by_type[ftype].append(f)
            principle = f.get('principle') or 'none'
            by_principle[principle].append(f)
            impact = f.get('impact') or f.get('impact_score') or 0
            by_impact[impact].append(f)
            ts = f.get('timestamp') or f.get('created_at')
            if ts:
                timeline[ts[:10]].append(f)  # group by date
        self.analytics['by_type'] = dict(by_type)
        self.analytics['by_principle'] = dict(by_principle)
        self.analytics['by_impact'] = dict(by_impact)
        self.analytics['timeline'] = dict(timeline)
        return self.analytics

    def to_json(self):
        return json.dumps(self.analytics, indent=2)

    def to_markdown(self):
        lines = ["# Feedback Analytics Report\n"]
        lines.append(f"- **Total Feedback Entries:** {len(self.analytics.get('all_feedback', []))}")
        lines.append(f"- **By Type:** { {k: len(v) for k,v in self.analytics.get('by_type', {}).items()} }")
        lines.append(f"- **By Principle:** { {k: len(v) for k,v in self.analytics.get('by_principle', {}).items()} }")
        lines.append(f"- **By Impact:** { {k: len(v) for k,v in self.analytics.get('by_impact', {}).items()} }")
        lines.append(f"- **Timeline (by day):** { {k: len(v) for k,v in self.analytics.get('timeline', {}).items()} }")
        return '\n'.join(lines)

class LessonsLearnedModule:
    """Aggregates and surfaces lessons learned from all feedback sources for post-project review."""
    def __init__(self, task_manager, workflow_manager, rag_system):
        self.task_manager = task_manager
        self.workflow_manager = workflow_manager
        self.rag_system = rag_system
        self.lessons = []
        self.assessment_history = []
        self.optimization_suggestions = []
        self.self_improvement_tasks = []
        self._collect_lessons()

    def _collect_lessons(self):
        self.lessons = []
        # Task feedback lessons
        try:
            conn = sqlite3.connect(self.task_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT lesson_learned, principle, feedback_text, created_at FROM task_feedback WHERE lesson_learned IS NOT NULL AND lesson_learned != ''")
            for row in cursor.fetchall():
                self.lessons.append({
                    'source': 'task',
                    'lesson': row[0],
                    'principle': row[1],
                    'feedback': row[2],
                    'timestamp': row[3]
                })
            conn.close()
        except Exception:
            pass
        # Workflow feedback lessons
        for step in getattr(self.workflow_manager, 'steps', {}).values():
            for fb in getattr(step, 'feedback', []):
                if fb.get('principle') or fb.get('text'):
                    self.lessons.append({
                        'source': 'workflow',
                        'lesson': fb.get('text', ''),
                        'principle': fb.get('principle', ''),
                        'feedback': fb.get('text', ''),
                        'timestamp': fb.get('timestamp', None)
                    })
        # RAG feedback lessons (if any)
        try:
            conn = sqlite3.connect(self.rag_system.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT user_feedback, query, created_at FROM rag_queries WHERE user_feedback IS NOT NULL")
            for row in cursor.fetchall():
                self.lessons.append({
                    'source': 'rag',
                    'lesson': row[1],
                    'principle': '',
                    'feedback': row[0],
                    'timestamp': row[2]
                })
            conn.close()
        except Exception:
            pass

    def export_lessons(self, format: str = 'json') -> str:
        self._collect_lessons()
        if format == 'json':
            return json.dumps(self.lessons, indent=2)
        elif format == 'markdown':
            lines = ["# Lessons Learned\n"]
            for l in self.lessons:
                lines.append(f"- **Source:** {l['source']} | **Lesson:** {l['lesson']} | **Principle:** {l['principle']} | **Feedback:** {l['feedback']} | **Timestamp:** {l['timestamp']}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def run_periodic_self_assessment(self, performance_monitor=None, objective_monitor=None):
        """Run a comprehensive self-assessment of the MCP system performance."""
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'optimization_suggestions': [],
            'self_improvement_tasks': [],
            'system_health': 'good'
        }
        
        # Collect current performance metrics
        if performance_monitor:
            current_metrics = performance_monitor.get_current_metrics()
            assessment['metrics']['system'] = current_metrics
            
            # Check for performance issues
            if current_metrics.get('cpu_usage', 0) > 80:
                assessment['optimization_suggestions'].append({
                    'type': 'performance',
                    'issue': 'High CPU usage detected',
                    'suggestion': 'Consider optimizing heavy operations or reducing concurrent tasks'
                })
                assessment['system_health'] = 'warning'
        
        if objective_monitor:
            try:
                stats = objective_monitor.compute_statistics(
                    self.workflow_manager, 
                    self.task_manager
                )
                assessment['metrics']['objective'] = stats
                
                # Analyze task completion rates
                if stats.get('task_completion_rate', 100) < 70:
                    assessment['optimization_suggestions'].append({
                        'type': 'workflow',
                        'issue': 'Low task completion rate',
                        'suggestion': 'Review task dependencies and complexity'
                    })
                
                # Check for bottlenecks
                if stats.get('avg_task_duration', 0) > 3600:  # 1 hour
                    assessment['optimization_suggestions'].append({
                        'type': 'efficiency',
                        'issue': 'Long average task duration',
                        'suggestion': 'Break down complex tasks into smaller subtasks'
                    })
            except Exception as e:
                assessment['metrics']['objective'] = {'error': str(e)}
        
        # Analyze lessons learned for patterns
        self._collect_lessons()
        if len(self.lessons) > 0:
            recent_lessons = [l for l in self.lessons if l.get('timestamp')]
            if recent_lessons:
                assessment['metrics']['lessons_count'] = len(recent_lessons)
                
                # Identify common issues
                principles = [l.get('principle', '') for l in recent_lessons if l.get('principle')]
                if principles:
                    from collections import Counter
                    common_principles = Counter(principles).most_common(3)
                    assessment['optimization_suggestions'].extend([
                        {
                            'type': 'learning',
                            'issue': f'Common principle: {principle}',
                            'suggestion': f'Focus on improving {principle} in future tasks'
                        }
                        for principle, count in common_principles
                    ])
        
        # Generate self-improvement tasks
        assessment['self_improvement_tasks'] = self._generate_self_improvement_tasks(assessment)
        
        # Store assessment
        self.assessment_history.append(assessment)
        
        # Keep only last 10 assessments
        if len(self.assessment_history) > 10:
            self.assessment_history = self.assessment_history[-10:]
        
        return assessment
    
    def _generate_self_improvement_tasks(self, assessment):
        """Generate specific tasks for self-improvement based on assessment."""
        tasks = []
        
        # Performance optimization tasks
        if assessment['system_health'] == 'warning':
            tasks.append({
                'title': 'Optimize system performance',
                'description': 'Investigate and resolve performance bottlenecks',
                'priority': 'high',
                'estimated_duration': '2 hours'
            })
        
        # Learning tasks based on lessons
        if assessment['metrics'].get('lessons_count', 0) > 5:
            tasks.append({
                'title': 'Review recent lessons learned',
                'description': 'Analyze patterns in recent feedback and lessons',
                'priority': 'medium',
                'estimated_duration': '1 hour'
            })
        
        # Workflow optimization tasks
        if any(s.get('type') == 'workflow' for s in assessment['optimization_suggestions']):
            tasks.append({
                'title': 'Optimize workflow processes',
                'description': 'Review and improve task dependencies and workflow efficiency',
                'priority': 'medium',
                'estimated_duration': '3 hours'
            })
        
        return tasks
    
    def schedule_periodic_assessment(self, interval_hours: int = 24):
        """Schedule periodic self-assessment to run automatically."""
        import threading
        import time
        
        def assessment_loop():
            while True:
                try:
                    self.run_periodic_self_assessment()
                    time.sleep(interval_hours * 3600)  # Convert hours to seconds
                except Exception as e:
                    print(f"Error in periodic assessment: {e}")
                    time.sleep(3600)  # Wait 1 hour before retrying
        
        thread = threading.Thread(target=assessment_loop, daemon=True)
        thread.start()
        return thread
    
    def get_latest_assessment(self):
        """Get the most recent self-assessment."""
        if self.assessment_history:
            return self.assessment_history[-1]
        return None
    
    def get_assessment_trends(self, days: int = 7):
        """Analyze trends in assessments over time."""
        if not self.assessment_history:
            return {}
        
        recent_assessments = [
            a for a in self.assessment_history 
            if (datetime.now() - datetime.fromisoformat(a['timestamp'])).days <= days
        ]
        
        if not recent_assessments:
            return {}
        
        trends = {
            'total_assessments': len(recent_assessments),
            'avg_suggestions_per_assessment': sum(len(a.get('optimization_suggestions', [])) for a in recent_assessments) / len(recent_assessments),
            'system_health_distribution': {},
            'common_issues': []
        }
        
        # Analyze system health trends
        health_counts = {}
        for a in recent_assessments:
            health = a.get('system_health', 'unknown')
            health_counts[health] = health_counts.get(health, 0) + 1
        trends['system_health_distribution'] = health_counts
        
        # Find common issues
        all_suggestions = []
        for a in recent_assessments:
            all_suggestions.extend(a.get('optimization_suggestions', []))
        
        if all_suggestions:
            from collections import Counter
            issue_types = [s.get('type', 'unknown') for s in all_suggestions]
            trends['common_issues'] = Counter(issue_types).most_common(5)
        
        return trends 

    def get_performance_summary(self):
        """Return a summary of performance metrics. See idea.txt for requirements."""
        return {}

    def optimize_database(self):
        """Optimize the database or return a safe fallback. See idea.txt for requirements."""
        return {"success": True, "message": "No-op"} 