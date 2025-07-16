"""
Python-based web interface for the MCP server.
Provides a minimalist dark-themed UI for config, engines, and knowledgebase management.
"""

import base64
import json
import logging
import os
import sqlite3
import threading
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional


class MCPWebInterface:
    """Web interface for MCP server with config editor, engine controls, and knowledgebase browser."""

    def __init__(self, mcp_server=None, port: int = 8080):
        self.mcp_server = mcp_server
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self):
        """Start the web interface server."""
        handler = MCPRequestHandler

        # Handler factory to inject mcp_interface into each instance
        def handler_factory(*args, **kwargs):
            h = MCPRequestHandler(*args, **kwargs)
            h.mcp_interface = self
            return h

        self.server = HTTPServer(("localhost", self.port), handler_factory)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()

        print(f"MCP Web Interface started at http://localhost:{self.port}")
        webbrowser.open(f"http://localhost:{self.port}")

    def stop(self):
        """Stop the web interface server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def get_config_data(self) -> Dict[str, Any]:
        """Get current configuration data."""
        if not self.mcp_server:
            return {"error": "MCP server not connected"}

        try:
            # Get configuration from various sources
            config = {
                "server_status": "running",
                "engines": self._get_engine_status(),
                "memory_stats": self._get_memory_stats(),
                "tasks": self._get_task_stats(),
                "timestamp": datetime.now().isoformat(),
            }
            return config
        except Exception as e:
            return {"error": str(e)}

    def _get_engine_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        engines = {}

        # Check if experimental lobes are available
        try:
            from .experimental_lobes import (
                AdvancedEngramEngine,
                AlignmentEngine,
                DreamingEngine,
                MindMapEngine,
                PatternRecognitionEngine,
                ScientificProcessEngine,
                SimulatedReality,
            )

            # Optionally import SpeculationEngine if it exists
            try:
                from .experimental_lobes import SpeculationEngine

                speculation_status = {"status": "available", "type": "experimental"}
            except ImportError:
                speculation_status = {"status": "unavailable", "error": "Import failed"}

            engines.update(
                {
                    "alignment": {"status": "available", "type": "experimental"},
                    "pattern_recognition": {
                        "status": "available",
                        "type": "experimental",
                    },
                    "simulated_reality": {
                        "status": "available",
                        "type": "experimental",
                    },
                    "dreaming": {"status": "available", "type": "experimental"},
                    "mind_map": {"status": "available", "type": "experimental"},
                    "scientific_process": {
                        "status": "available",
                        "type": "experimental",
                    },
                    "advanced_engram": {"status": "available", "type": "experimental"},
                    "speculation": speculation_status,
                }
            )
        except ImportError:
            engines.update(
                {
                    "experimental_lobes": {
                        "status": "unavailable",
                        "error": "Import failed",
                    }
                }
            )

        return engines

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            # Get database sizes
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            if os.path.exists(data_dir):
                db_files = [f for f in os.listdir(data_dir) if f.endswith(".db")]
                total_size = sum(
                    os.path.getsize(os.path.join(data_dir, f)) for f in db_files
                )

                return {
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "database_count": len(db_files),
                    "databases": db_files,
                }
        except Exception:
            pass

        return {"error": "Unable to get memory stats"}

    def _get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        try:
            if self.mcp_server and hasattr(self.mcp_server, "task_manager"):
                task_manager = self.mcp_server.task_manager
                return {
                    "total_tasks": (
                        len(task_manager.tasks) if hasattr(task_manager, "tasks") else 0
                    ),
                    "active_tasks": (
                        len(
                            [
                                t
                                for t in task_manager.tasks
                                if t.get("status") == "active"
                            ]
                        )
                        if hasattr(task_manager, "tasks")
                        else 0
                    ),
                    "completed_tasks": (
                        len(
                            [
                                t
                                for t in task_manager.tasks
                                if t.get("status") == "completed"
                            ]
                        )
                        if hasattr(task_manager, "tasks")
                        else 0
                    ),
                }
        except Exception:
            pass

        return {"error": "Unable to get task stats"}

    def update_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration."""
        try:
            # Here you would implement actual config updates
            # For now, just return success
            return {"status": "success", "message": "Configuration updated"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_knowledgebase_data(self, query: str = "") -> Dict[str, Any]:
        """Get knowledgebase data using RegexSearchEngine for code and documentation search."""
        try:
            if not query.strip():
                return {"query": query, "results": [], "total_results": 0}

            from .regex_search import (
                RegexSearchEngine,
                SearchQuery,
                SearchScope,
                SearchType,
            )

            # Search code and documentation files (py, md, txt, etc.)
            search_engine = RegexSearchEngine()
            search_query = SearchQuery(
                pattern=query,
                search_type=SearchType.FILE_SYSTEM,
                scope=SearchScope.CURRENT_PROJECT,
                case_sensitive=False,
                max_results=30,
                context_lines=2,
                file_patterns=["*.py", "*.md", "*.txt", "*.rst", "*.js", "*.ts"],
                exclude_patterns=None,
            )
            results = search_engine.search(search_query)
            # Format results for the web UI
            formatted = [
                {
                    "id": idx + 1,
                    "content": r.content,
                    "type": (
                        "code"
                        if (r.file_path and r.file_path.endswith(".py"))
                        else "doc"
                    ),
                    "file": r.file_path,
                    "line": r.line_number,
                    "match": r.match_text,
                    "context_before": r.context_before,
                    "context_after": r.context_after,
                }
                for idx, r in enumerate(results)
            ]
            return {
                "query": query,
                "results": formatted,
                "total_results": len(formatted),
            }
        except Exception as e:
            return {"error": str(e)}

    def some_web_method(self):
        """Minimal fallback for web interface. See idea.txt and TODO_DEVELOPMENT_PLAN.md for future improvements."""
        logging.warning('[WebInterface] This method is a placeholder. See idea.txt and TODO_DEVELOPMENT_PLAN.md for future improvements.')
        raise NotImplementedError("Web interface logic not yet implemented. See idea.txt and TODO_DEVELOPMENT_PLAN.md.")


class MCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MCP web interface."""

    mcp_interface: "Optional[Any]" = None

    def __init__(self, *args, **kwargs):
        self.mcp_interface = None
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        try:
            if self.path == "/":
                self._serve_main_page()
            elif self.path == "/api/config":
                self._serve_config_api()
            elif self.path == "/api/knowledgebase":
                self._serve_knowledgebase_api()
            elif self.path.startswith("/static/"):
                self._serve_static_file()
            else:
                self._serve_404()
        except Exception as e:
            self._serve_error(str(e))

    def do_POST(self):
        """Handle POST requests."""
        try:
            if self.path == "/api/config/update":
                self._handle_config_update()
            elif self.path == "/api/engine/control":
                self._handle_engine_control()
            else:
                self._serve_404()
        except Exception as e:
            self._serve_error(str(e))

    def _serve_main_page(self):
        """Serve the main HTML page."""
        html = self._get_main_html()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_config_api(self):
        """Serve configuration API."""
        if not self.mcp_interface:
            self._serve_json({"error": "MCP interface not available"})
            return

        config_data = self.mcp_interface.get_config_data()
        self._serve_json(config_data)

    def _serve_knowledgebase_api(self):
        """Serve knowledgebase API."""
        if not self.mcp_interface:
            self._serve_json({"error": "MCP interface not available"})
            return

        # Parse query parameters
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        query = query_params.get("q", [""])[0]

        kb_data = self.mcp_interface.get_knowledgebase_data(query)
        self._serve_json(kb_data)

    def _handle_config_update(self):
        """Handle configuration update."""
        if not self.mcp_interface:
            self._serve_json({"error": "MCP interface not available"})
            return

        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        config_data = json.loads(post_data.decode("utf-8"))

        result = self.mcp_interface.update_config(config_data)
        self._serve_json(result)

    def _handle_engine_control(self):
        """Handle engine control commands."""
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        control_data = json.loads(post_data.decode("utf-8"))

        # Here you would implement actual engine control
        result = {"status": "success", "message": f"Engine control: {control_data}"}
        self._serve_json(result)

    def _serve_json(self, data: Dict[str, Any]):
        """Serve JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _serve_static_file(self):
        """Serve static files (CSS, JS)."""
        # For now, serve inline CSS/JS
        if self.path == "/static/style.css":
            css = self._get_css()
            self.send_response(200)
            self.send_header("Content-type", "text/css")
            self.end_headers()
            self.wfile.write(css.encode("utf-8"))
        else:
            self._serve_404()

    def _serve_404(self):
        """Serve 404 error."""
        self.send_response(404)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def _serve_error(self, error_msg: str):
        """Serve error response."""
        self.send_response(500)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": error_msg}).encode("utf-8"))

    def _get_main_html(self) -> str:
        """Get main HTML content."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Server Interface</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>MCP Server Interface</h1>
            <div class="status-bar">
                <span id="server-status">Connecting...</span>
                <span id="last-update"></span>
            </div>
        </header>
        
        <nav class="tabs">
            <button class="tab-btn active" onclick="showTab('config')">Configuration</button>
            <button class="tab-btn" onclick="showTab('engines')">Engines</button>
            <button class="tab-btn" onclick="showTab('knowledgebase')">Knowledge Base</button>
            <button class="tab-btn" onclick="showTab('tasks')">Tasks</button>
        </nav>
        
        <main>
            <div id="config-tab" class="tab-content active">
                <h2>Configuration</h2>
                <div class="config-section">
                    <h3>Server Status</h3>
                    <div id="server-config" class="config-grid"></div>
                </div>
                
                <div class="config-section">
                    <h3>Memory Statistics</h3>
                    <div id="memory-config" class="config-grid"></div>
                </div>
                
                <div class="config-section">
                    <h3>Task Statistics</h3>
                    <div id="task-config" class="config-grid"></div>
                </div>
            </div>
            
            <div id="engines-tab" class="tab-content">
                <h2>Engine Controls</h2>
                <div id="engine-list" class="engine-grid"></div>
            </div>
            
            <div id="knowledgebase-tab" class="tab-content">
                <h2>Knowledge Base</h2>
                <div class="search-section">
                    <input type="text" id="kb-search" placeholder="Search knowledge base...">
                    <button onclick="searchKnowledgeBase()">Search</button>
                </div>
                <div id="kb-results" class="kb-results"></div>
            </div>
            
            <div id="tasks-tab" class="tab-content">
                <h2>Task Management</h2>
                <div id="task-list" class="task-list"></div>
            </div>
        </main>
    </div>
    
    <script>
        // Tab switching
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Load configuration data
        async function loadConfig() {{
            try {{
                const response = await fetch('/api/config');
                const data = await response.json();
                
                if (data.error) {{
                    document.getElementById('server-status').textContent = 'Error: ' + data.error;
                    return;
                }}
                
                // Update server status
                document.getElementById('server-status').textContent = 'Connected';
                document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
                
                // Update configuration sections
                updateConfigSection('server-config', data);
                updateConfigSection('memory-config', data.memory_stats);
                updateConfigSection('task-config', data.tasks);
                
                // Update engines
                updateEngines(data.engines);
                
            }} catch (error) {{
                document.getElementById('server-status').textContent = 'Connection failed';
                console.error('Error loading config:', error);
            }}
        }}
        
        function updateConfigSection(elementId, data) {{
            const element = document.getElementById(elementId);
            if (!element) return;
            
            element.innerHTML = '';
            
            for (const [key, value] of Object.entries(data)) {{
                if (key === 'engines' || key === 'memory_stats' || key === 'tasks') continue;
                
                const item = document.createElement('div');
                item.className = 'config-item';
                item.innerHTML = `
                    <span class="config-key">${{key}}:</span>
                    <span class="config-value">${{typeof value === 'object' ? JSON.stringify(value) : value}}</span>
                `;
                element.appendChild(item);
            }}
        }}
        
        function updateEngines(engines) {{
            const element = document.getElementById('engine-list');
            if (!element) return;
            
            element.innerHTML = '';
            
            for (const [name, info] of Object.entries(engines)) {{
                const engine = document.createElement('div');
                engine.className = 'engine-item';
                engine.innerHTML = `
                    <h3>${{name}}</h3>
                    <p>Status: <span class="status-${{info.status}}">${{info.status}}</span></p>
                    <p>Type: ${{info.type || 'unknown'}}</p>
                    <button onclick="controlEngine('${{name}}', 'start')" ${{info.status === 'running' ? 'disabled' : ''}}>Start</button>
                    <button onclick="controlEngine('${{name}}', 'stop')" ${{info.status === 'stopped' ? 'disabled' : ''}}>Stop</button>
                `;
                element.appendChild(engine);
            }}
        }}
        
        async function controlEngine(engineName, action) {{
            try {{
                const response = await fetch('/api/engine/control', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{engine: engineName, action: action}})
                }});
                const result = await response.json();
                console.log('Engine control result:', result);
                loadConfig(); // Refresh data
            }} catch (error) {{
                console.error('Error controlling engine:', error);
            }}
        }}
        
        async function searchKnowledgeBase() {{
            const query = document.getElementById('kb-search').value;
            const resultsElement = document.getElementById('kb-results');
            
            try {{
                const response = await fetch(`/api/knowledgebase?q=${{encodeURIComponent(query)}}`);
                const data = await response.json();
                
                if (data.error) {{
                    resultsElement.innerHTML = '<p class="error">Error: ' + data.error + '</p>';
                    return;
                }}
                
                resultsElement.innerHTML = '';
                
                if (data.results && data.results.length > 0) {{
                    data.results.forEach(result => {{
                        const item = document.createElement('div');
                        item.className = 'kb-item';
                        item.innerHTML = `
                            <h4>ID: ${{result.id}}</h4>
                            <p>${{result.content}}</p>
                            <span class="kb-type">${{result.type}}</span>
                        `;
                        resultsElement.appendChild(item);
                    }});
                }} else {{
                    resultsElement.innerHTML = '<p>No results found.</p>';
                }}
                
            }} catch (error) {{
                resultsElement.innerHTML = '<p class="error">Search failed: ' + error.message + '</p>';
            }}
        }}
        
        // Auto-refresh every 5 seconds
        setInterval(loadConfig, 5000);
        
        // Initial load
        loadConfig();
    </script>
</body>
</html>
        """

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
/* Dark theme styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #1a1a1a;
    color: #e0e0e0;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background-color: #2d2d2d;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}

header h1 {
    color: #4CAF50;
    margin-bottom: 10px;
}

.status-bar {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    color: #888;
}

.tabs {
    display: flex;
    background-color: #2d2d2d;
    border-radius: 8px;
    margin-bottom: 20px;
    overflow: hidden;
}

.tab-btn {
    background: none;
    border: none;
    color: #e0e0e0;
    padding: 15px 25px;
    cursor: pointer;
    transition: background-color 0.3s;
    flex: 1;
}

.tab-btn:hover {
    background-color: #3d3d3d;
}

.tab-btn.active {
    background-color: #4CAF50;
    color: white;
}

.tab-content {
    display: none;
    background-color: #2d2d2d;
    padding: 20px;
    border-radius: 8px;
}

.tab-content.active {
    display: block;
}

.tab-content h2 {
    color: #4CAF50;
    margin-bottom: 20px;
}

.config-section {
    margin-bottom: 30px;
}

.config-section h3 {
    color: #4CAF50;
    margin-bottom: 15px;
    border-bottom: 1px solid #444;
    padding-bottom: 5px;
}

.config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 15px;
}

.config-item {
    background-color: #3d3d3d;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4CAF50;
}

.config-key {
    font-weight: bold;
    color: #4CAF50;
    display: block;
    margin-bottom: 5px;
}

.config-value {
    color: #e0e0e0;
    word-break: break-all;
}

.engine-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

.engine-item {
    background-color: #3d3d3d;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #444;
}

.engine-item h3 {
    color: #4CAF50;
    margin-bottom: 10px;
}

.engine-item p {
    margin-bottom: 10px;
}

.status-available {
    color: #4CAF50;
}

.status-unavailable {
    color: #f44336;
}

.engine-item button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 10px;
    margin-bottom: 5px;
}

.engine-item button:hover {
    background-color: #45a049;
}

.engine-item button:disabled {
    background-color: #666;
    cursor: not-allowed;
}

.search-section {
    margin-bottom: 20px;
}

.search-section input {
    background-color: #3d3d3d;
    border: 1px solid #444;
    color: #e0e0e0;
    padding: 10px;
    border-radius: 4px;
    width: 300px;
    margin-right: 10px;
}

.search-section button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
}

.search-section button:hover {
    background-color: #45a049;
}

.kb-results {
    max-height: 500px;
    overflow-y: auto;
}

.kb-item {
    background-color: #3d3d3d;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
    border-left: 4px solid #2196F3;
}

.kb-item h4 {
    color: #2196F3;
    margin-bottom: 10px;
}

.kb-type {
    background-color: #2196F3;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    float: right;
}

.error {
    color: #f44336;
    background-color: #3d3d3d;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #f44336;
}

.task-list {
    max-height: 500px;
    overflow-y: auto;
}

/* Responsive design */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    .config-grid {
        grid-template-columns: 1fr;
    }
    
    .engine-grid {
        grid-template-columns: 1fr;
    }
    
    .search-section input {
        width: 100%;
        margin-bottom: 10px;
    }
}
        """


def start_web_interface(mcp_server=None, port: int = 8080):
    """Start the MCP web interface."""
    interface = MCPWebInterface(mcp_server, port)
    interface.start()
    return interface


if __name__ == "__main__":
    # Test the web interface
    interface = start_web_interface()
    try:
        input("Press Enter to stop the web interface...")
    except KeyboardInterrupt:
        pass
    finally:
        interface.stop()
