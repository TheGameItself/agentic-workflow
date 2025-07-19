#!/usr/bin/env python3
"""
MCP Stdio JSON-RPC Server
Entry point for portable, LLM/IDE-friendly MCP server using stdio for communication.

This server implements the full MCP protocol over stdio for seamless integration with:
- Cursor IDE
- VS Code
- Claude Desktop
- LMStudio
- Local AI services
- Any JSON-RPC compatible client

Features:
- Full MCP protocol compliance
- Robust error handling and logging
- Authentication and rate limiting
- Performance monitoring
- Graceful shutdown handling
"""
import sys
import asyncio
import os
import signal
import json
import logging
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .server import MCPServer

class MCPStdioServer:
    """Enhanced stdio JSON-RPC server for MCP protocol."""
    
    def __init__(self):
        self.server = MCPServer()
        self.running = True
        self.logger = self._setup_logging()
        self.request_id = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for stdio server."""
        logger = logging.getLogger("mcp_stdio")
        logger.setLevel(logging.WARNING)  # Reduce noise for stdio
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _create_error_response(self, code: int, message: str, data: Optional[str] = None, request_id: Optional[str] = None) -> str:
        """Create a JSON-RPC error response."""
        error = {
            "code": code,
            "message": message
        }
        if data:
            error["data"] = data
            
        response = {
            "jsonrpc": "2.0",
            "error": error,
            "id": request_id
        }
        return json.dumps(response)
    
    def _create_success_response(self, result: Any, request_id: Optional[str] = None) -> str:
        """Create a JSON-RPC success response."""
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
        return json.dumps(response)
    
    async def handle_request(self, request_data: str) -> str:
        """Handle incoming JSON-RPC request with enhanced error handling."""
        try:
            # Parse request
            try:
                request = json.loads(request_data)
            except json.JSONDecodeError as e:
                return self._create_error_response(
                    -32700, "Parse error", f"Invalid JSON: {str(e)}"
                )
            
            # Validate JSON-RPC structure
            if not isinstance(request, dict):
                return self._create_error_response(
                    -32600, "Invalid Request", "Request must be an object"
                )
            
            if request.get("jsonrpc") != "2.0":
                return self._create_error_response(
                    -32600, "Invalid Request", "jsonrpc must be '2.0'"
                )
            
            method = request.get("method")
            if not method:
                return self._create_error_response(
                    -32600, "Invalid Request", "method is required"
                )
            
            params = request.get("params", {})
            request_id = request.get("id")
            
            # Handle notification (no response needed)
            if request_id is None:
                await self.server.handle_request(request_data)
                return ""
            
            # Process request through main server
            try:
                result = await self.server.handle_request(request_data)
                return result
            except Exception as e:
                self.logger.error(f"Server error: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                return self._create_error_response(
                    -32603, "Internal error", str(e), request_id
                )
                
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(
                -32603, "Internal error", f"Unexpected error: {str(e)}"
            )
    
    async def run(self):
        """Main server loop."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("MCP Stdio JSON-RPC server starting...")
        
        # Send initialization notification
        init_notification = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {
                "version": "1.0.0",
                "capabilities": {
                    "workflow_management": True,
                    "task_management": True,
                    "memory_management": True,
                    "context_management": True,
                    "rag_system": True,
                    "performance_monitoring": True,
                    "experimental_lobes": True
                }
            }
        }
        print(json.dumps(init_notification), flush=True)
        
        # Main request loop
        while self.running:
            try:
                # Read request from stdin
                line = sys.stdin.readline()
                if not line:
                    break  # EOF
                
                line = line.strip()
                if not line:
                    continue
                
                # Process request
                response = await self.handle_request(line)
                if response:  # Only send response if not empty (notifications)
                    print(response, flush=True)
                    
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                error_response = self._create_error_response(
                    -32603, "Internal error", f"Server error: {str(e)}"
                )
                print(error_response, flush=True)
        
        self.logger.info("MCP Stdio server shutdown complete")

async def main():
    """Main entry point."""
    server = MCPStdioServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server interrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1) 