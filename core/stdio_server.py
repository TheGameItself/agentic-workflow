#!/usr/bin/env python3
"""
MCP Core System stdio JSON-RPC Server
Provides JSON-RPC interface to the MCP Core System over stdio.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

# Add core/src to path
core_src = Path(__file__).parent / "src"
sys.path.insert(0, str(core_src))

from mcp.core_system import initialize_core_system, shutdown_core_system, SystemConfiguration

# Configure logging to file
log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "stdio_server.log"),
    ]
)

logger = logging.getLogger("stdio_server")

class StdioJSONRPCServer:
    """JSON-RPC server that communicates over stdio."""
    
    def __init__(self):
        """Initialize the server."""
        self.system = None
        self.running = False
        self._error_responses = {
            'invalid_request': lambda id: {
                    'jsonrpc': '2.0',
                    'error': {
                        'code': -32600,
                        'message': 'Invalid Request'
                    },
                    'id': request_data.get('id')
                }

            # Extract method and params
            method = request_data.get('method')
            params = request_data.get('params', {})
            request_id = request_data.get('id')

            if not method:
                return {
                    'jsonrpc': '2.0',
                    'error': {
                        'code': -32601,
                        'message': 'Method not found'
                'id': id
                    },
