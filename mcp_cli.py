#!/usr/bin/env python3
"""
MCP Entry Point
Simple entry point for the MCP Agentic Workflow Accelerator.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp.cli import cli

if __name__ == '__main__':
    cli() 
    