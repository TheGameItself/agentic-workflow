#!/usr/bin/env python3
"""
MCP Entry Point
Main entry point for the MCP Agentic Workflow Accelerator.
"""

import sys
import os
from pathlib import Path

# Add core directory to Python path
project_root = Path(__file__).parent
core_path = project_root / "core"

if core_path.exists():
    sys.path.insert(0, str(core_path))

try:
    from cli import cli
except ImportError:
    print("❌ Error: Could not import MCP CLI from core directory.")
    print("Please ensure the core system is properly installed.")
    print("Try running: pip install -r requirements.txt")
    sys.exit(1)

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print("\n👋 MCP CLI interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ MCP CLI error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    