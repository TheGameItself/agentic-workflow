#!/usr/bin/env python3
"""
Simple test script for the Automatic Update System
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from mcp.automatic_update_system import AutomaticUpdateSystem, UpdateStatus, UpdateConfig
    print("✅ Successfully imported AutomaticUpdateSystem")
    
    # Test basic initialization
    update_system = AutomaticUpdateSystem()
    print(f"✅ Update system initialized with version: {update_system.current_version}")
    
    # Test status
    status = update_system.get_status()
    print(f"✅ Status retrieved: {status['update_status']}")
    
    # Test update check
    update_status, version_info = update_system.check_for_updates()
    print(f"✅ Update check completed: {update_status.value}")
    
    print("\n🎉 Automatic Update System is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1) 