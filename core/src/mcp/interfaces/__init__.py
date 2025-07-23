"""
Core interfaces for the MCP system.

This module defines the base interfaces used throughout the MCP system.
All components should implement the appropriate interfaces.

λinterface_definition(core_components)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import common dependencies
import os
import sys
import json
import logging
from datetime import datetime

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)