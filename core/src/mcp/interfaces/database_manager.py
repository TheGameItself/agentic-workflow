"""
Database Manager Interface for the MCP system.

This module defines the interface for database management components.

ℵdatabase_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


class IDatabaseManager(ABC):
    """Interface for database management components."""
    
    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        """Connect to a database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the database."""
        pass   
 
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass
    
    @abstractmethod
    def execute_transaction(self, queries: List[Tuple[str, Optional[Dict[str, Any]]]]) -> bool:
        """Execute multiple queries as a transaction."""
        pass
    
    @abstractmethod
    def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create a new table with the specified schema."""
        pass
    
    @abstractmethod
    def drop_table(self, table_name: str) -> bool:
        """Drop a table."""
        pass
    
    @abstractmethod
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        pass
    
    @abstractmethod
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the schema of a table."""
        pass
    
    @abstractmethod
    def backup_database(self, backup_path: str) -> bool:
        """Backup the database to the specified path."""
        pass
    
    @abstractmethod
    def restore_database(self, backup_path: str) -> bool:
        """Restore the database from the specified backup."""
        pass