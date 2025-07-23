"""
Database Manager Implementation for the MCP system.

This module provides a basic implementation of the IDatabaseManager interface using SQLite.

ℵdatabase_manager(implementation)
"""

import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import os
import json
import shutil
from datetime import datetime

from core.src.mcp.interfaces.database_manager import IDatabaseManager
from core.src.mcp.exceptions import MCPDatabaseError

logger = logging.getLogger(__name__)


class SQLiteDatabaseManager(IDatabaseManager):
    """Implementation of the IDatabaseManager interface using SQLite."""
    
    def __init__(self):
        """Initialize the database manager."""
        self.connection = None
        self.connection_string = None
        logger.info("SQLiteDatabaseManager initialized")
    
    def connect(self, connection_string: str) -> bool:
        """Connect to a database."""
        try:
            # Parse connection string
            if not connection_string.startswith("sqlite://"):
                raise MCPDatabaseError("Invalid connection string format. Expected: sqlite://path/to/database.db")
            
            # Extract database path
            db_path = connection_string[9:]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(db_path)
            self.connection.row_factory = sqlite3.Row
            self.connection_string = connection_string
            
            logger.info(f"Connected to database: {db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise MCPDatabaseError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self) -> bool:
        """Disconnect from the database."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.connection_string = None
                logger.info("Disconnected from database")
                return True
            else:
                logger.warning("Not connected to a database")
                return False
        except Exception as e:
            logger.error(f"Failed to disconnect from database: {str(e)}")
            raise MCPDatabaseError(f"Failed to disconnect from database: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        try:
            self._check_connection()
            
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Get results
            results = []
            for row in cursor.fetchall():
                results.append({key: row[key] for key in row.keys()})
            
            # Commit if not a SELECT query
            if not query.strip().upper().startswith("SELECT"):
                self.connection.commit()
            
            logger.info(f"Executed query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            raise MCPDatabaseError(f"Failed to execute query: {str(e)}")
    
    def execute_transaction(self, queries: List[Tuple[str, Optional[Dict[str, Any]]]]) -> bool:
        """Execute multiple queries as a transaction."""
        try:
            self._check_connection()
            
            cursor = self.connection.cursor()
            
            try:
                # Begin transaction
                self.connection.execute("BEGIN TRANSACTION")
                
                # Execute queries
                for query, params in queries:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                
                # Commit transaction
                self.connection.commit()
                logger.info(f"Executed transaction with {len(queries)} queries")
                return True
            except Exception as e:
                # Rollback transaction
                self.connection.rollback()
                logger.error(f"Transaction failed, rolled back: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to execute transaction: {str(e)}")
            raise MCPDatabaseError(f"Failed to execute transaction: {str(e)}")
    
    def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create a new table with the specified schema."""
        try:
            self._check_connection()
            
            # Build CREATE TABLE statement
            columns = []
            for column_name, column_type in schema.items():
                columns.append(f"{column_name} {column_type}")
            
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            
            # Execute query
            self.execute_query(query)
            
            logger.info(f"Created table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create table: {str(e)}")
            raise MCPDatabaseError(f"Failed to create table: {str(e)}")
    
    def drop_table(self, table_name: str) -> bool:
        """Drop a table."""
        try:
            self._check_connection()
            
            # Build DROP TABLE statement
            query = f"DROP TABLE IF EXISTS {table_name}"
            
            # Execute query
            self.execute_query(query)
            
            logger.info(f"Dropped table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop table: {str(e)}")
            raise MCPDatabaseError(f"Failed to drop table: {str(e)}")
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            self._check_connection()
            
            # Query for tables
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            results = self.execute_query(query)
            
            # Extract table names
            table_names = [result["name"] for result in results]
            
            logger.info(f"Listed {len(table_names)} tables")
            return table_names
        except Exception as e:
            logger.error(f"Failed to list tables: {str(e)}")
            raise MCPDatabaseError(f"Failed to list tables: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the schema of a table."""
        try:
            self._check_connection()
            
            # Query for table schema
            query = f"PRAGMA table_info({table_name})"
            results = self.execute_query(query)
            
            if not results:
                raise MCPDatabaseError(f"Table not found: {table_name}")
            
            # Extract column information
            schema = {}
            for column in results:
                schema[column["name"]] = column["type"]
            
            logger.info(f"Got schema for table: {table_name}")
            return schema
        except Exception as e:
            logger.error(f"Failed to get table schema: {str(e)}")
            raise MCPDatabaseError(f"Failed to get table schema: {str(e)}")
    
    def backup_database(self, backup_path: str) -> bool:
        """Backup the database to the specified path."""
        try:
            self._check_connection()
            
            # Get database path from connection string
            db_path = self.connection_string[9:]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)
            
            # Copy database file
            shutil.copy2(db_path, backup_path)
            
            logger.info(f"Backed up database to: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup database: {str(e)}")
            raise MCPDatabaseError(f"Failed to backup database: {str(e)}")
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore the database from the specified backup."""
        try:
            # Disconnect from current database
            if self.connection:
                self.disconnect()
            
            # Get database path from connection string
            db_path = self.connection_string[9:]
            
            # Copy backup file to database path
            shutil.copy2(backup_path, db_path)
            
            # Reconnect to database
            self.connect(self.connection_string)
            
            logger.info(f"Restored database from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore database: {str(e)}")
            raise MCPDatabaseError(f"Failed to restore database: {str(e)}")
    
    def _check_connection(self) -> None:
        """Check if connected to a database."""
        if not self.connection:
            raise MCPDatabaseError("Not connected to a database")