"""
Database Manager Test Suite
@{CORE.MCP.TESTS.DATABASE.001} Comprehensive validation suite for database manager implementations.
#{database,testing,validation,sqlite,transactions,backup_restore}
β(ℵ(Δ(database_manager_test_validation)))

This module provides comprehensive unit tests for the SQLiteDatabaseManager,
including transaction handling, schema operations, and backup/restore functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import sqlite3

from core.src.mcp.implementations.database_manager import SQLiteDatabaseManager
from core.src.mcp.exceptions import MCPDatabaseError


class TestSQLiteDatabaseManager(unittest.TestCase):
    """Test cases for the SQLiteDatabaseManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.connection_string = f"sqlite://{self.db_path}"
        
        # Create database manager
        self.db_manager = SQLiteDatabaseManager()
        self.db_manager.connect(self.connection_string)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Disconnect from database
        if self.db_manager.connection:
            self.db_manager.disconnect()
        
        # Remove temporary files
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_connect_disconnect(self):
        """β:connection_lifecycle(database_connectivity_validation) - Test connecting and disconnecting from a database."""
        # Disconnect
        result = self.db_manager.disconnect()
        self.assertTrue(result)
        self.assertIsNone(self.db_manager.connection)
        
        # Connect again
        result = self.db_manager.connect(self.connection_string)
        self.assertTrue(result)
        self.assertIsNotNone(self.db_manager.connection)
    
    def test_create_drop_table(self):
        """Ω:schema_management(table_lifecycle_validation) - Test creating and dropping a table."""
        # Create table
        schema = {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "value": "INTEGER"
        }
        result = self.db_manager.create_table("test_table", schema)
        self.assertTrue(result)
        
        # Verify table exists
        tables = self.db_manager.list_tables()
        self.assertIn("test_table", tables)
        
        # Drop table
        result = self.db_manager.drop_table("test_table")
        self.assertTrue(result)
        
        # Verify table no longer exists
        tables = self.db_manager.list_tables()
        self.assertNotIn("test_table", tables)
    
    def test_execute_query(self):
        """λ:query_execution(sql_operation_validation) - Test executing a query."""
        # Create table
        self.db_manager.create_table("test_table", {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "value": "INTEGER"
        })
        
        # Insert data
        self.db_manager.execute_query(
            "INSERT INTO test_table (name, value) VALUES (:name, :value)",
            {"name": "Test", "value": 42}
        )
        
        # Query data
        results = self.db_manager.execute_query(
            "SELECT * FROM test_table WHERE name = :name",
            {"name": "Test"}
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Test")
        self.assertEqual(results[0]["value"], 42)
    
    def test_execute_transaction(self):
        """Δ:transaction_processing(atomic_operation_validation) - Test executing a transaction."""
        # Create table
        self.db_manager.create_table("test_table", {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "value": "INTEGER"
        })
        
        # Define queries
        queries = [
            ("INSERT INTO test_table (name, value) VALUES (:name, :value)", {"name": "Item1", "value": 10}),
            ("INSERT INTO test_table (name, value) VALUES (:name, :value)", {"name": "Item2", "value": 20}),
            ("INSERT INTO test_table (name, value) VALUES (:name, :value)", {"name": "Item3", "value": 30})
        ]
        
        # Execute transaction
        result = self.db_manager.execute_transaction(queries)
        self.assertTrue(result)
        
        # Verify data was inserted
        results = self.db_manager.execute_query("SELECT COUNT(*) as count FROM test_table")
        self.assertEqual(results[0]["count"], 3)
    
    def test_transaction_rollback(self):
        """β:error_recovery(transaction_rollback_validation) - Test transaction rollback on error."""
        # Create table
        self.db_manager.create_table("test_table", {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "value": "INTEGER"
        })
        
        # Define queries with an error in the second query
        queries = [
            ("INSERT INTO test_table (name, value) VALUES (:name, :value)", {"name": "Item1", "value": 10}),
            ("INSERT INTO invalid_table (name, value) VALUES (:name, :value)", {"name": "Item2", "value": 20})  # This will fail
        ]
        
        # Execute transaction (should fail and rollback)
        with self.assertRaises(MCPDatabaseError):
            self.db_manager.execute_transaction(queries)
        
        # Verify no data was inserted (rollback occurred)
        results = self.db_manager.execute_query("SELECT COUNT(*) as count FROM test_table")
        self.assertEqual(results[0]["count"], 0)
    
    def test_get_table_schema(self):
        """ℵ:schema_introspection(metadata_retrieval_validation) - Test getting table schema."""
        # Create table
        schema = {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "value": "INTEGER"
        }
        self.db_manager.create_table("test_table", schema)
        
        # Get schema
        retrieved_schema = self.db_manager.get_table_schema("test_table")
        
        # Verify schema
        self.assertIn("id", retrieved_schema)
        self.assertIn("name", retrieved_schema)
        self.assertIn("value", retrieved_schema)
    
    def test_backup_restore(self):
        """τ:data_persistence(backup_restore_validation) - Test database backup and restore."""
        # Create table and insert data
        self.db_manager.create_table("test_table", {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL"
        })
        self.db_manager.execute_query(
            "INSERT INTO test_table (name) VALUES (:name)",
            {"name": "Test Data"}
        )
        
        # Create backup
        backup_path = os.path.join(self.temp_dir, "backup.db")
        result = self.db_manager.backup_database(backup_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(backup_path))
        
        # Clear original database
        self.db_manager.drop_table("test_table")
        
        # Restore from backup
        result = self.db_manager.restore_database(backup_path)
        self.assertTrue(result)
        
        # Verify data was restored
        results = self.db_manager.execute_query("SELECT * FROM test_table")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Test Data")


if __name__ == "__main__":
    unittest.main()


# τ:self_reference(test_database_manager_metadata)
{type:TestSuite, file:"test_database_manager.py", version:"1.0.0", checksum:"sha256:db_test_checksum", canonical_address:"test-database-manager", pfsus_compliant:true, lambda_operators:true, test_coverage:"comprehensive", file_format:"pytest.database.v1.0.0.py"}

# Dependencies: @{CORE.MCP.IMPLEMENTATIONS.DATABASE.001, CORE.MCP.EXCEPTIONS.001}
# Related: @{CORE.MCP.INTERFACES.DATABASE.001, CORE.MCP.TESTS.MEMORY.001}

%% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-22T00:00:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=pytest.database.v1.0.0.py; test_methods=8; coverage_areas=connection_lifecycle,schema_management,query_execution,transaction_processing,error_recovery,schema_introspection,data_persistence