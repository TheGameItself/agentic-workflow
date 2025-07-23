"""
Tests for the BasicMemoryManager implementation.

βtest_memory_manager(functionality_validation)
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json

from core.src.mcp.implementations.memory_manager import BasicMemoryManager
from core.src.mcp.exceptions import MCPMemoryError


class TestBasicMemoryManager(unittest.TestCase):
    """Test cases for the BasicMemoryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = BasicMemoryManager()
    
    def test_store_retrieve(self):
        """Test storing and retrieving data."""
        # Store data
        test_data = {"name": "Test", "value": 42}
        key = self.memory_manager.store(test_data)
        
        # Verify key is returned
        self.assertIsNotNone(key)
        
        # Retrieve data
        retrieved_data = self.memory_manager.retrieve(key)
        
        # Verify data is correct
        self.assertEqual(retrieved_data, test_data)
    
    def test_store_with_key(self):
        """Test storing data with a specified key."""
        test_data = {"name": "Test", "value": 42}
        key = "test-key"
        
        # Store data with key
        returned_key = self.memory_manager.store(test_data, key)
        
        # Verify key is returned correctly
        self.assertEqual(returned_key, key)
        
        # Retrieve data
        retrieved_data = self.memory_manager.retrieve(key)
        
        # Verify data is correct
        self.assertEqual(retrieved_data, test_data)
    
    def test_update(self):
        """Test updating data."""
        # Store initial data
        test_data = {"name": "Test", "value": 42}
        key = self.memory_manager.store(test_data)
        
        # Update data
        updated_data = {"name": "Updated", "value": 100}
        result = self.memory_manager.update(key, updated_data)
        
        # Verify update was successful
        self.assertTrue(result)
        
        # Retrieve updated data
        retrieved_data = self.memory_manager.retrieve(key)
        
        # Verify data is updated
        self.assertEqual(retrieved_data, updated_data)
    
    def test_delete(self):
        """Test deleting data."""
        # Store data
        test_data = {"name": "Test", "value": 42}
        key = self.memory_manager.store(test_data)
        
        # Delete data
        result = self.memory_manager.delete(key)
        
        # Verify deletion was successful
        self.assertTrue(result)
        
        # Verify key no longer exists
        with self.assertRaises(MCPMemoryError):
            self.memory_manager.retrieve(key)
    
    def test_search(self):
        """Test searching for data."""
        # Store multiple items
        self.memory_manager.store({"name": "Apple", "type": "fruit"}, "apple")
        self.memory_manager.store({"name": "Banana", "type": "fruit"}, "banana")
        self.memory_manager.store({"name": "Carrot", "type": "vegetable"}, "carrot")
        
        # Search for fruits
        results = self.memory_manager.search("fruit")
        
        # Verify results
        self.assertEqual(len(results), 2)
        keys = [result[0] for result in results]
        self.assertIn("apple", keys)
        self.assertIn("banana", keys)
    
    def test_list_keys(self):
        """Test listing keys."""
        # Store multiple items
        self.memory_manager.store({"name": "Apple"}, "apple")
        self.memory_manager.store({"name": "Banana"}, "banana")
        self.memory_manager.store({"name": "Carrot"}, "carrot")
        
        # List all keys
        keys = self.memory_manager.list_keys()
        
        # Verify keys
        self.assertEqual(len(keys), 3)
        self.assertIn("apple", keys)
        self.assertIn("banana", keys)
        self.assertIn("carrot", keys)
        
        # List keys with pattern
        filtered_keys = self.memory_manager.list_keys("a")
        
        # Verify filtered keys
        self.assertEqual(len(filtered_keys), 2)
        self.assertIn("apple", filtered_keys)
        self.assertIn("banana", filtered_keys)
    
    def test_clear(self):
        """Test clearing all data."""
        # Store multiple items
        self.memory_manager.store({"name": "Apple"}, "apple")
        self.memory_manager.store({"name": "Banana"}, "banana")
        
        # Clear all data
        result = self.memory_manager.clear()
        
        # Verify clear was successful
        self.assertTrue(result)
        
        # Verify no keys remain
        keys = self.memory_manager.list_keys()
        self.assertEqual(len(keys), 0)
    
    def test_get_metadata(self):
        """Test getting metadata."""
        # Store data with metadata
        test_data = {"name": "Test"}
        test_metadata = {"category": "test", "priority": "high"}
        key = self.memory_manager.store(test_data, metadata=test_metadata)
        
        # Get metadata
        metadata = self.memory_manager.get_metadata(key)
        
        # Verify metadata contains our values
        self.assertEqual(metadata["category"], "test")
        self.assertEqual(metadata["priority"], "high")
        
        # Verify metadata contains system values
        self.assertIn("created_at", metadata)
        self.assertIn("updated_at", metadata)
    
    def test_persistence(self):
        """Test persistence to disk."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create memory manager with storage path
            persistent_manager = BasicMemoryManager(temp_path)
            
            # Store data
            persistent_manager.store({"name": "Test"}, "test")
            
            # Create a new manager with the same storage path
            new_manager = BasicMemoryManager(temp_path)
            
            # Verify data was loaded
            retrieved_data = new_manager.retrieve("test")
            self.assertEqual(retrieved_data["name"], "Test")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()