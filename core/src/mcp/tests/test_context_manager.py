"""
Tests for the BasicContextManager implementation.

βtest_context_manager(functionality_validation)
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json

from core.src.mcp.implementations.context_manager import BasicContextManager
from core.src.mcp.exceptions import MCPContextError


class TestBasicContextManager(unittest.TestCase):
    """Test cases for the BasicContextManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context_manager = BasicContextManager()
    
    def test_create_get_context(self):
        """Test creating and getting a context."""
        # Create context
        initial_data = {"key": "value", "nested": {"subkey": "subvalue"}}
        context_id = self.context_manager.create_context("Test Context", initial_data)
        
        # Verify context was created
        self.assertIsNotNone(context_id)
        
        # Get context
        context = self.context_manager.get_context(context_id)
        
        # Verify context properties
        self.assertEqual(context["name"], "Test Context")
        self.assertEqual(context["data"], initial_data)
    
    def test_update_context_merge(self):
        """Test updating a context with merge."""
        # Create context
        initial_data = {"key1": "value1", "nested": {"subkey1": "subvalue1"}}
        context_id = self.context_manager.create_context("Test Context", initial_data)
        
        # Update context with merge
        update_data = {"key2": "value2", "nested": {"subkey2": "subvalue2"}}
        result = self.context_manager.update_context(context_id, update_data)
        
        # Verify update was successful
        self.assertTrue(result)
        
        # Get updated context
        context = self.context_manager.get_context(context_id)
        
        # Verify merged data
        expected_data = {
            "key1": "value1",
            "key2": "value2",
            "nested": {
                "subkey1": "subvalue1",
                "subkey2": "subvalue2"
            }
        }
        self.assertEqual(context["data"], expected_data)
    
    def test_update_context_replace(self):
        """Test updating a context with replace."""
        # Create context
        initial_data = {"key1": "value1", "nested": {"subkey1": "subvalue1"}}
        context_id = self.context_manager.create_context("Test Context", initial_data)
        
        # Update context with replace
        update_data = {"key2": "value2"}
        result = self.context_manager.update_context(context_id, update_data, merge=False)
        
        # Verify update was successful
        self.assertTrue(result)
        
        # Get updated context
        context = self.context_manager.get_context(context_id)
        
        # Verify replaced data
        expected_data = {"key2": "value2"}
        self.assertEqual(context["data"], expected_data)
    
    def test_delete_context(self):
        """Test deleting a context."""
        # Create context
        context_id = self.context_manager.create_context("Test Context")
        
        # Delete context
        result = self.context_manager.delete_context(context_id)
        
        # Verify deletion was successful
        self.assertTrue(result)
        
        # Verify context no longer exists
        with self.assertRaises(MCPContextError):
            self.context_manager.get_context(context_id)
    
    def test_list_contexts(self):
        """Test listing contexts."""
        # Create contexts
        context1_id = self.context_manager.create_context("Context 1")
        context2_id = self.context_manager.create_context("Context 2")
        
        # List contexts
        contexts = self.context_manager.list_contexts()
        
        # Verify contexts are listed
        self.assertEqual(len(contexts), 2)
        context_names = [c["name"] for c in contexts]
        self.assertIn("Context 1", context_names)
        self.assertIn("Context 2", context_names)
    
    def test_export_import_context(self):
        """Test exporting and importing a context."""
        # Create context
        initial_data = {"key": "value"}
        context_id = self.context_manager.create_context("Test Context", initial_data)
        
        # Export context
        exported = self.context_manager.export_context(context_id)
        
        # Import context
        new_context_id = self.context_manager.import_context(exported)
        
        # Get imported context
        imported_context = self.context_manager.get_context(new_context_id)
        
        # Verify imported context
        self.assertEqual(imported_context["name"], "Test Context")
        self.assertEqual(imported_context["data"], initial_data)
    
    def test_merge_contexts_override(self):
        """Test merging contexts with override strategy."""
        # Create contexts
        context1_id = self.context_manager.create_context("Context 1", {
            "key1": "value1",
            "common": "context1",
            "nested": {"subkey1": "subvalue1"}
        })
        
        context2_id = self.context_manager.create_context("Context 2", {
            "key2": "value2",
            "common": "context2",
            "nested": {"subkey2": "subvalue2"}
        })
        
        # Merge contexts
        merged_id = self.context_manager.merge_contexts([context1_id, context2_id], "override")
        
        # Get merged context
        merged_context = self.context_manager.get_context(merged_id)
        
        # Verify merged data
        expected_data = {
            "key1": "value1",
            "key2": "value2",
            "common": "context2",  # Override from context2
            "nested": {
                "subkey1": "subvalue1",
                "subkey2": "subvalue2"
            }
        }
        self.assertEqual(merged_context["data"], expected_data)
    
    def test_get_context_history(self):
        """Test getting context history."""
        # Create context
        initial_data = {"key": "value1"}
        context_id = self.context_manager.create_context("Test Context", initial_data)
        
        # Update context
        self.context_manager.update_context(context_id, {"key": "value2"})
        
        # Get history
        history = self.context_manager.get_context_history(context_id)
        
        # Verify history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["action"], "create")
        self.assertEqual(history[0]["data"], initial_data)
        self.assertEqual(history[1]["action"], "update")
        self.assertEqual(history[1]["data"], {"key": "value2"})
    
    def test_persistence(self):
        """Test persistence to disk."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create context manager with storage path
            persistent_manager = BasicContextManager(temp_path)
            
            # Create context
            context_id = persistent_manager.create_context("Test Context", {"key": "value"})
            
            # Create a new manager with the same storage path
            new_manager = BasicContextManager(temp_path)
            
            # Verify context was loaded
            context = new_manager.get_context(context_id)
            self.assertEqual(context["name"], "Test Context")
            self.assertEqual(context["data"]["key"], "value")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()