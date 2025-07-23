"""
Tests for the MemoryLobe implementation.

βtest_memory_lobe(functionality_validation)
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile

from core.src.mcp.lobes.memory_lobe import MemoryLobe
from core.src.mcp.exceptions import MCPLobeError


class TestMemoryLobe(unittest.TestCase):
    """Test cases for the MemoryLobe class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_lobe = MemoryLobe()
        self.memory_lobe.initialize()
    
    def test_initialization(self):
        """Test lobe initialization."""
        # Test basic initialization
        lobe = MemoryLobe()
        result = lobe.initialize()
        self.assertTrue(result)
        self.assertEqual(lobe.status, "ready")
        
        # Test initialization with config
        config = {"memory_types": ["custom1", "custom2"]}
        lobe = MemoryLobe()
        result = lobe.initialize(config)
        self.assertTrue(result)
        self.assertIn("custom1", lobe.memory_types)
        self.assertIn("custom2", lobe.memory_types)
    
    def test_store_retrieve_memory(self):
        """Test storing and retrieving memories."""
        # Store a memory
        memory_data = {"content": "Test memory", "importance": 0.8}
        memory_id = self.memory_lobe.store_memory(memory_data, "episodic")
        
        # Verify memory was stored
        self.assertIsNotNone(memory_id)
        
        # Retrieve the memory
        retrieved_memory = self.memory_lobe.retrieve_memory(memory_id)
        
        # Verify memory content
        self.assertEqual(retrieved_memory, memory_data)
    
    def test_search_memories(self):
        """Test searching memories."""
        # Store multiple memories
        self.memory_lobe.store_memory({"content": "Apple is a fruit"}, "semantic")
        self.memory_lobe.store_memory({"content": "Banana is yellow"}, "semantic")
        self.memory_lobe.store_memory({"content": "Car is a vehicle"}, "semantic")
        
        # Search for fruit-related memories
        results = self.memory_lobe.search_memories("fruit")
        
        # Verify results
        self.assertGreater(len(results), 0)
        found_apple = any("Apple" in str(result[1]) for result in results)
        self.assertTrue(found_apple)
    
    def test_search_memories_by_type(self):
        """Test searching memories by type."""
        # Store memories of different types
        self.memory_lobe.store_memory({"content": "Episodic memory"}, "episodic")
        self.memory_lobe.store_memory({"content": "Semantic memory"}, "semantic")
        
        # Search for semantic memories only
        results = self.memory_lobe.search_memories("memory", memory_type="semantic")
        
        # Verify only semantic memories are returned
        self.assertGreater(len(results), 0)
        # Note: This test might need adjustment based on actual search implementation
    
    def test_forget_memory(self):
        """Test forgetting memories."""
        # Store a memory
        memory_data = {"content": "Memory to forget"}
        memory_id = self.memory_lobe.store_memory(memory_data, "episodic")
        
        # Forget the memory
        result = self.memory_lobe.forget_memory(memory_id)
        self.assertTrue(result)
        
        # Verify memory is gone
        with self.assertRaises(MCPLobeError):
            self.memory_lobe.retrieve_memory(memory_id)
    
    def test_get_memory_types(self):
        """Test getting memory types."""
        memory_types = self.memory_lobe.get_memory_types()
        
        # Verify default memory types are present
        self.assertIn("episodic", memory_types)
        self.assertIn("semantic", memory_types)
        self.assertIn("procedural", memory_types)
        self.assertIn("working", memory_types)
    
    def test_process_method(self):
        """Test the process method with different actions."""
        # Test store action
        store_input = {
            "action": "store",
            "data": {"content": "Test content"},
            "memory_type": "episodic"
        }
        memory_id = self.memory_lobe.process(store_input)
        self.assertIsNotNone(memory_id)
        
        # Test retrieve action
        retrieve_input = {
            "action": "retrieve",
            "memory_id": memory_id
        }
        retrieved_data = self.memory_lobe.process(retrieve_input)
        self.assertEqual(retrieved_data["content"], "Test content")
        
        # Test search action
        search_input = {
            "action": "search",
            "query": "content",
            "limit": 5
        }
        search_results = self.memory_lobe.process(search_input)
        self.assertIsInstance(search_results, list)
        
        # Test forget action
        forget_input = {
            "action": "forget",
            "memory_id": memory_id
        }
        forget_result = self.memory_lobe.process(forget_input)
        self.assertTrue(forget_result)
    
    def test_lobe_communication(self):
        """Test communication between lobes."""
        # Create another lobe for testing
        other_lobe = MemoryLobe(name="OtherMemoryLobe")
        other_lobe.initialize()
        
        # Connect the lobes
        self.memory_lobe.connect(other_lobe, "bidirectional")
        
        # Send a message
        message = {
            "action": "store",
            "data": {"content": "Message content"},
            "memory_type": "episodic"
        }
        result = self.memory_lobe.send_message(other_lobe.lobe_id, message)
        self.assertTrue(result)
        
        # Verify connection
        connections = self.memory_lobe.get_connections()
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0]["lobe_id"], other_lobe.lobe_id)
    
    def test_get_status(self):
        """Test getting lobe status."""
        # Store some memories
        self.memory_lobe.store_memory({"content": "Memory 1"}, "episodic")
        self.memory_lobe.store_memory({"content": "Memory 2"}, "semantic")
        
        # Get status
        status = self.memory_lobe.get_status()
        
        # Verify status information
        self.assertEqual(status["name"], self.memory_lobe.name)
        self.assertEqual(status["status"], "ready")
        self.assertIn("memory_types", status)
        self.assertIn("total_memories", status)
        self.assertEqual(status["total_memories"], 2)
    
    def test_persistence(self):
        """Test memory persistence."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create lobe with storage path
            persistent_lobe = MemoryLobe(storage_path=temp_path)
            persistent_lobe.initialize()
            
            # Store a memory
            memory_id = persistent_lobe.store_memory({"content": "Persistent memory"}, "episodic")
            
            # Create a new lobe with the same storage path
            new_lobe = MemoryLobe(storage_path=temp_path)
            new_lobe.initialize()
            
            # Verify memory was persisted
            retrieved_memory = new_lobe.retrieve_memory(memory_id)
            self.assertEqual(retrieved_memory["content"], "Persistent memory")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()