"""
Unit tests for the Neural Network Models component.

This module contains tests for the ModelManager and ModelRegistry classes
to ensure proper functionality of the model management infrastructure.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.mcp.neural_network_models import ModelManager, ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """Test cases for the ModelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()

    def test_register_function(self):
        """Test registering a function with the registry."""
        # Register a function
        result = self.registry.register_function("test_function", ["mlp", "lstm"], "mlp")
        self.assertTrue(result)
        self.assertTrue(self.registry.is_registered("test_function"))
        self.assertEqual(self.registry.get_model_types("test_function"), ["mlp", "lstm"])
        self.assertEqual(self.registry.get_default_model("test_function"), "mlp")

        # Try to register the same function again
        result = self.registry.register_function("test_function")
        self.assertFalse(result)

    def test_register_with_default_values(self):
        """Test registering a function with default values."""
        result = self.registry.register_function("simple_function")
        self.assertTrue(result)
        self.assertEqual(self.registry.get_model_types("simple_function"), ["default"])
        self.assertEqual(self.registry.get_default_model("simple_function"), "default")

    def test_register_with_invalid_default(self):
        """Test registering a function with an invalid default model type."""
        result = self.registry.register_function("invalid_function", ["mlp", "lstm"], "cnn")
        self.assertFalse(result)
        self.assertFalse(self.registry.is_registered("invalid_function"))

    def test_register_model_dependency(self):
        """Test registering dependencies between functions."""
        # Register two functions
        self.registry.register_function("parent_function")
        self.registry.register_function("child_function")

        # Register dependency
        result = self.registry.register_model_dependency("child_function", "parent_function")
        self.assertTrue(result)
        self.assertEqual(self.registry.get_dependencies("child_function"), ["parent_function"])

        # Try to register dependency for unregistered function
        result = self.registry.register_model_dependency("unknown_function", "parent_function")
        self.assertFalse(result)

        # Try to register unregistered dependency
        result = self.registry.register_model_dependency("child_function", "unknown_function")
        self.assertFalse(result)

    def test_set_default_model(self):
        """Test setting the default model type for a function."""
        # Register a function with multiple model types
        self.registry.register_function("multi_model", ["mlp", "lstm", "cnn"], "mlp")

        # Change default model
        result = self.registry.set_default_model("multi_model", "lstm")
        self.assertTrue(result)
        self.assertEqual(self.registry.get_default_model("multi_model"), "lstm")

        # Try to set invalid model type as default
        result = self.registry.set_default_model("multi_model", "transformer")
        self.assertFalse(result)
        self.assertEqual(self.registry.get_default_model("multi_model"), "lstm")

        # Try to set default for unregistered function
        result = self.registry.set_default_model("unknown_function", "mlp")
        self.assertFalse(result)

    def test_get_all_registered_functions(self):
        """Test getting all registered functions."""
        # Register multiple functions
        self.registry.register_function("function1")
        self.registry.register_function("function2")
        self.registry.register_function("function3")

        # Get all registered functions
        functions = self.registry.get_all_registered_functions()
        self.assertEqual(len(functions), 3)
        self.assertIn("function1", functions)
        self.assertIn("function2", functions)
        self.assertIn("function3", functions)


class TestModelManager(unittest.TestCase):
    """Test cases for the ModelManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for models
        self.test_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(models_dir=self.test_dir)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test initialization of the ModelManager."""
        # Check that the models directory was created
        self.assertTrue(os.path.exists(self.test_dir))

        # Check that subdirectories were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "hormone")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "diffusion")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cascade")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "receptor")))

        # Check that the registry was initialized
        self.assertIsNotNone(self.model_manager.registry)
        self.assertTrue(self.model_manager.registry.is_registered("dopamine_production"))
        self.assertTrue(self.model_manager.registry.is_registered("spatial_diffusion"))

    def test_create_new_model(self):
        """Test creating a new model."""
        # Create a new model
        result = self.model_manager.create_new_model("test_model", "mlp")
        self.assertTrue(result)
        self.assertIn("test_model", self.model_manager.models)
        self.assertIn("test_model", self.model_manager.model_metadata)

        # Check that model files were created
        model_path = os.path.join(self.test_dir, "test_model.model")
        metadata_path = os.path.join(self.test_dir, "test_model.metadata")
        self.assertTrue(os.path.exists(model_path) or os.path.exists(os.path.join(self.test_dir, "hormone", "test_model.model")))

        # Try to create a model that already exists
        result = self.model_manager.create_new_model("test_model", "lstm")
        self.assertFalse(result)

    def test_get_model(self):
        """Test getting a model."""
        # Create a new model
        self.model_manager.create_new_model("get_test_model", "mlp")

        # Get the model
        model = self.model_manager.get_model("get_test_model")
        self.assertIsNotNone(model)
        self.assertEqual(model["type"], "mlp")
        self.assertEqual(model["function"], "get_test_model")

        # Try to get a non-existent model
        model = self.model_manager.get_model("nonexistent_model")
        self.assertIsNone(model)

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # Create a new model
        self.model_manager.create_new_model("save_load_model", "lstm")

        # Get the original model
        original_model = self.model_manager.models["save_load_model"]

        # Remove the model from memory
        del self.model_manager.models["save_load_model"]

        # Load the model
        result = self.model_manager.load_model("save_load_model")
        self.assertTrue(result)
        self.assertIn("save_load_model", self.model_manager.models)

        # Check that the loaded model has the same type
        loaded_model = self.model_manager.models["save_load_model"]
        self.assertEqual(loaded_model["type"], original_model["type"])
        self.assertEqual(loaded_model["function"], original_model["function"])

    def test_update_performance_metrics(self):
        """Test updating performance metrics for a model."""
        # Create a new model
        self.model_manager.create_new_model("metrics_model", "mlp")

        # Update metrics
        metrics = {"accuracy": 0.95, "latency": 10.5, "resource_usage": 0.3}
        self.model_manager.update_performance_metrics("metrics_model", metrics)

        # Get metrics
        updated_metrics = self.model_manager.get_performance_metrics("metrics_model")
        self.assertEqual(updated_metrics["accuracy"], 0.95)
        self.assertEqual(updated_metrics["latency"], 10.5)
        self.assertEqual(updated_metrics["resource_usage"], 0.3)

        # Check that overall score was calculated
        self.assertIn("overall_score", updated_metrics)

    def test_delete_model(self):
        """Test deleting a model."""
        # Create a new model
        self.model_manager.create_new_model("delete_model", "mlp")

        # Delete the model
        result = self.model_manager.delete_model("delete_model")
        self.assertTrue(result)
        self.assertNotIn("delete_model", self.model_manager.models)
        self.assertNotIn("delete_model", self.model_manager.model_metadata)

        # Try to delete a non-existent model
        result = self.model_manager.delete_model("nonexistent_model")
        self.assertFalse(result)

    def test_get_model_info(self):
        """Test getting model information."""
        # Create a new model for a registered function
        self.model_manager.registry.register_function("info_function", ["mlp", "lstm"], "mlp")
        self.model_manager.create_new_model("info_function", "mlp")

        # Get model info
        info = self.model_manager.get_model_info("info_function")
        self.assertEqual(info["function_name"], "info_function")
        self.assertEqual(info["model_type"], "mlp")
        self.assertTrue(info["registered"])
        self.assertEqual(info["valid_model_types"], ["mlp", "lstm"])
        self.assertEqual(info["default_model_type"], "mlp")
        self.assertTrue(info["loaded"])

        # Get info for non-existent model
        info = self.model_manager.get_model_info("nonexistent_model")
        self.assertEqual(info, {})

    def test_compare_implementations(self):
        """Test comparing neural and algorithmic implementations."""
        # Create a new model
        self.model_manager.create_new_model("compare_model", "mlp")

        # Update metrics
        metrics = {"accuracy": 0.95, "latency": 10.5, "resource_usage": 0.3}
        self.model_manager.update_performance_metrics("compare_model", metrics)

        # Compare implementations
        comparison = self.model_manager.compare_implementations("compare_model")
        self.assertEqual(comparison["function_name"], "compare_model")
        self.assertIn("neural", comparison["implementations"])
        self.assertIn("algorithmic", comparison["implementations"])
        self.assertIn("recommendation", comparison)

    def test_export_and_import_models(self):
        """Test exporting and importing models."""
        # Create a new model
        self.model_manager.create_new_model("export_model", "mlp")

        # Create a temporary export directory
        export_dir = tempfile.mkdtemp()

        try:
            # Export models
            result = self.model_manager.export_models(export_dir)
            self.assertTrue(result)

            # Check that files were exported
            exported_files = os.listdir(export_dir)
            self.assertTrue(any(f.startswith("export_model") for f in exported_files))

            # Create a new model manager
            new_manager = ModelManager(models_dir=tempfile.mkdtemp())

            # Import models
            result = new_manager.import_models(export_dir)
            self.assertTrue(result)

            # Check that the model was imported
            self.assertIn("export_model", new_manager.model_metadata)

            # Clean up the new manager's directory
            shutil.rmtree(new_manager.models_dir)
        finally:
            # Clean up the export directory
            shutil.rmtree(export_dir)

    def test_update_model_version(self):
        """Test updating model version."""
        # Create a new model
        self.model_manager.create_new_model("version_model", "mlp")

        # Update version
        result = self.model_manager.update_model_version("version_model", "2.0.0")
        self.assertTrue(result)

        # Check that version was updated
        version = self.model_manager.get_model_version("version_model")
        self.assertEqual(version, "2.0.0")

        # Try to update version for non-existent model
        result = self.model_manager.update_model_version("nonexistent_model", "2.0.0")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()