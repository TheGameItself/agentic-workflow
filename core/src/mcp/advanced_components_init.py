#!/usr/bin/env python3
"""
Advanced Components Initialization for MCP Core System
Handles initialization of cognitive architecture, creative engine, and learning manager.
"""

import logging
from typing import Optional

def initialize_advanced_components(core_system) -> bool:
    """
    Initialize advanced components for the core system.
    
    Args:
        core_system: The MCPCoreSystem instance
        
    Returns:
        bool: True if initialization successful
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize cognitive architecture
        from .cognitive_architecture import CognitiveArchitecture
        core_system.cognitive_architecture = CognitiveArchitecture()
        
        # Set neural components for cognitive processing
        if core_system.brain_state_integration and core_system.hormone_integration:
            core_system.cognitive_architecture.set_neural_components(
                brain_state=core_system.brain_state_integration,
                hormone_integration=core_system.hormone_integration
            )
        
        logger.info("Cognitive architecture initialized")
        
        # Initialize creative engine
        from .creative_engine import CreativeEngine
        core_system.creative_engine = CreativeEngine()
        
        # Set creativity level based on adaptation hormone
        if core_system.hormone_levels:
            adaptation_level = core_system.hormone_levels.get('adaptation', 0.5)
            core_system.creative_engine.set_creativity_level(adaptation_level)
        
        logger.info("Creative engine initialized")
        
        # Initialize learning manager
        from .learning_manager import LearningManager
        core_system.learning_manager = LearningManager()
        
        # Register learning callbacks
        def on_pattern_discovered(pattern):
            logger.info(f"New pattern discovered: {pattern.pattern}")
            # Add pattern to creative engine's concept pool
            if core_system.creative_engine:
                core_system.creative_engine.add_concept(pattern.pattern)
        
        def on_performance_improved(data):
            logger.info("Learning performance improved")
            # Adjust hormone levels for positive reinforcement
            if core_system.hormone_integration:
                core_system.hormone_integration.provide_feedback('efficiency', 0.1)
        
        def on_learning_phase_changed(data):
            logger.info(f"Learning phase changed to {data['new_phase'].value}")
            # Adjust cognitive state based on learning phase
            if core_system.cognitive_architecture:
                from .cognitive_architecture import CognitiveState
                if data['new_phase'].value == 'exploration':
                    core_system.cognitive_architecture.set_cognitive_state(CognitiveState.CREATIVE)
                elif data['new_phase'].value == 'exploitation':
                    core_system.cognitive_architecture.set_cognitive_state(CognitiveState.FOCUSED)
        
        # Register callbacks
        core_system.learning_manager.register_callback('pattern_discovered', on_pattern_discovered)
        core_system.learning_manager.register_callback('performance_improved', on_performance_improved)
        core_system.learning_manager.register_callback('learning_phase_changed', on_learning_phase_changed)
        
        logger.info("Learning manager initialized with callbacks")
        
        # Connect components for synergy
        _connect_advanced_components(core_system)
        
        logger.info("Advanced components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize advanced components: {e}")
        return False

def _connect_advanced_components(core_system):
    """Connect advanced components for synergistic operation."""
    logger = logging.getLogger(__name__)
    
    try:
        # Connect cognitive architecture with creative engine
        if core_system.cognitive_architecture and core_system.creative_engine:
            # Set creative engine's creativity level based on cognitive state
            def update_creativity_from_cognition():
                context = core_system.cognitive_architecture.get_cognitive_context()
                if context['state'] == 'creative':
                    core_system.creative_engine.set_creativity_level(0.8)
                elif context['state'] == 'focused':
                    core_system.creative_engine.set_creativity_level(0.4)
                else:
                    core_system.creative_engine.set_creativity_level(0.6)
            
            # This would be called periodically in a real implementation
            update_creativity_from_cognition()
        
        # Connect learning manager with perpetual training
        if core_system.learning_manager and core_system.perpetual_training_manager:
            # Add learning experiences to perpetual training data
            def add_learning_to_training(experience):
                if hasattr(experience, 'input_data') and hasattr(experience, 'expected_output'):
                    # Convert learning experience to training data
                    if isinstance(experience.input_data, dict):
                        core_system.perpetual_training_manager.add_metrics_sample(experience.input_data)
        
        logger.info("Advanced components connected successfully")
        
    except Exception as e:
        logger.error(f"Error connecting advanced components: {e}")

def update_advanced_components(core_system, metrics: dict):
    """
    Update advanced components with current system metrics.
    
    Args:
        core_system: The MCPCoreSystem instance
        metrics: Current system metrics
    """
    try:
        # Update cognitive architecture
        if core_system.cognitive_architecture:
            # Update hormone levels in cognitive context
            core_system.cognitive_architecture.update_hormone_levels(core_system.hormone_levels)
            
            # Process cognitive cycle
            core_system.cognitive_architecture.process_cognitive_cycle()
        
        # Update creative engine
        if core_system.creative_engine:
            # Adjust creativity based on system state
            stress_level = core_system.hormone_levels.get('stress', 0.0)
            adaptation_level = core_system.hormone_levels.get('adaptation', 0.5)
            
            # High stress reduces creativity, high adaptation increases it
            creativity_level = max(0.1, min(0.9, adaptation_level - stress_level * 0.3))
            core_system.creative_engine.set_creativity_level(creativity_level)
            
            # Add system concepts to creative pool
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value > 0:
                    concept = f"{key}_{int(value)}"
                    core_system.creative_engine.add_concept(concept, {'metric': key, 'value': value})
        
        # Update learning manager
        if core_system.learning_manager:
            # Add system metrics as learning experience
            core_system.learning_manager.add_experience(
                experience_type=core_system.learning_manager.LearningType.EXPERIENTIAL,
                input_data=metrics,
                reward=1.0 if core_system.hormone_levels.get('efficiency', 0.5) > 0.6 else 0.0
            )
        
        # Update perpetual training with new data
        if core_system.perpetual_training_manager:
            core_system.perpetual_training_manager.add_metrics_sample(metrics)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error updating advanced components: {e}")

def get_advanced_components_status(core_system) -> dict:
    """
    Get status of all advanced components.
    
    Args:
        core_system: The MCPCoreSystem instance
        
    Returns:
        dict: Status information for all advanced components
    """
    status = {
        'cognitive_architecture': None,
        'creative_engine': None,
        'learning_manager': None,
        'perpetual_training_manager': None,
        'model_factory': None
    }
    
    try:
        # Cognitive architecture status
        if core_system.cognitive_architecture:
            status['cognitive_architecture'] = core_system.cognitive_architecture.get_cognitive_context()
        
        # Creative engine status
        if core_system.creative_engine:
            status['creative_engine'] = core_system.creative_engine.get_creative_statistics()
        
        # Learning manager status
        if core_system.learning_manager:
            status['learning_manager'] = core_system.learning_manager.get_learning_statistics()
        
        # Perpetual training status
        if core_system.perpetual_training_manager:
            status['perpetual_training_manager'] = core_system.perpetual_training_manager.get_training_stats()
        
        # Model factory status
        if core_system.model_factory:
            status['model_factory'] = {
                'available_models': core_system.model_factory.get_available_models(),
                'registered_models': core_system.model_factory.registry.list_models()
            }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting advanced components status: {e}")
    
    return status