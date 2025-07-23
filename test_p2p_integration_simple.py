#!/usr/bin/env python3
"""
Simple test for P2P integration with cognitive architecture.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import core components
from core.src.mcp.cognitive_architecture import CognitiveArchitecture
from core.src.mcp.p2p_core_integration import P2PCoreIntegration
from core.src.mcp.p2p_network_bus import MessageBusEvent, MessagePriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_p2p_integration():
    """Test P2P integration with cognitive architecture."""
    logger.info("Starting P2P integration test")
    
    # Create cognitive architecture
    cognitive_architecture = CognitiveArchitecture()
    
    # Initialize cognitive architecture with P2P
    config = {
        'enable_p2p': True,
        'p2p_config': {
            'bootstrap_nodes': [('127.0.0.1', 8000)]
        }
    }
    
    # Initialize cognitive architecture
    success = await cognitive_architecture.initialize(config)
    logger.info(f"Cognitive architecture initialization: {'Success' if success else 'Failed'}")
    
    if success and cognitive_architecture.p2p_integration:
        # Add a thought to cognitive architecture
        thought_id = cognitive_architecture.add_thought(
            content="Test thought",
            priority=0.8,
            source="test",
            metadata={"test": True}
        )
        logger.info(f"Added thought: {thought_id}")
        
        # Get the thought
        thought = cognitive_architecture.get_thought(thought_id)
        logger.info(f"Retrieved thought: {thought.content}")
        
        # Share thought via P2P
        try:
            message_id = await cognitive_architecture.p2p_integration.network_bus.publish(
                event_type=MessageBusEvent.BROADCAST,
                content={
                    "thought_id": thought_id,
                    "thought_content": thought.content,
                    "thought_priority": thought.priority,
                    "thought_source": thought.source,
                    "thought_metadata": thought.metadata
                },
                priority=MessagePriority.NORMAL
            )
            logger.info(f"Published thought with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error publishing thought: {e}")
    
    # Shutdown cognitive architecture
    shutdown_success = await cognitive_architecture.shutdown()
    logger.info(f"Cognitive architecture shutdown: {'Success' if shutdown_success else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(test_p2p_integration())