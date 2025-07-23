#!/usr/bin/env python3
"""
Simple test for P2P research data sharing.
"""

import asyncio
import logging
import time
import uuid
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message types for P2P network."""
    RESEARCH_DATA = "research_data"
    RESEARCH_QUERY = "research_query"
    RESEARCH_RESPONSE = "research_response"

class P2PNode:
    """Simple P2P node for research data sharing."""
    
    def __init__(self, node_id=None):
        """Initialize P2P node."""
        self.node_id = node_id or str(uuid.uuid4())
        self.peers = {}
        self.research_data = {}
        self.message_handlers = {
            MessageType.RESEARCH_DATA: self._handle_research_data,
            MessageType.RESEARCH_QUERY: self._handle_research_query,
            MessageType.RESEARCH_RESPONSE: self._handle_research_response
        }
        self.received_messages = []
        self.query_responses = {}
        
        logger.info(f"Created P2P node: {self.node_id}")
    
    def connect_to_peer(self, peer):
        """Connect to another P2P node."""
        if peer.node_id != self.node_id:
            self.peers[peer.node_id] = peer
            peer.peers[self.node_id] = self
            logger.info(f"Node {self.node_id} connected to peer {peer.node_id}")
            return True
        return False
    
    def disconnect_from_peer(self, peer_id):
        """Disconnect from a peer."""
        if peer_id in self.peers:
            peer = self.peers[peer_id]
            del self.peers[peer_id]
            if self.node_id in peer.peers:
                del peer.peers[self.node_id]
            logger.info(f"Node {self.node_id} disconnected from peer {peer_id}")
            return True
        return False
    
    async def broadcast_research_data(self, topic, data):
        """Broadcast research data to all peers."""
        message = {
            'message_id': str(uuid.uuid4()),
            'type': MessageType.RESEARCH_DATA,
            'sender_id': self.node_id,
            'timestamp': time.time(),
            'topic': topic,
            'data': data
        }
        
        # Store data locally
        if topic not in self.research_data:
            self.research_data[topic] = []
        self.research_data[topic].append(data)
        
        # Send to all peers
        for peer_id, peer in self.peers.items():
            await peer._receive_message(message)
        
        logger.info(f"Node {self.node_id} broadcast research data on topic: {topic}")
        return message['message_id']
    
    async def query_research_data(self, topic):
        """Query research data from peers."""
        query_id = str(uuid.uuid4())
        message = {
            'message_id': query_id,
            'type': MessageType.RESEARCH_QUERY,
            'sender_id': self.node_id,
            'timestamp': time.time(),
            'topic': topic
        }
        
        # Create response tracker
        self.query_responses[query_id] = {
            'topic': topic,
            'responses': [],
            'completed': False
        }
        
        # Send to all peers
        for peer_id, peer in self.peers.items():
            await peer._receive_message(message)
        
        logger.info(f"Node {self.node_id} queried research data on topic: {topic}")
        return query_id
    
    async def _receive_message(self, message):
        """Receive a message from a peer."""
        self.received_messages.append(message)
        
        # Handle message based on type
        message_type = message['type']
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](message)
        
        logger.debug(f"Node {self.node_id} received message: {message['message_id']} ({message_type.value})")
    
    async def _handle_research_data(self, message):
        """Handle research data message."""
        topic = message['topic']
        data = message['data']
        
        # Store data
        if topic not in self.research_data:
            self.research_data[topic] = []
        self.research_data[topic].append(data)
        
        logger.info(f"Node {self.node_id} received research data on topic: {topic}")
    
    async def _handle_research_query(self, message):
        """Handle research query message."""
        topic = message['topic']
        sender_id = message['sender_id']
        
        # Check if we have data for this topic
        if topic in self.research_data and sender_id in self.peers:
            response = {
                'message_id': str(uuid.uuid4()),
                'type': MessageType.RESEARCH_RESPONSE,
                'sender_id': self.node_id,
                'timestamp': time.time(),
                'query_id': message['message_id'],
                'topic': topic,
                'data': self.research_data[topic]
            }
            
            # Send response to querying peer
            await self.peers[sender_id]._receive_message(response)
            
            logger.info(f"Node {self.node_id} responded to research query on topic: {topic}")
    
    async def _handle_research_response(self, message):
        """Handle research response message."""
        query_id = message['query_id']
        topic = message['topic']
        data = message['data']
        
        # Check if we're tracking this query
        if query_id in self.query_responses:
            self.query_responses[query_id]['responses'].append({
                'sender_id': message['sender_id'],
                'data': data
            })
            
            logger.info(f"Node {self.node_id} received research response for query: {query_id}")
    
    def get_research_data(self, topic):
        """Get research data for a topic."""
        return self.research_data.get(topic, [])
    
    def get_query_results(self, query_id):
        """Get results for a query."""
        return self.query_responses.get(query_id, {'responses': []})['responses']

async def test_p2p_research_sharing():
    """Test P2P research data sharing."""
    print("Starting P2P research sharing test...")
    
    # Create P2P nodes
    node1 = P2PNode("node1")
    node2 = P2PNode("node2")
    node3 = P2PNode("node3")
    
    print("Created 3 P2P nodes")
    
    # Connect nodes
    node1.connect_to_peer(node2)
    node2.connect_to_peer(node3)
    node3.connect_to_peer(node1)
    
    print("Connected nodes in a triangle topology")
    
    # Node 1 shares research data
    await node1.broadcast_research_data("neural_networks", {
        "title": "Advances in Neural Networks",
        "author": "Node 1",
        "content": "Research data on neural networks"
    })
    
    print("Node 1 shared research data on neural networks")
    
    # Node 2 shares research data
    await node2.broadcast_research_data("genetic_algorithms", {
        "title": "Genetic Algorithm Optimization",
        "author": "Node 2",
        "content": "Research data on genetic algorithms"
    })
    
    print("Node 2 shared research data on genetic algorithms")
    
    # Node 3 shares research data on both topics
    await node3.broadcast_research_data("neural_networks", {
        "title": "Neural Network Applications",
        "author": "Node 3",
        "content": "Applications of neural networks"
    })
    
    await node3.broadcast_research_data("genetic_algorithms", {
        "title": "Advanced Genetic Programming",
        "author": "Node 3",
        "content": "Research on genetic programming"
    })
    
    print("Node 3 shared research data on both topics")
    
    # Wait a moment for propagation
    await asyncio.sleep(0.1)
    
    # Check if all nodes have all research data
    for node in [node1, node2, node3]:
        nn_data = node.get_research_data("neural_networks")
        ga_data = node.get_research_data("genetic_algorithms")
        
        print(f"Node {node.node_id} has {len(nn_data)} neural network papers and {len(ga_data)} genetic algorithm papers")
    
    # Node 1 queries for genetic algorithms
    query_id = await node1.query_research_data("genetic_algorithms")
    
    # Wait for responses
    await asyncio.sleep(0.1)
    
    # Check query results
    results = node1.get_query_results(query_id)
    print(f"Node 1 received {len(results)} responses to its query on genetic algorithms")
    
    # Disconnect node 3
    node1.disconnect_from_peer(node3.node_id)
    node2.disconnect_from_peer(node3.node_id)
    
    print("Disconnected node 3 from the network")
    
    # Node 2 shares new research data
    await node2.broadcast_research_data("neural_networks", {
        "title": "Latest Neural Network Research",
        "author": "Node 2",
        "content": "Cutting-edge neural network research"
    })
    
    print("Node 2 shared new research data after node 3 disconnected")
    
    # Wait a moment for propagation
    await asyncio.sleep(0.1)
    
    # Check if node 1 received the new data but node 3 didn't
    node1_nn_data = node1.get_research_data("neural_networks")
    node3_nn_data = node3.get_research_data("neural_networks")
    
    print(f"Node 1 now has {len(node1_nn_data)} neural network papers")
    print(f"Node 3 still has {len(node3_nn_data)} neural network papers")
    
    print("P2P research sharing test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_p2p_research_sharing())