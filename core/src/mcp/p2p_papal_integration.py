"""
P2P Papal Election Integration

This module integrates the papal election chain of command system with the existing P2P network,
enabling hierarchical decision-making across distributed agents.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import asdict

from .papal_election_chain_of_command import (
    PapalElectionChainOfCommand,
    CardinalRole,
    SmokeSignal,
    Cardinal,
    ConclaveSession
)
from .p2p_network import P2PNetwork
from .genetic_data_exchange import GeneticDataExchange


class P2PPapalIntegration:
    """
    Integration layer between P2P network and papal election chain of command.
    
    Features:
    - Distributed cardinal registration
    - Cross-network conclave coordination
    - Smoke signal broadcasting
    - Genetic data exchange for performance updates
    - Hierarchical decision-making across network
    """
    
    def __init__(self, organism_id: str, port: int = 0):
        self.organism_id = organism_id
        self.papal_chain = PapalElectionChainOfCommand()
        self.p2p_network = P2PNetwork(organism_id, port)
        self.genetic_exchange = GeneticDataExchange()
        
        self.logger = logging.getLogger("P2PPapalIntegration")
        self.network_cardinals: Dict[str, Dict[str, Any]] = {}
        self.active_network_conclaves: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking for network agents
        self.network_performance: Dict[str, List[float]] = {}
        self.network_reputation: Dict[str, float] = {}
        
        self.logger.info(f"P2P Papal Integration initialized for {organism_id}")
    
    async def start(self):
        """Start the P2P papal integration system"""
        await self.p2p_network.start()
        
        # Register this node as a cardinal
        await self._register_self_as_cardinal()
        
        # Start background tasks
        asyncio.create_task(self._broadcast_status_periodic())
        asyncio.create_task(self._listen_for_network_updates())
        
        self.logger.info("P2P Papal Integration started")
    
    async def stop(self):
        """Stop the P2P papal integration system"""
        await self.p2p_network.stop()
        self.logger.info("P2P Papal Integration stopped")
    
    async def _register_self_as_cardinal(self):
        """Register this node as a cardinal in the network"""
        # Get local performance metrics
        performance_score = await self._get_local_performance()
        reputation = self._get_local_reputation()
        specialization = self._get_local_specialization()
        
        cardinal = await self.papal_chain.register_agent(
            self.organism_id,
            f"Node_{self.organism_id}",
            performance_score,
            specialization,
            reputation
        )
        
        # Broadcast cardinal registration to network
        await self._broadcast_cardinal_registration(cardinal)
        
        self.logger.info(f"Registered self as cardinal: {cardinal.role.value}")
    
    async def _get_local_performance(self) -> float:
        """Get local performance score"""
        # This would integrate with local performance monitoring
        # For now, return a default score
        return 0.75
    
    def _get_local_reputation(self) -> float:
        """Get local reputation score"""
        # This would integrate with reputation tracking
        # For now, return a default score
        return 0.70
    
    def _get_local_specialization(self) -> str:
        """Get local specialization"""
        # This would be determined by local capabilities
        return "general_ai"
    
    async def _broadcast_cardinal_registration(self, cardinal: Cardinal):
        """Broadcast cardinal registration to the network"""
        message = {
            "type": "cardinal_registration",
            "cardinal_data": asdict(cardinal),
            "timestamp": datetime.now().isoformat(),
            "source_id": self.organism_id
        }
        
        await self.p2p_network.broadcast_message(json.dumps(message))
    
    async def register_network_cardinal(self, cardinal_data: Dict[str, Any]):
        """Register a cardinal from the network"""
        cardinal_id = cardinal_data["id"]
        
        # Store network cardinal data
        self.network_cardinals[cardinal_id] = cardinal_data
        
        # Register in local papal chain if not already present
        if cardinal_id not in self.papal_chain.college:
            cardinal = await self.papal_chain.register_agent(
                cardinal_data["id"],
                cardinal_data["name"],
                cardinal_data["performance_score"],
                cardinal_data["specialization"],
                cardinal_data["reputation"]
            )
            
            self.logger.info(f"Registered network cardinal: {cardinal.name} ({cardinal.role.value})")
    
    async def initiate_network_conclave(self, topic: str, required_roles: List[CardinalRole] = None) -> str:
        """Initiate a conclave across the entire network"""
        # Create local conclave
        session_id = await self.papal_chain.initiate_conclave(topic, required_roles)
        
        # Broadcast conclave initiation
        message = {
            "type": "conclave_initiation",
            "session_id": session_id,
            "topic": topic,
            "required_roles": [role.value for role in required_roles] if required_roles else None,
            "timestamp": datetime.now().isoformat(),
            "source_id": self.organism_id
        }
        
        await self.p2p_network.broadcast_message(json.dumps(message))
        
        # Track network conclave
        self.active_network_conclaves[session_id] = {
            "session_id": session_id,
            "topic": topic,
            "network_votes": {},
            "consensus_reached": False,
            "smoke_signal": SmokeSignal.NO_SMOKE.value
        }
        
        self.logger.info(f"Initiated network conclave: {session_id} - {topic}")
        return session_id
    
    async def cast_network_vote(self, session_id: str, vote: Any) -> bool:
        """Cast a vote in a network conclave"""
        # Cast local vote
        success = await self.papal_chain.cast_vote(self.organism_id, vote)
        if not success:
            return False
        
        # Broadcast vote to network
        message = {
            "type": "conclave_vote",
            "session_id": session_id,
            "vote": vote,
            "timestamp": datetime.now().isoformat(),
            "source_id": self.organism_id
        }
        
        await self.p2p_network.broadcast_message(json.dumps(message))
        
        # Update network conclave tracking
        if session_id in self.active_network_conclaves:
            self.active_network_conclaves[session_id]["network_votes"][self.organism_id] = vote
        
        self.logger.info(f"Cast network vote in {session_id}: {vote}")
        return True
    
    async def _broadcast_status_periodic(self):
        """Periodically broadcast status updates"""
        while True:
            try:
                # Get current status
                status = await self.papal_chain.get_network_status()
                
                # Add network-specific information
                network_status = {
                    "type": "status_update",
                    "papal_status": status,
                    "network_cardinals_count": len(self.network_cardinals),
                    "active_network_conclaves": len(self.active_network_conclaves),
                    "timestamp": datetime.now().isoformat(),
                    "source_id": self.organism_id
                }
                
                await self.p2p_network.broadcast_message(json.dumps(network_status))
                
                # Wait before next broadcast
                await asyncio.sleep(30)  # Broadcast every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in status broadcast: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _listen_for_network_updates(self):
        """Listen for network updates and process them"""
        while True:
            try:
                # Get messages from network
                messages = await self.p2p_network.get_messages()
                
                for message in messages:
                    await self._process_network_message(message)
                
                await asyncio.sleep(1)  # Check for messages every second
                
            except Exception as e:
                self.logger.error(f"Error in network message processing: {e}")
                await asyncio.sleep(5)
    
    async def _process_network_message(self, message: str):
        """Process a message from the network"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "cardinal_registration":
                await self.register_network_cardinal(data["cardinal_data"])
                
            elif message_type == "conclave_initiation":
                await self._handle_conclave_initiation(data)
                
            elif message_type == "conclave_vote":
                await self._handle_conclave_vote(data)
                
            elif message_type == "status_update":
                await self._handle_status_update(data)
                
            elif message_type == "performance_update":
                await self._handle_performance_update(data)
                
        except json.JSONDecodeError:
            self.logger.warning("Received invalid JSON message")
        except Exception as e:
            self.logger.error(f"Error processing network message: {e}")
    
    async def _handle_conclave_initiation(self, data: Dict[str, Any]):
        """Handle conclave initiation from network"""
        session_id = data["session_id"]
        topic = data["topic"]
        required_roles = data.get("required_roles")
        
        # Convert role strings back to enums
        role_enums = None
        if required_roles:
            role_enums = [CardinalRole(role) for role in required_roles]
        
        # Create local conclave if not already present
        if not self.papal_chain.active_conclave or self.papal_chain.active_conclave.session_id != session_id:
            await self.papal_chain.initiate_conclave(topic, role_enums)
            
            # Track network conclave
            self.active_network_conclaves[session_id] = {
                "session_id": session_id,
                "topic": topic,
                "network_votes": {},
                "consensus_reached": False,
                "smoke_signal": SmokeSignal.NO_SMOKE.value
            }
        
        self.logger.info(f"Received conclave initiation: {session_id} - {topic}")
    
    async def _handle_conclave_vote(self, data: Dict[str, Any]):
        """Handle conclave vote from network"""
        session_id = data["session_id"]
        vote = data["vote"]
        source_id = data["source_id"]
        
        # Update network conclave tracking
        if session_id in self.active_network_conclaves:
            self.active_network_conclaves[session_id]["network_votes"][source_id] = vote
        
        # Check for network consensus
        await self._check_network_consensus(session_id)
        
        self.logger.info(f"Received vote from {source_id} in {session_id}: {vote}")
    
    async def _handle_status_update(self, data: Dict[str, Any]):
        """Handle status update from network"""
        source_id = data["source_id"]
        papal_status = data["papal_status"]
        
        # Update network cardinal information
        if source_id in papal_status.get("college_stats", {}):
            self.network_cardinals[source_id] = papal_status
        
        self.logger.debug(f"Received status update from {source_id}")
    
    async def _handle_performance_update(self, data: Dict[str, Any]):
        """Handle performance update from network"""
        source_id = data["source_id"]
        performance_score = data["performance_score"]
        reputation = data.get("reputation")
        
        # Update performance tracking
        if source_id not in self.network_performance:
            self.network_performance[source_id] = []
        self.network_performance[source_id].append(performance_score)
        
        if reputation is not None:
            self.network_reputation[source_id] = reputation
        
        # Update cardinal performance if registered
        if source_id in self.papal_chain.college:
            await self.papal_chain.update_agent_performance(source_id, performance_score, reputation)
        
        self.logger.info(f"Updated performance for {source_id}: {performance_score}")
    
    async def _check_network_consensus(self, session_id: str):
        """Check for consensus across the network"""
        if session_id not in self.active_network_conclaves:
            return
        
        conclave_data = self.active_network_conclaves[session_id]
        network_votes = conclave_data["network_votes"]
        
        if len(network_votes) < 2:  # Need at least 2 votes
            return
        
        # Calculate network consensus
        total_votes = len(network_votes)
        positive_votes = sum(1 for vote in network_votes.values() if vote)
        consensus_ratio = positive_votes / total_votes
        
        if consensus_ratio >= self.papal_chain.consensus_threshold:
            # Network consensus reached
            conclave_data["consensus_reached"] = True
            conclave_data["smoke_signal"] = SmokeSignal.WHITE_SMOKE.value
            
            # Broadcast consensus
            message = {
                "type": "network_consensus",
                "session_id": session_id,
                "consensus_ratio": consensus_ratio,
                "timestamp": datetime.now().isoformat(),
                "source_id": self.organism_id
            }
            
            await self.p2p_network.broadcast_message(json.dumps(message))
            
            self.logger.info(f"Network consensus reached in {session_id}: {consensus_ratio:.2f}")
    
    async def get_network_hierarchy(self) -> Dict[str, Any]:
        """Get the complete network hierarchy including all nodes"""
        local_hierarchy = self.papal_chain.get_chain_of_command()
        
        # Add network information
        network_hierarchy = {
            "local_hierarchy": local_hierarchy,
            "network_cardinals": self.network_cardinals,
            "active_network_conclaves": self.active_network_conclaves,
            "network_stats": {
                "total_network_nodes": len(self.network_cardinals) + 1,  # +1 for self
                "active_conclaves": len(self.active_network_conclaves),
                "network_performance": self.network_performance,
                "network_reputation": self.network_reputation
            }
        }
        
        return network_hierarchy
    
    async def broadcast_performance_update(self, performance_score: float, reputation: float = None):
        """Broadcast performance update to the network"""
        message = {
            "type": "performance_update",
            "performance_score": performance_score,
            "reputation": reputation,
            "timestamp": datetime.now().isoformat(),
            "source_id": self.organism_id
        }
        
        await self.p2p_network.broadcast_message(json.dumps(message))
        
        # Update local performance
        await self.papal_chain.update_agent_performance(self.organism_id, performance_score, reputation)
        
        self.logger.info(f"Broadcasted performance update: {performance_score}")


# Convenience functions for easy integration
async def create_p2p_papal_integration(organism_id: str, port: int = 0) -> P2PPapalIntegration:
    """Create and start a P2P papal integration instance"""
    integration = P2PPapalIntegration(organism_id, port)
    await integration.start()
    return integration


async def get_network_chain_of_command(integration: P2PPapalIntegration) -> Dict[str, Any]:
    """Get the complete network chain of command"""
    return await integration.get_network_hierarchy()


async def initiate_network_decision(integration: P2PPapalIntegration, topic: str) -> str:
    """Initiate a network-wide decision process"""
    return await integration.initiate_network_conclave(topic) 