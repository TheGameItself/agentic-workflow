"""
Integrated P2P Genetic Evolution System

Combines all components for comprehensive network-wide evolution:
- Full P2P network with DHT routing
- Genetic data exchange with advanced DNA features
- Engram transfer for neural network training
- Cross-pollination and network-wide optimization
- Real-time performance monitoring and adaptation
"""

import asyncio
import time
import random
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .p2p_network import P2PNetwork
from .genetic_data_exchange import GeneticDataExchange
from .engram_transfer_system import EngramTransferSystem
from .advanced_genetic_evolution import NetworkWideEvolution
from .genetic_network_orchestrator import GeneticNetworkOrchestrator, NetworkOrchestrationConfig


@dataclass
class IntegratedSystemConfig:
    """Configuration for integrated P2P genetic system"""
    # Network configuration
    p2p_port: int = 0  # Auto-assign port
    max_connections: int = 20
    bootstrap_nodes: List[tuple] = None
    
    # Genetic evolution configuration
    evolution_frequency: int = 100
    cross_pollination_rate: float = 0.4
    innovation_pressure: float = 0.3
    diversity_target: float = 0.8
    
    # Engram transfer configuration
    engram_sharing_enabled: bool = True
    pattern_sharing_threshold: float = 0.8
    skill_sharing_threshold: float = 0.75
    
    # Performance thresholds
    fitness_threshold: float = 0.7
    improvement_threshold: float = 0.05
    
    # Resource limits
    bandwidth_limit_mbps: float = 10.0
    storage_limit_gb: float = 1.0


class IntegratedP2PGeneticSystem:
    """Main integrated system combining all P2P genetic components"""
    
    def __init__(self, organism_id: str, config: IntegratedSystemConfig):
        self.organism_id = organism_id
        self.config = config
        
        # Initialize core components
        self.p2p_network = P2PNetwork(organism_id, config.p2p_port)
        self.genetic_exchange = GeneticDataExchange(organism_id)
        self.engram_system = EngramTransferSystem(organism_id, self.p2p_network)
        self.network_evolution = NetworkWideEvolution(self.genetic_exchange)
        
        # Orchestration configuration
        orchestration_config = NetworkOrchestrationConfig(
            evolution_frequency=config.evolution_frequency,
            cross_pollination_rate=config.cross_pollination_rate,
            innovation_pressure=config.innovation_pressure,
            diversity_target=config.diversity_target,
            performance_threshold=config.fitness_threshold
        )
        
        self.orchestrator = GeneticNetworkOrchestrator(organism_id, orchestration_config)
        
        # System state
        self.running = False
        self.connected_peers: Dict[str, Dict[str, Any]] = {}
        self.system_metrics = {
            'uptime': 0.0,
            'total_genetic_transfers': 0,
            'total_engram_transfers': 0,
            'successful_integrations': 0,
            'network_fitness': 0.0,
            'genetic_diversity': 0.0
        }
        
        # Set up P2P callbacks
        self._setup_p2p_callbacks()
        
        # Performance monitoring
        self.performance_history = []
        self.last_evolution_time = time.time()
    
    def _setup_p2p_callbacks(self):
        """Set up P2P network event callbacks"""
        self.p2p_network.on_genetic_data_received = self._handle_genetic_data_received
        self.p2p_network.on_node_connected = self._handle_node_connected
        self.p2p_network.on_node_disconnected = self._handle_node_disconnected
    
    async def start_system(self, bootstrap_nodes: Optional[List[Tuple[str, int]]] = None):
        """Start the integrated P2P genetic system"""
        print(f"Starting integrated P2P genetic system for {self.organism_id}")
        
        self.running = True
        start_time = time.time()
        
        # Start P2P network
        await self.p2p_network.start(bootstrap_nodes or self.config.bootstrap_nodes)
        
        # Initialize genetic material
        await self._initialize_genetic_material()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._evolution_loop()),
            asyncio.create_task(self._engram_sharing_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._network_maintenance_loop())
        ]
        
        print(f"Integrated system started in {time.time() - start_time:.2f} seconds")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("System tasks cancelled")
    
    async def stop_system(self):
        """Stop the integrated system"""
        print("Stopping integrated P2P genetic system...")
        
        self.running = False
        
        # Stop P2P network
        await self.p2p_network.stop()
        
        print("Integrated system stopped")
    
    async def _initialize_genetic_material(self):
        """Initialize genetic material and engrams"""
        # Create initial neural network genetic data
        neural_data = {
            'model_id': f'initial_model_{self.organism_id}',
            'architecture': {
                'input': {'size': 784, 'type': 'dense'},
                'hidden1': {'size': 256, 'activation': 'relu', 'dropout': 0.2},
                'hidden2': {'size': 128, 'activation': 'relu', 'dropout': 0.1},
                'output': {'size': 10, 'activation': 'softmax'}
            },
            'performance_metrics': {
                'accuracy': random.uniform(0.75, 0.85),
                'speed': random.uniform(0.7, 0.9),
                'efficiency': random.uniform(0.6, 0.8)
            },
            'training_params': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'optimizer': 'adam'
            }
        }
        
        # Create genetic packet
        genetic_packet = self.genetic_exchange.create_genetic_packet(
            'neural_network', neural_data, 'medium'
        )
        
        print(f"Initialized genetic material: {genetic_packet.packet_id}")
        
        # Create initial pattern recognition engram
        pattern_data = {
            'pattern_type': 'visual',
            'raw_features': np.random.randn(100).tolist(),
            'labels': ['digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4',
                      'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9'],
            'accuracy_metrics': {
                'training_accuracy': random.uniform(0.85, 0.95),
                'validation_accuracy': random.uniform(0.8, 0.9),
                'test_accuracy': random.uniform(0.75, 0.88)
            },
            'training_examples': [
                {'input': np.random.randn(28, 28).tolist(), 'output': f'digit_{i}'}
                for i in range(10)
            ]
        }
        
        engram = self.engram_system.create_pattern_recognition_engram(pattern_data)
        print(f"Created initial engram: {engram.engram_id}")
    
    async def _evolution_loop(self):
        """Main evolution loop"""
        evolution_cycle = 0
        
        while self.running:
            try:
                evolution_cycle += 1
                
                # Check if it's time for evolution
                if evolution_cycle % self.config.evolution_frequency == 0:
                    await self._perform_evolution_cycle()
                
                # Cross-pollination opportunities
                if evolution_cycle % 20 == 0:
                    await self._check_cross_pollination_opportunities()
                
                await asyncio.sleep(1.0)  # 1 second cycle
                
            except Exception as e:
                print(f"Error in evolution loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _perform_evolution_cycle(self):
        """Perform full evolution cycle"""
        print(f"Starting evolution cycle for {self.organism_id}")
        
        # Orchestrate network-wide evolution
        evolution_results = await self.network_evolution.orchestrate_network_evolution(1)
        
        if evolution_results['generations']:
            latest_gen = evolution_results['generations'][-1]
            network_metrics = latest_gen['network_metrics']
            
            # Update system metrics
            self.system_metrics['network_fitness'] = network_metrics.get('avg_fitness', 0.0)
            self.system_metrics['genetic_diversity'] = network_metrics.get('genetic_diversity', 0.0)
            
            # Record performance
            self.performance_history.append({
                'timestamp': time.time(),
                'fitness': network_metrics.get('avg_fitness', 0.0),
                'diversity': network_metrics.get('genetic_diversity', 0.0),
                'innovation_rate': network_metrics.get('innovation_rate', 0.0)
            })
            
            print(f"Evolution cycle completed - Fitness: {network_metrics.get('avg_fitness', 0.0):.3f}")
        
        self.last_evolution_time = time.time()
    
    async def _check_cross_pollination_opportunities(self):
        """Check for cross-pollination opportunities with connected peers"""
        if not self.connected_peers:
            return
        
        # Check if we should share genetic improvements
        if random.random() < self.config.cross_pollination_rate:
            await self._share_genetic_improvements()
        
        # Check if we should share engrams
        if self.config.engram_sharing_enabled and random.random() < 0.3:
            await self._share_best_engrams()
    
    async def _share_genetic_improvements(self):
        """Share genetic improvements with network peers"""
        # Get recent genetic improvements
        if self.genetic_exchange.fitness_history:
            recent_fitness = self.genetic_exchange.fitness_history[-1]
            
            if recent_fitness['fitness_score'] >= self.config.fitness_threshold:
                # Create genetic improvement data
                improvement_data = {
                    'improvement_type': 'fitness_optimization',
                    'fitness_score': recent_fitness['fitness_score'],
                    'genetic_modifications': {
                        'mutation_count': random.randint(1, 5),
                        'crossover_events': random.randint(0, 3),
                        'innovation_events': random.randint(0, 2)
                    },
                    'performance_metrics': {
                        'accuracy_improvement': random.uniform(0.02, 0.08),
                        'efficiency_gain': random.uniform(0.01, 0.05),
                        'stability_increase': random.uniform(0.01, 0.04)
                    },
                    'source_organism': self.organism_id,
                    'timestamp': time.time()
                }
                
                # Share with random peers
                target_peers = random.sample(list(self.connected_peers.keys()), 
                                           min(3, len(self.connected_peers)))
                
                for peer_id in target_peers:
                    success = await self.p2p_network.send_genetic_data(peer_id, improvement_data)
                    if success:
                        self.system_metrics['total_genetic_transfers'] += 1
                        print(f"Shared genetic improvement with {peer_id}")
    
    async def _share_best_engrams(self):
        """Share best performing engrams with network peers"""
        # Get best pattern recognition engrams
        best_engrams = []
        
        for engram in self.engram_system.pattern_engrams.values():
            if engram.generalization_score >= self.config.pattern_sharing_threshold:
                best_engrams.append(engram)
        
        if best_engrams:
            # Sort by generalization score
            best_engrams.sort(key=lambda e: e.generalization_score, reverse=True)
            
            # Share top engram with random peer
            if self.connected_peers:
                target_peer = random.choice(list(self.connected_peers.keys()))
                best_engram = best_engrams[0]
                
                success = await self.engram_system.transfer_engram(
                    best_engram.engram_id, target_peer
                )
                
                if success:
                    self.system_metrics['total_engram_transfers'] += 1
                    print(f"Shared engram {best_engram.engram_id} with {target_peer}")
    
    async def _engram_sharing_loop(self):
        """Background loop for engram sharing"""
        while self.running:
            try:
                # Periodic engram sharing
                if self.connected_peers and random.random() < 0.1:  # 10% chance
                    await self._share_best_engrams()
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in engram sharing loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance and adapt"""
        while self.running:
            try:
                # Update system metrics
                self.system_metrics['uptime'] = time.time() - self.last_evolution_time
                
                # Check network health
                network_stats = self.p2p_network.get_network_stats()
                
                # Adapt parameters based on performance
                if len(self.performance_history) >= 5:
                    recent_performance = self.performance_history[-5:]
                    avg_fitness = sum(p['fitness'] for p in recent_performance) / len(recent_performance)
                    
                    # Increase innovation pressure if fitness is stagnant
                    if avg_fitness < self.config.fitness_threshold:
                        self.config.innovation_pressure = min(0.5, self.config.innovation_pressure + 0.05)
                        print(f"Increased innovation pressure to {self.config.innovation_pressure:.3f}")
                
                await asyncio.sleep(60.0)  # Monitor every minute
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                await asyncio.sleep(120.0)
    
    async def _network_maintenance_loop(self):
        """Network maintenance and optimization"""
        while self.running:
            try:
                # Clean up stale peer data
                current_time = time.time()
                stale_peers = []
                
                for peer_id, peer_data in self.connected_peers.items():
                    if current_time - peer_data.get('last_seen', 0) > 300:  # 5 minutes
                        stale_peers.append(peer_id)
                
                for peer_id in stale_peers:
                    del self.connected_peers[peer_id]
                    print(f"Removed stale peer: {peer_id}")
                
                # Optimize network connections
                network_stats = self.p2p_network.get_network_stats()
                if network_stats['connected_nodes'] < 3:
                    print("Low peer count, attempting to discover more peers")
                    # Would implement peer discovery here
                
                await asyncio.sleep(120.0)  # Maintenance every 2 minutes
                
            except Exception as e:
                print(f"Error in network maintenance: {e}")
                await asyncio.sleep(300.0)
    
    async def _handle_genetic_data_received(self, genetic_data: Dict[str, Any], sender_id: str):
        """Handle received genetic data"""
        try:
            print(f"Received genetic data from {sender_id}: {genetic_data.get('improvement_type', 'unknown')}")
            
            # Process genetic improvement
            if genetic_data.get('fitness_score', 0) >= self.config.fitness_threshold:
                # Integrate genetic improvements
                success = await self._integrate_genetic_improvement(genetic_data)
                
                if success:
                    self.system_metrics['successful_integrations'] += 1
                    print(f"Successfully integrated genetic improvement from {sender_id}")
                
        except Exception as e:
            print(f"Error handling genetic data: {e}")
    
    async def _integrate_genetic_improvement(self, genetic_data: Dict[str, Any]) -> bool:
        """Integrate received genetic improvement"""
        try:
            # Create genetic packet from received data
            packet = self.genetic_exchange.create_genetic_packet(
                'genetic_improvement', genetic_data, 'medium'
            )
            
            # Simulate integration via genetic exchange
            encrypted_data = self.genetic_exchange._encrypt_packet(packet)
            success = await self.genetic_exchange.receive_genetic_data(
                encrypted_data, genetic_data.get('source_organism', 'unknown')
            )
            
            return success
            
        except Exception as e:
            print(f"Error integrating genetic improvement: {e}")
            return False
    
    async def _handle_node_connected(self, node_id: str):
        """Handle new node connection"""
        self.connected_peers[node_id] = {
            'connected_at': time.time(),
            'last_seen': time.time(),
            'genetic_exchanges': 0,
            'engram_exchanges': 0
        }
        
        print(f"New peer connected: {node_id}")
    
    async def _handle_node_disconnected(self, node_id: str):
        """Handle node disconnection"""
        if node_id in self.connected_peers:
            del self.connected_peers[node_id]
        
        print(f"Peer disconnected: {node_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        network_stats = self.p2p_network.get_network_stats()
        engram_stats = self.engram_system.get_transfer_statistics()
        genetic_diversity = self.genetic_exchange.calculate_genetic_diversity()
        
        return {
            'organism_id': self.organism_id,
            'system_metrics': self.system_metrics.copy(),
            'network_stats': network_stats,
            'engram_stats': engram_stats,
            'genetic_diversity': genetic_diversity,
            'connected_peers': len(self.connected_peers),
            'performance_history_length': len(self.performance_history),
            'config': {
                'evolution_frequency': self.config.evolution_frequency,
                'cross_pollination_rate': self.config.cross_pollination_rate,
                'innovation_pressure': self.config.innovation_pressure,
                'engram_sharing_enabled': self.config.engram_sharing_enabled
            },
            'last_evolution': self.last_evolution_time,
            'uptime': time.time() - self.last_evolution_time
        }


# Example usage and comprehensive testing
async def test_integrated_system():
    """Test the integrated P2P genetic system"""
    print("=" * 80)
    print("INTEGRATED P2P GENETIC SYSTEM TEST")
    print("=" * 80)
    
    # Create system configurations
    config1 = IntegratedSystemConfig(
        p2p_port=8001,
        evolution_frequency=50,
        cross_pollination_rate=0.5,
        innovation_pressure=0.3,
        engram_sharing_enabled=True
    )
    
    config2 = IntegratedSystemConfig(
        p2p_port=8002,
        evolution_frequency=60,
        cross_pollination_rate=0.4,
        innovation_pressure=0.25,
        engram_sharing_enabled=True
    )
    
    # Create integrated systems
    system1 = IntegratedP2PGeneticSystem("organism_alpha", config1)
    system2 = IntegratedP2PGeneticSystem("organism_beta", config2)
    
    # Start systems
    print("Starting integrated systems...")
    
    # Start system 1
    system1_task = asyncio.create_task(system1.start_system())
    await asyncio.sleep(2)  # Let first system start
    
    # Start system 2 with bootstrap to system 1
    bootstrap_nodes = [("127.0.0.1", 8001)]
    system2_task = asyncio.create_task(system2.start_system(bootstrap_nodes))
    
    # Let systems run and interact
    print("Systems running... allowing time for interaction")
    await asyncio.sleep(10)
    
    # Get system status
    status1 = system1.get_system_status()
    status2 = system2.get_system_status()
    
    print("\nSYSTEM 1 STATUS:")
    print(f"  Organism ID: {status1['organism_id']}")
    print(f"  Connected Peers: {status1['connected_peers']}")
    print(f"  Genetic Transfers: {status1['system_metrics']['total_genetic_transfers']}")
    print(f"  Engram Transfers: {status1['system_metrics']['total_engram_transfers']}")
    print(f"  Network Fitness: {status1['system_metrics']['network_fitness']:.3f}")
    print(f"  Genetic Diversity: {status1['genetic_diversity']:.3f}")
    
    print("\nSYSTEM 2 STATUS:")
    print(f"  Organism ID: {status2['organism_id']}")
    print(f"  Connected Peers: {status2['connected_peers']}")
    print(f"  Genetic Transfers: {status2['system_metrics']['total_genetic_transfers']}")
    print(f"  Engram Transfers: {status2['system_metrics']['total_engram_transfers']}")
    print(f"  Network Fitness: {status2['system_metrics']['network_fitness']:.3f}")
    print(f"  Genetic Diversity: {status2['genetic_diversity']:.3f}")
    
    # Stop systems
    print("\nStopping systems...")
    system1_task.cancel()
    system2_task.cancel()
    
    try:
        await system1_task
    except asyncio.CancelledError:
        pass
    
    try:
        await system2_task
    except asyncio.CancelledError:
        pass
    
    await system1.stop_system()
    await system2.stop_system()
    
    print("\n" + "=" * 80)
    print("INTEGRATED SYSTEM TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_integrated_system())