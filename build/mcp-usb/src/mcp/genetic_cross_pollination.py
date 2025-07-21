"""
Genetic Cross-Pollination System for P2P optimization sharing.

This module integrates the genetic data exchange system with the hormone system
and neural network models to enable cross-pollination of optimizations between
different instances of the MCP system.

It implements the genetic-inspired encoding for metadata and integration
instructions, ensuring secure and privacy-preserving sharing of optimizations.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from src.mcp.genetic_data_exchange import (
    P2PDataExchange, 
    DataType, 
    GeneticDataPacket,
    GeneticMetadata,
    PerformanceValidator
)
from src.mcp.hormone_system_integration import HormoneSystemIntegration
from src.mcp.brain_state_aggregator import BrainStateAggregator


class GeneticCrossPollination:
    """
    Genetic cross-pollination system for sharing optimizations between MCP instances.
    
    This class integrates the P2P data exchange system with the hormone system
    and neural network models to enable cross-pollination of optimizations.
    """
    
    def __init__(self, 
                 hormone_system: Optional[HormoneSystemIntegration] = None,
                 brain_state: Optional[BrainStateAggregator] = None):
        """
        Initialize the genetic cross-pollination system.
        
        Args:
            hormone_system: Optional hormone system for integration
            brain_state: Optional brain state aggregator for monitoring
        """
        self.logger = logging.getLogger("GeneticCrossPollination")
        
        # Initialize components
        self.hormone_system = hormone_system
        self.brain_state = brain_state
        self.p2p_exchange = P2PDataExchange(hormone_system)
        self.performance_validator = PerformanceValidator()
        
        # Configuration
        self.sharing_enabled = True
        self.auto_integration_enabled = False
        self.privacy_level = "high"  # high, medium, low
        
        # Tracking
        self.shared_optimizations = {}
        self.received_optimizations = {}
        self.integration_history = []
        
        # Create data directories
        self.data_dir = os.path.join("data", "genetic_cross_pollination")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("Genetic Cross-Pollination system initialized")
    
    async def share_neural_model_optimization(self, 
                                           model_name: str, 
                                           model_data: Dict[str, Any],
                                           performance_metrics: Dict[str, float]) -> bool:
        """
        Share a neural model optimization with peers.
        
        Args:
            model_name: Name of the neural model
            model_data: Neural model data
            performance_metrics: Performance metrics for the model
            
        Returns:
            True if shared successfully, False otherwise
        """
        try:
            self.logger.info(f"Sharing neural model optimization: {model_name}")
            
            # Check if sharing is enabled
            if not self.sharing_enabled:
                self.logger.info("Sharing is disabled")
                return False
            
            # Prepare data for sharing
            data = {
                "model_name": model_name,
                "model_data": model_data,
                "performance_metrics": performance_metrics,
                "timestamp": time.time()
            }
            
            # Get expected benefits based on performance metrics
            expected_benefits = {
                "accuracy": performance_metrics.get("accuracy", 0.0) * 0.8,  # Conservative estimate
                "latency": performance_metrics.get("latency_improvement", 0.0) * 0.8,
                "resource_usage": performance_metrics.get("resource_improvement", 0.0) * 0.8
            }
            
            # Prepare genetic packet
            packet = await self.p2p_exchange.prepare_data_for_sharing(
                data=data,
                data_type=DataType.NEURAL_MODEL,
                target_components=["neural_network_models"],
                purpose=f"Optimized neural model: {model_name}",
                expected_benefits=expected_benefits
            )
            
            if not packet:
                self.logger.error(f"Failed to prepare neural model for sharing: {model_name}")
                return False
            
            # Share packet
            success = await self.p2p_exchange.share_data_packet(packet)
            
            if success:
                # Track shared optimization
                self.shared_optimizations[packet.validation_checksum] = {
                    "type": "neural_model",
                    "name": model_name,
                    "timestamp": time.time(),
                    "metrics": performance_metrics
                }
                
                self.logger.info(f"Neural model optimization shared successfully: {model_name}")
                
                # Release dopamine if hormone system is available
                if self.hormone_system and hasattr(self.hormone_system, "emit_event"):
                    self.hormone_system.emit_event(
                        "optimization_shared",
                        {
                            "type": "neural_model",
                            "name": model_name,
                            "metrics": performance_metrics
                        }
                    )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sharing neural model optimization: {e}")
            return False
    
    async def share_hormone_profile_optimization(self, 
                                              profile_name: str,
                                              hormone_levels: Dict[str, float],
                                              hormone_trends: Dict[str, Any],
                                              performance_metrics: Dict[str, float]) -> bool:
        """
        Share a hormone profile optimization with peers.
        
        Args:
            profile_name: Name of the hormone profile
            hormone_levels: Hormone levels
            hormone_trends: Hormone trends
            performance_metrics: Performance metrics for the profile
            
        Returns:
            True if shared successfully, False otherwise
        """
        try:
            self.logger.info(f"Sharing hormone profile optimization: {profile_name}")
            
            # Check if sharing is enabled
            if not self.sharing_enabled:
                self.logger.info("Sharing is disabled")
                return False
            
            # Prepare data for sharing
            data = {
                "profile_name": profile_name,
                "hormone_levels": hormone_levels,
                "hormone_trends": hormone_trends,
                "performance_metrics": performance_metrics,
                "timestamp": time.time()
            }
            
            # Get expected benefits based on performance metrics
            expected_benefits = {
                "stability": performance_metrics.get("stability", 0.0) * 0.8,
                "responsiveness": performance_metrics.get("responsiveness", 0.0) * 0.8,
                "balance": performance_metrics.get("balance", 0.0) * 0.8
            }
            
            # Prepare genetic packet
            packet = await self.p2p_exchange.prepare_data_for_sharing(
                data=data,
                data_type=DataType.HORMONE_PROFILE,
                target_components=["hormone_system_controller", "hormone_system_integration"],
                purpose=f"Optimized hormone profile: {profile_name}",
                expected_benefits=expected_benefits
            )
            
            if not packet:
                self.logger.error(f"Failed to prepare hormone profile for sharing: {profile_name}")
                return False
            
            # Share packet
            success = await self.p2p_exchange.share_data_packet(packet)
            
            if success:
                # Track shared optimization
                self.shared_optimizations[packet.validation_checksum] = {
                    "type": "hormone_profile",
                    "name": profile_name,
                    "timestamp": time.time(),
                    "metrics": performance_metrics
                }
                
                self.logger.info(f"Hormone profile optimization shared successfully: {profile_name}")
                
                # Release dopamine if hormone system is available
                if self.hormone_system and hasattr(self.hormone_system, "emit_event"):
                    self.hormone_system.emit_event(
                        "optimization_shared",
                        {
                            "type": "hormone_profile",
                            "name": profile_name,
                            "metrics": performance_metrics
                        }
                    )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sharing hormone profile optimization: {e}")
            return False
    
    async def receive_optimization(self, packet_id: str) -> Optional[Dict[str, Any]]:
        """
        Receive an optimization from peers.
        
        Args:
            packet_id: ID of the packet to receive
            
        Returns:
            Optimization data if received successfully, None otherwise
        """
        try:
            self.logger.info(f"Receiving optimization: {packet_id}")
            
            # Receive packet
            packet = await self.p2p_exchange.receive_data_packet(packet_id)
            
            if not packet:
                self.logger.error(f"Failed to receive optimization: {packet_id}")
                return None
            
            # Decrypt and decompress data
            data = packet.decrypt_and_decompress()
            
            # Track received optimization
            self.received_optimizations[packet_id] = {
                "type": packet.metadata.data_type.value,
                "timestamp": time.time(),
                "metadata": packet.metadata.to_dict()
            }
            
            # Auto-integrate if enabled
            if self.auto_integration_enabled:
                asyncio.create_task(self.integrate_optimization(packet))
            
            self.logger.info(f"Optimization received successfully: {packet_id}")
            
            return {
                "packet_id": packet_id,
                "data_type": packet.metadata.data_type.value,
                "metadata": packet.metadata.to_dict(),
                "data": data
            }
            
        except Exception as e:
            self.logger.error(f"Error receiving optimization: {e}")
            return None
    
    async def integrate_optimization(self, packet: GeneticDataPacket) -> bool:
        """
        Integrate an optimization into the system.
        
        Args:
            packet: Genetic data packet to integrate
            
        Returns:
            True if integrated successfully, False otherwise
        """
        try:
            self.logger.info(f"Integrating optimization of type {packet.metadata.data_type.value}")
            
            # Check if brain state is in a good state for integration
            if self.brain_state:
                hormone_levels = self.brain_state.get_hormone_levels()
                
                # Check dopamine level (should be high for reward state)
                if hormone_levels.get("dopamine", 0.0) < 0.5:
                    self.logger.info("Integration delayed: dopamine level too low")
                    return False
                
                # Check cortisol level (should be low for low stress)
                if hormone_levels.get("cortisol", 0.0) > 0.7:
                    self.logger.info("Integration delayed: cortisol level too high")
                    return False
            
            # Integrate using P2P exchange
            success = await self.p2p_exchange.integrate_data_packet(packet)
            
            if success:
                # Record integration in history
                self.integration_history.append({
                    "packet_id": packet.validation_checksum,
                    "data_type": packet.metadata.data_type.value,
                    "timestamp": time.time(),
                    "success": True
                })
                
                # Release dopamine if hormone system is available
                if self.hormone_system and hasattr(self.hormone_system, "emit_event"):
                    self.hormone_system.emit_event(
                        "optimization_integrated",
                        {
                            "type": packet.metadata.data_type.value,
                            "benefits": packet.metadata.expected_benefits
                        }
                    )
            else:
                # Record failed integration
                self.integration_history.append({
                    "packet_id": packet.validation_checksum,
                    "data_type": packet.metadata.data_type.value,
                    "timestamp": time.time(),
                    "success": False
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error integrating optimization: {e}")
            
            # Record failed integration
            self.integration_history.append({
                "packet_id": packet.validation_checksum if packet else "unknown",
                "data_type": packet.metadata.data_type.value if packet else "unknown",
                "timestamp": time.time(),
                "success": False,
                "error": str(e)
            })
            
            return False
    
    def set_sharing_enabled(self, enabled: bool) -> None:
        """
        Enable or disable sharing of optimizations.
        
        Args:
            enabled: True to enable sharing, False to disable
        """
        self.sharing_enabled = enabled
        self.logger.info(f"Sharing {'enabled' if enabled else 'disabled'}")
    
    def set_auto_integration_enabled(self, enabled: bool) -> None:
        """
        Enable or disable automatic integration of received optimizations.
        
        Args:
            enabled: True to enable auto-integration, False to disable
        """
        self.auto_integration_enabled = enabled
        self.logger.info(f"Auto-integration {'enabled' if enabled else 'disabled'}")
    
    def set_privacy_level(self, level: str) -> None:
        """
        Set the privacy level for sharing optimizations.
        
        Args:
            level: Privacy level ('high', 'medium', or 'low')
        """
        if level in ["high", "medium", "low"]:
            self.privacy_level = level
            
            # Update privacy filter epsilon based on level
            if hasattr(self.p2p_exchange, "privacy_filter") and hasattr(self.p2p_exchange.privacy_filter, "epsilon"):
                if level == "high":
                    self.p2p_exchange.privacy_filter.epsilon = 0.5
                elif level == "medium":
                    self.p2p_exchange.privacy_filter.epsilon = 1.0
                elif level == "low":
                    self.p2p_exchange.privacy_filter.epsilon = 2.0
            
            self.logger.info(f"Privacy level set to {level}")
        else:
            self.logger.warning(f"Invalid privacy level: {level}")
    
    def get_shared_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all shared optimizations.
        
        Returns:
            Dictionary of shared optimizations
        """
        return self.shared_optimizations.copy()
    
    def get_received_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all received optimizations.
        
        Returns:
            Dictionary of received optimizations
        """
        return self.received_optimizations.copy()
    
    def get_integration_history(self) -> List[Dict[str, Any]]:
        """
        Get integration history.
        
        Returns:
            List of integration history entries
        """
        return self.integration_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the cross-pollination system.
        
        Returns:
            Dictionary of performance metrics
        """
        # Get metrics from P2P exchange
        metrics = self.p2p_exchange.get_performance_metrics()
        
        # Add cross-pollination specific metrics
        metrics.update({
            "shared_count": len(self.shared_optimizations),
            "received_count": len(self.received_optimizations),
            "integration_success_rate": self._calculate_integration_success_rate(),
            "privacy_level": {"high": 0.9, "medium": 0.7, "low": 0.5}.get(self.privacy_level, 0.7)
        })
        
        return metrics
    
    def _calculate_integration_success_rate(self) -> float:
        """
        Calculate the success rate of optimization integrations.
        
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if not self.integration_history:
            return 1.0  # No integrations attempted
        
        successful = sum(1 for entry in self.integration_history if entry.get("success", False))
        return successful / len(self.integration_history)


# Example usage
async def example_usage():
    # Create cross-pollination system
    cross_pollination = GeneticCrossPollination()
    
    # Configure system
    cross_pollination.set_privacy_level("high")
    cross_pollination.set_auto_integration_enabled(True)
    
    # Share a neural model optimization
    neural_model_data = {
        "architecture": "feedforward",
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "bias": [0.01, 0.02]
    }
    
    performance_metrics = {
        "accuracy": 0.85,
        "latency_improvement": 0.2,
        "resource_improvement": 0.15
    }
    
    success = await cross_pollination.share_neural_model_optimization(
        model_name="pattern_recognition_model",
        model_data=neural_model_data,
        performance_metrics=performance_metrics
    )
    
    print(f"Shared neural model optimization: {success}")
    
    # Get performance metrics
    metrics = cross_pollination.get_performance_metrics()
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


if __name__ == "__main__":
    import asyncio
    
    # Run example
    asyncio.run(example_usage())