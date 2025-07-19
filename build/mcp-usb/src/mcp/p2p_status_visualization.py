"""
P2P Status Visualization System

Implements red-green-white status bar for P2P user status visualization with:
- Green section (top-aligned): Idle users ready for queries
- Red section (bottom-aligned): Active online non-idle users  
- White section (middle divider): High-reputation capable query servers
- Real-time status updates and proportional bar segment sizing
- Reputation scoring system for identifying capable query servers
- User capability assessment and reliability tracking
- Hover tooltips and detailed status information display
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from collections import defaultdict, deque

# Import P2P system components
from .integrated_p2p_genetic_system import IntegratedP2PGeneticSystem
from .p2p_network import P2PNetworkNode, NodeStatus


class UserStatus(Enum):
    """User status types for P2P network"""
    IDLE = "idle"  # Green: Ready for queries
    ACTIVE = "active"  # Red: Online but busy
    HIGH_REPUTATION = "high_reputation"  # White: Capable query servers


@dataclass
class UserSegment:
    """Represents a segment in the status bar"""
    status: UserStatus
    count: int
    percentage: float
    tooltip: str
    reputation_score: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = 0.0


@dataclass
class StatusBarData:
    """Complete status bar data structure"""
    segments: List[UserSegment]
    total_users: int
    last_update: float
    network_health: float
    average_reputation: float
    high_reputation_threshold: float = 0.8


class ReputationScorer:
    """Calculates reputation scores for P2P network users"""
    
    def __init__(self):
        self.reputation_weights = {
            'uptime': 0.25,
            'successful_transfers': 0.30,
            'response_time': 0.20,
            'data_quality': 0.15,
            'network_contribution': 0.10
        }
        self.min_reputation = 0.0
        self.max_reputation = 1.0
        
    def calculate_user_reputation(self, user_data: Dict[str, Any]) -> float:
        """Calculate reputation score for a user based on multiple factors"""
        if not user_data:
            return self.min_reputation
            
        uptime_score = min(1.0, user_data.get('uptime', 0) / 86400)  # Normalize to 24 hours
        transfer_success_rate = user_data.get('successful_transfers', 0) / max(1, user_data.get('total_transfers', 1))
        response_time_score = max(0, 1 - (user_data.get('avg_response_time', 1000) / 5000))  # Normalize to 5 seconds
        data_quality_score = user_data.get('data_quality_score', 0.5)
        network_contribution = user_data.get('network_contribution_score', 0.5)
        
        reputation = (
            uptime_score * self.reputation_weights['uptime'] +
            transfer_success_rate * self.reputation_weights['successful_transfers'] +
            response_time_score * self.reputation_weights['response_time'] +
            data_quality_score * self.reputation_weights['data_quality'] +
            network_contribution * self.reputation_weights['network_contribution']
        )
        
        return max(self.min_reputation, min(self.max_reputation, reputation))
    
    def assess_user_capabilities(self, user_data: Dict[str, Any]) -> List[str]:
        """Assess user capabilities based on performance metrics"""
        capabilities = []
        
        if user_data.get('reputation_score', 0) >= 0.8:
            capabilities.append('high_reputation_server')
        
        if user_data.get('avg_response_time', 1000) < 2000:
            capabilities.append('fast_response')
            
        if user_data.get('successful_transfers', 0) > 100:
            capabilities.append('experienced')
            
        if user_data.get('uptime', 0) > 3600:  # More than 1 hour
            capabilities.append('stable')
            
        if user_data.get('data_quality_score', 0) > 0.9:
            capabilities.append('high_quality_data')
            
        return capabilities


class P2PStatusVisualizer:
    """
    Main P2P status visualization system with red-green-white status bar.
    
    Features:
    - Real-time status updates and proportional bar segment sizing
    - Reputation scoring system for identifying capable query servers
    - User capability assessment and reliability tracking
    - Hover tooltips and detailed status information display
    """
    
    def __init__(self, p2p_system: IntegratedP2PGeneticSystem, update_interval: float = 1.0):
        self.p2p_system = p2p_system
        self.update_interval = update_interval
        self.reputation_scorer = ReputationScorer()
        self.logger = logging.getLogger("P2PStatusVisualizer")
        
        # Status tracking
        self.current_status: Optional[StatusBarData] = None
        self.status_history: deque = deque(maxlen=100)  # Keep last 100 updates
        self.last_update_time = 0.0
        
        # Configuration
        self.high_reputation_threshold = 0.8
        self.idle_timeout = 300  # 5 minutes
        self.active_timeout = 60  # 1 minute
        
        # Performance tracking
        self.update_times: deque = deque(maxlen=50)
        
    async def start_monitoring(self):
        """Start continuous status monitoring"""
        self.logger.info("Starting P2P status monitoring")
        
        while True:
            try:
                start_time = time.time()
                
                # Update status
                await self.update_status()
                
                # Record update time
                update_duration = time.time() - start_time
                self.update_times.append(update_duration)
                
                # Log performance
                if len(self.update_times) >= 10:
                    avg_update_time = sum(self.update_times) / len(self.update_times)
                    if avg_update_time > 0.5:  # More than 500ms
                        self.logger.warning(f"Slow status updates: {avg_update_time:.3f}s average")
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in status monitoring: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def update_status(self):
        """Update the current status bar data"""
        try:
            # Get system status from P2P system
            system_status = self.p2p_system.get_system_status()
            
            # Classify users into segments
            user_segments = await self._classify_users(system_status)
            
            # Calculate network health
            network_health = self._calculate_network_health(system_status)
            
            # Calculate average reputation
            total_reputation = sum(seg.reputation_score for seg in user_segments)
            avg_reputation = total_reputation / len(user_segments) if user_segments else 0.0
            
            # Create status bar data
            self.current_status = StatusBarData(
                segments=user_segments,
                total_users=sum(seg.count for seg in user_segments),
                last_update=time.time(),
                network_health=network_health,
                average_reputation=avg_reputation,
                high_reputation_threshold=self.high_reputation_threshold
            )
            
            # Store in history
            self.status_history.append(self.current_status)
            self.last_update_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    async def _classify_users(self, system_status: Dict[str, Any]) -> List[UserSegment]:
        """Classify users into status segments"""
        connected_peers = system_status.get('connected_peers', 0)
        peer_data = system_status.get('peer_data', {})
        
        if connected_peers == 0:
            return []
        
        # Initialize counters
        idle_count = 0
        active_count = 0
        high_rep_count = 0
        
        # Classify each peer
        for peer_id, peer_info in peer_data.items():
            reputation = self.reputation_scorer.calculate_user_reputation(peer_info)
            capabilities = self.reputation_scorer.assess_user_capabilities(peer_info)
            
            # Update peer info with calculated values
            peer_info['reputation_score'] = reputation
            peer_info['capabilities'] = capabilities
            
            # Classify based on status and reputation
            last_seen = peer_info.get('last_seen', 0)
            time_since_seen = time.time() - last_seen
            
            if reputation >= self.high_reputation_threshold:
                high_rep_count += 1
            elif time_since_seen > self.idle_timeout:
                idle_count += 1
            else:
                active_count += 1
        
        # Create segments
        segments = []
        total = connected_peers
        
        if idle_count > 0:
            segments.append(UserSegment(
                status=UserStatus.IDLE,
                count=idle_count,
                percentage=(idle_count / total) * 100,
                tooltip=f"{idle_count} idle users ready for queries",
                reputation_score=0.5,  # Default for idle users
                capabilities=['available'],
                last_seen=time.time()
            ))
        
        if high_rep_count > 0:
            segments.append(UserSegment(
                status=UserStatus.HIGH_REPUTATION,
                count=high_rep_count,
                percentage=(high_rep_count / total) * 100,
                tooltip=f"{high_rep_count} high-reputation capable query servers",
                reputation_score=0.9,  # High reputation
                capabilities=['high_reputation_server', 'capable_query_server'],
                last_seen=time.time()
            ))
        
        if active_count > 0:
            segments.append(UserSegment(
                status=UserStatus.ACTIVE,
                count=active_count,
                percentage=(active_count / total) * 100,
                tooltip=f"{active_count} active online non-idle users",
                reputation_score=0.6,  # Moderate reputation
                capabilities=['online', 'busy'],
                last_seen=time.time()
            ))
        
        return segments
    
    def _calculate_network_health(self, system_status: Dict[str, Any]) -> float:
        """Calculate overall network health score"""
        try:
            # Get various health indicators
            uptime = system_status.get('uptime', 0)
            connected_peers = system_status.get('connected_peers', 0)
            genetic_diversity = system_status.get('genetic_diversity', 0.0)
            network_fitness = system_status.get('system_metrics', {}).get('network_fitness', 0.0)
            
            # Normalize uptime (consider 24 hours as healthy)
            uptime_score = min(1.0, uptime / 86400)
            
            # Normalize peer count (consider 10+ peers as healthy)
            peer_score = min(1.0, connected_peers / 10)
            
            # Combine scores
            health_score = (
                uptime_score * 0.3 +
                peer_score * 0.3 +
                genetic_diversity * 0.2 +
                network_fitness * 0.2
            )
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating network health: {e}")
            return 0.0
    
    def render_status_bar(self) -> str:
        """Render the status bar as a text-based visualization"""
        if not self.current_status:
            return "[NO DATA]"
        
        bar_parts = []
        
        for segment in self.current_status.segments:
            if segment.count == 0:
                continue
                
            # Create visual representation
            if segment.status == UserStatus.IDLE:
                color = "GREEN"
                symbol = "●"
            elif segment.status == UserStatus.HIGH_REPUTATION:
                color = "WHITE"
                symbol = "◆"
            else:  # ACTIVE
                color = "RED"
                symbol = "●"
            
            bar_parts.append(f"[{color} {symbol}x{segment.count}]")
        
        # Add network health indicator
        health_emoji = "🟢" if self.current_status.network_health > 0.7 else "🟡" if self.current_status.network_health > 0.4 else "🔴"
        
        return f"{' '.join(bar_parts)} {health_emoji} Health: {self.current_status.network_health:.1%}"
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information for tooltips and UI"""
        if not self.current_status:
            return {"error": "No status data available"}
        
        return {
            "status_bar": self.render_status_bar(),
            "segments": [
                {
                    "status": seg.status.value,
                    "count": seg.count,
                    "percentage": seg.percentage,
                    "tooltip": seg.tooltip,
                    "reputation_score": seg.reputation_score,
                    "capabilities": seg.capabilities
                }
                for seg in self.current_status.segments
            ],
            "network_health": self.current_status.network_health,
            "average_reputation": self.current_status.average_reputation,
            "total_users": self.current_status.total_users,
            "last_update": self.current_status.last_update,
            "update_frequency": self.update_interval,
            "performance": {
                "avg_update_time": sum(self.update_times) / len(self.update_times) if self.update_times else 0,
                "update_count": len(self.update_times)
            }
        }
    
    def get_status_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent status history for trend analysis"""
        history = []
        for status in list(self.status_history)[-limit:]:
            history.append({
                "timestamp": status.last_update,
                "total_users": status.total_users,
                "network_health": status.network_health,
                "average_reputation": status.average_reputation,
                "segments": [
                    {
                        "status": seg.status.value,
                        "count": seg.count,
                        "percentage": seg.percentage
                    }
                    for seg in status.segments
                ]
            })
        return history
    
    async def get_real_time_updates(self) -> Dict[str, Any]:
        """Get real-time status updates for streaming"""
        await self.update_status()
        return self.get_detailed_status()


class P2PStatusBarRenderer:
    """Renders P2P status bar in various formats"""
    
    @staticmethod
    def render_ascii_bar(segments: List[UserSegment], width: int = 50) -> str:
        """Render ASCII status bar"""
        if not segments:
            return "[" + " " * width + "]"
        
        bar = "["
        total_count = sum(seg.count for seg in segments)
        
        for segment in segments:
            if segment.count == 0:
                continue
                
            segment_width = int((segment.count / total_count) * width)
            
            if segment.status == UserStatus.IDLE:
                char = "█"  # Green block
            elif segment.status == UserStatus.HIGH_REPUTATION:
                char = "░"  # White block
            else:  # ACTIVE
                char = "▓"  # Red block
            
            bar += char * segment_width
        
        # Pad to full width
        bar += " " * (width - len(bar) + 1)
        bar += "]"
        
        return bar
    
    @staticmethod
    def render_html_bar(segments: List[UserSegment]) -> str:
        """Render HTML status bar for web interfaces"""
        if not segments:
            return '<div class="p2p-status-bar empty">No data available</div>'
        
        html = '<div class="p2p-status-bar">'
        
        for segment in segments:
            if segment.count == 0:
                continue
                
            percentage = segment.percentage
            color_class = {
                UserStatus.IDLE: "green",
                UserStatus.HIGH_REPUTATION: "white", 
                UserStatus.ACTIVE: "red"
            }[segment.status]
            
            html += f'''
                <div class="status-segment {color_class}" 
                     style="width: {percentage}%" 
                     title="{segment.tooltip}">
                    <span class="segment-count">{segment.count}</span>
                </div>
            '''
        
        html += '</div>'
        return html
    
    @staticmethod
    def render_json_status(status_data: StatusBarData) -> Dict[str, Any]:
        """Render status data as JSON for API responses"""
        return {
            "timestamp": status_data.last_update,
            "total_users": status_data.total_users,
            "network_health": status_data.network_health,
            "average_reputation": status_data.average_reputation,
            "segments": [
                {
                    "status": seg.status.value,
                    "count": seg.count,
                    "percentage": seg.percentage,
                    "tooltip": seg.tooltip,
                    "reputation_score": seg.reputation_score,
                    "capabilities": seg.capabilities
                }
                for seg in status_data.segments
            ],
            "high_reputation_threshold": status_data.high_reputation_threshold
        }


# Example usage and testing
async def test_p2p_status_visualization():
    """Test the P2P status visualization system"""
    
    # Create mock P2P system
    class MockP2PSystem:
        def get_system_status(self):
            return {
                'connected_peers': 15,
                'peer_data': {
                    f'peer_{i}': {
                        'uptime': 3600 + i * 100,
                        'successful_transfers': 50 + i * 10,
                        'total_transfers': 60 + i * 10,
                        'avg_response_time': 1000 + i * 50,
                        'data_quality_score': 0.8 + (i % 3) * 0.1,
                        'network_contribution_score': 0.7 + (i % 2) * 0.2,
                        'last_seen': time.time() - (i * 30)
                    }
                    for i in range(15)
                },
                'uptime': 7200,
                'genetic_diversity': 0.75,
                'system_metrics': {'network_fitness': 0.8}
            }
    
    mock_system = MockP2PSystem()
    visualizer = P2PStatusVisualizer(mock_system)
    
    # Test status update
    await visualizer.update_status()
    
    # Test rendering
    print("Status Bar:", visualizer.render_status_bar())
    print("Detailed Status:", visualizer.get_detailed_status())
    
    # Test ASCII rendering
    if visualizer.current_status:
        ascii_bar = P2PStatusBarRenderer.render_ascii_bar(visualizer.current_status.segments)
        print("ASCII Bar:", ascii_bar)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_p2p_status_visualization()) 