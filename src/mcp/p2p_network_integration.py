"""
P2P Network Integration and Status Visualization

This module implements P2P network integration with comprehensive status visualization:
- Red-green-white status bar for P2P user status visualization
- Green section (top-aligned) for idle users ready for queries
- Red section (bottom-aligned) for active online non-idle users
- White section (middle divider) for high-reputation capable query servers
- Real-time status updates and proportional bar segment sizing
- Reputation scoring system for identifying capable query servers
- User capability assessment and reliability tracking
- Hover tooltips and detailed status information display
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque

from .environmental_state import EnvironmentalState


class UserStatus(Enum):
    """User status types"""
    IDLE = "idle"                     # Ready for queries
    ACTIVE = "active"                 # Currently processing
    BUSY = "busy"                     # High load, limited availability
    OFFLINE = "offline"               # Not available
    MAINTENANCE = "maintenance"       # Under maintenance
    ERROR = "error"                   # Error state


class ServerCapability(Enum):
    """Server capability levels"""
    BASIC = "basic"                   # Basic query processing
    STANDARD = "standard"             # Standard processing capabilities
    ADVANCED = "advanced"             # Advanced processing capabilities
    EXPERT = "expert"                 # Expert-level capabilities
    MASTER = "master"                 # Master-level capabilities


class NetworkRegion(Enum):
    """Network regions for geographical distribution"""
    LOCAL = "local"                   # Local network
    REGIONAL = "regional"             # Regional network
    CONTINENTAL = "continental"       # Continental network
    GLOBAL = "global"                 # Global network


@dataclass
class UserProfile:
    """User profile for P2P network"""
    user_id: str
    username: str
    status: UserStatus = UserStatus.OFFLINE
    capability: ServerCapability = ServerCapability.BASIC
    region: NetworkRegion = NetworkRegion.LOCAL
    reputation_score: float = 0.5
    reliability_score: float = 0.5
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_seen: datetime = field(default_factory=datetime.now)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    total_queries_processed: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, new_status: UserStatus):
        """Update user status"""
        self.status = new_status
        self.last_seen = datetime.now()
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        
        # Update derived metrics
        if 'response_time' in metrics:
            self.average_response_time = metrics['response_time']
        
        if 'success_rate' in metrics:
            self.reliability_score = metrics['success_rate']
    
    def update_reputation(self, new_score: float):
        """Update reputation score"""
        self.reputation_score = max(0.0, min(1.0, new_score))
    
    def get_overall_score(self) -> float:
        """Calculate overall capability score"""
        # Weighted combination of reputation, reliability, and capability
        capability_weight = {
            ServerCapability.BASIC: 0.2,
            ServerCapability.STANDARD: 0.4,
            ServerCapability.ADVANCED: 0.6,
            ServerCapability.EXPERT: 0.8,
            ServerCapability.MASTER: 1.0
        }
        
        capability_score = capability_weight.get(self.capability, 0.2)
        
        overall_score = (
            self.reputation_score * 0.4 +
            self.reliability_score * 0.3 +
            capability_score * 0.3
        )
        
        return overall_score
    
    def is_available_for_queries(self) -> bool:
        """Check if user is available for queries"""
        return self.status in [UserStatus.IDLE, UserStatus.ACTIVE]
    
    def is_high_reputation_server(self) -> bool:
        """Check if user is a high-reputation server"""
        return (self.reputation_score >= 0.8 and 
                self.capability in [ServerCapability.EXPERT, ServerCapability.MASTER] and
                self.reliability_score >= 0.7)


@dataclass
class StatusBarSegment:
    """Represents a segment in the status bar"""
    segment_id: str
    user_id: str
    segment_type: str  # 'green', 'red', 'white'
    position: float  # 0.0 to 1.0
    width: float  # 0.0 to 1.0
    user_profile: UserProfile
    tooltip_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_tooltip_text(self) -> str:
        """Generate tooltip text for the segment"""
        profile = self.user_profile
        
        tooltip = f"""
User: {profile.username}
Status: {profile.status.value}
Capability: {profile.capability.value}
Reputation: {profile.reputation_score:.2f}
Reliability: {profile.reliability_score:.2f}
Region: {profile.region.value}
Uptime: {profile.uptime}
Queries: {profile.total_queries_processed}
Success Rate: {profile.successful_queries / max(profile.total_queries_processed, 1):.1%}
Avg Response: {profile.average_response_time:.3f}s
        """.strip()
        
        return tooltip


@dataclass
class NetworkMetrics:
    """Network-wide metrics"""
    total_users: int = 0
    active_users: int = 0
    idle_users: int = 0
    high_reputation_servers: int = 0
    average_reputation: float = 0.0
    average_reliability: float = 0.0
    network_health: float = 0.0
    query_success_rate: float = 0.0
    average_response_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class P2PNetworkIntegration:
    """
    P2P network integration with comprehensive status visualization.
    
    Features:
    - Real-time user status tracking
    - Reputation scoring and capability assessment
    - Status bar visualization with color-coded segments
    - Network health monitoring
    - Query routing and load balancing
    """
    
    def __init__(self, 
                 max_users: int = 1000,
                 update_interval: float = 1.0,
                 reputation_decay_rate: float = 0.01):
        self.max_users = max_users
        self.update_interval = update_interval
        self.reputation_decay_rate = reputation_decay_rate
        
        self.logger = logging.getLogger("P2PNetworkIntegration")
        
        # Core components
        self.users: Dict[str, UserProfile] = {}
        self.status_bar_segments: List[StatusBarSegment] = []
        self.network_metrics = NetworkMetrics()
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Event callbacks
        self.status_update_callbacks: List[callable] = []
        self.metrics_update_callbacks: List[callable] = []
        
        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.logger.info("P2P Network Integration initialized")
    
    async def start(self):
        """Start the P2P network integration"""
        if self.running:
            self.logger.warning("P2P network integration already running")
            return
        
        self.running = True
        self.update_task = asyncio.create_task(self._background_update_loop())
        self.logger.info("P2P network integration started")
    
    async def stop(self):
        """Stop the P2P network integration"""
        if not self.running:
            return
        
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("P2P network integration stopped")
    
    async def _background_update_loop(self):
        """Background loop for updates"""
        while self.running:
            try:
                await self._update_network_status()
                await self._update_reputation_scores()
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def register_user(self, user_id: str, username: str, 
                     capability: ServerCapability = ServerCapability.BASIC,
                     region: NetworkRegion = NetworkRegion.LOCAL) -> bool:
        """
        Register a new user in the P2P network.
        
        Args:
            user_id: Unique user identifier
            username: User display name
            capability: Server capability level
            region: Network region
            
        Returns:
            True if registration was successful
        """
        if len(self.users) >= self.max_users:
            self.logger.warning("Maximum users reached")
            return False
        
        if user_id in self.users:
            self.logger.warning(f"User {user_id} already registered")
            return False
        
        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            capability=capability,
            region=region
        )
        
        self.users[user_id] = user_profile
        self.logger.info(f"Registered user {username} ({user_id})")
        
        # Trigger status update
        asyncio.create_task(self._trigger_status_update())
        
        return True
    
    def unregister_user(self, user_id: str) -> bool:
        """Unregister a user from the P2P network"""
        if user_id not in self.users:
            self.logger.warning(f"User {user_id} not found")
            return False
        
        username = self.users[user_id].username
        del self.users[user_id]
        self.logger.info(f"Unregistered user {username} ({user_id})")
        
        # Trigger status update
        asyncio.create_task(self._trigger_status_update())
        
        return True
    
    def update_user_status(self, user_id: str, status: UserStatus, 
                          metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Update user status and metrics.
        
        Args:
            user_id: User identifier
            status: New status
            metrics: Optional performance metrics
            
        Returns:
            True if update was successful
        """
        if user_id not in self.users:
            self.logger.warning(f"User {user_id} not found")
            return False
        
        user_profile = self.users[user_id]
        user_profile.update_status(status)
        
        if metrics:
            user_profile.update_performance(metrics)
        
        # Trigger status update
        asyncio.create_task(self._trigger_status_update())
        
        return True
    
    def record_query_result(self, user_id: str, query_id: str, 
                          success: bool, response_time: float, 
                          result_quality: float = 1.0) -> bool:
        """
        Record query result for reputation calculation.
        
        Args:
            user_id: User who processed the query
            query_id: Query identifier
            success: Whether query was successful
            response_time: Response time in seconds
            result_quality: Quality of the result (0.0 to 1.0)
            
        Returns:
            True if recording was successful
        """
        if user_id not in self.users:
            self.logger.warning(f"User {user_id} not found")
            return False
        
        user_profile = self.users[user_id]
        
        # Update query statistics
        user_profile.total_queries_processed += 1
        if success:
            user_profile.successful_queries += 1
        
        # Update average response time
        if user_profile.average_response_time == 0.0:
            user_profile.average_response_time = response_time
        else:
            user_profile.average_response_time = (
                (user_profile.average_response_time + response_time) / 2
            )
        
        # Calculate new reputation score
        success_rate = user_profile.successful_queries / user_profile.total_queries_processed
        time_efficiency = max(0.0, 1.0 - (response_time / 10.0))  # Normalize to 10 seconds
        quality_factor = result_quality
        
        new_reputation = (success_rate * 0.5 + time_efficiency * 0.3 + quality_factor * 0.2)
        user_profile.update_reputation(new_reputation)
        
        # Record query history
        query_record = {
            'query_id': query_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'response_time': response_time,
            'result_quality': result_quality,
            'new_reputation': new_reputation
        }
        self.query_history.append(query_record)
        
        # Keep only recent history
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
        
        self.logger.debug(f"Recorded query result for user {user_id}: success={success}, "
                         f"response_time={response_time:.3f}s")
        
        return True
    
    async def _update_network_status(self):
        """Update network status and status bar segments"""
        # Clear existing segments
        self.status_bar_segments.clear()
        
        # Categorize users
        idle_users = []
        active_users = []
        high_reputation_servers = []
        
        for user_profile in self.users.values():
            if user_profile.is_available_for_queries():
                if user_profile.status == UserStatus.IDLE:
                    idle_users.append(user_profile)
                else:
                    active_users.append(user_profile)
            
            if user_profile.is_high_reputation_server():
                high_reputation_servers.append(user_profile)
        
        # Sort by overall score
        idle_users.sort(key=lambda u: u.get_overall_score(), reverse=True)
        active_users.sort(key=lambda u: u.get_overall_score(), reverse=True)
        high_reputation_servers.sort(key=lambda u: u.get_overall_score(), reverse=True)
        
        # Create status bar segments
        current_position = 0.0
        
        # Green section (idle users) - top-aligned
        if idle_users:
            segment_width = min(0.4, len(idle_users) / max(len(self.users), 1))
            for i, user_profile in enumerate(idle_users):
                segment = StatusBarSegment(
                    segment_id=f"green_{i}",
                    user_id=user_profile.user_id,
                    segment_type="green",
                    position=current_position,
                    width=segment_width / len(idle_users),
                    user_profile=user_profile,
                    tooltip_data={
                        'status': 'idle',
                        'capability': user_profile.capability.value,
                        'reputation': user_profile.reputation_score
                    }
                )
                self.status_bar_segments.append(segment)
                current_position += segment.width
        
        # White section (high-reputation servers) - middle divider
        if high_reputation_servers:
            segment_width = min(0.2, len(high_reputation_servers) / max(len(self.users), 1))
            for i, user_profile in enumerate(high_reputation_servers):
                segment = StatusBarSegment(
                    segment_id=f"white_{i}",
                    user_id=user_profile.user_id,
                    segment_type="white",
                    position=current_position,
                    width=segment_width / len(high_reputation_servers),
                    user_profile=user_profile,
                    tooltip_data={
                        'status': 'high_reputation',
                        'capability': user_profile.capability.value,
                        'reputation': user_profile.reputation_score
                    }
                )
                self.status_bar_segments.append(segment)
                current_position += segment.width
        
        # Red section (active users) - bottom-aligned
        if active_users:
            segment_width = min(0.4, len(active_users) / max(len(self.users), 1))
            for i, user_profile in enumerate(active_users):
                segment = StatusBarSegment(
                    segment_id=f"red_{i}",
                    user_id=user_profile.user_id,
                    segment_type="red",
                    position=current_position,
                    width=segment_width / len(active_users),
                    user_profile=user_profile,
                    tooltip_data={
                        'status': 'active',
                        'capability': user_profile.capability.value,
                        'reputation': user_profile.reputation_score
                    }
                )
                self.status_bar_segments.append(segment)
                current_position += segment.width
    
    async def _update_reputation_scores(self):
        """Update reputation scores with decay"""
        current_time = datetime.now()
        
        for user_profile in self.users.values():
            # Apply reputation decay for inactive users
            time_since_last_seen = current_time - user_profile.last_seen
            if time_since_last_seen > timedelta(hours=1):
                decay_factor = 1.0 - (self.reputation_decay_rate * time_since_last_seen.total_seconds() / 3600)
                new_reputation = user_profile.reputation_score * decay_factor
                user_profile.update_reputation(new_reputation)
    
    async def _update_metrics(self):
        """Update network-wide metrics"""
        if not self.users:
            return
        
        total_users = len(self.users)
        active_users = sum(1 for u in self.users.values() if u.is_available_for_queries())
        idle_users = sum(1 for u in self.users.values() if u.status == UserStatus.IDLE)
        high_reputation_servers = sum(1 for u in self.users.values() if u.is_high_reputation_server())
        
        reputation_scores = [u.reputation_score for u in self.users.values()]
        reliability_scores = [u.reliability_score for u in self.users.values()]
        
        # Calculate network health
        health_factors = []
        if total_users > 0:
            health_factors.append(active_users / total_users)  # Availability
            health_factors.append(sum(reputation_scores) / len(reputation_scores))  # Average reputation
            health_factors.append(sum(reliability_scores) / len(reliability_scores))  # Average reliability
        
        network_health = sum(health_factors) / len(health_factors) if health_factors else 0.0
        
        # Calculate query success rate
        total_queries = sum(u.total_queries_processed for u in self.users.values())
        successful_queries = sum(u.successful_queries for u in self.users.values())
        query_success_rate = successful_queries / total_queries if total_queries > 0 else 0.0
        
        # Calculate average response time
        response_times = [u.average_response_time for u in self.users.values() if u.average_response_time > 0]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Update metrics
        self.network_metrics = NetworkMetrics(
            total_users=total_users,
            active_users=active_users,
            idle_users=idle_users,
            high_reputation_servers=high_reputation_servers,
            average_reputation=sum(reputation_scores) / len(reputation_scores) if reputation_scores else 0.0,
            average_reliability=sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.0,
            network_health=network_health,
            query_success_rate=query_success_rate,
            average_response_time=average_response_time,
            last_updated=datetime.now()
        )
        
        # Record performance history
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'total_users': total_users,
            'active_users': active_users,
            'network_health': network_health,
            'query_success_rate': query_success_rate,
            'average_response_time': average_response_time
        }
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    async def _trigger_status_update(self):
        """Trigger status update callbacks"""
        for callback in self.status_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.get_status_bar_data())
                else:
                    callback(self.get_status_bar_data())
            except Exception as e:
                self.logger.error(f"Error in status update callback: {e}")
    
    async def _trigger_metrics_update(self):
        """Trigger metrics update callbacks"""
        for callback in self.metrics_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.network_metrics)
                else:
                    callback(self.network_metrics)
            except Exception as e:
                self.logger.error(f"Error in metrics update callback: {e}")
    
    def get_status_bar_data(self) -> Dict[str, Any]:
        """Get status bar visualization data"""
        return {
            'segments': [
                {
                    'segment_id': segment.segment_id,
                    'user_id': segment.user_id,
                    'segment_type': segment.segment_type,
                    'position': segment.position,
                    'width': segment.width,
                    'tooltip_data': segment.tooltip_data,
                    'user_info': {
                        'username': segment.user_profile.username,
                        'status': segment.user_profile.status.value,
                        'capability': segment.user_profile.capability.value,
                        'reputation': segment.user_profile.reputation_score,
                        'reliability': segment.user_profile.reliability_score
                    }
                }
                for segment in self.status_bar_segments
            ],
            'total_segments': len(self.status_bar_segments),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_network_metrics(self) -> NetworkMetrics:
        """Get current network metrics"""
        return self.network_metrics
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.users.get(user_id)
    
    def get_available_users(self) -> List[UserProfile]:
        """Get list of available users"""
        return [u for u in self.users.values() if u.is_available_for_queries()]
    
    def get_high_reputation_servers(self) -> List[UserProfile]:
        """Get list of high-reputation servers"""
        return [u for u in self.users.values() if u.is_high_reputation_server()]
    
    def find_best_server(self, requirements: Dict[str, Any]) -> Optional[UserProfile]:
        """
        Find the best server for a given query.
        
        Args:
            requirements: Query requirements (capability, region, etc.)
            
        Returns:
            Best matching user profile or None
        """
        available_users = self.get_available_users()
        if not available_users:
            return None
        
        # Filter by requirements
        filtered_users = available_users
        
        # Filter by capability
        if 'min_capability' in requirements:
            min_capability = requirements['min_capability']
            filtered_users = [u for u in filtered_users if u.capability.value >= min_capability.value]
        
        # Filter by region
        if 'preferred_region' in requirements:
            preferred_region = requirements['preferred_region']
            # Prefer users in the same region, but allow others
            filtered_users.sort(key=lambda u: 0 if u.region == preferred_region else 1)
        
        if not filtered_users:
            return None
        
        # Sort by overall score and return the best
        filtered_users.sort(key=lambda u: u.get_overall_score(), reverse=True)
        return filtered_users[0]
    
    def add_status_update_callback(self, callback: callable):
        """Add callback for status updates"""
        self.status_update_callbacks.append(callback)
    
    def add_metrics_update_callback(self, callback: callable):
        """Add callback for metrics updates"""
        self.metrics_update_callbacks.append(callback)
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history"""
        return self.performance_history.copy()
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get query history"""
        return self.query_history.copy()
    
    def save_network_state(self, filepath: str):
        """Save network state to file"""
        state = {
            'users': {
                uid: {
                    'user_id': profile.user_id,
                    'username': profile.username,
                    'status': profile.status.value,
                    'capability': profile.capability.value,
                    'region': profile.region.value,
                    'reputation_score': profile.reputation_score,
                    'reliability_score': profile.reliability_score,
                    'performance_metrics': profile.performance_metrics,
                    'resource_usage': profile.resource_usage,
                    'last_seen': profile.last_seen.isoformat(),
                    'uptime': profile.uptime.total_seconds(),
                    'total_queries_processed': profile.total_queries_processed,
                    'successful_queries': profile.successful_queries,
                    'failed_queries': profile.failed_queries,
                    'average_response_time': profile.average_response_time,
                    'metadata': profile.metadata
                }
                for uid, profile in self.users.items()
            },
            'network_metrics': {
                'total_users': self.network_metrics.total_users,
                'active_users': self.network_metrics.active_users,
                'idle_users': self.network_metrics.idle_users,
                'high_reputation_servers': self.network_metrics.high_reputation_servers,
                'average_reputation': self.network_metrics.average_reputation,
                'average_reliability': self.network_metrics.average_reliability,
                'network_health': self.network_metrics.network_health,
                'query_success_rate': self.network_metrics.query_success_rate,
                'average_response_time': self.network_metrics.average_response_time,
                'last_updated': self.network_metrics.last_updated.isoformat()
            },
            'performance_history': self.performance_history[-100:],  # Keep last 100 records
            'query_history': self.query_history[-100:]  # Keep last 100 records
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Network state saved to {filepath}")
    
    def load_network_state(self, filepath: str):
        """Load network state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load users
        self.users.clear()
        for uid, user_data in state.get('users', {}).items():
            user_profile = UserProfile(
                user_id=user_data['user_id'],
                username=user_data['username'],
                status=UserStatus(user_data['status']),
                capability=ServerCapability(user_data['capability']),
                region=NetworkRegion(user_data['region']),
                reputation_score=user_data['reputation_score'],
                reliability_score=user_data['reliability_score'],
                performance_metrics=user_data['performance_metrics'],
                resource_usage=user_data['resource_usage'],
                last_seen=datetime.fromisoformat(user_data['last_seen']),
                uptime=timedelta(seconds=user_data['uptime']),
                total_queries_processed=user_data['total_queries_processed'],
                successful_queries=user_data['successful_queries'],
                failed_queries=user_data['failed_queries'],
                average_response_time=user_data['average_response_time'],
                metadata=user_data['metadata']
            )
            self.users[uid] = user_profile
        
        # Load other state
        self.performance_history = state.get('performance_history', [])
        self.query_history = state.get('query_history', [])
        
        self.logger.info(f"Network state loaded from {filepath}")


# Import required enums
from enum import Enum
BrainLobe = Enum('BrainLobe', ['LEFT', 'RIGHT']) 