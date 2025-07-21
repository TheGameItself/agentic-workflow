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

# Add research tracking import
try:
    from .p2p_research_tracking import P2PResearchTracking, ResearchSource, ResearcherProfile, ResearchQuality, SourceType, ResearchDomain
    RESEARCH_TRACKING_AVAILABLE = True
except ImportError:
    RESEARCH_TRACKING_AVAILABLE = False
    P2PResearchTracking = None
    ResearchSource = None
    ResearcherProfile = None
    ResearchQuality = None
    SourceType = None
    ResearchDomain = None


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
    
    # Add research tracking fields
    research_reputation: float = 0.5
    research_contributions: int = 0
    expertise_domains: List[str] = field(default_factory=list)
    research_activity_streak: int = 0
    
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
    
    def update_research_reputation(self, new_score: float):
        """Update research reputation score"""
        self.research_reputation = max(0.0, min(1.0, new_score))
    
    def get_overall_score(self) -> float:
        """Calculate overall capability score including research reputation"""
        # Weighted combination of reputation, reliability, capability, and research reputation
        capability_weight = {
            ServerCapability.BASIC: 0.2,
            ServerCapability.STANDARD: 0.4,
            ServerCapability.ADVANCED: 0.6,
            ServerCapability.EXPERT: 0.8,
            ServerCapability.MASTER: 1.0
        }
        
        capability_score = capability_weight.get(self.capability, 0.2)
        
        overall_score = (
            self.reputation_score * 0.3 +
            self.reliability_score * 0.25 +
            capability_score * 0.25 +
            self.research_reputation * 0.2
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
    
    def is_research_expert(self) -> bool:
        """Check if user is a research expert"""
        return (self.research_reputation >= 0.8 and 
                self.research_contributions >= 10 and
                self.research_activity_streak >= 7)


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
    
    # Add research tracking metrics
    total_research_sources: int = 0
    high_quality_sources: int = 0
    active_researchers: int = 0
    average_research_reputation: float = 0.0
    research_experts: int = 0


class P2PNetworkIntegration:
    """
    P2P network integration with comprehensive status visualization and research tracking.
    
    Features:
    - Real-time user status tracking
    - Reputation scoring and capability assessment
    - Status bar visualization with color-coded segments
    - Network health monitoring
    - Query routing and load balancing
    - Research source tracking and quality assessment
    - Research reputation management
    """
    
    def __init__(self, 
                 max_users: int = 1000,
                 update_interval: float = 1.0,
                 reputation_decay_rate: float = 0.01,
                 enable_research_tracking: bool = True):
        self.max_users = max_users
        self.update_interval = update_interval
        self.reputation_decay_rate = reputation_decay_rate
        self.enable_research_tracking = enable_research_tracking and RESEARCH_TRACKING_AVAILABLE
        
        self.logger = logging.getLogger("P2PNetworkIntegration")
        
        # Core components
        self.users: Dict[str, UserProfile] = {}
        self.status_bar_segments: List[StatusBarSegment] = []
        self.network_metrics = NetworkMetrics()
        
        # Research tracking integration
        self.research_tracking: Optional[P2PResearchTracking] = None
        if self.enable_research_tracking:
            self.research_tracking = P2PResearchTracking(p2p_network=self)
            self.research_tracking.add_network_callback(self._on_research_metrics_update)
        
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
                     region: NetworkRegion = NetworkRegion.LOCAL,
                     expertise_domains: Optional[List[str]] = None) -> bool:
        """Register a new user with research expertise tracking"""
        if user_id in self.users:
            self.logger.warning(f"User {user_id} already registered")
            return False
        
        if len(self.users) >= self.max_users:
            self.logger.warning("Maximum users reached")
            return False
        
        user = UserProfile(
            user_id=user_id,
            username=username,
            capability=capability,
            region=region,
            expertise_domains=expertise_domains or []
        )
        
        self.users[user_id] = user
        
        # Register with research tracking if available
        if self.research_tracking:
            try:
                # Create researcher profile
                researcher_id = f"researcher_{user_id}"
                self.research_tracking.researcher_profiles[researcher_id] = ResearcherProfile(
                    researcher_id=researcher_id,
                    username=username,
                    user_id=user_id,
                    expertise_domains=[ResearchDomain(d) for d in expertise_domains] if expertise_domains else []
                )
            except Exception as e:
                self.logger.error(f"Error registering user with research tracking: {e}")
        
        self.logger.info(f"Registered user: {username} (ID: {user_id})")
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
                          result_quality: float = 1.0,
                          research_sources: Optional[List[str]] = None) -> bool:
        """Record query result with research source tracking"""
        if user_id not in self.users:
            self.logger.error(f"User {user_id} not found")
            return False
        
        user = self.users[user_id]
        
        # Update basic metrics
        user.total_queries_processed += 1
        if success:
            user.successful_queries += 1
        else:
            user.failed_queries += 1
        
        # Update response time
        if user.average_response_time == 0.0:
            user.average_response_time = response_time
        else:
            user.average_response_time = (user.average_response_time * 0.9 + response_time * 0.1)
        
        # Update reliability score
        success_rate = user.successful_queries / user.total_queries_processed
        user.reliability_score = success_rate
        
        # Update research tracking if sources provided
        if research_sources and self.research_tracking:
            try:
                researcher_id = f"researcher_{user_id}"
                if researcher_id in self.research_tracking.researcher_profiles:
                    researcher = self.research_tracking.researcher_profiles[researcher_id]
                    researcher.network_contributions += 1
                    researcher.last_activity = datetime.now()
                    
                    # Update research reputation based on result quality
                    if result_quality > 0.8:
                        researcher.research_reputation = min(1.0, researcher.research_reputation + 0.01)
                    elif result_quality < 0.5:
                        researcher.research_reputation = max(0.0, researcher.research_reputation - 0.01)
                    
                    # Update user's research reputation
                    user.research_reputation = researcher.research_reputation
                    user.research_contributions = researcher.network_contributions
            except Exception as e:
                self.logger.error(f"Error updating research tracking: {e}")
        
        self.logger.debug(f"Recorded query result for {user_id}: success={success}, quality={result_quality}")
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
        """Update network metrics including research tracking"""
        if not self.users:
            return
        
        # Update basic metrics
        self.network_metrics.total_users = len(self.users)
        self.network_metrics.active_users = len([u for u in self.users.values() if u.status == UserStatus.ACTIVE])
        self.network_metrics.idle_users = len([u for u in self.users.values() if u.status == UserStatus.IDLE])
        self.network_metrics.high_reputation_servers = len([u for u in self.users.values() if u.is_high_reputation_server()])
        
        # Calculate averages
        reputations = [u.reputation_score for u in self.users.values()]
        reliabilities = [u.reliability_score for u in self.users.values()]
        response_times = [u.average_response_time for u in self.users.values() if u.average_response_time > 0]
        
        self.network_metrics.average_reputation = np.mean(reputations) if reputations else 0.0
        self.network_metrics.average_reliability = np.mean(reliabilities) if reliabilities else 0.0
        self.network_metrics.average_response_time = np.mean(response_times) if response_times else 0.0
        
        # Calculate success rate
        total_queries = sum(u.total_queries_processed for u in self.users.values())
        successful_queries = sum(u.successful_queries for u in self.users.values())
        self.network_metrics.query_success_rate = successful_queries / total_queries if total_queries > 0 else 0.0
        
        # Calculate network health
        health_factors = [
            self.network_metrics.query_success_rate,
            self.network_metrics.average_reliability,
            min(1.0, self.network_metrics.active_users / max(self.network_metrics.total_users, 1))
        ]
        self.network_metrics.network_health = np.mean(health_factors)
        
        # Update research tracking metrics
        if self.research_tracking:
            research_stats = self.research_tracking.get_research_network_stats()
            self.network_metrics.total_research_sources = research_stats['total_sources']
            self.network_metrics.high_quality_sources = research_stats['high_quality_sources']
            self.network_metrics.active_researchers = research_stats['active_researchers']
            self.network_metrics.average_research_reputation = research_stats['average_research_reputation']
            self.network_metrics.research_experts = len([u for u in self.users.values() if u.is_research_expert()])
        
        self.network_metrics.last_updated = datetime.now()
        
        # Record performance history
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'total_users': self.network_metrics.total_users,
            'active_users': self.network_metrics.active_users,
            'network_health': self.network_metrics.network_health,
            'query_success_rate': self.network_metrics.query_success_rate,
            'average_response_time': self.network_metrics.average_response_time
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

    def _on_research_metrics_update(self, research_metrics: Dict[str, Any]):
        """Callback for research metrics updates"""
        self.logger.debug(f"Research metrics updated: {research_metrics}")
        # Additional processing can be added here if needed

    # Add new research tracking methods

    def add_research_source(self, 
                           title: str,
                           authors: List[str],
                           source_type: str,
                           domain: str,
                           url: Optional[str] = None,
                           doi: Optional[str] = None,
                           user_id: Optional[str] = None,
                           **kwargs) -> Optional[str]:
        """Add a research source through the P2P network"""
        if not self.research_tracking:
            self.logger.warning("Research tracking not enabled")
            return None
        
        try:
            source_type_enum = SourceType(source_type)
            domain_enum = ResearchDomain(domain)
            
            source_id = self.research_tracking.add_research_source(
                title=title,
                authors=authors,
                source_type=source_type_enum,
                domain=domain_enum,
                url=url,
                doi=doi,
                created_by=f"researcher_{user_id}" if user_id else None,
                **kwargs
            )
            
            # Update user's research contributions
            if user_id and user_id in self.users:
                self.users[user_id].research_contributions += 1
                self.users[user_id].research_activity_streak += 1
            
            return source_id
        except Exception as e:
            self.logger.error(f"Error adding research source: {e}")
            return None

    def validate_research_source(self,
                                source_id: str,
                                user_id: str,
                                validation_type: str,
                                confidence: float,
                                reasoning: str) -> bool:
        """Validate or dispute a research source"""
        if not self.research_tracking:
            self.logger.warning("Research tracking not enabled")
            return False
        
        try:
            researcher_id = f"researcher_{user_id}"
            success = self.research_tracking.validate_research_source(
                source_id=source_id,
                researcher_id=researcher_id,
                validation_type=validation_type,
                confidence=confidence,
                reasoning=reasoning
            )
            
            if success and user_id in self.users:
                self.users[user_id].research_contributions += 1
                self.users[user_id].research_activity_streak += 1
            
            return success
        except Exception as e:
            self.logger.error(f"Error validating research source: {e}")
            return False

    def get_research_sources(self,
                           domain: Optional[str] = None,
                           quality_level: Optional[str] = None,
                           source_type: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get research sources with filtering"""
        if not self.research_tracking:
            return []
        
        try:
            domain_enum = ResearchDomain(domain) if domain else None
            quality_enum = ResearchQuality(quality_level) if quality_level else None
            source_type_enum = SourceType(source_type) if source_type else None
            
            sources = self.research_tracking.get_research_sources(
                domain=domain_enum,
                quality_level=quality_enum,
                source_type=source_type_enum,
                limit=limit
            )
            
            # Convert to dictionary format for API
            return [
                {
                    'source_id': s.source_id,
                    'title': s.title,
                    'authors': s.authors,
                    'source_type': s.source_type.value,
                    'domain': s.domain.value,
                    'url': s.url,
                    'doi': s.doi,
                    'quality_score': s.quality_score,
                    'quality_level': s.quality_level.value,
                    'network_reputation': s.network_reputation,
                    'validation_count': s.validation_count,
                    'dispute_count': s.dispute_count,
                    'last_updated': s.last_updated.isoformat()
                }
                for s in sources
            ]
        except Exception as e:
            self.logger.error(f"Error getting research sources: {e}")
            return []

    def get_research_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive research network statistics"""
        if not self.research_tracking:
            return {}
        
        try:
            return self.research_tracking.get_research_network_stats()
        except Exception as e:
            self.logger.error(f"Error getting research network stats: {e}")
            return {}

    def get_top_researchers(self, domain: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top researchers by reputation"""
        if not self.research_tracking:
            return []
        
        try:
            domain_enum = ResearchDomain(domain) if domain else None
            researchers = self.research_tracking.get_top_researchers(domain=domain_enum, limit=limit)
            
            return [
                {
                    'researcher_id': r.researcher_id,
                    'username': r.username,
                    'user_id': r.user_id,
                    'research_reputation': r.research_reputation,
                    'total_sources': r.total_sources,
                    'validated_sources': r.validated_sources,
                    'expertise_domains': [d.value for d in r.expertise_domains],
                    'validation_accuracy': r.validation_accuracy,
                    'network_contributions': r.network_contributions,
                    'activity_streak': r.activity_streak,
                    'last_activity': r.last_activity.isoformat()
                }
                for r in researchers
            ]
        except Exception as e:
            self.logger.error(f"Error getting top researchers: {e}")
            return []

    def get_research_trends(self, domain: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get research trends and analysis"""
        if not self.research_tracking:
            return {}
        
        try:
            domain_enum = ResearchDomain(domain) if domain else None
            return self.research_tracking.get_research_trends(domain=domain_enum, days=days)
        except Exception as e:
            self.logger.error(f"Error getting research trends: {e}")
            return {}


# Import required enums
from enum import Enum
BrainLobe = Enum('BrainLobe', ['LEFT', 'RIGHT']) 