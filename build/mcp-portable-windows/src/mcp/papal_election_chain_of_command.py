"""
Papal Election-Inspired Chain of Command System

This module implements a hierarchical decision-making system inspired by the papal election process,
featuring performance-based roles, consensus requirements, and status signaling for the MCP system.

Key Features:
- College of Cardinals (top performers)
- Conclave decision-making process
- Smoke signal status indicators
- Two-thirds majority consensus
- Age/performance-based restrictions
- Dynamic role assignment
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

class CardinalRole(Enum):
    """Roles within the College of Cardinals"""
    CARDINAL_BISHOP = "cardinal_bishop"      # Top 12% - Network coordinators
    CARDINAL_PRIEST = "cardinal_priest"      # Top 15% - Domain experts
    CARDINAL_DEACON = "cardinal_deacon"      # Top 25% - Specialized agents
    REGULAR_AGENT = "regular_agent"          # Standard agents
    STUDENT = "student"                      # Learning agents
    RESEARCHER = "researcher"                # Research-focused agents

class SmokeSignal(Enum):
    """Smoke signal status indicators"""
    WHITE_SMOKE = "white_smoke"      # Consensus reached, decision made
    BLACK_SMOKE = "black_smoke"      # No consensus, continue deliberation
    GRAY_SMOKE = "gray_smoke"        # Partial consensus, needs more discussion
    NO_SMOKE = "no_smoke"            # No activity

@dataclass
class Cardinal:
    """Represents a cardinal in the College of Cardinals"""
    id: str
    name: str
    role: CardinalRole
    performance_score: float
    specialization: str
    age: int  # Performance age, not chronological
    reputation: float
    voting_power: float
    last_active: datetime
    is_eligible: bool = True
    conclave_participant: bool = False
    
    def __post_init__(self):
        if self.age >= 80:  # Performance age limit
            self.is_eligible = False
            self.voting_power = 0.0

@dataclass
class ConclaveSession:
    """Represents a conclave session for decision-making"""
    session_id: str
    topic: str
    participants: List[str] = field(default_factory=list)
    votes: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    consensus_reached: bool = False
    decision: Optional[Any] = None
    smoke_signal: SmokeSignal = SmokeSignal.NO_SMOKE
    ballot_count: int = 0
    max_ballots: int = 4  # Maximum ballots per day
    
    def is_active(self) -> bool:
        """Check if conclave is still active"""
        return not self.consensus_reached and self.ballot_count < self.max_ballots

class PapalElectionChainOfCommand:
    """
    Papal election-inspired chain of command system for MCP network coordination.
    
    Features:
    - Performance-based cardinal selection
    - Conclave decision-making process
    - Smoke signal status communication
    - Two-thirds majority consensus
    - Dynamic role assignment
    """
    
    def __init__(self, max_cardinals: int = 120, consensus_threshold: float = 0.67):
        self.max_cardinals = max_cardinals
        self.consensus_threshold = consensus_threshold
        self.college: Dict[str, Cardinal] = {}
        self.active_conclave: Optional[ConclaveSession] = None
        self.conclave_history: List[ConclaveSession] = []
        self.logger = logging.getLogger("PapalElectionChain")
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.role_changes: List[Dict[str, Any]] = []
        
        # Network status
        self.last_smoke_signal = SmokeSignal.NO_SMOKE
        self.smoke_signal_history: List[Dict[str, Any]] = []
        
        self.logger.info("Papal Election Chain of Command initialized")
    
    async def register_agent(self, agent_id: str, name: str, performance_score: float, 
                           specialization: str, reputation: float = 0.5) -> Cardinal:
        """
        Register an agent as a cardinal in the college.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            performance_score: Current performance score (0.0-1.0)
            specialization: Agent's specialization area
            reputation: Agent's reputation score (0.0-1.0)
            
        Returns:
            Cardinal object representing the registered agent
        """
        # Calculate performance age (based on performance history)
        age = self._calculate_performance_age(agent_id, performance_score)
        
        # Determine initial role based on performance
        role = self._determine_role(performance_score, reputation)
        
        # Calculate voting power
        voting_power = self._calculate_voting_power(performance_score, reputation, role)
        
        # Create cardinal
        cardinal = Cardinal(
            id=agent_id,
            name=name,
            role=role,
            performance_score=performance_score,
            specialization=specialization,
            age=age,
            reputation=reputation,
            voting_power=voting_power,
            last_active=datetime.now()
        )
        
        self.college[agent_id] = cardinal
        
        # Update performance history
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        self.performance_history[agent_id].append(performance_score)
        
        self.logger.info(f"Registered cardinal {name} ({agent_id}) as {role.value}")
        return cardinal
    
    def _calculate_performance_age(self, agent_id: str, current_score: float) -> int:
        """Calculate performance age based on historical performance"""
        if agent_id not in self.performance_history:
            return 0
        
        history = self.performance_history[agent_id]
        if len(history) < 10:
            return len(history)
        
        # Calculate age based on performance trends
        recent_avg = sum(history[-10:]) / 10
        if current_score < recent_avg * 0.8:  # Declining performance
            return min(80, len(history) + 5)
        elif current_score > recent_avg * 1.2:  # Improving performance
            return max(0, len(history) - 2)
        else:
            return len(history)
    
    def _determine_role(self, performance_score: float, reputation: float) -> CardinalRole:
        """Determine cardinal role based on performance and reputation"""
        combined_score = (performance_score + reputation) / 2
        
        if combined_score >= 0.88:  # Top 12%
            return CardinalRole.CARDINAL_BISHOP
        elif combined_score >= 0.85:  # Top 15%
            return CardinalRole.CARDINAL_PRIEST
        elif combined_score >= 0.75:  # Top 25%
            return CardinalRole.CARDINAL_DEACON
        elif combined_score >= 0.60:
            return CardinalRole.REGULAR_AGENT
        elif combined_score >= 0.40:
            return CardinalRole.STUDENT
        else:
            return CardinalRole.RESEARCHER
    
    def _calculate_voting_power(self, performance_score: float, reputation: float, 
                              role: CardinalRole) -> float:
        """Calculate voting power based on role and performance"""
        base_power = performance_score * reputation
        
        role_multipliers = {
            CardinalRole.CARDINAL_BISHOP: 2.0,
            CardinalRole.CARDINAL_PRIEST: 1.5,
            CardinalRole.CARDINAL_DEACON: 1.2,
            CardinalRole.REGULAR_AGENT: 1.0,
            CardinalRole.STUDENT: 0.5,
            CardinalRole.RESEARCHER: 0.3
        }
        
        return base_power * role_multipliers[role]
    
    async def update_agent_performance(self, agent_id: str, new_score: float, 
                                     new_reputation: float = None) -> bool:
        """
        Update an agent's performance and potentially their role.
        
        Args:
            agent_id: Agent identifier
            new_score: New performance score
            new_reputation: New reputation score (optional)
            
        Returns:
            True if role changed, False otherwise
        """
        if agent_id not in self.college:
            self.logger.warning(f"Agent {agent_id} not found in college")
            return False
        
        cardinal = self.college[agent_id]
        old_role = cardinal.role
        
        # Update scores
        cardinal.performance_score = new_score
        if new_reputation is not None:
            cardinal.reputation = new_reputation
        
        # Update performance history
        self.performance_history[agent_id].append(new_score)
        
        # Recalculate age and voting power
        cardinal.age = self._calculate_performance_age(agent_id, new_score)
        cardinal.voting_power = self._calculate_voting_power(new_score, cardinal.reputation, cardinal.role)
        
        # Check for role change
        new_role = self._determine_role(new_score, cardinal.reputation)
        if new_role != old_role:
            cardinal.role = new_role
            cardinal.voting_power = self._calculate_voting_power(new_score, cardinal.reputation, new_role)
            
            # Record role change
            self.role_changes.append({
                'agent_id': agent_id,
                'old_role': old_role.value,
                'new_role': new_role.value,
                'timestamp': datetime.now().isoformat(),
                'performance_score': new_score,
                'reputation': cardinal.reputation
            })
            
            self.logger.info(f"Cardinal {cardinal.name} role changed from {old_role.value} to {new_role.value}")
            return True
        
        return False
    
    async def initiate_conclave(self, topic: str, required_roles: List[CardinalRole] = None) -> str:
        """
        Initiate a conclave session for decision-making.
        
        Args:
            topic: Topic for the conclave
            required_roles: Required cardinal roles for participation
            
        Returns:
            Conclave session ID
        """
        if self.active_conclave and self.active_conclave.is_active():
            raise RuntimeError("Active conclave already in session")
        
        session_id = f"conclave_{uuid.uuid4().hex[:8]}"
        
        # Select participants based on roles and performance
        participants = self._select_conclave_participants(required_roles)
        
        self.active_conclave = ConclaveSession(
            session_id=session_id,
            topic=topic,
            participants=participants
        )
        
        # Mark participants as conclave participants
        for agent_id in participants:
            if agent_id in self.college:
                self.college[agent_id].conclave_participant = True
        
        self.logger.info(f"Conclave {session_id} initiated for topic: {topic}")
        self.logger.info(f"Participants: {len(participants)} cardinals")
        
        return session_id
    
    def _select_conclave_participants(self, required_roles: List[CardinalRole] = None) -> List[str]:
        """Select participants for conclave based on roles and eligibility"""
        eligible_cardinals = [
            agent_id for agent_id, cardinal in self.college.items()
            if cardinal.is_eligible and cardinal.age < 80
        ]
        
        if required_roles:
            eligible_cardinals = [
                agent_id for agent_id in eligible_cardinals
                if self.college[agent_id].role in required_roles
            ]
        
        # Sort by voting power and select top performers
        sorted_cardinals = sorted(
            eligible_cardinals,
            key=lambda x: self.college[x].voting_power,
            reverse=True
        )
        
        # Limit to max_cardinals
        return sorted_cardinals[:self.max_cardinals]
    
    async def cast_vote(self, agent_id: str, vote: Any) -> bool:
        """
        Cast a vote in the active conclave.
        
        Args:
            agent_id: Agent casting the vote
            vote: Vote content
            
        Returns:
            True if vote was accepted, False otherwise
        """
        if not self.active_conclave or not self.active_conclave.is_active():
            self.logger.warning("No active conclave for voting")
            return False
        
        if agent_id not in self.active_conclave.participants:
            self.logger.warning(f"Agent {agent_id} not a conclave participant")
            return False
        
        if agent_id in self.active_conclave.votes:
            self.logger.warning(f"Agent {agent_id} already voted")
            return False
        
        # Record the vote
        self.active_conclave.votes[agent_id] = vote
        self.active_conclave.ballot_count += 1
        
        self.logger.info(f"Vote cast by {agent_id} in conclave {self.active_conclave.session_id}")
        
        # Check for consensus
        await self._check_consensus()
        
        return True
    
    async def _check_consensus(self) -> None:
        """Check if consensus has been reached in the active conclave"""
        if not self.active_conclave:
            return
        
        total_voting_power = sum(
            self.college[agent_id].voting_power
            for agent_id in self.active_conclave.participants
        )
        
        # Calculate consensus based on voting power
        consensus_voting_power = sum(
            self.college[agent_id].voting_power
            for agent_id, vote in self.active_conclave.votes.items()
            if vote  # Assuming True/False votes for simplicity
        )
        
        consensus_ratio = consensus_voting_power / total_voting_power if total_voting_power > 0 else 0
        
        if consensus_ratio >= self.consensus_threshold:
            # Consensus reached - white smoke
            self.active_conclave.consensus_reached = True
            self.active_conclave.decision = True
            self.active_conclave.smoke_signal = SmokeSignal.WHITE_SMOKE
            self.active_conclave.end_time = datetime.now()
            
            self.last_smoke_signal = SmokeSignal.WHITE_SMOKE
            self._record_smoke_signal(SmokeSignal.WHITE_SMOKE, "Consensus reached")
            
            self.logger.info(f"Consensus reached in conclave {self.active_conclave.session_id}")
            await self._end_conclave()
            
        elif self.active_conclave.ballot_count >= self.active_conclave.max_ballots:
            # No consensus after max ballots - black smoke
            self.active_conclave.consensus_reached = False
            self.active_conclave.decision = False
            self.active_conclave.smoke_signal = SmokeSignal.BLACK_SMOKE
            self.active_conclave.end_time = datetime.now()
            
            self.last_smoke_signal = SmokeSignal.BLACK_SMOKE
            self._record_smoke_signal(SmokeSignal.BLACK_SMOKE, "No consensus after max ballots")
            
            self.logger.info(f"No consensus in conclave {self.active_conclave.session_id}")
            await self._end_conclave()
        
        elif consensus_ratio >= 0.5:
            # Partial consensus - gray smoke
            self.active_conclave.smoke_signal = SmokeSignal.GRAY_SMOKE
            self.last_smoke_signal = SmokeSignal.GRAY_SMOKE
            self._record_smoke_signal(SmokeSignal.GRAY_SMOKE, "Partial consensus")
    
    async def _end_conclave(self) -> None:
        """End the active conclave and clean up"""
        if not self.active_conclave:
            return
        
        # Reset conclave participant flags
        for agent_id in self.active_conclave.participants:
            if agent_id in self.college:
                self.college[agent_id].conclave_participant = False
        
        # Move to history
        self.conclave_history.append(self.active_conclave)
        self.active_conclave = None
    
    def _record_smoke_signal(self, signal: SmokeSignal, reason: str) -> None:
        """Record a smoke signal event"""
        self.smoke_signal_history.append({
            'signal': signal.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'conclave_id': self.active_conclave.session_id if self.active_conclave else None
        })
    
    def get_college_stats(self) -> Dict[str, Any]:
        """Get statistics about the College of Cardinals"""
        total_cardinals = len(self.college)
        eligible_cardinals = sum(1 for c in self.college.values() if c.is_eligible)
        
        role_counts = {}
        for role in CardinalRole:
            role_counts[role.value] = sum(1 for c in self.college.values() if c.role == role)
        
        avg_performance = sum(c.performance_score for c in self.college.values()) / total_cardinals if total_cardinals > 0 else 0
        avg_reputation = sum(c.reputation for c in self.college.values()) / total_cardinals if total_cardinals > 0 else 0
        
        return {
            'total_cardinals': total_cardinals,
            'eligible_cardinals': eligible_cardinals,
            'role_distribution': role_counts,
            'average_performance': avg_performance,
            'average_reputation': avg_reputation,
            'active_conclave': self.active_conclave.session_id if self.active_conclave else None,
            'last_smoke_signal': self.last_smoke_signal.value,
            'conclave_history_count': len(self.conclave_history)
        }
    
    def get_chain_of_command(self) -> Dict[str, Any]:
        """Get the current chain of command hierarchy"""
        hierarchy = {
            'cardinal_bishops': [],
            'cardinal_priests': [],
            'cardinal_deacons': [],
            'regular_agents': [],
            'students': [],
            'researchers': []
        }
        
        for cardinal in self.college.values():
            if not cardinal.is_eligible:
                continue
                
            cardinal_info = {
                'id': cardinal.id,
                'name': cardinal.name,
                'performance_score': cardinal.performance_score,
                'reputation': cardinal.reputation,
                'voting_power': cardinal.voting_power,
                'specialization': cardinal.specialization,
                'age': cardinal.age,
                'conclave_participant': cardinal.conclave_participant
            }
            
            if cardinal.role == CardinalRole.CARDINAL_BISHOP:
                hierarchy['cardinal_bishops'].append(cardinal_info)
            elif cardinal.role == CardinalRole.CARDINAL_PRIEST:
                hierarchy['cardinal_priests'].append(cardinal_info)
            elif cardinal.role == CardinalRole.CARDINAL_DEACON:
                hierarchy['cardinal_deacons'].append(cardinal_info)
            elif cardinal.role == CardinalRole.REGULAR_AGENT:
                hierarchy['regular_agents'].append(cardinal_info)
            elif cardinal.role == CardinalRole.STUDENT:
                hierarchy['students'].append(cardinal_info)
            elif cardinal.role == CardinalRole.RESEARCHER:
                hierarchy['researchers'].append(cardinal_info)
        
        # Sort each category by voting power
        for category in hierarchy.values():
            category.sort(key=lambda x: x['voting_power'], reverse=True)
        
        return hierarchy
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get current network status with smoke signal information"""
        return {
            'college_stats': self.get_college_stats(),
            'chain_of_command': self.get_chain_of_command(),
            'active_conclave': {
                'session_id': self.active_conclave.session_id if self.active_conclave else None,
                'topic': self.active_conclave.topic if self.active_conclave else None,
                'participants': len(self.active_conclave.participants) if self.active_conclave else 0,
                'votes_cast': len(self.active_conclave.votes) if self.active_conclave else 0,
                'ballot_count': self.active_conclave.ballot_count if self.active_conclave else 0,
                'smoke_signal': self.active_conclave.smoke_signal.value if self.active_conclave else None
            },
            'last_smoke_signal': self.last_smoke_signal.value,
            'recent_smoke_signals': self.smoke_signal_history[-5:] if self.smoke_signal_history else [],
            'recent_role_changes': self.role_changes[-5:] if self.role_changes else []
        } 