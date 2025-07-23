#!/usr/bin/env python3
"""
P2P Research Tracking System

This module implements comprehensive research source tracking, quality assessment,
and reputation management across the P2P network. It integrates with the existing
P2P network to provide research intelligence and collaborative knowledge sharing.

Features:
- Research source tracking and metadata management
- Quality assessment with multi-dimensional scoring
- Reputation system for researchers and sources
- Collaborative research validation
- Cross-network knowledge sharing
- Research trend analysis and prediction
"""

import asyncio
import logging
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import uuid
import re
from urllib.parse import urlparse
import requests
from concurrent.futures import ThreadPoolExecutor

from .p2p_network_integration import P2PNetworkIntegration, UserProfile, UserStatus


class ResearchQuality(Enum):
    """Research quality levels"""
    UNVERIFIED = "unverified"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent"
    PEER_REVIEWED = "peer_reviewed"


class SourceType(Enum):
    """Research source types"""
    ACADEMIC_PAPER = "academic_paper"
    CONFERENCE_PAPER = "conference_paper"
    JOURNAL_ARTICLE = "journal_article"
    PREPRINT = "preprint"
    TECHNICAL_REPORT = "technical_report"
    BLOG_POST = "blog_post"
    NEWS_ARTICLE = "news_article"
    WEBSITE = "website"
    BOOK = "book"
    THESIS = "thesis"
    PATENT = "patent"
    DATASET = "dataset"
    CODE_REPOSITORY = "code_repository"
    VIDEO = "video"
    PODCAST = "podcast"
    SOCIAL_MEDIA = "social_media"


class ResearchDomain(Enum):
    """Research domains"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    MACHINE_LEARNING = "machine_learning"
    NEUROSCIENCE = "neuroscience"
    COGNITIVE_SCIENCE = "cognitive_science"
    COMPUTER_SCIENCE = "computer_science"
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    BIOLOGY = "biology"
    PSYCHOLOGY = "psychology"
    PHILOSOPHY = "philosophy"
    ENGINEERING = "engineering"
    MEDICINE = "medicine"
    ECONOMICS = "economics"
    SOCIOLOGY = "sociology"
    OTHER = "other"


@dataclass
class ResearchSource:
    """Research source with comprehensive metadata"""
    source_id: str
    title: str
    authors: List[str]
    source_type: SourceType
    domain: ResearchDomain
    url: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    journal: Optional[str] = None
    conference: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    citations: int = 0
    downloads: int = 0
    views: int = 0
    
    # Quality metrics
    quality_score: float = 0.0
    quality_level: ResearchQuality = ResearchQuality.UNVERIFIED
    peer_reviewed: bool = False
    impact_factor: Optional[float] = None
    h_index: Optional[int] = None
    
    # Network metrics
    network_reputation: float = 0.0
    validation_count: int = 0
    dispute_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    validated_by: List[str] = field(default_factory=list)
    disputed_by: List[str] = field(default_factory=list)
    
    # Content analysis
    content_hash: Optional[str] = None
    word_count: int = 0
    reference_count: int = 0
    methodology_score: float = 0.0
    reproducibility_score: float = 0.0
    novelty_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Source type weight
        type_weights = {
            SourceType.PEER_REVIEWED: 1.0,
            SourceType.ACADEMIC_PAPER: 0.9,
            SourceType.JOURNAL_ARTICLE: 0.85,
            SourceType.CONFERENCE_PAPER: 0.8,
            SourceType.TECHNICAL_REPORT: 0.7,
            SourceType.PREPRINT: 0.6,
            SourceType.BOOK: 0.75,
            SourceType.THESIS: 0.7,
            SourceType.PATENT: 0.6,
            SourceType.DATASET: 0.8,
            SourceType.CODE_REPOSITORY: 0.7,
            SourceType.BLOG_POST: 0.4,
            SourceType.NEWS_ARTICLE: 0.3,
            SourceType.WEBSITE: 0.3,
            SourceType.VIDEO: 0.5,
            SourceType.PODCAST: 0.4,
            SourceType.SOCIAL_MEDIA: 0.2
        }
        
        type_score = type_weights.get(self.source_type, 0.5)
        scores.append(type_score)
        
        # Citation impact
        if self.citations > 0:
            citation_score = min(1.0, np.log10(self.citations + 1) / 3.0)
            scores.append(citation_score)
        
        # Network reputation
        scores.append(self.network_reputation)
        
        # Validation ratio
        total_checks = self.validation_count + self.dispute_count
        if total_checks > 0:
            validation_ratio = self.validation_count / total_checks
            scores.append(validation_ratio)
        
        # Methodology and reproducibility
        scores.append(self.methodology_score)
        scores.append(self.reproducibility_score)
        
        # Peer review bonus
        if self.peer_reviewed:
            scores.append(0.2)
        
        # Calculate weighted average
        weights = [0.2, 0.15, 0.25, 0.15, 0.1, 0.1, 0.05]
        final_score = np.average(scores[:len(weights)], weights=weights[:len(scores)])
        
        self.quality_score = max(0.0, min(1.0, final_score))
        
        # Update quality level
        if self.quality_score >= 0.9:
            self.quality_level = ResearchQuality.PEER_REVIEWED
        elif self.quality_score >= 0.8:
            self.quality_level = ResearchQuality.EXCELLENT
        elif self.quality_score >= 0.7:
            self.quality_level = ResearchQuality.HIGH
        elif self.quality_score >= 0.5:
            self.quality_level = ResearchQuality.MEDIUM
        elif self.quality_score >= 0.3:
            self.quality_level = ResearchQuality.LOW
        else:
            self.quality_level = ResearchQuality.UNVERIFIED
        
        return self.quality_score


@dataclass
class ResearcherProfile:
    """Researcher profile with reputation tracking"""
    researcher_id: str
    username: str
    user_id: str  # Links to P2P network user
    
    # Research metrics
    total_sources: int = 0
    validated_sources: int = 0
    disputed_sources: int = 0
    research_reputation: float = 0.5
    expertise_domains: List[ResearchDomain] = field(default_factory=list)
    
    # Quality metrics
    average_quality_score: float = 0.0
    validation_accuracy: float = 0.5
    dispute_accuracy: float = 0.5
    
    # Network metrics
    network_contributions: int = 0
    collaboration_count: int = 0
    citations_received: int = 0
    
    # Activity tracking
    last_activity: datetime = field(default_factory=datetime.now)
    activity_streak: int = 0
    total_activity_time: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_reputation(self, validation_result: bool, was_correct: bool):
        """Update reputation based on validation/dispute accuracy"""
        if validation_result:
            self.validated_sources += 1
            if was_correct:
                self.validation_accuracy = (self.validation_accuracy * (self.validated_sources - 1) + 1.0) / self.validated_sources
            else:
                self.validation_accuracy = (self.validation_accuracy * (self.validated_sources - 1) + 0.0) / self.validated_sources
        else:
            self.disputed_sources += 1
            if was_correct:
                self.dispute_accuracy = (self.dispute_accuracy * (self.disputed_sources - 1) + 1.0) / self.disputed_sources
            else:
                self.dispute_accuracy = (self.dispute_accuracy * (self.disputed_sources - 1) + 0.0) / self.disputed_sources
        
        self.total_sources = self.validated_sources + self.disputed_sources
        
        # Calculate overall research reputation
        if self.total_sources > 0:
            accuracy = (self.validation_accuracy + self.dispute_accuracy) / 2.0
            activity_factor = min(1.0, self.activity_streak / 30.0)  # Normalize to 30 days
            self.research_reputation = (accuracy * 0.7 + activity_factor * 0.3)


@dataclass
class ResearchValidation:
    """Research validation record"""
    validation_id: str
    source_id: str
    researcher_id: str
    validation_type: str  # 'validate' or 'dispute'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class P2PResearchTracking:
    """
    P2P Research Tracking System
    
    Features:
    - Research source tracking and metadata management
    - Quality assessment with multi-dimensional scoring
    - Reputation system for researchers and sources
    - Collaborative research validation
    - Cross-network knowledge sharing
    - Research trend analysis and prediction
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 p2p_network: Optional[P2PNetworkIntegration] = None,
                 update_interval: float = 5.0):
        
        if db_path is None:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'p2p_research_tracking.db')
        
        self.db_path = db_path
        self.p2p_network = p2p_network
        self.update_interval = update_interval
        
        self.logger = logging.getLogger("P2PResearchTracking")
        
        # Core data structures
        self.research_sources: Dict[str, ResearchSource] = {}
        self.researcher_profiles: Dict[str, ResearcherProfile] = {}
        self.validations: Dict[str, ResearchValidation] = {}
        
        # Network integration
        self.network_callbacks = []
        self.research_trends = defaultdict(list)
        self.domain_expertise = defaultdict(set)
        
        # Background tasks
        self.running = False
        self.update_task = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        self.logger.info("P2P Research Tracking System initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT,
                    source_type TEXT,
                    domain TEXT,
                    url TEXT,
                    doi TEXT,
                    publication_date TEXT,
                    publisher TEXT,
                    journal TEXT,
                    conference TEXT,
                    abstract TEXT,
                    keywords TEXT,
                    citations INTEGER DEFAULT 0,
                    downloads INTEGER DEFAULT 0,
                    views INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    quality_level TEXT DEFAULT 'unverified',
                    peer_reviewed BOOLEAN DEFAULT FALSE,
                    impact_factor REAL,
                    h_index INTEGER,
                    network_reputation REAL DEFAULT 0.0,
                    validation_count INTEGER DEFAULT 0,
                    dispute_count INTEGER DEFAULT 0,
                    last_updated TEXT,
                    created_by TEXT,
                    validated_by TEXT,
                    disputed_by TEXT,
                    content_hash TEXT,
                    word_count INTEGER DEFAULT 0,
                    reference_count INTEGER DEFAULT 0,
                    methodology_score REAL DEFAULT 0.0,
                    reproducibility_score REAL DEFAULT 0.0,
                    novelty_score REAL DEFAULT 0.0,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS researcher_profiles (
                    researcher_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    total_sources INTEGER DEFAULT 0,
                    validated_sources INTEGER DEFAULT 0,
                    disputed_sources INTEGER DEFAULT 0,
                    research_reputation REAL DEFAULT 0.5,
                    expertise_domains TEXT,
                    average_quality_score REAL DEFAULT 0.0,
                    validation_accuracy REAL DEFAULT 0.5,
                    dispute_accuracy REAL DEFAULT 0.5,
                    network_contributions INTEGER DEFAULT 0,
                    collaboration_count INTEGER DEFAULT 0,
                    citations_received INTEGER DEFAULT 0,
                    last_activity TEXT,
                    activity_streak INTEGER DEFAULT 0,
                    total_activity_time TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_validations (
                    validation_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    researcher_id TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    reasoning TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_domain ON research_sources(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_quality ON research_sources(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_type ON research_sources(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_validations_source ON research_validations(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_validations_researcher ON research_validations(researcher_id)")
    
    def _load_data(self):
        """Load existing data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load research sources
                cursor = conn.execute("SELECT * FROM research_sources")
                for row in cursor.fetchall():
                    source = self._row_to_research_source(row)
                    self.research_sources[source.source_id] = source
                
                # Load researcher profiles
                cursor = conn.execute("SELECT * FROM researcher_profiles")
                for row in cursor.fetchall():
                    profile = self._row_to_researcher_profile(row)
                    self.researcher_profiles[profile.researcher_id] = profile
                
                # Load validations
                cursor = conn.execute("SELECT * FROM research_validations")
                for row in cursor.fetchall():
                    validation = self._row_to_research_validation(row)
                    self.validations[validation.validation_id] = validation
                
                self.logger.info(f"Loaded {len(self.research_sources)} sources, "
                               f"{len(self.researcher_profiles)} researchers, "
                               f"{len(self.validations)} validations")
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
    
    def _row_to_research_source(self, row) -> ResearchSource:
        """Convert database row to ResearchSource object"""
        return ResearchSource(
            source_id=row[0],
            title=row[1],
            authors=json.loads(row[2]) if row[2] else [],
            source_type=SourceType(row[3]) if row[3] else SourceType.WEBSITE,
            domain=ResearchDomain(row[4]) if row[4] else ResearchDomain.OTHER,
            url=row[5],
            doi=row[6],
            publication_date=datetime.fromisoformat(row[7]) if row[7] else None,
            publisher=row[8],
            journal=row[9],
            conference=row[10],
            abstract=row[11],
            keywords=json.loads(row[12]) if row[12] else [],
            citations=row[13] or 0,
            downloads=row[14] or 0,
            views=row[15] or 0,
            quality_score=row[16] or 0.0,
            quality_level=ResearchQuality(row[17]) if row[17] else ResearchQuality.UNVERIFIED,
            peer_reviewed=bool(row[18]),
            impact_factor=row[19],
            h_index=row[20],
            network_reputation=row[21] or 0.0,
            validation_count=row[22] or 0,
            dispute_count=row[23] or 0,
            last_updated=datetime.fromisoformat(row[24]) if row[24] else datetime.now(),
            created_by=row[25],
            validated_by=json.loads(row[26]) if row[26] else [],
            disputed_by=json.loads(row[27]) if row[27] else [],
            content_hash=row[28],
            word_count=row[29] or 0,
            reference_count=row[30] or 0,
            methodology_score=row[31] or 0.0,
            reproducibility_score=row[32] or 0.0,
            novelty_score=row[33] or 0.0,
            metadata=json.loads(row[34]) if row[34] else {}
        )
    
    def _row_to_researcher_profile(self, row) -> ResearcherProfile:
        """Convert database row to ResearcherProfile object"""
        return ResearcherProfile(
            researcher_id=row[0],
            username=row[1],
            user_id=row[2],
            total_sources=row[3] or 0,
            validated_sources=row[4] or 0,
            disputed_sources=row[5] or 0,
            research_reputation=row[6] or 0.5,
            expertise_domains=[ResearchDomain(d) for d in json.loads(row[7])] if row[7] else [],
            average_quality_score=row[8] or 0.0,
            validation_accuracy=row[9] or 0.5,
            dispute_accuracy=row[10] or 0.5,
            network_contributions=row[11] or 0,
            collaboration_count=row[12] or 0,
            citations_received=row[13] or 0,
            last_activity=datetime.fromisoformat(row[14]) if row[14] else datetime.now(),
            activity_streak=row[15] or 0,
            total_activity_time=timedelta(seconds=row[16]) if row[16] else timedelta(0),
            metadata=json.loads(row[17]) if row[17] else {}
        )
    
    def _row_to_research_validation(self, row) -> ResearchValidation:
        """Convert database row to ResearchValidation object"""
        return ResearchValidation(
            validation_id=row[0],
            source_id=row[1],
            researcher_id=row[2],
            validation_type=row[3],
            confidence=row[4] or 0.5,
            reasoning=row[5],
            timestamp=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
            metadata=json.loads(row[7]) if row[7] else {}
        )
    
    async def start(self):
        """Start the research tracking system"""
        if self.running:
            return
        
        self.running = True
        self.update_task = asyncio.create_task(self._background_update_loop())
        self.logger.info("P2P Research Tracking System started")
    
    async def stop(self):
        """Stop the research tracking system"""
        if not self.running:
            return
        
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        self.logger.info("P2P Research Tracking System stopped")
    
    async def _background_update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                await self._update_research_metrics()
                await self._update_network_integration()
                await self._analyze_research_trends()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in background update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def add_research_source(self, 
                           title: str,
                           authors: List[str],
                           source_type: SourceType,
                           domain: ResearchDomain,
                           url: Optional[str] = None,
                           doi: Optional[str] = None,
                           created_by: Optional[str] = None,
                           **kwargs) -> str:
        """Add a new research source"""
        source_id = str(uuid.uuid4())
        
        # Generate content hash if URL is provided
        content_hash = None
        if url:
            content_hash = self._generate_content_hash(url)
        
        source = ResearchSource(
            source_id=source_id,
            title=title,
            authors=authors,
            source_type=source_type,
            domain=domain,
            url=url,
            doi=doi,
            created_by=created_by,
            content_hash=content_hash,
            **kwargs
        )
        
        # Calculate initial quality score
        source.calculate_quality_score()
        
        # Store in memory and database
        self.research_sources[source_id] = source
        self._save_research_source(source)
        
        # Update domain expertise tracking
        if created_by and created_by in self.researcher_profiles:
            profile = self.researcher_profiles[created_by]
            if domain not in profile.expertise_domains:
                profile.expertise_domains.append(domain)
            profile.total_sources += 1
            profile.last_activity = datetime.now()
            self._save_researcher_profile(profile)
        
        self.logger.info(f"Added research source: {title} (ID: {source_id})")
        return source_id
    
    def validate_research_source(self,
                                source_id: str,
                                researcher_id: str,
                                validation_type: str,
                                confidence: float,
                                reasoning: str) -> bool:
        """Validate or dispute a research source"""
        if source_id not in self.research_sources:
            self.logger.error(f"Research source not found: {source_id}")
            return False
        
        if researcher_id not in self.researcher_profiles:
            self.logger.error(f"Researcher profile not found: {researcher_id}")
            return False
        
        source = self.research_sources[source_id]
        researcher = self.researcher_profiles[researcher_id]
        
        # Create validation record
        validation_id = str(uuid.uuid4())
        validation = ResearchValidation(
            validation_id=validation_id,
            source_id=source_id,
            researcher_id=researcher_id,
            validation_type=validation_type,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Update source
        if validation_type == 'validate':
            source.validated_by.append(researcher_id)
            source.validation_count += 1
        else:  # dispute
            source.disputed_by.append(researcher_id)
            source.dispute_count += 1
        
        # Recalculate quality score
        source.calculate_quality_score()
        
        # Update researcher profile
        researcher.network_contributions += 1
        researcher.last_activity = datetime.now()
        
        # Store validation and update database
        self.validations[validation_id] = validation
        self._save_research_validation(validation)
        self._save_research_source(source)
        self._save_researcher_profile(researcher)
        
        self.logger.info(f"Research source {validation_type}d: {source_id} by {researcher_id}")
        return True
    
    def get_research_sources(self,
                           domain: Optional[ResearchDomain] = None,
                           quality_level: Optional[ResearchQuality] = None,
                           source_type: Optional[SourceType] = None,
                           limit: int = 100) -> List[ResearchSource]:
        """Get research sources with filtering"""
        sources = list(self.research_sources.values())
        
        if domain:
            sources = [s for s in sources if s.domain == domain]
        
        if quality_level:
            sources = [s for s in sources if s.quality_level == quality_level]
        
        if source_type:
            sources = [s for s in sources if s.source_type == source_type]
        
        # Sort by quality score and return top results
        sources.sort(key=lambda s: s.quality_score, reverse=True)
        return sources[:limit]
    
    def get_researcher_profile(self, researcher_id: str) -> Optional[ResearcherProfile]:
        """Get researcher profile"""
        return self.researcher_profiles.get(researcher_id)
    
    def get_top_researchers(self, domain: Optional[ResearchDomain] = None, limit: int = 10) -> List[ResearcherProfile]:
        """Get top researchers by reputation"""
        researchers = list(self.researcher_profiles.values())
        
        if domain:
            researchers = [r for r in researchers if domain in r.expertise_domains]
        
        researchers.sort(key=lambda r: r.research_reputation, reverse=True)
        return researchers[:limit]
    
    def get_research_trends(self, domain: Optional[ResearchDomain] = None, days: int = 30) -> Dict[str, Any]:
        """Get research trends and analysis"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_sources = [
            s for s in self.research_sources.values()
            if s.last_updated >= cutoff_date
        ]
        
        if domain:
            recent_sources = [s for s in recent_sources if s.domain == domain]
        
        trends = {
            'total_sources': len(recent_sources),
            'quality_distribution': defaultdict(int),
            'source_type_distribution': defaultdict(int),
            'top_authors': defaultdict(int),
            'average_quality': 0.0,
            'validation_rate': 0.0
        }
        
        if recent_sources:
            for source in recent_sources:
                trends['quality_distribution'][source.quality_level.value] += 1
                trends['source_type_distribution'][source.source_type.value] += 1
                
                for author in source.authors:
                    trends['top_authors'][author] += 1
            
            trends['average_quality'] = np.mean([s.quality_score for s in recent_sources])
            
            total_validations = sum(s.validation_count for s in recent_sources)
            total_checks = sum(s.validation_count + s.dispute_count for s in recent_sources)
            if total_checks > 0:
                trends['validation_rate'] = total_validations / total_checks
        
        return trends
    
    def _generate_content_hash(self, url: str) -> str:
        """Generate content hash for URL"""
        try:
            response = requests.get(url, timeout=10)
            content = response.text
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not generate content hash for {url}: {e}")
            return ""
    
    def _save_research_source(self, source: ResearchSource):
        """Save research source to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO research_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source.source_id,
                    source.title,
                    json.dumps(source.authors),
                    source.source_type.value,
                    source.domain.value,
                    source.url,
                    source.doi,
                    source.publication_date.isoformat() if source.publication_date else None,
                    source.publisher,
                    source.journal,
                    source.conference,
                    source.abstract,
                    json.dumps(source.keywords),
                    source.citations,
                    source.downloads,
                    source.views,
                    source.quality_score,
                    source.quality_level.value,
                    source.peer_reviewed,
                    source.impact_factor,
                    source.h_index,
                    source.network_reputation,
                    source.validation_count,
                    source.dispute_count,
                    source.last_updated.isoformat(),
                    source.created_by,
                    json.dumps(source.validated_by),
                    json.dumps(source.disputed_by),
                    source.content_hash,
                    source.word_count,
                    source.reference_count,
                    source.methodology_score,
                    source.reproducibility_score,
                    source.novelty_score,
                    json.dumps(source.metadata)
                ))
        except Exception as e:
            self.logger.error(f"Error saving research source: {e}")
    
    def _save_researcher_profile(self, profile: ResearcherProfile):
        """Save researcher profile to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO researcher_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.researcher_id,
                    profile.username,
                    profile.user_id,
                    profile.total_sources,
                    profile.validated_sources,
                    profile.disputed_sources,
                    profile.research_reputation,
                    json.dumps([d.value for d in profile.expertise_domains]),
                    profile.average_quality_score,
                    profile.validation_accuracy,
                    profile.dispute_accuracy,
                    profile.network_contributions,
                    profile.collaboration_count,
                    profile.citations_received,
                    profile.last_activity.isoformat(),
                    profile.activity_streak,
                    profile.total_activity_time.total_seconds(),
                    json.dumps(profile.metadata)
                ))
        except Exception as e:
            self.logger.error(f"Error saving researcher profile: {e}")
    
    def _save_research_validation(self, validation: ResearchValidation):
        """Save research validation to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO research_validations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    validation.validation_id,
                    validation.source_id,
                    validation.researcher_id,
                    validation.validation_type,
                    validation.confidence,
                    validation.reasoning,
                    validation.timestamp.isoformat(),
                    json.dumps(validation.metadata)
                ))
        except Exception as e:
            self.logger.error(f"Error saving research validation: {e}")
    
    async def _update_research_metrics(self):
        """Update research metrics and reputation scores"""
        for source in self.research_sources.values():
            source.calculate_quality_score()
            self._save_research_source(source)
        
        for researcher in self.researcher_profiles.values():
            # Update activity streak
            days_since_activity = (datetime.now() - researcher.last_activity).days
            if days_since_activity <= 1:
                researcher.activity_streak += 1
            else:
                researcher.activity_streak = 0
            
            self._save_researcher_profile(researcher)
    
    async def _update_network_integration(self):
        """Update P2P network integration"""
        if not self.p2p_network:
            return
        
        # Update network metrics with research data
        network_metrics = self.p2p_network.get_network_metrics()
        
        # Add research-specific metrics
        research_metrics = {
            'total_research_sources': len(self.research_sources),
            'high_quality_sources': len([s for s in self.research_sources.values() 
                                       if s.quality_level in [ResearchQuality.HIGH, ResearchQuality.EXCELLENT, ResearchQuality.PEER_REVIEWED]]),
            'active_researchers': len([r for r in self.researcher_profiles.values() 
                                     if r.activity_streak > 0]),
            'average_research_reputation': np.mean([r.research_reputation for r in self.researcher_profiles.values()]) if self.researcher_profiles else 0.0
        }
        
        # Trigger network callbacks
        for callback in self.network_callbacks:
            try:
                callback(research_metrics)
            except Exception as e:
                self.logger.error(f"Error in network callback: {e}")
    
    async def _analyze_research_trends(self):
        """Analyze research trends and patterns"""
        # Analyze domain popularity
        domain_counts = defaultdict(int)
        for source in self.research_sources.values():
            domain_counts[source.domain] += 1
        
        # Update trends
        self.research_trends['domain_popularity'] = dict(domain_counts)
        
        # Analyze quality trends
        quality_trends = defaultdict(int)
        for source in self.research_sources.values():
            quality_trends[source.quality_level] += 1
        
        self.research_trends['quality_distribution'] = dict(quality_trends)
    
    def add_network_callback(self, callback: callable):
        """Add callback for network updates"""
        self.network_callbacks.append(callback)
    
    def get_research_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive research network statistics"""
        return {
            'total_sources': len(self.research_sources),
            'total_researchers': len(self.researcher_profiles),
            'total_validations': len(self.validations),
            'quality_distribution': dict(self.research_trends.get('quality_distribution', {})),
            'domain_popularity': dict(self.research_trends.get('domain_popularity', {})),
            'average_quality_score': np.mean([s.quality_score for s in self.research_sources.values()]) if self.research_sources else 0.0,
            'average_research_reputation': np.mean([r.research_reputation for r in self.researcher_profiles.values()]) if self.researcher_profiles else 0.0,
            'active_researchers': len([r for r in self.researcher_profiles.values() if r.activity_streak > 0])
        } 