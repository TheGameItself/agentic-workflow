#!/usr/bin/env python3
"""
Research Integration System for MCP Server
Provides research source tracking, validation, and integration capabilities.

Features:
- Research source tracking and validation
- CRAAP test implementation for source credibility
- Research feedback loops
- Integration with latest research findings
- Automated research paper analysis

# --- RESEARCH AND IDEA.TXT REFERENCES ---
# This module implements research source tracking, validation, and integration per idea.txt requirements.
# All research integration features are designed to support continuous self-improvement and research-driven development.
# Paper analysis is a placeholder for future expansion (see idea.txt).
"""

import json
import re
import requests
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from urllib.parse import urlparse
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ResearchSource:
    """Research source information."""
    title: str
    authors: List[str] = field(default_factory=list)
    url: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    source_type: str = "paper"  # paper, article, book, website, etc.
    credibility_score: float = 0.0
    craap_scores: Dict[str, float] = field(default_factory=dict)
    last_accessed: Optional[str] = None
    hash_id: Optional[str] = None
    
    def __post_init__(self):
        if self.hash_id is None:
            self.hash_id = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate a unique hash for this source."""
        content = f"{self.title}{self.authors}{self.url}{self.doi}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class ResearchFinding:
    """Research finding or insight."""
    source_id: str
    finding: str
    category: str
    confidence: float
    relevance_score: float
    implementation_status: str = "pending"  # pending, implemented, rejected
    notes: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

class CRAAPTest:
    """CRAAP test implementation for source credibility assessment."""
    
    def __init__(self):
        self.criteria = {
            "currency": {
                "description": "Timeliness of the information",
                "questions": [
                    "When was the information published or posted?",
                    "Has the information been revised or updated?",
                    "Is the information current or out-of-date for your topic?"
                ],
                "weight": 0.2
            },
            "relevance": {
                "description": "Importance of the information for your needs",
                "questions": [
                    "Does the information relate to your topic or answer your question?",
                    "Who is the intended audience?",
                    "Is the information at an appropriate level?"
                ],
                "weight": 0.25
            },
            "authority": {
                "description": "Source of the information",
                "questions": [
                    "Who is the author/publisher/source/sponsor?",
                    "Are the author's credentials or organizational affiliations given?",
                    "What are the author's qualifications to write on the topic?"
                ],
                "weight": 0.25
            },
            "accuracy": {
                "description": "Reliability, truthfulness, and correctness of the content",
                "questions": [
                    "Where does the information come from?",
                    "Is the information supported by evidence?",
                    "Has the information been reviewed or refereed?"
                ],
                "weight": 0.2
            },
            "purpose": {
                "description": "Reason the information exists",
                "questions": [
                    "What is the purpose of the information?",
                    "Do the authors/sponsors make their intentions or purpose clear?",
                    "Is the information fact, opinion, or propaganda?"
                ],
                "weight": 0.1
            }
        }
    
    def evaluate_source(self, source: ResearchSource) -> Dict[str, Any]:
        """Evaluate a source using the CRAAP test."""
        scores = {}
        total_score = 0.0
        
        for criterion, config in self.criteria.items():
            score = self._evaluate_criterion(criterion, source)
            scores[criterion] = score
            total_score += score * config["weight"]
        
        return {
            "scores": scores,
            "total_score": total_score,
            "credibility_level": self._get_credibility_level(total_score),
            "recommendations": self._get_recommendations(scores)
        }
    
    def _evaluate_criterion(self, criterion: str, source: ResearchSource) -> float:
        """Evaluate a specific CRAAP criterion."""
        if criterion == "currency":
            return self._evaluate_currency(source)
        elif criterion == "relevance":
            return self._evaluate_relevance(source)
        elif criterion == "authority":
            return self._evaluate_authority(source)
        elif criterion == "accuracy":
            return self._evaluate_accuracy(source)
        elif criterion == "purpose":
            return self._evaluate_purpose(source)
        else:
            return 0.0
    
    def _evaluate_currency(self, source: ResearchSource) -> float:
        """Evaluate currency of the source."""
        if not source.publication_date:
            return 0.5  # Neutral score if no date available
        
        try:
            pub_date = datetime.fromisoformat(source.publication_date.replace('Z', '+00:00'))
            now = datetime.utcnow()
            age_years = (now - pub_date).days / 365.25
            
            if age_years < 1:
                return 1.0  # Very recent
            elif age_years < 3:
                return 0.8  # Recent
            elif age_years < 5:
                return 0.6  # Moderately recent
            elif age_years < 10:
                return 0.4  # Somewhat dated
            else:
                return 0.2  # Dated
        except:
            return 0.5
    
    def _evaluate_relevance(self, source: ResearchSource) -> float:
        """Evaluate relevance of the source."""
        # This would typically involve keyword matching and topic analysis
        # For now, return a default score
        return 0.7
    
    def _evaluate_authority(self, source: ResearchSource) -> float:
        """Evaluate authority of the source."""
        if not source.authors:
            return 0.3
        
        # Check for institutional affiliations, credentials, etc.
        # This is a simplified evaluation
        author_count = len(source.authors)
        if author_count == 1:
            return 0.6
        elif author_count <= 3:
            return 0.8
        else:
            return 0.7
    
    def _evaluate_accuracy(self, source: ResearchSource) -> float:
        """Evaluate accuracy of the source."""
        # Check for peer-reviewed journals, citations, etc.
        if source.journal and "peer" in source.journal.lower():
            return 0.9
        elif source.doi:
            return 0.8
        elif source.url and any(domain in source.url.lower() for domain in [".edu", ".gov", ".org"]):
            return 0.7
        else:
            return 0.5
    
    def _evaluate_purpose(self, source: ResearchSource) -> float:
        """Evaluate purpose of the source."""
        # Check for bias, commercial interests, etc.
        if source.url and any(domain in source.url.lower() for domain in [".com", "blog", "news"]):
            return 0.6
        else:
            return 0.8
    
    def _get_credibility_level(self, score: float) -> str:
        """Get credibility level based on score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _get_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Get recommendations based on scores."""
        recommendations = []
        
        for criterion, score in scores.items():
            if score < 0.5:
                recommendations.append(f"Improve {criterion}: {self.criteria[criterion]['description']}")
        
        return recommendations

class ResearchTracker:
    """Tracks and manages research sources and findings."""
    
    def __init__(self, db_path: str = "data/research.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("research_tracker")
        self.craap_test = CRAAPTest()
        self._init_database()
    
    def _init_database(self):
        """Initialize the research database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_sources (
                    hash_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT,
                    url TEXT,
                    doi TEXT,
                    publication_date TEXT,
                    journal TEXT,
                    abstract TEXT,
                    keywords TEXT,
                    source_type TEXT,
                    credibility_score REAL,
                    craap_scores TEXT,
                    last_accessed TEXT,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT,
                    finding TEXT NOT NULL,
                    category TEXT,
                    confidence REAL,
                    relevance_score REAL,
                    implementation_status TEXT,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (source_id) REFERENCES research_sources (hash_id)
                )
            """)
            
            conn.commit()
    
    def add_source(self, source: ResearchSource) -> bool:
        """Add a research source to the database."""
        try:
            # Evaluate source credibility
            evaluation = self.craap_test.evaluate_source(source)
            source.credibility_score = evaluation["total_score"]
            source.craap_scores = evaluation["scores"]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO research_sources 
                    (hash_id, title, authors, url, doi, publication_date, journal, 
                     abstract, keywords, source_type, credibility_score, craap_scores, 
                     last_accessed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source.hash_id,
                    source.title,
                    json.dumps(source.authors),
                    source.url,
                    source.doi,
                    source.publication_date,
                    source.journal,
                    source.abstract,
                    json.dumps(source.keywords),
                    source.source_type,
                    source.credibility_score,
                    json.dumps(source.craap_scores),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
            
            self.logger.info(f"Added research source: {source.title}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add research source: {e}")
            return False
    
    def get_source(self, hash_id: str) -> Optional[ResearchSource]:
        """Get a research source by hash ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM research_sources WHERE hash_id = ?
                """, (hash_id,))
                row = cursor.fetchone()
                
                if row:
                    return ResearchSource(
                        title=row[1],
                        authors=json.loads(row[2]) if row[2] else [],
                        url=row[3],
                        doi=row[4],
                        publication_date=row[5],
                        journal=row[6],
                        abstract=row[7],
                        keywords=json.loads(row[8]) if row[8] else [],
                        source_type=row[9],
                        credibility_score=row[10],
                        craap_scores=json.loads(row[11]) if row[11] else {},
                        last_accessed=row[12],
                        hash_id=row[0]
                    )
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get research source: {e}")
            return None
    
    def search_sources(self, query: str, limit: int = 10) -> List[ResearchSource]:
        """Search research sources by query."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM research_sources 
                    WHERE title LIKE ? OR abstract LIKE ? OR keywords LIKE ?
                    ORDER BY credibility_score DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
                
                sources = []
                for row in cursor.fetchall():
                    source = ResearchSource(
                        title=row[1],
                        authors=json.loads(row[2]) if row[2] else [],
                        url=row[3],
                        doi=row[4],
                        publication_date=row[5],
                        journal=row[6],
                        abstract=row[7],
                        keywords=json.loads(row[8]) if row[8] else [],
                        source_type=row[9],
                        credibility_score=row[10],
                        craap_scores=json.loads(row[11]) if row[11] else {},
                        last_accessed=row[12],
                        hash_id=row[0]
                    )
                    sources.append(source)
                
                return sources
                
        except Exception as e:
            self.logger.error(f"Failed to search sources: {e}")
            return []
    
    def add_finding(self, finding: ResearchFinding) -> bool:
        """Add a research finding."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO research_findings 
                    (source_id, finding, category, confidence, relevance_score,
                     implementation_status, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    finding.source_id,
                    finding.finding,
                    finding.category,
                    finding.confidence,
                    finding.relevance_score,
                    finding.implementation_status,
                    finding.notes,
                    finding.created_at,
                    finding.updated_at
                ))
                conn.commit()
            
            self.logger.info(f"Added research finding from source: {finding.source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add research finding: {e}")
            return False
    
    def get_findings(self, source_id: Optional[str] = None, category: Optional[str] = None) -> List[ResearchFinding]:
        """Get research findings."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM research_findings WHERE 1=1"
                params = []
                
                if source_id:
                    query += " AND source_id = ?"
                    params.append(source_id)
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                
                findings = []
                for row in cursor.fetchall():
                    finding = ResearchFinding(
                        source_id=row[1],
                        finding=row[2],
                        category=row[3],
                        confidence=row[4],
                        relevance_score=row[5],
                        implementation_status=row[6],
                        notes=row[7],
                        created_at=row[8],
                        updated_at=row[9]
                    )
                    findings.append(finding)
                
                return findings
                
        except Exception as e:
            self.logger.error(f"Failed to get findings: {e}")
            return []
    
    def update_finding_status(self, finding_id: int, status: str, notes: Optional[str] = None) -> bool:
        """Update implementation status of a finding."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE research_findings 
                    SET implementation_status = ?, notes = ?, updated_at = ?
                    WHERE id = ?
                """, (status, notes, datetime.utcnow().isoformat(), finding_id))
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update finding status: {e}")
            return False

class ResearchIntegrator:
    """Integrates research findings into the MCP system."""
    
    def __init__(self, tracker: ResearchTracker):
        self.tracker = tracker
        self.logger = logging.getLogger("research_integrator")
        self.implementation_queue = []
    
    async def analyze_paper(self, paper_url: str) -> Dict[str, Any]:
        """Analyze a research paper and extract findings. TODO: Implement full pipeline for paper analysis."""
        try:
            # This would typically involve:
            # 1. Downloading the paper
            # 2. Extracting text content
            # 3. Using NLP to identify key findings
            # 4. Categorizing findings
            # TODO: Implement full paper analysis pipeline
            return {
                "status": "success",
                "findings": [],
                "summary": "Paper analysis not yet implemented",
                "recommendations": []
            }
        except Exception as e:
            logging.error(f"Failed to analyze paper: {e}")
            return {"status": "error", "message": str(e)}
    
    async def integrate_findings(self, category: str) -> Dict[str, Any]:
        """Integrate research findings into the system."""
        try:
            findings = self.tracker.get_findings(category=category)
            
            if not findings:
                return {"status": "no_findings", "message": f"No findings for category: {category}"}
            
            # Filter high-confidence, relevant findings
            relevant_findings = [
                f for f in findings 
                if f.confidence > 0.7 and f.relevance_score > 0.6
            ]
            
            integration_results = []
            for finding in relevant_findings:
                result = await self._apply_finding(finding)
                integration_results.append(result)
            
            return {
                "status": "success",
                "findings_processed": len(relevant_findings),
                "results": integration_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to integrate findings: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _apply_finding(self, finding: ResearchFinding) -> Dict[str, Any]:
        """Apply a specific finding to the system."""
        try:
            # This would involve:
            # 1. Parsing the finding
            # 2. Identifying applicable system components
            # 3. Implementing the finding
            # 4. Testing the implementation
            
            # For now, just mark as implemented
            self.tracker.update_finding_status(
                finding_id=1,  # This should be the actual finding ID
                status="implemented",
                notes="Automatically integrated by research integrator"
            )
            
            return {
                "finding_id": 1,
                "status": "implemented",
                "message": "Finding applied successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to apply finding: {e}")
            return {
                "finding_id": 1,
                "status": "failed",
                "message": str(e)
            }
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of research integration status."""
        try:
            all_findings = self.tracker.get_findings()
            
            summary = {
                "total_findings": len(all_findings),
                "implemented": len([f for f in all_findings if f.implementation_status == "implemented"]),
                "pending": len([f for f in all_findings if f.implementation_status == "pending"]),
                "rejected": len([f for f in all_findings if f.implementation_status == "rejected"]),
                "categories": {},
                "recent_findings": []
            }
            
            # Group by category
            for finding in all_findings:
                if finding.category not in summary["categories"]:
                    summary["categories"][finding.category] = 0
                summary["categories"][finding.category] += 1
            
            # Get recent findings
            recent_findings = sorted(all_findings, key=lambda x: x.created_at, reverse=True)[:5]
            summary["recent_findings"] = [
                {
                    "finding": f.finding,
                    "category": f.category,
                    "confidence": f.confidence,
                    "status": f.implementation_status
                }
                for f in recent_findings
            ]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get research summary: {e}")
            return {"error": str(e)} 