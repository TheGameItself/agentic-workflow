#!/usr/bin/env python3
"""
Memory Quality Assessment System

This module implements comprehensive memory quality assessment, confidence and relevance scoring,
memory relationship detection and management, and memory consolidation and optimization processes.
"""

import sqlite3
import json
import re
import math
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter

# Optional numpy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

class MemoryQualityAssessment:
    """
    Comprehensive memory quality assessment system with advanced scoring algorithms,
    relationship detection, and memory optimization.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the memory quality assessment system."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'unified_memory.db')
        
        self.db_path = str(db_path)
        self._init_database()
        
        # Initialize scoring weights
        self.quality_weights = {
            'specificity': 0.25,
            'context_richness': 0.25,
            'structure': 0.20,
            'recency': 0.15,
            'access_frequency': 0.15
        }
        
        self.confidence_weights = {
            'source_reliability': 0.30,
            'consistency': 0.25,
            'verification': 0.25,
            'precision': 0.20
        }
        
        self.relevance_weights = {
            'semantic_similarity': 0.35,
            'contextual_match': 0.30,
            'temporal_relevance': 0.20,
            'user_feedback': 0.15
        }
        
        # Initialize relationship detection thresholds
        self.relationship_thresholds = {
            'semantic_similarity': 0.65,
            'contextual_overlap': 0.50,
            'temporal_proximity': 0.40,
            'reference_match': 0.75
        }
    
    def _init_database(self):
        """Initialize database tables for memory quality assessment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memory quality metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                specificity_score REAL DEFAULT 0.5,
                context_richness_score REAL DEFAULT 0.5,
                structure_score REAL DEFAULT 0.5,
                recency_score REAL DEFAULT 0.5,
                access_frequency_score REAL DEFAULT 0.5,
                source_reliability_score REAL DEFAULT 0.5,
                consistency_score REAL DEFAULT 0.5,
                verification_score REAL DEFAULT 0.5,
                precision_score REAL DEFAULT 0.5,
                semantic_similarity_score REAL DEFAULT 0.5,
                contextual_match_score REAL DEFAULT 0.5,
                temporal_relevance_score REAL DEFAULT 0.5,
                user_feedback_score REAL DEFAULT 0.5,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES advanced_memories (id)
            )
        """)
        
        # Memory relationships table (enhanced version)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_memory_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER NOT NULL,
                target_memory_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                semantic_similarity REAL DEFAULT 0.0,
                contextual_overlap REAL DEFAULT 0.0,
                temporal_proximity REAL DEFAULT 0.0,
                reference_match REAL DEFAULT 0.0,
                overall_strength REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.5,
                bidirectional BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_memory_id) REFERENCES advanced_memories (id),
                FOREIGN KEY (target_memory_id) REFERENCES advanced_memories (id)
            )
        """)
        
        conn.commit()
        conn.close()    def g
et_memory_quality_report(self, memory_id: int) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report for a memory.
        
        Args:
            memory_id: ID of the memory to assess
            
        Returns:
            Dictionary containing quality assessment results
        """
        # Get memory data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, text, memory_type, priority, context, tags, category,
                   quality_score, confidence_score, completeness_score, relevance_score,
                   memory_order, created_at, updated_at, last_accessed, access_count
            FROM advanced_memories WHERE id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {'error': f'Memory with ID {memory_id} not found'}
        
        # Parse memory data
        memory_id, text, memory_type, priority, context, tags_json, category, \
        quality_score, confidence_score, completeness_score, relevance_score, \
        memory_order, created_at, updated_at, last_accessed, access_count = row
        
        tags = json.loads(tags_json) if tags_json else []
        
        # Calculate enhanced quality metrics
        specificity = self._calculate_specificity_score(text)
        context_richness = self._calculate_context_richness_score(context, tags, category)
        structure = self._calculate_structure_score(text)
        recency = self._calculate_recency_score(created_at, updated_at)
        access_frequency = self._calculate_access_frequency_score(access_count)
        
        # Calculate enhanced confidence metrics
        source_reliability = self._calculate_source_reliability_score(text, context)
        consistency = 0.5  # Default value, would require related memories
        verification = self._calculate_verification_score(text, context)
        precision = self._calculate_precision_score(text)
        
        # Calculate enhanced relevance metrics
        semantic_similarity = 0.5  # Default value, would require query context
        contextual_match = 0.5  # Default value, would require current context
        temporal_relevance = self._calculate_temporal_relevance_score(created_at, last_accessed, access_count)
        user_feedback = 0.5  # Default value, would require feedback data
        
        # Calculate overall quality
        enhanced_quality_score = (specificity + context_richness + structure + recency + access_frequency) / 5
        enhanced_confidence_score = (source_reliability + consistency + verification + precision) / 4
        enhanced_relevance_score = (semantic_similarity + contextual_match + temporal_relevance + user_feedback) / 4
        enhanced_completeness_score = (enhanced_quality_score + enhanced_confidence_score) / 2
        
        overall_quality = (enhanced_quality_score + enhanced_confidence_score + 
                          enhanced_relevance_score + enhanced_completeness_score) / 4
        
        # Generate improvement suggestions
        suggestions = []
        if enhanced_quality_score < 0.7:
            suggestions.append("Add more specific details to improve quality")
        if enhanced_confidence_score < 0.7:
            suggestions.append("Include source references or verification details to increase confidence")
        if enhanced_relevance_score < 0.7:
            suggestions.append("Add relevant tags or context to improve relevance")
        if enhanced_completeness_score < 0.7:
            suggestions.append("Provide more comprehensive information for completeness")
        
        return {
            'memory_id': memory_id,
            'overall_quality': overall_quality,
            'quality_breakdown': {
                'quality_score': enhanced_quality_score,
                'confidence_score': enhanced_confidence_score,
                'completeness_score': enhanced_completeness_score,
                'relevance_score': enhanced_relevance_score
            },
            'detailed_metrics': {
                'quality': {
                    'specificity': specificity,
                    'context_richness': context_richness,
                    'structure': structure,
                    'recency': recency,
                    'access_frequency': access_frequency
                },
                'confidence': {
                    'source_reliability': source_reliability,
                    'consistency': consistency,
                    'verification': verification,
                    'precision': precision
                },
                'relevance': {
                    'semantic_similarity': semantic_similarity,
                    'contextual_match': contextual_match,
                    'temporal_relevance': temporal_relevance,
                    'user_feedback': user_feedback
                }
            },
            'content_info': {
                'text_length': len(text),
                'has_context': bool(context),
                'tag_count': len(tags),
                'category': category,
                'access_count': access_count,
                'created_at': created_at,
                'last_accessed': last_accessed
            },
            'improvement_suggestions': suggestions
        }    d
ef _calculate_specificity_score(self, text: str) -> float:
        """Calculate specificity score based on text content."""
        if not text:
            return 0.3
        
        # Base score
        score = 0.5
        
        # Check for specific details
        if re.search(r'\d+', text):  # Contains numbers
            score += 0.1
        
        # Check for specific terms
        specific_terms = ['specifically', 'exactly', 'precisely', 'particularly', 'namely']
        if any(term in text.lower() for term in specific_terms):
            score += 0.1
        
        # Check for detailed descriptions
        if len(text.split()) > 50:  # Longer text tends to be more specific
            score += 0.1
        
        # Check for technical terms (simple heuristic)
        technical_pattern = r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b'  # CamelCase words
        if re.search(technical_pattern, text):
            score += 0.1
        
        # Check for specific examples
        example_patterns = [r'for example', r'e\.g\.', r'such as', r'instance of']
        if any(re.search(pattern, text.lower()) for pattern in example_patterns):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_context_richness_score(self, context: str, tags: List[str], category: str) -> float:
        """Calculate context richness score."""
        # Base score
        score = 0.5
        
        # Context presence and length
        if context:
            score += 0.1
            if len(context) > 100:
                score += 0.1
        
        # Tags presence and count
        if tags:
            score += min(0.05 * len(tags), 0.2)  # Up to 0.2 for tags
        
        # Category specificity
        if category and category != 'general':
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structure score based on text organization."""
        if not text:
            return 0.3
        
        # Base score
        score = 0.5
        
        # Check for structured formatting
        if re.search(r'^\s*\d+\.', text, re.MULTILINE):  # Numbered lists
            score += 0.15
        
        if re.search(r'^\s*[-*•]', text, re.MULTILINE):  # Bullet points
            score += 0.15
        
        # Check for headings/sections
        if re.search(r'^#+\s+\w+', text, re.MULTILINE):  # Markdown headings
            score += 0.15
        
        # Check for key-value pairs
        if re.search(r'^\s*[A-Za-z]+\s*:\s*\w+', text, re.MULTILINE):
            score += 0.15
        
        # Check for code blocks or structured data
        if re.search(r'```|{|}|\[|\]', text):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_recency_score(self, created_at: str, updated_at: str) -> float:
        """Calculate recency score based on creation and update times."""
        # Default score if dates are missing
        if not created_at:
            return 0.5
        
        try:
            # Parse dates
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
                
            if updated_at and isinstance(updated_at, str):
                updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            elif updated_at:
                updated_date = updated_at
            else:
                updated_date = created_date
            
            # Use the most recent date
            most_recent = max(created_date, updated_date)
            
            # Calculate age in days
            now = datetime.now()
            age_days = (now - most_recent).days
            
            # Score based on recency
            if age_days < 1:  # Today
                return 1.0
            elif age_days < 7:  # Within a week
                return 0.9
            elif age_days < 30:  # Within a month
                return 0.7
            elif age_days < 90:  # Within 3 months
                return 0.5
            elif age_days < 365:  # Within a year
                return 0.3
            else:  # Older than a year
                return 0.1
                
        except (ValueError, TypeError):
            # Fallback if date parsing fails
            return 0.5
    
    def _calculate_access_frequency_score(self, access_count: int) -> float:
        """Calculate access frequency score."""
        # Score based on access count
        if access_count > 20:
            return 1.0
        elif access_count > 10:
            return 0.9
        elif access_count > 5:
            return 0.7
        elif access_count > 2:
            return 0.5
        elif access_count > 0:
            return 0.3
        else:
            return 0.1    def _
calculate_source_reliability_score(self, text: str, context: str) -> float:
        """Calculate source reliability score."""
        # Base score
        score = 0.5
        
        # Check for citations or references
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\w+,\s*\d{4}\)',  # (Author, 2023)
            r'according to',
            r'cited in',
            r'reference[sd]',
            r'source[sd]'
        ]
        
        if any(re.search(pattern, text.lower()) for pattern in citation_patterns):
            score += 0.2
        
        # Check for authoritative sources in context
        authority_indicators = [
            'official', 'documentation', 'research', 'study', 'paper', 
            'publication', 'journal', 'expert', 'verified'
        ]
        
        if any(indicator in context.lower() for indicator in authority_indicators):
            score += 0.2
        
        # Check for uncertainty indicators (negative impact)
        uncertainty_indicators = [
            'maybe', 'perhaps', 'possibly', 'might', 'could be', 
            'uncertain', 'unclear', 'not sure', 'guess'
        ]
        
        if any(indicator in text.lower() for indicator in uncertainty_indicators):
            score -= 0.2
        
        # Check for first-hand knowledge indicators
        firsthand_indicators = [
            'I observed', 'I tested', 'I verified', 'I confirmed',
            'direct observation', 'personal experience'
        ]
        
        if any(indicator in text.lower() for indicator in firsthand_indicators):
            score += 0.2
        
        return max(0.1, min(score, 1.0))
    
    def _calculate_verification_score(self, text: str, context: str) -> float:
        """Calculate verification score based on evidence of fact-checking."""
        # Base score
        score = 0.5
        
        # Check for verification indicators
        verification_indicators = [
            'verified', 'confirmed', 'tested', 'validated', 'proven',
            'checked', 'reviewed', 'examined', 'inspected', 'authenticated'
        ]
        
        if any(indicator in text.lower() for indicator in verification_indicators):
            score += 0.2
        
        if any(indicator in context.lower() for indicator in verification_indicators):
            score += 0.1
        
        # Check for evidence indicators
        evidence_indicators = [
            'evidence', 'proof', 'data', 'results', 'findings',
            'experiment', 'test', 'measurement', 'observation'
        ]
        
        if any(indicator in text.lower() for indicator in evidence_indicators):
            score += 0.2
        
        # Check for specific metrics or measurements
        if re.search(r'\d+(?:\.\d+)?\s*(?:%|percent|kg|mb|sec|ms|hours?|days?)', text.lower()):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_precision_score(self, text: str) -> float:
        """Calculate precision score based on specificity and detail."""
        if not text:
            return 0.3
        
        # Base score
        score = 0.5
        
        # Check for precise numbers
        if re.search(r'\d+\.\d+', text):  # Decimal numbers
            score += 0.15
        
        # Check for specific dates
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text):  # Date patterns
            score += 0.15
        
        # Check for precise terminology
        precision_indicators = [
            'exactly', 'precisely', 'specifically', 'accurately',
            'explicitly', 'clearly', 'definitely', 'certainly'
        ]
        
        if any(indicator in text.lower() for indicator in precision_indicators):
            score += 0.15
        
        # Check for detailed descriptions
        if len(text.split()) > 100:  # Longer, more detailed text
            score += 0.1
        
        # Check for technical precision
        technical_precision = [
            r'version \d+\.\d+\.\d+',  # Semantic versioning
            r'[A-Fa-f0-9]{6,}',  # Hex codes, hashes
            r'(?:\d{1,3}\.){3}\d{1,3}',  # IP addresses
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Email pattern
        ]
        
        if any(re.search(pattern, text) for pattern in technical_precision):
            score += 0.15
        
        return min(score, 1.0)   
 def _calculate_temporal_relevance_score(self, created_at: str, last_accessed: str, access_count: int) -> float:
        """Calculate temporal relevance score based on recency and access patterns."""
        # Default score if dates are missing
        if not created_at:
            return 0.5
        
        try:
            # Parse dates
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
                
            if last_accessed and isinstance(last_accessed, str):
                accessed_date = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            elif last_accessed:
                accessed_date = last_accessed
            else:
                accessed_date = created_date
            
            # Calculate age in days
            now = datetime.now()
            creation_age = (now - created_date).days
            last_access_age = (now - accessed_date).days
            
            # Score based on creation recency (30%)
            if creation_age < 7:  # Within a week
                creation_score = 1.0
            elif creation_age < 30:  # Within a month
                creation_score = 0.8
            elif creation_age < 90:  # Within 3 months
                creation_score = 0.6
            elif creation_age < 365:  # Within a year
                creation_score = 0.4
            else:  # Older than a year
                creation_score = 0.2
            
            # Score based on last access recency (40%)
            if last_access_age < 1:  # Today
                access_score = 1.0
            elif last_access_age < 7:  # Within a week
                access_score = 0.8
            elif last_access_age < 30:  # Within a month
                access_score = 0.6
            elif last_access_age < 90:  # Within 3 months
                access_score = 0.4
            else:  # Older than 3 months
                access_score = 0.2
            
            # Score based on access frequency (30%)
            if access_count > 10:
                frequency_score = 1.0
            elif access_count > 5:
                frequency_score = 0.8
            elif access_count > 2:
                frequency_score = 0.6
            elif access_count > 0:
                frequency_score = 0.4
            else:
                frequency_score = 0.2
            
            # Combined temporal relevance score
            return (creation_score * 0.3) + (access_score * 0.4) + (frequency_score * 0.3)
                
        except (ValueError, TypeError):
            # Fallback if date parsing fails
            return 0.5
    
    def detect_memory_relationships(self, memory_id: int, max_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Detect relationships between a memory and other memories in the system.
        
        Args:
            memory_id: ID of the memory to analyze
            max_candidates: Maximum number of candidate memories to check
            
        Returns:
            List of detected relationships with scores
        """
        # Get memory data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT text, context, tags, category FROM advanced_memories WHERE id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []
        
        source_text, source_context, source_tags_json, source_category = row
        source_tags = json.loads(source_tags_json) if source_tags_json else []
        
        # Get candidate memories
        cursor.execute("""
            SELECT id, text, context, tags, category, created_at
            FROM advanced_memories 
            WHERE id != ? 
            ORDER BY updated_at DESC
            LIMIT ?
        """, (memory_id, max_candidates))
        
        candidates = cursor.fetchall()
        conn.close()
        
        relationships = []
        
        for candidate in candidates:
            candidate_id, candidate_text, candidate_context, candidate_tags_json, candidate_category, candidate_created_at = candidate
            candidate_tags = json.loads(candidate_tags_json) if candidate_tags_json else []
            
            # Calculate relationship metrics
            semantic_similarity = self._calculate_text_similarity(source_text, candidate_text)
            contextual_overlap = self._calculate_contextual_overlap(source_context, candidate_context, source_tags, candidate_tags, source_category, candidate_category)
            temporal_proximity = self._calculate_temporal_proximity(source_created_at=None, target_created_at=candidate_created_at)
            reference_match = self._check_references(source_text, candidate_text, memory_id, candidate_id)
            
            # Calculate overall relationship strength
            overall_strength = (
                semantic_similarity * 0.4 +
                contextual_overlap * 0.3 +
                temporal_proximity * 0.1 +
                reference_match * 0.2
            )
            
            # Determine relationship type
            relationship_type = self._determine_relationship_type(
                source_text, candidate_text, 
                semantic_similarity, contextual_overlap, 
                temporal_proximity, reference_match
            )
            
            # Check if relationship is strong enough
            if overall_strength >= self.relationship_thresholds['semantic_similarity']:
                relationships.append({
                    'source_memory_id': memory_id,
                    'target_memory_id': candidate_id,
                    'relationship_type': relationship_type,
                    'semantic_similarity': semantic_similarity,
                    'contextual_overlap': contextual_overlap,
                    'temporal_proximity': temporal_proximity,
                    'reference_match': reference_match,
                    'overall_strength': overall_strength,
                    'bidirectional': self._is_bidirectional_relationship(relationship_type)
                })
        
        # Sort by overall strength
        relationships.sort(key=lambda x: x['overall_strength'], reverse=True)
        
        return relationships 
   def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and tokenize
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_contextual_overlap(self, context1: str, context2: str, 
                                     tags1: List[str], tags2: List[str],
                                     category1: str, category2: str) -> float:
        """Calculate contextual overlap between two memories."""
        # Compare contexts
        context_similarity = self._calculate_text_similarity(context1, context2)
        
        # Compare tags
        tag_similarity = 0.0
        if tags1 and tags2:
            common_tags = set(tags1).intersection(set(tags2))
            tag_similarity = len(common_tags) / max(len(tags1), len(tags2))
        
        # Compare categories
        category_similarity = 1.0 if category1 and category1 == category2 else 0.0
        
        # Combined contextual overlap score
        return (context_similarity * 0.4) + (tag_similarity * 0.4) + (category_similarity * 0.2)
    
    def _calculate_temporal_proximity(self, source_created_at: Optional[str], target_created_at: Optional[str]) -> float:
        """Calculate temporal proximity between two memories."""
        if not source_created_at or not target_created_at:
            return 0.5
        
        try:
            # Parse dates
            if isinstance(source_created_at, str):
                date1 = datetime.fromisoformat(source_created_at.replace('Z', '+00:00'))
            else:
                date1 = source_created_at
                
            if isinstance(target_created_at, str):
                date2 = datetime.fromisoformat(target_created_at.replace('Z', '+00:00'))
            else:
                date2 = target_created_at
            
            # Calculate time difference in days
            time_diff = abs((date1 - date2).total_seconds()) / 86400  # Convert to days
            
            # Score based on proximity
            if time_diff < 0.01:  # Less than 15 minutes
                return 1.0
            elif time_diff < 0.1:  # Less than 2.4 hours
                return 0.9
            elif time_diff < 1:  # Same day
                return 0.8
            elif time_diff < 7:  # Within a week
                return 0.6
            elif time_diff < 30:  # Within a month
                return 0.4
            elif time_diff < 90:  # Within 3 months
                return 0.2
            else:
                return 0.1
                
        except (ValueError, TypeError):
            return 0.5
    
    def _check_references(self, text1: str, text2: str, id1: int, id2: int) -> float:
        """Check for references between two memories."""
        # Check for direct references
        if str(id1) in text2 or str(id2) in text1:
            return 1.0
        
        # Extract potential titles (first sentence or first line)
        title1 = text1.split('.')[0].strip() if '.' in text1 else text1.split('\n')[0].strip()
        title2 = text2.split('.')[0].strip() if '.' in text2 else text2.split('\n')[0].strip()
        
        if title1 and len(title1) > 5 and title1 in text2:
            return 0.9
        
        if title2 and len(title2) > 5 and title2 in text1:
            return 0.9
        
        # Check for quote references
        quotes1 = re.findall(r'"([^"]+)"', text1)
        quotes2 = re.findall(r'"([^"]+)"', text2)
        
        for quote in quotes1:
            if len(quote) > 10 and quote in text2:
                return 0.8
        
        for quote in quotes2:
            if len(quote) > 10 and quote in text1:
                return 0.8
        
        return 0.0
    
    def _determine_relationship_type(self, text1: str, text2: str,
                                    semantic_similarity: float, contextual_overlap: float,
                                    temporal_proximity: float, reference_match: float) -> str:
        """Determine the type of relationship between two memories."""
        # Check for reference relationship
        if reference_match > 0.7:
            return 'reference'
        
        # Check for continuation relationship
        if temporal_proximity > 0.8 and semantic_similarity > 0.6:
            return 'continuation'
        
        # Check for elaboration relationship
        if semantic_similarity > 0.7 and len(text2) > len(text1) * 1.5:
            return 'elaboration'
        
        # Check for summary relationship
        if semantic_similarity > 0.7 and len(text1) > len(text2) * 1.5:
            return 'summary'
        
        # Check for contradiction relationship
        contradiction_score = self._check_contradictions(text1, text2)
        if contradiction_score > 0.6 and semantic_similarity > 0.5:
            return 'contradiction'
        
        # Default to semantic relationship based on similarity
        if semantic_similarity > 0.8:
            return 'strong_semantic'
        elif semantic_similarity > 0.6:
            return 'moderate_semantic'
        else:
            return 'weak_semantic'
    
    def _check_contradictions(self, text1: str, text2: str) -> float:
        """Check for contradictions between two texts."""
        # Simple contradiction detection based on negation patterns
        contradiction_score = 0.0
        
        # Look for opposite statements
        negation_words = ['not', 'no', "n't", 'never', 'none', 'neither', 'nor']
        
        # Extract sentences
        sentences1 = re.split(r'[.!?]', text1)
        sentences2 = re.split(r'[.!?]', text2)
        
        # Compare each sentence pair
        for s1 in sentences1:
            s1 = s1.strip().lower()
            if not s1:
                continue
                
            for s2 in sentences2:
                s2 = s2.strip().lower()
                if not s2:
                    continue
                
                # Check if sentences are similar but one has negation
                similarity = self._calculate_text_similarity(s1, s2)
                
                if similarity > 0.5:
                    s1_has_negation = any(neg in s1.split() for neg in negation_words)
                    s2_has_negation = any(neg in s2.split() for neg in negation_words)
                    
                    if s1_has_negation != s2_has_negation:
                        contradiction_score = max(contradiction_score, similarity)
        
        return contradiction_score
    
    def _is_bidirectional_relationship(self, relationship_type: str) -> bool:
        """Determine if a relationship type is bidirectional."""
        bidirectional_types = [
            'strong_semantic', 'moderate_semantic', 'weak_semantic',
            'contradiction', 'reference'
        ]
        
        return relationship_type in bidirectional_types 
   def store_memory_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        Store detected relationships in the database.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Number of relationships stored
        """
        if not relationships:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        
        for rel in relationships:
            # Check if relationship already exists
            cursor.execute("""
                SELECT id FROM enhanced_memory_relationships 
                WHERE source_memory_id = ? AND target_memory_id = ? AND relationship_type = ?
            """, (rel['source_memory_id'], rel['target_memory_id'], rel['relationship_type']))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing relationship
                cursor.execute("""
                    UPDATE enhanced_memory_relationships SET
                    semantic_similarity = ?,
                    contextual_overlap = ?,
                    temporal_proximity = ?,
                    reference_match = ?,
                    overall_strength = ?,
                    confidence = ?,
                    bidirectional = ?,
                    updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    rel['semantic_similarity'],
                    rel['contextual_overlap'],
                    rel['temporal_proximity'],
                    rel['reference_match'],
                    rel['overall_strength'],
                    rel.get('confidence', 0.5),
                    1 if rel.get('bidirectional', False) else 0,
                    existing[0]
                ))
            else:
                # Insert new relationship
                cursor.execute("""
                    INSERT INTO enhanced_memory_relationships (
                        source_memory_id, target_memory_id, relationship_type,
                        semantic_similarity, contextual_overlap, temporal_proximity,
                        reference_match, overall_strength, confidence, bidirectional
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel['source_memory_id'],
                    rel['target_memory_id'],
                    rel['relationship_type'],
                    rel['semantic_similarity'],
                    rel['contextual_overlap'],
                    rel['temporal_proximity'],
                    rel['reference_match'],
                    rel['overall_strength'],
                    rel.get('confidence', 0.5),
                    1 if rel.get('bidirectional', False) else 0
                ))
            
            stored_count += 1
            
            # If bidirectional, create reverse relationship
            if rel.get('bidirectional', False):
                # Check if reverse relationship already exists
                cursor.execute("""
                    SELECT id FROM enhanced_memory_relationships 
                    WHERE source_memory_id = ? AND target_memory_id = ? AND relationship_type = ?
                """, (rel['target_memory_id'], rel['source_memory_id'], rel['relationship_type']))
                
                existing_reverse = cursor.fetchone()
                
                if not existing_reverse:
                    # Insert reverse relationship
                    cursor.execute("""
                        INSERT INTO enhanced_memory_relationships (
                            source_memory_id, target_memory_id, relationship_type,
                            semantic_similarity, contextual_overlap, temporal_proximity,
                            reference_match, overall_strength, confidence, bidirectional
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rel['target_memory_id'],
                        rel['source_memory_id'],
                        rel['relationship_type'],
                        rel['semantic_similarity'],
                        rel['contextual_overlap'],
                        rel['temporal_proximity'],
                        rel['reference_match'],
                        rel['overall_strength'],
                        rel.get('confidence', 0.5),
                        1
                    ))
                    
                    stored_count += 1
        
        conn.commit()
        conn.close()
        
        return stored_count
    
    def consolidate_memories(self, memory_ids: List[int], consolidation_type: str = 'merge') -> Dict[str, Any]:
        """
        Consolidate multiple memories into a single memory.
        
        Args:
            memory_ids: List of memory IDs to consolidate
            consolidation_type: Type of consolidation ('merge', 'summarize', 'compress')
            
        Returns:
            Dictionary with consolidation results
        """
        if not memory_ids or len(memory_ids) < 2:
            return {'error': 'At least two memories are required for consolidation'}
        
        # Get memory data for all IDs
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        memories = []
        for memory_id in memory_ids:
            cursor.execute("""
                SELECT id, text, memory_type, priority, context, tags, category,
                       quality_score, confidence_score, completeness_score, relevance_score,
                       created_at, updated_at
                FROM advanced_memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                tags = json.loads(row[5]) if row[5] else []
                memories.append({
                    'id': row[0],
                    'text': row[1],
                    'memory_type': row[2],
                    'priority': row[3],
                    'context': row[4],
                    'tags': tags,
                    'category': row[6],
                    'quality_score': row[7],
                    'confidence_score': row[8],
                    'completeness_score': row[9],
                    'relevance_score': row[10],
                    'created_at': row[11],
                    'updated_at': row[12]
                })
        
        if len(memories) < 2:
            conn.close()
            return {'error': 'Could not find at least two valid memories'}
        
        # Sort memories by quality score
        memories.sort(key=lambda m: m.get('quality_score', 0), reverse=True)
        
        # Base the consolidated memory on the highest quality memory
        base_memory = memories[0]
        
        # Perform consolidation based on type
        if consolidation_type == 'merge':
            # Combine all text content
            all_text = [base_memory.get('text', '')]
            for memory in memories[1:]:
                text = memory.get('text', '')
                if text:
                    all_text.append(text)
            
            merged_text = "\n\n".join(all_text)
            
            # Combine contexts
            contexts = [m.get('context', '') for m in memories if m.get('context')]
            merged_context = '; '.join(set(contexts)) if contexts else base_memory.get('context', '')
            
            # Combine tags
            all_tags = []
            for memory in memories:
                all_tags.extend(memory.get('tags', []))
            merged_tags = list(set(all_tags))
            
            # Use the most common category
            categories = [m.get('category', '') for m in memories if m.get('category')]
            category_counts = Counter(categories)
            merged_category = category_counts.most_common(1)[0][0] if category_counts else base_memory.get('category', '')
            
            # Use highest priority
            priority = max(m.get('priority', 0.5) for m in memories)
            
            # Use most common memory type
            memory_types = [m.get('memory_type', 'general') for m in memories]
            memory_type_counts = Counter(memory_types)
            memory_type = memory_type_counts.most_common(1)[0][0]
            
            # Calculate quality scores
            quality_score = sum(m.get('quality_score', 0.5) for m in memories) / len(memories)
            confidence_score = sum(m.get('confidence_score', 0.5) for m in memories) / len(memories)
            completeness_score = sum(m.get('completeness_score', 0.5) for m in memories) / len(memories)
            relevance_score = sum(m.get('relevance_score', 0.5) for m in memories) / len(memories)
            
            # Calculate compression ratio
            total_original_length = sum(len(m.get('text', '')) for m in memories)
            compression_ratio = len(merged_text) / total_original_length if total_original_length > 0 else 1.0
            
            # Calculate information retention (simple heuristic)
            unique_words_original = set()
            for memory in memories:
                unique_words_original.update(re.findall(r'\b\w+\b', memory.get('text', '').lower()))
            
            unique_words_merged = set(re.findall(r'\b\w+\b', merged_text.lower()))
            information_retention = len(unique_words_merged) / len(unique_words_original) if unique_words_original else 1.0
            
            # Create new memory
            tags_json = json.dumps(merged_tags)
            
            cursor.execute("""
                INSERT INTO advanced_memories 
                (text, memory_type, priority, context, tags, category,
                 quality_score, confidence_score, completeness_score, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (merged_text, memory_type, priority, merged_context, tags_json, merged_category,
                  quality_score, confidence_score, completeness_score, relevance_score))
            
            new_memory_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return {
                'new_memory_id': new_memory_id,
                'compression_ratio': compression_ratio,
                'information_retention': information_retention,
                'merged_text': merged_text,
                'merged_context': merged_context,
                'merged_tags': merged_tags,
                'merged_category': merged_category
            }
        else:
            conn.close()
            return {'error': f'Consolidation type {consolidation_type} not implemented'}
    
    def optimize_memory_storage(self, max_candidates: int = 100, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Optimize memory storage by identifying and consolidating similar memories.
        
        Args:
            max_candidates: Maximum number of memories to analyze
            similarity_threshold: Minimum similarity threshold for consolidation
            
        Returns:
            Dictionary with optimization results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get candidate memories
        cursor.execute("""
            SELECT id FROM advanced_memories 
            ORDER BY quality_score DESC, updated_at DESC
            LIMIT ?
        """, (max_candidates,))
        
        memory_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not memory_ids:
            return {'error': 'No memories found for optimization'}
        
        # Find clusters of similar memories
        clusters = []
        processed_ids = set()
        
        for memory_id in memory_ids:
            if memory_id in processed_ids:
                continue
                
            # Find similar memories
            relationships = self.detect_memory_relationships(memory_id, max_candidates)
            
            # Filter by similarity threshold
            similar_memories = [r['target_memory_id'] for r in relationships 
                               if r['semantic_similarity'] >= similarity_threshold]
            
            if similar_memories:
                cluster = [memory_id] + similar_memories
                clusters.append(cluster)
                processed_ids.update(cluster)
        
        # Consolidate each cluster
        results = []
        for cluster in clusters:
            result = self.consolidate_memories(cluster, 'merge')
            if 'error' not in result:
                results.append({
                    'cluster': cluster,
                    'new_memory_id': result.get('new_memory_id'),
                    'compression_ratio': result.get('compression_ratio'),
                    'information_retention': result.get('information_retention')
                })
        
        return {
            'clusters_found': len(clusters),
            'memories_consolidated': sum(len(cluster) for cluster in clusters),
            'new_memories_created': len(results),
            'average_compression_ratio': sum(r.get('compression_ratio', 0) for r in results) / len(results) if results else 0,
            'average_information_retention': sum(r.get('information_retention', 0) for r in results) / len(results) if results else 0,
            'detailed_results': results
        }