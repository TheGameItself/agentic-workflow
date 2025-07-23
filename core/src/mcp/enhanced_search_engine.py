#!/usr/bin/env python3
"""
Enhanced Search Engine for MCP System
Combines grep-style regex search with EARS (Event-Action-Response-System) semantic search
"""

import re
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result"""
    content: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    context_lines: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    search_type: str = "unknown"

@dataclass
class EARSQuery:
    """EARS format query structure"""
    event: str
    action: str
    response: str
    system: str
    context: Optional[Dict[str, Any]] = None

class EnhancedSearchEngine:
    """Enhanced search engine with grep and EARS capabilities"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.search_cache = {}
        self.ears_patterns = self._load_ears_patterns()
        
    def _load_ears_patterns(self) -> Dict[str, re.Pattern]:
        """Load EARS pattern definitions"""
        patterns = {
            'event': re.compile(r'WHEN\s+(.+?)\s+THEN', re.IGNORECASE),
            'action': re.compile(r'THEN\s+(.+?)\s+SHALL', re.IGNORECASE),
            'response': re.compile(r'SHALL\s+(.+?)(?:\s+WHEN|\s*$)', re.IGNORECASE),
            'system': re.compile(r'SHALL\s+(.+?)\s+SHALL', re.IGNORECASE),
        }
        return patterns
    
    def grep_search(self, 
                   pattern: str, 
                   path: str = ".", 
                   file_patterns: str = "*.py,*.js,*.ts,*.md,*.txt,*.json,*.yaml,*.yml",
                   exclude_patterns: str = "__pycache__,*.pyc,*.git*,node_modules,.venv",
                   case_sensitive: bool = False,
                   invert_match: bool = False,
                   line_number: bool = True,
                   context_lines: int = 0,
                   max_results: int = 100) -> List[SearchResult]:
        """Perform grep-style regex search"""
        
        results = []
        search_path = Path(path).resolve()
        
        # Parse file patterns
        include_patterns = [p.strip() for p in file_patterns.split(",")]
        exclude_patterns = [p.strip() for p in exclude_patterns.split(",")]
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return []
        
        # Walk through files
        for root, dirs, files in os.walk(search_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                re.match(exclude, d) for exclude in exclude_patterns
            )]
            
            for file in files:
                # Check if file matches include patterns
                if not any(re.match(include, file) for include in include_patterns):
                    continue
                
                # Check if file should be excluded
                if any(re.match(exclude, file) for exclude in exclude_patterns):
                    continue
                
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        match = regex.search(line)
                        if (match and not invert_match) or (not match and invert_match):
                            # Get context lines
                            context = []
                            if context_lines > 0:
                                start = max(0, i - context_lines - 1)
                                end = min(len(lines), i + context_lines)
                                context = [line.strip() for line in lines[start:end]]
                            
                            result = SearchResult(
                                content=line.strip(),
                                file_path=str(file_path),
                                line_number=i if line_number else None,
                                context_lines=context if context else None,
                                search_type="grep"
                            )
                            results.append(result)
                            
                            if len(results) >= max_results:
                                return results
                                
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    continue
        
        return results
    
    def parse_ears_query(self, query: str) -> Optional[EARSQuery]:
        """Parse EARS format query"""
        try:
            # Extract components using regex patterns
            event_match = self.ears_patterns['event'].search(query)
            action_match = self.ears_patterns['action'].search(query)
            response_match = self.ears_patterns['response'].search(query)
            system_match = self.ears_patterns['system'].search(query)
            
            if not all([event_match, action_match, response_match]):
                return None
            
            return EARSQuery(
                event=event_match.group(1).strip(),
                action=action_match.group(1).strip(),
                response=response_match.group(1).strip(),
                system=system_match.group(1).strip() if system_match else "",
                context={"original_query": query}
            )
        except Exception as e:
            logger.error(f"Error parsing EARS query: {e}")
            return None
    
    def ears_search(self, 
                   query: str, 
                   scope: str = "all",
                   max_results: int = 20,
                   include_metadata: bool = True) -> List[SearchResult]:
        """Perform EARS format search"""
        
        # Parse EARS query
        ears_query = self.parse_ears_query(query)
        if not ears_query:
            logger.warning("Invalid EARS query format")
            return []
        
        results = []
        
        # Search in different scopes
        if scope in ["all", "docs"]:
            results.extend(self._search_docs_ears(ears_query, max_results))
        
        if scope in ["all", "code"]:
            results.extend(self._search_code_ears(ears_query, max_results))
        
        if scope in ["all", "tasks"]:
            results.extend(self._search_tasks_ears(ears_query, max_results))
        
        if scope in ["all", "memories"]:
            results.extend(self._search_memories_ears(ears_query, max_results))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.score or 0, reverse=True)
        
        return results[:max_results]
    
    def _search_docs_ears(self, ears_query: EARSQuery, max_results: int) -> List[SearchResult]:
        """Search documentation for EARS patterns"""
        results = []
        docs_path = self.project_root / "docs"
        
        if not docs_path.exists():
            return results
        
        for file_path in docs_path.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for EARS patterns in documentation
                score = self._calculate_ears_score(content, ears_query)
                if score > 0.3:  # Threshold for relevance
                    result = SearchResult(
                        content=content[:500] + "..." if len(content) > 500 else content,
                        file_path=str(file_path),
                        score=score,
                        search_type="ears_docs",
                        metadata={
                            "event_match": self._find_event_matches(content, ears_query.event),
                            "action_match": self._find_action_matches(content, ears_query.action),
                            "response_match": self._find_response_matches(content, ears_query.response)
                        }
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        return results
    
    def _search_code_ears(self, ears_query: EARSQuery, max_results: int) -> List[SearchResult]:
        """Search code for EARS patterns"""
        results = []
        
        # Search for function definitions, class methods, etc.
        code_patterns = [
            f"def.*{ears_query.action.lower().replace(' ', '_')}",
            f"class.*{ears_query.event.lower().replace(' ', '_')}",
            f"if.*{ears_query.event.lower()}",
            f"when.*{ears_query.event.lower()}",
        ]
        
        for pattern in code_patterns:
            grep_results = self.grep_search(
                pattern=pattern,
                path=str(self.project_root),
                file_patterns="*.py",
                max_results=max_results // len(code_patterns)
            )
            
            for result in grep_results:
                result.search_type = "ears_code"
                result.score = self._calculate_ears_score(result.content, ears_query)
                results.append(result)
        
        return results
    
    def _search_tasks_ears(self, ears_query: EARSQuery, max_results: int) -> List[SearchResult]:
        """Search task database for EARS patterns"""
        results = []
        
        # This would integrate with the task manager
        # For now, return empty results
        return results
    
    def _search_memories_ears(self, ears_query: EARSQuery, max_results: int) -> List[SearchResult]:
        """Search memory system for EARS patterns"""
        results = []
        
        # This would integrate with the memory manager
        # For now, return empty results
        return results
    
    def _calculate_ears_score(self, content: str, ears_query: EARSQuery) -> float:
        """Calculate relevance score for EARS query"""
        score = 0.0
        
        # Event matching
        if self._text_contains(content, ears_query.event):
            score += 0.4
        
        # Action matching
        if self._text_contains(content, ears_query.action):
            score += 0.3
        
        # Response matching
        if self._text_contains(content, ears_query.response):
            score += 0.3
        
        # System matching
        if ears_query.system and self._text_contains(content, ears_query.system):
            score += 0.2
        
        return min(score, 1.0)
    
    def _text_contains(self, text: str, pattern: str) -> bool:
        """Check if text contains pattern (case-insensitive)"""
        return pattern.lower() in text.lower()
    
    def _find_event_matches(self, content: str, event: str) -> List[str]:
        """Find event matches in content"""
        matches = []
        lines = content.split('\n')
        for line in lines:
            if self._text_contains(line, event):
                matches.append(line.strip())
        return matches[:5]  # Limit to 5 matches
    
    def _find_action_matches(self, content: str, action: str) -> List[str]:
        """Find action matches in content"""
        matches = []
        lines = content.split('\n')
        for line in lines:
            if self._text_contains(line, action):
                matches.append(line.strip())
        return matches[:5]  # Limit to 5 matches
    
    def _find_response_matches(self, content: str, response: str) -> List[str]:
        """Find response matches in content"""
        matches = []
        lines = content.split('\n')
        for line in lines:
            if self._text_contains(line, response):
                matches.append(line.strip())
        return matches[:5]  # Limit to 5 matches
    
    def semantic_search(self, 
                       query: str, 
                       scope: str = "all",
                       max_results: int = 10,
                       similarity_threshold: float = 0.5) -> List[SearchResult]:
        """Perform semantic search using vector embeddings"""
        # This would integrate with the vector memory system
        # For now, return empty results
        return []
    
    def search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get search history"""
        # This would integrate with a search history database
        # For now, return empty list
        return []
    
    def clear_cache(self) -> None:
        """Clear search cache"""
        self.search_cache.clear()
        logger.info("Search cache cleared") 