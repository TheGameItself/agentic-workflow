"""
Regex Search Module for MCP Agentic Workflow Accelerator

Provides comprehensive regex search capabilities for both file system and database content.
Supports SQLite regex operations using sqlean-regexp extension approach.
"""

import re
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
import ast


class SearchType(Enum):
    """Types of search operations."""
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    MEMORY = "memory"
    TASK = "task"
    RAG = "rag"
    COMBINED = "combined"


class SearchScope(Enum):
    """Search scope options."""
    CURRENT_PROJECT = "current_project"
    ALL_PROJECTS = "all_projects"
    SPECIFIC_PATH = "specific_path"
    DATABASE_ONLY = "database_only"


@dataclass
class SearchResult:
    """Represents a single search result."""
    content: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    match_text: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[int] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchQuery:
    """Represents a search query with all parameters."""
    pattern: str
    search_type: SearchType
    scope: SearchScope
    case_sensitive: bool = False
    multiline: bool = False
    dot_all: bool = False
    max_results: int = 100
    context_lines: int = 3
    file_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    project_path: Optional[str] = None
    database_path: Optional[str] = None


class RegexSearchEngine:
    """Main regex search engine for the MCP server."""
    
    def __init__(self, project_path: Optional[str] = None, database_path: Optional[str] = None):
        self.project_path = Path(project_path) if project_path else Path.cwd()
        # Use unified memory database as the main database
        self.database_path = database_path or "data/unified_memory.db"
        self.rag_database_path = "data/rag_system.db"
        self.search_history: List[Dict[str, Any]] = []
        self.cache: Dict[str, List[SearchResult]] = {}
        
        # Common file patterns to exclude
        self.default_exclude_patterns = [
            "*.pyc", "*.pyo", "__pycache__", "*.git*", "*.DS_Store",
            "*.log", "*.tmp", "*.cache", "node_modules", ".env*"
        ]
        
        # Common text file patterns to include
        self.default_include_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.html", "*.css",
            "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml",
            "*.cfg", "*.ini", "*.conf", "*.sh", "*.bash", "*.sql"
        ]
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform a comprehensive regex search based on the query parameters.
        
        Args:
            query: SearchQuery object with all search parameters
            
        Returns:
            List of SearchResult objects
        """
        # Check cache first
        cache_key = self._generate_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = []
        
        try:
            if query.search_type in [SearchType.FILE_SYSTEM, SearchType.COMBINED]:
                file_results = self._search_files(query)
                results.extend(file_results)
            
            if query.search_type in [SearchType.DATABASE, SearchType.MEMORY, 
                                   SearchType.TASK, SearchType.RAG, SearchType.COMBINED]:
                db_results = self._search_database(query)
                results.extend(db_results)
            
            # Sort results by relevance/confidence
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit results
            results = results[:query.max_results]
            
            # Cache results
            self.cache[cache_key] = results
            
            # Log search
            self._log_search(query, len(results))
            
        except Exception as e:
            print(f"Error during regex search: {e}")
            return []
        
        return results
    
    def _search_files(self, query: SearchQuery) -> List[SearchResult]:
        """Search files in the file system."""
        results = []
        
        # Determine search paths
        search_paths = self._get_search_paths(query)
        
        # Get file patterns
        include_patterns = query.file_patterns or self.default_include_patterns
        exclude_patterns = query.exclude_patterns or self.default_exclude_patterns
        
        # Compile regex pattern
        flags = 0
        if not query.case_sensitive:
            flags |= re.IGNORECASE
        if query.multiline:
            flags |= re.MULTILINE
        if query.dot_all:
            flags |= re.DOTALL
        
        try:
            pattern = re.compile(query.pattern, flags)
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return []
        
        # Search each path
        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue
            
            # Find matching files
            matching_files = self._find_matching_files(path, include_patterns, exclude_patterns)
            
            # Search each file
            for file_path in matching_files:
                file_results = self._search_single_file(file_path, pattern, query)
                results.extend(file_results)
        
        return results
    
    def _search_database(self, query: SearchQuery) -> List[SearchResult]:
        """Search database content using SQLite regex support."""
        results = []
        
        # Create regex function for SQLite
        def sqlite_regex(pattern, text):
            if text is None:
                return False
            try:
                flags = 0
                if not query.case_sensitive:
                    flags |= re.IGNORECASE
                if query.multiline:
                    flags |= re.MULTILINE
                if query.dot_all:
                    flags |= re.DOTALL
                return bool(re.search(pattern, text, flags))
            except re.error:
                return False
        
        # Search main database (unified memory)
        if os.path.exists(self.database_path):
            try:
                conn = sqlite3.connect(self.database_path)
                conn.create_function('regexp', 2, sqlite_regex)
                
                # Search different tables based on search type
                if query.search_type in [SearchType.MEMORY, SearchType.COMBINED]:
                    memory_results = self._search_memories_table(conn, query)
                    results.extend(memory_results)
                
                if query.search_type in [SearchType.TASK, SearchType.COMBINED]:
                    task_results = self._search_tasks_table(conn, query)
                    results.extend(task_results)
                
                conn.close()
                
            except Exception as e:
                print(f"Main database search error: {e}")
        
        # Search RAG database
        if query.search_type in [SearchType.RAG, SearchType.COMBINED] and os.path.exists(self.rag_database_path):
            try:
                conn = sqlite3.connect(self.rag_database_path)
                conn.create_function('regexp', 2, sqlite_regex)
                
                rag_results = self._search_rag_table(conn, query)
                results.extend(rag_results)
                
                conn.close()
                
            except Exception as e:
                print(f"RAG database search error: {e}")
        
        return results
    
    def _search_memories_table(self, conn: sqlite3.Connection, query: SearchQuery) -> List[SearchResult]:
        """Search the memories table."""
        results = []
        
        sql = """
        SELECT id, text, memory_type, priority, context, tags, created_at
        FROM memories 
        WHERE regexp(?, text) OR regexp(?, context) OR regexp(?, tags)
        ORDER BY priority DESC, created_at DESC
        LIMIT ?
        """
        
        try:
            cursor = conn.execute(sql, (query.pattern, query.pattern, query.pattern, query.max_results))
            
            for row in cursor.fetchall():
                result = SearchResult(
                    content=row[1],
                    source_type="memory",
                    source_id=row[0],
                    match_text=query.pattern,
                    metadata={
                        "memory_type": row[2],
                        "priority": row[3],
                        "context": row[4],
                        "tags": row[5],
                        "created_at": row[6]
                    }
                )
                results.append(result)
                
        except Exception as e:
            print(f"Error searching memories table: {e}")
        
        return results
    
    def _search_tasks_table(self, conn: sqlite3.Connection, query: SearchQuery) -> List[SearchResult]:
        """Search the tasks table."""
        results = []
        
        sql = """
        SELECT id, title, description, status, priority, created_at
        FROM tasks 
        WHERE regexp(?, title) OR regexp(?, description)
        ORDER BY priority DESC, created_at DESC
        LIMIT ?
        """
        
        try:
            cursor = conn.execute(sql, (query.pattern, query.pattern, query.max_results))
            
            for row in cursor.fetchall():
                result = SearchResult(
                    content=f"{row[1]}: {row[2] or ''}",
                    source_type="task",
                    source_id=row[0],
                    match_text=query.pattern,
                    metadata={
                        "title": row[1],
                        "description": row[2],
                        "status": row[3],
                        "priority": row[4],
                        "created_at": row[5]
                    }
                )
                results.append(result)
                
        except Exception as e:
            print(f"Error searching tasks table: {e}")
        
        return results
    
    def _search_rag_table(self, conn: sqlite3.Connection, query: SearchQuery) -> List[SearchResult]:
        """Search the RAG chunks table."""
        results = []
        
        sql = """
        SELECT id, content, source_type, source_id, project_id, metadata, created_at
        FROM rag_chunks 
        WHERE regexp(?, content) OR regexp(?, metadata)
        ORDER BY created_at DESC
        LIMIT ?
        """
        
        try:
            cursor = conn.execute(sql, (query.pattern, query.pattern, query.max_results))
            
            for row in cursor.fetchall():
                result = SearchResult(
                    content=row[1],
                    source_type="rag",
                    source_id=row[0],
                    match_text=query.pattern,
                    metadata={
                        "source_type": row[2],
                        "source_id": row[3],
                        "project_id": row[4],
                        "metadata": row[5],
                        "created_at": row[6]
                    }
                )
                results.append(result)
                
        except Exception as e:
            print(f"Error searching RAG table: {e}")
        
        return results
    
    def _search_single_file(self, file_path: Path, pattern: re.Pattern, query: SearchQuery) -> List[SearchResult]:
        """Search a single file for regex matches."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                matches = list(pattern.finditer(line))
                
                for match in matches:
                    # Get context lines
                    context_before = self._get_context_lines(lines, line_num - 1, query.context_lines, 'before')
                    context_after = self._get_context_lines(lines, line_num + 1, query.context_lines, 'after')
                    
                    result = SearchResult(
                        content=line.rstrip(),
                        file_path=str(file_path),
                        line_number=line_num,
                        column_start=match.start(),
                        column_end=match.end(),
                        match_text=match.group(),
                        context_before=context_before,
                        context_after=context_after,
                        source_type="file",
                        confidence=self._calculate_confidence(match, line)
                    )
                    results.append(result)
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        
        return results
    
    def _get_context_lines(self, lines: List[str], start_line: int, num_lines: int, direction: str) -> str:
        """Get context lines around a match."""
        if direction == 'before':
            start = max(0, start_line - num_lines)
            end = start_line
        else:  # after
            start = start_line
            end = min(len(lines), start_line + num_lines)
        
        context_lines = []
        for i in range(start, end):
            if 0 <= i < len(lines):
                context_lines.append(f"{i + 1:4d}: {lines[i].rstrip()}")
        
        return '\n'.join(context_lines)
    
    def _calculate_confidence(self, match: re.Match, line: str) -> float:
        """Calculate confidence score for a match."""
        # Base confidence
        confidence = 1.0
        
        # Boost for longer matches
        match_length = len(match.group())
        if match_length > 10:
            confidence += 0.2
        
        # Boost for matches at word boundaries
        if match.start() == 0 or not line[match.start() - 1].isalnum():
            confidence += 0.1
        if match.end() == len(line) or not line[match.end()].isalnum():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_search_paths(self, query: SearchQuery) -> List[str]:
        """Get list of paths to search based on scope."""
        if query.scope == SearchScope.CURRENT_PROJECT:
            return [str(self.project_path)]
        elif query.scope == SearchScope.SPECIFIC_PATH and query.project_path:
            return [query.project_path]
        elif query.scope == SearchScope.ALL_PROJECTS:
            # Search current project and parent directories
            paths = [str(self.project_path)]
            parent = self.project_path.parent
            while parent != parent.parent:
                paths.append(str(parent))
                parent = parent.parent
            return paths
        else:
            return [str(self.project_path)]
    
    def _find_matching_files(self, path: Path, include_patterns: List[str], 
                           exclude_patterns: List[str]) -> List[Path]:
        """Find files matching include/exclude patterns."""
        matching_files = []
        
        for include_pattern in include_patterns:
            # Convert glob pattern to regex
            regex_pattern = self._glob_to_regex(include_pattern)
            
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    # Check if file matches include pattern
                    if re.match(regex_pattern, str(file_path)):
                        # Check if file should be excluded
                        should_exclude = False
                        for exclude_pattern in exclude_patterns:
                            exclude_regex = self._glob_to_regex(exclude_pattern)
                            if re.match(exclude_regex, str(file_path)):
                                should_exclude = True
                                break
                        
                        if not should_exclude:
                            matching_files.append(file_path)
        
        return list(set(matching_files))  # Remove duplicates
    
    def _glob_to_regex(self, glob_pattern: str) -> str:
        """Convert glob pattern to regex pattern."""
        # Simple glob to regex conversion
        regex = glob_pattern.replace('.', r'\.')
        regex = regex.replace('*', r'.*')
        regex = regex.replace('?', r'.')
        return f"^{regex}$"
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for a search query."""
        query_dict = {
            'pattern': query.pattern,
            'search_type': query.search_type.value,
            'scope': query.scope.value,
            'case_sensitive': query.case_sensitive,
            'multiline': query.multiline,
            'dot_all': query.dot_all,
            'max_results': query.max_results,
            'context_lines': query.context_lines,
            'file_patterns': query.file_patterns,
            'exclude_patterns': query.exclude_patterns,
            'project_path': query.project_path
        }
        
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _log_search(self, query: SearchQuery, result_count: int):
        """Log search operation for history."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'pattern': query.pattern,
            'search_type': query.search_type.value,
            'scope': query.scope.value,
            'result_count': result_count,
            'max_results': query.max_results
        }
        
        self.search_history.append(log_entry)
        
        # Keep only last 100 searches
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
    
    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent search history."""
        return self.search_history[-limit:]
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'total_cached_results': sum(len(results) for results in self.cache.values()),
            'search_history_size': len(self.search_history)
        }


class RegexSearchFormatter:
    """Formats search results for different output formats."""
    
    @staticmethod
    def format_results(results: List[SearchResult], format_type: str = 'text', 
                      include_context: bool = True) -> str:
        """Format search results for output."""
        if format_type == 'json':
            return RegexSearchFormatter._format_json(results)
        elif format_type == 'compact':
            return RegexSearchFormatter._format_compact(results)
        else:
            return RegexSearchFormatter._format_text(results, include_context)
    
    @staticmethod
    def _format_text(results: List[SearchResult], include_context: bool) -> str:
        """Format results as human-readable text."""
        if not results:
            return "No matches found."
        
        output = [f"Found {len(results)} matches:\n"]
        
        for i, result in enumerate(results, 1):
            output.append(f"{i}. ")
            
            if result.file_path:
                output.append(f"File: {result.file_path}")
                if result.line_number:
                    output.append(f":{result.line_number}")
                output.append("\n")
            elif result.source_type:
                output.append(f"Source: {result.source_type}")
                if result.source_id:
                    output.append(f" (ID: {result.source_id})")
                output.append("\n")
            
            if result.match_text:
                output.append(f"   Match: {result.match_text}\n")
            
            if include_context and result.context_before:
                output.append(f"   Context before:\n{result.context_before}\n")
            
            output.append(f"   Content: {result.content}\n")
            
            if include_context and result.context_after:
                output.append(f"   Context after:\n{result.context_after}\n")
            
            if result.metadata:
                output.append(f"   Metadata: {json.dumps(result.metadata, indent=2)}\n")
            
            output.append("-" * 50 + "\n")
        
        return "".join(output)
    
    @staticmethod
    def _format_json(results: List[SearchResult]) -> str:
        """Format results as JSON."""
        json_results = []
        
        for result in results:
            json_result = {
                'content': result.content,
                'file_path': result.file_path,
                'line_number': result.line_number,
                'column_start': result.column_start,
                'column_end': result.column_end,
                'match_text': result.match_text,
                'context_before': result.context_before,
                'context_after': result.context_after,
                'source_type': result.source_type,
                'source_id': result.source_id,
                'confidence': result.confidence,
                'metadata': result.metadata
            }
            json_results.append(json_result)
        
        return json.dumps(json_results, indent=2)
    
    @staticmethod
    def _format_compact(results: List[SearchResult]) -> str:
        """Format results in compact format."""
        if not results:
            return "No matches found."
        
        output = [f"Found {len(results)} matches:\n"]
        
        for i, result in enumerate(results, 1):
            if result.file_path:
                output.append(f"{i}. {result.file_path}")
                if result.line_number:
                    output.append(f":{result.line_number}")
            elif result.source_type:
                output.append(f"{i}. {result.source_type}")
                if result.source_id:
                    output.append(f":{result.source_id}")
            
            if result.match_text:
                output.append(f" -> {result.match_text}")
            
            output.append("\n")
        
        return "".join(output)

def extract_docstrings_from_file(file_path: str) -> list:
    """Extract all docstrings from a Python file. Returns a list of (file, line, docstring) tuples."""
    docstrings = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                doc = ast.get_docstring(node)
                if doc:
                    if isinstance(node, ast.Module):
                        lineno = 1
                    else:
                        lineno = getattr(node, 'lineno', 1)
                    docstrings.append((file_path, lineno, doc))
    except Exception as e:
        print(f"[extract_docstrings_from_file] Error in {file_path}: {e}")
    return docstrings

def extract_all_docstrings(project_root: str) -> list:
    """Extract all docstrings from all Python files in the project."""
    import os
    all_docstrings = []
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_docstrings.extend(extract_docstrings_from_file(file_path))
    return all_docstrings 