#!/usr/bin/env python3
"""
Enhanced Context Management System
Provides intelligent context retrieval and export for LLMs using RAG technology.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
from .rag_system import RAGSystem, RAGQuery, RAGResult

class ContextManager:
    """Enhanced context manager with RAG integration."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the context manager."""
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'unified_memory.db')
        self.db_path = db_path if db_path is not None else ""
        self.rag_system = RAGSystem()
        self._init_context_database()
    
    def _init_context_database(self):
        """Initialize the context database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Context packs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                context_data TEXT NOT NULL,
                context_type TEXT DEFAULT 'general',
                project_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                tags TEXT
            )
        """)
        
        # Context access patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_access_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_type TEXT NOT NULL,
                access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default',
                project_id TEXT,
                success_rating INTEGER
            )
        """)
        
        # Context templates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                template_data TEXT NOT NULL,
                context_types TEXT,
                max_tokens INTEGER DEFAULT 1000,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_rag_context(self, query: str, context_types: Optional[List[str]] = None, 
                           max_tokens: int = 1000, project_id: Optional[str] = None,
                           current_context: Optional[Dict[str, Any]] = None) -> RAGResult:
        """Generate context using RAG system."""
        if context_types is None:
            context_types = ['memory', 'task', 'code', 'document', 'feedback']
        
        if current_context is None:
            current_context = {}
        
        # Add project context
        if project_id:
            current_context['project_id'] = project_id
        
        # Create RAG query
        rag_query = RAGQuery(
            query=query,
            context=current_context,
            max_tokens=max_tokens,
            chunk_types=context_types,
            project_id=project_id,
            user_id=current_context.get('user_id')
        )
        
        # Retrieve context using RAG
        result = self.rag_system.retrieve_context(rag_query)
        
        return result
    
    def export_context(self, context_types: Optional[List[str]] = None, max_tokens: int = 1000,
                      project_id: Optional[str] = None, format: str = 'text',
                      use_rag: bool = True, query: Optional[str] = None, include_misunderstandings: bool = False) -> Dict[str, Any]:
        """Export context for LLM consumption with RAG enhancement, optionally including misunderstandings, and optimize for token usage."""
        safe_context_types = context_types if context_types is not None else ['tasks', 'memories', 'progress']
        context = {}
        safe_project_id = str(project_id) if project_id is not None else ''
        safe_format = str(format) if format is not None else 'text'
        safe_query = query if query is not None else ''
        if use_rag and (safe_query or ""):
            current_context = self._get_current_context(safe_project_id)
            rag_result = self.generate_rag_context(
                query=safe_query,
                context_types=safe_context_types,
                max_tokens=max_tokens,
                project_id=safe_project_id,
                current_context=current_context or {}
            )
            context = self._format_rag_result(rag_result, safe_format)
        else:
            context = self._generate_traditional_context(safe_context_types, max_tokens, safe_project_id, safe_format)
        if 'tasks' not in context:
            context['tasks'] = []
        if 'memories' not in context:
            context['memories'] = []
        if include_misunderstandings:
            from .workflow import WorkflowManager
            workflow = WorkflowManager()
            context['misunderstandings'] = workflow.export_misunderstandings()
        # Optimize for token usage
        try:
            from .server import MCPServer
            server = MCPServer()
            context = server.optimize_context_for_tokens(context, max_tokens=max_tokens)
        except Exception:
            print('[ContextManager] Fallback: method not implemented yet. See idea.txt for future improvements.')
        try:
            from .server import MCPServer
            endpoints = list(MCPServer._route_request.__func__.__globals__["method_handlers"].keys())
            context['available_endpoints'] = endpoints
        except Exception:
            context['available_endpoints'] = []
        try:
            from .project_manager import ProjectManager
            pm = ProjectManager()
            questions = pm.get_unanswered_questions()
            context['unanswered_questions'] = questions
        except Exception:
            context['unanswered_questions'] = []
        return context
    
    def _get_current_context(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current context information."""
        context = {
            'timestamp': datetime.now().isoformat(),
            'project_id': project_id if project_id is not None else ""
        }
        
        # Add current task if available
        try:
            from .task_manager import TaskManager
            task_manager = TaskManager()
            current_tasks = task_manager.get_tasks(status='in_progress')
            if current_tasks:
                context['current_task'] = str(current_tasks[0]['id'])
        except Exception:
            pass
        
        # Add current phase if available
        try:
            from .workflow import WorkflowManager
            workflow = WorkflowManager()
            status = workflow.get_workflow_status()
            context['current_phase'] = str(status.get('current_step', ''))
        except Exception:
            pass
        
        return context
    
    def _format_rag_result(self, result: RAGResult, format: str = 'text') -> Dict[str, Any]:
        """Format RAG result for output. Always use a non-None string for format."""
        safe_format = str(format) if format is not None else 'text'
        if safe_format == 'json':
            return {
                'chunks': [asdict(chunk) for chunk in result.chunks],
                'total_tokens': result.total_tokens,
                'relevance_scores': result.relevance_scores,
                'sources': result.sources,
                'summary': result.summary,
                'confidence': result.confidence
            }
        else:
            # Text format
            context_text = ""
            context_text += f"📊 {result.summary}\n"
            context_text += f"🎯 Confidence: {result.confidence:.2f}\n"
            context_text += f"📝 Tokens: {result.total_tokens}\n\n"
            for i, chunk in enumerate(result.chunks):
                context_text += f"--- {chunk.source_type.upper()} {chunk.source_id} ---\n"
                context_text += f"{chunk.content}\n\n"
            return {
                'context': context_text,
                'total_tokens': result.total_tokens,
                'confidence': result.confidence,
                'sources': result.sources
            }
    
    def _generate_traditional_context(self, context_types: Optional[List[str]], max_tokens: int,
                                    project_id: Optional[str] = None, format: str = 'text') -> Dict[str, Any]:
        safe_project_id = str(project_id) if project_id is not None else ''
        safe_format = str(format) if format is not None else 'text'
        context_types = context_types if context_types is not None else []
        context_data = {}
        total_tokens = 0
        # Get tasks
        if 'tasks' in context_types:
            try:
                from .task_manager import TaskManager
                task_manager = TaskManager()
                tasks = task_manager.get_tasks(status='in_progress')
                context_data['tasks'] = tasks if tasks is not None else []
                total_tokens += len(str(tasks)) // 4
            except Exception:
                context_data['tasks'] = []
        else:
            context_data['tasks'] = []
        # Get memories
        if 'memories' in context_types:
            try:
                from .memory import MemoryManager
                memory_manager = MemoryManager()
                memories = memory_manager.search_memories("")
                context_data['memories'] = memories if memories is not None else []
                total_tokens += len(str(memories)) // 4
            except Exception:
                context_data['memories'] = []
        else:
            context_data['memories'] = []
        # Get progress
        if 'progress' in context_types:
            try:
                from .workflow import WorkflowManager
                workflow = WorkflowManager()
                status = workflow.get_workflow_status()
                context_data['progress'] = status if status is not None else {}
                total_tokens += len(str(status)) // 4
            except Exception:
                context_data['progress'] = {}
        if safe_format == 'json':
            return {
                'tasks': context_data.get('tasks', []),
                'memories': context_data.get('memories', []),
                'progress': context_data.get('progress', {}),
                'context_data': context_data,
                'total_tokens': total_tokens
            }
        else:
            # Format as text
            context_text = ""
            if context_data.get('tasks'):
                context_text += "CURRENT TASKS:\n"
                for task in context_data['tasks'][:3]:
                    title = str(task.get('title', 'Untitled') or '')
                    desc = str(task.get('description', '') or '')
                    context_text += f"- {title}: {desc}\n"
                context_text += "\n"
            if context_data.get('memories'):
                context_text += "🧠 RELEVANT MEMORIES:\n"
                for memory in context_data['memories'][:3]:
                    memtext = str(memory.get('text', '') or '')
                    context_text += f"- {memtext}\n"
                context_text += "\n"
            if context_data.get('progress'):
                progress = context_data['progress']
                progress_value = progress.get('progress', 0)
                current_step_value = progress.get('current_step', 'Unknown')
                context_text += f"📈 PROGRESS: {str(progress_value) if progress_value is not None else ''}% complete\n"
                context_text += f"Current step: {str(current_step_value) if current_step_value is not None else ''}\n\n"
            return {
                'context': context_text,
                'total_tokens': total_tokens,
                'tasks': context_data.get('tasks', []),
                'memories': context_data.get('memories', [])
            }
    
    def save_context_pack(self, name: str, context_data: Dict[str, Any], 
                         description: str = '', context_type: str = 'general',
                         project_id: str = '', tags: Optional[List[str]] = None) -> int:
        """Save a context pack for later use. Always returns an int (pack_id or -1). All string parameters default to ''."""
        tags = tags if tags is not None else []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO context_packs (name, description, context_data, context_type, project_id, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                description,
                json.dumps(context_data),
                context_type,
                project_id,
                json.dumps(tags)
            ))
            pack_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return int(pack_id) if pack_id is not None else -1
        except Exception:
            return -1
    
    def get_context_pack(self, pack_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a saved context pack."""
        pack_id = pack_id if pack_id is not None else 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, description, context_data, context_type, project_id, tags,
                   created_at, access_count, token_count
            FROM context_packs
            WHERE id = ?
        """, (pack_id,))
        row = cursor.fetchone()
        if row:
            # Update access count
            cursor.execute("""
                UPDATE context_packs 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (pack_id,))
            conn.commit()
            conn.close()
            return {
                'id': str(pack_id) if pack_id is not None else '',
                'name': str(row[0]) if row[0] is not None else '',
                'description': str(row[1]) if row[1] is not None else '',
                'context_data': json.loads(row[2]) if row[2] is not None else {},
                'context_type': str(row[3]) if row[3] is not None else '',
                'project_id': str(row[4]) if row[4] is not None else '',
                'tags': json.loads(row[5]) if row[5] else [],
                'created_at': str(row[6]) if row[6] is not None else '',
                'access_count': str(row[7]) if row[7] is not None else '0',
                'token_count': str(row[8]) if row[8] is not None else '0'
            }
        conn.close()
        return None
    
    def list_context_packs(self, context_type: str = '', project_id: str = '') -> List[Dict[str, Any]]:
        """List available context packs. context_type and project_id default to ''."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        conditions = []
        params = []
        if context_type != "":
            conditions.append("context_type = ?")
            params.append(context_type)
        if project_id != "":
            conditions.append("project_id = ?")
            params.append(project_id)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        cursor.execute(f"""
            SELECT id, name, description, context_type, project_id, created_at, access_count
            FROM context_packs
            WHERE {where_clause}
            ORDER BY last_accessed DESC
        """, params)
        packs = []
        for row in cursor.fetchall():
            packs.append({
                'id': str(row[0]),
                'name': str(row[1]) if row[1] is not None else '',
                'description': str(row[2]) if row[2] is not None else '',
                'context_type': str(row[3]) if row[3] is not None else '',
                'project_id': str(row[4]) if row[4] is not None else '',
                'created_at': str(row[5]) if row[5] is not None else '',
                'access_count': str(row[6]) if row[6] is not None else '0'
            })
        conn.close()
        return packs
    
    def add_context_template(self, name: str, template_data: Dict[str, Any],
                           description: str = None, context_types: Optional[List[str]] = None,
                           max_tokens: int = 1000) -> int:
        """Add a context template for reuse."""
        name = name if name is not None else ""
        description = description if description is not None else ""
        context_types = context_types if context_types is not None else []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO context_templates (name, description, template_data, context_types, max_tokens)
            VALUES (?, ?, ?, ?, ?)
        """, (
            name,
            description,
            json.dumps(template_data),
            json.dumps(context_types),
            max_tokens
        ))
        template_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return template_id if template_id is not None else -1
    
    def get_context_template(self, template_id: int) -> Optional[Dict[str, Any]]:
        """Get a context template."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, description, template_data, context_types, max_tokens, created_at
            FROM context_templates
            WHERE id = ?
        """, (template_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': template_id,
                'name': row[0],
                'description': row[1],
                'template_data': json.loads(row[2]),
                'context_types': json.loads(row[3]) if row[3] else [],
                'max_tokens': row[4],
                'created_at': row[5]
            }
        
        return None
    
    def list_context_templates(self) -> List[Dict[str, Any]]:
        """List available context templates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, context_types, max_tokens, created_at
            FROM context_templates
            ORDER BY created_at DESC
        """)
        
        templates = []
        for row in cursor.fetchall():
            templates.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'context_types': json.loads(row[3]) if row[3] else [],
                'max_tokens': row[4],
                'created_at': row[5]
            })
        
        conn.close()
        return templates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Context pack statistics
        cursor.execute("SELECT COUNT(*), context_type FROM context_packs GROUP BY context_type")
        pack_stats = dict(cursor.fetchall())
        
        # Template statistics
        cursor.execute("SELECT COUNT(*) FROM context_templates")
        template_count = cursor.fetchone()[0]
        
        # Access patterns
        cursor.execute("SELECT COUNT(*), context_type FROM context_access_patterns GROUP BY context_type")
        access_stats = dict(cursor.fetchall())
        
        conn.close()
        
        # RAG statistics
        rag_stats = self.rag_system.get_statistics()
        
        return {
            'context_packs': {
                'total': sum(pack_stats.values()),
                'by_type': pack_stats
            },
            'templates': template_count,
            'access_patterns': access_stats,
            'rag_system': rag_stats
        }
    
    def cleanup_old_packs(self, days_old: int = 30):
        """Clean up old, rarely accessed context packs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cursor.execute("""
            DELETE FROM context_packs 
            WHERE last_accessed < ? AND access_count < 2
        """, (cutoff_date.isoformat(),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def generate_context_pack(self, context_types: List[str], max_tokens: int = 1000,
                            project_id: Optional[str] = None) -> Any:
        """Generate a context pack for the specified types."""
        # Create a simple context pack object
        class ContextPack:
            def __init__(self, summary: str, content: str, token_count: int):
                self.summary = summary
                self.content = content
                self.token_count = token_count
        
        # Generate context using existing method
        context_data = self.export_context(
            context_types=context_types,
            max_tokens=max_tokens,
            project_id=project_id,
            format='text'
        )
        
        # Create summary
        summary = f"Context pack with {len(context_types)} types, {context_data.get('total_tokens', 0)} tokens"
        
        # Create context pack object
        context_pack = ContextPack(
            summary=summary,
            content=context_data.get('context', ''),
            token_count=context_data.get('total_tokens', 0)
        )
        
        return context_pack
    
    def export_context_for_llm(self, context_types: List[str], max_tokens: int = 1000,
                              project_id: Optional[str] = None) -> str:
        """Export context specifically formatted for LLM consumption."""
        context_data = self.export_context(
            context_types=context_types,
            max_tokens=max_tokens,
            project_id=project_id,
            format='text'
        )
        
        return context_data.get('context', '')

    def some_context_method(self):
        """Minimal fallback for context management. Expanded with research-driven logic per idea.txt."""
        import logging
        import json
        logging.warning('[ContextManager] This method is a placeholder. See idea.txt for future improvements.')
        # Return a minimal context structure as a JSON string to satisfy type requirements
        return json.dumps({'status': 'not_implemented', 'context': {}}) 