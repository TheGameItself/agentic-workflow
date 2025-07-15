#!/usr/bin/env python3
"""
Simplified MCP CLI Entry Point
A minimal CLI that works with system Python and available packages.
"""

import sys
import os
import json
import sqlite3
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_db_path():
    """Get the path to the main memory database."""
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / 'unified_memory.db')

def init_project(name, path=None):
    """Initialize a new project."""
    if path is None:
        path = os.getcwd()
    
    project_dir = os.path.join(path, name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Create config file
    config_file = os.path.join(project_dir, 'config.cfg')
    config_content = f"""[PROJECT]
name = {name}
created_at = {os.popen('date').read().strip()}
status = initializing

[ALIGNMENT]
project_goal = 
target_users = 
key_features = 
technical_constraints = 
timeline = 
success_metrics = 

[RESEARCH]
unknown_technologies = 
competitor_analysis = 
user_research_needed = 
technical_risks = 
compliance_requirements = 
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"[MCP] Project '{name}' initialized successfully!")
    print(f"[MCP] Project path: {project_dir}")
    print(f"[MCP] Configuration file: {config_file}")
    print("\n[MCP] Next steps:")
    print("  1. Navigate to the project directory")
    print("  2. Edit the config.cfg file to answer alignment questions")
    print("  3. Use 'python simple_mcp_cli.py show-questions' to see questions")

def show_questions():
    """Show configuration questions."""
    config_file = None
    
    # Look for config.cfg in current directory or parent directories
    current = os.getcwd()
    while current != os.path.dirname(current):
        test_config = os.path.join(current, 'config.cfg')
        if os.path.exists(test_config):
            config_file = test_config
            break
        current = os.path.dirname(current)
    
    if not config_file:
        print("[MCP] No project configuration found. Run 'init-project' first.")
        return
    
    print("[MCP] Configuration Questions:")
    print("=" * 50)
    
    # Simple config parser
    current_section = None
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                if current_section != 'PROJECT':
                    print(f"\n[{current_section.upper()}]")
            elif '=' in line and current_section and current_section != 'PROJECT':
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                status = "✓" if value else "○"
                print(f"  {status} {key}: {value or '(not answered)'}")

def add_memory(text, memory_type='general', priority=0.5):
    """Add a memory to the database."""
    db_path = get_db_path()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            memory_type TEXT DEFAULT 'general',
            priority REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        INSERT INTO memories (text, memory_type, priority)
        VALUES (?, ?, ?)
    """, (text, memory_type, priority))
    
    memory_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"[MCP] Memory added with ID: {memory_id}")

def search_memories(query, limit=10):
    """Search memories."""
    db_path = get_db_path()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, text, memory_type, priority, created_at
        FROM memories 
        WHERE text LIKE ?
        ORDER BY priority DESC, created_at DESC
        LIMIT ?
    """, (f'%{query}%', limit))
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("[MCP] No memories found matching your query.")
        return
    
    print(f"[MCP] Found {len(results)} memories:")
    for memory_id, text, memory_type, priority, created_at in results:
        print(f"  ID: {memory_id} | Type: {memory_type} | Priority: {priority}")
        print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  Created: {created_at}")
        print()

def create_task(title, description=None, priority=5):
    """Create a task."""
    db_path = get_db_path()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        INSERT INTO tasks (title, description, priority)
        VALUES (?, ?, ?)
    """, (title, description, priority))
    
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"[MCP] Task created with ID: {task_id}")

def list_tasks():
    """List all tasks."""
    db_path = get_db_path()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, description, status, priority, created_at
        FROM tasks 
        ORDER BY priority DESC, created_at DESC
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("[MCP] No tasks found.")
        return
    
    print(f"[MCP] Found {len(results)} tasks:")
    for task_id, title, description, status, priority, created_at in results:
        print(f"  ID: {task_id} | Status: {status} | Priority: {priority}")
        print(f"  Title: {title}")
        if description:
            print(f"  Description: {description[:100]}{'...' if len(description) > 100 else ''}")
        print(f"  Created: {created_at}")
        print()

def export_context(types='tasks,memories', max_tokens=1000):
    """Export context for LLM consumption."""
    db_path = get_db_path()
    
    context = {}
    
    if 'memories' in types:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT text, memory_type, priority
            FROM memories 
            ORDER BY priority DESC, created_at DESC
            LIMIT 20
        """)
        memories = cursor.fetchall()
        conn.close()
        
        context['memories'] = [
            {'text': text, 'type': memory_type, 'priority': priority}
            for text, memory_type, priority in memories
        ]
    
    if 'tasks' in types:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT title, description, status, priority
            FROM tasks 
            ORDER BY priority DESC, created_at DESC
            LIMIT 20
        """)
        tasks = cursor.fetchall()
        conn.close()
        
        context['tasks'] = [
            {'title': title, 'description': description, 'status': status, 'priority': priority}
            for title, description, status, priority in tasks
        ]
    
    # Convert to JSON and limit tokens
    context_json = json.dumps(context, indent=2)
    if len(context_json) > max_tokens:
        context_json = context_json[:max_tokens] + "..."
    
    print("[MCP] Context Export:")
    print("=" * 50)
    print(context_json)

def show_help():
    """Show help information."""
    print("MCP Agentic Workflow Accelerator - Simplified CLI")
    print("=" * 50)
    print()
    print("Available commands:")
    print("  init-project <name> [path]  - Initialize a new project")
    print("  show-questions              - Show configuration questions")
    print("  add-memory <text> [type] [priority] - Add a memory")
    print("  search-memories <query> [limit] - Search memories")
    print("  create-task <title> [description] [priority] - Create a task")
    print("  list-tasks                  - List all tasks")
    print("  export-context [types] [max-tokens] - Export context for LLM")
    print("  help                        - Show this help")
    print()
    print("Examples:")
    print("  python simple_mcp_cli.py init-project my_app")
    print("  python simple_mcp_cli.py add-memory 'User prefers dark theme' design 0.8")
    print("  python simple_mcp_cli.py create-task 'Design API' 'Create REST endpoints' 8")
    print("  python simple_mcp_cli.py export-context tasks,memories 500")

def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == 'init-project':
            if len(sys.argv) < 3:
                print("[MCP] Error: Project name required")
                return
            name = sys.argv[2]
            path = sys.argv[3] if len(sys.argv) > 3 else None
            init_project(name, path)
        
        elif command == 'show-questions':
            show_questions()
        
        elif command == 'add-memory':
            if len(sys.argv) < 3:
                print("[MCP] Error: Memory text required")
                return
            text = sys.argv[2]
            memory_type = sys.argv[3] if len(sys.argv) > 3 else 'general'
            priority = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
            add_memory(text, memory_type, priority)
        
        elif command == 'search-memories':
            if len(sys.argv) < 3:
                print("[MCP] Error: Search query required")
                return
            query = sys.argv[2]
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            search_memories(query, limit)
        
        elif command == 'create-task':
            if len(sys.argv) < 3:
                print("[MCP] Error: Task title required")
                return
            title = sys.argv[2]
            description = sys.argv[3] if len(sys.argv) > 3 else None
            priority = int(sys.argv[4]) if len(sys.argv) > 4 else 5
            create_task(title, description, priority)
        
        elif command == 'list-tasks':
            list_tasks()
        
        elif command == 'export-context':
            types = sys.argv[2] if len(sys.argv) > 2 else 'tasks,memories'
            max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
            export_context(types, max_tokens)
        
        elif command == 'help':
            show_help()
        
        else:
            print(f"[MCP] Unknown command: {command}")
            show_help()
    
    except Exception as e:
        print(f"[MCP] Error: {e}")

if __name__ == '__main__':
    main() 