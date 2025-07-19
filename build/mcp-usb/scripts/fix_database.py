#!/usr/bin/env python3
"""
Database Fix Script for MCP Agentic Workflow Accelerator
Fixes database schema issues and migrates existing data.
"""

import sqlite3
import os
import sys
from pathlib import Path

def fix_database():
    """Fix database schema issues."""
    print("🔧 Fixing database schema...")
    
    # Find the database file
    db_paths = [
        "data/unified_memory.db",
        "data/workflow.db",
        "data/tasks.db"
    ]
    
    for db_path in db_paths:
        if os.path.exists(db_path):
            print(f"  Fixing database: {db_path}")
            fix_single_database(db_path)
    
    print("✅ Database fixes completed!")

def fix_single_database(db_path):
    """Fix a single database file."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if tasks table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        if cursor.fetchone():
            # Check if is_meta column exists
            cursor.execute("PRAGMA table_info(tasks)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'is_meta' not in columns:
                print(f"    Adding is_meta column to tasks table...")
                cursor.execute("ALTER TABLE tasks ADD COLUMN is_meta BOOLEAN DEFAULT FALSE")
            
            if 'meta_type' not in columns:
                print(f"    Adding meta_type column to tasks table...")
                cursor.execute("ALTER TABLE tasks ADD COLUMN meta_type TEXT")
            
            # Check other tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_tags'")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(task_tags)")
                tag_columns = [column[1] for column in cursor.fetchall()]
                
                if 'task_id' not in tag_columns:
                    print(f"    Adding task_id column to task_tags table...")
                    cursor.execute("ALTER TABLE task_tags ADD COLUMN task_id INTEGER")
                
                if 'parent_tag_id' not in tag_columns:
                    print(f"    Adding parent_tag_id column to task_tags table...")
                    cursor.execute("ALTER TABLE task_tags ADD COLUMN parent_tag_id INTEGER")
                
                if 'color' not in tag_columns:
                    print(f"    Adding color column to task_tags table...")
                    cursor.execute("ALTER TABLE task_tags ADD COLUMN color TEXT")
                
                if 'icon' not in tag_columns:
                    print(f"    Adding icon column to task_tags table...")
                    cursor.execute("ALTER TABLE task_tags ADD COLUMN icon TEXT")
                
                if 'description' not in tag_columns:
                    print(f"    Adding description column to task_tags table...")
                    cursor.execute("ALTER TABLE task_tags ADD COLUMN description TEXT")
        
        # Create missing tables if they don't exist
        create_missing_tables(cursor)
        
        conn.commit()
        print(f"    Database {db_path} fixed successfully!")
        
    except Exception as e:
        print(f"    Error fixing database {db_path}: {e}")
        conn.rollback()
    finally:
        conn.close()

def create_missing_tables(cursor):
    """Create missing tables with correct schema."""
    
    # Tasks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            parent_id INTEGER,
            estimated_hours REAL DEFAULT 0.0,
            actual_hours REAL DEFAULT 0.0,
            accuracy_critical BOOLEAN DEFAULT FALSE,
            is_meta BOOLEAN DEFAULT FALSE,
            meta_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            due_date TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES tasks (id)
        )
    """)
    
    # Task dependencies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_dependencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            depends_on_task_id INTEGER,
            dependency_type TEXT DEFAULT 'blocks',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks (id),
            FOREIGN KEY (depends_on_task_id) REFERENCES tasks (id)
        )
    """)
    
    # Task notes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            note_text TEXT NOT NULL,
            line_number INTEGER,
            file_path TEXT,
            note_type TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks (id)
        )
    """)
    
    # Task progress table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            progress_percentage REAL DEFAULT 0.0,
            current_step TEXT,
            partial_completion_notes TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks (id)
        )
    """)
    
    # Task feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            feedback_text TEXT NOT NULL,
            lesson_learned TEXT,
            principle TEXT,
            impact_score INTEGER DEFAULT 0,
            feedback_type TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks (id)
        )
    """)
    
    # Task tags table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            tag_name TEXT NOT NULL,
            tag_type TEXT DEFAULT 'general',
            parent_tag_id INTEGER,
            color TEXT,
            icon TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks (id),
            FOREIGN KEY (parent_tag_id) REFERENCES task_tags (id)
        )
    """)
    
    # Memory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            memory_type TEXT DEFAULT 'general',
            priority REAL DEFAULT 0.5,
            context TEXT,
            tags TEXT,
            vector_data BLOB,
            encoder_type TEXT DEFAULT 'tfidf',
            memory_order INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Context packs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_packs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pack_name TEXT NOT NULL,
            context_type TEXT DEFAULT 'general',
            content TEXT NOT NULL,
            max_tokens INTEGER DEFAULT 1000,
            project_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Context templates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name TEXT NOT NULL,
            description TEXT,
            context_types TEXT,
            max_tokens INTEGER DEFAULT 1000,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

def main():
    """Main function."""
    print("🔧 MCP Agentic Workflow Accelerator - Database Fix Tool")
    print("=" * 60)
    
    try:
        fix_database()
        print("\n✅ All databases have been fixed!")
        print("You can now run the MCP server without issues.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 