#!/usr/bin/env python3
"""
MCP System Feature Demonstration
Shows key capabilities of the MCP agentic workflow system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp.server import MCPServer
from mcp.memory import MemoryManager
from mcp.workflow import WorkflowManager
from mcp.task_manager import TaskManager
from mcp.context_manager import ContextManager
from mcp.rag_system import RAGSystem

def demo_memory_system():
    """Demonstrate advanced memory capabilities"""
    print("🧠 MEMORY SYSTEM DEMO")
    print("=" * 50)
    
    memory_manager = MemoryManager()
    
    # Add different types of memories
    memories = [
        ("Python is a high-level programming language", "knowledge", 0.8),
        ("Use virtual environments for Python projects", "best_practice", 0.9),
        ("The project deadline is next Friday", "reminder", 0.7),
        ("User prefers concise explanations", "preference", 0.6),
        ("API endpoint /users returns user data", "technical", 0.8)
    ]
    
    for text, mem_type, priority in memories:
        mem_id = memory_manager.add_memory(text=text, memory_type=mem_type, priority=priority)
        print(f"✓ Added {mem_type} memory: {text[:50]}...")
    
    # Search memories
    results = memory_manager.search_memories("Python programming", limit=3)
    print(f"\n🔍 Search results for 'Python programming':")
    for result in results:
        print(f"  - {result['text']} (type: {result['memory_type']})")
    
    print()

def demo_workflow_system():
    """Demonstrate workflow orchestration"""
    print("🔄 WORKFLOW SYSTEM DEMO")
    print("=" * 50)
    
    workflow_manager = WorkflowManager()
    workflow_id = workflow_manager.create_workflow("demo_project", "/tmp/demo")
    
    # Create a custom workflow step
    from mcp.workflow import WorkflowStep
    custom_step = WorkflowStep(name="custom_analysis", description="Custom data analysis step")
    workflow_manager.register_step("custom_analysis", custom_step)
    
    # Start and complete steps
    workflow_manager.start_step("init")
    workflow_manager.complete_step("init")
    
    workflow_manager.start_step("custom_analysis")
    workflow_manager.add_step_feedback("custom_analysis", "Analysis completed successfully", impact=2)
    workflow_manager.complete_step("custom_analysis")
    
    # Show workflow status
    status = workflow_manager.get_workflow_status()
    print(f"✓ Workflow created: {workflow_id}")
    print(f"✓ Completed steps: {status.get('completed_steps', [])}")
    print(f"✓ Current step: {status.get('current_step', 'None')}")
    print()

def demo_task_system():
    """Demonstrate task management"""
    print("📋 TASK SYSTEM DEMO")
    print("=" * 50)
    
    task_manager = TaskManager()
    
    # Create a task hierarchy
    parent_task = task_manager.create_task(
        title="Build Web Application",
        description="Create a full-stack web application",
        priority=8,
        estimated_hours=40.0
    )
    
    subtasks = [
        ("Design Database Schema", "Create ERD and implement database", 7),
        ("Build Backend API", "Implement REST API endpoints", 8),
        ("Create Frontend UI", "Build responsive user interface", 6),
        ("Write Tests", "Unit and integration tests", 5)
    ]
    
    for title, desc, priority in subtasks:
        task_id = task_manager.create_task(
            title=title,
            description=desc,
            priority=priority,
            parent_id=parent_task
        )
        print(f"✓ Created subtask: {title}")
    
    # Update task progress
    task_manager.update_task_progress(parent_task, 25.0, "Database design completed")
    
    # Show task tree
    tree = task_manager.get_task_tree(root_task_id=parent_task)
    print(f"\n📊 Task tree for project:")
    for task in tree.get('all_tasks', {}).values():
        indent = "  " * (task.get('depth', 0))
        status = task.get('status', 'unknown')
        print(f"{indent}- {task['title']} ({status})")
    
    print()

def demo_context_system():
    """Demonstrate context management"""
    print("🎯 CONTEXT SYSTEM DEMO")
    print("=" * 50)
    
    context_manager = ContextManager()
    
    # Export optimized context
    context = context_manager.export_context(max_tokens=500)
    
    print("📄 Current Project Context:")
    print(f"  Tasks: {len(context.get('tasks', []))}")
    print(f"  Memories: {len(context.get('memories', []))}")
    print(f"  Total tokens: {context.get('total_tokens', 0)}")
    
    # Show context summary
    if 'context' in context:
        print(f"\n📝 Context Summary:")
        print(context['context'][:200] + "...")
    
    print()

def demo_rag_system():
    """Demonstrate RAG capabilities"""
    print("🔍 RAG SYSTEM DEMO")
    print("=" * 50)
    
    rag_system = RAGSystem()
    
    # Add some content to RAG
    content_items = [
        ("Python best practices include using virtual environments", "code", 1),
        ("REST APIs should follow HTTP status code conventions", "documentation", 2),
        ("Database normalization reduces data redundancy", "knowledge", 3)
    ]
    
    for content, content_type, source_id in content_items:
        rag_system.add_chunk(content, content_type, source_id)
        print(f"✓ Added {content_type} content: {content[:50]}...")
    
    # Query RAG system
    from mcp.rag_system import RAGQuery
    query = RAGQuery(
        query="How to structure a Python project?",
        context={},
        max_tokens=200,
        chunk_types=['code', 'documentation', 'knowledge'],
        project_id=None,
        user_id=None
    )
    results = rag_system.retrieve_context(query)
    
    print(f"\n🔍 RAG Query: '{query.query}'")
    print(f"📄 Results: {results.summary[:100]}...")
    print(f"📊 Confidence: {results.confidence:.2f}")
    print(f"📝 Sources: {len(results.sources)} chunks")
    print()

def demo_experimental_features():
    """Demonstrate experimental AI features"""
    print("🤖 EXPERIMENTAL FEATURES DEMO")
    print("=" * 50)
    
    try:
        from mcp.experimental_lobes import AlignmentEngine, PatternRecognitionEngine
        
        # Test alignment engine
        alignment_engine = AlignmentEngine(":memory:")
        original_text = "This is a very long and verbose explanation that could be more concise."
        aligned = alignment_engine.align(original_text, {"preference": "concise", "max_words": 10})
        print(f"✓ Text alignment: {aligned}")
        
        # Test pattern recognition
        pattern_engine = PatternRecognitionEngine(":memory:")
        data = ["pattern1", "pattern2", "pattern1", "pattern3"]
        patterns = pattern_engine.recognize_patterns(data)
        print(f"✓ Recognized {len(patterns)} patterns")
        
        print("✅ Experimental features working!")
        
    except Exception as e:
        print(f"⚠️  Experimental features: {e}")
    
    print()

def main():
    """Run all demonstrations"""
    print("🚀 MCP SYSTEM FEATURE DEMONSTRATION")
    print("=" * 60)
    print()
    
    demo_memory_system()
    demo_workflow_system()
    demo_task_system()
    demo_context_system()
    demo_rag_system()
    demo_experimental_features()
    
    print("🎉 DEMONSTRATION COMPLETE!")
    print("\nThe MCP system provides:")
    print("  • Advanced memory management with vector search")
    print("  • Sophisticated workflow orchestration")
    print("  • Hierarchical task management")
    print("  • Intelligent context optimization")
    print("  • RAG-powered information retrieval")
    print("  • Experimental AI features")
    print("\nReady for agentic development workflows! 🚀")

if __name__ == "__main__":
    main() 