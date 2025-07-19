from typing import Final, Literal, Dict, Any
from pathlib import Path
import click
import os
import sys
import asyncio
import json
import datetime
import time

from .memory import MemoryManager
from .workflow import WorkflowManager
from .project_manager import ProjectManager
from .task_manager import TaskManager
from .unified_memory import UnifiedMemoryManager
try:
    from .experimental_lobes import SimulatedReality
except ImportError:
    SimulatedReality = None  # Fallback for environments where not present
from .performance_monitor import ObjectivePerformanceMonitor
from .rag_system import RAGSystem
from .reminder_engine import EnhancedReminderEngine
from .advanced_memory import TFIDFEncoder, RaBitQEncoder
from .server import MCPServer
from .automatic_update_system import AutomaticUpdateSystem, UpdateStatus

FEATURE_FLAGS: Final[Dict[str, bool]] = {
    'experimental_lobes': False,
    'distributed_mode': False,
    'visual_workflow_explorer': False,
    'self_improving_server': False,
    'simulated_reality': False,
}

REALITY_PATH: Final[str] = 'data/simulated_reality.json'

# Initialize MCPServer to ensure logging is set up for all CLI commands
mcp_server = MCPServer()
logger = mcp_server.logger
logger.info('[MCP CLI] Logger initialized and ready (test log to trigger mcp.log creation).')

def find_project_root(start_path=None):
    """Walk up directories to find the project root (where config.cfg or .mcp exists)."""
    if start_path is None:
        start_path = os.getcwd()
    current = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current, 'config.cfg')) or os.path.exists(os.path.join(current, '.mcp')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

# Ensure src/ is in sys.path for relative imports
project_root = find_project_root()
if project_root:
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

@click.group()
@click.option('--project-path', default=None, help='Explicit path to the project directory (overrides auto-detection)')
@click.pass_context
def cli(ctx, project_path):
    """MCP Agentic Workflow Accelerator CLI - Advanced Edition"""
    ctx.ensure_object(dict)
    ctx.obj['PROJECT_PATH'] = project_path

def get_project_manager(ctx=None) -> ProjectManager:
    project_path = None
    if ctx and hasattr(ctx, 'obj') and ctx.obj.get('PROJECT_PATH'):
        project_path = ctx.obj['PROJECT_PATH']
    elif click.get_current_context().obj and 'PROJECT_PATH' in click.get_current_context().obj:
        project_path = click.get_current_context().obj['PROJECT_PATH']
    if project_path:
        return ProjectManager(project_path)
    current_dir = os.getcwd()
    manager = ProjectManager(current_dir)
    config_file = manager.find_project_config()
    if config_file:
        project_dir = config_file.parent
        return ProjectManager(str(project_dir))
    return manager

def get_workflow_manager(ctx=None):
    manager = get_project_manager(ctx)
    project_info = manager.get_project_info()
    if not project_info or 'path' not in project_info:
        return WorkflowManager()  # fallback to default/global
    db_path = os.path.join(project_info['path'], 'data', 'workflow.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return WorkflowManager(db_path=db_path)

# Project Management Commands
@cli.command()
@click.option('--name', prompt='Project name', help='Name of the new project')
@click.option('--path', default=None, help='Path for the project (defaults to current directory)')
def init_project(name, path):
    """Initialize a new project with MCP workflow."""
    manager = get_project_manager()
    result = manager.init_project(name, path)
    
    click.echo(f"[MCP] Project '{name}' initialized successfully!")
    click.echo(f"[MCP] Project path: {result['project_path']}")
    click.echo(f"[MCP] Configuration file: {result['config_file']}")
    click.echo("\n[MCP] Next steps:")
    for step in result['next_steps']:
        click.echo(f"  {step}")
    
    click.echo(f"\n[MCP] Use 'python mcp.py show-questions' to see alignment questions")
    click.echo(f"[MCP] Use 'python mcp.py answer-question' to provide answers")

@cli.command()
@click.pass_context
def show_questions(ctx):
    """Show all configuration questions for the current project."""
    manager = get_project_manager(ctx)
    questions = manager.get_questions()
    
    if not questions:
        click.echo("[MCP] No project configuration found. Run 'init-project' first.")
        click.echo("[MCP] Make sure you're in a project directory or run from the project root.")
        return
    
    click.echo("[MCP] Configuration Questions:")
    click.echo("=" * 50)
    
    for section, section_questions in questions.items():
        if section == 'PROJECT':
            continue  # Skip project metadata section
        click.echo(f"\n[{section.upper()}]")
        for key, value in section_questions.items():
            status = "✓" if value.strip() else "○"
            click.echo(f"  {status} {key}: {value or '(not answered)'}")

@cli.command()
@click.option('--section', prompt='Section', help='Configuration section')
@click.option('--key', prompt='Question key', help='Question key')
@click.option('--answer', prompt='Answer', help='Your answer')
def answer_question(section, key, answer):
    """Answer a configuration question."""
    manager = get_project_manager()
    success = manager.answer_question(section, key, answer)
    
    if success:
        click.echo(f"[MCP] Answer saved for {section}.{key}")
    else:
        click.echo(f"[MCP] Failed to save answer. Check section and key names.")
        click.echo("[MCP] Make sure you're in a project directory.")

@cli.command()
def project_status():
    """Show current project status and completion."""
    manager = get_project_manager()
    info = manager.get_project_info()
    summary = manager.generate_project_summary()
    validation = manager.validate_configuration()
    
    click.echo("[MCP] Project Status:")
    click.echo("=" * 30)
    click.echo(summary)
    
    if validation['errors']:
        click.echo("\n[MCP] Errors:")
        for error in validation['errors']:
            click.echo(f"  ❌ {error}")
    
    if validation['warnings']:
        click.echo("\n[MCP] Warnings:")
        for warning in validation['warnings']:
            click.echo(f"  ⚠️  {warning}")

# Workflow Commands
@cli.command()
def complete_init():
    """Complete the initialization phase and allow progression to research."""
    manager = get_project_manager()
    project_info = manager.get_project_info()
    if not project_info:
        click.echo("[MCP] No project found. Run 'init-project' first.")
        return
    
    workflow = get_workflow_manager()
    # Create workflow if it doesn't exist
    workflow_id = workflow.create_workflow(project_info['name'], project_info['path'])
    success = workflow.complete_init_step()
    if success:
        click.echo("[MCP] Initialization phase completed!")
        click.echo("[MCP] You can now start the research phase with 'start-research'")
    else:
        click.echo("[MCP] Initialization already completed or not in progress.")

@cli.command()
def start_research():
    """Start the research phase of the project."""
    manager = get_project_manager()
    project_info = manager.get_project_info()
    if not project_info:
        click.echo("[MCP] No project found. Run 'init-project' first.")
        return
    workflow = get_workflow_manager()
    if workflow.can_start_research():
        success = workflow.start_step('research')
        if success:
            click.echo("[MCP] Research phase started!")
            click.echo("[MCP] Use 'add-research-topic' to add research areas")
            click.echo("[MCP] Use 'add-finding' to record research findings")
        else:
            click.echo("[MCP] Failed to start research phase.")
    else:
        click.echo("[MCP] Cannot start research phase. Complete initialization first.")
        click.echo("[MCP] Use 'complete-init' to mark initialization as complete.")

@cli.command()
@click.option('--topic', prompt='Research topic', help='Topic to research')
@click.option('--priority', default=0.5, help='Priority (0.0-1.0)')
def add_research_topic(topic, priority):
    """Add a research topic."""
    workflow = get_workflow_manager()
    step = workflow.steps.get('research')
    if step and step.status.value == 'in_progress':
        step.add_research_topic(topic, priority)
        click.echo(f"[MCP] Research topic added: {topic}")
    else:
        click.echo("[MCP] Research phase not active. Start research first.")

@cli.command()
@click.option('--topic', prompt='Research topic', help='Topic this finding relates to')
@click.option('--finding', prompt='Finding', help='Research finding')
@click.option('--source', default=None, help='Source of the finding')
def add_finding(topic, finding, source):
    """Add a research finding."""
    workflow = get_workflow_manager()
    step = workflow.steps.get('research')
    if step and step.status.value == 'in_progress':
        step.add_finding(topic, finding, source)
        click.echo(f"[MCP] Finding added for topic: {topic}")
    else:
        click.echo("[MCP] Research phase not active. Start research first.")

@cli.command()
def start_planning():
    """Start the planning phase of the project."""
    workflow = get_workflow_manager()
    success = workflow.start_step('planning')
    if success:
        click.echo("[MCP] Planning phase started!")
        click.echo("[MCP] Use 'add-task' to add project tasks")
        click.echo("[MCP] Use 'set-architecture' to define system architecture")
    else:
        click.echo("[MCP] Cannot start planning phase. Complete research first.")

@cli.command()
@click.option('--description', prompt='Task description', help='Description of the task')
@click.option('--priority', default=0, help='Priority (0-10)')
def add_task(description, priority):
    """Add a project task."""
    workflow = get_workflow_manager()
    step = workflow.steps.get('planning')
    if step and step.status.value == 'in_progress':
        step.add_task(description, priority)
        click.echo(f"[MCP] Task added: {description}")
    else:
        click.echo("[MCP] Planning phase not active. Start planning first.")

@cli.command()
def start_development():
    """Start the development phase of the project."""
    workflow = get_workflow_manager()
    success = workflow.start_step('development')
    if success:
        click.echo("[MCP] Development phase started!")
        click.echo("[MCP] Use 'add-feature' to add features")
        click.echo("[MCP] Use 'add-bug' to track bugs")
    else:
        click.echo("[MCP] Cannot start development phase. Complete planning first.")

@cli.command()
def start_testing():
    """Start the testing phase of the project."""
    workflow = get_workflow_manager()
    success = workflow.start_step('testing')
    if success:
        click.echo("[MCP] Testing phase started!")
        click.echo("[MCP] Use 'add-test-case' to add test cases")
        click.echo("[MCP] Use 'add-test-result' to record test results")
    else:
        click.echo("[MCP] Cannot start testing phase. Complete development first.")

@cli.command()
def start_deployment():
    """Start the deployment phase of the project."""
    workflow = get_workflow_manager()
    success = workflow.start_step('deployment')
    if success:
        click.echo("[MCP] Deployment phase started!")
        click.echo("[MCP] Use 'add-environment' to add deployment environments")
        click.echo("[MCP] Use 'add-deployment' to record deployments")
    else:
        click.echo("[MCP] Cannot start deployment phase. Complete testing first.")

@cli.command()
def workflow_status():
    """Show current workflow status."""
    workflow = get_workflow_manager()
    status = workflow.get_workflow_status()
    click.echo("[MCP] Workflow Status:")
    click.echo("=" * 30)
    click.echo(f"Progress: {status['progress']:.1%}")
    click.echo(f"Current Step: {status['current_step'] or 'None'}")
    click.echo(f"Completed Steps: {', '.join(status['completed_steps']) if status['completed_steps'] else 'None'}")
    if status['current_step']:
        next_steps = workflow.get_next_steps(status['current_step'])
        click.echo(f"Possible Next Steps: {', '.join(next_steps) if next_steps else '(none)'}")
    click.echo("\n[MCP] Step Details:")
    for step_name, step_info in status['steps'].items():
        status_icon = "✅" if step_info['status'] == 'completed' else "⏳" if step_info['status'] == 'in_progress' else "○"
        click.echo(f"  {status_icon} {step_name}: {step_info['status']}")

# Memory Management Commands
@cli.command()
@click.option('--text', prompt='Memory text', help='Text content of the memory')
@click.option('--type', default='general', help='Memory type')
@click.option('--priority', default=0.5, help='Priority (0.0-1.0)')
@click.option('--context', default=None, help='Context for the memory')
@click.option('--tags', default=None, help='Comma-separated tags')
@click.option('--encoder', default='tfidf', type=click.Choice(['tfidf', 'rabitq']), help='Vector encoder for advanced memory (tfidf, rabitq)')
@click.option('--memory-order', default=1, type=int, help='Memory order: 1=first, 2=second, 3=third (see idea.txt for details)')
def add_memory(text: str, type: str, priority: float, context: str, tags: str, encoder: Literal['tfidf', 'rabitq'], memory_order: int) -> None:
    """Add a new memory (supports advanced vector encoding and memory order)."""
    if encoder == 'rabitq':
        encoder_instance = RaBitQEncoder()
    else:
        encoder_instance = TFIDFEncoder()
    manager = UnifiedMemoryManager(encoder=encoder_instance)
    tags_list = [t.strip() for t in tags.split(',')] if tags else []
    result = manager.add_memory(text, type, priority, context, tags_list, use_advanced=True, create_reminder=False, memory_order=memory_order)
    click.echo(f"[MCP] Memory added. Basic ID: {result['basic_memory_id']}, Advanced ID: {result['advanced_memory_id']}")
    click.echo(f"[MCP] Encoder used: {encoder_instance.name()}")
    click.echo(f"[MCP] Memory order: {memory_order}")
    if tags_list:
        click.echo(f"[MCP] Tags: {', '.join(tags_list)}")

@cli.command()
@click.option('--query', prompt='Search query', help='Query to search for')
@click.option('--limit', default=10, help='Maximum number of results')
@click.option('--type', default=None, help='Filter by memory type')
@click.option('--encoder', default='tfidf', type=click.Choice(['tfidf', 'rabitq']), help='Vector encoder for advanced memory search (tfidf, rabitq)')
@click.option('--memory-order', default=None, type=int, help='Filter by memory order (1=first, 2=second, 3=third)')
def search_memories(query, limit, type, encoder, memory_order):
    """Search memories by text content and memory order (supports advanced vector encoding)."""
    if encoder == 'rabitq':
        encoder_instance = RaBitQEncoder()
    else:
        encoder_instance = TFIDFEncoder()
    manager = UnifiedMemoryManager(encoder=encoder_instance)
    results = manager.search_memories(query, limit=limit, memory_type=type, memory_order=memory_order)
    if not results:
        click.echo("[MCP] No memories found matching your query.")
        return
    click.echo(f"[MCP] Found {len(results)} memories (Encoder: {encoder_instance.name()}):")
    click.echo("=" * 50)
    for i, memory in enumerate(results, 1):
        click.echo(f"{i}. Memory {memory['id']} (Order: {memory.get('memory_order', '?')}, Type: {memory.get('memory_type', 'unknown')})")
        click.echo(f"   Text: {memory.get('text', '')[:100]}...")
        click.echo(f"   Priority: {memory.get('priority', 0)}")
        click.echo(f"   Created: {memory.get('created_at', 'N/A')}")
        if memory.get('tags'):
            click.echo(f"   Tags: {', '.join(memory['tags'])}")
        click.echo()

@cli.command()
@click.option('--memory-id', prompt='Memory ID', type=int, help='ID of the memory')
def get_memory(memory_id):
    """Get a specific memory by ID."""
    manager = MemoryManager()
    memory = manager.get_memory(memory_id)
    
    if not memory:
        click.echo(f"[MCP] Memory {memory_id} not found.")
        return
    
    click.echo(f"[MCP] Memory {memory_id}:")
    click.echo("=" * 30)
    click.echo(f"Text: {memory['text']}")
    click.echo(f"Type: {memory['memory_type']}")
    click.echo(f"Priority: {memory['priority']}")
    click.echo(f"Context: {memory['context'] or 'None'}")
    if memory['tags']:
        click.echo(f"Tags: {', '.join(memory['tags'])}")
    click.echo(f"Created: {memory['created_at']}")
    click.echo(f"Updated: {memory['updated_at']}")

@cli.command()
@click.option('--encoder', default='tfidf', type=click.Choice(['tfidf', 'rabitq']), help='Vector encoder to report metrics for (tfidf, rabitq)')
def memory_metrics(encoder):
    """Show vector encoder compression and search accuracy metrics."""
    if encoder == 'rabitq':
        encoder_instance = RaBitQEncoder()
    else:
        encoder_instance = TFIDFEncoder()
    from .unified_memory import UnifiedMemoryManager
    manager = UnifiedMemoryManager(encoder=encoder_instance)
    adv_manager = manager.advanced_memory
    metrics = adv_manager.get_metrics()
    click.echo(f"[MCP] Encoder: {encoder_instance.name()}")
    click.echo(f"  Average Compression Ratio: {metrics['average_compression_ratio']:.2f}")
    click.echo(f"  Average Search Similarity: {metrics['average_search_similarity']:.3f}")
    click.echo(f"  Max Search Similarity: {metrics['max_search_similarity']:.3f}")
    click.echo(f"  Memories Added: {metrics['num_memories']}")
    click.echo(f"  Searches Performed: {metrics['num_searches']}")

# Task Management Commands
@cli.command()
@click.option('--title', prompt='Task title', help='Title of the task')
@click.option('--description', default=None, help='Description of the task')
@click.option('--priority', default=5, type=int, help='Priority (1-10)')
@click.option('--parent-id', default=None, type=int, help='Parent task ID')
@click.option('--estimated-hours', default=0.0, type=float, help='Estimated hours')
@click.option('--accuracy-critical', is_flag=True, help='Mark as accuracy-critical')
@click.option('--due-date', default=None, help='Due date (YYYY-MM-DD)')
@click.option('--tags', default=None, help='Comma-separated tags')
def create_task(title: str, description: str, priority: int, parent_id: int, estimated_hours: float, accuracy_critical: bool, due_date: str, tags: str) -> None:
    """Create a new task with full metadata."""
    from datetime import datetime
    
    task_manager = TaskManager()
    
    # Parse due date
    parsed_due_date = None
    if due_date:
        try:
            parsed_due_date = datetime.strptime(due_date, '%Y-%m-%d')
        except ValueError:
            click.echo("[MCP] Invalid due date format. Use YYYY-MM-DD")
            return
    
    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
    
    task_id = task_manager.create_task(
        title=title,
        description=description,
        priority=priority,
        parent_id=parent_id,
        estimated_hours=estimated_hours,
        accuracy_critical=accuracy_critical,
        due_date=parsed_due_date,
        tags=tag_list
    )
    
    click.echo(f"[MCP] Task created with ID {task_id}")
    click.echo(f"[MCP] Title: {title}")
    click.echo(f"[MCP] Priority: {priority}")
    if accuracy_critical:
        click.echo("[MCP] ⚠️  Marked as accuracy-critical")

@cli.command()
@click.option('--status', default=None, help='Filter by status')
@click.option('--priority-min', default=None, type=int, help='Minimum priority')
@click.option('--include-completed', is_flag=True, help='Include completed tasks')
@click.option('--tree', is_flag=True, help='Show as tree structure')
def list_tasks(status, priority_min, include_completed, tree):
    """List tasks with advanced filtering."""
    task_manager = TaskManager()
    
    if tree:
        task_tree = task_manager.get_task_tree(include_completed=include_completed)
        _display_task_tree(task_tree['root_tasks'])
    else:
        tasks = task_manager.get_tasks(status)
        
        # Apply priority filter
        if priority_min:
            tasks = [t for t in tasks if t['priority'] >= priority_min]
        
        if not tasks:
            click.echo("[MCP] No tasks found.")
            return
        
        click.echo(f"[MCP] Found {len(tasks)} tasks:")
        click.echo("=" * 60)
        
        for i, task in enumerate(tasks, 1):
            status_icon = _get_status_icon(task['status'])
            critical_marker = " 🔴" if task.get('accuracy_critical') else ""
            click.echo(f"{i}. {status_icon} Task {task['id']}{critical_marker}")
            click.echo(f"   Title: {task['title']}")
            click.echo(f"   Status: {task['status']}")
            click.echo(f"   Priority: {task['priority']}")
            click.echo(f"   Created: {task['created_at']}")
            if task['completed_at']:
                click.echo(f"   Completed: {task['completed_at']}")
            click.echo()

@cli.command()
@click.option('--task-id', prompt='Task ID', type=int, help='ID of the task')
@click.option('--progress', default=None, type=float, help='Progress percentage (0-100)')
@click.option('--current-step', default=None, help='Current step description')
@click.option('--notes', default=None, help='Partial completion notes')
def update_task_progress(task_id, progress, current_step, notes):
    """Update task progress with partial completion support."""
    task_manager = TaskManager()
    
    if progress is None:
        progress = click.prompt('Progress percentage (0-100)', type=float)
    
    success = task_manager.update_task_progress(
        task_id, progress, current_step, notes
    )
    
    if success:
        click.echo(f"[MCP] Task {task_id} progress updated to {progress}%")
        if current_step:
            click.echo(f"[MCP] Current step: {current_step}")
    else:
        click.echo(f"[MCP] Failed to update task {task_id}")

@cli.command()
@click.option('--task-id', prompt='Task ID', type=int, help='ID of the task')
@click.option('--note', prompt='Note text', help='Note content')
@click.option('--line-number', default=None, type=int, help='Line number reference')
@click.option('--file-path', default=None, help='File path reference')
@click.option('--note-type', default='general', help='Type of note')
def add_task_note(task_id, note, line_number, file_path, note_type):
    """Add a note to a task with line number and file path support."""
    task_manager = TaskManager()
    
    note_id = task_manager.add_task_note(
        task_id, note, line_number, file_path, note_type
    )
    
    click.echo(f"[MCP] Note added with ID {note_id}")
    click.echo(f"[MCP] Task: {task_id}")
    if line_number:
        click.echo(f"[MCP] Line: {line_number}")
    if file_path:
        click.echo(f"[MCP] File: {file_path}")

@cli.command()
@click.option('--task-id', prompt='Task ID', type=int, help='ID of the task')
@click.option('--depends-on', prompt='Dependency task ID', type=int, help='Task this depends on')
@click.option('--dependency-type', default='blocks', help='Type of dependency')
def add_task_dependency(task_id, depends_on, dependency_type):
    """Add a dependency between tasks."""
    task_manager = TaskManager()
    
    dependency_id = task_manager.add_task_dependency(
        task_id, depends_on, dependency_type
    )
    
    click.echo(f"[MCP] Dependency added with ID {dependency_id}")
    click.echo(f"[MCP] Task {task_id} now depends on task {depends_on}")

@cli.command()
def show_blocked_tasks():
    """Show tasks that are blocked by incomplete dependencies."""
    task_manager = TaskManager()
    blocked_tasks = task_manager.get_blocked_tasks()
    
    if not blocked_tasks:
        click.echo("[MCP] No blocked tasks found.")
        return
    
    click.echo(f"[MCP] Found {len(blocked_tasks)} blocked tasks:")
    click.echo("=" * 50)
    
    for i, task in enumerate(blocked_tasks, 1):
        click.echo(f"{i}. Task {task['id']} (P{task['priority']})")
        click.echo(f"   Title: {task['title']}")
        click.echo(f"   Status: {task['status']}")
        click.echo(f"   Blocked by: {', '.join(task['blocking_tasks'])}")
        click.echo()

@cli.command()
def show_critical_tasks():
    """Show tasks marked as accuracy-critical."""
    task_manager = TaskManager()
    critical_tasks = task_manager.get_accuracy_critical_tasks()
    
    if not critical_tasks:
        click.echo("[MCP] No accuracy-critical tasks found.")
        return
    
    click.echo(f"[MCP] Found {len(critical_tasks)} accuracy-critical tasks:")
    click.echo("=" * 50)
    
    for i, task in enumerate(critical_tasks, 1):
        click.echo(f"{i}. 🔴 Task {task['id']} (P{task['priority']})")
        click.echo(f"   Title: {task['title']}")
        click.echo(f"   Status: {task['status']}")
        click.echo(f"   Estimated: {task['estimated_hours']} hours")
        click.echo()

# Context Management Commands
@cli.command()
@click.option('--types', default='tasks,memories,progress', help='Context types (comma-separated)')
@click.option('--max-tokens', default=1000, type=int, help='Maximum tokens')
@click.option('--save', is_flag=True, help='Save context pack for later')
@click.option('--format', default='text', help='Output format (text/json)')
@click.option('--use-rag', is_flag=True, help='Use RAG system for intelligent retrieval')
@click.option('--query', default=None, help='Query for RAG retrieval')
def export_context(types, max_tokens, save, format, use_rag, query):
    """Export minimal, relevant context for LLM consumption with RAG enhancement."""
    from .context_manager import ContextManager
    
    context_manager = ContextManager()
    context_types = [t.strip() for t in types.split(',')]
    
    if use_rag and query:
        # Use RAG system
        result = context_manager.export_context(
            context_types=context_types,
            max_tokens=max_tokens,
            format=format,
            use_rag=True,
            query=query
        )
        
        click.echo(f"[MCP] RAG Context Export:")
        click.echo(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
        click.echo(f"📝 Tokens: {result.get('total_tokens', 0)}")
        click.echo(f"📊 Sources: {len(result.get('sources', []))}")
        click.echo()
        
        if format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result.get('context', 'No context generated'))
    else:
        # Use traditional context generation
        result = context_manager.export_context(
            context_types=context_types,
            max_tokens=max_tokens,
            format=format,
            use_rag=False
        )
        
        click.echo(f"[MCP] Traditional Context Export:")
        click.echo(f"📝 Tokens: {result.get('total_tokens', 0)}")
        click.echo()
        
        if format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result.get('context', 'No context generated'))
    
    if save:
        pack_id = context_manager.save_context_pack(
            name=f"Context Export {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            context_data=result,
            description=f"Auto-saved context export with {len(context_types)} types"
        )
        click.echo(f"[MCP] Context pack saved with ID: {pack_id}")

@cli.command()
@click.option('--query', prompt='RAG query', help='Query for intelligent context retrieval')
@click.option('--types', default='memory,task,code,document,feedback', help='Chunk types to search')
@click.option('--max-tokens', default=1000, type=int, help='Maximum tokens')
@click.option('--project-id', default=None, help='Project ID to scope search')
@click.option('--format', default='text', help='Output format (text/json)')
def rag_query(query, types, max_tokens, project_id, format):
    """Query the RAG system for intelligent context retrieval."""
    from .rag_system import RAGSystem, RAGQuery
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    rag_system = RAGSystem()
    chunk_types = [t.strip() for t in types.split(',')]
    
    # Create RAG query
    rag_query = RAGQuery(
        query=query,
        context={'project_id': project_id} if project_id else {},
        max_tokens=max_tokens,
        chunk_types=chunk_types,
        project_id=project_id,
        user_id=None
    )
    
    # Retrieve context
    result = rag_system.retrieve_context(rag_query)
    
    click.echo(f"[MCP] RAG Query Results:")
    click.echo(f"🎯 Confidence: {result.confidence:.2f}")
    click.echo(f"📝 Tokens: {result.total_tokens}")
    click.echo(f"📊 Chunks: {len(result.chunks)}")
    click.echo(f"📋 Summary: {result.summary}")
    click.echo()
    
    if format == 'json':
        # Convert to JSON-serializable format
        json_result = {
            'chunks': [
                {
                    'id': chunk.id,
                    'content': chunk.content,
                    'source_type': chunk.source_type,
                    'source_id': chunk.source_id,
                    'metadata': chunk.metadata
                }
                for chunk in result.chunks
            ],
            'total_tokens': result.total_tokens,
            'relevance_scores': result.relevance_scores,
            'sources': result.sources,
            'summary': result.summary,
            'confidence': result.confidence
        }
        click.echo(json.dumps(json_result, indent=2))
    else:
        # Text format
        for i, chunk in enumerate(result.chunks, 1):
            click.echo(f"--- Chunk {i}: {chunk.source_type.upper()} {chunk.source_id} ---")
            click.echo(f"Relevance: {result.relevance_scores[i-1]:.2f}")
            click.echo(f"Content: {chunk.content}")
            click.echo()

@cli.command()
@click.option('--content', prompt='Content to add', help='Content to add to RAG system')
@click.option('--type', prompt='Source type', help='Source type (memory/task/code/document/feedback)')
@click.option('--source-id', prompt='Source ID', type=int, help='Source ID')
@click.option('--project-id', default=None, help='Project ID')
@click.option('--metadata', default=None, help='JSON metadata')
def add_rag_chunk(content, type, source_id, project_id, metadata):
    """Add content to the RAG system for intelligent retrieval."""
    from .rag_system import RAGSystem
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    rag_system = RAGSystem()
    
    # Parse metadata if provided
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo("[MCP] Error: Invalid JSON metadata")
            return
    
    # Add chunk
    chunk_id = rag_system.add_chunk(
        content=content,
        source_type=type,
        source_id=source_id,
        project_id=project_id,
        metadata=metadata_dict
    )
    
    click.echo(f"[MCP] RAG chunk added with ID: {chunk_id}")

@cli.command()
@click.option('--query-id', prompt='Query ID', type=int, help='Query ID to provide feedback for')
@click.option('--score', prompt='Feedback score (1-5)', type=int, help='Feedback score')
@click.option('--feedback-text', default=None, help='Additional feedback text')
def rag_feedback(query_id, score, feedback_text):
    """Provide feedback on RAG query results to improve the system."""
    from .rag_system import RAGSystem
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    rag_system = RAGSystem()
    
    if not 1 <= score <= 5:
        click.echo("[MCP] Error: Score must be between 1 and 5")
        return
    
    rag_system.add_feedback(query_id, score, feedback_text)
    click.echo(f"[MCP] Feedback added for query {query_id}")

@cli.command()
def rag_statistics():
    """Show RAG system statistics."""
    from .rag_system import RAGSystem
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    rag_system = RAGSystem()
    stats = rag_system.get_statistics()
    
    click.echo("[MCP] RAG System Statistics:")
    click.echo("=" * 40)
    click.echo(f"📊 Total chunks: {stats['total_chunks']}")
    click.echo(f"🔍 Total queries: {stats['total_queries']}")
    click.echo(f"📝 Avg tokens per query: {stats['average_tokens_per_query']:.1f}")
    click.echo()
    
    click.echo("📋 Chunks by type:")
    for chunk_type, count in stats['chunks_by_type'].items():
        click.echo(f"  {chunk_type}: {count}")
    click.echo()
    
    if stats['top_patterns']:
        click.echo("🎯 Top relevance patterns:")
        for pattern, score, count in stats['top_patterns']:
            click.echo(f"  '{pattern}': {score:.2f} (used {count} times)")

@cli.command()
@click.option('--days', default=90, type=int, help='Days old to consider for cleanup')
def rag_cleanup(days):
    """Clean up old, rarely accessed RAG chunks."""
    from .rag_system import RAGSystem
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    rag_system = RAGSystem()
    deleted_count = rag_system.cleanup_old_chunks(days)
    
    click.echo(f"[MCP] Cleaned up {deleted_count} old RAG chunks (older than {days} days)")

@cli.command()
@click.option('--pack-id', prompt='Pack ID', type=int, help='Context pack ID')
def get_context_pack(pack_id):
    """Retrieve a saved context pack."""
    from .context_manager import ContextManager
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    context_manager = ContextManager()
    pack = context_manager.get_context_pack(pack_id)
    
    if pack:
        click.echo(f"[MCP] Context Pack: {pack['name']}")
        click.echo(f"Description: {pack['description'] or 'No description'}")
        click.echo(f"Type: {pack['context_type']}")
        click.echo(f"Project: {pack['project_id'] or 'Global'}")
        click.echo(f"Access count: {pack['access_count']}")
        click.echo(f"Created: {pack['created_at']}")
        click.echo()
        
        if isinstance(pack['context_data'], dict) and 'context' in pack['context_data']:
            click.echo(pack['context_data']['context'])
        else:
            click.echo(json.dumps(pack['context_data'], indent=2))
    else:
        click.echo(f"[MCP] Context pack {pack_id} not found")

@cli.command()
@click.option('--context-type', default=None, help='Filter by context type')
@click.option('--project-id', default=None, help='Filter by project ID')
def list_context_packs(context_type, project_id):
    """List available context packs."""
    from .context_manager import ContextManager
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    context_manager = ContextManager()
    # Fix: never pass None, always pass str
    context_type = context_type if context_type is not None else ''
    project_id = project_id if project_id is not None else ''
    packs = context_manager.list_context_packs(context_type, project_id)
    
    if packs:
        click.echo(f"[MCP] Found {len(packs)} context packs:")
        click.echo("=" * 60)
        
        for pack in packs:
            click.echo(f"ID: {pack['id']}")
            click.echo(f"Name: {pack['name']}")
            click.echo(f"Type: {pack['context_type']}")
            click.echo(f"Project: {pack['project_id'] or 'Global'}")
            click.echo(f"Access count: {pack['access_count']}")
            click.echo(f"Created: {pack['created_at']}")
            click.echo("-" * 40)
    else:
        click.echo("[MCP] No context packs found")

@cli.command()
@click.option('--name', prompt='Template name', help='Template name')
@click.option('--description', default=None, help='Template description')
@click.option('--types', default='tasks,memories,progress', help='Context types')
@click.option('--max-tokens', default=1000, type=int, help='Maximum tokens')
def add_context_template(name, description, types, max_tokens):
    """Add a context template for reuse."""
    from .context_manager import ContextManager
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    context_manager = ContextManager()
    context_types = [t.strip() for t in types.split(',')]
    
    template_data = {
        'context_types': context_types,
        'max_tokens': max_tokens,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    template_id = context_manager.add_context_template(
        name=name,
        template_data=template_data,
        description=description,
        context_types=context_types,
        max_tokens=max_tokens
    )
    
    click.echo(f"[MCP] Context template added with ID: {template_id}")

@cli.command()
def list_context_templates():
    """List available context templates."""
    from .context_manager import ContextManager
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    context_manager = ContextManager()
    templates = context_manager.list_context_templates()
    
    if templates:
        click.echo(f"[MCP] Found {len(templates)} context templates:")
        click.echo("=" * 60)
        
        for template in templates:
            click.echo(f"ID: {template['id']}")
            click.echo(f"Name: {template['name']}")
            click.echo(f"Description: {template['description'] or 'No description'}")
            click.echo(f"Types: {', '.join(template['context_types'])}")
            click.echo(f"Max tokens: {template['max_tokens']}")
            click.echo(f"Created: {template['created_at']}")
            click.echo("-" * 40)
    else:
        click.echo("[MCP] No context templates found")

# Regex Search Commands
@cli.command()
@click.option('--pattern', prompt='Regex pattern', help='Regular expression pattern to search for')
@click.option('--type', default='combined', help='Search type (file_system/database/memory/task/rag/combined)')
@click.option('--scope', default='current_project', help='Search scope (current_project/all_projects/specific_path/database_only)')
@click.option('--case-sensitive', is_flag=True, help='Case sensitive search')
@click.option('--multiline', is_flag=True, help='Multiline regex mode')
@click.option('--dot-all', is_flag=True, help='Dot matches all characters including newlines')
@click.option('--max-results', default=50, type=int, help='Maximum number of results')
@click.option('--context-lines', default=3, type=int, help='Number of context lines to show')
@click.option('--file-patterns', default=None, help='Comma-separated file patterns to include')
@click.option('--exclude-patterns', default=None, help='Comma-separated file patterns to exclude')
@click.option('--format', default='text', help='Output format (text/json/compact)')
@click.option('--project-path', default=None, help='Specific project path to search')
def regex_search(pattern, type, scope, case_sensitive, multiline, dot_all, 
                max_results, context_lines, file_patterns, exclude_patterns, 
                format, project_path):
    """Perform regex search across files and database content."""
    from .regex_search import RegexSearchEngine, SearchQuery, SearchType, SearchScope, RegexSearchFormatter
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    
    # Parse search type
    try:
        search_type = SearchType(type)
    except ValueError:
        click.echo(f"[MCP] Error: Invalid search type '{type}'. Valid types: file_system, database, memory, task, rag, combined")
        return
    
    # Parse search scope
    try:
        search_scope = SearchScope(scope)
    except ValueError:
        click.echo(f"[MCP] Error: Invalid search scope '{scope}'. Valid scopes: current_project, all_projects, specific_path, database_only")
        return
    
    # Parse file patterns
    include_patterns = None
    if file_patterns:
        include_patterns = [p.strip() for p in file_patterns.split(',')]
    
    exclude_patterns_list = None
    if exclude_patterns:
        exclude_patterns_list = [p.strip() for p in exclude_patterns.split(',')]
    
    # Create search query
    query = SearchQuery(
        pattern=pattern,
        search_type=search_type,
        scope=search_scope,
        case_sensitive=case_sensitive,
        multiline=multiline,
        dot_all=dot_all,
        max_results=max_results,
        context_lines=context_lines,
        file_patterns=include_patterns,
        exclude_patterns=exclude_patterns_list,
        project_path=project_path
    )
    
    # Perform search
    search_engine = RegexSearchEngine(project_path=project_path)
    results = search_engine.search(query)
    
    # Format and display results
    formatted_output = RegexSearchFormatter.format_results(results, format, include_context=True)
    click.echo(formatted_output)
    
    # Show summary
    click.echo(f"\n[MCP] Search completed: {len(results)} results found")
    if len(results) >= max_results:
        click.echo(f"[MCP] Note: Results limited to {max_results} matches")

@cli.command()
@click.option('--pattern', prompt='Regex pattern', help='Regular expression pattern to search for')
@click.option('--file-patterns', default='*.py,*.js,*.ts,*.md,*.txt', help='Comma-separated file patterns to include')
@click.option('--exclude-patterns', default='__pycache__,*.pyc,*.git*', help='Comma-separated file patterns to exclude')
@click.option('--case-sensitive', is_flag=True, help='Case sensitive search')
@click.option('--max-results', default=20, type=int, help='Maximum number of results')
@click.option('--context-lines', default=2, type=int, help='Number of context lines to show')
def search_files(pattern, file_patterns, exclude_patterns, case_sensitive, max_results, context_lines):
    """Search for regex patterns in project files only."""
    from .regex_search import RegexSearchEngine, SearchQuery, SearchType, SearchScope, RegexSearchFormatter
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    
    # Parse file patterns
    include_patterns = [p.strip() for p in file_patterns.split(',')]
    exclude_patterns_list = [p.strip() for p in exclude_patterns.split(',')]
    
    # Create search query
    query = SearchQuery(
        pattern=pattern,
        search_type=SearchType.FILE_SYSTEM,
        scope=SearchScope.CURRENT_PROJECT,
        case_sensitive=case_sensitive,
        max_results=max_results,
        context_lines=context_lines,
        file_patterns=include_patterns,
        exclude_patterns=exclude_patterns_list
    )
    
    # Perform search
    search_engine = RegexSearchEngine()
    results = search_engine.search(query)
    
    # Format and display results
    formatted_output = RegexSearchFormatter.format_results(results, 'text', include_context=True)
    click.echo(formatted_output)
    
    click.echo(f"\n[MCP] File search completed: {len(results)} results found")

@cli.command()
@click.option('--pattern', prompt='Regex pattern', help='Regular expression pattern to search for')
@click.option('--type', default='combined', help='Database search type (memory/task/rag/combined)')
@click.option('--case-sensitive', is_flag=True, help='Case sensitive search')
@click.option('--max-results', default=30, type=int, help='Maximum number of results')
def search_database(pattern, type, case_sensitive, max_results):
    """Search for regex patterns in database content only."""
    from .regex_search import RegexSearchEngine, SearchQuery, SearchType, SearchScope, RegexSearchFormatter
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    
    # Parse search type
    try:
        search_type = SearchType(type)
    except ValueError:
        click.echo(f"[MCP] Error: Invalid search type '{type}'. Valid types: memory, task, rag, combined")
        return
    
    # Create search query
    query = SearchQuery(
        pattern=pattern,
        search_type=search_type,
        scope=SearchScope.DATABASE_ONLY,
        case_sensitive=case_sensitive,
        max_results=max_results
    )
    
    # Perform search
    search_engine = RegexSearchEngine()
    results = search_engine.search(query)
    
    # Format and display results
    formatted_output = RegexSearchFormatter.format_results(results, 'text', include_context=False)
    click.echo(formatted_output)
    
    click.echo(f"\n[MCP] Database search completed: {len(results)} results found")

@cli.command()
@click.option('--limit', default=10, type=int, help='Number of recent searches to show')
def search_history(limit):
    """Show recent regex search history."""
    from .regex_search import RegexSearchEngine
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    search_engine = RegexSearchEngine()
    history = search_engine.get_search_history(limit)
    
    if history:
        click.echo(f"[MCP] Recent search history (last {len(history)} searches):")
        click.echo("=" * 60)
        
        for i, search in enumerate(reversed(history), 1):
            click.echo(f"{i}. Pattern: {search['pattern']}")
            click.echo(f"   Type: {search['search_type']}")
            click.echo(f"   Scope: {search['scope']}")
            click.echo(f"   Results: {search['result_count']}")
            click.echo(f"   Time: {search['timestamp']}")
            click.echo("-" * 40)
    else:
        click.echo("[MCP] No search history found")

@cli.command()
def search_cache_stats():
    """Show regex search cache statistics."""
    from .regex_search import RegexSearchEngine
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    search_engine = RegexSearchEngine()
    stats = search_engine.get_cache_stats()
    
    click.echo("[MCP] Regex Search Cache Statistics:")
    click.echo("=" * 40)
    click.echo(f"📊 Cache entries: {stats['cache_size']}")
    click.echo(f"🔍 Cached results: {stats['total_cached_results']}")
    click.echo(f"📝 Search history: {stats['search_history_size']} entries")

@cli.command()
def clear_search_cache():
    """Clear the regex search cache."""
    from .regex_search import RegexSearchEngine
    
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    search_engine = RegexSearchEngine()
    search_engine.clear_cache()
    
    click.echo("[MCP] Regex search cache cleared")

@cli.command()
def performance_report(milestone=None):
    """Generate a performance report for the current project (objective metrics)."""
    from .performance_monitor import ObjectivePerformanceMonitor
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    workflow = get_workflow_manager()
    manager = get_project_manager()
    project_info = manager.get_project_info()
    if not project_info or 'path' not in project_info:
        click.echo("[MCP] No project found. Run 'init-project' first.")
        return
    try:
        from .task_manager import TaskManager
        task_manager = TaskManager()
    except Exception:
        task_manager = None
    feedback_model = getattr(workflow, 'feedback_model', None)
    monitor = ObjectivePerformanceMonitor(project_info['path'])
    report = monitor.generate_report(workflow, task_manager, feedback_model, milestone=milestone)
    click.echo("[MCP] Performance report generated:")
    click.echo(json.dumps(report, indent=2))

@cli.command()
@click.option('--interval', default=5.0, type=float, help='Monitoring interval in seconds')
@click.option('--duration', default=300, type=int, help='Duration to monitor in seconds (0 for continuous)')
def start_monitoring(interval, duration):
    """Start real-time performance monitoring with Prometheus/Netdata integration."""
    from .performance_monitor import RealTimeMonitor
    
    try:
        # Get current project path
        current_dir = os.getcwd()
        
        # Start real-time monitoring
        monitor = RealTimeMonitor(collection_interval=interval)
        monitor.start_monitoring()
        
        click.echo(f"[MCP] Real-time monitoring started (interval: {interval}s)")
        click.echo("[MCP] Prometheus metrics available at: http://localhost:9090/metrics")
        click.echo("[MCP] Press Ctrl+C to stop monitoring")
        
        if duration > 0:
            import time
            time.sleep(duration)
            monitor.stop_monitoring()
            click.echo("[MCP] Monitoring completed")
        else:
            # Continuous monitoring
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop_monitoring()
                click.echo("\n[MCP] Monitoring stopped")
        
    except Exception as e:
        click.echo(f"[MCP] Failed to start monitoring: {e}")

@cli.command()
def monitoring_status():
    """Show current monitoring status and metrics."""
    from .performance_monitor import RealTimeMonitor
    
    try:
        monitor = RealTimeMonitor()
        current_metrics = monitor.get_current_metrics()
        
        if current_metrics:
            click.echo("[MCP] Current System Metrics:")
            click.echo("=" * 40)
            for metric, value in current_metrics.items():
                click.echo(f"📊 {metric}: {value:.2f}")
        else:
            click.echo("[MCP] No metrics available. Start monitoring first with 'start-monitoring'")
            
    except Exception as e:
        click.echo(f"[MCP] Error getting monitoring status: {e}")

@cli.command()
def get_alerts():
    """Show current system alerts."""
    from .performance_monitor import RealTimeMonitor
    
    try:
        monitor = RealTimeMonitor()
        alerts = monitor.alerting_system.get_alerts()
        
        if alerts:
            click.echo("[MCP] Current Alerts:")
            click.echo("=" * 40)
            for alert in alerts:
                severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡"
                click.echo(f"{severity_icon} {alert['metric_name']}: {alert['value']:.2f} - {alert['message']}")
        else:
            click.echo("[MCP] No active alerts")
            
    except Exception as e:
        click.echo(f"[MCP] Error getting alerts: {e}")

@cli.command()
@click.option('--backend', default='sqlitefaiss', type=click.Choice(['sqlitefaiss', 'milvus', 'annoy', 'qdrant']), help='Vector backend to use')
@click.option('--config', default=None, help='JSON configuration for the backend')
def vector_backend_status(backend, config):
    """Show status of vector backends."""
    from .vector_memory import get_vector_backend
    
    try:
        config_dict = json.loads(config) if config else {}
        vector_backend = get_vector_backend(backend, config_dict)
        stats = vector_backend.get_stats()
        
        click.echo(f"[MCP] Vector Backend Status ({backend}):")
        click.echo("=" * 40)
        for key, value in stats.items():
            click.echo(f"📊 {key}: {value}")
            
    except Exception as e:
        click.echo(f"[MCP] Error getting vector backend status: {e}")

@cli.command()
@click.option('--backend', default='sqlitefaiss', type=click.Choice(['sqlitefaiss', 'milvus', 'annoy', 'qdrant']), help='Vector backend to use')
@click.option('--config', default=None, help='JSON configuration for the backend')
def create_vector_index(backend, config):
    """Create or rebuild vector index for the specified backend."""
    from .vector_memory import get_vector_backend
    
    try:
        config_dict = json.loads(config) if config else {}
        vector_backend = get_vector_backend(backend, config_dict)
        vector_backend.create_index()
        
        click.echo(f"[MCP] Vector index created/rebuilt for {backend}")
        
    except Exception as e:
        click.echo(f"[MCP] Error creating vector index: {e}")

@cli.command()
@click.option('--backend', default='sqlitefaiss', type=click.Choice(['sqlitefaiss', 'milvus', 'annoy', 'qdrant']), help='Vector backend to use')
@click.option('--config', default=None, help='JSON configuration for the backend')
def load_vector_index(backend, config):
    """Load vector index into memory for the specified backend."""
    from .vector_memory import get_vector_backend
    
    try:
        config_dict = json.loads(config) if config else {}
        vector_backend = get_vector_backend(backend, config_dict)
        vector_backend.load_index()
        
        click.echo(f"[MCP] Vector index loaded for {backend}")
        
    except Exception as e:
        click.echo(f"[MCP] Error loading vector index: {e}")

@cli.command()
@click.option('--format', default='json', type=click.Choice(['json', 'markdown']), help='Output format')
def statistical_report(format):
    """Generate a statistical analysis report for the current project (task trends, feedback, resource usage)."""
    from .performance_monitor import ObjectivePerformanceMonitor
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    workflow = get_workflow_manager()
    manager = get_project_manager()
    project_info = manager.get_project_info()
    if not project_info or 'path' not in project_info:
        click.echo("[MCP] No project found. Run 'init-project' first.")
        return
    try:
        from .task_manager import TaskManager
        task_manager = TaskManager()
    except Exception:
        task_manager = None
    feedback_model = getattr(workflow, 'feedback_model', None)
    monitor = ObjectivePerformanceMonitor(project_info['path'])
    report = monitor.generate_statistical_report(workflow, task_manager, feedback_model, format=format)
    click.echo(report)

@cli.command('feedback-analytics-report')
@click.option('--format', type=click.Choice(['json', 'markdown']), default='json', help='Output format')
def feedback_analytics_report(format):
    """Generate and display the feedback analytics report."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    # Assume current project path is cwd
    project_path = os.getcwd()
    monitor = ObjectivePerformanceMonitor(project_path)
    task_manager = TaskManager()
    workflow_manager = WorkflowManager()
    rag_system = RAGSystem()
    reminder_engine = EnhancedReminderEngine()
    # System feedback: could be loaded from MCPServer if available, else empty
    system_feedback = []
    report = monitor.generate_feedback_analytics_report(
        task_manager, workflow_manager, rag_system, reminder_engine, system_feedback, format=format
    )
    click.echo(report)

@cli.command()
@click.option('--format', default='json', type=click.Choice(['json', 'markdown']), help='Output format')
def lessons_learned(format):
    """Show lessons learned from all feedback sources for post-project review."""
    from .performance_monitor import ObjectivePerformanceMonitor, LessonsLearnedModule
    try:
        task_manager = TaskManager()
        workflow_manager = WorkflowManager()
        rag_system = RAGSystem()
        lessons_module = LessonsLearnedModule(task_manager, workflow_manager, rag_system)
        output = lessons_module.export_lessons(format=format)
        click.echo(output)
    except Exception as e:
        click.echo(f"[MCP] Error generating lessons learned: {e}")

# Helper Functions
def _get_status_icon(status):
    """Get status icon for display."""
    icons = {
        'pending': '⏳',
        'in_progress': '🔄',
        'partial': '📝',
        'completed': '✅',
        'blocked': '🚫',
        'cancelled': '❌'
    }
    return icons.get(status, '❓')

def _display_task_tree(tasks, level=0):
    """Display tasks in tree structure."""
    indent = "  " * level
    
    for task in tasks:
        status_icon = _get_status_icon(task['status'])
        critical_marker = " 🔴" if task.get('accuracy_critical') else ""
        
        click.echo(f"{indent}{status_icon} {task['title']} (ID: {task['id']}, P{task['priority']}){critical_marker}")
        
        if task['children']:
            _display_task_tree(task['children'], level + 1)

@cli.command()
@click.argument('step_name')
@click.option('--after', default=None, help='Add after this step (optional)')
def add_workflow_step(step_name, after):
    """Dynamically add a workflow step at runtime."""
    workflow = get_workflow_manager()
    success = workflow.add_step(step_name, after=after)
    if success:
        click.echo(f"[MCP] Step '{step_name}' added after '{after}'.")
    else:
        click.echo(f"[MCP] Step '{step_name}' already exists.")

@cli.command()
@click.argument('step_name')
def remove_workflow_step(step_name):
    """Dynamically remove a workflow step at runtime."""
    workflow = get_workflow_manager()
    success = workflow.remove_step(step_name)
    if success:
        click.echo(f"[MCP] Step '{step_name}' removed.")
    else:
        click.echo(f"[MCP] Step '{step_name}' not found.")

@cli.command()
@click.argument('step_name')
@click.option('--config', prompt='Config as JSON', help='New config for the step (JSON)')
def modify_workflow_step(step_name, config):
    """Modify the configuration of a workflow step at runtime."""
    import json
    workflow = get_workflow_manager()
    try:
        config_dict = json.loads(config)
    except Exception as e:
        click.echo(f"[MCP] Invalid config JSON: {e}")
        return
    success = workflow.modify_step(step_name, config_dict)
    if success:
        click.echo(f"[MCP] Step '{step_name}' modified.")
    else:
        click.echo(f"[MCP] Step '{step_name}' not found.")

@cli.command()
@click.argument('step_name')
@click.argument('next_steps', nargs=-1)
def set_next_steps(step_name, next_steps):
    """Set possible next steps for a workflow step (branching/parallel)."""
    workflow = get_workflow_manager()
    success = workflow.set_next_steps(step_name, list(next_steps))
    if success:
        click.echo(f"[MCP] Next steps for '{step_name}' set to: {', '.join(next_steps)}")
    else:
        click.echo(f"[MCP] Step '{step_name}' not found.")

@cli.command()
@click.argument('step_name')
@click.argument('next_step')
def add_next_step(step_name, next_step):
    """Add a possible next step to a workflow step."""
    workflow = get_workflow_manager()
    success = workflow.add_next_step(step_name, next_step)
    if success:
        click.echo(f"[MCP] Next step '{next_step}' added to '{step_name}'.")
    else:
        click.echo(f"[MCP] Step '{step_name}' not found.")

@cli.command()
@click.argument('step_name')
@click.argument('next_step')
def remove_next_step(step_name, next_step):
    """Remove a possible next step from a workflow step."""
    workflow = get_workflow_manager()
    success = workflow.remove_next_step(step_name, next_step)
    if success:
        click.echo(f"[MCP] Next step '{next_step}' removed from '{step_name}'.")
    else:
        click.echo(f"[MCP] Step '{step_name}' or next step not found.")

@cli.command()
@click.argument('step_name')
def show_next_steps(step_name):
    """Show all possible next steps for a workflow step."""
    workflow = get_workflow_manager()
    next_steps = workflow.get_next_steps(step_name)
    click.echo(f"[MCP] Next steps for '{step_name}': {', '.join(next_steps) if next_steps else '(none)'}")

@cli.command()
def list_endpoints():
    """List all available MCP API endpoints and their schemas."""
    import requests
    import json
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    # Assume server is running locally on default port
    url = "http://localhost:8000/jsonrpc"
    payload = {
        "jsonrpc": "2.0",
        "method": "list_endpoints",
        "params": {},
        "id": 1
    }
    try:
        response = requests.post(url, json=payload)
        result = response.json().get('result', {})
        endpoints = result.get('endpoints', [])
        click.echo("[MCP] Available API Endpoints:")
        for ep in endpoints:
            click.echo(f"- {ep}")
        click.echo("\nUse 'get-endpoint-schema --endpoint <name>' for details.")
    except Exception as e:
        click.echo(f"[MCP] Error listing endpoints: {e}")

@cli.command()
@click.option('--endpoint', required=True, help='Endpoint name')
def get_endpoint_schema(endpoint):
    """Get the schema for a given MCP API endpoint."""
    import requests
    import json
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    url = "http://localhost:8000/jsonrpc"
    payload = {
        "jsonrpc": "2.0",
        "method": "get_endpoint_schema",
        "params": {"endpoint": endpoint},
        "id": 1
    }
    try:
        response = requests.post(url, json=payload)
        result = response.json().get('result', {})
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"[MCP] Error getting endpoint schema: {e}")

@cli.command()
@click.pass_context
def status(ctx):
    """Output all project, workflow, and task state as a single JSON blob for LLM/IDE integration."""
    manager = get_project_manager(ctx)
    project_info = manager.get_project_info()
    workflow = get_workflow_manager(ctx)
    workflow_status = getattr(workflow, '__dict__', {})
    data_dir = os.path.join(project_info.get('path', os.getcwd()), 'data')
    os.makedirs(data_dir, exist_ok=True)
    tasks_db_path = os.path.join(data_dir, 'tasks.db')
    task_manager = TaskManager(tasks_db_path)
    tasks = task_manager.get_tasks() if hasattr(task_manager, 'get_tasks') else []
    def make_serializable(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return obj
    def serialize_tasks(tasks):
        if isinstance(tasks, list):
            return [serialize_tasks(t) for t in tasks]
        if isinstance(tasks, dict):
            return {k: serialize_tasks(v) for k, v in tasks.items()}
        if isinstance(tasks, (str, int, float, bool)) or tasks is None:
            return tasks
        if isinstance(tasks, (datetime.datetime, datetime.date)):
            return tasks.isoformat()
        if isinstance(tasks, bytes):
            return tasks.decode('utf-8')
        # Fallback: convert to string
        return str(tasks)
    # Sanitize workflow_status for serialization
    workflow_status_sanitized = dict(workflow_status)
    if 'task_manager' in workflow_status_sanitized:
        workflow_status_sanitized['task_manager'] = str(workflow_status_sanitized['task_manager'].__class__.__name__)
    if 'steps' in workflow_status_sanitized:
        workflow_status_sanitized['steps'] = {k: str(v.__class__.__name__) for k, v in workflow_status_sanitized['steps'].items()}
    output = {
        'project': project_info,
        'workflow': workflow_status_sanitized,
        'tasks': serialize_tasks(tasks)
    }
    print('[DEBUG] output before serialization:', output)
    click.echo(json.dumps(output, indent=2))

@cli.command()
@click.argument('name')
@click.option('--attributes', default=None, help='JSON string of attributes')
def add_entity(name: str, attributes: str) -> None:
    """Add an entity to the simulated reality model."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
@click.argument('description')
@click.argument('timestamp')
@click.option('--entities', default=None, help='Comma-separated list of entity names')
def add_event(description, timestamp, entities):
    """Add an event to the simulated reality model."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
@click.argument('name')
@click.argument('value')
@click.argument('timestamp')
def add_state(name, value, timestamp):
    """Add a state to the simulated reality model."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
def query_entities():
    """Query all entities in the simulated reality model."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
def query_events():
    """Query all events in the simulated reality model."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
def query_states():
    """Query all states in the simulated reality model."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
def export_reality():
    """Export the entire simulated reality model as JSON."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
    return

@cli.command()
@click.pass_context
def auto_prompt(ctx):
    """Output all missing/needed info for the LLM to fill in, based on project state and config."""
    manager = get_project_manager(ctx)
    questions = manager.get_questions()
    missing = {}
    for section, section_questions in questions.items():
        for key, value in section_questions.items():
            if not value or (isinstance(value, str) and not value.strip()):
                if section not in missing:
                    missing[section] = []
                missing[section].append(key)
    if not missing:
        click.echo("[MCP] All config questions are answered.")
        return
    click.echo("[MCP] Missing/Needed Info (for LLM):")
    click.echo(json.dumps(missing, indent=2))

@cli.command()
@click.pass_context
def mindmap_export(ctx):
    """Export a mind map of all tasks, memories, and their crosslinks for LLM deep research and recall."""
    manager = get_project_manager(ctx)
    project_info = manager.get_project_info()
    data_dir = os.path.join(project_info.get('path', os.getcwd()), 'data')
    tasks_db_path = os.path.join(data_dir, 'tasks.db')
    task_manager = TaskManager(tasks_db_path)
    tasks = task_manager.get_tasks() if hasattr(task_manager, 'get_tasks') else []
    # For now, just output a simple tree of tasks by parent_id
    tree = {}
    by_id = {t['id']: t for t in tasks}
    for t in tasks:
        parent = t.get('parent_id')
        if parent and parent in by_id:
            by_id[parent].setdefault('subtasks', []).append(t)
        else:
            tree.setdefault('root', []).append(t)
    click.echo("[MCP] Mind Map Export (JSON):")
    click.echo(json.dumps(tree, indent=2))

@cli.command()
@click.option('--queries', prompt='Comma-separated queries', help='Comma-separated list of queries for batch vector search')
@click.option('--limit', default=10, help='Maximum number of results per query')
@click.option('--encoder', default='tfidf', type=click.Choice(['tfidf', 'rabitq']), help='Vector encoder for batch search (tfidf, rabitq)')
def batch_vector_search(queries, limit, encoder):
    """Batch vector search: run multiple queries at once (ANN-ready)."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    if encoder == 'rabitq':
        encoder_instance = RaBitQEncoder()
    else:
        encoder_instance = TFIDFEncoder()
    from .unified_memory import UnifiedMemoryManager
    manager = UnifiedMemoryManager(encoder=encoder_instance)
    query_list = [q.strip() for q in queries.split(',') if q.strip()]
    results = manager.batch_vector_search(query_list, limit=limit)
    for i, (query, res) in enumerate(zip(query_list, results), 1):
        click.echo(f"\nQuery {i}: '{query}'")
        if not res:
            click.echo("  No results found.")
            continue
        for j, memory in enumerate(res, 1):
            click.echo(f"  {j}. Memory {memory.get('id', '?')} (Similarity: {memory.get('similarity', 0):.3f})")
            click.echo(f"     Text: {memory.get('text', '')[:80]}...")
            click.echo(f"     Type: {memory.get('memory_type', '')}")
            click.echo(f"     Priority: {memory.get('priority', 0)}")
            click.echo(f"     Created: {memory.get('created_at', 'N/A')}")
            if memory.get('tags'):
                click.echo(f"     Tags: {', '.join(memory['tags'])}")

@cli.command()
@click.argument('flag_name')
@click.argument('state', type=click.Choice(['on', 'off']))
def set_feature_flag(flag_name: str, state: Literal['on', 'off']) -> None:
    """Enable or disable a feature flag at runtime."""
    if not FEATURE_FLAGS.get('simulated_reality', False):
        click.echo("[MCP] SimulatedReality is a planned/experimental feature. Enable the feature flag and implement the backend to use this command.")
        return
    if flag_name not in FEATURE_FLAGS:
        click.echo(f"[MCP] Unknown feature flag: {flag_name}")
        return
    FEATURE_FLAGS[flag_name] = (state == 'on')
    click.echo(f"[MCP] Feature flag '{flag_name}' set to {state}")

@cli.command('list-feature-flags')
def list_feature_flags():
    """List all feature flags and their current state."""
    for flag, enabled in FEATURE_FLAGS.items():
        click.echo(f"{flag}: {'on' if enabled else 'off'}")

@cli.command()
@click.option('--task-id', prompt='Task ID', type=int, help='Task ID to add feedback for')
@click.option('--feedback', prompt='Feedback text', help='Feedback content')
@click.option('--impact', default=0, type=int, help='Impact score (-5 to 5)')
@click.option('--principle', default=None, help='Learning principle')
@click.option('--rating', default=3, type=int, help='Rating (1-5)')
def add_task_feedback(task_id, feedback, impact, principle, rating):
    """Add feedback to a completed task. If the task is flagged or marked critical, a reminder will be created or updated automatically."""
    from .task_manager import TaskManager
    task_manager = TaskManager()
    try:
        feedback_id = task_manager.add_task_feedback(
            task_id=task_id,
            feedback_text=feedback,
            impact_score=impact,
            principle=principle,
            feedback_type='general'
        )
        click.echo(f"[MCP] Feedback added with ID: {feedback_id}")
        # Check if a reminder was created/updated
        from .unified_memory import UnifiedMemoryManager
        unified = UnifiedMemoryManager()
        reminders = unified.reminder_engine.get_reminders_by_task_id(task_id)
        if reminders:
            click.echo(f"[MCP] Reminder(s) for this task have been created or updated due to feedback (flagged/critical). Reminder IDs: {[r['id'] for r in reminders]}")
    except Exception as e:
        click.echo(f"[MCP] Error adding feedback: {e}")

@cli.command()
def dashboard():
    """Show a unified project health, risk, and progress KPI dashboard."""
    from .performance_monitor import ObjectivePerformanceMonitor
    try:
        manager = get_project_manager()
        workflow = get_workflow_manager()
        project_info = manager.get_project_info()
        task_manager = TaskManager()
        monitor = ObjectivePerformanceMonitor(project_info.get('path', os.getcwd()))
        # Health
        validation = manager.validate_configuration()
        completion = validation.get('completion_rate', 0.0)
        errors = validation.get('errors', [])
        warnings = validation.get('warnings', [])
        # Progress KPIs
        metrics = monitor.collect_metrics(workflow, task_manager)
        # Risk
        critical_tasks = [t for t in task_manager.get_tasks() if t.get('accuracy_critical') or 'critical' in (t.get('tags') or [])]
        flagged_tasks = [t for t in task_manager.get_tasks() if 'flagged' in (t.get('tags') or [])]
        # Output
        click.echo("[MCP] Project Dashboard")
        click.echo("=" * 40)
        click.echo(f"Project: {project_info.get('name', 'Unknown')}")
        click.echo(f"Status: {project_info.get('status', 'Unknown')}")
        click.echo(f"Completion: {completion:.1f}%")
        if errors:
            click.echo("\n❌ Errors:")
            for e in errors:
                click.echo(f"  - {e}")
        if warnings:
            click.echo("\n⚠️  Warnings:")
            for w in warnings:
                click.echo(f"  - {w}")
        click.echo(f"\n📈 Task Completion Rate: {metrics.get('task_completion_rate', 0):.1f}%")
        click.echo(f"🧠 Code Files: {metrics.get('code_file_count', 0)} | Test Files: {metrics.get('test_file_count', 0)}")
        click.echo(f"💾 Disk Usage: {metrics.get('disk_usage_mb', 0):.1f} MB")
        click.echo(f"📝 Avg Feedback Score: {metrics.get('avg_feedback_score', 'N/A')}")
        click.echo(f"\n🔴 Critical Tasks: {len(critical_tasks)} | 🚩 Flagged Tasks: {len(flagged_tasks)}")
        if critical_tasks:
            click.echo("  - Critical Task IDs: " + ', '.join(str(t['id']) for t in critical_tasks))
        if flagged_tasks:
            click.echo("  - Flagged Task IDs: " + ', '.join(str(t['id']) for t in flagged_tasks))
        click.echo("=" * 40)
        click.echo("[MCP] Dashboard complete.")
    except Exception as e:
        click.echo(f"[MCP] Error generating dashboard: {e}")

@cli.command()
def database_status():
    """Show database status and statistics."""
    from .database_manager import OptimizedDatabaseManager
    
    try:
        # Get current project path
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, 'data')
        db_path = os.path.join(data_dir, 'unified_memory.db')
        
        db_manager = OptimizedDatabaseManager(db_path)
        stats = db_manager.get_database_stats()
        
        click.echo("[MCP] Database Status:")
        click.echo("=" * 40)
        for key, value in stats.items():
            if key == 'file_size_mb':
                click.echo(f"📊 {key}: {value:.2f} MB")
            else:
                click.echo(f"📊 {key}: {value}")
                
    except Exception as e:
        click.echo(f"[MCP] Error getting database status: {e}")

@cli.command()
def optimize_database():
    """Run database optimization (VACUUM, ANALYZE, REINDEX)."""
    from .database_manager import OptimizedDatabaseManager
    
    try:
        # Get current project path
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, 'data')
        db_path = os.path.join(data_dir, 'unified_memory.db')
        
        click.echo("[MCP] Starting database optimization...")
        
        db_manager = OptimizedDatabaseManager(db_path)
        db_manager.optimize_database()
        
        click.echo("[MCP] Database optimization completed")
        
    except Exception as e:
        click.echo(f"[MCP] Error optimizing database: {e}")

@cli.command()
@click.option('--table', default=None, help='Specific table to show schema for')
def show_database_schema(table):
    """Show database schema information."""
    from .database_manager import OptimizedDatabaseManager
    
    try:
        # Get current project path
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, 'data')
        db_path = os.path.join(data_dir, 'unified_memory.db')
        
        db_manager = OptimizedDatabaseManager(db_path)
        
        if table:
            # Show specific table schema
            schema = db_manager.get_table_info(table)
            click.echo(f"[MCP] Schema for table '{table}':")
            click.echo("=" * 40)
            for column in schema:
                click.echo(f"📋 {column['name']} ({column['type']}) - {'NOT NULL' if column['notnull'] else 'NULL'}")
        else:
            # Show all tables
            tables = db_manager.execute_query("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            click.echo("[MCP] Database Tables:")
            click.echo("=" * 40)
            for table_info in tables:
                click.echo(f"📋 {table_info['name']}")
                
    except Exception as e:
        click.echo(f"[MCP] Error showing database schema: {e}")

# IDE Integration Commands
@cli.command()
@click.option('--ide', type=click.Choice(['cursor', 'vscode', 'claude', 'lmstudio', 'ollama']), prompt='IDE', help='Target IDE for configuration')
@click.option('--project-path', default=None, help='Project path (defaults to current directory)')
@click.option('--api-key', default='changeme', help='API key for authentication')
@click.option('--output-dir', default='config', help='Output directory for configuration files')
def setup_ide_integration(ide, project_path, api_key, output_dir):
    """Setup IDE integration for the MCP server."""
    project_path = project_path or os.getcwd()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Common environment variables
    env_vars = {
        "PYTHONPATH": f"{project_path}/src",
        "MCP_API_KEY": api_key,
        "MCP_PROJECT_PATH": project_path,
        "MCP_VECTOR_BACKEND": "sqlitefaiss",
        "MCP_LOG_LEVEL": "INFO",
        "MCP_DEBUG_MODE": "false"
    }
    
    # Common capabilities
    capabilities = {
        "workflow_management": True,
        "task_management": True,
        "memory_management": True,
        "context_management": True,
        "rag_system": True,
        "performance_monitoring": True,
        "experimental_lobes": True,
        "project_management": True,
        "self_improvement": True
    }
    
    # Common methods
    methods = [
        "create_project", "get_project_status", "update_configuration",
        "start_workflow_step", "create_task", "update_task", "get_tasks",
        "add_memory", "search_memories", "get_context", "rag_query",
        "get_performance", "optimize_system", "run_lobe", "get_lobe_status",
        "list_endpoints", "get_endpoint_schema"
    ]
    
    if ide == 'cursor':
        config = {
            "name": "Agentic Workflow MCP",
            "description": "Advanced MCP server for agentic development workflows",
            "command": "python",
            "args": ["-m", "src.mcp.mcp_stdio"],
            "env": env_vars,
            "capabilities": capabilities,
            "methods": methods,
            "version": "1.0.0",
            "author": "Agentic Workflow Team"
        }
        config_file = output_path / "cursor-mcp.json"
        
    elif ide == 'vscode':
        config = {
            "mcp.servers": {
                "agentic-workflow": {
                    "command": "python",
                    "args": ["-m", "src.mcp.mcp_stdio"],
                    "env": env_vars,
                    "capabilities": capabilities
                }
            },
            "mcp.autoConnect": True,
            "mcp.showNotifications": True,
            "mcp.logLevel": "info",
            "mcp.enableTelemetry": False,
            "mcp.experimentalFeatures": True
        }
        config_file = output_path / "vscode-settings.json"
        
    elif ide == 'claude':
        config = {
            "agentic-workflow": {
                "name": "Agentic Workflow MCP",
                "description": "Advanced MCP server for agentic development workflows",
                "command": "python",
                "args": ["-m", "src.mcp.mcp_stdio"],
                "env": env_vars,
                "capabilities": capabilities,
                "autoConnect": True,
                "enabled": True
            }
        }
        config_file = output_path / "claude-mcp.json"
        
    elif ide == 'lmstudio':
        config = {
            "mcp_servers": {
                "agentic-workflow": {
                    "command": "python",
                    "args": ["-m", "src.mcp.mcp_stdio"],
                    "env": env_vars
                }
            }
        }
        config_file = output_path / "lmstudio-config.json"
        
    elif ide == 'ollama':
        config = {
            "mcp_servers": {
                "agentic-workflow": {
                    "command": "python",
                    "args": ["-m", "src.mcp.mcp_stdio"],
                    "env": env_vars
                }
            }
        }
        config_file = output_path / "ollama-config.yaml"
    
    # Write configuration file
    with open(config_file, 'w') as f:
        if ide == 'ollama':
            try:
                import yaml
                yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                click.echo("[MCP] Warning: PyYAML not installed, using JSON format")
                json.dump(config, f, indent=2)
        else:
            json.dump(config, f, indent=2)
    
    click.echo(f"[MCP] {ide.upper()} configuration created: {config_file}")
    click.echo(f"[MCP] Copy this file to the appropriate location for {ide}:")
    
    if ide == 'cursor':
        click.echo("  - Open Cursor settings")
        click.echo("  - Navigate to 'AI' → 'Model Context Protocol'")
        click.echo("  - Import the configuration file")
    elif ide == 'vscode':
        click.echo("  - Copy to .vscode/settings.json in your workspace")
    elif ide == 'claude':
        click.echo("  - Copy to ~/.config/claude/mcp-servers.json")
    elif ide == 'lmstudio':
        click.echo("  - Add to LMStudio configuration")
    elif ide == 'ollama':
        click.echo("  - Add to Ollama configuration")

@cli.command()
@click.option('--ide', type=click.Choice(['cursor', 'vscode', 'claude', 'lmstudio', 'ollama']), prompt='IDE', help='IDE to test integration with')
def test_ide_integration(ide):
    """Test IDE integration by running a simple MCP request."""
    import subprocess
    
    # Create a test request
    test_request = {
        "jsonrpc": "2.0",
        "method": "list_endpoints",
        "params": {},
        "id": 1
    }
    
    click.echo(f"[MCP] Testing {ide.upper()} integration...")
    click.echo(f"[MCP] Sending test request: {json.dumps(test_request)}")
    
    try:
        # Run the stdio server with test input
        result = subprocess.run(
            [sys.executable, "-m", "src.mcp.mcp_stdio"],
            input=json.dumps(test_request) + "\n",
            text=True,
            capture_output=True,
            timeout=10,
            env={
                **os.environ,
                "PYTHONPATH": f"{os.getcwd()}/src",
                "MCP_API_KEY": "changeme",
                "MCP_PROJECT_PATH": os.getcwd(),
                "MCP_VECTOR_BACKEND": "sqlitefaiss"
            }
        )
        
        if result.returncode == 0:
            click.echo(f"[MCP] ✅ {ide.upper()} integration test successful!")
            click.echo(f"[MCP] Response: {result.stdout.strip()}")
        else:
            click.echo(f"[MCP] ❌ {ide.upper()} integration test failed!")
            click.echo(f"[MCP] Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        click.echo(f"[MCP] ❌ {ide.upper()} integration test timed out!")
    except Exception as e:
        click.echo(f"[MCP] ❌ {ide.upper()} integration test error: {e}")

@cli.command()
def list_ide_configurations():
    """List all available IDE configuration templates."""
    config_dir = Path("config")
    
    if not config_dir.exists():
        click.echo("[MCP] No IDE configurations found. Run 'setup-ide-integration' first.")
        return
    
    click.echo("[MCP] Available IDE configurations:")
    click.echo("=" * 40)
    
    for config_file in config_dir.glob("*.json"):
        ide_name = config_file.stem.replace("-", " ").title()
        click.echo(f"  📁 {ide_name}: {config_file}")
    
    for config_file in config_dir.glob("*.yaml"):
        ide_name = config_file.stem.replace("-", " ").title()
        click.echo(f"  📁 {ide_name}: {config_file}")
    
    click.echo("\n[MCP] To use these configurations:")
    click.echo("  1. Copy the appropriate file to your IDE's config directory")
    click.echo("  2. Update the paths and API keys as needed")
    click.echo("  3. Restart your IDE")

@cli.command()
@click.option('--ide', type=click.Choice(['cursor', 'vscode', 'claude', 'lmstudio', 'ollama']), prompt='IDE', help='IDE to generate documentation for')
def generate_ide_docs(ide):
    """Generate IDE-specific integration documentation."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    if ide == 'cursor':
        content = """# Cursor IDE Integration

## Setup Instructions

1. **Install the MCP server**:
   ```bash
   git clone <repository>
   cd agentic-workflow
   python -m pip install -e .
   ```

2. **Configure Cursor**:
   - Open Cursor settings
   - Navigate to "AI" → "Model Context Protocol"
   - Add a new server configuration using the provided JSON

3. **Environment Variables**:
   ```bash
   export MCP_API_KEY="your_api_key_here"
   export MCP_PROJECT_PATH="/path/to/your/project"
   export MCP_VECTOR_BACKEND="sqlitefaiss"
   ```

## Usage

Once configured, you can use MCP commands directly in Cursor's AI chat:

- Create projects: `/mcp create_project`
- Manage tasks: `/mcp create_task`
- Search memories: `/mcp search_memories`
- Query RAG: `/mcp rag_query`

## Troubleshooting

- Ensure PYTHONPATH is set correctly
- Check that the MCP server is accessible
- Verify API key configuration
"""
    elif ide == 'vscode':
        content = """# VS Code Integration

## Setup Instructions

1. **Install MCP Extension**:
   - Search for "Model Context Protocol" in VS Code extensions
   - Install the official MCP extension

2. **Configure Extension**:
   - Copy the provided settings to `.vscode/settings.json`
   - Update paths and API keys as needed

3. **Usage**:
   - Use Command Palette: `MCP: Connect to Server`
   - Select "agentic-workflow" server
   - Access MCP features through the MCP panel

## Features

- Integrated task management
- Memory search and management
- RAG system queries
- Performance monitoring
- Experimental lobes access

## Troubleshooting

- Check extension logs for errors
- Verify server configuration
- Ensure Python environment is correct
"""
    else:
        content = f"""# {ide.upper()} Integration

## Setup Instructions

1. **Install the MCP server**:
   ```bash
   git clone <repository>
   cd agentic-workflow
   python -m pip install -e .
   ```

2. **Configure {ide.upper()}**:
   - Use the provided configuration file
   - Update paths and API keys as needed

3. **Usage**:
   - Connect to the MCP server
   - Use available MCP methods

## Available Methods

- Project management
- Task management
- Memory management
- Context management
- RAG system
- Performance monitoring
- Experimental lobes

## Troubleshooting

- Check server logs
- Verify configuration
- Test connectivity
"""
    
    doc_file = docs_dir / f"{ide}-integration.md"
    with open(doc_file, 'w') as f:
        f.write(content)
    
    click.echo(f"[MCP] {ide.upper()} documentation generated: {doc_file}")

# Plugin Management Commands
@cli.command()
def list_plugins():
    """List all available plugins."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        # Discover plugins
        discovered = manager.discover_plugins()
        
        if not discovered:
            click.echo("[MCP] No plugins found.")
            click.echo("[MCP] Use 'create-plugin-template' to create a new plugin.")
            return
        
        click.echo("[MCP] Available Plugins:")
        click.echo("=" * 40)
        
        for plugin_name in discovered:
            plugin_info = manager.get_plugin_info(plugin_name)
            if plugin_info:
                status = "✅" if plugin_info.loaded else "⭕"
                enabled = "🟢" if plugin_info.enabled else "🔴"
                click.echo(f"  {status} {enabled} {plugin_name}")
                click.echo(f"     Version: {plugin_info.metadata.version}")
                click.echo(f"     Description: {plugin_info.metadata.description}")
                if plugin_info.error:
                    click.echo(f"     Error: {plugin_info.error}")
                click.echo()
            else:
                click.echo(f"  ⚠️  {plugin_name} (metadata not found)")
        
    except Exception as e:
        click.echo(f"[MCP] Error listing plugins: {e}")

@cli.command()
@click.argument('plugin_name')
def load_plugin(plugin_name):
    """Load a specific plugin."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        success = manager.load_plugin(plugin_name)
        if success:
            click.echo(f"[MCP] ✅ Plugin loaded successfully: {plugin_name}")
        else:
            click.echo(f"[MCP] ❌ Failed to load plugin: {plugin_name}")
            
    except Exception as e:
        click.echo(f"[MCP] Error loading plugin: {e}")

@cli.command()
@click.argument('plugin_name')
def unload_plugin(plugin_name):
    """Unload a specific plugin."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        success = manager.unload_plugin(plugin_name)
        if success:
            click.echo(f"[MCP] ✅ Plugin unloaded successfully: {plugin_name}")
        else:
            click.echo(f"[MCP] ❌ Failed to unload plugin: {plugin_name}")
            
    except Exception as e:
        click.echo(f"[MCP] Error unloading plugin: {e}")

@cli.command()
@click.argument('plugin_name')
def enable_plugin(plugin_name):
    """Enable a plugin."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        success = manager.enable_plugin(plugin_name)
        if success:
            click.echo(f"[MCP] ✅ Plugin enabled: {plugin_name}")
        else:
            click.echo(f"[MCP] ❌ Failed to enable plugin: {plugin_name}")
            
    except Exception as e:
        click.echo(f"[MCP] Error enabling plugin: {e}")

@cli.command()
@click.argument('plugin_name')
def disable_plugin(plugin_name):
    """Disable a plugin."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        success = manager.disable_plugin(plugin_name)
        if success:
            click.echo(f"[MCP] ✅ Plugin disabled: {plugin_name}")
        else:
            click.echo(f"[MCP] ❌ Failed to disable plugin: {plugin_name}")
            
    except Exception as e:
        click.echo(f"[MCP] Error disabling plugin: {e}")

@cli.command()
@click.argument('plugin_name')
@click.option('--template-type', default='basic', help='Template type (basic/advanced)')
def create_plugin_template(plugin_name, template_type):
    """Create a new plugin template."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        success = manager.create_plugin_template(plugin_name, template_type)
        if success:
            click.echo(f"[MCP] ✅ Plugin template created: {plugin_name}")
            click.echo(f"[MCP] Template location: plugins/{plugin_name}/")
            click.echo(f"[MCP] Edit the files and use 'load-plugin {plugin_name}' to test")
        else:
            click.echo(f"[MCP] ❌ Failed to create plugin template: {plugin_name}")
            
    except Exception as e:
        click.echo(f"[MCP] Error creating plugin template: {e}")

@cli.command()
@click.argument('plugin_path')
def install_plugin(plugin_path):
    """Install a plugin from a path or URL."""
    try:
        from .plugin_system import PluginManager
        manager = PluginManager()
        
        success = manager.install_plugin(plugin_path)
        if success:
            click.echo(f"[MCP] ✅ Plugin installed successfully from: {plugin_path}")
        else:
            click.echo(f"[MCP] ❌ Failed to install plugin from: {plugin_path}")
            
    except Exception as e:
        click.echo(f"[MCP] Error installing plugin: {e}")

@cli.command()
@click.argument('plugin_name')
@click.argument('method_name')
@click.option('--params', default='{}', help='Method parameters as JSON')
def call_plugin_method(plugin_name, method_name, params):
    """Call a method on a loaded plugin."""
    try:
        from .plugin_system import PluginManager
        import json
        
        manager = PluginManager()
        
        # Parse parameters
        try:
            method_params = json.loads(params)
        except json.JSONDecodeError:
            click.echo("[MCP] ❌ Invalid JSON parameters")
            return
        
        # Call plugin method
        result = manager.call_plugin_method(plugin_name, method_name, **method_params)
        
        if asyncio.iscoroutine(result):
            # Handle async methods
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(result)
        
        click.echo(f"[MCP] ✅ Method call result:")
        click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"[MCP] Error calling plugin method: {e}")

@cli.command()
def plugin_marketplace():
    """Search and browse the plugin marketplace. (Experimental, see idea.txt)"""
    try:
        from .plugin_system import PluginMarketplace
        click.echo("[MCP] Plugin Marketplace")
        click.echo("=" * 30)
        click.echo("This feature is under development. See idea.txt for future plans and research.")
        click.echo("  - Create plugins using 'create-plugin-template'")
        click.echo("  - Install plugins using 'install-plugin'")
        click.echo("  - Browse available plugins using 'list-plugins'")
    except Exception as e:
        click.echo(f"[MCP] Error accessing marketplace: {e}")

@cli.command()
@click.option('--project-root', default='.', help='Project root directory to index docstrings from')
def index_docstrings(project_root):
    """Extract and index all docstrings in the project for semantic search."""
    from .rag_system import RAGSystem
    rag = RAGSystem()
    rag.index_all_docstrings(project_root)
    click.echo(f"[MCP] Indexed all docstrings from {project_root} for semantic search.")

@cli.command()
@click.option('--query', prompt='Search query', help='Query for code/documentation search')
@click.option('--mode', default='both', type=click.Choice(['regex', 'semantic', 'both']), help='Search mode: regex, semantic, or both')
@click.option('--max-results', default=10, type=int, help='Maximum results to return')
def search_docs(query, mode, max_results):
    """Unified code/documentation search (regex and/or semantic)."""
    from .regex_search import RegexSearchEngine, SearchQuery, SearchType, SearchScope
    from .rag_system import RAGSystem, RAGQuery
    results = []
    if mode in ('regex', 'both'):
        search_engine = RegexSearchEngine()
        search_query = SearchQuery(
            pattern=query,
            search_type=SearchType.FILE_SYSTEM,
            scope=SearchScope.CURRENT_PROJECT,
            case_sensitive=False,
            max_results=max_results,
            context_lines=2,
            file_patterns=["*.py", "*.md", "*.txt", "*.rst", "*.js", "*.ts"],
            exclude_patterns=None
        )
        regex_results = search_engine.search(search_query)
        results.extend([
            {
                'type': 'regex',
                'file': r.file_path,
                'line': r.line_number,
                'match': r.match_text,
                'context': r.content
            } for r in regex_results
        ])
    if mode in ('semantic', 'both'):
        rag = RAGSystem()
        rag_query = RAGQuery(
            query=query,
            context={},
            max_tokens=1000,
            chunk_types=["docstring", "code", "document"],
            project_id=None,
            user_id=None
        )
        rag_result = rag.retrieve_context(rag_query)
        for chunk in rag_result.chunks[:max_results]:
            results.append({
                'type': 'semantic',
                'content': chunk.content,
                'file': chunk.metadata.get('file', ''),
                'line': chunk.metadata.get('line', ''),
                'score': chunk.relevance_score
            })
    click.echo(f"[MCP] Search results for '{query}':")
    for idx, r in enumerate(results):
        click.echo(f"[{idx+1}] {r}")

@cli.command()
@click.option('--lines', default=20, help='Number of lines to show from the end of the log file')
@click.option('--follow', is_flag=True, help='Follow the log file (like tail -f)')
@click.option('--level', default=None, type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Filter by log level')
def tail_logs(lines, follow, level):
    """Tail the MCP log file (mcp.log in project root)."""
    log_path = os.path.join(os.getcwd(), 'mcp.log')
    if not os.path.exists(log_path):
        click.echo('[MCP] No log file found at mcp.log. If logging to file is not enabled, please enable it in the server configuration.')
        return
    def filter_lines(log_lines):
        if not level:
            return log_lines
        prefix = f"{level}:"
        return [l for l in log_lines if f"{level}" in l]
    with open(log_path, 'r') as f:
        f.seek(0, os.SEEK_END)
        filesize = f.tell()
        f.seek(max(filesize - 8192, 0), os.SEEK_SET)
        lines_list = f.readlines()
        lines_list = filter_lines(lines_list)
        click.echo(''.join(lines_list[-lines:]))
        if follow:
            try:
                while True:
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        f.seek(where)
                    else:
                        if not level or level in line:
                            click.echo(line, nl=False)
            except KeyboardInterrupt:
                click.echo('\n[MCP] Log tailing stopped.')

@cli.command()
def check_updates():
    """Check for available updates."""
    try:
        update_system = AutomaticUpdateSystem()
        status, version_info = update_system.check_for_updates()
        
        if status == UpdateStatus.UPDATE_AVAILABLE and version_info:
            click.echo(f"[MCP] Update available!")
            click.echo(f"[MCP] Current version: {update_system.current_version}")
            click.echo(f"[MCP] New version: {version_info.version}")
            click.echo(f"[MCP] Release date: {version_info.release_date.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"[MCP] Size: {version_info.size} bytes")
            click.echo(f"[MCP] Critical: {'Yes' if version_info.critical else 'No'}")
            click.echo(f"[MCP] Auto-update: {'Yes' if version_info.auto_update else 'No'}")
            click.echo(f"[MCP] Changelog: {version_info.changelog}")
            click.echo(f"\n[MCP] Run 'perform-update' to install the update")
        elif status == UpdateStatus.UP_TO_DATE:
            click.echo(f"[MCP] System is up to date (version {update_system.current_version})")
        else:
            click.echo(f"[MCP] Failed to check for updates: {status.value}")
    except Exception as e:
        click.echo(f"[MCP] Error checking for updates: {str(e)}")

@cli.command()
def perform_update():
    """Perform system update."""
    try:
        update_system = AutomaticUpdateSystem()
        
        # Check if update is available
        status, version_info = update_system.check_for_updates()
        if status != UpdateStatus.UPDATE_AVAILABLE or not version_info:
            click.echo("[MCP] No update available")
            return
        
        click.echo(f"[MCP] Starting update to version {version_info.version}...")
        click.echo("[MCP] This may take a few minutes...")
        
        # Perform update
        success = asyncio.run(update_system.perform_update(version_info))
        
        if success:
            click.echo(f"[MCP] Successfully updated to version {version_info.version}")
            click.echo(f"[MCP] Previous version: {update_system.current_version}")
        else:
            click.echo("[MCP] Update failed")
    except Exception as e:
        click.echo(f"[MCP] Error performing update: {str(e)}")

@cli.command()
def update_status():
    """Show update system status."""
    try:
        update_system = AutomaticUpdateSystem()
        status = update_system.get_status()
        
        click.echo("[MCP] Update System Status:")
        click.echo("=" * 30)
        click.echo(f"Current version: {status['current_version']}")
        click.echo(f"Latest version: {status['latest_version'] or 'Unknown'}")
        click.echo(f"Update status: {status['update_status']}")
        click.echo(f"Auto-update enabled: {'Yes' if status['auto_update_enabled'] else 'No'}")
        click.echo(f"System running: {'Yes' if status['running'] else 'No'}")
        
        if status['last_check_time']:
            click.echo(f"Last check: {status['last_check_time']}")
        
        if status['update_history']:
            click.echo(f"\nRecent updates ({len(status['update_history'])}):")
            for update in status['update_history'][-3:]:  # Show last 3
                click.echo(f"  {update['version']} ({update['timestamp'][:10]})")
    except Exception as e:
        click.echo(f"[MCP] Error getting update status: {str(e)}")

@cli.command()
def enable_auto_update():
    """Enable automatic updates."""
    try:
        update_system = AutomaticUpdateSystem()
        update_system.config.auto_update_enabled = True
        click.echo("[MCP] Automatic updates enabled")
    except Exception as e:
        click.echo(f"[MCP] Error enabling automatic updates: {str(e)}")

@cli.command()
def disable_auto_update():
    """Disable automatic updates."""
    try:
        update_system = AutomaticUpdateSystem()
        update_system.config.auto_update_enabled = False
        click.echo("[MCP] Automatic updates disabled")
    except Exception as e:
        click.echo(f"[MCP] Error disabling automatic updates: {str(e)}")

@cli.command()
def start_update_service():
    """Start the automatic update service."""
    try:
        update_system = AutomaticUpdateSystem()
        update_system.start()
        click.echo("[MCP] Automatic update service started")
        click.echo("[MCP] The service will check for updates every 24 hours")
    except Exception as e:
        click.echo(f"[MCP] Error starting update service: {str(e)}")

@cli.command()
def stop_update_service():
    """Stop the automatic update service."""
    try:
        update_system = AutomaticUpdateSystem()
        update_system.stop()
        click.echo("[MCP] Automatic update service stopped")
    except Exception as e:
        click.echo(f"[MCP] Error stopping update service: {str(e)}")

if __name__ == '__main__':
    cli() 