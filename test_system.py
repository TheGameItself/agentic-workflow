from src.mcp.memory import MemoryManager
from src.mcp.workflow import WorkflowManager
from src.mcp.task_manager import TaskManager
from src.mcp.context_manager import ContextManager
from src.mcp.server import MCPServer
from src.mcp.experimental_lobes import test_experimental_lobes

# --- MemoryManager Tests ---


def test_add_and_get_memory():
    memory_manager = MemoryManager()
    mem_id = memory_manager.add_memory(text="test", memory_type="test", priority=1)
    mem = memory_manager.get_memory(mem_id)
    assert mem is not None
    assert mem.get("text") == "test"
    assert mem.get("memory_type") == "test"

def test_memory_param():
    memory_manager = MemoryManager()
    for text, memory_type, priority in [
        ("alpha", "type1", 1),
        ("beta", "type2", 2),
        ("gamma", "type3", 3),
    ]:
        mem_id = memory_manager.add_memory(text=text, memory_type=memory_type, priority=priority)
        mem = memory_manager.get_memory(mem_id)
        assert mem is not None
        assert mem.get("text") == text
        assert mem.get("memory_type") == memory_type
        assert mem.get("priority") == priority


# --- WorkflowManager Tests ---


def test_create_and_modify_workflow():
    workflow_manager = WorkflowManager()
    workflow_id = workflow_manager.create_workflow("proj", "/tmp/proj")
    assert isinstance(workflow_id, int)
    assert workflow_manager.modify_step("init", {"description": "Init step"})
    from src.mcp.workflow import WorkflowStep
    new_step = WorkflowStep(name="custom", description="Custom step")
    workflow_manager.register_step("custom", new_step)
    assert "custom" in workflow_manager.steps
    assert workflow_manager.modify_step("custom", {"description": "Updated custom step"})
    assert workflow_manager.steps["custom"].description == "Updated custom step"
    assert workflow_manager.remove_step("custom")
    assert "custom" not in workflow_manager.steps


def test_workflow_step_lifecycle():
    workflow_manager = WorkflowManager()
    assert workflow_manager.start_step("init")
    assert workflow_manager.steps["init"].status.name == "IN_PROGRESS"
    assert workflow_manager.complete_step("init")
    assert workflow_manager.steps["init"].status.name == "COMPLETED"
    workflow_manager.add_step_feedback("init", "Initial feedback", impact=1, principle="test-principle")
    feedbacks = workflow_manager.steps["init"].feedback
    assert any(fb['text'] == "Initial feedback" for fb in feedbacks)


def test_workflow_dependencies_and_next_steps():
    workflow_manager = WorkflowManager()
    from src.mcp.workflow import WorkflowStep, WorkflowStatus
    # Reset completed_steps to ensure clean state
    workflow_manager.completed_steps = []
    # Explicitly register 'init' step
    init_step = WorkflowStep(name="init", description="Initialization step")
    workflow_manager.register_step("init", init_step)
    dep_step = WorkflowStep(name="dep", description="Dependent step", dependencies=["init"])
    workflow_manager.register_step("dep", dep_step)
    # Reset all step statuses to NOT_STARTED
    for step in workflow_manager.steps.values():
        step.status = WorkflowStatus.NOT_STARTED
    assert not workflow_manager.start_step("dep"), f"dep should not start before init is complete, got: {workflow_manager.steps['dep'].status}"
    workflow_manager.start_step("init")
    workflow_manager.complete_step("init")
    assert workflow_manager.start_step("dep"), f"dep should start after init is complete, got: {workflow_manager.steps['dep'].status}"
    workflow_manager.set_next_steps("dep", ["testing"])
    next_steps = workflow_manager.get_next_steps("dep")
    print(f"[DEBUG] next_steps for 'dep': {next_steps}")
    assert "testing" in next_steps, f"'testing' not in next_steps: {next_steps}"
    assert workflow_manager.remove_next_step("dep", "testing"), f"Failed to remove 'testing' from next_steps: {next_steps}"
    next_steps_after = workflow_manager.get_next_steps("dep")
    print(f"[DEBUG] next_steps after removal for 'dep': {next_steps_after}")
    assert "testing" not in next_steps_after, f"'testing' still in next_steps after removal: {next_steps_after}"

def test_workflow_meta_partial_steps():
    workflow_manager = WorkflowManager()
    from src.mcp.workflow import WorkflowStep
    meta_step = WorkflowStep(name="meta", description="Meta step", is_meta=True)
    workflow_manager.register_step("meta", meta_step)
    assert workflow_manager.steps["meta"].is_meta
    workflow_manager.steps["meta"].set_partial_progress(0.5)
    assert workflow_manager.steps["meta"].get_partial_progress() == 0.5


def test_task_creation_and_dependencies():
    task_manager = TaskManager()
    t1 = task_manager.create_task(title="Task 1", description="desc1", priority=5)
    t2 = task_manager.create_task(title="Task 2", description="desc2", priority=8, parent_id=t1)
    dep_id = task_manager.add_task_dependency(t2, t1)
    assert isinstance(dep_id, int)
    tree = task_manager.get_task_tree(root_task_id=t1)
    assert str(t1) in [str(task['id']) for task in tree.get('root_tasks', [])] or t1 in tree.get('all_tasks', {})
    deps = task_manager.get_task_dependencies(t2)
    assert any(d.get('depends_on_task_id') == t1 or d.get('task_id') == t1 for d in deps)


def test_task_progress_and_notes():
    task_manager = TaskManager()
    t1 = task_manager.create_task(title="Task Progress", description="desc", priority=5)
    assert task_manager.update_task_progress(t1, 50.0, current_step="Halfway", partial_completion_notes="50% done")
    progress = task_manager.get_task_progress(t1)
    assert progress['progress_percentage'] == 50.0
    note_id = task_manager.add_task_note(t1, "Important note", line_number=42, file_path="/tmp/file.py")
    assert isinstance(note_id, int)
    notes = task_manager.get_task_notes(t1)
    assert any(n['note_text'] == "Important note" for n in notes)


def test_task_feedback_and_flags():
    task_manager = TaskManager()
    t1 = task_manager.create_task(title="Feedback Task", description="desc", priority=5)
    fb_id = task_manager.add_task_feedback(t1, "Needs improvement", lesson_learned="Lesson", principle="Principle", impact_score=-2)
    assert isinstance(fb_id, int)
    flags = task_manager.get_task_suggestions_and_flags(t1)
    assert 'flags' in flags
    task_manager.add_task_feedback(t1, "Great job", impact_score=3)
    flags2 = task_manager.get_task_suggestions_and_flags(t1)
    assert 'suggestions' in flags2


def test_task_meta_and_tags():
    task_manager = TaskManager()
    t1 = task_manager.create_task(title="Meta Task", description="desc", priority=5, is_meta=True, meta_type="review", tags=["urgent", "review"])
    tree = task_manager.get_task_tree(root_task_id=t1)
    root_task = tree.get('all_tasks', {}).get(t1)
    if root_task is None:
        root_task = tree.get('all_tasks', {}).get(str(t1))
    assert root_task is not None
    assert root_task.get('is_meta', False)
    tags = task_manager.get_tags(task_id=t1)
    assert any(tag['tag_name'] == "urgent" for tag in tags)
    tag_id = task_manager.create_tag("blocker", tag_type="critical")
    assert isinstance(tag_id, int)
    assert task_manager.update_tag(tag_id, tag_name="blocker-updated")
    tag = task_manager.get_tag(tag_id)
    assert tag['tag_name'] == "blocker-updated"


def test_create_and_get_task():
    task_manager = TaskManager()
    t1 = task_manager.create_task(title="Test Task", description="desc", priority=5)
    tree = task_manager.get_task_tree(root_task_id=t1)
    root_task = tree.get('all_tasks', {}).get(t1)
    if root_task is None:
        root_task = tree.get('all_tasks', {}).get(str(t1))
    assert root_task is not None, "Root task not found in task tree"
    assert root_task.get("title") == "Test Task"
    assert root_task.get("priority") == 5


def test_export_context():
    context_manager = ContextManager()
    ctx = context_manager.export_context()
    print(f"[DEBUG] export_context returned: {ctx}")
    assert isinstance(ctx, dict)
    assert "tasks" in ctx or "memories" in ctx, f"Context missing 'tasks' or 'memories': {ctx.keys()}"

def test_server_status():
    mcp_server = MCPServer()
    status = mcp_server.get_status()
    assert isinstance(status, dict)
    assert "is_running" in status
    assert "task_count" in status


def test_memory_empty_text():
    memory_manager = MemoryManager()
    mem_id = memory_manager.add_memory(text="", memory_type="empty", priority=0)
    mem = memory_manager.get_memory(mem_id)
    assert mem is not None
    assert mem.get("text") == ""
    assert mem.get("memory_type") == "empty"


def test_task_priority_range():
    task_manager = TaskManager()
    for priority in [-1, 0, 100]:
        t1 = task_manager.create_task(title="Edge Task", description="desc", priority=priority)
        tree = task_manager.get_task_tree(root_task_id=t1)
        root_task = tree.get('all_tasks', {}).get(t1)
        if root_task is None:
            root_task = tree.get('all_tasks', {}).get(str(t1))
        assert root_task is not None, "Root task not found in task tree"
        assert root_task.get("priority") == priority


def test_experimental_lobes_integration():
    try:
        test_experimental_lobes()
    except Exception as e:
        assert False, f"Experimental lobes test failed: {e}"

if __name__ == "__main__":
    tests = [
        test_add_and_get_memory,
        test_memory_param,
        test_create_and_modify_workflow,
        test_workflow_step_lifecycle,
        test_workflow_dependencies_and_next_steps,
        test_workflow_meta_partial_steps,
        test_task_creation_and_dependencies,
        test_task_progress_and_notes,
        test_task_feedback_and_flags,
        test_task_meta_and_tags,
        test_create_and_get_task,
        test_export_context,
        test_server_status,
        test_memory_empty_text,
        test_task_priority_range,
        test_experimental_lobes_integration,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} tests failed.") 