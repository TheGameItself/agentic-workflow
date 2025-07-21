import os
import sys
import sqlite3
from datetime import datetime

# Adjust path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from mcp.task_manager import TaskManager, TaskStatus
except ImportError as e:
    print(f"[ERROR] Could not import TaskManager: {e}")
    sys.exit(1)

def is_automatable(task):
    # Define logic for automatable tasks (stub: all tasks with 'automate' in title or description)
    title = (task.get('title') or '').lower()
    desc = (task.get('description') or '').lower()
    return 'automate' in title or 'automate' in desc or 'auto' in title or 'auto' in desc

def enact_task(task_manager, task):
    task_id = task['id']
    print(f"[ENACT] Task {task_id}: {task['title']} ...", end=' ')
    # Simulate action (replace with real logic as needed)
    # Here, just mark as completed
    task_manager.update_task_progress(task_id, 100.0, current_step='Auto-enacted', partial_completion_notes='Automatically enacted by script.')
    print("DONE")

def main():
    print("[INFO] Enacting all automatable tasks (no user input)...")
    tm = TaskManager()
    tasks = tm.get_tasks(include_completed=False)
    enacted = 0
    skipped = 0
    for task in tasks:
        if task['status'] == TaskStatus.COMPLETED.value:
            continue
        if is_automatable(task):
            enact_task(tm, task)
            enacted += 1
        else:
            print(f"[SKIP] Task {task['id']}: {task['title']} (not automatable)")
            skipped += 1
    print(f"[SUMMARY] Enacted: {enacted}, Skipped: {skipped}, Total: {len(tasks)}")

if __name__ == "__main__":
    main() 