#!/usr/bin/env python3
"""
Progress Tracking System for MCP Server
Provides visual progress tracking and status monitoring.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class ProgressTracker:
    """Track and display progress for various operations."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.progress_file = self.data_dir / "progress.json"
        self.load_progress()
        
    def load_progress(self):
        """Load existing progress data."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.progress = {}
        else:
            self.progress = {}
            
    def save_progress(self):
        """Save progress data."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save progress: {e}")
            
    def start_task(self, task_id: str, description: str, total_steps: int = 1):
        """Start tracking a new task."""
        self.progress[task_id] = {
            "description": description,
            "total_steps": total_steps,
            "current_step": 0,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "completed_steps": []
        }
        self.save_progress()
        
    def update_task(self, task_id: str, step: int, step_description: str = ""):
        """Update task progress."""
        if task_id not in self.progress:
            return
            
        self.progress[task_id]["current_step"] = step
        if step_description:
            self.progress[task_id]["completed_steps"].append({
                "step": step,
                "description": step_description,
                "timestamp": datetime.now().isoformat()
            })
        self.save_progress()
        
    def complete_task(self, task_id: str, success: bool = True):
        """Mark task as completed."""
        if task_id not in self.progress:
            return
            
        self.progress[task_id]["status"] = "completed" if success else "failed"
        self.progress[task_id]["end_time"] = datetime.now().isoformat()
        self.progress[task_id]["current_step"] = self.progress[task_id]["total_steps"]
        self.save_progress()
        
    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for a specific task."""
        return self.progress.get(task_id)
        
    def get_all_progress(self) -> Dict[str, Any]:
        """Get all progress data."""
        return self.progress
        
    def display_progress(self, task_id: Optional[str] = None):
        """Display progress for tasks."""
        if task_id:
            self._display_single_task(task_id)
        else:
            self._display_all_tasks()
            
    def _display_single_task(self, task_id: str):
        """Display progress for a single task."""
        task = self.get_task_progress(task_id)
        if not task:
            print(f"Task '{task_id}' not found.")
            return
            
        print(f"\n📊 Task: {task['description']}")
        print(f"Status: {task['status'].upper()}")
        
        if task['total_steps'] > 1:
            progress = (task['current_step'] / task['total_steps']) * 100
            bar_length = 30
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"Progress: [{bar}] {progress:.1f}% ({task['current_step']}/{task['total_steps']})")
        else:
            print(f"Progress: {task['current_step']}/{task['total_steps']}")
            
        if task['completed_steps']:
            print("\nCompleted steps:")
            for step in task['completed_steps'][-5:]:  # Show last 5 steps
                print(f"  ✓ {step['description']}")
                
    def _display_all_tasks(self):
        """Display progress for all tasks."""
        if not self.progress:
            print("No tasks tracked yet.")
            return
            
        print("\n📊 Progress Summary")
        print("=" * 50)
        
        for task_id, task in self.progress.items():
            status_icon = {
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(task['status'], "❓")
            
            print(f"{status_icon} {task['description']}")
            print(f"   Status: {task['status']}")
            
            if task['total_steps'] > 1:
                progress = (task['current_step'] / task['total_steps']) * 100
                print(f"   Progress: {progress:.1f}% ({task['current_step']}/{task['total_steps']})")
                
            if 'start_time' in task:
                start_time = datetime.fromisoformat(task['start_time'])
                print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            if 'end_time' in task:
                end_time = datetime.fromisoformat(task['end_time'])
                duration = end_time - start_time
                print(f"   Duration: {duration}")
                
            print()
            
    def clear_completed_tasks(self, older_than_days: int = 7):
        """Clear completed tasks older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        tasks_to_remove = []
        
        for task_id, task in self.progress.items():
            if task['status'] in ['completed', 'failed']:
                if 'end_time' in task:
                    end_time = datetime.fromisoformat(task['end_time'])
                    if end_time < cutoff_date:
                        tasks_to_remove.append(task_id)
                        
        for task_id in tasks_to_remove:
            del self.progress[task_id]
            
        if tasks_to_remove:
            self.save_progress()
            print(f"Cleared {len(tasks_to_remove)} old completed tasks.")
        else:
            print("No old completed tasks to clear.")


class ProgressBar:
    """Simple progress bar for CLI operations."""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._display()
        
    def _display(self):
        """Display the progress bar."""
        if self.total == 0:
            return
            
        progress = min(self.current / self.total, 1.0)
        filled_length = int(self.width * progress)
        bar = '█' * filled_length + '░' * (self.width - filled_length)
        
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
            eta_str = f"ETA: {timedelta(seconds=int(eta))}"
        else:
            eta_str = "ETA: --:--:--"
            
        print(f"\r{self.description} [{bar}] {progress*100:.1f}% ({self.current}/{self.total}) {eta_str}", end='', flush=True)
        
    def finish(self):
        """Finish the progress bar."""
        self.current = self.total
        self._display()
        print()  # New line after progress bar


def main():
    """CLI interface for progress tracking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progress Tracking System")
    parser.add_argument("command", choices=["show", "clear", "task"], help="Command to execute")
    parser.add_argument("--task-id", help="Task ID for specific operations")
    parser.add_argument("--days", type=int, default=7, help="Days threshold for clearing old tasks")
    
    args = parser.parse_args()
    
    tracker = ProgressTracker()
    
    if args.command == "show":
        tracker.display_progress(args.task_id)
    elif args.command == "clear":
        tracker.clear_completed_tasks(args.days)
    elif args.command == "task":
        if not args.task_id:
            print("Error: --task-id required for 'task' command")
            sys.exit(1)
        tracker.display_progress(args.task_id)


if __name__ == "__main__":
    main() 