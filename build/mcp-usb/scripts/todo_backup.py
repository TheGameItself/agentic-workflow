#!/usr/bin/env python3
import os
import re
from datetime import datetime

TODO_SRC = "TODO_DEVELOPMENT_PLAN.md"
TODO_TXT = "todo.txt"
LOG_FILE = "todo_backup.log"

STATUS_MAP = {
    "pending": "",
    "in_progress": "(A)",
    "completed": "x",
    "cancelled": "(C)"
}

def parse_todo_md(md_path):
    tasks = []
    with open(md_path, "r") as f:
        for line in f:
            m = re.match(r"- \[( |x)\] (.+)", line)
            if m:
                status = "completed" if m.group(1) == "x" else "pending"
                content = m.group(2).strip()
                tasks.append({"content": content, "status": status})
            elif line.strip().startswith("- "):
                content = line.strip()[2:]
                tasks.append({"content": content, "status": "pending"})
    return tasks

def refine_tasks(tasks):
    seen = set()
    refined = []
    for t in tasks:
        c = t["content"].strip()
        if c and c not in seen:
            # Speculative improvement: enforce actionable phrasing
            if not c[0].isupper():
                c = c.capitalize()
            if not c.endswith('.'):
                c += '.'
            refined.append({"content": c, "status": t["status"]})
            seen.add(c)
    return refined

def write_todo_txt(tasks, out_path):
    with open(out_path, "w") as f:
        for t in tasks:
            status = STATUS_MAP.get(t["status"], "")
            line = f"{status} {t['content']}"
            f.write(line.strip() + "\n")

def log_backup(tasks, log_path):
    with open(log_path, "a") as f:
        f.write(f"Backup at {datetime.now().isoformat()}\n")
        for t in tasks:
            f.write(f"- {t['status']}: {t['content']}\n")
        f.write("\n")

def main():
    if not os.path.exists(TODO_SRC):
        print(f"Source TODO file not found: {TODO_SRC}")
        return
    tasks = parse_todo_md(TODO_SRC)
    refined = refine_tasks(tasks)
    write_todo_txt(refined, TODO_TXT)
    log_backup(refined, LOG_FILE)
    print(f"Exported {len(refined)} tasks to {TODO_TXT}.")
    print(f"Backup log written to {LOG_FILE}.")

if __name__ == "__main__":
    main() 