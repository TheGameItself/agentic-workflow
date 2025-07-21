# MCP Frontend-Backend Integration

This document specifies the integration points and API endpoints for the MCP minimalist web UI to interact with the backend server.

## 1. Config File Management
- **Endpoint:** `/api/config`
- **Methods:**
  - `GET`: Fetch current config (config.cfg)
  - `PUT`: Update config (send new config content)
- **Example:**
  - `GET /api/config` → `{ "content": "...cfg file contents..." }`
  - `PUT /api/config` with `{ "content": "...new cfg..." }`

## 2. Engine/Lobe Status & Control
- **Endpoint:** `/api/engines`
- **Methods:**
  - `GET`: List all engines/lobes and their status
  - `POST`: Control an engine (start/stop/reload, etc.)
- **Example:**
  - `GET /api/engines` → `[ { "name": "MemoryLobe", "status": "active" }, ... ]`
  - `POST /api/engines` with `{ "name": "MemoryLobe", "action": "restart" }`

## 3. Knowledgebase Queries
- **Endpoint:** `/api/knowledgebase`
- **Methods:**
  - `GET`: List/search knowledgebase entries
  - `POST`: Add new entry
- **Example:**
  - `GET /api/knowledgebase?query=task` → `[ { "id": 1, "type": "task", ... }, ... ]`
  - `POST /api/knowledgebase` with `{ "type": "note", "content": "..." }`

## 4. Task/Workflow Management
- **Endpoint:** `/api/tasks`
- **Methods:**
  - `GET`: List all tasks
  - `POST`: Create new task
  - `PUT`: Update task status
- **Example:**
  - `GET /api/tasks` → `[ { "id": 1, "status": "pending", ... }, ... ]`
  - `PUT /api/tasks/1` with `{ "status": "completed" }`

## 5. Feedback & Reporting
- **Endpoint:** `/api/feedback`
- **Methods:**
  - `POST`: Submit user/LLM feedback
  - `GET`: Fetch periodic reports
- **Example:**
  - `POST /api/feedback` with `{ "type": "user", "content": "..." }`
  - `GET /api/feedback/reports` → `[ { "date": "2024-07-12", "metrics": { ... } }, ... ]`

## Scope Management Components & API Integration

### TaskTree
- **Purpose:** Display and manage the hierarchical task tree, including meta and partial tasks.
- **Backend Endpoints:**
  - `GET /api/tasks` — Fetch all tasks (with parent/child relationships, meta/partial info)
  - `PUT /api/tasks/:id` — Update a task (title, status, progress, etc.)
- **Example:**
  - `GET /api/tasks` → `[ { "id": 1, "title": "Meta Task", "status": "in_progress", "isMeta": true, "progress": 50, "children": [...] }, ... ]`
  - `PUT /api/tasks/1` with `{ "title": "Updated", "status": "partial", "progress": 60 }`

### MetaTaskEditor
- **Purpose:** Edit meta/partial tasks (title, status, progress, notes).
- **Backend Endpoints:**
  - `PUT /api/tasks/:id` — Update task details
  - `POST /api/tasks` — Create new task
- **Example:**
  - `PUT /api/tasks/2` with `{ "title": "Subtask", "status": "completed", "progress": 100 }`

### ProgressBar
- **Purpose:** Visualize progress for tasks and workflows.
- **Data:**
  - Receives a `percentage` prop from the parent (from task/workflow data).

---

**Note:**
- All endpoints accept/return JSON.
- The frontend will map backend data to the `TaskNode` interface for display and editing.
- See backend docs for full API details and data model.

---

- All endpoints accept/return JSON.
- For advanced workflows, JSON-RPC may be used (see backend docs).
- Update this document as endpoints are implemented. 