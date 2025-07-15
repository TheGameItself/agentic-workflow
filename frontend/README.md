# MCP Minimalist Dark-Themed Web UI

---

## ⚠️ Node.js ICU Error Blocker (libicui18n.so.75)

**Current Status:**
- The frontend build and dev server are currently **blocked** due to a missing ICU library required by Node.js: `libicui18n.so.75`.
- This is a host system issue and cannot be fixed from within the project directory.
- **No system files or settings will be changed by this project.**

**How to resolve on your system:**
- Install the required ICU version (e.g., `sudo pacman -S icu75` or `sudo apt-get install libicu75`), or
- Use a portable Node.js build from [nodejs.org](https://nodejs.org/en/download/) that bundles ICU, or
- Reinstall Node.js using a version manager (e.g., [nvm](https://github.com/nvm-sh/nvm)) to match your system libraries.
- See [Arch Linux forum](https://bbs.archlinux32.org/viewtopic.php?id=3085) and [Artix Linux forum](https://forum.artixlinux.org/index.php/topic,5634.0.html) for more details.

---

## Overview
This frontend is a minimalist, dark-themed web UI for the MCP server. It allows users to interactively and dynamically manage:
- The centralized config file (`config/config.cfg`)
- All MCP engines/lobes (memory, workflow, project, task, context, reminder, RAG, etc.)
- The knowledgebase (memories, engrams, tasks, feedback, etc.)

## Stack
- [Vite](https://vitejs.dev/) (React + TypeScript template)
- [React](https://react.dev/) 18+
- [Tailwind CSS](https://tailwindcss.com/) 3+
- [PostCSS](https://postcss.org/), [Autoprefixer](https://github.com/postcss/autoprefixer)

## Features
- Minimalist, dark-themed UI
- Config editor for `config.cfg`
- Engine/lobe status and controls
- Knowledgebase browser and search
- Extensible for future features (AB test, split-brain, feedback, etc.)

## Integration Points
- Communicates with MCP backend via API endpoints (to be documented)
- Reads/writes config, triggers workflows, queries knowledgebase

## Setup (when Node.js is fixed)
1. Install Node.js (v18+ recommended)
2. `cd frontend`
3. `npm install`
4. `npm run dev` to start the dev server
5. Open [http://localhost:5173](http://localhost:5173) in your browser

## Inspiration
- [vite-scaffold-2023](https://github.com/jblossomweb/vite-scaffold-2023)
- [react-ui-scaffold](https://github.com/cion-studio/react-ui-scaffold)

## Scope Management Components

### TaskTree
- Displays a hierarchical tree of tasks, including meta and partial tasks.
- Allows selection and editing of tasks.
- Will connect to backend API endpoints for fetching/updating the task tree.

### MetaTaskEditor
- Form for editing meta/partial tasks (title, status, progress, notes).
- Used for both creating and updating tasks.
- Will connect to backend API endpoints for saving changes.

### ProgressBar
- Minimalist dark-themed progress bar for visualizing task or workflow progress.
- Used in task lists, editors, and dashboards.

**Integration:**
- These components are designed to work with the MCP backend's task and workflow APIs (see INTEGRATION.md).
- Example usage and API integration will be added once the backend endpoints are connected.

---

> This UI is under active development. See `