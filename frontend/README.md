# MCP Minimalist Dark-Themed Web UI
## λ:frontend_overview(minimalist_dark_ui_system)

---

## β:nodejs_icu_error_blocker(system_dependency_issue)

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

## λ:system_overview(interactive_management_interface)
This frontend is a minimalist, dark-themed web UI for the MCP server. It allows users to interactively and dynamically manage:
- **i:config_management(centralized_configuration)**: The centralized config file (`config/config.cfg`)
- **Ω:engine_orchestration(lobe_management)**: All MCP engines/lobes (memory, workflow, project, task, context, reminder, RAG, etc.)
- **ℵ:knowledgebase_interface(data_management)**: The knowledgebase (memories, engrams, tasks, feedback, etc.)

## τ:technology_stack(modern_web_framework)
- **[Vite](https://vitejs.dev/)**: React + TypeScript template
- **[React](https://react.dev/)**: 18+
- **[Tailwind CSS](https://tailwindcss.com/)**: 3+
- **[PostCSS](https://postcss.org/)**, **[Autoprefixer](https://github.com/postcss/autoprefixer)**: CSS processing

## λ:feature_matrix(core_capabilities)
- **λ:ui_design(minimalist_dark_theme)**: Minimalist, dark-themed UI
- **i:config_editor(cfg_management)**: Config editor for `config.cfg`
- **β:engine_controls(status_monitoring)**: Engine/lobe status and controls
- **ℵ:knowledgebase_browser(search_interface)**: Knowledgebase browser and search
- **Δ:extensibility(future_features)**: Extensible for future features (AB test, split-brain, feedback, etc.)

## Δ:integration_points(backend_communication)
- **τ:api_communication(endpoint_interface)**: Communicates with MCP backend via API endpoints (to be documented)
- **i:config_operations(read_write_trigger)**: Reads/writes config, triggers workflows, queries knowledgebase

## Ω:setup_sequence(nodejs_dependency_resolution)
1. **τ:runtime_installation(nodejs_v18_plus)**: Install Node.js (v18+ recommended)
2. **λ:directory_navigation(frontend_context)**: `cd frontend`
3. **ℵ:dependency_installation(npm_packages)**: `npm install`
4. **β:development_server(local_testing)**: `npm run dev` to start the dev server
5. **τ:browser_access(localhost_5173)**: Open [http://localhost:5173](http://localhost:5173) in your browser

## i:inspiration_sources(reference_implementations)
- **[vite-scaffold-2023](https://github.com/jblossomweb/vite-scaffold-2023)**: Vite scaffolding reference
- **[react-ui-scaffold](https://github.com/cion-studio/react-ui-scaffold)**: React UI patterns

## Δ:scope_management_components(task_orchestration)

### ℵ:task_tree(hierarchical_task_display)
- **λ:display_functionality(tree_visualization)**: Displays a hierarchical tree of tasks, including meta and partial tasks
- **i:interaction_capabilities(selection_editing)**: Allows selection and editing of tasks
- **τ:backend_integration(api_connectivity)**: Will connect to backend API endpoints for fetching/updating the task tree

### i:meta_task_editor(task_modification_interface)
- **λ:form_interface(task_editing)**: Form for editing meta/partial tasks (title, status, progress, notes)
- **Δ:dual_purpose(create_update)**: Used for both creating and updating tasks
- **τ:persistence_layer(backend_api_integration)**: Will connect to backend API endpoints for saving changes

### β:progress_bar(visual_progress_indicator)
- **λ:visual_design(minimalist_dark_theme)**: Minimalist dark-themed progress bar for visualizing task or workflow progress
- **Δ:usage_contexts(multi_component_integration)**: Used in task lists, editors, and dashboards

### Ω:component_integration(backend_api_alignment)
- **τ:api_compatibility(mcp_backend_integration)**: These components are designed to work with the MCP backend's task and workflow APIs (see INTEGRATION.md)
- **i:future_development(example_usage_api_integration)**: Example usage and API integration will be added once the backend endpoints are connected

---

## τ:development_status(active_development_phase)
> This UI is under active development. Integration with MCP backend APIs is in progress.

## τ:self_reference(frontend_readme_metadata)
{type:Documentation, file:"frontend/README.md", version:"1.0.0", checksum:"sha256:frontend_readme_checksum", canonical_address:"frontend-readme", pfsus_compliant:true, lambda_operators:true, file_format:"readme.frontend.v1.0.0.md"}

@{visual-meta-start}
author = {MCP Core Team},
title = {MCP Minimalist Dark-Themed Web UI},
version = {1.0.0},
file_format = {readme.frontend.v1.0.0.md},
structure = { overview, error_blocker, system_overview, technology_stack, features, integration_points, setup, inspiration, components },
file_naming_standards = {pfsus_compliant, lambda_operators},
@{visual-meta-end}

%% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-22T18:56:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=readme.frontend.v1.0.0.md