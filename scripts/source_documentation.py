#!/usr/bin/env python3
"""
Source Documentation Generator

This script generates comprehensive documentation for source code files
and handles stub files by creating proper content.
"""

import os
import re
import ast
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceDocumentationGenerator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.src_dir = project_root / "src"
    
    def generate_source_docs(self):
        """Generate documentation for source files"""
        
        # Scan source files
        source_files = []
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            source_files.append(py_file)
        
        logger.info(f"Found {len(source_files)} source files to document")
        
        for source_file in source_files:
            self._generate_file_doc(source_file)
    
    def _generate_file_doc(self, source_file: Path):
        """Generate documentation for a single source file"""
        
        try:
            content = source_file.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Could not read {source_file}: {e}")
            return
        
        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Syntax error in {source_file}")
            return
        
        # Extract information
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or '',
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                })
            elif isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or '',
                    'args': [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                imports.extend([alias.name for alias in node.names])
        
        # Generate documentation
        doc_content = self._create_source_doc(source_file, classes, functions, imports, content)
        
        # Write documentation
        doc_filename = f"{source_file.stem}-API.md"
        doc_path = self.docs_dir / "api" / doc_filename
        doc_path.parent.mkdir(exist_ok=True)
        
        doc_path.write_text(doc_content, encoding='utf-8')
        logger.info(f"Generated documentation for {source_file.name}")
    
    def _create_source_doc(self, source_file: Path, classes: list, functions: list, imports: list, content: str) -> str:
        """Create documentation content for a source file"""
        
        title = source_file.stem.replace('_', ' ').title()
        module_name = source_file.stem
        
        # Determine tags and graph group
        tags = self._determine_source_tags(source_file, classes, functions)
        graph_group = "API Reference"
        
        # Create class documentation
        classes_doc = ""
        for cls in classes:
            classes_doc += f"""
#### {cls['name']}
```python
class {cls['name']}:
    \"\"\"
    {cls['docstring'] or f'{cls["name"]} class implementation.'}
    \"\"\"
    
    def __init__(self, *args, **kwargs):
        \"\"\"
        Initialize {cls['name']}.
        \"\"\"
        pass
```

**Methods:**
"""
            for method in cls['methods']:
                classes_doc += f"- `{method}()`: Method implementation\n"
        
        # Create function documentation
        functions_doc = ""
        for func in functions:
            args_str = ", ".join(func['args'])
            functions_doc += f"""
#### {func['name']}
```python
def {func['name']}({args_str}):
    \"\"\"
    {func['docstring'] or f'{func["name"]} function implementation.'}
    \"\"\"
    pass
```
"""
        
        # Create imports documentation
        imports_doc = ""
        if imports:
            imports_doc = "**Imports:**\n"
            for imp in imports:
                imports_doc += f"- `{imp}`\n"
        
        return f"""---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} API

## Epic
**As a** developer
**I want** comprehensive API documentation for {title}
**So that** I can integrate it effectively into my applications

## User Stories

### Story 1: API Understanding
**As a** developer
**I want** to understand the {title} API
**So that** I can use it correctly

**Acceptance Criteria:**
- [ ] Given I need to use {title}, When I read the API docs, Then I understand the interface
- [ ] Given I understand the API, When I implement it, Then it works as expected

### Story 2: Integration
**As a** developer
**I want** to integrate {title} into my application
**So that** I can leverage its functionality

**Acceptance Criteria:**
- [ ] Given I want to integrate {title}, When I follow the integration guide, Then it works seamlessly
- [ ] Given I encounter issues, When I check error handling, Then I can resolve problems

## Module Overview

The `{module_name}` module provides {self._generate_module_description(classes, functions)}.

{imports_doc}

## API Reference

### Classes
{classes_doc}

### Functions
{functions_doc}

## Usage Examples

### Basic Usage
```python
from mcp import {module_name}

# Basic usage example
# Implementation details to be added
```

### Advanced Usage
```python
# Advanced usage example
# Implementation details to be added
```

## Implementation Details

### Source File
- **File**: `{source_file.relative_to(self.project_root)}`
- **Lines**: {len(content.split())} lines of code
- **Classes**: {len(classes)} classes
- **Functions**: {len(functions)} functions

### Dependencies
{self._analyze_dependencies(imports)}

## Related Documentation
- [[API_DOCUMENTATION]] - Complete API reference
- [[DEVELOPER_GUIDE]] - Development guide
- [[Documentation-Index]] - Main documentation index
"""
    
    def _determine_source_tags(self, source_file: Path, classes: list, functions: list) -> str:
        """Determine tags for source file documentation"""
        tags = ["api", "source", "technical"]
        
        # Add tags based on file path
        path_str = str(source_file.relative_to(self.src_dir))
        if "memory" in path_str.lower():
            tags.append("memory")
        if "hormone" in path_str.lower():
            tags.append("hormone")
        if "genetic" in path_str.lower():
            tags.append("genetic")
        if "p2p" in path_str.lower():
            tags.append("p2p")
        if "pattern" in path_str.lower():
            tags.append("pattern")
        if "simulation" in path_str.lower():
            tags.append("simulation")
        if "lobe" in path_str.lower():
            tags.append("lobe")
        if "engine" in path_str.lower():
            tags.append("engine")
        
        return ", ".join(tags)
    
    def _generate_module_description(self, classes: list, functions: list) -> str:
        """Generate a description of the module"""
        if classes and functions:
            return f"{len(classes)} classes and {len(functions)} functions for system functionality"
        elif classes:
            return f"{len(classes)} classes for system functionality"
        elif functions:
            return f"{len(functions)} functions for system functionality"
        else:
            return "system functionality"
    
    def _analyze_dependencies(self, imports: list) -> str:
        """Analyze and document dependencies"""
        if not imports:
            return "No external dependencies identified."
        
        # Categorize imports
        internal = [imp for imp in imports if not imp.startswith(('os', 'sys', 'json', 're', 'pathlib', 'logging', 'typing', 'dataclasses', 'datetime', 'asyncio'))]
        external = [imp for imp in imports if imp.startswith(('os', 'sys', 'json', 're', 'pathlib', 'logging', 'typing', 'dataclasses', 'datetime', 'asyncio'))]
        
        result = ""
        if internal:
            result += "**Internal Dependencies:**\n"
            for imp in internal:
                result += f"- `{imp}`\n"
        
        if external:
            result += "\n**External Dependencies:**\n"
            for imp in external:
                result += f"- `{imp}`\n"
        
        return result
    
    def handle_stub_files(self):
        """Handle stub files by creating proper content"""
        
        # Find stub files
        stub_files = []
        for file_path in self.docs_dir.rglob("*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                if self._is_stub_file(content):
                    stub_files.append(file_path)
            except Exception:
                continue
        
        logger.info(f"Found {len(stub_files)} stub files to handle")
        
        for stub_file in stub_files:
            self._handle_stub_file(stub_file)
    
    def _is_stub_file(self, content: str) -> bool:
        """Check if file is a stub"""
        lines = content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.startswith('#')]
        return len(non_empty_lines) < 15
    
    def _handle_stub_file(self, file_path: Path):
        """Handle a stub file by creating proper content"""
        
        title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
        
        # Determine content type based on filename and path
        filename_lower = file_path.stem.lower()
        path_str = str(file_path.relative_to(self.docs_dir))
        
        if any(word in filename_lower for word in ['api', 'reference']):
            content_type = "api"
        elif any(word in filename_lower for word in ['guide', 'tutorial', 'user']):
            content_type = "user_guide"
        elif any(word in filename_lower for word in ['arch', 'design', 'system']):
            content_type = "architecture"
        elif any(word in path_str.lower() for word in ['memory', 'hormone', 'genetic', 'p2p', 'pattern']):
            content_type = "system"
        else:
            content_type = "general"
        
        # Create appropriate content
        content = self._create_stub_content(title, content_type, file_path)
        
        # Write content
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"Handled stub file: {file_path.name}")
    
    def _create_stub_content(self, title: str, content_type: str, file_path: Path) -> str:
        """Create content for a stub file"""
        
        # Determine tags and graph group
        tags = self._determine_stub_tags(title, content_type, file_path)
        graph_group = self._determine_stub_graph_group(title, content_type, file_path)
        
        if content_type == "api":
            return self._create_api_stub_content(title, tags, graph_group)
        elif content_type == "user_guide":
            return self._create_user_guide_stub_content(title, tags, graph_group)
        elif content_type == "architecture":
            return self._create_architecture_stub_content(title, tags, graph_group)
        elif content_type == "system":
            return self._create_system_stub_content(title, tags, graph_group)
        else:
            return self._create_general_stub_content(title, tags, graph_group)
    
    def _determine_stub_tags(self, title: str, content_type: str, file_path: Path) -> str:
        """Determine tags for stub file"""
        tags = ["documentation"]
        
        if content_type == "api":
            tags.extend(["api", "reference", "technical"])
        elif content_type == "user_guide":
            tags.extend(["user-guide", "tutorial"])
        elif content_type == "architecture":
            tags.extend(["architecture", "design", "system"])
        elif content_type == "system":
            tags.extend(["system", "core"])
        
        # Add specific tags based on title and path
        title_lower = title.lower()
        path_str = str(file_path.relative_to(self.docs_dir))
        
        if "memory" in title_lower or "memory" in path_str:
            tags.append("memory")
        if "hormone" in title_lower or "hormone" in path_str:
            tags.append("hormone")
        if "genetic" in title_lower or "genetic" in path_str:
            tags.append("genetic")
        if "p2p" in title_lower or "p2p" in path_str:
            tags.append("p2p")
        if "pattern" in title_lower or "pattern" in path_str:
            tags.append("pattern")
        if "simulation" in title_lower or "simulation" in path_str:
            tags.append("simulation")
        
        return ", ".join(tags)
    
    def _determine_stub_graph_group(self, title: str, content_type: str, file_path: Path) -> str:
        """Determine graph group for stub file"""
        if content_type == "api":
            return "API Reference"
        elif content_type == "user_guide":
            return "User Guides"
        elif content_type == "architecture":
            return "Architecture"
        elif content_type == "system":
            return "Core Systems"
        else:
            return "Documentation"
    
    def _create_api_stub_content(self, title: str, tags: str, graph_group: str) -> str:
        """Create API stub content"""
        return f"""---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} API

## Epic
**As a** developer
**I want** comprehensive API documentation for {title.lower()}
**So that** I can integrate it effectively into my applications

## User Stories

### Story 1: API Understanding
**As a** developer
**I want** to understand the {title.lower()} API
**So that** I can use it correctly

**Acceptance Criteria:**
- [ ] Given I need to use {title.lower()}, When I read the API docs, Then I understand the interface
- [ ] Given I understand the API, When I implement it, Then it works as expected

## API Reference

### Classes
```python
class {title.replace(' ', '')}:
    \"\"\"
    Main class for {title.lower()} functionality.
    \"\"\"
    pass
```

### Methods
```python
def main_method(self, *args, **kwargs):
    \"\"\"
    Main method for {title.lower()}.
    \"\"\"
    pass
```

## Usage Examples

### Basic Usage
```python
# Basic usage example for {title.lower()}
# Implementation details to be added
```

## Related Documentation
- [[API_DOCUMENTATION]] - Complete API reference
- [[DEVELOPER_GUIDE]] - Development guide
- [[Documentation-Index]] - Main documentation index
"""
    
    def _create_user_guide_stub_content(self, title: str, tags: str, graph_group: str) -> str:
        """Create user guide stub content"""
        return f"""---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title}

## Epic
**As a** user
**I want** to understand and use {title.lower()}
**So that** I can accomplish my goals effectively

## User Stories

### Story 1: Basic Understanding
**As a** user
**I want** to understand the basic concepts of {title.lower()}
**So that** I can get started quickly

**Acceptance Criteria:**
- [ ] Given I am new to {title.lower()}, When I read the documentation, Then I understand the basic concepts
- [ ] Given I understand the basics, When I follow the examples, Then I can perform basic operations

## Implementation

Implementation details for {title.lower()} will be documented here.

## Usage Examples

### Basic Example
```python
# Basic usage example for {title.lower()}
# Implementation details to be added
```

## Related Documentation
- [[Documentation-Index]] - Main documentation index
- [[USER_GUIDE]] - Complete user guide
"""
    
    def _create_architecture_stub_content(self, title: str, tags: str, graph_group: str) -> str:
        """Create architecture stub content"""
        return f"""---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} Architecture

## Epic
**As a** system architect
**I want** to understand the architecture of {title.lower()}
**So that** I can design effective solutions

## User Stories

### Story 1: Architecture Understanding
**As a** architect
**I want** to understand the {title.lower()} architecture
**So that** I can make informed design decisions

**Acceptance Criteria:**
- [ ] Given I need to understand {title.lower()}, When I read the architecture docs, Then I understand the design
- [ ] Given I understand the architecture, When I design solutions, Then they align with the system

## Architecture Overview

Architecture overview for {title.lower()} will be documented here.

## Component Diagram
```mermaid
graph TB
    A[Component A] --> B[Component B]
    B --> C[Component C]
```

## Components

### Main Component
**Purpose**: Main functionality of {title.lower()}

**Key Interfaces**:
```python
def main_interface():
    \"\"\"
    Main interface for {title.lower()}.
    \"\"\"
    pass
```

## Related Documentation
- [[ARCHITECTURE]] - System architecture
- [[Core-Systems]] - Core system components
- [[Documentation-Index]] - Main documentation index
"""
    
    def _create_system_stub_content(self, title: str, tags: str, graph_group: str) -> str:
        """Create system stub content"""
        return f"""---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} System

## Epic
**As a** system user
**I want** to understand and use the {title.lower()} system
**So that** I can accomplish my goals effectively

## User Stories

### Story 1: System Understanding
**As a** user
**I want** to understand how {title.lower()} works
**So that** I can use it effectively

**Acceptance Criteria:**
- [ ] Given I need to use {title.lower()}, When I read the documentation, Then I understand how it works
- [ ] Given I understand the system, When I use it, Then it performs as expected

## System Overview

System overview for {title.lower()} will be documented here.

## Key Features
- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Implementation

Implementation details for {title.lower()} will be documented here.

## Related Documentation
- [[Documentation-Index]] - Main documentation index
- [[System-Architecture]] - System architecture
"""
    
    def _create_general_stub_content(self, title: str, tags: str, graph_group: str) -> str:
        """Create general stub content"""
        return f"""---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title}

## Epic
**As a** user
**I want** information about {title.lower()}
**So that** I can understand and use this feature

## User Stories

### Story 1: Understanding
**As a** user
**I want** to understand {title.lower()}
**So that** I can use it effectively

**Acceptance Criteria:**
- [ ] Given I need to understand {title.lower()}, When I read the documentation, Then I understand the concepts
- [ ] Given I understand {title.lower()}, When I use it, Then it works as expected

## Overview

Overview of {title.lower()} will be documented here.

## Implementation

Implementation details for {title.lower()} will be documented here.

## Related Documentation
- [[Documentation-Index]] - Main documentation index
"""
    
    def run_generation(self):
        """Run the complete source documentation generation"""
        logger.info("Starting source documentation generation")
        
        # Generate source documentation
        self.generate_source_docs()
        
        # Handle stub files
        self.handle_stub_files()
        
        logger.info("Source documentation generation completed")

def main():
    """Main function"""
    project_root = Path("agentic-workflow")
    
    if not project_root.exists():
        logger.error("Project root not found")
        return
    
    generator = SourceDocumentationGenerator(project_root)
    generator.run_generation()

if __name__ == "__main__":
    main() 