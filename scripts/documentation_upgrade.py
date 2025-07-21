#!/usr/bin/env python3
"""
Documentation Upgrade Script

This script upgrades all documentation to EARS format and fixes the incomplete Obsidian vault.
It systematically:
1. Converts documentation to EARS format where appropriate
2. Creates missing documentation files
3. Fixes broken Obsidian references
4. Updates graph view groups and tags
5. Reorganizes documentation structure
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocFile:
    """Represents a documentation file with metadata"""
    path: Path
    title: str
    content: str
    tags: List[str]
    graph_group: str
    obsidian_links: List[str]
    missing_links: List[str]
    is_stub: bool

class DocumentationUpgrader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.src_dir = project_root / "src"
        self.obsidian_dir = self.docs_dir / ".obsidian"
        
        # EARS format templates
        self.ears_templates = {
            "user_guide": self._get_ears_user_guide_template(),
            "api_doc": self._get_ears_api_template(),
            "architecture": self._get_ears_architecture_template(),
            "development": self._get_ears_development_template(),
            "system": self._get_ears_system_template()
        }
        
        # Graph view groups mapping
        self.graph_groups = {
            "getting-started": "Getting Started",
            "user-guides": "User Guides", 
            "development": "Development",
            "architecture": "Architecture",
            "core-systems": "Core Systems",
            "api": "API Reference",
            "reference": "Reference",
            "guides": "Guides",
            "troubleshooting": "Troubleshooting",
            "examples": "Examples",
            "community": "Community",
            "release": "Release",
            "performance": "Performance",
            "memory": "Memory System",
            "hormone": "Hormone System",
            "genetic": "Genetic System",
            "p2p": "P2P Network",
            "pattern": "Pattern Recognition",
            "simulation": "Simulation Layer"
        }
        
        # Source files that need documentation
        self.source_files = self._scan_source_files()
        
    def _scan_source_files(self) -> Dict[str, List[Path]]:
        """Scan source files and categorize them"""
        source_files = {}
        
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            # Categorize by directory structure
            relative_path = py_file.relative_to(self.src_dir)
            category = relative_path.parts[0] if relative_path.parts else "misc"
            
            if category not in source_files:
                source_files[category] = []
            source_files[category].append(py_file)
            
        return source_files
    
    def _get_ears_user_guide_template(self) -> str:
        """EARS template for user guides"""
        return """---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title}

## Epic
**As a** {user_type}
**I want** {capability}
**So that** {benefit}

## User Stories

### Story 1: Basic Usage
**As a** {user_type}
**I want** {basic_capability}
**So that** {basic_benefit}

**Acceptance Criteria:**
- [ ] Given {precondition}, When {action}, Then {expected_result}
- [ ] Given {precondition}, When {action}, Then {expected_result}

### Story 2: Advanced Usage
**As a** {user_type}
**I want** {advanced_capability}
**So that** {advanced_benefit}

**Acceptance Criteria:**
- [ ] Given {precondition}, When {action}, Then {expected_result}
- [ ] Given {precondition}, When {action}, Then {expected_result}

## Implementation

{implementation_details}

## Related Documentation
{related_docs}
"""

    def _get_ears_api_template(self) -> str:
        """EARS template for API documentation"""
        return """---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} API

## Epic
**As a** developer
**I want** {api_capability}
**So that** {api_benefit}

## API Reference

### Classes

#### {main_class}
```python
class {main_class}:
    \"\"\"
    {class_description}
    \"\"\"
    
    def __init__(self, {params}):
        \"\"\"
        Initialize {main_class}.
        
        Args:
            {param_docs}
        \"\"\"
        pass
```

### Methods

#### {main_method}
```python
def {main_method}(self, {method_params}) -> {return_type}:
    \"\"\"
    {method_description}
    
    Args:
        {method_param_docs}
        
    Returns:
        {return_docs}
        
    Raises:
        {exceptions}
    \"\"\"
    pass
```

## Usage Examples

### Basic Usage
```python
{basic_example}
```

### Advanced Usage
```python
{advanced_example}
```

## Related Documentation
{related_docs}
"""

    def _get_ears_architecture_template(self) -> str:
        """EARS template for architecture documentation"""
        return """---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} Architecture

## Epic
**As a** system architect
**I want** {arch_capability}
**So that** {arch_benefit}

## Architecture Overview

### High-Level Design
{high_level_design}

### Component Diagram
```mermaid
{component_diagram}
```

### Data Flow
```mermaid
{data_flow_diagram}
```

## Components

### {component_name}
**Purpose**: {component_purpose}

**Key Interfaces**:
```python
{component_interfaces}
```

**Design Rationale**: {design_rationale}

## Integration Points

### Input Dependencies
- {input_deps}

### Output Dependencies  
- {output_deps}

### Cross-System Integration
- {cross_system_integration}

## Performance Considerations

### Scalability
{scalability_notes}

### Resource Usage
{resource_usage}

### Optimization Strategies
{optimization_strategies}

## Related Documentation
{related_docs}
"""

    def _get_ears_development_template(self) -> str:
        """EARS template for development documentation"""
        return """---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} Development Guide

## Epic
**As a** developer
**I want** {dev_capability}
**So that** {dev_benefit}

## Development Setup

### Prerequisites
{prerequisites}

### Installation
{installation_steps}

### Configuration
{configuration_steps}

## Development Workflow

### Code Standards
{code_standards}

### Testing Strategy
{testing_strategy}

### Debugging
{debugging_guide}

## API Development

### Creating New APIs
{api_creation_guide}

### Testing APIs
{api_testing_guide}

### Documentation Standards
{doc_standards}

## Related Documentation
{related_docs}
"""

    def _get_ears_system_template(self) -> str:
        """EARS template for system documentation"""
        return """---
tags: [{tags}]
graph-view-group: {graph_group}
---

# {title} System

## Epic
**As a** system administrator
**I want** {sys_capability}
**So that** {sys_benefit}

## System Overview

### Purpose
{system_purpose}

### Key Features
{key_features}

### System Requirements
{system_requirements}

## Configuration

### Environment Variables
{env_vars}

### Configuration Files
{config_files}

### Runtime Options
{runtime_options}

## Operation

### Startup Process
{startup_process}

### Shutdown Process
{shutdown_process}

### Monitoring
{monitoring_guide}

## Troubleshooting

### Common Issues
{common_issues}

### Diagnostic Tools
{diagnostic_tools}

### Recovery Procedures
{recovery_procedures}

## Related Documentation
{related_docs}
"""

    def scan_existing_docs(self) -> Dict[str, DocFile]:
        """Scan existing documentation files"""
        docs = {}
        
        for md_file in self.docs_dir.rglob("*.md"):
            if md_file.name.startswith("."):
                continue
                
            content = md_file.read_text(encoding='utf-8')
            
            # Extract metadata
            title = self._extract_title(content, md_file.name)
            tags = self._extract_tags(content)
            graph_group = self._determine_graph_group(md_file)
            obsidian_links = self._extract_obsidian_links(content)
            missing_links = self._find_missing_links(obsidian_links)
            is_stub = self._is_stub_file(content)
            
            docs[str(md_file.relative_to(self.docs_dir))] = DocFile(
                path=md_file,
                title=title,
                content=content,
                tags=tags,
                graph_group=graph_group,
                obsidian_links=obsidian_links,
                missing_links=missing_links,
                is_stub=is_stub
            )
            
        return docs
    
    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from markdown content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return filename.replace('.md', '').replace('-', ' ').title()
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from markdown frontmatter"""
        tags = []
        
        # Look for YAML frontmatter tags
        yaml_match = re.search(r'tags:\s*\[(.*?)\]', content)
        if yaml_match:
            tag_str = yaml_match.group(1)
            tags.extend([tag.strip() for tag in tag_str.split(',') if tag.strip()])
        
        # Look for hashtag tags
        hashtag_matches = re.findall(r'#(\w+)', content)
        tags.extend(hashtag_matches)
        
        return list(set(tags))
    
    def _determine_graph_group(self, file_path: Path) -> str:
        """Determine graph view group based on file path"""
        relative_path = file_path.relative_to(self.docs_dir)
        
        # Check directory-based grouping
        for dir_pattern, group in self.graph_groups.items():
            if dir_pattern in str(relative_path):
                return group
        
        # Default grouping based on filename
        filename = file_path.stem.lower()
        for pattern, group in self.graph_groups.items():
            if pattern in filename:
                return group
                
        return "Documentation"
    
    def _extract_obsidian_links(self, content: str) -> List[str]:
        """Extract all Obsidian-style links from content"""
        links = re.findall(r'\[\[([^\]]+)\]\]', content)
        return [link.split('|')[0].strip() for link in links]
    
    def _find_missing_links(self, links: List[str]) -> List[str]:
        """Find links that don't correspond to existing files"""
        missing = []
        
        for link in links:
            # Handle different link formats
            link_path = link
            if link.endswith('.md'):
                link_path = link
            else:
                link_path = f"{link}.md"
            
            # Check if file exists
            possible_paths = [
                self.docs_dir / link_path,
                self.docs_dir / f"{link_path}.md",
                self.docs_dir / link_path.replace('-', '_') / "README.md"
            ]
            
            if not any(path.exists() for path in possible_paths):
                missing.append(link)
                
        return missing
    
    def _is_stub_file(self, content: str) -> bool:
        """Check if file is a stub (minimal content)"""
        lines = content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.startswith('#')]
        return len(non_empty_lines) < 10
    
    def create_missing_docs(self, existing_docs: Dict[str, DocFile]) -> List[Path]:
        """Create missing documentation files"""
        created_files = []
        
        # Create docs for source files
        for category, files in self.source_files.items():
            for file_path in files:
                doc_name = self._generate_doc_name(file_path)
                doc_path = self.docs_dir / f"{doc_name}.md"
                
                if not doc_path.exists():
                    logger.info(f"Creating documentation for {file_path}")
                    content = self._create_source_doc(file_path, category)
                    doc_path.write_text(content, encoding='utf-8')
                    created_files.append(doc_path)
        
        # Create docs for missing links
        all_missing_links = set()
        for doc in existing_docs.values():
            all_missing_links.update(doc.missing_links)
        
        for link in all_missing_links:
            # Handle links that might include directory paths
            if '/' in link or '\\' in link:
                # Split path and create directories
                parts = link.replace('\\', '/').split('/')
                filename = parts[-1]
                dir_parts = parts[:-1]
                
                if dir_parts:
                    doc_dir = self.docs_dir / '/'.join(dir_parts)
                    doc_dir.mkdir(parents=True, exist_ok=True)
                    doc_path = doc_dir / f"{filename}.md"
                else:
                    doc_path = self.docs_dir / f"{filename}.md"
            else:
                doc_path = self.docs_dir / f"{link}.md"
            
            if not doc_path.exists():
                logger.info(f"Creating missing documentation: {link}")
                content = self._create_missing_doc(link)
                doc_path.write_text(content, encoding='utf-8')
                created_files.append(doc_path)
        
        return created_files
    
    def _generate_doc_name(self, file_path: Path) -> str:
        """Generate documentation name from source file"""
        name = file_path.stem
        # Convert snake_case to Title Case
        name = name.replace('_', ' ').title()
        return name
    
    def _create_source_doc(self, file_path: Path, category: str) -> str:
        """Create documentation for a source file"""
        # Read source file to understand its purpose
        try:
            source_content = file_path.read_text(encoding='utf-8')
        except:
            source_content = ""
        
        # Determine template type
        if "class" in source_content and "def" in source_content:
            template_type = "api_doc"
        elif "def" in source_content:
            template_type = "api_doc"
        else:
            template_type = "system"
        
        # Extract class and method information
        classes = re.findall(r'class\s+(\w+)', source_content)
        methods = re.findall(r'def\s+(\w+)', source_content)
        
        # Generate content
        title = self._generate_doc_name(file_path)
        tags = [category, "api", "source"]
        graph_group = self.graph_groups.get(category, "API Reference")
        
        template = self.ears_templates[template_type]
        content = template.format(
            title=title,
            tags=", ".join(tags),
            graph_group=graph_group,
            user_type="developer",
            capability=f"understand and use the {title}",
            benefit=f"integrate {title} into my applications",
            api_capability=f"access {title} functionality",
            api_benefit=f"build applications using {title}",
            main_class=classes[0] if classes else "MainClass",
            class_description=f"Main class for {title} functionality",
            params="",
            param_docs="",
            main_method=methods[0] if methods else "main_method",
            method_params="",
            method_description=f"Main method for {title}",
            method_param_docs="",
            return_type="Any",
            return_docs="",
            exceptions="",
            basic_example=f"# Basic usage example for {title}",
            advanced_example=f"# Advanced usage example for {title}",
            related_docs="[[API_DOCUMENTATION]] - Complete API reference\n[[DEVELOPER_GUIDE]] - Development guide"
        )
        
        return content
    
    def _create_missing_doc(self, link: str) -> str:
        """Create documentation for a missing link"""
        title = link.replace('-', ' ').title()
        tags = ["documentation", "placeholder"]
        graph_group = "Documentation"
        
        content = f"""---
tags: [{", ".join(tags)}]
graph-view-group: {graph_group}
---

# {title}

## Epic
**As a** user
**I want** information about {title.lower()}
**So that** I can understand and use this feature

## Overview

This documentation is being created for {title}. Please refer to the related documentation below for more information.

## Related Documentation

- [[Documentation-Index]] - Main documentation index
- [[README]] - Project overview

## Implementation Status

- [ ] Content needs to be written
- [ ] Examples need to be added
- [ ] API reference needs to be created
- [ ] Integration guide needs to be written

## Notes

This is a placeholder document that will be expanded with comprehensive documentation.
"""
        return content
    
    def upgrade_to_ears_format(self, existing_docs: Dict[str, DocFile]) -> List[Path]:
        """Upgrade existing documentation to EARS format"""
        upgraded_files = []
        
        for doc_path, doc in existing_docs.items():
            if doc.is_stub or self._needs_ears_upgrade(doc.content):
                logger.info(f"Upgrading {doc_path} to EARS format")
                
                # Determine template type
                template_type = self._determine_template_type(doc)
                
                # Upgrade content
                upgraded_content = self._upgrade_content_to_ears(doc, template_type)
                
                # Write upgraded content
                doc.path.write_text(upgraded_content, encoding='utf-8')
                upgraded_files.append(doc.path)
        
        return upgraded_files
    
    def _needs_ears_upgrade(self, content: str) -> bool:
        """Check if content needs EARS format upgrade"""
        # Check if already has EARS format
        if "Epic" in content and "As a" in content and "I want" in content:
            return False
        
        # Check if it's a comprehensive document that should be in EARS format
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.startswith('#')]
        return len(non_empty_lines) > 20  # Substantial content
    
    def _determine_template_type(self, doc: DocFile) -> str:
        """Determine which EARS template to use"""
        path_str = str(doc.path)
        filename = doc.path.stem.lower()
        
        if "user" in filename or "guide" in filename:
            return "user_guide"
        elif "api" in filename or "reference" in filename:
            return "api_doc"
        elif "arch" in filename or "design" in filename:
            return "architecture"
        elif "dev" in filename or "development" in filename:
            return "development"
        else:
            return "system"
    
    def _upgrade_content_to_ears(self, doc: DocFile, template_type: str) -> str:
        """Upgrade existing content to EARS format"""
        # Extract existing content sections
        sections = self._extract_content_sections(doc.content)
        
        template = self.ears_templates[template_type]
        
        # Map existing content to EARS format
        if template_type == "user_guide":
            return self._upgrade_to_user_guide(doc, sections, template)
        elif template_type == "api_doc":
            return self._upgrade_to_api_doc(doc, sections, template)
        elif template_type == "architecture":
            return self._upgrade_to_architecture(doc, sections, template)
        elif template_type == "development":
            return self._upgrade_to_development(doc, sections, template)
        else:
            return self._upgrade_to_system(doc, sections, template)
    
    def _extract_content_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from existing content"""
        sections = {}
        current_section = "overview"
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].lower().replace(' ', '_')
                current_content = []
            elif line.startswith('# '):
                sections['title'] = line[2:].strip()
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _upgrade_to_user_guide(self, doc: DocFile, sections: Dict[str, str], template: str) -> str:
        """Upgrade to user guide EARS format"""
        title = sections.get('title', doc.title)
        
        return template.format(
            title=title,
            tags=", ".join(doc.tags),
            graph_group=doc.graph_group,
            user_type="user",
            capability=f"use {title.lower()}",
            benefit=f"accomplish tasks with {title.lower()}",
            basic_capability=f"perform basic operations with {title.lower()}",
            basic_benefit=f"get started quickly",
            advanced_capability=f"perform advanced operations with {title.lower()}",
            advanced_benefit=f"maximize productivity",
            precondition="I have access to the system",
            action=f"I use {title.lower()}",
            expected_result="I can accomplish my goals",
            implementation_details=sections.get('overview', 'Implementation details to be added.'),
            related_docs="[[Documentation-Index]] - Main documentation index"
        )
    
    def _upgrade_to_api_doc(self, doc: DocFile, sections: Dict[str, str], template: str) -> str:
        """Upgrade to API documentation EARS format"""
        title = sections.get('title', doc.title)
        
        return template.format(
            title=title,
            tags=", ".join(doc.tags),
            graph_group=doc.graph_group,
            api_capability=f"access {title.lower()} APIs",
            api_benefit=f"integrate {title.lower()} into my applications",
            main_class=title.replace(' ', ''),
            class_description=f"Main class for {title}",
            params="",
            param_docs="",
            main_method="main_method",
            method_params="",
            method_description=f"Main method for {title}",
            method_param_docs="",
            return_type="Any",
            return_docs="",
            exceptions="",
            basic_example=sections.get('examples', '# Basic usage example'),
            advanced_example=sections.get('advanced_examples', '# Advanced usage example'),
            related_docs="[[API_DOCUMENTATION]] - Complete API reference"
        )
    
    def _upgrade_to_architecture(self, doc: DocFile, sections: Dict[str, str], template: str) -> str:
        """Upgrade to architecture EARS format"""
        title = sections.get('title', doc.title)
        
        return template.format(
            title=title,
            tags=", ".join(doc.tags),
            graph_group=doc.graph_group,
            arch_capability=f"understand {title.lower()} architecture",
            arch_benefit=f"design and implement {title.lower()} effectively",
            high_level_design=sections.get('overview', 'High-level design to be documented.'),
            component_diagram="graph TB\n    A[Component A] --> B[Component B]",
            data_flow_diagram="graph LR\n    A[Input] --> B[Process] --> C[Output]",
            component_name="MainComponent",
            component_purpose="Main system component",
            component_interfaces="def main_interface():\n    pass",
            design_rationale="Design rationale to be documented.",
            input_deps="External dependencies",
            output_deps="Output interfaces",
            cross_system_integration="Integration points",
            scalability_notes="Scalability considerations",
            resource_usage="Resource usage patterns",
            optimization_strategies="Optimization strategies",
            related_docs="[[ARCHITECTURE]] - System architecture"
        )
    
    def _upgrade_to_development(self, doc: DocFile, sections: Dict[str, str], template: str) -> str:
        """Upgrade to development EARS format"""
        title = sections.get('title', doc.title)
        
        return template.format(
            title=title,
            tags=", ".join(doc.tags),
            graph_group=doc.graph_group,
            dev_capability=f"develop with {title.lower()}",
            dev_benefit=f"create applications using {title.lower()}",
            prerequisites="Development prerequisites",
            installation_steps="Installation steps",
            configuration_steps="Configuration steps",
            code_standards="Code standards",
            testing_strategy="Testing strategy",
            debugging_guide="Debugging guide",
            api_creation_guide="API creation guide",
            api_testing_guide="API testing guide",
            doc_standards="Documentation standards",
            related_docs="[[DEVELOPER_GUIDE]] - Development guide"
        )
    
    def _upgrade_to_system(self, doc: DocFile, sections: Dict[str, str], template: str) -> str:
        """Upgrade to system EARS format"""
        title = sections.get('title', doc.title)
        
        return template.format(
            title=title,
            tags=", ".join(doc.tags),
            graph_group=doc.graph_group,
            sys_capability=f"operate {title.lower()}",
            sys_benefit=f"maintain {title.lower()} effectively",
            system_purpose=sections.get('overview', 'System purpose to be documented.'),
            key_features="Key system features",
            system_requirements="System requirements",
            env_vars="Environment variables",
            config_files="Configuration files",
            runtime_options="Runtime options",
            startup_process="Startup process",
            shutdown_process="Shutdown process",
            monitoring_guide="Monitoring guide",
            common_issues="Common issues",
            diagnostic_tools="Diagnostic tools",
            recovery_procedures="Recovery procedures",
            related_docs="[[Documentation-Index]] - Main documentation index"
        )
    
    def update_graph_view_config(self):
        """Update Obsidian graph view configuration"""
        graph_config_path = self.obsidian_dir / "graph.json"
        
        if not graph_config_path.exists():
            logger.warning("Graph config not found, creating default")
            self._create_default_graph_config()
            return
        
        # Read existing config
        with open(graph_config_path, 'r') as f:
            config = json.load(f)
        
        # Update color groups
        config['colorGroups'] = self._generate_color_groups()
        
        # Write updated config
        with open(graph_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Updated graph view configuration")
    
    def _create_default_graph_config(self):
        """Create default graph view configuration"""
        config = {
            "collapse-filter": True,
            "search": "",
            "showTags": True,
            "showAttachments": True,
            "hideUnresolved": False,
            "showOrphans": True,
            "collapse-color-groups": True,
            "colorGroups": self._generate_color_groups()
        }
        
        graph_config_path = self.obsidian_dir / "graph.json"
        with open(graph_config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _generate_color_groups(self) -> List[Dict]:
        """Generate color groups for graph view"""
        colors = [
            {"a": 1, "rgb": 5419488},    # Blue
            {"a": 1, "rgb": 14701138},   # Green
            {"a": 1, "rgb": 14725458},   # Yellow
            {"a": 1, "rgb": 11621088},   # Purple
            {"a": 1, "rgb": 15105570},   # Orange
            {"a": 1, "rgb": 5431378},    # Cyan
            {"a": 1, "rgb": 5431473},    # Magenta
            {"a": 1, "rgb": 16711935},   # Pink
            {"a": 1, "rgb": 16711680},   # Red
            {"a": 1, "rgb": 65280},      # Lime
            {"a": 1, "rgb": 8421504},    # Gray
            {"a": 1, "rgb": 2552550},    # Light Green
            {"a": 1, "rgb": 16753920},   # Gold
            {"a": 1, "rgb": 32896},      # Dark Blue
            {"a": 1, "rgb": 820},        # Dark Green
            {"a": 1, "rgb": 16744448},   # Dark Red
            {"a": 1, "rgb": 16761035},   # Light Orange
            {"a": 1, "rgb": 16776960},   # Yellow
            {"a": 1, "rgb": 8388736},    # Dark Purple
            {"a": 1, "rgb": 16711935}    # Pink
        ]
        
        color_groups = []
        color_index = 0
        
        for group_name, display_name in self.graph_groups.items():
            color_groups.append({
                "query": f"path:{group_name}",
                "color": colors[color_index % len(colors)]
            })
            color_index += 1
        
        # Add tag-based groups
        tag_groups = [
            "android", "api", "plugin", "network", "performance", 
            "memory", "hormone", "genetic", "p2p", "pattern", "simulation"
        ]
        
        for tag in tag_groups:
            color_groups.append({
                "query": f"tag:{tag}",
                "color": colors[color_index % len(colors)]
            })
            color_index += 1
        
        return color_groups
    
    def update_3d_graph_config(self):
        """Update 3D graph configuration"""
        config_path = self.obsidian_dir / "plugins" / "3d-graph" / "data.json"
        
        if not config_path.exists():
            logger.warning("3D graph config not found")
            return
        
        # Read existing config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update groups
        config['groups']['groups'] = self._generate_3d_groups()
        
        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Updated 3D graph configuration")
    
    def _generate_3d_groups(self) -> List[Dict]:
        """Generate groups for 3D graph"""
        groups = []
        colors = ["#746767", "#8B4513", "#2E8B57", "#4682B4", "#DDA0DD", "#F0E68C", "#FF6347", "#40E0D0"]
        
        for i, (group_name, display_name) in enumerate(self.graph_groups.items()):
            groups.append({
                "query": group_name,
                "color": colors[i % len(colors)]
            })
        
        return groups
    
    def fix_obsidian_links(self, existing_docs: Dict[str, DocFile]) -> List[Path]:
        """Fix broken Obsidian links"""
        fixed_files = []
        
        for doc_path, doc in existing_docs.items():
            if doc.missing_links:
                logger.info(f"Fixing links in {doc_path}")
                
                # Read content
                content = doc.path.read_text(encoding='utf-8')
                
                # Fix each missing link
                for missing_link in doc.missing_links:
                    # Try to find the correct file
                    correct_path = self._find_correct_file_path(missing_link)
                    if correct_path:
                        # Replace broken link with correct link
                        old_link = f"[[{missing_link}]]"
                        new_link = f"[[{correct_path}]]"
                        content = content.replace(old_link, new_link)
                
                # Write fixed content
                doc.path.write_text(content, encoding='utf-8')
                fixed_files.append(doc.path)
        
        return fixed_files
    
    def _find_correct_file_path(self, link: str) -> str:
        """Find the correct file path for a broken link"""
        # Try different variations
        variations = [
            link,
            f"{link}.md",
            link.replace('-', '_'),
            link.replace('_', '-'),
            link.lower(),
            link.title().replace(' ', '')
        ]
        
        for variation in variations:
            possible_paths = [
                self.docs_dir / variation,
                self.docs_dir / f"{variation}.md",
                self.docs_dir / variation / "README.md"
            ]
            
            for path in possible_paths:
                if path.exists():
                    return str(path.relative_to(self.docs_dir)).replace('.md', '')
        
        return None
    
    def reorganize_documentation(self):
        """Reorganize documentation structure"""
        logger.info("Reorganizing documentation structure")
        
        # Create new directory structure
        new_structure = {
            "getting-started": ["Quick-Start", "Installation", "First-Project-Walkthrough"],
            "user-guides": ["User-Guide", "CLI-Commands", "Configuration"],
            "development": ["Developer-Guide", "API-Reference", "Plugin-Development"],
            "architecture": ["System-Architecture", "Core-Systems", "Brain-Inspired-Systems"],
            "api": ["API-Documentation", "Advanced-API", "Integration-Guide"],
            "reference": ["CLI-Reference", "Configuration-Reference", "Error-Codes"],
            "guides": ["Performance-Guide", "Security-Guide", "Deployment-Guide"],
            "troubleshooting": ["Troubleshooting", "System-Diagnostics", "Recovery-Procedures"],
            "examples": ["Basic-Examples", "Advanced-Examples", "Integration-Examples"],
            "community": ["Community-Guide", "Contributing", "Support"],
            "release": ["Release-Notes", "Changelog", "Migration-Guide"]
        }
        
        # Move files to appropriate directories
        for directory, files in new_structure.items():
            dir_path = self.docs_dir / directory
            dir_path.mkdir(exist_ok=True)
            
            for file_name in files:
                source_path = self.docs_dir / f"{file_name}.md"
                target_path = dir_path / f"{file_name}.md"
                
                if source_path.exists() and not target_path.exists():
                    shutil.move(str(source_path), str(target_path))
                    logger.info(f"Moved {source_path} to {target_path}")
    
    def run_complete_upgrade(self):
        """Run the complete documentation upgrade process"""
        logger.info("Starting comprehensive documentation upgrade")
        
        # Step 1: Scan existing documentation
        logger.info("Step 1: Scanning existing documentation")
        existing_docs = self.scan_existing_docs()
        logger.info(f"Found {len(existing_docs)} existing documentation files")
        
        # Step 2: Create missing documentation
        logger.info("Step 2: Creating missing documentation")
        created_files = self.create_missing_docs(existing_docs)
        logger.info(f"Created {len(created_files)} new documentation files")
        
        # Step 3: Upgrade to EARS format
        logger.info("Step 3: Upgrading to EARS format")
        upgraded_files = self.upgrade_to_ears_format(existing_docs)
        logger.info(f"Upgraded {len(upgraded_files)} files to EARS format")
        
        # Step 4: Fix Obsidian links
        logger.info("Step 4: Fixing Obsidian links")
        fixed_files = self.fix_obsidian_links(existing_docs)
        logger.info(f"Fixed links in {len(fixed_files)} files")
        
        # Step 5: Update graph view configurations
        logger.info("Step 5: Updating graph view configurations")
        self.update_graph_view_config()
        self.update_3d_graph_config()
        
        # Step 6: Reorganize documentation structure
        logger.info("Step 6: Reorganizing documentation structure")
        self.reorganize_documentation()
        
        logger.info("Documentation upgrade completed successfully!")

def main():
    """Main function"""
    project_root = Path("agentic-workflow")
    
    if not project_root.exists():
        logger.error("Project root not found. Please run from the correct directory.")
        return
    
    upgrader = DocumentationUpgrader(project_root)
    upgrader.run_complete_upgrade()

if __name__ == "__main__":
    main() 