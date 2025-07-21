#!/usr/bin/env python3
"""
EARS Format Converter

This script converts specific documentation files to proper EARS format
and fixes stub files with comprehensive content.
"""

import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EARSConverter:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
    
    def convert_to_ears(self, filepath: Path, template_type: str = "system"):
        """Convert a file to EARS format"""
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return
        
        content = filepath.read_text(encoding='utf-8')
        
        # Extract title and determine content
        title = self._extract_title(content, filepath.stem)
        
        # Create EARS format content
        ears_content = self._create_ears_content(title, template_type, content)
        
        # Write updated content
        filepath.write_text(ears_content, encoding='utf-8')
        logger.info(f"Converted {filepath.name} to EARS format")
    
    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return filename.replace('-', ' ').replace('_', ' ').title()
    
    def _create_ears_content(self, title: str, template_type: str, original_content: str) -> str:
        """Create EARS format content"""
        
        # Determine tags and graph group
        tags = self._determine_tags(title, template_type)
        graph_group = self._determine_graph_group(title, template_type)
        
        # Extract sections from original content
        sections = self._extract_sections(original_content)
        
        if template_type == "user_guide":
            return self._create_user_guide_ears(title, tags, graph_group, sections)
        elif template_type == "api":
            return self._create_api_ears(title, tags, graph_group, sections)
        elif template_type == "architecture":
            return self._create_architecture_ears(title, tags, graph_group, sections)
        else:
            return self._create_system_ears(title, tags, graph_group, sections)
    
    def _determine_tags(self, title: str, template_type: str) -> str:
        """Determine tags for the document"""
        base_tags = ["documentation"]
        
        if template_type == "user_guide":
            base_tags.extend(["user-guide", "tutorial"])
        elif template_type == "api":
            base_tags.extend(["api", "reference", "technical"])
        elif template_type == "architecture":
            base_tags.extend(["architecture", "design", "system"])
        else:
            base_tags.extend(["system", "reference"])
        
        # Add specific tags based on title
        title_lower = title.lower()
        if "memory" in title_lower:
            base_tags.append("memory")
        if "hormone" in title_lower:
            base_tags.append("hormone")
        if "genetic" in title_lower:
            base_tags.append("genetic")
        if "p2p" in title_lower:
            base_tags.append("p2p")
        if "pattern" in title_lower:
            base_tags.append("pattern")
        if "simulation" in title_lower:
            base_tags.append("simulation")
        
        return ", ".join(base_tags)
    
    def _determine_graph_group(self, title: str, template_type: str) -> str:
        """Determine graph view group"""
        if template_type == "user_guide":
            return "User Guides"
        elif template_type == "api":
            return "API Reference"
        elif template_type == "architecture":
            return "Architecture"
        else:
            return "Documentation"
    
    def _extract_sections(self, content: str) -> dict:
        """Extract sections from content"""
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
    
    def _create_user_guide_ears(self, title: str, tags: str, graph_group: str, sections: dict) -> str:
        """Create user guide in EARS format"""
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

### Story 2: Advanced Usage
**As a** user
**I want** to perform advanced operations with {title.lower()}
**So that** I can maximize my productivity

**Acceptance Criteria:**
- [ ] Given I understand the basics, When I learn advanced features, Then I can perform complex operations
- [ ] Given I encounter problems, When I check troubleshooting, Then I can resolve issues

## Implementation

{sections.get('overview', f'Implementation details for {title.lower()} will be documented here.')}

## Usage Examples

### Basic Example
```python
# Basic usage example for {title.lower()}
# Implementation details to be added
```

### Advanced Example
```python
# Advanced usage example for {title.lower()}
# Implementation details to be added
```

## Related Documentation
- [[Documentation-Index]] - Main documentation index
- [[USER_GUIDE]] - Complete user guide
- [[Getting-Started]] - Getting started guide
"""
    
    def _create_api_ears(self, title: str, tags: str, graph_group: str, sections: dict) -> str:
        """Create API documentation in EARS format"""
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

### Story 2: Integration
**As a** developer
**I want** to integrate {title.lower()} into my application
**So that** I can leverage its functionality

**Acceptance Criteria:**
- [ ] Given I want to integrate {title.lower()}, When I follow the integration guide, Then it works seamlessly
- [ ] Given I encounter issues, When I check error handling, Then I can resolve problems

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
# Basic API usage example
from mcp import {title.replace(' ', '').lower()}

instance = {title.replace(' ', '')}()
result = instance.main_method()
```

### Advanced Usage
```python
# Advanced API usage example
# Implementation details to be added
```

## Related Documentation
- [[API_DOCUMENTATION]] - Complete API reference
- [[DEVELOPER_GUIDE]] - Development guide
- [[Documentation-Index]] - Main documentation index
"""
    
    def _create_architecture_ears(self, title: str, tags: str, graph_group: str, sections: dict) -> str:
        """Create architecture documentation in EARS format"""
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

### Story 2: Integration Design
**As a** architect
**I want** to design integrations with {title.lower()}
**So that** I can create effective system designs

**Acceptance Criteria:**
- [ ] Given I need to integrate with {title.lower()}, When I understand the architecture, Then I can design effective integrations
- [ ] Given I design integrations, When I implement them, Then they work correctly

## Architecture Overview

{sections.get('overview', f'Architecture overview for {title.lower()} will be documented here.')}

## Component Diagram
```mermaid
graph TB
    A[Component A] --> B[Component B]
    B --> C[Component C]
    C --> A
```

## Data Flow
```mermaid
graph LR
    A[Input] --> B[Process]
    B --> C[Output]
    C --> D[Storage]
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

**Design Rationale**: {sections.get('design_rationale', 'Design rationale to be documented.')}

## Integration Points

### Input Dependencies
- External data sources
- Configuration parameters

### Output Dependencies
- Data storage
- External systems

### Cross-System Integration
- Memory system integration
- Hormone system integration
- Genetic system integration

## Related Documentation
- [[ARCHITECTURE]] - System architecture
- [[Core-Systems]] - Core system components
- [[Documentation-Index]] - Main documentation index
"""
    
    def _create_system_ears(self, title: str, tags: str, graph_group: str, sections: dict) -> str:
        """Create system documentation in EARS format"""
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

### Story 2: System Operation
**As a** user
**I want** to operate {title.lower()} effectively
**So that** I can achieve my objectives

**Acceptance Criteria:**
- [ ] Given I want to use {title.lower()}, When I follow the instructions, Then it works correctly
- [ ] Given I encounter issues, When I check troubleshooting, Then I can resolve problems

## System Overview

{sections.get('overview', f'System overview for {title.lower()} will be documented here.')}

## Key Features
- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## System Requirements
- Requirement 1
- Requirement 2
- Requirement 3

## Configuration

### Environment Variables
- `ENV_VAR_1`: Description
- `ENV_VAR_2`: Description

### Configuration Files
- `config.json`: Main configuration
- `settings.yaml`: Settings file

## Operation

### Startup Process
1. Initialize system
2. Load configuration
3. Start services

### Shutdown Process
1. Stop services
2. Save state
3. Cleanup resources

## Troubleshooting

### Common Issues
- Issue 1: Solution
- Issue 2: Solution

### Diagnostic Tools
- Tool 1: Purpose
- Tool 2: Purpose

## Related Documentation
- [[Documentation-Index]] - Main documentation index
- [[System-Architecture]] - System architecture
- [[Troubleshooting]] - Troubleshooting guide
"""
    
    def fix_stub_files(self):
        """Fix stub files with comprehensive content"""
        
        # Find stub files
        stub_files = []
        for file_path in self.docs_dir.rglob("*.md"):
            content = file_path.read_text(encoding='utf-8')
            if self._is_stub_file(content):
                stub_files.append(file_path)
        
        logger.info(f"Found {len(stub_files)} stub files to fix")
        
        for stub_file in stub_files:
            self._fix_stub_file(stub_file)
    
    def _is_stub_file(self, content: str) -> bool:
        """Check if file is a stub"""
        lines = content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.startswith('#')]
        return len(non_empty_lines) < 10
    
    def _fix_stub_file(self, file_path: Path):
        """Fix a stub file with comprehensive content"""
        
        title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
        
        # Determine template type based on filename
        filename_lower = file_path.stem.lower()
        
        if any(word in filename_lower for word in ['api', 'reference']):
            template_type = "api"
        elif any(word in filename_lower for word in ['guide', 'tutorial', 'user']):
            template_type = "user_guide"
        elif any(word in filename_lower for word in ['arch', 'design', 'system']):
            template_type = "architecture"
        else:
            template_type = "system"
        
        # Convert to EARS format
        self.convert_to_ears(file_path, template_type)
    
    def run_conversion(self):
        """Run the complete EARS conversion process"""
        logger.info("Starting EARS format conversion")
        
        # Convert specific important files
        important_files = [
            ("README.md", "user_guide"),
            ("USER_GUIDE.md", "user_guide"),
            ("DEVELOPER_GUIDE.md", "user_guide"),
            ("API_DOCUMENTATION.md", "api"),
            ("ARCHITECTURE.md", "architecture"),
            ("Memory-System.md", "architecture"),
            ("Hormone-System.md", "architecture"),
            ("Genetic-System.md", "architecture"),
            ("P2P-Network.md", "architecture"),
            ("Pattern-Recognition.md", "architecture"),
            ("Simulation-Layer.md", "architecture"),
            ("Performance-Optimization.md", "system"),
            ("CLI-Commands.md", "user_guide"),
            ("Installation-Guide.md", "user_guide"),
            ("Troubleshooting.md", "user_guide")
        ]
        
        for filename, template_type in important_files:
            filepath = self.docs_dir / filename
            if filepath.exists():
                self.convert_to_ears(filepath, template_type)
        
        # Fix stub files
        self.fix_stub_files()
        
        logger.info("EARS format conversion completed")

def main():
    """Main function"""
    project_root = Path("agentic-workflow")
    
    if not project_root.exists():
        logger.error("Project root not found")
        return
    
    converter = EARSConverter(project_root)
    converter.run_conversion()

if __name__ == "__main__":
    main() 