#!/usr/bin/env python3
"""
Documentation Organization Script
Organize documentation files into proper structure and fix all issues.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set
import logging

class DocumentationOrganizer:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.logger = self._setup_logging()
        self.moved_files = []
        self.created_files = []
        self.fixed_files = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_directory_structure(self) -> None:
        """Create the proper directory structure for documentation."""
        directories = [
            'getting-started',
            'architecture',
            'core-systems',
            'api',
            'development',
            'user-guides',
            'cli',
            'performance-optimization',
            'testing',
            'troubleshooting',
            'release',
            'community',
            'guides',
            'configuration',
            'memory-system',
            'hormone-system',
            'genetic-system',
            'pattern-recognition',
            'p2p-network',
            'simulation-layer',
            'automatic-updates',
            'examples',
            'attachments'
        ]
        
        for dir_name in directories:
            dir_path = self.docs_path / dir_name
            dir_path.mkdir(exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
    
    def organize_files(self) -> None:
        """Organize files into appropriate directories."""
        # Define file organization rules
        file_organization = {
            # Getting started files
            'INSTALLATION_GUIDE.md': 'getting-started/',
            'QUICK_START.md': 'getting-started/',
            'Universal-Install-Wizard.md': 'getting-started/',
            'Android-Support.md': 'getting-started/',
            
            # Architecture files
            'ARCHITECTURE.md': 'architecture/',
            'BRAIN_INSPIRED_ARCHITECTURE.md': 'architecture/',
            'Plugin-Architecture.md': 'architecture/',
            'Core-System-Infrastructure.md': 'architecture/',
            
            # API files
            'API_DOCUMENTATION.md': 'api/',
            'Alignment-Engine-API.md': 'api/',
            'API-Reference.md': 'api/',
            
            # Development files
            'DEVELOPER_GUIDE.md': 'development/',
            'Code-Review.md': 'development/',
            'Core-Architecture.md': 'development/',
            'Development-Setup.md': 'development/',
            
            # User guide files
            'USER_GUIDE.md': 'user-guides/',
            
            # Performance files
            'Performance-Optimization.md': 'performance-optimization/',
            
            # Testing files
            'TESTING_GUIDE.md': 'testing/',
            
            # Troubleshooting files
            'Troubleshooting.md': 'troubleshooting/',
            
            # Release files
            'RELEASE_NOTES.md': 'release/',
            
            # Community files
            'Community-Updates.md': 'community/',
            'Bugtracker & Community.md': 'community/',
            
            # Memory system files
            'Memory-System.md': 'memory-system/',
            'EngramLobe.md': 'memory-system/',
            'MemoryLobe.md': 'memory-system/',
            'UnifiedMemoryManager.md': 'memory-system/',
            
            # Hormone system files
            'Hormone-System.md': 'hormone-system/',
            'hormone_system.md': 'hormone-system/',
            
            # Genetic system files
            'Genetic-System.md': 'genetic-system/',
            'genetic_codon_encoding.md': 'genetic-system/',
            
            # Pattern recognition files
            'Pattern-Recognition.md': 'pattern-recognition/',
            'PatternRecognitionLobe.md': 'pattern-recognition/',
            
            # P2P network files
            'P2P-Network.md': 'p2p-network/',
            'P2P_BENCHMARKING.md': 'p2p-network/',
            
            # Simulation layer files
            'Simulation-Layer.md': 'simulation-layer/',
            'SIMULATION_LAYER.md': 'simulation-layer/',
            
            # Automatic updates files
            'Automatic-Updates.md': 'automatic-updates/',
            
            # Examples files
            'Examples.md': 'examples/',
            
            # Attachments files
            'attachments.md': 'attachments/',
        }
        
        # Move files to appropriate directories
        for filename, target_dir in file_organization.items():
            source_path = self.docs_path / filename
            target_path = self.docs_path / target_dir / filename
            
            if source_path.exists() and not target_path.exists():
                try:
                    shutil.move(str(source_path), str(target_path))
                    self.moved_files.append(f"{filename} -> {target_dir}")
                    self.logger.info(f"Moved {filename} to {target_dir}")
                except Exception as e:
                    self.logger.error(f"Error moving {filename}: {e}")
    
    def fix_duplicate_readmes(self) -> None:
        """Fix duplicate README.md files by merging content."""
        readme_files = list(self.docs_path.rglob("README.md"))
        
        # Group by directory
        readme_groups = {}
        for readme in readme_files:
            parent_dir = readme.parent.name
            if parent_dir not in readme_groups:
                readme_groups[parent_dir] = []
            readme_groups[parent_dir].append(readme)
        
        # Process each group
        for dir_name, files in readme_groups.items():
            if len(files) > 1:
                # Sort by path depth to prioritize main directories
                files.sort(key=lambda x: len(x.parts))
                
                # Keep the first one, merge content from others
                primary_file = files[0]
                secondary_files = files[1:]
                
                # Read primary content
                with open(primary_file, 'r', encoding='utf-8') as f:
                    primary_content = f.read()
                
                # Merge content from secondary files
                for secondary_file in secondary_files:
                    try:
                        with open(secondary_file, 'r', encoding='utf-8') as f:
                            secondary_content = f.read()
                        
                        # Add content if it's different and meaningful
                        if secondary_content not in primary_content and len(secondary_content.strip()) > 50:
                            # Extract title and content
                            lines = secondary_content.split('\n')
                            title = None
                            content_start = 0
                            
                            for i, line in enumerate(lines):
                                if line.startswith('# ') and not title:
                                    title = line[2:].strip()
                                    content_start = i + 1
                                    break
                            
                            if title:
                                section_content = '\n'.join(lines[content_start:]).strip()
                                if section_content:
                                    primary_content += f"\n\n## {title}\n\n{section_content}"
                        
                        # Remove the duplicate file
                        os.remove(secondary_file)
                        self.logger.info(f"Removed duplicate README: {secondary_file}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {secondary_file}: {e}")
                
                # Write merged content back
                with open(primary_file, 'w', encoding='utf-8') as f:
                    f.write(primary_content)
                
                self.fixed_files.append(str(primary_file))
    
    def add_ears_format(self) -> None:
        """Add EARS format to all documentation files."""
        all_files = list(self.docs_path.rglob("*.md"))
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip if already has EARS format
                if 'tags:' in content and '---' in content:
                    continue
                
                # Determine tags based on path and content
                tags = self._determine_tags(file_path, content)
                
                # Create EARS header
                ears_header = f"""---
tags: [{', '.join(tags)}]
---
#tags: {' '.join(f'#{tag}' for tag in tags)}

"""
                
                # Add header to content
                new_content = ears_header + content
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                self.fixed_files.append(str(file_path))
                self.logger.info(f"Added EARS format to {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
    
    def _determine_tags(self, file_path: Path, content: str) -> List[str]:
        """Determine appropriate tags for a file."""
        tags = ['documentation']
        
        # Add tags based on path
        path_parts = [p.lower() for p in file_path.parts]
        
        # Directory-based tags
        if 'getting-started' in path_parts:
            tags.append('getting-started')
        if 'architecture' in path_parts:
            tags.append('architecture')
        if 'api' in path_parts:
            tags.append('api')
        if 'development' in path_parts:
            tags.append('development')
        if 'user-guides' in path_parts:
            tags.append('user-guide')
        if 'cli' in path_parts:
            tags.append('cli')
        if 'performance-optimization' in path_parts:
            tags.append('performance')
        if 'testing' in path_parts:
            tags.append('test')
        if 'troubleshooting' in path_parts:
            tags.append('troubleshooting')
        if 'release' in path_parts:
            tags.append('release')
        if 'community' in path_parts:
            tags.append('community')
        if 'guides' in path_parts:
            tags.append('guides')
        if 'configuration' in path_parts:
            tags.append('configuration')
        if 'memory-system' in path_parts:
            tags.append('memory')
        if 'hormone-system' in path_parts:
            tags.append('hormone')
        if 'genetic-system' in path_parts:
            tags.append('genetic')
        if 'pattern-recognition' in path_parts:
            tags.append('pattern-recognition')
        if 'p2p-network' in path_parts:
            tags.append('p2p')
        if 'simulation-layer' in path_parts:
            tags.append('simulation')
        if 'automatic-updates' in path_parts:
            tags.append('release')
        if 'examples' in path_parts:
            tags.append('examples')
        if 'attachments' in path_parts:
            tags.append('attachments')
        
        # Content-based tags
        content_lower = content.lower()
        if 'llm' in content_lower:
            tags.append('llm')
        if 'android' in content_lower:
            tags.append('android')
        if 'plugin' in content_lower:
            tags.append('plugin')
        if 'network' in content_lower:
            tags.append('network')
        if 'install' in content_lower:
            tags.append('install')
        if 'obsidian' in content_lower:
            tags.append('obsidian')
        if 'index' in content_lower:
            tags.append('index')
        
        return list(set(tags))  # Remove duplicates
    
    def create_missing_files(self) -> None:
        """Create missing files that are referenced in the documentation index."""
        missing_files = [
            'getting-started/README.md',
            'architecture/README.md',
            'core-systems/README.md',
            'api/README.md',
            'development/README.md',
            'user-guides/README.md',
            'cli/README.md',
            'performance-optimization/README.md',
            'testing/README.md',
            'troubleshooting/README.md',
            'release/README.md',
            'community/README.md',
            'guides/README.md',
            'configuration/README.md',
            'memory-system/README.md',
            'hormone-system/README.md',
            'genetic-system/README.md',
            'pattern-recognition/README.md',
            'p2p-network/README.md',
            'simulation-layer/README.md',
            'automatic-updates/README.md',
            'examples/README.md',
            'attachments/README.md'
        ]
        
        for file_path in missing_files:
            full_path = self.docs_path / file_path
            if not full_path.exists():
                # Create directory if needed
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create basic README content
                dir_name = full_path.parent.name
                title = dir_name.replace('-', ' ').title()
                
                content = f"""---
tags: [documentation, {dir_name.replace('-', '')}]
---
#tags: #documentation #{dir_name.replace('-', '')}

# {title}

## Overview

This directory contains documentation for {title.lower()}.

## Contents

- Documentation files for {title.lower()}
- Related guides and examples
- Configuration and setup information

## Related Links

- [[Documentation-Index.md|Documentation Index]]

---

**Graph View Group:** {title}

**Tags:** #documentation #{dir_name.replace('-', '')}
"""
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.created_files.append(str(full_path))
                self.logger.info(f"Created missing file: {full_path}")
    
    def run_organization(self) -> None:
        """Run the complete documentation organization process."""
        self.logger.info("Starting documentation organization")
        
        # Step 1: Create directory structure
        self.logger.info("Step 1: Creating directory structure")
        self.create_directory_structure()
        
        # Step 2: Organize files
        self.logger.info("Step 2: Organizing files")
        self.organize_files()
        
        # Step 3: Fix duplicate READMEs
        self.logger.info("Step 3: Fixing duplicate README files")
        self.fix_duplicate_readmes()
        
        # Step 4: Add EARS format
        self.logger.info("Step 4: Adding EARS format")
        self.add_ears_format()
        
        # Step 5: Create missing files
        self.logger.info("Step 5: Creating missing files")
        self.create_missing_files()
        
        # Summary
        self.logger.info("Documentation organization completed!")
        self.logger.info(f"Moved files: {len(self.moved_files)}")
        self.logger.info(f"Created files: {len(self.created_files)}")
        self.logger.info(f"Fixed files: {len(self.fixed_files)}")
        
        # Write summary report
        with open('documentation_organization_summary.md', 'w') as f:
            f.write(f"""# Documentation Organization Summary

## Overview
This report summarizes the documentation organization performed on the MCP project.

## Statistics
- **Moved files:** {len(self.moved_files)}
- **Created files:** {len(self.created_files)}
- **Fixed files:** {len(self.fixed_files)}

## Moved Files
{chr(10).join(f"- {move}" for move in self.moved_files)}

## Created Files
{chr(10).join(f"- {file}" for file in self.created_files)}

## Fixed Files
{chr(10).join(f"- {file}" for file in self.fixed_files)}

## Directory Structure
```
docs/
├── getting-started/
├── architecture/
├── core-systems/
├── api/
├── development/
├── user-guides/
├── cli/
├── performance-optimization/
├── testing/
├── troubleshooting/
├── release/
├── community/
├── guides/
├── configuration/
├── memory-system/
├── hormone-system/
├── genetic-system/
├── pattern-recognition/
├── p2p-network/
├── simulation-layer/
├── automatic-updates/
├── examples/
└── attachments/
```

## Next Steps
1. Review the organized structure
2. Update any remaining broken links
3. Test the Obsidian graph view
4. Validate EARS format compliance
5. Update the documentation index with new paths
""")

if __name__ == "__main__":
    organizer = DocumentationOrganizer()
    organizer.run_organization() 