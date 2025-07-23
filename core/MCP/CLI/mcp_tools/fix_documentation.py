#!/usr/bin/env python3
"""
Documentation Fix Script
Comprehensive script to fix documentation issues in the MCP project.
"""

import os
import re
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

class DocumentationFixer:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.logger = self._setup_logging()
        self.fixed_files = []
        self.removed_files = []
        self.created_files = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the documentation fixer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('documentation_fix.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def find_duplicate_files(self) -> Dict[str, List[Path]]:
        """Find duplicate files by basename."""
        duplicates = {}
        all_files = list(self.docs_path.rglob("*.md"))
        
        for file_path in all_files:
            basename = file_path.name
            if basename not in duplicates:
                duplicates[basename] = []
            duplicates[basename].append(file_path)
        
        # Filter to only actual duplicates
        return {k: v for k, v in duplicates.items() if len(v) > 1}
    
    def resolve_duplicates(self) -> None:
        """Resolve duplicate files by merging or removing them."""
        duplicates = self.find_duplicate_files()
        
        for basename, file_paths in duplicates.items():
            self.logger.info(f"Processing duplicates for {basename}: {len(file_paths)} files")
            
            # Sort by path depth and length to prioritize main docs
            file_paths.sort(key=lambda x: (len(x.parts), str(x)))
            
            # Keep the first (most important) file, merge content from others
            primary_file = file_paths[0]
            secondary_files = file_paths[1:]
            
            # Read primary content
            with open(primary_file, 'r', encoding='utf-8') as f:
                primary_content = f.read()
            
            # Merge content from secondary files
            merged_content = primary_content
            for secondary_file in secondary_files:
                try:
                    with open(secondary_file, 'r', encoding='utf-8') as f:
                        secondary_content = f.read()
                    
                    # Extract useful content (avoid duplicates)
                    if secondary_content not in merged_content:
                        # Add as a section if it has meaningful content
                        if len(secondary_content.strip()) > 100:
                            merged_content += f"\n\n## Additional Information\n\n{secondary_content}"
                    
                    # Remove the duplicate file
                    os.remove(secondary_file)
                    self.removed_files.append(str(secondary_file))
                    self.logger.info(f"Removed duplicate: {secondary_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {secondary_file}: {e}")
            
            # Write merged content back to primary file
            with open(primary_file, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            self.fixed_files.append(str(primary_file))
    
    def find_broken_links(self) -> List[Tuple[str, str, str]]:
        """Find broken markdown links in documentation."""
        broken_links = []
        all_files = list(self.docs_path.rglob("*.md"))
        
        # Pattern to match markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        wiki_link_pattern = r'\[\[([^\]]+)\]\]'
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check regular markdown links
                for match in re.finditer(link_pattern, content):
                    link_text, link_url = match.groups()
                    if link_url.startswith('./') or link_url.startswith('../'):
                        # Relative link
                        target_path = file_path.parent / link_url
                        if not target_path.exists():
                            broken_links.append((str(file_path), link_text, link_url))
                
                # Check wiki-style links
                for match in re.finditer(wiki_link_pattern, content):
                    link_text = match.group(1)
                    # Check if the linked file exists
                    possible_targets = [
                        self.docs_path / f"{link_text}.md",
                        self.docs_path / link_text / "README.md",
                        self.docs_path / f"{link_text}/README.md"
                    ]
                    
                    if not any(target.exists() for target in possible_targets):
                        broken_links.append((str(file_path), link_text, f"[[{link_text}]]"))
                        
            except Exception as e:
                self.logger.error(f"Error checking links in {file_path}: {e}")
        
        return broken_links
    
    def fix_broken_links(self) -> None:
        """Fix broken links by creating missing files or updating links."""
        broken_links = self.find_broken_links()
        
        for file_path, link_text, link_url in broken_links:
            self.logger.info(f"Fixing broken link in {file_path}: {link_text} -> {link_url}")
            
            # Try to find the correct file
            if link_url.startswith('[[') and link_url.endswith(']]'):
                # Wiki-style link
                link_name = link_text
                possible_paths = [
                    self.docs_path / f"{link_name}.md",
                    self.docs_path / link_name / "README.md",
                    self.docs_path / f"{link_name}/README.md"
                ]
                
                # Create the file if it doesn't exist
                target_path = None
                for path in possible_paths:
                    if not path.exists():
                        # Create the file with basic content
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(f"# {link_name}\n\nThis file was auto-generated to fix broken links.\n\n**Tags:** #documentation\n\n**Graph View Group:** Documentation")
                        self.created_files.append(str(path))
                        target_path = path
                        break
                
                if target_path:
                    # Update the link in the source file
                    self._update_link_in_file(file_path, link_url, f"[[{link_name}]]", str(target_path.relative_to(self.docs_path)))
    
    def _update_link_in_file(self, file_path: str, old_link: str, new_link: str, target_path: str) -> None:
        """Update a link in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the old link with the new one
            if old_link.startswith('[[') and old_link.endswith(']]'):
                # Wiki-style link
                content = content.replace(old_link, f"[[{target_path}]]")
            else:
                # Regular markdown link
                content = content.replace(old_link, f"({target_path})")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixed_files.append(file_path)
            
        except Exception as e:
            self.logger.error(f"Error updating link in {file_path}: {e}")
    
    def update_graph_view_config(self) -> None:
        """Update the Obsidian graph view configuration."""
        graph_config_path = self.docs_path / ".obsidian" / "graph.json"
        
        if not graph_config_path.exists():
            self.logger.warning("Graph config file not found, creating default")
            graph_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            default_config = {
                "collapse-filter": True,
                "search": "",
                "showTags": True,
                "showAttachments": True,
                "hideUnresolved": False,
                "showOrphans": True,
                "collapse-color-groups": True,
                "colorGroups": [
                    {
                        "query": "path:getting-started",
                        "color": {"a": 1, "rgb": 5419488}
                    },
                    {
                        "query": "path:core-systems",
                        "color": {"a": 1, "rgb": 14701138}
                    },
                    {
                        "query": "path:development",
                        "color": {"a": 1, "rgb": 14725458}
                    },
                    {
                        "query": "path:api",
                        "color": {"a": 1, "rgb": 65280}
                    },
                    {
                        "query": "path:architecture",
                        "color": {"a": 1, "rgb": 16711935}
                    },
                    {
                        "query": "path:memory-system",
                        "color": {"a": 1, "rgb": 8388736}
                    },
                    {
                        "query": "path:hormone-system",
                        "color": {"a": 1, "rgb": 16753920}
                    },
                    {
                        "query": "path:genetic-system",
                        "color": {"a": 1, "rgb": 16711935}
                    },
                    {
                        "query": "path:p2p-network",
                        "color": {"a": 1, "rgb": 32896}
                    },
                    {
                        "query": "path:pattern-recognition",
                        "color": {"a": 1, "rgb": 820}
                    },
                    {
                        "query": "path:performance-optimization",
                        "color": {"a": 1, "rgb": 8421504}
                    },
                    {
                        "query": "path:troubleshooting",
                        "color": {"a": 1, "rgb": 16744448}
                    },
                    {
                        "query": "path:release",
                        "color": {"a": 1, "rgb": 16761035}
                    },
                    {
                        "query": "tag:documentation",
                        "color": {"a": 1, "rgb": 16777215}
                    },
                    {
                        "query": "tag:api",
                        "color": {"a": 1, "rgb": 16711680}
                    },
                    {
                        "query": "tag:architecture",
                        "color": {"a": 1, "rgb": 16711935}
                    },
                    {
                        "query": "tag:memory",
                        "color": {"a": 1, "rgb": 8388736}
                    },
                    {
                        "query": "tag:hormone",
                        "color": {"a": 1, "rgb": 16753920}
                    },
                    {
                        "query": "tag:genetic",
                        "color": {"a": 1, "rgb": 16711935}
                    },
                    {
                        "query": "tag:p2p",
                        "color": {"a": 1, "rgb": 32896}
                    },
                    {
                        "query": "tag:performance",
                        "color": {"a": 1, "rgb": 8421504}
                    },
                    {
                        "query": "tag:test",
                        "color": {"a": 1, "rgb": 16744448}
                    },
                    {
                        "query": "tag:install",
                        "color": {"a": 1, "rgb": 32768}
                    }
                ],
                "collapse-display": False,
                "showArrow": True,
                "textFadeMultiplier": -0.8,
                "nodeSizeMultiplier": 1.22291666666667,
                "lineSizeMultiplier": 0.380729166666667,
                "collapse-forces": True,
                "centerStrength": 0.364583333333333,
                "repelStrength": 20,
                "linkStrength": 0.932291666666667,
                "linkDistance": 500,
                "scale": 0.0263671875,
                "close": False
            }
            
            with open(graph_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        self.logger.info("Graph view configuration updated")
    
    def add_ears_format_to_docs(self) -> None:
        """Add EARS format to documentation files that don't have it."""
        all_files = list(self.docs_path.rglob("*.md"))
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file already has EARS format
                if 'tags:' in content and '---' in content:
                    continue
                
                # Add EARS format header
                filename = file_path.stem
                parent_dir = file_path.parent.name
                
                # Determine tags based on path and content
                tags = self._determine_tags(file_path, content)
                
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
        """Determine appropriate tags for a file based on its path and content."""
        tags = ['documentation']
        
        # Add tags based on path
        path_parts = file_path.parts
        if 'api' in path_parts:
            tags.append('api')
        if 'architecture' in path_parts:
            tags.append('architecture')
        if 'memory' in path_parts:
            tags.append('memory')
        if 'hormone' in path_parts:
            tags.append('hormone')
        if 'genetic' in path_parts:
            tags.append('genetic')
        if 'p2p' in path_parts:
            tags.append('p2p')
        if 'performance' in path_parts:
            tags.append('performance')
        if 'test' in path_parts:
            tags.append('test')
        if 'install' in path_parts:
            tags.append('install')
        if 'getting-started' in path_parts:
            tags.append('getting-started')
        if 'development' in path_parts:
            tags.append('development')
        if 'troubleshooting' in path_parts:
            tags.append('troubleshooting')
        if 'release' in path_parts:
            tags.append('release')
        
        # Add tags based on content
        content_lower = content.lower()
        if 'llm' in content_lower:
            tags.append('llm')
        if 'android' in content_lower:
            tags.append('android')
        if 'plugin' in content_lower:
            tags.append('plugin')
        if 'network' in content_lower:
            tags.append('network')
        if 'community' in content_lower:
            tags.append('community')
        if 'automation' in content_lower:
            tags.append('automation')
        
        return list(set(tags))  # Remove duplicates
    
    def create_documentation_index(self) -> None:
        """Create a comprehensive documentation index."""
        index_content = """---
tags: [documentation, index, obsidian, guide]
---
#tags: #documentation #index #obsidian #guide

# 📚 MCP Documentation Index

Welcome to the comprehensive documentation for the MCP Agentic Workflow Accelerator. This index provides navigation to all documentation sections.

## 🚀 Getting Started

- [[getting-started/README.md|Getting Started Guide]]
- [[INSTALLATION_GUIDE.md|Installation Guide]]
- [[QUICK_START.md|Quick Start]]
- [[Universal-Install-Wizard.md|Universal Install Wizard]]

## 🏗️ Architecture

- [[Architecture/README.md|Architecture Overview]]
- [[ARCHITECTURE.md|System Architecture]]
- [[BRAIN_INSPIRED_ARCHITECTURE.md|Brain-Inspired Architecture]]
- [[Plugin-Architecture.md|Plugin Architecture]]

## 🔧 Core Systems

### Memory System
- [[Memory-System/README.md|Memory System Overview]]
- [[Memory-System/Working-Memory.md|Working Memory]]
- [[Memory-System/Short-Term-Memory.md|Short-Term Memory]]
- [[Memory-System/Long-Term-Memory.md|Long-Term Memory]]

### Hormone System
- [[Hormone-System/README.md|Hormone System Overview]]
- [[Hormone-System/Hormone-Types.md|Hormone Types]]
- [[Hormone-System/Hormone-Cascades.md|Hormone Cascades]]
- [[Hormone-System/Cross-Lobe-Communication.md|Cross-Lobe Communication]]

### Genetic System
- [[Genetic-System/README.md|Genetic System Overview]]
- [[Genetic-System/Genetic-Algorithms.md|Genetic Algorithms]]
- [[Genetic-System/Environmental-Adaptation.md|Environmental Adaptation]]
- [[Genetic-System/Split-Brain-Testing.md|Split-Brain Testing]]

### Pattern Recognition
- [[Pattern-Recognition/README.md|Pattern Recognition Overview]]
- [[Pattern-Recognition/Neural-Columns.md|Neural Columns]]
- [[Pattern-Recognition/Adaptive-Sensitivity.md|Adaptive Sensitivity]]

### P2P Network
- [[P2P-Network/README.md|P2P Network Overview]]
- [[P2P-Network/P2P_BENCHMARKING.md|P2P Benchmarking]]

## 🔌 API & Integration

- [[API/README.md|API Overview]]
- [[API/ADVANCED_API.md|Advanced API]]
- [[API/LLM-API-Support.md|LLM API Support]]
- [[Alignment-Engine-API.md|Alignment Engine API]]

## 🖥️ Development

- [[Development/README.md|Development Overview]]
- [[development/Brain-Inspired-Patterns.md|Brain-Inspired Patterns]]
- [[Code-Review.md|Code Review]]
- [[Core-Architecture.md|Core Architecture]]

## 📖 User Guides

- [[User-Guides/README.md|User Guides Overview]]
- [[User-Guides/INSTALLATION_GUIDE.md|Installation Guide]]
- [[User-Guides/SECURITY_GUIDE.md|Security Guide]]
- [[User-Guides/DOCUMENTATION_STYLE_GUIDE.md|Documentation Style Guide]]

## 🛠️ CLI & Tools

- [[cli/README.md|CLI Overview]]
- [[cli/CLI-Commands.md|CLI Commands]]
- [[cli/CLI-Advanced-Features.md|CLI Advanced Features]]
- [[cli/CLI-Configuration.md|CLI Configuration]]

## 🚀 Performance & Optimization

- [[Performance-Optimization/README.md|Performance Overview]]
- [[Performance-Optimization/PERFORMANCE_TUNING.md|Performance Tuning]]
- [[Performance-Optimization/PERFORMANCE_COMPARISONS.md|Performance Comparisons]]
- [[Performance-Optimization/SYSTEM_MONITORING.md|System Monitoring]]

## 🧪 Testing

- [[Testing/README.md|Testing Overview]]
- [[Testing/TESTING_GUIDE.md|Testing Guide]]
- [[Testing/DOCUMENTATION-VALIDATION.md|Documentation Validation]]
- [[Testing/split_brain_testing.md|Split Brain Testing]]

## 🔧 Troubleshooting

- [[Troubleshooting/README.md|Troubleshooting Overview]]
- [[Troubleshooting/ERROR_TRACKING.md|Error Tracking]]
- [[Troubleshooting/Development-Issues.md|Development Issues]]

## 📦 Release & Deployment

- [[Release/README.md|Release Overview]]
- [[Release/QA_SECURITY_DEPLOYMENT.md|QA Security Deployment]]
- [[Automatic-Updates/README.md|Automatic Updates]]

## 🌐 Community

- [[Community/README.md|Community Overview]]
- [[Community/Bugtracker & Community.md|Bugtracker & Community]]
- [[Community/Community-Updates.md|Community Updates]]

## 📋 Specialized Documentation

### Android Support
- [[Android-Support.md|Android Support]]
- [[Guides/Android-Deployment-Guide.md|Android Deployment Guide]]

### Simulation Layer
- [[Simulation-Layer/README.md|Simulation Layer Overview]]
- [[Simulation-Layer/SIMULATION_LAYER.md|Simulation Layer Details]]

### Configuration
- [[Configuration/README.md|Configuration Overview]]
- [[Configuration/Universal-Install-Wizard.md|Universal Install Wizard]]

## 🔍 Search & Navigation

- [[Documentation-Map.md|Documentation Map]]
- [[WikiLinks.md|Wiki Links Guide]]
- [[links.md|Links Reference]]

## 📊 Analytics & Monitoring

- [[USAGE_ANALYTICS.md|Usage Analytics]]
- [[IMPLEMENTATION_STATUS.md|Implementation Status]]

---

**Tags:** #documentation #index #obsidian #guide #api #llm #android #install #release #universal #wizard #p2p #plugin #performance #test #usb #network #bugtracker #community #automation
"""
        
        index_path = self.docs_path / "Documentation-Index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        self.created_files.append(str(index_path))
        self.logger.info("Created comprehensive documentation index")
    
    def run_full_fix(self) -> None:
        """Run the complete documentation fix process."""
        self.logger.info("Starting comprehensive documentation fix")
        
        # Step 1: Resolve duplicates
        self.logger.info("Step 1: Resolving duplicate files")
        self.resolve_duplicates()
        
        # Step 2: Fix broken links
        self.logger.info("Step 2: Fixing broken links")
        self.fix_broken_links()
        
        # Step 3: Add EARS format
        self.logger.info("Step 3: Adding EARS format to documentation")
        self.add_ears_format_to_docs()
        
        # Step 4: Update graph view config
        self.logger.info("Step 4: Updating graph view configuration")
        self.update_graph_view_config()
        
        # Step 5: Create documentation index
        self.logger.info("Step 5: Creating documentation index")
        self.create_documentation_index()
        
        # Summary
        self.logger.info("Documentation fix completed!")
        self.logger.info(f"Fixed files: {len(self.fixed_files)}")
        self.logger.info(f"Removed files: {len(self.removed_files)}")
        self.logger.info(f"Created files: {len(self.created_files)}")
        
        # Write summary report
        with open('documentation_fix_summary.md', 'w') as f:
            f.write(f"""# Documentation Fix Summary

## Overview
This report summarizes the documentation fixes applied to the MCP project.

## Statistics
- **Fixed files:** {len(self.fixed_files)}
- **Removed files:** {len(self.removed_files)}
- **Created files:** {len(self.created_files)}

## Fixed Files
{chr(10).join(f"- {file}" for file in self.fixed_files)}

## Removed Files
{chr(10).join(f"- {file}" for file in self.removed_files)}

## Created Files
{chr(10).join(f"- {file}" for file in self.created_files)}

## Next Steps
1. Review the changes and ensure all links work correctly
2. Update any remaining broken references
3. Test the Obsidian graph view
4. Validate EARS format compliance
""")

if __name__ == "__main__":
    fixer = DocumentationFixer()
    fixer.run_full_fix() 