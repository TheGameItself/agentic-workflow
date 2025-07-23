#!/usr/bin/env python3
"""
Documentation Cleanup Script
Clean up documentation by removing duplicates, fixing links, and organizing files.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set
import logging

class DocumentationCleanup:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.logger = self._setup_logging()
        self.removed_files = []
        self.fixed_files = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def remove_duplicate_readmes(self) -> None:
        """Remove duplicate README.md files, keeping the most relevant ones."""
        # Find all README.md files
        readme_files = list(self.docs_path.rglob("README.md"))
        
        # Group by directory depth and content similarity
        readme_groups = {}
        for readme in readme_files:
            depth = len(readme.parts) - len(self.docs_path.parts)
            if depth not in readme_groups:
                readme_groups[depth] = []
            readme_groups[depth].append(readme)
        
        # Keep the most relevant README in each directory
        for depth, files in readme_groups.items():
            if len(files) > 1:
                # Sort by path to prioritize main directories
                files.sort(key=lambda x: str(x))
                
                # Keep the first one, remove others
                keep_file = files[0]
                for remove_file in files[1:]:
                    try:
                        # Read content to see if it's different
                        with open(keep_file, 'r', encoding='utf-8') as f:
                            keep_content = f.read()
                        with open(remove_file, 'r', encoding='utf-8') as f:
                            remove_content = f.read()
                        
                        # If content is different, merge it
                        if remove_content not in keep_content and len(remove_content.strip()) > 50:
                            # Add as a section
                            with open(keep_file, 'a', encoding='utf-8') as f:
                                f.write(f"\n\n## Additional Information from {remove_file.name}\n\n{remove_content}")
                        
                        # Remove the duplicate file
                        os.remove(remove_file)
                        self.removed_files.append(str(remove_file))
                        self.logger.info(f"Removed duplicate README: {remove_file}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {remove_file}: {e}")
    
    def fix_broken_links(self) -> None:
        """Fix broken markdown links by creating missing files or updating links."""
        all_files = list(self.docs_path.rglob("*.md"))
        
        # Pattern to match markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        wiki_link_pattern = r'\[\[([^\]]+)\]\]'
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check wiki-style links
                for match in re.finditer(wiki_link_pattern, content):
                    link_text = match.group(1)
                    
                    # Skip if it's already a valid link
                    if '|' in link_text:
                        link_name = link_text.split('|')[0]
                    else:
                        link_name = link_text
                    
                    # Check if the linked file exists
                    possible_targets = [
                        self.docs_path / f"{link_name}.md",
                        self.docs_path / link_name / "README.md",
                        self.docs_path / f"{link_name}/README.md"
                    ]
                    
                    if not any(target.exists() for target in possible_targets):
                        # Create a basic file if it doesn't exist
                        target_path = self.docs_path / f"{link_name}.md"
                        if not target_path.exists():
                            # Create the file with basic content
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(target_path, 'w', encoding='utf-8') as f:
                                f.write(f"""---
tags: [documentation, auto-generated]
---
#tags: #documentation #auto-generated

# {link_name}

This file was auto-generated to fix broken links.

## Overview

This document provides information about {link_name}.

## Related Links

- [[Documentation-Index.md|Documentation Index]]

---

**Graph View Group:** Documentation

**Tags:** #documentation #auto-generated
""")
                            self.fixed_files.append(str(target_path))
                            self.logger.info(f"Created missing file: {target_path}")
                        
            except Exception as e:
                self.logger.error(f"Error checking links in {file_path}: {e}")
    
    def add_ears_format(self) -> None:
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
                
                # Determine tags based on path
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
        """Determine appropriate tags for a file."""
        tags = ['documentation']
        
        # Add tags based on path
        path_parts = [p.lower() for p in file_path.parts]
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
        if 'cli' in path_parts:
            tags.append('cli')
        if 'user-guides' in path_parts:
            tags.append('user-guide')
        if 'community' in path_parts:
            tags.append('community')
        
        return list(set(tags))
    
    def organize_files(self) -> None:
        """Organize files into proper directory structure."""
        # Create main directories if they don't exist
        main_dirs = [
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
            'configuration'
        ]
        
        for dir_name in main_dirs:
            dir_path = self.docs_path / dir_name
            dir_path.mkdir(exist_ok=True)
        
        # Move files to appropriate directories
        file_moves = [
            # Getting started files
            ('INSTALLATION_GUIDE.md', 'getting-started/'),
            ('QUICK_START.md', 'getting-started/'),
            ('Universal-Install-Wizard.md', 'getting-started/'),
            
            # Architecture files
            ('ARCHITECTURE.md', 'architecture/'),
            ('BRAIN_INSPIRED_ARCHITECTURE.md', 'architecture/'),
            ('Plugin-Architecture.md', 'architecture/'),
            ('Core-System-Infrastructure.md', 'architecture/'),
            
            # API files
            ('API_DOCUMENTATION.md', 'api/'),
            ('Alignment-Engine-API.md', 'api/'),
            ('API-Reference.md', 'api/'),
            
            # Development files
            ('DEVELOPER_GUIDE.md', 'development/'),
            ('Code-Review.md', 'development/'),
            ('Core-Architecture.md', 'development/'),
            ('Development-Setup.md', 'development/'),
            
            # User guide files
            ('USER_GUIDE.md', 'user-guides/'),
            
            # Performance files
            ('Performance-Optimization.md', 'performance-optimization/'),
            
            # Testing files
            ('TESTING_GUIDE.md', 'testing/'),
            
            # Troubleshooting files
            ('Troubleshooting.md', 'troubleshooting/'),
            
            # Release files
            ('RELEASE_NOTES.md', 'release/'),
            
            # Community files
            ('Community-Updates.md', 'community/'),
            ('Bugtracker & Community.md', 'community/'),
        ]
        
        for filename, target_dir in file_moves:
            source_path = self.docs_path / filename
            target_path = self.docs_path / target_dir / filename
            
            if source_path.exists() and not target_path.exists():
                try:
                    shutil.move(str(source_path), str(target_path))
                    self.logger.info(f"Moved {filename} to {target_dir}")
                except Exception as e:
                    self.logger.error(f"Error moving {filename}: {e}")
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary and unnecessary files."""
        # Remove temporary files
        temp_patterns = [
            '*.tmp',
            '*.bak',
            '*.old',
            '*~',
            '.#*'
        ]
        
        for pattern in temp_patterns:
            for file_path in self.docs_path.rglob(pattern):
                try:
                    os.remove(file_path)
                    self.removed_files.append(str(file_path))
                    self.logger.info(f"Removed temp file: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error removing {file_path}: {e}")
        
        # Remove empty directories
        for dir_path in sorted(self.docs_path.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    self.logger.info(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    self.logger.error(f"Error removing directory {dir_path}: {e}")
    
    def run_cleanup(self) -> None:
        """Run the complete cleanup process."""
        self.logger.info("Starting documentation cleanup")
        
        # Step 1: Remove duplicate README files
        self.logger.info("Step 1: Removing duplicate README files")
        self.remove_duplicate_readmes()
        
        # Step 2: Fix broken links
        self.logger.info("Step 2: Fixing broken links")
        self.fix_broken_links()
        
        # Step 3: Add EARS format
        self.logger.info("Step 3: Adding EARS format")
        self.add_ears_format()
        
        # Step 4: Organize files
        self.logger.info("Step 4: Organizing files")
        self.organize_files()
        
        # Step 5: Clean up temp files
        self.logger.info("Step 5: Cleaning up temporary files")
        self.cleanup_temp_files()
        
        # Summary
        self.logger.info("Documentation cleanup completed!")
        self.logger.info(f"Fixed files: {len(self.fixed_files)}")
        self.logger.info(f"Removed files: {len(self.removed_files)}")
        
        # Write summary report
        with open('documentation_cleanup_summary.md', 'w') as f:
            f.write(f"""# Documentation Cleanup Summary

## Overview
This report summarizes the documentation cleanup performed on the MCP project.

## Statistics
- **Fixed files:** {len(self.fixed_files)}
- **Removed files:** {len(self.removed_files)}

## Fixed Files
{chr(10).join(f"- {file}" for file in self.fixed_files)}

## Removed Files
{chr(10).join(f"- {file}" for file in self.removed_files)}

## Next Steps
1. Review the changes and ensure all links work correctly
2. Update any remaining broken references
3. Test the Obsidian graph view
4. Validate EARS format compliance
""")

if __name__ == "__main__":
    cleanup = DocumentationCleanup()
    cleanup.run_cleanup() 