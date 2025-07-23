#!/usr/bin/env python3
"""
Documentation Upgrade Script for MCP Agentic Workflow

This script:
1. Identifies missing documentation files referenced in WikiLinks
2. Creates missing files in EARS format
3. Updates Obsidian graph configuration
4. Reorganizes documentation structure
5. Adds proper tags and cross-references
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Dict, Set, Tuple
import click

class DocumentationUpgrader:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.obsidian_config_path = self.docs_path / ".obsidian"
        self.graph_config_path = self.obsidian_config_path / "graph.json"
        
        # EARS format template
        self.ears_template = """---
tags: [{tags}]
---
#tags: #{tag_list}

# {title}

## Event
{event_description}

## Action
{action_description}

## Result
{result_description}

## System
{system_description}

## Implementation

### Code References
{code_references}

### Configuration
{configuration}

### Integration Points
{integration_points}

## Related Documentation
{related_docs}

## Navigation
- [[Documentation-Index]] - Main documentation index
- [[{category}]] - {category} overview
"""

    def extract_wikilinks(self) -> Set[str]:
        """Extract all WikiLink references from documentation files."""
        wikilinks = set()
        
        for md_file in self.docs_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract WikiLinks: [[filename]] or [[filename|display text]]
                matches = re.findall(r'\[\[([^\]]+)\]\]', content)
                for match in matches:
                    # Remove display text if present
                    filename = match.split('|')[0].strip()
                    # Remove file extensions and paths
                    filename = filename.replace('.md', '').replace('docs/', '').replace('src/', '')
                    if filename and not filename.startswith('http'):
                        wikilinks.add(filename)
                        
            except Exception as e:
                print(f"Error reading {md_file}: {e}")
        
        return wikilinks

    def get_existing_files(self) -> Set[str]:
        """Get list of existing documentation files."""
        existing = set()
        
        for md_file in self.docs_path.rglob("*.md"):
            # Get filename without extension and path
            filename = md_file.stem
            existing.add(filename)
        
        return existing

    def identify_missing_files(self) -> Set[str]:
        """Identify missing documentation files."""
        referenced = self.extract_wikilinks()
        existing = self.get_existing_files()
        
        # Filter out common non-file references
        filtered_referenced = set()
        for ref in referenced:
            if (ref and 
                not ref.startswith('http') and 
                not ref.startswith('attachments/') and
                not ':' in ref and
                len(ref) > 1):
                filtered_referenced.add(ref)
        
        missing = filtered_referenced - existing
        return missing

    def categorize_file(self, filename: str) -> str:
        """Categorize a file based on its name and content."""
        filename_lower = filename.lower()
        
        # Core systems
        if any(keyword in filename_lower for keyword in ['memory', 'engram', 'vector']):
            return 'Memory-System'
        elif any(keyword in filename_lower for keyword in ['genetic', 'evolution', 'trigger']):
            return 'Genetic-System'
        elif any(keyword in filename_lower for keyword in ['hormone', 'cascade']):
            return 'Hormone-System'
        elif any(keyword in filename_lower for keyword in ['pattern', 'neural', 'column']):
            return 'Pattern-Recognition'
        elif any(keyword in filename_lower for keyword in ['p2p', 'network', 'peer']):
            return 'P2P-Network'
        elif any(keyword in filename_lower for keyword in ['performance', 'optimization']):
            return 'Performance-Optimization'
        elif any(keyword in filename_lower for keyword in ['simulation', 'reality']):
            return 'Simulation-Layer'
        
        # Architecture
        elif any(keyword in filename_lower for keyword in ['architecture', 'design', 'system']):
            return 'Architecture'
        elif any(keyword in filename_lower for keyword in ['api', 'interface']):
            return 'API'
        
        # Development
        elif any(keyword in filename_lower for keyword in ['development', 'code', 'programming']):
            return 'Development'
        elif any(keyword in filename_lower for keyword in ['test', 'testing']):
            return 'Testing'
        
        # User guides
        elif any(keyword in filename_lower for keyword in ['user', 'guide', 'tutorial']):
            return 'User-Guides'
        elif any(keyword in filename_lower for keyword in ['cli', 'command']):
            return 'CLI'
        
        # Configuration
        elif any(keyword in filename_lower for keyword in ['config', 'setup', 'install']):
            return 'Configuration'
        
        # Troubleshooting
        elif any(keyword in filename_lower for keyword in ['troubleshoot', 'error', 'issue']):
            return 'Troubleshooting'
        
        # Community
        elif any(keyword in filename_lower for keyword in ['community', 'collaboration']):
            return 'Community'
        
        # Release
        elif any(keyword in filename_lower for keyword in ['release', 'version', 'update']):
            return 'Release'
        
        # Default
        return 'Documentation'

    def generate_ears_content(self, filename: str, category: str) -> str:
        """Generate EARS format content for a documentation file."""
        
        # Convert filename to title
        title = filename.replace('-', ' ').replace('_', ' ').title()
        
        # Generate tags
        tags = [category.lower().replace('-', ''), 'documentation', 'ears']
        tag_list = ' #'.join(tags)
        
        # Generate EARS components based on filename
        filename_lower = filename.lower()
        
        if 'memory' in filename_lower:
            event = "Memory operation request"
            action = "Process memory storage, retrieval, or consolidation"
            result = "Memory operation completed with quality assessment"
            system = "Memory-System"
        elif 'genetic' in filename_lower:
            event = "Environmental condition change"
            action = "Activate genetic triggers and evolve adaptations"
            result = "Optimized genetic configuration for environment"
            system = "Genetic-System"
        elif 'hormone' in filename_lower:
            event = "Cross-lobe communication signal"
            action = "Release and diffuse hormones across lobes"
            result = "Coordinated lobe behavior and system modulation"
            system = "Hormone-System"
        elif 'pattern' in filename_lower:
            event = "Sensory input or data stream"
            action = "Analyze patterns using neural columns"
            result = "Pattern recognition and adaptive sensitivity"
            system = "Pattern-Recognition"
        elif 'p2p' in filename_lower:
            event = "Network collaboration request"
            action = "Establish P2P connection and exchange data"
            result = "Distributed collaboration and knowledge sharing"
            system = "P2P-Network"
        elif 'performance' in filename_lower:
            event = "Performance monitoring trigger"
            action = "Analyze system performance and optimize resources"
            result = "Optimized system performance and resource allocation"
            system = "Performance-Optimization"
        else:
            event = f"{title} operation request"
            action = f"Process {title.lower()} functionality"
            result = f"{title} operation completed successfully"
            system = category
        
        # Generate descriptions
        event_description = f"System receives a request to perform {title.lower()} operations."
        action_description = f"The system processes the {title.lower()} request using appropriate algorithms and neural networks."
        result_description = f"Successfully completes {title.lower()} operations with quality assessment and feedback."
        system_description = f"Integrates with {system} for coordinated system behavior."
        
        # Code references
        code_references = f"- [[src/mcp/{filename.lower().replace('-', '_')}.py]] - Main implementation"
        
        # Configuration
        configuration = f"- [[config/{filename.lower()}.json]] - Configuration settings"
        
        # Integration points
        integration_points = f"- [[{system}]] - System integration\n- [[Documentation-Index]] - Main documentation"
        
        # Related docs
        related_docs = f"- [[{system}]] - System overview\n- [[Documentation-Index]] - Complete documentation index"
        
        return self.ears_template.format(
            tags=', '.join(tags),
            tag_list=tag_list,
            title=title,
            event_description=event_description,
            action_description=action_description,
            result_description=result_description,
            system_description=system_description,
            code_references=code_references,
            configuration=configuration,
            integration_points=integration_points,
            related_docs=related_docs,
            category=category
        )

    def create_missing_file(self, filename: str) -> bool:
        """Create a missing documentation file."""
        category = self.categorize_file(filename)
        content = self.generate_ears_content(filename, category)
        
        # Determine file path
        file_path = self.docs_path / f"{filename}.md"
        
        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created: {file_path}")
            return True
        except Exception as e:
            print(f"Error creating {file_path}: {e}")
            return False

    def update_obsidian_graph_config(self):
        """Update Obsidian graph configuration with new categories."""
        if not self.graph_config_path.exists():
            print("Obsidian graph config not found, creating default...")
            self.create_default_graph_config()
            return
        
        try:
            with open(self.graph_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error reading graph config: {e}")
            return
        
        # Define new color groups
        new_groups = [
            {"query": "path:Memory-System", "color": {"a": 1, "rgb": 8388736}},
            {"query": "path:Genetic-System", "color": {"a": 1, "rgb": 16711935}},
            {"query": "path:Hormone-System", "color": {"a": 1, "rgb": 16753920}},
            {"query": "path:Pattern-Recognition", "color": {"a": 1, "rgb": 820}},
            {"query": "path:P2P-Network", "color": {"a": 1, "rgb": 32896}},
            {"query": "path:Performance-Optimization", "color": {"a": 1, "rgb": 8421504}},
            {"query": "path:Simulation-Layer", "color": {"a": 1, "rgb": 16711935}},
            {"query": "path:Architecture", "color": {"a": 1, "rgb": 16711935}},
            {"query": "path:API", "color": {"a": 1, "rgb": 65280}},
            {"query": "path:Development", "color": {"a": 1, "rgb": 14725458}},
            {"query": "path:Testing", "color": {"a": 1, "rgb": 16744448}},
            {"query": "path:User-Guides", "color": {"a": 1, "rgb": 15105570}},
            {"query": "path:CLI", "color": {"a": 1, "rgb": 16711680}},
            {"query": "path:Configuration", "color": {"a": 1, "rgb": 11621088}},
            {"query": "path:Troubleshooting", "color": {"a": 1, "rgb": 16744448}},
            {"query": "path:Community", "color": {"a": 1, "rgb": 2552550}},
            {"query": "path:Release", "color": {"a": 1, "rgb": 16761035}},
            {"query": "tag:ears", "color": {"a": 1, "rgb": 12345678}},
            {"query": "tag:documentation", "color": {"a": 1, "rgb": 16777215}},
        ]
        
        # Add new groups to existing config
        existing_groups = config.get('colorGroups', [])
        existing_queries = {group['query'] for group in existing_groups}
        
        for group in new_groups:
            if group['query'] not in existing_queries:
                existing_groups.append(group)
        
        config['colorGroups'] = existing_groups
        
        # Save updated config
        try:
            with open(self.graph_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print("Updated Obsidian graph configuration")
        except Exception as e:
            print(f"Error saving graph config: {e}")

    def create_default_graph_config(self):
        """Create default Obsidian graph configuration."""
        config = {
            "collapse-filter": True,
            "search": "",
            "showTags": True,
            "showAttachments": True,
            "hideUnresolved": False,
            "showOrphans": True,
            "collapse-color-groups": True,
            "colorGroups": [
                {"query": "path:Memory-System", "color": {"a": 1, "rgb": 8388736}},
                {"query": "path:Genetic-System", "color": {"a": 1, "rgb": 16711935}},
                {"query": "path:Hormone-System", "color": {"a": 1, "rgb": 16753920}},
                {"query": "path:Pattern-Recognition", "color": {"a": 1, "rgb": 820}},
                {"query": "path:P2P-Network", "color": {"a": 1, "rgb": 32896}},
                {"query": "path:Performance-Optimization", "color": {"a": 1, "rgb": 8421504}},
                {"query": "path:Simulation-Layer", "color": {"a": 1, "rgb": 16711935}},
                {"query": "path:Architecture", "color": {"a": 1, "rgb": 16711935}},
                {"query": "path:API", "color": {"a": 1, "rgb": 65280}},
                {"query": "path:Development", "color": {"a": 1, "rgb": 14725458}},
                {"query": "path:Testing", "color": {"a": 1, "rgb": 16744448}},
                {"query": "path:User-Guides", "color": {"a": 1, "rgb": 15105570}},
                {"query": "path:CLI", "color": {"a": 1, "rgb": 16711680}},
                {"query": "path:Configuration", "color": {"a": 1, "rgb": 11621088}},
                {"query": "path:Troubleshooting", "color": {"a": 1, "rgb": 16744448}},
                {"query": "path:Community", "color": {"a": 1, "rgb": 2552550}},
                {"query": "path:Release", "color": {"a": 1, "rgb": 16761035}},
                {"query": "tag:ears", "color": {"a": 1, "rgb": 12345678}},
                {"query": "tag:documentation", "color": {"a": 1, "rgb": 16777215}},
            ],
            "collapse-display": True,
            "showArrow": True,
            "textFadeMultiplier": -0.8,
            "nodeSizeMultiplier": 1.22291666666667,
            "lineSizeMultiplier": 0.380729166666667,
            "collapse-forces": True,
            "centerStrength": 0.364583333333333,
            "repelStrength": 20,
            "linkStrength": 0.40625,
            "linkDistance": 30,
            "scale": 0.0263671875,
            "close": True
        }
        
        # Ensure obsidian config directory exists
        self.obsidian_config_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.graph_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print("Created default Obsidian graph configuration")
        except Exception as e:
            print(f"Error creating graph config: {e}")

    def reorganize_documentation(self):
        """Reorganize documentation into proper directory structure."""
        categories = {
            'Memory-System': ['memory', 'engram', 'vector', 'working', 'long-term', 'short-term'],
            'Genetic-System': ['genetic', 'evolution', 'trigger', 'mutation'],
            'Hormone-System': ['hormone', 'cascade', 'diffusion'],
            'Pattern-Recognition': ['pattern', 'neural', 'column', 'sensory'],
            'P2P-Network': ['p2p', 'network', 'peer', 'collaboration'],
            'Performance-Optimization': ['performance', 'optimization', 'monitoring'],
            'Simulation-Layer': ['simulation', 'reality', 'entity', 'event'],
            'Architecture': ['architecture', 'design', 'system', 'infrastructure'],
            'API': ['api', 'interface', 'endpoint'],
            'Development': ['development', 'code', 'programming', 'debug'],
            'Testing': ['test', 'testing', 'validation'],
            'User-Guides': ['user', 'guide', 'tutorial', 'manual'],
            'CLI': ['cli', 'command', 'terminal'],
            'Configuration': ['config', 'setup', 'install', 'configuration'],
            'Troubleshooting': ['troubleshoot', 'error', 'issue', 'problem'],
            'Community': ['community', 'collaboration', 'team'],
            'Release': ['release', 'version', 'update', 'deployment']
        }
        
        # Create category directories
        for category in categories.keys():
            category_path = self.docs_path / category
            category_path.mkdir(exist_ok=True)
        
        # Move files to appropriate categories
        for md_file in self.docs_path.glob("*.md"):
            if md_file.parent == self.docs_path:  # Only process files in root docs directory
                filename_lower = md_file.stem.lower()
                
                # Find appropriate category
                target_category = None
                for category, keywords in categories.items():
                    if any(keyword in filename_lower for keyword in keywords):
                        target_category = category
                        break
                
                if target_category:
                    target_path = self.docs_path / target_category / md_file.name
                    try:
                        shutil.move(str(md_file), str(target_path))
                        print(f"Moved {md_file.name} to {target_category}/")
                    except Exception as e:
                        print(f"Error moving {md_file.name}: {e}")

    def run_upgrade(self, create_missing: bool = True, reorganize: bool = True, update_config: bool = True):
        """Run the complete documentation upgrade process."""
        print("🔍 Analyzing documentation structure...")
        
        if create_missing:
            print("\n📝 Creating missing documentation files...")
            missing_files = self.identify_missing_files()
            
            if missing_files:
                print(f"Found {len(missing_files)} missing files:")
                for filename in sorted(missing_files):
                    print(f"  - {filename}")
                
                created_count = 0
                for filename in sorted(missing_files):
                    if self.create_missing_file(filename):
                        created_count += 1
                
                print(f"\n✅ Created {created_count} missing documentation files")
            else:
                print("✅ No missing files found")
        
        if reorganize:
            print("\n📁 Reorganizing documentation structure...")
            self.reorganize_documentation()
            print("✅ Documentation reorganization complete")
        
        if update_config:
            print("\n⚙️ Updating Obsidian configuration...")
            self.update_obsidian_graph_config()
            print("✅ Obsidian configuration updated")
        
        print("\n🎉 Documentation upgrade complete!")

@click.command()
@click.option('--docs-path', default='docs', help='Path to documentation directory')
@click.option('--no-create', is_flag=True, help='Skip creating missing files')
@click.option('--no-reorganize', is_flag=True, help='Skip reorganizing structure')
@click.option('--no-config', is_flag=True, help='Skip updating Obsidian config')
def main(docs_path, no_create, no_reorganize, no_config):
    """Upgrade MCP documentation to EARS format and organize structure."""
    upgrader = DocumentationUpgrader(docs_path)
    
    upgrader.run_upgrade(
        create_missing=not no_create,
        reorganize=not no_reorganize,
        update_config=not no_config
    )

if __name__ == '__main__':
    main() 