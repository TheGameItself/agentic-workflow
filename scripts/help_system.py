#!/usr/bin/env python3
"""
Help System for MCP Server
Provides contextual help and documentation access.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class HelpSystem:
    """Comprehensive help system for MCP server."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.help_data = self._load_help_data()
        
    def _load_help_data(self) -> Dict[str, Any]:
        """Load help data from configuration."""
        help_file = self.project_root / "data" / "help_data.json"
        
        if help_file.exists():
            try:
                with open(help_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
                
        # Default help data
        return {
            "commands": {
                "server": {
                    "description": "Start the MCP server",
                    "usage": "python -m src.mcp.server [options]",
                    "options": [
                        "--port PORT: Server port (default: 8000)",
                        "--host HOST: Server host (default: localhost)",
                        "--debug: Enable debug mode",
                        "--config FILE: Configuration file path"
                    ],
                    "examples": [
                        "python -m src.mcp.server --port 9000",
                        "python -m src.mcp.server --debug"
                    ]
                },
                "cli": {
                    "description": "Run CLI commands",
                    "usage": "python mcp_cli.py <command> [options]",
                    "commands": [
                        "list-tasks: List all tasks",
                        "create-task: Create a new task",
                        "update-task: Update task progress",
                        "project-status: Show project status",
                        "list-tools: List available tools"
                    ],
                    "examples": [
                        "python mcp_cli.py list-tasks",
                        "python mcp_cli.py create-task 'My Task'"
                    ]
                },
                "setup": {
                    "description": "Run setup wizard",
                    "usage": "python scripts/setup_wizard.py",
                    "description_long": "Interactive setup wizard that guides you through initial configuration",
                    "steps": [
                        "Check system requirements",
                        "Setup Python environment",
                        "Configure IDE integrations",
                        "Initialize database",
                        "Setup plugins"
                    ]
                }
            },
            "concepts": {
                "mcp": {
                    "title": "Model Context Protocol (MCP)",
                    "description": "A protocol for AI assistants to interact with external tools and data sources",
                    "key_features": [
                        "Tool integration",
                        "Data access",
                        "Resource management",
                        "Standardized communication"
                    ]
                },
                "lobes": {
                    "title": "Brain-Inspired Lobes",
                    "description": "Modular components inspired by brain functions",
                    "types": [
                        "AlignmentEngine: LLM-based alignment",
                        "PatternRecognitionEngine: Neural column simulation",
                        "SimulatedReality: Entity/event/state tracking",
                        "DreamingEngine: Scenario simulation",
                        "MindMapEngine: Graph-based associations",
                        "ScientificProcessEngine: Hypothesis testing",
                        "SplitBrainABTest: Parallel agent teams",
                        "MultiLLMOrchestrator: Task routing",
                        "AdvancedEngramEngine: Dynamic coding models"
                    ]
                },
                "workflows": {
                    "title": "Workflow Management",
                    "description": "Multi-step task orchestration and management",
                    "features": [
                        "Task dependencies",
                        "Parallel execution",
                        "Progress tracking",
                        "Error handling",
                        "Dynamic adaptation"
                    ]
                }
            },
            "troubleshooting": {
                "common_issues": {
                    "node_icu_error": {
                        "title": "Node.js ICU Dependency Error",
                        "description": "Error: libicui18n.so.75: cannot open shared object file",
                        "cause": "Cursor AppImage conflicts with system Node.js",
                        "solutions": [
                            "Use alternative Node.js installation",
                            "Fix Cursor AppImage configuration",
                            "Use portable Node.js distribution"
                        ]
                    },
                    "import_errors": {
                        "title": "Import Errors",
                        "description": "ModuleNotFoundError or ImportError",
                        "solutions": [
                            "Activate virtual environment",
                            "Install missing dependencies",
                            "Check PYTHONPATH configuration"
                        ]
                    },
                    "database_errors": {
                        "title": "Database Connection Errors",
                        "description": "SQLite or database initialization issues",
                        "solutions": [
                            "Check file permissions",
                            "Verify database path",
                            "Reinitialize database"
                        ]
                    }
                }
            }
        }
        
    def show_help(self, topic: Optional[str] = None, subtopic: Optional[str] = None):
        """Show help for a specific topic."""
        if not topic:
            self.show_main_help()
        elif topic == "commands":
            self.show_commands_help(subtopic)
        elif topic == "concepts":
            self.show_concepts_help(subtopic)
        elif topic == "troubleshooting":
            self.show_troubleshooting_help(subtopic)
        elif topic == "docs":
            self.show_documentation_help()
        else:
            print(f"Unknown help topic: {topic}")
            self.show_main_help()
            
    def show_main_help(self):
        """Show main help menu."""
        print("🚀 MCP Server Help System")
        print("=" * 50)
        print()
        print("Available help topics:")
        print("  commands     - CLI commands and usage")
        print("  concepts     - Key concepts and architecture")
        print("  troubleshooting - Common issues and solutions")
        print("  docs         - Documentation files")
        print()
        print("Usage: python scripts/help_system.py <topic> [subtopic]")
        print("Example: python scripts/help_system.py commands server")
        
    def show_commands_help(self, command: Optional[str] = None):
        """Show help for commands."""
        commands = self.help_data["commands"]
        
        if not command:
            print("📋 Available Commands")
            print("=" * 30)
            for cmd, info in commands.items():
                print(f"  {cmd:<15} - {info['description']}")
            print()
            print("Use: python scripts/help_system.py commands <command>")
        else:
            if command not in commands:
                print(f"Unknown command: {command}")
                return
                
            cmd_info = commands[command]
            print(f"📋 Command: {command}")
            print("=" * 40)
            print(f"Description: {cmd_info['description']}")
            print(f"Usage: {cmd_info['usage']}")
            
            if 'options' in cmd_info:
                print("\nOptions:")
                for option in cmd_info['options']:
                    print(f"  {option}")
                    
            if 'commands' in cmd_info:
                print("\nSubcommands:")
                for subcmd in cmd_info['commands']:
                    print(f"  {subcmd}")
                    
            if 'examples' in cmd_info:
                print("\nExamples:")
                for example in cmd_info['examples']:
                    print(f"  {example}")
                    
            if 'description_long' in cmd_info:
                print(f"\nDetails: {cmd_info['description_long']}")
                
            if 'steps' in cmd_info:
                print("\nSteps:")
                for i, step in enumerate(cmd_info['steps'], 1):
                    print(f"  {i}. {step}")
                    
    def show_concepts_help(self, concept: Optional[str] = None):
        """Show help for concepts."""
        concepts = self.help_data["concepts"]
        
        if not concept:
            print("🧠 Key Concepts")
            print("=" * 20)
            for key, info in concepts.items():
                print(f"  {key:<20} - {info['title']}")
            print()
            print("Use: python scripts/help_system.py concepts <concept>")
        else:
            if concept not in concepts:
                print(f"Unknown concept: {concept}")
                return
                
            concept_info = concepts[concept]
            print(f"🧠 Concept: {concept_info['title']}")
            print("=" * 40)
            print(f"Description: {concept_info['description']}")
            
            if 'key_features' in concept_info:
                print("\nKey Features:")
                for feature in concept_info['key_features']:
                    print(f"  • {feature}")
                    
            if 'types' in concept_info:
                print("\nTypes:")
                for type_info in concept_info['types']:
                    print(f"  • {type_info}")
                    
            if 'features' in concept_info:
                print("\nFeatures:")
                for feature in concept_info['features']:
                    print(f"  • {feature}")
                    
    def show_troubleshooting_help(self, issue: Optional[str] = None):
        """Show troubleshooting help."""
        issues = self.help_data["troubleshooting"]["common_issues"]
        
        if not issue:
            print("🔧 Troubleshooting")
            print("=" * 20)
            for key, info in issues.items():
                print(f"  {key:<20} - {info['title']}")
            print()
            print("Use: python scripts/help_system.py troubleshooting <issue>")
        else:
            if issue not in issues:
                print(f"Unknown issue: {issue}")
                return
                
            issue_info = issues[issue]
            print(f"🔧 Issue: {issue_info['title']}")
            print("=" * 40)
            print(f"Description: {issue_info['description']}")
            
            if 'cause' in issue_info:
                print(f"\nCause: {issue_info['cause']}")
                
            if 'solutions' in issue_info:
                print("\nSolutions:")
                for i, solution in enumerate(issue_info['solutions'], 1):
                    print(f"  {i}. {solution}")
                    
    def show_documentation_help(self):
        """Show documentation files help."""
        print("📚 Documentation Files")
        print("=" * 30)
        
        doc_files = [
            ("README.md", "Project overview and quick start"),
            ("USER_GUIDE.md", "Comprehensive user guide"),
            ("API_DOCUMENTATION.md", "API reference and examples"),
            ("CLI_USAGE.md", "Command-line interface guide"),
            ("ADVANCED_API.md", "Advanced features and experimental lobes"),
            ("ARCHITECTURE.md", "System architecture and design"),
            ("DEVELOPER_GUIDE.md", "Development setup and guidelines"),
            ("PLUGIN_DEVELOPMENT.md", "Plugin development guide"),
            ("IDE_INTEGRATION.md", "IDE integration instructions")
        ]
        
        for filename, description in doc_files:
            file_path = self.project_root / filename
            status = "✅" if file_path.exists() else "❌"
            print(f"  {status} {filename:<25} - {description}")
            
        print()
        print("📖 Quick Access:")
        print("  cat README.md                    # View README")
        print("  cat USER_GUIDE.md               # View user guide")
        print("  cat API_DOCUMENTATION.md        # View API docs")
        print("  python scripts/help_system.py   # Interactive help")
        
    def search_help(self, query: str):
        """Search help content."""
        print(f"🔍 Searching for: {query}")
        print("=" * 40)
        
        results = []
        query_lower = query.lower()
        
        # Search in commands
        for cmd, info in self.help_data["commands"].items():
            if query_lower in cmd.lower() or query_lower in info["description"].lower():
                results.append(("command", cmd, info["description"]))
                
        # Search in concepts
        for concept, info in self.help_data["concepts"].items():
            if query_lower in concept.lower() or query_lower in info["title"].lower():
                results.append(("concept", concept, info["title"]))
                
        # Search in troubleshooting
        for issue, info in self.help_data["troubleshooting"]["common_issues"].items():
            if query_lower in issue.lower() or query_lower in info["title"].lower():
                results.append(("issue", issue, info["title"]))
                
        if results:
            print(f"Found {len(results)} results:")
            for result_type, key, description in results:
                print(f"  {result_type:<10} {key:<20} - {description}")
            print()
            print("Use: python scripts/help_system.py <type> <key>")
        else:
            print("No results found.")
            print("Try different keywords or check the main help menu.")


def main():
    """CLI interface for help system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Help System")
    parser.add_argument("topic", nargs="?", help="Help topic")
    parser.add_argument("subtopic", nargs="?", help="Help subtopic")
    parser.add_argument("--search", "-s", help="Search help content")
    
    args = parser.parse_args()
    
    help_system = HelpSystem()
    
    if args.search:
        help_system.search_help(args.search)
    else:
        help_system.show_help(args.topic, args.subtopic)


if __name__ == "__main__":
    main() 