#!/usr/bin/env python3
"""
ProjectLobe: Project Management Engine for MCP

This module implements the ProjectLobe, responsible for project initialization, configuration, and alignment.

CONFIGURATION POLICY:
- All project configuration must be centralized in config.cfg in the project root.
- config.cfg is the single source of truth for project and system configuration.
- All configuration questions, answers, and status are managed through this file.
- See ProjectManager methods for dynamic Q&A and config management.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class ProjectManager:
    """
    Manages project initialization and configuration.
    CONFIGURATION POLICY:
    - All configuration is centralized in config.cfg.
    - Use ProjectManager methods to add, answer, and retrieve config questions.
    - config.cfg is the single source of truth for project config.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize the project manager."""
        if base_path is None:
            base_path = os.getcwd()
        self.base_path = Path(base_path)
        self.config_file = None
        self.project_info = {}
    
    def find_project_config(self, search_path: Optional[str] = None) -> Optional[Path]:
        """Find the nearest project configuration file (config.cfg)."""
        if search_path is None:
            search_path = str(self.base_path)
        
        current_path = Path(search_path)
        
        # Search up the directory tree for config.cfg
        while current_path != current_path.parent:
            config_file = current_path / 'config.cfg'
            if config_file.exists():
                return config_file
            current_path = current_path.parent
        
        return None
    
    def init_project(self, name: str, path: Optional[str] = None) -> Dict[str, Any]:
        """Initialize a new project and create centralized config.cfg."""
        if path is None:
            path = os.path.join(str(self.base_path), name)
        
        project_path = Path(path)
        
        # Create project directory
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize project info
        self.project_info = {
            'name': name,
            'path': str(project_path),
            'created_at': datetime.now().isoformat(),
            'status': 'initializing',
            'steps_completed': [],
            'current_step': 'init'
        }
        
        # Create project structure
        self._create_project_structure(project_path)
        
        # Create initial configuration
        self.config_file = project_path / 'config.cfg'
        self._create_initial_config()
        
        # Create project metadata
        self._create_project_metadata(project_path)
        
        return {
            'project_name': name,
            'project_path': str(project_path),
            'status': 'initialized',
            'next_steps': self._get_initial_steps(),
            'config_file': str(self.config_file)
        }
    
    def _create_project_structure(self, project_path: Path):
        """Create the basic project directory structure."""
        directories = [
            'src',
            'tests',
            'docs',
            'data',
            'config',
            'scripts',
            'templates'
        ]
        
        for directory in directories:
            (project_path / directory).mkdir(exist_ok=True)
        
        # Create basic files
        (project_path / 'README.md').touch()
        (project_path / 'requirements.txt').touch()
    
    def _create_initial_config(self):
        """Create the initial centralized configuration file (config.cfg) with dynamic questions."""
        config_content = """[PROJECT]
name = {name}
created_at = {created_at}
status = initializing

[ALIGNMENT]
# These questions help align the LLM and user on project goals
# Fill in answers as the project progresses
project_goal = 
target_users = 
key_features = 
technical_constraints = 
timeline = 
success_metrics = 

[RESEARCH]
# Research questions to guide initial investigation
unknown_technologies = 
competitor_analysis = 
user_research_needed = 
technical_risks = 
compliance_requirements = 

[PLANNING]
# Planning questions for architecture and implementation
architecture_preferences = 
database_requirements = 
api_requirements = 
deployment_environment = 
scalability_needs = 

[DEVELOPMENT]
# Development preferences and constraints
programming_language = 
framework_preferences = 
testing_strategy = 
code_quality_standards = 
documentation_requirements = 

[DEPLOYMENT]
# Deployment and operations questions
hosting_preferences = 
domain_requirements = 
ssl_certificates = 
monitoring_needs = 
backup_strategy = 
""".format(
            name=self.project_info['name'],
            created_at=self.project_info['created_at']
        )
        
        if self.config_file is not None:
            with open(str(self.config_file), 'w') as f:
                f.write(config_content)
    
    def _create_project_metadata(self, project_path: Path):
        """Create project metadata file."""
        metadata = {
            'project_info': self.project_info,
            'workflow_steps': [
                'init',
                'research',
                'planning',
                'development',
                'testing',
                'deployment'
            ],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        metadata_file = project_path / 'project_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_initial_steps(self) -> List[str]:
        """Get the initial steps needed to complete project setup."""
        return [
            "1. Review and fill in the alignment questions in config.cfg",
            "2. Provide project idea and requirements",
            "3. Specify technical preferences and constraints",
            "4. Define success criteria and timeline",
            "5. Run 'python mcp.py start-research' to begin research phase"
        ]
    
    def add_question(self, section: str = '', question: str = '', key: str = ''):
        """Add a new question to the centralized config.cfg."""
        section = section or ''
        question = question or ''
        key = key or (question.lower().replace(' ', '_').replace('?', '').replace(':', ''))
        if not self.config_file or not self.config_file.exists():
            return False
        with open(str(self.config_file), 'r') as f:
            content = f.read()
        
        # Parse the content manually to handle comments properly
        lines = content.split('\n')
        new_lines = []
        section_found = False
        
        for line in lines:
            new_lines.append(line)
            if line.strip() == f'[{section.upper()}]':
                section_found = True
                # Add the new question after the section header
                new_lines.append(f'{key} = ')
        
        if not section_found:
            # Add new section
            new_lines.append(f'\n[{section.upper()}]')
            new_lines.append(f'{key} = ')
        
        # Write back to file
        with open(str(self.config_file), 'w') as f:
            f.write('\n'.join(new_lines))
        
        return True
    
    def answer_question(self, section: str, key: str, answer: str):
        """Answer a configuration question in the centralized config.cfg."""
        section = section or ''
        key = key or ''
        answer = answer or ''
        config_file = self.find_project_config()
        if not config_file or not config_file.exists():
            return False
        if config_file is None:
            return False
        # Read the current config file as text
        with open(str(config_file), 'r') as f:
            content = f.read()
        
        # Parse the content manually to handle comments properly
        lines = content.split('\n')
        new_lines = []
        in_section = False
        
        for line in lines:
            if line.strip().startswith('[') and line.strip().endswith(']'):
                in_section = line.strip()[1:-1].upper() == section.upper()
                new_lines.append(line)
            elif in_section and line.strip().startswith(f'{key} ='):
                new_lines.append(f'{key} = {answer}')
            else:
                new_lines.append(line)
        
        # Write back to file
        with open(str(config_file), 'w') as f:
            f.write('\n'.join(new_lines))
        
        return True
    
    def get_questions(self, section: str = '') -> Dict[str, Any]:
        """Get all questions from the centralized config.cfg."""
        section = section or ''
        config_file = self.find_project_config()
        if not config_file or not config_file.exists():
            return {}
        with open(str(config_file), 'r') as f:
            content = f.read()
        questions = {}
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                if section == '' or current_section.upper() == section.upper():
                    questions[current_section] = {}
            elif current_section and '=' in line and not line.startswith('#'):
                if section == '' or current_section.upper() == section.upper():
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    questions[current_section][key] = value
        return questions
    
    def get_unanswered_questions(self) -> List[Dict[str, str]]:
        """Get all unanswered questions from the centralized config.cfg."""
        all_questions = self.get_questions() or {}
        unanswered = []
        for section, questions in (all_questions or {}).items():
            questions = questions or {}
            for key, value in questions.items():
                key = key or ''
                value = value or ''
                if not value.strip():
                    unanswered.append({
                        'section': section,
                        'key': key,
                        'question': key.replace('_', ' ').title()
                    })
        return unanswered
    
    def update_project_status(self, status: str, current_step: str = ''):
        """Update the project status in the centralized config.cfg."""
        status = status or ''
        current_step = current_step or ''
        if not status:
            return False
        config_file = self.find_project_config()
        if not config_file or not config_file.exists():
            return False
        with open(str(config_file), 'r') as f:
            content = f.read()
        
        # Parse the content manually to handle comments properly
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            if line.strip().startswith('status ='):
                new_lines.append(f'status = {status}')
            else:
                new_lines.append(line)
        
        # Write back to file
        with open(str(config_file), 'w') as f:
            f.write('\n'.join(new_lines))
        
        # Update project info
        if current_step:
            self.project_info['current_step'] = current_step
            self.project_info['status'] = status
        
        return True
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get the current project information from the centralized config.cfg."""
        config_file = self.find_project_config()
        if not config_file or not config_file.exists():
            return {}
        if config_file is None:
            return {}
        questions = self.get_questions('PROJECT') or {}
        project_info = questions.get('PROJECT', {}) if 'PROJECT' in questions else {}
        project_info['path'] = str(config_file.parent)
        return project_info
    
    def create_task_list(self) -> List[Dict[str, Any]]:
        """Create a list of tasks based on the project configuration."""
        config = self.get_questions()
        tasks = []
        
        # Research tasks
        if 'RESEARCH' in config:
            research_tasks = [
                {
                    'title': 'Research unknown technologies',
                    'description': 'Investigate technologies mentioned in requirements',
                    'priority': 'high',
                    'section': 'research',
                    'key': 'unknown_technologies'
                },
                {
                    'title': 'Competitor analysis',
                    'description': 'Analyze existing solutions and competitors',
                    'priority': 'medium',
                    'section': 'research',
                    'key': 'competitor_analysis'
                },
                {
                    'title': 'User research',
                    'description': 'Conduct user research and gather requirements',
                    'priority': 'high',
                    'section': 'research',
                    'key': 'user_research_needed'
                }
            ]
            tasks.extend(research_tasks)
        
        # Planning tasks
        if 'PLANNING' in config:
            planning_tasks = [
                {
                    'title': 'Architecture design',
                    'description': 'Design system architecture and components',
                    'priority': 'high',
                    'section': 'planning',
                    'key': 'architecture_preferences'
                },
                {
                    'title': 'Database design',
                    'description': 'Design database schema and relationships',
                    'priority': 'medium',
                    'section': 'planning',
                    'key': 'database_requirements'
                },
                {
                    'title': 'API design',
                    'description': 'Design API endpoints and data structures',
                    'priority': 'medium',
                    'section': 'planning',
                    'key': 'api_requirements'
                }
            ]
            tasks.extend(planning_tasks)
        
        return tasks
    
    def generate_project_summary(self) -> str:
        """Generate a summary of the current project state."""
        config = self.get_questions()
        project_info = self.get_project_info()
        
        summary = f"Project: {project_info.get('name', 'Unknown')}\n"
        summary += f"Status: {project_info.get('status', 'Unknown')}\n"
        summary += f"Created: {project_info.get('created_at', 'Unknown')}\n\n"
        
        # Count answered vs unanswered questions
        total_questions = 0
        answered_questions = 0
        
        for section, questions in config.items():
            if section != 'PROJECT':
                summary += f"{section.title()}:\n"
                for key, value in questions.items():
                    total_questions += 1
                    if value.strip():
                        answered_questions += 1
                        summary += f"  ✅ {key}: {value}\n"
                    else:
                        summary += f"  ❌ {key}: (unanswered)\n"
                summary += "\n"
        
        completion_rate = (answered_questions / total_questions * 100) if total_questions > 0 else 0
        summary += f"Completion: {completion_rate:.1f}% ({answered_questions}/{total_questions} questions answered)\n"
        
        return summary
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current project configuration."""
        config = self.get_questions()
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completion_rate': 0.0
        }
        
        # Check required sections
        required_sections = ['ALIGNMENT', 'RESEARCH', 'PLANNING']
        for section in required_sections:
            if section not in config:
                validation['errors'].append(f"Missing required section: {section}")
                validation['is_valid'] = False
        
        # Check completion rate
        total_questions = 0
        answered_questions = 0
        
        for section, questions in config.items():
            if section != 'PROJECT':
                for key, value in questions.items():
                    total_questions += 1
                    if value.strip():
                        answered_questions += 1
        
        if total_questions > 0:
            validation['completion_rate'] = (answered_questions / total_questions) * 100
        
        # Add warnings for low completion
        if validation['completion_rate'] < 50:
            validation['warnings'].append(f"Low completion rate: {validation['completion_rate']:.1f}%")
        
        return validation

    def suggest_dynamic_questions(self, workflow_step: str, context: dict = {}, feedback: list = []):
        """Suggest and add dynamic questions based on workflow step, project state, and feedback."""
        context = context or {}
        feedback = feedback or []
        # Example: Add more sophisticated logic here for real use
        suggestions = []
        if workflow_step == 'research':
            if not self.get_questions('RESEARCH').get('RESEARCH', {}).get('unknown_technologies', '').strip():
                suggestions.append(('RESEARCH', 'Are there any new or emerging technologies to consider?', 'unknown_technologies'))
            if not self.get_questions('RESEARCH').get('RESEARCH', {}).get('competitor_analysis', '').strip():
                suggestions.append(('RESEARCH', 'Are there competitors or similar projects to analyze?', 'competitor_analysis'))
        elif workflow_step == 'planning':
            if not self.get_questions('PLANNING').get('PLANNING', {}).get('architecture_preferences', '').strip():
                suggestions.append(('PLANNING', 'What are your architecture preferences or constraints?', 'architecture_preferences'))
        # Add more rules as needed, including using feedback/context
        for section, question, key in suggestions:
            self.add_question(section, question, key)
        return suggestions

    def get_proactive_unanswered_questions(self, workflow_step: str = ''):
        """Return unanswered questions, optionally filtered or prioritized by workflow step."""
        workflow_step = workflow_step or ''
        unanswered = self.get_unanswered_questions()
        if workflow_step:
            # Optionally filter or prioritize by section relevant to the step
            relevant_sections = {
                'init': ['ALIGNMENT', 'PROJECT'],
                'research': ['RESEARCH'],
                'planning': ['PLANNING'],
                'development': ['DEVELOPMENT'],
                'deployment': ['DEPLOYMENT'],
            }.get(workflow_step, [])
            unanswered = [q for q in unanswered if q['section'].upper() in relevant_sections]
        return unanswered 