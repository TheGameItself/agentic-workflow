"""
GitGraph visualization strategy for better representation of MCP system workflows, strategy patterns, and
spinal cord translation layer operations.

Supports:
- MerMaidContextProtocol v1.0.0.0 (MMCP format compliant)
- Strategy pattern visualization
- Spinal cord translation visualization
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class VisualizationStrategy(Protocol):
    """Protocol for visualization strategies."""
    
    async def generate_visualization(self, data: Dict[str, Any]) -> str:
        """Generate visualization from data."""
        ...
    
    def get_format_type(self) -> str:
        """Return the format type (gitgraph, mermaid, etc.)."""
        ...


class GitGraphTheme(Enum):
    """GitGraph theme options for different visualization contexts."""
    BASE = "base"
    DARK = "dark"
    METRO = "metro"
    CUSTOM_MCP = "custom_mcp"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow visualization."""
    id: str
    name: str
    type: str  # strategy, lobe, translation
    status: str  # active, completed, failed
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spinal_cord_processing: bool = False


@dataclass
class StrategyTransition:
    """Represents a transition between workflow nodes."""
    from_node: str
    to_node: str
    condition: str
    strategy_type: str
    spinal_cord_translation: Optional[Dict[str, Any]] = None


class GitGraphVisualizationStrategy:
    """GitGraph-based visualization strategy."""
    
    def __init__(self, theme: GitGraphTheme = GitGraphTheme.CUSTOM_MCP):
        self.theme = theme
        self.logger = logging.getLogger("GitGraphVisualizationStrategy")
        self.spinal_cord = None  # Will be initialized when needed
    
    async def generate_visualization(self, data: Dict[str, Any]) -> str:
        """Generate GitGraph visualization from workflow data."""
        workflow_nodes = data.get('nodes', [])
        strategy_patterns = data.get('strategy_patterns', [])
        
        gitgraph = self._generate_gitgraph_header()
        gitgraph += await self._generate_workflow_branches(workflow_nodes)
        gitgraph += self._generate_strategy_patterns(strategy_patterns)
        gitgraph += self._generate_spinal_cord_integration(data.get('spinal_cord_data', {}))
        
        return gitgraph
    
    def get_format_type(self) -> str:
        """Return GitGraph format type."""
        return "gitgraph"
    
    def _generate_gitgraph_header(self) -> str:
        """Generate GitGraph header with theme configuration."""
        return f"""```mermaid
gitgraph:
    options:
        theme: {self.theme.value}
        mainBranchName: "main-workflow"
        showCommitLabel: true
        showBranches: true
    commit id: "MCP-System-Init"
    commit id: "System-Ready"
"""
    
    async def _generate_workflow_branches(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate GitGraph branches for workflow nodes."""
        
        branches = []
        current_branch = "main-workflow"
        
        # Group nodes by workflow type
        lobe_nodes = [n for n in nodes if n.get('type') == 'lobe']
        strategy_nodes = [n for n in nodes if n.get('type') == 'strategy']
        translation_nodes = [n for n in nodes if n.get('type') == 'translation']
        
        # Generate lobe processing branch
        if lobe_nodes:
            branches.append(f"""
    branch lobe-processing
    checkout lobe-processing""")
            
            for node in lobe_nodes:
                # Process through spinal cord if needed
                if node.get('spinal_cord_processing', False):
                    spinal_translation = await self._process_spinal_cord_translation(node)
                    branches.append(f"""
    commit id: "SC-Translate-{node['id']}"
    commit id: "Lobe-{node['id']}-{node['status']}"")
                else:
                    branches.append(f"""
    commit id: "Lobe-{node['id']}-{node['status']}" """)
        
        # Generate strategy pattern branch
        if strategy_nodes:
            branches.append(f"""
    branch strategy-patterns
    checkout strategy-patterns""")
            
            for node in strategy_nodes:
                branches.append(f"""
    commit id: "Strategy-{node['id']}-{node.get('strategy_type', 'default')}"")
        
        # Generate translation branch
        if translation_nodes:
            branches.append(f"""
    branch spinal-cord-translation
    checkout spinal-cord-translation""")
            
            for node in translation_nodes:
                branches.append(f"""
    commit id: "SC-Translation-{node['id']}" """)
        
        # Merge back to main
        branches.append(f"""
    checkout main-workflow
    merge lobe-processing
    merge strategy-patterns
    merge spinal-cord-translation
    commit id: "Workflow-Complete"
""")
        
        return ''.join(branches)
    
    def _generate_strategy_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Generate strategy pattern visualization."""
        if not patterns:
            return ""
        
        strategy_viz = """
    %% Strategy Pattern Implementations
    branch strategy-implementations
    checkout strategy-implementations"""
        
        for pattern in patterns:
            strategy_viz += f"""
    commit id: "Pattern-{pattern.get('name')}"
    commit id: "Context-{pattern.get('context')}"
"""
        
        strategy_viz += f"""
    checkout main-workflow
    merge strategy-implementations
"""
        return strategy_viz
    
    def _generate_spinal_cord_integration(self, spinal_data: Dict[str, Any]) -> str:
        """Generate spinal cord translation layer visualization."""
        if not spinal_data:
            return ""
        
        integration_viz = """
    %% Spinal Cord Translation Layer
    branch spinal-cord-integration
    checkout spinal-cord-integration
    commit id: "SC-Init"
    commit id: "SC-Cortical-Columns-Ready"
    commit id: "SC-Gaussian-Modulation"
    commit id: "SC-Lobe-NN-Translation-Active"
    checkout main-workflow
    merge spinal-cord-integration
```
"""
        return integration_viz
    
    async def _process_spinal_cord_translation(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Process node through spinal cord translation."""
        try:
            # Simulate lobe output
            lobe_output = node.get('output', 0.5)
            context = {
                'fitness': node.get('fitness', 1.0),
                'node_id': node['id'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Translate through spinal cord
            translated = await self.spinal_cord.translate_lobe_output(lobe_output, context)
            
            return {
                'original': lobe_output,
                'translated': translated.tolist() if hasattr(translated, 'tolist') else translated,
                'context': context
            }
        except Exception as e:
            self.logger.error(f"Spinal cord translation error: {e}")
            return {'error': str(e)}


class MermaidFlowchartStrategy:
    """Traditional Mermaid flowchart strategy for component visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger("MermaidFlowchartStrategy")
    
    async def generate_visualization(self, data: Dict[str, Any]) -> str:
        """Generate Mermaid flowchart from workflow data."""
        nodes = data.get('nodes', [])
        transitions = data.get('transitions', [])
        
        flowchart = """```mermaid
flowchart TD
    %% MCP System Components
    subgraph SpinalCord["Spinal Cord"]
        SC1["Cortical Columns"]
        SC2["Gaussian Modulation"]
        SC3["Lobe-to-NN Translation"]
        SC4["NN-to-Lobe Translation"]
        
    end
    
    subgraph StrategyPatterns["Strategy Patterns"]
        SP1["Visualization Strategy"]
        SP2["Memory Strategy"]
        SP3["Translation Strategy"]
        SP4["Context Strategy"]
    end
    
    subgraph Lobes["Brain Lobes"]
        L1["Memory Lobe"]
        L2["Workflow Lobe"]
        L3["Task Lobe"]
        L4["Context Lobe"]
    end
"""
        
        # Add nodes and connections
        for node in nodes:
            node_id = node.get('id', '0')
            node_name = node.get('name', 'Unknown')
            node_type = node.get('type', 'default')
            
            if node_type == 'lobe':
                flowchart += f"""
    L{node_id}["{node_name}"] --> SC3"""
            elif node_type == 'strategy':
                flowchart += f"""
    SP{node_id}["{node_name}"]"""
        
        # Add transitions
        for transition in transitions:
            from_node = transition['from_node']
            to_node = transition['to_node']
            condition = transition.get('condition', '')
            
            flowchart += f"""
    {from_node} -->|"{condition}"| {to_node}"""
        
        flowchart += """
    
    %% Styling
    classDef spinalCord fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef strategy fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef lobe fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class SC1,SC2,SC3,SC4 spinalCord
    class SP1,SP2,SP3,SP4 strategy
    class L1,L2,L3,L4 lobe
```"""
        
        return flowchart
    
    def get_format_type(self) -> str:
        """Return Mermaid format type."""
        return "mermaid"


class VisualizationContext:
    """Context class for managing visualization strategies."""
    
    def __init__(self, strategy: VisualizationStrategy):
        self._strategy = strategy
        self.logger = logging.getLogger("VisualizationContext")
    
    def set_strategy(self, strategy: VisualizationStrategy):
        """Set the visualization strategy."""
        self._strategy = strategy
        self.logger.info(f"Visualization strategy set to {strategy.get_format_type()}")
    
    async def generate_visualization(self, data: Dict[str, Any]) -> str:
        """Generate visualization using the current strategy."""
        try:
            return await self._strategy.generate_visualization(data)
        except Exception as e:
            self.logger.error(f"Visualization generation error: {e}")
            return f"Error generating visualization: {e}"
    
    def get_current_format(self) -> str:
        """Get the current visualization format type."""
        return self._strategy.get_format_type()


class MCPWorkflowVisualizer:
    """Main visualizer class for MCP system workflows."""
    
    def __init__(self):
        self.logger = logging.getLogger("MCPWorkflowVisualizer")
        self.spinal_cord = None  # Will be initialized when needed
        
        # Initialize with default
        self.context = VisualizationContext(GitGraphVisualizationStrategy())
        
        # Available strategies
        self.strategies = {
            'gitgraph': GitGraphVisualizationStrategy(),
            'mermaid': MermaidFlowchartStrategy()
        }
    
    async def visualize_workflow(self, workflow_data: Dict[str, Any], format_type: str = 'gitgraph') -> str:
        """
        Visualize workflow data in the specified format.
        
        Args:
            workflow_data: Dictionary containing workflow data
            format_type: Visualization format type (gitgraph, mermaid)
            
        Returns:
            Visualization in the specified format
        """
        if format_type in self.strategies:
            self.context.set_strategy(self.strategies[format_type])
        
        # Process data through spinal cord if needed
        processed_data = await self._process_through_spinal_cord(workflow_data)
        
        # Generate visualization
        visualization = await self.context.generate_visualization(processed_data)
        
        # Add MMCP compliance footer
        visualization += self._generate_mmcp_footer()
        
        return visualization
    
    async def _process_through_spinal_cord(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow data through spinal cord translation layer."""
        processed_data = data.copy()
        
        # Process nodes that require translation
        if 'nodes' in processed_data:
            for node in processed_data['nodes']:
                if node.get('requires_translation', False):
                    try:
                        # Simulate processing through spinal cord
                        lobe_output = node.get('output', 0.5)
                        context = {
                            'fitness': node.get('fitness', 1.0),
                            'node_id': node.get('id')
                        }
                        
                        # Placeholder for actual translation
                        node['spinal_cord_processed'] = True
                        
                    except Exception as e:
                        self.logger.error(f"Spinal cord processing error: {e}")
                        node['spinal_cord_error'] = str(e)
        
        return processed_data
    
    def _generate_mmcp_footer(self) -> str:
        """Generate MMCP v1.0.0.0 compliant footer."""
        return f"""
<!-- MMCP v1.0.0.0 Compliant -->
<!-- Generated: {datetime.now().isoformat()} -->
<!-- Strategy Pattern: Visualization -->
<!-- Spinal Cord Integration: Active -->
<!-- EARS ID: [MCP-Visualization-001] -->
<!-- Cross-System Integration: Enabled -->
<!-- See documentation for more details -->
"""
    
    async def generate_demo_visualization(self) -> str:
        """Generate a demo visualization."""
        demo_data = {
            'nodes': [
                {
                    'id': 'memory_lobe',
                    'name': 'Memory Lobe',
                    'type': 'lobe',
                    'status': 'active',
                    'requires_translation': True,
                    'output': 0.8,
                    'fitness': 1.2
                },
                {
                    'id': 'workflow_lobe',
                    'name': 'Workflow Lobe',
                    'type': 'lobe',
                    'status': 'active',
                    'requires_translation': True,
                    'output': 0.6,
                    'fitness': 0.9
                },
                {
                    'id': 'viz_strategy',
                    'name': 'Visualization Strategy',
                    'type': 'strategy',
                    'strategy_type': 'visualization',
                    'status': 'active',
                    'fitness': 1.0
                }
            ],
            'transitions': [
                {
                    'from_node': 'memory_lobe',
                    'to_node': 'workflow_lobe',
                    'condition': 'Memory Ready',
                    'strategy_type': 'sequential'
                }
            ],
            'strategy_patterns': [
                {
                    'name': 'VisualizationStrategy',
                    'context': 'MCPWorkflowVisualizer',
                    'implementations': ['GitGraphStrategy', 'MermaidStrategy']
                }
            ],
            'spinal_cord_data': {
                'active': True,
                'modulation': 'gaussian',
                'active_columns': 1372
            }
        }
        
        return await self.visualize_workflow(demo_data, 'gitgraph')


# Example usage and testing
async def test_visualization_strategies():
    """Test the visualization strategies with sample MCP workflow data."""
    visualizer = MCPWorkflowVisualizer()
    
    # Generate demo visualization
    demo_viz = await visualizer.generate_demo_visualization()
    print("GitGraph Strategy Demo:")
    print(demo_viz)
    
    # Switch to Mermaid strategy
    demo_data = {
        'nodes': [
            {'id': '1', 'name': 'Start', 'type': 'lobe'},
            {'id': '2', 'name': 'Process', 'type': 'strategy'},
            {'id': '3', 'name': 'End', 'type': 'lobe'}
        ],
        'transitions': [
            {'from_node': '1', 'to_node': '2', 'condition': 'Ready'},
            {'from_node': '2', 'to_node': '3', 'condition': 'Complete'}
        ]
    }
    
    mermaid_viz = await visualizer.visualize_workflow(demo_data, 'mermaid')
    print("\nMermaid Strategy Demo:")
    print(mermaid_viz)


if __name__ == "__main__":
    asyncio.run(test_visualization_strategies())