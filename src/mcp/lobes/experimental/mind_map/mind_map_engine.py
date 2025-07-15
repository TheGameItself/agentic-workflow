from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
from typing import Optional, Dict, Any, List
import logging

class MindMapEngine:
    """
    Mind Map Engine
    Implements graph-based memory association, dynamic context export, and visualization.
    See idea.txt for requirements and research references.
    """
    def __init__(self, db_path: Optional[str] = None, **kwargs):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        # Placeholder for graph structure (node: [edges])
        self.graph = {}

    def add_association(self, node_a: str, node_b: str, label: str = "related"):
        """Add an association (edge) between two nodes in the mind map graph."""
        if node_a not in self.graph:
            self.graph[node_a] = []
        self.graph[node_a].append({"target": node_b, "label": label})
        if node_b not in self.graph:
            self.graph[node_b] = []
        self.graph[node_b].append({"target": node_a, "label": label})

    def export_map(self, format: str = "dict") -> Dict[str, Any]:
        """
        Export the mind map as a graph structure for visualization or context association.
        - format: 'dict' (default) or 'mermaid' (for graph visualization)
        """
        if format == "mermaid":
            # Export as Mermaid.js graph for visualization
            lines = ["graph TD"]
            for node, edges in self.graph.items():
                for edge in edges:
                    lines.append(f'    {node}-->{edge["target"]}:::"{edge["label"]}"')
            return {"mermaid": "\n".join(lines)}
        # Default: export as dict
        return {"graph": self.graph}

    def add_node(self, name: str, attrs: dict = None):
        """Add a node to the mind map graph."""
        if name not in self.graph:
            self.graph[name] = []
        return len(self.graph)

    def add_edge(self, node_a: str, node_b: str, label: str = "related"):
        """Add an edge between two nodes in the mind map graph."""
        self.add_association(node_a, node_b, label)
        return True 