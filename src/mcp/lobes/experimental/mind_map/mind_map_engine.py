from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
from typing import Optional, Dict, Any, List, Tuple
import logging
import heapq

class MindMapEngine:
    """
    Mind Map Engine
    Implements graph-based memory association, dynamic context export, visualization, and research-driven graph algorithms.
    
    Research References:
    - idea.txt (graph-based memory, context association, visualization)
    - IJCAI 2023 (Knowledge Graphs for AI Reasoning)
    - SIGKDD 2023 (Efficient Path Finding in Knowledge Graphs)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Pluggable graph algorithms (path finding, centrality, dynamic edge weighting)
    - Advanced visualization and context export formats
    - Feedback-driven adaptation and continual learning
    - Integration with other lobes for cross-engine research and feedback
    """
    def __init__(self, db_path: Optional[str] = None, **kwargs):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        # Graph structure: node -> list of (target, label, weight)
        self.graph = {}
        self.node_attrs = {}
        self.logger = logging.getLogger("MindMapEngine")

    def add_association(self, node_a: str, node_b: str, label: str = "related", weight: float = 1.0):
        """Add an association (edge) between two nodes in the mind map graph."""
        if node_a not in self.graph:
            self.graph[node_a] = []
        self.graph[node_a].append({"target": node_b, "label": label, "weight": weight})
        if node_b not in self.graph:
            self.graph[node_b] = []
        self.graph[node_b].append({"target": node_a, "label": label, "weight": weight})
        self.logger.info(f"[MindMapEngine] Added association: {node_a} <-> {node_b} ({label}, weight={weight})")

    def add_node(self, name: str, attrs: Optional[dict] = None):
        """Add a node to the mind map graph."""
        if attrs is None:
            attrs = {}
        if name not in self.graph:
            self.graph[name] = []
        self.node_attrs[name] = attrs
        self.logger.info(f"[MindMapEngine] Added node: {name} with attrs: {attrs}")
        return len(self.graph)

    def add_edge(self, node_a: str, node_b: str, label: str = "related", weight: float = 1.0):
        """Add an edge between two nodes in the mind map graph."""
        self.add_association(node_a, node_b, label, weight)
        return True

    def export_map(self, format: str = "dict") -> Dict[str, Any]:
        """
        Export the mind map as a graph structure for visualization or context association.
        - format: 'dict' (default) or 'mermaid' (for graph visualization)
        - 'attrs': include node attributes
        """
        if format == "mermaid":
            lines = ["graph TD"]
            for node, edges in self.graph.items():
                for edge in edges:
                    lines.append(f'    {node}-->|{edge["label"]} ({edge["weight"]})|{edge["target"]}')
            return {"mermaid": "\n".join(lines)}
        elif format == "attrs":
            return {"graph": self.graph, "attrs": self.node_attrs}
        return {"graph": self.graph}

    def find_path(self, start: str, end: str) -> List[str]:
        """
        Find the shortest path between two nodes using Dijkstra's algorithm.
        Returns a list of node names representing the path.
        """
        if start not in self.graph or end not in self.graph:
            return []
        queue = [(0, start, [start])]
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == end:
                return path
            if node in visited:
                continue
            visited.add(node)
            for edge in self.graph[node]:
                if edge["target"] not in visited:
                    heapq.heappush(queue, (cost + edge["weight"], edge["target"], path + [edge["target"]]))
        return []

    def node_centrality(self) -> Dict[str, float]:
        """
        Compute degree centrality for each node (number of connections).
        Returns a dict of node -> centrality score.
        """
        return {node: len(edges) for node, edges in self.graph.items()}

    def update_edge_weight(self, node_a: str, node_b: str, new_weight: float):
        """
        Update the weight of the edge between node_a and node_b.
        """
        for edge in self.graph.get(node_a, []):
            if edge["target"] == node_b:
                edge["weight"] = new_weight
        for edge in self.graph.get(node_b, []):
            if edge["target"] == node_a:
                edge["weight"] = new_weight
        self.logger.info(f"[MindMapEngine] Updated edge weight: {node_a} <-> {node_b} to {new_weight}")

    def adapt_from_feedback(self, feedback: Dict[str, Any]):
        """
        Adapt graph structure or edge weights based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        self.logger.info(f"[MindMapEngine] Adapting from feedback: {feedback}")
        # Example: increase weight for positive feedback, decrease for negative
        if "edge" in feedback and "delta" in feedback:
            node_a, node_b = feedback["edge"]
            delta = feedback["delta"]
            for edge in self.graph.get(node_a, []):
                if edge["target"] == node_b:
                    edge["weight"] += delta
            for edge in self.graph.get(node_b, []):
                if edge["target"] == node_a:
                    edge["weight"] += delta
        self.working_memory.add({"feedback": feedback})

    def clear_memory(self):
        """Clear working memory for this engine."""
        self.working_memory.clear()
        self.logger.info("[MindMapEngine] Working memory cleared.")

    # TODO: Add path finding, centrality, and dynamic edge weighting algorithms.
    # TODO: Add demo/test methods for plugging in custom graph algorithms.
    # TODO: Document extension points and provide usage examples in README.md.
    # TODO: Integrate with other lobes for cross-engine research and feedback.
    # TODO: Add advanced feedback integration and continual learning.
    # See: idea.txt, IJCAI 2023, SIGKDD 2023, README.md, ARCHITECTURE.md 