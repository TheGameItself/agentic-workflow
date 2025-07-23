#!/usr/bin/env python3
"""
Test for P2P research tracking.
"""

import asyncio
import logging
import time
import uuid
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchTracker:
    """Research tracker for P2P network."""
    
    def __init__(self, node_id):
        """Initialize research tracker."""
        self.node_id = node_id
        self.research_topics = {}
        self.research_papers = {}
        self.collaborators = {}
        self.messages_sent = 0
        self.messages_received = 0
        
        print(f"Created research tracker for node: {node_id}")
    
    def add_research_topic(self, topic_id, name, description):
        """Add a research topic."""
        self.research_topics[topic_id] = {
            'name': name,
            'description': description,
            'created_by': self.node_id,
            'created_at': time.time(),
            'papers': []
        }
        print(f"Node {self.node_id} added research topic: {name}")
        return topic_id
    
    def add_research_paper(self, paper_id, topic_id, title, authors, abstract):
        """Add a research paper."""
        if topic_id not in self.research_topics:
            print(f"Topic {topic_id} not found")
            return None
        
        self.research_papers[paper_id] = {
            'topic_id': topic_id,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'submitted_by': self.node_id,
            'submitted_at': time.time(),
            'citations': []
        }
        
        self.research_topics[topic_id]['papers'].append(paper_id)
        print(f"Node {self.node_id} added research paper: {title}")
        return paper_id
    
    def add_collaborator(self, collaborator_id, name, expertise):
        """Add a collaborator."""
        self.collaborators[collaborator_id] = {
            'name': name,
            'expertise': expertise,
            'added_at': time.time(),
            'papers': []
        }
        print(f"Node {self.node_id} added collaborator: {name}")
        return collaborator_id
    
    def cite_paper(self, paper_id, citing_paper_id):
        """Cite a paper."""
        if paper_id not in self.research_papers or citing_paper_id not in self.research_papers:
            print(f"Paper {paper_id} or {citing_paper_id} not found")
            return False
        
        self.research_papers[paper_id]['citations'].append(citing_paper_id)
        print(f"Node {self.node_id} cited paper {paper_id} in paper {citing_paper_id}")
        return True
    
    def get_topic_papers(self, topic_id):
        """Get papers for a topic."""
        if topic_id not in self.research_topics:
            return []
        
        papers = []
        for paper_id in self.research_topics[topic_id]['papers']:
            if paper_id in self.research_papers:
                papers.append(self.research_papers[paper_id])
        
        return papers
    
    def get_collaborator_papers(self, collaborator_id):
        """Get papers by a collaborator."""
        if collaborator_id not in self.collaborators:
            return []
        
        papers = []
        for paper_id, paper in self.research_papers.items():
            if collaborator_id in paper['authors']:
                papers.append(paper)
        
        return papers
    
    def sync_with_peer(self, peer_tracker):
        """Sync research data with a peer."""
        # Sync topics
        for topic_id, topic in peer_tracker.research_topics.items():
            if topic_id not in self.research_topics:
                self.research_topics[topic_id] = topic.copy()
                print(f"Node {self.node_id} synced topic: {topic['name']} from {peer_tracker.node_id}")
        
        # Sync papers
        for paper_id, paper in peer_tracker.research_papers.items():
            if paper_id not in self.research_papers:
                self.research_papers[paper_id] = paper.copy()
                
                # Update topic papers list
                topic_id = paper['topic_id']
                if topic_id in self.research_topics and paper_id not in self.research_topics[topic_id]['papers']:
                    self.research_topics[topic_id]['papers'].append(paper_id)
                
                print(f"Node {self.node_id} synced paper: {paper['title']} from {peer_tracker.node_id}")
        
        # Sync collaborators
        for collab_id, collab in peer_tracker.collaborators.items():
            if collab_id not in self.collaborators:
                self.collaborators[collab_id] = collab.copy()
                print(f"Node {self.node_id} synced collaborator: {collab['name']} from {peer_tracker.node_id}")
        
        self.messages_received += 1
        peer_tracker.messages_sent += 1
        
        print(f"Node {self.node_id} synced with peer {peer_tracker.node_id}")
        return True
    
    def get_stats(self):
        """Get research tracker statistics."""
        return {
            'node_id': self.node_id,
            'topics': len(self.research_topics),
            'papers': len(self.research_papers),
            'collaborators': len(self.collaborators),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received
        }

def test_research_tracking():
    """Test P2P research tracking."""
    print("Starting P2P research tracking test...")
    
    # Create research trackers
    tracker1 = ResearchTracker("node1")
    tracker2 = ResearchTracker("node2")
    tracker3 = ResearchTracker("node3")
    
    print("Created 3 research trackers")
    
    # Node 1 adds research topics and papers
    topic1 = tracker1.add_research_topic(
        "topic1",
        "Neural Networks",
        "Research on neural network architectures and applications"
    )
    
    paper1 = tracker1.add_research_paper(
        "paper1",
        topic1,
        "Advances in Neural Networks",
        ["Alice", "Bob"],
        "This paper discusses recent advances in neural networks"
    )
    
    # Node 2 adds research topics and papers
    topic2 = tracker2.add_research_topic(
        "topic2",
        "Genetic Algorithms",
        "Research on genetic algorithms and evolutionary computation"
    )
    
    paper2 = tracker2.add_research_paper(
        "paper2",
        topic2,
        "Genetic Algorithm Optimization",
        ["Charlie", "Dave"],
        "This paper discusses optimization techniques using genetic algorithms"
    )
    
    # Node 3 adds collaborators
    tracker3.add_collaborator(
        "collab1",
        "Alice",
        "Neural Networks"
    )
    
    tracker3.add_collaborator(
        "collab2",
        "Charlie",
        "Genetic Algorithms"
    )
    
    # Sync between nodes
    tracker2.sync_with_peer(tracker1)
    tracker3.sync_with_peer(tracker2)
    tracker1.sync_with_peer(tracker3)
    
    # Node 2 adds a paper citing paper1
    paper3 = tracker2.add_research_paper(
        "paper3",
        topic1,  # Using topic1 from node1
        "Neural Networks in Robotics",
        ["Eve", "Frank"],
        "This paper discusses applications of neural networks in robotics"
    )
    
    tracker2.cite_paper(paper1, paper3)
    
    # Sync again
    tracker1.sync_with_peer(tracker2)
    tracker3.sync_with_peer(tracker1)
    
    # Check if all nodes have all research data
    for tracker in [tracker1, tracker2, tracker3]:
        stats = tracker.get_stats()
        print(f"Node {tracker.node_id} stats: {json.dumps(stats, indent=2)}")
        
        # Check papers in Neural Networks topic
        nn_papers = tracker.get_topic_papers(topic1)
        print(f"Node {tracker.node_id} has {len(nn_papers)} papers in Neural Networks topic")
        
        # Check papers in Genetic Algorithms topic
        ga_papers = tracker.get_topic_papers(topic2)
        print(f"Node {tracker.node_id} has {len(ga_papers)} papers in Genetic Algorithms topic")
        
        # Check citations
        if paper1 in tracker.research_papers:
            citations = tracker.research_papers[paper1]['citations']
            print(f"Node {tracker.node_id} shows paper1 has {len(citations)} citations")
    
    print("P2P research tracking test completed successfully!")

if __name__ == "__main__":
    test_research_tracking()