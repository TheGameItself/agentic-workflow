#!/usr/bin/env python3
"""
Creative Engine for MCP Core System
Implements creative problem solving and idea generation capabilities.
"""

import logging
import random
import time
import threading
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class CreativeMode(Enum):
    """Creative processing modes."""
    DIVERGENT = "divergent"      # Generate many ideas
    CONVERGENT = "convergent"    # Refine and combine ideas
    ASSOCIATIVE = "associative"  # Make connections
    ANALOGICAL = "analogical"    # Find analogies
    COMBINATORIAL = "combinatorial"  # Combine existing elements

@dataclass
class CreativeIdea:
    """Representation of a creative idea."""
    id: str
    content: str
    mode: CreativeMode
    originality: float = 0.5
    feasibility: float = 0.5
    value: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    source_ideas: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

class CreativeEngine:
    """
    Creative Engine for MCP Core System.
    
    Implements various creative thinking techniques and idea generation
    methods to enhance problem-solving capabilities.
    
    Features:
    - Multiple creative modes (divergent, convergent, associative, etc.)
    - Idea generation and evaluation
    - Creative combination of existing concepts
    - Analogical reasoning
    - Serendipitous discovery mechanisms
    """
    
    def __init__(self):
        """Initialize creative engine."""
        self.logger = logging.getLogger(__name__)
        
        # Idea storage
        self.ideas = []
        self.concept_pool = []
        
        # Creative state
        self.current_mode = CreativeMode.DIVERGENT
        self.creativity_level = 0.5
        
        # Knowledge base for creative combinations
        self.knowledge_base = {
            'concepts': [],
            'relationships': [],
            'patterns': [],
            'analogies': []
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def set_creativity_level(self, level: float):
        """Set the creativity level (0.0 to 1.0)."""
        with self.lock:
            self.creativity_level = max(0.0, min(1.0, level))
    
    def set_creative_mode(self, mode: CreativeMode):
        """Set the current creative mode."""
        with self.lock:
            self.current_mode = mode
            self.logger.info(f"Creative mode set to {mode.value}")
    
    def add_concept(self, concept: str, context: Optional[Dict[str, Any]] = None):
        """Add a concept to the knowledge base."""
        with self.lock:
            concept_entry = {
                'concept': concept,
                'context': context or {},
                'added_at': datetime.now()
            }
            self.concept_pool.append(concept_entry)
            self.knowledge_base['concepts'].append(concept_entry)
            
            # Limit pool size
            if len(self.concept_pool) > 1000:
                self.concept_pool.pop(0)
    
    def generate_idea(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> CreativeIdea:
        """Generate a creative idea based on a prompt."""
        with self.lock:
            if self.current_mode == CreativeMode.DIVERGENT:
                return self._generate_divergent_idea(prompt, context)
            elif self.current_mode == CreativeMode.CONVERGENT:
                return self._generate_convergent_idea(prompt, context)
            elif self.current_mode == CreativeMode.ASSOCIATIVE:
                return self._generate_associative_idea(prompt, context)
            elif self.current_mode == CreativeMode.ANALOGICAL:
                return self._generate_analogical_idea(prompt, context)
            elif self.current_mode == CreativeMode.COMBINATORIAL:
                return self._generate_combinatorial_idea(prompt, context)
            else:
                return self._generate_divergent_idea(prompt, context)
    
    def _generate_divergent_idea(self, prompt: str, context: Optional[Dict[str, Any]]) -> CreativeIdea:
        """Generate ideas through divergent thinking."""
        # Divergent thinking: generate many varied ideas
        variations = [
            f"What if we {prompt} differently?",
            f"How might we improve {prompt}?",
            f"What are alternative approaches to {prompt}?",
            f"What would happen if we reversed {prompt}?",
            f"How could we combine {prompt} with something unexpected?"
        ]
        
        # Add randomness based on creativity level
        if self.creativity_level > 0.7:
            variations.extend([
                f"What if {prompt} was alive?",
                f"How would aliens approach {prompt}?",
                f"What if {prompt} happened in a dream?"
            ])
        
        content = random.choice(variations)
        
        idea = CreativeIdea(
            id=f"idea_{int(time.time())}_{len(self.ideas)}",
            content=content,
            mode=CreativeMode.DIVERGENT,
            originality=self.creativity_level * 0.8 + random.random() * 0.2,
            feasibility=0.6,
            value=0.5,
            context=context or {}
        )
        
        self.ideas.append(idea)
        return idea
    
    def _generate_convergent_idea(self, prompt: str, context: Optional[Dict[str, Any]]) -> CreativeIdea:
        """Generate ideas through convergent thinking."""
        # Convergent thinking: refine and focus ideas
        if len(self.ideas) >= 2:
            # Combine recent ideas
            recent_ideas = self.ideas[-3:]
            combined_content = f"Refined approach: {prompt} by combining elements from previous ideas"
            
            source_ideas = [idea.id for idea in recent_ideas]
        else:
            combined_content = f"Focused solution for {prompt}"
            source_ideas = []
        
        idea = CreativeIdea(
            id=f"idea_{int(time.time())}_{len(self.ideas)}",
            content=combined_content,
            mode=CreativeMode.CONVERGENT,
            originality=0.4,
            feasibility=0.8,
            value=0.7,
            source_ideas=source_ideas,
            context=context or {}
        )
        
        self.ideas.append(idea)
        return idea
    
    def _generate_associative_idea(self, prompt: str, context: Optional[Dict[str, Any]]) -> CreativeIdea:
        """Generate ideas through associative thinking."""
        # Associative thinking: make connections between concepts
        if self.concept_pool:
            # Select random concepts for association
            num_concepts = min(3, len(self.concept_pool))
            selected_concepts = random.sample(self.concept_pool, num_concepts)
            
            concept_names = [c['concept'] for c in selected_concepts]
            content = f"Associate {prompt} with: {', '.join(concept_names)}"
        else:
            # Default associations
            associations = ["water", "growth", "connection", "transformation", "balance"]
            selected = random.choice(associations)
            content = f"Associate {prompt} with {selected}"
        
        idea = CreativeIdea(
            id=f"idea_{int(time.time())}_{len(self.ideas)}",
            content=content,
            mode=CreativeMode.ASSOCIATIVE,
            originality=0.6,
            feasibility=0.5,
            value=0.6,
            context=context or {}
        )
        
        self.ideas.append(idea)
        return idea
    
    def _generate_analogical_idea(self, prompt: str, context: Optional[Dict[str, Any]]) -> CreativeIdea:
        """Generate ideas through analogical reasoning."""
        # Analogical thinking: find analogies and metaphors
        analogies = [
            "like a tree growing",
            "like water flowing",
            "like a symphony",
            "like a puzzle",
            "like a dance",
            "like a conversation",
            "like breathing",
            "like a journey"
        ]
        
        selected_analogy = random.choice(analogies)
        content = f"{prompt} is {selected_analogy} - what does this suggest?"
        
        idea = CreativeIdea(
            id=f"idea_{int(time.time())}_{len(self.ideas)}",
            content=content,
            mode=CreativeMode.ANALOGICAL,
            originality=0.7,
            feasibility=0.4,
            value=0.6,
            context=context or {}
        )
        
        self.ideas.append(idea)
        return idea
    
    def _generate_combinatorial_idea(self, prompt: str, context: Optional[Dict[str, Any]]) -> CreativeIdea:
        """Generate ideas through combinatorial creativity."""
        # Combinatorial thinking: combine existing elements in new ways
        if len(self.concept_pool) >= 2:
            # Combine random concepts
            concepts = random.sample(self.concept_pool, 2)
            concept1 = concepts[0]['concept']
            concept2 = concepts[1]['concept']
            
            content = f"Combine {concept1} and {concept2} to address {prompt}"
        else:
            # Default combinations
            elements = ["technology", "nature", "art", "science", "emotion", "logic"]
            element1, element2 = random.sample(elements, 2)
            content = f"Combine {element1} and {element2} to address {prompt}"
        
        idea = CreativeIdea(
            id=f"idea_{int(time.time())}_{len(self.ideas)}",
            content=content,
            mode=CreativeMode.COMBINATORIAL,
            originality=0.8,
            feasibility=0.3,
            value=0.7,
            context=context or {}
        )
        
        self.ideas.append(idea)
        return idea
    
    def evaluate_idea(self, idea: CreativeIdea) -> Dict[str, float]:
        """Evaluate an idea on multiple dimensions."""
        # Simple evaluation based on mode and content
        evaluation = {
            'originality': idea.originality,
            'feasibility': idea.feasibility,
            'value': idea.value
        }
        
        # Adjust based on mode
        if idea.mode == CreativeMode.DIVERGENT:
            evaluation['originality'] *= 1.2
            evaluation['feasibility'] *= 0.8
        elif idea.mode == CreativeMode.CONVERGENT:
            evaluation['feasibility'] *= 1.2
            evaluation['originality'] *= 0.8
        elif idea.mode == CreativeMode.ANALOGICAL:
            evaluation['originality'] *= 1.1
            evaluation['value'] *= 1.1
        
        # Normalize values
        for key in evaluation:
            evaluation[key] = max(0.0, min(1.0, evaluation[key]))
        
        return evaluation
    
    def get_best_ideas(self, limit: int = 5) -> List[CreativeIdea]:
        """Get the best ideas based on evaluation."""
        with self.lock:
            if not self.ideas:
                return []
            
            # Evaluate all ideas
            evaluated_ideas = []
            for idea in self.ideas:
                evaluation = self.evaluate_idea(idea)
                score = (evaluation['originality'] + evaluation['feasibility'] + evaluation['value']) / 3
                evaluated_ideas.append((idea, score))
            
            # Sort by score
            evaluated_ideas.sort(key=lambda x: x[1], reverse=True)
            
            return [idea for idea, score in evaluated_ideas[:limit]]
    
    def brainstorm(self, topic: str, num_ideas: int = 10) -> List[CreativeIdea]:
        """Generate multiple ideas for a topic using different modes."""
        ideas = []
        
        # Use different modes for variety
        modes = list(CreativeMode)
        
        for i in range(num_ideas):
            # Cycle through modes
            mode = modes[i % len(modes)]
            self.set_creative_mode(mode)
            
            # Generate idea
            idea = self.generate_idea(topic)
            ideas.append(idea)
        
        return ideas
    
    def find_connections(self, idea1: CreativeIdea, idea2: CreativeIdea) -> List[str]:
        """Find connections between two ideas."""
        connections = []
        
        # Simple word-based connections
        words1 = set(idea1.content.lower().split())
        words2 = set(idea2.content.lower().split())
        
        common_words = words1.intersection(words2)
        if common_words:
            connections.append(f"Common concepts: {', '.join(common_words)}")
        
        # Mode-based connections
        if idea1.mode == idea2.mode:
            connections.append(f"Both use {idea1.mode.value} thinking")
        
        # Context-based connections
        if idea1.context and idea2.context:
            common_context = set(idea1.context.keys()).intersection(set(idea2.context.keys()))
            if common_context:
                connections.append(f"Shared context: {', '.join(common_context)}")
        
        return connections
    
    def get_creative_statistics(self) -> Dict[str, Any]:
        """Get statistics about creative output."""
        with self.lock:
            if not self.ideas:
                return {'total_ideas': 0}
            
            # Count by mode
            mode_counts = {}
            for mode in CreativeMode:
                mode_counts[mode.value] = sum(1 for idea in self.ideas if idea.mode == mode)
            
            # Average scores
            total_originality = sum(idea.originality for idea in self.ideas)
            total_feasibility = sum(idea.feasibility for idea in self.ideas)
            total_value = sum(idea.value for idea in self.ideas)
            
            num_ideas = len(self.ideas)
            
            return {
                'total_ideas': num_ideas,
                'mode_distribution': mode_counts,
                'average_originality': total_originality / num_ideas,
                'average_feasibility': total_feasibility / num_ideas,
                'average_value': total_value / num_ideas,
                'creativity_level': self.creativity_level,
                'current_mode': self.current_mode.value,
                'concepts_in_pool': len(self.concept_pool)
            }

# Global creative engine instance
_creative_engine: Optional[CreativeEngine] = None

def get_creative_engine() -> CreativeEngine:
    """Get the global creative engine instance."""
    global _creative_engine
    if _creative_engine is None:
        _creative_engine = CreativeEngine()
    return _creative_engine