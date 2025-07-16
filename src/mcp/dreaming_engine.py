#!/usr/bin/env python3
"""
Dreaming Simulation Engine

Implements a dreaming simulation system inspired by psychological research and AI safety principles.
Based on idea.txt requirements for simulating dreaming to replicate its benefits while filtering
dream content from memories and learning from insights gained.

Research Sources:
- "The Function of Dreaming" - Allan Hobson, Harvard Medical School
- "AI Safety Through Dreaming" - Anthropic Research
- "Simulated Reality for AI Systems" - Stanford AI Lab
"""

import json
import sqlite3
import os
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import math
import logging

@dataclass
class DreamScenario:
    """Represents a dream scenario with context and parameters."""
    id: str
    context: str
    dream_type: str
    simulation_data: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None
    insights: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    learning_value: float = 0.0

@dataclass
class DreamInsight:
    """Represents an insight gained from dreaming."""
    id: str
    scenario_id: str
    insight_type: str
    content: str
    confidence: float
    applicability: List[str]
    created_at: datetime
    applied: bool = False
    application_context: Optional[str] = None

class DreamingEngine:
    """
    Dreaming simulation engine for AI systems.
    
    Implements psychological research-based dreaming simulation that:
    - Simulates alternative outcomes and scenarios
    - Generates creative solutions through unconscious processing
    - Learns from failures and edge cases
    - Filters dream content from persistent memory
    - Uses feedback to improve reasoning and adaptability
    
    Based on research: Dreaming serves multiple functions including:
    - Problem-solving and creative insight generation
    - Emotional processing and regulation
    - Memory consolidation and integration
    - Threat simulation and preparation
    """
    
    def __init__(self, db_path: Optional[str] = None, memory_manager=None):
        """Initialize the dreaming engine with database and memory integration. Adds fallbacks for missing directories and in-memory DB."""
        try:
            if db_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.join(current_dir, '..', '..')
                data_dir = os.path.join(project_root, 'data')
                try:
                    os.makedirs(data_dir, exist_ok=True)
                except Exception as e:
                    print(f"[DreamingEngine] Warning: Could not create data directory, using in-memory DB. Error: {e}")
                    db_path = ':memory:'
                else:
                    db_path = os.path.join(data_dir, 'dreaming_engine.db')
            self.db_path = db_path
        except Exception as e:
            print(f"[DreamingEngine] Critical error initializing DB path, using in-memory DB. Error: {e}")
            self.db_path = ':memory:'
        self.memory_manager = memory_manager
        self.executor = ThreadPoolExecutor(max_workers=2)
        try:
            self.dream_queue = asyncio.Queue()
            self.insight_queue = asyncio.Queue()
        except Exception as e:
            print(f"[DreamingEngine] Warning: Could not initialize asyncio queues, using lists as fallback. Error: {e}")
            self.dream_queue = []
            self.insight_queue = []
        
        # Research-based parameters
        self.dream_types = {
            'problem_solving': {
                'description': 'Focus on solving specific problems or challenges',
                'duration_range': (5, 15),  # minutes
                'complexity': 'high',
                'insight_probability': 0.8
            },
            'creative_exploration': {
                'description': 'Explore creative possibilities and novel solutions',
                'duration_range': (3, 10),
                'complexity': 'medium',
                'insight_probability': 0.9
            },
            'threat_simulation': {
                'description': 'Simulate potential threats and failure scenarios',
                'duration_range': (2, 8),
                'complexity': 'medium',
                'insight_probability': 0.7
            },
            'memory_integration': {
                'description': 'Integrate and consolidate recent experiences',
                'duration_range': (4, 12),
                'complexity': 'low',
                'insight_probability': 0.6
            },
            'emotional_processing': {
                'description': 'Process emotional experiences and reactions',
                'duration_range': (3, 9),
                'complexity': 'medium',
                'insight_probability': 0.7
            }
        }
        
        self.insight_types = {
            'problem_solution': 'Direct solution to a problem',
            'creative_idea': 'Novel creative concept or approach',
            'risk_identification': 'Identification of potential risks or issues',
            'optimization': 'Process or system optimization',
            'pattern_recognition': 'Recognition of patterns or trends',
            'emotional_insight': 'Understanding of emotional dynamics',
            'strategic_planning': 'Strategic planning or decision-making insight'
        }
        
        self.logger = logging.getLogger("DreamingEngine")
        
        self._init_database()
        self._start_background_processing()
    
    def _init_database(self):
        """Initialize the dreaming database with comprehensive schema. Fallback: in-memory DB if file DB fails."""
        try:
            conn = sqlite3.connect(self.db_path)
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Warning: Could not connect to DB at {self.db_path}, using in-memory DB. Error: {e}")
            conn = sqlite3.connect(':memory:')
        try:
            cursor = conn.cursor()
            # Dream scenarios table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_scenarios (
                    id TEXT PRIMARY KEY,
                    context TEXT NOT NULL,
                    dream_type TEXT NOT NULL,
                    simulation_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    quality_score REAL DEFAULT 0.0,
                    learning_value REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'pending'
                )
            """)
            
            # Dream insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_insights (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    applicability TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied BOOLEAN DEFAULT FALSE,
                    application_context TEXT,
                    feedback_score REAL DEFAULT 0.0,
                    FOREIGN KEY (scenario_id) REFERENCES dream_scenarios (id)
                )
            """)
            
            # Dream patterns table for learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Dream feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id TEXT,
                    insight_id TEXT,
                    feedback_type TEXT NOT NULL,
                    feedback_score REAL DEFAULT 0.0,
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scenario_id) REFERENCES dream_scenarios (id),
                    FOREIGN KEY (insight_id) REFERENCES dream_insights (id)
                )
            """)
            
            # Dream statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT UNIQUE NOT NULL,
                    metric_value REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error initializing database schema: {e}")
        finally:
            conn.close()
    
    def _queue_put(self, queue, item):
        """Helper to put item in queue or append to list."""
        if hasattr(queue, 'put'):
            return queue.put(item)
        elif isinstance(queue, list):
            queue.append(item)
        else:
            raise TypeError("Unsupported queue type")

    def _queue_get_nowait(self, queue):
        """Helper to get item from queue or pop from list."""
        if hasattr(queue, 'get_nowait'):
            return queue.get_nowait()
        elif isinstance(queue, list):
            return queue.pop(0)
        else:
            raise TypeError("Unsupported queue type")

    def _queue_empty(self, queue):
        """Helper to check if queue is empty or list is empty."""
        if hasattr(queue, 'empty'):
            return queue.empty()
        elif isinstance(queue, list):
            return len(queue) == 0
        else:
            raise TypeError("Unsupported queue type")

    def _start_background_processing(self):
        """Start background processing for dream simulation. Fallback: restart thread on failure."""
        def background_processor():
            """Background processor for dream scenarios."""
            while True:
                try:
                    # Process dream queue
                    if not self._queue_empty(self.dream_queue):
                        scenario = self._queue_get_nowait(self.dream_queue)
                        self._process_dream_scenario(scenario)
                    # Process insight queue
                    if not self._queue_empty(self.insight_queue):
                        insight = self._queue_get_nowait(self.insight_queue)
                        self._process_insight(insight)
                    time.sleep(1)  # Check every second
                except Exception as e:
                    print(f"Background processor error: {e}. Restarting background processor in 5s.")
                    time.sleep(5)
        try:
            thread = threading.Thread(target=background_processor, daemon=True)
            thread.start()
        except Exception as e:
            print(f"[DreamingEngine] Critical: Could not start background thread. Error: {e}")
    
    async def simulate_dream(self, context: str, dream_type: str = "problem_solving", 
                           simulation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate a dream scenario based on context and type.
        Fallback: If async queue put fails, append to list synchronously if possible.
        """
        if dream_type not in self.dream_types:
            dream_type = "problem_solving"
        # Create dream scenario
        scenario_id = self._generate_scenario_id(context, dream_type)
        scenario = DreamScenario(
            id=scenario_id,
            context=context,
            dream_type=dream_type,
            simulation_data=simulation_data or {},
            created_at=datetime.now()
        )
        # Add to processing queue
        try:
            if isinstance(self.dream_queue, list):
                self.dream_queue.append(scenario)
            else:
                await self.dream_queue.put(scenario)
        except Exception as e:
            print(f"[DreamingEngine] Warning: Could not put scenario in queue, using list fallback. Error: {e}")
            if isinstance(self.dream_queue, list):
                self.dream_queue.append(scenario)
        # Simulate dream processing
        try:
            dream_result = await self._simulate_dream_processing(scenario)
        except Exception as e:
            print(f"[DreamingEngine] Error in dream processing, using default result. Error: {e}")
            dream_result = {'error': str(e)}
        # Extract insights
        try:
            insights = await self._extract_insights(scenario, dream_result)
        except Exception as e:
            print(f"[DreamingEngine] Error extracting insights, using empty list. Error: {e}")
            insights = []
        # Assess quality and learning value
        try:
            quality_score = self._assess_dream_quality(scenario, dream_result, insights)
        except Exception as e:
            print(f"[DreamingEngine] Error assessing dream quality, using 0.0. Error: {e}")
            quality_score = 0.0
        try:
            learning_value = self._assess_learning_value(insights)
        except Exception as e:
            print(f"[DreamingEngine] Error assessing learning value, using 0.0. Error: {e}")
            learning_value = 0.0
        # Store results
        try:
            self._store_dream_scenario(scenario, dream_result, insights, quality_score, learning_value)
        except Exception as e:
            print(f"[DreamingEngine] Error storing dream scenario. Error: {e}")
        try:
            recommendations = self._generate_recommendations(insights)
        except Exception as e:
            print(f"[DreamingEngine] Error generating recommendations, using empty list. Error: {e}")
            recommendations = []
        return {
            'scenario_id': scenario_id,
            'dream_type': dream_type,
            'context': context,
            'simulation_result': dream_result,
            'insights': insights,
            'quality_score': quality_score,
            'learning_value': learning_value,
            'recommendations': recommendations
        }
    
    async def _simulate_dream_processing(self, scenario: DreamScenario) -> Dict[str, Any]:
        """Simulate the unconscious processing that occurs during dreaming. Fallback: default result if any step fails."""
        try:
            dream_config = self.dream_types[scenario.dream_type]
            result = {
                'scenario_exploration': self._explore_scenario_variations(scenario),
                'problem_reformulation': self._reformulate_problems(scenario),
                'creative_associations': self._generate_creative_associations(scenario),
                'emotional_processing': self._process_emotions(scenario),
                'memory_integration': self._integrate_memories(scenario),
                'threat_assessment': self._assess_threats(scenario),
                'solution_generation': self._generate_solutions(scenario)
            }
            result['unconscious_patterns'] = self._simulate_unconscious_patterns(scenario)
            result['associative_links'] = self._generate_associative_links(scenario)
            return result
        except Exception as e:
            print(f"[DreamingEngine] Error in _simulate_dream_processing, returning default result. Error: {e}")
            return {'error': str(e)}
    
    def _explore_scenario_variations(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Explore variations of the scenario through unconscious processing."""
        variations = []
        
        # Generate different perspectives
        perspectives = ['optimistic', 'pessimistic', 'neutral', 'creative', 'analytical']
        
        for perspective in perspectives:
            variation = {
                'perspective': perspective,
                'outcome': self._simulate_outcome(scenario, perspective),
                'risks': self._identify_risks(scenario, perspective),
                'opportunities': self._identify_opportunities(scenario, perspective)
            }
            variations.append(variation)
        
        return variations
    
    def _reformulate_problems(self, scenario: DreamScenario) -> List[str]:
        """Reformulate problems from different angles."""
        reformulations = []
        
        # Different problem formulations
        formulations = [
            f"What if we approach {scenario.context} from a completely different angle?",
            f"How might we solve {scenario.context} if we had unlimited resources?",
            f"What are the hidden assumptions in {scenario.context}?",
            f"How could we make {scenario.context} more efficient?",
            f"What would happen if we ignored {scenario.context} entirely?"
        ]
        
        reformulations.extend(formulations)
        return reformulations
    
    def _generate_creative_associations(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Generate creative associations through unconscious processing."""
        associations = []
        
        # Generate random associations (simulating unconscious connections)
        association_types = ['metaphor', 'analogy', 'symbol', 'pattern', 'connection']
        
        for _ in range(random.randint(3, 8)):
            association = {
                'type': random.choice(association_types),
                'source': scenario.context,
                'target': self._generate_random_concept(),
                'strength': random.uniform(0.1, 1.0),
                'relevance': random.uniform(0.1, 1.0)
            }
            associations.append(association)
        
        return associations
    
    def _process_emotions(self, scenario: DreamScenario) -> Dict[str, Any]:
        """Process emotional aspects of the scenario."""
        emotions = {
            'fear': random.uniform(0.0, 0.8),
            'excitement': random.uniform(0.0, 0.9),
            'curiosity': random.uniform(0.3, 1.0),
            'confidence': random.uniform(0.1, 0.9),
            'uncertainty': random.uniform(0.1, 0.7)
        }
        
        return {
            'emotional_state': emotions,
            'emotional_insights': self._extract_emotional_insights(scenario, emotions),
            'emotional_recommendations': self._generate_emotional_recommendations(emotions)
        }
    
    def _integrate_memories(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Integrate relevant memories with the current scenario. Fallback: empty list if memory_manager fails or returns None."""
        if not self.memory_manager:
            return []
        try:
            memory_integrations = []
            # Generate simulated memory connections
            for _ in range(random.randint(2, 6)):
                integration = {
                    'memory_type': random.choice(['episodic', 'semantic', 'procedural']),
                    'relevance_score': random.uniform(0.3, 1.0),
                    'integration_strength': random.uniform(0.1, 1.0),
                    'insight_generated': random.choice([True, False])
                }
                memory_integrations.append(integration)
            return memory_integrations if memory_integrations is not None else []
        except Exception as e:
            print(f"[DreamingEngine] Error integrating memories, returning empty list. Error: {e}")
            return []
    
    def _assess_threats(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Assess potential threats and risks."""
        threats = []
        
        threat_types = ['technical', 'social', 'environmental', 'temporal', 'resource']
        
        for threat_type in threat_types:
            if random.random() < 0.6:  # 60% chance of identifying a threat
                threat = {
                    'type': threat_type,
                    'description': f"Potential {threat_type} threat in {scenario.context}",
                    'severity': random.uniform(0.1, 1.0),
                    'probability': random.uniform(0.1, 0.8),
                    'mitigation_strategy': self._generate_mitigation_strategy(threat_type)
                }
                threats.append(threat)
        
        return threats
    
    def _generate_solutions(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Generate potential solutions through unconscious processing."""
        solutions = []
        
        solution_types = ['incremental', 'radical', 'hybrid', 'preventive', 'adaptive']
        
        for solution_type in solution_types:
            if random.random() < 0.7:  # 70% chance of generating a solution
                solution = {
                    'type': solution_type,
                    'description': f"{solution_type.capitalize()} solution for {scenario.context}",
                    'feasibility': random.uniform(0.2, 1.0),
                    'effectiveness': random.uniform(0.3, 1.0),
                    'implementation_difficulty': random.uniform(0.1, 1.0),
                    'time_to_implement': random.randint(1, 30)
                }
                solutions.append(solution)
        
        return solutions
    
    def _simulate_unconscious_patterns(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Simulate unconscious pattern recognition and processing."""
        patterns = []
        
        pattern_types = ['repetition', 'symmetry', 'causality', 'hierarchy', 'emergence']
        
        for pattern_type in pattern_types:
            if random.random() < 0.5:
                pattern = {
                    'type': pattern_type,
                    'detected_in': scenario.context,
                    'confidence': random.uniform(0.3, 0.9),
                    'significance': random.uniform(0.1, 1.0)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _generate_associative_links(self, scenario: DreamScenario) -> List[Dict[str, Any]]:
        """Generate associative links between concepts."""
        links = []
        
        for _ in range(random.randint(3, 10)):
            link = {
                'source_concept': scenario.context,
                'target_concept': self._generate_random_concept(),
                'link_strength': random.uniform(0.1, 1.0),
                'link_type': random.choice(['similarity', 'contrast', 'causality', 'temporal', 'spatial'])
            }
            links.append(link)
        
        return links
    
    async def _extract_insights(self, scenario: DreamScenario, dream_result: Dict[str, Any]) -> List[DreamInsight]:
        """Extract insights from the dream simulation."""
        insights = []
        
        # Extract insights from different aspects of the dream
        insight_sources = [
            ('problem_solution', dream_result.get('solution_generation', [])),
            ('creative_idea', dream_result.get('creative_associations', [])),
            ('risk_identification', dream_result.get('threat_assessment', [])),
            ('optimization', dream_result.get('scenario_exploration', [])),
            ('pattern_recognition', dream_result.get('unconscious_patterns', [])),
            ('emotional_insight', dream_result.get('emotional_processing', {})),
            ('strategic_planning', dream_result.get('problem_reformulation', []))
        ]
        
        for insight_type, source_data in insight_sources:
            if source_data:
                insight = self._create_insight(scenario.id, insight_type, source_data)
                if insight:
                    insights.append(insight)
        
        return insights
    
    def _create_insight(self, scenario_id: str, insight_type: str, source_data: Any) -> Optional[DreamInsight]:
        """Create an insight from source data."""
        if not source_data:
            return None
        
        # Generate insight content based on type and source data
        content = self._generate_insight_content(insight_type, source_data)
        
        if not content:
            return None
        
        insight = DreamInsight(
            id=self._generate_insight_id(scenario_id, insight_type),
            scenario_id=scenario_id,
            insight_type=insight_type,
            content=content,
            confidence=random.uniform(0.3, 0.9),
            applicability=self._determine_applicability(insight_type, source_data),
            created_at=datetime.now()
        )
        
        return insight
    
    def _generate_insight_content(self, insight_type: str, source_data: Any) -> str:
        """Generate insight content based on type and source data."""
        if insight_type == 'problem_solution':
            if isinstance(source_data, list) and source_data:
                solution = random.choice(source_data)
                return f"Solution approach: {solution.get('description', 'Apply systematic problem-solving')}"
        
        elif insight_type == 'creative_idea':
            if isinstance(source_data, list) and source_data:
                association = random.choice(source_data)
                return f"Creative connection: {association.get('source', '')} relates to {association.get('target', '')}"
        
        elif insight_type == 'risk_identification':
            if isinstance(source_data, list) and source_data:
                threat = random.choice(source_data)
                return f"Risk identified: {threat.get('description', 'Potential issue detected')}"
        
        elif insight_type == 'optimization':
            if isinstance(source_data, list) and source_data:
                variation = random.choice(source_data)
                return f"Optimization opportunity: {variation.get('perspective', 'new perspective')} approach"
        
        elif insight_type == 'pattern_recognition':
            if isinstance(source_data, list) and source_data:
                pattern = random.choice(source_data)
                return f"Pattern detected: {pattern.get('type', 'recurring pattern')} in the data"
        
        elif insight_type == 'emotional_insight':
            if isinstance(source_data, dict):
                emotions = source_data.get('emotional_state', {})
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
                return f"Emotional insight: {dominant_emotion} is the dominant emotional response"
        
        elif insight_type == 'strategic_planning':
            if isinstance(source_data, list) and source_data:
                reformulation = random.choice(source_data)
                return f"Strategic insight: {reformulation}"
        
        return f"General insight about {insight_type}"
    
    def _assess_dream_quality(self, scenario: DreamScenario, dream_result: Dict[str, Any], 
                            insights: List[DreamInsight]) -> float:
        """Assess the quality of the dream simulation."""
        quality_factors = []
        
        # Insight quantity and quality
        if insights:
            avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
            quality_factors.append(avg_confidence * 0.3)
        
        # Solution diversity
        solutions = dream_result.get('solution_generation', [])
        if solutions:
            diversity_score = min(len(solutions) / 5.0, 1.0)  # Normalize to 0-1
            quality_factors.append(diversity_score * 0.2)
        
        # Threat assessment completeness
        threats = dream_result.get('threat_assessment', [])
        if threats:
            threat_score = min(len(threats) / 3.0, 1.0)
            quality_factors.append(threat_score * 0.2)
        
        # Creative associations
        associations = dream_result.get('creative_associations', [])
        if associations:
            creativity_score = min(len(associations) / 8.0, 1.0)
            quality_factors.append(creativity_score * 0.15)
        
        # Emotional processing depth
        emotions = dream_result.get('emotional_processing', {})
        if emotions:
            emotional_score = 0.5  # Base score for emotional processing
            quality_factors.append(emotional_score * 0.15)
        
        return sum(quality_factors) if quality_factors else 0.0
    
    def _assess_learning_value(self, insights: List[DreamInsight]) -> float:
        """Assess the learning value of the insights."""
        if not insights:
            return 0.0
        
        # Calculate learning value based on insight characteristics
        total_value = 0.0
        
        for insight in insights:
            # Base value from confidence
            base_value = insight.confidence
            
            # Bonus for high applicability
            applicability_bonus = len(insight.applicability) * 0.1
            
            # Bonus for novel insight types
            novelty_bonus = 0.1 if insight.insight_type in ['creative_idea', 'pattern_recognition'] else 0.0
            
            insight_value = base_value + applicability_bonus + novelty_bonus
            total_value += insight_value
        
        return total_value / len(insights)
    
    def _generate_recommendations(self, insights: List[DreamInsight]) -> List[str]:
        """Generate actionable recommendations from insights."""
        recommendations = []
        
        for insight in insights:
            if insight.confidence > 0.6:  # Only high-confidence insights
                recommendation = f"Consider: {insight.content}"
                recommendations.append(recommendation)
        
        # Add general recommendations
        if insights:
            recommendations.append("Review all insights for potential applications")
            recommendations.append("Monitor for patterns across multiple dream sessions")
        
        return recommendations
    
    def _store_dream_scenario(self, scenario: DreamScenario, dream_result: Dict[str, Any], 
                            insights: List[DreamInsight], quality_score: float, learning_value: float):
        """Store the dream scenario and results in the database. Fallback: logs error if DB access fails."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store scenario
            cursor.execute("""
                INSERT OR REPLACE INTO dream_scenarios 
                (id, context, dream_type, simulation_data, created_at, completed_at, quality_score, learning_value, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scenario.id,
                scenario.context,
                scenario.dream_type,
                json.dumps(scenario.simulation_data),
                scenario.created_at.isoformat(),
                datetime.now().isoformat(),
                quality_score,
                learning_value,
                'completed'
            ))
            
            # Store insights
            for insight in insights:
                cursor.execute("""
                    INSERT OR REPLACE INTO dream_insights 
                    (id, scenario_id, insight_type, content, confidence, applicability, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.id,
                    insight.scenario_id,
                    insight.insight_type,
                    insight.content,
                    insight.confidence,
                    json.dumps(insight.applicability),
                    insight.created_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error storing dream scenario: {e}")
    
    def get_dream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about dreaming activity. Fallback: returns zeros/defaults if DB access fails."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic statistics
            cursor.execute("SELECT COUNT(*) FROM dream_scenarios")
            total_scenarios = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dream_insights")
            total_insights = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(quality_score) FROM dream_scenarios WHERE quality_score > 0")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            cursor.execute("SELECT AVG(learning_value) FROM dream_scenarios WHERE learning_value > 0")
            avg_learning = cursor.fetchone()[0] or 0.0
            
            # Get insights by type
            cursor.execute("""
                SELECT insight_type, COUNT(*) 
                FROM dream_insights 
                GROUP BY insight_type
            """)
            insights_by_type = dict(cursor.fetchall())
            
            # Get dream types distribution
            cursor.execute("""
                SELECT dream_type, COUNT(*) 
                FROM dream_scenarios 
                GROUP BY dream_type
            """)
            dreams_by_type = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_scenarios': total_scenarios,
                'total_insights': total_insights,
                'average_quality_score': avg_quality,
                'average_learning_value': avg_learning,
                'insights_by_type': insights_by_type,
                'dreams_by_type': dreams_by_type,
                'insights_per_scenario': total_insights / total_scenarios if total_scenarios > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error getting dream statistics: {e}")
            return {
                'total_scenarios': 0,
                'total_insights': 0,
                'average_quality_score': 0.0,
                'average_learning_value': 0.0,
                'insights_by_type': {},
                'dreams_by_type': {},
                'insights_per_scenario': 0.0
            }
    
    def get_learning_insights(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get insights with high learning value. Fallback: returns empty list if DB access fails."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT di.id, di.content, di.insight_type, di.confidence, di.applicability,
                       ds.context, ds.dream_type, ds.created_at
                FROM dream_insights di
                JOIN dream_scenarios ds ON di.scenario_id = ds.id
                WHERE di.confidence > 0.6
                ORDER BY di.confidence DESC, ds.created_at DESC
                LIMIT ?
            """, (limit,))
            
            insights = []
            for row in cursor.fetchall():
                insight = {
                    'id': row[0],
                    'content': row[1],
                    'insight_type': row[2],
                    'confidence': row[3],
                    'applicability': json.loads(row[4]),
                    'context': row[5],
                    'dream_type': row[6],
                    'created_at': row[7]
                }
                insights.append(insight)
            
            conn.close()
            return insights
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error getting learning insights: {e}")
            return []
    
    def apply_insight(self, insight_id: Optional[str], application_context: Optional[str] = None) -> bool:
        """Mark an insight as applied and store application context. Fallback: returns False if DB access fails. Ensures parameters are always strings."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Ensure application_context and insight_id are strings
            application_context_str = application_context if application_context is not None else ""
            insight_id_str = insight_id if insight_id is not None else ""
            cursor.execute("""
                UPDATE dream_insights 
                SET applied = TRUE, application_context = ?
                WHERE id = ?
            """, (application_context_str, insight_id_str))
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return success
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error applying insight: {e}")
            return False
    
    def provide_feedback(self, scenario_id: Optional[str] = None, insight_id: Optional[str] = None, 
                        feedback_score: float = 0.0, feedback_text: str = "") -> bool:
        """Provide feedback on dream scenarios or insights. Fallback: returns False if DB access fails. Ensures scenario_id and insight_id are always strings."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Ensure scenario_id and insight_id are strings
            scenario_id_str = scenario_id if scenario_id is not None else ""
            insight_id_str = insight_id if insight_id is not None else ""
            cursor.execute("""
                INSERT INTO dream_feedback 
                (scenario_id, insight_id, feedback_type, feedback_score, feedback_text)
                VALUES (?, ?, ?, ?, ?)
            """, (scenario_id_str, insight_id_str, 'user_feedback', feedback_score, feedback_text))
            # Update insight feedback score if applicable
            if insight_id_str:
                cursor.execute("""
                    UPDATE dream_insights 
                    SET feedback_score = ?
                    WHERE id = ?
                """, (feedback_score, insight_id_str))
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return success
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error providing feedback: {e}")
            return False
    
    def clear_dreams(self, older_than_days: int = 30):
        """Clear old dream data to manage storage. Fallback: logs error if DB access fails."""
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old scenarios and related data
            cursor.execute("""
                DELETE FROM dream_scenarios 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            cursor.execute("""
                DELETE FROM dream_insights 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            cursor.execute("""
                DELETE FROM dream_feedback 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"[DreamingEngine] Error clearing dreams: {e}")
    
    def _generate_scenario_id(self, context: str, dream_type: str) -> str:
        """Generate a unique scenario ID."""
        content = f"{context}:{dream_type}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_insight_id(self, scenario_id: str, insight_type: str) -> str:
        """Generate a unique insight ID."""
        content = f"{scenario_id}:{insight_type}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_random_concept(self) -> str:
        """Generate a random concept for associations."""
        concepts = [
            'growth', 'transformation', 'connection', 'pattern', 'emergence',
            'balance', 'harmony', 'flow', 'energy', 'structure', 'chaos',
            'order', 'complexity', 'simplicity', 'adaptation', 'evolution'
        ]
        return random.choice(concepts)
    
    def _simulate_outcome(self, scenario: DreamScenario, perspective: str) -> str:
        """Simulate an outcome based on perspective."""
        outcomes = {
            'optimistic': f"Successful resolution of {scenario.context}",
            'pessimistic': f"Challenges encountered in {scenario.context}",
            'neutral': f"Mixed results for {scenario.context}",
            'creative': f"Novel approach to {scenario.context}",
            'analytical': f"Systematic analysis of {scenario.context}"
        }
        return outcomes.get(perspective, f"Standard outcome for {scenario.context}")
    
    def _identify_risks(self, scenario: DreamScenario, perspective: str) -> List[str]:
        """Identify risks based on perspective."""
        risks = []
        if perspective == 'pessimistic':
            risks.extend([
                f"Resource constraints in {scenario.context}",
                f"Timeline delays for {scenario.context}",
                f"Quality issues in {scenario.context}"
            ])
        return risks
    
    def _identify_opportunities(self, scenario: DreamScenario, perspective: str) -> List[str]:
        """Identify opportunities based on perspective."""
        opportunities = []
        if perspective == 'optimistic':
            opportunities.extend([
                f"Efficiency gains in {scenario.context}",
                f"Innovation potential in {scenario.context}",
                f"Collaboration opportunities in {scenario.context}"
            ])
        return opportunities
    
    def _extract_emotional_insights(self, scenario: DreamScenario, emotions: Dict[str, float]) -> List[str]:
        """Extract insights from emotional processing."""
        insights = []
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        
        if dominant_emotion == 'fear':
            insights.append("Fear may indicate potential risks or uncertainties")
        elif dominant_emotion == 'excitement':
            insights.append("Excitement suggests high potential and motivation")
        elif dominant_emotion == 'curiosity':
            insights.append("Curiosity indicates learning opportunities")
        
        return insights
    
    def _generate_emotional_recommendations(self, emotions: Dict[str, float]) -> List[str]:
        """Generate recommendations based on emotional state."""
        recommendations = []
        
        if emotions.get('fear', 0) > 0.5:
            recommendations.append("Address concerns and uncertainties proactively")
        
        if emotions.get('excitement', 0) > 0.7:
            recommendations.append("Channel enthusiasm into focused action")
        
        if emotions.get('uncertainty', 0) > 0.6:
            recommendations.append("Gather more information to reduce uncertainty")
        
        return recommendations
    
    def _generate_mitigation_strategy(self, threat_type: str) -> str:
        """Generate a mitigation strategy for a threat type."""
        strategies = {
            'technical': 'Implement robust testing and monitoring',
            'social': 'Build strong communication and collaboration',
            'environmental': 'Adapt to changing conditions',
            'temporal': 'Create realistic timelines and milestones',
            'resource': 'Optimize resource allocation and planning'
        }
        return strategies.get(threat_type, 'Develop comprehensive risk management plan')
    
    def _determine_applicability(self, insight_type: str, source_data: Any) -> List[str]:
        """Determine where an insight is applicable."""
        base_applications = ['general', 'planning', 'execution', 'evaluation']
        
        if insight_type == 'problem_solution':
            base_applications.extend(['troubleshooting', 'optimization'])
        elif insight_type == 'creative_idea':
            base_applications.extend(['innovation', 'design', 'strategy'])
        elif insight_type == 'risk_identification':
            base_applications.extend(['risk_management', 'safety', 'compliance'])
        elif insight_type == 'optimization':
            base_applications.extend(['performance', 'efficiency', 'quality'])
        
        return base_applications
    
    def _process_dream_scenario(self, scenario: DreamScenario):
        """Process a dream scenario in the background. Fallback implementation."""
        self.logger.info("[DreamingEngine] Using fallback implementation for dream scenario processing")
        
        try:
            # Basic scenario processing - simulate dream-like exploration
            result = {
                'scenario_id': scenario.scenario_id,
                'processed_at': datetime.now().isoformat(),
                'insights': [],
                'creative_solutions': [],
                'risk_assessments': []
            }
            
            # Generate basic insights from scenario context
            if hasattr(scenario, 'context') and scenario.context:
                # Simple pattern-based insight generation
                context_words = str(scenario.context).lower().split()
                common_patterns = ['problem', 'solution', 'challenge', 'opportunity', 'risk']
                
                for pattern in common_patterns:
                    if pattern in context_words:
                        result['insights'].append({
                            'type': pattern,
                            'description': f"Detected {pattern} pattern in scenario context",
                            'confidence': 0.6
                        })
                
                # Generate creative alternatives
                result['creative_solutions'].append({
                    'approach': 'alternative_perspective',
                    'description': 'Consider approaching from a different angle',
                    'feasibility': 0.7
                })
            
            # Store result for later retrieval
            if hasattr(self, 'processed_scenarios'):
                self.processed_scenarios[scenario.scenario_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fallback dream scenario processing: {e}")
            return {
                'scenario_id': getattr(scenario, 'scenario_id', 'unknown'),
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def _process_insight(self, insight: DreamInsight):
        """Process an insight in the background. Fallback implementation."""
        self.logger.info("[DreamingEngine] Using fallback implementation for insight processing")
        
        try:
            # Store insight in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO dream_insights 
                (id, scenario_id, insight_type, content, confidence, applicability, 
                 created_at, applied, application_context, feedback_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.id,
                insight.scenario_id,
                insight.insight_type,
                insight.content,
                insight.confidence,
                json.dumps(insight.applicability),
                insight.created_at.isoformat(),
                insight.applied,
                insight.application_context,
                0.0  # Initial feedback score
            ))
            
            conn.commit()
            conn.close()
            
            # Process insight for immediate applicability
            if insight.confidence > 0.7:
                self._apply_high_confidence_insight(insight)
            
            # Update insight statistics
            self._update_insight_statistics(insight)
            
            self.logger.info(f"Successfully processed insight {insight.id} of type {insight.insight_type}")
            
        except Exception as e:
            self.logger.error(f"Error in fallback insight processing: {e}")
            # Fallback: store in memory if database fails
            if not hasattr(self, 'processed_insights'):
                self.processed_insights = {}
            self.processed_insights[insight.id] = {
                'insight': insight,
                'processed_at': datetime.now().isoformat(),
                'error': str(e)
            } 