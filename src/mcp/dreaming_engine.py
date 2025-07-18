"""
DreamingEngine - Advanced scenario simulation and creative insight generation.

This module implements comprehensive dream cycle initiation, scenario simulation,
insight extraction, and dream contamination filtering to protect memory systems.
Part of the MCP Agentic Workflow Accelerator's brain-inspired architecture.
"""

import asyncio
import json
import logging
import random
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from .hormone_system_integration import HormoneSystem
from .genetic_trigger_system import GeneticTrigger


class DreamScenario:
    """Represents a dream scenario with context and simulation parameters."""
    
    def __init__(self, scenario_id: str, context: str, dream_type: str = 'exploration'):
        self.scenario_id = scenario_id
        self.context = context
        self.dream_type = dream_type
        self.created_at = datetime.now()
        self.simulation_data = {}
        self.quality_score = 0.0
        self.learning_value = 0.0
        self.completed = False


class DreamInsight:
    """Represents an insight extracted from dream simulation."""
    
    def __init__(self, insight_id: str, content: str, insight_type: str = 'general'):
        self.insight_id = insight_id
        self.content = content
        self.insight_type = insight_type
        self.confidence = 0.0
        self.applicability = []
        self.created_at = datetime.now()


class DreamingEngine:
    """
    Advanced dreaming engine for scenario simulation and insight generation.
    
    Implements comprehensive dream cycles with contamination filtering to protect
    memory systems while generating creative insights and alternative scenarios.
    """
    
    def __init__(self, db_path: str = "data/dreaming_engine.db", 
                 hormone_system: Optional[HormoneSystem] = None):
        self.db_path = db_path
        self.hormone_system = hormone_system
        self.logger = logging.getLogger(__name__)
        self.genetic_trigger = GeneticTrigger()
        
        # Dream state management
        self.active_dreams = {}
        self.dream_contamination_threshold = 0.3
        self.max_concurrent_dreams = 5
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize the dreaming engine database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create dream scenarios table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_scenarios (
                    id TEXT PRIMARY KEY,
                    context TEXT NOT NULL,
                    dream_type TEXT DEFAULT 'exploration',
                    simulation_data TEXT,
                    quality_score REAL DEFAULT 0.0,
                    learning_value REAL DEFAULT 0.0,
                    completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create dream insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_insights (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT,
                    content TEXT NOT NULL,
                    insight_type TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 0.0,
                    applicability TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scenario_id) REFERENCES dream_scenarios (id)
                )
            """)
            
            # Create meta-insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta_insights (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    practical_value REAL DEFAULT 0.0,
                    cluster_size INTEGER DEFAULT 0,
                    constituent_insights TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing dreaming engine database: {e}")
            
    async def initiate_dream_cycle(self, context: str, dream_type: str = 'exploration') -> Dict[str, Any]:
        """
        Initiate a comprehensive dream cycle with scenario simulation.
        
        Args:
            context: The context or prompt for the dream
            dream_type: Type of dream ('exploration', 'problem_solving', 'creative')
            
        Returns:
            Dict containing dream results and insights
        """
        try:
            # Check if we can start a new dream
            if len(self.active_dreams) >= self.max_concurrent_dreams:
                return {'status': 'error', 'message': 'Maximum concurrent dreams reached'}
                
            # Generate scenario
            scenario = await self._generate_dream_scenario(context, dream_type)
            self.active_dreams[scenario.scenario_id] = scenario
            
            # Release dreaming hormones
            if self.hormone_system:
                await self.hormone_system.release_hormone('melatonin', 0.8, 'dreaming_initiation')
                
            # Simulate the dream scenario
            simulation_result = await self._simulate_dream_scenario(scenario)
            
            # Extract insights
            insights = await self._extract_dream_insights(scenario, simulation_result)
            
            # Filter contamination
            filtered_insights = await self._filter_dream_contamination(insights)
            
            # Generate meta-insights
            meta_insights = await self._generate_meta_insights(filtered_insights)
            
            # Store results
            await self._store_dream_results(scenario, filtered_insights, meta_insights)
            
            # Update memory systems
            await self._update_memory_systems(filtered_insights, meta_insights)
            
            # Clean up
            del self.active_dreams[scenario.scenario_id]
            
            return {
                'status': 'success',
                'scenario_id': scenario.scenario_id,
                'insights_count': len(filtered_insights),
                'meta_insights_count': len(meta_insights),
                'quality_score': scenario.quality_score,
                'learning_value': scenario.learning_value
            }
            
        except Exception as e:
            self.logger.error(f"Error in dream cycle: {e}")
            return {'status': 'error', 'message': str(e)}
            
    async def _generate_dream_scenario(self, context: str, dream_type: str) -> DreamScenario:
        """Generate a dream scenario based on context and type."""
        scenario_id = self._generate_scenario_id(context, dream_type)
        scenario = DreamScenario(scenario_id, context, dream_type)
        
        # Add sensory details and narrative structure
        scenario.simulation_data = {
            'sensory_details': self._add_sensory_details({}),
            'narrative_structure': self._create_dream_narrative({}),
            'symbolic_content': self._generate_symbolic_content({}),
            'emotional_tone': random.uniform(0.3, 0.9)
        }
        
        return scenario
        
    async def _simulate_dream_scenario(self, scenario: DreamScenario) -> Dict[str, Any]:
        """Simulate the dream scenario and generate outcomes."""
        simulation_result = {
            'outcomes': [],
            'emotional_responses': {},
            'symbolic_interpretations': [],
            'narrative_progression': []
        }
        
        # Generate multiple possible outcomes
        for i in range(3):  # Generate 3 alternative outcomes
            outcome = self._simulate_outcome(scenario, f'outcome_{i}')
            simulation_result['outcomes'].append(outcome)
            
        # Add emotional processing
        simulation_result['emotional_responses'] = self._process_dream_emotions(scenario)
        
        # Generate symbolic interpretations
        simulation_result['symbolic_interpretations'] = self._interpret_dream_symbols(scenario)
        
        return simulation_result
        
    async def _extract_dream_insights(self, scenario: DreamScenario, 
                                    simulation_result: Dict[str, Any]) -> List[DreamInsight]:
        """Extract actionable insights from dream simulation."""
        insights = []
        
        # Extract insights from outcomes
        for outcome in simulation_result['outcomes']:
            insight_content = f"Alternative approach: {outcome.get('description', '')}"
            insight = DreamInsight(
                self._generate_insight_id(scenario.scenario_id, 'outcome'),
                insight_content,
                'alternative_solution'
            )
            insight.confidence = outcome.get('confidence', 0.5)
            insights.append(insight)
            
        # Extract emotional insights
        for emotion, value in simulation_result['emotional_responses'].items():
            if value > 0.7:  # High emotional significance
                insight_content = f"Strong emotional response to {emotion}: {value}"
                insight = DreamInsight(
                    self._generate_insight_id(scenario.scenario_id, emotion),
                    insight_content,
                    'emotional_pattern'
                )
                insight.confidence = value
                insights.append(insight)
                
        # Extract symbolic insights
        for symbol_interpretation in simulation_result['symbolic_interpretations']:
            insight = DreamInsight(
                self._generate_insight_id(scenario.scenario_id, 'symbolic'),
                symbol_interpretation,
                'symbolic_meaning'
            )
            insight.confidence = 0.6  # Symbolic insights have moderate confidence
            insights.append(insight)
            
        return insights
        
    async def _filter_dream_contamination(self, insights: List[DreamInsight]) -> List[DreamInsight]:
        """Filter out contaminated ights to pect memct memory systems.
        safe_inseghts = []
        contaminaghts = [] = []
        
        for insight in insights:
        def contamination_score ():assess_cont(insight)
            
            if contamination_score < self.dream_contamination_threshold:
                safe_insights.append(insight)
            else:
                contaminated_insights.append(insigdream_queue):
                
        # Log conta     self._ltering
        if con     ated_insights:ight queue
            self.logger.info(f"Filtered {len(contaminaght_queue):inated insights")
                        insight = self._queue_get_nowait(self.insight_queue)
        return safe_insights._process_insight(insight)
                    time.sleep(1)  # Check every second
    def _a      contamination(self, insight: DreamInsight) -> float:
        """Assess the int(f"Background processor error: {e}. Restarting background processor in 5s.")
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
        
        return total_value / len(insights) if insights else 0.0
    
    # ===== NEW COMPREHENSIVE UPGRADE FEATURES =====
    
    def initiate_dream_cycle(self, trigger_context: Dict[str, Any] = None, 
                           cycle_type: str = "natural") -> Dict[str, Any]:
        """
        Initiate a comprehensive dream cycle with multiple phases.
        
        Args:
            trigger_context: Context that triggered the dream cycle
            cycle_type: Type of cycle ('natural', 'induced', 'therapeutic', 'creative')
        
        Returns:
            Dict containing cycle results and insights
        """
        try:
            cycle_id = self._generate_cycle_id(trigger_context, cycle_type)
            
            # Phase 1: Pre-dream preparation
            preparation_result = self._prepare_dream_cycle(trigger_context, cycle_type)
            
            # Phase 2: Dream initiation and REM simulation
            rem_result = self._simulate_rem_phase(preparation_result)
            
            # Phase 3: Deep dream processing
            deep_dream_result = self._process_deep_dreams(rem_result)
            
            # Phase 4: Dream consolidation and insight extraction
            consolidation_result = self._consolidate_dream_insights(deep_dream_result)
            
            # Phase 5: Contamination filtering
            filtered_result = self._filter_dream_contamination(consolidation_result)
            
            # Phase 6: Memory integration (safe insights only)
            integration_result = self._integrate_safe_insights(filtered_result)
            
            # Store complete cycle
            cycle_result = {
                'cycle_id': cycle_id,
                'cycle_type': cycle_type,
                'trigger_context': trigger_context,
                'phases': {
                    'preparation': preparation_result,
                    'rem_simulation': rem_result,
                    'deep_processing': deep_dream_result,
                    'consolidation': consolidation_result,
                    'contamination_filtering': filtered_result,
                    'memory_integration': integration_result
                },
                'total_insights': len(filtered_result.get('safe_insights', [])),
                'contamination_filtered': len(consolidation_result.get('insights', [])) - len(filtered_result.get('safe_insights', [])),
                'cycle_quality': self._assess_cycle_quality(filtered_result),
                'completed_at': datetime.now()
            }
            
            self._store_dream_cycle(cycle_result)
            
            # Release melatonin to signal dream completion
            if hasattr(self, 'hormone_integration'):
                self.hormone_integration.emit_event(
                    "dream_cycle_completed",
                    {"cycle_id": cycle_id, "quality": cycle_result['cycle_quality']},
                    context={"source": "dreaming_engine"}
                )
            
            return cycle_result
            
        except Exception as e:
            self.logger.error(f"Error in dream cycle initiation: {e}")
            return {
                'error': str(e),
                'cycle_id': None,
                'cycle_type': cycle_type,
                'status': 'failed'
            }
    
    def _prepare_dream_cycle(self, trigger_context: Dict[str, Any], 
                           cycle_type: str) -> Dict[str, Any]:
        """Prepare for dream cycle by gathering relevant memories and context."""
        try:
            # Gather recent memories and experiences
            recent_memories = self._gather_recent_memories(hours=24)
            
            # Identify unresolved problems or tensions
            unresolved_issues = self._identify_unresolved_issues(trigger_context)
            
            # Select dream themes based on context and cycle type
            dream_themes = self._select_dream_themes(trigger_context, cycle_type, unresolved_issues)
            
            # Prepare dream environment parameters
            environment_params = self._prepare_dream_environment(cycle_type)
            
            return {
                'recent_memories': recent_memories,
                'unresolved_issues': unresolved_issues,
                'dream_themes': dream_themes,
                'environment_params': environment_params,
                'preparation_quality': self._assess_preparation_quality(recent_memories, unresolved_issues)
            }
            
        except Exception as e:
            self.logger.error(f"Error in dream cycle preparation: {e}")
            return {'error': str(e), 'preparation_quality': 0.0}
    
    def _simulate_rem_phase(self, preparation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate REM sleep phase with rapid scenario generation."""
        try:
            rem_scenarios = []
            dream_themes = preparation_result.get('dream_themes', [])
            
            # Generate multiple rapid scenarios (REM characteristic)
            for theme in dream_themes:
                for _ in range(random.randint(3, 7)):  # Multiple scenarios per theme
                    scenario = self._generate_rem_scenario(theme, preparation_result)
                    rem_scenarios.append(scenario)
            
            # Process scenarios in parallel (simulating rapid REM processing)
            processed_scenarios = []
            for scenario in rem_scenarios:
                processed = self._process_rem_scenario(scenario)
                processed_scenarios.append(processed)
            
            return {
                'rem_scenarios': processed_scenarios,
                'scenario_count': len(processed_scenarios),
                'rem_intensity': self._calculate_rem_intensity(processed_scenarios),
                'emotional_processing': self._extract_rem_emotions(processed_scenarios)
            }
            
        except Exception as e:
            self.logger.error(f"Error in REM phase simulation: {e}")
            return {'error': str(e), 'rem_scenarios': []}
    
    def _process_deep_dreams(self, rem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process deep dream scenarios with enhanced creativity and problem-solving."""
        try:
            deep_dreams = []
            rem_scenarios = rem_result.get('rem_scenarios', [])
            
            # Select most promising REM scenarios for deep processing
            selected_scenarios = self._select_scenarios_for_deep_processing(rem_scenarios)
            
            for scenario in selected_scenarios:
                deep_dream = self._create_deep_dream(scenario)
                
                # Enhanced processing for deep dreams
                deep_dream['creative_expansions'] = self._expand_creatively(deep_dream)
                deep_dream['problem_solutions'] = self._extract_problem_solutions(deep_dream)
                deep_dream['symbolic_meanings'] = self._extract_symbolic_meanings(deep_dream)
                deep_dream['emotional_insights'] = self._extract_deep_emotions(deep_dream)
                
                deep_dreams.append(deep_dream)
            
            return {
                'deep_dreams': deep_dreams,
                'processing_depth': self._calculate_processing_depth(deep_dreams),
                'creative_breakthroughs': self._identify_creative_breakthroughs(deep_dreams),
                'problem_solving_insights': self._extract_problem_solving_insights(deep_dreams)
            }
            
        except Exception as e:
            self.logger.error(f"Error in deep dream processing: {e}")
            return {'error': str(e), 'deep_dreams': []}
    
    def _consolidate_dream_insights(self, deep_dream_result: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate insights from all dream phases."""
        try:
            all_insights = []
            deep_dreams = deep_dream_result.get('deep_dreams', [])
            
            # Extract insights from each deep dream
            for dream in deep_dreams:
                insights = self._extract_consolidated_insights(dream)
                all_insights.extend(insights)
            
            # Rank insights by potential value
            ranked_insights = self._rank_insights_by_value(all_insights)
            
            # Group related insights
            insight_clusters = self._cluster_related_insights(ranked_insights)
            
            # Generate meta-insights from clusters
            meta_insights = self._generate_meta_insights(insight_clusters)
            
            return {
                'insights': ranked_insights,
                'insight_clusters': insight_clusters,
                'meta_insights': meta_insights,
                'consolidation_quality': self._assess_consolidation_quality(ranked_insights, meta_insights)
            }
            
        except Exception as e:
            self.logger.error(f"Error in dream insight consolidation: {e}")
            return {'error': str(e), 'insights': []}
    
    def _filter_dream_contamination(self, consolidation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out dream contamination to protect memory systems."""
        try:
            insights = consolidation_result.get('insights', [])
            meta_insights = consolidation_result.get('meta_insights', [])
            
            # Apply contamination filters
            safe_insights = []
            contaminated_insights = []
            
            for insight in insights:
                if self._is_insight_safe(insight):
                    safe_insights.append(insight)
                else:
                    contaminated_insights.append(insight)
                    self.logger.warning(f"Filtered contaminated insight: {insight.get('content', 'Unknown')}")
            
            # Filter meta-insights
            safe_meta_insights = []
            contaminated_meta_insights = []
            
            for meta_insight in meta_insights:
                if self._is_meta_insight_safe(meta_insight):
                    safe_meta_insights.append(meta_insight)
                else:
                    contaminated_meta_insights.append(meta_insight)
            
            # Generate contamination report
            contamination_report = self._generate_contamination_report(
                contaminated_insights, contaminated_meta_insights
            )
            
            return {
                'safe_insights': safe_insights,
                'safe_meta_insights': safe_meta_insights,
                'contaminated_insights': contaminated_insights,
                'contaminated_meta_insights': contaminated_meta_insights,
                'contamination_report': contamination_report,
                'safety_score': len(safe_insights) / max(len(insights), 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error in contamination filtering: {e}")
            return {'error': str(e), 'safe_insights': [], 'safety_score': 0.0}
    
    def _integrate_safe_insights(self, filtered_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate safe insights into memory systems."""
        try:
            safe_insights = filtered_result.get('safe_insights', [])
            safe_meta_insights = filtered_result.get('safe_meta_insights', [])
            
            integration_results = []
            
            # Integrate individual insights
            for insight in safe_insights:
                integration_result = self._integrate_single_insight(insight)
                integration_results.append(integration_result)
            
            # Integrate meta-insights
            meta_integration_results = []
            for meta_insight in safe_meta_insights:
                meta_integration_result = self._integrate_meta_insight(meta_insight)
                meta_integration_results.append(meta_integration_result)
            
            # Update memory manager if available
            if self.memory_manager:
                self._update_memory_manager(safe_insights, safe_meta_insights)
            
            return {
                'integration_results': integration_results,
                'meta_integration_results': meta_integration_results,
                'total_integrated': len(integration_results) + len(meta_integration_results),
                'integration_success_rate': self._calculate_integration_success_rate(integration_results, meta_integration_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in safe insight integration: {e}")
            return {'error': str(e), 'total_integrated': 0, 'integration_success_rate': 0.0}
    
    def _is_insight_safe(self, insight: Dict[str, Any]) -> bool:
        """Determine if an insight is safe for memory integration."""
        try:
            # Check for dream-specific contamination markers
            content = insight.get('content', '').lower()
            
            # Filter out clearly fantastical or impossible content
            contamination_markers = [
                'impossible', 'magical', 'supernatural', 'flying without aid',
                'talking animals', 'time travel', 'teleportation', 'mind reading',
                'infinite', 'eternal', 'omnipotent', 'defying physics'
            ]
            
            for marker in contamination_markers:
                if marker in content:
                    return False
            
            # Check confidence threshold
            confidence = insight.get('confidence', 0.0)
            if confidence < 0.3:  # Low confidence insights may be contaminated
                return False
            
            # Check for logical consistency
            if not self._check_logical_consistency(insight):
                return False
            
            # Check for practical applicability
            applicability = insight.get('applicability', [])
            if not applicability or len(applicability) == 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking insight safety: {e}")
            return False
    
    def _is_meta_insight_safe(self, meta_insight: Dict[str, Any]) -> bool:
        """Determine if a meta-insight is safe for memory integration."""
        try:
            # Meta-insights should be more abstract and less prone to contamination
            # but still need basic safety checks
            
            confidence = meta_insight.get('confidence', 0.0)
            if confidence < 0.4:  # Slightly higher threshold for meta-insights
                return False
            
            # Check for practical value
            practical_value = meta_insight.get('practical_value', 0.0)
            if practical_value < 0.3:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking meta-insight safety: {e}")
            return False
    
    def _check_logical_consistency(self, insight: Dict[str, Any]) -> bool:
        """Check if an insight is logically consistent."""
        try:
            # Basic logical consistency checks
            content = insight.get('content', '')
            
            # Check for contradictory statements
            contradictory_pairs = [
                ('always', 'never'), ('all', 'none'), ('everything', 'nothing'),
                ('impossible', 'certain'), ('definitely', 'maybe')
            ]
            
            content_lower = content.lower()
            for pair in contradictory_pairs:
                if pair[0] in content_lower and pair[1] in content_lower:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking logical consistency: {e}")
            return False
    
    def set_hormone_integration(self, hormone_integration):
        """Set hormone system integration for dream cycle coordination."""
        self.hormone_integration = hormone_integration
        self.logger.info("Hormone integration set for DreamingEngine")
    
    # Helper methods for the new functionality
    
    def _generate_cycle_id(self, trigger_context: Dict[str, Any], cycle_type: str) -> str:
        """Generate unique cycle ID."""
        timestamp = datetime.now().isoformat()
        context_hash = hashlib.md5(str(trigger_context).encode()).hexdigest()[:8]
        return f"cycle_{cycle_type}_{context_hash}_{timestamp}"
    
    def _gather_recent_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Gather recent memories for dream processing."""
        # Placeholder implementation - would integrate with actual memory system
        return [
            {'type': 'episodic', 'content': 'Recent task completion', 'timestamp': datetime.now() - timedelta(hours=2)},
            {'type': 'semantic', 'content': 'New concept learned', 'timestamp': datetime.now() - timedelta(hours=5)},
            {'type': 'procedural', 'content': 'Skill practice session', 'timestamp': datetime.now() - timedelta(hours=8)}
        ]
    
    def _identify_unresolved_issues(self, trigger_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify unresolved issues that might benefit from dream processing."""
        issues = []
        
        if trigger_context:
            # Extract potential issues from context
            if 'problems' in trigger_context:
                for problem in trigger_context['problems']:
                    issues.append({
                        'type': 'problem',
                        'description': problem,
                        'urgency': random.uniform(0.3, 0.9)
                    })
        
        # Add some default unresolved issues for demonstration
        default_issues = [
            {'type': 'optimization', 'description': 'Process efficiency improvement', 'urgency': 0.6},
            {'type': 'creative', 'description': 'Novel solution exploration', 'urgency': 0.4},
            {'type': 'learning', 'description': 'Knowledge gap identification', 'urgency': 0.5}
        ]
        
        issues.extend(default_issues)
        return issues
    
    def _select_dream_themes(self, trigger_context: Dict[str, Any], cycle_type: str, 
                           unresolved_issues: List[Dict[str, Any]]) -> List[str]:
        """Select appropriate dream themes based on context and issues."""
        themes = []
        
        # Base themes by cycle type
        theme_mapping = {
            'natural': ['memory_integration', 'emotional_processing', 'problem_solving'],
            'induced': ['creative_exploration', 'threat_simulation', 'optimization'],
            'therapeutic': ['emotional_processing', 'trauma_integration', 'healing'],
            'creative': ['creative_exploration', 'artistic_inspiration', 'innovation']
        }
        
        base_themes = theme_mapping.get(cycle_type, ['problem_solving'])
        themes.extend(base_themes)
        
        # Add themes based on unresolved issues
        for issue in unresolved_issues:
            if issue['type'] == 'problem':
                themes.append('problem_solving')
            elif issue['type'] == 'creative':
                themes.append('creative_exploration')
            elif issue['type'] == 'optimization':
                themes.append('optimization')
        
        return list(set(themes))  # Remove duplicates
    
    def _store_dream_cycle(self, cycle_result: Dict[str, Any]):
        """Store complete dream cycle results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store in dream_cycles table (create if not exists)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dream_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    cycle_type TEXT NOT NULL,
                    trigger_context TEXT,
                    total_insights INTEGER DEFAULT 0,
                    contamination_filtered INTEGER DEFAULT 0,
                    cycle_quality REAL DEFAULT 0.0,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cycle_data TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO dream_cycles 
                (cycle_id, cycle_type, trigger_context, total_insights, contamination_filtered, 
                 cycle_quality, completed_at, cycle_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_result['cycle_id'],
                cycle_result['cycle_type'],
                json.dumps(cycle_result.get('trigger_context', {})),
                cycle_result.get('total_insights', 0),
                cycle_result.get('contamination_filtered', 0),
                cycle_result.get('cycle_quality', 0.0),
                cycle_result['completed_at'].isoformat(),
                json.dumps(cycle_result)
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing dream cycle: {e}")
        finally:
            conn.close()
    
    # Additional helper methods would be implemented here...
    # For brevity, I'm including key methods and placeholders for others
    
    def _prepare_dream_environment(self, cycle_type: str) -> Dict[str, Any]:
        """Prepare dream environment parameters."""
        return {
            'creativity_level': 0.8 if cycle_type == 'creative' else 0.6,
            'logic_constraints': 0.3 if cycle_type == 'creative' else 0.7,
            'emotional_intensity': 0.7 if cycle_type == 'therapeutic' else 0.5,
            'memory_access_level': 0.9
        }
    
    def _assess_preparation_quality(self, recent_memories: List, unresolved_issues: List) -> float:
        """Assess quality of dream preparation."""
        memory_score = min(len(recent_memories) / 10.0, 1.0)
        issue_score = min(len(unresolved_issues) / 5.0, 1.0)
        return (memory_score + issue_score) / 2.0
    
    def _generate_rem_scenario(self, theme: str, preparation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a REM scenario based on theme."""
        return {
            'theme': theme,
            'scenario_id': f"rem_{theme}_{random.randint(1000, 9999)}",
            'content': f"REM scenario exploring {theme}",
            'emotional_intensity': random.uniform(0.3, 0.9),
            'creativity_level': random.uniform(0.5, 1.0),
            'preparation_context': preparation_result.get('environment_params', {})
        }
    
    def _process_rem_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Process a REM scenario."""
        scenario['processed'] = True
        scenario['processing_time'] = random.uniform(0.1, 0.5)  # REM is fast
        scenario['insights_generated'] = random.randint(1, 3)
        return scenario
    
    def _calculate_rem_intensity(self, scenarios: List[Dict[str, Any]]) -> float:
        """Calculate REM intensity from scenarios."""
        if not scenarios:
            return 0.0
        total_intensity = sum(s.get('emotional_intensity', 0.0) for s in scenarios)
        return total_intensity / len(scenarios)
    
    def _extract_rem_emotions(self, scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract emotional processing from REM scenarios."""
        emotions = {
            'excitement': 0.0,
            'anxiety': 0.0,
            'curiosity': 0.0,
            'satisfaction': 0.0
        }
        
        for scenario in scenarios:
            intensity = scenario.get('emotional_intensity', 0.0)
            # Distribute intensity across emotions based on theme
            theme = scenario.get('theme', '')
            if 'creative' in theme:
                emotions['excitement'] += intensity * 0.4
                emotions['curiosity'] += intensity * 0.3
            elif 'problem' in theme:
                emotions['anxiety'] += intensity * 0.3
                emotions['satisfaction'] += intensity * 0.4
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions    return total_value / len(insights)
    
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
   # Additional helper methods for comprehensive dream cycle functionality
    
    def _select_scenarios_for_deep_processing(self, rem_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select most promising REM scenarios for deep processing."""
        # Sort by creativity level and emotional intensity
        scored_scenarios = []
        for scenario in rem_scenarios:
            score = (scenario.get('creativity_level', 0.0) * 0.6 + 
                    scenario.get('emotional_intensity', 0.0) * 0.4)
            scored_scenarios.append((score, scenario))
        
        # Select top scenarios (up to 5)
        scored_scenarios.sort(key=lambda x: x[0], reverse=True)
        return [scenario for _, scenario in scored_scenarios[:5]]
    
    def _create_deep_dream(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep dream from a REM scenario."""
        return {
            'deep_dream_id': f"deep_{scenario.get('scenario_id', 'unknown')}",
            'base_scenario': scenario,
            'processing_depth': random.uniform(0.7, 1.0),
            'symbolic_content': self._generate_symbolic_content(scenario),
            'narrative_structure': self._create_dream_narrative(scenario),
            'sensory_details': self._add_sensory_details(scenario)
        }
    
    def _expand_creatively(self, deep_dream: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand dream creatively with novel elements."""
        expansions = []
        base_theme = deep_dream.get('base_scenario', {}).get('theme', 'unknown')
        
        for _ in range(random.randint(2, 5)):
            expansion = {
                'expansion_type': random.choice(['metaphorical', 'literal', 'symbolic', 'abstract']),
                'content': f"Creative expansion of {base_theme}",
                'novelty_score': random.uniform(0.5, 1.0),
                'coherence_score': random.uniform(0.3, 0.9)
            }
            expansions.append(expansion)
        
        return expansions
    
    def _extract_problem_solutions(self, deep_dream: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract problem solutions from deep dream."""
        solutions = []
        base_scenario = deep_dream.get('base_scenario', {})
        theme = base_scenario.get('theme', '')
        
        if 'problem' in theme:
            for _ in range(random.randint(1, 3)):
                solution = {
                    'solution_type': random.choice(['direct', 'indirect', 'creative', 'systematic']),
                    'description': f"Deep dream solution for {theme}",
                    'feasibility': random.uniform(0.4, 0.9),
                    'originality': random.uniform(0.6, 1.0)
                }
                solutions.append(solution)
        
        return solutions
    
    def _extract_symbolic_meanings(self, deep_dream: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract symbolic meanings from deep dream."""
        symbols = []
        symbolic_content = deep_dream.get('symbolic_content', [])
        
        for symbol_data in symbolic_content:
            symbol = {
                'symbol': symbol_data.get('symbol', 'unknown'),
                'meaning': symbol_data.get('interpretation', 'unclear'),
                'confidence': random.uniform(0.3, 0.8),
                'cultural_context': random.choice(['universal', 'personal', 'cultural', 'archetypal'])
            }
            symbols.append(symbol)
        
        return symbols
    
    def _extract_deep_emotions(self, deep_dream: Dict[str, Any]) -> Dict[str, Any]:
        """Extract deep emotional insights from dream."""
        base_emotions = deep_dream.get('base_scenario', {}).get('emotional_intensity', 0.5)
        
        return {
            'primary_emotion': random.choice(['wonder', 'fear', 'joy', 'sadness', 'anger', 'surprise']),
            'emotional_intensity': base_emotions,
            'emotional_complexity': random.uniform(0.4, 0.9),
            'emotional_resolution': random.uniform(0.2, 0.8),
            'cathartic_value': random.uniform(0.3, 0.9)
        }
    
    def _calculate_processing_depth(self, deep_dreams: List[Dict[str, Any]]) -> float:
        """Calculate overall processing depth."""
        if not deep_dreams:
            return 0.0
        
        total_depth = sum(dream.get('processing_depth', 0.0) for dream in deep_dreams)
        return total_depth / len(deep_dreams)
    
    def _identify_creative_breakthroughs(self, deep_dreams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify creative breakthroughs in deep dreams."""
        breakthroughs = []
        
        for dream in deep_dreams:
            expansions = dream.get('creative_expansions', [])
            for expansion in expansions:
                if expansion.get('novelty_score', 0.0) > 0.8:
                    breakthrough = {
                        'type': 'creative_breakthrough',
                        'content': expansion.get('content', ''),
                        'novelty_score': expansion.get('novelty_score', 0.0),
                        'source_dream': dream.get('deep_dream_id', 'unknown')
                    }
                    breakthroughs.append(breakthrough)
        
        return breakthroughs
    
    def _extract_problem_solving_insights(self, deep_dreams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract problem-solving insights from deep dreams."""
        insights = []
        
        for dream in deep_dreams:
            solutions = dream.get('problem_solutions', [])
            for solution in solutions:
                if solution.get('feasibility', 0.0) > 0.6 and solution.get('originality', 0.0) > 0.5:
                    insight = {
                        'type': 'problem_solving_insight',
                        'solution': solution.get('description', ''),
                        'feasibility': solution.get('feasibility', 0.0),
                        'originality': solution.get('originality', 0.0),
                        'source_dream': dream.get('deep_dream_id', 'unknown')
                    }
                    insights.append(insight)
        
        return insights
    
    def _extract_consolidated_insights(self, dream: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract consolidated insights from a single dream."""
        insights = []
        
        # Extract from creative expansions
        expansions = dream.get('creative_expansions', [])
        for expansion in expansions:
            if expansion.get('coherence_score', 0.0) > 0.6:
                insights.append({
                    'type': 'creative_insight',
                    'content': expansion.get('content', ''),
                    'confidence': expansion.get('coherence_score', 0.0),
                    'source': 'creative_expansion'
                })
        
        # Extract from problem solutions
        solutions = dream.get('problem_solutions', [])
        for solution in solutions:
            insights.append({
                'type': 'solution_insight',
                'content': solution.get('description', ''),
                'confidence': solution.get('feasibility', 0.0),
                'source': 'problem_solution'
            })
        
        # Extract from symbolic meanings
        symbols = dream.get('symbolic_meanings', [])
        for symbol in symbols:
            if symbol.get('confidence', 0.0) > 0.5:
                insights.append({
                    'type': 'symbolic_insight',
                    'content': f"Symbol {symbol.get('symbol', '')} means {symbol.get('meaning', '')}",
                    'confidence': symbol.get('confidence', 0.0),
                    'source': 'symbolic_meaning'
                })
        
        return insights
    
    def _rank_insights_by_value(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank insights by their potential value."""
        # Calculate value score for each insight
        for insight in insights:
            confidence = insight.get('confidence', 0.0)
            type_bonus = {
                'solution_insight': 0.3,
                'creative_insight': 0.2,
                'symbolic_insight': 0.1
            }.get(insight.get('type', ''), 0.0)
            
            insight['value_score'] = confidence + type_bonus
        
        # Sort by value score
        return sorted(insights, key=lambda x: x.get('value_score', 0.0), reverse=True)
    
    def _cluster_related_insights(self, insights: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group related insights into clusters."""
        clusters = []
        used_insights = set()
        
        for i, insight in enumerate(insights):
            if i in used_insights:
                continue
            
            cluster = [insight]
            used_insights.add(i)
            
            # Find related insights
            for j, other_insight in enumerate(insights[i+1:], i+1):
                if j in used_insights:
                    continue
                
                if self._are_insights_related(insight, other_insight):
                    cluster.append(other_insight)
                    used_insights.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_insights_related(self, insight1: Dict[str, Any], insight2: Dict[str, Any]) -> bool:
        """Check if two insights are related."""
        # Simple relatedness check based on type and content similarity
        if insight1.get('type') == insight2.get('type'):
            return True
        
        content1 = insight1.get('content', '').lower()
        content2 = insight2.get('content', '').lower()
        
        # Check for common words (simple similarity)
        words1 = set(content1.split())
        words2 = set(content2.split())
        common_words = words1.intersection(words2)
        
        return len(common_words) >= 2
    
    def _generate_meta_insights(self, insight_clusters: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate meta-insights from insight clusters."""
        meta_insights = []
        
        for cluster in insight_clusters:
            if len(cluster) >= 2:  # Only generate meta-insights for clusters with multiple insights
                meta_insight = {
                    'type': 'meta_insight',
                    'content': f"Pattern identified across {len(cluster)} related insights",
                    'cluster_size': len(cluster),
                    'confidence': sum(insight.get('confidence', 0.0) for insight in cluster) / len(cluster),
                    'practical_value': random.uniform(0.4, 0.8),
                    'constituent_insights': [insight.get('content', '') for insight in cluster]
                }
                meta_insights.append(meta_insight)
        
        return meta_insights
    
    def _assess_consolidation_quality(self, insights: List[Dict[str, Any]], 
                                    meta_insights: List[Dict[str, Any]]) -> float:
        """Assess the quality of insight consolidation."""
        if not insights:
            return 0.0
        
        # Base quality from insight confidence
        avg_confidence = sum(insight.get('confidence', 0.0) for insight in insights) / len(insights)
        
        # Bonus for meta-insights
        meta_bonus = min(len(meta_insights) * 0.1, 0.3)
        
        # Bonus for insight diversity
        insight_types = set(insight.get('type', '') for insight in insights)
        diversity_bonus = min(len(insight_types) * 0.05, 0.2)
        
        return avg_confidence + meta_bonus + diversity_bonus
    
    def _generate_contamination_report(self, contaminated_insights: List[Dict[str, Any]], 
                                     contaminated_meta_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """!")llyte a reted suc comple d contamreation."e 
   
    c_dream())_basio.run(test  asynci
    
  ))}") []'insights',t.get(len(resuled: {athts generigf"Ins print(     }")
  ', 0.0):.2fuevalrning_et('leault.g{res value: ningearint(f"L    pr)
    0):.2f}", 0.ity_score''qualget(e: {result.uality scor"Q    print(f')}")
    nknownio_id', 'Uscenaresult.get('esult: {r rm simulationDreaint(f" pr    
        )
   lving""problem_soype=m_t  drea          lem",
on prob optimizatiomplex"Solving c    context=m(
        mulate_drea engine.siawait = ult
        res_dream(): test_basicc defasyn
    
    ncioport asy   im
    
 lation...")mu dream sing basicti\nTes(" printation
    dream simulasicTest b    # )
    
.0):.2f}"_quality', 0leyct('ce_result.geality: {cycl"Cycle quint(f)
    prd', 0)}"lteren_fiontaminatioet('ccle_result.gd: {cyfilteretion ina"Contam  print(f  )}")
hts', 0insig'total_t.get(cycle_resul insights: {talrint(f"To
    p)}")', 'Unknown'('cycle_id.getultes_r{cyclempleted: co cycle eamrint(f"Dr p   
       )
 ative'
='creypecycle_t      t,
  ontexxt=trigger_cger_conte  trig(
      m_cycle_dreanitiatene.i= engiycle_result   c 
  }
   
    ed' motivatous and: 'curional_state'ti        'emo'],
w technique net', 'learnedajor projecompleted m ['cts':_evenrecent       '
 ed'],olution neede s', 'Creativmanceem perforystoptimize sto : ['How s'lemrob   'p = {
     extger_cont    trig")
    
ycle...am chensive drecompreg in"Testnt(on
    prie initiaticyclm # Test drea
    
    gEngine()Dreamingine =   engine
   dreaming en# Create":
    __main__me__ == "ing
if __natest usage and lexamp

# E")
}: {e}{insight.idt nsighg issinroceror prror(f"Erger.eogf.lel        s    on as e:
cepti except Exs
        requirementprocessingon specific  depend tion would Implementa          #id}")
  nsight. {isight: inocessing.info(f"Prself.logger      
      d processorbackgroune lled by thld be ca This wou      #
              try:"""
).ngund processi(backgroight cess an ins"""Pro        ):
eamInsightnsight: Drght(self, insiprocess_ief _    d    
: {e}")
ario.id} {sceneam scenariong dr processif"Errorr.error(ogge      self.l
      s e:eption aExc    except ments
    g requiressincific proceped on sould depention wlementa Imp       #}")
     .idrio{scenaario:  sceng dreamcessin.info(f"Prologger   self.
         rocessorkground pthe bac by e called would b  # This
            try:     
 "")."ocessingackground pr(bcenario s a dream sces""Pro    "
    enario):amScio: Dreself, scenario(narream_scerocess_d  def _p   
  close()
  conn.         
  lly:  fina")
      io: {e}dream scenaring or stor"Errror(fogger.er    self.l       :
 eption as except Exc  e
                  mmit()
    conn.co
                    
    ))            rmat()
ed_at.isofot.creat      insigh        
      y),pplicabilit.aighton.dumps(ins      js          dence,
    sight.confi    in          ent,
      insight.cont                    type,
ight_insight.ins                  
  enario_id, insight.sc                ght.id,
         insi      
         (    """,      ?)
       , ?,, ?, ?, ?, ? (?   VALUES            )
     eated_at cricability,, appl, confidence, contentsight_typerio_id, inid, scena       (            hts 
 _insigreamCE INTO dREPLAOR T INSER                   "
 te(""ecu   cursor.ex        ghts:
     insiht in for insig         ights
   ore ins   # St        
         ))
                leted'
comp       ',
         arning_value     le
           ity_score, qual          
     soformat(),ow().iatetime.n  d             t(),
 format.isocreated_ao. scenari      ,
         n_data)o.simulatios(scenarison.dump        j    ype,
    o.dream_t scenari               .context,
 scenario    
           id,ario.   scen          (
    ","    "     ?)
     ?,?, ?, ?, ?, ?, , ?,ALUES (?         V   atus)
    stlue, ning_valear_score,   quality               ed_at, 
 complet_at,eda, creatdation_ype, simulatm_tdreat,  contex (id,     
          cenarios INTO dream_sREPLACE NSERT OR  I           ""
    "execute( cursor.           e scenario
 Stor       # 
       )
         rsor(= conn.cu cursor       )
     self.db_pathct(e3.conneconn = sqlit            y:

        tr"""e.basdatasults in renario  scee dreamor"St      ""at):
  loing_value: flearn, aty_score: floitht], qualt[DreamInsigLishts:   insig                          
 y],[str, Anresult: Dictario, dream_DreamScen, scenario: (selfam_scenario_dreef _store 
    d
   dations recommenreturn 
            }")
   ntentight.co: {ins riskentialor potnd(f"Monitations.appeommend  rec             
 ht_type:.insign insightif 'risk' i                      
")
  .content}htsiging: {inimplement"Consider nd(fppeions.amendat recom             
  pe:ht_tyt.insiginsighlution' in 'so   if                  
   ")
 t.content}sight: {in insighconfidenced(f"High-ppens.aendation  recomm           
   :nce > 0.7onfideht.c    if insig       nsights:
 in iinsight for       
         = []
  mendationscom re"
       ights.""nsons from imendatible recomonaate acti"""Gener       tr]:
 -> List[st]) eamInsigh List[Drghts:f, insidations(selmmente_recoenera def _g
   
    ent plan')emmanag risk nsiverehecomp'Develop at_type, threget(s.rategieturn st       re    }
     ize usage'
d optimans urce source resoversifyDisource': '        're  ,
  uling'schedlexible me and fer ti 'Build buffemporal':     't       ies',
ingencprepare conts and or conditionl': 'Monitntame    'environ,
        gagement'holder enon and stakeati communic'Improvecial':        'so
     cols',g protoin testdundancy andplement reIm': ' 'technical          egies = {
 strat      """
  e.at typor a thren strategy fatioigGenerate mit"""
        str) -> str:eat_type: hry(self, tstrategigation_te_mitneraef _ge    
    ddations
enturn recomm
        re
        s")tienining opportuear lplorepend("Exdations.ap   recommen
         :0.0) > 0.8curiosity', ('ons.getif emoti            

     action")ctivet into produxcitemennnel eappend("Chandations.mecom      re     0) > 0.7:
 t', 0.temenciexons.get(' if emoti         
  ")
     techniquesanagementer stress mnsidappend("Comendations. recom        
    0.0) > 0.6:ety',xiions.get('anmotf e
        i]
        ns = [ndatio   recomme
     ing."""ocessal pr on emotionsedendations barate recommGene""       "
 str]:) -> List[float] Dict[str, , emotions:s(selftionecommendaional_rte_emotgeneraef _    d  
s
  rn insight retu      
         ")
ancenificnal sigioes high emote indicat]} responsnt_emotion[0{domina"Strong append(fnsights.         i.7:
   [1] > 0onant_emotidomin       if 
    ]}")
     emotion[0nant_is {domiio.context} cenarsponse to {stional reimary emof"Prts.append(   insigh        
    
 1])a x: x[ key=lambd.items(),ionsmax(emoton = t_emoti dominan    
   sights = []       in
 "g.""inocesstion prts from emoinsigh emotional xtract""E"     str]:
   List[loat]) -> [str, ftions: Dictnario, emoo: DreamSceari, scens(selfnal_insightact_emotiof _extr   de    
 e, [])
ctivspeet(per.gunitiespporterspective_o + pitiesse_opportunba return 
                     }
 ]
 ext}"io.contn {scenarty i opportunitimizationf"Opytical': ['anal         "],
   io.context}cenarnity in {son opportuovatie': [f"Inntiv     'crea       "],
io.context}l in {scenartiapotenat [f"Greistic': optim  '         es = {
 opportunitictive_ perspe       

        "]ext}cenario.cont in {sportunityearning opies = [f"Lit_opportun        base"""
rspective. specific peies from apportunit"Identify o   ""
     List[str]:r) -> : stvespectier po,ri: DreamScenaf, scenario(selitiesportunfy_opidentief _   
    dtive, [])
 get(perspecks.ive_riserspect_risks + pbase     return    
   
         }
    "]ontext}io.c{scenarisks in nnovation r [f"Iive':   'creat      xt}"],
   .contenarioceks in {sic risemat [f"Systanalytical':     '
       xt}"],enario.conteor {sccenario frst-case stic': [f"Wosimis       'pes    {
  = ctive_risks   perspe     
  
      ntext}"]scenario.coraints for {ste con, f"Resourctext}"scenario.conre in {al failuotentisks = [f"Pe_ri   bas"
     ve.""ic perspectiom a specifks frrisentify """Id        [str]:
) -> Listive: str, perspecteamScenarionario: Drscelf, _risks(sef _identifyde   
    
 ontext}")scenario.ccome for {nown outve, f"Unkperspecti.get(mesreturn outco}
            "
    o.context} {scenariforn tic solutio f"Systemaytical':nal        'a    ",
t}exenario.contoach to {scapprovel ative': f"N  'cre         ",
 text}rio.conna{sce for metcoanced oual': f"Bal      'neutr",
      xt}.conte{scenarion ications ing complChallengi f"istic':pessim        '   ",
 text}nario.conion of {sceute resol"Positivc': fmisti   'opti     es = {
         outcom   "
""e.ivpect persspecific a e fromutcommulate o"Si      ""str:
  r) -> stspective: , perenarioamScnario: Dreelf, sceutcome(se_odef _simulat
    pts)
    choice(conceturn random.
        re   ]'
      'story',game 'le',ft', 'puzz'gitranger', end', 's', 'frimeurney', 'ho         'jo  ce',
 'dan 'song', or', 'book',rr', 'mikeydoor', '', 'path', '  'bridge        hadow',
  ght', 's 'lione',nd', 'stire', 'wiee', 'ftain', 'tr', 'mounwater    '  
      s = [ncept  co
      ions."""or associatpt fncem corate a rando  """Genetr:
      -> sncept(self) ndom_coe_ra _generat  defs
    
  mainspecific_do_domains +  return base)
       ype, []sight_tcific.get(in_speains = typeom specific_d      
 }
        
        ion_making']isnt', 'dect_managemeojecs', 'pr': ['businesnningtrategic_pla         's
   adership'],tion', 'le, 'communicachology'': ['psyightinsnal_'emotio          n'],
  sificatioclasn', 'ctio', 'predilysisnadata_a: ['ognition'pattern_rec     '],
       _management' 'resourcerformance',peiency', 'effic: ['ion'mizat 'opti          
 lysis'],ng', 'ana 'planniecurity',on': ['sentificati_id      'risk     '],
 ation', 'artnov, 'in['design': dea'creative_i    '],
        earch'', 'resentg', 'managemngineerin: ['eution''problem_sol            = {
fic e_specityp  
        
      ing']nkcreative_thiolving', 'm_srobleal', 'p'geners = [e_domain     bas"
   sight.""s for an inity domainilabne applicDetermi"""]:
         List[strny) ->ta: Asource_datr, _type: snsightlity(self, icabiplietermine_ap
    def _d
    p}"tamimes_id}_{tcenariotype}_{st_ht_{insighsign f"in returt()
       w().isoformaetime.no = datamp      timest"""
  nsight ID. iate unique"""Gener
        > str:str) -ype: ght_tr, insi: stcenario_idf, sght_id(selrate_insigene  def _    
  "
imestamp}hash}_{t_{context_type}eam_drio_{n f"scenar    retur:8]
    st()[).hexdige.encode()xtb.md5(conte hashlih =ntext_has     comat()
   w().isofor.no = datetime   timestamp""
     enario ID."unique sc"Generate   ""
      -> str:tr) pe: sm_tytr, dreaext: self, contd(so_irinacete_sdef _genera   
    r methods
 helpemissing tions for  implementaeholder# Plac }

        
   rm(0.5, 0.9)unifom.andoidness': rry_viv     'senso'],
       nce patternssona, 'reiations'varty ensi, 'int shifts'odnal': ['mo     'emotio     
  s'],eeling 'movement f changes',rature 'tempeons',atiure sens': ['text    'tactile        
tones'],', 'musical bient soundsvoices', 'amdistant : ['itory''aud       '],
     iliar facesamhapes', 'fting s 'shiflors', co': ['bright 'visual       urn {
         ret"""
   sm.eam reali enhance drils toensory detad s"""Ad    
    [str, Any]:> Dictny]) -[str, Aario: Dict(self, scen_detailsd_sensoryf _ad  de }
    
        lex'])
 ical', 'compcyclalling', 'ising', 'f['rndom.choice(l_arc': rationa       'emo    
 8),orm(0.4, 0.om.unife': randive_coherenc  'narrat          
tanding",ed undersratteges with inoncludDream cion': "resolut          '",
  ion emerges or resolutsight': "Key in  'climax   ",
       ngocessionscious prthrough uncio evolves : "Scenardevelopment'         'ario",
   )} scen', 'unknown''themeenario.get(ins with {sceam begning': f"Drbegin     '     
  {eturn 
        rm."""p dreature for deeve strucate narrati  """Cre
      y]:[str, An) -> Dict[str, Any]enario: Dictself, sctive(ream_narracreate_df _de    
    rn symbols
etu    r  
              })
      )
   0.8.4,(0uniformandom.ity': r     'clar
           ),(0.3, 0.9iformndom.unce': raanional_reson     'emot        ",
   heme}t of {taspecnts f"Represeretation': interp  '          
    mbol,bol': sy        'sym     nd({
   s.appe    symbol       
 ls))):e_symbohem len(t(3,s, mineme_symbolm.sample(thrando in olr symb fo
       
        , 'path'])', 'mystery'e, ['unknownt(themge_mappings.ls = symbol_symbo    theme  
    
              }rt']
 'heaen', 'gard 'ocean',fire','storm', 'essing': [rocotional_p         'emm'],
   , 'albu', 'mirror'er', 'rivee, 'tr: ['library'n'y_integratio      'memor,
      , 'spiral']y', 'prism'erfled', 'buttnvas', 'setion': ['cae_exploraeativ    'cr,
        , 'door']n'mountairidge', '', 'key', 'b'mazeg': [lem_solvin      'prob     
 ings = { symbol_mappls
       priate symborote theme-app # Genera       
        )
n'', 'unknow('themeetscenario.ge = hem        tls = []
       symbo""
 reams." dr deepcontent fosymbolic Generate ""      "ny]]:
  Dict[str, A List[Any]) ->t[str, ario: Diccen(self, sntntelic_coe_symboerat   def _gen   
 
 e structureand narrativ content ic symbolthods forl helper me # Additiona   
   ty_bonus
 quantibonus + afety_dence + sn avg_confiur  ret           
  , 0.2)
 10.0sights) / len(safe_inonus = min(quantity_b        bonus
ntity ght qua# Insi
        3
        ore * 0.afety_scs = snu_bo      safety
  y bonus# Safet                
ghts)
safe_insien(ights) / linsafe_n ssight ior in0) fce', 0.nfidenight.get('co sum(inse =denc_confi   avg     fidence
t conrom insighlity fBase qua       #       
 
  urn 0.0        retights:
    nssafe_if not 
        i
        re', 0.0)'safety_scoresult.get(iltered_ty_score = fafe       s
 , [])_insights'safet('.geresultiltered_hts = f safe_insig""
       "m cycle. of a drealityverall quahe ossess t"A"      "> float:
  y]) -r, An Dict[st_result:teredillf, fe_quality(sessess_cycl _a
    def   ts)
 ll_resulsful / len(aces suc    return)
    ccess'= 'sustatus') =(' result.getsults ift in all_reresul= sum(1 for cessful       suc
      n 0.0
    uret           r
 esults:t all_rno       if ts
 ion_resulgrateta_inte mults +_restionts = integraresul all_    """
   ntegration.t ie of insighats re the succesatcul""Cal       "oat:
 flAny]]) -> ict[str, ist[Ds: Lresulttegration_  meta_in                                       ]], 
 , Anyct[strst[Dilts: Lion_resutintegra(self, is_ratetion_succestegralculate_indef _ca  
    {e}")
  : nager memory matingupdaror ror(f"Err.erelf.logge           ss e:
 ption axcept Exce
        e           ")
 hts-insigmeta)} ightse_meta_insnd {len(saf} insights ainsights)fe_ith {len(sary manager wdated memonfo(f"Uper.i.logg     self               
)
                         0.0)}
',luecal_va('practiht.getmeta_insig_value': cal 'practie',gin_enamingource': 'dreta={'smetada                 t',
       sigh'meta_inmemory_type=                        nt', ''),
'conteght.get(ta_insi content=me                     memory(
  anager.add_f.memory_m    sel       
         y'): 'add_memor_manager,(self.memoryf hasattr   eli            insight)
 ht(meta_insigmeta_nager.add_y_mamemor self.            
       ht'):sig_in, 'add_metaagermemory_manattr(self.f has     i
           ghts:meta_insisafe_t in  meta_insigh       for
     s-insight meta # Add                
 
      )                 )}
   ce', 0.0et('confident.g': insighonfidencegine', 'caming_enrce': 'dretadata={'sou     me          ,
         e='insight'ry_typ      memo            '),
      ontent', 'ight.get('cins    content=               
     mory(er.add_mey_manag self.memor                  ):
 d_memory'er, 'admanagelf.memory_sattr(s    elif ha      
      t(insight)dd_insighager.ay_manf.memor         sel           '):
ightdd_ins'ary_manager, emoself.mhasattr( if           ts:
     e_insighght in saf   for insi       er
  ry manags to memosightAdd in          # y:
          tr      

  eturn      rer:
      anaglf.memory_m not se
        if"ts.""nsighth safe i manager wiory mem"Update""     ny]]):
   t[str, Aic: List[Dhtse_meta_insig     saf                  ]], 
      r, Anyt[Dict[sthts: Lisnsiglf, safe_inager(semate_memory_daef _up
    d          }
 
     age'_stornsight_i': 'metaion_methodatgr  'inte          tr(e),
    'error': s             ',
   'errortatus':         's       {
   return     ")
      nsight: {e}g meta-iatintegrError in(f"rorogger.erelf.l     s       tion as e:
ep Excexcept         
     }
                 e'
 agsight_stora_ind': 'metration_metho     'integ        }",
   000, 9999)t(1in{random.randmeta_ f"_id':meta_insight    '           
 uccess','s'status':            
      return {
                    
   ose()  conn.cl      )
    it(conn.comm            
         )
         )
      ]))hts', [iguent_institget('cons_insight.umps(metajson.d           
     mat(),w().isofore.nodatetim                 0),
ize',r_set('clusteght.ginsi    meta_     ),
       , 0.0al_value'('practicsight.get   meta_in      0),
       ', 0.encet('confid_insight.ge meta               ),
'''content', et(a_insight.gmet             99)}",
   (1000, 99intndom.rand{ra f"meta_              "", (
       "
      , ?, ?, ?), ?, ?, ?  VALUES (?          
    )sightsuent_in constitt, created_aluster_size,e, cluractical_vafidence, pntent, con   (id, co        s 
     ightnsNTO meta_iPLACE I OR REERT INS           "
    execute(""     cursor.
                  """)
         
             )     hts TEXT
  ituent_insignst       co           ESTAMP,
  RENT_TIMDEFAULT CURt TIMESTAMP _a     created               FAULT 0,
GER DETEINze ter_si   clus        
         AULT 0.0, DEFALalue REcal_v     practi             .0,
   0FAULTL DEfidence REA        con          L,
  TEXT NOT NULntent    co                KEY,
  MARYT PRI     id TEX     
           (ta_insightsmeXISTS LE IF NOT EABCREATE T                ute("""
or.execrs  cu           exists
 if nots tablemeta-insightCreate      #   
                )
 or(ursor = conn.c  curs         .db_path)
 elfconnect(s3.ite  conn = sql
          lingcial handspe with ghtre meta-insi    # Sto        try:
  ""
      he system." t intohtta-insigte a me"Integra       ""r, Any]:
 > Dict[stAny]) -ict[str, ght: Da_insi(self, met_insightetagrate_m_inte
    def     }
    e'
        e_storag 'databas':_methodintegration          '),
      r': str(e 'erro       ,
        s': 'error'     'statu           n {
ur      ret
      ght: {e}")ating insitegrError inr(f"erroger.self.log      s e:
      xception a Eept  exc         
           }
     e'
     e_storagabasd': 'datthotion_me 'integra              99)}",
 int(1000, 99ndandom.raegrated_{rntf"iht_id':  'insig             cess',
  tus': 'sucta       's
         turn {      re 
              close()
    conn.      mit()
     onn.com          c  
        ))
         
       mat()fore.now().iso  datetim              
lity', [])),plicabi('apsight.getumps(in      json.d    ),
      .0dence', 0onfi('cght.get       insi      ', ''),
   nt'conteght.get(        insi,
        '), 'unknownet('type'.gsight   in          d",
   solidate  "con            
  9)}",, 9991000t(andinndom.r_{raedrat  f"integ         (
      """,      
      ?, ?, ?)?, ?, ?, ?, VALUES (      
          ated_at)ility, crecabplince, apidentent, confght_type, co, insiio_id scenar      (id,        nsights 
  eam_i INTO drEPLACE INSERT OR R               "
"cute("rsor.exe          cu        
     rsor()
 nn.cu= cocursor            
 h).db_patlfct(seite3.conne  conn = sql
           databaseinsight in    # Store   :
            try""
  he system."t into t insighnglegrate a si""Inte"        :
ny], A> Dict[str -])t[str, Anyinsight: Dicf, nsight(selingle_i_integrate_s   def 
    
 endationsurn recomm     ret     
   )
   s"chanisming mence scorve confide"Improappend(ons.recommendati          nt > 3:
  _couidencef low_conf   i  0.3)
   < .0) ence', 0'confidt(insight.ge       if                        
   ights d_ins contaminateht insig1 for inum( sount =nfidence_cco  low_  
         ion")
   neratm ge in dreaasy elementsduce fantend("Reappndations.mecom  re     s):
     ghtsiinated_inntamght in co) for insiwer(ent', '').loht.get('contig insinl' caf any('magi     i  
   
      s")ameterng dream parr adjusticonsideted - etecon dtaminatiigh cond("Hs.appenndationmme reco           5:
nsights) > taminated_i if len(con
               ons = []
datien    recomm"""
    is.tion analysn contaminad o basendationse recomme"Generat  "":
      str]-> List[) str, Any]]Dict[List[nsights: ed_iontaminatons(self, commendatimination_recrate_contadef _gene 
       ries
turn catego  re
              1
 l'] +=s['illogica  categorie        e:
      ls       e1
     l'] += ['impractica categories             []):
   ility',ab('applict.getnot insigh    elif       
   += 1fidence']ons['low_corieteg         ca    :
   ce < 0.3lif confiden   e          1
stical'] +='fantagories[ cate             ]):
  pernatural'agical', 'su, 'me'siblpos in ['imr markerntent focor in ny(marke   if a         
         0.0)
   onfidence', et('cnsight.gnce = i   confide      er()
   .low, '')('content'.getght insient =  cont        ts:
  _insighntaminatedin coight  for ins  
         }
            
 0l':impractica      '    : 0,
  logical'  'il
          0,onfidence':      'low_c   
    cal': 0, 'fantasti     
      = {egories   cat""
      ion found."ntaminatf cotypes otegorize "Ca      ""nt]:
  [str, iDicty]]) -> An[Dict[str, sights: Listed_intaminat connation(self,ami_contategorizedef _c
    
     }       insights)
ontaminated_endations(ction_recommminaate_conta self._generndations':comme        'rer now
    ing fot filterrfecsume pe  # Asess': 1.0,iveng_effect 'filterin          
 ghts),d_insiaminateination(contntam_coategorizes': self._cion_typetaminaton   'c
         _insights),d_metaaminateen(conthts': lnsigmeta_ied_minat      'conta    hts),
  nsigntaminated_icoghts': len(nated_insimi 'conta     ,
      nsights)d_meta_iontaminate(cs) + lenated_insightminnta len(co':minatedtotal_conta           'return {
     """
    ation.d contaminn filtere a report oateGener