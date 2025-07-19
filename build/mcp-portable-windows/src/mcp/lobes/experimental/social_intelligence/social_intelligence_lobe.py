"""
SocialIntelligenceLobe: Superior Temporal Sulcus analog for user relationship modeling,
collaboration optimization, empathy simulation, and trust building.

This lobe handles social aspects of the MCP system, enabling it to build and maintain
relationships with users, optimize collaborative workflows, and adapt its behavior
based on social context.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import math

class SocialIntelligenceLobe:
    """
    Social Intelligence Lobe implementation for user relationship modeling,
    collaboration optimization, empathy simulation, and trust building.
    """
    
    def __init__(self, event_bus=None, hormone_system=None):
        """Initialize the Social Intelligence Lobe."""
        # Set up logging
        self.logger = logging.getLogger("SocialIntelligenceLobe")
        
        # Store references to event bus and hormone system
        self.event_bus = event_bus
        self.hormone_system = hormone_system
        
        # Initialize user relationship models
        self.user_models = {}
        
        # Initialize collaboration metrics
        self.collaboration_metrics = {
            'trust_score': 0.7,  # Initial trust level (0-1)
            'rapport_level': 0.5,  # Initial rapport level (0-1)
            'interaction_history': [],  # History of interactions
            'preference_model': {},  # User preferences
            'communication_style': 'neutral',  # Default communication style
        }
        
        # Initialize empathy model
        self.empathy_model = {
            'user_sentiment': 'neutral',
            'detected_emotions': {},
            'context_awareness': 0.5,
            'adaptation_level': 0.5,
        }
        
        # Initialize trust building metrics
        self.trust_metrics = {
            'consistency_score': 0.8,
            'competence_score': 0.7,
            'transparency_score': 0.9,
            'reliability_history': [],
            'promise_keeping': 1.0,
        }
        
        # Initialize hormone production parameters
        self.hormone_production = {
            'oxytocin': {
                'base_level': 0.3,
                'positive_interaction_boost': 0.2,
                'trust_multiplier': 1.5,
                'current_level': 0.3,
            },
            'vasopressin': {
                'base_level': 0.2,
                'memory_formation_boost': 0.3,
                'relationship_strength_multiplier': 1.2,
                'current_level': 0.2,
            },
            'prolactin': {
                'base_level': 0.1,
                'protective_trigger_boost': 0.4,
                'care_situation_multiplier': 2.0,
                'current_level': 0.1,
            }
        }
        
        # Initialize connected lobes for hormone interactions
        self.connected_lobes = []
        
        # Set up event handlers if event bus is provided
        if self.event_bus:
            self._setup_event_handlers()
        
        self.logger.info("Social Intelligence Lobe initialized")
    
    def _setup_event_handlers(self):
        """Set up event handlers for social interactions."""
        self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
        self.event_bus.subscribe("collaboration_event", self._handle_collaboration_event)
        self.event_bus.subscribe("trust_signal", self._handle_trust_signal)
        self.event_bus.subscribe("emotion_detected", self._handle_emotion_detected)
        
        self.logger.info("Event handlers set up")
    
    def _handle_user_interaction(self, data):
        """Handle user interaction events."""
        user_id = data.get('user_id', 'default_user')
        interaction_type = data.get('type', 'general')
        sentiment = data.get('sentiment', 'neutral')
        
        # Update user model
        self._update_user_model(user_id, interaction_type, data)
        
        # Update interaction history
        self.collaboration_metrics['interaction_history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': interaction_type,
            'sentiment': sentiment,
            'context': data.get('context', {})
        })
        
        # Limit history size
        if len(self.collaboration_metrics['interaction_history']) > 100:
            self.collaboration_metrics['interaction_history'] = self.collaboration_metrics['interaction_history'][-100:]
        
        # Update empathy model based on interaction
        self._update_empathy_model(sentiment, data)
        
        # Produce hormones based on interaction
        self._produce_interaction_hormones(interaction_type, sentiment)
        
        self.logger.info(f"Processed user interaction: {interaction_type} with sentiment: {sentiment}")
    
    def _handle_collaboration_event(self, data):
        """Handle collaboration events."""
        event_type = data.get('type', 'general')
        success_level = data.get('success_level', 0.5)
        
        # Update collaboration metrics
        if event_type == 'successful_collaboration':
            self.collaboration_metrics['trust_score'] = min(1.0, 
                self.collaboration_metrics['trust_score'] + 0.05 * success_level)
            self.collaboration_metrics['rapport_level'] = min(1.0,
                self.collaboration_metrics['rapport_level'] + 0.03 * success_level)
        elif event_type == 'failed_collaboration':
            self.collaboration_metrics['trust_score'] = max(0.0,
                self.collaboration_metrics['trust_score'] - 0.03 * (1 - success_level))
        
        # Produce oxytocin for successful collaboration
        if success_level > 0.7:
            self._produce_hormone('oxytocin', 0.7 * success_level)
        
        self.logger.info(f"Processed collaboration event: {event_type} with success level: {success_level}")
    
    def _handle_trust_signal(self, data):
        """Handle trust-related signals."""
        signal_type = data.get('type', 'neutral')
        strength = data.get('strength', 0.5)
        
        # Update trust metrics
        if signal_type == 'promise_kept':
            self.trust_metrics['promise_keeping'] = min(1.0, 
                self.trust_metrics['promise_keeping'] + 0.05 * strength)
            self.trust_metrics['reliability_history'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'promise_kept',
                'strength': strength
            })
        elif signal_type == 'promise_broken':
            self.trust_metrics['promise_keeping'] = max(0.0,
                self.trust_metrics['promise_keeping'] - 0.1 * strength)
            self.trust_metrics['reliability_history'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'promise_broken',
                'strength': strength
            })
        
        # Limit history size
        if len(self.trust_metrics['reliability_history']) > 50:
            self.trust_metrics['reliability_history'] = self.trust_metrics['reliability_history'][-50:]
        
        # Produce vasopressin for trust building
        if signal_type == 'promise_kept':
            self._produce_hormone('vasopressin', 0.6 * strength)
        
        self.logger.info(f"Processed trust signal: {signal_type} with strength: {strength}")
    
    def _handle_emotion_detected(self, data):
        """Handle detected user emotions."""
        emotion = data.get('emotion', 'neutral')
        intensity = data.get('intensity', 0.5)
        
        # Update empathy model
        self.empathy_model['detected_emotions'][emotion] = intensity
        
        # Update user sentiment based on emotion
        if emotion in ['happy', 'satisfied', 'excited']:
            self.empathy_model['user_sentiment'] = 'positive'
        elif emotion in ['frustrated', 'angry', 'disappointed']:
            self.empathy_model['user_sentiment'] = 'negative'
        else:
            self.empathy_model['user_sentiment'] = 'neutral'
        
        # Produce prolactin for protective care when negative emotions detected
        if emotion in ['frustrated', 'angry', 'disappointed', 'confused']:
            self._produce_hormone('prolactin', 0.7 * intensity)
        
        self.logger.info(f"Processed emotion: {emotion} with intensity: {intensity}")
    
    def _update_user_model(self, user_id, interaction_type, data):
        """Update the user relationship model."""
        if user_id not in self.user_models:
            self.user_models[user_id] = {
                'interaction_count': 0,
                'preferences': {},
                'communication_style': 'neutral',
                'expertise_areas': {},
                'pain_points': {},
                'relationship_strength': 0.5,
                'last_interaction': None,
            }
        
        user_model = self.user_models[user_id]
        user_model['interaction_count'] += 1
        user_model['last_interaction'] = datetime.now().isoformat()
        
        # Update preferences if provided
        if 'preferences' in data:
            for pref_key, pref_value in data['preferences'].items():
                user_model['preferences'][pref_key] = pref_value
        
        # Update communication style if provided
        if 'communication_style' in data:
            user_model['communication_style'] = data['communication_style']
            # Also update global communication style to match current user
            self.collaboration_metrics['communication_style'] = data['communication_style']
        
        # Update expertise areas if provided
        if 'expertise_areas' in data:
            for area, level in data['expertise_areas'].items():
                user_model['expertise_areas'][area] = level
        
        # Update pain points if provided
        if 'pain_points' in data:
            for pain_point, severity in data['pain_points'].items():
                user_model['pain_points'][pain_point] = severity
        
        # Update relationship strength based on interaction frequency and recency
        days_since_last = 0
        if user_model['interaction_count'] > 1:
            try:
                last_dt = datetime.fromisoformat(user_model['last_interaction'])
                days_since_last = (datetime.now() - last_dt).days
            except (ValueError, TypeError):
                days_since_last = 0
        
        # Relationship strength increases with interactions but decays with time
        frequency_factor = min(1.0, user_model['interaction_count'] / 100)
        recency_factor = math.exp(-0.1 * days_since_last) if days_since_last > 0 else 1.0
        
        user_model['relationship_strength'] = 0.3 + (0.7 * frequency_factor * recency_factor)
    
    def _update_empathy_model(self, sentiment, data):
        """Update the empathy model based on interaction."""
        # Update context awareness based on available context data
        context_richness = len(data.get('context', {})) / 10  # Normalize by expected context size
        self.empathy_model['context_awareness'] = min(1.0, 0.3 + (0.7 * context_richness))
        
        # Update adaptation level based on recent interactions
        recent_interactions = self.collaboration_metrics['interaction_history'][-5:] if self.collaboration_metrics['interaction_history'] else []
        sentiment_changes = 0
        
        for i in range(1, len(recent_interactions)):
            if recent_interactions[i].get('sentiment') != recent_interactions[i-1].get('sentiment'):
                sentiment_changes += 1
        
        # More sentiment changes require higher adaptation
        adaptation_need = sentiment_changes / max(1, len(recent_interactions) - 1)
        current_adaptation = self.empathy_model['adaptation_level']
        
        # Gradually move toward needed adaptation level
        self.empathy_model['adaptation_level'] = current_adaptation + 0.2 * (adaptation_need - current_adaptation)
    
    def _produce_interaction_hormones(self, interaction_type, sentiment):
        """Produce hormones based on interaction type and sentiment."""
        if sentiment == 'positive':
            self._produce_hormone('oxytocin', 0.8)
            self._produce_hormone('vasopressin', 0.5)
        elif sentiment == 'negative':
            self._produce_hormone('prolactin', 0.7)
        
        if interaction_type == 'collaborative':
            self._produce_hormone('oxytocin', 0.9)
        elif interaction_type == 'teaching':
            self._produce_hormone('prolactin', 0.8)
        elif interaction_type == 'learning':
            self._produce_hormone('vasopressin', 0.7)
    
    def _produce_hormone(self, hormone_name, trigger_strength):
        """Produce a specific hormone."""
        if hormone_name not in self.hormone_production:
            self.logger.warning(f"Unknown hormone: {hormone_name}")
            return
        
        hormone = self.hormone_production[hormone_name]
        base_level = hormone['base_level']
        
        # Calculate release amount based on trigger strength and base level
        release_amount = base_level * trigger_strength
        
        # Apply specific multipliers based on hormone type
        if hormone_name == 'oxytocin':
            release_amount *= hormone['trust_multiplier'] * self.collaboration_metrics['trust_score']
        elif hormone_name == 'vasopressin':
            user_id = 'default_user'
            relationship_strength = self.user_models.get(user_id, {}).get('relationship_strength', 0.5)
            release_amount *= hormone['relationship_strength_multiplier'] * relationship_strength
        elif hormone_name == 'prolactin':
            if self.empathy_model['user_sentiment'] == 'negative':
                release_amount *= hormone['care_situation_multiplier']
        
        # Update current level
        hormone['current_level'] = min(1.0, hormone['current_level'] + release_amount)
        
        # Release hormone through hormone system if available
        if self.hormone_system:
            context = {
                'source': 'social_intelligence_lobe',
                'trigger_strength': trigger_strength,
                'user_sentiment': self.empathy_model['user_sentiment']
            }
            self.hormone_system.release_hormone('social_intelligence', hormone_name, release_amount, context)
            self.logger.info(f"Released {hormone_name} with amount {release_amount:.2f}")
        else:
            self.logger.info(f"Would release {hormone_name} with amount {release_amount:.2f} (no hormone system)")
    
    def process_hormone_reception(self, hormone_name, amount, source_lobe):
        """Process received hormones from other lobes."""
        self.logger.info(f"Received {hormone_name} with amount {amount:.2f} from {source_lobe}")
        
        # Process different hormones
        if hormone_name == 'dopamine':
            # Dopamine from task completion increases trust
            self.collaboration_metrics['trust_score'] = min(1.0, 
                self.collaboration_metrics['trust_score'] + 0.05 * amount)
        elif hormone_name == 'serotonin':
            # Serotonin from decision making increases rapport
            self.collaboration_metrics['rapport_level'] = min(1.0,
                self.collaboration_metrics['rapport_level'] + 0.03 * amount)
        elif hormone_name == 'cortisol':
            # Cortisol from stress decreases trust slightly
            self.collaboration_metrics['trust_score'] = max(0.0,
                self.collaboration_metrics['trust_score'] - 0.02 * amount)
    
    def decay_hormones(self, decay_rate=0.1):
        """Apply natural decay to hormone levels."""
        for hormone_name, hormone in self.hormone_production.items():
            hormone['current_level'] = max(hormone['base_level'], 
                hormone['current_level'] * (1 - decay_rate))
    
    def get_communication_style(self, user_id='default_user'):
        """Get the appropriate communication style for a user."""
        if user_id in self.user_models:
            return self.user_models[user_id].get('communication_style', 'neutral')
        return self.collaboration_metrics['communication_style']
    
    def get_trust_level(self, user_id='default_user'):
        """Get the current trust level for a user."""
        # Combine global trust metrics with user-specific relationship strength
        global_trust = self.collaboration_metrics['trust_score']
        
        if user_id in self.user_models:
            relationship_strength = self.user_models[user_id].get('relationship_strength', 0.5)
            return (global_trust + relationship_strength) / 2
        
        return global_trust
    
    def get_empathy_response(self, user_sentiment=None):
        """Get an appropriate empathetic response based on user sentiment."""
        if user_sentiment is None:
            user_sentiment = self.empathy_model['user_sentiment']
        
        response_type = 'neutral'
        
        if user_sentiment == 'positive':
            response_type = 'encouraging'
        elif user_sentiment == 'negative':
            response_type = 'supportive'
        
        # Adjust based on adaptation level
        if self.empathy_model['adaptation_level'] > 0.7:
            if response_type == 'encouraging':
                response_type = 'enthusiastic'
            elif response_type == 'supportive':
                response_type = 'problem_solving'
        
        return response_type
    
    def optimize_collaboration(self, task_type, user_id='default_user'):
        """Optimize collaboration approach for a specific task and user."""
        user_model = self.user_models.get(user_id, {})
        expertise_level = user_model.get('expertise_areas', {}).get(task_type, 0.5)
        
        collaboration_approach = {
            'guidance_level': 0.5,  # Default guidance level
            'autonomy_level': 0.5,  # Default autonomy level
            'explanation_depth': 0.5,  # Default explanation depth
            'check_frequency': 0.5,  # Default check-in frequency
        }
        
        # Adjust based on expertise level
        if expertise_level < 0.3:
            # Novice: more guidance, less autonomy, deeper explanations
            collaboration_approach['guidance_level'] = 0.8
            collaboration_approach['autonomy_level'] = 0.3
            collaboration_approach['explanation_depth'] = 0.9
            collaboration_approach['check_frequency'] = 0.7
        elif expertise_level > 0.7:
            # Expert: less guidance, more autonomy, briefer explanations
            collaboration_approach['guidance_level'] = 0.3
            collaboration_approach['autonomy_level'] = 0.9
            collaboration_approach['explanation_depth'] = 0.4
            collaboration_approach['check_frequency'] = 0.3
        
        # Adjust based on trust level
        trust_level = self.get_trust_level(user_id)
        if trust_level > 0.8:
            # High trust: more autonomy, less checking
            collaboration_approach['autonomy_level'] = min(1.0, collaboration_approach['autonomy_level'] + 0.1)
            collaboration_approach['check_frequency'] = max(0.1, collaboration_approach['check_frequency'] - 0.1)
        elif trust_level < 0.4:
            # Low trust: more checking, less autonomy
            collaboration_approach['autonomy_level'] = max(0.1, collaboration_approach['autonomy_level'] - 0.1)
            collaboration_approach['check_frequency'] = min(0.9, collaboration_approach['check_frequency'] + 0.2)
        
        return collaboration_approach
    
    def build_trust(self, action_type):
        """Execute trust-building actions."""
        if action_type == 'consistency':
            self.trust_metrics['consistency_score'] = min(1.0, self.trust_metrics['consistency_score'] + 0.05)
        elif action_type == 'competence':
            self.trust_metrics['competence_score'] = min(1.0, self.trust_metrics['competence_score'] + 0.05)
        elif action_type == 'transparency':
            self.trust_metrics['transparency_score'] = min(1.0, self.trust_metrics['transparency_score'] + 0.05)
        
        # Overall trust is a combination of all metrics
        overall_trust = (
            self.trust_metrics['consistency_score'] * 0.3 +
            self.trust_metrics['competence_score'] * 0.4 +
            self.trust_metrics['transparency_score'] * 0.2 +
            self.trust_metrics['promise_keeping'] * 0.1
        )
        
        self.collaboration_metrics['trust_score'] = overall_trust
        return overall_trust
    
    def simulate_empathy(self, user_state):
        """Simulate empathy for a given user state."""
        # Update empathy model based on user state
        self.empathy_model['user_sentiment'] = user_state.get('sentiment', 'neutral')
        self.empathy_model['detected_emotions'] = user_state.get('emotions', {})
        
        # Calculate empathy response
        response_type = self.get_empathy_response()
        
        # Produce appropriate hormones based on user state
        if user_state.get('sentiment') == 'negative':
            self._produce_hormone('prolactin', 0.8)
        elif user_state.get('sentiment') == 'positive':
            self._produce_hormone('oxytocin', 0.7)
        
        return {
            'response_type': response_type,
            'adaptation_level': self.empathy_model['adaptation_level'],
            'context_awareness': self.empathy_model['context_awareness']
        }
    
    def get_state(self):
        """Get the current state of the Social Intelligence Lobe."""
        return {
            'collaboration_metrics': self.collaboration_metrics,
            'empathy_model': self.empathy_model,
            'trust_metrics': self.trust_metrics,
            'hormone_levels': {name: data['current_level'] for name, data in self.hormone_production.items()},
            'user_count': len(self.user_models)
        }
    
    def save_state(self, file_path):
        """Save the current state to a file."""
        try:
            state = {
                'collaboration_metrics': self.collaboration_metrics,
                'empathy_model': self.empathy_model,
                'trust_metrics': self.trust_metrics,
                'hormone_production': self.hormone_production,
                'user_models': self.user_models
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"State saved to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self, file_path):
        """Load state from a file."""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"State file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.collaboration_metrics = state.get('collaboration_metrics', self.collaboration_metrics)
            self.empathy_model = state.get('empathy_model', self.empathy_model)
            self.trust_metrics = state.get('trust_metrics', self.trust_metrics)
            self.hormone_production = state.get('hormone_production', self.hormone_production)
            self.user_models = state.get('user_models', self.user_models)
            
            self.logger.info(f"State loaded from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Social Intelligence Lobe
    social_lobe = SocialIntelligenceLobe()
    
    # Simulate some interactions
    social_lobe._handle_user_interaction({
        'user_id': 'test_user',
        'type': 'collaborative',
        'sentiment': 'positive',
        'context': {'task': 'coding', 'project': 'MCP'},
        'preferences': {'communication_style': 'direct'}
    })
    
    social_lobe._handle_emotion_detected({
        'emotion': 'satisfied',
        'intensity': 0.8
    })
    
    social_lobe._handle_collaboration_event({
        'type': 'successful_collaboration',
        'success_level': 0.9
    })
    
    # Get collaboration approach
    approach = social_lobe.optimize_collaboration('coding', 'test_user')
    print("Optimized collaboration approach:", approach)
    
    # Get empathy response
    response = social_lobe.get_empathy_response()
    print("Empathy response type:", response)
    
    # Get current state
    state = social_lobe.get_state()
    print("Current hormone levels:", state['hormone_levels'])