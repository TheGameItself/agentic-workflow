#!/usr/bin/env python3
"""
Test suite for the Genetic Trigger System

This test file validates all components of the genetic trigger system including:
- Environmental state encoding
- Genetic sequence operations
- Codon-based activation patterns
- Epigenetic memory mechanisms
- Evolutionary adaptation and natural selection
- Population management and evolution
"""

import asyncio
import json
import os
import random
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from mcp.genetic_trigger_system import (
    EnvironmentalState,
    EpigeneticMarker,
    EpigeneticMemory,
    GeneticSequence,
    GeneticTrigger,
    GeneticTriggerPopulation,
    GeneticTriggerSystem,
    GeneticCodon
)


class TestEnvironmentalState(unittest.TestCase):
    """Test environmental state encoding and operations"""
    
    def test_environmental_state_creation(self):
        """Test creating environmental state"""
        env = EnvironmentalState(
            task_complexity=0.8,
            user_satisfaction=0.6,
            system_load=0.4,
            error_rate=0.2
        )
        
        self.assertEqual(env.task_complexity, 0.8)
        self.assertEqual(env.user_satisfaction, 0.6)
        self.assertIsInstance(env.timestamp, float)
    
    def test_environmental_state_serialization(self):
        """Test serialization and deserialization"""
        env = EnvironmentalState(
            task_complexity=0.7,
            creativity_demand=0.9,
            context_switches=5
        )
        
        # Test to_dict
        env_dict = env.to_dict()
        self.assertIsInstance(env_dict, dict)
        self.assertEqual(env_dict['task_complexity'], 0.7)
        self.assertEqual(env_dict['creativity_demand'], 0.9)
        
        # Test from_dict
        env_restored = EnvironmentalState.from_dict(env_dict)
        self.assertEqual(env_restored.task_complexity, 0.7)
        self.assertEqual(env_restored.creativity_demand, 0.9)
        self.assertEqual(env_restored.context_switches, 5)


class TestEpigeneticMemory(unittest.TestCase):
    """Test epigenetic memory mechanisms"""
    
    def test_epigenetic_marker_creation(self):
        """Test creating epigenetic markers"""
        marker = EpigeneticMarker()
        
        # Test default values
        self.assertIsInstance(marker.methylation_pattern, dict)
        self.assertIsInstance(marker.histone_modifications, dict)
        self.assertIsInstance(marker.stress_markers, dict)
    
    def test_expression_modifier_calculation(self):
        """Test expression modifier calculation"""
        marker = EpigeneticMarker()
        
        # Add some modifications
        marker.methylation_pattern['gene1'] = 0.8
        marker.histone_modifications['gene1'] = 0.5
        marker.stress_markers['gene1'] = 0.9
        
        modifier = marker.get_expression_modifier('gene1')
        self.assertIsInstance(modifier, float)
        self.assertGreater(modifier, 0.1)
        self.assertLess(modifier, 3.0)
    
    def test_epigenetic_memory_operations(self):
        """Test epigenetic memory operations"""
        memory = EpigeneticMemory()
        
        # Test stress marker addition
        memory.add_stress_marker('stress_gene', 0.9)
        self.assertEqual(memory.markers.stress_markers['stress_gene'], 0.9)
        
        # Test learning marker addition
        memory.add_learning_marker('learning_gene', 0.8)
        self.assertIn('learning_gene', memory.markers.histone_modifications)
    
    def test_epigenetic_inheritance(self):
        """Test inheritance mechanisms"""
        parent_memory = EpigeneticMemory()
        parent_memory.markers.methylation_pattern['inherited_gene'] = 0.7
        parent_memory.markers.imprinting_markers['imprint_gene'] = True
        
        child_memory = EpigeneticMemory()
        child_memory.inherit_from_parent(parent_memory)
        
        # Check inheritance occurred
        self.assertTrue(len(child_memory.inheritance_history) > 0)
        # Methylation should be inherited with some reduction
        if 'inherited_gene' in child_memory.markers.methylation_pattern:
            self.assertLess(
                child_memory.markers.methylation_pattern['inherited_gene'],
                parent_memory.markers.methylation_pattern['inherited_gene']
            )


class TestGeneticSequence(unittest.TestCase):
    """Test genetic sequence operations"""
    
    def test_sequence_creation(self):
        """Test creating genetic sequences"""
        seq = GeneticSequence("ATGCGATCGTAGC")
        
        self.assertEqual(seq.sequence, "ATGCGATCGTAGC")
        self.assertIsInstance(seq.codons, list)
        self.assertGreater(len(seq.codons), 0)
    
    def test_codon_extraction(self):
        """Test codon extraction from sequence"""
        seq = GeneticSequence("ATGCGATCGTAG")
        
        expected_codons = ["ATG", "CGA", "TCG", "TAG"]
        self.assertEqual(seq.codons, expected_codons)
    
    def test_regulatory_regions(self):
        """Test promoter and enhancer regions"""
        seq = GeneticSequence("ATGCGATCGTAGC")
        
        seq.add_promoter(0, 0.8)
        seq.add_enhancer(6, 0.6)
        
        self.assertEqual(seq.promoter_regions[0], 0.8)
        self.assertEqual(seq.enhancer_regions[6], 0.6)
        
        # Test expression level calculation
        expression = seq.get_expression_level(3)
        self.assertIsInstance(expression, float)
        self.assertGreater(expression, 0.0)
    
    def test_mutation(self):
        """Test genetic mutation"""
        original_seq = GeneticSequence("ATGCGATCGTAGC")
        mutated_seq = original_seq.mutate(mutation_rate=0.5)  # High rate for testing
        
        self.assertIsInstance(mutated_seq, GeneticSequence)
        # Sequences should be different with high mutation rate
        self.assertNotEqual(original_seq.sequence, mutated_seq.sequence)
    
    def test_crossover(self):
        """Test genetic crossover"""
        seq1 = GeneticSequence("ATGCGATCG")
        seq2 = GeneticSequence("TAGCGATAG")
        
        child1, child2 = seq1.crossover(seq2)
        
        self.assertIsInstance(child1, GeneticSequence)
        self.assertIsInstance(child2, GeneticSequence)
        self.assertNotEqual(child1.sequence, seq1.sequence)
        self.assertNotEqual(child2.sequence, seq2.sequence)


class TestGeneticTrigger(unittest.TestCase):
    """Test genetic trigger functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_env = EnvironmentalState(
            task_complexity=0.7,
            user_satisfaction=0.8,
            system_load=0.3,
            error_rate=0.1,
            learning_rate=0.6
        )
    
    def test_trigger_creation(self):
        """Test creating genetic triggers"""
        trigger = GeneticTrigger(self.test_env)
        
        self.assertIsInstance(trigger.trigger_id, str)
        self.assertEqual(trigger.formation_environment, self.test_env)
        self.assertIsInstance(trigger.dna_signature, str)
        self.assertIsInstance(trigger.genetic_sequence, GeneticSequence)
        self.assertIsInstance(trigger.codon_map, dict)
    
    def test_environment_encoding(self):
        """Test environmental state encoding to DNA"""
        trigger = GeneticTrigger(self.test_env)
        
        dna_signature = trigger.dna_signature
        self.assertIsInstance(dna_signature, str)
        self.assertGreater(len(dna_signature), 0)
        # Should only contain valid DNA bases
        valid_bases = set('ATGC')
        self.assertTrue(all(base in valid_bases for base in dna_signature))
    
    def test_activation_decision(self):
        """Test trigger activation logic"""
        trigger = GeneticTrigger(self.test_env)
        
        # Test with same environment (should have high similarity)
        should_activate = trigger.should_activate(self.test_env)
        self.assertIsInstance(should_activate, bool)
        
        # Test with different environment
        different_env = EnvironmentalState(
            task_complexity=0.1,
            user_satisfaction=0.2,
            system_load=0.9,
            error_rate=0.8
        )
        
        different_activation = trigger.should_activate(different_env)
        self.assertIsInstance(different_activation, bool)
        
        # Check activation history was recorded
        self.assertGreater(len(trigger.activation_history), 0)
    
    def test_fitness_updates(self):
        """Test fitness score updates"""
        trigger = GeneticTrigger(self.test_env)
        initial_fitness = trigger.fitness_score
        
        # Update with good performance
        trigger.update_fitness(0.9)
        self.assertGreater(trigger.fitness_score, initial_fitness)
        
        # Check performance history
        self.assertGreater(len(trigger.performance_history), 0)
        self.assertEqual(trigger.performance_history[-1]['performance'], 0.9)
    
    def test_trigger_mutation(self):
        """Test trigger mutation"""
        original_trigger = GeneticTrigger(self.test_env)
        mutated_trigger = original_trigger.mutate(mutation_rate=0.1)
        
        self.assertIsInstance(mutated_trigger, GeneticTrigger)
        self.assertNotEqual(mutated_trigger.trigger_id, original_trigger.trigger_id)
        self.assertEqual(mutated_trigger.generation, original_trigger.generation + 1)
        self.assertIn(original_trigger.trigger_id, mutated_trigger.parent_ids)
    
    def test_trigger_crossover(self):
        """Test trigger crossover"""
        trigger1 = GeneticTrigger(self.test_env)
        trigger2 = GeneticTrigger(self.test_env)
        
        child1, child2 = trigger1.crossover(trigger2)
        
        self.assertIsInstance(child1, GeneticTrigger)
        self.assertIsInstance(child2, GeneticTrigger)
        self.assertIn(trigger1.trigger_id, child1.parent_ids)
        self.assertIn(trigger2.trigger_id, child1.parent_ids)
    
    def test_trigger_serialization(self):
        """Test trigger serialization"""
        trigger = GeneticTrigger(self.test_env)
        
        # Test to_dict
        trigger_dict = trigger.to_dict()
        self.assertIsInstance(trigger_dict, dict)
        self.assertEqual(trigger_dict['trigger_id'], trigger.trigger_id)
        
        # Test from_dict
        restored_trigger = GeneticTrigger.from_dict(trigger_dict)
        self.assertEqual(restored_trigger.trigger_id, trigger.trigger_id)
        self.assertEqual(restored_trigger.fitness_score, trigger.fitness_score)


class TestGeneticTriggerPopulation(unittest.TestCase):
    """Test population management and evolution"""
    
    def setUp(self):
        """Set up test population"""
        # Use temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.population = GeneticTriggerPopulation(
            population_size=20,
            database_path=self.temp_db.name
        )
    
    def tearDown(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_population_initialization(self):
        """Test population initialization"""
        self.assertGreater(len(self.population.triggers), 0)
        self.assertLessEqual(len(self.population.triggers), self.population.population_size)
    
    def test_trigger_evaluation(self):
        """Test trigger evaluation against environment"""
        test_env = EnvironmentalState(
            task_complexity=0.5,
            user_satisfaction=0.7,
            error_rate=0.2
        )
        
        activated_triggers = self.population.evaluate_triggers(test_env)
        self.assertIsInstance(activated_triggers, list)
        # All activated triggers should be from the population
        for trigger in activated_triggers:
            self.assertIn(trigger, self.population.triggers)
    
    def test_fitness_updates(self):
        """Test fitness updates for population"""
        if len(self.population.triggers) > 0:
            trigger = self.population.triggers[0]
            original_fitness = trigger.fitness_score
            
            self.population.update_fitness(trigger.trigger_id, 0.9)
            self.assertGreater(trigger.fitness_score, original_fitness)
    
    def test_population_evolution(self):
        """Test population evolution"""
        # Ensure we have enough triggers
        while len(self.population.triggers) < 10:
            env = EnvironmentalState(
                task_complexity=random.random(),
                user_satisfaction=random.random()
            )
            trigger = GeneticTrigger(env)
            self.population.add_trigger(trigger)
        
        # Set some fitness scores
        for i, trigger in enumerate(self.population.triggers[:5]):
            trigger.fitness_score = 0.8 + i * 0.02  # High fitness
        
        for i, trigger in enumerate(self.population.triggers[5:10]):
            trigger.fitness_score = 0.2 + i * 0.02  # Low fitness
        
        original_generation = self.population.generation
        self.population.evolve_population()
        
        self.assertEqual(self.population.generation, original_generation + 1)
        self.assertLessEqual(len(self.population.triggers), self.population.population_size)
    
    def test_population_stats(self):
        """Test population statistics"""
        stats = self.population.get_population_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('population_size', stats)
        self.assertIn('avg_fitness', stats)
        self.assertIn('genetic_diversity', stats)
        
        self.assertEqual(stats['population_size'], len(self.population.triggers))
    
    def test_horizontal_gene_transfer(self):
        """Test horizontal gene transfer"""
        if len(self.population.triggers) >= 2:
            source = self.population.triggers[0]
            target = self.population.triggers[1]
            
            # Set up source with high-performing codon
            source.codon_map['ATG'] = 0.9
            original_target_codon = target.codon_map.get('ATG', 0.5)
            
            self.population.horizontal_gene_transfer(source, target, transfer_rate=1.0)
            
            # Target should have been influenced by source
            if 'ATG' in target.codon_map:
                self.assertNotEqual(target.codon_map['ATG'], original_target_codon)


class TestGeneticTriggerSystem(unittest.TestCase):
    """Test the complete genetic trigger system"""
    
    def setUp(self):
        """Set up test system"""
        # Use temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.system = GeneticTriggerSystem(database_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsInstance(self.system.population, GeneticTriggerPopulation)
        self.assertIsInstance(self.system.current_environment, EnvironmentalState)
        self.assertIsInstance(self.system.active_triggers, list)
    
    async def test_environment_updates(self):
        """Test environment updates"""
        await self.system.update_environment(
            task_complexity=0.8,
            user_satisfaction=0.6,
            error_rate=0.3
        )
        
        self.assertEqual(self.system.current_environment.task_complexity, 0.8)
        self.assertEqual(self.system.current_environment.user_satisfaction, 0.6)
        self.assertEqual(self.system.current_environment.error_rate, 0.3)
    
    async def test_trigger_evaluation(self):
        """Test trigger evaluation"""
        # Update environment
        await self.system.update_environment(
            task_complexity=0.7,
            creativity_demand=0.8
        )
        
        # Should have evaluated triggers
        self.assertIsInstance(self.system.active_triggers, list)
        self.assertGreater(self.system.evaluation_count, 0)
    
    def test_performance_reporting(self):
        """Test performance reporting"""
        if len(self.system.population.triggers) > 0:
            trigger = self.system.population.triggers[0]
            original_fitness = trigger.fitness_score
            
            self.system.report_performance(trigger.trigger_id, 0.9)
            self.assertGreater(trigger.fitness_score, original_fitness)
    
    def test_trigger_creation(self):
        """Test creating triggers from environment"""
        env = EnvironmentalState(
            task_complexity=0.6,
            creativity_demand=0.8
        )
        
        original_count = len(self.system.population.triggers)
        new_trigger = self.system.create_trigger_from_environment(env)
        
        self.assertIsInstance(new_trigger, GeneticTrigger)
        self.assertEqual(len(self.system.population.triggers), original_count + 1)
    
    def test_system_stats(self):
        """Test system statistics"""
        stats = self.system.get_system_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('population_stats', stats)
        self.assertIn('current_environment', stats)
        self.assertIn('active_triggers', stats)
        self.assertIn('evaluation_count', stats)
    
    def test_active_trigger_behaviors(self):
        """Test getting active trigger behaviors"""
        # Create a trigger and make it active
        env = EnvironmentalState(task_complexity=0.8)
        trigger = GeneticTrigger(env)
        self.system.active_triggers = [trigger]
        
        behaviors = self.system.get_active_trigger_behaviors()
        
        self.assertIsInstance(behaviors, dict)
        if trigger.trigger_id in behaviors:
            self.assertIsInstance(behaviors[trigger.trigger_id], list)


class TestGeneticCodons(unittest.TestCase):
    """Test genetic codon mappings"""
    
    def test_codon_enum(self):
        """Test genetic codon enumeration"""
        # Test start codon
        self.assertEqual(GeneticCodon.ATG.value, "start_pathway")
        
        # Test stop codons
        self.assertEqual(GeneticCodon.TAA.value, "stop_process")
        self.assertEqual(GeneticCodon.TAG.value, "pause_process")
        self.assertEqual(GeneticCodon.TGA.value, "halt_process")
        
        # Test behavior codons
        self.assertEqual(GeneticCodon.GCN.value, "structural_stability")
        self.assertEqual(GeneticCodon.CGN.value, "activation_signal")
    
    def test_codon_coverage(self):
        """Test that we have good codon coverage"""
        codon_values = [codon.value for codon in GeneticCodon]
        
        # Should have diverse behaviors
        self.assertIn("learning_initiation", codon_values)
        self.assertIn("pattern_recognition", codon_values)
        self.assertIn("decision_making", codon_values)
        self.assertIn("creative_synthesis", codon_values)
        self.assertIn("error_detection", codon_values)


async def run_integration_test():
    """Run integration test of the complete system"""
    print("Running Genetic Trigger System Integration Test...")
    
    # Create system
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        system = GeneticTriggerSystem(database_path=temp_db.name)
        
        # Test different environmental scenarios
        scenarios = [
            {
                'name': 'High Complexity Task',
                'env': {'task_complexity': 0.9, 'user_satisfaction': 0.5, 'error_rate': 0.3}
            },
            {
                'name': 'Creative Work',
                'env': {'creativity_demand': 0.8, 'collaboration_level': 0.6, 'time_pressure': 0.4}
            },
            {
                'name': 'System Stress',
                'env': {'system_load': 0.9, 'error_rate': 0.7, 'resource_availability': 0.2}
            },
            {
                'name': 'Learning Phase',
                'env': {'learning_rate': 0.8, 'user_satisfaction': 0.9, 'task_complexity': 0.6}
            }
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Update environment
            await system.update_environment(**scenario['env'])
            
            # Check active triggers
            active_count = len(system.active_triggers)
            print(f"Active triggers: {active_count}")
            
            # Get behaviors
            behaviors = system.get_active_trigger_behaviors()
            for trigger_id, behavior_list in list(behaviors.items())[:3]:  # Show first 3
                print(f"Trigger {trigger_id[:8]}...: {behavior_list[:2]}")
            
            # Simulate performance feedback
            for trigger in system.active_triggers[:3]:  # Feedback for first 3
                performance = random.uniform(0.4, 0.9)
                system.report_performance(trigger.trigger_id, performance)
                print(f"Performance feedback: {performance:.3f}")
        
        # Force evolution
        print(f"\n--- Evolution ---")
        await system.evolve_population()
        
        # Final stats
        stats = system.get_system_stats()
        pop_stats = stats['population_stats']
        print(f"Final population size: {pop_stats['population_size']}")
        print(f"Average fitness: {pop_stats['avg_fitness']:.3f}")
        print(f"Genetic diversity: {pop_stats['genetic_diversity']:.3f}")
        print(f"Total evaluations: {stats['evaluation_count']}")
        
        print("\nIntegration test completed successfully!")
        
    finally:
        # Clean up
        try:
            os.unlink(temp_db.name)
        except:
            pass


if __name__ == '__main__':
    # Run unit tests
    print("Running Genetic Trigger System Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*60)
    asyncio.run(run_integration_test())