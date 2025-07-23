"""
Split-brain A/B test for genetic trigger system: left vs right lobe comparison.
"""
import pytest
import sys
import os

# Add src root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from mcp.left_lobes.genetic_trigger import GeneticTrigger as LeftGeneticTrigger
from mcp.right_lobes.genetic_trigger import GeneticTrigger as RightGeneticTrigger

@pytest.fixture
def sample_environment():
    return {"temp": 25.0, "humidity": 0.5, "active": True, "label": "test"}

@pytest.fixture
def sample_pathway():
    return {"neuron_count": 128, "activation": "relu"}

def test_split_brain_ab_activation_and_mutation(sample_environment, sample_pathway):
    left = LeftGeneticTrigger(sample_environment, sample_pathway)
    right = RightGeneticTrigger(sample_environment, sample_pathway)
    # Run activation (sync for code path)
    left_result = left.code_impl.evaluate_activation(sample_environment, left.formation_environment)
    right_result = right.code_impl.evaluate_activation(sample_environment, right.formation_environment)
    # Mutate both
    left_mutated = left.mutate()
    right_mutated = right.mutate()
    # Compare results
    assert isinstance(left_result, bool)
    assert isinstance(right_result, bool)
    assert left_mutated is not None
    assert right_mutated is not None
    # Log for manual review
    print(f"Left activation: {left_result}, Right activation: {right_result}")
    print(f"Left mutated env: {left_mutated.formation_environment}")
    print(f"Right mutated env: {right_mutated.formation_environment}")

def test_split_brain_ab_performance(sample_environment, sample_pathway):
    left_activations = 0
    right_activations = 0
    left_mutation_rates = []
    right_mutation_rates = []
    left_fitness = []
    right_fitness = []
    rounds = 100
    for _ in range(rounds):
        left = LeftGeneticTrigger(sample_environment, sample_pathway)
        right = RightGeneticTrigger(sample_environment, sample_pathway)
        # Activation
        if left.code_impl.evaluate_activation(sample_environment, left.formation_environment):
            left_activations += 1
        if right.code_impl.evaluate_activation(sample_environment, right.formation_environment):
            right_activations += 1
        # Mutation
        left_mutated = left.mutate()
        right_mutated = right.mutate()
        left_mutation_rates.append(left_mutated.mutation_rate)
        right_mutation_rates.append(right_mutated.mutation_rate)
        left_fitness.append(left_mutated.fitness_score)
        right_fitness.append(right_mutated.fitness_score)
    # Print summary
    print(f"Left activations: {left_activations}/{rounds}")
    print(f"Right activations: {right_activations}/{rounds}")
    print(f"Left avg mutation rate: {sum(left_mutation_rates)/rounds:.4f}")
    print(f"Right avg mutation rate: {sum(right_mutation_rates)/rounds:.4f}")
    print(f"Left avg fitness: {sum(left_fitness)/rounds:.4f}")
    print(f"Right avg fitness: {sum(right_fitness)/rounds:.4f}")
    # Assert both lobes are exercised
    assert left_activations > 0
    assert right_activations > 0
    assert len(left_mutation_rates) == rounds
    assert len(right_mutation_rates) == rounds

@pytest.mark.asyncio
async def test_resource_aware_scheduling(sample_environment, sample_pathway, caplog):
    from mcp.left_lobes.genetic_trigger import GeneticTrigger as LeftGeneticTrigger
    # Low resource: should not adapt
    left = LeftGeneticTrigger(sample_environment, sample_pathway)
    with caplog.at_level('INFO'):
        left.force_resource_adaptation({'cpu': 0.2, 'memory': 0.2, 'activity': 0.5})
    assert left.mutation_rate == 0.05
    # High resource: should adapt
    left2 = LeftGeneticTrigger(sample_environment, sample_pathway)
    with caplog.at_level('INFO'):
        left2.force_resource_adaptation({'cpu': 0.9, 'memory': 0.9, 'activity': 0.9})
    assert left2.mutation_rate > 0.05
    assert any('[TEST] Resource high' in r for r in caplog.text.splitlines())
    # Very high memory: should trigger consolidation
    left3 = LeftGeneticTrigger(sample_environment, sample_pathway)
    with caplog.at_level('INFO'):
        left3.force_resource_adaptation({'cpu': 0.5, 'memory': 0.99, 'activity': 0.5})
    assert any('[TEST] Memory very high' in r for r in caplog.text.splitlines())

def test_multi_point_crossover_and_fitness(sample_environment, sample_pathway):
    from mcp.left_lobes.genetic_trigger import GeneticTrigger as LeftGeneticTrigger
    parent1 = LeftGeneticTrigger(sample_environment, sample_pathway)
    parent2 = LeftGeneticTrigger({"temp": 30.0, "humidity": 0.7, "active": False, "label": "alt"}, {"neuron_count": 256, "activation": "sigmoid"})
    child = LeftGeneticTrigger.multi_point_crossover(parent1, parent2, points=2)
    assert isinstance(child, LeftGeneticTrigger)
    # Fitness landscape
    test_envs = [sample_environment, {"temp": 30.0, "humidity": 0.7, "active": False, "label": "alt"}]
    fitness = child.evaluate_fitness_landscape(test_envs)
    assert 0.0 <= fitness <= 1.0

def test_crossover_edge_cases():
    from mcp.left_lobes.genetic_trigger import GeneticTrigger as LeftGeneticTrigger
    # Both empty
    parent1 = LeftGeneticTrigger({}, {})
    parent2 = LeftGeneticTrigger({}, {})
    child = LeftGeneticTrigger.multi_point_crossover(parent1, parent2, points=2)
    assert isinstance(child, LeftGeneticTrigger)
    # Mismatched keys
    parent3 = LeftGeneticTrigger({"a": 1}, {"x": 1})
    parent4 = LeftGeneticTrigger({"b": 2, "c": 3}, {"y": 2})
    child2 = LeftGeneticTrigger.multi_point_crossover(parent3, parent4, points=2)
    assert isinstance(child2, LeftGeneticTrigger)
    # Short environments
    parent5 = LeftGeneticTrigger({"a": 1}, {"x": 1})
    parent6 = LeftGeneticTrigger({"b": 2}, {"y": 2})
    child3 = LeftGeneticTrigger.multi_point_crossover(parent5, parent6, points=2)
    assert isinstance(child3, LeftGeneticTrigger) 