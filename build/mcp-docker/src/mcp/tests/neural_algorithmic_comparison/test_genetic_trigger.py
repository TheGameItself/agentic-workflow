"""
Tests for the GeneticTrigger system (dual code/neural, hormone, mutation, epigenetic integration).
"""
import sys
import os
import pytest

# Ensure src root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from mcp.genetic_trigger_system.genetic_trigger import GeneticTrigger
from mcp.genetic_data_exchange import GeneticDataExchange, GeneticChromosome, GeneticElement, GeneticElementType, ChromatinState

@pytest.fixture
def sample_environment():
    return {"temp": 25.0, "humidity": 0.5, "active": True, "label": "test"}

@pytest.fixture
def sample_pathway():
    return {"neuron_count": 128, "activation": "relu"}

def test_genetic_trigger_instantiation(sample_environment, sample_pathway):
    trigger = GeneticTrigger(sample_environment, sample_pathway)
    assert trigger.dna_signature
    assert trigger.pathway_config == sample_pathway
    assert trigger.formation_environment == sample_environment

def test_similarity_self(sample_environment, sample_pathway):
    trigger = GeneticTrigger(sample_environment, sample_pathway)
    sim = trigger.calculate_similarity(sample_environment)
    assert sim == pytest.approx(1.0, abs=1e-6)

def test_mutate_returns_new_trigger(sample_environment, sample_pathway):
    trigger = GeneticTrigger(sample_environment, sample_pathway)
    mutated = trigger.mutate()
    assert mutated is not trigger
    assert mutated.parent_ids == [trigger.id]

def test_to_dict_and_from_dict(sample_environment, sample_pathway):
    trigger = GeneticTrigger(sample_environment, sample_pathway)
    d = trigger.to_dict()
    restored = GeneticTrigger.from_dict(d)
    assert restored.dna_signature == trigger.dna_signature
    assert restored.pathway_config == trigger.pathway_config
    assert restored.formation_environment == trigger.formation_environment

def test_codon_encoding_decoding(sample_environment):
    codon = GeneticTrigger.encode_dict_to_codon(sample_environment)
    decoded = GeneticTrigger.decode_codon_to_dict(codon)
    assert decoded == sample_environment

def test_epigenetic_marker_storage(sample_environment, sample_pathway):
    trigger = GeneticTrigger(sample_environment, sample_pathway)
    marker_key = "test_marker"
    codon = trigger.encode_dict_to_codon(sample_environment)
    trigger.epigenetic_markers.set_marker(marker_key, codon)
    retrieved = trigger.epigenetic_markers.get_marker(marker_key)
    decoded = trigger.decode_codon_to_dict(retrieved)
    assert decoded == sample_environment

def test_prompt_circuit_expression_architecture():
    gde = GeneticDataExchange(organism_id="test")
    circuit = {
        "nodes": [
            {"id": "n1", "type": "input"},
            {"id": "n2", "type": "hidden"},
            {"id": "n3", "type": "output"}
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"}
        ]
    }
    chrom = gde.encode_prompt_circuit(
        circuit,
        layout="hierarchical",
        interruption_points=[1],
        alignment_hooks=["cross_lobe_sync"],
        reputation=0.9,
        universal_hooks=["external_integration"]
    )
    # Check that special elements are present
    ids = [e.element_id for e in chrom.elements]
    assert "layout" in ids
    assert "interrupt_1" in ids
    assert "align_cross_lobe_sync" in ids
    assert "universal_external_integration" in ids
    # Check structural_variants fields
    sv = chrom.structural_variants[0]
    assert sv["layout"] == "hierarchical"
    assert sv["reputation"] == 0.9
    # Test B* search scaffold
    path = gde.scaffold_b_star_search(circuit, "n1", "n3")
    assert path == ["n1", "n3"]
    # Test condition chain scaffold
    assert gde.scaffold_condition_chain(chrom) is True

def test_evolutionary_genetic_development():
    gde = GeneticDataExchange(organism_id="test")
    # Create two chromosomes
    elements1 = [GeneticElement(f"e{i}", GeneticElementType.EXON, "A"*10, i, 10, 1.0, ChromatinState.EUCHROMATIN) for i in range(5)]
    elements2 = [GeneticElement(f"e{i}", GeneticElementType.EXON, "C"*10, i, 10, 1.0, ChromatinState.EUCHROMATIN) for i in range(3)] + [GeneticElement(f"eX{i}", GeneticElementType.EXON, "G"*10, i+3, 10, 1.0, ChromatinState.EUCHROMATIN) for i in range(2)]
    chrom1 = GeneticChromosome("chrom1", elements1)
    chrom2 = GeneticChromosome("chrom2", elements2)
    # Cross-pollination
    child = gde.cross_pollinate(chrom1, chrom2)
    assert isinstance(child, GeneticChromosome)
    # Compatibility/diversity
    sv = child.structural_variants[0]
    assert 0.0 <= sv["compatibility"] <= 1.0
    assert 0.0 <= sv["diversity"] <= 1.0
    # Fitness
    fitness = gde.evaluate_fitness_multiobjective(child, {"obj1": 0.7, "obj2": 0.9})
    assert 0.0 <= fitness <= 1.0
    # Diversity in population
    pop = [chrom1, chrom2, child]
    diversity = gde.calculate_diversity(pop)
    assert 0.0 <= diversity <= 1.0
    # Lineage tracking
    lineage = gde.track_lineage(child)
    assert isinstance(lineage, list)
    # Rollback (stub)
    ancestor_id = "chrom1"
    rolled = gde.rollback_to_ancestor(child, ancestor_id)
    assert isinstance(rolled, GeneticChromosome)
    # Neighborhood analysis (stub)
    neighbors = gde.neighborhood_analysis(pop, child)
    assert isinstance(neighbors, list)
