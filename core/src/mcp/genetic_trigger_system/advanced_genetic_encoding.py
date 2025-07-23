"""
Advanced Genetic Sequence Encoding for Prompt Circuits

This module implements sophisticated genetic sequence encoding for prompt circuits,
including DNA-inspired encoding, expression quality metrics, and dynamic optimization.

Features:
- DNA-inspired encoding for prompt circuit structures and execution paths
- Expression quality metrics to evaluate genetic sequence effectiveness
- Dynamic self-improvement mechanisms through recursive optimization
- Cross-reference indexing between related genetic elements
- Walkback mechanisms for error correction and optimization
- Context-aware genetic exploration for project-specific adaptation
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from .environmental_state import EnvironmentalState


class ExpressionQuality(Enum):
    """Quality levels for genetic expression"""
    EXCELLENT = "excellent"    # High performance, optimal expression
    GOOD = "good"             # Good performance, minor optimizations possible
    AVERAGE = "average"       # Average performance, significant room for improvement
    POOR = "poor"            # Poor performance, major optimization needed
    FAILED = "failed"        # Failed expression, requires redesign


class GeneticElementType(Enum):
    """Types of genetic elements in the sequence"""
    PROMOTER = "promoter"           # Promoter region for expression initiation
    ENHANCER = "enhancer"           # Enhancer region for expression amplification
    CODING_SEQUENCE = "coding"      # Main coding sequence
    REGULATORY = "regulatory"       # Regulatory elements
    TERMINATOR = "terminator"       # Terminator region
    SPLICING = "splicing"           # Splicing signals
    SILENCER = "silencer"           # Silencer elements


@dataclass
class GeneticElement:
    """Represents a genetic element in the sequence"""
    element_id: str
    element_type: GeneticElementType
    sequence: str
    position: int
    strength: float = 1.0
    context_dependency: Dict[str, float] = field(default_factory=dict)
    cross_references: List[str] = field(default_factory=list)
    expression_history: List[float] = field(default_factory=list)
    last_optimized: Optional[datetime] = None
    
    def add_expression_result(self, quality: float):
        """Add expression quality result to history"""
        self.expression_history.append(quality)
        if len(self.expression_history) > 100:
            self.expression_history = self.expression_history[-100:]
    
    def get_average_expression(self) -> float:
        """Get average expression quality"""
        if not self.expression_history:
            return 0.5
        return sum(self.expression_history) / len(self.expression_history)
    
    def needs_optimization(self, threshold: float = 0.6) -> bool:
        """Check if element needs optimization"""
        return self.get_average_expression() < threshold


@dataclass
class PromptCircuit:
    """Represents a prompt circuit with genetic encoding"""
    circuit_id: str
    genetic_sequence: str
    elements: List[GeneticElement] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def add_element(self, element: GeneticElement):
        """Add a genetic element to the circuit"""
        self.elements.append(element)
        self.elements.sort(key=lambda e: e.position)
    
    def get_expression_quality(self) -> ExpressionQuality:
        """Get overall expression quality"""
        if not self.performance_metrics:
            return ExpressionQuality.AVERAGE
        
        # Calculate weighted quality score
        weights = {
            'accuracy': 0.4,
            'efficiency': 0.3,
            'adaptability': 0.2,
            'stability': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.performance_metrics:
                total_score += self.performance_metrics[metric] * weight
                total_weight += weight
        
        if total_weight == 0:
            return ExpressionQuality.AVERAGE
        
        quality_score = total_score / total_weight
        
        if quality_score >= 0.9:
            return ExpressionQuality.EXCELLENT
        elif quality_score >= 0.7:
            return ExpressionQuality.GOOD
        elif quality_score >= 0.5:
            return ExpressionQuality.AVERAGE
        elif quality_score >= 0.3:
            return ExpressionQuality.POOR
        else:
            return ExpressionQuality.FAILED
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        self.last_used = datetime.now()
        self.usage_count += 1
    
    def add_cross_reference(self, element_id: str, reference_id: str):
        """Add cross-reference between elements"""
        if element_id not in self.cross_references:
            self.cross_references[element_id] = []
        if reference_id not in self.cross_references[element_id]:
            self.cross_references[element_id].append(reference_id)


class AdvancedGeneticEncoder:
    """
    Advanced genetic encoder for prompt circuits.
    
    Features:
    - DNA-inspired encoding with promoter, enhancer, coding, and regulatory regions
    - Context-aware sequence generation
    - Cross-reference indexing
    - Expression quality tracking
    - Dynamic optimization
    """
    
    def __init__(self, sequence_length: int = 1000, mutation_rate: float = 0.05):
        self.sequence_length = sequence_length
        self.mutation_rate = mutation_rate
        self.logger = logging.getLogger("AdvancedGeneticEncoder")
        
        # Genetic alphabet and codon tables
        self.bases = ['A', 'T', 'G', 'C']
        self.codons = [''.join(codon) for codon in itertools.product(self.bases, repeat=3)]
        self.codon_to_byte = {codon: i for i, codon in enumerate(self.codons)}
        
        # Expression quality tracking
        self.expression_quality_history: List[Dict[str, Any]] = []
        self.optimization_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Cross-reference database
        self.cross_reference_db: Dict[str, Set[str]] = defaultdict(set)
        
        self.logger.info("Advanced Genetic Encoder initialized")
    
    def encode_prompt_circuit(self, prompt_data: Dict[str, Any], 
                            context: EnvironmentalState) -> PromptCircuit:
        """
        Encode prompt data into a genetic circuit.
        
        Args:
            prompt_data: Prompt structure and content
            context: Environmental context for encoding
            
        Returns:
            Encoded prompt circuit
        """
        circuit_id = f"circuit_{uuid.uuid4().hex[:8]}"
        
        # Generate genetic sequence
        genetic_sequence = self._generate_genetic_sequence(prompt_data, context)
        
        # Create circuit
        circuit = PromptCircuit(
            circuit_id=circuit_id,
            genetic_sequence=genetic_sequence
        )
        
        # Add genetic elements
        elements = self._create_genetic_elements(prompt_data, context)
        for element in elements:
            circuit.add_element(element)
        
        # Set execution path
        circuit.execution_path = self._determine_execution_path(elements, context)
        
        # Set context requirements
        circuit.context_requirements = self._extract_context_requirements(context)
        
        self.logger.info(f"Encoded prompt circuit {circuit_id} with {len(elements)} elements")
        
        return circuit
    
    def _generate_genetic_sequence(self, prompt_data: Dict[str, Any], 
                                 context: EnvironmentalState) -> str:
        """Generate genetic sequence from prompt data and context"""
        # Create a structured genetic sequence
        sequence_parts = []
        
        # Promoter region (context-dependent)
        promoter = self._generate_promoter_region(context)
        sequence_parts.append(promoter)
        
        # Enhancer region (performance-dependent)
        enhancer = self._generate_enhancer_region(context)
        sequence_parts.append(enhancer)
        
        # Coding sequence (prompt content)
        coding = self._generate_coding_sequence(prompt_data)
        sequence_parts.append(coding)
        
        # Regulatory elements (context-specific)
        regulatory = self._generate_regulatory_elements(context)
        sequence_parts.append(regulatory)
        
        # Terminator region
        terminator = self._generate_terminator_region()
        sequence_parts.append(terminator)
        
        # Combine and ensure proper length
        full_sequence = "".join(sequence_parts)
        
        # Pad or truncate to target length
        if len(full_sequence) < self.sequence_length:
            full_sequence += self._generate_random_sequence(self.sequence_length - len(full_sequence))
        else:
            full_sequence = full_sequence[:self.sequence_length]
        
        return full_sequence
    
    def _generate_promoter_region(self, context: EnvironmentalState) -> str:
        """Generate promoter region based on context"""
        # Context-dependent promoter generation
        complexity_factor = context.task_complexity
        pressure_factor = context.adaptation_pressure
        
        # Base promoter sequence
        promoter = "TATAAA"  # TATA box
        
        # Add context-specific elements
        if complexity_factor > 0.7:
            promoter += "GCGCGC"  # High complexity promoter
        elif complexity_factor < 0.3:
            promoter += "ATATAT"  # Low complexity promoter
        
        if pressure_factor > 0.7:
            promoter += "CCCCCC"  # High pressure promoter
        elif pressure_factor < 0.3:
            promoter += "GGGGGG"  # Low pressure promoter
        
        return promoter
    
    def _generate_enhancer_region(self, context: EnvironmentalState) -> str:
        """Generate enhancer region based on performance context"""
        performance = context.get_overall_performance()
        
        # Performance-dependent enhancer
        if performance > 0.8:
            enhancer = "GCCGCC" * 3  # High performance enhancer
        elif performance > 0.6:
            enhancer = "ATCGAT" * 2  # Medium performance enhancer
        else:
            enhancer = "TAGCTA" * 1  # Low performance enhancer
        
        return enhancer
    
    def _generate_coding_sequence(self, prompt_data: Dict[str, Any]) -> str:
        """Generate coding sequence from prompt data"""
        # Encode prompt structure and content
        coding_parts = []
        
        # Encode prompt type
        prompt_type = prompt_data.get('type', 'general')
        type_encoding = self._encode_string(prompt_type)
        coding_parts.append(type_encoding)
        
        # Encode prompt complexity
        complexity = prompt_data.get('complexity', 0.5)
        complexity_encoding = self._encode_float(complexity)
        coding_parts.append(complexity_encoding)
        
        # Encode prompt content hash
        content = prompt_data.get('content', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()[:20]
        content_encoding = self._encode_string(content_hash)
        coding_parts.append(content_encoding)
        
        return "".join(coding_parts)
    
    def _generate_regulatory_elements(self, context: EnvironmentalState) -> str:
        """Generate regulatory elements based on context"""
        regulatory_parts = []
        
        # System load regulation
        load = context.get_overall_load()
        if load > 0.8:
            regulatory_parts.append("SILENCER_HIGH_LOAD")
        elif load < 0.2:
            regulatory_parts.append("ENHANCER_LOW_LOAD")
        
        # Hormone level regulation
        hormone_level = context.get_overall_hormone_level()
        if hormone_level > 0.7:
            regulatory_parts.append("ENHANCER_HIGH_HORMONE")
        elif hormone_level < 0.3:
            regulatory_parts.append("SILENCER_LOW_HORMONE")
        
        # Encode regulatory elements
        return self._encode_string("_".join(regulatory_parts))
    
    def _generate_terminator_region(self) -> str:
        """Generate terminator region"""
        return "AATAAA"  # Standard terminator
    
    def _generate_random_sequence(self, length: int) -> str:
        """Generate random genetic sequence"""
        return ''.join(random.choices(self.bases, k=length))
    
    def _encode_string(self, text: str) -> str:
        """Encode string to genetic sequence"""
        # Simple encoding: convert to bytes, then to genetic sequence
        text_bytes = text.encode('utf-8')
        genetic_sequence = ""
        
        for byte in text_bytes:
            # Map byte to codon
            codon_index = byte % len(self.codons)
            genetic_sequence += self.codons[codon_index]
        
        return genetic_sequence
    
    def _encode_float(self, value: float) -> str:
        """Encode float value to genetic sequence"""
        # Convert float to bytes, then encode
        import struct
        float_bytes = struct.pack('f', value)
        return self._encode_string(float_bytes.hex())
    
    def _create_genetic_elements(self, prompt_data: Dict[str, Any], 
                               context: EnvironmentalState) -> List[GeneticElement]:
        """Create genetic elements for the circuit"""
        elements = []
        position = 0
        
        # Promoter element
        promoter = GeneticElement(
            element_id=f"promoter_{uuid.uuid4().hex[:8]}",
            element_type=GeneticElementType.PROMOTER,
            sequence=self._generate_promoter_region(context),
            position=position,
            strength=1.0 + context.task_complexity * 0.5
        )
        elements.append(promoter)
        position += 100
        
        # Enhancer element
        enhancer = GeneticElement(
            element_id=f"enhancer_{uuid.uuid4().hex[:8]}",
            element_type=GeneticElementType.ENHANCER,
            sequence=self._generate_enhancer_region(context),
            position=position,
            strength=context.get_overall_performance()
        )
        elements.append(enhancer)
        position += 100
        
        # Coding sequence element
        coding = GeneticElement(
            element_id=f"coding_{uuid.uuid4().hex[:8]}",
            element_type=GeneticElementType.CODING_SEQUENCE,
            sequence=self._generate_coding_sequence(prompt_data),
            position=position,
            strength=0.8
        )
        elements.append(coding)
        position += 200
        
        # Regulatory elements
        regulatory = GeneticElement(
            element_id=f"regulatory_{uuid.uuid4().hex[:8]}",
            element_type=GeneticElementType.REGULATORY,
            sequence=self._generate_regulatory_elements(context),
            position=position,
            strength=0.6
        )
        elements.append(regulatory)
        position += 100
        
        # Terminator element
        terminator = GeneticElement(
            element_id=f"terminator_{uuid.uuid4().hex[:8]}",
            element_type=GeneticElementType.TERMINATOR,
            sequence=self._generate_terminator_region(),
            position=position,
            strength=1.0
        )
        elements.append(terminator)
        
        return elements
    
    def _determine_execution_path(self, elements: List[GeneticElement], 
                                context: EnvironmentalState) -> List[str]:
        """Determine execution path through genetic elements"""
        path = []
        
        # Always start with promoter
        path.append("promoter")
        
        # Add enhancer if performance is good
        if context.get_overall_performance() > 0.6:
            path.append("enhancer")
        
        # Add coding sequence
        path.append("coding")
        
        # Add regulatory elements based on context
        if context.is_high_stress():
            path.append("regulatory_silencer")
        elif context.is_optimal():
            path.append("regulatory_enhancer")
        
        # Always end with terminator
        path.append("terminator")
        
        return path
    
    def _extract_context_requirements(self, context: EnvironmentalState) -> Dict[str, Any]:
        """Extract context requirements from environmental state"""
        return {
            'min_performance': 0.5,
            'max_load': 0.8,
            'min_hormone_level': 0.3,
            'complexity_range': (0.3, 0.8),
            'pressure_threshold': 0.7
        }
    
    async def optimize_circuit(self, circuit: PromptCircuit, 
                             performance_feedback: Dict[str, float]) -> PromptCircuit:
        """
        Optimize circuit based on performance feedback.
        
        Args:
            circuit: Circuit to optimize
            performance_feedback: Performance metrics and feedback
            
        Returns:
            Optimized circuit
        """
        self.logger.info(f"Optimizing circuit {circuit.circuit_id}")
        
        # Update circuit performance
        circuit.update_performance(performance_feedback)
        
        # Identify elements needing optimization
        elements_to_optimize = [
            element for element in circuit.elements 
            if element.needs_optimization()
        ]
        
        if not elements_to_optimize:
            self.logger.info("No elements need optimization")
            return circuit
        
        # Create optimized circuit
        optimized_circuit = PromptCircuit(
            circuit_id=f"{circuit.circuit_id}_optimized_{uuid.uuid4().hex[:4]}",
            genetic_sequence=circuit.genetic_sequence,
            context_requirements=circuit.context_requirements.copy(),
            cross_references=circuit.cross_references.copy()
        )
        
        # Optimize each element
        for element in circuit.elements:
            if element in elements_to_optimize:
                optimized_element = await self._optimize_element(element, performance_feedback)
                optimized_circuit.add_element(optimized_element)
            else:
                optimized_circuit.add_element(element)
        
        # Update execution path
        optimized_circuit.execution_path = self._determine_execution_path(
            optimized_circuit.elements, 
            EnvironmentalState(timestamp=datetime.now().isoformat())
        )
        
        # Record optimization
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'original_circuit_id': circuit.circuit_id,
            'optimized_circuit_id': optimized_circuit.circuit_id,
            'elements_optimized': len(elements_to_optimize),
            'performance_improvement': self._calculate_improvement(circuit, optimized_circuit)
        }
        optimized_circuit.optimization_history.append(optimization_record)
        
        self.logger.info(f"Circuit optimization complete: {len(elements_to_optimize)} elements optimized")
        
        return optimized_circuit
    
    async def _optimize_element(self, element: GeneticElement, 
                              performance_feedback: Dict[str, float]) -> GeneticElement:
        """Optimize a genetic element"""
        # Create optimized element
        optimized_element = GeneticElement(
            element_id=f"{element.element_id}_opt_{uuid.uuid4().hex[:4]}",
            element_type=element.element_type,
            sequence=element.sequence,
            position=element.position,
            strength=element.strength,
            context_dependency=element.context_dependency.copy(),
            cross_references=element.cross_references.copy(),
            expression_history=element.expression_history.copy()
        )
        
        # Apply optimization based on element type
        if element.element_type == GeneticElementType.PROMOTER:
            optimized_element = await self._optimize_promoter(optimized_element, performance_feedback)
        elif element.element_type == GeneticElementType.ENHANCER:
            optimized_element = await self._optimize_enhancer(optimized_element, performance_feedback)
        elif element.element_type == GeneticElementType.CODING_SEQUENCE:
            optimized_element = await self._optimize_coding(optimized_element, performance_feedback)
        elif element.element_type == GeneticElementType.REGULATORY:
            optimized_element = await self._optimize_regulatory(optimized_element, performance_feedback)
        
        optimized_element.last_optimized = datetime.now()
        
        return optimized_element
    
    async def _optimize_promoter(self, element: GeneticElement, 
                               performance_feedback: Dict[str, float]) -> GeneticElement:
        """Optimize promoter element"""
        # Adjust promoter strength based on performance
        accuracy = performance_feedback.get('accuracy', 0.5)
        efficiency = performance_feedback.get('efficiency', 0.5)
        
        # Increase strength if performance is good
        if accuracy > 0.7 and efficiency > 0.7:
            element.strength = min(2.0, element.strength * 1.2)
        elif accuracy < 0.5 or efficiency < 0.5:
            element.strength = max(0.1, element.strength * 0.8)
        
        return element
    
    async def _optimize_enhancer(self, element: GeneticElement, 
                               performance_feedback: Dict[str, float]) -> GeneticElement:
        """Optimize enhancer element"""
        # Adjust enhancer based on performance
        performance = performance_feedback.get('accuracy', 0.5)
        
        if performance > 0.8:
            # High performance: strengthen enhancer
            element.strength = min(2.0, element.strength * 1.3)
        elif performance < 0.4:
            # Low performance: weaken enhancer
            element.strength = max(0.1, element.strength * 0.7)
        
        return element
    
    async def _optimize_coding(self, element: GeneticElement, 
                             performance_feedback: Dict[str, float]) -> GeneticElement:
        """Optimize coding sequence element"""
        # For coding sequences, we might need to modify the sequence itself
        # This is a simplified optimization
        accuracy = performance_feedback.get('accuracy', 0.5)
        
        if accuracy < 0.5:
            # Low accuracy: try to improve sequence
            element.strength = max(0.1, element.strength * 0.9)
        
        return element
    
    async def _optimize_regulatory(self, element: GeneticElement, 
                                 performance_feedback: Dict[str, float]) -> GeneticElement:
        """Optimize regulatory element"""
        # Adjust regulatory strength based on context
        adaptability = performance_feedback.get('adaptability', 0.5)
        
        if adaptability > 0.7:
            element.strength = min(2.0, element.strength * 1.1)
        elif adaptability < 0.3:
            element.strength = max(0.1, element.strength * 0.9)
        
        return element
    
    def _calculate_improvement(self, original: PromptCircuit, 
                             optimized: PromptCircuit) -> float:
        """Calculate performance improvement between circuits"""
        if not original.performance_metrics or not optimized.performance_metrics:
            return 0.0
        
        # Calculate improvement for each metric
        improvements = []
        for metric in ['accuracy', 'efficiency', 'adaptability', 'stability']:
            if metric in original.performance_metrics and metric in optimized.performance_metrics:
                original_val = original.performance_metrics[metric]
                optimized_val = optimized.performance_metrics[metric]
                if original_val > 0:
                    improvement = (optimized_val - original_val) / original_val
                    improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    async def walkback_optimization(self, circuit: PromptCircuit, 
                                  target_performance: Dict[str, float]) -> PromptCircuit:
        """
        Walkback optimization: revert to previous successful configurations.
        
        Args:
            circuit: Circuit to walkback
            target_performance: Target performance metrics
            
        Returns:
            Walkback-optimized circuit
        """
        self.logger.info(f"Performing walkback optimization on circuit {circuit.circuit_id}")
        
        # Find best historical performance
        best_performance = None
        best_configuration = None
        
        for record in circuit.optimization_history:
            if 'performance_improvement' in record:
                improvement = record['performance_improvement']
                if best_performance is None or improvement > best_performance:
                    best_performance = improvement
                    best_configuration = record
        
        if best_configuration:
            # Revert to best configuration
            walkback_circuit = PromptCircuit(
                circuit_id=f"{circuit.circuit_id}_walkback_{uuid.uuid4().hex[:4]}",
                genetic_sequence=circuit.genetic_sequence,
                context_requirements=circuit.context_requirements.copy()
            )
            
            # Restore best element configurations
            for element in circuit.elements:
                restored_element = GeneticElement(
                    element_id=element.element_id,
                    element_type=element.element_type,
                    sequence=element.sequence,
                    position=element.position,
                    strength=element.strength * 1.1,  # Slight boost
                    context_dependency=element.context_dependency.copy(),
                    cross_references=element.cross_references.copy()
                )
                walkback_circuit.add_element(restored_element)
            
            self.logger.info(f"Walkback optimization complete, restored best configuration")
            return walkback_circuit
        
        return circuit
    
    def add_cross_reference(self, element_id: str, reference_id: str):
        """Add cross-reference between genetic elements"""
        self.cross_reference_db[element_id].add(reference_id)
        self.cross_reference_db[reference_id].add(element_id)
    
    def get_related_elements(self, element_id: str) -> Set[str]:
        """Get related genetic elements"""
        return self.cross_reference_db.get(element_id, set())
    
    def get_expression_quality_statistics(self) -> Dict[str, Any]:
        """Get expression quality statistics"""
        if not self.expression_quality_history:
            return {}
        
        qualities = [record['quality'] for record in self.expression_quality_history]
        
        return {
            'total_expressions': len(qualities),
            'average_quality': sum(qualities) / len(qualities),
            'excellent_count': sum(1 for q in qualities if q >= 0.9),
            'good_count': sum(1 for q in qualities if 0.7 <= q < 0.9),
            'average_count': sum(1 for q in qualities if 0.5 <= q < 0.7),
            'poor_count': sum(1 for q in qualities if 0.3 <= q < 0.5),
            'failed_count': sum(1 for q in qualities if q < 0.3)
        }


# Import itertools for codon generation
import itertools 