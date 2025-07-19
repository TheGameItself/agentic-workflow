# Epigenetic Memory System

Epigenetic memory in the MCP system stores markers for environmental and hormone feedback, enabling adaptive, context-sensitive genetic behavior.

## Purpose
- Store markers for activation, hormone feedback, and adaptation
- Enable context-aware mutation and trigger selection

## Integration
- Used by GeneticTrigger to store/retrieve encoded markers
- Integrates with hormone system for feedback-driven adaptation
- Uses 256-codon encoding for robust storage

## Usage
- See `EpigeneticMemory.set_marker` and `get_marker`
- Markers are set on activation and hormone feedback events

## Related Docs
- [[genetic_trigger_system]]
- [[genetic_codon_encoding]]
- [[hormone_system]]
- [[split_brain_testing]] 