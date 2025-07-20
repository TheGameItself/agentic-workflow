# Genetic Trigger System

The Genetic Trigger System enables adaptive, biologically-inspired optimization in the MCP architecture. It encodes environmental and pathway data using a 256-codon (4-letter) DNA-inspired scheme, supports dual code/neural implementations, and integrates with hormone and epigenetic memory systems.

## Architecture
- Dual implementation: code and neural network alternatives
- Hormone system integration: triggers cascades, receives feedback
- Epigenetic memory: stores markers for adaptation and feedback
- Split-brain A/B testing: left/right lobe comparison for evolutionary improvement

## Codon Encoding
- Uses 256 unique 4-letter codons (A/T/G/C) for bijective byte mapping
- Enables robust, lossless encoding/decoding of genetic data
- See [[genetic_codon_encoding.md]] for details

## Hormone & Epigenetic Integration
- Triggers hormone cascades on activation
- Stores/retrieves epigenetic markers for environmental and hormone feedback
- Adapts mutation rate based on hormone levels (e.g., cortisol)
- See [[hormone_system.md]] and [[epigenetic_memory.md]]

## Split-Brain A/B Testing
- Implements left/right lobe folder structure
- Runs parallel genetic trigger variants for performance comparison
- See [[split_brain_testing.md]] for usage and results

## Related Docs
- [[split_brain_testing.md]]
- [[genetic_codon_encoding.md]]
- [[hormone_system.md]]
- [[epigenetic_memory.md]]
- [[memory_system.md]]
- [[neural-network-models/README|Neural Network Models]] - Self-improving neural networks 