# Hormone System

The hormone system in MCP enables cross-lobe communication, feedback, and adaptive regulation of genetic triggers and memory.

## Integration
- GeneticTrigger triggers hormone cascades on activation
- Receives hormone feedback (e.g., cortisol) to adapt mutation rates
- Stores hormone feedback as epigenetic markers

## Feedback & Adaptation
- High cortisol increases mutation rate (stress adaptation)
- Hormone levels influence genetic and memory system behavior

## Usage
- See `HormoneSystemInterface.release_genetic_hormones`
- Feedback is passed to `GeneticTrigger.receive_memory_feedback`

## Related Docs
- [[genetic_trigger_system]]
- [[epigenetic_memory]]
- [[split_brain_testing]] 