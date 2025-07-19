# Split-Brain A/B Testing

Split-brain A/B testing enables parallel evaluation of left and right lobe genetic trigger implementations. This supports evolutionary improvement and robust self-optimization in the MCP system.

## Architecture
- Left and right lobe folders: `left_lobes/`, `right_lobes/`
- Each lobe can evolve independently (e.g., different mutation strategies)
- Test runner instantiates both on the same input and compares outcomes

## Usage
- See `test_split_brain_ab.py` for example test
- Both lobes are exercised and compared for activation and mutation
- Results are logged for performance tracking

## Benefits
- Enables evolutionary selection of superior genetic strategies
- Supports robust, fault-tolerant adaptation

## Performance Benchmarking

The split-brain A/B test runner collects and compares performance metrics for both left and right lobes:
- **Activation count**: Number of times each lobe activates on the same input
- **Average mutation rate**: Mean mutation rate after mutation events
- **Average fitness**: Mean fitness score after mutation events

### Example Results
```
Left activations: 100/100
Right activations: 100/100
Left avg mutation rate: 0.0501
Right avg mutation rate: 0.0499
Left avg fitness: 0.5000
Right avg fitness: 0.5000
```

These metrics enable direct comparison and evolutionary selection of superior genetic strategies.

## Related Docs
- [[genetic_trigger_system]]
- [[genetic_codon_encoding]]
- [[hormone_system]]
- [[epigenetic_memory]]
- [[genetic_performance_benchmarking]] 