# MCP System Upgrade - Internal Rules and Tools

## Core Development Rules

### Code Quality Standards
- Line length: 88 characters (black formatter)
- Use black for code formatting
- Pass flake8 checks
- Use mypy with complete type annotations
- Use isort for import sorting
- Include docstrings for all public methods and classes
- Maximum 2000 lines per file
- Follow brain-inspired naming convention (e.g., `engram_engine.py`)

### Architecture Requirements
- Modular design: Each lobe/engine must be self-contained
- Plugin support: Extensible via `plugins/` directory
- Database separation: Each engine maintains its own SQLite database
- Memory efficiency: Vector compression for large data, <10GB target
- Local-only operation: No external API dependencies
- Cross-platform: Support Windows, Linux, and macOS

### Implementation Patterns
- Neural fallbacks: Implement both code and neural network solutions for critical functions
- Hormone system: Use hormone-inspired signaling for cross-lobe communication
- Genetic trigger system: Environmental adaptation through genetic-like mechanisms
- Three-tier memory: Implement hierarchical memory structure
- Split-brain testing: A/B testing via left/right lobe folders

## Hormone System Implementation

### Hormone Types & Functions
- Dopamine: Reward signaling (0.8-1.0 on task completion)
- Serotonin: Confidence and decision stability (0.5-0.9)
- Cortisol: Stress response and priority adjustment (0.7-0.95)
- Adrenaline: Urgency detection and acceleration
- Oxytocin: Collaboration and trust metrics (0.7-0.95)
- Growth hormone: Learning rate adaptation (0.6-0.9)
- GABA: Inhibitory control and noise reduction
- Vasopressin: Memory consolidation and learning enhancement (0.8-1.0)
- Thyroid hormones: Processing speed regulation
- Norepinephrine: Attention and focus enhancement
- Acetylcholine: Learning and neural plasticity
- Endorphins: System satisfaction and well-being metrics

### Diffusion Model
- Autocrine: 80% local effect on source lobe
- Paracrine: 40% effect on connected lobes
- Endocrine: 60% systemic circulation
- Decay rate: 5% per time step
- Circulation rate: 10% distribution per time step

## Genetic System Architecture

### Core Components
- GeneticTrigger: Environment-responsive activation system
- DNA signature: Encoded environmental conditions
- Codon map: 256-codon system for activation patterns
- Epigenetic markers: Environmental memory system
- Performance tracking: Comparative implementation metrics
- Hormone interface: Cross-system communication
- Mutation controller: Adaptive evolution system

### Required Methods
- encode_environment(): Convert environment to genetic signature
- should_activate(): Determine trigger activation
- calculate_environmental_similarity(): Environment matching algorithm
- build_codon_activation_map(): Extended 256-codon system
- apply_crossover(): Genetic crossover with configurable strategies
- mutate_sequence(): Controlled mutation with adaptive rates
- evaluate_fitness(): Multi-objective fitness evaluation

## Neural Network Integration

### Training and Optimization
- Store neural architectures in genetic sequences
- Implement controlled mutation with adaptive rates
- Support multiple crossover strategies
- Pass optimized weights through genetic lineage
- Implement neural architecture search (NAS)
- Support neuroevolution of augmenting topologies (NEAT)
- Use genetic programming for evolving neural network activation functions

### Performance Requirements
- Bandwidth: <5% of available network capacity
- Storage: <2% of available disk space
- Privacy compliance: 100% screening success rate
- Performance gains: >5% measurable improvement
- Offline operation: Full functionality without network
- Memory usage: <1GB RAM for genetic operations
- CPU utilization: <25% sustained load

## P2P Network Architecture

### Network Hierarchy Structure
- Chain of command: Top 12% of non-specialized clients lead network coordination
- Expert team: Top 15% of specialized agents provide domain expertise
- Dynamic adjustment: Roles shift based on real-time performance
- Peer review network: Scientific lobes collaborate across network for validation

### Genetic Data Exchange Protocol
- When: Temporal triggers and integration timing
- Where: Target lobe/engine for integration
- How: Integration methods and validation procedures
- Why: Purpose and expected performance benefits
- What: Data type, content summary, and metadata
- Order: Sequential integration dependencies

### Security & Privacy Standards
- Data sanitization pipeline with multiple stages
- Content-addressable storage with cryptographic hashing
- Lightweight consensus for data verification
- Encrypted P2P channels with key rotation
- Automatic auditing for privacy compliance

## Advanced Cognitive Systems

### Dreaming Engine
- Simulate scenarios for creative insights
- Generate dream-like processing without memory contamination
- Extract insights from simulated scenarios
- Filter dream contamination from memory systems
- Generate creative solutions to problems

### Scientific Process Engine
- Formulate hypotheses based on observations
- Design experiments to test hypotheses
- Execute experiments and collect results
- Analyze results and draw conclusions
- Validate findings against research
- Update knowledge base with validated findings

### Multi-LLM Orchestrator
- Distribute reasoning tasks across multiple LLMs
- Aggregate responses from different models
- Cross-validate outputs for accuracy
- Resolve conflicts between different outputs
- Optimize model selection based on task type

### Pattern Recognition Engine
- Create neural columns for different input types
- Process sensory input through neural columns
- Learn pattern associations from examples
- Predict pattern completions from partial inputs
- Adapt column sensitivity based on feedback

## Error Handling & Resilience

### Strategies
- Circuit breaker pattern for failing implementations
- Fallback logging for implementation switches
- Resource monitoring for memory management
- Comprehensive validation testing
- Graceful degradation for partial failures
- Self-healing for recovery from corruption
- Checkpointing for rollback capability
- Anomaly detection for problematic sequences

## Monitoring & Visualization

### Components
- Real-time dashboard for hormone levels and brain state
- Historical analysis tools for trend identification
- Cascade visualization for hormone interactions
- Performance comparison between implementations
- Anomaly highlighting for unusual patterns
- Lobe detail views for individual components
- Time-series visualizations for system metrics

## Tools and Utilities

### Development Tools
- Black: Code formatting
- Flake8: Linting
- Mypy: Type checking
- Isort: Import sorting
- Pytest: Testing framework
- PFSUS Standards Enforcer: Automated PFSUS compliance validation and fixing

### Runtime Tools
- Performance tracker: Monitor and compare implementations
- Hormone diffusion engine: Manage hormone distribution
- Genetic trigger system: Handle environmental adaptation
- Brain state aggregator: Monitor system-wide state
- Resource optimization engine: Manage resource allocation
- Security system: Detect and respond to threats
- Monitoring system: Visualize system behavior