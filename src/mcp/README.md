## Backend Features, Stubs, and Research Sources (2024-07)

### Key Features
- Modular, brain-inspired architecture (lobes/engines: memory, workflow, project, task, context, reminder, RAG, performance, pattern recognition, alignment, simulated reality)
- Centralized, dynamic config management via `config.cfg` and ProjectManager
- Unified workflow/task system with meta/partial task support, feedback, and dynamic Q&A
- Robust, extensible API endpoints for config, engines, knowledgebase, tasks, feedback, and reporting
- Security: API key authentication, per-IP rate limiting
- Monitoring: Prometheus/Netdata integration
- Fully local, portable, and secure design
- Self-improvement and research-driven iteration (AB test, split-brain, multi-LLM, etc.)

### Stubs & Planned Work
- Simulated reality engine (SimulatedRealityLobe)
- Advanced pattern recognition (PatternRecognitionLobe)
- Multi-LLM orchestration (experimental_lobes.py)
- Deep research, dreaming/simulation, scientific process, alignment engine
- Periodic reporting, dynamic tagging, bulk action safety, centralized config UI
- **Task Proposal Lobe (TaskProposalLobe):**
  - Proposes new tasks, tests, and evaluation steps based on project state, feedback, and research-driven heuristics.
  - Can be queried by other lobes or the main server for next actions.
  - Includes robust fallback logic and clear docstrings referencing idea.txt.
  - TODO: Expand with advanced heuristics, LLM-driven task generation, and integration with feedback loops.

### Research & References
- [WAIT/Hermes prompt queueing](https://arxiv.org/abs/2504.11320), [arXiv:2506.14851](https://arxiv.org/abs/2506.14851)
- [Absolute Zero self-learning model](https://medium.com/@santhosraj14/absolute-zero-the-future-of-ai-self-learning-without-human-data-and-uh-oh-moment-4562f337f508)
- [Clean Code Best Practices](https://hackernoon.com/how-to-write-clean-code-and-save-your-sanity)
- [Split-brain/AB test architectures](https://dev.to/topcat/create-react-ui-lib-component-library-speedrun-25bp)
- [Vite/React UI scaffolding](https://github.com/jblossomweb/vite-scaffold-2023)
- [Arch Linux ICU issues](https://bbs.archlinux32.org/viewtopic.php?id=3085)
- See `idea.txt` for the full vision and requirements

### Research: Advanced Engram Storage, Dynamic Coding Models, and Feedback-Driven Selection (2025)

- **ATLAS & DeepTransformers (2025):**
  - ATLAS introduces a high-capacity, deep neural memory module that learns to optimally memorize long contexts using the Omega rule and polynomial feature mappings, outperforming Transformers and RNNs on long-context and recall-intensive tasks. DeepTransformers generalize Transformer architectures with deep memory and locally optimal memory updates ([ATLAS, 2025](https://www.emergentmind.com/papers/2505.23735)).
  - Key innovations: sliding-window context optimization, polynomial kernel feature mapping, Muon optimizer for locally optimal memory updates, and chunk-wise parallelization for efficient training.
- **Neurocomputational Engram Models:**
  - Engram theory posits that sparse populations of neurons encode long-term memory via lasting physical and biochemical changes. Computational models leverage sparsity, synaptic plasticity, and sparse distributed memory to achieve efficient, interference-resistant representations ([Engram Memory Encoding, 2025](https://arxiv.org/abs/2506.01659)).
  - Techniques: sparse regularization, engram gating, spiking neural networks, and biologically inspired architectures.
- **Cognitive Weave (2025):**
  - Introduces a multi-layered spatio-temporal resonance graph (STRG) for memory, managing information as semantically rich "insight particles" dynamically enriched and interconnected. Cognitive refinement autonomously synthesizes higher-level knowledge aggregates, improving long-horizon planning and multi-session coherence ([Cognitive Weave, 2025](https://arxiv.org/abs/2506.08098)).
- **Hopfield and Local Learning Models:**
  - Dense Hopfield networks with temporal kernels enable exponential storage capacity and efficient sequential retrieval, supporting long-sequence memory and time-series modeling ([Hopfield, 2025](https://arxiv.org/abs/2507.01052)).
  - Objective-free local learning and emergent language structure models use hierarchical Hopfield memory chains for compositional memory and dynamic retokenization ([Objective-Free Local Learning, 2025](https://arxiv.org/abs/2506.23293)).
- **M+ and Latent-Space Memory:**
  - M+ extends MemoryLLM with scalable long-term memory and a co-trained retriever, dynamically retrieving relevant information during text generation and extending knowledge retention to over 160k tokens ([M+, 2025](https://arxiv.org/abs/2502.00592)).
- **Design Implications for MCP:**
  - Integrate deep, high-capacity memory modules (e.g., ATLAS, Hopfield, Cognitive Weave) for engram storage and retrieval.
  - Use sparse, plastic, and feedback-driven selection mechanisms for efficient, robust memory.
  - Support dynamic coding, diffusion models, and context-aware memory updates.
  - Modularize memory as a lobe/engine, with extensibility for future research and AB testing.
  - Cross-reference all design and implementation with idea.txt and latest research.

### Research: Advanced Engram Storage (2024-07)

- **Biological Principles:**
  - Engram neurons are recruited based on intrinsic excitability, synaptic plasticity, and neuromodulation. Memory consolidation is dynamic, with engram composition changing over time and selectivity increasing via inhibitory plasticity ([Nature 2023](https://www.nature.com/articles/s41380-023-02137-5), [bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.03.15.484429v1)).
- **Dynamic Coding Models:**
  - Use small, parallel neural networks (NNs) for encoding/decoding memory chunks, with synthetic selection algorithms trained on feedback to optimize storage and retrieval. Batch processing and mutagenic selection improve adaptability and compression.
- **Diffusion Models:**
  - Integrate diffusion-based generative models for engram merging, pattern completion, and robust recall. These models support flexible, context-aware memory retrieval and can be tuned for feedback-driven selection.
- **Feedback-Driven Selection:**
  - Implement feedback loops (from LLM/user/task outcomes) to dynamically select, reinforce, or prune engram representations. Use objective performance metrics and synthetic selection to avoid runaway effects and preserve original models for comparison.
- **Design Implications:**
  - Modularize engram storage/merging as a dedicated engine/lobe. Support pluggable backends for future extensibility. Document all research sources and update stubs as new methods are implemented.

### Research: Simulated Reality, Dreaming/Simulation, and Mind Map Engines (2024-07)

- **Simulated Reality Engines:**
  - State-of-the-art agentic systems now incorporate internal simulated realities, enabling agents to model, test, and reflect on hypothetical scenarios before acting. These systems use multi-agent architectures (e.g., Model Context Protocol, LangGraph, CrewAI) to orchestrate reasoning, planning, and self-correction in a virtual environment ([AI's State of the Art, 2025](https://medium.com/ai-simplified-in-plain-english/the-frontier-of-intelligence-ais-state-of-the-art-in-june-2025-f072dc909f6a)).
  - Simulated reality is used for "dreaming" (offline learning, scenario rehearsal, error recovery) and for mind map construction (associative, non-linear context linking).
- **Dreaming/Simulation:**
  - Inspired by biological dreaming, agentic systems use "dreaming" engines to simulate alternative outcomes, learn from failures, and generate creative solutions. These engines operate asynchronously, filter out dream content from persistent memory, and use feedback to improve reasoning and adaptability ([Project E.V.E.](https://community.openai.com/t/beyond-sci-fi-designing-ai-for-real-inner-life-why-tomorrow-s-intelligence-wont-just-think-it-will-feel-reflect-and-render-itself/1265039#post_4)).
  - Dreaming modules can leverage emotional reasoning, multi-perspective debate (e.g., alter-ego models), and symbolic self-rendering to enhance self-reflection and alignment.
- **Mind Map Engines:**
  - Modern mind map engines in agentic systems use graph-based memory and context association, supporting deep research, recall, and dynamic context export. These engines are integrated with memory, workflow, and alignment lobes for unified context management.
  - Mind maps are constructed and updated proactively, using pattern recognition, feedback, and associative learning to link related ideas, tasks, and engrams.
- **Design Implications:**
  - Modularize simulated reality, dreaming, and mind map engines as dedicated lobes. Support asynchronous operation, feedback-driven improvement, and integration with memory and workflow systems. Document all research sources and update stubs as new methods are implemented.

### Research: Multi-LLM Orchestration, AB Testing, and Split-Brain Architectures (2024-07)

- **Multi-LLM Orchestration:**
  - State-of-the-art agentic systems use hierarchical, modular multi-agent frameworks (e.g., OmniNova, AgentOrchestra, HASHIRU) to coordinate teams of specialized LLM agents for complex task decomposition, planning, and execution ([OmniNova](https://arxiv.org/html/2503.20028v1), [AgentOrchestra](https://arxiv.org/abs/2506.12508), [HASHIRU](https://arxiv.org/abs/2506.04255)).
  - Central orchestrators ("CEO" or "conductor" agents) dynamically allocate tasks, select models (local/remote, small/large), and manage resource constraints for optimal performance and cost.
  - Dynamic task routing, explicit sub-goal formulation, and inter-agent communication enable flexible, scalable, and efficient workflows.
- **AB Testing and Split-Brain Architectures:**
  - AB testing is implemented by running parallel agent teams ("left" and "right" lobes) with different strategies or model variants, comparing outcomes to select the best approach. This supports self-improvement, safety, and alignment.
  - Split-brain architectures modularize learning and adaptation, allowing independent evolution and cross-evaluation of agent subsystems. This mirrors biological split-brain research and supports robust, fail-safe operation.
- **Best Practices:**
  - Use hierarchical planning and supervision to decompose tasks and allocate them to specialized agents.
  - Prioritize local, resource-efficient models when possible, but flexibly use external APIs or larger models as needed.
  - Integrate autonomous tool creation, memory, and feedback-driven improvement for continual self-optimization.
  - Document all research sources and update stubs as new methods are implemented.

### Research: Multi-LLM Orchestration, AB Testing, and Split-Brain Architectures (2025)

- **Evolving Orchestration and Hierarchical Multi-Agent Frameworks:**
  - State-of-the-art agentic systems use dynamic, evolving orchestration paradigms (e.g., puppeteer-style orchestrators, hierarchical agent teams) to coordinate multiple LLMs and specialized agents for complex, adaptive workflows ([Multi-Agent Collaboration, 2025](https://arxiv.org/abs/2505.19591), [AgentOrchestra, 2025](https://arxiv.org/abs/2506.12508)).
  - Central orchestrators dynamically sequence, prioritize, and allocate tasks to agent teams, enabling flexible, scalable, and efficient collective reasoning. Hierarchical planning and explicit sub-goal formulation are key for generalization and adaptability.
- **Mixture-of-Agents and Agentic Neural Networks:**
  - Mixture-of-Agents (MoA) architectures leverage layered teams of LLM agents, where each agent takes outputs from previous layers as auxiliary information, achieving state-of-the-art performance on reasoning and generation benchmarks ([Mixture-of-Agents, 2025](https://openreview.net/forum?id=h0ZfDIrj7T)).
  - Agentic Neural Networks (ANN) conceptualize multi-agent collaboration as a layered neural network, with agents as nodes and teams as layers. ANN uses forward (task decomposition) and backward (feedback-driven refinement) phases, enabling self-evolving, neuro-symbolic, and data-driven multi-agent systems ([Agentic Neural Networks, 2025](https://arxiv.org/abs/2506.09046)).
- **Concurrent Reasoning and Cyclic AB Testing:**
  - Group Think and similar frameworks enable multiple concurrent reasoning agents to collaborate at token-level granularity, dynamically adapting reasoning trajectories and reducing redundancy ([Group Think, 2025](https://arxiv.org/abs/2505.11107)).
  - AB testing and split-brain architectures are implemented by running parallel agent teams ("left" and "right" lobes) with different strategies or model variants, comparing outcomes to select the best approach. Cyclic, feedback-driven evaluation supports self-improvement, safety, and alignment.
- **Design Implications for MCP:**
  - Modularize orchestration, AB testing, and split-brain as dedicated lobes/engines, supporting dynamic agent allocation, feedback-driven selection, and extensibility for future research.
  - Leverage evolving, hierarchical, and mixture-of-agents designs for robust, adaptive, and explainable agentic workflows.
  - Cross-reference all design and implementation with idea.txt and latest research.

### Assessment: Unfinished Implementation, Missing Features, and Technical Debt (2025)

#### **Current Stubs and Planned Work**
- **Experimental Lobes (src/mcp/experimental_lobes.py):**
  - `AlignmentEngine`: Basic preference alignment (stub), needs LLM-based alignment and AB testing
  - `PatternRecognitionEngine`: Neural column simulation (stub), needs batch NN processing and proactive prompting
  - `SimulatedReality`: Entity/event/state tracking (stub), needs integration with other lobes and feedback learning
  - `DreamingEngine`: Alternative scenario simulation (stub), needs dream filtering and learning mechanisms
  - `MindMapEngine`: Graph-based memory association (stub), needs dynamic context export and visualization
  - `ScientificProcessEngine`: Hypothesis testing (stub), needs evidence tracking and truth determination
  - `SplitBrainABTest`: Parallel agent teams (stub), needs comparison and selection mechanisms
  - `MultiLLMOrchestrator`: Task routing and aggregation (stub), needs actual LLM calls and feedback analytics
  - `AdvancedEngramEngine`: Dynamic coding models (stub), needs diffusion models and feedback-driven selection

#### **Information Flow and Modularity**
- **Project Structure:** 
  - Lobe registry in `lobes.py` provides modular architecture, but some lobes are stubbed
  - Information flow is represented through workflow steps and task dependencies
  - Need: Regular reorganization and refactoring per idea.txt priority order
- **Self-Improvement Mechanisms:**
  - Feedback loops implemented in workflow, task, and reminder systems
  - Self-improvement API (`_handle_self_improve`) exists but needs enhancement
  - Need: Autonomous reorganization, split-brain AB testing, and feedback-driven adaptation

#### **Technical Debt and Missing Features**
- **Frontend Integration:** 
  - Node.js ICU error blocks frontend development (documented in frontend/README.md)
  - Config editor, engine controls, and knowledgebase browser are stubbed
  - Need: Resolve ICU dependency or implement alternative UI approach
- **Testing and Documentation:**
  - TODO comments in `workflow.py` for expanding tests and documenting unified API
  - Missing comprehensive test coverage for all workflow/task operations
  - Need: Complete test suite and API documentation
- **Performance and Monitoring:**
  - Prometheus/Netdata integration stubbed in AB-test variations
  - Need: Implement real monitoring and performance optimization
- **Advanced Features:**
  - Vector backend abstraction (planned: Milvus, Annoy, Qdrant)
  - Advanced engram features (dynamic coding models, diffusion models)
  - Periodic reporting and dynamic tagging systems

#### **Alignment with idea.txt Requirements**
- **Implemented:** Modular brain-inspired architecture, centralized config, unified workflow/task system, feedback loops, self-improvement API
- **Stubbed:** Advanced engram features, simulated reality, dreaming/simulation, mind maps, multi-LLM orchestration, AB testing, split-brain architectures
- **Missing:** Autonomous reorganization, comprehensive testing, frontend UI, advanced monitoring, periodic reporting
- **Technical Debt:** Code quality, documentation, test coverage, performance optimization

#### **Next Development Priorities**
1. **Complete Core Features:** Implement stubbed experimental lobes with basic functionality
2. **Resolve Frontend Blocker:** Fix Node.js ICU issue or implement alternative UI
3. **Enhance Self-Improvement:** Implement autonomous reorganization and split-brain AB testing
4. **Improve Testing:** Complete test suite and API documentation
5. **Optimize Performance:** Implement real monitoring and vector backend options
6. **Regular Refactoring:** Follow idea.txt priority order for reorganization

---

## Stdio JSON-RPC Server for LLM/IDE Integration

The MCP server now supports a fully portable, LLM/IDE-friendly stdio JSON-RPC mode for optimal integration.

### Usage

```bash
python -m mcp.mcp_stdio
```

- Reads JSON-RPC requests from stdin
- Writes JSON-RPC responses to stdout
- All MCP endpoints are available (see `list_endpoints`)
- Designed for seamless use by LLMs, IDEs, and agentic tools

### Example JSON-RPC Request

```json
{
  "jsonrpc": "2.0",
  "method": "get_project_status",
  "params": {},
  "id": 1
}
```

### Example Response

```json
{
  "jsonrpc": "2.0",
  "result": { ... },
  "id": 1
}
```

### Features
- Fully local, portable, and secure
- Auto-prompting: MCP can proactively provide next steps and context
- Dynamic endpoint discovery (`list_endpoints`)
- Streaming support (planned)

See `idea.txt` for the vision and requirements this fulfills.

## LLM Inference Optimization

- MCP now includes a prompt queue and batch scheduler for LLM requests, inspired by WAIT/Hermes research ([arXiv:2504.11320](https://arxiv.org/abs/2504.11320), [arXiv:2506.14851](https://arxiv.org/abs/2506.14851)).
- Prompts are queued and processed in configurable batches for optimal throughput and latency.
- Queue status is available via the `get_prompt_queue_status` API endpoint.
- Hooks for speculative decoding and prewarming are stubbed for future work.

## Clean Code & Project Structure

- Refactored for modularity, maintainability, and clarity.
- See [Clean Code Best Practices](https://hackernoon.com/how-to-write-clean-code-and-save-your-sanity).

## Advanced Token Budgeting & Semantic Preservation

- MCP now optimizes all context exports for minimal token usage while preserving semantic meaning, using advanced summarization and truncation.
- This enables efficient LLM↔MCP interaction, especially for models with strict context limits.
- See [Advanced Guide: Optimizing LLMs with MCP](https://joelotepawembo.medium.com/advanced-guide-optimizing-large-language-models-with-model-context-protocol-mcp-performance-2020184dd605).

## Modular Vector Backend Architecture (Planned)

The MCP server is being refactored to support a modular `VectorBackend` interface, enabling pluggable vector storage and search backends:
- **SQLite/FAISS** (default, portable, local)
- **Milvus** (remote, scalable, high-performance)
- **Annoy** (lightweight, research/embedded)

### Planned Features
- Backend selection via config or CLI (`--vector-backend`)
- Drop-in replacement of vector backend without code changes
- Full compatibility with MCP tools and natural language commands (see [Milvus + MCP docs](https://milvus.io/docs/milvus_and_mcp.md))
- Optional dependency management for Milvus/Annoy
- Batch and ANN-ready APIs

### Example User Stories
- As a user, I want to run the MCP server on a USB drive with no dependencies, so I can use vector search anywhere (SQLite/FAISS backend).
- As a researcher, I want to benchmark Milvus vs. FAISS vs. Annoy for my workflow, so I can choose the best backend for my needs.
- As a team, we want to use a remote Milvus instance for shared, scalable vector search, but fall back to local FAISS if offline.
- As a developer, I want to add a new backend (e.g., Qdrant) by implementing the VectorBackend interface, so the system is future-proof.

### Next Steps
- Draft the VectorBackend interface and document in code and planning docs.
- Implement SQLite/FAISS backend as default.
- Add Milvus and Annoy backends as optional modules.
- Update CLI/config to support backend selection.
- Document pros/cons and usage for each backend.

## Security Best Practices

- Enterprise-grade security patterns are recommended, including threat modeling and tool poisoning mitigation.
- See [arXiv:2504.08623](https://arxiv.org/abs/2504.08623) for the latest research.

## Source Quality and Research Standards

All research, best practice, and external references in this project are held to the highest standards of credibility. Only peer-reviewed, academic, or authoritative sources are cited or referenced in code, documentation, and research workflows. All sources must pass the CRAAP test (Currency, Relevance, Authority, Accuracy, Purpose) as recommended by academic libraries ([NWACC Library](https://library.nwacc.edu/sourceevaluation/craap), [Merritt College Library](https://merritt.libguides.com/CRAAP_Test)).

- When adding research findings, always include a citation to a peer-reviewed journal, academic database, or official documentation (e.g., [Google Scholar](https://scholar.google.com/), [JSTOR](https://www.jstor.org/), [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)).
- Avoid referencing non-authoritative sources (e.g., social media, unverified blogs) except for illustrative or historical context.
- For security, architecture, and workflow best practices, cite standards from reputable organizations (e.g., IEEE, ACM, NIST, arXiv preprints with high citation counts).
- All sources must be evaluated using the CRAAP test. See [NWACC Library CRAAP Guide](https://library.nwacc.edu/sourceevaluation/craap) and [Merritt College CRAAP Test](https://merritt.libguides.com/CRAAP_Test) for details.

## Example: Research Reference in Code

When documenting a research-driven feature or best practice, use a comment like:

```python
# See: https://www.aresearchguide.com/find-sources-research-paper.html (Source evaluation best practices)
# See: https://arxiv.org/abs/2504.08623 (Security best practices)
# See: https://library.nwacc.edu/sourceevaluation/craap (CRAAP test for source evaluation)
```

## Iterative Design and Research

The MCP project is built on the principle of iterative design, as recommended by leading research handbooks and academic guides ([Zaposa Handbook](https://handbook.zaposa.com/articles/iterative-design/), [RMCAD LibGuides](https://rmcad.libguides.com/blogs/system/Research-is-an-iterative-process), [Dovetail](https://dovetail.com/product-development/iterative-design/), [ResearchOps Community](https://medium.com/researchops-community/breaking-the-double-diamond-with-iterative-discovery-7cd1c71c4f59)).

- **Iterative Cycles:** All research and development is conducted in cycles, with each iteration including rich data collection, deliberate documentation, reflection, and stakeholder feedback.
- **Emergence and Adaptation:** The project plan is flexible, allowing for new discoveries and directions to emerge at each stage.
- **Risk Mitigation:** Iterative design helps manage risks related to staffing, timing, and access, and supports both scaling up and deepening of research.
- **Reflection and Planning:** After each major milestone, the team documents lessons learned and adapts the plan for the next iteration.
- **Benefits:** Rapid issue resolution, adaptability, progress visibility, better product-market fit, and reduced documentation overhead.
- **Risks:** Over-iteration can lead to scope creep and delays; each iteration should have clear goals and review checkpoints.
- **Collaborative Output:** The MCP project emphasizes shared, continuously evolving outputs and cross-functional feedback, breaking down silos and fostering teamwork.

For more, see [Iterative Design – Zaposa Handbook](https://handbook.zaposa.com/articles/iterative-design/), [Research is an Iterative Process – RMCAD LibGuides](https://rmcad.libguides.com/blogs/system/Research-is-an-iterative-process), [Dovetail](https://dovetail.com/product-development/iterative-design/), and [ResearchOps Community](https://medium.com/researchops-community/breaking-the-double-diamond-with-iterative-discovery-7cd1c71c4f59).

## Alignment with idea.txt and Iterative Development Prompt

This project is developed in strict alignment with the vision and requirements in `idea.txt`. All features, stubs, and research-driven components are referenced to idea.txt, and the iterative development prompt is followed for all planning, implementation, and review cycles.

- All stubs and planned features are intentional and documented for future research and extensibility (see `idea.txt`).
- All research references are peer-reviewed or authoritative, and all sources are evaluated using the CRAAP test.
- Technical debt, missing features, and next steps are tracked and prioritized according to the iterative development prompt in `idea.txt`.
- The project is continuously refactored and improved based on research, feedback, and alignment with idea.txt.

For details on the iterative development process, see the end of `idea.txt` (ITTERATIVE_DEVELOPMENT_PROMPT.MD).

## Experimental Lobes and Research-Driven Development

All experimental lobes (pattern recognition, simulated reality, dreaming, mind map, scientific process, multi-LLM, advanced engram, etc.) are intentional stubs, documented in code and this README. They are designed for future research, AB testing, and extensibility, as required by idea.txt and the latest research (see AutoFlow: https://arxiv.org/abs/2407.12821, AFlow: https://arxiv.org/abs/2410.10762).

Each stub includes robust fallback logic (NotImplementedError with clear messages) and TODOs referencing the relevant research and idea.txt section. This approach ensures the codebase is robust, extensible, and ready for rapid research-driven iteration.

### Research References
- AutoFlow: Automated Workflow Generation for Large Language Model Agents ([arXiv:2407.12821](https://arxiv.org/abs/2407.12821))
- AFlow: Automating Agentic Workflow Generation ([arXiv:2410.10762](https://arxiv.org/abs/2410.10762))
- WAIT/Hermes prompt queueing ([arXiv:2504.11320](https://arxiv.org/abs/2504.11320))
- Group Think: Concurrent Reasoning and Cyclic AB Testing ([arXiv:2505.11107](https://arxiv.org/abs/2505.11107))
- Clean Code Best Practices (https://hackernoon.com/how-to-write-clean-code-and-save-your-sanity)

See code comments and docstrings for inline references and TODOs.

## Code/Documentation Search Functionality

The MCP server now supports unified code and documentation search, including both regex-based and semantic (RAG) search over code, markdown, and Python docstrings.

### Features
- **Docstring Extraction and Indexing:**
  - Extracts all Python docstrings in the project and indexes them for semantic search.
- **Unified Search:**
  - Search code, markdown, and docstrings using both regex and semantic (RAG) methods.
  - CLI and API support.

### Usage

#### Index all docstrings for semantic search
```bash
python -m mcp.cli index_docstrings --project-root .
```

#### Unified code/documentation search
```bash
python -m mcp.cli search_docs --query "your search term" --mode both --max-results 10
```
- `--mode` can be `regex`, `semantic`, or `both` (default: both)
- Results include file, line, context, and type (regex or semantic)

See the CLI help for more options.

---

## Advanced RL-Based Optimization and Self-Improvement (Planned)

The MCP server is being refactored to support RL-based optimization, failstate handling, and self-improvement, inspired by the latest research:
- Qwen2.5-Coder-7B-PPO: RL for code correctness and performance ([arXiv:2505.11480](https://arxiv.org/abs/2505.11480))
- ACECode: RL for efficiency and correctness ([arXiv:2412.17264](https://arxiv.org/abs/2412.17264))
- ACECODER: RL with automated test-case synthesis ([arXiv:2502.01718](https://arxiv.org/abs/2502.01718))
- CURE: Co-evolving coder and unit tester ([arXiv:2506.03136](https://arxiv.org/abs/2506.03136))
- ReCode: RL for dynamic API adaptation ([arXiv:2506.20495](https://arxiv.org/abs/2506.20495))

### Planned Features
- RL hooks for workflow/task optimization and failstate recovery
- Reward models for correctness, efficiency, and feedback
- Reranker modules for self-improvement and solution selection
- Automated test-case synthesis for robust failstate detection
- Dynamic knowledge and API adaptation

See TODOs and stubs in workflow.py and related modules for integration points.

### Additional Planned Features (2025)
- **PerfRL**: SLM-based RL optimization for efficient, portable code improvement ([arXiv:2312.05657](https://arxiv.org/abs/2312.05657))
- **CompilerDream**: Compiler world model for general code optimization and self-improvement ([arXiv:2404.16077](https://arxiv.org/abs/2404.16077))
- **Survey**: See [Enhancing Code LLMs with RL: A Survey](https://arxiv.org/abs/2412.20367) for a comprehensive review of RL-driven code optimization and failstate handling

All RL-based optimization and self-improvement features are planned to be modular, supporting both SLMs and LLMs, and leveraging test-based feedback and reward design for robust, adaptive agentic workflows.

### July 2024: Robust Fallbacks and Research-Driven Stubs

- All major stubs in core modules (workflow, experimental lobes, unified memory, vector memory, research integration) now include:
  - Robust fallback logic (logging, clear error/minimal result, no crashes)
  - Docstrings referencing idea.txt, TODO_DEVELOPMENT_PLAN.md, and relevant research
  - TODOs for future expansion and research-driven extensibility
- This approach ensures the codebase is robust, extensible, and ready for rapid research-driven iteration.
- See `idea.txt` and `TODO_DEVELOPMENT_PLAN.md` for the guiding vision and roadmap.
- For details, see code comments and docstrings in each module.

## Internal Neural Network Management Flow

The MCP system implements a robust, research-driven, and brain-inspired neural network (NN) management flow, designed for reliability, extensibility, and continuous self-improvement. This flow is aligned with the principles in `idea.txt` and best practices from leading research, including [Karpathy's Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).

### 1. Data-First Approach
- **Thorough Data Inspection:** Before any NN code is run, the system inspects and visualizes data distributions, outliers, and label quality. Scripts are provided for filtering, sorting, and analyzing data characteristics. All findings and assumptions are logged in working memory and experiment logs.

### 2. Experiment Tracking and Baselines
- **Experiment Registry:** Every training run, configuration, and result is logged in a robust, queryable format (e.g., SQLite, JSON, or the engram engine).
- **Baseline Models:** The system starts with a simple, trustworthy baseline (e.g., linear classifier or tiny ConvNet) to verify the pipeline. Random seeds are fixed for reproducibility. All unnecessary complexity (augmentation, regularization) is disabled at this stage.
- **Metric Visualization:** Loss, accuracy, and predictions are visualized and checked for sanity at every step.

### 3. Incremental Complexity and Hypothesis-Driven Development
- **One Change at a Time:** Complexity is added incrementally—only one new feature, layer, or trick at a time. Each change is accompanied by a hypothesis and validated with experiments.
- **No Unverified Complexity:** The system never introduces untested complexity. Each change is tested and documented.

### 4. Regularization and Generalization
- **Overfitting First:** Once the model can overfit the training set, regularization is introduced: more real data, data augmentation, dropout, weight decay, early stopping, etc.
- **Feature Visualization:** First-layer weights and activations are visualized to ensure meaningful learning.

### 5. Hyperparameter Tuning
- **Random Search Preferred:** Hyperparameters (learning rate, batch size, regularization, model size) are tuned using random search or Bayesian optimization. All settings and results are logged.

### 6. Feedback Integration and AB Testing
- **Feedback Loops:** The MCP's working memory and task system propose, track, and review all NN training experiments. User and LLM feedback is integrated into the training and evaluation process.
- **AB Testing and Split-Brain:** The system supports AB testing and split-brain architectures to compare different training strategies and select the best.

### 7. Self-Improvement and Continuous Refactoring
- **Periodic Review:** The training pipeline is periodically reviewed and refactored based on new research and project needs.
- **Automated Sanity Checks:** The system automates checks for common pitfalls (label leakage, data imbalance, overfitting).
- **Documentation and Knowledge Base:** All assumptions, failures, and learnings are documented in the project's knowledge base and experiment logs.

### 8. Finalization and Production Readiness
- **Ensembling and Distillation:** Ensembles are used for final accuracy boosts, and distilled into single models for efficiency if needed.
- **Long Training Runs:** Models are allowed to train for extended periods to maximize performance.
- **Comprehensive Testing:** All features, scripts, and methods are tested for robustness and alignment with `idea.txt`.

---

**References:**
- [A Recipe for Training Neural Networks – Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/)
- `idea.txt` and `TODO_DEVELOPMENT_PLAN.md`

This flow ensures that all neural network training and management in MCP is transparent, reproducible, and continuously improving, with every step logged, reviewed, and aligned with the project's research-driven vision. 