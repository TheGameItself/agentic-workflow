# MCP Server Architecture Documentation

## Overview

The MCP Server is designed as a modular, extensible system that provides intelligent development assistance through multiple cognitive engines and advanced memory management. The architecture is inspired by human brain functions and implements a "split-brain" approach for continuous self-improvement.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Client    │  │   IDE       │  │   Web UI    │         │
│  │  Interface  │  │ Integration │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    API Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   REST API  │  │  MCP Stdio  │  │  WebSocket  │         │
│  │             │  │   Server    │  │   Server    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Core Services                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Workflow   │  │    Task     │  │   Memory    │         │
│  │  Manager    │  │  Manager    │  │  Manager    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                  Experimental Lobes                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Alignment  │  │   Pattern   │  │  Simulated  │         │
│  │   Engine    │  │ Recognition │  │   Reality   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Dreaming   │  │   Mind Map  │  │ Scientific  │         │
│  │   Engine    │  │   Engine    │  │  Process    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Split     │  │   Multi-    │  │  Advanced   │         │
│  │   Brain     │  │    LLM      │  │   Engram    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   SQLite    │  │   Vector    │  │   File      │         │
│  │  Database   │  │  Database   │  │   System    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

For the full, up-to-date project file tree, see [`docs/FILETREE.txt`](FILETREE.txt).

## Core Components

### 1. API Layer

The API layer provides multiple interfaces for client interaction:

#### REST API
- **Purpose**: Standard HTTP-based API for web clients and external integrations
- **Features**: JSON-RPC 2.0 compliant, authentication, rate limiting
- **Endpoints**: `/api/v1/` for all operations

#### MCP Studio Server
- **Purpose**: Standard MCP protocol implementation for IDE integration
- **Features**: JSON-RPC over stdio, bidirectional communication
- **Protocol**: Model Context Protocol specification

#### WebSocket Server
- **Purpose**: Real-time communication for web interfaces
- **Features**: Live updates, event streaming, bidirectional messaging

### 1.1 Cross-Lobe Communication System

The system implements brain-inspired cross-lobe communication for coordinated processing:

#### Sensory Data Sharing (Implemented)
- **Purpose**: Standardized sensory data format for cross-lobe communication
- **Features**: 
  - Hormone-triggered sensory data propagation with priority adjustment
  - Propagation rule registration and management
  - Standardized data format across all lobes (`cross_lobe_sensory_data`)
  - Real-time cross-lobe data synchronization
  - Performance monitoring and statistics tracking
- **Implementation Status**: Fully implemented in Pattern Recognition Engine with comprehensive testing
- **Data Flow**: Sensory Input → Processing → Standardization → Rule Matching → Hormone-Based Propagation → Target Lobes
- **Key Components**:
  - `SensoryDataPropagator`: Manages propagation rules and data routing
  - `AdaptiveSensitivityManager`: Handles cross-column learning and hormone modulation
  - Hormone-based priority adjustment system for dynamic data routing

#### P2P Genetic Data Exchange (Implemented)
- **Purpose**: Secure, decentralized sharing of training data, engrams, and system optimizations
- **Features**:
  - 256-codon genetic encoding system for metadata and integration instructions
  - Privacy-preserving data sanitization pipeline
  - Cryptographic hashing and content-addressable storage
  - Multi-stage validation and integrity checking
- **Implementation Status**: Complete P2P network with DHT routing and genetic data exchange
- **Key Components**:
  - `P2PNetworkNode`: Handles network connectivity and data distribution
  - `GeneticDataExchange`: Manages genetic encoding and decoding
  - `EngramTransferManager`: Handles compressed memory structure sharing
  - `GeneticNetworkOrchestrator`: Coordinates network-wide genetic operations

### 2. Core Services

#### Implementation Switching Monitor
- **Purpose**: Manages automatic switching between algorithmic and neural implementations
- **Key Features**:
  - Performance metrics tracking and comparison
  - Automatic implementation selection based on performance
  - Fallback mechanisms for neural network failures
  - Model persistence and loading
  - Configurable switching thresholds
  - Stability period enforcement to prevent oscillation
- **Data Model**: Component → Implementation → Performance Metrics → Switching History

#### Workflow Manager
- **Purpose**: Manages project workflows and step progression
- **Key Features**:
  - Step lifecycle management (init, start, complete, fail)
  - Dependency tracking between steps
  - Feedback collection and analysis
  - Meta-step support for complex workflows
- **Data Model**: Workflow → Steps → Feedback

#### Task Manager
- **Purpose**: Manages individual tasks and their relationships
- **Key Features**:
  - Hierarchical task organization
  - Dependency management
  - Progress tracking with partial completion
  - Note and feedback collection
  - Tag-based organization
- **Data Model**: Task → Subtasks → Dependencies → Notes

#### Memory Manager
- **Purpose**: Intelligent memory storage and retrieval
- **Key Features**:
  - Vector-based semantic search
  - Memory type classification
  - Priority-based retrieval
  - Automatic compression and optimization
  - Multi-backend support (SQLite, Milvus, Qdrant, Annoy)
- **Data Model**: Memory → Vector → Metadata → Tags

#### Context Manager
- **Purpose**: Intelligent context management and optimization
- **Key Features**:
  - Context relevance scoring
  - Dynamic context chunking
  - Compression and decompression
  - Version control and history
  - Token optimization
- **Data Model**: Context → Chunks → Relevance → History

### 3. Experimental Lobes

The experimental lobes implement advanced cognitive functions inspired by human brain research:

#### Alignment Engine
- **Purpose**: Ensures system behavior aligns with user preferences
- **Implementation**: LLM-based preference analysis and feedback learning
- **Features**:
  - Preference modeling and tracking
  - Alignment scoring and suggestions
  - Feedback-driven model updates
  - Multi-dimensional preference analysis

#### Pattern Recognition Engine
- **Purpose**: Identifies patterns in data and behavior
- **Implementation**: Neural column simulation with batch processing
- **Features**:
  - Success/failure pattern recognition
  - Time complexity analysis
  - Resource usage patterns
  - Collaborative behavior patterns
  - Adaptive pattern learning

#### Simulated Reality Engine
- **Purpose**: Maintains a mental model of the external world
- **Implementation**: Entity-relationship-state tracking system
- **Features**:
  - Entity creation and management
  - Event tracking and causality
  - State monitoring and updates
  - Reality querying and inference
  - Temporal relationship modeling

#### Dreaming Engine
- **Purpose**: Generates insights through scenario simulation
- **Implementation**: Multi-agent simulation with insight extraction
- **Features**:
  - Scenario definition and simulation
  - Multi-iteration scenario testing
  - Insight extraction and analysis
  - Creative problem solving
  - Risk assessment and mitigation

#### Mind Map Engine
- **Purpose**: Creates and manages conceptual associations
- **Implementation**: Graph-based knowledge representation
- **Features**:
  - Node creation and management
  - Association strength calculation
  - Path finding and traversal
  - Centrality analysis
  - Dynamic graph updates

#### Scientific Process Engine
- **Purpose**: Implements scientific methodology for hypothesis testing
- **Implementation**: Structured hypothesis-experiment-analysis workflow
- **Features**:
  - Hypothesis creation and validation
  - Experimental design and execution
  - Statistical analysis and interpretation
  - Result validation and conclusion
  - Knowledge accumulation

#### Split Brain AB Test Engine
- **Purpose**: Implements parallel testing of different approaches
- **Implementation**: A/B testing with multi-dimensional comparison
- **Features**:
  - Parallel approach testing
  - Multi-metric comparison
  - Statistical significance analysis
  - Winner determination and recommendations
  - Continuous learning and adaptation

#### Multi-LLM Orchestrator
- **Purpose**: Manages multiple LLM instances for specialized tasks
- **Implementation**: Task routing and load balancing system
- **Features**:
  - Specialized LLM routing
  - Parallel task processing
  - Performance monitoring
  - Load balancing and optimization
  - Fallback mechanisms

#### Advanced Engram Engine
- **Purpose**: Manages dynamic coding models for memory compression
- **Implementation**: Neural network-based memory encoding/decoding
- **Features**:
  - Dynamic model creation and training
  - Memory compression and decompression
  - Engram merging and optimization
  - Predictive modeling
  - Adaptive learning

### 4. Data Layer

#### SQLite Database
- **Purpose**: Primary relational data storage
- **Schema**: Normalized relational schema for tasks, workflows, memories
- **Features**: ACID compliance, transaction support, backup/recovery

#### Vector Database
- **Purpose**: High-dimensional vector storage for semantic search
- **Backends**: SQLiteFAISS, Milvus, Qdrant, Annoy
- **Features**: Similarity search, clustering, dimensionality reduction

#### File System
- **Purpose**: Large file storage and management
- **Features**: Chunked storage, compression, deduplication

## Data Flow

### 1. Request Processing Flow

```
Client Request → API Layer → Authentication → Rate Limiting → 
Core Service → Experimental Lobe → Data Layer → Response
```

### 2. Memory Processing Flow

```
Input → Preprocessing → Vectorization → Storage → 
Indexing → Retrieval → Relevance Scoring → Context Assembly
```

### 3. Learning Flow

```
Feedback → Analysis → Pattern Recognition → Model Update → 
Performance Assessment → Optimization → Deployment
```

## Security Architecture

### 1. Authentication & Authorization
- **API Key Authentication**: Secure API key management
- **Rate Limiting**: Request throttling and abuse prevention
- **Input Validation**: Comprehensive input sanitization
- **Access Control**: Role-based access control (RBAC)

### 2. Data Security
- **Encryption**: Data encryption at rest and in transit
- **Backup Security**: Encrypted backup storage
- **Audit Logging**: Comprehensive audit trail
- **Data Privacy**: GDPR-compliant data handling

### 3. System Security
- **Secure Configuration**: Environment-based configuration
- **Dependency Scanning**: Regular security vulnerability scanning
- **Container Security**: Secure container deployment
- **Network Security**: Firewall and network isolation

## Performance Architecture

### 1. Scalability
- **Horizontal Scaling**: Multi-instance deployment support
- **Load Balancing**: Request distribution across instances
- **Caching**: Multi-level caching (memory, disk, distributed)
- **Database Optimization**: Connection pooling, query optimization

### 2. Monitoring
- **Metrics Collection**: Prometheus/Netdata integration
- **Real-time Monitoring**: System health and performance tracking
- **Alerting**: Automated alerting and notification system
- **Performance Analysis**: Detailed performance profiling

### 3. Optimization
- **Memory Management**: Efficient memory usage and garbage collection
- **CPU Optimization**: Multi-threading and async processing
- **I/O Optimization**: Efficient database and file operations
- **Network Optimization**: Connection pooling and compression

## Deployment Architecture

### 1. Portable Deployment
- **Self-contained Environment**: Complete Python environment
- **Cross-platform Support**: Linux, macOS, Windows
- **Dependency Bundling**: All dependencies included
- **Configuration Management**: Environment-specific configuration

### 2. Container Deployment
- **Docker Support**: Containerized deployment
- **Kubernetes Support**: Orchestrated deployment
- **Service Mesh**: Advanced networking and security
- **Auto-scaling**: Dynamic resource allocation

### 3. Cloud Deployment
- **Multi-cloud Support**: AWS, Azure, GCP compatibility
- **Serverless Options**: Function-as-a-Service deployment
- **Managed Services**: Database and storage integration
- **CDN Integration**: Global content delivery

## Plugin Architecture

### 1. Plugin System
- **Dynamic Loading**: Runtime plugin loading and unloading
- **API Extensions**: Custom API endpoint registration
- **Event System**: Plugin event subscription and publishing
- **Configuration Management**: Plugin-specific configuration

### 2. Plugin Development
- **SDK**: Comprehensive plugin development kit
- **Documentation**: Detailed plugin development guides
- **Testing Framework**: Plugin testing and validation tools
- **Marketplace**: Plugin distribution and discovery

## Integration Architecture

### 1. IDE Integration
- **VS Code**: Full VS Code extension support
- **Cursor**: Native Cursor integration
- **Claude Desktop**: Claude desktop app integration
- **LMStudio**: Local model studio integration
- **Ollama**: Ollama model integration

### 2. External Integrations
- **Git Integration**: Version control system integration
- **CI/CD Integration**: Continuous integration/deployment
- **Cloud Services**: AWS, Azure, GCP service integration
- **Monitoring Tools**: Prometheus, Grafana, etc.

## Development Architecture

### 1. Code Organization
- **Modular Design**: Clear separation of concerns
- **Dependency Injection**: Loose coupling between components
- **Interface-based Design**: Contract-based component interaction
- **Test-driven Development**: Comprehensive testing strategy

### 2. Quality Assurance
- **Static Analysis**: Code quality and security analysis
- **Unit Testing**: Comprehensive unit test coverage
- **Integration Testing**: End-to-end integration testing
- **Performance Testing**: Load and stress testing

### 3. Documentation
- **API Documentation**: Comprehensive API reference
- **Architecture Documentation**: System design documentation
- **User Guides**: End-user documentation
- **Developer Guides**: Developer onboarding and reference

## Future Architecture

### 1. Planned Enhancements
- **Distributed Computing**: Multi-node deployment support
- **Advanced AI Models**: Integration with cutting-edge AI models
- **Real-time Collaboration**: Multi-user collaborative features
- **Advanced Analytics**: Deep learning-based analytics

### 2. Research Integration
- **Academic Research**: Integration with academic research findings
- **Industry Best Practices**: Adoption of industry standards
- **Open Source Contributions**: Community-driven development
- **Continuous Innovation**: Ongoing research and development

# Memory Architecture (Updated)

## Overview
The MCP system now uses a three-tier memory architecture:
- **WorkingMemory**: Context-sensitive, temporary storage for feedback, adaptation, and short-lived data. Used by all lobes/engines for immediate, volatile information.
- **ShortTermMemory**: Stores recent, high-priority, or volatile information (e.g., recent tasks, user interactions, session data). Used for short-term recall and rapid adaptation.
- **LongTermMemory**: Persistent, structured, and research-driven storage (e.g., vector DBs, engrams, knowledge, historical logs). Used for long-term retention, research, and analytics.

## Integration
All lobes/engines should:
- Use **WorkingMemory** for context-sensitive, feedback, and temporary storage.
- Use **ShortTermMemory** for recent, high-priority, or volatile information.
- Use **LongTermMemory** for persistent, structured, and research-driven storage.

## Extensibility & Fallbacks
- Each memory type supports robust fallbacks and extensibility hooks.
- See `src/mcp/lobes/shared_lobes/working_memory.py` for implementation details.

## Research References
- See `idea.txt` for requirements and research sources.
- See `README.md` and `RESEARCH_SOURCES.md` for further reading.

## Example Usage
```
from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory, ShortTermMemory, LongTermMemory

wm = WorkingMemory()
wm.add({"context": "session", "data": "temp"})

stm = ShortTermMemory()
stm.add({"task": "recent", "priority": 1})

ltm = LongTermMemory()
ltm.add("knowledge_1", {"fact": "persistent"})
```

All lobes/engines should be updated to use the appropriate memory type for their needs.

## Integration Points (Updated)

- [[ThreeTierMemoryManager]]: Integrates WorkingMemory, ShortTermMemory, and LongTermMemory with hormone/genetic trigger hooks.
- [[UnifiedMemoryManager]]: Combines all memory features and supports advanced engram operations.
- [[MemoryLobe]]: Provides cross-lobe integration and hormone/genetic trigger hooks.
- [[GeneticTriggerSystem]], [[GeneticDataExchange]], [[IntegratedP2PGeneticSystem]]: Full genetic system integration, cross-pollination, and P2P data exchange.
- [[SharedSimulationState]], [[SimulatedReality]], [[PhysicsEngine]]: Simulation layer with shared state and cross-engine coordination.
- [[ResourceOptimizationIntegration]], [[PredictiveResourceAllocationIntegration]]: Resource optimization and predictive allocation integration.
- Experimental lobes: [[AdvancedEngramEngine]], [[DreamingEngine]], [[MindMapEngine]], etc. are cross-linked and referenced.
- Documentation is Obsidian-friendly and all features are cross-linked for maximum explorability.

## Planned Enhancements

- **Periodic Reporting & Dynamic Tagging**: Planned for future implementation. Will automate periodic QA/security reporting and dynamic tagging across all lobes and integration points.

This architecture documentation provides a comprehensive overview of the MCP server's design, implementation, and future direction.

# MCP System Architecture (Updated)

## Overview
This document reflects the latest MCP System Upgrade, including:
- Three-Tier Memory Manager (WorkingMemory, ShortTermMemory, LongTermMemory)
- Simulation Layer: WebSocialEngine
- Genetic Layer: Dual code/neural GeneticTrigger, hormone integration, A/B testing
- Integration Layer: P2P benchmarking, async processing, global performance projection

## Three-Tier Memory Manager
- See [unified_memory.py](../src/mcp/unified_memory.py)
- Integrates WorkingMemory, ShortTermMemory, LongTermMemory
- Automatic tier transitions, cross-tier search, and consolidation
- Used by all lobes/engines for memory operations

## Simulation Layer: WebSocialEngine
- See [web_social_engine.py](../src/mcp/lobes/web_social_engine.py)
- Interfaces: crawl_and_analyze_content, interact_with_social_media, manage_digital_identity, handle_captcha_challenges, generate_secure_credentials, assess_source_credibility

## Genetic Layer: GeneticTrigger System
- See [genetic_trigger.py](../src/mcp/genetic_trigger_system/genetic_trigger.py)
- Dual code/neural implementation, hormone system integration, split-brain A/B testing
- Environmental adaptation, mutation, crossover, performance tracking

## Integration Layer: P2P Benchmarking & Async
- See [p2p_network.py](../src/mcp/p2p_network.py), [integrated_p2p_genetic_system.py](../src/mcp/integrated_p2p_genetic_system.py)
- Secure benchmarking, async status visualization, global performance projection

## Cross-References
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- [USER_GUIDE.md](USER_GUIDE.md)
- [ADVANCED_API.md](ADVANCED_API.md)

## For details on each component, see the respective source files and linked documentation. 