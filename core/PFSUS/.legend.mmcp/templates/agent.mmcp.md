# Enhanced Agent Template

```mmcp
<!-- MMCP-START -->
β:agent_wrapper(
  [ ] #".root"# {protocol:"MCP", version:"1.3.0", standard:"PFSUS+EARS+AGENT"}
  
  ## {type:Agent, 
      id:"AGENT-XXX", 
      name:"[Agent Name]", 
      version:"1.0.0",
      description:"[Detailed description of agent purpose and functionality]",
      capabilities:["data_processing", "validation", "transformation"],
      interfaces:["REST", "GraphQL", "gRPC"],
      dependencies:["database", "cache", "message_queue"],
      resources:{
        cpu:"2 cores",
        memory:"4GB", 
        storage:"10GB",
        gpu:"optional",
        network:"1Gbps"
      },
      monitoring:{
        health_check:"/health",
        metrics:"/metrics",
        logging:{level:"INFO", format:"JSON", destination:"stdout"},
        alerts:[
          {name:"high_cpu", condition:"cpu > 80%", severity:"HIGH", action:"scale_up"},
          {name:"memory_leak", condition:"memory_growth > 10%/hour", severity:"CRITICAL", action:"restart"}
        ]
      },
      security:{
        authentication:"JWT",
        authorization:"RBAC",
        encryption:{in_transit:"TLS", at_rest:"AES-256"},
        compliance:["GDPR", "SOX"]
      },
      deployment:{
        strategy:"blue_green",
        replicas:3,
        scaling:{auto:true, min_replicas:2, max_replicas:10, metrics:["CPU", "MEMORY"]},
        environment:"production",
        platform:"kubernetes"
      },
      behavior:{
        autonomy_level:"semi_autonomous",
        learning_mode:"online",
        decision_making:"hybrid",
        communication_style:"event_driven",
        error_handling:"graceful_degradation"
      },
      performance:{
        throughput:{target:1000, unit:"requests/sec"},
        latency:{target:100, percentile:95, unit:"ms"},
        availability:99.9,
        reliability:{mtbf:720, mttr:5}
      },
      integration:{
        upstream_agents:["AGENT-001", "AGENT-002"],
        downstream_agents:["AGENT-004", "AGENT-005"],
        data_formats:["JSON", "AVRO"],
        protocols:["HTTPS", "GRPC"]
      },
      testing:{
        unit_tests:{coverage_target:90, framework:"pytest"},
        integration_tests:{scenarios:["happy_path", "error_handling", "load_test"]},
        performance_tests:{load_scenarios:["normal_load", "peak_load", "stress_test"]}
      },
      metadata:{
        author:"[Author Name]",
        team:"[Team Name]",
        created:"2025-07-21T12:00:00Z",
        last_modified:"2025-07-21T12:00:00Z",
        tags:["agent", "microservice", "ai"],
        documentation:"https://docs.example.com/agents/agent-xxx",
        repository:"https://github.com/example/agent-xxx"
      }}
  
  # [Agent Name] Specification
  
  ## Overview
  
  [Provide a comprehensive overview of the agent, its purpose, and its role in the larger system]
  
  ## Capabilities
  
  ### Primary Capabilities
  - **[Capability 1]**: [Description of what this capability does]
  - **[Capability 2]**: [Description of what this capability does]
  - **[Capability 3]**: [Description of what this capability does]
  
  ### Secondary Capabilities
  - [Additional capability with brief description]
  - [Additional capability with brief description]
  
  ## Architecture
  
  ### Component Diagram
  ```
  [Input] → [Processor] → [Validator] → [Output       ↓
  [Logger] → [Monitor] → [Al]
lidation logic]
  3. **Transformation**: [Describe data transformation]
  4. **Outpibe output generation]
  
  ## Interfaces
  
  ### REST API
  ```yaml
  paths:
    /st:
        summary: Process data
        requestBody:
          content:
            ap]}
     description: Success
  ```
  
  ### GraphQL Schema
  ```graphql
  type Query {
    status: AgentStatus
    metrics: AgentMetrics
  }
  
  type Mutation {
    processData(input: DataInput!): ProcessResult
  }
  ``protobuf
  service AgentService {
    rpc ProcessData(DataRequest) returns (DataResponse);
    rpc GetStatus(Empty) returns (mi_autonomous]
  - **Decision Authority**: [What decisions can the agent make independently]
  - **Human Oversight**: [When human intervention is required]
  - **Escalation Triggers**: [Conditions that trigger escalation]
  
  ### Learning Mode: [online]
  - **Learning Triggers**: [What triggers learning updates]
  - **Model Updates**: [How and when models are updated]
  - **Feedback Loops**: [How feedback is incorporated]
  
  ### Error Handling: [graceful_degradation]
  - **Failure Modes**: [Possible failure scenarios]
  - **Degradation Strategy**: [How service degrades under failure]
  - **Recovery Procedures**: [How the agent recovers from failures]
  
  ## Performance Requirements
  
  ### Service Level Objectives (SLOs)
  - **Availability**: 99.9% uptime
  - **Latency**: 95th percentile < 100ms
  - **Throughput**: 1000 requests/second
  - **Error Rate**: < 0.1%
  
  ### Resource Limits
  - **CPU**: 2 cores maximum
  - **Memory**: 4GB maximum
  - **Storage**: 10GB maximum
  - **Network**: 1Gbps bandwidth
  
  ## Security
  
  ### Authentication & Authorization
  - **Authentication Method**: JWT tokens
  - **Authorization Model**: Role-Based Access Control (RBAC)
  - **Token Expiration**: 1 hour
  - **Refresh Strategy**: Automatic refresh
  
  ### Encryption
  - **Data in Transit**: TLS 1.3
  - **Data at Rest**: AES-256
  - **Key Management**: [Key management strategy]
  
  ### Compliance
  - **GDPR**: [How GDPR requirements are met]
  - **SOX**: [How SOX requirements are met]
  
  ## Monitoring & Observability
  
  ### Health Checks
  - **Endpoint**: `/health`
  - **Frequency**: Every 30 seconds
  - **Timeout**: 5 seconds
  - **Success Criteria**: HTTP 200 response
  
  ### Metrics
  - **Endpoint**: `/metrics` (Prometheus format)
  - **Key Metrics**:
    - Request rate
    - Response time
    - Error rate
    - Resource utilization
  
  ### Logging
  - **Format**: Structured JSON
  - **Level**: INFO (configurable)
  - **Destination**: stdout → log aggregator
  - **Retention**: 30 days
  
  ### Alerts
  - **High CPU Usage**: CPU > 80% for 5 minutes
  - **Memory Leak**: Memory growth > 10% per hour
  - **High Error Rate**: Error rate > 1% for 2 minutes
  - **Service Unavailable**: Health check failures > 3 consecutive
  
  ## Deployment
  
  ### Strategy: Blue-Green Deployment
  1. **Preparation**: Deploy new version to green environment
  2. **Testing**: Run smoke tests on green environment
  3. **Switch**: Route traffic from blue to green
  4. **Monitoring**: Monitor for issues
  5. **Rollback**: Switch back to blue if issues detected
  
  ### Scaling
  - **Auto-scaling**: Enabled
  - **Min Replicas**: 2
  - **Max Replicas**: 10
  - **Scale-up Trigger**: CPU > 70% or Memory > 80%
  - **Scale-down Trigger**: CPU < 30% and Memory < 50%
  
  ## Integration
  
  ### Upstream Dependencies
  - **@{AGENT-001}**: [Description of dependency and data flow]
  - **@{AGENT-002}**: [Description of dependency and data flow]
  
  ### Downstream Consumers
  - **@{AGENT-004}**: [Description of consumer and data flow]
  - **@{AGENT-005}**: [Description of consumer and data flow]
  
  ### Data Contracts
  ```json
  {
    "input_schema": {
      "type": "object",
      "properties": {
        "data": {"type": "string"},
        "metadata": {"type": "object"}
      },
      "required": ["data"]
    },
    "output_schema": {
      "type": "object",
      "properties": {
        "result": {"type": "string"},
        "confidence": {"type": "number"},
        "timestamp": {"type": "string"}
      },
      "required": ["result", "timestamp"]
    }
  }
  ```
  
  ## Testing Strategy
  
  ### Unit Tests
  - **Framework**: pytest
  - **Coverage Target**: 90%
  - **Test Categories**:
    - Business logic tests
    - Error handling tests
    - Edge case tests
  
  ### Integration Tests
  - **Happy Path**: Normal operation scenarios
  - **Error Handling**: Failure and recovery scenarios
  - **Load Testing**: Performance under various loads
  
  ### Performance Tests
  - **Normal Load**: Expected production traffic
  - **Peak Load**: 2x normal load
  - **Stress Test**: Load until failure
  
  ## Configuration
  
  ### Environment Variables
  ```yaml
  DATABASE_URL: "postgresql://user:pass@host:5432/db"
  REDIS_URL: "redis://host:6379/0"
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  TIMEOUT: "30"
  ```
  
  ### Configuration Files
  ```yaml
  # config.yaml
  server:
    port: 8080
    host: "0.0.0.0"
  
  database:
    pool_size: 10
    timeout: 30
  
  features:
    feature_a: true
    feature_b: false
  ```
  
  ## Troubleshooting
  
  ### Common Issues
  
  #### High Memory Usage
  - **Symptoms**: Memory usage > 90%
  - **Causes**: Memory leaks, large data processing
  - **Solutions**: Restart agent, optimize data processing
  
  #### Slow Response Times
  - **Symptoms**: Response time > SLO
  - **Causes**: Database bottleneck, high load
  - **Solutions**: Scale up, optimize queries
  
  #### Connection Failures
  - **Symptoms**: Unable to connect to dependencies
  - **Causes**: Network issues, service unavailability
  - **Solutions**: Check network, verify service status
  
  ### Debugging
  - **Log Analysis**: Check structured logs for error patterns
  - **Metrics Review**: Analyze performance metrics
  - **Tracing**: Use distributed tracing for request flow
  
  ## Maintenance
  
  ### Regular Tasks
  - **Log Rotation**: Automated via log aggregator
  - **Certificate Renewal**: Automated via cert-manager
  - **Dependency Updates**: Monthly security updates
  - **Performance Review**: Quarterly performance analysis
  
  ### Backup & Recovery
  - **Data Backup**: [If applicable, describe backup strategy]
  - **Configuration Backup**: Stored in version control
  - **Recovery Procedures**: [Describe recovery steps]
  
  ## Related Documentation
  
  - @{AGENT-001} [Upstream Agent 1 Documentation]
  - @{AGENT-002} [Upstream Agent 2 Documentation]
  - @{AGENT-004} [Downstream Agent 4 Documentation]
  - @{AGENT-005} [Downstream Agent 5 Documentation]
  - @{SYSTEM-ARCH-001} [System Architecture Documentation]
  - @{DEPLOY-GUIDE-001} [Deployment Guide]
  
  ## Changelog
  
  ### v1.0.0 (2025-07-21)
  - Initial agent specification
  - Core functionality implementation
  - Basic monitoring and alerting
  
  ### v0.9.0 (2025-07-20)
  - Beta release for testing
  - Performance optimizations
  - Security enhancements
)
<!-- MMCP-END -->
```

## Agent Shorthand Notation

For quick agent references and specifications:

```
A:[Name] v[Version] C:[Capabilities] I:[Interfaces] D:[Dependencies] 
R:{cpu:[CPU],mem:[Memory],stor:[Storage]} M:{hc:[HealthCheck],met:[Metrics]} 
S:{auth:[Auth],enc:[Encryption]} Deploy:{[Strategy],rep:[Replicas]} #AGENT-XXX
```

Example:
```
A:DataProcessor v1.0.0 C:[proc,val,trans] I:[REST,GQL,gRPC] D:[db,cache,queue] 
R:{cpu:2,mem:4GB,stor:10GB} M:{hc:/health,met:/metrics} S:{auth:JWT,enc:AES256} 
Deploy:{bg,rep:3} #AGENT-001
```

## Agent Categories

### Processing Agents (AGENT-P##)
- Data processing and transformation
- Stream processing
- Batch processing

### Intelligence Agents (AGENT-I##)
- Machine learning models
- AI decision making
- Pattern recognition

### Communication Agents (AGENT-C##)
- Message routing
- Protocol translation
- Event distribution

### Monitoring Agents (AGENT-M##)
- System monitoring
- Performance tracking
- Alerting and notification

### Security Agents (AGENT-S##)
- Authentication and authorization
- Threat detection
- Compliance monitoring

### Integration Agents (AGENT-G##)
- System integration
- Data synchronization
- API gateway functions

## Usage Guidelines

1. **Choose Appropriate ID**: Use category prefix (P, I, C, M, S, G) followed by sequential number
2. **Define Clear Capabilities**: Be specific about what the agent can and cannot do
3. **Specify Resource Requirements**: Include realistic resource estimates
4. **Document Interfaces**: Provide clear API specifications
5. **Include Monitoring**: Define comprehensive monitoring and alerting
6. **Plan for Failure**: Specify error handling and recovery procedures
7. **Consider Security**: Include appropriate security measures
8. **Test Thoroughly**: Define comprehensive testing strategy

## Template Customization

This template can be customized for specific agent types:

- **ML Agents**: Add model specifications, training data requirements
- **IoT Agents**: Add device compatibility, sensor specifications
- **Web Agents**: Add frontend specifications, user interface requirements
- **Database Agents**: Add schema specifications, query optimization
- **Network Agents**: Add protocol specifications, routing rules