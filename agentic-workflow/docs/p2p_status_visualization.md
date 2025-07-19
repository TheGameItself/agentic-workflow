# P2P Status Visualization System

The P2P Status Visualization System provides real-time monitoring and visualization of peer-to-peer network status using a sophisticated red-green-white status bar architecture. This system enables comprehensive network health monitoring, user capability assessment, and reputation-based query server identification.

## Architecture Overview

### Red-Green-White Status Bar

The system implements a three-segment status bar that provides immediate visual feedback on network status:

- **Green Section (Top-aligned)**: Idle users ready for queries
- **White Section (Middle divider)**: High-reputation capable query servers  
- **Red Section (Bottom-aligned)**: Active online non-idle users

### Core Components

#### P2PStatusVisualizer
Main visualization system that manages real-time status updates and proportional bar segment sizing.

#### ReputationScorer
Calculates reputation scores for P2P network users based on multiple performance factors.

#### P2PStatusBarRenderer
Provides multiple rendering formats (ASCII, HTML, JSON) for different interface requirements.

## Key Features

### Real-Time Status Updates
- Continuous monitoring with configurable update intervals
- Proportional bar segment sizing based on actual user counts
- Performance tracking and optimization

### Reputation Scoring System
The system calculates user reputation based on multiple weighted factors:

- **Uptime (25%)**: Normalized to 24-hour periods
- **Successful Transfers (30%)**: Success rate of data transfers
- **Response Time (20%)**: Average response time (normalized to 5 seconds)
- **Data Quality (15%)**: Quality score of shared data
- **Network Contribution (10%)**: Overall contribution to network health

### User Capability Assessment
Automatically assesses user capabilities based on performance metrics:

- `high_reputation_server`: Reputation score ≥ 0.8
- `fast_response`: Average response time < 2 seconds
- `experienced`: > 100 successful transfers
- `stable`: Uptime > 1 hour
- `high_quality_data`: Data quality score > 0.9

### Network Health Monitoring
Calculates overall network health using multiple indicators:

- Uptime score (normalized to 24 hours)
- Peer connectivity (normalized to 10+ peers)
- Genetic diversity metrics
- Network fitness scores

## Usage Examples

### Basic Status Visualization

```python
from mcp.p2p_status_visualization import P2PStatusVisualizer

# Create visualizer with P2P system
visualizer = P2PStatusVisualizer(p2p_system)

# Update status
await visualizer.update_status()

# Render status bar
status_bar = visualizer.render_status_bar()
print(status_bar)
# Output: [GREEN ●x5] [WHITE ◆x3] [RED ●x2] 🟢 Health: 80.0%

# Get detailed status
details = visualizer.get_detailed_status()
print(f"Total users: {details['total_users']}")
print(f"Network health: {details['network_health']:.1%}")
```

### Reputation Scoring

```python
from mcp.p2p_status_visualization import ReputationScorer

scorer = ReputationScorer()

# Calculate user reputation
user_data = {
    'uptime': 86400,  # 24 hours
    'successful_transfers': 100,
    'total_transfers': 110,
    'avg_response_time': 1000,
    'data_quality_score': 0.9,
    'network_contribution_score': 0.8
}

reputation = scorer.calculate_user_reputation(user_data)
capabilities = scorer.assess_user_capabilities(user_data)

print(f"Reputation: {reputation:.3f}")
print(f"Capabilities: {capabilities}")
```

### Multiple Rendering Formats

```python
from mcp.p2p_status_visualization import P2PStatusBarRenderer

# Create test segments
segments = [
    UserSegment(UserStatus.IDLE, 5, 50.0, "5 idle users", 0.5),
    UserSegment(UserStatus.HIGH_REPUTATION, 3, 30.0, "3 high-rep servers", 0.9),
    UserSegment(UserStatus.ACTIVE, 2, 20.0, "2 active users", 0.6)
]

# ASCII rendering
ascii_bar = P2PStatusBarRenderer.render_ascii_bar(segments, width=30)
print(f"ASCII: {ascii_bar}")

# HTML rendering
html_bar = P2PStatusBarRenderer.render_html_bar(segments)
print(f"HTML: {html_bar[:100]}...")

# JSON rendering
status_data = StatusBarData(segments, 10, time.time(), 0.8, 0.7)
json_status = P2PStatusBarRenderer.render_json_status(status_data)
print(f"JSON: {json_status}")
```

### Continuous Monitoring

```python
import asyncio

async def monitor_network():
    visualizer = P2PStatusVisualizer(p2p_system)
    
    # Start continuous monitoring
    await visualizer.start_monitoring()

# Run monitoring in background
asyncio.create_task(monitor_network())
```

## Configuration

### Update Intervals
Configure monitoring frequency based on network requirements:

```python
# High-frequency updates for real-time applications
visualizer = P2PStatusVisualizer(p2p_system, update_interval=0.5)

# Standard updates for most applications
visualizer = P2PStatusVisualizer(p2p_system, update_interval=1.0)

# Low-frequency updates for resource-constrained environments
visualizer = P2PStatusVisualizer(p2p_system, update_interval=5.0)
```

### Reputation Thresholds
Adjust reputation thresholds for different network requirements:

```python
# High-reputation threshold for critical applications
visualizer.high_reputation_threshold = 0.9

# Standard threshold for general use
visualizer.high_reputation_threshold = 0.8

# Lower threshold for development/testing
visualizer.high_reputation_threshold = 0.6
```

### Timeout Configuration
Configure timeouts for different user states:

```python
# Idle timeout (5 minutes)
visualizer.idle_timeout = 300

# Active timeout (1 minute)
visualizer.active_timeout = 60
```

## Performance Considerations

### Update Frequency
- **Real-time applications**: 0.5-1.0 second intervals
- **Standard monitoring**: 1.0-5.0 second intervals
- **Resource-constrained**: 5.0+ second intervals

### Memory Usage
- Status history limited to 100 updates by default
- Performance metrics limited to 50 samples
- Configurable limits for different deployment scenarios

### Network Impact
- Minimal bandwidth usage for status updates
- Efficient data structures for real-time processing
- Caching mechanisms for repeated queries

## Integration with Genetic System

The P2P status visualization system integrates seamlessly with the genetic expression architecture:

### Genetic-Aware Status Updates
```python
# Genetic interruption points based on P2P status
def p2p_health_condition(context):
    return visualizer.current_status.network_health < 0.5

def p2p_optimization_handler(context):
    return {"action": "optimize_network", "reason": "low_health"}

architecture.add_interruption_point(
    "p2p_health_check",
    InterruptionType.PERFORMANCE,
    p2p_health_condition,
    p2p_optimization_handler
)
```

### Reputation-Based Genetic Selection
```python
# Use P2P reputation for genetic sequence selection
top_sequences = architecture.get_top_performing_sequences(limit=10)
p2p_peer_sequences = [
    seq for seq in top_sequences 
    if "peer" in seq.sequence_id and seq.overall_score > 0.8
]
```

## Error Handling

### Graceful Degradation
- Continues operation with partial data
- Fallback to default values for missing metrics
- Comprehensive error logging and monitoring

### Timeout Management
- Configurable timeouts for all operations
- Automatic retry mechanisms for failed updates
- Performance monitoring for slow operations

### Data Validation
- Input validation for all user data
- Sanitization of network metrics
- Confidence scoring for reputation calculations

## Monitoring and Debugging

### Performance Metrics
```python
# Get system performance metrics
metrics = visualizer.get_performance_metrics()
print(f"Average update time: {metrics['avg_update_time']:.3f}s")
print(f"Update count: {metrics['update_count']}")
```

### Status History
```python
# Get recent status history for trend analysis
history = visualizer.get_status_history(limit=10)
for entry in history:
    print(f"Timestamp: {entry['timestamp']}")
    print(f"Total users: {entry['total_users']}")
    print(f"Network health: {entry['network_health']:.1%}")
```

### Debug Logging
```python
import logging

# Enable debug logging
logging.getLogger("P2PStatusVisualizer").setLevel(logging.DEBUG)

# Monitor specific events
visualizer.logger.info("Status update completed")
visualizer.logger.warning("Slow status updates detected")
visualizer.logger.error("Error in status monitoring")
```

## Related Documentation

- [[genetic_expression_architecture]]: Advanced genetic expression architecture
- [[p2p_network]]: P2P network implementation
- [[integrated_p2p_genetic_system]]: Integrated P2P and genetic systems
- [[genetic_trigger_system]]: Genetic trigger system
- [[monitoring_system]]: System monitoring and visualization

## Future Enhancements

### Planned Features
- GUI-based status visualization
- WebSocket real-time updates
- Advanced analytics and trend analysis
- Machine learning-based reputation prediction
- Distributed status aggregation

### Performance Optimizations
- Parallel status updates
- Intelligent caching strategies
- Adaptive update frequencies
- Resource-aware monitoring

---

For more information, see:
- [[ARCHITECTURE.md]]
- [[P2P_BENCHMARKING.md]]
- [[genetic_performance_benchmarking]] 