#!/usr/bin/env python3
"""
Simple test script for neural network models without MCP dependencies.
"""

import os
import sys
import numpy as np
import random
from pathlib import Path

# Create data directories
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

print("Testing neural network models...")

# Test hormone system
print("\n=== Testing Hormone System ===")

# Simple hormone system implementation
class SimpleHormoneSystem:
    def __init__(self):
        self.hormone_levels = {
            'stress': 0.0,
            'efficiency': 1.0,
            'adaptation': 0.5,
            'stability': 1.0
        }
    
    def update_hormone_levels(self, metrics):
        # Stress hormone - increases with high resource usage
        if 'cpu_usage' in metrics or 'memory_usage' in metrics:
            cpu = metrics.get('cpu_usage', 0.0) / 100.0
            memory = metrics.get('memory_usage', 0.0) / 100.0
            resource_stress = (cpu + memory) / 2.0
            
            self.hormone_levels['stress'] = (
                self.hormone_levels['stress'] * 0.9 + resource_stress * 0.1
            )
        
        # Efficiency hormone - decreases with errors and high response times
        if 'error_count' in metrics or 'response_time' in metrics:
            error_factor = 0.0
            if 'error_count' in metrics:
                error_factor = max(0.0, min(1.0, metrics['error_count'] / 100.0))
            
            time_factor = 0.0
            if 'response_time' in metrics:
                time_factor = max(0.0, min(1.0, metrics['response_time'] / 10.0))
            
            efficiency_impact = (error_factor + time_factor) / 2.0
            
            self.hormone_levels['efficiency'] = (
                self.hormone_levels['efficiency'] * 0.95 - efficiency_impact * 0.05
            )
        
        # Clamp all hormone levels to valid range
        for hormone in self.hormone_levels:
            self.hormone_levels[hormone] = max(0.0, min(1.0, self.hormone_levels[hormone]))
        
        return self.hormone_levels

# Create hormone system
hormone_system = SimpleHormoneSystem()

# Test with some metrics
metrics = {
    'cpu_usage': 75.0,
    'memory_usage': 80.0,
    'error_count': 10,
    'response_time': 1.0
}

hormone_levels = hormone_system.update_hormone_levels(metrics)
print(f"Hormone levels: {hormone_levels}")

# Test with different metrics
metrics = {
    'cpu_usage': 30.0,
    'memory_usage': 40.0,
    'error_count': 2,
    'response_time': 0.2
}

hormone_levels = hormone_system.update_hormone_levels(metrics)
print(f"Updated hormone levels: {hormone_levels}")

# Test brain state system
print("\n=== Testing Brain State System ===")

class SimpleBrainState:
    def __init__(self, state_dim=16):
        self.state_dim = state_dim
        self.current_state = np.zeros(state_dim)
        self.hormone_levels = {
            'stress': 0.0,
            'efficiency': 1.0,
            'adaptation': 0.5,
            'stability': 1.0
        }
    
    def update_brain_state(self, metrics, hormone_levels=None):
        # Use provided hormone levels or default
        hormones = hormone_levels or self.hormone_levels
        
        # Create feature vector from metrics and hormones
        features = []
        
        # Add metrics
        for key in ['cpu_usage', 'memory_usage', 'error_count', 'response_time']:
            features.append(metrics.get(key, 0.0))
        
        # Add hormone levels
        for hormone in ['stress', 'efficiency', 'adaptation', 'stability']:
            features.append(hormones.get(hormone, 0.5))
        
        # Normalize features
        features = [f / 100.0 if i < 2 else f for i, f in enumerate(features)]
        
        # Ensure we have enough features
        while len(features) < self.state_dim:
            features.append(0.0)
        
        # Take only what we need
        features = features[:self.state_dim]
        
        # Update state with some momentum
        self.current_state = 0.8 * self.current_state + 0.2 * np.array(features)
        
        return self.current_state

# Create brain state system
brain_state = SimpleBrainState()

# Test with metrics and hormone levels
state = brain_state.update_brain_state(metrics, hormone_levels)
print(f"Brain state: {state}")

# Test with different metrics
metrics = {
    'cpu_usage': 90.0,
    'memory_usage': 85.0,
    'error_count': 20,
    'response_time': 3.0
}

state = brain_state.update_brain_state(metrics, hormone_levels)
print(f"Updated brain state: {state}")

print("\nAll tests completed successfully!")