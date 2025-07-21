"""
Database Schemas: SQL schemas for hormone system, genetic triggers, and monitoring.

This module defines the SQL schemas for the hormone system, genetic triggers, and
monitoring databases, providing a consistent structure for storing and retrieving data.

References:
- Requirements 1.1, 1.4, 5.1, 5.2 from MCP System Upgrade specification
"""

import sqlite3
from typing import Dict, List, Optional, Tuple, Union

# Schema version for tracking database migrations
SCHEMA_VERSION = 1


# Hormone System Schema
HORMONE_SYSTEM_SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Hormone definitions
CREATE TABLE IF NOT EXISTS hormones (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    base_decay_rate REAL NOT NULL,
    base_diffusion_coefficient REAL NOT NULL,
    receptor_affinity REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hormone levels over time
CREATE TABLE IF NOT EXISTS hormone_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hormone_id TEXT NOT NULL,
    level REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hormone_id) REFERENCES hormones(id)
);

-- Hormone release events
CREATE TABLE IF NOT EXISTS hormone_releases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hormone_id TEXT NOT NULL,
    source_lobe TEXT NOT NULL,
    quantity REAL NOT NULL,
    context TEXT,  -- JSON context
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hormone_id) REFERENCES hormones(id)
);

-- Hormone cascade definitions
CREATE TABLE IF NOT EXISTS hormone_cascades (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    trigger_hormone_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trigger_hormone_id) REFERENCES hormones(id)
);

-- Hormone cascade steps
CREATE TABLE IF NOT EXISTS hormone_cascade_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cascade_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    hormone_id TEXT NOT NULL,
    effect_type TEXT NOT NULL,  -- 'release', 'inhibit', 'amplify'
    effect_magnitude REAL NOT NULL,
    condition TEXT,  -- Optional condition for step activation
    FOREIGN KEY (cascade_id) REFERENCES hormone_cascades(id),
    FOREIGN KEY (hormone_id) REFERENCES hormones(id)
);

-- Hormone cascade activations
CREATE TABLE IF NOT EXISTS hormone_cascade_activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cascade_id TEXT NOT NULL,
    trigger_release_id INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cascade_id) REFERENCES hormone_cascades(id),
    FOREIGN KEY (trigger_release_id) REFERENCES hormone_releases(id)
);

-- Lobe receptor configurations
CREATE TABLE IF NOT EXISTS lobe_receptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lobe_id TEXT NOT NULL,
    hormone_id TEXT NOT NULL,
    sensitivity REAL NOT NULL,
    adaptation_rate REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hormone_id) REFERENCES hormones(id),
    UNIQUE(lobe_id, hormone_id)
);

-- Receptor sensitivity history
CREATE TABLE IF NOT EXISTS receptor_sensitivity_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lobe_id TEXT NOT NULL,
    hormone_id TEXT NOT NULL,
    sensitivity REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hormone_id) REFERENCES hormones(id)
);

-- Hormone diffusion parameters
CREATE TABLE IF NOT EXISTS diffusion_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hormone_id TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    parameter_value REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hormone_id) REFERENCES hormones(id),
    UNIQUE(hormone_id, parameter_name)
);

-- Lobe positions for diffusion calculations
CREATE TABLE IF NOT EXISTS lobe_positions (
    lobe_id TEXT PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    radius REAL NOT NULL,
    permeability REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Implementation performance metrics
CREATE TABLE IF NOT EXISTS implementation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    implementation_type TEXT NOT NULL,  -- 'algorithmic' or 'neural'
    accuracy REAL NOT NULL,
    speed REAL NOT NULL,
    resource_usage REAL NOT NULL,
    error_rate REAL NOT NULL,
    confidence_score REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Implementation switching events
CREATE TABLE IF NOT EXISTS implementation_switches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    from_implementation TEXT NOT NULL,
    to_implementation TEXT NOT NULL,
    reason TEXT NOT NULL,
    performance_improvement REAL NOT NULL,
    context TEXT,  -- JSON context
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial schema version
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- Insert default hormones
INSERT OR IGNORE INTO hormones (id, name, description, base_decay_rate, base_diffusion_coefficient, receptor_affinity)
VALUES 
    ('dopamine', 'Dopamine', 'Reward signaling', 0.05, 0.15, 0.7),
    ('serotonin', 'Serotonin', 'Confidence and decision stability', 0.03, 0.12, 0.6),
    ('cortisol', 'Cortisol', 'Stress response and priority adjustment', 0.02, 0.08, 0.8),
    ('adrenaline', 'Adrenaline', 'Urgency detection and acceleration', 0.08, 0.2, 0.7),
    ('oxytocin', 'Oxytocin', 'Collaboration and trust metrics', 0.04, 0.1, 0.9),
    ('growth_hormone', 'Growth Hormone', 'Learning rate adaptation', 0.01, 0.05, 0.6),
    ('gaba', 'GABA', 'Inhibitory control and noise reduction', 0.06, 0.1, 0.5),
    ('vasopressin', 'Vasopressin', 'Memory consolidation and learning enhancement', 0.03, 0.08, 0.8),
    ('thyroid', 'Thyroid Hormones', 'Processing speed regulation', 0.01, 0.05, 0.5),
    ('norepinephrine', 'Norepinephrine', 'Attention and focus enhancement', 0.07, 0.15, 0.7),
    ('acetylcholine', 'Acetylcholine', 'Learning and neural plasticity', 0.04, 0.1, 0.6),
    ('endorphins', 'Endorphins', 'System satisfaction and well-being metrics', 0.05, 0.12, 0.7);
"""


# Genetic Trigger System Schema
GENETIC_TRIGGER_SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Environmental signals
CREATE TABLE IF NOT EXISTS environmental_signals (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Environmental contexts
CREATE TABLE IF NOT EXISTS environmental_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_data TEXT NOT NULL  -- JSON representation of environment
);

-- Genetic triggers
CREATE TABLE IF NOT EXISTS genetic_triggers (
    id TEXT PRIMARY KEY,
    dna_signature TEXT NOT NULL,
    formation_environment_id INTEGER NOT NULL,
    activation_threshold REAL NOT NULL,
    creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activation TIMESTAMP,
    mutation_count INTEGER DEFAULT 0,
    FOREIGN KEY (formation_environment_id) REFERENCES environmental_contexts(id)
);

-- Codon activation maps
CREATE TABLE IF NOT EXISTS codon_maps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id TEXT NOT NULL,
    codon TEXT NOT NULL,
    activation_strength REAL NOT NULL,
    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers(id)
);

-- Performance history for genetic triggers
CREATE TABLE IF NOT EXISTS trigger_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id TEXT NOT NULL,
    performance_score REAL NOT NULL,
    context TEXT,  -- JSON context
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers(id)
);

-- Trigger activations
CREATE TABLE IF NOT EXISTS trigger_activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id TEXT NOT NULL,
    environment_id INTEGER NOT NULL,
    activation_score REAL NOT NULL,
    activated BOOLEAN NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers(id),
    FOREIGN KEY (environment_id) REFERENCES environmental_contexts(id)
);

-- Epigenetic memory
CREATE TABLE IF NOT EXISTS epigenetic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id TEXT NOT NULL,
    methylation_key TEXT NOT NULL,
    methylation_value REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers(id)
);

-- Histone modifications
CREATE TABLE IF NOT EXISTS histone_modifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id TEXT NOT NULL,
    gene_key TEXT NOT NULL,
    modification_type TEXT NOT NULL,  -- 'activating' or 'repressive'
    modification_value REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers(id)
);

-- Chromatin accessibility
CREATE TABLE IF NOT EXISTS chromatin_accessibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_id TEXT NOT NULL,
    gene_id TEXT NOT NULL,
    accessibility_state TEXT NOT NULL,  -- 'open', 'closed', 'facultative'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trigger_id) REFERENCES genetic_triggers(id)
);

-- Genetic mutations
CREATE TABLE IF NOT EXISTS genetic_mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_trigger_id TEXT NOT NULL,
    child_trigger_id TEXT NOT NULL,
    mutation_type TEXT NOT NULL,
    mutation_description TEXT,
    fitness_before REAL,
    fitness_after REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_trigger_id) REFERENCES genetic_triggers(id),
    FOREIGN KEY (child_trigger_id) REFERENCES genetic_triggers(id)
);

-- Insert initial schema version
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- Insert default environmental signals
INSERT OR IGNORE INTO environmental_signals (id, name, description)
VALUES 
    ('performance_change', 'Performance Change', 'Changes in system performance metrics'),
    ('learning_opportunity', 'Learning Opportunity', 'Opportunities for learning and adaptation'),
    ('stress_condition', 'Stress Condition', 'System under stress or resource constraints'),
    ('collaboration_request', 'Collaboration Request', 'Requests for collaboration between components'),
    ('adaptation_need', 'Adaptation Need', 'Need for adaptation to changing conditions'),
    ('optimization_trigger', 'Optimization Trigger', 'Triggers for optimization processes'),
    ('network_change', 'Network Change', 'Changes in network topology or connectivity');
"""


# Monitoring System Schema
MONITORING_SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- System states
CREATE TABLE IF NOT EXISTS system_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    state_data TEXT NOT NULL  -- JSON representation of system state
);

-- Hormone level snapshots
CREATE TABLE IF NOT EXISTS hormone_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_state_id INTEGER NOT NULL,
    hormone_levels TEXT NOT NULL,  -- JSON mapping hormone IDs to levels
    FOREIGN KEY (system_state_id) REFERENCES system_states(id)
);

-- Neural performance metrics
CREATE TABLE IF NOT EXISTS neural_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_state_id INTEGER NOT NULL,
    component TEXT NOT NULL,
    metrics TEXT NOT NULL,  -- JSON performance metrics
    FOREIGN KEY (system_state_id) REFERENCES system_states(id)
);

-- Genetic trigger activations
CREATE TABLE IF NOT EXISTS genetic_activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_state_id INTEGER NOT NULL,
    trigger_id TEXT NOT NULL,
    environment TEXT NOT NULL,  -- JSON environment state
    behavior_changes TEXT,  -- JSON list of behavior changes
    FOREIGN KEY (system_state_id) REFERENCES system_states(id)
);

-- Implementation switches
CREATE TABLE IF NOT EXISTS implementation_switches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_state_id INTEGER NOT NULL,
    component TEXT NOT NULL,
    old_impl TEXT NOT NULL,
    new_impl TEXT NOT NULL,
    reason TEXT NOT NULL,
    comparison TEXT NOT NULL,  -- JSON comparison result
    FOREIGN KEY (system_state_id) REFERENCES system_states(id)
);

-- Performance reports
CREATE TABLE IF NOT EXISTS performance_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    report_data TEXT NOT NULL,  -- JSON report data
    report_type TEXT NOT NULL
);

-- Anomalies
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_state_id INTEGER NOT NULL,
    anomaly_type TEXT NOT NULL,
    severity REAL NOT NULL,
    description TEXT NOT NULL,
    related_components TEXT,  -- JSON list of related components
    FOREIGN KEY (system_state_id) REFERENCES system_states(id)
);

-- Visualizations
CREATE TABLE IF NOT EXISTS visualizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    visualization_type TEXT NOT NULL,
    data TEXT NOT NULL,  -- JSON visualization data
    parameters TEXT  -- JSON visualization parameters
);

-- Time series data
CREATE TABLE IF NOT EXISTS time_series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_name TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    value REAL NOT NULL,
    context TEXT  -- JSON context
);

-- Insert initial schema version
INSERT OR IGNORE INTO schema_version (version) VALUES (1);
"""


def initialize_database(db_path: str, schema: str) -> None:
    """
    Initialize a database with the given schema.
    
    Args:
        db_path: Path to the database file
        schema: SQL schema to initialize the database with
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.commit()
    conn.close()


def initialize_all_databases() -> None:
    """Initialize all databases with their schemas."""
    initialize_database("data/hormone_system.db", HORMONE_SYSTEM_SCHEMA)
    initialize_database("data/genetic_trigger_system.db", GENETIC_TRIGGER_SCHEMA)
    initialize_database("data/monitoring_system.db", MONITORING_SCHEMA)


def check_schema_version(db_path: str) -> int:
    """
    Check the schema version of a database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Schema version, or 0 if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            return 0
    except sqlite3.OperationalError:
        return 0


def migrate_database(db_path: str, schema: str, target_version: int) -> bool:
    """
    Migrate a database to the target schema version.
    
    Args:
        db_path: Path to the database file
        schema: SQL schema to migrate to
        target_version: Target schema version
        
    Returns:
        True if migration was successful, False otherwise
    """
    current_version = check_schema_version(db_path)
    
    if current_version == target_version:
        return True
    
    if current_version > target_version:
        print(f"Warning: Database {db_path} has schema version {current_version}, "
              f"but target version is {target_version}")
        return False
    
    # For now, just reinitialize the database
    # In a real implementation, we would have migration scripts
    initialize_database(db_path, schema)
    
    return check_schema_version(db_path) == target_version


if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize all databases
    initialize_all_databases()
    
    # Check schema versions
    hormone_version = check_schema_version("data/hormone_system.db")
    genetic_version = check_schema_version("data/genetic_trigger_system.db")
    monitoring_version = check_schema_version("data/monitoring_system.db")
    
    print(f"Hormone system schema version: {hormone_version}")
    print(f"Genetic trigger system schema version: {genetic_version}")
    print(f"Monitoring system schema version: {monitoring_version}")