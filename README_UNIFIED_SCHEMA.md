# Unified Reasoning Engine Schema Documentation

## Overview

This document describes the comprehensive schema for logging the Unified Reasoning Engine's complete reasoning process. The schema is based on AI system design principles and mathematical foundations from linear algebra, calculus, and statistics.

## Schema Architecture

### 🏗️ Core Design Principles

1. **Mathematical Rigor**: Every reasoning step is mathematically grounded
2. **Complete Traceability**: Full lineage from input to output
3. **Multi-Expert Validation**: Cross-validation between different reasoning approaches
4. **Information Theoretic Foundations**: Entropy, compression, and complexity metrics
5. **AI System Design**: Performance metrics, validation, and reproducibility

### 📊 Schema Sections

#### 1. Session Metadata
```json
{
  "session_id": "UUID for unique identification",
  "timestamp": "ISO 8601 timestamp", 
  "engine_version": "Version tracking for reproducibility",
  "model_configuration": {
    "temperature": "LLM temperature parameter",
    "max_tokens": "Token limit for generation"
  }
}
```

**Mathematical Foundation**: Session identification uses UUID4 for collision resistance (~10^36 unique IDs).

#### 2. Input Analysis  
```json
{
  "structural_analysis": {
    "shannon_entropy": "H(X) = -Σ p(xi)log(p(xi))",
    "compression_ratio": "Kolmogorov complexity approximation",
    "concept_graph": "Graph-theoretic representation G(V,E)"
  },
  "domain_detection": {
    "confidence_score": "Bayesian posterior probability"
  }
}
```

**Mathematical Foundation**:
- **Shannon Entropy**: Measures information content and predictability
- **Graph Theory**: Concepts represented as nodes, relationships as edges
- **Bayesian Inference**: Domain classification with uncertainty quantification

#### 3. Reasoning Process
```json
{
  "expert_analyses": [
    {
      "confidence_score": "Expert certainty ∈ [0,1]",
      "mathematical_formulation": "Domain-specific mathematical representation",
      "causal_chain": "Directed causal relationships"
    }
  ],
  "reasoning_paths": [
    {
      "step_by_step_process": "Detailed reasoning steps with mathematical justification",
      "causal_graph": "DAG representing causal structure"
    }
  ]
}
```

**Mathematical Foundation**:
- **Linear Algebra**: Vector spaces, transformations, eigenanalysis
- **Calculus**: Derivatives, integrals, optimization conditions  
- **Statistics**: Probability distributions, Bayesian inference
- **Graph Theory**: Directed Acyclic Graphs (DAGs) for causal reasoning

#### 4. Mathematical Foundations
```json
{
  "linear_algebra_components": {
    "eigenanalysis": "Spectral decomposition A = QΛQ^T",
    "vector_spaces": "Basis vectors and transformations"
  },
  "calculus_components": {
    "optimization_conditions": "∇f(x) = 0 for critical points",
    "lagrange_multipliers": "Constrained optimization"
  },
  "statistical_components": {
    "bayesian_inference": "P(H|E) = P(E|H)P(H)/P(E)",
    "confidence_intervals": "Statistical uncertainty bounds"
  }
}
```

#### 5. Output Synthesis
```json
{
  "confidence_assessment": {
    "overall_confidence": "Weighted combination of expert confidences",
    "uncertainty_sources": "Identified sources of epistemic uncertainty"
  },
  "weighted_combination": {
    "combination_formula": "∑(wi * ai) / ∑wi",
    "expert_weights": "Confidence-based weighting scheme"
  }
}
```

**Mathematical Foundation**:
- **Weighted Averages**: Confidence-weighted expert consensus
- **Uncertainty Propagation**: Error propagation through reasoning chain
- **Information Fusion**: Combining multiple information sources

#### 6. Performance Metrics
```json
{
  "latency_metrics": {
    "total_latency_seconds": "End-to-end processing time",
    "component_latencies": "Per-component timing breakdown"
  },
  "computational_metrics": {
    "memory_peak_mb": "Maximum memory usage",
    "token_consumption": "LLM token usage tracking"
  }
}
```

#### 7. Validation Metrics
```json
{
  "mathematical_validation": {
    "dimensional_analysis": "Unit consistency verification",
    "numerical_stability": "Condition number analysis"
  },
  "logical_validation": {
    "consistency_check": "Logical contradiction detection",
    "completeness_score": "Reasoning chain completeness"
  }
}
```

## 🔢 Mathematical Foundations

### Information Theory
- **Shannon Entropy**: H(X) = -Σ p(xi)log₂(p(xi))
- **Mutual Information**: I(X;Y) = H(X) - H(X|Y)  
- **Kolmogorov Complexity**: K(x) ≈ |compress(x)|

### Linear Algebra
- **Eigendecomposition**: A = QΛQ⁻¹
- **Singular Value Decomposition**: A = UΣVᵀ
- **Vector Space Transformations**: T: V → W

### Calculus & Optimization  
- **Critical Points**: ∇f(x) = 0
- **Lagrange Multipliers**: ∇f = λ∇g for constrained optimization
- **KKT Conditions**: Necessary conditions for constrained optimization

### Statistics & Probability
- **Bayes' Theorem**: P(H|E) = P(E|H)P(H)/P(E)
- **Confidence Intervals**: x̄ ± t(α/2) × s/√n
- **Hypothesis Testing**: p-values and significance levels

### Graph Theory
- **Directed Acyclic Graphs**: For causal relationships
- **Graph Metrics**: Density, clustering coefficient, path length
- **Network Analysis**: Centrality measures, community detection

## 🚀 Usage Examples

### Basic Usage
```python
from unified_reasoning_engine import unified_engine
from reasoning_logger import ReasoningSessionManager

# Run with comprehensive logging
result = unified_engine.reason("Solve x² + 5x + 6 = 0", enable_logging=True)

# Access session data
session_id = result['session_id']
print(f"Detailed log saved for session: {session_id}")
```

### Advanced Usage with Session Management
```python
with ReasoningSessionManager() as session:
    session_id = session.start_reasoning_session(
        "Complex optimization problem...",
        engine_version="1.0.0"
    )
    
    # Manual logging of specific steps
    session.log_step("input_analysis", analysis_data)
    session.log_step("expert_analysis", expert_data)
    
    # Session automatically saved on exit
```

## 📁 Log File Structure

Each reasoning session generates a JSON file with the following structure:

```
reasoning_logs/
├── reasoning_session_20240115_143022_a1b2c3d4.json
├── reasoning_session_20240115_143156_e5f6g7h8.json
└── unified_reasoning_schema.json (schema definition)
```

### Sample Log Entry
```json
{
  "session_metadata": {
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "timestamp": "2024-01-15T14:30:22.123Z",
    "engine_version": "1.0.0"
  },
  "input_analysis": {
    "structural_analysis": {
      "shannon_entropy": 4.23,
      "compression_ratio": 0.67,
      "token_count": 45
    }
  },
  "reasoning_process": {
    "expert_analyses": [
      {
        "expert_type": "linear_algebra",
        "confidence_score": 0.85,
        "mathematical_formulation": "Ax = b system solution"
      }
    ]
  }
}
```

## 🔍 Schema Validation

The schema includes comprehensive validation:

1. **Type Validation**: All fields have strict type requirements
2. **Range Validation**: Confidence scores ∈ [0,1], probabilities sum to 1
3. **Mathematical Consistency**: Dimensional analysis, unit checking
4. **Logical Consistency**: Contradiction detection between expert analyses

## 🎯 Key Benefits

1. **Complete Auditability**: Every reasoning step is logged and traceable
2. **Mathematical Rigor**: All computations have mathematical foundations
3. **Reproducibility**: Full environment and configuration capture
4. **Performance Analysis**: Detailed timing and resource usage
5. **Quality Assurance**: Multi-level validation and consistency checking
6. **Research Value**: Rich dataset for AI reasoning research

## 📚 References

- Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- Boyd, S. & Vandenberghe, L. (2004). "Convex Optimization"  
- Pearl, J. (2009). "Causality: Models, Reasoning and Inference"
- Russell, S. & Norvig, P. (2020). "Artificial Intelligence: A Modern Approach"

## 🔧 Implementation Notes

- **JSON Schema Version**: Draft-07 compliance
- **File Encoding**: UTF-8 for international character support
- **Precision**: Floating point numbers stored with full precision
- **Graph Serialization**: NetworkX graphs converted to edge lists
- **Memory Efficiency**: Large objects compressed when possible

---

*This schema represents the state-of-the-art in reasoning system logging, combining mathematical rigor with practical AI system design principles.* 