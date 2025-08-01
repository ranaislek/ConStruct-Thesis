# Edge-Insertion Structure Guide

## Overview

This document explains the organization of the **edge-insertion mechanism** (flipped edge-deletion) and the new "at least" constraints in the ConStruct codebase.

## Directory Structure

```
ConStruct/projector/is_ring/
├── is_ring_count/                    # Edge-deletion: "at most" constraints
│   ├── is_ring_count.py             # has_at_most_n_rings, ring_count_projector
│   ├── check_dataset_ring_count.py  # Dataset analysis for "at most"
│   └── qm9_ring_count_violations/   # Analysis results
│
├── is_ring_length/                   # Edge-deletion: "at most" constraints  
│   ├── is_ring_length.py            # has_rings_of_length_at_most, ring_length_projector
│   ├── check_dataset_ring_length.py # Dataset analysis for "at most"
│   └── qm9_ring_length_violations/  # Analysis results
│
├── is_ring_count_at_least/          # Edge-insertion: "at least" constraints
│   ├── is_ring_count_at_least.py    # has_at_least_n_rings, ring_count_at_least_projector
│   ├── check_dataset_ring_count_at_least.py # Dataset analysis for "at least"
│   └── qm9_ring_count_at_least_violations/ # Analysis results
│
└── is_ring_length_at_least/         # Edge-insertion: "at least" constraints
    ├── is_ring_length_at_least.py   # has_rings_of_length_at_least, ring_length_at_least_projector
    ├── check_dataset_ring_length_at_least.py # Dataset analysis for "at least"
    └── qm9_ring_length_at_least_violations/ # Analysis results
```

## Mechanism Comparison

### Edge-Deletion (Original)
- **Purpose**: "At most" constraints (e.g., max_rings ≤ 3)
- **Absorbing State**: no-edge (index 0)
- **Forward Diffusion**: Edges disappear toward no-edge state
- **Reverse Diffusion**: Edges are added back while respecting constraints
- **Natural Bias**: Toward acyclic graphs

### Edge-Insertion (New)
- **Purpose**: "At least" constraints (e.g., min_rings ≥ 2)
- **Absorbing State**: edge (index 1)
- **Forward Diffusion**: Edges appear toward edge state
- **Reverse Diffusion**: Edges are removed while preserving constraints
- **Natural Bias**: Toward connected graphs

## Implementation Files

### 1. Transition Mechanisms (`ConStruct/diffusion/noise_model.py`)

**Edge-Deletion Transition**:
```python
class AbsorbingEdgesTransition(MarginalTransition):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        # Edges absorb to no-edge state (index 0)
        self.E_marginals[0] = 1  # Edge-deletion: absorb to no-edge state
```

**Edge-Insertion Transition**:
```python
class EdgeInsertionTransition(MarginalTransition):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        # Edges absorb to edge state (index 1)
        self.E_marginals[1] = 1  # Edge-insertion: absorb to edge state
```

### 2. Projectors (`ConStruct/projector/projector_utils.py`)

**Edge-Deletion Projectors**:
```python
class RingCountAtMostProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        cycles = nx.cycle_basis(nx_graph)
        return len(cycles) <= self.max_rings  # At most N rings

class RingLengthAtMostProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        cycles = nx.cycle_basis(nx_graph)
        return all(len(cycle) <= self.max_ring_length for cycle in cycles)
```

**Edge-Insertion Projectors**:
```python
class RingCountAtLeastProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        cycles = nx.cycle_basis(nx_graph)
        return len(cycles) >= self.min_rings  # At least N rings

class RingLengthAtLeastProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        cycles = nx.cycle_basis(nx_graph)
        return any(len(cycle) >= self.min_ring_length for cycle in cycles)
```

### 3. Ring Constraint Utilities

**Edge-Deletion Utilities**:
- `is_ring_count/is_ring_count.py`: `has_at_most_n_rings`, `ring_count_projector`
- `is_ring_length/is_ring_length.py`: `has_rings_of_length_at_most`, `ring_length_projector`

**Edge-Insertion Utilities**:
- `is_ring_count_at_least/is_ring_count_at_least.py`: `has_at_least_n_rings`, `ring_count_at_least_projector`
- `is_ring_length_at_least/is_ring_length_at_least.py`: `has_rings_of_length_at_least`, `ring_length_at_least_projector`

## Configuration Examples

### Edge-Insertion with Ring Count "At Least"
```yaml
# configs/experiment/edge_insertion_ring_count_at_least.yaml
model:
  transition: edge_insertion  # Edge-insertion transition
  rev_proj: ring_count_at_least  # At least N rings projector
  min_rings: 2  # At least 2 rings
```

### Edge-Insertion with Ring Length "At Least"
```yaml
# configs/experiment/edge_insertion_ring_length_at_least.yaml
model:
  transition: edge_insertion  # Edge-insertion transition
  rev_proj: ring_length_at_least  # At least N ring length projector
  min_ring_length: 5  # At least 5-ring length
```

## Testing

### Dataset Analysis
```bash
# Analyze ring count "at least" distribution
python ConStruct/projector/is_ring/is_ring_count_at_least/check_dataset_ring_count_at_least.py

# Analyze ring length "at least" distribution  
python ConStruct/projector/is_ring/is_ring_length_at_least/check_dataset_ring_length_at_least.py
```

### Constraint Testing
```bash
# Test ring count "at least" constraints
python ConStruct/projector/is_ring/is_ring_count_at_least/check_dataset_ring_count_at_least.py

# Test ring length "at least" constraints
python ConStruct/projector/is_ring/is_ring_length_at_least/check_dataset_ring_length_at_least.py
```

## Key Principles

### 1. **Mechanism Separation**
- **NEVER** mix edge-deletion and edge-insertion in the same experiment
- Edge-deletion: Use with "at most" constraints
- Edge-insertion: Use with "at least" constraints

### 2. **Theoretical Soundness**
- Each mechanism preserves the theoretical soundness of the diffusion process
- Edge-deletion: Forward removes edges, reverse adds edges
- Edge-insertion: Forward adds edges, reverse removes edges

### 3. **Constraint Compatibility**
- Edge-deletion transitions + "at most" projectors = ✅ Compatible
- Edge-insertion transitions + "at least" projectors = ✅ Compatible
- Edge-deletion transitions + "at least" projectors = ❌ Incompatible
- Edge-insertion transitions + "at most" projectors = ❌ Incompatible

### 4. **Organization Pattern**
- Follow the existing pattern: `is_ring_[constraint_type]/`
- Separate "at most" and "at least" versions clearly
- Use descriptive names that indicate the constraint type
- Include dataset analysis and testing for each constraint type

## Usage Workflow

1. **Choose Mechanism**: Decide between edge-deletion ("at most") or edge-insertion ("at least")
2. **Select Constraints**: Choose appropriate ring count or ring length constraints
3. **Configure**: Use the corresponding configuration file
4. **Test**: Run the appropriate dataset analysis and constraint tests
5. **Validate**: Ensure transition-projector compatibility

This structure ensures clean separation of concerns and maintains the theoretical soundness of the diffusion process while providing clear organization for both edge-deletion and edge-insertion mechanisms. 