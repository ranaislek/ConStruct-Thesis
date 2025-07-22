# ConStruct Transition Mechanism Architecture

## Overview

ConStruct supports multiple transition mechanisms for molecular graph diffusion, each designed for different constraint types. This document explains the architectural separation and proper usage of each mechanism.

## Transition Mechanisms

### 1. Edge-Deletion Transitions (`absorbing_edges`)

**Purpose**: Handle "at most" constraints (e.g., max_rings ≤ 3)

**Mathematical Properties**:
- **Absorbing State**: no-edge (index 0)
- **Forward Diffusion**: Edges progressively disappear toward no-edge state
- **Reverse Diffusion**: Edges are added back while respecting constraints
- **Natural Bias**: Toward acyclic graphs

**Implementation**:
```python
class AbsorbingEdgesTransition(MarginalTransition):
    """Edge-Deletion Transition: Edges absorb to no-edge state (index 0)."""
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        # Edges absorb to no-edge state (index 0)
        self.E_marginals[0] = 1  # Edge-deletion: absorb to no-edge state
```

**Compatible Projectors**:
- `RingCountAtMostProjector` - Ensures at most N rings
- `RingLengthAtMostProjector` - Ensures max ring length

**Configuration**:
```yaml
model:
  transition: absorbing_edges
  rev_proj: ring_count_at_most
  max_rings: 3
```

### 2. Edge-Insertion Transitions (`edge_insertion`)

**Purpose**: Handle "at least" constraints (e.g., min_rings ≥ 2)

**Mathematical Properties**:
- **Absorbing State**: edge (index 1)
- **Forward Diffusion**: Edges progressively appear toward edge state
- **Reverse Diffusion**: Edges are removed while preserving constraints
- **Natural Bias**: Toward connected graphs

**Implementation**:
```python
class EdgeInsertionTransition(MarginalTransition):
    """Edge-Insertion Transition: Edges absorb to edge state (index 1)."""
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        # Edges absorb to edge state (index 1)
        self.E_marginals[1] = 1  # Edge-insertion: absorb to edge state
```

**Compatible Projectors**:
- `RingCountAtLeastProjector` - Ensures at least N rings
- `RingLengthAtLeastProjector` - Ensures min ring length

**Configuration**:
```yaml
model:
  transition: absorbing_edges_insertion
  rev_proj: ring_count_at_least
  min_rings: 2
```

### 3. Marginal Transitions (`marginal`)

**Purpose**: Use dataset marginals without artificial bias

**Mathematical Properties**:
- **No Absorbing State**: Uses natural dataset distribution
- **Forward Diffusion**: Natural noise addition
- **Reverse Diffusion**: Natural denoising
- **Natural Bias**: None (follows dataset)

**Compatible Projectors**: Any projector (no artificial bias)

**Configuration**:
```yaml
model:
  transition: marginal
  rev_proj: ring_count  # Can use any projector
  max_rings: 3
```

## Critical Architectural Rules

### 1. Mechanism Separation

**NEVER mix edge-deletion and edge-insertion mechanisms in the same experiment:**

```python
# ❌ INCOMPATIBLE - Will raise validation error
model:
  transition: absorbing_edges  # Edge-deletion
  rev_proj: ring_count_at_least  # Edge-insertion projector

# ✅ COMPATIBLE
model:
  transition: absorbing_edges  # Edge-deletion
  rev_proj: ring_count_at_most  # Edge-deletion projector
```

### 2. Transition-Projector Compatibility

The framework automatically validates combinations:

```python
# Edge-deletion transitions
edge_deletion_transitions = ["absorbing_edges"]
at_most_projectors = ["ring_count", "ring_count_at_most", "ring_length", "ring_length_at_most"]

# Edge-insertion transitions  
        edge_insertion_transitions = ["edge_insertion"]
at_least_projectors = ["ring_count_at_least", "ring_length_at_least", "ring_count_insertion"]

# Marginal transitions can use any projector
marginal_transitions = ["marginal", "uniform", "absorbing"]
```

### 3. Theoretical Soundness

Each mechanism preserves the theoretical soundness of the diffusion process:

- **Edge-Deletion**: Forward process removes edges, reverse process adds edges
- **Edge-Insertion**: Forward process adds edges, reverse process removes edges
- **Marginal**: Natural diffusion without artificial bias

## Implementation Details

### Transition Matrix Differences

**Edge-Deletion Transition Matrix**:
```python
# Standard absorbing transition: edges absorb to no-edge state (index 0)
q_e = be_abs * Pe + (1 - be_abs) * torch.eye(self.E_classes, device=dev).unsqueeze(0)
```

**Edge-Insertion Transition Matrix**:
```python
# Custom transition matrix: edges absorb to edge state (index 1)
# State 0 (no edge): can transition to any edge type with probability be_abs
# States 1-4 (edge types): absorbing (stay as edge)
q_e = torch.zeros(batch_size, self.E_classes, self.E_classes, device=dev)
q_e[:, 0, 0] = 1 - be_abs.squeeze()  # 0->0: stay no edge
q_e[:, 0, 1:] = be_abs.squeeze().unsqueeze(-1) / (self.E_classes - 1)  # 0->1,2,3,4: become edge
for i in range(1, self.E_classes):
    q_e[:, i, i] = 1  # i->i: stay as edge type i (absorbing)
```

### Projector Differences

**Edge-Deletion Projector**:
```python
def valid_graph_fn(self, nx_graph):
    cycles = nx.cycle_basis(nx_graph)
    return len(cycles) <= self.max_rings  # At most N rings
```

**Edge-Insertion Projector**:
```python
def valid_graph_fn(self, nx_graph):
    cycles = nx.cycle_basis(nx_graph)
    return len(cycles) >= self.min_rings  # At least N rings
```

## Usage Examples

### Example 1: Generate molecules with at most 2 rings
```yaml
model:
  transition: absorbing_edges  # Edge-deletion
  rev_proj: ring_count_at_most
  max_rings: 2
```

### Example 2: Generate molecules with at least 1 ring
```yaml
model:
  transition: edge_insertion  # Edge-insertion
  rev_proj: ring_count_at_least
  min_rings: 1
```

### Example 3: Generate molecules with natural ring distribution
```yaml
model:
  transition: marginal  # No artificial bias
  rev_proj: ring_count
  max_rings: 3
```

## Validation and Error Handling

The framework includes automatic validation:

```python
def _validate_transition_projector_compatibility(self, cfg):
    """Validate that transition and projector combinations are compatible."""
    # Checks for incompatible combinations and raises descriptive errors
```

**Example Error Messages**:
```
INCOMPATIBLE: Edge-deletion transition 'absorbing_edges' cannot be used 
with 'at least' projector 'ring_count_at_least'. Use 'at most' projectors instead.
```

## Migration Guide

### From Old Configurations

**Old (deprecated)**:
```yaml
model:
  transition: absorbing_edges
  rev_proj: ring_count  # Ambiguous
```

**New (explicit)**:
```yaml
model:
  transition: absorbing_edges
  rev_proj: ring_count_at_most  # Explicit "at most"
```

### Backward Compatibility

Legacy names are still supported:
- `edge_insertion` → `edge_insertion` (no change needed)
- `ring_count` → `ring_count_at_most`
- `ring_count_insertion` → `ring_count_at_least`

## Best Practices

1. **Choose the right mechanism for your constraint type**:
   - "At most" constraints → Edge-deletion
   - "At least" constraints → Edge-insertion
   - No artificial bias → Marginal

2. **Use explicit projector names**:
   - `ring_count_at_most` instead of `ring_count`
   - `ring_count_at_least` for edge-insertion

3. **Validate your configuration**:
   - The framework will catch incompatible combinations
   - Check error messages for guidance

4. **Document your choice**:
   - Add comments explaining why you chose a mechanism
   - Reference this document for architectural decisions

## Future Extensions

The modular architecture allows for easy extension:

1. **New Transition Types**: Add new classes inheriting from `NoiseModel`
2. **New Projectors**: Add new classes inheriting from `AbstractProjector`
3. **New Constraints**: Implement new constraint types following the pattern

All extensions should respect the mechanism separation principle to preserve theoretical soundness. 