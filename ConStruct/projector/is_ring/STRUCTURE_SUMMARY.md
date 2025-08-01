# Fixed Edge-Insertion Structure Summary

## Overview

This document summarizes the **fixed structure** for edge-insertion constraints without duplications. All function names are now unique and clearly indicate their purpose.

## Directory Structure (Fixed)

```
ConStruct/projector/is_ring/
├── is_ring_count_at_most/           # Edge-deletion: "at most" constraints
│   ├── is_ring_count_at_most.py    # has_at_most_n_rings, ring_count_at_most_projector, count_rings_at_most
│   ├── check_dataset_ring_count_at_most.py # Dataset analysis for "at most"
│   └── qm9_ring_count_violations/  # Analysis results
│
├── is_ring_length_at_most/          # Edge-deletion: "at most" constraints  
│   ├── is_ring_length_at_most.py   # has_rings_of_length_at_most, ring_length_at_most_projector, get_max_ring_length_at_most
│   ├── check_dataset_ring_length_at_most.py # Dataset analysis for "at most"
│   └── qm9_ring_length_violations/ # Analysis results
│
├── is_ring_count_at_least/          # Edge-insertion: "at least" constraints
│   ├── is_ring_count_at_least.py   # has_at_least_n_rings, ring_count_at_least_projector, count_rings_at_least
│   ├── check_dataset_ring_count_at_least.py # Dataset analysis for "at least"
│   └── qm9_ring_count_at_least_violations/ # Analysis results
│
└── is_ring_length_at_least/         # Edge-insertion: "at least" constraints
    ├── is_ring_length_at_least.py   # has_rings_of_length_at_least, ring_length_at_least_projector, get_min_ring_length_at_least
    ├── check_dataset_ring_length_at_least.py # Dataset analysis for "at least"
    └── qm9_ring_length_at_least_violations/ # Analysis results
```

## Function Names (Fixed - No Duplications)

### Edge-Deletion Functions ("At Most")
- `has_at_most_n_rings(graph, n)` - Check if graph has at most n rings
- `ring_count_at_most_projector(graph, max_rings)` - Remove edges to satisfy max rings
- `count_rings_at_most(graph)` - Count total rings for "at most" constraints
- `has_rings_of_length_at_most(graph, max_length)` - Check if all rings ≤ max_length
- `ring_length_at_most_projector(graph, max_length)` - Remove edges to satisfy max ring length
- `get_max_ring_length_at_most(graph)` - Get length of largest ring for "at most" constraints

### Edge-Insertion Functions ("At Least")
- `has_at_least_n_rings(graph, n)` - Check if graph has at least n rings
- `ring_count_at_least_projector(graph, min_rings)` - Add edges to satisfy min rings
- `count_rings_at_least(graph)` - Count total rings for "at least" constraints
- `has_rings_of_length_at_least(graph, min_length)` - Check if any ring ≥ min_length
- `ring_length_at_least_projector(graph, min_length)` - Add edges to satisfy min ring length
- `get_min_ring_length_at_least(graph)` - Get length of smallest ring for "at least" constraints

## Key Fixes Applied

### 1. **Eliminated Function Name Duplications**
- **Before**: Both files had `count_rings()` function
- **After**: 
  - Edge-deletion: `count_rings_at_most()` 
  - Edge-insertion: `count_rings_at_least()`

### 2. **Made Function Names More Specific**
- **Before**: Both files had `get_min_ring_length()` function
- **After**:
  - Edge-deletion: `get_max_ring_length_at_most()` 
  - Edge-insertion: `get_min_ring_length_at_least()`

### 3. **Renamed Old Functions for Consistency**
- **Before**: `ring_count_projector()` and `ring_length_projector()`
- **After**: `ring_count_at_most_projector()` and `ring_length_at_most_projector()`

### 4. **Renamed Transition Class for Clarity**
- **Before**: `AbsorbingEdgesInsertionTransition` 
- **After**: `EdgeInsertionTransition`
- **Reason**: More descriptive and consistent with the mechanism

### 5. **Updated All Import Statements**
- Fixed `__init__.py` files to import correct function names
- Updated test files to use new function names
- Updated `projector_utils.py` to use new function names
- Updated all transition class references
- Ensured all references use the correct, non-duplicated names

### 6. **Maintained Clear Separation**
- Edge-deletion functions: Focus on "at most" constraints
- Edge-insertion functions: Focus on "at least" constraints
- No overlap in function names or purposes

## Configuration Examples (Updated)

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

## Testing (Updated)

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

## Benefits of the Fixed Structure

### 1. **No Function Name Conflicts**
- Each function has a unique, descriptive name
- Clear indication of whether it's for "at most" or "at least" constraints
- No import conflicts or naming collisions

### 2. **Clear Organization**
- Edge-deletion and edge-insertion mechanisms are completely separate
- Each directory has its own purpose and functions
- Easy to understand which functions to use for which constraints

### 3. **Maintainable Code**
- No duplications to maintain
- Clear separation of concerns
- Easy to extend with new constraint types

### 4. **Theoretical Soundness**
- Each mechanism preserves the theoretical soundness of the diffusion process
- Edge-deletion: Forward removes edges, reverse adds edges
- Edge-insertion: Forward adds edges, reverse removes edges

## Usage Guidelines

1. **For "At Most" Constraints**: Use edge-deletion functions and projectors
2. **For "At Least" Constraints**: Use edge-insertion functions and projectors
3. **Never Mix**: Don't use edge-deletion with "at least" constraints or vice versa
4. **Clear Naming**: Function names clearly indicate their purpose and constraint type

This fixed structure ensures clean, maintainable code without duplications while preserving the theoretical soundness of both edge-deletion and edge-insertion mechanisms. 