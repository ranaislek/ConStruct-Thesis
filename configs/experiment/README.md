# Organized Experiment Configurations

## Overview

This directory contains all experiment configurations organized by constraint type and experiment level.

## Directory Structure

```
configs/experiment/
├── debug/                          # Debug-level experiments (quick testing)
│   ├── no_constraint/             # No constraint experiments
│   ├── edge_deletion/             # Edge-deletion constraints ("at most")
│   │   ├── ring_count_at_most/   # Ring count "at most" constraints
│   │   └── ring_length_at_most/  # Ring length "at most" constraints
│   └── edge_insertion/            # Edge-insertion constraints ("at least")
│       ├── ring_count_at_least/  # Ring count "at least" constraints
│       └── ring_length_at_least/ # Ring length "at least" constraints
├── thesis/                         # Thesis-level experiments (full-scale)
│   ├── no_constraint/             # No constraint experiments
│   ├── edge_deletion/             # Edge-deletion constraints ("at most")
│   │   ├── ring_count_at_most/   # Ring count "at most" constraints
│   │   └── ring_length_at_most/  # Ring length "at most" constraints
│   └── edge_insertion/            # Edge-insertion constraints ("at least")
│       ├── ring_count_at_least/  # Ring count "at least" constraints
│       └── ring_length_at_least/ # Ring length "at least" constraints
├── legacy/                         # Legacy constraint configurations
└── README.md                      # This file
```

## Constraint Types

### No Constraint
- **Purpose**: Baseline training without any constraints
- **Transition**: `absorbing_edges`
- **Projector**: `null` (no constraint)
- **Use Case**: Generate molecules without any structural constraints

### Edge-Deletion Constraints ("At Most")
- **Purpose**: Limit maximum ring count or ring length
- **Transition**: `absorbing_edges`
- **Projectors**: `ring_count_at_most`, `ring_length_at_most`
- **Use Case**: Generate molecules with limited ring complexity

### Edge-Insertion Constraints ("At Least")
- **Purpose**: Ensure minimum ring count or ring length
- **Transition**: `edge_insertion`
- **Projectors**: `ring_count_at_least`, `ring_length_at_least`
- **Use Case**: Generate molecules with guaranteed ring structures

## Experiment Levels

### Debug Level
- **Purpose**: Quick testing and validation
- **Training**: 300 epochs, smaller model (1 layer, 8 dimensions)
- **Sampling**: 500-1000 samples
- **Resource Usage**: Lower computational requirements

### Thesis Level
- **Purpose**: Full-scale experiments and research
- **Training**: 1000 epochs, larger model (4 layers, 128 dimensions)
- **Sampling**: 1000-2000 samples
- **Resource Usage**: Higher computational requirements

## Configuration Naming Convention

### No Constraint Configurations
- `qm9_debug_no_constraint.yaml` - Debug, no constraints
- `qm9_thesis_no_constraint.yaml` - Thesis, no constraints

### Edge-Deletion Configurations
- `qm9_debug_ring_count_at_most_{N}.yaml` - Debug, max N rings
- `qm9_thesis_ring_count_at_most_{N}.yaml` - Thesis, max N rings
- `qm9_debug_ring_length_at_most_{N}.yaml` - Debug, max N-ring length
- `qm9_thesis_ring_length_at_most_{N}.yaml` - Thesis, max N-ring length

### Edge-Insertion Configurations
- `qm9_debug_ring_count_at_least_{N}.yaml` - Debug, min N rings
- `qm9_thesis_ring_count_at_least_{N}.yaml` - Thesis, min N rings
- `qm9_debug_ring_length_at_least_{N}.yaml` - Debug, min N-ring length
- `qm9_thesis_ring_length_at_least_{N}.yaml` - Thesis, min N-ring length

## Usage Examples

### Running Debug Experiments
```bash
# No constraint (debug)
python ConStruct/main.py \
  --config-name experiment/debug/no_constraint/qm9_debug_no_constraint.yaml \
  --config-path configs/

# Ring count at most 2 (debug)
python ConStruct/main.py \
  --config-name experiment/debug/edge_deletion/ring_count_at_most/qm9_debug_ring_count_at_most_2.yaml \
  --config-path configs/

# Ring length at least 5 (debug)
python ConStruct/main.py \
  --config-name experiment/debug/edge_insertion/ring_length_at_least/qm9_debug_ring_length_at_least_5.yaml \
  --config-path configs/
```

### Running Thesis Experiments
```bash
# No constraint (thesis)
python ConStruct/main.py \
  --config-name experiment/thesis/no_constraint/qm9_thesis_no_constraint.yaml \
  --config-path configs/

# Ring count at most 3 (thesis)
python ConStruct/main.py \
  --config-name experiment/thesis/edge_deletion/ring_count_at_most/qm9_thesis_ring_count_at_most_3.yaml \
  --config-path configs/

# Ring count at least 2 (thesis)
python ConStruct/main.py \
  --config-name experiment/thesis/edge_insertion/ring_count_at_least/qm9_thesis_ring_count_at_least_2.yaml \
  --config-path configs/
```

## Constraint Values

### Ring Count Constraints
- **At Most**: 0, 1, 2, 3, 4, 5 (based on QM9 distribution)
- **At Least**: 1, 2, 3 (reasonable minimums)

### Ring Length Constraints
- **At Most**: 3, 4, 5, 6 (based on QM9 distribution)
- **At Least**: 4, 5, 6 (reasonable minimums)

## QM9 Dataset Considerations

- **Ring Count**: Most molecules have 0-2 rings, max found was 6
- **Ring Length**: Most rings are 4-6 atoms, violations start at 7+
- **Chemical Validity**: All constraints preserve chemical validity

## Legacy Configurations

The `legacy/` directory contains older constraint configurations:
- `planar.yaml` - Planarity constraint
- `tree.yaml` - Tree constraint
- `lobster.yaml` - Lobster constraint
- `low_tls.yaml` - Low TLS constraint
- `high_tls.yaml` - High TLS constraint

These are kept for reference but not actively used in new experiments.
