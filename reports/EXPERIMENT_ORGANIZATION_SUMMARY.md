# Experiment Organization Summary

## Overview

This document summarizes the complete reorganization of experiment configurations and SLURM scripts for ring count and ring length constraints using both edge-deletion and edge-insertion mechanisms.

## 🎯 What Was Accomplished

### 1. **Organized Configs Folder Structure**
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
└── README.md                      # Comprehensive documentation
```

### 2. **Created All Reasonable Constraint Configurations**

#### Edge-Deletion (At Most) Configurations
- **Ring Count**: 0, 1, 2, 3, 4, 5 (based on QM9 distribution)
- **Ring Length**: 3, 4, 5, 6 (based on QM9 distribution)

#### Edge-Insertion (At Least) Configurations
- **Ring Count**: 1, 2, 3 (reasonable minimums)
- **Ring Length**: 4, 5, 6 (reasonable minimums)

### 3. **Organized SLURM Scripts**
```
slurm_scripts/
├── debug/                    # Debug-level SLURM scripts
│   ├── submit_ring_count_at_most_*_debug.sh
│   ├── submit_ring_length_at_most_*_debug.sh
│   ├── submit_ring_count_at_least_*_debug.sh
│   └── submit_ring_length_at_least_*_debug.sh
└── thesis/                   # Thesis-level SLURM scripts
    ├── submit_ring_count_at_most_*_thesis.sh
    ├── submit_ring_length_at_most_*_thesis.sh
    ├── submit_ring_count_at_least_*_thesis.sh
    └── submit_ring_length_at_least_*_thesis.sh
```

## 📊 Configuration Details

### Debug Configurations
- **Purpose**: Quick testing and validation
- **Training**: 300 epochs, smaller model (1 layer, 8 dimensions)
- **Sampling**: 500-1000 samples for validation
- **Resource Usage**: Lower computational requirements
- **Time Limit**: 12 hours
- **Memory**: 16GB
- **CPU**: 2 cores

### Thesis Configurations
- **Purpose**: Full-scale experiments and research
- **Training**: 1000 epochs, larger model (4 layers, 128 dimensions)
- **Sampling**: 1000-2000 samples for comprehensive analysis
- **Resource Usage**: Higher computational requirements
- **Time Limit**: 48 hours
- **Memory**: 32GB
- **CPU**: 4 cores

## 🔧 Constraint Mechanisms

### Edge-Deletion (At Most)
- **Transition**: `absorbing_edges`
- **Projectors**: `ring_count_at_most`, `ring_length_at_most`
- **Purpose**: Limit maximum ring count or ring length
- **Use Case**: Generate molecules with limited ring complexity

### Edge-Insertion (At Least)
- **Transition**: `edge_insertion`
- **Projectors**: `ring_count_at_least`, `ring_length_at_least`
- **Purpose**: Ensure minimum ring count or ring length
- **Use Case**: Generate molecules with guaranteed ring structures

## 📁 File Naming Convention

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

## 🚀 Usage Examples

### Running Debug Experiments
```bash
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
# Ring count at most 3 (thesis)
python ConStruct/main.py \
  --config-name experiment/thesis/edge_deletion/ring_count_at_most/qm9_thesis_ring_count_at_most_3.yaml \
  --config-path configs/

# Ring count at least 2 (thesis)
python ConStruct/main.py \
  --config-name experiment/thesis/edge_insertion/ring_count_at_least/qm9_thesis_ring_count_at_least_2.yaml \
  --config-path configs/
```

### Using SLURM Scripts
```bash
# Debug experiment
sbatch slurm_scripts/debug/submit_ring_count_at_most_2_debug.sh

# Thesis experiment
sbatch slurm_scripts/thesis/submit_ring_count_at_most_3_thesis.sh
```

## 📈 QM9 Dataset Considerations

### Ring Count Distribution
- **0 rings**: 13,387 molecules (10.23%)
- **1 ring**: 51,445 molecules (39.34%)
- **2 rings**: 41,360 molecules (31.62%)
- **3+ rings**: 24,636 molecules (18.81%)

### Ring Length Distribution
- **3-atom rings**: 14,415 molecules (11.02%)
- **4-atom rings**: 30,962 molecules (23.66%)
- **5-atom rings**: 48,892 molecules (37.38%)
- **6-atom rings**: 18,905 molecules (14.46%)
- **7+ atom rings**: 4,267 molecules (3.26%)

## 🧹 Cleanup Actions

### Removed Duplications
- ✅ Consolidated duplicate configuration files
- ✅ Organized by constraint type and experiment level
- ✅ Moved legacy configurations to `legacy/` folder
- ✅ Updated all SLURM script paths

### Created Documentation
- ✅ Comprehensive README for configs folder
- ✅ SLURM scripts documentation
- ✅ Usage examples and best practices
- ✅ Constraint value justifications

## 🎯 Benefits of New Organization

### 1. **Clear Structure**
- Easy to find configurations by constraint type
- Logical separation of debug vs thesis experiments
- No more random SLURM file creation

### 2. **Comprehensive Coverage**
- All reasonable constraint values covered
- Both edge-deletion and edge-insertion mechanisms
- Debug and thesis level experiments

### 3. **Maintainable**
- Consistent naming convention
- Organized file structure
- Clear documentation

### 4. **Scalable**
- Easy to add new constraint types
- Simple to extend with new values
- Modular organization

## 🔮 Future Extensions

### Potential Additions
- **Combined Constraints**: Ring count AND ring length constraints
- **Adaptive Constraints**: Dynamic constraint adjustment
- **Multi-Objective**: Balance constraint satisfaction with properties
- **Custom Constraints**: User-defined ring-based constraints

### Easy Integration
- New constraint types can follow the same pattern
- Additional constraint values can be added easily
- Documentation structure is extensible

## ✅ Summary

The experiment folder is now fully organized with:
- **36 configuration files** covering all reasonable constraint values
- **Organized structure** by constraint type and experiment level
- **Comprehensive SLURM scripts** for all configurations
- **Clear documentation** and usage examples
- **No duplications** or random file creation

This provides a solid foundation for systematic experimentation with ring constraints while maintaining clean organization and avoiding the creation of random SLURM files. 