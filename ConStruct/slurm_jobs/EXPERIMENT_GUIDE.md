# ConStruct Experiment Guide

## Overview

This guide explains how to use the organized experiment structure for testing different constraint types in the ConStruct framework.

## 🏗️ **Experiment Structure**

### **Directory Organization**

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
└── thesis/                         # Thesis-level experiments (full-scale)
    ├── no_constraint/             # No constraint experiments
    ├── edge_deletion/             # Edge-deletion constraints ("at most")
    │   ├── ring_count_at_most/   # Ring count "at most" constraints
    │   └── ring_length_at_most/  # Ring length "at most" constraints
    └── edge_insertion/            # Edge-insertion constraints ("at least")
        ├── ring_count_at_least/  # Ring count "at least" constraints
        └── ring_length_at_least/ # Ring length "at least" constraints

ConStruct/slurm_jobs/
├── debug/                          # Debug-level SLURM scripts
│   ├── no_constraint/             # No constraint scripts
│   ├── edge_deletion/             # Edge-deletion scripts
│   └── edge_insertion/            # Edge-insertion scripts
└── thesis/                         # Thesis-level SLURM scripts
    ├── no_constraint/             # No constraint scripts
    ├── edge_deletion/             # Edge-deletion scripts
    └── edge_insertion/            # Edge-insertion scripts
```

## 🔬 **Constraint Types**

### **1. Edge-Deletion Constraints ("At Most")**

**Purpose**: Limit maximum ring count or ring length

**Configuration**:
```yaml
model:
  transition: absorbing_edges        # Edge-deletion transition
  rev_proj: ring_count_at_most      # Ring count "at most" projector
  max_rings: 3                      # Maximum 3 rings allowed
  # OR
  rev_proj: ring_length_at_most     # Ring length "at most" projector
  max_ring_length: 5                # Maximum 5-ring length
```

**Use Cases**:
- Generate molecules with limited ring complexity
- Create acyclic or simple ring structures
- Control molecular complexity

**Available Values**:
- **Ring Count**: 0, 1, 2, 3, 4, 5 (based on QM9 distribution)
- **Ring Length**: 3, 4, 5, 6 (based on QM9 distribution)

### **2. Edge-Insertion Constraints ("At Least")**

**Purpose**: Ensure minimum ring count or ring length

**Configuration**:
```yaml
model:
  transition: edge_insertion         # Edge-insertion transition
  rev_proj: ring_count_at_least     # Ring count "at least" projector
  min_rings: 2                      # At least 2 rings required
  # OR
  rev_proj: ring_length_at_least    # Ring length "at least" projector
  min_ring_length: 5                # At least 5-ring length required
```

**Use Cases**:
- Generate molecules with guaranteed ring structures
- Create aromatic compounds
- Ensure specific ring topologies

**Available Values**:
- **Ring Count**: 1, 2, 3 (reasonable minimums)
- **Ring Length**: 4, 5, 6 (reasonable minimums)

### **3. No Constraint (Baseline)**

**Purpose**: Baseline training without any constraints

**Configuration**:
```yaml
model:
  transition: absorbing_edges        # Standard edge-deletion transition
  rev_proj: null                    # No constraint projector
```

**Use Cases**:
- Baseline comparison
- Unconstrained molecular generation
- Control experiments

## 🧪 **Running Experiments**

### **Method 1: Direct Python Execution**

#### **Debug Experiments (Quick Testing)**
```bash
# No constraint baseline
python ConStruct/main.py \
  --config-name experiment/debug/no_constraint/qm9_debug_no_constraint.yaml \
  --config-path configs/

# Ring count at most 2
python ConStruct/main.py \
  --config-name experiment/debug/edge_deletion/ring_count_at_most/qm9_debug_ring_count_at_most_2.yaml \
  --config-path configs/

# Ring count at least 1
python ConStruct/main.py \
  --config-name experiment/debug/edge_insertion/ring_count_at_least/qm9_debug_ring_count_at_least_1.yaml \
  --config-path configs/
```

#### **Thesis Experiments (Full-Scale)**
```bash
# Ring count at most 3
python ConStruct/main.py \
  --config-name experiment/thesis/edge_deletion/ring_count_at_most/qm9_thesis_ring_count_at_most_3.yaml \
  --config-path configs/

# Ring length at least 5
python ConStruct/main.py \
  --config-name experiment/thesis/edge_insertion/ring_length_at_least/qm9_thesis_ring_length_at_least_5.yaml \
  --config-path configs/
```

### **Method 2: SLURM Job Submission**

#### **Debug Experiments**
```bash
# No constraint baseline
sbatch ConStruct/slurm_jobs/debug/no_constraint/submit_no_constraint_debug.sh

# Ring count at most 2
sbatch ConStruct/slurm_jobs/debug/edge_deletion/ring_count_at_most/submit_ring_count_at_most_2_debug.sh

# Ring count at least 1
sbatch ConStruct/slurm_jobs/debug/edge_insertion/ring_count_at_least/submit_ring_count_at_least_1_debug.sh
```

#### **Thesis Experiments**
```bash
# Ring count at most 3
sbatch ConStruct/slurm_jobs/thesis/edge_deletion/ring_count_at_most/submit_ring_count_at_most_3_thesis.sh

# Ring length at least 5
sbatch ConStruct/slurm_jobs/thesis/edge_insertion/ring_length_at_least/submit_ring_length_at_least_5_thesis.sh
```

## 📊 **Resource Requirements**

### **Debug Level**
- **Time**: 12 hours
- **Memory**: 16GB
- **CPU**: 2 cores
- **GPU**: 1 GPU
- **Training**: 300 epochs, smaller model (1 layer, 8 dimensions)
- **Sampling**: 500-1000 samples

### **Thesis Level**
- **Time**: 48 hours
- **Memory**: 32GB
- **CPU**: 4 cores
- **GPU**: 1 GPU
- **Training**: 1000 epochs, larger model (4 layers, 128 dimensions)
- **Sampling**: 1000-2000 samples

## 🔍 **Monitoring and Validation**

### **Log Files**
All experiments create log files in the `logs/` directory:
- `logs/{job_name}_{job_id}.out` - Standard output
- `logs/{job_name}_{job_id}.err` - Error output

### **Constraint Validation**
The framework automatically validates constraint satisfaction during sampling:
```
[Constraint Check] 85/100 molecules satisfy ring_count_at_most ≤ 3
[Constraint Check] 92/100 molecules satisfy ring_length_at_least ≥ 5
```

### **Wandb Integration**
All experiments log to Weights & Biases for monitoring:
- Training metrics
- Sampling metrics
- Constraint satisfaction rates
- Generated molecule examples

## ⚠️ **Important Notes**

### **1. Transition-Projector Compatibility**
The framework automatically validates that you don't mix incompatible mechanisms:

```yaml
# ✅ COMPATIBLE
transition: absorbing_edges
rev_proj: ring_count_at_most

# ❌ INCOMPATIBLE (will raise error)
transition: absorbing_edges
rev_proj: ring_count_at_least
```

### **2. Constraint Values**
Choose constraint values based on QM9 dataset statistics:
- **Ring Count**: Most molecules have 0-2 rings, max found was 6
- **Ring Length**: Most rings are 4-6 atoms, violations start at 7+

### **3. Best Practices**
1. **Start with Debug**: Use debug experiments for quick validation
2. **Monitor Resources**: Check resource usage during execution
3. **Check Logs**: Monitor log files for errors or issues
4. **Use Appropriate Level**: Choose debug vs thesis based on your needs

## 🚀 **Quick Start Examples**

### **Example 1: Test Edge-Insertion Constraints**
```bash
# Quick test with minimum ring count
sbatch ConStruct/slurm_jobs/debug/edge_insertion/ring_count_at_least/submit_ring_count_at_least_1_debug.sh

# Full experiment with minimum ring length
sbatch ConStruct/slurm_jobs/thesis/edge_insertion/ring_length_at_least/submit_ring_length_at_least_5_thesis.sh
```

### **Example 2: Test Edge-Deletion Constraints**
```bash
# Quick test with maximum ring count
sbatch ConStruct/slurm_jobs/debug/edge_deletion/ring_count_at_most/submit_ring_count_at_most_2_debug.sh

# Full experiment with maximum ring length
sbatch ConStruct/slurm_jobs/thesis/edge_deletion/ring_length_at_most/submit_ring_length_at_most_4_thesis.sh
```

### **Example 3: Baseline Comparison**
```bash
# No constraint baseline
sbatch ConStruct/slurm_jobs/debug/no_constraint/submit_no_constraint_debug.sh
```

## 📚 **Additional Resources**

- **Configs Documentation**: `configs/experiment/README.md`
- **SLURM Documentation**: `ConStruct/slurm_jobs/README.md`
- **Legacy Configs**: `configs/experiment/legacy/README.md`
- **Constraint Implementation**: `ConStruct/projector/projector_utils.py`
- **Sampling Metrics**: `ConStruct/metrics/sampling_molecular_metrics.py`

## 🆘 **Troubleshooting**

### **Common Issues**

1. **Constraint Validation Errors**
   - Check that transition and projector are compatible
   - Verify constraint values are reasonable for QM9 dataset

2. **Resource Issues**
   - Debug experiments use fewer resources than thesis
   - Monitor GPU memory usage during training

3. **Path Errors**
   - Ensure you're running from the correct directory
   - Check that config paths in SLURM scripts are correct

### **Getting Help**
- Check log files for detailed error messages
- Monitor Wandb for training progress
- Verify constraint satisfaction rates in logs 