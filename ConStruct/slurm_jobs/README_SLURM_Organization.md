# SLURM Jobs Organization

This document explains the organized structure of SLURM job files for the ConStruct project.

## **Folder Structure**

```
ConStruct/slurm_jobs/
├── templates/           # Template files for easy modification
├── experiments/         # Real experiment files (24h, full settings)
├── debug/              # Quick debug tests (10min, minimal settings)
└── tests/              # Validation and testing files
```

## **File Categories**

### **1. Templates (`templates/`)**
**Purpose**: Base templates for creating new SLURM jobs
- `template_real_experiment.slurm` - Template for real experiments
- `template_debug_test.slurm` - Template for debug tests

**Usage**: Copy and modify for new experiments

### **2. Real Experiments (`experiments/`)**
**Purpose**: Full experiments with complete settings
- `qm9_ring_count_real.slurm` - Ring count constraint (max 0 rings)
- `qm9_ring_length_real.slurm` - Ring length constraint (max 6 atoms/ring)

**Settings**:
- 24h time limit
- 4 CPUs, 20GB RAM, 1 GPU
- 1000 epochs, batch_size=64, 1000 samples
- Uses `qm9.yaml` config (no overrides needed)

### **3. Debug Tests (`debug/`)**
**Purpose**: Quick testing and validation
- `qm9_ring_count_debug.slurm` - Ring count constraint test
- `qm9_ring_length_debug.slurm` - Ring length constraint test

**Settings**:
- 10min time limit
- 2 CPUs, 4GB RAM, 1 GPU
- 1 epoch, batch_size=4, 5 samples
- Uses `qm9_debug.yaml` config

### **4. Validation Tests (`tests/`)**
**Purpose**: Comprehensive testing and validation
- `qm9_constraint_validation.slurm` - Tests both constraints

**Settings**:
- 15min time limit
- Tests multiple constraints in sequence

## **Quick Start Guide**

### **For Real Experiments**
```bash
# Ring count constraint (no rings allowed)
sbatch ConStruct/slurm_jobs/experiments/qm9_ring_count_real.slurm

# Ring length constraint (max 6 atoms per ring)
sbatch ConStruct/slurm_jobs/experiments/qm9_ring_length_real.slurm
```

### **For Quick Testing**
```bash
# Debug ring count constraint
sbatch ConStruct/slurm_jobs/debug/qm9_ring_count_debug.slurm

# Debug ring length constraint
sbatch ConStruct/slurm_jobs/debug/qm9_ring_length_debug.slurm
```

### **For Constraint Validation**
```bash
# Comprehensive constraint test
sbatch ConStruct/slurm_jobs/tests/qm9_constraint_validation.slurm
```

## **Creating New Experiments**

### **Using Templates**
1. Copy a template:
   ```bash
   cp ConStruct/slurm_jobs/templates/template_real_experiment.slurm \
      ConStruct/slurm_jobs/experiments/qm9_my_new_experiment.slurm
   ```

2. Modify the file:
   - Change `EXPERIMENT_NAME` to your experiment name
   - Change `CONSTRAINT_TYPE` to your constraint (e.g., `ring_count`)
   - Change `CONSTRAINT_PARAM` to your parameter (e.g., `model.max_rings`)
   - Change `CONSTRAINT_VALUE` to your value (e.g., `1`)

### **Example: Ring Count with 1 Ring**
```bash
# Copy template
cp ConStruct/slurm_jobs/templates/template_real_experiment.slurm \
   ConStruct/slurm_jobs/experiments/qm9_ring_count_1_real.slurm

# Edit the file to change:
# - job-name=qm9_ring_count_1_real
# - model.rev_proj=ring_count
# - model.max_rings=1
```

## **Key Features**

### **✅ Benefits of This Organization**
- **Clear separation** between debug, test, and real experiments
- **Consistent naming** convention
- **Template-based** creation for easy modification
- **No config overrides** needed (uses updated YAML configs)
- **Constraint checking** integrated in all experiments
- **Proper resource allocation** for each type

### **📊 Resource Allocation**
| Type | Time | CPUs | RAM | GPU | Purpose |
|------|------|------|-----|-----|---------|
| Debug | 10min | 2 | 4GB | 1 | Quick testing |
| Test | 15min | 2 | 4GB | 1 | Validation |
| Real | 24h | 4 | 20GB | 1 | Full experiments |

### **🔧 Configuration Usage**
- **Real experiments**: Use `qm9.yaml` (1000 epochs, full settings)
- **Debug tests**: Use `qm9_debug.yaml` (1 epoch, minimal settings)
- **No overrides**: All settings come from config files

## **Monitoring and Logs**

### **Log File Locations**
- Real experiments: `ConStruct/logs/qm9_EXPERIMENT_NAME_%j.out`
- Debug tests: `ConStruct/logs/debug_qm9_CONSTRAINT_NAME_%j.out`
- Validation tests: `ConStruct/logs/qm9_constraint_validation_%j.out`

### **Checking Job Status**
```bash
# Check running jobs
squeue -u rislek

# Monitor specific job
tail -f ConStruct/logs/qm9_ring_count_real_902988.out
```

## **Notes**

- All SLURM files use the **updated config system** (no parameter overrides)
- **Constraint checking** is automatically included in all experiments
- **Templates** make it easy to create new experiments
- **Debug tests** are perfect for quick validation
- **Real experiments** are ready for overnight runs 