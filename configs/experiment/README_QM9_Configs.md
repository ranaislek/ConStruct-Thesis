# QM9 Configuration Files Guide

This document explains all the QM9-related configuration files and their purposes.

## **File Structure**

### **Dataset Config**
- `configs/dataset/qm9.yaml` - Dataset-specific settings for QM9

### **Experiment Configs**
- `configs/experiment/qm9.yaml` - **Main QM9 config** (recommended for most experiments)
- `configs/experiment/qm9_debug.yaml` - **Debug version** for quick testing
- `configs/experiment/qm9_no_h.yaml` - QM9 without hydrogen atoms
- `configs/experiment/qm9_with_h.yaml` - QM9 with hydrogen atoms

## **Configuration Details**

### **1. Main QM9 Config (`qm9.yaml`)**
**Purpose**: Standard QM9 experiment with constraint support
**Key Settings**:
- `remove_h: true` (no hydrogen atoms)
- `n_epochs: 1000`
- `batch_size: 64`
- `diffusion_steps: 500`
- `samples_to_generate: 1000`
- **Constraint support**: ring_count, ring_length

**Usage**:
```bash
python ConStruct/main.py +experiment=qm9 model.rev_proj=ring_count model.max_rings=0
```

### **2. Debug QM9 Config (`qm9_debug.yaml`)**
**Purpose**: Quick testing and debugging
**Key Settings**:
- `n_epochs: 1`
- `batch_size: 4`
- `diffusion_steps: 10`
- `samples_to_generate: 5`
- **Fast execution** for testing pipeline

**Usage**:
```bash
python ConStruct/main.py +experiment=qm9_debug model.rev_proj=ring_count model.max_rings=0
```

### **3. QM9 No Hydrogen (`qm9_no_h.yaml`)**
**Purpose**: QM9 experiment without hydrogen atoms (same as main config)
**Key Settings**:
- `remove_h: true`
- Same settings as main qm9.yaml
- **Use this if you want to be explicit about no hydrogen**

### **4. QM9 With Hydrogen (`qm9_with_h.yaml`)**
**Purpose**: QM9 experiment including hydrogen atoms
**Key Settings**:
- `remove_h: false`
- Same settings as main qm9.yaml
- **Use this if you want to include hydrogen atoms**

## **Constraint Settings**

All configs support these constraints:

### **Ring Count Constraint**
```yaml
model:
    rev_proj: ring_count
    max_rings: 0  # Maximum number of rings allowed
```

### **Ring Length Constraint**
```yaml
model:
    rev_proj: ring_length
    max_ring_length: 6  # Maximum ring size allowed
```

## **Recommended Usage**

### **For Real Experiments**
```bash
# Ring count constraint
python ConStruct/main.py +experiment=qm9 model.rev_proj=ring_count model.max_rings=0

# Ring length constraint  
python ConStruct/main.py +experiment=qm9 model.rev_proj=ring_length model.max_ring_length=6
```

### **For Quick Testing**
```bash
# Debug version
python ConStruct/main.py +experiment=qm9_debug model.rev_proj=ring_count model.max_rings=0
```

### **For Hydrogen Experiments**
```bash
# With hydrogen atoms
python ConStruct/main.py +experiment=qm9_with_h model.rev_proj=ring_count model.max_rings=0
```

## **Key Differences**

| Config | Hydrogen | Epochs | Batch Size | Purpose |
|--------|----------|--------|------------|---------|
| `qm9.yaml` | No | 1000 | 64 | **Main experiments** |
| `qm9_debug.yaml` | No | 1 | 4 | **Quick testing** |
| `qm9_no_h.yaml` | No | 1000 | 64 | **Explicit no-H** |
| `qm9_with_h.yaml` | Yes | 1000 | 64 | **With hydrogen** |

## **Notes**

- All configs now have **consistent settings** for training and sampling
- **Constraint checking** is integrated into all configs
- **Debug config** uses minimal settings for fast testing
- **Main config** (`qm9.yaml`) is recommended for most experiments 