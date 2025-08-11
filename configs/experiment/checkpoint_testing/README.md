# 🧪 Checkpoint Testing Configuration

This directory contains specialized configurations for testing ConStruct metrics with real trained model checkpoints.

## 📁 Configuration Files

### **1. No Constraint Testing**
- **File**: `qm9_checkpoint_test_no_constraint.yaml`
- **Checkpoint**: `checkpoints/qm9_debug_no_constraint/last.ckpt`
- **Purpose**: Test metrics with unconstrained molecular generation
- **Expected**: Natural molecular distribution with various ring counts and lengths

### **2. Ring Count Constraint Testing**
- **File**: `qm9_checkpoint_test_ring_count_0.yaml`
- **Checkpoint**: `checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt`
- **Purpose**: Test metrics with ring count ≤ 0 constraint
- **Expected**: All generated molecules should have 0 rings (acyclic)

### **3. Ring Length Constraint Testing**
- **File**: `qm9_checkpoint_test_ring_length_3.yaml`
- **Checkpoint**: `checkpoints/qm9_debug_ring_length_at_most_3/last.ckpt`
- **Purpose**: Test metrics with ring length ≤ 3 constraint
- **Expected**: All rings should have ≤ 3 atoms

## 🚀 Usage

### **Individual Testing**

#### **No Constraint**
```bash
cd ConStruct
source activate construct-env

python main.py \
    +experiment=checkpoint_testing/qm9_checkpoint_test_no_constraint \
    general.resume=checkpoints/qm9_debug_no_constraint/last.ckpt \
    general.test_only=checkpoints/qm9_debug_no_constraint/last.ckpt
```

#### **Ring Count ≤ 0**
```bash
python main.py \
    +experiment=checkpoint_testing/qm9_checkpoint_test_ring_count_0 \
    general.resume=checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt \
    general.test_only=checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt
```

#### **Ring Length ≤ 3**
```bash
python main.py \
    +experiment=checkpoint_testing/qm9_checkpoint_test_ring_length_3 \
    general.resume=checkpoints/qm9_debug_ring_length_at_most_3/last.ckpt \
    general.test_only=checkpoints/qm9_debug_ring_length_at_most_3/last.ckpt
```

### **SLURM Batch Testing**

#### **Individual Jobs**
```bash
# Submit individual test jobs
sbatch ../ConStruct/slurm_jobs/checkpoint_testing/test_no_constraint.slurm
sbatch ../ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_0.slurm
sbatch ../ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_3.slurm
```

#### **All Tests Together**
```bash
# Submit comprehensive testing suite
sbatch ../ConStruct/slurm_jobs/checkpoint_testing/run_all_tests.slurm
```

## 📊 Expected Results

### **No Constraint Model**
- **Ring Count Distribution**: Natural distribution (0-5+ rings)
- **Ring Length Distribution**: Natural distribution (3-8+ atoms)
- **Metrics**: Standard molecular validity, uniqueness, novelty

### **Ring Count ≤ 0 Model**
- **Ring Count**: 100% should have 0 rings
- **Ring Count Satisfaction**: 100%
- **Metrics**: All molecules should be acyclic

### **Ring Length ≤ 3 Model**
- **Ring Length**: All rings ≤ 3 atoms
- **Ring Length Satisfaction**: 100%
- **Metrics**: Small ring constraint properly enforced

## 🎯 What This Tests

1. **✅ Checkpoint Loading**: All your checkpoints work
2. **✅ Model Creation**: Models initialize with correct dimensions
3. **✅ Constraint Enforcement**: Constraints are properly applied
4. **✅ Metric Calculation**: Your metrics work with real model outputs
5. **✅ Edge Case Handling**: Real data processing
6. **✅ Output Generation**: Sample generation and saving

## 🚀 Quick Start

1. **Test Individual Config**: Run one of the individual commands above
2. **Submit SLURM Job**: Use `sbatch` with any of the SLURM files
3. **Check Results**: Look in `outputs/checkpoint_testing/` for results
4. **Review Logs**: Check `logs/checkpoint_testing/` for detailed logs

This setup gives you comprehensive testing of your ConStruct metrics with real, trained models! 🎉
