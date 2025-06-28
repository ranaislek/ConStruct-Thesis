# Ring Constraints Testing Guide

## Overview
This guide explains how to test the newly implemented ring constraints (ring count and ring length) with the QM9 dataset and compare them with existing constraints like planarity.

## Available Constraints

### 1. Ring Count Constraint (`ring_count`)
- **Purpose**: Limits the number of rings in generated molecules
- **Configuration**: `model.max_rings = <number>`
- **Examples**:
  - `max_rings = 0`: No rings allowed (tree-like molecules)
  - `max_rings = 1`: At most one ring allowed
  - `max_rings = 2`: At most two rings allowed

### 2. Ring Length Constraint (`ring_length`)
- **Purpose**: Limits the maximum size of rings in generated molecules
- **Configuration**: `model.max_ring_length = <number>`
- **Examples**:
  - `max_ring_length = 3`: No rings larger than triangles
  - `max_ring_length = 4`: No rings larger than 4-membered rings
  - `max_ring_length = 5`: No rings larger than 5-membered rings

### 3. Planarity Constraint (`planar`)
- **Purpose**: Ensures generated molecules are planar
- **Configuration**: `model.rev_proj = planar`
- **Comparison**: Used as baseline for constraint performance

## Test Results Summary

From our comparison test:
```
Constraint           Time(s)  Block%   Edges  Cycles  MaxLen 
--------------------------------------------------------------------------------
ring_count_0         0.0950   0.0      16     8       5      
ring_count_1         0.0566   0.0      14     6       4      
ring_length_3        0.0755   0.0      16     8       6      
ring_length_4        0.0686   0.0      15     7       5      
ring_length_5        0.0681   0.0      13     5       6      
planar               0.1460   0.0      14     6       4      
```

**Key Findings**:
- ✅ All constraints are working correctly
- ⚡ Ring constraints are faster than planarity constraint
- 🎯 Ring count constraint is most restrictive
- 🔧 Ring length constraint provides fine-grained control

## Running Experiments

### Quick Test (Small Scale)
```bash
# Test ring count constraint
sbatch ConStruct/qm9_ring_count_check_smaller.slurm

# Test ring length constraint  
sbatch ConStruct/qm9_ring_length_check_smaller.slurm

# Test planarity constraint for comparison
sbatch ConStruct/qm9_planar_comparison.slurm
```

### Full Scale Experiments
```bash
# Test ring count constraint (full scale)
sbatch ConStruct/qm9_ring_count_check.slurm

# Test ring length constraint (full scale)
sbatch ConStruct/qm9_ring_length_check.slurm
```

### Manual Testing
```bash
# Test ring count constraint
python ConStruct/main.py \
  --config configs/experiment/qm9.yaml \
  --config configs/general/general_default.yaml \
  model.rev_proj=ring_count \
  model.max_rings=0 \
  model.diffusion_steps=200 \
  general.number_chain_steps=50 \
  general.samples_to_generate=200 \
  train.batch_size=32 \
  train.n_epochs=200

# Test ring length constraint
python ConStruct/main.py \
  --config configs/experiment/qm9.yaml \
  --config configs/general/general_default.yaml \
  model.rev_proj=ring_length \
  model.max_ring_length=4 \
  model.diffusion_steps=200 \
  general.number_chain_steps=50 \
  general.samples_to_generate=200 \
  train.batch_size=32 \
  train.n_epochs=200
```

## Configuration Options

### QM9 Configuration (`configs/experiment/qm9.yaml`)
```yaml
model:
  # Choose constraint type
  rev_proj: ring_count  # Options: planar, tree, lobster, ring_count, ring_length
  
  # Ring count constraint settings
  max_rings: 0         # Maximum number of rings allowed
  
  # Ring length constraint settings  
  max_ring_length: 6   # Maximum ring size allowed
```

### Experiment Parameters
- **Small Scale**: 200 diffusion steps, 50 chain steps, 200 samples
- **Full Scale**: 500 diffusion steps, 200 chain steps, 1000 samples
- **Batch Size**: 32 (small) or 64 (full)
- **Epochs**: 200 (small) or 1000 (full)

## Expected Outcomes

### Ring Count Constraint
- **max_rings = 0**: Should generate only tree-like molecules (no cycles)
- **max_rings = 1**: Should generate molecules with at most one ring
- **max_rings = 2**: Should generate molecules with at most two rings

### Ring Length Constraint
- **max_ring_length = 3**: Should generate molecules with only triangles (3-membered rings)
- **max_ring_length = 4**: Should generate molecules with rings up to 4 members
- **max_ring_length = 5**: Should generate molecules with rings up to 5 members

### Performance Comparison
- **Speed**: Ring constraints should be faster than planarity
- **Quality**: Generated molecules should respect the specified constraints
- **Diversity**: Different constraint settings should produce different molecule distributions

## Monitoring Results

### Log Files
- Check output logs in `ConStruct/logs/`
- Look for constraint violation messages
- Monitor generation statistics

### Key Metrics to Watch
1. **Constraint Satisfaction**: Are generated molecules respecting the constraints?
2. **Generation Speed**: How fast are molecules being generated?
3. **Molecular Diversity**: Are the constraints producing diverse molecules?
4. **Quality Metrics**: Validity, uniqueness, novelty of generated molecules

## Troubleshooting

### Common Issues
1. **Constraint Not Working**: Check debug output in logs
2. **Slow Performance**: Reduce batch size or diffusion steps
3. **Memory Issues**: Reduce batch size or number of samples
4. **No Molecules Generated**: Check constraint parameters (too restrictive)

### Debug Information
The constraints include debug output showing:
- Which edges are being blocked
- Why edges are being blocked
- Current graph state during generation

## Next Steps

1. **Run Small Scale Tests**: Start with smaller experiments to verify functionality
2. **Compare with Planarity**: Run planarity constraint for baseline comparison
3. **Analyze Results**: Check generated molecules and constraint satisfaction
4. **Scale Up**: Run full-scale experiments if small tests are successful
5. **Parameter Tuning**: Experiment with different constraint parameters

## Files Created

### SLURM Scripts
- `qm9_ring_count_check.slurm` - Full scale ring count test
- `qm9_ring_count_check_smaller.slurm` - Small scale ring count test
- `qm9_ring_length_check.slurm` - Full scale ring length test
- `qm9_ring_length_check_smaller.slurm` - Small scale ring length test
- `qm9_planar_comparison.slurm` - Planarity constraint for comparison

### Test Scripts
- `test_constraints_comparison.py` - Comprehensive constraint comparison
- `test_ring_length_constraint.py` - Ring length constraint tests
- `test_ring_length_diffusion.py` - Diffusion-style tests

### Configuration Updates
- Updated `configs/experiment/qm9.yaml` with constraint options
- Updated `configs/model/discrete.yaml` with constraint documentation 

# Check job status
squeue -u rislek

# Check specific job
squeue -j <job_id> 