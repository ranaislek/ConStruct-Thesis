# Organized SLURM Scripts

This directory contains SLURM scripts organized by experiment level and constraint type, matching the configs folder structure.

## Directory Structure

```
ConStruct/slurm_jobs/
├── debug/                          # Debug-level SLURM scripts
│   ├── no_constraint/             # No constraint scripts
│   ├── edge_deletion/             # Edge-deletion constraints ("at most")
│   │   ├── ring_count_at_most/   # Ring count "at most" scripts
│   │   └── ring_length_at_most/  # Ring length "at most" scripts
│   └── edge_insertion/            # Edge-insertion constraints ("at least")
│       ├── ring_count_at_least/  # Ring count "at least" scripts
│       └── ring_length_at_least/ # Ring length "at least" scripts
└── thesis/                         # Thesis-level SLURM scripts
    ├── no_constraint/             # No constraint scripts
    ├── edge_deletion/             # Edge-deletion constraints ("at most")
    │   ├── ring_count_at_most/   # Ring count "at most" scripts
    │   └── ring_length_at_most/  # Ring length "at most" scripts
    └── edge_insertion/            # Edge-insertion constraints ("at least")
        ├── ring_count_at_least/  # Ring count "at least" scripts
        └── ring_length_at_least/ # Ring length "at least" scripts
```

## Constraint Types

### No Constraint Scripts
- **Purpose**: Baseline training without any constraints
- **Transition**: `absorbing_edges`
- **Use Case**: Generate molecules without any structural constraints

### Edge-Deletion (At Most) Scripts
- **Purpose**: Limit maximum ring count or ring length
- **Transition**: `absorbing_edges`
- **Use Case**: Generate molecules with limited ring complexity

### Edge-Insertion (At Least) Scripts
- **Purpose**: Ensure minimum ring count or ring length
- **Transition**: `edge_insertion`
- **Use Case**: Generate molecules with guaranteed ring structures

## Resource Requirements

### Debug Scripts
- **Time**: 12 hours
- **Memory**: 16GB
- **CPU**: 2 cores
- **GPU**: 1 GPU

### Thesis Scripts
- **Time**: 48 hours
- **Memory**: 32GB
- **CPU**: 4 cores
- **GPU**: 1 GPU

## Usage Examples

### Debug Experiments
```bash
# No constraint (debug)
sbatch ConStruct/slurm_jobs/debug/no_constraint/submit_no_constraint_debug.sh

# Ring count at most 2 (debug)
sbatch ConStruct/slurm_jobs/debug/edge_deletion/ring_count_at_most/submit_ring_count_at_most_2_debug.sh

# Ring length at least 5 (debug)
sbatch ConStruct/slurm_jobs/debug/edge_insertion/ring_length_at_least/submit_ring_length_at_least_5_debug.sh
```

### Thesis Experiments
```bash
# No constraint (thesis)
sbatch ConStruct/slurm_jobs/thesis/no_constraint/submit_no_constraint_thesis.sh

# Ring count at most 3 (thesis)
sbatch ConStruct/slurm_jobs/thesis/edge_deletion/ring_count_at_most/submit_ring_count_at_most_3_thesis.sh

# Ring count at least 2 (thesis)
sbatch ConStruct/slurm_jobs/thesis/edge_insertion/ring_count_at_least/submit_ring_count_at_least_2_thesis.sh
```

## Script Naming Convention

### No Constraint Scripts
- `submit_no_constraint_{level}.sh` - No constraint training

### Edge-Deletion Scripts
- `submit_ring_count_at_most_{N}_{level}.sh` - Ring count at most N
- `submit_ring_length_at_most_{N}_{level}.sh` - Ring length at most N

### Edge-Insertion Scripts
- `submit_ring_count_at_least_{N}_{level}.sh` - Ring count at least N
- `submit_ring_length_at_least_{N}_{level}.sh` - Ring length at least N

Where `{level}` is either `debug` or `thesis`.

## Constraint Values

### Ring Count Constraints
- **At Most**: 0, 1, 2, 3, 4, 5 (based on QM9 distribution)
- **At Least**: 1, 2, 3 (reasonable minimums)

### Ring Length Constraints
- **At Most**: 3, 4, 5, 6 (based on QM9 distribution)
- **At Least**: 4, 5, 6 (reasonable minimums)

## Monitoring

All scripts create log files in the `logs/` directory:
- `logs/{job_name}_{job_id}.out` - Standard output
- `logs/{job_name}_{job_id}.err` - Error output

## Best Practices

1. **Start with Debug**: Use debug scripts for quick validation
2. **Monitor Resources**: Check resource usage during execution
3. **Check Logs**: Monitor log files for errors or issues
4. **Use Appropriate Level**: Choose debug vs thesis based on your needs
