# QM9 Ring Count Constraint Analysis Report

## Overview

This report summarizes the results of analyzing the QM9 dataset for the ring count constraint using the script `check_dataset_ring_count.py` with `max_rings=5`.

## Methodology

- **Dataset:** QM9 (train, val, test splits)
- **Constraint:** Maximum allowed rings per molecule = 5
- **Tool:** `count_rings()` function from `ConStruct.projector.is_ring.is_ring_count`
- **Process:**
  - For each molecule, count the number of rings.
  - Record the distribution of ring counts.
  - Identify and visualize molecules with more than 5 rings.
  - Output a summary and a JSON report.

## Results

### Ring Count Distribution

| Number of Rings | Molecule Count |
|-----------------|---------------|
| 0               | 13,387        |
| 1               | 51,445        |
| 2               | 41,360        |
| 3               | 19,795        |
| 4               | 4,385         |
| 5               | 447           |
| 6               | 9             |

- **Total molecules analyzed:** ~130,000+

### Violations

- **Number of molecules with more than 5 rings:** 9
- **Percentage of dataset violating constraint:** ~0.007%
- **Action:** Visualizations were saved for each violating molecule.

### Example Output
```
Found 9 graphs with more than 5 rings
Ring count distribution: {0: 13387, 1: 51445, 2: 41360, 3: 19795, 4: 4385, 5: 447, 6: 9}
```

## Interpretation

- The **ring count constraint (max 5 rings)** is almost always satisfied in QM9.
- Only a tiny fraction (9 out of ~130,000+) of molecules violate this constraint.
- Most molecules are simple, with 0–2 rings.
- Setting `max_rings=5` for generation or filtering will only exclude rare outliers.

## Recommendation

- **`max_rings=5` is a safe and reasonable constraint** for QM9-based molecular generation or filtering.
- If strict constraint satisfaction is required, removing or correcting the 9 violating molecules is trivial.

## Next Steps

- Run the **ring length constraint analysis** to check for molecules with large rings.
- Use the results to set or justify a `max_ring_length` constraint for QM9. 