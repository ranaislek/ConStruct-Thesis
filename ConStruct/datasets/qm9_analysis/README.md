# QM9 Dataset Structural Analysis

This directory contains the comprehensive analysis of the QM9 dataset's structural properties, including ring count, ring length, and planarity distributions.

## 📊 Analysis Overview

**Dataset Size**: 97,732 molecules  
**Total Rings**: 68,847  
**Average Rings per Molecule**: 0.704  
**Planarity Rate**: 100.0%

## 📁 Directory Structure

```
qm9_analysis/
├── data/
│   └── qm9_dataset_analysis.json          # Raw analysis data
├── reports/
│   ├── qm9_dataset_analysis_report.txt    # Detailed analysis report
│   └── qm9_dataset_summary.txt            # Executive summary
├── plots/
│   ├── ring_count_distribution.png        # Distribution of molecules by ring count
│   ├── ring_length_distribution.png       # Distribution of rings by length
│   ├── constraint_satisfaction_rates.png  # Constraint satisfaction rates
│   ├── cumulative_ring_count.png          # Cumulative ring count distribution
│   └── planarity_distribution.png         # Planarity distribution
├── complete_qm9_analysis.py               # Complete analysis script
└── README.md                              # This file
```

## 🎯 Key Findings

### Ring Count Distribution
- **10.3%** of molecules are **acyclic** (0 rings)
- **39.2%** have **1 ring**
- **31.6%** have **2 rings**
- **15.2%** have **3 rings**
- **3.4%** have **4 rings**
- **0.4%** have **5 rings**
- **0.0%** have **6 rings**

### Ring Length Distribution
- **3-atom rings**: 28,456 rings
- **4-atom rings**: 15,436 rings
- **5-atom rings**: 36,525 rings
- **6-atom rings**: 14,256 rings
- **7-atom rings**: 1,425 rings
- **8-atom rings**: 149 rings

### Constraint Satisfaction Rates

#### Ring Count Constraints
- **≤0 rings** (acyclic): **10.3%**
- **≤1 ring**: **49.5%**
- **≤2 rings**: **81.1%**
- **≤3 rings**: **96.3%**
- **≤4 rings**: **99.6%**
- **≤5 rings**: **100.0%**

#### Ring Length Constraints
- **≤3 atoms**: **21.2%**
- **≤4 atoms**: **44.9%**
- **≤5 atoms**: **82.2%**
- **≤6 atoms**: **96.8%**
- **≤7 atoms**: **98.5%**
- **≤8 atoms**: **99.8%**

## 🔬 Implications for Constraint Experiments

### Ring Count Constraints
- **Acyclic constraint (≤0 rings)** is very restrictive - only 10.3% of molecules satisfy it
- **≤1 ring constraint** is moderately restrictive - 49.5% satisfy it
- **≤2 rings constraint** is more lenient - 81.1% satisfy it
- **≤3+ rings constraints** are very lenient - 96%+ satisfy them

### Ring Length Constraints
- **≤3 atoms constraint** is very restrictive - only 21.2% satisfy it
- **≤4 atoms constraint** is moderately restrictive - 44.9% satisfy it
- **≤5 atoms constraint** is more lenient - 82.2% satisfy it
- **≤6 atoms constraint** is very lenient - 96.8% satisfy it
- **≤7 atoms constraint** is extremely lenient - 98.5% satisfy it
- **≤8 atoms constraint** is nearly universal - 99.8% satisfy it

### Planarity
- **100% of QM9 molecules are planar** - this constraint will not restrict the dataset

## 📈 Visualization Files

1. **ring_count_distribution.png** - Shows the distribution of molecules by number of rings
2. **ring_length_distribution.png** - Shows the distribution of rings by length (3-8 atoms)
3. **constraint_satisfaction_rates.png** - Shows satisfaction rates for different constraint thresholds
4. **cumulative_ring_count.png** - Shows cumulative distribution of ring counts
5. **planarity_distribution.png** - Shows planarity distribution (100% planar)

## 🎯 Recommendations for Thesis Experiments

### Most Challenging Constraints
1. **Acyclic (≤0 rings)** - Only 10.3% satisfaction rate
2. **Ring length ≤3** - Only 21.2% satisfaction rate
3. **Ring length ≤4** - Only 44.9% satisfaction rate

### Moderate Constraints
1. **≤1 ring** - 49.5% satisfaction rate
2. **Ring length ≤5** - 82.2% satisfaction rate

### Lenient Constraints
1. **≤2 rings** - 81.1% satisfaction rate
2. **≤3 rings** - 96.3% satisfaction rate
3. **Ring length ≤6** - 96.8% satisfaction rate
4. **Ring length ≤7** - 98.5% satisfaction rate
5. **Ring length ≤8** - 99.8% satisfaction rate

### No Constraint Effect
- **Planarity** - 100% satisfaction rate (no restriction)

## 📊 Data Files

- **qm9_dataset_analysis.json** - Complete analysis data in JSON format
- **qm9_dataset_analysis_report.txt** - Detailed textual report
- **qm9_dataset_summary.txt** - Executive summary

## 🚀 Usage

To run the complete analysis:

```bash
cd /home/rislek/ConStruct-Thesis/ConStruct/datasets/qm9_analysis/
python complete_qm9_analysis.py
```

This will:
1. Load the QM9 dataset
2. Analyze ring properties, planarity, and connectivity
3. Calculate constraint satisfaction rates
4. Generate visualization plots
5. Save detailed reports and data files

---

*Analysis completed on QM9 dataset with 97,732 molecules*
