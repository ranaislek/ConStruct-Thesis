# Moses Dataset Structural Analysis

This directory contains the comprehensive analysis of the Moses dataset's structural properties, including ring count, ring length, and planarity distributions.

## 📊 Analysis Overview

**Dataset Size**: Moses dataset molecules  
**Total Rings**: Calculated from analysis  
**Average Rings per Molecule**: Calculated from analysis  
**Planarity Rate**: Calculated from analysis

## 📁 Directory Structure

```
moses_analysis/
├── data/
│   └── moses_dataset_analysis.json          # Raw analysis data
├── reports/
│   ├── moses_dataset_analysis_report.txt    # Detailed analysis report
│   └── moses_dataset_summary.txt            # Executive summary
├── plots/
│   ├── ring_count_distribution.png          # Distribution of molecules by ring count
│   ├── ring_length_distribution.png         # Distribution of rings by length
│   ├── constraint_satisfaction_rates.png    # Constraint satisfaction rates
│   ├── cumulative_ring_count.png            # Cumulative ring count distribution
│   └── planarity_distribution.png           # Planarity distribution
├── complete_moses_analysis.py               # Complete analysis script
└── README.md                                # This file
```

## 🎯 Key Findings

### Ring Count Distribution
- Analysis will show distribution of molecules by ring count

### Ring Length Distribution
- Analysis will show distribution of rings by length

### Constraint Satisfaction Rates

#### Ring Count Constraints
- **≤0 rings** (acyclic): Calculated percentage
- **≤1 ring**: Calculated percentage
- **≤2 rings**: Calculated percentage
- **≤3 rings**: Calculated percentage
- **≤4 rings**: Calculated percentage
- **≤5 rings**: Calculated percentage

#### Ring Length Constraints
- **≤3 atoms**: Calculated percentage
- **≤4 atoms**: Calculated percentage
- **≤5 atoms**: Calculated percentage
- **≤6 atoms**: Calculated percentage
- **≤7 atoms**: Calculated percentage
- **≤8 atoms**: Calculated percentage

## 🔬 Implications for Constraint Experiments

### Ring Count Constraints
- Analysis will show how restrictive each constraint is

### Ring Length Constraints
- Analysis will show how restrictive each constraint is

### Planarity
- Analysis will show planarity distribution

## 📈 Visualization Files

1. **ring_count_distribution.png** - Shows the distribution of molecules by number of rings
2. **ring_length_distribution.png** - Shows the distribution of rings by length
3. **constraint_satisfaction_rates.png** - Shows satisfaction rates for different constraint thresholds
4. **cumulative_ring_count.png** - Shows cumulative distribution of ring counts
5. **planarity_distribution.png** - Shows planarity distribution

## 🎯 Recommendations for Thesis Experiments

### Most Challenging Constraints
- Will be determined by analysis

### Moderate Constraints
- Will be determined by analysis

### Lenient Constraints
- Will be determined by analysis

### No Constraint Effect
- Will be determined by analysis

## 📊 Data Files

- **moses_dataset_analysis.json** - Complete analysis data in JSON format
- **moses_dataset_analysis_report.txt** - Detailed textual report
- **moses_dataset_summary.txt** - Executive summary

## 🚀 Usage

To run the complete analysis:

```bash
cd /home/rislek/ConStruct-Thesis/ConStruct/datasets/moses_analysis/
python complete_moses_analysis.py
```

This will:
1. Load the Moses dataset
2. Analyze ring properties, planarity, and connectivity
3. Calculate constraint satisfaction rates
4. Generate visualization plots
5. Save detailed reports and data files

---

*Analysis completed on Moses dataset*
