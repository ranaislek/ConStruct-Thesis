# Dataset Analysis Best Practices

## 🎯 **Recommended Approach: Comprehensive Dataset Analysis**

### **Why Analyze Both Training and Full Dataset?**

#### **1. Training Dataset Analysis**
- **Purpose**: Understand what the model learns from
- **Use case**: Compare generated molecules vs training distribution
- **Paper alignment**: Models are trained on training split only
- **Key insight**: Generated molecules should match training distribution

#### **2. Full Dataset Analysis**
- **Purpose**: Understand complete dataset properties
- **Use case**: Comprehensive dataset characterization
- **Paper alignment**: Full dataset analysis for constraint understanding
- **Key insight**: Understand dataset-wide constraints and properties

#### **3. Comparison Analysis**
- **Purpose**: Identify differences between training and full dataset
- **Use case**: Validate training split representativeness
- **Paper alignment**: Ensure training data is representative
- **Key insight**: Training split should be representative of full dataset

## 📊 **Analysis Structure**

### **Directory Organization**
```
dataset_analysis/
├── train_analysis/          # Training dataset analysis
│   ├── plots/
│   ├── reports/
│   └── data/
├── full_analysis/           # Full dataset analysis
│   ├── plots/
│   ├── reports/
│   └── data/
├── comparison/              # Comparison analysis
│   ├── plots/
│   ├── reports/
│   └── data/
├── plots/                   # Combined visualizations
├── reports/                 # Combined reports
└── data/                    # Combined data files
```

### **Analysis Components**

#### **1. Ring Count Analysis**
- Distribution of molecules by ring count
- Constraint satisfaction rates (≤0, ≤1, ≤2, ≤3, ≤4, ≤5 rings)
- Comparison between training and full dataset

#### **2. Ring Length Analysis**
- Distribution of rings by length (3-8 atoms)
- Constraint satisfaction rates (≤3, ≤4, ≤5, ≤6, ≤7, ≤8 atoms)
- Comparison between training and full dataset

#### **3. Planarity Analysis**
- Planar vs non-planar molecule distribution
- Planarity rates for each dataset
- Comparison between training and full dataset

#### **4. Constraint Satisfaction Analysis**
- Ring count constraint satisfaction rates
- Ring length constraint satisfaction rates
- Comparison between training and full dataset

## 🔬 **Paper Alignment**

### **ConStruct Paper Approach**
The original ConStruct paper analyzes:
1. **Training data distribution** - What the model learns from
2. **Full dataset properties** - Complete dataset characterization
3. **Constraint satisfaction** - How well constraints work on real data

### **Best Practices**
1. **Always analyze training split** - This is what the model learns from
2. **Analyze full dataset** - Understand complete dataset properties
3. **Compare distributions** - Ensure training split is representative
4. **Use consistent methods** - Same analysis for both splits

## 📈 **Expected Results**

### **Training vs Full Dataset**
- **Similar distributions**: Training split should be representative
- **Minor differences**: Due to random splitting
- **Consistent constraint rates**: Both should show similar patterns

### **Key Metrics**
- **Ring count distribution**: Should be similar between splits
- **Ring length distribution**: Should be similar between splits
- **Planarity rates**: Should be identical (100% for QM9)
- **Constraint satisfaction**: Should be similar between splits

## 🚀 **Usage**

### **Run Complete Analysis**
```bash
# QM9 Analysis
cd ConStruct/datasets/qm9_analysis/
python complete_qm9_analysis.py

# Moses Analysis  
cd ConStruct/datasets/moses_analysis/
python complete_moses_analysis.py

# Guacamol Analysis
cd ConStruct/datasets/guacamol_analysis/
python complete_guacamol_analysis.py
```

### **Output Structure**
Each analysis produces:
1. **Training dataset analysis** - What model learns from
2. **Full dataset analysis** - Complete dataset properties
3. **Comparison analysis** - Training vs full dataset differences
4. **Visualizations** - Plots for each analysis type
5. **Reports** - Detailed textual reports
6. **Data files** - JSON data for further analysis

## 📋 **Analysis Checklist**

### **Before Running Experiments**
- [ ] Analyze training dataset distribution
- [ ] Analyze full dataset properties
- [ ] Compare training vs full dataset
- [ ] Validate training split representativeness
- [ ] Document constraint satisfaction rates
- [ ] Create visualizations for paper

### **For Each Dataset**
- [ ] QM9 analysis completed
- [ ] Moses analysis completed
- [ ] Guacamol analysis completed
- [ ] Comparison analysis completed
- [ ] Reports generated
- [ ] Visualizations created

## 🎯 **Key Insights**

### **Training Dataset Analysis**
- **Purpose**: Understand what the model learns from
- **Focus**: Ring count, ring length, planarity distributions
- **Use**: Compare generated molecules vs training data

### **Full Dataset Analysis**
- **Purpose**: Understand complete dataset properties
- **Focus**: Comprehensive dataset characterization
- **Use**: Understand dataset-wide constraints

### **Comparison Analysis**
- **Purpose**: Validate training split representativeness
- **Focus**: Differences between training and full dataset
- **Use**: Ensure training data is representative

## 📊 **Expected Findings**

### **QM9 Dataset**
- **Training size**: ~78,000 molecules
- **Full size**: ~97,000 molecules
- **Planarity**: 100% planar
- **Ring distribution**: Similar between splits

### **Moses Dataset**
- **Training size**: ~1,200,000 molecules
- **Full size**: ~1,500,000 molecules
- **Planarity**: 100% planar
- **Ring distribution**: Similar between splits

### **Guacamol Dataset**
- **Training size**: ~800,000 molecules
- **Full size**: ~1,000,000 molecules
- **Planarity**: 100% planar
- **Ring distribution**: Similar between splits

## 🔧 **Technical Notes**

### **NetworkX vs RDKit**
- **Dataset analysis**: Use NetworkX (real molecules don't have spurious cycles)
- **Generated molecule analysis**: Use RDKit (filter out non-chemical cycles)
- **Consistency**: Same method for both training and full dataset analysis

### **Performance Considerations**
- **Large datasets**: Process in batches
- **Memory usage**: Monitor during analysis
- **Time**: Full dataset analysis takes longer

---

*This approach ensures comprehensive dataset understanding and paper alignment* 