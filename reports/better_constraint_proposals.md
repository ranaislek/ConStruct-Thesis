# 🎯 Better Constraint Proposals
## Achieving 100% Validity, Connectivity, and Constraint Satisfaction

**Date**: July 21, 2025  
**Project**: ConStruct - Improved Constraint Design  
**Status**: Proposal Phase

---

## 📋 Current Problems

### 1. **Low Validity Rate (45-60%)**
- **Issue**: Many invalid SMILES generated
- **Root cause**: Insufficient training, poor atom valency
- **Impact**: Constraint analysis meaningless if molecules invalid

### 2. **High Disconnected Rate (30-70%)**
- **Issue**: Multiple separate molecular fragments
- **Root cause**: Edge-insertion creating fragments instead of rings
- **Impact**: Not realistic molecules for drug discovery

### 3. **Poor Constraint Satisfaction (0-62.5%)**
- **Issue**: Constraints not being enforced properly
- **Root cause**: Complex constraints need more training
- **Impact**: Constraint system not working as intended

---

## 🚀 Better Constraint Proposals

### 1. **Simplified Ring Constraints**

#### A. **Single Ring Requirement**
```yaml
model:
  min_rings: 1                    # Exactly 1 ring
  min_ring_length: 5              # 5-membered ring (common)
  rev_proj: ring_count_exact      # Exact ring count
  transition: edge_insertion
```
**Advantages**:
- **Easier to satisfy**: Single ring is common
- **5-membered rings**: Very common in molecules
- **Exact constraint**: Clearer target than "at least"

#### B. **Ring Size Constraints**
```yaml
model:
  min_rings: 1
  ring_size: 6                    # Exactly 6-membered ring
  rev_proj: ring_size_exact       # Exact ring size
  transition: edge_insertion
```
**Advantages**:
- **Specific target**: Exactly 6 atoms
- **Aromatic potential**: 6-membered rings can be aromatic
- **Drug-like**: Common in pharmaceutical compounds

### 2. **Bond Type Constraints**

#### A. **Aromatic Ring Requirement**
```yaml
model:
  min_rings: 1
  aromatic_rings: 1               # At least 1 aromatic ring
  rev_proj: aromatic_ring_at_least
  transition: edge_insertion
```
**Advantages**:
- **Drug-like molecules**: Aromatic rings common in drugs
- **Stable structures**: Aromatic rings are chemically stable
- **Clear validation**: Easy to check aromaticity

#### B. **Bond Type Distribution**
```yaml
model:
  min_single_bonds: 3             # At least 3 single bonds
  min_double_bonds: 1             # At least 1 double bond
  rev_proj: bond_type_distribution
  transition: edge_insertion
```
**Advantages**:
- **Realistic molecules**: Natural bond distributions
- **Easy to satisfy**: Common in organic molecules
- **Good starting point**: Build up to more complex constraints

### 3. **Molecular Property Constraints**

#### A. **Molecular Weight Range**
```yaml
model:
  min_molecular_weight: 100       # At least 100 g/mol
  max_molecular_weight: 500       # At most 500 g/mol
  rev_proj: molecular_weight_range
  transition: edge_insertion
```
**Advantages**:
- **Drug-like size**: Typical drug molecular weights
- **Easy validation**: Simple to calculate and check
- **Realistic target**: Common in pharmaceutical compounds

#### B. **Atom Count Constraints**
```yaml
model:
  min_atoms: 8                    # At least 8 atoms
  max_atoms: 20                   # At most 20 atoms
  rev_proj: atom_count_range
  transition: edge_insertion
```
**Advantages**:
- **Manageable size**: Not too small, not too large
- **Easy to generate**: Reasonable complexity
- **Good validation**: Simple to count atoms

### 4. **Hybrid Constraints**

#### A. **Simple Ring + Size**
```yaml
model:
  min_rings: 1                    # Exactly 1 ring
  min_atoms: 8                    # At least 8 atoms
  max_atoms: 15                   # At most 15 atoms
  rev_proj: hybrid_ring_size
  transition: edge_insertion
```
**Advantages**:
- **Balanced complexity**: Not too simple, not too hard
- **Multiple targets**: Ring + size constraints
- **Good starting point**: Build up from here

#### B. **Aromatic + Size**
```yaml
model:
  aromatic_rings: 1               # At least 1 aromatic ring
  min_atoms: 10                   # At least 10 atoms
  max_atoms: 20                   # At most 20 atoms
  rev_proj: aromatic_size_hybrid
  transition: edge_insertion
```
**Advantages**:
- **Drug-like**: Aromatic rings + reasonable size
- **Stable structures**: Aromatic compounds are stable
- **Realistic target**: Common in pharmaceutical compounds

---

## 🔧 Implementation Strategy

### 1. **Phase 1: Foundation (100% Validity)**
```yaml
# Start with simple, achievable constraints
model:
  min_atoms: 6                    # At least 6 atoms
  max_atoms: 12                   # At most 12 atoms
  rev_proj: atom_count_range
  transition: edge_insertion

train:
  n_epochs: 100                   # Much longer training
  batch_size: 128                 # Larger batches
  num_workers: 4                  # Better data loading
```
**Goal**: Achieve 100% valid SMILES

### 2. **Phase 2: Connectivity (100% Connected)**
```yaml
# Add connectivity constraints
model:
  min_atoms: 6
  max_atoms: 12
  enforce_connectivity: true       # New constraint
  rev_proj: connected_molecules
  transition: edge_insertion
```
**Goal**: Achieve 100% connected molecules

### 3. **Phase 3: Ring Constraints (100% Satisfaction)**
```yaml
# Add ring constraints
model:
  min_atoms: 8
  max_atoms: 15
  min_rings: 1                    # Exactly 1 ring
  ring_size: 5                    # 5-membered ring
  enforce_connectivity: true
  rev_proj: single_ring_exact
  transition: edge_insertion
```
**Goal**: Achieve 100% constraint satisfaction

### 4. **Phase 4: Advanced Constraints**
```yaml
# More complex constraints
model:
  aromatic_rings: 1               # Aromatic ring
  min_atoms: 10
  max_atoms: 20
  enforce_connectivity: true
  rev_proj: aromatic_ring_at_least
  transition: edge_insertion
```
**Goal**: Complex but achievable constraints

---

## 📊 Expected Improvements

### 1. **Validity Rate**
- **Current**: 45-60%
- **Target**: >95%
- **Method**: Longer training, better loss functions

### 2. **Connectivity Rate**
- **Current**: 30-70% disconnected
- **Target**: <5% disconnected
- **Method**: Connectivity penalties, better edge placement

### 3. **Constraint Satisfaction**
- **Current**: 0-62.5%
- **Target**: >90%
- **Method**: Simpler constraints, better training

---

## 🎯 Recommended New Constraints

### 1. **Start Simple: Atom Count**
```yaml
model:
  min_atoms: 6
  max_atoms: 12
  rev_proj: atom_count_range
  transition: edge_insertion
```
**Expected**: 100% validity, 100% satisfaction

### 2. **Add Connectivity**
```yaml
model:
  min_atoms: 6
  max_atoms: 12
  enforce_connectivity: true
  rev_proj: connected_atom_count
  transition: edge_insertion
```
**Expected**: 100% validity, 100% connected, 100% satisfaction

### 3. **Add Single Ring**
```yaml
model:
  min_atoms: 8
  max_atoms: 15
  min_rings: 1
  ring_size: 5
  enforce_connectivity: true
  rev_proj: single_ring_exact
  transition: edge_insertion
```
**Expected**: 100% validity, 100% connected, 100% satisfaction

### 4. **Add Aromatic Ring**
```yaml
model:
  min_atoms: 10
  max_atoms: 20
  aromatic_rings: 1
  enforce_connectivity: true
  rev_proj: aromatic_ring_at_least
  transition: edge_insertion
```
**Expected**: 100% validity, 100% connected, 100% satisfaction

---

## 🚀 Implementation Plan

### 1. **Immediate Actions**
1. **Cancel current experiments** (not achieving targets)
2. **Create new configs** with simpler constraints
3. **Submit foundation experiments** (atom count only)
4. **Monitor validity rates** (target >95%)

### 2. **Short-term Goals**
1. **Achieve 100% validity** with atom count constraints
2. **Add connectivity constraints** once validity achieved
3. **Add simple ring constraints** once connectivity achieved
4. **Scale up complexity** gradually

### 3. **Long-term Vision**
1. **Complex ring constraints** (aromatic, multiple rings)
2. **Molecular property constraints** (weight, logP)
3. **Drug-like constraints** (functional groups, pharmacophores)
4. **Production pipeline** for constraint-based generation

---

## 📈 Success Metrics

### 1. **Foundation Metrics**
- **Validity Rate**: >95% valid SMILES
- **Connectivity Rate**: <5% disconnected
- **Constraint Satisfaction**: >90% for simple constraints

### 2. **Advanced Metrics**
- **Ring Constraints**: >90% satisfaction
- **Aromatic Constraints**: >90% satisfaction
- **Molecular Properties**: Within target ranges

### 3. **Quality Metrics**
- **Diversity**: Good variety in generated molecules
- **Novelty**: Molecules not in training set
- **Realistic Properties**: Drug-like characteristics

---

**Proposal Status**: Ready for Implementation  
**Next Phase**: Create new configs with simpler constraints  
**Research Focus**: Achieving 100% validity first, then adding complexity 