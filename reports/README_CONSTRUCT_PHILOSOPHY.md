# ConStruct Philosophy: Structural-Only Constraints and Post-Generation Validation

This codebase has been refactored to follow the **ConStruct philosophy** of enforcing only structural constraints during generation and measuring chemical properties post-generation. This enables honest, critical analysis of the gap between structural satisfaction and chemical validity.

## 🎯 **ConStruct Philosophy**

### **Core Principles**

1. **Structural-Only Constraints**: Projectors enforce ONLY structural constraints (ring count, ring length, planarity)
2. **No Chemical Enforcement**: Chemical validity (valency, connectivity, atom types) is NOT enforced during generation
3. **Post-Generation Measurement**: Chemical properties are measured separately after generation
4. **Honest Gap Analysis**: This reveals the true relationship between structural and chemical constraints

### **Why This Matters for Your Thesis**

- **Reproducible Results**: Clear separation of structural vs chemical constraints
- **Critical Analysis**: Honest measurement of the gap between structural satisfaction and chemical validity
- **ConStruct Alignment**: Follows the original paper's methodology for planarity experiments
- **Thesis-Ready**: Provides systematic data for analyzing constraint satisfaction vs chemical reality

## 🔄 **Transition Mechanisms**

### **Edge-Deletion Transitions** (`absorbing_edges`)

- **Absorbing State**: No-edge (index 0)
- **Forward Process**: Edges progressively disappear
- **Reverse Process**: Edges are added while preserving constraints
- **Natural Bias**: Toward sparse graphs
- **Constraints**: "At most" (e.g., at most N rings)

### **Edge-Insertion Transitions** (`edge_insertion`)

- **Absorbing State**: Edge (index 1)
- **Forward Process**: Edges progressively appear
- **Reverse Process**: Edges are removed while preserving constraints
- **Natural Bias**: Toward connected graphs
- **Constraints**: "At least" (e.g., at least N rings)

## 🏗️ **Structural Constraints**

### **Ring Count Constraints**

```yaml
# Edge-deletion: "At most N rings"
rev_proj: 'ring_count_at_most'
max_rings: 2

# Edge-insertion: "At least N rings"
rev_proj: 'ring_count_at_least'
min_rings: 2
```

### **Ring Length Constraints**

```yaml
# Edge-deletion: "All rings at most N length"
rev_proj: 'ring_length_at_most'
max_ring_length: 6

# Edge-insertion: "All rings at least N length"
rev_proj: 'ring_length_at_least'
min_ring_length: 4
```

### **Other Structural Constraints**

```yaml
# Planarity
rev_proj: 'planar'

# Tree structure
rev_proj: 'tree'

# Lobster structure
rev_proj: 'lobster'
```

## 🧪 **Post-Generation Validation**

### **Chemical Properties Measured**

- **Valency Violations**: Check if atoms exceed expected valency
- **Connectivity**: Single vs multiple components
- **Atom Type Validity**: Valid atom symbols
- **RDKit Validity**: Molecular structure validation
- **Molecular Descriptors**: Weight, logP, HBD, HBA, etc.

### **Configuration**

```yaml
# Enable post-generation validation
post_gen_validation: True
post_gen_validation_config:
  enable_rdkit: True
  enable_valency_check: True
  enable_connectivity_check: True
```

### **Output Analysis**

The validation provides honest metrics:

```
POST-GENERATION CHEMICAL VALIDATION SUMMARY
============================================================
Total graphs generated: 100
Structurally valid: 95 (95.0%)
Chemically valid: 78 (78.0%)
RDKit valid: 65 (65.0%)

CHEMICAL VALIDITY ISSUES (Post-Generation):
  Valency violations: 15
  Connectivity issues: 8
  Atom type issues: 2

CONSTRUCT PHILOSOPHY ANALYSIS:
  ✓ Structural constraints were enforced during generation
  ✓ Chemical properties were measured post-generation
  ✓ Gap analysis enables honest, critical thesis results
============================================================
```

## 📋 **Configuration Examples**

### **Edge-Deletion Example**

```yaml
# configs/model/edge_deletion_example.yaml
transition: 'absorbing_edges'
rev_proj: 'ring_count_at_most'
max_rings: 2
post_gen_validation: True
```

### **Edge-Insertion Example**

```yaml
# configs/model/edge_insertion_example.yaml
transition: 'edge_insertion'
rev_proj: 'ring_count_at_least'
min_rings: 2
post_gen_validation: True
```

## 🚀 **Usage Instructions**

### **1. Choose Your Transition Type**

```bash
# Edge-deletion (sparse graphs, "at most" constraints)
python main.py model=discrete model.transition=absorbing_edges model.rev_proj=ring_count_at_most model.max_rings=2

# Edge-insertion (connected graphs, "at least" constraints)
python main.py model=discrete model.transition=edge_insertion model.rev_proj=ring_count_at_least model.min_rings=2
```

### **2. Enable Post-Generation Validation**

```bash
python main.py model=discrete model.post_gen_validation=True
```

### **3. Run Training**

```bash
python main.py model=discrete dataset=qm9 train.batch_size=32
```

### **4. Generate and Validate**

```bash
python main.py model=discrete dataset=qm9 test=True
```

## 📊 **Thesis Analysis Framework**

### **Key Metrics for Analysis**

1. **Structural Satisfaction Rate**: % of graphs satisfying structural constraints
2. **Chemical Validity Rate**: % of graphs that are chemically valid
3. **Gap Analysis**: Difference between structural and chemical satisfaction
4. **Constraint Type Impact**: How different constraints affect chemical validity
5. **Transition Mechanism Impact**: Edge-deletion vs edge-insertion performance

### **Research Questions**

- How well do structural constraints correlate with chemical validity?
- Which constraint types produce the most chemically valid molecules?
- What is the trade-off between structural satisfaction and chemical reality?
- How do different transition mechanisms affect the structural-chemical gap?

## 🔬 **Code Structure**

### **Core Components**

- **`EdgeInsertionTransition`**: Edge-insertion mechanism with structural-only constraints
- **`RingCountAtLeastProjector`**: "At least" ring count enforcement
- **`RingLengthAtLeastProjector`**: "At least" ring length enforcement
- **`PostGenerationValidator`**: Post-generation chemical validation
- **Configuration Files**: Clear separation of edge-deletion vs edge-insertion

### **Key Files**

```
ConStruct/
├── diffusion/
│   └── noise_model.py              # Edge-insertion transition
├── projector/
│   └── projector_utils.py          # Structural-only projectors
├── metrics/
│   └── post_generation_validation.py  # Post-generation validation
└── diffusion_model_discrete.py     # Main model with validation integration
```

## 🎓 **Thesis Integration**

### **For Your Thesis**

1. **Systematic Experiments**: Run both edge-deletion and edge-insertion with various constraints
2. **Gap Analysis**: Measure the structural-chemical validity gap for each configuration
3. **Critical Discussion**: Analyze why structural constraints don't guarantee chemical validity
4. **ConStruct Comparison**: Compare your results with the original ConStruct planarity experiments
5. **Future Directions**: Propose improvements based on your findings

### **Expected Outcomes**

- **Honest Results**: Clear separation of what was enforced vs what was measured
- **Critical Insights**: Understanding of structural-chemical relationship
- **Reproducible Analysis**: Systematic framework for constraint evaluation
- **Thesis-Ready Data**: Comprehensive metrics for your analysis

## 🔍 **Validation and Debugging**

### **Transition-Projector Compatibility**

The system automatically validates that you're using compatible transitions and projectors:

```bash
# ✅ Valid: Edge-deletion with "at most" projector
transition: 'absorbing_edges'
rev_proj: 'ring_count_at_most'

# ✅ Valid: Edge-insertion with "at least" projector  
transition: 'edge_insertion'
rev_proj: 'ring_count_at_least'

# ❌ Invalid: Edge-deletion with "at least" projector
transition: 'absorbing_edges'
rev_proj: 'ring_count_at_least'  # Will raise error
```

### **Debugging Tips**

1. **Check Transition Type**: Ensure you're using the right transition for your constraints
2. **Validate Configuration**: Use the compatibility checker
3. **Monitor Validation**: Watch the post-generation validation output
4. **Analyze Gaps**: Focus on the structural-chemical validity gap

## 📈 **Expected Results**

### **Typical Validation Output**

```
POST-GENERATION CHEMICAL VALIDATION SUMMARY
============================================================
Total graphs generated: 1000
Structurally valid: 950 (95.0%)  ← What was enforced
Chemically valid: 780 (78.0%)    ← What was measured
RDKit valid: 650 (65.0%)         ← Molecular validity

CHEMICAL VALIDITY ISSUES:
  Valency violations: 150         ← Gap analysis
  Connectivity issues: 80         ← Gap analysis  
  Atom type issues: 20           ← Gap analysis

CONSTRUCT PHILOSOPHY ANALYSIS:
  ✓ Structural constraints were enforced during generation
  ✓ Chemical properties were measured post-generation
  ✓ Gap analysis enables honest, critical thesis results
============================================================
```

This framework provides the honest, critical analysis needed for your thesis while following the ConStruct philosophy of structural-only constraints with post-generation chemical validation. 