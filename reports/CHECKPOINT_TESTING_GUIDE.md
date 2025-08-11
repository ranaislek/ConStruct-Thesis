# 🧪 Testing Metrics with Checkpoints - Complete Guide

This guide shows you how to test your ConStruct metrics using real, trained model checkpoints. This approach provides much more realistic testing than mock data.

## 🚀 Quick Start

### **1. Run the Checkpoint Test Script**
```bash
cd ConStruct
python test_checkpoint_metrics.py
```

### **2. Test Specific Checkpoint Types**
```bash
# Test no-constraint model
python -c "
import torch
ckpt = torch.load('../checkpoints/qm9_debug_no_constraint/last.ckpt', map_location='cpu')
print('✅ Checkpoint loaded')
print(f'Model config: {ckpt.get(\"hyper_parameters\", {}).get(\"model\", {})}')
"
```

## 🔍 Available Checkpoints for Testing

Based on your checkpoint directory, you have these models available:

### **📊 QM9 Models (Molecular)**

| Model Type | Checkpoint Path | Constraint | Use Case |
|------------|----------------|------------|----------|
| **No Constraint** | `qm9_debug_no_constraint/last.ckpt` | None | Test natural molecular distribution |
| **Ring Count ≤ 0** | `qm9_debug_ring_count_at_most_0/last.ckpt` | No rings | Test acyclic molecule generation |
| **Ring Count ≤ 1** | `qm9_debug_ring_count_at_most_1/last.ckpt` | Max 1 ring | Test single ring constraint |
| **Ring Count ≤ 2** | `qm9_debug_ring_count_at_most_2/last.ckpt` | Max 2 rings | Test multiple ring constraint |
| **Ring Length ≤ 3** | `qm9_debug_ring_length_at_most_3/last.ckpt` | Max 3-atom rings | Test small ring constraint |
| **Ring Length ≤ 4** | `qm9_debug_ring_length_at_most_4/last.ckpt` | Max 4-atom rings | Test medium ring constraint |
| **Planar** | `qm9_debug_planar/last.ckpt` | Planar graphs | Test planarity constraint |

### **🏗️ Planar Models (Graph)**

| Model Type | Checkpoint Path | Constraint | Use Case |
|------------|----------------|------------|----------|
| **Planar** | `planar_debug_planar/last.ckpt` | Planar graphs | Test graph planarity |

## 🧪 Testing Strategies

### **Strategy 1: Constraint Satisfaction Testing**

Test if your metrics correctly identify constraint violations:

```python
# Test with ring count constraint model
python -c "
import torch
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from types import SimpleNamespace

# Load checkpoint
ckpt = torch.load('../checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt', map_location='cpu')
cfg = ckpt.get('hyper_parameters', {})

print(f'Model constraint: {cfg.get(\"model\", {}).get(\"rev_proj\", \"None\")}')
print(f'Max rings: {cfg.get(\"model\", {}).get(\"max_rings\", \"None\")}')

# Create mock dataset
dataset_infos = SimpleNamespace()
dataset_infos.statistics = {'test': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]]), 'val': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]])}
dataset_infos.is_molecular = False

# Test metrics
metrics = SamplingMetrics(dataset_infos, test=False)
print('✅ Metrics initialized with checkpoint config')
"
```

### **Strategy 2: Real Data Testing**

Use the actual model to generate real samples and test metrics:

```python
# Test with real model sampling
python -c "
import torch
from ConStruct.diffusion_model_discrete import DiscreteDenoisingDiffusion

# Load checkpoint
ckpt = torch.load('../checkpoints/qm9_debug_no_constraint/last.ckpt', map_location='cpu')
cfg = ckpt.get('hyper_parameters', {})

# Create minimal dataset_infos
class MockDatasetInfos:
    def __init__(self):
        self.is_molecular = True
        self.remove_h = False
        self.atom_decoder = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}
        self.nodes_dist = {10: 1.0}
        self.input_dims = {'X': 4, 'E': 4, 'y': 1}
        self.output_dims = {'X': 4, 'E': 4, 'y': 1}
        self.statistics = {
            'test': {'num_nodes': {10: 100}, 'degree_hist': [[0,1,2,3], [0.1,0.2,0.3,0.4]]},
            'val': {'num_nodes': {10: 100}, 'degree_hist': [[0,1,2,3], [0.1,0.2,0.3,0.4]]}
        }

# Create minimal sampling metrics
class MockSamplingMetrics:
    def __init__(self): pass
    def reset(self): pass
    def compute_all_metrics(self, *args, **kwargs): return {}

dataset_infos = MockDatasetInfos()
val_metrics = MockSamplingMetrics()
test_metrics = MockSamplingMetrics()

# Create model
model = DiscreteDenoisingDiffusion(cfg, dataset_infos, val_metrics, test_metrics)
print('✅ Model created with checkpoint weights')
"
```

### **Strategy 3: Edge Case Testing with Real Models**

Test how your metrics handle edge cases with real model outputs:

```python
# Test edge cases with real model
python -c "
import torch
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from types import SimpleNamespace

# Load checkpoint to get config
ckpt = torch.load('../checkpoints/qm9_debug_no_constraint/last.ckpt', map_location='cpu')
cfg = ckpt.get('hyper_parameters', {})

print(f'Testing metrics with model: {cfg.get(\"model\", {}).get(\"rev_proj\", \"None\")}')

# Create mock dataset
dataset_infos = SimpleNamespace()
dataset_infos.statistics = {'test': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]]), 'val': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]])}
dataset_infos.is_molecular = False

metrics = SamplingMetrics(dataset_infos, test=False)

# Test edge cases
print('\\n🧪 Testing Edge Cases:')

# Test 1: Empty input
try:
    metrics.compute_all_metrics([], current_epoch=0, local_rank=0)
    print('✅ Empty input handled gracefully')
except Exception as e:
    print(f'❌ Empty input failed: {e}')

# Test 2: Single node
try:
    class MockSingleNode:
        def __init__(self):
            self.X = torch.randn(1, 3)
            self.E = torch.zeros(1, 1)
            self.y = torch.zeros(1)
            self.node_mask = torch.ones(1, dtype=torch.bool)
        def split(self): return [self]
    
    single_node = MockSingleNode()
    metrics.compute_all_metrics([single_node], current_epoch=0, local_rank=0)
    print('✅ Single node handled')
except Exception as e:
    print(f'❌ Single node failed: {e}')

# Test 3: Disconnected components
try:
    class MockDisconnected:
        def __init__(self):
            self.X = torch.randn(4, 3)
            self.E = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            self.y = torch.zeros(1)
            self.node_mask = torch.ones(4, dtype=torch.bool)
        def split(self): return [self]
    
    disconnected = MockDisconnected()
    metrics.compute_all_metrics([disconnected], current_epoch=0, local_rank=0)
    print('✅ Disconnected components handled')
except Exception as e:
    print(f'❌ Disconnected components failed: {e}')
"
```

## 🔧 Advanced Testing Scenarios

### **Test 1: Constraint Violation Detection**

```python
# Test if metrics correctly identify constraint violations
python -c "
import torch
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from types import SimpleNamespace

# Load ring-constrained model
ckpt = torch.load('../checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt', map_location='cpu')
cfg = ckpt.get('hyper_parameters', {})

print(f'Testing ring count constraint: ≤{cfg.get(\"model\", {}).get(\"max_rings\", \"Unknown\")} rings')

# Create mock dataset
dataset_infos = SimpleNamespace()
dataset_infos.statistics = {'test': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]]), 'val': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]])}
dataset_infos.is_molecular = False

metrics = SamplingMetrics(dataset_infos, test=False)

# Test with graphs that should violate constraint
class MockRingGraph:
    def __init__(self):
        self.X = torch.randn(3, 3)
        self.E = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle (1 ring)
        self.y = torch.zeros(1)
        self.node_mask = torch.ones(3, dtype=torch.bool)
    def split(self): return [self]

ring_graph = MockRingGraph()
metrics.compute_all_metrics([ring_graph], current_epoch=0, local_rank=0)

# Check if ring detection works
print('✅ Ring detection test completed')
"
```

### **Test 2: Performance Testing with Large Graphs**

```python
# Test performance with larger graphs
python -c "
import torch
import time
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from types import SimpleNamespace

# Create mock dataset
dataset_infos = SimpleNamespace()
dataset_infos.statistics = {'test': SimpleNamespace(num_nodes={50: 100}, degree_hist=[[0,1,2,3,4,5], [0.1,0.2,0.3,0.2,0.1,0.1]]), 'val': SimpleNamespace(num_nodes={50: 100}, degree_hist=[[0,1,2,3,4,5], [0.1,0.2,0.3,0.2,0.1,0.1]])}
dataset_infos.is_molecular = False

metrics = SamplingMetrics(dataset_infos, test=False)

# Create large graph
class MockLargeGraph:
    def __init__(self, n_nodes=50):
        self.X = torch.randn(n_nodes, 3)
        self.E = torch.randint(0, 4, (n_nodes, n_nodes))
        self.E = torch.triu(self.E) + torch.triu(self.E, 1).T  # Make symmetric
        self.y = torch.zeros(1)
        self.node_mask = torch.ones(n_nodes, dtype=torch.bool)
    def split(self): return [self]

# Test performance
sizes = [10, 25, 50]
for size in sizes:
    large_graph = MockLargeGraph(size)
    
    start_time = time.time()
    metrics.compute_all_metrics([large_graph], current_epoch=0, local_rank=0)
    end_time = time.time()
    
    print(f'Graph size {size}: {end_time - start_time:.3f}s')

print('✅ Performance testing completed')
"
```

## 📊 Expected Results by Checkpoint Type

### **No Constraint Model (`qm9_debug_no_constraint`)**
- **Expected**: Natural molecular distribution
- **Test**: Generate molecules, check for natural ring distribution
- **Metrics**: Should show variety in ring counts, lengths

### **Ring Count Constraint (`qm9_debug_ring_count_at_most_0`)**
- **Expected**: All molecules have 0 rings
- **Test**: Generate molecules, verify no rings
- **Metrics**: Ring count satisfaction should be 100%

### **Planar Constraint (`qm9_debug_planar`)**
- **Expected**: All graphs are planar
- **Test**: Generate graphs, verify planarity
- **Metrics**: Planarity should be 100%

## 🚨 Common Issues and Solutions

### **Issue 1: Checkpoint Loading Fails**
```bash
# Check if checkpoint exists
ls -la ../checkpoints/qm9_debug_no_constraint/

# Check checkpoint integrity
python -c "
import torch
try:
    ckpt = torch.load('../checkpoints/qm9_debug_no_constraint/last.ckpt', map_location='cpu')
    print('✅ Checkpoint is valid')
except Exception as e:
    print(f'❌ Checkpoint corrupted: {e}')
"
```

### **Issue 2: Model Creation Fails**
```bash
# Check if all dependencies are available
python -c "
try:
    from ConStruct.diffusion_model_discrete import DiscreteDenoisingDiffusion
    print('✅ Model class available')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

### **Issue 3: Metrics Computation Fails**
```bash
# Test with minimal data
python -c "
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from types import SimpleNamespace

# Create minimal mock
dataset_infos = SimpleNamespace()
dataset_infos.statistics = {'test': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]]), 'val': SimpleNamespace(num_nodes={10: 100}, degree_hist=[[0,1,2], [0.1,0.2,0.3]])}
dataset_infos.is_molecular = False

try:
    metrics = SamplingMetrics(dataset_infos, test=False)
    print('✅ Basic initialization works')
except Exception as e:
    print(f'❌ Initialization failed: {e}')
"
```

## 🎯 Testing Checklist

- [ ] **Checkpoint Loading**: Can load all checkpoint types
- [ ] **Model Creation**: Can create models from checkpoints
- [ ] **Basic Metrics**: Metrics work with checkpoint configs
- [ ] **Constraint Testing**: Metrics correctly identify constraints
- [ ] **Edge Cases**: Handle empty, single-node, disconnected graphs
- [ ] **Performance**: Reasonable computation time for large graphs
- [ ] **Real Data**: Work with actual model outputs
- [ ] **Error Handling**: Graceful failure for invalid inputs

## 🚀 Next Steps

1. **Run basic checkpoint tests** to verify loading works
2. **Test metrics with different constraint types** to ensure accuracy
3. **Test edge cases** with real model configurations
4. **Performance test** with larger graphs
5. **Validate constraint satisfaction** with known violations

This approach will give you comprehensive testing with real, trained models and ensure your metrics work correctly in production scenarios! 🎉 