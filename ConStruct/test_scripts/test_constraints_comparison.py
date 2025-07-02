#!/usr/bin/env python3
"""
Comprehensive test script to compare different constraints on QM9 dataset.
This will help evaluate how well our ring constraints work compared to planarity.
"""

import sys
import torch
import networkx as nx
import numpy as np
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ConStruct.projector.projector_utils import RingCountProjector, RingLengthProjector, PlanarProjector
from ConStruct.utils import PlaceHolder

def create_qm9_like_data(batch_size=4, n_nodes=9):
    """Create data similar to QM9 molecules."""
    X = torch.zeros(batch_size, n_nodes, 1, dtype=torch.long)
    E = torch.zeros(batch_size, n_nodes, n_nodes, 1)  # Single edge type
    charges = torch.zeros(batch_size, n_nodes, dtype=torch.long)
    y = torch.zeros(batch_size, 1)
    node_mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    
    z_t = PlaceHolder(X=X, E=E, charges=charges, y=y)
    z_t.node_mask = node_mask
    return z_t

def test_constraint_performance(constraint_type, **kwargs):
    """Test the performance of a specific constraint."""
    print(f"\n{'='*60}")
    print(f"Testing {constraint_type} constraint")
    print(f"{'='*60}")
    
    # Create test data
    z_t = create_qm9_like_data()
    
    # Create projector based on constraint type
    if constraint_type == "ring_count":
        projector = RingCountProjector(z_t, max_rings=kwargs.get('max_rings', 0))
    elif constraint_type == "ring_length":
        projector = RingLengthProjector(z_t, max_ring_length=kwargs.get('max_ring_length', 4))
    elif constraint_type == "planar":
        projector = PlanarProjector(z_t)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")
    
    # Test edge addition performance
    start_time = time.time()
    edges_blocked = 0
    edges_allowed = 0
    
    # Simulate adding edges randomly (like in diffusion)
    for i in range(20):  # Test 20 edge additions
        z_s = z_t.copy()
        
        # Add a random edge
        u, v = np.random.choice(9, 2, replace=False)
        z_s.E[0, u, v] = torch.tensor([1.0])
        z_s.E[0, v, u] = torch.tensor([1.0])
        
        # Apply constraint
        projector.project(z_s)
        
        # Check if edge was blocked
        edge_exists = z_s.E[0, u, v, 0] > 0.5
        if edge_exists:
            edges_allowed += 1
        else:
            edges_blocked += 1
        
        # Update z_t for next iteration
        z_t = z_s.copy()
        projector = type(projector)(z_t, **kwargs)
    
    end_time = time.time()
    
    # Analyze final graph
    adj_matrix = (z_t.E[0, :, :, 0] > 0.5).cpu().numpy()
    G = nx.from_numpy_array(adj_matrix)
    cycles = nx.cycle_basis(G)
    max_cycle_length = max([len(cycle) for cycle in cycles]) if cycles else 0
    
    print(f"Performance Results:")
    print(f"  Time taken: {end_time - start_time:.4f} seconds")
    print(f"  Edges allowed: {edges_allowed}")
    print(f"  Edges blocked: {edges_blocked}")
    print(f"  Blocking rate: {edges_blocked/(edges_allowed+edges_blocked)*100:.1f}%")
    print(f"  Final graph edges: {len(G.edges())}")
    print(f"  Final cycles: {len(cycles)}")
    print(f"  Max cycle length: {max_cycle_length}")
    
    return {
        'time': end_time - start_time,
        'edges_allowed': edges_allowed,
        'edges_blocked': edges_blocked,
        'blocking_rate': edges_blocked/(edges_allowed+edges_blocked)*100,
        'final_edges': len(G.edges()),
        'final_cycles': len(cycles),
        'max_cycle_length': max_cycle_length
    }

def compare_all_constraints():
    """Compare all available constraints."""
    print("Comprehensive Constraint Comparison Test")
    print("=" * 80)
    
    results = {}
    
    # Test Ring Count Constraint
    results['ring_count_0'] = test_constraint_performance("ring_count", max_rings=0)
    results['ring_count_1'] = test_constraint_performance("ring_count", max_rings=1)
    
    # Test Ring Length Constraint
    results['ring_length_3'] = test_constraint_performance("ring_length", max_ring_length=3)
    results['ring_length_4'] = test_constraint_performance("ring_length", max_ring_length=4)
    results['ring_length_5'] = test_constraint_performance("ring_length", max_ring_length=5)
    
    # Test Planarity Constraint
    results['planar'] = test_constraint_performance("planar")
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("CONSTRAINT COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Constraint':<20} {'Time(s)':<8} {'Block%':<8} {'Edges':<6} {'Cycles':<7} {'MaxLen':<7}")
    print(f"{'-'*80}")
    
    for name, result in results.items():
        print(f"{name:<20} {result['time']:<8.4f} {result['blocking_rate']:<8.1f} "
              f"{result['final_edges']:<6} {result['final_cycles']:<7} {result['max_cycle_length']:<7}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print(f"{'='*80}")
    
    # Find best performing constraints
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    most_blocking = max(results.items(), key=lambda x: x[1]['blocking_rate'])
    
    print(f"Fastest constraint: {fastest[0]} ({fastest[1]['time']:.4f}s)")
    print(f"Most restrictive: {most_blocking[0]} ({most_blocking[1]['blocking_rate']:.1f}% blocking)")
    
    # Ring count vs Ring length comparison
    ring_count_0 = results['ring_count_0']
    ring_length_3 = results['ring_length_3']
    
    print(f"\nRing Count (0) vs Ring Length (3) comparison:")
    print(f"  Ring Count: {ring_count_0['blocking_rate']:.1f}% blocking, {ring_count_0['final_cycles']} cycles")
    print(f"  Ring Length: {ring_length_3['blocking_rate']:.1f}% blocking, {ring_length_3['final_cycles']} cycles")
    
    return results

if __name__ == "__main__":
    print("Starting Constraint Comparison Test...")
    results = compare_all_constraints()
    print("\n🎉 Constraint comparison completed!") 
    
    
# to run:

"""
  cd /home/rislek/ConStruct-Thesis
  python test_scripts/test_my_module.py
  
"""