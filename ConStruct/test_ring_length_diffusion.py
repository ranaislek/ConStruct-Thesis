#!/usr/bin/env python3
"""
Test the ring length constraint in a diffusion-style sampling process.
This simulates how the constraint works during actual molecule generation.
"""

import sys
import torch
import networkx as nx
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ConStruct.projector.projector_utils import RingLengthProjector
from ConStruct.utils import PlaceHolder

def create_noise_state(batch_size=1, n_nodes=6):
    """Create a noisy state (like z_T in diffusion)."""
    X = torch.zeros(batch_size, n_nodes, 1, dtype=torch.long)
    E = torch.zeros(batch_size, n_nodes, n_nodes, 1)  # Single edge type
    charges = torch.zeros(batch_size, n_nodes, dtype=torch.long)
    y = torch.zeros(batch_size, 1)
    node_mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    
    z_t = PlaceHolder(X=X, E=E, charges=charges, y=y)
    z_t.node_mask = node_mask
    return z_t

def simulate_diffusion_step(z_t, z_s, projector):
    """Simulate one diffusion step with projection."""
    print(f"  Current graph edges: {list(z_t.E[0].nonzero(as_tuple=False).cpu().numpy())}")
    print(f"  Proposed new edges: {list(z_s.E[0].nonzero(as_tuple=False).cpu().numpy())}")
    
    # Apply projection
    projector.project(z_s)
    
    # Check what edges remain
    final_edges = list(z_s.E[0].nonzero(as_tuple=False).cpu().numpy())
    print(f"  Final edges after projection: {final_edges}")
    
    # Convert to NetworkX for analysis
    adj_matrix = (z_s.E[0, :, :, 0] > 0.5).cpu().numpy()
    G = nx.from_numpy_array(adj_matrix)
    cycles = nx.cycle_basis(G)
    max_cycle_length = max([len(cycle) for cycle in cycles]) if cycles else 0
    
    print(f"  Cycles: {cycles}")
    print(f"  Max cycle length: {max_cycle_length}")
    print()
    
    return z_s

def test_ring_length_in_diffusion():
    """Test ring length constraint during diffusion sampling."""
    print("Testing ring length constraint in diffusion sampling...")
    
    # Test parameters
    n_nodes = 6
    max_ring_length = 4  # No rings larger than 4 nodes
    
    # Start with noise (no edges)
    z_t = create_noise_state(n_nodes=n_nodes)
    projector = RingLengthProjector(z_t, max_ring_length=max_ring_length)
    
    print(f"Starting with {n_nodes} nodes, max_ring_length={max_ring_length}")
    print("=" * 50)
    
    # Simulate diffusion steps - gradually add edges
    edges_to_add = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]  # Would create a 6-cycle
    
    for i, (u, v) in enumerate(edges_to_add):
        print(f"Step {i+1}: Adding edge {u}-{v}")
        
        # Create z_s with the new edge
        z_s = z_t.copy()
        z_s.E[0, u, v] = torch.tensor([1.0])
        z_s.E[0, v, u] = torch.tensor([1.0])
        
        # Apply projection
        z_s = simulate_diffusion_step(z_t, z_s, projector)
        
        # Update z_t for next step
        z_t = z_s.copy()
        projector = RingLengthProjector(z_t, max_ring_length=max_ring_length)
    
    # Final analysis
    print("Final analysis:")
    adj_matrix = (z_t.E[0, :, :, 0] > 0.5).cpu().numpy()
    G = nx.from_numpy_array(adj_matrix)
    cycles = nx.cycle_basis(G)
    max_cycle_length = max([len(cycle) for cycle in cycles]) if cycles else 0
    
    print(f"Final graph edges: {list(G.edges())}")
    print(f"Final cycles: {cycles}")
    print(f"Max cycle length: {max_cycle_length}")
    
    # Check constraint satisfaction
    constraint_satisfied = max_cycle_length <= max_ring_length
    print(f"Constraint satisfied (max_cycle_length <= {max_ring_length}): {constraint_satisfied}")
    
    if constraint_satisfied:
        print("✅ Ring length constraint working correctly in diffusion!")
        return True
    else:
        print("❌ Ring length constraint failed in diffusion!")
        return False

def test_multiple_scenarios():
    """Test multiple scenarios with different constraints."""
    print("Testing multiple ring length constraint scenarios...")
    print("=" * 60)
    
    scenarios = [
        (6, 3, "No triangles allowed"),
        (6, 4, "No 4+ cycles allowed"), 
        (6, 5, "No 5+ cycles allowed"),
        (8, 6, "No 6+ cycles allowed"),
    ]
    
    all_passed = True
    
    for n_nodes, max_ring_length, description in scenarios:
        print(f"\nScenario: {description}")
        print(f"Nodes: {n_nodes}, Max ring length: {max_ring_length}")
        print("-" * 40)
        
        # Create test edges that would violate the constraint
        edges_to_add = [(i, (i+1) % n_nodes) for i in range(n_nodes)]  # Creates n_nodes-cycle
        
        # Start with noise
        z_t = create_noise_state(n_nodes=n_nodes)
        projector = RingLengthProjector(z_t, max_ring_length=max_ring_length)
        
        # Add edges one by one
        for i, (u, v) in enumerate(edges_to_add):
            z_s = z_t.copy()
            z_s.E[0, u, v] = torch.tensor([1.0])
            z_s.E[0, v, u] = torch.tensor([1.0])
            
            projector.project(z_s)
            z_t = z_s.copy()
            projector = RingLengthProjector(z_t, max_ring_length=max_ring_length)
        
        # Check final result
        adj_matrix = (z_t.E[0, :, :, 0] > 0.5).cpu().numpy()
        G = nx.from_numpy_array(adj_matrix)
        cycles = nx.cycle_basis(G)
        max_cycle_length = max([len(cycle) for cycle in cycles]) if cycles else 0
        
        constraint_satisfied = max_cycle_length <= max_ring_length
        print(f"Final max cycle length: {max_cycle_length}")
        print(f"Constraint satisfied: {constraint_satisfied}")
        
        if not constraint_satisfied:
            all_passed = False
            print("❌ FAILED")
        else:
            print("✅ PASSED")
    
    if all_passed:
        print("\n🎉 All scenarios passed! Ring length constraint working correctly.")
    else:
        print("\n❌ Some scenarios failed!")
    
    return all_passed

if __name__ == "__main__":
    print("Testing Ring Length Constraint in Diffusion Process")
    print("=" * 60)
    
    # Test 1: Basic diffusion simulation
    test1_passed = test_ring_length_in_diffusion()
    
    print("\n" + "=" * 60)
    
    # Test 2: Multiple scenarios
    test2_passed = test_multiple_scenarios()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Basic diffusion test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Multiple scenarios test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Ring length constraint is working correctly in diffusion.")
    else:
        print("\n❌ Some tests failed! Ring length constraint needs fixing.") 