import torch
import networkx as nx
import numpy as np
from ConStruct.projector.is_ring_count import (
    has_rings_of_length_at_most,
    ring_length_projector,
    get_max_ring_length,
)
from ConStruct.projector.projector_utils import RingLengthProjector, get_adj_matrix
from ConStruct.utils import PlaceHolder


def test_ring_length_functions():
    """Test the core ring length functions."""
    print("Testing core ring length functions...")
    
    # Test 1: Graph with no rings
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3)])
    assert has_rings_of_length_at_most(G1, 3) == True
    assert get_max_ring_length(G1) == 0
    print("✓ No rings test passed")
    
    # Test 2: Graph with small ring (triangle)
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 0)])
    assert has_rings_of_length_at_most(G2, 3) == True
    assert has_rings_of_length_at_most(G2, 2) == False
    assert get_max_ring_length(G2) == 3
    print("✓ Triangle test passed")
    
    # Test 3: Graph with larger ring
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    assert has_rings_of_length_at_most(G3, 4) == False
    assert has_rings_of_length_at_most(G3, 5) == True
    assert get_max_ring_length(G3) == 5
    print("✓ 5-cycle test passed")
    
    # Test 4: Graph with multiple rings
    G4 = nx.Graph()
    G4.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 2)])
    # Has a triangle (length 3) and a 4-cycle (length 4)
    assert has_rings_of_length_at_most(G4, 3) == False
    assert has_rings_of_length_at_most(G4, 4) == True
    assert get_max_ring_length(G4) == 4
    print("✓ Multiple rings test passed")
    
    print("All core ring length function tests passed!\n")


def test_ring_length_projector():
    """Test the ring length projector function."""
    print("Testing ring length projector...")
    
    # Test 1: Projector should break large rings
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])  # 5-cycle
    assert get_max_ring_length(G) == 5
    
    G_projected = ring_length_projector(G.copy(), max_length=4)
    assert get_max_ring_length(G_projected) <= 4
    assert len(G_projected.edges()) < len(G.edges())  # Should have removed at least one edge
    print("✓ Large ring projection test passed")
    
    # Test 2: Projector should not affect small rings
    G_small = nx.Graph()
    G_small.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle
    G_small_projected = ring_length_projector(G_small.copy(), max_length=3)
    assert get_max_ring_length(G_small_projected) == 3
    assert len(G_small_projected.edges()) == len(G_small.edges())  # Should not remove edges
    print("✓ Small ring preservation test passed")
    
    # Test 3: Projector should handle multiple large rings
    G_multi = nx.Graph()
    G_multi.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # 5-cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5)   # Another 5-cycle
    ])
    assert get_max_ring_length(G_multi) == 5
    
    G_multi_projected = ring_length_projector(G_multi.copy(), max_length=4)
    assert get_max_ring_length(G_multi_projected) <= 4
    print("✓ Multiple large rings projection test passed")
    
    print("All ring length projector tests passed!\n")


def test_ring_length_constraint_generic(n_nodes, max_ring_length, edges_to_add, expected_blocked_edges):
    """Generic test for ring length constraint with any number of nodes and any constraint."""
    print(f"\nTesting ring length constraint: {n_nodes} nodes, max_ring_length={max_ring_length}")
    
    # Create initial state with n_nodes
    z_t = PlaceHolder(
        X=torch.zeros((1, n_nodes, 1)),
        E=torch.zeros((1, n_nodes, n_nodes, 1)),
        y=torch.zeros((1, 1)),
        node_mask=torch.ones((1, n_nodes)),
    )
    
    for i, (u, v) in enumerate(edges_to_add):
        print(f"Adding edge {u}-{v}...")
        # z_t: all previous edges (not including the new one)
        z_t_step = z_t.copy()
        for j in range(i):
            uu, vv = edges_to_add[j]
            z_t_step.E[0, uu, vv] = torch.tensor([1.0])
            z_t_step.E[0, vv, uu] = torch.tensor([1.0])
        # z_s: copy of z_t, with the new edge added
        z_s = z_t_step.copy()
        z_s.E[0, u, v] = torch.tensor([1.0])
        z_s.E[0, v, u] = torch.tensor([1.0])
        projector = RingLengthProjector(z_t_step, max_ring_length=max_ring_length)
        projector.project(z_s)
        edge_exists = z_s.E[0, u, v, 0] > 0.5
        
        should_be_blocked = (u, v) in expected_blocked_edges
        if should_be_blocked:
            assert not edge_exists, f"Edge {u}-{v} should be blocked (would create ring > {max_ring_length})"
            print(f"  ✓ Edge {u}-{v} blocked (would create ring > {max_ring_length})")
        else:
            assert edge_exists, f"Edge {u}-{v} should be allowed"
            print(f"  ✓ Edge {u}-{v} allowed")
    
    print(f"✓ Ring length constraint test passed for {n_nodes} nodes, max_ring_length={max_ring_length}")


def test_ring_length_projector_class():
    """Test the RingLengthProjector class with generic scenarios."""
    print("Testing RingLengthProjector class...")
    
    # Test 1: 4 nodes, max_ring_length=3 (should block 4-cycle)
    edges_4cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    expected_blocked_4cycle = [(3, 0)]  # Last edge creates 4-cycle
    test_ring_length_constraint_generic(4, 3, edges_4cycle, expected_blocked_4cycle)
    
    # Test 2: 5 nodes, max_ring_length=4 (should block 5-cycle)
    edges_5cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    expected_blocked_5cycle = [(4, 0)]  # Last edge creates 5-cycle
    test_ring_length_constraint_generic(5, 4, edges_5cycle, expected_blocked_5cycle)
    
    # Test 3: 6 nodes, max_ring_length=5 (should block 6-cycle)
    edges_6cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    expected_blocked_6cycle = [(5, 0)]  # Last edge creates 6-cycle
    test_ring_length_constraint_generic(6, 5, edges_6cycle, expected_blocked_6cycle)
    
    print("All RingLengthProjector class tests passed!\n")


if __name__ == "__main__":
    # Minimal test for has_rings_of_length_at_most
    print("\nMinimal test for has_rings_of_length_at_most:")
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    print("Cycle basis:", nx.cycle_basis(G))
    print("has_rings_of_length_at_most(G, 3):", has_rings_of_length_at_most(G, 3))
    print("has_rings_of_length_at_most(G, 4):", has_rings_of_length_at_most(G, 4))

    print("Running ring length constraint tests...\n")
    
    test_ring_length_functions()
    test_ring_length_projector()
    test_ring_length_projector_class()
    
    print("🎉 All ring length constraint tests passed!") 