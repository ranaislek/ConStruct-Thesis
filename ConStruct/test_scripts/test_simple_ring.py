#!/usr/bin/env python3
"""
Simple test to verify the ring_count_projector function works.
"""

import networkx as nx
from ConStruct.projector.is_ring_count import ring_count_projector, count_rings

def test_ring_count_projector():
    """Test the ring_count_projector function directly."""
    
    print("Testing ring_count_projector function...")
    
    # Create a graph with a ring: 0-1-2-0
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    print(f"Original graph has {count_rings(G)} rings")
    print(f"Edges: {list(G.edges())}")
    
    # Apply ring count projector with max_rings=0
    G_projected = ring_count_projector(G.copy(), max_rings=0)
    
    print(f"After projection with max_rings=0: {count_rings(G_projected)} rings")
    print(f"Edges: {list(G_projected.edges())}")
    
    if count_rings(G_projected) == 0:
        print("✅ ring_count_projector correctly removed rings!")
    else:
        print("❌ ring_count_projector failed to remove rings!")
        return False
    
    return True

if __name__ == "__main__":
    test_ring_count_projector() 