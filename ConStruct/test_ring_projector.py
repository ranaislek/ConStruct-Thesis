#!/usr/bin/env python3
"""
Simple unit test for RingCountProjector.
This tests the core logic without running the full training pipeline.
"""

import sys
import torch
import networkx as nx
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ConStruct.projector.projector_utils import RingCountProjector
from ConStruct.utils import PlaceHolder

def create_test_data(batch_size=2, n_nodes=4):
    """Create test data with no edges (noise distribution)."""
    X = torch.zeros(batch_size, n_nodes, 1, dtype=torch.long)  # Single atom type
    E = torch.zeros(batch_size, n_nodes, n_nodes, 2)  # 2 edge types (no edge, single bond)
    charges = torch.zeros(batch_size, n_nodes, dtype=torch.long)
    y = torch.zeros(batch_size, 1)
    node_mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    
    z_t = PlaceHolder(X=X, E=E, charges=charges, y=y)
    z_t.node_mask = node_mask
    return z_t

def test_ring_count_projector():
    """Test the RingCountProjector with simple test cases."""
    
    print("Testing RingCountProjector...")
    
    # Test with max_rings=0 (should block all rings)
    print("Testing with max_rings=0...")
    z_t = create_test_data()
    projector = RingCountProjector(z_t, max_rings=0)
    
    # Create a new state with edges that would create a ring
    z_s = z_t.copy()
    # Add edges to create a ring: 0-1-2-0
    z_s.E[0, 0, 1, 1] = 1  # Edge between nodes 0 and 1
    z_s.E[0, 1, 0, 1] = 1  # Undirected edge
    z_s.E[0, 1, 2, 1] = 1  # Edge between nodes 1 and 2
    z_s.E[0, 2, 1, 1] = 1  # Undirected edge
    z_s.E[0, 2, 0, 1] = 1  # This creates a ring
    z_s.E[0, 0, 2, 1] = 1  # This creates a ring
    
    print(f"Before projection - ring edge values: {z_s.E[0, 2, 0, 1]}, {z_s.E[0, 0, 2, 1]}")
    
    # Apply projection
    projector.project(z_s)
    
    print(f"After projection - ring edge values: {z_s.E[0, 2, 0, 1]}, {z_s.E[0, 0, 2, 1]}")
    
    # Check if the ring-creating edge was removed
    if z_s.E[0, 2, 0, 1] == 0 and z_s.E[0, 0, 2, 1] == 0:
        print("✅ RingCountProjector correctly blocked ring formation!")
    else:
        print("❌ RingCountProjector failed to block ring formation!")
        print(f"Expected: 0, 0. Got: {z_s.E[0, 2, 0, 1]}, {z_s.E[0, 0, 2, 1]}")
        return False
    
    # Test with max_rings=1 (should allow one ring)
    print("Testing with max_rings=1...")
    z_t = create_test_data()  # Fresh data
    projector = RingCountProjector(z_t, max_rings=1)
    
    # Create a new state with edges that would create a ring
    z_s = z_t.copy()
    # Add edges to create a ring: 0-1-2-0
    z_s.E[1, 0, 1, 1] = 1  # Edge between nodes 0 and 1
    z_s.E[1, 1, 0, 1] = 1  # Undirected edge
    z_s.E[1, 1, 2, 1] = 1  # Edge between nodes 1 and 2
    z_s.E[1, 2, 1, 1] = 1  # Undirected edge
    z_s.E[1, 2, 0, 1] = 1  # This creates a ring
    z_s.E[1, 0, 2, 1] = 1  # This creates a ring
    
    # Apply projection
    projector.project(z_s)
    
    # Check if the ring-creating edge was kept (since max_rings=1)
    if z_s.E[1, 2, 0, 1] == 1 and z_s.E[1, 0, 2, 1] == 1:
        print("✅ RingCountProjector correctly allowed ring formation with max_rings=1!")
    else:
        print("❌ RingCountProjector incorrectly blocked ring formation with max_rings=1!")
        return False
    
    print("✅ All RingCountProjector tests passed!")
    return True

if __name__ == "__main__":
    test_ring_count_projector() 