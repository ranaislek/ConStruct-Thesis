#!/usr/bin/env python3
"""
Test that simulates how RingCountProjector is used in the actual diffusion process.
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

def create_test_data(batch_size=1, n_nodes=4):
    """Create test data with no edges (noise distribution)."""
    X = torch.zeros(batch_size, n_nodes, 1, dtype=torch.long)
    E = torch.zeros(batch_size, n_nodes, n_nodes, 2)  # 2 edge types (no edge, single bond)
    charges = torch.zeros(batch_size, n_nodes, dtype=torch.long)
    y = torch.zeros(batch_size, 1)
    node_mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    
    z_t = PlaceHolder(X=X, E=E, charges=charges, y=y)
    z_t.node_mask = node_mask
    return z_t

def test_diffusion_style():
    """Test RingCountProjector in a way that simulates the diffusion process."""
    
    print("Testing RingCountProjector in diffusion style...")
    
    # Start with no edges (noise distribution)
    z_t = create_test_data()
    projector = RingCountProjector(z_t, max_rings=0)
    
    # Simulate diffusion process: add edges one by one
    z_s = z_t.copy()
    
    print("Step 1: Adding edge 0-1")
    z_s.E[0, 0, 1, 1] = 1
    z_s.E[0, 1, 0, 1] = 1
    projector.project(z_s)
    print(f"After step 1: edge 0-1 = {z_s.E[0, 0, 1, 1]}")
    
    print("Step 2: Adding edge 1-2")
    z_s.E[0, 1, 2, 1] = 1
    z_s.E[0, 2, 1, 1] = 1
    projector.project(z_s)
    print(f"After step 2: edge 1-2 = {z_s.E[0, 1, 2, 1]}")
    
    print("Step 3: Adding edge 2-0 (this would create a ring)")
    z_s.E[0, 2, 0, 1] = 1
    z_s.E[0, 0, 2, 1] = 1
    projector.project(z_s)
    print(f"After step 3: edge 2-0 = {z_s.E[0, 2, 0, 1]}")
    
    # Check if the ring-creating edge was blocked
    if z_s.E[0, 2, 0, 1] == 0:
        print("✅ RingCountProjector correctly blocked ring formation in diffusion style!")
        return True
    else:
        print("❌ RingCountProjector failed to block ring formation in diffusion style!")
        return False

if __name__ == "__main__":
    test_diffusion_style() 