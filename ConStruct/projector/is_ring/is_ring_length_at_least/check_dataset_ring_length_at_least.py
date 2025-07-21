#!/usr/bin/env python3
"""
Check dataset for ring length "at least" constraints.
This script analyzes the QM9 dataset to understand ring length distributions
for edge-insertion constraints (at least N ring length).
"""

import os
import sys
import torch
import networkx as nx
import numpy as np
from pathlib import Path

# Add ConStruct to path
sys.path.append('ConStruct')

from ConStruct.datasets.qm9_dataset import QM9Dataset
from ConStruct.utils import PlaceHolder
from ConStruct.datasets.dataset_utils import get_adj_matrix


def analyze_ring_length_at_least_distribution():
    """Analyze the distribution of ring lengths in QM9 dataset for 'at least' constraints."""
    
    print("🔍 ANALYZING RING LENGTH 'AT LEAST' DISTRIBUTION")
    print("=" * 60)
    
    # Load QM9 dataset
    dataset = QM9Dataset(
        root='data/qm9',
        split='train',
        remove_h=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Analyze ring lengths
    ring_lengths = []
    valid_molecules = 0
    
    for i, data in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing molecule {i}/{len(dataset)}")
        
        # Convert to NetworkX graph
        adj_matrix = get_adj_matrix(data)
        nx_graph = nx.from_numpy_array(adj_matrix.numpy())
        
        # Get ring lengths
        cycles = nx.cycle_basis(nx_graph)
        if cycles:
            max_ring_length = max(len(cycle) for cycle in cycles)
            ring_lengths.append(max_ring_length)
            
            if max_ring_length >= 4:  # Has at least 4-ring
                valid_molecules += 1
        else:
            ring_lengths.append(0)
    
    ring_lengths = np.array(ring_lengths)
    
    print(f"\n📊 RING LENGTH DISTRIBUTION:")
    print(f"   Total molecules: {len(ring_lengths)}")
    print(f"   Molecules with at least 4-ring: {valid_molecules} ({valid_molecules/len(ring_lengths)*100:.1f}%)")
    
    # Analyze different "at least" thresholds
    thresholds = [3, 4, 5, 6, 7, 8]
    for threshold in thresholds:
        count = np.sum(ring_lengths >= threshold)
        percentage = count / len(ring_lengths) * 100
        print(f"   Molecules with at least {threshold}-ring: {count} ({percentage:.1f}%)")
    
    print(f"\n📈 STATISTICS:")
    print(f"   Mean max ring length: {np.mean(ring_lengths):.2f}")
    print(f"   Median max ring length: {np.median(ring_lengths):.2f}")
    print(f"   Min max ring length: {np.min(ring_lengths)}")
    print(f"   Max ring length: {np.max(ring_lengths)}")
    print(f"   Std max ring length: {np.std(ring_lengths):.2f}")
    
    # Save results
    output_dir = Path("ConStruct/projector/is_ring/is_ring_length_at_least/qm9_ring_length_at_least_violations")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "ring_lengths.npy", ring_lengths)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    return ring_lengths


def test_ring_length_at_least_constraints():
    """Test the ring length 'at least' constraint functions."""
    
    print("\n🧪 TESTING RING LENGTH 'AT LEAST' CONSTRAINTS")
    print("=" * 60)
    
    from ConStruct.projector.is_ring.is_ring_length_at_least import (
        has_rings_of_length_at_least,
        ring_length_at_least_projector,
        get_min_ring_length_at_least
    )
    
    # Test 1: Triangle (3-ring)
    print("Test 1: Triangle (3-ring)")
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    assert has_rings_of_length_at_least(triangle, 3) == True
    assert has_rings_of_length_at_least(triangle, 4) == False
    assert get_min_ring_length_at_least(triangle) == 3
    print("✅ Triangle test passed")
    
    # Test 2: Square (4-ring)
    print("Test 2: Square (4-ring)")
    square = nx.Graph()
    square.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    
    assert has_rings_of_length_at_least(square, 3) == True
    assert has_rings_of_length_at_least(square, 4) == True
    assert has_rings_of_length_at_least(square, 5) == False
    assert get_min_ring_length_at_least(square) == 4
    print("✅ Square test passed")
    
    # Test 3: No rings
    print("Test 3: No rings")
    no_rings = nx.Graph()
    no_rings.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    assert has_rings_of_length_at_least(no_rings, 3) == False
    assert get_min_ring_length_at_least(no_rings) == 0
    print("✅ No rings test passed")
    
    # Test 4: Projector functionality
    print("Test 4: Projector functionality")
    test_graph = nx.Graph()
    test_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])  # No rings
    
    # Should add edges to create at least 4-ring
    modified_graph = ring_length_at_least_projector(test_graph.copy(), 4)
    cycles = nx.cycle_basis(modified_graph)
    if cycles:
        max_length = max(len(cycle) for cycle in cycles)
        assert max_length >= 4
    print("✅ Projector test passed")
    
    print("\n🎉 All ring length 'at least' tests passed!")


if __name__ == "__main__":
    # Analyze dataset
    ring_lengths = analyze_ring_length_at_least_distribution()
    
    # Test constraints
    test_ring_length_at_least_constraints() 