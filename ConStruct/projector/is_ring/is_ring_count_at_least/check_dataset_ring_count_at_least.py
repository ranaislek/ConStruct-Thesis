#!/usr/bin/env python3
"""
Check dataset for ring count "at least" constraints.
This script analyzes the QM9 dataset to understand ring count distributions
for edge-insertion constraints (at least N rings).
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


def analyze_ring_count_at_least_distribution():
    """Analyze the distribution of ring counts in QM9 dataset for 'at least' constraints."""
    
    print("🔍 ANALYZING RING COUNT 'AT LEAST' DISTRIBUTION")
    print("=" * 60)
    
    # Load QM9 dataset
    dataset = QM9Dataset(
        root='data/qm9',
        split='train',
        remove_h=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Analyze ring counts
    ring_counts = []
    valid_molecules = 0
    
    for i, data in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing molecule {i}/{len(dataset)}")
        
        # Convert to NetworkX graph
        adj_matrix = get_adj_matrix(data)
        nx_graph = nx.from_numpy_array(adj_matrix.numpy())
        
        # Count rings
        cycles = nx.cycle_basis(nx_graph)
        ring_count = len(cycles)
        ring_counts.append(ring_count)
        
        if ring_count > 0:  # Has at least 1 ring
            valid_molecules += 1
    
    ring_counts = np.array(ring_counts)
    
    print(f"\n📊 RING COUNT DISTRIBUTION:")
    print(f"   Total molecules: {len(ring_counts)}")
    print(f"   Molecules with at least 1 ring: {valid_molecules} ({valid_molecules/len(ring_counts)*100:.1f}%)")
    
    # Analyze different "at least" thresholds
    thresholds = [1, 2, 3, 4, 5]
    for threshold in thresholds:
        count = np.sum(ring_counts >= threshold)
        percentage = count / len(ring_counts) * 100
        print(f"   Molecules with at least {threshold} rings: {count} ({percentage:.1f}%)")
    
    print(f"\n📈 STATISTICS:")
    print(f"   Mean ring count: {np.mean(ring_counts):.2f}")
    print(f"   Median ring count: {np.median(ring_counts):.2f}")
    print(f"   Min ring count: {np.min(ring_counts)}")
    print(f"   Max ring count: {np.max(ring_counts)}")
    print(f"   Std ring count: {np.std(ring_counts):.2f}")
    
    # Save results
    output_dir = Path("ConStruct/projector/is_ring/is_ring_count_at_least/qm9_ring_count_at_least_violations")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "ring_counts.npy", ring_counts)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    return ring_counts


def test_ring_count_at_least_constraints():
    """Test the ring count 'at least' constraint functions."""
    
    print("\n🧪 TESTING RING COUNT 'AT LEAST' CONSTRAINTS")
    print("=" * 60)
    
    from ConStruct.projector.is_ring.is_ring_count_at_least import (
        has_at_least_n_rings,
        ring_count_at_least_projector,
        count_rings_at_least
    )
    
    # Test 1: Simple triangle (1 ring)
    print("Test 1: Triangle (1 ring)")
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    assert has_at_least_n_rings(triangle, 1) == True
    assert has_at_least_n_rings(triangle, 2) == False
    assert count_rings_at_least(triangle) == 1
    print("✅ Triangle test passed")
    
    # Test 2: Two triangles (2 rings)
    print("Test 2: Two triangles (2 rings)")
    two_triangles = nx.Graph()
    two_triangles.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
    
    assert has_at_least_n_rings(two_triangles, 1) == True
    assert has_at_least_n_rings(two_triangles, 2) == True
    assert has_at_least_n_rings(two_triangles, 3) == False
    assert count_rings_at_least(two_triangles) == 2
    print("✅ Two triangles test passed")
    
    # Test 3: No rings
    print("Test 3: No rings")
    no_rings = nx.Graph()
    no_rings.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    assert has_at_least_n_rings(no_rings, 1) == False
    assert count_rings_at_least(no_rings) == 0
    print("✅ No rings test passed")
    
    # Test 4: Projector functionality
    print("Test 4: Projector functionality")
    test_graph = nx.Graph()
    test_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])  # No rings
    
    # Should add edges to create at least 1 ring
    modified_graph = ring_count_at_least_projector(test_graph.copy(), 1)
    assert count_rings_at_least(modified_graph) >= 1
    print("✅ Projector test passed")
    
    print("\n🎉 All ring count 'at least' tests passed!")


if __name__ == "__main__":
    # Analyze dataset
    ring_counts = analyze_ring_count_at_least_distribution()
    
    # Test constraints
    test_ring_count_at_least_constraints() 