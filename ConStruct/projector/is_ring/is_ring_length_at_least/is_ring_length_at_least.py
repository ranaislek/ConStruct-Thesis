###############################################################################
#
# Ring length "at least" constraint functionality for molecular graph generation
# This is the flipped mechanism of edge-deletion for edge-insertion transitions
#
###############################################################################

import networkx as nx

__all__ = ["has_rings_of_length_at_least", "ring_length_at_least_projector", "get_min_ring_length_at_least"]


def has_rings_of_length_at_least(graph, min_length):
    """Return True if the graph has at least one ring of length >= min_length."""
    cycles = nx.cycle_basis(graph)
    for cycle in cycles:
        if len(cycle) >= min_length:
            return True
    return False


def ring_length_at_least_projector(graph, min_length):
    """
    Edge-Insertion Projector: If the graph has no rings of length >= min_length, 
    add edges to create larger rings.
    SIMPLE APPROACH: Create a ring of exact length using existing nodes.
    """
    cycles = nx.cycle_basis(graph)
    
    # Check if we already have a ring of sufficient length
    for cycle in cycles:
        if len(cycle) >= min_length:
            return graph  # Already satisfies constraint
    
    # Need to create a ring of at least min_length
    nodes = list(graph.nodes())
    
    # Strategy: Create a ring of exactly min_length using existing nodes
    # This ensures we get a ring of the required length without tensor issues
    
    if len(nodes) >= min_length:
        # Create a cycle of min_length using the first min_length nodes
        for i in range(min_length):
            u = nodes[i]
            v = nodes[(i + 1) % min_length]
            graph.add_edge(u, v)
    else:
        # Not enough nodes to create a ring of min_length
        # Create the largest possible ring with available nodes
        if len(nodes) >= 3:
            for i in range(len(nodes)):
                u = nodes[i]
                v = nodes[(i + 1) % len(nodes)]
                graph.add_edge(u, v)
        else:
            # Not enough nodes for any ring, add edges to create a chain
            for i in range(len(nodes) - 1):
                graph.add_edge(nodes[i], nodes[i + 1])
    
    return graph


def get_min_ring_length_at_least(graph):
    """Return the length of the smallest ring in the graph for 'at least' constraints, or 0 if no rings."""
    cycles = nx.cycle_basis(graph)
    if not cycles:
        return 0
    return min(len(cycle) for cycle in cycles) 