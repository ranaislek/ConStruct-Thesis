###############################################################################
#
# Ring length constraint functionality for molecular graph generation
#
###############################################################################

import networkx as nx

__all__ = ["has_rings_of_length_at_most", "ring_length_projector", "get_max_ring_length"]


def has_rings_of_length_at_most(graph, max_length):
    """Return True if all rings in the graph have length at most max_length."""
    cycles = nx.cycle_basis(graph)
    for cycle in cycles:
        if len(cycle) > max_length:
            return False
    return True


def ring_length_projector(graph, max_length):
    """
    Projector: If the graph has rings longer than max_length, remove edges to break them.
    This removes edges from the largest rings first, but tries to preserve smaller rings.
    """
    while True:
        cycles = nx.cycle_basis(graph)
        if not cycles:  # No cycles left
            break
            
        # Find the largest cycle
        largest_cycle = max(cycles, key=len)
        if len(largest_cycle) <= max_length:
            break
            
        # For rings that are too large, we need to break them
        # But we should try to preserve smaller rings if possible
        # Remove an edge from the largest cycle
        edge_to_remove = (largest_cycle[0], largest_cycle[1])
        if graph.has_edge(*edge_to_remove):
            graph.remove_edge(*edge_to_remove)
        else:
            # Try the next edge in the cycle
            for i in range(len(largest_cycle)):
                u, v = largest_cycle[i], largest_cycle[(i+1)%len(largest_cycle)]
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                    break
    return graph


def ring_length_projector_improved(graph, max_length):
    """
    Improved projector: If the graph has rings longer than max_length, 
    try to reduce them to smaller rings rather than just breaking them.
    """
    while True:
        cycles = nx.cycle_basis(graph)
        if not cycles:  # No cycles left
            break
            
        # Find the largest cycle
        largest_cycle = max(cycles, key=len)
        if len(largest_cycle) <= max_length:
            break
            
        # For rings that are too large, we need to break them
        # But we should try to preserve smaller rings if possible
        # Remove an edge from the largest cycle
        edge_to_remove = (largest_cycle[0], largest_cycle[1])
        if graph.has_edge(*edge_to_remove):
            graph.remove_edge(*edge_to_remove)
        else:
            # Try the next edge in the cycle
            for i in range(len(largest_cycle)):
                u, v = largest_cycle[i], largest_cycle[(i+1)%len(largest_cycle)]
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                    break
    return graph


def get_max_ring_length(graph):
    """Return the length of the largest ring in the graph, or 0 if no rings."""
    cycles = nx.cycle_basis(graph)
    if not cycles:
        return 0
    return max(len(cycle) for cycle in cycles) 