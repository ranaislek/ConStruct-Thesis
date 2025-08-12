###############################################################################
#
# Ring count constraint functionality for molecular graph generation
#
###############################################################################

import networkx as nx

__all__ = ["has_at_most_n_rings", "ring_count_at_most_projector", "count_rings_at_most"]


def has_at_most_n_rings(graph, n):
    """Return True if the graph has cycle rank ≤ n (structural constraint)."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) <= n


def ring_count_at_most_projector(graph, max_rings):
    """
    Edge-Deletion Projector: Enforces structural constraint: cycle rank (basis size) ≤ max_rings.
    GENTLE APPROACH: Only remove edges from excess cycles, preserve as many cycles as possible.
    """
    while True:
        cycles = nx.cycle_basis(graph)
        if len(cycles) <= max_rings:
            break
        
        # Only remove edges from EXCESS rings (not all rings!)
        excess_rings = len(cycles) - max_rings
        rings_to_break = cycles[-excess_rings:]  # Take the last excess rings
        
        for cycle in rings_to_break:
            if len(nx.cycle_basis(graph)) <= max_rings:
                break
            # Remove the first edge in the cycle
            edge_to_remove = (cycle[0], cycle[1])
            if graph.has_edge(*edge_to_remove):
                graph.remove_edge(*edge_to_remove)
            else:
                # Try the next edge in the cycle
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1)%len(cycle)]
                    if graph.has_edge(u, v):
                        graph.remove_edge(u, v)
                        break
    return graph


def count_rings_at_most(graph):
    """Return the cycle rank (basis size) in the graph for 'at most' constraints."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) 