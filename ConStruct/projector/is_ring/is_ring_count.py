###############################################################################
#
# Ring count constraint functionality for molecular graph generation
#
###############################################################################

import networkx as nx

__all__ = ["has_at_most_n_rings", "ring_count_projector", "count_rings"]


def has_at_most_n_rings(graph, n):
    """Return True if the graph has at most n rings (cycles)."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) <= n


def ring_count_projector(graph, max_rings):
    """
    Projector: If the graph has more than max_rings, remove edges to break cycles.
    This is a robust approach: remove edges from all cycles until compliant.
    """
    while True:
        cycles = nx.cycle_basis(graph)
        # print(f"[DEBUG] Cycles before removal: {cycles}")
        if len(cycles) <= max_rings:
            break
        # Remove an edge from each cycle until the constraint is satisfied
        for cycle in cycles:
            if len(nx.cycle_basis(graph)) <= max_rings:
                break
            # Remove the first edge in the cycle
            edge_to_remove = (cycle[0], cycle[1])
            # print(f"[DEBUG] Removing edge: {edge_to_remove}")
            if graph.has_edge(*edge_to_remove):
                graph.remove_edge(*edge_to_remove)
            else:
                # Try the next edge in the cycle
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1)%len(cycle)]
                    if graph.has_edge(u, v):
                        # print(f"[DEBUG] Removing edge: {(u, v)}")
                        graph.remove_edge(u, v)
                        break
        # print(f"[DEBUG] Cycles after removal: {nx.cycle_basis(graph)}")
    return graph


def count_rings(graph):
    """Return the number of rings (cycles) in the graph."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) 