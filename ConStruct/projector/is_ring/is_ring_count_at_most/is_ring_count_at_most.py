###############################################################################
#
# Ring count constraint functionality for molecular graph generation
#
###############################################################################

import networkx as nx
from ConStruct.projector.graph_cycles import (
    count_simple_cycles,
    enumerate_simple_cycles_unique,
)

__all__ = ["has_at_most_n_rings", "ring_count_at_most_projector", "count_rings_at_most"]


def has_at_most_n_rings(graph, n):
    """Return True if the graph has ring count ≤ n (counts ALL unique simple cycles)."""
    return count_simple_cycles(graph) <= n


def ring_count_at_most_projector(graph, max_rings):
    """
    Edge-Deletion Projector: Enforces structural constraint: total number of unique simple
    cycles ≤ max_rings. Removes edges from cycles until count ≤ max_rings.
    """
    while True:
        total = count_simple_cycles(graph)
        if total <= max_rings:
            break

        # Break one edge from one of the cycles (heuristic: last enumerated)
        cycles = list(enumerate_simple_cycles_unique(graph))
        if not cycles:
            break
        cyc = cycles[-1]
        edge_to_remove = (cyc[0], cyc[1])
        if graph.has_edge(*edge_to_remove):
            graph.remove_edge(*edge_to_remove)
        else:
            for i in range(len(cyc)):
                u, v = cyc[i], cyc[(i + 1) % len(cyc)]
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                    break
    return graph


def count_rings_at_most(graph):
    """Return the total number of unique simple cycles in the graph for 'at most' constraints."""
    return count_simple_cycles(graph) 