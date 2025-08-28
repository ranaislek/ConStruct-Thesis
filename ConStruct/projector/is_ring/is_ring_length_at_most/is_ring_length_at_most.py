###############################################################################
#
# Ring length constraint functionality for molecular graph generation
#
###############################################################################

from __future__ import annotations

import networkx as nx
from typing import Iterable, List, Set, Tuple
from ConStruct.projector.graph_cycles import (
    enumerate_simple_cycles_unique,
    max_simple_cycle_length,
)

__all__ = [
    "has_rings_of_length_at_most",
    "ring_length_at_most_projector",
    "get_max_ring_length_at_most",
]


def has_rings_of_length_at_most(graph, max_length):
    """Return True if every simple cycle length ≤ max_length (structural constraint)."""
    max_nodes_cap = max_length + 8
    return max_simple_cycle_length(graph, max_len=max_nodes_cap) <= max_length


def ring_length_at_most_projector(graph, max_length):
    """
    Edge-Deletion Projector: Enforces structural constraint: max ring length ≤ max_length.
    This removes edges from the largest rings first, but tries to preserve smaller rings.
    """
    while True:
        max_len = max_simple_cycle_length(graph)
        if max_len == 0:
            break
        if max_len <= max_length:
            break

        for cyc in enumerate_simple_cycles_unique(graph):
            if len(cyc) == max_len:
                u, v = cyc[0], cyc[1]
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                else:
                    for i in range(len(cyc)):
                        a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                        if graph.has_edge(a, b):
                            graph.remove_edge(a, b)
                            break
                break
    return graph


def ring_length_at_most_projector_improved(graph, max_length):
    return ring_length_at_most_projector(graph, max_length)


def get_max_ring_length_at_most(graph):
    return max_simple_cycle_length(graph) 