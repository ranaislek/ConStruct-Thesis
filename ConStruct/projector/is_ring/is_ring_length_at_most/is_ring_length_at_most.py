###############################################################################
#
# Ring length constraint functionality for molecular graph generation
#
###############################################################################

from __future__ import annotations

import networkx as nx
from typing import Iterable, List, Set, Tuple
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, max_ring_length_exceeds, fast_ring_length_prescreen

__all__ = [
    "has_rings_of_length_at_most",
    "ring_length_at_most_projector",
    "get_max_ring_length_at_most",
    "ring_length_at_most_sanitize",
]


def has_rings_of_length_at_most(G, L: int) -> bool:
    """
    FIXED: Check if all rings/cycles in graph have length at most L.
    
    CRITICAL FIX: Always uses full cycle enumeration for 100% correctness.
    Removed unreliable fast pre-screening that was causing constraint violations.
    
    This ensures the same 100% constraint satisfaction as planarity and ring count projectors.
    
    Args:
        G: NetworkX graph
        L: Maximum allowed cycle length
        
    Returns:
        True if all cycles have length ≤ L, False otherwise
        
    Performance: O(exponential) for complex fused ring systems, but guaranteed correctness
    """
    # OPTION 1: Always use full validation for guaranteed correctness
    # This matches the approach used by working projectors (planarity, ring count)
    return not max_ring_length_exceeds(G, L)
    
    # OPTION 2: Use fixed fast pre-screening (commented out for maximum safety)
    # Uncomment only after extensive testing confirms no false positives
    #
    # prescreen_result = fast_ring_length_prescreen(G, L)
    # if prescreen_result is not None:
    #     return prescreen_result  # Only when mathematically certain
    # 
    # # Complex case: requires full cycle enumeration for correctness
    # return not max_ring_length_exceeds(G, L)


def ring_length_at_most_projector(graph, max_length):
    """
    Edge-Deletion Projector: Enforces structural constraint: max ring length ≤ max_length.
    
    DEPRECATED: Use ring_length_at_most_sanitize instead for offline cleanup.
    This function is kept for backward compatibility.
    """
    return ring_length_at_most_sanitize(graph, max_length)


def ring_length_at_most_sanitize(G, L: int):
    """
    Sanitizer: Remove one edge on a longest cycle until all cycles <= L
    """
    def find_longest_cycle():
        best = None
        for cyc in enumerate_simple_cycles_unique(G):
            if best is None or len(cyc) > len(best):
                best = cyc
        return best

    longest = find_longest_cycle()
    while longest and len(longest) > L:
        # remove an edge from that cycle; simple heuristic
        a, b = longest[0], longest[1]
        if G.has_edge(a, b):
            G.remove_edge(a, b)
        else:
            # fallback: remove first existing edge along the cycle
            for i in range(len(longest)):
                x, y = longest[i], longest[(i+1) % len(longest)]
                if G.has_edge(x, y):
                    G.remove_edge(x, y)
                    break
        longest = find_longest_cycle()
    return G


def get_max_ring_length_at_most(graph):
    """Return maximum simple cycle length with no enumeration cap."""
    max_len = 0
    for cyc in enumerate_simple_cycles_unique(graph, max_len=None):
        if len(cyc) > max_len:
            max_len = len(cyc)
    return max_len 