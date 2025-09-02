###############################################################################
#
# Ring count constraint functionality for molecular graph generation
#
###############################################################################

import networkx as nx
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, simple_cycle_count_exceeds, fast_ring_count_prescreen

__all__ = ["has_at_most_n_rings", "ring_count_at_most_projector", "count_rings_at_most", "ring_count_at_most_sanitize"]


def has_at_most_n_rings(G, K: int) -> bool:
    """
    Check if graph has at most K rings/cycles.
    
    Uses fast pre-screening to avoid expensive cycle enumeration when possible,
    but falls back to exact validation for complex cases to ensure correctness.
    
    Args:
        G: NetworkX graph
        K: Maximum allowed number of rings/cycles
        
    Returns:
        True if graph has ≤ K cycles, False otherwise
        
    Performance: O(1) for simple cases, O(exponential) for complex fused ring systems
    """
    # Fast pre-screening: skip expensive enumeration for trivial cases
    prescreen_result = fast_ring_count_prescreen(G, K)
    if prescreen_result is not None:
        return prescreen_result  # Definitive answer from pre-screening
    
    # Complex case: requires full cycle enumeration for correctness
    return not simple_cycle_count_exceeds(G, K)


def ring_count_at_most_projector(graph, max_rings):
    """
    Edge-Deletion Projector: Enforces structural constraint: total number of unique simple
    cycles ≤ max_rings. Removes edges from cycles until count ≤ max_rings.
    
    DEPRECATED: Use ring_count_at_most_sanitize instead for offline cleanup.
    This function is kept for backward compatibility.
    """
    return ring_count_at_most_sanitize(graph, max_rings)


def ring_count_at_most_sanitize(G, K: int):
    """
    Sanitizer: Greedily remove the edge that appears in the most cycles until ≤K
    """
    # Greedily remove the edge that appears in the most cycles until <=K
    while simple_cycle_count_exceeds(G, K):
        edge_freq = {}
        for cyc in enumerate_simple_cycles_unique(G):
            m = len(cyc)
            for i in range(m):
                a, b = cyc[i], cyc[(i+1) % m]
                e = tuple(sorted((a, b)))
                edge_freq[e] = edge_freq.get(e, 0) + 1
        
        if edge_freq:
            e_remove = max(edge_freq.items(), key=lambda kv: kv[1])[0]
            if G.has_edge(*e_remove):
                G.remove_edge(*e_remove)
            else:
                # fallback: remove first available edge from first cycle
                for cyc in enumerate_simple_cycles_unique(G):
                    a, b = cyc[0], cyc[1]
                    if G.has_edge(a, b):
                        G.remove_edge(a, b)
                        break
        else:
            break
    return G


def count_rings_at_most(graph):
    """Return the total number of unique simple cycles in the graph for 'at most' constraints."""
    # Remove any cycle counting caps - we want exact counts
    cycles = list(enumerate_simple_cycles_unique(graph, max_len=None))
    return len(cycles) 