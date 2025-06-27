import networkx as nx

def has_at_most_n_rings(graph, n):
    """Return True if the graph has at most n rings (cycles)."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) <= n

def ring_count_projector(graph, max_rings):
    """
    Projector: If the graph has more than max_rings, remove edges to break cycles.
    This is a simple greedy approach.
    """
    while True:
        cycles = nx.cycle_basis(graph)
        if len(cycles) <= max_rings:
            break
        # Remove an edge from the first cycle found
        cycle = cycles[0]
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

def count_rings(graph):
    """Return the number of rings (cycles) in the graph."""
    cycles = nx.cycle_basis(graph)
    return len(cycles)
