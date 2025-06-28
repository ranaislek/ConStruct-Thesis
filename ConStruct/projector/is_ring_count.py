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
    This removes edges from the largest rings first.
    """
    while True:
        cycles = nx.cycle_basis(graph)
        if not cycles:  # No cycles left
            break
            
        # Find the largest cycle
        largest_cycle = max(cycles, key=len)
        if len(largest_cycle) <= max_length:
            break
            
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
