###############################################################################
#
# Ring count "at least" constraint functionality for molecular graph generation
# This is the flipped mechanism of edge-deletion for edge-insertion transitions
#
###############################################################################

import networkx as nx

__all__ = ["has_at_least_n_rings", "ring_count_at_least_projector", "count_rings_at_least"]


def has_at_least_n_rings(graph, n):
    """Return True if the graph has at least n rings (cycles)."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) >= n


def ring_count_at_least_projector(graph, min_rings):
    """
    Edge-Insertion Projector: If the graph has fewer than min_rings, add edges to create cycles.
    AGGRESSIVE APPROACH: Add edges strategically to create the required number of rings.
    """
    cycles = nx.cycle_basis(graph)
    current_rings = len(cycles)
    
    if current_rings >= min_rings:
        return graph  # Already satisfies constraint
    
    # Need to add rings
    rings_needed = min_rings - current_rings
    
    # Strategy: Add edges to create rings
    # 1. First, try to connect disconnected components
    components = list(nx.connected_components(graph))
    
    # Keep trying until we have enough rings
    attempts = 0
    max_attempts = rings_needed * 10  # Prevent infinite loops
    
    while current_rings < min_rings and attempts < max_attempts:
        attempts += 1
        
        if len(components) <= 1:
            # Need to create a ring within a single component
            nodes = list(graph.nodes())
            
            # Strategy 1: Try to add edges to create new cycles
            ring_created = False
            
            # Try to find a path and close it
            for start_node in nodes:
                for end_node in nodes:
                    if start_node != end_node and not graph.has_edge(start_node, end_node):
                        try:
                            path = nx.shortest_path(graph, start_node, end_node)
                            if len(path) >= 3:
                                # Add edge to close the path into a ring
                                graph.add_edge(start_node, end_node)
                                ring_created = True
                                break
                        except nx.NetworkXNoPath:
                            continue
                if ring_created:
                    break
            
            # Strategy 2: If no ring created, try adding edges to create new cycles
            if not ring_created:
                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i < j and not graph.has_edge(node1, node2):
                            # Check if adding this edge creates a new cycle
                            graph.add_edge(node1, node2)
                            new_cycles = nx.cycle_basis(graph)
                            if len(new_cycles) > current_rings:
                                ring_created = True
                                break
                            else:
                                # Remove the edge if it didn't create a cycle
                                graph.remove_edge(node1, node2)
                    if ring_created:
                        break
            
            # Strategy 3: If still no ring created, add a new node and create a cycle
            if not ring_created and len(nodes) >= 3:
                # Add a new node and connect it to create a new cycle
                new_node = max(nodes) + 1
                # Connect to two existing nodes to create a new triangle
                graph.add_edge(new_node, nodes[0])
                graph.add_edge(new_node, nodes[1])
                ring_created = True
        else:
            # Connect two components to create a ring
            if len(components) >= 2:
                comp1, comp2 = components[0], components[1]
                node1 = list(comp1)[0]
                node2 = list(comp2)[0]
                graph.add_edge(node1, node2)
                components = list(nx.connected_components(graph))
        
        # Update ring count and components
        current_rings = len(nx.cycle_basis(graph))
        components = list(nx.connected_components(graph))
    
    return graph


def count_rings_at_least(graph):
    """Return the number of rings (cycles) in the graph for 'at least' constraints."""
    cycles = nx.cycle_basis(graph)
    return len(cycles) 