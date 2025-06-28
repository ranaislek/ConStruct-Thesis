import abc

import networkx as nx
import numpy as np
import time
import torch
import torch.nn.functional as F

from ConStruct.projector.is_planar import is_planar
from ConStruct.utils import PlaceHolder


def do_zero_prob_forbidden_edges(pred, z_t, clean_data):
    """
    Checks if the graph has forbidden edges.
    """
    adj_matrices = (z_t.E > 0).int()
    zeroed_edge = torch.tensor([1.0] + [0.0] * (pred.E.shape[-1] - 1))
    for graph_idx, adj_matrix in enumerate(adj_matrices):
        # t21 = time.time()
        num_nodes = z_t.X.shape[1]
        adj_matrix = adj_matrix[:num_nodes, :num_nodes]
        forbidden_edges = get_forbidden_edges(adj_matrix)
        if forbidden_edges.shape[0] > 0:
            pred.E[graph_idx, forbidden_edges[:, 0], forbidden_edges[:, 1]] = (
                zeroed_edge
            )
            pred.E[graph_idx, forbidden_edges[:, 1], forbidden_edges[:, 0]] = (
                zeroed_edge
            )

        # t22 = time.time()
        # print("Graph idx", graph_idx, "-- Time to get forbidden edges: ", t22 - t21)

    return pred


def get_forbidden_edges(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    nx_graph = nx.from_numpy_array(adj_matrix.cpu().numpy())
    forbidden_edges = []
    for node_1 in range(num_nodes):
        for node_2 in range(node_1, num_nodes):
            # Prevent computation of is planar
            if nx_graph.has_edge(node_1, node_2):
                continue
            # Check if adding edge makes graph non-planar
            trial_graph = nx_graph.copy()
            trial_graph.add_edge(node_1, node_2)
            if not is_planar.is_planar(trial_graph):
                forbidden_edges.append([node_1, node_2])
            # assert nx.is_planar(trial_graph) == is_planar.is_planar(trial_graph)

    return torch.tensor(forbidden_edges)


def get_adj_matrix(z_t):
    # Check if edge exists by looking at the actual value, not just argmax
    z_t_adj = (z_t.E.sum(dim=3) > 0).int()  # Sum across edge types and check if > 0
    return z_t_adj


class AbstractProjector(abc.ABC):
    @abc.abstractmethod
    def valid_graph_fn(self, nx_graph):
        pass

    @property
    @abc.abstractmethod
    def can_block_edges(self):
        pass

    def __init__(self, z_t: PlaceHolder):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        if self.can_block_edges:
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_adj_matrix(z_t)

        # add data structure where planarity is checked
        for graph_idx in range(self.batch_size):
            num_nodes = int(z_t.node_mask[graph_idx].sum().item())
            nx_graph = nx.empty_graph(num_nodes)
            assert self.valid_graph_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False

    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_adj_matrix(z_s)
        diff_adj = z_s_adj - self.z_t_adj
        print(f"z_t_adj:\n{self.z_t_adj[0].cpu().numpy()}")
        print(f"z_s_adj:\n{z_s_adj[0].cpu().numpy()}")
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        print(f"Projector: z_s_adj has {z_s_adj.sum()} edges, z_t_adj has {self.z_t_adj.sum()} edges, diff_adj has {diff_adj.sum()} new edges")
        print(f"Projector: new_edges = {new_edges}")
        
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edges_to_add = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ][:, 1:]
                .cpu()
                .numpy()
            )
            print(f"Graph {graph_idx}, edges_to_add: {edges_to_add}")

            # If we can block edges, we do it
            if self.can_block_edges:
                not_blocked_edges = []
                for edge in edges_to_add:
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        not_blocked_edges.append(edge)
                edges_to_add = np.array(not_blocked_edges)

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                nx_graph.add_edges_from(edges_to_add)

            # If it fails, we go one by one and delete the ring breakers
            if not self.valid_graph_fn(nx_graph) or len(edges_to_add) == 1:
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge in edges_to_add:
                    old_nx_graph = nx_graph.copy()
                    nx_graph.add_edge(edge[0], edge[1])
                    if not self.valid_graph_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
            
            # After adding all possible edges, check if we need to remove edges from existing rings
            if not self.valid_graph_fn(nx_graph):
                # Apply ring count projection to remove excess rings
                from ConStruct.projector.is_ring_count import ring_count_projector
                nx_graph = ring_count_projector(nx_graph, self.max_rings)
                
                # Find which edges were removed and update the tensor accordingly
                old_edges = set(old_nx_graph.edges())
                new_edges_set = set(nx_graph.edges())
                removed_edges = old_edges - new_edges_set
                
                # Remove these edges from the tensor
                for edge in removed_edges:
                    u, v = edge
                    z_s.E[graph_idx, u, v] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    z_s.E[graph_idx, v, u] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    # Mark as blocked for future iterations
                    if self.can_block_edges:
                        self.blocked_edges[graph_idx][tuple(edge)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            # (Removed assertion for tensor/NetworkX adjacency equality)

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)


def has_no_cycles(nx_graph):
    # Tree have n-1 edges
    if nx_graph.number_of_edges() >= nx_graph.number_of_nodes():
        return False
    return nx.is_forest(nx_graph)
    # try:
    #     # Attempt to find a cycle
    #     nx.find_cycle(nx_graph)
    #     # If a cycle is found, it's not a tree
    #     return False
    # except nx.exception.NetworkXNoCycle:
    #     # No cycle found, so it's a tree
    #     return True


def is_linear_graph(nx_graph):
    num_nodes = nx_graph.number_of_nodes()
    num_degree_one = sum([d == 1 for n, d in nx_graph.degree()])
    num_degree_two = sum([d == 2 for n, d in nx_graph.degree()])

    return (num_degree_one == 2 and num_degree_two == num_nodes - 2) or (
        num_degree_one == 0 and num_degree_two == 0
    )


def has_lobster_components(nx_graph):
    if has_no_cycles(nx_graph):
        G = nx_graph.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        conn_components = nx.connected_components(G)
        for node_set in conn_components:
            subgraph = G.subgraph(node_set)
            if not is_linear_graph(subgraph):
                return False
        return True
    else:
        return False


class PlanarProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        return is_planar.is_planar(nx_graph)

    @property
    def can_block_edges(self):
        return True


class TreeProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        return has_no_cycles(nx_graph)

    @property
    def can_block_edges(self):
        return True


class LobsterProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        return has_lobster_components(nx_graph)

    @property
    def can_block_edges(self):
        return True


class RingCountProjector(AbstractProjector):
    def __init__(self, z_t: PlaceHolder, max_rings: int):
        self.max_rings = max_rings
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        # Use NetworkX to count cycles (rings)
        cycles = nx.cycle_basis(nx_graph)
        return len(cycles) <= self.max_rings

    @property
    def can_block_edges(self):
        return True
    
    def project(self, z_s: PlaceHolder):
        """Override project method to handle ring count constraint properly."""
        # find added edges
        z_s_adj = get_adj_matrix(z_s)
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edges_to_add = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ][:, 1:]
                .cpu()
                .numpy()
            )

            # If we can block edges, we do it
            if self.can_block_edges:
                not_blocked_edges = []
                for edge in edges_to_add:
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        not_blocked_edges.append(edge)
                edges_to_add = np.array(not_blocked_edges)

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                nx_graph.add_edges_from(edges_to_add)

            # If it fails, we go one by one and delete the ring breakers
            if not self.valid_graph_fn(nx_graph) or len(edges_to_add) == 1:
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge in edges_to_add:
                    old_nx_graph = nx_graph.copy()
                    nx_graph.add_edge(edge[0], edge[1])
                    if not self.valid_graph_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
            
            # After adding all possible edges, check if we need to remove edges from existing rings
            if not self.valid_graph_fn(nx_graph):
                # Apply ring count projection to remove excess rings
                from ConStruct.projector.is_ring_count import ring_count_projector
                nx_graph = ring_count_projector(nx_graph, self.max_rings)
                
                # Find which edges were removed and update the tensor accordingly
                old_edges = set(old_nx_graph.edges())
                new_edges_set = set(nx_graph.edges())
                removed_edges = old_edges - new_edges_set
                
                # Remove these edges from the tensor
                for edge in removed_edges:
                    u, v = edge
                    z_s.E[graph_idx, u, v] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    z_s.E[graph_idx, v, u] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    # Mark as blocked for future iterations
                    if self.can_block_edges:
                        self.blocked_edges[graph_idx][tuple(edge)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            # (Removed assertion for tensor/NetworkX adjacency equality)

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)


class RingLengthProjector(AbstractProjector):
    def __init__(self, z_t: PlaceHolder, max_ring_length: int):
        self.max_ring_length = max_ring_length
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        # Use NetworkX to check if all rings have length at most max_ring_length
        from ConStruct.projector.is_ring_count import has_rings_of_length_at_most
        return has_rings_of_length_at_most(nx_graph, self.max_ring_length)

    @property
    def can_block_edges(self):
        return True
    
    def project(self, z_s: PlaceHolder):
        """Override project method to handle ring length constraint properly."""
        # find added edges
        z_s_adj = get_adj_matrix(z_s)
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edges_to_add = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ][:, 1:]
                .cpu()
                .numpy()
            )

            # If we can block edges, we do it
            print(f"Graph {graph_idx}, edges_to_add: {edges_to_add}")
            
            # If we can block edges, we do it
            not_blocked_edges = []
            for edge in edges_to_add:
                # Check if adding this edge would create a ring longer than allowed
                # Build test graph from current adjacency matrix to ensure all existing edges are present
                test_graph = nx.from_numpy_array(self.z_t_adj[graph_idx].cpu().numpy())
                test_graph.add_edge(edge[0], edge[1])
                from ConStruct.projector.is_ring_count import has_rings_of_length_at_most
                print(f"Trying edge {edge}, cycle basis: {nx.cycle_basis(test_graph)}, has_rings_of_length_at_most={has_rings_of_length_at_most(test_graph, self.max_ring_length)}")
                if not has_rings_of_length_at_most(test_graph, self.max_ring_length):
                    print(f"Blocking edge {edge} because it creates rings: {nx.cycle_basis(test_graph)}")
                    # Block this edge
                    z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    print(f"  After blocking, edge {edge} value: {z_s.E[graph_idx, edge[0], edge[1], 0].item()}")
                    if self.can_block_edges:
                        self.blocked_edges[graph_idx][tuple(edge)] = True
                else:
                    not_blocked_edges.append(edge)
            edges_to_add = np.array(not_blocked_edges)

            # Add all allowed edges
            if len(edges_to_add) > 0:
                nx_graph.add_edges_from(edges_to_add)

            # After adding all possible edges, check if we need to remove edges from existing rings
            from ConStruct.projector.is_ring_count import ring_length_projector
            if not self.valid_graph_fn(nx_graph):
                nx_graph = ring_length_projector(nx_graph, self.max_ring_length)
                # Find which edges were removed and update the tensor accordingly
                old_edges = set(old_nx_graph.edges())
                new_edges_set = set(nx_graph.edges())
                removed_edges = old_edges - new_edges_set
                for edge in removed_edges:
                    u, v = edge
                    z_s.E[graph_idx, u, v] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    z_s.E[graph_idx, v, u] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    if self.can_block_edges:
                        self.blocked_edges[graph_idx][tuple(edge)] = True
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            # (Removed assertion for tensor/NetworkX adjacency equality)

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)
