import abc

import networkx as nx
import numpy as np
import time
import torch
import torch.nn.functional as F
from rdkit import Chem
from typing import List, Optional, Tuple, Dict, Any

from ConStruct.projector.is_planar import is_planar
from ConStruct.projector.is_ring.is_ring_count_at_most.is_ring_count_at_most import ring_count_at_most_projector
from ConStruct.projector.is_ring.is_ring_length_at_most.is_ring_length_at_most import has_rings_of_length_at_most
from ConStruct.projector.is_ring.is_ring_count_at_least.is_ring_count_at_least import ring_count_at_least_projector
from ConStruct.projector.is_ring.is_ring_length_at_least.is_ring_length_at_least import ring_length_at_least_projector
from ConStruct.utils import PlaceHolder
from ConStruct.diffusion.extra_features import ExtraFeatures
from ConStruct.diffusion.extra_features_molecular import ExtraMolecularFeatures


def resize_placeholder_tensor(z_s: PlaceHolder, new_size: int, graph_idx: int = 0) -> PlaceHolder:
    """
    Resize PlaceHolder tensor to accommodate new nodes for dynamic tensor resizing.
    
    Args:
        z_s: PlaceHolder tensor to resize
        new_size: New number of nodes (must be >= current size)
        graph_idx: Index of the graph being modified
    
    Returns:
        Resized PlaceHolder tensor
    """
    current_size = z_s.X.shape[1]
    
    if new_size <= current_size:
        return z_s  # No resizing needed
    
    # Resize X tensor (node features)
    new_X = torch.zeros(z_s.X.shape[0], new_size, z_s.X.shape[2], device=z_s.X.device)
    new_X[:, :current_size, :] = z_s.X
    z_s.X = new_X
    
    # Resize E tensor (edge features)
    new_E = torch.zeros(z_s.E.shape[0], new_size, new_size, z_s.E.shape[3], device=z_s.E.device)
    new_E[:, :current_size, :current_size, :] = z_s.E
    z_s.E = new_E
    
    # Resize node mask if it exists
    if hasattr(z_s, 'node_mask') and z_s.node_mask is not None:
        new_mask = torch.zeros(z_s.node_mask.shape[0], new_size, device=z_s.node_mask.device)
        new_mask[:, :current_size] = z_s.node_mask
        z_s.node_mask = new_mask
    
    # Resize extra features if they exist
    if hasattr(z_s, 'extra_features') and z_s.extra_features is not None:
        if isinstance(z_s.extra_features, ExtraMolecularFeatures):
            # Resize charge tensor
            if hasattr(z_s.extra_features, 'charge') and z_s.extra_features.charge is not None:
                new_charge = torch.zeros(z_s.extra_features.charge.shape[0], new_size, device=z_s.extra_features.charge.device)
                new_charge[:, :current_size] = z_s.extra_features.charge
                z_s.extra_features.charge = new_charge
            
            # Resize is_aromatic tensor
            if hasattr(z_s.extra_features, 'is_aromatic') and z_s.extra_features.is_aromatic is not None:
                new_aromatic = torch.zeros(z_s.extra_features.is_aromatic.shape[0], new_size, device=z_s.extra_features.is_aromatic.device)
                new_aromatic[:, :current_size] = z_s.extra_features.is_aromatic
                z_s.extra_features.is_aromatic = new_aromatic
    
    return z_s


def update_tensor_from_graph(z_s: PlaceHolder, graph_idx: int, nx_graph: nx.Graph) -> None:
    """
    Synchronize tensor with NetworkX graph after resizing.
    
    Args:
        z_s: PlaceHolder tensor to update
        graph_idx: Index of the graph being updated
        nx_graph: NetworkX graph with the updated structure
    """
    # Clear existing edges for this graph
    z_s.E[graph_idx] = torch.zeros_like(z_s.E[graph_idx])
    
    # Add all edges from graph
    for u, v in nx_graph.edges():
        if u != v and u < z_s.E.shape[1] and v < z_s.E.shape[2]:
            z_s.E[graph_idx, u, v, 1] = 1  # single bond
            z_s.E[graph_idx, v, u, 1] = 1  # undirected


def create_ring_with_new_nodes(graph: nx.Graph, min_length: int, start_node_idx: int) -> nx.Graph:
    """
    Create a ring of specified length using new nodes, and connect it to the existing graph if possible.
    Args:
        graph: NetworkX graph to modify
        min_length: Minimum length of the ring to create
        start_node_idx: Starting index for new nodes
    Returns:
        Modified graph with new ring, connected to existing graph if possible
    """
    # Add new nodes to graph
    new_nodes = []
    for i in range(min_length):
        new_node = start_node_idx + i
        graph.add_node(new_node)
        new_nodes.append(new_node)
    # Create ring using new nodes
    for i in range(min_length):
        u = new_nodes[i]
        v = new_nodes[(i + 1) % min_length]
        graph.add_edge(u, v)
    # Connect to existing graph (maintain connectivity)
    existing_nodes = [n for n in graph.nodes() if n < start_node_idx]
    if existing_nodes:
        # Connect all isolated existing nodes to the new ring
        for i, existing_node in enumerate(existing_nodes):
            # Connect to a different new node for each existing node to distribute connections
            new_node_idx = i % len(new_nodes)
            graph.add_edge(existing_node, new_nodes[new_node_idx])
    return graph


def is_chemically_valid_graph(nx_graph, atom_types, atom_decoder):
    """
    Check if a NetworkX graph can be converted to a chemically valid RDKit molecule.
    This ensures that the graph structure is chemically meaningful.
    """
    try:
        # Convert NetworkX graph to RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for i, atom_type in enumerate(atom_types):
            if atom_type == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom_type.item())])
            mol.AddAtom(a)
        
        # Add bonds
        for edge in nx_graph.edges():
            u, v = edge
            if u != v:  # No self-loops
                mol.AddBond(int(u), int(v), Chem.rdchem.BondType.SINGLE)
        
        # Try to convert to a valid molecule
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


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
            
            # For edge-insertion with "at least" constraints:
            # - Start with fully connected graph (many rings)
            # - This ensures we have enough rings to satisfy min_rings constraint
            if num_nodes >= 3:  # Need at least 3 nodes for a ring
                nx_graph = nx.complete_graph(num_nodes)
            else:
                nx_graph = nx.empty_graph(num_nodes)
            
            # For edge-insertion, we don't validate the initial graph
            # because we'll be removing edges to satisfy constraints
            # assert self.valid_graph_fn(nx_graph)  # REMOVED: Don't validate initial graph
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
        # print(f"z_t_adj:\n{self.z_t_adj[0].cpu().numpy()}")
        # print(f"z_s_adj:\n{z_s_adj[0].cpu().numpy()}")
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        # print(f"Projector: z_s_adj has {z_s_adj.sum()} edges, z_t_adj has {self.z_t_adj.sum()} edges, diff_adj has {diff_adj.sum()} new edges")
        # print(f"Projector: new_edges = {new_edges}")
        
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
            # print(f"Graph {graph_idx}, edges_to_add: {edges_to_add}")

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
                nx_graph = ring_count_at_most_projector(nx_graph, self.max_rings)
                
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


class RingCountAtMostProjector(AbstractProjector):
    """
    Edge-Deletion Projector: Ensures graphs have at most N rings.
    
    This projector is designed for edge-deletion transitions (absorbing_edges).
    - Forward diffusion: Edges progressively disappear toward no-edge state
    - Reverse diffusion: Edges are added back while respecting max ring constraint
    - Use with transitions: absorbing_edges
    
    Mathematical properties:
    - Constraint type: "at most" (e.g., at most N rings)
    - Natural bias: toward acyclic graphs
    - Projection: Remove edges to break excess rings
    """
    def __init__(self, z_t: PlaceHolder, max_rings: int, atom_decoder=None):
        self.max_rings = max_rings
        self.atom_decoder = atom_decoder
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        # Use NetworkX to count cycles (rings)
        cycles = nx.cycle_basis(nx_graph)
        # For edge-deletion with "at most" constraints:
        # - We start from sparse graphs (few rings)
        # - We add edges until we have exactly max_rings
        # - We NEVER allow graphs with more than max_rings
        return len(cycles) <= self.max_rings  # At most N rings (NO EXCEPTIONS!)

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
                    
                    # Check both ring constraint AND chemical validity
                    ring_valid = self.valid_graph_fn(nx_graph)
                    chem_valid = True
                    if self.atom_decoder is not None:
                        # Get atom types for this graph
                        atom_types = z_s.X[graph_idx]
                        chem_valid = is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder)
                    
                    if not ring_valid or not chem_valid:
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
            atom_types = z_s.X[graph_idx] if self.atom_decoder is not None else None
            
            # Reconstruct NetworkX graph from current tensor to ensure synchronization
            current_adj = get_adj_matrix(z_s)[graph_idx].cpu().numpy()
            reconstructed_graph = nx.from_numpy_array(current_adj)
            # print(f"[DEBUG] Before ring_count_at_most_projector: {list(nx.cycle_basis(reconstructed_graph))}")
            
            # Apply ring count projection until the graph is valid
            while not self.valid_graph_fn(reconstructed_graph):
                reconstructed_graph = ring_count_at_most_projector(reconstructed_graph, self.max_rings)
                # print(f"[DEBUG] After ring_count_at_most_projector: {list(nx.cycle_basis(reconstructed_graph))}")
            
            # Synchronize z_s.E with reconstructed_graph after projection is complete
            # Set all edges to zero, then set to 1 for all edges in the projected graph (single bond type)
            z_s.E[graph_idx] = torch.zeros_like(z_s.E[graph_idx])
            for u, v in reconstructed_graph.edges():
                if u != v:
                    z_s.E[graph_idx, u, v, 1] = 1  # single bond
                    z_s.E[graph_idx, v, u, 1] = 1  # undirected
            # Mark as blocked for all non-edges
            n = reconstructed_graph.number_of_nodes()
            current_edges = set(reconstructed_graph.edges())
            for u in range(n):
                for v in range(n):
                    if u != v and (u, v) not in current_edges and (v, u) not in current_edges:
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            # Update the nx_graph to match the reconstructed graph
            nx_graph = reconstructed_graph
            
            # Final check: ensure the graph is both ring-compliant and chemically valid
            if self.atom_decoder is not None:
                atom_types = z_s.X[graph_idx]
                if not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder):
                    # If chemically invalid, remove edges until valid
                    while not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder) and nx_graph.number_of_edges() > 0:
                        # Remove a random edge
                        edge_to_remove = list(nx_graph.edges())[0]
                        nx_graph.remove_edge(*edge_to_remove)
                        # Update tensor
                        u, v = edge_to_remove
                        z_s.E[graph_idx, u, v] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, v, u] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # Mark as blocked
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)


class RingLengthAtMostProjector(AbstractProjector):
    def __init__(self, z_t: PlaceHolder, max_ring_length: int, atom_decoder=None):
        self.max_ring_length = max_ring_length
        self.atom_decoder = atom_decoder
        super().__init__(z_t)

    def valid_graph_fn(self, nx_graph):
        # Use NetworkX to check if all rings have length at most max_ring_length
        return has_rings_of_length_at_most(nx_graph, self.max_ring_length)  # At most N ring length

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
            not_blocked_edges = []
            for edge in edges_to_add:
                # Check if adding this edge would create a ring longer than allowed
                # Build test graph from current adjacency matrix to ensure all existing edges are present
                test_graph = nx.from_numpy_array(self.z_t_adj[graph_idx].cpu().numpy())
                test_graph.add_edge(edge[0], edge[1])
                
                # Check both ring length constraint AND chemical validity
                ring_valid = has_rings_of_length_at_most(test_graph, self.max_ring_length)
                chem_valid = True
                if self.atom_decoder is not None:
                    # Get atom types for this graph
                    atom_types = z_s.X[graph_idx]
                    chem_valid = is_chemically_valid_graph(test_graph, atom_types, self.atom_decoder)
                
                if not ring_valid or not chem_valid:
                    # Block this edge
                    z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                        torch.tensor(0), num_classes=z_s.E.shape[-1]
                    )
                    if self.can_block_edges:
                        self.blocked_edges[graph_idx][tuple(edge)] = True
                else:
                    not_blocked_edges.append(edge)
            edges_to_add = np.array(not_blocked_edges)

            # Add all allowed edges
            if len(edges_to_add) > 0:
                nx_graph.add_edges_from(edges_to_add)

            # After adding all possible edges, check if we need to remove edges from existing rings
            from ConStruct.projector.is_ring.is_ring_length_at_most.is_ring_length_at_most import ring_length_at_most_projector
            
            # Reconstruct NetworkX graph from current tensor to ensure synchronization
            current_adj = get_adj_matrix(z_s)[graph_idx].cpu().numpy()
            reconstructed_graph = nx.from_numpy_array(current_adj)
            # print(f"[DEBUG] Before ring_length_at_most_projector: {list(nx.cycle_basis(reconstructed_graph))}")
            
            # Apply ring length projection until the graph is valid
            while not self.valid_graph_fn(reconstructed_graph):
                reconstructed_graph = ring_length_at_most_projector(reconstructed_graph, self.max_ring_length)
                # print(f"[DEBUG] After ring_length_at_most_projector: {list(nx.cycle_basis(reconstructed_graph))}")
            
            # Synchronize z_s.E with reconstructed_graph after projection is complete
            # Set all edges to zero, then set to 1 for all edges in the projected graph (single bond type)
            z_s.E[graph_idx] = torch.zeros_like(z_s.E[graph_idx])
            for u, v in reconstructed_graph.edges():
                if u != v:
                    z_s.E[graph_idx, u, v, 1] = 1  # single bond
                    z_s.E[graph_idx, v, u, 1] = 1  # undirected
            # Mark as blocked for all non-edges
            n = reconstructed_graph.number_of_nodes()
            current_edges = set(reconstructed_graph.edges())
            for u in range(n):
                for v in range(n):
                    if u != v and (u, v) not in current_edges and (v, u) not in current_edges:
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            # Update the nx_graph to match the reconstructed graph
            nx_graph = reconstructed_graph
            
            # Final check: ensure the graph is both ring-compliant and chemically valid
            if self.atom_decoder is not None:
                atom_types = z_s.X[graph_idx]
                if not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder):
                    # If chemically invalid, remove edges until valid
                    while not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder) and nx_graph.number_of_edges() > 0:
                        # Remove a random edge
                        edge_to_remove = list(nx_graph.edges())[0]
                        nx_graph.remove_edge(*edge_to_remove)
                        # Update tensor
                        u, v = edge_to_remove
                        z_s.E[graph_idx, u, v] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, v, u] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # Mark as blocked
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            # Synchronize z_s.E with nx_graph after projection is complete
            # Set all edges to zero, then set to 1 for all edges in the projected graph (single bond type)
            z_s.E[graph_idx] = torch.zeros_like(z_s.E[graph_idx])
            for u, v in nx_graph.edges():
                if u != v:
                    z_s.E[graph_idx, u, v, 1] = 1  # single bond
                    z_s.E[graph_idx, v, u, 1] = 1  # undirected
            # Mark as blocked for all non-edges
            n = nx_graph.number_of_nodes()
            current_edges = set(nx_graph.edges())
            for u in range(n):
                for v in range(n):
                    if u != v and (u, v) not in current_edges and (v, u) not in current_edges:
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)


class RingCountAtLeastProjector(AbstractProjector):
    """
    Edge-Insertion Projector: Ensures graphs have at least N rings.
    
    This projector is designed for edge-insertion transitions (edge_insertion).
    - Forward diffusion: Edges progressively appear toward edge state
    - Reverse diffusion: Edges are removed while preserving min ring constraint
    - Use with transitions: edge_insertion
    
    Mathematical properties:
    - Constraint type: "at least" (e.g., at least N rings)
    - Natural bias: toward connected graphs
    - Projection: ADD edges to satisfy min ring constraint
    
    CRITICAL: This mechanism is separate from edge-deletion and should not
    be mixed with RingCountAtMostProjector in the same experiment.
    """
    
    def __init__(self, z_t: PlaceHolder, min_rings: int, atom_decoder=None):
        self.min_rings = min_rings
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        # Store z_t for edge-insertion projection
        self.z_t = z_t

    def valid_graph_fn(self, nx_graph):
        # Use NetworkX to count cycles (rings)
        cycles = nx.cycle_basis(nx_graph)
        # For edge-insertion with "at least" constraints:
        # - We start from fully connected graphs (many rings)
        # - We remove edges until we have exactly min_rings
        # - We NEVER allow graphs with fewer than min_rings
        return len(cycles) >= self.min_rings  # At least N rings (NO EXCEPTIONS!)

    @property
    def can_block_edges(self):
        return True
    
    def project(self, z_s: PlaceHolder):
        """Edge-insertion projection: ADD edges to satisfy ring constraints."""
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Check current ring count
            cycles = nx.cycle_basis(nx_graph)
            current_rings = len(cycles)
            
            # If we already have enough rings, no action needed
            if current_rings >= self.min_rings:
                continue
            
            # We need to ADD edges to create more rings
            rings_needed = self.min_rings - current_rings
            
            # Strategy: Add edges to create rings
            # 1. First, try to connect disconnected components
            components = list(nx.connected_components(nx_graph))
            
            # Keep trying until we have enough rings
            attempts = 0
            max_attempts = rings_needed * 10  # Prevent infinite loops
            
            while current_rings < self.min_rings and attempts < max_attempts:
                attempts += 1
                
                if len(components) <= 1:
                    # Need to create a ring within a single component
                    nodes = list(nx_graph.nodes())
                    
                    # Strategy 1: Try to add edges to create new cycles
                    ring_created = False
                    
                    # Try to find a path and close it
                    for start_node in nodes:
                        for end_node in nodes:
                            if start_node != end_node and not nx_graph.has_edge(start_node, end_node):
                                try:
                                    path = nx.shortest_path(nx_graph, start_node, end_node)
                                    if len(path) >= 3:
                                        # Add edge to close the path into a ring
                                        nx_graph.add_edge(start_node, end_node)
                                        # Update tensor
                                        z_s.E[graph_idx, start_node, end_node, 1] = 1  # single bond
                                        z_s.E[graph_idx, end_node, start_node, 1] = 1  # undirected
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
                                if i < j and not nx_graph.has_edge(node1, node2):
                                    # Check if adding this edge creates a new cycle
                                    nx_graph.add_edge(node1, node2)
                                    new_cycles = nx.cycle_basis(nx_graph)
                                    if len(new_cycles) > current_rings:
                                        # Update tensor
                                        z_s.E[graph_idx, node1, node2, 1] = 1  # single bond
                                        z_s.E[graph_idx, node2, node1, 1] = 1  # undirected
                                        ring_created = True
                                        break
                                    else:
                                        # Remove the edge if it didn't create a cycle
                                        nx_graph.remove_edge(node1, node2)
                            if ring_created:
                                break
                    
                    # Strategy 3: If still no ring created, add new nodes and create a cycle
                    if not ring_created:
                        # Calculate how many new nodes we need for a triangle
                        new_nodes_needed = max(0, 3 - len(nodes))
                        if new_nodes_needed > 0:
                            # Resize tensor to accommodate new nodes
                            new_size = len(nodes) + new_nodes_needed
                            resize_placeholder_tensor(z_s, new_size, graph_idx)
                            
                            # Create triangle with new nodes
                            new_nodes = []
                            for i in range(3):
                                if i < len(nodes):
                                    new_nodes.append(nodes[i])
                                else:
                                    new_node = len(nodes) + i - len(nodes)
                                    new_nodes.append(new_node)
                                    nx_graph.add_node(new_node)
                            
                            # Create triangle
                            nx_graph.add_edge(new_nodes[0], new_nodes[1])
                            nx_graph.add_edge(new_nodes[1], new_nodes[2])
                            nx_graph.add_edge(new_nodes[2], new_nodes[0])
                            
                            # Update tensor
                            for i in range(3):
                                for j in range(i + 1, 3):
                                    u, v = new_nodes[i], new_nodes[j]
                                    z_s.E[graph_idx, u, v, 1] = 1  # single bond
                                    z_s.E[graph_idx, v, u, 1] = 1  # undirected
                            
                            ring_created = True
                        elif len(nodes) >= 3:
                            # Use existing nodes to create triangle
                            nx_graph.add_edge(nodes[0], nodes[1])
                            nx_graph.add_edge(nodes[1], nodes[2])
                            nx_graph.add_edge(nodes[2], nodes[0])
                            
                            # Update tensor
                            for i in range(3):
                                for j in range(i + 1, 3):
                                    u, v = nodes[i], nodes[j]
                                    z_s.E[graph_idx, u, v, 1] = 1  # single bond
                                    z_s.E[graph_idx, v, u, 1] = 1  # undirected
                            
                            ring_created = True
                else:
                    # Connect two components to create a ring
                    if len(components) >= 2:
                        comp1, comp2 = components[0], components[1]
                        node1 = list(comp1)[0]
                        node2 = list(comp2)[0]
                        nx_graph.add_edge(node1, node2)
                        # Update tensor
                        z_s.E[graph_idx, node1, node2, 1] = 1  # single bond
                        z_s.E[graph_idx, node2, node1, 1] = 1  # undirected
                        components = list(nx.connected_components(nx_graph))
                
                # Update ring count and components
                current_rings = len(nx.cycle_basis(nx_graph))
                components = list(nx.connected_components(nx_graph))
            
            # CRITICAL STEP: Apply ring count projection until the graph is valid
            # This is the key step that was missing!
            reconstructed_graph = nx.from_numpy_array(get_adj_matrix(z_s)[graph_idx].cpu().numpy())
            
            # Apply ring count projection until the graph is valid
            while not self.valid_graph_fn(reconstructed_graph):
                from ConStruct.projector.is_ring.is_ring_count_at_least.is_ring_count_at_least import ring_count_at_least_projector
                reconstructed_graph = ring_count_at_least_projector(reconstructed_graph, self.min_rings)
            
            # Synchronize z_s.E with reconstructed_graph after projection is complete
            update_tensor_from_graph(z_s, graph_idx, reconstructed_graph)
            # Mark as blocked for all non-edges
            n = reconstructed_graph.number_of_nodes()
            current_edges = set(reconstructed_graph.edges())
            for u in range(n):
                for v in range(n):
                    if u != v and (u, v) not in current_edges and (v, u) not in current_edges:
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            # Update the nx_graph to match the reconstructed graph
            nx_graph = reconstructed_graph
            
            # Final check: ensure the graph is both ring-compliant and chemically valid
            if self.atom_decoder is not None:
                atom_types = z_s.X[graph_idx]
                if not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder):
                    # If chemically invalid, remove edges until valid
                    while not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder) and nx_graph.number_of_edges() > 0:
                        # Remove a random edge
                        edge_to_remove = list(nx_graph.edges())[0]
                        nx_graph.remove_edge(*edge_to_remove)
                        # Update tensor
                        u, v = edge_to_remove
                        z_s.E[graph_idx, u, v] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, v, u] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # Mark as blocked
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)


class RingLengthAtLeastProjector(AbstractProjector):
    """
    Edge-Insertion Projector: Ensures graphs have rings of at least N length.
    
    This projector is designed for edge-insertion transitions (edge_insertion).
    - Forward diffusion: Edges progressively appear toward edge state
    - Reverse diffusion: Edges are removed while preserving min ring length constraint
    - Use with transitions: edge_insertion
    
    Mathematical properties:
    - Constraint type: "at least" (e.g., at least N ring length)
    - Natural bias: toward connected graphs
    - Projection: ADD edges to satisfy min ring length constraint
    
    CRITICAL: This mechanism is separate from edge-deletion and should not
    be mixed with RingLengthAtMostProjector in the same experiment.
    """
    
    def __init__(self, z_t: PlaceHolder, min_ring_length: int, atom_decoder=None):
        self.min_ring_length = min_ring_length
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        # Store z_t for edge-insertion projection
        self.z_t = z_t

    def valid_graph_fn(self, nx_graph):
        # Use NetworkX to check if all rings have length at least min_ring_length
        cycles = nx.cycle_basis(nx_graph)
        # For edge-insertion with "at least" constraints:
        # - We start from fully connected graphs
        # - We remove edges until all rings have at least min_ring_length
        # - We NEVER allow rings shorter than min_ring_length
        for cycle in cycles:
            if len(cycle) < self.min_ring_length:
                return False
        return True  # All rings have at least min_ring_length (NO EXCEPTIONS!)

    @property
    def can_block_edges(self):
        return True
    
    def project(self, z_s: PlaceHolder):
        """Edge-insertion projection with dynamic tensor resizing: ADD edges to satisfy ring length constraints."""
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Check current ring lengths
            cycles = nx.cycle_basis(nx_graph)
            
            # Check if we already have rings of sufficient length
            has_sufficient_rings = True
            for cycle in cycles:
                if len(cycle) >= self.min_ring_length:
                    has_sufficient_rings = True
                    break
            else:
                has_sufficient_rings = False
            
            if has_sufficient_rings:
                continue
            
            # We need to create rings of sufficient length
            # Strategy: Try with existing nodes first, then resize tensor if needed
            nodes = list(nx_graph.nodes())
            current_size = len(nodes)
            
            # Try to create ring with existing nodes first
            ring_created = False
            if current_size >= self.min_ring_length:
                # Try to create ring using existing nodes
                for i in range(current_size):
                    for j in range(i + 1, current_size):
                        if not nx_graph.has_edge(nodes[i], nodes[j]):
                            # Try to create a path and close it
                            try:
                                path = nx.shortest_path(nx_graph, nodes[i], nodes[j])
                                if len(path) == self.min_ring_length - 1:
                                    nx_graph.add_edge(nodes[i], nodes[j])
                                    ring_created = True
                                    break
                            except nx.NetworkXNoPath:
                                continue
                    if ring_created:
                        break
            
            # If no ring created with existing nodes, resize tensor and add new nodes
            if not ring_created:
                # Calculate how many new nodes we need
                new_nodes_needed = max(0, self.min_ring_length - current_size)
                if new_nodes_needed > 0:
                    # Resize tensor to accommodate new nodes
                    new_size = current_size + new_nodes_needed
                    resize_placeholder_tensor(z_s, new_size, graph_idx)
                    
                    # Create ring with new nodes
                    nx_graph = create_ring_with_new_nodes(nx_graph, self.min_ring_length, current_size)
                    ring_created = True
                else:
                    # Fallback: create largest possible ring with existing nodes
                    if current_size >= 3:
                        for i in range(current_size):
                            u = nodes[i]
                            v = nodes[(i + 1) % current_size]
                            nx_graph.add_edge(u, v)
                        ring_created = True
            
            # Update tensor with new edges
            update_tensor_from_graph(z_s, graph_idx, nx_graph)
            
            # CRITICAL STEP: Apply ring length projection until the graph is valid
            reconstructed_graph = nx.from_numpy_array(get_adj_matrix(z_s)[graph_idx].cpu().numpy())
            
            # Apply ring length projection until the graph is valid
            while not self.valid_graph_fn(reconstructed_graph):
                # Use enhanced projector with tensor resizing capability
                reconstructed_graph = self._enhanced_ring_length_projector(
                    reconstructed_graph, self.min_ring_length, z_s, graph_idx
                )
            
            # Synchronize z_s.E with reconstructed_graph after projection is complete
            update_tensor_from_graph(z_s, graph_idx, reconstructed_graph)
            
            # Mark as blocked for all non-edges
            n = reconstructed_graph.number_of_nodes()
            current_edges = set(reconstructed_graph.edges())
            for u in range(n):
                for v in range(n):
                    if u != v and (u, v) not in current_edges and (v, u) not in current_edges:
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            # Update the nx_graph to match the reconstructed graph
            nx_graph = reconstructed_graph
            
            # Final check: ensure the graph is both ring-compliant and chemically valid
            if self.atom_decoder is not None:
                atom_types = z_s.X[graph_idx]
                if not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder):
                    # If chemically invalid, remove edges until valid
                    while not is_chemically_valid_graph(nx_graph, atom_types, self.atom_decoder) and nx_graph.number_of_edges() > 0:
                        # Remove a random edge
                        edge_to_remove = list(nx_graph.edges())[0]
                        nx_graph.remove_edge(*edge_to_remove)
                        # Update tensor
                        u, v = edge_to_remove
                        z_s.E[graph_idx, u, v] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, v, u] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # Mark as blocked
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)
    
    def _enhanced_ring_length_projector(self, graph: nx.Graph, min_length: int, z_s: PlaceHolder, graph_idx: int) -> nx.Graph:
        """
        Enhanced ring length projector with dynamic tensor resizing capability.
        
        Args:
            graph: NetworkX graph to modify
            min_length: Minimum ring length required
            z_s: PlaceHolder tensor for resizing
            graph_idx: Index of the graph being modified
        
        Returns:
            Modified graph with rings of sufficient length
        """
        cycles = nx.cycle_basis(graph)
        
        # Check if we already have rings of sufficient length
        for cycle in cycles:
            if len(cycle) >= min_length:
                return graph
        
        nodes = list(graph.nodes())
        current_size = len(nodes)
        
        # Try to create ring with existing nodes first
        ring_created = False
        if current_size >= min_length:
            # Try to create ring using existing nodes
            for i in range(current_size):
                for j in range(i + 1, current_size):
                    if not graph.has_edge(nodes[i], nodes[j]):
                        # Try to create a path and close it
                        try:
                            path = nx.shortest_path(graph, nodes[i], nodes[j])
                            if len(path) == min_length - 1:
                                graph.add_edge(nodes[i], nodes[j])
                                ring_created = True
                                break
                        except nx.NetworkXNoPath:
                            continue
                if ring_created:
                    break
        
        # If no ring created with existing nodes, resize tensor and add new nodes
        if not ring_created:
            # Calculate how many new nodes we need
            new_nodes_needed = max(0, min_length - current_size)
            if new_nodes_needed > 0:
                # Resize tensor to accommodate new nodes
                new_size = current_size + new_nodes_needed
                resize_placeholder_tensor(z_s, new_size, graph_idx)
                
                # Create ring with new nodes
                graph = create_ring_with_new_nodes(graph, min_length, current_size)
            else:
                # Fallback: create largest possible ring with existing nodes
                if current_size >= 3:
                    for i in range(current_size):
                        u = nodes[i]
                        v = nodes[(i + 1) % current_size]
                        graph.add_edge(u, v)
        
        return graph
