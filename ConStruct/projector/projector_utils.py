import abc
import logging

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_simple_graph_from_edge_tensor(edge_tensor: torch.Tensor, mask: torch.Tensor):
    """
    Unified graph construction helper for consistent graph building across all components.
    
    Args:
        edge_tensor: [N, N, C] or [N, N] - edge tensor with channels
        mask: [N] boolean or 0/1 tensor - nodes to keep
        
    Returns:
        networkx.Graph: simple, undirected graph with no parallel edges
        
    Notes:
        - Channels: 0 = no-edge, 1..3 = bond types we enforce; ignore others
        - Ensures consistent graph construction across projectors, WandB, and tables
        - Uses structural semantics (cycle-basis based) for ring constraints
    """
    n = int(mask.sum().item()) if mask is not None else edge_tensor.shape[0]
    et = edge_tensor[:n, :n]
    
    if et.dim() == 3:
        # Multi-channel edge tensor: sum across bond types 1-3 (ignore channel 0 and others)
        adj = (et[..., 1:4].sum(dim=-1) > 0).int()
    else:
        # Single-channel edge tensor: treat as binary
        adj = (et > 0).int()
    
    # Convert to NetworkX graph
    g = nx.from_numpy_array(adj.cpu().numpy())
    
    # Ensure simple undirected graph (no self-loops, no parallel edges)
    g = nx.Graph(g)
    
    return g


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


def check_valence_constraints(nx_graph, atom_types, atom_decoder):
    """
    Check if adding an edge would violate valence constraints.
    Returns True if the edge addition is chemically valid.
    """
    try:
        # Get current valence for each atom
        valence_count = {}
        for node in nx_graph.nodes():
            valence_count[node] = 0
        
        # Count current bonds
        for edge in nx_graph.edges():
            u, v = edge
            valence_count[u] += 1
            valence_count[v] += 1
        
        # Define maximum valence for common atoms (for edge addition checking)
        max_valence = {
            'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6,
            'Cl': 1, 'Br': 1, 'I': 1, 'H': 1
        }
        
        # Check if any atom exceeds maximum valence
        for node, valence in valence_count.items():
            if atom_types[node] == -1:
                continue
            atom_symbol = atom_decoder[int(atom_types[node].item())]
            
            if atom_symbol in max_valence and valence > max_valence[atom_symbol]:
                return False
        
        return True
    except:
        return False


def can_add_edge_safely(nx_graph, u, v, atom_types, atom_decoder):
    """
    Check if adding edge (u,v) would be chemically valid.
    Returns True if the edge can be safely added.
    """
    # Check if edge already exists
    if nx_graph.has_edge(u, v):
        return False
    
    # Check if adding edge would violate valence constraints
    trial_graph = nx_graph.copy()
    trial_graph.add_edge(u, v)
    
    return check_valence_constraints(trial_graph, atom_types, atom_decoder)


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
    # Only consider edge types 1, 2, 3 (single, double, triple bonds), not 0 (no edge)
    z_t_adj = (z_t.E[:, :, :, 1:4].sum(dim=3) > 0).int()  # Sum across edge types 1,2,3 and check if > 0
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

        # Initialize total blocked edges counter
        self.total_blocked = 0

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_adj_matrix(z_t)

        # add data structure where planarity is checked
        for graph_idx in range(self.batch_size):
            # Use unified graph construction helper
            edge_mat = z_t.E[graph_idx]
            mask = z_t.node_mask[graph_idx]
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            # For edge-insertion, we validate the initial graph
            # because we need to add edges to satisfy constraints
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list with bounds checking
            if self.can_block_edges:
                num_nodes = nx_graph.number_of_nodes()
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False

    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_adj_matrix(z_s)
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        
        # Process each graph in the batch
        new_edges = diff_adj.nonzero(as_tuple=False)
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edges_to_add = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ][:, 1:]
            )
            # FIX: Ensure proper GPU tensor handling
            if edges_to_add.is_cuda:
                edges_to_add = edges_to_add.cpu()
            edges_to_add = edges_to_add.numpy()

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
                        self.total_blocked += 1
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
                        self.total_blocked += 1
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

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
    def __init__(self, z_t: PlaceHolder):
        super().__init__(z_t)
        # Initialize planarity statistics
        self.total_blocked = 0
        self.total_edges_checked = 0
        self.verbose = False  # Set to True for detailed debugging
        
        # Print mode information once per class instance
        if not hasattr(self.__class__, '_printed_planar_mode'):
            # print(f"🔧 PlanarProjector initialized")
            # print(f"   🎯 Enforcing planarity constraint")
            self.__class__._printed_planar_mode = True

    def valid_graph_fn(self, nx_graph):
        return is_planar(nx_graph)

    @property
    def can_block_edges(self):
        return True
    
    def project(self, z_s: PlaceHolder):
        """
        Project graphs to satisfy planarity constraint.
        
        This implements planarity enforcement by blocking edges that would
        create non-planar graphs during reverse diffusion.
        """
        # Get the current adjacency matrix from z_s
        current_adj = get_adj_matrix(z_s)
        
        # Track projection statistics
        total_edges_checked = 0
        edges_blocked = 0
        
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Find edges that are present in z_s but not in the original z_t
            original_adj = self.z_t_adj[graph_idx]
            new_edges = torch.where(current_adj[graph_idx] > original_adj)
            
            for i in range(len(new_edges[0])):
                u, v = new_edges[0][i].item(), new_edges[1][i].item()
                if u >= v:  # Skip duplicate edges (undirected graph)
                    continue
                    
                edge_tuple = (u, v)
                total_edges_checked += 1
                
                # Add edge and check if it makes the graph non-planar
                nx_graph.add_edge(u, v)
                if not self.valid_graph_fn(nx_graph):
                    # Remove the edge if it violates planarity
                    nx_graph.remove_edge(u, v)
                    z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                    z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                    edges_blocked += 1
                    self.total_blocked += 1

            self.nx_graphs_list[graph_idx] = nx_graph



        self.z_t_adj = get_adj_matrix(z_s)


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
    Edge-Deletion Projector: Ensures graphs have at most N cycles (cycle rank ≤ K).
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (cycle rank) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Mathematical Construction:
    - Constraint: "Cycle rank ≤ K" (structural only)
    - Forward diffusion: Edges progressively disappear toward no-edge state
    - Reverse diffusion: Edges are added while preserving max cycle rank constraint
    - Natural bias: toward sparse graphs (fewer edges = fewer cycle possibilities)
    
    Structural Constraint:
    - Count cycles using NetworkX cycle_basis() (cycle rank = basis size)
    - Remove edges that break excess cycles
    - Block edge additions that would exceed maximum cycle rank
    
    CRITICAL: Chemical validity (valency, connectivity, atom types) is NOT enforced.
    These properties are measured separately after generation using RDKit.
    
    Usage:
    - Transition: 'absorbing_edges'
    - Config: rev_proj: 'ring_count_at_most', max_rings: K, use_incremental: bool
    - Post-generation: Run RDKit validation to measure chemical properties
    
    MODES:
    - Baseline mode: Full cycle enumeration after each candidate edge removal
    - Efficient mode: Incremental enforcement using shortest-path detection and blocked-edge hash set
    """
    
    def __init__(self, z_t: PlaceHolder, max_rings: int, atom_decoder=None, use_incremental=False):
        self.max_rings = max_rings
        self.use_incremental = use_incremental
        # Note: atom_decoder is kept for compatibility but NOT used for chemical validation
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        
        # Only print mode information once per class instance
        if not hasattr(self.__class__, '_printed_mode'):
            # print(f"🔧 RingCountAtMostProjector initialized: max_rings={max_rings}, use_incremental={use_incremental}")
            # if self.use_incremental:
            #     print(f"   🚀 Using INCREMENTAL mode (efficient)")
            # else:
            #     print(f"   🐌 Using BASELINE mode (full recomputation)")
            self.__class__._printed_mode = True
        
        # Initialize blocked edges set for efficient mode
        if self.use_incremental:
            self.blocked_edges = {i: set() for i in range(self.batch_size)}
            # Initialize current ring counts correctly
            self.current_ring_counts = {}
            for i in range(self.batch_size):
                # Reconstruct the graph from the adjacency matrix to get correct ring count
                adj_matrix = self.z_t_adj[i].cpu().numpy()
                nx_graph = nx.from_numpy_array(adj_matrix)
                self.current_ring_counts[i] = len(nx.cycle_basis(nx_graph))
        
        # Verbose logging flag (set to True for detailed debugging)
        self.verbose = False

    def valid_graph_fn(self, nx_graph):
        """
        Check if graph satisfies structural constraint: cycle rank ≤ K.
        
        This function ONLY checks structural properties (cycle rank) and does
        NOT enforce chemical validity, valency, or connectivity.
        
        Args:
            nx_graph: NetworkX graph to validate
            
        Returns:
            bool: True if graph has cycle rank ≤ max_rings
        """
        # Use NetworkX to count cycles (cycle rank = basis size) - structural constraint only
        cycles = nx.cycle_basis(nx_graph)
        cycle_rank = len(cycles)
        is_valid = cycle_rank <= self.max_rings
        # print(f"   🔍 RingCountAtMostProjector.valid_graph_fn(): {cycle_rank} cycles, max={self.max_rings}, valid={is_valid}")
        return is_valid  # Cycle rank ≤ K

    @property
    def can_block_edges(self):
        """Can block edge additions that would violate structural constraint."""
        return True
    
    def project(self, z_s: PlaceHolder):
        """
        Project graphs to satisfy ring-count-at-most constraint.
        
        This implements both baseline mode (full recomputation) and efficient mode
        (incremental checking with blocked-edge hashing) as described in the explanation.
        """
        # Timing for projection
        projection_start_time = time.time()
        
        # Get the current adjacency matrix from z_s
        current_adj = get_adj_matrix(z_s)
        
        # Track projection statistics
        total_edges_checked = 0
        edges_blocked = 0
        
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Find edges that are present in z_s but not in the original z_t
            original_adj = self.z_t_adj[graph_idx]
            new_edges = torch.where(current_adj[graph_idx] > original_adj)
            
            for i in range(len(new_edges[0])):
                u, v = new_edges[0][i].item(), new_edges[1][i].item()
                if u >= v:  # Skip duplicate edges (undirected graph)
                    continue
                    
                edge_tuple = (u, v)
                total_edges_checked += 1
                
                if self.use_incremental:
                    if edge_tuple in self.blocked_edges[graph_idx]:
                        # Blocked already, reject immediately
                        z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        edges_blocked += 1
                        self.total_blocked += 1
                        continue

                    # Check if adding this edge would create a new ring
                    if nx.has_path(nx_graph, u, v):
                        # This edge would create a new ring
                        if self.current_ring_counts[graph_idx] + 1 > self.max_rings:
                            # Block permanently
                            self.blocked_edges[graph_idx].add(edge_tuple)
                            z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                            z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                            edges_blocked += 1
                            self.total_blocked += 1
                        else:
                            # Allow the edge
                            nx_graph.add_edge(u, v)
                            self.current_ring_counts[graph_idx] += 1
                    else:
                        # No path exists, so no new ring created
                        nx_graph.add_edge(u, v)
                else:
                    # Baseline mode: add edge and check if valid
                    nx_graph.add_edge(u, v)
                    if not self.valid_graph_fn(nx_graph):
                        # Remove the edge if it violates constraint
                        nx_graph.remove_edge(u, v)
                        z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        edges_blocked += 1
                        self.total_blocked += 1

            self.nx_graphs_list[graph_idx] = nx_graph

        # Log projection timing (only at key timesteps to reduce noise)
        projection_time = time.time() - projection_start_time
        if hasattr(self, 'current_timestep') and self.current_timestep % 100 == 0:
            logger.info(f"⏱️ RingCountAtMostProjector: projection_time={projection_time:.4f}s, mode={'INCREMENTAL' if self.use_incremental else 'BASELINE'}")

        self.z_t_adj = get_adj_matrix(z_s)


class RingLengthAtMostProjector(AbstractProjector):
    """
    Edge-Deletion Projector: Ensures graphs have max basis-cycle length ≤ L.
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (max basis-cycle length) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Mathematical Construction:
    - Constraint: "Max basis-cycle length ≤ L" (structural only)
    - Forward diffusion: Edges progressively disappear toward no-edge state
    - Reverse diffusion: Edges are added while preserving max basis-cycle length constraint
    - Natural bias: toward sparse graphs (fewer edges = fewer cycle possibilities)
    
    Structural Constraint:
    - Check basis-cycle lengths using NetworkX cycle_basis()
    - Block edge additions that would create cycles longer than maximum
    - Remove edges that create cycles longer than allowed
    
    CRITICAL: Chemical validity (valency, connectivity, atom types) is NOT enforced.
    These properties are measured separately after generation using RDKit.
    
    Usage:
    - Transition: 'absorbing_edges'
    - Config: rev_proj: 'ring_length_at_most', max_ring_length: L, use_incremental: bool
    - Post-generation: Run RDKit validation to measure chemical properties
    
    MODES:
    - Baseline mode: Full cycle enumeration after each candidate edge removal
    - Efficient mode: Incremental enforcement using shortest-path detection and blocked-edge hash set
    """
    
    def __init__(self, z_t: PlaceHolder, max_ring_length: int, atom_decoder=None, use_incremental=False):
        self.max_ring_length = max_ring_length
        self.use_incremental = use_incremental
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        
        # Only print mode information once per class instance
        if not hasattr(self.__class__, '_printed_mode'):
            # print(f"🔧 RingLengthAtMostProjector initialized: max_ring_length={max_ring_length}, use_incremental={use_incremental}")
            # if self.use_incremental:
            #     print(f"   🚀 Using INCREMENTAL mode (efficient)")
            # else:
            #     print(f"   🐌 Using BASELINE mode (full recomputation)")
            self.__class__._printed_mode = True
        
        # Initialize blocked edges set for efficient mode
        if self.use_incremental:
            self.blocked_edges = {i: set() for i in range(self.batch_size)}
        
        # Verbose logging flag (set to True for detailed debugging)
        self.verbose = False

    def valid_graph_fn(self, nx_graph):
        """Check if max basis-cycle length ≤ L."""
        cycles = nx.cycle_basis(nx_graph)
        for cycle in cycles:
            if len(cycle) > self.max_ring_length:
                return False
        return True

    @property
    def can_block_edges(self):
        """Can block edge additions that would violate constraint."""
        return True
    
    def project(self, z_s: PlaceHolder):
        """
        Project graphs to satisfy ring-length-at-most constraint.
        
        This implements both baseline mode (full recomputation) and efficient mode
        (incremental checking with blocked-edge hashing) as described in the explanation.
        """
        # Timing for projection
        projection_start_time = time.time()
        
        # Get the current adjacency matrix from z_s
        current_adj = get_adj_matrix(z_s)
        
        # Track projection statistics
        total_edges_checked = 0
        edges_blocked = 0
        
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Find edges that are present in z_s but not in the original z_t
            original_adj = self.z_t_adj[graph_idx]
            new_edges = torch.where(current_adj[graph_idx] > original_adj)
            
            for i in range(len(new_edges[0])):
                u, v = new_edges[0][i].item(), new_edges[1][i].item()
                if u >= v:  # Skip duplicate edges (undirected graph)
                    continue
                    
                edge_tuple = (u, v)
                total_edges_checked += 1
                
                if self.use_incremental:
                    if edge_tuple in self.blocked_edges[graph_idx]:
                        z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        # Edge already blocked
                        edges_blocked += 1
                        continue

                    # Check if adding this edge would create a ring longer than allowed
                    try:
                        path_length = nx.shortest_path_length(nx_graph, u, v)
                        potential_cycle_length = path_length + 1
                    except nx.NetworkXNoPath:
                        potential_cycle_length = 0

                    if potential_cycle_length > self.max_ring_length:
                        # Block permanently
                        self.blocked_edges[graph_idx].add(edge_tuple)
                        z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        # Edge blocked - would create ring longer than allowed
                        edges_blocked += 1
                    else:
                        # Allow the edge
                        nx_graph.add_edge(u, v)
                        # Edge allowed
                else:
                    # Baseline mode: add edge and check if valid
                    nx_graph.add_edge(u, v)
                    if not self.valid_graph_fn(nx_graph):
                        # Remove the edge if it violates constraint
                        nx_graph.remove_edge(u, v)
                        z_s.E[graph_idx, u, v] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        z_s.E[graph_idx, v, u] = F.one_hot(torch.tensor(0), num_classes=z_s.E.shape[-1])
                        edges_blocked += 1

            self.nx_graphs_list[graph_idx] = nx_graph

        # Log projection timing (only at key timesteps to reduce noise)
        projection_time = time.time() - projection_start_time
        if hasattr(self, 'current_timestep') and self.current_timestep % 100 == 0:
            logger.info(f"⏱️ RingLengthAtMostProjector: projection_time={projection_time:.4f}s, mode={'INCREMENTAL' if self.use_incremental else 'BASELINE'}")

        self.z_t_adj = get_adj_matrix(z_s)


class RingCountAtLeastProjector(AbstractProjector):
    """
    Edge-Insertion Projector: Ensures graphs have at least N rings.
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (ring count) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Mathematical Construction:
    - Constraint: "At least N rings" (structural only)
    - Forward diffusion: Edges progressively appear toward edge state
    - Reverse diffusion: Edges are removed while preserving min ring constraint
    - Natural bias: toward connected graphs (more edges = more ring possibilities)
    
    Structural Constraint:
    - Count cycles using NetworkX cycle_basis()
    - Block edge removals that would drop ring count below minimum
    - Allow edge additions that create new rings
    
    CRITICAL: Chemical validity (valency, connectivity, atom types) is NOT enforced.
    These properties are measured separately after generation using RDKit.
    
    Usage:
    - Transition: 'edge_insertion'
    - Config: rev_proj: 'ring_count_at_least', min_rings: N
    - Post-generation: Run RDKit validation to measure chemical properties
    """
    
    def __init__(self, z_t: PlaceHolder, min_rings: int, atom_decoder=None):
        self.min_rings = min_rings
        # Note: atom_decoder is kept for compatibility but NOT used for chemical validation
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        # Store z_t for edge-insertion projection
        self.z_t = z_t

    def valid_graph_fn(self, nx_graph):
        """
        Check if graph satisfies structural constraint: at least N rings.
        
        This function ONLY checks structural properties (ring count) and does
        NOT enforce chemical validity, valency, or connectivity.
        
        Args:
            nx_graph: NetworkX graph to validate
            
        Returns:
            bool: True if graph has at least min_rings cycles
        """
        # Use NetworkX to count cycles (rings) - structural constraint only
        cycles = nx.cycle_basis(nx_graph)
        
        # For edge-insertion with "at least" constraints:
        # - We start from fully connected graphs (many rings)
        # - We remove edges until we have exactly min_rings
        # - We NEVER allow graphs with fewer than min_rings
        return len(cycles) >= self.min_rings  # At least N rings (NO EXCEPTIONS!)

    @property
    def can_block_edges(self):
        """Can block edge removals that would violate structural constraint."""
        return True
    
    def project(self, z_s: PlaceHolder):
        """
        Edge-insertion projection: Add edges to satisfy minimum ring constraints.
        
        IMPROVED ALGORITHM: Uses strategic ring creation instead of random edge additions.
        
        Strategy:
        1. Find existing paths that could form rings when connected
        2. Prioritize edges that connect nodes with existing paths
        3. Use graph theory to identify ring-forming opportunities
        4. Ensure proper tensor synchronization
        
        This projector ONLY enforces structural constraints (ring count) and does
        NOT enforce chemical validity, valency, or connectivity. Chemical properties
        are measured separately after generation.
        
        Args:
            z_s: PlaceHolder tensor to project
        """
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Check current ring count (structural constraint only)
            cycles = nx.cycle_basis(nx_graph)
            current_rings = len(cycles)
            
            # If we already have enough rings, no action needed
            if current_rings >= self.min_rings:
                continue
            
            # ===== IMPROVED STRATEGIC RING CREATION =====
            attempts = 0
            max_attempts = 100  # Increased for better coverage
            
            while current_rings < self.min_rings and attempts < max_attempts:
                attempts += 1
                ring_created = False
                
                # Strategy 1: Find nodes that could form rings when connected
                nodes = list(nx_graph.nodes())
                n_nodes = len(nodes)
                
                if n_nodes < 3:
                    # Need at least 3 nodes to form a ring
                    continue
                
                # Strategy 2: Look for nodes that are close but not directly connected
                # These are more likely to form rings when connected
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        node1, node2 = nodes[i], nodes[j]
                        
                        # Skip if edge already exists
                        if nx_graph.has_edge(node1, node2):
                            continue
                        
                        # Strategy 3: Check if connecting these nodes would create a cycle
                        # by looking for existing paths between them
                        try:
                            # Check if there's already a path between these nodes
                            path_exists = nx.has_path(nx_graph, node1, node2)
                            
                            if path_exists:
                                # Connecting nodes with existing path creates a cycle!
                                nx_graph.add_edge(node1, node2)
                                new_cycles = nx.cycle_basis(nx_graph)
                                
                                if len(new_cycles) > current_rings:
                                    # SUCCESS: This edge created a new ring
                                    # Edge successfully added
                                    z_s.E[graph_idx, node1, node2, 1] = 1  # single bond
                                    z_s.E[graph_idx, node2, node1, 1] = 1  # undirected
                                    ring_created = True
                                    current_rings = len(new_cycles)
                                    break
                                else:
                                    # Remove edge if it didn't create a ring
                                    nx_graph.remove_edge(node1, node2)
                        except nx.NetworkXNoPath:
                            # No path exists, try next pair
                            continue
                    
                if ring_created:
                                                        # print(f"🔍 DEBUG: Ring created in attempt {attempts}")  # Detailed debugging - commented for clean output
                    break
                
                # Strategy 4: If no rings created with existing paths, try creating triangles
                if not ring_created and n_nodes >= 3:
                                                    # print(f"🔍 DEBUG: Trying triangle creation strategy")  # Detailed debugging - commented for clean output
                    for i in range(n_nodes):
                        for j in range(i + 1, n_nodes):
                            for k in range(j + 1, n_nodes):
                                node1, node2, node3 = nodes[i], nodes[j], nodes[k]
                                
                                # Check if we can create a triangle
                                edges_to_add = []
                                if not nx_graph.has_edge(node1, node2):
                                    edges_to_add.append((node1, node2))
                                if not nx_graph.has_edge(node2, node3):
                                    edges_to_add.append((node2, node3))
                                if not nx_graph.has_edge(node3, node1):
                                    edges_to_add.append((node3, node1))
                                
                                # Add all edges to create triangle
                                for u, v in edges_to_add:
                                    nx_graph.add_edge(u, v)
                                    z_s.E[graph_idx, u, v, 1] = 1  # single bond
                                    z_s.E[graph_idx, v, u, 1] = 1  # undirected
                                
                                # Check if this created a ring
                                new_cycles = nx.cycle_basis(nx_graph)
                                if len(new_cycles) > current_rings:
                                    ring_created = True
                                    current_rings = len(new_cycles)
                                    break
                                else:
                                    # Remove edges if no ring created
                                    for u, v in edges_to_add:
                                        nx_graph.remove_edge(u, v)
                                        z_s.E[graph_idx, u, v, 1] = 0
                                        z_s.E[graph_idx, v, u, 1] = 0
                            
                            if ring_created:
                                break
                        
                        if ring_created:
                            break
                
                # Strategy 5: If still no rings, try connecting nodes with degree 1
                # These are leaf nodes that can be connected to form rings
                if not ring_created:
                    leaf_nodes = [n for n in nx_graph.nodes() if nx_graph.degree(n) == 1]
                    if len(leaf_nodes) >= 2:
                        for i in range(len(leaf_nodes)):
                            for j in range(i + 1, len(leaf_nodes)):
                                node1, node2 = leaf_nodes[i], leaf_nodes[j]
                                
                                if not nx_graph.has_edge(node1, node2):
                                    nx_graph.add_edge(node1, node2)
                                    z_s.E[graph_idx, node1, node2, 1] = 1
                                    z_s.E[graph_idx, node2, node1, 1] = 1
                                    
                                    new_cycles = nx.cycle_basis(nx_graph)
                                    if len(new_cycles) > current_rings:
                                        ring_created = True
                                        current_rings = len(new_cycles)
                                        break
                                    else:
                                        nx_graph.remove_edge(node1, node2)
                                        z_s.E[graph_idx, node1, node2, 1] = 0
                                        z_s.E[graph_idx, node2, node1, 1] = 0
                            
                            if ring_created:
                                break
                
                # Update ring count for next iteration
                if not ring_created:
                    current_rings = len(nx.cycle_basis(nx_graph))
            
            # Mark edges as blocked to prevent future violations
            n = nx_graph.number_of_nodes()
            current_edges = set(nx_graph.edges())
            for u in range(n):
                for v in range(n):
                    if u != v and (u, v) not in current_edges and (v, u) not in current_edges:
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][(u, v)] = True
            
            self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

        # Print summary statistics
        if hasattr(self, 'current_timestep') and self.current_timestep == 0:
            print(f"🔍 FINAL: Total edges added for ring count at least across all timesteps: {getattr(self, 'total_added', 0)}")
        
        # store modified z_s
        self.z_t_adj = get_adj_matrix(z_s)
        # RingCountAtLeastProjector.project() completed


class RingLengthAtLeastProjector(AbstractProjector):
    """
    Edge-Insertion Projector: Ensures graphs have rings of at least N length.
    
    CONSTRUCT PHILOSOPHY: This projector enforces ONLY structural constraints
    (ring length) and does NOT enforce chemical validity, valency, or connectivity.
    Chemical properties are measured post-generation.
    
    Mathematical Construction:
    - Constraint: "All rings have length at least N" (structural only)
    - Forward diffusion: Edges progressively appear toward edge state
    - Reverse diffusion: Edges are removed while preserving min ring length constraint
    - Natural bias: toward connected graphs (more edges = more ring possibilities)
    
    Structural Constraint:
    - Check ring lengths using NetworkX cycle_basis()
    - Block edge removals that would create rings shorter than minimum
    - Allow edge additions that create rings of sufficient length
    
    CRITICAL: Chemical validity (valency, connectivity, atom types) is NOT enforced.
    These properties are measured separately after generation using RDKit.
    
    Usage:
    - Transition: 'edge_insertion'
    - Config: rev_proj: 'ring_length_at_least', min_ring_length: N
    - Post-generation: Run RDKit validation to measure chemical properties
    """
    
    def __init__(self, z_t: PlaceHolder, min_ring_length: int, atom_decoder=None):
        self.min_ring_length = min_ring_length
        # Note: atom_decoder is kept for compatibility but NOT used for chemical validation
        self.atom_decoder = atom_decoder
        super().__init__(z_t)
        # Store z_t for edge-insertion projection
        self.z_t = z_t

    def valid_graph_fn(self, nx_graph):
        """
        Check if graph satisfies structural constraint: all rings have length at least N.
        
        This function ONLY checks structural properties (ring length) and does
        NOT enforce chemical validity, valency, or connectivity.
        
        Args:
            nx_graph: NetworkX graph to validate
            
        Returns:
            bool: True if all rings have length at least min_ring_length
        """
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
        """Can block edge removals that would violate structural constraint."""
        return True
    
    def project(self, z_s: PlaceHolder):
        """Edge-insertion projection with dynamic tensor resizing: ADD edges to satisfy ring length constraints."""
        # Process each graph in the batch
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            # Check current ring lengths (structural constraint only)
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
            
            # ===== STRUCTURAL-ONLY CONSTRAINT ENFORCEMENT =====
            # Strategy: Create rings of sufficient length (structural constraint only)
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
                                    # ===== NO CHEMICAL VALIDITY CHECKS =====
                                    # We do NOT check valency, connectivity, or atom types here
                                    # These properties are measured post-generation
                                    
                                    # Add edge to create ring of sufficient length
                                    nx_graph.add_edge(nodes[i], nodes[j])
                                    
                                    # Update tensor
                                    z_s.E[graph_idx, nodes[i], nodes[j], 1] = 1  # single bond
                                    z_s.E[graph_idx, nodes[j], nodes[i], 1] = 1  # undirected
                                    
                                    ring_created = True
                                    break
                            except nx.NetworkXNoPath:
                                continue
                    
                    if ring_created:
                        break
            
            # If we couldn't create ring with existing nodes, resize tensor and add new nodes
            if not ring_created:
                    # Resize tensor to accommodate new nodes
                new_size = max(current_size + self.min_ring_length, current_size * 2)
                z_s = resize_placeholder_tensor(z_s, new_size, graph_idx)
                    
                    # Create ring with new nodes
                nx_graph = self._enhanced_ring_length_projector(
                    nx_graph, self.min_ring_length, z_s, graph_idx
                )
            
            # Mark edges as blocked to prevent future violations
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
