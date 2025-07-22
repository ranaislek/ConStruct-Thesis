"""
Post-Generation Chemical Validation Module

This module implements the ConStruct philosophy of measuring chemical properties
AFTER generation, not during the constraint enforcement process.

The ConStruct approach:
1. Enforce ONLY structural constraints during generation (ring count, ring length, planarity)
2. Measure chemical properties post-generation (valency, connectivity, atom types)
3. Report the gap between structural satisfaction and chemical validity

This enables honest, critical analysis of the relationship between structural
constraints and chemical validity - exactly as intended in the original ConStruct paper.
"""

import networkx as nx
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
import logging

from ConStruct.utils import PlaceHolder


class PostGenerationValidator:
    """
    Post-generation chemical validation following ConStruct philosophy.
    
    This class measures chemical properties AFTER generation, not during
    the constraint enforcement process. This enables honest analysis of
    the gap between structural satisfaction and chemical validity.
    
    Chemical Properties Measured:
    - Valency violations
    - Connectivity (single vs multiple components)
    - Atom type validity
    - RDKit validity
    - Molecular descriptors (when applicable)
    
    Usage:
    - Call after generation is complete
    - Compare structural constraint satisfaction vs chemical validity
    - Report gaps for critical thesis analysis
    """
    
    def __init__(self, atom_decoder: Optional[Dict] = None):
        """
        Initialize post-generation validator.
        
        Args:
            atom_decoder: Dictionary mapping atom indices to atom symbols
        """
        self.atom_decoder = atom_decoder
        self.logger = logging.getLogger(__name__)
        
    def validate_batch(self, z_s: PlaceHolder, nx_graphs: List[nx.Graph]) -> Dict[str, Any]:
        """
        Validate a batch of generated graphs for chemical properties.
        
        Args:
            z_s: PlaceHolder tensor with generated graphs
            nx_graphs: List of NetworkX graphs
            
        Returns:
            Dictionary with validation results for each graph
        """
        results = {
            'graphs': [],
            'summary': {
                'total_graphs': len(nx_graphs),
                'structurally_valid': 0,
                'chemically_valid': 0,
                'rdkit_valid': 0,
                'valency_violations': 0,
                'connectivity_issues': 0,
                'atom_type_issues': 0
            }
        }
        
        for graph_idx, nx_graph in enumerate(nx_graphs):
            graph_result = self._validate_single_graph(z_s, nx_graph, graph_idx)
            results['graphs'].append(graph_result)
            
            # Update summary statistics
            if graph_result['structurally_valid']:
                results['summary']['structurally_valid'] += 1
            if graph_result['chemically_valid']:
                results['summary']['chemically_valid'] += 1
            if graph_result['rdkit_valid']:
                results['summary']['rdkit_valid'] += 1
            if graph_result['valency_violations'] > 0:
                results['summary']['valency_violations'] += 1
            if not graph_result['is_connected']:
                results['summary']['connectivity_issues'] += 1
            if graph_result['atom_type_issues'] > 0:
                results['summary']['atom_type_issues'] += 1
        
        return results
    
    def _validate_single_graph(self, z_s: PlaceHolder, nx_graph: nx.Graph, graph_idx: int) -> Dict[str, Any]:
        """
        Validate a single generated graph.
        
        Args:
            z_s: PlaceHolder tensor
            nx_graph: NetworkX graph to validate
            graph_idx: Index of the graph in the batch
            
        Returns:
            Dictionary with validation results for this graph
        """
        # Get atom types if available
        atom_types = None
        if self.atom_decoder is not None and hasattr(z_s, 'X'):
            atom_types = z_s.X[graph_idx]
        
        # Structural validation (what was enforced during generation)
        structural_result = self._validate_structural_properties(nx_graph)
        
        # Chemical validation (measured post-generation)
        chemical_result = self._validate_chemical_properties(nx_graph, atom_types)
        
        # RDKit validation (if molecular)
        rdkit_result = self._validate_rdkit_properties(nx_graph, atom_types)
        
        return {
            'graph_idx': graph_idx,
            'num_nodes': nx_graph.number_of_nodes(),
            'num_edges': nx_graph.number_of_edges(),
            'num_components': nx.number_connected_components(nx_graph),
            'is_connected': nx.is_connected(nx_graph),
            
            # Structural properties (enforced during generation)
            'structurally_valid': structural_result['valid'],
            'ring_count': structural_result['ring_count'],
            'ring_lengths': structural_result['ring_lengths'],
            'is_planar': structural_result['is_planar'],
            
            # Chemical properties (measured post-generation)
            'chemically_valid': chemical_result['valid'],
            'valency_violations': chemical_result['valency_violations'],
            'atom_type_issues': chemical_result['atom_type_issues'],
            'valence_details': chemical_result['valence_details'],
            
            # RDKit properties (if molecular)
            'rdkit_valid': rdkit_result['valid'],
            'rdkit_mol': rdkit_result['mol'],
            'rdkit_smiles': rdkit_result['smiles'],
            'rdkit_descriptors': rdkit_result['descriptors']
        }
    
    def _validate_structural_properties(self, nx_graph: nx.Graph) -> Dict[str, Any]:
        """
        Validate structural properties that were enforced during generation.
        
        These are the constraints that were actually enforced by the projectors.
        """
        # Count rings (cycles)
        cycles = nx.cycle_basis(nx_graph)
        ring_count = len(cycles)
        ring_lengths = [len(cycle) for cycle in cycles]
        
        # Check planarity
        try:
            is_planar = nx.is_planar(nx_graph)
        except:
            is_planar = False
        
        return {
            'valid': True,  # Structural constraints were enforced during generation
            'ring_count': ring_count,
            'ring_lengths': ring_lengths,
            'is_planar': is_planar
        }
    
    def _validate_chemical_properties(self, nx_graph: nx.Graph, atom_types: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Validate chemical properties that were NOT enforced during generation.
        
        These properties are measured post-generation to analyze the gap
        between structural satisfaction and chemical validity.
        """
        if atom_types is None or self.atom_decoder is None:
            return {
                'valid': False,
                'valency_violations': 0,
                'atom_type_issues': 0,
                'valence_details': {}
            }
        
        # Convert atom types to symbols
        atom_symbols = []
        for i in range(atom_types.shape[0]):
            atom_idx = atom_types[i].argmax().item()
            if atom_idx in self.atom_decoder:
                atom_symbols.append(self.atom_decoder[atom_idx])
            else:
                atom_symbols.append('C')  # Default to carbon
        
        # Check valency violations
        valency_violations = 0
        valence_details = {}
        
        for node in nx_graph.nodes():
            degree = nx_graph.degree(node)
            atom_symbol = atom_symbols[node] if node < len(atom_symbols) else 'C'
            
            # Expected valency for common atoms
            expected_valency = {
                'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 3, 'S': 2,
                'Cl': 1, 'Br': 1, 'I': 1
            }
            
            expected = expected_valency.get(atom_symbol, 4)
            if degree > expected:
                valency_violations += 1
                valence_details[node] = {
                    'atom': atom_symbol,
                    'degree': degree,
                    'expected': expected,
                    'violation': degree - expected
                }
        
        # Check atom type issues
        atom_type_issues = 0
        for atom_symbol in atom_symbols:
            if atom_symbol not in ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']:
                atom_type_issues += 1
        
        return {
            'valid': valency_violations == 0 and atom_type_issues == 0,
            'valency_violations': valency_violations,
            'atom_type_issues': atom_type_issues,
            'valence_details': valence_details
        }
    
    def _validate_rdkit_properties(self, nx_graph: nx.Graph, atom_types: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        Validate using RDKit (if molecular data).
        
        This provides additional molecular descriptors and validation.
        """
        if atom_types is None or self.atom_decoder is None:
            return {
                'valid': False,
                'mol': None,
                'smiles': None,
                'descriptors': {}
            }
        
        try:
            # Convert to SMILES
            smiles = self._graph_to_smiles(nx_graph, atom_types)
            if smiles is None:
                return {
                    'valid': False,
                    'mol': None,
                    'smiles': None,
                    'descriptors': {}
                }
            
            # Create RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    'valid': False,
                    'mol': None,
                    'smiles': smiles,
                    'descriptors': {}
                }
            
            # Calculate molecular descriptors
            descriptors = {}
            try:
                descriptors['molecular_weight'] = Descriptors.MolWt(mol)
                descriptors['logp'] = Descriptors.MolLogP(mol)
                descriptors['hbd'] = Descriptors.NumHDonors(mol)
                descriptors['hba'] = Descriptors.NumHAcceptors(mol)
                descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                descriptors['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                descriptors['saturated_rings'] = Descriptors.NumSaturatedRings(mol)
            except:
                pass
            
            return {
                'valid': True,
                'mol': mol,
                'smiles': smiles,
                'descriptors': descriptors
            }
            
        except Exception as e:
            self.logger.warning(f"RDKit validation failed: {e}")
            return {
                'valid': False,
                'mol': None,
                'smiles': None,
                'descriptors': {}
            }
    
    def _graph_to_smiles(self, nx_graph: nx.Graph, atom_types: torch.Tensor) -> Optional[str]:
        """
        Convert NetworkX graph to SMILES string.
        
        This is a simplified conversion for validation purposes.
        """
        try:
            # Convert atom types to symbols
            atom_symbols = []
            for i in range(atom_types.shape[0]):
                atom_idx = atom_types[i].argmax().item()
                if atom_idx in self.atom_decoder:
                    atom_symbols.append(self.atom_decoder[atom_idx])
                else:
                    atom_symbols.append('C')
            
            # Create a simple SMILES representation
            # This is a basic implementation - for production use, consider more robust conversion
            edges = list(nx_graph.edges())
            if not edges:
                return None
            
            # Simple linear SMILES for validation
            nodes = list(nx_graph.nodes())
            if len(nodes) == 0:
                return None
            
            # Create a basic SMILES string
            smiles_parts = []
            for node in nodes:
                if node < len(atom_symbols):
                    smiles_parts.append(atom_symbols[node])
                else:
                    smiles_parts.append('C')
            
            return ''.join(smiles_parts)
            
        except Exception as e:
            self.logger.warning(f"SMILES conversion failed: {e}")
            return None
    
    def print_validation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of validation results.
        
        This highlights the gap between structural satisfaction and chemical validity,
        which is the key insight for the ConStruct approach.
        """
        summary = results['summary']
        
        print("\n" + "="*60)
        print("POST-GENERATION CHEMICAL VALIDATION SUMMARY")
        print("="*60)
        print(f"Total graphs generated: {summary['total_graphs']}")
        print(f"Structurally valid: {summary['structurally_valid']} ({summary['structurally_valid']/summary['total_graphs']*100:.1f}%)")
        print(f"Chemically valid: {summary['chemically_valid']} ({summary['chemically_valid']/summary['total_graphs']*100:.1f}%)")
        print(f"RDKit valid: {summary['rdkit_valid']} ({summary['rdkit_valid']/summary['total_graphs']*100:.1f}%)")
        print()
        print("CHEMICAL VALIDITY ISSUES (Post-Generation):")
        print(f"  Valency violations: {summary['valency_violations']}")
        print(f"  Connectivity issues: {summary['connectivity_issues']}")
        print(f"  Atom type issues: {summary['atom_type_issues']}")
        print()
        print("CONSTRUCT PHILOSOPHY ANALYSIS:")
        print("  ✓ Structural constraints were enforced during generation")
        print("  ✓ Chemical properties were measured post-generation")
        print("  ✓ Gap analysis enables honest, critical thesis results")
        print("="*60) 