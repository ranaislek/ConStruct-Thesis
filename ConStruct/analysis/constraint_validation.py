#!/usr/bin/env python3
"""
Constraint Validation Script
===========================

This script validates generated molecules for:
1. 100% Constraint Enforcement
2. Molecular Validity
3. Graph Connectedness
4. Quality Metrics

Usage:
    python constraint_validation.py --experiment_name <name> --output_dir <path>
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Chemical validation will be limited.")
    RDKIT_AVAILABLE = False

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch/PyTorch Geometric not available.")
    TORCH_AVAILABLE = False


class ConstraintValidator:
    """Validates molecular constraints and properties."""
    
    def __init__(self, experiment_name: str, output_dir: str):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.results = {
            'constraint_satisfaction': {},
            'validity_metrics': {},
            'graph_properties': {},
            'quality_metrics': {}
        }
    
    def load_generated_molecules(self) -> List[str]:
        """Load generated SMILES from experiment output."""
        # Look for generated molecules in various possible locations
        possible_paths = [
            self.output_dir / f"{self.experiment_name}" / "generated_molecules.txt",
            self.output_dir / f"{self.experiment_name}" / "samples" / "smiles.txt",
            self.output_dir / f"{self.experiment_name}" / "final_samples" / "smiles.txt",
        ]
        
        molecules = []
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    molecules = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(molecules)} molecules from {path}")
                break
        
        if not molecules:
            print(f"Warning: No molecules found in expected locations for {self.experiment_name}")
            return []
        
        return molecules
    
    def check_ring_count_constraint(self, molecules: List[str], min_rings: int) -> Dict:
        """Check if molecules satisfy ring count constraint."""
        if not RDKIT_AVAILABLE:
            return {'satisfaction_rate': 0.0, 'error': 'RDKit not available'}
        
        satisfied = 0
        total_valid = 0
        ring_counts = []
        
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    total_valid += 1
                    rings = mol.GetRingInfo()
                    num_rings = rings.NumRings()
                    ring_counts.append(num_rings)
                    
                    if num_rings >= min_rings:
                        satisfied += 1
            except:
                continue
        
        satisfaction_rate = satisfied / total_valid if total_valid > 0 else 0.0
        
        return {
            'satisfaction_rate': satisfaction_rate,
            'total_molecules': len(molecules),
            'valid_molecules': total_valid,
            'satisfied_constraint': satisfied,
            'ring_count_distribution': ring_counts,
            'min_rings_required': min_rings
        }
    
    def check_ring_length_constraint(self, molecules: List[str], min_ring_length: int) -> Dict:
        """Check if molecules satisfy ring length constraint."""
        if not RDKIT_AVAILABLE:
            return {'satisfaction_rate': 0.0, 'error': 'RDKit not available'}
        
        satisfied = 0
        total_valid = 0
        shortest_rings = []
        
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    total_valid += 1
                    rings = mol.GetRingInfo()
                    
                    # Find shortest ring length
                    min_ring_size = float('inf')
                    for ring in rings.AtomRings():
                        min_ring_size = min(min_ring_size, len(ring))
                    
                    if min_ring_size != float('inf'):
                        shortest_rings.append(min_ring_size)
                        if min_ring_size >= min_ring_length:
                            satisfied += 1
            except:
                continue
        
        satisfaction_rate = satisfied / total_valid if total_valid > 0 else 0.0
        
        return {
            'satisfaction_rate': satisfaction_rate,
            'total_molecules': len(molecules),
            'valid_molecules': total_valid,
            'satisfied_constraint': satisfied,
            'shortest_ring_distribution': shortest_rings,
            'min_ring_length_required': min_ring_length
        }
    
    def assess_molecular_validity(self, molecules: List[str]) -> Dict:
        """Assess molecular validity using RDKit."""
        if not RDKIT_AVAILABLE:
            return {'validity_rate': 0.0, 'error': 'RDKit not available'}
        
        valid_smiles = 0
        valid_molecules = 0
        valid_chemistry = 0
        molecular_weights = []
        logp_values = []
        
        for smiles in molecules:
            try:
                # Check SMILES validity
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles += 1
                    
                    # Check chemical reasonableness
                    try:
                        # Calculate molecular weight
                        mw = Descriptors.MolWt(mol)
                        molecular_weights.append(mw)
                        
                        # Calculate logP
                        logp = Descriptors.MolLogP(mol)
                        logp_values.append(logp)
                        
                        # Basic chemical validity checks
                        if (0 < mw < 1000 and  # Reasonable molecular weight
                            -5 < logp < 10):     # Reasonable logP
                            valid_chemistry += 1
                        
                        valid_molecules += 1
                    except:
                        pass
            except:
                continue
        
        validity_rate = valid_smiles / len(molecules) if molecules else 0.0
        chemistry_rate = valid_chemistry / valid_molecules if valid_molecules > 0 else 0.0
        
        return {
            'validity_rate': validity_rate,
            'chemistry_rate': chemistry_rate,
            'total_molecules': len(molecules),
            'valid_smiles': valid_smiles,
            'valid_molecules': valid_molecules,
            'valid_chemistry': valid_chemistry,
            'molecular_weight_stats': {
                'mean': np.mean(molecular_weights) if molecular_weights else 0,
                'std': np.std(molecular_weights) if molecular_weights else 0,
                'min': np.min(molecular_weights) if molecular_weights else 0,
                'max': np.max(molecular_weights) if molecular_weights else 0
            },
            'logp_stats': {
                'mean': np.mean(logp_values) if logp_values else 0,
                'std': np.std(logp_values) if logp_values else 0,
                'min': np.min(logp_values) if logp_values else 0,
                'max': np.max(logp_values) if logp_values else 0
            }
        }
    
    def check_graph_properties(self, molecules: List[str]) -> Dict:
        """Check graph properties like connectedness."""
        if not RDKIT_AVAILABLE:
            return {'connectedness_rate': 0.0, 'error': 'RDKit not available'}
        
        connected_graphs = 0
        total_valid = 0
        component_counts = []
        
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    total_valid += 1
                    
                    # Convert to graph
                    adj_matrix = Chem.GetAdjacencyMatrix(mol)
                    G = nx.from_numpy_array(adj_matrix)
                    
                    # Check connectedness
                    if nx.is_connected(G):
                        connected_graphs += 1
                    
                    # Count components
                    num_components = nx.number_connected_components(G)
                    component_counts.append(num_components)
                    
            except:
                continue
        
        connectedness_rate = connected_graphs / total_valid if total_valid > 0 else 0.0
        
        return {
            'connectedness_rate': connectedness_rate,
            'total_molecules': len(molecules),
            'valid_molecules': total_valid,
            'connected_graphs': connected_graphs,
            'component_distribution': component_counts,
            'avg_components': np.mean(component_counts) if component_counts else 0
        }
    
    def calculate_quality_metrics(self, molecules: List[str]) -> Dict:
        """Calculate quality metrics like diversity and novelty."""
        if not RDKIT_AVAILABLE:
            return {'diversity_rate': 0.0, 'error': 'RDKit not available'}
        
        # Remove duplicates
        unique_molecules = list(set(molecules))
        diversity_rate = len(unique_molecules) / len(molecules) if molecules else 0.0
        
        # Calculate molecular properties for unique molecules
        properties = []
        for smiles in unique_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    properties.append({
                        'smiles': smiles,
                        'molecular_weight': mw,
                        'logp': logp,
                        'hbd': hbd,
                        'hba': hba
                    })
            except:
                continue
        
        return {
            'diversity_rate': diversity_rate,
            'total_molecules': len(molecules),
            'unique_molecules': len(unique_molecules),
            'properties': properties
        }
    
    def run_validation(self, constraint_type: str, constraint_value: int) -> Dict:
        """Run complete validation pipeline."""
        print(f"Running validation for {self.experiment_name}")
        print(f"Constraint: {constraint_type} = {constraint_value}")
        
        # Load molecules
        molecules = self.load_generated_molecules()
        if not molecules:
            return {'error': 'No molecules found'}
        
        print(f"Loaded {len(molecules)} molecules for analysis")
        
        # Run constraint checks
        if constraint_type == 'ring_count':
            constraint_results = self.check_ring_count_constraint(molecules, constraint_value)
        elif constraint_type == 'ring_length':
            constraint_results = self.check_ring_length_constraint(molecules, constraint_value)
        else:
            constraint_results = {'error': f'Unknown constraint type: {constraint_type}'}
        
        # Run other validations
        validity_results = self.assess_molecular_validity(molecules)
        graph_results = self.check_graph_properties(molecules)
        quality_results = self.calculate_quality_metrics(molecules)
        
        # Compile results
        results = {
            'experiment_name': self.experiment_name,
            'constraint_type': constraint_type,
            'constraint_value': constraint_value,
            'constraint_satisfaction': constraint_results,
            'validity_metrics': validity_results,
            'graph_properties': graph_results,
            'quality_metrics': quality_results
        }
        
        # Save results
        output_file = self.output_dir / f"{self.experiment_name}_validation_results.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Validation results saved to {output_file}")
        return results
    
    def print_summary(self, results: Dict):
        """Print validation summary."""
        print("\n" + "="*60)
        print(f"VALIDATION SUMMARY: {self.experiment_name}")
        print("="*60)
        
        # Constraint satisfaction
        if 'constraint_satisfaction' in results:
            cs = results['constraint_satisfaction']
            if 'satisfaction_rate' in cs:
                print(f"✅ Constraint Satisfaction: {cs['satisfaction_rate']:.2%}")
                print(f"   - Satisfied: {cs.get('satisfied_constraint', 0)}")
                print(f"   - Total Valid: {cs.get('valid_molecules', 0)}")
        
        # Validity
        if 'validity_metrics' in results:
            vm = results['validity_metrics']
            print(f"🧪 Molecular Validity: {vm.get('validity_rate', 0):.2%}")
            print(f"   - Chemistry Rate: {vm.get('chemistry_rate', 0):.2%}")
        
        # Graph properties
        if 'graph_properties' in results:
            gp = results['graph_properties']
            print(f"🔗 Graph Connectedness: {gp.get('connectedness_rate', 0):.2%}")
        
        # Quality
        if 'quality_metrics' in results:
            qm = results['quality_metrics']
            print(f"📊 Diversity Rate: {qm.get('diversity_rate', 0):.2%}")
            print(f"   - Unique Molecules: {qm.get('unique_molecules', 0)}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate molecular constraints')
    parser.add_argument('--experiment_name', required=True, help='Experiment name')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--constraint_type', choices=['ring_count', 'ring_length'], 
                       required=True, help='Type of constraint')
    parser.add_argument('--constraint_value', type=int, required=True, 
                       help='Constraint value (e.g., min rings or min ring length)')
    
    args = parser.parse_args()
    
    validator = ConstraintValidator(args.experiment_name, args.output_dir)
    results = validator.run_validation(args.constraint_type, args.constraint_value)
    validator.print_summary(results)


if __name__ == "__main__":
    main() 