#!/usr/bin/env python3
"""
Simple Constraint Checker
========================

Quick validation of generated molecules for constraint satisfaction.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: pip install rdkit")
    RDKIT_AVAILABLE = False


def load_smiles_from_file(filepath: str) -> List[str]:
    """Load SMILES strings from file."""
    molecules = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    molecules.append(line)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return molecules


def check_ring_count(smiles: str, min_rings: int) -> bool:
    """Check if molecule has at least min_rings rings."""
    if not RDKIT_AVAILABLE:
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        rings = mol.GetRingInfo()
        num_rings = rings.NumRings()
        return num_rings >= min_rings
    except:
        return False


def check_ring_length(smiles: str, min_length: int) -> bool:
    """Check if molecule has rings with at least min_length atoms."""
    if not RDKIT_AVAILABLE:
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        rings = mol.GetRingInfo()
        for ring in rings.AtomRings():
            if len(ring) >= min_length:
                return True
        return False
    except:
        return False


def validate_molecules(molecules: List[str], constraint_type: str, constraint_value: int) -> Dict:
    """Validate molecules against constraint."""
    if not molecules:
        return {'error': 'No molecules provided'}
    
    valid_molecules = 0
    constraint_satisfied = 0
    valid_smiles = 0
    
    for smiles in molecules:
        # Check SMILES validity
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles += 1
                valid_molecules += 1
                
                # Check constraint
                if constraint_type == 'ring_count':
                    if check_ring_count(smiles, constraint_value):
                        constraint_satisfied += 1
                elif constraint_type == 'ring_length':
                    if check_ring_length(smiles, constraint_value):
                        constraint_satisfied += 1
        except:
            continue
    
    total = len(molecules)
    validity_rate = valid_smiles / total if total > 0 else 0.0
    constraint_rate = constraint_satisfied / valid_molecules if valid_molecules > 0 else 0.0
    
    return {
        'total_molecules': total,
        'valid_smiles': valid_smiles,
        'valid_molecules': valid_molecules,
        'constraint_satisfied': constraint_satisfied,
        'validity_rate': validity_rate,
        'constraint_satisfaction_rate': constraint_rate,
        'constraint_type': constraint_type,
        'constraint_value': constraint_value
    }


def main():
    """Main validation function."""
    if len(sys.argv) < 4:
        print("Usage: python simple_constraint_check.py <smiles_file> <constraint_type> <constraint_value>")
        print("Example: python simple_constraint_check.py molecules.txt ring_count 2")
        sys.exit(1)
    
    smiles_file = sys.argv[1]
    constraint_type = sys.argv[2]
    constraint_value = int(sys.argv[3])
    
    # Load molecules
    molecules = load_smiles_from_file(smiles_file)
    print(f"Loaded {len(molecules)} molecules from {smiles_file}")
    
    # Validate
    results = validate_molecules(molecules, constraint_type, constraint_value)
    
    # Print results
    print("\n" + "="*50)
    print("CONSTRAINT VALIDATION RESULTS")
    print("="*50)
    print(f"Total molecules: {results['total_molecules']}")
    print(f"Valid SMILES: {results['valid_smiles']} ({results['validity_rate']:.2%})")
    print(f"Constraint satisfied: {results['constraint_satisfied']} ({results['constraint_satisfaction_rate']:.2%})")
    print(f"Constraint: {constraint_type} >= {constraint_value}")
    print("="*50)
    
    # Save results
    output_file = f"validation_results_{constraint_type}_{constraint_value}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main() 