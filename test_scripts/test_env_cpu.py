#!/usr/bin/env python3
"""
Local/CPU ConStruct Environment Quick Sanity Check
Checks RDKit, torch (CPU), torch-geometric, fcd, and your local package.
Run after following the Bulletproof Setup Instructions.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def print_banner(msg):
    print(f"\n{'=' * 10} {msg} {'=' * 10}")

def check_rdkit():
    print_banner("RDKit Import/Functionality")
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CCO')
        assert mol is not None
        print("✓ RDKit import OK, SMILES parsing works")
        return True
    except Exception as e:
        print(f"✗ RDKit check failed: {e}")
        return False

def check_torch_cpu():
    print_banner("PyTorch (CPU)")
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print("⚠️  CUDA detected, but this test expects a CPU-only environment. Continuing anyway.")
        else:
            # CPU tensor operation
            x = torch.rand(128, 128)
            y = torch.mm(x, x)
            print("✓ CPU tensor ops work")
        return True
    except Exception as e:
        print(f"✗ torch check failed: {e}")
        return False

def check_torch_geometric():
    print_banner("torch-geometric")
    try:
        import torch_geometric
        print(f"✓ torch_geometric {torch_geometric.__version__}")
        return True
    except Exception as e:
        print(f"✗ torch-geometric check failed: {e}")
        return False

def check_fcd():
    print_banner("fcd import/load_ref_model")
    try:
        import fcd
        has_load = hasattr(fcd, 'load_ref_model')
        print(f"✓ fcd import (load_ref_model: {has_load})")
        m = fcd.load_ref_model()
        print(f"✓ fcd.load_ref_model() returns model: {m is not None}")
        return True
    except Exception as e:
        print(f"✗ fcd check failed: {e}")
        return False

def check_fcd_score():
    print_banner("fcd get_fcd Score")
    try:
        import fcd
        score = fcd.get_fcd(['CCO', 'CCC'], ['CCO', 'CCN'])
        print(f"✓ fcd.get_fcd returns: {score} (float)")
        return True
    except Exception as e:
        print(f"✗ fcd.get_fcd check failed: {e}")
        return False

def check_graph_tool():
    print_banner("graph-tool import")
    try:
        import graph_tool
        print("✓ graph-tool import OK")
        return True
    except Exception as e:
        print(f"✗ graph-tool check failed: {e}")
        return False

def check_construct_package():
    print_banner("Your ConStruct Package")
    try:
        import ConStruct
        print("✓ ConStruct package import OK")
        return True
    except Exception as e:
        print(f"✗ ConStruct package import failed: {e}")
        return False

def main():
    print("\nConStruct CPU Environment Sanity Check")
    print("=" * 60)
    print(f"Python version: {sys.version.split()[0]}\n")

    tests = [
        check_rdkit,
        check_torch_cpu,
        check_torch_geometric,
        check_fcd,
        check_fcd_score,
        #check_graph_tool,
        check_construct_package,
    ]
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉  ALL CHECKS PASSED: Your CPU environment is ready for ConStruct!")
        print("You are safe to run your experiments (CPU mode). Good luck!")
    else:
        print("❌  Some checks failed. Paste the errors here for troubleshooting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
