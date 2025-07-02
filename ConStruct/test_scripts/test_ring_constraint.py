#!/usr/bin/env python3
"""
Quick test script to verify the ring count constraint works correctly.
This runs locally without SLURM and uses minimal parameters for fast testing.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ConStruct.main import main
from omegaconf import OmegaConf
import hydra

def test_ring_constraint():
    """Test the ring count constraint with minimal parameters."""
    
    # Set up minimal config for quick testing
    config = {
        "general": {
            "name": "test_ring_constraint",
            "wandb": "disabled",
            "gpus": 0,  # Use CPU for quick testing
            "sample_every_val": 1,
            "samples_to_generate": 5,
            "samples_to_save": 2,
            "chains_to_save": 1,
            "number_chain_steps": 10,
            "final_model_samples_to_generate": 5,
            "final_model_samples_to_save": 2,
            "final_model_chains_to_save": 1,
        },
        "train": {
            "batch_size": 2,
            "n_epochs": 2,  # Very few epochs for quick test
            "save_model": False,
        },
        "model": {
            "diffusion_steps": 10,  # Very few steps for quick test
            "n_layers": 1,
            "hidden_mlp_dims": {"X": 8, "E": 4, "y": 4},
            "hidden_dims": {"dx": 8, "de": 4, "dy": 4, "n_head": 2, "dim_ffX": 8, "dim_ffE": 4, "dim_ffy": 8},
            "rev_proj": "ring_count",
            "max_rings": 0,
            "transition": "absorbing_edges",
        },
        "dataset": {
            "name": "qm9",
            "remove_h": True,
            "datadir": "data/qm9",
            "subset": 5,  # Use only 5 samples for quick test
            "batch_size": 2,
            "num_workers": 0,
        }
    }
    
    # Save config to temporary file
    config_path = "ConStruct/test_config.yaml"
    OmegaConf.save(config, config_path)
    
    print("Testing ring count constraint with minimal parameters...")
    print(f"Config saved to: {config_path}")
    print("This will test if the RingCountProjector works correctly.")
    print("Expected behavior: Generated molecules should have 0 rings (tree-like structures).")
    
    try:
        # Run the test
        main(config_path)
        print("\n✅ Test completed successfully!")
        print("Check the generated molecules to verify they have 0 rings.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("This indicates an issue with the ring count constraint implementation.")
        return False
    
    return True

if __name__ == "__main__":
    test_ring_constraint() 