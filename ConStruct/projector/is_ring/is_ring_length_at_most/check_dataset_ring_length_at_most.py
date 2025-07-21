try:
    import graph_tool as gt
except ModuleNotFoundError:
    print("Graph tool not found, non molecular datasets cannot be used")
import os
import pathlib

import hydra
import json
import networkx as nx
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch_geometric as tg
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from ConStruct.analysis.visualization import Visualizer
from ConStruct.utils import PlaceHolder
from ConStruct import utils
from ConStruct.projector.is_ring.is_ring_length_at_most.is_ring_length_at_most import get_max_ring_length_at_most


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset

    if dataset_config.name == "planar":
        from ConStruct.datasets.spectre_dataset import (
            PlanarDataModule,
            SpectreDatasetInfos,
        )

        datamodule = PlanarDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule)

    elif dataset_config.name == "qm9":
        from ConStruct.datasets.qm9_dataset import QM9DataModule, QM9Infos

        datamodule = QM9DataModule(cfg)
        dataset_infos = QM9Infos(datamodule=datamodule, cfg=cfg)
    elif dataset_config.name == "guacamol":
        from ConStruct.datasets.guacamol_dataset import (
            GuacamolDataModule,
            GuacamolInfos,
        )

        datamodule = GuacamolDataModule(cfg)
        dataset_infos = GuacamolInfos(datamodule=datamodule, cfg=cfg)
    elif dataset_config.name == "moses":
        from ConStruct.datasets.moses_dataset import MosesDataModule, MosesInfos

        datamodule = MosesDataModule(cfg)
        dataset_infos = MosesInfos(datamodule=datamodule, cfg=cfg)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg.dataset))

    datamodules = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    visualizer = Visualizer(dataset_infos=dataset_infos)

    # Get max_ring_length from config, default to 6 if not specified
    max_ring_length = getattr(cfg.model, "max_ring_length", 6)
    print(f"Checking for graphs with rings longer than {max_ring_length}")

    # Create main output directory for better organization
    main_output_dir = f"{dataset_config.name}_ring_length_violations"
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Output will be organized in: {main_output_dir}")

    graphs_with_too_large_rings = {}
    ring_length_distribution = {}
    print(f"Dataset loaded: {cfg.dataset.name}")
    
    for datamodule_str, datamodule in datamodules.items():
        print(f"Starting analysing {datamodule_str} datamodule.")
        dataset_idx = 0
        for batch_idx, batch in enumerate(tqdm(datamodule)):
            for graph in batch.to_data_list():
                nx_graph = to_networkx(graph, to_undirected=True)
                assert (
                    to_dense_adj(graph.edge_index).numpy()
                    == nx.to_numpy_array(nx_graph)
                ).all()
                
                # Get the length of the largest ring in the graph
                max_ring_len = get_max_ring_length_at_most(nx_graph)
                
                # Track ring length distribution
                if max_ring_len not in ring_length_distribution:
                    ring_length_distribution[max_ring_len] = 0
                ring_length_distribution[max_ring_len] += 1
                
                if max_ring_len > max_ring_length:
                    print(
                        f"Found graph with ring of length {max_ring_len} (max allowed: {max_ring_length}). Edge index: {graph.edge_index}. Num nodes: {graph.num_nodes}"
                    )
                    # Create organized directory structure
                    violation_dir = os.path.join(
                        main_output_dir,
                        f"{datamodule_str}_violations",
                        f"ring_length_{max_ring_len}_atoms",
                        f"graph_{dataset_idx}"
                    )
                    os.makedirs(violation_dir, exist_ok=True)
                    
                    saving_path = violation_dir

                    # put graph in required format for visualization
                    dense_graph = utils.to_dense(graph, dataset_infos)
                    graph_to_plot = dense_graph.collapse(dataset_infos.collapse_charges)
                    visualizer.visualize(
                        path=saving_path,
                        graphs=graph_to_plot,
                        atom_decoder=(
                            dataset_infos.atom_decoder
                            if hasattr(dataset_infos, "atom_decoder")
                            else None
                        ),
                        num_graphs_to_visualize=1,
                    )
                    # Storing smiles as well
                    if dataset_infos.is_molecular:
                        smiles = getattr(graph, 'smiles', None)
                        if smiles is None:
                            # Try to get SMILES from dataset_infos if available
                            smiles = "Unknown"
                        graphs_with_too_large_rings[f"{datamodule_str}_{dataset_idx}"] = {
                            "smiles": smiles,
                            "max_ring_length": max_ring_len,
                            "max_allowed": max_ring_length,
                            "output_path": violation_dir
                        }

                dataset_idx += 1

    # Write results to file in the main output directory
    results = {
        "graphs_with_too_large_rings": graphs_with_too_large_rings,
        "ring_length_distribution": ring_length_distribution,
        "max_ring_length_allowed": max_ring_length,
        "total_graphs_analyzed": sum(ring_length_distribution.values())
    }
    
    results_file = os.path.join(main_output_dir, f"ring_length_analysis_{dataset_config.name}.json")
    with open(results_file, "w") as outfile:
        json.dump(results, outfile, indent=2)

    print(f"Found {len(graphs_with_too_large_rings)} graphs with rings longer than {max_ring_length}")
    print(f"Ring length distribution: {dict(sorted(ring_length_distribution.items()))}")
    print(f"Results saved to: {main_output_dir}")


if __name__ == "__main__":
    main() 