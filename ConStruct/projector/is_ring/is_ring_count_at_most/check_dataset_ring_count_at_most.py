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
from ConStruct.projector.is_ring.is_ring_count_at_most.is_ring_count_at_most import count_rings_at_most


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

    # Get max_rings from config, default to 5 if not specified
    max_rings = getattr(cfg.model, "max_rings", 5)
    print(f"Checking for graphs with more than {max_rings} rings")

    # Create main output directory for better organization
    main_output_dir = f"{dataset_config.name}_ring_count_violations"
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Output will be organized in: {main_output_dir}")

    graphs_with_too_many_rings = {}
    ring_count_distribution = {}
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
                
                # Count rings in the graph
                num_rings = count_rings_at_most(nx_graph)
                
                # Track ring count distribution
                if num_rings not in ring_count_distribution:
                    ring_count_distribution[num_rings] = 0
                ring_count_distribution[num_rings] += 1
                
                if num_rings > max_rings:
                    print(
                        f"Found graph with {num_rings} rings (max allowed: {max_rings}). Edge index: {graph.edge_index}. Num nodes: {graph.num_nodes}"
                    )
                    # Create organized directory structure
                    violation_dir = os.path.join(
                        main_output_dir,
                        f"{datamodule_str}_violations",
                        f"ring_count_{num_rings}_rings",
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
                        graphs_with_too_many_rings[f"{datamodule_str}_{dataset_idx}"] = {
                            "smiles": smiles,
                            "num_rings": num_rings,
                            "max_allowed": max_rings,
                            "output_path": violation_dir
                        }

                dataset_idx += 1

    # Write results to file in the main output directory
    results = {
        "graphs_with_too_many_rings": graphs_with_too_many_rings,
        "ring_count_distribution": ring_count_distribution,
        "max_rings_allowed": max_rings,
        "total_graphs_analyzed": sum(ring_count_distribution.values())
    }
    
    results_file = os.path.join(main_output_dir, f"ring_count_analysis_{dataset_config.name}.json")
    with open(results_file, "w") as outfile:
        json.dump(results, outfile, indent=2)

    print(f"Found {len(graphs_with_too_many_rings)} graphs with more than {max_rings} rings")
    print(f"Ring count distribution: {dict(sorted(ring_count_distribution.items()))}")
    print(f"Results saved to: {main_output_dir}")


if __name__ == "__main__":
    main() 