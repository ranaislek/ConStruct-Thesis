import os
from collections import Counter
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torchmetrics import MeanMetric, MeanAbsoluteError, Metric, CatMetric
import torch
import wandb
import torch.nn as nn

from ConStruct.utils import PlaceHolder
import fcd
import numpy as np
import signal
import time
from tabulate import tabulate
from rdkit.Chem import AllChem
import networkx as nx

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

RDLogger.DisableLog("rdApp.*")


class Molecule:
    def __init__(self, graph: PlaceHolder, atom_decoder):
        """atom_decoder: extracted from dataset_infos."""
        self.atom_types = graph.X.long()
        self.bond_types = graph.E.long()
        self.charges = graph.charges.long()
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.atom_decoder = atom_decoder
        self.num_nodes = len(graph.X)
        self.num_atom_types = len(atom_decoder)
        self.device = self.atom_types.device

    def build_molecule(self, atom_decoder):
        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None
        return mol

    def check_stability(self, debug=False):
        e = self.bond_types.clone()
        e[e == 4] = 1.5
        e[e < 0] = 0
        valencies = torch.sum(e, dim=-1).long()

        n_stable_at = 0
        mol_stable = True
        for i, (atom_type, valency, charge) in enumerate(
            zip(self.atom_types, valencies, self.charges)
        ):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds = allowed_bonds[self.atom_decoder[atom_type]]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == valency
            elif type(possible_bonds) == dict:
                expected_bonds = (
                    possible_bonds[charge]
                    if charge in possible_bonds.keys()
                    else possible_bonds[0]
                )
                is_stable = (
                    expected_bonds == valency
                    if type(expected_bonds) == int
                    else valency in expected_bonds
                )
            else:
                is_stable = valency in possible_bonds
            if not is_stable:
                mol_stable = False
            if not is_stable and debug:
                print(
                    f"Invalid atom {self.atom_decoder[atom_type]}: valency={valency}, charge={charge}"
                )
                print()
            n_stable_at += int(is_stable)

        return mol_stable, n_stable_at, len(self.atom_types)


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, test, cfg=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.is_molecular = dataset_infos.is_molecular
        self.atom_decoder = dataset_infos.atom_decoder
        self.cfg = cfg

        self.test = test
        self.cfg = cfg
        self.is_molecular = dataset_infos.is_molecular
        self.remove_h = dataset_infos.remove_h
        self.train_smiles = set(dataset_infos.train_smiles)
        self.test_smiles = (
            dataset_infos.test_smiles if test else dataset_infos.val_smiles
        )
        self.test_fcd_stats = (
            dataset_infos.test_fcd_stats if test else dataset_infos.val_fcd_stats
        )
        
        # Timing tracking
        self.total_sampling_time = 0.0
        self.avg_sampling_time = 0.0
        self.num_batches = 0

        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(dataset_infos.train_smiles)
        self.validity_metric = MeanMetric()
        # self.uniqueness_metric = UniquenessMetric()
        self.novelty_metric = MeanMetric(nan_strategy="ignore")

        self.charge_w1 = MeanMetric()
        self.valency_w1 = MeanMetric()

        # for fcd
        self.val_smiles = list(
            dataset_infos.test_smiles if test else dataset_infos.val_smiles
        )
        self.val_fcd_mu, self.val_fcd_sigma = (
            dataset_infos.test_fcd_stats if test else dataset_infos.val_fcd_stats
        )

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,  # self.uniqueness_metric,
            self.novelty_metric,
            self.atom_stable,
            self.mol_stable,
            self.charge_w1,
            self.valency_w1,
        ]:
            metric.reset()

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        all_smiles = []
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=True
                    )
                    
                    # below code handles disconnected molecules by selecting the largest valid molecular fragment.
                    if len(mol_frags) > 1:
                        error_message[4] += 1  # count the number of disconnected molecules
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol, canonical=True)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    all_smiles.append("error")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    all_smiles.append("error")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
                    all_smiles.append("error")
        print(
            f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
            f" -- No error {error_message[-1]}"
        )
        self.validity_metric.update(
            value=len(valid) / len(generated), weight=len(generated)
        )
        return valid, all_smiles, error_message

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""

        valid_mols, all_smiles, error_message = self.compute_validity(generated)
        validity = self.validity_metric.compute().item()

        # self.uniqueness_metric.update(valid_mols)
        if len(valid_mols) > 0:
            uniqueness = len(set(valid_mols)) / len(valid_mols)
        else:
            uniqueness = 0.0
        unique = list(set(valid_mols))

        if self.train_smiles is not None and len(unique) > 0:
            novel = [s for s in unique if s not in self.train_smiles]
            self.novelty_metric.update(
                value=len(novel) / len(unique), weight=len(unique)
            )
            novelty = self.novelty_metric.compute().item()
        else:
            print("No valid molecules")
            novelty = 0.0
        # num_molecules = int(self.validity_metric.weight.item())
        # print(f"Validity over {num_molecules} molecules: {validity * 100 :.2f}%")

        key = "val_sampling" if not self.test else "test_sampling"
        dic = {
            f"{key}/Validity": validity * 100,
            f"{key}/Uniqueness": uniqueness * 100 if uniqueness != 0 else 0,
            f"{key}/Novelty": novelty * 100 if novelty != 0 else 0,
        }

        return all_smiles, dic

    def forward(self, generated_graphs: list[PlaceHolder], current_epoch, local_rank, computed_metrics=None):
        molecules = []

        for batch in generated_graphs:
            graphs = batch.split()
            for graph in graphs:
                molecule = Molecule(graph, atom_decoder=self.dataset_infos.atom_decoder)
                molecules.append(molecule)

        if not self.dataset_infos.remove_h:
            print(f"Analyzing molecule stability on {local_rank}...")
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = mol.check_stability()
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

            stability_dict = {
                "mol_stable": self.mol_stable.compute().item(),
                "atm_stable": self.atom_stable.compute().item(),
            }
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
                if wandb.run:
                    wandb.log(stability_dict, commit=False)

        # Validity, uniqueness, novelty
        all_generated_smiles, metrics = self.evaluate(molecules)
        
        # No chemical validation metrics in initial commit - removed to match exactly

        # Calculate FCD BEFORE table generation
        to_log_fcd = self.compute_fcd(generated_smiles=all_generated_smiles)
        metrics.update(to_log_fcd)

        if len(all_generated_smiles) > 0 and local_rank == 0:
            print("Some generated smiles: " + " ".join(all_generated_smiles[:10]))
            # --- Constraint check integration ---
            constraint_type = getattr(self.dataset_infos, "constraint_type", None)
            constraint_value = getattr(self.dataset_infos, "constraint_value", None)
            # Fallback: try to infer from self.cfg if available
            if constraint_type is None and hasattr(self, "cfg") and self.cfg is not None:
                constraint_type = getattr(self.cfg.model, "rev_proj", None)
                # Map old names to explicit ones
                if constraint_type == "ring_length_at_most":
                    constraint_value = getattr(self.cfg.model, "max_ring_length", None)
                elif constraint_type == "ring_length_at_least":
                    constraint_value = getattr(self.cfg.model, "min_ring_length", None)
                elif constraint_type == "ring_count_at_most":
                    constraint_value = getattr(self.cfg.model, "max_rings", None)
                elif constraint_type == "ring_count_at_least":
                    constraint_value = getattr(self.cfg.model, "min_rings", None)
            # List of constraint types that are enforced
            enforced_types = ["ring_count_at_most", "ring_count_at_least", "ring_length_at_most", "ring_length_at_least", "planar", "tree", "lobster"]
            boolean_constraints = ["planar", "tree", "lobster"]
            if constraint_type in enforced_types and (constraint_value is not None or constraint_type in boolean_constraints):
                # Constraint is enforced, only print enforcement check
                check_ring_constraints(all_generated_smiles, constraint_type, constraint_value, logger=lambda x: None)
                
                # Generate and print table report
                print("\n" + "="*80)
                print("🎯 TABLE-BASED CONSTRAINT ANALYSIS")
                print("="*80)
                # Use computed_metrics if available, otherwise fall back to local metrics
                if computed_metrics is not None:
                    # Merge so locally computed keys (like FCD) override or complement global ones (like Disconnected)
                    metrics_to_pass = {**computed_metrics, **metrics}
                else:
                    metrics_to_pass = metrics
                table_report = self.generate_table_report(constraint_type, constraint_value, all_generated_smiles, metrics_to_pass)
                print(table_report)
                print("="*80)
            else:
                # No constraint enforced, show comprehensive analysis for all constraint types
                
                # Generate and print comprehensive table report
                print("\n" + "="*80)
                print("🎯 COMPREHENSIVE CONSTRAINT ANALYSIS (NO ENFORCED CONSTRAINT)")
                print("="*80)
                # Use computed_metrics if available, otherwise fall back to local metrics
                if computed_metrics is not None:
                    metrics_to_pass = {**computed_metrics, **metrics}
                else:
                    metrics_to_pass = metrics
                comprehensive_report = self.generate_comprehensive_report(all_generated_smiles, metrics_to_pass)
                print(comprehensive_report)
                print("="*80)

        # Save in any case in the graphs folder
        os.makedirs("graphs", exist_ok=True)
        textfile = open(
            f"graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt", "w"
        )
        textfile.writelines(all_generated_smiles)
        textfile.close()
        # Save in the root folder if test_model
        if self.test:
            filename = f"final_smiles_GR{local_rank}_{0}.txt"
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f"final_smiles_GR{local_rank}_{i}.txt"
                else:
                    break
            with open(filename, "w") as fp:
                for smiles in all_generated_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print(f"All smiles saved on rank {local_rank}")
        # Compute statistics
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )

        key = "val_sampling" if not self.test else "test_sampling"
        if self.is_molecular:
            # the molecule validity is calculated with charge information
            charge_w1, charge_w1_per_class = charge_distance(
                molecules, stat.charge_types, stat.atom_types, self.dataset_infos
            )
            self.charge_w1(charge_w1)
            metrics[f"{key}/ChargeW1"] = self.charge_w1.compute().item()
            valency_w1, valency_w1_per_class = valency_distance(
                molecules,
                stat.valencies,
                stat.atom_types,
                self.dataset_infos.atom_encoder,
            )
            self.valency_w1(valency_w1)
            # TODO: (line below) torch lightning stalls for multi-gpu sampling is number of samples to generate is <=2
            metrics[f"{key}/ValencyW1"] = self.valency_w1.compute().item()

        # if local_rank == 0:
        #     print(f"Sampling metrics", {k: round(val, 3) for k, val in metrics.items()})
        # if local_rank == 0:
        #     print(f"Molecular metrics computed.")

        return metrics

    def compute_fcd(self, generated_smiles):
        fcd_model = fcd.load_ref_model()
        generated_smiles = [
            smile
            for smile in fcd.canonical_smiles(generated_smiles)
            if smile is not None
        ]

        if len(generated_smiles) <= 1:
            print("Not enough (<=1) valid smiles for FCD computation.")
            fcd_score = -1
        else:
            try:
                gen_activations = fcd.get_predictions(fcd_model, generated_smiles)
                gen_mu = np.mean(gen_activations, axis=0)
                gen_sigma = np.cov(gen_activations.T)
                target_mu = self.val_fcd_mu
                target_sigma = self.val_fcd_sigma
                try:
                    fcd_score = fcd.calculate_frechet_distance(
                        mu1=gen_mu,
                        sigma1=gen_sigma,
                        mu2=target_mu,
                        sigma2=target_sigma,
                    )
                except ValueError as e:
                    eps = 1e-6
                    print(f"Error in FCD computation: {e}. Increasing eps to {eps}")
                    eps_sigma = eps * np.eye(gen_sigma.shape[0])
                    gen_sigma = gen_sigma + eps_sigma
                    target_sigma = self.val_fcd_sigma + eps_sigma
                    fcd_score = fcd.calculate_frechet_distance(
                        mu1=gen_mu,
                        sigma1=gen_sigma,
                        mu2=target_mu,
                        sigma2=target_sigma,
                    )
            except Exception as e:
                print(f"FCD calculation failed: {e}")
                fcd_score = -1

        key = "val_sampling" if not self.test else "test_sampling"
        return {f"{key}/fcd score": fcd_score}

    # No chemical validation metrics method in initial commit - removed to match exactly

    def generate_table_report(self, constraint_type: str, constraint_value: int, all_generated_smiles: List[str], computed_metrics: Dict = None) -> str:
        """
        Generate a comprehensive table report for constraint satisfaction metrics.
        Includes both ALL molecules (structural analysis) and VALID molecules (chemical analysis).
        
        Args:
            constraint_type: Type of constraint (e.g., 'ring_count_at_most', 'planar')
            constraint_value: Value of the constraint (e.g., 1)
            all_generated_smiles: List of all generated SMILES strings
            
        Returns:
            Formatted table string with both structural and chemical analysis
        """
        
        # Initialize table variables to prevent UnboundLocalError
        structural_ring_length_table = ""
        chemical_ring_length_table = ""
        
        # Extract valid molecules first (same as in evaluate method)
        valid_smiles = [s for s in all_generated_smiles if s and s != "error"]
        total_molecules = len(all_generated_smiles)
        valid_molecules = len(valid_smiles)
        
        # Calculate validity rate (based on total generated)
        validity_rate = (valid_molecules / total_molecules * 100) if total_molecules > 0 else 0.0
        
        # ============================================================================
        # STRUCTURAL ANALYSIS (ALL MOLECULES) - ConStruct Philosophy
        # ============================================================================
        
        # Get structural constraint satisfaction results (using ALL molecules)
        if constraint_type == "planar":
            # For planar constraints, use the existing function for ALL molecules
            structural_results = check_planarity_constraint_all_molecules(all_generated_smiles, logger=lambda x: None)
            structural_satisfaction_rate = structural_results['satisfaction_rate'] * 100
        else:
            # For ring-based constraints, use the ALL molecules function
            structural_results = check_ring_constraints_all_molecules(all_generated_smiles, constraint_type, constraint_value, logger=lambda x: None)
            structural_satisfaction_rate = structural_results['satisfaction_rate'] * 100
        
        # ============================================================================
        # CHEMICAL ANALYSIS (VALID MOLECULES ONLY) - Traditional Analysis
        # ============================================================================
        
        # Get chemical constraint satisfaction results (using valid molecules only)
        if constraint_type == "planar":
            # For planar constraints, use the existing function for valid molecules
            chemical_results = check_planarity_constraint(valid_smiles, logger=lambda x: None)
            chemical_satisfaction_rate = chemical_results['satisfaction_rate'] * 100
        else:
            # For ring-based constraints, use the existing function for valid molecules
            chemical_results = check_ring_constraints(valid_smiles, constraint_type, constraint_value, logger=lambda x: None)
            chemical_satisfaction_rate = chemical_results['satisfaction_rate'] * 100
        
        # ============================================================================
        # QUALITY METRICS (VALID MOLECULES ONLY)
        # ============================================================================
        
        # Use computed metrics from the original wandb-submitting calculations
        if computed_metrics is not None:
            # Extract metrics from the computed_metrics dictionary
            key = "test_sampling" if self.test else "val_sampling"
            uniqueness_rate = computed_metrics.get(f"{key}/Uniqueness", 0.0)
            novelty_rate = computed_metrics.get(f"{key}/Novelty", 0.0)
            fcd_score = computed_metrics.get(f"{key}/fcd score", 0.0)
            
            # Debug: Print the computed_metrics to see what's available
            print(f"DEBUG: Available computed_metrics keys: {list(computed_metrics.keys())}")
            print(f"DEBUG: Looking for FCD key: {key}/fcd score")
            print(f"DEBUG: FCD score found: {fcd_score}")
            
            # Calculate unique and novel molecules for display (based on valid molecules only)
            if len(valid_smiles) > 0:
                unique_smiles = set(valid_smiles)
                unique_molecules = len(unique_smiles)
                uniqueness_rate = (unique_molecules / len(valid_smiles) * 100)  # Based on valid molecules
            else:
                uniqueness_rate = 0.0
                unique_molecules = 0
            
            # Calculate novelty (based on valid molecules only, same as evaluate method)
            if self.train_smiles is not None and len(unique_smiles) > 0:
                novel_smiles = [s for s in unique_smiles if s not in self.train_smiles]
                novel_molecules = len(novel_smiles)
                novelty_rate = (len(novel_smiles) / len(unique_smiles) * 100)  # Based on unique valid molecules
            else:
                novelty_rate = 0.0
                novel_molecules = 0
        else:
            # Fallback to original calculation if computed_metrics not provided
            if len(valid_smiles) > 0:
                unique_smiles = set(valid_smiles)
                unique_molecules = len(unique_smiles)
                uniqueness_rate = (unique_molecules / len(valid_smiles) * 100)  # Based on valid molecules
            else:
                uniqueness_rate = 0.0
                unique_molecules = 0
            
            # Calculate novelty (based on valid molecules only, same as evaluate method)
            if self.train_smiles is not None and len(unique_smiles) > 0:
                novel_smiles = [s for s in unique_smiles if s not in self.train_smiles]
                novel_molecules = len(novel_smiles)
                novelty_rate = (len(novel_smiles) / len(unique_smiles) * 100)  # Based on unique valid molecules
            else:
                novelty_rate = 0.0
                novel_molecules = 0
            
            # Fallback FCD calculation
            fcd_score = 0.0
        
        # ============================================================================
        # CREATE COMPREHENSIVE TABLES
        # ============================================================================
        
        # Get disconnected molecules rate from the original Disconnected metric (same as initial commit)
        key = "test_sampling" if self.test else "val_sampling"
        if computed_metrics is not None:
            # Use the original Disconnected metric from sampling_metrics.py
            disconnected_rate = computed_metrics.get(f"{key}/Disconnected", 0.0)
        else:
            # Fallback to 0.0 if computed_metrics not available
            disconnected_rate = 0.0
        
        # Create main metrics table with both structural and chemical analysis
        constraint_label = "planar" if constraint_type == "planar" else f"{constraint_type} ≤ {constraint_value}"
        
        metrics_data = [
            ["Total Molecules Generated", f"{total_molecules:,}"],
            ["Valid Molecules (RDKit)", f"{valid_molecules:,} ({validity_rate:.2f}%)"],
            ["Invalid Molecules", f"{total_molecules - valid_molecules:,} ({(total_molecules - valid_molecules) / total_molecules * 100:.2f}%)"],
            ["Disconnected Molecules", f"{disconnected_rate:.2f}%"],
            ["Constraint Type", constraint_label],
        ]
        
        metrics_table = tabulate(metrics_data, headers=["Metric", "Value"], 
                               tablefmt="grid", numalign="left")
        
        # Create structural constraint verification table (ALL molecules)
        structural_verification_data = []
        if constraint_type == "ring_count_at_most":
            # Check ring_count_at_most for values 0-6 using ALL molecules
            for test_value in range(7):  # 0 to 6
                test_results = check_ring_constraints_all_molecules(all_generated_smiles, "ring_count_at_most", test_value, logger=lambda x: None)
                satisfied = test_results['satisfied_molecules']
                total = test_results['total_molecules']
                rate = test_results['satisfaction_rate'] * 100
                
                # Highlight the enforced constraint
                if test_value == constraint_value:
                    structural_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", "⭐ ENFORCED"])
                else:
                    structural_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", ""])
                    
        elif constraint_type == "ring_length_at_most":
            # Check ring_length_at_most for values 3-8 using ALL molecules
            for test_value in range(3, 9):  # 3 to 8
                test_results = check_ring_constraints_all_molecules(all_generated_smiles, "ring_length_at_most", test_value, logger=lambda x: None)
                satisfied = test_results['satisfied_molecules']
                total = test_results['total_molecules']
                rate = test_results['satisfaction_rate'] * 100
                
                # Highlight the enforced constraint
                if test_value == constraint_value:
                    structural_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", "⭐ ENFORCED"])
                else:
                    structural_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", ""])
        
        elif constraint_type == "planar":
            # For planar, just show the single result
            structural_verification_data.append([f"planar", f"{structural_results['satisfied_molecules']}/{structural_results['total_molecules']}", f"{structural_satisfaction_rate:.1f}%", "⭐ ENFORCED"])
        
        structural_verification_table = ""
        if structural_verification_data:
            structural_verification_table = "\n\n🏗️ STRUCTURAL CONSTRAINT ENFORCEMENT (ALL Molecules - ConStruct Philosophy):\n"
            structural_verification_table += tabulate(structural_verification_data, 
                                                    headers=["Constraint", "Satisfied", "Rate", "Status"], 
                                                    tablefmt="grid", numalign="left")
        
        # Create chemical constraint verification table (VALID molecules only)
        chemical_verification_data = []
        if constraint_type == "ring_count_at_most":
            # Check ring_count_at_most for values 0-6 using valid molecules only
            for test_value in range(7):  # 0 to 6
                test_results = check_ring_constraints(valid_smiles, "ring_count_at_most", test_value, logger=lambda x: None)
                satisfied = test_results['satisfied_molecules']
                total = test_results['total_molecules']
                rate = test_results['satisfaction_rate'] * 100
                
                # Highlight the enforced constraint
                if test_value == constraint_value:
                    chemical_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", "⭐ ENFORCED"])
                else:
                    chemical_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", ""])
                    
        elif constraint_type == "ring_length_at_most":
            # Check ring_length_at_most for values 3-8 using valid molecules only
            for test_value in range(3, 9):  # 3 to 8
                test_results = check_ring_constraints(valid_smiles, "ring_length_at_most", test_value, logger=lambda x: None)
                satisfied = test_results['satisfied_molecules']
                total = test_results['total_molecules']
                rate = test_results['satisfaction_rate'] * 100
                
                # Highlight the enforced constraint
                if test_value == constraint_value:
                    chemical_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", "⭐ ENFORCED"])
                else:
                    chemical_verification_data.append([f"≤{test_value}", f"{satisfied}/{total}", f"{rate:.1f}%", ""])
        
        elif constraint_type == "planar":
            # For planar, just show the single result
            chemical_verification_data.append([f"planar", f"{chemical_results['satisfied_molecules']}/{chemical_results['total_molecules']}", f"{chemical_satisfaction_rate:.1f}%", "⭐ ENFORCED"])
        
        chemical_verification_table = ""
        if chemical_verification_data:
            chemical_verification_table = "\n\n🧪 CHEMICAL CONSTRAINT ENFORCEMENT (Valid Molecules Only):\n"
            chemical_verification_table += tabulate(chemical_verification_data, 
                                                  headers=["Constraint", "Satisfied", "Rate", "Status"], 
                                                  tablefmt="grid", numalign="left")
        
        # Create ring count distribution tables (both ALL and VALID molecules)
        structural_distribution_table = ""
        chemical_distribution_table = ""
        if constraint_type in ["ring_count_at_most", "ring_length_at_most"]:
            # Structural distribution (ALL molecules)
            structural_results = check_ring_constraints_all_molecules(all_generated_smiles, constraint_type, constraint_value, logger=lambda x: None)
            structural_ring_counts = structural_results.get('ring_counts', [])
            if structural_ring_counts:
                from collections import Counter
                structural_ring_count_dist = Counter(structural_ring_counts)
                
                structural_ring_data = []
                for ring_count in sorted(structural_ring_count_dist.keys()):
                    count = structural_ring_count_dist[ring_count]
                    percentage = (count / total_molecules * 100) if total_molecules > 0 else 0.0
                    structural_ring_data.append([ring_count, count, f"{percentage:.1f}%"])
                
                structural_distribution_table = "\n\n📊 STRUCTURAL RING COUNT DISTRIBUTION (ALL Molecules):\n"
                structural_distribution_table += tabulate(structural_ring_data, 
                                                       headers=["Rings", "Count", "Percentage"], 
                                                       tablefmt="grid", numalign="left")
            
            # Chemical distribution (VALID molecules only)
            chemical_ring_counts = chemical_results.get('ring_counts', [])
            if chemical_ring_counts:
                from collections import Counter
                chemical_ring_count_dist = Counter(chemical_ring_counts)
                
                chemical_ring_data = []
                for ring_count in sorted(chemical_ring_count_dist.keys()):
                    count = chemical_ring_count_dist[ring_count]
                    percentage = (count / valid_molecules * 100) if valid_molecules > 0 else 0.0
                    chemical_ring_data.append([ring_count, count, f"{percentage:.1f}%"])
                
                chemical_distribution_table = "\n\n📊 CHEMICAL RING COUNT DISTRIBUTION (Valid Molecules Only):\n"
                chemical_distribution_table += tabulate(chemical_ring_data, 
                                                      headers=["Rings", "Count", "Percentage"], 
                                                      tablefmt="grid", numalign="left")
            
            # Add ring length distribution tables for ring_length_at_most constraint
            if constraint_type == "ring_length_at_most":
                # Structural ring length distribution (ALL molecules)
                structural_ring_lengths = structural_results.get('ring_lengths', [])
                if structural_ring_lengths:
                    from collections import Counter
                    structural_ring_length_dist = Counter(structural_ring_lengths)
                    
                    structural_ring_length_data = []
                    for ring_length in sorted(structural_ring_length_dist.keys()):
                        count = structural_ring_length_dist[ring_length]
                        percentage = (count / total_molecules * 100) if total_molecules > 0 else 0.0
                        structural_ring_length_data.append([ring_length, count, f"{percentage:.1f}%"])
                    
                    structural_ring_length_table = "\n\n📊 STRUCTURAL RING LENGTH DISTRIBUTION (ALL Molecules):\n"
                    structural_ring_length_table += tabulate(structural_ring_length_data, 
                                                           headers=["Ring Length", "Count", "Percentage"], 
                                                           tablefmt="grid", numalign="left")
                else:
                    structural_ring_length_table = ""
                
                # Chemical ring length distribution (VALID molecules only)
                chemical_ring_lengths = chemical_results.get('ring_lengths', [])
                if chemical_ring_lengths:
                    from collections import Counter
                    chemical_ring_length_dist = Counter(chemical_ring_lengths)
                    
                    chemical_ring_length_data = []
                    for ring_length in sorted(chemical_ring_length_dist.keys()):
                        count = chemical_ring_length_dist[ring_length]
                        percentage = (count / valid_molecules * 100) if valid_molecules > 0 else 0.0
                        chemical_ring_length_data.append([ring_length, count, f"{percentage:.1f}%"])
                    
                    chemical_ring_length_table = "\n\n📊 CHEMICAL RING LENGTH DISTRIBUTION (Valid Molecules Only):\n"
                    chemical_ring_length_table += tabulate(chemical_ring_length_data, 
                                                          headers=["Ring Length", "Count", "Percentage"], 
                                                          tablefmt="grid", numalign="left")
                else:
                    chemical_ring_length_table = ""
            else:
                structural_ring_length_table = ""
                chemical_ring_length_table = ""
        
        # Create comprehensive quality metrics table (ALL molecules)
        # Note: Novelty and uniqueness are only calculated for valid molecules
        all_molecules_quality_data = [
            ["Total Molecules Generated", f"{total_molecules:,}"],
            ["Valid Molecules (RDKit)", f"{valid_molecules:,} ({validity_rate:.1f}%)"],
            ["Invalid Molecules", f"{total_molecules - valid_molecules:,} ({(total_molecules - valid_molecules) / total_molecules * 100:.1f}%)"],
            ["Disconnected Molecules", f"{disconnected_rate:.1f}%"],
            ["Average Molecular Weight", f"{self._calculate_avg_molecular_weight(all_generated_smiles):.1f}"],
            ["Average Number of Atoms", f"{self._calculate_avg_atom_count(all_generated_smiles):.1f}"],
            ["Average Number of Bonds", f"{self._calculate_avg_bond_count(all_generated_smiles):.1f}"],
            ["Average Ring Count", f"{self._calculate_avg_ring_count(all_generated_smiles):.1f}"],
        ]
        
        all_molecules_quality_table = "\n\n📊 MOLECULAR QUALITY METRICS (ALL Molecules):\n"
        all_molecules_quality_table += tabulate(all_molecules_quality_data, headers=["Metric", "Value"], 
                                              tablefmt="grid", numalign="left")
        
        # Create detailed quality metrics table (VALID molecules only)
        valid_molecules_quality_data = [
            ["Valid Molecules (RDKit)", f"{valid_molecules:,} ({validity_rate:.2f}%)"],
            ["Unique Molecules", f"{unique_molecules:,} ({uniqueness_rate:.2f}%)"],
            ["Novel Molecules", f"{novel_molecules:,} ({novelty_rate:.2f}%)"],
            ["FCD Score", f"{fcd_score:.3f}"],
            ["Average Molecular Weight", f"{self._calculate_avg_molecular_weight(valid_smiles):.1f}"],
            ["Average Number of Atoms", f"{self._calculate_avg_atom_count(valid_smiles):.1f}"],
            ["Average Number of Bonds", f"{self._calculate_avg_bond_count(valid_smiles):.1f}"],
            ["Average Ring Count", f"{self._calculate_avg_ring_count(valid_smiles):.1f}"],
            ["Average Ring Length", f"{self._calculate_avg_ring_length(valid_smiles):.1f}"],
            ["Average Valency", f"{self._calculate_avg_valency(valid_smiles):.1f}"],
        ]
        
        valid_molecules_quality_table = "\n\n🎯 MOLECULAR QUALITY METRICS (Valid Molecules Only):\n"
        valid_molecules_quality_table += tabulate(valid_molecules_quality_data, headers=["Metric", "Value"], 
                                                tablefmt="grid", numalign="left")
        
        # Create timing table with actual values
        timing_data = [
            ["Total Sampling Time", f"{getattr(self, 'total_sampling_time', 0):.2f} seconds"],
            ["Average Time per Batch", f"{getattr(self, 'avg_sampling_time', 0):.2f} seconds"],
            ["Number of Batches", f"{getattr(self, 'num_batches', 0)}"],
        ]
        
        timing_table = "\n\n⏱️ TIMING METRICS:\n"
        timing_table += tabulate(timing_data, headers=["Metric", "Value"], 
                               tablefmt="grid", numalign="left")
        
        # Create gap analysis table
        gap_data = [
            ["Structural Satisfaction (ALL)", f"{structural_satisfaction_rate:.1f}%"],
            ["Chemical Satisfaction (Valid)", f"{chemical_satisfaction_rate:.1f}%"],
            ["Gap (Structural - Chemical)", f"{structural_satisfaction_rate - chemical_satisfaction_rate:.1f}%"],
            ["Gap Explanation", "Shows difference between structural enforcement and chemical validity"],
        ]
        
        gap_table = "\n\n🔍 GAP ANALYSIS (ConStruct Philosophy):\n"
        gap_table += tabulate(gap_data, headers=["Metric", "Value"], 
                             tablefmt="grid", numalign="left")
        
        # No chemical validation table in initial commit - removed to match exactly
        
        # Combine all tables
        full_report = f"""🎯 COMPREHENSIVE CONSTRAINT SATISFACTION ANALYSIS
{'='*80}

{metrics_table}{structural_verification_table}{chemical_verification_table}{structural_distribution_table}{chemical_distribution_table}{structural_ring_length_table}{chemical_ring_length_table}{all_molecules_quality_table}{valid_molecules_quality_table}{gap_table}{timing_table}

{'='*80}
"""
        
        return full_report

    def generate_comprehensive_report(self, all_generated_smiles: List[str], computed_metrics: Dict = None) -> str:
        """
        Generates a comprehensive report for all constraint types, showing satisfaction rates.
        Uses computed metrics from the original wandb-submitting calculations to avoid duplication.
        
        Args:
            all_generated_smiles: List of all generated SMILES strings
            computed_metrics: Dictionary of already-computed metrics from the original calculations
            
        Returns:
            Formatted comprehensive report string
        """
        
        # Extract valid molecules first (same as in evaluate method)
        valid_smiles = [s for s in all_generated_smiles if s and s != "error"]
        total_molecules = len(all_generated_smiles)
        valid_molecules = len(valid_smiles)
        
        # Calculate validity rate (based on total generated) - will be overridden by wandb value if available
        validity_rate = (valid_molecules / total_molecules * 100) if total_molecules > 0 else 0.0
        
        # Use computed metrics from the original wandb-submitting calculations
        if computed_metrics is not None:
            # Extract metrics from the computed_metrics dictionary
            key = "test_sampling" if self.test else "val_sampling"
            uniqueness_rate = computed_metrics.get(f"{key}/Uniqueness", 0.0)
            novelty_rate = computed_metrics.get(f"{key}/Novelty", 0.0)
            fcd_score = computed_metrics.get(f"{key}/fcd score", 0.0)
            validity_rate = computed_metrics.get(f"{key}/Validity", 0.0)
            
            # Calculate molecule counts for display (based on valid molecules only)
            if len(valid_smiles) > 0:
                unique_smiles = set(valid_smiles)
                unique_molecules = len(unique_smiles)
                # Use exact uniqueness rate from wandb, calculate molecule count
                unique_molecules = int((uniqueness_rate / 100) * len(valid_smiles))
            else:
                unique_molecules = 0
            
            # Calculate novel molecules count (based on valid molecules only, same as evaluate method)
            if self.train_smiles is not None and len(unique_smiles) > 0:
                novel_smiles = [s for s in unique_smiles if s not in self.train_smiles]
                novel_molecules = len(novel_smiles)
                # Use exact novelty rate from wandb, calculate molecule count
                novel_molecules = int((novelty_rate / 100) * unique_molecules)
            else:
                novel_molecules = 0
        else:
            # Fallback to original calculation if computed_metrics not provided
            if len(valid_smiles) > 0:
                unique_smiles = set(valid_smiles)
                unique_molecules = len(unique_smiles)
                uniqueness_rate = (unique_molecules / len(valid_smiles) * 100)  # Based on valid molecules
            else:
                uniqueness_rate = 0.0
                unique_molecules = 0
            
            # Calculate novelty (based on valid molecules only, same as evaluate method)
            if self.train_smiles is not None and len(unique_smiles) > 0:
                novel_smiles = [s for s in unique_smiles if s not in self.train_smiles]
                novel_molecules = len(novel_smiles)
                novelty_rate = (len(novel_smiles) / len(unique_smiles) * 100)  # Based on unique valid molecules
            else:
                novelty_rate = 0.0
                novel_molecules = 0
            
            # Fallback FCD calculation
            fcd_score = 0.0
        
        # ============================================================================
        # CREATE COMPREHENSIVE REPORT
        # ============================================================================
        
        # Get disconnected molecules rate from the original Disconnected metric (same as initial commit)
        key = "test_sampling" if self.test else "val_sampling"
        if computed_metrics is not None:
            # Use the original Disconnected metric from sampling_metrics.py
            disconnected_rate = computed_metrics.get(f"{key}/Disconnected", 0.0)
        else:
            # Fallback to 0.0 if computed_metrics not available
            disconnected_rate = 0.0
        
        # Create comprehensive quality metrics table (ALL molecules)
        # Note: Novelty and uniqueness are only calculated for valid molecules
        all_molecules_quality_data = [
            ["Total Molecules Generated", f"{total_molecules:,}"],
            ["Valid Molecules (RDKit)", f"{valid_molecules:,} ({validity_rate:.2f}%)"],
            ["Invalid Molecules", f"{total_molecules - valid_molecules:,} ({(total_molecules - valid_molecules) / total_molecules * 100:.2f}%)"],
            ["Disconnected Molecules", f"{disconnected_rate:.2f}%"],
            ["Average Molecular Weight", f"{self._calculate_avg_molecular_weight(all_generated_smiles):.1f}"],
            ["Average Number of Atoms", f"{self._calculate_avg_atom_count(all_generated_smiles):.1f}"],
            ["Average Number of Bonds", f"{self._calculate_avg_bond_count(all_generated_smiles):.1f}"],
            ["Average Ring Count", f"{self._calculate_avg_ring_count(all_generated_smiles):.1f}"],
        ]
        
        # Generate detailed quality metrics table (VALID molecules only)
        valid_molecules_quality_data = [
            ["Valid Molecules (RDKit)", f"{valid_molecules:,} ({validity_rate:.2f}%)"],
            ["Disconnected Molecules", f"{disconnected_rate:.2f}%"],
            ["Unique Molecules", f"{unique_molecules:,} ({uniqueness_rate:.2f}%)"],
            ["Novel Molecules", f"{novel_molecules:,} ({novelty_rate:.2f}%)"],
            ["FCD Score", f"{fcd_score:.3f}"],
            ["Average Molecular Weight", f"{self._calculate_avg_molecular_weight(valid_smiles):.1f}"],
            ["Average Number of Atoms", f"{self._calculate_avg_atom_count(valid_smiles):.1f}"],
            ["Average Number of Bonds", f"{self._calculate_avg_bond_count(valid_smiles):.1f}"],
            ["Average Ring Count", f"{self._calculate_avg_ring_count(valid_smiles):.1f}"],
            ["Average Ring Length", f"{self._calculate_avg_ring_length(valid_smiles):.1f}"],
            ["Average Valency", f"{self._calculate_avg_valency(valid_smiles):.1f}"],
        ]
        
        # Generate timing metrics table
        timing_data = [
            ["Total Sampling Time", f"{getattr(self, 'total_sampling_time', 0):.2f} seconds"],
            ["Average Time per Batch", f"{getattr(self, 'avg_sampling_time', 0):.2f} seconds"],
            ["Number of Batches", f"{getattr(self, 'num_batches', 0)}"],
        ]
        
        # For no-constraint baseline, analyze ALL constraint types to show baseline performance
        # Structural constraint analysis (ALL molecules)
        structural_constraint_data = []
        constraint_types = [
            ("ring_count_at_most", 0), ("ring_count_at_most", 1), ("ring_count_at_most", 2),
            ("ring_count_at_most", 3), ("ring_count_at_most", 4), ("ring_count_at_most", 5),
            ("ring_length_at_most", 3), ("ring_length_at_most", 4), ("ring_length_at_most", 5),
            ("ring_length_at_most", 6), ("ring_length_at_most", 7), ("ring_length_at_most", 8),
            ("planar", None)
        ]
        
        for constraint_type, constraint_value in constraint_types:
            if constraint_type == "planar":
                # For planar constraints, analyze ALL molecules for structural planarity
                structural_results = check_planarity_constraint_all_molecules(all_generated_smiles, logger=lambda x: None)
                structural_constraint_data.append({
                    "Constraint Type": constraint_type,
                    "Constraint Value": "Planar",
                    "Total Molecules": structural_results['total_molecules'],
                    "Satisfied Molecules": structural_results['satisfied_molecules'],
                    "Satisfaction Rate": f"{structural_results['satisfaction_rate'] * 100:.1f}%"
                })
            else:
                # For ring-based constraints, use the ALL molecules function
                structural_results = check_ring_constraints_all_molecules(all_generated_smiles, constraint_type, constraint_value, logger=lambda x: None)
                structural_constraint_data.append({
                    "Constraint Type": constraint_type,
                    "Constraint Value": f"≤{constraint_value}",
                    "Total Molecules": structural_results['total_molecules'],
                    "Satisfied Molecules": structural_results['satisfied_molecules'],
                    "Satisfaction Rate": f"{structural_results['satisfaction_rate'] * 100:.1f}%"
                })
        
        # Chemical constraint analysis (VALID molecules only)
        chemical_constraint_data = []
        
        for constraint_type, constraint_value in constraint_types:
            if constraint_type == "planar":
                # For planar constraints, check planarity using the same method as constraint enforcement
                chemical_results = check_planarity_constraint(valid_smiles, logger=lambda x: None)
                chemical_constraint_data.append({
                    "Constraint Type": constraint_type,
                    "Constraint Value": "Planar",
                    "Total Molecules": chemical_results['total_molecules'],
                    "Satisfied Molecules": chemical_results['satisfied_molecules'],
                    "Satisfaction Rate": f"{chemical_results['satisfaction_rate'] * 100:.1f}%"
                })
            else:
                # For ring-based constraints, use the existing function for valid molecules
                chemical_results = check_ring_constraints(valid_smiles, constraint_type, constraint_value, logger=lambda x: None)
                chemical_constraint_data.append({
                    "Constraint Type": constraint_type,
                    "Constraint Value": f"≤{constraint_value}",
                    "Total Molecules": chemical_results['total_molecules'],
                    "Satisfied Molecules": chemical_results['satisfied_molecules'],
                    "Satisfaction Rate": f"{chemical_results['satisfaction_rate'] * 100:.1f}%"
                })
        
        # Create gap analysis table
        gap_data = []
        for i in range(len(structural_constraint_data)):
            structural_rate = float(structural_constraint_data[i]["Satisfaction Rate"].replace('%', ''))
            chemical_rate = float(chemical_constraint_data[i]["Satisfaction Rate"].replace('%', ''))
            gap = structural_rate - chemical_rate
            gap_data.append({
                "Constraint": f"{structural_constraint_data[i]['Constraint Type']} {structural_constraint_data[i]['Constraint Value']}",
                "Structural (ALL)": f"{structural_rate:.1f}%",
                "Chemical (Valid)": f"{chemical_rate:.1f}%",
                "Gap": f"{gap:.1f}%"
            })
        
        # Combine all tables
        full_report = ""
        
        # ALL Molecules quality metrics table
        full_report += "📊 MOLECULAR QUALITY METRICS (ALL Molecules)\n"
        full_report += tabulate(all_molecules_quality_data, headers=["Metric", "Value"], tablefmt="grid", numalign="left")
        full_report += "\n\n"
        
        # Valid Molecules quality metrics table
        full_report += "🎯 MOLECULAR QUALITY METRICS (Valid Molecules Only)\n"
        full_report += tabulate(valid_molecules_quality_data, headers=["Metric", "Value"], tablefmt="grid", numalign="left")
        full_report += "\n\n"
        
        # Timing table
        full_report += "⏱️ TIMING METRICS\n"
        full_report += tabulate(timing_data, headers=["Metric", "Value"], tablefmt="grid", numalign="left")
        full_report += "\n\n"
        
        # Structural constraint satisfaction table
        full_report += "🏗️ STRUCTURAL CONSTRAINT SATISFACTION (ALL Molecules - ConStruct Philosophy)\n"
        full_report += tabulate(structural_constraint_data, headers="keys", tablefmt="grid", numalign="left")
        full_report += "\n\n"
        
        # Chemical constraint satisfaction table
        full_report += "🧪 CHEMICAL CONSTRAINT SATISFACTION (Valid Molecules Only)\n"
        full_report += tabulate(chemical_constraint_data, headers="keys", tablefmt="grid", numalign="left")
        full_report += "\n\n"
        
        # No chemical validation metrics in initial commit - removed to match exactly
        
        # Gap analysis table
        full_report += "🔍 GAP ANALYSIS (Structural vs Chemical Satisfaction)\n"
        full_report += tabulate(gap_data, headers="keys", tablefmt="grid", numalign="left")
        
        return full_report

    def record_sampling_time(self, sampling_time: float):
        """Record sampling time for timing metrics."""
        self.total_sampling_time += sampling_time
        self.num_batches += 1
        self.avg_sampling_time = self.total_sampling_time / self.num_batches
    
    def _calculate_avg_molecular_weight(self, smiles_list: List[str]) -> float:
        """Calculate average molecular weight for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_weight = 0.0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Try to sanitize the molecule first
                        try:
                            Chem.SanitizeMol(mol)
                        except:
                            pass  # Continue even if sanitization fails
                        
                        # Try different methods to get molecular weight
                        weight = 0.0
                        try:
                            # Method 1: Use Descriptors.MolWt (proper import)
                            weight = Descriptors.MolWt(mol)
                        except Exception as e:
                            try:
                                # Method 2: Manual calculation as fallback
                                weight = 0.0
                                for atom in mol.GetAtoms():
                                    weight += atom.GetMass()
                            except Exception as e2:
                                continue
                        
                        if weight > 0:  # Only count if we got a valid weight
                            total_weight += weight
                            valid_count += 1
                except Exception as e:
                    continue
        
        return total_weight / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_atom_count(self, smiles_list: List[str]) -> float:
        """Calculate average number of atoms for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_atoms = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        total_atoms += mol.GetNumAtoms()
                        valid_count += 1
                except:
                    continue
        
        return total_atoms / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_bond_count(self, smiles_list: List[str]) -> float:
        """Calculate average number of bonds for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_bonds = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        total_bonds += mol.GetNumBonds()
                        valid_count += 1
                except:
                    continue
        
        return total_bonds / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_ring_count(self, smiles_list: List[str]) -> float:
        """Calculate average ring count for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_rings = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        ring_info = mol.GetRingInfo()
                        total_rings += ring_info.NumRings()
                        valid_count += 1
                except:
                    continue
        
        return total_rings / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_ring_length(self, smiles_list: List[str]) -> float:
        """Calculate average ring length for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_ring_length = 0
        total_rings = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        ring_info = mol.GetRingInfo()
                        rings = ring_info.AtomRings()
                        for ring in rings:
                            total_ring_length += len(ring)
                            total_rings += 1
                        valid_count += 1
                except:
                    continue
        
        return total_ring_length / total_rings if total_rings > 0 else 0.0
    
    def _calculate_avg_valency(self, smiles_list: List[str]) -> float:
        """Calculate average valency for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_valency = 0
        total_atoms = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        for atom in mol.GetAtoms():
                            total_valency += atom.GetTotalValence()
                            total_atoms += 1
                        valid_count += 1
                except:
                    continue
        
        return total_valency / total_atoms if total_atoms > 0 else 0.0
    
    # Note: Novelty and uniqueness calculations are only done on valid molecules in the evaluate() method
    # These helper methods were incorrectly calculating on all molecules and have been removed


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()
    return w1, w1_per_class


def valency_distance(
    molecules, target_valencies, atom_types_probabilities, atom_encoder
):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(
        max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values()
    )
    max_valency_generated = max(
        max(vals.keys()) if len(vals) > 0 else -1
        for vals in generated_valencies.values()
    )
    max_valency = int(max(max_valency_target, max_valency_generated))

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[int(atom_encoder[atom_type]), int(valency)] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[int(atom_type), int(valency)] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    # print('debugging for molecular_metrics - valency_distance')
    # print('cs_target', cs_target)
    # print('cs_generated', cs_generated)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


#
# class UniquenessMetric(Metric):
#     is_differentiable = False
#     higher_is_better = True
#     full_state_update = True
#     def __init__(self):
#         """ Check if the number of unique molecules by concatenating the smiles. """
#         super().__init__(compute_on_cpu=True)
#         # add the smiles as a state
#         self.add_state("smiles", default=[], dist_reduce_fx=None)
#
#     def update(self, smiles_list):
#         self.smiles.extend(smiles_list)
#
#     def compute(self):
#         print(f"Computing uniqueness over {len(self.smiles)} smiles")
#         if len(self.smiles) == 0:
#             return 0.0
#         return len(set(self.smiles)) / len(self.smiles)


def smiles_from_generated_samples_file(generated_samples_file, atom_decoder):
    """ """
    smiles_list = []
    with open(generated_samples_file, "r") as f:
        # Repeat until the end of the file
        while True:
            # Check if we reached the end of the file
            line = f.readline()
            print("First line", line)
            if not line:
                break
            else:
                N = int(line.split("=")[1])

            # Extract X (labels or coordinates of the nodes)
            f.readline()
            X = list(map(int, f.readline().split()))
            X = torch.tensor(X)[:N]

            # Extract charges
            f.readline()
            charges = list(map(int, f.readline().split()))
            charges = torch.tensor(charges)
            N_before_mask = len(charges)
            charges = charges[:N]

            f.readline()
            E = []
            for i in range(N_before_mask):
                E.append(list(map(int, f.readline().split())))
            E = torch.tensor(E)[:N, :N]

            graph = PlaceHolder(X=X, E=E, charges=charges, y=None)
            f.readline()
            mol = Molecule(graph, atom_decoder)
            rdkit_mol = mol.rdkit_mol
            # Try to convert to smiles
            try:
                smiles = Chem.MolToSmiles(rdkit_mol)
            except:
                smiles = None
            print(smiles)
            smiles_list.append(smiles)

        # Save the smiles list to file
        with open(generated_samples_file.split(".")[0] + ".smiles", "w") as f:
            for smiles in smiles_list:
                f.write(smiles + "\n")
    return smiles


def check_ring_constraints(smiles_list, constraint_type, constraint_value, logger=print, timeout_seconds=30):
    total = 0
    passed = 0
    
    # Collect distribution data for better analysis
    ring_counts = []
    ring_lengths = []
    
    # Use a more robust approach without signal handling
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    def check_single_molecule(smi):
        """Check a single molecule for constraint satisfaction without signal handling."""
        if smi is None or smi == "error":
            return None
            
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
                
            if constraint_type == "ring_length_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    max_len = max((len(r) for r in ring_info.AtomRings()), default=0)
                    ring_lengths.append(max_len)
                    return max_len <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_length_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    min_len = min((len(r) for r in ring_info.AtomRings()), default=float('inf'))
                    ring_lengths.append(min_len)
                    return min_len >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
                    
            elif constraint_type == "ring_count_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_count_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            else:
                return None
                
        except Exception as e:
            logger(f"[WARNING] RDKit parsing failed for SMILES: {smi}, error: {e}")
            return None
    
    # Process molecules in batches to avoid overwhelming the system
    batch_size = 50
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        
        # Use ThreadPoolExecutor for timeout handling without signals
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                # Submit all molecules in the batch
                future_to_smi = {executor.submit(check_single_molecule, smi): smi for smi in batch}
                
                # Process results with timeout
                for future in future_to_smi:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        if result is not None:
                            total += 1
                            if result:
                                passed += 1
                    except TimeoutError:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check timed out for SMILES: {smi}")
                        continue
                    except Exception as e:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check failed for SMILES: {smi}, error: {e}")
                        continue
                        
            except Exception as e:
                logger(f"[WARNING] Batch processing failed: {e}")
                continue
    
    # Determine the correct operator based on constraint type
    if constraint_type in ["ring_count_at_least", "ring_length_at_least"]:
        operator = "≥"
    else:
        operator = "≤"
    
    # Calculate satisfaction rate
    satisfaction_rate = (passed / total * 100) if total > 0 else 0.0
    
    # Enhanced output with distribution information
    # logger(f"[Constraint Check] {passed}/{total} molecules satisfy {constraint_type} {operator} {constraint_value} ({satisfaction_rate:.1f}%)")
    
    # Add distribution information for better understanding
    # if ring_counts and constraint_type.startswith("ring_count"):
    #     from collections import Counter
    #     count_dist = Counter(ring_counts)
    #     # Format distribution in a readable way
    #     dist_str = ", ".join([f"{count} rings: {freq} molecules" for count, freq in sorted(count_dist.items())])
    #     logger(f"[Distribution] Ring counts: {dist_str}")
    # elif ring_lengths and constraint_type.startswith("ring_length"):
    #     from collections import Counter
    #     length_dist = Counter(ring_lengths)
    #     # Format distribution in a readable way
    #     dist_str = ", ".join([f"{length}-atom rings: {freq} rings" for length, freq in sorted(length_dist.items())])
    #     logger(f"[Distribution] Ring lengths: {dist_str}")
    
    # Return results for potential use
    return {
        'constraint_type': constraint_type,
        'constraint_value': constraint_value,
        'total_molecules': total,
        'satisfied_molecules': passed,
        'satisfaction_rate': satisfaction_rate / 100.0,  # Return as decimal
        'ring_counts': ring_counts,
        'ring_lengths': ring_lengths
    }


def analyze_all_molecules_constraints(all_generated_smiles: List[str], constraint_type: str, constraint_value: int) -> Dict:
    """
    Analyze constraint satisfaction for ALL molecules (including invalid ones).
    This provides structural constraint analysis following ConStruct philosophy.
    
    Args:
        all_generated_smiles: List of all generated SMILES (including None and "error")
        constraint_type: Type of constraint
        constraint_value: Value of the constraint
        
    Returns:
        Dictionary with structural constraint analysis results
    """
    total_molecules = len(all_generated_smiles)
    valid_molecules = len([s for s in all_generated_smiles if s and s != "error"])
    
    # Analyze ALL molecules for structural constraints
    structural_ring_counts = []
    structural_ring_lengths = []
    structural_planar_count = 0
    structural_non_planar_count = 0
    
    for smiles in all_generated_smiles:
        if smiles and smiles != "error":
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Ring count analysis
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    structural_ring_counts.append(n_rings)
                    
                    # Ring length analysis
                    if ring_info and ring_info.NumRings() > 0:
                        for ring in ring_info.AtomRings():
                            structural_ring_lengths.append(len(ring))
                    
                    # Planarity analysis
                    try:
                        # Generate 3D conformer for planarity check
                        AllChem.EmbedMolecule(mol)
                        AllChem.MMFFOptimizeMolecule(mol)
                        conf = mol.GetConformer()
                        coords = conf.GetPositions()
                        
                        if len(coords) >= 3:
                            # Calculate the normal vector of the plane formed by first 3 atoms
                            v1 = coords[1] - coords[0]
                            v2 = coords[2] - coords[0]
                            normal = np.cross(v1, v2)
                            normal = normal / np.linalg.norm(normal)
                            
                            # Check if all other atoms are close to this plane
                            planar = True
                            for i in range(3, len(coords)):
                                v = coords[i] - coords[0]
                                distance = abs(np.dot(v, normal))
                                if distance > 0.5:  # Threshold for planarity
                                    planar = False
                                    break
                            
                            if planar:
                                structural_planar_count += 1
                            else:
                                structural_non_planar_count += 1
                        else:
                            structural_non_planar_count += 1
                    except:
                        structural_non_planar_count += 1
            except:
                pass
    
    # Calculate structural constraint rates
    structural_ring_count_rates = {}
    for max_rings in range(7):  # 0 to 6 rings
        satisfied = sum(1 for count in structural_ring_counts if count <= max_rings)
        rate = satisfied / total_molecules if total_molecules > 0 else 0.0
        structural_ring_count_rates[f"≤{max_rings}"] = rate
    
    # Calculate structural ring length rates
    structural_ring_length_rates = {}
    for max_length in range(3, 9):  # 3 to 8 atoms
        satisfied = sum(1 for length in structural_ring_lengths if length <= max_length)
        rate = satisfied / len(structural_ring_lengths) if structural_ring_lengths else 0.0
        structural_ring_length_rates[f"≤{max_length}"] = rate
    
    # Calculate planarity rate
    structural_planarity_rate = structural_planar_count / total_molecules if total_molecules > 0 else 0.0
    
    return {
        'total_molecules': total_molecules,
        'valid_molecules': valid_molecules,
        'structural_ring_counts': structural_ring_counts,
        'structural_ring_lengths': structural_ring_lengths,
        'structural_planar_count': structural_planar_count,
        'structural_non_planar_count': structural_non_planar_count,
        'structural_ring_count_rates': structural_ring_count_rates,
        'structural_ring_length_rates': structural_ring_length_rates,
        'structural_planarity_rate': structural_planarity_rate
    }

def check_ring_constraints_all_molecules(smiles_list, constraint_type, constraint_value, logger=print, timeout_seconds=30):
    """
    Check ring constraints for ALL molecules (including invalid ones).
    This provides structural constraint analysis.
    """
    total = len(smiles_list)
    passed = 0
    
    # Collect distribution data for structural analysis
    ring_counts = []
    ring_lengths = []
    
    def check_single_molecule(smi):
        """Check a single molecule for constraint satisfaction."""
        if smi is None or smi == "error":
            return None
            
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
                
            if constraint_type == "ring_length_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    max_len = max((len(r) for r in ring_info.AtomRings()), default=0)
                    ring_lengths.append(max_len)
                    return max_len <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_length_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    min_len = min((len(r) for r in ring_info.AtomRings()), default=float('inf'))
                    ring_lengths.append(min_len)
                    return min_len >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
                    
            elif constraint_type == "ring_count_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_count_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            else:
                return None
                
        except Exception as e:
            logger(f"[WARNING] RDKit parsing failed for SMILES: {smi}, error: {e}")
            return None
    
    # Process molecules in batches
    batch_size = 50
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                future_to_smi = {executor.submit(check_single_molecule, smi): smi for smi in batch}
                
                for future in future_to_smi:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        if result is not None:
                            if result:
                                passed += 1
                    except TimeoutError:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check timed out for SMILES: {smi}")
                        continue
                    except Exception as e:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check failed for SMILES: {smi}, error: {e}")
                        continue
                        
            except Exception as e:
                logger(f"[WARNING] Batch processing failed: {e}")
                continue
    
    # Determine the correct operator based on constraint type
    if constraint_type in ["ring_count_at_least", "ring_length_at_least"]:
        operator = "≥"
    else:
        operator = "≤"
    
    # Calculate satisfaction rate (structural - based on ALL molecules)
    satisfaction_rate = (passed / total * 100) if total > 0 else 0.0
    
    # Enhanced output with distribution information
    logger(f"[Structural Constraint Check] {passed}/{total} molecules satisfy {constraint_type} {operator} {constraint_value} ({satisfaction_rate:.1f}%)")
    
    # Add distribution information
    if ring_counts and constraint_type.startswith("ring_count"):
        from collections import Counter
        count_dist = Counter(ring_counts)
        dist_str = ", ".join([f"{count} rings: {freq} molecules" for count, freq in sorted(count_dist.items())])
        logger(f"[Structural Distribution] Ring counts: {dist_str}")
    elif ring_lengths and constraint_type.startswith("ring_length"):
        from collections import Counter
        length_dist = Counter(ring_lengths)
        dist_str = ", ".join([f"{length}-atom rings: {freq} rings" for length, freq in sorted(length_dist.items())])
        logger(f"[Structural Distribution] Ring lengths: {dist_str}")
    
    return {
        'constraint_type': constraint_type,
        'constraint_value': constraint_value,
        'total_molecules': total,
        'satisfied_molecules': passed,
        'satisfaction_rate': satisfaction_rate / 100.0,
        'ring_counts': ring_counts,
        'ring_lengths': ring_lengths
    }


def check_planarity_constraint(smiles_list, logger=print):
    """
    Check planarity constraint for valid molecules only.
    Uses the existing is_planar function from the codebase.
    
    Args:
        smiles_list: List of SMILES strings (valid molecules only)
        logger: Logger function
        
    Returns:
        Dictionary with planarity results
    """
    satisfied_molecules = 0
    total_molecules = 0
    
    for smi in smiles_list:
        if smi and smi != "error":
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    total_molecules += 1
                    # Use existing planarity check from codebase
                    try:
                        from ConStruct.projector.is_planar import is_planar
                        import networkx as nx
                        
                        # Convert RDKit molecule to NetworkX graph
                        adj_matrix = Chem.GetAdjacencyMatrix(mol)
                        nx_graph = nx.from_numpy_array(adj_matrix)
                        
                        if is_planar(nx_graph):
                            satisfied_molecules += 1
                    except Exception as e:
                        # If planarity check fails, assume non-planar
                        pass
            except:
                pass
    
    satisfaction_rate = (satisfied_molecules / total_molecules) if total_molecules > 0 else 0.0
    
    return {
        'satisfied_molecules': satisfied_molecules,
        'total_molecules': total_molecules,
        'satisfaction_rate': satisfaction_rate
    }


def check_planarity_constraint_all_molecules(smiles_list, logger=print):
    """
    Check planarity constraint for ALL molecules (including invalid ones).
    Uses the existing is_planar function from the codebase.
    
    Args:
        smiles_list: List of SMILES strings (ALL molecules)
        logger: Logger function
        
    Returns:
        Dictionary with planarity results
    """
    satisfied_molecules = 0
    total_molecules = 0
    
    for smi in smiles_list:
        if smi and smi != "error":
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    total_molecules += 1
                    # Use existing planarity check from codebase
                    try:
                        from ConStruct.projector.is_planar import is_planar
                        import networkx as nx
                        
                        # Convert RDKit molecule to NetworkX graph
                        adj_matrix = Chem.GetAdjacencyMatrix(mol)
                        nx_graph = nx.from_numpy_array(adj_matrix)
                        
                        if is_planar(nx_graph):
                            satisfied_molecules += 1
                    except Exception as e:
                        # If planarity check fails, assume non-planar
                        pass
            except:
                pass
    
    satisfaction_rate = (satisfied_molecules / total_molecules) if total_molecules > 0 else 0.0
    
    return {
        'satisfied_molecules': satisfied_molecules,
        'total_molecules': total_molecules,
        'satisfaction_rate': satisfaction_rate
    }


if __name__ == "__main__":
    file = ""
    atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    smiles_list = smiles_from_generated_samples_file(file, atom_decoder)
    print(smiles_list)
