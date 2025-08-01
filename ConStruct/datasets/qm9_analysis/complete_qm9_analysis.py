#!/usr/bin/env python3
"""
Complete QM9 Dataset Analysis Script
====================================

This script performs comprehensive analysis of the QM9 dataset including:
1. TRAINING DATASET ANALYSIS - What the model learns from
2. FULL DATASET ANALYSIS - Complete dataset characterization
3. COMPARISON ANALYSIS - Training vs Full dataset differences

Features:
- Ring count and ring length analysis
- Planarity analysis  
- Constraint satisfaction rates
- Visualization plots
- Detailed reports for each analysis type

All results are saved to the qm9_analysis directory structure.
"""

import os
import sys
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time

# Add ConStruct to path
sys.path.append('/home/rislek/ConStruct-Thesis')

def create_output_dir():
    """Create output directory structure"""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different analysis types
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "train_analysis").mkdir(exist_ok=True)
    (output_dir / "full_analysis").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)
    
    return output_dir

def load_qm9_dataset(split="train"):
    """Load QM9 dataset using ConStruct's proper loading mechanism"""
    try:
        from ConStruct.datasets.qm9_dataset import QM9Dataset
        
        print(f"🔍 Loading QM9 dataset ({split} split) using ConStruct's dataset loader...")
        
        # Load specified split
        dataset = QM9Dataset(
            split=split,
            root="/home/rislek/ConStruct-Thesis/data/qm9",
            remove_h=True,
            target_prop=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
        )
        
        print(f"✅ Loaded {split} dataset with {len(dataset)} individual molecules")
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def load_full_qm9_datasets():
    """Load all QM9 dataset splits"""
    try:
        from ConStruct.datasets.qm9_dataset import QM9Dataset
        
        print("🔍 Loading full QM9 dataset (train + val + test)...")
        
        datasets = {}
        for split in ["train", "val", "test"]:
            dataset = QM9Dataset(
                split=split,
                root="/home/rislek/ConStruct-Thesis/data/qm9",
                remove_h=True,
                target_prop=None,
                transform=None,
                pre_transform=None,
                pre_filter=None,
            )
            datasets[split] = dataset
            print(f"✅ Loaded {split} split: {len(dataset)} molecules")
        
        return datasets
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None

def analyze_ring_properties(dataset, dataset_name="dataset"):
    """Analyze ring count and ring length properties"""
    print(f"\n🔍 Analyzing ring properties for {dataset_name}...")
    
    ring_count_dist = Counter()
    ring_length_dist = Counter()
    max_ring_length_per_mol = []
    total_rings = 0
    
    for i, mol in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processing molecule {i}/{len(dataset)}")
        
        # Convert PyTorch Geometric Data to NetworkX graph
        edge_index = mol.edge_index.cpu().numpy()
        num_nodes = mol.x.shape[0]
        
        # Create adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for edge in edge_index.T:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1
        
        nx_graph = nx.from_numpy_array(adj_matrix)
        
        # Count rings
        cycles = nx.cycle_basis(nx_graph)
        ring_count = len(cycles)
        ring_count_dist[ring_count] += 1
        total_rings += ring_count
        
        # Analyze ring lengths
        max_length_in_mol = 0
        for cycle in cycles:
            ring_length = len(cycle)
            ring_length_dist[ring_length] += 1
            max_length_in_mol = max(max_length_in_mol, ring_length)
        
        max_ring_length_per_mol.append(max_length_in_mol)
    
    return ring_count_dist, ring_length_dist, max_ring_length_per_mol, total_rings

def analyze_planarity(dataset, dataset_name="dataset"):
    """Analyze planarity properties"""
    print(f"\n🔍 Analyzing planarity for {dataset_name}...")
    
    try:
        from ConStruct.projector.is_planar import is_planar
        planar_count = 0
        non_planar_count = 0
        
        for i, mol in enumerate(dataset):
            if i % 1000 == 0:
                print(f"  Processing molecule {i}/{len(dataset)}")
            
            # Convert PyTorch Geometric Data to NetworkX graph
            edge_index = mol.edge_index.cpu().numpy()
            num_nodes = mol.x.shape[0]
            
            # Create adjacency matrix
            adj_matrix = np.zeros((num_nodes, num_nodes))
            for edge in edge_index.T:
                adj_matrix[edge[0], edge[1]] = 1
                adj_matrix[edge[1], edge[0]] = 1
            
            nx_graph = nx.from_numpy_array(adj_matrix)
            
            if is_planar(nx_graph):
                planar_count += 1
            else:
                non_planar_count += 1
        
        return planar_count, non_planar_count
    except ImportError:
        print("⚠️ Planarity analysis not available (is_planar module not found)")
        return None, None

def calculate_constraint_rates(ring_count_dist, ring_length_dist, max_ring_length_per_mol):
    """Calculate rates for different constraint thresholds"""
    print("\n📊 Calculating constraint satisfaction rates...")
    
    total_molecules = sum(ring_count_dist.values())
    
    # Ring count rates (≤0, ≤1, ≤2, ≤3, ≤4, ≤5)
    ring_count_rates = {}
    for max_rings in [0, 1, 2, 3, 4, 5]:
        count = sum(ring_count_dist[i] for i in range(max_rings + 1))
        rate = count / total_molecules
        ring_count_rates[f"≤{max_rings}"] = rate
        print(f"  Ring count ≤{max_rings}: {count}/{total_molecules} = {rate:.3f} ({rate*100:.1f}%)")
    
    # Ring length rates (≤3, ≤4, ≤5, ≤6, ≤7, ≤8)
    ring_length_rates = {}
    for max_length in [3, 4, 5, 6, 7, 8]:
        count = sum(1 for max_len in max_ring_length_per_mol if max_len <= max_length)
        rate = count / total_molecules
        ring_length_rates[f"≤{max_length}"] = rate
        print(f"  Ring length ≤{max_length}: {count}/{total_molecules} = {rate:.3f} ({rate*100:.1f}%)")
    
    return ring_count_rates, ring_length_rates

def create_plots(ring_count_dist, ring_length_dist, planar_count, non_planar_count, 
                ring_count_rates, ring_length_rates, output_dir, analysis_name="analysis"):
    """Create comprehensive visualization plots"""
    print(f"\n📈 Creating visualization plots for {analysis_name}...")
    
    # Set style
    plt.style.use('default')
    
    # 1. Ring Count Distribution
    plt.figure(figsize=(10, 6))
    ring_counts = sorted(ring_count_dist.keys())
    ring_count_values = [ring_count_dist[c] for c in ring_counts]
    bars = plt.bar(ring_counts, ring_count_values, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.title(f'QM9 {analysis_name}: Ring Count Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Rings', fontsize=12)
    plt.ylabel('Number of Molecules', fontsize=12)
    plt.xticks(ring_counts)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, ring_count_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ring_count_values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ring_count_distribution_{analysis_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Ring Length Distribution
    plt.figure(figsize=(10, 6))
    ring_lengths = sorted(ring_length_dist.keys())
    ring_length_values = [ring_length_dist[l] for l in ring_lengths]
    bars = plt.bar(ring_lengths, ring_length_values, alpha=0.7, color='lightcoral', edgecolor='darkred')
    plt.title(f'QM9 {analysis_name}: Ring Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Ring Length (Number of Atoms)', fontsize=12)
    plt.ylabel('Number of Rings', fontsize=12)
    plt.xticks(ring_lengths)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, ring_length_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ring_length_values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ring_length_distribution_{analysis_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Planarity Distribution
    if planar_count is not None:
        plt.figure(figsize=(8, 8))
        labels = ['Planar', 'Non-Planar']
        sizes = [planar_count, non_planar_count]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title(f'QM9 {analysis_name}: Planarity Distribution', fontsize=14, fontweight='bold')
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"planarity_distribution_{analysis_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Constraint Satisfaction Rates
    plt.figure(figsize=(12, 8))
    
    # Ring count rates
    plt.subplot(2, 1, 1)
    ring_count_constraints = ['≤0', '≤1', '≤2', '≤3', '≤4', '≤5']
    ring_count_rates_list = [ring_count_rates[c] for c in ring_count_constraints]
    
    bars = plt.bar(ring_count_constraints, ring_count_rates_list, alpha=0.7, color='gold', edgecolor='orange')
    plt.title(f'Ring Count Constraint Satisfaction Rates ({analysis_name})', fontsize=12, fontweight='bold')
    plt.ylabel('Satisfaction Rate', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, ring_count_rates_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ring length rates
    plt.subplot(2, 1, 2)
    ring_length_constraints = ['≤3', '≤4', '≤5', '≤6', '≤7', '≤8']
    ring_length_rates_list = [ring_length_rates[c] for c in ring_length_constraints]
    
    bars = plt.bar(ring_length_constraints, ring_length_rates_list, alpha=0.7, color='lightblue', edgecolor='navy')
    plt.title(f'Ring Length Constraint Satisfaction Rates ({analysis_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Constraint', fontsize=10)
    plt.ylabel('Satisfaction Rate', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, ring_length_rates_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"constraint_satisfaction_rates_{analysis_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved plots to {output_dir}/")

def save_results(ring_count_dist, ring_length_dist, ring_count_rates, ring_length_rates, 
                planar_count, non_planar_count, total_rings, total_molecules, output_dir, analysis_name="analysis"):
    """Save analysis results to files"""
    print(f"\n💾 Saving {analysis_name} results...")
    
    # Prepare results dictionary
    results = {
        'analysis_type': analysis_name,
        'dataset_info': {
            'total_molecules': total_molecules,
            'total_rings': total_rings,
            'avg_rings_per_molecule': total_rings / total_molecules if total_molecules > 0 else 0
        },
        'ring_count_distribution': dict(ring_count_dist),
        'ring_length_distribution': dict(ring_length_dist),
        'ring_count_rates': ring_count_rates,
        'ring_length_rates': ring_length_rates,
        'planarity': {
            'planar_molecules': planar_count,
            'non_planar_molecules': non_planar_count,
            'planarity_rate': planar_count / (planar_count + non_planar_count) if planar_count is not None else None
        }
    }
    
    # Save as JSON
    with open(output_dir / f"qm9_{analysis_name}_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed text report
    with open(output_dir / f"qm9_{analysis_name}_analysis_report.txt", 'w') as f:
        f.write(f"QM9 {analysis_name.title()} Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Analysis type: {analysis_name}\n")
        f.write(f"Total molecules analyzed: {total_molecules:,}\n")
        f.write(f"Total rings found: {total_rings:,}\n")
        f.write(f"Average rings per molecule: {total_rings/total_molecules:.3f}\n\n")
        
        f.write("RING COUNT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Distribution of molecules by ring count:\n")
        for count, num_mols in sorted(ring_count_dist.items()):
            percentage = (num_mols / total_molecules) * 100
            f.write(f"  {count} rings: {num_mols:,} molecules ({percentage:.1f}%)\n")
        
        f.write("\nRING LENGTH ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Distribution of rings by length:\n")
        for length, num_rings in sorted(ring_length_dist.items()):
            f.write(f"  {length}-atom rings: {num_rings:,} rings\n")
        
        f.write("\nCONSTRAINT SATISFACTION RATES\n")
        f.write("-" * 30 + "\n")
        f.write("Ring Count Constraints:\n")
        for constraint, rate in ring_count_rates.items():
            f.write(f"  {constraint}: {rate:.3f} ({rate*100:.1f}%)\n")
        
        f.write("\nRing Length Constraints:\n")
        for constraint, rate in ring_length_rates.items():
            f.write(f"  {constraint}: {rate:.3f} ({rate*100:.1f}%)\n")
        
        if planar_count is not None:
            f.write(f"\nPLANARITY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Planar molecules: {planar_count:,}\n")
            f.write(f"Non-planar molecules: {non_planar_count:,}\n")
            f.write(f"Planarity rate: {planar_count/(planar_count+non_planar_count):.3f} ({(planar_count/(planar_count+non_planar_count))*100:.1f}%)\n")
        
        f.write(f"\n\nAnalysis completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Saved {analysis_name} results to {output_dir}/")

def create_comparison_analysis(train_results, full_results, output_dir):
    """Create comparison analysis between train and full dataset"""
    print("\n📊 Creating comparison analysis...")
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Ring count comparison
    plt.subplot(2, 2, 1)
    train_rates = [train_results['ring_count_rates'][f"≤{i}"] for i in range(6)]
    full_rates = [full_results['ring_count_rates'][f"≤{i}"] for i in range(6)]
    x = range(6)
    
    plt.plot(x, train_rates, 'o-', label='Training Dataset', linewidth=2, markersize=8)
    plt.plot(x, full_rates, 's-', label='Full Dataset', linewidth=2, markersize=8)
    plt.title('Ring Count Constraint Satisfaction Comparison', fontweight='bold')
    plt.xlabel('Maximum Rings')
    plt.ylabel('Satisfaction Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ring length comparison
    plt.subplot(2, 2, 2)
    train_length_rates = [train_results['ring_length_rates'][f"≤{i}"] for i in range(3, 9)]
    full_length_rates = [full_results['ring_length_rates'][f"≤{i}"] for i in range(3, 9)]
    x = range(3, 9)
    
    plt.plot(x, train_length_rates, 'o-', label='Training Dataset', linewidth=2, markersize=8)
    plt.plot(x, full_length_rates, 's-', label='Full Dataset', linewidth=2, markersize=8)
    plt.title('Ring Length Constraint Satisfaction Comparison', fontweight='bold')
    plt.xlabel('Maximum Ring Length')
    plt.ylabel('Satisfaction Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dataset size comparison
    plt.subplot(2, 2, 3)
    sizes = [train_results['dataset_info']['total_molecules'], full_results['dataset_info']['total_molecules']]
    labels = ['Training Dataset', 'Full Dataset']
    colors = ['lightblue', 'lightcoral']
    
    bars = plt.bar(labels, sizes, color=colors, alpha=0.7)
    plt.title('Dataset Size Comparison', fontweight='bold')
    plt.ylabel('Number of Molecules')
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    # Average rings comparison
    plt.subplot(2, 2, 4)
    avg_rings = [train_results['dataset_info']['avg_rings_per_molecule'], 
                 full_results['dataset_info']['avg_rings_per_molecule']]
    
    bars = plt.bar(labels, avg_rings, color=colors, alpha=0.7)
    plt.title('Average Rings per Molecule Comparison', fontweight='bold')
    plt.ylabel('Average Rings')
    
    # Add value labels
    for bar, avg in zip(bars, avg_rings):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_rings)*0.01,
                f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison report
    with open(output_dir / "comparison_report.txt", 'w') as f:
        f.write("QM9 Training vs Full Dataset Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET SIZE COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training dataset: {train_results['dataset_info']['total_molecules']:,} molecules\n")
        f.write(f"Full dataset: {full_results['dataset_info']['total_molecules']:,} molecules\n")
        f.write(f"Difference: {full_results['dataset_info']['total_molecules'] - train_results['dataset_info']['total_molecules']:,} molecules\n\n")
        
        f.write("RING COUNT COMPARISON\n")
        f.write("-" * 25 + "\n")
        for i in range(6):
            train_rate = train_results['ring_count_rates'][f"≤{i}"]
            full_rate = full_results['ring_count_rates'][f"≤{i}"]
            diff = full_rate - train_rate
            f.write(f"≤{i} rings: Train={train_rate:.3f}, Full={full_rate:.3f}, Diff={diff:+.3f}\n")
        
        f.write("\nRING LENGTH COMPARISON\n")
        f.write("-" * 27 + "\n")
        for i in range(3, 9):
            train_rate = train_results['ring_length_rates'][f"≤{i}"]
            full_rate = full_results['ring_length_rates'][f"≤{i}"]
            diff = full_rate - train_rate
            f.write(f"≤{i} atoms: Train={train_rate:.3f}, Full={full_rate:.3f}, Diff={diff:+.3f}\n")
        
        f.write(f"\n\nComparison completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Saved comparison analysis to {output_dir}/")

def main():
    """Main analysis function"""
    print("🔬 Complete QM9 Dataset Analysis")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"📁 Output directory: {output_dir}")
    
    # ===== 1. TRAINING DATASET ANALYSIS =====
    print("\n" + "="*50)
    print("📚 TRAINING DATASET ANALYSIS")
    print("="*50)
    
    train_dataset = load_qm9_dataset("train")
    if train_dataset is None:
        print("❌ Failed to load training dataset. Exiting.")
        return
    
    # Analyze training dataset
    train_ring_count_dist, train_ring_length_dist, train_max_ring_length_per_mol, train_total_rings = analyze_ring_properties(train_dataset, "Training Dataset")
    train_planar_count, train_non_planar_count = analyze_planarity(train_dataset, "Training Dataset")
    train_ring_count_rates, train_ring_length_rates = calculate_constraint_rates(train_ring_count_dist, train_ring_length_dist, train_max_ring_length_per_mol)
    
    # Save training analysis
    train_results = {
        'dataset_info': {
            'total_molecules': len(train_dataset),
            'total_rings': train_total_rings,
            'avg_rings_per_molecule': train_total_rings / len(train_dataset)
        },
        'ring_count_rates': train_ring_count_rates,
        'ring_length_rates': train_ring_length_rates
    }
    
    save_results(train_ring_count_dist, train_ring_length_dist, train_ring_count_rates, train_ring_length_rates,
                train_planar_count, train_non_planar_count, train_total_rings, len(train_dataset), 
                output_dir / "train_analysis", "training")
    
    create_plots(train_ring_count_dist, train_ring_length_dist, train_planar_count, train_non_planar_count,
                train_ring_count_rates, train_ring_length_rates, output_dir / "train_analysis", "Training")
    
    # ===== 2. FULL DATASET ANALYSIS =====
    print("\n" + "="*50)
    print("🌐 FULL DATASET ANALYSIS")
    print("="*50)
    
    full_datasets = load_full_qm9_datasets()
    if full_datasets is None:
        print("❌ Failed to load full datasets. Exiting.")
        return
    
    # Combine all splits for full dataset analysis
    full_dataset = []
    for split_name, dataset in full_datasets.items():
        full_dataset.extend(dataset)
    
    print(f"📊 Combined dataset: {len(full_dataset)} molecules")
    
    # Analyze full dataset
    full_ring_count_dist, full_ring_length_dist, full_max_ring_length_per_mol, full_total_rings = analyze_ring_properties(full_dataset, "Full Dataset")
    full_planar_count, full_non_planar_count = analyze_planarity(full_dataset, "Full Dataset")
    full_ring_count_rates, full_ring_length_rates = calculate_constraint_rates(full_ring_count_dist, full_ring_length_dist, full_max_ring_length_per_mol)
    
    # Save full dataset analysis
    full_results = {
        'dataset_info': {
            'total_molecules': len(full_dataset),
            'total_rings': full_total_rings,
            'avg_rings_per_molecule': full_total_rings / len(full_dataset)
        },
        'ring_count_rates': full_ring_count_rates,
        'ring_length_rates': full_ring_length_rates
    }
    
    save_results(full_ring_count_dist, full_ring_length_dist, full_ring_count_rates, full_ring_length_rates,
                full_planar_count, full_non_planar_count, full_total_rings, len(full_dataset), 
                output_dir / "full_analysis", "full")
    
    create_plots(full_ring_count_dist, full_ring_length_dist, full_planar_count, full_non_planar_count,
                full_ring_count_rates, full_ring_length_rates, output_dir / "full_analysis", "Full")
    
    # ===== 3. COMPARISON ANALYSIS =====
    print("\n" + "="*50)
    print("📊 COMPARISON ANALYSIS")
    print("="*50)
    
    create_comparison_analysis(train_results, full_results, output_dir / "comparison")
    
    # ===== 4. SUMMARY =====
    print("\n" + "="*50)
    print("📋 ANALYSIS SUMMARY")
    print("="*50)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Training dataset analysis: {len(train_dataset):,} molecules")
    print(f"✅ Full dataset analysis: {len(full_dataset):,} molecules")
    print(f"✅ Comparison analysis completed")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("   ├── train_analysis/ (training dataset analysis)")
    print("   ├── full_analysis/ (full dataset analysis)")
    print("   ├── comparison/ (comparison analysis)")
    print("   ├── plots/ (visualization charts)")
    print("   ├── reports/ (detailed analysis reports)")
    print("   └── data/ (JSON data files)")
    print(f"⏱️  Analysis completed in {duration:.1f} seconds")
    print("✅ Complete analysis finished!")

if __name__ == "__main__":
    main() 