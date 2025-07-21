Here’s how you should **update your original README** so that **no one loses days of their life** fighting dependency hell again. I’m not deleting anything, just heavily commenting, revising, and adding modern cluster best-practices, **with big warnings and explanations where needed**.

---
## Generative Modelling of Structurally Constrained Graphs

---

### 🚦 Bulletproof Environment Setup Instructions (with `fcd`)
> **Read this section before touching the old instructions below!**
> 
> These steps are based on real-world cluster, GPU, RDKit, PyTorch, and graph-tool nightmares.
>  
> **You MUST follow the order and warnings below, or your environment will break.**

---

### 1. **Create and Activate Your Conda Environment**

```bash
conda create -y -c conda-forge -n construct python=3.9 rdkit=2023.03.2
conda activate construct
```

### 2. **Check RDKit Works**

```bash
python -c "from rdkit import Chem"
# No error means it's fine.
```

### 3. **Install graph-tool (optional)**

```bash
conda install -c conda-forge graph-tool=2.45
python -c "import graph_tool as gt"
```
⚠️ NOTE:
* *graph-tool* is **only required for non-molecular datasets** (e.g., tree, planar, lobster).
* If you work only with molecular datasets (QM9, etc.), you can skip installing graph-tool to avoid compatibility headaches.

### 4. **Install PyTorch (CUDA 11.8), then torch-geometric**

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.3.1
python -c "import torch; print(torch.cuda.is_available())"
# Should print True if GPU is visible.
```

### 5. **Install fcd (Fréchet ChemNet Distance, CODE ONLY -> don't pip install fcd !!!)**

```bash
pip install --no-deps fcd
# Do NOT install dependencies here, or you WILL break torch/rdkit versions!
# This gives you fcd.load_ref_model, fcd.get_fcd, etc.
```

### 6. **Install the Rest of Your Requirements**

```bash
pip install -r requirements.txt
# (If requirements.txt has torch or rdkit, double check they don't get downgraded!)
```

### 7. **Install Your Own Package (Editable Dev Mode, If Needed)**

```bash
pip install -e .
```

### 8. **Compile ORCA if Needed**

```bash
cd ./ConStruct/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
cd -
```

### 9. **Test Everything**

```bash
python -c "import fcd; print(hasattr(fcd, 'load_ref_model'))"
python -c "import torch; print(torch.cuda.is_available())"
```

Both should print `True` or not error.

---

#### ⚠️ **CRITICAL WARNINGS!**

* **Never** use `pip install fcd` (without `--no-deps`) after torch/rdkit, or you’ll nuke your versions.
* **Never** install `cuda` libraries via conda. Cluster GPUs already have drivers.
* **Never** install both `fcd` and `fcd_torch` in the same env unless you know why.
* **Always** check for libstdc++ or libgomp errors (see troubleshooting below).

---

### 🔗 **Summary Table**

| Step                    | Command                                                                       |
| ----------------------- | ----------------------------------------------------------------------------- |
| Create env              | `conda create -y -c conda-forge -n construct python=3.9 rdkit=2023.03.2`      |
| Activate env            | `conda activate construct`                                                    |
| Install graph-tool      | `conda install -c conda-forge graph-tool=2.45`                                |
| Install PyTorch         | `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118` |
| Install torch-geometric | `pip install torch-geometric==2.3.1`                                          |
| Install fcd             | `pip install --no-deps fcd`                                                   |
| Other packages          | `pip install -r requirements.txt`                                             |
| Your package            | `pip install -e .`                                                            |
| Compile ORCA            | `g++ -O2 -std=c++11 -o orca orca.cpp`                                         |

---

## 🆘 **Troubleshooting**

### fcd/rdkit libstdc++ error:

If you get something like
`ImportError: ... libstdc++.so.6: version 'GLIBCXX_3.4.29' not found ...`
run:

```bash
find $CONDA_PREFIX -name "libstdc++.so.6"
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python -c "from rdkit import Chem; import fcd; print(hasattr(fcd, 'load_ref_model'))"
```

If it fixes things, **make it permanent**:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"' > $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libstdcxx.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libstdcxx.sh
```

### graph-tool/libgomp error:

If you see
`libgomp-a34b3233.so.1: version 'GOMP_5.0' not found (required by ...)`
run:

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
python test_env.py
```

If it works, make it permanent:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"' > $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libgomp.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libgomp.sh
```

If still not working, add to every SLURM script after `conda activate`:

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
```

---

## **Quick Cluster Sanity Check Script**

Paste and run these one by one in your **(construct)** environment:

1. **RDKit Basic Import**

   ```bash
   python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO') is not None)"
   ```
2. **PyTorch + CUDA Check**

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
3. **torch-geometric Check**

   ```bash
   python -c "import torch_geometric; print(torch_geometric.__version__)"
   ```
4. **fcd Import and Model Load**

   ```bash
   python -c "import fcd; print(hasattr(fcd, 'load_ref_model')); m = fcd.load_ref_model(); print(m is not None)"
   ```
5. **Your Own Package Import**

   ```bash
   python -c "import ConStruct; print('ConStruct imported\!')"
   ```
6. **(Optional) Try a minimal fcd score calculation**

   ```bash
   python -c "import fcd; s = fcd.get_fcd(['CCO', 'CCC'], ['CCO', 'CCN']); print('FCD score:', s)"
   ```

If all these work: **your env is cluster-proof**.

---

## --- Original README instructions below ---
## (kept for reference, see above for robust setup) 

### \[LEGACY/REFERENCE] Environment installation

> **WARNING:** The below steps are error-prone on modern clusters.
> **Follow the bulletproof steps above for reliable cluster installs!**

This code was tested with PyTorch 2.0.1, cuda 11.8 and torch\_geometrics 2.3.1

* Download anaconda/miniconda if needed

* Create a rdkit environment that directly contains rdkit:

  `conda create -c conda-forge -n construct rdkit=2023.03.2 python=3.9`

* `conda activate construct`

* Check that this line does not return an error:

  `python3 -c 'from rdkit import Chem'`

* Install graph-tool ([https://graph-tool.skewed.de/](https://graph-tool.skewed.de/)):

  `conda install -c conda-forge graph-tool=2.45`

* Check that this line does not return an error:

  `python3 -c 'import graph_tool as gt' `

* ~~Install the nvcc drivers for your cuda version. For example:~~
  **(DO NOT DO THIS ON A CLUSTER, drivers are managed by the system)**

  ```diff
  - conda install -c "nvidia/label/cuda-11.8.0" cuda
  ```

* Install a corresponding version of pytorch, for example:

  `pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118`

* Install other packages using the requirement file:

  `pip install -r requirements.txt`

* Run:

  `pip install -e .`

* Navigate to the ./ConStruct/analysis/orca directory and compile orca.cpp:

  `g++ -O2 -std=c++11 -o orca orca.cpp`

---

## Run the code

### 🚀 **Organized Experiment Structure**

The codebase now includes a comprehensive, organized experiment structure for testing different constraint types:

#### **Directory Structure**
```
configs/experiment/
├── debug/                          # Debug-level experiments (quick testing)
│   ├── no_constraint/             # No constraint experiments
│   ├── edge_deletion/             # Edge-deletion constraints ("at most")
│   │   ├── ring_count_at_most/   # Ring count "at most" constraints
│   │   └── ring_length_at_most/  # Ring length "at most" constraints
│   └── edge_insertion/            # Edge-insertion constraints ("at least")
│       ├── ring_count_at_least/  # Ring count "at least" constraints
│       └── ring_length_at_least/ # Ring length "at least" constraints
└── thesis/                         # Thesis-level experiments (full-scale)
    ├── no_constraint/             # No constraint experiments
    ├── edge_deletion/             # Edge-deletion constraints ("at most")
    │   ├── ring_count_at_most/   # Ring count "at most" constraints
    │   └── ring_length_at_most/  # Ring length "at most" constraints
    └── edge_insertion/            # Edge-insertion constraints ("at least")
        ├── ring_count_at_least/  # Ring count "at least" constraints
        └── ring_length_at_least/ # Ring length "at least" constraints

ConStruct/slurm_jobs/
├── debug/                          # Debug-level SLURM scripts
│   ├── no_constraint/             # No constraint scripts
│   ├── edge_deletion/             # Edge-deletion scripts
│   └── edge_insertion/            # Edge-insertion scripts
└── thesis/                         # Thesis-level SLURM scripts
    ├── no_constraint/             # No constraint scripts
    ├── edge_deletion/             # Edge-deletion scripts
    └── edge_insertion/            # Edge-insertion scripts
```

#### **Constraint Types**

**Edge-Deletion Constraints ("At Most")**:
- **Purpose**: Limit maximum ring count or ring length
- **Transition**: `absorbing_edges`
- **Projectors**: `ring_count_at_most`, `ring_length_at_most`
- **Use Case**: Generate molecules with limited ring complexity

**Edge-Insertion Constraints ("At Least")**:
- **Purpose**: Ensure minimum ring count or ring length
- **Transition**: `edge_insertion`
- **Projectors**: `ring_count_at_least`, `ring_length_at_least`
- **Use Case**: Generate molecules with guaranteed ring structures

**No Constraint**:
- **Purpose**: Baseline training without any constraints
- **Transition**: `absorbing_edges`
- **Projector**: `null`
- **Use Case**: Generate molecules without structural constraints

### 🧪 **Running Experiments**

#### **Direct Python Execution**
```bash
# Debug experiments
python ConStruct/main.py \
  --config-name experiment/debug/no_constraint/qm9_debug_no_constraint.yaml \
  --config-path configs/

# Thesis experiments
python ConStruct/main.py \
  --config-name experiment/thesis/edge_insertion/ring_count_at_least/qm9_thesis_ring_count_at_least_2.yaml \
  --config-path configs/
```

#### **SLURM Job Submission**
```bash
# Debug experiments
sbatch ConStruct/slurm_jobs/debug/edge_insertion/ring_count_at_least/submit_ring_count_at_least_1_debug.sh

# Thesis experiments
sbatch ConStruct/slurm_jobs/thesis/edge_deletion/ring_count_at_most/submit_ring_count_at_most_3_thesis.sh
```

#### **Legacy Usage** (for backward compatibility)
* All code is currently launched through `python3 main.py`. Check hydra documentation ([https://hydra.cc/](https://hydra.cc/)) for overriding default parameters.
* To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first before launching full experiments.
* To run the diffusion model: `python3 main.py`
* You can specify the dataset with `python3 main.py dataset=tree`. Look at `configs/dataset` for the list of datasets that are currently available
* To reproduce the experiments in the paper, please add the flag `+experiment` to get the correct configuration: `python3 main.py +experiment=<dataset_name>`
* To test the obtained models, specify the path to a model with the flag `general.test_only`, it will load the model and test it, e.g., `python3 main.py +experiment=tree general.test_only=<path>`
* The projector is controlled by the flag `model.rev_proj` (options: `planar`, `tree`, `lobster`, `ring_count_at_most`, `ring_count_at_least`, `ring_length_at_most`, `ring_length_at_least`)
* The transition mechanism is set through `model.transition` (options: `absorbing_edges`, `edge_insertion`, `marginal`, `uniform`).

---

```
**If you ever want this as a standalone README or script, let me know and I’ll format it for you.**
```

---

If you follow this, **you’ll never waste time on cluster setup hell again**.
