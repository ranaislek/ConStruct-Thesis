import os, statistics
from typing import Dict, Any, List, Optional, Tuple

# -------- formatting (no arrows, consistent precision) --------
def pct(x: Optional[float]) -> str:
    if x is None: return "NA"
    if 0.0 <= x <= 1.0: x *= 100.0
    return f"{x:.2f}%"

def f3(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"

# --- new: robust scalar coercion for hist/arrays/tensors/W&B Histogram ---
def to_scalar(value) -> Optional[float]:
    """
    Convert metric payloads (float/int/list/tuple/np.ndarray/torch.Tensor/dict/W&B Histogram-like)
    into a single scalar suitable for table printing. Returns None if not possible.
    For histograms (e.g., abs_diff_deg_hist), we return the L1 sum (sum of absolute bins).
    """
    if value is None:
        return None
    # numeric fast-path
    if isinstance(value, (int, float)):
        return float(value)
    # torch
    try:
        import torch  # type: ignore
        if isinstance(value, torch.Tensor):
            return float(value.detach().abs().sum().cpu().item())
    except Exception:
        pass
    # numpy
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.ndarray):
            return float(np.abs(value).sum())
    except Exception:
        pass
    # list/tuple -> sum abs
    if isinstance(value, (list, tuple)):
        try:
            return float(sum(abs(float(v)) for v in value))
        except Exception:
            return None
    # dict -> sum abs of values
    if isinstance(value, dict):
        try:
            return float(sum(abs(float(v)) for v in value.values()))
        except Exception:
            return None
    # W&B Histogram-like: has .histogram (list of counts) or .to_json()
    try:
        if hasattr(value, "histogram"):
            h = getattr(value, "histogram", None)
            if isinstance(h, (list, tuple)):
                return float(sum(abs(float(v)) for v in h))
        if hasattr(value, "to_json"):
            j = value.to_json()
            h = j.get("histogram") if isinstance(j, dict) else None
            if isinstance(h, (list, tuple)):
                return float(sum(abs(float(v)) for v in h))
    except Exception:
        pass
    # last resort: try float cast
    try:
        return float(value)
    except Exception:
        return None

def f2(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.2f}"

def f1(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.1f}"

def intc(x: Optional[int]) -> str:
    return "NA" if x is None else f"{int(x):,}"

def arr_pct_str(counts: Optional[List[int]], total: Optional[int], take: int) -> str:
    if not counts or total in (None,0) or len(counts) < take: return "NA"
    return " / ".join([f"{(100.0*c/total):.1f}" for c in counts[:take]])

def split_prefix(split: str) -> str:
    assert split in {"val","test"}
    return f"{split}_sampling"

# -------- constraint detection & captions --------
def detect_constraint_kind(metrics: Dict[str, Any], split: str, cfg=None) -> str:
    p = split_prefix(split)
    # Check for enforced constraints (satisfaction metrics)
    if f"{p}/ring_count_satisfaction" in metrics: return "ring_count"
    if f"{p}/ring_length_satisfaction" in metrics: return "ring_length"
    # Check configuration for planarity constraint
    if cfg is not None and hasattr(cfg.model, 'rev_proj') and cfg.model.rev_proj == "planar":
        return "planarity"
    # For no-constraint trainings, even if planarity is high, it's not enforced
    return "none"

def constraint_caption(kind: str, meta: Dict[str, Any]) -> str:
    if kind == "ring_count":  return f"Cycle rank ≤ {meta.get('max_rings','k')}"
    if kind == "ring_length": return f"Cycle length ≤ {meta.get('max_ring_length','L')}"
    if kind == "planarity":   return "Planar molecules only"
    return "No structural constraint"

# -------- Core (main) --------
def collect_core(split: str, metrics: Dict[str, Any], N_total: Optional[int], cfg=None) -> List[Tuple[str,str]]:
    p = split_prefix(split)
    rows = []
    rows.append(("Molecules generated (N)", intc(N_total)))
    rows.append(("FCD",                 f3(metrics.get(f"{p}/fcd score"))))
    rows.append(("Unique (%)",          pct(metrics.get(f"{p}/Uniqueness"))))
    rows.append(("Novel (%)",           pct(metrics.get(f"{p}/Novelty"))))
    rows.append(("Valid (%)",           pct(metrics.get(f"{p}/Validity"))))
    rows.append(("Disconnected (%)",    pct(metrics.get(f"{p}/Disconnected"))))
    # Property satisfied (%)
    kind = detect_constraint_kind(metrics, split, cfg)
    prop = None
    if kind == "planarity":   prop = metrics.get(f"{p}/planarity")
    elif kind == "ring_count":   prop = metrics.get(f"{p}/ring_count_satisfaction")
    elif kind == "ring_length":  prop = metrics.get(f"{p}/ring_length_satisfaction")
    rows.append(("Property satisfied (%)", "—" if kind=="none" else pct(prop)))
    # V.U.N. (%), over generated set
    u = metrics.get(f"{p}/Uniqueness"); n = metrics.get(f"{p}/Novelty"); v = metrics.get(f"{p}/Validity")
    if None not in (u,n,v):
        if u>1: u/=100.0
        if n>1: n/=100.0
        if v>1: v/=100.0
        rows.append(("V.U.N. (%)", f"{100.0*(u*n*v):.2f}%"))
    return rows

def core_definitions_md() -> str:
    return (
        "\n**Definitions**: "
        "**FCD** — Fréchet ChemNet Distance computed on valid canonical SMILES; lower is better. "
        "**Unique/Novel/Valid** — proportions computed over all generated molecules (not only valid). "
        "**Disconnected** — share of graphs with >1 connected component. "
        "**Property satisfied** — share of generated graphs meeting the enforced structural constraint. "
        "**V.U.N.** — product of Valid × Unique × Novel (in [0,100]%).\n"
    )

# -------- Structural (main) --------
def collect_structural(split: str, metrics: Dict[str, Any], N_total: Optional[int],
                       ring_counts_all: Optional[List[int]]=None,
                       ring_lengths_all: Optional[List[int]]=None,
                       max_rings: Optional[int]=None,
                       max_ring_length: Optional[int]=None,
                       cfg=None) -> List[Tuple[str,str]]:
    p = split_prefix(split)
    rows = []
    kind = detect_constraint_kind(metrics, split, cfg)

    # 1) Satisfaction row
    if kind == "planarity":
        rows.append(("Planarity satisfied (%)", pct(metrics.get(f"{p}/planarity"))))
    elif kind == "ring_count":
        rows.append(("Ring count satisfied (%)", pct(metrics.get(f"{p}/ring_count_satisfaction"))))
    elif kind == "ring_length":
        rows.append(("Ring length satisfied (%)", pct(metrics.get(f"{p}/ring_length_satisfaction"))))
    else:
        rows.append(("Constraint", "No structural constraint"))

    # 2) Distributions (per-molecule)
    # For ring count constraints: show only ring count distribution
    if kind == "ring_count" and ring_counts_all is not None:
        if max_rings is not None:
            # Constrained training: show up to max_rings, then calculate >max_rings
            for i in range(max_rings + 1):
                count = ring_counts_all[i] if i < len(ring_counts_all) else 0
                label = f"Cycle rank {i} (%)"
                pct_val = (100.0 * count / N_total) if N_total else 0.0
                rows.append((label, f"{pct_val:.1f}%"))
            # Calculate >max_rings from actual data to detect constraint violations
            sum_gt = 0
            for i in range(max_rings + 1, len(ring_counts_all)):
                sum_gt += ring_counts_all[i]
            pct_gt = (100.0 * sum_gt / N_total) if N_total else 0.0
            rows.append((f"Cycle rank >{max_rings} (%)", f"{pct_gt:.1f}%"))
        else:
            # Unconstrained training: show all natural distribution
            for i, count in enumerate(ring_counts_all):
                label = f"Cycle rank {i} (%)" if i < 9 else "Cycle rank 9+ (%)"
                pct_val = (100.0 * count / N_total) if N_total else 0.0
                rows.append((label, f"{pct_val:.1f}%"))

    # For ring length constraints: show only ring length distribution
    elif kind == "ring_length" and ring_lengths_all is not None:
        # index 0 = acyclic
        count0 = ring_lengths_all[0] if len(ring_lengths_all) > 0 else 0
        pct0 = (100.0 * count0 / N_total) if N_total else 0.0
        rows.append(("Acyclic (max len 0) (%)", f"{pct0:.1f}%"))

        # indices 1..10 => lengths 3..12 ; index 11 => >12
        # For constrained trainings, show only up to enforced value and set bigger values to 0
        # For unconstrained trainings, show all natural distribution
        if kind == "ring_length" and max_ring_length is not None:
            # Constrained training: show up to max_ring_length, then calculate >max_ring_length
            L = max_ring_length
            # 3..L
            for length in range(3, L+1):
                idx = (length - 3) + 1  # map 3→1
                count = ring_lengths_all[idx] if idx < len(ring_lengths_all) else 0
                pct_val = (100.0 * count / N_total) if N_total else 0.0
                rows.append((f"Cycle length {length} (max) (%)", f"{pct_val:.1f}%"))
            # Calculate >L from actual data to detect constraint violations
            sum_gt = 0
            for idx in range((L - 3) + 2, len(ring_lengths_all)):  # indices whose length > L
                sum_gt += ring_lengths_all[idx]
            pct_gt = (100.0 * sum_gt / N_total) if N_total else 0.0
            rows.append((f"Cycle length >{L} (max) (%)", f"{pct_gt:.1f}%"))
        else:
            # Natural distribution: show 3..12 and >12
            for length in range(3, 13):
                idx = (length - 3) + 1
                count = ring_lengths_all[idx] if idx < len(ring_lengths_all) else 0
                pct_val = (100.0 * count / N_total) if N_total else 0.0
                rows.append((f"Cycle length {length} (max) (%)", f"{pct_val:.1f}%"))
            # >12
            count_gt = ring_lengths_all[-1] if len(ring_lengths_all) > 0 else 0
            pct_gt = (100.0 * count_gt / N_total) if N_total else 0.0
            rows.append(("Cycle length >12 (max) (%)", f"{pct_gt:.1f}%"))

    # For planarity constraints: show only planarity satisfaction, no ring distributions
    elif kind == "planarity":
        # Planarity constraints don't show ring distributions
        pass

    # For no-constraint trainings: show both distributions (natural)
    elif kind == "none":
        # Show ring count distribution
        if ring_counts_all is not None:
            for i, count in enumerate(ring_counts_all):
                label = f"Cycle rank {i} (%)" if i < 9 else "Cycle rank 9+ (%)"
                pct_val = (100.0 * count / N_total) if N_total else 0.0
                rows.append((label, f"{pct_val:.1f}%"))
        
        # Show ring length distribution
        if ring_lengths_all is not None:
            # index 0 = acyclic
            count0 = ring_lengths_all[0] if len(ring_lengths_all) > 0 else 0
            pct0 = (100.0 * count0 / N_total) if N_total else 0.0
            rows.append(("Acyclic (max len 0) (%)", f"{pct0:.1f}%"))

            # Natural distribution: show 3..12 and >12
            for length in range(3, 13):
                idx = (length - 3) + 1
                count = ring_lengths_all[idx] if idx < len(ring_lengths_all) else 0
                pct_val = (100.0 * count / N_total) if N_total else 0.0
                rows.append((f"Cycle length {length} (max) (%)", f"{pct_val:.1f}%"))
            # >12
            count_gt = ring_lengths_all[-1] if len(ring_lengths_all) > 0 else 0
            pct_gt = (100.0 * count_gt / N_total) if N_total else 0.0
            rows.append(("Cycle length >12 (max) (%)", f"{pct_gt:.1f}%"))

    return rows

# -------- Alignment (appendix) --------
def collect_alignment(split: str, metrics: Dict[str, Any]) -> List[Tuple[str,str]]:
    p = split_prefix(split)
    rows = []
    rows.append(("NumNodes W1",   f3(to_scalar(metrics.get(f"{p}/NumNodesW1")))))
    rows.append(("NodeTypes TV",  f3(to_scalar(metrics.get(f"{p}/NodeTypesTV")))))
    rows.append(("EdgeTypes TV",  f3(to_scalar(metrics.get(f"{p}/EdgeTypesTV")))))
    # If available, reduce a histogram (abs diff per degree) to a single L1 sum
    if f"{p}/abs_diff_deg_hist" in metrics:
        deg_l1 = to_scalar(metrics.get(f"{p}/abs_diff_deg_hist"))
        rows.append(("Degree hist |Δ| (L1 sum)", f3(deg_l1)))
    return rows

# -------- Chemistry (appendix; from valid SMILES, optional) --------
def chemistry_from_smiles(smiles: List[str]) -> Dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return {}
    mols = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is not None: mols.append(m)
        except Exception:
            pass
    if not mols: return {}
    mw = [Descriptors.MolWt(m) for m in mols]
    heavy = [m.GetNumHeavyAtoms() for m in mols]
    comps = {"C":0,"N":0,"O":0,"F":0}
    total_atoms = 0
    for m in mols:
        for a in m.GetAtoms():
            sym = a.GetSymbol()
            if sym in comps: comps[sym]+=1
            total_atoms += 1
    comp_pct = {k: (100.0*v/total_atoms) if total_atoms>0 else 0.0 for k,v in comps.items()}
    def meanstd(x):
        return (statistics.mean(x), statistics.pstdev(x) if len(x)>1 else 0.0)
    mw_mu, mw_sd = meanstd(mw)
    hv_mu, hv_sd = meanstd(heavy)
    return {
        "N_valid_for_chem": len(mols),
        "MW_mean": mw_mu, "MW_sd": mw_sd,
        "Heavy_mean": hv_mu, "Heavy_sd": hv_sd,
        "Comp_C": comp_pct["C"], "Comp_N": comp_pct["N"], "Comp_O": comp_pct["O"], "Comp_F": comp_pct["F"],
    }

def collect_chemistry(chem: Dict[str, Any]) -> List[Tuple[str,str]]:
    if not chem: return []
    return [
        ("Valid molecules used (chem)", intc(chem.get("N_valid_for_chem"))),
        ("Molecular weight (Da) mean ± sd", f"{f1(chem.get('MW_mean'))} ± {f1(chem.get('MW_sd'))}"),
        ("Heavy atoms mean ± sd",            f"{f1(chem.get('Heavy_mean'))} ± {f1(chem.get('Heavy_sd'))}"),
        ("Atom composition % (C/N/O/F)",     f"{f1(chem.get('Comp_C'))}/{f1(chem.get('Comp_N'))}/{f1(chem.get('Comp_O'))}/{f1(chem.get('Comp_F'))}"),
    ]

# -------- Timing (appendix) --------
# Align with ConStruct: report sampling time for N graphs and per-graph time (ms).
def collect_timing(t: Dict[str, Any], N_total: Optional[int], device_meta: Dict[str, str]) -> List[Tuple[str,str]]:
    rows = []
    samp_s = t.get("sampling_sec")         # total sampling seconds for this split
    rows.append(("Sampling time (s) for N", f"{f2(samp_s)} for N={intc(N_total)}"))
    per_ms = None
    if samp_s is not None and N_total not in (None, 0):
        per_ms = 1000.0 * float(samp_s) / int(N_total)
    rows.append(("Per-graph sampling time (ms)", f1(per_ms)))
    rows.append(("Sampling batches",            f2(t.get("sampling_batches")) if t.get("sampling_batches") is not None else "NA"))
    
    # Handle projection metrics with better defaults
    proj_ms_mean = t.get("proj_ms_mean")
    if proj_ms_mean is not None and proj_ms_mean == 0.0:
        rows.append(("Projector mean time (ms)", "0.0 (no projection)"))
    else:
        rows.append(("Projector mean time (ms)", f1(proj_ms_mean)))
    
    proj_share_pct = t.get("proj_share_pct")
    if proj_share_pct is not None and proj_share_pct == 0.0:
        rows.append(("Projector share of sampling (%)", "0.0% (no projection)"))
    else:
        rows.append(("Projector share of sampling (%)", f1(proj_share_pct)))
    
    rows.append(("Wall-clock (h:mm)",           t.get("wall_clock_hhmm","NA")))
    # Device/context for reproducibility
    dev = device_meta or {}
    ctx = ", ".join([f"{k}={v}" for k,v in dev.items() if v])
    rows.append(("Environment", ctx if ctx else "NA"))
    return rows

# -------- Renderers (titles self-contained) --------
def render_table_md(title: str, subtitle: str, rows: List[Tuple[str,str]], tail_note: Optional[str]=None) -> str:
    s = [f"### {title}", f"{subtitle}", "", "| Metric | Value |", "|---|---:|"]
    for k,v in rows: s.append(f"| {k} | {v} |")
    if tail_note: s.append(tail_note)
    s.append("")  # newline
    return "\n".join(s)

def render_table_console(title: str, subtitle: str, rows: List[Tuple[str,str]], tail_note: Optional[str]=None) -> str:
    """Render a nicely formatted table for console output with proper alignment."""
    if not rows:
        return f"{title}\n{subtitle}\n(No data available)"
    
    # Find the maximum width for the metric column
    max_metric_width = max(len(str(k)) for k, _ in rows)
    
    # Set a reasonable maximum width for values to prevent very wide tables
    max_value_width = max(len(str(v)) for _, v in rows)
    value_width = min(50, max(20, max_value_width + 2))  # Between 20 and 50 characters
    
    # Create the table
    lines = []
    lines.append(title)
    lines.append(subtitle)
    lines.append("")
    
    # Create separator line
    separator = "+" + "-" * (max_metric_width + 2) + "+" + "-" * value_width + "+"
    lines.append(separator)
    
    # Header
    header = f"| {('Metric').ljust(max_metric_width)} | {'Value'.ljust(value_width)} |"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    for metric, value in rows:
        metric_str = str(metric).ljust(max_metric_width)
        value_str = str(value)
        
        # Truncate long values and add ellipsis
        if len(value_str) > value_width - 1:
            value_str = value_str[:value_width-4] + "..."
        
        value_str = value_str.ljust(value_width)
        row = f"| {metric_str} | {value_str} |"
        lines.append(row)
    
    lines.append(separator)
    
    # Add tail note if provided
    if tail_note:
        lines.append("")
        lines.append(tail_note)
    
    lines.append("")  # final newline
    return "\n".join(lines)

def write_tables(outdir: str, split: str, exp_name: str, dataset: str, constraint_str: str,
                 N_total: Optional[int], diffusion_steps: Optional[int],
                 eval_meta: Dict[str, Any],
                 core: List[Tuple[str,str]],
                 structural: List[Tuple[str,str]],
                 alignment: List[Tuple[str,str]],
                 timing: List[Tuple[str,str]],
                 include_alignment: bool = True,
                 include_timing: bool = True):
    os.makedirs(outdir, exist_ok=True)
    hdr = f"{exp_name} — {split.upper()} — {dataset}"
    sub = f"Constraint: {constraint_str} · N={intc(N_total)} · d={intc(diffusion_steps)} · sampler={eval_meta.get('sampler','reverse')}"
    # Structural subtitle explains the calculation:
    sub_struct = (
        sub
        + " · Structural cycles via NetworkX cycle_basis; "
          "ring-length rows = per-molecule MAX basis-cycle length; "
          "aromatic bonds included"
    )
    with open(os.path.join(outdir, f"{split}_core.md"), "w") as f:
        f.write(render_table_md(hdr, sub, core, tail_note=core_definitions_md()))
    with open(os.path.join(outdir, f"{split}_structural.md"), "w") as f:
        f.write(render_table_md(hdr+" — Structural", sub_struct, structural))
    if include_alignment and alignment:
        with open(os.path.join(outdir, f"{split}_alignment.md"), "w") as f:
            f.write(render_table_md(hdr+" — Alignment (Appendix)", sub, alignment))
    if include_timing and timing:
        with open(os.path.join(outdir, f"{split}_timing.md"), "w") as f:
            f.write(render_table_md(hdr+" — Timing (Appendix)", sub, timing)) 