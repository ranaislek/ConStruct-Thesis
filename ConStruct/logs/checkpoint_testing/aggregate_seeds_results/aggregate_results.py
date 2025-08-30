import re
import numpy as np
import glob
import os

# Regex to capture metrics from your .out files (fallback, non-table style)
patterns = {
    "FCD": r"\bFCD\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Unique": r"\bUnique\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Novel": r"\bNovel\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Valid": r"\bValid\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "Disconnected": r"\bDisconnected\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    "VUN": r"\bV\.U\.N\.\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
    # Satisfied can appear with different prefixes (Property/Planarity/Ring count/Cycle rank)
    "Satisfied": r"\b(?:Property|Planarity|Ring count|Cycle rank) satisfied\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?",
}

# Labels as shown in the table-style summaries
TABLE_LABELS = {
    "FCD": r"^\|\s*FCD\s*\|",
    "Unique": r"^\|\s*Unique\s*\(\%\)\s*\|",
    "Novel": r"^\|\s*Novel\s*\(\%\)\s*\|",
    "Valid": r"^\|\s*Valid\s*\(\%\)\s*\|",
    "Disconnected": r"^\|\s*Disconnected\s*\(\%\)\s*\|",
    "VUN": r"^\|\s*V\.U\.N\.\s*\(\%\)\s*\|",
    # Satisfied row in top summary table (generic)
    "Satisfied": r"^\|\s*(?:Property|Planarity|Ring count|Cycle rank) satisfied\s*\(\%\)\s*\|",
}

# New patterns for structural constraint satisfaction
STRUCTURAL_PATTERNS = {
    "Ring_length_satisfied": r"^\|\s*Ring length satisfied\s*\(\%\)\s*\|",
    "Acyclic": r"^\|\s*Acyclic\s*\(max len 0\)\s*\(\%\)\s*\|",
    "Cycle_length_3": r"^\|\s*Cycle length 3\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_4": r"^\|\s*Cycle length 4\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_5": r"^\|\s*Cycle length 5\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_6": r"^\|\s*Cycle length 6\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_7": r"^\|\s*Cycle length 7\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_8": r"^\|\s*Cycle length 8\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_gt3": r"^\|\s*Cycle length >3\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_gt4": r"^\|\s*Cycle length >4\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_gt5": r"^\|\s*Cycle length >5\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_gt6": r"^\|\s*Cycle length >6\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_gt7": r"^\|\s*Cycle length >7\s*\(max\)\s*\(\%\)\s*\|",
    "Cycle_length_gt8": r"^\|\s*Cycle length >8\s*\(max\)\s*\(\%\)\s*\|",
}

def _parse_table_style(text):
    results = {}
    lines = text.splitlines()
    for line in lines:
        for key, label_regex in TABLE_LABELS.items():
            if re.search(label_regex, line):
                # Expect a markdown-like row: | Label | Value |
                # Split on '|' and take the value column (index 2 when trimmed)
                parts = [p.strip() for p in line.split('|')]
                # After split of "| A | B |" we get: ['', ' A ', ' B ', '']
                if len(parts) >= 3:
                    value_str = parts[2]
                    # Remove trailing percentage sign if present
                    value_str = value_str.rstrip('%').strip()
                    # Extract the first number
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value_str)
                    if m:
                        try:
                            if key not in results:
                                results[key] = []
                            results[key].append(float(m.group(1)))
                        except ValueError:
                            pass
    return results

def _parse_structural_constraints(text):
    """Parse structural constraint satisfaction metrics"""
    results = {}
    lines = text.splitlines()
    
    in_structural_section = False
    in_table = False
    
    for line in lines:
        if "🏗️ STRUCTURAL CONSTRAINT SATISFACTION:" in line:
            in_structural_section = True
            continue
            
        if in_structural_section and "| Metric" in line:
            in_table = True
            continue
            
        if in_table and line.strip().startswith("+---"):
            continue
            
        if in_table and line.strip() == "":
            in_table = False
            in_structural_section = False
            break
            
        if in_table and "|" in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                metric_name = parts[1].strip()
                value_str = parts[2].strip()
                
                # Extract percentage value
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value_str)
                if m:
                    try:
                        value = float(m.group(1))
                        
                        # Map metric names to our standardized keys
                        if "Ring length satisfied" in metric_name:
                            key = "Ring_length_satisfied"
                        elif "Acyclic" in metric_name:
                            key = "Acyclic"
                        elif "Cycle length 3" in metric_name:
                            key = "Cycle_length_3"
                        elif "Cycle length 4" in metric_name:
                            key = "Cycle_length_4"
                        elif "Cycle length 5" in metric_name:
                            key = "Cycle_length_5"
                        elif "Cycle length 6" in metric_name:
                            key = "Cycle_length_6"
                        elif "Cycle length 7" in metric_name:
                            key = "Cycle_length_7"
                        elif "Cycle length 8" in metric_name:
                            key = "Cycle_length_8"
                        elif "Cycle length >3" in metric_name:
                            key = "Cycle_length_gt3"
                        elif "Cycle length >4" in metric_name:
                            key = "Cycle_length_gt4"
                        elif "Cycle length >5" in metric_name:
                            key = "Cycle_length_gt5"
                        elif "Cycle length >6" in metric_name:
                            key = "Cycle_length_gt6"
                        elif "Cycle length >7" in metric_name:
                            key = "Cycle_length_gt7"
                        elif "Cycle length >8" in metric_name:
                            key = "Cycle_length_gt8"
                        else:
                            continue
                            
                        if key not in results:
                            results[key] = []
                        results[key].append(value)
                    except ValueError:
                        pass
    
    return results

def _parse_fallback_regex(text):
    results = {}
    for key, pat in patterns.items():
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            try:
                if key not in results:
                    results[key] = []
                results[key].append(float(match.group(1)))
            except ValueError:
                pass
    return results

def parse_file(filename):
    with open(filename, "r") as f:
        text = f.read()
    
    # Parse molecular metrics
    results = _parse_table_style(text)
    fallback = _parse_fallback_regex(text)
    for k, v in fallback.items():
        results.setdefault(k, v)
    
    # Parse structural constraints
    structural_results = _parse_structural_constraints(text)
    results.update(structural_results)
    
    return results

def aggregate(files):
    # Initialize with all possible keys
    all_keys = list(patterns.keys()) + list(STRUCTURAL_PATTERNS.keys())
    data = {key: [] for key in all_keys}
    
    for f in files:
        res = parse_file(f)
        for k, v in res.items():
            if k in data:
                # v is now a list of values from the file
                if isinstance(v, list):
                    data[k].extend(v)
                else:
                    data[k].append(v)
    
    summary = {}
    for k, vals in data.items():
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            summary[k] = f"{mean} ± {std}"
    
    return summary

def generate_report() -> str:
    lines = []
    
    # Process all constraint types
    constraint_types = [
        "no_constraint",
        "planar",
        "ring_count_0",
        "ring_count_1",
        "ring_count_2",
        "ring_count_3",
        "ring_count_4",
        "ring_count_5",
        "ring_length_3",
        "ring_length_4",
        "ring_length_5",
        "ring_length_6",
        "ring_length_7",
        "ring_length_8",
    ]
    
    for constraint in constraint_types:
        files = glob.glob(f"test_{constraint}_*.out")
        if not files:
            continue
        
        lines.append("")
        lines.append(f"=== {constraint} ===")
        summary = aggregate(files)
        lines.append(str(summary))
        
        # For ring length constraints, show distribution that should sum to 100%
        if constraint.startswith("ring_length_"):
            lines.append("Ring Length Distribution (should sum to 100%):")
            acyclic = summary.get("Acyclic", "0 ± 0")
            lines.append(f"  Acyclic: {acyclic}")
            
            # Show cycle lengths that are present
            for length in [3, 4, 5, 6, 7, 8]:
                key = f"Cycle_length_{length}"
                if key in summary:
                    lines.append(f"  Cycle length {length}: {summary[key]}")
            
            # Show violations
            for gt_length in [3, 4, 5, 6, 7, 8]:
                key = f"Cycle_length_gt{gt_length}"
                if key in summary:
                    lines.append(f"  Cycle length >{gt_length}: {summary[key]}")
    
    return "\n".join(lines).rstrip() + "\n"

if __name__ == "__main__":
    report = generate_report()
    print(report)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "aggregate_summary.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Saved aggregate summary to: {out_path}")
