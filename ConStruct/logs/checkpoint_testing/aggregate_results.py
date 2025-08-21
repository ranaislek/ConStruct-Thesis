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
    # Prefer table-style metrics
    results = _parse_table_style(text)
    # Fill any missing with fallback patterns
    fallback = _parse_fallback_regex(text)
    for k, v in fallback.items():
        results.setdefault(k, v)
    return results


def aggregate(files):
    data = {key: [] for key in patterns.keys()}
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
            summary[k] = f"{mean:.2f} ± {std:.2f}"
    return summary


def generate_report() -> str:
    lines = []
    # Example: all ring_count_0 runs
    files = glob.glob("test_ring_count_0_*.out")
    lines.append("Ring Count ≤ 0:")
    lines.append(str(aggregate(files)))

    # You can repeat for different constraints
    for constraint in [
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
    ]:
        files = glob.glob(f"test_{constraint}_*.out")
        if not files:
            continue
        lines.append("")
        lines.append(f"=== {constraint} ===")
        lines.append(str(aggregate(files)))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    report = generate_report()
    print(report)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "aggregate_summary.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Saved aggregate summary to: {out_path}")
