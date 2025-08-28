#!/usr/bin/env python3
"""
Shared utilities for structural ring (cycle) detection on undirected molecular graphs.
- enumerate_simple_cycles_unique: unique simple cycles (no orientation duplicates)
- count_simple_cycles: total number of unique simple cycles
- max_simple_cycle_length: maximum simple cycle length, or 0 if acyclic

Note: Structural only; no chemical validity checks.
"""
from __future__ import annotations
from typing import Iterable, List, Set, Tuple, Optional
import networkx as nx

__all__ = [
    "enumerate_simple_cycles_unique",
    "count_simple_cycles",
    "max_simple_cycle_length",
]


def _canonical_cycle(cycle: List[int]) -> Tuple[int, ...]:
    """Canonicalize a cycle for undirected graphs.
    - Rotate both orientations so the smallest node is first
    - Choose the lexicographically smaller orientation
    """
    if not cycle:
        return tuple()
    # Remove duplicated closing node if present
    if len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    if not cycle:
        return tuple()
    m = len(cycle)
    # Forward orientation rotated to put smallest node first
    min_node = min(cycle)
    f_idx = cycle.index(min_node)
    forward = cycle[f_idx:] + cycle[:f_idx]
    # Reverse orientation also rotated to put smallest node first
    rc = list(reversed(cycle))
    r_idx = rc.index(min_node)
    backward = rc[r_idx:] + rc[:r_idx]
    return tuple(forward) if forward <= backward else tuple(backward)


def enumerate_simple_cycles_unique(graph: nx.Graph, max_len: Optional[int] = None) -> Iterable[List[int]]:
    """Enumerate unique simple cycles for an undirected graph.
    Uses networkx.simple_cycles on a directed view and deduplicates by canonical form.
    max_len: if provided, skip cycles longer than this threshold.
    """
    if graph.number_of_edges() == 0 or graph.number_of_nodes() < 3:
        return []
    # Use directed view for simple_cycles
    directed = graph.to_directed()
    raw_cycles = list(nx.simple_cycles(directed))

    seen: Set[Tuple[int, ...]] = set()
    unique_cycles: List[List[int]] = []
    for cyc in raw_cycles:
        if max_len is not None and len(cyc) > max_len:
            continue
        can = _canonical_cycle(cyc)
        if len(can) >= 3 and can not in seen:
            seen.add(can)
            unique_cycles.append(list(can))
    return unique_cycles


def count_simple_cycles(graph: nx.Graph, max_len: Optional[int] = None) -> int:
    return len(list(enumerate_simple_cycles_unique(graph, max_len=max_len)))


def max_simple_cycle_length(graph: nx.Graph, max_len: Optional[int] = None) -> int:
    cycles = enumerate_simple_cycles_unique(graph, max_len=max_len)
    max_len_found = 0
    for cyc in cycles:
        if len(cyc) > max_len_found:
            max_len_found = len(cyc)
    return max_len_found 