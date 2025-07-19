# Ring Count Violations

This directory contains molecular graphs that violate the constraint criteria.

## Directory Structure

- `train_violations/` - Violations from training set
- `val_violations/` - Violations from validation set  
- `test_violations/` - Violations from test set

Each violation is organized by the type and severity of the violation.

## Files

Each violation folder contains:
- `graph_0.png` - Visualization of the molecular graph
- Additional metadata about the violation

## Analysis

These violations were identified by running the constraint analysis scripts:
- Ring length violations: `check_dataset_ring_length.py`
- Ring count violations: `check_dataset_ring_count.py`

Generated on: Thu 17 Jul 2025 09:26:50 PM CEST
