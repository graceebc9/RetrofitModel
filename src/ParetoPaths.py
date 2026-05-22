"""
src/ParetoPaths.py
==================

Single source of truth for every path in the pareto pipeline.

Layout produced:
    <root>/
    ├── manifest.json
    ├── data/<bucket>/<budget>_<loft>/
    │       summary.csv
    │       baseline.csv
    │       baseline_summary.csv
    │       epc_summary.csv          (EPC mode only)
    │       selected/
    │           eq{N}.csv
    │           epc.csv              (EPC mode only)
    ├── views/
    │   ├── per_scenario_per_budget/<bucket>/<budget>_<loft>/*.png
    │   ├── per_scenario_across_budgets/<bucket>/loft_<loft>/*.png
    │   ├── across_scenarios_per_budget/<budget>_<loft>/
    │   │       stability.csv
    │   │       envelope.csv
    │   │       pareto_overlay.png
    │   └── opt_vs_epc/<bucket>/<budget>_<loft>_eq{N}/*.png
    └── logs/<bucket>/<budget>_<loft>/
            summary_<timestamp>.log
            detail_<timestamp>.log

Every reader and writer in the pipeline goes through this module.
Hand-rolling os.path.join elsewhere is a bug.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass


MILLION = 1_000_000
SCHEMA_VERSION = 2  # bump on layout changes; manifest records this


# ============================================================================
# Slug helpers — canonical string forms for the path components
# ============================================================================

def budget_slug(budget: float) -> str:
    """£1M -> '1M', £2.5M -> '2.5M', £500k -> '0.5M'."""
    return f"{budget / MILLION:g}M"


def loft_slug(loft: float) -> str:
    """0.65 -> 'loft0.65'. Loft is a probability so :g is fine."""
    return f"loft{loft:g}"


def eq_slug(equity_floor: float) -> str:
    """0 -> 'eq0', 25 -> 'eq25'. Always integer-formatted."""
    return f"eq{int(equity_floor)}"


def slice_slug(budget: float, loft: float) -> str:
    """The (budget, loft) slice that most artefacts are keyed by."""
    return f"{budget_slug(budget)}_{loft_slug(loft)}"


# ============================================================================
# Inputs — where the upstream optimiser dumps its data
# ============================================================================

LOGS_SUBDIR = 'split_scenarios_logs'
MEANS_SUBDIR = 'split_scenarios_means'
PER_RUN_MEANS_PATTERN = 'per_run_means_*.parquet'


def input_csv_glob(input_base_dir: str, bucket: str) -> str:
    """CSV glob for one cost-scenario bucket (interventions logs)."""
    return os.path.join(input_base_dir, LOGS_SUBDIR, bucket, '*')


def per_run_parquet_glob(input_base_dir: str, bucket: str) -> str:
    """Per-run parquet glob for one cost-scenario bucket."""
    return os.path.join(
        input_base_dir, MEANS_SUBDIR, bucket, PER_RUN_MEANS_PATTERN,
    )


# ============================================================================
# Top-level run folders
# ============================================================================

def manifest_path(root: str) -> str:
    return os.path.join(root, 'manifest.json')


def data_root(root: str) -> str:
    return os.path.join(root, 'data')


def views_root(root: str) -> str:
    return os.path.join(root, 'views')


def logs_root(root: str) -> str:
    return os.path.join(root, 'logs')


# ============================================================================
# Data layer — per-bucket, per-(budget, loft) raw outputs
# ============================================================================

def bucket_data_dir(root: str, bucket: str) -> str:
    return os.path.join(data_root(root), bucket)


def slice_data_dir(root: str, bucket: str, budget: float, loft: float) -> str:
    """The raw-output folder for one (bucket, budget, loft) solve."""
    return os.path.join(bucket_data_dir(root, bucket), slice_slug(budget, loft))


def summary_csv(root: str, bucket: str, budget: float, loft: float) -> str:
    """Pareto sweep summary — one row per equity floor."""
    return os.path.join(slice_data_dir(root, bucket, budget, loft), 'summary.csv')


def baseline_csv(root: str, bucket: str, budget: float, loft: float) -> str:
    """Per-pair baseline (preselect-best-cpt) selection."""
    return os.path.join(slice_data_dir(root, bucket, budget, loft), 'baseline.csv')


def baseline_summary_csv(
    root: str, bucket: str, budget: float, loft: float,
) -> str:
    """One-row baseline stats — same schema family as summary.csv rows."""
    return os.path.join(
        slice_data_dir(root, bucket, budget, loft), 'baseline_summary.csv',
    )


def epc_summary_csv(root: str, bucket: str, budget: float, loft: float) -> str:
    """One-row EPC fallback stats. Only present in EPC mode."""
    return os.path.join(
        slice_data_dir(root, bucket, budget, loft), 'epc_summary.csv',
    )


def selected_dir(root: str, bucket: str, budget: float, loft: float) -> str:
    """Folder holding one CSV per equity floor (and EPC selection)."""
    return os.path.join(slice_data_dir(root, bucket, budget, loft), 'selected')


def selected_csv(
    root: str, bucket: str, budget: float, loft: float, equity_floor: float,
) -> str:
    return os.path.join(
        selected_dir(root, bucket, budget, loft), f'{eq_slug(equity_floor)}.csv',
    )


def epc_selected_csv(
    root: str, bucket: str, budget: float, loft: float,
) -> str:
    """EPC random-targeted selection rows. Only present in EPC mode."""
    return os.path.join(selected_dir(root, bucket, budget, loft), 'epc.csv')


# ============================================================================
# Views layer — finished plots and comparison artefacts
# ============================================================================

def per_scenario_per_budget_dir(
    root: str, bucket: str, budget: float, loft: float,
) -> str:
    """Stage 1 — per-bucket per-budget plots from run_pareto."""
    return os.path.join(
        views_root(root),
        'per_scenario_per_budget',
        bucket,
        slice_slug(budget, loft),
    )


def per_scenario_across_budgets_dir(
    root: str, bucket: str, loft: float,
) -> str:
    """Stage 3 — per-bucket overlays across all budgets at one loft."""
    return os.path.join(
        views_root(root),
        'per_scenario_across_budgets',
        bucket,
        loft_slug(loft),
    )


def across_scenarios_per_budget_dir(
    root: str, budget: float, loft: float,
) -> str:
    """Stage 2 — cross-bucket comparison at one (budget, loft) slice."""
    return os.path.join(
        views_root(root),
        'across_scenarios_per_budget',
        slice_slug(budget, loft),
    )


def opt_vs_epc_dir(
    root: str, bucket: str, budget: float, loft: float, equity_floor: float,
) -> str:
    """Stage 4 — Opt.T vs EPC comparison at one full grid point."""
    return os.path.join(
        views_root(root),
        'opt_vs_epc',
        bucket,
        f'{slice_slug(budget, loft)}_{eq_slug(equity_floor)}',
    )


# ============================================================================
# Logs layer — per-slice debug logs, kept out of data/ and views/
# ============================================================================

def slice_log_dir(root: str, bucket: str, budget: float, loft: float) -> str:
    return os.path.join(
        logs_root(root), bucket, slice_slug(budget, loft),
    )


# ============================================================================
# Required-artefact lists — used by completeness checks
# ============================================================================

# Files a non-EPC slice must produce to count as complete.
SLICE_REQUIRED_FILES = (
    'summary.csv',
    'baseline.csv',
    'baseline_summary.csv',
)

# Extra files an EPC slice must additionally produce.
SLICE_EPC_EXTRA_FILES = (
    'epc_summary.csv',
)


def required_slice_paths(
    root: str, bucket: str, budget: float, loft: float, *, epc: bool,
) -> list[str]:
    """Absolute paths of files a complete slice must contain."""
    base = slice_data_dir(root, bucket, budget, loft)
    files = list(SLICE_REQUIRED_FILES)
    if epc:
        files.extend(SLICE_EPC_EXTRA_FILES)
    return [os.path.join(base, f) for f in files]


# ============================================================================
# Discovery helpers — for downstream readers that walk the tree
# ============================================================================

def existing_buckets(root: str) -> list[str]:
    """Buckets that have any data on disk under <root>/data/."""
    d = data_root(root)
    if not os.path.isdir(d):
        return []
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
    )


def existing_slices(root: str, bucket: str) -> list[str]:
    """Slice slugs (e.g. '25M_loft0.65') that exist under one bucket."""
    d = bucket_data_dir(root, bucket)
    if not os.path.isdir(d):
        return []
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
    )


# ============================================================================
# Parsed-slice convenience type
# ============================================================================

@dataclass(frozen=True)
class SliceKey:
    """A (bucket, budget, loft) tuple; the natural unit of work."""
    bucket: str
    budget: float
    loft: float

    @property
    def slug(self) -> str:
        return slice_slug(self.budget, self.loft)