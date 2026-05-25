"""
Pareto Knapsack Retrofit Analysis — per-cost_scenario runs with comparison.

Changes vs. previous version:
  - Runs the full Pareto sweep independently for each cost_scenario bucket
    (optimistic / central / pessimistic) produced by the upstream
    SPLIT_BY_COST_SCENARIO=1 preprocessing.
  - Output layout grows a `cost_scenario={bucket}/` level under the
    pareto_runs folder. Each bucket's run is identical in structure to the
    previous single-output run.
  - New `comparison/` directory at the top level, holding three artefacts
    per (budget, loft_prob) combination:
        * stability.csv     : pairwise Jaccard on UPN sets and on
                              (UPN, intervention) tuples across buckets.
        * envelope.csv      : range of total_cost / total_abatement /
                              cpex_per_ton across the 3 buckets per
                              equity_floor.
        * pareto_overlay.png: 3 fronts on one axis with their own bands.
  - Per-run epistemic propagation now reads the bucket's own parquets so
    epistemic_std within a bucket reflects the other 6 factors only.
    Cost_scenario uncertainty is captured by the cross-bucket envelope.
  - Baselines are per-bucket (each bucket has its own preselect baseline,
    since the selection score column shifts with cost assumptions).

Optimiser objective unchanged: knapsack runs on the means
(`mean_total_capex`, `mean_total_co2_saved`).
"""

from __future__ import annotations

import os
import sys
import glob
import gc
import json
import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.personas import load_personas
from src.utils import is_running_on_hpc
from src.PartetoEpc import select_epc_algo_pareto
from src.GreedyEpcVis import run_epc_vis
from src.PostPareto import post_proc_pareto
from src.ParetoUtills import *  # noqa: F401,F403
from src.ParetoKnapsack import (
    multichoice_knapsack,
    preselect_best_cpt,
    DEFAULT_HIGH_EQUITY_PERSONAS,
)

# ============================================================================
# CONSTANTS
# ============================================================================

MILLION_FACTOR = 1_000_000
# RHO = 0.45  #

# Selection-score column name from the new preprocessing.
SELECTION_SCORE_COL = 'capex_per_net_ton_aleatoric_sigma'

# Cost / carbon column names used by the optimiser objective.
COST_COL = 'mean_total_capex'
CARBON_COL = 'mean_total_co2_saved'

# Aleatoric-std columns used for portfolio-level closed-form aleatoric std.
COST_ALEATORIC_STD_COL = 'aleatoric_std_total_capex'
CARBON_ALEATORIC_STD_COL = 'aleatoric_std_total_co2_saved'

# New layout: under <root>/{epc,non_epc}/ live two sibling trees, one for the
# CSV interventions logs and one for the per-run means parquets. Each tree is
# then split by cost_scenario bucket.
LOGS_SUBDIR = 'split_scenarios_logs'
MEANS_SUBDIR = 'split_scenarios_means'
PER_RUN_MEANS_PATTERN = 'per_run_means_*.parquet'

# Cost scenario buckets produced by the upstream SPLIT_BY_COST_SCENARIO=1.
DEFAULT_COST_SCENARIOS = [ 'pessimistic', 'central', 'optimistic'  ]

# Colours for cross-bucket overlay plots.
BUCKET_COLOURS = {
    'optimistic':  '#2e7d32',
    'central':     '#1976d2',
    'pessimistic': '#c62828',
}


class EPCSelectionError(RuntimeError):
    """Raised when the EPC random-selection fallback produces no results."""


# ============================================================================
# BUDGET NAME FORMATTING — single source of truth
# ============================================================================

def budget_label(budget: int) -> str:
    """Canonical folder-name label for a budget in pounds."""
    return f"{budget / MILLION_FACTOR:g}"


def budgets_tag(budgets: list[int]) -> str:
    """Tag used for grouping visualisations across budgets."""
    return '_'.join(budget_label(b) for b in budgets) + 'M'


# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class RunConfig:
    setting_name: str
    input_base_dir: str
    base_dir: str
    mip_gap: float
    budgets: list[int]
    loft_probs: list[float]
    equity_floors: list[int]
    epc_run: bool
    run_greedy_runs: bool
    force_rerun: bool
    test_mode: bool
    test_sample_size: int
    test_min_per_stratum: int
    test_seed: int
    cost_scenarios: list[str] = field(
        default_factory=lambda: list(DEFAULT_COST_SCENARIOS)
    )
    verbose: bool = True

    @property
    def root(self) -> str:
        """Top of the run folder. Replaces pareto_runs_folder."""
        parts = [self.base_dir]
        if self.epc_run:
            parts.append('pareto_runs_epc' if not self.test_mode
                         else 'pareto_runs')
        else:
            parts.append('pareto_runs')
        parts.append(self.setting_name + ('_epc' if self.epc_run and self.test_mode else ''))
        if self.test_mode:
            parts.append(f'samples_{self.test_sample_size}')
        return os.path.join(*parts)

# Path table now points at the *parent* of the logs/ and means/ trees.
# Layout under each entry:
#   <input_base_dir>/split_scenarios_logs/<bucket>/*.csv
#   <input_base_dir>/split_scenarios_means/<bucket>/per_run_means_*.parquet
PATHS = {
    ('local', True): (
        '/Volumes/T9/2025_10_RetrofitModel/15_split_costs_opt/epc',
        '/Volumes/T9/2025_10_RetrofitModel/15_split_costs_opt/2_greedy_results/NE/all_domestic',
    ),
    ('local', False): (
        '/Volumes/T9/2025_10_RetrofitModel/15_split_costs_opt/non_epc',
        '/Volumes/T9/2025_10_RetrofitModel/15_split_costs_opt/2_greedy_results/NE/all_domestic',
    ),
    ('hpc', True): (
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/'
        '2_optimized_priorities_epc/risk_sigma_1.0/epc',
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v9/NE/epc',
    ),
    ('hpc', False): (
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/'
        '4_optimized_priorities/risk_sigma_1.0/non_epc',
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/5_pareto/v9/NE/all_domestic',
    ),
}

LOCAL_DEFAULTS = dict(
    budgets=[200_000_000, 1_000_000, 25_000_000, 50_000_000, 100_000_000],
    loft_probs=[0.65, 0.95],
    equity_floors=[0, 25, 50, 75,87, 100],
)

 

# LOCAL_DEFAULTS = dict(
#     budgets=[200_000_000, 1_000_000, 25_000_000, 50_000_000, 100_000_000],
#     loft_probs=[0.65, 0.95],
#     equity_floors=[0, 12, 25,37,  50, 62, 75, 87, 100],
# )

 

HPC_DEFAULTS = dict(
    budgets=[1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000],
    loft_probs=[0.95, 0.65],
    equity_floors=[0, 10, 25, 35, 50, 60, 75, 100],
)


def _parse_cost_scenarios_env() -> list[str]:
    """Comma-separated env override; falls back to default."""
    raw = os.getenv('COST_SCENARIOS', '').strip()
    if not raw:
        return list(DEFAULT_COST_SCENARIOS)
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    bad = [p for p in parts if p not in DEFAULT_COST_SCENARIOS]
    if bad:
        raise ValueError(
            f"Unknown cost_scenarios in env: {bad}. "
            f"Expected subset of {DEFAULT_COST_SCENARIOS}."
        )
    return parts


def resolve_config() -> RunConfig:
    """Resolve environment variables + host into a single config object."""
    running_locally = not is_running_on_hpc()
    env_key = 'local' if running_locally else 'hpc'
    epc_run = os.getenv('EPC_YN', 'N').upper() == 'Y'
    test_mode = os.getenv('TEST_MODE', 'N').upper() == 'Y'

    input_base_dir, base_dir = PATHS[(env_key, epc_run)]
    defaults = LOCAL_DEFAULTS if running_locally else HPC_DEFAULTS
    setting_name = 'local' if running_locally else 'v10'
    if test_mode:
        setting_name = f'{setting_name}_TEST'

    return RunConfig(
        setting_name=setting_name,
        input_base_dir=input_base_dir,
        base_dir=base_dir,
        budgets=defaults['budgets'],
        loft_probs=defaults['loft_probs'],
        equity_floors=defaults['equity_floors'],
        epc_run=epc_run,
        mip_gap=float(os.getenv('MIP_GAP', '0.01')),
        run_greedy_runs=os.getenv('RUN_GREEDY_RUNS_YN', 'Y').upper() != 'N',
        force_rerun=os.getenv('FORCE_RERUN', 'N').upper() == 'Y',
        test_mode=test_mode,
        test_sample_size=int(os.getenv('TEST_SAMPLE_SIZE', '500')),
        test_min_per_stratum=int(os.getenv('TEST_MIN_PER_STRATUM', '1')),
        test_seed=int(os.getenv('TEST_SEED', '42')),
        cost_scenarios=_parse_cost_scenarios_env(),
        verbose=os.getenv('VERBOSE', 'Y').upper() == 'Y',
    )


# ============================================================================
# DATA LOADING & DESCRIPTION
# ============================================================================

def load_data_simple(files: list[str]) -> pd.DataFrame:
    """Concatenate CSVs, logging per-file row counts."""
    frames = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  loaded {os.path.basename(f):<50} {len(df):>10,} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _per_run_dataset(per_run_glob: str):
    """
    Open the per-run parquet artefacts as a lazy pyarrow Dataset.

    Returns None if no files match — callers fall back to aleatoric-only.
    """
    import pyarrow.dataset as ds

    print(per_run_glob)
    files = glob.glob(per_run_glob)
    if not files:
        print(f"  [warn] No per-run means files matched {per_run_glob}")
        return None

    try:
        return ds.dataset(files, format='parquet')
    except Exception as e:
        print(f"  [warn] Failed to open per-run dataset: {e}")
        return None


def _per_run_slice_for_selection(
    dataset,
    selected_df: pd.DataFrame,
    upn_col: str = 'upn',
    intervention_col: str = 'intervention',
) -> pd.DataFrame:
    """
    Predicate-pushdown read of only the rows needed for one selection.
    """
    cols = ['upn', 'scenario', 'run_idx', 'cost_run_mean', 'co2_run_mean']
    if dataset is None or selected_df is None or selected_df.empty:
        return pd.DataFrame(columns=cols)

    import pyarrow as pa
    import pyarrow.compute as pc

    sel_upns = selected_df[upn_col].astype(str).unique().tolist()
    sel_scenarios = selected_df[intervention_col].astype(str).unique().tolist()
    if not sel_upns or not sel_scenarios:
        return pd.DataFrame(columns=cols)

    expr = pc.field('upn').isin(pa.array(sel_upns))
    expr = expr & pc.field('scenario').isin(pa.array(sel_scenarios))

    try:
        table = dataset.to_table(columns=cols, filter=expr)
    except Exception as e:
        print(f"  [warn] Predicate-pushdown read failed ({e}); "
              f"returning empty per-run slice.")
        return pd.DataFrame(columns=cols)

    out = table.to_pandas()
    if out.empty:
        return out

    sel_keys = (
        selected_df[[upn_col, intervention_col]]
        .rename(columns={intervention_col: 'scenario'})
        .drop_duplicates()
    )
    out = out.merge(sel_keys, on=[upn_col, 'scenario'], how='inner')
    return out


def describe_input(df: pd.DataFrame, pkg_col: str = 'intervention') -> None:
    """Optional diagnostic prints. Gated by cfg.verbose upstream."""
    per_building = df.groupby('upn')[pkg_col].apply(set)

    for a, b in [
        ('joint_heat_loft_decay', 'loft_installation'),
        ('joint_heat_wall_decay', 'wall_installation'),
    ]:
        has_a = per_building.apply(lambda s, a=a: a in s)
        has_b = per_building.apply(lambda s, b=b: b in s)
        print(pd.crosstab(has_a, has_b, rownames=[f'has_{a}'], colnames=[f'has_{b}']))

    print(f"\nDistinct interventions: {df[pkg_col].nunique()}")
    print(df[pkg_col].value_counts())

    menus = df.groupby('upn')[pkg_col].apply(lambda s: tuple(sorted(s.unique())))
    print(f"\nDistinct intervention menus: {menus.nunique()}")
    print("Menu size → count of buildings:")
    print(menus.apply(len).value_counts().sort_index())


def drop_upn_postcode_collisions(
    df: pd.DataFrame, upn_col: str = 'upn', threshold: int = 100,
) -> pd.DataFrame:
    """Drop UPNs that map to >1 postcode; hard-fail if there are too many."""
    counts = df.groupby(upn_col)['postcode'].nunique()
    bad = counts[counts > 1].index
    if len(bad) == 0:
        print("UPN-postcode collisions: 0")
        return df
    if len(bad) > threshold:
        raise ValueError(
            f"{len(bad)} UPN-postcode collisions — too many to be noise. "
            f"Investigate upstream join."
        )
    before = len(df)
    df = df[~df[upn_col].isin(bad)].reset_index(drop=True)
    print(f"UPN-postcode collisions: dropped {len(bad)} UPNs "
          f"({before - len(df)} rows)")
    return df


# ============================================================================
# PORTFOLIO-LEVEL UNCERTAINTY
# ============================================================================

def _portfolio_aleatoric_std(selected_df: pd.DataFrame, ale_col: str) -> float:
    if selected_df.empty or ale_col not in selected_df.columns:
        return 0.0
    return float(np.sqrt(np.sum(selected_df[ale_col].fillna(0).to_numpy() ** 2)))


def _portfolio_epistemic_totals_wide(
    per_run_slice: pd.DataFrame,
    value_col: str,
) -> np.ndarray:
    if per_run_slice is None or per_run_slice.empty:
        return np.array([])
    if value_col not in per_run_slice.columns:
        return np.array([])
    totals = (
        per_run_slice
        .groupby('run_idx')[value_col]
        .sum()
        .sort_index()
    )
    return totals.to_numpy()


# def compute_portfolio_uncertainty(
#     selected_df: pd.DataFrame,
#     per_run_dataset,
#     upn_col: str = 'upn',
#     intervention_col: str = 'intervention',
# ) -> dict:
#     """
#     Portfolio-level uncertainty fields for one optimiser solution.

#     Within a cost_scenario bucket the epistemic std reflects only the
#     other 6 factors. Cost_scenario uncertainty is expressed as the
#     cross-bucket envelope, computed in the comparison stage.
#     """
#     out = {
#         'total_cost_aleatoric_std': 0.0,
#         'total_cost_epistemic_std': 0.0,
#         'total_abatement_aleatoric_std': 0.0,
#         'total_abatement_epistemic_std': 0.0,
#         'epistemic_share_cost': float('nan'),
#         'epistemic_share_carbon': float('nan'),
#         'cpex_per_ton_p16': float('nan'),
#         'cpex_per_ton_median': float('nan'),
#         'cpex_per_ton_p84': float('nan'),
#         'per_run_totals_cost': [],
#         'per_run_totals_carbon': [],
#     }

#     if selected_df is None or selected_df.empty:
#         return out

#     out['total_cost_aleatoric_std'] = _portfolio_aleatoric_std(
#         selected_df, COST_ALEATORIC_STD_COL
#     )
#     out['total_abatement_aleatoric_std'] = _portfolio_aleatoric_std(
#         selected_df, CARBON_ALEATORIC_STD_COL
#     )

#     per_run_slice = _per_run_slice_for_selection(
#         per_run_dataset, selected_df,
#         upn_col=upn_col, intervention_col=intervention_col,
#     )
#     per_run_cost = _portfolio_epistemic_totals_wide(
#         per_run_slice, value_col='cost_run_mean',
#     )
#     per_run_carbon = _portfolio_epistemic_totals_wide(
#         per_run_slice, value_col='co2_run_mean',
#     )

#     if per_run_cost.size > 0:
#         out['total_cost_epistemic_std'] = float(np.std(per_run_cost, ddof=1))
#         out['per_run_totals_cost'] = per_run_cost.tolist()
#     if per_run_carbon.size > 0:
#         out['total_abatement_epistemic_std'] = float(np.std(per_run_carbon, ddof=1))
#         out['per_run_totals_carbon'] = per_run_carbon.tolist()

#     def _share(ale: float, epi: float) -> float:
#         denom = ale ** 2 + epi ** 2
#         if denom <= 0:
#             return float('nan')
#         return float(epi ** 2 / denom)

#     out['epistemic_share_cost'] = _share(
#         out['total_cost_aleatoric_std'], out['total_cost_epistemic_std']
#     )
#     out['epistemic_share_carbon'] = _share(
#         out['total_abatement_aleatoric_std'], out['total_abatement_epistemic_std']
#     )

#     if (per_run_cost.size > 0 and per_run_carbon.size > 0
#             and per_run_cost.size == per_run_carbon.size):
#         valid = per_run_carbon > 0
#         n_runs_dropped_zero_carbon = valid.sum() < per_run_carbon.size
#         print('n_runs_dropped_zero_carbon')
#         print(n_runs_dropped_zero_carbon)
#         if valid.any():

#             ratios = per_run_cost[valid] / per_run_carbon[valid]
#             out['cpex_per_ton_p16'] = float(np.percentile(ratios, 16))
#             out['cpex_per_ton_median'] = float(np.percentile(ratios, 50))
#             out['cpex_per_ton_p84'] = float(np.percentile(ratios, 84))

#     return out

def compute_portfolio_uncertainty(
    selected_df: pd.DataFrame,
    per_run_dataset,
    upn_col: str = 'upn',
    intervention_col: str = 'intervention',
) -> dict:
    """
    Portfolio-level uncertainty fields for one optimiser solution.

    Within a cost_scenario bucket the epistemic std reflects only the
    other 6 factors. Cost_scenario uncertainty is expressed as the
    cross-bucket envelope, computed in the comparison stage.
    """
    out = {
        'total_cost_aleatoric_std': 0.0,
        'total_cost_epistemic_std': 0.0,
        'total_abatement_aleatoric_std': 0.0,
        'total_abatement_epistemic_std': 0.0,
        'epistemic_share_cost': float('nan'),
        'epistemic_share_carbon': float('nan'),
        'cpex_per_ton_p16': float('nan'),
        'cpex_per_ton_median': float('nan'),
        'cpex_per_ton_p84': float('nan'),
        'per_run_totals_cost': [],
        'per_run_totals_carbon': [],
        # NEW diagnostics
        'n_runs': 0,
        'n_pairs_selected': 0,
        'n_pairs_in_slice': 0,
        'per_run_slice_complete': True,
    }

    if selected_df is None or selected_df.empty:
        return out

    out['total_cost_aleatoric_std'] = _portfolio_aleatoric_std(
        selected_df, COST_ALEATORIC_STD_COL
    )
    out['total_abatement_aleatoric_std'] = _portfolio_aleatoric_std(
        selected_df, CARBON_ALEATORIC_STD_COL
    )

    per_run_slice = _per_run_slice_for_selection(
        per_run_dataset, selected_df,
        upn_col=upn_col, intervention_col=intervention_col,
    )

    # ---------- NEW: per-run slice completeness check ----------
    n_selected = len(selected_df)
    out['n_pairs_selected'] = n_selected

    if per_run_slice is not None and not per_run_slice.empty:
        runs_per_pair = (
            per_run_slice
            .groupby([upn_col, 'scenario'])['run_idx']
            .nunique()
        )
        K = int(runs_per_pair.max()) if len(runs_per_pair) else 0
        out['n_runs'] = K
        out['n_pairs_in_slice'] = int(len(runs_per_pair))

        if len(runs_per_pair) < n_selected:
            out['per_run_slice_complete'] = False
            missing = n_selected - len(runs_per_pair)
            print(f"  [warn] {missing} of {n_selected} selected "
                  f"(upn, intervention) pairs are absent from the per-run "
                  f"dataset entirely; epistemic std will be biased low.")
        elif (runs_per_pair < K).any():
            out['per_run_slice_complete'] = False
            ragged = int((runs_per_pair < K).sum())
            print(f"  [warn] {ragged} of {len(runs_per_pair)} pairs have "
                  f"fewer than {K} runs; epistemic std will be biased.")
    # -----------------------------------------------------------

    per_run_cost = _portfolio_epistemic_totals_wide(
        per_run_slice, value_col='cost_run_mean',
    )
    per_run_carbon = _portfolio_epistemic_totals_wide(
        per_run_slice, value_col='co2_run_mean',
    )

    if per_run_cost.size > 0:
        out['total_cost_epistemic_std'] = float(np.std(per_run_cost, ddof=1))
        out['per_run_totals_cost'] = per_run_cost.tolist()
    if per_run_carbon.size > 0:
        out['total_abatement_epistemic_std'] = float(np.std(per_run_carbon, ddof=1))
        out['per_run_totals_carbon'] = per_run_carbon.tolist()

    def _share(ale: float, epi: float) -> float:
        denom = ale ** 2 + epi ** 2
        if denom <= 0:
            return float('nan')
        return float(epi ** 2 / denom)

    out['epistemic_share_cost'] = _share(
        out['total_cost_aleatoric_std'], out['total_cost_epistemic_std']
    )
    out['epistemic_share_carbon'] = _share(
        out['total_abatement_aleatoric_std'], out['total_abatement_epistemic_std']
    )

    if (per_run_cost.size > 0 and per_run_carbon.size > 0
            and per_run_cost.size == per_run_carbon.size):
        valid = per_run_carbon > 0
        if valid.any():
            ratios = per_run_cost[valid] / per_run_carbon[valid]
            out['cpex_per_ton_p16'] = float(np.percentile(ratios, 16))
            out['cpex_per_ton_median'] = float(np.percentile(ratios, 50))
            out['cpex_per_ton_p84'] = float(np.percentile(ratios, 84))

    return out

# ============================================================================
# PARETO SUMMARY PLOTS (per bucket — unchanged from previous version)
# ============================================================================

PERSONA_COLORS = {
    'high_risk':   '#d32f2f',
    'med_risk':    '#f57c00',
    'middle_risk': '#fbc02d',
    'low_risk':    '#66bb6a',
    'v_low_risk':  '#42a5f5',
}

PERSONA_LABELS = {
    'high_risk':   'High risk',
    'med_risk':    'Med risk',
    'middle_risk': 'Middle',
    'low_risk':    'Low risk',
    'v_low_risk':  'V. low risk',
}


def _get_cmap(name, n):
    try:
        return plt.colormaps.get_cmap(name).resampled(n)
    except AttributeError:
        return plt.cm.get_cmap(name, n)


def plot_pareto_summary(all_stats, baseline_stats, output_dir, budget,
                        bucket_label: Optional[str] = None):
    """Per-bucket summary plots. Unchanged behaviour, optional bucket title."""
    feasible = [s for s in all_stats if s['status'] in ('Optimal', 'Not Solved')]
    if not feasible:
        print("No feasible solutions to plot.")
        return

    eq_floors = [s['equity_floor_pct'] for s in feasible]
    title_suffix = f" — {bucket_label}" if bucket_label else ""

    def save_fig(fig, name):
        fig.tight_layout()
        path = os.path.join(output_dir, f'pareto_{name}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved {name}.png")
        plt.close(fig)

    # Plot 1: Pareto front
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        abatements = np.array([s['total_abatement'] for s in feasible])
        ale_std = np.array([s.get('total_abatement_aleatoric_std', 0.0) or 0.0
                            for s in feasible])
        epi_std = np.array([s.get('total_abatement_epistemic_std', 0.0) or 0.0
                            for s in feasible])

        ax.fill_between(eq_floors, abatements - epi_std, abatements + epi_std,
                        color='#1976d2', alpha=0.18, label='± epistemic σ')
        ax.fill_between(eq_floors, abatements - ale_std, abatements + ale_std,
                        color='#1976d2', alpha=0.45, label='± aleatoric σ')
        ax.plot(eq_floors, abatements, 'o-', color='#1976d2', linewidth=2,
                markersize=7, label='Multi-choice knapsack (mean)')

        if baseline_stats.get('total_abatement'):
            ax.axhline(baseline_stats['total_abatement'], color='#d32f2f',
                       linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f"Baseline: {baseline_stats['total_abatement']:.0f} tCO2")

        ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
        ax.set_ylabel('Total CO₂ abatement (tonnes)', fontsize=11)
        ax.set_title(f'Pareto Front — Budget £{budget/1e6:.1f}M{title_suffix}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'front_abatement')
    except Exception as e:
        print(f"  Plot 1 failed: {e}")

    # Plot 2: £/tCO2 vs equity floor
    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        cpex_mean = [s.get('cpex_per_ton') for s in feasible]
        ax.plot(eq_floors, cpex_mean, 's-', color='#f57c00', linewidth=2,
                markersize=7, label='£/tCO₂ (mean of totals)')

        med = np.array([s.get('cpex_per_ton_median', np.nan) for s in feasible],
                       dtype=float)
        p16 = np.array([s.get('cpex_per_ton_p16', np.nan) for s in feasible],
                       dtype=float)
        p84 = np.array([s.get('cpex_per_ton_p84', np.nan) for s in feasible],
                       dtype=float)
        if not np.all(np.isnan(med)):
            ax.fill_between(eq_floors, p16, p84,
                            color='#f57c00', alpha=0.20,
                            label='P16–P84 across epistemic runs')
            ax.plot(eq_floors, med, 'd--', color='#bf360c', linewidth=1.5,
                    markersize=5, label='Median across runs')

        if baseline_stats.get('cpex_per_ton'):
            ax.axhline(baseline_stats['cpex_per_ton'], color='#d32f2f',
                       linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f"Baseline: £{baseline_stats['cpex_per_ton']:,.0f}/t")

        ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
        ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
        ax.set_title(f'Cost-Effectiveness — Budget £{budget/1e6:.1f}M{title_suffix}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'front_cpex')
    except Exception as e:
        print(f"  Plot 2 failed: {e}")

    personas_order = ['high_risk', 'med_risk', 'middle_risk', 'low_risk', 'v_low_risk']
    width = 0.7

    for plot_idx, (key, ylabel, divisor, suffix) in enumerate([
        ('buildings', 'Number of buildings retrofitted', 1, 'persona_buildings'),
        ('spend',     'Spend (£M)',                     1e6, 'persona_spend'),
        ('abatement', 'CO₂ abatement (tonnes)',         1, 'persona_abatement'),
    ], start=3):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(eq_floors))
            bottom = np.zeros(len(eq_floors))
            for p in personas_order:
                vals = [s['persona_breakdown'].get(p, {}).get(key, 0) / divisor
                        for s in feasible]
                ax.bar(x, vals, width, bottom=bottom,
                       label=PERSONA_LABELS.get(p, p),
                       color=PERSONA_COLORS.get(p, '#999'),
                       edgecolor='white', linewidth=0.5)
                bottom += np.array(vals)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{e:.0f}%' for e in eq_floors], rotation=45)
            ax.set_xlabel('Equity floor', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Persona Split — {key.title()} — '
                         f'£{budget/1e6:.1f}M{title_suffix}',
                         fontsize=13, fontweight='bold')
            ax.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            save_fig(fig, suffix)
        except Exception as e:
            print(f"  Plot {plot_idx} failed: {e}")

    deciles_order = list(range(1, 11))
    decile_colors = plt.cm.viridis(np.linspace(0, 1, 10))

    for plot_idx, (key, ylabel, divisor, suffix) in enumerate([
        ('buildings', 'Number of buildings retrofitted', 1,   'decile_buildings'),
        ('spend',     'Spend (£M)',                     1e6, 'decile_spend'),
        ('abatement', 'CO₂ abatement (tonnes)',         1,   'decile_abatement'),
    ], start=3):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(eq_floors))
            bottom = np.zeros(len(eq_floors))
            width_local = 0.75

            for i, d in enumerate(deciles_order):
                vals = [s['percentile_breakdown'].get(d, {}).get(key, 0) / divisor
                        for s in feasible]
                ax.bar(x, vals, width_local, bottom=bottom,
                       label=f'Decile {d}',
                       color=decile_colors[i],
                       edgecolor='white', linewidth=0.5)
                bottom += np.array(vals)

            ax.set_xticks(x)
            ax.set_xticklabels([f'{e:.0f}%' for e in eq_floors], rotation=45)
            ax.set_xlabel('Equity floor', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Decile Split — {key.title()} — '
                         f'£{budget/1e6:.1f}M{title_suffix}',
                         fontsize=13, fontweight='bold')
            ax.legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            save_fig(fig, suffix)
        except Exception as e:
            print(f"  Plot {plot_idx} failed: {e}")

    try:
        all_interventions = set()
        for s in feasible:
            all_interventions.update(s.get('intervention_breakdown', {}).keys())
        all_interventions = sorted(all_interventions)

        if all_interventions:
            intv_cmap = _get_cmap('tab10', max(len(all_interventions), 1))
            intv_colors = {intv: intv_cmap(i) for i, intv in enumerate(all_interventions)}

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(eq_floors))
            bottom = np.zeros(len(eq_floors))
            for intv in all_interventions:
                vals = [s.get('intervention_breakdown', {}).get(intv, {}).get('buildings', 0)
                        for s in feasible]
                label = intv.replace('_', ' ').title()
                ax.bar(x, vals, width, bottom=bottom, label=label,
                       color=intv_colors[intv], edgecolor='white', linewidth=0.5)
                bottom += np.array(vals)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{e:.0f}%' for e in eq_floors], rotation=45)
            ax.set_xlabel('Equity floor', fontsize=11)
            ax.set_ylabel('Number of buildings', fontsize=11)
            ax.set_title(f'Intervention Mix — £{budget/1e6:.1f}M{title_suffix}',
                         fontsize=13, fontweight='bold')
            ax.legend(title='Intervention', bbox_to_anchor=(1.05, 1),
                      loc='upper left', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            save_fig(fig, 'intervention_mix')
    except Exception as e:
        print(f"  Plot 6 failed: {e}")

    print("Pareto summary plots complete.")


# ============================================================================
# LOGGING HELPER
# ============================================================================

def setup_loggers(output_dir, million_budget, prob_loft, bucket=None):
    """Fresh file loggers, scoped per bucket to avoid handler collisions."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{bucket}" if bucket else ""

    summary_logger = logging.getLogger(
        f'summary_{million_budget}_{prob_loft}{suffix}'
    )
    summary_logger.handlers.clear()
    summary_logger.setLevel(logging.INFO)
    summary_logger.propagate = False
    sh = logging.FileHandler(os.path.join(output_dir, f'summary_log_{timestamp}.log'))
    summary_logger.addHandler(sh)

    detail_logger = logging.getLogger(
        f'detail_{million_budget}_{prob_loft}{suffix}'
    )
    detail_logger.handlers.clear()
    detail_logger.setLevel(logging.INFO)
    detail_logger.propagate = False
    dh = logging.FileHandler(os.path.join(output_dir, f'detail_log_{timestamp}.log'))
    detail_logger.addHandler(dh)
    detail_logger.addHandler(logging.StreamHandler())

    return summary_logger, detail_logger


# ============================================================================
# STRATIFIED SAMPLING FOR TEST MODE
# ============================================================================

def stratified_sample_buildings(
    df,
    personas,
    target_n,
    upn_col='upn',
    postcode_col='postcode',
    persona_col='meta_socio_persona',
    intervention_col='intervention',
    decile_col='avg_gas_percentile',
    min_per_stratum=1,
    seed=42,
    logger=None,
):
    rng = np.random.default_rng(seed)

    def _log(msg):
        print(msg)
        if logger is not None:
            logger.info(msg)

    menus = (
        df.groupby(upn_col)[intervention_col]
          .apply(lambda s: tuple(sorted(s.unique())))
          .rename('menu')
    )

    upn_features = (
        df[[upn_col, postcode_col, decile_col]]
          .drop_duplicates(subset=[upn_col])
          .set_index(upn_col)
    )

    building_df = pd.concat([menus, upn_features], axis=1).reset_index()

    personas_small = personas[[postcode_col, persona_col]].drop_duplicates(
        subset=[postcode_col]
    )
    building_df = building_df.merge(personas_small, on=postcode_col, how='left')

    n_missing_persona = building_df[persona_col].isna().sum()
    if n_missing_persona > 0:
        _log(f"  [sampler] {n_missing_persona} buildings have no persona — excluded.")
        building_df = building_df.dropna(subset=[persona_col])

    n_missing_decile = building_df[decile_col].isna().sum()
    if n_missing_decile > 0:
        _log(f"  [sampler] {n_missing_decile} buildings have no decile — excluded.")
        building_df = building_df.dropna(subset=[decile_col])

    n_buildings_total = len(building_df)
    if n_buildings_total == 0:
        raise ValueError("No buildings left to sample after persona/decile joins.")

    if target_n >= n_buildings_total:
        _log(f"  [sampler] Requested {target_n} >= available "
             f"{n_buildings_total}; returning all buildings.")
        sampled_upns = building_df[upn_col].tolist()
    else:
        building_df['_stratum'] = list(
            zip(building_df[persona_col], building_df['menu'], building_df[decile_col])
        )
        strata_sizes = building_df['_stratum'].value_counts()
        n_strata = len(strata_sizes)
        _log(f"  [sampler] {n_strata} non-empty strata across "
             f"{n_buildings_total:,} buildings.")

        floor_alloc = {s: min(min_per_stratum, sz) for s, sz in strata_sizes.items()}
        floor_total = sum(floor_alloc.values())

        if floor_total >= target_n:
            alloc = floor_alloc
        else:
            remainder = target_n - floor_total
            headroom = {s: sz - floor_alloc[s] for s, sz in strata_sizes.items()}
            total_headroom = sum(headroom.values())

            if total_headroom == 0:
                alloc = floor_alloc
            else:
                raw = {
                    s: remainder * (headroom[s] / total_headroom)
                    for s in strata_sizes.index
                }
                floored = {s: int(np.floor(v)) for s, v in raw.items()}
                leftover = remainder - sum(floored.values())
                fracs = sorted(
                    raw.items(),
                    key=lambda kv: kv[1] - np.floor(kv[1]),
                    reverse=True,
                )
                for s, _ in fracs[:leftover]:
                    floored[s] += 1

                alloc = {s: floor_alloc[s] + floored[s] for s in strata_sizes.index}
                alloc = {s: min(alloc[s], strata_sizes[s]) for s in strata_sizes.index}

        sampled_upns = []
        for stratum, n_take in alloc.items():
            if n_take <= 0:
                continue
            upns_in_stratum = building_df.loc[
                building_df['_stratum'] == stratum, upn_col
            ].to_numpy()
            if len(upns_in_stratum) <= n_take:
                picked = upns_in_stratum
            else:
                picked = rng.choice(upns_in_stratum, size=n_take, replace=False)
            sampled_upns.extend(picked.tolist())

        _log(f"  [sampler] Target {target_n} → realised {len(sampled_upns):,} "
             f"buildings across {sum(1 for v in alloc.values() if v > 0)} strata.")

    sampled_upns_set = set(sampled_upns)
    sampled_df = df[df[upn_col].isin(sampled_upns_set)].copy()

    persona_counts = (
        building_df[building_df[upn_col].isin(sampled_upns_set)]
        [persona_col].value_counts().to_dict()
    )
    decile_counts = (
        building_df[building_df[upn_col].isin(sampled_upns_set)]
        [decile_col].value_counts().to_dict()
    )

    _log(f"  [sampler] Persona breakdown in sample: {persona_counts}")
    _log(f"  [sampler] Sampled rows: {len(sampled_df):,} "
         f"(from {len(df):,}); UPNs: {sampled_df[upn_col].nunique():,}")

    sample_info = {
        'target_n': target_n,
        'realised_n_buildings': sampled_df[upn_col].nunique(),
        'realised_n_rows': len(sampled_df),
        'persona_breakdown': persona_counts,
        'decile_breakdown': decile_counts,
        'seed': seed,
    }

    return sampled_df, sample_info


def _scale_budgets_for_test(budgets, n_sampled, n_total):
    """Scale budgets proportionally to the sample fraction (floor £100k)."""
    if n_total == 0:
        return budgets
    frac = n_sampled / n_total
    scaled = [max(100_000, int(b * frac)) for b in budgets]
    seen = set()
    out = []
    for b in scaled:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


# ============================================================================
# SKIP-IF-EXISTS
# ============================================================================
 

def comparison_is_complete(output_dir: str) -> bool:
    """True if the cross-bucket comparison artefacts exist for this slice."""
    for name in ('stability.csv', 'envelope.csv', 'pareto_overlay.png'):
        p = os.path.join(output_dir, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
    return True


# ============================================================================
# PIPELINE STAGES (per bucket)
# ============================================================================

def load_and_prepare_data(
    cfg: RunConfig,
    prob_loft: float,
    bucket: str,
    personas: pd.DataFrame,
):
    """Load CSVs and per-run dataset for one (loft, bucket) pair."""
    input_glob = cfg.bucket_input_glob(bucket)
    per_run_glob = cfg.bucket_per_run_glob(bucket)

    files = [x for x in glob.glob(input_glob) if f'loft_{prob_loft}' in x]
    print(f'\n[{bucket}] glob: {input_glob}')
    print(f'[{bucket}] Found {len(files)} files for loft prob {prob_loft}')
    if not files:
        return pd.DataFrame(), pd.DataFrame(), None, None

    print(f"\n[{bucket}] Loading input data...")
    res_df = load_data_simple(files)
    print(f'[{bucket}] res_df shape: {res_df.shape}, '
          f'n_upns: {res_df["upn"].nunique()}')

    print(f"\n[{bucket}] Opening per-run dataset (lazy)...")
    per_run_dataset = _per_run_dataset(per_run_glob)
    if per_run_dataset is not None:
        try:
            n_files = len(per_run_dataset.files)
        except Exception:
            n_files = '?'
        print(f"  Per-run dataset: {n_files} parquet file(s) registered.")
    else:
        print("  [warn] No per-run dataset; epistemic uncertainty will be 0.")

    sample_info = None
    if cfg.test_mode:
        print(f"\n[{bucket}][TEST_MODE] Sampling ~{cfg.test_sample_size} buildings...")
        n_before = res_df['upn'].nunique()
        # Same seed across buckets so sampled UPN sets coincide.
        res_df, sample_info = stratified_sample_buildings(
            df=res_df, personas=personas,
            target_n=cfg.test_sample_size,
            min_per_stratum=cfg.test_min_per_stratum,
            seed=cfg.test_seed,
        )
        n_after = res_df['upn'].nunique()
        cfg.budgets = _scale_budgets_for_test(
            cfg.budgets, n_sampled=n_after, n_total=n_before,
        )
        print(f"[{bucket}][TEST_MODE] Scaled budgets: "
              f"{[f'£{b/1e6:.2f}M' for b in cfg.budgets]}")

    if cfg.verbose:
        describe_input(res_df)

    res_df = drop_upn_postcode_collisions(res_df)
    res_df = validate_multipackage_input(  # noqa: F405
        res_df, personas,
        upn_col='upn',
        min_packages=MIN_PACKAGES_PER_BUILDING,  # noqa: F405
        max_packages=MAX_PACKAGES_PER_BUILDING,  # noqa: F405
    )

    df = res_df.merge(personas, on='postcode', how='inner')
    validate_post_merge(df, upn_col='upn',  # noqa: F405
                        max_packages=MAX_PACKAGES_PER_BUILDING)  # noqa: F405

    df = df[df['premise_type'] != 'Domestic_outbuilding']
    df = df[~df['premise_type'].isna()]
    gc.collect()
    print(f"[{bucket}] After premise filtering: {len(df):,} rows "
          f"({df['upn'].nunique():,} buildings)")

    df_buildings = build_building_level_view(df, upn_col='upn')  # noqa: F405
    return df, df_buildings, sample_info, per_run_dataset


def run_all_budgets(
    cfg: RunConfig,
    df: pd.DataFrame,
    df_buildings: pd.DataFrame,
    per_run_dataset,
    mip_gap: float,
    prob_loft: float,
    bucket: str,
    sample_info: Optional[dict],
) -> None:
    """Pareto sweep for every budget at a fixed (loft, bucket) pair."""
    bucket_root = cfg.bucket_output_dir(bucket)
    for budget in cfg.budgets:
        output_dir = os.path.join(
            bucket_root,
            f'budget_{budget_label(budget)}M__loft_{prob_loft}__mip_{mip_gap}',
        )
        os.makedirs(output_dir, exist_ok=True)

        if (not cfg.force_rerun and not cfg.test_mode
                and run_is_complete(output_dir, cfg.equity_floors)):
            print(f"\n[SKIP] [{bucket}] Budget £{budget_label(budget)}M "
                  f"loft={prob_loft} already complete: {output_dir}")
            continue

        summary_logger, detail_logger = setup_loggers(
            output_dir, budget_label(budget), prob_loft, bucket=bucket,
        )
        summary_logger.info(
            f'[{bucket}] Starting Pareto: Budget £{budget:,}, Loft {prob_loft}'
        )
        if cfg.test_mode and sample_info is not None:
            summary_logger.info(f'TEST_MODE sample_info: {sample_info}')

        _, all_stats, _ = run_pareto(
            df_all_packages=df,
            df_buildings=df_buildings,
            per_run_dataset=per_run_dataset,
            budget=budget,
            mip_gap=mip_gap,
            equity_floors=cfg.equity_floors,
            high_equity_personas=DEFAULT_HIGH_EQUITY_PERSONAS,
            output_dir=output_dir,
            loft_prob=prob_loft,
            detail_logger=detail_logger,
            summary_logger=summary_logger,
            bucket_label=bucket,
        )
        summary_logger.info("Pareto analysis complete!")
        print(f"✓ [{bucket}] Results saved to: {output_dir}")

        if cfg.epc_run:
              _run_epc_fallback(
                    df, budget, output_dir, detail_logger, summary_logger,
                    per_run_dataset=per_run_dataset,   # NEW — already in scope
                )


 
def _run_epc_fallback(
    df: pd.DataFrame,
    cfg: RunConfig,
    key: SliceKey,
    *,
    detail_logger,
    summary_logger,
    per_run_dataset=None,
) -> None:
    selected, stats = select_epc_algo_pareto(
        df_knapsack=df, budget=key.budget,
        cost_col=COST_COL, carbon_col=CARBON_COL,
        logger=detail_logger,
    )
    if selected.empty:
        detail_logger.info('EPC selection empty')
        raise EPCSelectionError(
            f'EPC random selection produced no rows at budget £{key.budget:,}'
        )

    unc = compute_portfolio_uncertainty(
        selected_df=selected,
        per_run_dataset=per_run_dataset,
        upn_col='upn',
        intervention_col='intervention',
    )
    stats.update(unc)

    selected.to_csv(
        epc_selected_csv(cfg.root, key.bucket, key.budget, key.loft),
        index=False,
    )
    pd.DataFrame([stats]).to_csv(
        epc_summary_csv(cfg.root, key.bucket, key.budget, key.loft),
        index=False,
    )
    summary_logger.info(
        f"EPC summary written to "
        f"{epc_summary_csv(cfg.root, key.bucket, key.budget, key.loft)}"
    )
# ============================================================================
# CROSS-BUCKET COMPARISON
# ============================================================================

def _read_bucket_pareto_summary(
    cfg: RunConfig, bucket: str, budget: float, loft: float,
) -> Optional[pd.DataFrame]:
    path = summary_csv(cfg.root, bucket, budget, loft)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] Failed to read {path}: {e}")
        return None


def _read_bucket_selected(
    cfg: RunConfig, bucket: str, budget: int, prob_loft: float, eps: int,
) -> Optional[pd.DataFrame]:
    """Load one bucket's selected_projects_eq{eps}.csv. None if missing."""
    path = os.path.join(
        cfg.bucket_output_dir(bucket),
        f'budget_{budget_label(budget)}M__loft_{prob_loft}__mip_{cfg.mip_gap}',
        f'selected_projects_eq{eps:.0f}.csv',
    )
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] Failed to read {path}: {e}")
        return None


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity. Returns NaN if both sets are empty."""
    if not a and not b:
        return float('nan')
    union = a | b
    if not union:
        return float('nan')
    return len(a & b) / len(union)


def _stability_for_slice(
    cfg: RunConfig, budget: int, prob_loft: float,
) -> pd.DataFrame:
    """
    Pairwise UPN-set and (UPN, intervention)-set Jaccard across buckets,
    one row per (eps, bucket_a, bucket_b).
    """
    rows = []
    buckets = cfg.cost_scenarios

    for eps in cfg.equity_floors:
        sels = {}
        for b in buckets:
            df = _read_bucket_selected(cfg, b, budget, prob_loft, eps)
            if df is None or df.empty:
                continue
            upn_set = set(df['upn'].astype(str))
            if 'intervention' in df.columns:
                pair_set = set(zip(df['upn'].astype(str), df['intervention'].astype(str)))
            else:
                pair_set = set()
            sels[b] = (upn_set, pair_set, len(df))

        present = [b for b in buckets if b in sels]
        for i, ba in enumerate(present):
            for bb in present[i + 1:]:
                ua, pa_, na = sels[ba]
                ub, pb_, nb = sels[bb]
                rows.append({
                    'budget': budget,
                    'loft_prob': prob_loft,
                    'equity_floor_pct': eps,
                    'bucket_a': ba,
                    'bucket_b': bb,
                    'n_a': na,
                    'n_b': nb,
                    'jaccard_upn': _jaccard(ua, ub),
                    'jaccard_pair': _jaccard(pa_, pb_),
                    'n_intersection_upn': len(ua & ub),
                    'n_union_upn': len(ua | ub),
                })
        # Log buckets that were dropped at this eps.
        missing = [b for b in buckets if b not in sels]
        if missing:
            print(f"  [stability] eps={eps}%: missing/empty selections for "
                  f"{missing}; dropped pairs.")

    return pd.DataFrame(rows)


def _envelope_for_slice(
    cfg: RunConfig, budget: int, prob_loft: float,
) -> pd.DataFrame:
    """
    Cross-bucket min/median/max for total_cost, total_abatement,
    cpex_per_ton at each equity_floor.
    """
    bucket_summaries = {}
    for b in cfg.cost_scenarios:
        df = _read_bucket_pareto_summary(cfg, b, budget, prob_loft)
        if df is None:
            continue
        df = df[df['status'].isin(['Optimal', 'Not Solved'])].copy()
        bucket_summaries[b] = df

    if not bucket_summaries:
        return pd.DataFrame()

    rows = []
    for eps in cfg.equity_floors:
        per_bucket_vals = {}
        for b, df in bucket_summaries.items():
            sub = df[df['equity_floor_pct'] == eps]
            if sub.empty:
                continue
            r = sub.iloc[0]
            per_bucket_vals[b] = {
                'total_cost': r.get('total_cost'),
                'total_abatement': r.get('total_abatement'),
                'cpex_per_ton': r.get('cpex_per_ton'),
            }

        if not per_bucket_vals:
            continue

        out = {
            'budget': budget,
            'loft_prob': prob_loft,
            'equity_floor_pct': eps,
            'n_buckets_present': len(per_bucket_vals),
        }
        for metric in ('total_cost', 'total_abatement', 'cpex_per_ton'):
            arr = np.array(
                [v[metric] for v in per_bucket_vals.values() if v[metric] is not None],
                dtype=float,
            )
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                out[f'{metric}_min'] = np.nan
                out[f'{metric}_median'] = np.nan
                out[f'{metric}_max'] = np.nan
                out[f'{metric}_range'] = np.nan
                out[f'{metric}_range_pct_of_median'] = np.nan
            else:
                out[f'{metric}_min'] = float(np.min(arr))
                out[f'{metric}_median'] = float(np.median(arr))
                out[f'{metric}_max'] = float(np.max(arr))
                out[f'{metric}_range'] = float(np.max(arr) - np.min(arr))
                med = float(np.median(arr))
                out[f'{metric}_range_pct_of_median'] = (
                    float((np.max(arr) - np.min(arr)) / med * 100)
                    if med != 0 else np.nan
                )
        # Also keep per-bucket point values for traceability.
        for b in cfg.cost_scenarios:
            v = per_bucket_vals.get(b, {})
            for metric in ('total_cost', 'total_abatement', 'cpex_per_ton'):
                out[f'{metric}__{b}'] = v.get(metric)
        rows.append(out)

    return pd.DataFrame(rows)


def _plot_pareto_overlay(
    cfg: RunConfig, budget: int, prob_loft: float, output_dir: str,
) -> None:
    """Overlay the 3 Pareto fronts (one line per bucket) with their bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False

    for b in cfg.cost_scenarios:
        df = _read_bucket_pareto_summary(cfg, b, budget, prob_loft)
        if df is None:
            continue
        df = df[df['status'].isin(['Optimal', 'Not Solved'])].sort_values(
            'equity_floor_pct'
        )
        if df.empty:
            continue

        eqs = df['equity_floor_pct'].to_numpy()
        means = df['total_abatement'].to_numpy()
        ale = df.get('total_abatement_aleatoric_std',
                     pd.Series(np.zeros(len(df)))).fillna(0).to_numpy()
        epi = df.get('total_abatement_epistemic_std',
                     pd.Series(np.zeros(len(df)))).fillna(0).to_numpy()
        total_std = np.sqrt(ale ** 2 + epi ** 2)

        colour = BUCKET_COLOURS.get(b, '#555555')
        ax.fill_between(eqs, means - total_std, means + total_std,
                        color=colour, alpha=0.18)
        ax.plot(eqs, means, 'o-', color=colour, linewidth=2, markersize=6,
                label=f'{b}')
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"  [overlay] No data to plot for budget=£{budget/1e6:.1f}M, "
              f"loft={prob_loft}")
        return

    ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
    ax.set_ylabel('Total CO₂ abatement (tonnes)', fontsize=11)
    ax.set_title(
        f'Pareto Fronts Across Cost Scenarios — '
        f'£{budget/1e6:.1f}M, loft={prob_loft}',
        fontsize=13, fontweight='bold',
    )
    ax.legend(title='Cost scenario', fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, 'pareto_overlay.png')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved overlay: {out}")


def run_comparison(cfg: RunConfig) -> None:
    """
    Build cross-bucket comparison artefacts. Runs after all bucket sweeps
    have completed for a given (budget, loft_prob).
    """
    print("\n" + "=" * 80)
    print("CROSS-BUCKET COMPARISON")
    print("=" * 80)

    for prob_loft in cfg.loft_probs:
        for budget in cfg.budgets:
            slice_dir = os.path.join(
                cfg.comparison_dir,
                f'budget_{budget_label(budget)}M__loft_{prob_loft}__mip_{cfg.mip_gap}',
            )
            os.makedirs(slice_dir, exist_ok=True)

            if (not cfg.force_rerun and not cfg.test_mode
                    and comparison_is_complete(slice_dir)):
                print(f"[SKIP] comparison already complete: {slice_dir}")
                continue

            print(f"\nComparison slice: budget=£{budget/1e6:.1f}M, "
                  f"loft={prob_loft}")

            # Stability
            stab = _stability_for_slice(cfg, budget, prob_loft)
            stab_path = os.path.join(slice_dir, 'stability.csv')
            stab.to_csv(stab_path, index=False)
            print(f"  Saved {stab_path} ({len(stab)} rows)")

            # Envelope
            env = _envelope_for_slice(cfg, budget, prob_loft)
            env_path = os.path.join(slice_dir, 'envelope.csv')
            env.to_csv(env_path, index=False)
            print(f"  Saved {env_path} ({len(env)} rows)")

            # Overlay plot
            _plot_pareto_overlay(cfg, budget, prob_loft, slice_dir)


# ============================================================================
# PARETO RUNNER
# ============================================================================
def run_pareto(
    df_all_packages: pd.DataFrame,
    df_buildings: pd.DataFrame,
    per_run_dataset,
    cfg: RunConfig,
    key: SliceKey,
    equity_floors: list,
    high_equity_personas: list,
    view_dir: str,
    cost_col: str = COST_COL,
    carbon_col: str = CARBON_COL,
    upn_col: str = 'upn',
    intervention_col: str = 'intervention',
    persona_col: str = 'meta_socio_persona',
    time_limit_seconds: int = 600,
    detail_logger: Optional[logging.Logger] = None,
    summary_logger: Optional[logging.Logger] = None,
):
    """
    Run the equity-floor sweep for one (bucket, budget, loft) slice.

    Solves multichoice knapsack at each equity floor, propagates per-run
    uncertainty through the fixed selection, computes a baseline, and
    writes outputs to the slice's data folder. Stage 1 plots go to
    view_dir.

    Returns
    -------
    pareto_df : DataFrame
        One row per equity floor with the standard summary columns.
    all_stats : list[dict]
        Full stats dicts (richer than pareto_df, including per-run totals).
    baseline_stats : dict
        One-dict summary for the preselect-best-cpt baseline.
    """
    output_dir = slice_data_dir(cfg.root, key.bucket, key.budget, key.loft)
    sel_dir = selected_dir(cfg.root, key.bucket, key.budget, key.loft)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sel_dir, exist_ok=True)
    os.makedirs(view_dir, exist_ok=True)

    if summary_logger:
        summary_logger.info(
            f"Starting Pareto sweep [bucket={key.bucket}]: "
            f"budget=£{key.budget:,.0f}, equity_floors={equity_floors}"
        )

    # ── Equity-floor sweep ────────────────────────────────────────────────
    all_stats = []
    for eps in equity_floors:
        print(f"\n{'='*60}")
        print(f"[{key.bucket}] Equity floor: {eps}% to {high_equity_personas}")
        print(f"{'='*60}")

        selected_df, stats = multichoice_knapsack(
            df_all_packages=df_all_packages,
            budget=key.budget,
            equity_floor_pct=eps,
            mip_gap=cfg.mip_gap,
            high_equity_personas=high_equity_personas,
            upn_col=upn_col,
            persona_col=persona_col,
            cost_col=cost_col,
            carbon_col=carbon_col,
            time_limit_seconds=time_limit_seconds,
            logger=detail_logger,
        )

        unc = compute_portfolio_uncertainty(
            selected_df=selected_df,
            per_run_dataset=per_run_dataset,
            upn_col=upn_col,
            intervention_col=intervention_col,
        )
        stats.update(unc)
        stats['cost_scenario'] = key.bucket

        if summary_logger:
            summary_logger.info(
                f"  eq={eps}% uncertainty: "
                f"abatement σ_ale={unc['total_abatement_aleatoric_std']:.1f}, "
                f"σ_epi={unc['total_abatement_epistemic_std']:.1f}, "
                f"epistemic_share={unc['epistemic_share_carbon']}"
            )
            if not unc['per_run_slice_complete']:
                summary_logger.warning(
                    f"  eq={eps}%: per-run slice incomplete "
                    f"({unc['n_pairs_in_slice']}/{unc['n_pairs_selected']} "
                    f"pairs found; epistemic std biased low)."
                )

        all_stats.append(stats)

        if not selected_df.empty:
            selected_df.to_csv(
                selected_csv(cfg.root, key.bucket, key.budget, key.loft, eps),
                index=False,
            )

        if stats['status'] not in ('Optimal', 'Not Solved'):
            print(f"  Infeasible at {eps}% — stopping sweep.")
            if summary_logger:
                summary_logger.info(f"Infeasible at equity_floor={eps}%")
            break

    # ── Baseline (preselect best aleatoric-σ £/tCO₂) ──────────────────────
    print(f"\n{'='*60}")
    print(f"[{key.bucket}] BASELINE: pre-select best aleatoric-σ £/tCO2")
    print(f"{'='*60}")

    df_preselected = preselect_best_cpt(
        df_all_packages, upn_col=upn_col,
        cost_col=cost_col, carbon_col=carbon_col,
        score_col=SELECTION_SCORE_COL,
    )
    baseline_selected, baseline_stats = multichoice_knapsack(
        df_all_packages=df_preselected,
        budget=key.budget,
        equity_floor_pct=0,
        mip_gap=cfg.mip_gap,
        high_equity_personas=high_equity_personas,
        upn_col=upn_col,
        persona_col=persona_col,
        cost_col=cost_col,
        carbon_col=carbon_col,
        time_limit_seconds=time_limit_seconds,
        logger=detail_logger,
    )
    baseline_stats['method'] = 'pre_select_best_cpt'
    baseline_stats['cost_scenario'] = key.bucket
    baseline_unc = compute_portfolio_uncertainty(
        selected_df=baseline_selected,
        per_run_dataset=per_run_dataset,
        upn_col=upn_col,
        intervention_col=intervention_col,
    )
    baseline_stats.update(baseline_unc)

    baseline_selected.to_csv(
        baseline_csv(cfg.root, key.bucket, key.budget, key.loft),
        index=False,
    )
    pd.DataFrame([baseline_stats]).to_csv(
        baseline_summary_csv(cfg.root, key.bucket, key.budget, key.loft),
        index=False,
    )

    # ── Summary CSV ───────────────────────────────────────────────────────
    pareto_df = pd.DataFrame(all_stats)
    summary_cols = [
        'equity_floor_pct', 'status', 'n_retrofitted', 'n_high_equity',
        'total_cost', 'total_abatement', 'cpex_per_ton',
        'total_cost_aleatoric_std', 'total_cost_epistemic_std',
        'total_abatement_aleatoric_std', 'total_abatement_epistemic_std',
        'epistemic_share_cost', 'epistemic_share_carbon',
        'cpex_per_ton_p16', 'cpex_per_ton_median', 'cpex_per_ton_p84',
        'n_runs', 'n_pairs_selected', 'n_pairs_in_slice',
        'per_run_slice_complete',
        'high_eq_spend_pct', 'high_eq_abatement_pct', 'solve_time_s',
        'cost_scenario',
    ]
    available_cols = [c for c in summary_cols if c in pareto_df.columns]
    pareto_df[available_cols].to_csv(
        summary_csv(cfg.root, key.bucket, key.budget, key.loft),
        index=False,
    )

    # ── Stage 1 plots ─────────────────────────────────────────────────────
    plot_pareto_summary(
        all_stats, baseline_stats, view_dir, key.budget,
        bucket_label=key.bucket,
    )

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"[{key.bucket}] PARETO FRONT SUMMARY")
    print(f"{'#'*60}")
    print(pareto_df[available_cols].to_string(index=False))

    if baseline_stats.get('total_abatement'):
        print(f"\n[{key.bucket}] Baseline: "
              f"{baseline_stats['total_abatement']:.1f} tCO2, "
              f"£{baseline_stats['cpex_per_ton']:,.0f}/t, "
              f"{baseline_stats['high_eq_spend_pct']:.1f}% high-eq spend")
        if all_stats and all_stats[0].get('total_abatement'):
            improvement = (
                (all_stats[0]['total_abatement']
                 - baseline_stats['total_abatement'])
                / baseline_stats['total_abatement'] * 100
            )
            print(f"  Improvement: +{improvement:.1f}% abatement vs baseline")

    if summary_logger:
        summary_logger.info(
            f"Pareto sweep complete. Data: {output_dir}, plots: {view_dir}"
        )

    return pareto_df, all_stats, baseline_stats

# ============================================================================
# POST-PROCESSING (per bucket)
# ============================================================================

def run_post_processing(cfg: RunConfig) -> None:
    """Existing per-bucket post-processing, looped across the 3 buckets."""
    print("\n" + "=" * 80)
    print("POST PROCESSING (per bucket)")
    print("=" * 80)

    for bucket in cfg.cost_scenarios:
        bucket_root = cfg.bucket_output_dir(bucket)
        for loft_val in cfg.loft_probs:
            vis_folder = os.path.join(
                bucket_root, 'pareto_vis',
                f'budgets{budgets_tag(cfg.budgets)}_loft{loft_val}',
            )
            os.makedirs(vis_folder, exist_ok=True)
            try:
                print(' post proc pareto')
                post_proc_pareto(
                    BUDGETS=cfg.budgets,
                    EQUITY_FLOORS=cfg.equity_floors,
                    LOFT_VALUE=loft_val,
                    BASE_PATH=bucket_root,
                    OUTPUT_PATH=vis_folder,
                    MIP_GAP=cfg.mip_gap,
                )
            except Exception as e:
                print(f"  [warn] post_proc_pareto failed for "
                      f"{bucket} loft={loft_val}: {e}")
        
        if cfg.epc_run:
            print('starting epc post proce')
            for equity_floor in cfg.equity_floors:
                for loft_val in cfg.loft_probs:
                    for budget in cfg.budgets:
                        try:
                            # run_epc_vis(
                            #     bucket_root,
                            #     base_dir_outputs=os.path.join(
                            #         bucket_root, 'greedy_vis_epc_pareto',
                            #     ),
                            #     million_budget=budget / MILLION_FACTOR,
                            #     prob_loft=loft_val,
                            #     equity_floor=equity_floor,
                            # )
                            run_epc_vis(
                                bucket_root,
                                base_dir_outputs=os.path.join(bucket_root, 'greedy_vis_epc_pareto'),
                                million_budget=budget / MILLION_FACTOR,
                                prob_loft=loft_val,
                                equity_floor=equity_floor,
                                mip_gap=cfg.mip_gap,   # add this
                            )
                        except Exception as e:
                            print(f"  [warn] run_epc_vis failed: {e}")


# ============================================================================
# MAIN
# ============================================================================
from src.ParetoPaths import (
    SliceKey,
    bucket_data_dir,
    slice_data_dir,
    summary_csv,
    selected_csv,
    selected_dir,
    baseline_csv,
    baseline_summary_csv,
    epc_summary_csv,
    epc_selected_csv,
    per_scenario_per_budget_dir,
    slice_log_dir,
)
from src.ParetoManifest import (
    new_manifest,
    write_manifest,
    read_manifest,
    record_slice,
    is_slice_recorded,
    assert_compatible_with_cfg,
    ManifestSchemaError,
)


def main() -> None:
    cfg = resolve_config()
    _print_run_header(cfg)

    manifest = _open_or_create_manifest(cfg)

    if cfg.run_greedy_runs:
        _run_all_slices(cfg, manifest)
    else:
        print('Set to skip runs (RUN_GREEDY_RUNS_YN=N).')

    if len(cfg.cost_scenarios) >= 2:
        run_comparison(cfg)
    else:
        print(f"\n[skip comparison] Only {len(cfg.cost_scenarios)} bucket(s) "
              f"configured; nothing to compare.")

    run_post_processing(cfg)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)


def _open_or_create_manifest(cfg: RunConfig):
    """Resume an existing run or start a fresh one."""
    try:
        manifest = read_manifest(cfg.root)
    except ManifestSchemaError as e:
        print(f"\n[FATAL] {e}")
        print("Run scripts/migrate_pareto_outputs.py to upgrade the run, "
              "or delete the folder to start fresh.")
        sys.exit(2)

    if manifest is None:
        manifest = new_manifest(cfg)
        write_manifest(cfg.root, manifest)
        print(f"\nStarted new run: {manifest.run_id}")
    else:
        assert_compatible_with_cfg(manifest, cfg)
        print(f"\nResuming run: {manifest.run_id} "
              f"({len(manifest.slices)} slices already recorded)")

    return manifest


def _run_all_slices(cfg: RunConfig, manifest) -> None:
    """The main solve loop, one slice at a time."""
    print("\nLoading personas...")
    personas = load_personas().drop_duplicates()

    # Outer loop is loft, then bucket — preserves cache locality on
    # per_run_dataset, which is bucket-scoped and expensive to open.
    for prob_loft in cfg.loft_probs:
        for bucket in cfg.cost_scenarios:
            if _bucket_loft_fully_recorded(cfg, manifest, bucket, prob_loft):
                print(f"\n[SKIP] All budgets recorded for "
                      f"loft={prob_loft}, bucket={bucket}.")
                continue

            df, df_buildings, sample_info, per_run_dataset = (
                load_and_prepare_data(cfg, prob_loft, bucket, personas)
            )
            if df.empty:
                print(f"[{bucket}] No data; skipping.")
                continue

            _greedy_sanity_print(df, bucket)

            for budget in cfg.budgets:
                key = SliceKey(bucket=bucket, budget=budget, loft=prob_loft)

                if not cfg.force_rerun and is_slice_recorded(cfg.root, key):
                    print(f"[SKIP] {key.slug} already in manifest")
                    continue

                _run_one_slice(
                    cfg, key, df, df_buildings, per_run_dataset,
                    sample_info=sample_info,
                )

            del df, df_buildings, per_run_dataset
            gc.collect()


def _bucket_loft_fully_recorded(
    cfg: RunConfig, manifest, bucket: str, loft: float,
) -> bool:
    """Cheap pre-check: every budget at this (bucket, loft) is in manifest."""
    if cfg.force_rerun or cfg.test_mode:
        return False
    return all(
        SliceKey(bucket=bucket, budget=b, loft=loft).slug in manifest.slices
        for b in cfg.budgets
    )


def _run_one_slice(
    cfg: RunConfig,
    key: SliceKey,
    df: pd.DataFrame,
    df_buildings: pd.DataFrame,
    per_run_dataset,
    sample_info: Optional[dict],
) -> None:
    """Solve one (bucket, budget, loft) slice and record it."""
    output_dir = slice_data_dir(cfg.root, key.bucket, key.budget, key.loft)
    log_dir = slice_log_dir(cfg.root, key.bucket, key.budget, key.loft)
    view_dir = per_scenario_per_budget_dir(
        cfg.root, key.bucket, key.budget, key.loft,
    )
    for d in (output_dir, log_dir, view_dir, selected_dir(
            cfg.root, key.bucket, key.budget, key.loft)):
        os.makedirs(d, exist_ok=True)

    summary_logger, detail_logger = setup_loggers(
        log_dir, key.budget, key.loft, bucket=key.bucket,
    )
    summary_logger.info(
        f'[{key.bucket}] Starting Pareto: '
        f'Budget £{key.budget:,}, Loft {key.loft}'
    )
    if cfg.test_mode and sample_info is not None:
        summary_logger.info(f'TEST_MODE sample_info: {sample_info}')

    _, all_stats, _ = run_pareto(
        df_all_packages=df,
        df_buildings=df_buildings,
        per_run_dataset=per_run_dataset,
        cfg=cfg,
        key=key,
        equity_floors=cfg.equity_floors,
        high_equity_personas=DEFAULT_HIGH_EQUITY_PERSONAS,
        view_dir=view_dir,
        detail_logger=detail_logger,
        summary_logger=summary_logger,
    )
    summary_logger.info("Pareto analysis complete!")
    print(f"✓ [{key.bucket}] Slice complete: {output_dir}")

    if cfg.epc_run:
        _run_epc_fallback(
            df, cfg, key,
            detail_logger=detail_logger,
            summary_logger=summary_logger,
            per_run_dataset=per_run_dataset,
        )

    n_solved = sum(1 for s in all_stats if s.get('status') in ('Optimal', 'Not Solved'))
    n_infeasible = len(all_stats) - n_solved
    record_slice(
        cfg.root, key,
        n_equity_floors_solved=n_solved,
        n_equity_floors_infeasible=n_infeasible,
        epc_mode=cfg.epc_run,
    )


def _print_run_header(cfg: RunConfig) -> None:
    print("\n" + "=" * 80)
    print("PARETO KNAPSACK ANALYSIS — PER COST_SCENARIO")
    print(f"  Mode:                {'EPC' if cfg.epc_run else 'standard'}")
    print(f"  Cost scenarios:      {cfg.cost_scenarios}")
    print(f"  High-equity personas: {DEFAULT_HIGH_EQUITY_PERSONAS}")
    print(f"  Budgets:             "
          f"{[budget_slug(b) for b in cfg.budgets]}")
    print(f"  Loft probs:          {cfg.loft_probs}")
    print(f"  Equity floors:       {cfg.equity_floors}")
    print(f"  Force rerun:         {cfg.force_rerun}")
    print(f"  Test mode:           {cfg.test_mode}"
          + (f" (n={cfg.test_sample_size}, seed={cfg.test_seed})"
             if cfg.test_mode else ''))
    print("=" * 80)


def _greedy_sanity_print(df: pd.DataFrame, bucket: str) -> None:
    """Diagnostic £1M greedy print. Side-effecting only, never raises."""
    try:
        best = preselect_best_cpt(
            df, upn_col='upn', cost_col=COST_COL, carbon_col=CARBON_COL,
        )
    except Exception:
        return
    if best.empty:
        return
    best['_is_high_eq'] = best['meta_socio_persona'].isin(
        DEFAULT_HIGH_EQUITY_PERSONAS
    )
    best = best.sort_values(COST_COL)
    best['cumcost'] = best[COST_COL].cumsum()
    picked = best[best['cumcost'] <= 1_000_000]
    if picked.empty or picked[COST_COL].sum() <= 0:
        return
    pct = (
        100 * picked.loc[picked['_is_high_eq'], COST_COL].sum()
        / picked[COST_COL].sum()
    )
    print(f"[{bucket}] Greedy £1M sanity: {len(picked)} bldgs, "
          f"{picked['_is_high_eq'].sum()} high-eq ({pct:.1f}% spend)")

if __name__ == "__main__":
    main()