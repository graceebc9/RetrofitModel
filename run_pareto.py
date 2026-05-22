"""
Pareto Knapsack Retrofit Analysis — with portfolio-level uncertainty.

Changes vs. previous version:
  - Reads the new decomposed-uncertainty preprocessing outputs:
      mean_*, aleatoric_std_*, epistemic_std_* per metric per scenario.
  - Loads the per-run building means parquet artefacts and uses them to
    compute portfolio-level epistemic uncertainty for every (budget,
    equity_floor) solution. Aleatoric uncertainty is computed in closed
    form from the selected rows.
  - Stats dicts now carry portfolio-level std fields plus a percentile-
    based £/tCO2 envelope (median, P16, P84) derived from the per-run
    portfolio totals — sidesteps the ratio-of-Gaussians problem.
  - `pareto_full.json` keeps the per-run portfolio totals per equity floor
    so plots can be reproduced or extended downstream.
  - Pareto-front and £/tCO2 plots show uncertainty bands.
  - `preselect_best_cpt` is invoked on the new aleatoric-sigma column so
    the baseline lives in the same risk regime as the preprocessing filter.

Optimiser objective unchanged: knapsack runs on the means
(`mean_total_capex`, `mean_total_co2_saved`). The marginal-building filter
lives upstream in preprocessing via `capex_per_net_ton_aleatoric_sigma`.
"""

from __future__ import annotations

import os
import sys
import glob
import gc
import json
import logging
import datetime
from dataclasses import dataclass
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
RHO = 0.45  # Discount factor used in post-processing NPV calculations.

# Selection-score column name from the new preprocessing.
# Used by `preselect_best_cpt` so the baseline is consistent with the
# preprocessing's risk-adjusted filter on cost-per-tonne.
SELECTION_SCORE_COL = 'capex_per_net_ton_aleatoric_sigma'

# Cost / carbon column names used by the optimiser objective.
COST_COL = 'mean_total_capex'
CARBON_COL = 'mean_total_co2_saved'

# Aleatoric-std columns used for portfolio-level closed-form aleatoric std.
COST_ALEATORIC_STD_COL = 'aleatoric_std_total_capex'
CARBON_ALEATORIC_STD_COL = 'aleatoric_std_total_co2_saved'

# Filename pattern of the per-run means parquet artefacts.
PER_RUN_MEANS_GLOB = 'per_run_means/per_run_means_*.parquet'


class EPCSelectionError(RuntimeError):
    """Raised when the EPC random-selection fallback produces no results."""


# ============================================================================
# BUDGET NAME FORMATTING — single source of truth
# ============================================================================

def budget_label(budget: int) -> str:
    """
    Canonical folder-name label for a budget in pounds.

    £1_000_000       -> '1'
    £2_500_000       -> '2.5'
    £10_000_000      -> '10'
    £100_000_000     -> '100'
    £500_000         -> '0.5'
    """
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
    input_files_path: str
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
    verbose: bool = True

    @property
    def pareto_runs_folder(self) -> str:
        parts = [self.base_dir]
        if self.epc_run:
            parts.append('pareto_runs_epc' if not self.test_mode else 'pareto_runs')
        else:
            parts.append('pareto_runs')
        parts.append(self.setting_name + ('_epc' if self.epc_run and self.test_mode else ''))
        if self.test_mode:
            parts.append(f'samples_{self.test_sample_size}')
        return os.path.join(*parts)

    @property
    def per_run_means_glob(self) -> str:
        """Where to find per-run means parquets, derived from the input path."""
        # input_files_path is a glob like '.../processed_all_scenarios/*'.
        # The per-run artefacts sit alongside in 'per_run_means/'.
        input_dir = os.path.dirname(self.input_files_path.rstrip('/*'))
        return os.path.join(input_dir, 'processed_all_scenarios', PER_RUN_MEANS_GLOB)


# Path table: (environment, epc_run) -> (input_glob, base_dir)
PATHS = {
    ('local', True): (
        '/Volumes/T9/2025_10_RetrofitModel/14_new_runs_compressed/1_all_int_epc/'
        'risk_sigma_1.0/processed_all_scenarios/*',
        '/Volumes/T9/2025_10_RetrofitModel/14_new_runs_compressed/2_greedy_results/NE/all_domestic',
    ),
    ('local', False): (
        '/Volumes/T9/2025_10_RetrofitModel/14_new_runs_compressed/1_all_interventions/'
        'risk_sigma_1.0/processed_all_scenarios/*',
        '/Volumes/T9/2025_10_RetrofitModel/14_new_runs_compressed/2_greedy_results/NE/all_domestic',
    ),
    ('hpc', True): (
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/'
        '2_optimized_priorities_epc/risk_sigma_1.0/processed_all_scenarios/*',
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v9/NE/epc',
    ),
    ('hpc', False): (
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/'
        '4_optimized_priorities/risk_sigma_1.0/processed_all_scenarios/*',
        '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/5_pareto/v9/NE/all_domestic',
    ),
}

# high mips / slower
LOCAL_DEFAULTS = dict(
    budgets=[1_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000],
    loft_probs=[0.65, 0.95],
    equity_floors=[0, 25, 50, 75, 100],
)

HPC_DEFAULTS = dict(
    budgets=[1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000],
    loft_probs=[0.95, 0.65],
    equity_floors=[0, 10, 25, 35, 50, 60, 75, 100],
)


def resolve_config() -> RunConfig:
    """Resolve environment variables + host into a single config object."""
    running_locally = not is_running_on_hpc()
    env_key = 'local' if running_locally else 'hpc'
    epc_run = os.getenv('EPC_YN', 'N').upper() == 'Y'
    test_mode = os.getenv('TEST_MODE', 'N').upper() == 'Y'

    input_path, base_dir = PATHS[(env_key, epc_run)]
    defaults = LOCAL_DEFAULTS if running_locally else HPC_DEFAULTS
    setting_name = 'local' if running_locally else 'v10'
    if test_mode:
        setting_name = f'{setting_name}_TEST'

    return RunConfig(
        setting_name=setting_name,
        input_files_path=input_path,
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


def _per_run_dataset(cfg: RunConfig):
    """
    Open the per-run parquet artefacts as a lazy pyarrow Dataset rather
    than loading them into memory. At full scale these files together
    can be tens of GB, but we only need a tiny slice (the selected upns
    for one solve) at a time.

    Returns None if no files exist, so callers can fall back to
    aleatoric-only uncertainty without crashing.
    """
    import pyarrow.dataset as ds  # local import: optional dep at runtime

    pattern = cfg.per_run_means_glob
    print(pattern)
    files = glob.glob(pattern)
    if not files:
        print(f"  [warn] No per-run means files matched {pattern}")
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
    Pull only the rows needed for one selection: the (upn, scenario)
    pairs the optimiser chose. Uses pyarrow predicate-pushdown +
    column projection so the read scans the parquet without
    materialising the whole frame.

    Returns an empty wide-format DataFrame if anything is missing.
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

    # Pre-filter on upn AND scenario at the parquet layer. This is the
    # whole point of the lazy read -- both predicates are pushed down so
    # we only materialise rows that match.
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

    # Final inner join on (upn, scenario) — the parquet may have rows
    # for non-selected scenarios per upn even after the scenario isin
    # filter, when those upns also appear in selected_df with different
    # scenarios. The cheap join here keeps semantics identical to the
    # old eager-load path.
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
    """
    Closed-form aleatoric std of the portfolio total for a metric:
        sigma_P^ale = sqrt(sum_i sigma_i^2)
    where sigma_i is each selected building's per-metric aleatoric std.

    Returns 0.0 if the column is missing or the frame is empty.
    """
    if selected_df.empty or ale_col not in selected_df.columns:
        return 0.0
    return float(np.sqrt(np.sum(selected_df[ale_col].fillna(0).to_numpy() ** 2)))


def _portfolio_epistemic_totals_wide(
    per_run_slice: pd.DataFrame,
    value_col: str,
) -> np.ndarray:
    """
    Given a wide-format per-run slice already restricted to the selected
    (upn, scenario) pairs, sum `value_col` across selections within each
    run_idx to get one portfolio total per epistemic world.

    Returns
    -------
    np.ndarray of shape (n_runs,). Empty array if input is empty or
    `value_col` is missing.
    """
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


def compute_portfolio_uncertainty(
    selected_df: pd.DataFrame,
    per_run_dataset,
    upn_col: str = 'upn',
    intervention_col: str = 'intervention',
) -> dict:
    """
    Compute portfolio-level uncertainty fields for one optimiser solution.

    Aleatoric: closed-form sum-of-variances over selected rows. Scales
    like sqrt(n) with portfolio size, washes out at scale.

    Epistemic: for each epistemic run, sum the per-run building means
    across the selected (upn, scenario) pairs to get one portfolio
    total per run. The std across runs captures the cross-building
    correlation induced by shared global parameters automatically -- no
    correlation modelling needed.

    £/tCO2 is reported as percentiles of the per-run ratio
    (cost_total_r / carbon_total_r), avoiding the ratio-of-Gaussians
    problem.

    Parameters
    ----------
    selected_df : DataFrame
        Optimiser output (one row per selected (upn, scenario)).
    per_run_dataset : pyarrow.dataset.Dataset or None
        Lazy handle to the wide-format per-run parquets. Only the rows
        matching `selected_df` are read.

    Returns a dict suitable for merging into the optimiser's stats dict.
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
    }

    if selected_df is None or selected_df.empty:
        return out

    # ----- Aleatoric (closed form on selected rows) -----
    out['total_cost_aleatoric_std'] = _portfolio_aleatoric_std(
        selected_df, COST_ALEATORIC_STD_COL
    )
    out['total_abatement_aleatoric_std'] = _portfolio_aleatoric_std(
        selected_df, CARBON_ALEATORIC_STD_COL
    )

    # ----- Epistemic (lazy slice + per-run totals) -----
    per_run_slice = _per_run_slice_for_selection(
        per_run_dataset,
        selected_df,
        upn_col=upn_col,
        intervention_col=intervention_col,
    )
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

    # ----- Epistemic share of total variance, per metric -----
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

    # ----- Percentile-based £/tCO2 envelope from per-run ratios -----
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
# PARETO SUMMARY PLOTS
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
    """matplotlib >=3.9 safe cmap accessor."""
    try:
        return plt.colormaps.get_cmap(name).resampled(n)
    except AttributeError:
        return plt.cm.get_cmap(name, n)


def plot_pareto_summary(all_stats, baseline_stats, output_dir, budget):
    """Generate summary plots from the Pareto sweep results."""
    feasible = [s for s in all_stats if s['status'] in ('Optimal', 'Not Solved')]
    if not feasible:
        print("No feasible solutions to plot.")
        return

    eq_floors = [s['equity_floor_pct'] for s in feasible]

    def save_fig(fig, name):
        fig.tight_layout()
        path = os.path.join(output_dir, f'pareto_{name}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved {name}.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 1: Pareto front — total abatement vs equity floor
    # Now with aleatoric (light) and epistemic (darker) uncertainty bands.
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        abatements = np.array([s['total_abatement'] for s in feasible])
        ale_std = np.array([
            s.get('total_abatement_aleatoric_std', 0.0) or 0.0 for s in feasible
        ])
        epi_std = np.array([
            s.get('total_abatement_epistemic_std', 0.0) or 0.0 for s in feasible
        ])

        # Bands centred on the mean abatement.
        ax.fill_between(
            eq_floors,
            abatements - epi_std, abatements + epi_std,
            color='#1976d2', alpha=0.18, label='± epistemic σ',
        )
        ax.fill_between(
            eq_floors,
            abatements - ale_std, abatements + ale_std,
            color='#1976d2', alpha=0.45, label='± aleatoric σ',
        )
        ax.plot(eq_floors, abatements, 'o-', color='#1976d2', linewidth=2,
                markersize=7, label='Multi-choice knapsack (mean)')

        if baseline_stats.get('total_abatement'):
            ax.axhline(
                baseline_stats['total_abatement'], color='#d32f2f',
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f"Baseline (pre-select): {baseline_stats['total_abatement']:.0f} tCO2",
            )

        ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
        ax.set_ylabel('Total CO₂ abatement (tonnes)', fontsize=11)
        ax.set_title(f'Pareto Front — Budget £{budget/1e6:.1f}M',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'front_abatement')
    except Exception as e:
        print(f"  Plot 1 failed: {e}")

    # ------------------------------------------------------------------
    # Plot 2: £/tCO2 vs equity floor — percentile band from per-run ratios
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        # Mean line for backwards comparability.
        cpex_mean = [s.get('cpex_per_ton') for s in feasible]
        ax.plot(eq_floors, cpex_mean, 's-', color='#f57c00', linewidth=2,
                markersize=7, label='£/tCO₂ (mean of totals)')

        # Percentile band where available.
        med = np.array([s.get('cpex_per_ton_median', np.nan) for s in feasible],
                       dtype=float)
        p16 = np.array([s.get('cpex_per_ton_p16', np.nan) for s in feasible],
                       dtype=float)
        p84 = np.array([s.get('cpex_per_ton_p84', np.nan) for s in feasible],
                       dtype=float)
        if not np.all(np.isnan(med)):
            ax.fill_between(
                eq_floors, p16, p84,
                color='#f57c00', alpha=0.20,
                label='P16–P84 across epistemic runs',
            )
            ax.plot(eq_floors, med, 'd--', color='#bf360c', linewidth=1.5,
                    markersize=5, label='Median across runs')

        if baseline_stats.get('cpex_per_ton'):
            ax.axhline(
                baseline_stats['cpex_per_ton'], color='#d32f2f',
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f"Baseline: £{baseline_stats['cpex_per_ton']:,.0f}/t",
            )

        ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
        ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
        ax.set_title(f'Cost-Effectiveness vs Equity — Budget £{budget/1e6:.1f}M',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'front_cpex')
    except Exception as e:
        print(f"  Plot 2 failed: {e}")

    personas_order = ['high_risk', 'med_risk', 'middle_risk', 'low_risk', 'v_low_risk']
    width = 0.7

    # ------------------------------------------------------------------
    # Plot 3-5: Persona splits (unchanged)
    # ------------------------------------------------------------------
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
            ax.set_title(f'Persona Split — {key.title()} — Budget £{budget/1e6:.1f}M',
                         fontsize=13, fontweight='bold')
            ax.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            save_fig(fig, suffix)
        except Exception as e:
            print(f"  Plot {plot_idx} failed: {e}")

    # ------------------------------------------------------------------
    # Decile splits (unchanged)
    # ------------------------------------------------------------------
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
            ax.set_title(f'Decile Split — {key.title()} — Budget £{budget/1e6:.1f}M',
                         fontsize=13, fontweight='bold')
            ax.legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            save_fig(fig, suffix)
        except Exception as e:
            print(f"  Plot {plot_idx} failed: {e}")

    # ------------------------------------------------------------------
    # Plot 6: Intervention mix (unchanged)
    # ------------------------------------------------------------------
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
            ax.set_title(f'Intervention Mix — Budget £{budget/1e6:.1f}M',
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

def setup_loggers(output_dir, million_budget, prob_loft):
    """
    Create fresh file loggers for this run. Clears any stale handlers from
    prior iterations in the same Python process to avoid duplicate writes.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_logger = logging.getLogger(f'summary_{million_budget}_{prob_loft}')
    summary_logger.handlers.clear()
    summary_logger.setLevel(logging.INFO)
    summary_logger.propagate = False
    sh = logging.FileHandler(os.path.join(output_dir, f'summary_log_{timestamp}.log'))
    summary_logger.addHandler(sh)

    detail_logger = logging.getLogger(f'detail_{million_budget}_{prob_loft}')
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
    """
    Draw a stratified sample of *buildings* (UPNs), jointly stratified on:
      - persona (from the personas table, joined on postcode),
      - intervention-package menu (the set of interventions available
        for that building in `df`), and
      - gas percentile decile.
    """
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
        _log(f"  [sampler] {n_missing_persona} buildings have no persona "
             f"(postcode not in personas table) — excluded from sample.")
        building_df = building_df.dropna(subset=[persona_col])

    n_missing_decile = building_df[decile_col].isna().sum()
    if n_missing_decile > 0:
        _log(f"  [sampler] {n_missing_decile} buildings have no decile "
             f"recorded — excluded from sample.")
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
        _log(f"  [sampler] {n_strata} non-empty (persona × menu × decile) strata "
             f"across {n_buildings_total:,} buildings.")

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

        _log(f"  [sampler] Target {target_n} buildings → realised "
             f"{len(sampled_upns):,} buildings across "
             f"{sum(1 for v in alloc.values() if v > 0)} strata.")

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
         f"(from {len(df):,}); sampled UPNs: "
         f"{sampled_df[upn_col].nunique():,}")

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

def run_is_complete(output_dir: str, equity_floors: list) -> bool:
    """Return True if a (budget, loft_prob) run already produced its outputs."""
    summary_path = os.path.join(output_dir, 'pareto_summary.csv')
    full_path = os.path.join(output_dir, 'pareto_full.json')
    baseline_path = os.path.join(output_dir, 'baseline_preselect.csv')

    for p in (summary_path, full_path, baseline_path):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False

    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return False

    if summary_df.empty or 'status' not in summary_df.columns:
        return False

    feasible = summary_df[summary_df['status'].isin(['Optimal', 'Not Solved'])]
    if feasible.empty:
        return False

    if equity_floors and 'equity_floor_pct' in summary_df.columns:
        if equity_floors[0] not in summary_df['equity_floor_pct'].values:
            return False

    return True


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def load_and_prepare_data(
    cfg: RunConfig,
    prob_loft: float,
    personas: pd.DataFrame,
):
    """
    Load raw package data and open the per-run dataset handle for one
    loft probability.

    Returns
    -------
    df : DataFrame
        Package-level frame with personas joined.
    df_buildings : DataFrame
        Building-level view.
    sample_info : dict or None
        Diagnostics from test-mode sampling, else None.
    per_run_dataset : pyarrow.dataset.Dataset or None
        Lazy handle to the wide-format per-run parquets. None if files
        are missing — uncertainty fields will then report aleatoric only.
    """
    files = [x for x in glob.glob(cfg.input_files_path) if f'loft_{prob_loft}' in x]
    print(cfg.input_files_path)
    print(f'\nFound {len(files)} files for loft prob {prob_loft}')
    if not files:
        return pd.DataFrame(), pd.DataFrame(), None, None

    print("\nLoading input data...")
    res_df = load_data_simple(files)
    print(f'res_df shape: {res_df.shape}, n_upns: {res_df["upn"].nunique()}')

    print("\nOpening per-run dataset (lazy) for epistemic propagation...")
    per_run_dataset = _per_run_dataset(cfg)
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
        print(f"\n[TEST_MODE] Sampling ~{cfg.test_sample_size} buildings...")
        n_before = res_df['upn'].nunique()
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
        print(f"[TEST_MODE] Scaled budgets: "
              f"{[f'£{b/1e6:.2f}M' for b in cfg.budgets]}")
        # No need to filter the per-run dataset -- the per-call slice
        # filters by selected upns at read time anyway.

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
    print(f"After premise filtering: {len(df):,} rows "
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
    sample_info: Optional[dict],
) -> None:
    """Run the Pareto sweep for every budget at a fixed loft probability."""
    for budget in cfg.budgets:
        output_dir = os.path.join(
            cfg.pareto_runs_folder,
            f'budget_{budget_label(budget)}M__loft_{prob_loft}__mip_{mip_gap}',
        )
        os.makedirs(output_dir, exist_ok=True)

        if (not cfg.force_rerun and not cfg.test_mode
                and run_is_complete(output_dir, cfg.equity_floors)):
            print(f"\n[SKIP] Budget £{budget_label(budget)}M "
                  f"loft={prob_loft} already complete: {output_dir}")
            continue

        summary_logger, detail_logger = setup_loggers(
            output_dir, budget_label(budget), prob_loft,
        )
        summary_logger.info(
            f'Starting Pareto: Budget £{budget:,}, Loft {prob_loft}'
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
        )
        summary_logger.info("Pareto analysis complete!")
        print(f"✓ Results saved to: {output_dir}")

        if cfg.epc_run:
            _run_epc_fallback(df, budget, output_dir, detail_logger, summary_logger)


def _run_epc_fallback(
    df: pd.DataFrame, budget: int, output_dir: str,
    detail_logger: logging.Logger, summary_logger: logging.Logger,
) -> None:
    epc_path = os.path.join(output_dir, 'epc_random_selection.csv')
    selected, remaining = select_epc_algo_pareto(
        df_knapsack=df, budget=budget,
        cost_col=COST_COL,
        carbon_col=CARBON_COL,
        logger=detail_logger,
    )
    if selected.empty:
        detail_logger.info('EPC selection empty')
        raise EPCSelectionError(
            f'EPC random selection produced no rows at budget £{budget:,}'
        )
    selected['remaining_funds'] = remaining
    selected.to_csv(epc_path, index=False)
    summary_logger.info(f"EPC random saved to: {epc_path}")


def run_post_processing(cfg: RunConfig) -> None:
    print("\n" + "=" * 80)
    print("POST PROCESSING")
    print("=" * 80)

    for loft_val in cfg.loft_probs:
        vis_folder = os.path.join(
            cfg.pareto_runs_folder, 'pareto_vis',
            f'budgets{budgets_tag(cfg.budgets)}_loft{loft_val}',
        )
        os.makedirs(vis_folder, exist_ok=True)
        post_proc_pareto(
            BUDGETS=cfg.budgets,
            EQUITY_FLOORS=cfg.equity_floors,
            LOFT_VALUE=loft_val,
            BASE_PATH=cfg.pareto_runs_folder,
            OUTPUT_PATH=vis_folder,
            MIP_GAP=cfg.mip_gap,
        )

    if cfg.epc_run:
        for equity_floor in cfg.equity_floors:
            for loft_val in cfg.loft_probs:
                for budget in cfg.budgets:
                    run_epc_vis(
                        cfg.pareto_runs_folder,
                        base_dir_outputs=os.path.join(
                            cfg.pareto_runs_folder, 'greedy_vis_epc_pareto',
                        ),
                        million_budget=budget / MILLION_FACTOR,
                        prob_loft=loft_val,
                        equity_floor=equity_floor,
                    )


# ============================================================================
# PARETO RUNNER
# ============================================================================

def run_pareto(
    df_all_packages,
    df_buildings,
    per_run_dataset,
    budget,
    equity_floors,
    high_equity_personas,
    output_dir,
    loft_prob,
    mip_gap,
    cost_col=COST_COL,
    carbon_col=CARBON_COL,
    upn_col='upn',
    intervention_col='intervention',
    persona_col='meta_socio_persona',
    time_limit_seconds=600,
    detail_logger=None,
    summary_logger=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if summary_logger:
        summary_logger.info(
            f"Starting Pareto sweep: budget=£{budget:,.0f}, "
            f"equity_floors={equity_floors}, "
            f"high_equity_personas={high_equity_personas}"
        )

    # ------------------------------------------------------------------
    # 1. Pareto sweep
    # ------------------------------------------------------------------
    all_stats = []
    for eps in equity_floors:
        print(f"\n{'='*60}")
        print(f"Equity floor: {eps}% of spend to {high_equity_personas}")
        print(f"{'='*60}")

        selected_df, stats = multichoice_knapsack(
            df_all_packages=df_all_packages,
            budget=budget,
            equity_floor_pct=eps,
            mip_gap=mip_gap,
            high_equity_personas=high_equity_personas,
            upn_col=upn_col,
            persona_col=persona_col,
            cost_col=cost_col,
            carbon_col=carbon_col,
            time_limit_seconds=time_limit_seconds,
            logger=detail_logger,
        )

        # ----- Portfolio-level uncertainty (lazy slice on per-run parquets) -----
        unc = compute_portfolio_uncertainty(
            selected_df=selected_df,
            per_run_dataset=per_run_dataset,
            upn_col=upn_col,
            intervention_col=intervention_col,
        )
        stats.update(unc)
        if summary_logger:
            summary_logger.info(
                f"  eq={eps}% uncertainty: "
                f"abatement σ_ale={unc['total_abatement_aleatoric_std']:.1f}, "
                f"σ_epi={unc['total_abatement_epistemic_std']:.1f}, "
                f"epistemic_share={unc['epistemic_share_carbon']}"
            )

        all_stats.append(stats)

        if not selected_df.empty:
            eq_label = f"{eps:.0f}"
            selected_path = os.path.join(
                output_dir, f'selected_projects_eq{eq_label}.csv'
            )
            selected_df.to_csv(selected_path, index=False)

        if stats["status"] not in ("Optimal", "Not Solved"):
            print(f"  Infeasible at {eps}% — stopping sweep.")
            if summary_logger:
                summary_logger.info(f"Infeasible at equity_floor={eps}%")
            break

    # ------------------------------------------------------------------
    # 2. Baseline (preselect best aleatoric-penalised £/tCO2 per building)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BASELINE: pre-select best aleatoric-σ £/tCO2 per building")
    print(f"{'='*60}")

    try:
        df_preselected = preselect_best_cpt(
            df_all_packages, upn_col=upn_col,
            cost_col=cost_col, carbon_col=carbon_col,
            score_col=SELECTION_SCORE_COL,
        )
    except TypeError:
        df_preselected = preselect_best_cpt(
            df_all_packages, upn_col=upn_col,
            cost_col=cost_col, carbon_col=carbon_col,
        )

    baseline_selected, baseline_stats = multichoice_knapsack(
        df_all_packages=df_preselected,
        budget=budget,
        equity_floor_pct=0,
        mip_gap=mip_gap,
        high_equity_personas=high_equity_personas,
        upn_col=upn_col,
        persona_col=persona_col,
        cost_col=cost_col,
        carbon_col=carbon_col,
        time_limit_seconds=time_limit_seconds,
        logger=detail_logger,
    )
    baseline_stats["method"] = "pre_select_best_cpt"
    baseline_unc = compute_portfolio_uncertainty(
        selected_df=baseline_selected,
        per_run_dataset=per_run_dataset,
        upn_col=upn_col,
        intervention_col=intervention_col,
    )
    baseline_stats.update(baseline_unc)
    baseline_selected.to_csv(
        os.path.join(output_dir, 'baseline_preselect.csv'), index=False
    )

    # ------------------------------------------------------------------
    # 3. Save summary CSV (now with std and percentile columns)
    # ------------------------------------------------------------------
    pareto_df = pd.DataFrame(all_stats)
    summary_cols = [
        "equity_floor_pct", "status", "n_retrofitted", "n_high_equity",
        "total_cost", "total_abatement", "cpex_per_ton",
        "total_cost_aleatoric_std", "total_cost_epistemic_std",
        "total_abatement_aleatoric_std", "total_abatement_epistemic_std",
        "epistemic_share_cost", "epistemic_share_carbon",
        "cpex_per_ton_p16", "cpex_per_ton_median", "cpex_per_ton_p84",
        "high_eq_spend_pct", "high_eq_abatement_pct", "solve_time_s",
    ]
    available_cols = [c for c in summary_cols if c in pareto_df.columns]
    pareto_df[available_cols].to_csv(
        os.path.join(output_dir, 'pareto_summary.csv'), index=False
    )

    # Full JSON keeps the per-run portfolio totals so plots can be redone.
    with open(os.path.join(output_dir, 'pareto_full.json'), 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'baseline_stats.json'), 'w') as f:
        json.dump(baseline_stats, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # 4. Plots (now with uncertainty bands)
    # ------------------------------------------------------------------
    plot_pareto_summary(all_stats, baseline_stats, output_dir, budget)

    # ------------------------------------------------------------------
    # 5. Console summary
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print("PARETO FRONT SUMMARY")
    print(f"{'#'*60}")
    print(pareto_df[available_cols].to_string(index=False))

    if baseline_stats.get("total_abatement"):
        print(f"\nBaseline (pre-select): "
              f"{baseline_stats['total_abatement']:.1f} tCO2, "
              f"£{baseline_stats['cpex_per_ton']:,.0f}/t, "
              f"{baseline_stats['high_eq_spend_pct']:.1f}% high-eq spend")
        if all_stats and all_stats[0].get("total_abatement"):
            improvement = (
                (all_stats[0]["total_abatement"] - baseline_stats["total_abatement"])
                / baseline_stats["total_abatement"] * 100
            )
            print(f"  Multi-choice improvement: +{improvement:.1f}% abatement "
                  f"vs pre-select method")

    if summary_logger:
        summary_logger.info(f"Pareto sweep complete. Results in: {output_dir}")

    return pareto_df, all_stats, baseline_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    running_locally = not is_running_on_hpc()

    epc_yn = os.getenv('EPC_YN')
    epc_run = epc_yn == 'Y'
    print('Running greedy for EPC' if epc_run else 'Running greedy for normal')
    
    # --- NEW: HPC Array / Single Input Logic ---
    # These will be strings from os.getenv, so we convert them if they exist
    env_budget = os.getenv('SINGLE_BUDGET')
    env_loft = os.getenv('SINGLE_LOFT_PROB')

    run_g_yn = os.getenv('RUN_GREEDY_RUNS_YN')
    run_greedy_runs = run_g_yn != 'N'

    if TEST_MODE:
        print("\n" + "!" * 80)
        print(f"!! TEST_MODE ENABLED — stratified sample of "
              f"~{TEST_SAMPLE_SIZE} buildings (seed={TEST_SEED})")
        print(f"!! Outputs will be written to a separate `_TEST` folder.")
        print("!" * 80)

    # Configuration
    if running_locally:
        setting_name = 'local'
        budgets = [1_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000]
        loft_probs = [0.65]
        equity_floors = list(range(0, 105, 25))

        if epc_run:
            INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities_epc/risk_sigma_1.0/processed_best_only/*'
            BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/5_greedy_results_epc/NE/all_domestic'
        else:
            INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities/risk_sigma_1.0/processed_best_only/*'
            INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/1_all_interventions/risk_sigma_1.0/processed_all_scenarios/*'

            BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/2_greedy_results/NE/all_domestic'
    else:
        setting_name = 'v10'
        budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
        loft_probs = [0.95, 0.65]
        equity_floors = list(range(0, 105, 50))

        if epc_run:
            INPUT_FILES_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/2_optimized_priorities_epc/risk_sigma_1.0/processed_all_scenarios/*'
            BASE_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v9/NE/epc'
        else:
            # FIX: was pointing at local /Volumes/T9 — clearly wrong on HPC.
            INPUT_FILES_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_optimized_priorities/risk_sigma_1.0/processed_all_scenarios/*'
            BASE_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/5_pareto/v9/NE/all_domestic'

        print(f'Starting {INPUT_FILES_PATH}')

    
    # OVERRIDE with single values if provided by the HPC job script
    if env_budget:
        budgets = [int(float(env_budget))]
        print(f"HPC Input: Single Budget set to £{budgets[0]:,}")
    
    if env_loft:
        loft_probs = [float(env_loft)]
        print(f"HPC Input: Single Loft Prob set to {loft_probs[0]}")
        
    # In test mode, redirect outputs to a `_TEST` sibling folder so we
    # never clobber a real run's artefacts.
    if TEST_MODE:
        setting_name = f'{setting_name}_TEST'

    input_files = glob.glob(INPUT_FILES_PATH)
    pareto_runs_folder = os.path.join(BASE_DIR, 'pareto_runs', setting_name)

    print("\n" + "=" * 80)
    print("PARETO KNAPSACK ANALYSIS — ε-CONSTRAINT ON EQUITY SPEND")
    print(f"  Mode:                {'EPC' if cfg.epc_run else 'standard'}")
    print(f"  High-equity personas: {DEFAULT_HIGH_EQUITY_PERSONAS}")
    print(f"  Budgets:             {[budget_label(b) + 'M' for b in cfg.budgets]}")
    print(f"  Loft probs:          {cfg.loft_probs}")
    print(f"  Equity floors:       {cfg.equity_floors}")
    print(f"  Force rerun:         {cfg.force_rerun}")
    print(f"  Test mode:           {cfg.test_mode}"
          + (f" (n={cfg.test_sample_size}, seed={cfg.test_seed})"
             if cfg.test_mode else ''))
    print("=" * 80)

    if cfg.run_greedy_runs:
        print("\nLoading personas...")
        personas = load_personas().drop_duplicates()

        for prob_loft in cfg.loft_probs:
            # Cheap early skip: all budgets done for this loft?
            if not cfg.force_rerun and not cfg.test_mode:
                all_done = all(
                    run_is_complete(
                        os.path.join(
                            cfg.pareto_runs_folder,
                            f'budget_{budget_label(b)}M__loft_{prob_loft}__mip_{cfg.mip_gap}',
                        ),
                        cfg.equity_floors,
                    )
                    for b in cfg.budgets
                )
                if all_done:
                    print(f"\n[SKIP] All budgets complete for loft={prob_loft}. "
                          f"Set FORCE_RERUN=Y to redo.")
                    continue

            df, df_buildings, sample_info, per_run_dataset = load_and_prepare_data(
                cfg, prob_loft, personas,
            )
            if df.empty:
                continue

            # Diagnostic: trivial £1M greedy as a sanity print.
            try:
                best = preselect_best_cpt(
                    df, upn_col='upn', cost_col=COST_COL, carbon_col=CARBON_COL,
                )
            except Exception:
                best = pd.DataFrame()
            if not best.empty:
                best['_is_high_eq'] = best['meta_socio_persona'].isin(
                    DEFAULT_HIGH_EQUITY_PERSONAS
                )
                best = best.sort_values(COST_COL)
                best['cumcost'] = best[COST_COL].cumsum()
                picked = best[best['cumcost'] <= 1_000_000]
                if not picked.empty and picked[COST_COL].sum() > 0:
                    print(
                        f"Greedy picks (£1M sanity): {len(picked)} bldgs, "
                        f"{picked['_is_high_eq'].sum()} high-eq "
                        f"({100 * picked.loc[picked['_is_high_eq'], COST_COL].sum() / picked[COST_COL].sum():.1f}% spend)"
                    )

            run_all_budgets(
                cfg, df, df_buildings, per_run_dataset=per_run_dataset,
                prob_loft=prob_loft, sample_info=sample_info,
                mip_gap=cfg.mip_gap,
            )
    else:
        print('Set to skip runs (RUN_GREEDY_RUNS_YN=N).')

    run_post_processing(cfg)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()