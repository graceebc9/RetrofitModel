"""
Pareto Knapsack Retrofit Analysis — refactored.

Changes vs. previous version:
  - `million_budget` / `budgets_tag` now produced by one helper (fixes
    the str.replace('.0', '') bug that silently mangled round-million
    budget folder names).
  - Config resolved once into a dataclass; no more duplicated branches.
  - `main` decomposed into small, testable functions.
  - Diagnostic prints behind a `verbose` flag.
  - Specific exception types.
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
RHO = 0.45  # Discount factor used in post-processing NPV calculations.


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
    £10_000_000      -> '10'      (previously '1' — BUG)
    £100_000_000     -> '100'     (previously '1' — BUG)
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


# Path table: (environment, epc_run) -> (input_glob, base_dir)
PATHS = {
    ('local', True): (
        '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/1_all_int_epc/'
        'risk_sigma_1.0/processed_all_scenarios/*',
        '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/2_greedy_results/NE/all_domestic',
    ),
    ('local', False): (
        '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/1_all_interventions/'
        'risk_sigma_1.0/processed_all_scenarios/*',
        '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/2_greedy_results/NE/all_domestic',
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
# low mips 
# LOCAL_DEFAULTS = dict(
#     budgets=[1_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000],
#     loft_probs=[0.95, 0.65],
#     equity_floors=[0, 10, 25, 35, 50, 60, 75, 85, 100],
# )

# high mips /slower 
LOCAL_DEFAULTS = dict(
    budgets=[1_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000],
    loft_probs=[0.95, 0.65],
    equity_floors=[0,  25,  50,   75, 100],
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
        mip_gap =float(os.getenv('MIP_GAP', '0.01')), 
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

    # Plot 1: Pareto front — total abatement vs equity floor
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        abatements = [s['total_abatement'] for s in feasible]
        ax.plot(eq_floors, abatements, 'o-', color='#1976d2', linewidth=2,
                markersize=7, label='Multi-choice knapsack')
        if baseline_stats.get('total_abatement'):
            ax.axhline(baseline_stats['total_abatement'], color='#d32f2f',
                       linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f"Baseline (pre-select): {baseline_stats['total_abatement']:.0f} tCO2")
        ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
        ax.set_ylabel('Total CO₂ abatement (tonnes)', fontsize=11)
        ax.set_title(f'Pareto Front — Budget £{budget/1e6:.1f}M', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'front_abatement')
    except Exception as e:
        print(f"  Plot 1 failed: {e}")

    # Plot 2: £/tCO2 vs equity floor
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        cpex = [s['cpex_per_ton'] for s in feasible]
        ax.plot(eq_floors, cpex, 's-', color='#f57c00', linewidth=2, markersize=7,
                label='Multi-choice knapsack')
        if baseline_stats.get('cpex_per_ton'):
            ax.axhline(baseline_stats['cpex_per_ton'], color='#d32f2f',
                       linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f"Baseline: £{baseline_stats['cpex_per_ton']:,.0f}/t")
        ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
        ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
        ax.set_title(f'Cost-Effectiveness vs Equity — Budget £{budget/1e6:.1f}M',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'front_cpex')
    except Exception as e:
        print(f"  Plot 2 failed: {e}")

    personas_order = ['high_risk', 'med_risk', 'middle_risk', 'low_risk', 'v_low_risk']
    width = 0.7

    # Plot 3–5: Persona splits
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
            ax.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            save_fig(fig, suffix)
        except Exception as e:
            print(f"  Plot {plot_idx} failed: {e}")


 

    # 1. Define deciles 1-10 and a color palette
    deciles_order = list(range(1, 11))
    # Use a sequential colormap (like YlGnBu or viridis) to represent the 1-10 scale
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
            width = 0.75

            # 2. Iterate over deciles 1 through 10
            for i, d in enumerate(deciles_order):
                # Pulling from s['avg_gas_percentile'] which now contains decile keys (1, 2, etc.)
                vals = [s['percentile_breakdown'].get(d, {}).get(key, 0) / divisor
                        for s in feasible]
                
                ax.bar(x, vals, width, bottom=bottom,
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
            
            # Place legend on the side because 10 decile labels take up space
            ax.legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            save_fig(fig, suffix)
        except Exception as e:
            print(f"  Plot {plot_idx} failed: {e}")
    
    # Plot 6: Intervention mix
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

    Parameters
    ----------
    df : DataFrame
        The package-level frame (one row per UPN × intervention).
    personas : DataFrame
        Personas keyed by postcode.
    target_n : int
        Target sample size in *buildings* (UPNs). The realised sample
        may be slightly larger because each stratum contributes at
        least `min_per_stratum` buildings (when available).
    upn_col, postcode_col, persona_col, intervention_col, decile_col : str
        Column names.
    min_per_stratum : int
        Minimum buildings per stratum when the stratum is non-empty.
    seed : int
        RNG seed.
    logger : logging.Logger or None

    Returns
    -------
    sampled_df : DataFrame
        All rows of `df` whose UPN is in the sampled set.
    sample_info : dict
        Diagnostics: stratum counts, realised sample size, etc.
    """
    rng = np.random.default_rng(seed)

    def _log(msg):
        print(msg)
        if logger is not None:
            logger.info(msg)

    # 1. Build per-building stratum labels.
    #    menu = sorted tuple of distinct interventions available to that UPN.
    menus = (
        df.groupby(upn_col)[intervention_col]
          .apply(lambda s: tuple(sorted(s.unique())))
          .rename('menu')
    )

    # Extract building-level features (postcode AND decile)
    upn_features = (
        df[[upn_col, postcode_col, decile_col]]
          .drop_duplicates(subset=[upn_col])
          .set_index(upn_col)
    )

    building_df = pd.concat([menus, upn_features], axis=1).reset_index()

    # Attach persona via postcode.
    personas_small = personas[[postcode_col, persona_col]].drop_duplicates(
        subset=[postcode_col]
    )
    building_df = building_df.merge(personas_small, on=postcode_col, how='left')

    # Handle missing personas
    n_missing_persona = building_df[persona_col].isna().sum()
    if n_missing_persona > 0:
        _log(f"  [sampler] {n_missing_persona} buildings have no persona "
             f"(postcode not in personas table) — excluded from sample.")
        building_df = building_df.dropna(subset=[persona_col])

    # Handle missing deciles
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
        # 2. Form strata (Persona x Menu x Decile)
        building_df['_stratum'] = list(
            zip(building_df[persona_col], building_df['menu'], building_df[decile_col])
        )
        strata_sizes = building_df['_stratum'].value_counts()
        n_strata = len(strata_sizes)
        _log(f"  [sampler] {n_strata} non-empty (persona × menu × decile) strata "
             f"across {n_buildings_total:,} buildings.")

        # 3. Proportional allocation with a floor.
        floor_alloc = {
            s: min(min_per_stratum, sz) for s, sz in strata_sizes.items()
        }
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

                alloc = {
                    s: floor_alloc[s] + floored[s]
                    for s in strata_sizes.index
                }
                alloc = {
                    s: min(alloc[s], strata_sizes[s])
                    for s in strata_sizes.index
                }

        # 4. Draw per stratum.
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

    # Diagnostics
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
    """
    Scale budgets proportionally to the sample fraction so the test run
    is solvable quickly but still exercises the constraint. A floor of
    £100k ensures we don't collapse to a trivial budget.
    """
    if n_total == 0:
        return budgets
    frac = n_sampled / n_total
    scaled = [max(100_000, int(b * frac)) for b in budgets]
    # Deduplicate while preserving order.
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
    """
    Return True if a (budget, loft_prob) run has already produced its
    expected outputs in `output_dir`.
    """
    summary_path = os.path.join(output_dir, 'pareto_summary.csv')
    full_path = os.path.join(output_dir, 'pareto_full.json')
    baseline_path = os.path.join(output_dir, 'baseline_preselect.csv')

    for p in (summary_path, full_path, baseline_path):
        if not os.path.exists(p):
            return False
        if os.path.getsize(p) == 0:
            return False

    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return False

    if summary_df.empty:
        return False

    if 'status' not in summary_df.columns:
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
    cfg: RunConfig, prob_loft: float, personas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Load raw package data for one loft probability, apply sampling (if
    test mode), validate, merge personas, filter premises, and build
    the building-level view.

    Returns
    -------
    df : DataFrame
        Package-level frame with personas joined.
    df_buildings : DataFrame
        Building-level view.
    sample_info : dict or None
        Diagnostics from test-mode sampling, else None.
    """
    files = [x for x in glob.glob(cfg.input_files_path) if f'loft_{prob_loft}' in x]
    print(f'\nFound {len(files)} files for loft prob {prob_loft}')
    if not files:
        return pd.DataFrame(), pd.DataFrame(), None

    print("\nLoading input data...")
    res_df = load_data_simple(files)
    print(f'res_df shape: {res_df.shape}, n_upns: {res_df["upn"].nunique()}')

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
        # Scale budgets to keep the small ones binding on the sample.
        cfg.budgets = _scale_budgets_for_test(
            cfg.budgets, n_sampled=n_after, n_total=n_before,
        )
        print(f"[TEST_MODE] Scaled budgets: "
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
    print(f"After premise filtering: {len(df):,} rows "
          f"({df['upn'].nunique():,} buildings)")

    df_buildings = build_building_level_view(df, upn_col='upn')  # noqa: F405
    return df, df_buildings, sample_info


def run_all_budgets(
    cfg: RunConfig, df: pd.DataFrame, df_buildings: pd.DataFrame, mip_gap:float , 
    prob_loft: float, sample_info: Optional[dict],
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
            budget=budget,
            mip_gap=mip_gap ,
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
        cost_col='mean_total_capex',
        carbon_col='mean_total_co2_saved',
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
            RHO=RHO,
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
    budget,
    equity_floors,
    high_equity_personas,
    output_dir,
    loft_prob,
    mip_gap,
    cost_col='mean_total_capex',
    carbon_col='mean_total_co2_saved',
    upn_col='upn',
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

    # 1. Run Pareto sweep
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
        all_stats.append(stats)

        if not selected_df.empty:
            eq_label = f"{eps:.0f}"
            selected_path = os.path.join(output_dir, f'selected_projects_eq{eq_label}.csv')
            selected_df.to_csv(selected_path, index=False)
            try:
                None
                # plot_greedy_distribution_analysis(
                #     baseline_df=df_buildings,
                #     selected_df=selected_df,
                #     scenario_name=f'pareto_eq{eq_label}_loft{loft_prob}',
                #     output_dir=output_dir,
                # )
            except Exception as e:
                print(f"  Plot failed for eq={eps}: {e}")

        if stats["status"] not in ("Optimal", "Not Solved"):
            print(f"  Infeasible at {eps}% — stopping sweep.")
            if summary_logger:
                summary_logger.info(f"Infeasible at equity_floor={eps}%")
            break

    # 2. Baseline
    print(f"\n{'='*60}")
    print("BASELINE: pre-select best £/tCO2 per building (old method)")
    print(f"{'='*60}")

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
    baseline_selected.to_csv(os.path.join(output_dir, 'baseline_preselect.csv'), index=False)

    # 3. Save summary
    pareto_df = pd.DataFrame(all_stats)
    summary_cols = [
        "equity_floor_pct", "status", "n_retrofitted", "n_high_equity",
        "total_cost", "total_abatement", "cpex_per_ton",
        "high_eq_spend_pct", "high_eq_abatement_pct", "solve_time_s",
    ]
    available_cols = [c for c in summary_cols if c in pareto_df.columns]
    pareto_df[available_cols].to_csv(
        os.path.join(output_dir, 'pareto_summary.csv'), index=False
    )

    with open(os.path.join(output_dir, 'pareto_full.json'), 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'baseline_stats.json'), 'w') as f:
        json.dump(baseline_stats, f, indent=2, default=str)

    # 4. Plots
    plot_pareto_summary(all_stats, baseline_stats, output_dir, budget)

    # 5. Print summary
    print(f"\n{'#'*60}")
    print("PARETO FRONT SUMMARY")
    print(f"{'#'*60}")
    print(pareto_df[available_cols].to_string(index=False))

    if baseline_stats.get("total_abatement"):
        print(f"\nBaseline (old method): "
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

def main() -> None:
    cfg = resolve_config()

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
                            f'budget_{budget_label(b)}M__loft_{prob_loft}',
                        ),
                        cfg.equity_floors,
                    )
                    for b in cfg.budgets
                )
                if all_done:
                    print(f"\n[SKIP] All budgets complete for loft={prob_loft}. "
                          f"Set FORCE_RERUN=Y to redo.")
                    continue

            df, df_buildings, sample_info = load_and_prepare_data(
                cfg, prob_loft, personas,
            )
            if df.empty:
                continue

             
            best = preselect_best_cpt(df, upn_col='upn',
                                    cost_col='mean_total_capex',
                                    carbon_col='mean_total_co2_saved')
            best['_is_high_eq'] = best['meta_socio_persona'].isin(DEFAULT_HIGH_EQUITY_PERSONAS)

            # At £1M budget the solver will pick the cheapest-£/tCO2 packages.
            # Sort and take them greedily until budget runs out.
            best = best.sort_values('mean_total_capex') # or by cpt
            best['cumcost'] = best['mean_total_capex'].cumsum()
            picked = best[best['cumcost'] <= 1_000_000]
            print(f"Greedy picks: {len(picked)} bldgs, "
                f"{picked['_is_high_eq'].sum()} high-eq "
                f"({100*picked.loc[picked['_is_high_eq'], 'mean_total_capex'].sum()/picked['mean_total_capex'].sum():.1f}% spend)")

            run_all_budgets(cfg, df, df_buildings, prob_loft=prob_loft, sample_info=sample_info, mip_gap=cfg.mip_gap)
    else:
        print('Set to skip runs (RUN_GREEDY_RUNS_YN=N).')

    run_post_processing(cfg)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# """
# Greedy/Exact Algorithm Analysis for Retrofit Scenarios
# =======================================================
# UPDATED: Multi-choice knapsack with ε-constraint on equity.

# Key changes from previous version:
#   - No longer pre-selects "best" package per building — the solver
#     jointly picks buildings AND packages.
#   - Replaces equity_factor weighting with ε-constraint:
#     "at least X% of total spend must go to high/med risk personas".
#   - Sweeps equity_floor_pct to trace the Pareto front.
#   - Saves per-sweep results + Pareto summary.
#   - Skips a (budget, loft_prob) combination if pareto_summary.csv
#     already exists for it (unless FORCE_RERUN=Y).
#   - NEW: TEST_MODE support — run on a stratified sample of buildings
#     (stratified jointly on persona and intervention-package menu) to
#     sanity-check the pipeline end-to-end before kicking off a full run.
# """


# import os
# import sys
# import glob
# import gc
# import json
# import logging
# import datetime
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Add custom module path
# sys.path.append('/Users/gracecolverd/RetrofitModel')

# # from src.GreedyAlgo import plot_greedy_distribution_analysis
# from src.personas import load_personas
# from src.utils import is_running_on_hpc
# # from src.EPCAlgo import select_epc_algo
# from src.PartetoEpc import select_epc_algo_pareto
# from src.GreedyEpcVis import run_epc_vis
# from src.PostPareto import post_proc_pareto
# from src.ParetoUtills import *

# from src.ParetoKnapsack import (
#     multichoice_knapsack,
#     preselect_best_cpt,
#     DEFAULT_HIGH_EQUITY_PERSONAS,
# )

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# MILLION_FACTOR = 1_000_000
# RHO = 0.45

# # FORCE_RERUN=Y to redo completed runs. Default is to skip.
# FORCE_RERUN = os.getenv("FORCE_RERUN", "N").upper() == "Y"

# # ---------------------------------------------------------------------------
# # TEST MODE
# # ---------------------------------------------------------------------------
# # TEST_MODE=Y       → run on a small stratified sample of buildings.
# # TEST_SAMPLE_SIZE  → target number of buildings in the sample (default 500).
# # TEST_MIN_PER_STRATUM → at least this many buildings per (persona, menu)
# #                        stratum, subject to availability (default 1).
# # TEST_SEED         → RNG seed for reproducibility (default 42).
# #
# # Outputs go to a separate folder (suffix `_TEST`) so they never clobber
# # a real run. Budgets are also scaled down (see `_scale_budgets_for_test`).
# TEST_MODE = os.getenv("TEST_MODE", "N").upper() == "Y"
# TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", "500"))
# TEST_MIN_PER_STRATUM = int(os.getenv("TEST_MIN_PER_STRATUM", "1"))
# TEST_SEED = int(os.getenv("TEST_SEED", "42"))


# def load_data_simple(files):
#     res = []
#     for f in files:
#         df = pd.read_csv(f)
#         res.append(df)
#     return pd.concat(res)





# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     running_locally = not is_running_on_hpc()

#     epc_yn = os.getenv('EPC_YN')
#     epc_run = epc_yn == 'Y'
#     print('Running greedy for EPC' if epc_run else 'Running greedy for normal')

#     run_g_yn = os.getenv('RUN_GREEDY_RUNS_YN')
#     run_greedy_runs = run_g_yn != 'N'

#     if TEST_MODE:
#         print("\n" + "!" * 80)
#         print(f"!! TEST_MODE ENABLED — stratified sample of "
#               f"~{TEST_SAMPLE_SIZE} buildings (seed={TEST_SEED})")
#         print(f"!! Outputs will be written to a separate `_TEST` folder.")
#         print("!" * 80)

#     # Configuration
#     if running_locally:
#         setting_name = 'local'
#         budgets = [1_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000]
#         # budgets = [ 200_000_000]
#         loft_probs = [0.95, 0.65]
#         # loft_probs = [0.65]
#         # equity_floors = list(range(10, 95, 25))
#         equity_floors = [0,10, 25, 35, 50, 60, 75,100] 
#         # equity_floors = [0,    60, 100] 
#         # equity_floors = [75]

#         if epc_run:
#             INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities_epc/risk_sigma_1.0/processed_best_only/*'
#             INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/1_all_int_epc/risk_sigma_1.0/processed_all_scenarios/*'
            
#             BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/2_greedy_results/NE/all_domestic'
#         else:
#             INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities/risk_sigma_1.0/processed_best_only/*'
#             INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/1_all_interventions/risk_sigma_1.0/processed_all_scenarios/*'

#             BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/2_greedy_results/NE/all_domestic'
#     else:
#         setting_name = 'v10'
#         budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
#         loft_probs = [0.95, 0.65]
#         # equity_floors = list(range(0, 105, 50)
#         equity_floors = [0, 10, 25, 35, 50, 60, 75,100] 

#         if epc_run:
#             INPUT_FILES_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/2_optimized_priorities_epc/risk_sigma_1.0/processed_all_scenarios/*'
#             BASE_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v9/NE/epc'
#         else:
#             # FIX: was pointing at local /Volumes/T9 — clearly wrong on HPC.
#             INPUT_FILES_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_optimized_priorities/risk_sigma_1.0/processed_all_scenarios/*'
#             BASE_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/5_pareto/v9/NE/all_domestic'

#         print(f'Starting {INPUT_FILES_PATH}')

#     # In test mode, redirect outputs to a `_TEST` sibling folder so we
#     # never clobber a real run's artefacts.
#     if TEST_MODE:
#         setting_name = f'{setting_name}_TEST'

#     input_files = glob.glob(INPUT_FILES_PATH)
#     if TEST_MODE:
#            if epc_run:
#                pareto_runs_folder = os.path.join(BASE_DIR, 'pareto_runs', f'{setting_name}_epc', f'samples_{str(TEST_SAMPLE_SIZE)}'  )
#            else:
#                pareto_runs_folder = os.path.join(BASE_DIR, 'pareto_runs', setting_name, f'samples_{str(TEST_SAMPLE_SIZE)}'  )
#     else:
#         if epc_run:
#             pareto_runs_folder = os.path.join(BASE_DIR, 'pareto_runs_epc', setting_name   )
#         else:
#             pareto_runs_folder = os.path.join(BASE_DIR, 'pareto_runs', setting_name   )

#     print("\n" + "=" * 80)
#     print("PARETO KNAPSACK ANALYSIS — ε-CONSTRAINT ON EQUITY SPEND")
#     print(f"  High-equity personas: {DEFAULT_HIGH_EQUITY_PERSONAS}")
#     print(f"  Equity floors: {equity_floors}")
#     print(f"  Force rerun: {FORCE_RERUN}")
#     print(f"  Test mode:   {TEST_MODE}")
#     print("=" * 80)

#     if run_greedy_runs:
#         for prob_loft in loft_probs:
#             # --------------------------------------------------------------
#             # SKIP CHECK: if *all* budgets for this loft are already done,
#             # skip the expensive data load. (Disabled in test mode — we
#             # always want to run the test end-to-end.)
#             # --------------------------------------------------------------
#             if not FORCE_RERUN and not TEST_MODE:
#                 all_done = True
#                 for budget in budgets:
#                     million_budget = str(budget / MILLION_FACTOR).replace('.0', '')
#                     output_dir = os.path.join(
#                         pareto_runs_folder,
#                         f'budget_{million_budget}M__loft_{prob_loft}'
#                     )
#                     if not run_is_complete(output_dir, equity_floors):
#                         all_done = False
#                         break
#                 if all_done:
#                     print(f"\n[SKIP] All budgets already complete for loft={prob_loft}. "
#                           f"Set FORCE_RERUN=Y to redo.")
#                     continue

#             files_to_use = [x for x in input_files if f'loft_{prob_loft}' in x]
#             print(f'\nFound {len(files_to_use)} files with loft prob {prob_loft}')

#             if not files_to_use:
#                 print(f"  No input files for loft={prob_loft}, skipping.")
#                 continue

#             print("\nLoading input data...")
#             res_df = load_data_simple(files_to_use)
#             print(f'res_df shape: {res_df.shape}')
#             print(f'num upns: {res_df.upn.nunique()}')

#             print("\nLoading personas...")
#             personas = load_personas()
#             personas = personas.drop_duplicates()
#             print('peronas loded')

#             # ----------------------------------------------------------------
#             # TEST MODE — stratified sample on (persona, intervention menu)
#             # ----------------------------------------------------------------
#             if TEST_MODE:
#                 print(f"\n[TEST_MODE] Stratified-sampling ~{TEST_SAMPLE_SIZE} "
#                       f"buildings (seed={TEST_SEED})...")
#                 n_upns_before = res_df['upn'].nunique()
#                 res_df, sample_info = stratified_sample_buildings(
#                     df=res_df,
#                     personas=personas,
#                     target_n=TEST_SAMPLE_SIZE,
#                     upn_col='upn',
#                     postcode_col='postcode',
#                     persona_col='meta_socio_persona',
#                     intervention_col='intervention',
#                     min_per_stratum=TEST_MIN_PER_STRATUM,
#                     seed=TEST_SEED,
#                 )
#                 n_upns_after = res_df['upn'].nunique()
#                 print(f"[TEST_MODE] Sampled {n_upns_after:,} / {n_upns_before:,} "
#                       f"buildings ({res_df.shape[0]:,} rows).")

#                 # Scale budgets so at least the small ones are feasible on
#                 # the mini-sample; the large ones will just retrofit the
#                 # whole sample, which is a useful smoke-test too.
#                 budgets = _scale_budgets_for_test(
#                     budgets, n_sampled=n_upns_after, n_total=n_upns_before
#                 )
#                 print(f"[TEST_MODE] Scaled budgets: "
#                       f"{[f'£{b/1e6:.2f}M' for b in budgets]}")

#             # validations
#             per_building = res_df.groupby('upn')['intervention'].apply(set)

#             # How often do the loft packages co-occur?
#             has_loft_decay = per_building.apply(lambda s: 'joint_heat_loft_decay' in s)
#             has_loft_install = per_building.apply(lambda s: 'loft_installation' in s)

#             print(pd.crosstab(has_loft_decay, has_loft_install,
#                             rownames=['has_loft_decay'], colnames=['has_loft_install']))
#             has_wall_decay = per_building.apply(lambda s: 'joint_heat_wall_decay' in s)
#             has_wall_install = per_building.apply(lambda s: 'wall_installation' in s)
#             print(pd.crosstab(has_wall_decay, has_wall_install,
#                             rownames=['has_wall_decay'], colnames=['has_wall_install']))
#             wall_install_buildings = per_building[per_building.apply(
#                 lambda s: 'wall_installation' in s
#             )]
#             print(f"Buildings with wall_installation: {len(wall_install_buildings)}")

#             also_wall_decay = wall_install_buildings.apply(
#                 lambda s: 'joint_heat_wall_decay' in s
#             ).sum()
#             print(f"  of which also have joint_heat_wall_decay: {also_wall_decay}")

#             wall_decay_buildings = per_building[per_building.apply(
#                 lambda s: 'joint_heat_wall_decay' in s and 'wall_installation' not in s
#             )]
#             print(f"Buildings with wall_decay but no wall_install: {len(wall_decay_buildings)}")

#             PKG_COL = 'intervention'

#             print(f"Distinct interventions: {res_df[PKG_COL].nunique()}")
#             print(res_df[PKG_COL].value_counts())

#             building_menus = (
#                 res_df.groupby('upn')[PKG_COL]
#                 .apply(lambda s: tuple(sorted(s.unique())))
#             )
#             menu_counts = building_menus.value_counts()
#             print(f"\nDistinct intervention menus: {len(menu_counts)}")
#             print(f"Menu size → count of buildings:")
#             print(building_menus.apply(len).value_counts().sort_index())

#             print("\nAll menus (or top 30):")
#             for menu, n in menu_counts.head(30).items():
#                 print(f"  {n:>7,}  ({len(menu)}) {menu}")

#             upn_col = 'upn'
#             # Drop UPN collisions
#             upn_postcode_counts = res_df.groupby(upn_col)['postcode'].nunique()
#             bad_upns = upn_postcode_counts[upn_postcode_counts > 1].index
#             if len(bad_upns) > 0:
#                 before = len(res_df)
#                 res_df = res_df[~res_df[upn_col].isin(bad_upns)].reset_index(drop=True)
#                 print(f"UPN-postcode collisions:     "
#                     f"{len(bad_upns)} UPNs dropped ({before - len(res_df)} rows)")
#                 if len(bad_upns) > 100:
#                     raise ValueError(
#                         f"{len(bad_upns)} UPN-postcode collisions — "
#                         f"too many to be noise. Investigate upstream join."
#                     )
#             else:
#                 print("UPN-postcode collisions:     0")

#             res_df = validate_multipackage_input(
#                 res_df, personas,
#                 upn_col='upn',
#                 min_packages=MIN_PACKAGES_PER_BUILDING,
#                 max_packages=MAX_PACKAGES_PER_BUILDING,
#             )

#             df = res_df.merge(personas, on='postcode', how='inner')
#             validate_post_merge(df, upn_col='upn', max_packages=MAX_PACKAGES_PER_BUILDING)

#             df = df[df['premise_type'] != 'Domestic_outbuilding']
#             df = df[~df['premise_type'].isna()]
#             gc.collect()
#             print(f"After premise filtering: {len(df):,} rows ({df['upn'].nunique():,} buildings)")
#             df_buildings = build_building_level_view(df, upn_col='upn')

#             for budget in budgets:
#                 million_budget = str(budget / MILLION_FACTOR).replace('.0', '')
#                 output_dir = os.path.join(
#                     pareto_runs_folder,
#                     f'budget_{million_budget}M__loft_{prob_loft}'
#                 )
#                 os.makedirs(output_dir, exist_ok=True)

#                 # Per-budget skip check (disabled in test mode).
#                 if not FORCE_RERUN and not TEST_MODE and run_is_complete(output_dir, equity_floors):
#                     print(f"\n[SKIP] Budget £{million_budget}M loft={prob_loft} "
#                           f"already complete: {output_dir}")
#                     continue

#                 summary_logger, detail_logger = setup_loggers(
#                     output_dir, million_budget, prob_loft
#                 )
#                 summary_logger.info(
#                     f'Starting Pareto analysis: Budget £{budget:,}, '
#                     f'Loft Probability {prob_loft}'
#                 )
#                 if TEST_MODE:
#                     summary_logger.info(
#                         f'TEST_MODE sample_info: {sample_info}'
#                     )

#                 pareto_df, all_stats, baseline_stats = run_pareto(
#                     df_all_packages=df,
#                     df_buildings=df_buildings,
#                     budget=budget,
#                     equity_floors=equity_floors,
#                     high_equity_personas=DEFAULT_HIGH_EQUITY_PERSONAS,
#                     output_dir=output_dir,
#                     loft_prob=prob_loft,
#                     cost_col='mean_total_capex',
#                     carbon_col='mean_total_co2_saved',
#                     upn_col='upn',
#                     persona_col='meta_socio_persona',
#                     time_limit_seconds=600,
#                     detail_logger=detail_logger,
#                     summary_logger=summary_logger,
#                 )

#                 summary_logger.info("Pareto analysis complete!")
#                 print(f"✓ Results saved to: {output_dir}")

#                 if epc_run:
#                     epc_random_path = os.path.join(output_dir, 'epc_random_selection.csv')
#                     epc_random_selected_df, epc_random_remaining = select_epc_algo_pareto(
#                         df_knapsack=df,
#                         budget=budget,
#                         cost_col='mean_total_capex',
#                         # efficiency_column='capex_per_net_ton',
#                         carbon_col='mean_total_co2_saved',
#                         logger=detail_logger,
#                     )
#                     epc_random_selected_df['remaining_funds'] = epc_random_remaining
#                     if epc_random_selected_df.empty:
#                         detail_logger.info('EPC selection empty')
#                         raise Exception('EPC selection empty')
#                     epc_random_selected_df.to_csv(epc_random_path, index=False)
#                     summary_logger.info(f"EPC random saved to: {epc_random_path}")

#         print("\n" + "=" * 80)
#         print("PARETO RUNS COMPLETE!")
#         print("=" * 80)
#     else:
#         print('Set to skip runs.')

#     # POST PROCESSING
#     print("\n" + "=" * 80)
#     print("POST PROCESSING")
#     print("=" * 80)

#     for loft_val in loft_probs:
#         budgets_tag = '_'.join(str(int(b / MILLION_FACTOR)) for b in budgets)
#         viss_fold = os.path.join(
#             pareto_runs_folder, 'pareto_vis',
#             f'budgets{budgets_tag}M_loft{loft_val}'
#         )
#         os.makedirs(viss_fold, exist_ok=True)

#         post_proc_pareto(
#             BUDGETS=budgets,
#             EQUITY_FLOORS=equity_floors,
#             LOFT_VALUE=loft_val,
#             BASE_PATH=pareto_runs_folder,
#             OUTPUT_PATH=viss_fold,
#             RHO=RHO,
#         )

#     if epc_run:
#         for equity_floor in equity_floors:
#             for LOFT_VALUE in loft_probs:
#                 for budget in budgets:
#                     million_budget = budget / MILLION_FACTOR
#                     run_epc_vis(
#                         pareto_runs_folder,
#                         base_dir_outputs = os.path.join(pareto_runs_folder, 'greedy_vis_epc_pareto'),
#                         million_budget= million_budget,
#                          prob_loft=  LOFT_VALUE, 
#                           equity_floor=equity_floor,
#                     )

#     print("\n" + "=" * 80)
#     print("ALL ANALYSES COMPLETE!")
#     print("=" * 80)


# if __name__ == "__main__":
#     main()