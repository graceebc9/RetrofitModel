"""
Greedy/Exact Algorithm Analysis for Retrofit Scenarios
=======================================================
UPDATED: Multi-choice knapsack with ε-constraint on equity.

Key changes from previous version:
  - No longer pre-selects "best" package per building — the solver
    jointly picks buildings AND packages.
  - Replaces equity_factor weighting with ε-constraint: 
    "at least X% of total spend must go to high/med risk personas".
  - Sweeps equity_floor_pct to trace the Pareto front.
  - Saves per-sweep results + Pareto summary.
"""

import os
import sys
import glob
import gc
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add custom module path
sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.validate import validate
from src.GreedyAlgo import plot_greedy_distribution_analysis
from src.personas import load_personas
from src.RetrofitEquity import EQUITY_WEIGHTS
from src.utils import is_running_on_hpc
from src.EPCAlgo import select_epc_algo
from src.GreedyEpcVis import run_epc_vis
from src.PostPareto import post_proc_pareto 

# NEW: import the multi-choice knapsack solver
from src.ParetoKnapsack import (
    multichoice_knapsack,
    pareto_sweep,
    preselect_best_cpt,
    DEFAULT_HIGH_EQUITY_PERSONAS,
    ALL_PERSONAS,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

milion_factor = 1_000_000
RHO = 0.45

# Algorithm toggle: "greedy" uses old single-objective, "pareto" uses new
ALGO = os.getenv("ALGO", "pareto").lower()
assert ALGO in ("greedy", "exact", "pareto"), \
    f"ALGO must be 'greedy', 'exact', or 'pareto', got '{ALGO}'"


def load_data_simple(files):
    res = []
    for f in files:
        df = pd.read_csv(f)
        res.append(df)
    return pd.concat(res)


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


def plot_pareto_summary(all_stats, baseline_stats, output_dir, budget):
    """
    Generate summary plots from the Pareto sweep results.

    Produces 5 plots saved to output_dir:
      1. Pareto front: abatement vs equity floor
      2. £/tCO2 vs equity floor
      3. Persona split — buildings (stacked bar)
      4. Persona split — spend (stacked bar)
      5. Persona split — abatement (stacked bar)
      6. Intervention mix (stacked bar)
    """
    # Filter to feasible solutions only
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
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        abatements = [s['total_abatement'] for s in feasible]
        ax.plot(eq_floors, abatements, 'o-', color='#1976d2', linewidth=2,
                markersize=7, label='Multi-choice knapsack')

        # Baseline reference line
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

    # ------------------------------------------------------------------
    # Plot 2: £/tCO2 vs equity floor
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Plot 3: Persona split — number of buildings (stacked bar)
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        personas_order = ['high_risk', 'med_risk', 'middle_risk', 'low_risk', 'v_low_risk']
        x = np.arange(len(eq_floors))
        width = 0.7
        bottom = np.zeros(len(eq_floors))

        for p in personas_order:
            vals = [s['persona_breakdown'].get(p, {}).get('buildings', 0) for s in feasible]
            ax.bar(x, vals, width, bottom=bottom, label=PERSONA_LABELS.get(p, p),
                   color=PERSONA_COLORS.get(p, '#999'), edgecolor='white', linewidth=0.5)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.0f}%' for e in eq_floors], rotation=45)
        ax.set_xlabel('Equity floor', fontsize=11)
        ax.set_ylabel('Number of buildings retrofitted', fontsize=11)
        ax.set_title(f'Persona Split — Buildings — Budget £{budget/1e6:.1f}M',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'persona_buildings')
    except Exception as e:
        print(f"  Plot 3 failed: {e}")

    # ------------------------------------------------------------------
    # Plot 4: Persona split — spend (stacked bar)
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(eq_floors))
        bottom = np.zeros(len(eq_floors))

        for p in personas_order:
            vals = [s['persona_breakdown'].get(p, {}).get('spend', 0) / 1e6
                    for s in feasible]
            ax.bar(x, vals, width, bottom=bottom, label=PERSONA_LABELS.get(p, p),
                   color=PERSONA_COLORS.get(p, '#999'), edgecolor='white', linewidth=0.5)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.0f}%' for e in eq_floors], rotation=45)
        ax.set_xlabel('Equity floor', fontsize=11)
        ax.set_ylabel('Spend (£M)', fontsize=11)
        ax.set_title(f'Persona Split — Spend — Budget £{budget/1e6:.1f}M',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'persona_spend')
    except Exception as e:
        print(f"  Plot 4 failed: {e}")

    # ------------------------------------------------------------------
    # Plot 5: Persona split — abatement (stacked bar)
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(eq_floors))
        bottom = np.zeros(len(eq_floors))

        for p in personas_order:
            vals = [s['persona_breakdown'].get(p, {}).get('abatement', 0) for s in feasible]
            ax.bar(x, vals, width, bottom=bottom, label=PERSONA_LABELS.get(p, p),
                   color=PERSONA_COLORS.get(p, '#999'), edgecolor='white', linewidth=0.5)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.0f}%' for e in eq_floors], rotation=45)
        ax.set_xlabel('Equity floor', fontsize=11)
        ax.set_ylabel('CO₂ abatement (tonnes)', fontsize=11)
        ax.set_title(f'Persona Split — Abatement — Budget £{budget/1e6:.1f}M',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'persona_abatement')
    except Exception as e:
        print(f"  Plot 5 failed: {e}")

    # ------------------------------------------------------------------
    # Plot 6: Intervention mix (stacked bar)
    # ------------------------------------------------------------------
    try:
        # Collect all intervention names across all feasible solutions
        all_interventions = set()
        for s in feasible:
            all_interventions.update(s.get('intervention_breakdown', {}).keys())
        all_interventions = sorted(all_interventions)

        intv_cmap = plt.cm.get_cmap('tab10', len(all_interventions))
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
        ax.legend(title='Intervention', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        save_fig(fig, 'intervention_mix')
    except Exception as e:
        print(f"  Plot 6 failed: {e}")

    print("Pareto summary plots complete.")


# ============================================================================
# PARETO RUNNER (NEW)
# ============================================================================

def run_pareto(
    df_all_packages: pd.DataFrame,
    budget: float,
    equity_floors: list,
    high_equity_personas: set,
    output_dir: str,
    loft_prob: float,
    cost_col: str = 'mean_total_capex',
    carbon_col: str = 'mean_total_co2_saved',
    upn_col: str = 'upn',
    persona_col: str = 'meta_socio_persona',
    time_limit_seconds: int = 600,
    detail_logger=None,
    summary_logger=None,
):
    """
    Run the full Pareto sweep for a given budget and save all outputs.

    For each equity floor value, solves a multi-choice knapsack:
      - Picks at most one package per building (all packages considered)
      - Maximises total CO2 abatement
      - Subject to: total spend ≤ budget
      - Subject to: % spend on high-equity personas ≥ equity_floor

    Saves:
      - pareto_summary.csv: one row per equity floor with key metrics
      - pareto_full.json: full stats including persona/intervention breakdowns
      - selected_projects_eq{X}.csv: selected buildings for each equity floor
      - baseline_preselect.csv: old method result for comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    if summary_logger:
        summary_logger.info(
            f"Starting Pareto sweep: budget=£{budget:,.0f}, "
            f"equity_floors={equity_floors}, "
            f"high_equity_personas={high_equity_personas}"
        )

    # ------------------------------------------------------------------
    # 1. Run Pareto sweep (multi-choice knapsack, all packages)
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
            high_equity_personas=high_equity_personas,
            upn_col=upn_col,
            persona_col=persona_col,
            cost_col=cost_col,
            carbon_col=carbon_col,
            time_limit_seconds=time_limit_seconds,
            logger=detail_logger,
        )
        all_stats.append(stats)

        # Save selected projects for this equity floor
        if not selected_df.empty:
            eq_label = f"{eps:.0f}"
            selected_path = os.path.join(
                output_dir, f'selected_projects_eq{eq_label}.csv'
            )
            selected_df.to_csv(selected_path, index=False)

            # Generate distribution plots
            try:
                plot_greedy_distribution_analysis(
                    baseline_df=df_all_packages,
                    selected_df=selected_df,
                    scenario_name=f'pareto_eq{eq_label}_loft{loft_prob}',
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"  Plot failed for eq={eps}: {e}")

        # Stop if infeasible
        if stats["status"] not in ("Optimal", "Not Solved"):
            print(f"  Infeasible at {eps}% — stopping sweep.")
            if summary_logger:
                summary_logger.info(f"Infeasible at equity_floor={eps}%")
            break

    # ------------------------------------------------------------------
    # 2. Run baseline comparison (old pre-select method)
    # ------------------------------------------------------------------
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
        high_equity_personas=high_equity_personas,
        upn_col=upn_col,
        persona_col=persona_col,
        cost_col=cost_col,
        carbon_col=carbon_col,
        time_limit_seconds=time_limit_seconds,
        logger=detail_logger,
    )
    baseline_stats["method"] = "pre_select_best_cpt"

    baseline_path = os.path.join(output_dir, 'baseline_preselect.csv')
    baseline_selected.to_csv(baseline_path, index=False)

    # ------------------------------------------------------------------
    # 3. Save Pareto summary
    # ------------------------------------------------------------------
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

    # Full stats with breakdowns
    with open(os.path.join(output_dir, 'pareto_full.json'), 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)

    with open(os.path.join(output_dir, 'baseline_stats.json'), 'w') as f:
        json.dump(baseline_stats, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # 4. Pareto front plots — persona & intervention breakdown
    # ------------------------------------------------------------------
    plot_pareto_summary(all_stats, baseline_stats, output_dir, budget)

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print("PARETO FRONT SUMMARY")
    print(f"{'#'*60}")
    print(pareto_df[available_cols].to_string(index=False))

    if baseline_stats.get("total_abatement"):
        print(f"\nBaseline (old method): "
              f"{baseline_stats['total_abatement']:.1f} tCO2, "
              f"£{baseline_stats['cpex_per_ton']:,.0f}/t, "
              f"{baseline_stats['high_eq_spend_pct']:.1f}% high-eq spend")

        # Compare multi-choice vs pre-select at equity_floor=0
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
    print(f"\nAlgorithm: {ALGO.upper()}")

    running_locally = not is_running_on_hpc()

    epc_yn = os.getenv('EPC_YN')
    epc_run = epc_yn == 'Y'
    print('Running greedy for EPC' if epc_run else 'Running greedy for normal')

    run_g_yn = os.getenv('RUN_GREEDY_RUNS_YN')
    run_greedy_runs = run_g_yn != 'N'

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    if running_locally:
        setting_name = 'local'
        budgets = [1_000_000]
        budgets= [ 1_000_000,  25_000_000,  50_000_000, 100_000_000, 200_000_000] 
        loft_probs = [0.65]

        # NEW: equity floors replace equity_factors
        # "at least X% of spend must go to high_risk + med_risk buildings"
        equity_floors = list(range(0, 105, 5))  # 0%, 5%, 10%, ..., 100%

        if epc_run:
            INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities_epc/risk_sigma_1.0/processed_best_only/*'
            BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/5_greedy_results_epc/NE/all_domestic'
        else:
            INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities/risk_sigma_1.0/processed_best_only/*'
            BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/5_greedy_results/NE/all_domestic'
    else:
        setting_name = 'v10'

        if epc_run:
            INPUT_FILES_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/2_optimized_priorities_epc/risk_sigma_1.0/processed_best_only/*'
            BASE_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v9/NE/epc'
        else:
            INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/4_optimized_priorities/risk_sigma_1.0/processed_best_only/*'
            BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/5_greedy_results/NE/all_domestic'

        print(f'Starting {INPUT_FILES_PATH}')

        budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
        loft_probs = [0.95, 0.65]
        equity_floors = list(range(0, 105, 50))

    input_files = glob.glob(INPUT_FILES_PATH)

    # NEW: folder naming for Pareto results
    pareto_runs_folder = os.path.join(BASE_DIR, f'pareto_runs', setting_name)

    print("\n" + "=" * 80)
    print(f"PARETO KNAPSACK ANALYSIS — ε-CONSTRAINT ON EQUITY SPEND")
    print(f"  High-equity personas: {DEFAULT_HIGH_EQUITY_PERSONAS}")
    print(f"  Equity floors: {equity_floors}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Data loading + prep (same as before)
    # ------------------------------------------------------------------
    if run_greedy_runs:
        for prob_loft in loft_probs:
            files_to_use = [x for x in input_files if f'loft_{prob_loft}' in x]
            print(f'\nFound {len(files_to_use)} files with loft prob {prob_loft}')

            print("\nLoading input data...")
            res_df = load_data_simple(files_to_use)
            res_df = res_df.drop_duplicates()
            print(f'res_df shape: {res_df.shape}')
            print(f'num upns: {res_df.upn.nunique()}')

            print("\nLoading personas...")
            personas = load_personas()
            personas = personas.drop_duplicates()

            # === VALIDATION ===
            print("\n=== PRE-MERGE VALIDATION ===")
            print(f"res_df rows: {len(res_df)}")
            print(f"personas rows: {len(personas)}")

            res_dupes = res_df['upn'].duplicated().sum()
            print(f"Duplicate upn in res_df: {res_dupes}")
            if 0 < res_dupes < 50:
                res_df = res_df.drop_duplicates(subset='upn', keep='first')
                print("✓ Deduplicated")

            personas_dupes = personas['postcode'].duplicated().sum()
            print(f"Duplicate postcodes in personas: {personas_dupes}")

            common_postcodes = set(res_df['postcode']) & set(personas['postcode'])
            print(f"Postcodes in common: {len(common_postcodes)}")

            df = res_df.merge(personas, on='postcode', how='inner')
            print(f"\n=== POST-MERGE ===")
            print(f"After persona merge: {len(df)} rows")
            print(f"Unique UPNs: {df['upn'].nunique()}")

            if df['upn'].nunique() < len(df):
                print(f"⚠️ UPNs duplicated: {len(df) - df['upn'].nunique()} extra rows")

            # Filter
            df = df[df['premise_type'] != 'Domestic_outbuilding']
            df = df[~df['premise_type'].isna()]
            gc.collect()
            print(f"After filtering: {len(df)} rows")

            # ==============================================================
            # KEY CHANGE: Instead of pre-selecting one package per building 
            # and sweeping equity_factor, we now feed ALL packages to the 
            # solver and sweep equity_floor_pct.
            #
            # The input df should have multiple rows per upn (one per 
            # intervention/package). If your current data already has 
            # one-best-per-building, you need to go back to the step that 
            # loads all scenarios/packages.
            # ==============================================================

            for budget in budgets:
                million_budget = str(budget / milion_factor).replace('.0', '')

                output_dir = os.path.join(
                    pareto_runs_folder,
                    f'budget_{million_budget}M__loft_{prob_loft}'
                )
                os.makedirs(output_dir, exist_ok=True)

                # Set up logging — no longer depends on equity_factor
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                summary_logger = logging.getLogger(f'summary_{million_budget}_{prob_loft}')
                summary_logger.setLevel(logging.INFO)
                summary_handler = logging.FileHandler(
                    os.path.join(output_dir, f'summary_log_{timestamp}.log')
                )
                summary_logger.addHandler(summary_handler)

                detail_logger = logging.getLogger(f'detail_{million_budget}_{prob_loft}')
                detail_logger.setLevel(logging.INFO)
                detail_handler = logging.FileHandler(
                    os.path.join(output_dir, f'detail_log_{timestamp}.log')
                )
                detail_logger.addHandler(detail_handler)
                # Also print to console
                if not detail_logger.handlers or not any(
                    isinstance(h, logging.StreamHandler) for h in detail_logger.handlers
                ):
                    detail_logger.addHandler(logging.StreamHandler())

                summary_logger.info(
                    f'Starting Pareto analysis: Budget £{budget:,}, '
                    f'Loft Probability {prob_loft}'
                )

                # ----------------------------------------------------------
                # RUN PARETO SWEEP
                # ----------------------------------------------------------
                pareto_df, all_stats, baseline_stats = run_pareto(
                    df_all_packages=df,
                    budget=budget,
                    equity_floors=equity_floors,
                    high_equity_personas=DEFAULT_HIGH_EQUITY_PERSONAS,
                    output_dir=output_dir,
                    loft_prob=prob_loft,
                    cost_col='mean_total_capex',
                    carbon_col='mean_total_co2_saved',
                    upn_col='upn',
                    persona_col='meta_socio_persona',
                    time_limit_seconds=600,
                    detail_logger=detail_logger,
                    summary_logger=summary_logger,
                )

                summary_logger.info("Pareto analysis complete!")
                print(f"✓ Results saved to: {output_dir}")

                # EPC comparison (if enabled)
                if epc_run:
                    epc_random_path = os.path.join(
                        output_dir, 'epc_random_selection.csv'
                    )
                    epc_random_selected_df, epc_random_remaining = select_epc_algo(
                        df_knapsack=df,
                        budget=budget,
                        cost_column='mean_total_capex',
                        efficiency_column='capex_per_net_ton',
                        carbon_col='mean_total_co2_saved',
                        logger=detail_logger,
                    )
                    epc_random_selected_df['remaining_funds'] = epc_random_remaining
                    if epc_random_selected_df.empty:
                        detail_logger.info('EPC selection empty')
                        raise Exception('EPC selection empty')
                    epc_random_selected_df.to_csv(epc_random_path, index=False)
                    summary_logger.info(f"EPC random saved to: {epc_random_path}")

        print("\n" + "=" * 80)
        print("PARETO RUNS COMPLETE!")
        print("=" * 80)
    else:
        print('Set to skip runs.')

    # ------------------------------------------------------------------
    # POST PROCESSING
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POST PROCESSING")
    print("=" * 80)

    # NOTE: post_proc_greedy expects the old folder structure with
    # equity_factor folders. For the new Pareto structure you'll want
    # a new post-processing script that reads pareto_summary.csv from
    # each budget folder and produces combined Pareto front plots.
    # 
    # Example of what that looks like:
    #   for each budget folder:
    #     pareto_summary.csv has the front
    #     pareto_full.json has persona + intervention breakdowns
    #
    

    for loft_val in loft_probs: 
        viss_fold = os.path.join(pareto_runs_folder, 'pareto_vis', f'budget{str(budgets)}_{loft_val} ' )
        os.makedirs(viss_fold, exist_ok=True)
        
        post_proc_pareto(
        BUDGETS=budgets,
        EQUITY_FLOORS=equity_floors,
        LOFT_VALUE=loft_val,
        BASE_PATH=pareto_runs_folder,
        OUTPUT_PATH=viss_fold,
        RHO=RHO,
        )

    if epc_run:
        for LOFT_VALUE in loft_probs:
            for budget in budgets:
                million_budget = budget / milion_factor
                run_epc_vis(
                    pareto_runs_folder,
                    os.path.join(BASE_DIR, 'greedy_vis_epc_pareto'),
                    million_budget, LOFT_VALUE, equity_factor=0,
                )

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)


main()