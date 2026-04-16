# ==============================================================================
# 0. IMPORTS
# ==============================================================================

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .RetrofitGreedyAnalysis import (
    plot_greedy_comparison_main,
    plot_carbon_by_persona,
    plot_count_by_group,
    plot_metric_by_group,
)
from .Sankey import run_sankey_greedy


# ==============================================================================
# 1. CONSTANTS
# ==============================================================================

MILLION_FACTOR = 1_000_000


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================

def load_data(budgets, equity_weights, loft_val, base_path):
    """
    Load per-project results and derive equity metrics for every
    (budget, equity_weight) combination.

    Returns
    -------
    equity_df : pd.DataFrame   – one row per (budget, equity_weight)
    results_df : pd.DataFrame  – row-level project data with scenario tags
    """
    equity_dfs = []
    results_dfs = []
    loaded = 0
    total = len(budgets) * len(equity_weights)

    print("--- Starting Data Loading ---")

    for budget in budgets:
        for ew in equity_weights:
            budg = str(budget / MILLION_FACTOR).replace('.0', '')
            dir_name = f'budget_{budg}M__loft_{loft_val}__equity_{ew}'
            scenario_label = f'budget_{budg}M_equity_{ew}'
            dir_path = os.path.join(base_path, dir_name)
            results_file = os.path.join(dir_path, 'selected_projects.csv')

            print(results_file)

            # --- Load project-level results ---
            results_df_temp = pd.read_csv(results_file)

            if 'scenario' in results_df_temp.columns:
                results_df_temp = results_df_temp.rename(
                    columns={'scenario': 'intervention'}
                )

            results_df_temp['scenario'] = scenario_label
            results_df_temp['budget'] = budget
            results_df_temp['equity_weight'] = ew
            results_dfs.append(results_df_temp)

            # --- Derive equity metrics ---
            counts = results_df_temp.groupby('meta_socio_persona')['upn'].count()
            pcts = counts / counts.sum()

            equity_row = {
                'scenario': scenario_label,
                'budget': budget,
                'equity_weight': ew,
                'high_risk_count': counts.get('high_risk', 0),
                'high_risk_pct': pcts.get('high_risk', 0.0),
                'med_risk_count': counts.get('med_risk', 0),
                'med_risk_pct': pcts.get('med_risk', 0.0),
                'middle_risk_count': counts.get('middle_risk', 0),
                'middle_risk_pct': pcts.get('middle_risk', 0.0),
                'low_risk_count': counts.get('low_risk', 0),
                'low_risk_pct': pcts.get('low_risk', 0.0),
                'v_low_risk_count': counts.get('v_low_risk', 0),
                'v_low_risk_pct': pcts.get('v_low_risk', 0.0),
            }

            # Herfindahl concentration index (0 = equal, 1 = concentrated)
            proportions = counts / len(results_df_temp)
            equity_row['equity_concentration'] = (proportions ** 2).sum()

            equity_dfs.append(equity_row)

            print(f"  Loaded: budget=£{budget/1e6:.1f}M, equity_weight={ew}")
            loaded += 1

    # --- Combine ---
    equity_df = pd.DataFrame()
    if equity_dfs:
        equity_df = pd.DataFrame(equity_dfs).drop_duplicates()
        print(f"\n  Combined {len(equity_dfs)} equity records ({len(equity_df)} unique)")
    else:
        print("\n  No equity tracking data loaded!")

    results_df = pd.DataFrame()
    if results_dfs:
        results_df = pd.concat(results_dfs, ignore_index=True).drop_duplicates()
        print(f"  Combined {len(results_dfs)} results files ({len(results_df):,} rows)")
    else:
        print("\n  No results data loaded!")

    print(f"\n{'='*70}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*70}")
    print(f"  Budgets:      {budgets}")
    print(f"  Equity wts:   {equity_weights}")
    print(f"  Combinations: {loaded}/{total} loaded")
    print(f"{'='*70}\n")

    return equity_df, results_df


# ==============================================================================
# 3. AGGREGATION
# ==============================================================================

def aggregate_results(df, rho=0.5):
    """
    Aggregate project-level results to scenario level using ratio-of-sums.

    Produces mean + three uncertainty bands (uncorrelated, partially
    correlated, fully correlated) for total capex, total CO2, and
    cost-effectiveness (capex per net ton).

    Parameters
    ----------
    df : pd.DataFrame
        Project-level data with columns: scenario, upn, mean_total_capex,
        std_total_capex, mean_total_co2_saved, std_total_co2_saved.
    rho : float
        Correlation parameter for partial-correlation band (0–1).
    """
    if df.empty:
        print("Warning: Results dataframe is empty. Cannot aggregate.")
        return pd.DataFrame()

    df = df.copy()

    # Variance columns for uncorrelated path
    df['var_total_capex'] = df['std_total_capex'] ** 2
    df['var_total_co2'] = df['std_total_co2_saved'] ** 2

    agg_dict = {
        'upn': 'count',
        'mean_total_capex': 'sum',
        'mean_total_co2_saved': 'sum',
        'var_total_capex': 'sum',
        'var_total_co2': 'sum',
        'std_total_capex': 'sum',        # for correlated path
        'std_total_co2_saved': 'sum',     # for correlated path
    }

    df_agg = df.groupby('scenario').agg(agg_dict).reset_index()
    df_agg = df_agg.rename(columns={'upn': 'num_buildings_sum'})

    # ── Totals uncertainty ────────────────────────────────────────────────
    # Uncorrelated
    df_agg['total_capex_std_uncorr'] = np.sqrt(df_agg['var_total_capex'])
    df_agg['total_co2_std_uncorr'] = np.sqrt(df_agg['var_total_co2'])

    # Correlated (= sum of stds)
    df_agg = df_agg.rename(columns={
        'std_total_capex': 'total_capex_std_corr',
        'std_total_co2_saved': 'total_co2_std_corr',
    })

    # Partially correlated
    df_agg['total_capex_std_partial'] = np.sqrt(
        (1 - rho) * df_agg['total_capex_std_uncorr'] ** 2
        + rho * df_agg['total_capex_std_corr'] ** 2
    )
    df_agg['total_co2_std_partial'] = np.sqrt(
        (1 - rho) * df_agg['total_co2_std_uncorr'] ** 2
        + rho * df_agg['total_co2_std_corr'] ** 2
    )

    # ── Efficiency (capex per net ton) ────────────────────────────────────
    df_agg['mean_capex_per_net_ton'] = (
        df_agg['mean_total_capex'] / df_agg['mean_total_co2_saved']
    )

    # Error propagation for Z = X / Y
    for tag, capex_std, co2_std in [
        ('uncorr', 'total_capex_std_uncorr', 'total_co2_std_uncorr'),
        ('corr', 'total_capex_std_corr', 'total_co2_std_corr'),
        ('partial', 'total_capex_std_partial', 'total_co2_std_partial'),
    ]:
        cv_capex = df_agg[capex_std] / df_agg['mean_total_capex']
        cv_co2 = df_agg[co2_std] / df_agg['mean_total_co2_saved']
        df_agg[f'std_capex_per_net_ton_{tag}'] = (
            df_agg['mean_capex_per_net_ton'] * np.sqrt(cv_capex ** 2 + cv_co2 ** 2)
        )

    # ── Select & order columns ────────────────────────────────────────────
    cols_order = [
        'scenario',
        'num_buildings_sum',
        'mean_capex_per_net_ton',
        'std_capex_per_net_ton_uncorr',
        'std_capex_per_net_ton_partial',
        'std_capex_per_net_ton_corr',
        'mean_total_capex',
        'total_capex_std_uncorr',
        'total_capex_std_partial',
        'total_capex_std_corr',
        'mean_total_co2_saved',
        'total_co2_std_uncorr',
        'total_co2_std_partial',
        'total_co2_std_corr',
    ]

    return df_agg[cols_order]


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def post_proc_greedy(
    BUDGETS, EQUITY_WEIGHTS, LOFT_VALUE, BASE_PATH, OUTPUT_PATH, RHO=0.5,
):
    """
    End-to-end pipeline: load → aggregate → merge → plot.

    Parameters
    ----------
    BUDGETS : list[float]
        Budget values (in raw £, e.g. 50_000_000).
    EQUITY_WEIGHTS : list[float]
        Equity weight values (0–1).
    LOFT_VALUE : float
        Loft insulation parameter.
    BASE_PATH : str
        Root directory containing scenario sub-folders.
    OUTPUT_PATH : str
        Directory for plots and CSVs.
    RHO : float
        Partial-correlation parameter (0 = uncorrelated, 1 = fully correlated).
    """
    # ── 1. Load ───────────────────────────────────────────────────────────
    equity_df, results_df = load_data(
        budgets=BUDGETS,
        equity_weights=EQUITY_WEIGHTS,
        loft_val=LOFT_VALUE,
        base_path=BASE_PATH,
    )

    if results_df.empty or equity_df.empty:
        print("Critical error: No data was loaded. Exiting.")
        return

    # ── 2. Aggregate ──────────────────────────────────────────────────────
    results_agg = aggregate_results(results_df, rho=RHO)
    print(f'\nUsing correlation parameter rho = {RHO}')

    if results_agg.empty:
        print("Critical error: Aggregation failed. Exiting.")
        return

    # ── 3. Merge ──────────────────────────────────────────────────────────
    # equity_df already has one row per scenario with bare column names
    # (high_risk_pct, equity_concentration, etc.) — no _mean/_std suffixes —
    # which is exactly what the plotting functions expect.
    comparison_df = results_agg.merge(equity_df, on='scenario', how='left')

    # Readable labels
    scenario_map = {
        f'budget_{b/1e6:.0f}M_equity_{e}': f'£{b/1e6:.0f}M, Equity={e}'
        for b in BUDGETS
        for e in EQUITY_WEIGHTS
    }
    comparison_df['scenario_label'] = comparison_df['scenario'].map(scenario_map)

    # Sort by equity weight then budget
    temp_map = {
        f'budget_{b/1e6:.0f}M_equity_{e}': (e, b)
        for b in BUDGETS
        for e in EQUITY_WEIGHTS
    }
    sort_keys = comparison_df['scenario'].map(temp_map)

    if sort_keys.notna().all():
        comparison_df['sort_equity'] = sort_keys.str[0]
        comparison_df['sort_budget'] = sort_keys.str[1]
        comparison_df = comparison_df.sort_values(
            ['sort_equity', 'sort_budget']
        ).drop(columns=['sort_equity', 'sort_budget'])
    else:
        missing = comparison_df.loc[sort_keys.isna(), 'scenario'].tolist()
        print(f"Warning: Could not sort — {len(missing)} scenarios "
              f"not in map: {missing[:5]}")

    # ── 4. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"EQUITY WEIGHTING COMPARISON SUMMARY (rho = {RHO})")
    print(f"{'='*80}")

    display_cols = [
        'scenario_label',
        'mean_total_co2_saved',
        'total_co2_std_uncorr',
        'total_co2_std_partial',
        'total_co2_std_corr',
        'num_buildings_sum',
        'mean_capex_per_net_ton',
        'std_capex_per_net_ton_uncorr',
        'std_capex_per_net_ton_partial',
        'std_capex_per_net_ton_corr',
        'high_risk_pct',
        'equity_concentration',
        'med_risk_pct',
        'middle_risk_pct',
        'low_risk_pct',
        'v_low_risk_pct',
    ]
    display_cols = [c for c in display_cols if c in comparison_df.columns]

    if display_cols:
        print(comparison_df[display_cols].to_string(index=False))
    else:
        print("Could not find key columns to display.")
    print()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    comparison_df.to_csv(
        os.path.join(OUTPUT_PATH, f'comparison_df_rho_{RHO}.csv'), index=False,
    )

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print(f"--- Generating plots in: {OUTPUT_PATH} ---")

    scenario_colors = plot_greedy_comparison_main(
        comparison_df, output_dir=OUTPUT_PATH,
        y_axis_zero=True, loft_val=LOFT_VALUE, rho=RHO,
    )

    plot_carbon_by_persona(
        results_df, scenario_colors,
        os.path.join(OUTPUT_PATH, f"12_carbon_per_persona_loft_{LOFT_VALUE}.png"),
        y_axis_zero=True,
    )

    plot_metric_by_group(
        results_df, scenario_colors,
        filename=os.path.join(
            OUTPUT_PATH, f"12b_carbon_metapersona__loft_{LOFT_VALUE}.png"
        ),
        value_col='mean_total_co2_saved',
        metric_stat='sum',
        group_col='meta_socio_persona',
        xlabel='Socio-economic Persona',
        ylabel='Total Carbon Saved (Ton)',
        y_axis_zero=True,
    )

    plot_metric_by_group(
        results_df, scenario_colors,
        filename=os.path.join(
            OUTPUT_PATH, f"13_mean_cost_per_Ton_per_persona_loft_{LOFT_VALUE}.png"
        ),
        value_col='mean_capex_per_net_ton',
        group_col='meta_socio_persona',
        xlabel='Socio-economic Persona',
        ylabel='Total Cost per Ton Saved (£)',
        y_axis_zero=True,
    )

    plot_metric_by_group(
        results_df, scenario_colors,
        filename=os.path.join(
            OUTPUT_PATH,
            f"13BB_sigma_cost_per_Ton_per_persona_loft_{LOFT_VALUE}.png",
        ),
        value_col='capex_per_net_ton_sigma',
        group_col='meta_socio_persona',
        xlabel='Socio-economic Persona',
        ylabel='Std Cost per Ton Saved (£)',
        y_axis_zero=True,
    )

    plot_metric_by_group(
        results_df, scenario_colors,
        filename=os.path.join(
            OUTPUT_PATH,
            f"14_cost_per_intervention_per_persona__loft_{LOFT_VALUE}.png",
        ),
        value_col='mean_total_capex',
        group_col='meta_socio_persona',
        xlabel='Socio-economic Persona',
        ylabel='Total Cost per Intervention (£)',
        y_axis_zero=True,
    )

    plot_count_by_group(
        results_df, scenario_colors,
        filename=os.path.join(
            OUTPUT_PATH, f"15_counts_persona__loft_{LOFT_VALUE}.png"
        ),
        group_col='meta_socio_persona',
        xlabel='Socio-economic Persona',
        ylabel='Number of Projects',
        y_axis_zero=True,
    )

    print("  Plotting complete.")