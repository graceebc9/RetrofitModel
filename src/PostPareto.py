# ==============================================================================
# 0. IMPORTS
# ==============================================================================

import os
import glob
import json
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


# ==============================================================================
# 1. CONSTANTS
# ==============================================================================

MILLION_FACTOR = 1_000_000


# ==============================================================================
# 2. DATA LOADING — NEW PARETO STRUCTURE
# ==============================================================================

def load_pareto_data(budgets, equity_floors, loft_val, base_path):
    """
    Load per-project results from the new Pareto folder structure.

    Folder layout:
        base_path/
            budget_1M__loft_0.65/
                selected_projects_eq0.csv
                selected_projects_eq5.csv
                ...
                pareto_summary.csv
            budget_10M__loft_0.65/
                ...

    Returns
    -------
    equity_df : pd.DataFrame
        One row per (budget, equity_floor) — persona shares and counts.
    results_df : pd.DataFrame
        Row-level project data with scenario tags for every
        (budget, equity_floor) combination.
    """
    equity_dfs = []
    results_dfs = []
    loaded = 0
    total = len(budgets) * len(equity_floors)

    print("--- Starting Data Loading (Pareto structure) ---")

    for budget in budgets:
        budg = str(budget / MILLION_FACTOR).replace('.0', '')
        dir_name = f'budget_{budg}M__loft_{loft_val}'
        dir_path = os.path.join(base_path, dir_name)

        if not os.path.isdir(dir_path):
            print(f"  ⚠️ Missing directory: {dir_path}")
            continue

        for ef in equity_floors:
            eq_label = f"{ef:.0f}"
            scenario_label = f'budget_{budg}M_eq_{eq_label}'
            results_file = os.path.join(
                dir_path, f'selected_projects_eq{eq_label}.csv'
            )

            if not os.path.isfile(results_file):
                print(f"  ⚠️ Missing file: {results_file}")
                continue

            # --- Load project-level results ---
            results_df_temp = pd.read_csv(results_file)

            if 'scenario' in results_df_temp.columns:
                results_df_temp = results_df_temp.rename(
                    columns={'scenario': 'intervention'}
                )

            results_df_temp['scenario'] = scenario_label
            results_df_temp['budget'] = budget
            results_df_temp['equity_floor_pct'] = ef
            results_dfs.append(results_df_temp)

            # --- Derive equity metrics ---
            counts = results_df_temp.groupby('meta_socio_persona')['upn'].count()
            pcts = counts / counts.sum() if counts.sum() > 0 else counts * 0

            equity_row = {
                'scenario': scenario_label,
                'budget': budget,
                'equity_floor_pct': ef,
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

            # Herfindahl concentration index
            if len(results_df_temp) > 0:
                proportions = counts / len(results_df_temp)
                equity_row['equity_concentration'] = (proportions ** 2).sum()
            else:
                equity_row['equity_concentration'] = 0.0

            equity_dfs.append(equity_row)

            print(f"  Loaded: budget=£{budget/1e6:.1f}M, equity_floor={ef}%")
            loaded += 1

    # --- Combine ---
    equity_df = pd.DataFrame()
    if equity_dfs:
        equity_df = pd.DataFrame(equity_dfs).drop_duplicates()
        print(f"\n  Combined {len(equity_dfs)} equity records "
              f"({len(equity_df)} unique)")

    results_df = pd.DataFrame()
    if results_dfs:
        results_df = pd.concat(results_dfs, ignore_index=True).drop_duplicates()
        print(f"  Combined {len(results_dfs)} results files "
              f"({len(results_df):,} rows)")

    print(f"\n{'='*70}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*70}")
    print(f"  Budgets:        {budgets}")
    print(f"  Equity floors:  {equity_floors}")
    print(f"  Combinations:   {loaded}/{total} loaded")
    print(f"{'='*70}\n")

    return equity_df, results_df


def load_pareto_summaries(budgets, loft_val, base_path):
    """
    Load the pareto_summary.csv from each budget folder into a single 
    long-format DataFrame. Useful for the Pareto front overlay plot.
    """
    summaries = []
    for budget in budgets:
        budg = str(budget / MILLION_FACTOR).replace('.0', '')
        dir_name = f'budget_{budg}M__loft_{loft_val}'
        summary_file = os.path.join(base_path, dir_name, 'pareto_summary.csv')

        if not os.path.isfile(summary_file):
            print(f"  ⚠️ Missing summary: {summary_file}")
            continue

        s = pd.read_csv(summary_file)
        s['budget'] = budget
        s['budget_label'] = f'£{budget/1e6:.0f}M'
        summaries.append(s)

    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


# ==============================================================================
# 3. AGGREGATION (unchanged — scenario-agnostic)
# ==============================================================================

def aggregate_results(df, rho=0.5):
    """
    Aggregate project-level results to scenario level using ratio-of-sums.

    Produces mean + three uncertainty bands (uncorrelated, partially
    correlated, fully correlated) for total capex, total CO2, and
    cost-effectiveness (capex per net ton).
    """
    if df.empty:
        print("Warning: Results dataframe is empty. Cannot aggregate.")
        return pd.DataFrame()

    df = df.copy()

    df['var_total_capex'] = df['std_total_capex'] ** 2
    df['var_total_co2'] = df['std_total_co2_saved'] ** 2

    agg_dict = {
        'upn': 'count',
        'mean_total_capex': 'sum',
        'mean_total_co2_saved': 'sum',
        'var_total_capex': 'sum',
        'var_total_co2': 'sum',
        'std_total_capex': 'sum',
        'std_total_co2_saved': 'sum',
    }

    df_agg = df.groupby('scenario').agg(agg_dict).reset_index()
    df_agg = df_agg.rename(columns={'upn': 'num_buildings_sum'})

    df_agg['total_capex_std_uncorr'] = np.sqrt(df_agg['var_total_capex'])
    df_agg['total_co2_std_uncorr'] = np.sqrt(df_agg['var_total_co2'])

    df_agg = df_agg.rename(columns={
        'std_total_capex': 'total_capex_std_corr',
        'std_total_co2_saved': 'total_co2_std_corr',
    })

    df_agg['total_capex_std_partial'] = np.sqrt(
        (1 - rho) * df_agg['total_capex_std_uncorr'] ** 2
        + rho * df_agg['total_capex_std_corr'] ** 2
    )
    df_agg['total_co2_std_partial'] = np.sqrt(
        (1 - rho) * df_agg['total_co2_std_uncorr'] ** 2
        + rho * df_agg['total_co2_std_corr'] ** 2
    )

    df_agg['mean_capex_per_net_ton'] = (
        df_agg['mean_total_capex'] / df_agg['mean_total_co2_saved']
    )

    for tag, capex_std, co2_std in [
        ('uncorr', 'total_capex_std_uncorr', 'total_co2_std_uncorr'),
        ('corr', 'total_capex_std_corr', 'total_co2_std_corr'),
        ('partial', 'total_capex_std_partial', 'total_co2_std_partial'),
    ]:
        cv_capex = df_agg[capex_std] / df_agg['mean_total_capex']
        cv_co2 = df_agg[co2_std] / df_agg['mean_total_co2_saved']
        df_agg[f'std_capex_per_net_ton_{tag}'] = (
            df_agg['mean_capex_per_net_ton']
            * np.sqrt(cv_capex ** 2 + cv_co2 ** 2)
        )

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
# 4. CROSS-BUDGET PARETO PLOTS (NEW)
# ==============================================================================

def plot_pareto_front_overlay(pareto_summaries_df, output_dir, loft_val):
    """
    Overlay Pareto fronts for multiple budgets on one chart.
    Equity floor on x-axis, total abatement on y-axis, one curve per budget.
    """
    if pareto_summaries_df.empty:
        print("  No Pareto summaries to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(pareto_summaries_df['budget'].unique())
    cmap = plt.colormaps.get_cmap('viridis')
    colors = [cmap(i / max(len(budgets) - 1, 1)) for i in range(len(budgets))]

    for i, budget in enumerate(budgets):
        sub = pareto_summaries_df[pareto_summaries_df['budget'] == budget].copy()
        sub = sub.sort_values('equity_floor_pct')
        sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]

        ax.plot(
            sub['equity_floor_pct'],
            sub['total_abatement'],
            'o-', color=colors[i], linewidth=2, markersize=6,
            label=f'£{budget/1e6:.0f}M',
        )

    ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
    ax.set_ylabel('Total CO₂ abatement (tonnes)', fontsize=11)
    ax.set_title(
        f'Pareto Fronts Across Budgets — Loft {loft_val}',
        fontsize=13, fontweight='bold',
    )
    ax.legend(title='Budget', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'01_pareto_fronts_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def plot_cpex_vs_equity_overlay(pareto_summaries_df, output_dir, loft_val):
    """£/tCO2 vs equity floor, one curve per budget."""
    if pareto_summaries_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(pareto_summaries_df['budget'].unique())
    cmap = plt.colormaps.get_cmap('viridis')
    colors = [cmap(i / max(len(budgets) - 1, 1)) for i in range(len(budgets))]

    for i, budget in enumerate(budgets):
        sub = pareto_summaries_df[pareto_summaries_df['budget'] == budget].copy()
        sub = sub.sort_values('equity_floor_pct')
        sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]

        ax.plot(
            sub['equity_floor_pct'],
            sub['cpex_per_ton'],
            's-', color=colors[i], linewidth=2, markersize=6,
            label=f'£{budget/1e6:.0f}M',
        )

    ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
    ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
    ax.set_title(
        f'Cost-Effectiveness vs Equity — Loft {loft_val}',
        fontsize=13, fontweight='bold',
    )
    ax.legend(title='Budget', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'02_cpex_vs_equity_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def plot_tradeoff_scatter(pareto_summaries_df, output_dir, loft_val):
    """
    Scatter of £/tCO2 vs total abatement across all (budget, eq_floor)
    combinations. Colour = budget, marker size = equity floor.
    """
    if pareto_summaries_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(pareto_summaries_df['budget'].unique())
    cmap = plt.colormaps.get_cmap('viridis')
    colors = [cmap(i / max(len(budgets) - 1, 1)) for i in range(len(budgets))]

    for i, budget in enumerate(budgets):
        sub = pareto_summaries_df[pareto_summaries_df['budget'] == budget].copy()
        sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]

        sizes = 30 + (sub['equity_floor_pct'] / 100.0) * 150
        ax.scatter(
            sub['total_abatement'],
            sub['cpex_per_ton'],
            s=sizes, color=colors[i], alpha=0.7,
            edgecolor='white', linewidth=0.5,
            label=f'£{budget/1e6:.0f}M',
        )

    ax.set_xlabel('Total CO₂ abatement (tonnes)', fontsize=11)
    ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
    ax.set_title(
        f'Trade-off Space (marker size = equity floor) — Loft {loft_val}',
        fontsize=13, fontweight='bold',
    )
    ax.legend(title='Budget', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'03_tradeoff_scatter_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def post_proc_pareto(
    BUDGETS, EQUITY_FLOORS, LOFT_VALUE, BASE_PATH, OUTPUT_PATH, RHO=0.5,
):
    """
    End-to-end pipeline for the new Pareto structure.

    Parameters
    ----------
    BUDGETS : list[float]
        Budget values in raw £ (e.g. 50_000_000).
    EQUITY_FLOORS : list[float]
        Equity floor percentages (e.g. [0, 10, 20, ..., 100]).
    LOFT_VALUE : float
        Loft insulation parameter.
    BASE_PATH : str
        Root containing `budget_XM__loft_Y/` folders.
    OUTPUT_PATH : str
        Directory for plots and CSVs.
    RHO : float
        Partial-correlation parameter (0–1).
    """
    # ── 1. Load project-level data ────────────────────────────────────────
    equity_df, results_df = load_pareto_data(
        budgets=BUDGETS,
        equity_floors=EQUITY_FLOORS,
        loft_val=LOFT_VALUE,
        base_path=BASE_PATH,
    )

    if results_df.empty or equity_df.empty:
        print("Critical error: No data loaded. Exiting.")
        return

    # ── 2. Load Pareto summaries for front plots ──────────────────────────
    pareto_summaries_df = load_pareto_summaries(
        budgets=BUDGETS, loft_val=LOFT_VALUE, base_path=BASE_PATH,
    )

    # ── 3. Aggregate project-level to scenario-level ──────────────────────
    results_agg = aggregate_results(results_df, rho=RHO)
    print(f'\nUsing correlation parameter rho = {RHO}')

    if results_agg.empty:
        print("Critical error: Aggregation failed. Exiting.")
        return

    # ── 4. Merge ──────────────────────────────────────────────────────────
    comparison_df = results_agg.merge(equity_df, on='scenario', how='left')

    # Readable labels
    scenario_map = {
        f'budget_{b/1e6:.0f}M_eq_{e:.0f}': f'£{b/1e6:.0f}M, EqFloor={e}%'
        for b in BUDGETS
        for e in EQUITY_FLOORS
    }
    comparison_df['scenario_label'] = comparison_df['scenario'].map(scenario_map)

    # Sort by equity floor then budget
    temp_map = {
        f'budget_{b/1e6:.0f}M_eq_{e:.0f}': (e, b)
        for b in BUDGETS
        for e in EQUITY_FLOORS
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

    # ── 5. Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"PARETO COMPARISON SUMMARY (rho = {RHO})")
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
    print()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    comparison_df.to_csv(
        os.path.join(OUTPUT_PATH, f'comparison_df_rho_{RHO}.csv'),
        index=False,
    )

    if not pareto_summaries_df.empty:
        pareto_summaries_df.to_csv(
            os.path.join(OUTPUT_PATH, 'pareto_summaries_all_budgets.csv'),
            index=False,
        )

    # ── 6. Cross-budget Pareto plots (NEW) ────────────────────────────────
    print(f"\n--- Generating Pareto overlay plots in: {OUTPUT_PATH} ---")

    plot_pareto_front_overlay(pareto_summaries_df, OUTPUT_PATH, LOFT_VALUE)
    plot_cpex_vs_equity_overlay(pareto_summaries_df, OUTPUT_PATH, LOFT_VALUE)
    plot_tradeoff_scatter(pareto_summaries_df, OUTPUT_PATH, LOFT_VALUE)

    # ── 7. Reuse existing scenario-comparison plots ───────────────────────
    print(f"\n--- Generating scenario comparison plots ---")

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

    if 'capex_per_net_ton_sigma' in results_df.columns:
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

    print("\n  Plotting complete.")