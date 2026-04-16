import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------------------------
# NOTE: If running this file directly (not as part of a package), comment out
# the relative import below and define cluster_names locally.
# ---------------------------------------------------------------------------
# from .personas import cluster_names

label_cleaner_map = {
    'high_risk': 'High Risk',
    'med_risk': 'Medium Risk',
    'middle_risk': 'Middle Risk',
    'low_risk': 'Low Risk',
    'v_low_risk': 'Very Low Risk',
}

# ── Column-name constants ─────────────────────────────────────────────────────
total_co2_saved_col = 'mean_total_co2_saved'
total_co2_saved_col_std = 'total_co2_std_uncorr'
total_co2_full_corr = 'total_co2_std_corr'
total_co2_std_partial = 'total_co2_std_partial'

capex_per_net_ton_mean_col = 'mean_capex_per_net_ton'
capex_per_net_ton_std_col = 'std_capex_per_net_ton_corr'
capex_per_net_ton_std_col_uncorr = 'std_capex_per_net_ton_uncorr'
capex_per_net_ton_std_partial = 'std_capex_per_net_ton_partial'

scenario_label_map = {
    'budget_1m_equity_0': 'equity_0',
    'budget_1m_equity_0.2': 'equity_0.2',
    'budget_1m_equity_0.4': 'equity_0.4',
    'budget_1m_equity_0.6': 'equity_0.6',
    'budget_1m_equity_0.8': 'equity_0.8',
    'budget_1m_equity_1': 'equity_1',
    'budget_10m_equity_0': 'equity_0',
    'budget_10m_equity_0.2': 'equity_0.2',
    'budget_10m_equity_0.4': 'equity_0.4',
    'budget_10m_equity_0.6': 'equity_0.6',
    'budget_10m_equity_0.8': 'equity_0.8',
    'budget_10m_equity_1': 'equity_1',
    'budget_100m_equity_0': 'equity_0',
    'budget_100m_equity_0.2': 'equity_0.2',
    'budget_100m_equity_0.4': 'equity_0.4',
    'budget_100m_equity_0.6': 'equity_0.6',
    'budget_100m_equity_0.8': 'equity_0.8',
    'budget_100m_equity_1': 'equity_1',
    'budget_50m_equity_0': 'equity_0',
    'budget_50m_equity_0.2': 'equity_0.2',
    'budget_50m_equity_0.4': 'equity_0.4',
    'budget_50m_equity_0.6': 'equity_0.6',
    'budget_50m_equity_0.8': 'equity_0.8',
    'budget_50m_equity_1': 'equity_1',
    'budget_80m_equity_0': 'equity_0',
    'budget_80m_equity_0.2': 'equity_0.2',
    'budget_80m_equity_0.4': 'equity_0.4',
    'budget_80m_equity_0.6': 'equity_0.6',
    'budget_80m_equity_0.8': 'equity_0.8',
    'budget_80m_equity_1': 'equity_1',
    'budget_10000000000_equity_0': 'equity_0',
    'budget_10000000000_equity_0.2': 'equity_0.2',
    'budget_10000000000_equity_0.4': 'equity_0.4',
    'budget_10000000000_equity_0.6': 'equity_0.6',
    'budget_10000000000_equity_0.8': 'equity_0.8',
    'budget_10000000000_equity_1': 'equity_1',
}


# ==============================================================================
# 1. BUDGET PARSING HELPER  (single source of truth for sort order)
# ==============================================================================

def _parse_budget_value(val):
    """Convert a budget string like '£50M', '10m', '1000000' to a float."""
    s = str(val).lower().strip().replace('£', '').replace(',', '')
    multiplier = 1
    if s.endswith('m'):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith('k'):
        multiplier = 1_000
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return 0.0


def get_sorted_budget_list(df, budget_col='budget'):
    """Return unique budgets from *df* sorted numerically."""
    return sorted(df[budget_col].unique(), key=_parse_budget_value)


# ==============================================================================
# 2. DATA PREPROCESSING
# ==============================================================================

def flatten_columns(df):
    """Flatten MultiIndex columns into single-level strings."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat_name = '_'.join(str(c) for c in col if c)
            new_cols.append(flat_name)
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


def preprocess_dataframe(df):
    """Flatten columns, extract budget / equity_weight, sort."""
    df_flat = flatten_columns(df)

    try:
        df_flat['equity_weight'] = (
            df_flat['scenario'].str.split('_').str[-1].astype(float)
        )
    except Exception as e:
        print(f"Error: Could not extract equity weight: {e}")
        raise

    try:
        df_flat['budget'] = df_flat['scenario'].str.split('_').str[1]
    except Exception as e:
        print(f"Error: Could not extract budget: {e}")
        raise

    df_flat = df_flat.sort_values(['budget', 'equity_weight']).reset_index(drop=True)
    return df_flat


# ==============================================================================
# 3. COLOUR HELPERS
# ==============================================================================

def get_color_palette(n_colors):
    if n_colors <= 3:
        return ['#e74c3c', '#f39c12', '#27ae60'][:n_colors]
    return sns.color_palette('Set3', n_colors).as_hex()


def create_scenario_colors(scenarios, results_agg):
    equity_weights = sorted(results_agg['equity_weight'].unique())
    colors = get_color_palette(len(equity_weights))
    weight_to_color = dict(zip(equity_weights, colors))

    scenario_colors = {}
    for scenario in scenarios:
        weight = results_agg.loc[
            results_agg['scenario'] == scenario, 'equity_weight'
        ].iloc[0]
        scenario_colors[scenario] = weight_to_color[weight]
    return scenario_colors


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

def plot_greedy_comparison_main(
    df_raw, output_dir, y_axis_zero=False, loft_val=None, rho=None
):
    """Generate all combined-budget plots and save to *output_dir*."""
    df_processed = preprocess_dataframe(df_raw)

    scenarios = df_processed['scenario'].unique()
    equity_weights = sorted(df_processed['equity_weight'].unique())
    scenario_colors = create_scenario_colors(scenarios, df_processed)

    unique_budgets = get_sorted_budget_list(df_processed)
    budget_label = (
        f"Budget £{unique_budgets[0]}" if len(unique_budgets) == 1
        else "All Budgets"
    )

    results_subset = df_processed.copy()
    equity_subset = df_processed.copy()

    print("Generating plots...")

    plot_all_metrics(
        df_processed,
        output_dir=os.path.join(output_dir, 'single_plots'),
        y_axis_zero=y_axis_zero,
    )

    # ── Carbon savings (3 correlation variants) ───────────────────────────
    plot_carbon_savings_vs_equity(
        results_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"1_carbon_vs_equity_{loft_val}_UNCORR.png"),
        y_axis_zero=y_axis_zero,
    )
    plot_carbon_savings_vs_equity(
        results_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"1_carbon_vs_equity_{loft_val}_CORR.png"),
        y_axis_zero=y_axis_zero,
        co2_std=total_co2_full_corr,
    )
    plot_carbon_savings_vs_equity(
        results_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"1_carbon_vs_equity_{loft_val}_PARTIAL_CORR_{rho}.png"),
        y_axis_zero=y_axis_zero,
        co2_std=total_co2_std_partial,
    )

    # ── Cost effectiveness (2 variants) ───────────────────────────────────
    plot_cost_effectiveness_vs_equity(
        results_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"2_cost_effectiveness_vs_equity_{loft_val}_CORR.png"),
        y_axis_zero=y_axis_zero,
    )
    plot_cost_effectiveness_vs_equity(
        results_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"2_cost_effectiveness_vs_equity_{loft_val}_UNCORR.png"),
        y_axis_zero=y_axis_zero,
        capex_std=capex_per_net_ton_std_col_uncorr,
    )

    # ── Vulnerable coverage & equity concentration ────────────────────────
    plot_vulnerable_coverage_vs_equity(
        equity_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"3_vulnerable_coverage_vs_equity_{loft_val}.png"),
        y_axis_zero=y_axis_zero,
    )
    plot_equity_concentration_vs_weight(
        equity_subset, equity_weights, budget_label,
        os.path.join(output_dir, f"4_equity_concentration_vs_weight_{loft_val}.png"),
        y_axis_zero=y_axis_zero,
    )

    # ── Pareto fronts (3 correlation variants) ────────────────────────────
    for suffix, std_col in [
        (f"_{loft_val}", total_co2_saved_col_std),
        (f"_{loft_val}_FULLCORR", total_co2_full_corr),
        (f"_{loft_val}_partialCORR_{rho}", total_co2_std_partial),
    ]:
        plot_pareto_front(
            results_subset, equity_subset, scenarios, scenario_colors,
            budget_label,
            os.path.join(output_dir, f"6_pareto_front{suffix}.png"),
            y_axis_zero=y_axis_zero,
            ycol=total_co2_saved_col,
            ycol_std=std_col,
        )

    # ── Tradeoff efficiency ───────────────────────────────────────────────
    plot_tradeoff_efficiency(
        results_subset, equity_subset, scenarios, scenario_colors,
        budget_label,
        os.path.join(output_dir, f"8_tradeoff_efficiency_{loft_val}.png"),
        y_axis_zero=y_axis_zero,
    )

    # ── Radar chart ───────────────────────────────────────────────────────
    plot_radar_chart(
        results_subset, equity_subset,
        os.path.join(output_dir, f"9_radar_chart_{loft_val}.png"),
        scenario_colors,
    )

    # ── Pareto: buildings vs CO2 (3 variants) ─────────────────────────────
    for suffix, std_col in [
        (f"_{loft_val}", total_co2_saved_col_std),
        (f"_{loft_val}_FULLCORR", total_co2_full_corr),
        (f"_{loft_val}_PARTIALCORR_{rho}", total_co2_std_partial),
    ]:
        plot_pareto_front_flexi(
            results_subset, equity_subset, scenarios, scenario_colors,
            budget_label,
            filename=os.path.join(output_dir, f"10_pareto_bcounts{suffix}.png"),
            x_col='num_buildings_sum',
            y_col=total_co2_saved_col,
            y_col_std=std_col,
            x_label='Number of buildings',
            y_label='CO2 Saved (kton)',
            x_scaler=1.0,
            y_scaler=1e3,
            y_axis_zero=False,
        )

    # ── Pareto: cost effectiveness (3 variants) ──────────────────────────
    for suffix, std_col in [
        (f"_loft_{loft_val}_UNCORR", capex_per_net_ton_std_col),
        (f"_loft_{loft_val}_CORR", capex_per_net_ton_std_col_uncorr),
        (f"_loft_{loft_val}_PARTIALCORR_{rho}", capex_per_net_ton_std_partial),
    ]:
        plot_pareto_retrofit_carbon_by_costeff(
            results_subset, equity_subset, scenarios, scenario_colors,
            budget_label,
            os.path.join(output_dir, f"11_pareto_cost_eff{suffix}.png"),
            y_axis_zero=y_axis_zero,
            capex_std=std_col,
        )

    print(f"Done! All plots saved to '{output_dir}'.")
    return scenario_colors


# ==============================================================================
# 5. INDIVIDUAL PLOT FUNCTIONS  (combined-budget only)
# ==============================================================================

def plot_all_metrics(df, output_dir='scenario_plots', y_axis_zero=False):
    """Plot every *_mean column against equity_weight and save."""
    os.makedirs(output_dir, exist_ok=True)

    x = df['equity_weight']
    mean_cols = [c for c in df.columns if c.endswith('_mean')]

    print(f"Found {len(mean_cols)} metrics to plot. Saving to '{output_dir}'...")

    for mean_col in mean_cols:
        y_mean = df[mean_col]
        metric_base = mean_col[:-5]  # strip '_mean'
        std_col = f"{metric_base}_std"

        plt.figure(figsize=(12, 7))
        plt.plot(x, y_mean, marker='o', linestyle='-', label='Mean Value')

        if std_col in df.columns:
            y_std = df[std_col]
            plt.fill_between(
                x, y_mean - y_std, y_mean + y_std,
                alpha=0.2, label='Mean ± 1 Std. Dev.', color='blue',
            )

        plt.xlabel("Equity Weight", fontsize=12)
        plt.ylabel(mean_col, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x)

        if y_axis_zero:
            plt.ylim(bottom=0)

        plt.tight_layout()
        safe_filename = mean_col.replace(' ', '_').replace('/', '_') + '.png'
        plt.savefig(os.path.join(output_dir, safe_filename))
        plt.close()

    print(f"Done! All single-metric plots saved to '{output_dir}'.")


# ── Plot 1 ────────────────────────────────────────────────────────────────────

def plot_carbon_savings_vs_equity(
    results_subset, equity_weights, budget_label, filename,
    y_axis_zero=False, co2_std=total_co2_saved_col_std,
):
    """Carbon Savings vs Equity Weight (all budgets combined)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets_to_plot = get_sorted_budget_list(results_subset)

    for budget_val in budgets_to_plot:
        subset = results_subset[results_subset['budget'] == budget_val].sort_values('equity_weight')
        weights = subset['equity_weight'].values
        means = subset[total_co2_saved_col].values / 1e3
        stds = subset[co2_std].values / 1e3

        label = f'£{budget_val}' if len(budgets_to_plot) > 1 else None
        ax.errorbar(
            weights, means, yerr=stds, fmt='o-', markersize=10,
            linewidth=2, capsize=5, label=label, alpha=0.7,
        )

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)

    if len(budgets_to_plot) > 1:
        ax.legend(fontsize=12, title="Budget")
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 2 ────────────────────────────────────────────────────────────────────

def plot_cost_effectiveness_vs_equity(
    results_subset, equity_weights, budget_label, filename,
    y_axis_zero=False,
    capex_col=capex_per_net_ton_mean_col,
    capex_std=capex_per_net_ton_std_col,
):
    """Cost Effectiveness vs Equity Weight (all budgets combined)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = get_sorted_budget_list(results_subset)

    for budget_val in budgets:
        subset = results_subset[results_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset[capex_col].values
        stds = subset[capex_std].values

        label = f'£{budget_val}' if len(budgets) > 1 else None
        ax.errorbar(
            weights, means, yerr=stds, fmt='o-', markersize=10,
            linewidth=2, capsize=5, label=label, alpha=0.7,
        )

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost per Net Ton CO2 (£/ton)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(budgets) > 1:
        ax.legend(fontsize=12)
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 3 ────────────────────────────────────────────────────────────────────

def plot_vulnerable_coverage_vs_equity(
    equity_subset, equity_weights, budget_label, filename, y_axis_zero=False,
):
    """Vulnerable Coverage vs Equity Weight (all budgets combined)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = get_sorted_budget_list(equity_subset)

    for budget_val in budgets:
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset['high_risk_pct'].values * 100

        label = f'£{budget_val}' if len(budgets) > 1 else None
        ax.errorbar(
            weights, means, fmt='o-', markersize=10,
            linewidth=2, capsize=5, label=label, alpha=0.7,
        )

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(budgets) > 1:
        ax.legend(fontsize=12)
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 4 ────────────────────────────────────────────────────────────────────

def plot_equity_concentration_vs_weight(
    equity_subset, equity_weights, budget_label, filename, y_axis_zero=False,
):
    """Equity Concentration vs Equity Weight (all budgets combined)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = get_sorted_budget_list(equity_subset)

    for budget_val in budgets:
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset['equity_concentration'].values

        label = f'£{budget_val}' if len(budgets) > 1 else None
        ax.errorbar(
            weights, means, fmt='o-', markersize=10,
            linewidth=2, capsize=5, label=label, alpha=0.7,
        )

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Equity Concentration Index', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(budgets) > 1:
        ax.legend(fontsize=12)
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 6 – Pareto front ────────────────────────────────────────────────────

def plot_pareto_front(
    results_subset, equity_subset, scenarios, scenario_colors,
    budget_label, filename, y_axis_zero=False,
    ycol=None, ycol_std=None,
):
    """Pareto Front – Equity vs Carbon (combined plot only)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    all_budgets = get_sorted_budget_list(results_subset)

    for budget_val in all_budgets:
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()

        vuln_means, co2_means, co2_stds = [], [], []
        weights, colors = [], []

        for scen in scenarios_subset:
            if scen not in equity_subset['scenario'].values:
                continue

            eq_row = equity_subset[equity_subset['scenario'] == scen].iloc[0]
            res_row = results_subset[results_subset['scenario'] == scen].iloc[0]

            vuln_means.append(eq_row['high_risk_pct'])
            co2_means.append(res_row[ycol] / 1e3)
            co2_stds.append(res_row[ycol_std] / 1e3)
            weights.append(eq_row['equity_weight'])
            colors.append(scenario_colors[scen])

        if not weights:
            continue

        # Scatter with error bars (label weights on first budget only to
        # avoid duplicate legend entries)
        is_first = (budget_val == all_budgets[0])
        for i in range(len(vuln_means)):
            ax.errorbar(
                vuln_means[i], co2_means[i], yerr=co2_stds[i],
                fmt='o', markersize=12, capsize=5, color=colors[i],
                label=f'EW={weights[i]}' if is_first else None,
            )

        # Connecting line
        idx = np.argsort(weights)
        ax.plot(
            [vuln_means[j] for j in idx],
            [co2_means[j] for j in idx],
            '--', alpha=0.5, linewidth=2, label=f'Budget £{budget_val}',
        )

    ax.set_xlabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(), fontsize=11,
            ncol=1 if len(by_label) <= 6 else 2,
        )

    ax.grid(True, alpha=0.3)
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 7 – Vulnerable groups coverage ──────────────────────────────────────

def plot_vulnerable_groups_coverage(equity_subset, filename, y_axis_zero=False):
    """Most Vulnerable Groups Coverage (combined plot only)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = get_sorted_budget_list(equity_subset)
    all_equity_weights = sorted(equity_subset['equity_weight'].unique())

    for budget_val in budgets:
        subset = equity_subset[equity_subset['budget'] == budget_val]
        scenarios = subset.sort_values('equity_weight')['scenario'].values
        deprived = [
            subset.loc[subset['scenario'] == s, 'high_risk_pct'].iloc[0] * 100
            for s in scenarios
        ]
        ew_labels = [
            str(subset.loc[subset['scenario'] == s, 'equity_weight'].iloc[0])
            for s in scenarios
        ]
        x_pos = np.arange(len(scenarios))
        ax.bar(x_pos, deprived, 0.6, label=f'£{budget_val}', alpha=0.8)

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(all_equity_weights)))
    ax.set_xticklabels([str(w) for w in all_equity_weights])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 8 – Tradeoff efficiency ─────────────────────────────────────────────

def plot_tradeoff_efficiency(
    results_subset, equity_subset, scenarios, scenario_colors,
    budget_label, filename, y_axis_zero=False,
):
    """Equity-Carbon Tradeoff Efficiency (combined plot only)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = get_sorted_budget_list(results_subset)
    bar_width = 0.8 / max(len(budgets), 1)

    for i, budget_val in enumerate(budgets):
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()

        if len(scenarios_subset) < 2:
            continue

        sorted_scenarios = sorted(
            scenarios_subset,
            key=lambda s: equity_subset.loc[
                equity_subset['scenario'] == s, 'equity_weight'
            ].iloc[0],
        )
        base_scenario = sorted_scenarios[0]

        base_vuln = equity_subset.loc[
            equity_subset['scenario'] == base_scenario, 'high_risk_pct'
        ].iloc[0]
        base_co2 = results_subset.loc[
            results_subset['scenario'] == base_scenario, total_co2_saved_col
        ].iloc[0] / 1e3

        tradeoff_ratios, tradeoff_labels = [], []

        for scen in sorted_scenarios[1:]:
            eq_row = equity_subset[equity_subset['scenario'] == scen].iloc[0]
            res_row = results_subset[results_subset['scenario'] == scen].iloc[0]

            vuln_cov = eq_row['high_risk_pct'] * 100
            co2_saved = res_row[total_co2_saved_col] / 1e3

            vuln_gain = vuln_cov - base_vuln
            co2_loss = base_co2 - co2_saved

            ratio = vuln_gain / co2_loss if co2_loss > 0.001 else 0.0
            tradeoff_ratios.append(ratio)
            tradeoff_labels.append(str(eq_row['equity_weight']))

        offset = (i - len(budgets) / 2 + 0.5) * bar_width
        x_pos = np.arange(len(tradeoff_ratios))
        ax.bar(
            x_pos + offset, tradeoff_ratios, bar_width,
            label=f'£{budget_val}', alpha=0.7,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tradeoff_labels)

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable % Gain per kton CO2 Lost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    if len(budgets) > 1:
        ax.legend(fontsize=12)
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 9 – Radar chart ─────────────────────────────────────────────────────

def plot_radar_chart(results_subset, equity_subset, filename, scenario_colors):
    """Normalised radar chart (combined plot only, one subplot per budget)."""
    budgets = get_sorted_budget_list(equity_subset)

    for budget_val in budgets:
        eq_sub = equity_subset[equity_subset['budget'] == budget_val]
        res_sub = results_subset[results_subset['budget'] == budget_val]
        scenarios = eq_sub['scenario'].unique()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')

        metrics = [
            'Carbon\nSavings', 'Cost\nEfficiency', 'Vulnerable\nCoverage',
            'Equity\nBalance', '# Buildings',
        ]
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Normalisation denominators (avoid division by zero)
        norm_co2 = res_sub[total_co2_saved_col].max() or 1
        norm_cost = (1 / res_sub[capex_per_net_ton_mean_col]).max() or 1
        norm_vuln = eq_sub['high_risk_pct'].max() or 1
        norm_equity = eq_sub['equity_concentration'].max() or 1
        norm_bldg = res_sub['num_buildings_sum'].max() or 1

        for scen in scenarios:
            r_row = res_sub[res_sub['scenario'] == scen].iloc[0]
            e_row = eq_sub[eq_sub['scenario'] == scen].iloc[0]

            values = [
                r_row[total_co2_saved_col] / norm_co2,
                (1 / r_row[capex_per_net_ton_mean_col]) / norm_cost,
                e_row['high_risk_pct'] / norm_vuln,
                1 - (e_row['equity_concentration'] / norm_equity),
                r_row['num_buildings_sum'] / norm_bldg,
            ]
            values += values[:1]

            label = f'EW={e_row["equity_weight"]}'
            color = scenario_colors[scen]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=label,
                    color=color, markersize=8)
            ax.fill(angles, values, alpha=0.2, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(
            loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11,
            ncol=1 if len(scenarios) <= 5 else 2,
        )
        ax.grid(True)

        base, ext = os.path.splitext(filename)
        rl_filename = f'{base}_budget{budget_val}{ext}'
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {rl_filename}")


# ── Plot 10 – Flexible Pareto ────────────────────────────────────────────────

def plot_pareto_front_flexi(
    results_subset, equity_subset, scenarios, scenario_colors,
    budget_label, filename,
    x_col='high_risk_pct', y_col='total_co2_saved_col',
    y_col_std='', x_label='Vulnerable Coverage (%)',
    y_label='CO2 Saved (kton)', x_scaler=1.0, y_scaler=1e3,
    y_axis_zero=False,
):
    """Flexible-axis Pareto front (combined plot only)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    all_budgets = get_sorted_budget_list(results_subset)

    for budget_val in all_budgets:
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()

        x_vals, y_vals, y_stds = [], [], []
        weights, colors = [], []

        for scen in scenarios_subset:
            if scen not in equity_subset['scenario'].values:
                continue

            # Resolve column source
            x_src = (
                equity_subset if x_col in equity_subset.columns
                else results_subset
            )
            y_src = (
                results_subset if y_col in results_subset.columns
                else equity_subset
            )

            x_row = x_src[x_src['scenario'] == scen].iloc[0]
            y_row = y_src[y_src['scenario'] == scen].iloc[0]
            eq_row = equity_subset[equity_subset['scenario'] == scen].iloc[0]

            x_vals.append(x_row[x_col] / x_scaler)
            y_vals.append(y_row[y_col] / y_scaler)
            y_stds.append(y_row[y_col_std] / y_scaler)
            weights.append(eq_row['equity_weight'])
            colors.append(scenario_colors[scen])

        if not weights:
            continue

        is_first = (budget_val == all_budgets[0])
        for i in range(len(x_vals)):
            ax.errorbar(
                x_vals[i], y_vals[i], yerr=y_stds[i],
                fmt='o', markersize=12, capsize=5, color=colors[i],
                label=f'EW={weights[i]}' if is_first else None,
            )

        idx = np.argsort(weights)
        ax.plot(
            [x_vals[j] for j in idx], [y_vals[j] for j in idx],
            '--', alpha=0.5, linewidth=2, label=f'Budget £{budget_val}',
        )

    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(), fontsize=11,
            ncol=1 if len(by_label) <= 6 else 2,
        )

    ax.grid(True, alpha=0.3)
    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Plot 11 – Pareto: retrofit cost-effectiveness ────────────────────────────

def plot_pareto_retrofit_carbon_by_costeff(
    results_subset, equity_subset, scenarios, scenario_colors,
    budget_label, filename, y_axis_zero=False,
    capex_std=capex_per_net_ton_std_col,
):
    """Buildings Retrofitted vs Cost-Effectiveness (combined plot only)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    all_budgets = get_sorted_budget_list(results_subset)

    for budget_val in all_budgets:
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()

        bldg_means, cost_means, cost_stds = [], [], []
        weights, colors = [], []

        for scen in scenarios_subset:
            if scen not in equity_subset['scenario'].values:
                continue

            eq_row = equity_subset[equity_subset['scenario'] == scen].iloc[0]
            res_row = subset[subset['scenario'] == scen].iloc[0]

            bldg_means.append(res_row['num_buildings_sum'])
            cost_means.append(res_row[capex_per_net_ton_mean_col])
            cost_stds.append(res_row[capex_std])
            weights.append(eq_row['equity_weight'])
            colors.append(scenario_colors[scen])

        if not weights:
            continue

        idx = np.argsort(weights)
        bldg_sorted = [bldg_means[j] for j in idx]
        cost_sorted = [cost_means[j] for j in idx]
        cost_std_sorted = [cost_stds[j] for j in idx]
        colors_sorted = [colors[j] for j in idx]
        weights_sorted = [weights[j] for j in idx]

        is_first = (budget_val == all_budgets[0])
        for i in range(len(bldg_sorted)):
            ax.errorbar(
                bldg_sorted[i], cost_sorted[i], yerr=cost_std_sorted[i],
                fmt='o', markersize=12, capsize=5, color=colors_sorted[i],
                label=f'EW={weights_sorted[i]}' if is_first else None,
            )

        ax.plot(
            bldg_sorted, cost_sorted,
            '--', alpha=0.5, linewidth=2, label=f'Budget £{budget_val}',
        )

    ax.set_xlabel('Number of Buildings', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost Effectiveness (£/net ton CO2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if y_axis_zero:
        ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(), fontsize=11,
            ncol=1 if len(by_label) <= 6 else 2,
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── Bar-chart helpers (for use with row-level project DataFrames) ─────────────

def plot_carbon_by_persona(
    selected_projects_df, scenario_colors, filename,
    scenario_label_map=None, y_axis_zero=True,
):
    """Distribution of total carbon saved by persona (all budgets combined)."""
    if scenario_label_map is None:
        scenario_label_map = {}

    carbon_grouped = selected_projects_df.groupby(
        ['scenario', 'persona_name']
    )['mean_total_co2_saved'].sum().reset_index()

    plot_data = carbon_grouped.pivot(
        index='persona_name', columns='scenario',
        values='mean_total_co2_saved',
    ).fillna(0).sort_index()

    personas = plot_data.index.tolist()
    scenarios = plot_data.columns.tolist()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(personas))
    n_scen = len(scenarios)
    width = 0.8 / n_scen

    for i, scen in enumerate(scenarios):
        offset = (i - n_scen / 2 + 0.5) * width
        ax.bar(
            x + offset, plot_data[scen], width,
            label=scenario_label_map.get(scen, scen),
            color=scenario_colors.get(scen, f'C{i}'),
            alpha=0.7,
        )

    ax.set_xlabel('Socio-economic Persona', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Carbon Saved (tons CO₂)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(personas, fontsize=11, rotation=45, ha='right')
    ax.legend(fontsize=11, ncol=min(3, max(1, (n_scen + 2) // 3)))
    ax.grid(True, alpha=0.3, axis='y')

    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_metric_by_group(
    selected_projects_df, scenario_colors, filename,
    value_col='mean_total_co2_saved',
    group_col='meta_socio_persona',
    metric_stat='mean',
    xlabel='Socio-economic Persona',
    ylabel=None,
    group_label_map=None,
    scenario_label_map=None,
    y_axis_zero=True,
):
    """Generic grouped-bar chart of any metric (all budgets combined)."""
    if group_label_map is None:
        group_label_map = {}
    if scenario_label_map is None:
        scenario_label_map = {}

    if ylabel is None:
        stat_name = "Total" if metric_stat == 'sum' else metric_stat.capitalize()
        ylabel = f"{stat_name} {value_col.replace('_', ' ').title()}"

    plot_data = selected_projects_df.pivot_table(
        index=group_col, columns='scenario', values=value_col,
        aggfunc=metric_stat, fill_value=0,
    ).sort_index()

    groups = plot_data.index.tolist()
    group_labels = [label_cleaner_map.get(g, str(g)) for g in groups]
    scenarios = plot_data.columns.tolist()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(groups))
    n_scen = len(scenarios)
    width = 0.8 / n_scen

    for i, scen in enumerate(scenarios):
        offset = (i - n_scen / 2 + 0.5) * width
        ax.bar(
            x + offset, plot_data[scen], width,
            label=scenario_label_map.get(scen, scen),
            color=scenario_colors.get(scen, f'C{i}'),
            alpha=0.7,
        )

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11, rotation=45, ha='right')
    ax.legend(fontsize=11, ncol=min(3, max(1, (n_scen + 2) // 3)))
    ax.grid(True, alpha=0.3, axis='y')

    if y_axis_zero:
        ax.set_ylim(bottom=0)

    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    rl_filename = f'{base}_{metric_stat}{ext}'

    plt.tight_layout()
    plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ({metric_stat}): {rl_filename}")


def plot_count_by_group(
    selected_projects_df, scenario_colors, filename,
    group_col='meta_socio_persona',
    xlabel='Socio-economic Persona',
    ylabel='Number of Projects',
    y_axis_zero=True,
):
    """Count of items by group (all budgets combined)."""
    grouped = selected_projects_df.groupby(
        ['scenario', group_col]
    ).size().reset_index(name='count')

    groups = sorted(grouped[group_col].unique())
    group_labels = [label_cleaner_map.get(g, str(g)) for g in groups]
    scenarios = grouped['scenario'].unique()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(groups))
    n_scen = len(scenarios)
    width = 0.8 / n_scen

    for i, scen in enumerate(scenarios):
        scen_data = grouped[grouped['scenario'] == scen]
        means = [
            scen_data.loc[scen_data[group_col] == g, 'count'].iloc[0]
            if not scen_data.loc[scen_data[group_col] == g].empty else 0
            for g in groups
        ]
        offset = (i - n_scen / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=scenario_label_map.get(scen, scen),
            color=scenario_colors.get(scen, f'C{i}'),
            alpha=0.7,
        )

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.legend(fontsize=11, ncol=min(3, (n_scen + 2) // 3))
    ax.grid(True, alpha=0.3, axis='y')

    if y_axis_zero:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")