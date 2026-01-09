"""
Pareto Analysis: Calculate difference between optimization methods and generate visualizations.

Unified notation:
- Loft probability markers: ● (circle) for 0.95, ▲ (triangle) for 0.65
- Budget: £M on axes, integer values (1, 10, 50, 80, 100)
- CO2: kilotonnes (kt) with comma formatting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# GLOBAL STYLE CONFIGURATION
# =============================================================================
MARKER_LOFT_095 = 'o'      # Circle for loft probability 0.95
MARKER_LOFT_065 = '^'      # Triangle for loft probability 0.65
LINESTYLE_LOFT_095 = '-'   # Solid line for 0.95
LINESTYLE_LOFT_065 = '--'  # Dashed line for 0.65

BUDGET_LABEL = 'Budget (£M)'
CO2_LABEL_KT = 'CO2 Improvement (kt)'
CO2_LABEL_PCT = 'Improvement (%)'
EQUITY_LABEL = 'Equity Factor'


def get_loft_style(loft_prob):
    """Return consistent marker and linestyle for a given loft probability."""
    if loft_prob == 0.95:
        return MARKER_LOFT_095, LINESTYLE_LOFT_095
    else:
        return MARKER_LOFT_065, LINESTYLE_LOFT_065


def format_budget_ticks(budget_m_values):
    """Format budget values as integers for tick labels."""
    return [f'{int(b)}' for b in budget_m_values]


def calc_diff(df1, df2, column):
    """Calculate absolute and percentage difference for a column between two dataframes."""
    total_df1 = df1[column].sum()
    total_df2 = df2[column].sum()
    diff = total_df2 - total_df1
    diff_pct = (diff / total_df1) * 100 if total_df1 != 0 else 0
    return diff, diff_pct


def aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, equity_factors,
                                million_factor=1_000_000):
    """Loop through file structure, aggregate totals, and return a Meta DataFrame."""
    print('Starting aggregation...')
    meta_results = []

    for prob_loft in loft_probs:
        print(f'Processing loft probability: {prob_loft}')
        for budget in budgets:
            million_budget = budget / million_factor
            for equity_factor in equity_factors:
                folder_name = f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}'
                output_dir = os.path.join(greedy_runs_folder, folder_name)

                selected_path = os.path.join(output_dir, 'selected_projects.csv')
                epc_random_path = os.path.join(output_dir, 'epc_random_selection.csv')

                if os.path.exists(selected_path) and os.path.exists(epc_random_path):
                    try:
                        df_opt = pd.read_csv(selected_path)
                        df_epc = pd.read_csv(epc_random_path)

                        row = {
                            'budget_raw': budget,
                            'budget_m': million_budget,
                            'loft_prob': prob_loft,
                            'equity_factor': equity_factor,
                            'scenario_id': folder_name
                        }

                        col = 'total_co2_saved'
                        diff, diff_pct = calc_diff(df_opt, df_epc, col)
                        row['co2_diff'] = diff
                        row['diff_pct'] = diff_pct
                        meta_results.append(row)

                    except Exception as e:
                        print(f'Error processing {folder_name}: {e}')
                else:
                    print(f'Missing files in {folder_name}')

    meta_df = pd.DataFrame(meta_results)
    meta_df.to_csv('diff_epc_opt.csv', index=False)
    print('Results saved to diff_epc_opt.csv')
    return meta_df


def create_all_configs_plot(df, output_dir):
    """Create Pareto plot with all configurations colored by equity factor."""
    fig, ax = plt.subplots(figsize=(14, 9))

    equity_factors = sorted(df['equity_factor'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(equity_factors)))

    for idx, eq in enumerate(equity_factors):
        for loft in sorted(df['loft_prob'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            marker, linestyle = get_loft_style(loft)
            label = f'Equity {eq}, Loft {loft}'

            ax.plot(subset['budget_m'], subset['improvement_pct'],
                    marker=marker, linestyle=linestyle, linewidth=2.5, markersize=8,
                    color=colors[idx], alpha=0.8, label=label)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'pareto_all_configs_colored.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_heatmap_pct(df, loft_prob, pct_vmin, pct_vmax, output_dir):
    """Create percentage improvement heatmap for a specific loft probability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df[df['loft_prob'] == loft_prob].pivot(
        index='equity_factor', columns='budget_m', values='improvement_pct'
    )
    # Rename columns to integer format
    pivot.columns = [int(c) for c in pivot.columns]

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=pct_vmin, vmax=pct_vmax, center=(pct_vmin + pct_vmax) / 2,
                ax=ax, cbar_kws={'label': CO2_LABEL_PCT},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel(BUDGET_LABEL, fontsize=13)
    ax.set_ylabel(EQUITY_LABEL, fontsize=13)

    plt.tight_layout()
    loft_str = str(loft_prob).replace('.', '')
    filepath = os.path.join(output_dir, f'heatmap_pct_loft{loft_str}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_heatmap_abs(df, loft_prob, abs_vmin, abs_vmax, output_dir):
    """Create absolute CO2 improvement heatmap (in kilotonnes) for a specific loft probability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to kilotonnes
    df_kt = df.copy()
    df_kt['co2_improvement_kt'] = df_kt['co2_improvement'] / 1000

    pivot = df_kt[df_kt['loft_prob'] == loft_prob].pivot(
        index='equity_factor', columns='budget_m', values='co2_improvement_kt'
    )
    # Rename columns to integer format
    pivot.columns = [int(c) for c in pivot.columns]

    # Convert min/max to kilotonnes
    kt_vmin = abs_vmin / 1000
    kt_vmax = abs_vmax / 1000

    sns.heatmap(pivot, annot=True, fmt=',.1f', cmap='RdYlGn',
                vmin=kt_vmin, vmax=kt_vmax, center=(kt_vmin + kt_vmax) / 2,
                ax=ax, cbar_kws={'label': CO2_LABEL_KT},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel(BUDGET_LABEL, fontsize=13)
    ax.set_ylabel(EQUITY_LABEL, fontsize=13)

    plt.tight_layout()
    loft_str = str(loft_prob).replace('.', '')
    filepath = os.path.join(output_dir, f'heatmap_abs_loft{loft_str}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_boxplot(df, output_dir):
    """Create box plot of improvements by budget."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_box = df.copy()
    df_box['budget_m_int'] = df_box['budget_m'].astype(int)
    budget_order = sorted(df_box['budget_m_int'].unique())

    sns.boxplot(data=df_box, x='budget_m_int', y='improvement_pct',
                order=budget_order, ax=ax, palette='Set2', hue='budget_m_int',
                )

    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'boxplot_improvements.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


# def create_loft_comparison(df, output_dir):
#     """Create loft probability comparison plot with unified symbols in legend."""
#     fig, ax = plt.subplots(figsize=(12, 8))

#     budgets = sorted(df['budget_m'].unique())
#     colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(budgets)))

#     for idx, budget in enumerate(budgets):
#         budget_int = int(budget)

#         # Loft 0.95
#         loft_95 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.95)].sort_values('equity_factor')
#         marker_95, ls_95 = get_loft_style(0.95)
#         ax.plot(loft_95['equity_factor'], loft_95['improvement_pct'],
#                 marker=marker_95, linewidth=2.5, markersize=8,
#                 color=colors[idx], linestyle=ls_95,
#                 label=f'£{budget_int}M (Loft 0.95 ●)', alpha=0.9)

#         # Loft 0.65
#         loft_65 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.65)].sort_values('equity_factor')
#         marker_65, ls_65 = get_loft_style(0.65)
#         ax.plot(loft_65['equity_factor'], loft_65['improvement_pct'],
#                 marker=marker_65, linewidth=2.5, markersize=8,
#                 color=colors[idx], linestyle=ls_65,
#                 label=f'£{budget_int}M (Loft 0.65 ▲)', alpha=0.9)

#     ax.set_xlabel(EQUITY_LABEL, fontsize=14, fontweight='bold')
#     ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
#     ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.95)
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
#     ax.tick_params(labelsize=12)

#     plt.tight_layout()
#     filepath = os.path.join(output_dir, 'loft_comparison.png')
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✓ Saved: {filepath}")



def create_loft_comparison(df, output_dir):
    """Create loft probability comparison plot with unified symbols in legend."""
    fig, ax = plt.subplots(figsize=(12, 8))

    budgets = sorted(df['budget_m'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(budgets)))

    # Store handles for custom legend
    budget_handles = []
    budget_labels = []

    for idx, budget in enumerate(budgets):
        budget_int = int(budget)

        # Loft 0.95
        loft_95 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.95)].sort_values('equity_factor')
        marker_95, ls_95 = get_loft_style(0.95)
        ax.plot(loft_95['equity_factor'], loft_95['improvement_pct'],
                marker=marker_95, linewidth=2.5, markersize=8,
                color=colors[idx], linestyle=ls_95, alpha=0.9)

        # Loft 0.65
        loft_65 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.65)].sort_values('equity_factor')
        marker_65, ls_65 = get_loft_style(0.65)
        ax.plot(loft_65['equity_factor'], loft_65['improvement_pct'],
                marker=marker_65, linewidth=2.5, markersize=8,
                color=colors[idx], linestyle=ls_65, alpha=0.9)

        # Create dummy line without marker for legend (color only)
        budget_line, = ax.plot([], [], linestyle='-', linewidth=2.5,
                               color=colors[idx], marker='')
        budget_handles.append(budget_line)
        budget_labels.append(f'£{budget_int}M')

    # Create symbol legend entries (using dummy plots)
    symbol_handle_95, = ax.plot([], [], marker=MARKER_LOFT_095, linestyle=LINESTYLE_LOFT_095,
                                 color='gray', linewidth=2.5, markersize=8, label='')
    symbol_handle_65, = ax.plot([], [], marker=MARKER_LOFT_065, linestyle=LINESTYLE_LOFT_065,
                                 color='gray', linewidth=2.5, markersize=8, label='')

    # Build combined legend: budgets first, then symbols
    all_handles = budget_handles + [symbol_handle_95, symbol_handle_65]
    all_labels = budget_labels + ['Loft 0.95', 'Loft 0.65']

    ax.set_xlabel(EQUITY_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.legend(all_handles, all_labels, fontsize=10, ncol=2, loc='center left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'loft_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")

def create_pareto_frontier(df, output_dir, annotated=True):
    """Create Pareto frontier plot with optional annotations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all configurations with transparency
    for loft in sorted(df['loft_prob'].unique()):
        for eq in sorted(df['equity_factor'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            marker, _ = get_loft_style(loft)
            ax.plot(subset['budget_m'], subset['improvement_pct'],
                    marker=marker, alpha=0.2, linewidth=1,
                    color='gray', markersize=6)

    # Highlight Pareto frontier points
    pareto_points = df.loc[df.groupby('budget_m')['improvement_pct'].idxmax()].sort_values('budget_m')
    colors = plt.cm.viridis(np.linspace(0, 1, len(pareto_points)))

    for idx, (i, row) in enumerate(pareto_points.iterrows()):
        ax.scatter(row['budget_m'], row['improvement_pct'],
                   s=500, c=[colors[idx]], alpha=0.9,
                   edgecolors='black', linewidth=3, zorder=100)

        if annotated:
            ax.annotate(f"Equity: {row['equity_factor']}\nLoft: {row['loft_prob']}",
                        xy=(row['budget_m'], row['improvement_pct']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                                  alpha=0.8, edgecolor='black', linewidth=2))

    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    suffix = 'annotated' if annotated else 'clean'
    filepath = os.path.join(output_dir, f'pareto_frontier_{suffix}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_cost_benefit_analysis(df, output_dir):
    """Create cost-benefit analysis scatter plot with CO2 in kilotonnes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to kilotonnes
    df_kt = df.copy()
    df_kt['co2_improvement_kt'] = df_kt['co2_improvement'] / 1000

    colors_rg = ['red', 'yellow', 'green']
    cmap_rg = LinearSegmentedColormap.from_list('red_green', colors_rg, N=100)

    for loft in sorted(df['loft_prob'].unique()):
        subset = df_kt[df_kt['loft_prob'] == loft].copy()
        marker, _ = get_loft_style(loft)
        label = f'Loft Prob: {loft}'

        equity_norm = (subset['equity_factor'] - df['equity_factor'].min()) / \
                      (df['equity_factor'].max() - df['equity_factor'].min())

        scatter = ax.scatter(subset['budget_m'], subset['co2_improvement_kt'],
                             c=equity_norm, s=200, alpha=0.6,
                             cmap=cmap_rg, marker=marker, edgecolors='black',
                              linewidth=1,
                             label=label, vmin=0, vmax=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{EQUITY_LABEL}', fontsize=12, fontweight='bold')
    equity_values = sorted(df['equity_factor'].unique())
    equity_norm_ticks = (np.array(equity_values) - df['equity_factor'].min()) / \
                        (df['equity_factor'].max() - df['equity_factor'].min())
    cbar.set_ticks(equity_norm_ticks)
    cbar.set_ticklabels([f'{v:.1f}' for v in equity_values])
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Total {CO2_LABEL_KT}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Format x-axis as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cost_benefit_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    RISK_PENALTY_SIGMA = 1.0
    SETTING_NAME = 'lcoal'
    BASE_DIR = f'/Volumes/T9/2025_10_RetrofitModel/4_gredy_epc/risk_{RISK_PENALTY_SIGMA}/'
    OUTPUT_DIR = '/Volumes/T9/2025_10_RetrofitModel/4_gredy_epc/meta_summary'

    greedy_runs_folder = os.path.join(BASE_DIR, 'greedy_runs', SETTING_NAME)
    budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
    loft_probs = [0.95, 0.65]
    equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # AGGREGATE DATA
    # =========================================================================
    df = aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, equity_factors)

    # Create improvement metrics (positive = optimization is better)
    df['co2_improvement'] = -df['co2_diff']
    df['improvement_pct'] = -df['diff_pct']

    # Calculate global min/max for consistent color scales
    pct_vmin = df['improvement_pct'].min()
    pct_vmax = df['improvement_pct'].max()
    abs_vmin = df['co2_improvement'].min()
    abs_vmax = df['co2_improvement'].max()

    # =========================================================================
    # GENERATE ALL FIGURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    # Figure 0: All configurations colored
    create_all_configs_plot(df, OUTPUT_DIR)

    # Figures 1-2: Percentage heatmaps
    for loft in loft_probs:
        create_heatmap_pct(df, loft, pct_vmin, pct_vmax, OUTPUT_DIR)

    # Figures 3-4: Absolute CO2 heatmaps (in kilotonnes)
    for loft in loft_probs:
        create_heatmap_abs(df, loft, abs_vmin, abs_vmax, OUTPUT_DIR)

    # Figure 5: Box plot
    create_boxplot(df, OUTPUT_DIR)

    # Figure 6: Loft comparison
    create_loft_comparison(df, OUTPUT_DIR)

    # Figures 7-8: Pareto frontiers
    create_pareto_frontier(df, OUTPUT_DIR, annotated=True)
    create_pareto_frontier(df, OUTPUT_DIR, annotated=False)

    # Figure 9: Cost-benefit analysis
    create_cost_benefit_analysis(df, OUTPUT_DIR)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles saved:")
    print("  0. pareto_all_configs_colored.png")
    print("  1. heatmap_pct_loft095.png")
    print("  2. heatmap_pct_loft065.png")
    print("  3. heatmap_abs_loft095.png")
    print("  4. heatmap_abs_loft065.png")
    print("  5. boxplot_improvements.png")
    print("  6. loft_comparison.png")
    print("  7. pareto_frontier_annotated.png")
    print("  8. pareto_frontier_clean.png")
    print("  9. cost_benefit_analysis.png")


if __name__ == '__main__':
    main()