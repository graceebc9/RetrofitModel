import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

 

"""
Calculate the difference between two methods and plot a Pareto graph
"""


def calc_diff(df1, df2, column):
    """Calculate absolute and percentage difference for a column between two dataframes."""
    total_df1 = df1[column].sum()
    total_df2 = df2[column].sum()
    diff = total_df2 - total_df1
    diff_pct = (diff / total_df1) * 100 if total_df1 != 0 else 0
    return diff, diff_pct


def aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, equity_factors, million_factor=1000000):
    """
    Loop through file structure, aggregate totals, and return a Meta DataFrame.
    """
    print('Starting aggregation...')
    meta_results = []

    for prob_loft in loft_probs:
        print(f'Starting {prob_loft}')
        for budget in budgets:
            print(f'Starting budget {budget}')
            million_budget = budget / million_factor
            for equity_factor in equity_factors:
                
                # Reconstruct the directory path
                folder_name = f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}'
                output_dir = os.path.join(greedy_runs_folder, folder_name)
                
                # Define file paths
                selected_path = os.path.join(output_dir, 'selected_projects.csv')      # Optimization (Smart)
                epc_random_path = os.path.join(output_dir, 'epc_random_selection.csv')  # Baseline (Random/EPC)
                
                # Check if files exist before processing
                if os.path.exists(selected_path) and os.path.exists(epc_random_path):
                    try:
                        # Load data
                        df_opt = pd.read_csv(selected_path)
                        df_epc = pd.read_csv(epc_random_path)
                        
                        # Create a dictionary for this scenario
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


if __name__ == '__main__':
    RISK_PENALTY_SIGMA = 1.0
    SETTING_NAME = 'lcoal'
    INPUT_FILES_PATH=f'/Volumes/T9/2025_10_RetrofitModel/3_optimiseD_iroiities/epc/risk_sigma_{RISK_PENALTY_SIGMA}__processed_best_only/*'
    
    BASE_DIR=f'/Volumes/T9/2025_10_RetrofitModel/4_gredy_epc/risk_{RISK_PENALTY_SIGMA}/'
    
    greedy_runs_folder = os.path.join(BASE_DIR, 'greedy_runs', SETTING_NAME)
    budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
    loft_probs = [0.95, 0.65]
    equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
    
    df = aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, equity_factors)



    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    BASE_DIR = '/Volumes/T9/2025_10_RetrofitModel/4_gredy_epc/meta_summary' 
 
    os.makedirs(BASE_DIR, exist_ok=True)
    # ============================================================================
    # LOAD AND PREPARE DATA
    # ============================================================================
 

    # Create improvement metrics (positive = optimization is better)
    df['co2_improvement'] = -df['co2_diff']
    df['improvement_pct'] = -df['diff_pct']

    # Determine global min/max for consistent color scales
    pct_vmin = df['improvement_pct'].min()
    pct_vmax = df['improvement_pct'].max()
    abs_vmin = df['co2_improvement'].min()
    abs_vmax = df['co2_improvement'].max()

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get unique equity factors and create a colormap
    equity_factors = sorted(df['equity_factor'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(equity_factors)))

    # Plot each configuration with different colors
    for idx, eq in enumerate(equity_factors):
        for loft in sorted(df['loft_prob'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            
            # Line style: solid for 0.95, dashed for 0.65
            linestyle = '-' if loft == 0.95 else '--'
            marker = 'o' if loft == 0.95 else '^'
            
            label = f'Equity {eq}, Loft {loft}'
            
            ax.plot(subset['budget_m'], subset['improvement_pct'], 
                marker=marker, linestyle=linestyle, linewidth=2.5, markersize=8,
                color=colors[idx], alpha=0.8, label=label)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_xlabel('Budget (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=9, ncol=2, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f'pareto_all_configs_colored.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pareto_all_configs_colored.png")

    # ============================================================================
    # FIGURE 1: HEATMAP - PERCENTAGE IMPROVEMENT (Loft = 0.95)
    # ============================================================================
    fig1, ax = plt.subplots(figsize=(12, 8))
    pivot_095 = df[df['loft_prob'] == 0.95].pivot(index='equity_factor', 
                                                    columns='budget_m', 
                                                    values='improvement_pct')
    sns.heatmap(pivot_095, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=pct_vmin, vmax=pct_vmax, center=(pct_vmin+pct_vmax)/2,
                ax=ax, cbar_kws={'label': 'Improvement (%)'}, 
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Budget (Millions)', fontsize=13)
    ax.set_ylabel('Equity Factor', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_pct_loft095.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'heatmap_pct_loft095.png')}")

    # ============================================================================
    # FIGURE 2: HEATMAP - PERCENTAGE IMPROVEMENT (Loft = 0.65)
    # ============================================================================
    fig2, ax = plt.subplots(figsize=(12, 8))
    pivot_065 = df[df['loft_prob'] == 0.65].pivot(index='equity_factor', 
                                                    columns='budget_m', 
                                                    values='improvement_pct')
    sns.heatmap(pivot_065, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=pct_vmin, vmax=pct_vmax, center=(pct_vmin+pct_vmax)/2,
                ax=ax, cbar_kws={'label': 'Improvement (%)'}, 
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Budget (£M)', fontsize=13)
    ax.set_ylabel('Equity Factor', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_pct_loft065.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'heatmap_pct_loft065.png')}")

    # ============================================================================
    # FIGURE 3: HEATMAP - ABSOLUTE IMPROVEMENT (Loft = 0.95)
    # ============================================================================
    fig3, ax = plt.subplots(figsize=(12, 8))
    pivot_abs_095 = df[df['loft_prob'] == 0.95].pivot(index='equity_factor', 
                                                        columns='budget_m', 
                                                        values='co2_improvement')
    sns.heatmap(pivot_abs_095, annot=True, fmt='.0f', cmap='RdYlGn', 
                vmin=abs_vmin, vmax=abs_vmax, center=(abs_vmin+abs_vmax)/2,
                ax=ax, cbar_kws={'label': 'CO2 Improvement (tons)'}, 
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Budget (£M)', fontsize=13)
    ax.set_ylabel('Equity Factor', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_abs_loft095.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'heatmap_abs_loft095.png')}")

    # ============================================================================
    # FIGURE 4: HEATMAP - ABSOLUTE IMPROVEMENT (Loft = 0.65)
    # ============================================================================
    fig4, ax = plt.subplots(figsize=(12, 8))
    pivot_abs_065 = df[df['loft_prob'] == 0.65].pivot(index='equity_factor', 
                                                        columns='budget_m', 
                                                        values='co2_improvement')
    sns.heatmap(pivot_abs_065, annot=True, fmt='.0f', cmap='RdYlGn', 
                vmin=abs_vmin, vmax=abs_vmax, center=(abs_vmin+abs_vmax)/2,
                ax=ax, cbar_kws={'label': 'CO2 Improvement (tons)'}, 
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Budget (£M)', fontsize=13)
    ax.set_ylabel('Equity Factor', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_abs_loft065.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'heatmap_abs_loft065.png')}")

    # ============================================================================
    # FIGURE 5: BOX PLOT - IMPROVEMENTS BY BUDGET
    # ============================================================================
    fig5, ax = plt.subplots(figsize=(10, 6))
    df_box = df.copy()
    df_box['budget_m_str'] = df_box['budget_m'].astype(str) + 'M'
    budget_order = ['1.0M', '10.0M', '50.0M', '80.0M', '100.0M']
    sns.boxplot(data=df_box, x='budget_m_str', y='improvement_pct', 
                order=budget_order, ax=ax, palette='Set2', hue='budget_m_str',  )
    ax.set_xlabel('Budget (£)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'boxplot_improvements.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'boxplot_improvements.png')}")

    # ============================================================================
    # FIGURE 6: LOFT PROBABILITY COMPARISON
    # ============================================================================
    fig6, ax = plt.subplots(figsize=(12, 8))
    budgets = sorted(df['budget_m'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(budgets)))  # Start at 0.2 to avoid too light colors

    # Plot each budget with progressively darker colors
    for idx, budget in enumerate(budgets):
        loft_95 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.95)].sort_values('equity_factor')
        loft_65 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.65)].sort_values('equity_factor')
        
        equity_factors = loft_95['equity_factor'].values
        
        # Solid line for Loft 0.95, dashed for Loft 0.65
        ax.plot(equity_factors, loft_95['improvement_pct'].values, 
                marker='o', linewidth=2.5, markersize=8,
                color=colors[idx], linestyle='-',
                label=f'{budget}M (Loft 0.95)', alpha=0.9)
        ax.plot(equity_factors, loft_65['improvement_pct'].values, 
                marker='s', linewidth=2.5, markersize=8, 
                color=colors[idx], linestyle='--',
                label=f'{budget}M (Loft 0.65)', alpha=0.9)

    ax.set_xlabel('Equity Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'loft_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'loft_comparison.png')}")

    # ============================================================================
    # FIGURE 7: PARETO FRONTIER WITH ANNOTATIONS
    # ============================================================================
    fig7, ax = plt.subplots(figsize=(12, 8))

    # Plot all configurations with transparency
    for loft in sorted(df['loft_prob'].unique()):
        for eq in sorted(df['equity_factor'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            marker = 'o' if loft == 0.95 else '^'
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
        ax.annotate(f"Equity: {row['equity_factor']}\nLoft: {row['loft_prob']}", 
                    xy=(row['budget_m'], row['improvement_pct']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                            alpha=0.8, edgecolor='black', linewidth=2))

    ax.set_xlabel('Budget (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'pareto_frontier_annotated.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'pareto_frontier_annotated.png')}")

    # ============================================================================
    # FIGURE 8: PARETO FRONTIER WITHOUT ANNOTATIONS
    # ============================================================================
    fig8, ax = plt.subplots(figsize=(12, 8))

    # Plot all configurations
    for loft in sorted(df['loft_prob'].unique()):
        for eq in sorted(df['equity_factor'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            marker = 'o' if loft == 0.95 else '^'
            ax.plot(subset['budget_m'], subset['improvement_pct'], 
                marker=marker, alpha=0.3, linewidth=1,
                color='gray', markersize=6)

    # Highlight Pareto frontier points (no annotations)
    pareto_points = df.loc[df.groupby('budget_m')['improvement_pct'].idxmax()].sort_values('budget_m')
    colors = plt.cm.viridis(np.linspace(0, 1, len(pareto_points)))

    for idx, (i, row) in enumerate(pareto_points.iterrows()):
        ax.scatter(row['budget_m'], row['improvement_pct'], 
                s=500, c=[colors[idx]], alpha=0.9, 
                edgecolors='black', linewidth=3, zorder=100)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_xlabel('Budget (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'pareto_frontier_clean.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'pareto_frontier_clean.png')}")

    # ============================================================================
    # FIGURE 9: COST-BENEFIT ANALYSIS
    # ============================================================================
    fig9, ax = plt.subplots(figsize=(12, 8))

    # Create red-green colormap
    colors_rg = ['red', 'yellow', 'green']
    n_bins = 100
    cmap_rg = LinearSegmentedColormap.from_list('red_green', colors_rg, N=n_bins)

    for loft in sorted(df['loft_prob'].unique()):
        subset = df[df['loft_prob'] == loft].copy()
        marker = 'o' if loft == 0.95 else '^'
        label = f'Loft Prob: {loft}'
        
        equity_norm_subset = (subset['equity_factor'] - df['equity_factor'].min()) / \
                            (df['equity_factor'].max() - df['equity_factor'].min())
        
        scatter = ax.scatter(subset['budget_raw']/1e6, subset['co2_improvement'], 
                            c=equity_norm_subset, s=250, alpha=0.8, 
                            cmap=cmap_rg, marker=marker, 
                            edgecolors='black', linewidth=2,
                            label=label, vmin=0, vmax=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Equity Factor (Green = Higher)', fontsize=12, fontweight='bold')
    equity_values = sorted(df['equity_factor'].unique())
    equity_norm_ticks = (np.array(equity_values) - df['equity_factor'].min()) / \
                        (df['equity_factor'].max() - df['equity_factor'].min())
    cbar.set_ticks(equity_norm_ticks)
    cbar.set_ticklabels([f'{v:.1f}' for v in equity_values])
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('Budget (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total CO2 Improvement (tons)', fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'cost_benefit_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(BASE_DIR, 'cost_benefit_analysis.png')}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {BASE_DIR}")
    print("\nFiles saved:")
    print("  1. heatmap_pct_loft095.png")
    print("  2. heatmap_pct_loft065.png")
    print("  3. heatmap_abs_loft095.png")
    print("  4. heatmap_abs_loft065.png")
    print("  5. boxplot_improvements.png")
    print("  6. loft_comparison.png")
    print("  7. pareto_frontier_annotated.png")
    print("  8. pareto_frontier_clean.png")
    print("  9. cost_benefit_analysis.png")