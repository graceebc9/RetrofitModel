import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
import pandas as pd

def true_greedy_knapsack(df_knapsack, budget, cost_column='cost of interventon_mean', efficiency_column='cost_per_net_ton_co2_kg'):
    """
    Selects the most cost-effective buildings to receive interventions until the budget is exhausted.
    
    Parameters:
    -----------
    df_knapsack : DataFrame with one row (best intervention) per building (upn).
    budget : float, total budget available (in £).
    cost_column : str, name of the column containing the absolute cost of the intervention.
    efficiency_column : str, name of the column containing the cost-effectiveness metric 
                        (lower is better, e.g., cost_per_net_ton_co2_kg).
    
    Returns:
    --------
    selected : DataFrame of selected interventions.
    remaining : float, remaining budget.
    """
    
    # 1. Sort by the cost-effectiveness metric (ascending: lower cost per CO2 saved is better)
    # This is the core of the greedy strategy.
    df_sorted = df_knapsack.sort_values(efficiency_column, ascending=True).reset_index(drop=True)
    
    selected_interventions = []
    remaining_budget = budget
    
    print(f"Starting true greedy selection with budget: £{budget:,.0f}")
    
    # 2. Iterate through the sorted, most cost-effective interventions
    for idx, row in df_sorted.iterrows():
        project_cost = row[cost_column]
        
        # Check if we can afford the current most cost-effective project
        if project_cost <= remaining_budget:
            selected_interventions.append(row)
            remaining_budget -= project_cost
        else:
            # Crucial for efficiency: Since the list is sorted, 
            # if we can't afford the current project, we can't afford any subsequent, 
            # less cost-effective projects either. Stop iteration immediately.
            break 
            
    selected_df = pd.DataFrame(selected_interventions)
    
    total_spent = budget - remaining_budget
    total_co2 = selected_df['total_ton_co2_saved'].sum()
    
    if len(selected_df) > 0:
        print("\n✅ Selection Complete:")
        print(f"  Buildings covered: {len(selected_df):,}")
        print(f"  Total spent: £{total_spent:,.0f}")
        print(f"  Total CO2 saved: {total_co2:,.2f} tons")
        print(f"  Cost per ton CO2 (Achieved): £{total_spent/total_co2:,.2f}")
    
    return selected_df, remaining_budget


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os  # <-- Added for directory and path handling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_greedy_distribution_analysis(baseline_df, selected_df, 
                                      scenario_name="Greedy Selection", 
                                      output_dir=None):
    """
    Calculates and plots distributions, including averages and std dev
    across epistemic runs.
    
    Generates 6 separate plots. If output_dir is provided, saves each
    plot as a PNG file instead of displaying it.

    Parameters:
    -----------
    baseline_df : DataFrame
        Contains the best intervention for ALL buildings.
    selected_df : DataFrame
        Contains the projects chosen by the greedy algorithm. 
        MUST contain an 'epistemic_run' column.
    scenario_name : str
        Name for the scenario (e.g., "50% Budget"), used in titles and filenames.
    output_dir : str or Path, optional
        If provided, saves plots to this directory instead of showing them.
    """
    
    print("=" * 60)
    print(f"ANALYSIS: {scenario_name}")
    print("=" * 60)
    
    if 'epistemic_run' not in selected_df.columns:
        print("ERROR: 'epistemic_run' column not found in selected_df.")
        print("Cannot perform average-per-run analysis.")
        return

    # ----------------------------------------
    # 1. Calculate Data Distributions
    # ----------------------------------------
    
    n_runs = selected_df['epistemic_run'].nunique()
    all_runs = selected_df['epistemic_run'].unique()
    all_deciles_sorted = np.sort(selected_df['avg_gas_percentile'].unique())
    all_scenarios_sorted = np.sort(selected_df['scenario'].unique())
    
    print(f"Found {n_runs} epistemic runs.")

    # --- Baseline & Aggregate Decile Comparison (Plots A, B, C) ---
    baseline_decile_dist = baseline_df['avg_gas_percentile'].value_counts().sort_index()
    baseline_decile_pct = (baseline_decile_dist / len(baseline_df) * 100).rename('Baseline (100%)')

    # Note: For comparison, we use the *total* selected distribution
    total_scenario_decile_dist = selected_df['avg_gas_percentile'].value_counts().sort_index()
    total_scenario_decile_pct = (total_scenario_decile_dist / len(selected_df) * 100).rename(scenario_name)

    comparison_df = pd.concat([baseline_decile_pct, total_scenario_decile_pct], axis=1).fillna(0)
    diff_from_baseline = comparison_df[scenario_name] - comparison_df['Baseline (100%)']
    
    print("\nGas Decile Distribution Comparison (%):")
    print(comparison_df.round(1))

    # --- Per-Run Scenario Distribution (Plot D) ---
    scenario_counts_per_run = selected_df.groupby('epistemic_run')['scenario'].value_counts().unstack(level='scenario', fill_value=0)
    # Reindex to ensure all runs/scenarios are present for mean/std calc
    scenario_counts_per_run = scenario_counts_per_run.reindex(index=all_runs, fill_value=0)
    scenario_counts_per_run = scenario_counts_per_run.reindex(columns=all_scenarios_sorted, fill_value=0)
    
    scenario_dist_avg = scenario_counts_per_run.mean(axis=0)
    scenario_dist_std = scenario_counts_per_run.std(axis=0)
    
    print("\nAverage Intervention Distribution (per Run):")
    print(pd.DataFrame({'Mean': scenario_dist_avg, 'StdDev': scenario_dist_std}).round(1))

    # --- Per-Run Intervention Mix per Decile (Plots E, F) ---
    counts_per_run_decile = selected_df.pivot_table(
        index=['epistemic_run', 'avg_gas_percentile'], 
        columns='scenario', 
        aggfunc='size', 
        fill_value=0
    )
    # Reindex to ensure all combinations are present
    full_multi_index = pd.MultiIndex.from_product(
        [all_runs, all_deciles_sorted], 
        names=['epistemic_run', 'avg_gas_percentile']
    )
    counts_per_run_decile = counts_per_run_decile.reindex(index=full_multi_index, fill_value=0)
    counts_per_run_decile = counts_per_run_decile.reindex(columns=all_scenarios_sorted, fill_value=0)
    
    intervention_decile_avg = counts_per_run_decile.groupby(level='avg_gas_percentile').mean()
    intervention_decile_std = counts_per_run_decile.groupby(level='avg_gas_percentile').std()

    print("\nAverage Intervention Mix per Decile (per Run):")
    print(intervention_decile_avg.round(1))

    
    # ----------------------------------------
    # 2. Setup Plot Saving or Showing
    # ----------------------------------------
    
    filename_prefix = scenario_name.replace(' ', '_').replace('%', 'pct').replace('.', '')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nPlots will be saved to: {output_dir}")

    def save_or_show(fig, plot_name):
        """Helper function to either save or show the plot."""
        fig.tight_layout()
        if output_dir:
            filename = f"{filename_prefix}_{plot_name}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=150)
            print(f"  ... saved {filename}")
            plt.close(fig)
        else:
            plt.show()

    # ----------------------------------------
    # 3. Generate Plots (Separately)
    # ----------------------------------------

    # --- A. Gas Decile Comparison Bar Chart (Aggregate) ---
    try:
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        comparison_df.T.plot(kind='bar', stacked=False, ax=ax_a, 
                             colormap='tab10', edgecolor='black', linewidth=0.5)
        ax_a.set_xlabel('Scenario', fontsize=11)
        ax_a.set_ylabel('Percentage of Buildings (%)', fontsize=11)
        ax_a.set_title('A. Gas Decile Distribution: Baseline vs. Selection (All Runs)', fontsize=12, fontweight='bold')
        ax_a.legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_a.tick_params(axis='x', rotation=0)
        ax_a.grid(axis='y', alpha=0.3)
        save_or_show(fig_a, "A_Decile_Distribution_Aggregate")
    except Exception as e:
        print(f"Failed to create Plot A: {e}")
        if 'fig_a' in locals(): plt.close(fig_a)

    # --- B. Heatmap of Decile Distribution (Aggregate) ---
    try:
        fig_b, ax_b = plt.subplots(figsize=(8, 5))
        cmap = plt.cm.RdYlGn
        im = ax_b.imshow(comparison_df.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=comparison_df.T.values.max() + 5)

        ax_b.set_xticks(np.arange(len(comparison_df.index)))
        ax_b.set_yticks(np.arange(len(comparison_df.columns)))
        ax_b.set_xticklabels(comparison_df.index)
        ax_b.set_yticklabels(comparison_df.columns)
        ax_b.tick_params(axis='x', rotation=45)

        for i in range(len(comparison_df.columns)):
            for j in range(len(comparison_df.index)):
                value = comparison_df.T.values[i, j]
                ax_b.text(j, i, f'{value:.1f}', ha="center", va="center", 
                          color="black", fontsize=9)

        ax_b.set_title('B. Gas Decile Distribution Heatmap (%) (All Runs)', fontsize=12, fontweight='bold', pad=10)
        ax_b.set_xlabel('Gas Decile', fontsize=11)
        ax_b.set_ylabel('Scenario', fontsize=11)

        cbar = plt.colorbar(im, ax=ax_b)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
        save_or_show(fig_b, "B_Decile_Heatmap_Aggregate")
    except Exception as e:
        print(f"Failed to create Plot B: {e}")
        if 'fig_b' in locals(): plt.close(fig_b)

    # --- C. Difference from Baseline (Bias Analysis - Aggregate) ---
    try:
        fig_c, ax_c = plt.subplots(figsize=(8, 6))
        colors = ['red' if x < 0 else 'green' for x in diff_from_baseline]
        ax_c.bar(range(len(diff_from_baseline)), diff_from_baseline, color=colors, edgecolor='black', alpha=0.7)
        ax_c.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax_c.set_xlabel('Gas Decile', fontsize=11)
        ax_c.set_ylabel('Difference from Baseline (%)', fontsize=11)
        ax_c.set_title('C. Decile Bias vs. Full Baseline (All Runs)', fontsize=12, fontweight='bold')
        ax_c.set_xticks(range(len(diff_from_baseline)))
        ax_c.set_xticklabels(diff_from_baseline.index, rotation=45)
        ax_c.grid(axis='y', alpha=0.3)
        save_or_show(fig_c, "C_Decile_Bias_Aggregate")
    except Exception as e:
        print(f"Failed to create Plot C: {e}")
        if 'fig_c' in locals(): plt.close(fig_c)

    # --- D. Scenario Selection Distribution (Average per Run) ---
    try:
        fig_d, ax_d = plt.subplots(figsize=(8, 6))
        scenario_dist_avg.plot(
            kind='bar', ax=ax_d, 
            color='steelblue', edgecolor='black', alpha=0.7,
            yerr=scenario_dist_std, capsize=4  # <-- Added error bars
        )
        ax_d.set_xlabel('Intervention Scenario', fontsize=11)
        ax_d.set_ylabel('Avg. Number of Buildings Selected (per Run)', fontsize=11) # <-- Updated label
        ax_d.set_title(f'D. Intervention Distribution for {scenario_name} (Avg. per Run)', fontsize=12, fontweight='bold') # <-- Updated title
        ax_d.tick_params(axis='x', rotation=45)
        ax_d.grid(axis='y', alpha=0.3)
        save_or_show(fig_d, "D_Intervention_Distribution_Average")
    except Exception as e:
        print(f"Failed to create Plot D: {e}")
        if 'fig_d' in locals(): plt.close(fig_d)

    # --- E. Intervention Mix per Decile (Average per Run) ---
    try:
        fig_e, ax_e = plt.subplots(figsize=(9, 6))
        intervention_decile_avg.plot(  # <-- Using average data
            kind='bar', stacked=True, ax=ax_e, 
            colormap='tab20', edgecolor='black', linewidth=0.5
        )
        ax_e.set_xlabel('Gas Decile', fontsize=11)
        ax_e.set_ylabel('Avg. Number of Interventions Selected (per Run)', fontsize=11) # <-- Updated label
        ax_e.set_title(f'E. Intervention Mix per Decile for {scenario_name} (Avg. per Run)', fontsize=12, fontweight='bold') # <-- Updated title
        ax_e.legend(title='Intervention', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_e.tick_params(axis='x', rotation=45)
        ax_e.grid(axis='y', alpha=0.3)
        save_or_show(fig_e, "E_Intervention_Mix_Average")
    except Exception as e:
        print(f"Failed to create Plot E: {e}")
        if 'fig_e' in locals(): plt.close(fig_e)

    # --- F. NEW: Intervention Mix Standard Deviation per Decile ---
    try:
        fig_f, ax_f = plt.subplots(figsize=(9, 6))
        intervention_decile_std.plot( # <-- Using std dev data
            kind='bar', stacked=False, ax=ax_f, # <-- Grouped bar
            colormap='tab20', edgecolor='black', linewidth=0.5, alpha=0.8
        )
        ax_f.set_xlabel('Gas Decile', fontsize=11)
        ax_f.set_ylabel('Std. Dev. of Intervention Count (across Runs)', fontsize=11)
        ax_f.set_title(f'F. Variability: Intervention Mix Std. Dev. per Decile', fontsize=12, fontweight='bold')
        ax_f.legend(title='Intervention', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_f.tick_params(axis='x', rotation=45)
        ax_f.grid(axis='y', alpha=0.3)
        save_or_show(fig_f, "F_Intervention_Mix_StdDev")
    except Exception as e:
        print(f"Failed to create Plot F: {e}")
        if 'fig_f' in locals(): plt.close(fig_f)

    print("\nAnalysis plotting complete.")
# def plot_greedy_distribution_analysis(baseline_df, selected_df, scenario_name="Greedy Selection"):
#     """
#     Calculates and plots the Gas Decile and Scenario distributions for the 
#     selected projects against the full baseline.

#     Parameters:
#     -----------
#     baseline_df : DataFrame (e.g., your 'baseline_selection') 
#                   containing the best intervention for ALL buildings.
#     selected_df : DataFrame (e.g., your 'selected_projects_df') 
#                   containing the projects chosen by the greedy algorithm.
#     scenario_name : str, name for the scenario (e.g., "50% Budget").
#     """
    
#     print("=" * 60)
#     print(f"ANALYSIS: {scenario_name}")
#     print("=" * 60)
    
#     # ----------------------------------------
#     # 1. Calculate Baseline and Scenario Distributions
#     # ----------------------------------------
    
#     # Gas Decile Distribution
#     baseline_decile_dist = baseline_df['avg_gas_percentile'].value_counts().sort_index()
#     baseline_decile_pct = (baseline_decile_dist / len(baseline_df) * 100).rename('Baseline (100%)')

#     scenario_decile_dist = selected_df['avg_gas_percentile'].value_counts().sort_index()
#     scenario_decile_pct = (scenario_decile_dist / len(selected_df) * 100).rename(scenario_name)

#     # Combine for comparison
#     comparison_df = pd.concat([baseline_decile_pct, scenario_decile_pct], axis=1).fillna(0)
    
#     # Scenario Distribution (for selected projects)
#     scenario_dist = selected_df['scenario'].value_counts().sort_index()
    
#     print("\nGas Decile Distribution Comparison (%):")
#     print(comparison_df.round(1))
    
#     # ----------------------------------------
#     # 2. Plotting (Recreating Figure 2 structure)
#     # ----------------------------------------
    
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))

#     # --- A. Gas Decile Comparison Bar Chart (Comparison_df.T plot) ---
#     comparison_df.T.plot(kind='bar', stacked=False, ax=axes[0, 0], 
#                          colormap='tab10', edgecolor='black', linewidth=0.5)
#     axes[0, 0].set_xlabel('Scenario', fontsize=11)
#     axes[0, 0].set_ylabel('Percentage of Buildings (%)', fontsize=11)
#     axes[0, 0].set_title('Gas Decile Distribution: Baseline vs. Selection', fontsize=12, fontweight='bold')
#     axes[0, 0].legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
#     axes[0, 0].tick_params(axis='x', rotation=0)
#     axes[0, 0].grid(axis='y', alpha=0.3)
    
#     # --- B. Heatmap of Decile Distribution ---
#     cmap = plt.cm.RdYlGn # Use a color map for visual impact
#     im = axes[0, 1].imshow(comparison_df.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=25)

#     axes[0, 1].set_xticks(np.arange(len(comparison_df.index)))
#     axes[0, 1].set_yticks(np.arange(len(comparison_df.columns)))
#     axes[0, 1].set_xticklabels(comparison_df.index)
#     axes[0, 1].set_yticklabels(comparison_df.columns)
#     axes[0, 1].tick_params(axis='x', rotation=45)

#     # Add text labels to the heatmap cells
#     for i in range(len(comparison_df.columns)):
#         for j in range(len(comparison_df.index)):
#             value = comparison_df.T.values[i, j]
#             axes[0, 1].text(j, i, f'{value:.1f}',
#                            ha="center", va="center", 
#                            color="white" if value > 12 else "black",
#                            fontsize=8)

#     axes[0, 1].set_title('Gas Decile Distribution Heatmap (%)', fontsize=12, fontweight='bold', pad=10)
#     axes[0, 1].set_xlabel('Gas Decile', fontsize=11)
#     axes[0, 1].set_ylabel('Scenario', fontsize=11)

#     cbar = plt.colorbar(im, ax=axes[0, 1])
#     cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
    
#     # --- C. Difference from Baseline (Bias Analysis) ---
#     diff_from_baseline = comparison_df[scenario_name] - comparison_df['Baseline (100%)']
#     colors = ['red' if x < 0 else 'green' for x in diff_from_baseline]
#     axes[1, 0].bar(range(len(diff_from_baseline)), diff_from_baseline, color=colors, edgecolor='black', alpha=0.7)
#     axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
#     axes[1, 0].set_xlabel('Gas Decile', fontsize=11)
#     axes[1, 0].set_ylabel('Difference from Baseline (%)', fontsize=11)
#     axes[1, 0].set_title('Decile Bias: Difference from Full Coverage Baseline', fontsize=12, fontweight='bold')
#     axes[1, 0].set_xticks(range(len(diff_from_baseline)))
#     axes[1, 0].set_xticklabels(diff_from_baseline.index, rotation=45)
#     axes[1, 0].grid(axis='y', alpha=0.3)
    
#     # --- D. Scenario Selection Distribution ---
#     scenario_dist.plot(kind='bar', ax=axes[1, 1], color='steelblue', edgecolor='black', alpha=0.7)
#     axes[1, 1].set_xlabel('Intervention Scenario', fontsize=11)
#     axes[1, 1].set_ylabel('Number of Buildings Selected', fontsize=11)
#     axes[1, 1].set_title(f'Intervention Scenario Distribution for {scenario_name}', fontsize=12, fontweight='bold')
#     axes[1, 1].tick_params(axis='x', rotation=45)
#     axes[1, 1].grid(axis='y', alpha=0.3)

#     plt.tight_layout()
#     plt.show()



# def multi_intervention_greedy_knapsack(df_interventions, budget, 
#                                         cost_column='cost of interventon_mean', 
#                                         efficiency_column='cost_per_net_ton_co2_kg'):
#     """
#     Selects the most cost-effective *single interventions* until the budget is exhausted,
#     allowing multiple, distinct interventions per building (UPN).
    
#     Parameters:
#     -----------
#     df_interventions : DataFrame with one row per single, standalone intervention 
#                        (e.g., one row for 'UPN 1 - Loft', one for 'UPN 1 - Wall').
#     budget : float, total budget available (in £).
#     cost_column : str, name of the column containing the absolute cost of the intervention.
#     efficiency_column : str, name of the column containing the cost-effectiveness metric 
#                         (lower is better, e.g., cost_per_net_ton_co2_kg).
    
#     Returns:
#     --------
#     selected : DataFrame of selected interventions.
#     remaining : float, remaining budget.
#     """
    
#     # 1. Sort by the cost-effectiveness metric (ascending: lower cost per CO2 saved is better)
#     # The greedy core remains the same.
#     df_sorted = df_interventions.sort_values(efficiency_column, ascending=True).reset_index(drop=True)
    
#     selected_interventions = []
#     remaining_budget = budget
    
#     print(f"Starting multi-intervention greedy selection with budget: £{budget:,.0f}")
    
#     # 2. Iterate through the sorted, most cost-effective interventions
#     for idx, row in df_sorted.iterrows():
#         project_cost = row[cost_column]
        
#         # Check if we can afford the current project
#         if project_cost <= remaining_budget:
#             # Since each row is a unique, standalone intervention, we just select it.
#             # This is the step that now implicitly allows multiple selections per UPN.
#             selected_interventions.append(row)
#             remaining_budget -= project_cost
#         else:
#             # Crucial for efficiency: Stop iteration immediately.
#             break 
            
#     selected_df = pd.DataFrame(selected_interventions)
    
#     # --- Reporting (same as before) ---
#     total_spent = budget - remaining_budget
    
#     # We now need to ensure 'total_ton_co2_saved' is the name of the column 
#     # for the CO2 saved by *that specific intervention*.
#     total_co2 = selected_df['total_ton_co2_saved'].sum() 
    
#     if len(selected_df) > 0:
#         # Note: 'Buildings covered' is now replaced by 'Interventions selected'
#         # To get buildings, you'd use selected_df['UPN'].nunique()
#         print("\n✅ Selection Complete:")
#         print(f"  Interventions selected: {len(selected_df):,}")
#         print(f"  Unique Buildings covered: {selected_df['UPN'].nunique():,}")
#         print(f"  Total spent: £{total_spent:,.0f}")
#         print(f"  Total CO2 saved: {total_co2:,.2f} tons")
#         print(f"  Cost per ton CO2 (Achieved): £{total_spent/total_co2:,.2f}")
    
#     return selected_df, remaining_budget
 