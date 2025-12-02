import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
import pandas as pd

import pandas as pd
import logging
from typing import Tuple

def true_greedy_knapsack(df_knapsack: pd.DataFrame, 
                         budget: float, 
                         cost_column: str = 'cost_of_intervention_mean', 
                         efficiency_column: str = 'cost_per_net_ton_co2_kg',
                         logger: logging.Logger = None, 
                         carbon_col: str = 'total_ton_co2_saved') -> Tuple[pd.DataFrame, float]:
    """
    Selects the most cost-effective buildings to receive interventions until 
    the budget is exhausted.
    
    This implementation iterates through all potential projects to fill the 
    knapsack, skipping items that are too large but allowing smaller, less
    efficient items to be selected later.
    
    Parameters:
    -----------
    df_knapsack : DataFrame with one row (best intervention) per building (upn).
    budget : float, total budget available (in £).
    cost_column : str, name of the column containing the absolute cost.
    efficiency_column : str, name of the column for cost-effectiveness 
                        (lower is better).
    logger : logging.Logger, optional logger object.
    
    Returns:
    --------
    selected_df : DataFrame of selected interventions.
    remaining_budget : float, remaining budget.
    """
    
    # 1. Sort by the cost-effectiveness metric (ascending)
    df_sorted = df_knapsack.sort_values(efficiency_column, ascending=True)
    
    selected_indices = []
    remaining_budget = budget
    
    if logger:
        logger.info(f"Starting true greedy selection with budget: £{budget:,.0f}")
        logger.info(f"Sorting {len(df_sorted)} projects by '{efficiency_column}'")

    # 2. Iterate using itertuples() for performance.
    # This is much faster than iterrows().
    for row in df_sorted.itertuples(index=True):
        # Use getattr() to dynamically access the column from the tuple
        project_cost = getattr(row, cost_column)
        
        # Check if we can afford this project
        if project_cost <= remaining_budget:
            selected_indices.append(row.Index) # Add the row's original index
            remaining_budget -= project_cost
        
        # *** LOGIC FIX: ***
        # We intentionally DO NOT have an 'else: break' here.
        # We must continue iterating to find smaller projects that may
        # be less efficient but still fit in the remaining budget.

    # 3. Create the final DataFrame from the selected indices
    # Use .loc to select all at once (fast) and .copy() to avoid warnings
    selected_df = df_sorted.loc[selected_indices].copy()
    
    total_spent = budget - remaining_budget
    
    # 4. Log the results
    if logger:
        if not selected_df.empty:
            # This 'total_ton_co2_saved' column name is hardcoded based on
            # the context of the main script (RANK_COL_CO2_SAVED)
            total_co2 = selected_df[carbon_col].sum()
            
            logger.info("\n✅ Selection Complete:")
            logger.info(f"  Buildings covered: {len(selected_df):,}")
            logger.info(f"  Total spent: £{total_spent:,.0f} (Budget: £{budget:,.0f})")
            logger.info(f"  Total CO2 saved: {total_co2:,.2f} tons")
            if total_co2 > 0:
                logger.info(f"  Cost per ton CO2 (Achieved): £{total_spent/total_co2:,.2f}")
        else:
            logger.warning("\n⚠️ No interventions selected (budget may be insufficient for any single project)")
    
    return selected_df, remaining_budget


# def true_greedy_knapsack(df_knapsack, budget, cost_column='cost of interventon_mean', efficiency_column='cost_per_net_ton_co2_kg'):
#     """
#     Selects the most cost-effective buildings to receive interventions until the budget is exhausted.
    
#     Parameters:
#     -----------
#     df_knapsack : DataFrame with one row (best intervention) per building (upn).
#     budget : float, total budget available (in £).
#     cost_column : str, name of the column containing the absolute cost of the intervention.
#     efficiency_column : str, name of the column containing the cost-effectiveness metric 
#                         (lower is better, e.g., cost_per_net_ton_co2_kg).
    
    
#     Returns:
#     --------
#     selected : DataFrame of selected interventions.
#     remaining : float, remaining budget.
#     """
    
#     # 1. Sort by the cost-effectiveness metric (acending: lower cost per CO2 saved is better)
#     # This is the core of the greedy strategy.
#     df_sorted = df_knapsack.sort_values(efficiency_column, ascending=True).reset_index(drop=True)
    
#     selected_interventions = []
#     remaining_budget = budget
    
#     print(f"Starting true greedy selection with budget: £{budget:,.0f}")
    
#     # 2. Iterate through the sorted, most cost-effective interventions
#     for idx, row in df_sorted.iterrows():
#         project_cost = row[cost_column]
        
#         # Check if we can afford the current most cost-effective project
#         if project_cost <= remaining_budget:
#             selected_interventions.append(row)
#             remaining_budget -= project_cost
#         else:
#             # Crucial for efficiency: Since the list is sorted, 
#             # if we can't afford the current project, we can't afford any subsequent, 
#             # less cost-effective projects either. Stop iteration immediately.
#             break 
            
#     selected_df = pd.DataFrame(selected_interventions)
    
#     total_spent = budget - remaining_budget
 
    
#     if len(selected_df) > 0:
#         total_co2 = selected_df['total_ton_co2_saved'].sum()
#         print("\n✅ Selection Complete:")
#         print(f"  Buildings covered: {len(selected_df):,}")
#         print(f"  Total spent: £{total_spent:,.0f}")
#         print(f"  Total CO2 saved: {total_co2:,.2f} tons")
#         print(f"  Cost per ton CO2 (Achieved): £{total_spent/total_co2:,.2f}")
#     else:
#         print("\n⚠️ No interventions selected (budget insufficient)")
    
#     return selected_df, remaining_budget


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os  # <-- Added for directory and path handling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_greedy_distribution_analysis(baseline_df, selected_df, 
                                      scenario_name="intervention", 
                                      output_dir=None):
    """
    Calculates and plots distributions for a single result set 
    (pre-processed or single run).
    
    Generates 5 separate plots (A, B, C, D, E). 
    Plot F (Std Dev) has been removed as this is a single run analysis.

    Parameters:
    -----------
    baseline_df : DataFrame
        Contains the best intervention for ALL buildings (the pool).
    selected_df : DataFrame
        Contains the projects chosen by the greedy algorithm.
    scenario_name : str
        Name for the scenario, used in titles and filenames.
    output_dir : str or Path, optional
        If provided, saves plots to this directory instead of showing them.
    """
    
    print("=" * 60)
    print(f"ANALYSIS: {scenario_name} (Single Result Set)")
    print("=" * 60)

    # Validate required columns
    required_cols = ['avg_gas_percentile', 'intervention']
    for col in required_cols:
        if col not in selected_df.columns:
            print(f"ERROR: '{col}' column not found in selected_df.")
            return

    # ----------------------------------------
    # 1. Calculate Data Distributions
    # ----------------------------------------
    
    # Get sorted unique values for consistent plotting order
    all_deciles_sorted = np.sort(selected_df['avg_gas_percentile'].unique())
    all_scenarios_sorted = np.sort(selected_df['intervention'].unique())

    # --- Baseline & Selected Decile Comparison (Plots A, B, C) ---
    # Calculate % distribution of buildings across gas deciles
    baseline_decile_dist = baseline_df['avg_gas_percentile'].value_counts().sort_index()
    baseline_decile_pct = (baseline_decile_dist / len(baseline_df) * 100).rename('Baseline (100%)')

    selected_decile_dist = selected_df['avg_gas_percentile'].value_counts().sort_index()
    selected_decile_pct = (selected_decile_dist / len(selected_df) * 100).rename(scenario_name)

    comparison_df = pd.concat([baseline_decile_pct, selected_decile_pct], axis=1).fillna(0)
    diff_from_baseline = comparison_df[scenario_name] - comparison_df['Baseline (100%)']
    
    print("\nGas Decile Distribution Comparison (%):")
    print(comparison_df.round(1))

    # --- Scenario Counts (Plot D) ---
    # Simply count how many times each scenario appears in the selection
    scenario_counts = selected_df['intervention'].value_counts().reindex(all_scenarios_sorted, fill_value=0)
    
    print("\nIntervention Counts:")
    print(scenario_counts)

    # --- Intervention Mix per Decile (Plot E) ---
    # Crosstab to show which interventions are selected within each decile
    counts_per_decile = pd.crosstab(
        selected_df['avg_gas_percentile'], 
        selected_df['intervention']
    )
    
    # Reindex to ensure strictly consistent axes even if some scenarios/deciles are missing in this specific run
    counts_per_decile = counts_per_decile.reindex(index=all_deciles_sorted, fill_value=0)
    counts_per_decile = counts_per_decile.reindex(columns=all_scenarios_sorted, fill_value=0)

    print("\nIntervention Mix per Decile (Counts):")
    print(counts_per_decile)

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
    # 3. Generate Plots
    # ----------------------------------------

    # --- A. Gas Decile Comparison Bar Chart ---
    try:
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        comparison_df.T.plot(kind='bar', stacked=False, ax=ax_a, 
                             colormap='tab10', edgecolor='black', linewidth=0.5)
        ax_a.set_xlabel('Dataset', fontsize=11)
        ax_a.set_ylabel('Percentage of Buildings (%)', fontsize=11)
        ax_a.set_title(f'A. Gas Decile Distribution: Baseline vs. {scenario_name}', fontsize=12, fontweight='bold')
        ax_a.legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_a.tick_params(axis='x', rotation=0)
        ax_a.grid(axis='y', alpha=0.3)
        save_or_show(fig_a, "A_Decile_Distribution")
    except Exception as e:
        print(f"Failed to create Plot A: {e}")

    # --- B. Heatmap of Decile Distribution ---
    try:
        fig_b, ax_b = plt.subplots(figsize=(8, 5))
        comparison_df_numeric = comparison_df.astype('float64')

        cmap = plt.cm.RdYlGn
        # vmin/vmax logic handles coloring
        im = ax_b.imshow(comparison_df_numeric.T.values, cmap=cmap, aspect='auto', 
                         vmin=0, vmax=comparison_df_numeric.T.values.max() * 1.1)

        ax_b.set_xticks(np.arange(len(comparison_df_numeric.index)))
        ax_b.set_yticks(np.arange(len(comparison_df_numeric.columns)))
        ax_b.set_xticklabels(comparison_df_numeric.index)
        ax_b.set_yticklabels(comparison_df_numeric.columns)
        ax_b.tick_params(axis='x', rotation=45)

        # Annotate values
        for i in range(len(comparison_df_numeric.columns)):
            for j in range(len(comparison_df_numeric.index)):
                value = comparison_df_numeric.T.values[i, j]
                # Dynamic text color based on value for readability
                text_color = "black" if value < (comparison_df_numeric.values.max()/1.5) else "white"
                ax_b.text(j, i, f'{value:.1f}', ha="center", va="center", 
                          color=text_color, fontsize=9, fontweight='bold')

        ax_b.set_title('B. Gas Decile Distribution Heatmap (%)', fontsize=12, fontweight='bold', pad=10)
        ax_b.set_xlabel('Gas Decile', fontsize=11)
        
        cbar = plt.colorbar(im, ax=ax_b)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
        save_or_show(fig_b, "B_Decile_Heatmap")
    except Exception as e:
        print(f"Failed to create Plot B: {e}")

    # --- C. Difference from Baseline (Bias Analysis) ---
    try:
        fig_c, ax_c = plt.subplots(figsize=(8, 6))
        # Color: Green if over-indexed compared to baseline, Red if under-indexed (or vice versa depending on preference)
        # Here: Green = Positive Difference, Red = Negative Difference
        colors = ['green' if x >= 0 else 'red' for x in diff_from_baseline]
        
        ax_c.bar(range(len(diff_from_baseline)), diff_from_baseline, color=colors, edgecolor='black', alpha=0.7)
        ax_c.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax_c.set_xlabel('Gas Decile', fontsize=11)
        ax_c.set_ylabel('Difference from Baseline (%)', fontsize=11)
        ax_c.set_title(f'C. Decile Bias: {scenario_name} vs. Baseline', fontsize=12, fontweight='bold')
        ax_c.set_xticks(range(len(diff_from_baseline)))
        ax_c.set_xticklabels(diff_from_baseline.index, rotation=45)
        ax_c.grid(axis='y', alpha=0.3)
        save_or_show(fig_c, "C_Decile_Bias")
    except Exception as e:
        print(f"Failed to create Plot C: {e}")

    # --- D. Scenario Selection Distribution (Absolute Counts) ---
    try:
        fig_d, ax_d = plt.subplots(figsize=(8, 6))
        scenario_counts.plot(
            kind='bar', ax=ax_d, 
            color='steelblue', edgecolor='black', alpha=0.8
        )
        ax_d.set_xlabel('Intervention Scenario', fontsize=11)
        ax_d.set_ylabel('Total Buildings Selected', fontsize=11)
        ax_d.set_title(f'D. Intervention Distribution for {scenario_name}', fontsize=12, fontweight='bold')
        ax_d.tick_params(axis='x', rotation=45)
        ax_d.grid(axis='y', alpha=0.3)
        
        # Add count labels on top of bars
        for p in ax_d.patches:
            ax_d.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.01), fontsize=9)
            
        save_or_show(fig_d, "D_Intervention_Distribution")
    except Exception as e:
        print(f"Failed to create Plot D: {e}")

    # --- E. Intervention Mix per Decile (Stacked Counts) ---
    try:
        fig_e, ax_e = plt.subplots(figsize=(9, 6))
        counts_per_decile.plot(
            kind='bar', stacked=True, ax=ax_e, 
            colormap='tab20', edgecolor='black', linewidth=0.5
        )
        ax_e.set_xlabel('Gas Decile', fontsize=11)
        ax_e.set_ylabel('Number of Buildings Selected', fontsize=11)
        ax_e.set_title(f'E. Intervention Mix per Decile for {scenario_name}', fontsize=12, fontweight='bold')
        ax_e.legend(title='Intervention', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_e.tick_params(axis='x', rotation=45)
        ax_e.grid(axis='y', alpha=0.3)
        save_or_show(fig_e, "E_Intervention_Mix")
    except Exception as e:
        print(f"Failed to create Plot E: {e}")

    print("\nAnalysis plotting complete.")

