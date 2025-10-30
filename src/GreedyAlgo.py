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

def plot_greedy_distribution_analysis(baseline_df, selected_df, scenario_name="Greedy Selection"):
    """
    Calculates and plots the Gas Decile and Scenario distributions for the 
    selected projects against the full baseline.

    Parameters:
    -----------
    baseline_df : DataFrame (e.g., your 'baseline_selection') 
                  containing the best intervention for ALL buildings.
    selected_df : DataFrame (e.g., your 'selected_projects_df') 
                  containing the projects chosen by the greedy algorithm.
    scenario_name : str, name for the scenario (e.g., "50% Budget").
    """
    
    print("=" * 60)
    print(f"ANALYSIS: {scenario_name}")
    print("=" * 60)
    
    # ----------------------------------------
    # 1. Calculate Baseline and Scenario Distributions
    # ----------------------------------------
    
    # Gas Decile Distribution
    baseline_decile_dist = baseline_df['avg_gas_percentile'].value_counts().sort_index()
    baseline_decile_pct = (baseline_decile_dist / len(baseline_df) * 100).rename('Baseline (100%)')

    scenario_decile_dist = selected_df['avg_gas_percentile'].value_counts().sort_index()
    scenario_decile_pct = (scenario_decile_dist / len(selected_df) * 100).rename(scenario_name)

    # Combine for comparison
    comparison_df = pd.concat([baseline_decile_pct, scenario_decile_pct], axis=1).fillna(0)
    
    # Scenario Distribution (for selected projects)
    scenario_dist = selected_df['scenario'].value_counts().sort_index()
    
    print("\nGas Decile Distribution Comparison (%):")
    print(comparison_df.round(1))
    
    # ----------------------------------------
    # 2. Plotting (Recreating Figure 2 structure)
    # ----------------------------------------
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- A. Gas Decile Comparison Bar Chart (Comparison_df.T plot) ---
    comparison_df.T.plot(kind='bar', stacked=False, ax=axes[0, 0], 
                         colormap='tab10', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Scenario', fontsize=11)
    axes[0, 0].set_ylabel('Percentage of Buildings (%)', fontsize=11)
    axes[0, 0].set_title('Gas Decile Distribution: Baseline vs. Selection', fontsize=12, fontweight='bold')
    axes[0, 0].legend(title='Gas Decile', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # --- B. Heatmap of Decile Distribution ---
    cmap = plt.cm.RdYlGn # Use a color map for visual impact
    im = axes[0, 1].imshow(comparison_df.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=25)

    axes[0, 1].set_xticks(np.arange(len(comparison_df.index)))
    axes[0, 1].set_yticks(np.arange(len(comparison_df.columns)))
    axes[0, 1].set_xticklabels(comparison_df.index)
    axes[0, 1].set_yticklabels(comparison_df.columns)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Add text labels to the heatmap cells
    for i in range(len(comparison_df.columns)):
        for j in range(len(comparison_df.index)):
            value = comparison_df.T.values[i, j]
            axes[0, 1].text(j, i, f'{value:.1f}',
                           ha="center", va="center", 
                           color="white" if value > 12 else "black",
                           fontsize=8)

    axes[0, 1].set_title('Gas Decile Distribution Heatmap (%)', fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Gas Decile', fontsize=11)
    axes[0, 1].set_ylabel('Scenario', fontsize=11)

    cbar = plt.colorbar(im, ax=axes[0, 1])
    cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
    
    # --- C. Difference from Baseline (Bias Analysis) ---
    diff_from_baseline = comparison_df[scenario_name] - comparison_df['Baseline (100%)']
    colors = ['red' if x < 0 else 'green' for x in diff_from_baseline]
    axes[1, 0].bar(range(len(diff_from_baseline)), diff_from_baseline, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Gas Decile', fontsize=11)
    axes[1, 0].set_ylabel('Difference from Baseline (%)', fontsize=11)
    axes[1, 0].set_title('Decile Bias: Difference from Full Coverage Baseline', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(range(len(diff_from_baseline)))
    axes[1, 0].set_xticklabels(diff_from_baseline.index, rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # --- D. Scenario Selection Distribution ---
    scenario_dist.plot(kind='bar', ax=axes[1, 1], color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Intervention Scenario', fontsize=11)
    axes[1, 1].set_ylabel('Number of Buildings Selected', fontsize=11)
    axes[1, 1].set_title(f'Intervention Scenario Distribution for {scenario_name}', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()



def multi_intervention_greedy_knapsack(df_interventions, budget, 
                                        cost_column='cost of interventon_mean', 
                                        efficiency_column='cost_per_net_ton_co2_kg'):
    """
    Selects the most cost-effective *single interventions* until the budget is exhausted,
    allowing multiple, distinct interventions per building (UPN).
    
    Parameters:
    -----------
    df_interventions : DataFrame with one row per single, standalone intervention 
                       (e.g., one row for 'UPN 1 - Loft', one for 'UPN 1 - Wall').
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
    # The greedy core remains the same.
    df_sorted = df_interventions.sort_values(efficiency_column, ascending=True).reset_index(drop=True)
    
    selected_interventions = []
    remaining_budget = budget
    
    print(f"Starting multi-intervention greedy selection with budget: £{budget:,.0f}")
    
    # 2. Iterate through the sorted, most cost-effective interventions
    for idx, row in df_sorted.iterrows():
        project_cost = row[cost_column]
        
        # Check if we can afford the current project
        if project_cost <= remaining_budget:
            # Since each row is a unique, standalone intervention, we just select it.
            # This is the step that now implicitly allows multiple selections per UPN.
            selected_interventions.append(row)
            remaining_budget -= project_cost
        else:
            # Crucial for efficiency: Stop iteration immediately.
            break 
            
    selected_df = pd.DataFrame(selected_interventions)
    
    # --- Reporting (same as before) ---
    total_spent = budget - remaining_budget
    
    # We now need to ensure 'total_ton_co2_saved' is the name of the column 
    # for the CO2 saved by *that specific intervention*.
    total_co2 = selected_df['total_ton_co2_saved'].sum() 
    
    if len(selected_df) > 0:
        # Note: 'Buildings covered' is now replaced by 'Interventions selected'
        # To get buildings, you'd use selected_df['UPN'].nunique()
        print("\n✅ Selection Complete:")
        print(f"  Interventions selected: {len(selected_df):,}")
        print(f"  Unique Buildings covered: {selected_df['UPN'].nunique():,}")
        print(f"  Total spent: £{total_spent:,.0f}")
        print(f"  Total CO2 saved: {total_co2:,.2f} tons")
        print(f"  Cost per ton CO2 (Achieved): £{total_spent/total_co2:,.2f}")
    
    return selected_df, remaining_budget
 