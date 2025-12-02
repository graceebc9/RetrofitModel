import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

label_cleaner_map = {
    'high_deprived':  'High Deprivation' , 
    'med_deprived': 'Medium Deprivation', 
    'low_deprived': 'Low Deprivation', 
}

name_mapping = { 
    0: "Struggling Lone Parents",
    1: "Secure Established Families",
    2: "Solvent Working Households",
    3: "Modest Solo Dwellers",
    4: "At-Risk Singles",
    5: "Senior Citizens",
    6: "Isolated and Deprived",
    7: "Younger Strugglers",
    8: "The Squeezed Middle"
}

total_co2_saved_col = 'total_co2_saved_robust_sum'
# total_co2_saved_col_std = 'total_co2_saved_robust_sum_std'
capex_per_net_ton_mean_col = 'capex_per_net_ton_mean'
# capex_per_net_ton_std_col = 'capex_per_net_ton_mean_std'

scenario_label_map = {
    'budget_1000000_equity_0': 'equity_0', 
    'budget_1000000_equity_0.2': 'equity_0.2', 
    'budget_1000000_equity_0.4': 'equity_0.4', 
    'budget_1000000_equity_0.6': 'equity_0.6',
    'budget_1000000_equity_0.8': 'equity_0.8', 
    'budget_1000000_equity_1': 'equity_1',
    
    'budget_10000000_equity_0': 'equity_0', 
    'budget_10000000_equity_0.2': 'equity_0.2', 
    'budget_10000000_equity_0.4': 'equity_0.4', 
    'budget_10000000_equity_0.6': 'equity_0.6',
    'budget_10000000_equity_0.8': 'equity_0.8', 
    'budget_10000000_equity_1': 'equity_1',
    
    'budget_100000000_equity_0': 'equity_0', 
    'budget_100000000_equity_0.2': 'equity_0.2', 
    'budget_100000000_equity_0.4': 'equity_0.4', 
    'budget_100000000_equity_0.6': 'equity_0.6',
    'budget_100000000_equity_0.8': 'equity_0.8', 
    'budget_100000000_equity_1': 'equity_1',
    
    'budget_500000000_equity_0': 'equity_0', 
    'budget_500000000_equity_0.2': 'equity_0.2', 
    'budget_500000000_equity_0.4': 'equity_0.4', 
    'budget_500000000_equity_0.6': 'equity_0.6',
    'budget_500000000_equity_0.8': 'equity_0.8', 
    'budget_500000000_equity_1': 'equity_1',
    
    'budget_1000000000_equity_0': 'equity_0', 
    'budget_1000000000_equity_0.2': 'equity_0.2', 
    'budget_1000000000_equity_0.4': 'equity_0.4', 
    'budget_1000000000_equity_0.6': 'equity_0.6',
    'budget_1000000000_equity_0.8': 'equity_0.8', 
    'budget_1000000000_equity_1': 'equity_1',
    
    'budget_5000000000_equity_0': 'equity_0', 
    'budget_5000000000_equity_0.2': 'equity_0.2', 
    'budget_5000000000_equity_0.4': 'equity_0.4', 
    'budget_5000000000_equity_0.6': 'equity_0.6',
    'budget_5000000000_equity_0.8': 'equity_0.8', 
    'budget_5000000000_equity_1': 'equity_1',
    
    'budget_10000000000_equity_0': 'equity_0', 
    'budget_10000000000_equity_0.2': 'equity_0.2', 
    'budget_10000000000_equity_0.4': 'equity_0.4', 
    'budget_10000000000_equity_0.6': 'equity_0.6',
    'budget_10000000000_equity_0.8': 'equity_0.8', 
    'budget_10000000000_equity_1': 'equity_1',
}


 
# plot main 
def plot_greedy_compairosn_main(df_raw, output_dir, y_axis_zero=False):
    """
    Main plotting function.
    
    Args:
        df_raw (pd.DataFrame): The raw input dataframe.
        output_dir (str): The directory to save plots.
        y_axis_zero (bool, optional): If True, sets the y-axis minimum to 0
                                      for all plots. Defaults to False.
    """
    df_processed = preprocess_dataframe(df_raw)

    scenarios = df_processed['scenario'].unique()
    equity_weights = sorted(df_processed['equity_weight'].unique())
    scenario_colors = create_scenario_colors(scenarios, df_processed)
    
    # Create a general budget label (for titles)
    budget_label = "All Budgets"
    unique_budgets = df_processed['budget'].unique()
    if len(unique_budgets) == 1:
        budget_label = f"Budget £{unique_budgets[0]/1e6:.0f}M"

    # Set subsets for plotting (in this simple case, they are the same)
    # You might have different DFs in your real workflow
    results_subset = df_processed.copy()
    equity_subset = df_processed.copy()

    # --- Call all new plotting functions ---
    print("Generating plots...")
    plot_all_metrics(df_processed, 
                     output_dir=os.path.join(output_dir, 'single_plots'),
                     y_axis_zero=y_axis_zero) 
    
    plot_carbon_savings_vs_equity(results_subset, equity_weights, budget_label, 
                                  os.path.join(output_dir, "1_carbon_vs_equity.png"),
                                  y_axis_zero=y_axis_zero)
    
    plot_cost_effectiveness_vs_equity(results_subset, equity_weights, budget_label, 
                                      os.path.join(output_dir, "2_cost_effectiveness_vs_equity.png"),
                                      y_axis_zero=y_axis_zero)
    
    plot_vulnerable_coverage_vs_equity(equity_subset, equity_weights, budget_label, 
                                       os.path.join(output_dir, "3_vulnerable_coverage_vs_equity.png"),
                                       y_axis_zero=y_axis_zero)
    
    plot_equity_concentration_vs_weight(equity_subset, equity_weights, budget_label, 
                                        os.path.join(output_dir, "4_equity_concentration_vs_weight.png"),
                                        y_axis_zero=y_axis_zero)
    
 
    
    plot_pareto_front(results_subset, equity_subset, scenarios, scenario_colors, budget_label, 
                      os.path.join(output_dir, "6_pareto_front.png"),
                      y_axis_zero=y_axis_zero)
    
    plot_vulnerable_groups_coverage(equity_subset,
                                    os.path.join(output_dir, "7_vulnerable_groups_coverage.png"),
                                    y_axis_zero=y_axis_zero)
    
    plot_tradeoff_efficiency(results_subset, equity_subset, scenarios, scenario_colors, budget_label, 
                             os.path.join(output_dir, "8_tradeoff_efficiency.png"),
                             y_axis_zero=y_axis_zero)
    
    plot_radar_chart(results_subset, equity_subset, 
                     os.path.join(output_dir, "9_radar_chart.png"),scenario_colors
                    ) 

    plot_pareto_retrofit_carbon_by_budget(results_subset, equity_subset, scenarios, scenario_colors, budget_label,
                                 os.path.join(output_dir, "10_pareto_bcounts.png") ,
                                y_axis_zero=y_axis_zero)
    
    plot_pareto_retrofit_carbon_by_costeff(results_subset, equity_subset, scenarios, scenario_colors, budget_label,
                                 os.path.join(output_dir, "11_pareto_cost_eff.png") ,
                                y_axis_zero=y_axis_zero)



    print(f"Done! All 9 plots saved to '{output_dir}'.")
    return scenario_colors
# ==============================================================================
# 2. DATA PREPROCESSING
# ==============================================================================

def flatten_columns(df):
    """
    Flattens the MultiIndex columns into a single, usable string per column.
    e.g., ('cost_of_intervention_mean_mean', 'mean') -> 'cost_of_intervention_mean_mean_mean'
    e.g., ('scenario', '') -> 'scenario'
    """
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Filter out empty strings from tuple and join
            flat_name = '_'.join(str(c) for c in col if c)
            new_cols.append(flat_name)
        else:
            new_cols.append(str(col))
    
    df.columns = new_cols
    
    # Handle potential duplicate columns (e.g., 'scenario' from tuple and string)
    # We'll just keep the first instance of each
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df

def preprocess_dataframe(df):
    """
    Applies all preprocessing steps: flattening columns,
    extracting budget and equity weight, and sorting.
    """
    # 1. Flatten columns
    df_flat = flatten_columns(df)
    
    # 2. Extract numeric equity weight from scenario string
    try:
        df_flat['equity_weight'] = df_flat['scenario'].str.split('_').str[-1].astype(float)
    except Exception as e:
        print(f"Error: Could not extract equity weight: {e}")
        raise e
        
    # 3. Extract numeric budget from scenario string
    try:
        df_flat['budget'] = df_flat['scenario'].str.split('_').str[1].astype(float)
    except Exception as e:
        print(f"Error: Could not extract budget: {e}")
        raise e
        
    # 4. Sort by budget, then equity weight
    df_flat = df_flat.sort_values(['budget', 'equity_weight']).reset_index(drop=True)
    
    return df_flat

# ==============================================================================
# 3. COLOR HELPER FUNCTIONS
# ==============================================================================

def get_color_palette(n_colors):
    """Generate a color palette with n colors"""
    if n_colors <= 3:
        return ['#e74c3c', '#f39c12', '#27ae60'][:n_colors]
    else:
        # Use seaborn color palette for more colors
        return sns.color_palette('Set2', n_colors).as_hex()

def create_scenario_colors(scenarios, results_agg):
    """Create color mapping for scenarios based on equity weights"""
    # Get unique equity weights and sort them
    equity_weights = sorted(results_agg['equity_weight'].unique())
    colors = get_color_palette(len(equity_weights))
    weight_to_color = dict(zip(equity_weights, colors))
    
    scenario_colors = {}
    for scenario in scenarios:
        # Extract equity weight from scenario
        weight = results_agg[results_agg['scenario'] == scenario]['equity_weight'].iloc[0]
        scenario_colors[scenario] = weight_to_color[weight]
    
    return scenario_colors

# ==============================================================================
# 4. INDIVIDUAL PLOT FUNCTIONS (UPDATED FOR FLATTENED COLUMNS)
# ==============================================================================

def plot_all_metrics(df, output_dir='scenario_plots', y_axis_zero=False):
    """
    Loops through all mean metrics, plots them against 'equity_weight',
    and saves each plot to the output directory.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get the X-axis for all plots
    x = df['equity_weight']
    
    # 3. Find all columns that represent a 'mean'
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    
    print(f"Found {len(mean_cols)} metrics to plot. Saving to '{output_dir}'...")
    
    # 4. Loop, plot, and save
    for mean_col in mean_cols:
        # 5. Get the Y-axis data (the mean value)
        y_mean = df[mean_col]
        
        # 6. Find the corresponding standard deviation column
        # e.g., 'vulnerable_pct_mean' -> 'vulnerable_pct_std'
        metric_base = mean_col[:-5] # Remove '_mean'
        std_col = f"{metric_base}_std"
        
        # 7. Create a new figure for this plot
        plt.figure(figsize=(12, 7))
        
        # 8. Plot the mean line
        plt.plot(x, y_mean, marker='o', linestyle='-', label='Mean Value')
        
        # 9. If we found a std column, plot the error band
        if std_col in df.columns:
            y_std = df[std_col]
            plt.fill_between(
                x, 
                y_mean - y_std, 
                y_mean + y_std, 
                alpha=0.2, 
                label='Mean +/- 1 Std. Dev.',
                color='blue'
            )
        
        # 10. Make the plot look good
        plt.title(f"Impact of Equity Weight on\n{mean_col}", fontsize=16)
        plt.xlabel("Equity Weight", fontsize=12)
        plt.ylabel(mean_col, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x) # Ensure all equity weights are shown as ticks
        
        # *** ADDED: Set y-axis to 0 if toggled ***
        if y_axis_zero:
            plt.ylim(bottom=0)
            
        plt.tight_layout()
        
        # 11. Save the plot
        safe_filename = mean_col.replace(' ', '_').replace('/', '_') + '.png'
        save_path = os.path.join(output_dir, safe_filename)
        plt.savefig(save_path)
        
        # 12. Close the figure to free memory
        plt.close()

    print(f"Done! All plots saved to '{output_dir}'.")


def plot_carbon_savings_vs_equity(results_subset, equity_weights, budget_label, filename, y_axis_zero=False):
    """Plot 1: Carbon Savings vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in results_subset['budget'].unique():
        subset = results_subset[results_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        # *** UPDATED COLUMN NAMES ***
        means = subset[total_co2_saved_col].values / 1e3
        stds = subset[total_co2_saved_col].values / 1e3
        
        label = f'£{budget_val/1e6:.0f}M' if len(results_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    ax.set_title(f'Carbon Savings vs Equity Weight\n{budget_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(results_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    # *** ADDED: Set y-axis to 0 if toggled ***
    if y_axis_zero:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_cost_effectiveness_vs_equity(results_subset, equity_weights, budget_label, filename, y_axis_zero=False):
    """Plot 2: Cost Effectiveness vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in results_subset['budget'].unique():
        subset = results_subset[results_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        # *** UPDATED COLUMN NAMES ***
        means = subset[capex_per_net_ton_mean_col].values
        # stds = subset[capex_per_net_ton_std_col].values
        
        label = f'£{budget_val/1e6:.0f}M' if len(results_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost per Ton CO2 (£/kg)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cost Effectiveness vs Equity Weight\n{budget_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(results_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    # *** ADDED: Set y-axis to 0 if toggled ***
    if y_axis_zero:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_vulnerable_coverage_vs_equity(equity_subset, equity_weights, budget_label, filename, y_axis_zero=False):
    """Plot 3: Vulnerable Coverage vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in equity_subset['budget'].unique():
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        # *** These columns were simple strings, so NO change needed ***
        means = subset['high_deprived_pct'].values * 100
        # stds = subset['high_deprived_pct_std'].values * 100
        
        label = f'£{budget_val/1e6:.0f}M' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means,fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Vulnerable Population Coverage vs Equity Weight\n{budget_label}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(equity_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
        
    # *** ADDED: Set y-axis to 0 if toggled ***
    if y_axis_zero:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_equity_concentration_vs_weight(equity_subset, equity_weights, budget_label, filename, y_axis_zero=False):
    """Plot 4: Equity Concentration vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in equity_subset['budget'].unique():
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        # *** These columns were simple strings, so NO change needed ***
        means = subset['equity_concentration'].values
        # stds = subset['equity_concentration_std'].values
        
        label = f'£{budget_val/1e6:.0f}M' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means,  fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Equity Concentration Index', fontsize=14, fontweight='bold')
    ax.set_title(f'Equity Concentration vs Equity Weight\n(lower = more equitable)\n{budget_label}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(equity_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
        
    # *** ADDED: Set y-axis to 0 if toggled ***
    if y_axis_zero:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


    print(f"Saved: {filename}")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_carbon_by_persona(selected_projects_df, scenario_colors, filename, 
                           scenario_label_map=None, y_axis_zero=True):
    """
    Plot distribution of total carbon saved by persona (socioeconomic group).
    
    Parameters:
    - selected_projects_df: DataFrame with columns 'persona_name', 
      'total_co2_saved_robust', 'scenario', 'budget'
    - scenario_colors: Dict mapping scenario names to colors
    - filename: Base output filename (e.g., 'output.png')
    - scenario_label_map: (Optional) Dict to rename scenarios for the legend
    - y_axis_zero: Whether to start y-axis at 0
    """
    if scenario_label_map is None:
        scenario_label_map = {}

    budgets = selected_projects_df['budget'].unique()
    
    for budget in budgets:
        # 1. Filter data for current budget
        equity_subset = selected_projects_df[selected_projects_df['budget'] == budget].copy()
        
        # 2. Aggregate data
        # Group by scenario and persona, summing the robust carbon savings
        carbon_grouped = equity_subset.groupby(
            ['scenario', 'persona_name']
        )['total_co2_saved_robust'].sum().reset_index()
        
        # 3. Pivot for plotting
        # This creates a matrix where index=Personas, columns=Scenarios
        # fillna(0) handles cases where a persona exists in one scenario but not another
        plot_data = carbon_grouped.pivot(
            index='persona_name', 
            columns='scenario', 
            values='total_co2_saved_robust'
        ).fillna(0)
        
        # Sort personas alphabetically (or by index)
        plot_data = plot_data.sort_index()
        
        personas = plot_data.index.tolist()
        scenarios = plot_data.columns.tolist()
        
        # 4. Create Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(personas))
        n_scenarios = len(scenarios)
        width = 0.8 / n_scenarios
        
        for i, scenario in enumerate(scenarios):
            # Calculate bar positions
            offset = (i - n_scenarios / 2 + 0.5) * width
            
            # Get clean label
            label_name = scenario_label_map.get(scenario, scenario)
            
            # Get color
            color = scenario_colors.get(scenario, f'C{i}')
            
            ax.bar(x + offset, 
                   plot_data[scenario], 
                   width, 
                   label=label_name,
                   color=color,
                   alpha=0.7)
        
        # 5. Styling
        ax.set_xlabel('Socio-economic Persona', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Carbon Saved (tons CO₂)', fontsize=14, fontweight='bold')
        ax.set_title(f'Distribution of Total Carbon Saved by Persona\n(Budget: £{budget/1e6:.1f}M)', 
                     fontsize=16, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(personas, fontsize=11, rotation=45, ha='right')
        
        # Smart legend positioning
        ax.legend(fontsize=11, ncol=min(3, max(1, (n_scenarios + 2) // 3)))
        ax.grid(True, alpha=0.3, axis='y')
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        # 6. Save File
        base, ext = os.path.splitext(filename)
        # Handle case where filename might not have an extension
        if not ext: ext = ".png"
        
        rl_filename = f'{base}_budget{budget/1_000_000:.1f}M{ext}'
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to: {rl_filename}")

# def plot_carbon_by_persona(selected_projects_df, scenario_colors, filename, y_axis_zero=True):
#     """
#     Plot distribution of total carbon saved by persona (socioeconomic group).
#     Shows mean across epistemic runs with error bars for SD.
    
#     Parameters:
#     - selected_projects_df: Row-level dataframe with columns including 
#       'meta_socio_persona', 'total_ton_co2_saved_mean', 'scenario', 'epistemic_run'
#     - scenario_colors: Dict mapping scenario names to colors
#     - filename: Output filename
#     - y_axis_zero: Whether to start y-axis at 0
#     """
 
#     def plot_carbon_by_persona_single(selected_projects_df,budget_to_plot,  scenario_colors, filename, y_axis_zero=True):
        
#         rl_filename =f'{filename.split(".png")[0]}_budget{budget_to_plot/1_000_000}M.png'
#         equity_subset = selected_projects_df[selected_projects_df['budget'] == budget_to_plot]
#         print('equity_subset cols: ' , equity_subset.columns.tolist() ) 
#         # Step 1: Sum carbon saved per persona per epistemic run
        
#         # equity_subset['persona_name'] =  equity_subset['cluster'].map(name_mapping)
#         carbon_by_run = equity_subset.groupby(
#             ['scenario', 'persona_name']
#         )['total_co2_saved_robust'].sum().reset_index()
#         carbon_by_run.columns = ['scenario', 'persona', 'mean_carbon']
        
#         # # Step 2: Calculate mean and SD across epistemic runs
#         # carbon_stats = carbon_by_run.groupby(['scenario', 'persona']).agg(
#         #     mean_carbon=('total_co2_saved_robust', 'mean'),
#         #     # sd_carbon=('total_carbon_saved', 'std'),
#         #     n_runs=('total_co2_saved_robust', 'count')
#         # ).reset_index()
        
#         # Step 3: Create plot
#         scenarios = carbon_by_run['scenario'].unique()
#         personas = sorted(carbon_by_run['persona'].unique())
        
 
        
        
#         fig, ax = plt.subplots(figsize=(14, 7))
        
#         x = np.arange(len(personas))
#         n_scenarios = len(scenarios)
#         width = 0.8 / n_scenarios
        
#         for i, scenario in enumerate(scenarios):
#             scenario_data = carbon_by_run[carbon_by_run['scenario'] == scenario]
            
#             # Ensure we have data for all personas (fill with 0 if missing)
#             means = []
            
#             for persona in personas:
#                 print(persona) 
#                 persona_row = scenario_data[scenario_data['persona'] == persona]
#                 if len(persona_row) > 0:
#                     means.append(persona_row['mean_carbon'].iloc[0])
                    
#                 else:
#                     means.append(0)
                    
            
#             offset = (i - n_scenarios/2 + 0.5) * width
            
#             # Extract scenario label (you might want to customize this)
#             label = scenario
#             clean_scenario_name = scenario_label_map.get(scenario, scenario)
#             ax.bar(x + offset, means, width, 
                
#                 label=clean_scenario_name,
#                 color=scenario_colors.get(scenario, f'C{i}'),
#                 alpha=0.7,
#                 capsize=3)
        
#         ax.set_xlabel('Socio-economic Persona', fontsize=14, fontweight='bold')
#         ax.set_ylabel('Total Carbon Saved (tons CO₂)', fontsize=14, fontweight='bold')
#         ax.set_title('Distribution of Total Carbon Saved by Persona\n(Mean ± SD across epistemic runs)', 
#                     fontsize=16, fontweight='bold')
#         ax.set_xticks(x)
#         #ax.set_xticklabels(persona_labels, fontsize=11)
#         ax.legend(fontsize=11, ncol=min(3, (n_scenarios + 2) // 3))
#         ax.grid(True, alpha=0.3, axis='y')
        
#         if y_axis_zero:
#             ax.set_ylim(bottom=0)
        
#         plt.tight_layout()
#         plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"Saved: {filename}")
    
#     budgets = selected_projects_df['budget'].unique()
#     print(selected_projects_df.columns.tolist() )
#     if len(budgets) > 1:
#         print(f"Warning: Plot 5 (Socio-economic) only plotting for first budget: £{budgets[0]/1e6:.0f}M")
#     for budget_plot in budgets:
#         plot_carbon_by_persona_single(selected_projects_df, budget_plot, scenario_colors, filename, y_axis_zero)
      

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# def plot_metric_by_group_mean(selected_projects_df, scenario_colors, filename, 
#                          value_col='total_co2_saved_robust',
#                          group_col='meta_socio_persona',
#                          xlabel='Socio-economic Persona',
#                          ylabel='Total Carbon Saved (tons CO₂)',
#                          title='Distribution of Total Carbon Saved by Persona',
#                          y_axis_zero=True):
#     """
#     Generic function to plot distribution of any metric by a specific group.
#     Shows mean across epistemic runs with error bars for SD.
    
#     Parameters:
#     - selected_projects_df: Row-level dataframe.
#     - scenario_colors: Dict mapping scenario names to colors.
#     - filename: Output filename base.
#     - value_col: The column name of the metric to sum and plot (e.g., carbon, cost).
#     - group_col: The column name to group by (e.g., meta_socio_persona, region).
#     - xlabel: Custom label for the X-axis.
#     - ylabel: Custom label for the Y-axis.
#     - title: Custom main title for the plot.
#     - y_axis_zero: Whether to start y-axis at 0.
#     """
 
#     def plot_single_budget(df, budget_to_plot):
#         # Create specific filename for this budget
#         budget_str = f"{budget_to_plot/1e6:.0f}M"
#         rl_filename = f'{filename.split(".png")[0]}_budget{budget_str}.png'
        
#         subset = df[df['budget'] == budget_to_plot].copy()
        
#         # Step 1: Sum flexible metric per flexible group per epistemic run
#         grouped_by_run = subset.groupby(
#             ['scenario', group_col]
#         )[value_col].mean().reset_index()
        
#         # Standardize internal column names for generic processing
#         grouped_by_run.columns = ['scenario', 'group', 'mean_val']
        
#         # # Step 2: Calculate mean and SD across epistemic runs
#         # run_stats = grouped_by_run.groupby(['scenario', 'group']).agg(
#         #     mean_val=('total_value', 'mean'),
#         #     sd_val=('total_value', 'std'),
#         #     n_runs=('total_value', 'count')
#         # ).reset_index()
        
#         # Step 3: Create plot
#         scenarios = grouped_by_run['scenario'].unique()
#         print('Scenarios: ') 
#         print( scenarios)

#         # Sort groups for consistent plotting order
#         groups = sorted(grouped_by_run['group'].unique())
        
#         # Optional: Map common messy labels to cleaner ones (safe to keep even if not using personas)
 

        
#         # .get(g, str(g)) ensures it falls back to the original value if not in the map
#         group_labels = [label_cleaner_map.get(g, str(g)) for g in groups]
        
#         fig, ax = plt.subplots(figsize=(14, 7))
        
#         x = np.arange(len(groups))
#         n_scenarios = len(scenarios)
#         width = 0.8 / n_scenarios

#         for i, scenario in enumerate(scenarios):
#             scenario_data = grouped_by_run[grouped_by_run['scenario'] == scenario]
            
#             # Ensure data exists for all groups (fill with 0 if missing for a scenario)
#             means = []
#             sds = []
#             for group in groups:
#                 group_row = scenario_data[scenario_data['group'] == group]
#                 if not group_row.empty:
#                     means.append(group_row['mean_val'].iloc[0])
#                     # sds.append(group_row['sd_val'].iloc[0])
#                 else:
#                     means.append(0)
#                     # sds.append(0)
            
#             offset = (i - n_scenarios/2 + 0.5) * width
            
#             clean_scenario_name = scenario_label_map.get(scenario, scenario)
            
#             ax.bar(x + offset, means, width, 
#                 # yerr=sds,
#                 label=clean_scenario_name,
#                 color=scenario_colors.get(scenario, f'C{i}'),
#                 alpha=0.7,
#                 capsize=3)
        
#         # Apply flexible labels
#         ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
#         ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
#         ax.set_title(f'{title}\n(Mean ± SD across epistemic runs)', 
#                     fontsize=16, fontweight='bold')
#         ax.set_xticks(x)
#         ax.set_xticklabels(group_labels, fontsize=11)
#         ax.legend(fontsize=11, ncol=min(3, (n_scenarios + 2) // 3))
#         ax.grid(True, alpha=0.3, axis='y')
        
#         if y_axis_zero:
#             ax.set_ylim(bottom=0)
        
#         plt.tight_layout()
#         plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"Saved: {rl_filename}")
    
#     # Main loop over budgets
#     budgets = selected_projects_df['budget'].unique()
#     for budget in budgets:
#         plot_single_budget(selected_projects_df, budget)



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_metric_by_group(selected_projects_df, scenario_colors, filename, 
                         value_col='total_co2_saved_robust',
                         group_col='meta_socio_persona',
                         metric_stat='mean',  # <--- NEW TOGGLE ('mean', 'sum', 'count', etc.)
                         xlabel='Socio-economic Persona',
                         ylabel=None,         # Set to None to auto-generate based on stat
                         title=None,          # Set to None to auto-generate based on stat
                         group_label_map=None,
                         scenario_label_map=None,
                         y_axis_zero=True):
    """
    Generic function to plot distribution of any metric by group using a specific statistic (mean/sum).
    
    Parameters:
    - metric_stat: 'mean', 'sum', 'median', 'count', etc. (passed to pivot_table)
    - ylabel: Y-axis label. If None, auto-generates based on metric_stat.
    - title: Plot title. If None, auto-generates based on metric_stat.
    """
    
    if group_label_map is None: group_label_map = {}
    if scenario_label_map is None: scenario_label_map = {}

    # Auto-generate dynamic labels if not provided
    if ylabel is None:
        stat_name = "Total" if metric_stat == 'sum' else metric_stat.capitalize()
        ylabel = f"{stat_name} {value_col.replace('_', ' ').title()}"
        
    if title is None:
        stat_name = "Total" if metric_stat == 'sum' else metric_stat.capitalize()
        title = f"Distribution of {stat_name} {value_col.replace('_', ' ')} by Group"

    budgets = selected_projects_df['budget'].unique()
    
    for budget in budgets:
        # 1. Filter for current budget
        subset = selected_projects_df[selected_projects_df['budget'] == budget].copy()
        
        # 2. Pivot and Aggregate
        # pivot_table handles grouping, pivoting, and aggregation in one step
        # fill_value=0 replaces NaNs with 0 immediately
        plot_data = subset.pivot_table(
            index=group_col, 
            columns='scenario', 
            values=value_col,
            aggfunc=metric_stat,
            fill_value=0
        )
        
        # Sort groups (index)
        plot_data = plot_data.sort_index()
        
        groups = plot_data.index.tolist()
        scenarios = plot_data.columns.tolist()
        
        # 3. Create Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(groups))
        n_scenarios = len(scenarios)
        width = 0.8 / n_scenarios
        
        for i, scenario in enumerate(scenarios):
            offset = (i - n_scenarios/2 + 0.5) * width
            
            # Get display names and colors
            display_label = scenario_label_map.get(scenario, scenario)
            color = scenario_colors.get(scenario, f'C{i}')
            
            ax.bar(x + offset, 
                   plot_data[scenario], 
                   width, 
                   label=display_label,
                   color=color,
                   alpha=0.7)
        
        # 4. Styling
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{title}\n(Budget: £{budget/1e6:.1f}M)', 
                    fontsize=16, fontweight='bold')
        
        # Clean x-axis labels
        clean_labels = [group_label_map.get(g, str(g)) for g in groups]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, fontsize=11, rotation=45, ha='right')
        
        ax.legend(fontsize=11, ncol=min(3, max(1, (n_scenarios + 2) // 3)))
        ax.grid(True, alpha=0.3, axis='y')
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        # 5. Save File
        base, ext = os.path.splitext(filename)
        if not ext: ext = ".png"
        
        # Add the stat type to the filename so they don't overwrite each other
        rl_filename = f'{base}_{metric_stat}_budget{budget/1_000_000:.1f}M{ext}'
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ({metric_stat}): {rl_filename}")

def plot_count_by_group(selected_projects_df, scenario_colors, filename, 
                       group_col='meta_socio_persona',
                       xlabel='Socio-economic Persona',
                       ylabel='Number of Projects',
                       title='Distribution of Project Count by Persona',
                       y_axis_zero=True):
    """
    Generic function to plot count of items by a specific group.
    Shows mean count across epistemic runs with error bars for SD.
    
    Parameters:
    - selected_projects_df: Row-level dataframe.
    - scenario_colors: Dict mapping scenario names to colors.
    - filename: Output filename base.
    - group_col: The column name to group by (e.g., meta_socio_persona, region).
    - xlabel: Custom label for the X-axis.
    - ylabel: Custom label for the Y-axis.
    - title: Custom main title for the plot.
    - y_axis_zero: Whether to start y-axis at 0.
    """
 
    def plot_single_budget(df, budget_to_plot):
        # Create specific filename for this budget
        budget_str = f"{budget_to_plot/1e6:.0f}M"
        rl_filename = f'{filename.split(".png")[0]}_budget{budget_str}.png'
        
        subset = df[df['budget'] == budget_to_plot].copy()
        
        # Step 1: Count items per group per epistemic run
        grouped_by_run = subset.groupby(
            ['scenario', group_col]
        ).size().reset_index(name='count')
        
        # Standardize internal column names for generic processing
        grouped_by_run.columns = ['scenario', 'group', 'mean_count']
        
        # # Step 2: Calculate mean and SD of counts across epistemic runs
        # run_stats = grouped_by_run.groupby(['scenario', 'group']).agg(
        #     mean_count=('total_count', 'mean'),
        #     sd_count=('total_count', 'std'),
        #     n_runs=('total_count', 'count')
        # ).reset_index()
        
        # Step 3: Create plot
        scenarios = grouped_by_run['scenario'].unique()
        # Sort groups for consistent plotting order
        groups = sorted(grouped_by_run['group'].unique())
        
        # Optional: Map common messy labels to cleaner ones
 
        group_labels = [label_cleaner_map.get(g, str(g)) for g in groups]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(groups))
        n_scenarios = len(scenarios)
        width = 0.8 / n_scenarios
        
        for i, scenario in enumerate(scenarios):
            scenario_data = grouped_by_run[grouped_by_run['scenario'] == scenario]
            
            # Ensure data exists for all groups (fill with 0 if missing for a scenario)
            means = []
            # sds = []
            for group in groups:
                group_row = scenario_data[scenario_data['group'] == group]
                if not group_row.empty:
                    means.append(group_row['mean_count'].iloc[0])
                    # sds.append(group_row['sd_count'].iloc[0])
                else:
                    means.append(0)
                    # sds.append(0)
            
            offset = (i - n_scenarios/2 + 0.5) * width
            
            clean_scenario_name = scenario_label_map.get(scenario, scenario)
            
            ax.bar(x + offset, means, width, 
                # yerr=sds,
                label=clean_scenario_name,
                color=scenario_colors.get(scenario, f'C{i}'),
                alpha=0.7,
                capsize=3)
        
        # Apply flexible labels
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{title}\n(Mean ± SD across epistemic runs)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=11)
        ax.legend(fontsize=11, ncol=min(3, (n_scenarios + 2) // 3))
        ax.grid(True, alpha=0.3, axis='y')
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {rl_filename}")
    
    # Main loop over budgets
    budgets = selected_projects_df['budget'].unique()
    for budget in budgets:
        plot_single_budget(selected_projects_df, budget)

def plot_pareto_front(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename, y_axis_zero=False):
    """
    Plot 6: Pareto Front - Equity vs Carbon.
    
    1. Plots a combined chart with all budgets (saves as 'filename').
    2. Plots a separate chart for each individual budget (saves as 'filename_budget_X').
    """
    
    # --- PART 1: Plot Combined Pareto for all budgets (as before) ---
    
    fig_comb, ax_comb = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Get all unique budgets from the subset
    all_budgets = sorted(results_subset['budget'].unique())
    
    # Plot each budget as a separate series
    for budget_val in all_budgets:
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()
        
        vuln_means = []
        # vuln_stds = []
        co2_means = []
        co2_stds = []
        weights = []
        colors = []

        for scenario in scenarios_subset:
            # Skip if scenario is missing from either dataframe
            if scenario not in equity_subset['scenario'].values or \
               scenario not in results_subset['scenario'].values:
                continue
            
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
            
            vuln_means.append(equity_row['high_deprived_pct'] )
            # vuln_stds.append(equity_row['high_deprived_pct_std'] )
            co2_means.append(results_row[total_co2_saved_col] / 1e3)
            # co2_stds.append(results_row[total_co2_saved_col] / 1e3)
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Skip if this budget had no valid scenarios
        if not weights:
            continue
            
        label = f'£{budget_val/1e6:.0f}M'
        
        # Plot points with error bars
        for i in range(len(vuln_means)): # Use vuln_means len as it matches collected data
            ax_comb.errorbar(vuln_means[i], co2_means[i],
                             fmt='o', markersize=12, capsize=5,
                             color=colors[i], 
                             label=f'EW={weights[i]}' if budget_val == all_budgets[0] else None) # Only label weights once

        # Draw connecting line
        sorted_indices = np.argsort(weights)
        vuln_means_sorted = [vuln_means[i] for i in sorted_indices]
        co2_means_sorted = [co2_means[i] for i in sorted_indices]
        ax_comb.plot(vuln_means_sorted, co2_means_sorted, '--', alpha=0.5, linewidth=2, label=f'Tradeoff Curve ({label})')
    
    ax_comb.set_xlabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax_comb.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    ax_comb.set_title(f'Equity-Carbon Tradeoff Curve\n{budget_label}', fontsize=16, fontweight='bold')
    
    # Consolidate legends
    handles, labels = ax_comb.get_legend_handles_labels()
    if handles: # Only show legend if there's something to plot
        by_label = dict(zip(labels, handles))
        ax_comb.legend(by_label.values(), by_label.keys(), fontsize=11, 
                       ncol=1 if len(by_label) <= 6 else 2)
    
    ax_comb.grid(True, alpha=0.3)
    
    if y_axis_zero:
        ax_comb.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig_comb) # Explicitly close the combined figure
    print(f"Saved Combined Plot: {filename}")
    
    # --- PART 2: Plot individual Pareto for each budget ---
    
    # Get base filename and extension (e.g., "plots/pareto" and ".png")
    base_filename, file_extension = os.path.splitext(filename)
    
    for budget_val in all_budgets:
        # Create a new, separate figure for each budget
        fig_ind, ax_ind = plt.figure(figsize=(10, 8)), plt.gca()
        
        # Filter data for this specific budget
        budget_results_subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_for_budget = budget_results_subset['scenario'].unique()
        
        # Filter equity data to only include scenarios relevant to this budget
        budget_equity_subset = equity_subset[equity_subset['scenario'].isin(scenarios_for_budget)]

        vuln_means_ind = []
        # vuln_stds_ind = []
        co2_means_ind = []
        # co2_stds_ind = []
        weights_ind = []
        colors_ind = []

        for scenario in scenarios_for_budget:
            # Skip if scenario is missing from this budget's filtered data
            if scenario not in budget_equity_subset['scenario'].values:
                continue
            
            equity_row = budget_equity_subset[budget_equity_subset['scenario'] == scenario].iloc[0]
            results_row = budget_results_subset[budget_results_subset['scenario'] == scenario].iloc[0]
            
            vuln_means_ind.append(equity_row['high_deprived_pct'] )
            # vuln_stds_ind.append(equity_row['high_deprived_pct_std'] )
            co2_means_ind.append(results_row[total_co2_saved_col] / 1e3)
            # co2_stds_ind.append(results_row[total_co2_saved_col_std] / 1e3)
            weights_ind.append(equity_row['equity_weight'])
            colors_ind.append(scenario_colors[scenario])

        # Skip this budget if no valid data was found
        if not weights_ind:
            plt.close(fig_ind) # Close the empty figure
            continue
            
        # Plot points with error bars
        for i in range(len(vuln_means_ind)):
            ax_ind.errorbar(vuln_means_ind[i], co2_means_ind[i],
                            fmt='o', markersize=12, capsize=5,
                            color=colors_ind[i], 
                            label=f'EW={weights_ind[i]}') # Label weights for every individual plot

        # Draw connecting line
        sorted_indices_ind = np.argsort(weights_ind)
        vuln_means_sorted_ind = [vuln_means_ind[i] for i in sorted_indices_ind]
        co2_means_sorted_ind = [co2_means_ind[i] for i in sorted_indices_ind]
        ax_ind.plot(vuln_means_sorted_ind, co2_means_sorted_ind, '--', alpha=0.5, linewidth=2, label='Tradeoff Curve')
        
        ax_ind.set_xlabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
        ax_ind.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
        
        # Create a specific title for this budget
        budget_specific_label = f'£{budget_val/1e6:.0f}M Budget'
        ax_ind.set_title(f'Equity-Carbon Tradeoff ({budget_specific_label})\n{budget_label}', fontsize=16, fontweight='bold')
        
        # Consolidate legends
        handles_ind, labels_ind = ax_ind.get_legend_handles_labels()
        by_label_ind = dict(zip(labels_ind, handles_ind))
        ax_ind.legend(by_label_ind.values(), by_label_ind.keys(), fontsize=11, 
                      ncol=1 if len(by_label_ind) <= 6 else 2)
        
        ax_ind.grid(True, alpha=0.3)
        
        if y_axis_zero:
            ax_ind.set_ylim(bottom=0)
        
        # Create the new indexed filename
        new_filename = f"{base_filename}_budget_{int(budget_val)}{file_extension}"
        
        plt.tight_layout()
        plt.savefig(new_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_ind) # Close the individual figure
        print(f"Saved Individual Plot: {new_filename}")


def plot_vulnerable_groups_coverage(equity_subset, filename, y_axis_zero=False):
    """Plot 7: Most Vulnerable Groups Coverage"""
    

    def plot_one_budget(budget_to_plot,  equity_subset, filename, y_axis_zero):
        rl_filename = f'{filename.split(".png")[0]}_budget{budget_to_plot}.png'
        equity_subset = equity_subset[equity_subset['budget'] == budget_to_plot]
        budget_label = f'Budget £{budget_to_plot/1e6:.0f}M'
        scenarios = equity_subset['scenario'].unique()
        equity_weights = sorted(equity_subset['equity_weight'].unique())
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        x_pos = np.arange(len(scenarios))
        width = 0.35
        
        # *** These columns were simple strings, so NO change needed ***
        deprived_means = [equity_subset[equity_subset['scenario'] == s]['high_deprived_pct'].iloc[0] * 100 
                        for s in scenarios]

        
        bars1 = ax.bar(x_pos - width/2, deprived_means, width, label='Deprived', 
                    color='#c0392b', alpha=0.8)
 
        
        ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Most Vulnerable Groups Coverage\n{budget_label}', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{w}' for w in equity_weights])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # *** ADDED: Set y-axis to 0 if toggled ***
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {rl_filename}")
    

    for budget_to_plot in equity_subset['budget'].unique():
        plot_one_budget(budget_to_plot, equity_subset, filename, y_axis_zero)

def plot_tradeoff_efficiency(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename, y_axis_zero=False):
    """Plot 8: Tradeoff Efficiency"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot for each budget
    bar_width = 0.8 / len(results_subset['budget'].unique())
    
    for i, budget_val in enumerate(results_subset['budget'].unique()):
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()
        
        if len(scenarios_subset) >= 2:
            sorted_scenarios = sorted(scenarios_subset, 
                                     key=lambda s: equity_subset[equity_subset['scenario'] == s]['equity_weight'].iloc[0])
            base_scenario = sorted_scenarios[0]
            
            # *** UPDATED COLUMN NAMES ***
            base_vuln = equity_subset[equity_subset['scenario'] == base_scenario]['high_deprived_pct'].iloc[0] 
            base_co2 = results_subset[results_subset['scenario'] == base_scenario][total_co2_saved_col].iloc[0] / 1e3
            
            tradeoff_scenarios = []
            tradeoff_ratios = []
            tradeoff_labels = []
            
            for scenario in sorted_scenarios[1:]:
                equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
                results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
                
                # *** UPDATED COLUMN NAMES ***
                vuln_cov = equity_row['high_deprived_pct'] * 100
                co2_saved = results_row[total_co2_saved_col] / 1e3
                
                vuln_gain = vuln_cov - base_vuln
                co2_loss = base_co2 - co2_saved
                
                if co2_loss > 0.001:
                    ratio = vuln_gain / co2_loss
                else:
                    ratio = 0  # No loss, infinite gain (or no change)
                
                tradeoff_scenarios.append(scenario)
                tradeoff_ratios.append(ratio)
                tradeoff_labels.append(f'{equity_row["equity_weight"]}')
            
            offset = (i - len(results_subset['budget'].unique())/2 + 0.5) * bar_width
            x_pos = np.arange(len(tradeoff_ratios))
            label = f'£{budget_val/1e6:.0f}M'
            
            bars = ax.bar(x_pos + offset, tradeoff_ratios, bar_width, label=label, alpha=0.7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(tradeoff_labels)

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable % Gain per kton CO2 Lost', fontsize=14, fontweight='bold')
    ax.set_title(f'Equity-Carbon Tradeoff Efficiency\n(vs. EW=0 baseline)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    if len(results_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    # *** ADDED: Set y-axis to 0 if toggled ***
    if y_axis_zero:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_radar_chart(results_subset, equity_subset,  filename, scenario_colors):
    """
    Plot 9: Multi-Metric Radar Chart
    Note: y_axis_zero has no effect here as the axis is normalized 0-1.
    """
    
    def plot_one_budget(results_subset, budget_to_plot,  equity_subset, filename,  scenario_colors):
        rl_filename = f'{filename.split(".png")[0]}_budget{budget_to_plot}.png'
        equity_subset = equity_subset[equity_subset['budget'] == budget_to_plot]
        results_subset = results_subset[results_subset['budget'] == budget_to_plot]
        budget_label = f'Budget £{budget_to_plot/1e6:.0f}M'
        scenarios = equity_subset['scenario'].unique()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        metrics = ['Carbon\nSavings', 'Cost\nEfficiency', 'Vulnerable\nCoverage', 
                'Equity\nBalance', '# Buildings']
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # *** UPDATED COLUMN NAMES for normalization ***
        norm_co2 = results_subset[total_co2_saved_col].max()
        norm_cost_eff = (1 / results_subset[capex_per_net_ton_mean_col]).max()
        norm_vuln = equity_subset['high_deprived_pct'].max()
        norm_equity = equity_subset['equity_concentration'].max()
        norm_buildings = results_subset['num_buildings_sum'].max()

        for scenario in scenarios:
            results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            
            # Normalize metrics (0-1, higher is better)
            # *** UPDATED COLUMN NAMES ***
            carbon_norm = results_row[total_co2_saved_col] / norm_co2
            cost_eff_norm = (1 / results_row[capex_per_net_ton_mean_col]) / norm_cost_eff
            vuln_norm = equity_row['high_deprived_pct'] / norm_vuln
            equity_norm = 1 - (equity_row['equity_concentration'] / norm_equity)
            buildings_norm = results_row['num_buildings_sum'] / norm_buildings
            
            values = [carbon_norm, cost_eff_norm, vuln_norm, equity_norm, buildings_norm]
            values += values[:1]
            
            weight = equity_row['equity_weight']
            label = f'EW={weight}'
            
            ax.plot(angles, values, 'o-', linewidth=2.5, label=label, 
                    color=scenario_colors[scenario], markersize=8)
            ax.fill(angles, values, alpha=0.2, color=scenario_colors[scenario])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1) # Radar chart y-axis (radius) is hard-coded 0-1
        ax.set_title(f'Normalized Performance Comparison\n{budget_label}', fontsize=16, 
                    fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11,
                ncol=1 if len(scenarios) <= 5 else 2)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {rl_filename}")

    for bugdet in equity_subset['budget'].unique():
        plot_one_budget(results_subset, bugdet,  equity_subset, filename , scenario_colors)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os # Added for splitting filename

def plot_pareto_retrofit_carbon_by_budget(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename, y_axis_zero=False):
    """
    Plot: Pareto Front - Buildings Retrofitted vs Carbon
    Generates a SEPARATE plot file for each budget.
    """
    
    # Get the base filename and extension to create unique names
    base_name, extension = os.path.splitext(filename)
    
    # Loop over each budget and create a unique plot
    for budget_val in sorted(results_subset['budget'].unique()):
        
        # Create a new figure for each budget
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        
        # --- Data collection for this specific budget ---
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()
        
        buildings_means = []
        buildings_stds = []
        co2_means = []
        co2_stds = []
        weights = []
        colors = []

        for scenario in scenarios_subset:
            # Ensure scenario exists in both subsets before trying to access
            if scenario not in equity_subset['scenario'].values or scenario not in results_subset['scenario'].values:
                print(f"Warning: Scenario {scenario} not found. Skipping.")
                continue
                
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
            
            buildings_means.append(results_row['num_buildings_sum'])
            # buildings_stds.append(results_row['num_buildings_sum_std'])
            co2_means.append(results_row[total_co2_saved_col] / 1e3)
            # co2_stds.append(results_row[total_co2_saved_col_std] / 1e3)
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Specific label for this budget
        current_budget_label = f'£{budget_val/1e6:.0f}M'
        
        # --- Plotting for this specific budget ---
        
        # Plot points with error bars
        for i in range(len(scenarios_subset)):
            ax.errorbar(buildings_means[i], co2_means[i] , 
                       fmt='o', markersize=12, capsize=5,
                       color=colors[i], 
                       label=f'EW={weights[i]}') # Label weights on every plot

        # Draw connecting line
        sorted_indices = np.argsort(weights)
        buildings_means_sorted = [buildings_means[i] for i in sorted_indices]
        co2_means_sorted = [co2_means[i] for i in sorted_indices]
        ax.plot(buildings_means_sorted, co2_means_sorted, '--', alpha=0.5, linewidth=2, label='Tradeoff Curve')
    
        # --- Formatting for this specific plot ---
        ax.set_xlabel('Number of Buildings Retrofitted', fontsize=14, fontweight='bold')
        ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
        
        # Updated title to include the specific budget
        ax.set_title(f'Retrofit-Carbon Tradeoff Curve ({current_budget_label})\n{budget_label}', fontsize=16, fontweight='bold')
        
        # Consolidate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=11, 
                  ncol=1 if len(by_label) <= 6 else 2)
        
        ax.grid(True, alpha=0.3)
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        # --- Save and close this specific plot ---
        
        # Create the new filename
        budget_suffix = f"_budget_{budget_val/1e6:.0f}M"
        new_filename = f"{base_name}{budget_suffix}{extension}"
        
        plt.tight_layout()
        plt.savefig(new_filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved: {new_filename}")


import os
import matplotlib.pyplot as plt
import numpy as np

# NOTE: The imports for os, plt, and np are assumed based on the original function's content.
# You will need to ensure they are imported in your script.

def plot_pareto_retrofit_carbon_by_costeff(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename, y_axis_zero=False):
    """
    Plot: Pareto Front - Buildings Retrofitted vs Cost-Effectiveness
    Generates a SEPARATE plot file for each budget.
    
    
    """
    
    # Get the base filename and extension to create unique names
    base_name, extension = os.path.splitext(filename)
    
    # Loop over each budget and create a unique plot
    for budget_val in sorted(results_subset['budget'].unique()):
        
        # Create a new figure for each budget
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        
        # --- Data collection for this specific budget ---
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()
        
        buildings_means = []
        buildings_stds = []
        # CHANGED: Renamed lists for clarity (cost-effectiveness)
        cost_eff_means = []
        cost_eff_stds = []
        weights = []
        colors = []

        for scenario in scenarios_subset:
            # Ensure scenario exists in both subsets before trying to access
            if scenario not in equity_subset['scenario'].values or scenario not in results_subset['scenario'].values:
                print(f"Warning: Scenario {scenario} not found. Skipping.")
                continue
                
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            # CHANGED: Grab the results_row for the *specific budget*
            results_row = subset[subset['scenario'] == scenario].iloc[0]
            
            buildings_means.append(results_row['num_buildings_sum'])
            # buildings_stds.append(results_row['num_buildings_sum_std'])
            
            # CHANGED: Use the new cost-effectiveness variables for the Y-axis
            cost_eff_means.append(results_row[capex_per_net_ton_mean_col])
            # cost_eff_stds.append(results_row[capex_per_net_ton_std_col])
            
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Specific label for this budget
        current_budget_label = f'£{budget_val/1e6:.0f}M'
        
        # --- Plotting for this specific budget ---
        
        # Plot points with error bars
        for i in range(len(scenarios_subset)):
            # CHANGED: Use cost_eff_means and cost_eff_stds for Y-axis and y-error
            ax.errorbar(buildings_means[i], cost_eff_means[i], 
                      
                       fmt='o', markersize=12, capsize=5,
                       color=colors[i], 
                       label=f'EW={weights[i]}') # Label weights on every plot

        # Draw connecting line
        sorted_indices = np.argsort(weights)
        buildings_means_sorted = [buildings_means[i] for i in sorted_indices]
        # CHANGED: Use cost_eff_means for the sorted Y-axis data
        cost_eff_means_sorted = [cost_eff_means[i] for i in sorted_indices]
        # CHANGED: Plot the new Y-axis data
        ax.plot(buildings_means_sorted, cost_eff_means_sorted, '--', alpha=0.5, linewidth=2, label='Tradeoff Curve')
    
        # --- Formatting for this specific plot ---
        ax.set_xlabel('Number of Buildings Retrofitted', fontsize=14, fontweight='bold')
        # CHANGED: Updated Y-axis label
        ax.set_ylabel('Cost Effectiveness (£ / net ton CO2)', fontsize=14, fontweight='bold')
        
        # CHANGED: Updated title to reflect new Y-axis
        ax.set_title(f'Retrofit-Cost-Effectiveness Tradeoff ({current_budget_label})\n{budget_label}', fontsize=16, fontweight='bold')
        
        # Consolidate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=11, 
                  ncol=1 if len(by_label) <= 6 else 2)
        
        ax.grid(True, alpha=0.3)
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        # --- Save and close this specific plot ---
        
        # Create the new filename
        budget_suffix = f"_budget_{budget_val/1e6:.0f}M"
        # CHANGED: Added a suffix to the base name to reflect new metric
        new_filename = f"{base_name}_cost_eff{budget_suffix}{extension}"
        
        plt.tight_layout()
        plt.savefig(new_filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved: {new_filename}")