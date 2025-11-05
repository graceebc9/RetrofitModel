import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

 
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
    
    plot_socioeconomic_distribution(equity_subset, scenario_colors, 
                                    os.path.join(output_dir, "5_socioeconomic_distribution.png"),
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

    print(f"Done! All 9 plots saved to '{output_dir}'.")
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
        return sns.color_palette('husl', n_colors).as_hex()

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
        means = subset['total_ton_co2_saved_mean_sum_mean'].values / 1e3
        stds = subset['total_ton_co2_saved_mean_sum_std'].values / 1e3
        
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
        means = subset['cost_per_net_ton_co2_kg_mean_mean_mean'].values
        stds = subset['cost_per_net_ton_co2_kg_mean_mean_std'].values
        
        label = f'£{budget_val/1e6:.0f}M' if len(results_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
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
        means = subset['vulnerable_pct_mean'].values * 100
        stds = subset['vulnerable_pct_std'].values * 100
        
        label = f'£{budget_val/1e6:.0f}M' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
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
        means = subset['equity_concentration_mean'].values
        stds = subset['equity_concentration_std'].values
        
        label = f'£{budget_val/1e6:.0f}M' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
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

def plot_socioeconomic_distribution(equity_subset, scenario_colors, filename, y_axis_zero=False):
    """Plot 5: Socio-economic Distribution by Equity Weight"""
    
    # This plot is complex with budgets, so we'll just plot for the *first* budget found
    # Or you could adapt this to create one plot per budget
    
  
    def plot_one_budget(equity_subset, budget_to_plot, filename, scenario_colors, y_axis_zero):
        # budget_to_plot = budgets[0]
        rl_filename =f'{filename.split(".png")[0]}_budget{budget_to_plot}.png'
        equity_subset = equity_subset[equity_subset['budget'] == budget_to_plot]
        budget_label = f'Budget £{budget_to_plot/1e6:.0f}M'
        scenarios = equity_subset['scenario'].unique()
        
        fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
        
        # *** These columns were simple strings, so NO change needed ***
        socio_groups = ['deprived_pct', 'struggling_pct', 'lower middle_pct', 
                        'upper middle_pct', 'affluent_pct', 'student_pct']
        socio_labels = ['Deprived', 'Struggling', 'Lower\nMiddle', 
                        'Upper\nMiddle', 'Affluent', 'Student']
        
        x = np.arange(len(socio_labels))
        n_scenarios = len(scenarios)
        width = 0.8 / n_scenarios
        
        for i, scenario in enumerate(scenarios):
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            means = [equity_row[f'{group}_mean'] for group in socio_groups]
            offset = (i - n_scenarios/2 + 0.5) * width
            
            weight = equity_row['equity_weight']
            label = f'EW={weight}'
            ax.bar(x + offset, means, width, label=label, 
                color=scenario_colors[scenario], alpha=0.7)
        
        ax.set_xlabel('Socio-economic Group', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Socio-economic Distribution by Equity Weight\n{budget_label}', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(socio_labels, fontsize=11)
        ax.legend(fontsize=11, ncol=min(2, (n_scenarios + 1) // 2))
        ax.grid(True, alpha=0.3, axis='y')
        
        # *** ADDED: Set y-axis to 0 if toggled ***
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {rl_filename}")
    budgets = equity_subset['budget'].unique()
    if len(budgets) > 1:
        print(f"Warning: Plot 5 (Socio-economic) only plotting for first budget: £{budgets[0]/1e6:.0f}M")
    for budget_plot in budgets:
        plot_one_budget(equity_subset, budget_plot, filename, scenario_colors, y_axis_zero)
    

import matplotlib.pyplot as plt
import numpy as np
import os  # <-- Required for splitting filenames

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
        vuln_stds = []
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
            
            vuln_means.append(equity_row['vulnerable_pct_mean'] )
            vuln_stds.append(equity_row['vulnerable_pct_std'] )
            co2_means.append(results_row['total_ton_co2_saved_mean_sum_mean'] / 1e3)
            co2_stds.append(results_row['total_ton_co2_saved_mean_sum_std'] / 1e3)
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Skip if this budget had no valid scenarios
        if not weights:
            continue
            
        label = f'£{budget_val/1e6:.0f}M'
        
        # Plot points with error bars
        for i in range(len(vuln_means)): # Use vuln_means len as it matches collected data
            ax_comb.errorbar(vuln_means[i], co2_means[i], xerr=vuln_stds[i], yerr=co2_stds[i],
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
        vuln_stds_ind = []
        co2_means_ind = []
        co2_stds_ind = []
        weights_ind = []
        colors_ind = []

        for scenario in scenarios_for_budget:
            # Skip if scenario is missing from this budget's filtered data
            if scenario not in budget_equity_subset['scenario'].values:
                continue
            
            equity_row = budget_equity_subset[budget_equity_subset['scenario'] == scenario].iloc[0]
            results_row = budget_results_subset[budget_results_subset['scenario'] == scenario].iloc[0]
            
            vuln_means_ind.append(equity_row['vulnerable_pct_mean'] )
            vuln_stds_ind.append(equity_row['vulnerable_pct_std'] )
            co2_means_ind.append(results_row['total_ton_co2_saved_mean_sum_mean'] / 1e3)
            co2_stds_ind.append(results_row['total_ton_co2_saved_mean_sum_std'] / 1e3)
            weights_ind.append(equity_row['equity_weight'])
            colors_ind.append(scenario_colors[scenario])

        # Skip this budget if no valid data was found
        if not weights_ind:
            plt.close(fig_ind) # Close the empty figure
            continue
            
        # Plot points with error bars
        for i in range(len(vuln_means_ind)):
            ax_ind.errorbar(vuln_means_ind[i], co2_means_ind[i], xerr=vuln_stds_ind[i], yerr=co2_stds_ind[i],
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
        deprived_means = [equity_subset[equity_subset['scenario'] == s]['deprived_pct_mean'].iloc[0] * 100 
                        for s in scenarios]
        struggling_means = [equity_subset[equity_subset['scenario'] == s]['struggling_pct_mean'].iloc[0] * 100 
                            for s in scenarios]
        
        bars1 = ax.bar(x_pos - width/2, deprived_means, width, label='Deprived', 
                    color='#c0392b', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, struggling_means, width, label='Struggling', 
                    color='#e67e22', alpha=0.8)
        
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
            base_vuln = equity_subset[equity_subset['scenario'] == base_scenario]['vulnerable_pct_mean'].iloc[0] 
            base_co2 = results_subset[results_subset['scenario'] == base_scenario]['total_ton_co2_saved_mean_sum_mean'].iloc[0] / 1e3
            
            tradeoff_scenarios = []
            tradeoff_ratios = []
            tradeoff_labels = []
            
            for scenario in sorted_scenarios[1:]:
                equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
                results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
                
                # *** UPDATED COLUMN NAMES ***
                vuln_cov = equity_row['vulnerable_pct_mean'] * 100
                co2_saved = results_row['total_ton_co2_saved_mean_sum_mean'] / 1e3
                
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
        norm_co2 = results_subset['total_ton_co2_saved_mean_sum_mean'].max()
        norm_cost_eff = (1 / results_subset['cost_per_net_ton_co2_kg_mean_mean_mean']).max()
        norm_vuln = equity_subset['vulnerable_pct_mean'].max()
        norm_equity = equity_subset['equity_concentration_mean'].max()
        norm_buildings = results_subset['num_buildings_sum_mean'].max()

        for scenario in scenarios:
            results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            
            # Normalize metrics (0-1, higher is better)
            # *** UPDATED COLUMN NAMES ***
            carbon_norm = results_row['total_ton_co2_saved_mean_sum_mean'] / norm_co2
            cost_eff_norm = (1 / results_row['cost_per_net_ton_co2_kg_mean_mean_mean']) / norm_cost_eff
            vuln_norm = equity_row['vulnerable_pct_mean'] / norm_vuln
            equity_norm = 1 - (equity_row['equity_concentration_mean'] / norm_equity)
            buildings_norm = results_row['num_buildings_sum_mean'] / norm_buildings
            
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
            
            buildings_means.append(results_row['num_buildings_sum_mean'])
            buildings_stds.append(results_row['num_buildings_sum_std'])
            co2_means.append(results_row['total_ton_co2_saved_mean_sum_mean'] / 1e3)
            co2_stds.append(results_row['total_ton_co2_saved_mean_sum_std'] / 1e3)
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Specific label for this budget
        current_budget_label = f'£{budget_val/1e6:.0f}M'
        
        # --- Plotting for this specific budget ---
        
        # Plot points with error bars
        for i in range(len(scenarios_subset)):
            ax.errorbar(buildings_means[i], co2_means[i], xerr=buildings_stds[i], yerr=co2_stds[i],
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