import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

label_cleaner_map = {
    'high_risk' : 'High Risk' , 
    'med_risk': 'Medium Risk' , 
    'middle_risk': 'Middle Risk' , 
    'low_risk' : 'Low Risk', 
    'v_low_risk': 'Very Low Risk', 
}

from .personas import cluster_names 
 
# total_co2_saved_col = 'total_co2_saved_sum'
total_co2_saved_col = 'mean_total_co2_saved'
total_co2_saved_col_std ='total_co2_std_uncorr' 
total_co2_full_corr =  'total_co2_std_corr'

# capex_per_net_ton_mean_col = 'capex_per_net_ton_mean'
capex_per_net_ton_mean_col = 'mean_capex_per_net_ton'
capex_per_net_ton_std_col = 'std_capex_per_net_ton_corr'
capex_per_net_ton_std_col_unorr = 'std_capex_per_net_ton_uncorr'



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
    
    'budget_50m_equity_0': 'equity_0', 
    'budget_50m_equity_0.2': 'equity_0.2', 
    'budget_50m_equity_0.4': 'equity_0.4', 
    'budget_50m_equity_0.6': 'equity_0.6',
    'budget_50m_equity_0.8': 'equity_0.8', 
    'budget_50m_equity_1': 'equity_1',
    
    'budget_10000000000_equity_0': 'equity_0', 
    'budget_10000000000_equity_0.2': 'equity_0.2', 
    'budget_10000000000_equity_0.4': 'equity_0.4', 
    'budget_10000000000_equity_0.6': 'equity_0.6',
    'budget_10000000000_equity_0.8': 'equity_0.8', 
    'budget_10000000000_equity_1': 'equity_1',
}


def get_sorted_budget_list(df, budget_col='budget'):
    """
    Extracts unique budgets from the dataframe and returns them 
    sorted numerically (handling 'm' for million, 'k' for thousand).
    """
    unique_budgets = df[budget_col].unique()

    def parse_budget_str(val):
        s = str(val).lower().strip().replace('£', '')
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
            return 0 # Push unparseable items to the start
            
    return sorted(unique_budgets, key=parse_budget_str)

 
# plot main 
def plot_greedy_compairosn_main(df_raw, output_dir, y_axis_zero=False, loft_val = None  ):
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
    
    
    budget_label = "All Budgets"
    # unique_budgets = df_processed['budget'].unique()
    unique_budgets = sorted(df_processed['budget'].unique(), key=lambda x: float(x.replace('£', '').replace('M', '')) * 1_000_000  if isinstance(x, str) and 'm' in x.lower() else 0 )
    if len(unique_budgets) == 1:
        budget_label = f"Budget £{unique_budgets[0]}"

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
                                  os.path.join(output_dir, f"1_carbon_vs_equity_{loft_val}_UNCORR.png"),
                                  y_axis_zero=y_axis_zero)
    
    plot_carbon_savings_vs_equity(results_subset, equity_weights, budget_label, 
                                  os.path.join(output_dir, f"1_carbon_vs_equity_{loft_val}_CORR.png"),
                                  y_axis_zero=y_axis_zero, co2_std =total_co2_full_corr  )
    
    plot_cost_effectiveness_vs_equity(results_subset, equity_weights, budget_label, 
                                      os.path.join(output_dir, f"2_cost_effectiveness_vs_equity_{loft_val}_CORR.png"),
                                      y_axis_zero=y_axis_zero)
        
    plot_cost_effectiveness_vs_equity(results_subset, equity_weights, budget_label, 
                                      os.path.join(output_dir, f"2_cost_effectiveness_vs_equity_{loft_val}_UNCORR.png"),
                                      y_axis_zero=y_axis_zero, capex_std = capex_per_net_ton_std_col_unorr)
    
    plot_vulnerable_coverage_vs_equity(equity_subset, equity_weights, budget_label, 
                                       os.path.join(output_dir, f"3_vulnerable_coverage_vs_equity_{loft_val}.png"),
                                       y_axis_zero=y_axis_zero)
    
    plot_equity_concentration_vs_weight(equity_subset, equity_weights, budget_label, 
                                        os.path.join(output_dir, f"4_equity_concentration_vs_weight_{loft_val}.png"),
                                        y_axis_zero=y_axis_zero)
    
 
    
    plot_pareto_front(results_subset, equity_subset, scenarios, scenario_colors, budget_label, 
                      os.path.join(output_dir, f"6_pareto_front_{loft_val}.png"),
                      y_axis_zero=y_axis_zero, 
                         ycol=total_co2_saved_col,
        ycol_std= total_co2_saved_col_std, )

    plot_pareto_front(results_subset, equity_subset, scenarios, scenario_colors, budget_label, 
                      os.path.join(output_dir, f"6_pareto_front_{loft_val}_FULLCORR.png"),
                      y_axis_zero=y_axis_zero, 
                         ycol=total_co2_saved_col,
        ycol_std= total_co2_full_corr, )
    
 
    
    plot_tradeoff_efficiency(results_subset, equity_subset, scenarios, scenario_colors, budget_label, 
                             os.path.join(output_dir, f"8_tradeoff_efficiency_{loft_val}.png"),
                             y_axis_zero=y_axis_zero)
    
    plot_radar_chart(results_subset, equity_subset, 
                     os.path.join(output_dir, f"9_radar_chart_{loft_val}.png"),scenario_colors
                    ) 

 
    plot_pareto_front_flexi(
        results_subset, 
        equity_subset, 
        scenarios, 
        scenario_colors, 
        budget_label, 
        filename=   os.path.join(output_dir, f"10_pareto_bcounts_{loft_val}.png" ) , 
        x_col='num_buildings_sum', 
        
        y_col=total_co2_saved_col,
        y_col_std= total_co2_saved_col_std, 
        x_label='Number of buildings',
        y_label='CO2 Saved (kton)',
        x_scaler=1.0,
        y_scaler=1e3, 
        y_axis_zero=False
        )
    
    plot_pareto_front_flexi(
        results_subset, 
        equity_subset, 
        scenarios, 
        scenario_colors, 
        budget_label, 
        filename=   os.path.join(output_dir, f"10_pareto_bcounts_{loft_val}_FULLCORR.png" ) , 
        x_col='num_buildings_sum', 
        
        y_col=total_co2_saved_col,
        y_col_std= total_co2_full_corr, 
        x_label='Number of buildings',
        y_label='CO2 Saved (kton)',
        x_scaler=1.0,
        y_scaler=1e3, 
        y_axis_zero=False
        )
    
    plot_pareto_retrofit_carbon_by_costeff(results_subset, equity_subset, scenarios, scenario_colors, budget_label,
                                 os.path.join(output_dir, f"11_pareto_cost_eff__loft_{loft_val}_UNCORR.png") ,
                                y_axis_zero=y_axis_zero)

    plot_pareto_retrofit_carbon_by_costeff(results_subset, equity_subset, scenarios, scenario_colors, budget_label,
                                 os.path.join(output_dir, f"11_pareto_cost_eff__loft_{loft_val}_CORR.png") ,
                                y_axis_zero=y_axis_zero, capex_std = capex_per_net_ton_std_col_unorr)



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
        df_flat['budget'] = df_flat['scenario'].str.split('_').str[1]
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
        return sns.color_palette('Set3', n_colors).as_hex()

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
        # plt.title(f"Impact of Equity Weight on\n{mean_col}", fontsize=16)
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


 

def plot_cost_effectiveness_vs_equity(results_subset, equity_weights, budget_label, filename, y_axis_zero=False, capex_col =capex_per_net_ton_mean_col, capex_std = capex_per_net_ton_std_col):
    """Plot 2: Cost Effectiveness vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in results_subset['budget'].unique():
        subset = results_subset[results_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        # *** UPDATED COLUMN NAMES ***
        means = subset[capex_col].values
        stds = subset[capex_std].values
        
        label = f'£{budget_val}' if len(results_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, fmt='o-', markersize=10, yerr= stds, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost per Ton CO2 (£/kg)', fontsize=14, fontweight='bold')
    # ax.set_title(f'Cost Effectiveness vs Equity Weight\n{budget_label}', fontsize=16, fontweight='bold')
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


def plot_carbon_savings_vs_equity(results_subset, equity_weights, budget_label, filename, budget_order=None, y_axis_zero=False, co2_std =total_co2_saved_col_std):
    """
    Plot 1: Carbon Savings vs Equity Weight
    Args:
        budget_order (list): Optional list of budgets in the desired sort order.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Determine Order: Use provided order, or default to simple sort
    if budget_order is None:
        budgets_to_plot = sorted(results_subset['budget'].unique())
    else:
        # Filter to ensure we only try to plot budgets that actually exist in this subset
        available = set(results_subset['budget'].unique())
        budgets_to_plot = [b for b in budget_order if b in available]

    for budget_val in budgets_to_plot:
        subset = results_subset[results_subset['budget'] == budget_val]
        
        # Sort X-axis to ensure clean lines
        subset = subset.sort_values(by='equity_weight')
        
        weights = subset['equity_weight'].values
        # Note: Ensure 'total_co2_saved_col' is defined globally
        means = subset[total_co2_saved_col].values / 1e3
        stds = subset[co2_std].values / 1e3
        # Plot
        label = f'£{budget_val}' if len(budgets_to_plot) > 1 else None
        ax.errorbar(weights, means, fmt='o-', markersize=10, yerr = stds, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    
    if len(budgets_to_plot) > 1:
        # Legend now follows the exact order of the loop
        ax.legend(fontsize=12, title="Budget")
    
    if y_axis_zero:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(f"Saved: {filename}")


def plot_vulnerable_coverage_vs_equity(equity_subset, equity_weights, budget_label, filename, y_axis_zero=False):
    """Plot 3: Vulnerable Coverage vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in equity_subset['budget'].unique():
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        print('subset')
        print(subset.columns.tolist() )
        # *** These columns were simple strings, so NO change needed ***
        means = subset['high_risk_pct'].values * 100
        # stds = subset['high_risk_pct_std'].values * 100
        
        label = f'£{budget_val}' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means,fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    # ax.set_title(f'Vulnerable Population Coverage vs Equity Weight\n{budget_label}', 
                #  fontsize=16, fontweight='bold')
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
    # print(f"Saved: {filename}")

def plot_equity_concentration_vs_weight(equity_subset, equity_weights, budget_label, filename, y_axis_zero=False):
    """Plot 4: Equity Concentration vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in equity_subset['budget'].unique():
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        
        # *** These columns were simple strings, so NO change needed ***
        means = subset['equity_concentration'].values
        # stds = subset['equity_concentration_std'].values
        
        label = f'£{budget_val}' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means,  fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Equity Concentration Index', fontsize=14, fontweight='bold')
    # ax.set_title(f'Equity Concentration vs Equity Weight\n(lower = more equitable)\n{budget_label}', fontsize=16, fontweight='bold')
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


    # print(f"Saved: {filename}")

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
      'total_co2_saved', 'scenario', 'budget'
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
        )['mean_total_co2_saved'].sum().reset_index()
        
        # 3. Pivot for plotting
        # This creates a matrix where index=Personas, columns=Scenarios
        # fillna(0) handles cases where a persona exists in one scenario but not another
        plot_data = carbon_grouped.pivot(
            index='persona_name', 
            columns='scenario', 
            values='mean_total_co2_saved', 
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
        # ax.set_title(f'Distribution of Total Carbon Saved by Persona\n(Budget: £{budget})', 
                    #  fontsize=16, fontweight='bold')
        
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
        
        rl_filename = f'{base}_budget{budget}{ext}'
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # print(f"Saved plot to: {rl_filename}")
 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_metric_by_group(selected_projects_df, scenario_colors, filename, 
                         value_col='mean_total_co2_saved',
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
        
    # if title is None:
    #     stat_name = "Total" if metric_stat == 'sum' else metric_stat.capitalize()
    #     title = f"Distribution of {stat_name} {value_col.replace('_', ' ')} by Group"

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
        # groups = sorted(grouped_by_run['group'].unique())
        group_labels = [label_cleaner_map.get(g, str(g)) for g in groups]
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
  
        
        # Clean x-axis labels
        # clean_labels = [group_label_map.get(g, str(g)) for g in groups]
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=11, rotation=45, ha='right')
        
        ax.legend(fontsize=11, ncol=min(3, max(1, (n_scenarios + 2) // 3)))
        ax.grid(True, alpha=0.3, axis='y')
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        # 5. Save File
        base, ext = os.path.splitext(filename)
        if not ext: ext = ".png"
        
        # Add the stat type to the filename so they don't overwrite each other
        rl_filename = f'{base}_{metric_stat}_budget{budget}{ext}'
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        # print(f"Saved ({metric_stat}): {rl_filename}")

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
        budget_str = f"{budget_to_plot}"
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
        # ax.set_title(f'{title}\n(Mean ± SD across epistemic runs)', 
        #             fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=11)
        ax.legend(fontsize=11, ncol=min(3, (n_scenarios + 2) // 3))
        ax.grid(True, alpha=0.3, axis='y')
        
        if y_axis_zero:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        # print(f"Saved: {rl_filename}")
    
    # Main loop over budgets
    budgets = selected_projects_df['budget'].unique()
    for budget in budgets:
        plot_single_budget(selected_projects_df, budget)

def plot_pareto_front(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename, y_axis_zero=False, ycol= None ,ycol_std=None  ):
    """
    Plot 6: Pareto Front - Equity vs Carbon.
    
    1. Plots a combined chart with all budgets (saves as 'filename').
    2. Plots a separate chart for each individual budget (saves as 'filename_budget_X').
    """
    
    # --- PART 1: Plot Combined Pareto for all budgets (as before) ---
    fig_comb, ax_comb = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Get all unique budgets from the subset
    # all_budgets = sorted(results_subset['budget'].unique())
    all_budgets = sorted(results_subset['budget'].unique(), key=lambda x: float(x.replace('£', '').replace('M', '')) * 1_000_000  if isinstance(x, str) and 'm' in x.lower() else 0 )
    
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
            print('equity_row')
            print(equity_row)
            vuln_means.append(equity_row['high_risk_pct'] )
            # vuln_stds.append(equity_row['high_risk_pct_std'] )
            co2_means.append(results_row[ycol] / 1e3)
            co2_stds.append(results_row[ycol_std] / 1e3)
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Skip if this budget had no valid scenarios
        if not weights:
            continue
            
        label = f'£{budget_val}'
        
        # Plot points with error bars
        for i in range(len(vuln_means)): # Use vuln_means len as it matches collected data
            ax_comb.errorbar(vuln_means[i], co2_means[i], yerr= co2_stds[i] , 
                             fmt='o', markersize=12, capsize=5,
                             color=colors[i], 
                             label=f'EW={weights[i]}' if budget_val == all_budgets[0] else None) # Only label weights once

        # Draw connecting line
        sorted_indices = np.argsort(weights)
        vuln_means_sorted = [vuln_means[i] for i in sorted_indices]
        co2_means_sorted = [co2_means[i] for i in sorted_indices]
        ax_comb.plot(vuln_means_sorted, co2_means_sorted , '--', alpha=0.5, linewidth=2, label=f'Budget {label}')
    
    ax_comb.set_xlabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax_comb.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    # ax_comb.set_title(f'Equity-Carbon Tradeoff Curve\n{budget_label}', fontsize=16, fontweight='bold')
    
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
     
def plot_vulnerable_groups_coverage(equity_subset, filename, y_axis_zero=False):
    """Plot 7: Most Vulnerable Groups Coverage"""
    

    def plot_one_budget(budget_to_plot,  equity_subset, filename, y_axis_zero):
        rl_filename = f'{filename.split(".png")[0]}_budget{budget_to_plot}.png'
        equity_subset = equity_subset[equity_subset['budget'] == budget_to_plot]
        budget_label = f'Budget £{budget_to_plot}'
        scenarios = equity_subset['scenario'].unique()
        equity_weights = sorted(equity_subset['equity_weight'].unique())
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        x_pos = np.arange(len(scenarios))
        width = 0.35
        
        # *** These columns were simple strings, so NO change needed ***
        deprived_means = [equity_subset[equity_subset['scenario'] == s]['high_risk_pct'].iloc[0] * 100 
                        for s in scenarios]

        
        bars1 = ax.bar(x_pos - width/2, deprived_means, width, label='Deprived', 
                    color='#c0392b', alpha=0.8)
 
        
        ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
        # ax.set_title(f'Most Vulnerable Groups Coverage\n{budget_label}', fontsize=16, fontweight='bold')
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
            base_vuln = equity_subset[equity_subset['scenario'] == base_scenario]['high_risk_pct'].iloc[0] 
            base_co2 = results_subset[results_subset['scenario'] == base_scenario][total_co2_saved_col].iloc[0] / 1e3
            
            tradeoff_scenarios = []
            tradeoff_ratios = []
            tradeoff_labels = []
            
            for scenario in sorted_scenarios[1:]:
                equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
                results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
                
                # *** UPDATED COLUMN NAMES ***
                vuln_cov = equity_row['high_risk_pct'] * 100
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
            label = f'£{budget_val}'
            
            bars = ax.bar(x_pos + offset, tradeoff_ratios, bar_width, label=label, alpha=0.7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(tradeoff_labels)

    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable % Gain per kton CO2 Lost', fontsize=14, fontweight='bold')
    # ax.set_title(f'Equity-Carbon Tradeoff Efficiency\n(vs. EW=0 baseline)', fontsize=16, fontweight='bold')
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
    # print(f"Saved: {filename}")

def plot_radar_chart(results_subset, equity_subset,  filename, scenario_colors):
    """
    Plot 9: Multi-Metric Radar Chart
    Note: y_axis_zero has no effect here as the axis is normalized 0-1.
    """
    
    def plot_one_budget(results_subset, budget_to_plot,  equity_subset, filename,  scenario_colors):
        rl_filename = f'{filename.split(".png")[0]}_budget{budget_to_plot}.png'
        equity_subset = equity_subset[equity_subset['budget'] == budget_to_plot]
        results_subset = results_subset[results_subset['budget'] == budget_to_plot]
        budget_label = f'Budget £{budget_to_plot}'
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
        norm_vuln = equity_subset['high_risk_pct'].max()
        norm_equity = equity_subset['equity_concentration'].max()
        norm_buildings = results_subset['num_buildings_sum'].max()

        for scenario in scenarios:
            results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            
            # Normalize metrics (0-1, higher is better)
            # *** UPDATED COLUMN NAMES ***
            carbon_norm = results_row[total_co2_saved_col] / norm_co2
            cost_eff_norm = (1 / results_row[capex_per_net_ton_mean_col]) / norm_cost_eff
            vuln_norm = equity_row['high_risk_pct'] / norm_vuln
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
        # ax.set_title(f'Normalized Performance Comparison\n{budget_label}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11,
                ncol=1 if len(scenarios) <= 5 else 2)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(rl_filename, dpi=300, bbox_inches='tight')
        plt.close()
        # print(f"Saved: {rl_filename}")

    for bugdet in equity_subset['budget'].unique():
        plot_one_budget(results_subset, bugdet,  equity_subset, filename , scenario_colors)

def plot_pareto_front_flexi(
    results_subset, 
    equity_subset, 
    scenarios, 
    scenario_colors, 
    budget_label, 
    filename, 
    x_col='high_risk_pct', 
    y_col='total_co2_saved_col',
    y_col_std = '', 
    x_label='Vulnerable Coverage (%)',
    y_label='CO2 Saved (kton)',
    x_scaler=1.0,
    y_scaler=1e3, # Dividing by 1000 as per your original code
    y_axis_zero=False
        ):
    """
    Plot 6: Pareto Front - Flexible axes.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # --- PART 1: Plot Combined Pareto for all budgets ---
    fig_comb, ax_comb = plt.subplots(figsize=(10, 8))
    
    # all_budgets = sorted(results_subset['budget'].unique())
    all_budgets = sorted(results_subset['budget'].unique(), key=lambda x: float(x.replace('£', '').replace('M', '')) * 1_000_000  if isinstance(x, str) and 'm' in x.lower() else 0 )
    
    for budget_val in all_budgets:
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()
        
        x_vals = []
        y_vals = []
        weights = []
        colors = []
        y_stds=[]

        for scenario in scenarios_subset:
            if scenario not in equity_subset['scenario'].values or \
               scenario not in results_subset['scenario'].values:
                continue
            
            # Identify which dataframe holds the requested columns
            # Checks equity_subset first, then results_subset
            if x_col in equity_subset.columns:
                x_row_source = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            else:
                x_row_source = results_subset[results_subset['scenario'] == scenario].iloc[0]

            if y_col in results_subset.columns:
                y_row_source = results_subset[results_subset['scenario'] == scenario].iloc[0]
            else:
                y_row_source = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            
            x_vals.append(x_row_source[x_col] / x_scaler)
            y_vals.append(y_row_source[y_col] / y_scaler)
            y_stds.append(y_row_source[y_col_std] / y_scaler)
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        if not weights:
            continue
            
        label = f'£{budget_val}'
        
        # Plot points
        for i in range(len(x_vals)):
            ax_comb.errorbar(x_vals[i], y_vals[i], yerr = y_stds[i], 
                             fmt='o', markersize=12, capsize=5,
                             color=colors[i], 
                             label=f'EW={weights[i]}' if budget_val == all_budgets[0] else None)

        # Draw connecting line
        sorted_indices = np.argsort(weights)
        x_sorted = [x_vals[i] for i in sorted_indices]
        y_sorted = [y_vals[i] for i in sorted_indices]
        ax_comb.plot(x_sorted, y_sorted, '--', alpha=0.5, linewidth=2, label=f'Budget {label}')
    
    ax_comb.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax_comb.set_ylabel(y_label, fontsize=14, fontweight='bold')
    
    handles, labels = ax_comb.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax_comb.legend(by_label.values(), by_label.keys(), fontsize=11, ncol=1 if len(by_label) <= 6 else 2)
    
    ax_comb.grid(True, alpha=0.3)
    if y_axis_zero: ax_comb.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig_comb)
    
    # --- PART 2: Plot individual Pareto for each budget ---
    base_filename, file_extension = os.path.splitext(filename)
    
    for budget_val in all_budgets:
        fig_ind, ax_ind = plt.subplots(figsize=(10, 8))
        budget_results_subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_for_budget = budget_results_subset['scenario'].unique()

        x_vals_ind, y_vals_ind, weights_ind, colors_ind = [], [], [], []

        for scenario in scenarios_for_budget:
            if scenario not in equity_subset['scenario'].values:
                continue
            
            # Source data selection (same logic as above)
            x_src = equity_subset if x_col in equity_subset.columns else results_subset
            y_src = results_subset if y_col in results_subset.columns else equity_subset
            
            x_row = x_src[x_src['scenario'] == scenario].iloc[0]
            y_row = y_src[y_src['scenario'] == scenario].iloc[0]
            eq_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            
            x_vals_ind.append(x_row[x_col] / x_scaler)
            y_vals_ind.append(y_row[y_col] / y_scaler)
            weights_ind.append(eq_row['equity_weight'])
            colors_ind.append(scenario_colors[scenario])

        if not weights_ind:
            print('closign fig')
            plt.close(fig_ind)
            continue
            
        for i in range(len(x_vals_ind)):
            ax_ind.errorbar(x_vals_ind[i], y_vals_ind[i], fmt='o', markersize=12, color=colors_ind[i], label=f'EW={weights_ind[i]}')

        sorted_idx = np.argsort(weights_ind)
        ax_ind.plot([x_vals_ind[i] for i in sorted_idx], [y_vals_ind[i] for i in sorted_idx], '--', alpha=0.5, linewidth=2, label='Tradeoff Curve')
        
        ax_ind.set_xlabel(x_label, fontsize=14, fontweight='bold')
        ax_ind.set_ylabel(y_label, fontsize=14, fontweight='bold')
        
        handles_ind, labels_ind = ax_ind.get_legend_handles_labels()
        by_label_ind = dict(zip(labels_ind, handles_ind))
        ax_ind.legend(by_label_ind.values(), by_label_ind.keys(), fontsize=11)
        
        ax_ind.grid(True, alpha=0.3)
        if y_axis_zero: ax_ind.set_ylim(bottom=0)
        
        new_filename = f"{base_filename}_budget_{budget_val}{file_extension}"
        plt.tight_layout()
        plt.savefig(new_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)


 

def plot_pareto_retrofit_carbon_by_costeff(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename, y_axis_zero=False, capex_std = capex_per_net_ton_std_col) :
    """
    Plot: Pareto Front - Buildings Retrofitted vs Cost-Effectiveness
    
    Updates:
    1. Generates a SEPARATE plot file for each budget (original functionality).
    2. Generates a COMBINED plot file with all budgets overlayed (new functionality).
    """
    
    # Get the base filename and extension to create unique names
    base_name, extension = os.path.splitext(filename)
    
    # --- NEW: Initialize the Combined "Master" Figure ---
    fig_all, ax_all = plt.subplots(figsize=(10, 8))
    
    # Get sorted unique budgets
    # unique_budgets = sorted(results_subset['budget'].unique())
    unique_budgets = sorted(results_subset['budget'].unique(), key=lambda x: float(x.replace('£', '').replace('M', '')) * 1_000_000  if isinstance(x, str) and 'm' in x.lower() else 0 )
    # Loop over each budget and create a unique plot
    for budget_val in unique_budgets:
        
        # Create a new figure for the individual budget
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # --- Data collection for this specific budget ---
        subset = results_subset[results_subset['budget'] == budget_val]
        scenarios_subset = subset['scenario'].unique()
        
        buildings_means = []
        cost_eff_means = []
        cost_eff_stds=[] 
        weights = []
        colors = []

        for scenario in scenarios_subset:
            # Ensure scenario exists in both subsets
            if scenario not in equity_subset['scenario'].values or scenario not in results_subset['scenario'].values:
                print(f"Warning: Scenario {scenario} not found. Skipping.")
                continue
                
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            results_row = subset[subset['scenario'] == scenario].iloc[0]
            
            buildings_means.append(results_row['num_buildings_sum'])
            
            cost_eff_means.append(results_row[capex_per_net_ton_mean_col])
            cost_eff_stds.append(results_row[capex_std])
            weights.append(equity_row['equity_weight'])
            colors.append(scenario_colors[scenario])
        
        # Sort data by weight to draw clean lines
        sorted_indices = np.argsort(weights)
        buildings_means_sorted = [buildings_means[i] for i in sorted_indices]
        cost_eff_means_sorted = [cost_eff_means[i] for i in sorted_indices]
        cost_eff_stds_sorted = [cost_eff_stds[i] for i in sorted_indices]
 
               
        for i in range(len(scenarios_subset)):
            ax_all.errorbar(buildings_means_sorted[i], cost_eff_means_sorted[i], yerr= cost_eff_stds_sorted[i], 
                                    fmt='o', markersize=12, capsize=5,
                                    color=colors[i], 
                                     label=f'EW={weights[i]}' ) 

        
        ax_all.plot(buildings_means_sorted, cost_eff_means_sorted , '--', alpha=0.5, linewidth=2, label=f'Budget {budget_val}')


        # # Plot the line for this budget on the master plot
        # # We label the line with the Budget Value so the legend identifies the curve
        # ax_all.errorbar(buildings_means_sorted, cost_eff_means_sorted, yerr=cost_eff_stds_sorted, 
        #             fmt = 'o', # Line with markers
        #             alpha=0.7, 
        #             linewidth=2, 
        #             label=f'Budget £{budget_val}') 
        
        # # Optional: If you want the specific scenario colors on the master plot points
        # # we can overlay the scatter points on top of the line
        # for i in range(len(scenarios_subset)):
        #     ax_all.scatter(buildings_means[i], cost_eff_means[i], color=colors[i], s=50, zorder=3, label=f'EW={weights[i]}' )

    
        # ax.legend(by_label.values(), by_label.keys(), fontsize=11, ncol=1 if len(by_label) <= 6 else 2)
        
    # ---------------------------------------------------------
    # Finalize and Save the COMBINED Plot
    # ---------------------------------------------------------
    ax_all.set_xlabel('Number of Buildings', fontsize=14, fontweight='bold')
    ax_all.set_ylabel('Cost Effectiveness (£ / net ton CO2)', fontsize=14, fontweight='bold')
    
    
    ax_all.grid(True, alpha=0.3)
    if y_axis_zero:
        ax_all.set_ylim(bottom=0)
    
    # Add legend (distinguishing budgets)
    # ax_all.legend(fontsize=12, loc='best')
    handles, labels = ax_all.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    if handles:
        by_label = dict(zip(labels, handles))
        ax_all.legend(by_label.values(), by_label.keys(), fontsize=11, ncol=1 if len(by_label) <= 6 else 2)

    combined_filename = f"{base_name}_cost_eff_ALL_BUDGETS{extension}"
    plt.figure(fig_all.number) # Set context to figure
    plt.tight_layout()
    fig_all.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_all)
    print(f"Saved Combined: {combined_filename}")
 