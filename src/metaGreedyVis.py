import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Define the columns you want to sum up and compare
METRICS = [
    'total_capex', 
    'weighted_capex_per_net_ton', 
    'total_co2_saved'
]

def aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, equity_factors, milion_factor=1000000):
    """
    Loops through file structure, aggregates totals, and returns a Meta DataFrame.
    """
    print('starting agg scenario ')
    meta_results = []
    
    print("Starting Aggregation...")

    for prob_loft in loft_probs:
        for budget in budgets:
            million_budget = budget / milion_factor
            for equity_factor in equity_factors:
                
                # 1. Reconstruct the directory path exactly as in your snippet
                folder_name = f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}'
                output_dir = os.path.join(greedy_runs_folder, folder_name)
                
                # 2. Define file paths
                selected_path = os.path.join(output_dir, 'selected_projects.csv')     # Optimization (Smart)
                epc_random_path = os.path.join(output_dir, 'epc_random_selection.csv') # Baseline (Random/EPC)
                
                # 3. Check if files exist before processing
                if os.path.exists(selected_path) and os.path.exists(epc_random_path):
                    try:
                        # Load Data
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
                        
                        # Calculate Totals for each metric
                        for metric in METRICS:
                            if metric in df_opt.columns and metric in df_epc.columns:
                                val_opt = df_opt[metric].sum()
                                val_epc = df_epc[metric].sum()
                                
                                row[f'{metric}_OPT'] = val_opt
                                row[f'{metric}_EPC'] = val_epc
                                row[f'{metric}_diff'] = val_opt - val_epc
                                row[f'{metric}_pct_gain'] = ((val_opt - val_epc) / val_epc * 100) if val_epc != 0 else 0
                        
                        meta_results.append(row)
                        print(f"Processed: {folder_name}")
                        
                    except Exception as e:
                        print(f"Error reading {folder_name}: {e}")
                else:
                    print(f"Skipping missing: {folder_name}")

    # Create final DataFrame
    meta_df = pd.DataFrame(meta_results)
    print('meta df made and returning')
    return meta_df


def plot_meta_comparisons(meta_df, output_dir=None):
    """
    Plots the aggregated trends: Optimization vs EPC across budgets.
    """
    print('starting meta com[aisoron]')
    sns.set_style("whitegrid")
    
    # Ensure rows are sorted by budget for clean lines
    meta_df = meta_df.sort_values('budget_m')
    
    # We might have different loft/equity groups. 
    # For simplicity, let's plot for each unique combination of loft/equity.
    groups = meta_df.groupby(['loft_prob', 'equity_factor'])

    for (loft, equity), group_df in groups:
        
        # Define suffix for title/filename
        scenario_suffix = f"Loft {loft} - Equity {equity}"
        
        for metric in METRICS:
            # Check if columns exist
            if f'{metric}_OPT' not in group_df.columns: 
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            # X-Axis: Budget in Millions
            x = group_df['budget_m']
            
            # --- Top Plot: Absolute Values ---
            ax1.plot(x, group_df[f'{metric}_OPT'], marker='o', linewidth=3, label='Optimised (Smart)', color='#2ecc71')
            ax1.plot(x, group_df[f'{metric}_EPC'], marker='s', linewidth=3, label='Random (EPC)', color='#e74c3c', linestyle='--')
            
            # Labels and Formatting
            metric_title = metric.replace('_', ' ').title().replace('Kwh', 'kWh').replace('Kg', 'kg')
            ax1.set_ylabel(metric_title, fontsize=12, fontweight='bold')
            ax1.set_title(f'Performance Comparison: {metric_title}\n({scenario_suffix})', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Add text labels on the Smart points
            for x_val, y_val in zip(x, group_df[f'{metric}_OPT']):
                ax1.text(x_val, y_val, f'{y_val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # --- Bottom Plot: The "Lift" (Difference) ---
            # Calculate % improvement
            pct_gain = group_df[f'{metric}_pct_gain']
            
            bars = ax2.bar(x.astype(str), pct_gain, color='#3498db', alpha=0.8, edgecolor='black', width=0.5)
            
            ax2.set_ylabel('% Improvement over EPC', fontsize=12)
            ax2.set_xlabel('Budget (Millions)', fontsize=12, fontweight='bold')
            ax2.axhline(0, color='black', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='y')

            # Add labels to bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'+{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            
            # Save or Show
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fname = f"meta_trend_{metric}_loft{loft}_eq{equity}.png"
                plt.savefig(os.path.join(output_dir, fname), dpi=300)
                print(f"Saved plot: {fname}")
            else:
                plt.show()
            plt.close()





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
METRICS = [
    'total_capex', 
    'weighted_capex_per_net_ton', 
    'total_co2_saved'
]

# ==========================================
# PART 1: Diagnostic Plotting Functions
# ==========================================

def generate_scenario_diagnostics(df_opt, df_epc, output_dir):
    """
    Runs all diagnostic plots for a single scenario folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. HEATMAP: Socio-Persona vs Energy Rating (Targeting Bias)
    plot_targeting_heatmap(df_opt, df_epc, output_dir)
    
    # 2. PLOT A: Gas Percentile (The "Fuel Guzzler" Validation)
    plot_gas_validation(df_opt, df_epc, output_dir)
    
    # 3. PLOT B: Social Equity (Who gets the CapEx?)
    plot_social_equity(df_opt, df_epc, output_dir)
    
    # 4. PLOT C: Worst-First (EPC Band Transition)
    plot_epc_bands(df_opt, df_epc, output_dir)
    
    # 5. PLOT D: Cost-Effectiveness Bubble Plot (Using weighted_capex_per_net_ton)
    plot_efficiency_scatter(df_opt, df_epc, output_dir)

def plot_targeting_heatmap(df_opt, df_epc, output_dir):
    """Heatmap showing where Smart Optimization focuses compared to Random."""
    opt_counts = df_opt.groupby(['meta_socio_persona', 'CURRENT_ENERGY_RATING']).size().unstack(fill_value=0)
    epc_counts = df_epc.groupby(['meta_socio_persona', 'CURRENT_ENERGY_RATING']).size().unstack(fill_value=0)
    
    # Align indexes and columns
    all_personas = sorted(set(opt_counts.index) | set(epc_counts.index))
    all_ratings = sorted(set(opt_counts.columns) | set(epc_counts.columns))
    opt_counts = opt_counts.reindex(index=all_personas, columns=all_ratings, fill_value=0)
    epc_counts = epc_counts.reindex(index=all_personas, columns=all_ratings, fill_value=0)
    
    # Difference (Smart - Random)
    diff_matrix = opt_counts - epc_counts

    plt.figure(figsize=(12, 8))
    sns.heatmap(diff_matrix, annot=True, fmt='d', cmap='RdBu_r', center=0, linewidths=.5)
    plt.title('Selection Bias: Smart vs Random\n(Red = Smart Targets More)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_heatmap_targeting.png", dpi=200)
    plt.close()

def plot_gas_validation(df_opt, df_epc, output_dir):
    """KDE showing gas usage of selected homes."""
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(data=df_opt, x='avg_gas_percentile', fill=True, label='Smart Opt', color='green', alpha=0.3, linewidth=2)
    sns.kdeplot(data=df_epc, x='avg_gas_percentile', fill=True, label='Random (EPC)', color='gray', alpha=0.3, linewidth=2, linestyle='--')
    
    plt.title('Validation: Gas Usage of Selected Homes', fontsize=14, fontweight='bold')
    plt.xlabel('Gas Usage Percentile (0-100)', fontsize=12)
    plt.ylabel('Density of Selection', fontsize=12)
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_gas_validation_kde.png", dpi=200)
    plt.close()

def plot_social_equity(df_opt, df_epc, output_dir):
    """Stacked Bar of Total CapEx by Persona."""
    # Sum total_capex by persona (convert to Millions for readability)
    s_opt = df_opt.groupby('meta_socio_persona')['total_capex'].sum() / 1e6 
    s_epc = df_epc.groupby('meta_socio_persona')['total_capex'].sum() / 1e6
    
    df_compare = pd.DataFrame({'Smart Opt': s_opt, 'Random': s_epc}).fillna(0)
    
    df_compare.plot(kind='bar', figsize=(12, 6), color=['#2ecc71', '#95a5a6'], edgecolor='black', alpha=0.8)
    
    plt.title('Social Equity: Total CapEx Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Total CapEx (£ Millions)', fontsize=12)
    plt.xlabel('Socio Persona', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_social_equity_capex.png", dpi=200)
    plt.close()

def plot_epc_bands(df_opt, df_epc, output_dir):
    """Bar chart comparing EPC bands selected."""
    band_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    c_opt = df_opt['CURRENT_ENERGY_RATING'].value_counts()
    c_epc = df_epc['CURRENT_ENERGY_RATING'].value_counts()
    
    df_compare = pd.DataFrame({'Smart Opt': c_opt, 'Random': c_epc}).reindex(band_order).fillna(0)
    
    df_compare.plot(kind='bar', width=0.8, figsize=(10, 6), color=['#2ecc71', '#95a5a6'], edgecolor='black')
    
    plt.title('"Worst First" Check: Selected EPC Bands', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Homes Treated', fontsize=12)
    plt.xlabel('Starting EPC Band', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_epc_band_transition.png", dpi=200)
    plt.close()

def plot_efficiency_scatter(df_opt, df_epc, output_dir):
    """Scatter of Cost Efficiency (weighted_capex_per_net_ton) vs Gas Usage."""
    plt.figure(figsize=(12, 7))
    
    # We plot the Smart Selection
    # Y-Axis = weighted_capex_per_net_ton (Lower is better)
    # X-Axis = avg_gas_percentile (Higher is better targeting)
    
    sns.scatterplot(
        data=df_opt, 
        x='avg_gas_percentile', 
        y='weighted_capex_per_net_ton', 
        hue='meta_socio_persona',
        size='total_capex',
        sizes=(20, 200),
        alpha=0.6,
        palette='viridis'
    )
    
    plt.title('Sweet Spot Analysis: Gas Usage vs Efficiency (Smart Selection)', fontsize=14, fontweight='bold')
    plt.xlabel('Gas Percentile (Higher = More Usage)', fontsize=12)
    plt.ylabel('Weighted CapEx Per Net Ton (£/tCO2) (Lower is Better)', fontsize=12)
    
    # Add a benchmark line for Random/EPC median efficiency
    random_median = df_epc['weighted_capex_per_net_ton'].median()
    if pd.notna(random_median):
        plt.axhline(random_median, color='red', linestyle='--', label=f'Median Random Efficiency (£{random_median:,.0f}/t)')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f'Saving to {output_dir}')
    plt.savefig(f"{output_dir}/5_efficiency_scatter.png", dpi=200)
    plt.close()

# ==========================================
# PART 2: Meta-Analysis Plotting (Pareto)
# ==========================================

def plot_pareto_frontier(meta_df, output_dir):
    """Plots Pareto Frontier: Total CapEx vs Total CO2 Saved."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Sort for cleaner lines
    df_sorted = meta_df.sort_values(by='total_capex_OPT')
    groups = df_sorted.groupby(['loft_prob', 'equity_factor'])

    for (loft, equity), group_df in groups:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # X-Axis: Investment (Millions)
        x_opt = group_df['total_capex_OPT'] / 1e6
        x_epc = group_df['total_capex_EPC'] / 1e6
        
        # Y-Axis: Benefit (Total CO2 Saved)
        y_opt = group_df['total_co2_saved_OPT']
        y_epc = group_df['total_co2_saved_EPC']
        
        # Plot Curves
        ax.plot(x_opt, y_opt, marker='o', markersize=8, linewidth=3, 
                color='#2ecc71', label='Optimised Strategy (Pareto Front)')
        ax.plot(x_epc, y_epc, marker='s', markersize=8, linewidth=2, linestyle='--', 
                color='#95a5a6', label='Standard EPC Strategy')

        # Fill Gap
        ax.fill_between(x_opt, y_opt, y_epc, color='#2ecc71', alpha=0.1, label='Optimization Value Add')

        # Formatting
        ax.set_title(f'Pareto Frontier: Cost vs Carbon Impact\nLoft {loft} - Equity {equity}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total CapEx (£ Millions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total CO2 Saved (kg)', fontsize=12, fontweight='bold')
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        print(f'saving to {output_dir}')
        plt.savefig(f"{output_dir}/pareto_co2_loft{loft}_eq{equity}.png", dpi=300)
        plt.close()

# ==========================================
# PART 3: Main Processing Loop
# ==========================================

def run_full_analysis_pipeline(greedy_runs_folder, budgets, loft_probs, equity_factors):
    meta_results = []
    
    print("Starting Analysis Pipeline...")

    for prob_loft in loft_probs:
        for budget in budgets:
            million_budget = budget / 1_000_000
            for equity_factor in equity_factors:
                
                # 1. Setup Paths
                folder_name = f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}'
                scenario_dir = os.path.join(greedy_runs_folder, folder_name)
                
                selected_path = os.path.join(scenario_dir, 'selected_projects.csv')
                epc_random_path = os.path.join(scenario_dir, 'epc_random_selection.csv')
                
                # 2. Check & Load
                if os.path.exists(selected_path) and os.path.exists(epc_random_path):
                    print(f"--> Processing: {folder_name}")
                    
                    try:
                        df_opt = pd.read_csv(selected_path)
                        df_epc = pd.read_csv(epc_random_path)
                        
                        # --- A. GENERATE PER-SCENARIO DIAGNOSTICS ---
                        plots_output_dir = os.path.join(scenario_dir, 'diagnostic_plots')
                        generate_scenario_diagnostics(df_opt, df_epc, plots_output_dir)
                        
                        # --- B. AGGREGATE META DATA ---
                        row = {
                            'budget_m': million_budget,
                            'loft_prob': prob_loft,
                            'equity_factor': equity_factor,
                        }
                        
                        # Dynamically add all metrics with suffixes
                        for m in METRICS:
                            if m in df_opt.columns:
                                row[f'{m}_OPT'] = df_opt[m].sum()
                                row[f'{m}_EPC'] = df_epc[m].sum()
                        
                        # Special handling for weighted metrics (Summing them usually implies calculating a weighted average, 
                        # but for meta-plotting total cost vs total co2, we rely on the totals. 
                        # If you need an average efficiency for the whole scenario, we calculate it here:)
                        if 'total_co2_saved' in df_opt.columns and 'total_capex' in df_opt.columns:
                             # Re-calculate overall scenario efficiency (Cost / CO2)
                            row['efficiency_OPT'] = row['total_capex_OPT'] / row['total_co2_saved_OPT'] if row['total_co2_saved_OPT'] > 0 else 0
                            row['efficiency_EPC'] = row['total_capex_EPC'] / row['total_co2_saved_EPC'] if row['total_co2_saved_EPC'] > 0 else 0

                        meta_results.append(row)
                        
                    except Exception as e:
                        print(f"    Error in {folder_name}: {e}")
                else:
                    print(f"    Skipping (Files not found): {folder_name}")

    # --- C. GENERATE META PLOTS ---
    if meta_results:
        print("--> Generating Meta-Analysis (Pareto)...")
        meta_df = pd.DataFrame(meta_results)
        
        # Save Meta CSV
        meta_df.to_csv(os.path.join(greedy_runs_folder, "meta_optimization_summary.csv"), index=False)
        
        # Run the Pareto Plotter
        plot_pareto_frontier(
            meta_df, 
            output_dir=os.path.join(greedy_runs_folder, "meta_plots")
        )
        print("Done!")
    else:
        print("No results found to aggregate.")

# --- Execute ---
