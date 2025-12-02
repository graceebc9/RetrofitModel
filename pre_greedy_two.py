import pandas as pd
import numpy as np
import glob
from pathlib import Path
from src.RetrofitAnalysisUtils import load_data, prepare_data_for_postanalysis
import os 
import seaborn as sns 
import matplotlib.pyplot as plt 

# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v7/NE'
OUTPUT_BASE_DIR = 'optimized_priorities'

RISK_QUANTILE = 0.15 

# SYMMETRIC CAP
# We ignore any single run outside this window.
# i.e., Ignore if Cost > £200k OR Cost < -£200k
ABS_COST_CAP = 200000.0

# Simulation Parameters
YEARS = 5
N_SIMULATIONS = 5000
GAS_CARBON_FACTOR = 0.18       
ELEC_CARBON_FACTOR = 0.19338

METRICS = {
    'hp_only': 'heat_pump_only_cost_per_total_energy_ton_heat_pump_only_mean',
    'loft':    'loft_installation_cost_per_total_energy_ton_loft_installation_mean',
    'wall':    'wall_installation_cost_per_total_energy_ton_wall_installation_mean',
    'heat_wall': 'joint_heat_wall_decay_cost_per_total_energy_ton_joint_heat_wall_decay_mean'
}

SCENARIO_LIST = [
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'wall_installation', 
    'join_heat_ins_decay', 
    'heat_pump_only', 
    'loft_installation'
]

def plot_comparison(df, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Filter for cleaner plot (zoom to 95th percentile of costs)
    if not df.empty:
        plot_cap = df['cost_robust'].quantile(0.95)
        plot_df = df[df['cost_robust'] < plot_cap]
        
        # Scatter Plot
        sns.scatterplot(
            data=plot_df,
            x='cost_mean',
            y='cost_robust',
            alpha=0.4,
            s=15,
            color='blue',
            label='Buildings'
        )

        # Reference Line (y=x)
        max_val = plot_df['cost_robust'].max()
        plt.plot([0, max_val], [0, max_val], 'r--', label='Zero Risk Line (Mean == Robust)')
        
        plt.title(f'Risk Analysis: Mean Cost vs Robust Cost (P{int(RISK_QUANTILE*100)})\nPoints higher above the red line have higher uncertainty')
        plt.xlabel('Mean Cost (£/Ton) - [Neutral View]')
        plt.ylabel('Robust Cost (£/Ton) - [Conservative View]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f"{output_dir}/risk_comparison_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close() 
        print(f"Saved comparison plot to: {plot_path}")

def prepare_data_for_postanalysis_greedy(filepath):
    """
    Wrapper for your actual data preparation function
    """
    pre_df = pd.read_csv(filepath)
    proc_df = prepare_data_for_postanalysis(
            pre_df, 
            SCENARIO_LIST, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
    return proc_df

def generate_best_intervention_summary(collected_dfs, output_dir):
    """
    Takes a list of dataframes (one per scenario), combines them,
    and finds the single most cost-effective intervention per building.
    """
    print("\n--- Generating 'Best Intervention' Summary ---")
    
    if not collected_dfs:
        print("No data available to summarize.")
        return

    # Combine all scenarios into one large dataframe
    master_df = pd.concat(collected_dfs, ignore_index=True)
    
    # Sort: Primary key is UPN, Secondary is Cost (Ascending)
    # This puts the cheapest intervention for every building at the top of its group
    master_df.sort_values(by=['upn', 'cost_robust'], ascending=[True, True], inplace=True)
    
    # Group by UPN and take the first record (which is the cheapest due to sort)
    best_df = master_df.groupby('upn').first().reset_index()
    
    # Select specific columns requested
    output_df = best_df[['upn', 'cost_robust', 'intervention']]
    
    # Save
    save_path = f"{output_dir}/best_intervention_per_building.csv"
    output_df.to_csv(save_path, index=False)
    
    print(f"Summary generated for {len(output_df)} buildings.")
    print("Top 5 Example Rows:")
    print(output_df.head(5).to_string(index=False))
    print(f"Saved Global Summary to: {save_path}")

def process_all_scenarios(metrics_dict, scenario_list, output_base_dir):
    files = glob.glob(f"{LOG_DIR}/*.csv")
    files = files[0:5] 
    
    # Structure: {'hp_only': [df1, df2...], 'loft': [df1, df2...] }
    data_storage = {key: [] for key in metrics_dict.keys()}

    print(f"Found {len(files)} files. Starting batch processing...")

    # --- PHASE 1: LOAD AND PROCESS FILES ---
    for f in files:
        print(f"Processing: {f}")
        
        # Load Data ONCE per file
        full_df = prepare_data_for_postanalysis_greedy(f)
        
        # Iterate over all metrics 
        for metric_name, metric_col in metrics_dict.items():
            
            sub_df = full_df[['upn', metric_col]].copy()
            
            # Remove Monsters
            mask_monster = (sub_df[metric_col].abs() > ABS_COST_CAP) | (sub_df[metric_col].isna())
            if mask_monster.sum() > 0:
                sub_df.loc[mask_monster, metric_col] = np.nan
            
            # Calculate Robust Metrics
            agg = sub_df.groupby('upn')[metric_col].agg([
                ('valid_runs', 'count'),
                ('raw_mean', 'mean'),
                ('raw_robust_score', lambda x: x.quantile(RISK_QUANTILE)) 
            ]).reset_index()
            
            # Stability Filter
            agg = agg[agg['valid_runs'] > 50]
            
            data_storage[metric_name].append(agg)

    # --- PHASE 2: AGGREGATE, RANK AND SAVE PER SCENARIO ---
    print("\nBatch processing complete. Generating reports...")

    # We will collect the finished dataframes here to pass to the summary function
    all_finished_dfs = []

    for metric_name, agg_list in data_storage.items():
        if not agg_list:
            print(f"No valid data found for {metric_name}")
            continue

        current_output_dir = f'{output_base_dir}/{metric_name}'
        os.makedirs(current_output_dir, exist_ok=True)
        
        print(f"\n--- Generating Results for: {metric_name} ---")
        
        # Combine chunks
        final_df = pd.concat(agg_list, ignore_index=True)
        
        # --- SIGN FLIP & CALCULATIONS ---
        final_df['optimizer_cost'] = final_df['raw_robust_score'] * -1
        final_df['cost_robust'] = final_df['raw_robust_score'] * -1
        final_df['cost_mean'] = final_df['raw_mean'] * -1
        final_df['risk_premium'] = final_df['cost_robust'] - final_df['cost_mean']
        
        # Add Intervention Name (Crucial for the summary step)
        final_df['intervention'] = metric_name

        # Sort and Rank
        final_df = final_df.sort_values(by='optimizer_cost', ascending=True)
        final_df['priority_rank'] = range(1, len(final_df) + 1)
        final_df['rank'] = range(1, len(final_df) + 1)
        
        # Save individual scenario lists
        final_df.to_csv(f"{current_output_dir}/flipped_priority_list.csv", index=False)
        final_df.to_csv(f"{current_output_dir}/robust_priority_list_with_mean.csv", index=False)
        plot_comparison(final_df, current_output_dir)

        # Add to collection for the global summary
        all_finished_dfs.append(final_df)

    # --- PHASE 3: GENERATE BEST INTERVENTION SUMMARY ---
    generate_best_intervention_summary(all_finished_dfs, output_base_dir)

if __name__ == "__main__":
    process_all_scenarios(METRICS, SCENARIO_LIST, OUTPUT_BASE_DIR)