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
OUTPUT_DIR = 'optimized_priorities'


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


def plot_comparison(df):
    plt.figure(figsize=(10, 6))
    
    # Filter for cleaner plot (zoom to 95th percentile of costs)
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
    # If a point is ON this line, Robust == Mean (Zero Risk)
    max_val = plot_df['cost_robust'].max()
    plt.plot([0, max_val], [0, max_val], 'r--', label='Zero Risk Line (Mean == Robust)')
    
    plt.title(f'Risk Analysis: Mean Cost vs Robust Cost (P{int(RISK_QUANTILE*100)})\nPoints higher above the red line have higher uncertainty')
    plt.xlabel('Mean Cost (£/Ton) - [Neutral View]')
    plt.ylabel('Robust Cost (£/Ton) - [Conservative View]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f"{OUTPUT_DIR}/risk_comparison_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to: {plot_path}")


    

def prepare_data_for_postanalysis_greedy(filepath ):
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



def generate_flipped_priority_list(METRIC_COL, metric_name, scenario_list , OUTPUT_DIR):
    files = glob.glob(f"{LOG_DIR}/*.csv")
    files = files[0:5]
    all_aggs = []
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"Processing {len(files)} files...")
    print(f"Metric: {METRIC_COL}")
    print(f"Logic: Negative Raw Values -> P{int(RISK_QUANTILE*100)} (Conservative) -> Sign Flip -> Sort Min")

    for f in files:
        print(f) 
    
        # Load Data
        df = prepare_data_for_postanalysis_greedy(f  , ) 
        
        # --- STEP 1: REMOVE MONSTERS ---
        # If Raw Value is < -200,000 (i.e., Cost is > 200k/ton), it's an outlier/error.
        # We also filter positive numbers (which would imply negative savings? Error?)
        mask_monster = (df[METRIC_COL].abs() > ABS_COST_CAP) | (df[METRIC_COL].isna())
        
        if mask_monster.sum() > 0:
            df.loc[mask_monster, METRIC_COL] = np.nan
        
        # --- STEP 2: CALCULATE ROBUST METRICS ---
        agg = df.groupby('upn')[METRIC_COL].agg([
            ('valid_runs', 'count'),
            ('raw_mean', 'mean'),
            # The Critical Step: Grab the conservative (more negative) end of the spread
            ('raw_robust_score', lambda x: x.quantile(RISK_QUANTILE)) 
        ]).reset_index()
        
   
        # --- STEP 3: STABILITY FILTER ---
        agg = agg[agg['valid_runs'] > 50]
        
        all_aggs.append(agg)
            
 

    if not all_aggs:
        print("No valid data found.")
        return

    # Combine
    final_df = pd.concat(all_aggs, ignore_index=True)
    
    # --- STEP 4: SIGN FLIP & SORT ---
    
    # Create the "Optimizer Cost" (Positive £/Ton)
    # -500 (Raw) -> 500 (Optimizer Cost)
    final_df['optimizer_cost'] = final_df['raw_robust_score'] * -1
    
    final_df['cost_robust'] = final_df['raw_robust_score'] * -1
    final_df['cost_mean'] = final_df['raw_mean'] * -1
    
    # Calculate the "Risk Premium" (How much extra did we penalize this building?)
    final_df['risk_premium'] = final_df['cost_robust'] - final_df['cost_mean']
    
    # Sort Ascending: Smallest Positive Cost (Cheapest) -> Top Priority
    final_df = final_df.sort_values(by='optimizer_cost', ascending=True)
    
    # Add Rank
    final_df['priority_rank'] = range(1, len(final_df) + 1)
    # Add Rank
    final_df['rank'] = range(1, len(final_df) + 1)
    
    # --- REPORTING ---
    print("\n=== TOP 5 PRIORITY BUILDINGS (Most Cost Effective) ===")
    # Expected: Low Positive 'optimizer_cost' (e.g., 50, 100)
    print(final_df[['priority_rank', 'upn', 'optimizer_cost', 'raw_robust_score']].head(5).to_string(index=False))
    
    print("\n=== BOTTOM 5 BUILDINGS (Most Expensive/Risky) ===")
    # Expected: High Positive 'optimizer_cost' (e.g., 20000, 50000)
    print(final_df[['priority_rank', 'upn', 'optimizer_cost', 'raw_robust_score']].tail(5).to_string(index=False))
    
    # Save
    save_path = f"{OUTPUT_DIR}/flipped_priority_list.csv"
    final_df.to_csv(save_path, index=False)
    print(f"\nSaved Priority List: {save_path}")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    # Display the comparison
    cols_to_show = ['rank', 'upn', 'cost_robust', 'cost_mean', 'risk_premium']
    
    print("\n=== TOP 5 CANDIDATES (Robust Sort) ===")
    print("Notice: 'cost_robust' is used for ranking, but 'cost_mean' is shown for context.")
    print(final_df[cols_to_show].head(5).to_string(index=False))
    
    print("\n=== BOTTOM 5 CANDIDATES (Risky/Expensive) ===")
    print(final_df[cols_to_show].tail(5).to_string(index=False))
    
    # Save CSV
    save_path = f"{OUTPUT_DIR}/robust_priority_list_with_mean.csv"
    final_df.to_csv(save_path, index=False)
    print(f"\nSaved list to: {save_path}")

    # ========================================================================
    # VISUALIZATION: THE "RISK CONE"
    # ========================================================================
    plot_comparison(final_df)
    return final_df


if __name__ == "__main__":
 
    
    
    for metric_name, METRIC_COL in METRICS.items(): 
        OUTPUT_DIR = f'optimized_priorities/{metric_name}'
        os.makedirs(OUTPUT_DIR , exist_ok=True) 
        generate_flipped_priority_list(METRIC_COL, metric_name, SCENARIO_LIST , OUTPUT_DIR)