import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from src.RetrofitAnalysisUtils import load_data , prepare_data_for_postanalysis
# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# UPDATE THIS PATH to where your CSV files are located
# e.g., '/home/gb669/rds/hpc-work/.../NE/' or local path
LOG_DIR =  '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v7/NE'


YEARS = 5
N_SIMULATIONS = 5000

GAS_CARBON_FACTOR=0.18      
ELEC_CARBON_FACTOR=0.19338  

 


# The metric columns you want to analyze (from your previous snippet)
METRICS = {
    'hp_only': 'heat_pump_only_cost_per_total_energy_ton_heat_pump_only_mean',
    'loft':    'loft_installation_cost_per_total_energy_ton_loft_installation_mean',
    'wall':    'wall_installation_cost_per_total_energy_ton_wall_installation_mean',
    'heat_wall': 'joint_heat_wall_decay_cost_per_total_energy_ton_joint_heat_wall_decay_mean'
}

# THRESHOLDS FOR ANALYSIS
STRICT_CV_CUTOFF = 0.5      # The old method
HYBRID_SD_FLOOR = 500.0     # The "Safe absolute risk" (e.g. £500)
HYBRID_CV_RELAXED = 1.5     # The relaxed CV for walls/lofts if needed

# ============================================================================
# 2. ANALYSIS ENGINE
# ============================================================================



def prepare_data_for_postanalysis_greedy(pre_df, scenario_list, years, gas_carbon_factor, elec_carbon_factor):
    """
    Placeholder for your actual data preparation function
    Replace with your implementation
    """
    proc_df = prepare_data_for_postanalysis(
            pre_df, 
            scenario_list, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
    return proc_df
    
    
def analyze_log_file(filepath, metric_alias, metric_col):
    """
    Reads a single log file, groups by UPN, and calculates stats.
    """
    print(f"Reading {Path(filepath).name}...")
    
 
    pre_df = pd.read_csv(filepath)
 
    
    df = prepare_data_for_postanalysis_greedy(
            pre_df, 
            scenario_list, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
    
    
    # Group by UPN (Building ID) to get stats across the 70 epistemic runs
    # We calculate Mean and Std Dev across the runs
    stats = df.groupby('upn')[metric_col].agg(['mean', 'std']).reset_index()
    
    # Handle zeros to avoid division by zero
    stats['mean'] = stats['mean'].replace(0, 0.001) 
    
    # Calculate CV
    stats['cv'] = (stats['std'] / stats['mean']).abs()
    stats['intervention'] = metric_alias
    
    return stats

def run_diagnostics():
    all_stats = []
    
    # Locate files
    files = glob.glob(f"{LOG_DIR}/*.csv")
    
    if not files:
        print(f"No files found in {LOG_DIR}")
        return

    # Process each metric definition
    for alias, col_name in METRICS.items():
        print(f"\n--- Analyzing Intervention: {alias.upper()} ---")
        
        # Find files containing this metric (or just loop all if structure is consistent)
        # Here we just grab the first file that matches or loop all if they are split
        # tailored to your specific file structure (adjust if one file contains all metrics)
        
        # Simplified: Loop through all found files, try to extract the specific metric
        metric_dfs = []
        for f in files[:35]: # LIMIT to 5 files for speed testing, remove [:5] for full run
            res = analyze_log_file(f, alias, col_name)
            if res is not None:
                metric_dfs.append(res)
        
        if not metric_dfs:
            continue
            
        # Combine data for this intervention
        combined = pd.concat(metric_dfs, ignore_index=True)
        
        # --- APPLY LOGIC TESTS ---
        
        # 1. OLD METHOD (Strict)
        n_strict = (combined['cv'] < STRICT_CV_CUTOFF).sum()
        pct_strict = (n_strict / len(combined)) * 100
        
        # 2. HYBRID METHOD (SD Floor OR Relaxed CV)
        # Pass if: (SD is small) OR (CV is acceptable)
        pass_hybrid = (combined['std'] < HYBRID_SD_FLOOR) | (combined['cv'] < STRICT_CV_CUTOFF)
        n_hybrid = pass_hybrid.sum()
        pct_hybrid = (n_hybrid / len(combined)) * 100
        
        # 3. WALL SPECIFIC (Relaxed CV only)
        pass_wall = (combined['cv'] < HYBRID_CV_RELAXED)
        n_wall = pass_wall.sum()
        
        print(f"  Total Buildings: {len(combined)}")
        print(f"  [Old] Strict CV < {STRICT_CV_CUTOFF}:  {n_strict} ({pct_strict:.1f}%)")
        print(f"  [New] Hybrid (SD < £{HYBRID_SD_FLOOR}): {n_hybrid} ({pct_hybrid:.1f}%)")
        print(f"  -> Buildings 'Saved' by Hybrid: {n_hybrid - n_strict}")
        
        all_stats.append(combined)

    if not all_stats:
        return

    # Combine everything for plotting
    final_df = pd.concat(all_stats, ignore_index=True)
    
    # ========================================================================
    # 3. VISUALIZATION
    # ========================================================================
    print("\nGenerating Plots...")
    sns.set_style("whitegrid")
    
    # PLOT 1: The "Small Mean Trap" (CV vs Mean)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=final_df, 
        x='mean', 
        y='cv', 
        hue='intervention', 
        alpha=0.4,
        s=15
    )
    plt.xscale('log') # Crucial for seeing Lofts vs Heat Pumps
    plt.axhline(STRICT_CV_CUTOFF, color='red', linestyle='--', label=f'Strict Cutoff ({STRICT_CV_CUTOFF})')
    plt.title('Diagnostic 1: The "Small Mean" Trap\n(Notice how CV spikes as Mean Cost decreases)')
    plt.xlabel('Mean Cost per Ton (Log Scale)')
    plt.ylabel('Coefficient of Variation (CV)')
    plt.legend()
    plt.savefig('{output_dir}/diagnostic_cv_vs_mean.png', dpi=150)
    print("Saved: diagnostic_cv_vs_mean.png")

    # PLOT 2: The "Actual Risk" (SD vs Mean)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=final_df, 
        x='mean', 
        y='std', 
        hue='intervention', 
        alpha=0.4,
        s=15
    )
    plt.axhline(HYBRID_SD_FLOOR, color='green', linestyle='--', label=f'Hybrid Floor (£{HYBRID_SD_FLOOR})')
    plt.title('Diagnostic 2: Actual Financial Risk\n(Points below Green Line are stable regardless of CV)')
    plt.xlabel('Mean Cost per Ton')
    plt.ylabel('Standard Deviation (Uncertainty in £)')
    plt.legend()
    plt.savefig('{output_dir}/diagnostic_sd_vs_mean.png', dpi=150)
    print("Saved: diagnostic_sd_vs_mean.png")

if __name__ == "__main__":
    output_dir = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/pre_proc_greedy'
    YEARS = 5
    N_SIMULATIONS = 5000

    GAS_CARBON_FACTOR=0.18      
    ELEC_CARBON_FACTOR=0.19338  
    scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']

    run_diagnostics()