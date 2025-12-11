import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import gc # Garbage Collection interface
from src.RetrofitAnalysis import process_batch_robust
from src.RetrofitUtils import safe_load 
import csv
# ==============================================================================
# CONFIGURATION
# ==============================================================================
GAS_CARBON_FACTOR = 0.18
ELEC_CARBON_FACTOR = 0.19338
YEARS = 5

# PATTERN to find your files
LOG_FILE_PATTERN  = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/*csv'
REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
# LIST OF SCENARIOS to test
SCENARIOS_TO_TEST  = [
    'wall_installation',
    'loft_installation',
    'heat_pump_only',
    'joint_heat_wall_decay',
    'joint_heat_loft_decay',
    'join_heat_ins_decay'
]


is_hpc=True 

# PLOTTING LIMIT (To prevent plotting library crashes)
MAX_PLOT_POINTS = 100000 
op_dir = '2_sigma_test_no_regrets'
ERROR_LOG_FILE = f'{op_dir}/stock_summary/processing_errors.txt'

os.makedirs(op_dir,  exist_ok=True ) 
test=False  
# ==============================================================================
# 1. DATA PREPROCESSOR (Optimized)
# ==============================================================================
def extract_scenario_columns(df_agg, scenario_name):
    """
    Extracts ONLY the columns needed for calculation to save memory.
    Returns a lightweight DataFrame.
    """
    # Construct column names
    col_gas_mean  = f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_mean'
    col_elec_mean = f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_mean'
    col_gas_std   = f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_std'
    col_elec_std  = f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_std'
    col_cost_mean = f'{scenario_name}_cost_{scenario_name}_mean'
    
    # Fast fail if column missing
    if col_cost_mean not in df_agg.columns:
        return None
    
    # specific copy of only needed raw columns to avoid fragmentation
    needed_cols = [col for col in [col_gas_mean, col_elec_mean, col_gas_std, col_elec_std, col_cost_mean] if col in df_agg.columns]
    temp_df = df_agg[needed_cols].copy()
    
    # 1. Capex
    total_capex = temp_df[col_cost_mean]

    # 2. Mean CO2 Savings
    gas_savings = temp_df[col_gas_mean] * YEARS * GAS_CARBON_FACTOR if col_gas_mean in temp_df.columns else 0
    elec_savings = temp_df[col_elec_mean] * YEARS * ELEC_CARBON_FACTOR if col_elec_mean in temp_df.columns else 0
    total_co2_kg = (gas_savings + elec_savings)
    
    # Flip negative savings
    if total_co2_kg.mean() < 0:
        total_co2_kg = total_co2_kg * -1
        
    co2_saved_tonnes = total_co2_kg / 1000.0

    # 3. Total Uncertainty
    
    gas_std_t = (temp_df[col_gas_std] * YEARS * GAS_CARBON_FACTOR / 1000.0) if col_gas_std in temp_df.columns else 0
    elec_std_t = (temp_df[col_elec_std] * YEARS * ELEC_CARBON_FACTOR / 1000.0) if col_elec_std in temp_df.columns else 0
    # NEW (Correct for correlated downscaling):
    total_std_tonnes = gas_std_t + elec_std_t
    
    # Create the result dataframe (Minimal memory footprint)
    # Using float32 to save 50% memory compared to float64
    result = pd.DataFrame({
        'total_capex': total_capex.astype('float32'),
        'co2_saved_net_tonnes_mean': co2_saved_tonnes.astype('float32'),
        'total_std_tonnes': total_std_tonnes.astype('float32')
    })
    
    # Drop NaNs immediately
    result.dropna(inplace=True)
    return result

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================
def calculate_sigma_sensitivity(df, sigma_values, scenario_name):
    """
    Expands the dataframe by len(sigma_values).
    Only run this immediately before plotting.
    """
    
    
    NO_REGRETS_CAPS = {
    'loft_installation': 1000.0,   
    # Wall is tricky. 
    # Cavity is cheap (~£1k), Solid is expensive (~£8k-£10k).
    'wall_installation': 2000.0,    
    'heat_pump_only': 0.0,       
    } 
    cap_limit = NO_REGRETS_CAPS.get(scenario_name, 0.0)

    results = []
    
    # Pre-convert columns to numpy arrays for speed
    capex = df['total_capex'].values
    mean_save = df['co2_saved_net_tonnes_mean'].values
    std_save = df['total_std_tonnes'].values
        
    # 3. IDENTIFY "CHEAP" HOMES ONCE
    # If True, these homes get a "Free Pass" on uncertainty
    is_cheap = capex < cap_limit
    
    for s in sigma_values:
            # A. Calculate Strict Savings (The default penalized value)
            savings_strict = mean_save - (s * std_save)
            
            # B. Calculate "No-Regrets" Savings (Ignore Risk -> Sigma=0)
            savings_mean = mean_save
            
            # C. Apply Logic:
            # If is_cheap is True -> Use Mean (Ignore Risk)
            # If is_cheap is False -> Use Strict (Apply Risk)
            robust_savings = np.where(is_cheap, savings_mean, savings_strict)
            
            # 4. VIABILITY CHECK
            is_viable = robust_savings > 0.01
            
            # 5. METRICS (Safe Division)
            capex_per_ton = np.full_like(robust_savings, np.nan)
            mask = is_viable
            capex_per_ton[mask] = capex[mask] / robust_savings[mask]
            
            # 6. STORE RESULTS
            temp = pd.DataFrame({
                'capex_per_net_ton': capex_per_ton,
                'is_viable': is_viable,
                'Sigma_Setting': s,
                # Optional: Add flag to see which logic was used
                'is_no_regret': is_cheap 
            })
            results.append(temp)
            
    return pd.concat(results, ignore_index=True)

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
def plot_impacts(long_df, scenario_name, save_path=None):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- MEMORY SAFETY CHECK ---
    # If we have too many points, violin plots render extremely slowly.
    # We sample randomly for the visual, but calculate viability on the full set.
    
    # Filter for plot A (Capex)
    valid_capex = long_df.dropna(subset=['capex_per_net_ton'])
    valid_capex = valid_capex[valid_capex['capex_per_net_ton'] < 10000] # Cut extreme outliers
    
    if len(valid_capex) > MAX_PLOT_POINTS:
        print(f"  [Info] Downsampling plot data from {len(valid_capex)} to {MAX_PLOT_POINTS} for performance...")
        plot_data = valid_capex.sample(n=MAX_PLOT_POINTS, random_state=42)
    else:
        plot_data = valid_capex

    # --- PLOT A: VIOLIN PLOT ---
    if plot_data.empty:
        print(f"  [Warning] No valid capex data for {scenario_name}.")
    else:
        sns.violinplot(
            data=plot_data, 
            x='Sigma_Setting', 
            y='capex_per_net_ton',
            palette="viridis",
            inner="quartile",
            ax=axes[0]
        )
    
    axes[0].set_title(f"Capex/Ton Distribution: {scenario_name}\n(Based on {len(valid_capex)} aggregate datapoints)")
    axes[0].set_xlabel("Sigma (Std Devs Subtracted)")
    axes[0].set_ylabel("Capex per Net Ton (£)")
    
    # --- PLOT B: VIABILITY ---
    # Calculate on FULL dataset (long_df), not the sampled one
    viability = long_df.groupby('Sigma_Setting')['is_viable'].mean() * 100
    
    sns.barplot(
        x=viability.index, 
        y=viability.values, 
        palette="magma", 
        ax=axes[1]
    )
    
    axes[1].set_title(f"% Viable Interventions: {scenario_name}")
    axes[1].set_ylabel("% Viable (>0 Robust Savings)")
    axes[1].set_xlabel("Sigma Setting")
    if len(axes[1].containers) > 0:
        axes[1].bar_label(axes[1].containers[0], fmt='%.1f%%')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [Saved] {save_path}")
        plt.close() 
    else:
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os  # <--- Added for directory handling


import logging
def diagnose_cliff_edge(df, scenario_name="Intervention", output_dir=None):
    """
    Diagnoses why measures fail the Sigma test using pre-processed columns:
    ['total_capex', 'co2_saved_net_tonnes_mean', 'total_std_tonnes']
    
    Args:
        df: DataFrame containing the columns above.
        scenario_name: Name of the scenario (used for title and filename).
        output_dir: (Optional) Folder path to save the image and log. If None, shows plot instead.
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        clean_name = scenario_name.replace(" ", "_").lower()
        log_filename = f"{clean_name}_diagnosis.txt"
        log_path = os.path.join(output_dir, log_filename)
        log_file = open(log_path, 'w')
    else:
        log_file = None
    
    def log(message):
        """Helper to write to both file and console"""
        if log_file:
            log_file.write(message + '\n')
        else:
            print(message)
    
    log(f"\n=== DIAGNOSING: {scenario_name} ===")
    
    # 1. CLEAN & PREPARE
    # Filter out absolute garbage (negative savings) to clean up plot
    plot_data = df.copy()
    plot_data = plot_data[plot_data['co2_saved_net_tonnes_mean'] > 0.001]
    
    # 2. CALCULATE METRICS
    # Coefficient of Variation (Noise / Signal)
    plot_data['CV'] = plot_data['total_std_tonnes'] / plot_data['co2_saved_net_tonnes_mean']
    
    # 3. LOG STATS
    log(f"Datapoints Analyzed: {len(plot_data)}")
    log(f"Mean Savings:    {plot_data['co2_saved_net_tonnes_mean'].mean():.3f} t")
    log(f"Mean Uncertainty: {plot_data['total_std_tonnes'].mean():.3f} t")
    log(f"Avg Signal-to-Noise (CV): {plot_data['CV'].mean():.2f}")
    log("-" * 30)
    log("VIABILITY CHECKS:")
    
    # Check how many pass at different Sigmas
    pass_sigma_05 = (plot_data['co2_saved_net_tonnes_mean'] - 0.5 * plot_data['total_std_tonnes']) > 0
    pass_sigma_10 = (plot_data['co2_saved_net_tonnes_mean'] - 1.0 * plot_data['total_std_tonnes']) > 0
    pass_sigma_20 = (plot_data['co2_saved_net_tonnes_mean'] - 2.0 * plot_data['total_std_tonnes']) > 0
    
    log(f"  Pass Sigma 0.5: {pass_sigma_05.mean()*100:.1f}%")
    log(f"  Pass Sigma 1.0: {pass_sigma_10.mean()*100:.1f}%")
    log(f"  Pass Sigma 2.0: {pass_sigma_20.mean()*100:.1f}%")
    
    # 4. VISUALIZATION
    plt.figure(figsize=(10, 8))
    
    # Downsample for speed if dataset is huge (>50k)
    if len(plot_data) > 50000:
        plot_data = plot_data.sample(50000, random_state=42)
        log("  (Plot downsampled to 50k points for speed)")
    
    sns.scatterplot(
        data=plot_data,
        x='co2_saved_net_tonnes_mean',
        y='total_std_tonnes',
        alpha=0.15,
        s=15,
        edgecolor=None,
        color='teal'
    )
    
    # --- ADD THE CLIFF EDGE LINES ---
    limit = plot_data['co2_saved_net_tonnes_mean'].quantile(0.99)
    x_line = np.linspace(0, limit, 100)
    
    # Line 1: Sigma = 1.0
    plt.plot(x_line, x_line, 'r--', linewidth=2, label='Sigma = 1.0 Limit (Std = Mean)')
    
    # Line 2: Sigma = 0.5
    plt.plot(x_line, x_line * 2, 'orange', linestyle='--', linewidth=2, label='Sigma = 0.5 Limit (Std = 2*Mean)')
    plt.title(f"Why {scenario_name} Fails: Signal vs. Noise\n(Points above lines are filtered out)")
    plt.xlabel("Predicted Savings (Signal) [Tonnes]")
    plt.ylabel("Uncertainty (Noise) [Tonnes]")
    plt.xlim(0, limit)
    plt.ylim(0, limit * 2.5) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 5. SAVE OR SHOW
    if output_dir:
        # Clean filename (replace spaces with underscores)
        clean_name = scenario_name.replace(" ", "_").lower()
        filename = f"{clean_name}_signal_noise_diagnosis.png"
        full_path = os.path.join(output_dir, filename)
        
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        log(f"Plot saved to: {full_path}")
        log(f"Log saved to: {log_path}")
        plt.close()  # Close plot to free memory
        
        # Close log file
        if log_file:
            log_file.close()
    else:
        plt.show()


# ==============================================================================
# MAIN RUN
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Initialize Storage Dictionary
    # We store a list of DataFrames for each scenario
    scenario_accumulators = {scen: [] for scen in SCENARIOS_TO_TEST}
    
    all_files = glob.glob(LOG_FILE_PATTERN)
    if test:
        all_files=all_files[0:5]
    
    total_files = len(all_files)
    print(f"Found {total_files} files. Starting sequential processing...")
    
    
    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
            print("Loaded headers from reference file.")
        except Exception as e:
            print(f"Warning: Could not read headers: {e}")
    
    # 2. Sequential Load & Extract
    for i, filepath in enumerate(all_files):
        if i % 50 == 0: 
            print(f"Processing file {i}/{total_files}...", end='\r')
            
        try:
            # Load raw
            df_raw = safe_load(filepath, headers, ERROR_LOG_FILE)
            
            # Process Columns (your external function)
            df_processed = process_batch_robust(df_raw, SCENARIOS_TO_TEST)
            
            # Extract only what we need for each scenario and store it
            for scen in SCENARIOS_TO_TEST:
                mini_df = extract_scenario_columns(df_processed, scen)
                if mini_df is not None and not mini_df.empty:
                    scenario_accumulators[scen].append(mini_df)
                    
            # FREE MEMORY
            del df_raw
            del df_processed
            # Force garbage collection occasionally
            if i % 100 == 0:
                gc.collect()

        except Exception as e:
            print(f"\nError processing {os.path.basename(filepath)}: {e}")

    print("\nFile processing complete. Starting Aggregation and Plotting...")
    print("-" * 50)

    # 3. Aggregate & Plot per Scenario
    sigmas = [0, 0.5, 1.0, 1.5, 2.0, 3.0]

    for scenario in SCENARIOS_TO_TEST:
        data_chunks = scenario_accumulators[scenario]
        
        if not data_chunks:
            print(f"Skipping {scenario}: No data found.")
            continue
            
        print(f"Aggregating data for: {scenario} ({len(data_chunks)} chunks)...")
        
        # Concat all chunks for this specific scenario
        combined_df = pd.concat(data_chunks, ignore_index=True)
        
        # Clear the list from memory immediately
        scenario_accumulators[scenario] = [] 
        gc.collect()
        
        print(f"  Total Rows: {len(combined_df)}")
        
        # Calculate Sensitivity (Expands data size by len(sigmas))
        results = calculate_sigma_sensitivity(combined_df, sigmas, scenario)
        

        diagnose_cliff_edge(combined_df, scenario, output_dir= op_dir)
        
        
        # Free the combined raw input, keep only results
        del combined_df
        gc.collect()
        
        # Plot
        filename = f"{op_dir}/sensitivity_AGGREGATED_{scenario}.png"
        plot_impacts(results, scenario_name=scenario, save_path=filename)
        
        # Final cleanup for this loop
        del results
        gc.collect()

    print("-" * 50)
    print("All aggregated figures generated.")