import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import random
import os
from src.RetrofitAnalysis import process_batch_robust
# ==============================================================================
# CONFIGURATION
# ==============================================================================
GAS_CARBON_FACTOR = 0.18
ELEC_CARBON_FACTOR = 0.19338
YEARS = 5

# PATTERN to find your files
LOG_FILE_PATTERN  = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/*csv'

# LIST OF SCENARIOS to test
# Make sure these match the prefixes in your column names (e.g. 'heat_pump_only_cost...')
SCENARIOS_TO_TEST  = [
    'wall_installation',
    'loft_installation',
    'heat_pump_only',
    'joint_heat_wall_decay',
    'joint_heat_loft_decay',
    'join_heat_ins_decay'
]

# ==============================================================================
# 1. DATA PREPROCESSOR
# ==============================================================================
def preprocess_agg_data(df_agg, scenario_name):
    """
    Extracts columns specific to 'scenario_name' from the aggregated dataframe.
    """
    df = df_agg.copy()
    
    # Construct column names based on the scenario prefix
    col_gas_mean = f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_mean'
    col_elec_mean = f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_mean'
    col_gas_std = f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_std'
    col_elec_std = f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_std'
    col_cost_mean = f'{scenario_name}_cost_{scenario_name}_mean'
    
    # Check if this scenario actually exists in the dataframe
    if col_cost_mean not in df.columns:
        print(f"  [Skipping] Column {col_cost_mean} not found.")
        return pd.DataFrame() # Return empty if data missing
    
    df['total_capex'] = df[col_cost_mean]

    # Calculate Mean CO2 Savings
    gas_savings = df[col_gas_mean] * YEARS * GAS_CARBON_FACTOR if col_gas_mean in df.columns else 0
    elec_savings = df[col_elec_mean] * YEARS * ELEC_CARBON_FACTOR if col_elec_mean in df.columns else 0
    
    total_co2_kg = (gas_savings + elec_savings)
    
    # Flip negative savings to positive if needed
    if total_co2_kg.mean() < 0:
        total_co2_kg = total_co2_kg * -1
        
    df['co2_saved_net_tonnes_mean'] = total_co2_kg / 1000.0

    # Calculate Total Uncertainty
    gas_std_t = (df[col_gas_std] * YEARS * GAS_CARBON_FACTOR / 1000.0) if col_gas_std in df.columns else 0
    elec_std_t = (df[col_elec_std] * YEARS * ELEC_CARBON_FACTOR / 1000.0) if col_elec_std in df.columns else 0
    
    df['total_std_tonnes'] = np.sqrt(gas_std_t**2 + elec_std_t**2)
    
    df = df.dropna(subset=['total_capex', 'co2_saved_net_tonnes_mean'])
    
    return df[['upn', 'total_capex', 'co2_saved_net_tonnes_mean', 'total_std_tonnes']]

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================
def calculate_sigma_sensitivity(df, sigma_values):
    results = []
    
    col_capex = 'total_capex'
    col_mean_save = 'co2_saved_net_tonnes_mean'
    col_std_save = 'total_std_tonnes'

    for s in sigma_values:
        temp = df.copy()
        
        # Robust Savings calculation
        temp['robust_savings'] = temp[col_mean_save] - (s * temp[col_std_save])
        temp['is_viable'] = temp['robust_savings'] > 0.01
        
        # Calculate Metric
        temp['capex_per_net_ton'] = np.where(
            temp['is_viable'],
            temp[col_capex] / temp['robust_savings'],
            np.nan 
        )
        
        temp['Sigma_Setting'] = s
        results.append(temp)
        
    return pd.concat(results, ignore_index=True)

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
def plot_impacts(long_df, scenario_name, save_path=None):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT A: VIOLIN PLOT ---
    # Filter extreme outliers for readability
    plot_data = long_df[long_df['capex_per_net_ton'] < 10000].copy()
    
    if plot_data.empty:
        print(f"  [Warning] All data points filtered out for {scenario_name}.")
    else:
        sns.violinplot(
            data=plot_data, 
            x='Sigma_Setting', 
            y='capex_per_net_ton',
            palette="viridis",
            inner="quartile",
            ax=axes[0]
        )
        
    axes[0].set_title(f"Capex/Ton Distribution: {scenario_name}")
    axes[0].set_xlabel("Sigma (Std Devs Subtracted)")
    axes[0].set_ylabel("Capex per Net Ton (£)")
    
    # --- PLOT B: VIABILITY ---
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

# ==============================================================================
# MAIN RUN
# ==============================================================================
if __name__ == "__main__":
    
    # --- A. FIND FILES ---
    all_files = glob.glob(LOG_FILE_PATTERN)
    print(f"Found {len(all_files)} files.")
    
    if not all_files:
        print("No files found.")
        exit()

    # --- B. SELECT RANDOM SAMPLE ---
    num_to_select = 10
    if len(all_files) > num_to_select:
        selected_files = random.sample(all_files, num_to_select)
    else:
        selected_files = all_files
    
    print(f"Processing {len(selected_files)} files...")

    # --- C. BATCH LOAD DATA (ONCE) ---
    # We load and run the robust processor once per file, then store result in memory.
    # This prevents reading the CSVs 10 times if you have 10 scenarios.
    loaded_data_frames = []
    
    for filepath in selected_files:
        try:
            df_raw = pd.read_csv(filepath)
            
            
            
            df_processed = process_batch_robust(df_raw, SCENARIOS_TO_TEST) 
            
            loaded_data_frames.append(df_processed)
        except Exception as e:
            print(f"  Error loading file {os.path.basename(filepath)}: {e}")

    if not loaded_data_frames:
        print("No data loaded successfully. Exiting.")
        exit()

    print("Data loaded. Starting scenario analysis loop...")
    print("-" * 50)

    # --- D. LOOP THROUGH SCENARIOS ---
    sigmas = [0, 0.5, 1.0, 1.5, 2.0, 3.0]

    for scenario in SCENARIOS_TO_TEST:
        print(f"Analyzing: {scenario}")
        
        scenario_results = []
        
        # 1. Extract data for this specific scenario from all loaded files
        for df_agg in loaded_data_frames:
            clean_part = preprocess_agg_data(df_agg, scenario)
            if not clean_part.empty:
                scenario_results.append(clean_part)
        
        # 2. If we have data, run analysis
        if scenario_results:
            combined_df = pd.concat(scenario_results, ignore_index=True)
            
            # Sensitivity Calculation
            results = calculate_sigma_sensitivity(combined_df, sigmas)
            
            # Plot and Save
            filename = f"sensitivity_{scenario}.png"
            plot_impacts(results, scenario_name=scenario, save_path=filename)
        else:
            print(f"  [Skipped] No valid data found for scenario: {scenario}")
            
    print("-" * 50)
    print("Batch processing complete.")