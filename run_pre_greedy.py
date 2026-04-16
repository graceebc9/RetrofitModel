import pandas as pd
import numpy as np
import glob
from pathlib import Path
import os
import gc
import logging
import csv
import sys
from src.utils import is_running_on_hpc 
from src.RetrofitUtils import  filter_typology

# ============================================================================
# CONFIGURATION
# ============================================================================
is_hpc = is_running_on_hpc() 

epc_yn= os.getenv('EPC_YN')  

if epc_yn =='Y':
    is_epc = True 
else:
    is_epc = False 
    
RISK_PENALTY_SIGMA = float(os.getenv('SIGMA')  )  

if is_hpc:
    # Update this path if necessary to match your actual data location
    if not is_epc:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE'
    else:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/v10_logs_with_epc'
        # Use the file you confirmed works as the Source of Truth for headers
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE/120_log_file.csv'
else: 
    if is_epc:
        LOG_DIR='/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge'

    else:
        LOG_DIR = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE'
    REFERENCE_FILE = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/120_log_file.csv'

if is_epc:
    OUTPUT_BASE_DIR = f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/processed_best_only'
    OUTPUT_BASE_DIR = f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/processed_all_scenarios'
    
    LOG_FILE_PATH = f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/processing_log.txt'
    ERROR_LOG_FILE = f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/epc_processing_errors.txt'
else:
    OUTPUT_BASE_DIR = f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/processed_best_only'
    OUTPUT_BASE_DIR = f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/processed_all_scenarios'
    LOG_FILE_PATH = f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/processing_log.txt'
    ERROR_LOG_FILE = f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/processing_errors.txt'

# --- NEW PARAMETER ---
# 0.35 means 35% of buildings already have loft insulation and cannot get it again.
#LOFT_INSULATION_EXISTING_PERCENT = 0.95 

loft = int(os.getenv('LOFT')) 
if loft ==1 :
    loft_perc_list = [0.95] 
else:
    loft_perc_list = [0.65] 

# CUTOFFS & PARAMETERS
ABS_COST_CAP = 200000.0

 

SCENARIO_LIST = [
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'wall_installation',
    'join_heat_ins_decay',
    'heat_pump_only',
    'loft_installation'
]

if is_epc:
    COLS_KEEP = [ 'postcode', 'premise_type', 'avg_gas_percentile' , 'CURRENT_ENERGY_RATING' , 'POTENTIAL_ENERGY_RATING',  'CURRENT_ENERGY_EFFICIENCY' , 'POTENTIAL_ENERGY_EFFICIENCY', 'INSPECTION_DATE' ] 
else:
    COLS_KEEP = [  'postcode', 'premise_type', 'avg_gas_percentile' ] 

# ============================================================================
# HELPER: ERROR LOGGER
# ============================================================================
def log_error_to_file(filename, error_msg):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] FILE: {filename}\nERROR: {error_msg}\n{'-'*40}\n")
 
     

# ============================================================================
# 1. ROBUST AGGREGATION
# ============================================================================


import pandas as pd
import numpy as np
import glob
from pathlib import Path
import os
import gc
import logging
import csv
import sys
from src.utils import is_running_on_hpc 
from src.RetrofitUtils import  filter_typology

 
# ============================================================================
# 1. ROBUST AGGREGATION (EVE'S LAW IMPLEMENTATION)
# ============================================================================

def pool_epistemic_runs_robust(df, scenarios, id_col='upn'):
    """
    Applies the Law of Total Variance (Eve's Law) to combine multiple runs per UPN.
    
    Logic:
    Total Variance = Mean of Variances (Aleatoric) + Variance of Means (Epistemic)
    """
    print("... Pooling runs using Eve's Law (Total Variance) ...")
    
    # Define the metric structure
    # We need both the MEAN and the STD from the raw runs to do this correctly
    metrics_map = {
        'capex_per_net_ton':  '{sc}_capex_per_net_ton_co2_{sc}_{stat}',
        'co2': '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}',
        'capex':   '{sc}_cost_{sc}_{stat}'
    }
    df = df.copy() 
    # 1. Build dictionary of columns to process
    # Structure: { 'new_col_name_base': {'mean_col': 'raw_mean_col', 'std_col': 'raw_std_col'} }
    cols_to_process = {}
    
    missing_cols = []

    for scn in scenarios:
        for metric_name, pattern in metrics_map.items():
            # We construct the expected column names for the Run Mean and Run Std
           
            raw_mean_col = pattern.format(sc=scn, stat='mean')
            raw_std_col = pattern.format(sc=scn, stat='std')
            
            if raw_mean_col in df.columns and raw_std_col in df.columns:
                base_name = f"{scn}_{metric_name}_robust" 
                
                cols_to_process[base_name] = {
                    'mean': raw_mean_col,
                    'std': raw_std_col
                }
                
                cols_to_process[base_name] = {
                    'mean': raw_mean_col,
                    'std': raw_std_col
                }
            else:
                # Log missing only if it's completely absent (optional)
                pass

    if not cols_to_process:
        print("Warning: No matching scenario columns found for aggregation.")
        return pd.DataFrame()

    # 2. Pre-calculate Variances for the Aleatoric part (Sigma^2)
    # We create temporary columns for variances to speed up the groupby
    temp_var_cols = []
    for base, cols in cols_to_process.items():
        var_col = f"_tmp_var_{base}"
        # Variance = Std^2
        df[var_col] = df[cols['std']] ** 2
        cols['var_col'] = var_col
        temp_var_cols.append(var_col)

    # 3. Define Aggregation Dictionary
    # We need:
    # A) Mean of the Run Means (The Central Estimate)
    # B) Variance of the Run Means (The Epistemic Uncertainty)
    # C) Mean of the Run Variances (The Aleatoric Uncertainty)
    agg_dict = {}
    for base, cols in cols_to_process.items():
        agg_dict[cols['mean']] = ['mean', 'var']  # get mean of means, and var of means
        agg_dict[cols['var_col']] = ['mean']      # get mean of variances

    # 4. Perform Groupby
    grouped = df.groupby(id_col).agg(agg_dict)

    # 5. Reconstruct the Total Mean and Total Std
    final_stats = pd.DataFrame(index=grouped.index)

    for base, cols in cols_to_process.items():
        # Retrieve the aggregated parts
        # Note: grouped columns are MultiIndex: (OriginalCol, AggFunc)
        
        # A. Total Mean = Average of the individual run means
        mu_total = grouped[(cols['mean'], 'mean')]
        
        # B. Epistemic Variance = Variance of the individual run means
        # fillna(0) handles cases with only 1 run where var is NaN
        var_epistemic = grouped[(cols['mean'], 'var')].fillna(0)
        
        # C. Aleatoric Variance = Average of the individual run variances
        var_aleatoric = grouped[(cols['var_col'], 'mean')]
        
        # D. Total Variance = Epistemic + Aleatoric
        var_total = var_epistemic + var_aleatoric
        
        # E. Total Std = Sqrt(Total Variance)
        std_total = np.sqrt(var_total)
        
        # Assign to final dataframe using your expected naming convention
        final_stats[f"{base}_mean"] = mu_total
        final_stats[f"{base}_std"] = std_total

    # 6. Re-attach Metadata (First value)
    meta_cols = [c for c in COLS_KEEP if c in df.columns]
    df_meta = df.groupby(id_col)[meta_cols].first()
    
    df_final = pd.concat([df_meta, final_stats], axis=1).reset_index()
    
    print('df_final col')
    print(df_final.columns.tolist() ) 
    
    return df_final

# ============================================================================
# ============================================================================


 
def add_sigma_columns(df_out, scenarios, sigma=1):
    """
    Generates 'Sigma' columns (Mean + Sigma * Std) for Capex, Energy, and Cost 
    across provided scenarios.
    
    Args:
        df (pd.DataFrame): DataFrame containing the mean and std columns.
        scenarios (list): List of scenario strings (e.g. 'wall_installation').
        sigma (float): The multiplier for the standard deviation (default=1).
        
    Returns:
        pd.DataFrame: The original dataframe with new sigma columns attached.
    """
    # Avoid modifying the original dataframe in place unexpectedly
    # df_out = df.copy()
    print('Starting to add sigma cols')
    metric_patterns = {
        'capex_per_net_ton':  '{sc}_capex_per_net_ton_robust_{stat}',
        'co2': '{sc}_co2_robust_{stat}',
        'capex':   '{sc}_capex_robust_{stat}'
    }

    new_columns = {} 

    # Calculation Loop
    for sc in scenarios:
        for metric, pattern in metric_patterns.items():
            
            # Construct expected column names based on the pattern
            mean_col = pattern.format(sc=sc, stat='mean')
            std_col  = pattern.format(sc=sc, stat='std')
            
            # Check if source columns exist
            if mean_col in df_out.columns and std_col in df_out.columns:
                
                # OPTIMIZATION: Use numpy arrays for speed
                mean_vals = df_out[mean_col].to_numpy()
                std_vals  = df_out[std_col].to_numpy()
                
                # Define new column name (e.g. wall_installation_capex_1sigma)
                sigma_str = str(float(sigma))
                new_col_name = f"{sc}_{metric}_robust_{sigma_str}sigma"
                
                new_columns[new_col_name] = mean_vals + (sigma * std_vals)

    # Merger Phase
    if new_columns:
        new_data_df = pd.DataFrame(new_columns, index=df_out.index)
        df_out = pd.concat([df_out, new_data_df], axis=1)
        
    return df_out
 

def apply_physical_filters_for_optimisation(df, sc ):
    capex_col = f'{sc}_capex_per_net_ton_robust_mean'
    energy_col = f'{sc}_co2_robust_mean'
    
    return df[(df[capex_col] > 0) & (df[energy_col] > 0.1)]
    
    

# ============================================================================
# 2. EXISTING MEASURES LOGIC
# ============================================================================
def apply_existing_measures_constraint(df, percent_existing):
    unique_upns = df['upn'].unique()
    n_existing = int(len(unique_upns) * percent_existing)
    rng = np.random.default_rng(seed=42) 
    existing_loft_upns = set(rng.choice(unique_upns, size=n_existing, replace=False))
    return existing_loft_upns

# ============================================================================
# 3. PROCESSING PIPELINE (UPDATED WITH LOADING LOGIC)
# ============================================================================

def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

def process_single_file(filepath, output_dir, LOFT_INSULATION_EXISTING_PERCENT, SIGMA_VAL):
    # NO_REGRETS_CAPS = {
    # 'loft_installation': 1000.0,   
    # # Wall is tricky. 
    # # Cavity is cheap (~£1k), Solid is expensive (~£8k-£10k).
    # 'wall_installation': 2000.0,    
    # 'heat_pump_only': 0.0,       
    # } 
    
    filename = Path(filepath).stem
    logging.info(f"--> Processing: {filename}")
    
    # -------------------------------------------------------------
    # A. ROBUST LOAD (Updated Logic)
    # -------------------------------------------------------------
  
    print('running for epc')
    raw_df = pd.read_csv(filepath ) 
    
    print('Starting to clean typolgoies ')
    clean_df = filter_typology(raw_df )
    print(f'Shape before: {raw_df.shape}, shape after: {clean_df.shape} ' ) 
    
    # -------------------------------------------------------------
    # B. AGGREGATE
    # -------------------------------------------------------------
    
    print('Starting to  aggregate') 
    agg_df = pool_epistemic_runs_robust(clean_df, SCENARIO_LIST, id_col='upn')
    
    # --- Identify buildings that already have loft insulation ---
    disqualified_loft_upns = apply_existing_measures_constraint(agg_df, LOFT_INSULATION_EXISTING_PERCENT)
    logging.info(f"   Excluding loft options for {len(disqualified_loft_upns)} buildings ({LOFT_INSULATION_EXISTING_PERCENT*100}%)")

    
    agg_df= add_sigma_columns(agg_df, SCENARIO_LIST, sigma=SIGMA_VAL)
    print('done sgima add')
    
    print('agg_df sigma cols' ) 
    print(agg_df.columns.tolist() ) 
    all_interventions = []

    # -------------------------------------------------------------
    # D. CALCULATE ROBUST SCORES & FILTER
    # -------------------------------------------------------------
                      
    for scn in SCENARIO_LIST:
        print(scn) 
        wdf = apply_physical_filters_for_optimisation(agg_df, scn )
        # --- CONSTRAINT CHECK ---
        is_loft_scenario = 'loft' in scn.lower()


        # 2. Extract
        sub_df = wdf[COLS_KEEP + ['upn']].copy()
        sub_df['intervention'] = scn
        sub_df['capex_per_net_ton_sigma'] = wdf[f'{scn}_capex_per_net_ton_robust_{float(RISK_PENALTY_SIGMA)}sigma']
        
        
        sub_df['mean_capex_per_net_ton'] = wdf[f'{scn}_capex_per_net_ton_robust_mean']
        sub_df['std_capex_per_net_ton'] = wdf[f'{scn}_capex_per_net_ton_robust_std']
        
        sub_df['mean_total_co2_saved'] = wdf[f'{scn}_co2_robust_mean']
        sub_df['std_total_co2_saved'] = wdf[f'{scn}_co2_robust_std']
        
        sub_df['mean_total_capex'] = wdf[f'{scn}_capex_robust_mean'] 
        sub_df['std_total_capex'] = wdf[f'{scn}_capex_robust_std'] 
        
        # 3. Filter Monsters
        mask_valid = (
            (sub_df['capex_per_net_ton_sigma'] > 0) & 
            (sub_df['capex_per_net_ton_sigma'] <= ABS_COST_CAP) &
            (sub_df['capex_per_net_ton_sigma'].notna())
        )
        
        print('sub_df cols')
        print(sub_df.columns.tolist() ) 
        # --- APPLY CONSTRAINT FILTER ---
        if is_loft_scenario:
            mask_allowed = ~sub_df['upn'].isin(disqualified_loft_upns)
            mask_valid = mask_valid & mask_allowed

        clean_df = sub_df[mask_valid].copy()
        print('clean is here') 
        if not clean_df.empty:
            all_interventions.append(clean_df)

    # -------------------------------------------------------------
    # E. KEEP ALL INTERVENTIONS (no best-only selection)
    # -------------------------------------------------------------
    if all_interventions:
        print('Combining all interventions')
        combined_df = pd.concat(all_interventions, ignore_index=True)
        # Sort for readability: by upn, then by risk-adjusted score ascending
        combined_df.sort_values(
            by=['upn', 'capex_per_net_ton_sigma'],
            ascending=[True, True],
            inplace=True
        )
 
        # Optional: add a rank column so downstream code can still find "best" easily
        combined_df['rank_within_upn'] = (
            combined_df.groupby('upn')['capex_per_net_ton_sigma']
            .rank(method='first', ascending=True)
            .astype(int)
        )
 
        output_path = os.path.join(
            output_dir,
            f"all_interventions_{filename}_loft_{LOFT_INSULATION_EXISTING_PERCENT}.csv"
        )
        print(f'Saving to {output_path}')
        combined_df.to_csv(output_path, index=False)
    else:
        logging.warning(f"No valid interventions found for {filename}")
 
    del raw_df, agg_df, all_interventions
    gc.collect()
 

def run_pipeline():
    setup_logging()
    
    # Clean previous error log
    if os.path.exists(ERROR_LOG_FILE):
        os.remove(ERROR_LOG_FILE)
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD REFERENCE HEADERS (ONCE)
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # 2. RUN BATCH
    # ---------------------------------------------------------
    files = glob.glob(f"{LOG_DIR}/*.csv")
    print(f"Found {len(files)} files to process.")
    
    for LOFT_INSULATION_EXISTING_PERCENT in loft_perc_list:
        print(f'Starting loft {LOFT_INSULATION_EXISTING_PERCENT}') 
        for f in files:
            # Pass master_headers to the function
            process_single_file(f, OUTPUT_BASE_DIR, LOFT_INSULATION_EXISTING_PERCENT, RISK_PENALTY_SIGMA)

if __name__ == "__main__":
    run_pipeline()
    print('Pipeline complete') 