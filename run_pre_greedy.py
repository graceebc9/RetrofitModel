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
    LOG_FILE_PATH = f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/processing_log.txt'
    ERROR_LOG_FILE = f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/epc_processing_errors.txt'
else:
    OUTPUT_BASE_DIR = f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/processed_best_only'
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
    COLS_KEEP = ['upn', 'postcode', 'premise_type', 'avg_gas_percentile' , 'CURRENT_ENERGY_RATING' , 'POTENTIAL_ENERGY_RATING',  'CURRENT_ENERGY_EFFICIENCY' , 'POTENTIAL_ENERGY_EFFICIENCY', 'INSPECTION_DATE' ] 
else:
    COLS_KEEP = ['upn', 'postcode', 'premise_type', 'avg_gas_percentile' ] 

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


def pool_epistemic_runs_robust(df, scenarios, id_col='upn'):
    """
    Groups multiple rows (runs) per UPN and calculates Mean/Std 
    for every scenario column simultaneously.
    """
    
    # 1. Identify all target columns (Metrics x Scenarios)
    #    We build a single list of all columns we want to process.
    metric_patterns = {
        'capex':  '{sc}_capex_per_net_ton_co2_{sc}_{stat}',
        'energy': '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}',
        'cost':   '{sc}_cost_{sc}_{stat}'
    }
    
    target_cols = []
    for scn in scenarios:
        for metric, pattern in metric_patterns.items():
            col_name = pattern.format(sc=scn, stat='p50')
            if col_name in df.columns:
                target_cols.append(col_name)

    if not target_cols:
        print("Warning: No matching scenario columns found.")
        return pd.DataFrame()

    # 2. Group by UPN and Aggregate Vertically
    #    This collapses the rows (runs) while keeping columns distinct.
    #    We get both 'mean' and 'std' for every column in the list.
    grouped = df.groupby(id_col)[target_cols].agg(['mean', 'std'])

    # 3. Flatten the MultiIndex Columns
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

    # 4. (Optional) Re-attach static metadata columns
    #    We pick the 'first' row's value for metadata since they don't change per run.
    meta_cols = [c for c in COLS_KEEP if c in df.columns]
    df_meta = df.groupby(id_col)[meta_cols].first()
    df_final = pd.concat([df_meta, grouped], axis=1)
    if 'upn' not in df_final.columns.tolist() :
        sys.exit('upn missing')
    return df_final

 
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
        'capex':  '{sc}_capex_per_net_ton_co2_{sc}_p50_{stat}',
        'energy': '{sc}_total_energy_abs_co2_ton_samples_{sc}_p50_{stat}',
        'cost':   '{sc}_cost_{sc}_p50_{stat}'
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
                new_col_name = f"{sc}_{metric}_p50_{sigma_str}sigma"
                
                new_columns[new_col_name] = mean_vals + (sigma * std_vals)

    # Merger Phase
    if new_columns:
        new_data_df = pd.DataFrame(new_columns, index=df_out.index)
        df_out = pd.concat([df_out, new_data_df], axis=1)
        
    return df_out
 

def apply_physical_filters_for_optimisation(df, sc ):
    capex_col = f'{sc}_capex_per_net_ton_co2_{sc}_p50_mean'
    energy_col = f'{sc}_total_energy_abs_co2_ton_samples_{sc}_p50_mean'
    return df[( df[capex_col] > 0 ) & (df[energy_col]) > 0.1   ]
    
    

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

def process_single_file(filepath, output_dir, master_headers, LOFT_INSULATION_EXISTING_PERCENT, SIGMA_VAL):
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
    if master_headers:
        try:
            # Check if file is empty
            if os.path.getsize(filepath) == 0:
                log_error_to_file(filepath, "File is empty")
                return

            # 1. Peek at the first row to check headers/columns
            with open(filepath, 'r') as f:
                first_row = next(csv.reader(f))
                
            expected_cols = len(master_headers)
            
            # 2. Basic Column Count Sanity Check
            if len(first_row) != expected_cols:
                msg = f"Skipping: Column count mismatch (Found {len(first_row)} vs Expected {expected_cols})"
                logging.warning(msg)
                log_error_to_file(filepath, msg)
                return

            # 3. Prepare Load Options
            # 'on_bad_lines': 'skip' prevents crash on lines with too many commas
            # 'low_memory': False prevents MixedType warnings
            load_opts = {
                'on_bad_lines': 'skip', 
                'low_memory': False
            }

            # 4. Load Conditional on Header Existence
            if first_row == master_headers:
                # File HAS headers
                print('file has headers')
                raw_df = pd.read_csv(filepath, header=0, **load_opts)
            else:
                # File MISSING headers - Inject them
                logging.info(f"   Injecting headers into {filename}")
                raw_df = pd.read_csv(filepath, header=None, names=master_headers, **load_opts)

            # 5. Verify UPN exists after load
            if 'upn' not in raw_df.columns:
                msg = "UPN column missing after load"
                log_error_to_file(filepath, msg)
                return

        except pd.errors.ParserError as e:
            log_error_to_file(filepath, f"CSV Parser Error: {e}")
            return
        except Exception as e:
            log_error_to_file(filepath, f"General Load Error: {e}")
            return
    else:
        print('running for epc')
        raw_df = pd.read_csv(filepath ) 
    
    print('Starting to clean typolgoies ')
    clean_df = filter_typology(raw_df )
    print(f'Shape before: {raw_df.shape}, shape after: {clean_df.shape} ' ) 
    
    # -------------------------------------------------------------
    # B. AGGREGATE
    # -------------------------------------------------------------
    try:
        print('Starting to  aggregate') 
        agg_df = pool_epistemic_runs_robust(clean_df, SCENARIO_LIST, id_col='upn')
        
        # --- Identify buildings that already have loft insulation ---
        disqualified_loft_upns = apply_existing_measures_constraint(agg_df, LOFT_INSULATION_EXISTING_PERCENT)
        logging.info(f"   Excluding loft options for {len(disqualified_loft_upns)} buildings ({LOFT_INSULATION_EXISTING_PERCENT*100}%)")

        
        agg_df= add_sigma_columns(agg_df, SCENARIO_LIST, sigma=SIGMA_VAL)
        print('done sgima add')
        
        
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
            sub_df = wdf[COLS_KEEP].copy()
            sub_df['intervention'] = scn
            sub_df['capex_per_net_ton_sigma'] = wdf[f'{scn}_capex_p50_{float(RISK_PENALTY_SIGMA)}sigma']
            
            
            sub_df['mean_capex_per_net_ton'] = wdf[f'{scn}_capex_per_net_ton_co2_{scn}_p50_mean']
            sub_df['std_capex_per_net_ton'] = wdf[f'{scn}_capex_per_net_ton_co2_{scn}_p50_std']
            
            sub_df['mean_total_co2_saved'] = wdf[f'{scn}_total_energy_abs_co2_ton_samples_{scn}_p50_mean']
            sub_df['std_total_co2_saved'] = wdf[f'{scn}_total_energy_abs_co2_ton_samples_{scn}_p50_std']
            
            sub_df['mean_total_capex'] = wdf[f'{scn}_cost_{scn}_p50_mean'] 
            sub_df['std_total_capex'] = wdf[f'{scn}_cost_{scn}_p50_std'] 
            
            # 3. Filter Monsters
            mask_valid = (
                (sub_df['capex_per_net_ton_sigma'] > 0) & 
                (sub_df['capex_per_net_ton_sigma'] <= ABS_COST_CAP) &
                (sub_df['capex_per_net_ton_sigma'].notna())
            )
            
            # --- APPLY CONSTRAINT FILTER ---
            if is_loft_scenario:
                mask_allowed = ~sub_df['upn'].isin(disqualified_loft_upns)
                mask_valid = mask_valid & mask_allowed

            clean_df = sub_df[mask_valid].copy()
            print('clean is here') 
            if not clean_df.empty:
                all_interventions.append(clean_df)

        # -------------------------------------------------------------
        # E. SELECT BEST
        # -------------------------------------------------------------
        if all_interventions:
            print('Startng all interventions') 
            combined_df = pd.concat(all_interventions, ignore_index=True)
            combined_df.sort_values(by='capex_per_net_ton_sigma', ascending=True, inplace=True)
            best_only_df = combined_df.drop_duplicates(subset=['upn'], keep='first')
            
            output_path = os.path.join(output_dir, f"best_intervention_{filename}_loft_{LOFT_INSULATION_EXISTING_PERCENT}.csv")
            print(f'Saving t0 {output_path}' ) 
            best_only_df.to_csv(output_path, index=False)
        else:
            logging.warning(f"No valid interventions found for {filename}")

        del raw_df, agg_df , all_interventions
        gc.collect()

    except Exception as e:
        print(e) 
        log_error_to_file(filepath, f"Processing Error (Post-Load): {e}")
        return

def run_pipeline():
    setup_logging()
    
    # Clean previous error log
    if os.path.exists(ERROR_LOG_FILE):
        os.remove(ERROR_LOG_FILE)
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD REFERENCE HEADERS (ONCE)
    # ---------------------------------------------------------
    if not is_epc:
        try:
            logging.info(f"Loading reference headers from: {REFERENCE_FILE}")
            with open(REFERENCE_FILE, 'r') as f:
                master_headers = next(csv.reader(f))
        except Exception as e:
            logging.error(f"CRITICAL: Could not load reference file. {e}")
            return
    else:
        master_headers=None 

    # ---------------------------------------------------------
    # 2. RUN BATCH
    # ---------------------------------------------------------
    files = glob.glob(f"{LOG_DIR}/*.csv")
    print(f"Found {len(files)} files to process.")
    
    for LOFT_INSULATION_EXISTING_PERCENT in loft_perc_list:
        print(f'Starting loft {LOFT_INSULATION_EXISTING_PERCENT}') 
        for f in files:
            # Pass master_headers to the function
            process_single_file(f, OUTPUT_BASE_DIR, master_headers, LOFT_INSULATION_EXISTING_PERCENT, RISK_PENALTY_SIGMA)

if __name__ == "__main__":
    run_pipeline()
    print('Pipeline complete') 