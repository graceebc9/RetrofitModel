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

# ============================================================================
# CONFIGURATION
# ============================================================================
is_hpc = is_running_on_hpc() 
is_epc = True 

if is_hpc:
    # Update this path if necessary to match your actual data location
    if not is_epc:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE'
    else:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/v8_logs_with_epc'
        # Use the file you confirmed works as the Source of Truth for headers
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
else: 
    if is_epc:
        LOG_DIR='/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge'

    else:
        LOG_DIR = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE'
    REFERENCE_FILE = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/130_log_file.csv'

if is_epc:
    OUTPUT_BASE_DIR = '2_optimized_priorities_epc/processed_best_only'
    LOG_FILE_PATH = '2_optimized_priorities_epc/processing_log.txt'
    ERROR_LOG_FILE = '2_optimized_priorities_epc/epc_processing_errors.txt'
else:
    OUTPUT_BASE_DIR = '2_optimized_priorities/processed_best_only'
    LOG_FILE_PATH = '2_optimized_priorities/processing_log.txt'
    ERROR_LOG_FILE = '2_optimized_priorities/processing_errors.txt'

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
RISK_PENALTY_SIGMA = 0  
YEARS = 5
GAS_CARBON_FACTOR = 0.18
ELEC_CARBON_FACTOR = 0.19338

SCENARIO_LIST = [
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'wall_installation',
    'join_heat_ins_decay',
    'heat_pump_only',
    'loft_installation'
]

COLS_KEEP = ['upn', 'postcode', 'premise_type', 'avg_gas_percentile' ] 

# ============================================================================
# HELPER: ERROR LOGGER
# ============================================================================
def log_error_to_file(filename, error_msg):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] FILE: {filename}\nERROR: {error_msg}\n{'-'*40}\n")

# ============================================================================
# METRIC CALCULATION (Unchanged)
# ============================================================================
def prepare_data_for_postanalysis_greedy(df_input, scenarios, n_years, factor_gas, factor_elec, 
                                  include_extremes=False, cols_to_keep=None):
    if isinstance(scenarios, str): scenarios = [scenarios]
    if cols_to_keep is None: cols_to_keep = []
    
    dtype_float = 'float32'
    output_data = {}
    
    # 1. PRESERVE REQUESTED COLUMNS
    for col in cols_to_keep:
        if col in df_input.columns:
            output_data[col] = df_input[col]

    # 2. STANDARDIZE INPUT COLUMNS
    rename_map = {
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_mean': 'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_mean',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_std': 'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_std',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_p5': 'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_p5',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_p50': 'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_p50',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_p95': 'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_p95'
    }
    df_src = df_input.rename(columns=rename_map)

    # 3. DEFINE STATS
    stats = ['mean', 'std', 'p50']
    if include_extremes: stats.extend(['p5', 'p95'])
    
    for scn in scenarios:
        is_heat = 'heat' in scn.lower()
        for stat in stats:
            col_src_gas_kwh  = f'{scn}_gas_saving_abs_kwh_{scn}_{stat}'
            col_src_elec_kwh = f'{scn}_elec_saving_abs_kwh_{scn}_{stat}'
            col_src_cost     = f'{scn}_cost_{scn}_{stat}'
            col_src_cost_avg = f'{scn}_cost_{scn}_mean' 
            
            if col_src_cost in df_src.columns: series_cost = df_src[col_src_cost]
            elif col_src_cost_avg in df_src.columns: series_cost = df_src[col_src_cost_avg]
            else: continue 

            if col_src_gas_kwh not in df_src.columns: continue

            val_gas_tonnes = (df_src[col_src_gas_kwh] * n_years * factor_gas) / 1000
            val_elec_tonnes = 0
            if is_heat and col_src_elec_kwh in df_src.columns:
                val_elec_tonnes = (df_src[col_src_elec_kwh] * n_years * factor_elec) / 1000

            val_saved_net = (val_gas_tonnes + val_elec_tonnes) * -1
            val_saved_gas = val_gas_tonnes * -1

            col_out_cost      = f'cost_total_{scn}_{stat}'
            col_out_saved_net = f'co2_saved_net_tonnes_{scn}_{stat}'
            col_out_saved_gas = f'co2_saved_gas_tonnes_{scn}_{stat}'
            col_out_cpt_net   = f'capex_per_net_ton_{scn}_{stat}'
            col_out_cpt_gas   = f'capex_per_gas_ton_{scn}_{stat}'

            output_data[col_out_cost]      = series_cost.astype(dtype_float)
            output_data[col_out_saved_net] = val_saved_net.astype(dtype_float)
            output_data[col_out_saved_gas] = val_saved_gas.astype(dtype_float)

            ratio_net = series_cost / val_saved_net
            output_data[col_out_cpt_net] = ratio_net.replace([np.inf, -np.inf], 0).fillna(0).astype(dtype_float)

            ratio_gas = series_cost / val_saved_gas
            output_data[col_out_cpt_gas] = ratio_gas.replace([np.inf, -np.inf], 0).fillna(0).astype(dtype_float)

    return pd.DataFrame(output_data, index=df_input.index)

# ============================================================================
# 1. ROBUST AGGREGATION
# ============================================================================
def pool_epistemic_runs_robust(df, scenarios, id_col='upn'):
    agg_map = {}
    calc_cols = []
    
    for col in COLS_KEEP:
        if col in df.columns and col != id_col:
            agg_map[col] = 'first'

    metrics = ['cost', 'gas_saving_abs_kwh', 'elec_saving_abs_kwh']
    
    for scn in scenarios:
        for m in metrics:
            base_col = f'{scn}_{m}_{scn}'
            col_mean = f'{base_col}_mean'
            col_std  = f'{base_col}_std'
            
            if col_mean in df.columns:
                agg_map[col_mean] = 'mean'
                if col_std in df.columns:
                    calc_cols.append({'base': base_col, 'mean': col_mean, 'std': col_std})

    grouped = df.groupby(id_col)
    df_agg = grouped.agg(agg_map)
    
    for item in calc_cols:
        var_of_means = grouped[item['mean']].var(ddof=1).fillna(0)
        temp_var = df[[id_col, item['std']]].copy()
        temp_var['var'] = temp_var[item['std']] ** 2
        mean_of_vars = temp_var.groupby(id_col)['var'].mean()
        
        pooled_std = np.sqrt(mean_of_vars + var_of_means)
        df_agg[item['std']] = pooled_std.astype('float32')

    return df_agg.reset_index()

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

def process_single_file(filepath, output_dir, master_headers, LOFT_INSULATION_EXISTING_PERCENT):
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
    # -------------------------------------------------------------
    # B. AGGREGATE
    # -------------------------------------------------------------
    try:
        agg_df = pool_epistemic_runs_robust(raw_df, SCENARIO_LIST, id_col='upn')

        # --- Identify buildings that already have loft insulation ---
        disqualified_loft_upns = apply_existing_measures_constraint(agg_df, LOFT_INSULATION_EXISTING_PERCENT)
        logging.info(f"   Excluding loft options for {len(disqualified_loft_upns)} buildings ({LOFT_INSULATION_EXISTING_PERCENT*100}%)")

        # -------------------------------------------------------------
        # C. CALC METRICS
        # -------------------------------------------------------------
        df_metrics = prepare_data_for_postanalysis_greedy(
            df_input=agg_df,
            scenarios=SCENARIO_LIST,
            n_years=YEARS,
            factor_gas=GAS_CARBON_FACTOR,
            factor_elec=ELEC_CARBON_FACTOR,
            include_extremes=False, 
            cols_to_keep=COLS_KEEP
        )
        
        all_interventions = []

        # -------------------------------------------------------------
        # D. CALCULATE ROBUST SCORES & FILTER
        # -------------------------------------------------------------
        for scn in SCENARIO_LIST:
            col_cost_mean  = f'cost_total_{scn}_mean'
            col_saved_mean = f'co2_saved_net_tonnes_{scn}_mean'
            
            if col_cost_mean not in df_metrics.columns: continue

            # --- CONSTRAINT CHECK ---
            is_loft_scenario = 'loft' in scn.lower()
            
            # 1. Penalty Calculation
            raw_gas_std = f'{scn}_gas_saving_abs_kwh_{scn}_std'
            raw_elec_std = f'{scn}_elec_saving_abs_kwh_{scn}_std'
            
            if raw_gas_std not in agg_df.columns: continue

            gas_std_t = (agg_df[raw_gas_std] * YEARS * GAS_CARBON_FACTOR) / 1000
            elec_std_t = 0
            if raw_elec_std in agg_df.columns:
                elec_std_t = (agg_df[raw_elec_std] * YEARS * ELEC_CARBON_FACTOR) / 1000
                
            total_std_t = np.sqrt(gas_std_t**2 + elec_std_t**2)
            robust_savings = df_metrics[col_saved_mean] - (RISK_PENALTY_SIGMA * total_std_t)
            robust_metric = df_metrics[col_cost_mean] / robust_savings

            # 2. Extract
            sub_df = df_metrics[COLS_KEEP].copy()
            sub_df['intervention'] = scn
            sub_df['total_capex'] = df_metrics[col_cost_mean]
            sub_df['total_co2_saved_robust'] = robust_savings
            sub_df['capex_per_net_ton'] = robust_metric
            
            # 3. Filter Monsters
            mask_valid = (
                (sub_df['capex_per_net_ton'] > 0) & 
                (sub_df['capex_per_net_ton'] <= ABS_COST_CAP) &
                (sub_df['capex_per_net_ton'].notna())
            )
            
            # --- APPLY CONSTRAINT FILTER ---
            if is_loft_scenario:
                mask_allowed = ~sub_df['upn'].isin(disqualified_loft_upns)
                mask_valid = mask_valid & mask_allowed

            clean_df = sub_df[mask_valid].copy()
            
            if not clean_df.empty:
                all_interventions.append(clean_df)

        # -------------------------------------------------------------
        # E. SELECT BEST
        # -------------------------------------------------------------
        if all_interventions:
            combined_df = pd.concat(all_interventions, ignore_index=True)
            combined_df.sort_values(by='capex_per_net_ton', ascending=True, inplace=True)
            best_only_df = combined_df.drop_duplicates(subset=['upn'], keep='first')
            
            output_path = os.path.join(output_dir, f"best_intervention_{filename}_loft_{LOFT_INSULATION_EXISTING_PERCENT}.csv")
            best_only_df.to_csv(output_path, index=False)
        else:
            logging.warning(f"No valid interventions found for {filename}")

        del raw_df, agg_df, df_metrics, all_interventions
        gc.collect()

    except Exception as e:
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
            process_single_file(f, OUTPUT_BASE_DIR, master_headers, LOFT_INSULATION_EXISTING_PERCENT)

if __name__ == "__main__":
    run_pipeline()
    print('Pipeline complete') 