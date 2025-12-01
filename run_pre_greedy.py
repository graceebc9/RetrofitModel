import pandas as pd
import numpy as np
import glob
from pathlib import Path
from src.RetrofitAnalysisUtils import load_data, prepare_data_for_postanalysis
import os
import gc
import logging
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v7/NE'
OUTPUT_BASE_DIR = 'optimized_priorities/processed_chunks'
LOG_FILE_PATH = 'optimized_priorities/processing_log.txt'
STATS_FILE_PATH = 'optimized_priorities/processing_stats.csv'

RISK_QUANTILE = 0.15

# SYMMETRIC CAP
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

cols_keep = ['postcode', 'premise_type'] 

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    # Make sure dir exists
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    
    # Configure logging to both File and Console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )

def append_stats_to_csv(stat_dict):
    """
    Appends a dictionary of statistics to the global CSV file.
    Creates the file with headers if it doesn't exist.
    """
    file_exists = os.path.isfile(STATS_FILE_PATH)
    
    fieldnames = [
        'filename', 
        'scenario', 
        'total_rows_loaded', 
        'monsters_removed', 
        'buildings_found', 
        'buildings_filtered_low_runs', 
        'buildings_kept'
    ]
    
    with open(STATS_FILE_PATH, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(stat_dict)

# ============================================================================
# PROCESSING LOGIC
# ============================================================================

def prepare_data_for_postanalysis_greedy(filepath):
    try:
        pre_df = pd.read_csv(filepath)
        proc_df = prepare_data_for_postanalysis(
                pre_df,
                SCENARIO_LIST,
                YEARS,
                GAS_CARBON_FACTOR,
                ELEC_CARBON_FACTOR
            )
        return proc_df
    except Exception as e:
        logging.error(f"Failed to load or prepare {filepath}: {e}")
        return None

def process_single_file(filepath, output_dir):
    filename = Path(filepath).stem
    logging.info(f"--> Processing File: {filename}")

    # 1. Load Data
    full_df = prepare_data_for_postanalysis_greedy(filepath)
    
    if full_df is None or full_df.empty:
        logging.warning(f"Skipping {filename} (Empty dataframe or load error)")
        return
    if 'postcode' not in full_df.columns:
         raise Exception('Postcode not present' ) 

    processed_chunks = []

    # 2. Iterate over metrics
    for metric_name, metric_col in METRICS.items():
        if metric_col not in full_df.columns:
            logging.warning(f"Column {metric_col} not found in {filename}")
            continue

        # Create a lightweight copy
        sub_df = full_df[['upn', metric_col] + cols_keep].copy()
        
        # --- STATS: Initial Count ---
        total_rows = len(sub_df)

        # --- A. Remove Monsters ---
        mask_monster = (sub_df[metric_col].abs() > ABS_COST_CAP) | (sub_df[metric_col].isna())
        n_monsters = mask_monster.sum()
        
        if n_monsters > 0:
            sub_df.loc[mask_monster, metric_col] = np.nan

        # --- B. Aggregation (Reduce to 1 row per building) ---
        agg = sub_df.groupby(cols_keep + ['upn'])[metric_col].agg([
            ('valid_runs', 'count'),
            ('raw_mean', 'mean'),
            ('raw_robust_score', lambda x: x.quantile(RISK_QUANTILE))
        ]).reset_index()

        # --- STATS: Building Counts ---
        n_buildings_total = len(agg)

        # --- C. Stability Filter ---
        # Keep only buildings with > 50 valid runs
        agg_filtered = agg[agg['valid_runs'] > 50].copy()
        
        n_buildings_kept = len(agg_filtered)
        n_filtered_out = n_buildings_total - n_buildings_kept

        # --- LOGGING STATS ---
        stats = {
            'filename': filename,
            'scenario': metric_name,
            'total_rows_loaded': total_rows,
            'monsters_removed': n_monsters,
            'buildings_found': n_buildings_total,
            'buildings_filtered_low_runs': n_filtered_out,
            'buildings_kept': n_buildings_kept
        }
        append_stats_to_csv(stats)
        
        logging.info(f"   [{metric_name}] Monsters: {n_monsters} | Filtered Buildings: {n_filtered_out} | Kept: {n_buildings_kept}")

        if agg_filtered.empty:
            continue

        # --- D. Formatting ---
        agg_filtered['intervention'] = metric_name
        agg_filtered['optimizer_cost'] = agg_filtered['raw_robust_score'] * -1
        agg_filtered['cost_robust'] = agg_filtered['raw_robust_score'] * -1
        agg_filtered['cost_mean'] = agg_filtered['raw_mean'] * -1
        agg_filtered['risk_premium'] = agg_filtered['cost_robust'] - agg_filtered['cost_mean']

        processed_chunks.append(agg_filtered)

    # 3. Save
    if processed_chunks:
        final_file_df = pd.concat(processed_chunks, ignore_index=True)
        final_file_df.sort_values(by=['upn', 'intervention'], inplace=True)
        
        output_path = os.path.join(output_dir, f"processed_{filename}.csv")
        final_file_df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to: {output_path}")
    else:
        logging.warning(f"No valid data remaining for {filename}")

    # 4. Cleanup
    del full_df
    del processed_chunks
    gc.collect()

def run_pipeline():
    setup_logging()
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Clean/Reset Stats File if you want to start fresh every run
    # if os.path.exists(STATS_FILE_PATH):
    #     os.remove(STATS_FILE_PATH)

    files = glob.glob(f"{LOG_DIR}/*.csv")
    
    logging.info(f"Found {len(files)} log files to process.")

    for i, f in enumerate(files):
        try:
            process_single_file(f, OUTPUT_BASE_DIR)
        except Exception as e:
            logging.critical(f"CRITICAL ERROR processing {f}: {e}")
            continue
        
    logging.info("Pipeline execution finished.")

if __name__ == "__main__":
    run_pipeline()