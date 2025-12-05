import logging 


import logging
import random # Needed for aleatoric sampling
import pandas as pd 
import os 

import random 
import os
import csv
import logging
import pandas as pd
from pathlib import Path




def safe_load(filepath, master_headers, ERROR_LOG_FILE):
    def log_error_to_file(filename, error_msg):
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ERROR_LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] FILE: {filename}\nERROR: {error_msg}\n{'-'*40}\n")

    filename = Path(filepath).stem
    logging.info(f"--> Processing: {filename}")
    
    # -------------------------------------------------------------
    # A. ROBUST LOAD (Updated Logic)
    # -------------------------------------------------------------
    if master_headers:
        try:
            # Check if file is empty
            if os.path.getsize(filepath) == 0:
                print('Empty file')
                log_error_to_file(filepath, "File is empty")
                return

            # 1. Peek at the first row to check headers/columns
            with open(filepath, 'r') as f:
                first_row = next(csv.reader(f))
                
            expected_cols = len(master_headers)
            
            # 2. Basic Column Count Sanity Check
            if len(first_row) != expected_cols:
                msg = f"Skipping: Column count mismatch (Found {len(first_row)} vs Expected {expected_cols})"
                print(msg)
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
            print('error')
            log_error_to_file(filepath, f"CSV Parser Error: {e}")
            return
        except Exception as e:
            log_error_to_file(filepath, f"General Load Error: {e}")
            return
    else:
        print('running for epc')
        raw_df = pd.read_csv(filepath ) 
    return raw_df 



# def log_error_to_file(error_log_file, filename, error_msg):
#     """Log errors to a file with timestamp."""
#     timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open(error_log_file, 'a') as f:
#         f.write(f"[{timestamp}] FILE: {filename}\n"
#                 f"ERROR: {error_msg}\n{'-'*40}\n")

def load_master(error_log_file, output_base_dir, reference_file):
    """Load reference headers from master CSV file."""
    # Clean previous error log
    if os.path.exists(error_log_file):
        os.remove(error_log_file)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Load reference headers
    try:
        logging.info(f"Loading reference headers from: {reference_file}")
        with open(reference_file, 'r') as f:
            master_headers = next(csv.reader(f))
        return master_headers
    except Exception as e:
        logging.error(f"CRITICAL: Could not load reference file. {e}")
        return None

def clean_logs(filepath, master_headers, error_log_file):
    """Load and validate CSV file against master headers."""
    filename = os.path.basename(filepath)
    
    try:
        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            log_error_to_file(error_log_file, filename, "File is empty")
            return None
        
        # Peek at first row to check headers/columns
        with open(filepath, 'r') as f:
            first_row = next(csv.reader(f))
        
        # Validate column count
        expected_cols = len(master_headers)
        if len(first_row) != expected_cols:
            msg = f"Column count mismatch (Found {len(first_row)} vs Expected {expected_cols})"
            logging.warning(msg)
            log_error_to_file(error_log_file, filename, msg)
            return None
        
        # Load CSV with error handling
        load_opts = {
            'on_bad_lines': 'skip',
            'low_memory': False
        }
        
        # Load with or without headers
        if first_row == master_headers:
            raw_df = pd.read_csv(filepath, header=0, **load_opts)
        else:
            logging.info(f"Injecting headers into {filename}")
            raw_df = pd.read_csv(filepath, header=None, names=master_headers, **load_opts)
        
        # Verify UPN column exists
        if 'upn' not in raw_df.columns:
            log_error_to_file(error_log_file, filename, "UPN column missing after load")
            return None
        
        return raw_df
        
    except pd.errors.ParserError as e:
        log_error_to_file(error_log_file, filename, f"CSV Parser Error: {e}")
        return None
    except Exception as e:
        log_error_to_file(error_log_file, filename, f"General Load Error: {e}")
        return None
    

def calc_est_flats_building(
    building_footprint_area: float, 
    typology_col: str, 
    floor_count: float, 
    fp_mean: float, 
    fp_std: float, 
    eff_mean: float, 
    eff_std: float
) -> int:
    """
    Runs ONE   sample for the number of flats in a building.
    
    This is the INNER LOOP of the 2D Monte Carlo. It takes the epistemic
    parameters (the means and std devs) as arguments.
    """

    
    # --- 1. Handle House Typologies (Deterministic) ---
    house_typologies = [
        'Small low terraces', 'Tall terraces 3-4 storeys', 'Large semi detached',
        'Standard size detached', 'Standard size semi detached', 'Planned balanced mixed estates', 
        '2 storeys terraces with t rear extension', 'Semi type house in multiples',
        'Large detached', 'Very large detached', 'Linked and step linked premises',
        'Domestic outbuilding',
        
    ]
    
    if typology_col in house_typologies or typology_col == 'all_unknown_typology' or typology_col is  None:
        return 1
 
    # --- 2. Run Aleatoric Sample (for Flats) ---
 
    # --- 2. Validate Inputs Before Sampling ---
    if any(param is None for param in [fp_mean, fp_std, eff_mean, eff_std]):
        logging.warning(
            f"Missing epistemic parameters for typology '{typology_col}'. "
            f"fp_mean={fp_mean}, fp_std={fp_std}, eff_mean={eff_mean}, eff_std={eff_std}. "
            f"Defaulting to 1 flat."
        )
        raise Exception (f'Epistemic scenario missing params Missing epistemic parameters for typology {typology_col} ' )

    try:

        # check inputs are present: 
        
        # --- Aleatoric Sampling ---
        # Sample from the distributions defined by the epistemic parameters
        
        # We must clip the samples to avoid nonsensical values
        # e.g., negative footprint or efficiency > 1.0
        
        sampled_footprint = max(20.0, random.normalvariate(fp_mean, fp_std))
        sampled_efficiency = max(0.50, min(0.95, random.normalvariate(eff_mean, eff_std)))
        
        # --- Calculation ---
        usable_area_per_floor = float(building_footprint_area) * sampled_efficiency
        flats_per_floor = usable_area_per_floor / sampled_footprint
        total_flats = float(floor_count) * flats_per_floor
        
        return max(1, round(total_flats))
        
    except (TypeError, ZeroDivisionError, ValueError) as e:
        print(f'building_footprint_area: {building_footprint_area}, floor_count: {floor_count}, typology_col: {typology_col}, sampled_footprint: {sampled_footprint}, sampled_efficiency: {sampled_efficiency} ' )
        # Log the error and return a default
        logging.error(f"Error in aleatoric sample for typology {typology_col}: {e}. Defaulting to 1.")
        return 1


# def calculate_estimated_flats_per_building(building_footprint_area, typology_col, floor_count):
#     """Calculate estimated number of flats based on building characteristics."""
#     house_typologies = [
#         'Small low terraces', 'Tall terraces 3-4 storeys', 'Large semi detached',
#         'Standard size detached', 'Standard size semi detached',
#         '2 storeys terraces with t rear extension', 'Semi type house in multiples',
#         'Large detached', 'Very large detached', 'Linked and step linked premises',
#         'Domestic outbuilding',
#     ]
    
#     if typology_col in house_typologies or typology_col == 'all_unknown_typology':
#         return 1
    
#     typical_flat_footprints = {
#         'Medium height flats 5-6 storeys': 50,
#         '3-4 storey and smaller flats': 60,

#         'Tall flats 6-15 storeys': 45,
#         'Very tall point block flats': 40,
#         'Planned balanced mixed estates': 65,
#     }
    
#     efficiency_factors = {
#         'Medium height flats 5-6 storeys': 0.75,
#         '3-4 storey and smaller flats': 0.80,
#         'Tall flats 6-15 storeys': 0.70,
#         'Very tall point block flats': 0.65,
#         'Planned balanced mixed estates': 0.80,
#     }
    
#     flat_footprint = typical_flat_footprints.get(typology_col, 50)
#     efficiency = efficiency_factors.get(typology_col, 0.75)
    
#     try:
        
#         usable_area_per_floor = building_footprint_area * efficiency
#         flats_per_floor = usable_area_per_floor / flat_footprint
#         total_flats = float(floor_count) * float(flats_per_floor)
#         return max(1, round(total_flats))
#     except (TypeError, ZeroDivisionError, ValueError) as e:
#         # E: Replaced magic number -999 with 1 and logged the error
#         logging.error(f"Error calculating flats for typology {typology_col}: {e}. Defaulting to 1.")
#         return 1


