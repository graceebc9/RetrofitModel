import pandas as pd
import glob
import os

# --- Configuration ---

# 1. Point this to your log files (e.g., "data/logs/*.csv")
#    We will process these ONE BY ONE.
from src.utils import is_running_on_hpc 
from src.RetrofitUtils import clean_logs, load_master

running_locally = not is_running_on_hpc()

if running_locally:
    LOG_FILE_PATTERN= '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/*csv'
    new_log_epc_dir = '/Users/gracecolverd/RetrofitModel/test/new_log_epc'
    lk = pd.read_csv('/Volumes/T9/2024_Data_downloads/Eng_wales_boundary_shapefiles/Local_Authority_District_to_Region_(December_2022)_Lookup_in_England.csv')
    epc_pattern = '/Volumes/T9/2024_Data_downloads/2025_epc_database/all-domestic-certificates'
else:
    LOG_FILE_PATTERN = "/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/*file.csv"
    new_log_epc_dir='/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/v9_logs_with_epc'

    epc_pattern = '/home/gb669/rds/hpc-work/energy_map/data/epc_database/all-domestic-certificates'
    lk = pd.read_csv('/home/gb669/rds/hpc-work/energy_map/RetrofitModel/lookup_data_ew/Local_Authority_District_to_Region_(December_2022)_Lookup_in_England.csv')

nelads = lk[lk['RGN22NM']=='North East'].LAD22CD.unique().tolist()
neepcs = []
for ne in nelads:
    epcs_files = glob.glob(f'{epc_pattern}/*{ne}*/certificates.csv')
    if len(epcs_files) > 0 : 
        for x in epcs_files :
            neepcs.append(x)

 
EPC_COLS_TO_KEEP = [ 'UPRN',
'INSPECTION_DATE',
'CURRENT_ENERGY_RATING',
 'POTENTIAL_ENERGY_RATING',
 'CURRENT_ENERGY_EFFICIENCY',
 'POTENTIAL_ENERGY_EFFICIENCY',]

 
LOG_UPRN_COL_NAME = 'uprn'

 
EPC_UPRN_COL_NAME = 'UPRN' 

os.makedirs(new_log_epc_dir, exist_ok=True)
 

# --- End Configuration ---


def load_all_epc_data(epc_files_list, columns_to_load, uprn_col_name):
    """
    Loads all EPC files from a list into a single, combined DataFrame.
    """
    print("Loading all EPC data into memory...")
    all_epc_data = []
    
    if not epc_files_list:
        print(f"Warning: No EPC files provided to load.")
        return pd.DataFrame()

    for epc_file in epc_files_list:
        print(f"Processing EPC file: {epc_file}...")
        try:
            df_epc = pd.read_csv(epc_file, usecols=columns_to_load)
            all_epc_data.append(df_epc)
        except ValueError as e:
            print(f"  > Skipping {epc_file}: Could not read. Check columns? Error: {e}")
        except Exception as e:
            print(f"  > An unexpected error occurred with {epc_file}: {e}")

    if not all_epc_data:
        print("No EPC data was loaded.")
        return pd.DataFrame()

    # Combine all EPC data into one big DataFrame
    print("Combining all EPC data...")
    df_all_epcs = pd.concat(all_epc_data, ignore_index=True)
    
    # Ensure UPRNs are strings for merging
    df_all_epcs[uprn_col_name] = df_all_epcs[uprn_col_name].astype(str)
    
    # Rename the UPRN column to 'uprn' for a consistent merge key
    if uprn_col_name != 'uprn':
        df_all_epcs = df_all_epcs.rename(columns={uprn_col_name: 'uprn'})

 
    initial_rows = len(df_all_epcs)
 
    # Sort by date (newest first) before dropping duplicates
    df_all_epcs.sort_values('INSPECTION_DATE', ascending=False, inplace=True)

 
    df_all_epcs.drop_duplicates(subset=['uprn'], keep='first', inplace=True)
    
 
    final_rows = len(df_all_epcs)
    
    if initial_rows > final_rows:
        print(f"Removed {initial_rows - final_rows} duplicate UPRNs from EPC data.")

    print(f"Loaded {len(df_all_epcs)} unique EPC records into memory.")
    return df_all_epcs
    
def process_logs_against_epcs(log_file_pattern, df_all_epcs, log_uprn_col, 
                               error_log_file, new_log_epc_dir, reference_file):
    """
    Iterates through log files one-by-one and merges them against 
    the in-memory EPC DataFrame. Logs all failures to error log.
    """
    
    log_files = glob.glob(log_file_pattern)
    if not log_files:
        print(f"Warning: No log files found at {log_file_pattern}")
        return pd.DataFrame()
    
    if df_all_epcs.empty:
        print("Warning: EPC data is empty. Cannot perform merge.")
        return pd.DataFrame()
    
    # Load master headers
    print('Loading master headers...')
    master_headers = load_master(error_log_file, new_log_epc_dir, reference_file)
    
    if master_headers is None:
        print("CRITICAL: Could not load master headers. Aborting.")
        return pd.DataFrame()
    
    print(f"Master headers: {master_headers}")
    print("\n--- Starting Log File Processing (One by One) ---")
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        new_log_path = os.path.join(new_log_epc_dir, filename)
        
        # Check if output file already exists
        if os.path.exists(new_log_path):
            print(f"Skipping {filename} - already processed")
            continue
        
        print(f"Processing: {filename}...")
        
        try:
            # Load and validate log file
            df_log = clean_logs(log_file, master_headers, error_log_file)
            
            # Check if load was successful
            if df_log is None:
                print(f"  ❌ Failed to load {filename} (see error log)")
                continue
            
            if df_log.empty:
                log_error_to_file(error_log_file, filename, "File loaded but contains no data")
                print(f"  ❌ No data in {filename}")
                continue
            
            # Remove duplicates
            original_rows = len(df_log)
            df_log = df_log.drop_duplicates()
            if len(df_log) < original_rows:
                print(f"  Removed {original_rows - len(df_log)} duplicate rows")
            
            # Prepare UPRN column for merge
            if log_uprn_col not in df_log.columns:
                log_error_to_file(error_log_file, filename, 
                                f"Missing UPRN column: '{log_uprn_col}'")
                print(f"  ❌ Missing UPRN column")
                continue
            
            df_log[log_uprn_col] = df_log[log_uprn_col].astype(str)
            
            if log_uprn_col != 'uprn':
                df_log = df_log.rename(columns={log_uprn_col: 'uprn'})
            
            # Perform merge
            merged_chunk = pd.merge(
                df_log,
                df_all_epcs,
                on='uprn',
                how='inner'
            )
            
            # Save results
            if not merged_chunk.empty:
                merged_chunk.to_csv(new_log_path, index=False)
                print(f"  ✓ Saved {len(merged_chunk)} matched rows to {filename}")
            else:
                log_error_to_file(error_log_file, filename, 
                                "No UPRN matches found with EPC data")
                print(f"  ⚠ No matches found for {filename}")
        
        except KeyError as e:
            log_error_to_file(error_log_file, filename, f"Column error: {e}")
            print(f"  ❌ Column error in {filename}")
            continue
        
        except pd.errors.MergeError as e:
            log_error_to_file(error_log_file, filename, f"Merge error: {e}")
            print(f"  ❌ Merge failed for {filename}")
            continue
        
        except Exception as e:
            log_error_to_file(error_log_file, filename, 
                            f"Unexpected error during processing: {e}")
            print(f"  ❌ Unexpected error in {filename}")
            continue
    
    print("\n--- Log File Processing Complete ---")
    
# --- Main script execution ---
if __name__ == "__main__":
    # 2. Load ALL EPC data into memory
    df_epc_data = load_all_epc_data(
        epc_files_list=neepcs, 
        columns_to_load=EPC_COLS_TO_KEEP,
        uprn_col_name=EPC_UPRN_COL_NAME
    )
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/130_log_file.csv'
    ERROR_LOG_FILE = 'epc_merge_logs/processing_errors.txt'
    os.makedirs('epc_merge_logs', exist_ok=True) 
    print(df_epc_data.shape)

    if not df_epc_data.empty:
        # 3. Process Log files one-by-one against the EPC data
        final_merged_data = process_logs_against_epcs(
            log_file_pattern=LOG_FILE_PATTERN,
            df_all_epcs=df_epc_data,
 
            log_uprn_col=LOG_UPRN_COL_NAME, 
            
            error_log_file=ERROR_LOG_FILE,
            new_log_epc_dir=new_log_epc_dir,
            reference_file=REFERENCE_FILE , 
        )

 
 