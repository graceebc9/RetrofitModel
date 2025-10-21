import pandas as pd 

import glob 

def join_three_pcds(df, df_col,  pc_df  , pcds_cols):
    # merge on any one of three columns in pc_map 
    final_d = [] 
    for col in pcds_cols:
        d = df.merge(pc_df , right_on = col, left_on = df_col  )
        final_d.append(d)
    # Concatenate the results
    merged_final = pd.concat(final_d ).drop_duplicates()
    
    if len(df) != len(merged_final):
        print('Warning: some postcodes not matched')
    return merged_final 


def load_and_concatenate_data(file_pattern):
    """Load all CSV files matching the pattern and concatenate them."""
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    dataframes = [pd.read_csv(file) for file in files]
    return pd.concat(dataframes, ignore_index=True)
 


import os
from datetime import datetime
import json
import os
import socket

def is_running_on_hpc():
    """
    Detect if running on HPC by checking:
    1. SLURM job environment variable (for submitted jobs)
    2. Login node hostname pattern (for interactive testing)
    """
    # Check if running in a SLURM job
    if 'SLURM_JOB_ID' in os.environ:
        return True
    
    # Check if on a login node
    hostname = socket.gethostname().lower()
    if 'login' in hostname:
        return True
    
    return False

 


def log_configuration(log_filepath, **config_params):
    """
    Log configuration parameters to a text file with recursive object logging.
    
    Args:
        log_filepath: Path to the log file
        **config_params: Configuration parameters to log as keyword arguments
    """
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    def format_value(value, indent=2):
        """Recursively format values, including nested objects."""
        indent_str = " " * indent
        
        if isinstance(value, (dict)):
            result = "{\n"
            for k, v in value.items():
                result += f"{indent_str}  {k}: {format_value(v, indent + 2)}\n"
            result += f"{indent_str}}}"
            return result
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]"
            # For short lists, keep on one line
            if len(value) <= 5 and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                return json.dumps(value)
            result = "[\n"
            for item in value:
                result += f"{indent_str}  {format_value(item, indent + 2)},\n"
            result += f"{indent_str}]"
            return result
        elif hasattr(value, '__dict__'):
            # For objects, recursively log their attributes
            result = f"{type(value).__name__}:\n"
            for attr, attr_value in vars(value).items():
                if not attr.startswith('_'):
                    result += f"{indent_str}  {attr}: {format_value(attr_value, indent + 2)}\n"
            return result.rstrip('\n')
        else:
            return str(value)
    
    with open(log_filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CONFIGURATION LOG\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Host: {os.getenv('HOSTNAME', 'unknown')}\n")
        f.write(f"SLURM Job ID: {os.getenv('SLURM_JOB_ID', 'N/A')}\n")
        f.write(f"SLURM Array Task ID: {os.getenv('SLURM_ARRAY_TASK_ID', 'N/A')}\n")
        f.write(f"Job name if given: {config_params['job_name']}  \n")
        f.write("="*60 + "\n\n")
        
        for key, value in config_params.items():
            f.write(f"{key}:\n")
            f.write(f"  {format_value(value, 2)}\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"Configuration logged to: {log_filepath}")


# import os
# from datetime import datetime
# import json

# def log_configuration(log_filepath, **config_params):
#     """
#     Log configuration parameters to a text file.
    
#     Args:
#         log_filepath: Path to the log file
#         **config_params: Configuration parameters to log as keyword arguments
#     """
#     os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
#     with open(log_filepath, 'w') as f:
#         f.write("="*60 + "\n")
#         f.write("CONFIGURATION LOG\n")
#         f.write("="*60 + "\n")
#         f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Host: {os.getenv('HOSTNAME', 'unknown')}\n")
#         f.write(f"SLURM Job ID: {os.getenv('SLURM_JOB_ID', 'N/A')}\n")
#         f.write(f"SLURM Array Task ID: {os.getenv('SLURM_ARRAY_TASK_ID', 'N/A')}\n")
#         f.write("="*60 + "\n\n")
        
#         for key, value in config_params.items():
#             f.write(f"{key}:\n")
            
#             # Handle different types
#             if isinstance(value, (dict, list, tuple)):
#                 f.write(f"  {json.dumps(value, indent=2, default=str)}\n")
#             elif hasattr(value, '__dict__'):
#                 # For objects like RetrofitConfig, log their attributes
#                 f.write("  Object attributes:\n")
#                 for attr, attr_value in vars(value).items():
#                     if not attr.startswith('_'):
#                         f.write(f"    {attr}: {attr_value}\n")
#             else:
#                 f.write(f"  {value}\n")
#             f.write("\n")
        
#         f.write("="*60 + "\n")
    
#     print(f"Configuration logged to: {log_filepath}")


 