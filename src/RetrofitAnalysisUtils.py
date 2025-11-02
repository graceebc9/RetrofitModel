
"""
Energy Retrofit Analysis Script - Multi-Scenario Version
Processes energy and carbon savings data for multiple retrofit scenarios with uncertainty analysis.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import gc 

# Add RetrofitModel to path
sys.path.append('/rds/user/gb669/hpc-work/energy_map/RetrofitModel')
from src.validate import validate_single_scenario_new

from src.RetrofitPostProcess import clean_post_process 
from src.visualisations import run_vis_new 

from src.RetrofitAnalysis import run_meta_portoflio
import psutil
import tracemalloc
from functools import wraps




def get_scenario_columns(scenario_list):
    
    STATIC_COLUMNS = [
        # Building identifiers
        'postcode', 
        'upn',                          # Unique property reference number
        'premise_type',                 # Type of building (excludes 'Domestic outbuilding')
        'epistemic_run_id',             # Run identifier for uncertainty analysis
        'premise_age',
        # Decile and grouping columns
        'avg_gas_percentile',           # Gas usage decile for grouping
        'total_gas_derived',
        'total_elec_derived',
        # Building characteristics
        'conservation_area_bool',       # Conservation area flag
        'inferred_insulation_type',     # Type of insulation (for wall scenarios)
        'epistemic__cost_scenario',
        
 
    ]
    
    # Define dtypes for static columns
    STATIC_DTYPES = {
        'postcode': 'str',
        'upn': 'Int64',                            # Large integer identifier
        'premise_type': 'object',                  # Categorical
        'epistemic_run_id': 'Int8',                # Small integer (<100)
        'avg_gas_percentile': 'Int64',             # Integer percentile
        'conservation_area_bool': 'bool',          # Boolean flag
        'inferred_insulation_type': 'object',      # Categorical
          'total_gas_derived': 'float64',
          'total_elec_derived': 'float64',
          'epistemic__cost_scenario': 'object', 
          'premise_age': 'object', 

    }
    
    fin_cols = STATIC_COLUMNS.copy()
    dtypes = STATIC_DTYPES.copy()
    


    # Add scenario columns and their dtypes
    for scenario_name in scenario_list:
        columns = [
            f'{scenario_name}_cost_{scenario_name}_mean',
            f'{scenario_name}_cost_{scenario_name}_p50',
            f'{scenario_name}_cost_{scenario_name}_p95',
            f'{scenario_name}_cost_{scenario_name}_p5',
            f'{scenario_name}_cost_{scenario_name}_std',
            f'{scenario_name}_{scenario_name}_gas_mean',
            f'{scenario_name}_{scenario_name}_gas_p5',
            f'{scenario_name}_{scenario_name}_gas_p50',
            f'{scenario_name}_{scenario_name}_gas_p95',
            f'{scenario_name}_{scenario_name}_gas_std',
           
        ]
        fin_cols += columns
        for col in columns:
            dtypes[col] = 'float64'
        if 'heat' in scenario_name :
            cols =  [ 
             f'{scenario_name}_{scenario_name}_electricity_mean',
            f'{scenario_name}_{scenario_name}_electricity_std',
            f'{scenario_name}_{scenario_name}_electricity_p5',
            f'{scenario_name}_{scenario_name}_electricity_p50',
            f'{scenario_name}_{scenario_name}_electricity_p95',
                    ]
            fin_cols += cols
            for col in cols:
                dtypes[col] = 'float64'
 
    
    return fin_cols, dtypes

# Add near the top after other imports
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def log_memory(stage_name):
    """Log memory usage at a specific stage."""
    mem = get_memory_usage()
    print(f"\n[MEMORY] {stage_name}")
    print(f"  RSS: {mem['rss_mb']:.2f} MB")
    print(f"  VMS: {mem['vms_mb']:.2f} MB")
    print(f"  Percent: {mem['percent']:.2f}%")
    return mem

def memory_profiler(func):
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"\n{'='*60}")
        print(f"[MEMORY PROFILE] Starting: {func_name}")
        mem_before = get_memory_usage()
        print(f"  Memory before: {mem_before['rss_mb']:.2f} MB")
        
        result = func(*args, **kwargs)
        
        mem_after = get_memory_usage()
        mem_delta = mem_after['rss_mb'] - mem_before['rss_mb']
        print(f"  Memory after: {mem_after['rss_mb']:.2f} MB")
        print(f"  Memory delta: {mem_delta:+.2f} MB")
        print(f"{'='*60}\n")
        
        return result
    return wrapper

@memory_profiler
def load_data(input_pattern, scenario_list):
    """Load and concatenate CSV files matching the pattern."""
    print(f"Loading data from: {input_pattern}")
    files = glob.glob(input_pattern)
    print(f"Found {len(files)} files")
    
    files =[x for x in files if 'failed' not in x]

    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")

    fin_cols, dtypes = get_scenario_columns(scenario_list)
   
    res = []
    for f in files:
        df = pd.read_csv(f, usecols=fin_cols, dtype=dtypes )
        res.append(df)

    res_df = pd.concat(res, ignore_index=True)
    print(f"Loaded {len(res_df)} rows")
    return res_df
