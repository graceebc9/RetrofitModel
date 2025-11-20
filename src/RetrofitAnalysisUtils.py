"""
Updated data loading utilities for new column format.
Works with pre-processed data that includes absolute kWh savings and cost metrics.
"""
import sys 
from typing import List 
import pandas as pd
import glob
import psutil
from functools import wraps


def get_scenario_columns(scenario_list, epc=False):
    """
    Generate column names and dtypes for the new data format.
    
    New format includes:
    - Absolute kWh savings (gas and electricity)
    - Percentage savings
    - Cost metrics (total cost and cost per kWh)
    - All with percentiles (p5, p50, p95) and statistics (mean, std)
    """
    
    STATIC_COLUMNS = [
        # Building identifiers
        'postcode', 
        'upn',                          # Unique property reference number
        'premise_type',                 # Type of building (excludes 'Domestic outbuilding')
        'epistemic_run_id',             # Run identifier for uncertainty analysis
        'premise_age',
        
        # Decile and grouping columns
        'avg_gas_percentile',           # Gas usage decile for grouping
        'total_gas_derived',            # Total gas consumption
        'total_elec_derived',           # Total electricity consumption
        
        # Building characteristics
        'conservation_area_bool',       # Conservation area flag
        'inferred_insulation_type',     # Type of insulation (for wall scenarios)
        'epistemic__cost_scenario',     # Cost scenario identifier
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
    
    # Define statistics suffixes
    stats = ['mean', 'std', 'p5', 'p50', 'p95']
    
    # Add scenario columns and their dtypes
    for scenario_name in scenario_list:
        
        # 1. COST COLUMNS (same format as before)
        cost_cols = [f'{scenario_name}_cost_{scenario_name}_{stat}' for stat in stats]
        fin_cols.extend(cost_cols)
        for col in cost_cols:
            dtypes[col] = 'float64'
        
        # 2. COST EFFICIENCY METRICS (new in updated format)
        cost_efficiency_cols = [
            f'{scenario_name}_cost_per_gas_kwh_{scenario_name}_mean',
            f'{scenario_name}_cost_per_gas_kwh_{scenario_name}_std',
            f'{scenario_name}_cost_per_gas_kwh_{scenario_name}_p5',
            f'{scenario_name}_cost_per_gas_kwh_{scenario_name}_p50',
            f'{scenario_name}_cost_per_gas_kwh_{scenario_name}_p95',
            f'{scenario_name}_cost_per_total_energy_kwh_{scenario_name}_mean',
            f'{scenario_name}_cost_per_total_energy_kwh_{scenario_name}_std',
            f'{scenario_name}_cost_per_total_energy_kwh_{scenario_name}_p5',
            f'{scenario_name}_cost_per_total_energy_kwh_{scenario_name}_p50',
            f'{scenario_name}_cost_per_total_energy_kwh_{scenario_name}_p95',
        ]
        fin_cols.extend(cost_efficiency_cols)
        for col in cost_efficiency_cols:
            dtypes[col] = 'float64'
        
        # 3. GAS SAVINGS - ABSOLUTE (kWh)
        gas_abs_cols = [f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_{stat}' for stat in stats]
        fin_cols.extend(gas_abs_cols)
        for col in gas_abs_cols:
            dtypes[col] = 'float64'
        
        # 4. GAS SAVINGS - PERCENTAGE
        gas_perc_cols = [f'{scenario_name}_gas_saving_perc_{scenario_name}_{stat}' for stat in stats]
        fin_cols.extend(gas_perc_cols)
        for col in gas_perc_cols:
            dtypes[col] = 'float64'
        
        # 5. ELECTRICITY COLUMNS (only for heat pump scenarios)
        if 'heat' in scenario_name.lower():
            
            # Electricity savings - absolute (kWh)
            elec_abs_cols = [f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_{stat}' for stat in stats]
            fin_cols.extend(elec_abs_cols)
            for col in elec_abs_cols:
                dtypes[col] = 'float64'
            
            # Electricity savings  
            elec_perc_cols = [
                f'{scenario_name}_elec_saving_perc_{scenario_name}_{stat}' for stat in stats
            ]
 
            fin_cols.extend(elec_perc_cols)
  
            for col in elec_perc_cols  :
                dtypes[col] = 'float64'
        if epc: 
            epc_cols = ['CURRENT_ENERGY_RATING',
                                'POTENTIAL_ENERGY_RATING',
                                'CURRENT_ENERGY_EFFICIENCY',
                                'POTENTIAL_ENERGY_EFFICIENCY',
                                'INSPECTION_DATE']
            fin_cols.extend(epc_cols)
            for x in epc_cols:
                dtypes[x] = 'object'

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
def load_data(input_pattern, scenario_list, validate_columns=True, epc=False , number=None):
    """
    Load and concatenate CSV files matching the pattern.
    
    Parameters:
    -----------
    input_pattern : str
        Glob pattern for input CSV files
    scenario_list : list
        List of scenario names to load
    validate_columns : bool
        If True, print warning for missing columns instead of failing
        
    Returns:
    --------
    pd.DataFrame
        Concatenated dataframe with all scenarios
    """
    print(f"Loading data from: {input_pattern}")
    files = glob.glob(input_pattern)
    print(f"Found {len(files)} files")
    
    # Filter out failed files
    files = [x for x in files if 'failed' not in x]
    print(f"After filtering: {len(files)} files")

    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")

    if number:
        files=files[0:number]

    # Get expected columns
    fin_cols, dtypes = get_scenario_columns(scenario_list, epc)
    print(f"\nExpecting {len(fin_cols)} columns for {len(scenario_list)} scenarios")
    
    res = []
    for i, f in enumerate(files):
        print(f"Loading file {i+1}/{len(files)}: {f}")
        
        if validate_columns and i == 0:
            # On first file, check which columns exist
            first_df = pd.read_csv(f, nrows=0)  # Just read header
            available_cols = first_df.columns.tolist()
            
            # Find which expected columns are missing
            missing_cols = [col for col in fin_cols if col not in available_cols]
            
            if missing_cols:
                print(f"\n⚠️  WARNING: {len(missing_cols)} expected columns not found in data:")
                print(f"Missing columns: {missing_cols[:10]}...")  # Show first 10
                
                # Adjust columns and dtypes to only include available ones
                fin_cols_available = [col for col in fin_cols if col in available_cols]
                dtypes_available = {k: v for k, v in dtypes.items() if k in available_cols}
                
                print(f"\nProceeding with {len(fin_cols_available)} available columns")
                fin_cols = fin_cols_available
                dtypes = dtypes_available
        
        try:
            df = pd.read_csv(f, usecols=fin_cols, dtype=dtypes)
            res.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            # Try without dtype specification
            try:
                df = pd.read_csv(f, usecols=fin_cols)
                res.append(df)
                print(f"  → Loaded without dtype specification")
            except Exception as e2:
                print(f"  → Failed to load: {e2}")
                continue

    if len(res) == 0:
        raise ValueError("No files successfully loaded!")

    res_df = pd.concat(res, ignore_index=True)
    print(f"\n✓ Successfully loaded {len(res_df)} rows from {len(res)} files")
    print(f"Columns in dataframe: {len(res_df.columns)}")
    
    return res_df

import sys
import pandas as pd
import numpy as np

def prepare_data_for_postanalysis(df, scenario_list, years, gas_carbon_factor, elec_carbon_factor):
    """
    Prepare data for post work by adding CO2 conversion columns.
    
    Optimized to:
    1. Store all new columns as float32 to save memory.
    2. Avoid redundant df.copy() to reduce peak memory usage.
    3. Collect all new columns and add them at once to avoid fragmentation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with new column format
    scenario_list : list
        List of scenario names
    years : int
        Number of years for CO2 calculation
    gas_carbon_factor : float
        Gas carbon factor (kg CO2/kWh)
    elec_carbon_factor : float
        Electricity carbon factor (kg CO2/kWh)
        
    Returns:
    --------
    pd.DataFrame
        Data with added CO2 columns (in tonnes) needed for greedy algorithm
    """
    print("\n" + "="*80)
    print("PREPARING DATA FOR GREEDY ALGORITHM (Optimized for float32)")
    print("="*80)
    
    if isinstance(scenario_list, str):
         scenario_list=[scenario_list]
  
    # Define the target data type for all new columns
    dtype = 'float32'
    
  
    # 2. Rename columns. This operation returns a new DataFrame.
    #    This is now our 'prepared' DataFrame.
    df_prep = df.rename(columns={
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_mean': 'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_mean',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_std':'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_std',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_p5':'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_p5',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_p50':'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_p50',
        'join_heat_ins_decay_elec_saving_perc__join_heat_ins_decay_p95':'join_heat_ins_decay_elec_saving_perc_join_heat_ins_decay_p95'
    })

    stats = ['mean', 'std', 'p5', 'p50', 'p95']
    
    # Dictionary to collect all new columns (as float32 Series)
    new_columns = {}
    
    for scenario_name in scenario_list:
        print(f"\nProcessing scenario: {scenario_name}")
        
        for stat in stats:
            # Add total kwh savings 
            gas_col = f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_{stat}'
            elec_col = f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_{stat}'
            net_total_col = f'{scenario_name}_net_total_saving_abs_kwh_{scenario_name}_{stat}'
            
            if 'heat' in scenario_name.lower():
                # 3. Cast all new columns to float32
                new_columns[net_total_col] = (df_prep[elec_col] + df_prep[gas_col]).astype(dtype)
            else:
                # 3. Cast all new columns to float32
                new_columns[net_total_col] = df_prep[gas_col].astype(dtype)
            
            # Convert gas kWh to CO2 (in tonnes)
            co2_col = f'gas_total_tonne_co2_saved_{scenario_name}_{years}yr_{stat}'
            co2kg_col = f'gas_{years}yr_kg_co2_saved_{scenario_name}_{stat}'
            kwh_col = f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_{stat}'

            if kwh_col in df_prep.columns:
                # 3. Cast all new columns to float32
                new_columns[co2_col] = ((df_prep[kwh_col] * years * gas_carbon_factor) / 1000).astype(dtype)
                new_columns[co2kg_col] = (df_prep[kwh_col] * years * gas_carbon_factor).astype(dtype)
                print(f"  Created: {co2_col}")
            else:
                print(f"  ⚠️  Missing: {kwh_col}")
                sys.exit() 

            co2_col = f'total_tonne_co2_saved_{scenario_name}_{years}yr_{stat}'
            kwh_col = f'{scenario_name}_net_total_saving_abs_kwh_{scenario_name}_{stat}'

            # Use the newly created column from new_columns dict
            if kwh_col in new_columns:
                # 3. Cast all new columns to float32
                new_columns[co2_col] = ((new_columns[kwh_col] * years * gas_carbon_factor) / 1000).astype(dtype)
                print(f"  Created: {co2_col}")
            else:
                print(f"  ⚠️  Missing: {kwh_col}")
                sys.exit() 

            # Convert cost per ton 
            for ff in ['gas', 'total_energy']:
                cost_per_ton_col = f'{scenario_name}_cost_per_{ff}_ton_{scenario_name}_{stat}'
                cost_per_kwh_col = f'{scenario_name}_cost_per_{ff}_kwh_{scenario_name}_{stat}'
                # 3. Cast all new columns to float32
                new_columns[cost_per_ton_col] = (df_prep[cost_per_kwh_col] * 1000 / (years * gas_carbon_factor)).astype(dtype)
                print(f"  Created: {cost_per_ton_col}")
            
            # Cost in millions
            cost_col = f'{scenario_name}_cost_{scenario_name}_mean'
            mill_col = f'{scenario_name}_cost_{scenario_name}_{stat}_mill'
            # 3. Cast all new columns to float32
            new_columns[mill_col] = (df_prep[cost_col] / 1_000_000).astype(dtype)
            print(f"  Created: {mill_col} (in millions)")

        # Convert electricity kWh to CO2 (in tonnes, for heat scenarios)
        if 'heat' in scenario_name.lower():
            for stat in stats:
                kwh_col = f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_{stat}'
                co2_col = f'elec_total_tonne_co2_saved_{scenario_name}_{years}yr_{stat}'
                
                if kwh_col in df_prep.columns:
                    # 3. Cast all new columns to float32
                    new_columns[co2_col] = ((df_prep[kwh_col] * years * elec_carbon_factor) / 1000).astype(dtype)
                    print(f"  Created: {co2_col}")
        
        # Add total CO2 savings column (gas + elec) for ranking (in tonnes)
        # <<< REVERTED to your original column names >>>
        gas_co2_mean = f'gas_{years}yr_tonne_co2_saved_{scenario_name}_mean'
        total_co2_col = f'{scenario_name}_total_co2_saved_{years}yr_mean'
        
        # Check if the column exists either in df_prep or new_columns
        if gas_co2_mean in new_columns:
            # 3. Cast all new columns to float32
            new_columns[total_co2_col] = new_columns[gas_co2_mean].copy().astype(dtype)
            
            # Add electricity CO2 if heat scenario
            if 'heat' in scenario_name.lower():
                # <<< REVERTED to your original column names >>>
                elec_co2_mean = f'elec_{years}yr_tonne_co2_saved_{scenario_name}_mean'
                if elec_co2_mean in new_columns:
                    # 4. Replace in-place add with a new assignment to ensure dtype
                    new_columns[total_co2_col] = (new_columns[total_co2_col] + new_columns[elec_co2_mean]).astype(dtype)
            
            print(f"  Created: {total_co2_col} (in tonnes)")
        
        # Ensure cost column exists (should already be there)
        cost_col = f'{scenario_name}_cost_{scenario_name}_mean'
        if cost_col not in df_prep.columns:
            print(f"  ⚠️  WARNING: Missing cost column: {cost_col}")

    # Add all new columns at once using concat - this avoids fragmentation
    print(f"\nAdding {len(new_columns)} new columns to dataframe (all as {dtype})...")
    
    # 5. Use df_prep (the renamed df) as the base for the concat
    new_cols_df = pd.DataFrame(new_columns, index=df_prep.index)
    df_prep_final = pd.concat([df_prep, new_cols_df], axis=1)
    
    print("\n✓ Data preparation complete (CO2 values in tonnes)")
    print(f"Total columns: {len(df_prep_final.columns)}")
    
    return df_prep_final

