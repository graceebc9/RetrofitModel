"""
Required Columns Analysis for Energy Retrofit Script
Extracts and organizes all columns needed from the dataframe, indexed by scenario.
"""

# ============================================================================
# STATIC COLUMNS (Required for all scenarios)
# ============================================================================

STATIC_COLUMNS = [
    # Building identifiers
    'upn',                          # Unique property reference number
    'premise_type',                 # Type of building (excludes 'Domestic outbuilding')
    'epistemic_run_id',             # Run identifier for uncertainty analysis
    
    # Decile and grouping columns
    'avg_gas_percentile',           # Gas usage decile for grouping
    
    # Building characteristics
    'conservation_area_bool',       # Conservation area flag
    'inferred_insulation_type',     # Type of insulation (for wall scenarios)
    
    # Wall intervention columns (for wall_installation scenario)
    'absolute_reduction_cavity',         # Cavity wall reduction potential
    'absolute_reduction_solid_internal', # Solid internal wall reduction potential
    'absolute_reduction_solid_external', # Solid external wall reduction potential
]


# ============================================================================
# SCENARIO-SPECIFIC COLUMN PATTERNS
# ============================================================================

def get_scenario_columns(scenario_name, measure_type, years=5):
    """
    Generate all required columns for a specific scenario.
    
    Parameters:
    -----------
    scenario_name : str
        Name of the scenario (e.g., 'heat_pump_only', 'wall_installation')
    measure_type : str
        Measure type identifier (e.g., 'heat_pump', 'joint_heat_ins_decay')
    years : int
        Number of years for projections (default: 5)
    
    Returns:
    --------
    dict : Dictionary with column categories and their required columns
    """
    
    columns = {
        # =====================================================================
        # COST COLUMNS (from clean_post_process output)
        # =====================================================================
        'cost': [
            f'{scenario_name}_cost_{scenario_name}_mean',
            f'{scenario_name}_cost_{scenario_name}_std',
        ],
        
        # =====================================================================
        # GAS CONSUMPTION COLUMNS (percentage reduction)
        # =====================================================================
        'gas_percentage': [
            f'{scenario_name}_{scenario_name}_gas_mean',
            f'{scenario_name}_{scenario_name}_gas_std',
        ],
        
        # =====================================================================
        # ELECTRICITY CONSUMPTION COLUMNS (percentage change)
        # =====================================================================
        'electricity_percentage': [
            f'{scenario_name}_{scenario_name}_electricity_mean',
            f'{scenario_name}_{scenario_name}_electricity_std',
        ],
        
        # =====================================================================
        # CARBON SAVINGS - GAS (absolute tonnes CO2)
        # =====================================================================
        'gas_carbon_savings': [
            f'gas_total_tonne_co2_saved_{scenario_name}_{years}yr_mean',
            f'gas_total_tonne_co2_saved_{scenario_name}_{years}yr_std',
        ],
        
        # =====================================================================
        # CARBON SAVINGS - ELECTRICITY (absolute tonnes CO2)
        # =====================================================================
        'elec_carbon_savings': [
            f'elec_total_tonne_co2_saved_{scenario_name}_{years}yr_mean',
            f'elec_total_tonne_co2_saved_{scenario_name}_{years}yr_std',
        ],
        
        # =====================================================================
        # NET CARBON SAVINGS (absolute tonnes CO2)
        # =====================================================================
        'net_carbon_savings': [
            f'total_tonne_co2_saved_{scenario_name}_{years}yr_mean',
            f'total_tonne_co2_saved_{scenario_name}_{years}yr_std',
        ],
        
        # =====================================================================
        # UNCERTAINTY ANALYSIS COLUMNS (from raw data before post-processing)
        # =====================================================================
        'uncertainty_raw': [
            f'gas_{years}yr_kg_co2_saved_{measure_type}_mean',
            f'gas_{years}yr_kg_co2_saved_{measure_type}_p50',
            f'gas_{years}yr_kg_co2_saved_{measure_type}_p95',
            f'gas_{years}yr_kg_co2_saved_{measure_type}_p5',
            f'gas_{years}yr_kg_co2_saved_{measure_type}_std',
        ],
    }
    
    return columns


# ============================================================================
# COLUMN REQUIREMENTS BY FUNCTION
# ============================================================================

FUNCTION_COLUMN_REQUIREMENTS = {
    'load_data': {
        'description': 'Initial data loading - all columns needed',
        'columns': 'ALL'
    },
    
    'analyze_uncertainty': {
        'description': 'Epistemic vs aleatoric uncertainty analysis',
        'required': [
            'upn',
            'epistemic_run_id',
            # Plus uncertainty_raw columns (see get_scenario_columns)
        ]
    },
    
    'clean_post_process': {
        'description': 'Processes energy and carbon metrics',
        'required': [
            'upn',
            'premise_type',
            'epistemic_run_id',
            # Plus uncertainty_raw columns (input)
            # Produces: cost, gas_percentage, electricity_percentage, carbon_savings (output)
        ]
    },
    
    'run_vis_new': {
        'description': 'Creates all visualizations',
        'required': [
            'premise_type',
            'avg_gas_percentile',
            'conservation_area_bool',
            'inferred_insulation_type',
            'epistemic_run_id',
            # Plus all processed columns from clean_post_process
        ]
    },
    
    'run_meta_portoflio': {
        'description': 'Portfolio-level metrics',
        'required': [
            'epistemic_run_id',
            # Plus carbon_savings columns
        ]
    },
}


# ============================================================================
# MINIMAL COLUMN SET FOR MEMORY OPTIMIZATION
# ============================================================================

def get_minimal_columns_for_scenario(scenario_name, measure_type, years=5):
    """
    Get the absolute minimum columns needed to process a scenario.
    Use this to reduce memory usage by only loading necessary columns.
    
    Returns:
    --------
    list : List of column names to load
    """
    
    # Start with static columns
    minimal_cols = STATIC_COLUMNS.copy()
    
    # Add scenario-specific columns
    scenario_cols = get_scenario_columns(scenario_name, measure_type, years)
    
    # Add uncertainty_raw columns (input to clean_post_process)
    minimal_cols.extend(scenario_cols['uncertainty_raw'])
    
    # Note: The processed columns (cost, gas_percentage, etc.) are CREATED
    # by clean_post_process, so we don't need them in the input
    
    return minimal_cols


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    # Example 1: Get columns for heat pump scenario
    print("="*80)
    print("EXAMPLE 1: Heat Pump Scenario Columns")
    print("="*80)
    scenario = 'heat_pump_only'
    measure = 'heat_pump'
    
    cols = get_scenario_columns(scenario, measure, years=5)
    print(f"\nScenario: {scenario}")
    print(f"Measure Type: {measure}")
    print("\nColumn Categories:")
    for category, column_list in cols.items():
        print(f"\n{category.upper()}:")
        for col in column_list:
            print(f"  - {col}")
    
    # Example 2: Get minimal columns for loading
    print("\n" + "="*80)
    print("EXAMPLE 2: Minimal Columns for Memory-Efficient Loading")
    print("="*80)
    minimal = get_minimal_columns_for_scenario(scenario, measure, years=5)
    print(f"\nTotal columns needed: {len(minimal)}")
    print("\nColumns to load:")
    for col in minimal:
        print(f"  - {col}")
    
    # Example 3: Multiple scenarios
    print("\n" + "="*80)
    print("EXAMPLE 3: Columns for Multiple Scenarios")
    print("="*80)
    
    scenarios_config = [
        ('heat_pump_only', 'heat_pump'),
        ('wall_installation', 'wall'),
        ('join_heat_ins_decay', 'joint_heat_ins_decay'),
    ]
    
    all_columns_needed = set(STATIC_COLUMNS)
    
    for scenario_name, measure_type in scenarios_config:
        scenario_cols = get_scenario_columns(scenario_name, measure_type, years=5)
        # Add uncertainty_raw columns (these are the input columns we need)
        all_columns_needed.update(scenario_cols['uncertainty_raw'])
    
    print(f"\nTotal unique columns needed for all scenarios: {len(all_columns_needed)}")
    print("\nAll columns:")
    for col in sorted(all_columns_needed):
        print(f"  - {col}")
    
    # Example 4: Create usecols parameter for pd.read_csv
    print("\n" + "="*80)
    print("EXAMPLE 4: Using with pandas read_csv")
    print("="*80)
    
    print("\nCode snippet:")
    print("""
# Load only necessary columns to save memory
scenarios_config = [
    ('heat_pump_only', 'heat_pump'),
    ('wall_installation', 'wall'),
]

# Get all columns needed
all_columns = set(STATIC_COLUMNS)
for scenario_name, measure_type in scenarios_config:
    minimal_cols = get_minimal_columns_for_scenario(scenario_name, measure_type, years=5)
    all_columns.update(minimal_cols)

# Load data with only necessary columns
df = pd.read_csv('data.csv', usecols=list(all_columns), low_memory=False)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    """)


# ============================================================================
# COLUMN VALIDATION FUNCTION
# ============================================================================

def validate_columns_present(df, scenario_name, measure_type, years=5):
    """
    Validate that all required columns are present in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to validate
    scenario_name : str
        Scenario name
    measure_type : str
        Measure type
    years : int
        Number of years
    
    Returns:
    --------
    tuple : (bool, list) - (all_present, missing_columns)
    """
    required = get_minimal_columns_for_scenario(scenario_name, measure_type, years)
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\n[WARNING] Missing {len(missing)} required columns:")
        for col in missing:
            print(f"  - {col}")
        return False, missing
    else:
        print(f"\n[SUCCESS] All {len(required)} required columns present")
        return True, []


# ============================================================================
# MEMORY-OPTIMIZED LOADING FUNCTION
# ============================================================================

def load_data_with_minimal_columns(file_pattern, scenarios_config, years=5):
    """
    Load data with only the columns needed for the specified scenarios.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern for CSV files
    scenarios_config : list of tuples
        [(scenario_name, measure_type), ...]
    years : int
        Number of years for projections
    
    Returns:
    --------
    pd.DataFrame : Loaded data with minimal columns
    """
    import glob
    
    # Determine all columns needed
    all_columns = set(STATIC_COLUMNS)
    for scenario_name, measure_type in scenarios_config:
        minimal_cols = get_minimal_columns_for_scenario(scenario_name, measure_type, years)
        all_columns.update(minimal_cols)
    
    columns_to_load = list(all_columns)
    
    print(f"\n[MEMORY OPT] Loading only {len(columns_to_load)} columns instead of all columns")
    print("This will significantly reduce memory usage!")
    
    # Load files
    files = glob.glob(file_pattern)
    print(f"\nFound {len(files)} files")
    
    dfs = []
    for i, f in enumerate(files, 1):
        # Try to load with specified columns
        try:
            df = pd.read_csv(f, usecols=columns_to_load, low_memory=False)
            dfs.append(df)
            if i % 10 == 0:
                print(f"Loaded {i}/{len(files)} files")
        except ValueError as e:
            # If columns don't exist in file, load without usecols
            print(f"[WARNING] File {f} doesn't have all columns, loading all columns")
            df = pd.read_csv(f, low_memory=False)
            # Select only columns that exist
            available_cols = [col for col in columns_to_load if col in df.columns]
            dfs.append(df[available_cols])
    
    result = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(result)} rows with {len(result.columns)} columns")
    print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return result