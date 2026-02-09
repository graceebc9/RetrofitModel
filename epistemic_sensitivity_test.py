"""
Module: sensitivity_test_fixed.py

Corrected sensitivity analysis for epistemic factors.

KEY FIX: Uses the SAME baseline LHS sample for all runs, only overwriting 
the fixed factor for each test. This isolates the effect of each factor properly.

Previous bug: Generated new LHS samples for each run, making comparisons invalid.

-- added random seed to sampler 
-- added skip_existing check to avoid re-running completed analyses

Usage:
    python epistemic_sensitivity_test.py
    python epistemic_sensitivity_test.py --all
    python epistemic_sensitivity_test.py --n-postcodes 5
    python epistemic_sensitivity_test.py --force  # Force re-run even if results exist
"""
import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================

# Test configuration
RANDOM_SEED_OUTER = 42

SCENARIOS = ['wall_installation', 'loft_installation', 'heat_pump_only']

# Paths 
PC_SHP_PATH = '/rds/user/gb669/hpc-work/energy_map/data/postcode_polygons/codepoint-poly_5267291'
BUILDING_PATH = '/rds/user/gb669/hpc-work/energy_map/data/building_files/UKBuildings_Edition_15_new_format_upn.gpkg'
location_input_data_folder = '/home/gb669/rds/hpc-work/energy_map/data/input_data'
onsud_path_base = '/home/gb669/rds/hpc-work/energy_map/data/onsud_files/Data'

GAS_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_gas_2022.csv'
ELEC_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_all_meters_electricity_2022.csv'

 
# Output directory
OUTPUT_DIR = '5_sensitivity_results'

 
# Key metrics for sensitivity ranking
KEY_METRICS = [
    # Cost metrics
    'wall_installation_cost_wall_installation_mean',
    'loft_installation_cost_loft_installation_mean',
    'heat_pump_only_cost_heat_pump_only_mean',
    # Energy/CO2 metrics
    'wall_installation_total_energy_abs_co2_ton_samples_wall_installation_mean',
    'loft_installation_total_energy_abs_co2_ton_samples_loft_installation_mean',
    'heat_pump_only_total_energy_abs_co2_ton_samples_heat_pump_only_mean',
    # Gas percentage savings (if available)
    'wall_installation_gas_perc_wall_installation_mean',
    'loft_installation_gas_perc_loft_installation_mean',
    'heat_pump_only_gas_perc_heat_pump_only_mean',
]

# Central/default values for each factor
# SYNCHRONIZED with RetrofitEpistemic.py - only includes factors that are actually sampled
FACTOR_DEFAULTS = {
    'time_scale_bias': 1.0,
    'decile_misclassification_bias': 0.0,
    'solid_wall_internal_improvement_factor': 0.10,
    'solid_wall_external_improvement_factor': 0.20,
    'age_band_multipliers_uncertainty': 1.0,
    'cost_scenario': 'central',
    'area_based_choice': 'mode',
        'flat_fp_mean': 55,
    'flat_fp_std': 8,
    'flat_eff_mean': 0.75,
    'flat_eff_std': 0.05,
}

# ========================================
# IMPORTS (after config)
# ========================================

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.RetrofitEpistemic import generate_epistemic_scenarios_lhs
from src.RetrofitScenarioGenerator2DMC import RetrofitScenarioGenerator2DMC
from src.RetrofitModel2D import RetrofitModel2D
from src.RetrofitConfig import RetrofitConfig
from src.postcode_utils import load_ids_from_file, load_onsud_data, find_data_pc_joint
from src.conservation import load_conservation_shapefile
from src.RetrofitDownscale import load_scaled_gas_elec
from src.pre_process_buildings import pre_process_building_data
from src.retrofit_calc2D import get_conservation_area


# ========================================
# ARGUMENT PARSING
# ========================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Epistemic factor sensitivity analysis (corrected)')
    parser.add_argument(
        '--batch', 
        type=str, 
        default='batches/NE/batch_130.txt',
        help='Path to batch file (e.g., batches/NE/batch_10.txt)'
    )
    parser.add_argument(
        '--batch_name', 
        type=str, 
        default='130',
        help='Path to batch file (e.g., batches/NE/batch_10.txt)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='sensitivity_results',
        help='Base output directory for results'
    )
    parser.add_argument(
        '--n-postcodes',
        type=int,
        default=-1,
        help='Number of postcodes to process from batch (default: 1, use -1 for all)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all postcodes in batch (equivalent to --n-postcodes -1)'
    )
    parser.add_argument(
        '--n-epistemic',
        type=int,
        default=50,
        help='Number of epistemic runs (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if results already exist'
    )
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.n_postcodes = -1
    
    return args


# ========================================
# LOGGING SETUP
# ========================================

def setup_logging_for_batch(output_dir: str, batch_label: str, timestamp: str) -> logging.Logger:
    """Setup logging to both console and file."""
    
    logger = logging.getLogger(f'sensitivity_test_{batch_label}')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    log_file = os.path.join(output_dir, f'sensitivity_log_{batch_label}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# ========================================
# DATA LOADING
# ========================================

def load_test_data(batch_path: str, n_postcodes: int, logger: logging.Logger):
    """Load all required data for testing."""
    logger.info(f"Loading test data for batch: {batch_path}")
    
    # Derive ONSUD path from batch path
    batch_dir = os.path.dirname(batch_path)
    batch_num = os.path.basename(batch_path).replace('batch_', '').replace('.txt', '')
    onsud_path = os.path.join(batch_dir, f'onsud_{batch_num}.csv')
    
    logger.info(f"Using ONSUD path: {onsud_path}")
    
    # Conservation areas
    conservation_data = load_conservation_shapefile(
        path=f'{root_dir}/src/global_avs/Conservation_Areas_-5503574965118299320'
    )
    
    # ONSUD data
    onsud_data = load_onsud_data(onsud_path, PC_SHP_PATH)
    
    # Scaled gas/elec
    scaled_gas_elec_data = load_scaled_gas_elec()
    
    # Gas deciles
    gas_deciles = pd.read_csv(f'{root_dir}/src/global_avs/neb_unfil_final_gas_deciles.csv')
    
    # Postcodes
    all_postcodes = load_ids_from_file(batch_path)
    if n_postcodes == -1:
        postcodes = all_postcodes
    else:
        postcodes = all_postcodes[:n_postcodes]
    logger.info(f"Testing with {len(postcodes)} postcode(s): {postcodes[:5]}{'...' if len(postcodes) > 5 else ''}")
    
    return {
        'conservation_data': conservation_data,
        'onsud_data': onsud_data,
        'scaled_gas_elec_data': scaled_gas_elec_data,
        'gas_deciles': gas_deciles,
        'postcodes': postcodes,
        'batch_path': batch_path,
    }


# ========================================
# BUILDING DATA PREPARATION
# ========================================
 

def prepare_building_data(pc: str, data: dict, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Prepare building data for a single postcode."""
    
    energy_columns = [
        'gas_scaled_scaled_area_max', 'elec_scaled_scaled_area_max',
        'gas_scaled_scaled_area_min', 'elec_scaled_scaled_area_min',
        'gas_scaled_scaled_area_mode', 'elec_scaled_scaled_area_mode',
    ]
    
    pc = pc.strip()
    
    # Get building data
    uprn_match = find_data_pc_joint(pc, data['onsud_data'], input_gpk=BUILDING_PATH)
    if uprn_match is None or uprn_match.empty:
        logger.warning(f"No building data for {pc}")
        return None
    
    # Add conservation area
    uprn_match = get_conservation_area(uprn_match, data['conservation_data'])
    
    # Pre-process
    building_data = pre_process_building_data(uprn_match)
    
    # Add gas decile
    pc_decile = data['gas_deciles'][data['gas_deciles']['postcode'] == pc]
    if pc_decile.empty:
        logger.warning(f"No gas decile for {pc}")
        return None
    building_data['avg_gas_percentile'] = pc_decile['avg_gas_decile'].values[0]
    
    # Merge energy data
    energy = data['scaled_gas_elec_data'][data['scaled_gas_elec_data']['postcode'] == pc]
    building_data = building_data.merge(energy, on='upn')
    
    # Check energy columns
    for col in energy_columns:
        if col not in building_data.columns:
            logger.warning(f"Missing energy column: {col}")
            return None
    
    building_data['postcode'] = pc
    
    #filter out unknown typolgoyies 
    building_data = building_data[building_data['premise_type']!='Unknown'] 
    
    excl_typs = [ 'Tall flats 6-15 storeys', 
       'Very tall point block flats', 
       'Domestic outbuilding',
       'all_unknown_typology',
         'Medium height flats 5-6 storeys', 
       ] 
    building_data = building_data[~building_data['premise_type'].isin(excl_typs) ] 
    building_data = building_data[~building_data['premise_type'].isna()] 
    return building_data


# ========================================
# MODEL RUNNER (with fixed epistemic df)
# ========================================

def create_sampler_from_df(epistemic_df: pd.DataFrame) -> Callable:
    """
    Create a sampler function that returns a pre-defined DataFrame.
    This ensures the SAME epistemic scenarios are used across runs.
    """
    def fixed_sampler(n_runs: int, random_seed: int ) -> pd.DataFrame:
        # Ignore n_runs, return the pre-defined df
        return epistemic_df.copy()
    return fixed_sampler


def run_model_with_epistemic_df(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    epistemic_df: pd.DataFrame,
    n_epistemic_runs: int,
    logger: logging.Logger,
    region: str = 'NE',
) -> Optional[pd.DataFrame]:
    """
    Run model with a specific epistemic DataFrame.
    
    KEY: Uses a pre-defined epistemic_df instead of generating new samples.
    """
    
    # Create sampler that returns the fixed df
    fixed_sampler = create_sampler_from_df(epistemic_df)
    
    scenario_generator = RetrofitScenarioGenerator2DMC(
        n_epistemic_runs=n_epistemic_runs,
        epistemic_sampler=fixed_sampler
    )
    
    RetrofitModel2D.retrofit_config = retrofit_config
    
    results = scenario_generator.process_dataframe_scenarios(
        df=building_data.copy(),
        region=region,
        model_class=RetrofitModel2D,
        random_seed=RANDOM_SEED_OUTER,
        scenarios=SCENARIOS,
    )
    
    if isinstance(results, dict) and 'error' in results:
        logger.warning(f"Error: {results['error']}")
        return None
    
    return results

# ========================================
# DIAGNOSTICS
# ========================================

def diagnose_factor_usage(
    building_data: pd.DataFrame,
    baseline_epistemic_df: pd.DataFrame,
    factor_name: str,
    logger: logging.Logger
):
    """
    Diagnose why a factor might not be affecting outputs.
    """
    logger.info("=" * 70)
    logger.info(f"DIAGNOSTIC: {factor_name}")
    logger.info("=" * 70)
    
    # 1. Check if factor varies in epistemic sample
    factor_values = baseline_epistemic_df[factor_name]
    logger.info(f"Factor range: {factor_values.min():.4f} to {factor_values.max():.4f}")
    logger.info(f"Factor mean: {factor_values.mean():.4f}, std: {factor_values.std():.4f}")
    logger.info(f"Factor unique values: {factor_values.nunique()}")
    
    if factor_values.std() < 0.001:
        logger.warning(f"⚠️  Factor has very low variance - might be effectively constant!")
    
 
    return True


def add_factor_diagnostics_to_main(logger: logging.Logger):
    """Add this call in your main function after loading data"""
    
    # After preparing building data and baseline epistemic sample:
    problematic_factors = [
        'solid_wall_internal_improvement_factor',
        'solid_wall_external_improvement_factor',
    ]
    
    for factor in problematic_factors:
        diagnose_factor_usage(
            combined_building_data,
            baseline_epistemic_df,
            factor,
            logger
        )
        
        


# ========================================
# METRICS
# ========================================

def identify_output_metrics(df: pd.DataFrame, logger: logging.Logger) -> List[str]:
    """Identify key output metrics based on known column patterns."""
    
    metric_cols = []
    
    for scenario in SCENARIOS:
        patterns = [
            # Cost
            f'{scenario}_cost_{scenario}_mean',
            f'{scenario}_cost_{scenario}_std',
            # Energy/CO2
            f'{scenario}_total_energy_abs_co2_ton_samples_{scenario}_mean',
            f'{scenario}_total_energy_abs_co2_ton_samples_{scenario}_std',
            # CAPEX per tCO2
            f'{scenario}_capex_per_net_ton_co2_{scenario}_mean',
            f'{scenario}_capex_per_net_ton_co2_{scenario}_std',
            # Gas percentage savings
            f'{scenario}_gas_perc_{scenario}_mean',
            f'{scenario}_gas_perc_{scenario}_std',
        ]
        
        for pattern in patterns:
            if pattern in df.columns:
                metric_cols.append(pattern)
    
    # Fallback to auto-detection
    if not metric_cols:
        logger.warning("No known metric patterns found, falling back to auto-detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['epistemic_run_id', 'epistemic__', 'upn', 'index']
        
        for col in numeric_cols:
            if not any(pattern in col for pattern in exclude_patterns):
                if df[col].std() > 0:
                    metric_cols.append(col)
    
    return metric_cols


def compute_variance_summary(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, float]:
    """Compute variance metrics for output columns."""
    summary = {}
    
    for col in metric_cols:
        if col in df.columns:
            summary[f'{col}__var'] = df[col].var()
            summary[f'{col}__std'] = df[col].std()
            summary[f'{col}__cv'] = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else np.nan
    
    return summary


# ========================================
# VALIDATION
# ========================================

def validate_factor_defaults(epistemic_df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Validate that FACTOR_DEFAULTS matches the columns in the epistemic DataFrame.
    Returns True if valid, False otherwise.
    """
    epistemic_cols = set(epistemic_df.columns)
    default_factors = set(FACTOR_DEFAULTS.keys())
    
    missing_in_defaults = epistemic_cols - default_factors
    missing_in_epistemic = default_factors - epistemic_cols
    
    is_valid = True
    
    if missing_in_defaults:
        logger.warning(f"Factors in epistemic sample but NOT in FACTOR_DEFAULTS: {missing_in_defaults}")
        logger.warning("These factors will NOT be tested in sensitivity analysis!")
        is_valid = False
    
    if missing_in_epistemic:
        logger.error(f"Factors in FACTOR_DEFAULTS but NOT in epistemic sample: {missing_in_epistemic}")
        logger.error("These factors cannot be tested - they don't exist in the model!")
        is_valid = False
    
    if is_valid:
        logger.info(f"✓ FACTOR_DEFAULTS synchronized with epistemic sampler ({len(default_factors)} factors)")
    
    return is_valid


# ========================================
# CHECK FOR EXISTING RESULTS
# ========================================

def find_existing_output_dir(output_base_dir: str, batch_label: str) -> Optional[str]:
    """
    Find an existing output directory for this batch (for resuming partial runs).
    
    Returns the path to existing results directory if found, None otherwise.
    """
    if not os.path.exists(output_base_dir):
        return None
    
    # Look for any existing output directories for this batch
    existing_dirs = [d for d in os.listdir(output_base_dir) 
                    if d.startswith(batch_label) and os.path.isdir(os.path.join(output_base_dir, d))]
    
    if not existing_dirs:
        return None
    
    # Sort by timestamp (newest first) and return the most recent
    existing_dirs.sort(reverse=True)
    return os.path.join(output_base_dir, existing_dirs[0])


def check_completed_runs(output_dir: str) -> Dict[str, bool]:
    """
    Check which runs have already been completed in the output directory.
    
    Returns a dict mapping configuration names to completion status.
    """
    completed = {}
    
    if not os.path.exists(output_dir):
        return completed
    
    # Check for baseline
    baseline_file = os.path.join(output_dir, 'baseline_results.csv')
    completed['baseline'] = os.path.exists(baseline_file)
    
    # Check for each factor
    for factor in FACTOR_DEFAULTS.keys():
        factor_file = os.path.join(output_dir, f'fixed_{factor}_results.csv')
        completed[factor] = os.path.exists(factor_file)
    
    return completed


def is_fully_complete(output_dir: str) -> bool:
    """Check if all runs are complete (baseline + all factors)."""
    completed = check_completed_runs(output_dir)
    
    if not completed.get('baseline', False):
        return False
    
    for factor in FACTOR_DEFAULTS.keys():
        if not completed.get(factor, False):
            return False
    
    return True


# ========================================
# MAIN SENSITIVITY TEST (CORRECTED)
# ========================================

def run_sensitivity_test(
    batch_path: str,
    output_base_dir: str,
    n_postcodes: int,
    n_epistemic_runs: int = 20,
    random_seed: int = 42,
    skip_existing: bool = True,
):
    """
    Run sensitivity analysis with CORRECTED methodology.
    
    KEY FIX: Generate ONE baseline LHS sample, then for each factor test,
    use the SAME sample with only that factor overwritten.
    
    Args:
        batch_path: Path to batch file containing postcodes.
        output_base_dir: Base directory for output results.
        n_postcodes: Number of postcodes to process (-1 for all).
        n_epistemic_runs: Number of epistemic runs.
        random_seed: Random seed for reproducibility.
        skip_existing: If True, skip individual runs that are already complete.
    """
    
    # Update global random seed
    global RANDOM_SEED_OUTER
    RANDOM_SEED_OUTER = random_seed
    
    batch_label = os.path.basename(batch_path).replace('.txt', '')
    
    # Check for existing output directory to resume
    existing_dir = find_existing_output_dir(output_base_dir, batch_label)
    
    if existing_dir and skip_existing:
        # Check if fully complete
        if is_fully_complete(existing_dir):
            print(f"=" * 70)
            print(f"SKIPPING: All results already exist for batch '{batch_label}'")
            print(f"Existing results path: {existing_dir}")
            print(f"Use --force flag to re-run and overwrite.")
            print(f"=" * 70)
            
            # Load and return existing results
            summary_file = os.path.join(existing_dir, 'sensitivity_summary.csv')
            ranking_file = os.path.join(existing_dir, 'sensitivity_ranking.csv')
            
            summary_df = pd.read_csv(summary_file)
            ranking_df = pd.read_csv(ranking_file)
            
            return summary_df, ranking_df
        else:
            # Partial completion - will resume
            output_dir = existing_dir
            completed_runs = check_completed_runs(output_dir)
            n_completed = sum(completed_runs.values())
            n_total = 1 + len(FACTOR_DEFAULTS)  # baseline + factors
            print(f"=" * 70)
            print(f"RESUMING: Found partial results for batch '{batch_label}'")
            print(f"Completed: {n_completed}/{n_total} runs")
            print(f"Output directory: {output_dir}")
            print(f"=" * 70)
            
            # Extract timestamp from existing dir for logging
            timestamp = existing_dir.split('_')[-2] + '_' + existing_dir.split('_')[-1]
    else:
        # Fresh run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_base_dir, f'{batch_label}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        completed_runs = {}
    
    # Setup logging
    logger = setup_logging_for_batch(output_dir, batch_label, timestamp)
    
    logger.info("=" * 70)
    logger.info("EPISTEMIC FACTOR SENSITIVITY ANALYSIS (CORRECTED METHODOLOGY)")
    logger.info("=" * 70)
    logger.info(f"Batch: {batch_path}")
    logger.info(f"N epistemic runs: {n_epistemic_runs}")
    logger.info(f"N postcodes: {n_postcodes if n_postcodes != -1 else 'all'}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Skip existing: {skip_existing}")
    if completed_runs:
        logger.info(f"Resuming with {sum(completed_runs.values())} completed runs")
    logger.info("")
    logger.info("METHODOLOGY: Same baseline LHS sample used for all runs.")
    logger.info("Each factor test overwrites ONLY that factor in the baseline sample.")
    logger.info("=" * 70)
    
    # Log factors being tested
    logger.info(f"Factors to test ({len(FACTOR_DEFAULTS)}):")
    for factor, default in FACTOR_DEFAULTS.items():
        status = "✓ DONE" if completed_runs.get(factor, False) else "pending"
        logger.info(f"  - {factor}: default = {default} [{status}]")
    logger.info("")
    
    # Load data
    data = load_test_data(batch_path, n_postcodes, logger)
    
    # Prepare building data for all postcodes
    logger.info("Preparing building data...")
    all_building_data = []
    for pc in data['postcodes']:
        bd = prepare_building_data(pc, data, logger)
        if bd is not None:
            all_building_data.append(bd)
    
    if not all_building_data:
        logger.error("No valid building data found")
        return None, None
    
    combined_building_data = pd.concat(all_building_data, ignore_index=True)
    logger.info(f"Combined building data shape: {combined_building_data.shape}")
    
    
    # Retrofit config
    retrofit_config = RetrofitConfig(
        existing_intervention_probs={
            'loft_insulation': 0,
            'floor_insulation': 0,
            'window_upgrades': 0,
            'roof_scaling_factor': 0.8,
        }
    )
    
    # =========================================================
    # KEY FIX: Generate ONE baseline epistemic sample with random_seed
    # (or load existing one if resuming)
    # =========================================================
    baseline_epistemic_path = f'{output_dir}/baseline_epistemic_scenarios.csv'
    
    if os.path.exists(baseline_epistemic_path):
        logger.info("Loading existing baseline epistemic sample...")
        baseline_epistemic_df = pd.read_csv(baseline_epistemic_path)
    else:
        logger.info("Generating baseline epistemic sample (used for ALL runs)...")
        baseline_epistemic_df = generate_epistemic_scenarios_lhs(
            N_epistemic_runs=n_epistemic_runs,
            random_seed=random_seed,
        )
        baseline_epistemic_df.to_csv(baseline_epistemic_path, index=False)
            
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING DIAGNOSTICS ON ZERO-SENSITIVITY FACTORS")
        logger.info("=" * 70)
        
        problematic_factors = [
            'solid_wall_internal_improvement_factor',
            'solid_wall_external_improvement_factor',
        ]
        
        for factor in problematic_factors:
            if factor in baseline_epistemic_df.columns:
                diagnose_factor_usage(
                    combined_building_data,
                    baseline_epistemic_df,
                    factor,
                    logger
                )
    
    # Validate factor synchronization
    validate_factor_defaults(baseline_epistemic_df, logger)
    
    logger.info(f"Baseline epistemic scenarios:\n{baseline_epistemic_df.to_string()}")
    
    results_summary = []
    metric_cols = None
    
    # =========================================================
    # BASELINE RUN
    # =========================================================
    baseline_file = f'{output_dir}/baseline_results.csv'
    
    if skip_existing and completed_runs.get('baseline', False):
        logger.info("=" * 60)
        logger.info("SKIPPING BASELINE (already complete)")
        logger.info("=" * 60)
        baseline_df = pd.read_csv(baseline_file)
        logger.info(f"Loaded baseline: {len(baseline_df)} rows")
    else:
        logger.info("=" * 60)
        logger.info("Running BASELINE (all factors vary)")
        logger.info("=" * 60)
        
        baseline_df = run_model_with_epistemic_df(
            building_data=combined_building_data,
            retrofit_config=retrofit_config,
            epistemic_df=baseline_epistemic_df,
            n_epistemic_runs=n_epistemic_runs,
            logger=logger,
        )
        
        if baseline_df is not None:
            baseline_df.to_csv(baseline_file, index=False)
            logger.info(f"Baseline complete: {len(baseline_df)} rows")
        else:
            logger.error("Baseline run failed")
            return None, None
    
    # Identify metrics
    metric_cols = identify_output_metrics(baseline_df, logger)
    logger.info(f"Detected {len(metric_cols)} output metrics")
    logger.debug(f"Metrics: {metric_cols}")
    
    baseline_summary = compute_variance_summary(baseline_df, metric_cols)
    baseline_summary['configuration'] = 'baseline'
    baseline_summary['fixed_factor'] = None
    results_summary.append(baseline_summary)
    
    # =========================================================
    # FIXED FACTOR RUNS (using SAME baseline sample)
    # =========================================================
    for factor, default_value in FACTOR_DEFAULTS.items():
        factor_file = f'{output_dir}/fixed_{factor}_results.csv'
        
        if skip_existing and completed_runs.get(factor, False):
            logger.info("=" * 60)
            logger.info(f"SKIPPING {factor} (already complete)")
            logger.info("=" * 60)
            fixed_df = pd.read_csv(factor_file)
            logger.info(f"Loaded {factor}: {len(fixed_df)} rows")
        else:
            logger.info("=" * 60)
            logger.info(f"Running with {factor} FIXED to {default_value}")
            logger.info("=" * 60)
            
            # Verify factor exists in baseline
            if factor not in baseline_epistemic_df.columns:
                logger.warning(f"Factor '{factor}' not in epistemic sample - skipping")
                continue
            
            # KEY: Copy baseline and overwrite ONLY this factor
            fixed_epistemic_df = baseline_epistemic_df.copy()
            fixed_epistemic_df[factor] = default_value
            
            logger.debug(f"Fixed epistemic df:\n{fixed_epistemic_df[[factor]].to_string()}")
            
            fixed_df = run_model_with_epistemic_df(
                building_data=combined_building_data,
                retrofit_config=retrofit_config,
                epistemic_df=fixed_epistemic_df,
                n_epistemic_runs=n_epistemic_runs,
                logger=logger,
            )
            
            if fixed_df is not None:
                fixed_df.to_csv(factor_file, index=False)
                logger.info(f"Completed: {factor}")
            else:
                logger.warning(f"Failed: {factor}")
                continue
        
        fixed_summary = compute_variance_summary(fixed_df, metric_cols)
        fixed_summary['configuration'] = f'fixed_{factor}'
        fixed_summary['fixed_factor'] = factor
        results_summary.append(fixed_summary)
    
    # =========================================================
    # COMPILE SUMMARY
    # =========================================================
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f'{output_dir}/sensitivity_summary.csv', index=False)
    
    # =========================================================
    # COMPUTE SENSITIVITY RANKING
    # =========================================================
    all_rankings = []
    
    for key_metric in KEY_METRICS:
        var_col = f'{key_metric}__var'
        
        if var_col not in summary_df.columns:
            continue
        
        baseline_var = summary_df[summary_df['configuration'] == 'baseline'][var_col].values[0]
        
        if baseline_var == 0 or np.isnan(baseline_var):
            logger.warning(f"Baseline variance is 0 or NaN for {key_metric}")
            continue
        
        for _, row in summary_df[summary_df['configuration'] != 'baseline'].iterrows():
            factor = row['fixed_factor']
            fixed_var = row[var_col]
            var_reduction = baseline_var - fixed_var
            var_reduction_pct = (var_reduction / baseline_var) * 100
            
            all_rankings.append({
                'metric': key_metric,
                'factor': factor,
                'baseline_var': baseline_var,
                'fixed_var': fixed_var,
                'var_reduction': var_reduction,
                'var_reduction_pct': var_reduction_pct,
            })
    
    ranking_df = pd.DataFrame(all_rankings)
    ranking_df.to_csv(f'{output_dir}/sensitivity_ranking.csv', index=False)
    
    # =========================================================
    # OUTPUT RESULTS
    # =========================================================
    output_lines = []
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append(f"SENSITIVITY RANKING BY METRIC (Batch: {batch_label})")
    output_lines.append("=" * 70)
    output_lines.append("Higher % = factor contributes more to output uncertainty")
    output_lines.append("(Negative values should NOT occur with corrected methodology)")
    output_lines.append("")
    
    for key_metric in KEY_METRICS:
        metric_ranking = ranking_df[ranking_df['metric'] == key_metric].sort_values(
            'var_reduction_pct', ascending=False
        )
        
        if metric_ranking.empty:
            continue
        
        # Extract scenario and metric type
        if '_cost_' in key_metric:
            scenario = key_metric.split('_cost_')[0]
            metric_type = 'cost'
        elif '_total_energy_abs_co2_ton_samples_' in key_metric:
            scenario = key_metric.split('_total_energy_abs_co2_ton_samples_')[0]
            metric_type = 'energy/CO2'
        elif '_gas_perc_' in key_metric:
            scenario = key_metric.split('_gas_perc_')[0]
            metric_type = 'gas %'
        else:
            scenario = key_metric
            metric_type = ''
        
        output_lines.append(f"\n{scenario.upper()} ({metric_type})")
        output_lines.append("-" * 50)
        output_lines.append(f"{'Factor':<45} {'Var Reduction %':>12}")
        output_lines.append("-" * 50)
        
        for _, row in metric_ranking.iterrows():
            pct = row['var_reduction_pct']
            flag = " ⚠️" if pct < 0 else ""
            output_lines.append(f"{row['factor']:<45} {pct:>11.1f}%{flag}")
    
    # Average ranking
    if not ranking_df.empty:
        avg_ranking = ranking_df.groupby('factor')['var_reduction_pct'].mean().sort_values(ascending=False)
        
        output_lines.append("\n" + "=" * 70)
        output_lines.append("AVERAGE SENSITIVITY ACROSS ALL METRICS")
        output_lines.append("=" * 70)
        output_lines.append(f"{'Factor':<45} {'Avg Var Reduction %':>12}")
        output_lines.append("-" * 57)
        for factor, pct in avg_ranking.items():
            flag = " ⚠️" if pct < 0 else ""
            output_lines.append(f"{factor:<45} {pct:>11.1f}%{flag}")
        
        avg_ranking.to_csv(f'{output_dir}/sensitivity_avg_ranking.csv')
    
    # Print and log
    for line in output_lines:
        print(line)
        logger.info(line)
    
    logger.info(f"\nAll results saved to: {output_dir}/")
    
    return summary_df, ranking_df


# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Running sensitivity test (CORRECTED) for batch: {args.batch}")
    print(f"Output directory: {args.output}")
    print(f"N postcodes: {args.n_postcodes if args.n_postcodes != -1 else 'all'}")
    print(f"N epistemic runs: {args.n_epistemic}")
    print(f"Random seed: {args.seed}")
    print(f"Skip existing: {not args.force}")
    
    output_base_dir= os.path.join(args.output, args.batch_name) 
    run_sensitivity_test(
        batch_path=args.batch,
        output_base_dir=output_base_dir, 
        n_postcodes=args.n_postcodes,
        n_epistemic_runs=args.n_epistemic,
        random_seed=args.seed,
        skip_existing=not args.force,
    )