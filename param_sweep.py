"""
Module: wall_improvement_sweep.py

Parameter sweep to identify break-even solid wall improvement factors.

Tests a range of improvement factors for internal and external wall insulation
separately, measuring impact on £/tCO2 (cost per ton CO2 saved).

Usage:
    # Single batch, 1 postcode (quick test)
    python wall_improvement_sweep.py
    
    # All postcodes in single batch
    python wall_improvement_sweep.py --all
    
    # Specific number of postcodes
    python wall_improvement_sweep.py --n-postcodes 10
    
    # Run across ALL batches
    python wall_improvement_sweep.py --all-batches --all
    
    # All batches with custom batch file
    python wall_improvement_sweep.py --all-batches --batch-file my_batches.txt --all
    
    # SAMPLING OPTIONS (recommended for large runs):
    
    # Sample 3 random postcodes from each batch, max 500 buildings total
    python wall_improvement_sweep.py --all-batches --sample-per-batch 3 --max-buildings 500
    
    # Sample 5 random postcodes from each batch, max 1000 buildings
    python wall_improvement_sweep.py --all-batches --sample-per-batch 5 --max-buildings 1000
    
    # All batches, but cap at 2000 buildings total
    python wall_improvement_sweep.py --all-batches --all --max-buildings 2000
    
    # Set random seed for reproducibility
    python wall_improvement_sweep.py --all-batches --sample-per-batch 3 --max-buildings 500 --seed 123
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================

N_EPISTEMIC_RUNS = 5
RANDOM_SEED_OUTER = 42

# Only test wall_installation scenario
SCENARIOS = ['wall_installation']


# Paths 
PC_SHP_PATH = '/rds/user/gb669/hpc-work/energy_map/data/postcode_polygons/codepoint-poly_5267291'
BUILDING_PATH = '/rds/user/gb669/hpc-work/energy_map/data/building_files/UKBuildings_Edition_15_new_format_upn.gpkg'
location_input_data_folder = '/home/gb669/rds/hpc-work/energy_map/data/input_data'
onsud_path_base = '/home/gb669/rds/hpc-work/energy_map/data/onsud_files/Data'

GAS_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_gas_2022.csv'
ELEC_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_all_meters_electricity_2022.csv'

# Parameter sweep ranges
# Current literature values: internal ~0.1, external ~0.2
# These are IMPROVEMENT FACTORS over cavity wall baseline
# 
# Cavity wall savings (from literature):
#   - Percentile 5: ~10% gas savings
#   - Percentile 9: ~17% gas savings
#
# With improvement factor X, solid wall savings = cavity * (1 + X)
#   - Factor 0.2 → 17% * 1.2 = 20% savings
#   - Factor 0.5 → 17% * 1.5 = 26% savings  
#   - Factor 1.0 → 17% * 2.0 = 34% savings
#   - Factor 1.5 → 17% * 2.5 = 43% savings
#   - Factor 2.0 → 17% * 3.0 = 52% savings
#
# Extended ranges to find break-even

INTERNAL_WALL_FACTORS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
EXTERNAL_WALL_FACTORS = [0.20, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]

# Reference cavity wall values for logging (percentile 5 as typical)
CAVITY_WALL_REFERENCE = {
    'percentile': 5,
    'mean': -0.0975109,  # ~10% gas savings
}

# Key metric for comparison
COST_PER_TCO2_METRIC = 'wall_installation_capex_per_net_ton_co2_wall_installation_mean'

# Threshold for "attractive" (£/tCO2) - adjust based on your policy context
# UK carbon price ~£50-100/tCO2, social cost of carbon ~£250/tCO2
ATTRACTIVE_THRESHOLDS = [50, 100, 150, 200, 250, 300, 1000]

# ========================================
# IMPORTS
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
    parser = argparse.ArgumentParser(description='Wall improvement factor parameter sweep')
    parser.add_argument(
        '--batch', 
        type=str, 
        default='batches/NE/batch_10.txt',
        help='Path to batch file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='wall_sweep_results',
        help='Base output directory'
    )
    parser.add_argument(
        '--n-postcodes',
        type=int,
        default=1,
        help='Number of postcodes per batch (-1 for all)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all postcodes in batch'
    )
    parser.add_argument(
        '--all-batches',
        action='store_true',
        help='Run across ALL batches listed in batch_paths.txt'
    )
    parser.add_argument(
        '--batch-file',
        type=str,
        default='batch_paths.txt',
        help='File containing list of batch paths (used with --all-batches)'
    )
    parser.add_argument(
        '--max-buildings',
        type=int,
        default=None,
        help='Maximum total buildings to sample across all batches (default: no limit)'
    )
    parser.add_argument(
        '--sample-per-batch',
        type=int,
        default=None,
        help='Randomly sample N postcodes from each batch (overrides --n-postcodes)'
    )
    parser.add_argument(
        '--n-epistemic',
        type=int,
        default=5,
        help='Number of epistemic runs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    args = parser.parse_args()
    
    if args.all:
        args.n_postcodes = -1
    
    # sample-per-batch overrides n-postcodes
    if args.sample_per_batch is not None:
        args.n_postcodes = args.sample_per_batch
        args.random_sample = True
    else:
        args.random_sample = False
    
    return args


# ========================================
# LOGGING
# ========================================

def setup_logging(output_dir: str, timestamp: str) -> logging.Logger:
    logger = logging.getLogger('wall_sweep')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # Console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    # File
    file_handler = logging.FileHandler(f'{output_dir}/wall_sweep_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


def load_all_batch_paths(batch_file: str) -> List[str]:
    """Load all batch paths from batch_paths.txt file."""
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"Batch file not found: {batch_file}")
    
    with open(batch_file, 'r') as f:
        batch_paths = [line.strip() for line in f if line.strip()]
    
    # Remove duplicates while preserving order
    batch_paths = list(dict.fromkeys(batch_paths))
    
    return batch_paths


# ========================================
# DATA LOADING
# ========================================

def load_test_data(batch_path: str, n_postcodes: int, logger: logging.Logger, 
                   random_sample: bool = False, seed: int = 42):
    logger.info(f"Loading data for batch: {batch_path}")
    
    batch_dir = os.path.dirname(batch_path)
    batch_num = os.path.basename(batch_path).replace('batch_', '').replace('.txt', '')
    onsud_path = os.path.join(batch_dir, f'onsud_{batch_num}.csv')
    
    # Extract region from batch path (e.g., 'batches/NE/batch_10.txt' -> 'NE')
    region = os.path.basename(batch_dir)
    logger.info(f"Detected region: {region}")
    
    conservation_data = load_conservation_shapefile(
        path=f'{root_dir}/src/global_avs/Conservation_Areas_-5503574965118299320'
    )
    onsud_data = load_onsud_data(onsud_path, PC_SHP_PATH)
    scaled_gas_elec_data = load_scaled_gas_elec()
    gas_deciles = pd.read_csv(f'{root_dir}/src/global_avs/neb_unfil_final_gas_deciles.csv')
    
    # Load all postcodes first
    all_postcodes = load_ids_from_file(batch_path)
    
    # Select postcodes based on mode
    if n_postcodes == -1:
        postcodes = all_postcodes
    elif random_sample and n_postcodes < len(all_postcodes):
        # Random sample
        np.random.seed(seed)
        postcodes = list(np.random.choice(all_postcodes, size=min(n_postcodes, len(all_postcodes)), replace=False))
        logger.info(f"Randomly sampled {len(postcodes)} postcodes from {len(all_postcodes)}")
    else:
        # Take first N
        postcodes = all_postcodes[:n_postcodes]
    
    logger.info(f"Using {len(postcodes)} postcodes")
    
    return {
        'conservation_data': conservation_data,
        'onsud_data': onsud_data,
        'scaled_gas_elec_data': scaled_gas_elec_data,
        'gas_deciles': gas_deciles,
        'postcodes': postcodes,
        'region': region,
    }


def prepare_building_data(pc: str, data: dict, logger: logging.Logger) -> Optional[pd.DataFrame]:
    energy_columns = [
        'gas_scaled_scaled_area_max', 'elec_scaled_scaled_area_max',
        'gas_scaled_scaled_area_min', 'elec_scaled_scaled_area_min',
        'gas_scaled_scaled_area_mode', 'elec_scaled_scaled_area_mode',
    ]
    
    pc = pc.strip()
    uprn_match = find_data_pc_joint(pc, data['onsud_data'], input_gpk=BUILDING_PATH)
    if uprn_match is None or uprn_match.empty:
        return None
    
    uprn_match = get_conservation_area(uprn_match, data['conservation_data'])
    building_data = pre_process_building_data(uprn_match)
    
    pc_decile = data['gas_deciles'][data['gas_deciles']['postcode'] == pc]
    if pc_decile.empty:
        return None
    building_data['avg_gas_percentile'] = pc_decile['avg_gas_decile'].values[0]
    
    energy = data['scaled_gas_elec_data'][data['scaled_gas_elec_data']['postcode'] == pc]
    building_data = building_data.merge(energy, on='upn')
    
    for col in energy_columns:
        if col not in building_data.columns:
            return None
    
    building_data['postcode'] = pc
    building_data['region'] = data.get('region', 'NE')  # Add region
    return building_data


# ========================================
# MODEL RUNNER
# ========================================

def create_sampler_from_df(epistemic_df: pd.DataFrame) -> Callable:
    def fixed_sampler(n_runs: int) -> pd.DataFrame:
        return epistemic_df.copy()
    return fixed_sampler


def calculate_actual_savings(improvement_factor: float, cavity_wall_means: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate actual solid wall savings from improvement factor.
    
    Formula: solid_mean = cavity_mean * (1 + improvement_factor)
    
    For negative means (savings), higher factor = more negative = more savings
    """
    solid_means = {}
    for percentile, cavity_mean in cavity_wall_means.items():
        if cavity_mean >= 0:
            solid_means[percentile] = cavity_mean * (1 - improvement_factor)
        else:
            solid_means[percentile] = cavity_mean * (1 + improvement_factor)
    return solid_means


def log_improvement_factor_table(logger: logging.Logger):
    """Log a table showing actual savings for each improvement factor."""
    
    # Cavity wall baseline values (from RetrofitEnergy)
    cavity_wall_means = {
        0: 0.1059182,
        1: 0.02026381,
        2: -0.023164,
        3: -0.0518756,
        4: -0.0752905,
        5: -0.0975109,
        6: -0.1179157,
        7: -0.1360034,
        8: -0.1537655,
        9: -0.1738108,
    }
    
    logger.info("\n" + "=" * 90)
    logger.info("IMPROVEMENT FACTOR → ACTUAL GAS SAVINGS CONVERSION TABLE")
    logger.info("=" * 90)
    logger.info("Cavity wall baseline (from literature):")
    logger.info(f"  Percentile 5 mean: {cavity_wall_means[5]:.1%} gas savings")
    logger.info(f"  Percentile 9 mean: {cavity_wall_means[9]:.1%} gas savings")
    logger.info("")
    logger.info("Formula: solid_wall_savings = cavity_savings × (1 + improvement_factor)")
    logger.info("")
    
    # Header
    header = f"{'Factor':<10}"
    for p in [3, 5, 7, 9]:
        header += f"{'P' + str(p) + ' Savings':<15}"
    logger.info(header)
    logger.info("-" * 70)
    
    # All factors
    all_factors = sorted(set(INTERNAL_WALL_FACTORS) | set(EXTERNAL_WALL_FACTORS))
    
    for factor in all_factors:
        solid_means = calculate_actual_savings(factor, cavity_wall_means)
        row = f"{factor:<10.2f}"
        for p in [3, 5, 7, 9]:
            savings_pct = abs(solid_means[p]) * 100
            row += f"{savings_pct:<15.1f}%"
        logger.info(row)
    
    logger.info("")
    logger.info("Note: Values show % gas reduction for buildings at each percentile")
    logger.info("=" * 90)


def run_model_with_wall_factors(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    internal_factor: float,
    external_factor: float,
    n_epistemic_runs: int,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Run model with specific wall improvement factors."""
    
    # Determine region - use most common if multiple
    if 'region' in building_data.columns:
        region = building_data['region'].mode().iloc[0]
    else:
        region = 'NE'
    
    # Generate epistemic scenarios with fixed wall factors
    np.random.seed(RANDOM_SEED_OUTER)
    epistemic_df = generate_epistemic_scenarios_lhs(n_epistemic_runs)
    
    # Override wall factors
    epistemic_df['solid_wall_internal_improvement_factor'] = internal_factor
    epistemic_df['solid_wall_external_improvement_factor'] = external_factor
    
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
# MAIN SWEEP
# ========================================

def run_parameter_sweep(
    batch_paths: List[str],
    output_base_dir: str,
    n_postcodes: int,
    n_epistemic_runs: int = 5,
    max_buildings: int = None,
    random_sample: bool = False,
    seed: int = 42,
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base_dir, f'sweep_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir, timestamp)
    
    logger.info("=" * 70)
    logger.info("WALL IMPROVEMENT FACTOR PARAMETER SWEEP")
    logger.info("=" * 70)
    logger.info(f"Number of batches: {len(batch_paths)}")
    logger.info(f"Postcodes per batch: {n_postcodes if n_postcodes != -1 else 'all'}")
    logger.info(f"Random sampling: {random_sample}")
    logger.info(f"Max buildings: {max_buildings if max_buildings else 'no limit'}")
    logger.info(f"Internal factors: {INTERNAL_WALL_FACTORS}")
    logger.info(f"External factors: {EXTERNAL_WALL_FACTORS}")
    logger.info(f"Metric: £/tCO2")
    
    # Log the conversion table showing actual savings
    log_improvement_factor_table(logger)
    
    # =========================================================
    # LOAD AND COMBINE DATA FROM ALL BATCHES
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA FROM ALL BATCHES")
    logger.info("=" * 70)
    
    all_building_data = []
    batches_processed = 0
    batches_failed = 0
    total_buildings_loaded = 0
    max_reached = False
    
    # Shuffle batches if random sampling to get better coverage
    if random_sample:
        np.random.seed(seed)
        batch_paths = list(np.random.permutation(batch_paths))
    
    for batch_path in batch_paths:
        if max_buildings and total_buildings_loaded >= max_buildings:
            logger.info(f"Reached max buildings limit ({max_buildings}), stopping batch loading")
            max_reached = True
            break
            
        logger.info(f"Loading batch: {batch_path}")
        
        try:
            data = load_test_data(batch_path, n_postcodes, logger, 
                                  random_sample=random_sample, seed=seed + batches_processed)
            
            for pc in data['postcodes']:
                if max_buildings and total_buildings_loaded >= max_buildings:
                    break
                    
                bd = prepare_building_data(pc, data, logger)
                if bd is not None:
                    # Check if adding this would exceed limit
                    if max_buildings and (total_buildings_loaded + len(bd)) > max_buildings:
                        # Take only what we need
                        remaining = max_buildings - total_buildings_loaded
                        bd = bd.head(remaining)
                        all_building_data.append(bd)
                        total_buildings_loaded += len(bd)
                        logger.info(f"  Added {len(bd)} buildings (truncated to reach max)")
                        break
                    else:
                        all_building_data.append(bd)
                        total_buildings_loaded += len(bd)
            
            batches_processed += 1
            logger.info(f"  Total buildings so far: {total_buildings_loaded}")
            
        except Exception as e:
            logger.warning(f"Failed to load batch {batch_path}: {e}")
            batches_failed += 1
            continue
    
    logger.info(f"Batches processed: {batches_processed}, failed: {batches_failed}")
    if max_reached:
        logger.info(f"Stopped early due to max_buildings limit")
    
    if not all_building_data:
        logger.error("No valid building data from any batch")
        return None, None
    
    combined_building_data = pd.concat(all_building_data, ignore_index=True)
    logger.info(f"Total buildings across all batches: {len(combined_building_data)}")
    
    # Count wall types
    if 'inferred_wall_type' in combined_building_data.columns:
        wall_counts = combined_building_data['inferred_wall_type'].value_counts()
        logger.info(f"Wall types in data:\n{wall_counts}")
    
    # Count regions
    if 'region' in combined_building_data.columns:
        region_counts = combined_building_data['region'].value_counts()
        logger.info(f"Regions in data:\n{region_counts}")
    
    retrofit_config = RetrofitConfig(
        existing_intervention_probs={
            'loft_insulation': 0,
            'floor_insulation': 0,
            'window_upgrades': 0,
            'roof_scaling_factor': 0.8,
        }
    )
    
    # =========================================================
    # SWEEP 1: Internal Wall Improvement (External fixed at 0.2)
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP 1: INTERNAL WALL IMPROVEMENT FACTOR")
    logger.info("(External fixed at 0.20)")
    logger.info("=" * 70)
    
    internal_results = []
    
    for internal_factor in INTERNAL_WALL_FACTORS:
        # Calculate actual savings at percentile 5 for reference
        actual_savings_p5 = abs(-0.0975109 * (1 + internal_factor)) * 100
        logger.info(f"Testing internal_factor = {internal_factor} (≈{actual_savings_p5:.0f}% gas savings at P5)")
        
        results_df = run_model_with_wall_factors(
            building_data=combined_building_data,
            retrofit_config=retrofit_config,
            internal_factor=internal_factor,
            external_factor=0.20,  # Fixed
            n_epistemic_runs=n_epistemic_runs,
            logger=logger,
        )
        
        if results_df is not None and COST_PER_TCO2_METRIC in results_df.columns:
            cost_per_tco2 = results_df[COST_PER_TCO2_METRIC]
            
            # Filter to solid wall internal buildings only if possible
            # For now, use all results
            
            internal_results.append({
                'internal_factor': internal_factor,
                'external_factor': 0.20,
                'approx_gas_savings_pct_p5': actual_savings_p5,
                'cost_per_tco2_mean': cost_per_tco2.mean(),
                'cost_per_tco2_median': cost_per_tco2.median(),
                'cost_per_tco2_std': cost_per_tco2.std(),
                'cost_per_tco2_p10': cost_per_tco2.quantile(0.10),
                'cost_per_tco2_p25': cost_per_tco2.quantile(0.25),
                'cost_per_tco2_p75': cost_per_tco2.quantile(0.75),
                'cost_per_tco2_p90': cost_per_tco2.quantile(0.90),
                'n_buildings': len(results_df),
                'pct_below_100': (cost_per_tco2 < 100).mean() * 100,
                'pct_below_200': (cost_per_tco2 < 200).mean() * 100,
                'pct_below_300': (cost_per_tco2 < 300).mean() * 100,
            })
            
            logger.info(f"  Mean £/tCO2: {cost_per_tco2.mean():.1f}")
            logger.info(f"  Median £/tCO2: {cost_per_tco2.median():.1f}")
            logger.info(f"  % below £200/tCO2: {(cost_per_tco2 < 200).mean() * 100:.1f}%")
    
    internal_df = pd.DataFrame(internal_results)
    internal_df.to_csv(f'{output_dir}/internal_wall_sweep.csv', index=False)
    
    # =========================================================
    # SWEEP 2: External Wall Improvement (Internal fixed at 0.1)
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP 2: EXTERNAL WALL IMPROVEMENT FACTOR")
    logger.info("(Internal fixed at 0.10)")
    logger.info("=" * 70)
    
    external_results = []
    
    for external_factor in EXTERNAL_WALL_FACTORS:
        # Calculate actual savings at percentile 5 for reference
        actual_savings_p5 = abs(-0.0975109 * (1 + external_factor)) * 100
        logger.info(f"Testing external_factor = {external_factor} (≈{actual_savings_p5:.0f}% gas savings at P5)")
        
        results_df = run_model_with_wall_factors(
            building_data=combined_building_data,
            retrofit_config=retrofit_config,
            internal_factor=0.10,  # Fixed
            external_factor=external_factor,
            n_epistemic_runs=n_epistemic_runs,
            logger=logger,
        )
        
        if results_df is not None and COST_PER_TCO2_METRIC in results_df.columns:
            cost_per_tco2 = results_df[COST_PER_TCO2_METRIC]
            
            external_results.append({
                'internal_factor': 0.10,
                'external_factor': external_factor,
                'approx_gas_savings_pct_p5': actual_savings_p5,
                'cost_per_tco2_mean': cost_per_tco2.mean(),
                'cost_per_tco2_median': cost_per_tco2.median(),
                'cost_per_tco2_std': cost_per_tco2.std(),
                'cost_per_tco2_p10': cost_per_tco2.quantile(0.10),
                'cost_per_tco2_p25': cost_per_tco2.quantile(0.25),
                'cost_per_tco2_p75': cost_per_tco2.quantile(0.75),
                'cost_per_tco2_p90': cost_per_tco2.quantile(0.90),
                'n_buildings': len(results_df),
                'pct_below_100': (cost_per_tco2 < 100).mean() * 100,
                'pct_below_200': (cost_per_tco2 < 200).mean() * 100,
                'pct_below_300': (cost_per_tco2 < 300).mean() * 100,
            })
            
            logger.info(f"  Mean £/tCO2: {cost_per_tco2.mean():.1f}")
            logger.info(f"  Median £/tCO2: {cost_per_tco2.median():.1f}")
            logger.info(f"  % below £200/tCO2: {(cost_per_tco2 < 200).mean() * 100:.1f}%")
    
    external_df = pd.DataFrame(external_results)
    external_df.to_csv(f'{output_dir}/external_wall_sweep.csv', index=False)
    
    # =========================================================
    # SUMMARY
    # =========================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: BREAK-EVEN ANALYSIS")
    logger.info("=" * 70)
    
    print("\n" + "=" * 80)
    print("INTERNAL WALL IMPROVEMENT SWEEP")
    print("=" * 80)
    print(f"{'Factor':<10} {'~Gas Savings':<15} {'Mean £/tCO2':<15} {'Median £/tCO2':<15} {'% < £200':<10}")
    print("-" * 65)
    for _, row in internal_df.iterrows():
        gas_sav = f"{row['approx_gas_savings_pct_p5']:.0f}%"
        print(f"{row['internal_factor']:<10.2f} {gas_sav:<15} {row['cost_per_tco2_mean']:<15.1f} {row['cost_per_tco2_median']:<15.1f} {row['pct_below_200']:<10.1f}%")
    
    print("\n" + "=" * 80)
    print("EXTERNAL WALL IMPROVEMENT SWEEP")
    print("=" * 80)
    print(f"{'Factor':<10} {'~Gas Savings':<15} {'Mean £/tCO2':<15} {'Median £/tCO2':<15} {'% < £200':<10}")
    print("-" * 65)
    for _, row in external_df.iterrows():
        gas_sav = f"{row['approx_gas_savings_pct_p5']:.0f}%"
        print(f"{row['external_factor']:<10.2f} {gas_sav:<15} {row['cost_per_tco2_mean']:<15.1f} {row['cost_per_tco2_median']:<15.1f} {row['pct_below_200']:<10.1f}%")
    
    # Find break-even points
    print("\n" + "=" * 80)
    print("BREAK-EVEN POINTS (where >50% buildings below threshold)")
    print("=" * 80)
    print("Shows the minimum improvement factor (and approx. gas savings) needed")
    print("")
    
    for threshold in ATTRACTIVE_THRESHOLDS:
        # Internal
        internal_df[f'pct_below_{threshold}'] = internal_df.apply(
            lambda x: x.get(f'pct_below_{threshold}', 0), axis=1
        )
        breakeven_internal = internal_df[internal_df[f'pct_below_{threshold}'] >= 50]
        if not breakeven_internal.empty:
            min_factor = breakeven_internal['internal_factor'].min()
            gas_savings = breakeven_internal[breakeven_internal['internal_factor'] == min_factor]['approx_gas_savings_pct_p5'].values[0]
            print(f"Internal wall @ £{threshold}/tCO2: factor >= {min_factor:.2f} (≈{gas_savings:.0f}% gas savings)")
        else:
            print(f"Internal wall @ £{threshold}/tCO2: not achieved in tested range")
        
        # External
        external_df[f'pct_below_{threshold}'] = external_df.apply(
            lambda x: x.get(f'pct_below_{threshold}', 0), axis=1
        )
        breakeven_external = external_df[external_df[f'pct_below_{threshold}'] >= 50]
        if not breakeven_external.empty:
            min_factor = breakeven_external['external_factor'].min()
            gas_savings = breakeven_external[breakeven_external['external_factor'] == min_factor]['approx_gas_savings_pct_p5'].values[0]
            print(f"External wall @ £{threshold}/tCO2: factor >= {min_factor:.2f} (≈{gas_savings:.0f}% gas savings)")
        else:
            print(f"External wall @ £{threshold}/tCO2: not achieved in tested range")
    
    logger.info(f"\nResults saved to: {output_dir}/")
    
    return internal_df, external_df


# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    args = parse_args()
    
    # Determine which batches to process
    if args.all_batches:
        print(f"Loading all batches from: {args.batch_file}")
        batch_paths = load_all_batch_paths(args.batch_file)
        print(f"Found {len(batch_paths)} batches")
    else:
        batch_paths = [args.batch]
        print(f"Running on single batch: {args.batch}")
    
    print(f"Output directory: {args.output}")
    print(f"N postcodes per batch: {args.n_postcodes if args.n_postcodes != -1 else 'all'}")
    print(f"Random sampling: {args.random_sample}")
    print(f"Max buildings: {args.max_buildings if args.max_buildings else 'no limit'}")
    print(f"N epistemic runs: {args.n_epistemic}")
    print(f"Random seed: {args.seed}")
    
    run_parameter_sweep(
        batch_paths=batch_paths,
        output_base_dir=args.output,
        n_postcodes=args.n_postcodes,
        n_epistemic_runs=args.n_epistemic,
        max_buildings=args.max_buildings,
        random_sample=args.random_sample,
        seed=args.seed,
    )