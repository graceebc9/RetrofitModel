"""
Module: wall_improvement_sweep_v4.py

Parameter sweep with CORRECT stratification and CORRECT epistemic aggregation:
- Wall type: solid_wall vs cavity_wall
- Insulation type (for solid walls): internal vs external from 'inferred_insulation_type'
- Gas consumption decile
- Premise type

Key model understanding:
- Solid wall savings = cavity_wall_savings × (1 + improvement_factor)
- inferred_wall_type: 'solid_wall' or 'cavity_wall'
- inferred_insulation_type: 'internal' or 'external' (determines which factor applies)
- £/tCO2 is over 5 years (not lifetime)

Two separate concepts:
1. prob_external (default 0.5): Probability that a solid wall building gets external
   (vs internal) insulation. This is just for categorizing buildings - 50/50 split.
   
2. solid_wall_internal/external_improvement_factor: The ENERGY SAVINGS multiplier
   applied to cavity wall baseline savings. This is what we're sweeping to find
   break-even points.

Epistemic aggregation method:
- Inner (aleatoric): 5,000 samples per building per epistemic run
- Outer (epistemic): N runs with different parameter draws
- Per building: use p50 (median) of aleatoric distribution
- Aggregate across epistemic runs: mean(p50) + 1*std(p50) = conservative estimate
- Then compute summary statistics across buildings
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

# ========================================
# CONFIGURATION
# ========================================

N_EPISTEMIC_RUNS = 50
RANDOM_SEED_OUTER = 42

SCENARIOS = ['wall_installation']

# Paths 
PC_SHP_PATH = '/rds/user/gb669/hpc-work/energy_map/data/postcode_polygons/codepoint-poly_5267291'
BUILDING_PATH = '/rds/user/gb669/hpc-work/energy_map/data/building_files/UKBuildings_Edition_15_new_format_upn.gpkg'
location_input_data_folder = '/home/gb669/rds/hpc-work/energy_map/data/input_data'
onsud_path_base = '/home/gb669/rds/hpc-work/energy_map/data/onsud_files/Data'

GAS_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_gas_2022.csv'
ELEC_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_all_meters_electricity_2022.csv'

# Parameter sweep ranges
INTERNAL_WALL_FACTORS = [ 1.00, 2.00,  3.00, 4,  5,  6,   7, 8 ,  9 ,  10 ]
EXTERNAL_WALL_FACTORS = [ 1.00,   2.00,  3.00, 4, 5,  6,  7,   8 ,  9 , 10 ]

# Key metric - using p50 (median of aleatoric distribution)
COST_PER_TCO2_METRIC = 'wall_installation_capex_per_net_ton_co2_wall_installation_p50'

# Columns to preserve from input data
PRESERVE_COLS = [
    'upn', 'postcode', 'premise_type_filled', 'avg_gas_percentile',
    'inferred_wall_type', 'inferred_insulation_type', 'region'
]

# Thresholds for break-even analysis (5-year £/tCO2)
# Note: For 30-year equivalent, divide by 6
ATTRACTIVE_THRESHOLDS_5YR = [500, 800, 1000, 1500, 2000]

# Epistemic aggregation parameter
N_STD_CONSERVATIVE = 1.0  # mean + N_STD * std for conservative estimate

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
from src.PreProcessRetrofit import vectorized_process_buildings 
from src.RetrofitConfig import RetrofitConfig

# ========================================
# ARGUMENT PARSING
# ========================================

def parse_args():
    parser = argparse.ArgumentParser(description='Wall improvement factor parameter sweep v4')
    parser.add_argument('--batch', type=str, default='batches/NE/batch_10.txt', help='Path to batch file')
    parser.add_argument('--output', type=str, default='wall_sweep_results_v5', help='Base output directory')
    parser.add_argument('--n-postcodes', type=int, default=100, help='Number of postcodes per batch (-1 for all)')
    parser.add_argument('--all', action='store_true', help='Process all postcodes in batch')
    parser.add_argument('--all-batches', action='store_true', help='Run across ALL batches')
    parser.add_argument('--batch-file', type=str, default='batch_paths.txt', help='File containing batch paths')
    parser.add_argument('--max-buildings', type=int, default=1000, help='Maximum total buildings')
    parser.add_argument('--sample-per-batch', type=int, default=None, help='Randomly sample N postcodes per batch')
    parser.add_argument('--n-epistemic', type=int, default=5, help='Number of epistemic runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--prob-external', type=float, default=0.5, 
                        help='Probability of external (vs internal) insulation for solid walls (default: 0.5)')
    parser.add_argument('--n-std', type=float, default=1.0,
                        help='Number of std devs to add for conservative estimate (default: 1.0)')
    args = parser.parse_args()
    
    if args.all:
        args.n_postcodes = -1
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
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    file_handler = logging.FileHandler(f'{output_dir}/wall_sweep_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


def load_all_batch_paths(batch_file: str) -> List[str]:
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"Batch file not found: {batch_file}")
    with open(batch_file, 'r') as f:
        batch_paths = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(batch_paths))


# ========================================
# DATA LOADING
# ========================================

def load_test_data(batch_path: str, n_postcodes: int, logger: logging.Logger, 
                   random_sample: bool = False, seed: int = 42):
    logger.info(f"Loading data for batch: {batch_path}")
    
    batch_dir = os.path.dirname(batch_path)
    batch_num = os.path.basename(batch_path).replace('batch_', '').replace('.txt', '')
    onsud_path = os.path.join(batch_dir, f'onsud_{batch_num}.csv')
    region = os.path.basename(batch_dir)
    logger.info(f"Detected region: {region}")
    
    conservation_data = load_conservation_shapefile(
        path=f'{root_dir}/src/global_avs/Conservation_Areas_-5503574965118299320'
    )
    onsud_data = load_onsud_data(onsud_path, PC_SHP_PATH)
    scaled_gas_elec_data = load_scaled_gas_elec()
    gas_deciles = pd.read_csv(f'{root_dir}/src/global_avs/neb_unfil_final_gas_deciles.csv')
    
    all_postcodes = load_ids_from_file(batch_path)
    
    if n_postcodes == -1:
        postcodes = all_postcodes
    elif random_sample and n_postcodes < len(all_postcodes):
        np.random.seed(seed)
        postcodes = list(np.random.choice(all_postcodes, size=min(n_postcodes, len(all_postcodes)), replace=False))
        logger.info(f"Randomly sampled {len(postcodes)} postcodes from {len(all_postcodes)}")
    else:
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
    building_data['region'] = data.get('region', 'NE')
    return building_data


# ========================================
# EPISTEMIC AGGREGATION
# ========================================

def compute_building_conservative_estimate(
    df: pd.DataFrame, 
    metric_col: str = COST_PER_TCO2_METRIC,
    building_id: str = 'upn',
    n_std: float = 1.0,
) -> pd.Series:
    """
    For each building, compute conservative estimate: mean + n_std*std across epistemic runs.
    
    Args:
        df: Results with multiple epistemic runs per building
        metric_col: The p50 column from aleatoric distribution
        building_id: Column identifying unique buildings
        n_std: Number of standard deviations to add (1.0 = conservative)
    
    Returns:
        Series indexed by building_id with conservative estimate
    """
    grouped = df.groupby(building_id)[metric_col]
    
    means = grouped.mean()
    stds = grouped.std().fillna(0)  # fillna for buildings with single run
    
    conservative = means + n_std * stds
    
    return conservative


# ========================================
# STRATIFIED AGGREGATION
# ========================================

def compute_stats(
    df: pd.DataFrame, 
    metric_col: str = COST_PER_TCO2_METRIC,
    building_id: str = 'upn',
    n_std: float = N_STD_CONSERVATIVE,
) -> dict:
    """
    Compute statistics across buildings using conservative epistemic estimate.
    
    Per building: mean(p50) + n_std * std(p50) across epistemic runs
    Then: distribution statistics across buildings
    """
    if df.empty:
        return {
            'n': 0, 'n_valid': 0,
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'p10': np.nan, 'p25': np.nan, 'p75': np.nan, 'p90': np.nan,
            'pct_below_1000': np.nan, 'pct_below_2000': np.nan,
            'pct_below_3000': np.nan, 'pct_below_5000': np.nan,
        }
    
    building_values = compute_building_conservative_estimate(
        df, metric_col, building_id, n_std=n_std
    )
    
    if len(building_values) == 0:
        return {
            'n': 0, 'n_valid': 0,
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'p10': np.nan, 'p25': np.nan, 'p75': np.nan, 'p90': np.nan,
            'pct_below_1000': np.nan, 'pct_below_2000': np.nan,
            'pct_below_3000': np.nan, 'pct_below_5000': np.nan,
        }
    
    valid = building_values[np.isfinite(building_values) & (building_values.abs() < 1e6)]
    
    if len(valid) == 0:
        return {
            'n': 0, 'n_valid': 0,
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'p10': np.nan, 'p25': np.nan, 'p75': np.nan, 'p90': np.nan,
            'pct_below_1000': np.nan, 'pct_below_2000': np.nan,
            'pct_below_3000': np.nan, 'pct_below_5000': np.nan,
        }
    
    return {
        'n': len(building_values),
        'n_valid': len(valid),
        'mean': valid.mean(),
        'median': valid.median(),
        'std': valid.std(),
        'p10': valid.quantile(0.10),
        'p25': valid.quantile(0.25),
        'p75': valid.quantile(0.75),
        'p90': valid.quantile(0.90),
        'pct_below_1000': (valid < 1000).mean() * 100,
        'pct_below_2000': (valid < 2000).mean() * 100,
        'pct_below_3000': (valid < 3000).mean() * 100,
        'pct_below_5000': (valid < 5000).mean() * 100,
    }


def compute_epistemic_sensitivity(
    df: pd.DataFrame, 
    metric_col: str = COST_PER_TCO2_METRIC,
    building_id: str = 'upn',
    epistemic_col: str = 'epistemic_run_id',
) -> dict:
    """
    Compute how summary statistics vary across epistemic runs.
    
    Returns the mean and range of key statistics across epistemic scenarios.
    """
    if epistemic_col not in df.columns:
        return {}
    
    run_stats = []
    for run_id, run_df in df.groupby(epistemic_col):
        # For each run, just use the p50 values directly (no cross-run aggregation)
        values = run_df.groupby(building_id)[metric_col].first()  # Should be one per building per run
        valid = values[np.isfinite(values) & (values.abs() < 1e6)]
        
        if len(valid) > 0:
            run_stats.append({
                'run': run_id,
                'median': valid.median(),
                'mean': valid.mean(),
                'p25': valid.quantile(0.25),
                'p75': valid.quantile(0.75),
                'pct_below_2000': (valid < 2000).mean() * 100,
                'pct_below_3000': (valid < 3000).mean() * 100,
            })
    
    if not run_stats:
        return {}
    
    run_df = pd.DataFrame(run_stats)
    
    return {
        'epistemic_median_mean': run_df['median'].mean(),
        'epistemic_median_std': run_df['median'].std(),
        'epistemic_median_min': run_df['median'].min(),
        'epistemic_median_max': run_df['median'].max(),
        'epistemic_pct_below_2000_mean': run_df['pct_below_2000'].mean(),
        'epistemic_pct_below_2000_std': run_df['pct_below_2000'].std(),
        'epistemic_pct_below_3000_mean': run_df['pct_below_3000'].mean(),
        'epistemic_pct_below_3000_std': run_df['pct_below_3000'].std(),
        'n_epistemic_runs': len(run_stats),
    }


def create_building_category(row) -> str:
    """Create a combined category for stratification."""
    wall_type = row.get('inferred_wall_type', 'unknown')
    ins_type = row.get('inferred_insulation_type', 'unknown')
    
    if wall_type == 'cavity_wall':
        return 'cavity_wall'
    elif wall_type == 'solid_wall':
        if ins_type == 'internal':
            return 'solid_wall_internal'
        elif ins_type == 'external':
            return 'solid_wall_external'
        else:
            return f'solid_wall_{ins_type}'
    else:
        return f'{wall_type}_{ins_type}'


def aggregate_by_building_category(
    results_df: pd.DataFrame, 
    metric_col: str,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Aggregate results by combined wall type + insulation type category."""
    
    results_df = results_df.copy()
    if 'building_category' not in results_df.columns:
        results_df['building_category'] = results_df.apply(create_building_category, axis=1)
    
    rows = []
    for category in results_df['building_category'].unique():
        mask = results_df['building_category'] == category
        subset = results_df.loc[mask]
        
        # Main stats (epistemic-aggregated with conservative estimate)
        stats = compute_stats(subset, metric_col, n_std=n_std)
        stats['building_category'] = category
        
        # Epistemic sensitivity
        sensitivity = compute_epistemic_sensitivity(subset, metric_col)
        stats.update(sensitivity)
        
        rows.append(stats)
    
    return pd.DataFrame(rows)


def aggregate_by_gas_decile(
    results_df: pd.DataFrame, 
    metric_col: str,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Aggregate results by gas consumption decile."""
    if 'avg_gas_percentile' not in results_df.columns:
        return pd.DataFrame()
    
    df = results_df.copy()
    df['gas_decile_bin'] = pd.cut(
        df['avg_gas_percentile'],
        bins=[-0.1, 2, 4, 6, 8, 10.1],
        labels=['0-2 (low)', '2-4', '4-6', '6-8', '8-10 (high)']
    )
    
    rows = []
    for decile in df['gas_decile_bin'].dropna().unique():
        mask = df['gas_decile_bin'] == decile
        subset = df.loc[mask]
        
        stats = compute_stats(subset, metric_col, n_std=n_std)
        stats['gas_decile'] = decile
        
        sensitivity = compute_epistemic_sensitivity(subset, metric_col)
        stats.update(sensitivity)
        
        rows.append(stats)
    
    return pd.DataFrame(rows)


def aggregate_by_premise_type(
    results_df: pd.DataFrame, 
    metric_col: str,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Aggregate results by premise/building type."""
    if 'premise_type_filled' not in results_df.columns:
        return pd.DataFrame()
    
    rows = []
    for ptype in results_df['premise_type_filled'].unique():
        mask = results_df['premise_type_filled'] == ptype
        subset = results_df.loc[mask]
        
        stats = compute_stats(subset, metric_col, n_std=n_std)
        stats['premise_type_filled'] = ptype
        
        sensitivity = compute_epistemic_sensitivity(subset, metric_col)
        stats.update(sensitivity)
        
        rows.append(stats)
    
    return pd.DataFrame(rows)


def aggregate_category_by_gas_decile(
    results_df: pd.DataFrame, 
    metric_col: str,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Cross-tabulation: building category x gas decile."""
    if 'avg_gas_percentile' not in results_df.columns:
        return pd.DataFrame()
    
    df = results_df.copy()
    if 'building_category' not in df.columns:
        df['building_category'] = df.apply(create_building_category, axis=1)
    df['gas_decile_bin'] = pd.cut(
        df['avg_gas_percentile'],
        bins=[-0.1, 2, 4, 6, 8, 10.1],
        labels=['0-2', '2-4', '4-6', '6-8', '8-10']
    )
    
    rows = []
    for category in df['building_category'].unique():
        for decile in df['gas_decile_bin'].dropna().unique():
            mask = (df['building_category'] == category) & (df['gas_decile_bin'] == decile)
            if mask.sum() > 0:
                subset = df.loc[mask]
                stats = compute_stats(subset, metric_col, n_std=n_std)
                stats['building_category'] = category
                stats['gas_decile'] = decile
                rows.append(stats)
    
    return pd.DataFrame(rows)


def aggregate_category_by_premise(
    results_df: pd.DataFrame, 
    metric_col: str,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Cross-tabulation: building category x premise type."""
    if 'premise_type_filled' not in results_df.columns:
        return pd.DataFrame()
    
    df = results_df.copy()
    if 'building_category' not in df.columns:
        df['building_category'] = df.apply(create_building_category, axis=1)
    
    rows = []
    for category in df['building_category'].unique():
        for ptype in df['premise_type_filled'].unique():
            mask = (df['building_category'] == category) & (df['premise_type_filled'] == ptype)
            if mask.sum() > 0:
                subset = df.loc[mask]
                stats = compute_stats(subset, metric_col, n_std=n_std)
                stats['building_category'] = category
                stats['premise_type_filled'] = ptype
                rows.append(stats)
    
    return pd.DataFrame(rows)


# ========================================
# MODEL RUNNER
# ========================================

def create_sampler_from_df(epistemic_df: pd.DataFrame) -> Callable:
    def fixed_sampler(n_runs: int) -> pd.DataFrame:
        return epistemic_df.copy()
    return fixed_sampler


def run_model_with_wall_factors(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    internal_factor: float,
    external_factor: float,
    n_epistemic_runs: int,
    logger: logging.Logger,
    epistemic_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Run model with specific wall improvement factors, preserving building characteristics."""
    
    if 'region' in building_data.columns:
        region = building_data['region'].mode().iloc[0]
    else:
        region = 'NE'
    
    np.random.seed(RANDOM_SEED_OUTER)
    
    epistemic_df = epistemic_df.copy()
    epistemic_df['solid_wall_internal_improvement_factor'] = internal_factor
    epistemic_df['solid_wall_external_improvement_factor'] = external_factor
    
    fixed_sampler = create_sampler_from_df(epistemic_df)
    
    scenario_generator = RetrofitScenarioGenerator2DMC(
        n_epistemic_runs=n_epistemic_runs,
        epistemic_sampler=fixed_sampler
    )
    
    RetrofitModel2D.retrofit_config = retrofit_config
    
    # Store original columns for merging back
    original_data = building_data[PRESERVE_COLS].copy() if all(c in building_data.columns for c in PRESERVE_COLS[:2]) else None
    
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
    
    # Merge back preserved columns if they're missing
    if results is not None and original_data is not None:
        for col in PRESERVE_COLS:
            if col in original_data.columns and col not in results.columns:
                if len(original_data) == len(results):
                    results[col] = original_data[col].values
    
    return results


# ========================================
# SWEEP CONFIGURATION
# ========================================

@dataclass
class SweepConfig:
    """Configuration for a parameter sweep run."""
    n_postcodes: int
    n_epistemic_runs: int = 5
    max_buildings: Optional[int] = None
    random_sample: bool = False
    seed: int = 42
    prob_external: float = 0.5
    n_std: float = 1.0  # For conservative epistemic aggregation


@dataclass
class SweepParameters:
    """Parameters for a single sweep iteration."""
    internal_factor: float
    external_factor: float
    sweep_type: str  # 'internal' or 'external'


# ========================================
# DATA LOADING (BATCHED)
# ========================================

def load_all_batches(
    batch_paths: List[str],
    config: SweepConfig,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Load and combine building data from all batches."""
    
    if config.random_sample:
        np.random.seed(config.seed)
        batch_paths = list(np.random.permutation(batch_paths))

    all_building_data = []
    total_loaded = 0

    for batch_idx, batch_path in enumerate(batch_paths):
        if config.max_buildings and total_loaded >= config.max_buildings:
            logger.info(f"Reached max buildings limit ({config.max_buildings})")
            break

        batch_data = _load_single_batch(
            batch_path, batch_idx, config, total_loaded, logger
        )
        
        if batch_data is not None:
            all_building_data.extend(batch_data)
            total_loaded = sum(len(bd) for bd in all_building_data)
            logger.info(f"Batch {batch_idx+1}/{len(batch_paths)}: total buildings = {total_loaded}")

    if not all_building_data:
        logger.error("No valid building data")
        return None

    combined = pd.concat(all_building_data, ignore_index=True)
    logger.info(f"\nTotal buildings: {len(combined)}")
    return combined


def _load_single_batch(
    batch_path: str,
    batch_idx: int,
    config: SweepConfig,
    total_loaded: int,
    logger: logging.Logger,
) -> Optional[List[pd.DataFrame]]:
    """Load building data from a single batch."""
    try:
        data = load_test_data(
            batch_path, 
            config.n_postcodes, 
            logger,
            random_sample=config.random_sample, 
            seed=config.seed + batch_idx
        )
        
        batch_buildings = []
        for pc in data['postcodes']:
            if config.max_buildings and total_loaded >= config.max_buildings:
                break
                
            bd = prepare_building_data(pc, data, logger)
            if bd is None:
                continue
                
            bd = _trim_to_limit(bd, total_loaded, config.max_buildings)
            batch_buildings.append(bd)
            total_loaded += len(bd)
            
        return batch_buildings
        
    except Exception as e:
        logger.warning(f"Failed to load batch {batch_path}: {e}")
        return None


def _trim_to_limit(
    df: pd.DataFrame, 
    current_count: int, 
    max_count: Optional[int]
) -> pd.DataFrame:
    """Trim dataframe if adding it would exceed max_count."""
    if max_count and (current_count + len(df)) > max_count:
        remaining = max_count - current_count
        return df.head(remaining)
    return df

def check_batch_complete(output_base_dir: str, batch_path: str) -> bool:
    """Check if a batch already has completed results in any sweep directory."""
    # Extract batch number from path like 'batches/NE/batch_10.txt'
    batch_num = os.path.basename(batch_path).replace('batch_', '').replace('.txt', '')
    print(f'batch num {batch_num}') 
    batch_dir = output_base_dir
    print(batch_dir)
    if not os.path.exists(batch_dir):
        return False
    
    # Check for any sweep_* directory with required files
    required_files = ['sweep_by_building_category.csv', 'detailed_results.parquet']
    
    for item in os.listdir(batch_dir):
        if item.startswith('sweep_'):
            sweep_dir = os.path.join(batch_dir, item)
            if os.path.isdir(sweep_dir):
                if all(os.path.exists(os.path.join(sweep_dir, f)) for f in required_files):
                    return True
    
    return False

# ========================================
# PREPROCESSING
# ========================================

EXCLUDED_PREMISE_TYPES = [
    'Unknown', 
    'Domestic outbuilding', 
    'Medium height flats 5-6 storeys', 
    'Tall flats 6-15 storeys',
]

COLUMN_MAPPING = {
    'age_band': 'premise_age_bucketed',
    'floor_count': 'fc_filled',
    'gross_external_area': None,
    'gross_internal_area': None,
    'footprint_area': 'premise_area',
    'footprint_circumference': 'perimeter_length',
    'flat_count': 'est_num_flats',
    'building_type': 'premise_type_filled',
    'building_footprint_area': 'premise_area',
    'avg_gas_percentile': 'avg_gas_percentile',
    'cons_bool': 'conservation_area_bool',
    'inferred_insulation_type': 'inferred_insulation_type',
    'inferred_wall_type': 'inferred_wall_type',
}


def preprocess_buildings(
    df: pd.DataFrame,
    config: SweepConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Apply vectorized processing and filter buildings."""
    
    logger.info("\nPreprocessing buildings to get inferred_insulation_type...")
    logger.info(f"Using prob_external = {config.prob_external:.2f}")

    retrofit_config = RetrofitConfig(
        existing_intervention_probs={
            'loft_insulation': 0,
            'floor_insulation': 0,
            'window_upgrades': 0,
            'roof_scaling_factor': 0.8,
        }
    )

    df = vectorized_process_buildings(
        result_df=df,
        col_mapping=COLUMN_MAPPING,
        config=retrofit_config,
        random_seed=RANDOM_SEED_OUTER,
        prob_external=config.prob_external,
    )
    
    # Filter out excluded premise types
    df = df[~df['premise_type_filled'].isin(EXCLUDED_PREMISE_TYPES)]
    
    # Add building category
    df['building_category'] = df.apply(create_building_category, axis=1)
    
    logger.info("Successfully applied vectorized_process_buildings")
    _log_preprocessing_summary(df, logger)
    
    return df


def _log_preprocessing_summary(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Log summary statistics after preprocessing."""
    
    if 'inferred_insulation_type' in df.columns:
        logger.info(f"inferred_insulation_type values:\n{df['inferred_insulation_type'].value_counts()}")
    else:
        logger.warning("inferred_insulation_type column not found!")
        logger.info(f"Available columns: {list(df.columns)}")

    logger.info(f"\nBuilding category distribution:\n{df['building_category'].value_counts()}")

    if 'inferred_wall_type' in df.columns:
        logger.info(f"\nWall type distribution:\n{df['inferred_wall_type'].value_counts()}")

    if 'inferred_insulation_type' in df.columns and 'inferred_wall_type' in df.columns:
        solid_mask = df['inferred_wall_type'] == 'solid_wall'
        if solid_mask.any():
            logger.info(f"\nInsulation type distribution (solid walls):")
            logger.info(df.loc[solid_mask, 'inferred_insulation_type'].value_counts())

    if 'premise_type_filled' in df.columns:
        logger.info(f"\nPremise type distribution:\n{df['premise_type_filled'].value_counts()}")

    if 'avg_gas_percentile' in df.columns:
        logger.info(
            f"\nGas percentile: mean={df['avg_gas_percentile'].mean():.1f}, "
            f"median={df['avg_gas_percentile'].median():.1f}"
        )


# ========================================
# PARAMETER SWEEPS
# ========================================

def run_sweep(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    sweep_params: SweepParameters,
    n_epistemic_runs: int,
    epistemic_df: pd.DataFrame,
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> Tuple[Optional[pd.DataFrame], List[dict]]:
    """Run a single parameter sweep iteration."""
    
    logger.info(f"\n--- Testing {sweep_params.sweep_type}_factor = "
                f"{getattr(sweep_params, f'{sweep_params.sweep_type}_factor')} ---")

    results_df = run_model_with_wall_factors(
        building_data=building_data,
        retrofit_config=retrofit_config,
        internal_factor=sweep_params.internal_factor,
        external_factor=sweep_params.external_factor,
        n_epistemic_runs=n_epistemic_runs,
        logger=logger,
        epistemic_df=epistemic_df,
    )

    if results_df is None or COST_PER_TCO2_METRIC not in results_df.columns:
        logger.warning(f"No results for {sweep_params.sweep_type}_factor")
        return None, []

    # Tag results with sweep parameters
    results_df['internal_factor'] = sweep_params.internal_factor
    results_df['external_factor'] = sweep_params.external_factor
    results_df['sweep_type'] = sweep_params.sweep_type

    # Aggregate by building category
    aggregated = _aggregate_and_log_results(
        results_df, sweep_params, logger, n_std
    )

    return results_df, aggregated


def _aggregate_and_log_results(
    results_df: pd.DataFrame,
    sweep_params: SweepParameters,
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> List[dict]:
    """Aggregate results by building category and log summary."""
    
    cat_stats = aggregate_by_building_category(results_df, COST_PER_TCO2_METRIC, n_std=n_std)
    aggregated = []
    
    logger.info(f"  Results by building category (using mean + {n_std}*std conservative estimate):")
    for _, row in cat_stats.iterrows():
        row_dict = row.to_dict()
        row_dict['internal_factor'] = sweep_params.internal_factor
        row_dict['external_factor'] = sweep_params.external_factor
        row_dict['sweep_type'] = sweep_params.sweep_type
        row_dict['n_std'] = n_std
        aggregated.append(row_dict)
        
        logger.info(
            f"    {row['building_category']}: n={row['n']}, "
            f"median={row['median']:.0f}, pct<2000={row['pct_below_2000']:.1f}%"
        )
    
    return aggregated


def run_internal_sweep(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    n_epistemic_runs: int,
    epistemic_df: pd.DataFrame,
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> Tuple[List[pd.DataFrame], List[dict]]:
    """Run sweep over internal wall improvement factors."""
    
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP 1: INTERNAL WALL IMPROVEMENT FACTOR")
    logger.info("(External fixed at 0.20)")
    logger.info("Focus on: solid_wall_internal buildings")
    logger.info(f"Epistemic aggregation: mean(p50) + {n_std}*std(p50)")
    logger.info("=" * 70)

    all_detailed = []
    all_aggregated = []

    for internal_factor in INTERNAL_WALL_FACTORS:
        params = SweepParameters(
            internal_factor=internal_factor,
            external_factor=0.20,
            sweep_type='internal',
        )
        
        results_df, aggregated = run_sweep(
            building_data, retrofit_config, params,
            n_epistemic_runs, epistemic_df, logger, n_std
        )
        
        if results_df is not None:
            all_detailed.append(results_df)
            all_aggregated.extend(aggregated)

    return all_detailed, all_aggregated


def run_external_sweep(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    n_epistemic_runs: int,
    epistemic_df: pd.DataFrame,
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> Tuple[List[pd.DataFrame], List[dict]]:
    """Run sweep over external wall improvement factors."""
    
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP 2: EXTERNAL WALL IMPROVEMENT FACTOR")
    logger.info("(Internal fixed at 0.10)")
    logger.info("Focus on: solid_wall_external buildings")
    logger.info(f"Epistemic aggregation: mean(p50) + {n_std}*std(p50)")
    logger.info("=" * 70)

    all_detailed = []
    all_aggregated = []

    for external_factor in EXTERNAL_WALL_FACTORS:
        params = SweepParameters(
            internal_factor=0.10,
            external_factor=external_factor,
            sweep_type='external',
        )
        
        results_df, aggregated = run_sweep(
            building_data, retrofit_config, params,
            n_epistemic_runs, epistemic_df, logger, n_std
        )
        
        if results_df is not None:
            all_detailed.append(results_df)
            all_aggregated.extend(aggregated)

    return all_detailed, all_aggregated


# ========================================
# OUTPUT
# ========================================

def save_results(
    all_results: List[dict],
    all_detailed_results: List[pd.DataFrame],
    output_dir: str,
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Save sweep results to disk."""
    
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{output_dir}/sweep_by_building_category.csv', index=False)
    logger.info("Saved: sweep_by_building_category.csv")

    if all_detailed_results:
        full_results = pd.concat(all_detailed_results, ignore_index=True)
        full_results.to_parquet(f'{output_dir}/detailed_results.parquet')
        logger.info(f"Saved: detailed_results.parquet ({len(full_results)} rows)")
        generate_additional_summaries(full_results, output_dir, logger, n_std)

    print_summary_tables(results_df, logger, n_std)
    return results_df


def setup_output_directory(output_base_dir: str) -> Tuple[str, str]:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base_dir, f'sweep_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, timestamp


def log_run_header(config: SweepConfig, n_batches: int, logger: logging.Logger) -> None:
    """Log the run configuration header."""
    logger.info("=" * 70)
    logger.info("WALL IMPROVEMENT FACTOR PARAMETER SWEEP v4")
    logger.info("=" * 70)
    logger.info(f"Number of batches: {n_batches}")
    logger.info(f"Postcodes per batch: {config.n_postcodes if config.n_postcodes != -1 else 'all'}")
    logger.info(f"Max buildings: {config.max_buildings if config.max_buildings else 'no limit'}")
    logger.info(f"Internal factors: {INTERNAL_WALL_FACTORS}")
    logger.info(f"External factors: {EXTERNAL_WALL_FACTORS}")
    logger.info(f"Epistemic aggregation: mean(p50) + {config.n_std}*std(p50)")
    logger.info(f"Note: £/tCO2 is over 5-year horizon")


def generate_additional_summaries(
    full_results: pd.DataFrame, 
    output_dir: str, 
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> None:
    """Generate additional stratified summary CSVs."""
    
    metric = COST_PER_TCO2_METRIC
    
    # By building category x gas decile
    logger.info("Generating building category x gas decile summaries...")
    cat_gas_results = []
    for (int_f, ext_f, sweep), group in full_results.groupby(['internal_factor', 'external_factor', 'sweep_type']):
        cross_stats = aggregate_category_by_gas_decile(group, metric, n_std=n_std)
        if not cross_stats.empty:
            cross_stats['internal_factor'] = int_f
            cross_stats['external_factor'] = ext_f
            cross_stats['sweep_type'] = sweep
            cat_gas_results.append(cross_stats)
    
    if cat_gas_results:
        cat_gas_df = pd.concat(cat_gas_results, ignore_index=True)
        cat_gas_df.to_csv(f'{output_dir}/category_x_gas_decile.csv', index=False)
        logger.info(f"Saved: category_x_gas_decile.csv")
    
    # By building category x premise type
    logger.info("Generating building category x premise type summaries...")
    cat_premise_results = []
    for (int_f, ext_f, sweep), group in full_results.groupby(['internal_factor', 'external_factor', 'sweep_type']):
        cross_stats = aggregate_category_by_premise(group, metric, n_std=n_std)
        if not cross_stats.empty:
            cross_stats['internal_factor'] = int_f
            cross_stats['external_factor'] = ext_f
            cross_stats['sweep_type'] = sweep
            cat_premise_results.append(cross_stats)
    
    if cat_premise_results:
        cat_premise_df = pd.concat(cat_premise_results, ignore_index=True)
        cat_premise_df.to_csv(f'{output_dir}/category_x_premise_type.csv', index=False)
        logger.info(f"Saved: category_x_premise_type.csv")
    
    # By gas decile only
    gas_results = []
    for (int_f, ext_f, sweep), group in full_results.groupby(['internal_factor', 'external_factor', 'sweep_type']):
        gas_stats = aggregate_by_gas_decile(group, metric, n_std=n_std)
        if not gas_stats.empty:
            gas_stats['internal_factor'] = int_f
            gas_stats['external_factor'] = ext_f
            gas_stats['sweep_type'] = sweep
            gas_results.append(gas_stats)
    
    if gas_results:
        gas_df = pd.concat(gas_results, ignore_index=True)
        gas_df.to_csv(f'{output_dir}/by_gas_decile.csv', index=False)
        logger.info(f"Saved: by_gas_decile.csv")
    
    # By premise type only
    premise_results = []
    for (int_f, ext_f, sweep), group in full_results.groupby(['internal_factor', 'external_factor', 'sweep_type']):
        premise_stats = aggregate_by_premise_type(group, metric, n_std=n_std)
        if not premise_stats.empty:
            premise_stats['internal_factor'] = int_f
            premise_stats['external_factor'] = ext_f
            premise_stats['sweep_type'] = sweep
            premise_results.append(premise_stats)
    
    if premise_results:
        premise_df = pd.concat(premise_results, ignore_index=True)
        premise_df.to_csv(f'{output_dir}/by_premise_type.csv', index=False)
        logger.info(f"Saved: by_premise_type.csv")
    
    # Epistemic sensitivity summary
    logger.info("Generating epistemic sensitivity summary...")
    epistemic_summary = []
    for (int_f, ext_f, sweep), group in full_results.groupby(['internal_factor', 'external_factor', 'sweep_type']):
        if 'building_category' not in group.columns:
            group = group.copy()
            group['building_category'] = group.apply(create_building_category, axis=1)
        
        for category in group['building_category'].unique():
            subset = group[group['building_category'] == category]
            sensitivity = compute_epistemic_sensitivity(subset, metric)
            if sensitivity:
                sensitivity['internal_factor'] = int_f
                sensitivity['external_factor'] = ext_f
                sensitivity['sweep_type'] = sweep
                sensitivity['building_category'] = category
                epistemic_summary.append(sensitivity)
    
    if epistemic_summary:
        epistemic_df = pd.DataFrame(epistemic_summary)
        epistemic_df.to_csv(f'{output_dir}/epistemic_sensitivity.csv', index=False)
        logger.info(f"Saved: epistemic_sensitivity.csv")


def print_summary_tables(
    results_df: pd.DataFrame, 
    logger: logging.Logger,
    n_std: float = N_STD_CONSERVATIVE,
) -> None:
    """Print nicely formatted summary tables."""
    
    logger.info("\n" + "=" * 100)
    logger.info(f"SUMMARY TABLES (£/tCO2 over 5 years, conservative estimate: mean + {n_std}*std)")
    logger.info("=" * 100)
    
    # INTERNAL SWEEP - focus on solid_wall_internal
    print("\n" + "=" * 110)
    print("INTERNAL WALL IMPROVEMENT SWEEP")
    print("(External factor fixed at 0.20)")
    print(f"Epistemic aggregation: mean(p50) + {n_std}*std(p50) per building")
    print("=" * 110)
    print(f"{'Factor':<8} {'Building Category':<25} {'N':>6} {'Median':>12} {'Mean':>12} {'P25':>10} {'P75':>10} {'%<2000':>8}")
    print("-" * 110)
    
    internal_df = results_df[results_df['sweep_type'] == 'internal']
    for factor in sorted(internal_df['internal_factor'].unique()):
        factor_data = internal_df[internal_df['internal_factor'] == factor]
        for _, row in factor_data.sort_values('building_category').iterrows():
            cat = row.get('building_category', 'unknown')
            n = row.get('n', 0)
            median = row.get('median', np.nan)
            mean = row.get('mean', np.nan)
            p25 = row.get('p25', np.nan)
            p75 = row.get('p75', np.nan)
            pct2000 = row.get('pct_below_2000', np.nan)
            
            # Highlight the relevant category
            highlight = " <--" if cat == 'solid_wall_internal' else ""
            print(f"{factor:<8.2f} {cat:<25} {n:>6} {median:>12.0f} {mean:>12.0f} {p25:>10.0f} {p75:>10.0f} {pct2000:>7.1f}%{highlight}")
        print("-" * 110)
    
    # EXTERNAL SWEEP - focus on solid_wall_external
    print("\n" + "=" * 110)
    print("EXTERNAL WALL IMPROVEMENT SWEEP")
    print("(Internal factor fixed at 0.10)")
    print(f"Epistemic aggregation: mean(p50) + {n_std}*std(p50) per building")
    print("=" * 110)
    print(f"{'Factor':<8} {'Building Category':<25} {'N':>6} {'Median':>12} {'Mean':>12} {'P25':>10} {'P75':>10} {'%<2000':>8}")
    print("-" * 110)
    
    external_df = results_df[results_df['sweep_type'] == 'external']
    for factor in sorted(external_df['external_factor'].unique()):
        factor_data = external_df[external_df['external_factor'] == factor]
        for _, row in factor_data.sort_values('building_category').iterrows():
            cat = row.get('building_category', 'unknown')
            n = row.get('n', 0)
            median = row.get('median', np.nan)
            mean = row.get('mean', np.nan)
            p25 = row.get('p25', np.nan)
            p75 = row.get('p75', np.nan)
            pct2000 = row.get('pct_below_2000', np.nan)
            
            # Highlight the relevant category
            highlight = " <--" if cat == 'solid_wall_external' else ""
            print(f"{factor:<8.2f} {cat:<25} {n:>6} {median:>12.0f} {mean:>12.0f} {p25:>10.0f} {p75:>10.0f} {pct2000:>7.1f}%{highlight}")
        print("-" * 110)
    
    # FOCUSED SUMMARY - just the relevant categories
    print("\n" + "=" * 90)
    print("FOCUSED SUMMARY: Relevant Building Categories Only")
    print(f"(Conservative estimate: mean + {n_std}*std across epistemic runs)")
    print("=" * 90)
    
    print("\nINTERNAL FACTOR SWEEP (solid_wall_internal buildings):")
    print(f"{'Factor':<10} {'N':>8} {'Median £/tCO2':>15} {'% < £2000':>12} {'% < £3000':>12}")
    print("-" * 60)
    solid_int = internal_df[internal_df['building_category'] == 'solid_wall_internal']
    for factor in sorted(solid_int['internal_factor'].unique()):
        row = solid_int[solid_int['internal_factor'] == factor].iloc[0] if len(solid_int[solid_int['internal_factor'] == factor]) > 0 else None
        if row is not None:
            print(f"{factor:<10.2f} {row['n']:>8} {row['median']:>15.0f} {row['pct_below_2000']:>11.1f}% {row['pct_below_3000']:>11.1f}%")
    
    print("\nEXTERNAL FACTOR SWEEP (solid_wall_external buildings):")
    print(f"{'Factor':<10} {'N':>8} {'Median £/tCO2':>15} {'% < £2000':>12} {'% < £3000':>12}")
    print("-" * 60)
    solid_ext = external_df[external_df['building_category'] == 'solid_wall_external']
    for factor in sorted(solid_ext['external_factor'].unique()):
        row = solid_ext[solid_ext['external_factor'] == factor].iloc[0] if len(solid_ext[solid_ext['external_factor'] == factor]) > 0 else None
        if row is not None:
            print(f"{factor:<10.2f} {row['n']:>8} {row['median']:>15.0f} {row['pct_below_2000']:>11.1f}% {row['pct_below_3000']:>11.1f}%")
    
    # Break-even analysis
    print("\n" + "=" * 90)
    print("BREAK-EVEN ANALYSIS (50% of buildings below threshold)")
    print("Note: Values are 5-year £/tCO2. Divide by 6 for ~30-year equivalent.")
    print("=" * 90)
    
    for threshold in ATTRACTIVE_THRESHOLDS_5YR:
        pct_col = f'pct_below_{threshold}' if f'pct_below_{threshold}' in results_df.columns else None
        
        # Internal
        if pct_col and len(solid_int) > 0 and pct_col in solid_int.columns:
            above_50 = solid_int[solid_int[pct_col] >= 50]
            if not above_50.empty:
                min_factor = above_50['internal_factor'].min()
                print(f"  Internal solid wall @ £{threshold}/tCO2 (5yr): factor >= {min_factor:.2f}")
            else:
                print(f"  Internal solid wall @ £{threshold}/tCO2 (5yr): not achieved in range")
        
        # External
        if pct_col and len(solid_ext) > 0 and pct_col in solid_ext.columns:
            above_50 = solid_ext[solid_ext[pct_col] >= 50]
            if not above_50.empty:
                min_factor = above_50['external_factor'].min()
                print(f"  External solid wall @ £{threshold}/tCO2 (5yr): factor >= {min_factor:.2f}")
            else:
                print(f"  External solid wall @ £{threshold}/tCO2 (5yr): not achieved in range")
        
        print()


# ========================================
# MAIN ENTRY POINT
# ========================================

def run_parameter_sweep(
    batch_paths: List[str],
    output_base_dir: str,
    n_postcodes: int,
    n_epistemic_runs: int = 5,
    max_buildings: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
    prob_external: float = 0.5,
    n_std: float = 1.0,
     skip_existing: bool = True,  # NEW
) -> Optional[pd.DataFrame]:
    """
    Run wall improvement factor parameter sweep across building data.
    
    Performs two sweeps:
    1. Internal wall factors (with external fixed at 0.20)
    2. External wall factors (with internal fixed at 0.10)
    
    Epistemic aggregation:
    - Per building: mean(p50) + n_std * std(p50) across epistemic runs
    - Then compute summary statistics across buildings
    
    Returns aggregated results by building category.
    """
    
    if skip_existing:
        original_count = len(batch_paths)
        batch_paths = [
            bp for bp in batch_paths 
            if not check_batch_complete(output_base_dir, bp)
        ]
        skipped = original_count - len(batch_paths)
        print(f'Skipped: {skipped}') 
        if skipped > 0:
            print(f"Skipping {skipped} already-completed batches")
        
        if not batch_paths:
            print("All batches already complete!")
            return None
    
    config = SweepConfig(
        n_postcodes=n_postcodes,
        n_epistemic_runs=n_epistemic_runs,
        max_buildings=max_buildings,
        random_sample=random_sample,
        seed=seed,
        prob_external=prob_external,
        n_std=n_std,
    )

    # Setup
    output_dir, timestamp = setup_output_directory(output_base_dir)
    logger = setup_logging(output_dir, timestamp)
    log_run_header(config, len(batch_paths), logger)

    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA FROM ALL BATCHES")
    logger.info("=" * 70)
    
    building_data = load_all_batches(batch_paths, config, logger)
    if building_data is None:
        return None

    # Preprocess
    building_data = preprocess_buildings(building_data, config, logger)

    # Configure retrofit model
    retrofit_config = RetrofitConfig(
        existing_intervention_probs={
            'loft_insulation': 0,
            'floor_insulation': 0,
            'window_upgrades': 0,
            'roof_scaling_factor': 0.8,
        }
    )
    
    epistemic_df = generate_epistemic_scenarios_lhs(n_epistemic_runs)

    # Run sweeps
    internal_detailed, internal_agg = run_internal_sweep(
        building_data, retrofit_config, n_epistemic_runs, epistemic_df, logger, n_std
    )
    
    external_detailed, external_agg = run_external_sweep(
        building_data, retrofit_config, n_epistemic_runs, epistemic_df, logger, n_std
    )

    # Combine and save
    all_detailed = internal_detailed + external_detailed
    all_aggregated = internal_agg + external_agg
    
    return save_results(all_aggregated, all_detailed, output_dir, logger, n_std)


# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    args = parse_args()
    
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
    print(f"Prob external insulation: {args.prob_external} (for solid wall internal/external split)")
    print(f"Conservative estimate: mean(p50) + {args.n_std}*std(p50)")
    
    run_parameter_sweep(
        batch_paths=batch_paths,
        output_base_dir=args.output,
        n_postcodes=args.n_postcodes,
        n_epistemic_runs=args.n_epistemic,
        max_buildings=args.max_buildings,
        random_sample=args.random_sample,
        seed=args.seed,
        prob_external=args.prob_external,
        n_std=args.n_std,
    )