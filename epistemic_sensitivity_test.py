"""
Module: sensitivity_test_simple.py

Simplified sensitivity analysis for epistemic factors.
- Loads first batch only
- Tests 3 scenarios: wall_installation, loft_installation, heat_pump_only
- Auto-detects numeric output columns for variance comparison

Usage:
    python sensitivity_test_simple.py
"""

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
N_EPISTEMIC_RUNS = 5
RANDOM_SEED_OUTER = 42
SCENARIOS = ['wall_installation', 'loft_installation', 'heat_pump_only']

# Paths 
PC_SHP_PATH = '/rds/user/gb669/hpc-work/energy_map/data/postcode_polygons/codepoint-poly_5267291'
BUILDING_PATH = '/rds/user/gb669/hpc-work/energy_map/data/building_files/UKBuildings_Edition_15_new_format_upn.gpkg'
location_input_data_folder = '/home/gb669/rds/hpc-work/energy_map/data/input_data'
onsud_path_base = '/home/gb669/rds/hpc-work/energy_map/data/onsud_files/Data'

GAS_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_gas_2022.csv'
ELEC_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_all_meters_electricity_2022.csv'

# First batch
TEST_BATCH_PATH = 'batches/NE/batch_10.txt'
TEST_ONSUD_PATH = 'batches/NE/onsud_10.csv'

# Output directory
OUTPUT_DIR = '5_sensitivity_results'

# ========================================
# IMPORTS (after path setup)
# ========================================

# Add project root to path if needed
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.RetrofitEpistemic import generate_epistemic_scenarios_lhs, FACTOR_DEFAULTS
from src.RetrofitScenarioGenerator2DMC import RetrofitScenarioGenerator2DMC
from src.RetrofitModel2D import RetrofitModel2D
from src.RetrofitConfig import RetrofitConfig
from src.postcode_utils import load_ids_from_file, load_onsud_data, find_data_pc_joint
from src.conservation import load_conservation_shapefile
from src.RetrofitDownscale import load_scaled_gas_elec
from src.pre_process_buildings import pre_process_building_data
from src.retrofit_calc2D import get_conservation_area


# ========================================
# FACTOR DEFAULTS (central values)
# ========================================

FACTOR_DEFAULTS = {
    'time_scale_bias': 1.0,
    'decile_misclassification_bias': 0.0,
    'solid_wall_internal_improvement_factor': 0.10,
    'solid_wall_external_improvement_factor': 0.20,
    'regional_multipliers_uncertainty': 1.0,
    'age_band_multipliers_uncertainty': 1.0,
    'cost_scenario': 'central',
    'external_wall_probability': 0.5,
    'flat_fp_mean': 55,
    'flat_fp_std': 8,
    'flat_eff_mean': 0.75,
    'flat_eff_std': 0.05,
    'area_based_choice': 'mode',
}


# ========================================
# HELPER FUNCTIONS
# ========================================

def create_fixed_sampler(fixed_factors: Dict[str, Any]) -> Callable:
    """Create sampler with specific factors fixed."""
    def fixed_sampler(n_runs: int) -> pd.DataFrame:
        return generate_epistemic_scenarios_lhs(n_runs, fixed_factors=fixed_factors)
    return fixed_sampler


def load_test_data():
    """Load all required data for testing."""
    logger.info("Loading test data...")
    
    # Conservation areas
    conservation_data = load_conservation_shapefile(
        path=f'{root_dir}/src/global_avs/Conservation_Areas_-5503574965118299320'
    )
    
    # ONSUD data
    onsud_data = load_onsud_data(TEST_ONSUD_PATH, PC_SHP_PATH)
    
    # Scaled gas/elec
    scaled_gas_elec_data = load_scaled_gas_elec()
    
    # Gas deciles
    gas_deciles = pd.read_csv(f'{root_dir}/src/global_avs/neb_unfil_final_gas_deciles.csv')
    
    # Postcodes - just first one
    postcodes = load_ids_from_file(TEST_BATCH_PATH) 
    logger.info(f"Testing with postcode: {postcodes}")
    
    return {
        'conservation_data': conservation_data,
        'onsud_data': onsud_data,
        'scaled_gas_elec_data': scaled_gas_elec_data,
        'gas_deciles': gas_deciles,
        'postcodes': postcodes,
    }


def prepare_building_data(pc: str, data: dict) -> Optional[pd.DataFrame]:
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
    
    return building_data


def run_model_with_sampler(
    building_data: pd.DataFrame,
    retrofit_config: RetrofitConfig,
    epistemic_sampler: Callable,
    region: str = 'NE',
) -> Optional[pd.DataFrame]:
    """Run model with a specific epistemic sampler."""
    
    scenario_generator = RetrofitScenarioGenerator2DMC(
        n_epistemic_runs=N_EPISTEMIC_RUNS,
        epistemic_sampler=epistemic_sampler
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


def identify_output_metrics(df: pd.DataFrame) -> List[str]:
    """Identify key output metrics based on known column patterns."""
    
    # Known metric patterns:
    # - {sc}_capex_per_net_ton_co2_{sc}_{stat}
    # - {sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}
    # - {sc}_cost_{sc}_{stat}
    
    metric_cols = []
    
    for scenario in SCENARIOS:
        # Focus on 'mean' stats for variance comparison
        patterns = [
            f'{scenario}_capex_per_net_ton_co2_{scenario}_mean',
            f'{scenario}_total_energy_abs_co2_ton_samples_{scenario}_mean',
            f'{scenario}_cost_{scenario}_mean',
            # Also include std to see uncertainty propagation
            f'{scenario}_capex_per_net_ton_co2_{scenario}_std',
            f'{scenario}_total_energy_abs_co2_ton_samples_{scenario}_std',
            f'{scenario}_cost_{scenario}_std',
        ]
        
        for pattern in patterns:
            if pattern in df.columns:
                metric_cols.append(pattern)
    
    # If no matches, fall back to auto-detection
    if not metric_cols:
        logger.warning("No known metric patterns found, falling back to auto-detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['epistemic_run_id', 'epistemic__', 'upn', 'index']
        
        for col in numeric_cols:
            if not any(pattern in col for pattern in exclude_patterns):
                if df[col].std() > 0:
                    metric_cols.append(col)
    
    return metric_cols


# Key metrics for sensitivity ranking (one per scenario)
KEY_METRICS = [
    'wall_installation_cost_wall_installation_mean',
    'loft_installation_cost_loft_installation_mean',
    'heat_pump_only_cost_heat_pump_only_mean',
]


def compute_variance_summary(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, float]:
    """Compute variance metrics for output columns."""
    summary = {}
    
    for col in metric_cols:
        if col in df.columns:
            summary[f'{col}__var'] = df[col].var()
            summary[f'{col}__std'] = df[col].std()
            summary[f'{col}__cv'] = df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan
    
    return summary


# ========================================
# MAIN SENSITIVITY TEST
# ========================================

def run_sensitivity_test():
    """Run sensitivity analysis: baseline + each factor fixed."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load data
    data = load_test_data()
    pc = data['postcodes'][0]
    
    # Prepare building data (do this once)
    building_data = prepare_building_data(pc, data)
    if building_data is None:
        logger.error("Failed to prepare building data")
        return
    
    logger.info(f"Building data shape: {building_data.shape}")
    
    # Retrofit config
    retrofit_config = RetrofitConfig(
        existing_intervention_probs={
            'loft_insulation': 0,
            'floor_insulation': 0,
            'window_upgrades': 0,
            'roof_scaling_factor': 0.8,
        }
    )
    
    results_summary = []
    metric_cols = None  # Will be detected from first run
    
    # === BASELINE ===
    logger.info("=" * 60)
    logger.info("Running BASELINE (all factors vary)")
    logger.info("=" * 60)
    
    baseline_df = run_model_with_sampler(
        building_data=building_data,
        retrofit_config=retrofit_config,
        epistemic_sampler=generate_epistemic_scenarios_lhs,
    )
    
    if baseline_df is not None:
        baseline_df.to_csv(f'{OUTPUT_DIR}/baseline_{timestamp}.csv', index=False)
        
        # Identify metrics from baseline
        metric_cols = identify_output_metrics(baseline_df)
        logger.info(f"Detected {len(metric_cols)} output metrics")
        logger.info(f"Sample metrics: {metric_cols[:10]}")
        
        baseline_summary = compute_variance_summary(baseline_df, metric_cols)
        baseline_summary['configuration'] = 'baseline'
        baseline_summary['fixed_factor'] = None
        results_summary.append(baseline_summary)
    else:
        logger.error("Baseline run failed")
        return
    
    # === FIXED FACTOR RUNS ===
    for factor, default_value in FACTOR_DEFAULTS.items():
        logger.info("=" * 60)
        logger.info(f"Running with {factor} FIXED to {default_value}")
        logger.info("=" * 60)
        
        fixed_sampler = create_fixed_sampler({factor: default_value})
        
        fixed_df = run_model_with_sampler(
            building_data=building_data,
            retrofit_config=retrofit_config,
            epistemic_sampler=fixed_sampler,
        )
        
        if fixed_df is not None:
            fixed_df.to_csv(f'{OUTPUT_DIR}/fixed_{factor}_{timestamp}.csv', index=False)
            
            fixed_summary = compute_variance_summary(fixed_df, metric_cols)
            fixed_summary['configuration'] = f'fixed_{factor}'
            fixed_summary['fixed_factor'] = factor
            results_summary.append(fixed_summary)
            
            logger.info(f"Completed: {factor}")
        else:
            logger.warning(f"Failed: {factor}")
    
    # === COMPILE SUMMARY ===
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f'{OUTPUT_DIR}/sensitivity_summary_{timestamp}.csv', index=False)
    
    # === COMPUTE SENSITIVITY RANKING ===
    logger.info("=" * 60)
    logger.info("SENSITIVITY RANKING")
    logger.info("=" * 60)
    
    # Rank by each key metric
    all_rankings = []
    
    for key_metric in KEY_METRICS:
        var_col = f'{key_metric}__var'
        
        if var_col not in summary_df.columns:
            logger.warning(f"Metric not found: {key_metric}")
            continue
        
        baseline_var = summary_df[summary_df['configuration'] == 'baseline'][var_col].values[0]
        
        if baseline_var == 0:
            logger.warning(f"Baseline variance is 0 for {key_metric}")
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
    ranking_df.to_csv(f'{OUTPUT_DIR}/sensitivity_ranking_{timestamp}.csv', index=False)
    
    # Print summary for each metric
    print("\n" + "=" * 70)
    print("SENSITIVITY RANKING BY METRIC")
    print("=" * 70)
    print("Higher % = factor contributes more to output uncertainty\n")
    
    for key_metric in KEY_METRICS:
        metric_ranking = ranking_df[ranking_df['metric'] == key_metric].sort_values(
            'var_reduction_pct', ascending=False
        )
        
        if metric_ranking.empty:
            continue
        
        # Extract scenario name for cleaner display
        scenario = key_metric.split('_cost_')[0] if '_cost_' in key_metric else key_metric
        
        print(f"\n{scenario.upper()} (cost)")
        print("-" * 50)
        print(f"{'Factor':<45} {'Var Reduction %':>12}")
        print("-" * 50)
        
        for _, row in metric_ranking.head(10).iterrows():
            print(f"{row['factor']:<45} {row['var_reduction_pct']:>11.1f}%")
    
    # Also create a summary: average ranking across all metrics
    avg_ranking = ranking_df.groupby('factor')['var_reduction_pct'].mean().sort_values(ascending=False)
    
    print("\n" + "=" * 70)
    print("AVERAGE SENSITIVITY ACROSS ALL METRICS")
    print("=" * 70)
    print(f"{'Factor':<45} {'Avg Var Reduction %':>12}")
    print("-" * 57)
    for factor, pct in avg_ranking.items():
        print(f"{factor:<45} {pct:>11.1f}%")
    
    avg_ranking.to_csv(f'{OUTPUT_DIR}/sensitivity_avg_ranking_{timestamp}.csv')
    
    logger.info(f"\nResults saved to {OUTPUT_DIR}/")
    return summary_df, ranking_df


if __name__ == "__main__":
    run_sensitivity_test()