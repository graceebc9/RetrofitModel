import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION - TUNE THESE PARAMETERS
# ============================================================================

CONFIG = {
    # CV threshold method: 'min', 'percentile_only', 'absolute_only'
    'threshold_method': 'min',
    
    # Absolute CV threshold (used if method is 'min' or 'absolute_only')
    'cv_threshold': 0.5,
    
    # Percentile to use (used if method is 'min' or 'percentile_only')
    'percentile': 95,
    
    # Cleaning strategy: 'intersection' (all metrics stable) or 'union' (any metric stable)
    'cleaning_strategy': 'intersection',
    
    # Allow per-metric thresholds (overrides global settings)
    'per_metric_thresholds': {
        # 'hp_only': {'cv_threshold': 0.3, 'percentile': 90},
        # 'heat_wall': {'cv_threshold': 0.5, 'percentile': 95},
    },
    
    # Output settings
    'save_pass1_results': True,
    'save_pass2_results': True,
    'save_cleaned_files': True,
    'output_dir': 'cleaned_logs'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_data_for_postanalysis_greedy(pre_df, scenario_list, years, gas_carbon_factor, elec_carbon_factor):
    """
    Placeholder for your actual data preparation function
    Replace with your implementation
    """
    proc_df = prepare_data_for_postanalysis(
            pre_df, 
            scenario_list, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
    return proc_df


def get_threshold_for_metric(metric_name, config):
    """
    Get threshold settings for a specific metric
    Allows per-metric overrides
    """
    if metric_name in config['per_metric_thresholds']:
        return config['per_metric_thresholds'][metric_name]
    else:
        return {
            'cv_threshold': config['cv_threshold'],
            'percentile': config['percentile']
        }


def calculate_cutoff(cv_values, method, cv_threshold, percentile):
    """
    Calculate CV cutoff based on specified method
    
    Methods:
    - 'min': min(absolute_threshold, percentile)
    - 'percentile_only': use percentile only
    - 'absolute_only': use absolute threshold only
    """
    if method == 'min':
        cv_percentile = cv_values.quantile(percentile / 100)
        return min(cv_threshold, cv_percentile), cv_percentile
    
    elif method == 'percentile_only':
        cv_percentile = cv_values.quantile(percentile / 100)
        return cv_percentile, cv_percentile
    
    elif method == 'absolute_only':
        cv_percentile = cv_values.quantile(percentile / 100)
        return cv_threshold, cv_percentile
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")


# ============================================================================
# PASS 1: Calculate spread across all files
# ============================================================================

def pass1_calculate_all_spreads(files_list, metrics_dict, config):
    """
    PASS 1: Process all files to calculate spread metrics
    WITHOUT filtering - just gather statistics
    """
    all_spread_by_metric = {name: [] for name in metrics_dict.keys()}
    
    print(f"{'='*70}")
    print(f"PASS 1: CALCULATING SPREAD METRICS ACROSS ALL FILES")
    print(f"{'='*70}")
    print(f"Total files: {len(files_list)}")
    
    for i, filepath in enumerate(files_list):
        print(f"\n[{i+1}/{len(files_list)}] Processing: {Path(filepath).name}")
        
        # Load and prepare data
        pre_df = pd.read_csv(filepath)
        
        # You'll need to define these variables or pass them as parameters
        df = prepare_data_for_postanalysis_greedy(
            pre_df, 
            scenario_list, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
        # Calculate spread for each metric
        for metric_name, metric_col in metrics_dict.items():
            if metric_col not in df.columns:
                print(df.columns)
                print(f"  WARNING: [{metric_name}] Column '{metric_col}' not found, skipping")
                continue
                
            spread = df.groupby('upn')[metric_col].agg([
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('q25', lambda x: x.quantile(0.25)),
                ('q75', lambda x: x.quantile(0.75)),
                ('count', 'count')
            ]).reset_index()
            
            spread['range'] = spread['max'] - spread['min']
            spread['iqr'] = spread['q75'] - spread['q25']
            spread['cv'] = (spread['std'] / spread['mean']).abs()
            spread['relative_range'] = (spread['range'] / spread['mean']).abs()
            spread['metric_name'] = metric_name
            spread['source_file'] = Path(filepath).name
            
            all_spread_by_metric[metric_name].append(spread)
            
            print(f"  [{metric_name}] Buildings: {len(spread)}, "
                  f"Median CV: {spread['cv'].median():.3f}, Max CV: {spread['cv'].max():.3f}")
    
    # Combine all spreads
    combined_spread_by_metric = {
        metric_name: pd.concat(spreads, ignore_index=True)
        for metric_name, spreads in all_spread_by_metric.items()
        if len(spreads) > 0
    }
    
    return combined_spread_by_metric


# ============================================================================
# Determine global CV cutoffs
# ============================================================================

def determine_global_cv_cutoffs(combined_spread_by_metric, config):
    """
    Determine CV cutoffs based on ALL buildings across ALL files
    Supports tunable threshold methods and per-metric overrides
    """
    cutoffs = {}
    cutoff_details = {}
    
    print(f"\n{'='*70}")
    print(f"DETERMINING GLOBAL CV CUTOFFS")
    print(f"{'='*70}")
    print(f"Method: {config['threshold_method']}")
    
    for metric_name, spread_df in combined_spread_by_metric.items():
        # Get threshold settings for this metric
        metric_config = get_threshold_for_metric(metric_name, config)
        
        # Calculate cutoff
        cutoff, cv_percentile = calculate_cutoff(
            spread_df['cv'],
            config['threshold_method'],
            metric_config['cv_threshold'],
            metric_config['percentile']
        )
        
        cutoffs[metric_name] = cutoff
        
        n_stable = (spread_df['cv'] <= cutoff).sum()
        n_total = len(spread_df)
        
        # Store details
        cutoff_details[metric_name] = {
            'cutoff': cutoff,
            'cv_percentile': cv_percentile,
            'absolute_threshold': metric_config['cv_threshold'],
            'percentile': metric_config['percentile'],
            'method': config['threshold_method'],
            'n_total': n_total,
            'n_stable': n_stable,
            'pct_stable': 100 * n_stable / n_total
        }
        
        print(f"\n[{metric_name}]")
        print(f"  Total buildings (all files): {n_total}")
        print(f"  CV {metric_config['percentile']}th percentile: {cv_percentile:.3f}")
        print(f"  Absolute threshold: {metric_config['cv_threshold']:.3f}")
        print(f"  Method: {config['threshold_method']}")
        print(f"  ✓ GLOBAL CUTOFF: {cutoff:.3f}")
        print(f"  Buildings that would be stable: {n_stable} ({100*n_stable/n_total:.1f}%)")
        print(f"  Buildings that would be filtered: {n_total-n_stable} ({100*(n_total-n_stable)/n_total:.1f}%)")
    
    # Save cutoffs for reference
    cutoff_df = pd.DataFrame.from_dict(cutoff_details, orient='index').reset_index()
    cutoff_df.rename(columns={'index': 'metric'}, inplace=True)
    cutoff_df.to_csv('global_cv_cutoffs.csv', index=False)
    print(f"\nGlobal cutoffs saved to: global_cv_cutoffs.csv")
    
    return cutoffs, cutoff_details


# ============================================================================
# Create cleaned log file
# ============================================================================

def create_cleaned_log_file_multi_metric(filepath, spread_results, strategy, output_dir):
    """
    Create a cleaned log file with only stable buildings
    
    Args:
        filepath: Original file path
        spread_results: Dict of {metric_name: spread_df with 'is_stable' column}
        strategy: 'intersection' or 'union'
        output_dir: Directory to save cleaned files
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load original data
    df = pd.read_csv(filepath)
    
    if strategy == 'intersection':
        # Keep only UPNs that are stable in ALL metrics
        stable_upns = None
        
        for metric_name, spread in spread_results.items():
            metric_stable_upns = set(spread[spread['is_stable']]['upn'])
            
            if stable_upns is None:
                stable_upns = metric_stable_upns
            else:
                stable_upns &= metric_stable_upns
        
        stable_upns = list(stable_upns) if stable_upns else []
        
    elif strategy == 'union':
        # Keep UPNs that are stable in ANY metric
        stable_upns = set()
        
        for metric_name, spread in spread_results.items():
            metric_stable_upns = set(spread[spread['is_stable']]['upn'])
            stable_upns |= metric_stable_upns
        
        stable_upns = list(stable_upns)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Filter dataframe
    df_cleaned = df[df['upn'].isin(stable_upns)]
    
    # Save cleaned file
    original_name = Path(filepath).stem
    cleaned_filepath = output_path / f"{original_name}_cleaned.csv"
    df_cleaned.to_csv(cleaned_filepath, index=False)
    
    print(f"  Created cleaned file: {cleaned_filepath.name}")
    print(f"    Original buildings: {df['upn'].nunique()}")
    print(f"    Stable buildings: {len(stable_upns)}")
    print(f"    Retention rate: {100*len(stable_upns)/df['upn'].nunique():.1f}%")
    
    return cleaned_filepath


# ============================================================================
# PASS 2: Apply cutoffs and create cleaned files
# ============================================================================

def pass2_apply_cutoffs_and_clean(files_list, metrics_dict, global_cutoffs, config):
    """
    PASS 2: Apply the global CV cutoffs to filter and clean each file
    """
    all_spread_by_metric = {name: [] for name in metrics_dict.keys()}
    cleaned_files = []
    
    print(f"\n{'='*70}")
    print(f"PASS 2: APPLYING GLOBAL CUTOFFS AND CREATING CLEANED FILES")
    print(f"{'='*70}")
    print(f"Strategy: {config['cleaning_strategy'].upper()} - "
          f"Buildings kept if {'ALL' if config['cleaning_strategy']=='intersection' else 'ANY'} metrics stable")
    
    for i, filepath in enumerate(files_list):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(files_list)}] File: {Path(filepath).name}")
        print(f"{'='*70}")
        
        # Load and prepare data
        pre_df = pd.read_csv(filepath)
        df = pre_df  # Replace with your prepare_data function if needed
        
        # Calculate spread and apply global cutoffs
        spread_results = {}
        
        for metric_name, metric_col in metrics_dict.items():
            if metric_col not in df.columns:
                continue
            
            spread = df.groupby('upn')[metric_col].agg([
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('count', 'count')
            ]).reset_index()
            
            spread['cv'] = (spread['std'] / spread['mean']).abs()
            spread['metric_name'] = metric_name
            spread['source_file'] = Path(filepath).name
            
            # Apply GLOBAL cutoff
            cutoff = global_cutoffs[metric_name]
            spread['is_stable'] = spread['cv'] <= cutoff
            spread['cv_cutoff'] = cutoff
            
            n_stable = spread['is_stable'].sum()
            print(f"  [{metric_name}] Cutoff: {cutoff:.3f} -> "
                  f"Stable: {n_stable}/{len(spread)} ({100*n_stable/len(spread):.1f}%)")
            
            spread_results[metric_name] = spread
            all_spread_by_metric[metric_name].append(spread)
        
        # Create cleaned file
        if config['save_cleaned_files']:
            cleaned_filepath = create_cleaned_log_file_multi_metric(
                filepath, 
                spread_results, 
                config['cleaning_strategy'],
                config['output_dir']
            )
            cleaned_files.append(cleaned_filepath)
    
    # Combine all spreads
    combined_spread_by_metric = {
        metric_name: pd.concat(spreads, ignore_index=True)
        for metric_name, spreads in all_spread_by_metric.items()
        if len(spreads) > 0
    }
    
    return combined_spread_by_metric, cleaned_files


# ============================================================================
# Main pipeline
# ============================================================================

def process_all_files_with_global_cutoffs(files_list, metrics_dict, config=None):
    """
    Main pipeline with TWO-PASS approach for consistent CV cutoffs
    """
    if config is None:
        config = CONFIG
    
    print(f"{'='*70}")
    print(f"PRE-OPTIMIZATION PIPELINE WITH GLOBAL CV CUTOFFS")
    print(f"{'='*70}")
    print(f"Total files: {len(files_list)}")
    print(f"Metrics: {len(metrics_dict)}")
    print(f"Threshold method: {config['threshold_method']}")
    print(f"CV threshold: {config['cv_threshold']}")
    print(f"Percentile: {config['percentile']}")
    print(f"Cleaning strategy: {config['cleaning_strategy']}")
    
    # PASS 1: Calculate spreads across all files
    combined_spreads_pass1 = pass1_calculate_all_spreads(files_list, metrics_dict, config)
    
    # Save pass 1 results
    if config['save_pass1_results']:
        for metric_name, spread_df in combined_spreads_pass1.items():
            spread_df.to_csv(f'pass1_spread_metrics_{metric_name}.csv', index=False)
    
    # Determine global cutoffs
    global_cutoffs, cutoff_details = determine_global_cv_cutoffs(
        combined_spreads_pass1, 
        config
    )
    
    # PASS 2: Apply global cutoffs to filter files
    combined_spreads_pass2, cleaned_files = pass2_apply_cutoffs_and_clean(
        files_list, 
        metrics_dict, 
        global_cutoffs,
        config
    )
    
    # Save final results
    if config['save_pass2_results']:
        for metric_name, spread_df in combined_spreads_pass2.items():
            spread_df.to_csv(f'spread_metrics_{metric_name}.csv', index=False)
        
        all_combined = pd.concat(combined_spreads_pass2.values(), ignore_index=True)
        all_combined.to_csv('spread_metrics_all.csv', index=False)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE - FINAL SUMMARY")
    print(f"{'='*70}")
    
    for metric_name, spread_df in combined_spreads_pass2.items():
        n_stable = spread_df['is_stable'].sum()
        n_total = len(spread_df)
        cutoff = global_cutoffs[metric_name]
        
        print(f"\n[{metric_name}] Global cutoff: {cutoff:.3f}")
        print(f"  Total buildings: {n_total}")
        print(f"  Stable: {n_stable} ({100*n_stable/n_total:.1f}%)")
        print(f"  Filtered: {n_total-n_stable} ({100*(n_total-n_stable)/n_total:.1f}%)")
        print(f"  CV - Min: {spread_df['cv'].min():.3f}, "
              f"Median: {spread_df['cv'].median():.3f}, "
              f"Max: {spread_df['cv'].max():.3f}")
    
    print(f"\n{'='*70}")
    print(f"FILES SAVED:")
    print(f"{'='*70}")
    print(f"  - global_cv_cutoffs.csv (CV cutoffs used)")
    if config['save_pass1_results']:
        print(f"  - pass1_spread_metrics_*.csv (one per metric, pre-filtering)")
    if config['save_pass2_results']:
        print(f"  - spread_metrics_*.csv (one per metric, post-filtering)")
        print(f"  - spread_metrics_all.csv (all metrics combined)")
    if config['save_cleaned_files']:
        print(f"  - {config['output_dir']}/ (filtered log files)")
    
    return combined_spreads_pass2, cleaned_files, global_cutoffs, cutoff_details


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
import glob 
from src.RetrofitAnalysisUtils import load_data , prepare_data_for_postanalysis

if __name__ == "__main__":
    
    YEARS = 5
    N_SIMULATIONS = 5000

    GAS_CARBON_FACTOR=0.18      
    ELEC_CARBON_FACTOR=0.19338  
    scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']

    # Define your metrics
    METRIC_SHORTCUTS = {
        'hp_only': 'heat_pump_only_cost_per_total_energy_ton_heat_pump_only_mean',
        'heat_wall': 'joint_heat_wall_decay_cost_per_total_energy_ton_joint_heat_wall_decay_mean',
        'wall': 'wall_installation_cost_per_total_energy_ton_wall_installation_mean',
        'heat_ins': 'join_heat_ins_decay_cost_per_total_energy_ton_join_heat_ins_decay_mean',
        'loft': 'loft_installation_cost_per_total_energy_ton_loft_installation_mean',
        'join_heat_ins_decay': 'joint_heat_loft_decay_cost_per_total_energy_ton_joint_heat_loft_decay_mean'
    }
    
    # Define your files
    files_list = glob.glob( '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*csv' ) 
    
    # ========================================================================
    # OPTION 1: Use default config
    # ========================================================================
    # all_spread_metrics, cleaned_files_list, global_cutoffs, cutoff_details = \
    #     process_all_files_with_global_cutoffs(
    #         files_list=files_list,
    #         metrics_dict=METRIC_SHORTCUTS
    #     )
    
    # # ========================================================================
    # # OPTION 2: Custom config
    # # ========================================================================
    custom_config = {
        'threshold_method': 'min',  # Options: 'min', 'percentile_only', 'absolute_only'
        'cv_threshold': 0.4,  # Stricter threshold
        'percentile': 90,  # More conservative (keep top 90%)
        'cleaning_strategy': 'union',  # Keep only if ALL metrics stable
        'per_metric_thresholds': {
            'hp_only': {'cv_threshold': 0.3, 'percentile': 85},  # Stricter for heat pump
            'loft': {'cv_threshold': 0.5, 'percentile': 95},  # More lenient for loft
        },
        'save_pass1_results': True,
        'save_pass2_results': True,
        'save_cleaned_files': True,
        'output_dir': 'cleaned_logs'
    }
    
    all_spread_metrics, cleaned_files_list, global_cutoffs, cutoff_details = \
        process_all_files_with_global_cutoffs(
            files_list=files_list,
            metrics_dict=METRIC_SHORTCUTS,
            config=custom_config
        )
    
    # # ========================================================================
    # # OPTION 3: Quick parameter override (using default config as base)
    # # ========================================================================
    # quick_config = CONFIG.copy()
    # quick_config['cv_threshold'] = 0.3
    # quick_config['percentile'] = 85
    # quick_config['threshold_method'] = 'percentile_only'
    
    # all_spread_metrics, cleaned_files_list, global_cutoffs, cutoff_details = \
    #     process_all_files_with_global_cutoffs(
    #         files_list=files_list,
    #         metrics_dict=METRIC_SHORTCUTS,
    #         config=quick_config
    #     )