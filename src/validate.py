# src/validate_all_scenarios.py

import pandas as pd
import numpy as np
from typing import List, Dict
import re


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

import os
from datetime import datetime

def validate(df, output_dir):
    # Create output directory if it doesn't exist
    output_dir = f'{output_dir}/validation_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'validation_log_{timestamp}.txt')
    
    def log_print(message=''):
        """Print to console and write to log file"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    log_print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Validate all scenarios
    summary = validate_all_scenarios(df, verbose=True)

    # Show cost comparison
    log_print(f"\n{'='*80}")
    log_print("COST COMPARISON (Most to Least Expensive)")
    log_print(f"{'='*80}\n")
    cost_comp = compare_all_scenarios(df, metric='cost')
    if len(cost_comp) > 0:
        log_print(cost_comp.to_string(index=False))
    else:
        log_print("No cost data found")

    # Show energy comparison
    log_print(f"\n{'='*80}")
    log_print("ENERGY SAVINGS COMPARISON")
    log_print(f"{'='*80}\n")
    energy_comp = compare_all_scenarios(df, metric='energy')
    if len(energy_comp) > 0:
        log_print(energy_comp.to_string(index=False))
    else:
        log_print("No energy data found")

    # Save summary
    summary_path = os.path.join(output_dir, f'validation_summary_{timestamp}.csv')
    summary.to_csv(summary_path, index=False)
    log_print(f"\n✅ Summary saved to: {summary_path}")

    validate_percentiles(df)

    # Validate wall energy columns
    validation_df, invalid_rows = validate_wall_energy_columns(df)

    # Filter for actual invalid rows: not domestic outbuildings AND not already insulated
    actual_invalid = invalid_rows[
        (invalid_rows['premise_type'] != 'Domestic outbuilding') & 
        (invalid_rows['wall_insulated'] == False)
    ]
    
    log_print(f"\nActual invalid rows (excluding outbuildings and already insulated): {len(actual_invalid):,}")
    
    # Save actual invalid rows to CSV
    if len(actual_invalid) > 0:
        invalid_path = os.path.join(output_dir, f'invalid_rows_{timestamp}.csv')
        actual_invalid.to_csv(invalid_path, index=False)
        log_print(f"Invalid rows saved to: {invalid_path}")
        log_print("\nSample of invalid rows:")
        log_print(actual_invalid[['postcode', 'premise_age', 'inferred_wall_type', 
                                   'inferred_insulation_type', 'wall_insulated']].head(10).to_string(index=False))
    
    log_print(f"\n✅ Validation log saved to: {log_file}")

# def validate(df):


#     print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

#     # Validate all scenarios
#     summary = validate_all_scenarios(df, verbose=True)

#     # Show cost comparison
#     print(f"\n{'='*80}")
#     print("COST COMPARISON (Most to Least Expensive)")
#     print(f"{'='*80}\n")
#     cost_comp = compare_all_scenarios(df, metric='cost')
#     if len(cost_comp) > 0:
#         print(cost_comp.to_string(index=False))
#     else:
#         print("No cost data found")

#     # Show energy comparison
#     print(f"\n{'='*80}")
#     print("ENERGY SAVINGS COMPARISON")
#     print(f"{'='*80}\n")
#     energy_comp = compare_all_scenarios(df, metric='energy')
#     if len(energy_comp) > 0:
#         print(energy_comp.to_string(index=False))
#     else:
#         print("No energy data found")

#     # Save summary
#     summary.to_csv('validation_summary.csv', index=False)
#     print(f"\n✅ Summary saved to: validation_summary.csv")

#     validate_percentiles(df)

#     validation_df, invalid_rows = validate_wall_energy_columns(df)


#     inv_shape = invalid_rows[(invalid_rows['premise_type']!='Domestic outbuilding') & (invalid_rows['wall_insulated'] ==False) ][['postcode', 'premise_age', 'inferred_wall_type', 'inferred_insulation_type', 'wall_insulated']].shape
#     print('exclusing outbuilds and insualted: ', inv_shape)


def validate_wall_energy_columns(df):
    """
    Validate that energy columns are populated correctly based on wall type and insulation type.
    
    Rules:
    - If inferred_wall_type is 'cavity wall': only cavity column should be non-NaN
    - If inferred_wall_type is 'solid wall' and inferred_insulation_type is 'external': 
      only external column should be non-NaN
    - If inferred_wall_type is 'solid wall' and inferred_insulation_type is 'internal': 
      only internal column should be non-NaN
    
    Returns:
    - validation_df: DataFrame with validation results for each row
    - invalid_rows: DataFrame containing only rows that failed validation
    """
    print('Validate wall energy columns that we dont have much overlap ')
    
    validation_df = df.copy()
    
    # Short column names
    cavity_col = 'wall_installation_energy_cavity_wall_percentile_gas_mean'
    external_col = 'wall_installation_energy_solid_wall_external_percentile_gas_mean'
    internal_col = 'wall_installation_energy_solid_wall_internal_percentile_gas_mean'
    
    # Create boolean masks for NaN values
    is_cavity_nan = validation_df[cavity_col].isna()
    is_external_nan = validation_df[external_col].isna()
    is_internal_nan = validation_df[internal_col].isna()
    
    # Normalize strings to lowercase for comparison
    wall_type_lower = validation_df['inferred_wall_type'].str.lower()
    insulation_type_lower = validation_df['inferred_insulation_type'].str.lower()
    
    # Initialize columns
    validation_df['is_valid'] = False
    validation_df['validation_error'] = ''
    
    # Rule 1: Cavity wall - only cavity should be non-NaN
    cavity_mask = wall_type_lower == 'cavity_wall'
    cavity_valid = cavity_mask & ~is_cavity_nan & is_external_nan & is_internal_nan
    validation_df.loc[cavity_valid, 'is_valid'] = True
    
    # Cavity wall errors
    cavity_invalid = cavity_mask & ~cavity_valid
    validation_df.loc[cavity_invalid & is_cavity_nan, 'validation_error'] += 'cavity column is NaN; '
    validation_df.loc[cavity_invalid & ~is_external_nan, 'validation_error'] += 'external column is not NaN; '
    validation_df.loc[cavity_invalid & ~is_internal_nan, 'validation_error'] += 'internal column is not NaN; '
    
    # Rule 2: Solid wall + external insulation - only external should be non-NaN
    external_mask = (wall_type_lower == 'solid_wall') & (insulation_type_lower == 'external_wall_insulation')
    external_valid = external_mask & is_cavity_nan & ~is_external_nan & is_internal_nan
    validation_df.loc[external_valid, 'is_valid'] = True
    
    # External insulation errors
    external_invalid = external_mask & ~external_valid
    validation_df.loc[external_invalid & ~is_cavity_nan, 'validation_error'] += 'cavity column is not NaN; '
    validation_df.loc[external_invalid & is_external_nan, 'validation_error'] += 'external column is NaN; '
    validation_df.loc[external_invalid & ~is_internal_nan, 'validation_error'] += 'internal column is not NaN; '
    
    # Rule 3: Solid wall + internal insulation - only internal should be non-NaN
    internal_mask = (wall_type_lower == 'solid_wall') & (insulation_type_lower == 'internal_wall_insulation')
    internal_valid = internal_mask & is_cavity_nan & is_external_nan & ~is_internal_nan
    validation_df.loc[internal_valid, 'is_valid'] = True
    
    # Internal insulation errors
    internal_invalid = internal_mask & ~internal_valid
    validation_df.loc[internal_invalid & ~is_cavity_nan, 'validation_error'] += 'cavity column is not NaN; '
    validation_df.loc[internal_invalid & ~is_external_nan, 'validation_error'] += 'external column is not NaN; '
    validation_df.loc[internal_invalid & is_internal_nan, 'validation_error'] += 'internal column is NaN; '
    
    # Handle special error cases
    wall_type_nan = validation_df['inferred_wall_type'].isna()
    validation_df.loc[wall_type_nan, 'validation_error'] = 'Wall type is NaN'
    
    solid_wall_no_insulation = (wall_type_lower == 'solid_wall') & validation_df['inferred_insulation_type'].isna()
    validation_df.loc[solid_wall_no_insulation, 'validation_error'] = 'Solid wall but insulation type is NaN'
    
    # Clean up trailing semicolons and spaces
    validation_df['validation_error'] = validation_df['validation_error'].str.rstrip('; ')
    
    # Get invalid rows
    invalid_rows = validation_df[~validation_df['is_valid']].copy()
    
    # Print summary
    total_rows = len(validation_df)
    valid_count = validation_df['is_valid'].sum()
    invalid_count = total_rows - valid_count
    
    print(f"Validation Summary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Valid rows: {valid_count} ({100*valid_count/total_rows:.1f}%)")
    print(f"  Invalid rows: {invalid_count} ({100*invalid_count/total_rows:.1f}%)")
    
    if invalid_count > 0:
        print(f"\nError breakdown:")
        error_counts = invalid_rows['validation_error'].value_counts()
        for error, count in error_counts.items():
            print(f"  {error}: {count}")
    
    return validation_df, invalid_rows



def detect_scenarios(df: pd.DataFrame) -> List[str]:
    """
    Detect unique scenario prefixes from DataFrame columns.
    
    Works with new column formats like:
      'wall_installation_cost_cavity_wall_percentile_mean'
      'wall_installation_energy_cavity_wall_percentile_gas_mean'
      'loft_installation_cost_loft_percentile_p5'
    
    Returns:
    --------
    list : Scenario names (e.g. ['wall_installation', 'loft_installation'])
    """
    scenarios = set()

    for col in df.columns:
        # Match prefix before _cost_ or _energy_
        m = re.match(r"^([a-z0-9_]+)_(?:cost|energy)_", col, flags=re.IGNORECASE)
        if m:
            scenarios.add(m.group(1))

    return sorted(list(scenarios))



def safe_numeric_column(series: pd.Series) -> pd.Series:
    """
    Convert a series to numeric, handling mixed types.
    
    Parameters:
    -----------
    series : pd.Series
        Series that may contain mixed types
    
    Returns:
    --------
    pd.Series : Numeric series with non-numeric values as NaN
    """
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(series, errors='coerce')


def safe_mean(series: pd.Series) -> float:
    """Safely calculate mean, handling non-numeric values."""
    numeric_series = safe_numeric_column(series)
    valid_values = numeric_series.dropna()
    return valid_values.mean() if len(valid_values) > 0 else np.nan


def safe_median(series: pd.Series) -> float:
    """Safely calculate median, handling non-numeric values."""
    numeric_series = safe_numeric_column(series)
    valid_values = numeric_series.dropna()
    return valid_values.median() if len(valid_values) > 0 else np.nan


def safe_min(series: pd.Series) -> float:
    """Safely calculate min, handling non-numeric values."""
    numeric_series = safe_numeric_column(series)
    valid_values = numeric_series.dropna()
    return valid_values.min() if len(valid_values) > 0 else np.nan


def safe_max(series: pd.Series) -> float:
    """Safely calculate max, handling non-numeric values."""
    numeric_series = safe_numeric_column(series)
    valid_values = numeric_series.dropna()
    return valid_values.max() if len(valid_values) > 0 else np.nan


def validate_all_scenarios(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Validate all scenarios in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe with multiple scenario columns
    verbose : bool
        If True, print detailed output
    
    Returns:
    --------
    pd.DataFrame : Summary of validation results for all scenarios
    """
    
    scenarios = detect_scenarios(df)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-SCENARIO VALIDATION")
        print(f"{'='*80}")
        print(f"Total rows: {len(df):,}")
        print(f"Scenarios detected: {len(scenarios)}")
        print(f"  {', '.join(scenarios)}")
        print(f"{'='*80}\n")
    
    results = []
    
    for scenario in scenarios:
        if verbose:
            print(f"\n{'#'*80}")
            print(f"# SCENARIO: {scenario.upper()}")
            print(f"{'#'*80}\n")
        
        scenario_results = validate_single_scenario(df, scenario, verbose=verbose)
        scenario_results['scenario'] = scenario
        results.append(scenario_results)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL SCENARIOS")
        print(f"{'='*80}\n")
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}\n")
    
    return summary_df

def validate_single_scenario(df: pd.DataFrame, scenario: str, verbose: bool = True) -> Dict:
    """
    Validate a single scenario (supports new '..._cost_...' and '..._energy_...' column formats).
    """

    results = {
        'scenario': scenario,
        'rows_total': len(df)
    }

    # === COST COLUMNS ===
    cost_cols = [c for c in df.columns if c.startswith(f"{scenario}_cost_")]
    if len(cost_cols) == 0:
        if verbose:
            print(f"❌ No cost columns found for scenario: {scenario}")
    else:
        mean_cols = [c for c in cost_cols if c.endswith("_mean")]
        for mean_col in mean_cols:
            base = mean_col.removesuffix("_mean")
            p5_col = f"{base}_p5"
            p95_col = f"{base}_p95"
            std_col = f"{base}_std"

            # Load numeric safely
            cost_numeric = safe_numeric_column(df[mean_col])
            valid = cost_numeric.dropna()

            results[f"{base}_valid_rows"] = len(valid)
            results[f"{base}_mean"] = valid.mean() if len(valid) else np.nan
            results[f"{base}_median"] = valid.median() if len(valid) else np.nan
            results[f"{base}_min"] = valid.min() if len(valid) else np.nan
            results[f"{base}_max"] = valid.max() if len(valid) else np.nan
            results[f"{base}_negative_count"] = (cost_numeric < 0).sum()
            results[f"{base}_nan_count"] = cost_numeric.isna().sum()

            # Coeff of variation
            if std_col in df.columns:
                std_mean = safe_mean(df[std_col])
                results[f"{base}_cv"] = std_mean / results[f"{base}_mean"] if results[f"{base}_mean"] > 0 else np.nan

            # P95 / mean ratio
            if p95_col in df.columns:
                p95_mean = safe_mean(df[p95_col])
                results[f"{base}_p95_mean_ratio"] = p95_mean / results[f"{base}_mean"] if results[f"{base}_mean"] > 0 else np.nan

            if verbose:
                print(f"\n💰 COST: {base}")
                if not np.isnan(results[f"{base}_mean"]):
                    print(f"   Mean: £{results[f'{base}_mean']:,.0f}")
                    print(f"   Range: £{results[f'{base}_min']:,.0f} - £{results[f'{base}_max']:,.0f}")
                else:
                    print("   ⚠️ No valid cost data")
                if results[f"{base}_negative_count"] > 0:
                    print(f"   ⚠️ {results[f'{base}_negative_count']:,} negative values")
                else:
                    print("   ✅ All costs positive")

    # === ENERGY COLUMNS ===
    energy_cols = [c for c in df.columns if c.startswith(f"{scenario}_energy_")]
    if len(energy_cols) == 0:
        if verbose:
            print(f"\n❌ No energy columns found for scenario: {scenario}")
    else:
        # Group by gas/electricity
        energy_groups = {}
        for c in energy_cols:
            m = re.match(rf"{re.escape(scenario)}_energy_(.+?)_(gas|electricity)_(mean|p5|p95|std)$", c)
            if m:
                group_key = f"{m.group(1)}_{m.group(2)}"  # e.g. "cavity_wall_percentile_gas"
                energy_groups.setdefault(group_key, []).append(c)

        for energy_key, cols in energy_groups.items():
            base = f"{scenario}_energy_{energy_key}"
            mean_col = f"{base}_mean"
            if mean_col not in df.columns:
                continue

            energy_numeric = safe_numeric_column(df[mean_col])
            valid = energy_numeric.dropna()

            results[f"{base}_valid_rows"] = len(valid)
            results[f"{base}_mean"] = valid.mean() if len(valid) else np.nan
            results[f"{base}_median"] = valid.median() if len(valid) else np.nan
            results[f"{base}_min"] = valid.min() if len(valid) else np.nan
            results[f"{base}_max"] = valid.max() if len(valid) else np.nan
            results[f"{base}_negative_count"] = (energy_numeric < 0).sum()
            results[f"{base}_positive_count"] = (energy_numeric > 0).sum()
            results[f"{base}_nan_count"] = energy_numeric.isna().sum()

            # Identify if it's a percentile (savings negative)
            is_percentile = "percentile" in energy_key.lower()
            results[f"{base}_is_percentile"] = is_percentile
            if is_percentile:
                results[f"{base}_pct_with_savings"] = results[f"{base}_negative_count"] / len(df)
            else:
                results[f"{base}_pct_with_savings"] = results[f"{base}_positive_count"] / len(df)

            # Reasonable ranges
            if not np.isnan(results[f"{base}_mean"]):
                if is_percentile:
                    results[f"{base}_reasonable_range"] = -0.30 < results[f"{base}_mean"] < 0.20
                else:
                    results[f"{base}_reasonable_range"] = 0.02 < results[f"{base}_mean"] < 0.30
            else:
                results[f"{base}_reasonable_range"] = False

            if verbose:
                print(f"\n⚡ ENERGY ({energy_key}):")
                if not np.isnan(results[f"{base}_mean"]):
                    print(f"   Mean: {results[f'{base}_mean']:.1%}")
                    print(f"   Range: {results[f'{base}_min']:.1%} - {results[f'{base}_max']:.1%}")
                else:
                    print("   ⚠️ No valid energy data")
                if is_percentile:
                    print(f"   Negative (savings): {results[f'{base}_negative_count']:,} ({results[f'{base}_pct_with_savings']:.1%})")
                else:
                    print(f"   Positive (savings): {results[f'{base}_positive_count']:,} ({results[f'{base}_pct_with_savings']:.1%})")
                if results[f"{base}_reasonable_range"]:
                    print("   ✅ Mean in reasonable range")
                else:
                    print("   ⚠️ Mean outside typical range")

    return results


def compare_all_scenarios(df: pd.DataFrame, metric: str = 'cost') -> pd.DataFrame:
    """
    Compare a specific metric across all scenarios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    metric : str
        'cost' or 'energy'
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    
    scenarios = detect_scenarios(df)
    
    comparison_data = []
    
    for scenario in scenarios:
        if metric == 'cost':
            col = f'cost_{scenario}_cost_total_mean'
            if col in df.columns:
                comparison_data.append({
                    'scenario': scenario,
                    'mean': safe_mean(df[col]),
                    'median': safe_median(df[col]),
                    'p5': safe_mean(df[f'cost_{scenario}_cost_total_p5']) if f'cost_{scenario}_cost_total_p5' in df.columns else None,
                    'p95': safe_mean(df[f'cost_{scenario}_cost_total_p95']) if f'cost_{scenario}_cost_total_p95' in df.columns else None,
                })
        elif metric == 'energy':
            col = f'energy_{scenario}_energy_gas_mean'
            if col in df.columns:
                comparison_data.append({
                    'scenario': scenario,
                    'mean': safe_mean(df[col]),
                    'median': safe_median(df[col]),
                    'p5': safe_mean(df[f'energy_{scenario}_energy_gas_p5']) if f'energy_{scenario}_energy_gas_p5' in df.columns else None,
                    'p95': safe_mean(df[f'energy_{scenario}_energy_gas_p95']) if f'energy_{scenario}_energy_gas_p95' in df.columns else None,
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        if metric == 'cost':
            comparison_df = comparison_df.sort_values('mean', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('mean', ascending=True)
    
    return comparison_df


def validate_percentiles(res_df):
    print('Starting validate percentiles for wall and loft only ')
    # Example
    scenarios = ['wall_installation', 'loft_installation']
    interventions = ['solid_wall_percentile', 'cavity_wall_percentile', 'loft_percentile']

    # If you want to include energy and cost columns
    typp = ['energy', 'cost']   
    energy_types =['gas' , 'electricity']

    invalid_rows = []

    for scenario in scenarios:
        for intervention in interventions:
            for e in typp:
                if e=='energy':
                    for g in energy_types:
                        # find matching columns
                        p5_cols = [c for c in res_df.columns if scenario in c and intervention in c and e in c and '_p5' in c and g in c]
                        p50_cols = [c for c in res_df.columns if scenario in c and intervention in c and e in c and '_p50' in c and g in c]
                        p95_cols = [c for c in res_df.columns if scenario in c and intervention in c and e in c and '_p95' in c and g in c]
                        
                        # Ensure we found columns
                        if not (p5_cols and p50_cols and p95_cols):
                            continue
                        
                        # Check all columns
                        for col5, col50, col95 in zip(p5_cols, p50_cols, p95_cols):
                            invalid = res_df[(res_df[col5] >res_df[col50]) | (res_df[col50] > res_df[col95])]
                            if not invalid.empty:
                                invalid_rows.append({
                                    'scenario': scenario,
                                    'intervention': intervention,
                                    'energy_type': e,
                                    'p5': col5,
                                    'p50': col50,
                                    'p95': col95,
                                    'invalid_count': len(invalid)
                                })
                else: 

  

                    # find matching columns
                    p5_cols = [c for c in res_df.columns if scenario in c and intervention in c and e in c and '_p5' in c]
                    p50_cols = [c for c in res_df.columns if scenario in c and intervention in c and e in c and '_p50' in c]
                    p95_cols = [c for c in res_df.columns if scenario in c and intervention in c and e in c and '_p95' in c]
                    
                    # Ensure we found columns
                    if not (p5_cols and p50_cols and p95_cols):
                        continue
                    
                    # Check all columns
                    for col5, col50, col95 in zip(p5_cols, p50_cols, p95_cols):
                        invalid = res_df[(res_df[col5] >res_df[col50]) | (res_df[col50] > res_df[col95])]
                        if not invalid.empty:
                            invalid_rows.append({
                                'scenario': scenario,
                                'intervention': intervention,
                                'energy_type': e,
                                'p5': col5,
                                'p50': col50,
                                'p95': col95,
                                'invalid_count': len(invalid)
                            })

    invalid_df = pd.DataFrame(invalid_rows)
    if invalid_df.empty:
        print('Validation passed')
    else:
        raise Exception('Percentiles invalid')
