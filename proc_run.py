import glob
import pandas as pd
import numpy as np
from src.validate import validate
from src.visualisations import (
    plot_col_reduction_by_decile,
    plot_building_counts_by_age_band
)
from src.costs import calculate_and_plot_measure_analysis


# Configuration
DATA_DIR = '/Volumes/T9/2024_Data_downloads/2025_10_RetrofitModel/1_data_runs/NE'
OUTPUT_DIR_base = '/Volumes/T9/2024_Data_downloads/2025_10_RetrofitModel/2_costs_analysis'

N_MONTE_CARLO_RUNS = 100
GAS_CARBON_FACTOR_2022 = 0.2  # kg CO2/kWh
GAS_PRICE = 0.07  # £/kWh


def load_and_concatenate_data(file_pattern):
    """Load all CSV files matching the pattern and concatenate them."""
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    dataframes = [pd.read_csv(file) for file in files]
    return pd.concat(dataframes, ignore_index=True)
 
def proc_measures_optimized(df, measure_type, scenario_name, gas_carbon_factor_22, n_monte, gas_price):
    """
    Optimized version that calculates measures in-place without copying dataframe.
    Returns column names that were added.
    """
    # Pre-calculate common values to avoid repeated calculations
    total_gas_5y = df['total_gas_derived'] * 5
    sqrt_n_monte = np.sqrt(n_monte)
    
    # Get column prefixes
    energy_mean_col = f'{scenario_name}_installation_energy_{measure_type}_percentile_gas_mean'
    energy_std_col = f'{scenario_name}_installation_energy_{measure_type}_percentile_gas_std'
    cost_mean_col = f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'
    cost_std_col = f'{scenario_name}_installation_cost_{measure_type}_percentile_std'
    
    # Calculate 5-year kWh changes
    kwh_change_mean = total_gas_5y * df[energy_mean_col]
    kwh_change_std = total_gas_5y * df[energy_std_col]
    
    df[f'{scenario_name}_{measure_type}_5y_kwh_change'] = kwh_change_mean
    df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] = kwh_change_std
    
    # Calculate cost standard error
    df[f'{scenario_name}_{measure_type}_installation_cost_percentile_se'] = (
        df[cost_std_col] / sqrt_n_monte
    )
    
    # Calculate carbon savings
    carbon_saved_mean = kwh_change_mean * gas_carbon_factor_22
    carbon_saved_std = kwh_change_std * gas_carbon_factor_22
    carbon_se = carbon_saved_std / sqrt_n_monte
    
    df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean'] = carbon_saved_mean
    df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_std'] = carbon_saved_std
    df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] = carbon_se
    df[f'{scenario_name}_{measure_type}_5yr_carbon_saved_r_se'] = (
        carbon_se / carbon_saved_mean
    )
    
    # Calculate cost per kg saved
    cost_mean = df[cost_mean_col]
    cost_per_kg = cost_mean / carbon_saved_mean
    
    df[f'{scenario_name}_{measure_type}_cost_per_kg_saved'] = cost_per_kg
    
    # Vectorized calculation of combined standard error
    cost_se = df[f'{scenario_name}_{measure_type}_installation_cost_percentile_se']
    combined_variance = (cost_se / cost_mean)**2 + (carbon_se / carbon_saved_mean)**2
    
    df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_se'] = (
        np.abs(cost_per_kg) * np.sqrt(combined_variance)
    )
    df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_std'] = (
        df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_se'] * sqrt_n_monte
    )
    
    # Calculate cost savings
    cost_savings_mean = kwh_change_mean * gas_price
    cost_savings_std = kwh_change_std * gas_price
    
    df[f'{scenario_name}_{measure_type}_5yr_cost_savings_mean'] = cost_savings_mean
    df[f'{scenario_name}_{measure_type}_5yr_cost_savings_std'] = cost_savings_std
    
    # Calculate payback
    df[f'{scenario_name}_{measure_type}_5y_payback'] = cost_mean + cost_savings_mean
    df[f'{scenario_name}_{measure_type}_5yr_savings_std'] = np.sqrt(
        df[cost_std_col]**2 + cost_savings_std**2
    )


def calculate_absolute_reductions(df):
    """Calculate absolute energy reductions for different wall types."""
    # Calculate all reductions in vectorized operations
    total_gas = df['total_gas_derived']
    
    # Cavity wall reductions
    df['absolute_reduction_cavity'] = (
        total_gas * df['wall_installation_energy_cavity_wall_percentile_gas_mean']
    )
    df['absolute_reduction_std_cavity'] = (
        total_gas * df['wall_installation_energy_cavity_wall_percentile_gas_std']
    )
    
    # Solid wall internal reductions
    df['absolute_reduction_solid_internal'] = (
        total_gas * df['wall_installation_energy_solid_wall_internal_percentile_gas_mean']
    )
    df['absolute_reduction_std_solid_internal'] = (
        total_gas * df['wall_installation_energy_solid_wall_internal_percentile_gas_std']
    )
    
    # Solid wall external reductions
    df['absolute_reduction_solid_external'] = (
        total_gas * df['wall_installation_energy_solid_wall_external_percentile_gas_mean']
    )
    df['absolute_reduction_std_solid_external'] = (
        total_gas * df['wall_installation_energy_solid_wall_external_percentile_gas_std']
    )


def main( ):
    """Main execution function."""
    # Load and validate data
    name='test1'
    file_pattern = f"{DATA_DIR}/*.csv"
    df = load_and_concatenate_data(file_pattern)
    
    print('DataFame loaded ', df.shape)
    OUTPUT_DIR=f'{OUTPUT_DIR_base}/{name}'
    validate(df, OUTPUT_DIR)

    # Calculate absolute reductions
    calculate_absolute_reductions(df)
    
    # Plot building counts by age band
    fig = plot_building_counts_by_age_band(
        df,
        groupby_cols=['avg_gas_percentile', 'premise_age_bucketed'],
        cavity_col='absolute_reduction_cavity',
        solid_internal_col='absolute_reduction_solid_internal',
        solid_external_col='absolute_reduction_solid_external',
        decile_label='Gas Usage Decile',
        age_label='premise_age_bucketed',
        title='Building Counts by Wall Insulation Type and Age Band',
        age_band_order=None,
        figsize=(18, 10),
        show_plot=False,
        return_data=False
    )
    fig.savefig(f'{OUTPUT_DIR}/age_band.png', dpi=300, bbox_inches='tight')
    
    print('Plotting single intervention figs ')
    # plosingle measures
    scenario_name='loft'
    measure_type='loft'

    calculate_and_plot_measure_analysis(df, scenario_name, measure_type, N_MONTE_CARLO_RUNS, GAS_CARBON_FACTOR_2022, 
                                        GAS_PRICE, save_figs=True, output_dir=OUTPUT_DIR)
    
    scenario_name='wall'
    for measure_type in ['cavity_wall', 'solid_wall_internal', 'solid_wall_external']:
        calculate_and_plot_measure_analysis(df, scenario_name, measure_type, N_MONTE_CARLO_RUNS, GAS_CARBON_FACTOR_2022, 
                                        GAS_PRICE, save_figs=True, output_dir=OUTPUT_DIR)

    print('starting grouped plots')
    # Process all measures in-place (MUCH faster - no dataframe copying/concatenation)
    proc_measures_optimized(df, 'loft', 'loft', GAS_CARBON_FACTOR_2022, N_MONTE_CARLO_RUNS, GAS_PRICE)
    print('Df shape after loft', df.shape)
    for measure in ['cavity_wall', 'solid_wall_internal', 'solid_wall_external']:
        proc_measures_optimized(df, measure, 'wall', GAS_CARBON_FACTOR_2022, N_MONTE_CARLO_RUNS, GAS_PRICE)
    
    # df now contains all the calculated columns
    print('Df shape after wall', df.shape)
    # Save or return as needed
    df.to_csv(f'{OUTPUT_DIR}/processed_interventions_tables.csv')
    return df


if __name__ == '__main__':
    result_df = main()
    print(result_df.head)

    