import pandas as pd 
from src.visualisations import plot_col_reduction_by_decile
import numpy as np 


def calculate_and_plot_measure_analysis(df, scenario_name, measure_type, n_monte, gas_carbon_factor_22, 
                                        GAS_PRICE, save_figs=True, output_dir='./figures'):
    """
    Calculate energy savings, carbon reduction, and cost metrics for a home energy measure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with energy usage and measure cost data
    measure_type : str
        Type of measure (e.g., 'loft', 'cavity_wall', 'solar_panels')
    n_monte : int
        Number of Monte Carlo simulations
    gas_carbon_factor_22 : float
        Gas carbon factor for 2022
    GAS_PRICE : float
        Price per kWh of gas
    save_figs : bool
        Whether to save the generated figures
    output_dir : str
        Directory to save figures
        
    Returns:
    --------
    df : pd.DataFrame
        Updated dataframe with calculated columns
    figs : dict
        Dictionary of generated figures
    """
    import os
    
    # Create output directory if saving
    if save_figs:
        os.makedirs(output_dir, exist_ok=True)
    
    df = pre_proc(df,measure_type, scenario_name , gas_carbon_factor_22, n_monte, GAS_PRICE)
    run_figs_seperate(df, scenario_name, measure_type, save_figs , output_dir)
    return df 


def pre_proc(df,measure_type, scenario_name , gas_carbon_factor_22, n_monte, GAS_PRICE):
    # Calculate 5-year kWh changes
    df[f'5y_kwh_change_{measure_type}'] = (
        (df['total_gas_derived'] * 5) * 
        df[f'{scenario_name}_installation_energy_{measure_type}_percentile_gas_mean']
    )
    df[f'5y_kwh_change_{measure_type}_std'] = (
        (df['total_gas_derived'] * 5) * 
        df[f'{scenario_name}_installation_energy_{measure_type}_percentile_gas_std']
    )

    # Calculate cost standard error
    df[f'{scenario_name}_installation_cost_{measure_type}_percentile_se'] = (
        df[f'{scenario_name}_installation_cost_{measure_type}_percentile_std'] / np.sqrt(n_monte)
    )

    # Calculate carbon savings
    df['5yr_kg_co2_saved_mean'] = (
        df[f'5y_kwh_change_{measure_type}'] * gas_carbon_factor_22
    )
    df['5yr_kg_co2_saved_std'] = (
        df[f'5y_kwh_change_{measure_type}_std'] * gas_carbon_factor_22
    )
    df['5yr_carbon_se'] = df['5yr_kg_co2_saved_std'] / np.sqrt(n_monte)
    df['5yr_carbon_saved_r_se'] = df['5yr_carbon_se'] / df['5yr_kg_co2_saved_mean']

    # Calculate cost per kg saved
    df['cost_per_kg_saved'] = (
        df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'] / 
        df['5yr_kg_co2_saved_mean']
    )

    df['cost_per_kg_saved_se'] = (
        np.abs(df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'] / 
               df['5yr_kg_co2_saved_mean']) * 
        np.sqrt(
            (df[f'{scenario_name}_installation_cost_{measure_type}_percentile_se'] / 
             df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'])**2 + 
            (df['5yr_carbon_se'] / df['5yr_kg_co2_saved_mean'])**2
        )
    )
    df['cost_per_kg_saved_std'] = df['cost_per_kg_saved_se'] * np.sqrt(n_monte)

    # Calculate cost savings
    df['5yr_cost_savings_mean'] = df[f'5y_kwh_change_{measure_type}'] * GAS_PRICE
    df['5yr_cost_savings_std'] = df[f'5y_kwh_change_{measure_type}_std'] * GAS_PRICE

    # Calculate payback
    df[f'{measure_type}_5y_payback'] = (
        df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'] + 
        df['5yr_cost_savings_mean']
    )
    df[f'{measure_type}_5yr_savings_std'] = np.sqrt(
        (df[f'{scenario_name}_installation_cost_{measure_type}_percentile_std']**2) + 
        (df['5yr_cost_savings_std']**2)
    )
    return df 


def proc_measures(df, measure_type, scenario_name, gas_carbon_factor_22, n_monte, GAS_PRICE):
    # Calculate 5-year kWh changes
    df[f'{scenario_name}_{measure_type}_5y_kwh_change'] = (
        (df['total_gas_derived'] * 5) * 
        df[f'{scenario_name}_installation_energy_{measure_type}_percentile_gas_mean']
    )
    df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] = (
        (df['total_gas_derived'] * 5) * 
        df[f'{scenario_name}_installation_energy_{measure_type}_percentile_gas_std']
    )
    
    # Calculate cost standard error
    df[f'{scenario_name}_{measure_type}_installation_cost_percentile_se'] = (
        df[f'{scenario_name}_installation_cost_{measure_type}_percentile_std'] / np.sqrt(n_monte)
    )
    
    # Calculate carbon savings
    df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean'] = (
        df[f'{scenario_name}_{measure_type}_5y_kwh_change'] * gas_carbon_factor_22
    )
    df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_std'] = (
        df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] * gas_carbon_factor_22
    )
    df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] = (
        df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_std'] / np.sqrt(n_monte)
    )
    df[f'{scenario_name}_{measure_type}_5yr_carbon_saved_r_se'] = (
        df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] / 
        df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean']
    )
    
    # Calculate cost per kg saved
    df[f'{scenario_name}_{measure_type}_cost_per_kg_saved'] = (
        df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'] / 
        df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean']
    )
    
    df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_se'] = (
        np.abs(df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'] / 
               df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean']) * 
        np.sqrt(
            (df[f'{scenario_name}_{measure_type}_installation_cost_percentile_se'] / 
             df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'])**2 + 
            (df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] / 
             df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean'])**2
        )
    )
    df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_std'] = (
        df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_se'] * np.sqrt(n_monte)
    )
    
    # Calculate cost savings
    df[f'{scenario_name}_{measure_type}_5yr_cost_savings_mean'] = (
        df[f'{scenario_name}_{measure_type}_5y_kwh_change'] * GAS_PRICE
    )
    df[f'{scenario_name}_{measure_type}_5yr_cost_savings_std'] = (
        df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] * GAS_PRICE
    )
    
    # Calculate payback
    df[f'{scenario_name}_{measure_type}_5y_payback'] = (
        df[f'{scenario_name}_installation_cost_{measure_type}_percentile_mean'] + 
        df[f'{scenario_name}_{measure_type}_5yr_cost_savings_mean']
    )
    df[f'{scenario_name}_{measure_type}_5yr_savings_std'] = np.sqrt(
        (df[f'{scenario_name}_installation_cost_{measure_type}_percentile_std']**2) + 
        (df[f'{scenario_name}_{measure_type}_5yr_cost_savings_std']**2)
    )
    
    return df


def run_figs_seperate(df, scenario_name, measure_type, save_figs , output_dir):
    
    # Dictionary to store figures
    figs = {}
    
    # Plot 1: Installation costs
    fig1 = plot_col_reduction_by_decile(
        df, 
        groupby_col='avg_gas_percentile',
        mean_col=f'{scenario_name}_installation_cost_{measure_type}_percentile_mean',
        std_col=f'{scenario_name}_installation_cost_{measure_type}_percentile_std',
        percentage=False,
        groupby_label='Gas Usage Decile',
        ylabel=f'{measure_type.replace("_", " ").title()} Cost (£)'
    )
    figs['installation_cost'] = fig1
    if save_figs:
        fig1.savefig(f'{output_dir}/{scenario_name}_{measure_type}_installation_cost.png', 
                     dpi=300, bbox_inches='tight')
    
    # Plot 2: Carbon savings
    fig2 = plot_col_reduction_by_decile(
        df, 
        groupby_col='avg_gas_percentile',
        mean_col='5yr_kg_co2_saved_mean',
        std_col='5yr_kg_co2_saved_std',
        percentage=False,
        groupby_label='Gas Usage Decile',
        ylabel='kWh gas saved (kWh)',
        costs=False
    )
    figs['carbon_savings'] = fig2
    if save_figs:
        fig2.savefig(f'{output_dir}/{scenario_name}_{measure_type}_carbon_savings.png', 
                     dpi=300, bbox_inches='tight')
    
    # Plot 3: Cost per kg saved (filtered)
    pl = df[df['avg_gas_percentile'] > 3].copy()
    fig3 = plot_col_reduction_by_decile(
        pl, 
        groupby_col='avg_gas_percentile',
        mean_col='cost_per_kg_saved',
        std_col='cost_per_kg_saved_std',
        percentage=False,
        groupby_label='Gas Usage Decile',
        ylabel='Costs per kWh gas saved (kWh)',
        costs=False
    )
    figs['cost_per_kg_saved'] = fig3
    if save_figs:
        fig3.savefig(f'{output_dir}/{scenario_name}_{measure_type}_cost_per_kg_saved.png', 
                     dpi=300, bbox_inches='tight')
    
    return df, figs


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
