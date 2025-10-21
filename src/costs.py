import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pandas as pd 
from src.visualisations import plot_col_reduction_by_decile
import numpy as np 
from typing import Optional 


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional 


def calculate_and_plot_measure_analysis(
    df, 
    scenario_name, 
    measure_type, 
    n_monte, 
    gas_carbon_factor_22, 
    GAS_PRICE, 
    save_figs=True, 
    output_dir='./figures', 
    total_cost_mean_col=None, 
    total_cost_std_col=None,
    years=5
):
    """
    Calculate energy savings, carbon reduction, and cost metrics for a home energy measure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with energy usage and measure cost data
    scenario_name : str
        Name of the scenario being analyzed
    measure_type : str
        Type of measure (e.g., 'loft', 'cavity_wall', 'solar_panels')
    n_monte : int
        Number of Monte Carlo simulations
    gas_carbon_factor_22 : float
        Gas carbon factor for 2022 (kg CO2 per kWh)
    GAS_PRICE : float
        Price per kWh of gas
    save_figs : bool
        Whether to save the generated figures
    output_dir : str
        Directory to save figures
    total_cost_mean_col : str, optional
        Custom column name for total cost mean
    total_cost_std_col : str, optional
        Custom column name for total cost std
    years : int
        Time horizon for calculations (default: 5)
        
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
    
    # Calculate all metrics
    calculate_energy_savings_metrics(
        df, 
        measure_type, 
        scenario_name, 
        gas_carbon_factor_22, 
        n_monte, 
        GAS_PRICE,
        years=years,
        total_cost_mean_col=total_cost_mean_col,
        total_cost_std_col=total_cost_std_col
    )
    
    if df.empty:
        raise Exception('DF empty')
    
    # Generate plots
    figs = run_figs_seperate(
        df, 
        scenario_name, 
        measure_type, 
        save_figs, 
        output_dir,
        years=years
    )
    
    return df, figs


def calculate_energy_savings_metrics(
    df: pd.DataFrame,
    measure_type: str,
    scenario_name: str,
    gas_carbon_factor: float,
    elec_carbon_factor: float, 
    n_simulations: int,
    gas_price: float,
    years: int = 5,
    total_cost_mean_col: Optional[str] = None,
    total_cost_std_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate comprehensive energy savings, carbon reduction, and cost metrics.
    
    Merges functionality from both proc_measures and calculate_energy_savings_metrics,
    with flexible cost column naming for combo measures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with energy and cost data
    measure_type : str
        Type of energy efficiency measure (e.g., 'insulation', 'heating')
    scenario_name : str
        Name of the scenario being analyzed
    gas_carbon_factor : float
        Carbon emission factor for gas (kg CO2 per kWh)
    n_simulations : int
        Number of Monte Carlo simulations used
    gas_price : float
        Price per kWh of gas
    years : int, optional
        Time horizon for calculations (default: 5)
    total_cost_mean_col : str, optional
        Custom column name for total cost mean (e.g., for combo measures)
        If None, uses default: '{scenario_name}_cost_{measure_type}_mean'
    total_cost_std_col : str, optional
        Custom column name for total cost std (e.g., for combo measures)
        If None, uses default: '{scenario_name}_cost_{measure_type}_std'
    
    Returns:
    --------
    pd.DataFrame
        Input dataframe with additional calculated columns (modifies in place)
    """
    
    # ==================================================================
    # STEP 1: Determine cost column names
    # ==================================================================
    cost_mean_col = (
        total_cost_mean_col if total_cost_mean_col 
        else f'{scenario_name}_cost_{measure_type}_mean'
    )
    cost_std_col = (
        total_cost_std_col if total_cost_std_col 
        else f'{scenario_name}_cost_{measure_type}_std'
    )
    
    # ==================================================================
    # STEP 2: Calculate multi-year kWh changes
    # ==================================================================
    df[f'{years}yr_kwh_change_{measure_type}'] = (
        df['total_gas_derived'] * years * 
        df[f'{scenario_name}_energy_{measure_type}_gas_mean']
    )
    df[f'{years}yr_kwh_change_{measure_type}_std'] = (
        df['total_gas_derived'] * years * 
        df[f'{scenario_name}_energy_{measure_type}_gas_std']
    )
    if scenario_name in ['heat_pump_only' , 'join_heat_ins_decay' ]:
        df[f'{years}yr_kwh_change_{measure_type}_electricity'] = (
        df['total_elec_derived'] * years * 
        df[f'{scenario_name}_energy_{measure_type}_electricity_mean']
      )
        df[f'{years}yr_kwh_change_{measure_type}_electricity_std'] = (
            df['total_elec_derived'] * years * 
            df[f'{scenario_name}_energy_{measure_type}_electricity_std']
        )
    
    # ==================================================================
    # STEP 3: Calculate cost standard error
    # ==================================================================
    df[f'{scenario_name}_cost_{measure_type}_se'] = (
        df[cost_std_col] / np.sqrt(n_simulations)
    )
    
    # ==================================================================
    # STEP 4: Calculate carbon savings metrics
    # ==================================================================
    df[f'{years}yr_kg_co2_saved_mean'] = (
        df[f'{years}yr_kwh_change_{measure_type}'] * gas_carbon_factor
    )
    df[f'{years}yr_kg_co2_saved_std'] = (
        df[f'{years}yr_kwh_change_{measure_type}_std'] * gas_carbon_factor
    )
    df[f'{years}yr_carbon_se'] = (
        df[f'{years}yr_kg_co2_saved_std'] / np.sqrt(n_simulations)
    )
    df[f'{years}yr_carbon_saved_r_se'] = (
        df[f'{years}yr_carbon_se'] / df[f'{years}yr_kg_co2_saved_mean']
    )

    if scenario_name in ['heat_pump_only' , 'join_heat_ins_decay' ]:
            df[f'{years}yr_kg_co2_saved_electricity_mean'] = (
            df[f'{years}yr_kwh_change_{measure_type}_electricity'] * elec_carbon_factor
        )
            df[f'{years}yr_kg_co2_saved_electricity_std'] = (
                df[f'{years}yr_kwh_change_{measure_type}_electricity_std'] * elec_carbon_factor
            )
    
    # ==================================================================
    # STEP 5: Calculate cost savings (from gas bill reductions)
    # ==================================================================
    df[f'{years}yr_cost_savings_mean'] = (
        df[f'{years}yr_kwh_change_{measure_type}'] * gas_price
    )
    df[f'{years}yr_cost_savings_std'] = (
        df[f'{years}yr_kwh_change_{measure_type}_std'] * gas_price
    )
    
    # ==================================================================
    # STEP 6: Calculate cost per kg CO2 saved (excluding bill savings)
    # ==================================================================
    df[f'{years}yr_cost_per_kg_saved'] = (
        df[cost_mean_col] / df[f'{years}yr_kg_co2_saved_mean']
    )
    
    # Standard error using error propagation
    df[f'{years}yr_cost_per_kg_saved_se'] = (
        np.abs(df[cost_mean_col] / df[f'{years}yr_kg_co2_saved_mean']) * 
        np.sqrt(
            (df[f'{scenario_name}_cost_{measure_type}_se'] / df[cost_mean_col])**2 + 
            (df[f'{years}yr_carbon_se'] / df[f'{years}yr_kg_co2_saved_mean'])**2
        )
    )
    df[f'{years}yr_cost_per_kg_saved_std'] = (
        df[f'{years}yr_cost_per_kg_saved_se'] * np.sqrt(n_simulations)
    )
    

    # now do it with the 
    df[f'net_cost'] = df[cost_mean_col] + df['5yr_cost_savings_mean']


    # ==================================================================
    # STEP 7: Calculate payback (net cost after savings)
    # ==================================================================
    df[f'{years}yr_payback'] = (
        df[cost_mean_col] + df[f'{years}yr_cost_savings_mean']
    )
    
    # Combined uncertainty (installation cost + savings uncertainty)
    df[f'{years}yr_savings_std'] = np.sqrt(
        df[cost_std_col]**2 + df[f'{years}yr_cost_savings_std']**2
    )
    
    return df


def run_figs_seperate(df, scenario_name, measure_type, save_figs, output_dir, years=5):
    """
    Generate plots for measure analysis with dynamic column naming.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with calculated metrics
    scenario_name : str
        Name of the scenario
    measure_type : str
        Type of measure
    save_figs : bool
        Whether to save figures
    output_dir : str
        Directory to save figures
    years : int
        Time horizon used in calculations (default: 5)
        
    Returns:
    --------
    figs : dict
        Dictionary of generated figures
    """
    from src.visualisations import plot_col_reduction_by_decile
    
    # Dictionary to store figures
    figs = {}
    
    # ==================================================================
    # Plot 1: Installation costs
    # ==================================================================
    fig1 = plot_col_reduction_by_decile(
        df, 
        groupby_col='avg_gas_percentile',
        mean_col=f'{scenario_name}_cost_{measure_type}_mean',
        std_col=f'{scenario_name}_cost_{measure_type}_std',
        percentage=False,
        groupby_label='Gas Usage Decile',
        ylabel=f'{measure_type.replace("_", " ").title()} Installation Cost (£)'
    )
    figs['installation_cost'] = fig1
    if save_figs:
        fig1.savefig(
            f'{output_dir}/{scenario_name}_{measure_type}_installation_costs.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig1)
    
    # ==================================================================
    # Plot 2: Carbon savings (using dynamic column names)
    # ==================================================================
    fig2 = plot_col_reduction_by_decile(
        df, 
        groupby_col='avg_gas_percentile',
        mean_col=f'{years}yr_kg_co2_saved_mean',  # Dynamic
        std_col=f'{years}yr_kg_co2_saved_std',    # Dynamic
        percentage=False,
        groupby_label='Gas Usage Decile',
        ylabel=f'{years}-Year CO2 Saved (kg)',
        costs=False
    )
    figs['carbon_savings'] = fig2
    if save_figs:
        fig2.savefig(
            f'{output_dir}/{scenario_name}_{measure_type}_kgcarbon_savings.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig2)
    
    # ==================================================================
    # Plot 3: Cost per kg saved (filtered for higher usage homes)
    # ==================================================================
    pl = df[df['avg_gas_percentile'] > 3].copy()
    
    if not pl.empty:
        fig3 = plot_col_reduction_by_decile(
            pl, 
            groupby_col='avg_gas_percentile',
            mean_col=f'{years}yr_cost_per_kg_saved',  # Dynamic
            std_col=f'{years}yr_cost_per_kg_saved_std',  # Dynamic
            percentage=False,
            groupby_label='Gas Usage Decile',
            ylabel=f'Cost per kg CO2 Saved (£/kg)',
            costs=False
        )
        figs['cost_per_kg_saved'] = fig3
        if save_figs:
            fig3.savefig(
                f'{output_dir}/{scenario_name}_{measure_type}_cost_per_kg_saved.png', 
                dpi=300, bbox_inches='tight'
            )
            plt.close(fig3)
    
    # ==================================================================
    # Plot 4: Payback analysis
    # ==================================================================
    # fig4 = plot_col_reduction_by_decile(
    #     df, 
    #     groupby_col='avg_gas_percentile',
    #     mean_col=f'{years}yr_payback',
    #     std_col=f'{years}yr_savings_std',
    #     percentage=False,
    #     groupby_label='Gas Usage Decile',
    #     ylabel=f'{years}-Year Net Cost (£)',
    #     costs=True
    # )
    # figs['payback'] = fig4
    # if save_figs:
    #     fig4.savefig(
    #         f'{output_dir}/{scenario_name}_{measure_type}_payback.png', 
    #         dpi=300, bbox_inches='tight'
    #     )
    #     plt.close(fig4)
    
    return figs



# def calculate_and_plot_measure_analysis(df, scenario_name, measure_type, n_monte, gas_carbon_factor_22, 
#                                         GAS_PRICE, save_figs=True, output_dir='./figures', total_cost_mean_col=None, total_cost_std_col=None  ):
#     """
#     Calculate energy savings, carbon reduction, and cost metrics for a home energy measure.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input dataframe with energy usage and measure cost data
#     measure_type : str
#         Type of measure (e.g., 'loft', 'cavity_wall', 'solar_panels')
#     n_monte : int
#         Number of Monte Carlo simulations
#     gas_carbon_factor_22 : float
#         Gas carbon factor for 2022
#     GAS_PRICE : float
#         Price per kWh of gas
#     save_figs : bool
#         Whether to save the generated figures
#     output_dir : str
#         Directory to save figures
        
#     Returns:
#     --------
#     df : pd.DataFrame
#         Updated dataframe with calculated columns
#     figs : dict
#         Dictionary of generated figures
#     """
#     import os
    
#     # Create output directory if saving
#     if save_figs:
#         os.makedirs(output_dir, exist_ok=True)
    
#     calculate_energy_savings_metrics(df,measure_type, scenario_name , gas_carbon_factor_22, n_monte, GAS_PRICE, total_cost_mean_col, total_cost_std_col)
#     if df.empty:
#         raise Exception('DF empty ')
#     run_figs_seperate(df, scenario_name, measure_type, save_figs , output_dir)
#     return df 



# def calculate_energy_savings_metrics(
#     df: pd.DataFrame,
#     measure_type: str,
#     scenario_name: str,
#     gas_carbon_factor: float,
#     n_simulations: int,
#     # gas_price: float,
#     years: int = 5,
#     total_cost_mean_col: Optional[str] = None,
#     total_cost_std_col: Optional[str] = None
# ) -> pd.DataFrame:
#     """
#     Calculate energy savings, carbon reduction, and cost metrics for energy efficiency measures.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input dataframe with energy and cost data
#     measure_type : str
#         Type of energy efficiency measure (e.g., 'insulation', 'heating')
#     scenario_name : str
#         Name of the scenario being analyzed
#     gas_carbon_factor : float
#         Carbon emission factor for gas (kg CO2 per kWh)
#     n_simulations : int
#         Number of Monte Carlo simulations used
#     gas_price : float
#         Price per kWh of gas
#     years : int, optional
#         Time horizon for calculations (default: 5)
#     total_cost_mean_col : str, optional
#         Custom column name for total cost mean (e.g., for combo measures)
#         If None, uses default pattern: '{scenario_name}_cost_{measure_type}_mean'
#     total_cost_std_col : str, optional
#         Custom column name for total cost std (e.g., for combo measures)
#         If None, uses default pattern: '{scenario_name}_cost_{measure_type}_std'
    
#     Returns:
#     --------
#     pd.DataFrame
#         Input dataframe with additional calculated columns
#     """
    
#     # Determine cost column names
#     cost_mean_col = (
#         total_cost_mean_col if total_cost_mean_col 
#         else f'{scenario_name}_cost_{measure_type}_mean'
#     )
#     cost_std_col = (
#         total_cost_std_col if total_cost_std_col 
#         else f'{scenario_name}_cost_{measure_type}_std'
#     )
    
#     # Calculate multi-year kWh changes
#     df[f'{years}yr_kwh_change_{measure_type}'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_energy_{measure_type}_gas_mean']
#     )
#     df[f'{years}yr_kwh_change_{measure_type}_std'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_energy_{measure_type}_gas_std']
#     )
    
#     # Calculate cost standard error
#     df[f'{scenario_name}_cost_{measure_type}_se'] = (
#         df[cost_std_col] / np.sqrt(n_simulations)
#     )
    
#     # Calculate carbon savings metrics
#     df[f'{years}yr_kg_co2_saved_mean'] = (
#         df[f'{years}yr_kwh_change_{measure_type}'] * gas_carbon_factor
#     )
#     df[f'{years}yr_kg_co2_saved_std'] = (
#         df[f'{years}yr_kwh_change_{measure_type}_std'] * gas_carbon_factor
#     )
#     df[f'{years}yr_carbon_se'] = (
#         df[f'{years}yr_kg_co2_saved_std'] / np.sqrt(n_simulations)
#     )


# def proc_measures(df, measure_type, scenario_name, gas_carbon_factor_22, n_monte, GAS_PRICE):
#     # Calculate 5-year kWh changes
#     df[f'{scenario_name}_{measure_type}_5y_kwh_change'] = (
#         (df['total_gas_derived'] * 5) * 
#         df[f'{scenario_name}_energy_{measure_type}_gas_mean']
#     )
#     df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] = (
#         (df['total_gas_derived'] * 5) * 
#         df[f'{scenario_name}_energy_{measure_type}_gas_std']
#     )
    
#     # Calculate cost standard error
#     df[f'{scenario_name}_{measure_type}_cost_se'] = (
#         df[f'{scenario_name}_cost_{measure_type}_std'] / np.sqrt(n_monte)
#     )
    
#     # Calculate carbon savings
#     df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean'] = (
#         df[f'{scenario_name}_{measure_type}_5y_kwh_change'] * gas_carbon_factor_22
#     )
#     df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_std'] = (
#         df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] * gas_carbon_factor_22
#     )
#     df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] = (
#         df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_std'] / np.sqrt(n_monte)
#     )
#     df[f'{scenario_name}_{measure_type}_5yr_carbon_saved_r_se'] = (
#         df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] / 
#         df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean']
#     )
    
#     # Calculate cost per kg saved
#     df[f'{scenario_name}_{measure_type}_cost_per_kg_saved'] = (
#         df[f'{scenario_name}_cost_{measure_type}_mean'] / 
#         df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean']
#     )
    
#     df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_se'] = (
#         np.abs(df[f'{scenario_name}_cost_{measure_type}_mean'] / 
#                df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean']) * 
#         np.sqrt(
#             (df[f'{scenario_name}_{measure_type}_cost_se'] / 
#              df[f'{scenario_name}_cost_{measure_type}_mean'])**2 + 
#             (df[f'{scenario_name}_{measure_type}_5yr_carbon_se'] / 
#              df[f'{scenario_name}_{measure_type}_5yr_kg_co2_saved_mean'])**2
#         )
#     )
#     df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_std'] = (
#         df[f'{scenario_name}_{measure_type}_cost_per_kg_saved_se'] * np.sqrt(n_monte)
#     )
    
#     # Calculate cost savings
#     df[f'{scenario_name}_{measure_type}_5yr_cost_savings_mean'] = (
#         df[f'{scenario_name}_{measure_type}_5y_kwh_change'] * GAS_PRICE
#     )
#     df[f'{scenario_name}_{measure_type}_5yr_cost_savings_std'] = (
#         df[f'{scenario_name}_{measure_type}_5y_kwh_change_std'] * GAS_PRICE
#     )
    
#     # Calculate payback
#     df[f'{scenario_name}_{measure_type}_5y_payback'] = (
#         df[f'{scenario_name}_cost_{measure_type}_mean'] + 
#         df[f'{scenario_name}_{measure_type}_5yr_cost_savings_mean']
#     )
#     df[f'{scenario_name}_{measure_type}_5yr_savings_std'] = np.sqrt(
#         (df[f'{scenario_name}_cost_{measure_type}_std']**2) + 
#         (df[f'{scenario_name}_{measure_type}_5yr_cost_savings_std']**2)
#     )
    
#     return df


# def run_figs_seperate(df, scenario_name, measure_type, save_figs , output_dir):
    
#     # Dictionary to store figures
#     figs = {}
    
#     # Plot 1: Installation costs
#     fig1 = plot_col_reduction_by_decile(
#         df, 
#         groupby_col='avg_gas_percentile',
#         mean_col=f'{scenario_name}_cost_{measure_type}_mean',
#         std_col=f'{scenario_name}_cost_{measure_type}_std',
#         percentage=False,
#         groupby_label='Gas Usage Decile',
#         ylabel=f'{measure_type.replace("_", " ").title()} Cost (£)'
#     )
#     figs['installation_cost'] = fig1
#     if save_figs:
#         fig1.savefig(f'{output_dir}/{scenario_name}_{measure_type}_installation_cost.png', 
#                      dpi=300, bbox_inches='tight')
    
#     # Plot 2: Carbon savings
#     fig2 = plot_col_reduction_by_decile(
#         df, 
#         groupby_col='avg_gas_percentile',
#         mean_col='5yr_kg_co2_saved_mean',
#         std_col='5yr_kg_co2_saved_std',
#         percentage=False,
#         groupby_label='Gas Usage Decile',
#         ylabel='kWh gas saved (kWh)',
#         costs=False
#     )
#     figs['carbon_savings'] = fig2
#     if save_figs:
#         fig2.savefig(f'{output_dir}/{scenario_name}_{measure_type}_carbon_savings.png', 
#                      dpi=300, bbox_inches='tight')
    
#     # Plot 3: Cost per kg saved (filtered)
#     pl = df[df['avg_gas_percentile'] > 3].copy()
#     fig3 = plot_col_reduction_by_decile(
#         pl, 
#         groupby_col='avg_gas_percentile',
#         mean_col='cost_per_kg_saved',
#         std_col='cost_per_kg_saved_std',
#         percentage=False,
#         groupby_label='Gas Usage Decile',
#         ylabel='Costs per kWh gas saved (kWh)',
#         costs=False
#     )
#     figs['cost_per_kg_saved'] = fig3
#     if save_figs:
#         fig3.savefig(f'{output_dir}/{scenario_name}_{measure_type}_cost_per_kg_saved.png', 
#                      dpi=300, bbox_inches='tight')
    
#     return df, figs





# # def pre_proc(df,measure_type, scenario_name , gas_carbon_factor_22, n_monte, GAS_PRICE):
# #     # Calculate 5-year kWh changes
# #     df[f'5y_kwh_change_{measure_type}'] = (
# #         (df['total_gas_derived'] * 5) * 
# #         df[f'{scenario_name}_energy_{measure_type}_gas_mean']
# #     )
# #     df[f'5y_kwh_change_{measure_type}_std'] = (
# #         (df['total_gas_derived'] * 5) * 
# #         df[f'{scenario_name}_energy_{measure_type}_gas_std']
# #     )

# #     # Calculate cost standard error
# #     df[f'{scenario_name}_cost_{measure_type}_se'] = (
# #         df[f'{scenario_name}_cost_{measure_type}_std'] / np.sqrt(n_monte)
# #     )

# #     # Calculate carbon savings
# #     df['5yr_kg_co2_saved_mean'] = (
# #         df[f'5y_kwh_change_{measure_type}'] * gas_carbon_factor_22
# #     )
# #     df['5yr_kg_co2_saved_std'] = (
# #         df[f'5y_kwh_change_{measure_type}_std'] * gas_carbon_factor_22
# #     )
# #     df['5yr_carbon_se'] = df['5yr_kg_co2_saved_std'] / np.sqrt(n_monte)
# #     df['5yr_carbon_saved_r_se'] = df['5yr_carbon_se'] / df['5yr_kg_co2_saved_mean']

# #     # Calculate cost per kg saved
# #     df['cost_per_kg_saved'] = (
# #         df[f'{scenario_name}_cost_{measure_type}_mean'] / 
# #         df['5yr_kg_co2_saved_mean']
# #     )

# #     df['cost_per_kg_saved_se'] = (
# #         np.abs(df[f'{scenario_name}_cost_{measure_type}_mean'] / 
# #                df['5yr_kg_co2_saved_mean']) * 
# #         np.sqrt(
# #             (df[f'{scenario_name}_cost_{measure_type}_se'] / 
# #              df[f'{scenario_name}_cost_{measure_type}_mean'])**2 + 
# #             (df['5yr_carbon_se'] / df['5yr_kg_co2_saved_mean'])**2
# #         )
# #     )
# #     df['cost_per_kg_saved_std'] = df['cost_per_kg_saved_se'] * np.sqrt(n_monte)

# #     # Calculate cost savings
# #     df['5yr_cost_savings_mean'] = df[f'5y_kwh_change_{measure_type}'] * GAS_PRICE
# #     df['5yr_cost_savings_std'] = df[f'5y_kwh_change_{measure_type}_std'] * GAS_PRICE

# #     # Calculate payback
# #     df[f'{measure_type}_5y_payback'] = (
# #         df[f'{scenario_name}_cost_{measure_type}_mean'] + 
# #         df['5yr_cost_savings_mean']
# #     )
# #     df[f'{measure_type}_5yr_savings_std'] = np.sqrt(
# #         (df[f'{scenario_name}_cost_{measure_type}_std']**2) + 
# #         (df['5yr_cost_savings_std']**2)
# #     )
# #     return df 