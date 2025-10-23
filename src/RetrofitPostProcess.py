

import numpy as np 


def process_multiple_scenarios(df, scenarios_config, years, n_simulations, 
                                GAS_CARBON_FACTOR_2022, elec_carbon_factor):
    """
    Process energy and carbon savings data for multiple measure scenarios.
    
    Parameters:
    - df: DataFrame with energy consumption data for all scenarios
    - scenarios_config: List of tuples (measure_type, scenario_name) or dict {measure_type: scenario_name}
                       e.g., [('heat_pump', 'heat_pump_only'), 
                              ('insulation', 'join_heat_ins_decay')]
    - years: Number of years for projections
    - n_simulations: Number of Monte Carlo simulations
    - GAS_CARBON_FACTOR_2022: Carbon factor for gas (kg CO2/kWh)
    - elec_carbon_factor: Carbon factor for electricity (kg CO2/kWh)
    
    Returns:
    - df: DataFrame with all scenarios processed
    """
    
    # Convert dict to list of tuples if needed
    if isinstance(scenarios_config, dict):
        scenarios_config = list(scenarios_config.items())
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Process each scenario
    for measure_type, scenario_name in scenarios_config:
        print(f"Processing scenario: {scenario_name} (measure type: {measure_type})")
        
        df_processed = clean_post_proccess(
            df=df_processed,
            measure_type=measure_type,
            scenario_name=scenario_name,
            years=years,
            n_simulations=n_simulations,
            GAS_CARBON_FACTOR_2022=GAS_CARBON_FACTOR_2022,
            elec_carbon_factor=elec_carbon_factor
        )
    
    return df_processed



def clean_post_proccess(df, measure_type, scenario_name, years, n_simulations, 
                        GAS_CARBON_FACTOR_2022, elec_carbon_factor):
    """
    Process energy and carbon savings data for different measure scenarios.
    
    Parameters:
    - df: DataFrame with energy consumption data
    - measure_type: Type of energy efficiency measure
    - scenario_name: Name of the scenario (e.g., 'heat_pump_only', 'join_heat_ins_decay')
    - years: Number of years for projections
    - n_simulations: Number of Monte Carlo simulations
    - GAS_CARBON_FACTOR_2022: Carbon factor for gas (kg CO2/kWh)
    - elec_carbon_factor: Carbon factor for electricity (kg CO2/kWh)
    """
    
    stats = ['mean', 'p5', 'p50', 'p95', 'std']
    if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
        fuels = ['gas', 'elec']
    else:
        fuels = ['gas']
    # ==================================================================
    # Gas energy changes
    # ==================================================================
    for stat in stats:
        df[f'gas_{years}yr_kwh_change_{measure_type}_{stat}'] = (
            df['total_gas_derived'] * years * 
            df[f'{scenario_name}_{scenario_name}_gas_{stat}']
        )
    
    # ==================================================================
    # Electricity energy changes (for heat pump scenarios only)
    # ==================================================================
    if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
        for stat in stats:
            df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] = (
                df['total_elec_derived'] * years * 
                df[f'{scenario_name}_{scenario_name}_electricity_{stat}']
            )
    
    # ==================================================================
    # Gas carbon savings metrics
    # ==================================================================
    for stat in stats:
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_{stat}'] = (
            df[f'gas_{years}yr_kwh_change_{measure_type}_{stat}'] * GAS_CARBON_FACTOR_2022
        )
    
    # Standard error and relative standard error for gas
    df[f'gas_{years}yr_kg_co2_saved_{measure_type}_se'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std'] / np.sqrt(n_simulations)
    )
    df[f'gas_{years}yr_kg_co2_saved_{measure_type}_r_se'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_se'] / df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean']
    )

    # ==================================================================
    # Electricity carbon savings (heat pump scenarios only)
    # ==================================================================
    if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
        for stat in stats:
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_{stat}'] = (
                df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] * elec_carbon_factor
            )
        
        # Standard error for electricity
        df[f'elec_{years}yr_kg_co2_saved_{measure_type}_se'] = (
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_std'] / np.sqrt(n_simulations)
        )
        
        # Net carbon savings (gas + electricity) - NOW INDEXED
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean'] + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_mean']
        )
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] = np.sqrt(
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std']**2 + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_std']**2
        )
    else:
        # For non-heat pump scenarios, total equals gas only - NOW INDEXED
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] = df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean']
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] = df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std']

    # ==================================================================
    # Convert to tonnes - NOW INDEXED
    # ==================================================================

    for fuel in fuels:
        df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_mean'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_mean'] / 1000
        df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_std'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_std'] / 1000
        df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_p50'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_p50'] / 1000
        df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_p95'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_p95'] / 1000
        df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_p5'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_p5'] / 1000

    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean'] = df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] / 1000
    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std'] = df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] / 1000

    # ==================================================================
    # Cost per tonne CO2 metrics - NOW INDEXED
    # ==================================================================
    cost_mean = (df[f'{scenario_name}_cost_{scenario_name}_mean'] / 1000 ) 
    cost_std = (df[f'{scenario_name}_cost_{scenario_name}_std'] / 1000) 
    
    # Cost per net ton CO2
    df[f'cost_per_net_ton_co2_{measure_type}_thousands'] = cost_mean / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean']
    df[f'cost_per_net_ton_co2_{measure_type}_std_thousands'] = df[f'cost_per_net_ton_co2_{measure_type}_thousands'] * np.sqrt(
        (cost_std / cost_mean)**2 + 
        (df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std'] / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean'])**2
    )

    # Cost per gas ton reductions
    df[f'cost_per_gas_ton_reductions_{measure_type}_th'] = cost_mean / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean']
    df[f'cost_per_gas_ton_co2_{measure_type}_std_th'] = df[f'cost_per_gas_ton_reductions_{measure_type}_th'] * np.sqrt(
        (cost_std / cost_mean)**2 + 
        (df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_std'] / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean'])**2
    )

    return df


# def clean_post_proccess(df, measure_type, scenario_name, years, n_simulations, 
#          GAS_CARBON_FACTOR_2022, elec_carbon_factor):
#     """
#     Process energy and carbon savings data for different measure scenarios.
    
#     Parameters:
#     - df: DataFrame with energy consumption data
#     - measure_type: Type of energy efficiency measure
#     - scenario_name: Name of the scenario (e.g., 'heat_pump_only', 'join_heat_ins_decay')
#     - years: Number of years for projections
#     - n_simulations: Number of Monte Carlo simulations
#     - GAS_CARBON_FACTOR_2022: Carbon factor for gas (kg CO2/kWh)
#     - elec_carbon_factor: Carbon factor for electricity (kg CO2/kWh)
#     """
    
#     # ==================================================================
#     # Gas energy changes
#     # ==================================================================
#     df[f'gas_{years}yr_kwh_change_{measure_type}_mean'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_{scenario_name}_gas_mean']
#     )
#     df[f'gas_{years}yr_kwh_change_{measure_type}_p50'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_{scenario_name}_gas_p50']
#     )
#     df[f'gas_{years}yr_kwh_change_{measure_type}_p95'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_{scenario_name}_gas_p95']
#     )
#     df[f'gas_{years}yr_kwh_change_{measure_type}_p5'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_{scenario_name}_gas_p5']
#     )
#     df[f'gas_{years}yr_kwh_change_{measure_type}_std'] = (
#         df['total_gas_derived'] * years * 
#         df[f'{scenario_name}_{scenario_name}_gas_std']
#     )
    
#     # ==================================================================
#     # Electricity energy changes (for heat pump scenarios only)
#     # ==================================================================
#     if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
#         for stat in ['mean', 'p5', 'p50', 'p95', 'stat']:
#             df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] = (
#                 df['total_elec_derived'] * years * 
#                 df[f'{scenario_name}_{scenario_name}_electricity_{stat}']
#             )
#         # df[f'elec_{years}yr_kwh_change_{measure_type}_mean'] = (
#         #     df['total_elec_derived'] * years * 
#         #     df[f'{scenario_name}_{scenario_name}_electricity_mean']
#         # )
#         # df[f'elec_{years}yr_kwh_change_{measure_type}_std'] = (
#         #     df['total_elec_derived'] * years * 
#         #     df[f'{scenario_name}_{scenario_name}_electricity_std']
#         # )
    
#     # ==================================================================
#     # Gas carbon savings metrics
#     # ==================================================================
#     df[f'gas_{years}yr_kg_co2_saved_mean'] = (
#         df[f'gas_{years}yr_kwh_change_{measure_type}_mean'] * GAS_CARBON_FACTOR_2022  # Fixed
#     )

#     df[f'gas_{years}yr_kg_co2_saved_p50'] = (
#         df[f'gas_{years}yr_kwh_change_{measure_type}_p50'] * GAS_CARBON_FACTOR_2022  # Fixed
#     )
#     df[f'gas_{years}yr_kg_co2_saved_p95'] = (
#         df[f'gas_{years}yr_kwh_change_{measure_type}_p95'] * GAS_CARBON_FACTOR_2022  # Fixed
#     )

#     df[f'gas_{years}yr_kg_co2_saved_p5'] = (
#         df[f'gas_{years}yr_kwh_change_{measure_type}_p5'] * GAS_CARBON_FACTOR_2022  # Fixed
#     )
#     df[f'gas_{years}yr_kg_co2_saved_std'] = (
#         df[f'gas_{years}yr_kwh_change_{measure_type}_std'] * GAS_CARBON_FACTOR_2022  # Fixed
#     )
#     df[f'gas_{years}yr_kg_co2_saved_se'] = (
#         df[f'gas_{years}yr_kg_co2_saved_std'] / np.sqrt(n_simulations)
#     )
#     df[f'gas_{years}yr_kg_co2_saved_r_se'] = (  # Relative standard error
#         df[f'gas_{years}yr_kg_co2_saved_se'] / df[f'gas_{years}yr_kg_co2_saved_mean']  # Fixed
#     )

#     # ==================================================================
#     # Electricity carbon savings (heat pump scenarios only)
#     # ==================================================================
#     if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
#         df[f'elec_{years}yr_kg_co2_saved_mean'] = (
#             df[f'elec_{years}yr_kwh_change_{measure_type}_mean'] * elec_carbon_factor
#         )

#         df[f'elec_{years}yr_kg_co2_saved_p50'] = (
#             df[f'elec_{years}yr_kwh_change_{measure_type}_p50'] * elec_carbon_factor  # Fixed
#         )
#         df[f'elec_{years}yr_kg_co2_saved_p95'] = (
#             df[f'elec_{years}yr_kwh_change_{measure_type}_p95'] * elec_carbon_factor  # Fixed
#         )

#         df[f'elec_{years}yr_kg_co2_saved_p5'] = (
#             df[f'elec_{years}yr_kwh_change_{measure_type}_p5'] * elec_carbon_factor  # Fixed
#     )
#         df[f'elec_{years}yr_kg_co2_saved_std'] = (
#             df[f'elec_{years}yr_kwh_change_{measure_type}_std'] * elec_carbon_factor
#         )
#         df[f'elec_{years}yr_kg_co2_saved_se'] = (  # Fixed: consistent naming
#             df[f'elec_{years}yr_kg_co2_saved_std'] / np.sqrt(n_simulations)
#         )   
        
#         # Net carbon savings (gas + electricity)
#         df[f'total_kg_co2_saved_{years}yr_mean'] = (
#             df[f'gas_{years}yr_kg_co2_saved_mean'] + 
#             df[f'elec_{years}yr_kg_co2_saved_mean']
#         )
#         df[f'total_kg_co2_saved_{years}yr_std'] = np.sqrt(
#             df[f'gas_{years}yr_kg_co2_saved_std']**2 + 
#             df[f'elec_{years}yr_kg_co2_saved_std']**2
#         )
#     else:
#         # For non-heat pump scenarios, total equals gas only
#         df[f'total_kg_co2_saved_{years}yr_mean'] = df[f'gas_{years}yr_kg_co2_saved_mean']
#         df[f'total_kg_co2_saved_{years}yr_std'] = df[f'gas_{years}yr_kg_co2_saved_std']


    
#     df[f'gas_total_tonne_co2_saved_{years}yr_mean'] =  df[f'gas_{years}yr_kg_co2_saved_mean'] / 1000
#     df[f'gas_total_tonne_co2_saved_{years}yr_std'] =   df[f'gas_{years}yr_kg_co2_saved_std']/ 1000

#     df[f'total_tonne_co2_saved_{years}yr_mean'] =  df[f'total_kg_co2_saved_{years}yr_mean'] / 1000
#     df[f'total_tonne_co2_saved_{years}yr_std'] =   df[f'total_kg_co2_saved_{years}yr_std']/ 1000

#     # cost per kg co2 saings 
#     df['cost_per_net_ton_co2'] = df[f'{scenario_name}_cost_{scenario_name}_mean' ] /   df[f'total_tonne_co2_saved_{years}yr_mean']
#     df['cost_per_net_ton_co2_std'] = df['cost_per_net_ton_co2'] *  np.sqrt( ( df[f'{scenario_name}_cost_{scenario_name}_std' ]/df[f'{scenario_name}_cost_{scenario_name}_mean' ])**2 + (  df[f'total_tonne_co2_saved_{years}yr_std']/  df[f'total_tonne_co2_saved_{years}yr_mean'])**2 ) 

#     df['cost_per_gas_ton_redutions'] = df[f'{scenario_name}_cost_{scenario_name}_mean' ] /   df[f'gas_total_tonne_co2_saved_{years}yr_mean']
#     df['cost_per_gas_ton_co2_std'] = df['cost_per_gas_ton_redutions'] *  np.sqrt( ( df[f'{scenario_name}_cost_{scenario_name}_std' ]/df[f'{scenario_name}_cost_{scenario_name}_mean' ])**2 + (  df[f'gas_total_tonne_co2_saved_{years}yr_std']/  df[f'gas_total_tonne_co2_saved_{years}yr_mean'])**2 ) 

#     return df