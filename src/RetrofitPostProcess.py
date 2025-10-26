

import pandas as pd 
import glob 
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import sys
sys.path.append('/rds/user/gb669/hpc-work/energy_map/RetrofitModel')
from src.validate import validate

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
    
    df_processed = df[df['premise_type']!='Domestic outbuilding'].copy()
    if df_processed.shape[0]== df.shape[0]:
        raise Exception('No rows filtered out')
    # Process eachscenario
    for measure_type, scenario_name in scenarios_config:
        print(f"Processing scenario: {scenario_name} (measure type: {measure_type})")
        
        df_processed = clean_post_process(
            df=df_processed,
            measure_type=measure_type,
            scenario_name=scenario_name,
            years=years,
            n_simulations=n_simulations,
            GAS_CARBON_FACTOR_2022=GAS_CARBON_FACTOR_2022,
            elec_carbon_factor=elec_carbon_factor
        )
    
    return df_processed


def clean_post_process(df, measure_type, scenario_name, years, n_simulations, 
                        GAS_CARBON_FACTOR_2022, elec_carbon_factor):
    """
    Process energy and carbon savings data for different measure scenarios.
    
    All costs are kept in POUNDS (£) throughout.
    Carbon savings are in TONNES with negative values indicating reductions.
    
    Parameters:
    - df: DataFrame with energy consumption data
    - measure_type: Type of energy efficiency measure
    - scenario_name: Name of the scenario (e.g., 'heat_pump_only', 'join_heat_ins_decay')
    - years: Number of years for projections
    - n_simulations: Number of Monte Carlo simulations
    - GAS_CARBON_FACTOR_2022: Carbon factor for gas (kg CO2/kWh)
    - elec_carbon_factor: Carbon factor for electricity (kg CO2/kWh)
    """
    elec_scenarios = ['heat_pump_only', 'join_heat_ins_decay', 'join_heat_ins_add']
    stats = ['mean', 'p5', 'p50', 'p95', 'std']
    
    if scenario_name in elec_scenarios: 
        fuels = ['gas', 'elec']
    else:
        fuels = ['gas']
    
    # ==================================================================
    # ENERGY CHANGES - GAS
    # ==================================================================
    for stat in stats:
        df[f'gas_{years}yr_kwh_change_{measure_type}_{stat}'] = (
            df['total_gas_derived'] * years * 
            df[f'{scenario_name}_{scenario_name}_gas_{stat}']
        )
    
    # ==================================================================
    # ENERGY CHANGES - ELECTRICITY (for heat pump scenarios only)
    # ==================================================================
    if scenario_name in elec_scenarios:
        for stat in stats:
            df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] = (
                df['total_elec_derived'] * years * 
                df[f'{scenario_name}_{scenario_name}_electricity_{stat}']
            )
    
    # ==================================================================
    # CARBON SAVINGS - GAS
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
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_se'] / 
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean']
    )

    # ==================================================================
    # CARBON SAVINGS - ELECTRICITY (heat pump scenarios only)
    # ==================================================================
    if scenario_name in elec_scenarios:
        for stat in stats:
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_{stat}'] = (
                df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] * elec_carbon_factor
            )
        
        # Standard error for electricity
        df[f'elec_{years}yr_kg_co2_saved_{measure_type}_se'] = (
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_std'] / np.sqrt(n_simulations)
        )
        
        # Net carbon savings (gas + electricity)
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean'] + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_mean']
        )
             
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p95'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p95'] + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_p95']
        )

        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p50'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p50'] + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_p50']
        )
        
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p5'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p5'] + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_p5']
        )

        df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] = np.sqrt(
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std']**2 + 
            df[f'elec_{years}yr_kg_co2_saved_{measure_type}_std']**2
        )
    else:
        # For non-heat pump scenarios, total equals gas only
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean']
        )
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p95'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p95']
        )
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p50'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p50']
        )
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p5'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p5']
        )
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] = (
            df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std']
        )

    # ==================================================================
    # CONVERT KG TO TONNES
    # ==================================================================
    
    # Gas only (in tonnes)
    df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean'] / 1000
    )
    df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_std'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std'] / 1000
    )
    df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p50'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p50'] / 1000
    )
    df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p95'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p95'] / 1000
    )
    df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p5'] = (
        df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p5'] / 1000
    )

    # Total (gas + elec, in tonnes)
    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean'] = (
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] / 1000
    )
    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std'] = (
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] / 1000
    )
    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p50'] = (
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p50'] / 1000
    )
    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p95'] = (
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p95'] / 1000
    )
    df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p5'] = (
        df[f'total_kg_co2_saved_{measure_type}_{years}yr_p5'] / 1000
    )

    # ==================================================================
    # COST PER TONNE CO2 - NET (GAS + ELECTRICITY)
    # All costs in POUNDS (£)
    # ==================================================================
    
    # Get cost columns (already in pounds)
    cost_mean = df[f'{scenario_name}_cost_{scenario_name}_mean']
    cost_std = df[f'{scenario_name}_cost_{scenario_name}_std']
    cost_p50 = df[f'{scenario_name}_cost_{scenario_name}_p50']
    cost_p5 = df[f'{scenario_name}_cost_{scenario_name}_p5']
    cost_p95 = df[f'{scenario_name}_cost_{scenario_name}_p95']
    
    # Get carbon savings columns (in tonnes, negative = reduction)
    carbon_mean = df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean']
    carbon_std = df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std']
    carbon_p50 = df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p50']
    carbon_p95 = df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p95']
    carbon_p5 = df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p5']
    
    # Safety check: only calculate where carbon savings are meaningful
    # # Avoid division by near-zero values
    # min_carbon_threshold = 0.01  # At least 10 kg CO2
    # valid_mask = carbon_mean.abs() > min_carbon_threshold
    
    # Cost per tonne - NET (in £/tCO2)
    df[f'cost_per_net_ton_co2_{measure_type}_mean'] = cost_mean / carbon_mean
    
    df[f'cost_per_net_ton_co2_{measure_type}_p50'] =   cost_p50 / carbon_p50
 
    df[f'cost_per_net_ton_co2_{measure_type}_p95'] = cost_p95 / carbon_p95 
    
    df[f'cost_per_net_ton_co2_{measure_type}_p5'] =  cost_p5 / carbon_p5
 
    
    # Propagate uncertainty using error propagation
    df[f'cost_per_net_ton_co2_{measure_type}_std'] =   df[f'cost_per_net_ton_co2_{measure_type}_mean'] * np.sqrt(
            (cost_std / cost_mean)**2 + 
            (carbon_std / carbon_mean)**2 ) 
 
    
    # Convert to thousands for easier reading (in £k/tCO2)
    df[f'cost_per_net_ton_co2_{measure_type}_mean_thousands'] = (
        df[f'cost_per_net_ton_co2_{measure_type}_mean'] / 1000
    )
    df[f'cost_per_net_ton_co2_{measure_type}_std_thousands'] = (
        df[f'cost_per_net_ton_co2_{measure_type}_std'] / 1000
    )
    df[f'cost_per_net_ton_co2_{measure_type}_p50_thousands'] = (
        df[f'cost_per_net_ton_co2_{measure_type}_p50'] / 1000
    )
    df[f'cost_per_net_ton_co2_{measure_type}_p95_thousands'] = (
        df[f'cost_per_net_ton_co2_{measure_type}_p95'] / 1000
    )
    df[f'cost_per_net_ton_co2_{measure_type}_p5_thousands'] = (
        df[f'cost_per_net_ton_co2_{measure_type}_p5'] / 1000
    )

    # ==================================================================
    # COST PER TONNE CO2 - GAS ONLY
    # All costs in POUNDS (£)
    # ==================================================================
    
    # Get gas-only carbon savings
    gas_carbon_mean = df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean']
    gas_carbon_std = df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_std']
    gas_carbon_p50 = df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p50']
    gas_carbon_p95 = df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p95']
    gas_carbon_p5 = df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p5']
    
    # Safety check for gas carbon savings
    # valid_gas_mask = gas_carbon_mean.abs() > min_carbon_threshold
    
    # Cost per tonne - GAS ONLY (in £/tCO2)
    df[f'cost_per_gas_ton_reductions_{measure_type}_mean'] =  cost_mean / gas_carbon_mean
 
    
    df[f'cost_per_gas_ton_reductions_{measure_type}_p50'] =   cost_p50 / gas_carbon_p50 
 
    df[f'cost_per_gas_ton_reductions_{measure_type}_p95'] =  cost_p95 / gas_carbon_p95
 
    
    df[f'cost_per_gas_ton_reductions_{measure_type}_p5'] =   cost_p5 / gas_carbon_p5
        
    
    df[f'cost_per_gas_ton_co2_{measure_type}_std'] =  df[f'cost_per_gas_ton_reductions_{measure_type}_mean'] * np.sqrt(
            (cost_std / cost_mean)**2 + 
            (gas_carbon_std / gas_carbon_mean)**2
        )
    # Convert to thousands (in £k/tCO2)
    df[f'cost_per_gas_ton_reductions_{measure_type}_mean_thousands'] = (
        df[f'cost_per_gas_ton_reductions_{measure_type}_mean'] / 1000
    )
    df[f'cost_per_gas_ton_co2_{measure_type}_std_thousands'] = (
        df[f'cost_per_gas_ton_co2_{measure_type}_std'] / 1000
    )
    df[f'cost_per_gas_ton_reductions_{measure_type}_p50_thousands'] = (
        df[f'cost_per_gas_ton_reductions_{measure_type}_p50'] / 1000
    )
    df[f'cost_per_gas_ton_reductions_{measure_type}_p95_thousands'] = (
        df[f'cost_per_gas_ton_reductions_{measure_type}_p95'] / 1000
    )
    df[f'cost_per_gas_ton_reductions_{measure_type}_p5_thousands'] = (
        df[f'cost_per_gas_ton_reductions_{measure_type}_p5'] / 1000
    )

    # ==================================================================
    # # DIAGNOSTIC OUTPUT
    # # ==================================================================
    # print(f"\n{'='*70}")
    # print(f"SCENARIO: {scenario_name} ({measure_type})")
    # print(f"{'='*70}")
    
    # # Show a sample building
    # sample_idx = df.index[0] if len(df) > 0 else None
    # if sample_idx is not None:
    #     print(f"\nSample building (index {sample_idx}):")
    #     print(f"  Cost (mean): £{cost_mean.iloc[0]:,.0f}")
    #     print(f"  Carbon saved (total): {carbon_mean.iloc[0]:.3f} tonnes")
    #     print(f"  Carbon saved (gas only): {gas_carbon_mean.iloc[0]:.3f} tonnes")
    #     print(f"  Cost per tonne (net): £{df[f'cost_per_net_ton_co2_{measure_type}_mean'].iloc[0]:,.0f}/tCO2")
    #     print(f"  Cost per tonne (net, thousands): £{df[f'cost_per_net_ton_co2_{measure_type}_mean_thousands'].iloc[0]:,.1f}k/tCO2")
    
    # # Summary statistics
    # print(f"\nSummary statistics (all buildings):")
    # # print(f"  Valid cost-per-ton calculations: {valid_mask.sum()} / {len(df)}")
    # print(f"  Median carbon saved (total): {carbon_mean.median():.3f} tonnes")
    # print(f"  Median cost per tonne (net): £{df[f'cost_per_net_ton_co2_{measure_type}_mean'].median():,.0f}/tCO2")
    
    # # # Identify problematic cases
    # # problematic = (~valid_mask).sum()
    # # if problematic > 0:
    # #     print(f"\n⚠️  WARNING: {problematic} buildings excluded due to minimal carbon savings (<{min_carbon_threshold} tonnes)")

    return df


# import numpy as np 


# def process_multiple_scenarios(df, scenarios_config, years, n_simulations, 
#                                 GAS_CARBON_FACTOR_2022, elec_carbon_factor):
#     """
#     Process energy and carbon savings data for multiple measure scenarios.
    
#     Parameters:
#     - df: DataFrame with energy consumption data for all scenarios
#     - scenarios_config: List of tuples (measure_type, scenario_name) or dict {measure_type: scenario_name}
#                        e.g., [('heat_pump', 'heat_pump_only'), 
#                               ('insulation', 'join_heat_ins_decay')]
#     - years: Number of years for projections
#     - n_simulations: Number of Monte Carlo simulations
#     - GAS_CARBON_FACTOR_2022: Carbon factor for gas (kg CO2/kWh)
#     - elec_carbon_factor: Carbon factor for electricity (kg CO2/kWh)
    
#     Returns:
#     - df: DataFrame with all scenarios processed
#     """
    
#     # Convert dict to list of tuples if needed
#     if isinstance(scenarios_config, dict):
#         scenarios_config = list(scenarios_config.items())
    
#     # Make a copy to avoid modifying original
#     df_processed = df.copy()
    
#     # Process each scenario
#     for measure_type, scenario_name in scenarios_config:
#         print(f"Processing scenario: {scenario_name} (measure type: {measure_type})")
        
#         df_processed = clean_post_proccess(
#             df=df_processed,
#             measure_type=measure_type,
#             scenario_name=scenario_name,
#             years=years,
#             n_simulations=n_simulations,
#             GAS_CARBON_FACTOR_2022=GAS_CARBON_FACTOR_2022,
#             elec_carbon_factor=elec_carbon_factor
#         )
    
#     return df_processed



# def clean_post_proccess(df, measure_type, scenario_name, years, n_simulations, 
#                         GAS_CARBON_FACTOR_2022, elec_carbon_factor):
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
#     elec_scenarios = ['heat_pump_only', 'join_heat_ins_decay', 'join_heat_ins_add']
#     stats = ['mean', 'p5', 'p50', 'p95', 'std']
#     if scenario_name in elec_scenarios: 
#         fuels = ['gas', 'elec']
#     else:
#         fuels = ['gas']
#     # ==================================================================
#     # Convert cost to millions  
#     # ==================================================================
#     million = 1_000_000
 
#     df[f'{scenario_name}_cost_{scenario_name}_mean_mill'] = (df[f'{scenario_name}_cost_{scenario_name}_mean'] / million ) 
#     df[f'{scenario_name}_cost_{scenario_name}_std_mill'] = (df[f'{scenario_name}_cost_{scenario_name}_std'] / million ) 
    
     
#     df[f'{scenario_name}_cost_{scenario_name}_p50_mill'] = (df[f'{scenario_name}_cost_{scenario_name}_p50'] / million ) 
#     df[f'{scenario_name}_cost_{scenario_name}_p5_mill'] = (df[f'{scenario_name}_cost_{scenario_name}_p5'] / million ) 
#     df[f'{scenario_name}_cost_{scenario_name}_p95_mill'] = (df[f'{scenario_name}_cost_{scenario_name}_p95'] / million ) 
    
#     # ==================================================================
#     # Gas energy changes
#     # ==================================================================
#     for stat in stats:
#         df[f'gas_{years}yr_kwh_change_{measure_type}_{stat}'] = (
#             df['total_gas_derived'] * years * 
#             df[f'{scenario_name}_{scenario_name}_gas_{stat}']
#         )
    
#     # ==================================================================
#     # Electricity energy changes (for heat pump scenarios only)
#     # ==================================================================
#     if scenario_name in elec_scenarios :
#         for stat in stats:
#             df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] = (
#                 df['total_elec_derived'] * years * 
#                 df[f'{scenario_name}_{scenario_name}_electricity_{stat}']
#             )
    
#     # ==================================================================
#     # Gas carbon savings metrics
#     # ==================================================================
#     for stat in stats:
#         df[f'gas_{years}yr_kg_co2_saved_{measure_type}_{stat}'] = (
#             df[f'gas_{years}yr_kwh_change_{measure_type}_{stat}'] * GAS_CARBON_FACTOR_2022
#         )
    
#     # Standard error and relative standard error for gas
#     df[f'gas_{years}yr_kg_co2_saved_{measure_type}_se'] = (
#         df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std'] / np.sqrt(n_simulations)
#     )
#     df[f'gas_{years}yr_kg_co2_saved_{measure_type}_r_se'] = (
#         df[f'gas_{years}yr_kg_co2_saved_{measure_type}_se'] / df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean']
#     )

#     # ==================================================================
#     # Electricity carbon savings (heat pump scenarios only)
#     # ==================================================================
#     if scenario_name in elec_scenarios :
#         for stat in stats:
#             df[f'elec_{years}yr_kg_co2_saved_{measure_type}_{stat}'] = (
#                 df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] * elec_carbon_factor
#             )
        
#         # Standard error for electricity
#         df[f'elec_{years}yr_kg_co2_saved_{measure_type}_se'] = (
#             df[f'elec_{years}yr_kg_co2_saved_{measure_type}_std'] / np.sqrt(n_simulations)
#         )
        
#         # Net carbon savings (gas + electricity) - NOW INDEXED
#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] = (
#             df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean'] + 
#             df[f'elec_{years}yr_kg_co2_saved_{measure_type}_mean']
#         )
             
#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_p95'] = (
#             df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p95'] + 
#             df[f'elec_{years}yr_kg_co2_saved_{measure_type}_p95']
#         )

#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_p50'] = (
#             df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p50'] + 
#             df[f'elec_{years}yr_kg_co2_saved_{measure_type}_p50']
#         )

#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] = np.sqrt(
#             df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std']**2 + 
#             df[f'elec_{years}yr_kg_co2_saved_{measure_type}_std']**2
#         )
#     else:
#         # For non-heat pump scenarios, total equals gas only - NOW INDEXED
#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] = df[f'gas_{years}yr_kg_co2_saved_{measure_type}_mean']
#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_p95'] = df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p95']
#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_p50'] = df[f'gas_{years}yr_kg_co2_saved_{measure_type}_p50']
#         df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] = df[f'gas_{years}yr_kg_co2_saved_{measure_type}_std']

#     # ==================================================================
#     # Convert to tonnes - NOW INDEXED
#     # ==================================================================

#     for fuel in fuels:
#         df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_mean'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_mean'] / 1000
#         df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_std'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_std'] / 1000
#         df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_p50'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_p50'] / 1000
#         df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_p95'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_p95'] / 1000
#         df[f'{fuel}_total_tonne_co2_saved_{measure_type}_{years}yr_p5'] = df[f'{fuel}_{years}yr_kg_co2_saved_{measure_type}_p5'] / 1000

#     df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean'] = df[f'total_kg_co2_saved_{measure_type}_{years}yr_mean'] / 1000
#     df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p50'] = df[f'total_kg_co2_saved_{measure_type}_{years}yr_p50'] / 1000
#     df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p95'] = df[f'total_kg_co2_saved_{measure_type}_{years}yr_p95'] / 1000
#     df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std'] = df[f'total_kg_co2_saved_{measure_type}_{years}yr_std'] / 1000

#     # ==================================================================
#     # Cost per tonne CO2 metrics - NOW INDEXED
#     # ==================================================================
#     cost_mean_mill = df[f'{scenario_name}_cost_{scenario_name}_mean_mill']
#     cost_p95_mill = df[f'{scenario_name}_cost_{scenario_name}_p95_mill']
#     cost_p50_mill = df[f'{scenario_name}_cost_{scenario_name}_p50_mill']
#     cost_std_mill = df[f'{scenario_name}_cost_{scenario_name}_std_mill']
    
#     cost_mean = df[f'{scenario_name}_cost_{scenario_name}_mean']  
#     cost_std =  df[f'{scenario_name}_cost_{scenario_name}_std'] 
#     cost_p50 =  df[f'{scenario_name}_cost_{scenario_name}_p50']
#     cost_p95 = df[f'{scenario_name}_cost_{scenario_name}_p95']
    
#     # Cost per net ton CO2
#     df[f'cost_per_net_ton_co2_{measure_type}_mean_thousands'] = cost_mean / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean']
#     df[f'cost_per_net_ton_co2_{measure_type}_p50_thousands'] = cost_p50 / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p50']
#     df[f'cost_per_net_ton_co2_{measure_type}_p95_thousands'] = cost_p95 / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p95']
#     df[f'cost_per_net_ton_co2_{measure_type}_std_thousands'] = df[f'cost_per_net_ton_co2_{measure_type}_mean_thousands'] * np.sqrt(
#         (cost_std / cost_mean)**2 + 
#         (df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std'] / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean'])**2
#     )

#     df[f'cost_per_net_ton_co2_{measure_type}_mean_mill'] = cost_mean_mill / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean']
#     df[f'cost_per_net_ton_co2_{measure_type}_p50_mill'] = cost_p50_mill / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p50']
#     df[f'cost_per_net_ton_co2_{measure_type}_p95_mill'] = cost_p95_mill / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_p95']
#     df[f'cost_per_net_ton_co2_{measure_type}_std_mill'] = df[f'cost_per_net_ton_co2_{measure_type}_mean_mill'] * np.sqrt(
#         (cost_std_mill / cost_mean_mill)**2 + 
#         (df[f'total_tonne_co2_saved_{measure_type}_{years}yr_std'] / df[f'total_tonne_co2_saved_{measure_type}_{years}yr_mean'])**2
#     )

#     # Cost per gas ton reductions
#     df[f'cost_per_gas_ton_reductions_{measure_type}_mean'] = cost_mean / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean']
#     df[f'cost_per_gas_ton_reductions_{measure_type}_p50'] = cost_p50 / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p50']
#     df[f'cost_per_gas_ton_reductions_{measure_type}_p95'] = cost_p95 / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p95']
#     df[f'cost_per_gas_ton_co2_{measure_type}_std'] = df[f'cost_per_gas_ton_reductions_{measure_type}_mean'] * np.sqrt(
#         (cost_std / cost_mean)**2 + 
#         (df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_std'] / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean'])**2
#     )
        
#     df[f'cost_per_gas_ton_reductions_{measure_type}_mean_mill'] = cost_mean_mill / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean']
#     df[f'cost_per_gas_ton_reductions_{measure_type}_p50_mill'] = cost_p50_mill / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p50']
#     df[f'cost_per_gas_ton_reductions_{measure_type}_p95_mill'] = cost_p95_mill / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_p95']
#     df[f'cost_per_gas_ton_co2_{measure_type}_std_mill'] = df[f'cost_per_gas_ton_reductions_{measure_type}_mean_mill'] * np.sqrt(
#         (cost_std_mill / cost_mean_mill)**2 + 
#         (df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_std'] / df[f'gas_total_tonne_co2_saved_{measure_type}_{years}yr_mean'])**2
#     )

#     return df


# # def clean_post_proccess(df, measure_type, scenario_name, years, n_simulations, 
# #          GAS_CARBON_FACTOR_2022, elec_carbon_factor):
# #     """
# #     Process energy and carbon savings data for different measure scenarios.
    
# #     Parameters:
# #     - df: DataFrame with energy consumption data
# #     - measure_type: Type of energy efficiency measure
# #     - scenario_name: Name of the scenario (e.g., 'heat_pump_only', 'join_heat_ins_decay')
# #     - years: Number of years for projections
# #     - n_simulations: Number of Monte Carlo simulations
# #     - GAS_CARBON_FACTOR_2022: Carbon factor for gas (kg CO2/kWh)
# #     - elec_carbon_factor: Carbon factor for electricity (kg CO2/kWh)
# #     """
    
# #     # ==================================================================
# #     # Gas energy changes
# #     # ==================================================================
# #     df[f'gas_{years}yr_kwh_change_{measure_type}_mean'] = (
# #         df['total_gas_derived'] * years * 
# #         df[f'{scenario_name}_{scenario_name}_gas_mean']
# #     )
# #     df[f'gas_{years}yr_kwh_change_{measure_type}_p50'] = (
# #         df['total_gas_derived'] * years * 
# #         df[f'{scenario_name}_{scenario_name}_gas_p50']
# #     )
# #     df[f'gas_{years}yr_kwh_change_{measure_type}_p95'] = (
# #         df['total_gas_derived'] * years * 
# #         df[f'{scenario_name}_{scenario_name}_gas_p95']
# #     )
# #     df[f'gas_{years}yr_kwh_change_{measure_type}_p5'] = (
# #         df['total_gas_derived'] * years * 
# #         df[f'{scenario_name}_{scenario_name}_gas_p5']
# #     )
# #     df[f'gas_{years}yr_kwh_change_{measure_type}_std'] = (
# #         df['total_gas_derived'] * years * 
# #         df[f'{scenario_name}_{scenario_name}_gas_std']
# #     )
    
# #     # ==================================================================
# #     # Electricity energy changes (for heat pump scenarios only)
# #     # ==================================================================
# #     if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
# #         for stat in ['mean', 'p5', 'p50', 'p95', 'stat']:
# #             df[f'elec_{years}yr_kwh_change_{measure_type}_{stat}'] = (
# #                 df['total_elec_derived'] * years * 
# #                 df[f'{scenario_name}_{scenario_name}_electricity_{stat}']
# #             )
# #         # df[f'elec_{years}yr_kwh_change_{measure_type}_mean'] = (
# #         #     df['total_elec_derived'] * years * 
# #         #     df[f'{scenario_name}_{scenario_name}_electricity_mean']
# #         # )
# #         # df[f'elec_{years}yr_kwh_change_{measure_type}_std'] = (
# #         #     df['total_elec_derived'] * years * 
# #         #     df[f'{scenario_name}_{scenario_name}_electricity_std']
# #         # )
    
# #     # ==================================================================
# #     # Gas carbon savings metrics
# #     # ==================================================================
# #     df[f'gas_{years}yr_kg_co2_saved_mean'] = (
# #         df[f'gas_{years}yr_kwh_change_{measure_type}_mean'] * GAS_CARBON_FACTOR_2022  # Fixed
# #     )

# #     df[f'gas_{years}yr_kg_co2_saved_p50'] = (
# #         df[f'gas_{years}yr_kwh_change_{measure_type}_p50'] * GAS_CARBON_FACTOR_2022  # Fixed
# #     )
# #     df[f'gas_{years}yr_kg_co2_saved_p95'] = (
# #         df[f'gas_{years}yr_kwh_change_{measure_type}_p95'] * GAS_CARBON_FACTOR_2022  # Fixed
# #     )

# #     df[f'gas_{years}yr_kg_co2_saved_p5'] = (
# #         df[f'gas_{years}yr_kwh_change_{measure_type}_p5'] * GAS_CARBON_FACTOR_2022  # Fixed
# #     )
# #     df[f'gas_{years}yr_kg_co2_saved_std'] = (
# #         df[f'gas_{years}yr_kwh_change_{measure_type}_std'] * GAS_CARBON_FACTOR_2022  # Fixed
# #     )
# #     df[f'gas_{years}yr_kg_co2_saved_se'] = (
# #         df[f'gas_{years}yr_kg_co2_saved_std'] / np.sqrt(n_simulations)
# #     )
# #     df[f'gas_{years}yr_kg_co2_saved_r_se'] = (  # Relative standard error
# #         df[f'gas_{years}yr_kg_co2_saved_se'] / df[f'gas_{years}yr_kg_co2_saved_mean']  # Fixed
# #     )

# #     # ==================================================================
# #     # Electricity carbon savings (heat pump scenarios only)
# #     # ==================================================================
# #     if scenario_name in ['heat_pump_only', 'join_heat_ins_decay']:
# #         df[f'elec_{years}yr_kg_co2_saved_mean'] = (
# #             df[f'elec_{years}yr_kwh_change_{measure_type}_mean'] * elec_carbon_factor
# #         )

# #         df[f'elec_{years}yr_kg_co2_saved_p50'] = (
# #             df[f'elec_{years}yr_kwh_change_{measure_type}_p50'] * elec_carbon_factor  # Fixed
# #         )
# #         df[f'elec_{years}yr_kg_co2_saved_p95'] = (
# #             df[f'elec_{years}yr_kwh_change_{measure_type}_p95'] * elec_carbon_factor  # Fixed
# #         )

# #         df[f'elec_{years}yr_kg_co2_saved_p5'] = (
# #             df[f'elec_{years}yr_kwh_change_{measure_type}_p5'] * elec_carbon_factor  # Fixed
# #     )
# #         df[f'elec_{years}yr_kg_co2_saved_std'] = (
# #             df[f'elec_{years}yr_kwh_change_{measure_type}_std'] * elec_carbon_factor
# #         )
# #         df[f'elec_{years}yr_kg_co2_saved_se'] = (  # Fixed: consistent naming
# #             df[f'elec_{years}yr_kg_co2_saved_std'] / np.sqrt(n_simulations)
# #         )   
        
# #         # Net carbon savings (gas + electricity)
# #         df[f'total_kg_co2_saved_{years}yr_mean'] = (
# #             df[f'gas_{years}yr_kg_co2_saved_mean'] + 
# #             df[f'elec_{years}yr_kg_co2_saved_mean']
# #         )
# #         df[f'total_kg_co2_saved_{years}yr_std'] = np.sqrt(
# #             df[f'gas_{years}yr_kg_co2_saved_std']**2 + 
# #             df[f'elec_{years}yr_kg_co2_saved_std']**2
# #         )
# #     else:
# #         # For non-heat pump scenarios, total equals gas only
# #         df[f'total_kg_co2_saved_{years}yr_mean'] = df[f'gas_{years}yr_kg_co2_saved_mean']
# #         df[f'total_kg_co2_saved_{years}yr_std'] = df[f'gas_{years}yr_kg_co2_saved_std']


    
# #     df[f'gas_total_tonne_co2_saved_{years}yr_mean'] =  df[f'gas_{years}yr_kg_co2_saved_mean'] / 1000
# #     df[f'gas_total_tonne_co2_saved_{years}yr_std'] =   df[f'gas_{years}yr_kg_co2_saved_std']/ 1000

# #     df[f'total_tonne_co2_saved_{years}yr_mean'] =  df[f'total_kg_co2_saved_{years}yr_mean'] / 1000
# #     df[f'total_tonne_co2_saved_{years}yr_std'] =   df[f'total_kg_co2_saved_{years}yr_std']/ 1000

# #     # cost per kg co2 saings 
# #     df['cost_per_net_ton_co2'] = df[f'{scenario_name}_cost_{scenario_name}_mean' ] /   df[f'total_tonne_co2_saved_{years}yr_mean']
# #     df['cost_per_net_ton_co2_std'] = df['cost_per_net_ton_co2'] *  np.sqrt( ( df[f'{scenario_name}_cost_{scenario_name}_std' ]/df[f'{scenario_name}_cost_{scenario_name}_mean' ])**2 + (  df[f'total_tonne_co2_saved_{years}yr_std']/  df[f'total_tonne_co2_saved_{years}yr_mean'])**2 ) 

# #     df['cost_per_gas_ton_redutions'] = df[f'{scenario_name}_cost_{scenario_name}_mean' ] /   df[f'gas_total_tonne_co2_saved_{years}yr_mean']
# #     df['cost_per_gas_ton_co2_std'] = df['cost_per_gas_ton_redutions'] *  np.sqrt( ( df[f'{scenario_name}_cost_{scenario_name}_std' ]/df[f'{scenario_name}_cost_{scenario_name}_mean' ])**2 + (  df[f'gas_total_tonne_co2_saved_{years}yr_std']/  df[f'gas_total_tonne_co2_saved_{years}yr_mean'])**2 ) 

# #     return df