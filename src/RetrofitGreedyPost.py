# ==============================================================================
# 0. IMPORTS
# ==============================================================================

# Standard library imports
import sys
from pathlib import Path
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from .RetrofitGreedyAnalysis import plot_greedy_comparison_main, plot_carbon_by_persona, plot_count_by_group,  plot_metric_by_group
 
from .Sankey import run_sankey_greedy 


# ==============================================================================
# 2. DATA LOADING FUNCTION
# ==============================================================================
million_factor=1_000_000

def load_data(budgets, equity_weights, loft_val, base_path):
    """
    Loads equity and results data from files based on configuration.
    """
    equity_dfs = []
    results_dfs = []
    loaded_combinations = 0
    total_combinations = len(budgets) * len(equity_weights)

    print("--- Starting Data Loading ---")

    for budget in budgets:
        for equity_weight in equity_weights:
            budg = str(budget/million_factor).replace('.0','') 
            # Construct scenario info
            dir_name = f'budget_{budg }M__loft_{loft_val}__equity_{equity_weight}'
            scenario_label = f'budget_{budg}M_equity_{equity_weight}'
            dir_path = os.path.join(base_path , dir_name) 
            
            # Define file paths
      
            results_file = os.path.join( dir_path , 'selected_projects.csv') 

            print(results_file) 
            # try:           
            # Load combined results data
            results_df_temp = pd.read_csv(results_file)
            # Rename 'scenario' column if it exists, as it's ambiguous
            if 'scenario' in results_df_temp.columns:
                results_df_temp = results_df_temp.rename(
                    columns={'scenario': 'intervention'}
                )
            results_df_temp['scenario'] = scenario_label
            results_df_temp['budget'] = budget
            results_df_temp['equity_weight'] = equity_weight
            results_dfs.append(results_df_temp)
            

            # Load equity tracking data
            # Get counts and percentages
            counts = results_df_temp.groupby('meta_socio_persona')['upn'].count()
            pcts = results_df_temp.groupby('meta_socio_persona')['upn'].count() / results_df_temp.groupby('meta_socio_persona')['upn'].count().sum()
            
            equity_df_temp = pd.DataFrame({
                'high_risk_count': [counts.get('high_risk', 0)],
                'high_risk_pct': [pcts.get('high_risk', 0.0)],
                
                'med_risk_count': [counts.get('med_risk', 0)],
                'med_risk_pct': [pcts.get('med_risk', 0.0)],
 
                'middle_risk_count': [counts.get('middle_risk', 0)],
                'middle_risk_pct': [pcts.get('middle_risk', 0.0)],
                
                'low_risk_count': [counts.get('low_risk', 0)],
                'low_risk_pct': [pcts.get('low_risk', 0.0)],

                 'v_low_risk_count': [counts.get('v_low_risk', 0)],
                'v_low_risk_pct': [pcts.get('v_low_risk', 0.0)],
 
            })
            equity_df_temp['scenario'] = scenario_label
            equity_df_temp['budget'] = budget
            equity_df_temp['equity_weight'] = equity_weight
            # Calculate concentration index (Herfindahl index: 0 = perfect equality, 1 = concentrated)            
            
            persona_counts = results_df_temp['meta_socio_persona'].value_counts()
            total = len(results_df_temp)
            proportions = persona_counts / total
            concentration = (proportions ** 2).sum()

            equity_df_temp['equity_concentration'] =  concentration
            
            equity_dfs.append(equity_df_temp)


            print(f"✓ Loaded: budget=${budget/1e6:.1f}M, equity_weight={equity_weight}")
            loaded_combinations += 1
                
            # except FileNotFoundError:
            #     print(f"✗ Missing: budget=${budget/1e6:.1f}M, equity_weight={equity_weight}")
            # except Exception as e:
            #     print(f"✗ Error: budget=${budget/1e6:.1f}M, equity_weight={equity_weight}: {str(e)}")

    # --- Combine all dataframes ---
    equity_df = pd.DataFrame()
    if equity_dfs:
        equity_df = pd.concat(equity_dfs, ignore_index=True)
        equity_df= equity_df.drop_duplicates() 
        print(f"\n✓ Combined {len(equity_dfs)} equity tracking files")
        print(f"  Total equity tracking records: {len(equity_df):,}")
    else:
        print("\n✗ No equity tracking data loaded!")

    results_df = pd.DataFrame()
    if results_dfs:
        results_df = pd.concat(results_dfs, ignore_index=True)
        results_df=results_df.drop_duplicates() 
        print(f"✓ Combined {len(results_dfs)} results files")
        print(f"  Total results records: {len(results_df):,}")
    else:
        print("\n✗ No results data loaded!")

    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)
    print(f"Budgets analyzed: {budgets}")
    print(f"Equity weights analyzed: {equity_weights}")
    print(f"Total combinations: {total_combinations}")
    print(f"Successfully loaded: {loaded_combinations} combinations")
    print("="*70 + "\n")

    return equity_df, results_df


# ==============================================================================
# 3. DATA AGGREGATION FUNCTIONS
# ==============================================================================

def flatten_multiindex_cols(df):
    """Flattens hierarchical columns (e.g., from .agg()) into a single level."""
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) and col[1] else col[0] 
                  for col in df.columns.values]
    return df



import numpy as np
import pandas as pd

def aggregate_results(df, rho=0.5):
    """
    Aggregates results using METHOD 2 (Ratio of Sums).
    
    Returns 3 Key Metrics, each with Uncorrelated, Correlated, and Partially Correlated (rho) Std:
    1. Efficiency (Capex per Net Ton) -> Calculated as Total Cost / Total Carbon
    2. Total Capex -> Sum of costs
    3. Total Carbon -> Sum of savings
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with project-level data
    rho : float, default=0.5
        Correlation parameter (0 = uncorrelated, 1 = fully correlated)
        Used to calculate partially correlated uncertainty
    """
    if df.empty:
        print("Warning: Results dataframe is empty. Cannot aggregate.")
        return pd.DataFrame()
        
    df = df.copy()

    # ---------------------------------------------------------
    # 1. PRE-CALCULATION
    # ---------------------------------------------------------
    # Create Variance columns (Sigma^2) for the Uncorrelated path.
    # (For Correlated path, we will just sum the Stds directly).
    df['var_total_capex'] = df['std_total_capex'] ** 2
    df['var_total_co2'] = df['std_total_co2_saved'] ** 2

    # ---------------------------------------------------------
    # 2. AGGREGATE (SUMMATION)
    # ---------------------------------------------------------
    agg_dict = {
        
        'upn':  'count', 
        # 'num_buildings': ('upn', 'count'),
        
        # Means (Simple Sums)
        'mean_total_capex': 'sum',
        'mean_total_co2_saved': 'sum',
        
        # Uncorrelated Uncertainty (Sum of Variances)
        'var_total_capex': 'sum',
        'var_total_co2': 'sum',
        
        # Correlated Uncertainty (Sum of Stds)
        'std_total_capex': 'sum',
        'std_total_co2_saved': 'sum'
    }

    df_agg = df.groupby('scenario').agg(agg_dict).reset_index()

    # ---------------------------------------------------------
    # 3. POST-CALCULATION
    # ---------------------------------------------------------
    
    # === A. TOTALS (Capex & CO2) ===
    
    # 1. Uncorrelated Std = Sqrt(Sum of Variances)
    df_agg['total_capex_std_uncorr'] = np.sqrt(df_agg['var_total_capex'])
    df_agg['total_co2_std_uncorr'] = np.sqrt(df_agg['var_total_co2'])
    
    # 2. Correlated Std = Sum of Stds (Rename for clarity)
    df_agg.rename(columns={
        'std_total_capex': 'total_capex_std_corr',
        'std_total_co2_saved': 'total_co2_std_corr'
    }, inplace=True)
    
    # 3. Partially Correlated Std = Interpolation between uncorrelated and correlated
    # Formula: σ_partial = √[(1-ρ)·σ_uncorr² + ρ·σ_corr²]
    df_agg['total_capex_std_partial'] = np.sqrt(
        (1 - rho) * df_agg['total_capex_std_uncorr']**2 + 
        rho * df_agg['total_capex_std_corr']**2
    )
    df_agg['total_co2_std_partial'] = np.sqrt(
        (1 - rho) * df_agg['total_co2_std_uncorr']**2 + 
        rho * df_agg['total_co2_std_corr']**2
    )
    
    # === B. EFFICIENCY (Capex per Net Ton) ===
    
    # 1. The Mean (Ratio of Sums)
    # "How much is the whole portfolio costing per ton?"
    df_agg['mean_capex_per_net_ton'] = df_agg['mean_total_capex'] / df_agg['mean_total_co2_saved']
    
    # 2. The Uncertainty (Error Propagation for Division)
    # Formula: Z = X/Y -> Sigma_Z = Z * Sqrt( (Sigma_X/X)^2 + (Sigma_Y/Y)^2 )
    
    # ... Uncorrelated Path
    cv_capex_u = df_agg['total_capex_std_uncorr'] / df_agg['mean_total_capex']
    cv_co2_u   = df_agg['total_co2_std_uncorr'] / df_agg['mean_total_co2_saved']
    
    df_agg['std_capex_per_net_ton_uncorr'] = (
        df_agg['mean_capex_per_net_ton'] * np.sqrt(cv_capex_u**2 + cv_co2_u**2)
    )

    # ... Correlated Path
    cv_capex_c = df_agg['total_capex_std_corr'] / df_agg['mean_total_capex']
    cv_co2_c   = df_agg['total_co2_std_corr'] / df_agg['mean_total_co2_saved']
    
    df_agg['std_capex_per_net_ton_corr'] = (
        df_agg['mean_capex_per_net_ton'] * np.sqrt(cv_capex_c**2 + cv_co2_c**2)
    )
    
    # ... Partially Correlated Path (using rho)
    cv_capex_p = df_agg['total_capex_std_partial'] / df_agg['mean_total_capex']
    cv_co2_p   = df_agg['total_co2_std_partial'] / df_agg['mean_total_co2_saved']
    
    df_agg['std_capex_per_net_ton_partial'] = (
        df_agg['mean_capex_per_net_ton'] * np.sqrt(cv_capex_p**2 + cv_co2_p**2)
    )

    # ---------------------------------------------------------
    # 4. CLEANUP
    # ---------------------------------------------------------
    # Select and order columns logically
    cols_order = [
        'scenario',
        'num_buildings_sum',
        
        # Efficiency - all three correlation assumptions
        'mean_capex_per_net_ton', 
        'std_capex_per_net_ton_uncorr', 
        'std_capex_per_net_ton_partial',
        'std_capex_per_net_ton_corr',
        
        # Total Capex - all three correlation assumptions
        'mean_total_capex', 
        'total_capex_std_uncorr',
        'total_capex_std_partial', 
        'total_capex_std_corr',
        
        # Total Carbon - all three correlation assumptions
        'mean_total_co2_saved', 
        'total_co2_std_uncorr',
        'total_co2_std_partial',
        'total_co2_std_corr'
    ]

    df_agg = df_agg.rename(columns={"upn": 'num_buildings_sum'})
    
    return df_agg[cols_order]

 

def aggregate_equity(df, group_cols=['scenario']):
    """Aggregate equity metrics across epistemic runs."""
    if df.empty:
        print("Warning: Equity dataframe is empty. Cannot aggregate.")
        return pd.DataFrame()
        

    agg_dict = {
        
        'equity_concentration': ['mean', 'std'],
        
        'high_risk_pct': ['mean', 'std'],
        'high_risk_count': ['mean', 'std'],
        
        'med_risk_count': ['mean', 'std'],
        'med_risk_pct': ['mean', 'std'],

         'middle_risk_pct': ['mean', 'std'],
        'middle_risk_count': ['mean', 'std'],
        
        'low_risk_count': ['mean', 'std'],
        'low_risk_pct': ['mean', 'std'],
        'v_low_risk_count': ['mean', 'std'],
        'v_low_risk_pct': ['mean', 'std'],
        
 
        
    }
    aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
    aggregated = flatten_multiindex_cols(aggregated)
    
    return aggregated


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def post_proc_greedy(BUDGETS, EQUITY_WEIGHTS, LOFT_VALUE, BASE_PATH, OUTPUT_PATH, RHO=0.5):
    """
    Main function to run the data loading, aggregation, and plotting.
    
    Parameters:
    -----------
    BUDGETS : list
        List of budget values to analyze
    EQUITY_WEIGHTS : list
        List of equity weights to analyze
    LOFT_VALUE : float
        Loft value parameter
    BASE_PATH : str
        Base path for input data
    OUTPUT_PATH : str
        Path for output files
    RHO : float, default=0.5
        Correlation parameter (0 = uncorrelated, 1 = fully correlated)
        Controls the partial correlation calculation
    """
    # --- 1. Load Data ---
    equity_df, results_df = load_data(
        budgets=BUDGETS,
        equity_weights=EQUITY_WEIGHTS,
        loft_val=LOFT_VALUE,
        base_path=BASE_PATH
    )
    print('results_df cols')
    print(results_df.columns.tolist() )
    if results_df.empty or equity_df.empty:
        print("Critical error: No data was loaded. Exiting.")
        return

    # --- 2. Aggregate Data with rho parameter ---
    results_agg = aggregate_results(results_df, rho=RHO)
    print(f'\nUsing correlation parameter ρ = {RHO}')
    print('results cols ')
    print(results_agg.columns.tolist() )
    
    equity_agg = equity_df

    if results_agg.empty or equity_agg.empty:
        print("Critical error: Aggregation failed. Exiting.")
        return
    
    print('results_df cols')
    print(results_df.columns.tolist() )

    results_df.to_csv('testresults.csv')
    # --- 3. Merge & Format ---
    comparison_df = results_agg.merge(equity_agg, on='scenario', how='left')

        
    scenario_map = {
        f'budget_{b/1e6:.0f}M_equity_{e}': f'£{b/1e6:.0f}M, Equity={e}'
        for b in BUDGETS
        for e in EQUITY_WEIGHTS
    }
        
    # Add clean labels for plotting
    comparison_df['scenario_label'] = comparison_df['scenario'].map(scenario_map)
    print('scenario_map')
    print(scenario_map)
    
    

    
    temp_map = {f'budget_{b/1e6:.0f}M_equity_{e}': (e, b) for b in BUDGETS for e in EQUITY_WEIGHTS}
    sort_keys = comparison_df['scenario'].map(temp_map)
    
    if sort_keys.notna().all():
        comparison_df['sort_equity'] = sort_keys.str[0]
        comparison_df['sort_budget'] = sort_keys.str[1]
        comparison_df = comparison_df.sort_values(
            ['sort_equity', 'sort_budget']
        ).drop(columns=['sort_equity', 'sort_budget'])
    else:
        # Check what's in sort_keys
        print("Sort keys:")
        print(sort_keys)

        # Check which ones are NaN
        print("\nNaN values:")
        print(sort_keys.isna())

        # See how many are missing
        print(f"\nTotal NaN count: {sort_keys.isna().sum()}")
        print("Warning: Could not sort dataframe. Scenario map mismatch.")

    # --- 4. Print Summary ---
    print("\n" + "=" * 80)
    print(f"EQUITY WEIGHTING COMPARISON SUMMARY (ρ = {RHO})")
    print("=" * 80)
    # Define columns to display for a cleaner summary
    
    display_cols = [
        'scenario_label',
        'mean_total_co2_saved',
        'total_co2_std_uncorr',
        'total_co2_std_partial',
        'total_co2_std_corr',
        'num_buildings_sum',
        'mean_capex_per_net_ton', 
        'std_capex_per_net_ton_uncorr',
        'std_capex_per_net_ton_partial',
        'std_capex_per_net_ton_corr',
        'high_risk_pct_mean',
        'equity_concentration_mean',
        'med_risk_pct_mean',
        'middle_risk_pct_mean',
        'low_risk_pct_mean',
        'v_low_risk_pct_mean'
    ]
    
    # Filter for columns that actually exist in the final dataframe
    display_cols = [col for col in display_cols if col in comparison_df.columns]
    
    if display_cols:
        print(comparison_df[display_cols].to_string(index=False))
    else:
        print("Could not find key columns to display in summary.")
    print("\n")
    # save comparison df 
    comparison_df.to_csv(f'{OUTPUT_PATH}/comparison_df_rho_{RHO}.csv', index=False)

    # --- 5. Plot Results ---
    print(f"--- Generating plots in: {OUTPUT_PATH} ---")
    scenario_colors = plot_greedy_comparison_main(comparison_df, output_dir=OUTPUT_PATH, y_axis_zero=True, loft_val=LOFT_VALUE, rho=RHO)

         
    plot_carbon_by_persona(results_df, scenario_colors, 
                           os.path.join(OUTPUT_PATH, f"12_carbon_per_persona_loft_{LOFT_VALUE}.png"), 
                           y_axis_zero=True)

    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, f"12b_carbon_metapersona__loft_{LOFT_VALUE}.png"), 
                         value_col='mean_total_co2_saved',
                         metric_stat='sum',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Carbon Saved (Ton)', 
                        #  title='Distribution of Total Carbon Ton by Persona',
                         y_axis_zero=True)

    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, f"13_mean_cost_per_Ton_per_persona_loft_{LOFT_VALUE}.png"), 
                         value_col='mean_capex_per_net_ton',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Cost per Ton Saved (£)',
                        #  title='Distribution of Total Cost per Ton by Persona',
                         y_axis_zero=True)
    
    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, f"13BB_sigma_cost_per_Ton_per_persona_loft_{LOFT_VALUE}.png"), 
                         value_col='capex_per_net_ton_sigma',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Cost per Ton Saved (£)',
                        #  title='Distribution of Total Cost per Ton by Persona',
                         y_axis_zero=True)
    
    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, f"14_cost_per_intervention_prr_persona__loft_{LOFT_VALUE}.png"), 
                         value_col='mean_total_capex',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Cost per Intervention (£)',
                        #  title='Distribution of Total Cost per Intervention by Persona',
                         y_axis_zero=True)
    
    plot_count_by_group(results_df, scenario_colors, 
                        filename=os.path.join(OUTPUT_PATH, f"15_counts_persona__loft_{LOFT_VALUE}.png"), 
                       group_col='meta_socio_persona',
                       xlabel='Socio-economic Persona',
                       ylabel='Number of Projects',
                    #    title='Distribution of Project Count by Persona',
                       y_axis_zero=True)

    print("✓ Plotting complete.")