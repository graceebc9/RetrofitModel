 
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

# Local application/library specific imports
# Note: Using sys.path.append is brittle. Consider making RetrofitModel
# an installable package (e.g., with `pip install -e .`)
 
from .RetrofitGreedyAnalysis import plot_greedy_compairosn_main, plot_carbon_by_persona, plot_count_by_group,  plot_metric_by_group
 
from .Sankey import run_sankey_greedy 


# ==============================================================================
# 2. DATA LOADING FUNCTION
# ==============================================================================

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
            # Construct scenario info
            dir_name = f'budget_{budget}__loft_{loft_val}__equity_{equity_weight}'
            scenario_label = f'budget_{budget}_equity_{equity_weight}'
            dir_path = os.path.join(base_path , dir_name) 
            
            # Define file paths
      
            results_file = os.path.join( dir_path , 'selected_projects.csv') 

            print(results_file) 
            try:
     
                
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

                # Create dataframe with separate columns
                equity_df_temp = pd.DataFrame({
                    'high_deprived_count': [counts.iloc[0]],
                    'high_deprived_pct': [pcts.iloc[0]],
                    'low_deprived_count': [counts.iloc[1]],
                    'low_deprived_pct': [pcts.iloc[1]],
                    'med_deprived_count': [counts.iloc[2]],
                    'med_deprived_pct': [pcts.iloc[2]]
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
                
            except FileNotFoundError:
                print(f"✗ Missing: budget=${budget/1e6:.1f}M, equity_weight={equity_weight}")
            except Exception as e:
                print(f"✗ Error: budget=${budget/1e6:.1f}M, equity_weight={equity_weight}: {str(e)}")

    # --- Combine all dataframes ---
    equity_df = pd.DataFrame()
    if equity_dfs:
        equity_df = pd.concat(equity_dfs, ignore_index=True)
        print(f"\n✓ Combined {len(equity_dfs)} equity tracking files")
        print(f"  Total equity tracking records: {len(equity_df):,}")
    else:
        print("\n✗ No equity tracking data loaded!")

    results_df = pd.DataFrame()
    if results_dfs:
        results_df = pd.concat(results_dfs, ignore_index=True)
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

def aggregate_results(df):
    """Aggregate metrics across epistemic runs."""
    if df.empty:
        print("Warning: Results dataframe is empty. Cannot aggregate.")
        return pd.DataFrame()
        
    # 1. Count number of buildings per scenario/epistemic run
    df['num_buildings'] = 1  # Each row is a building
    
    # # 2. Calculate totals per epistemic run
    df_summary = df.groupby(['scenario']).agg({
        'total_capex': ['mean', 'sum'],  # mean per bldg, sum = total spent
        'total_co2_saved_robust': 'sum',  # total CO2 saved across all bldgs
        'capex_per_net_ton': 'mean', # avg cost effectiveness
        'weighted_capex_per_net_ton': 'mean',  # avg weighted cost
        'remaining_funds': 'first',  # should be same for all in run
        'num_buildings': 'sum'  # total number of buildings retrofitted
    }).reset_index()
    
    # Flatten column names (e.g., ('cost_of_intervention_mean', 'mean') -> 'cost_of_intervention_mean_mean')
    df_summary = flatten_multiindex_cols(df_summary)
    
    # # 3. Now aggregate stats (mean, std) across all epistemic runs
    # agg_dict = {
    #     'total_capex_mean': ['mean', 'std'], # avg cost per building
    #     'total_capex_sum': ['mean', 'std'], # avg total budget spent
    #     'total_co2_saved_robust_sum': ['mean', 'std'], # avg total CO2 saved
    #     'capex_per_net_ton_mean': ['mean', 'std'], # avg cost effectiveness
    #     'weighted_capex_per_net_ton_mean': ['mean', 'std'], # avg weighted cost
    #     'remaining_funds_first': ['mean', 'std'], # avg remaining funds
    #     'num_buildings_sum': ['mean', 'std'] # avg number of buildings
    # }
    
    # aggregated = df_summary.groupby('scenario').agg(agg_dict).reset_index()
    
    # Fix: Flatten the final aggregated columns
    # aggregated = flatten_multiindex_cols(aggregated)
    
    return df_summary


def aggregate_equity(df, group_cols=['scenario']):
    """Aggregate equity metrics across epistemic runs."""
    if df.empty:
        print("Warning: Equity dataframe is empty. Cannot aggregate.")
        return pd.DataFrame()
        
    # agg_dict = {
    #     'vulnerable_pct': ['mean', 'std'],
    #     'equity_concentration': ['mean', 'std'],
    #     'deprived_count': ['mean', 'std'],
    #     'struggling_count': ['mean', 'std'],
    #     'lower middle_count': ['mean', 'std'],
    #     'upper middle_count': ['mean', 'std'],
    #     'affluent_count': ['mean', 'std'],
    #     'student_count': ['mean', 'std'],
    #     'deprived_pct': ['mean', 'std'],
    #     'struggling_pct': ['mean', 'std'],
    #     'lower middle_pct': ['mean', 'std'],
    #     'upper middle_pct': ['mean', 'std'],
    #     'affluent_pct': ['mean', 'std'],
    #     'student_pct': ['mean', 'std']
    # }
    agg_dict = {
        'high_deprived_pct': ['mean', 'std'],
        'equity_concentration': ['mean', 'std'],
        'high_deprived_count': ['mean', 'std'],
        'med_deprived_count': ['mean', 'std'],
        'med_deprived_pct': ['mean', 'std'],
        'low_deprived_count': ['mean', 'std'],
        'low_deprived_pct': ['mean', 'std'],
        
    }
    aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
    aggregated = flatten_multiindex_cols(aggregated)
    
    return aggregated


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def post_proc_greedy(BUDGETS, EQUITY_WEIGHTS, LOFT_VALUE, BASE_PATH, OUTPUT_PATH ):
    """
    Main function to run the data loading, aggregation, and plotting.
    """
    # --- 1. Load Data ---
    equity_df, results_df = load_data(
        budgets=BUDGETS,
        equity_weights=EQUITY_WEIGHTS,
        loft_val=LOFT_VALUE,
        base_path=BASE_PATH
    )
    
    if results_df.empty or equity_df.empty:
        print("Critical error: No data was loaded. Exiting.")
        return

    # --- 2. Aggregate Data ---
    results_agg = aggregate_results(results_df)
    # equity_agg = aggregate_equity(equity_df)
    equity_agg= equity_df

    if results_agg.empty or equity_agg.empty:
        print("Critical error: Aggregation failed. Exiting.")
        return
    
    results_df.to_csv('testresults.csv')
    # --- 3. Merge & Format ---
    comparison_df = results_agg.merge(equity_agg, on='scenario', how='left')
    print('comparison_df')
    print(comparison_df.columns.tolist())
    # Fix: Create scenario map with correct keys (matching the 'scenario' col)
    scenario_map = {
        f'budget_{b}_equity_{e}': f'${b/1e6:.0f}M, Equity={e}'
        for b in BUDGETS
        for e in EQUITY_WEIGHTS
    }
    
    # Add clean labels for plotting
    comparison_df['scenario_label'] = comparison_df['scenario'].map(scenario_map)
    
    # Fix: Sort the final dataframe based on parameter values for a logical order
    # Create temporary sorting keys from the 'scenario' string
    temp_map = {f'budget_{b}_equity_{e}': (e, b) for b in BUDGETS for e in EQUITY_WEIGHTS}
    sort_keys = comparison_df['scenario'].map(temp_map)
    
    if sort_keys.notna().all():
        comparison_df['sort_equity'] = sort_keys.str[0]
        comparison_df['sort_budget'] = sort_keys.str[1]
        comparison_df = comparison_df.sort_values(
            ['sort_equity', 'sort_budget']
        ).drop(columns=['sort_equity', 'sort_budget'])
    else:
        print("Warning: Could not sort dataframe. Scenario map mismatch.")

    # --- 4. Print Summary ---
    print("\n" + "=" * 80)
    print("EQUITY WEIGHTING COMPARISON SUMMARY (MEAN ACROSS EPISTEMIC RUNS)")
    print("=" * 80)
    # Define columns to display for a cleaner summary
    
    display_cols = [
        'scenario_label',
        'total_co2_saved_robust_sum',
        'num_buildings_sum_mean',
        'high_deprived_pct_mean',
        'equity_concentration_mean',
        'med_deprived_pct_mean',
        'low_deprived_pct_mean'
    ]
    
    # Filter for columns that actually exist in the final dataframe
    display_cols = [col for col in display_cols if col in comparison_df.columns]
    
    if display_cols:
        print(comparison_df[display_cols].to_string(index=False))
    else:
        print("Could not find key columns to display in summary.")
    print("\n")
    # save compariosn df 
    comparison_df.to_csv(f'{OUTPUT_PATH}/comparison_df.csv', index=False)

    # --- 5. Plot Results ---
    print(f"--- Generating plots in: {OUTPUT_PATH} ---")
    scenario_colors = plot_greedy_compairosn_main(comparison_df, output_dir=OUTPUT_PATH, y_axis_zero=True )

         
    plot_carbon_by_persona(results_df, scenario_colors, 
                           os.path.join(OUTPUT_PATH, "12_carbon_per_persona.png") 
                           , y_axis_zero=True)

    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, "12b_carbon_metapersona.png")  , 
                         value_col='total_co2_saved_robust',
                         metric_stat='sum',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Carbon Saved (Ton)', 
                         title='Distribution of Total Carbon Ton by Persona',
                         y_axis_zero=True)

    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, "13_cost_per_Ton_per_persona.png")  , 
                         value_col='capex_per_net_ton',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Cost per Ton Saved (£)',
                         title='Distribution of Total Cost per Ton by Persona',
                         y_axis_zero=True)
    
    plot_metric_by_group(results_df, scenario_colors, 
                         filename=os.path.join(OUTPUT_PATH, "14_cost_per_intervention_prr_persona.png")  , 
                         value_col='total_capex',
                         group_col='meta_socio_persona',
                         xlabel='Socio-economic Persona',
                         ylabel='Total Cost per Intervention (£)',
                         title='Distribution of Total Cost per Intervention by Persona',
                         y_axis_zero=True)
    
    plot_count_by_group(results_df, scenario_colors, 
                        filename=os.path.join(OUTPUT_PATH, "15_counts_persona.png"), 
                       group_col='meta_socio_persona',
                       xlabel='Socio-economic Persona',
                       ylabel='Number of Projects',
                       title='Distribution of Project Count by Persona',
                       y_axis_zero=True)

    print("✓ Plotting complete.")



 