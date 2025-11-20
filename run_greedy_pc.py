"""
Greedy Algorithm Analysis for Retrofit Scenarios (Updated for new column format)
Processes multiple epistemic runs and selects optimal retrofit projects within budget constraints.

Key Changes:
- Works with new column format (gas_saving_abs_kwh, etc.)
- No longer uses process_multiple_scenarios() - data already processed
- Adds CO2 conversion from kWh savings
"""

import os
import sys
import glob


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add custom module path
sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.validate import validate
# from src.RetrofitPostProcess import process_multiple_scenarios  # NO LONGER NEEDED
from src.GreedyAlgo import true_greedy_knapsack, plot_greedy_distribution_analysis
# from src.RetrofitGreedy import run_greedy_algo 
from src.RetrofitGreedy_EPC import run_greedy_algo_epc
# from src.RetrofitEquity import EQUITY_WEIGHTS, calculate_social_equity_score, calculate_scenario_persona_metrics
from src.RetrofitGreedyUtils import setup_logging
from src.RetrofitAnalysisUtils import load_data , prepare_data_for_postanalysis
from src.RetrofitGreedyPost import post_proc_greedy 
from src.GreedyEpcVis import run_mode_comparison
# ============================================================================
# DATA LOADING (UPDATED FOR NEW FORMAT)
# ============================================================================
 

def load_personas(path):
    """Load persona/demographic data."""
    personas = pd.read_csv(path)
    return personas

# ============================================================================
# MAIN EXECUTION
# ============================================================================
from src.utils import is_running_on_hpc 

def main():
    """
    Main execution function for greedy algorithm analysis.
    """
    # Configuration
    running_locally = not is_running_on_hpc()
    if running_locally:
        personas_path='/Users/gracecolverd/RetrofitModel/NE_region_personas.csv'
        BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy_epc'
        # INPUT_FILES_PATH = '/Users/gracecolverd/Downloads/all/*.csv'
        INPUT_FILES_PATH='/Users/gracecolverd/RetrofitModel/test/new_log_epc/*.csv'
        scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
        run_greedy_runs=True   
        budgets = [1_000_000, 10_000_000, 100_000_000]
        loft_probs = [0.65, 0.95]
        equity_factor =  0.8 
        number = 5 

 
    else:
        BASE_DIR = os.getenv('BASE_DIR')
        INPUT_FILES_PATH = os.getenv('INPUT_FILES_PATH') 
        personas_path='/home/gb669/rds/hpc-work/energy_map/RetrofitModel/personas/NE_region_personas.csv'
        scenario_list =  ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
        
        run_g_yn=os.getenv('RUN_GREEDY_RUNS_YN') 
        
        if run_g_yn=='N':
            run_greedy_runs=False
        else:
            run_greedy_runs=True 
        BUDGET_SETTING = os.getenv('BUDGET_SETTING')
        
        if BUDGET_SETTING=='1':
            budgets=[1_000_000]
        elif BUDGET_SETTING=='2':
            budgets=[10_000_000]
        elif BUDGET_SETTING=='3':
            budgets=[50_000_000]
        elif BUDGET_SETTING=='4':
            budgets=[80_000_000]
        elif BUDGET_SETTING=='5':
            budgets=[100_000_000]
            
        else:
            budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
            
        loft_setting = os.getenv('loft_setting') 
        if loft_setting=='1':
            loft_probs = [0.65 ]
        elif loft_setting=='2':
            loft_probs = [0.95 ] 
        else: 
            loft_probs = [0.65, 0.95] 
 
        equity_factor = float( os.getenv('equity_factor') ) 
 
        number=None  

    YEARS = 5
    N_SIMULATIONS = 5000
 
    # Carbon factors (kg CO2/kWh)
    GAS_CARBON_FACTOR=0.18      
    ELEC_CARBON_FACTOR=0.19338  
    targeted_or_epc='targeted'
    
    
    if number:
        greedy_runs_folder = os.path.join(BASE_DIR, f'greedy_combo_ef{equity_factor}_{number}' , 'runs') 
    else:
        greedy_runs_folder = os.path.join(BASE_DIR, f'greedy_combo_ef{equity_factor}', 'runs')
 
    
       
    epc=True 

    if run_greedy_runs: 
        print("\n" + "="*80)
        print("GREEDY ALGORITHM ANALYSIS - UPDATED FOR NEW COLUMN FORMAT")
        print("="*80)
        
        # Load and concatenate input data
        print("\nLoading input data...")
        if  number:
            res_df = load_data(INPUT_FILES_PATH, scenario_list, epc=epc, number=number)
        else:    
            res_df = load_data(INPUT_FILES_PATH, scenario_list, epc=epc)
        epc_col = 'CURRENT_ENERGY_RATING'   
        if epc:
            if epc_col not in res_df.columns:
                raise Exception('Log missing epc col')
            
        print("\nLoading personas...")
        personas = load_personas(path=personas_path) 
        res_df = res_df.merge(personas, on='postcode', how='inner')
        print(f"After persona merge: {len(res_df)} rows")
        print(res_df.columns.tolist() )

        # UPDATED: Use prepare_data_for_greedy instead of process_multiple_scenarios
        print("\nPreparing data for greedy algorithm...")
        proc_df = prepare_data_for_postanalysis(
            res_df, 
            scenario_list, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
  
        
        # Filter data
        # Option 2: Filter in place (destroys proc_df but saves even more memory)
        proc_df = proc_df[
            (proc_df['premise_type'] != 'Domestic_outbuilding') & 
            (~proc_df['premise_type'].isna())
        ]
        df = proc_df 

        print(f"After filtering: {len(df)} rows")
        if epc:
            if epc_col not in df.columns:
                raise Exception('Log missing epc col dfdf')
        
        for budget in budgets:
            for prob_loft in loft_probs:
                # Create output directory
                output_dir = os.path.join(
                    greedy_runs_folder, 
                    f'budget_{budget}__loft_{prob_loft}__equity_{equity_factor}'
                )
                os.makedirs(output_dir, exist_ok=True)
                
                # Set up logging
                summary_logger, detail_logger = setup_logging(
                    output_dir, budget, prob_loft, equity_factor
                )
                
                summary_logger.info(
                    f'Starting analysis: Budget £{budget:,}, '
                    f'Loft Probability {prob_loft}, '
                    f'Equity Factor {equity_factor}'
                )
                    
     
                baseline_path = os.path.join(output_dir, f'all_modes_equity_tracking.csv')
                combined_path = os.path.join(output_dir, f'all_modes_selected_projects.csv')

                if os.path.exists(baseline_path) and os.path.exists(combined_path):
                    print(f"✓ Results already exist for this configuration, skipping...")
                    print(f"  Existing files found in: {output_dir}")
                    continue
                
                print(f"\n{'='*80}")
                print(f"Starting analysis:")
                print(f"  Budget: £{budget:,}")
                print(f"  Loft Probability: {prob_loft}")
                print(f"  Equity Factor: {equity_factor}")
                print(f"{'='*80}")
            
                # Run greedy algorithm
                try:
                        combined_results = run_greedy_algo_epc(
                        budget, 
                        prob_loft, 
                        df, 
                        scenario_list, 
                        summary_logger, 
                        detail_logger, 
                        equity_factor, 
                        output_dir,  
                        
                    )
             
                    
                except Exception as e:
                    summary_logger.error(f"Error in analysis: {e}")
                    print(f"✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    # Clear handlers to avoid duplicate logging in next iteration
                    summary_logger.handlers.clear()
                    detail_logger.handlers.clear()
        
        print("\n" + "="*80)
        print("Greedy RUNS  COMPLETE!")
        print("="*80)
    else:
        print('Set to skip runs. goin to wrap up ')
    print("\n" + "="*80)
    print("Start post proces ") 
    print("="*80)

    for LOFT_VALUE in loft_probs:
        if number:
            OUTPUT_PATH=os.path.join(BASE_DIR, f'greedy_combo_ef{equity_factor}_{number}', 'vis',  f'loft_val{LOFT_VALUE}')
        else:
            OUTPUT_PATH=os.path.join(BASE_DIR, f'greedy_combo_ef{equity_factor}', 'vis',  f'loft_val{LOFT_VALUE}')
        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok = True )
        for budget in budgets:
            inp_dir = os.path.join(
                        greedy_runs_folder, 
                        f'budget_{budget}__loft_{LOFT_VALUE}__equity_{equity_factor}'
                    )
            op_folder = os.path.join(OUTPUT_PATH  , f'output_{budget/1_000_000}m_budget' ) 
            df_projects_50m = pd.read_csv(os.path.join(inp_dir, 'all_modes_selected_projects.csv') ) 
            df_equity_50m = pd.read_csv(os.path.join(inp_dir, 'all_modes_equity_tracking.csv') ) 
            run_mode_comparison(df_projects_50m, df_equity_50m,op_folder  )
            # post_proc_greedy(budgets, equity_factors, LOFT_VALUE, greedy_runs_folder, OUTPUT_PATH , targeted_or_epc)
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

    # ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
# run_mode_comparison(df_projects_10m, df_equity_10m, 'output_10m_budget')
# run_mode_comparison(df_projects_50m, df_equity_50m, 'output_50m_budget')