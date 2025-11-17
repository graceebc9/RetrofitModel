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

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltn

# Add custom module path
sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.validate import validate
# from src.RetrofitPostProcess import process_multiple_scenarios  # NO LONGER NEEDED
from src.GreedyAlgo import true_greedy_knapsack, plot_greedy_distribution_analysis
from src.RetrofitGreedy import run_greedy_algo 
# from src.RetrofitEquity import EQUITY_WEIGHTS, calculate_social_equity_score, calculate_scenario_persona_metrics
from src.RetrofitGreedyUtils import setup_logging
from src.RetrofitAnalysisUtils import load_data , prepare_data_for_postanalysis
from src.RetrofitGreedyPost import post_proc_greedy 
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
        BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy'
        # INPUT_FILES_PATH = '/Users/gracecolverd/Downloads/all/*.csv'
        INPUT_FILES_PATH='/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/*.csv'
        scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
 

        setting_name = 'lcoal'
        run_greedy_runs=True 
 
        budgets = [1_000_000, 10_000_000, 100_000_000]
        loft_probs = [0.65]
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]

        run_greedy_runs=True  
    else:
        BASE_DIR = os.getenv('BASE_DIR')
        INPUT_FILES_PATH = os.getenv('INPUT_FILES_PATH') 
        personas_path='/home/gb669/rds/hpc-work/energy_map/RetrofitModel/personas/NE_region_personas.csv'
        scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
        setting_name = 'v5'
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
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    

    YEARS = 5
    N_SIMULATIONS = 5000
 
    GAS_CARBON_FACTOR=0.18      
    ELEC_CARBON_FACTOR=0.19338  
    number=None
    if number:
        greedy_runs_folder = os.path.join(BASE_DIR, f'greedy_runs_{number}' , setting_name ) 
    else:
        greedy_runs_folder = os.path.join(BASE_DIR, 'greedy_runs', setting_name )
 
       

    if run_greedy_runs: 
        print("\n" + "="*80)
        print("GREEDY ALGORITHM ANALYSIS - UPDATED FOR NEW COLUMN FORMAT")
        print("="*80)
        
        # Load and concatenate input data
        print("\nLoading input data...")
        if number:
            res_df = load_data(INPUT_FILES_PATH, scenario_list, number )
        else:
            res_df = load_data(INPUT_FILES_PATH, scenario_list)
        
        print("\nLoading personas...")
        personas = load_personas(path=personas_path) 
        res_df = res_df.merge(personas, on='postcode', how='inner')
        print(f"After persona merge: {len(res_df)} rows")
        
        print('res df shape: ', res_df.shape)
        # UPDATED: Use prepare_data_for_greedy instead of process_multiple_scenarios
        print("\nPreparing data for greedy algorithm...")
        proc_df = prepare_data_for_postanalysis(
            res_df, 
            scenario_list, 
            YEARS, 
            GAS_CARBON_FACTOR, 
            ELEC_CARBON_FACTOR
        )
        print('proc df shape: ', proc_df.shape )
        
        
        # Filter data
        print("\nFiltering data...")
        # Option 2: Filter in place (destroys proc_df but saves even more memory)
        print("\nFiltering data...")
        # Filter in steps to avoid large boolean array
        mask1 = proc_df['premise_type'] != 'Domestic_outbuilding'
        proc_df = proc_df[mask1]
        del mask1
        gc.collect()

        mask2 = ~proc_df['premise_type'].isna()
        proc_df = proc_df[mask2]
        del mask2
        print(f"After filtering: {len(proc_df)} rows")
        df = proc_df  # Just a reference, no copy
        print(f"After filtering: {len(df)} rows")

      
 
        for budget in budgets:
            for prob_loft in loft_probs:
                for equity_factor in equity_factors: 
                  
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
                       
                    baseline_path = os.path.join(output_dir, f'baseline_selection.csv')
                    combined_path = os.path.join(output_dir, f'combined_results.csv')
                    combined_path = os.path.join(output_dir, f'equity_tracking_with_ranges.csv')

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
                        baseline_selection, combined_results = run_greedy_algo(
                            budget, 
                            prob_loft, 
                            df, 
                            scenario_list, 
                            summary_logger, 
                            detail_logger, 
                            equity_factor, 
                            output_dir,  
                        )
                        # check if empty 
                        if baseline_selection.empty:
                            raise Exception('Baselin results empty ')
                        if combined_results.empty: 
                            raise Exception('Baselin results empty ')
                        
                        # Save results to CSV
                        baseline_path = os.path.join(output_dir, f'baseline_selection.csv')
                        combined_path = os.path.join(output_dir, f'combined_results.csv')
                        
                        baseline_selection.to_csv(baseline_path, index=False)
                        combined_results.to_csv(combined_path, index=False)
                        
                        summary_logger.info(f"Baseline selection saved to: {baseline_path}")
                        summary_logger.info(f"Combined results saved to: {combined_path}")
                        
                        # Generate visualization
                        summary_logger.info("\nGenerating visualization...")
                        plot_greedy_distribution_analysis(
                            baseline_df=baseline_selection,
                            selected_df=combined_results,
                            scenario_name=f'£{budget:,} Budget - All Epistemic Runs',
                            output_dir=output_dir
                        )
                        
                        summary_logger.info("Analysis complete!")
                        print(f"✓ Results saved to: {output_dir}")
                        
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
            OUTPUT_PATH=os.path.join(BASE_DIR, f'greedy_vis_num{number}', f'loft_val{LOFT_VALUE}_budget{budgets}', setting_name)
        else:
            OUTPUT_PATH=os.path.join(BASE_DIR, 'greedy_vis', f'loft_val{LOFT_VALUE}_budget{budgets}', setting_name)
 
        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok = True )
        post_proc_greedy(budgets, equity_factors, LOFT_VALUE, greedy_runs_folder, OUTPUT_PATH )
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
 