"""
Greedy Algorithm Analysis for Retrofit Scenarios (Updated for Pre-Processed Chunks)
Processes pre-calculated retrofit project files (chunks) and selects optimal projects within budget constraints.

Key Changes:
- Loads pre-processed CSV chunks instead of raw simulation logs.
- Skips redundant `prepare_data_for_postanalysis`.
- Preserves logic for Equity, Budget, and Risk.
"""

import os
import sys
import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add custom module path
sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.validate import validate
from src.GreedyAlgo import true_greedy_knapsack, plot_greedy_distribution_analysis
from src.RetrofitGreedy import run_greedy_algo 
from src.RetrofitGreedyUtils import setup_logging
# from src.RetrofitAnalysisUtils import load_data , prepare_data_for_postanalysis
from src.RetrofitGreedyPost import post_proc_greedy 
from src.personas import load_personas  
from src.RetrofitEquity import EQUITY_WEIGHTS  
from src.utils import is_running_on_hpc 
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_data_simple(files):
    res = [] 
    for f in files:
        df = pd.read_csv(f)
        res.append(df)
    return pd.concat(res)


def add_equity_weight(scenario_df, equity_factor , capex_col='capex_per_net_ton' ): 
    EQUITY_WEIGHT_COL = 'equity_weight'
    PERSONA_COL = 'meta_socio_persona'
    scenario_df[EQUITY_WEIGHT_COL] = scenario_df[PERSONA_COL].map(EQUITY_WEIGHTS)
    scenario_df['weighted_capex_per_net_ton'] = (
        scenario_df[capex_col] * (1 + (scenario_df[EQUITY_WEIGHT_COL] - 1) * equity_factor)
    )
    print('equity weight added')
    return scenario_df


def main():
    """
    Main execution function for greedy algorithm analysis using pre-processed data.
    """
    # Configuration
    running_locally = not is_running_on_hpc()
    
    if running_locally:
        
        
        BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy'
        
        INPUT_FILES_PATH='/Users/gracecolverd/RetrofitModel/optimized_priorities/processed_best_only/*.csv'
        
        # scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
 

        setting_name = 'lcoal'
        run_greedy_runs=True 
 
        budgets = [1_000_000, 10_000_000, 100_000_000]
        loft_probs = [0.65, 0.95]
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]

        run_greedy_runs=False   
    else:
        BASE_DIR = os.getenv('BASE_DIR')
        INPUT_FILES_PATH = os.getenv('INPUT_FILES_PATH') 
        
        
        setting_name = 'v7'
        run_g_yn=os.getenv('RUN_GREEDY_RUNS_YN') 
        
        if run_g_yn == 'N':
            run_greedy_runs = False
        else:
            run_greedy_runs = True 
            
        BUDGET_SETTING = os.getenv('BUDGET_SETTING' )
        
        if BUDGET_SETTING == '1':
            budgets = [1_000_000]
        elif BUDGET_SETTING == '2':
            budgets = [10_000_000]
        elif BUDGET_SETTING == '3':
            budgets = [50_000_000]
        elif BUDGET_SETTING == '4':
            budgets = [80_000_000]
        elif BUDGET_SETTING == '5':
            budgets = [100_000_000]
        else:
            budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
        
        budgets = [1_000_000] 
        
        loft_setting = os.getenv('loft_setting')
        
        if loft_setting == '1':
            loft_probs = [0.65]
        elif loft_setting == '2':
            loft_probs = [0.95] 
        else: 
            loft_probs = [0.65, 0.95] 
            
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]
        equity_factors=[1]
        number = os.getenv("NUMBER")
        try: 
            number=int(number)
            print(number ) 
        except:
            number= None 
    

    # YEARS = 5
    # N_SIMULATIONS = 5000
 
    # GAS_CARBON_FACTOR=0.18      
    # ELEC_CARBON_FACTOR=0.19338  
    
    input_files = glob.glob(INPUT_FILES_PATH)
    number=None
    if number:
        greedy_runs_folder = os.path.join(BASE_DIR, f'greedy_runs_{number}', setting_name) 
    else:
        greedy_runs_folder = os.path.join(BASE_DIR, 'greedy_runs', setting_name )
 
       
    print("\n" + "="*80)
    print("GREEDY ALGORITHM ANALYSIS - UPDATED FOR NEW COLUMN FORMAT")
    print("="*80)
    if run_greedy_runs: 

        for prob_loft in loft_probs:
            files_to_use =  [x for x in input_files if f'loft_{prob_loft}' in x ]
            print(f'Found {len(files_to_use)} files with loft prob {prob_loft}')

            
            # Load and concatenate input data
            print("\nLoading input data...")
            if number:
                res_df = load_data_simple(files_to_use, number )
            else:
                res_df = load_data_simple(files_to_use )
            
            print("\nLoading personas...")
            personas = load_personas( ) 
            df = res_df.merge(personas, on='postcode', how='inner')
            print(f"After persona merge: {len(df)} rows")
    
    
            
            
            
            print("\nFiltering data...")
            # Filter in steps to avoid large boolean array
            mask1 = df['premise_type'] != 'Domestic_outbuilding'
            df = df[mask1]
            del mask1
            gc.collect()

            mask2 = ~df['premise_type'].isna()
            df = df[mask2]
            del mask2
            
            
            print(f"After filtering: {len(df)} rows")
            print('cols with personas')
            print(df.columns.tolist() )
    
            for budget in budgets:
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
                    
                    # combined_path = os.path.join(output_dir, f'equity_tracking_with_ranges.csv')

                    # if os.path.exists(baseline_path) and os.path.exists(combined_path):
                    #     print(f"✓ Results already exist for this configuration, skipping...")
                    #     print(f"  Existing files found in: {output_dir}")
                    #     continue
                    
                    print(f"\n{'='*80}")
                    print(f"Starting analysis:")
                    print(f"  Budget: £{budget:,}")
                    print(f"  Loft Probability: {prob_loft}")
                    print(f"  Equity Factor: {equity_factor}")
                    print(f"{'='*80}")
                
                    df = add_equity_weight(df, equity_factor , capex_col='capex_per_net_ton' )

                    # Run greedy algorithm
                    baseline_selection = df 


                    try:
                        selected_projects_df, remaining_funds = true_greedy_knapsack(
                            df_knapsack=baseline_selection,
                            budget=budget,
                            cost_column='total_capex',
                            efficiency_column='weighted_capex_per_net_ton' ,
                            carbon_col='total_co2_saved_robust'
                        )
                        selected_projects_df['remaining_funds'] = remaining_funds

                        if baseline_selection.empty:
                            raise Exception('Baselin results empty ')
                   
                        # Save results to CSV
                        baseline_path = os.path.join(output_dir, f'baseline_selection.csv')
                        selected_path = os.path.join(output_dir, f'selected_projects.csv')
                        
                        baseline_selection.to_csv(baseline_path, index=False)
                        selected_projects_df.to_csv(selected_path, index=False)
                        
                        summary_logger.info(f"Baseline selection saved to: {baseline_path}")
                        summary_logger.info(f"Selected projects results saved to: {selected_path}")
                        
                        # Generate visualization
                        summary_logger.info("\nGenerating visualization...")
                        plot_greedy_distribution_analysis(
                            baseline_df=baseline_selection,
                            selected_df=selected_projects_df,
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
        print('Set to skip runs. going to wrap up ')

    # ------------------------------------------------------------------------
    # PART 3: POST PROCESSING (VISUALIZATION)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Start post process ") 
    print("="*80)
    
    for LOFT_VALUE in loft_probs:
        if number:
            OUTPUT_PATH = os.path.join(BASE_DIR, f'greedy_vis_num{number}', f'loft_val{LOFT_VALUE}_budget{budgets}', setting_name)
        else:
            OUTPUT_PATH = os.path.join(BASE_DIR, 'greedy_vis', f'loft_val{LOFT_VALUE}_budget{budgets}', setting_name)

        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        post_proc_greedy(budgets, equity_factors, LOFT_VALUE, greedy_runs_folder, OUTPUT_PATH)

    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()