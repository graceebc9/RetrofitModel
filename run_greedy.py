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
from src.EPCAlgo import select_epc_algo 
from src.GreedyEpcVis import run_epc_vis 


# ============================================================================
# MAIN EXECUTION
# ============================================================================

milion_factor = 1_000_000

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

RISK_PENALTY_SIGMA = float(os.getenv('SIGMA')  )  

def main():
    """
    Main execution function for greedy algorithm analysis using pre-processed data.
    """
    # Configuration
    running_locally = not is_running_on_hpc()
    

    epc_yn = os.getenv('EPC_YN')
    if epc_yn =='Y':
        print('Runnig greedy for EPC' ) 
        epc_run = True 
    else:
        epc_run = False 
        print('Runnig greedy for normal' ) 

    run_g_yn=os.getenv('RUN_GREEDY_RUNS_YN') 

    if run_g_yn == 'N':
        run_greedy_runs = False
    else:
        run_greedy_runs = True 


    if running_locally:
        BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy'
        
        
        setting_name = 'lcoal'
        # run_greedy_runs=True  
        # budgets = [1_000_000, 10_000_000, 100_000_000]
        budgets = [ 1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000] 
        budgets = [ 1_000_000, 10_000_000, 50_000_000, 80_000_000,  100_000_000]
        # budgets = [  50_000_000,80_000_000]
        loft_probs = [0.95, 0.65 ]
        # loft_probs = [0.65, 0.95]
        
        
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1 , 1.2,1.4 ]
        # equity_factors = [ 1.4  ]
        # equity_factors=[0.8] 

        # epc_run = True  
        epc_yn = os.getenv('EPC_YN')
        if epc_run:
            INPUT_FILES_PATH=f'/Volumes/T9/2025_10_RetrofitModel/3_optimiseD_iroiities/epc/risk_sigma_{RISK_PENALTY_SIGMA}__processed_best_only/*'
            # INPUT_FILES_PATH= '/Users/gracecolverd/Downloads/risk_sigma1_epc__processed_best_only/*csv'
            BASE_DIR=f'/Volumes/T9/2025_10_RetrofitModel/4_gredy_epc/risk_{RISK_PENALTY_SIGMA}/'
            # BASE_DIR = '/Users/gracecolverd/RetrofitModel/3_greedy_optimisation/epc'
        else:
            INPUT_FILES_PATH=f'/Volumes/T9/2025_10_RetrofitModel/3_optimiseD_iroiities/risk_sigma_{RISK_PENALTY_SIGMA}__processed_best_only/*.csv'
            BASE_DIR=f'/Volumes/T9/2025_10_RetrofitModel/4_gredy/risk_{RISK_PENALTY_SIGMA}/'

    else:
        BASE_DIR = os.getenv('BASE_DIR')
        sigma_value = float(os.getenv('SIGMA')) 
        
        if epc_run:
            INPUT_FILES_PATH=f'/home/gb669/rds/hpc-work/energy_map/RetrofitModel/2_optimized_priorities_epc/risk_sigma_{sigma_value}/processed_best_only/*'
            BASE_DIR=f'/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v8/NE/epc/risk_sigma_{sigma_value}'
        else:
            INPUT_FILES_PATH=f'/home/gb669/rds/hpc-work/energy_map/RetrofitModel/2_optimized_priorities/risk_sigma_{sigma_value}/processed_best_only/*'
            BASE_DIR=f'/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_greedy_optimisation/v8/NE/all_domestic/risk_sigma_{sigma_value}'

        print(f'Starting {INPUT_FILES_PATH}') 
        setting_name = 'v8'
        run_g_yn=os.getenv('RUN_GREEDY_RUNS_YN') 
        
        if run_g_yn == 'N':
            run_greedy_runs = False
        else:
            run_greedy_runs = True 

        budgets = [1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000]
        
        loft_probs = [ 0.65] 
        
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1 , 1.2, 1.4, 1.6, 1.8 ,2  ]
        
        equity_factor=float(os.getenv('EQUITY_FACTOR'))
        equity_factors = [equity_factor]
        
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
            print('res dsf shap0e: ' )
            print(res_df.shape)
            print('num upns')
            print(len(res_df.upn.unique() )  ) 
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
                million_budget = str(budget/milion_factor).replace('.0','')
                for equity_factor in equity_factors: 
                    print(f'Million biudget: {million_budget}')
                    output_dir = os.path.join(greedy_runs_folder, f'budget_{million_budget}M__loft_{prob_loft}__equity_{equity_factor}'  )
                    os.makedirs(output_dir, exist_ok=True)
                    print(f'saving to {output_dir}')
                    # Set up logging
                    summary_logger, detail_logger = setup_logging(
                        output_dir, budget, prob_loft, equity_factor
                    )
                    print(detail_logger)
                    
                    summary_logger.info(
                        f'Starting analysis: Budget £{budget:,}, '
                        f'Loft Probability {prob_loft}, '
                        f'Equity Factor {equity_factor}'
                    )
                        
                    baseline_path = os.path.join(output_dir, f'baseline_selection.csv')
                    
         
                    
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
                        print(f'Million biudget: {million_budget}')
                        selected_projects_df, remaining_funds = true_greedy_knapsack(
                            df_knapsack=baseline_selection,
                            budget=budget,
                            cost_column='total_capex',
                            efficiency_column='weighted_capex_per_net_ton' ,
                            carbon_col='total_co2_saved',
                            logger=detail_logger, 
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
                        print(f'budget: {budget}')
                        print(f'Million biudget: {million_budget}')
                        # Generate visualization
                        summary_logger.info("\nGenerating visualization...")
                        
                        plot_greedy_distribution_analysis(
                            baseline_df=baseline_selection,
                            selected_df=selected_projects_df,
                            scenario_name=f'budget_{million_budget}M__loft{prob_loft}__equity{equity_factor}',
                            output_dir=output_dir,
                            
                        )
                        
                        summary_logger.info("Analysis complete!")
                        print(f"✓ Results saved to: {output_dir}")

                        if epc_run:
                            epc_random_path = os.path.join(output_dir, f'epc_random_selection.csv')
                            epc_random_selected_df, epc_random_remaining_budget = select_epc_algo( 
                                                                            df_knapsack=baseline_selection,
                                                                        budget=budget,
                                                                        cost_column='total_capex',
                                                                        efficiency_column='weighted_capex_per_net_ton' ,
                                                                        carbon_col='total_co2_saved', 
                                                                        logger=detail_logger)
                            
                            epc_random_selected_df['remaining_funds'] = epc_random_remaining_budget
                            if epc_random_selected_df.empty:
                                detail_logger.info('EPC selection empty')
                                raise Exception('EPC selection empty')
                            
                            epc_random_selected_df.to_csv(epc_random_path, index=False) 
                            summary_logger.info(f"EPC RAndom selection saved to: {epc_random_path}")

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
    post_proc_meta=False 
    post_proc_epc=True 

    if post_proc_meta: 
        for LOFT_VALUE in loft_probs:
            if number:
                OUTPUT_PATH = os.path.join(BASE_DIR, f'greedy_vis_num{number}', f'loft_val{LOFT_VALUE}_budget{budgets}', setting_name)
            else:
                OUTPUT_PATH = os.path.join(BASE_DIR, 'greedy_vis', f'loft_val{LOFT_VALUE}_budget{budgets}', setting_name)

            # Ensure output directory exists
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            post_proc_greedy(budgets, equity_factors, LOFT_VALUE, greedy_runs_folder, OUTPUT_PATH)
        
    if post_proc_epc:
        if epc_run:
            for LOFT_VALUE in loft_probs:
                for budget in budgets:
                    million_budget= budget / milion_factor
                    for equity_factor in equity_factors:
                        
                        run_epc_vis(greedy_runs_folder, os.path.join(BASE_DIR, 'greedy_vis_epc') , million_budget  , LOFT_VALUE , equity_factor    )
            
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()