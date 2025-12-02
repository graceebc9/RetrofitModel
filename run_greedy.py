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


from src.RetrofitGreedyPost import post_proc_greedy 
from src.personas import load_personas  
from src.utils import is_running_on_hpc 

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for greedy algorithm analysis using pre-processed data.
    """
    # Configuration
    running_locally = not is_running_on_hpc()
    
    if running_locally:
       
        BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy'
        # UPDATED: Path now points to the processed chunks folder
        PROCESSED_DATA_PATH = '/Users/gracecolverd/RetrofitModel/optimized_priorities/processed_chunks/*.csv'
        
        scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
        setting_name = 'local'
        run_greedy_runs = True 
        budgets = [1_000_000, 10_000_000, 100_000_000]
        loft_probs = [0.65]
        equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]

    else:
        BASE_DIR = os.getenv('BASE_DIR')
        
        PROCESSED_DATA_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/optimized_priorities/processed_chunks/*'
        
        scenario_list = ['joint_heat_loft_decay','joint_heat_wall_decay','wall_installation', 'join_heat_ins_decay', 'heat_pump_only', 'loft_installation']
        setting_name = 'v7'
        run_g_yn = os.getenv('RUN_GREEDY_RUNS_YN') 
        
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
    
    # These parameters are implicitly handled in the pre-processing, 
    # but kept here if needed for calculations.
    YEARS = 5
    N_SIMULATIONS = 5000
    GAS_CARBON_FACTOR = 0.18       
    ELEC_CARBON_FACTOR = 0.19338  
    
    
    if number:
        greedy_runs_folder = os.path.join(BASE_DIR, f'greedy_runs_{number}', setting_name) 
    else:
        greedy_runs_folder = os.path.join(BASE_DIR, 'greedy_runs', setting_name)

    # ------------------------------------------------------------------------
    # PART 1: LOAD PRE-PROCESSED DATA
    # ------------------------------------------------------------------------
    if run_greedy_runs: 
        print("\n" + "="*80)
        print("GREEDY ALGORITHM ANALYSIS - USING PRE-PROCESSED CHUNKS")
        print("="*80)
        
        # 1. Load Data from Chunks (Replacing load_data + prepare_data)
        print(f"\nLoading processed chunks from: {PROCESSED_DATA_PATH}")
        chunk_files = glob.glob(PROCESSED_DATA_PATH)
        
        if not chunk_files:
            raise FileNotFoundError(f"No processed chunk files found at {PROCESSED_DATA_PATH}")
            
        df_list = []
        for f in chunk_files:
            # Read only essential columns if possible to save memory
            # Assuming chunks have: upn, intervention, cost_robust, cost_mean, postcode (if preserved)
            try:
                chunk = pd.read_csv(f)
                df_list.append(chunk)
            except Exception as e:
                print(f"Warning: Failed to read {f}: {e}")

        proc_df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(proc_df)} rows from {len(chunk_files)} files.")
        
        # Free memory
        del df_list
        gc.collect()

        # 2. Merge Personas
        # Note: This relies on 'postcode' being present in the processed chunks.
        # If processed chunks only have UPN, you may need to join with a master building file first.
        print("\nLoading personas...")
        personas = load_personas() 
        
        if 'postcode' in proc_df.columns:
            proc_df = proc_df.merge(personas, on='postcode', how='inner')
            print(f"After persona merge: {len(proc_df)} rows")
        else:
            print("WARNING: 'postcode' column missing from processed chunks. Cannot merge Personas.")
            print("Equity analysis relying on persona data will likely fail.")

        # 3. Apply Filters
        print("\nFiltering data...")
        # Check if premise_type exists (it might have been filtered out during pre-processing)
        if 'premise_type' in proc_df.columns:
            mask1 = proc_df['premise_type'] != 'Domestic_outbuilding'
            proc_df = proc_df[mask1]
            mask2 = ~proc_df['premise_type'].isna()
            proc_df = proc_df[mask2]
            print(f"After filtering premise types: {len(proc_df)} rows")
        else:
            print("Note: 'premise_type' column not found. Assuming pre-processing already filtered buildings.")

        # Create reference
        df = proc_df 
        print(f"Final dataset size ready for Greedy: {len(df)} rows")

 



        # This loop acts as a Knapsack solver: trying to fit the most value (Carbon Savings/Equity)
        # into the fixed capacity (Budget).

        # --------------------------------------------------------------------
        # PART 2: RUN GREEDY LOOPS
        # --------------------------------------------------------------------
        prob_loft='None'
        
        for budget in budgets:
           # -- for prob_loft in loft_probs:
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
                    
                    # Check if already done
                    if os.path.exists(baseline_path) and os.path.exists(combined_path):
                        print(f"✓ Results already exist for this configuration, skipping...")
                        # Clear loggers for next iteration
                        summary_logger.handlers.clear()
                        detail_logger.handlers.clear()
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
                            
                            df, 
                            scenario_list, 
                            summary_logger, 
                            detail_logger, 
                            equity_factor, 
                            output_dir,  
                        )
                        
                        # check if empty 
                        if baseline_selection.empty:
                            raise Exception('Baseline results empty')
                        if combined_results.empty: 
                            raise Exception('Combined results empty')
                        
                        # Save results to CSV
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