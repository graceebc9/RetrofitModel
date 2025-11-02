"""
Greedy Algorithm Analysis for Retrofit Scenarios
Processes multiple epistemic runs and selects optimal retrofit projects within budget constraints.
"""

import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add custom module path
sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.validate import validate
from src.RetrofitPostProcess import process_multiple_scenarios
from src.GreedyAlgo import true_greedy_knapsack, plot_greedy_distribution_analysis
from src.RetrofitAnalysisUtils import load_data 
from src.RetrofitGreedy import run_greedy_algo 

from src.RetrofitEquity import EQUITY_WEIGHTS,  calculate_social_equity_score , calculate_scenario_persona_metrics

def load_personas(path = '/Users/gracecolverd/RetrofitModel/NE_region_personas.csv'):

    personas= pd.read_csv(path)
    
    return personas 

def setup_logging(output_dir, budget, prob_loft):
    """
    Set up logging with separate files for detailed logs and summary statistics.
    
    Args:
        output_dir: Directory to save log files
        budget: Budget amount for naming
        prob_loft: Probability of existing loft insulation for naming
    
    Returns:
        tuple: (summary_logger, detail_logger)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed log
    detail_log_path = os.path.join(output_dir, f'detailed_log_{budget}_loft{prob_loft}_{timestamp}.log')
    detail_logger = logging.getLogger(f'detail_{budget}_{prob_loft}')
    detail_logger.setLevel(logging.DEBUG)
    detail_handler = logging.FileHandler(detail_log_path)
    detail_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    detail_logger.addHandler(detail_handler)
    
    # Create summary log
    summary_log_path = os.path.join(output_dir, f'SUMMARY_{budget}_loft{prob_loft}_{timestamp}.txt')
    summary_logger = logging.getLogger(f'summary_{budget}_{prob_loft}')
    summary_logger.setLevel(logging.INFO)
    summary_handler = logging.FileHandler(summary_log_path)
    summary_handler.setFormatter(logging.Formatter('%(message)s'))
    summary_logger.addHandler(summary_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    summary_logger.addHandler(console_handler)
    
    return summary_logger, detail_logger







def main():
    """
    Main execution function for greedy algorithm analysis.
    """
    # Configuration
    BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy'
    INPUT_FILES_PATH = '/Volumes/T9/2025_10_RetrofitModel/all/*.csv'
    
    YEARS = 5
    N_SIMULATIONS = 5000
    ELEC_CARBON_FACTOR = 0.2
    GAS_CARBON_FACTOR = 0.2

    
    scenarios_config = [
        ("wall_installation", "wall_installation"),
        ("loft_installation", "loft_installation"),
        ("join_heat_ins_decay", "join_heat_ins_decay"),
        ("heat_pump_only", "heat_pump_only")
    ]
    
    scenario_list = ['wall_installation', 'loft_installation', 'join_heat_ins_decay', 'heat_pump_only']
    
    # Load and concatenate input data
    res_df =load_data(INPUT_FILES_PATH, scenario_list )
    personas = load_personas(path = '/Users/gracecolverd/RetrofitModel/NE_region_personas.csv') 
    res_df = res_df.merge(personas, on='postcode', how='inner')

    
    print("Processing scenarios...")
    proc_df = process_multiple_scenarios(
        res_df, scenarios_config, YEARS, N_SIMULATIONS,
        GAS_CARBON_FACTOR, ELEC_CARBON_FACTOR, gas_col='deriv'
    )
    
    # Filter data
    pdf = proc_df[proc_df['premise_type'] != 'Domestic_outbuilding'].copy()
    pdf = pdf[~pdf['premise_type'].isna()]
    df = pdf.copy()
    
    # Run analysis for different budget and loft probability combinations
    local= True 

    if local:
        budgets = [10_000_000, 100_000_000]
        loft_probs = [0.65 ]
    else:
        budgets = [10_000_000, 100_000_000, 1_000_000_000]
        loft_probs = [0.65, 0.95]
    
    for budget in budgets:
        for prob_loft in loft_probs:
       
            for equity_factor in [ 0,  0.2,   0.4,   0.6,   0.8 , 1 ]: 
                print(f"\n{'='*80}")
                print(f"Starting analysis: Budget £{budget:,}, Loft Probability {prob_loft} and equity: {equity_factor} ")
                print(f"{'='*80}")
                
                # Create output directory
                output_dir = os.path.join(BASE_DIR, f'budget_{budget}__loft_{prob_loft}__equity_{equity_factor}')
                os.makedirs(output_dir, exist_ok=True)
                
                # Set up logging
                summary_logger, detail_logger = setup_logging(output_dir, budget, prob_loft)
                
                summary_logger.info(f'Starting analysis: Budget £{budget:,}, Loft Probability {prob_loft}')
                
                # Run greedy algorithm
                baseline_selection, combined_results = run_greedy_algo(
                    budget, prob_loft, df, scenario_list, summary_logger, detail_logger, equity_factor, output_dir,  
            )
            
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
                print(f"Results saved to: {output_dir}")
                
                # Clear handlers to avoid duplicate logging in next iteration
                summary_logger.handlers.clear()
                detail_logger.handlers.clear()


if __name__ == "__main__":
    main()