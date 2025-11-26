"""
Energy Retrofit Analysis Script - Multi-Scenario Version (Updated for new column names)
Processes energy and carbon savings data for multiple retrofit scenarios with uncertainty analysis.
Works with pre-processed data that already includes absolute kWh savings and cost metrics.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import gc 

# Add RetrofitModel to path
sys.path.append('/rds/user/gb669/hpc-work/energy_map/RetrofitModel')
from src.validate import validate_single_scenario_new

# Note: clean_post_process is NO LONGER NEEDED - data is already processed
# from src.RetrofitPostProcess import clean_post_process 
from src.visualisations import run_vis_new 
from src.RetrofitAnalysis_meta import  run_meta
from src.RetrofitAnalysis import run_meta_portoflio

import psutil
import tracemalloc
from functools import wraps

from src.RetrofitAnalysisUtils import load_data, memory_profiler, log_memory, prepare_data_for_postanalysis

 


@memory_profiler
def analyze_uncertainty(df, scenario_name, measure_type, years, output_dir):
    """Perform epistemic vs aleatoric uncertainty analysis using CO2 columns."""
    print("\n" + "="*60)
    print(f"UNCERTAINTY ANALYSIS - {scenario_name}")
    print("="*60)
    
    # Use CO2 columns (created in add_co2_columns function)
    GAS_MEAN_COL = f'gas_{years}yr_kg_co2_saved_{scenario_name}_mean'
    GAS_P50_COL = f'gas_{years}yr_kg_co2_saved_{scenario_name}_p50'
    GAS_P95_COL = f'gas_{years}yr_kg_co2_saved_{scenario_name}_p95'
    GAS_P5_COL = f'gas_{years}yr_kg_co2_saved_{scenario_name}_p5'
    GAS_STD_COL = f'gas_{years}yr_kg_co2_saved_{scenario_name}_std'

    # Check if columns exist
    required_cols = [GAS_P50_COL, GAS_P95_COL, GAS_STD_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"WARNING: Missing columns for uncertainty analysis: {missing_cols}")
        print("Skipping uncertainty analysis for this scenario.")
        return None

    # Building-level metrics
    building_metrics = df.groupby('upn').agg(
        EPISTEMIC_GAS_STD=(GAS_P50_COL, 'std'),
        MEAN_ALEATORIC_GAS_STD=(GAS_STD_COL, 'mean'),
        MEAN_ALEATORIC_GAS_SPREAD=(GAS_P95_COL, 'mean'),
        MEAN_P50_GAS_BEST_ESTIMATE=(GAS_P50_COL, 'mean'),
        FINAL_GAS_MEAN_P50=(GAS_P50_COL, 'mean'),
        FINAL_GAS_MEAN_P95=(GAS_P95_COL, 'mean')
    ).reset_index()

    building_metrics['MEAN_ALEATORIC_GAS_RANGE'] = (
        building_metrics['FINAL_GAS_MEAN_P95'] - building_metrics['FINAL_GAS_MEAN_P50']
    )

    # Save building metrics
    building_metrics.to_csv(
        os.path.join(output_dir, 'building_uncertainty_metrics.csv'), 
        index=False
    )
    print(f"\nSaved building metrics to: {output_dir}/building_uncertainty_metrics.csv")

    # Plot: Distribution of uncertainties
    plt.figure(figsize=(10, 6))
    sns.histplot(building_metrics['EPISTEMIC_GAS_STD'], kde=True, 
                 label='Epistemic $\\sigma$ (StdDev of P50s)')
    sns.histplot(building_metrics['MEAN_ALEATORIC_GAS_STD'], kde=True, 
                 label='Aleatoric $\\sigma$ (Mean of STDs)', color='orange')
    plt.title(f'Distribution of Epistemic vs. Aleatoric Uncertainty\n({scenario_name}, Total {years}yr Gas KG CO2 Savings)')
    plt.xlabel('Uncertainty ($\\sigma$) in KG CO2/Year')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/uncertainty_distribution.png")

    # Plot: Scatter plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x='EPISTEMIC_GAS_STD', 
        y='MEAN_ALEATORIC_GAS_RANGE', 
        data=building_metrics, 
        alpha=0.6
    )
    max_val = max(
        building_metrics['EPISTEMIC_GAS_STD'].max(), 
        building_metrics['MEAN_ALEATORIC_GAS_RANGE'].max()
    )
    plt.plot([0, max_val], [0, max_val], 'r--', label='Epistemic = Aleatoric')
    plt.title(f'Epistemic vs. Aleatoric Uncertainty\n({scenario_name}, Total {years}yr KG CO2 Gas Savings)')
    plt.xlabel('Epistemic Uncertainty ($\\sigma$ of P50 across runs)')
    plt.ylabel('Aleatoric Uncertainty (Mean P95 - P50 spread)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epistemic_vs_aleatoric_scatter.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/epistemic_vs_aleatoric_scatter.png")

    return building_metrics


@memory_profiler
def process_single_scenario(df, scenario_name, measure_type, years, n_simulations,
                            gas_carbon_factor, elec_carbon_factor, output_dir, save_processed_data=False):
    """Process a single scenario and generate all outputs."""
    print("\n" + "="*80)
    print(f"PROCESSING SCENARIO: {scenario_name} (measure type: {measure_type})")
    print("="*80)

    # Create scenario-specific output directory
    scenario_output_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)
    print(f"Output directory: {scenario_output_dir}")

    # Convert kWh savings to CO2 savings (data already has kWh, we add CO2)
    print("\nAdding CO2 conversion columns...")
    df_processed = prepare_data_for_postanalysis(
        df.copy(), 
        scenario_name, 
        years,
        gas_carbon_factor,
        elec_carbon_factor
    )

    # Calculate and print summary statistics
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS - {scenario_name}")
    print("="*60)

    # Updated column names for new format
    scenario_cols = [
        f'{scenario_name}_cost_{scenario_name}_mean',
        f'{scenario_name}_cost_{scenario_name}_std',
        f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_mean',
        f'{scenario_name}_gas_saving_abs_kwh_{scenario_name}_std',
        f'{scenario_name}_gas_saving_perc_{scenario_name}_mean',
        f'{scenario_name}_gas_saving_perc_{scenario_name}_std',
    ]

    # Add electricity columns if they exist (for heat pump scenarios)
    elec_cols = [
        f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_mean',
        f'{scenario_name}_elec_saving_abs_kwh_{scenario_name}_std',
    ]

    available_cols = [col for col in scenario_cols + elec_cols if col in df_processed.columns]

    if available_cols:
        desc_stats = df_processed[available_cols].describe()
        print(desc_stats)
        desc_stats.to_csv(os.path.join(scenario_output_dir, 'descriptive_statistics.csv'))
        print(f"\nSaved: {scenario_output_dir}/descriptive_statistics.csv")
    else:
        print(f"WARNING: No expected columns found for {scenario_name}")
        print(f"Available columns: {df_processed.columns.tolist()}")

    # Total costs
    cost_col = f'{scenario_name}_cost_{scenario_name}_mean'
    if cost_col in df_processed.columns:
        mean_total_costs = df_processed.groupby('epistemic_run_id')[cost_col].sum().mean()
        std_total_costs = df_processed.groupby('epistemic_run_id')[cost_col].sum().std()

        mill = 1000000
        print(f"\nMean Total Costs for {scenario_name}: £{mean_total_costs/mill:.2f}M")
        print(f"Std Total Costs for {scenario_name}: £{std_total_costs/mill:.2f}M")
    else:
        print(f"WARNING: Cost column not found: {cost_col}")
        mean_total_costs = 0
        std_total_costs = 0

    # Validate
    print("\nRunning validation...")
    try:
        validate_single_scenario_new(df_processed, scenario_name)
        print("Validation passed!")
    except Exception as e:
        print(f"Validation warning: {e}")

    # Save processed data
    if save_processed_data:
        df_processed.to_csv(os.path.join(scenario_output_dir, 'processed_data.csv'), index=False)
        print(f"Saved processed data to: {scenario_output_dir}/processed_data.csv")

    # Perform analyses
    building_metrics = analyze_uncertainty(df_processed, scenario_name, measure_type, years, scenario_output_dir)
    
 
    portfolio_metrics = run_meta_portoflio(scenario_output_dir, df_processed, scenario_name, years=years)
 
 

    # Create visualizations
    try:
        run_vis_new(df_processed, scenario_name, scenario_output_dir)
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")

    # Save summary report
    print("\n" + "="*60)
    print(f"GENERATING SUMMARY REPORT - {scenario_name}")
    print("="*60)

    with open(os.path.join(scenario_output_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"ENERGY RETROFIT ANALYSIS - SUMMARY REPORT\n")
        f.write(f"Scenario: {scenario_name}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Measure Type: {measure_type}\n")
        f.write(f"Years: {years}\n")
        f.write(f"N Simulations: {n_simulations}\n\n")

        f.write("TOTAL COSTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean: £{mean_total_costs/1000000:.2f}M\n")
        f.write(f"Std:  £{std_total_costs/1000000:.2f}M\n\n")

        f.write("\n\nOUTPUT FILES\n")
        f.write("-"*60 + "\n")
        for file in sorted(os.listdir(scenario_output_dir)):
            f.write(f"- {file}\n")

    print(f"\nSaved summary report to: {scenario_output_dir}/summary_report.txt")
    print(f"\n{scenario_name} ANALYSIS COMPLETE!")

    return df_processed


def main():
    """Main execution function."""
    tracemalloc.start()
    log_memory("Script start")
    parser = argparse.ArgumentParser(
        description='Energy Retrofit Analysis Script - Multi-Scenario (Updated for new column format)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument(
        '--input-pattern',
        type=str,
        required=True,
        help='Glob pattern for input CSV files (e.g., "/path/to/data/*.csv")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this analysis run (default: timestamp)'
    )

    # Scenario configuration
    parser.add_argument(
        '--scenarios',
        type=str,
        nargs='+',
        required=True,
        help='List of scenario names (e.g., heat_pump_only join_heat_ins_decay)'
    )
    parser.add_argument(
        '--measure-types',
        type=str,
        nargs='+',
        required=True,
        help='List of measure types corresponding to scenarios (e.g., heat_pump joint_heat_ins_decay)'
    )

    # Analysis parameters
    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Number of years for projections'
    )
    parser.add_argument(
        '--n-simulations',
        type=int,
        default=5000,
        help='Number of Monte Carlo simulations'
    )
    parser.add_argument(
        '--gas-carbon-factor',
        type=float,
        default=0.18,
        help='Gas carbon factor (kg CO2/kWh)'
    )
    parser.add_argument(
        '--elec-carbon-factor',
        type=float,
        default=0.19338,
        help='Electricity carbon factor (kg CO2/kWh)'
    )
    parser.add_argument(
        '--save-processed-data',
        action='store_true',
        help='Save processed data to CSV (can be large)'
    )

    args = parser.parse_args()

    # Validate that scenarios and measure_types have same length
    if len(args.scenarios) != len(args.measure_types):
        raise ValueError("Number of scenarios must match number of measure types")

    # Create output directory
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("ENERGY RETROFIT ANALYSIS - MULTI-SCENARIO (UPDATED)")
    print("="*80)
    print(f"Run name: {args.run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Years: {args.years}")
    print("="*80 + "\n")

    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("Analysis Configuration\n")
        f.write("="*60 + "\n\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Load data once
    log_memory("Before loading data")
    res_df = load_data(args.input_pattern, args.scenarios, number=6)
    log_memory("After loading data")

    # Filter out domestic outbuildings
    pl = res_df[res_df['premise_type'] != 'Domestic outbuilding'].copy()
    print(f"After filtering: {len(pl)} rows")
    print(f"Available columns: {pl.columns.tolist()[:20]}...")  # Print first 20 columns
    log_memory("After filtering")
    
    del res_df
    gc.collect()
    log_memory("After freeing original dataframe")

    proc_df = prepare_data_for_postanalysis(
        pl, 
        args.scenarios,
        args.years, 
        args.gas_carbon_factor, 
        args.elec_carbon_factor,
    )
    # process meta on all scenarios 
    print('starting meta comparison plots')
    run_meta(output_dir, args, proc_df)
    del proc_df

    log_memory("meta end")


    # Process each scenario
    for i, (scenario_name, measure_type) in enumerate(zip(args.scenarios, args.measure_types)):
        print(f"\n{'#'*80}")
        print(f"# SCENARIO {i+1}/{len(args.scenarios)}: {scenario_name}")
        print(f"{'#'*80}")
        
        process_single_scenario(
            df=pl,
            scenario_name=scenario_name,
            measure_type=measure_type,
            years=args.years,
            n_simulations=args.n_simulations,
            gas_carbon_factor=args.gas_carbon_factor,
            elec_carbon_factor=args.elec_carbon_factor,
            output_dir=output_dir,
            save_processed_data=args.save_processed_data
        )
        
        gc.collect() 
        log_memory(f"After scenario {i+1}/{len(args.scenarios)} + GC")

    log_memory("Script end")

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n[TRACEMALLOC] Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()

    print("\n" + "="*80)
    print("ALL SCENARIOS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nScenario folders:")
    for scenario in args.scenarios:
        scenario_dir = os.path.join(output_dir, scenario)
        if os.path.exists(scenario_dir):
            print(f"  - {scenario}/")
            print(f"    Files: {len(os.listdir(scenario_dir))}")


if __name__ == "__main__":
    main()