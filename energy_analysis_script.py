#!/usr/bin/env python3
"""
Energy Retrofit Analysis Script - Multi-Scenario Version
Processes energy and carbon savings data for multiple retrofit scenarios with uncertainty analysis.
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

# Add RetrofitModel to path
sys.path.append('/rds/user/gb669/hpc-work/energy_map/RetrofitModel')
from src.validate import validate_single_scenario_new

from src.RetrofitPostProcess import clean_post_proccess 
from src.visualisations import run_vis_new 

from src.RetrofitAnalysis import run_meta_portoflio
 
def load_data(input_pattern):
    """Load and concatenate CSV files matching the pattern."""
    print(f"Loading data from: {input_pattern}")
    files = glob.glob(input_pattern)
    print(f"Found {len(files)} files")
    
    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")
    
    res = []
    for f in files:
        df = pd.read_csv(f)
        res.append(df)
    
    res_df = pd.concat(res, ignore_index=True)
    print(f"Loaded {len(res_df)} rows")
    return res_df


def analyze_uncertainty(df, scenario_name, measure_type, years, output_dir):
    """Perform epistemic vs aleatoric uncertainty analysis."""
    print("\n" + "="*60)
    print(f"UNCERTAINTY ANALYSIS - {scenario_name}")
    print("="*60)
    
    # Updated column names with measure_type indexing
    GAS_MEAN_COL = f'gas_{years}yr_kg_co2_saved_{measure_type}_mean'
    GAS_P50_COL = f'gas_{years}yr_kg_co2_saved_{measure_type}_p50'
    GAS_P95_COL = f'gas_{years}yr_kg_co2_saved_{measure_type}_p95'
    GAS_P5_COL = f'gas_{years}yr_kg_co2_saved_{measure_type}_p5'
    GAS_STD_COL = f'gas_{years}yr_kg_co2_saved_{measure_type}_std'
    
    # Building-level metrics
    building_metrics = df.groupby('verisk_building_id').agg(
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
 

# def create_visualizations(df, scenario_name, measure_type, years, output_dir):
#     """Create comprehensive visualizations for the scenario."""
#     print("\n" + "="*60)
#     print(f"CREATING VISUALIZATIONS - {scenario_name}")
#     print("="*60)
    
#     # Updated column names with measure_type indexing
#     GAS_MEAN_COL = f'gas_{years}yr_kg_co2_saved_{measure_type}_mean'
#     TOTAL_CO2_COL = f'total_kg_co2_saved_{measure_type}_{years}yr_mean'
#     COST_COL = f'{scenario_name}_cost_{scenario_name}_mean'
#     COST_PER_NET_TON_COL = f'cost_per_net_ton_co2_{measure_type}_mean_thousands'
#     COST_PER_GAS_TON_COL = f'cost_per_gas_ton_reductions_{measure_type}_mean_th'
    
#     # Installation costs by decile
#     plt.figure(figsize=(10, 6))
#     plot_col_reduction_by_decile(
#         df, 
#         mean_col = COST_COL,
#         std_col = f'{scenario_name}_cost_{scenario_name}_std',
#         groupby_col='avg_gas_percentile',
#         # scenario_name=scenario_name
#     )
#     plt.title(f'Installation Costs by Avg Gas Consumption Decile - {scenario_name}')
#     plt.ylabel('Installation Cost (£)')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{scenario_name}_installation_costs_by_decile.png'), dpi=300)
#     plt.close()
#     print(f"Saved: {output_dir}/{scenario_name}_installation_costs_by_decile.png")
    
#     # CO2 savings by decile
#     plt.figure(figsize=(10, 6))
#     plot_col_reduction_by_decile(
#         df, 
#         mean_col =f'total_kg_co2_saved_{measure_type}_{years}yr_mean',
#         std_col = f'total_kg_co2_saved_{measure_type}_{years}yr_std',
#         groupby_col='avg_gas_percentile',
#         groupby_label='Avg Gas Consumption Decile',
#         # scenario_name=scenario_name
#     )
#     plt.title(f'Total CO2 Savings by Avg Gas Consumption Decile - {scenario_name}')
#     plt.ylabel('CO2 Savings (kg CO2)')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{scenario_name}_co2_savings_by_decile.png'), dpi=300)
#     plt.close()
#     print(f"Saved: {output_dir}/{scenario_name}_co2_savings_by_decile.png")
    
#     # Gas percentile analysis
#     plt.figure(figsize=(12, 6))
#     gas_cols = [
#         f'gas_{years}yr_kg_co2_saved_{measure_type}_p5',
#         f'gas_{years}yr_kg_co2_saved_{measure_type}_p50',
#         f'gas_{years}yr_kg_co2_saved_{measure_type}_p95'
#     ]
    
#     if all(col in df.columns for col in gas_cols):
#         df[gas_cols].boxplot()
#         plt.title(f'Gas Savings Percentile Analysis - {scenario_name}')
#         plt.ylabel('Gas Savings (kg CO2)')
#         plt.xticks([1, 2, 3], ['P5', 'P50', 'P95'])
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'{scenario_name}_gas_percentile_analysis.png'), dpi=300)
#         plt.close()
#         print(f"Saved: {output_dir}/{scenario_name}_gas_percentile_analysis.png")
    
#     # Cost per gas ton by decile
#     if COST_PER_GAS_TON_COL in df.columns:
#         plt.figure(figsize=(10, 6))
#         plot_col_reduction_by_decile(
#             df, 
#             mean_col = f'cost_per_gas_ton_reductions_{measure_type}_mean_th',
#             std_col = f'cost_per_gas_ton_reductions_{measure_type}_std_th',
#             groupby_col='avg_gas_percentile',
#             groupby_label='Avg Gas Consumption Decile',
#             # scenario_name=scenario_name
#         )
#         plt.title(f'Cost per Gas Ton CO2 Reduction by Avg Gas Decile - {scenario_name}')
#         plt.ylabel('Cost per Ton CO2 (£/ton, thousands)')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'{scenario_name}_cost_per_gas_ton_by_decile.png'), dpi=300)
#         plt.close()
#         print(f"Saved: {output_dir}/{scenario_name}_cost_per_gas_ton_by_decile.png")
    
#     # Cost per net ton by decile
#     if COST_PER_NET_TON_COL in df.columns:
#         plt.figure(figsize=(10, 6))
#         plot_col_reduction_by_decile(
#             df, 
#             mean_col = COST_PER_NET_TON_COL,
#             std_col = f'cost_per_net_ton_co2_{measure_type}_std_thousands',
#             groupby_col='avg_gas_percentile',
#             groupby_label='AVg Gas Consumption Decile',
#             # scenario_name=scenario_name
#         )
#         plt.title(f'Cost per Net Ton CO2 Reduction by Avg Gas Percentile - {scenario_name}')
#         plt.ylabel('Cost per Ton CO2 (£/ton, thousands)')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir,  f'{scenario_name}_cost_per_net_ton_by_decile.png'), dpi=300)
#         plt.close()
#         print(f"Saved: {output_dir}/{scenario_name}_cost_per_net_ton_by_decile.png")


def process_single_scenario(df, scenario_name, measure_type, years, n_simulations,
                            gas_carbon_factor, elec_carbon_factor, output_dir):
    """Process a single scenario and generate all outputs."""
    print("\n" + "="*80)
    print(f"PROCESSING SCENARIO: {scenario_name} (measure type: {measure_type})")
    print("="*80)
    
    # Create scenario-specific output directory
    scenario_output_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)
    print(f"Output directory: {scenario_output_dir}")
    
    # Process data
    print("\nProcessing energy and carbon metrics...")
    df_processed = clean_post_proccess(
        df, 
        measure_type, 
        scenario_name, 
        years=years,
        GAS_CARBON_FACTOR_2022=gas_carbon_factor,
        elec_carbon_factor=elec_carbon_factor,
        n_simulations=n_simulations
    )
    
    # Calculate and print summary statistics
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS - {scenario_name}")
    print("="*60)
    
    scenario_cols = [
        f'{scenario_name}_cost_{scenario_name}_mean',
        f'{scenario_name}_cost_{scenario_name}_std',
        f'{scenario_name}_{scenario_name}_gas_mean',
        f'{scenario_name}_{scenario_name}_gas_std',
    ]
    
    # Add electricity columns if they exist
    elec_cols = [
        f'{scenario_name}_{scenario_name}_electricity_mean',
        f'{scenario_name}_{scenario_name}_electricity_std',
    ]
    
    available_cols = [col for col in scenario_cols + elec_cols if col in df_processed.columns]
    
    if available_cols:
        desc_stats = df_processed[available_cols].describe()
        print(desc_stats)
        desc_stats.to_csv(os.path.join(scenario_output_dir, 'descriptive_statistics.csv'))
        print(f"\nSaved: {scenario_output_dir}/descriptive_statistics.csv")
    
    # Total costs
    cost_col = f'{scenario_name}_cost_{scenario_name}_mean'
    if cost_col in df_processed.columns:
        mean_total_costs = df_processed.groupby('epistemic_run_id')[cost_col].sum().mean()
        std_total_costs = df_processed.groupby('epistemic_run_id')[cost_col].sum().std()
        
        mill = 1000000
        print(f"\nMean Total Costs for {scenario_name}: £{mean_total_costs/mill:.2f}M")
        print(f"Std Total Costs for {scenario_name}: £{std_total_costs/mill:.2f}M")
    else:
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
    df_processed.to_csv(os.path.join(scenario_output_dir, 'processed_data.csv'), index=False)
    print(f"Saved processed data to: {scenario_output_dir}/processed_data.csv")
    print(df_processed.columns.tolist() )
    # Perform analyses
    building_metrics = analyze_uncertainty(df_processed, scenario_name, measure_type, years, scenario_output_dir)
    portfolio_metrics = run_meta_portoflio(scenario_output_dir, df_processed, scenario_name,  years=5 )
    
    # Create visualizations
    # create_visualizations(df_processed, scenario_name, measure_type, years, scenario_output_dir)
    run_vis_new(df_processed, scenario_name, scenario_output_dir)
    
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
        
        # f.write("PORTFOLIO METRICS\n")
        # f.write("-"*60 + "\n")
        # for k, v in portfolio_metrics.items():
        #     f.write(f"{k:<35}: {v:,.0f} KG CO2/year\n")
        
        f.write("\n\nOUTPUT FILES\n")
        f.write("-"*60 + "\n")
        for file in sorted(os.listdir(scenario_output_dir)):
            f.write(f"- {file}\n")
    
    print(f"\nSaved summary report to: {scenario_output_dir}/summary_report.txt")
    print(f"\n{scenario_name} ANALYSIS COMPLETE!")
    
    return df_processed


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Energy Retrofit Analysis Script - Multi-Scenario',
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
    print("ENERGY RETROFIT ANALYSIS - MULTI-SCENARIO")
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
    res_df = load_data(args.input_pattern)
    
    # Filter out domestic outbuildings
    pl = res_df[res_df['premise_type'] != 'Domestic outbuilding'].copy()
    print(f"After filtering: {len(pl)} rows")
    
    # Process each scenario
    all_results = {}
    for scenario_name, measure_type in zip(args.scenarios, args.measure_types):
        df_result = process_single_scenario(
            df=pl.copy(),
            scenario_name=scenario_name,
            measure_type=measure_type,
            years=args.years,
            n_simulations=args.n_simulations,
            gas_carbon_factor=args.gas_carbon_factor,
            elec_carbon_factor=args.elec_carbon_factor,
            output_dir=output_dir
        )
        all_results[scenario_name] = df_result
    
    print("\n" + "="*80)
    print("ALL SCENARIOS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nScenario folders:")
    for scenario in args.scenarios:
        scenario_dir = os.path.join(output_dir, scenario)
        print(f"  - {scenario}/")
        print(f"    Files: {len(os.listdir(scenario_dir))}")


if __name__ == "__main__":
    main()