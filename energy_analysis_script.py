#!/usr/bin/env python3
"""
Energy Retrofit Analysis Script
Processes energy and carbon savings data for retrofit scenarios with uncertainty analysis.
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
from src.visualisations import plot_col_reduction_by_decile
from src.RetrofitPostProcess import clean_post_proccess 

from src.visualisations import plot_building_counts_by_age_band 

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


def analyze_uncertainty(df, scenario_name, output_dir):
    """Perform epistemic vs aleatoric uncertainty analysis."""
    print("\n" + "="*60)
    print("UNCERTAINTY ANALYSIS")
    print("="*60)
    
    GAS_MEAN_COL = f'gas_5yr_kg_co2_saved_mean'
    GAS_P50_COL = f'gas_5yr_kg_co2_saved_p50'
    GAS_P95_COL = f'gas_5yr_kg_co2_saved_p95'
    GAS_P5_COL = f'gas_5yr_kg_co2_saved_p5'
    GAS_STD_COL = f'gas_5yr_kg_co2_saved_std'
    
    # Building-level metrics
    building_metrics = df.groupby('verisk_building_id_x').agg(
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
    plt.title('Distribution of Epistemic vs. Aleatoric Uncertainty (Total 5kyr Gas KG CO2 Savings)')
    plt.xlabel('Uncertainty ($\\sigma$) in MWh/Year')
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
    plt.title('Epistemic vs. Aleatoric Uncertainty for Total 5yr KG CO2 Gas Savings')
    plt.xlabel('Epistemic Uncertainty ($\\sigma$ of P50 across runs)')
    plt.ylabel('Aleatoric Uncertainty (Mean P95 - P50 spread)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epistemic_vs_aleatoric_scatter.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/epistemic_vs_aleatoric_scatter.png")
    
    return building_metrics

def analyze_portfolio(df, scenario_name, output_dir):
    """Perform portfolio-level analysis for gas and electricity."""
    print("\n" + "="*60)
    print("PORTFOLIO ANALYSIS")
    print("="*60)
    
    GAS_MEAN_COL = 'gas_5yr_kg_co2_saved_mean'
    GAS_P50_COL = 'gas_5yr_kg_co2_saved_p50'
    GAS_P95_COL = 'gas_5yr_kg_co2_saved_p95'
    GAS_P5_COL = 'gas_5yr_kg_co2_saved_p5'
    GAS_STD_COL = 'gas_5yr_kg_co2_saved_std'
    ELEC_P50_COL = 'elec_5yr_kg_co2_saved_p50'
    ELEC_P95_COL='elec_5yr_kg_co2_saved_p95'

    # Aggregate portfolio totals by epistemic run
    portfolio_summary = df.groupby('epistemic_run_id').agg(
        TOTAL_GAS_SAVINGS_P50=(GAS_P50_COL, 'sum'), 
        TOTAL_GAS_SAVINGS_P95=(GAS_P95_COL, 'sum'),
        TOTAL_ELEC_SAVINGS_P50=(ELEC_P50_COL, 'sum'),
        TOTAL_ELEC_SAVINGS_P95=(ELEC_P95_COL, 'sum')
    ).reset_index()
    
    # Save portfolio summary
    portfolio_summary.to_csv(
        os.path.join(output_dir, 'portfolio_summary.csv'), 
        index=False
    )
    print(f"Saved portfolio summary to: {output_dir}/portfolio_summary.csv")
    
    # Calculate final metrics for gas
    final_gas_metrics = {
        'Gas_Best_Estimate (P50 Mean)': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].mean(),
        'Gas_Epistemic_StdDev': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].std(),
        'Gas_Epistemic_Min': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].min(),
        'Gas_Epistemic_Max': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].max(),
        'Gas_High_Risk_Estimate (P95 Mean)': portfolio_summary['TOTAL_GAS_SAVINGS_P95'].mean(),
    }
    
    # Calculate final metrics for electricity
    final_elec_metrics = {
        'Elec_Best_Estimate (P50 Mean)': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].mean(),
        'Elec_Epistemic_StdDev': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].std(),
        'Elec_Epistemic_Min': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].min(),
        'Elec_Epistemic_Max': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].max(),
        'Elec_High_Risk_Estimate (P95 Mean)': portfolio_summary['TOTAL_ELEC_SAVINGS_P95'].mean(),
    }
    
    # Combine metrics
    final_portfolio_metrics = {**final_gas_metrics, **final_elec_metrics}
    
    # Print gas metrics
    print("\n--- Final Portfolio Uncertainty Summary (Gas Savings) ---")
    for k, v in final_gas_metrics.items():
        print(f"{k:<35}: {v:,.0f} MWh/year")
    
    # Print electricity metrics
    print("\n--- Final Portfolio Uncertainty Summary (Electricity Savings) ---")
    for k, v in final_elec_metrics.items():
        print(f"{k:<35}: {v:,.0f} MWh/year")
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'portfolio_metrics.txt'), 'w') as f:
        f.write("Final Portfolio Uncertainty Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write("GAS SAVINGS\n")
        f.write("-"*60 + "\n")
        for k, v in final_gas_metrics.items():
            f.write(f"{k:<35}: {v:,.0f} MWh/year\n")
        
        f.write("\n\nELECTRICITY SAVINGS\n")
        f.write("-"*60 + "\n")
        for k, v in final_elec_metrics.items():
            f.write(f"{k:<35}: {v:,.0f} MWh/year\n")
    
    # Create side-by-side plots for gas and electricity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gas plot
    sns.histplot(portfolio_summary['TOTAL_GAS_SAVINGS_P50'], kde=True, bins=10, ax=ax1)
    ax1.axvline(
        portfolio_summary['TOTAL_GAS_SAVINGS_P50'].mean(), 
        color='r', 
        linestyle='--', 
        label=f"Mean: {final_gas_metrics['Gas_Best_Estimate (P50 Mean)']:,.0f}"
    )
    ax1.set_title('Portfolio Gas Savings: Epistemic Uncertainty Distribution')
    ax1.set_xlabel('Total Gas Energy Savings (MWh/Year)')
    ax1.set_ylabel('Frequency (Number of Epistemic Runs)')
    ax1.legend()
    
    # Electricity plot
    sns.histplot(portfolio_summary['TOTAL_ELEC_SAVINGS_P50'], kde=True, bins=10, ax=ax2)
    ax2.axvline(
        portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].mean(), 
        color='r', 
        linestyle='--', 
        label=f"Mean: {final_elec_metrics['Elec_Best_Estimate (P50 Mean)']:,.0f}"
    )
    ax2.set_title('Portfolio Electricity Savings: Epistemic Uncertainty Distribution')
    ax2.set_xlabel('Total Electricity Energy Savings (MWh/Year)')
    ax2.set_ylabel('Frequency (Number of Epistemic Runs)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_distribution.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/portfolio_distribution.png")
    
    return portfolio_summary, final_portfolio_metrics

def create_visualizations(df, scenario_name, years, output_dir):
    """Create all visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Installation costs by decile
    print("Creating installation costs plot...")
    fig = plot_col_reduction_by_decile(
        df, 
        mean_col=f'{scenario_name}_cost_{scenario_name}_mean',
        std_col=f'{scenario_name}_cost_{scenario_name}_std',
        costs=True, 
        ylabel='Total Installation Costs'
    )
    plt.savefig(os.path.join(output_dir, 'installation_costs_by_decile.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/installation_costs_by_decile.png")
    
    # Net CO2 savings by decile
    print("Creating CO2 savings plot...")
    fig = plot_col_reduction_by_decile(
        df, 
        mean_col=f'total_tonne_co2_saved_{years}yr_mean',
        std_col=f'total_tonne_co2_saved_{years}yr_std',
        costs=False, 
        ylabel=f'Net tonne CO2 Saved over {years} yrs'
    )
    plt.savefig(os.path.join(output_dir, 'co2_savings_by_decile.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/co2_savings_by_decile.png")
    
    # Violin plot by gas percentile
    print("Creating gas percentile analysis plot...")
    plt.figure(figsize=(12, 7))
    sns.violinplot(
        data=df,
        x='avg_gas_percentile',
        y='cost_per_gas_ton_redutions'
    )
    plt.title('Analysis by Avg Gas Percentile (Cost per Ton Reduction)')
    plt.ylabel('Cost per Ton CO2 Reduction (£)')
    plt.xlabel('Average Gas Percentile')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gas_percentile_analysis.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/gas_percentile_analysis.png")
    
    # Cost per gas ton saved (filtered)
    print("Creating cost per gas ton plot...")
    pldf = df[df['avg_gas_percentile'] > 3].copy()
    pldf['flip_cost_per_gas_ton_redutions'] = -pldf['cost_per_gas_ton_redutions']
    fig = plot_col_reduction_by_decile(
        pldf, 
        mean_col='flip_cost_per_gas_ton_redutions',
        std_col='cost_per_gas_ton_co2_std',
        costs=True, 
        ylabel=f'Cost Per Gas Ton Saved over {years} yrs'
    )
    plt.savefig(os.path.join(output_dir, 'cost_per_gas_ton_by_decile.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/cost_per_gas_ton_by_decile.png")
    
    # Cost per net ton saved (filtered)
    print("Creating cost per net ton plot...")
    pldf['flip_sign_cost_per_net_ton_co2'] = -pldf['cost_per_net_ton_co2']
    fig = plot_col_reduction_by_decile(
        pldf, 
        mean_col='flip_sign_cost_per_net_ton_co2',
        std_col='cost_per_net_ton_co2_std',
        costs=True, 
        ylabel=f'Cost Per Net Ton Saved over {years} yrs'
    )
    plt.savefig(os.path.join(output_dir, 'cost_per_net_ton_by_decile.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/cost_per_net_ton_by_decile.png")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Energy Retrofit Analysis Script',
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
    
    # Analysis parameters
    parser.add_argument(
        '--scenario',
        type=str,
        default='join_heat_ins_decay',
        help='Scenario name'
    )
    parser.add_argument(
        '--measure-type',
        type=str,
        default='joint_heat_ins_decay',
        help='Measure type'
    )
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
        '--gas-price',
        type=float,
        default=0.07,
        help='Gas price (£/kWh)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ENERGY RETROFIT ANALYSIS")
    print("="*60)
    print(f"Run name: {args.run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Scenario: {args.scenario}")
    print(f"Years: {args.years}")
    print("="*60 + "\n")
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("Analysis Configuration\n")
        f.write("="*60 + "\n\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Load data
    res_df = load_data(args.input_pattern)
    
    # Filter out domestic outbuildings
    pl = res_df[res_df['premise_type'] != 'Domestic outbuilding'].copy()
    print(f"After filtering: {len(pl)} rows")
    
    # Process data
    print("\nProcessing energy and carbon metrics...")
    df_joint = clean_post_proccess(
        pl, 
        args.measure_type, 
        args.scenario, 
        years=args.years,
        GAS_CARBON_FACTOR_2022=args.gas_carbon_factor,
        elec_carbon_factor=args.elec_carbon_factor,
        n_simulations=args.n_simulations
    )
    print(df_joint.columns.tolist())
    # Calculate and print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    scenario_cols = [
        f'{args.scenario}_cost_{args.scenario}_mean',
        f'{args.scenario}_cost_{args.scenario}_std',
        f'{args.scenario}_{args.scenario}_gas_mean',
        f'{args.scenario}_{args.scenario}_electricity_mean',
        f'{args.scenario}_{args.scenario}_gas_std',
        f'{args.scenario}_{args.scenario}_electricity_std',
    ]
    
    desc_stats = res_df[scenario_cols].describe()
    print(desc_stats)
    desc_stats.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'))
    print(f"\nSaved: {output_dir}/descriptive_statistics.csv")
    
    # Total costs
    mean_total_costs = pl.groupby('epistemic_run_id')[
        f'{args.scenario}_cost_{args.scenario}_mean'
    ].sum().mean()
    std_total_costs = pl.groupby('epistemic_run_id')[
        f'{args.scenario}_cost_{args.scenario}_mean'
    ].sum().std()
    
    mill = 1000000
    print(f"\nMean Total Costs: £{mean_total_costs/mill:.2f}M")
    print(f"Std Total Costs: £{std_total_costs/mill:.2f}M")
    
    # Validate
    print("\nRunning validation...")
    try:
        validate_single_scenario_new(df_joint, args.scenario)
        print("Validation passed!")
    except Exception as e:
        print(f"Validation warning: {e}")
    
    # Save processed data
    df_joint.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    print(f"Saved processed data to: {output_dir}/processed_data.csv")
    
    # Perform analyses
    building_metrics = analyze_uncertainty(df_joint, args.scenario, output_dir)
    portfolio_summary, portfolio_metrics = analyze_portfolio(
        df_joint, args.scenario, output_dir
    )
    
    # Create visualizations
    create_visualizations(df_joint, args.scenario, args.years, output_dir)
    
    # Save summary report
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("ENERGY RETROFIT ANALYSIS - SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Run Name: {args.run_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scenario: {args.scenario}\n")
        f.write(f"Years: {args.years}\n")
        f.write(f"N Simulations: {args.n_simulations}\n\n")
        
        f.write("TOTAL COSTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean: £{mean_total_costs/mill:.2f}M\n")
        f.write(f"Std:  £{std_total_costs/mill:.2f}M\n\n")
        
        f.write("PORTFOLIO METRICS\n")
        f.write("-"*60 + "\n")
        for k, v in portfolio_metrics.items():
            f.write(f"{k:<35}: {v:,.0f} MWh/year\n")
        
        f.write("\n\nOUTPUT FILES\n")
        f.write("-"*60 + "\n")
        f.write("- processed_data.csv\n")
        f.write("- building_uncertainty_metrics.csv\n")
        f.write("- portfolio_summary.csv\n")
        f.write("- portfolio_metrics.txt\n")
        f.write("- descriptive_statistics.csv\n")
        f.write("- uncertainty_distribution.png\n")
        f.write("- epistemic_vs_aleatoric_scatter.png\n")
        f.write("- portfolio_distribution.png\n")
        f.write("- installation_costs_by_decile.png\n")
        f.write("- co2_savings_by_decile.png\n")
        f.write("- gas_percentile_analysis.png\n")
        f.write("- cost_per_gas_ton_by_decile.png\n")
        f.write("- cost_per_net_ton_by_decile.png\n")
    
    print(f"\nSaved summary report to: {output_dir}/summary_report.txt")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")


if __name__ == "__main__":
    main()