 
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from .RetrofitPostProcess import clean_post_process 

def run_meta_portoflio(base_op, df_processed, scenraio,  years=5 ):
    """
    Run costs and energy portolfio level epistemic analysis
    """
    metrics={} 
    
    for i,  dd in df_processed.groupby('epistemic__cost_scenario'):
        suff = f'{scenraio}_{i}'
        op = os.path.join(base_op, scenraio, suff )
        os.makedirs(op , exist_ok=True) 
        metrics[i]= analyze_portfolio_costs(dd, scenraio, scenraio, op  , suff)
            
    # ADD THIS: Compare all cost scenarios
    suff_comparison = f'{scenraio}_cost_comparison'
    op_comparison = os.path.join(base_op, scenraio, suff_comparison)
    os.makedirs(op_comparison, exist_ok=True)
    
    cost_comparison = compare_cost_scenarios(
        df_processed, scenraio, scenraio, op_comparison, suff_comparison
    )
    metrics['cost_scenario_comparison'] = cost_comparison

    wall_cost_comparison = compare_by_wall_type(
                                                df_processed=df_processed,
                                                scenario_name=scenraio , 
                                                measure_type=scenraio,
                                                output_dir=op_comparison , 
                                                name_suffix='cost_wall_type',
                                                input_col_split='inferred_insulation_type'  
                                            )
    
    wall_cost_comparison = compare_by_wall_type_2d(
                                                df_processed=df_processed,
                                                scenario_name=scenraio , 
                                                measure_type=scenraio,
                                                output_dir=op_comparison , 
                                                name_suffix='cost_wall_type_scenario',
                                                input_col_split='inferred_insulation_type'  ,
                                                cost_scenario_col='epistemic__cost_scenario',
                                            )
    
    suff = f'{scenraio}_energy'
    op = os.path.join(base_op, scenraio, suff)
    os.makedirs(op, exist_ok=True)

    # energy portoflios sum 
    suff = f'{scenraio}_energy'
    op = os.path.join(base_op, scenraio, suff )
    os.makedirs(op,exist_ok=True) 
 
    energy_metrics = analyze_portfolio_energy(df_processed, scenraio, scenraio, years, op  , suff)
    metrics[suff] = energy_metrics
    return metrics 

def analyze_portfolio_energy(df, scenario_name, measure_type, years, output_dir, name_suffix):
    """Perform portfolio-level analysis for gas and electricity."""
    print("\n" + "="*60)
    print(f"PORTFOLIO ANALYSIS - {scenario_name}")
    print("="*60)
    
    # Updated column names with measure_type indexing
    base = f'total_tonne_co2_saved_{measure_type}_{years}yr'
    GAS_MEAN_COL = f'gas_{base}_mean'
    GAS_P50_COL = f'gas_{base}_p50'
    GAS_P95_COL = f'gas_{base}_p95'
    GAS_P5_COL = f'gas_{base}_p5'
    GAS_STD_COL = f'gas_{base}_std'
    ELEC_P50_COL = f'elec_{base}_p50'
    ELEC_P95_COL = f'elec_{base}_p95'

    # Check if electricity columns exist (for heat pump scenarios)
    has_elec = ELEC_P50_COL in df.columns

    # Aggregate portfolio totals by epistemic run
    agg_dict = {
      GAS_P50_COL: 'sum', 
        GAS_P95_COL: 'sum',
    }
    rename_dict= {
    GAS_P50_COL: 'TOTAL_GAS_SAVINGS_P50',
    GAS_P95_COL: 'TOTAL_GAS_SAVINGS_P95'} 
    
    if has_elec:
        agg_dict[ ELEC_P50_COL] = 'sum'
        agg_dict[ELEC_P95_COL] =   'sum'
        rename_dict[ELEC_P50_COL] =  'TOTAL_ELEC_SAVINGS_P50'
        rename_dict[ELEC_P95_COL] =  'TOTAL_ELEC_SAVINGS_P95' 
    
 
    portfolio_summary = df.groupby('epistemic_run_id').agg(agg_dict).reset_index()
    portfolio_summary = portfolio_summary.rename(columns=rename_dict) 
    
    # Save portfolio summary
    portfolio_summary.to_csv(
        os.path.join(output_dir, f'{name_suffix}_portfolio_summary.csv'), 
        index=False
    )
    print(f"Saved portfolio summary to: {output_dir}/portfolio_summary.csv")
    
    # Calculate final metrics for gas
    final_gas_metrics = {
        'Gas_Best_Estimate (P50 Mean)': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].mean(),
        'Gas_Epistemic_StdDev': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].std(),
        'Gas_Epistemic_Max': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].min(),
        'Gas_Epistemic_Min': portfolio_summary['TOTAL_GAS_SAVINGS_P50'].max(),
        'Gas_High_Risk_Estimate (P95 Mean)': portfolio_summary['TOTAL_GAS_SAVINGS_P95'].mean(),
    }
    
    # Calculate final metrics for electricity (if applicable)
    final_elec_metrics = {}
    if has_elec:
        final_elec_metrics = {
            'Elec_Best_Estimate (P50 Mean)': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].mean(),
            'Elec_Epistemic_StdDev': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].std(),
            'Elec_Epistemic_Max': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].min(),
            'Elec_Epistemic_Min': portfolio_summary['TOTAL_ELEC_SAVINGS_P50'].max(),
            'Elec_High_Risk_Estimate (P95 Mean)': portfolio_summary['TOTAL_ELEC_SAVINGS_P95'].mean(),
        }
    
    # Combine metrics
    final_portfolio_metrics = {**final_gas_metrics, **final_elec_metrics}
    
    # Print gas metrics
    print("\n--- Final Portfolio Uncertainty Summary (Gas Savings) ---")
    for k, v in final_gas_metrics.items():
        print(f"{k:<35}: {v:,.0f} TON CO2 over whole time")
    
    # Print electricity metrics
    if has_elec:
        print("\n--- Final Portfolio Uncertainty Summary (Electricity Savings) ---")
        for k, v in final_elec_metrics.items():
            print(f"{k:<35}: {v:,.0f} TON CO2 over whole time")
    
    # Save metrics to file
    with open(os.path.join(output_dir, f'{name_suffix}_portfolio_metrics.txt'), 'w') as f:
        f.write("Final Portfolio Uncertainty Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write("GAS SAVINGS\n")
        f.write("-"*60 + "\n")
        for k, v in final_gas_metrics.items():
            f.write(f"{k:<35}: {v:,.0f} TON CO2 over whole time\n")
        
        if has_elec:
            f.write("\n\nELECTRICITY SAVINGS\n")
            f.write("-"*60 + "\n")
            for k, v in final_elec_metrics.items():
                f.write(f"{k:<35}: {v:,.0f} TON CO2 over whole time\n")
    
    # Create plots
    if has_elec:
        # Side-by-side plots for gas and electricity
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gas plot
        sns.histplot(portfolio_summary['TOTAL_GAS_SAVINGS_P50'], kde=True, bins=10, ax=ax1)
        ax1.axvline(
            portfolio_summary['TOTAL_GAS_SAVINGS_P50'].mean(), 
            color='r', 
            linestyle='--', 
            label=f"Mean: {final_gas_metrics['Gas_Best_Estimate (P50 Mean)']:,.0f}"
        )
        ax1.set_title(f'Portfolio Gas Savings: Epistemic Uncertainty\n({scenario_name})')
        ax1.set_xlabel('Total Gas CO2 Savings (TON CO2)')
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
        ax2.set_title(f'Portfolio Electricity Savings: Epistemic Uncertainty\n({scenario_name})')
        ax2.set_xlabel('Total Electricity CO2 Savings (TON CO2/Year)')
        ax2.set_ylabel('Frequency (Number of Epistemic Runs)')
        ax2.legend()
        
        plt.tight_layout()
    else:
        # Single plot for gas only
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        sns.histplot(portfolio_summary['TOTAL_GAS_SAVINGS_P50'], kde=True, bins=10, ax=ax1)
        ax1.axvline(
            portfolio_summary['TOTAL_GAS_SAVINGS_P50'].mean(), 
            color='r', 
            linestyle='--', 
            label=f"Mean: {final_gas_metrics['Gas_Best_Estimate (P50 Mean)']:,.0f}"
        )
        ax1.set_title(f'Portfolio Gas Savings: Epistemic Uncertainty\n({scenario_name})')
        ax1.set_xlabel('Total Gas CO2 Savings (KG CO2/Year)')
        ax1.set_ylabel('Frequency (Number of Epistemic Runs)')
        ax1.legend()
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{name_suffix}_portfolio_distribution.png'), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/{name_suffix}_portfolio_distribution.png")
    
    return  final_portfolio_metrics

def analyze_portfolio_costs(df, scenario_name, measure_type, output_dir, name_suffix):

    # Column names
    COST_PER_GAS_P95_COL = f'cost_per_gas_ton_reductions_{measure_type}_p95'
    COST_PER_GAS_P50_COL = f'cost_per_gas_ton_reductions_{measure_type}_p50'
    TOTAL_P50_COL = f'cost_per_net_ton_co2_{measure_type}_p50'
    TOTAL_P95_COL = f'cost_per_net_ton_co2_{measure_type}_p95'
    
    cost_p50 = f'{scenario_name}_cost_{scenario_name}_p50_mill'
    cost_p95 = f'{scenario_name}_cost_{scenario_name}_p95_mill'
    
    # CO2 savings columns (need these to calculate cost per ton correctly)
    base = f'total_tonne_co2_saved_{measure_type}_5yr'  # Adjust years as needed
    GAS_P50_COL = f'gas_{base}_p50'
    GAS_P95_COL = f'gas_{base}_p95'
    
    # Check if electricity columns exist
    ELEC_P50_COL = f'elec_{base}_p50'
    has_elec = ELEC_P50_COL in df.columns
    
    # Aggregate portfolio totals by epistemic run
    # Sum total costs and CO2 savings (NOT cost per ton values)
    agg_dict = {
        cost_p50: 'sum', 
        cost_p95: 'sum',
        GAS_P50_COL: 'sum',
        GAS_P95_COL: 'sum',
    }
    
    if has_elec:
        agg_dict[ELEC_P50_COL] = 'sum'
        ELEC_P95_COL = f'elec_{base}_p95'
        agg_dict[ELEC_P95_COL] = 'sum'
    
    portfolio_summary = df.groupby('epistemic_run_id').agg(agg_dict).reset_index()
    
    # CALCULATE cost per ton at portfolio level (not by summing individual ratios)
    portfolio_summary['Cost_per_Ton_Gas_P50'] = (
        portfolio_summary[cost_p50] / portfolio_summary[GAS_P50_COL]
    )
    portfolio_summary['Cost_per_Ton_Gas_P95'] = (
        portfolio_summary[cost_p95] / portfolio_summary[GAS_P95_COL]
    )
    
    # Calculate NET cost per ton (accounting for electricity if applicable)
    if has_elec:
        portfolio_summary['Total_Net_CO2_P50'] = (
            portfolio_summary[GAS_P50_COL] - portfolio_summary[ELEC_P50_COL]
        )
        portfolio_summary['Total_Net_CO2_P95'] = (
            portfolio_summary[GAS_P95_COL] - portfolio_summary[ELEC_P95_COL]
        )
    else:
        portfolio_summary['Total_Net_CO2_P50'] = portfolio_summary[GAS_P50_COL]
        portfolio_summary['Total_Net_CO2_P95'] = portfolio_summary[GAS_P95_COL]
    
    portfolio_summary['Cost_per_Net_Ton_P50'] = (
        portfolio_summary[cost_p50] / portfolio_summary['Total_Net_CO2_P50']
    )
    portfolio_summary['Cost_per_Net_Ton_P95'] = (
        portfolio_summary[cost_p95] / portfolio_summary['Total_Net_CO2_P95']
    )
    
    # Rename for clarity
    portfolio_summary = portfolio_summary.rename(columns={
        cost_p50: 'Total_Costs_P50',
        cost_p95: 'Total_Costs_P95',
    })
    
    # Save portfolio summary
    portfolio_summary.to_csv(
        os.path.join(output_dir, f'{name_suffix}_costs_portfolio_summary.csv'), 
        index=False
    )
    print(f"Saved portfolio summary to: {output_dir}/{name_suffix}_costs_portfolio_summary.csv")

    # Calculate final metrics
    final_gas_metrics = {
        'Cost_per_TonCO2_Gas_Best_Estimate (P50 Mean)': portfolio_summary['Cost_per_Ton_Gas_P50'].mean(),
        'Cost_per_TonCO2_Gas_Epistemic_StdDev': portfolio_summary['Cost_per_Ton_Gas_P50'].std(),
        'Cost_per_TonCO2_Gas_Epistemic_Min': portfolio_summary['Cost_per_Ton_Gas_P50'].min(),
        'Cost_per_TonCO2_Gas_Epistemic_Max': portfolio_summary['Cost_per_Ton_Gas_P50'].max(),
        'Cost_per_TonCO2_Gas_High_Risk (P95 Mean)': portfolio_summary['Cost_per_Ton_Gas_P95'].mean(),
    }
    
    final_net_metrics = {
        'Net Cost per Ton CO2 (P50 Mean)': portfolio_summary['Cost_per_Net_Ton_P50'].mean(),
        'Net Cost per Ton CO2 StdDev': portfolio_summary['Cost_per_Net_Ton_P50'].std(),
        'Net Cost per Ton CO2 Min': portfolio_summary['Cost_per_Net_Ton_P50'].min(),
        'Net Cost per Ton CO2 Max': portfolio_summary['Cost_per_Net_Ton_P50'].max(),
        'Net Cost per Ton CO2 High Risk (P95 Mean)': portfolio_summary['Cost_per_Net_Ton_P95'].mean(),
    }

    final_cost_metrics = {
        'Total Costs (P50 Mean)': portfolio_summary['Total_Costs_P50'].mean(),
        'Total Costs Epistemic STD': portfolio_summary['Total_Costs_P50'].std(),
        'Total Costs Epistemic Min': portfolio_summary['Total_Costs_P50'].min(),
        'Total Costs Epistemic Max': portfolio_summary['Total_Costs_P50'].max(),
        'Total Costs High Risk (P95)': portfolio_summary['Total_Costs_P95'].mean(),
    }
    
    # Combine metrics
    final_portfolio_metrics = {**final_gas_metrics, **final_net_metrics, **final_cost_metrics}
    
    # Print and save metrics (rest of your code remains the same)
    print(f"\n--- Final Cost Portfolio Uncertainty Summary: {name_suffix} ---")
    for k, v in final_gas_metrics.items():
        print(f"{k:<50}: £{v:,.2f} per TON CO2")
    for k, v in final_net_metrics.items():
        print(f"{k:<50}: £{v:,.2f} per TON CO2")
    for k, v in final_cost_metrics.items():
        print(f"{k:<50}: £{v:,.2f}M")

    # Save metrics to file
    with open(os.path.join(output_dir, f'{name_suffix}_costs_portfolio_metrics.txt'), 'w') as f:
        f.write(f"Final Portfolio Uncertainty Summary: {name_suffix}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Cost per Ton CO2 (GAS) (£ per Ton)\n")
        f.write("-"*60 + "\n")
        for k, v in final_gas_metrics.items():
            f.write(f"{k:<50}: £{v:,.2f} per ton\n")
        
        f.write("\n\nCost per Net Ton CO2 (£ per ton)\n")
        f.write("-"*60 + "\n")
        for k, v in final_net_metrics.items():
            f.write(f"{k:<50}: £{v:,.2f} per ton\n")
        
        f.write("\n\nTotal Costs (£M)\n")
        f.write("-"*60 + "\n")
        for k, v in final_cost_metrics.items():
            f.write(f"{k:<50}: £{v:,.2f}M\n")

    # # Plotting code (your existing code with updated column names)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # sns.histplot(portfolio_summary['Cost_per_Ton_Gas_P50'], kde=True, bins=10, ax=ax1)
    # ax1.axvline(
    #     portfolio_summary['Cost_per_Ton_Gas_P50'].mean(), 
    #     color='r', 
    #     linestyle='--', 
    #     label=f"Mean: £{final_gas_metrics['Cost_per_TonCO2_Gas_Best_Estimate (P50 Mean)']:,.2f}"
    # )
    # ax1.set_title(f'Portfolio Cost per Ton CO2 Removal: Epistemic Uncertainty\n({scenario_name})')
    # ax1.set_xlabel('Cost per Ton CO2 Savings (£/TON)')
    # ax1.set_ylabel('Frequency (Number of Epistemic Runs)')
    # ax1.legend()
    
    # sns.histplot(portfolio_summary['Cost_per_Net_Ton_P50'], kde=True, bins=10, ax=ax2)
    # ax2.axvline(
    #     portfolio_summary['Cost_per_Net_Ton_P50'].mean(), 
    #     color='r', 
    #     linestyle='--', 
    #     label=f"Mean: £{final_net_metrics['Net Cost per Ton CO2 (P50 Mean)']:,.2f}"
    # )
    # ax2.set_title(f'Portfolio Net Cost per TON CO2: Epistemic Uncertainty\n({scenario_name})')
    # ax2.set_xlabel('Cost per Net TON CO2 (£/TON)')
    # ax2.set_ylabel('Frequency (Number of Epistemic Runs)')
    # ax2.legend()
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'{name_suffix}_cost_per_ton_distributions.png'), 
    #             dpi=300, bbox_inches='tight')
    # plt.close()
    
    # fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # sns.histplot(portfolio_summary['Total_Costs_P50'], kde=True, bins=10, ax=ax3)
    # ax3.axvline(
    #     portfolio_summary['Total_Costs_P50'].mean(), 
    #     color='r', 
    #     linestyle='--', 
    #     label=f"Mean: £{final_cost_metrics['Total Costs (P50 Mean)']:,.2f}M"
    # )
    # ax3.set_title(f'Portfolio Total Costs: Epistemic Uncertainty\n({scenario_name})')
    # ax3.set_xlabel('Total Costs (£M)')
    # ax3.set_ylabel('Frequency (Number of Epistemic Runs)')
    # ax3.legend()
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'{name_suffix}_total_costs_distribution.png'), 
    #             dpi=300, bbox_inches='tight')
    # plt.close()
    
    return final_portfolio_metrics

def compare_cost_scenarios(df_processed, scenario_name, measure_type, output_dir, name_suffix):
    """
    Compare different epistemic cost scenarios with overlapping histograms.
    """
    print("\n" + "="*60)
    print(f"COST SCENARIO COMPARISON - {scenario_name}")
    print("="*60)
    
    # Column names
    cost_p50 = f'{scenario_name}_cost_{scenario_name}_p50'
    cost_p95 = f'{scenario_name}_cost_{scenario_name}_p95'
    cost_p50_m =  f'{scenario_name}_cost_{scenario_name}_p50_mill'
    cost_p95_m =  f'{scenario_name}_cost_{scenario_name}_p95_mill'
    
    # CO2 savings columns
    base = f'total_tonne_co2_saved_{measure_type}_5yr'
    GAS_P50_COL = f'gas_{base}_p50'
    GAS_P95_COL = f'gas_{base}_p95'
    
    # Check if electricity columns exist
    ELEC_P50_COL = f'elec_{base}_p50'
    has_elec = ELEC_P50_COL in df_processed.columns
    
    net_co2_col_p50 = f'total_tonne_co2_saved_{measure_type}_5yr_p50'
    net_co2_col_p95 = f'total_tonne_co2_saved_{measure_type}_5yr_p95'

    # Prepare aggregation dict
    agg_dict = {
        cost_p50: 'sum', 
        cost_p95: 'sum',
        GAS_P50_COL: 'sum',
        GAS_P95_COL: 'sum',
        cost_p50_m: 'sum', 
        cost_p95_m: 'sum',
        net_co2_col_p50: 'sum', 
        net_co2_col_p95: 'sum',


    }
    
    if has_elec:
        ELEC_P95_COL = f'elec_{base}_p95'
        agg_dict[ELEC_P50_COL] = 'sum'
        agg_dict[ELEC_P95_COL] = 'sum'
    
    # Store data for each cost scenario
    scenario_data = {}
    cost_scenarios = df_processed['epistemic__cost_scenario'].unique()
    
    for cost_scenario in cost_scenarios:
        df_scenario = df_processed[df_processed['epistemic__cost_scenario'] == cost_scenario]
        
        # Aggregate by epistemic run
        portfolio_summary = df_scenario.groupby('epistemic_run_id').agg(agg_dict).reset_index()
        
        # Calculate cost per ton metrics
        portfolio_summary['Cost_per_Ton_Gas_P50'] = (
            portfolio_summary[cost_p50] / portfolio_summary[GAS_P50_COL]
        )
        portfolio_summary['Cost_per_Ton_Gas_P95'] = (
            portfolio_summary[cost_p95] / portfolio_summary[GAS_P95_COL]
        )
        

        portfolio_summary['Total_Net_CO2_P50'] = (
            portfolio_summary[net_co2_col_p50] 
        )
        portfolio_summary['Total_Net_CO2_P95'] = (
            portfolio_summary[net_co2_col_p95]
        )
 
        
        portfolio_summary['Cost_per_Net_Ton_P50'] = (
            portfolio_summary[cost_p50] / portfolio_summary['Total_Net_CO2_P50']
        )
        portfolio_summary['Cost_per_Net_Ton_P95'] = (
            portfolio_summary[cost_p95] / portfolio_summary['Total_Net_CO2_P95']
        )
        
        portfolio_summary['Total_Costs_P50'] = portfolio_summary[cost_p50_m]
        portfolio_summary['Total_Costs_P95'] = portfolio_summary[cost_p95_m]
        
        scenario_data[cost_scenario] = portfolio_summary
    
    # Create comparison plots with overlapping histograms
    metrics_to_plot = [
        ('Cost_per_Ton_Gas_P50', 'Cost per Ton Gas CO2 (£/TON)', 'cost_per_ton_gas_p50'),
        ('Cost_per_Net_Ton_P50', 'Cost per Net Ton CO2 (£/TON)', 'cost_per_net_ton_p50'),
        ('Total_Costs_P50', 'Total Costs (£M)', 'total_costs_p50'),
    ]
    
    # Color palette for different scenarios
    colors = plt.cm.Set2(np.linspace(0, 1, len(cost_scenarios)))
    
    for metric_col, metric_label, file_suffix in metrics_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        for idx, (cost_scenario, portfolio_summary) in enumerate(scenario_data.items()):
            # Plot histogram with transparency
            sns.histplot(
                portfolio_summary[metric_col], 
                kde=True, 
                bins=15, 
                ax=ax,
                alpha=0.5,
                color=colors[idx],
                label=f'{cost_scenario} (μ={portfolio_summary[metric_col].mean():,.2f})',
                stat='density'
            )
            
            # Add mean line
            ax.axvline(
                portfolio_summary[metric_col].mean(), 
                color=colors[idx], 
                linestyle='--', 
                linewidth=2,
                alpha=0.8
            )
        
        ax.set_title(f'Cost Scenario Comparison: {metric_label}\n({scenario_name})', fontsize=14, fontweight='bold')
        ax.set_xlabel(metric_label, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{name_suffix}_comparison_{file_suffix}.png'), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        print(f"Saved: {output_dir}/{name_suffix}_comparison_{file_suffix}.png")
    
    # Create summary statistics table
    summary_stats = []
    for cost_scenario, portfolio_summary in scenario_data.items():
        stats = {
            'Cost_Scenario': cost_scenario,
            'Cost_per_Ton_Gas_Mean': portfolio_summary['Cost_per_Ton_Gas_P50'].mean(),
            'Cost_per_Ton_Gas_Std': portfolio_summary['Cost_per_Ton_Gas_P50'].std(),
            'Cost_per_Net_Ton_Mean': portfolio_summary['Cost_per_Net_Ton_P50'].mean(),
            'Cost_per_Net_Ton_Std': portfolio_summary['Cost_per_Net_Ton_P50'].std(),
            'Total_Costs_Mean': portfolio_summary['Total_Costs_P50'].mean(),
            'Total_Costs_Std': portfolio_summary['Total_Costs_P50'].std(),
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(
        os.path.join(output_dir, f'{name_suffix}_cost_scenario_comparison.csv'),
        index=False
    )
    print(f"\nSaved comparison summary: {output_dir}/{name_suffix}_cost_scenario_comparison.csv")
    
    # Print summary
    print("\n--- Cost Scenario Comparison Summary ---")
    print(summary_df.to_string(index=False))
    
    return summary_df

def compare_by_wall_type(df_processed, scenario_name, measure_type, output_dir, name_suffix, input_col_split='inferred_insulation_type'):
    """
    Compare costs across different wall types (or other categorical splits) with overlapping histograms.
    
    Parameters:
    -----------
    input_col_split : str
        Column name to split the analysis by (default: 'inferred_insulation_type')
    """
    print("\n" + "="*60)
    print(f"WALL TYPE COMPARISON - {scenario_name}")
    print("="*60)
    
    # Check if split column exists
    if input_col_split not in df_processed.columns:
        print(f"Warning: Column '{input_col_split}' not found in dataframe. Skipping wall type comparison.")
        return None
    
    # Column names
    cost_p50 = f'{scenario_name}_cost_{scenario_name}_p50'
    cost_p95 = f'{scenario_name}_cost_{scenario_name}_p95'
    cost_p50_m = f'{scenario_name}_cost_{scenario_name}_p50_mill'
    cost_p95_m = f'{scenario_name}_cost_{scenario_name}_p95_mill'
    
    # CO2 savings columns
    base = f'total_tonne_co2_saved_{measure_type}_5yr'
    GAS_P50_COL = f'gas_{base}_p50'
    GAS_P95_COL = f'gas_{base}_p95'
    
    # Check if electricity columns exist
    ELEC_P50_COL = f'elec_{base}_p50'
    has_elec = ELEC_P50_COL in df_processed.columns
    
    net_co2_col_p50 = f'total_tonne_co2_saved_{measure_type}_5yr_p50'
    net_co2_col_p95 = f'total_tonne_co2_saved_{measure_type}_5yr_p95'

    # Prepare aggregation dict
    agg_dict = {
        cost_p50: 'sum', 
        cost_p95: 'sum',
        GAS_P50_COL: 'sum',
        GAS_P95_COL: 'sum',
        cost_p50_m: 'sum', 
        cost_p95_m: 'sum',
        net_co2_col_p50: 'sum', 
        net_co2_col_p95: 'sum',
    }
    
    if has_elec:
        ELEC_P95_COL = f'elec_{base}_p95'
        agg_dict[ELEC_P50_COL] = 'sum'
        agg_dict[ELEC_P95_COL] = 'sum'
    
    # Store data for each wall type
    wall_type_data = {}
    wall_types = df_processed[input_col_split].unique()
    wall_types = [wt for wt in wall_types if pd.notna(wt)]  # Remove NaN values
    
    print(f"\nFound {len(wall_types)} unique values in '{input_col_split}': {wall_types}")
    
    for wall_type in wall_types:
        df_wall = df_processed[df_processed[input_col_split] == wall_type]
        
        # Aggregate by epistemic run
        portfolio_summary = df_wall.groupby('epistemic_run_id').agg(agg_dict).reset_index()
        
        # Calculate cost per ton metrics
        portfolio_summary['Cost_per_Ton_Gas_P50'] = (
            portfolio_summary[cost_p50] / portfolio_summary[GAS_P50_COL]
        )
        portfolio_summary['Cost_per_Ton_Gas_P95'] = (
            portfolio_summary[cost_p95] / portfolio_summary[GAS_P95_COL]
        )
        
        portfolio_summary['Total_Net_CO2_P50'] = portfolio_summary[net_co2_col_p50]
        portfolio_summary['Total_Net_CO2_P95'] = portfolio_summary[net_co2_col_p95]
        
        portfolio_summary['Cost_per_Net_Ton_P50'] = (
            portfolio_summary[cost_p50] / portfolio_summary['Total_Net_CO2_P50']
        )
        portfolio_summary['Cost_per_Net_Ton_P95'] = (
            portfolio_summary[cost_p95] / portfolio_summary['Total_Net_CO2_P95']
        )
        
        portfolio_summary['Total_Costs_P50'] = portfolio_summary[cost_p50_m]
        portfolio_summary['Total_Costs_P95'] = portfolio_summary[cost_p95_m]
        
        wall_type_data[wall_type] = portfolio_summary
    
    # Create comparison plots with overlapping histograms
    metrics_to_plot = [
        ('Cost_per_Ton_Gas_P50', 'Cost per Ton Gas CO2 (£/TON)', 'cost_per_ton_gas_p50'),
        ('Cost_per_Net_Ton_P50', 'Cost per Net Ton CO2 (£/TON)', 'cost_per_net_ton_p50'),
        ('Total_Costs_P50', 'Total Costs (£M)', 'total_costs_p50'),
    ]
    
    # Color palette for different wall types
    colors = plt.cm.tab10(np.linspace(0, 1, len(wall_types)))
    
    for metric_col, metric_label, file_suffix in metrics_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        for idx, (wall_type, portfolio_summary) in enumerate(wall_type_data.items()):
            # Skip if no valid data
            valid_data = portfolio_summary[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_data) == 0:
                print(f"Warning: No valid data for {wall_type} in {metric_col}")
                continue
            
            # Plot histogram with transparency
            sns.histplot(
                valid_data, 
                kde=True, 
                bins=15, 
                ax=ax,
                alpha=0.5,
                color=colors[idx],
                label=f'{wall_type} (μ={valid_data.mean():,.2f}, n={len(valid_data)})',
                stat='density'
            )
            
            # Add mean line
            ax.axvline(
                valid_data.mean(), 
                color=colors[idx], 
                linestyle='--', 
                linewidth=2,
                alpha=0.8
            )
        
        ax.set_title(f'{input_col_split.replace("_", " ").title()} Comparison: {metric_label}\n({scenario_name})', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(metric_label, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{name_suffix}_{input_col_split}_comparison_{file_suffix}.png'
        plt.savefig(
            os.path.join(output_dir, filename), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        print(f"Saved: {output_dir}/{filename}")
    
    # Create summary statistics table
    summary_stats = []
    for wall_type, portfolio_summary in wall_type_data.items():
        stats = {
            input_col_split: wall_type,
            'N_Properties': len(portfolio_summary),
            'Cost_per_Ton_Gas_Mean': portfolio_summary['Cost_per_Ton_Gas_P50'].replace([np.inf, -np.inf], np.nan).mean(),
            'Cost_per_Ton_Gas_Std': portfolio_summary['Cost_per_Ton_Gas_P50'].replace([np.inf, -np.inf], np.nan).std(),
            'Cost_per_Net_Ton_Mean': portfolio_summary['Cost_per_Net_Ton_P50'].replace([np.inf, -np.inf], np.nan).mean(),
            'Cost_per_Net_Ton_Std': portfolio_summary['Cost_per_Net_Ton_P50'].replace([np.inf, -np.inf], np.nan).std(),
            'Total_Costs_Mean': portfolio_summary['Total_Costs_P50'].mean(),
            'Total_Costs_Std': portfolio_summary['Total_Costs_P50'].std(),
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    csv_filename = f'{name_suffix}_{input_col_split}_comparison.csv'
    summary_df.to_csv(
        os.path.join(output_dir, csv_filename),
        index=False
    )
    print(f"\nSaved comparison summary: {output_dir}/{csv_filename}")
    
    # Print summary
    print(f"\n--- {input_col_split.replace('_', ' ').title()} Comparison Summary ---")
    print(summary_df.to_string(index=False))
    
    return summary_df



def compare_by_wall_type_2d(df_processed, scenario_name, measure_type, output_dir, name_suffix, 
                        input_col_split='inferred_insulation_type',
                        cost_scenario_col='epistemic__cost_scenario'):
    """
    Compare costs across different wall types and cost scenarios using combined color/pattern coding.
    
    Parameters:
    -----------
    input_col_split : str
        Column name to split the analysis by (default: 'inferred_insulation_type')
    cost_scenario_col : str
        Column name for cost scenario split (default: 'epistemic__cost_scenario')
    
    Visual Encoding:
    ---------------
    - Colors: Wall types (consistent across all scenarios)
    - Line styles: Cost scenarios (solid, dashed, dotted, dashdot)
    - Hatching: Cost scenarios (for bar patterns)
    """
    print("\n" + "="*60)
    print(f"WALL TYPE × COST SCENARIO COMPARISON - {scenario_name}")
    print("="*60)
    
    # Check if split columns exist
    if input_col_split not in df_processed.columns:
        print(f"Warning: Column '{input_col_split}' not found. Skipping.")
        return None
    
    if cost_scenario_col not in df_processed.columns:
        print(f"Warning: Column '{cost_scenario_col}' not found. Will proceed without cost scenario split.")
        cost_scenarios = [None]
    else:
        cost_scenarios = df_processed[cost_scenario_col].unique()
        cost_scenarios = [cs for cs in cost_scenarios if pd.notna(cs)]
        print(f"\nFound {len(cost_scenarios)} cost scenarios: {cost_scenarios}")
    
    # Column definitions (same as before)
    cost_p50 = f'{scenario_name}_cost_{scenario_name}_p50'
    cost_p95 = f'{scenario_name}_cost_{scenario_name}_p95'
    cost_p50_m = f'{scenario_name}_cost_{scenario_name}_p50_mill'
    cost_p95_m = f'{scenario_name}_cost_{scenario_name}_p95_mill'
    
    base = f'total_tonne_co2_saved_{measure_type}_5yr'
    GAS_P50_COL = f'gas_{base}_p50'
    GAS_P95_COL = f'gas_{base}_p95'
    ELEC_P50_COL = f'elec_{base}_p50'
    has_elec = ELEC_P50_COL in df_processed.columns
    net_co2_col_p50 = f'total_tonne_co2_saved_{measure_type}_5yr_p50'
    net_co2_col_p95 = f'total_tonne_co2_saved_{measure_type}_5yr_p95'

    agg_dict = {
        cost_p50: 'sum', cost_p95: 'sum',
        GAS_P50_COL: 'sum', GAS_P95_COL: 'sum',
        cost_p50_m: 'sum', cost_p95_m: 'sum',
        net_co2_col_p50: 'sum', net_co2_col_p95: 'sum',
    }
    if has_elec:
        ELEC_P95_COL = f'elec_{base}_p95'
        agg_dict[ELEC_P50_COL] = 'sum'
        agg_dict[ELEC_P95_COL] = 'sum'
    
    # Get wall types
    wall_types = df_processed[input_col_split].unique()
    wall_types = [wt for wt in wall_types if pd.notna(wt)]
    print(f"\nFound {len(wall_types)} wall types: {wall_types}")
    
    # Nested data structure
    nested_data = {}
    
    for cost_scenario in cost_scenarios:
        nested_data[cost_scenario] = {}
        
        if cost_scenario is not None:
            df_scenario = df_processed[df_processed[cost_scenario_col] == cost_scenario]
        else:
            df_scenario = df_processed.copy()
        
        for wall_type in wall_types:
            df_wall = df_scenario[df_scenario[input_col_split] == wall_type]
            
            if len(df_wall) == 0:
                continue
            
            portfolio_summary = df_wall.groupby('epistemic_run_id').agg(agg_dict).reset_index()
            
            # Calculate metrics
            portfolio_summary['Cost_per_Ton_Gas_P50'] = (
                portfolio_summary[cost_p50] / portfolio_summary[GAS_P50_COL]
            )
            portfolio_summary['Cost_per_Ton_Gas_P95'] = (
                portfolio_summary[cost_p95] / portfolio_summary[GAS_P95_COL]
            )
            portfolio_summary['Total_Net_CO2_P50'] = portfolio_summary[net_co2_col_p50]
            portfolio_summary['Total_Net_CO2_P95'] = portfolio_summary[net_co2_col_p95]
            portfolio_summary['Cost_per_Net_Ton_P50'] = (
                portfolio_summary[cost_p50] / portfolio_summary['Total_Net_CO2_P50']
            )
            portfolio_summary['Cost_per_Net_Ton_P95'] = (
                portfolio_summary[cost_p95] / portfolio_summary['Total_Net_CO2_P95']
            )
            portfolio_summary['Total_Costs_P50'] = portfolio_summary[cost_p50_m]
            portfolio_summary['Total_Costs_P95'] = portfolio_summary[cost_p95_m]
            
            nested_data[cost_scenario][wall_type] = portfolio_summary
    
    # ========================================================================
    # VISUAL ENCODING SETUP
    # ========================================================================
    
    # Colors for wall types (consistent across scenarios)
    wall_type_colors = {}
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, wall_type in enumerate(wall_types):
        wall_type_colors[wall_type] = base_colors[idx % 10]
    
    # Line styles for cost scenarios
    line_styles = ['-', '--', '-.', ':']
    scenario_line_styles = {}
    for idx, scenario in enumerate(cost_scenarios):
        scenario_line_styles[scenario] = line_styles[idx % len(line_styles)]
    
    # Hatching patterns for cost scenarios (for histogram bars)
    hatch_patterns = ['', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo', '...', '***']
    scenario_hatches = {}
    for idx, scenario in enumerate(cost_scenarios):
        scenario_hatches[scenario] = hatch_patterns[idx % len(hatch_patterns)]
    
    # Alpha adjustments for better visibility
    base_alpha = 0.3 if len(cost_scenarios) > 2 else 0.4
    
    # ========================================================================
    # CREATE COMBINED PLOTS
    # ========================================================================
    
    metrics_to_plot = [
        ('Cost_per_Ton_Gas_P50', 'Cost per Ton Gas CO2 (£/TON)', 'cost_per_ton_gas_p50'),
        ('Cost_per_Net_Ton_P50', 'Cost per Net Ton CO2 (£/TON)', 'cost_per_net_ton_p50'),
        ('Total_Costs_P50', 'Total Costs (£M)', 'total_costs_p50'),
    ]
    
    for metric_col, metric_label, file_suffix in metrics_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Track legend entries to avoid duplicates
        legend_entries = []
        legend_labels = []
        
        for cost_scenario in cost_scenarios:
            scenario_data = nested_data[cost_scenario]
            scenario_label = cost_scenario if cost_scenario is not None else 'All'
            
            for wall_type in wall_types:
                if wall_type not in scenario_data:
                    continue
                    
                portfolio_summary = scenario_data[wall_type]
                
                # Skip if no valid data
                valid_data = portfolio_summary[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) == 0:
                    continue
                
                # Combination label
                combo_label = f'{wall_type} | {scenario_label}'
                
                # Get color and style
                color = wall_type_colors[wall_type]
                linestyle = scenario_line_styles[cost_scenario]
                hatch = scenario_hatches[cost_scenario]
                
                # Plot histogram with both color (wall type) and hatch (cost scenario)
                n, bins, patches = ax.hist(
                    valid_data,
                    bins=15,
                    alpha=base_alpha,
                    color=color,
                    edgecolor=color,
                    linewidth=1.5,
                    label=combo_label,
                    density=True
                )
                
                # Apply hatching pattern to bars
                for patch in patches:
                    patch.set_hatch(hatch)
                
                # Plot KDE line with matching color and line style
                from scipy import stats
                kde = stats.gaussian_kde(valid_data)
                x_range = np.linspace(valid_data.min(), valid_data.max(), 200)
                kde_values = kde(x_range)
                
                line = ax.plot(
                    x_range,
                    kde_values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    alpha=0.9
                )[0]
                
                # Add mean line
                mean_val = valid_data.mean()
                ax.axvline(
                    mean_val,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.7
                )
                
                # Store for legend
                legend_entries.append(line)
                legend_labels.append(f'{combo_label} (μ={mean_val:,.0f}, n={len(valid_data)})')
        
        # Title and labels
        ax.set_title(
            f'{input_col_split.replace("_", " ").title()} × Cost Scenario: {metric_label}\n'
            f'({scenario_name})\n'
            f'Color = Wall Type | Line Style = Cost Scenario',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.set_xlabel(metric_label, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        
        # Enhanced legend with visual guide
        # Split legend into two columns if many combinations
        n_combos = len(legend_entries)
        ncol = 2 if n_combos > 6 else 1
        
        legend = ax.legend(
            legend_entries,
            legend_labels,
            loc='upper right',
            fontsize=8,
            ncol=ncol,
            framealpha=0.95,
            edgecolor='black',
            title='Wall Type | Cost Scenario (mean, count)',
            title_fontsize=9
        )
        
        # Add visual guide for encoding
        guide_text = (
            'Visual Encoding:\n'
            f'• Colors = Wall Types: {", ".join(wall_types)}\n'
            f'• Line Styles = Scenarios: {", ".join([str(s) if s else "All" for s in cost_scenarios])}'
        )
        ax.text(
            0.02, 0.98, guide_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{name_suffix}_{input_col_split}_by_cost_scenario_combined_{file_suffix}.png'
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print(f"Saved: {output_dir}/{filename}")
    
    # ========================================================================
    # ALTERNATIVE: Separate Legend Visualization
    # ========================================================================
    # Create a separate figure showing the encoding scheme
    fig_legend, (ax_color, ax_style) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Wall type color legend
    ax_color.set_title('Wall Type Colors', fontsize=12, fontweight='bold')
    for idx, (wall_type, color) in enumerate(wall_type_colors.items()):
        ax_color.barh(idx, 1, color=color, alpha=0.7, edgecolor='black')
        ax_color.text(0.5, idx, wall_type, ha='center', va='center', fontsize=10, fontweight='bold')
    ax_color.set_xlim(0, 1)
    ax_color.set_ylim(-0.5, len(wall_types) - 0.5)
    ax_color.axis('off')
    
    # Cost scenario line style legend
    ax_style.set_title('Cost Scenario Line Styles', fontsize=12, fontweight='bold')
    for idx, (scenario, linestyle) in enumerate(scenario_line_styles.items()):
        scenario_label = scenario if scenario is not None else 'All'
        ax_style.plot([0, 1], [idx, idx], linestyle=linestyle, linewidth=3, color='black')
        ax_style.text(1.1, idx, f'{scenario_label} ({hatch_patterns[idx]})', 
                     va='center', fontsize=10)
    ax_style.set_xlim(0, 1)
    ax_style.set_ylim(-0.5, len(cost_scenarios) - 0.5)
    ax_style.axis('off')
    
    plt.tight_layout()
    legend_filename = f'{name_suffix}_{input_col_split}_encoding_legend.png'
    plt.savefig(
        os.path.join(output_dir, legend_filename),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    print(f"Saved encoding legend: {output_dir}/{legend_filename}")
    
    # ========================================================================
    # SUMMARY STATISTICS TABLE
    # ========================================================================
    summary_stats = []
    for cost_scenario in cost_scenarios:
        for wall_type, portfolio_summary in nested_data[cost_scenario].items():
            stats = {
                'cost_scenario': cost_scenario if cost_scenario is not None else 'All',
                input_col_split: wall_type,
                'N_Properties': len(portfolio_summary),
                'Cost_per_Ton_Gas_Mean': portfolio_summary['Cost_per_Ton_Gas_P50'].replace([np.inf, -np.inf], np.nan).mean(),
                'Cost_per_Ton_Gas_Std': portfolio_summary['Cost_per_Ton_Gas_P50'].replace([np.inf, -np.inf], np.nan).std(),
                'Cost_per_Net_Ton_Mean': portfolio_summary['Cost_per_Net_Ton_P50'].replace([np.inf, -np.inf], np.nan).mean(),
                'Cost_per_Net_Ton_Std': portfolio_summary['Cost_per_Net_Ton_P50'].replace([np.inf, -np.inf], np.nan).std(),
                'Total_Costs_Mean': portfolio_summary['Total_Costs_P50'].mean(),
                'Total_Costs_Std': portfolio_summary['Total_Costs_P50'].std(),
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    csv_filename = f'{name_suffix}_{input_col_split}_by_cost_scenario_comparison.csv'
    summary_df.to_csv(
        os.path.join(output_dir, csv_filename),
        index=False
    )
    print(f"\nSaved comparison summary: {output_dir}/{csv_filename}")
    
    print(f"\n--- {input_col_split.replace('_', ' ').title()} × Cost Scenario Summary ---")
    print(summary_df.to_string(index=False))
    
    return summary_df