"""
Retrofit Scenario Analysis Script - Memory Optimized Version
Analyzes cost-effectiveness vs total impact strategies for building retrofits
Split by cost scenarios with optimized memory usage
"""

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import glob
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add custom module path
sys.path.append('/rds/user/gb669/hpc-work/energy_map/RetrofitModel')
from src.validate import validate
from src.RetrofitPostProcess import process_multiple_scenarios

# ============================================================================
# CONFIGURATION
# ============================================================================
# Input/output paths
DATA_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/all_v2/NE/*.csv'
BASE_OUTPUT_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/retrofit_scenario_analysis/2_wrap_up/all_v2'

# Analysis parameters
YEARS = 5
N_SIMULATIONS = 5000
ELEC_CARBON_FACTOR = 0.2
GAS_CARBON_FACTOR = 0.2

# Scenarios to analyze
SCENARIOS_CONFIG = [
    ("wall_installation", "wall_installation"),
    ("loft_installation", "loft_installation"),
    ("join_heat_ins_decay", "join_heat_ins_decay"),
    ("heat_pump_only", "heat_pump_only")
]
SCENARIO_LIST = ['wall_installation', 'loft_installation', 'join_heat_ins_decay', 'heat_pump_only']
EPISTEMIC_COL = 'epistemic_run_id'
COST_SCENARIO_COL = 'epistemic__cost_scenario'

# Memory optimization settings
CHUNK_SIZE = 100000  # Process CSV files in chunks if they're large
USE_CATEGORICAL = True  # Convert string columns to categorical to save memory

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

from src.RetrofitAnalysisUtils import load_data, memory_profiler, log_memory


# ============================================================================
# MEMORY OPTIMIZATION UTILITIES
# ============================================================================
def optimize_dtypes(df):
    """
    Optimize dataframe memory usage by downcasting numeric types
    and converting object columns to categorical where appropriate
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert low-cardinality object columns to categorical
    if USE_CATEGORICAL:
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df)
            if num_unique / num_total < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Memory optimized: {initial_memory:.2f} MB -> {final_memory:.2f} MB "
          f"(saved {initial_memory - final_memory:.2f} MB, {(1 - final_memory/initial_memory)*100:.1f}%)")
    
    return df


def report_memory():
    """Report current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"  Current memory usage: {mem_info.rss / 1024**2:.2f} MB")


# ============================================================================
# DATA LOADING WITH MEMORY OPTIMIZATION
# ============================================================================
print("Loading data files with memory optimization...")
print(f"  Found {len(glob.glob(DATA_PATH))} CSV files")

res_df = load_data(DATA_PATH, SCENARIO_LIST )
 


gc.collect()

print(f"  Loaded data: {len(res_df)} rows, {len(res_df.columns)} columns")
print(f"  Memory usage before optimization: {res_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize memory before processing
#res_df = optimize_dtypes(res_df)

print("Processing scenarios...")
proc_df = process_multiple_scenarios(
    res_df, 
    SCENARIOS_CONFIG, 
    YEARS, 
    N_SIMULATIONS,
    GAS_CARBON_FACTOR, 
    ELEC_CARBON_FACTOR, 
    gas_col='deriv'
)

# Clear original dataframe
del res_df
gc.collect()

print("Filtering data...")
proc_df = proc_df[proc_df['premise_use'] != 'Domestic_outbuilding'].copy()
proc_df = proc_df[~proc_df['premise_type'].isna()]

# Optimize filtered dataframe
proc_df = optimize_dtypes(proc_df)
report_memory()

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def run_two_strategy_analysis(wdf, epi_run):
    """
    Performs two optimization strategies - Memory efficient version
    """
    df_positive = wdf.copy()
    df_positive['actual_total_ton_co2_saved'] = df_positive['total_ton_co2_saved']
    
    if df_positive.empty:
        return None

    # Strategy 1: Pure Cost-Effectiveness
    strategy1 = (df_positive
                 .sort_values('cost_per_net_ton_co2_kg', ascending=True)
                 .drop_duplicates(subset='upn', keep='first'))
    
    # Strategy 2: Pure Impact
    strategy2 = (df_positive
                 .sort_values('actual_total_ton_co2_saved', ascending=False)
                 .drop_duplicates(subset='upn', keep='first'))

    # Comparison table
    comparison_data = {
        'epistemic_run': [epi_run, epi_run],
        'Strategy': ['Cost-Effectiveness', 'Total Impact'],
        'Total_CO2_tons': [
            strategy1['actual_total_ton_co2_saved'].sum(),
            strategy2['actual_total_ton_co2_saved'].sum()
        ],
        'Avg_CO2_per_building': [
            strategy1['actual_total_ton_co2_saved'].mean(),
            strategy2['actual_total_ton_co2_saved'].mean()
        ],
        'Avg_Cost_per_kg': [
            strategy1['cost_per_net_ton_co2_kg'].mean(),
            strategy2['cost_per_net_ton_co2_kg'].mean()
        ],
        'Buildings_Covered': [
            len(strategy1),
            len(strategy2)
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    
    # Clean up
    del df_positive
    
    return comparison_df, strategy1, strategy2


def prepare_visualization_data(selections_dict, strategy_name):
    """
    Combines all epistemic runs and adds decile information - Memory efficient
    """
    all_data = []
    
    for epi_run, df in selections_dict.items():
        df_copy = df[['upn', 'avg_gas_percentile', 'cost', 'total_ton_co2_saved', 
                      'cost_per_net_ton_co2_kg', 'scenario']].copy()
        
        # Create deciles from percentiles (1-10)
        df_copy['avg_gas_decile'] = pd.cut(
            df_copy['avg_gas_percentile'], 
            bins=10, 
            labels=range(1, 11),
            include_lowest=True
        ).astype(np.int8)  # Use int8 instead of int
        df_copy['strategy'] = strategy_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()
    
    return optimize_dtypes(combined_df)


def run_analysis_for_epistemic_runs(df_subset, output_path):
    """
    Run the main analysis loop - Memory optimized
    """
    epistemic_runs = df_subset[EPISTEMIC_COL].unique()

    # Store results
    all_epistemic_results = []
    all_cost_eff_selections = {}
    all_impact_selections = {}

    print(f"  Processing {len(epistemic_runs)} epistemic runs...")
    
    for idx, epi_run in enumerate(epistemic_runs):
        if idx % 10 == 0 and idx > 0:
            print(f"    Processed {idx}/{len(epistemic_runs)} epistemic runs...")
            gc.collect()  # Periodic garbage collection
        
        # Filter data for this epistemic run - only keep necessary columns
        necessary_cols = ['upn', 'avg_gas_percentile', EPISTEMIC_COL]
        
        # Add scenario-specific columns dynamically
        for scenario in SCENARIO_LIST:
            necessary_cols.extend([
                f'total_tonne_co2_saved_{scenario}_5yr_mean',
                f'cost_per_net_ton_co2_{scenario}_mean',
                f'{scenario}_cost_{scenario}_mean'
            ])
        
        # Filter to only necessary columns that exist
        available_cols = [col for col in necessary_cols if col in df_subset.columns]
        epi_df = df_subset[df_subset[EPISTEMIC_COL] == epi_run][available_cols].copy()
        
        res = []
        
        for scenario in SCENARIO_LIST:
            col_to_check = f'total_tonne_co2_saved_{scenario}_5yr_mean'
            cost_col = f'cost_per_net_ton_co2_{scenario}_mean'
            cost_absolute_col = f'{scenario}_cost_{scenario}_mean'
            
            if col_to_check not in epi_df.columns:
                continue
            
            # Extract relevant columns for this scenario
            scenario_df = epi_df[[
                'upn', 
                'avg_gas_percentile',
                cost_absolute_col,
                col_to_check,
                cost_col
            ]].copy()
            
            # Filter: Only keep rows where CO2 is NEGATIVE (saves CO2)
            scenario_df = scenario_df[scenario_df[col_to_check] < 0].copy()
            
            if scenario_df.empty:
                continue
            
            # Flip signs for optimization
            scenario_df['total_ton_co2_saved'] = -scenario_df[col_to_check].astype(np.float32)
            scenario_df['cost_per_net_ton_co2_kg'] = -scenario_df[cost_col].astype(np.float32)
            
            # Rename and add metadata
            scenario_df = scenario_df.rename(columns={cost_absolute_col: 'cost'})
            scenario_df['scenario'] = scenario
            scenario_df['epistemic_run'] = epi_run
            
            # Keep only needed columns
            scenario_df = scenario_df[[
                'upn', 'avg_gas_percentile', 'cost', 
                'total_ton_co2_saved', 'cost_per_net_ton_co2_kg',
                'scenario', 'epistemic_run'
            ]]
            
            res.append(scenario_df)
        
        # Clean up
        del epi_df
        
        if not res:
            continue
        
        # Combine all scenarios for this run
        res_df = pd.concat(res, ignore_index=True)
        del res
        
        # Filter out rows with NaN cost effectiveness
        wdf = res_df[~res_df['cost_per_net_ton_co2_kg'].isna()].copy()
        del res_df
        
        if wdf.empty:
            continue

        # Run analysis
        result = run_two_strategy_analysis(wdf, epi_run)
        del wdf
        
        if result is None:
            continue
            
        comparison_df, cost_eff_selection, impact_selection = result
        
        all_epistemic_results.append(comparison_df)
        all_cost_eff_selections[epi_run] = cost_eff_selection
        all_impact_selections[epi_run] = impact_selection

    return all_epistemic_results, all_cost_eff_selections, all_impact_selections


def generate_visualizations(all_data, summary_by_decile, output_path):
    """
    Generate all visualizations - Memory efficient (close plots immediately)
    """
    cost_eff_summary = summary_by_decile[summary_by_decile['strategy'] == 'Cost-Effectiveness']
    impact_summary = summary_by_decile[summary_by_decile['strategy'] == 'Total Impact']
    
    # ============================================================================
    # VISUALIZATION 1: MAIN COMPARISON BY DECILE
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Retrofit Analysis: Cost-Effectiveness vs Total Impact Strategy', 
                 fontsize=16, fontweight='bold')

    # Plot 1: CO2 Savings (Cost-Effectiveness)
    ax1 = axes[0, 0]
    pivot_co2_ce = cost_eff_summary.pivot(index='decile', columns='scenario', values='total_co2_saved')
    pivot_co2_ce.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('CO2 Savings by Decile - Cost-Effectiveness Strategy', fontweight='bold')
    ax1.set_xlabel('Gas Usage Decile (1=lowest, 10=highest)')
    ax1.set_ylabel('Total CO2 Saved (tons)')
    ax1.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    # Plot 2: CO2 Savings (Total Impact)
    ax2 = axes[0, 1]
    pivot_co2_ti = impact_summary.pivot(index='decile', columns='scenario', values='total_co2_saved')
    pivot_co2_ti.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('CO2 Savings by Decile - Total Impact Strategy', fontweight='bold')
    ax2.set_xlabel('Gas Usage Decile (1=lowest, 10=highest)')
    ax2.set_ylabel('Total CO2 Saved (tons)')
    ax2.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    # Plot 3: Total Costs (Cost-Effectiveness)
    ax3 = axes[1, 0]
    pivot_cost_ce = cost_eff_summary.pivot(index='decile', columns='scenario', values='total_cost')
    pivot_cost_ce.plot(kind='bar', ax=ax3, width=0.8, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    ax3.set_title('Total Costs by Decile - Cost-Effectiveness Strategy', fontweight='bold')
    ax3.set_xlabel('Gas Usage Decile (1=lowest, 10=highest)')
    ax3.set_ylabel('Total Cost (£)')
    ax3.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

    # Plot 4: Total Costs (Total Impact)
    ax4 = axes[1, 1]
    pivot_cost_ti = impact_summary.pivot(index='decile', columns='scenario', values='total_cost')
    pivot_cost_ti.plot(kind='bar', ax=ax4, width=0.8, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    ax4.set_title('Total Costs by Decile - Total Impact Strategy', fontweight='bold')
    ax4.set_xlabel('Gas Usage Decile (1=lowest, 10=highest)')
    ax4.set_ylabel('Total Cost (£)')
    ax4.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_path}/retrofit_comparison_by_decile.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: retrofit_comparison_by_decile.png")
    plt.close(fig)
    del fig, axes
    gc.collect()

    # ============================================================================
    # VISUALIZATION 2: INTERVENTION DISTRIBUTION
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Distribution of Interventions by Decile', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    pivot_buildings_ce = cost_eff_summary.pivot(index='decile', columns='scenario', values='num_buildings')
    pivot_buildings_ce.plot(kind='bar', ax=ax1, width=0.8, stacked=True)
    ax1.set_title('Cost-Effectiveness Strategy', fontweight='bold')
    ax1.set_xlabel('Gas Usage Decile')
    ax1.set_ylabel('Number of Buildings')
    ax1.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    ax2 = axes[1]
    pivot_buildings_ti = impact_summary.pivot(index='decile', columns='scenario', values='num_buildings')
    pivot_buildings_ti.plot(kind='bar', ax=ax2, width=0.8, stacked=True)
    ax2.set_title('Total Impact Strategy', fontweight='bold')
    ax2.set_xlabel('Gas Usage Decile')
    ax2.set_ylabel('Number of Buildings')
    ax2.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_path}/intervention_distribution_by_decile.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: intervention_distribution_by_decile.png")
    plt.close(fig)
    del fig, axes
    gc.collect()

    # ============================================================================
    # VISUALIZATION 3: HEATMAPS
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Heatmap Analysis: Scenario Performance by Decile', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    sns.heatmap(pivot_co2_ce, annot=True, fmt='.0f', cmap='YlGn', ax=ax1, cbar_kws={'label': 'CO2 Saved (tons)'})
    ax1.set_title('CO2 Savings - Cost-Effectiveness', fontweight='bold')
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Decile')

    ax2 = axes[0, 1]
    sns.heatmap(pivot_co2_ti, annot=True, fmt='.0f', cmap='YlGn', ax=ax2, cbar_kws={'label': 'CO2 Saved (tons)'})
    ax2.set_title('CO2 Savings - Total Impact', fontweight='bold')
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Decile')

    ax3 = axes[1, 0]
    sns.heatmap(pivot_cost_ce/1000, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Cost (£k)'})
    ax3.set_title('Total Costs - Cost-Effectiveness', fontweight='bold')
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Decile')

    ax4 = axes[1, 1]
    sns.heatmap(pivot_cost_ti/1000, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Cost (£k)'})
    ax4.set_title('Total Costs - Total Impact', fontweight='bold')
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Decile')

    plt.tight_layout()
    plt.savefig(f'{output_path}/scenario_heatmaps_by_decile.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: scenario_heatmaps_by_decile.png")
    plt.close(fig)
    del fig, axes
    gc.collect()

    # ============================================================================
    # VISUALIZATION 4: SCATTER
    # ============================================================================
    fig, ax = plt.subplots(figsize=(14, 8))

    for strategy in ['Cost-Effectiveness', 'Total Impact']:
        strategy_data = summary_by_decile[summary_by_decile['strategy'] == strategy]
        strategy_summary = strategy_data.groupby('scenario').agg({
            'total_cost': 'sum',
            'total_co2_saved': 'sum'
        }).reset_index()
        
        strategy_summary['cost_per_ton'] = strategy_summary['total_cost'] / strategy_summary['total_co2_saved']
        
        marker = 'o' if strategy == 'Cost-Effectiveness' else 's'
        ax.scatter(strategy_summary['total_co2_saved'], 
                   strategy_summary['cost_per_ton'],
                   s=200, alpha=0.6, marker=marker, label=strategy)
        
        for idx, row in strategy_summary.iterrows():
            ax.annotate(row['scenario'], 
                       (row['total_co2_saved'], row['cost_per_ton']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Total CO2 Saved (tons)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost per Ton CO2 (£/ton)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Effectiveness Analysis: Scenarios by Strategy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_path}/cost_effectiveness_scatter.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: cost_effectiveness_scatter.png")
    plt.close(fig)
    del fig, ax
    gc.collect()


# ============================================================================
# MAIN ANALYSIS - SPLIT BY COST SCENARIO
# ============================================================================
print("\n" + "=" * 80)
print("STARTING MEMORY-OPTIMIZED ANALYSIS")
print("=" * 80)

# Check if cost scenario column exists
if COST_SCENARIO_COL not in proc_df.columns:
    print(f"ERROR: Column '{COST_SCENARIO_COL}' not found in dataframe!")
    print(f"Available columns: {proc_df.columns.tolist()}")
    exit()

# Get unique cost scenarios
cost_scenarios = proc_df[COST_SCENARIO_COL].unique()
print(f"\nFound {len(cost_scenarios)} cost scenarios: {cost_scenarios}")
report_memory()

# Loop through each cost scenario
for cost_scenario in cost_scenarios:
    print("\n" + "=" * 80)
    print(f"PROCESSING COST SCENARIO: {cost_scenario}")
    print("=" * 80)
    
    # Create output directory
    output_path = os.path.join(BASE_OUTPUT_PATH, f'cost_scenario_{cost_scenario}')
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Filter data for this cost scenario
    df_cost_scenario = proc_df[proc_df[COST_SCENARIO_COL] == cost_scenario].copy()
    print(f"Data size: {len(df_cost_scenario)} rows")
    report_memory()
    
    # Run analysis
    all_epistemic_results, all_cost_eff_selections, all_impact_selections = run_analysis_for_epistemic_runs(
        df_cost_scenario, output_path
    )
    
    # Clear the cost scenario dataframe immediately
    del df_cost_scenario
    gc.collect()
    
    # ============================================================================
    # RESULTS AND VISUALIZATION
    # ============================================================================
    if all_epistemic_results:
        final_comparison_df = pd.concat(all_epistemic_results, ignore_index=True)
        
        print("\n" + "-" * 80)
        print(f"COMPARISON FOR COST SCENARIO: {cost_scenario}")
        print("-" * 80)
        summary_stats = final_comparison_df.groupby('Strategy').agg(
            Avg_Total_CO2_tons=('Total_CO2_tons', 'mean'),
            Avg_Avg_CO2_per_building=('Avg_CO2_per_building', 'mean'),
            Avg_Avg_Cost_per_kg=('Avg_Cost_per_kg', 'mean'),
            Avg_Buildings_Covered=('Buildings_Covered', 'mean')
        )
        print(summary_stats)
        
        # Prepare visualization data
        print("\n  Preparing visualization data...")
        cost_eff_data = prepare_visualization_data(all_cost_eff_selections, 'Cost-Effectiveness')
        impact_data = prepare_visualization_data(all_impact_selections, 'Total Impact')
        all_data = pd.concat([cost_eff_data, impact_data], ignore_index=True)
        
        # Clear intermediate data
        del cost_eff_data, impact_data
        gc.collect()

        print(f"  Total records: {len(all_data)}")
        
        # Aggregate by strategy, scenario, and decile
        summary_by_decile = all_data.groupby(
            ['strategy', 'scenario', 'avg_gas_decile']
        ).agg({
            'cost': 'sum',
            'total_ton_co2_saved': 'sum',
            'upn': 'count',
            'cost_per_net_ton_co2_kg': 'mean'
        }).reset_index()

        summary_by_decile.columns = [
            'strategy', 'scenario', 'decile', 
            'total_cost', 'total_co2_saved', 'num_buildings', 'avg_cost_per_kg'
        ]

        # Generate visualizations
        print("\n  Generating visualizations...")
        generate_visualizations(all_data, summary_by_decile, output_path)

        # Export summary tables
        print("\n  Exporting summary tables...")
        overall_summary = all_data.groupby(['strategy', 'scenario']).agg({
            'cost': 'sum',
            'total_ton_co2_saved': 'sum',
            'upn': 'count',
            'cost_per_net_ton_co2_kg': 'mean'
        }).reset_index()

        overall_summary.columns = [
            'Strategy', 'Scenario', 'Total_Cost', 'Total_CO2_Saved', 
            'Num_Buildings', 'Avg_Cost_per_kg'
        ]
        overall_summary['Cost_per_ton_CO2'] = overall_summary['Total_Cost'] / overall_summary['Total_CO2_Saved']

        summary_by_decile.to_csv(f'{output_path}/detailed_summary_by_decile.csv', index=False)
        print("  ✓ Saved: detailed_summary_by_decile.csv")

        overall_summary.to_csv(f'{output_path}/overall_summary_by_strategy.csv', index=False)
        print("  ✓ Saved: overall_summary_by_strategy.csv")
        
        # Clean up after this cost scenario
        del all_data, summary_by_decile, overall_summary, final_comparison_df
        del all_epistemic_results, all_cost_eff_selections, all_impact_selections
        gc.collect()
        
        print(f"\n  ✓ Completed cost scenario: {cost_scenario}")
        report_memory()
    else:
        print(f"\n  ✗ No valid results for cost scenario: {cost_scenario}")

# Final cleanup
del proc_df
gc.collect()

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================
print("\n" + "=" * 80)
print("ALL COST SCENARIOS COMPLETED!")
print("=" * 80)
print(f"\nResults saved in: {BASE_OUTPUT_PATH}")
report_memory()