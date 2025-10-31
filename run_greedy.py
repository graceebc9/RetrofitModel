

import pandas as pd 
import glob 
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import sys
sys.path.append('/Users/gracecolverd/RetrofitModel')

 

import numpy as np 

from src.RetrofitPostProcess import process_multiple_scenarios 
from src.RetrofitAnalysisUtils import load_data 
import glob 



from src.RetrofitPostProcess import process_multiple_scenarios 
from src.GreedyAlgo import true_greedy_knapsack, plot_greedy_distribution_analysis

def assign_random_loft(df, prob_loft):
 
    modern_ages = ['Post 1999']
    #  = 0.65

    # For modern ages: True, for non-modern ages: random with 65% probability
    df['already_loft'] = np.where(
        df['premise_age'].isin(modern_ages),
        True,
        np.random.random(len(df)) < prob_loft
    )
    return df 

import pandas as pd
import numpy as np
from datetime import datetime
import logging

def run_greedy_algo_(scenario_budget, prob_loft, df ):


    # Identify epistemic run column
    epistemic_col = 'epistemic_run_id'
    epistemic_runs = df[epistemic_col].unique()

    logging.debug(f"Starting analysis with {len(epistemic_runs)} epistemic runs and {len(scenario_list)} scenarios")
    logging.debug(f"Total buildings in dataset: {len(df)}")
    logging.debug(f"Scenarios: {', '.join(scenario_list)}")

    # Store results for each epistemic run AND scenario
    all_epistemic_results = []
    scenario_exclusion_stats = []
    scenario_performance_log = []

    # Loop through each epistemic run
    for epi_idx, epi_run in enumerate(epistemic_runs, 1):
        logging.debug(f"\n{'='*80}")
        logging.debug(f"Processing epistemic run {epi_idx}/{len(epistemic_runs)}: {epi_run}")
        logging.debug(f"{'='*80}")
        
        # Filter data for this epistemic run
        epi_df = df[df[epistemic_col] == epi_run].copy()
        
        # make 65% have loft insulation already 
        epi_df = assign_random_loft(epi_df, prob_loft )
        logging.debug(f"Buildings in this epistemic run: {len(epi_df)}")
        
        res = []
        epi_scenario_stats = []
        
        for scenario_idx, scenario in enumerate(scenario_list, 1):
            logging.debug(f"\n  --- Scenario {scenario_idx}/{len(scenario_list)}: {scenario} ---")
            
            # Exclude UPNs with positive values for THIS scenario only- i think maybe this doest even matter if we rank on cost per ton 
            col_to_check = f'total_tonne_co2_saved_{scenario}_5yr_mean'
            mask = epi_df[col_to_check] > 0
            bad_upns_for_scenario = epi_df.loc[mask, 'upn'].unique().tolist()
            
            # Log exclusion statistics
            n_excluded_buildings = len(bad_upns_for_scenario)
            n_excluded_records = mask.sum()
            pct_excluded = (n_excluded_records / len(epi_df)) * 100 if len(epi_df) > 0 else 0
            
            logging.debug(f"    Excluded {n_excluded_buildings} UPNs ({n_excluded_records} records, {pct_excluded:.1f}%) with positive CO2 savings")
            
            scenario_exclusion_stats.append({
                'epistemic_run': epi_run,
                'scenario': scenario,
                'n_upns_excluded': n_excluded_buildings,
                'n_records_excluded': n_excluded_records,
                'pct_records_excluded': pct_excluded
            })
            
            # Filter out bad UPNs for this scenario
            scenario_df = epi_df[~epi_df['upn'].isin(bad_upns_for_scenario)].copy()

            if scenario == 'loft_installation':
                logging.debug('Removing existing loft insulation')
                scenario_df=scenario_df[scenario_df['already_loft']==False]

            logging.debug(f"    Remaining buildings for {scenario}: {len(scenario_df)}")
            
            # Flip signs for optimization
            cols = [f'total_tonne_co2_saved_{scenario}_5yr_mean',
                    f'cost_per_net_ton_co2_{scenario}_mean']
            for col in cols:
                scenario_df[f'flip_sign_{col}'] = -scenario_df[col]
            
            # Keep individual building records
            df_rank = scenario_df[[
                'upn',
                'avg_gas_percentile',
                f'{scenario}_cost_{scenario}_mean',
                f'flip_sign_total_tonne_co2_saved_{scenario}_5yr_mean',
                f'flip_sign_cost_per_net_ton_co2_{scenario}_mean'
            ]].copy()
            
            df_rank['scenario'] = scenario
            df_rank['epistemic_run'] = epi_run
            df_rank.columns = ['upn', 'avg_gas_percentile', 'cost of interventon_mean',
                            'total_ton_co2_saved', 'cost_per_net_ton_co2_kg',
                            'scenario', 'epistemic_run']
            
            # Log scenario statistics before selection
            valid_scenario_data = df_rank[~df_rank['cost_per_net_ton_co2_kg'].isna()]
            if len(valid_scenario_data) > 0:
                logging.debug(f"    Valid buildings for optimization: {len(valid_scenario_data)}")
                logging.debug(f"    Cost/tonne CO2 - Min: £{valid_scenario_data['cost_per_net_ton_co2_kg'].min():.2f}, "
                            f"Median: £{valid_scenario_data['cost_per_net_ton_co2_kg'].median():.2f}, "
                            f"Mean: £{valid_scenario_data['cost_per_net_ton_co2_kg'].mean():.2f}, "
                            f"Max: £{valid_scenario_data['cost_per_net_ton_co2_kg'].max():.2f}")
                logging.debug(f"    Total potential CO2 savings: {valid_scenario_data['total_ton_co2_saved'].sum():,.0f} tonnes")
                logging.debug(f"    Total potential cost: £{valid_scenario_data['cost of interventon_mean'].sum():,.0f}")
            else:
                logging.warning(f"    No valid data for scenario {scenario}!")
            
            res.append(df_rank)
        
        # Combine all scenarios for this epistemic run
        res_df = pd.concat(res)
        wdf = res_df[~res_df['cost_per_net_ton_co2_kg'].isna()]
        
        logging.debug(f"\n  Combined dataset for epistemic run {epi_run}:")
        logging.debug(f"    Total valid building-scenario combinations: {len(wdf)}")
        logging.debug(f"    Unique buildings: {wdf['upn'].nunique()}")
        
        # Log scenario distribution
        scenario_counts = wdf.groupby('scenario').size()
        logging.debug(f"    Buildings per scenario:")
        for sc, count in scenario_counts.items():
            logging.debug(f"      {sc}: {count}")
        
        
        logging.debug(f"\n  Running greedy knapsack with budget: £{scenario_budget:,}")
        
        baseline_selection = (wdf
                            .sort_values('cost_per_net_ton_co2_kg', ascending=True)
                            .drop_duplicates(subset='upn', keep='first')
                            .reset_index(drop=True))
        
        logging.debug(f"    Candidate projects after deduplication: {len(baseline_selection)}")
        
        selected_projects_df, remaining_funds = true_greedy_knapsack(
            df_knapsack=baseline_selection,
            budget=scenario_budget,
            cost_column='cost of interventon_mean',
            efficiency_column='cost_per_net_ton_co2_kg'
        )
        
        # Detailed logging of selection results
        selected_projects_df['epistemic_run'] = epi_run
        selected_projects_df['remaining_funds'] = remaining_funds
        
        total_cost = selected_projects_df['cost of interventon_mean'].sum()
        total_co2_saved = selected_projects_df['total_ton_co2_saved'].sum()
        avg_cost_per_tonne = total_cost / total_co2_saved if total_co2_saved > 0 else 0
        
        logging.debug(f"\n  RESULTS for epistemic run {epi_run}:")
        logging.debug(f"    Projects selected: {len(selected_projects_df)}")
        logging.debug(f"    Budget used: £{total_cost:,.0f} ({(total_cost/scenario_budget)*100:.1f}%)")
        logging.debug(f"    Remaining funds: £{remaining_funds:,.0f}")
        logging.debug(f"    Total CO2 saved: {total_co2_saved:,.0f} tonnes")
        logging.debug(f"    Average cost per tonne CO2: £{avg_cost_per_tonne:.2f}")
        
        # Scenario breakdown
        scenario_breakdown = selected_projects_df.groupby('scenario').agg({
            'upn': 'count',
            'cost of interventon_mean': 'sum',
            'total_ton_co2_saved': 'sum'
        }).rename(columns={'upn': 'n_projects'})
        scenario_breakdown['avg_cost_per_tonne'] = (
            scenario_breakdown['cost of interventon_mean'] / scenario_breakdown['total_ton_co2_saved']
        )
        
        logging.debug(f"\n  Scenario breakdown:")
        for sc in scenario_breakdown.index:
            row = scenario_breakdown.loc[sc]
            logging.debug(f"    {sc}:")
            logging.debug(f"      Projects: {row['n_projects']:.0f}")
            logging.debug(f"      Cost: £{row['cost of interventon_mean']:,.0f}")
            logging.debug(f"      CO2 saved: {row['total_ton_co2_saved']:,.0f} tonnes")
            logging.debug(f"      Avg cost/tonne: £{row['avg_cost_per_tonne']:.2f}")
        
        # Store performance metrics
        for sc in scenario_breakdown.index:
            row = scenario_breakdown.loc[sc]
            scenario_performance_log.append({
                'epistemic_run': epi_run,
                'scenario': sc,
                'n_projects': row['n_projects'],
                'total_cost': row['cost of interventon_mean'],
                'total_co2_saved': row['total_ton_co2_saved'],
                'avg_cost_per_tonne': row['avg_cost_per_tonne']
            })
        
        all_epistemic_results.append(selected_projects_df)

    # ============================================================================
    # COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS
    # ============================================================================

    logging.info(f"\n\n{'='*80}")
    logging.info("COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS")
    logging.info(f"{'='*80}")

    # Aggregate results across all epistemic runs
    combined_results = pd.concat(all_epistemic_results, ignore_index=True)

    # 1. Summary by epistemic run
    logging.info("\n1. SUMMARY BY EPISTEMIC RUN")
    logging.info("-" * 80)
    summary_by_run = combined_results.groupby('epistemic_run').agg({
        'upn': 'count',
        'cost of interventon_mean': 'sum',
        'total_ton_co2_saved': 'sum',
        'remaining_funds': 'first'
    }).rename(columns={'upn': 'n_projects'})

    for epi_run in summary_by_run.index:
        row = summary_by_run.loc[epi_run]
        logging.info(f"  Run {epi_run}:")
        logging.info(f"    Projects: {row['n_projects']:.0f}")
        logging.info(f"    Total cost: £{row['cost of interventon_mean']:,.0f}")
        logging.info(f"    CO2 saved: {row['total_ton_co2_saved']:,.0f} tonnes")
        logging.info(f"    Remaining: £{row['remaining_funds']:,.0f}")

    # 2. Overall statistics
    logging.info("\n2. OVERALL STATISTICS ACROSS EPISTEMIC RUNS")
    logging.info("-" * 80)
    logging.info(f"  Mean projects selected: {summary_by_run['n_projects'].mean():.1f} ± {summary_by_run['n_projects'].std():.1f}")
    logging.info(f"  Range: {summary_by_run['n_projects'].min():.0f} - {summary_by_run['n_projects'].max():.0f}")
    logging.info(f"  Mean CO2 saved: {summary_by_run['total_ton_co2_saved'].mean():,.0f} ± {summary_by_run['total_ton_co2_saved'].std():,.0f} tonnes")
    logging.info(f"  Range: {summary_by_run['total_ton_co2_saved'].min():,.0f} - {summary_by_run['total_ton_co2_saved'].max():,.0f} tonnes")
    logging.info(f"  Mean cost: £{summary_by_run['cost of interventon_mean'].mean():,.0f} ± £{summary_by_run['cost of interventon_mean'].std():,.0f}")

    # 3. Scenario performance analysis
    logging.info("\n3. SCENARIO PERFORMANCE ANALYSIS")
    logging.info("-" * 80)
    scenario_perf_df = pd.DataFrame(scenario_performance_log)

    scenario_summary = scenario_perf_df.groupby('scenario').agg({
        'n_projects': ['mean', 'std', 'min', 'max'],
        'total_co2_saved': ['mean', 'std', 'sum'],
        'avg_cost_per_tonne': ['mean', 'std', 'min', 'max']
    })

    for scenario in scenario_list:
        if scenario in scenario_summary.index:
            logging.info(f"\n  {scenario.upper()}:")
            stats = scenario_summary.loc[scenario]
            logging.info(f"    Projects per run: {stats[('n_projects', 'mean')]:.1f} ± {stats[('n_projects', 'std')]:.1f} "
                        f"(range: {stats[('n_projects', 'min')]:.0f}-{stats[('n_projects', 'max')]:.0f})")
            logging.info(f"    CO2 saved per run: {stats[('total_co2_saved', 'mean')]:,.0f} ± {stats[('total_co2_saved', 'std')]:,.0f} tonnes")
            logging.info(f"    Total CO2 saved (all runs): {stats[('total_co2_saved', 'sum')]:,.0f} tonnes")
            logging.info(f"    Avg cost/tonne: £{stats[('avg_cost_per_tonne', 'mean')]:.2f} ± £{stats[('avg_cost_per_tonne', 'std')]:.2f}")
            logging.info(f"    Cost/tonne range: £{stats[('avg_cost_per_tonne', 'min')]:.2f} - £{stats[('avg_cost_per_tonne', 'max')]:.2f}")

    # 4. Scenario selection frequency
    logging.info("\n4. SCENARIO SELECTION FREQUENCY")
    logging.info("-" * 80)
    scenario_freq = combined_results.groupby('scenario').size()
    total_selections = scenario_freq.sum()
    for scenario in scenario_list:
        if scenario in scenario_freq.index:
            count = scenario_freq[scenario]
            pct = (count / total_selections) * 100
            logging.info(f"  {scenario}: {count} selections ({pct:.1f}%)")

    # 5. Building robustness analysis
    logging.info("\n5. BUILDING ROBUSTNESS ANALYSIS")
    logging.info("-" * 80)
    building_selection_freq = combined_results.groupby('upn').agg({
        'epistemic_run': 'count',
        'scenario': lambda x: x.mode()[0] if len(x) > 0 else None,
        'cost_per_net_ton_co2_kg': 'mean',
        'total_ton_co2_saved': 'mean'
    }).rename(columns={'epistemic_run': 'times_selected'})

    building_selection_freq['selection_rate'] = (
        building_selection_freq['times_selected'] / len(epistemic_runs)
    )

    # Robustness categories
    always_selected = (building_selection_freq['selection_rate'] == 1).sum()
    mostly_selected = ((building_selection_freq['selection_rate'] > 0.5) & 
                    (building_selection_freq['selection_rate'] < 1)).sum()
    sometimes_selected = ((building_selection_freq['selection_rate'] > 0) & 
                        (building_selection_freq['selection_rate'] <= 0.5)).sum()

    logging.debug(f"  Buildings selected in ALL {len(epistemic_runs)} runs: {always_selected}")
    logging.debug(f"  Buildings selected in >50% of runs: {mostly_selected}")
    logging.debug(f"  Buildings selected in ≤50% of runs: {sometimes_selected}")
    logging.debug(f"  Total unique buildings selected: {len(building_selection_freq)}")

    # Top 10 most robust selections
    logging.info(f"\n  Top 10 most robust buildings (selected most frequently):")
    top_buildings = building_selection_freq.nlargest(10, 'selection_rate')
    for idx, (upn, row) in enumerate(top_buildings.iterrows(), 1):
        logging.info(f"    {idx}. UPN {upn}: selected {row['times_selected']}/{len(epistemic_runs)} times "
                    f"({row['selection_rate']*100:.0f}%), scenario: {row['scenario']}, "
                    f"avg cost/tonne: £{row['cost_per_net_ton_co2_kg']:.2f}")

    # 6. Exclusion statistics analysis
    logging.info("\n6. EXCLUSION STATISTICS ANALYSIS")
    logging.info("-" * 80)
    exclusion_df = pd.DataFrame(scenario_exclusion_stats)
    exclusion_summary = exclusion_df.groupby('scenario').agg({
        'n_upns_excluded': ['mean', 'std'],
        'pct_records_excluded': ['mean', 'std']
    })

    for scenario in scenario_list:
        if scenario in exclusion_summary.index:
            stats = exclusion_summary.loc[scenario]
            logging.info(f"  {scenario}:")
            logging.info(f"    Avg UPNs excluded: {stats[('n_upns_excluded', 'mean')]:.1f} ± {stats[('n_upns_excluded', 'std')]:.1f}")
            logging.info(f"    Avg % records excluded: {stats[('pct_records_excluded', 'mean')]:.1f}% ± {stats[('pct_records_excluded', 'std')]:.1f}%")

    return baseline_selection , combined_results


YEARS=5
N_SIMULATIONS=5000 
ELEC_CARBON_FACTOR = GAS_CARBON_FACTOR = 0.2 
scenarios_config = [("wall_installation", "wall_installation") ,  ("loft_installation","loft_installation"), ("join_heat_ins_decay","join_heat_ins_decay"),   ("heat_pump_only","heat_pump_only") ]
 
scenario_list = ['wall_installation', 'loft_installation', 'join_heat_ins_decay', 'heat_pump_only']


pathp =  '/Users/gracecolverd/Downloads/all/*.csv'
# files  = glob.glob( '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/testing/NE/*.csv')

res_df = load_data(pathp, scenario_list)

proc_df = process_multiple_scenarios(res_df, scenarios_config, YEARS, N_SIMULATIONS, 
                                GAS_CARBON_FACTOR, ELEC_CARBON_FACTOR, gas_col='deriv')


# Basic filtering (no UPN exclusions yet)
pdf = proc_df[proc_df['premise_use'] != 'Domestic_outbuilding'].copy()
pdf = pdf[~pdf['premise_type'].isna()]
df = pdf.copy()

 
base_Dir = '/Users/gracecolverd/RetrofitModel/test'
import os 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'greedy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
for budget in [10_000_000, 100_000_000, 100_000_000]:
    for prob_loft in [0.65, 0.95]: 
        op_path = os.path.join(base_Dir, f'{budget}_probloft{prob_loft}')
        
        os.makedirs(op_path, exist_ok=True )
        
        logging.info(f'Starting {budget} and loft {prob_loft}')
        baseline_selection , combined_results  = run_greedy_algo_(budget, prob_loft, df )
        
        logging.info("\nGenerating visualization...")
        plot_greedy_distribution_analysis(
            baseline_df=baseline_selection,
            selected_df=combined_results,
            scenario_name=f'£{budget:,} Budget - All Epistemic Runs',
            output_dir = op_path, 
            )