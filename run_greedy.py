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


def assign_random_loft(df, prob_loft):
    """
    Assign loft insulation status to buildings.
    Modern buildings (post-1999) are assumed to have loft insulation.
    Older buildings are randomly assigned based on probability.
    
    Args:
        df: DataFrame with building data
        prob_loft: Probability that older buildings have existing loft insulation
    
    Returns:
        DataFrame with 'already_loft' column added
    """
    modern_ages = ['Post 1999']
    
    df['already_loft'] = np.where(
        df['premise_age'].isin(modern_ages),
        True,
        np.random.random(len(df)) < prob_loft
    )
    return df




def run_greedy_algo(scenario_budget, prob_loft, df, scenario_list, summary_logger, detail_logger, equity_factor, output_dir):
    """
    Run greedy knapsack algorithm across multiple epistemic runs to select optimal retrofit projects.
    
    Args:
        scenario_budget: Total budget available
        prob_loft: Probability of existing loft insulation in older buildings
        df: Input DataFrame with building and scenario data
        scenario_list: List of retrofit scenarios to consider
        summary_logger: Logger for summary statistics
        detail_logger: Logger for detailed processing information
        equity_factor: 0 for pure carbon, 1 for pure equity 
    
    Returns:
        tuple: (baseline_selection DataFrame, combined_results DataFrame)
    """
    # Identify epistemic runs
    epistemic_col = 'epistemic_run_id'
    epistemic_runs = df[epistemic_col].unique()
    
    detail_logger.info(f"Starting analysis with {len(epistemic_runs)} epistemic runs and {len(scenario_list)} scenarios")
    detail_logger.info(f"Total buildings in dataset: {len(df)}")
    detail_logger.info(f"Scenarios: {', '.join(scenario_list)}")
    detail_logger.info(f"Equity factor: {equity_factor}")
    
    # Storage for results
    all_epistemic_results = []
    scenario_exclusion_stats = []
    scenario_performance_log = []
    equity_tracking = [] 
    
    # Process each epistemic run
    for epi_idx, epi_run in enumerate(epistemic_runs, 1):
        detail_logger.info(f"\n{'='*80}")
        detail_logger.info(f"Processing epistemic run {epi_idx}/{len(epistemic_runs)}: {epi_run}")
        detail_logger.info(f"{'='*80}")
        
        # Filter data for this epistemic run
        epi_df = df[df[epistemic_col] == epi_run].copy()
        epi_df = assign_random_loft(epi_df, prob_loft)
        detail_logger.info(f"Buildings in this epistemic run: {len(epi_df)}")
        
        res = []
        
        # Process each scenario
        for scenario_idx, scenario in enumerate(scenario_list, 1):
            detail_logger.info(f"\n  --- Scenario {scenario_idx}/{len(scenario_list)}: {scenario} ---")
            
            # Exclude buildings with positive CO2 values (these increase emissions)
            col_to_check = f'total_tonne_co2_saved_{scenario}_5yr_mean'
            mask = epi_df[col_to_check] > 0
            bad_upns_for_scenario = epi_df.loc[mask, 'upn'].unique().tolist()
            
            # Log exclusion statistics
            n_excluded_buildings = len(bad_upns_for_scenario)
            n_excluded_records = mask.sum()
            pct_excluded = (n_excluded_records / len(epi_df)) * 100 if len(epi_df) > 0 else 0
            
            detail_logger.info(f"    Excluded {n_excluded_buildings} UPNs ({n_excluded_records} records, {pct_excluded:.1f}%) with positive CO2 values")
            
            scenario_exclusion_stats.append({
                'epistemic_run': epi_run,
                'scenario': scenario,
                'n_upns_excluded': n_excluded_buildings,
                'n_records_excluded': n_excluded_records,
                'pct_records_excluded': pct_excluded
            })
            
            # Filter out excluded buildings
            scenario_df = epi_df[~epi_df['upn'].isin(bad_upns_for_scenario)].copy()
            
            # For loft installation, only consider buildings without existing loft insulation
            if scenario == 'loft_installation':
                detail_logger.info('    Removing buildings with existing loft insulation')
                scenario_df = scenario_df[scenario_df['already_loft'] == False]
            
            detail_logger.info(f"    Remaining buildings for {scenario}: {len(scenario_df)}")
            
            # Flip signs for optimization
            scenario_df[f'flip_sign_total_tonne_co2_saved_{scenario}_5yr_mean'] = -scenario_df[f'total_tonne_co2_saved_{scenario}_5yr_mean']
            scenario_df[f'flip_sign_cost_per_net_ton_co2_{scenario}_mean'] = -scenario_df[f'cost_per_net_ton_co2_{scenario}_mean']
            
            # Apply equity weighting
            scenario_df['equity_weight'] = scenario_df['meta_socio_persona'].map(EQUITY_WEIGHTS)
            scenario_df[f'flip_sign_weighted_cost_per_ton_{scenario}'] = (
                scenario_df[f'flip_sign_cost_per_net_ton_co2_{scenario}_mean'] * 
                (1 + (scenario_df['equity_weight'] - 1) * equity_factor)
            )
            
            # Prepare ranking dataframe WITH PERSONA
            df_rank = scenario_df[[
                'upn',
                'meta_socio_persona',  # NEW: Include persona
                'avg_gas_percentile',
                f'{scenario}_cost_{scenario}_mean',
                f'flip_sign_total_tonne_co2_saved_{scenario}_5yr_mean',
                f'flip_sign_cost_per_net_ton_co2_{scenario}_mean',
                f'flip_sign_weighted_cost_per_ton_{scenario}'
            ]].copy()
            
            df_rank['scenario'] = scenario
            df_rank['epistemic_run'] = epi_run
            df_rank.columns = [
                'upn', 'meta_socio_persona', 'avg_gas_percentile', 
                'cost of interventon_mean', 'total_ton_co2_saved', 
                'cost_per_net_ton_co2_kg', 'weighted_cost_per_net_ton',
                'scenario', 'epistemic_run'
            ]

            # Log scenario statistics
            valid_scenario_data = df_rank[~df_rank['cost_per_net_ton_co2_kg'].isna()]
            if len(valid_scenario_data) > 0:
                detail_logger.info(f"    Valid buildings for optimization: {len(valid_scenario_data)}")
                detail_logger.info(f"    Cost/tonne CO2 - Min: £{valid_scenario_data['cost_per_net_ton_co2_kg'].min():.2f}, "
                                 f"Median: £{valid_scenario_data['cost_per_net_ton_co2_kg'].median():.2f}, "
                                 f"Mean: £{valid_scenario_data['cost_per_net_ton_co2_kg'].mean():.2f}, "
                                 f"Max: £{valid_scenario_data['cost_per_net_ton_co2_kg'].max():.2f}")
                detail_logger.info(f"    Total potential CO2 savings: {valid_scenario_data['total_ton_co2_saved'].sum():,.0f} tonnes")
                detail_logger.info(f"    Total potential cost: £{valid_scenario_data['cost of interventon_mean'].sum():,.0f}")
            else:
                detail_logger.warning(f"    No valid data for scenario {scenario}!")
            
            res.append(df_rank)
        
        # Combine all scenarios for this epistemic run
        res_df = pd.concat(res)
        wdf = res_df[~res_df['cost_per_net_ton_co2_kg'].isna()]
        
        detail_logger.info(f"\n  Combined dataset for epistemic run {epi_run}:")
        detail_logger.info(f"    Total valid building-scenario combinations: {len(wdf)}")
        detail_logger.info(f"    Unique buildings: {wdf['upn'].nunique()}")
        
        # Log scenario distribution
        scenario_counts = wdf.groupby('scenario').size()
        detail_logger.info(f"    Buildings per scenario:")
        for sc, count in scenario_counts.items():
            detail_logger.info(f"      {sc}: {count}")
        
        # Run greedy knapsack algorithm
        detail_logger.info(f"\n  Running greedy knapsack with budget: £{scenario_budget:,}")
        
        baseline_selection = (wdf
                            .sort_values('weighted_cost_per_net_ton', ascending=True)
                            .drop_duplicates(subset='upn', keep='first')
                            .reset_index(drop=True))
        
        detail_logger.info(f"    Candidate projects after deduplication: {len(baseline_selection)}")
        
        selected_projects_df, remaining_funds = true_greedy_knapsack(
            df_knapsack=baseline_selection,
            budget=scenario_budget,
            cost_column='cost of interventon_mean',
            efficiency_column='weighted_cost_per_net_ton'
        )
        
        # Log selection results
        selected_projects_df['epistemic_run'] = epi_run
        selected_projects_df['remaining_funds'] = remaining_funds
        
        total_cost = selected_projects_df['cost of interventon_mean'].sum()
        total_co2_saved = selected_projects_df['total_ton_co2_saved'].sum()
        avg_cost_per_tonne = total_cost / total_co2_saved if total_co2_saved > 0 else 0
        
        detail_logger.info(f"\n  RESULTS for epistemic run {epi_run}:")
        detail_logger.info(f"    Projects selected: {len(selected_projects_df)}")
        detail_logger.info(f"    Budget used: £{total_cost:,.0f} ({(total_cost/scenario_budget)*100:.1f}%)")
        detail_logger.info(f"    Remaining funds: £{remaining_funds:,.0f}")
        detail_logger.info(f"    Total CO2 saved: {total_co2_saved:,.0f} tonnes")
        detail_logger.info(f"    Average cost per tonne CO2: £{avg_cost_per_tonne:.2f}")
        
        # Scenario breakdown
        scenario_breakdown = selected_projects_df.groupby('scenario').agg({
            'upn': 'count',
            'cost of interventon_mean': 'sum',
            'total_ton_co2_saved': 'sum'
        }).rename(columns={'upn': 'n_projects'})
        scenario_breakdown['avg_cost_per_tonne'] = (
            scenario_breakdown['cost of interventon_mean'] / scenario_breakdown['total_ton_co2_saved']
        )
        
        detail_logger.info(f"\n  Scenario breakdown:")
        for sc in scenario_breakdown.index:
            row = scenario_breakdown.loc[sc]
            detail_logger.info(f"    {sc}:")
            detail_logger.info(f"      Projects: {row['n_projects']:.0f}")
            detail_logger.info(f"      Cost: £{row['cost of interventon_mean']:,.0f}")
            detail_logger.info(f"      CO2 saved: {row['total_ton_co2_saved']:,.0f} tonnes")
            detail_logger.info(f"      Avg cost/tonne: £{row['avg_cost_per_tonne']:.2f}")
        
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

        # ============================================================
        # NEW: EQUITY ANALYSIS FOR THIS EPISTEMIC RUN
        # ============================================================
        
        # Overall equity metrics
        overall_equity = calculate_social_equity_score(selected_projects_df)
        
        detail_logger.info(f"\n  EQUITY ANALYSIS for epistemic run {epi_run}:")
        detail_logger.info(f"    Vulnerable groups investment: {overall_equity['vulnerable_investment_pct']:.1f}%")
        detail_logger.info(f"    Equity concentration index: {overall_equity['equity_concentration']:.3f}")
        detail_logger.info(f"\n    Persona breakdown:")
        
        for persona, stats in overall_equity['persona_breakdown'].items():
            detail_logger.info(f"      {persona}: {stats['count']} projects ({stats['pct']:.1f}%)")
        
        # Scenario-specific equity analysis
        detail_logger.info(f"\n  Equity breakdown by scenario:")
        for scenario in scenario_list:
            scenario_equity = calculate_scenario_persona_metrics(selected_projects_df, scenario)
            if scenario_equity:
                metrics = scenario_equity['equity_metrics']
                detail_logger.info(f"\n    {scenario}:")
                detail_logger.info(f"      Vulnerable: {metrics['vulnerable_investment_pct']:.1f}%")
                
                for persona, stats in metrics['persona_breakdown'].items():
                    if stats['count'] > 0:
                        persona_stats = scenario_equity['persona_stats'].loc[persona]
                        detail_logger.info(f"        {persona}: {stats['count']} projects ({stats['pct']:.1f}%), "
                                         f"£{persona_stats['cost of interventon_mean']:,.0f}, "
                                         f"{persona_stats['total_ton_co2_saved']:,.0f} tonnes CO2")
                
                # Store for cross-run analysis
                equity_tracking.append({
                    'epistemic_run': epi_run,
                    'scenario': scenario,
                    'vulnerable_pct': metrics['vulnerable_investment_pct'],
                    'equity_concentration': metrics['equity_concentration'],
                    **{f'{persona}_count': metrics['persona_breakdown'][persona]['count'] 
                       for persona in EQUITY_WEIGHTS.keys()},
                    **{f'{persona}_pct': metrics['persona_breakdown'][persona]['pct'] 
                       for persona in EQUITY_WEIGHTS.keys()}
                })
        
        
        
        all_epistemic_results.append(selected_projects_df)
    
    # ============================================================================
    # COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS - SAVE TO SUMMARY FILE
    # ============================================================================
    
    combined_results = pd.concat(all_epistemic_results, ignore_index=True)
    
    summary_logger.info(f"\n{'='*80}")
    summary_logger.info("COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft}")
    summary_logger.info(f"{'='*80}\n")
    
    # 1. Summary by epistemic run
    summary_logger.info("1. SUMMARY BY EPISTEMIC RUN")
    summary_logger.info("-" * 80)
    summary_by_run = combined_results.groupby('epistemic_run').agg({
        'upn': 'count',
        'cost of interventon_mean': 'sum',
        'total_ton_co2_saved': 'sum',
        'remaining_funds': 'first'
    }).rename(columns={'upn': 'n_projects'})
    
    for epi_run in summary_by_run.index:
        row = summary_by_run.loc[epi_run]
        summary_logger.info(f"  Run {epi_run}:")
        summary_logger.info(f"    Projects: {row['n_projects']:.0f}")
        summary_logger.info(f"    Total cost: £{row['cost of interventon_mean']:,.0f}")
        summary_logger.info(f"    CO2 saved: {row['total_ton_co2_saved']:,.0f} tonnes")
        summary_logger.info(f"    Remaining: £{row['remaining_funds']:,.0f}\n")
    
    # 2. Overall statistics
    summary_logger.info("\n2. OVERALL STATISTICS ACROSS EPISTEMIC RUNS")
    summary_logger.info("-" * 80)
    summary_logger.info(f"  Mean projects selected: {summary_by_run['n_projects'].mean():.1f} ± {summary_by_run['n_projects'].std():.1f}")
    summary_logger.info(f"  Range: {summary_by_run['n_projects'].min():.0f} - {summary_by_run['n_projects'].max():.0f}")
    summary_logger.info(f"  Mean CO2 saved: {summary_by_run['total_ton_co2_saved'].mean():,.0f} ± {summary_by_run['total_ton_co2_saved'].std():,.0f} tonnes")
    summary_logger.info(f"  Range: {summary_by_run['total_ton_co2_saved'].min():,.0f} - {summary_by_run['total_ton_co2_saved'].max():,.0f} tonnes")
    summary_logger.info(f"  Mean cost: £{summary_by_run['cost of interventon_mean'].mean():,.0f} ± £{summary_by_run['cost of interventon_mean'].std():,.0f}")
    
    # 3. Scenario performance analysis
    summary_logger.info("\n\n3. SCENARIO PERFORMANCE ANALYSIS")
    summary_logger.info("-" * 80)
    scenario_perf_df = pd.DataFrame(scenario_performance_log)
    
    scenario_summary = scenario_perf_df.groupby('scenario').agg({
        'n_projects': ['mean', 'std', 'min', 'max'],
        'total_co2_saved': ['mean', 'std', 'sum'],
        'avg_cost_per_tonne': ['mean', 'std', 'min', 'max']
    })
    
    for scenario in scenario_list:
        if scenario in scenario_summary.index:
            summary_logger.info(f"\n  {scenario.upper()}:")
            stats = scenario_summary.loc[scenario]
            summary_logger.info(f"    Projects per run: {stats[('n_projects', 'mean')]:.1f} ± {stats[('n_projects', 'std')]:.1f} "
                              f"(range: {stats[('n_projects', 'min')]:.0f}-{stats[('n_projects', 'max')]:.0f})")
            summary_logger.info(f"    CO2 saved per run: {stats[('total_co2_saved', 'mean')]:,.0f} ± {stats[('total_co2_saved', 'std')]:,.0f} tonnes")
            summary_logger.info(f"    Total CO2 saved (all runs): {stats[('total_co2_saved', 'sum')]:,.0f} tonnes")
            summary_logger.info(f"    Avg cost/tonne: £{stats[('avg_cost_per_tonne', 'mean')]:.2f} ± £{stats[('avg_cost_per_tonne', 'std')]:.2f}")
            summary_logger.info(f"    Cost/tonne range: £{stats[('avg_cost_per_tonne', 'min')]:.2f} - £{stats[('avg_cost_per_tonne', 'max')]:.2f}")
    
    # 4. Scenario selection frequency
    summary_logger.info("\n\n4. SCENARIO SELECTION FREQUENCY")
    summary_logger.info("-" * 80)
    scenario_freq = combined_results.groupby('scenario').size()
    total_selections = scenario_freq.sum()
    for scenario in scenario_list:
        if scenario in scenario_freq.index:
            count = scenario_freq[scenario]
            pct = (count / total_selections) * 100
            summary_logger.info(f"  {scenario}: {count} selections ({pct:.1f}%)")
    
    # 5. Building robustness analysis
    summary_logger.info("\n\n5. BUILDING ROBUSTNESS ANALYSIS")
    summary_logger.info("-" * 80)
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
    
    summary_logger.info(f"  Buildings selected in ALL {len(epistemic_runs)} runs: {always_selected}")
    summary_logger.info(f"  Buildings selected in >50% of runs: {mostly_selected}")
    summary_logger.info(f"  Buildings selected in ≤50% of runs: {sometimes_selected}")
    summary_logger.info(f"  Total unique buildings selected: {len(building_selection_freq)}")
    
    # Top 10 most robust selections
    summary_logger.info(f"\n  Top 10 most robust buildings (selected most frequently):")
    top_buildings = building_selection_freq.nlargest(10, 'selection_rate')
    for idx, (upn, row) in enumerate(top_buildings.iterrows(), 1):
        summary_logger.info(f"    {idx}. UPN {upn}: selected {row['times_selected']}/{len(epistemic_runs)} times "
                          f"({row['selection_rate']*100:.0f}%), scenario: {row['scenario']}, "
                          f"avg cost/tonne: £{row['cost_per_net_ton_co2_kg']:.2f}")
    
    # 6. Exclusion statistics analysis
    summary_logger.info("\n\n6. EXCLUSION STATISTICS ANALYSIS")
    summary_logger.info("-" * 80)
    exclusion_df = pd.DataFrame(scenario_exclusion_stats)
    exclusion_summary = exclusion_df.groupby('scenario').agg({
        'n_upns_excluded': ['mean', 'std'],
        'pct_records_excluded': ['mean', 'std']
    })
    
    for scenario in scenario_list:
        if scenario in exclusion_summary.index:
            stats = exclusion_summary.loc[scenario]
            summary_logger.info(f"  {scenario}:")
            summary_logger.info(f"    Avg UPNs excluded: {stats[('n_upns_excluded', 'mean')]:.1f} ± {stats[('n_upns_excluded', 'std')]:.1f}")
            summary_logger.info(f"    Avg % records excluded: {stats[('pct_records_excluded', 'mean')]:.1f}% ± {stats[('pct_records_excluded', 'std')]:.1f}%")
    
     
         
    # ============================================================================
    # COMPREHENSIVE EQUITY ANALYSIS ACROSS ALL EPISTEMIC RUNS
    # ============================================================================
    
 
    
    summary_logger.info(f"\n{'='*80}")
    summary_logger.info("SOCIAL EQUITY ANALYSIS ACROSS ALL EPISTEMIC RUNS")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft} | Equity Factor: {equity_factor}")
    summary_logger.info(f"{'='*80}\n")
    
    # 1. Overall equity metrics
    summary_logger.info("1. OVERALL EQUITY METRICS")
    summary_logger.info("-" * 80)
    
    overall_equity_all = calculate_social_equity_score(combined_results)
    summary_logger.info(f"  Total projects across all runs: {overall_equity_all['total_count']}")
    summary_logger.info(f"  Vulnerable groups (deprived + struggling): {overall_equity_all['vulnerable_count']} "
                       f"({overall_equity_all['vulnerable_investment_pct']:.1f}%)")
    summary_logger.info(f"  Equity concentration index: {overall_equity_all['equity_concentration']:.3f}\n")
    
    summary_logger.info("  Persona distribution across all selections:")
    for persona, stats in overall_equity_all['persona_breakdown'].items():
        summary_logger.info(f"    {persona.ljust(15)}: {stats['count']:5d} projects ({stats['pct']:5.1f}%)")
    
    # 2. Equity metrics by epistemic run
    summary_logger.info("\n\n2. EQUITY METRICS BY EPISTEMIC RUN")
    summary_logger.info("-" * 80)
    
    equity_by_run = combined_results.groupby('epistemic_run').apply(
        lambda x: pd.Series({
            'vulnerable_pct': calculate_social_equity_score(x)['vulnerable_investment_pct'],
            'concentration': calculate_social_equity_score(x)['equity_concentration'],
            'n_projects': len(x)
        })
    )
    
    summary_logger.info(f"  Mean vulnerable investment: {equity_by_run['vulnerable_pct'].mean():.1f}% "
                       f"± {equity_by_run['vulnerable_pct'].std():.1f}%")
    summary_logger.info(f"  Range: {equity_by_run['vulnerable_pct'].min():.1f}% - "
                       f"{equity_by_run['vulnerable_pct'].max():.1f}%")
    summary_logger.info(f"  Mean concentration index: {equity_by_run['concentration'].mean():.3f} "
                       f"± {equity_by_run['concentration'].std():.3f}")
    
    # 3. Equity metrics by scenario
    summary_logger.info("\n\n3. EQUITY METRICS BY SCENARIO")
    summary_logger.info("-" * 80)
    
    equity_tracking_df = pd.DataFrame(equity_tracking)
    
    for scenario in scenario_list:
        scenario_equity_data = equity_tracking_df[equity_tracking_df['scenario'] == scenario]
        
        if len(scenario_equity_data) > 0:
            summary_logger.info(f"\n  {scenario.upper()}:")
            summary_logger.info(f"    Vulnerable investment: {scenario_equity_data['vulnerable_pct'].mean():.1f}% "
                              f"± {scenario_equity_data['vulnerable_pct'].std():.1f}%")
            summary_logger.info(f"    Concentration: {scenario_equity_data['equity_concentration'].mean():.3f} "
                              f"± {scenario_equity_data['equity_concentration'].std():.3f}")
            
            summary_logger.info(f"\n    Persona distribution:")
            for persona in EQUITY_WEIGHTS.keys():
                count_col = f'{persona}_count'
                pct_col = f'{persona}_pct'
                if count_col in scenario_equity_data.columns:
                    mean_count = scenario_equity_data[count_col].mean()
                    mean_pct = scenario_equity_data[pct_col].mean()
                    std_pct = scenario_equity_data[pct_col].std()
                    summary_logger.info(f"      {persona.ljust(15)}: {mean_count:5.1f} projects "
                                      f"({mean_pct:5.1f}% ± {std_pct:4.1f}%)")
    
    # 4. Compare equity factor impact (if equity_factor != 0)
    if equity_factor > 0:
        summary_logger.info("\n\n4. EQUITY FACTOR IMPACT")
        summary_logger.info("-" * 80)
        summary_logger.info(f"  Equity factor applied: {equity_factor}")
        summary_logger.info(f"  This prioritizes investment in:")
        for persona, weight in sorted(EQUITY_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
            if weight > 1.0:
                summary_logger.info(f"    {persona}: {weight}x weighting")
    
    # 5. Cost and CO2 efficiency by persona
    summary_logger.info("\n\n5. COST & CO2 EFFICIENCY BY PERSONA")
    summary_logger.info("-" * 80)
    
    persona_efficiency = combined_results.groupby('meta_socio_persona').agg({
        'cost of interventon_mean': ['sum', 'mean'],
        'total_ton_co2_saved': ['sum', 'mean'],
        'cost_per_net_ton_co2_kg': 'mean',
        'upn': 'count'
    }).rename(columns={'upn': 'n_projects'})
    
    for persona in EQUITY_WEIGHTS.keys():
        if persona in persona_efficiency.index:
            stats = persona_efficiency.loc[persona]
            total_cost = stats[('cost of interventon_mean', 'sum')]
            total_co2 = stats[('total_ton_co2_saved', 'sum')]
            n_proj = stats[('n_projects', 'count')]
            avg_cost_per_tonne = stats[('cost_per_net_ton_co2_kg', 'mean')]
            
            summary_logger.info(f"\n  {persona}:")
            summary_logger.info(f"    Projects: {n_proj:.0f}")
            summary_logger.info(f"    Total cost: £{total_cost:,.0f}")
            summary_logger.info(f"    Total CO2 saved: {total_co2:,.0f} tonnes")
            summary_logger.info(f"    Avg cost per tonne CO2: £{avg_cost_per_tonne:.2f}")
    
    summary_logger.info(f"\n{'='*80}\n")
    
    # Save equity tracking data
    equity_tracking_df.to_csv(
        os.path.join(output_dir, 'equity_tracking.csv'), 
        index=False
    )
    
    summary_logger.info(f"\n{'='*80}\n")
    
    return baseline_selection, combined_results


def main():
    """
    Main execution function for greedy algorithm analysis.
    """
    # Configuration
    BASE_DIR = '/Users/gracecolverd/RetrofitModel/test/greedy'
    INPUT_FILES_PATH = '/Users/gracecolverd/Downloads/all/*.csv'
    
    YEARS = 5
    N_SIMULATIONS = 5000
    ELEC_CARBON_FACTOR = 0.2
    GAS_CARBON_FACTOR = 0.2
    equity_factor = 1 
    
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
        loft_probs = [0.65, 0.95]
    else:
        budgets = [10_000_000, 100_000_000, 1_000_000_000]
        loft_probs = [0.65, 0.95]
    
    for budget in budgets:
        for prob_loft in loft_probs:
            for equity_factor in [0.5]: 
                print(f"\n{'='*80}")
                print(f"Starting analysis: Budget £{budget:,}, Loft Probability {prob_loft} and equity: {equity_factor}")
                print(f"{'='*80}")
                
                # Create output directory
                output_dir = os.path.join(BASE_DIR, f'budget_{budget}_loft_{prob_loft}_equity: {equity_factor}')
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