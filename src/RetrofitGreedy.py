



import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Any

 
import os 

from src.RetrofitEquity import EQUITY_WEIGHTS,  calculate_social_equity_score , calculate_scenario_persona_metrics
import pandas as pd
import numpy as np
from src.GreedyAlgo import true_greedy_knapsack, plot_greedy_distribution_analysis 

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


 
 


# =============================================================================
# CONSTANTS (Centralized "Magic Strings")
# =============================================================================
# --- Input Columns ---
EPISTEMIC_COL = 'epistemic_run_id'
UPN_COL = 'upn'
PERSONA_COL = 'meta_socio_persona'
LOFT_EXISTS_COL = 'already_loft'
GAS_PERCENTILE_COL = 'avg_gas_percentile'

# --- Scenario/Dynamic Column Templates ---
CO2_SAVED_TPL = 'total_tonne_co2_saved_{scenario}_5yr_mean'
COST_PER_TON_TPL = 'cost_per_net_ton_co2_{scenario}_mean'
COST_TPL = '{scenario}_cost_{scenario}_mean'

# --- Scenario Names ---
LOFT_SCENARIO = 'loft_installation'

# --- Internal/Calculated Columns ---
FLIP_CO2_TPL = 'flip_sign_total_tonne_co2_saved_{scenario}_5yr_mean'
FLIP_COST_PER_TON_TPL = 'flip_sign_cost_per_net_ton_co2_{scenario}_mean'
EQUITY_WEIGHT_COL = 'equity_weight'
WEIGHTED_COST_PER_TON_TPL = 'flip_sign_weighted_cost_per_ton_{scenario}'

# --- Final Rank DataFrame Columns (Standardized) ---
RANK_COL_UPN = 'upn'
RANK_COL_PERSONA = 'meta_socio_persona'
RANK_COL_GAS_PCT = 'avg_gas_percentile'
RANK_COL_COST = 'cost_of_intervention_mean'
RANK_COL_CO2_SAVED = 'total_ton_co2_saved'
RANK_COL_COST_PER_TON = 'cost_per_net_ton_co2_kg'
RANK_COL_WEIGHTED_COST = 'weighted_cost_per_net_ton'
RANK_COL_SCENARIO = 'scenario'
RANK_COL_EPI_RUN = 'epistemic_run'

# List of final columns for easy renaming
FINAL_RANK_COLS = [
    RANK_COL_UPN,
    RANK_COL_PERSONA,
    RANK_COL_GAS_PCT,
    RANK_COL_COST,
    RANK_COL_CO2_SAVED,
    RANK_COL_COST_PER_TON,
    RANK_COL_WEIGHTED_COST,
    RANK_COL_SCENARIO,
    RANK_COL_EPI_RUN
]


# =============================================================================
# HELPER FUNCTION: PROCESS ONE SCENARIO
# =============================================================================

def _process_scenario(epi_df: pd.DataFrame, 
                      scenario: str, 
                      equity_factor: float, 
                      detail_logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filters, weights, and formats data for a single scenario within an epistemic run.
    """
    detail_logger.info(f"\n  --- Processing scenario: {scenario} ---")
     
    # # DEBUG: Print available columns related to this scenario
    # scenario_cols = [col for col in epi_df.columns if scenario in col.lower()]
    # detail_logger.debug(f"    Available columns containing '{scenario}':")
    # for col in scenario_cols:
    #     detail_logger.debug(f"      - {col}")
    
    # --- Define dynamic column names ---
    co2_col = CO2_SAVED_TPL.format(scenario=scenario)
    cost_col = COST_TPL.format(scenario=scenario)
    cost_per_ton_col = COST_PER_TON_TPL.format(scenario=scenario)
    
    # # DEBUG: Show what columns we're looking for
    # detail_logger.debug(f"    Looking for columns:")
    # detail_logger.debug(f"      - {co2_col}")
    # detail_logger.debug(f"      - {cost_col}")
    # detail_logger.debug(f"      - {cost_per_ton_col}")
    
    # --- Define dynamic column names ---
    co2_col = CO2_SAVED_TPL.format(scenario=scenario)
    cost_col = COST_TPL.format(scenario=scenario)
    cost_per_ton_col = COST_PER_TON_TPL.format(scenario=scenario)
    
    # Check if required columns exist
    required_cols = [co2_col, cost_col, cost_per_ton_col, PERSONA_COL, GAS_PERCENTILE_COL]
    if not all(col in epi_df.columns for col in required_cols):
        detail_logger.warning(f"    Missing required columns for scenario {scenario}. Skipping.")
        # Return empty results
        stats = {
            'epistemic_run': epi_df[EPISTEMIC_COL].iloc[0] if not epi_df.empty else 'unknown',
            'scenario': scenario,
            'n_upns_excluded': 0, 'n_records_excluded': 0, 'pct_records_excluded': 0
        }
        return pd.DataFrame(columns=FINAL_RANK_COLS), stats
        
    # --- 1. Exclude buildings with positive CO2 values (emissions increase) ---
    mask = epi_df[co2_col] > 0
    bad_upns_for_scenario = epi_df.loc[mask, UPN_COL].unique().tolist()
    
    n_excluded_buildings = len(bad_upns_for_scenario)
    n_excluded_records = mask.sum()
    pct_excluded = (n_excluded_records / len(epi_df)) * 100 if len(epi_df) > 0 else 0
    
    detail_logger.info(f"    Excluded {n_excluded_buildings} UPNs ({n_excluded_records} records, {pct_excluded:.1f}%) with positive CO2 values")
    
    exclusion_stats = {
        'scenario': scenario,
        'n_upns_excluded': n_excluded_buildings,
        'n_records_excluded': n_excluded_records,
        'pct_records_excluded': pct_excluded
    }
    
    scenario_df = epi_df[~epi_df[UPN_COL].isin(bad_upns_for_scenario)].copy()
    
    # --- 2. Handle scenario-specific logic ---
    if scenario == LOFT_SCENARIO:
        detail_logger.info('    Removing buildings with existing loft insulation')
        scenario_df = scenario_df[scenario_df[LOFT_EXISTS_COL] == False]
    
    detail_logger.info(f"    Remaining buildings for {scenario}: {len(scenario_df)}")
    
    if scenario_df.empty:
        return pd.DataFrame(columns=FINAL_RANK_COLS), exclusion_stats

    # --- 3. Apply calculations ---
    flip_co2_col = FLIP_CO2_TPL.format(scenario=scenario)
    flip_cost_per_ton_col = FLIP_COST_PER_TON_TPL.format(scenario=scenario)
    weighted_cost_col = WEIGHTED_COST_PER_TON_TPL.format(scenario=scenario)
    
    # Flip signs for optimization (greedy algorithm minimizes)
    scenario_df[flip_co2_col] = -scenario_df[co2_col]
    scenario_df[flip_cost_per_ton_col] = -scenario_df[cost_per_ton_col]
    
    # Apply equity weighting
    scenario_df[EQUITY_WEIGHT_COL] = scenario_df[PERSONA_COL].map(EQUITY_WEIGHTS)
    scenario_df[weighted_cost_col] = (
        scenario_df[flip_cost_per_ton_col] * (1 + (scenario_df[EQUITY_WEIGHT_COL] - 1) * equity_factor)
    )
    
    # --- 4. Prepare ranking dataframe ---
    source_cols = [
        UPN_COL,
        PERSONA_COL,
        GAS_PERCENTILE_COL,
        cost_col,
        flip_co2_col,
        flip_cost_per_ton_col,
        weighted_cost_col
    ]
    
    df_rank = scenario_df[source_cols].copy()
    df_rank[RANK_COL_SCENARIO] = scenario
    
    # Set standardized column names
    df_rank.columns = FINAL_RANK_COLS[:-1] # All except epistemic_run

    # --- 5. Log scenario statistics ---
    valid_scenario_data = df_rank[~df_rank[RANK_COL_COST_PER_TON].isna()]
    if len(valid_scenario_data) > 0:
        detail_logger.info(f"    Valid buildings for optimization: {len(valid_scenario_data)}")
        detail_logger.info(f"    Cost/tonne CO2 - Min: £{valid_scenario_data[RANK_COL_COST_PER_TON].min():.2f}, "
                         f"Median: £{valid_scenario_data[RANK_COL_COST_PER_TON].median():.2f}, "
                         f"Max: £{valid_scenario_data[RANK_COL_COST_PER_TON].max():.2f}")
    else:
        detail_logger.warning(f"    No valid data for scenario {scenario}!")
    
    return df_rank, exclusion_stats


# =============================================================================
# HELPER FUNCTION: LOG EPISTEMIC RUN EQUITY
# =============================================================================

def _log_epistemic_run_equity_analysis(selected_projects_df: pd.DataFrame,
                                      epi_run: str,
                                      detail_logger: logging.Logger) -> Dict[str, Any]:
    """
    Performs and logs the equity analysis for a single, completed epistemic run.
    The epistemic run contains different selected options (retrofit interventions),
    and is analyzed as a single cohesive unit.
    Returns tracking data for the comprehensive summary.
    """
    # DEBUG: Check what columns we actually have
    detail_logger.info(f"\n  DEBUG - Columns in selected_projects_df:")
    detail_logger.info(f"    {selected_projects_df.columns.tolist()}")
    
    # --- Equity metrics for the entire epistemic run ---
    equity_metrics = calculate_social_equity_score(selected_projects_df)
    
    detail_logger.info(f"\n  EQUITY ANALYSIS for epistemic run {epi_run}:")
    detail_logger.info(f"    Vulnerable groups investment: {equity_metrics['vulnerable_investment_pct']:.1f}%")
    detail_logger.info(f"    Equity concentration index: {equity_metrics['equity_concentration']:.3f}")
    detail_logger.info(f"\n    Persona breakdown:")
    
    for persona, stats in equity_metrics['persona_breakdown'].items():
        detail_logger.info(f"      {persona}: {stats['count']} projects ({stats['pct']:.1f}%)")
    
    # Store for cross-run analysis
    equity_tracking = {
        'epistemic_run': epi_run,
        'vulnerable_pct': equity_metrics['vulnerable_investment_pct'],
        'equity_concentration': equity_metrics['equity_concentration'],
        **{f'{persona}_count': equity_metrics['persona_breakdown'].get(persona, {}).get('count', 0) 
           for persona in EQUITY_WEIGHTS.keys()},
        **{f'{persona}_pct': equity_metrics['persona_breakdown'].get(persona, {}).get('pct', 0) 
           for persona in EQUITY_WEIGHTS.keys()},
    }
            
    return equity_tracking

# def _log_epistemic_run_equity_analysis(selected_projects_df: pd.DataFrame,
#                                       epi_run: str,
#                                       scenario_list: List[str],
#                                       detail_logger: logging.Logger) -> List[Dict[str, Any]]:
#     """
#     Performs and logs the equity analysis for a single, completed epistemic run.
#     Returns tracking data for the comprehensive summary.
#     """
#     # DEBUG: Check what columns we actually have
#     detail_logger.info(f"\n  DEBUG - Columns in selected_projects_df:")
#     detail_logger.info(f"    {selected_projects_df.columns.tolist()}")

#     equity_tracking = [] 
    
#     # --- Overall equity metrics ---
#     overall_equity = calculate_social_equity_score(selected_projects_df)
    
#     detail_logger.info(f"\n  EQUITY ANALYSIS for epistemic run {epi_run}:")
#     detail_logger.info(f"    Vulnerable groups investment: {overall_equity['vulnerable_investment_pct']:.1f}%")
#     detail_logger.info(f"    Equity concentration index: {overall_equity['equity_concentration']:.3f}")
#     detail_logger.info(f"\n    Persona breakdown:")
    
#     for persona, stats in overall_equity['persona_breakdown'].items():
#         detail_logger.info(f"      {persona}: {stats['count']} projects ({stats['pct']:.1f}%)")
    
#     # --- Scenario-specific equity analysis ---
#     detail_logger.info(f"\n  Equity breakdown by scenario:")
#     for scenario in scenario_list:
#         scenario_equity = calculate_scenario_persona_metrics(selected_projects_df, scenario)
#         if scenario_equity:
#             metrics = scenario_equity['equity_metrics']
#             detail_logger.info(f"\n    {scenario}:")
#             detail_logger.info(f"      Vulnerable: {metrics['vulnerable_investment_pct']:.1f}%")
            
#             for persona, stats in metrics['persona_breakdown'].items():
#                 if stats['count'] > 0:
#                     persona_stats = scenario_equity['persona_stats'].loc[persona]
#                     detail_logger.info(f"        {persona}: {stats['count']} projects ({stats['pct']:.1f}%), "
#                                      f"£{persona_stats[RANK_COL_COST]:,.0f}, "
#                                      f"{persona_stats[RANK_COL_CO2_SAVED]:,.0f} tonnes CO2")
            
#             # Store for cross-run analysis
#             equity_tracking.append({
#                 'epistemic_run': epi_run,
#                 'scenario': scenario,
#                 'vulnerable_pct': metrics['vulnerable_investment_pct'],
#                 'equity_concentration': metrics['equity_concentration'],
#                 **{f'{persona}_count': metrics['persona_breakdown'].get(persona, {}).get('count', 0) for persona in EQUITY_WEIGHTS.keys()},
#                 **{f'{persona}_pct': metrics['persona_breakdown'].get(persona, {} ).get('pct', 0 )  for persona in EQUITY_WEIGHTS.keys()},
#             })
            
#     return equity_tracking


# =============================================================================
# HELPER FUNCTION: PROCESS ONE EPISTEMIC RUN
# =============================================================================

def _process_epistemic_run(epi_run: str,
                          epi_df_full: pd.DataFrame,
                          scenario_list: List[str],
                          prob_loft: float,
                          equity_factor: float,
                          scenario_budget: float,
                          detail_logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict], pd.DataFrame]:
    """
    Runs the full analysis for a single epistemic run, from scenario processing
    to knapsack selection and equity analysis.
    """
    # --- 1. Prepare data for this run ---
    epi_df = epi_df_full.copy()
    epi_df = assign_random_loft(epi_df, prob_loft)
    detail_logger.info(f"Buildings in this epistemic run: {len(epi_df)}")
    
    all_scenario_ranks = []
    all_exclusion_stats = []
    
    # --- 2. Process all scenarios --- theses are the retrofit intevention scenarios (e/g. wall , loft. heat pump )
    for scenario in scenario_list:
        df_rank, exclusion_stats = _process_scenario(
            epi_df, scenario, equity_factor, detail_logger
        )
        if not df_rank.empty:
            all_scenario_ranks.append(df_rank)
        
        exclusion_stats['epistemic_run'] = epi_run
        all_exclusion_stats.append(exclusion_stats)

    if not all_scenario_ranks:
        detail_logger.warning(f"No valid data for any scenario in epistemic run {epi_run}. Skipping.")
        return pd.DataFrame(), [], [], [], pd.DataFrame()
        
    # --- 3. Combine scenarios and filter ---
    res_df = pd.concat(all_scenario_ranks)
    res_df[RANK_COL_EPI_RUN] = epi_run
    
    # Filter for valid, optimizable projects
    wdf = res_df[~res_df[RANK_COL_COST_PER_TON].isna()]
    
    detail_logger.info(f"\n  Combined dataset for epistemic run {epi_run}:")
    detail_logger.info(f"    Total valid building-scenario combinations: {len(wdf)}")
    detail_logger.info(f"    Unique buildings: {wdf[RANK_COL_UPN].nunique()}")

    # --- 4. Run greedy knapsack ---
    detail_logger.info(f"\n  Running greedy knapsack with budget: £{scenario_budget:,}")
    
    # Get the best *potential* project for each building
    baseline_selection = (wdf
                        .sort_values(RANK_COL_WEIGHTED_COST, ascending=True)
                        .drop_duplicates(subset=RANK_COL_UPN, keep='first')
                        .reset_index(drop=True))
    
    detail_logger.info(f"    Candidate projects after deduplication: {len(baseline_selection)}")
    
    selected_projects_df, remaining_funds = true_greedy_knapsack(
        df_knapsack=baseline_selection,
        budget=scenario_budget,
        cost_column=RANK_COL_COST,
        efficiency_column=RANK_COL_WEIGHTED_COST
    )
    
    # --- 5. Log selection results ---
    selected_projects_df=selected_projects_df.copy() 
    detail_logger.info(selected_projects_df.columns.tolist() )
    selected_projects_df[RANK_COL_EPI_RUN] = epi_run
    selected_projects_df['remaining_funds'] = remaining_funds
    
    total_cost = selected_projects_df[RANK_COL_COST].sum()
    total_co2_saved = selected_projects_df[RANK_COL_CO2_SAVED].sum()
    
    detail_logger.info(f"\n  RESULTS for epistemic run {epi_run}:")
    detail_logger.info(f"    Projects selected: {len(selected_projects_df)}")
    detail_logger.info(f"    Budget used: £{total_cost:,.0f} ({(total_cost/scenario_budget)*100:.1f}%)")
    detail_logger.info(f"    Total CO2 saved: {total_co2_saved:,.0f} tonnes")
    
    # --- 6. Log scenario breakdown and collect performance stats ---
    scenario_performance_log = []
    if not selected_projects_df.empty:
        scenario_breakdown = selected_projects_df.groupby(RANK_COL_SCENARIO).agg({
            RANK_COL_UPN: 'count',
            RANK_COL_COST: 'sum',
            RANK_COL_CO2_SAVED: 'sum'
        }).rename(columns={RANK_COL_UPN: 'n_projects'})
        
        scenario_breakdown['avg_cost_per_tonne'] = (
            scenario_breakdown[RANK_COL_COST] / scenario_breakdown[RANK_COL_CO2_SAVED]
        )
        
        detail_logger.info(f"\n  Scenario breakdown:")
        for sc in scenario_breakdown.index:
            row = scenario_breakdown.loc[sc]
            detail_logger.info(f"    {sc}: {row['n_projects']:.0f} projects, "
                             f"Cost: £{row[RANK_COL_COST]:,.0f}, "
                             f"CO2: {row[RANK_COL_CO2_SAVED]:,.0f} tonnes")
            
            # Store performance metrics
            scenario_performance_log.append({
                'epistemic_run': epi_run,
                'scenario': sc,
                'n_projects': row['n_projects'],
                'total_cost': row[RANK_COL_COST],
                'total_co2_saved': row[RANK_COL_CO2_SAVED],
                'avg_cost_per_tonne': row['avg_cost_per_tonne']
            })
            
    # --- 7. Perform and log equity analysis ---
    equity_tracking = _log_epistemic_run_equity_analysis(
        selected_projects_df, epi_run, detail_logger
    )
    
    return selected_projects_df, all_exclusion_stats, scenario_performance_log, equity_tracking, baseline_selection


# =============================================================================
# HELPER FUNCTION: LOG COMPREHENSIVE SUMMARY
# =============================================================================

def log_comprehensive_summary(combined_results: pd.DataFrame,
                              all_exclusion_stats: List[Dict],
                              all_scenario_performance: List[Dict],
                              all_equity_tracking: List[Dict],
                              scenario_budget: float,
                              prob_loft: float,
                              equity_factor: float,
                              scenario_list: List[str],
                              epistemic_runs: List[str],
                              summary_logger: logging.Logger,
                              output_dir: str):
    """
    Takes all collated results and logs the final comprehensive analysis.
    This function is purely for reporting and file output.
    """
    
    # Convert list-of-dicts to DataFrames for analysis
    scenario_perf_df = pd.DataFrame(all_scenario_performance)
    exclusion_df = pd.DataFrame(all_exclusion_stats)
    equity_tracking_df = pd.DataFrame(all_equity_tracking)
    summary_logger.info('equity_tracking_df')
    summary_logger.info(equity_tracking_df.head())
    n_epi_runs = len(epistemic_runs)
    
    summary_logger.info(f"\n{'='*80}")
    summary_logger.info("COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft} | Equity Factor: {equity_factor}")
    summary_logger.info(f"{'='*80}\n")
    
    # --- 1. Summary by epistemic run ---
    summary_logger.info("1. SUMMARY BY EPISTEMIC RUN")
    summary_logger.info("-" * 80)
    summary_by_run = combined_results.groupby(RANK_COL_EPI_RUN).agg({
        RANK_COL_UPN: 'count',
        RANK_COL_COST: 'sum',
        RANK_COL_CO2_SAVED: 'sum',
        'remaining_funds': 'first'
    }).rename(columns={RANK_COL_UPN: 'n_projects'})
    
    # --- 2. Overall statistics ---
    summary_logger.info("\n2. OVERALL STATISTICS ACROSS EPISTEMIC RUNS")
    summary_logger.info("-" * 80)
    summary_logger.info(f"  Mean projects selected: {summary_by_run['n_projects'].mean():.1f} ± {summary_by_run['n_projects'].std():.1f}")
    summary_logger.info(f"  Mean CO2 saved: {summary_by_run[RANK_COL_CO2_SAVED].mean():,.0f} ± {summary_by_run[RANK_COL_CO2_SAVED].std():,.0f} tonnes")
    
    # --- 3. Scenario performance analysis ---
    if not scenario_perf_df.empty:
        summary_logger.info("\n\n3. SCENARIO PERFORMANCE ANALYSIS")
        summary_logger.info("-" * 80)
        scenario_summary = scenario_perf_df.groupby('scenario').agg({
            'n_projects': ['mean', 'std', 'min', 'max'],
            'total_co2_saved': ['mean', 'std', 'sum'],
            'avg_cost_per_tonne': ['mean', 'std', 'min', 'max']
        })
        
        for scenario in scenario_list:
            if scenario in scenario_summary.index:
                summary_logger.info(f"\n  {scenario.upper()}:")
                stats = scenario_summary.loc[scenario]
                summary_logger.info(f"    Projects per run: {stats[('n_projects', 'mean')]:.1f} ± {stats[('n_projects', 'std')]:.1f}")
                summary_logger.info(f"    CO2 saved per run: {stats[('total_co2_saved', 'mean')]:,.0f} ± {stats[('total_co2_saved', 'std')]:,.0f} tonnes")

    # --- 4. Scenario selection frequency ---
    summary_logger.info("\n\n4. SCENARIO SELECTION FREQUENCY")
    summary_logger.info("-" * 80)
    scenario_freq = combined_results.groupby(RANK_COL_SCENARIO).size()
    total_selections = scenario_freq.sum()
    if total_selections > 0:
        for scenario in scenario_list:
            if scenario in scenario_freq.index:
                count = scenario_freq[scenario]
                pct = (count / total_selections) * 100
                summary_logger.info(f"  {scenario}: {count} selections ({pct:.1f}%)")

    # --- 5. Building robustness analysis ---
    summary_logger.info("\n\n5. BUILDING ROBUSTNESS ANALYSIS")
    summary_logger.info("-" * 80)
    building_selection_freq = combined_results.groupby(RANK_COL_UPN).agg({
        RANK_COL_EPI_RUN: 'count',
        RANK_COL_SCENARIO: lambda x: x.mode()[0] if len(x) > 0 else None,
        RANK_COL_COST_PER_TON: 'mean',
        RANK_COL_CO2_SAVED: 'mean'
    }).rename(columns={RANK_COL_EPI_RUN: 'times_selected'})
    
    building_selection_freq['selection_rate'] = (
        building_selection_freq['times_selected'] / n_epi_runs
    )
    
    summary_logger.info(f"  Total unique buildings selected: {len(building_selection_freq)}")
    summary_logger.info(f"  Buildings selected in ALL {n_epi_runs} runs: {(building_selection_freq['selection_rate'] == 1).sum()}")
    
    # --- 6. Exclusion statistics analysis ---
    if not exclusion_df.empty:
        summary_logger.info("\n\n6. EXCLUSION STATISTICS ANALYSIS")
        summary_logger.info("-" * 80)
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

      # ============================================================
    # COMPREHENSIVE EQUITY ANALYSIS
    # ============================================================
    
    summary_logger.info(f"\n{'='*80}")
    summary_logger.info("SOCIAL EQUITY ANALYSIS ACROSS ALL EPISTEMIC RUNS")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Equity Factor: {equity_factor}")
    summary_logger.info(f"{'='*80}\n")
    
    # --- 1. Overall equity metrics ---
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
    
    # --- 2. Equity metrics by epistemic run (variability) ---
    summary_logger.info("\n\n2. EQUITY METRICS BY EPISTEMIC RUN (VARIABILITY)")
    summary_logger.info("-" * 80)
    
    if not equity_tracking_df.empty and 'epistemic_run' in equity_tracking_df.columns:
        # Now each row is one epistemic run
        summary_logger.info(f"  Mean vulnerable investment: {equity_tracking_df['vulnerable_pct'].mean():.1f}% "
                           f"± {equity_tracking_df['vulnerable_pct'].std():.1f}%")
        summary_logger.info(f"  Mean concentration index: {equity_tracking_df['equity_concentration'].mean():.3f} "
                           f"± {equity_tracking_df['equity_concentration'].std():.3f}")
        
        # Show persona variability across runs
        summary_logger.info(f"\n  Persona distribution variability across runs:")
        for persona in EQUITY_WEIGHTS.keys():
            pct_col = f'{persona}_pct'
            if pct_col in equity_tracking_df.columns:
                mean_pct = equity_tracking_df[pct_col].mean()
                std_pct = equity_tracking_df[pct_col].std()
                summary_logger.info(f"    {persona.ljust(15)}: {mean_pct:5.1f}% ± {std_pct:4.1f}%")

    # --- 3. Cost and CO2 efficiency by persona ---
    summary_logger.info("\n\n3. COST & CO2 EFFICIENCY BY PERSONA")
    summary_logger.info("-" * 80)
    
    persona_efficiency = combined_results.groupby(RANK_COL_PERSONA).agg({
        RANK_COL_COST: ['sum', 'mean'],
        RANK_COL_CO2_SAVED: ['sum', 'mean'],
        RANK_COL_COST_PER_TON: 'mean',
        RANK_COL_UPN: 'count'
    }).rename(columns={RANK_COL_UPN: 'n_projects'})
    
    for persona in EQUITY_WEIGHTS.keys():
        if persona in persona_efficiency.index:
            stats = persona_efficiency.loc[persona]
            summary_logger.info(f"\n  {persona}:")
            summary_logger.info(f"    Projects: {stats[('n_projects', 'count')]:.0f}")
            summary_logger.info(f"    Avg cost per tonne CO2: £{stats[(RANK_COL_COST_PER_TON, 'mean')]:.2f}")
    
    # --- Save equity tracking data ---
    try:
        if not equity_tracking_df.empty:
            equity_tracking_df.to_csv(
                os.path.join(output_dir, 'equity_tracking.csv'), 
                index=False
            )
            summary_logger.info(f"\nEquity tracking saved to {output_dir}")
    except Exception as e:
        summary_logger.error(f"Failed to save equity_tracking.csv: {e}")
        
    summary_logger.info(f"\n{'='*80}\n")


# =============================================================================
# MAIN ORCHESTRATOR FUNCTION (Refactored)
# =============================================================================

def run_greedy_algo(scenario_budget: float, 
                    prob_loft: float, 
                    df: pd.DataFrame, 
                    scenario_list: List[str], 
                    summary_logger: logging.Logger, 
                    detail_logger: logging.Logger, 
                    equity_factor: float, 
                    output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run greedy knapsack algorithm across multiple epistemic runs to select optimal retrofit projects.
    
    Args:
        (Same as original)
    
    Returns:
        tuple: (baseline_selection DataFrame from *last* run, combined_results DataFrame)
    """
    epistemic_runs = df[EPISTEMIC_COL].unique()
    
    detail_logger.info(f"Starting analysis with {len(epistemic_runs)} epistemic runs and {len(scenario_list)} scenarios")
    detail_logger.info(f"Total buildings in dataset: {len(df)}")
    detail_logger.info(f"Equity factor: {equity_factor}")
    
    # --- Storage for results from all runs ---
    all_epistemic_results = []
    all_exclusion_stats = []
    all_scenario_performance = []
    all_equity_tracking = []
    
    baseline_selection = pd.DataFrame() # To store the last run's baseline
    
    # --- Process each epistemic run ---
    for epi_idx, epi_run in enumerate(epistemic_runs, 1):
        detail_logger.info(f"\n{'='*80}")
        detail_logger.info(f"Processing epistemic run {epi_idx}/{len(epistemic_runs)}: {epi_run}")
        detail_logger.info(f"{'='*80}")
        
        # Filter data for this epistemic run
        epi_df_full = df[df[EPISTEMIC_COL] == epi_run]
        
        # --- Call helper to process the entire run ---
        selected_df, ex_stats, perf_stats, eq_stats, baseline_sel = _process_epistemic_run(
            epi_run=epi_run,
            epi_df_full=epi_df_full,
            scenario_list=scenario_list,
            prob_loft=prob_loft,
            equity_factor=equity_factor,
            scenario_budget=scenario_budget,
            detail_logger=detail_logger
        )
        
        # --- Collect results ---
        if not selected_df.empty:
            all_epistemic_results.append(selected_df)
            all_exclusion_stats.extend(ex_stats)
            all_scenario_performance.extend(perf_stats)
            all_equity_tracking.append(eq_stats)
            baseline_selection = baseline_sel # Store baseline from the last successful run

    if not all_epistemic_results:
        detail_logger.error("No results generated from any epistemic run.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Combine all epistemic run results ---
    combined_results = pd.concat(all_epistemic_results, ignore_index=True)
    detail_logger.info('all_equity_tracking')
    detail_logger.info(len(all_equity_tracking))
    
    # --- Call helper to log the comprehensive summary ---
    log_comprehensive_summary(
        combined_results=combined_results,
        all_exclusion_stats=all_exclusion_stats,
        all_scenario_performance=all_scenario_performance,
        all_equity_tracking=all_equity_tracking,
        scenario_budget=scenario_budget,
        prob_loft=prob_loft,
        equity_factor=equity_factor,
        scenario_list=scenario_list,
        epistemic_runs=epistemic_runs,
        summary_logger=summary_logger,
        output_dir=output_dir
    )
    
    # Preserve the original's strange return signature
    return baseline_selection, combined_results


# =============================================================================
# EXAMPLE USAGE (Main script block)
# =============================================================================

# if __name__ == "__main__":
#     # --- 1. Create mock data ---
#     N_BUILDINGS = 1000
#     N_EPI_RUNS = 5
#     SCENARIOS = ['loft_installation', 'wall_insulation', 'heat_pump']
    
#     data = []
#     for epi_run in range(N_EPI_RUNS):
#         for i in range(N_BUILDINGS):
#             base = {
#                 'upn': i,
#                 'epistemic_run_id': epi_run,
#                 'meta_socio_persona': np.random.choice(list(EQUITY_WEIGHTS.keys())),
#                 'avg_gas_percentile': np.random.rand(),
#             }
#             for s in SCENARIOS:
#                 # Add scenario-specific mock data
#                 base[f'total_tonne_co2_saved_{s}_5yr_mean'] = np.random.uniform(-0.5, 5.0) # Some negative
#                 base[f'cost_of_intervention_{s}_mean'] = np.random.uniform(1000, 15000)
#                 base[f'cost_per_net_ton_co2_{s}_mean'] = np.random.uniform(50, 500)
            
#             data.append(base)
            
#     mock_df = pd.DataFrame(data)

#     # --- 2. Set up mock loggers ---
#     # Set up a console logger for demonstration
#     mock_logger = logging.getLogger('mock_logger')
#     mock_logger.setLevel(logging.INFO)
#     if not mock_logger.hasHandlers():
#         handler = logging.StreamHandler()
#         formatter = logging.Formatter('%(message)s')
#         handler.setFormatter(formatter)
#         mock_logger.addHandler(handler)
        
#     # --- 3. Set parameters ---
#     BUDGET = 5_000_000
#     LOFT_PROB = 0.3
#     EQUITY_FACTOR = 0.5 # 0 = pure carbon, 1 = pure equity
#     OUTPUT_DIR = '.' # Current directory
    
#     print(f"--- Running mock analysis with {N_EPI_RUNS} runs and {N_BUILDINGS} buildings ---")
    
#     # --- 4. Run the refactored function ---
#     last_baseline, all_results = run_greedy_algo(
#         scenario_budget=BUDGET,
#         prob_loft=LOFT_PROB,
#         df=mock_df,
#         scenario_list=SCENARIOS,
#         summary_logger=mock_logger,
#         detail_logger=mock_logger,
#         equity_factor=EQUITY_FACTOR,
#         output_dir=OUTPUT_DIR
#     )
    
#     print("\n\n--- ANALYSIS COMPLETE ---")
#     print(f"Total projects selected across all runs: {len(all_results)}")
#     if not all_results.empty:
#         print(f"Mean CO2 saved per project: {all_results[RANK_COL_CO2_SAVED].mean():.2f} tonnes")
#         print(f"Mean cost per project: £{all_results[RANK_COL_COST].mean():,.2f}")
    
#     print(f"\nBaseline candidates from *last* run: {len(last_baseline)}")



# # def run_greedy_algo(scenario_budget, prob_loft, df, scenario_list, summary_logger, detail_logger, equity_factor, output_dir):
# #     """
# #     Run greedy knapsack algorithm across multiple epistemic runs to select optimal retrofit projects.
    
# #     Args:
# #         scenario_budget: Total budget available
# #         prob_loft: Probability of existing loft insulation in older buildings
# #         df: Input DataFrame with building and scenario data
# #         scenario_list: List of retrofit scenarios to consider
# #         summary_logger: Logger for summary statistics
# #         detail_logger: Logger for detailed processing information
# #         equity_factor: 0 for pure carbon, 1 for pure equity 
    
# #     Returns:
# #         tuple: (baseline_selection DataFrame, combined_results DataFrame)
# #     """
# #     # Identify epistemic runs
# #     epistemic_col = 'epistemic_run_id'
# #     epistemic_runs = df[epistemic_col].unique()
    
# #     detail_logger.info(f"Starting analysis with {len(epistemic_runs)} epistemic runs and {len(scenario_list)} scenarios")
# #     detail_logger.info(f"Total buildings in dataset: {len(df)}")
# #     detail_logger.info(f"Scenarios: {', '.join(scenario_list)}")
# #     detail_logger.info(f"Equity factor: {equity_factor}")
    
# #     # Storage for results
# #     all_epistemic_results = []
# #     scenario_exclusion_stats = []
# #     scenario_performance_log = []
# #     equity_tracking = [] 
    
# #     # Process each epistemic run
# #     for epi_idx, epi_run in enumerate(epistemic_runs, 1):
# #         detail_logger.info(f"\n{'='*80}")
# #         detail_logger.info(f"Processing epistemic run {epi_idx}/{len(epistemic_runs)}: {epi_run}")
# #         detail_logger.info(f"{'='*80}")
        
# #         # Filter data for this epistemic run
# #         epi_df = df[df[epistemic_col] == epi_run].copy()
# #         epi_df = assign_random_loft(epi_df, prob_loft)
# #         detail_logger.info(f"Buildings in this epistemic run: {len(epi_df)}")
        
# #         res = []
        
# #         # Process each scenario
# #         for scenario_idx, scenario in enumerate(scenario_list, 1):
# #             detail_logger.info(f"\n  --- Scenario {scenario_idx}/{len(scenario_list)}: {scenario} ---")
            
# #             # Exclude buildings with positive CO2 values (these increase emissions)
# #             col_to_check = f'total_tonne_co2_saved_{scenario}_5yr_mean'
# #             mask = epi_df[col_to_check] > 0
# #             bad_upns_for_scenario = epi_df.loc[mask, 'upn'].unique().tolist()
            
# #             # Log exclusion statistics
# #             n_excluded_buildings = len(bad_upns_for_scenario)
# #             n_excluded_records = mask.sum()
# #             pct_excluded = (n_excluded_records / len(epi_df)) * 100 if len(epi_df) > 0 else 0
            
# #             detail_logger.info(f"    Excluded {n_excluded_buildings} UPNs ({n_excluded_records} records, {pct_excluded:.1f}%) with positive CO2 values")
            
# #             scenario_exclusion_stats.append({
# #                 'epistemic_run': epi_run,
# #                 'scenario': scenario,
# #                 'n_upns_excluded': n_excluded_buildings,
# #                 'n_records_excluded': n_excluded_records,
# #                 'pct_records_excluded': pct_excluded
# #             })
            
# #             # Filter out excluded buildings
# #             scenario_df = epi_df[~epi_df['upn'].isin(bad_upns_for_scenario)].copy()
            
# #             # For loft installation, only consider buildings without existing loft insulation
# #             if scenario == 'loft_installation':
# #                 detail_logger.info('    Removing buildings with existing loft insulation')
# #                 scenario_df = scenario_df[scenario_df['already_loft'] == False]
            
# #             detail_logger.info(f"    Remaining buildings for {scenario}: {len(scenario_df)}")
            
# #             # Flip signs for optimization
# #             scenario_df[f'flip_sign_total_tonne_co2_saved_{scenario}_5yr_mean'] = -scenario_df[f'total_tonne_co2_saved_{scenario}_5yr_mean']
# #             scenario_df[f'flip_sign_cost_per_net_ton_co2_{scenario}_mean'] = -scenario_df[f'cost_per_net_ton_co2_{scenario}_mean']
            
# #             # Apply equity weighting
# #             scenario_df['equity_weight'] = scenario_df['meta_socio_persona'].map(EQUITY_WEIGHTS)
# #             scenario_df[f'flip_sign_weighted_cost_per_ton_{scenario}'] = (
# #                 scenario_df[f'flip_sign_cost_per_net_ton_co2_{scenario}_mean'] * 
# #                 (1 + (scenario_df['equity_weight'] - 1) * equity_factor)
# #             )
            
# #             # Prepare ranking dataframe WITH PERSONA
# #             df_rank = scenario_df[[
# #                 'upn',
# #                 'meta_socio_persona',  # NEW: Include persona
# #                 'avg_gas_percentile',
# #                 f'{scenario}_cost_{scenario}_mean',
# #                 f'flip_sign_total_tonne_co2_saved_{scenario}_5yr_mean',
# #                 f'flip_sign_cost_per_net_ton_co2_{scenario}_mean',
# #                 f'flip_sign_weighted_cost_per_ton_{scenario}'
# #             ]].copy()
            
# #             df_rank['scenario'] = scenario
# #             df_rank['epistemic_run'] = epi_run
# #             df_rank.columns = [
# #                 'upn', 'meta_socio_persona', 'avg_gas_percentile', 
# #                 'cost of interventon_mean', 'total_ton_co2_saved', 
# #                 'cost_per_net_ton_co2_kg', 'weighted_cost_per_net_ton',
# #                 'scenario', 'epistemic_run'
# #             ]

# #             # Log scenario statistics
# #             valid_scenario_data = df_rank[~df_rank['cost_per_net_ton_co2_kg'].isna()]
# #             if len(valid_scenario_data) > 0:
# #                 detail_logger.info(f"    Valid buildings for optimization: {len(valid_scenario_data)}")
# #                 detail_logger.info(f"    Cost/tonne CO2 - Min: £{valid_scenario_data['cost_per_net_ton_co2_kg'].min():.2f}, "
# #                                  f"Median: £{valid_scenario_data['cost_per_net_ton_co2_kg'].median():.2f}, "
# #                                  f"Mean: £{valid_scenario_data['cost_per_net_ton_co2_kg'].mean():.2f}, "
# #                                  f"Max: £{valid_scenario_data['cost_per_net_ton_co2_kg'].max():.2f}")
# #                 detail_logger.info(f"    Total potential CO2 savings: {valid_scenario_data['total_ton_co2_saved'].sum():,.0f} tonnes")
# #                 detail_logger.info(f"    Total potential cost: £{valid_scenario_data['cost of interventon_mean'].sum():,.0f}")
# #             else:
# #                 detail_logger.warning(f"    No valid data for scenario {scenario}!")
            
# #             res.append(df_rank)
        
# #         # Combine all scenarios for this epistemic run
# #         res_df = pd.concat(res)
# #         wdf = res_df[~res_df['cost_per_net_ton_co2_kg'].isna()]
        
# #         detail_logger.info(f"\n  Combined dataset for epistemic run {epi_run}:")
# #         detail_logger.info(f"    Total valid building-scenario combinations: {len(wdf)}")
# #         detail_logger.info(f"    Unique buildings: {wdf['upn'].nunique()}")
        
# #         # Log scenario distribution
# #         scenario_counts = wdf.groupby('scenario').size()
# #         detail_logger.info(f"    Buildings per scenario:")
# #         for sc, count in scenario_counts.items():
# #             detail_logger.info(f"      {sc}: {count}")
        
# #         # Run greedy knapsack algorithm
# #         detail_logger.info(f"\n  Running greedy knapsack with budget: £{scenario_budget:,}")
        
# #         baseline_selection = (wdf
# #                             .sort_values('weighted_cost_per_net_ton', ascending=True)
# #                             .drop_duplicates(subset='upn', keep='first')
# #                             .reset_index(drop=True))
        
# #         detail_logger.info(f"    Candidate projects after deduplication: {len(baseline_selection)}")
        
# #         selected_projects_df, remaining_funds = true_greedy_knapsack(
# #             df_knapsack=baseline_selection,
# #             budget=scenario_budget,
# #             cost_column='cost of interventon_mean',
# #             efficiency_column='weighted_cost_per_net_ton'
# #         )
        
# #         # Log selection results
# #         selected_projects_df['epistemic_run'] = epi_run
# #         selected_projects_df['remaining_funds'] = remaining_funds
        
# #         total_cost = selected_projects_df['cost of interventon_mean'].sum()
# #         total_co2_saved = selected_projects_df['total_ton_co2_saved'].sum()
# #         avg_cost_per_tonne = total_cost / total_co2_saved if total_co2_saved > 0 else 0
        
# #         detail_logger.info(f"\n  RESULTS for epistemic run {epi_run}:")
# #         detail_logger.info(f"    Projects selected: {len(selected_projects_df)}")
# #         detail_logger.info(f"    Budget used: £{total_cost:,.0f} ({(total_cost/scenario_budget)*100:.1f}%)")
# #         detail_logger.info(f"    Remaining funds: £{remaining_funds:,.0f}")
# #         detail_logger.info(f"    Total CO2 saved: {total_co2_saved:,.0f} tonnes")
# #         detail_logger.info(f"    Average cost per tonne CO2: £{avg_cost_per_tonne:.2f}")
        
# #         # Scenario breakdown
# #         scenario_breakdown = selected_projects_df.groupby('scenario').agg({
# #             'upn': 'count',
# #             'cost of interventon_mean': 'sum',
# #             'total_ton_co2_saved': 'sum'
# #         }).rename(columns={'upn': 'n_projects'})
# #         scenario_breakdown['avg_cost_per_tonne'] = (
# #             scenario_breakdown['cost of interventon_mean'] / scenario_breakdown['total_ton_co2_saved']
# #         )
        
# #         detail_logger.info(f"\n  Scenario breakdown:")
# #         for sc in scenario_breakdown.index:
# #             row = scenario_breakdown.loc[sc]
# #             detail_logger.info(f"    {sc}:")
# #             detail_logger.info(f"      Projects: {row['n_projects']:.0f}")
# #             detail_logger.info(f"      Cost: £{row['cost of interventon_mean']:,.0f}")
# #             detail_logger.info(f"      CO2 saved: {row['total_ton_co2_saved']:,.0f} tonnes")
# #             detail_logger.info(f"      Avg cost/tonne: £{row['avg_cost_per_tonne']:.2f}")
        
# #         # Store performance metrics
# #         for sc in scenario_breakdown.index:
# #             row = scenario_breakdown.loc[sc]
# #             scenario_performance_log.append({
# #                 'epistemic_run': epi_run,
# #                 'scenario': sc,
# #                 'n_projects': row['n_projects'],
# #                 'total_cost': row['cost of interventon_mean'],
# #                 'total_co2_saved': row['total_ton_co2_saved'],
# #                 'avg_cost_per_tonne': row['avg_cost_per_tonne']
# #             })

# #         # ============================================================
# #         # NEW: EQUITY ANALYSIS FOR THIS EPISTEMIC RUN
# #         # ============================================================
        
# #         # Overall equity metrics
# #         overall_equity = calculate_social_equity_score(selected_projects_df)
        
# #         detail_logger.info(f"\n  EQUITY ANALYSIS for epistemic run {epi_run}:")
# #         detail_logger.info(f"    Vulnerable groups investment: {overall_equity['vulnerable_investment_pct']:.1f}%")
# #         detail_logger.info(f"    Equity concentration index: {overall_equity['equity_concentration']:.3f}")
# #         detail_logger.info(f"\n    Persona breakdown:")
        
# #         for persona, stats in overall_equity['persona_breakdown'].items():
# #             detail_logger.info(f"      {persona}: {stats['count']} projects ({stats['pct']:.1f}%)")
        
# #         # Scenario-specific equity analysis
# #         detail_logger.info(f"\n  Equity breakdown by scenario:")
# #         for scenario in scenario_list:
# #             scenario_equity = calculate_scenario_persona_metrics(selected_projects_df, scenario)
# #             if scenario_equity:
# #                 metrics = scenario_equity['equity_metrics']
# #                 detail_logger.info(f"\n    {scenario}:")
# #                 detail_logger.info(f"      Vulnerable: {metrics['vulnerable_investment_pct']:.1f}%")
                
# #                 for persona, stats in metrics['persona_breakdown'].items():
# #                     if stats['count'] > 0:
# #                         persona_stats = scenario_equity['persona_stats'].loc[persona]
# #                         detail_logger.info(f"        {persona}: {stats['count']} projects ({stats['pct']:.1f}%), "
# #                                          f"£{persona_stats['cost of interventon_mean']:,.0f}, "
# #                                          f"{persona_stats['total_ton_co2_saved']:,.0f} tonnes CO2")
                
# #                 # Store for cross-run analysis
# #                 equity_tracking.append({
# #                     'epistemic_run': epi_run,
# #                     'scenario': scenario,
# #                     'vulnerable_pct': metrics['vulnerable_investment_pct'],
# #                     'equity_concentration': metrics['equity_concentration'],
# #                     # **{f'{persona}_count': metrics['persona_breakdown'][persona]['count'] 
# #                        **{f'{persona}_count': metrics['persona_breakdown'].get(persona, {}).get('count', 0) for persona in EQUITY_WEIGHTS.keys()},
# #                     **{f'{persona}_pct': metrics['persona_breakdown'].get(persona, {} ).get('pct', 0 )  for persona in EQUITY_WEIGHTS.keys()},
# #                 })
        
        
        
# #         all_epistemic_results.append(selected_projects_df)
    
# #     # ============================================================================
# #     # COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS - SAVE TO SUMMARY FILE
# #     # ============================================================================
    
# #     combined_results = pd.concat(all_epistemic_results, ignore_index=True)
    
# #     summary_logger.info(f"\n{'='*80}")
# #     summary_logger.info("COMPREHENSIVE ANALYSIS ACROSS ALL EPISTEMIC RUNS")
# #     summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft}")
# #     summary_logger.info(f"{'='*80}\n")
    
# #     # 1. Summary by epistemic run
# #     summary_logger.info("1. SUMMARY BY EPISTEMIC RUN")
# #     summary_logger.info("-" * 80)
# #     summary_by_run = combined_results.groupby('epistemic_run').agg({
# #         'upn': 'count',
# #         'cost of interventon_mean': 'sum',
# #         'total_ton_co2_saved': 'sum',
# #         'remaining_funds': 'first'
# #     }).rename(columns={'upn': 'n_projects'})
    
# #     for epi_run in summary_by_run.index:
# #         row = summary_by_run.loc[epi_run]
# #         summary_logger.info(f"  Run {epi_run}:")
# #         summary_logger.info(f"    Projects: {row['n_projects']:.0f}")
# #         summary_logger.info(f"    Total cost: £{row['cost of interventon_mean']:,.0f}")
# #         summary_logger.info(f"    CO2 saved: {row['total_ton_co2_saved']:,.0f} tonnes")
# #         summary_logger.info(f"    Remaining: £{row['remaining_funds']:,.0f}\n")
    
# #     # 2. Overall statistics
# #     summary_logger.info("\n2. OVERALL STATISTICS ACROSS EPISTEMIC RUNS")
# #     summary_logger.info("-" * 80)
# #     summary_logger.info(f"  Mean projects selected: {summary_by_run['n_projects'].mean():.1f} ± {summary_by_run['n_projects'].std():.1f}")
# #     summary_logger.info(f"  Range: {summary_by_run['n_projects'].min():.0f} - {summary_by_run['n_projects'].max():.0f}")
# #     summary_logger.info(f"  Mean CO2 saved: {summary_by_run['total_ton_co2_saved'].mean():,.0f} ± {summary_by_run['total_ton_co2_saved'].std():,.0f} tonnes")
# #     summary_logger.info(f"  Range: {summary_by_run['total_ton_co2_saved'].min():,.0f} - {summary_by_run['total_ton_co2_saved'].max():,.0f} tonnes")
# #     summary_logger.info(f"  Mean cost: £{summary_by_run['cost of interventon_mean'].mean():,.0f} ± £{summary_by_run['cost of interventon_mean'].std():,.0f}")
    
# #     # 3. Scenario performance analysis
# #     summary_logger.info("\n\n3. SCENARIO PERFORMANCE ANALYSIS")
# #     summary_logger.info("-" * 80)
# #     scenario_perf_df = pd.DataFrame(scenario_performance_log)
    
# #     scenario_summary = scenario_perf_df.groupby('scenario').agg({
# #         'n_projects': ['mean', 'std', 'min', 'max'],
# #         'total_co2_saved': ['mean', 'std', 'sum'],
# #         'avg_cost_per_tonne': ['mean', 'std', 'min', 'max']
# #     })
    
# #     for scenario in scenario_list:
# #         if scenario in scenario_summary.index:
# #             summary_logger.info(f"\n  {scenario.upper()}:")
# #             stats = scenario_summary.loc[scenario]
# #             summary_logger.info(f"    Projects per run: {stats[('n_projects', 'mean')]:.1f} ± {stats[('n_projects', 'std')]:.1f} "
# #                               f"(range: {stats[('n_projects', 'min')]:.0f}-{stats[('n_projects', 'max')]:.0f})")
# #             summary_logger.info(f"    CO2 saved per run: {stats[('total_co2_saved', 'mean')]:,.0f} ± {stats[('total_co2_saved', 'std')]:,.0f} tonnes")
# #             summary_logger.info(f"    Total CO2 saved (all runs): {stats[('total_co2_saved', 'sum')]:,.0f} tonnes")
# #             summary_logger.info(f"    Avg cost/tonne: £{stats[('avg_cost_per_tonne', 'mean')]:.2f} ± £{stats[('avg_cost_per_tonne', 'std')]:.2f}")
# #             summary_logger.info(f"    Cost/tonne range: £{stats[('avg_cost_per_tonne', 'min')]:.2f} - £{stats[('avg_cost_per_tonne', 'max')]:.2f}")
    
# #     # 4. Scenario selection frequency
# #     summary_logger.info("\n\n4. SCENARIO SELECTION FREQUENCY")
# #     summary_logger.info("-" * 80)
# #     scenario_freq = combined_results.groupby('scenario').size()
# #     total_selections = scenario_freq.sum()
# #     for scenario in scenario_list:
# #         if scenario in scenario_freq.index:
# #             count = scenario_freq[scenario]
# #             pct = (count / total_selections) * 100
# #             summary_logger.info(f"  {scenario}: {count} selections ({pct:.1f}%)")
    
# #     # 5. Building robustness analysis
# #     summary_logger.info("\n\n5. BUILDING ROBUSTNESS ANALYSIS")
# #     summary_logger.info("-" * 80)
# #     building_selection_freq = combined_results.groupby('upn').agg({
# #         'epistemic_run': 'count',
# #         'scenario': lambda x: x.mode()[0] if len(x) > 0 else None,
# #     #     'cost_per_net_ton_co2_kg': 'mean',
# #     #     'total_ton_co2_saved': 'mean'
# #     # }).rename(columns={'epistemic_run': 'times_selected'})
    
# #     # building_selection_freq['selection_rate'] = (
# #     #     building_selection_freq['times_selected'] / len(epistemic_runs)
# #     # )
    
# #     # # Robustness categories
# #     # always_selected = (building_selection_freq['selection_rate'] == 1).sum()
# #     # mostly_selected = ((building_selection_freq['selection_rate'] > 0.5) & 
# #     #                   (building_selection_freq['selection_rate'] < 1)).sum()
# #     # sometimes_selected = ((building_selection_freq['selection_rate'] > 0) & 
# #     #                      (building_selection_freq['selection_rate'] <= 0.5)).sum()
    
# #     # summary_logger.info(f"  Buildings selected in ALL {len(epistemic_runs)} runs: {always_selected}")
# #     # summary_logger.info(f"  Buildings selected in >50% of runs: {mostly_selected}")
# #     # summary_logger.info(f"  Buildings selected in ≤50% of runs: {sometimes_selected}")
# #     # summary_logger.info(f"  Total unique buildings selected: {len(building_selection_freq)}")
    
# #     # # Top 10 most robust selections
# #     # summary_logger.info(f"\n  Top 10 most robust buildings (selected most frequently):")
# #     # top_buildings = building_selection_freq.nlargest(10, 'selection_rate')
# #     # for idx, (upn, row) in enumerate(top_buildings.iterrows(), 1):
# #     #     summary_logger.info(f"    {idx}. UPN {upn}: selected {row['times_selected']}/{len(epistemic_runs)} times "
# #     #                       f"({row['selection_rate']*100:.0f}%), scenario: {row['scenario']}, "
# #     #                       f"avg cost/tonne: £{row['cost_per_net_ton_co2_kg']:.2f}")
    
# #     # # 6. Exclusion statistics analysis
# #     # summary_logger.info("\n\n6. EXCLUSION STATISTICS ANALYSIS")
# #     # summary_logger.info("-" * 80)
# #     # exclusion_df = pd.DataFrame(scenario_exclusion_stats)
# #     # exclusion_summary = exclusion_df.groupby('scenario').agg({
# #     #     'n_upns_excluded': ['mean', 'std'],
# #     #     'pct_records_excluded': ['mean', 'std']
# #     # })
    
# #     # for scenario in scenario_list:
# #     #     if scenario in exclusion_summary.index:
# #     #         stats = exclusion_summary.loc[scenario]
# #     #         summary_logger.info(f"  {scenario}:")
# #     #         summary_logger.info(f"    Avg UPNs excluded: {stats[('n_upns_excluded', 'mean')]:.1f} ± {stats[('n_upns_excluded', 'std')]:.1f}")
# #     #         summary_logger.info(f"    Avg % records excluded: {stats[('pct_records_excluded', 'mean')]:.1f}% ± {stats[('pct_records_excluded', 'std')]:.1f}%")
    
     
         
# #     # # ============================================================================
# #     # # COMPREHENSIVE EQUITY ANALYSIS ACROSS ALL EPISTEMIC RUNS
# #     # # ============================================================================
    
 
    
# #     # summary_logger.info(f"\n{'='*80}")
# #     # summary_logger.info("SOCIAL EQUITY ANALYSIS ACROSS ALL EPISTEMIC RUNS")
# #     # summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft} | Equity Factor: {equity_factor}")
# #     # summary_logger.info(f"{'='*80}\n")
    
# #     # # 1. Overall equity metrics
# #     # summary_logger.info("1. OVERALL EQUITY METRICS")
# #     # summary_logger.info("-" * 80)
    
# #     # overall_equity_all = calculate_social_equity_score(combined_results)
# #     # summary_logger.info(f"  Total projects across all runs: {overall_equity_all['total_count']}")
# #     # summary_logger.info(f"  Vulnerable groups (deprived + struggling): {overall_equity_all['vulnerable_count']} "
# #     #                    f"({overall_equity_all['vulnerable_investment_pct']:.1f}%)")
# #     # summary_logger.info(f"  Equity concentration index: {overall_equity_all['equity_concentration']:.3f}\n")
    
# #     # summary_logger.info("  Persona distribution across all selections:")
# #     # for persona, stats in overall_equity_all['persona_breakdown'].items():
# #     #     summary_logger.info(f"    {persona.ljust(15)}: {stats['count']:5d} projects ({stats['pct']:5.1f}%)")
    
# #     # # 2. Equity metrics by epistemic run
# #     # summary_logger.info("\n\n2. EQUITY METRICS BY EPISTEMIC RUN")
# #     # summary_logger.info("-" * 80)
    
# #     # equity_by_run = combined_results.groupby('epistemic_run').apply(
# #     #     lambda x: pd.Series({
# #     #         'vulnerable_pct': calculate_social_equity_score(x)['vulnerable_investment_pct'],
# #     #         'concentration': calculate_social_equity_score(x)['equity_concentration'],
# #     #         'n_projects': len(x)
# #     #     })
# #     # )
    
# #     # summary_logger.info(f"  Mean vulnerable investment: {equity_by_run['vulnerable_pct'].mean():.1f}% "
# #     #                    f"± {equity_by_run['vulnerable_pct'].std():.1f}%")
# #     # summary_logger.info(f"  Range: {equity_by_run['vulnerable_pct'].min():.1f}% - "
# #     #                    f"{equity_by_run['vulnerable_pct'].max():.1f}%")
# #     # summary_logger.info(f"  Mean concentration index: {equity_by_run['concentration'].mean():.3f} "
# #     #                    f"± {equity_by_run['concentration'].std():.3f}")
    
# #     # # 3. Equity metrics by scenario
# #     # summary_logger.info("\n\n3. EQUITY METRICS BY SCENARIO")
# #     # summary_logger.info("-" * 80)
    
# #     # equity_tracking_df = pd.DataFrame(equity_tracking)
    
# #     # for scenario in scenario_list:
# #     #     scenario_equity_data = equity_tracking_df[equity_tracking_df['scenario'] == scenario]
        
# #     #     if len(scenario_equity_data) > 0:
# #     #         summary_logger.info(f"\n  {scenario.upper()}:")
# #     #         summary_logger.info(f"    Vulnerable investment: {scenario_equity_data['vulnerable_pct'].mean():.1f}% "
# #     #                           f"± {scenario_equity_data['vulnerable_pct'].std():.1f}%")
# #     #         summary_logger.info(f"    Concentration: {scenario_equity_data['equity_concentration'].mean():.3f} "
# #     #                           f"± {scenario_equity_data['equity_concentration'].std():.3f}")
            
# #     #         summary_logger.info(f"\n    Persona distribution:")
# #     #         for persona in EQUITY_WEIGHTS.keys():
# #     #             count_col = f'{persona}_count'
# #     #             pct_col = f'{persona}_pct'
# #     #             if count_col in scenario_equity_data.columns:
# #     #                 mean_count = scenario_equity_data[count_col].mean()
# #     #                 mean_pct = scenario_equity_data[pct_col].mean()
# #     #                 std_pct = scenario_equity_data[pct_col].std()
# #     #                 summary_logger.info(f"      {persona.ljust(15)}: {mean_count:5.1f} projects "
# #     #                                   f"({mean_pct:5.1f}% ± {std_pct:4.1f}%)")
    
# #     # # 4. Compare equity factor impact (if equity_factor != 0)
# #     # if equity_factor > 0:
# #     #     summary_logger.info("\n\n4. EQUITY FACTOR IMPACT")
# #     #     summary_logger.info("-" * 80)
# #     #     summary_logger.info(f"  Equity factor applied: {equity_factor}")
# #     #     summary_logger.info(f"  This prioritizes investment in:")
# #     #     for persona, weight in sorted(EQUITY_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
# #     #         if weight > 1.0:
# #     #             summary_logger.info(f"    {persona}: {weight}x weighting")
    
# #     # # 5. Cost and CO2 efficiency by persona
# #     # summary_logger.info("\n\n5. COST & CO2 EFFICIENCY BY PERSONA")
# #     # summary_logger.info("-" * 80)
    
# #     # persona_efficiency = combined_results.groupby('meta_socio_persona').agg({
# #     #     'cost of interventon_mean': ['sum', 'mean'],
# #     #     'total_ton_co2_saved': ['sum', 'mean'],
# #     #     'cost_per_net_ton_co2_kg': 'mean',
# #     #     'upn': 'count'
# #     # }).rename(columns={'upn': 'n_projects'})
    
# #     # for persona in EQUITY_WEIGHTS.keys():
# #     #     if persona in persona_efficiency.index:
# #     #         stats = persona_efficiency.loc[persona]
# #     #         total_cost = stats[('cost of interventon_mean', 'sum')]
# #     #         total_co2 = stats[('total_ton_co2_saved', 'sum')]
# #     #         n_proj = stats[('n_projects', 'count')]
# #     #         avg_cost_per_tonne = stats[('cost_per_net_ton_co2_kg', 'mean')]
            
# #     #         summary_logger.info(f"\n  {persona}:")
# #     #         summary_logger.info(f"    Projects: {n_proj:.0f}")
# #     #         summary_logger.info(f"    Total cost: £{total_cost:,.0f}")
# #     #         summary_logger.info(f"    Total CO2 saved: {total_co2:,.0f} tonnes")
# #     #         summary_logger.info(f"    Avg cost per tonne CO2: £{avg_cost_per_tonne:.2f}")
    
# #     # summary_logger.info(f"\n{'='*80}\n")
    
# #     # # Save equity tracking data
# #     # equity_tracking_df.to_csv(
# #     #     os.path.join(output_dir, 'equity_tracking.csv'), 
# #     #     index=False
# #     # )
    
# #     # summary_logger.info(f"\n{'='*80}\n")
    
# #     # return baseline_selection, combined_results