


import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Any

 
import os 

from src.RetrofitEquity import EQUITY_WEIGHTS,  calculate_social_equity_score 
import pandas as pd
import numpy as np
from src.GreedyAlgo import true_greedy_knapsack  

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
# CONSTANTS
# =============================================================================
# --- Input Columns ---
EPISTEMIC_COL = 'epistemic_run_id'
UPN_COL = 'upn'
PERSONA_COL = 'meta_socio_persona'
LOFT_EXISTS_COL = 'already_loft'
GAS_PERCENTILE_COL = 'avg_gas_percentile'

# --- Scenario/Dynamic Column Templates (with stat_measure placeholder) ---
CO2_SAVED_TPL = 'total_tonne_co2_saved_{scenario}_5yr_{stat_measure}'
COST_PER_TON_TPL = '{scenario}_cost_per_total_energy_ton_{scenario}_{stat_measure}'
COST_TPL = '{scenario}_cost_{scenario}_{stat_measure}'

# --- Valid statistical measures ---
VALID_STAT_MEASURES = ['mean', 'p95', 'p50', 'p5']
OPTIMIZATION_STAT = 'mean'  # Always optimize on mean
RANGE_STATS = ['p5', 'p95']  # Capture these for uncertainty ranges

# --- Scenario Names ---
LOFT_SCENARIO = 'loft_installation'

# --- Internal/Calculated Columns ---
FLIP_CO2_TPL = 'flip_sign_total_tonne_co2_saved_{scenario}_5yr_{stat_measure}'
FLIP_COST_PER_TON_TPL = 'flip_sign_cost_per_net_ton_co2_{scenario}_{stat_measure}'
EQUITY_WEIGHT_COL = 'equity_weight'
WEIGHTED_COST_PER_TON_TPL = 'flip_sign_weighted_cost_per_ton_{scenario}'

# --- Final Rank DataFrame Columns (Standardized) ---
RANK_COL_UPN = 'upn'
RANK_COL_PERSONA = 'meta_socio_persona'
RANK_COL_GAS_PCT = 'avg_gas_percentile'
RANK_COL_COST = 'cost_of_intervention_mean'
RANK_COL_COST_P95 = 'cost_of_intervention_p95'
RANK_COL_COST_P5 = 'cost_of_intervention_p5'

RANK_COL_CO2_SAVED = 'total_ton_co2_saved_mean'
RANK_COL_CO2_SAVED_P95 = 'total_ton_co2_saved_p95'
RANK_COL_CO2_SAVED_P5 = 'total_ton_co2_saved_p5'

RANK_COL_COST_PER_TON = 'cost_per_net_ton_co2_kg_mean'
RANK_COL_COST_PER_TON_P95 = 'cost_per_net_ton_co2_kg_p95'
RANK_COL_COST_PER_TON_P5 = 'cost_per_net_ton_co2_kg_p5'

RANK_COL_WEIGHTED_COST = 'weighted_cost_per_net_ton'
RANK_COL_SCENARIO = 'scenario'
RANK_COL_EPI_RUN = 'epistemic_run'

# List of final columns for easy renaming
FINAL_RANK_COLS = [
    RANK_COL_UPN,
    RANK_COL_PERSONA,
    RANK_COL_GAS_PCT,
    RANK_COL_COST,
    RANK_COL_COST_P95,
    RANK_COL_COST_P5,
    RANK_COL_CO2_SAVED,
    RANK_COL_CO2_SAVED_P95,
    RANK_COL_CO2_SAVED_P5,
    RANK_COL_COST_PER_TON,
    RANK_COL_COST_PER_TON_P95,
    RANK_COL_COST_PER_TON_P5,
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
    Always optimizes on MEAN values but captures P95 and P5 for range analysis.
    
    Args:
        epi_df: DataFrame with building data for this epistemic run
        scenario: Name of the retrofit scenario
        equity_factor: Weighting factor for equity (0 = pure carbon, 1 = pure equity)
        detail_logger: Logger for detailed information
    
    Returns:
        tuple: (DataFrame with ranked projects, Dict with exclusion statistics)
    """
    detail_logger.info(f"\n  --- Processing scenario: {scenario} (optimizing on {OPTIMIZATION_STAT}, capturing ranges) ---")
    
    # --- Define dynamic column names for ALL statistical measures ---
    # Optimization columns (mean)
    co2_col_mean = CO2_SAVED_TPL.format(scenario=scenario, stat_measure=OPTIMIZATION_STAT)
    cost_col_mean = COST_TPL.format(scenario=scenario, stat_measure=OPTIMIZATION_STAT)
    cost_per_ton_col_mean = COST_PER_TON_TPL.format(scenario=scenario, stat_measure=OPTIMIZATION_STAT)
    
    # Range columns (p95 and p5)
    co2_col_p95 = CO2_SAVED_TPL.format(scenario=scenario, stat_measure='p95')
    cost_col_p95 = COST_TPL.format(scenario=scenario, stat_measure='p95')
    cost_per_ton_col_p95 = COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p95')
    
    co2_col_p5 = CO2_SAVED_TPL.format(scenario=scenario, stat_measure='p5')
    cost_col_p5 = COST_TPL.format(scenario=scenario, stat_measure='p5')
    cost_per_ton_col_p5 = COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p5')
    
    # Check if required columns exist
    required_cols = [
        co2_col_mean, cost_col_mean, cost_per_ton_col_mean,
        co2_col_p95, cost_col_p95, cost_per_ton_col_p95,
        co2_col_p5, cost_col_p5, cost_per_ton_col_p5,
        PERSONA_COL, GAS_PERCENTILE_COL
    ]
    missing_cols = [col for col in required_cols if col not in epi_df.columns]
    
    if missing_cols:
        detail_logger.warning(f"    Missing required columns for scenario {scenario}: {missing_cols}. Skipping.")
        # Return empty results
        stats = {
            'epistemic_run': epi_df[EPISTEMIC_COL].iloc[0] if not epi_df.empty else 'unknown',
            'scenario': scenario,
            'n_upns_excluded': 0, 'n_records_excluded': 0, 'pct_records_excluded': 0
        }
        return pd.DataFrame(columns=FINAL_RANK_COLS), stats
        
    # --- 1. Exclude buildings with positive CO2 values (emissions increase) ---
    # Use MEAN for filtering decisions
    mask = epi_df[co2_col_mean] > 0
    bad_upns_for_scenario = epi_df.loc[mask, UPN_COL].unique().tolist()
    
    n_excluded_buildings = len(bad_upns_for_scenario)
    n_excluded_records = mask.sum()
    pct_excluded = (n_excluded_records / len(epi_df)) * 100 if len(epi_df) > 0 else 0
    
    detail_logger.info(f"    Excluded {n_excluded_buildings} UPNs ({n_excluded_records} records, {pct_excluded:.1f}%) with positive CO2 values (mean)")
    
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

    # --- 3. Apply calculations (using MEAN for optimization) ---
    
    # --- Define optimization/reporting columns ---
    # We will flip ALL CO2 and Cost/Ton columns to be positive.
    # The greedy algorithm will minimize the positive 'weighted_cost_col'.
    
    # --- Flipped (Positive) Column Names ---
    flip_co2_col_mean = FLIP_CO2_TPL.format(scenario=scenario, stat_measure='mean')
    flip_co2_col_p95 = FLIP_CO2_TPL.format(scenario=scenario, stat_measure='p95')
    flip_co2_col_p5 = FLIP_CO2_TPL.format(scenario=scenario, stat_measure='p5')
    
    flip_cost_per_ton_col_mean = FLIP_COST_PER_TON_TPL.format(scenario=scenario, stat_measure='mean')
    flip_cost_per_ton_col_p95 = FLIP_COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p95')
    flip_cost_per_ton_col_p5 = FLIP_COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p5')
    
    weighted_cost_col = WEIGHTED_COST_PER_TON_TPL.format(scenario=scenario)
    
    # --- Perform the sign flip for ALL stats ---
    # CO2 values are negative (savings), so -1 * value = positive
    scenario_df[flip_co2_col_mean] = -scenario_df[co2_col_mean]
    scenario_df[flip_co2_col_p95] = -scenario_df[co2_col_p95]
    scenario_df[flip_co2_col_p5] = -scenario_df[co2_col_p5]
    
    # Cost/Ton values are negative (cost/negative_savings), so -1 * value = positive
    scenario_df[flip_cost_per_ton_col_mean] = -scenario_df[cost_per_ton_col_mean]
    scenario_df[flip_cost_per_ton_col_p95] = -scenario_df[cost_per_ton_col_p95]
    scenario_df[flip_cost_per_ton_col_p5] = -scenario_df[cost_per_ton_col_p5]
    
    # Apply equity weighting (on the positive mean cost/ton)
    scenario_df[EQUITY_WEIGHT_COL] = scenario_df[PERSONA_COL].map(EQUITY_WEIGHTS)
    scenario_df[weighted_cost_col] = (
        scenario_df[flip_cost_per_ton_col_mean] * (1 + (scenario_df[EQUITY_WEIGHT_COL] - 1) * equity_factor)
    )
    
    # --- 4. Prepare ranking dataframe with ALL statistical measures ---
    source_cols = [
        UPN_COL,
        PERSONA_COL,
        GAS_PERCENTILE_COL,
        cost_col_mean,
        cost_col_p95,
        cost_col_p5,
        flip_co2_col_mean,
        flip_co2_col_p95, 
        flip_co2_col_p5, 
 
        flip_cost_per_ton_col_mean, 
        flip_cost_per_ton_col_p95,
        flip_cost_per_ton_col_p5,
    
        weighted_cost_col
    ]
    
    df_rank = scenario_df[source_cols].copy()
    df_rank[RANK_COL_SCENARIO] = scenario
    
    # Set standardized column names (excluding epistemic_run which is added later)
    df_rank.columns = FINAL_RANK_COLS[:-1]

    # --- 5. Log scenario statistics with ranges ---
    valid_scenario_data = df_rank[~df_rank[RANK_COL_COST_PER_TON].isna()]
    if len(valid_scenario_data) > 0:
        detail_logger.info(f"    Valid buildings for optimization: {len(valid_scenario_data)}")
        detail_logger.info(f"    Cost/tonne CO2 (mean) - Min: £{valid_scenario_data[RANK_COL_COST_PER_TON].min():.2f}, "
                         f"Median: £{valid_scenario_data[RANK_COL_COST_PER_TON].median():.2f}, "
                         f"Max: £{valid_scenario_data[RANK_COL_COST_PER_TON].max():.2f}")
        detail_logger.info(f"    Cost/tonne CO2 (p95)  - Min: £{valid_scenario_data[RANK_COL_COST_PER_TON_P95].min():.2f}, "
                         f"Median: £{valid_scenario_data[RANK_COL_COST_PER_TON_P95].median():.2f}, "
                         f"Max: £{valid_scenario_data[RANK_COL_COST_PER_TON_P95].max():.2f}")
        detail_logger.info(f"    Cost/tonne CO2 (p5)   - Min: £{valid_scenario_data[RANK_COL_COST_PER_TON_P5].min():.2f}, "
                         f"Median: £{valid_scenario_data[RANK_COL_COST_PER_TON_P5].median():.2f}, "
                         f"Max: £{valid_scenario_data[RANK_COL_COST_PER_TON_P5].max():.2f}")
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


# =============================================================================
# HELPER FUNCTION: CREATE VISUALIZATIONS
# =============================================================================

def _create_range_visualizations(summary_by_run: pd.DataFrame,
                                 output_dir: str,
                                 summary_logger: logging.Logger):
    """
    Generates and saves error bar plots for Cost and CO2, showing the
    mean, p5, and p95 ranges for each epistemic run.
    """
    summary_logger.info("\n\n7. GENERATING VISUALIZATIONS")
    summary_logger.info("-" * 80)
    
    try:
        # --- Plot 1: CO2 Saved Ranges ---
        
        # Sort by mean CO2 for a cleaner plot
        plot_df_co2 = summary_by_run.sort_values(RANK_COL_CO2_SAVED).reset_index()
        
        # Calculate asymmetric errors: (mean - p5_value, p95_value - mean)
        lower_err_co2 = plot_df_co2[RANK_COL_CO2_SAVED] - plot_df_co2[RANK_COL_CO2_SAVED_P5]
        upper_err_co2 = plot_df_co2[RANK_COL_CO2_SAVED_P95] - plot_df_co2[RANK_COL_CO2_SAVED]
        y_errors_co2 = [lower_err_co2, upper_err_co2]

        plt.figure(figsize=(15, 7))
        plt.errorbar(
            x=plot_df_co2.index,
            y=plot_df_co2[RANK_COL_CO2_SAVED],
            yerr=y_errors_co2,
            fmt='o',         # 'o' for the mean value marker
            capsize=5,       # Cap on error bars
            label='Mean (with p5-p95 range)',
            alpha=0.8
        )
        plt.xlabel('Epistemic Run (Sorted by Mean CO2 Saved)')
        plt.ylabel('Total CO2 Saved (Tonnes)')
        plt.title('Total CO2 Saved per Epistemic Run (Mean with p5-p95 Range)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path_co2 = os.path.join(output_dir, 'epistemic_run_co2_ranges.png')
        plt.savefig(plot_path_co2)
        plt.close()
        summary_logger.info(f"Saved CO2 range plot to {plot_path_co2}")

        # --- Plot 2: Cost Ranges ---
        
        # Sort by mean Cost
        plot_df_cost = summary_by_run.sort_values(RANK_COL_COST).reset_index()

        lower_err_cost = plot_df_cost[RANK_COL_COST] - plot_df_cost[RANK_COL_COST_P5]
        upper_err_cost = plot_df_cost[RANK_COL_COST_P95] - plot_df_cost[RANK_COL_COST]
        y_errors_cost = [lower_err_cost, upper_err_cost]

        plt.figure(figsize=(15, 7))
        plt.errorbar(
            x=plot_df_cost.index,
            y=plot_df_cost[RANK_COL_COST],
            yerr=y_errors_cost,
            fmt='o',
            ecolor='orange',
            color='darkorange',
            capsize=5,
            label='Mean (with p5-p95 range)',
            alpha=0.8
        )
        plt.xlabel('Epistemic Run (Sorted by Mean Cost)')
        plt.ylabel('Total Cost (£)')
        plt.title('Total Cost per Epistemic Run (Mean with p5-p95 Range)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path_cost = os.path.join(output_dir, 'epistemic_run_cost_ranges.png')
        plt.savefig(plot_path_cost)
        plt.close()
        summary_logger.info(f"Saved Cost range plot to {plot_path_cost}")
        
    except Exception as e:
        summary_logger.error(f"Failed to generate visualizations: {e}")


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
    Always optimizes on MEAN but captures P95 and P5 ranges.
    
    Args:
        epi_run: Epistemic run identifier
        epi_df_full: DataFrame with building data for this run
        scenario_list: List of retrofit scenarios to consider
        prob_loft: Probability of existing loft insulation
        equity_factor: Weighting factor for equity (0 = pure carbon, 1 = pure equity)
        scenario_budget: Total budget available
        detail_logger: Logger for detailed information
    
    Returns:
        tuple: (selected projects DF, exclusion stats, performance stats, equity tracking, baseline selection DF)
    """
    # --- 1. Prepare data for this run ---
    epi_df = epi_df_full.copy()
    epi_df = assign_random_loft(epi_df, prob_loft)
    detail_logger.info(f"Buildings in this epistemic run: {len(epi_df)}")
    
    all_scenario_ranks = []
    all_exclusion_stats = []
    
    # --- 2. Process all scenarios --- these are the retrofit intervention scenarios (e.g. wall, loft, heat pump)
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
    
    # Filter for valid, optimizable projects (using MEAN)
    wdf = res_df[~res_df[RANK_COL_COST_PER_TON].isna()]
    
    detail_logger.info(f"\n  Combined dataset for epistemic run {epi_run}:")
    detail_logger.info(f"    Total valid building-scenario combinations: {len(wdf)}")
    detail_logger.info(f"    Unique buildings: {wdf[RANK_COL_UPN].nunique()}")

    # --- 4. Run greedy knapsack (using MEAN values for optimization) ---
    detail_logger.info(f"\n  Running greedy knapsack with budget: £{scenario_budget:,} (optimizing on MEAN)")
    
    # Get the best *potential* project for each building
    baseline_selection = (wdf
                        .sort_values(RANK_COL_WEIGHTED_COST, ascending=True)
                        .drop_duplicates(subset=RANK_COL_UPN, keep='first')
                        .reset_index(drop=True))
    
    detail_logger.info(f"    Candidate projects after deduplication: {len(baseline_selection)}")
    
    selected_projects_df, remaining_funds = true_greedy_knapsack(
        df_knapsack=baseline_selection,
        budget=scenario_budget,
        cost_column=RANK_COL_COST,  # Using MEAN cost
        efficiency_column=RANK_COL_WEIGHTED_COST  # Using weighted MEAN
    )
    
    # --- 5. Log selection results with ranges ---
    selected_projects_df = selected_projects_df.copy() 
    detail_logger.info(selected_projects_df.columns.tolist())
    selected_projects_df[RANK_COL_EPI_RUN] = epi_run
    selected_projects_df['remaining_funds'] = remaining_funds
    
    # Calculate totals for all stat measures
    total_cost_mean = selected_projects_df[RANK_COL_COST].sum()
    total_cost_p95 = selected_projects_df[RANK_COL_COST_P95].sum()
    total_cost_p5 = selected_projects_df[RANK_COL_COST_P5].sum()
    
    total_co2_saved_mean = selected_projects_df[RANK_COL_CO2_SAVED].sum()
    total_co2_saved_p95 = selected_projects_df[RANK_COL_CO2_SAVED_P95].sum()
    total_co2_saved_p5 = selected_projects_df[RANK_COL_CO2_SAVED_P5].sum()
    
    detail_logger.info(f"\n  RESULTS for epistemic run {epi_run} (optimized on MEAN):")
    detail_logger.info(f"    Projects selected: {len(selected_projects_df)}")
    detail_logger.info(f"    Budget used (mean): £{total_cost_mean:,.0f} ({(total_cost_mean/scenario_budget)*100:.1f}%)")
    detail_logger.info(f"    Budget range: £{total_cost_p5:,.0f} (p5) to £{total_cost_p95:,.0f} (p95)")
    detail_logger.info(f"    Total CO2 saved (mean): {total_co2_saved_mean:,.0f} tonnes")
    detail_logger.info(f"    CO2 saved range: {total_co2_saved_p5:,.0f} (p5) to {total_co2_saved_p95:,.0f} (p95) tonnes")
    
    # --- 6. Log scenario breakdown and collect performance stats ---
    scenario_performance_log = []
    if not selected_projects_df.empty:
        scenario_breakdown = selected_projects_df.groupby(RANK_COL_SCENARIO).agg({
            RANK_COL_UPN: 'count',
            RANK_COL_COST: 'sum',
            RANK_COL_COST_P95: 'sum',
            RANK_COL_COST_P5: 'sum',
            RANK_COL_CO2_SAVED: 'sum',
            RANK_COL_CO2_SAVED_P95: 'sum',
            RANK_COL_CO2_SAVED_P5: 'sum'
        }).rename(columns={RANK_COL_UPN: 'n_projects'})
        
        scenario_breakdown['avg_cost_per_tonne_mean'] = (
            scenario_breakdown[RANK_COL_COST] / scenario_breakdown[RANK_COL_CO2_SAVED]
        )
        scenario_breakdown['avg_cost_per_tonne_p95'] = (
            scenario_breakdown[RANK_COL_COST_P95] / scenario_breakdown[RANK_COL_CO2_SAVED_P95]
        )
        scenario_breakdown['avg_cost_per_tonne_p5'] = (
            scenario_breakdown[RANK_COL_COST_P5] / scenario_breakdown[RANK_COL_CO2_SAVED_P5]
        )
        
        detail_logger.info(f"\n  Scenario breakdown:")
        for sc in scenario_breakdown.index:
            row = scenario_breakdown.loc[sc]
            detail_logger.info(f"    {sc}: {row['n_projects']:.0f} projects")
            detail_logger.info(f"      Cost (mean): £{row[RANK_COL_COST]:,.0f}, "
                             f"Range: £{row[RANK_COL_COST_P5]:,.0f} to £{row[RANK_COL_COST_P95]:,.0f}")
            detail_logger.info(f"      CO2 (mean): {row[RANK_COL_CO2_SAVED]:,.0f} tonnes, "
                             f"Range: {row[RANK_COL_CO2_SAVED_P5]:,.0f} to {row[RANK_COL_CO2_SAVED_P95]:,.0f} tonnes")
            
            # Store performance metrics with ranges
            scenario_performance_log.append({
                'epistemic_run': epi_run,
                'scenario': sc,
                'n_projects': row['n_projects'],
                'total_cost_mean': row[RANK_COL_COST],
                'total_cost_p95': row[RANK_COL_COST_P95],
                'total_cost_p5': row[RANK_COL_COST_P5],
                'total_co2_saved_mean': row[RANK_COL_CO2_SAVED],
                'total_co2_saved_p95': row[RANK_COL_CO2_SAVED_P95],
                'total_co2_saved_p5': row[RANK_COL_CO2_SAVED_P5],
                'avg_cost_per_tonne_mean': row['avg_cost_per_tonne_mean'],
                'avg_cost_per_tonne_p95': row['avg_cost_per_tonne_p95'],
                'avg_cost_per_tonne_p5': row['avg_cost_per_tonne_p5']
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
    Now includes range analysis with P95 and P5 values.
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
    summary_logger.info(f"Optimization Method: Greedy on MEAN values")
    summary_logger.info(f"Ranges Captured: P5, MEAN, P95")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft} | Equity Factor: {equity_factor}")
    summary_logger.info(f"{'='*80}\n")
    
    # --- 1. Summary by epistemic run with ranges ---
    summary_logger.info("1. SUMMARY BY EPISTEMIC RUN (with uncertainty ranges)")
    summary_logger.info("-" * 80)
    summary_by_run = combined_results.groupby(RANK_COL_EPI_RUN).agg({
        RANK_COL_UPN: 'count',
        RANK_COL_COST: 'sum',
        RANK_COL_COST_P95: 'sum',
        RANK_COL_COST_P5: 'sum',
        RANK_COL_CO2_SAVED: 'sum',
        RANK_COL_CO2_SAVED_P95: 'sum',
        RANK_COL_CO2_SAVED_P5: 'sum',
        'remaining_funds': 'first'
    }).rename(columns={RANK_COL_UPN: 'n_projects'})
    
    for run in summary_by_run.index:
        row = summary_by_run.loc[run]
        summary_logger.info(f"\n  {run}:")
        summary_logger.info(f"    Projects: {row['n_projects']:.0f}")
        summary_logger.info(f"    Cost: £{row[RANK_COL_COST]:,.0f} (mean), Range: £{row[RANK_COL_COST_P5]:,.0f} to £{row[RANK_COL_COST_P95]:,.0f}")
        summary_logger.info(f"    CO2: {row[RANK_COL_CO2_SAVED]:,.0f} tonnes (mean), Range: {row[RANK_COL_CO2_SAVED_P5]:,.0f} to {row[RANK_COL_CO2_SAVED_P95]:,.0f}")
    
    # --- 2. Overall statistics with uncertainty ---
    summary_logger.info("\n2. OVERALL STATISTICS ACROSS EPISTEMIC RUNS (with uncertainty)")
    summary_logger.info("-" * 80)
    summary_logger.info(f"  Mean projects selected: {summary_by_run['n_projects'].mean():.1f} ± {summary_by_run['n_projects'].std():.1f}")
    summary_logger.info(f"  Mean CO2 saved (mean): {summary_by_run[RANK_COL_CO2_SAVED].mean():,.0f} ± {summary_by_run[RANK_COL_CO2_SAVED].std():,.0f} tonnes")
    summary_logger.info(f"  Mean CO2 saved (p95): {summary_by_run[RANK_COL_CO2_SAVED_P95].mean():,.0f} ± {summary_by_run[RANK_COL_CO2_SAVED_P95].std():,.0f} tonnes")
    summary_logger.info(f"  Mean CO2 saved (p5):  {summary_by_run[RANK_COL_CO2_SAVED_P5].mean():,.0f} ± {summary_by_run[RANK_COL_CO2_SAVED_P5].std():,.0f} tonnes")
    
    # --- 3. Scenario performance analysis with ranges ---
    if not scenario_perf_df.empty:
        summary_logger.info("\n\n3. SCENARIO PERFORMANCE ANALYSIS (with uncertainty ranges)")
        summary_logger.info("-" * 80)
        scenario_summary = scenario_perf_df.groupby('scenario').agg({
            'n_projects': ['mean', 'std', 'min', 'max'],
            'total_co2_saved_mean': ['mean', 'std', 'sum'],
            'total_co2_saved_p95': ['mean', 'std', 'sum'],
            'total_co2_saved_p5': ['mean', 'std', 'sum'],
            'avg_cost_per_tonne_mean': ['mean', 'std', 'min', 'max'],
            'avg_cost_per_tonne_p95': ['mean', 'std', 'min', 'max'],
            'avg_cost_per_tonne_p5': ['mean', 'std', 'min', 'max']
        })
        
        for scenario in scenario_list:
            if scenario in scenario_summary.index:
                summary_logger.info(f"\n  {scenario.upper()}:")
                stats = scenario_summary.loc[scenario]
                summary_logger.info(f"    Projects per run: {stats[('n_projects', 'mean')]:.1f} ± {stats[('n_projects', 'std')]:.1f}")
                summary_logger.info(f"    CO2 saved per run (mean): {stats[('total_co2_saved_mean', 'mean')]:,.0f} ± {stats[('total_co2_saved_mean', 'std')]:,.0f} tonnes")
                summary_logger.info(f"    CO2 saved range: {stats[('total_co2_saved_p5', 'mean')]:,.0f} (p5) to {stats[('total_co2_saved_p95', 'mean')]:,.0f} (p95) tonnes")
                summary_logger.info(f"    Cost per tonne (mean): £{stats[('avg_cost_per_tonne_mean', 'mean')]:.2f} ± £{stats[('avg_cost_per_tonne_mean', 'std')]:.2f}")
                summary_logger.info(f"    Cost per tonne range: £{stats[('avg_cost_per_tonne_p5', 'mean')]:.2f} (p5) to £{stats[('avg_cost_per_tonne_p95', 'mean')]:.2f} (p95)")

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
        RANK_COL_COST_PER_TON_P95: 'mean',
        RANK_COL_COST_PER_TON_P5: 'mean',
        RANK_COL_CO2_SAVED: 'mean',
        RANK_COL_CO2_SAVED_P95: 'mean',
        RANK_COL_CO2_SAVED_P5: 'mean'
    }).rename(columns={RANK_COL_EPI_RUN: 'times_selected'})
    
    building_selection_freq['selection_rate'] = (
        building_selection_freq['times_selected'] / n_epi_runs
    )
    
    summary_logger.info(f"  Total unique buildings selected: {len(building_selection_freq)}")
    summary_logger.info(f"  Buildings selected in ALL {n_epi_runs} runs: {(building_selection_freq['selection_rate'] == 1).sum()}")
    summary_logger.info(f"  Buildings selected in >50% of runs: {(building_selection_freq['selection_rate'] > 0.5).sum()}")
    
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

    # --- 3. Cost and CO2 efficiency by persona with ranges ---
    summary_logger.info("\n\n3. COST & CO2 EFFICIENCY BY PERSONA (with uncertainty ranges)")
    summary_logger.info("-" * 80)
    
    persona_efficiency = combined_results.groupby(RANK_COL_PERSONA).agg({
        RANK_COL_COST: ['sum', 'mean'],
        RANK_COL_COST_P95: ['sum', 'mean'],
        RANK_COL_COST_P5: ['sum', 'mean'],
        RANK_COL_CO2_SAVED: ['sum', 'mean'],
        RANK_COL_CO2_SAVED_P95: ['sum', 'mean'],
        RANK_COL_CO2_SAVED_P5: ['sum', 'mean'],
        RANK_COL_COST_PER_TON: 'mean',
        RANK_COL_COST_PER_TON_P95: 'mean',
        RANK_COL_COST_PER_TON_P5: 'mean',
        RANK_COL_UPN: 'count'
    }).rename(columns={RANK_COL_UPN: 'n_projects'})
    
    for persona in EQUITY_WEIGHTS.keys():
        if persona in persona_efficiency.index:
            stats = persona_efficiency.loc[persona]
            summary_logger.info(f"\n  {persona}:")
            summary_logger.info(f"    Projects: {stats[('n_projects', 'count')]:.0f}")
            summary_logger.info(f"    Avg cost per tonne CO2 (mean): £{stats[(RANK_COL_COST_PER_TON, 'mean')]:.2f}")
            summary_logger.info(f"    Avg cost per tonne range: £{stats[(RANK_COL_COST_PER_TON_P5, 'mean')]:.2f} to £{stats[(RANK_COL_COST_PER_TON_P95, 'mean')]:.2f}")
    
    # vis ranges 
    _create_range_visualizations(summary_by_run, output_dir, summary_logger)

    # --- Save outputs with ranges ---
    try:
        if not equity_tracking_df.empty:
            equity_tracking_df.to_csv(
                os.path.join(output_dir, 'equity_tracking_with_ranges.csv'), 
                index=False
            )
        if not scenario_perf_df.empty:
            scenario_perf_df.to_csv(
                os.path.join(output_dir, 'scenario_performance_with_ranges.csv'),
                index=False
            )
        combined_results.to_csv(
            os.path.join(output_dir, 'selected_projects_with_ranges.csv'),
            index=False
        )
        summary_logger.info(f"\nAll outputs saved to {output_dir}")
    except Exception as e:
        summary_logger.error(f"Failed to save output files: {e}")
        
    summary_logger.info(f"\n{'='*80}\n")


# =============================================================================
# MAIN ORCHESTRATOR FUNCTION (Simplified - always uses MEAN for optimization)
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
    
    OPTIMIZATION STRATEGY:
    - Always optimizes using MEAN values for costs and CO2 savings
    - Captures P95 and P5 values for all selected projects to show uncertainty ranges
    - This allows analysis of best/worst case scenarios without re-running the algorithm
    
    Args:
        scenario_budget: Total budget available for retrofits
        prob_loft: Probability that older buildings have existing loft insulation (0-1)
        df: Input DataFrame with building and scenario data (must include epistemic runs)
        scenario_list: List of retrofit scenarios to consider (e.g., ['loft_installation', 'wall_insulation'])
        summary_logger: Logger for summary statistics (writes to summary file)
        detail_logger: Logger for detailed processing information (writes to detail file)
        equity_factor: Weighting factor for social equity (0 = pure carbon optimization, 1 = pure equity)
        output_dir: Directory path for saving output files
    
    Returns:
        tuple: (baseline_selection DataFrame from last run, combined_results DataFrame from all runs)
               Both DataFrames include mean, p95, and p5 values for costs and CO2 savings
        
    Example:
        >>> baseline, results = run_greedy_algo(
        ...     scenario_budget=5_000_000,
        ...     prob_loft=0.3,
        ...     df=building_data,
        ...     scenario_list=['loft_installation', 'wall_insulation'],
        ...     summary_logger=summary_log,
        ...     detail_logger=detail_log,
        ...     equity_factor=0.5,
        ...     output_dir='./outputs'
        ... )
        >>> 
        >>> # Access mean values (used for optimization)
        >>> print(results['cost_of_intervention_mean'].sum())
        >>> 
        >>> # Access uncertainty ranges
        >>> print(f"Cost range: {results['cost_of_intervention_p5'].sum()} to {results['cost_of_intervention_p95'].sum()}")
        >>> print(f"CO2 range: {results['total_ton_co2_saved_p5'].sum()} to {results['total_ton_co2_saved_p95'].sum()}")
    """
    
    epistemic_runs = df[EPISTEMIC_COL].unique()
    
    detail_logger.info(f"Starting analysis with {len(epistemic_runs)} epistemic runs and {len(scenario_list)} scenarios")
    detail_logger.info(f"Optimization strategy: Greedy on MEAN, capturing P95/P5 ranges")
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
    
    # Preserve the original's return signature
    return baseline_selection, combined_results
 
