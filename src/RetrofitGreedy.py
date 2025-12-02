import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Any

from src.RetrofitEquity import EQUITY_WEIGHTS, calculate_social_equity_score 
from src.GreedyAlgo import true_greedy_knapsack  
from .Sankey import run_sankey_greedy 

# =============================================================================
# CONSTANTS
# =============================================================================

# --- Input Columns ---
UPN_COL = 'upn'
PERSONA_COL = 'meta_socio_persona'
LOFT_EXISTS_COL = 'already_loft'
GAS_PERCENTILE_COL = 'avg_gas_percentile'

# --- Scenario/Dynamic Column Templates ---
# No {stat_measure} needed, data is pre-averaged
CO2_SAVED_TPL = 'total_tonne_co2_saved_{scenario}_5yr'
COST_PER_TON_TPL = '{scenario}_cost_per_total_energy_ton_{scenario}'
COST_TPL = '{scenario}_cost_{scenario}'

# Metrics
COST_COL = 'optimizer_cost' 
CO2_COL = 'raw_mean' # Assumed POSITIVE (Savings) based on pre-processing
PERSONA_COL = 'meta_socio_persona' 

# --- Internal/Calculated Columns ---
FLIP_CO2_TPL = 'flip_sign_total_tonne_co2_saved_{scenario}_5yr'
FLIP_COST_PER_TON_TPL = 'flip_sign_cost_per_net_ton_co2_{scenario}'
EQUITY_WEIGHT_COL = 'equity_weight'
COST_PER_TON_COL = 'cost_per_net_ton_co2' # Calculated directly
WEIGHTED_COST_PER_TON_COL = 'weighted_cost_per_net_ton'

# --- Final Rank DataFrame Columns (UPDATED: Removed Epi Run) ---
RANK_COL_UPN = 'upn'
RANK_COL_PERSONA = 'meta_socio_persona'
RANK_COL_GAS_PCT = 'avg_gas_percentile'
RANK_COL_COST = 'cost_of_intervention'
RANK_COL_CO2_SAVED = 'total_ton_co2_saved'
RANK_COL_COST_PER_TON = 'cost_per_net_ton_co2_kg'
RANK_COL_WEIGHTED_COST = 'weighted_cost_per_net_ton'
RANK_COL_SCENARIO = 'scenario'

# List of final columns for easy renaming
FINAL_RANK_COLS = [
    RANK_COL_UPN,
    RANK_COL_PERSONA,
    RANK_COL_COST,
    RANK_COL_CO2_SAVED,
    RANK_COL_COST_PER_TON,
    RANK_COL_WEIGHTED_COST,
    RANK_COL_SCENARIO
]

# def assign_random_loft(df, prob_loft):
#     """
#     Assign loft insulation status to buildings.
#     """
#     modern_ages = ['Post 1999']
    
#     # Check if column already exists to avoid overwriting if pre-processed
#     if 'already_loft' in df.columns:
#          return dfff

#     df['already_loft'] = np.where(
#         df['premise_age'].isin(modern_ages),
#         True,
#         np.random.random(len(df)) < prob_loft
#     )
#     return df

# =============================================================================
# HELPER FUNCTION: PREPARE AND RANK DATA (Vectorized)
# =============================================================================

def _prepare_and_rank_data(df: pd.DataFrame, 
                          equity_factor: float,
                          detail_logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filters, weights, and formats data for a single scenario.
    """
    detail_logger.info(f"\n  --- Processing scenario: {scenario} ---")
    
    # --- Define dynamic column names ---
    co2_col = CO2_SAVED_TPL.format(scenario=scenario)
    cost_col = COST_TPL.format(scenario=scenario)
    cost_per_ton_col = COST_PER_TON_TPL.format(scenario=scenario)
    
    # Check if required columns exist
    required_cols = [co2_col, cost_col, cost_per_ton_col, PERSONA_COL]
    
    # Handle optional Gas Percentile
    if GAS_PERCENTILE_COL in epi_df.columns:
        required_cols.append(GAS_PERCENTILE_COL)

    missing_cols = [col for col in required_cols if col not in epi_df.columns]
    
    if missing_cols:
        detail_logger.warning(f"    Missing required columns for scenario {scenario}: {missing_cols}. Skipping.")
        stats = {
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
    
    # 1. Validate Columns
    required_cols = [COST_COL, CO2_COL, PERSONA_COL, INTERVENTION_COL, UPN_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        detail_logger.error(f"CRITICAL: Missing columns: {missing}")
        return pd.DataFrame(), {}

    # --- 3. Apply calculations ---
    
    # --- Flipped (Positive) Column Names ---
    flip_co2_col = FLIP_CO2_TPL.format(scenario=scenario)
    flip_cost_per_ton_col = FLIP_COST_PER_TON_TPL.format(scenario=scenario)
    weighted_cost_col = WEIGHTED_COST_PER_TON_TPL.format(scenario=scenario)
    
    # --- Perform the sign flip (CO2 savings are negative in input) ---
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
        GAS_PERCENTILE_COL if GAS_PERCENTILE_COL in scenario_df else UPN_COL, 
        cost_col,
        flip_co2_col,
        flip_cost_per_ton_col,
        weighted_cost_col
    ]
    
    if GAS_PERCENTILE_COL not in scenario_df:
         scenario_df[GAS_PERCENTILE_COL] = 0 

    df_rank = scenario_df[source_cols].copy()
    df_rank[RANK_COL_SCENARIO] = scenario
    
    # Set standardized column names
    df_rank.columns = FINAL_RANK_COLS

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
# HELPER FUNCTION: LOG EQUITY ANALYSIS
# =============================================================================

# CHANGED: Removed epi_run parameter
def _log_equity_analysis(selected_projects_df: pd.DataFrame,
                                    detail_logger: logging.Logger) -> Dict[str, Any]:
    """
    Performs and logs the equity analysis.
    """
    equity_metrics = calculate_social_equity_score(selected_projects_df)
    
    detail_logger.info(f"\n  EQUITY ANALYSIS:")
    detail_logger.info(f"    Vulnerable groups investment: {equity_metrics['vulnerable_investment_pct']:.1f}%")
    detail_logger.info(f"    Equity concentration index: {equity_metrics['equity_concentration']:.3f}")
    
    # Store for analysis
    equity_tracking = {
        'vulnerable_pct': equity_metrics['vulnerable_investment_pct'],
        'equity_concentration': equity_metrics['equity_concentration'],
        **{f'{persona}_count': equity_metrics['persona_breakdown'].get(persona, {}).get('count', 0) 
        for persona in EQUITY_WEIGHTS.keys()},
        **{f'{persona}_pct': equity_metrics['persona_breakdown'].get(persona, {}).get('pct', 0) 
        for persona in EQUITY_WEIGHTS.keys()},
    }
            
    return equity_tracking

# =============================================================================
# HELPER FUNCTION: PROCESS BATCH (Deterministic)
# =============================================================================

# CHANGED: Renamed from _process_single_run, removed run_id arg
def _process_optimization_batch(epi_df_full: pd.DataFrame,
                        scenario_list: List[str],
                        prob_loft: float,
                        equity_factor: float,
                        scenario_budget: float,
                        detail_logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict], Dict, pd.DataFrame]:
    """
    Runs the full analysis for the loaded dataset (deterministic).
    """
    # --- 1. Prepare data ---
    epi_df = epi_df_full.copy()
    # epi_df = assign_random_loft(epi_df, prob_loft)
    detail_logger.info(f"Buildings in this processing batch: {len(epi_df)}")
    
    all_scenario_ranks = []
    all_exclusion_stats = []
    
    # --- 2. Process all scenarios --- 
    for scenario in scenario_list:
        df_rank, exclusion_stats = _process_scenario(
            epi_df, scenario, equity_factor, detail_logger
        )
        if not df_rank.empty:
            all_scenario_ranks.append(df_rank)
        
        all_exclusion_stats.append(exclusion_stats)

    if not all_scenario_ranks:
        detail_logger.warning(f"No valid data for any scenario. Skipping.")
        return pd.DataFrame(), [], [], {}, pd.DataFrame()
        
    # --- 3. Combine scenarios and filter ---
    res_df = pd.concat(all_scenario_ranks)
    del all_scenario_ranks # Memory cleanup
    
    # Filter for valid, optimizable projects
    wdf = res_df[~res_df[RANK_COL_COST_PER_TON].isna()]
    
    detail_logger.info(f"\n  Combined dataset for optimization:")
    detail_logger.info(f"    Total valid building-scenario combinations: {len(wdf)}")
    detail_logger.info(f"    Unique buildings: {wdf[RANK_COL_UPN].nunique()}")

    # --- 4. Run greedy knapsack ---
    detail_logger.info(f"\n  Running greedy knapsack with budget: £{scenario_budget:,}")
    

    # Get the best *potential* project for each building
    baseline_selection = (wdf
                        .sort_values(RANK_COL_WEIGHTED_COST, ascending=True)
                        .drop_duplicates(subset=RANK_COL_UPN, keep='first')
                        .reset_index(drop=True))
    
    del wdf
    del res_df
    
    detail_logger.info(f"    Candidate projects after deduplication: {len(baseline_selection)}")
    
    selected_projects_df, remaining_funds = true_greedy_knapsack(
        df_knapsack=baseline_selection,
        budget=scenario_budget,
        cost_column=RANK_COL_COST,
        efficiency_column=RANK_COL_WEIGHTED_COST 
    )
    
    # --- 5. Log selection results ---
    selected_projects_df = selected_projects_df.copy() 
    selected_projects_df['remaining_funds'] = remaining_funds
    
    total_cost = selected_projects_df[RANK_COL_COST].sum()
    total_co2_saved = selected_projects_df[RANK_COL_CO2_SAVED].sum()
    
    detail_logger.info(f"\n  RESULTS:")
    detail_logger.info(f"    Projects selected: {len(selected_projects_df)}")
    detail_logger.info(f"    Budget used: £{total_cost:,.0f} ({(total_cost/scenario_budget)*100:.1f}%)")
    detail_logger.info(f"    Total CO2 saved: {total_co2:,.0f} tonnes")
    
    # --- 6. Log scenario breakdown ---
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
            detail_logger.info(f"    {sc}: {row['n_projects']:.0f} projects")
            detail_logger.info(f"      Cost: £{row[RANK_COL_COST]:,.0f}")
            detail_logger.info(f"      CO2: {row[RANK_COL_CO2_SAVED]:,.0f} tonnes")
            
            scenario_performance_log.append({
                'scenario': sc,
                'n_projects': row['n_projects'],
                'total_cost': row[RANK_COL_COST],
                'total_co2_saved': row[RANK_COL_CO2_SAVED],
                'avg_cost_per_tonne': row['avg_cost_per_tonne']
            })
            
    # --- 7. Perform equity analysis ---
    # CHANGED: Removed ID arg
    equity_tracking = _log_equity_analysis(
        selected_projects_df, detail_logger
    )
    
    return selected_projects_df, all_exclusion_stats, scenario_performance_log, equity_tracking, baseline_selection


# =============================================================================
# HELPER FUNCTION: LOG COMPREHENSIVE SUMMARY
# =============================================================================

def log_comprehensive_summary(combined_results: pd.DataFrame,
                            equity_tracking: Dict,
                            scenario_budget: float,
                            equity_factor: float,
                            scenario_list: List[str],
                            summary_logger: logging.Logger,
                            output_dir: str):
    """
    Logs the final analysis.
    CHANGED: Removed "robustness" and "variability" logic (std dev) as we only have one run.
    """
    
    scenario_perf_df = pd.DataFrame(all_scenario_performance)
    exclusion_df = pd.DataFrame(all_exclusion_stats)
    equity_tracking_df = pd.DataFrame(all_equity_tracking)
    
    summary_logger.info(f"\n{'='*80}")
    summary_logger.info("COMPREHENSIVE ANALYSIS (Deterministic / Pre-Averaged)")
    summary_logger.info(f"Optimization Method: Greedy Knapsack")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Loft Probability: {prob_loft} | Equity Factor: {equity_factor}")
    summary_logger.info(f"{'='*80}\n")
    
    # --- 1. Overall statistics ---
    total_projects = len(combined_results)
    total_cost = combined_results[RANK_COL_COST].sum()
    total_co2 = combined_results[RANK_COL_CO2_SAVED].sum()
    
    summary_logger.info("1. OVERALL STATISTICS")
    summary_logger.info("-" * 80)
    summary_logger.info(f"  Total Projects Selected: {total_projects}")
    summary_logger.info(f"  Total Cost: £{total_cost:,.0f}")
    summary_logger.info(f"  Total CO2 Saved: {total_co2:,.0f} tonnes")
    
    # --- 2. Scenario performance ---
    if not scenario_perf_df.empty:
        summary_logger.info("\n2. SCENARIO PERFORMANCE ANALYSIS")
        summary_logger.info("-" * 80)
        
        # Simple groupby in case there are multiple entries for same scenario
        scenario_summary = scenario_perf_df.groupby('scenario').agg({
            'n_projects': 'sum',
            'total_co2_saved': 'sum',
            'avg_cost_per_tonne': 'mean' 
        })
        
        for scenario in scenario_list:
            if scenario in scenario_summary.index:
                summary_logger.info(f"\n  {scenario.upper()}:")
                stats = scenario_summary.loc[scenario]
                summary_logger.info(f"    Total Projects: {stats['n_projects']:.0f}")
                summary_logger.info(f"    Total CO2 Saved: {stats['total_co2_saved']:,.0f} tonnes")
                summary_logger.info(f"    Avg Cost/Tonne: £{stats['avg_cost_per_tonne']:.2f}")

    # --- 3. Exclusion statistics ---
    if not exclusion_df.empty:
        summary_logger.info("\n3. EXCLUSION STATISTICS")
        summary_logger.info("-" * 80)
        exclusion_summary = exclusion_df.groupby('scenario').agg({
            'n_upns_excluded': 'sum',
            'pct_records_excluded': 'mean'
        })
        
        for scenario in scenario_list:
            if scenario in exclusion_summary.index:
                stats = exclusion_summary.loc[scenario]
                summary_logger.info(f"  {scenario}: Excluded {stats['n_upns_excluded']} UPNs ({stats['pct_records_excluded']:.1f}%)")

    # --- 4. Equity Analysis ---
    summary_logger.info("\n4. SOCIAL EQUITY ANALYSIS")
    summary_logger.info("-" * 80)
    
    # Since we have one run, we can just use the first entry of equity tracking or recalc
    if not equity_tracking_df.empty:
        # Use values from the tracking DF (which came from _log_equity_analysis)
        latest_equity = equity_tracking_df.iloc[0]
        summary_logger.info(f"  Vulnerable groups investment: {latest_equity['vulnerable_pct']:.1f}%")
        summary_logger.info(f"  Equity concentration index: {latest_equity['equity_concentration']:.3f}\n")
        
        summary_logger.info("  Persona distribution:")
        for persona in EQUITY_WEIGHTS.keys():
            count_col = f'{persona}_count'
            pct_col = f'{persona}_pct'
            if count_col in latest_equity and pct_col in latest_equity:
                summary_logger.info(f"    {persona.ljust(15)}: {int(latest_equity[count_col]):5d} projects ({latest_equity[pct_col]:5.1f}%)")

    # --- Save outputs ---
    try:
        combined_results.to_csv(
            os.path.join(output_dir, 'selected_projects.csv'),
            index=False
        )
        if not equity_tracking_df.empty:
            equity_tracking_df.to_csv(os.path.join(output_dir, 'equity_metrics.csv'), index=False)
        
        summary_logger.info(f"\nAll outputs saved to {output_dir}")
    except Exception as e:
        summary_logger.error(f"Failed to save output files: {e}")
        
    summary_logger.info(f"\n{'='*80}\n")


# =============================================================================
# MAIN ORCHESTRATOR FUNCTION
# =============================================================================

def run_greedy_algo(scenario_budget: float, 
                    df: pd.DataFrame, 
                    scenario_list: List[str], # Kept in signature for compatibility, but unused for filtering
                    summary_logger: logging.Logger, 
                    detail_logger: logging.Logger, 
                    equity_factor: float, 
                    output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run greedy knapsack algorithm on pre-processed single-value data.
    
    UPDATED: 
    - No loop over epistemic runs.
    - No ID handling.
    """
    
    detail_logger.info(f"Starting analysis on pre-processed data.")
    detail_logger.info(f"Total buildings in dataset: {len(df)}")
    detail_logger.info(f"Optimization Strategy: Greedy Knapsack (Deterministic)")
    detail_logger.info(f"Equity factor: {equity_factor}")

    # --- Process the single dataset ---
    # CHANGED: Called the new batch function without run_id
    selected_df, ex_stats, perf_stats, eq_stats, baseline_sel = _process_optimization_batch(
        epi_df_full=df,
        scenario_list=scenario_list,
        prob_loft=prob_loft,
        equity_factor=equity_factor,
        scenario_budget=scenario_budget,
        detail_logger=detail_logger
    )

    # --- Run Sankey Diagram ---
    try:
        print(f'Running sankey saved in {output_dir}')
        run_sankey_greedy(selected_df, output_dir)
    except Exception as e:
        detail_logger.error(f"Sankey generation failed: {e}")

    # --- Save Results ---
    if selected_df.empty:
        detail_logger.error("No projects selected. Aborting.")
        return pd.DataFrame(), pd.DataFrame()

    # Wrap results in lists for the summary logger function compatibility
    all_exclusion_stats = ex_stats
    all_scenario_performance = perf_stats
    all_equity_tracking = [eq_stats] 

    # --- Call helper to log the comprehensive summary ---
    log_comprehensive_summary(
        combined_results=selected_df,
        all_exclusion_stats=all_exclusion_stats,
        all_scenario_performance=all_scenario_performance,
        all_equity_tracking=all_equity_tracking,
        scenario_budget=scenario_budget,
        equity_factor=equity_factor,
        scenario_list=scenario_list,
        summary_logger=summary_logger,
        output_dir=output_dir
    )
    
    return baseline_sel, selected_df
