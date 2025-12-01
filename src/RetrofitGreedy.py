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
POSTCODE_COL = 'postcode' 
INTERVENTION_COL = 'intervention' # e.g. 'loft_installation', 'wall_insulation'

# Metrics
COST_COL = 'optimizer_cost' 
CO2_COL = 'raw_mean' # Assumed POSITIVE (Savings) based on pre-processing
PERSONA_COL = 'meta_socio_persona' 

# --- Internal/Calculated Columns ---
EQUITY_WEIGHT_COL = 'equity_weight'
COST_PER_TON_COL = 'cost_per_net_ton_co2' # Calculated directly
WEIGHTED_COST_PER_TON_COL = 'weighted_cost_per_net_ton'

# --- Final Rank DataFrame Columns ---
RANK_COL_UPN = 'upn'
RANK_COL_PERSONA = 'meta_socio_persona'
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

# =============================================================================
# HELPER FUNCTION: PREPARE AND RANK DATA (Vectorized)
# =============================================================================

def _prepare_and_rank_data(df: pd.DataFrame, 
                          equity_factor: float,
                          detail_logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs vectorized calculations on the entire dataset at once.
    
    Logic Changes:
    1. No sign flipping (CO2 assumed positive savings).
    2. No deduplication (Input assumed to be 1 row per building).
    3. No scenario loop (Input contains mixed interventions).
    """
    detail_logger.info(f"  --- Preparing data (Vectorized Processing) ---")
    
    # 1. Validate Columns
    required_cols = [COST_COL, CO2_COL, PERSONA_COL, INTERVENTION_COL, UPN_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        detail_logger.error(f"CRITICAL: Missing columns: {missing}")
        return pd.DataFrame(), {}

    # 2. Filter Invalid CO2 (Safety Check)
    # Even if pre-processed, ensure we don't divide by zero or have negative savings
    # (assuming raw_mean should be > 0 for savings)
    initial_count = len(df)
    mask_valid_co2 = df[CO2_COL] > 0.0001 # Tolerance for zero division
    
    df_clean = df[mask_valid_co2].copy()
    
    n_excluded = initial_count - len(df_clean)
    pct_excluded = (n_excluded / initial_count * 100) if initial_count > 0 else 0
    if n_excluded > 0:
        detail_logger.warning(f"    Excluded {n_excluded} rows ({pct_excluded:.1f}%) with <= 0 CO2 savings.")
    
    if df_clean.empty:
        return pd.DataFrame(columns=FINAL_RANK_COLS), {'n_excluded': n_excluded}

    # 3. Vectorized Calculations
    
    # A. Calculate Cost Per Ton (Direct)
    df_clean[COST_PER_TON_COL] = df_clean[COST_COL] / df_clean[CO2_COL]
    
    # B. Apply Equity Weights
    # Map personas to weights. Fill missing personas with weight 1.0 (neutral)
    df_clean[EQUITY_WEIGHT_COL] = df_clean[PERSONA_COL].map(EQUITY_WEIGHTS).fillna(1.0)
    
    # Formula: Adjusted_Cost = Cost * (1 + (Weight - 1) * EquityFactor)
    df_clean[WEIGHTED_COST_PER_TON_COL] = (
        df_clean[COST_PER_TON_COL] * (1 + (df_clean[EQUITY_WEIGHT_COL] - 1) * equity_factor)
    )

    # 4. Format for Ranking
    # Select specific columns and rename them to standard names
    df_rank = df_clean[[
        UPN_COL,
        PERSONA_COL,
        COST_COL,
        CO2_COL,
        COST_PER_TON_COL,
        WEIGHTED_COST_PER_TON_COL,
        INTERVENTION_COL 
    ]].copy()
    
    df_rank.columns = FINAL_RANK_COLS
    
    return df_rank, {'n_excluded': n_excluded}

# =============================================================================
# HELPER FUNCTION: LOG EQUITY ANALYSIS
# =============================================================================

def _log_equity_analysis(selected_projects_df: pd.DataFrame,
                                    detail_logger: logging.Logger) -> Dict[str, Any]:
    """
    Performs and logs the equity analysis.
    """
    if selected_projects_df.empty:
        return {}

    equity_metrics = calculate_social_equity_score(selected_projects_df)
    
    detail_logger.info(f"\n  EQUITY ANALYSIS:")
    detail_logger.info(f"    Vulnerable groups investment: {equity_metrics['vulnerable_investment_pct']:.1f}%")
    detail_logger.info(f"    Equity concentration index: {equity_metrics['equity_concentration']:.3f}")
    
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
# HELPER FUNCTION: LOG COMPREHENSIVE SUMMARY
# =============================================================================

def log_comprehensive_summary(combined_results: pd.DataFrame,
                            equity_tracking: Dict,
                            scenario_budget: float,
                            equity_factor: float,
                            summary_logger: logging.Logger,
                            output_dir: str):
    """
    Logs the final analysis stats.
    """
    summary_logger.info(f"\n{'='*80}")
    summary_logger.info("COMPREHENSIVE ANALYSIS (Pre-Processed / Single-Pass)")
    summary_logger.info(f"Optimization Method: Greedy Knapsack")
    summary_logger.info(f"Budget: £{scenario_budget:,} | Equity Factor: {equity_factor}")
    summary_logger.info(f"{'='*80}\n")
    
    # 1. Overall statistics
    total_projects = len(combined_results)
    total_cost = combined_results[RANK_COL_COST].sum()
    total_co2 = combined_results[RANK_COL_CO2_SAVED].sum()
    
    summary_logger.info("1. OVERALL STATISTICS")
    summary_logger.info("-" * 80)
    summary_logger.info(f"  Total Projects Selected: {total_projects}")
    summary_logger.info(f"  Total Cost: £{total_cost:,.0f}")
    summary_logger.info(f"  Total CO2 Saved: {total_co2:,.0f} tonnes")
    
    # 2. Scenario Breakdown (Calculated from results)
    if not combined_results.empty:
        summary_logger.info("\n2. INTERVENTION BREAKDOWN")
        summary_logger.info("-" * 80)
        breakdown = combined_results.groupby(RANK_COL_SCENARIO).agg({
            RANK_COL_UPN: 'count',
            RANK_COL_COST: 'sum',
            RANK_COL_CO2_SAVED: 'sum'
        }).rename(columns={RANK_COL_UPN: 'Count'})
        
        breakdown['Avg Cost/Ton'] = breakdown[RANK_COL_COST] / breakdown[RANK_COL_CO2_SAVED]
        summary_logger.info("\n" + breakdown.to_string(float_format=lambda x: f"{x:,.1f}"))

    # 3. Equity Analysis
    summary_logger.info("\n3. SOCIAL EQUITY ANALYSIS")
    summary_logger.info("-" * 80)
    
    if equity_tracking:
        summary_logger.info(f"  Vulnerable groups investment: {equity_tracking.get('vulnerable_pct',0):.1f}%")
        summary_logger.info(f"  Equity concentration index: {equity_tracking.get('equity_concentration',0):.3f}\n")
        
        summary_logger.info("  Persona distribution:")
        for persona in EQUITY_WEIGHTS.keys():
            count_col = f'{persona}_count'
            pct_col = f'{persona}_pct'
            if count_col in equity_tracking:
                summary_logger.info(f"    {persona.ljust(15)}: {int(equity_tracking[count_col]):5d} projects ({equity_tracking[pct_col]:5.1f}%)")

    # Save outputs
    try:
        combined_results.to_csv(os.path.join(output_dir, 'selected_projects.csv'), index=False)
        pd.DataFrame([equity_tracking]).to_csv(os.path.join(output_dir, 'equity_metrics.csv'), index=False)
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
    Run greedy knapsack algorithm on pre-processed, pre-selected single-row-per-building data.
    """
    
    detail_logger.info(f"Starting analysis on pre-processed data.")
    detail_logger.info(f"Total rows in dataset: {len(df)}")
    detail_logger.info(f"Optimization Strategy: Greedy Knapsack (Weighted Cost/Ton)")
    detail_logger.info(f"Equity factor: {equity_factor}")

    # --- 1. Prepare and Rank (Vectorized) ---
    df_ranked, exclusion_stats = _prepare_and_rank_data(
        df=df,
        equity_factor=equity_factor,
        detail_logger=detail_logger
    )
    
    if df_ranked.empty:
        detail_logger.error("No valid data remaining after preparation. Aborting.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 2. Run Greedy Knapsack ---
    # Since input is already 1 row per building (pre-selected), 
    # we don't need deduplication logic. Just sort and cut.
    
    # Ensure sorted by Weighted Cost (Cheapest/Most Efficient first)
    df_ranked.sort_values(by=RANK_COL_WEIGHTED_COST, ascending=True, inplace=True)
    
    detail_logger.info(f"\n  Running greedy knapsack with budget: £{scenario_budget:,}")
    detail_logger.info(f"  Candidate projects: {len(df_ranked)}")
    
    selected_df, remaining_funds = true_greedy_knapsack(
        df_knapsack=df_ranked,
        budget=scenario_budget,
        cost_column=RANK_COL_COST,
        efficiency_column=RANK_COL_WEIGHTED_COST 
    )
    
    selected_df = selected_df.copy()
    selected_df['remaining_funds'] = remaining_funds

    # --- 3. Run Sankey Diagram ---
    try:
        print(f'Running sankey saved in {output_dir}')
        run_sankey_greedy(selected_df, output_dir)
    except Exception as e:
        detail_logger.error(f"Sankey generation failed: {e}")

    # --- 4. Equity Analysis ---
    equity_tracking = _log_equity_analysis(selected_df, detail_logger)

    # --- 5. Summary Logs ---
    log_comprehensive_summary(
        combined_results=selected_df,
        equity_tracking=equity_tracking,
        scenario_budget=scenario_budget,
        equity_factor=equity_factor,
        summary_logger=summary_logger,
        output_dir=output_dir
    )
    
    # Return empty df as baseline_selection since there was no comparison step
    return df_ranked, selected_df