# RetrofitGreedy.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Any
import gc  # For garbage collection
import glob # For finding result files

# --- Custom Imports ---
from src.RetrofitEquity import EQUITY_WEIGHTS, calculate_social_equity_score
from src.GreedyAlgo import true_greedy_knapsack
from src.EPCAlgo import select_epc_algo

# =============================================================================
# CONSTANTS
# =============================================================================
# --- Input Columns ---
EPISTEMIC_COL = 'epistemic_run_id'
UPN_COL = 'upn'
PERSONA_COL = 'meta_socio_persona'
LOFT_EXISTS_COL = 'already_loft'
GAS_PERCENTILE_COL = 'avg_gas_percentile'
EPC_COL = 'CURRENT_ENERGY_RATING'

# --- Scenario/Dynamic Column Templates ---
CO2_SAVED_TPL = 'total_tonne_co2_saved_{scenario}_5yr_{stat_measure}'
COST_PER_TON_TPL = '{scenario}_cost_per_total_energy_ton_{scenario}_{stat_measure}'
COST_TPL = '{scenario}_cost_{scenario}_{stat_measure}'

# --- Valid statistical measures ---
OPTIMIZATION_STAT = 'mean'

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
RANK_COL_MODE = 'selection_mode' # NEW: To track Baseline vs Targeted vs EPC

# Base list of final columns (Always include EPC now as we always run all 3)
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
    RANK_COL_EPI_RUN,
    EPC_COL
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assign_random_loft(df, prob_loft):
    """Assign loft insulation status based on age and probability."""
    modern_ages = ['Post 1999']
    df[LOFT_EXISTS_COL] = np.where(
        df['premise_age'].isin(modern_ages),
        True,
        np.random.random(len(df)) < prob_loft
    )
    return df

def _process_scenario(epi_df: pd.DataFrame,
                      scenario: str,
                      equity_factor: float,
                      detail_logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filters, weights, and formats data for a single scenario."""

    # --- Define dynamic column names ---
    co2_col_mean = CO2_SAVED_TPL.format(scenario=scenario, stat_measure=OPTIMIZATION_STAT)
    cost_col_mean = COST_TPL.format(scenario=scenario, stat_measure=OPTIMIZATION_STAT)
    cost_per_ton_col_mean = COST_PER_TON_TPL.format(scenario=scenario, stat_measure=OPTIMIZATION_STAT)

    co2_col_p95 = CO2_SAVED_TPL.format(scenario=scenario, stat_measure='p95')
    cost_col_p95 = COST_TPL.format(scenario=scenario, stat_measure='p95')
    cost_per_ton_col_p95 = COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p95')

    co2_col_p5 = CO2_SAVED_TPL.format(scenario=scenario, stat_measure='p5')
    cost_col_p5 = COST_TPL.format(scenario=scenario, stat_measure='p5')
    cost_per_ton_col_p5 = COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p5')

    required_cols = [
        co2_col_mean, cost_col_mean, cost_per_ton_col_mean,
        co2_col_p95, cost_col_p95, cost_per_ton_col_p95,
        co2_col_p5, cost_col_p5, cost_per_ton_col_p5,
        PERSONA_COL, GAS_PERCENTILE_COL, EPC_COL
    ]

    missing_cols = [col for col in required_cols if col not in epi_df.columns]
    if missing_cols:
        detail_logger.warning(f"    Missing required columns for {scenario}: {missing_cols}. Skipping.")
        return pd.DataFrame(columns=FINAL_RANK_COLS), {'scenario': scenario, 'n_records_excluded': 0}

    # --- Filtering ---
    mask = epi_df[co2_col_mean] > 0
    bad_upns = epi_df.loc[mask, UPN_COL].unique().tolist()
    n_excluded = mask.sum()
    
    exclusion_stats = {
        'scenario': scenario,
        'n_upns_excluded': len(bad_upns),
        'n_records_excluded': n_excluded,
        'pct_records_excluded': (n_excluded / len(epi_df) * 100) if len(epi_df) > 0 else 0
    }

    # --- OPTIMIZATION: This is the ONLY main copy we will make ---
    scenario_df = epi_df[~epi_df[UPN_COL].isin(bad_upns)].copy()

    if scenario == LOFT_SCENARIO:
        scenario_df = scenario_df[~scenario_df[LOFT_EXISTS_COL]]

    if scenario_df.empty:
        del scenario_df
        return pd.DataFrame(columns=FINAL_RANK_COLS), exclusion_stats

    # --- Calculations & Sign Flipping (OPERATE IN-PLACE on scenario_df) ---
    flip_co2_mean = FLIP_CO2_TPL.format(scenario=scenario, stat_measure='mean')
    flip_co2_p95 = FLIP_CO2_TPL.format(scenario=scenario, stat_measure='p95')
    flip_co2_p5 = FLIP_CO2_TPL.format(scenario=scenario, stat_measure='p5')

    flip_cpt_mean = FLIP_COST_PER_TON_TPL.format(scenario=scenario, stat_measure='mean')
    flip_cpt_p95 = FLIP_COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p95')
    flip_cpt_p5 = FLIP_COST_PER_TON_TPL.format(scenario=scenario, stat_measure='p5')

    weighted_cost_col = WEIGHTED_COST_PER_TON_TPL.format(scenario=scenario)

    for orig, flipped in [(co2_col_mean, flip_co2_mean), (co2_col_p95, flip_co2_p95), (co2_col_p5, flip_co2_p5),
                          (cost_per_ton_col_mean, flip_cpt_mean), (cost_per_ton_col_p95, flip_cpt_p95), (cost_per_ton_col_p5, flip_cpt_p5)]:
        scenario_df[flipped] = -scenario_df[orig]

    scenario_df[EQUITY_WEIGHT_COL] = scenario_df[PERSONA_COL].map(EQUITY_WEIGHTS)
    scenario_df[weighted_cost_col] = (
        scenario_df[flip_cpt_mean] * (1 + (scenario_df[EQUITY_WEIGHT_COL] - 1) * equity_factor)
    )
    scenario_df[RANK_COL_SCENARIO] = scenario

    # --- OPTIMIZATION: Rename columns IN-PLACE ---
    rename_map = {
        UPN_COL: RANK_COL_UPN,
        PERSONA_COL: RANK_COL_PERSONA,
        GAS_PERCENTILE_COL: RANK_COL_GAS_PCT,
        cost_col_mean: RANK_COL_COST,
        cost_col_p95: RANK_COL_COST_P95,
        cost_col_p5: RANK_COL_COST_P5,
        flip_co2_mean: RANK_COL_CO2_SAVED,
        flip_co2_p95: RANK_COL_CO2_SAVED_P95,
        flip_co2_p5: RANK_COL_CO2_SAVED_P5,
        flip_cpt_mean: RANK_COL_COST_PER_TON,
        flip_cpt_p95: RANK_COL_COST_PER_TON_P95,
        flip_cpt_p5: RANK_COL_COST_PER_TON_P5,
        weighted_cost_col: RANK_COL_WEIGHTED_COST,
        RANK_COL_SCENARIO: RANK_COL_SCENARIO,
        EPC_COL: EPC_COL,
        EPISTEMIC_COL: RANK_COL_EPI_RUN,
    }
    scenario_df.rename(columns=rename_map, inplace=True)
    
    # --- OPTIMIZATION: Select final columns and make ONE small copy ---
    final_cols_to_keep = [col for col in FINAL_RANK_COLS if col in scenario_df.columns]
    df_rank_final = scenario_df[final_cols_to_keep].copy()

    # --- OPTIMIZATION: Delete the large intermediate dataframe ---
    del scenario_df

    return df_rank_final, exclusion_stats

def _log_epistemic_run_equity_analysis(selected_projects_df: pd.DataFrame,
                                    epi_run: str,
                                    mode: str,
                                    detail_logger: logging.Logger) -> Dict[str, Any]:
    """Logs equity analysis for a specific run and mode."""
    equity_metrics = calculate_social_equity_score(selected_projects_df)
    
    detail_logger.info(f"    [Mode: {mode.upper()}] Equity Analysis:")
    detail_logger.info(f"      Vulnerable investment: {equity_metrics['vulnerable_investment_pct']:.1f}%")
    detail_logger.info(f"      Concentration index: {equity_metrics['equity_concentration']:.3f}")
    
    return {
        'epistemic_run': epi_run,
        'selection_mode': mode,
        'vulnerable_pct': equity_metrics['vulnerable_investment_pct'],
        'equity_concentration': equity_metrics['equity_concentration'],
        **{f'{persona}_count': equity_metrics['persona_breakdown'].get(persona, {}).get('count', 0) 
        for persona in EQUITY_WEIGHTS.keys()},
        **{f'{persona}_pct': equity_metrics['persona_breakdown'].get(persona, {}).get('pct', 0) 
        for persona in EQUITY_WEIGHTS.keys()},
    }

def _process_epistemic_run(epi_run: str,
                           epi_df_full: pd.DataFrame, # This is already a copy
                           scenario_list: List[str],
                           prob_loft: float,
                           equity_factor: float,
                           scenario_budget: float,
                           detail_logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
    """Runs all three scenarios (Baseline, Targeted, EPC) for one epistemic run."""
    
    # 1. Prepare the universal pool of interventions for this run
    # epi_df_full is *already* a copy, just assign_random_loft to it
    epi_df = assign_random_loft(epi_df_full, prob_loft)
    
    all_ranks, all_excl = [], []
    for scen in scenario_list:
        df_r, excl = _process_scenario(epi_df, scen, equity_factor, detail_logger)
        if not df_r.empty: all_ranks.append(df_r)
        excl['epistemic_run'] = epi_run
        all_excl.append(excl)

    if not all_ranks:
        del epi_df # Free memory
        return pd.DataFrame(), [], [], []

    # We are done with epi_df, free its memory
    del epi_df
    
    wdf = pd.concat(all_ranks)
    
    # We are done with the list of DFs, free it
    del all_ranks
    
    wdf[RANK_COL_EPI_RUN] = epi_run
    wdf = wdf.dropna(subset=[RANK_COL_COST, RANK_COL_COST_PER_TON]) 

    run_results = []
    run_perf = []
    run_equity = []

    # --- MODE 1: BASELINE (Pure Cost Effectiveness) ---
    detail_logger.info(f"  > Running BASELINE (Pure Cost) selection...")
    baseline_pool = (wdf.sort_values(RANK_COL_COST_PER_TON, ascending=True)
                        .drop_duplicates(subset=RANK_COL_UPN, keep='first'))
    
    res_base, rem_base = true_greedy_knapsack(
        df_knapsack=baseline_pool,
        budget=scenario_budget,
        cost_column=RANK_COL_COST,
        efficiency_column=RANK_COL_COST_PER_TON # Pure cost efficiency
    )
    if not res_base.empty:
        res_base = res_base.copy()
        res_base[RANK_COL_MODE] = 'baseline'
        res_base['remaining_funds'] = rem_base
        run_results.append(res_base)
        run_equity.append(_log_epistemic_run_equity_analysis(res_base, epi_run, 'baseline', detail_logger))
    
    del baseline_pool
    if 'res_base' in locals(): del res_base

    # --- MODE 2: TARGETED (Equity Weighted) ---
    detail_logger.info(f"  > Running TARGETED (Equity Weighted) selection...")
    targeted_pool = (wdf.sort_values(RANK_COL_WEIGHTED_COST, ascending=True)
                        .drop_duplicates(subset=RANK_COL_UPN, keep='first'))
    
    res_targ, rem_targ = true_greedy_knapsack(
        df_knapsack=targeted_pool,
        budget=scenario_budget,
        cost_column=RANK_COL_COST,
        efficiency_column=RANK_COL_WEIGHTED_COST # Weighted efficiency
    )
    if not res_targ.empty:
        res_targ = res_targ.copy()
        res_targ[RANK_COL_MODE] = 'targeted'
        res_targ['remaining_funds'] = rem_targ
        run_results.append(res_targ)
        run_equity.append(_log_epistemic_run_equity_analysis(res_targ, epi_run, 'targeted', detail_logger))

    del targeted_pool
    if 'res_targ' in locals(): del res_targ

    # --- MODE 3: EPC ---
    detail_logger.info(f"  > Running EPC selection...")
    res_epc, rem_epc = select_epc_algo(
        df_knapsack=wdf, # Pass the full wdf
        budget=scenario_budget,
        cost_column=RANK_COL_COST,
        efficiency_column=RANK_COL_COST_PER_TON 
    )
    if not res_epc.empty:
        res_epc = res_epc.copy()
        res_epc[RANK_COL_MODE] = 'epc'
        res_epc['remaining_funds'] = rem_epc
        run_results.append(res_epc)
        run_equity.append(_log_epistemic_run_equity_analysis(res_epc, epi_run, 'epc', detail_logger))
    
    if 'res_epc' in locals(): del res_epc
    
    # We are now done with wdf, free it
    del wdf

    combined_run_df = pd.concat(run_results) if run_results else pd.DataFrame()
    
    del run_results

    # Generate basic performance logs per mode
    if not combined_run_df.empty:
        grp = combined_run_df.groupby([RANK_COL_MODE, RANK_COL_SCENARIO])
        for (mode, sc), row in grp:
             run_perf.append({
                'epistemic_run': epi_run, 
                'selection_mode': mode,
                'scenario': sc,
                'n_projects': len(row),
                'total_cost': row[RANK_COL_COST].sum(),
                'total_co2': row[RANK_COL_CO2_SAVED].sum(),
            })

    gc.collect()

    return combined_run_df, all_excl, run_perf, run_equity


def _aggregate_results_from_disk(temp_dir: str, detail_logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all individual Parquet results from the temp_dir and concatenates them.
    """
    
    # 1. Find all 'result' (res) files
    res_files = glob.glob(os.path.join(temp_dir, "res_*.parquet"))
    
    # 2. Find all 'equity' (eq) files
    eq_files = glob.glob(os.path.join(temp_dir, "eq_*.parquet"))

    if not res_files:
        detail_logger.warning("No result files found in cache. Returning empty data.")
        return pd.DataFrame(), pd.DataFrame()

    detail_logger.info(f"Aggregating {len(res_files)} result files from cache...")
    combined_results = pd.concat(
        (pd.read_parquet(f) for f in res_files), 
        ignore_index=True
    )
    
    detail_logger.info(f"Aggregating {len(eq_files)} equity files from cache...")
    combined_equity = pd.DataFrame()
    if eq_files:
        combined_equity = pd.concat(
            (pd.read_parquet(f) for f in eq_files), 
            ignore_index=True
        )

    return combined_results, combined_equity

def log_comprehensive_summary(combined_results: pd.DataFrame, 
                              combined_equity: pd.DataFrame, 
                              budget: float, 
                              equity_factor: float, 
                              output_dir: str, 
                              summary_logger: logging.Logger):
    """Logs final comprehensive analysis split by mode."""
    summary_logger.info(f"\n{'='*80}\nCOMPREHENSIVE ANALYSIS (All Modes)\nBudget: £{budget:,} | Equity Factor: {equity_factor}\n{'='*80}")
    
    if combined_results.empty:
        summary_logger.warning("No projects selected across any run/mode.")
        return

    # Group by Mode AND Epistemic Run for stats
    summary = combined_results.groupby([RANK_COL_MODE, RANK_COL_EPI_RUN]).agg({
        RANK_COL_UPN: 'count',
        RANK_COL_COST: 'sum',
        RANK_COL_CO2_SAVED: 'sum'
    }).reset_index()

    # Average across epistemic runs for each mode
    mode_means = summary.groupby(RANK_COL_MODE).agg({
        RANK_COL_UPN: 'mean',
        RANK_COL_COST: 'mean',
        RANK_COL_CO2_SAVED: 'mean'
    })

    summary_logger.info("\nMean Performance per Mode (averaged across epistemic runs):")
    summary_logger.info(mode_means.to_string(float_format=lambda x: f"{x:,.1f}"))

    # Save full detailed results
    combined_results.to_csv(os.path.join(output_dir, 'all_modes_selected_projects.csv'), index=False)
    
    if not combined_equity.empty:
        combined_equity.to_csv(os.path.join(output_dir, 'all_modes_equity_tracking.csv'), index=False)

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_greedy_algo_epc(scenario_budget: float,
                    prob_loft: float,
                    df: pd.DataFrame,
                    scenario_list: List[str],
                    summary_logger: logging.Logger,
                    detail_logger: logging.Logger,
                    equity_factor: float,
                    output_dir: str) -> pd.DataFrame:
    """
    Main entry point with checkpointing.
    Saves results for each run to disk and skips if they already exist.
    """
    epistemic_runs = df[EPISTEMIC_COL].unique()
    detail_logger.info(f"Starting multi-mode analysis: {len(epistemic_runs)} runs for df with shape {df.shape}")

    # --- Define a cache directory for intermediate results ---
    temp_dir = os.path.join(output_dir, "epistemic_run_cache")
    os.makedirs(temp_dir, exist_ok=True)
    detail_logger.info(f"Using cache directory: {temp_dir}")

    for i, epi_run in enumerate(epistemic_runs, 1):
        
        # --- Define output filenames for this run ---
        # We use a unique, safe identifier for the run.
        # If epi_run is a string with weird characters, this might need sanitizing.
        # For now, let's assume it's a simple string or int.
        safe_epi_run = str(epi_run).replace(os.path.sep, '_') # Basic sanitization
        res_file = os.path.join(temp_dir, f"res_{safe_epi_run}.parquet")
        eq_file = os.path.join(temp_dir, f"eq_{safe_epi_run}.parquet")
        
        # --- CHECKPOINT LOGIC ---
        if os.path.exists(res_file):
            detail_logger.info(f"--- Run {i}/{len(epistemic_runs)}: {epi_run} (SKIPPING, result file found) ---")
            continue
        
        # --- If not skipped, process as normal ---
        detail_logger.info(f"\n--- Run {i}/{len(epistemic_runs)}: {epi_run} (Processing) ---")
        
        epi_df_subset = df[df[EPISTEMIC_COL] == epi_run].copy()
        
        res, excl, perf, eq = _process_epistemic_run(
            epi_run, epi_df_subset, scenario_list,
            prob_loft, equity_factor, scenario_budget, detail_logger
        )
        
        del epi_df_subset  # Free memory from the subset
        
        # --- Save results to disk instead of appending ---
        if not res.empty:
            res.to_parquet(res_file)
            
            if eq:
                pd.DataFrame(eq).to_parquet(eq_file)

        # --- Clean up all in-memory objects from this loop ---
        del res, excl, perf, eq
        
        if i % 10 == 0:
             gc.collect() # Force garbage collection every 10 runs

    # --- All processing is done. Delete the huge input DF. ---
    detail_logger.info("All epistemic runs processed. Deleting main dataframe.")
    del df
    gc.collect()

    # --- Aggregate all saved files from disk ---
    detail_logger.info("Aggregating all cached results...")
    combined_results, combined_equity = _aggregate_results_from_disk(temp_dir, detail_logger)
    
    if combined_results.empty:
        summary_logger.warning("Aggregation resulted in an empty DataFrame. Check cache.")
        return pd.DataFrame()

    detail_logger.info("Aggregation complete. Logging comprehensive summary.")
    log_comprehensive_summary(
        combined_results, combined_equity, scenario_budget, 
        equity_factor, output_dir, summary_logger
    )

    detail_logger.info("Analysis complete.")
    
    # (Optional) You could clean up the temp_dir here if you want
    # import shutil
    # shutil.rmtree(temp_dir)

    return combined_results