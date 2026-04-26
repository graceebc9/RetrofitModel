"""
Retrofit pre-processing pipeline with split aleatoric/epistemic uncertainty.

For each input file (one region's per-building per-run results), this script:

  1. Aggregates the N epistemic runs per building using Eve's law, but
     preserves the decomposition rather than collapsing it:
        - mean              : average of run means (central estimate)
        - aleatoric_std     : sqrt(mean of within-run variances)
                              -> building-level irreducible variability
        - epistemic_std     : std of within-run means
                              -> sensitivity to global / shared parameters

  2. Builds the building-level selection score using the aleatoric std only:
        capex_per_net_ton_aleatoric_sigma = mean + sigma * aleatoric_std
     This filters marginal buildings (e.g. fuel-poor with mean cost/tonne
     near zero) without baking systemic epistemic risk into per-building
     selection. Epistemic uncertainty is propagated at the portfolio level
     in a downstream step.

  3. Saves two artifacts per input file:
       (a) all_interventions_<file>.csv
           - one row per (building, scenario) with mean / aleatoric_std /
             epistemic_std for each of capex_per_net_ton, co2, capex,
             plus the aleatoric-sigma selection score and a rank.
       (b) per_run_means_<file>.parquet
           - WIDE-format table with one row per (upn, scenario, run_idx),
             holding cost_run_mean and co2_run_mean per row. This is the
             input to the portfolio-level epistemic propagation step
             (compute portfolio total per run, take std across runs).
             capex_per_net_ton is intentionally excluded -- it is not
             needed downstream, and dropping it cuts the artefact size
             by ~3x. Compressed snappy parquet keeps full-region runs
             tractable.

Author: Grace Colverd, revised 2026.
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
import os
import gc
import logging
import sys

from src.utils import is_running_on_hpc
from src.RetrofitUtils import filter_typology


# ============================================================================
# CONFIGURATION
# ============================================================================
is_hpc = is_running_on_hpc()

epc_yn = os.getenv('EPC_YN')
is_epc = (epc_yn == 'Y')

# sigma applied to the aleatoric std for the building-level selection score.
RISK_PENALTY_SIGMA = float(os.getenv('SIGMA'))

# Loft existing-measures fraction (some buildings already insulated and
# therefore disqualified from loft scenarios).
loft_flag = int(os.getenv('LOFT'))
loft_perc_list = [0.95] if loft_flag == 1 else [0.65]

# Hard cap on the (penalised) cost-per-tonne for a building to be selectable.
ABS_COST_CAP = 200000.0

SCENARIO_LIST = [
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'wall_installation',
    'join_heat_ins_decay',
    'heat_pump_only',
    'loft_installation',
]

# Metadata columns to carry through aggregation.
if is_epc:
    COLS_KEEP = [
        'postcode', 'premise_type', 'avg_gas_percentile',
        'CURRENT_ENERGY_RATING', 'POTENTIAL_ENERGY_RATING',
        'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY',
        'INSPECTION_DATE',
    ]
else:
    COLS_KEEP = ['postcode', 'premise_type', 'avg_gas_percentile']

# Metrics: maps a short name to the {sc}/{stat} pattern used by the upstream
# RetrofitModel2D output columns.
METRICS_MAP = {
    'capex_per_net_ton': '{sc}_capex_per_net_ton_co2_{sc}_{stat}',
    'co2':               '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}',
    'capex':             '{sc}_cost_{sc}_{stat}',
}

# ----------------------------------------------------------------------------
# Path configuration. Kept identical in spirit to the original script;
# anything portable should ideally move to a config file later.
# ----------------------------------------------------------------------------
if is_hpc:
    if not is_epc:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE'
    else:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/v10_logs_with_epc'
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE/120_log_file.csv'
else:
    if is_epc:
        LOG_DIR = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge'
    else:
        LOG_DIR = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE'
    REFERENCE_FILE = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/120_log_file.csv'

if is_epc:
    OUTPUT_BASE_DIR = (
        f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/'
        f'processed_all_scenarios'
    )
    LOG_FILE_PATH = (
        f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/'
        f'processing_log.txt'
    )
    ERROR_LOG_FILE = (
        f'4_optimized_priorities_epc/risk_sigma_{RISK_PENALTY_SIGMA}/'
        f'epc_processing_errors.txt'
    )
else:
    OUTPUT_BASE_DIR = (
        f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/'
        f'processed_all_scenarios'
    )
    LOG_FILE_PATH = (
        f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/'
        f'processing_log.txt'
    )
    ERROR_LOG_FILE = (
        f'4_optimized_priorities/risk_sigma_{RISK_PENALTY_SIGMA}/'
        f'processing_errors.txt'
    )

# Subdirectory for the per-run building means (portfolio-level inputs).
PER_RUN_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'per_run_means')


# ============================================================================
# HELPERS
# ============================================================================
def log_error_to_file(filename: str, error_msg: str) -> None:
    """Append an error entry to the error log."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(
            f"[{timestamp}] FILE: {filename}\n"
            f"ERROR: {error_msg}\n"
            f"{'-' * 40}\n"
        )


def setup_logging() -> None:
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()],
    )


# ============================================================================
# 1. AGGREGATION (EVE'S LAW WITH PRESERVED DECOMPOSITION)
# ============================================================================
def pool_epistemic_runs_decomposed(
    df: pd.DataFrame,
    scenarios: list,
    id_col: str = 'upn',
) -> pd.DataFrame:
    """
    Apply the Law of Total Variance to combine epistemic runs per building,
    keeping the aleatoric and epistemic components separate rather than
    summing them.

    For each (scenario, metric) we return three quantities:
        mean           = E_runs[ E_inner[Y] ]               # central estimate
        aleatoric_std  = sqrt( E_runs[ Var_inner[Y] ] )     # building-level noise
        epistemic_std  = sqrt( Var_runs[ E_inner[Y] ] )     # global-parameter noise

    The total std under Eve's law is sqrt(aleatoric_std^2 + epistemic_std^2),
    but we deliberately do not emit it here. Aleatoric is the right signal for
    building-level selection (catching marginal / take-back-prone buildings),
    epistemic is the right signal for portfolio-level systemic uncertainty.
    """
    logging.info("Pooling runs with split aleatoric/epistemic decomposition")
    df = df.copy()

    # Identify which (scenario, metric) pairs actually have both mean and std
    # columns present in the input.
    cols_to_process = {}
    for scn in scenarios:
        for metric_name, pattern in METRICS_MAP.items():
            raw_mean_col = pattern.format(sc=scn, stat='mean')
            raw_std_col = pattern.format(sc=scn, stat='std')
            if raw_mean_col in df.columns and raw_std_col in df.columns:
                base_name = f"{scn}_{metric_name}"
                cols_to_process[base_name] = {
                    'scenario': scn,
                    'metric': metric_name,
                    'mean_col': raw_mean_col,
                    'std_col': raw_std_col,
                }

    if not cols_to_process:
        logging.warning("No matching scenario columns found for aggregation.")
        return pd.DataFrame()

    # Pre-compute per-run variances (sigma^2) so groupby can average them.
    tmp_var_cols = []
    for base, cols in cols_to_process.items():
        var_col = f"_tmp_var_{base}"
        df[var_col] = df[cols['std_col']] ** 2
        cols['var_col'] = var_col
        tmp_var_cols.append(var_col)

    # Build the aggregation dictionary:
    #   - mean of run means -> central estimate
    #   - var of run means  -> epistemic variance
    #   - mean of run variances -> aleatoric variance
    agg_dict = {}
    for base, cols in cols_to_process.items():
        agg_dict[cols['mean_col']] = ['mean', 'var']
        agg_dict[cols['var_col']] = ['mean']

    grouped = df.groupby(id_col).agg(agg_dict)

    # Reconstruct the three quantities per (scenario, metric).
    final_stats = pd.DataFrame(index=grouped.index)
    for base, cols in cols_to_process.items():
        mu_total = grouped[(cols['mean_col'], 'mean')]
        var_epistemic = grouped[(cols['mean_col'], 'var')].fillna(0)
        var_aleatoric = grouped[(cols['var_col'], 'mean')]

        final_stats[f"{base}_mean"] = mu_total
        final_stats[f"{base}_aleatoric_std"] = np.sqrt(var_aleatoric)
        final_stats[f"{base}_epistemic_std"] = np.sqrt(var_epistemic)

    # Re-attach metadata (first value per upn).
    meta_cols = [c for c in COLS_KEEP if c in df.columns]
    df_meta = df.groupby(id_col)[meta_cols].first()

    df_final = pd.concat([df_meta, final_stats], axis=1).reset_index()

    # Tidy up the temporary variance columns on the original dataframe.
    df.drop(columns=tmp_var_cols, inplace=True)

    logging.info(f"Aggregated to {len(df_final)} unique buildings.")
    return df_final


# ============================================================================
# 2. PER-RUN MEANS FOR PORTFOLIO-LEVEL EPISTEMIC PROPAGATION
# ============================================================================
# Schema: WIDE format. One row per (upn, scenario, run_idx).
# Columns: upn, scenario, run_idx, cost_run_mean, co2_run_mean.
#
# This is denser than long-format by ~3x (no row duplication per metric)
# and we drop capex_per_net_ton from this artefact entirely -- the
# downstream optimiser only needs cost and co2 per-run totals to compute
# portfolio-level epistemic uncertainty, and uses percentile ratios for
# £/tCO2 rather than the per-run capex_per_net_ton values.
#
# At full scale (~5e5 buildings x 6 scenarios x 50 runs) this fits in
# memory and on disk where long-format with 3 metrics did not.
PER_RUN_COST_COL = 'cost_run_mean'
PER_RUN_CO2_COL = 'co2_run_mean'


def extract_per_run_means(
    df: pd.DataFrame,
    scenarios: list,
    id_col: str = 'upn',
) -> pd.DataFrame:
    """
    Return a wide-format dataframe of per-run building means for the two
    metrics needed downstream (capex, co2):

        upn | scenario | run_idx | cost_run_mean | co2_run_mean

    Each input row is already one (upn, run) pair; we read the per-run
    means out of the upstream {scn}_..._mean columns and reshape so each
    (upn, scenario, run_idx) is a single row.

    `run_idx` is a within-upn enumeration of the runs (0..N-1). We do not
    require the upstream to provide a stable run id because the epistemic
    propagation only needs to align rows across buildings *within the same
    epistemic world*. As long as runs are emitted in a consistent order
    per upn, position within the group is sufficient.

    Wide format + dropping capex_per_net_ton brings the artefact down to
    ~1/3 of the long-format size, which is what makes full-scale runs
    feasible.
    """
    logging.info("Extracting per-run building means for portfolio propagation")

    df = df.copy()
    # Stable ordering -> consistent run_idx alignment across buildings.
    df = df.sort_values(id_col, kind='mergesort').reset_index(drop=True)
    df['run_idx'] = df.groupby(id_col).cumcount()

    # The two metrics we keep. capex_per_net_ton is intentionally excluded.
    keep_metrics = {
        'capex': PER_RUN_COST_COL,
        'co2': PER_RUN_CO2_COL,
    }

    pieces = []
    for scn in scenarios:
        per_scn = {id_col: df[id_col].to_numpy(),
                   'run_idx': df['run_idx'].to_numpy()}
        any_found = False
        for metric_name, out_col in keep_metrics.items():
            mean_col = METRICS_MAP[metric_name].format(sc=scn, stat='mean')
            if mean_col not in df.columns:
                # Mark missing so we don't emit a half-populated row.
                per_scn[out_col] = None
                continue
            per_scn[out_col] = df[mean_col].to_numpy()
            any_found = True

        if not any_found:
            continue

        piece = pd.DataFrame(per_scn)
        piece['scenario'] = scn
        # Reorder for downstream readability.
        piece = piece[[id_col, 'scenario', 'run_idx',
                       PER_RUN_COST_COL, PER_RUN_CO2_COL]]
        # Drop rows where both metrics are NaN (i.e. scenario truly absent).
        valid = piece[[PER_RUN_COST_COL, PER_RUN_CO2_COL]].notna().any(axis=1)
        piece = piece[valid]
        if not piece.empty:
            pieces.append(piece)

    cols = [id_col, 'scenario', 'run_idx', PER_RUN_COST_COL, PER_RUN_CO2_COL]
    if not pieces:
        logging.warning("No per-run mean columns found to extract.")
        return pd.DataFrame(columns=cols)

    wide_df = pd.concat(pieces, ignore_index=True)

    # Downcast where safe to shrink the on-disk parquet further.
    for c in (PER_RUN_COST_COL, PER_RUN_CO2_COL):
        if c in wide_df.columns:
            wide_df[c] = pd.to_numeric(wide_df[c], downcast='float')
    wide_df['run_idx'] = pd.to_numeric(wide_df['run_idx'], downcast='unsigned')

    n_runs_avg = (
        wide_df.groupby([id_col, 'scenario'])['run_idx'].count().mean()
        if not wide_df.empty else 0
    )
    logging.info(
        f"Per-run table (wide): {len(wide_df):,} rows "
        f"({wide_df[id_col].nunique():,} buildings x "
        f"{wide_df['scenario'].nunique()} scenarios x "
        f"~{n_runs_avg:.0f} runs)"
    )
    return wide_df


# ============================================================================
# 3. BUILDING-LEVEL SELECTION SCORE (ALEATORIC-ONLY PENALTY)
# ============================================================================
def add_aleatoric_sigma_columns(
    df: pd.DataFrame,
    scenarios: list,
    sigma: float,
) -> pd.DataFrame:
    """
    Add the building-level risk-adjusted selection score:

        {scn}_{metric}_aleatoric_{sigma}sigma = mean + sigma * aleatoric_std

    Aleatoric std is used (not total std) because:
      - it captures building-level variability that does NOT average out
        across the portfolio, so it's the right signal to penalise individual
        marginal buildings (e.g. fuel-poor buildings near zero net CO2);
      - epistemic std applies symmetrically to all buildings under any given
        global-parameter draw, so it does not distinguish good from bad
        picks at selection time -- it is propagated at the portfolio level.
    """
    logging.info(f"Adding aleatoric-sigma selection columns (sigma={sigma})")
    new_cols = {}
    sigma_str = str(float(sigma))

    for scn in scenarios:
        for metric in METRICS_MAP.keys():
            mean_col = f"{scn}_{metric}_mean"
            ale_col = f"{scn}_{metric}_aleatoric_std"
            if mean_col in df.columns and ale_col in df.columns:
                new_name = f"{scn}_{metric}_aleatoric_{sigma_str}sigma"
                new_cols[new_name] = (
                    df[mean_col].to_numpy() + sigma * df[ale_col].to_numpy()
                )

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ============================================================================
# 4. PHYSICAL FILTER + EXISTING-MEASURES CONSTRAINT
# ============================================================================
def apply_physical_filters_for_optimisation(
    df: pd.DataFrame,
    scn: str,
) -> pd.DataFrame:
    """Drop rows where the central estimates are non-physical for this scn."""
    capex_col = f'{scn}_capex_per_net_ton_mean'
    co2_col = f'{scn}_co2_mean'
    return df[(df[capex_col] > 0) & (df[co2_col] > 0.1)]


def disqualified_loft_upns(
    df: pd.DataFrame,
    percent_existing: float,
) -> set:
    """
    Return the set of upns deemed to already have loft insulation, sampled
    deterministically. These upns are excluded from any loft-touching scenario.
    """
    unique_upns = df['upn'].unique()
    n_existing = int(len(unique_upns) * percent_existing)
    rng = np.random.default_rng(seed=42)
    return set(rng.choice(unique_upns, size=n_existing, replace=False))


# ============================================================================
# 5. PROCESS A SINGLE INPUT FILE
# ============================================================================
def process_single_file(
    filepath: str,
    output_dir: str,
    per_run_output_dir: str,
    loft_existing_pct: float,
    sigma_val: float,
) -> None:
    filename = Path(filepath).stem
    logging.info(f"--> Processing: {filename}")

    try:
        # ----- Load -----
        raw_df = pd.read_csv(filepath)
        logging.info(f"   Loaded raw shape: {raw_df.shape}")

        # ----- Typology cleaning -----
        typo_df = filter_typology(raw_df)
        logging.info(
            f"   Typology filter: {raw_df.shape[0]:,} -> {typo_df.shape[0]:,} rows"
        )

        # ----- Per-run means artifact (BEFORE aggregation) -----
        # We extract from the typology-cleaned but unaggregated frame so each
        # row is still one (upn, run) pair.
        per_run_df = extract_per_run_means(typo_df, SCENARIO_LIST, id_col='upn')

        # ----- Aggregation: split aleatoric / epistemic -----
        agg_df = pool_epistemic_runs_decomposed(
            typo_df, SCENARIO_LIST, id_col='upn'
        )
        if agg_df.empty:
            logging.warning(f"   Empty aggregation for {filename}; skipping.")
            return

        # ----- Existing-measures disqualification (loft scenarios) -----
        disqualified_loft = disqualified_loft_upns(agg_df, loft_existing_pct)
        logging.info(
            f"   Disqualifying loft for {len(disqualified_loft):,} buildings "
            f"({loft_existing_pct * 100:.0f}%)"
        )

        # ----- Selection-score columns (aleatoric only) -----
        agg_df = add_aleatoric_sigma_columns(agg_df, SCENARIO_LIST, sigma=sigma_val)

        # ----- Per-scenario filtering and column extraction -----
        sigma_str = str(float(sigma_val))
        all_interventions = []

        for scn in SCENARIO_LIST:
            wdf = apply_physical_filters_for_optimisation(agg_df, scn)
            is_loft_scenario = 'loft' in scn.lower()

            # Build per-scenario output rows.
            base_cols = [c for c in COLS_KEEP if c in wdf.columns] + ['upn']
            sub_df = wdf[base_cols].copy()
            sub_df['intervention'] = scn

            # Selection score (aleatoric-penalised cost per net tonne).
            sub_df['capex_per_net_ton_aleatoric_sigma'] = wdf[
                f'{scn}_capex_per_net_ton_aleatoric_{sigma_str}sigma'
            ]

            # Decomposed uncertainty: capex per net tonne.
            sub_df['mean_capex_per_net_ton'] = wdf[f'{scn}_capex_per_net_ton_mean']
            sub_df['aleatoric_std_capex_per_net_ton'] = wdf[
                f'{scn}_capex_per_net_ton_aleatoric_std'
            ]
            sub_df['epistemic_std_capex_per_net_ton'] = wdf[
                f'{scn}_capex_per_net_ton_epistemic_std'
            ]

            # Decomposed uncertainty: total CO2 saved.
            sub_df['mean_total_co2_saved'] = wdf[f'{scn}_co2_mean']
            sub_df['aleatoric_std_total_co2_saved'] = wdf[f'{scn}_co2_aleatoric_std']
            sub_df['epistemic_std_total_co2_saved'] = wdf[f'{scn}_co2_epistemic_std']

            # Decomposed uncertainty: total capex.
            sub_df['mean_total_capex'] = wdf[f'{scn}_capex_mean']
            sub_df['aleatoric_std_total_capex'] = wdf[f'{scn}_capex_aleatoric_std']
            sub_df['epistemic_std_total_capex'] = wdf[f'{scn}_capex_epistemic_std']

            # Validity mask on the selection score.
            mask_valid = (
                (sub_df['capex_per_net_ton_aleatoric_sigma'] > 0)
                & (sub_df['capex_per_net_ton_aleatoric_sigma'] <= ABS_COST_CAP)
                & (sub_df['capex_per_net_ton_aleatoric_sigma'].notna())
            )
            if is_loft_scenario:
                mask_valid &= ~sub_df['upn'].isin(disqualified_loft)

            kept = sub_df[mask_valid].copy()
            logging.info(
                f"   {scn}: {len(sub_df):,} -> {len(kept):,} after filters"
            )
            if not kept.empty:
                all_interventions.append(kept)

        # ----- Combine, rank, save -----
        if all_interventions:
            combined_df = pd.concat(all_interventions, ignore_index=True)
            combined_df.sort_values(
                by=['upn', 'capex_per_net_ton_aleatoric_sigma'],
                ascending=[True, True],
                inplace=True,
            )
            combined_df['rank_within_upn'] = (
                combined_df.groupby('upn')['capex_per_net_ton_aleatoric_sigma']
                .rank(method='first', ascending=True)
                .astype(int)
            )

            os.makedirs(output_dir, exist_ok=True)
            interventions_path = os.path.join(
                output_dir,
                f"all_interventions_{filename}_loft_{loft_existing_pct}.csv",
            )
            combined_df.to_csv(interventions_path, index=False)
            logging.info(f"   Saved interventions: {interventions_path}")
        else:
            logging.warning(f"   No valid interventions for {filename}")

        # ----- Save per-run means artifact -----
        if not per_run_df.empty:
            os.makedirs(per_run_output_dir, exist_ok=True)
            per_run_path = os.path.join(
                per_run_output_dir,
                f"per_run_means_{filename}.parquet",
            )
            # Snappy compression + downcasted dtypes keep these files
            # compact at full scale.
            per_run_df.to_parquet(
                per_run_path, index=False, compression='snappy',
            )
            logging.info(f"   Saved per-run means: {per_run_path}")

    except Exception as e:
        logging.error(f"Failed on {filename}: {e}", exc_info=True)
        log_error_to_file(filename, str(e))

    finally:
        # Best-effort cleanup; not all names exist on every code path.
        for name in ('raw_df', 'typo_df', 'agg_df', 'per_run_df', 'all_interventions'):
            if name in locals():
                del locals()[name]
        gc.collect()


# ============================================================================
# 6. PIPELINE ENTRYPOINT
# ============================================================================
def run_pipeline() -> None:
    setup_logging()

    # Reset error log.
    if os.path.exists(ERROR_LOG_FILE):
        os.remove(ERROR_LOG_FILE)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(PER_RUN_OUTPUT_DIR, exist_ok=True)

    files = glob.glob(f"{LOG_DIR}/*.csv")
    logging.info(f"Found {len(files)} files in {LOG_DIR}")

    for loft_pct in loft_perc_list:
        logging.info(f"=== Loft existing fraction: {loft_pct} ===")
        for f in files:
            process_single_file(
                filepath=f,
                output_dir=OUTPUT_BASE_DIR,
                per_run_output_dir=PER_RUN_OUTPUT_DIR,
                loft_existing_pct=loft_pct,
                sigma_val=RISK_PENALTY_SIGMA,
            )

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()