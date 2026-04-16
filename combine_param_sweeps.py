#!/usr/bin/env python
"""
combine_sweep_results.py - Memory-efficient combination of parquet files WITH visualization.

Updated to use Law of Total Variance for uncertainty aggregation and
confidence interval threshold crossings (μ-σ / μ / μ+σ).

Now includes building census logging at raw and filtered stages for paper reporting.

Usage:
    # Full pipeline (process + plot)
    python combine_sweep_results.py
    
    # Plots only from existing results
    python combine_sweep_results.py --plots-only --output-dir wall_param_sweep/results/combined_12:34:56

    python combine_param_sweeps.py --results-dir /home/gb669/rds/hpc-work/energy_map/RetrofitModel/wall_param_sweep_v10_n50_p10_v3/results  --output-dir /home/gb669/rds/hpc-work/energy_map/RetrofitModel/wall_param_sweep_v10_n50_p10_v3/output
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Optional, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '.')

from param_sweep import (
    compute_building_total_variance,
    compute_threshold_crossings,
    create_building_category,
    COST_MEAN_COL,
    COST_STD_COL,
    ATTRACTIVE_THRESHOLDS_5YR,
)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

KEEP_COLS = [
    'upn', 'postcode', 'region',
    'premise_type_filled', 'avg_gas_percentile',
    'inferred_wall_type', 'inferred_insulation_type',
    'building_category',
    'internal_factor', 'external_factor', 'sweep_type',
]

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

SOLID_WALL_INTERNAL = 'solid_wall_internal'
SOLID_WALL_EXTERNAL = 'solid_wall_external'
CAVITY_WALL = 'cavity_wall'
SWEEP_INTERNAL = 'internal'
SWEEP_EXTERNAL = 'external'

PALETTE = {
    SOLID_WALL_INTERNAL: '#1f77b4',
    SOLID_WALL_EXTERNAL: '#ff7f0e',
    CAVITY_WALL: '#2ca02c',
}

THRESHOLDS = [800, 1600, 2400]

CATEGORY_MAP = {
    'solid_wall_internal_wall_insulation': SOLID_WALL_INTERNAL,
    'solid_wall_external_wall_insulation': SOLID_WALL_EXTERNAL,
    'solid_wall_internal': SOLID_WALL_INTERNAL,
    'solid_wall_external': SOLID_WALL_EXTERNAL,
}

GAS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MIN_SAMPLE_SIZE = 10

print(f'Starting with min_sample={MIN_SAMPLE_SIZE}')


def normalize_building_category(df: pd.DataFrame) -> pd.DataFrame:
    """Apply category mapping to standardize building_category values."""
    if 'building_category' in df.columns:
        df = df.copy()
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
    return df


# ==========================================
# BUILDING CENSUS FUNCTIONS
# ==========================================

def _add_total_row(df: pd.DataFrame, count_col: str = 'n_distinct_upns') -> pd.DataFrame:
    """Append a total row summing the count column."""
    total = pd.DataFrame([{
        **{c: 'TOTAL' if df[c].dtype == object else '' for c in df.columns if c != count_col},
        count_col: df[count_col].sum(),
    }])
    return pd.concat([df, total], ignore_index=True)


def compute_building_census(
    df: pd.DataFrame,
    stage_label: str,
    output_dir: Path,
    attrition_ref: Optional[dict] = None,
) -> dict:
    """Compute and save building census tables (distinct UPN counts) for all typology groupings.
    
    Args:
        df: DataFrame with at least 'upn' and 'building_category' columns.
        stage_label: 'raw' or 'filtered' — used in filenames.
        output_dir: Directory to save CSVs.
        attrition_ref: If provided (dict of group_key -> n_distinct_upns from raw stage),
                       an 'attrition' and 'attrition_pct' column is added to filtered tables.
    
    Returns:
        dict mapping group_key tuples to n_distinct_upns (for use as attrition_ref in
        a subsequent filtered call).
    """
    census_dir = output_dir / 'census'
    census_dir.mkdir(exist_ok=True)
    
    upn_col = 'upn'
    ref_lookup = {}  # returned for attrition tracking
    
    # ---- 1. By building_category ----
    grp1 = (
        df.groupby('building_category')[upn_col]
        .nunique()
        .reset_index()
        .rename(columns={upn_col: 'n_distinct_upns'})
        .sort_values('building_category')
    )
    for _, row in grp1.iterrows():
        ref_lookup[('category', row['building_category'])] = row['n_distinct_upns']
    
    if attrition_ref:
        grp1['n_raw_upns'] = grp1['building_category'].map(
            lambda c: attrition_ref.get(('category', c), np.nan)
        )
        grp1['n_dropped'] = grp1['n_raw_upns'] - grp1['n_distinct_upns']
        grp1['pct_retained'] = (grp1['n_distinct_upns'] / grp1['n_raw_upns'] * 100).round(1)
    
    grp1 = _add_total_row(grp1)
    path1 = census_dir / f'census_{stage_label}_by_category.csv'
    grp1.to_csv(path1, index=False)
    print(f"  Census saved: {path1} ({len(grp1) - 1} categories + total)")
    
    # ---- 2. By building_category × avg_gas_percentile ----
    if 'avg_gas_percentile' in df.columns:
        grp2 = (
            df.groupby(['building_category', 'avg_gas_percentile'])[upn_col]
            .nunique()
            .reset_index()
            .rename(columns={upn_col: 'n_distinct_upns'})
            .sort_values(['building_category', 'avg_gas_percentile'])
        )
        for _, row in grp2.iterrows():
            ref_lookup[('cat_gas', row['building_category'], row['avg_gas_percentile'])] = row['n_distinct_upns']
        
        if attrition_ref:
            grp2['n_raw_upns'] = grp2.apply(
                lambda r: attrition_ref.get(('cat_gas', r['building_category'], r['avg_gas_percentile']), np.nan),
                axis=1,
            )
            grp2['n_dropped'] = grp2['n_raw_upns'] - grp2['n_distinct_upns']
            grp2['pct_retained'] = (grp2['n_distinct_upns'] / grp2['n_raw_upns'] * 100).round(1)
        
        grp2 = _add_total_row(grp2)
        path2 = census_dir / f'census_{stage_label}_by_category_x_gas.csv'
        grp2.to_csv(path2, index=False)
        print(f"  Census saved: {path2} ({len(grp2) - 1} groups + total)")
    
    # ---- 3. By building_category × premise_type_filled ----
    if 'premise_type_filled' in df.columns:
        grp3 = (
            df.groupby(['building_category', 'premise_type_filled'])[upn_col]
            .nunique()
            .reset_index()
            .rename(columns={upn_col: 'n_distinct_upns'})
            .sort_values(['building_category', 'premise_type_filled'])
        )
        for _, row in grp3.iterrows():
            ref_lookup[('cat_prem', row['building_category'], row['premise_type_filled'])] = row['n_distinct_upns']
        
        if attrition_ref:
            grp3['n_raw_upns'] = grp3.apply(
                lambda r: attrition_ref.get(('cat_prem', r['building_category'], r['premise_type_filled']), np.nan),
                axis=1,
            )
            grp3['n_dropped'] = grp3['n_raw_upns'] - grp3['n_distinct_upns']
            grp3['pct_retained'] = (grp3['n_distinct_upns'] / grp3['n_raw_upns'] * 100).round(1)
        
        grp3 = _add_total_row(grp3)
        path3 = census_dir / f'census_{stage_label}_by_category_x_premise.csv'
        grp3.to_csv(path3, index=False)
        print(f"  Census saved: {path3} ({len(grp3) - 1} groups + total)")
    
    # ---- 4. Summary to console ----
    total_upns = df[upn_col].nunique()
    print(f"\n  [{stage_label.upper()} CENSUS] Total distinct UPNs: {total_upns:,}")
    for _, row in grp1[grp1['building_category'] != 'TOTAL'].iterrows():
        pct = row['n_distinct_upns'] / total_upns * 100
        print(f"    {row['building_category']}: {row['n_distinct_upns']:,} ({pct:.1f}%)")
    
    return ref_lookup


def compute_building_census_chunked(
    csv_path: Path,
    stage_label: str,
    output_dir: Path,
    attrition_ref: Optional[dict] = None,
    chunksize: int = 200_000,
) -> dict:
    """Memory-efficient census from a large CSV, reading in chunks.
    
    Collects sets of unique UPNs per group across chunks, then counts.
    """
    census_dir = output_dir / 'census'
    census_dir.mkdir(exist_ok=True)
    
    # Accumulators: group_key -> set of UPNs
    by_cat = {}
    by_cat_gas = {}
    by_cat_prem = {}
    
    cols_needed = ['upn', 'building_category']
    # Peek at columns to know what's available
    sample = pd.read_csv(csv_path, nrows=1)
    has_gas = 'avg_gas_percentile' in sample.columns
    has_prem = 'premise_type_filled' in sample.columns
    if has_gas:
        cols_needed.append('avg_gas_percentile')
    if has_prem:
        cols_needed.append('premise_type_filled')
    
    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunksize, usecols=cols_needed),
        desc=f"Census ({stage_label})",
    ):
        for _, row in chunk.iterrows():
            upn = row['upn']
            cat = row['building_category']
            
            if pd.isna(cat):
                continue
            
            # By category
            by_cat.setdefault(cat, set()).add(upn)
            
            # By category × gas
            if has_gas and pd.notna(row.get('avg_gas_percentile')):
                gas = row['avg_gas_percentile']
                by_cat_gas.setdefault((cat, gas), set()).add(upn)
            
            # By category × premise
            if has_prem and pd.notna(row.get('premise_type_filled')):
                prem = row['premise_type_filled']
                by_cat_prem.setdefault((cat, prem), set()).add(upn)
    
    # Build DataFrames and save
    ref_lookup = {}
    
    # ---- 1. By category ----
    rows1 = []
    for cat, upns in sorted(by_cat.items()):
        n = len(upns)
        rows1.append({'building_category': cat, 'n_distinct_upns': n})
        ref_lookup[('category', cat)] = n
    grp1 = pd.DataFrame(rows1)
    
    if attrition_ref:
        grp1['n_raw_upns'] = grp1['building_category'].map(
            lambda c: attrition_ref.get(('category', c), np.nan)
        )
        grp1['n_dropped'] = grp1['n_raw_upns'] - grp1['n_distinct_upns']
        grp1['pct_retained'] = (grp1['n_distinct_upns'] / grp1['n_raw_upns'] * 100).round(1)
    
    grp1 = _add_total_row(grp1)
    path1 = census_dir / f'census_{stage_label}_by_category.csv'
    grp1.to_csv(path1, index=False)
    print(f"  Census saved: {path1} ({len(grp1) - 1} categories + total)")
    
    # ---- 2. By category × gas ----
    if by_cat_gas:
        rows2 = []
        for (cat, gas), upns in sorted(by_cat_gas.items()):
            n = len(upns)
            rows2.append({'building_category': cat, 'avg_gas_percentile': gas, 'n_distinct_upns': n})
            ref_lookup[('cat_gas', cat, gas)] = n
        grp2 = pd.DataFrame(rows2)
        
        if attrition_ref:
            grp2['n_raw_upns'] = grp2.apply(
                lambda r: attrition_ref.get(('cat_gas', r['building_category'], r['avg_gas_percentile']), np.nan),
                axis=1,
            )
            grp2['n_dropped'] = grp2['n_raw_upns'] - grp2['n_distinct_upns']
            grp2['pct_retained'] = (grp2['n_distinct_upns'] / grp2['n_raw_upns'] * 100).round(1)
        
        grp2 = _add_total_row(grp2)
        path2 = census_dir / f'census_{stage_label}_by_category_x_gas.csv'
        grp2.to_csv(path2, index=False)
        print(f"  Census saved: {path2} ({len(grp2) - 1} groups + total)")
    
    # ---- 3. By category × premise ----
    if by_cat_prem:
        rows3 = []
        for (cat, prem), upns in sorted(by_cat_prem.items()):
            n = len(upns)
            rows3.append({'building_category': cat, 'premise_type_filled': prem, 'n_distinct_upns': n})
            ref_lookup[('cat_prem', cat, prem)] = n
        grp3 = pd.DataFrame(rows3)
        
        if attrition_ref:
            grp3['n_raw_upns'] = grp3.apply(
                lambda r: attrition_ref.get(('cat_prem', r['building_category'], r['premise_type_filled']), np.nan),
                axis=1,
            )
            grp3['n_dropped'] = grp3['n_raw_upns'] - grp3['n_distinct_upns']
            grp3['pct_retained'] = (grp3['n_distinct_upns'] / grp3['n_raw_upns'] * 100).round(1)
        
        grp3 = _add_total_row(grp3)
        path3 = census_dir / f'census_{stage_label}_by_category_x_premise.csv'
        grp3.to_csv(path3, index=False)
        print(f"  Census saved: {path3} ({len(grp3) - 1} groups + total)")
    
    # ---- Console summary ----
    total_upns = len(set().union(*by_cat.values())) if by_cat else 0
    print(f"\n  [{stage_label.upper()} CENSUS] Total distinct UPNs: {total_upns:,}")
    for cat in sorted(by_cat.keys()):
        n = len(by_cat[cat])
        pct = n / total_upns * 100 if total_upns > 0 else 0
        print(f"    {cat}: {n:,} ({pct:.1f}%)")
    
    return ref_lookup


# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================

def reduce_single_parquet(
    filepath: Path,
    mean_col: str = COST_MEAN_COL,
    std_col: str = COST_STD_COL,
) -> pd.DataFrame:
    """Load one parquet file and reduce to per-building combined uncertainty estimates.
    
    For each building x factor combination, applies Law of Total Variance across
    epistemic runs:
        μ = E_θ[μ_θ]
        σ² = E_θ[σ²_θ] + Var_θ[μ_θ]
    
    Returns one row per building x factor with combined_mean, combined_std,
    and the variance decomposition.
    """
    df = pd.read_parquet(filepath)
    
    if 'building_category' not in df.columns:
        df['building_category'] = df.apply(create_building_category, axis=1)
    
    df = normalize_building_category(df)
    
    # Check required columns
    if mean_col not in df.columns or std_col not in df.columns:
        print(f"  Warning: missing {mean_col} or {std_col} in {filepath}")
        return pd.DataFrame()
    
    group_cols = ['upn', 'internal_factor', 'external_factor', 'sweep_type']
    reduced_rows = []
    
    for keys, group in df.groupby(group_cols):
        upn, int_f, ext_f, sweep = keys
        
        means = group[mean_col].values
        stds = group[std_col].values
        
        # Law of Total Variance
        combined_mean = np.mean(means)
        aleatoric_var = np.mean(stds ** 2)
        epistemic_var = np.var(means, ddof=0) if len(means) > 1 else 0.0
        total_var = aleatoric_var + epistemic_var
        combined_std = np.sqrt(total_var)
        
        row = {
            'upn': upn,
            'internal_factor': int_f,
            'external_factor': ext_f,
            'sweep_type': sweep,
            'combined_mean': combined_mean,
            'combined_std': combined_std,
            'aleatoric_var': aleatoric_var,
            'epistemic_var': epistemic_var,
            'n_runs': len(means),
        }
        
        for col in KEEP_COLS:
            if col in group.columns and col not in row:
                row[col] = group[col].iloc[0]
        
        reduced_rows.append(row)
    
    return pd.DataFrame(reduced_rows)


def process_all_parquets(
    results_dir: Path,
    output_path: Path,
) -> int:
    """Process all parquet files one by one, append reduced data to output."""
    parquet_files = sorted(results_dir.glob('batch_*/sweep_*/detailed_results.parquet'))

    print(f"Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        return 0
    
    if output_path.exists():
        output_path.unlink()
    
    n_processed = 0
    total_buildings = 0
    
    for filepath in tqdm(parquet_files, desc="Processing"):
        try:
            batch_num = int(filepath.parts[-3].replace('batch_', ''))
            reduced = reduce_single_parquet(filepath)
            
            if reduced.empty:
                continue
            
            reduced['source_batch'] = batch_num
            
            if n_processed == 0:
                reduced.to_csv(output_path, index=False)
            else:
                reduced.to_csv(output_path, mode='a', header=False, index=False)
            
            total_buildings += len(reduced)
            n_processed += 1
            
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
            continue
    
    print(f"\nProcessed {n_processed} files")
    print(f"Total building-factor combinations: {total_buildings}")
    
    return n_processed


def _filter_valid_buildings(df):
    """Filter to valid buildings (positive mean, finite).
    
    Returns (valid_means, valid_stds, valid_df, n_dropped).
    """
    means = df['combined_mean']
    stds = df['combined_std']

    valid_mask = (
        np.isfinite(means) &
        np.isfinite(stds) &
        (means > 0) &
        (means.abs() < 1e6)
    )
    valid_means = means[valid_mask]
    valid_stds = stds[valid_mask]
    valid_df = df[valid_mask]
    n_dropped = len(means) - len(valid_means)

    return valid_means, valid_stds, valid_df, n_dropped


def _compute_group_stats(valid_means, valid_stds, valid_df):
    """Compute the standard set of stats for a group of buildings.
    
    Includes:
    - Distributional stats on combined_mean (median, percentiles)
    - median_optimistic: median of per-building (μ - σ)
    - median_pessimistic: median of per-building (μ + σ)
    - Threshold crossings at optimistic/central/pessimistic
    - Average uncertainty decomposition
    
    Returns a dict of stats, or None if no valid data.
    """
    if len(valid_means) == 0:
        return None

    # Per-building optimistic/pessimistic values
    optimistic_values = valid_means - valid_stds
    pessimistic_values = valid_means + valid_stds

    result = {
        'n_valid': len(valid_means),
        'n_distinct_upns': valid_df['upn'].nunique(),
        'mean': valid_means.mean(),
        'median': valid_means.median(),
        'median_optimistic': optimistic_values.median(),
        'median_pessimistic': pessimistic_values.median(),
        'std': valid_means.std(),
        'p10': valid_means.quantile(0.10),
        'p25': valid_means.quantile(0.25),
        'p75': valid_means.quantile(0.75),
        'p90': valid_means.quantile(0.90),
        # Average uncertainty components
        'avg_combined_std': valid_stds.mean(),
        'avg_aleatoric_std': np.sqrt(valid_df['aleatoric_var']).mean(),
        'avg_epistemic_std': np.sqrt(valid_df['epistemic_var']).mean(),
        'avg_n_runs': valid_df['n_runs'].mean(),
    }

    # Threshold crossings at three confidence levels
    for threshold in ATTRACTIVE_THRESHOLDS_5YR:
        crossings = compute_threshold_crossings(
            valid_means.values, valid_stds.values, threshold
        )
        result[f'optimistic_pct_below_{threshold}'] = crossings['optimistic_pct']
        result[f'central_pct_below_{threshold}'] = crossings['central_pct']
        result[f'pessimistic_pct_below_{threshold}'] = crossings['pessimistic_pct']

    return result


def compute_final_statistics(
    reduced_csv_path: Path,
    output_dir: Path,
    raw_census_ref: Optional[dict] = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Compute final aggregated statistics from reduced data.
    
    Produces four output CSVs:
    - sweep_by_building_category.csv: per factor x building_category
    - category_x_gas_decile.csv: per factor x building_category x avg_gas_percentile
    - category_x_premise_type.csv: per factor x building_category x premise_type_filled
    - premise_x_gas_decile.csv: per factor x building_category x premise_type_filled x avg_gas_percentile
    
    Also runs a filtered building census (after validity filtering) with attrition
    tracking relative to the raw census.
    
    For each group:
    - Cross-building summary stats (median, percentiles)
    - n_distinct_upns: number of unique buildings in the group
    - median_optimistic / median_pessimistic (median of per-building μ±σ)
    - Threshold crossings at three confidence levels (μ-σ / μ / μ+σ)
    - Average uncertainty decomposition
    """
    print("\nComputing final statistics...")
    
    # Gather unique sweep parameter combinations
    sweep_params = set()
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize,
                             usecols=['internal_factor', 'external_factor', 'sweep_type']):
        for _, row in chunk.drop_duplicates().iterrows():
            sweep_params.add((row['internal_factor'], row['external_factor'], row['sweep_type']))
    
    print(f"Found {len(sweep_params)} sweep parameter combinations")
    
    all_results = []
    all_gas_results = []
    all_premise_results = []
    all_premise_gas_results = []
    
    # Collect all valid buildings for the filtered census
    filtered_buildings_rows = []
    
    for int_f, ext_f, sweep in tqdm(sweep_params, desc="Aggregating"):
        buildings = []
        
        for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
            mask = (
                (chunk['internal_factor'] == int_f) &
                (chunk['external_factor'] == ext_f) &
                (chunk['sweep_type'] == sweep)
            )
            if mask.any():
                buildings.append(chunk.loc[mask])
        
        if not buildings:
            continue
        
        df = pd.concat(buildings, ignore_index=True)
        
        sweep_keys = {
            'internal_factor': int_f,
            'external_factor': ext_f,
            'sweep_type': sweep,
        }
        
        for category in df['building_category'].dropna().unique():
            cat_df = df[df['building_category'] == category]
            
            valid_means, valid_stds, valid_df, n_dropped = _filter_valid_buildings(cat_df)
            
            # Accumulate valid buildings for filtered census
            # (use only first sweep param combo to avoid double-counting across sweeps)
            if len(valid_df) > 0:
                filtered_buildings_rows.append(valid_df[
                    [c for c in ['upn', 'building_category', 'avg_gas_percentile', 'premise_type_filled']
                     if c in valid_df.columns]
                ])
            
            stats = _compute_group_stats(valid_means, valid_stds, valid_df)
            if stats is None:
                continue
            
            # --- Main results (sweep_by_building_category) ---
            result = {
                **sweep_keys,
                'building_category': category,
                'n': len(cat_df),
                'n_dropped_non_positive': n_dropped,
                **stats,
            }
            all_results.append(result)
            
            # --- Gas decile breakdown (category_x_gas_decile) ---
            if 'avg_gas_percentile' in valid_df.columns:
                for gas_val in valid_df['avg_gas_percentile'].dropna().unique():
                    gas_sub = valid_df[valid_df['avg_gas_percentile'] == gas_val]
                    g_means, g_stds, g_df, _ = _filter_valid_buildings(gas_sub)
                    g_stats = _compute_group_stats(g_means, g_stds, g_df)
                    if g_stats is None:
                        continue
                    all_gas_results.append({
                        **sweep_keys,
                        'building_category': category,
                        'gas_decile': gas_val,
                        'n': len(gas_sub),
                        **g_stats,
                    })
            
            # --- Premise type breakdown (category_x_premise_type) ---
            if 'premise_type_filled' in valid_df.columns:
                for ptype in valid_df['premise_type_filled'].dropna().unique():
                    prem_sub = valid_df[valid_df['premise_type_filled'] == ptype]
                    p_means, p_stds, p_df, _ = _filter_valid_buildings(prem_sub)
                    p_stats = _compute_group_stats(p_means, p_stds, p_df)
                    if p_stats is None:
                        continue
                    all_premise_results.append({
                        **sweep_keys,
                        'building_category': category,
                        'premise_type_filled': ptype,
                        'n': len(prem_sub),
                        **p_stats,
                    })
                    
                    # --- Premise x gas decile (premise_x_gas_decile) ---
                    if 'avg_gas_percentile' in p_df.columns:
                        for gas_val in p_df['avg_gas_percentile'].dropna().unique():
                            pg_sub = p_df[p_df['avg_gas_percentile'] == gas_val]
                            pg_means, pg_stds, pg_df, _ = _filter_valid_buildings(pg_sub)
                            pg_stats = _compute_group_stats(pg_means, pg_stds, pg_df)
                            if pg_stats is None:
                                continue
                            all_premise_gas_results.append({
                                **sweep_keys,
                                'building_category': category,
                                'premise_type_filled': ptype,
                                'gas_decile': gas_val,
                                'n': len(pg_sub),
                                **pg_stats,
                            })
    
    # --- Filtered building census ---
    if filtered_buildings_rows:
        print("\n--- Filtered Building Census (after validity filtering) ---")
        filtered_all = pd.concat(filtered_buildings_rows, ignore_index=True).drop_duplicates(subset=['upn'])
        compute_building_census(
            filtered_all,
            stage_label='filtered',
            output_dir=output_dir,
            attrition_ref=raw_census_ref,
        )
    
    # Save main results
    results_df = pd.DataFrame(all_results)
    output_path = output_dir / 'sweep_by_building_category.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(results_df)} rows)")
    
    # Save gas decile results
    gas_df = pd.DataFrame(all_gas_results)
    gas_path = output_dir / 'category_x_gas_decile.csv'
    gas_df.to_csv(gas_path, index=False)
    print(f"Saved: {gas_path} ({len(gas_df)} rows)")
    
    # Save premise type results
    premise_df = pd.DataFrame(all_premise_results)
    premise_path = output_dir / 'category_x_premise_type.csv'
    premise_df.to_csv(premise_path, index=False)
    print(f"Saved: {premise_path} ({len(premise_df)} rows)")
    
    # Save premise x gas decile results
    premise_gas_df = pd.DataFrame(all_premise_gas_results)
    premise_gas_path = output_dir / 'premise_x_gas_decile.csv'
    premise_gas_df.to_csv(premise_gas_path, index=False)
    print(f"Saved: {premise_gas_path} ({len(premise_gas_df)} rows)")
    
    return results_df


def compute_epistemic_stats_from_parquets(
    results_dir: Path,
    mean_col: str = COST_MEAN_COL,
) -> Optional[pd.DataFrame]:
    """
    Compute epistemic statistics by reading parquet files.
    Calculates the median cost per run to isolate run-level variance.
    """
    parquet_files = sorted(results_dir.glob('results/batch_*/sweep_*/detailed_results.parquet'))
    
    if not parquet_files:
        return None

    all_run_medians = []

    for filepath in tqdm(parquet_files, desc="Computing epistemic stats"):
        try:
            df = pd.read_parquet(filepath)

            if 'building_category' not in df.columns:
                df['building_category'] = df.apply(create_building_category, axis=1)

            df = normalize_building_category(df)

            if 'epistemic_run_id' not in df.columns:
                continue

            # Process Internal Wall Sweeps
            internal_mask = df['sweep_type'] == SWEEP_INTERNAL
            if internal_mask.any():
                internal_df = df[internal_mask]
                for (factor, run_id, cat), group in internal_df.groupby(
                    ['internal_factor', 'epistemic_run_id', 'building_category']
                ):
                    median_cost = group[mean_col].median()
                    all_run_medians.append({
                        'factor': factor,
                        'sweep_type': SWEEP_INTERNAL,
                        'building_category': cat,
                        'run_id': run_id,
                        'median_cost': median_cost,
                    })

            # Process External Wall Sweeps
            external_mask = df['sweep_type'] == SWEEP_EXTERNAL
            if external_mask.any():
                external_df = df[external_mask]
                for (factor, run_id, cat), group in external_df.groupby(
                    ['external_factor', 'epistemic_run_id', 'building_category']
                ):
                    median_cost = group[mean_col].median()
                    all_run_medians.append({
                        'factor': factor,
                        'sweep_type': SWEEP_EXTERNAL,
                        'building_category': cat,
                        'run_id': run_id,
                        'median_cost': median_cost,
                    })

        except Exception as e:
            print(f"\nError processing {filepath} for epistemic stats: {e}")
            continue

    if not all_run_medians:
        return None

    return pd.DataFrame(all_run_medians)


def load_existing_statistics(output_dir: Path) -> pd.DataFrame:
    """Load existing statistics CSV if it exists."""
    stats_path = output_dir / 'sweep_by_building_category.csv'
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    print(f"Loading existing statistics from: {stats_path}")
    return pd.read_csv(stats_path)


def print_summary(results_df: pd.DataFrame):
    """Print summary tables with optimistic/central/pessimistic thresholds."""
    results_df = results_df.copy()
    results_df['internal_factor'] = pd.to_numeric(results_df['internal_factor'], errors='coerce')
    results_df['external_factor'] = pd.to_numeric(results_df['external_factor'], errors='coerce')

    print("\n" + "=" * 110)
    print("COMBINED RESULTS SUMMARY")
    print("Aggregation: Law of Total Variance, threshold crossings at μ-σ (optimistic) / μ (central) / μ+σ (pessimistic)")

    # Internal sweep
    print("\nINTERNAL FACTOR SWEEP (solid_wall_internal buildings):")
    print(f"{'Factor':<10} {'N':>6} {'UPNs':>8} {'Median':>10} {'AvgStd':>10} "
          f"{'Opt%<£2k':>10} {'Cen%<£2k':>10} {'Pes%<£2k':>10}")
    print("-" * 90)
    
    internal = results_df[
        (results_df['sweep_type'] == 'internal') &
        (results_df['building_category'] == 'solid_wall_internal')
    ].sort_values('internal_factor')
    
    for _, row in internal.iterrows():
        print(f"{row['internal_factor']:<10.2f} {row['n']:>6.0f} "
              f"{row.get('n_distinct_upns', np.nan):>8.0f} "
              f"{row['median']:>10.0f} "
              f"{row.get('avg_combined_std', np.nan):>10.0f} "
              f"{row.get('optimistic_pct_below_2000', np.nan):>9.1f}% "
              f"{row.get('central_pct_below_2000', np.nan):>9.1f}% "
              f"{row.get('pessimistic_pct_below_2000', np.nan):>9.1f}%")

    # External sweep
    print("\nEXTERNAL FACTOR SWEEP (solid_wall_external buildings):")
    print(f"{'Factor':<10} {'N':>6} {'UPNs':>8} {'Median':>10} {'AvgStd':>10} "
          f"{'Opt%<£2k':>10} {'Cen%<£2k':>10} {'Pes%<£2k':>10}")
    print("-" * 90)
    
    external = results_df[
        (results_df['sweep_type'] == 'external') &
        (results_df['building_category'] == 'solid_wall_external')
    ].sort_values('external_factor')
    
    for _, row in external.iterrows():
        print(f"{row['external_factor']:<10.2f} {row['n']:>6.0f} "
              f"{row.get('n_distinct_upns', np.nan):>8.0f} "
              f"{row['median']:>10.0f} "
              f"{row.get('avg_combined_std', np.nan):>10.0f} "
              f"{row.get('optimistic_pct_below_2000', np.nan):>9.1f}% "
              f"{row.get('central_pct_below_2000', np.nan):>9.1f}% "
              f"{row.get('pessimistic_pct_below_2000', np.nan):>9.1f}%")

    # Break-even analysis
    print("\n" + "=" * 110)
    print("BREAK-EVEN ANALYSIS (50% of buildings below threshold)")
    print("Reports factor at which each confidence level crosses 50%")
    print("Note: Values are 5-year £/tCO2. Divide by 6 for ~30-year equivalent.")
    print("=" * 110)
    
    for threshold in [1000, 1500, 2000, 3000]:
        opt_col = f'optimistic_pct_below_{threshold}'
        cen_col = f'central_pct_below_{threshold}'
        pes_col = f'pessimistic_pct_below_{threshold}'
        
        print(f"\n  Threshold: £{threshold}/tCO2 (5yr)")
        
        for label, data, factor_col in [
            ('Internal solid wall', internal, 'internal_factor'),
            ('External solid wall', external, 'external_factor'),
        ]:
            if len(data) == 0:
                continue
            
            factors = []
            for level_name, col in zip(['optimistic', 'central', 'pessimistic'], [opt_col, cen_col, pes_col]):
                if col in data.columns:
                    above_50 = data[data[col] >= 50]
                    if not above_50.empty:
                        min_f = above_50[factor_col].min()
                        factors.append(f"{level_name}={min_f:.0f}")
                    else:
                        max_val = data[col].max()
                        factors.append(f"{level_name}=n/a (max {max_val:.0f}%)")
            
            print(f"    {label}: {' | '.join(factors)}")

    approx_unique = results_df['n'].sum() / max(len(results_df['internal_factor'].unique()) * 2, 1)
    print(f"\nTotal unique buildings processed: {approx_unique:.0f} (approx)")


# ==========================================
# VISUALIZATION HELPER FUNCTIONS
# ==========================================

def filter_sweep(df: pd.DataFrame, sweep_type: str, building_category: str) -> pd.DataFrame:
    mask = (df['sweep_type'] == sweep_type) & (df['building_category'] == building_category)
    return df[mask].copy()


def get_factor_column(sweep_type: str) -> str:
    return f'{sweep_type}_factor'


def clean_premise_name(name: str) -> str:
    if pd.isna(name):
        return 'Unknown'
    return name.replace('_', ' ').title()


def get_gas_colors(n_bins: int) -> np.ndarray:
    return cm.YlOrRd(np.linspace(0.3, 1, n_bins))


def clean_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    df = df.copy()
    if 'building_category' in df.columns:
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
    if 'median' in df.columns:
        df = df[df['median'] < 100000]
    return df


def get_premise_types_to_plot(agg: pd.DataFrame, max_types: int = 6) -> List[str]:
    all_premises = sorted(agg['Premise Type'].unique())
    
    priority_types = [
        'Detached', 'Semi Detached', 'Terraces',
        'Terraced', 'Bungalow', 'Flat', 'Maisonette', 'Estates', 'Semi Type'
    ]
    
    ordered_premises = []
    for ptype in priority_types:
        matches = [p for p in all_premises if ptype.lower() in p.lower()]
        ordered_premises.extend(matches)
    
    remaining = [p for p in all_premises if p not in ordered_premises]
    ordered_premises.extend(remaining)
    
    seen = set()
    unique_premises = []
    for p in ordered_premises:
        if p not in seen:
            seen.add(p)
            unique_premises.append(p)
    
    return unique_premises[:max_types]


# ==========================================
# CLI ARGUMENT PARSING
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine sweep results and generate visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (process parquets + compute stats + generate plots)
  python combine_sweep_results.py

  # Plots only from existing results directory
  python combine_sweep_results.py --plots-only --output-dir wall_param_sweep/results/combined_12:34:56

  # Skip parquet processing, recompute stats and plots
  python combine_sweep_results.py --skip-processing --output-dir wall_param_sweep/results/combined_12:34:56

  # Just recompute stats (no plots)
  python combine_sweep_results.py --skip-processing --no-plots --output-dir wall_param_sweep/results/combined_12:34:56
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Existing output directory to use (required for --plots-only and --skip-processing)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='wall_param_sweep/results',
        help='Directory containing raw parquet results (default: wall_param_sweep/results)'
    )

    parser.add_argument(
        '--plots-only',
        action='store_true',
        help='Only generate plots from existing reduced CSV and statistics'
    )

    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip parquet processing, use existing reduced CSV'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--skip-epistemic',
        action='store_true',
        help='Skip epistemic sensitivity plot (requires reading raw parquets)'
    )

    parser.add_argument(
        '--min-sample',
        type=int,
        default=MIN_SAMPLE_SIZE,
        help=f'Minimum sample size for intersection plots (default: {MIN_SAMPLE_SIZE})'
    )

    return parser.parse_args()


# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        
        if not output_dir.exists():
            if args.plots_only or args.skip_processing:
                print(f"Error: Output directory does not exist: {output_dir}")
                sys.exit(1)
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.plots_only or args.skip_processing:
            print("Error: --output-dir is required when using --plots-only or --skip-processing")
            sys.exit(1)
        now = datetime.datetime.now()
        output_dir = results_dir / f'combined_{str(now.time()).replace(":", "-")}'
        output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}")

    reduced_csv = output_dir / 'reduced_building_estimates.csv'
    stats_csv = output_dir / 'sweep_by_building_category.csv'

    # Step 1: Process parquets (unless skipped)
    if args.plots_only:
        print("\n=== PLOTS ONLY MODE ===")
        results_df = load_existing_statistics(output_dir)
    elif args.skip_processing:
        print("\n=== SKIP PROCESSING MODE ===")
        if not reduced_csv.exists():
            print(f"Error: Reduced CSV not found: {reduced_csv}")
            sys.exit(1)
        
        # Raw census from existing reduced CSV
        print("\n--- Raw Building Census (all reduced buildings) ---")
        raw_census_ref = compute_building_census_chunked(
            reduced_csv, stage_label='raw', output_dir=output_dir
        )
        
        results_df = compute_final_statistics(reduced_csv, output_dir, raw_census_ref=raw_census_ref)
        print_summary(results_df)
    else:
        print("\n=== FULL PROCESSING MODE ===")
        n_processed = process_all_parquets(results_dir, reduced_csv)

        if n_processed == 0:
            print("No files processed!")
            return

        # Raw census from freshly created reduced CSV
        print("\n--- Raw Building Census (all reduced buildings) ---")
        raw_census_ref = compute_building_census_chunked(
            reduced_csv, stage_label='raw', output_dir=output_dir
        )

        results_df = compute_final_statistics(reduced_csv, output_dir, raw_census_ref=raw_census_ref)
        print_summary(results_df)

        # Convert to parquet for efficient downstream use
        print("\nConverting reduced data to parquet...")
        for chunk_idx, chunk in enumerate(pd.read_csv(reduced_csv, chunksize=500_000)):
            chunk.to_parquet(
                output_dir / f'reduced_chunk_{chunk_idx}.parquet',
                index=False
            )

    print("\nDone!")


if __name__ == '__main__':
    main()