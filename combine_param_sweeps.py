#!/usr/bin/env python
"""
combine_sweep_results.py - Memory-efficient combination of parquet files WITH visualization.

Strategy:
1. Process each parquet file one at a time
2. Reduce to per-building conservative estimates (much smaller)
3. Append to a single output file incrementally
4. Final aggregation on the reduced data
5. Generate visualizations from the aggregated results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
from typing import Optional, Tuple, List

sys.path.insert(0, '.')

from param_sweep import (
    compute_building_conservative_estimate,
    create_building_category,
    COST_PER_TCO2_METRIC,
    N_STD_CONSERVATIVE,
)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# Columns to keep for final aggregation
KEEP_COLS = [
    'upn', 'postcode', 'region',
    'premise_type_filled', 'avg_gas_percentile',
    'inferred_wall_type', 'inferred_insulation_type',
    'building_category',
    'internal_factor', 'external_factor', 'sweep_type',
]

# Visualization settings
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Wall type constants
SOLID_WALL_INTERNAL = 'solid_wall_internal'
SOLID_WALL_EXTERNAL = 'solid_wall_external'
CAVITY_WALL = 'cavity_wall'

# Sweep type constants
SWEEP_INTERNAL = 'internal'
SWEEP_EXTERNAL = 'external'

PALETTE = {
    SOLID_WALL_INTERNAL: '#1f77b4',  # Blue
    SOLID_WALL_EXTERNAL: '#ff7f0e',  # Orange
    CAVITY_WALL: '#2ca02c',          # Green
}

THRESHOLDS = [800, 1600, 2400]

# Category mapping - maps raw data values to standardized names
CATEGORY_MAP = {
    'solid_wall_internal_wall_insulation': SOLID_WALL_INTERNAL,
    'solid_wall_external_wall_insulation': SOLID_WALL_EXTERNAL,
    'solid_wall_internal': SOLID_WALL_INTERNAL,
    'solid_wall_external': SOLID_WALL_EXTERNAL,
}


def normalize_building_category(df: pd.DataFrame) -> pd.DataFrame:
    """Apply category mapping to standardize building_category values."""
    if 'building_category' in df.columns:
        df = df.copy()
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
    return df

# Gas bin configuration
GAS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Minimum sample size for intersection plots
MIN_SAMPLE_SIZE = 10


# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================

def reduce_single_parquet(
    filepath: Path,
    metric_col: str = COST_PER_TCO2_METRIC,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """
    Load one parquet file and reduce to per-building conservative estimates.
    
    Input: Full results with multiple epistemic runs per building per factor
    Output: One row per building per factor combination
    """
    df = pd.read_parquet(filepath)
    
    # Ensure building_category exists
    if 'building_category' not in df.columns:
        df['building_category'] = df.apply(create_building_category, axis=1)
    
    # Normalize building categories to standard names
    df = normalize_building_category(df)
    
    # Group by building + sweep parameters
    group_cols = ['upn', 'internal_factor', 'external_factor', 'sweep_type']
    
    reduced_rows = []
    
    for keys, group in df.groupby(group_cols):
        upn, int_f, ext_f, sweep = keys
        
        # Compute conservative estimate for this building
        values = group[metric_col]
        conservative = values.mean() + n_std * values.std() if len(values) > 1 else values.mean()
        
        # Get building characteristics (should be same across group)
        row = {
            'upn': upn,
            'internal_factor': int_f,
            'external_factor': ext_f,
            'sweep_type': sweep,
            'conservative_estimate': conservative,
        }
        
        # Add building characteristics
        for col in KEEP_COLS:
            if col in group.columns and col not in row:
                row[col] = group[col].iloc[0]
        
        reduced_rows.append(row)
    
    return pd.DataFrame(reduced_rows)


def process_all_parquets(
    results_dir: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
) -> int:
    """
    Process all parquet files one by one, append reduced data to output.
    
    Returns number of files processed.
    """
    parquet_files = sorted(results_dir.glob('batch_*/sweep_*/detailed_results.parquet'))
    parquet_files = parquet_files[0:15]
    print(f"Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        return 0
    
    # Remove existing output file
    if output_path.exists():
        output_path.unlink()
    
    n_processed = 0
    total_buildings = 0
    
    for filepath in tqdm(parquet_files, desc="Processing"):
        try:
            # Extract batch number from path
            batch_num = int(filepath.parts[-3].replace('batch_', ''))
            
            # Reduce this file
            reduced = reduce_single_parquet(filepath, n_std=n_std)
            reduced['source_batch'] = batch_num
            
            # Append to output (parquet doesn't support append, so use CSV for intermediate)
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


def compute_final_statistics(
    reduced_csv_path: Path,
    output_dir: Path,
    chunksize: int = 100_000,
):
    """
    Compute final aggregated statistics from reduced data.
    
    Uses chunked reading if needed for large files.
    """
    print("\nComputing final statistics...")
    
    # First pass: get unique sweep parameter combinations
    sweep_params = set()
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize, 
                             usecols=['internal_factor', 'external_factor', 'sweep_type']):
        for _, row in chunk.drop_duplicates().iterrows():
            sweep_params.add((row['internal_factor'], row['external_factor'], row['sweep_type']))
    
    print(f"Found {len(sweep_params)} sweep parameter combinations")
    
    # For each combination, compute statistics
    all_results = []
    
    for int_f, ext_f, sweep in tqdm(sweep_params, desc="Aggregating"):
        # Collect all buildings for this parameter combo
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
        
        # Aggregate by building category
        for category in df['building_category'].dropna().unique():
            cat_df = df[df['building_category'] == category]
            values = cat_df['conservative_estimate']
            valid = values[np.isfinite(values) & (values.abs() < 1e6)]
            
            if len(valid) == 0:
                continue
            
            all_results.append({
                'internal_factor': int_f,
                'external_factor': ext_f,
                'sweep_type': sweep,
                'building_category': category,
                'n': len(values),
                'n_valid': len(valid),
                'mean': valid.mean(),
                'median': valid.median(),
                'std': valid.std(),
                'p10': valid.quantile(0.10),
                'p25': valid.quantile(0.25),
                'p75': valid.quantile(0.75),
                'p90': valid.quantile(0.90),
                'pct_below_1000': (valid < 1000).mean() * 100,
                'pct_below_2000': (valid < 2000).mean() * 100,
                'pct_below_3000': (valid < 3000).mean() * 100,
                'pct_below_5000': (valid < 5000).mean() * 100,
            })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'sweep_by_building_category.csv', index=False)
    print(f"Saved: {output_dir / 'sweep_by_building_category.csv'}")
    
    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print summary tables."""
    
    print("\n" + "=" * 90)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 90)
    
    # Internal sweep - solid_wall_internal
    print("\nINTERNAL FACTOR SWEEP (solid_wall_internal buildings):")
    print(f"{'Factor':<10} {'N':>8} {'Median £/tCO2':>15} {'% < £2000':>12} {'% < £3000':>12}")
    print("-" * 60)
    
    internal = results_df[
        (results_df['sweep_type'] == 'internal') & 
        (results_df['building_category'] == 'solid_wall_internal')
    ].sort_values('internal_factor')
    
    for _, row in internal.iterrows():
        print(f"{row['internal_factor']:<10.2f} {row['n']:>8.0f} {row['median']:>15.0f} "
              f"{row['pct_below_2000']:>11.1f}% {row['pct_below_3000']:>11.1f}%")
    
    # External sweep - solid_wall_external
    print("\nEXTERNAL FACTOR SWEEP (solid_wall_external buildings):")
    print(f"{'Factor':<10} {'N':>8} {'Median £/tCO2':>15} {'% < £2000':>12} {'% < £3000':>12}")
    print("-" * 60)
    
    external = results_df[
        (results_df['sweep_type'] == 'external') & 
        (results_df['building_category'] == 'solid_wall_external')
    ].sort_values('external_factor')
    
    for _, row in external.iterrows():
        print(f"{row['external_factor']:<10.2f} {row['n']:>8.0f} {row['median']:>15.0f} "
              f"{row['pct_below_2000']:>11.1f}% {row['pct_below_3000']:>11.1f}%")
    
    # Total buildings
    print(f"\nTotal unique buildings processed: {results_df['n'].sum() / len(results_df['internal_factor'].unique()) / 2:.0f} (approx)")


# ==========================================
# VISUALIZATION HELPER FUNCTIONS
# ==========================================

def filter_sweep(df: pd.DataFrame, sweep_type: str, building_category: str) -> pd.DataFrame:
    """Filter dataframe by sweep type and building category."""
    mask = (df['sweep_type'] == sweep_type) & (df['building_category'] == building_category)
    return df[mask].copy()


def get_factor_column(sweep_type: str) -> str:
    """Return the appropriate factor column name for a sweep type."""
    return f'{sweep_type}_factor'


def clean_premise_name(name: str) -> str:
    """Convert premise_type_filled to display name."""
    if pd.isna(name):
        return 'Unknown'
    return name.replace('_', ' ').title()


def get_gas_colors(n_bins: int) -> np.ndarray:
    """
    Return colors for gas bins.
    Yellow (low gas usage) -> Red (high gas usage)
    """
    return cm.YlOrRd(np.linspace(0.3, 1, n_bins))


def clean_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Standardizes category names and removes outliers."""
    if df is None:
        return None

    df = df.copy()

    # Apply Mapping
    if 'building_category' in df.columns:
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)

    # Remove Infinity/Outliers for plotting
    if 'median' in df.columns:
        df = df[df['median'] < 100000]

    return df


def get_premise_types_to_plot(agg: pd.DataFrame, max_types: int = 6) -> List[str]:
    """
    Get list of premise types to plot, prioritizing common building types.
    """
    all_premises = sorted(agg['Premise Type'].unique())
    
    priority_types = [
        'Detached', 'Semi Detached', 'End Terrace', 'Mid Terrace',
        'Terraced', 'Bungalow', 'Flat', 'Maisonette'
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
# CORE PLOTTING FUNCTIONS
# ==========================================

def plot_cost_efficiency_curve(df: pd.DataFrame, output_path: Path, n_std: float = N_STD_CONSERVATIVE) -> None:
    """Plot 1: Cost efficiency curves for internal vs external insulation."""
    if df is None or df.empty:
        print("Skipping Plot 1: No data available")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    internal_data = filter_sweep(df, SWEEP_INTERNAL, SOLID_WALL_INTERNAL)
    internal_data = internal_data.sort_values('internal_factor')

    external_data = filter_sweep(df, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)
    external_data = external_data.sort_values('external_factor')

    min_x = 0.5

    if not internal_data.empty:
        ax.plot(
            np.array(internal_data['internal_factor']),
            np.array(internal_data['median']),
            marker='o', label='Solid Wall (Internal)',
            color=PALETTE[SOLID_WALL_INTERNAL], linewidth=2.5
        )
        min_x = internal_data['internal_factor'].min()

    if not external_data.empty:
        ax.plot(
            np.array(external_data['external_factor']),
            np.array(external_data['median']),
            marker='s', label='Solid Wall (External)',
            color=PALETTE[SOLID_WALL_EXTERNAL], linewidth=2.5
        )
        if internal_data.empty:
            min_x = external_data['external_factor'].min()

    # Thresholds
    for thr in THRESHOLDS:
        ax.axhline(thr, color='green', linestyle='--', alpha=0.5)
        ax.text(min_x, thr + 50, f'£{thr}/tCO2', color='gray', fontsize=9)

    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_title(f"Cost Efficiency by Wall Type\n(Conservative: mean + {n_std}×std)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '1_cost_efficiency_curve.png', dpi=300)
    plt.close()
    print("Saved Plot 1: Cost Efficiency Curve")


def plot_viability_percentage(df: pd.DataFrame, output_path: Path, n_std: float = N_STD_CONSERVATIVE) -> None:
    """Plot 2: Percentage of properties viable at £2000/tCO2 threshold."""
    if df is None or df.empty:
        print("Skipping Plot 2: No data available")
        return

    col = 'pct_below_2000'
    if col not in df.columns:
        print(f"Skipping Plot 2: Column '{col}' not found")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    int_data = filter_sweep(df, SWEEP_INTERNAL, SOLID_WALL_INTERNAL)
    int_data = int_data.sort_values('internal_factor')

    ext_data = filter_sweep(df, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)
    ext_data = ext_data.sort_values('external_factor')

    if not int_data.empty:
        ax.plot(
            np.array(int_data['internal_factor']),
            np.array(int_data[col]),
            marker='o', color=PALETTE[SOLID_WALL_INTERNAL],
            label='Solid Wall (Internal)', linewidth=2.5
        )

    if not ext_data.empty:
        ax.plot(
            np.array(ext_data['external_factor']),
            np.array(ext_data[col]),
            marker='s', color=PALETTE[SOLID_WALL_EXTERNAL],
            label='Solid Wall (External)', linewidth=2.5
        )

    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("% Viable (< £2000/tCO2)", fontsize=14)
    ax.set_title(f"Viability Ramp by Wall Type\n(Conservative: mean + {n_std}×std)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '2_viability_ramp.png', dpi=300)
    plt.close()
    print("Saved Plot 2: Viability Ramp")


def plot_viability_multi_threshold(df: pd.DataFrame, output_path: Path, n_std: float = N_STD_CONSERVATIVE) -> None:
    """Plot 3: Viability percentages at multiple thresholds."""
    if df is None or df.empty:
        print("Skipping Plot 3: No data available")
        return

    threshold_cols = ['pct_below_1000', 'pct_below_2000', 'pct_below_3000', 'pct_below_5000']
    available_cols = [c for c in threshold_cols if c in df.columns]
    
    if not available_cols:
        print("Skipping Plot 3: No threshold columns found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall', axes[0]),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall', axes[1]),
    ]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_cols)))

    for sweep_type, building_cat, title, ax in configs:
        data = filter_sweep(df, sweep_type, building_cat)
        factor_col = get_factor_column(sweep_type)
        data = data.sort_values(factor_col)

        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        for i, col in enumerate(available_cols):
            threshold = col.replace('pct_below_', '£')
            ax.plot(
                np.array(data[factor_col]),
                np.array(data[col]),
                marker='o', label=f'< {threshold}',
                color=colors[i], linewidth=2
            )

        ax.set_xlabel("Improvement Factor", fontsize=12)
        ax.set_ylabel("% of Buildings Viable", fontsize=12)
        ax.set_title(f"{title}", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(title="Cost Threshold")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Viability at Multiple Thresholds\n(Conservative: mean + {n_std}×std)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / '3_viability_multi_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 3: Multi-Threshold Viability")


def plot_gas_stratification(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = 100_000,
) -> None:
    """Plot 4a: Gas decile impact on external wall insulation efficiency."""
    
    # Read and filter for external wall data
    filtered_chunks = []
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
        chunk = normalize_building_category(chunk)
        mask = (chunk['sweep_type'] == SWEEP_EXTERNAL) & (chunk['building_category'] == SOLID_WALL_EXTERNAL)
        if mask.any():
            filtered_chunks.append(chunk[mask])
    
    if not filtered_chunks:
        print("Skipping Plot 4a: No external wall data")
        return
    
    subset = pd.concat(filtered_chunks, ignore_index=True)
    
    if subset.empty:
        print("Skipping Plot 4a: No external wall data")
        return

    # Create gas bins
    subset = subset.copy()
    subset['gas_decile'] = subset['avg_gas_percentile'].astype(int)
    
    # Aggregate by factor and gas decile
    agg = subset.groupby(['external_factor', 'gas_decile']).agg(
        median=('conservative_estimate', 'median'),
        count=('conservative_estimate', 'count')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    
    deciles = sorted(agg['gas_decile'].unique())
    colors = get_gas_colors(len(deciles))

    for i, decile in enumerate(deciles):
        group = agg[agg['gas_decile'] == decile].sort_values('external_factor')
        if not group.empty:
            ax.plot(
                np.array(group['external_factor']),
                np.array(group['median']),
                marker='o', label=f'Decile {decile}', color=colors[i], linewidth=2
            )

    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2", fontsize=14)
    ax.set_title(f"Gas Usage Impact on External Wall Insulation\n(Conservative: mean + {n_std}×std)", 
                 fontsize=14, fontweight='bold')
    ax.legend(title="Gas Decile", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add threshold lines
    for thr in THRESHOLDS:
        ax.axhline(thr, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / '4a_gas_decile_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 4a: Gas Decile Impact")


def plot_premise_stratification(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = 100_000,
) -> None:
    """Plot 4b: Premise type impact on internal wall insulation efficiency."""
    
    # Read and filter for internal wall data
    filtered_chunks = []
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
        chunk = normalize_building_category(chunk)
        mask = (chunk['sweep_type'] == SWEEP_INTERNAL) & (chunk['building_category'] == SOLID_WALL_INTERNAL)
        if mask.any():
            filtered_chunks.append(chunk[mask])
    
    if not filtered_chunks:
        print("Skipping Plot 4b: No internal wall data")
        return
    
    subset = pd.concat(filtered_chunks, ignore_index=True)
    
    if subset.empty:
        print("Skipping Plot 4b: No internal wall data")
        return

    # Clean premise types
    subset = subset.copy()
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)
    
    # Aggregate by factor and premise type
    agg = subset.groupby(['internal_factor', 'Premise Type']).agg(
        median=('conservative_estimate', 'median'),
        count=('conservative_estimate', 'count')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    
    premises = sorted(agg['Premise Type'].unique())

    for premise in premises:
        group = agg[agg['Premise Type'] == premise].sort_values('internal_factor')
        if not group.empty:
            ax.plot(
                np.array(group['internal_factor']),
                np.array(group['median']),
                marker='o', label=premise, linewidth=2
            )

    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2", fontsize=14)
    ax.set_title(f"Premise Type Impact on Internal Wall Insulation\n(Conservative: mean + {n_std}×std)", 
                 fontsize=14, fontweight='bold')
    ax.legend(title="Premise Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add threshold lines
    for thr in THRESHOLDS:
        ax.axhline(thr, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / '4b_premise_type_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 4b: Premise Type Impact")


def prepare_intersection_data(
    reduced_csv_path: Path,
    sweep_type: str,
    building_category: str,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    chunksize: int = 100_000,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Prepare data for intersection plots from reduced CSV.
    """
    factor_col = get_factor_column(sweep_type)
    
    # Read and filter in chunks
    filtered_chunks = []
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
        chunk = normalize_building_category(chunk)
        mask = (chunk['sweep_type'] == sweep_type) & (chunk['building_category'] == building_category)
        if mask.any():
            filtered_chunks.append(chunk[mask])
    
    if not filtered_chunks:
        return None, factor_col
    
    subset = pd.concat(filtered_chunks, ignore_index=True)
    
    if subset.empty:
        return None, factor_col

    # Create gas bins and clean premise types
    subset = subset.copy()
    subset['gas_bin'] = subset['avg_gas_percentile'].astype(int)
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

    # Aggregate
    agg = subset.groupby([factor_col, 'Premise Type', 'gas_bin']).agg(
        median_cost=('conservative_estimate', 'median'),
        mean_cost=('conservative_estimate', 'mean'),
        sample_count=('conservative_estimate', 'count')
    ).reset_index()

    # Filter small samples
    before_count = len(agg)
    agg = agg[agg['sample_count'] >= min_sample_size]
    after_count = len(agg)
    
    if before_count > after_count:
        print(f"  Filtered {before_count - after_count} bins with < {min_sample_size} buildings")

    return agg, factor_col


def plot_intersection_grid(
    agg: pd.DataFrame,
    factor_col: str,
    title: str,
    output_file: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """
    Create a grid of subplots showing premise type x gas decile interaction.
    """
    premises = get_premise_types_to_plot(agg)
    n_plots = len(premises)
    
    if n_plots == 0:
        print(f"No premise types to plot for {title}")
        return

    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True)
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    gas_labels = GAS_LABELS
    colors = get_gas_colors(len(gas_labels))
    
    if shared_ylim is None:
        y_min = 0
        y_max = agg['median_cost'].quantile(0.95)
        y_max = max(y_max, 3500) * 1.1
        shared_ylim = (y_min, y_max)
    
    for i, premise in enumerate(premises):
        ax = axes[i]
        p_data = agg[agg['Premise Type'] == premise]
        
        for j, gas_bin in enumerate(gas_labels):
            g_data = p_data[p_data['gas_bin'] == gas_bin].sort_values(factor_col)
            if not g_data.empty:
                ax.plot(
                    np.array(g_data[factor_col]),
                    np.array(g_data['median_cost']),
                    marker='o', markersize=6, label=str(gas_bin),
                    color=colors[j], linewidth=2
                )
        
        ax.set_title(f"{premise}", fontsize=12, fontweight='bold')
        ax.set_xlabel(factor_col.replace('_', ' ').title())
        
        if i % cols == 0:
            ax.set_ylabel("Median £/tCO2")
        
        ax.grid(True, alpha=0.3)
        
        for thr in [800, 1600, 2400]:
            if thr <= shared_ylim[1]:
                ax.axhline(thr, color='green', linestyle=':', alpha=0.5, linewidth=1)
    
    for ax in axes[:n_plots]:
        ax.set_ylim(shared_ylim)
    
    for k in range(len(premises), len(axes)):
        axes[k].axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=colors[j], marker='o', markersize=6, linewidth=2)
        for j in range(len(gas_labels))
    ]
    
    fig.legend(
        legend_handles, [str(g) for g in gas_labels],
        loc='upper center', bbox_to_anchor=(0.5, 1.12),
        ncol=len(gas_labels), title="Gas Consumption Decile", fontsize=10
    )
    
    plt.suptitle(f"{title}\n(Conservative: mean + {n_std}×std)", fontsize=14, fontweight='bold', y=1.15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_intersection_internal(
    reduced_csv_path: Path,
    output_path: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot 5a: Intersection plot for INTERNAL wall insulation."""
    agg, factor_col = prepare_intersection_data(
        reduced_csv_path, SWEEP_INTERNAL, SOLID_WALL_INTERNAL, min_sample_size
    )
    
    if agg is None or agg.empty:
        print("Skipping Plot 5a: No data for internal wall intersection")
        return

    plot_intersection_grid(
        agg=agg, factor_col=factor_col,
        title="Internal Wall Insulation: Premise Type × Gas Usage",
        output_file=output_path / '5a_intersection_internal.png',
        shared_ylim=shared_ylim, n_std=n_std
    )


def plot_intersection_external(
    reduced_csv_path: Path,
    output_path: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot 5b: Intersection plot for EXTERNAL wall insulation."""
    agg, factor_col = prepare_intersection_data(
        reduced_csv_path, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, min_sample_size
    )
    
    if agg is None or agg.empty:
        print("Skipping Plot 5b: No data for external wall intersection")
        return

    plot_intersection_grid(
        agg=agg, factor_col=factor_col,
        title="External Wall Insulation: Premise Type × Gas Usage",
        output_file=output_path / '5b_intersection_external.png',
        shared_ylim=shared_ylim, n_std=n_std
    )


def plot_intersection_heatmap(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = 100_000,
) -> None:
    """Plot 6: Combined heatmap showing cost efficiency across premise types and gas deciles."""
    
    # Load all reduced data
    all_data = []
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
        chunk = normalize_building_category(chunk)
        all_data.append(chunk)
    
    if not all_data:
        print("Skipping Plot 6: No data available")
        return
    
    df = pd.concat(all_data, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall Insulation', axes[0]),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall Insulation', axes[1]),
    ]

    for sweep_type, building_cat, title, ax in configs:
        subset = filter_sweep(df, sweep_type, building_cat)
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        subset = subset.copy()
        
        # Create gas bins and clean premise types
        subset['gas_decile'] = subset['avg_gas_percentile'].astype(int)
        subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

        # Use middle factor value for comparison
        factor_col = get_factor_column(sweep_type)
        mid_factor = subset[factor_col].median()
        
        # Filter to records near the median factor
        factor_range = subset[factor_col].max() - subset[factor_col].min()
        tolerance = factor_range * 0.1 if factor_range > 0 else 0.1
        mid_subset = subset[abs(subset[factor_col] - mid_factor) <= tolerance]

        if mid_subset.empty:
            mid_subset = subset.copy()

        # Pivot: each cell is median cost across buildings in that premise/gas combination
        pivot = mid_subset.pivot_table(
            values='conservative_estimate',
            index='Premise Type',
            columns='gas_decile',
            aggfunc='median'
        )

        if pivot.empty:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        # Plot heatmap
        sns.heatmap(
            pivot,
            ax=ax,
            cmap='RdYlGn_r',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': '£/tCO2'},
            linewidths=0.5
        )
        
        ax.set_title(f"{title}\n(Factor ≈ {mid_factor:.1f})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Gas Consumption Decile")
        ax.set_ylabel("Building Type")

    plt.suptitle(
        f"Cost Efficiency Heatmap: Premise Type vs Gas Usage\n(Conservative: mean + {n_std}×std)",
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path / '6_intersection_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 6: Intersection Heatmap")


def compute_epistemic_stats_from_parquets(
    results_dir: Path,
    metric_col: str = COST_PER_TCO2_METRIC,
) -> Optional[pd.DataFrame]:
    """
    Compute epistemic statistics by reading parquet files.
    
    For each (factor, epistemic_run), computes the median across buildings.
    Returns a DataFrame with columns: factor, sweep_type, building_category, run_id, median_cost
    """
    parquet_files = sorted(results_dir.glob('batch_*/sweep_*/detailed_results.parquet'))
    parquet_files = parquet_files[0:15]  # Match the main processing limit
    
    if not parquet_files:
        return None
    
    all_run_medians = []
    
    for filepath in tqdm(parquet_files, desc="Computing epistemic stats"):
        try:
            df = pd.read_parquet(filepath)
            
            # Ensure building_category exists
            if 'building_category' not in df.columns:
                df['building_category'] = df.apply(create_building_category, axis=1)
            
            # Normalize building categories
            df = normalize_building_category(df)
            
            # Check for epistemic_run_id
            if 'epistemic_run_id' not in df.columns:
                continue
            
            # For internal sweep
            internal_mask = df['sweep_type'] == SWEEP_INTERNAL
            if internal_mask.any():
                internal_df = df[internal_mask]
                for (factor, run_id, cat), group in internal_df.groupby(
                    ['internal_factor', 'epistemic_run_id', 'building_category']
                ):
                    median_cost = group[metric_col].median()
                    all_run_medians.append({
                        'factor': factor,
                        'sweep_type': SWEEP_INTERNAL,
                        'building_category': cat,
                        'run_id': run_id,
                        'median_cost': median_cost,
                    })
            
            # For external sweep
            external_mask = df['sweep_type'] == SWEEP_EXTERNAL
            if external_mask.any():
                external_df = df[external_mask]
                for (factor, run_id, cat), group in external_df.groupby(
                    ['external_factor', 'epistemic_run_id', 'building_category']
                ):
                    median_cost = group[metric_col].median()
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


def plot_epistemic_sensitivity(
    results_dir: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
) -> None:
    """
    Plot 7: Show how results vary across epistemic runs.
    Saves separate figures for Internal and External walls with shared Y-axis scale.
    """
    print("Computing epistemic statistics from parquet files...")
    epistemic_df = compute_epistemic_stats_from_parquets(results_dir)
    
    if epistemic_df is None or epistemic_df.empty:
        print("Skipping Plot 7: No epistemic data available")
        return

    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall', 'internal'),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall', 'external'),
    ]

    # Pre-calculate data to find global max Y
    processed_data = []
    global_max_y = 0
    thresholds = [800, 1600, 2400, 3200]

    for sweep_type, building_cat, title, file_suffix in configs:
        subset = epistemic_df[
            (epistemic_df['sweep_type'] == sweep_type) & 
            (epistemic_df['building_category'] == building_cat)
        ]
        
        if subset.empty:
            processed_data.append(None)
            continue

        # Summarize: mean ± std of median costs across epistemic runs
        summary = subset.groupby('factor')['median_cost'].agg(['mean', 'std']).reset_index()
        summary['std'] = summary['std'].fillna(0)
        
        # Track the highest value for shared axis
        current_max = (summary['mean'] + summary['std']).max()
        if current_max > global_max_y:
            global_max_y = current_max

        processed_data.append({
            'title': title,
            'summary': summary,
            'suffix': file_suffix
        })

    # Determine shared Y limit
    y_limit_top = max(global_max_y, max(thresholds)) * 1.1

    # Generate and save separate plots
    for data in processed_data:
        if data is None:
            continue

        summary = data['summary']
        
        fig, ax = plt.subplots(figsize=(10, 7))

        factors = summary['factor'].values
        means = summary['mean'].values
        stds = summary['std'].values
        
        # Plot fill for uncertainty band
        ax.fill_between(
            factors,
            means - stds,
            means + stds,
            alpha=0.3,
            label='±1 std (epistemic)'
        )
        # Plot mean line
        ax.plot(factors, means, 'o-', linewidth=2, label='Mean across runs')
        
        # Add threshold lines
        for thr in thresholds:
            ax.axhline(thr, color='green', linestyle='--', alpha=0.5)
            if thr < y_limit_top:
                ax.text(factors.min(), thr + 50, f'£{thr}', fontsize=9, color='gray')
        
        ax.set_xlabel("Improvement Factor", fontsize=14)
        ax.set_ylabel('Median £/tCO2 (across buildings)', fontsize=14)
        ax.set_title(f"Epistemic Uncertainty: {data['title']}\n(Variation across {int(summary['factor'].count())} epistemic runs)", 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=y_limit_top)

        plt.tight_layout()
        filename = f"7_epistemic_sensitivity_{data['suffix']}.png"
        plt.savefig(output_path / filename, dpi=300)
        plt.close()
        print(f"Saved Plot 7 ({data['title']}): {filename}")


def calculate_shared_ylim(
    reduced_csv_path: Path,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> Optional[Tuple[float, float]]:
    """Calculate shared y-axis limit across both internal and external data."""
    all_medians = []
    
    for sweep_type, building_cat in [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)
    ]:
        agg, _ = prepare_intersection_data(
            reduced_csv_path, sweep_type, building_cat, min_sample_size
        )
        if agg is not None and not agg.empty:
            all_medians.extend(agg['median_cost'].dropna().tolist())
    
    if not all_medians:
        return None
    
    y_max = np.percentile(all_medians, 95)
    y_max = max(y_max, 3500) * 1.1
    
    return (0, y_max)


def plot_distribution_comparison(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = 100_000
) -> None:
    """Plot 8: Distribution comparison (violin/box plots) for each wall type."""
    
    # Read all data
    all_data = []
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
        chunk = normalize_building_category(chunk)
        all_data.append(chunk)
    
    if not all_data:
        print("Skipping Plot 8: No data available")
        return
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Filter to valid data
    df = df[np.isfinite(df['conservative_estimate']) & (df['conservative_estimate'] < 10000)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall', axes[0]),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall', axes[1]),
    ]
    
    for sweep_type, building_cat, title, ax in configs:
        subset = filter_sweep(df, sweep_type, building_cat)
        factor_col = get_factor_column(sweep_type)
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue
        
        # Select a few factor values to compare
        factors = sorted(subset[factor_col].unique())
        if len(factors) > 5:
            indices = np.linspace(0, len(factors) - 1, 5, dtype=int)
            factors = [factors[i] for i in indices]
        
        plot_data = subset[subset[factor_col].isin(factors)]
        
        sns.boxplot(
            data=plot_data, x=factor_col, y='conservative_estimate',
            ax=ax, palette='viridis'
        )
        
        # Add threshold lines
        for thr in THRESHOLDS:
            ax.axhline(thr, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Improvement Factor", fontsize=12)
        ax.set_ylabel("£/tCO2 (Conservative)", fontsize=12)
        ax.set_title(f"{title}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Cost Distribution by Improvement Factor\n(Conservative: mean + {n_std}×std)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / '8_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 8: Distribution Comparison")


# ==========================================
# MAIN VISUALIZATION ORCHESTRATOR
# ==========================================

def generate_all_visualizations(
    results_df: pd.DataFrame,
    reduced_csv_path: Path,
    output_dir: Path,
    results_dir: Path,
    n_std: float = N_STD_CONSERVATIVE,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> None:
    """Generate all visualizations from the processed data."""
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    print(f"Output directory: {plots_dir}")
    print(f"Conservative estimate: mean + {n_std}×std")
    print(f"Minimum sample size: {min_sample_size}")
    
    # Clean the summary dataframe
    results_df = clean_dataframe(results_df)
    
    # Generate basic plots from summary statistics
    print("\n--- Basic Plots (1-3) ---")
    plot_cost_efficiency_curve(results_df, plots_dir, n_std)
    plot_viability_percentage(results_df, plots_dir, n_std)
    plot_viability_multi_threshold(results_df, plots_dir, n_std)
    
    # Generate stratification plots from reduced data
    print("\n--- Stratification Plots (4a-4b) ---")
    plot_gas_stratification(reduced_csv_path, plots_dir, n_std)
    plot_premise_stratification(reduced_csv_path, plots_dir, n_std)
    
    # Generate intersection plots from reduced data
    print("\n--- Intersection Plots (5a-5b) ---")
    shared_ylim = calculate_shared_ylim(reduced_csv_path, min_sample_size)
    print(f"Shared y-axis: {shared_ylim}")
    
    plot_intersection_internal(reduced_csv_path, plots_dir, shared_ylim, min_sample_size, n_std)
    plot_intersection_external(reduced_csv_path, plots_dir, shared_ylim, min_sample_size, n_std)
    
    # Heatmap
    print("\n--- Heatmap Plot (6) ---")
    plot_intersection_heatmap(reduced_csv_path, plots_dir, n_std)
    
    # Epistemic sensitivity (requires raw parquet files)
    print("\n--- Epistemic Sensitivity Plot (7) ---")
    plot_epistemic_sensitivity(results_dir, plots_dir, n_std)
    
    # Distribution comparison
    print("\n--- Distribution Plot (8) ---")
    plot_distribution_comparison(reduced_csv_path, plots_dir, n_std)
    
    print("\n" + "=" * 50)
    print(f"Visualization complete. Plots saved to: {plots_dir}")
    print("=" * 50)


# ==========================================
# MAIN
# ==========================================

def main():
    results_dir = Path('wall_param_sweep/results')
    output_dir = results_dir / 'combined'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    reduced_csv = output_dir / 'reduced_building_estimates.csv'
    
    # Step 1: Process all parquets one by one
    n_processed = process_all_parquets(results_dir, reduced_csv)
    
    if n_processed == 0:
        print("No files processed!")
        return
    
    # Step 2: Compute final statistics
    results_df = compute_final_statistics(reduced_csv, output_dir)
    
    # Step 3: Print summary
    print_summary(results_df)
    
    # Step 4: Generate visualizations
    generate_all_visualizations(
        results_df=results_df,
        reduced_csv_path=reduced_csv,
        output_dir=output_dir,
        results_dir=results_dir,
        n_std=N_STD_CONSERVATIVE,
        min_sample_size=MIN_SAMPLE_SIZE,
    )
    
    # Step 5: Convert reduced CSV to parquet for efficiency
    print("\nConverting reduced data to parquet...")
    for chunk_idx, chunk in enumerate(pd.read_csv(reduced_csv, chunksize=500_000)):
        chunk.to_parquet(
            output_dir / f'reduced_chunk_{chunk_idx}.parquet',
            index=False
        )
    print("Done!")


if __name__ == '__main__':
    main()