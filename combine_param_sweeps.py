#!/usr/bin/env python
"""
combine_sweep_results.py - Memory-efficient combination of parquet files WITH visualization.

Usage:
    # Full pipeline (process + plot)
    python combine_sweep_results.py
    
    # Plots only from existing results
    python combine_sweep_results.py --plots-only --output-dir wall_param_sweep/results/combined_12:34:56
    
    # Recompute statistics and plots (skip parquet processing)
    python combipct_below_2000ne_sweep_results.py --skip-processing --output-dir wall_param_sweep/results/combined_12:34:56
"""

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
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

print(f'Starting with n_std_conservative={N_STD_CONSERVATIVE}, min_sample={MIN_SAMPLE_SIZE}')


def normalize_building_category(df: pd.DataFrame) -> pd.DataFrame:
    """Apply category mapping to standardize building_category values."""
    if 'building_category' in df.columns:
        df = df.copy()
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
    return df


# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================

def reduce_single_parquet(
    filepath: Path,
    metric_col: str = COST_PER_TCO2_METRIC,
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """Load one parquet file and reduce to per-building conservative estimates.
    
    group over upn, get the mean ,std and the conservative value per building across epistemic runs
    
    """
    df = pd.read_parquet(filepath)
    
    if 'building_category' not in df.columns:
        df['building_category'] = df.apply(create_building_category, axis=1)
    
    df = normalize_building_category(df)
    
    group_cols = ['upn', 'internal_factor', 'external_factor', 'sweep_type']
    reduced_rows = []
    
    for keys, group in df.groupby(group_cols):
        upn, int_f, ext_f, sweep = keys
        values = group[metric_col]
        conservative = values.mean() + n_std * values.std() if len(values) > 1 else values.mean()
        
        row = {
            'upn': upn,
            'internal_factor': int_f,
            'external_factor': ext_f,
            'sweep_type': sweep,
            'conservative_estimate': conservative,
            'mean_val': values.mean(),
            'std_val': values.std(),
        }
        
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
            reduced = reduce_single_parquet(filepath, n_std=n_std)
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


def compute_final_statistics(
    reduced_csv_path: Path,
    output_dir: Path,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Compute final aggregated statistics from reduced data
    only uses conservative estimate."""
    print("\nComputing final statistics...")
    
    sweep_params = set()
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize,
                             usecols=['internal_factor', 'external_factor', 'sweep_type']):
        for _, row in chunk.drop_duplicates().iterrows():
            sweep_params.add((row['internal_factor'], row['external_factor'], row['sweep_type']))
    
    print(f"Found {len(sweep_params)} sweep parameter combinations")
    
    all_results = []
    
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
        
        for category in df['building_category'].dropna().unique():
            cat_df = df[df['building_category'] == category]
            values = cat_df['mean_val']
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
    output_path = output_dir / 'sweep_by_building_category.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return results_df



def compute_epistemic_stats_from_parquets(
    results_dir: Path,
    metric_col: str = COST_PER_TCO2_METRIC,
) -> Optional[pd.DataFrame]:
    """
    Compute epistemic statistics by reading parquet files.
    Calculates the median cost per run to isolate run-level variance.
    """
    parquet_files = sorted(results_dir.glob('results/batch_*/sweep_*/detailed_results.parquet'))
    parquet_files=parquet_files[0:5]
    
    # --- NOTE: Removed hard limit for production runs ---
    # parquet_files = parquet_files[0:50] 
    
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
                # Group by Factor AND Run ID to get one stat per epistemic run
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

            # Process External Wall Sweeps
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
    

def load_existing_statistics(output_dir: Path) -> pd.DataFrame:
    """Load existing statistics CSV if it exists."""
    stats_path = output_dir / 'sweep_by_building_category.csv'
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    print(f"Loading existing statistics from: {stats_path}")
    return pd.read_csv(stats_path)


def print_summary(results_df: pd.DataFrame):
    """Print summary tables."""
    results_df = results_df.copy()
    results_df['internal_factor'] = pd.to_numeric(results_df['internal_factor'], errors='coerce')
    results_df['external_factor'] = pd.to_numeric(results_df['external_factor'], errors='coerce')

    print("\n" + "=" * 90)
    print("COMBINED RESULTS SUMMARY")

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

    print(f"\nTotal unique buildings processed: {results_df['n'].sum() / len(results_df['internal_factor'].unique()) / 2:.0f} (approx)")


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
        '--n-std',
        type=float,
        default=N_STD_CONSERVATIVE,
        help=f'Number of standard deviations for conservative estimate (default: {N_STD_CONSERVATIVE})'
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
import os 

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

    print("\n=== FULL PROCESSING MODE ===")
    # Full pipeline
    n_processed = process_all_parquets(results_dir, reduced_csv, n_std=args.n_std)

    if n_processed == 0:
        print("No files processed!")
        return

    results_df = compute_final_statistics(reduced_csv, output_dir)
    print_summary(results_df)

    # Convert to parquet
    print("\nConverting reduced data to parquet...")
    for chunk_idx, chunk in enumerate(pd.read_csv(reduced_csv, chunksize=500_000)):
        chunk.to_parquet(
            output_dir / f'reduced_chunk_{chunk_idx}.parquet',
            index=False
        )

   

    print("\nDone!")


if __name__ == '__main__':
    main()