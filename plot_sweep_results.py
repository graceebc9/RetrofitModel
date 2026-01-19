#!/usr/bin/env python
"""
plot_sweep_results.py - Visualization of sweep analysis results.

This script generates all visualizations from aggregated sweep results.
Requires aggregate_sweep_results.py to have been run first.

Usage:
    # Generate all plots from existing results
    python plot_sweep_results.py --output-dir wall_param_sweep/results/combined_12:34:56
    
    # Include epistemic sensitivity plot (requires raw parquets)
    python plot_sweep_results.py --output-dir wall_param_sweep/results/combined_12:34:56 --include-epistemic
    
    # Custom minimum sample size for intersection plots
    python plot_sweep_results.py --output-dir wall_param_sweep/results/combined_12:34:56 --min-sample 20
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '.')

from param_sweep import (
    create_building_category,
    COST_PER_TCO2_METRIC,
    N_STD_CONSERVATIVE,
)

from sweep_config import (
    CATEGORY_MAP,
    SOLID_WALL_INTERNAL,
    SOLID_WALL_EXTERNAL,
    SWEEP_INTERNAL,
    SWEEP_EXTERNAL,
    PALETTE,
    THRESHOLDS,
    GAS_LABELS,
    MIN_SAMPLE_SIZE,
    DEFAULT_CHUNKSIZE,
)

# ==========================================
# PLOT STYLING
# ==========================================

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_building_category(df: pd.DataFrame) -> pd.DataFrame:
    """Apply category mapping to standardize building_category values."""
    if 'building_category' in df.columns:
        df = df.copy()
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
    return df


def filter_sweep(df: pd.DataFrame, sweep_type: str, building_category: str) -> pd.DataFrame:
    """Filter DataFrame by sweep type and building category."""
    mask = (df['sweep_type'] == sweep_type) & (df['building_category'] == building_category)
    return df[mask].copy()


def get_factor_column(sweep_type: str) -> str:
    """Get the factor column name for a sweep type."""
    return f'{sweep_type}_factor'


def clean_premise_name(name: str) -> str:
    """Clean premise type name for display."""
    if pd.isna(name):
        return 'Unknown'
    return name.replace('_', ' ').title()


def get_gas_colors(n_bins: int) -> np.ndarray:
    """Get color palette for gas deciles."""
    return cm.YlOrRd(np.linspace(0.3, 1, n_bins))


def clean_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Clean DataFrame for plotting."""
    if df is None:
        return None
    df = df.copy()
    if 'building_category' in df.columns:
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
    if 'median' in df.columns:
        df = df[df['median'] < 100000]
    return df


def get_premise_types_to_plot(agg: pd.DataFrame, max_types: int = 6) -> List[str]:
    """Get ordered list of premise types to plot."""
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


# def prepare_intersection_data(
#     reduced_csv_path: Path,
#     sweep_type: str,
#     building_category: str,
#     min_sample_size: int = MIN_SAMPLE_SIZE,
#     chunksize: int = DEFAULT_CHUNKSIZE,
# ) -> Tuple[Optional[pd.DataFrame], str]:
#     """Prepare data for intersection plots from reduced CSV."""
#     factor_col = get_factor_column(sweep_type)

#     filtered_chunks = []
#     for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
#         chunk = normalize_building_category(chunk)
#         mask = (chunk['sweep_type'] == sweep_type) & (chunk['building_category'] == building_category)
#         if mask.any():
#             filtered_chunks.append(chunk[mask])

#     if not filtered_chunks:
#         return None, factor_col

#     subset = pd.concat(filtered_chunks, ignore_index=True)

#     if subset.empty:
#         return None, factor_col

#     subset = subset.copy()
#     subset['gas_bin'] = subset['avg_gas_percentile'].astype(int)
#     subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)
#     # Build aggregation dict
#     agg_dict = {
#         'median_cost': ('conservative_estimate', 'median'),
#         'mean_cost': ('conservative_estimate', 'mean'),
#         'std_cost': ('conservative_estimate', 'std'),
#         'n_buildings': ('conservative_estimate', 'count'),
#     }
    
#     # Add raw mean/std aggregations for plots 9a/9b if columns exist
#     if 'mean_val' in subset.columns:
#         agg_dict['raw_mean_cost'] = ('mean_val', 'mean')
#     if 'std_val' in subset.columns:
#         agg_dict['raw_std_cost'] = ('std_val', 'mean')  # Mean of per-building stds

#     agg = subset.groupby([factor_col, 'Premise Type', 'gas_bin']).agg(**agg_dict).reset_index()
    
#     before_count = len(agg)
#     agg = agg[agg['n_buildings'] >= min_sample_size]
#     after_count = len(agg)

#     if before_count > after_count:
#         print(f"  Filtered {before_count - after_count} bins with < {min_sample_size} buildings")

#     return agg, factor_col

def prepare_intersection_data(
    reduced_csv_path: Path,
    sweep_type: str,
    building_category: str,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Prepare data for intersection plots from reduced CSV."""
    factor_col = get_factor_column(sweep_type)

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

    subset = subset.copy()
    
    # --- STATISTICAL PRE-PROCESSING ---
    # We need the variance (sigma^2), not just std, to average it correctly.
    if 'std_val' in subset.columns:
        subset['var_val'] = subset['std_val'] ** 2
    else:
        # Fallback if std_val missing, though it implies data issues
        subset['var_val'] = 0 
        
    subset['gas_bin'] = subset['avg_gas_percentile'].astype(int)
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

    # --- UPDATED AGGREGATION ---
    # We need specific components to build the robust metric later
    agg_dict = {
        'n_buildings': ('conservative_estimate', 'count'),
        # Component 1: Central Tendency of the Group
        'group_mean': ('mean_val', 'mean'), 
        # Component 2: Heterogeneity (How different are the buildings?)
        'between_building_var': ('mean_val', 'var'),
        # Component 3: Model Noise (Average internal uncertainty)
        'within_building_var': ('var_val', 'mean') 
    }
    
    # Perform Aggregation
    agg = subset.groupby([factor_col, 'Premise Type', 'gas_bin']).agg(**agg_dict).reset_index()
    
    # Filter for sample size
    before_count = len(agg)
    agg = agg[agg['n_buildings'] >= min_sample_size]
    after_count = len(agg)

    if before_count > after_count:
        print(f"  Filtered {before_count - after_count} bins with < {min_sample_size} buildings")

    # --- STATISTICAL POST-PROCESSING ---
    # Calculate the Total Group Variance (Eve's Law)
    # Total Var = Average(Internal Noise) + Variance(Building Means)
    agg['total_group_variance'] = agg['within_building_var'].fillna(0) + agg['between_building_var'].fillna(0)
    agg['total_group_std'] = np.sqrt(agg['total_group_variance'])
    
    # Create the Robust Metric (Conservative Estimate)
    agg['robust_conservative_estimate'] = agg['group_mean'] + agg['total_group_std']

    # Rename for compatibility with your existing plotting code if needed
    # Or keep distinct to compare "Old Mean" vs "Robust Estimate"
    agg['mean_cost'] = agg['robust_conservative_estimate'] 

    # group_mean total_group_std
    return agg, factor_col


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
            all_medians.extend(agg['robust_conservative_estimate'].dropna().tolist())

    if not all_medians:
        return None

    y_max = np.percentile(all_medians, 95)
    y_max = max(y_max, 3500) * 1.1

    return (0, y_max)


def calculate_shared_ylim_with_std(
    reduced_csv_path: Path,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> Optional[Tuple[float, float]]:
    """Calculate shared y-axis limit using mean + std bounds
    use cols group_mean total_group_std 
    
    ."""
    all_upper_bounds = []

    for sweep_type, building_cat in [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)
    ]:
        agg, _ = prepare_intersection_data(
            reduced_csv_path, sweep_type, building_cat, min_sample_size
        )
        if agg is not None and not agg.empty:
            upper = agg['group_mean'] + agg['total_group_std'].fillna(0)
            all_upper_bounds.extend(upper.dropna().tolist())

    if not all_upper_bounds:
        return None

    y_max = np.percentile(all_upper_bounds, 95)
    y_max = max(y_max, 3500) * 1.1

    return (0, y_max)


# ==========================================
# PLOT 1: Cost Efficiency Curve
# ==========================================

def plot_cost_efficiency_curve(
    df: pd.DataFrame,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot cost efficiency curves for internal vs external insulation."""
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

    for thr in THRESHOLDS:
        ax.axhline(thr, color='green', linestyle='--', alpha=0.5)
        ax.text(min_x, thr + 50, f'£{thr}/tCO2', color='gray', fontsize=9)

    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '1_cost_efficiency_curve.png', dpi=300)
    plt.close()

    # Save underlying data
    plot_data = pd.concat([
        internal_data[['internal_factor', 'building_category', 'n', 'n_valid', 'median', 'mean', 'std']].assign(wall_type='internal'),
        external_data[['external_factor', 'building_category', 'n', 'n_valid', 'median', 'mean', 'std']].assign(wall_type='external')
    ], ignore_index=True)
    plot_data.to_csv(output_path / '1_cost_efficiency_curve_data.csv', index=False)

    print("Saved Plot 1: Cost Efficiency Curve")


# ==========================================
# PLOT 2: Viability Percentage
# ==========================================

def plot_viability_percentage(
    df: pd.DataFrame,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot percentage of properties viable at £2000/tCO2 threshold."""
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
    
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '2_viability_ramp.png', dpi=300)
    plt.close()

    # Save underlying data
    threshold_cols = [c for c in ['pct_below_1000', 'pct_below_2000', 'pct_below_3000', 'pct_below_5000'] if c in df.columns]
    plot_data = pd.concat([
        int_data[['internal_factor', 'building_category', 'n', 'n_valid'] + threshold_cols].assign(wall_type='internal'),
        ext_data[['external_factor', 'building_category', 'n', 'n_valid'] + threshold_cols].assign(wall_type='external')
    ], ignore_index=True)
    plot_data.to_csv(output_path / '2_viability_ramp_data.csv', index=False)

    print("Saved Plot 2: Viability Ramp")


# ==========================================
# PLOT 3: Multi-Threshold Viability
# ==========================================

def plot_viability_multi_threshold(
    df: pd.DataFrame,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot viability percentages at multiple thresholds."""
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
    viability_data_list = []

    for sweep_type, building_cat, title, ax in configs:
        data = filter_sweep(df, sweep_type, building_cat)
        factor_col = get_factor_column(sweep_type)
        data = data.sort_values(factor_col)

        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        data_copy = data[[factor_col, 'building_category', 'n', 'n_valid'] + available_cols].copy()
        data_copy['wall_type'] = sweep_type
        viability_data_list.append(data_copy)

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

    if viability_data_list:
        viability_df = pd.concat(viability_data_list, ignore_index=True)
        viability_df.to_csv(output_path / '3_viability_multi_threshold_data.csv', index=False)

    print("Saved Plot 3: Multi-Threshold Viability")


# ==========================================
# PLOT 4a: Gas Decile Stratification
# ==========================================

def plot_gas_stratification(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """Plot gas decile impact on external wall insulation efficiency."""
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

    subset = subset.copy()
    subset['gas_decile'] = subset['avg_gas_percentile'].astype(int)

    agg = subset.groupby(['external_factor', 'gas_decile']).agg(
        median=('conservative_estimate', 'median'),
        mean=('conservative_estimate', 'mean'),
        std=('conservative_estimate', 'std'),
        n_buildings=('conservative_estimate', 'count')
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
 
    ax.legend(title="Gas Decile", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    for thr in THRESHOLDS:
        ax.axhline(thr, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / '4a_gas_decile_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

    agg.to_csv(output_path / '4a_gas_decile_impact_data.csv', index=False)

    print("Saved Plot 4a: Gas Decile Impact")


# ==========================================
# PLOT 4b: Premise Type Stratification
# ==========================================

def plot_premise_stratification(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """Plot premise type impact on internal wall insulation efficiency."""
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

    subset = subset.copy()
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

    agg = subset.groupby(['internal_factor', 'Premise Type']).agg(
        median=('conservative_estimate', 'median'),
        mean=('conservative_estimate', 'mean'),
        std=('conservative_estimate', 'std'),
        n_buildings=('conservative_estimate', 'count')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(20, 7))

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
 
    ax.legend(title="Premise Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    for thr in THRESHOLDS:
        ax.axhline(thr, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / '4b_premise_type_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

    agg.to_csv(output_path / '4b_premise_type_impact_data.csv', index=False)

    print("Saved Plot 4b: Premise Type Impact")


# ==========================================
# PLOT 5a/5b: Intersection Grid (Median)
# ==========================================

def plot_intersection_grid(
    agg: pd.DataFrame,
    factor_col: str,
    title: str,
    output_file: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Create a grid of subplots showing premise type x gas decile interaction."""
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
        y_max = agg['robust_conservative_estimate'].quantile(0.95)
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
                    np.array(g_data['robust_conservative_estimate']),
                    marker='o', markersize=6, label=str(gas_bin),
                    color=colors[j], linewidth=2
                )

        ax.set_title(f"{premise}", fontsize=12, fontweight='bold')
        ax.set_xlabel(factor_col.replace('_', ' ').title())

        if i % cols == 0:
            ax.set_ylabel("robust_conservative_estimate £/tCO2")

        ax.grid(True, alpha=0.3)

        for thr in [800, 1600, 2400]:
            if thr <= shared_ylim[1]:
                ax.axhline(thr, color='green', linestyle=':', alpha=0.5, linewidth=1)

    for ax in axes[:n_plots]:
        ax.set_ylim(shared_ylim)

    for k in range(len(premises), len(axes)):
        axes[k].axis('off')

    legend_handles = [
        Line2D([0], [0], color=colors[j], marker='o', markersize=6, linewidth=2)
        for j in range(len(gas_labels))
    ]

    fig.legend(
        legend_handles, [str(g) for g in gas_labels],
        loc='upper center', bbox_to_anchor=(0.5, 1.12),
        ncol=len(gas_labels), title="Gas Consumption Decile", fontsize=10
    )

    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    csv_file = output_file.with_suffix('.csv').name.replace('.csv', '_data.csv')
    agg.to_csv(output_file.parent / csv_file, index=False)

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


# ==========================================
# PLOT 6: Intersection Heatmap
# ==========================================

# def plot_intersection_heatmap(
#     reduced_csv_path: Path,
#     output_path: Path,
#     n_std: float = N_STD_CONSERVATIVE,
#     chunksize: int = DEFAULT_CHUNKSIZE,
# ) -> None:
#     """Plot combined heatmap showing cost efficiency across premise types and gas deciles."""
#     all_data = []
#     for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
#         chunk = normalize_building_category(chunk)
#         all_data.append(chunk)

#     if not all_data:
#         print("Skipping Plot 6: No data available")
#         return

#     df = pd.concat(all_data, ignore_index=True)

#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))

#     configs = [
#         (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall Insulation', axes[0]),
#         (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall Insulation', axes[1]),
#     ]

#     heatmap_data_list = []

#     for sweep_type, building_cat, title, ax in configs:
#         subset = filter_sweep(df, sweep_type, building_cat)

#         if subset.empty:
#             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
#             ax.set_title(title)
#             continue

#         subset = subset.copy()
#         subset['gas_decile'] = subset['avg_gas_percentile'].astype(int)
#         subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

#         factor_col = get_factor_column(sweep_type)
#         mid_factor = subset[factor_col].median()

#         factor_range = subset[factor_col].max() - subset[factor_col].min()
#         tolerance = factor_range * 0.1 if factor_range > 0 else 0.1
#         mid_subset = subset[abs(subset[factor_col] - mid_factor) <= tolerance]

#         if mid_subset.empty:
#             mid_subset = subset.copy()

#         pivot = mid_subset.pivot_table(
#             values='conservative_estimate',
#             index='Premise Type',
#             columns='gas_decile',
#             aggfunc='median'
#         )

#         agg_data = mid_subset.groupby(['Premise Type', 'gas_decile']).agg(
#             median_cost=('conservative_estimate', 'median'),
#             mean_cost=('conservative_estimate', 'mean'),
#             std_cost=('conservative_estimate', 'std'),
#             n_buildings=('conservative_estimate', 'count')
#         ).reset_index()
#         agg_data['sweep_type'] = sweep_type
#         agg_data['factor_used'] = mid_factor
#         heatmap_data_list.append(agg_data)

#         if pivot.empty:
#             ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
#             ax.set_title(title)
#             continue

#         sns.heatmap(
#             pivot,
#             ax=ax,
#             cmap='RdYlGn_r',
#             annot=True,
#             fmt='.0f',
#             cbar_kws={'label': '£/tCO2'},
#             linewidths=0.5
#         )


#         ax.set_xlabel("Gas Consumption Decile")
#         ax.set_ylabel("Building Type")

#     plt.suptitle(
#         f"Cost Efficiency Heatmap: Premise Type vs Gas Usage\n(Conservative: mean + {n_std}×std)",
#         fontsize=14,
#         fontweight='bold',
#         y=1.02
#     )
#     plt.tight_layout()
#     plt.savefig(output_path / '6_intersection_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     if heatmap_data_list:
#         heatmap_df = pd.concat(heatmap_data_list, ignore_index=True)
#         heatmap_df.to_csv(output_path / '6_intersection_heatmap_data.csv', index=False)

#     print("Saved Plot 6: Intersection Heatmap")


# ==========================================
# PLOT 7: Epistemic Sensitivity
# ==========================================

def compute_epistemic_stats_from_parquets(
    results_dir: Path,
    metric_col: str = COST_PER_TCO2_METRIC,
) -> Optional[pd.DataFrame]:
    """Compute epistemic statistics by reading parquet files."""
    parquet_files = sorted(results_dir.glob('batch_*/sweep_*/detailed_results.parquet'))

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
    """Plot how results vary across epistemic runs."""
    print("Computing epistemic statistics from parquet files...")
    epistemic_df = compute_epistemic_stats_from_parquets(results_dir)

    if epistemic_df is None or epistemic_df.empty:
        print("Skipping Plot 7: No epistemic data available")
        return

    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall', 'internal'),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall', 'external'),
    ]

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

        n_epistemic_runs = subset['run_id'].nunique()

        summary = subset.groupby('factor')['median_cost'].agg(['mean', 'std']).reset_index()
        summary['std'] = summary['std'].fillna(0)

        current_max = (summary['mean'] + summary['std']).max()
        if current_max > global_max_y:
            global_max_y = current_max

        processed_data.append({
            'title': title,
            'summary': summary,
            'suffix': file_suffix,
            'n_epistemic_runs': n_epistemic_runs,
        })

    y_limit_top = max(global_max_y, max(thresholds)) * 1.1

    for data in processed_data:
        if data is None:
            continue

        summary = data['summary']

        fig, ax = plt.subplots(figsize=(10, 7))

        factors = summary['factor'].values
        means = summary['mean'].values
        stds = summary['std'].values

        ax.fill_between(
            factors,
            means - stds,
            means + stds,
            alpha=0.3,
            label='±1 std (epistemic)'
        )
        ax.plot(factors, means, 'o-', linewidth=2, label='Mean across runs')

        for thr in thresholds:
            ax.axhline(thr, color='green', linestyle='--', alpha=0.5)
            if thr < y_limit_top:
                ax.text(factors.min(), thr + 50, f'£{thr}', fontsize=9, color='gray')

        ax.set_xlabel("Improvement Factor", fontsize=14)
        ax.set_ylabel('Median £/tCO2 (across buildings)', fontsize=14)
        ax.set_title(f"Epistemic Uncertainty: {data['title']}\n(Variation across {data['n_epistemic_runs']} epistemic runs)",
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=y_limit_top)

        plt.tight_layout()
        filename = f"7_epistemic_sensitivity_{data['suffix']}.png"
        plt.savefig(output_path / filename, dpi=300)
        plt.close()

        summary_df = summary.copy()
        summary_df['wall_type'] = data['suffix']
        summary_df['n_epistemic_runs'] = data['n_epistemic_runs']
        summary_df.to_csv(output_path / f"7_epistemic_sensitivity_{data['suffix']}_data.csv", index=False)

        print(f"Saved Plot 7 ({data['title']}): {filename}")


# ==========================================
# PLOT 8: Distribution Comparison
# ==========================================

def plot_distribution_comparison(
    reduced_csv_path: Path,
    output_path: Path,
    n_std: float = N_STD_CONSERVATIVE,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """Plot distribution comparison (box plots) for each wall type."""
    all_data = []
    for chunk in pd.read_csv(reduced_csv_path, chunksize=chunksize):
        chunk = normalize_building_category(chunk)
        all_data.append(chunk)

    if not all_data:
        print("Skipping Plot 8: No data available")
        return

    df = pd.concat(all_data, ignore_index=True)
    df = df[np.isfinite(df['conservative_estimate']) & (df['conservative_estimate'] < 10000)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall', axes[0]),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall', axes[1]),
    ]

    distribution_data_list = []

    for sweep_type, building_cat, title, ax in configs:
        subset = filter_sweep(df, sweep_type, building_cat)
        factor_col = get_factor_column(sweep_type)

        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        factors = sorted(subset[factor_col].unique())
        if len(factors) > 5:
            indices = np.linspace(0, len(factors) - 1, 5, dtype=int)
            factors = [factors[i] for i in indices]

        plot_data = subset[subset[factor_col].isin(factors)]

        agg_data = plot_data.groupby(factor_col).agg(
            median_cost=('conservative_estimate', 'median'),
            mean_cost=('conservative_estimate', 'mean'),
            std_cost=('conservative_estimate', 'std'),
            p25=('conservative_estimate', lambda x: x.quantile(0.25)),
            p75=('conservative_estimate', lambda x: x.quantile(0.75)),
            n_buildings=('conservative_estimate', 'count')
        ).reset_index()
        agg_data['wall_type'] = sweep_type
        agg_data['building_category'] = building_cat
        distribution_data_list.append(agg_data)

        sns.boxplot(
            data=plot_data, x=factor_col, y='conservative_estimate',
            ax=ax, palette='viridis'
        )

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

    if distribution_data_list:
        distribution_df = pd.concat(distribution_data_list, ignore_index=True)
        distribution_df.to_csv(output_path / '8_distribution_comparison_data.csv', index=False)

    print("Saved Plot 8: Distribution Comparison")


# ==========================================
# PLOT 9a/9b: Intersection Grid with Std
# ==========================================

def plot_intersection_grid_with_std(
    agg: pd.DataFrame,
    factor_col: str,
    title: str,
    output_file: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Create a grid showing premise type x gas decile with mean ± std bands
    use cols # group_mean total_group_std
    ."""
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
        y_max = (agg['group_mean'] + agg['total_group_std'].fillna(0)).quantile(0.95)
        y_max = max(y_max, 3500) * 1.1
        shared_ylim = (0, y_max)

    for i, premise in enumerate(premises):
        ax = axes[i]
        p_data = agg[agg['Premise Type'] == premise]

        for j, gas_bin in enumerate(gas_labels):
            g_data = p_data[p_data['gas_bin'] == gas_bin].sort_values(factor_col)
            if not g_data.empty and len(g_data) > 0:
                factors = np.array(g_data[factor_col])
                # means = np.array(g_data['mean_cost'])
                # stds = np.array(g_data['std_cost'].fillna(0))
                means = np.array(g_data['group_mean'])
                stds = np.array(g_data['total_group_std'].fillna(0))

                ax.fill_between(
                    factors,
                    means - stds,
                    means + stds,
                    alpha=0.15,
                    color=colors[j],
                    linewidth=0,
                )

                ax.plot(
                    factors,
                    means,
                    marker='o', markersize=5, label=str(gas_bin),
                    color=colors[j], linewidth=1.5
                )

        ax.set_title(f"{premise}", fontsize=12, fontweight='bold')
        ax.set_xlabel(factor_col.replace('_', ' ').title())

        if i % cols == 0:
            ax.set_ylabel("Mean £/tCO2 (±1 std)")

        ax.grid(True, alpha=0.3)

        for thr in [800, 1600, 2400]:
            if thr <= shared_ylim[1]:
                ax.axhline(thr, color='green', linestyle=':', alpha=0.5, linewidth=1)

    for ax in axes[:n_plots]:
        ax.set_ylim(shared_ylim)

    for k in range(len(premises), len(axes)):
        axes[k].axis('off')

    legend_handles = [
        Line2D([0], [0], color=colors[j], marker='o', markersize=5, linewidth=1.5)
        for j in range(len(gas_labels))
    ]

    fig.legend(
        legend_handles, [str(g) for g in gas_labels],
        loc='upper center', bbox_to_anchor=(0.5, 1.12),
        ncol=len(gas_labels), title="Gas Consumption Decile", fontsize=10
    )

    plt.suptitle(f"{title}\n(Mean ± 1 std across buildings)", fontsize=14, fontweight='bold', y=1.15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    csv_file = output_file.with_suffix('.csv').name.replace('.csv', '_data.csv')
    agg.to_csv(output_file.parent / csv_file, index=False)

    print(f"Saved: {output_file}")


def plot_intersection_internal_with_std(
    reduced_csv_path: Path,
    output_path: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot 9a: Intersection for INTERNAL wall with mean ± std."""
    agg, factor_col = prepare_intersection_data(
        reduced_csv_path, SWEEP_INTERNAL, SOLID_WALL_INTERNAL, min_sample_size
    )

    if agg is None or agg.empty:
        print("Skipping Plot 9a: No data for internal wall intersection")
        return

    plot_intersection_grid_with_std(
        agg=agg, factor_col=factor_col,
        title="Internal Wall Insulation: Premise Type × Gas Usage",
        output_file=output_path / '9a_intersection_internal_mean_std.png',
        shared_ylim=shared_ylim, n_std=n_std
    )


def plot_intersection_external_with_std(
    reduced_csv_path: Path,
    output_path: Path,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot 9b: Intersection for EXTERNAL wall with mean ± std."""
    agg, factor_col = prepare_intersection_data(
        reduced_csv_path, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, min_sample_size
    )

    if agg is None or agg.empty:
        print("Skipping Plot 9b: No data for external wall intersection")
        return

    plot_intersection_grid_with_std(
        agg=agg, factor_col=factor_col,
        title="External Wall Insulation: Premise Type × Gas Usage",
        output_file=output_path / '9b_intersection_external_mean_std.png',
        shared_ylim=shared_ylim, n_std=n_std
    )


# ==========================================
# MAIN ORCHESTRATOR
# ==========================================

def generate_all_visualizations(
    results_df: pd.DataFrame,
    reduced_csv_path: Path,
    output_dir: Path,
    results_dir: Path,
    n_std: float = N_STD_CONSERVATIVE,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    include_epistemic: bool = False,
) -> None:
    """Generate all visualizations from processed data."""
    plots_dir = output_dir / 'plots_combined'
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    print(f"Output directory: {plots_dir}")
    print(f"Conservative estimate: mean + {n_std}×std")
    print(f"Minimum sample size: {min_sample_size}")

    results_df = clean_dataframe(results_df)

    print("\n--- Basic Plots (1-3) ---")
    plot_cost_efficiency_curve(results_df, plots_dir, n_std)
    plot_viability_percentage(results_df, plots_dir, n_std)
    plot_viability_multi_threshold(results_df, plots_dir, n_std)

    print("\n--- Stratification Plots (4a-4b) ---")
    plot_gas_stratification(reduced_csv_path, plots_dir, n_std)
    plot_premise_stratification(reduced_csv_path, plots_dir, n_std)

    print("\n--- Intersection Plots (5a-5b) ---")
    shared_ylim = calculate_shared_ylim(reduced_csv_path, min_sample_size)
    print(f"Shared y-axis: {shared_ylim}")
    #shared_ylim=(0, 15000)
    plot_intersection_internal(reduced_csv_path, plots_dir, shared_ylim, min_sample_size, n_std)
    plot_intersection_external(reduced_csv_path, plots_dir, shared_ylim, min_sample_size, n_std)

    print("\n--- Mean ± Std Intersection Plots (9a-9b) ---")
    shared_ylim_std = calculate_shared_ylim_with_std(reduced_csv_path, min_sample_size)
    print(f"Shared y-axis (mean±std): {shared_ylim_std}")
    #shared_ylim_std=(0, 15000)
    plot_intersection_internal_with_std(reduced_csv_path, plots_dir, shared_ylim_std, min_sample_size, n_std)
    plot_intersection_external_with_std(reduced_csv_path, plots_dir, shared_ylim_std, min_sample_size, n_std)

    print("\n--- Heatmap Plot (6) ---")
    # plot_intersection_heatmap(reduced_csv_path, plots_dir, n_std)

    if include_epistemic:
        print("\n--- Epistemic Sensitivity Plot (7) ---")
        plot_epistemic_sensitivity(results_dir, plots_dir, n_std)
    else:
        print("\n--- Skipping Epistemic Sensitivity Plot (7) ---")
        print("  (Use --include-epistemic to enable, requires raw parquets)")

    print("\n--- Distribution Plot (8) ---")
    plot_distribution_comparison(reduced_csv_path, plots_dir, n_std)

    print("\n" + "=" * 50)
    print(f"Visualization complete. Plots saved to: {plots_dir}")
    print("=" * 50)


# ==========================================
# CLI ARGUMENT PARSING
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from aggregated sweep results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots
  python plot_sweep_results.py --output-dir wall_param_sweep/results/combined_12:34:56

  # Include epistemic sensitivity plot (requires raw parquets)
  python plot_sweep_results.py --output-dir wall_param_sweep/results/combined_12:34:56 --include-epistemic

  # Custom minimum sample size
  python plot_sweep_results.py --output-dir wall_param_sweep/results/combined_12:34:56 --min-sample 20
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory containing aggregated results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='wall_param_sweep/results',
        help='Directory containing raw parquet results (for epistemic plot)'
    )

    parser.add_argument(
        '--include-epistemic',
        action='store_true',
        help='Include epistemic sensitivity plot (requires raw parquets)'
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

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    reduced_csv = output_dir / 'reduced_building_estimates.csv'
    stats_csv = output_dir / 'sweep_by_building_category.csv'

    if not reduced_csv.exists():
        print(f"Error: Reduced CSV not found: {reduced_csv}")
        print("Run aggregate_sweep_results.py first.")
        sys.exit(1)

    if not stats_csv.exists():
        print(f"Error: Statistics CSV not found: {stats_csv}")
        print("Run aggregate_sweep_results.py first.")
        sys.exit(1)

    print(f"Loading statistics from: {stats_csv}")
    results_df = pd.read_csv(stats_csv)

    generate_all_visualizations(
        results_df=results_df,
        reduced_csv_path=reduced_csv,
        output_dir=output_dir,
        results_dir=results_dir,
        n_std=args.n_std,
        min_sample_size=args.min_sample,
        include_epistemic=args.include_epistemic,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()