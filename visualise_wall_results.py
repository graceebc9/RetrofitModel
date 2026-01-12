"""
Module: visualise_wall_results_expanded.py
Purpose: Visualise the outputs from wall_improvement_sweep_v3.py
Updates:
1. Expanded intersection plots: separate plots for internal and external factors
2. Each plot shows premise types with gas decile breakdown
3. Refactored with helper functions to reduce code duplication
4. Added constants for magic strings
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from typing import Optional, Dict, List, Tuple

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

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

THRESHOLDS = [800, 1500, 2200]

# MAPPING: Matches CSV data labels -> Script keys
CATEGORY_MAP = {
    'solid_wall_internal_wall_insulation': SOLID_WALL_INTERNAL,
    'solid_wall_external_wall_insulation': SOLID_WALL_EXTERNAL,
    'solid_wall_internal': SOLID_WALL_INTERNAL,
    'solid_wall_external': SOLID_WALL_EXTERNAL,
}

# Metric column name
COST_METRIC = 'wall_installation_capex_per_net_ton_co2_wall_installation_mean'

# Gas bin configuration
GAS_BINS = [-0.1, 2, 4, 6, 8, 10.1]
GAS_LABELS = ['0-2 (Low)', '2-4', '4-6', '6-8', '8-10 (High)']

# Minimum sample size for intersection plots (filters out noisy bins)
MIN_SAMPLE_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser(description='Visualise Wall Sweep Results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the sweep output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--min-samples', type=int, default=MIN_SAMPLE_SIZE,
                        help=f'Minimum sample size per bin for intersection plots (default: {MIN_SAMPLE_SIZE})')
    return parser.parse_args()


# ==========================================
# HELPER FUNCTIONS
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


def create_building_category_if_missing(row: pd.Series) -> str:
    """Helper to recreate category if missing in parquet."""
    if 'building_category' in row.index and pd.notna(row.get('building_category')):
        return row['building_category']
    # Fallback logic mirroring the sweep script
    w_type = row.get('inferred_wall_type', 'unknown')
    i_type = row.get('inferred_insulation_type', 'unknown')
    if w_type == 'solid_wall':
        return f'solid_wall_{i_type}'
    return w_type


def get_gas_colors(n_bins: int) -> np.ndarray:
    """
    Return colors for gas bins.
    Yellow (low gas usage) -> Red (high gas usage)
    High gas usage typically means better cost efficiency for insulation.
    """
    return cm.YlOrRd(np.linspace(0.3, 1, n_bins))


def load_data(input_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Loads CSV summaries AND the detailed parquet if available."""
    data = {}

    # 1. Load CSV Summaries
    csv_paths = {
        'main': 'sweep_by_building_category.csv',
        'gas': 'category_x_gas_decile.csv',
        'premise': 'category_x_premise_type.csv'
    }

    for key, filename in csv_paths.items():
        path = os.path.join(input_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if df.empty:
                print(f"Warning: {key} file exists but is empty")
                data[key] = None
            else:
                data[key] = clean_dataframe(df)
                print(f"Loaded {key}: {len(data[key])} rows")
        else:
            print(f"Warning: Could not find {path}")
            data[key] = None

    # 2. Load Detailed Parquet (for intersection plots)
    parquet_path = os.path.join(input_dir, 'detailed_results.parquet')
    if os.path.exists(parquet_path):
        print(f"Loading detailed parquet from {parquet_path}...")
        try:
            # Define columns we need
            required_cols = [
                'building_category', 'sweep_type', 'internal_factor', 'external_factor',
                'premise_type_filled', 'avg_gas_percentile', COST_METRIC,
                'inferred_wall_type', 'inferred_insulation_type'
            ]
            
            # Try loading only required columns first
            try:
                df_full = pd.read_parquet(parquet_path, columns=required_cols)
            except Exception:
                # Fall back to loading all if column subset fails
                df_full = pd.read_parquet(parquet_path)
            
            # Standardize category
            if 'building_category' not in df_full.columns or df_full['building_category'].isna().all():
                df_full['building_category'] = df_full.apply(create_building_category_if_missing, axis=1)
            df_full['building_category'] = df_full['building_category'].replace(CATEGORY_MAP)
            
            data['detailed'] = df_full
            print(f"Loaded detailed parquet: {len(df_full)} rows")
        except Exception as e:
            print(f"Failed to load parquet: {e}")
            data['detailed'] = None
    else:
        print("Detailed parquet not found. Intersection plots will be skipped.")
        data['detailed'] = None

    return data


# =========================================================
# BASIC PLOTTING FUNCTIONS
# =========================================================

def plot_cost_efficiency_curve(df: pd.DataFrame, output_path: str) -> None:
    """Plot 1: Cost efficiency curves for internal vs external insulation."""
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    internal_data = filter_sweep(df, SWEEP_INTERNAL, SOLID_WALL_INTERNAL)
    internal_data = internal_data.sort_values('internal_factor')

    external_data = filter_sweep(df, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)
    external_data = external_data.sort_values('external_factor')

    min_x = 0.5  # Default

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
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '1_cost_efficiency_curve.png'), dpi=300)
    plt.close()
    print("Saved Plot 1: Cost Efficiency Curve")


def plot_viability_percentage(df: pd.DataFrame, output_path: str) -> None:
    """Plot 2: Percentage of properties viable at £2000/tCO2 threshold."""
    if df is None:
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
            label='Solid Wall (Internal)'
        )

    if not ext_data.empty:
        ax.plot(
            np.array(ext_data['external_factor']),
            np.array(ext_data[col]),
            marker='s', color=PALETTE[SOLID_WALL_EXTERNAL],
            label='Solid Wall (External)'
        )

    
    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("% Viable", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '2_viability_ramp.png'), dpi=300)
    plt.close()
    print("Saved Plot 2: Viability Ramp")


def plot_gas_stratification(df: pd.DataFrame, output_path: str) -> None:
    """Plot 3: Gas decile impact on external wall insulation efficiency."""
    if df is None:
        return

    subset = filter_sweep(df, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)

    if subset.empty:
        print("Skipping Plot 3: No external wall data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    deciles = sorted(subset['gas_decile'].unique())
    colors = get_gas_colors(len(deciles))

    for i, decile in enumerate(deciles):
        group = subset[subset['gas_decile'] == decile].sort_values('external_factor')
        ax.plot(
            np.array(group['external_factor']),
            np.array(group['median']),
            marker='o', label=decile, color=colors[i], linewidth=2
        )

    
    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.legend(title="Gas Decile")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '3_gas_decile_impact.png'), dpi=300)
    plt.close()
    print("Saved Plot 3: Gas Decile Impact")


def plot_premise_stratification(df: pd.DataFrame, output_path: str) -> None:
    """Plot 4: Premise type impact on internal wall insulation efficiency."""
    if df is None:
        return

    subset = filter_sweep(df, SWEEP_INTERNAL, SOLID_WALL_INTERNAL)

    if subset.empty:
        print("Skipping Plot 4: No internal wall data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)
    premises = sorted(subset['Premise Type'].unique())

    for premise in premises:
        group = subset[subset['Premise Type'] == premise].sort_values('internal_factor')
        ax.plot(
            np.array(group['internal_factor']),
            np.array(group['median']),
            marker='o', label=premise
        )

    
    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.legend(title="Premise Type")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '4_premise_type_impact.png'), dpi=300)
    plt.close()
    print("Saved Plot 4: Premise Type Impact")


# =========================================================
# EXPANDED INTERSECTION PLOTS
# =========================================================

def prepare_intersection_data(
    df: pd.DataFrame,
    sweep_type: str,
    building_category: str,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Prepare data for intersection plots.
    Returns aggregated data and the factor column name.
    
    Args:
        df: Detailed results dataframe
        sweep_type: 'internal' or 'external'
        building_category: Building category to filter on
        min_sample_size: Minimum number of buildings required per bin (default from MIN_SAMPLE_SIZE)
    
    Returns:
        Tuple of (aggregated dataframe, factor column name)
    """
    if df is None or COST_METRIC not in df.columns:
        return None, ''

    subset = filter_sweep(df, sweep_type, building_category)

    if subset.empty:
        return None, ''

    factor_col = get_factor_column(sweep_type)

    # Create gas bins
    subset['gas_bin'] = pd.cut(
        subset['avg_gas_percentile'],
        bins=GAS_BINS,
        labels=GAS_LABELS
    )

    # Clean premise types
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

    # Aggregate: Group by Factor, Premise, Gas Bin -> Median AND Count
    agg = subset.groupby([factor_col, 'Premise Type', 'gas_bin']).agg(
        median_cost=(COST_METRIC, 'median'),
        sample_count=(COST_METRIC, 'count')
    ).reset_index()

    # Filter out bins with insufficient sample size
    before_count = len(agg)
    agg = agg[agg['sample_count'] >= min_sample_size]
    after_count = len(agg)
    
    if before_count > after_count:
        filtered_count = before_count - after_count
        print(f"  Filtered {filtered_count} bins with < {min_sample_size} samples "
              f"({sweep_type} {building_category})")

    return agg, factor_col


def get_premise_types_to_plot(agg: pd.DataFrame, max_types: int = 6) -> List[str]:
    """
    Get list of premise types to plot, prioritizing common building types.
    Returns up to max_types premises.
    """
    all_premises = sorted(agg['Premise Type'].unique())
    
    # Priority order for common UK building types
    priority_types = [
        'Detached',
        'Semi Detached', 
        'End Terrace',
        'Mid Terrace',
        'Terraced',
        'Bungalow',
        'Flat',
        'Maisonette'
    ]
    
    # Get premises in priority order
    ordered_premises = []
    for ptype in priority_types:
        matches = [p for p in all_premises if ptype.lower() in p.lower()]
        ordered_premises.extend(matches)
    
    # Add any remaining premises not in priority list
    remaining = [p for p in all_premises if p not in ordered_premises]
    ordered_premises.extend(remaining)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_premises = []
    for p in ordered_premises:
        if p not in seen:
            seen.add(p)
            unique_premises.append(p)
    
    return unique_premises[:max_types]


def plot_intersection_grid(
    agg: pd.DataFrame,
    factor_col: str,
    title: str,
    output_file: str,
    shared_ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Create a grid of subplots showing premise type x gas decile interaction.
    
    Args:
        agg: Aggregated dataframe with columns [factor_col, 'Premise Type', 'gas_bin', 'median_cost']
        factor_col: Name of the factor column (e.g., 'internal_factor')
        title: Plot title
        output_file: Full path to save the plot
        shared_ylim: Optional (min, max) tuple for shared y-axis. If None, calculated from data.
    """
    premises = get_premise_types_to_plot(agg)
    n_plots = len(premises)
    
    if n_plots == 0:
        print(f"No premise types to plot for {title}")
        return

    # Calculate grid dimensions
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    # Create figure with shared y-axis
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True)
    
    # Handle single plot case
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Gas colors: Yellow (low gas) -> Red (high gas)
    gas_labels = [l for l in GAS_LABELS if l in agg['gas_bin'].unique()]
    colors = get_gas_colors(len(gas_labels))

    # Calculate shared y-axis limits if not provided
    if shared_ylim is None:
        y_min = 0
        y_max = agg['median_cost'].quantile(0.95)  # Use 95th percentile to avoid outliers
        y_max = max(y_max, 3500)  # Ensure we show at least up to £3000 threshold
        # Add 10% padding
        y_max = y_max * 1.1
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
                    marker='o',
                    markersize=6,
                    label=gas_bin,
                    color=colors[j],
                    linewidth=2
                )

        ax.set_title(f"{premise}", fontsize=12, fontweight='bold')
        ax.set_xlabel(factor_col.replace('_', ' ').title())
        
        # Only show y-label on leftmost plots
        if i % cols == 0:
            ax.set_ylabel("Median £/tCO2")
            ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        ax.grid(True, alpha=0.3)

        # Add threshold lines
        for thr in [800, 1500, 2200]:
            if thr <= shared_ylim[1]:
                ax.axhline(thr, color='green', linestyle=':', alpha=0.5, linewidth=1)
                # Add threshold label only on first subplot
                if i == 0 or i == 3:
                    ax.text(ax.get_xlim()[0], thr + (shared_ylim[1] * 0.02), 
                            f'£{thr}', color='gray', fontsize=8, va='bottom')

    # Apply shared y-axis limits
    for ax in axes[:n_plots]:
        ax.set_ylim(shared_ylim)

    # Turn off unused axes
    for k in range(len(premises), len(axes)):
        axes[k].axis('off')

    # Single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(gas_labels),
        title="Gas Consumption Decile",
        fontsize=10
    )


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_intersection_internal(
    df: pd.DataFrame, 
    output_path: str,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> None:
    """
    Plot 5a: Intersection of Premise Type x Gas Decile for INTERNAL wall insulation.
    """
    agg, factor_col = prepare_intersection_data(df, SWEEP_INTERNAL, SOLID_WALL_INTERNAL, min_sample_size)
    
    if agg is None or agg.empty:
        print("Skipping Plot 5a: No data for internal wall intersection")
        return

    plot_intersection_grid(
        agg=agg,
        factor_col=factor_col,
        title="Internal Wall Insulation: Building Form vs Gas Usage",
        output_file=os.path.join(output_path, '5a_intersection_internal.png'),
        shared_ylim=shared_ylim
    )


def plot_intersection_external(
    df: pd.DataFrame, 
    output_path: str,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> None:
    """
    Plot 5b: Intersection of Premise Type x Gas Decile for EXTERNAL wall insulation.
    """
    agg, factor_col = prepare_intersection_data(df, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, min_sample_size)
    
    if agg is None or agg.empty:
        print("Skipping Plot 5b: No data for external wall intersection")
        return

    plot_intersection_grid(
        agg=agg,
        factor_col=factor_col,
        title="External Wall Insulation: Building Form vs Gas Usage",
        output_file=os.path.join(output_path, '5b_intersection_external.png'),
        shared_ylim=shared_ylim
    )


def calculate_shared_ylim_for_intersections(
    df: pd.DataFrame,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> Optional[Tuple[float, float]]:
    """
    Calculate a shared y-axis limit across both internal and external intersection data.
    This allows direct visual comparison between the two plot types.
    """
    if df is None or COST_METRIC not in df.columns:
        return None
    
    all_medians = []
    
    for sweep_type, building_cat in [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL)
    ]:
        agg, _ = prepare_intersection_data(df, sweep_type, building_cat, min_sample_size)
        if agg is not None and not agg.empty:
            all_medians.extend(agg['median_cost'].dropna().tolist())
    
    if not all_medians:
        return None
    
    y_min = 0
    y_max = np.percentile(all_medians, 95)  # 95th percentile to avoid outliers
    y_max = max(y_max, 3500)  # Ensure we show at least up to £3000 threshold
    y_max = y_max * 1.1  # Add 10% padding
    
    return (y_min, y_max)


def plot_intersection_combined_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot 6: Combined heatmap showing cost efficiency across premise types and gas deciles.
    Creates side-by-side heatmaps for internal and external insulation.
    """
    if df is None or COST_METRIC not in df.columns:
        print("Skipping Plot 6: No detailed data available")
        return

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

        # Create gas bins
        subset['gas_bin'] = pd.cut(
            subset['avg_gas_percentile'],
            bins=GAS_BINS,
            labels=GAS_LABELS
        )
        subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

        # Pivot for heatmap: rows = premise type, cols = gas bin, values = median cost
        # Use middle factor value for comparison
        factor_col = get_factor_column(sweep_type)
        mid_factor = subset[factor_col].median()
        
        # Filter to records near the median factor
        factor_range = subset[factor_col].max() - subset[factor_col].min()
        tolerance = factor_range * 0.1 if factor_range > 0 else 0.1
        mid_subset = subset[abs(subset[factor_col] - mid_factor) <= tolerance]

        if mid_subset.empty:
            mid_subset = subset

        pivot = mid_subset.pivot_table(
            values=COST_METRIC,
            index='Premise Type',
            columns='gas_bin',
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
            cmap='RdYlGn_r',  # Red (high cost) -> Green (low cost)
            annot=True,
            fmt='.0f',
            cbar_kws={'label': '£/tCO2'},
            linewidths=0.5
        )
        
        
        ax.set_xlabel("Gas Consumption Decile")
        ax.set_ylabel("Building Type")

    plt.suptitle(
        "Cost Efficiency Heatmap: Premise Type vs Gas Usage",
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '6_intersection_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 6: Intersection Heatmap")


def plot_viability_matrix(df: pd.DataFrame, output_path: str, threshold: float = 2000) -> None:
    """
    Plot 7: Viability matrix showing % of properties below threshold for each premise/gas combination.
    """
    if df is None or COST_METRIC not in df.columns:
        print("Skipping Plot 7: No detailed data available")
        return

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

        # Create gas bins
        subset['gas_bin'] = pd.cut(
            subset['avg_gas_percentile'],
            bins=GAS_BINS,
            labels=GAS_LABELS
        )
        subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

        # Calculate viability (% below threshold) for each combination
        subset['viable'] = subset[COST_METRIC] < threshold

        viability = subset.groupby(['Premise Type', 'gas_bin'])['viable'].mean() * 100
        viability = viability.reset_index()
        
        pivot = viability.pivot(index='Premise Type', columns='gas_bin', values='viable')

        if pivot.empty:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        # Plot heatmap
        sns.heatmap(
            pivot,
            ax=ax,
            cmap='RdYlGn',  # Red (low viability) -> Green (high viability)
            annot=True,
            fmt='.0f',
            vmin=0,
            vmax=100,
            cbar_kws={'label': '% Viable'},
            linewidths=0.5
        )

        ax.set_title(f"{title}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Gas Consumption Decile")
        ax.set_ylabel("Building Type")

    plt.suptitle(
        f"Viability Matrix: % Properties Below £{threshold}/tCO2",
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '7_viability_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 7: Viability Matrix")


# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_args()

    # Setup Output Directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading from: {args.input_dir}")
    print(f"Saving plots to: {output_dir}")
    print("=" * 50)

    # Load Data
    data = load_data(args.input_dir)
    print("=" * 50)

    # Generate Basic Plots (1-4)
    print("\nGenerating basic plots...")
    if data['main'] is not None:
        plot_cost_efficiency_curve(data['main'], output_dir)
        plot_viability_percentage(data['main'], output_dir)

    if data['gas'] is not None:
        plot_gas_stratification(data['gas'], output_dir)

    if data['premise'] is not None:
        plot_premise_stratification(data['premise'], output_dir)

    # Generate Expanded Intersection Plots (5-7)
    print("\nGenerating intersection plots...")
    if data['detailed'] is not None:
        min_samples = args.min_samples
        print(f"Using minimum sample size: {min_samples}")
        
        # Calculate shared y-axis limits across both internal and external
        shared_ylim = calculate_shared_ylim_for_intersections(data['detailed'], min_samples)
        print(f"Using shared y-axis: {shared_ylim}")
        
        plot_intersection_internal(data['detailed'], output_dir, shared_ylim=shared_ylim, min_sample_size=min_samples)
        plot_intersection_external(data['detailed'], output_dir, shared_ylim=shared_ylim, min_sample_size=min_samples)
        plot_intersection_combined_heatmap(data['detailed'], output_dir)
        plot_viability_matrix(data['detailed'], output_dir)
    else:
        print("Skipping intersection plots: detailed parquet not available")

    print("\n" + "=" * 50)
    print("Visualization complete.")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()