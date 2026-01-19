"""
Module: visualise_wall_results_v4.py
Purpose: Visualise the outputs from wall_improvement_sweep_v4.py

Key update: Correct epistemic aggregation
- Per building: mean(p50) + n_std * std(p50) across epistemic runs
- Then aggregate across buildings for summary stats

Updates from v3:
1. Changed metric from _mean to _p50
2. Added compute_building_conservative_estimate() to collapse epistemic runs
3. All aggregations now properly collapse per-building first
4. Added n_std parameter for conservative estimate tuning
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

THRESHOLDS = [800, 1600, 2400]

# MAPPING: Matches CSV data labels -> Script keys
CATEGORY_MAP = {
    'solid_wall_internal_wall_insulation': SOLID_WALL_INTERNAL,
    'solid_wall_external_wall_insulation': SOLID_WALL_EXTERNAL,
    'solid_wall_internal': SOLID_WALL_INTERNAL,
    'solid_wall_external': SOLID_WALL_EXTERNAL,
}

# Metric column name - NOW USING P50
COST_METRIC = 'wall_installation_capex_per_net_ton_co2_wall_installation_p50'

# Epistemic aggregation parameter
N_STD_CONSERVATIVE = 1.0  # mean + N_STD * std for conservative estimate

# Gas bin configuration

GAS_LABELS = [0,1,2,3,4,5,6,7,8,9]

# Minimum sample size for intersection plots (filters out noisy bins)
MIN_SAMPLE_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser(description='Visualise Wall Sweep Results v4')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the sweep output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--min-samples', type=int, default=MIN_SAMPLE_SIZE,
                        help=f'Minimum sample size per bin for intersection plots (default: {MIN_SAMPLE_SIZE})')
    parser.add_argument('--n-std', type=float, default=N_STD_CONSERVATIVE,
                        help=f'Number of std devs for conservative estimate (default: {N_STD_CONSERVATIVE})')
    return parser.parse_args()


# ==========================================
# EPISTEMIC AGGREGATION
# ==========================================

def compute_building_conservative_estimate(
    df: pd.DataFrame, 
    metric_col: str = COST_METRIC,
    building_id: str = 'upn',
    n_std: float = N_STD_CONSERVATIVE,
) -> pd.DataFrame:
    """
    Collapse epistemic runs per building: mean(p50) + n_std * std(p50).
    Returns one row per building with the conservative estimate.
    
    Args:
        df: DataFrame with multiple rows per building (one per epistemic run)
        metric_col: The p50 metric column
        building_id: Column identifying unique buildings
        n_std: Number of std devs to add for conservative estimate
    
    Returns:
        DataFrame with one row per building, metric_col replaced with conservative estimate
    """
    if df.empty:
        return df
    
    # Columns to preserve (take first value since constant per building within a sweep)
    preserve_cols = ['building_category', 'sweep_type', 'internal_factor', 'external_factor',
                     'premise_type_filled', 'avg_gas_percentile', 'inferred_wall_type', 
                     'inferred_insulation_type', 'postcode', 'region']
    
    # Filter to columns that exist
    preserve_cols = [c for c in preserve_cols if c in df.columns]
    
    # Group by building AND sweep parameters (factor values define different sweep points)
    group_cols = [building_id]
    if 'internal_factor' in df.columns:
        group_cols.append('internal_factor')
    if 'external_factor' in df.columns:
        group_cols.append('external_factor')
    if 'sweep_type' in df.columns:
        group_cols.append('sweep_type')
    
    grouped = df.groupby(group_cols)
    
    # Compute mean and std of metric across epistemic runs
    agg_dict = {
        metric_col: ['mean', 'std', 'count'],
    }
    
    # Add first-value aggregation for preserved columns
    for col in preserve_cols:
        if col not in group_cols:
            agg_dict[col] = 'first'
    
    result = grouped.agg(agg_dict)
    
    # Flatten column names
    result.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in result.columns]
    
    # Compute conservative estimate
    mean_col = f'{metric_col}_mean'
    std_col = f'{metric_col}_std'
    count_col = f'{metric_col}_count'
    
    result[std_col] = result[std_col].fillna(0)
    result[metric_col] = result[mean_col] + n_std * result[std_col]
    
    # Keep epistemic stats for reference
    result['epistemic_mean'] = result[mean_col]
    result['epistemic_std'] = result[std_col]
    result['epistemic_n'] = result[count_col]
    
    # Clean up intermediate columns
    result = result.drop(columns=[mean_col, std_col, count_col], errors='ignore')
    
    # Flatten preserved column names (remove '_first' suffix)
    for col in preserve_cols:
        if col not in group_cols:
            old_name = f'{col}_first'
            if old_name in result.columns:
                result = result.rename(columns={old_name: col})
    
    result = result.reset_index()
    
    return result


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


def load_data(input_dir: str, n_std: float = N_STD_CONSERVATIVE) -> Dict[str, Optional[pd.DataFrame]]:
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
                'upn', 'building_category', 'sweep_type', 'internal_factor', 'external_factor',
                'premise_type_filled', 'avg_gas_percentile', COST_METRIC,
                'inferred_wall_type', 'inferred_insulation_type', 'epistemic_run_id'
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
            
            # Count raw rows before collapsing
            n_raw = len(df_full)
            n_epistemic = df_full['epistemic_run_id'].nunique() if 'epistemic_run_id' in df_full.columns else 'unknown'
            
            # COLLAPSE EPISTEMIC RUNS
            print(f"  Raw rows: {n_raw:,} (buildings × {n_epistemic} epistemic runs)")
            print(f"  Collapsing epistemic runs with n_std={n_std}...")
            
            df_collapsed = compute_building_conservative_estimate(df_full, n_std=n_std)
            
            print(f"  Collapsed to: {len(df_collapsed):,} building-factor combinations")
            
            data['detailed'] = df_collapsed
            data['detailed_raw'] = df_full  # Keep raw for epistemic sensitivity analysis
            
        except Exception as e:
            print(f"Failed to load parquet: {e}")
            import traceback
            traceback.print_exc()
            data['detailed'] = None
            data['detailed_raw'] = None
    else:
        print("Detailed parquet not found. Intersection plots will be skipped.")
        data['detailed'] = None
        data['detailed_raw'] = None

    return data


# =========================================================
# BASIC PLOTTING FUNCTIONS
# =========================================================

def plot_cost_efficiency_curve(df: pd.DataFrame, output_path: str, n_std: float = N_STD_CONSERVATIVE) -> None:
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


def plot_viability_percentage(df: pd.DataFrame, output_path: str, n_std: float = N_STD_CONSERVATIVE) -> None:
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
    ax.set_ylabel("% Viable (< £2000/tCO2)", fontsize=14)
    
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
    
    NOTE: df should already be collapsed (one row per building-factor combination)
    from compute_building_conservative_estimate().
    
    Returns aggregated data and the factor column name.
    """
    if df is None or COST_METRIC not in df.columns:
        return None, ''

    subset = filter_sweep(df, sweep_type, building_category)

    if subset.empty:
        return None, ''

    factor_col = get_factor_column(sweep_type)

    # Create gas bins
    subset = subset.copy()
    subset['gas_bin'] =  subset['avg_gas_percentile']
 

    # Clean premise types
    subset['Premise Type'] = subset['premise_type_filled'].apply(clean_premise_name)

    # Aggregate: Group by Factor, Premise, Gas Bin -> Median AND Count
    # Now each row represents one building (epistemic already collapsed)
    agg = subset.groupby([factor_col, 'Premise Type', 'gas_bin']).agg(
        median_cost=(COST_METRIC, 'median'),
        mean_cost=(COST_METRIC, 'mean'),
        sample_count=(COST_METRIC, 'count')  # This is now actual building count
    ).reset_index()

    # Filter out bins with insufficient sample size
    before_count = len(agg)
    agg = agg[agg['sample_count'] >= min_sample_size]
    after_count = len(agg)
    
    if before_count > after_count:
        filtered_count = before_count - after_count
        print(f"  Filtered {filtered_count} bins with < {min_sample_size} buildings "
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
    output_file: str,
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
    # Use ALL gas labels for consistent legend, not just those in current data
    gas_labels = GAS_LABELS
 
    colors = get_gas_colors(len(gas_labels))
    
    # Calculate shared y-axis limits if not provided
    if shared_ylim is None:
        y_min = 0
        y_max = agg['median_cost'].quantile(0.95)
        y_max = max(y_max, 3500)
        y_max = y_max * 1.1
        shared_ylim = (y_min, y_max)
    
    for i, premise in enumerate(premises):
        ax = axes[i]
        p_data = agg[agg['Premise Type'] == premise]
        for j, gas_bin in enumerate(gas_labels):
            print(gas_bin) 
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
        for thr in [800, 1600, 2400]:
            if thr <= shared_ylim[1]:
                ax.axhline(thr, color='green', linestyle=':', alpha=0.5, linewidth=1)
                if i == 0 or i == 3:
                    ax.text(ax.get_xlim()[0], thr + (shared_ylim[1] * 0.02), 
                            f'£{thr}', color='gray', fontsize=8, va='bottom')
    
    # Apply shared y-axis limits
    for ax in axes[:n_plots]:
        ax.set_ylim(shared_ylim)
    
    # Turn off unused axes
    for k in range(len(premises), len(axes)):
        axes[k].axis('off')
    
    # Create legend handles manually for ALL gas bins (not just those in current plot)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=colors[j], marker='o', markersize=6, linewidth=2)
        for j in range(len(gas_labels))
    ]
    
    fig.legend(
        legend_handles, gas_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
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
    min_sample_size: int = MIN_SAMPLE_SIZE,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot 5a: Intersection of Premise Type x Gas Decile for INTERNAL wall insulation."""
    agg, factor_col = prepare_intersection_data(df, SWEEP_INTERNAL, SOLID_WALL_INTERNAL, min_sample_size)
    
    if agg is None or agg.empty:
        print("Skipping Plot 5a: No data for internal wall intersection")
        return

    plot_intersection_grid(
        agg=agg,
        factor_col=factor_col,
        output_file=os.path.join(output_path, '5a_intersection_internal.png'),
        shared_ylim=shared_ylim,
        n_std=n_std
    )


def plot_intersection_external(
    df: pd.DataFrame, 
    output_path: str,
    shared_ylim: Optional[Tuple[float, float]] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """Plot 5b: Intersection of Premise Type x Gas Decile for EXTERNAL wall insulation."""
    agg, factor_col = prepare_intersection_data(df, SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, min_sample_size)
    
    if agg is None or agg.empty:
        print("Skipping Plot 5b: No data for external wall intersection")
        return

    plot_intersection_grid(
        agg=agg,
        factor_col=factor_col,
       
        output_file=os.path.join(output_path, '5b_intersection_external.png'),
        shared_ylim=shared_ylim,
        n_std=n_std
    )


def calculate_shared_ylim_for_intersections(
    df: pd.DataFrame,
    min_sample_size: int = MIN_SAMPLE_SIZE
) -> Optional[Tuple[float, float]]:
    """Calculate a shared y-axis limit across both internal and external intersection data."""
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
    y_max = np.percentile(all_medians, 95)
    y_max = max(y_max, 3500)
    y_max = y_max * 1.1
    
    return (y_min, y_max)

 

# =========================================================
# HEATMAPS
# =========================================================

def plot_intersection_combined_heatmap(
    df: pd.DataFrame, 
    output_path: str,
    n_std: float = N_STD_CONSERVATIVE
) -> None:
    """
    Plot 6: Combined heatmap showing cost efficiency across premise types and gas deciles.
    
    NOTE: df should already be collapsed (one row per building-factor combination).
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

        subset = subset.copy()
        
        # Create gas bins
        subset['gas_bin'] = pd.cut(
            subset['avg_gas_percentile'],
            bins=GAS_BINS,
            labels=GAS_LABELS
        )
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
    plt.savefig(os.path.join(output_path, '6_intersection_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Plot 6: Intersection Heatmap")

 

# =========================================================
# EPISTEMIC SENSITIVITY PLOT
# =========================================================
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_epistemic_sensitivity(
    df_raw: pd.DataFrame,
    output_path: str
) -> None:
    """
    Plot 8: Show how results vary across epistemic runs.
    Saves separate figures for Internal and External walls but maintains a shared Y-axis scale.
    """
    if df_raw is None or COST_METRIC not in df_raw.columns:
        print("Skipping Plot 8: No raw data available")
        return
    
    if 'epistemic_run_id' not in df_raw.columns:
        print("Skipping Plot 8: No epistemic_run_id column")
        return

    # Configuration for the two separate plots
    configs = [
        (SWEEP_INTERNAL, SOLID_WALL_INTERNAL, 'Internal Wall', 'internal'),
        (SWEEP_EXTERNAL, SOLID_WALL_EXTERNAL, 'External Wall', 'external'),
    ]

    # --- Step 1: Pre-calculate data to find the Global Max Y ---
    processed_data = []
    global_max_y = 0
    thresholds = [800, 1600, 2400, 3200]

    for sweep_type, building_cat, title, file_suffix in configs:
        subset = filter_sweep(df_raw, sweep_type, building_cat)
        
        if subset.empty:
            processed_data.append(None)
            continue

        factor_col = get_factor_column(sweep_type)
        
        # Compute median across buildings for each epistemic run
        run_medians = []
        for (factor_val, run_id), group in subset.groupby([factor_col, 'epistemic_run_id']):
            median_cost = group[COST_METRIC].median()
            run_medians.append({
                'factor': factor_val,
                'run_id': run_id,
                'median_cost': median_cost
            })
        
        run_df = pd.DataFrame(run_medians)
        
        if run_df.empty:
            processed_data.append(None)
            continue
        
        # Summarize mean ± std
        summary = run_df.groupby('factor')['median_cost'].agg(['mean', 'std']).reset_index()
        summary['std'] = summary['std'].fillna(0)
        
        # Track the highest value (Mean + Std) to set shared axis later
        current_max = (summary['mean'] + summary['std']).max()
        if current_max > global_max_y:
            global_max_y = current_max

        # Store processed data for the plotting step
        processed_data.append({
            'title': title,
            'summary': summary,
            'factor_col': factor_col,
            'suffix': file_suffix
        })

    # Determine Shared Y Limit (Max of data or Max threshold, plus padding)
    y_limit_top = max(global_max_y, max(thresholds)) * 1.1

    # --- Step 2: Generate and Save Separate Plots ---
    for data in processed_data:
        # Handle cases with no data
        if data is None:
            continue

        summary = data['summary']
        
        # Create a new figure for each plot
        fig, ax = plt.subplots(figsize=(8, 6))

        factors = summary['factor'].values
        means = summary['mean'].values
        stds = summary['std'].values
        
        # Plot Fill
        ax.fill_between(
            factors,
            means - stds,
            means + stds,
            alpha=0.3,
            label='±1 std (epistemic)'
        )
        # Plot Mean Line
        ax.plot(factors, means, 'o-', linewidth=2, label='Mean across runs')
        
        # Add threshold lines
        for thr in thresholds:
            ax.axhline(thr, color='green', linestyle='--', alpha=0.5)
            # Only add text if within reasonable view
            if thr < y_limit_top:
                ax.text(factors.min(), thr + 50, f'£{thr}', fontsize=8, color='gray')
        
        # Formatting
        ax.set_xlabel(f"{data['factor_col'].replace('_', ' ').title()}")
        ax.set_ylabel('Median £/tCO2')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # --- Apply Shared Y-Axis Manually ---
        ax.set_ylim(bottom=0, top=y_limit_top)

        plt.tight_layout()
        filename = f"8_epistemic_sensitivity_{data['suffix']}.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300)
        plt.close() # Close figure to free memory
        print(f"Saved Plot 8 ({data['title']})")

# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_args()
    n_std = args.n_std

    # Setup Output Directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading from: {args.input_dir}")
    print(f"Saving plots to: {output_dir}")
    print(f"Conservative estimate: mean + {n_std}×std")
    print("=" * 50)

    # Load Data (includes epistemic collapse)
    data = load_data(args.input_dir, n_std=n_std)
    print("=" * 50)

    # Generate Basic Plots (1-4) from pre-aggregated CSVs
    print("\nGenerating basic plots...")
    if data['main'] is not None:
        plot_cost_efficiency_curve(data['main'], output_dir, n_std=n_std)
        plot_viability_percentage(data['main'], output_dir, n_std=n_std)

    if data['gas'] is not None:
        plot_gas_stratification(data['gas'], output_dir)

    if data['premise'] is not None:
        plot_premise_stratification(data['premise'], output_dir)

    # Generate Expanded Intersection Plots (5-8) from collapsed parquet
    print("\nGenerating intersection plots...")
    if data['detailed'] is not None:
        min_samples = args.min_samples
        print(f"Using minimum sample size: {min_samples}")
        
        # Calculate shared y-axis limits
        shared_ylim = calculate_shared_ylim_for_intersections(data['detailed'], min_samples)
        print(f"Using shared y-axis: {shared_ylim}")
        
        plot_intersection_internal(data['detailed'], output_dir, shared_ylim=shared_ylim, 
                                   min_sample_size=min_samples, n_std=n_std)
        plot_intersection_external(data['detailed'], output_dir, shared_ylim=shared_ylim, 
                                   min_sample_size=min_samples, n_std=n_std)
        
        
        
       
        
    else:
        print("Skipping intersection plots: detailed parquet not available")

    # Plot epistemic sensitivity from raw data
    if data.get('detailed_raw') is not None:
        print("\nGenerating epistemic sensitivity plot...")
        plot_epistemic_sensitivity(data['detailed_raw'], output_dir)

    print("\n" + "=" * 50)
    print("Visualization complete.")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()