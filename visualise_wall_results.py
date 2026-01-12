"""
Module: visualise_wall_results.py
Purpose: Visualise the outputs from wall_improvement_sweep_v3.py
Updates:
1. Added 'plot_intersection_premise_gas' to visualize the interaction between form and usage.
2. Loads 'detailed_results.parquet' for granular analysis.
3. Uses robust manual plotting (ax.plot) with np.array() to avoid pandas/matplotlib errors.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

# ==========================================
# CONFIGURATION
# ==========================================

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

PALETTE = {
    'solid_wall_internal': '#1f77b4',  # Blue
    'solid_wall_external': '#ff7f0e',  # Orange
    'cavity_wall': '#2ca02c',          # Green
}

THRESHOLDS = [1000, 2000, 3000, 4000]

# MAPPING: Matches CSV data labels -> Script keys
CATEGORY_MAP = {
    'solid_wall_internal_wall_insulation': 'solid_wall_internal',
    'solid_wall_external_wall_insulation': 'solid_wall_external',
    'solid_wall_internal': 'solid_wall_internal',
    'solid_wall_external': 'solid_wall_external'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Visualise Wall Sweep Results')
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Path to the sweep output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots')
    return parser.parse_args()

def clean_dataframe(df):
    """Standardizes category names and removes outliers."""
    if df is None: return None
    
    # Apply Mapping
    if 'building_category' in df.columns:
        df['building_category'] = df['building_category'].replace(CATEGORY_MAP)
        
    # Remove Infinity/Outliers for plotting
    if 'median' in df.columns:
        df = df[df['median'] < 100000] 
        
    return df

def load_data(input_dir):
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
            data[key] = clean_dataframe(df)
            print(f"Loaded {key}: {len(data[key])} rows")
        else:
            print(f"Warning: Could not find {path}")
            data[key] = None

    # 2. Load Detailed Parquet (for intersection plot)
    parquet_path = os.path.join(input_dir, 'detailed_results.parquet')
    if os.path.exists(parquet_path):
        print(f"Loading detailed parquet from {parquet_path} (this may take a moment)...")
        # Only load necessary columns to save memory
        cols = [
            'building_category', 'sweep_type', 'internal_factor', 'external_factor', 
            'premise_type_filled', 'avg_gas_percentile', 
            'wall_installation_capex_per_net_ton_co2_wall_installation_mean'
        ]
        # Handle case where columns might differ slightly
        try:
            df_full = pd.read_parquet(parquet_path)
            # Standardize category immediately
            df_full['building_category'] = df_full.apply(create_building_category_if_missing, axis=1)
            df_full['building_category'] = df_full['building_category'].replace(CATEGORY_MAP)
            data['detailed'] = df_full
            print(f"Loaded detailed parquet: {len(df_full)} rows")
        except Exception as e:
            print(f"Failed to load parquet: {e}")
            data['detailed'] = None
    else:
        print("Detailed parquet not found. Intersection plot will be skipped.")
        data['detailed'] = None

    return data

def create_building_category_if_missing(row):
    """Helper to recreate category if missing in parquet"""
    if 'building_category' in row: return row['building_category']
    # Fallback logic mirroring the sweep script
    w_type = row.get('inferred_wall_type', 'unknown')
    i_type = row.get('inferred_insulation_type', 'unknown')
    if w_type == 'solid_wall': return f'solid_wall_{i_type}'
    return w_type

# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_cost_efficiency_curve(df, output_path):
    if df is None: return

    fig, ax = plt.subplots(figsize=(10, 7))

    internal_data = df[
        (df['sweep_type'] == 'internal') & 
        (df['building_category'] == 'solid_wall_internal')
    ].sort_values('internal_factor')

    external_data = df[
        (df['sweep_type'] == 'external') & 
        (df['building_category'] == 'solid_wall_external')
    ].sort_values('external_factor')

    if not internal_data.empty:
        ax.plot(np.array(internal_data['internal_factor']), 
                np.array(internal_data['median']), 
                marker='o', label='Solid Wall (Internal)', color=PALETTE['solid_wall_internal'], linewidth=2.5)
    
    if not external_data.empty:
        ax.plot(np.array(external_data['external_factor']), 
                np.array(external_data['median']), 
                marker='s', label='Solid Wall (External)', color=PALETTE['solid_wall_external'], linewidth=2.5)

    # Thresholds
    min_x = 0
    if not internal_data.empty: min_x = internal_data['internal_factor'].min()
    elif not external_data.empty: min_x = external_data['external_factor'].min()

    for thr in THRESHOLDS:
        ax.axhline(thr, color='gray', linestyle='--', alpha=0.5)
        ax.text(min_x, thr + 50, f'£{thr}/tCO2', color='gray', fontsize=9)

    ax.set_title("Cost Efficiency: Improvement Factor vs Cost per tCO2", fontsize=16)
    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '1_cost_efficiency_curve.png'), dpi=300)
    plt.close()
    print("Saved Plot 1")

def plot_viability_percentage(df, output_path):
    if df is None: return

    fig, ax = plt.subplots(figsize=(10, 7))
    col = 'pct_below_2000'
    if col not in df.columns: return

    int_data = df[(df['sweep_type'] == 'internal') & (df['building_category'] == 'solid_wall_internal')].sort_values('internal_factor')
    ext_data = df[(df['sweep_type'] == 'external') & (df['building_category'] == 'solid_wall_external')].sort_values('external_factor')

    if not int_data.empty:
        ax.plot(np.array(int_data['internal_factor']), np.array(int_data[col]), 
                marker='o', color=PALETTE['solid_wall_internal'], label='Solid Wall (Internal)')

    if not ext_data.empty:
        ax.plot(np.array(ext_data['external_factor']), np.array(ext_data[col]), 
                marker='s', color=PALETTE['solid_wall_external'], label='Solid Wall (External)')

    ax.set_title(f"Market Viability (< £2000/tCO2)", fontsize=16)
    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("% Viable", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '2_viability_ramp.png'), dpi=300)
    plt.close()
    print("Saved Plot 2")

def plot_gas_stratification(df, output_path):
    if df is None: return

    subset = df[
        (df['sweep_type'] == 'external') & 
        (df['building_category'] == 'solid_wall_external')
    ].copy()

    if subset.empty: return

    fig, ax = plt.subplots(figsize=(10, 7))
    deciles = sorted(subset['gas_decile'].unique())
    colors = cm.YlOrRd(np.linspace(0.3, 1, len(deciles)))
    
    for i, decile in enumerate(deciles):
        group = subset[subset['gas_decile'] == decile].sort_values('external_factor')
        ax.plot(np.array(group['external_factor']), np.array(group['median']), 
                marker='o', label=decile, color=colors[i], linewidth=2)

    ax.set_title("Gas Impact: Solid Wall (External) Efficiency", fontsize=16)
    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    
    ax.legend(title="Gas Decile")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '3_gas_decile_impact.png'), dpi=300)
    plt.close()
    print("Saved Plot 3")

def plot_premise_stratification(df, output_path):
    if df is None: return

    subset = df[
        (df['sweep_type'] == 'internal') & 
        (df['building_category'] == 'solid_wall_internal')
    ].copy()

    if subset.empty: return

    fig, ax = plt.subplots(figsize=(10, 7))
    subset['Premise Type'] = subset['premise_type_filled'].str.replace('_', ' ').str.title()
    premises = sorted(subset['Premise Type'].unique())
    
    for premise in premises:
        group = subset[subset['Premise Type'] == premise].sort_values('internal_factor')
        ax.plot(np.array(group['internal_factor']), np.array(group['median']), marker='o', label=premise)

    ax.set_title("Form Factor: Solid Wall (Internal) Efficiency", fontsize=16)
    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    
    ax.legend(title="Premise Type")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '4_premise_type_impact.png'), dpi=300)
    plt.close()
    print("Saved Plot 4")

def plot_intersection_premise_gas(df, output_path):
    """
    Plot 5: Intersection of Premise Type x Gas Decile.
    Creates a grid of plots (one per premise type) showing lines for gas deciles.
    """
    if df is None: 
        print("Skipping Plot 5: Detailed parquet data needed.")
        return

    # 1. Filter for Solid Wall Internal (to keep analysis focused)
    metric = 'wall_installation_capex_per_net_ton_co2_wall_installation_mean'
    if metric not in df.columns:
        print(f"Skipping Plot 5: Metric column {metric} not found.")
        return

    subset = df[
        (df['sweep_type'] == 'internal') & 
        (df['building_category'] == 'solid_wall_internal')
    ].copy()

    if subset.empty: 
        print("Skipping Plot 5: No matching data (Solid Wall Internal).")
        return

    # 2. Create Gas Bins
    subset['gas_bin'] = pd.cut(
        subset['avg_gas_percentile'],
        bins=[-0.1, 2, 4, 6, 8, 10.1],
        labels=['0-2 (Low)', '2-4', '4-6', '6-8', '8-10 (High)']
    )

    # 3. Clean Premise Types
    subset['Premise Type'] = subset['premise_type_filled'].str.replace('_', ' ').str.title()
    
    # 4. Aggregate: Group by Factor, Premise, Gas Bin -> Median
    agg = subset.groupby(['internal_factor', 'Premise Type', 'gas_bin'])[metric].median().reset_index()
    agg = agg.rename(columns={metric: 'median_cost'})

    # 5. Setup Grid Plot
    premises = sorted(agg['Premise Type'].unique())
    # Keep only top 4 common types to avoid clutter if many exist
    common_types = ['Detached', 'Semi Detached', 'Terraced'] 
    premises = [p for p in premises if any(c in p for c in common_types)][:6]

    n_plots = len(premises)
    if n_plots == 0: return

    cols = 2
    rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), constrained_layout=True)
    axes = axes.flatten()

    # Gas Colors
    gas_labels = sorted(agg['gas_bin'].unique())
    colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(gas_labels))) # Red (High Cost/Low Gas) to Green? 
    # Actually standard Gas logic: High Gas = Low Cost = Good. 
    # Let's use standard Sequential: Light Yellow (Low Gas) -> Dark Red (High Gas)
    colors = cm.YlOrRd(np.linspace(0.3, 1, len(gas_labels)))

    for i, premise in enumerate(premises):
        ax = axes[i]
        p_data = agg[agg['Premise Type'] == premise]
        
        for j, gas_bin in enumerate(gas_labels):
            g_data = p_data[p_data['gas_bin'] == gas_bin].sort_values('internal_factor')
            if not g_data.empty:
                ax.plot(
                    np.array(g_data['internal_factor']),
                    np.array(g_data['median_cost']),
                    marker='.',
                    label=gas_bin,
                    color=colors[j],
                    linewidth=2
                )
        
        ax.set_title(f"{premise}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Internal Factor")
        ax.set_ylabel("Median £/tCO2")
        
        ax.grid(True, alpha=0.3)
        # Add thresholds
        for thr in [1000, 2000]:
            ax.axhline(thr, color='gray', linestyle=':', alpha=0.5)

    # Turn off unused axes
    for k in range(i + 1, len(axes)):
        axes[k].axis('off')

    # Single Legend
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(gas_labels), title="Gas Consumption Decile")

    plt.suptitle("Intersection: Building Form vs Gas Usage (Solid Wall Internal)", fontsize=18, y=1.05)
    
    out_file = os.path.join(output_path, '5_intersection_premise_gas.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Plot 5: {out_file}")

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

    # Load Data
    data = load_data(args.input_dir)

    # Generate Standard Plots
    if data['main'] is not None:
        plot_cost_efficiency_curve(data['main'], output_dir)
        plot_viability_percentage(data['main'], output_dir)
    
    if data['gas'] is not None:
        plot_gas_stratification(data['gas'], output_dir)
    
    if data['premise'] is not None:
        plot_premise_stratification(data['premise'], output_dir)

    # Generate Intersection Plot (New!)
    if data['detailed'] is not None:
        plot_intersection_premise_gas(data['detailed'], output_dir)

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()