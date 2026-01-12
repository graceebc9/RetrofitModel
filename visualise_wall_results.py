"""
Module: visualise_wall_results.py
Purpose: Visualise the CSV outputs from wall_improvement_sweep_v3.py
Fixes:
1. Maps long category names to short keys.
2. Replaces sns.lineplot with manual ax.plot loops to allow np.array() wrapping.
   (This fixes the 'Multi-dimensional indexing' error completely).
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

THRESHOLDS = [1000, 2000, 3000]

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
    paths = {
        'main': os.path.join(input_dir, 'sweep_by_building_category.csv'),
        'gas': os.path.join(input_dir, 'category_x_gas_decile.csv'),
        'premise': os.path.join(input_dir, 'category_x_premise_type.csv')
    }
    
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[key] = clean_dataframe(df)
            print(f"Loaded {key}: {len(data[key])} rows")
        else:
            print(f"Warning: Could not find {path}")
            data[key] = None
    return data

def plot_cost_efficiency_curve(df, output_path):
    """Plot 1: Median Cost vs Improvement Factor."""
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
    ax.set_ylim(0, 10000)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '1_cost_efficiency_curve.png'), dpi=300)
    plt.close()
    print("Saved Plot 1")

def plot_viability_percentage(df, output_path):
    """Plot 2: % Viable (<£2000) vs Factor."""
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
    """Plot 3: Efficiency by Gas Decile (External only). MANUAL LOOP."""
    if df is None: return

    # Filter data
    subset = df[
        (df['sweep_type'] == 'external') & 
        (df['building_category'] == 'solid_wall_external')
    ].copy()

    if subset.empty: return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get unique deciles and sort them
    deciles = sorted(subset['gas_decile'].unique())
    
    # Create colors using matplotlib colormap
    colors = cm.YlOrRd(np.linspace(0.3, 1, len(deciles)))
    
    # Manual loop instead of sns.lineplot
    for i, decile in enumerate(deciles):
        group = subset[subset['gas_decile'] == decile].sort_values('external_factor')
        ax.plot(
            np.array(group['external_factor']), 
            np.array(group['median']), 
            marker='o', 
            label=decile, 
            color=colors[i], 
            linewidth=2
        )

    ax.set_title("Gas Impact: Solid Wall (External) Efficiency", fontsize=16)
    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)
    ax.legend(title="Gas Decile")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '3_gas_decile_impact.png'), dpi=300)
    plt.close()
    print("Saved Plot 3")

def plot_premise_stratification(df, output_path):
    """Plot 4: Efficiency by Premise Type (Internal only). MANUAL LOOP."""
    if df is None: return

    subset = df[
        (df['sweep_type'] == 'internal') & 
        (df['building_category'] == 'solid_wall_internal')
    ].copy()

    if subset.empty: return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    subset['Premise Type'] = subset['premise_type_filled'].str.replace('_', ' ').str.title()
    
    # Get unique premises
    premises = sorted(subset['Premise Type'].unique())
    
    # Manual loop
    for premise in premises:
        group = subset[subset['Premise Type'] == premise].sort_values('internal_factor')
        ax.plot(
            np.array(group['internal_factor']), 
            np.array(group['median']), 
            marker='o', 
            label=premise
        )

    ax.set_title("Form Factor: Solid Wall (Internal) Efficiency", fontsize=16)
    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)
    ax.legend(title="Premise Type")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '4_premise_type_impact.png'), dpi=300)
    plt.close()
    print("Saved Plot 4")

def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_data(args.input_dir)
    
    plot_cost_efficiency_curve(data['main'], output_dir)
    plot_viability_percentage(data['main'], output_dir)
    plot_gas_stratification(data['gas'], output_dir)
    plot_premise_stratification(data['premise'], output_dir)
    
    print("\nVisualization complete.")

if __name__ == "__main__":
    main()