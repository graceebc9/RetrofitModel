"""
Module: visualise_wall_results.py
Purpose: Visualise the CSV outputs from wall_improvement_sweep_v3.py
Fixes: Added .values to ax.plot calls to avoid pandas Multi-dimensional indexing errors.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Colors
PALETTE = {
    'solid_wall_internal': '#1f77b4',  # Blue
    'solid_wall_external': '#ff7f0e',  # Orange
    'cavity_wall': '#2ca02c',          # Green
}

# The thresholds drawn as horizontal lines
THRESHOLDS = [1000, 2000, 3000]

def parse_args():
    parser = argparse.ArgumentParser(description='Visualise Wall Sweep Results')
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Path to the sweep output directory (e.g. wall_sweep_results_v3/sweep_202X...)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots. Defaults to input-dir/plots')
    return parser.parse_args()

def load_data(input_dir):
    """Loads the various aggregated CSV files."""
    paths = {
        'main': os.path.join(input_dir, 'sweep_by_building_category.csv'),
        'gas': os.path.join(input_dir, 'category_x_gas_decile.csv'),
        'premise': os.path.join(input_dir, 'category_x_premise_type.csv')
    }
    
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure strictly numeric metric
            if 'median' in df.columns:
                df = df[df['median'] < 100000] # Remove extreme outliers/infinity
            data[key] = df
        else:
            print(f"Warning: Could not find {path}")
            data[key] = None
    return data

def plot_cost_efficiency_curve(df, output_path):
    """
    Plot 1: The main cost curve.
    Shows how Median £/tCO2 decreases as Improvement Factor increases.
    """
    if df is None: return

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Filter for the Internal Sweep (Solid Wall Internal)
    internal_data = df[
        (df['sweep_type'] == 'internal') & 
        (df['building_category'] == 'solid_wall_internal')
    ].sort_values('internal_factor')

    # 2. Filter for the External Sweep (Solid Wall External)
    external_data = df[
        (df['sweep_type'] == 'external') & 
        (df['building_category'] == 'solid_wall_external')
    ].sort_values('external_factor')

    # Plot Lines - USING .values TO FIX INDEXING ERROR
    if not internal_data.empty:
        ax.plot(internal_data['internal_factor'].values, internal_data['median'].values, 
                marker='o', label='Solid Wall (Internal Ins.)', color=PALETTE['solid_wall_internal'], linewidth=2.5)
    
    if not external_data.empty:
        ax.plot(external_data['external_factor'].values, external_data['median'].values, 
                marker='s', label='Solid Wall (External Ins.)', color=PALETTE['solid_wall_external'], linewidth=2.5)

    # Reference Lines (Thresholds)
    # Get min x for annotation positioning
    min_x = 0
    if not internal_data.empty:
        min_x = internal_data['internal_factor'].min()
    elif not external_data.empty:
        min_x = external_data['external_factor'].min()

    for thr in THRESHOLDS:
        ax.axhline(thr, color='gray', linestyle='--', alpha=0.5)
        ax.text(min_x, thr + 50, f'£{thr}/tCO2', color='gray', fontsize=9)

    # Formatting
    ax.set_title("Cost Efficiency: Improvement Factor vs Cost per tCO2 (5yr)", fontsize=16, pad=20)
    ax.set_xlabel("Improvement Factor (Multiplier on Savings)", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year Horizon)", fontsize=14)
    ax.set_ylim(0, 10000)  # Cap Y-axis to keep readable
    ax.legend()
    
    # Secondary X-Axis explanation
    plt.figtext(0.5, 0.01, "Factor 1.0 = Standard Physics. Factor 2.0 = 2x Savings.", ha="center", fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '1_cost_efficiency_curve.png'), dpi=300)
    plt.close()
    print("Generated Plot 1: Cost Efficiency Curve")

def plot_viability_percentage(df, output_path):
    """
    Plot 2: What % of stock becomes viable (<£2000/tCO2) as factor increases?
    """
    if df is None: return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    threshold_col = 'pct_below_2000' # Ensure this column matches your CSV
    if threshold_col not in df.columns:
        print(f"Skipping Plot 2: {threshold_col} not found in data")
        return

    # Internal Sweep
    int_data = df[(df['sweep_type'] == 'internal') & (df['building_category'] == 'solid_wall_internal')].sort_values('internal_factor')
    if not int_data.empty:
        ax.plot(int_data['internal_factor'].values, int_data[threshold_col].values, 
                marker='o', color=PALETTE['solid_wall_internal'], label='Solid Wall (Internal)')

    # External Sweep
    ext_data = df[(df['sweep_type'] == 'external') & (df['building_category'] == 'solid_wall_external')].sort_values('external_factor')
    if not ext_data.empty:
        ax.plot(ext_data['external_factor'].values, ext_data[threshold_col].values, 
                marker='s', color=PALETTE['solid_wall_external'], label='Solid Wall (External)')

    ax.set_title(f"Market Viability: % of Buildings < £2000/tCO2", fontsize=16)
    ax.set_xlabel("Improvement Factor", fontsize=14)
    ax.set_ylabel("% of Housing Stock Viable", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '2_viability_ramp.png'), dpi=300)
    plt.close()
    print("Generated Plot 2: Viability Ramp")

def plot_gas_stratification(df, output_path):
    """
    Plot 3: Efficiency by Gas Decile.
    Focuses on 'solid_wall_external' sweep to keep plot clean.
    """
    print('starting plot 3 .. ') 
    if df is None: 
        print('df isu noen' ) 
        return

    # Filter for External Insulation sweep only, looking at Solid Wall External buildings
    subset = df[
        (df['sweep_type'] == 'external') & 
        (df['building_category'] == 'solid_wall_external')
    ].copy()
    
    if subset.empty: 
        print('subset emtpy' ) 
        return
    
    print('startging plot"') 
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a sequential palette for gas deciles
    gas_palette = sns.color_palette("YlOrRd", n_colors=subset['gas_decile'].nunique())
    
    # seaborn handles pandas dataframes natively well, usually no need for .values here
    sns.lineplot(
        data=subset, 
        x='external_factor', 
        y='median', 
        hue='gas_decile', 
        palette=gas_palette,
        marker='o',
        linewidth=2,
        ax=ax
    )

    ax.set_title("Gas Impact: Solid Wall (External) Efficiency by Usage", fontsize=16)
    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)
    
    # Add annotation explaining the trend
    plt.figtext(0.5, 0.01, "High gas users (Darker Red) have better ROI (Lower £/tCO2)", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '3_gas_decile_impact.png'), dpi=300)
    plt.close()
    print("Generated Plot 3: Gas Stratification")

def plot_premise_stratification(df, output_path):
    """
    Plot 4: Efficiency by Premise Type.
    Focuses on 'solid_wall_internal' sweep.
    """
    if df is None: return

    # Filter for Internal Insulation sweep
    subset = df[
        (df['sweep_type'] == 'internal') & 
        (df['building_category'] == 'solid_wall_internal')
    ].copy()

    if subset.empty: return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Clean up premise type labels if needed
    subset['Premise Type'] = subset['premise_type_filled'].str.replace('_', ' ').str.title()

    sns.lineplot(
        data=subset,
        x='internal_factor',
        y='median',
        hue='Premise Type',
        marker='o',
        ax=ax
    )

    ax.set_title("Form Factor: Solid Wall (Internal) Efficiency by Type", fontsize=16)
    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '4_premise_type_impact.png'), dpi=300)
    plt.close()
    print("Generated Plot 4: Premise Type Stratification")

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

    # Generate Plots
    plot_cost_efficiency_curve(data['main'], output_dir)
    plot_viability_percentage(data['main'], output_dir)
    plot_gas_stratification(data['gas'], output_dir)
    plot_premise_stratification(data['premise'], output_dir)

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()