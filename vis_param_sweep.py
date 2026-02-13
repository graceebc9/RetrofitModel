"""
Module: vis_param_sweep.py
Purpose: Visualise the CSV outputs from combine_sweep_results.py / param_sweep.py

Plots generated:
  1  - Cost efficiency curve (median only)
  1b - Cost efficiency curve with confidence bands (median_optimistic / median / median_pessimistic)
  2  - Viability % ramp with 3 confidence lines (optimistic / central / pessimistic)
  3  - Gas decile impact (median only)
  3b - Gas decile impact with confidence bands
  4  - Premise type impact (median only)
  4b - Premise type impact with confidence bands
  5_<premise>_internal - Per-premise gas decile breakdown (internal sweep)
  5_<premise>_external - Per-premise gas decile breakdown (external sweep)

python vis_param_sweep.py --input-dir <output_dir> --output-dir <vis_dir>

python vis_param_sweep.py --input-dir /home/gb669/rds/hpc-work/energy_map/RetrofitModel/wall_param_sweep_v10_n50_p10_v3/output --output-dir /home/gb669/rds/hpc-work/energy_map/RetrofitModel/wall_param_sweep_v10_n50_p10_v3/vis
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

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

PALETTE = {
    'solid_wall_internal': '#1f77b4',
    'solid_wall_external': '#ff7f0e',
    'cavity_wall': '#2ca02c',
}

THRESHOLDS = [1000, 2000, 3000]

CONFIDENCE_STYLES = {
    'optimistic':  {'linestyle': '--', 'alpha': 0.6, 'label_suffix': '(μ-σ optimistic)'},
    'central':     {'linestyle': '-',  'alpha': 1.0, 'label_suffix': '(μ central)'},
    'pessimistic': {'linestyle': ':',  'alpha': 0.6, 'label_suffix': '(μ+σ pessimistic)'},
}

VIABILITY_COLS = {
    'optimistic':  'optimistic_pct_below_2000',
    'central':     'central_pct_below_2000',
    'pessimistic': 'pessimistic_pct_below_2000',
}

COST_MEDIAN_COLS = {
    'optimistic':  'median_optimistic',
    'central':     'median',
    'pessimistic': 'median_pessimistic',
}

MIN_SAMPLE_SIZE = 10


# ==========================================
# HELPERS
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description='Visualise Wall Sweep Results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the sweep output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots. Defaults to input-dir/plots')
    parser.add_argument('--min-sample', type=int, default=MIN_SAMPLE_SIZE,
                        help=f'Min sample size for sub-group plots (default: {MIN_SAMPLE_SIZE})')
    return parser.parse_args()


def load_data(input_dir):
    """Loads the various aggregated CSV files."""
    paths = {
        'main': os.path.join(input_dir, 'sweep_by_building_category.csv'),
        'gas': os.path.join(input_dir, 'category_x_gas_decile.csv'),
        'premise': os.path.join(input_dir, 'category_x_premise_type.csv'),
        'premise_gas': os.path.join(input_dir, 'premise_x_gas_decile.csv'),
    }
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'median' in df.columns:
                df = df[df['median'] < 100000]
            data[key] = df
        else:
            print(f"Warning: Could not find {path}")
            data[key] = None
    return data


def _add_threshold_lines(ax, min_x):
    for thr in THRESHOLDS:
        ax.axhline(thr, color='gray', linestyle='--', alpha=0.5)
        ax.text(min_x, thr + 50, f'£{thr}/tCO2', color='gray', fontsize=9)


def _get_min_x(int_data, ext_data):
    min_x = 0
    if int_data is not None and not int_data.empty:
        min_x = int_data['internal_factor'].min()
    elif ext_data is not None and not ext_data.empty:
        min_x = ext_data['external_factor'].min()
    return min_x


def _clean_premise_name(name):
    if pd.isna(name):
        return 'Unknown'
    return str(name).replace('_', ' ').title()


def _safe_filename(name):
    return str(name).lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')


def _has_confidence_cols(df):
    """Check if df has the median_optimistic/pessimistic columns."""
    return df is not None and 'median_optimistic' in df.columns and 'median_pessimistic' in df.columns


# ==========================================
# PLOT 1: Cost Efficiency Curve (median only)
# ==========================================

def plot_cost_efficiency_curve(df, output_path):
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    int_data = df[
        (df['sweep_type'] == 'internal') &
        (df['building_category'] == 'solid_wall_internal')
    ].sort_values('internal_factor')

    ext_data = df[
        (df['sweep_type'] == 'external') &
        (df['building_category'] == 'solid_wall_external')
    ].sort_values('external_factor')

    if not int_data.empty:
        ax.plot(int_data['internal_factor'].values, int_data['median'].values,
                marker='o', label='Solid Wall (Internal Ins.)',
                color=PALETTE['solid_wall_internal'], linewidth=2.5)
    if not ext_data.empty:
        ax.plot(ext_data['external_factor'].values, ext_data['median'].values,
                marker='s', label='Solid Wall (External Ins.)',
                color=PALETTE['solid_wall_external'], linewidth=2.5)

    _add_threshold_lines(ax, _get_min_x(int_data, ext_data))

    
    ax.set_xlabel("Improvement Factor (Multiplier on Savings)", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year Horizon)", fontsize=14)
    ax.set_ylim(0, 10000)
    ax.legend()
 

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '1_cost_efficiency_curve.png'), dpi=300)
    plt.close()
    print("Generated Plot 1: Cost Efficiency Curve")


# ==========================================
# PLOT 1b: Cost Efficiency with Confidence
# ==========================================

def plot_cost_efficiency_confidence(df, output_path):
    if not _has_confidence_cols(df):
        print("Skipping Plot 1b: median_optimistic/median_pessimistic not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, sweep_type, cat, factor_col, marker in [
        (axes[0], 'internal', 'solid_wall_internal', 'internal_factor', 'o'),
        (axes[1], 'external', 'solid_wall_external', 'external_factor', 's'),
    ]:
        data = df[
            (df['sweep_type'] == sweep_type) &
            (df['building_category'] == cat)
        ].sort_values(factor_col)

        if data.empty:
            continue

        color = PALETTE[cat]
        x = data[factor_col].values

        for level, col in COST_MEDIAN_COLS.items():
            if col not in data.columns:
                continue
            style = CONFIDENCE_STYLES[level]
            ax.plot(x, data[col].values, marker=marker, markersize=4,
                    color=color, linestyle=style['linestyle'],
                    alpha=style['alpha'], linewidth=2, label=style['label_suffix'])

        ax.fill_between(x, data['median_optimistic'].values,
                         data['median_pessimistic'].values,
                         color=color, alpha=0.12)

        _add_threshold_lines(ax, x.min())
        label = 'Internal' if sweep_type == 'internal' else 'External'
        ax.set_title(f"{label} Insulation", fontsize=14)
        ax.set_xlabel(f"{label} Improvement Factor", fontsize=12)
        ax.set_ylim(0, 10000)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Median £ / tCO2 (5-Year)", fontsize=12)
 
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '1b_cost_efficiency_confidence.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Plot 1b: Cost Efficiency with Confidence Bands")


# ==========================================
# PLOT 2: Viability Percentage (3 lines)
# ==========================================

def plot_viability_percentage(df, output_path):
    if df is None:
        return

    available = {k: v for k, v in VIABILITY_COLS.items() if v in df.columns}
    if not available:
        print(f"Skipping Plot 2: none of {list(VIABILITY_COLS.values())} found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, sweep_type, cat, factor_col, marker in [
        (axes[0], 'internal', 'solid_wall_internal', 'internal_factor', 'o'),
        (axes[1], 'external', 'solid_wall_external', 'external_factor', 's'),
    ]:
        data = df[
            (df['sweep_type'] == sweep_type) &
            (df['building_category'] == cat)
        ].sort_values(factor_col)

        if data.empty:
            continue

        color = PALETTE[cat]
        for level_key, col_name in available.items():
            style = CONFIDENCE_STYLES[level_key]
            ax.plot(data[factor_col].values, data[col_name].values,
                    marker=marker, markersize=4, color=color,
                    linestyle=style['linestyle'], alpha=style['alpha'],
                    linewidth=2, label=style['label_suffix'])

        
        label = 'Internal' if sweep_type == 'internal' else 'External'

        
        ax.set_title(f"{label} Insulation", fontsize=14)
        ax.set_xlabel(f"{label} Improvement Factor", fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("% of Housing Stock < £2000/tCO2", fontsize=12)

    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '2_viability_ramp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Plot 2: Viability Ramp (optimistic / central / pessimistic)")


# ==========================================
# PLOT 3: Gas Stratification (median only)
# ==========================================

def plot_gas_stratification(df, output_path):
    if df is None:
        return

    subset = df[
        (df['sweep_type'] == 'external') &
        (df['building_category'] == 'solid_wall_external')
    ].copy()
    if subset.empty:
        print("Skipping Plot 3: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    gas_palette = sns.color_palette("YlOrRd", n_colors=subset['gas_decile'].nunique())
    sns.lineplot(data=subset, x='external_factor', y='median',
                 hue='gas_decile', palette=gas_palette, marker='o', linewidth=2, ax=ax)

    
    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)
    plt.figtext(0.5, 0.01, "High gas users (Darker Red) have better ROI (Lower £/tCO2)",
                ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '3_gas_decile_impact.png'), dpi=300)
    plt.close()
    print("Generated Plot 3: Gas Stratification")


# ==========================================
# PLOT 3b: Gas Stratification with Confidence
# ==========================================

def plot_gas_stratification_confidence(df, output_path):
    if not _has_confidence_cols(df):
        print("Skipping Plot 3b: median_optimistic/median_pessimistic not found")
        return

    subset = df[
        (df['sweep_type'] == 'external') &
        (df['building_category'] == 'solid_wall_external')
    ].copy()
    if subset.empty:
        print("Skipping Plot 3b: no data")
        return

    deciles = sorted(subset['gas_decile'].unique())
    colors = sns.color_palette("YlOrRd", n_colors=len(deciles))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, decile in enumerate(deciles):
        dec_data = subset[subset['gas_decile'] == decile].sort_values('external_factor')
        if dec_data.empty:
            continue
        x = dec_data['external_factor'].values
        color = colors[i]
        ax.plot(x, dec_data['median'].values, marker='o', markersize=3,
                color=color, linestyle='-', linewidth=1.8, label=f'Decile {decile}')
        ax.fill_between(x, dec_data['median_optimistic'].values,
                         dec_data['median_pessimistic'].values, color=color, alpha=0.10)

    ax.set_title("Gas Impact with Uncertainty: Solid Wall (External)", fontsize=16)
    ax.set_xlabel("External Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)
    ax.legend(fontsize=9, ncol=2)
 

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '3b_gas_decile_confidence.png'), dpi=300)
    plt.close()
    print("Generated Plot 3b: Gas Stratification with Confidence")


# ==========================================
# PLOT 4: Premise Type (median only)
# ==========================================

def plot_premise_stratification(df, output_path):
    if df is None:
        return

    subset = df[
        (df['sweep_type'] == 'internal') &
        (df['building_category'] == 'solid_wall_internal')
    ].copy()
    if subset.empty:
        return

    subset['Premise Type'] = subset['premise_type_filled'].apply(_clean_premise_name)

    fig, ax = plt.subplots(figsize=(10, 7))
        
    sns.lineplot(data=subset, x='internal_factor', y='median',
                 hue='Premise Type', marker='o', ax=ax)

    _add_threshold_lines(ax, subset['internal_factor'].min())
    ax.set_title("Internal", fontsize=16)
    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '4_premise_type_impact.png'), dpi=300)
    plt.close()
    print("Generated Plot 4: Premise Type Stratification")


# ==========================================
# PLOT 4b: Premise Type with Confidence
# ==========================================

def plot_premise_stratification_confidence(df, output_path):
    if not _has_confidence_cols(df):
        print("Skipping Plot 4b: median_optimistic/median_pessimistic not found")
        return

    subset = df[
        (df['sweep_type'] == 'internal') &
        (df['building_category'] == 'solid_wall_internal')
    ].copy()
    if subset.empty:
        return

    subset['Premise Type'] = subset['premise_type_filled'].apply(_clean_premise_name)
    premise_types = sorted(subset['Premise Type'].unique())
    colors = sns.color_palette("tab10", n_colors=len(premise_types))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, ptype in enumerate(premise_types):
        pt_data = subset[subset['Premise Type'] == ptype].sort_values('internal_factor')
        if pt_data.empty:
            continue
        x = pt_data['internal_factor'].values
        color = colors[i]
        ax.plot(x, pt_data['median'].values, marker='o', markersize=3,
                color=color, linestyle='-', linewidth=1.8, label=ptype)
        ax.fill_between(x, pt_data['median_optimistic'].values,
                         pt_data['median_pessimistic'].values, color=color, alpha=0.10)

     

    _add_threshold_lines(ax, subset['internal_factor'].min())

    ax.set_title("Form Factor with Uncertainty: Solid Wall (Internal)", fontsize=16)
    ax.set_xlabel("Internal Wall Improvement Factor", fontsize=14)
    ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=14)
    ax.set_ylim(0, 8000)
    ax.legend(fontsize=9, ncol=2)
   

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, '4b_premise_type_confidence.png'), dpi=300)
    plt.close()
    print("Generated Plot 4b: Premise Type with Confidence")


# ==========================================
# PLOT 5: Per-Premise Gas Decile Breakdown
# ==========================================

def plot_per_premise_gas_decile(df, output_path, min_sample=MIN_SAMPLE_SIZE):
    """Generate one figure per premise type showing gas decile breakdown.
    
    For each premise type, plots median cost vs factor for each gas decile,
    for both internal and external sweeps.
    """
    if df is None:
        print("Skipping Plot 5: premise_x_gas_decile data not available")
        return

    has_confidence = _has_confidence_cols(df)

    # Get all premise types present
    premise_types = df['premise_type_filled'].dropna().unique()
    print(f"Generating per-premise gas decile plots for {len(premise_types)} premise types...")

    sweep_configs = [
        ('internal', 'solid_wall_internal', 'internal_factor'),
        ('external', 'solid_wall_external', 'external_factor'),
    ]

    for ptype in sorted(premise_types):
        clean_name = _clean_premise_name(ptype)
        safe_name = _safe_filename(ptype)

        for sweep_type, cat, factor_col in sweep_configs:
            subset = df[
                (df['sweep_type'] == sweep_type) &
                (df['building_category'] == cat) &
                (df['premise_type_filled'] == ptype)
            ].copy()

            if subset.empty:
                continue

            # Filter out small sample sizes
            subset = subset[subset['n'] >= min_sample]
            if subset.empty:
                continue

            deciles = sorted(subset['gas_decile'].unique())
            if len(deciles) < 2:
                continue

            colors = sns.color_palette("YlOrRd", n_colors=len(deciles))

            fig, ax = plt.subplots(figsize=(10, 7))

            for i, decile in enumerate(deciles):
                dec_data = subset[subset['gas_decile'] == decile].sort_values(factor_col)
                if dec_data.empty:
                    continue

                x = dec_data[factor_col].values
                color = colors[i]

                ax.plot(x, dec_data['median'].values, marker='o', markersize=3,
                        color=color, linestyle='-', linewidth=1.8,
                        label=f'Gas decile {decile}')

                if has_confidence:
                    ax.fill_between(x,
                                     dec_data['median_optimistic'].values,
                                     dec_data['median_pessimistic'].values,
                                     color=color, alpha=0.10)

            label = 'Internal' if sweep_type == 'internal' else 'External'
                  
            _add_threshold_lines(ax, subset[factor_col].min())
            ax.set_title(f"{clean_name}", fontsize=14)
            ax.set_xlabel(f"{label} Wall Improvement Factor", fontsize=12)
            ax.set_ylabel("Median £ / tCO2 (5-Year)", fontsize=12)
            ax.set_ylim(0, 8000)
            ax.legend(fontsize=9, ncol=2)

 

            plt.tight_layout()
            fname = f'5_{safe_name}_{sweep_type}.png'
            plt.savefig(os.path.join(output_path, fname), dpi=300)
            plt.close()
            print(f"  Generated: {fname}")

    print("Generated Plot 5: Per-Premise Gas Decile Breakdown")


# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_args()

    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading from: {args.input_dir}")
    print(f"Saving plots to: {output_dir}")

    data = load_data(args.input_dir)

    # Plot 1: Cost efficiency (median only)
    plot_cost_efficiency_curve(data['main'], output_dir)

    # Plot 1b: Cost efficiency with confidence bands
    plot_cost_efficiency_confidence(data['main'], output_dir)

    # Plot 2: Viability % (3 confidence lines)
    plot_viability_percentage(data['main'], output_dir)

    # Plot 3: Gas stratification (median only)
    plot_gas_stratification(data['gas'], output_dir)

    # Plot 3b: Gas stratification with confidence
    plot_gas_stratification_confidence(data['gas'], output_dir)

    # Plot 4: Premise type (median only)
    plot_premise_stratification(data['premise'], output_dir)

    # Plot 4b: Premise type with confidence
    plot_premise_stratification_confidence(data['premise'], output_dir)

    # Plot 5: Per-premise x gas decile
    plot_per_premise_gas_decile(data['premise_gas'], output_dir,
                                min_sample=args.min_sample)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()