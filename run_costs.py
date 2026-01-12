#!/usr/bin/env python3
# ==============================================================================
# Script Name: Retrofit Analysis & Plotting (Updated)
# Last Updated: 2026-01-12
# Description: Aggregates retrofit simulation data, generates summary plots
#              (Median, Variance, & Risk Analysis), and exports data tables.
# ==============================================================================

import sys
import pandas as pd
import numpy as np
import glob
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.ticker import FuncFormatter

# Try import, fallback to False if running locally without src module
try:
    from src.utils import is_running_on_hpc
except ImportError:
    def is_running_on_hpc(): return False

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- Plot Generation Toggles ---
PLOT_TOGGLES = {
    'Median_Summary': False,        # The original "Median of Medians" bar chart
    'Variance_Box': False,          # Box plot of Run Totals (P50 only)
    'Variance_Bar': False,          # Bar chart of Run Totals (Mean +/- Std)
    'Variance_Range': True,         # Bar chart of Run Totals (Mean with P5-P95 interval)
    'Risk_Comparison': True         # NEW: Compares Sum(P5) vs Sum(P50) vs Sum(P95)
}

# Timestamp for file naming
TODAY = datetime.datetime.now().strftime("%Y_%m_%d")

is_hpc = is_running_on_hpc()
is_epc = False     

OUTPUT_DIR = f'1_summary_results_{TODAY}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)




SCENARIOS = [
    'loft_installation',
    'wall_installation',
    'heat_pump_only',
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'join_heat_ins_decay',
]

SCENARIO_DISPLAY_NAMES = {
    'loft_installation': 'Loft Installation',
    'wall_installation': 'Wall Insulation',
    'heat_pump_only': 'Heat Pump Only',
    'joint_heat_loft_decay': 'HP + Loft (Decay)',
    'joint_heat_wall_decay': 'HP + Wall (Decay)',
    'join _heat_ins_decay': 'HP + All Insulation (Decay)',
   
}
# --- Path config ---
if is_hpc:
    if not is_epc:
        LOG_FILE_PATTERN = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/*csv'
    else:
        LOG_FILE_PATTERN = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/v9_logs_with_epc/*csv'
else: 
    if is_epc:
        LOG_FILE_PATTERN = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge/*csv'
    else:
        LOG_FILE_PATTERN = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*.csv'

# --- Metric Definitions ---
# 1. For Median Plots (Per building stats)
METRICS_INFO = {
    'Capex':  {'pattern': '{sc}_capex_per_net_ton_co2_{sc}_{stat}', 'ylabel': 'Capex (£/ton)'},
    'Energy': {'pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}', 'ylabel': 'Energy (Ton CO2/5yr)'},
    'Cost':   {'pattern': '{sc}_cost_{sc}_{stat}', 'ylabel': 'Cost (£)'}
}

# 2. For Variance/Risk Plots (Global Sums)
# Note: We now define the BASE pattern without the specific statistic (_p50)
VARIANCE_METRICS = {
    'Total_Cost': {
        'base_col_pattern': '{sc}_cost_{sc}', 
        'ylabel': 'Total Cost (£M)',
        'scale_factor': 1_000_000,
        'unit': 'M'
    },
    'Total_Carbon_Removed': {
        'base_col_pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}', 
        'ylabel': 'Total Carbon Removed (kTons)',
        'scale_factor': 1_000,
        'unit': 'k'
    }
}

# ==============================================================================
# 2. DATA COLLECTION
# ==============================================================================

def collect_data(file_pattern):
    """
    Reads CSVs and collects data. 
    Now collects P5, P50, and P95 sums for variance metrics.
    """
    # Standard Median Data Store
    summary_store = {
        metric: {sc: {'p5': [], 'p50': [], 'p95': []} for sc in SCENARIOS} 
        for metric in METRICS_INFO
    }

    # Variance Data Store (Run Totals)
    # Structure: {'Total_Cost': {'loft': {'p5': [run0_sum, run1_sum...], 'p50': [...], 'p95': [...]}}}
    variance_records = {
        v_name: {sc: {'p5': [], 'p50': [], 'p95': []} for sc in SCENARIOS}
        for v_name in VARIANCE_METRICS
    }
    
    # Track Run IDs to ensure we group correctly
    # We will store: (run_id, p5_sum, p50_sum, p95_sum) then aggregate later
    temp_variance_data = {
        v_name: [] for v_name in VARIANCE_METRICS
    }

    log_files = glob.glob(file_pattern)
    
    if not log_files:
        print(f"Error: No files found matching pattern: {file_pattern}")
        sys.exit(1)

    print(f"Found {len(log_files)} files. Collecting data...")
 
    for i, file_path in enumerate(log_files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(log_files)}...")

        try:
            chunk = pd.read_csv(file_path)
            
            # --- A. Standard Median Plot Data (Building Level Distribution) ---
            if PLOT_TOGGLES['Median_Summary']:
                for metric_name, info in METRICS_INFO.items():
                    for sc in SCENARIOS:
                        c_p5  = info['pattern'].format(sc=sc, stat='p5')
                        c_p50 = info['pattern'].format(sc=sc, stat='p50')
                        c_p95 = info['pattern'].format(sc=sc, stat='p95')
                        
                        if c_p50 in chunk.columns:
                            # Loose validity check: if P50 exists, we take what we can get
                            # (Strict check removed to avoid dropping P50 if P5 missing)
                            valid = chunk.dropna(subset=[c_p50])
                            if not valid.empty:
                                summary_store[metric_name][sc]['p50'].extend(valid[c_p50].tolist())
                                if c_p5 in valid.columns:
                                    summary_store[metric_name][sc]['p5'].extend(valid[c_p5].tolist())
                                if c_p95 in valid.columns:
                                    summary_store[metric_name][sc]['p95'].extend(valid[c_p95].tolist())

            # --- B. Variance & Risk Data (Run Totals) ---
            if any([PLOT_TOGGLES['Variance_Box'], PLOT_TOGGLES['Variance_Bar'], 
                    PLOT_TOGGLES['Variance_Range'], PLOT_TOGGLES['Risk_Comparison']]):
                
                if 'epistemic_run_id' in chunk.columns:
                    for v_name, v_info in VARIANCE_METRICS.items():
                        for sc in SCENARIOS:
                            # Construct column names
                            col_base = v_info['base_col_pattern'].format(sc=sc)
                            col_p5   = f"{col_base}_p5"
                            col_p50  = f"{col_base}_p50"
                            col_p95  = f"{col_base}_p95"
                            
                            cols_to_fetch = []
                            if col_p5 in chunk.columns: cols_to_fetch.append(col_p5)
                            if col_p50 in chunk.columns: cols_to_fetch.append(col_p50)
                            if col_p95 in chunk.columns: cols_to_fetch.append(col_p95)
                            
                            if not cols_to_fetch:
                                continue

                            # Group by run_id within this file chunk
                            cols_with_id = ['epistemic_run_id'] + cols_to_fetch
                            subset = chunk[cols_with_id].dropna(subset=['epistemic_run_id'])
                            
                            if not subset.empty:
                                partial_sums = subset.groupby('epistemic_run_id')[cols_to_fetch].sum().reset_index()
                                
                                for _, row in partial_sums.iterrows():
                                    record = {
                                        'scenario': sc,
                                        'epistemic_run_id': row['epistemic_run_id'],
                                    }
                                    if col_p5 in row: record['val_p5'] = row[col_p5]
                                    if col_p50 in row: record['val_p50'] = row[col_p50]
                                    if col_p95 in row: record['val_p95'] = row[col_p95]
                                    
                                    temp_variance_data[v_name].append(record)
            
            del chunk
            
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    gc.collect()
    
    # --- Final Aggregation of Variance Data ---
    # We need to sum across files because one run_id might be split across files (if not unique)
    # BUT normally run_ids are unique sets. Assuming run_ids are globally unique per file or we want to aggregate them.
    # The current logic assumes partial sums need to be added together for the same run_id.
    
    final_variance_records = {}
    
    print("Aggregating partial sums for variance plots...")
    for v_name, records in temp_variance_data.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        
        # Summing partials for the same run_id and scenario
        # This handles if buildings for Run 0 are split across multiple CSVs
        agg_cols = {}
        if 'val_p5' in df.columns: agg_cols['val_p5'] = 'sum'
        if 'val_p50' in df.columns: agg_cols['val_p50'] = 'sum'
        if 'val_p95' in df.columns: agg_cols['val_p95'] = 'sum'
        
        if not agg_cols:
            continue
            
        df_agg = df.groupby(['scenario', 'epistemic_run_id']).agg(agg_cols).reset_index()
        final_variance_records[v_name] = df_agg

    return summary_store, final_variance_records

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def generate_median_plots(data_store, output_dir):
    print(f"Generating Median Summary plots...")
    for metric_name, info in METRICS_INFO.items():
        bar_heights = []
        lower_errors, upper_errors = [], []
        valid_scenarios = []
        table_rows = []
        
        for sc in SCENARIOS:
            d = data_store[metric_name][sc]
            if len(d['p50']) > 0:
                med_p50 = np.median(d['p50'])
                # If P5/P95 missing, just use P50 (no error bar)
                med_p5  = np.median(d['p5']) if d['p5'] else med_p50
                med_p95 = np.median(d['p95']) if d['p95'] else med_p50
                
                bar_heights.append(med_p50)
                lower_errors.append(med_p50 - med_p5)
                upper_errors.append(med_p95 - med_p50)
                valid_scenarios.append(sc)
                table_rows.append({
                    'Scenario': SCENARIO_DISPLAY_NAMES.get(sc, sc),
                    'Median_P5': med_p5, 'Median_P50': med_p50, 'Median_P95': med_p95
                })

        if valid_scenarios:
            pd.DataFrame(table_rows).to_csv(os.path.join(output_dir, f"{metric_name}_median_summary.csv"), index=False)
            plt.figure(figsize=(10, 6))
            plt.bar(valid_scenarios, bar_heights, yerr=[lower_errors, upper_errors], 
                    capsize=5, color='cornflowerblue', alpha=0.8, edgecolor='black')
            plt.ylabel(info['ylabel'], fontweight='bold')
            clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in valid_scenarios]
            plt.xticks(ticks=range(len(valid_scenarios)), labels=clean_labels, rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric_name}_median_plot.png"), dpi=300)
            plt.close()

def generate_variance_range_plots_range(df, v_name, output_dir):
    """
    Standard Variance Plot: Uses P50 data only.
    Shows the distribution of the portfolio Sum(P50) across runs.
    """
    print(f"Generating Variance Range plots (P50 Only)...")
    info = VARIANCE_METRICS[v_name]
    
    if 'val_p50' not in df.columns:
        print(f"Skipping {v_name} range plot: p50 data missing.")
        return

    # Unit Conversion
    vals = df['val_p50'] / info['scale_factor']
    
    # Stats per scenario
    summary = df.copy()
    summary['scaled_val'] = vals
    stats = summary.groupby('scenario')['scaled_val'].agg(
        mean='mean',
        p5=lambda x: np.percentile(x, 5),
        p95=lambda x: np.percentile(x, 95)
    ).reindex(SCENARIOS).dropna()

    if stats.empty: return

    # Save Stats
    stats.to_csv(os.path.join(output_dir, f"Variance_Range_P50_{v_name}.csv"))

    # Plot
    lower = stats['mean'] - stats['p5']
    upper = stats['p95'] - stats['mean']
    
    plt.figure(figsize=(10, 6))
    plt.bar(stats.index, stats['mean'], yerr=[lower, upper], 
            capsize=5, color='mediumseagreen', alpha=0.7, edgecolor='black',
            error_kw={'elinewidth': 1.5})
    
    plt.ylabel(info['ylabel'], fontweight='bold')
    clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in stats.index]
    plt.xticks(ticks=range(len(stats.index)), labels=clean_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Variance_Runs_Range_P50_{v_name}.png"), dpi=300)
    plt.close()

def generate_risk_comparison_plots(df, v_name, output_dir):
    """
    NEW: Compares the Portfolio Totals using building-level P5 vs P50 vs P95.
    Groups bars by Scenario, showing 3 bars per scenario.
    """
    print(f"Generating Risk Comparison plots (P5 vs P50 vs P95)...")
    info = VARIANCE_METRICS[v_name]
    
    # Check available columns
    avail_metrics = []
    if 'val_p5' in df.columns: avail_metrics.append('val_p5')
    if 'val_p50' in df.columns: avail_metrics.append('val_p50')
    if 'val_p95' in df.columns: avail_metrics.append('val_p95')
    
    if not avail_metrics:
        return

    # 1. Calculate the Mean Total across runs for each metric type
    # (We average out the epistemic noise to get the stable Portfolio Risk Profile)
    agg_dict = {col: 'mean' for col in avail_metrics}
    scenario_means = df.groupby('scenario').agg(agg_dict).reindex(SCENARIOS).dropna()
    
    # Unit Conversion
    scenario_means = scenario_means / info['scale_factor']
    
    # Save Table
    csv_name = f"Risk_Comparison_{v_name}.csv"
    scenario_means.to_csv(os.path.join(output_dir, csv_name))
    
    # 2. Plotting Grouped Bar Chart
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(scenario_means))
    width = 0.25
    
    # Mapping for colors and labels
    metric_map = {
        'val_p5':  {'label': 'Sum P5',  'color': '#2ca02c', 'offset': -width},
        'val_p50': {'label': 'Sum P50',    'color': '#1f77b4', 'offset': 0},
        'val_p95': {'label': 'Sum P95', 'color': '#d62728', 'offset': width}
    }
    
    for col in avail_metrics:
        props = metric_map[col]
        plt.bar(x + props['offset'], scenario_means[col], width, 
                label=props['label'], color=props['color'], alpha=0.8, edgecolor='black')

    plt.ylabel(info['ylabel'], fontweight='bold')
    plt.xlabel('Scenario', fontweight='bold')
    plt.legend(title="Building-Level Assumption")
    
    clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in scenario_means.index]
    plt.xticks(x, clean_labels, rotation=45, ha='right')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"Risk_Comparison_Bar_{v_name}.png"), dpi=300)
    plt.close()

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    print(f"========================================")
    print(f"Retrofit Analysis - {TODAY}")
    print(f"========================================")
    print(f"Input Pattern: {LOG_FILE_PATTERN}")
    print(f"Output Dir:    {OUTPUT_DIR}")
    print(f"Toggles:       {PLOT_TOGGLES}")
    print(f"----------------------------------------")

    summary_data, variance_data = collect_data(LOG_FILE_PATTERN)
    
    # Generate requested plots
    if PLOT_TOGGLES['Median_Summary']:
        generate_median_plots(summary_data, OUTPUT_DIR)
        
    for v_name, df_agg in variance_data.items():
        if df_agg.empty: continue
        
        if PLOT_TOGGLES['Variance_Range']:
            generate_variance_range_plots_range(df_agg, v_name, OUTPUT_DIR)
            
        if PLOT_TOGGLES['Risk_Comparison']:
            generate_risk_comparison_plots(df_agg, v_name, OUTPUT_DIR)
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()