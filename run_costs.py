#!/usr/bin/env python3
# ==============================================================================
# Script Name: Retrofit Analysis & Plotting (Updated with Risk Error Bars)
# Last Updated: 2026-01-12
# Description: Aggregates retrofit simulation data. 
#              Updated to show Epistemic Error Bars on the Risk Comparison Plots.
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
    'Median_Summary': True,        # The original "Median of Medians" bar chart
    'Variance_Box': True,          # Box plot of Run Totals (P50 only)
    'Variance_Bar': True,          # Bar chart of Run Totals (Mean +/- Std)
    'Variance_Range': True,         # Bar chart of Run Totals (Mean with P5-P95 interval)
    'Risk_Comparison': True         # Compares Sum(P5) vs Sum(P50) vs Sum(P95) with Error Bars
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
    'joint_heat_ins_decay',
]

SCENARIO_DISPLAY_NAMES = {
    'loft_installation': 'Loft Installation',
    'wall_installation': 'Wall Insulation',
    'heat_pump_only': 'Heat Pump Only',
    'joint_heat_loft_decay': 'HP + Loft (Decay)',
    'joint_heat_wall_decay': 'HP + Wall (Decay)',
    'join_heat_ins_decay': 'HP + All Insulation (Decay)',
    
}

# --- Path config ---
if is_hpc:
    if not is_epc:
        LOG_FILE_PATTERN = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE/*csv'
    else:
        LOG_FILE_PATTERN = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/v10_logs_with_epc/*csv'
else: 
    if is_epc:
        LOG_FILE_PATTERN = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge/*csv'
    else:
        LOG_FILE_PATTERN = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*.csv'

# --- Metric Definitions ---
METRICS_INFO = {
    'Capex':  {'pattern': '{sc}_capex_per_net_ton_co2_{sc}_{stat}', 'ylabel': 'Capex (£/ton)'},
    'Energy': {'pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}', 'ylabel': 'Energy (Ton CO2/5yr)'},
    'Cost':   {'pattern': '{sc}_cost_{sc}_{stat}', 'ylabel': 'Cost (£)'}
}

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
    summary_store = {
        metric: {sc: {'p5': [], 'p50': [], 'p95': []} for sc in SCENARIOS} 
        for metric in METRICS_INFO
    }

    # Temporary store for partial sums: {v_name: [ {scenario, run_id, val_p5, val_p50...} ]}
    temp_variance_data = {v_name: [] for v_name in VARIANCE_METRICS}

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
            
            # --- A. Median Plots Data ---
            if PLOT_TOGGLES['Median_Summary']:
                for metric_name, info in METRICS_INFO.items():
                    for sc in SCENARIOS:
                        c_p50 = info['pattern'].format(sc=sc, stat='p50')
                        c_p5  = info['pattern'].format(sc=sc, stat='p5')
                        c_p95 = info['pattern'].format(sc=sc, stat='p95')
                        
                        if c_p50 in chunk.columns:
                            valid = chunk.dropna(subset=[c_p50])
                            if not valid.empty:
                                summary_store[metric_name][sc]['p50'].extend(valid[c_p50].tolist())
                                if c_p5 in valid.columns:
                                    summary_store[metric_name][sc]['p5'].extend(valid[c_p5].tolist())
                                if c_p95 in valid.columns:
                                    summary_store[metric_name][sc]['p95'].extend(valid[c_p95].tolist())

            # --- B. Variance & Risk Data ---
            if any([PLOT_TOGGLES['Variance_Range'], PLOT_TOGGLES['Risk_Comparison']]):
                if 'epistemic_run_id' in chunk.columns:
                    for v_name, v_info in VARIANCE_METRICS.items():
                        for sc in SCENARIOS:
                            col_base = v_info['base_col_pattern'].format(sc=sc)
                            col_p5   = f"{col_base}_p5"
                            col_p50  = f"{col_base}_p50"
                            col_p95  = f"{col_base}_p95"
                            
                            cols_to_fetch = []
                            if col_p5 in chunk.columns: cols_to_fetch.append(col_p5)
                            if col_p50 in chunk.columns: cols_to_fetch.append(col_p50)
                            if col_p95 in chunk.columns: cols_to_fetch.append(col_p95)
                            
                            if not cols_to_fetch: continue

                            cols_with_id = ['epistemic_run_id'] + cols_to_fetch
                            subset = chunk[cols_with_id].dropna(subset=['epistemic_run_id'])
                            
                            if not subset.empty:
                                partial_sums = subset.groupby('epistemic_run_id')[cols_to_fetch].sum().reset_index()
                                for _, row in partial_sums.iterrows():
                                    record = {'scenario': sc, 'epistemic_run_id': row['epistemic_run_id']}
                                    if col_p5 in row: record['val_p5'] = row[col_p5]
                                    if col_p50 in row: record['val_p50'] = row[col_p50]
                                    if col_p95 in row: record['val_p95'] = row[col_p95]
                                    temp_variance_data[v_name].append(record)
            del chunk
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    gc.collect()
    
    # --- Final Aggregation ---
    final_variance_records = {}
    print("Aggregating partial sums for variance plots...")
    for v_name, records in temp_variance_data.items():
        if not records: continue
        df = pd.DataFrame(records)
        agg_cols = {}
        if 'val_p5' in df.columns: agg_cols['val_p5'] = 'sum'
        if 'val_p50' in df.columns: agg_cols['val_p50'] = 'sum'
        if 'val_p95' in df.columns: agg_cols['val_p95'] = 'sum'
        
        if agg_cols:
            df_agg = df.groupby(['scenario', 'epistemic_run_id']).agg(agg_cols).reset_index()
            final_variance_records[v_name] = df_agg

    return summary_store, final_variance_records

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def generate_median_plots(data_store, output_dir):
    """
    Generates Median of Medians plots AND saves data tables.
    """
    print(f"Generating Median Summary plots and tables...")
    
    for metric_name, info in METRICS_INFO.items():
        bar_heights = []
        lower_errors = []
        upper_errors = []
        valid_scenarios = []
        
        table_rows = []

        for sc in SCENARIOS:
            d = data_store[metric_name][sc]
            if len(d['p50']) > 0:
                med_p5  = np.median(d['p5'])
                med_p50 = np.median(d['p50'])
                med_p95 = np.median(d['p95'])



                bar_heights.append(med_p50)
                # Error bars are relative to the median for matplotlib
                lower_errors.append(med_p50 - med_p5)
                upper_errors.append(med_p95 - med_p50)
                valid_scenarios.append(sc)

                # Store data for table
                table_rows.append({
                    'Scenario_Name': SCENARIO_DISPLAY_NAMES.get(sc, sc),
                    'Scenario_ID': sc,
                    'Metric': metric_name,
                    'Median_Lower_P5': med_p5,
                    'Median_Central_P50': med_p50,
                    'Median_Upper_P95': med_p95,
                    'Sample_Count': len(d['p50'])
                })

        # --- Save Table ---
        if table_rows:
            df_table = pd.DataFrame(table_rows)
            csv_name = f"{metric_name}_median_summary.csv"
            df_table.to_csv(os.path.join(output_dir, csv_name), index=False)
            print(f"Saved Table: {csv_name}")

        # --- Plotting ---
        if valid_scenarios:

            plt.figure(figsize=(10, 6))
            asymmetric_err = [lower_errors, upper_errors]
            plt.bar(valid_scenarios, bar_heights, yerr=asymmetric_err, 
                    capsize=5, color='cornflowerblue', alpha=0.8, edgecolor='black')
            
            plt.ylabel(info['ylabel'], fontweight='bold')
            plt.xlabel('Scenario', fontweight='bold')
            # plt.title Removed
            
            clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in valid_scenarios]
            plt.xticks(ticks=range(len(valid_scenarios)), labels=clean_labels, rotation=45, ha='right')
            
            ax = plt.gca()
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
            
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"{metric_name}_median_plot.png"), dpi=300)
            plt.close()

def generate_variance_range_plots_range(df, v_name, output_dir):
    """ Standard P50 Variance Plot """
    print(f"Generating Variance Range plots (P50 Only)...")
    info = VARIANCE_METRICS[v_name]
    if 'val_p50' not in df.columns: return

    vals = df['val_p50'] / info['scale_factor']
    df_temp = df.copy()
    df_temp['scaled_val'] = vals
    
    stats = df_temp.groupby('scenario')['scaled_val'].agg(
        mean='mean',
        p5=lambda x: np.percentile(x, 5),
        p95=lambda x: np.percentile(x, 95)
    ).reindex(SCENARIOS).dropna()

    if stats.empty: return
    stats.to_csv(os.path.join(output_dir, f"Variance_Range_P50_{v_name}.csv"))

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
    UPDATED: Compares Portfolio Sum(P5) vs Sum(P50) vs Sum(P95).
    NOW INCLUDES ERROR BARS showing the variation of these sums across Epistemic Runs.
    """
    print(f"Generating Risk Comparison plots (P5/P50/P95 with Epistemic Ranges)...")
    info = VARIANCE_METRICS[v_name]
    
    # 1. Prepare Data Containers
    # We need separate aggregations for each "Risk Bar" type
    # metrics_to_plot: (column_name, label, color, offset)
    bar_configs = [
        ('val_p5',  'Sum P5',  '#2ca02c', -0.25),
        ('val_p50', 'Sum P50',    '#1f77b4', 0),
        ('val_p95', 'Sum P95', '#d62728', 0.25)
    ]
    
    stats_storage = [] # To save to CSV later

    plt.figure(figsize=(12, 6))
    
    # We will iterate through SCENARIOS to ensure x-axis alignment
    # But for plotting efficiency, we iterate by "Bar Type"
    
    valid_scenarios = [s for s in SCENARIOS if s in df['scenario'].unique()]
    x_indexes = np.arange(len(valid_scenarios))
    
    for col_name, label, color, offset in bar_configs:
        if col_name not in df.columns:
            continue
            
        # Extract data for this metric
        # Scale units (e.g. to Millions)
        sub_df = df.copy()
        sub_df[col_name] = sub_df[col_name] / info['scale_factor']
        
        # Calculate P5, Mean, P95 across Epistemic Runs for this specific metric
        grouped = sub_df.groupby('scenario')[col_name].agg(
            mean='mean',
            p5=lambda x: np.percentile(x, 5),
            p95=lambda x: np.percentile(x, 95)
        ).reindex(valid_scenarios)
        
        # Store for CSV
        grouped['metric_type'] = label
        stats_storage.append(grouped)
        
        # Calculate asymmetric error bars
        # yerr shape must be [2, N] -> [[lower], [upper]]
        lower_err = grouped['mean'] - grouped['p5']
        upper_err = grouped['p95'] - grouped['mean']
        
        # Plot Bar with Error Whiskers
        plt.bar(x_indexes + offset, grouped['mean'], width=0.25,
                yerr=[lower_err, upper_err],
                label=label, color=color, alpha=0.8, edgecolor='black',
                capsize=3, error_kw={'elinewidth': 1, 'alpha': 0.7})

    # Export Stats Table
    if stats_storage:
        full_stats = pd.concat(stats_storage)
        full_stats.to_csv(os.path.join(output_dir, f"Risk_Comparison_Stats_{v_name}.csv"))

    # Plot formatting
    plt.ylabel(info['ylabel'], fontweight='bold')
    plt.xlabel('Scenario', fontweight='bold')
    plt.legend(title="Building-Level Assumption")
    
    clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in valid_scenarios]
    plt.xticks(x_indexes, clean_labels, rotation=45, ha='right')
    
    ax = plt.gca()
    # Add comma separator to y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"Risk_Comparison_Bar_Errors_{v_name}.png"), dpi=300)
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
        # Assuming you kept the original function or I can include it if needed
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