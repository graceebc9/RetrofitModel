#!/usr/bin/env python3
# ==============================================================================
# Script Name: Retrofit Analysis & Plotting
# Last Updated: 2026-01-05
# Description: Aggregates retrofit simulation data, generates summary plots 
#              (Median & Variance), and exports data tables for publication.
#              (No Plot Titles Version)
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

# Timestamp for file naming
TODAY = datetime.datetime.now().strftime("%Y_%m_%d")

is_hpc = is_running_on_hpc()
is_epc = False    

OUTPUT_DIR = f'1_summary_results_{TODAY}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = [
    'loft_installation',
    'wall_installation',
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'join_heat_ins_decay', # Check spelling in filenames vs this list
    'heat_pump_only',
]

# --- Dictionary for Clean Plot Labels ---
SCENARIO_DISPLAY_NAMES = {
    'loft_installation': 'Loft Installation',
    'wall_installation': 'Wall Insulation',
    'joint_heat_loft_decay': 'HP + Loft (Decay)',
    'joint_heat_wall_decay': 'HP + Wall (Decay)',
    'join_heat_ins_decay': 'HP + All Insulation (Decay)',
    'heat_pump_only': 'Heat Pump Only',
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
METRICS_INFO = {
    'Capex':  {'pattern': '{sc}_capex_per_net_ton_co2_{sc}_{stat}', 'ylabel': 'Capex (£/ton)'},
    'Energy': {'pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}', 'ylabel': 'Energy (Ton CO2/5yr)'},
    'Cost':   {'pattern': '{sc}_cost_{sc}_{stat}', 'ylabel': 'Cost (£)'}
}

# UPDATED: Labels now reflect Millions and kTONS
VARIANCE_METRICS = {
    'Total_Cost': {
        'col_pattern': '{sc}_cost_{sc}_p50', 
        'ylabel': 'Total Cost (£M)'
    },
    'Total_Carbon_Removed': {
        'col_pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_p50', 
        'ylabel': 'Total Carbon Removed (kTons)'
    }
}

# ==============================================================================
# 2. DATA COLLECTION
# ==============================================================================

def collect_data(file_pattern):
    """
    Reads CSVs and collects data for both plotting types.
    """
    summary_store = {
        metric: {sc: {'p5': [], 'p50': [], 'p95': []} for sc in SCENARIOS} 
        for metric in METRICS_INFO
    }

    variance_records = {
        'Total_Cost': [],
        'Total_Carbon_Removed': []
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
            
            # --- A. Standard Median Plot Data ---
            for metric_name, info in METRICS_INFO.items():
                for sc in SCENARIOS:
                    c_p5  = info['pattern'].format(sc=sc, stat='p5')
                    c_p50 = info['pattern'].format(sc=sc, stat='p50')
                    c_p95 = info['pattern'].format(sc=sc, stat='p95')
                    
                    if c_p50 in chunk.columns:
                        valid = chunk[[c_p5, c_p50, c_p95]].dropna()
                        if not valid.empty:
                            summary_store[metric_name][sc]['p5'].extend(valid[c_p5].tolist())
                            summary_store[metric_name][sc]['p50'].extend(valid[c_p50].tolist())
                            summary_store[metric_name][sc]['p95'].extend(valid[c_p95].tolist())

            # --- B. Variance Plot Data (Run Totals) ---
            if 'epistemic_run_id' in chunk.columns:
                for v_name, v_info in VARIANCE_METRICS.items():
                    for sc in SCENARIOS:
                        col = v_info['col_pattern'].format(sc=sc)
                        
                        if col in chunk.columns:
                            subset = chunk[['epistemic_run_id', col]].dropna()
                            if not subset.empty:
                                # PARTIAL SUM: Collapse buildings in this file -> 1 row per run_id
                                partial_sums = subset.groupby('epistemic_run_id')[col].sum().reset_index()
                                
                                for _, row in partial_sums.iterrows():
                                    variance_records[v_name].append({
                                        'scenario': sc,
                                        'epistemic_run_id': row['epistemic_run_id'],
                                        'value': row[col]
                                    })
            
            del chunk
            
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    gc.collect()
    return summary_store, variance_records

# ==============================================================================
# 3. PLOTTING & SAVING TABLES
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

def generate_variance_plots(variance_records, output_dir):
    """
    Boxplot of Global Totals to show distribution across runs.
    """
    print(f"Generating Variance (Box) plots and tables...")
    sns.set_theme(style="whitegrid")

    for v_name, records in variance_records.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        df_total = df.groupby(['scenario', 'epistemic_run_id'])['value'].sum().reset_index()
        
        # --- UNIT CONVERSION LOGIC ---
        if v_name == 'Total_Cost':
            # Convert to Millions
            df_total['value'] = df_total['value'] / 1_000_000
        elif v_name == 'Total_Carbon_Removed':
            # Convert to KTONS (Thousands of Tons)
            df_total['value'] = df_total['value'] / 1_000

        # Add clean names for the CSV export
        df_total['Scenario_Display'] = df_total['scenario'].map(SCENARIO_DISPLAY_NAMES).fillna(df_total['scenario'])

        # --- Save Raw Data Table ---
        csv_name = f"Variance_Raw_Runs_{v_name}.csv"
        df_total.to_csv(os.path.join(output_dir, csv_name), index=False)
        print(f"Saved Table: {csv_name}")
        
        # --- Plotting ---
        plt.figure(figsize=(12, 8))
        
        # hue='scenario' added to fix FutureWarning
        sns.boxplot(x='scenario', y='value', hue='scenario', data=df_total, 
                    palette="Set3", showfliers=False, linewidth=1.2,
                    order=SCENARIOS, legend=False) 
        
        sns.stripplot(x='scenario', y='value', data=df_total, 
                      color=".2", alpha=0.4, jitter=True, size=4,
                      order=SCENARIOS) 
        
        info = VARIANCE_METRICS[v_name]
        plt.ylabel(info['ylabel'], fontsize=12, fontweight='bold')
        plt.xlabel('Scenario', fontsize=12, fontweight='bold')
        # plt.title Removed
        
        clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in SCENARIOS]
        plt.xticks(ticks=range(len(SCENARIOS)), labels=clean_labels, rotation=45, ha='right')

        ax = plt.gca()
        # Changed formatter to show decimal places since we are now in Millions/KTONS
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.ylim(bottom=0) 
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"Variance_Runs_Box_{v_name}.png"), dpi=300)
        plt.close()

def generate_variance_bar_plots(variance_records, output_dir):
    """
    Bar chart of Global Totals (Mean across runs) with Std Dev.
    Includes 'count' (number of runs) in the CSV.
    """
    print(f"Generating Variance (Bar + Std) plots and tables...")
    
    for v_name, records in variance_records.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        df_total = df.groupby(['scenario', 'epistemic_run_id'])['value'].sum().reset_index()

        # --- UNIT CONVERSION LOGIC ---
        if v_name == 'Total_Cost':
            # Convert to Millions
            df_total['value'] = df_total['value'] / 1_000_000
        elif v_name == 'Total_Carbon_Removed':
            # Convert to KTONS (Thousands of Tons)
            df_total['value'] = df_total['value'] / 1_000
        
        # Added 'count'
        summary = df_total.groupby('scenario')['value'].agg(['mean', 'std', 'count']).reindex(SCENARIOS)
        summary = summary.dropna()

        if summary.empty:
            continue
            
        # --- Save Summary Table ---
        summary_export = summary.copy()
        
        # Fix for TypeError: Use list comprehension
        summary_export['Scenario_Display'] = [SCENARIO_DISPLAY_NAMES.get(x, x) for x in summary_export.index]
        
        # Rename columns for clarity in the CSV
        summary_export.rename(columns={'count': 'Run_Count', 'mean': 'Mean_Value', 'std': 'Std_Dev'}, inplace=True)
        
        csv_name = f"Variance_Stats_MeanStd_{v_name}.csv"
        summary_export.to_csv(os.path.join(output_dir, csv_name))
        print(f"Saved Table: {csv_name} (with run counts)")

        # --- Plotting ---
        scenarios_present = summary.index.tolist()
        means = summary['mean']
        stds = summary['std']
        
        plt.figure(figsize=(10, 6))
        
        plt.bar(scenarios_present, means, yerr=stds, 
                capsize=5, color='mediumseagreen', alpha=0.7, edgecolor='black',
                )
        
        info = VARIANCE_METRICS[v_name]
        plt.ylabel(info['ylabel'], fontweight='bold')
        plt.xlabel('Scenario', fontweight='bold')
        # plt.title Removed
        
        clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in scenarios_present]
        plt.xticks(ticks=range(len(scenarios_present)), labels=clean_labels, rotation=45, ha='right')
        
        ax = plt.gca()
        # Changed formatter to show decimal places since we are now in Millions/KTONS
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.2f}'))
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"Variance_Runs_Bar_{v_name}.png"), dpi=300)
        plt.close()

def generate_variance_range_plots_range(variance_records, output_dir):
    """
    Generates Bar plots showing the Mean Global Total with P5-P95 Confidence Intervals.
    This quantifies the 'range' of likely outcomes across epistemic runs.
    """
    print(f"Generating Variance Range plots (Mean with P5-P95 intervals)...")
    
    for v_name, records in variance_records.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        # 1. Aggregate to get one total per run
        df_total = df.groupby(['scenario', 'epistemic_run_id'])['value'].sum().reset_index()

        # 2. Unit Conversion
        if v_name == 'Total_Cost':
            df_total['value'] = df_total['value'] / 1_000_000  # Millions
        elif v_name == 'Total_Carbon_Removed':
            df_total['value'] = df_total['value'] / 1_000      # kTons
        
        # 3. Calculate Statistics: Mean, P5, P95
        # We use a custom aggregation function
        summary = df_total.groupby('scenario')['value'].agg(
            mean='mean',
            p5=lambda x: np.percentile(x, 5),
            p95=lambda x: np.percentile(x, 95),
            count='count'
        ).reindex(SCENARIOS)
        
        summary = summary.dropna()

        if summary.empty:
            continue
            
        # 4. Save Summary Table (Crucial for verifying the numbers)
        summary_export = summary.copy()
        summary_export['Scenario_Display'] = [SCENARIO_DISPLAY_NAMES.get(x, x) for x in summary_export.index]
        
        csv_name = f"Variance_Stats_Range_P5_P95_{v_name}.csv"
        summary_export.to_csv(os.path.join(output_dir, csv_name))
        print(f"Saved Table: {csv_name}")

        # 5. Prepare Error Bars for Matplotlib
        # Matplotlib requires error bars to be relative lengths (distance from mean), not absolute coordinates.
        # shape: [2, N] -> [[lower_errors], [upper_errors]]
        lower_error = summary['mean'] - summary['p5']
        upper_error = summary['p95'] - summary['mean']
        asymmetric_error = [lower_error, upper_error]

        # 6. Plotting
        scenarios_present = summary.index.tolist()
        means = summary['mean']
        
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        plt.bar(scenarios_present, means, yerr=asymmetric_error, 
                capsize=5, color='mediumseagreen', alpha=0.7, edgecolor='black',
                error_kw={'elinewidth': 1.5, 'capthick': 1.5}) # Thicker error bars for visibility
        
        # Optional: Overlay individual runs as faint dots to show density
        # (Uncomment the lines below if you want to see the dots too)
        # sns.stripplot(x='scenario', y='value', data=df_total, 
        #               order=scenarios_present, color='black', alpha=0.3, jitter=True, size=3)

        info = VARIANCE_METRICS[v_name]
        plt.ylabel(info['ylabel'], fontweight='bold')
        plt.xlabel('Scenario', fontweight='bold')
        
        clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in scenarios_present]
        plt.xticks(ticks=range(len(scenarios_present)), labels=clean_labels, rotation=45, ha='right')
        
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.2f}'))
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"Variance_Runs_Range_P5P95_{v_name}.png"), dpi=300)
        plt.close()

def main():
    print(f"========================================")
    print(f"Retrofit Analysis - {TODAY}")
    print(f"========================================")
    print(f"Input Pattern: {LOG_FILE_PATTERN}")
    print(f"Output Dir:    {OUTPUT_DIR}")
    print(f"----------------------------------------")

    summary_data, variance_data = collect_data(LOG_FILE_PATTERN)
    generate_variance_range_plots_range(variance_data, OUTPUT_DIR)
    
    #generate_median_plots(summary_data, OUTPUT_DIR)
    #generate_variance_plots(variance_data, OUTPUT_DIR)
    #generate_variance_bar_plots(variance_data, OUTPUT_DIR)
    
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()