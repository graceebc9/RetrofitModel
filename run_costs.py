import sys 
import pandas as pd
import numpy as np
import glob
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import is_running_on_hpc 
import csv

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
is_hpc = is_running_on_hpc() 
is_epc = False   

OUTPUT_DIR = '1_summary_results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = [
    'loft_installation',
    'wall_installation',
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'join_heat_ins_decay',
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

VARIANCE_METRICS = {
    'Total_Cost': {
        'col_pattern': '{sc}_cost_{sc}_p50', 
        'ylabel': 'Total Cost (£)',
        'title': 'Variance of Total Cost (Sum across all buildings) per Run'
    },
    'Total_Carbon_Removed': {
        'col_pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_p50', 
        'ylabel': 'Total Carbon Removed (Tons)',
        'title': 'Variance of Total Carbon Removed per Run'
    }
}

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

        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    return summary_store, variance_records

def generate_median_plots(data_store, output_dir):
    """Original plotting function: Median of Medians."""
    print(f"Generating Median Summary plots...")
    for metric_name, info in METRICS_INFO.items():
        bar_heights = []
        lower_errors = []
        upper_errors = []
        valid_scenarios = []
        
        # Iterate through SCENARIOS list to ensure correct order
        for sc in SCENARIOS:
            d = data_store[metric_name][sc]
            if len(d['p50']) > 0:
                med_p5  = np.median(d['p5'])
                med_p50 = np.median(d['p50'])
                med_p95 = np.median(d['p95'])
                
                bar_heights.append(med_p50)
                lower_errors.append(med_p50 - med_p5)
                upper_errors.append(med_p95 - med_p50)
                valid_scenarios.append(sc)

        if valid_scenarios:
            plt.figure(figsize=(10, 6))
            asymmetric_err = [lower_errors, upper_errors]
            plt.bar(valid_scenarios, bar_heights, yerr=asymmetric_err, 
                    capsize=5, color='cornflowerblue', alpha=0.8, edgecolor='black')
            
            plt.ylabel(info['ylabel'])
            plt.xlabel('Scenario')
            
            # --- Apply Clean Labels ---
            clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in valid_scenarios]
            plt.xticks(ticks=range(len(valid_scenarios)), labels=clean_labels, rotation=45, ha='right')
            
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"{metric_name}_median_plot.png"))
            plt.close()

def generate_variance_plots(variance_records, output_dir):
    """
    Boxplot of Global Totals to show distribution across runs.
    """
    print(f"Generating Variance (Box) plots...")
    sns.set_theme(style="whitegrid")

    for v_name, records in variance_records.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        df_total = df.groupby(['scenario', 'epistemic_run_id'])['value'].sum().reset_index()
        
        plt.figure(figsize=(12, 8))
        
        sns.boxplot(x='scenario', y='value', data=df_total, 
                    palette="Set3", showfliers=False, linewidth=1.2,
                    order=SCENARIOS) 
        
        sns.stripplot(x='scenario', y='value', data=df_total, 
                      color=".2", alpha=0.4, jitter=True, size=4,
                      order=SCENARIOS) 
        
        info = VARIANCE_METRICS[v_name]
        plt.ylabel(info['ylabel'], fontsize=12)
        plt.xlabel('Scenario', fontsize=12)
        
        # --- Apply Clean Labels ---
        clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in SCENARIOS]
        plt.xticks(ticks=range(len(SCENARIOS)), labels=clean_labels, rotation=45, ha='right')

        plt.ylim(bottom=0) 
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filename = f"Variance_Runs_Box_{v_name}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def generate_variance_bar_plots(variance_records, output_dir):
    """
    NEW: Bar chart of Global Totals (Mean across runs) with Standard Deviation error bars.
    """
    print(f"Generating Variance (Bar + Std) plots...")
    
    for v_name, records in variance_records.items():
        if not records:
            continue
            
        df = pd.DataFrame(records)
        
        # 1. Aggregate partial chunks to get TOTAL sum per run_id
        df_total = df.groupby(['scenario', 'epistemic_run_id'])['value'].sum().reset_index()
        
        # 2. Calculate Mean and Std Dev across the runs for each scenario
        #    We manually reindex to SCENARIOS to ensure the X-axis order matches config
        summary = df_total.groupby('scenario')['value'].agg(['mean', 'std']).reindex(SCENARIOS)
        
        # Drop scenarios that might be missing (NaNs) just for the plot
        summary = summary.dropna()
        
        if summary.empty:
            continue

        scenarios_present = summary.index.tolist()
        means = summary['mean']
        stds = summary['std']
        
        plt.figure(figsize=(10, 6))
        
        # Plot Bar with Error Bars (Std Dev)
        # Using a different color (mediumseagreen) to distinguish from the median plots
        plt.bar(scenarios_present, means, yerr=stds, 
                capsize=5, color='mediumseagreen', alpha=0.7, edgecolor='black',
                label='Mean across Runs (± 1 Std Dev)')
        
        info = VARIANCE_METRICS[v_name]
        plt.ylabel(info['ylabel'])
        plt.title(f"Mean {v_name.replace('_', ' ')} across Runs")
        plt.xlabel('Scenario')
        
        # Clean Labels
        clean_labels = [SCENARIO_DISPLAY_NAMES.get(sc, sc) for sc in scenarios_present]
        plt.xticks(ticks=range(len(scenarios_present)), labels=clean_labels, rotation=45, ha='right')
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        filename = f"Variance_Runs_Bar_{v_name}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

def main():
    print(f"--- Configuration ---")
    print(f"Input Pattern: {LOG_FILE_PATTERN}")
    print(f"Output Dir:    {OUTPUT_DIR}")
    print(f"---------------------")

    summary_data, variance_data = collect_data(LOG_FILE_PATTERN)
    
    generate_median_plots(summary_data, OUTPUT_DIR)
    generate_variance_plots(variance_data, OUTPUT_DIR)      # Box plots
    generate_variance_bar_plots(variance_data, OUTPUT_DIR)  # New Bar plots
    
    print("Done.")

if __name__ == "__main__":
    main()