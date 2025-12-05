import pandas as pd
import numpy as np
import glob
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import is_running_on_hpc 
import csv
from src.RetrofitUtils import safe_load 


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
is_hpc = is_running_on_hpc() 
is_epc = False   



OUTPUT_DIR = 'processed_results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS=[
    'loft_installation',
    'wall_installation',
    'joint_heat_loft_decay',
 'joint_heat_wall_decay',
 'join_heat_ins_decay',
 'heat_pump_only',
 ]

YEARS = 5
GAS_FACTOR = 0.18
ELEC_FACTOR = 0.19338
RISK_PENALTY_SIGMA = 1.0

if is_hpc:
    # Update this path if necessary to match your actual data location
    if not is_epc:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/*csv'
    else:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/v8_logs_with_epc/*csv'
        # Use the file you confirmed works as the Source of Truth for headers
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
else: 
    if is_epc:
        LOG_DIR='/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge/*csv'

    else:
        LOG_DIR = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/*csv'
    REFERENCE_FILE = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/114_log_file.csv'


OUTPUT_DIR= '1_summary_results/'
os.makedirs(OUTPUT_DIR,exist_ok=True)
ERROR_LOG_FILE = '1_summary_results/processing_errors.txt'

# ==============================================================================
# 2. THE STATS ACCUMULATOR CLASS
# ==============================================================================
class GlobalStatsAccumulator:
    """
    Tracks running totals of sums and counts to calculate 
    Global Means and Average Uncertainties without RAM overhead.
    """
    def __init__(self, scenarios):
        self.scenarios = scenarios
        # Structure: {scenario: {metric: {'sum': 0.0, 'count': 0}}}
        self.data = {scn: {} for scn in scenarios}
        
    def update(self, df_agg):
        """
        Takes an aggregated batch (one file) and updates running totals.
        """
        for scn in self.scenarios:
            # Metrics to track
            metrics = {
                'cost_mean': f'{scn}_cost_{scn}_mean',
                'cost_std':  f'{scn}_cost_{scn}_std',
                'gas_mean':  f'{scn}_gas_saving_abs_kwh_{scn}_mean',
                'gas_std':   f'{scn}_gas_saving_abs_kwh_{scn}_std',
                'elec_mean': f'{scn}_elec_saving_abs_kwh_{scn}_mean',
                'elec_std':  f'{scn}_elec_saving_abs_kwh_{scn}_std'
            }
            
            for key, col in metrics.items():
                if col not in df_agg.columns: continue
                
                # Initialize key if first time seeing it
                if key not in self.data[scn]:
                    self.data[scn][key] = {'sum': 0.0, 'count': 0}
                
                # Update Totals
                valid_data = df_agg[col].dropna()
                self.data[scn][key]['sum'] += valid_data.sum()
                self.data[scn][key]['count'] += len(valid_data)

    def get_summary_dataframe(self):
        """
        Calculates final averages and returns a clean DataFrame for plotting.
        """
        rows = []
        for scn in self.scenarios:
            stats = self.data[scn]
            row = {'Scenario': scn}
            
            # Helper to safely get average
            def get_avg(metric_key):
                if metric_key not in stats or stats[metric_key]['count'] == 0: return 0
                return stats[metric_key]['sum'] / stats[metric_key]['count']

            # Helper to get count (using cost_mean as proxy for valid buildings)
            def get_count():
                if 'cost_mean' not in stats: return 0
                return stats['cost_mean']['count']

            # 1. Retrieve Raw Averages & Count
            N = get_count()
            mu_cost = get_avg('cost_mean')
            sig_cost = get_avg('cost_std')
            
            mu_gas_kwh = get_avg('gas_mean')
            sig_gas_kwh = get_avg('gas_std')
            
            mu_elec_kwh = get_avg('elec_mean')
            sig_elec_kwh = get_avg('elec_std')

            # 2. Convert to Carbon Tonnes
            mu_gas_t = (mu_gas_kwh * YEARS * GAS_FACTOR) / 1000
            sig_gas_t = (sig_gas_kwh * YEARS * GAS_FACTOR) / 1000
            
            mu_elec_t = (mu_elec_kwh * YEARS * ELEC_FACTOR) / 1000
            sig_elec_t = (sig_elec_kwh * YEARS * ELEC_FACTOR) / 1000

            # 3. Combine Energy (Net Savings)
            mu_savings = (mu_gas_t + mu_elec_t) * -1
            sig_savings = np.sqrt(sig_gas_t**2 + sig_elec_t**2)

            # 4. Calculate Capex per Ton (Error Propagation)
            if mu_savings > 0.01:
                mu_metric = mu_cost / mu_savings
                # Rel Variance Formula
                rel_var_c = (sig_cost / mu_cost)**2 if mu_cost > 0 else 0
                rel_var_s = (sig_savings / mu_savings)**2
                sig_metric = mu_metric * np.sqrt(rel_var_c + rel_var_s)
            else:
                mu_metric = 0
                sig_metric = 0
            
            # Store
            row.update({
                'Count': int(N),
                'Cost_Mean': mu_cost, 'Cost_Std': sig_cost,
                'Savings_Mean': mu_savings, 'Savings_Std': sig_savings,
                'Metric_Mean': mu_metric, 'Metric_Std': sig_metric
            })
            rows.append(row)
            
        return pd.DataFrame(rows)

# ==============================================================================
# 3. HELPER: AGGREGATION LOGIC
# ==============================================================================
def process_batch_robust(df, scenarios, id_col='upn'):
    """
    Simplified robust aggregator that returns the df_agg needed for the accumulator
    """
    agg_map = {}
    calc_tasks = []
    
    # Identify Mean/Std columns
    for scn in scenarios:
        cols = [f'{scn}_cost_{scn}', f'{scn}_gas_saving_abs_kwh_{scn}', f'{scn}_elec_saving_abs_kwh_{scn}']
        for base in cols:
            col_mean = f'{base}_mean'
            col_std = f'{base}_std'
            if col_mean in df.columns:
                agg_map[col_mean] = 'mean'
                if col_std in df.columns:
                    calc_tasks.append({'mean': col_mean, 'std': col_std})

    # Group
    grouped = df.groupby(id_col)
    df_agg = grouped.agg(agg_map)
    
    # Robust Pooling (Law of Total Variance)
    for task in calc_tasks:
        var_epistemic = grouped[task['mean']].var(ddof=1).fillna(0)
        temp = df[[id_col, task['std']]].copy()
        temp['var'] = temp[task['std']] ** 2
        var_aleatoric = temp.groupby(id_col)['var'].mean().fillna(0)
        
        pooled_std = np.sqrt(var_aleatoric + var_epistemic)
        df_agg[task['std']] = pooled_std.astype('float32')
        
    return df_agg.reset_index()

# ==============================================================================
# 4. MAIN PROCESSING PIPELINE
# ==============================================================================
def run_pipeline():
    files = glob.glob(LOG_DIR)
    print(f"Found {len(files)} log files.")
    
    if not is_epc:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                master_headers = next(csv.reader(f))
                print(f'Master headers:  {master_headers}' )
        except Exception as e:
            
            return
    else:
        master_headers=None 
    
    # Initialize Accumulator
    accumulator = GlobalStatsAccumulator(SCENARIOS)
    
    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)}...")
        
        try:
            # A. Load
            df = safe_load(file_path, master_headers, ERROR_LOG_FILE)
            if df.empty: continue
            
            # B. Aggregate (Robust Stats)
            df_agg = process_batch_robust(df, SCENARIOS)
            
            # C. Update Global Stats (Accumulate)
            accumulator.update(df_agg)
            
            # D. Cleanup
            del df, df_agg
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # ==========================================================================
    # 5. GENERATE DATA
    # ==========================================================================
    print("Calculating final statistics...")
    df_summary = accumulator.get_summary_dataframe()
    
    # Create Labels for Plotting (Include Count in Label)
    df_summary['Label'] = (
        df_summary['Scenario'].str.replace('joint_', '').str.replace('_', '\n') + 
        '\n(n=' + df_summary['Count'].astype(str) + ')'
    )

    # ==========================================================================
    # 5A. SAVE SUMMARY TABLE
    # ==========================================================================
    csv_filename = 'global_summary_stats.csv'
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    print(f"Saving summary data to: {csv_path}")
    df_summary.to_csv(csv_path, index=False)
    
    # ==========================================================================
    # 6. FINAL VISUALIZATION
    # ==========================================================================
    print("Generating Summary Figures...")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = sns.color_palette("viridis", len(df_summary))
    x = np.arange(len(df_summary))
    width = 0.6
    
    # --- PANEL 1: COST ---
    axes[0].bar(x, df_summary['Cost_Mean'], yerr=df_summary['Cost_Std'], 
                capsize=5, color=colors, alpha=0.9, width=width)
    axes[0].set_title('Mean Cost (£)', fontweight='bold')
    axes[0].set_ylabel('£')
    
    # --- PANEL 2: SAVINGS ---
    axes[1].bar(x, df_summary['Savings_Mean'], yerr=df_summary['Savings_Std'], 
                capsize=5, color=colors, alpha=0.9, width=width)
    axes[1].set_title('Mean Savings (tCO2 / 5yrs)', fontweight='bold')
    axes[1].set_ylabel('tCO2')
    
    # --- PANEL 3: METRIC ---
    axes[2].bar(x, df_summary['Metric_Mean'], yerr=df_summary['Metric_Std'], 
                capsize=5, color=colors, alpha=0.9, width=width)
    axes[2].set_title('Cost Effectiveness (£/tCO2)', fontweight='bold')
    axes[2].set_ylabel('£/tCO2')
    
    # Formatting
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary['Label'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f"Global Summary across {len(files)} files", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'global_summary_plot.png'))
    plt.show()
    
    # Print Table (Added Count Column)
    print("\nFINAL GLOBAL STATS:")
    print(df_summary[['Scenario', 'Count', 'Cost_Mean', 'Metric_Mean']].round(2))

if __name__ == "__main__":
    run_pipeline()