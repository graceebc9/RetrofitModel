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

# Update these paths to match your environment
OUTPUT_DIR = '1_summary_results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, 'processing_errors.txt')

SCENARIOS=[
    'loft_installation',
    'wall_installation',
    'joint_heat_loft_decay',
 'joint_heat_wall_decay',
 'join_heat_ins_decay',
 'heat_pump_only',
 ]

YEARS = 5
GAS_FACTOR = 0.18        # kgCO2e/kWh
ELEC_FACTOR = 0.19338    # kgCO2e/kWh

if is_hpc:
    if not is_epc:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/*csv'
        REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
    else:
        LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/v8_logs_with_epc/*csv'
        REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
else: 
    if is_epc:
        LOG_DIR='/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/epc_merge/*csv'
        REFERENCE_FILE = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/114_log_file.csv'
    else:
        LOG_DIR = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/*csv'
        REFERENCE_FILE = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/114_log_file.csv'

# ==============================================================================
# 2. THE STATS ACCUMULATOR CLASS
# ==============================================================================
class GlobalStatsAccumulator:
    """
    Tracks running totals of sums AND sums of variances to calculate 
    Portfolio Totals and Global Means without RAM overhead.
    """
    def __init__(self, scenarios):
        self.scenarios = scenarios
        # Structure: {scenario: {metric: {'sum': 0.0, 'var_sum': 0.0, 'count': 0}}}
        self.data = {scn: {} for scn in scenarios}
        
    def update(self, df_agg):
        """
        Takes an aggregated batch (one file) and updates running totals.
        """
        for scn in self.scenarios:
            # Metrics mapping: Internal Name -> CSV Column Name
            metrics = {
                'cost': {'mean': f'{scn}_cost_{scn}_mean', 'std': f'{scn}_cost_{scn}_std'},
                'gas':  {'mean': f'{scn}_gas_saving_abs_kwh_{scn}_mean', 'std': f'{scn}_gas_saving_abs_kwh_{scn}_std'},
                'elec': {'mean': f'{scn}_elec_saving_abs_kwh_{scn}_mean', 'std': f'{scn}_elec_saving_abs_kwh_{scn}_std'}
            }
            
            for key, cols in metrics.items():
                mean_col = cols['mean']
                std_col = cols['std']
                
                if mean_col not in df_agg.columns: continue
                
                # Initialize key if first time seeing it
                if key not in self.data[scn]:
                    self.data[scn][key] = {'sum': 0.0, 'var_sum': 0.0, 'count': 0}
                
                # Filter valid data
                valid_df = df_agg[[mean_col, std_col]].dropna()
                
                if valid_df.empty: continue

                # Update Sum of Means (for Totals)
                self.data[scn][key]['sum'] += valid_df[mean_col].sum()
                
                # Update Sum of Variances (Sigma^2) (for Total Error Bars)
                # We square the std_dev to get variance, then sum variances
                self.data[scn][key]['var_sum'] += (valid_df[std_col] ** 2).sum()
                
                self.data[scn][key]['count'] += len(valid_df)

    def get_summary_dataframe(self):
        """
        Calculates Final Totals and Averages.
        """
        rows = []
        for scn in self.scenarios:
            stats = self.data[scn]
            row = {'Scenario': scn}
            
            # Helper to safely retrieve stats
            def get_stats(metric_key):
                if metric_key not in stats or stats[metric_key]['count'] == 0: 
                    return 0.0, 0.0, 0
                
                total_sum = stats[metric_key]['sum']
                # Total Uncertainty = sqrt(Sum of Variances)
                total_std = np.sqrt(stats[metric_key]['var_sum'])
                count = stats[metric_key]['count']
                return total_sum, total_std, count

            # 1. Raw Stats
            tot_cost, tot_cost_std, N = get_stats('cost')
            tot_gas_kwh, tot_gas_std_kwh, _ = get_stats('gas')
            tot_elec_kwh, tot_elec_std_kwh, _ = get_stats('elec')

            # 2. Convert to Carbon Totals (Tonnes)
            # Factor applies to both Mean and Std linearly
            tot_gas_t = (tot_gas_kwh * YEARS * GAS_FACTOR) / 1000.0
            tot_gas_std_t = (tot_gas_std_kwh * YEARS * GAS_FACTOR) / 1000.0
            
            tot_elec_t = (tot_elec_kwh * YEARS * ELEC_FACTOR) / 1000.0
            tot_elec_std_t = (tot_elec_std_kwh * YEARS * ELEC_FACTOR) / 1000.0

            # 3. Net Carbon Savings
            # Flip sign: Input negative (reduction) -> Output positive savings
            tot_savings = (tot_gas_t + tot_elec_t) * -1
            # Propagate error for addition: sqrt(sigma_gas^2 + sigma_elec^2)
            tot_savings_std = np.sqrt(tot_gas_std_t**2 + tot_elec_std_t**2)

            # 4. Cost Effectiveness (Total Cost / Total Carbon)
            if tot_savings > 1.0: 
                metric_mean = tot_cost / tot_savings
                # Rel Variance Formula for division
                rel_var_c = (tot_cost_std / tot_cost)**2 if tot_cost > 0 else 0
                rel_var_s = (tot_savings_std / tot_savings)**2
                metric_std = metric_mean * np.sqrt(rel_var_c + rel_var_s)
            else:
                metric_mean = 0
                metric_std = 0
            
            row.update({
                'Count': N,
                'Total_Cost': tot_cost, 
                'Total_Cost_Std': tot_cost_std,
                'Total_Savings': tot_savings, 
                'Total_Savings_Std': tot_savings_std,
                'Metric_Mean': metric_mean, 
                'Metric_Std': metric_std
            })
            rows.append(row)
            
        return pd.DataFrame(rows)

# ==============================================================================
# 3. HELPER: AGGREGATION LOGIC
# ==============================================================================
def process_batch_robust(df, scenarios, id_col='upn'):
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
    
    accumulator = GlobalStatsAccumulator(SCENARIOS)
    
    # Header handling
    if not is_epc and os.path.exists(REFERENCE_FILE):
        try:
            with open(REFERENCE_FILE, 'r') as f:
                master_headers = next(csv.reader(f))
        except Exception:
            master_headers = None
    else:
        master_headers = None 

    # Processing Loop
    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)}...")
        try:
            df = safe_load(file_path, master_headers, ERROR_LOG_FILE)
            if df.empty: continue
            
            df_agg = process_batch_robust(df, SCENARIOS)
            accumulator.update(df_agg)
            
            del df, df_agg
            gc.collect()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # ==========================================================================
    # 5. FINAL VISUALIZATION (BAR CHARTS FOR TOTALS)
    # ==========================================================================
    print("Generating Summary Figures...")
    df_summary = accumulator.get_summary_dataframe()
    
    if df_summary.empty:
        print("No data accumulated.")
        return

    # Clean labels
    df_summary['Label'] = df_summary['Scenario'].str.replace('joint_', '').str.replace('_', '\n')
    
    # Setup Plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7)) # 3 Panels
    colors = sns.color_palette("viridis", len(df_summary))
    x = np.arange(len(df_summary))
    width = 0.6
    
    # --- PANEL 1: TOTAL COST ---
    axes[0].bar(x, df_summary['Total_Cost'], yerr=df_summary['Total_Cost_Std'], 
                capsize=5, color=colors, alpha=0.9, width=width)
    axes[0].set_title('Total Portfolio Cost (£)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('£ (Millions/Billions)')
    # Optional: Format Y-axis to be more readable (e.g. Millions)
    # axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1e6:.1f}M'))

    # --- PANEL 2: TOTAL CARBON ---
    axes[1].bar(x, df_summary['Total_Savings'], yerr=df_summary['Total_Savings_Std'], 
                capsize=5, color=colors, alpha=0.9, width=width)
    axes[1].set_title('Total Carbon Saved (tCO2e)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Tonnes CO2e')

    # --- PANEL 3: COST EFFECTIVENESS ---
    axes[2].bar(x, df_summary['Metric_Mean'], yerr=df_summary['Metric_Std'], 
                capsize=5, color=colors, alpha=0.9, width=width)
    axes[2].set_title('Cost Effectiveness (£/tCO2e)', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('£ / tCO2e')
    
    # Common Formatting
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary['Label'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, which='major')
    
    plt.suptitle("Total Portfolio Impact", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'total_portfolio_bars.png'), bbox_inches='tight')
    plt.show()
    
    # print("\nFINAL PORTFOLIO TOTALS:")
    # print(df_summary[['Scenario', 'Total_Cost', 'Total_Savings', 'Metric_Mean']].round(2))

if __name__ == "__main__":
    run_pipeline()