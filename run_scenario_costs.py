import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import gc
import csv
from src.utils import is_running_on_hpc 
# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

SCENARIOS = [
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'wall_installation',
    'join_heat_ins_decay',
    'heat_pump_only',
    'loft_installation'
]

OUTPUT_BASE = '2_stock_results/vis_outputs/'
os.makedirs(OUTPUT_BASE, exist_ok=True)


is_hpc = is_running_on_hpc() 
if is_hpc:
    LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/*csv'
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
else:
    # Example local path
    LOG_DIR = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/*csv'
    REFERENCE_FILE = None

GROUP_COLS = ['avg_gas_percentile', 'premise_type', 'inferred_insulation_type']


TYPOLOGIES = [
    'Small low terraces', '3-4 storey and smaller flats',
    'Tall terraces 3-4 storeys', 'Large semi detached', 'Standard size detached',
    'Standard size semi detached', '2 storeys terraces with t rear extension',
    'Semi type house in multiples', 'Large detached',
      'Planned balanced mixed estates',
    'Linked and step linked premises',
]
# ==============================================================================
# 2. THE GROUPED ACCUMULATOR CLASS
# ==============================================================================
class GroupedStatsAccumulator:
    def __init__(self, scenario_name):
        self.scenario = scenario_name
        self.cost_col = f'{scenario_name}_cost_{scenario_name}_mean' 
        self.data = {}

    def update(self, df):
        if self.cost_col not in df.columns:
            return

        df_subset = df.dropna(subset=GROUP_COLS + [self.cost_col])
        if df_subset.empty:
            return

        grouped = df_subset.groupby(GROUP_COLS)
        
        for name, group in grouped:
            n = len(group)
            sum_c = group[self.cost_col].sum()
            sum_sq_c = (group[self.cost_col] ** 2).sum()

            if name not in self.data:
                self.data[name] = {'n': 0, 'sum_cost': 0.0, 'sum_sq_cost': 0.0}
            
            self.data[name]['n'] += n
            self.data[name]['sum_cost'] += sum_c
            self.data[name]['sum_sq_cost'] += sum_sq_c

    def get_detailed_df(self):
        rows = []
        for (dec, prem, wall), stats in self.data.items():
            rows.append({
                'avg_gas_percentile': dec,
                'premise_type': prem,
                'inferred_insulation_type': wall,
                'n': stats['n'],
                'sum_cost': stats['sum_cost'],
                'sum_sq_cost': stats['sum_sq_cost']
            })
        return pd.DataFrame(rows)

# ==============================================================================
# 3. HELPER: CALCULATE MEAN & STD FROM SUMS
# ==============================================================================
def aggregate_and_calculate_stats(detailed_df, group_by_cols):
    if detailed_df.empty:
        return pd.DataFrame()

    # Group by the desired columns (aggregating away the others)
    grouped = detailed_df.groupby(group_by_cols).agg({
        'n': 'sum',
        'sum_cost': 'sum',
        'sum_sq_cost': 'sum'
    }).reset_index()
    
    grouped['mean_cost'] = grouped['sum_cost'] / grouped['n']
    
    # Calculate Variance and Standard Error
    variance = (grouped['sum_sq_cost'] / grouped['n']) - (grouped['mean_cost'] ** 2)
    variance = variance.clip(lower=0)
    
    grouped['std_cost'] = np.sqrt(variance)
    grouped['se_cost'] = grouped['std_cost'] / np.sqrt(grouped['n'])
    grouped['ci_error'] = 2 * grouped['se_cost'] 
    
    return grouped

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================
def plot_costs_by_decile(df, scenario_name, output_path):
    if df.empty: return

    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    df = df.sort_values(by=['inferred_insulation_type', 'decile_numeric'])
    
    wall_types = sorted(df['inferred_insulation_type'].unique())
    deciles = sorted(df['decile_numeric'].dropna().unique())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_types = len(wall_types)
    if n_types == 0: 
        plt.close(); return

    # If only 1 type (not split by wall), make bar wider for aesthetics
    bar_width = 0.6 if n_types == 1 else 0.8 / n_types
    
    x_pos = np.arange(len(deciles))
    colors = plt.cm.viridis(np.linspace(0, 1, n_types))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('decile_numeric').reindex(deciles).reset_index()
        
        # Center the bar if n=1, otherwise offset
        offset = 0 if n_types == 1 else (i - n_types/2 + 0.5) * bar_width
        
        ax.bar(x_pos + offset, subset['mean_cost'], width=bar_width, 
               yerr=subset['ci_error'], label=w_type, color=colors[i], capsize=3, alpha=0.8)

    ax.set_xlabel('Gas Usage Decile')
    ax.set_ylabel('Avg Installation Costs (£)')
    ax.set_title(f'Cost by Decile\n({scenario_name})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([int(d) for d in deciles])
    
    # Only show legend if we actually have multiple wall types
    if n_types > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_costs_by_premise_type(df, scenario_name, output_path):
    if df.empty: return

    df['premise_type'] = pd.Categorical(df['premise_type'], categories=TYPOLOGIES, ordered=True)
    df = df.sort_values(by=['inferred_insulation_type', 'premise_type'])
    
    present_types = [t for t in TYPOLOGIES if t in df['premise_type'].unique()]
    wall_types = sorted(df['inferred_insulation_type'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    n_types = len(wall_types)
    if n_types == 0: 
        plt.close(); return

    bar_width = 0.6 if n_types == 1 else 0.8 / n_types
    
    x_pos = np.arange(len(present_types))
    colors = plt.cm.viridis(np.linspace(0, 1, n_types))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('premise_type').reindex(present_types).reset_index()
        
        offset = 0 if n_types == 1 else (i - n_types/2 + 0.5) * bar_width
        
        ax.bar(x_pos + offset, subset['mean_cost'], width=bar_width, 
               yerr=subset['ci_error'], label=w_type, color=colors[i], capsize=3, alpha=0.8)

    ax.set_xlabel('Premise Type')
    ax.set_ylabel('Avg Installation Costs (£)')
    ax.set_title(f'Cost by Premise Type\n({scenario_name})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(present_types, rotation=45, ha='right')
    
    if n_types > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

# ==============================================================================
# 5. LOADING HELPER
# ==============================================================================
def safe_load(filepath, headers=None):
    try:
        if headers:
            return pd.read_csv(filepath, names=headers, header=0)
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return pd.DataFrame()

# ==============================================================================
# 6. MAIN PIPELINE
# ==============================================================================
def run_pipeline():
    print(LOG_DIR)
    files = glob.glob(LOG_DIR)
    
    
    print(f"Found {len(files)} files.")
    
    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
        except:
            print("Warning: Could not read headers.")

    accumulators = {scn: GroupedStatsAccumulator(scn) for scn in SCENARIOS}
    
    for i, file_path in enumerate(files):
        if i % 10 == 0: print(f"Processing file {i}/{len(files)}...")
        
        df = safe_load(file_path, headers)
        if df.empty: continue
        
        if not set(GROUP_COLS).issubset(df.columns):
            continue
            
        df = df[df['premise_type'].isin(TYPOLOGIES)]
        
        for scn_name, acc in accumulators.items():
            if acc.cost_col in df.columns:
                acc.update(df[GROUP_COLS + [acc.cost_col]])
        
        del df
        gc.collect()

    print("\nAccumulation complete. Generating outputs for each scenario...")

    # 4. GENERATE OUTPUTS PER SCENARIO
    for scn_name, acc in accumulators.items():
        print(f"\n--- Processing results for: {scn_name} ---")
        
        detailed_df = acc.get_detailed_df()
        
        if detailed_df.empty:
            print(f"No data accumulated for {scn_name}. Skipping.")
            continue
        
        # ----------------------------------------------------------------------
        # NEW LOGIC: DETERMINE GROUPING BASED ON SCENARIO NAME
        # ----------------------------------------------------------------------
        is_wall_scenario = 'wall' in scn_name
        
        if is_wall_scenario:
            # Keep split by wall type
            group_keys_decile = ['avg_gas_percentile', 'inferred_insulation_type']
            group_keys_premise = ['premise_type', 'inferred_insulation_type']
        else:
            # Aggregate across all wall types (drop wall type from keys)
            group_keys_decile = ['avg_gas_percentile']
            group_keys_premise = ['premise_type']

        # --- A. Decile Plot ---
        df_decile = aggregate_and_calculate_stats(detailed_df, group_keys_decile)
        
        # If we didn't split by wall, add the column back as a dummy value so plotting works
        if not is_wall_scenario:
            df_decile['inferred_insulation_type'] = 'All'

        csv_path = os.path.join(OUTPUT_BASE, f'{scn_name}_stats_by_decile.csv')
        df_decile.to_csv(csv_path, index=False)
        
        plot_costs_by_decile(
            df_decile, 
            scn_name,
            os.path.join(OUTPUT_BASE, f'{scn_name}_costs_by_decile.png')
        )

        # --- B. Premise Type Plot ---
        df_premise = aggregate_and_calculate_stats(detailed_df, group_keys_premise)
        
        # If we didn't split by wall, add the column back as a dummy value
        if not is_wall_scenario:
            df_premise['inferred_insulation_type'] = 'All'

        csv_path = os.path.join(OUTPUT_BASE, f'{scn_name}_stats_by_premise.csv')
        df_premise.to_csv(csv_path, index=False)
        
        plot_costs_by_premise_type(
            df_premise, 
            scn_name,
            os.path.join(OUTPUT_BASE, f'{scn_name}_costs_by_premise.png')
        )
    
    print("\nDone. All scenario outputs saved.")

if __name__ == "__main__":
    run_pipeline()