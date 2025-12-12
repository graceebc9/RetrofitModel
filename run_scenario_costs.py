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
    LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/*csv'
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/130_log_file.csv'
else:
    # Example local path
    LOG_DIR = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*.csv'
    REFERENCE_FILE = None

# We use these keys to store the raw data initially
# Index 0: gas, 1: premise, 2: wall
GROUP_COLS = ['avg_gas_percentile', 'premise_type', 'inferred_insulation_type']

TYPOLOGIES = [
    'Small low terraces', '3-4 storey and smaller flats',
    'Tall terraces 3-4 storeys', 'Large semi detached', 'Standard size detached',
    'Standard size semi detached', '2 storeys terraces with t rear extension',
    'Semi type house in multiples', 'Large detached',
    'Planned balanced mixed estates',
    'Linked and step linked premises',
]

SCENARIO_DISPLAY_NAMES = {
    'loft_installation': 'Loft Installation',
    'wall_installation': 'Wall Insulation',
    'joint_heat_loft_decay': 'HP + Loft (Decay)',
    'joint_heat_wall_decay': 'HP + Wall (Decay)',
    'join_heat_ins_decay': 'HP + All Insulation (Decay)',
    'heat_pump_only': 'Heat Pump Only',
}

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

        if 'upn' not in df.columns:
            df['upn'] = df.index 

        df_subset = df.dropna(subset=GROUP_COLS + [self.cost_col])
        if df_subset.empty:
            return

        grouped = df_subset.groupby(GROUP_COLS)[[self.cost_col, 'upn']]
        
        for name, group in grouped:
            costs = group[self.cost_col].tolist()
            upns = group['upn'].tolist()
            pairs = list(zip(costs, upns))
            
            if name not in self.data:
                self.data[name] = []
            
            self.data[name].extend(pairs)

    def get_raw_data(self):
        return self.data

# ==============================================================================
# 3. HELPER: AGGREGATE RAW DATA & CALCULATE STATS
# ==============================================================================
def compute_grouped_stats(raw_data_dict, group_indices, col_names):
    merged_data = {}
    
    for key_tuple, pairs in raw_data_dict.items():
        new_key = tuple(key_tuple[i] for i in group_indices)
        if new_key not in merged_data:
            merged_data[new_key] = []
        merged_data[new_key].extend(pairs)

    rows = []
    for key, pairs in merged_data.items():
        if not pairs: 
            continue
            
        costs, upns = zip(*pairs)
        arr = np.array(costs)
        median_val = np.median(arr)
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)
        unique_count = len(set(upns))
        
        row_dict = dict(zip(col_names, key))
        row_dict.update({
            'median_cost': median_val,
            'p5': p5,
            'p95': p95,
            'count': unique_count
        })
        rows.append(row_dict)
        
    return pd.DataFrame(rows)

# ==============================================================================
# 4. PLOTTING FUNCTIONS (UPDATED TO LINE CHARTS)
# ==============================================================================
def plot_costs_by_decile(df, scenario_name, output_path):
    """
    Updated: Line Chart with Shaded Error Bands (Fill Between).
    """
    if df.empty: return

    # clean_name = SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)
    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    df = df.sort_values(by=['inferred_insulation_type', 'decile_numeric'])
    
    wall_types = sorted(df['inferred_insulation_type'].unique())
    deciles = sorted(df['decile_numeric'].dropna().unique())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(wall_types)))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        # Ensure we have a continuous index for plotting, though for deciles we use the value directly
        subset = subset.set_index('decile_numeric').reindex(deciles).reset_index()
        
        # Data for plotting
        x_vals = subset['decile_numeric']
        y_vals = subset['median_cost']
        y_p5 = subset['p5']
        y_p95 = subset['p95']
        
        # Plot Median Line
        ax.plot(x_vals, y_vals, label=w_type, color=colors[i], marker='o', linewidth=2)
        
        # Plot Error Band (Shaded Area)
        ax.fill_between(x_vals, y_p5, y_p95, color=colors[i], alpha=0.2, linewidth=0)

    ax.set_xlabel('Gas Usage Decile')
    ax.set_ylabel('Median Installation Costs (£)')
    
    # Ensure all integers appear on x-axis
    ax.set_xticks(deciles)
    ax.set_xticklabels([int(d) for d in deciles])
    
    if len(wall_types) > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.1) # Light grid for x-axis helps line charts
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_costs_by_premise_type(df, scenario_name, output_path):
    """
    Updated: Line Chart with Shaded Error Bands.
    Note: Premise types are categorical, so we map them to integer indices for the x-axis.
    """
    if df.empty: return

    # clean_name = SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)
    df['premise_type'] = pd.Categorical(df['premise_type'], categories=TYPOLOGIES, ordered=True)
    df = df.sort_values(by=['inferred_insulation_type', 'premise_type'])
    
    present_types = [t for t in TYPOLOGIES if t in df['premise_type'].unique()]
    wall_types = sorted(df['inferred_insulation_type'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(wall_types)))
    x_indices = np.arange(len(present_types)) # Convert categories to 0, 1, 2...
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('premise_type').reindex(present_types).reset_index()
        
        # Data
        y_vals = subset['median_cost']
        y_p5 = subset['p5']
        y_p95 = subset['p95']
        
        # Plot Median Line
        ax.plot(x_indices, y_vals, label=w_type, color=colors[i], marker='o', linewidth=2)
        
        # Plot Error Band
        ax.fill_between(x_indices, y_p5, y_p95, color=colors[i], alpha=0.15, linewidth=0)

        # Add Count Annotations
        # Only add annotations if it's the "top" layer or if counts differ significantly
        # Here we add them for every point, slightly offset if needed
        for x_idx, row_idx in enumerate(subset.index):
            row = subset.loc[row_idx]
            if pd.isna(row['median_cost']): continue
            
            y_top = row['p95']
            count_n = row['count']
            
            if pd.notna(y_top) and pd.notna(count_n):
                # We place the text slightly above the p95 band
                ax.text(x_indices[x_idx], y_top * 1.05, f"n={int(count_n)}", 
                        rotation=0, ha='center', va='bottom', fontsize=8, color='black')

    ax.set_xlabel('Premise Type')
    ax.set_ylabel('Median Installation Costs (£)')
    
    # Map the integers back to strings for the X-ticks
    ax.set_xticks(x_indices)
    ax.set_xticklabels(present_types, rotation=45, ha='right')
    
    ax.margins(y=0.15)
    
    if len(wall_types) > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.1)
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
    print(f"Scanning: {LOG_DIR}")
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
    
    # 1. COLLECT RAW DATA
    for i, file_path in enumerate(files):
        if i % 10 == 0: print(f"Processing file {i}/{len(files)}...")
        
        df = safe_load(file_path, headers)
        if df.empty: continue
        
        if not set(GROUP_COLS).issubset(df.columns):
            continue
            
        df = df[df['premise_type'].isin(TYPOLOGIES)]
        
        for scn_name, acc in accumulators.items():
            if acc.cost_col in df.columns:
                acc.update(df[GROUP_COLS + [acc.cost_col, 'upn']])
        
        del df
        gc.collect()

    print("\nAccumulation complete. Generating Median outputs...")

    # 2. CALCULATE STATS AND GENERATE PLOTS
    for scn_name, acc in accumulators.items():
        print(f"\n--- Processing results for: {scn_name} ---")
        
        raw_data = acc.get_raw_data()
        if not raw_data:
            print(f"No data accumulated for {scn_name}. Skipping.")
            continue
        
        is_wall_scenario = 'wall' in scn_name
        
        # --- A. Decile Plot (No Counts) ---
        if is_wall_scenario:
            indices = [0, 2] # Gas, Wall
            col_names = ['avg_gas_percentile', 'inferred_insulation_type']
        else:
            indices = [0]    # Gas
            col_names = ['avg_gas_percentile']

        df_decile = compute_grouped_stats(raw_data, indices, col_names)
        
        if not is_wall_scenario:
            df_decile['inferred_insulation_type'] = 'All'

        csv_path = os.path.join(OUTPUT_BASE, f'{scn_name}_stats_by_decile.csv')
        df_decile.to_csv(csv_path, index=False)
        
        plot_costs_by_decile(
            df_decile, 
            scn_name,
            os.path.join(OUTPUT_BASE, f'{scn_name}_median_costs_by_decile.png')
        )

        # --- B. Premise Type Plot (WITH Counts) ---
        if is_wall_scenario:
            indices = [1, 2] # Premise, Wall
            col_names = ['premise_type', 'inferred_insulation_type']
        else:
            indices = [1]    # Premise
            col_names = ['premise_type']

        df_premise = compute_grouped_stats(raw_data, indices, col_names)
        
        if not is_wall_scenario:
            df_premise['inferred_insulation_type'] = 'All'

        csv_path = os.path.join(OUTPUT_BASE, f'{scn_name}_stats_by_premise.csv')
        df_premise.to_csv(csv_path, index=False)
        
        plot_costs_by_premise_type(
            df_premise, 
            scn_name,
            os.path.join(OUTPUT_BASE, f'{scn_name}_median_costs_by_premise.png')
        )
    
    print("\nDone. All scenario outputs saved.")

if __name__ == "__main__":
    run_pipeline()