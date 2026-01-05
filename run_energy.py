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

# Column Patterns (Dynamic based on scenario name {sc})
CO2_METRIC_PATTERNS = {
    'net_co2': {
        'pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_p50',
        'label': 'Net CO2 Removal (Tons/5 years)',
    },
    'gas_co2': {
        'pattern': '{sc}_gas_abs_ton_co2_samples_{sc}_p50',
        'label': 'Gas: CO2 Removal (Tons/5 years)',
    },
    'elec_co2': {
        'pattern': '{sc}_elec_abs_ton_co2_samples_{sc}_p50',
        'label': 'Elec: CO2 Removal (Tons/5 years)',
    }
}

OUTPUT_BASE = '3_stock_results/epistemic'
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Determine environment and paths
is_hpc = is_running_on_hpc() 
if is_hpc:
    LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/*csv'
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/130_log_file.csv'
else:
    # Local Test Path
    LOG_DIR = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*.csv'
    REFERENCE_FILE = None

# Grouping Keys (Raw Data Storage)
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
# 2. DATA ACCUMULATOR CLASS (GENERIC)
# ==============================================================================
class GroupedStatsAccumulator:
    def __init__(self, scenario_name, metric_name, target_col):
        """
        metric_name: e.g., 'cost', 'net_co2' (used for file naming)
        target_col: The actual column name in the CSV to read.
        """
        self.scenario = scenario_name
        self.metric = metric_name
        self.target_col = target_col
        # Dictionary to store raw lists
        # Key: (gas, premise, wall) -> Value: [(value, upn, run_id), ...]
        self.data = {}

    def update(self, df):
        # Skip if this specific column doesn't exist in the current file
        if self.target_col not in df.columns:
            return

        # Ensure necessary columns exist; fill defaults if missing
        if 'upn' not in df.columns: 
            df['upn'] = df.index 
        if 'epistemic_run_id' not in df.columns: 
            df['epistemic_run_id'] = 0 

        df_subset = df.dropna(subset=GROUP_COLS + [self.target_col])
        if df_subset.empty:
            return

        # Extract relevant columns
        grouped = df_subset.groupby(GROUP_COLS)[[self.target_col, 'upn', 'epistemic_run_id']]
        
        for name, group in grouped:
            values = group[self.target_col].tolist()
            upns = group['upn'].tolist()
            runs = group['epistemic_run_id'].tolist()
            
            # Store as list of tuples: (value, upn, run_id)
            triplets = list(zip(values, upns, runs))
            
            if name not in self.data:
                self.data[name] = []
            
            self.data[name].extend(triplets)

    def get_raw_data(self):
        return self.data

# ==============================================================================
# 3. STATS CALCULATION (EPISTEMIC / RUN LEVEL)
# ==============================================================================
def compute_epistemic_stats(raw_data_dict, group_indices, col_names):
    """
    1. Groups raw data by `epistemic_run_id`.
    2. Calculates the MEDIAN Value per Run.
    3. Calculates Variance (P5/P95) across those Run Medians.
    """
    merged_data = {}
    
    # 1. Merge raw data based on the specific plot grouping
    for key_tuple, triplets in raw_data_dict.items():
        new_key = tuple(key_tuple[i] for i in group_indices)
        if new_key not in merged_data:
            merged_data[new_key] = []
        merged_data[new_key].extend(triplets)

    rows = []
    for key, triplets in merged_data.items():
        if not triplets: continue
        
        # triplets structure: [(value, upn, run_id), ...]
        df_temp = pd.DataFrame(triplets, columns=['value', 'upn', 'run_id'])
        
        # --- KEY LOGIC ---
        # 1. Calculate MEDIAN Value per Run ID
        run_medians = df_temp.groupby('run_id')['value'].median()
        
        # 2. Calculate Stats across these Medians
        arr = run_medians.values
        
        median_of_medians = np.median(arr)
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)
        
        row_dict = dict(zip(col_names, key))
        row_dict.update({
            'median_val': median_of_medians,
            'p5': p5,
            'p95': p95,
            'count': df_temp['upn'].nunique()
        })
        rows.append(row_dict)
        
    return pd.DataFrame(rows)

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================
def plot_stats_by_decile(df, scenario_name, metric_name, y_label, output_path):
    if df.empty: return

    clean_name = SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)
    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    df = df.sort_values(by=['inferred_insulation_type', 'decile_numeric'])
    
    wall_types = sorted(df['inferred_insulation_type'].unique())
    deciles = sorted(df['decile_numeric'].dropna().unique())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    n_types = len(wall_types)
    if n_types == 0: plt.close(); return

    bar_width = 0.6 if n_types == 1 else 0.8 / n_types
    x_pos = np.arange(len(deciles))
    colors = plt.cm.viridis(np.linspace(0, 1, n_types))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('decile_numeric').reindex(deciles).reset_index()
        
        offset = 0 if n_types == 1 else (i - n_types/2 + 0.5) * bar_width
        
        # Error Bars: Epistemic Variance (P5 to P95 of Run Medians)
        lower_err = subset['median_val'] - subset['p5']
        upper_err = subset['p95'] - subset['median_val']
        asymmetric_err = [lower_err.fillna(0), upper_err.fillna(0)]
        
        ax.bar(x_pos + offset, subset['median_val'], width=bar_width, 
               yerr=asymmetric_err, label=w_type, color=colors[i], capsize=3, alpha=0.8)

    ax.set_xlabel('Gas Usage Decile')
    ax.set_ylabel(f'Median {y_label}')
    ax.set_title(f'{clean_name} - {metric_name}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([int(d) for d in deciles])
    
    if n_types > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Decile Plot: {output_path}")

def plot_stats_by_premise_type(df, scenario_name, metric_name, y_label, output_path):
    if df.empty: return

    clean_name = SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)
    df['premise_type'] = pd.Categorical(df['premise_type'], categories=TYPOLOGIES, ordered=True)
    df = df.sort_values(by=['inferred_insulation_type', 'premise_type'])
    
    present_types = [t for t in TYPOLOGIES if t in df['premise_type'].unique()]
    wall_types = sorted(df['inferred_insulation_type'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 9))
    n_types = len(wall_types)
    if n_types == 0: plt.close(); return

    bar_width = 0.6 if n_types == 1 else 0.8 / n_types
    x_pos = np.arange(len(present_types))
    colors = plt.cm.viridis(np.linspace(0, 1, n_types))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('premise_type').reindex(present_types).reset_index()
        
        offset = 0 if n_types == 1 else (i - n_types/2 + 0.5) * bar_width
        
        # Error Bars: Epistemic Variance (P5 to P95 of Run Medians)
        lower_err = subset['median_val'] - subset['p5']
        upper_err = subset['p95'] - subset['median_val']
        asymmetric_err = [lower_err.fillna(0), upper_err.fillna(0)]
        
        ax.bar(x_pos + offset, subset['median_val'], width=bar_width, 
               yerr=asymmetric_err, label=w_type, color=colors[i], capsize=3, alpha=0.8)

    ax.set_xlabel('Premise Type')
    ax.set_ylabel(f'Median {y_label}')
    ax.set_title(f'{clean_name} - {metric_name}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(present_types, rotation=45, ha='right')
    ax.margins(y=0.15)
    
    if n_types > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Premise Plot: {output_path}")

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

    # ---------------------------------------------------------
    # Initialize Accumulators for Cost + Dynamic CO2 Metrics
    # ---------------------------------------------------------
    # Structure: accumulators[(scenario, metric_key)] = AccumulatorObject
    accumulators = {}

    for scn in SCENARIOS:
        # 1. Standard Cost Metric
        cost_col_name = f'{scn}_cost_{scn}_p50'
        accumulators[(scn, 'cost')] = GroupedStatsAccumulator(scn, 'cost', cost_col_name)

        # 2. CO2 Metrics (Dynamic column generation based on patterns)
        for key, config in CO2_METRIC_PATTERNS.items():
            # Inject scenario name into pattern: e.g. "{sc}_gas..." -> "loft_gas..."
            target_col = config['pattern'].format(sc=scn)
            accumulators[(scn, key)] = GroupedStatsAccumulator(scn, key, target_col)
    
    # ---------------------------------------------------------
    # 1. LOAD AND ACCUMULATE
    # ---------------------------------------------------------
    for i, file_path in enumerate(files):
        if i % 10 == 0: print(f"Processing file {i}/{len(files)}...")
        
        df = safe_load(file_path, headers)
        if df.empty: continue
        
        # Check required grouping columns
        req_cols = GROUP_COLS
        if not set(req_cols).issubset(df.columns):
            continue
            
        df = df[df['premise_type'].isin(TYPOLOGIES)]
        
        # Update all relevant accumulators for this file
        for (scn_name, metric_key), acc in accumulators.items():
            # The accumulator handles checking if its specific target_col exists
            acc.update(df)
        
        del df
        gc.collect()

    print("\nAccumulation complete. Generating Epistemic Variance plots...")

    # ---------------------------------------------------------
    # 2. COMPUTE STATS & PLOT
    # ---------------------------------------------------------
    for (scn_name, metric_key), acc in accumulators.items():
        # Get raw accumulated data
        raw_data = acc.get_raw_data()
        
        # Skip if no data found for this specific combo
        if not raw_data:
            # Only print if it's cost, otherwise it might be too verbose if some cols are missing
            if metric_key == 'cost': 
                print(f"Skipping {scn_name} - {metric_key} (No Data)")
            continue
        
        print(f"Processing results for: {scn_name} [{metric_key}]")

        is_wall_scenario = 'wall' in scn_name
        
        # Determine Label for Y-Axis
        if metric_key == 'cost':
            y_lbl = 'Installation Costs (£)'
        else:
            y_lbl = CO2_METRIC_PATTERNS[metric_key]['label']

        # --- A. Decile Plot (Epistemic) ---
        idx_decile = [0, 2] if is_wall_scenario else [0]
        col_decile = ['avg_gas_percentile', 'inferred_insulation_type'] if is_wall_scenario else ['avg_gas_percentile']
        
        df_decile = compute_epistemic_stats(raw_data, idx_decile, col_decile)
        
        if not is_wall_scenario: 
            df_decile['inferred_insulation_type'] = 'All'
            
        # Save CSV Stats
        df_decile.to_csv(os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_epistemic_stats_decile.csv'), index=False)
        
        # Plot
        plot_stats_by_decile(
            df_decile, 
            scn_name,
            metric_key,
            y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_epistemic_var_decile.png')
        )

        # --- B. Premise Type Plot (Epistemic) ---
        idx_premise = [1, 2] if is_wall_scenario else [1]
        col_premise = ['premise_type', 'inferred_insulation_type'] if is_wall_scenario else ['premise_type']

        df_premise = compute_epistemic_stats(raw_data, idx_premise, col_premise)
        
        if not is_wall_scenario: 
            df_premise['inferred_insulation_type'] = 'All'
            
        # Save CSV Stats
        df_premise.to_csv(os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_epistemic_stats_premise.csv'), index=False)
        
        # Plot
        plot_stats_by_premise_type(
            df_premise, 
            scn_name,
            metric_key,
            y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_epistemic_var_premise.png')
        )
    
    print("\nDone. All epistemic outputs saved.")

if __name__ == "__main__":
    run_pipeline()