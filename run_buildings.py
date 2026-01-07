import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob
import gc
import csv
from src.utils import is_running_on_hpc
from tqdm import tqdm
# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

 
SCENARIOS = [ 'wall_installation',  'heat_pump_only', 'loft_installation'] 

METRIC_PATTERNS = {
    'cost': {
        'pattern': '{sc}_cost_{sc}_mean', 
        'label': 'Installation Cost (£)',
        'type': 'cost'
    },
    'net_co2': {
        'pattern': '{sc}_total_energy_abs_co2_ton_samples_{sc}_p50', 
        'label': 'Net CO2 Removal (Tons/5 years)',
        'type': 'co2'
    },
    'gas_co2': {
        'pattern': '{sc}_gas_abs_ton_co2_samples_{sc}_p50', 
        'label': 'Gas: CO2 Removal (Tons/5 years)',
        'type': 'co2'
    },
    'elec_co2': {
        'pattern': '{sc}_elec_abs_ton_co2_samples_{sc}_p50', 
        'label': 'Elec: CO2 Removal (Tons/5 years)',
        'type': 'co2'
    }
}

OUTPUT_BASE = '3_stock_results/buildings_test/'
os.makedirs(OUTPUT_BASE, exist_ok=True)

is_hpc = is_running_on_hpc() 
if is_hpc:
    LOG_DIR = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/*csv'
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/130_log_file.csv'
else:
    LOG_DIR = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*.csv'
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
    def __init__(self, scenario_name, metric_key, target_col):
        self.scenario = scenario_name
        self.metric_key = metric_key
        self.target_col = target_col
        self.data = {}

    def update(self, df):
        if self.target_col not in df.columns:
            return

        if 'upn' not in df.columns:
            df['upn'] = df.index 

        df_subset = df.dropna(subset=GROUP_COLS + [self.target_col])
        if df_subset.empty:
            return

        grouped = df_subset.groupby(GROUP_COLS)[[self.target_col, 'upn']]
        
        for name, group in grouped:
            values = group[self.target_col].tolist()
            upns = group['upn'].tolist()
            pairs = list(zip(values, upns))
            
            if name not in self.data:
                self.data[name] = []
            
            self.data[name].extend(pairs)

    def get_raw_data(self):
        return self.data

# ==============================================================================
# 3. HELPER: AGGREGATE RAW DATA
# ==============================================================================
def compute_grouped_stats(raw_data_dict, group_indices, col_names):
    """
    Flattens the raw dictionary keys based on the indices requested.
    e.g. if key is (Decile, Premise, Wall) and we want Decile only,
    we pass group_indices=[0].
    """
    merged_data = {}
    
    for key_tuple, pairs in raw_data_dict.items():
        # Extract only the parts of the key we want (e.g., just Decile)
        new_key = tuple(key_tuple[i] for i in group_indices)
        if new_key not in merged_data:
            merged_data[new_key] = []
        merged_data[new_key].extend(pairs)

    rows = []
    for key, pairs in merged_data.items():
        if not pairs: 
            continue
            
        vals, upns = zip(*pairs)
        arr = np.array(vals)
        
        row_dict = dict(zip(col_names, key))
        row_dict.update({
            'median_val': np.median(arr),
            'p5': np.percentile(arr, 5),
            'p95': np.percentile(arr, 95),
            'count': len(set(upns))
        })
        rows.append(row_dict)
        
    return pd.DataFrame(rows)

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================

# --- A. Standard Single-Metric Plots ---
def plot_metric_by_decile(df, scenario_name, metric_label, output_path):
    if df.empty: return
    
    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    df = df.sort_values(by=['inferred_insulation_type', 'decile_numeric'])
    
    wall_types = sorted(df['inferred_insulation_type'].unique())
    deciles = sorted(df['decile_numeric'].dropna().unique())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(wall_types)))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('decile_numeric').reindex(deciles).reset_index()
        
        ax.plot(subset['decile_numeric'], subset['median_val'], 
                label=w_type, color=colors[i], marker='o', linewidth=2)
        ax.fill_between(subset['decile_numeric'], subset['p5'], subset['p95'], 
                        color=colors[i], alpha=0.2, linewidth=0)

    ax.set_xlabel('Gas Usage Decile')
    ax.set_ylabel(f'Median {metric_label}')
    
    ax.set_xticks(deciles)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        handles.append(Patch(facecolor='grey', alpha=0.2, label='p5 - p95 Range'))
        ax.legend(handles=handles, title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_metric_by_premise(df, scenario_name, metric_label, output_path):
    if df.empty: return

    df['premise_type'] = pd.Categorical(df['premise_type'], categories=TYPOLOGIES, ordered=True)
    df = df.sort_values(by=['inferred_insulation_type', 'premise_type'])
    
    present_types = [t for t in TYPOLOGIES if t in df['premise_type'].unique()]
    wall_types = sorted(df['inferred_insulation_type'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(wall_types)))
    x_indices = np.arange(len(present_types))
    
    for i, w_type in enumerate(wall_types):
        subset = df[df['inferred_insulation_type'] == w_type]
        subset = subset.set_index('premise_type').reindex(present_types).reset_index()
        
        ax.plot(x_indices, subset['median_val'], label=w_type, color=colors[i], marker='o')
        ax.fill_between(x_indices, subset['p5'], subset['p95'], color=colors[i], alpha=0.15, linewidth=0)

    ax.set_xlabel('Premise Type')
    ax.set_ylabel(f'Median {metric_label}')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(present_types, rotation=45, ha='right')
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        handles.append(Patch(facecolor='grey', alpha=0.2, label='p5 - p95 Range'))
        ax.legend(handles=handles, title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# --- B. NEW Comparison Plots (Net vs Gas vs Elec) ---

def plot_co2_compare_decile_x(stats_dict, scenario_name, output_base):
    """
    X-Axis: Gas Decile.
    Series: Net CO2, Gas CO2, Elec CO2.
    """
    metrics_cfg = {
        'net_co2':  {'label': 'Net CO2',  'color': 'black', 'style': '-'},
        'gas_co2':  {'label': 'Gas CO2',  'color': 'blue',   'style': '--'},
        'elec_co2': {'label': 'Elec CO2', 'color': 'red',  'style': ':'}
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    has_data = False
    
    for m_key, cfg in metrics_cfg.items():
        if m_key not in stats_dict: continue
        df = stats_dict[m_key]
        if df.empty: continue
        has_data = True
        
        df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
        df = df.sort_values('decile_numeric')
        
        ax.plot(df['decile_numeric'], df['median_val'], 
                label=cfg['label'], color=cfg['color'], linestyle=cfg['style'], marker='o')
        ax.fill_between(df['decile_numeric'], df['p5'], df['p95'], 
                        color=cfg['color'], alpha=0.1, linewidth=0)

    if has_data:
        
        ax.set_xlabel('Gas Usage Decile')
        ax.set_ylabel('Median CO2 Removal (Tons/5 years)')
        ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            handles.append(Patch(facecolor='grey', alpha=0.2, label='p5 - p95 Range'))
            ax.legend(handles=handles, title='Wall Type')
        
        ax.grid(True, alpha=0.3)
        
        out_name = f'{scenario_name}_COMPARE_X_Decile.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, out_name), dpi=300)
        print(f"Saved: {out_name}")
    plt.close()

def plot_co2_compare_premise_x(stats_dict, scenario_name, output_base):
    """
    X-Axis: Premise Type.
    Series: Net CO2, Gas CO2, Elec CO2.
    """
    metrics_cfg = {
        'net_co2':  {'label': 'Net CO2',  'color': 'black', 'style': '-'},
        'gas_co2':  {'label': 'Gas CO2',  'color': 'blue',   'style': '--'},
        'elec_co2': {'label': 'Elec CO2', 'color': 'red', 'style': ':'}
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    has_data = False
    
    # Use global typologies to ensure consistent X-axis order
    x_map = {t: i for i, t in enumerate(TYPOLOGIES)}
    
    for m_key, cfg in metrics_cfg.items():
        if m_key not in stats_dict: continue
        df = stats_dict[m_key]
        if df.empty: continue
        has_data = True
        
        # Categorical sort
        df['premise_type'] = pd.Categorical(df['premise_type'], categories=TYPOLOGIES, ordered=True)
        df = df.sort_values('premise_type').dropna(subset=['premise_type'])
        
        # Map premise strings to integer indices for plotting lines
        x_vals = [x_map[t] for t in df['premise_type']]
        
        ax.plot(x_vals, df['median_val'], 
                label=cfg['label'], color=cfg['color'], linestyle=cfg['style'], marker='s')
        ax.fill_between(x_vals, df['p5'], df['p95'], 
                        color=cfg['color'], alpha=0.1, linewidth=0)

    if has_data:
        
        ax.set_ylabel('Median CO2 Removal (Tons/5 years)')
        ax.set_xticks(range(len(TYPOLOGIES)))
        ax.set_xticklabels(TYPOLOGIES, rotation=45, ha='right')
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            handles.append(Patch(facecolor='grey', alpha=0.2, label='p5 - p95 Range'))
            ax.legend(handles=handles, title='Wall Type')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.05)
        
        out_name = f'{scenario_name}_COMPARE_X_Premise.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, out_name), dpi=300)
        print(f"Saved: {out_name}")
    plt.close()

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

    # Initialize accumulators
    accumulators = {}
    for scn in SCENARIOS:
        for m_key, m_cfg in METRIC_PATTERNS.items():
            col_name = m_cfg['pattern'].format(sc=scn)
            accumulators[(scn, m_key)] = GroupedStatsAccumulator(scn, m_key, col_name)
    print('Loading raw data') 
    # 1. COLLECT RAW DATA
    for i, file_path in enumerate(files):
        if i % 10 == 0: print(f"Processing file {i}/{len(files)}...")
        
        df = safe_load(file_path, headers)
        if df.empty: continue
        
        # Ensure required columns exist
        if not set(GROUP_COLS).issubset(df.columns):
            continue
            
        df = df[df['premise_type'].isin(TYPOLOGIES)]
        
        for acc in accumulators.values():
            acc.update(df)
        
        del df
        gc.collect()

    print("\nAccumulation complete. Generating outputs...")

    # 2. GENERATE PLOTS
    for scn in SCENARIOS:
        print(f"\n=== Processing Scenario: {scn} ===")
        
        # Containers for the Comparison Plots
        # These will store simplified dataframes (flattened wall types)
        decile_compare_collection = {}
        premise_compare_collection = {}

        for m_key, m_cfg in METRIC_PATTERNS.items():
            acc = accumulators.get((scn, m_key))
            if not acc: continue
            
            raw_data = acc.get_raw_data()
            if not raw_data: continue
            
            # Identify if this scenario has specific wall types (requires index 2)
            is_wall_scenario = 'wall' in scn
            
            # --- A. DECILE PROCESSING ---
            # 1. Standard Plot (Splits by Wall Type if applicable)
            idx_std = [0, 2] if is_wall_scenario else [0]
            cols_std = ['avg_gas_percentile', 'inferred_insulation_type'] if is_wall_scenario else ['avg_gas_percentile']
            df_std = compute_grouped_stats(raw_data, idx_std, cols_std)
            if not is_wall_scenario: df_std['inferred_insulation_type'] = 'All'
            
            plot_metric_by_decile(
                df_std, scn, m_cfg['label'],
                os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_decile.png')
            )
            
            # 2. Comparison Prep (Flatten Wall Types -> One line per metric)
            # We strictly group by Decile [0] only
            df_compare_dec = compute_grouped_stats(raw_data, [0], ['avg_gas_percentile'])
            decile_compare_collection[m_key] = df_compare_dec

            # --- B. PREMISE PROCESSING ---
            # 1. Standard Plot
            idx_std = [1, 2] if is_wall_scenario else [1]
            cols_std = ['premise_type', 'inferred_insulation_type'] if is_wall_scenario else ['premise_type']
            df_std = compute_grouped_stats(raw_data, idx_std, cols_std)
            if not is_wall_scenario: df_std['inferred_insulation_type'] = 'All'

            plot_metric_by_premise(
                df_std, scn, m_cfg['label'],
                os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_premise.png')
            )
            
            # 2. Comparison Prep (Flatten Wall Types)
            # We strictly group by Premise [1] only
            df_compare_prem = compute_grouped_stats(raw_data, [1], ['premise_type'])
            premise_compare_collection[m_key] = df_compare_prem

        # --- C. EXECUTE COMPARISON PLOTS ---
        # Generate the two requested comparison plots for this scenario
        print(f"  Generating Comparison Plots for {scn}...")
        plot_co2_compare_decile_x(decile_compare_collection, scn, OUTPUT_BASE)
        plot_co2_compare_premise_x(premise_compare_collection, scn, OUTPUT_BASE)
        
    print("\nDone. All scenario outputs saved.")

if __name__ == "__main__":
    run_pipeline()