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

# Column Patterns
# {sc} will be replaced by the scenario name
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
        'label': 'Gas: CO2 Changes (Tons/5 years)',
        'type': 'co2'
    },
    'elec_co2': {
        'pattern': '{sc}_elec_abs_ton_co2_samples_{sc}_p50', 
        'label': 'Elec: CO2 Changes (Tons/5 years)',
        'type': 'co2'
    }
}

OUTPUT_BASE = '3_stock_results/buildings/'
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

SCENARIO_DISPLAY_NAMES = {
    'loft_installation': 'Loft Installation',
    'wall_installation': 'Wall Insulation',
    'joint_heat_loft_decay': 'HP + Loft (Decay)',
    'joint_heat_wall_decay': 'HP + Wall (Decay)',
    'join_heat_ins_decay': 'HP + All Insulation (Decay)',
    'heat_pump_only': 'Heat Pump Only',
}

# ==============================================================================
# 2. THE GROUPED ACCUMULATOR CLASS (GENERIC)
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
            
        vals, upns = zip(*pairs)
        arr = np.array(vals)
        median_val = np.median(arr)
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)
        unique_count = len(set(upns))
        
        row_dict = dict(zip(col_names, key))
        row_dict.update({
            'median_val': median_val,
            'p5': p5,
            'p95': p95,
            'count': unique_count
        })
        rows.append(row_dict)
        
    return pd.DataFrame(rows)

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================
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
        
        x_vals = subset['decile_numeric']
        y_vals = subset['median_val']
        y_p5 = subset['p5']
        y_p95 = subset['p95']
        
        ax.plot(x_vals, y_vals, label=w_type, color=colors[i], marker='o', linewidth=2)
        ax.fill_between(x_vals, y_p5, y_p95, color=colors[i], alpha=0.2, linewidth=0)

    ax.set_xlabel('Gas Usage Decile')
    ax.set_ylabel(f'Median {metric_label}')
    
    ax.set_xticks(deciles)
    ax.set_xticklabels([int(d) for d in deciles])
    
    if len(wall_types) > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

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
        
        y_vals = subset['median_val']
        y_p5 = subset['p5']
        y_p95 = subset['p95']
        
        ax.plot(x_indices, y_vals, label=w_type, color=colors[i], marker='o', linewidth=2)
        ax.fill_between(x_indices, y_p5, y_p95, color=colors[i], alpha=0.15, linewidth=0)

    ax.set_xlabel('Premise Type')
    ax.set_ylabel(f'Median {metric_label}')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(present_types, rotation=45, ha='right')
    ax.margins(y=0.15)
    
    if len(wall_types) > 1 or wall_types[0] != 'All':
        ax.legend(title='Wall Type')
        
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_co2_comparison(stats_dict, scenario_name, output_base):
    """
    Plots Net, Gas, and Elec CO2 on the same chart (Decile view).
    Generates one plot per Wall Type (or just one if 'All').
    """
    # 1. Check if we have data for all three
    for m in ['net_co2', 'gas_co2', 'elec_co2']:
        if m not in stats_dict:
            return

    # 2. Get the dataframes
    df_net = stats_dict['net_co2'].copy()
    df_gas = stats_dict['gas_co2'].copy()
    df_elec = stats_dict['elec_co2'].copy()

    # 3. Align and Prepare
    for df in [df_net, df_gas, df_elec]:
        df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')

    # Get unique wall types present in Net (assuming others match)
    wall_types = sorted(df_net['inferred_insulation_type'].unique())
    deciles = sorted(df_net['decile_numeric'].dropna().unique())

    # Map for colors/labels
    metrics_cfg = {
        'Net CO2': {'df': df_net, 'color': 'black', 'style': '-'},
        'Gas CO2': {'df': df_gas, 'color': 'red', 'style': '--'},
        'Elec CO2': {'df': df_elec, 'color': 'blue', 'style': '--'}
    }

    # 4. Generate one plot per wall type
    for w_type in wall_types:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for label, cfg in metrics_cfg.items():
            df_curr = cfg['df']
            subset = df_curr[df_curr['inferred_insulation_type'] == w_type]
            subset = subset.set_index('decile_numeric').reindex(deciles).reset_index()

            x_vals = subset['decile_numeric']
            y_vals = subset['median_val']
            y_p5 = subset['p5']
            y_p95 = subset['p95']

            ax.plot(x_vals, y_vals, label=label, color=cfg['color'], linestyle=cfg['style'], marker='o')
            ax.fill_between(x_vals, y_p5, y_p95, color=cfg['color'], alpha=0.1, linewidth=0)

        
        ax.set_xlabel('Gas Usage Decile')
        ax.set_ylabel('CO2 Reduction (Tons)')
        ax.set_xticks(deciles)
        ax.set_xticklabels([int(d) for d in deciles])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        safe_w_type = w_type.replace(' ', '_').replace('/', '_')
        out_name = f'{scenario_name}_CO2_COMPARE_{safe_w_type}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, out_name), dpi=300)
        plt.close()
        print(f"Saved Comparison: {out_name}")

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
    files=files[0:5]
    print(f"Found {len(files)} files.")
    
    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
        except:
            print("Warning: Could not read headers.")

    # Initialize accumulators: (scenario, metric_key) -> Accumulator
    accumulators = {}
    for scn in SCENARIOS:
        for m_key, m_cfg in METRIC_PATTERNS.items():
            # Dynamic column name generation
            col_name = m_cfg['pattern'].format(sc=scn)
            accumulators[(scn, m_key)] = GroupedStatsAccumulator(scn, m_key, col_name)
    
    # 1. COLLECT RAW DATA
    for i, file_path in enumerate(files):
        if i % 10 == 0: print(f"Processing file {i}/{len(files)}...")
        
        df = safe_load(file_path, headers)
        if df.empty: continue
        
        if not set(GROUP_COLS).issubset(df.columns):
            continue
            
        df = df[df['premise_type'].isin(TYPOLOGIES)]
        
        # Update all accumulators
        for acc in accumulators.values():
            acc.update(df)
        
        del df
        gc.collect()

    print("\nAccumulation complete. Generating outputs...")

    # 2. CALCULATE STATS AND GENERATE PLOTS
    # We will group results by scenario to allow for the Comparison Plot at the end of each scenario loop
    for scn in SCENARIOS:
        print(f"\n=== Processing Scenario: {scn} ===")
        
        # Store calculated stats for this scenario here to use in the combined plot later
        # Key: metric_key (e.g. 'net_co2'), Value: DataFrame (Decile aggregated)
        scenario_decile_stats = {} 

        # Loop through metrics: Cost, Net CO2, Gas CO2, Elec CO2
        for m_key, m_cfg in METRIC_PATTERNS.items():
            acc = accumulators.get((scn, m_key))
            if not acc: continue
            
            raw_data = acc.get_raw_data()
            if not raw_data:
                if m_key == 'cost': print(f"  No data for {m_key}")
                continue
                
            is_wall_scenario = 'wall' in scn
            
            # --- A. Decile Plot ---
            idx_decile = [0, 2] if is_wall_scenario else [0]
            cols_decile = ['avg_gas_percentile', 'inferred_insulation_type'] if is_wall_scenario else ['avg_gas_percentile']
            
            df_decile = compute_grouped_stats(raw_data, idx_decile, cols_decile)
            if not is_wall_scenario: df_decile['inferred_insulation_type'] = 'All'
            
            # Save CSV
            df_decile.to_csv(os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_stats_decile.csv'), index=False)
            
            # Plot
            plot_metric_by_decile(
                df_decile, scn, m_cfg['label'],
                os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_decile.png')
            )
            
            # Store for combined plot
            scenario_decile_stats[m_key] = df_decile

            # --- B. Premise Plot ---
            idx_premise = [1, 2] if is_wall_scenario else [1]
            cols_premise = ['premise_type', 'inferred_insulation_type'] if is_wall_scenario else ['premise_type']
            
            df_premise = compute_grouped_stats(raw_data, idx_premise, cols_premise)
            if not is_wall_scenario: df_premise['inferred_insulation_type'] = 'All'

            # Save CSV
            df_premise.to_csv(os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_stats_premise.csv'), index=False)
            
            # Plot
            plot_metric_by_premise(
                df_premise, scn, m_cfg['label'],
                os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_premise.png')
            )

        # --- C. GENERATE COMBINED CO2 PLOT ---
        # Only if we have the necessary data for this scenario
        print(f"  Generating CO2 Comparison Plot for {scn}...")
        plot_co2_comparison(scenario_decile_stats, scn, OUTPUT_BASE)
    
    print("\nDone. All scenario outputs saved.")

if __name__ == "__main__":
    run_pipeline()