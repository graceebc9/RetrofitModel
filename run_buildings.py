import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob
import gc
import csv
import pickle
from src.utils import is_running_on_hpc
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

SCENARIOS = ['wall_installation', 'heat_pump_only', 'loft_installation']

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

CHECKPOINT_PATH = os.path.join(OUTPUT_BASE, 'checkpoint.pkl')
CHECKPOINT_INTERVAL = 50

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
# 2. BUILD COLUMN LIST (for usecols optimization)
# ==============================================================================

def get_all_metric_columns():
    """Build list of all metric columns we need across all scenarios."""
    cols = set()
    for scn in SCENARIOS:
        for m_key, m_cfg in METRIC_PATTERNS.items():
            cols.add(m_cfg['pattern'].format(sc=scn))
    return list(cols)

def get_required_columns():
    """All columns needed for processing."""
    base_cols = GROUP_COLS + ['upn']
    metric_cols = get_all_metric_columns()
    return base_cols + metric_cols

# ==============================================================================
# 3. SIMPLIFIED ACCUMULATOR (no groupby - just stores data)
# ==============================================================================

class GroupedStatsAccumulator:
    def __init__(self, scenario_name, metric_key, target_col):
        self.scenario = scenario_name
        self.metric_key = metric_key
        self.target_col = target_col
        self.data = {}

    def add_group_data(self, group_key, values, upns):
        """Add pre-grouped data directly."""
        pairs = list(zip(values, upns))
        if group_key not in self.data:
            self.data[group_key] = []
        self.data[group_key].extend(pairs)

    def get_raw_data(self):
        return self.data

# ==============================================================================
# 4. SINGLE GROUPBY PROCESSOR
# ==============================================================================

def process_file_single_groupby(df, accumulators, metric_col_map):
    """
    Do ONE groupby, then distribute values to all relevant accumulators.
    
    metric_col_map: {(scenario, metric_key): column_name}
    """
    if 'upn' not in df.columns:
        df['upn'] = df.index

    # Filter to valid rows (all group cols present)
    df_valid = df.dropna(subset=GROUP_COLS)
    if df_valid.empty:
        return

    # Single groupby
    grouped = df_valid.groupby(GROUP_COLS)

    for group_key, group_df in grouped:
        upns = group_df['upn'].tolist()
        
        # Distribute to each accumulator
        for (scn, m_key), col_name in metric_col_map.items():
            if col_name not in group_df.columns:
                continue
            
            # Get non-null values for this metric
            valid_mask = group_df[col_name].notna()
            if not valid_mask.any():
                continue
                
            values = group_df.loc[valid_mask, col_name].tolist()
            valid_upns = group_df.loc[valid_mask, 'upn'].tolist()
            
            acc = accumulators.get((scn, m_key))
            if acc:
                acc.add_group_data(group_key, values, valid_upns)

# ==============================================================================
# 5. CHECKPOINTING
# ==============================================================================

def save_checkpoint(accumulators, processed_files):
    """Save current state to checkpoint file."""
    checkpoint_data = {
        'accumulator_data': {
            key: acc.get_raw_data() 
            for key, acc in accumulators.items()
        },
        'processed_files': processed_files
    }
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"  [Checkpoint saved: {len(processed_files)} files processed]")

def load_checkpoint(accumulators):
    """Load checkpoint if exists. Returns set of processed files."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    
    print(f"Found checkpoint, resuming...")
    with open(CHECKPOINT_PATH, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Restore accumulator data
    for key, data in checkpoint_data['accumulator_data'].items():
        if key in accumulators:
            accumulators[key].data = data
    
    processed = checkpoint_data['processed_files']
    print(f"  Restored {len(processed)} processed files from checkpoint")
    return processed

def delete_checkpoint():
    """Remove checkpoint file on successful completion."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint deleted.")

# ==============================================================================
# 6. HELPER: AGGREGATE RAW DATA
# ==============================================================================

def compute_grouped_stats(raw_data_dict, group_indices, col_names):
    """
    Flattens the raw dictionary keys based on the indices requested.
    """
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
# 7. PLOTTING FUNCTIONS (unchanged)
# ==============================================================================

def plot_metric_by_decile(df, scenario_name, metric_label, output_path):
    if df.empty:
        return

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
    if df.empty:
        return

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


def plot_co2_compare_decile_x(stats_dict, scenario_name, output_base):
    """X-Axis: Gas Decile. Series: Net CO2, Gas CO2, Elec CO2."""
    metrics_cfg = {
        'net_co2': {'label': 'Net CO2', 'color': 'black', 'style': '-'},
        'gas_co2': {'label': 'Gas CO2', 'color': 'blue', 'style': '--'},
        'elec_co2': {'label': 'Elec CO2', 'color': 'red', 'style': ':'}
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    has_data = False

    for m_key, cfg in metrics_cfg.items():
        if m_key not in stats_dict:
            continue
        df = stats_dict[m_key]
        if df.empty:
            continue
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
            ax.legend(handles=handles, title='CO2 Type')

        ax.grid(True, alpha=0.3)

        out_name = f'{scenario_name}_COMPARE_X_Decile.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, out_name), dpi=300)
        print(f"Saved: {out_name}")
    plt.close()


def plot_co2_compare_premise_x(stats_dict, scenario_name, output_base):
    """X-Axis: Premise Type. Series: Net CO2, Gas CO2, Elec CO2."""
    metrics_cfg = {
        'net_co2': {'label': 'Net CO2', 'color': 'black', 'style': '-'},
        'gas_co2': {'label': 'Gas CO2', 'color': 'blue', 'style': '--'},
        'elec_co2': {'label': 'Elec CO2', 'color': 'red', 'style': ':'}
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    has_data = False

    x_map = {t: i for i, t in enumerate(TYPOLOGIES)}

    for m_key, cfg in metrics_cfg.items():
        if m_key not in stats_dict:
            continue
        df = stats_dict[m_key]
        if df.empty:
            continue
        has_data = True

        df['premise_type'] = pd.Categorical(df['premise_type'], categories=TYPOLOGIES, ordered=True)
        df = df.sort_values('premise_type').dropna(subset=['premise_type'])

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
            ax.legend(handles=handles, title='CO2 Type')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.05)

        out_name = f'{scenario_name}_COMPARE_X_Premise.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, out_name), dpi=300)
        print(f"Saved: {out_name}")
    plt.close()

# ==============================================================================
# 8. LOADING HELPER
# ==============================================================================

def safe_load(filepath, headers=None, usecols=None):
    try:
        if headers:
            # When using custom headers, we need to handle usecols differently
            df = pd.read_csv(filepath, names=headers, header=0)
            if usecols:
                # Filter to only columns that exist
                existing_cols = [c for c in usecols if c in df.columns]
                df = df[existing_cols]
            return df
        else:
            if usecols:
                # Only load columns that exist - read header first
                with open(filepath, 'r') as f:
                    file_headers = next(csv.reader(f))
                existing_cols = [c for c in usecols if c in file_headers]
                return pd.read_csv(filepath, usecols=existing_cols)
            return pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return pd.DataFrame()

# ==============================================================================
# 9. MAIN PIPELINE
# ==============================================================================

def run_pipeline():
    print(f"Scanning: {LOG_DIR}")
    files = glob.glob(LOG_DIR)
    print(f"Found {len(files)} files.")

    # Get headers if on HPC
    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
        except:
            print("Warning: Could not read headers.")

    # Build column requirements
    required_cols = get_required_columns()
    
    # Build metric column map: {(scenario, metric_key): column_name}
    metric_col_map = {}
    for scn in SCENARIOS:
        for m_key, m_cfg in METRIC_PATTERNS.items():
            col_name = m_cfg['pattern'].format(sc=scn)
            metric_col_map[(scn, m_key)] = col_name

    # Initialize accumulators
    accumulators = {}
    for scn in SCENARIOS:
        for m_key, m_cfg in METRIC_PATTERNS.items():
            col_name = m_cfg['pattern'].format(sc=scn)
            accumulators[(scn, m_key)] = GroupedStatsAccumulator(scn, m_key, col_name)

    # Load checkpoint if exists
    processed_files = load_checkpoint(accumulators)
    
    # Filter to unprocessed files
    files_to_process = [f for f in files if f not in processed_files]
    print(f"Files remaining: {len(files_to_process)}")

    # 1. COLLECT RAW DATA
    print('Loading and processing files...')
    for i, file_path in enumerate(tqdm(files_to_process, desc="Processing files")):
        df = safe_load(file_path, headers, usecols=required_cols)
        if df.empty:
            processed_files.add(file_path)
            continue

        # Ensure required group columns exist
        if not set(GROUP_COLS).issubset(df.columns):
            processed_files.add(file_path)
            continue

        # Filter to valid typologies
        df = df[df['premise_type'].isin(TYPOLOGIES)]

        # Single groupby, distribute to all accumulators
        process_file_single_groupby(df, accumulators, metric_col_map)

        processed_files.add(file_path)
        
        del df
        gc.collect()

        # Checkpoint every N files
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(accumulators, processed_files)

    # Final checkpoint before plotting
    save_checkpoint(accumulators, processed_files)

    print("\nAccumulation complete. Generating outputs...")
    final_results = {}
    for (scn, m_key), acc in accumulators.items():
        final_results[(scn, m_key)] = acc.get_raw_data()
    
    with open(os.path.join(OUTPUT_BASE, 'run_buildings_final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    print("Final results saved to final_results.pkl")

    # 2. GENERATE PLOTS
    for scn in SCENARIOS:
        print(f"\n=== Processing Scenario: {scn} ===")

        decile_compare_collection = {}
        premise_compare_collection = {}

        for m_key, m_cfg in METRIC_PATTERNS.items():
            acc = accumulators.get((scn, m_key))
            if not acc:
                continue

            raw_data = acc.get_raw_data()
            if not raw_data:
                continue

            is_wall_scenario = 'wall' in scn

            # --- A. DECILE PROCESSING ---
            idx_std = [0, 2] if is_wall_scenario else [0]
            cols_std = ['avg_gas_percentile', 'inferred_insulation_type'] if is_wall_scenario else ['avg_gas_percentile']
            df_std = compute_grouped_stats(raw_data, idx_std, cols_std)
            if not is_wall_scenario:
                df_std['inferred_insulation_type'] = 'All'

            plot_metric_by_decile(
                df_std, scn, m_cfg['label'],
                os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_decile.png')
            )

            df_compare_dec = compute_grouped_stats(raw_data, [0], ['avg_gas_percentile'])
            decile_compare_collection[m_key] = df_compare_dec

            # --- B. PREMISE PROCESSING ---
            idx_std = [1, 2] if is_wall_scenario else [1]
            cols_std = ['premise_type', 'inferred_insulation_type'] if is_wall_scenario else ['premise_type']
            df_std = compute_grouped_stats(raw_data, idx_std, cols_std)
            if not is_wall_scenario:
                df_std['inferred_insulation_type'] = 'All'

            plot_metric_by_premise(
                df_std, scn, m_cfg['label'],
                os.path.join(OUTPUT_BASE, f'{scn}_{m_key}_premise.png')
            )

            df_compare_prem = compute_grouped_stats(raw_data, [1], ['premise_type'])
            premise_compare_collection[m_key] = df_compare_prem

        # --- C. EXECUTE COMPARISON PLOTS ---
        print(f"  Generating Comparison Plots for {scn}...")
        plot_co2_compare_decile_x(decile_compare_collection, scn, OUTPUT_BASE)
        plot_co2_compare_premise_x(premise_compare_collection, scn, OUTPUT_BASE)

    
    
    print("\nDone. All scenario outputs saved.")


if __name__ == "__main__":
    run_pipeline()