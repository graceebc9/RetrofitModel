import os
import glob
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from src.utils import is_running_on_hpc 
from src.RetrofitUtils import safe_load 
import matplotlib.ticker as mtick
# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Update these paths for your environment
SCENARIO_NAME = 'stock_summary'      # Just for folder naming
INPUT_PATTERN = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/*csv'
OUTPUT_BASE_DIR = '2_stock_results/stock_summary_new/'
ERROR_LOG_FILE = '2_stock_results/stock_summary_new/processing_errors.txt'

# HPC / Local path toggles
is_hpc = is_running_on_hpc() 
if is_hpc:
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/130_log_file.csv'
else:
    # Example local path
    INPUT_PATTERN = '/Users/gracecolverd/RetrofitModel/1_intermediate_data_2D/retrofit_scenario/all/NE/*csv'
    REFERENCE_FILE = None

# Predefined order for consistency
TYPOLOGIES = [
    'Small low terraces', '3-4 storey and smaller flats',
    'Tall terraces 3-4 storeys', 'Large semi detached', 'Standard size detached',
    'Standard size semi detached', '2 storeys terraces with t rear extension',
    'Semi type house in multiples', 'Large detached',
      'Planned balanced mixed estates',
    'Linked and step linked premises',
]

# ==============================================================================
# 2. HELPER: SAFE LOAD
# ==============================================================================
# ==============================================================================
# 3. ROBUST STOCK ACCUMULATOR
# ==============================================================================
class StockCountAccumulator:
    def __init__(self):
        self.group_keys = [
            'premise_age_bucketed', 
            'premise_type', 
            'avg_gas_percentile',
            'inferred_insulation_type',
            'conservation_area_bool'
        ]
        self.data_store = []
        # --- DATA QUALITY TRACKER ---
        self.dq_report = {
            'processed_successfully': 0,
            'empty_files': 0,
            'missing_grouping_cols': 0,
            'missing_upn': 0,
            'load_errors': 0
        }

    def process_file(self, file_path, headers=None):
        try:
            df = safe_load(file_path, headers, ERROR_LOG_FILE)
            
            if df is None or df.empty:
                self.dq_report['empty_files'] += 1
                return

            # 1. Column Verification
            current_keys = [k for k in self.group_keys if k in df.columns]
            if not current_keys:
                self.dq_report['missing_grouping_cols'] += 1
                return

            if 'upn' not in df.columns:
                self.dq_report['missing_upn'] += 1
                # We continue, but log the warning
            else:
                df.drop_duplicates(subset=['upn'], keep='first', inplace=True)

            # 2. Aggregation with named output
            grouped = df.groupby(current_keys).size().reset_index(name='count')
            
            if not grouped.empty:
                self.data_store.append(grouped)
                self.dq_report['processed_successfully'] += 1
            
            del df
            gc.collect()
            
        except Exception as e:
            self.dq_report['load_errors'] += 1
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    def finalize(self):
        print("\n" + "="*30)
        print("DATA QUALITY SUMMARY")
        print("="*30)
        for reason, count in self.dq_report.items():
            print(f"{reason.replace('_', ' ').title()}: {count}")
        print("="*30 + "\n")

        if not self.data_store:
            return pd.DataFrame()
            
        full_df = pd.concat(self.data_store, ignore_index=True)
        keys_present = [k for k in self.group_keys if k in full_df.columns]
        
        # Final safety check before sum
        if 'count' not in full_df.columns:
            return pd.DataFrame()

        return full_df.groupby(keys_present)['count'].sum().reset_index()


# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================

def plot_counts_by_age_band(df, output_dir):
    """
    Generates one stacked bar plot per Age Band:
    X-axis: Gas Decile
    Y-axis: Count
    Stack/Color: Insulation Type
    """
    required_cols = ['premise_age_bucketed', 'avg_gas_percentile', 'inferred_insulation_type']
    if not set(required_cols).issubset(df.columns):
        print("Skipping Age Band plots (required columns missing)")
        return

    # Ensure output dir
    age_dir = os.path.join(output_dir, 'by_age_band')
    os.makedirs(age_dir, exist_ok=True)
    
    # Ensure numeric decile for sorting
    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    
    # --- Y-AXIS UNIFICATION LOGIC ---
    # Calculate the max height of any bar across ALL age bands.
    grouped_totals = df.groupby(['premise_age_bucketed', 'decile_numeric'])['count'].sum()
    global_max_y = grouped_totals.max() * 1.05 if not grouped_totals.empty else 0
    print(f"Global Y-Axis Max for Age Plots set to: {global_max_y:.0f}")

    # --- COLOR CONSISTENCY LOGIC ---
    # 1. Get all unique insulation types across the entire dataframe
    all_insul_types = sorted(df['inferred_insulation_type'].dropna().unique())
    
    # 2. Reorder so 'cavity' types come first
    cavity_types = [t for t in all_insul_types if 'cavity' in str(t).lower()]
    other_types = [t for t in all_insul_types if t not in cavity_types]
    master_order = cavity_types + other_types
    
    # 3. Create a fixed color map to ensure consistency across plots
    # We generate a list of colors from the viridis colormap
    colors_array = plt.cm.viridis(np.linspace(0, 1, len(master_order)))
    type_color_map = {t: colors_array[i] for i, t in enumerate(master_order)}
    
    age_bands = df['premise_age_bucketed'].unique()
    
    print(f"\nGenerating plots for {len(age_bands)} Age Bands...")
    
    for age in age_bands:
        if pd.isna(age): continue
        
        # Filter for this age band
        subset = df[df['premise_age_bucketed'] == age]
        
        # Pivot data for stacked bar chart
        # Index (X-axis) = Decile, Columns (Stacks) = Insulation Type, Values = Count
        plot_data = subset.groupby(['decile_numeric', 'inferred_insulation_type'])['count'].sum().unstack(fill_value=0)
        
        if plot_data.empty:
            continue
            
        # Reorder columns to match master_order (filtering for those present)
        # This ensures 'cavity' is always bottom and colors are consistent
        present_cols = [c for c in master_order if c in plot_data.columns]
        plot_data = plot_data[present_cols]
        
        # Get corresponding colors for the present columns
        current_colors = [type_color_map[c] for c in present_cols]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_data.plot(kind='bar', stacked=True, ax=ax, color=current_colors, alpha=0.9, width=0.8, edgecolor='black')
        
        # Apply consistent Y-limit
        ax.set_ylim(0, global_max_y)
        # This adds commas (e.g. 10,000) to the Y axis
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # <--- ADDED FORMATTER HERE
        
        
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # <--- ADDED FORMATTER HERE
        
        plt.xlabel('Gas Consumption Decile)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Insulation Type',  loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"count_age_{str(age).replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(os.path.join(age_dir, filename), dpi=300)
        plt.close()

def plot_counts_by_premise_decile(df, output_dir):
    """
    Generates one stacked plot per Premise Type:
    X-axis: Decile (avg_gas_percentile)
    Y-axis: Count
    Stack/Color: Conservation Area (True/False)
    """
    if 'premise_type' not in df.columns or 'avg_gas_percentile' not in df.columns:
        print("Skipping Decile plots (columns missing)")
        return
        
    stack_col = 'conservation_area_bool'
    if stack_col not in df.columns:
        print(f"Warning: '{stack_col}' missing. Plots will not be stacked.")
        # Create dummy column to prevent crash if missing
        df[stack_col] = 'Unknown'

    # Ensure output dir
    decile_dir = os.path.join(output_dir, 'by_premise_type_decile')
    os.makedirs(decile_dir, exist_ok=True)
    df[stack_col] = df[stack_col].astype(str)
    # Ensure numeric decile
    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    
    # --- Y-AXIS UNIFICATION LOGIC ---
    # Calculate max bar height across ALL premise types.
    # We sum up counts regardless of stacking to get total bar height
    grouped_totals = df.groupby(['premise_type', 'decile_numeric'])['count'].sum()
    global_max_y = grouped_totals.max() * 1.05 if not grouped_totals.empty else 0
    print(f"Global Y-Axis Max for Premise Plots set to: {global_max_y:.0f}")

    types = df['premise_type'].unique()
    print(f"\nGenerating plots for {len(types)} Premise Types...")
    
    for p_type in types:
        if pd.isna(p_type): continue
        
        # Filter for this premise type
        subset = df[df['premise_type'] == p_type]
        
        # Pivot data for stacked bar chart
        # Index (X-axis) = Decile, Columns (Stacks) = Conservation, Values = Count
        plot_data = subset.groupby(['decile_numeric', stack_col])['count'].sum().unstack(fill_value=0)
        
        if plot_data.empty:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        # Use a distinct colormap for conservation status (e.g., 'Paired' or specific colors)
        plot_data.plot(kind='bar', stacked=True, ax=ax, colormap='Paired', alpha=0.9, width=0.8, edgecolor='black')
        
        # Apply consistent Y-limit
        ax.set_ylim(0, global_max_y)
        # This adds commas (e.g. 10,000) to the Y axis
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # <--- ADDED FORMATTER HERE
        
        plt.xlabel('Gas Consumption Decile')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Conservation Area', loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Sanitize filename
        safe_name = "".join([c if c.isalnum() else "_" for c in str(p_type)])
        filename = f"count_decile_{safe_name}.png"
        plt.savefig(os.path.join(decile_dir, filename), dpi=300)
        plt.close()


# ==============================================================================
# 5. MAIN PIPELINE (ROBUST VERSION)
# ==============================================================================

def run_loading_pipeline():
    files = glob.glob(INPUT_PATTERN)
    files=files[0:3]
    if not files:
        print(f"No files found matching pattern: {INPUT_PATTERN}")
        return

    print(f"Found {len(files)} files. Starting processing...")
    
    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
        except Exception as e:
            print(f"Warning: Reference file error: {e}")

    accumulator = StockCountAccumulator()
    
    for i, f in enumerate(files):
        if i % 10 == 0: 
            print(f"Processing {i}/{len(files)}: {os.path.basename(f)}")
        accumulator.process_file(f, headers)
        
    # Finalize returns the safe, aggregated dataframe
    df_counts = accumulator.finalize()
    
    # --- ROBUST CHECK FOR KEYERROR ---
    if df_counts is not None and not df_counts.empty and 'count' in df_counts.columns:
        print(f"Aggregation Complete. Final Shape: {df_counts.shape}")
        
        total_buildings = df_counts['count'].sum()
        print(f"Total Unique Buildings Counted: {total_buildings:,}")
        
        # Save and Plot
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        df_counts.to_csv(os.path.join(OUTPUT_BASE_DIR, 'raw_stock_counts.csv'), index=False)
        
        print("Generating plots...")
        plot_counts_by_age_band(df_counts, OUTPUT_BASE_DIR)
        plot_counts_by_premise_decile(df_counts, OUTPUT_BASE_DIR)
    else:
        print("CRITICAL: No data survived the quality checks. Skipping plotting.")

if __name__ == "__main__":
    run_loading_pipeline()