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
INPUT_PATTERN = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/*csv'
OUTPUT_BASE_DIR = '1_processed_results/stock_summary/'
ERROR_LOG_FILE = '1_summary_results/stock_summary/processing_errors.txt'

# HPC / Local path toggles
is_hpc = is_running_on_hpc() 
if is_hpc:
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/v8/NE/130_log_file.csv'
else:
    # Example local path
    INPUT_PATTERN = '/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all/NE/*csv'
    REFERENCE_FILE = None

# Predefined order for consistency
TYPOLOGIES = [
    'Medium height flats 5-6 storeys', 'Small low terraces', '3-4 storey and smaller flats',
    'Tall terraces 3-4 storeys', 'Large semi detached', 'Standard size detached',
    'Standard size semi detached', '2 storeys terraces with t rear extension',
    'Semi type house in multiples', 'Tall flats 6-15 storeys', 'Large detached',
    'Very tall point block flats', 'Very large detached', 'Planned balanced mixed estates',
    'Linked and step linked premises'
]

# ==============================================================================
# 2. HELPER: SAFE LOAD
# ==============================================================================


# ==============================================================================
# 3. STOCK ACCUMULATOR (Counts Only)
# ==============================================================================
class StockCountAccumulator:
    """
    Reads chunks of building data, DEDUPLICATES based on 'upn' (per file), 
    and counts occurrences by Age Band, Premise Type, Decile, etc.
    """
    def __init__(self):
        self.group_keys = [
            'premise_age_bucketed', 
            'premise_type', 
            'avg_gas_percentile',
            'inferred_insulation_type',
            'conservation_area_bool'
        ]
        self.data_store = []

    def process_file(self, file_path, headers=None):
        try:
            df = safe_load(file_path, headers, ERROR_LOG_FILE)
            
            if df.empty: return

            # --- 1. LOCAL DEDUPLICATION ---
            # We only care about duplicates inside THIS specific file
            initial_count = len(df)
            if 'upn' in df.columns:
                df.drop_duplicates(subset=['upn'], keep='first', inplace=True)
            else:
                print(f"Warning: 'upn' column missing in {os.path.basename(file_path)}")

            unique_count = len(df)
            
            # --- 2. LOGGING ---
            # Log the count for this specific file as requested
            dropped = initial_count - unique_count
            print(f"  -> {os.path.basename(file_path)}: {unique_count} unique UPNs (dropped {dropped} duplicates)")

            # --- 3. FILTERING ---
            # Filter valid typologies if column exists
            if 'premise_type' in df.columns:
                df = df[df['premise_type'].isin(TYPOLOGIES)]
            
            # Ensure required columns exist for grouping
            current_keys = [k for k in self.group_keys if k in df.columns]
            if not current_keys:
                return

            # --- 4. AGGREGATION ---
            # Count occurrences (size) and store
            grouped = df.groupby(current_keys).size().reset_index(name='count')
            self.data_store.append(grouped)
            
            # Cleanup to save memory
            del df
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def finalize(self):
        print("Finalizing aggregation...")
        if not self.data_store:
            return pd.DataFrame()
            
        # Concatenate all chunk counts
        full_df = pd.concat(self.data_store, ignore_index=True)
        
        # Sum the counts across chunks
        keys_present = [k for k in self.group_keys if k in full_df.columns]
        
        final_df = full_df.groupby(keys_present)['count'].sum().reset_index()
        
        return final_df

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
        plot_data.plot(kind='bar', stacked=True, ax=ax, color=current_colors, alpha=0.9, width=0.8)
        
        # Apply consistent Y-limit
        ax.set_ylim(0, global_max_y)
        
        
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # <--- ADDED FORMATTER HERE
        
        plt.xlabel('Gas Consumption Decile)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Insulation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
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
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use a distinct colormap for conservation status (e.g., 'Paired' or specific colors)
        plot_data.plot(kind='bar', stacked=True, ax=ax, colormap='Paired', alpha=0.9, width=0.8)
        
        # Apply consistent Y-limit
        ax.set_ylim(0, global_max_y)
        # This adds commas (e.g. 10,000) to the Y axis
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # <--- ADDED FORMATTER HERE
        
        plt.xlabel('Gas Consumption Decile')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Conservation Area', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Sanitize filename
        safe_name = "".join([c if c.isalnum() else "_" for c in str(p_type)])
        filename = f"count_decile_{safe_name}.png"
        plt.savefig(os.path.join(decile_dir, filename), dpi=300)
        plt.close()

# ==============================================================================
# 5. MAIN PIPELINE
# ==============================================================================

def run_loading_pipeline():
    files = glob.glob(INPUT_PATTERN)
    print(f"Found {len(files)} files.")
    #files=files[0:5]
    
    # 1. Get Headers if needed (HPC fix)
    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
            print("Loaded headers from reference file.")
        except Exception as e:
            print(f"Warning: Could not read headers: {e}")

    # 2. Initialize Accumulator
    accumulator = StockCountAccumulator()
    
    # 3. Process Files
    for i, f in enumerate(files):
        if i % 10 == 0: print(f"Processing {i}/{len(files)}: {os.path.basename(f)}")
        accumulator.process_file(f, headers)
        
    # 4. Get Aggregated DataFrame
    df_counts = accumulator.finalize()
    
    print(f"Aggregation Complete. Shape: {df_counts.shape}")
    print(f"Total Buildings Counted: {df_counts['count'].sum():,}")
    
    # 5. Generate Plots
    if not df_counts.empty:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        # Save raw counts csv
        df_counts.to_csv(os.path.join(OUTPUT_BASE_DIR, 'raw_stock_counts.csv'), index=False)
        
        # Run specific plotting logic
        plot_counts_by_age_band(df_counts, OUTPUT_BASE_DIR)
        plot_counts_by_premise_decile(df_counts, OUTPUT_BASE_DIR)
        
        print(f"\nAll outputs saved to: {OUTPUT_BASE_DIR}")
    else:
        print("DataFrame is empty. Check input paths.")

if __name__ == "__main__":
    run_loading_pipeline()