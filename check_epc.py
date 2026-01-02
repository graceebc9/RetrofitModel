import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import kruskal
from src.personas import load_personas

# =======================
# CONFIGURATION
# =======================
LOFT = 0.95
RISK_PENALTY_SIGMA = 1.0
LOG_FOLDER = f'/Volumes/T9/2025_10_RetrofitModel/3_optimiseD_iroiities/epc/risk_sigma_{RISK_PENALTY_SIGMA}__processed_best_only'
OUTPUT_DIR = f'/Volumes/T9/2025_10_RetrofitModel/4_gredy_epc/summary/portfolio_analysis_plots/loft_{LOFT}'

# Column Mappings
COL_UPN = 'upn'
POSTCODE_COL = 'postcode'
COL_GAS = 'avg_gas_percentile'
COL_PERSONA = 'meta_socio_persona'
COL_EPC = 'CURRENT_ENERGY_RATING'
persona_order = ['low_deprived', 'med_deprived', 'high_deprived']

def process_portfolio_data():
    """
    1. Loads all log files.
    2. Dedupes and joins with Persona data.
    3. Checks overlap.
    4. Performs Statistical Tests (with Effect Size).
    5. Generates portfolio-wide visualizations.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- STEP 1: LOAD & AGGREGATE LOG FILES ---
    print(f"Searching for log files in: {LOG_FOLDER}")
    all_files = glob.glob(os.path.join(LOG_FOLDER, f"*_log_file_loft_{LOFT}.csv"))
    
    if not all_files:
        print("No log files found. Check path.")
        return

    print(f"Found {len(all_files)} files. Loading...")
    
    list_of_dfs = []
    for filename in all_files:
        try:
            # Only load necessary columns to save memory
            df = pd.read_csv(filename, usecols=[COL_UPN, POSTCODE_COL, COL_GAS, COL_EPC])
            list_of_dfs.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not list_of_dfs:
        return

    # Concatenate all log files
    raw_df = pd.concat(list_of_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(raw_df)}")

    # Drop duplicates
    unique_df = raw_df.drop_duplicates(subset=[COL_UPN]).copy()
    print(f"Unique UPRNs found: {len(unique_df)}")

    # --- STEP 2: LOAD PERSONA DATA & JOIN ---
    print("Loading Persona Data...")
    persona_df = load_personas() 

    # Join with Persona Data
    merged_df = pd.merge(unique_df, persona_df, on=POSTCODE_COL , how='left')

    # --- STEP 3: CHECK DUPLICATES & OVERLAP ---
    print("-" * 30)
    print("DATA QUALITY CHECK")
    print("-" * 30)
    
    uprn_counts = merged_df[COL_UPN].value_counts()
    multi_uprns = uprn_counts[uprn_counts > 1]
    
    print(f"Total Records: {len(merged_df)}")
    print(f"Unique UPRNs: {merged_df[COL_UPN].nunique()}")
    
    missing_personas = merged_df[COL_PERSONA].isna().sum()
    print(f"Records missing Persona/EPC data: {missing_personas} ({(missing_personas/len(merged_df))*100:.1f}%)")

    # Save the consolidated master dataset
    master_csv_path = os.path.join(OUTPUT_DIR, 'master_portfolio_consolidated.csv')
    merged_df.to_csv(master_csv_path, index=False)
    print(f"Saved consolidated data to: {master_csv_path}")

    # --- STEP 4: STATISTICAL ANALYSIS ---
    perform_statistical_tests(merged_df)

    # --- STEP 5: VISUALIZATIONS ---
    generate_plots(merged_df)


    # 6 interactiosn
    analyze_interactions(merged_df)

def perform_statistical_tests(df):
    """
    Performs Kruskal-Wallis H-test and calculates Eta-squared effect size.
    """
    print("-" * 30)
    print("STATISTICAL ANALYSIS (Non-Parametric)")
    print("-" * 30)
    
    # Filter data: Need Persona and Gas, drop NaNs
    stats_df = df.dropna(subset=[COL_PERSONA, COL_GAS])
    
    groups = []
    
    # Group by Persona
    grouped = stats_df.groupby(COL_PERSONA)
    
    for name, group in grouped:
        gas_values = group[COL_GAS].values
        groups.append(gas_values)
        print(f"  > Persona '{name}': n={len(gas_values)}, Median Gas %={pd.Series(gas_values).median():.2f}")

    if len(groups) < 2:
        print("Not enough groups for comparison.")
        return

    # 1. Perform Kruskal-Wallis Test
    H_stat, p_value = kruskal(*groups)

    # 2. Calculate Effect Size (Eta-squared)
    # Formula: eta^2 = (H - k + 1) / (n - k)
    # where H = H-statistic, k = number of groups, n = total observations
    n = len(stats_df)
    k = len(groups)
    
    if n > k:
        eta_squared = (H_stat - k + 1) / (n - k)
    else:
        eta_squared = 0 # Edge case
        
    # Clamp negative values to 0 (can happen if H < k-1 due to sampling)
    eta_squared = max(0, eta_squared)

    # Interpretation
    if eta_squared < 0.01:
        effect_desc = "Negligible"
    elif eta_squared < 0.06:
        effect_desc = "Small"
    elif eta_squared < 0.14:
        effect_desc = "Medium"
    else:
        effect_desc = "Large"

    print("\n--- TEST RESULTS: Kruskal-Wallis H-test ---")
    print(f"H-statistic: {H_stat:.4f}")
    print(f"P-value:     {p_value:.4e}")
    print(f"N (samples): {n}")
    print(f"k (groups):  {k}")
    
    print("\n--- EFFECT SIZE ---")
    print(f"Eta-squared: {eta_squared:.4f}")
    print(f"Magnitude:   {effect_desc}")

    alpha = 0.05
    if p_value < alpha:
        print("\nCONCLUSION: Significant difference found (p < 0.05).")
        print(f"The gas usage distribution varies between personas with a *{effect_desc}* effect size.")
    else:
        print("\nCONCLUSION: No significant difference found (p >= 0.05).")
    
    # Save stats to text file
    with open(os.path.join(OUTPUT_DIR, 'statistical_results.txt'), 'w') as f:
        f.write("Kruskal-Wallis Test Results\n")
        f.write("===========================\n")
        f.write(f"H-statistic: {H_stat}\n")
        f.write(f"P-value: {p_value}\n")
        f.write(f"N: {n}, k: {k}\n")
        f.write(f"Eta-squared: {eta_squared}\n")
        f.write(f"Effect Size: {effect_desc}\n")
        f.write(f"Significant at 0.05 level: {p_value < 0.05}\n")

def generate_plots(df):
    """Generates Heatmaps, Boxplots, Histograms, and EPC Comparisons."""
    sns.set_theme(style="whitegrid")
    
    # Filter out rows where Persona or EPC is missing
    plot_df = df.dropna(subset=[COL_PERSONA, COL_EPC])

    # Reorder EPCs A-G
    valid_epc = [x for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] if x in plot_df[COL_EPC].unique()]

    # ---------------------------------------------
    # PLOTS (Same as before)
    # ---------------------------------------------
    
    # 1. Heatmap Counts
    plt.figure(figsize=(12, 8))
    heatmap_data_counts = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC])
    heatmap_data_counts = heatmap_data_counts[valid_epc]
    sns.heatmap(heatmap_data_counts, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    # plt.title('Portfolio Composition: Socio-Persona vs EPC Rating (Count)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_heatmap_counts.png'), dpi=300)
    plt.close()

    # 2. Heatmap Percentage
    plt.figure(figsize=(12, 8))
    heatmap_data_pct = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC], normalize='index') * 100
    heatmap_data_pct = heatmap_data_pct[valid_epc]
    sns.heatmap(heatmap_data_pct, annot=True, fmt='.1f', cmap='Blues', linewidths=.5)
    # plt.title('Portfolio Composition: Socio-Persona vs EPC Rating (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_heatmap_percentage.png'), dpi=300)
    plt.close()

    # 3. Gas Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=plot_df, x=COL_GAS, kde=True, bins=30, hue=COL_EPC, hue_order=valid_epc, multiple="stack", palette="viridis", edgecolor=".3", linewidth=.5)
    # plt.title('Distribution of Gas Usage Percentiles by EPC Rating', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_gas_histogram.png'), dpi=300)
    plt.close()

    # 4. Gas Boxplot Persona
    plt.figure(figsize=(12, 6))
    order = plot_df.groupby(COL_PERSONA)[COL_GAS].median().sort_values().index
    sns.boxplot(data=plot_df, x=COL_PERSONA, y=COL_GAS, order=order, palette="Blues")
    # plt.title('Gas Usage Intensity by Socio-Persona', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_gas_boxplot_persona.png'), dpi=300)
    plt.close()

    # 5. Stacked Bar
    props = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC], normalize='index') * 100
    props = props[valid_epc]
    props.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='RdYlGn_r', edgecolor='black', alpha=0.8)
    # plt.title('EPC Rating Distribution within each Persona', fontsize=14, fontweight='bold')
    plt.legend(title='EPC Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_epc_stacked_bar.png'), dpi=300)
    plt.close()

    # 6. Gas vs EPC Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x=COL_EPC, y=COL_GAS, order=valid_epc, palette="RdYlGn_r")
    plt.title('Gas Usage Percentile vs EPC Rating', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_gas_vs_epc_boxplot.png'), dpi=300)
    plt.close()

    # 7. Gas vs EPC Boxplot - Grouped by Socio Persona
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=plot_df, 
        x=COL_EPC, 
        y=COL_GAS, 
        hue=COL_PERSONA,
        order=valid_epc, 
        hue_order=persona_order,
        palette="Set2"  # or use a different palette that works well with 3 categories
    )
    # plt.title('Gas Usage Decile vs EPC Rating by Socio-Economic Persona', 
            # fontsize=14, fontweight='bold')
    plt.xlabel('EPC Rating', fontsize=12)
    plt.ylim(-1, 11)
    plt.ylabel('Gas Usage Percentile', fontsize=12)
    plt.legend(title='Socio Persona', loc='upper center',  ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'duo_comparison_gas_vs_epc_boxplot_by_persona.png'), dpi=300)
    plt.close()

    print("Visualization Complete. Plots saved to output directory.")

    print("Visualization Complete. Plots saved to output directory.")

 

# def analyze_interactions(df):
#     """
#     Focuses on the interaction between Socio-Persona and EPC Rating.
#     """
#     sns.set_theme(style="whitegrid")
    
#     # Clean Data
#     plot_df = df.dropna(subset=[COL_PERSONA, COL_EPC, COL_GAS]).copy()
    
#     # Filter only valid EPCs
#     valid_epc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
#     plot_df = plot_df[plot_df[COL_EPC].isin(valid_epc)]

#     # Create Simplified EPC Groups for cleaner Interaction Plots
#     def group_epc(rating):
#         if rating in ['A', 'B', 'C']: return 'Efficient (A-C)'
#         if rating in ['D', 'E']: return 'Average (D-E)'
#         if rating in ['F', 'G']: return 'Inefficient (F-G)'
#         return 'Unknown'
    
#     plot_df['EPC_Group'] = plot_df[COL_EPC].apply(group_epc)

#     # Order Personas by median gas usage (Low -> High consumption)
#     persona_order = plot_df.groupby(COL_PERSONA)[COL_GAS].median().sort_values().index.tolist()

#     print("-" * 30)
#     print("INTERACTION ANALYSIS")
#     print("-" * 30)

#     # ---------------------------------------------
#     # PLOT 1: Heatmap of MEDIAN Gas Percentile
#     # ---------------------------------------------
#     # This directly answers: "Where are the hotspots?" 
#     # (e.g. Is High Deprivation + EPC G worse than Low Deprivation + EPC C?)
    
#     plt.figure(figsize=(12, 8))
    
#     # Pivot table: Rows=Persona, Cols=EPC, Values=Median Gas
#     pivot_median = pd.pivot_table(
#         plot_df, 
#         values=COL_GAS, 
#         index=COL_PERSONA, 
#         columns=COL_EPC, 
#         aggfunc='median'
#     )
    
#     # Reorder columns to A -> G
#     pivot_median = pivot_median[valid_epc]
    
#     sns.heatmap(pivot_median, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=.5)
    
#     plt.title('Median Gas Percentile by Persona and EPC Rating\n(Red = High Consumption, Green = Low)', fontsize=14, fontweight='bold')
#     plt.xlabel('EPC Rating', fontsize=12)
#     plt.ylabel('Socio Persona', fontsize=12)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_heatmap_median_gas.png'), dpi=300)
#     plt.close()

#     # ---------------------------------------------
#     # PLOT 2: Interaction Point Plot
#     # ---------------------------------------------
#     # This visualizes the diverging trends.
#     # X-axis = Persona (sorted by consumption), Y-axis = Gas %, Lines = EPC Groups
    
#     plt.figure(figsize=(14, 8))
    
#     sns.pointplot(
#         data=plot_df, 
#         x=COL_PERSONA, 
#         y=COL_GAS, 
#         hue='EPC_Group', 
#         hue_order=['Efficient (A-C)', 'Average (D-E)', 'Inefficient (F-G)'],
#         order=persona_order,
#         capsize=.1, 
#         errorbar=('ci', 95), # 95% Confidence Interval
#         palette={'Efficient (A-C)': 'green', 'Average (D-E)': 'orange', 'Inefficient (F-G)': 'red'}
#     )
    
#     plt.title('Interaction Effect: How EPC Rating impacts Consumption across Personas', fontsize=16, fontweight='bold')
#     plt.ylabel('Average Gas Percentile', fontsize=12)
#     plt.xlabel('Socio Persona (Ordered by Low -> High Consumption)', fontsize=12)
#     plt.xticks(rotation=45, ha='right')
#     plt.legend(title='EPC Group')
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_pointplot.png'), dpi=300)
#     plt.close()

#     # ---------------------------------------------
#     # STATS: Pivot Table Printout
#     # ---------------------------------------------
#     print("\n--- Median Gas Percentile Matrix ---")
#     print(pivot_median)
    
#     # Save table
#     pivot_median.to_csv(os.path.join(OUTPUT_DIR, 'stats_median_gas_matrix.csv'))

#     print("\nVisualization Complete. 'interaction_heatmap_median_gas.png' is the key chart.")


def analyze_interactions(df):
    """
    Focuses on the interaction between Socio-Persona and EPC Rating.
    """
    sns.set_theme(style="whitegrid")
    
    # Clean Data
    plot_df = df.dropna(subset=[COL_PERSONA, COL_EPC, COL_GAS]).copy()
    
    # Filter only valid EPCs
    valid_epc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    plot_df = plot_df[plot_df[COL_EPC].isin(valid_epc)]

    # Create Simplified EPC Groups for cleaner Interaction Plots
    def group_epc(rating):
        if rating in ['A', 'B', 'C']: return 'Efficient (A-C)'
        if rating in ['D', 'E']: return 'Average (D-E)'
        if rating in ['F', 'G']: return 'Inefficient (F-G)'
        return 'Unknown'
    
    plot_df['EPC_Group'] = plot_df[COL_EPC].apply(group_epc)

    # ---------------------------------------------------------
    # UPDATE: Set explicit order for Socio Personas
    # ---------------------------------------------------------
    persona_order = ['low_deprived', 'med_deprived', 'high_deprived']
    
    # (Optional) Filter to ensure we only plot these specific personas
    # Remove this line if you want to keep others but just prioritize the order
    plot_df = plot_df[plot_df[COL_PERSONA].isin(persona_order)]

    print("-" * 30)
    print("INTERACTION ANALYSIS")
    print("-" * 30)

    # ---------------------------------------------
    # PLOT 1: Heatmap of MEDIAN Gas Percentile
    # ---------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Pivot table: Rows=Persona, Cols=EPC, Values=Median Gas
    pivot_median = pd.pivot_table(
        plot_df, 
        values=COL_GAS, 
        index=COL_PERSONA, 
        columns=COL_EPC, 
        aggfunc='median'
    )
    
    # Reorder columns to A -> G
    pivot_median = pivot_median[valid_epc]
    
    # UPDATE: Reorder rows to the specific persona order
    pivot_median = pivot_median.reindex(persona_order)
    
    sns.heatmap(pivot_median, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=.5)
    
    # plt.title('Median Gas Percentile by Persona and EPC Rating\n(Red = High Consumption, Green = Low)', fontsize=14, fontweight='bold')
    plt.xlabel('EPC Rating', fontsize=12)
    plt.ylabel('Socio Persona', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_heatmap_median_gas.png'), dpi=300)
    plt.close()

    # ---------------------------------------------
    # PLOT 2: Interaction Point Plot
    # ---------------------------------------------
    plt.figure(figsize=(14, 8))
    
    sns.pointplot(
        data=plot_df, 
        x=COL_PERSONA, 
        y=COL_GAS, 
        hue='EPC_Group', 
        hue_order=['Efficient (A-C)', 'Average (D-E)', 'Inefficient (F-G)'],
        order=persona_order, # UPDATE: Use the fixed order here
        capsize=.1, 
        errorbar=('ci', 95), 
        palette={'Efficient (A-C)': 'green', 'Average (D-E)': 'orange', 'Inefficient (F-G)': 'red'}
    )
    
    # plt.title('Interaction Effect: How EPC Rating impacts Consumption across Personas', fontsize=16, fontweight='bold')
    plt.ylabel('Average Gas Percentile', fontsize=12)
    plt.xlabel('Meta Socio Persona', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='EPC Group')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_pointplot.png'), dpi=300)
    plt.close()

    # ---------------------------------------------
    # STATS: Pivot Table Printout
    # ---------------------------------------------
    print("\n--- Median Gas Percentile Matrix ---")
    print(pivot_median)
    
    # Save table
    pivot_median.to_csv(os.path.join(OUTPUT_DIR, 'stats_median_gas_matrix.csv'))

    print("\nVisualization Complete. 'interaction_heatmap_median_gas.png' is the key chart.")


if __name__ == "__main__":
    process_portfolio_data()