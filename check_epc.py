import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# ---------------------------------------------------------
# CONSISTENT COLOURING CONFIGURATION
# ---------------------------------------------------------
persona_order = ['low_deprived', 'med_deprived', 'high_deprived']

# Define specific colors for each persona to ensure consistency across all plots
PERSONA_COLORS = {
    'low_deprived': '#009E73',  # Blue
    'med_deprived': '#E69F00',  # Orange
    'high_deprived': '#D55E00'  # Red
}


# colors = {
#     'green': '#009E73',   # Bluish green
#     'orange': '#E69F00',  # Orange
#     'red': '#D55E00'      # Vermillion/red-orange
# }
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
    generate_distribution_plots(merged_df)
    perform_statistical_tests(merged_df)

    # --- STEP 5: VISUALIZATIONS ---
    generate_plots(merged_df)


    # 6 interactions
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
    n = len(stats_df)
    k = len(groups)
    
    if n > k:
        eta_squared = (H_stat - k + 1) / (n - k)
    else:
        eta_squared = 0 
        
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
    # PLOTS
    # ---------------------------------------------
    
    # 1. Heatmap Counts (Persona vs EPC)
    plt.figure(figsize=(12, 8))
    heatmap_data_counts = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC])
    heatmap_data_counts = heatmap_data_counts[valid_epc]
    heatmap_data_counts = heatmap_data_counts.reindex(persona_order)
    
    # Updated: fmt=',d' for comma separators
    sns.heatmap(heatmap_data_counts, annot=True, fmt=',d', cmap='YlGnBu', linewidths=.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_heatmap_counts.png'), dpi=300)
    plt.close()

    # 2. Heatmap Percentage
    plt.figure(figsize=(12, 8))
    heatmap_data_pct = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC], normalize='index') * 100
    heatmap_data_pct = heatmap_data_pct[valid_epc]
    heatmap_data_pct = heatmap_data_pct.reindex(persona_order)
    sns.heatmap(heatmap_data_pct, annot=True, fmt='.1f', cmap='Blues', linewidths=.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_heatmap_percentage.png'), dpi=300)
    plt.close()

    # 3. Gas Histogram (Hue is EPC)
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=plot_df, x=COL_GAS, kde=True, bins=30, hue=COL_EPC, hue_order=valid_epc, multiple="stack", palette="viridis", edgecolor=".3", linewidth=.5)
    
    # Updated: Format Y-axis with commas
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_gas_histogram.png'), dpi=300)
    plt.close()

    # 4. Gas Boxplot Persona -- UPDATED COLOURS
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=plot_df, 
        x=COL_PERSONA, 
        y=COL_GAS, 
        order=persona_order, 
        palette=PERSONA_COLORS 
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_gas_boxplot_persona.png'), dpi=300)
    plt.close()

    # 5. Stacked Bar (EPC distribution within Persona)
    props = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC], normalize='index') * 100
    props = props[valid_epc]
    props = props.reindex(persona_order)
    props.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='RdYlGn_r', edgecolor='black', alpha=0.8)
    plt.legend(title='EPC Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_epc_stacked_bar.png'), dpi=300)
    plt.close()

    # 6. Gas vs EPC Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x=COL_EPC, y=COL_GAS, order=valid_epc, palette="RdYlGn_r")
    plt.title('Gas Consumption Decile vs EPC Rating', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_gas_vs_epc_boxplot.png'), dpi=300)
    plt.close()

    # 7. Gas vs EPC Boxplot - Grouped by Socio Persona -- UPDATED COLOURS
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=plot_df, 
        x=COL_EPC, 
        y=COL_GAS, 
        hue=COL_PERSONA,
        order=valid_epc, 
        hue_order=persona_order,
        palette=PERSONA_COLORS 
    )
    plt.xlabel('EPC Rating', fontsize=12)
    plt.ylim(-1, 11)
    plt.ylabel('Gas Consumption Decile', fontsize=12)
    plt.legend(title='Socio Persona', loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'duo_comparison_gas_vs_epc_boxplot_by_persona.png'), dpi=300)
    plt.close()

    print("Visualization Complete. Plots saved to output directory.")

 
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

    # Create Simplified EPC Groups
    def group_epc(rating):
        if rating in ['A', 'B', 'C']: return 'Efficient (A-C)'
        if rating in ['D', 'E']: return 'Average (D-E)'
        if rating in ['F', 'G']: return 'Inefficient (F-G)'
        return 'Unknown'
    
    plot_df['EPC_Group'] = plot_df[COL_EPC].apply(group_epc)

    # Ensure correct order
    plot_df = plot_df[plot_df[COL_PERSONA].isin(persona_order)]

    print("-" * 30)
    print("INTERACTION ANALYSIS")
    print("-" * 30)

    # ---------------------------------------------
    # PLOT 1: Heatmap of MEDIAN Gas Percentile
    # ---------------------------------------------
    plt.figure(figsize=(12, 8))
    
    pivot_median = pd.pivot_table(
        plot_df, 
        values=COL_GAS, 
        index=COL_PERSONA, 
        columns=COL_EPC, 
        aggfunc='median'
    )
    
    pivot_median = pivot_median[valid_epc]
    pivot_median = pivot_median.reindex(persona_order)
    
    sns.heatmap(pivot_median, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=.5, cbar_kws={'label': 'Median Gas Decile'})
    
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
        order=persona_order,
        capsize=.1, 
        errorbar=('ci', 95), 
        palette={'Efficient (A-C)': 'green', 'Average (D-E)': 'orange', 'Inefficient (F-G)': 'red'}
    )
    
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
    
    pivot_median.to_csv(os.path.join(OUTPUT_DIR, 'stats_median_gas_matrix.csv'))
    print("\nVisualization Complete. 'interaction_heatmap_median_gas.png' is the key chart.")


def generate_distribution_plots(df):
    """
    Generates basic distribution plots to understand the dataset composition.
    """
    print("-" * 30)
    print("GENERATING DISTRIBUTION PLOTS")
    print("-" * 30)
    
    sns.set_theme(style="whitegrid")
    
    # Clean data for plotting
    plot_df = df.dropna(subset=[COL_PERSONA, COL_EPC, COL_GAS]).copy()
    
    # Valid EPC order
    valid_epc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    plot_df = plot_df[plot_df[COL_EPC].isin(valid_epc)]
    
    # ---------------------------------------------
    # PLOT 1: Distribution of Socio-Personas (Count) -- UPDATED COLOURS & FORMATTING
    # ---------------------------------------------
    plt.figure(figsize=(10, 6))
    persona_counts = plot_df[COL_PERSONA].value_counts().reindex(persona_order)
    
    ax = sns.barplot(
        x=persona_counts.index, 
        y=persona_counts.values, 
        palette=PERSONA_COLORS ,
         edgecolor='black', 
    )
    plt.xlabel('Socio Persona', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Updated: Format Y-axis with commas
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Labels already have commas in previous version, keeping them:
    for i, v in enumerate(persona_counts.values):
        ax.text(i, v + max(persona_counts.values)*0.01, f'{v:,}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_persona_counts.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------
    # PLOT 2: Distribution of EPC Ratings (Count) -- UPDATED FORMATTING
    # ---------------------------------------------
    plt.figure(figsize=(10, 6))
    epc_counts = plot_df[COL_EPC].value_counts().reindex(valid_epc)
    ax = sns.barplot(x=epc_counts.index, y=epc_counts.values, palette='RdYlGn_r',  edgecolor='black', )
    plt.xlabel('EPC Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Updated: Format Y-axis with commas
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    for i, v in enumerate(epc_counts.values):
        ax.text(i, v + max(epc_counts.values)*0.01, f'{v:,}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_epc_counts.png'), dpi=300)
    plt.close()

   
    # ---------------------------------------------
    # PLOT 2b: Distribution of EPC Ratings (Grouped by Persona) -- UPDATED COLOURS & FORMATTING
    # ---------------------------------------------
    plt.figure(figsize=(12, 6))

    epc_persona_counts = pd.crosstab(plot_df[COL_EPC], plot_df[COL_PERSONA])
    epc_persona_counts = epc_persona_counts.reindex(valid_epc)
    epc_persona_counts = epc_persona_counts[persona_order] 

    colors = [PERSONA_COLORS[p] for p in epc_persona_counts.columns]

    ax = epc_persona_counts.plot(
        kind='bar', 
        figsize=(12, 6), 
        color=colors, 
        edgecolor='black', 
        # alpha=0.8, 
        width=0.8
    )

    plt.xlabel('EPC Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Socio Persona', loc='upper right')
    plt.xticks(rotation=0)
    
    # Updated: Format Y-axis with commas
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Updated: Format bar labels with commas
    for container in ax.containers:
        # Create labels with comma formatting
        labels = [f'{rect.get_height():,.0f}' for rect in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_epc_counts_grouped.png'), dpi=300)
    plt.close()

    epc_persona_counts.to_csv(os.path.join(OUTPUT_DIR, 'dist_epc_persona_counts_table.csv'))
    print(f"Saved EPC × Persona counts table to: dist_epc_persona_counts_table.csv")

    # ---------------------------------------------
    # PLOT 3: Overall Gas Percentile Distribution -- UPDATED FORMATTING
    # ---------------------------------------------
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data=plot_df, x=COL_GAS, kde=True, bins=50, 
                 color='steelblue', edgecolor='black', alpha=0.7)
    
    plt.axvline(plot_df[COL_GAS].median(), color='red', linestyle='--', 
                linewidth=2, label=f'Median: {plot_df[COL_GAS].median():.2f}')
    plt.axvline(plot_df[COL_GAS].mean(), color='orange', linestyle='--', 
                linewidth=2, label=f'Mean: {plot_df[COL_GAS].mean():.2f}')
    plt.xlabel('Gas Usage Decile', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
    # Updated: Format Y-axis with commas
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_gas_percentile_overall.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------
    # PLOT 4: Gas Percentile Distribution by Persona -- UPDATED COLOURS
    # ---------------------------------------------
    plt.figure(figsize=(14, 6))
    for persona in persona_order:
        subset = plot_df[plot_df[COL_PERSONA] == persona]
        if not subset.empty:
            sns.kdeplot(
                data=subset, 
                x=COL_GAS, 
                label=persona, 
                linewidth=2, 
                 
                color=PERSONA_COLORS[persona] 
            )
    
    plt.xlabel('Gas Consumption Decile', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Socio Persona')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_gas_by_persona_kde.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------
    # PLOT 5: Gas Percentile Distribution by EPC
    # ---------------------------------------------
    plt.figure(figsize=(14, 6))
    for epc in valid_epc:
        subset = plot_df[plot_df[COL_EPC] == epc]
        if len(subset) > 0:
            sns.kdeplot(data=subset, x=COL_GAS, label=f'EPC {epc}', linewidth=2)
    
    plt.xlabel('Gas Consumption Decile', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='EPC Rating')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_gas_by_epc_kde.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------
    # PLOT 6: Sample Size Matrix (Persona × EPC) -- UPDATED FORMATTING
    # ---------------------------------------------
    plt.figure(figsize=(12, 8))
    sample_sizes = pd.crosstab(plot_df[COL_PERSONA], plot_df[COL_EPC])
    sample_sizes = sample_sizes[valid_epc]
    sample_sizes = sample_sizes.reindex(persona_order)
    
    # Updated: fmt=',d' for comma separators
    sns.heatmap(sample_sizes, annot=True, fmt=',d', cmap='Purples', 
                linewidths=.5, cbar_kws={'label': 'Sample Size'})
    plt.xlabel('EPC Rating', fontsize=12)
    plt.ylabel('Socio Persona', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_sample_sizes_matrix.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------
    # Print Summary Statistics
    # ---------------------------------------------
    print("\n--- DATASET SUMMARY STATISTICS ---")
    print(f"Total Records: {len(plot_df):,}")
    print(f"\nGas Percentile Statistics:")
    print(f"  Mean:   {plot_df[COL_GAS].mean():.2f}")
    print(f"  Median: {plot_df[COL_GAS].median():.2f}")
    print(f"  Std:    {plot_df[COL_GAS].std():.2f}")
    print(f"  Min:    {plot_df[COL_GAS].min():.2f}")
    print(f"  Max:    {plot_df[COL_GAS].max():.2f}")
    
    print(f"\nPersona Distribution:")
    for persona in persona_order:
        count = (plot_df[COL_PERSONA] == persona).sum()
        pct = (count / len(plot_df)) * 100
        print(f"  {persona}: {count:,} ({pct:.1f}%)")
    
    print(f"\nEPC Rating Distribution:")
    for epc in valid_epc:
        count = (plot_df[COL_EPC] == epc).sum()
        pct = (count / len(plot_df)) * 100
        print(f"  {epc}: {count:,} ({pct:.1f}%)")
    
    print("\nDistribution plots saved successfully.")


if __name__ == "__main__":
    process_portfolio_data()