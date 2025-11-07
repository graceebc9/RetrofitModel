import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
MODE_ORDER = ['baseline', 'targeted', 'epc']
MODE_COLORS = {'baseline': '#3498db', 'targeted': '#e74c3c', 'epc': '#2ecc71'}
sns.set_theme(style="whitegrid", context="talk") # 'talk' context for readable, larger labels

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

def prepare_data(projects_df, equity_df):
    """
    Aggregates house-level project data to mode-level and merges with equity metrics.
    """
    print("  -> Aggregating project data...")
    # Aggregate granular projects up to the run/mode level
    agg_projects = projects_df.groupby(['epistemic_run', 'selection_mode']).agg(
        total_cost=('cost_of_intervention_mean', 'sum'),
        mean_cost_per_intervention = ('cost_of_intervention_mean', 'mean'),
        total_co2=('total_ton_co2_saved_mean', 'sum'),
        count=('upn', 'count')
    ).reset_index()

    # Calculate efficiency
    agg_projects['cost_per_ton'] = agg_projects['total_cost'] / agg_projects['total_co2']

    print("  -> Merging with equity data...")
    # Merge with the already-aggregated equity dataframe
    # Ensure equity_df has 'epistemic_run' and 'selection_mode' columns for merging
    full_df = pd.merge(agg_projects, equity_df, on=['epistemic_run', 'selection_mode'], how='inner')
    
    return full_df

# ==============================================================================
# 2. PLOTTING FUNCTIONS
# ==============================================================================

def plot_bar_comparison(df, y_col, y_label, title, filename):
    """Standard bar chart comparing means with error bars (std dev across runs)."""
    plt.figure(figsize=(8, 6))
    
    # Calculate stats for explicit error bars
    stats = df.groupby('selection_mode')[y_col].agg(['mean', 'std']).reindex(MODE_ORDER)
    
    ax = sns.barplot(x=stats.index, y=stats['mean'], palette=MODE_COLORS, order=MODE_ORDER,
                     edgecolor=".2", capsize=.1, errcolor=".2")
    
    # Manually Plot error bars to ensure they use the 'std' we calculated
    plt.errorbar(x=np.arange(len(MODE_ORDER)), y=stats['mean'], yerr=stats['std'], 
                 fmt='none', c='.2', capsize=10)

    plt.title(title, fontweight='bold')
    plt.ylabel(y_label)
    plt.xlabel("") # Mode names are self-explanatory
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_box_distribution(df, y_col, y_label, title, filename):
    """Box plot showing the spread of outcomes across different epistemic runs."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='selection_mode', y=y_col, palette=MODE_COLORS, order=MODE_ORDER)
    sns.stripplot(data=df, x='selection_mode', y=y_col, color='k', alpha=0.3, jitter=True, order=MODE_ORDER)
    
    plt.title(title, fontweight='bold')
    plt.ylabel(y_label)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_scatter_tradeoff(df, x_col, y_col, x_label, y_label, title, filename):
    """Scatter plot to visualize trade-offs between two metrics for every run."""
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='selection_mode', style='selection_mode',
                    palette=MODE_COLORS, s=150, alpha=0.8)
    
    plt.title(title, fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_radar_summary(df, filename):
    """Radar chart of normalized mean performance (0-1 scale, outside is better)."""
    # 1. Define metrics and whether 'more' is better (True) or worse (False)
    metrics = {
        'CO2 Saved': ('total_co2', True),
        'Cost Efficiency': ('cost_per_ton', False), # Lower cost/ton is better
        'Vulnerable %': ('vulnerable_pct', True),
        'Equity (Gini)': ('equity_concentration', False), # Lower Gini is better
        'Retrofit Count': ('count', True)
    }
    
    # 2. Calculate means and normalize
    means = df.groupby('selection_mode')[list(m[0] for m in metrics.values())].mean().reindex(MODE_ORDER)
    normalized = pd.DataFrame(index=means.index)
    
    for label, (col, higher_better) in metrics.items():
        min_val, max_val = means[col].min(), means[col].max()
        denom = max_val - min_val if max_val != min_val else 1
        if higher_better:
            normalized[label] = (means[col] - min_val) / denom
        else:
            normalized[label] = (max_val - means[col]) / denom # Invert so 1 is "best"

    # 3. Plot
    categories = list(normalized.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=7)
    plt.ylim(0, 1.05)

    for mode in MODE_ORDER:
        values = normalized.loc[mode].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=mode.title(), color=MODE_COLORS[mode])
        ax.fill(angles, values, color=MODE_COLORS[mode], alpha=0.2)

    plt.title("Relative Performance (Normalized)\nOuter Edge = Best in Comparison", size=15, y=1.08, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
def plot_persona_counts(equity_df, filename):
    """
    New Vis 2: Grouped bar chart of interventions per persona per mode.
    Compatible with older Seaborn versions by using 'ci' instead of 'errorbar'.
    """
    # 1. Define the columns to melt and their desired display order
    persona_map = {
        'deprived_count': 'Deprived',
        'struggling_count': 'Struggling',
        'lower middle_count': 'Lower Mid',
        'upper middle_count': 'Upper Mid',
        'affluent_count': 'Affluent',
        'student_count': 'Student'
    }
    
    # 2. Melt the dataframe to long format for seaborn grouping
    melted = equity_df.melt(
        id_vars=['epistemic_run', 'selection_mode'],
        value_vars=list(persona_map.keys()),
        var_name='Persona_Raw',
        value_name='Intervention Count'
    )
    
    # 3. Clean up persona names and set categorical order
    melted['Persona'] = melted['Persona_Raw'].map(persona_map)
    persona_order = list(persona_map.values())

    # 4. Plot
    plt.figure(figsize=(12, 7))
    
    # CHANGE HERE: replaced errorbar='sd' with ci='sd' for compatibility
    sns.barplot(data=melted, x='Persona', y='Intervention Count', hue='selection_mode',
                palette=MODE_COLORS, hue_order=MODE_ORDER, order=persona_order,
                ci='sd', capsize=.05, edgecolor=".2")
    
    plt.title('Interventions by Persona Category', fontweight='bold')
    plt.xlabel("")
    plt.ylabel('Mean Intervention Count')
    plt.legend(title='Scenario', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_intervention_types(projects_df, output_dir,  intervention_col='scenario' ):
    """
    Generates stacked and grouped bar charts of intervention types per mode.
    """
    print(f"  -> Generating intervention type analysis using column: '{intervention_col}'...")

    # 1. Aggregate: Mean count of each intervention type per scenario across runs
    # Group by mode, run, and intervention type, then count.
    counts = projects_df.groupby(['selection_mode', 'epistemic_run', intervention_col]).size().reset_index(name='count')
    # Average these counts across the different epistemic runs
    avg_counts = counts.groupby(['selection_mode', intervention_col])['count'].mean().reset_index()

    # --- Plot A: Stacked Bar (Mix per Scenario) ---
    # Pivot for stacked plotting: Index=Mode, Columns=Intervention Type, Values=Mean Count
    pivot_df = avg_counts.pivot(index='selection_mode', columns=intervention_col, values='count').fillna(0)
    pivot_df = pivot_df.reindex(MODE_ORDER) # Ensure consistent x-axis order

    # Using standard pandas/matplotlib for easier stacking
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis', edgecolor='.2')
    plt.title('Mean Intervention Mix per Scenario', fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Mean Count of Interventions')
    plt.legend(title='Intervention Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9_stacked_interventions.png'), dpi=300)
    plt.close()

    # --- Plot B: Grouped Bar (Comparison per Type) ---
    plt.figure(figsize=(12, 8))
    # Order intervention types on X-axis by total overall count for neatness
    type_order = avg_counts.groupby(intervention_col)['count'].sum().sort_values(ascending=False).index

    sns.barplot(data=avg_counts, x=intervention_col, y='count', hue='selection_mode',
                palette=MODE_COLORS, hue_order=MODE_ORDER, order=type_order,
                edgecolor=".2", ci=None) # ci=None as we pre-calculated means

    plt.title('Comparison of Intervention Uptake by Scenario', fontweight='bold')
    plt.xlabel('Intervention Type')
    plt.ylabel('Mean Count')
    plt.xticks(rotation=45, ha='right') # Rotate labels to prevent overlap
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_grouped_interventions.png'), dpi=300)
    plt.close()


def plot_epc_ratings(projects_df, output_dir):
    """
    Generates comparison of CURRENT_ENERGY_RATING across scenarios.
    """
    print("  -> Generating EPC rating analysis...")
    rating_col = 'CURRENT_ENERGY_RATING'
    # Explicit order for EPC ratings
    epc_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    # Aggregate mean counts per rating per scenario
    counts = projects_df.groupby(['selection_mode', 'epistemic_run', rating_col]).size().reset_index(name='count')
    avg_counts = counts.groupby(['selection_mode', rating_col])['count'].mean().reset_index()
    
    # Filter to only include ratings that actually exist in the data to avoid empty plot space if some are missing
    existing_ratings = [r for r in epc_order if r in avg_counts[rating_col].unique()]

    # --- Plot A: Grouped Bar (Absolute Counts) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_counts, x=rating_col, y='count', hue='selection_mode',
                palette=MODE_COLORS, hue_order=MODE_ORDER, order=existing_ratings,
                edgecolor=".2", ci=None)
    plt.title('Selection by Current EPC Rating', fontweight='bold')
    plt.xlabel('Current Energy Rating'); plt.ylabel('Mean Count Selected')
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_grouped_epc.png'), dpi=300)
    plt.close()

    # --- Plot B: 100% Stacked Bar (Proportional Mix) ---
    pivot_df = avg_counts.pivot(index='selection_mode', columns=rating_col, values='count').fillna(0).reindex(MODE_ORDER)
    # Normalize rows to 100%
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Use a diverging colormap (RdYlGn reversed) so A is green, G is red
    pivot_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='RdYlGn_r', edgecolor='.2')
    plt.title('Proportion of Selected EPC Ratings', fontweight='bold')
    plt.xlabel(''); plt.ylabel('Percentage of Selection (%)')
    plt.legend(title='EPC Rating', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_stacked_epc_pct.png'), dpi=300)
    plt.close()
# ==============================================================================
# 3. MAIN ORCHESTRATOR
# ==============================================================================

def run_mode_comparison(projects_df, equity_df, output_dir):
    """
    Main function to generate all comparison plots for a single budget scenario.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting comparison. Output: {output_dir}")
    
    # 1. Prep
    df = prepare_data(projects_df, equity_df)
    df.to_csv(os.path.join(output_dir, "comparison_summary_data.csv"), index=False)

    plot_bar_comparison(df, 'count', 'Number of Retrofits', 
                        'Total Intervention Count', os.path.join(output_dir, '0_bar_count.png'))
    # 2. Efficiency Plots
    print("  -> Generating efficiency plots...")
    plot_bar_comparison(df, 'total_co2', 'Total CO2 Saved (Tonnes)', 
                        'Mean Carbon Savings', os.path.join(output_dir, '1_bar_co2.png'))
    plot_bar_comparison(df, 'cost_per_ton', 'Cost per Tonne (£/tCO2)', 
                        'Cost Effectiveness (Lower is Better)', os.path.join(output_dir, '2a_bar_cost_eff.png'))

    plot_bar_comparison(df, 'total_cost', 'Total Cost (£)', 'Mean Total Cost', os.path.join(output_dir, '2b_bar_total_cost.png'))
    plot_bar_comparison(df, 'mean_cost_per_intervention', 'Mean Total Cost per Intervention (£)', 'Mean Total Cost per Intervention', os.path.join(output_dir, '2c_bar_total_avg_cost.png'))

    
    # 3. Equity Plots
    print("  -> Generating equity plots...")
    plot_box_distribution(df, 'vulnerable_pct', 'Vulnerable Households (%)', 
                          'Vulnerable Selection Variance', os.path.join(output_dir, '3_box_vulnerable.png'))
    plot_box_distribution(df, 'equity_concentration', 'Concentration Index (Gini)', 
                          'Equity Concentration (Lower is Fairer)', os.path.join(output_dir, '4_box_gini.png'))

    # 4. Trade-off & Summary Plots
    print("  -> Generating summary plots...")
    plot_scatter_tradeoff(df, 'vulnerable_pct', 'total_co2', 'Vulnerable %', 'CO2 Saved (Tonnes)',
                          'Trade-off: Equity vs. Carbon', os.path.join(output_dir, '5_scatter_tradeoff.png'))
    plot_radar_summary(df, os.path.join(output_dir, '6_radar_summary.png'))
    
    plot_persona_counts(equity_df, os.path.join(output_dir, '7_persona_summary.png'))
    
    plot_intervention_types(projects_df, output_dir ) 
     
    plot_epc_ratings(projects_df, output_dir)
    # --- NEW VIS: Average Cost per Intervention ---
    
    print("Done.")

