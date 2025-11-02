import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Define parameters
budgets = [10000000]  # Add or modify your budgets here
equity_weights = [0, 0.2, 0.6, 0.8, 1] 
loft_value = 0.65  # Update if needed
base_path = '/Users/gracecolverd/RetrofitModel/test/greedy'

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
equity_dfs = []
results_dfs = []

for budget in budgets:
    for equity_weight in equity_weights:
        # Construct the directory path
        dir_path = f'{base_path}/budget_{budget}_loft_{loft_value}_equity: {equity_weight}'
        
        # Create scenario label
        scenario_label = f'budget_{budget}_equity_{equity_weight}'
        
        # try:
        # Load equity tracking data
        equity_df_temp = pd.read_csv(f'{dir_path}/equity_tracking.csv')
        # equity_df_temp['intervention'] = equity_df_temp['scenario']
        equity_df_temp['scenario'] = scenario_label
        equity_df_temp['budget'] = budget
        equity_df_temp['equity_weight'] = equity_weight
        equity_dfs.append(equity_df_temp)
        
        # Load combined results data
        results_df_temp = pd.read_csv(f'{dir_path}/combined_results.csv')
        results_df_temp['intervention'] = results_df_temp['scenario']
        results_df_temp['scenario'] = scenario_label
        results_df_temp['budget'] = budget
        results_df_temp['equity_weight'] = equity_weight
        results_dfs.append(results_df_temp)
        
        print(f"✓ Loaded data for budget=${budget/1e6:.1f}M, equity_weight={equity_weight}")
            
        # except FileNotFoundError as e:
        #     print(f"✗ Missing data for budget=${budget/1e6:.1f}M, equity_weight={equity_weight}")
        #     continue
        # except Exception as e:
        #     print(f"✗ Error loading budget=${budget/1e6:.1f}M, equity_weight={equity_weight}: {str(e)}")
        #     continue

# Combine all dataframes
if equity_dfs:
    equity_df = pd.concat(equity_dfs, ignore_index=True)
    print(f"\n✓ Combined {len(equity_dfs)} equity tracking files")
    print(f"  Total equity tracking records: {len(equity_df):,}")
else:
    print("\n✗ No equity tracking data loaded!")
    equity_df = pd.DataFrame()

if results_dfs:
    results_df = pd.concat(results_dfs, ignore_index=True)
    print(f"✓ Combined {len(results_dfs)} results files")
    print(f"  Total results records: {len(results_df):,}")
else:
    print("✗ No results data loaded!")
    results_df = pd.DataFrame()

# ==============================================================================
# 3. CREATE SCENARIO MAPPINGS
# ==============================================================================
# Create scenario mapping for cleaner labels
scenario_map = {
    f'budget_{b}_equity_{e}': f'${b/1e6:.0f}M, Equity={e}'
    for b in budgets
    for e in equity_weights
}

# Add mapped labels to dataframes
if not equity_df.empty:
    equity_df['scenario_label'] = equity_df['scenario'].map(scenario_map)
if not results_df.empty:
    results_df['scenario_label'] = results_df['scenario'].map(scenario_map)

print("\n" + "="*70)
print("DATA LOADING COMPLETE")
print("="*70)
print(f"Budgets analyzed: {budgets}")
print(f"Equity weights analyzed: {equity_weights}")
print(f"Total combinations: {len(budgets) * len(equity_weights)}")
print(f"Successfully loaded: {len(equity_dfs)} combinations")




def aggregate_results(df ):
    """Aggregate metrics across epistemic runs"""
    
    # Count number of buildings per scenario/epistemic run
    df['num_buildings'] = 1 # Each row is a building
    
    # Calculate total budget spent per epistemic run
    df_summary = df.groupby(['scenario', 'scenario_label', 'epistemic_run']).agg({
        # NOTE: If 'cost of interventon_mean' is the actual cost per building (as implied by raw data), 
        # then 'sum' correctly calculates the total spent per run.
        'cost_of_intervention_mean': ['mean', 'sum'], # mean per building, sum = total spent
        'total_ton_co2_saved': 'sum', # total CO2 saved across all buildings
        'cost_per_net_ton_co2_kg': 'mean', # average cost effectiveness (mean of ratios across buildings)
        'weighted_cost_per_net_ton': 'mean', # average weighted cost
        'remaining_funds': 'first', # should be same for all buildings in a run
        'num_buildings': 'sum' # total number of buildings retrofitted
    }).reset_index()
    
    # Flatten column names
    df_summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in df_summary.columns.values]
    
    # Now aggregate across epistemic runs
    agg_dict = {
        'cost_of_intervention_mean_mean': ['mean', 'std'], # avg cost per building
        'cost_of_intervention_mean_sum': ['mean', 'std'], # total budget spent
        'total_ton_co2_saved_sum': ['mean', 'std'], # total CO2 saved
        'cost_per_net_ton_co2_kg_mean': ['mean', 'std'], # avg cost effectiveness
        'weighted_cost_per_net_ton_mean': ['mean', 'std'], # avg weighted cost
        'remaining_funds_first': ['mean', 'std'], # remaining funds
        'num_buildings_sum': ['mean', 'std'] # number of buildings
    }
    
    aggregated = df_summary.groupby(['scenario', 'scenario_label']).agg(agg_dict).reset_index()
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in aggregated.columns.values]
    
    return aggregated

def aggregate_equity(df, group_cols=['scenario', 'scenario_label']):
    """Aggregate equity metrics across epistemic runs"""
    
    # ... (function body unchanged)
    agg_dict = {
        'vulnerable_pct': ['mean', 'std'],
        'equity_concentration': ['mean', 'std'],
        'deprived_count': ['mean', 'std'],
        'struggling_count': ['mean', 'std'],
        'lower middle_count': ['mean', 'std'],
        'upper middle_count': ['mean', 'std'],
        'affluent_count': ['mean', 'std'],
        'student_count': ['mean', 'std'],
        'deprived_pct': ['mean', 'std'],
        'struggling_pct': ['mean', 'std'],
        'lower middle_pct': ['mean', 'std'],
        'upper middle_pct': ['mean', 'std'],
        'affluent_pct': ['mean', 'std'],
        'student_pct': ['mean', 'std']
    }
    
    aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in aggregated.columns.values]
    
    return aggregated

# Aggregate results
results_agg = aggregate_results(results_df)
equity_agg = aggregate_equity(equity_df)

# Merge for comprehensive view
comparison_df = results_agg.merge(equity_agg, on='scenario', how='left')

# Sort by equity weight for better visualization
equity_order = ['Equity Weight = 0', 'Equity Weight = 0.5', 'Equity Weight = 1.0']
results_agg['sort_order'] = results_agg['scenario'].map({s: i for i, s in enumerate(equity_order)})
results_agg = results_agg.sort_values('sort_order').drop('sort_order', axis=1)
equity_agg['sort_order'] = equity_agg['scenario'].map({s: i for i, s in enumerate(equity_order)})
equity_agg = equity_agg.sort_values('sort_order').drop('sort_order', axis=1)

print("=" * 80)
print("EQUITY WEIGHTING COMPARISON SUMMARY")
print("=" * 80)
print(comparison_df.to_string())
print("\n")


 
# ==============================================================================
# 3. CREATE COMPARISON METRICS TABLE
# ==============================================================================

def create_comparison_table(results_agg, equity_agg):
    """Create formatted comparison table"""
    # ... (function body unchanged)
    comparison = []
    
    for _, row in results_agg.iterrows():
        scenario = row['scenario_label']
        equity_row = equity_agg[equity_agg['scenario_label'] == scenario].iloc[0]
        
        comparison.append({
            'Scenario': scenario,
            'Buildings Retrofitted': f"{row['num_buildings_sum_mean']:.0f} ± {row['num_buildings_sum_std']:.0f}",
            'Total Budget Spent (£M)': f"{row['cost_of_intervention_mean_sum_mean']/1e6:.2f} ± {row['cost_of_intervention_mean_sum_std']/1e6:.2f}",
            'Avg Cost/Building (£k)': f"{row['cost_of_intervention_mean_mean_mean']/1e3:.1f} ± {row['cost_of_intervention_mean_mean_std']/1e3:.1f}",
            'Total CO2 Saved (kton)': f"{row['total_ton_co2_saved_sum_mean']/1e3:.2f} ± {row['total_ton_co2_saved_sum_std']/1e3:.2f}",
            'Cost/Ton CO2 (£/kg)': f"{row['cost_per_net_ton_co2_kg_mean_mean']:.2f} ± {row['cost_per_net_ton_co2_kg_mean_std']:.2f}",
            'Vulnerable Coverage (%)': f"{equity_row['vulnerable_pct_mean']*100:.1f} ± {equity_row['vulnerable_pct_std']*100:.1f}",
            'Equity Concentration': f"{equity_row['equity_concentration_mean']:.3f} ± {equity_row['equity_concentration_std']:.3f}",
            'Deprived Coverage (%)': f"{equity_row['deprived_pct_mean']*100:.1f} ± {equity_row['deprived_pct_std']*100:.1f}",
            'Remaining Funds (£M)': f"{row['remaining_funds_first_mean']/1e6:.2f} ± {row['remaining_funds_first_std']/1e6:.2f}"
        })
    
    return pd.DataFrame(comparison)

comparison_table = create_comparison_table(results_agg, equity_agg)
print("\n" + "=" * 80)
print("KEY METRICS COMPARISON")
print("=" * 80)
print(comparison_table.to_string(index=False))
print("\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==============================================================================
# 4. PREPARE DATA FOR VISUALIZATION
# ==============================================================================

def prepare_aggregated_data(results_df, equity_df):
    """Aggregate data across runs for each scenario"""
    
    # Aggregate results by scenario - using sum for totals, mean for per-building metrics
    results_agg = results_df.groupby(['scenario', 'budget', 'equity_weight']).agg({
        'total_ton_co2_saved': ['sum', 'std'],  # Total across all runs
        'cost_per_net_ton_co2_kg': ['mean', 'std'],  # Average cost
        'cost_of_intervention_mean': ['mean', 'std'],  # Average intervention cost
        'num_buildings': ['sum', 'std']  # Total buildings
    }).reset_index()
    
    # Flatten column names
    results_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in results_agg.columns.values]
    
    # Aggregate equity by scenario
    equity_agg = equity_df.groupby(['scenario', 'budget', 'equity_weight']).agg({
        'vulnerable_pct': ['mean', 'std'],
        'equity_concentration': ['mean', 'std'],
        'deprived_pct': ['mean', 'std'],
        'struggling_pct': ['mean', 'std'],
        'lower middle_pct': ['mean', 'std'],
        'upper middle_pct': ['mean', 'std'],
        'affluent_pct': ['mean', 'std'],
        'student_pct': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    equity_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                         for col in equity_agg.columns.values]
    
    return results_agg, equity_agg

# Prepare aggregated data
results_agg, equity_agg = prepare_aggregated_data(results_df, equity_df)

# Print column names to verify
print("\nResults aggregated columns:")
print(results_agg.columns.tolist())
print("\nEquity aggregated columns:")
print(equity_agg.columns.tolist())

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================

def get_color_palette(n_colors):
    """Generate a color palette with n colors"""
    if n_colors <= 3:
        return ['#e74c3c', '#f39c12', '#27ae60'][:n_colors]
    else:
        # Use seaborn color palette for more colors
        return sns.color_palette('husl', n_colors).as_hex()

def create_scenario_colors(scenarios, results_agg):
    """Create color mapping for scenarios based on equity weights"""
    # Get unique equity weights and sort them
    equity_weights = sorted(results_agg['equity_weight'].unique())
    colors = get_color_palette(len(equity_weights))
    weight_to_color = dict(zip(equity_weights, colors))
    
    scenario_colors = {}
    for scenario in scenarios:
        # Extract equity weight from scenario
        weight = results_agg[results_agg['scenario'] == scenario]['equity_weight'].iloc[0]
        scenario_colors[scenario] = weight_to_color[weight]
    
    return scenario_colors

# ==============================================================================
# 6. INDIVIDUAL PLOT FUNCTIONS
# ==============================================================================

def plot_carbon_savings_vs_equity(results_subset, equity_weights, budget_label, filename):
    """Plot 1: Carbon Savings vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in results_subset['budget'].unique():
        subset = results_subset[results_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset['total_ton_co2_saved_sum'].values / 1e3
        stds = subset['total_ton_co2_saved_std'].values / 1e3
        
        label = f'£{budget_val/1e6:.0f}M' if len(results_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    ax.set_title(f'Carbon Savings vs Equity Weight\n{budget_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(results_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_cost_effectiveness_vs_equity(results_subset, equity_weights, budget_label, filename):
    """Plot 2: Cost Effectiveness vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in results_subset['budget'].unique():
        subset = results_subset[results_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset['cost_per_net_ton_co2_kg_mean'].values
        stds = subset['cost_per_net_ton_co2_kg_std'].values
        
        label = f'£{budget_val/1e6:.0f}M' if len(results_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost per Ton CO2 (£/kg)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cost Effectiveness vs Equity Weight\n{budget_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(results_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_vulnerable_coverage_vs_equity(equity_subset, equity_weights, budget_label, filename):
    """Plot 3: Vulnerable Coverage vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in equity_subset['budget'].unique():
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset['vulnerable_pct_mean'].values * 100
        stds = subset['vulnerable_pct_std'].values * 100
        
        label = f'£{budget_val/1e6:.0f}M' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Vulnerable Population Coverage vs Equity Weight\n{budget_label}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(equity_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_equity_concentration_vs_weight(equity_subset, equity_weights, budget_label, filename):
    """Plot 4: Equity Concentration vs Equity Weight"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    for budget_val in equity_subset['budget'].unique():
        subset = equity_subset[equity_subset['budget'] == budget_val]
        weights = subset['equity_weight'].values
        means = subset['equity_concentration_mean'].values
        stds = subset['equity_concentration_std'].values
        
        label = f'£{budget_val/1e6:.0f}M' if len(equity_subset['budget'].unique()) > 1 else None
        ax.errorbar(weights, means, yerr=stds, fmt='o-', markersize=10, 
                    linewidth=2, capsize=5, label=label, alpha=0.7)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Equity Concentration Index', fontsize=14, fontweight='bold')
    ax.set_title(f'Equity Concentration vs Equity Weight\n(lower = more equitable)\n{budget_label}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(equity_weights)
    if len(equity_subset['budget'].unique()) > 1:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_socioeconomic_distribution(equity_subset, scenarios, scenario_colors, budget_label, filename):
    """Plot 5: Socio-economic Distribution by Equity Weight"""
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    
    socio_groups = ['deprived_pct', 'struggling_pct', 'lower middle_pct', 
                    'upper middle_pct', 'affluent_pct', 'student_pct']
    socio_labels = ['Deprived', 'Struggling', 'Lower\nMiddle', 
                    'Upper\nMiddle', 'Affluent', 'Student']
    
    x = np.arange(len(socio_labels))
    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios
    
    for i, scenario in enumerate(scenarios):
        equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
        means = [equity_row[f'{group}_mean'] * 100 for group in socio_groups]
        offset = (i - n_scenarios/2 + 0.5) * width
        
        weight = equity_row['equity_weight']
        label = f'EW={weight}'
        ax.bar(x + offset, means, width, label=label, 
               color=scenario_colors[scenario], alpha=0.7)
    
    ax.set_xlabel('Socio-economic Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Socio-economic Distribution by Equity Weight\n{budget_label}', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(socio_labels, fontsize=11)
    ax.legend(fontsize=11, ncol=min(2, (n_scenarios + 1) // 2))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_pareto_front(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename):
    """Plot 6: Pareto Front - Equity vs Carbon"""
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    for scenario in scenarios:
        equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
        results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
        
        vuln_mean = equity_row['vulnerable_pct_mean'] * 100
        vuln_std = equity_row['vulnerable_pct_std'] * 100
        co2_mean = results_row['total_ton_co2_saved_sum'] / 1e3
        co2_std = results_row['total_ton_co2_saved_std'] / 1e3
        
        weight = equity_row['equity_weight']
        label = f'EW={weight}'
        
        ax.errorbar(vuln_mean, co2_mean, xerr=vuln_std, yerr=co2_std,
                   fmt='o', markersize=12, capsize=5,
                   label=label, color=scenario_colors[scenario])
    
    # Draw connecting line (sorted by equity weight)
    sorted_scenarios = sorted(scenarios, 
                             key=lambda s: equity_subset[equity_subset['scenario'] == s]['equity_weight'].iloc[0])
    vuln_means = [equity_subset[equity_subset['scenario'] == s]['vulnerable_pct_mean'].iloc[0] * 100 
                  for s in sorted_scenarios]
    co2_means = [results_subset[results_subset['scenario'] == s]['total_ton_co2_saved_sum'].iloc[0] / 1e3 
                 for s in sorted_scenarios]
    ax.plot(vuln_means, co2_means, '--', alpha=0.3, color='gray', linewidth=2)
    
    ax.set_xlabel('Vulnerable Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CO2 Saved (kton)', fontsize=14, fontweight='bold')
    ax.set_title(f'Equity-Carbon Tradeoff Curve\n{budget_label}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, ncol=min(2, (len(scenarios) + 1) // 2))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_vulnerable_groups_coverage(equity_subset, scenarios, equity_weights, budget_label, filename):
    """Plot 7: Most Vulnerable Groups Coverage"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    deprived_means = [equity_subset[equity_subset['scenario'] == s]['deprived_pct_mean'].iloc[0] * 100 
                      for s in scenarios]
    struggling_means = [equity_subset[equity_subset['scenario'] == s]['struggling_pct_mean'].iloc[0] * 100 
                        for s in scenarios]
    
    bars1 = ax.bar(x_pos - width/2, deprived_means, width, label='Deprived', 
                   color='#c0392b', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, struggling_means, width, label='Struggling', 
                   color='#e67e22', alpha=0.8)
    
    ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Most Vulnerable Groups Coverage\n{budget_label}', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{w}' for w in equity_weights])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_tradeoff_efficiency(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename):
    """Plot 8: Tradeoff Efficiency"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    if len(scenarios) >= 2:
        # Use minimum equity weight as baseline
        sorted_scenarios = sorted(scenarios, 
                                 key=lambda s: equity_subset[equity_subset['scenario'] == s]['equity_weight'].iloc[0])
        base_scenario = sorted_scenarios[0]
        
        base_vuln = equity_subset[equity_subset['scenario'] == base_scenario]['vulnerable_pct_mean'].iloc[0] * 100
        base_co2 = results_subset[results_subset['scenario'] == base_scenario]['total_ton_co2_saved_sum'].iloc[0] / 1e3
        
        tradeoff_scenarios = []
        tradeoff_ratios = []
        tradeoff_labels = []
        
        for scenario in sorted_scenarios[1:]:
            equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
            results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
            
            vuln_cov = equity_row['vulnerable_pct_mean'] * 100
            co2_saved = results_row['total_ton_co2_saved_sum'] / 1e3
            
            vuln_gain = vuln_cov - base_vuln
            co2_loss = base_co2 - co2_saved
            
            if co2_loss > 0:
                ratio = vuln_gain / co2_loss
            else:
                ratio = vuln_gain / 0.001  # Avoid division by zero
            
            tradeoff_scenarios.append(scenario)
            tradeoff_ratios.append(ratio)
            tradeoff_labels.append(f'{equity_row["equity_weight"]}')
        
        colors_for_bars = [scenario_colors[s] for s in tradeoff_scenarios]
        bars = ax.bar(range(len(tradeoff_ratios)), tradeoff_ratios, 
                     color=colors_for_bars, alpha=0.7)
        
        ax.set_xlabel('Equity Weight', fontsize=14, fontweight='bold')
        ax.set_ylabel('Vulnerable % Gain per kton CO2 Lost', fontsize=14, fontweight='bold')
        ax.set_title(f'Equity-Carbon Tradeoff Efficiency\n{budget_label}', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(tradeoff_ratios)))
        ax.set_xticklabels(tradeoff_labels)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_radar_chart(results_subset, equity_subset, scenarios, scenario_colors, budget_label, filename):
    """Plot 9: Multi-Metric Radar Chart"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    metrics = ['Carbon\nSavings', 'Cost\nEfficiency', 'Vulnerable\nCoverage', 
               'Equity\nBalance', '# Buildings']
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for scenario in scenarios:
        results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
        equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
        
        # Normalize metrics (0-1, higher is better)
        carbon_norm = results_row['total_ton_co2_saved_sum'] / results_subset['total_ton_co2_saved_sum'].max()
        cost_eff_norm = (1 / results_row['cost_per_net_ton_co2_kg_mean']) / (1 / results_subset['cost_per_net_ton_co2_kg_mean']).max()
        vuln_norm = equity_row['vulnerable_pct_mean'] / equity_subset['vulnerable_pct_mean'].max()
        equity_norm = 1 - (equity_row['equity_concentration_mean'] / equity_subset['equity_concentration_mean'].max())
        buildings_norm = results_row['num_buildings_sum'] / results_subset['num_buildings_sum'].max()
        
        values = [carbon_norm, cost_eff_norm, vuln_norm, equity_norm, buildings_norm]
        values += values[:1]
        
        weight = equity_row['equity_weight']
        label = f'EW={weight}'
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=label, 
                color=scenario_colors[scenario], markersize=8)
        ax.fill(angles, values, alpha=0.2, color=scenario_colors[scenario])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(f'Normalized Performance Comparison\n{budget_label}', fontsize=16, 
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11,
              ncol=1 if len(scenarios) <= 5 else 2)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# ==============================================================================
# 7. CREATE ALL VISUALIZATIONS
# ==============================================================================

def create_all_visualizations(results_agg, equity_agg, budget=None, op_path='') :
    """Create all individual plots for a given budget"""
    os.makedirs(op_path, exist_ok=True )
    # Filter by budget if specified
    if budget is not None:
        results_subset = results_agg[results_agg['budget'] == budget].copy()
        equity_subset = equity_agg[equity_agg['budget'] == budget].copy()
        budget_label = f'Budget: £{budget/1e6:.0f}M'
        file_suffix = f'_budget_{int(budget/1e6)}M'
    else:
        results_subset = results_agg.copy()
        equity_subset = equity_agg.copy()
        budget_label = 'All Budgets'
        file_suffix = '_all_budgets'
    
    scenarios = results_subset['scenario'].unique()
    equity_weights = sorted(results_subset['equity_weight'].unique())
    
    # Create color mapping
    scenario_colors = create_scenario_colors(scenarios, results_subset)
    
    print(f"\n{'='*70}")
    print(f"Creating visualizations for: {budget_label}")
    print(f"{'='*70}")
    
    # Create all plots
    plot_carbon_savings_vs_equity(
        results_subset, equity_weights, budget_label, 
        f'{op_path}/01_carbon_savings_vs_equity{file_suffix}.png'
    )
    
    plot_cost_effectiveness_vs_equity(
        results_subset, equity_weights, budget_label,
        f'{op_path}/02_cost_effectiveness_vs_equity{file_suffix}.png'
    )
    
    plot_vulnerable_coverage_vs_equity(
        equity_subset, equity_weights, budget_label,
        f'{op_path}/03_vulnerable_coverage_vs_equity{file_suffix}.png'
    )
    
    plot_equity_concentration_vs_weight(
        equity_subset, equity_weights, budget_label,
        f'{op_path}/04_equity_concentration_vs_weight{file_suffix}.png'
    )
    
    plot_socioeconomic_distribution(
        equity_subset, scenarios, scenario_colors, budget_label,
        f'{op_path}/05_socioeconomic_distribution{file_suffix}.png'
    )
    
    plot_pareto_front(
        results_subset, equity_subset, scenarios, scenario_colors, budget_label,
        f'{op_path}/06_pareto_front{file_suffix}.png'
    )
    
    plot_vulnerable_groups_coverage(
        equity_subset, scenarios, equity_weights, budget_label,
        f'{op_path}/07_vulnerable_groups_coverage{file_suffix}.png'
    )
    
    plot_tradeoff_efficiency(
        results_subset, equity_subset, scenarios, scenario_colors, budget_label,
        f'{op_path}/08_tradeoff_efficiency{file_suffix}.png'
    )
    
    plot_radar_chart(
        results_subset, equity_subset, scenarios, scenario_colors, budget_label,
        f'{op_path}/09_radar_chart{file_suffix}.png'
    )

# ==============================================================================
# 8. GENERATE PLOTS FOR ALL BUDGETS
# ==============================================================================

unique_budgets = results_agg['budget'].unique()
op_path= '/Users/gracecolverd/RetrofitModel/test/greedy_vis'
if len(unique_budgets) == 1:
    # Single budget analysis
    create_all_visualizations(results_agg, equity_agg, budget=unique_budgets[0], op_path = f'{op_path}/single')
else:
    # Create plots for each budget
    for budget in unique_budgets:
        op_path_b = f'{op_path}/{budget}'
        create_all_visualizations(results_agg, equity_agg, budget=budget)
    
    # Also create combined plots for all budgets
    create_all_visualizations(results_agg, equity_agg, budget=None, op_path = f'{op_path}/combo')

# ==============================================================================
# 9. DETAILED STATISTICAL COMPARISON
# ==============================================================================

def statistical_comparison(results_df, equity_df, budget=None):
    """Perform statistical tests between equity weightings"""
    
    # Filter by budget if specified
    if budget is not None:
        results_subset = results_df[results_df['budget'] == budget]
        equity_subset = equity_df[equity_df['budget'] == budget]
        print(f"\n{'=' * 80}")
        print(f"STATISTICAL COMPARISONS (t-tests) - Budget: £{budget/1e6:.0f}M")
        print(f"{'=' * 80}")
    else:
        results_subset = results_df
        equity_subset = equity_df
        print(f"\n{'=' * 80}")
        print("STATISTICAL COMPARISONS (t-tests) - All Budgets")
        print(f"{'=' * 80}")
    
    scenarios = results_subset['scenario'].unique()
    
    # Results metrics
    metrics = ['total_ton_co2_saved', 'cost_per_net_ton_co2_kg', 'cost of interventon_mean']
    
    for metric in metrics:
        if metric not in results_subset.columns:
            continue
            
        print(f"\n{metric.upper()}:")
        print("-" * 80)
        
        for i, s1 in enumerate(scenarios):
            for s2 in scenarios[i+1:]:
                data1 = results_subset[results_subset['scenario'] == s1][metric].dropna().values
                data2 = results_subset[results_subset['scenario'] == s2][metric].dropna().values
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                mean1, mean2 = np.mean(data1), np.mean(data2)
                if mean1 != 0:
                    pct_change = ((mean2 - mean1) / mean1) * 100
                else:
                    pct_change = 0
                
                print(f"{s1} vs {s2}:")
                print(f"  t={t_stat:.3f}, p={p_value:.4f} {sig}")
                print(f"  Change: {pct_change:+.1f}%")
    
    # Equity metrics
    print(f"\nEQUITY METRICS:")
    print("-" * 80)
    
    equity_metrics = ['vulnerable_pct', 'equity_concentration', 'deprived_pct']
    
    for metric in equity_metrics:
        if metric not in equity_subset.columns:
            continue
            
        print(f"\n{metric}:")
        for i, s1 in enumerate(scenarios):
            for s2 in scenarios[i+1:]:
                data1 = equity_subset[equity_subset['scenario'] == s1][metric].dropna().values
                data2 = equity_subset[equity_subset['scenario'] == s2][metric].dropna().values
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                mean1, mean2 = np.mean(data1), np.mean(data2)
                if mean1 != 0:
                    pct_change = ((mean2 - mean1) / mean1) * 100
                else:
                    pct_change = 0
                
                print(f"  {s1} vs {s2}: t={t_stat:.3f}, p={p_value:.4f} {sig}, Change: {pct_change:+.1f}%")

# Run statistical comparisons
for budget in unique_budgets:
    statistical_comparison(results_df, equity_df, budget=budget)

# ==============================================================================
# 10. TRADEOFF SUMMARY
# ==============================================================================

def calculate_tradeoffs(results_agg, equity_agg, budget=None):
    """Calculate key tradeoff metrics"""
    
    # Filter by budget if specified
    if budget is not None:
        results_subset = results_agg[results_agg['budget'] == budget]
        equity_subset = equity_agg[equity_agg['budget'] == budget]
        print(f"\n{'=' * 80}")
        print(f"TRADEOFF ANALYSIS SUMMARY - Budget: £{budget/1e6:.0f}M")
        print(f"{'=' * 80}")
    else:
        results_subset = results_agg
        equity_subset = equity_agg
        print(f"\n{'=' * 80}")
        print("TRADEOFF ANALYSIS SUMMARY - All Budgets")
        print(f"{'=' * 80}")
    
    # Sort scenarios by equity weight
    scenarios = sorted(results_subset['scenario'].unique(), 
                      key=lambda s: results_subset[results_subset['scenario'] == s]['equity_weight'].iloc[0])
    
    base_scenario = scenarios[0]  # Lowest equity weight
    base_results = results_subset[results_subset['scenario'] == base_scenario].iloc[0]
    base_equity = equity_subset[equity_subset['scenario'] == base_scenario].iloc[0]
    
    base_co2 = base_results['total_ton_co2_saved_sum']
    base_vuln = base_equity['vulnerable_pct_mean'] * 100
    
    print(f"\nBaseline (Equity Weight = {base_results['equity_weight']}):")
    print(f"  CO2 Saved: {base_co2/1e3:.2f} kton")
    print(f"  Vulnerable Coverage: {base_vuln:.1f}%")
    print(f"  Cost per Ton: £{base_results['cost_per_net_ton_co2_kg_mean']:.2f}/kg")
    
    for scenario in scenarios[1:]:
        results_row = results_subset[results_subset['scenario'] == scenario].iloc[0]
        equity_row = equity_subset[equity_subset['scenario'] == scenario].iloc[0]
        
        co2_saved = results_row['total_ton_co2_saved_sum']
        vuln_cov = equity_row['vulnerable_pct_mean'] * 100
        
        co2_loss = base_co2 - co2_saved
        vuln_gain = vuln_cov - base_vuln
        
        print(f"\nEquity Weight = {results_row['equity_weight']}:")
        print(f"  CO2 Saved: {co2_saved/1e3:.2f} kton ({-co2_loss/1e3:+.2f} kton vs baseline)")
        print(f"  Vulnerable Coverage: {vuln_cov:.1f}% ({vuln_gain:+.1f}% vs baseline)")
        print(f"  Cost per Ton: £{results_row['cost_per_net_ton_co2_kg_mean']:.2f}/kg")
        
        if co2_loss > 0:
            efficiency = vuln_gain / (co2_loss / 1e3)
            print(f"  Tradeoff: {vuln_gain:.1f}% vulnerable gain for {co2_loss/1e3:.2f} kton CO2 lost")
            print(f"  Efficiency: {efficiency:.2f}% vulnerable per kton CO2")
        elif co2_loss < 0:
            print(f"  Win-win: Gained both {vuln_gain:.1f}% vulnerable AND {-co2_loss/1e3:.2f} kton CO2!")

# Calculate tradeoffs for each budget
for budget in unique_budgets:
    calculate_tradeoffs(results_agg, equity_agg, budget=budget)

# ==============================================================================
# 11. CREATE COMPARISON TABLE
# ==============================================================================

def create_comparison_table(results_agg, equity_agg):
    """Create summary comparison table"""
    
    comparison_data = []
    
    for _, results_row in results_agg.iterrows():
        scenario = results_row['scenario']
        equity_row = equity_agg[equity_agg['scenario'] == scenario].iloc[0]
        
        comparison_data.append({
            'scenario': scenario,
            'budget': results_row['budget'],
            'equity_weight': results_row['equity_weight'],
            'co2_saved_kton': results_row['total_ton_co2_saved_sum'] / 1e3,
            'co2_saved_std_kton': results_row['total_ton_co2_saved_std'] / 1e3,
            'cost_per_ton_kg': results_row['cost_per_net_ton_co2_kg_mean'],
            'vulnerable_pct': equity_row['vulnerable_pct_mean'] * 100,
            'deprived_pct': equity_row['deprived_pct_mean'] * 100,
            'struggling_pct': equity_row['struggling_pct_mean'] * 100,
            'equity_concentration': equity_row['equity_concentration_mean'],
            'num_buildings': results_row['num_buildings_sum']
        })
    
    comparison_table = pd.DataFrame(comparison_data)
    return comparison_table

comparison_table = create_comparison_table(results_agg, equity_agg)

# ==============================================================================
# 12. EXPORT SUMMARY REPORT
# ==============================================================================

def export_summary_report(comparison_table, results_agg, equity_agg):
    """Export comprehensive summary to CSV"""
    
    # Create detailed export
    export_df = results_agg.merge(equity_agg, on=['scenario', 'budget', 'equity_weight'])
    export_df.to_csv('equity_weighting_comparison_detailed.csv', index=False)
    comparison_table.to_csv('equity_weighting_comparison_summary.csv', index=False)
    
    print("\n" + "=" * 80)
    print("EXPORTED FILES")
    print("=" * 80)
    print("\nCSV Files:")
    print("- equity_weighting_comparison_detailed.csv")
    print("- equity_weighting_comparison_summary.csv")
    
    print("\nVisualization Files (9 plots per budget):")
    for budget in unique_budgets:
        suffix = f'_budget_{int(budget/1e6)}M'
        print(f"\n  Budget £{budget/1e6:.0f}M:")
        for i in range(1, 10):
            print(f"  - {i:02d}_*{suffix}.png")
    
    if len(unique_budgets) > 1:
        print("\n  All Budgets Combined:")
        for i in range(1, 10):
            print(f"  - {i:02d}_*_all_budgets.png")

export_summary_report(comparison_table, results_agg, equity_agg)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Analyzed {len(unique_budgets)} budget(s) with {len(results_agg['equity_weight'].unique())} equity weight(s)")
print(f"Total scenarios: {len(results_agg)}")
print(f"Total plots created: {9 * (len(unique_budgets) + (1 if len(unique_budgets) > 1 else 0))}")