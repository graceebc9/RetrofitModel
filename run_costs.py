

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob 
YEARS=5
N_SIMULATIONS=5000

# Carbon factors (kg CO2/kWh)
GAS_CARBON_FACTOR=0.18      
ELEC_CARBON_FACTOR=0.19338  


files  = glob.glob( '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/mega_all_scenarios/NE/*.csv')
 


res =[]
for f in files:
    df = pd.read_csv(f)
    res.append(df)
res_df = pd.concat(res)


scenarios = ["wall_installation",  "loft_installation", "join_heat_ins_decay",  "heat_pump_only"]
df_processed=res_df.copy() 

for scenario_name in scenarios: 
        print(scenario_name)
        measure_type=scenario_name
        df_processed = clean_post_proccess(
        df_processed, 
        measure_type, 
        scenario_name, 
        years=YEARS,
        GAS_CARBON_FACTOR_2022=gas_carbon_factor,
        elec_carbon_factor=elec_carbon_factor,
        n_simulations=n_simulations
    )
    
pdf=df_processed[df_processed['premise_type']!='Domestic outbuilding']
# ============================================================================
# STEP 0: DATA VALIDATION
# ============================================================================

print("="*80)
print("DATA VALIDATION")
print("="*80)
print(f"Original data shape: {pdf.shape}")
print(f"Unique buildings: {pdf['verisk_building_id'].nunique()}")

# ============================================================================
# STEP 1: AGGREGATE ACROSS EPISTEMIC RUNS
# ============================================================================

df_agg = pdf.groupby('verisk_building_id').agg({
    # Mean of means
    'cost_per_net_ton_co2_loft_installation_mean_thousands': 'mean',
    'cost_per_net_ton_co2_wall_installation_mean_thousands': 'mean',
    'cost_per_net_ton_co2_join_heat_ins_decay_mean_thousands': 'mean',
    'cost_per_net_ton_co2_heat_pump_only_mean_thousands': 'mean',
    
    # Propagate aleatory std (average the variances, then sqrt)
    'cost_per_net_ton_co2_loft_installation_std_thousands': lambda x: np.sqrt(np.mean(x**2)),
    'cost_per_net_ton_co2_wall_installation_std_thousands': lambda x: np.sqrt(np.mean(x**2)),
    'cost_per_net_ton_co2_join_heat_ins_decay_std_thousands': lambda x: np.sqrt(np.mean(x**2)),
    'cost_per_net_ton_co2_heat_pump_only_std_thousands': lambda x: np.sqrt(np.mean(x**2)),
    
    # Carbon savings (negative = reduction, positive = increase)
    'total_tonne_co2_saved_loft_installation_5yr_mean': 'mean',
    'total_tonne_co2_saved_wall_installation_5yr_mean': 'mean',
    'total_tonne_co2_saved_join_heat_ins_decay_5yr_mean': 'mean',
    'total_tonne_co2_saved_heat_pump_only_5yr_mean': 'mean',
    
    # Building characteristics
    'premise_type': 'first',
    'avg_gas_percentile': 'first'
}).reset_index()

# ============================================================================
# STEP 2: RESHAPE TO LONG FORMAT
# ============================================================================

scenarios = {
    'Wall Insulation': {
        'cost': 'cost_per_net_ton_co2_wall_installation_mean_thousands',
        'std': 'cost_per_net_ton_co2_wall_installation_std_thousands',
        'carbon': 'total_tonne_co2_saved_wall_installation_5yr_mean'
    },
    'Loft Insulation': {
        'cost': 'cost_per_net_ton_co2_loft_installation_mean_thousands',
        'std': 'cost_per_net_ton_co2_loft_installation_std_thousands',
        'carbon': 'total_tonne_co2_saved_loft_installation_5yr_mean'
    },
    'Heat Pump + Insulation': {
        'cost': 'cost_per_net_ton_co2_join_heat_ins_decay_mean_thousands',
        'std': 'cost_per_net_ton_co2_join_heat_ins_decay_std_thousands',
        'carbon': 'total_tonne_co2_saved_join_heat_ins_decay_5yr_mean'
    },
    'Heat Pump Only': {
        'cost': 'cost_per_net_ton_co2_heat_pump_only_mean_thousands',
        'std': 'cost_per_net_ton_co2_heat_pump_only_std_thousands',
        'carbon': 'total_tonne_co2_saved_heat_pump_only_5yr_mean'
    }
}

df_long = []
for scenario_name, cols in scenarios.items():
    temp = df_agg[['verisk_building_id', 'premise_type', 'avg_gas_percentile']].copy()
    temp['intervention'] = scenario_name
    
    # Convert from thousands to actual values
    temp['cost_per_ton'] = df_agg[cols['cost']] * 1000
    temp['cost_per_ton_std'] = df_agg[cols['std']] * 1000
    
    # Keep original sign: negative = reduction (good)
    temp['carbon_savings_raw'] = df_agg[cols['carbon']]
    
    # For MACC display: convert to positive reductions
    # Multiply by -1 so reductions become positive
    temp['carbon_reduction'] = -df_agg[cols['carbon']]
    
    df_long.append(temp)

df_long = pd.concat(df_long, ignore_index=True)

# ============================================================================
# STEP 3: ANALYZE PROBLEMATIC MEASURES
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS OF COUNTERPRODUCTIVE MEASURES")
print("="*80)

# Buildings where measures INCREASE emissions (positive carbon_savings_raw)
counterproductive = df_long[df_long['carbon_savings_raw'] > 0]
print(f"\nTotal counterproductive interventions: {len(counterproductive)} ({100*len(counterproductive)/len(df_long):.1f}%)")
print(f"  Emissions increase: {counterproductive['carbon_savings_raw'].sum():,.1f} tCO₂")

print("\nBy intervention type:")
for intervention in df_long['intervention'].unique():
    subset = counterproductive[counterproductive['intervention'] == intervention]
    print(f"\n{intervention}:")
    print(f"  Count: {len(subset)}")
    if len(subset) > 0:
        print(f"  By gas percentile:")
        percentile_dist = subset.groupby('avg_gas_percentile').size().sort_index()
        for pct, count in percentile_dist.items():
            print(f"    Percentile {pct}: {count} buildings")

# ============================================================================
# STEP 4: CREATE MACC (ONLY BENEFICIAL MEASURES)
# ============================================================================

print("\n" + "="*80)
print("CREATING MACC CURVE")
print("="*80)

# Filter to only measures that REDUCE emissions (negative carbon_savings_raw)
# Also filter out extreme outliers and invalid data
df_macc = df_long[
    (df_long['carbon_savings_raw'] < 0) &  # Only actual reductions
    np.isfinite(df_long['cost_per_ton']) &
    np.isfinite(df_long['carbon_reduction']) &
    (df_long['carbon_reduction'] > 0.01) &  # Minimum threshold
    (df_long['cost_per_ton'].abs() < 100000)  # Remove extreme outliers (>£100k/ton)
].copy()

print(f"Interventions included in MACC: {len(df_macc)}")
print(f"Interventions excluded (counterproductive or invalid): {len(df_long) - len(df_macc)}")

# Sort by cost-effectiveness and create cumulative savings
df_macc = df_macc.sort_values('cost_per_ton').reset_index(drop=True)
df_macc['cumulative_reduction'] = df_macc['carbon_reduction'].cumsum()
df_macc['cumulative_start'] = df_macc['cumulative_reduction'].shift(1).fillna(0)

# ============================================================================
# STEP 5: PLOT MACC
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

colors = {
    'Wall Insulation': '#2E86AB',       # Blue
    'Loft Insulation': '#06A77D',       # Green
    'Heat Pump Only': '#A23B72',        # Purple
    'Heat Pump + Insulation': '#F18F01' # Orange
}

# Plot bars
for idx, row in df_macc.iterrows():
    ax.bar(row['cumulative_start'] + row['carbon_reduction']/2,
           row['cost_per_ton'],
           width=row['carbon_reduction'],
           color=colors[row['intervention']],
           edgecolor='white',
           linewidth=0.3,
           alpha=0.8)

# Add zero line
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

# Labels and title
ax.set_xlabel('Cumulative Carbon Reduction (tCO₂)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cost per Tonne CO₂ Reduced (£/tCO₂)', fontsize=13, fontweight='bold')
ax.set_title('Marginal Abatement Cost Curve - NE England Retrofit\n(Beneficial Measures Only)', 
             fontsize=15, fontweight='bold', pad=20)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors.keys()]
ax.legend(handles=legend_elements, loc='best', fontsize=11)

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Format axes
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

plt.tight_layout()
plt.savefig('macc_main_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 6: SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("MACC SUMMARY STATISTICS")
print("="*80)
print(f"Total buildings analyzed: {len(df_agg)}")
print(f"Total beneficial interventions: {len(df_macc)}")
print(f"Counterproductive interventions excluded: {len(df_long) - len(df_macc)}")

print(f"\nTotal carbon reduction potential: {df_macc['carbon_reduction'].sum():,.1f} tCO₂")
print(f"Mean cost per ton: £{df_macc['cost_per_ton'].mean():,.0f}/tCO₂")
print(f"Median cost per ton: £{df_macc['cost_per_ton'].median():,.0f}/tCO₂")

# Percentiles
print(f"\nCost percentiles:")
for p in [10, 25, 50, 75, 90]:
    val = df_macc['cost_per_ton'].quantile(p/100)
    print(f"  {p}th percentile: £{val:,.0f}/tCO₂")

# Negative cost measures
negative_cost = df_macc[df_macc['cost_per_ton'] < 0]
print(f"\nMeasures with negative cost (net savings):")
print(f"  Count: {len(negative_cost)} ({100*len(negative_cost)/len(df_macc):.1f}%)")
print(f"  Carbon reduction: {negative_cost['carbon_reduction'].sum():,.1f} tCO₂")
if len(negative_cost) > 0:
    print(f"  Average cost: £{negative_cost['cost_per_ton'].mean():,.0f}/tCO₂")

# Positive cost measures
positive_cost = df_macc[df_macc['cost_per_ton'] >= 0]
print(f"\nMeasures with positive cost:")
print(f"  Count: {len(positive_cost)} ({100*len(positive_cost)/len(df_macc):.1f}%)")
print(f"  Carbon reduction: {positive_cost['carbon_reduction'].sum():,.1f} tCO₂")
if len(positive_cost) > 0:
    print(f"  Average cost: £{positive_cost['cost_per_ton'].mean():,.0f}/tCO₂")

print("\nBy intervention type:")
summary_by_type = df_macc.groupby('intervention').agg({
    'verisk_building_id': 'count',
    'carbon_reduction': 'sum',
    'cost_per_ton': ['mean', 'median', 'min', 'max'],
}).round(0)
summary_by_type.columns = ['Count', 'Total_Reduction_tCO2', 'Mean_Cost', 'Median_Cost', 'Min_Cost', 'Max_Cost']
print(summary_by_type)

# ============================================================================
# STEP 7: ANALYSIS BY GAS PERCENTILE
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS BY GAS USAGE PERCENTILE")
print("="*80)

print("\nLoft Insulation - buildings by gas percentile:")
loft_data = df_long[df_long['intervention'] == 'Loft Insulation']
loft_summary = loft_data.groupby('avg_gas_percentile').agg({
    'carbon_savings_raw': ['count', 'mean'],
    'cost_per_ton': 'mean'
}).round(2)
loft_summary.columns = ['Count', 'Avg_Carbon_Change_tCO2', 'Avg_Cost_per_ton']
loft_summary['%_Beneficial'] = (loft_data.groupby('avg_gas_percentile')['carbon_savings_raw']
                                 .apply(lambda x: 100 * (x < 0).sum() / len(x))).round(1)
print(loft_summary)

print("\nWall Insulation - buildings by gas percentile:")
wall_data = df_long[df_long['intervention'] == 'Wall Insulation']
wall_summary = wall_data.groupby('avg_gas_percentile').agg({
    'carbon_savings_raw': ['count', 'mean'],
    'cost_per_ton': 'mean'
}).round(2)
wall_summary.columns = ['Count', 'Avg_Carbon_Change_tCO2', 'Avg_Cost_per_ton']
wall_summary['%_Beneficial'] = (wall_data.groupby('avg_gas_percentile')['carbon_savings_raw']
                                .apply(lambda x: 100 * (x < 0).sum() / len(x))).round(1)
print(wall_summary)

print("="*80)

# ============================================================================
# STEP 8: DIAGNOSTIC PLOTS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cost Distribution by Intervention Type (Beneficial Measures Only)', 
             fontsize=14, fontweight='bold')

for idx, (scenario, color) in enumerate(colors.items()):
    ax = axes[idx // 2, idx % 2]
    data = df_macc[df_macc['intervention'] == scenario]['cost_per_ton']
    
    if len(data) > 0:
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, 
                  label=f'Median: £{data.median():,.0f}')
        ax.axvline(data.mean(), color='darkred', linestyle=':', linewidth=2,
                  label=f'Mean: £{data.mean():,.0f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Cost per Tonne CO₂ (£/tCO₂)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{scenario} (n={len(data)})')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(scenario)

plt.tight_layout()
plt.savefig('macc_diagnostics_corrected.png', dpi=300, bbox_inches='tight')
plt.show()