import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.RetrofitUtils import filter_typology

# ==========================================
# CONFIGURATION
# ==========================================
# Point this to ONE of your raw log files (not the processed output)
RAW_FILE_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/120_log_file.csv' 
SIGMA = 1
 
SCENARIO='wall_installation' 
# ==========================================
# PROCESSING
# ==========================================
print(f"Loading {RAW_FILE_PATH}...")
try:
    df = pd.read_csv(RAW_FILE_PATH, on_bad_lines='skip', low_memory=False)
except:
    pass # Handle headerless if needed

df = filter_typology(df)

# We need TWO metrics: Cost/Ton and Absolute Carbon
# 1. Cost per Ton (The Ratio)
col_cost_ton = f'{SCENARIO}_capex_per_net_ton_co2_{SCENARIO}_p50'
# 2. Absolute Carbon (The Denominator)
col_abs_co2  = f'{SCENARIO}_total_energy_abs_co2_ton_samples_{SCENARIO}_p50'

# Grouping
grouped = df.groupby('upn')[[col_cost_ton, col_abs_co2]].agg(['mean', 'std'])

# Flatten columns
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# Rename for clarity
data = pd.DataFrame()
data['mean_cost_ton'] = grouped[f'{col_cost_ton}_mean']
data['std_cost_ton']  = grouped[f'{col_cost_ton}_std']
data['mean_abs_co2']  = grouped[f'{col_abs_co2}_mean'] # The size of the impact

# Calculate Robust Cost
data['robust_cost_ton'] = data['mean_cost_ton'] + (SIGMA * data['std_cost_ton'])

# Filter for the "Problem Zone" (Small to Medium buildings)
# We limit X to see the 'near zero' effect clearly
plot_data = data[
    (data['mean_abs_co2'] > 0) & 
    (data['mean_abs_co2'] < 5.0) &  # Zoom in on small impact (<5 tons)
    (data['mean_cost_ton'] < 20000)
].copy()

# ==========================================
# PLOTTING
# ==========================================
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(9, 6))

# 1. Plot the "Shift" (Arrows from Mean -> Robust)
# This explicitly visualizes the "correction"
ax.quiver(
    plot_data['mean_abs_co2'],      # X
    plot_data['mean_cost_ton'],     # Y (Start)
    np.zeros(len(plot_data)),       # No change in X
    plot_data['robust_cost_ton'] - plot_data['mean_cost_ton'], # Change in Y (The Penalty)
    color='gray',
    alpha=0.3,
    angles='xy', scale_units='xy', scale=1,
    width=0.002,
    headwidth=3
)

# 2. Plot the "Robust" Endpoints (The final decision value)
scatter = ax.scatter(
    plot_data['mean_abs_co2'], 
    plot_data['robust_cost_ton'], 
    c=plot_data['std_cost_ton'], 
    cmap='magma_r', # Red = High Risk
    s=25, 
    label='Robust Value (~P85)',
    zorder=3
)

# 3. Highlight the "Trap" (Small Impact, High Uncertainty)
ax.axvspan(0, 1.0, color='red', alpha=0.1, label='Small Impact Zone (Unstable)')

# Annotations
ax.set_title(f'Method Failure at Small Impact: Carbon Scale vs. Uncertainty ({SCENARIO})', fontsize=13)
ax.set_xlabel('Absolute Carbon Savings (Tons CO2/5yr)', fontsize=12)
ax.set_ylabel('Capex Efficiency (£/Ton)', fontsize=12)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Standard deviation across epistemic runs', rotation=270, labelpad=15)

ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
plt.savefig('capex_scatter.png')
