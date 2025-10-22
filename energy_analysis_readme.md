# Energy Retrofit Analysis Script

## Description

This script converts the original Jupyter notebook into an executable Python script that performs comprehensive energy retrofit analysis including:
- Energy and carbon savings calculations
- Uncertainty analysis (epistemic vs aleatoric)
- Portfolio-level metrics
- Cost-effectiveness analysis
- Multiple visualizations

All outputs (plots, CSVs, reports) are automatically saved to a specified directory.

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn --break-system-packages
```

Make the script executable:

```bash
chmod +x energy_analysis_script.py
```

## Usage

### Basic Usage

```bash
python energy_analysis_script.py \
    --input-pattern "/path/to/input/data/*.csv" \
    --output-dir "/path/to/output" \
    --run-name "my_analysis_run"
```

### Full Example with All Parameters

```bash
python energy_analysis_script.py \
    --input-pattern "/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/testing/NE/*.csv" \
    --output-dir "/rds/user/gb669/hpc-work/outputs/retrofit_analysis" \
    --run-name "NE_heat_pump_analysis_2025" \
    --scenario "join_heat_ins_decay" \
    --measure-type "joint_heat_ins_decay" \
    --years 5 \
    --n-simulations 5000 \
    --gas-carbon-factor 0.18 \
    --elec-carbon-factor 0.19338 \
    --gas-price 0.07
```

## Parameters

### Required Parameters

- `--input-pattern`: Glob pattern for input CSV files (e.g., `"/path/to/*.csv"`)
- `--output-dir`: Base output directory for results

### Optional Parameters

- `--run-name`: Name for this analysis run (default: timestamp in format YYYYMMDD_HHMMSS)
- `--scenario`: Scenario name (default: "join_heat_ins_decay")
- `--measure-type`: Measure type (default: "joint_heat_ins_decay")
- `--years`: Number of years for projections (default: 5)
- `--n-simulations`: Number of Monte Carlo simulations (default: 5000)
- `--gas-carbon-factor`: Gas carbon factor in kg CO2/kWh (default: 0.18)
- `--elec-carbon-factor`: Electricity carbon factor in kg CO2/kWh (default: 0.19338)
- `--gas-price`: Gas price in £/kWh (default: 0.07)

## Output Files

The script creates a subdirectory with your run name (or timestamp) containing:

### Data Files
- `processed_data.csv` - Complete processed dataset with all calculated metrics
- `building_uncertainty_metrics.csv` - Building-level uncertainty metrics
- `portfolio_summary.csv` - Portfolio-level summary statistics
- `descriptive_statistics.csv` - Descriptive statistics of key variables
- `config.txt` - Configuration parameters used for the run
- `portfolio_metrics.txt` - Detailed portfolio metrics
- `summary_report.txt` - Comprehensive summary report

### Visualizations
- `uncertainty_distribution.png` - Distribution of epistemic vs aleatoric uncertainty
- `epistemic_vs_aleatoric_scatter.png` - Scatter plot comparing uncertainty types
- `portfolio_distribution.png` - Portfolio-level epistemic uncertainty distribution
- `installation_costs_by_decile.png` - Installation costs by energy decile
- `co2_savings_by_decile.png` - CO2 savings by energy decile
- `gas_percentile_analysis.png` - Cost analysis by gas percentile
- `cost_per_gas_ton_by_decile.png` - Cost per gas ton saved by decile
- `cost_per_net_ton_by_decile.png` - Cost per net ton saved by decile

## Example Output Structure

```
/path/to/output/
└── my_run_name/
    ├── config.txt
    ├── summary_report.txt
    ├── processed_data.csv
    ├── building_uncertainty_metrics.csv
    ├── portfolio_summary.csv
    ├── portfolio_metrics.txt
    ├── descriptive_statistics.csv
    ├── uncertainty_distribution.png
    ├── epistemic_vs_aleatoric_scatter.png
    ├── portfolio_distribution.png
    ├── installation_costs_by_decile.png
    ├── co2_savings_by_decile.png
    ├── gas_percentile_analysis.png
    ├── cost_per_gas_ton_by_decile.png
    └── cost_per_net_ton_by_decile.png
```

## Running Multiple Scenarios

You can run multiple scenarios by changing the input pattern and run name:

```bash
# Scenario 1: Northeast region
python energy_analysis_script.py \
    --input-pattern "/data/NE/*.csv" \
    --output-dir "/outputs" \
    --run-name "NE_scenario_5yr"

# Scenario 2: Southeast region
python energy_analysis_script.py \
    --input-pattern "/data/SE/*.csv" \
    --output-dir "/outputs" \
    --run-name "SE_scenario_5yr"
```

## Troubleshooting

### No files found error
Ensure your input pattern is quoted and uses the correct path:
```bash
--input-pattern "/absolute/path/to/data/*.csv"
```

### Missing dependencies
Install required packages:
```bash
pip install pandas numpy matplotlib seaborn --break-system-packages
```

### Import errors for RetrofitModel
Ensure the path to RetrofitModel is correct in the script (line 19):
```python
sys.path.append('/rds/user/gb669/hpc-work/energy_map/RetrofitModel')
```

## Notes

- The script automatically filters out "Domestic outbuilding" premise types
- All plots are saved at 300 DPI for publication quality
- Progress messages are printed to console during execution
- The script validates the data processing using `validate_single_scenario_new`