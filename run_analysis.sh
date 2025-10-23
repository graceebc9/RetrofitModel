#!/bin/bash
# Script to run energy retrofit analysis - Multi-Scenario Version
# Modify the paths and parameters as needed

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Detect environment and set paths accordingly
ON_HPC=false  # Set to true if running on HPC, false for local

if [ "$ON_HPC" = true ]; then
    # HPC paths
    INPUT_PATTERN="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/testing/NE/*.csv"
    OUTPUT_DIR="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/testing/retrofit_analysis"
    SCRIPT_PATH="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/energy_retrofit_analysis_script.py"
else
    # Local paths
    INPUT_PATTERN='/Users/gracecolverd/RetrofitModel/intermediate_data_2D/retrofit_scenario/all_scenarios/NE/*.csv'
    OUTPUT_DIR='/Volumes/T9/2024_Data_downloads/2025_10_RetrofitModel/1_data_runs/m2d/NE_all_scenarios'
    SCRIPT_PATH='energy_analysis_script.py'
fi

# Run name (optional - will use timestamp if not provided)
RUN_NAME="NE_all_scenarios_5yr_$(date +%Y%m%d)"

# ==============================================================================
# SCENARIO CONFIGURATION
# ==============================================================================
# Define scenarios and their corresponding measure types
# Note: The order matters - each scenario is paired with its measure type

# Option 1: Single scenario  
# SCENARIOS="join_heat_ins_decay"
# MEASURE_TYPES="joint_heat_ins_decay"

# Option 2: Multiple scenarios 
SCENARIOS="wall_installation loft_installation join_heat_ins_decay heat_pump_only"
MEASURE_TYPES="wall_installation loft_installation join_heat_ins_decay heat_pump_only"

 
# ==============================================================================
# ANALYSIS PARAMETERS
# ==============================================================================

YEARS=5
N_SIMULATIONS=5000

# Carbon factors (kg CO2/kWh)
GAS_CARBON_FACTOR=0.18      
ELEC_CARBON_FACTOR=0.19338  

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

echo "======================================================================"
echo "Energy Retrofit Analysis - Multi-Scenario"
echo "======================================================================"
echo "Input: ${INPUT_PATTERN}"
echo "Output: ${OUTPUT_DIR}/${RUN_NAME}"
echo "Scenarios: ${SCENARIOS}"
echo "Measure Types: ${MEASURE_TYPES}"
echo "Years: ${YEARS}"
echo "======================================================================"
echo ""

# Check if input files exist
file_count=$(ls -1 ${INPUT_PATTERN} 2>/dev/null | wc -l)
if [ $file_count -eq 0 ]; then
    echo "ERROR: No input files found matching pattern: ${INPUT_PATTERN}"
    exit 1
fi
echo "Found ${file_count} input files"
echo ""

# Run the analysis
python ${SCRIPT_PATH} \
    --input-pattern "${INPUT_PATTERN}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --scenarios ${SCENARIOS} \
    --measure-types ${MEASURE_TYPES} \
    --years ${YEARS} \
    --n-simulations ${N_SIMULATIONS} \
    --gas-carbon-factor ${GAS_CARBON_FACTOR} \
    --elec-carbon-factor ${ELEC_CARBON_FACTOR}

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Analysis completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}/${RUN_NAME}"
    echo "======================================================================"
    
    # List scenario directories and their contents
    echo ""
    echo "Scenario directories created:"
    for scenario in ${SCENARIOS}; do
        if [ -d "${OUTPUT_DIR}/${RUN_NAME}/${scenario}" ]; then
            echo ""
            echo "--- ${scenario} ---"
            file_count=$(ls -1 "${OUTPUT_DIR}/${RUN_NAME}/${scenario}" | wc -l)
            echo "  Files generated: ${file_count}"
            
            # List key files
            echo "  Key outputs:"
            for file in processed_data.csv summary_report.txt portfolio_summary.csv; do
                if [ -f "${OUTPUT_DIR}/${RUN_NAME}/${scenario}/${file}" ]; then
                    size=$(ls -lh "${OUTPUT_DIR}/${RUN_NAME}/${scenario}/${file}" | awk '{print $5}')
                    echo "    - ${file} (${size})"
                fi
            done
            
            # Count plots
            plot_count=$(ls -1 "${OUTPUT_DIR}/${RUN_NAME}/${scenario}"/*.png 2>/dev/null | wc -l)
            echo "    - ${plot_count} plots (.png files)"
        fi
    done
    
    echo ""
    echo "======================================================================"
    
else
    echo ""
    echo "======================================================================"
    echo "Analysis failed! Check error messages above."
    echo "======================================================================"
    exit 1
fi