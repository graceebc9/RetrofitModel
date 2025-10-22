#!/bin/bash
# Sample script to run energy retrofit analysis
# Modify the paths and parameters as needed

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input data path (use glob pattern)
# if on_hpc_bool:
# INPUT_PATTERN="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/testing/NE/*.csv"
# OUTPUT_DIR="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/testing/retrofit_analysis"
# else: 
INPUT_PATTERN='/Volumes/T9/2024_Data_downloads/2025_10_RetrofitModel/1_data_runs/m2d/NE_min/*.csv'
OUTPUT_DIR='/Volumes/T9/2024_Data_downloads/2025_10_RetrofitModel/1_data_runs/m2d/NE1_retrofit-analysis.csv'



# Run name (optional - will use timestamp if not provided)
RUN_NAME="NE_heat_pump_5yr_$(date +%Y%m%d)"

# Analysis parameters
SCENARIO="join_heat_ins_decay"
MEASURE_TYPE="joint_heat_ins_decay"
YEARS=5
N_SIMULATIONS=5000

# Carbon factors and prices
GAS_CARBON_FACTOR=0.18      # kg CO2/kWh
ELEC_CARBON_FACTOR=0.19338  # kg CO2/kWh
GAS_PRICE=0.07              # £/kWh

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

echo "======================================================================"
echo "Energy Retrofit Analysis"
echo "======================================================================"
echo "Input: ${INPUT_PATTERN}"
echo "Output: ${OUTPUT_DIR}/${RUN_NAME}"
echo "======================================================================"
echo ""

python energy_analysis_script.py \
    --input-pattern "${INPUT_PATTERN}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --scenario "${SCENARIO}" \
    --measure-type "${MEASURE_TYPE}" \
    --years ${YEARS} \
    --n-simulations ${N_SIMULATIONS} \
    --gas-carbon-factor ${GAS_CARBON_FACTOR} \
    --elec-carbon-factor ${ELEC_CARBON_FACTOR} \
    --gas-price ${GAS_PRICE}

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Analysis completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}/${RUN_NAME}"
    echo "======================================================================"
    
    # Optional: List output files
    echo ""
    echo "Generated files:"
    ls -lh "${OUTPUT_DIR}/${RUN_NAME}/"
else
    echo ""
    echo "======================================================================"
    echo "Analysis failed! Check error messages above."
    echo "======================================================================"
    exit 1
fi