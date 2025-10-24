#!/bin/bash
# Script to run energy retrofit analysis - Multi-Scenario Version
# Each scenario is submitted as a separate SLURM job
# Modify the paths and parameters as needed

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Detect environment and set paths accordingly
ON_HPC=true  # Set to true if running on HPC, false for local

if [ "$ON_HPC" = true ]; then
    # HPC paths
    INPUT_PATTERN="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/mega_all_scenarios/NE/*.csv"
    OUTPUT_DIR="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/retrofit_scenario/mega_all_scenarios/NE/retrofit_analysis"
    SCRIPT_PATH="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/energy_analysis_script.py"
    LOG_DIR="/rds/user/gb669/hpc-work/energy_map/RetrofitModel/post_analysis_logs_slurm"
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
# Define scenarios and their corresponding measure types as arrays
# Note: The order matters - each scenario is paired with its measure type
# Each scenario will be submitted as a separate job (when running on HPC)

# Option 1: Single scenario  
SCENARIOS=("wall_installation")
MEASURE_TYPES=("wall_installation")

# Option 2: Multiple scenarios (use arrays for pairing)
# SCENARIOS=("wall_installation" "loft_installation" "join_heat_ins_decay" "heat_pump_only")
# MEASURE_TYPES=("wall_installation" "loft_installation" "join_heat_ins_decay" "heat_pump_only")

 
# ==============================================================================
# ANALYSIS PARAMETERS
# ==============================================================================

YEARS=5
N_SIMULATIONS=5000

# Carbon factors (kg CO2/kWh)
GAS_CARBON_FACTOR=0.18      
ELEC_CARBON_FACTOR=0.19338  

# ==============================================================================
# SLURM CONFIGURATION (HPC only)
# ==============================================================================

SLURM_ACCOUNT="CULLEN-SL3-CPU"
SLURM_PARTITION="icelake"
SLURM_TIME="00:45:00"
SLURM_NODES=1
SLURM_NTASKS=1
SLURM_MEM="250G"

# ==============================================================================
# VALIDATE CONFIGURATION
# ==============================================================================

# Check that scenarios and measure types have the same length
if [ ${#SCENARIOS[@]} -ne ${#MEASURE_TYPES[@]} ]; then
    echo "ERROR: Number of scenarios (${#SCENARIOS[@]}) must match number of measure types (${#MEASURE_TYPES[@]})"
    exit 1
fi

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

echo "======================================================================"
echo "Energy Retrofit Analysis - Multi-Scenario (Separate Jobs)"
echo "======================================================================"
echo "Input: ${INPUT_PATTERN}"
echo "Output: ${OUTPUT_DIR}/${RUN_NAME}"
echo "Number of scenarios: ${#SCENARIOS[@]}"
echo "Scenarios: ${SCENARIOS[@]}"
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

if [ "$ON_HPC" = true ]; then
    # ==============================================================================
    # HPC MODE: Create and submit separate SLURM job for each scenario
    # ==============================================================================
    
    echo "Running on HPC - submitting ${#SCENARIOS[@]} separate jobs to SLURM..."
    echo ""
    
    # Create log directory if it doesn't exist
    mkdir -p "${LOG_DIR}"
    
    # Array to store submitted job IDs
    declare -a JOB_IDS
    
    # Loop through each scenario and submit a separate job
    for i in "${!SCENARIOS[@]}"; do
        SCENARIO="${SCENARIOS[$i]}"
        MEASURE_TYPE="${MEASURE_TYPES[$i]}"
        JOB_NAME="${RUN_NAME}_${SCENARIO}"
        
        echo "----------------------------------------------------------------------"
        echo "Preparing job for scenario: ${SCENARIO}"
        echo "  Measure type: ${MEASURE_TYPE}"
        echo "  Job name: ${JOB_NAME}"
        
        # Create SLURM submission script for this scenario
        SLURM_SCRIPT="${LOG_DIR}/submit_${JOB_NAME}.sh"
        
        cat > "${SLURM_SCRIPT}" << EOF
#!/bin/bash
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --mail-type=NONE
#SBATCH --mem=${SLURM_MEM}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --job-name=${JOB_NAME}

# Load required modules (modify as needed for your HPC environment)
# module load python/3.9  # Uncomment and modify as needed

echo "======================================================================"
echo "SLURM Job Information"
echo "======================================================================"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Job Name: \${SLURM_JOB_NAME}"
echo "Scenario: ${SCENARIO}"
echo "Measure Type: ${MEASURE_TYPE}"
echo "Node: \${SLURM_NODELIST}"
echo "Start Time: \$(date)"
echo "======================================================================"
echo ""

# Initialize conda
CONDA_BASE=/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh
source \$CONDA_BASE/etc/profile.d/conda.sh

# Activate the nebula environment
conda activate /home/gb669/.conda/envs/nebula

# Set environment variables
export SLURM_SUBMIT_DIR='/home/gb669/rds/hpc-work/energy_map/RetrofitModel'
cd \$SLURM_SUBMIT_DIR


# Run the Python analysis for this scenario only
python ${SCRIPT_PATH} \\
    --input-pattern "${INPUT_PATTERN}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --run-name "${RUN_NAME}" \\
    --scenarios ${SCENARIO} \\
    --measure-types ${MEASURE_TYPE} \\
    --years ${YEARS} \\
    --n-simulations ${N_SIMULATIONS} \\
    --gas-carbon-factor ${GAS_CARBON_FACTOR} \\
    --elec-carbon-factor ${ELEC_CARBON_FACTOR}

# Check if the script completed successfully
if [ \$? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Analysis completed successfully for scenario: ${SCENARIO}"
    echo "Results saved to: ${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}"
    echo "======================================================================"
    
    # List output files
    if [ -d "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}" ]; then
        echo ""
        echo "Files generated:"
        file_count=\$(ls -1 "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}" | wc -l)
        echo "  Total files: \${file_count}"
        
        # List key files
        echo ""
        echo "  Key outputs:"
        for file in processed_data.csv summary_report.txt portfolio_summary.csv; do
            if [ -f "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}/\${file}" ]; then
                size=\$(ls -lh "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}/\${file}" | awk '{print \$5}')
                echo "    - \${file} (\${size})"
            fi
        done
        
        # Count plots
        plot_count=\$(ls -1 "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}"/*.png 2>/dev/null | wc -l)
        echo "    - \${plot_count} plots (.png files)"
    fi
    
    echo ""
    echo "======================================================================"
    echo "End Time: \$(date)"
    echo "======================================================================"
    
else
    echo ""
    echo "======================================================================"
    echo "Analysis failed for scenario: ${SCENARIO}"
    echo "Check error messages above."
    echo "======================================================================"
    exit 1
fi
EOF

        # Make the SLURM script executable
        chmod +x "${SLURM_SCRIPT}"
        
        # Submit the job
        echo "  Submitting to SLURM..."
        JOB_OUTPUT=$(sbatch "${SLURM_SCRIPT}" 2>&1)
        
        if [ $? -eq 0 ]; then
            # Extract job ID from output (format: "Submitted batch job 12345")
            JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
            JOB_IDS+=("${JOB_ID}")
            echo "  ✓ Job submitted successfully! Job ID: ${JOB_ID}"
            echo "  SLURM script: ${SLURM_SCRIPT}"
        else
            echo "  ✗ ERROR: Failed to submit job"
            echo "  Error output: ${JOB_OUTPUT}"
        fi
        
        echo ""
    done
    
    # Summary
    echo "======================================================================"
    echo "All Jobs Submitted!"
    echo "======================================================================"
    echo "Total jobs submitted: ${#JOB_IDS[@]}"
    echo "Job IDs: ${JOB_IDS[@]}"
    echo ""
    echo "Monitor jobs:"
    echo "  All jobs:      squeue -u \$USER"
    echo "  This run:      squeue -u \$USER --name=${RUN_NAME}_*"
    echo "  Specific job:  squeue -j <job_id>"
    echo ""
    echo "View logs:"
    echo "  Log directory: ${LOG_DIR}/"
    echo "  Pattern:       ${LOG_DIR}/${RUN_NAME}_*"
    echo ""
    echo "Cancel jobs:"
    echo "  All jobs:      scancel -u \$USER --name=${RUN_NAME}_*"
    echo "  Specific job:  scancel <job_id>"
    echo "  All listed:    scancel ${JOB_IDS[@]}"
    echo "======================================================================"

else
    # ==============================================================================
    # LOCAL MODE: Run each scenario sequentially
    # ==============================================================================
    
    echo "Running locally (scenarios will run sequentially)..."
    echo ""
    
    # Loop through each scenario
    for i in "${!SCENARIOS[@]}"; do
        SCENARIO="${SCENARIOS[$i]}"
        MEASURE_TYPE="${MEASURE_TYPES[$i]}"
        
        echo "======================================================================"
        echo "Running scenario $((i+1))/${#SCENARIOS[@]}: ${SCENARIO}"
        echo "======================================================================"
        echo "Measure type: ${MEASURE_TYPE}"
        echo ""
        
        # Run the analysis for this scenario
        python ${SCRIPT_PATH} \
            --input-pattern "${INPUT_PATTERN}" \
            --output-dir "${OUTPUT_DIR}" \
            --run-name "${RUN_NAME}" \
            --scenarios ${SCENARIO} \
            --measure-types ${MEASURE_TYPE} \
            --years ${YEARS} \
            --n-simulations ${N_SIMULATIONS} \
            --gas-carbon-factor ${GAS_CARBON_FACTOR} \
            --elec-carbon-factor ${ELEC_CARBON_FACTOR}

        # Check if the script completed successfully
        if [ $? -eq 0 ]; then
            echo ""
            echo "----------------------------------------------------------------------"
            echo "✓ Scenario '${SCENARIO}' completed successfully!"
            echo "----------------------------------------------------------------------"
            
            # List output files
            if [ -d "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}" ]; then
                file_count=$(ls -1 "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}" | wc -l)
                echo "  Files generated: ${file_count}"
                
                # List key files
                echo "  Key outputs:"
                for file in processed_data.csv summary_report.txt portfolio_summary.csv; do
                    if [ -f "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}/${file}" ]; then
                        size=$(ls -lh "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}/${file}" | awk '{print $5}')
                        echo "    - ${file} (${size})"
                    fi
                done
                
                # Count plots
                plot_count=$(ls -1 "${OUTPUT_DIR}/${RUN_NAME}/${SCENARIO}"/*.png 2>/dev/null | wc -l)
                echo "    - ${plot_count} plots (.png files)"
            fi
            echo ""
            
        else
            echo ""
            echo "======================================================================"
            echo "✗ ERROR: Scenario '${SCENARIO}' failed!"
            echo "======================================================================"
            echo "Stopping execution. Check error messages above."
            echo "======================================================================"
            exit 1
        fi
    done
    
    # Final summary
    echo ""
    echo "======================================================================"
    echo "All Scenarios Completed Successfully!"
    echo "======================================================================"
    echo "Total scenarios processed: ${#SCENARIOS[@]}"
    echo "Results saved to: ${OUTPUT_DIR}/${RUN_NAME}"
    echo ""
    echo "Scenario directories:"
    for scenario in "${SCENARIOS[@]}"; do
        echo "  - ${OUTPUT_DIR}/${RUN_NAME}/${scenario}"
    done
    echo "======================================================================"
fi