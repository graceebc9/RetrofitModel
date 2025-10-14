"""
Module: main.py
 
Copyright (c) 2024 Grace Colverd
This work is licensed under CC BY-NC-SA 4.0
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

For commercial licensing options, contact: gb669@cam.ac.uk. 
"""

from datetime import datetime
import os
from src.logging_config import setup_logging, get_logger
from src.utils import is_running_on_hpc 

# Use it in your script
running_locally = not is_running_on_hpc()
if running_locally:
    base_path = '/Users/gracecolverd/RetrofitModel/notebook'
else:
    base_path = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel'
    os.makedirs(f'{base_path}/logs', exist_ok=True )

log_config_filepath = f"{base_path}/logs/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_path = f"{base_path}/logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

setup_logging(log_level='INFO', log_path=log_path)
logger = get_logger(__name__)


from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent

import os
import sys
import logging
import pandas as pd

from src.RetrofitModel import RetrofitModel
from src.RetrofitConfig import RetrofitConfig
from src.split_onsud_file import split_onsud_and_postcodes
from src.postcode_utils import load_ids_from_file, load_onsud_data
from src.retrofit_proc import run_retrofit_calc_with_uncertainty
from src.logging_config import get_logger, setup_logging
from src.conservation import load_conservation_shapefile 

# ========================================
# DATA PATHS - UPDATE AS NEEDED
# ========================================

if running_locally:

    PC_SHP_PATH = '/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998/codepoint-poly_5267291'

    # Location of building stock dataset (we use Verisk buildings from Edina)
    BUILDING_PATH = '/Volumes/T9/2024_Data_downloads/Versik_building_data/2024_03_22_updated_data/UKBuildings_Edition_15_new_format_upn.gpkg'

    # Location of the input data folder
    # If in this repo: location_input_data_folder = 'input_data_sources'
    # If stored elsewhere (e.g. external hard drive):
    location_input_data_folder = '/Volumes/T9/2024_Data_downloads/2024_11_nebula_paper_data/'
    onsud_path_base = os.path.join(location_input_data_folder, 'ONS_UPRN_database/ONSUD_DEC_2022/Data')
    GAS_PATH = os.path.join(location_input_data_folder, 'energy_data/Postcode_level_gas_2022.csv')
    ELEC_PATH = os.path.join(location_input_data_folder, 'energy_data/Postcode_level_all_meters_electricity_2022.csv')
    TEMP_1KM_PATH = os.path.join(location_input_data_folder, 'climate_data/tas_hadukgrid_uk_1km_mon_202201-202212.nc')
else: 
    PC_SHP_PATH = '/rds/user/gb669/hpc-work/energy_map/data/postcode_polygons/codepoint-poly_5267291'
    BUILDING_PATH = '/rds/user/gb669/hpc-work/energy_map/data/building_files/UKBuildings_Edition_15_new_format_upn.gpkg'
    location_input_data_folder = '/home/gb669/rds/hpc-work/energy_map/data/input_data'
    onsud_path_base = '/home/gb669/rds/hpc-work/energy_map/data/onsud_files/Data'

    GAS_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_gas_2022.csv'
    ELEC_PATH='/home/gb669/rds/hpc-work/energy_map/data/input_data_sources/energy_data/Postcode_level_all_meters_electricity_2022.csv'





# Output directory - do not update if you want to save in the repo
OUTPUT_DIR = 'final_dataset'


# ========================================
# CONFIGURATION
# ========================================

batch_size = 500
log_size = 100
n_monte_carlo = 100
scenarios = ['wall_installation', 'loft_installation']
job_name='testing'
region_list = ['NE'] if running_locally else [os.getenv('REGION_LIST')]

STAGE0_split_onsud = False

retrofit_config = RetrofitConfig(
    energy_cost_per_kwh=0.07,
    existing_intervention_probs={
        'loft_insulation': 0,
        'floor_insulation': 0,
        'window_upgrades': 0,
        'roof_scaling_factor': 0.8,
        'external_wall_occurence': 0.5,
    }
)

retrofig_model = RetrofitModel(
    retrofit_config,
    n_samples=n_monte_carlo
)

from src.utils import log_configuration

 
log_configuration(
    log_config_filepath,
    batch_size=batch_size,
    log_size=log_size,
    n_monte_carlo=n_monte_carlo,
    scenarios=scenarios,
    running_locally=running_locally,
    region_list=region_list,
    STAGE0_split_onsud=STAGE0_split_onsud,
    retrofit_config=retrofit_config,
    retrofig_model=retrofig_model,
    job_name=job_name,
    # Environment variables
    slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID'),
    region_env=os.getenv('REGION_LIST'),
) 

# ========================================
# HELPER FUNCTIONS
# ========================================

 

def gen_batch_ids(batch_ids: list, log_file: str, logger: logging.Logger) -> list:
    """Generate batch IDs, excluding already processed ones."""
    if not os.path.exists(log_file):
        logger.info('No existing log file found, processing all IDs')
        return batch_ids

    logger.info('Found existing log file, removing already processed IDs')
    logger.info(f'Log file: {log_file}')
    logger.debug(f'Original batch size: {len(batch_ids)}')

    log = pd.read_csv(log_file)
    proc_id = log.postcode.unique().tolist()
    batch_ids = [x for x in batch_ids if x not in proc_id]

    logger.info(f'Reduced batch size after removing processed IDs: {len(batch_ids)}')
    return batch_ids


def postcode_main(batch_path, data_dir, path_to_onsud_file, path_to_pcshp, INPUT_GPK,
                  retrofit_config, retrofig_model, scenarios, neb_pcs, 
                  conservation_data, 
                  region_label, batch_label, attr_lab, log_size=100, ):
    """Main processing function."""

    # Setup logging
    proc_dir = os.path.join(data_dir, attr_lab, region_label)
    os.makedirs(proc_dir, exist_ok=True)

    logger.info(f'Starting processing for region: {region_label}')
    logger.debug(f'Processing batch: {batch_path}')
    logger.info(f'Saving to proc_dir {proc_dir}')

    # Setup log file
    log_file = os.path.join(proc_dir, f'{batch_label}_log_file.csv')
    logger.debug(f'Using log file: {log_file}')


    # Load ONSUD data
    logger.debug('Loading ONSUD data')
    onsud_data = load_onsud_data(path_to_onsud_file, path_to_pcshp)
    logger.debug('ONSUD data loaded successfully')

    # Load and filter postcodes
    postcodes = load_ids_from_file(batch_path)
    logger.debug(f'Loaded batch IDs (postcodes): {len(postcodes)} items')
    postcodes = gen_batch_ids(postcodes, log_file, logger)
    postcodes = [x for x in postcodes if x in neb_pcs]
    logger.info(f'New length of postcodes to process: {len(postcodes)} ')
    # Log processing parameters
    logger.debug('Processing parameters:')
    parameters = {
        'Batch size': len(postcodes),
        'Input GPK': INPUT_GPK,
        'SubBatch log limit': log_size,
        'Batch label': batch_label,
        'RetrofitConfig_energy_cost_per_kwh': retrofit_config.energy_cost_per_kwh,
        'RetrofitConfig_existing_intervention_probs': retrofit_config.existing_intervention_probs,
        'RetrofitModelSettings': RetrofitModel.n_samples,
        'Scenarios': scenarios,
        'Output dir': data_dir,
    }
    for param, value in parameters.items():
        logger.debug(f'{param}: {value}')

    run_retrofit_calc_with_uncertainty(
        pcs_list=postcodes,
        data=onsud_data,
        INPUT_GPK=INPUT_GPK,
        batch_size=log_size,
        batch_label=batch_label,
        log_file=log_file,
        retrofit_config=retrofit_config,
        retrofig_model=retrofig_model,
        scenarios=scenarios,
        conservation_data=conservation_data,
    )
    logger.info('Batch processing completed successfully')


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    log_file = os.path.join(OUTPUT_DIR, 'processing.log')
    logger.info("Starting data processing pipeline")
    logger.debug(f"Using output directory: {OUTPUT_DIR}")

    # Validate input paths
    required_paths = {
        'ONSUD base path': onsud_path_base,
        'Postcode shapefile path': PC_SHP_PATH,
        'Building data path': BUILDING_PATH,
        'Gas data path': GAS_PATH,
        'Electricity data path': ELEC_PATH
    }

    # load conservation areas 
    conservation_data = load_conservation_shapefile(path =f'{root_dir}/RetrofitModel/src/global_avs/Conservation_Areas_-5503574965118299320')

    for name, path in required_paths.items():
        if not os.path.exists(path):
            logger.error(f"{name} not found at: {path}")
            raise FileNotFoundError(f"{name} not found at: {path}")
        logger.debug(f"Verified {name} at: {path}")

    # Split ONSUD data if required
    if STAGE0_split_onsud:
        logger.info("Starting ONSUD splitting process")
        for region in region_list:
            logger.info(f"Processing region: {region}")
            onsud_path = os.path.join(onsud_path_base, f'ONSUD_DEC_2022_{region}.csv')
            split_onsud_and_postcodes(onsud_path, PC_SHP_PATH, batch_size)
            logger.info(f"Successfully split ONSUD data for region {region}")
    else:
        logger.info("ONSUD splitting disabled, proceeding to postcode calculations")

    # Load NEBULA postcodes
    neb_pcs = pd.read_csv(f'{root_dir}/RetrofitModel/src/global_avs/neb_dom_pcs.csv')
    neb_pcs = neb_pcs['postcode'].tolist()

    # Load batch paths
    all_batch_paths = list(set(load_ids_from_file('batch_paths.txt')))

    # Determine which batch(es) to process
    if not running_locally:
        # HPC mode: expect batch path as command line argument
        if len(sys.argv) < 2:
            logger.error("HPC mode requires batch path as command line argument")
            logger.error("Usage: python main.py <batch_path>")
            raise ValueError("Missing batch path argument for HPC mode")
        
        batch_path_arg = sys.argv[1]
        logger.info(f"HPC mode: Received batch path argument: {batch_path_arg}")
        
        # Validate the batch path exists in our list
        if batch_path_arg not in all_batch_paths:
            logger.warning(f"Batch path not in batch_paths.txt, but proceeding: {batch_path_arg}")
        
        batch_paths = [batch_path_arg]
        logger.info(f"HPC mode: Processing single batch: {batch_paths[0]}")
    else:
        # Local mode: process all batches
        batch_paths = all_batch_paths
        logger.info(f"Local mode: Processing all {len(batch_paths)} batches")

    print(f"Batch paths to process: {batch_paths}")

    # Process each batch
    for i, batch_path in enumerate(batch_paths, 1):
        logger.info(f"Processing batch {i}/{len(batch_paths)}: {batch_path}")

        label = batch_path.split('/')[-2]
        logger.info(f'Batch path label {label}' )

        batch_id = batch_path.split('/')[-1].split('.')[0].split('_')[-1]
        logger.info(f'Batch patch id : {batch_id}')
        onsud_path = os.path.join(os.path.dirname(batch_path), f'onsud_{batch_id}.csv')
        logger.info(f'Processing onsud path {onsud_path}')
        
        postcode_main(
            batch_path=batch_path,
            data_dir='intermediate_data',
            path_to_onsud_file=onsud_path,
            path_to_pcshp=PC_SHP_PATH,
            INPUT_GPK=BUILDING_PATH,
            retrofit_config=retrofit_config,
            retrofig_model=retrofig_model,
            scenarios=scenarios,
            region_label=label,
            neb_pcs=neb_pcs,
            batch_label=batch_id,
            attr_lab='retrofit_scenario',
            log_size=log_size,
            conservation_data=conservation_data,
        )
        logger.info(f"Successfully processed batch: {batch_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error in main program: {str(e)}", exc_info=True)
        raise