"""
Module: main.py
Description: Run the data processing pipeline to generate NEBULA dataset. This works in two stages:
1- Batch the ONSUD files / postcodes. 
    We split the Regional files into batches of 10k postcodes, and find the associated UPRNs associated with them. 
    Batches stored in dataset/batches/
2- Run the fuel calculations for each batch of postcodes. This involves finding all the buildings associated, calculating the per building metrics, and pulling in the gas and electricity data. 
3 - unify the results from the log fiels 
    To protect against timeout we log results in dataset/proc_dir/fuel
    this stage extracts all results, stores a processing log and processes the final dataset

Key features:
 - you can run logging with DEBUG to see more detailed logs 
 - batches were to enable multi processing on a HPC 

Outputs:
final_dataset/Unfiltered_processed_data.csv: whole dataset with no filters, includes mixed use and domestic postcodes 
final_dataset/NEBULA_data_filtered.csv: filtered to wholly residential and applies thresholds / filters (UPRN to gas match and thresholds for gas and elec EUI etc.)
{fuel/type/age}_log_file.csv: details the count of postcodes for each region/batch combo for themes. If runnning for subset of dataset can check here to see counts align with batch size. If counts are missing, re-run stage 1


Copyright (c) 2024 Grace Colverd
This work is licensed under CC BY-NC-SA 4.0
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

For commercial licensing options, contact: gb669@cam.ac.uk. 
"""

#########################################   Data paths SOME TO BE UPDATED   ########################################################################### 
import os 
import sys
import pandas as pd 

from src.RetrofitModel import RetrofitModel
from src.RetrofitConfig import RetrofitConfig 

# Location of postcode shapefiles (we use codepoint edina)
PC_SHP_PATH = '/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998/codepoint-poly_5267291' 
# Location of building stock dataset (we use Verisk buildings from Edina)
BUILDING_PATH = '/Volumes/T9/2024_Data_downloads/Versik_building_data/2024_03_22_updated_data/UKBuildings_Edition_15_new_format_upn.gpkg'

# Location of the input data folder (update if not within this repo)
# If in this repo
location_input_data_folder = 'input_data_sources'
# if stored elsewhere e.g. external hard drive
location_input_data_folder = '/Volumes/T9/2024_Data_downloads/2024_11_nebula_paper_data/'

# Do not need to update if you download our zip file, unzip and place in location_input_data_folder
onsud_path_base = os.path.join(location_input_data_folder, 'ONS_UPRN_database/ONSUD_DEC_2022/Data')
GAS_PATH = os.path.join(location_input_data_folder, 'energy_data/Postcode_level_gas_2022.csv')
ELEC_PATH = os.path.join(location_input_data_folder, 'energy_data/Postcode_level_all_meters_electricity_2022.csv')
TEMP_1KM_PATH = os.path.join(location_input_data_folder, 'climate_data/tas_hadukgrid_uk_1km_mon_202201-202212.nc')
# Output directory, do not update if you want to save in the repo
OUTPUT_DIR = 'final_dataset'
# OUTPUT_DIR = 'tests'

n_monte_carlo=100


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
    retrofit_config , 
    n_samples = 1000
) 
scenarios = ['wall_installation', 'loft_installation']

#########################################   Regions to run, YOU CAN UPDATE   ###################################################################### 
batch_size = 500
log_size = 500
running_locally = os.getenv('SLURM_ARRAY_TASK_ID') is None
region_list = ['NE'] if running_locally else [os.getenv('REGION_LIST')]

STAGE0_split_onsud =False 

#########################################    Script      ###################################################################################### 

from src.split_onsud_file import split_onsud_and_postcodes
from src.postcode_utils import load_ids_from_file
from src.retrofit_proc import run_retrofit_calc_with_uncertainty  
# from src.post_process import  apply_filters, unify_dataset, final_rename
import os
import logging 

from src.logging_config import get_logger, setup_logging


setup_logging()  
logger = get_logger(__name__)


def gen_batch_ids(batch_ids: list, log_file: str, logger: logging.Logger) -> list:
    """Generate batch IDs, excluding already processed ones."""
    if os.path.exists(log_file):

        logger.info('Found existing log file, removing already processed IDs')
        logger.info(f'Log file: {log_file}')
        logger.debug(f'Original batch size: {len(batch_ids)}')
        
        log = pd.read_csv(log_file)
        proc_id = log.postcode.unique().tolist()
        batch_ids = [x for x in batch_ids if x not in proc_id]
        
        logger.info(f'Reduced batch size after removing processed IDs: {len(batch_ids)}')
        return batch_ids
    else:
        logger.info('No existing log file found, processing all IDs')
        return batch_ids

from src.postcode_utils import load_onsud_data, load_ids_from_file

# def run_retrofit_process(batch_ids, onsud_data, INPUT_GPK, subbatch_size, batch_label, log_file ):
#     """Process type data."""
#     print(log_file)
    
#     run_retrofit_calc_with_uncertainty(batch_ids, onsud_data, INPUT_GPK, subbatch_size,
#                   batch_label, log_file,   )
    

def postcode_main(batch_path, data_dir, path_to_onsud_file, path_to_pcshp, INPUT_GPK, 
                  retrofit_config,
                  retrofig_model,
                  scenarios,
                neb_pcs , 
         region_label, batch_label, attr_lab, n_monte_carlo=100,  log_size=100):
    """Main processing function."""
    
    # Setup logging
    proc_dir = os.path.join(data_dir, attr_lab, region_label)
    os.makedirs(proc_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    
    logger.info(f'Starting processing for region: {region_label}')
    logger.debug(f'Processing batch: {batch_path}')
    
    # Setup log file
    log_file = os.path.join(proc_dir, f'{batch_label}_log_file.csv')
    logger.debug(f'Using log file: {log_file}')
    
    # Load ONSUD data
    logger.debug('Loading ONSUD data')
    onsud_data = load_onsud_data(path_to_onsud_file, path_to_pcshp)
    logger.debug('ONSUD data loaded successfully')

    
    # Load and filter batch IDs
    batch_ids = load_ids_from_file(batch_path)
    print('batch ids which are posttcodesd ')
    print(batch_ids)
    batch_ids = gen_batch_ids(batch_ids, log_file, logger)

    batch_ids = [x for x in batch_ids if x in neb_pcs ]
    
    # Log processing parameters
    logger.debug('Processing parameters:')
    parameters = {
        'Batch size': len(batch_ids),
        'Input GPK': INPUT_GPK,
        'SubBatch log limit': log_size,
        'Batch label': batch_label,
        'retofitConfig_energy_cost_per_kwh': retrofit_config.energy_cost_per_kwh,
        'retofitConfig_existing_intervention_probs': retrofit_config.existing_intervention_probs,
        'RetrofitModelSettings': RetrofitModel.n_samples , 
        'Scenarios': scenarios,


        'Output dir:' : data_dir, 
    }
    for param, value in parameters.items():
        logger.debug(f'{param}: {value}')
    
    run_retrofit_calc_with_uncertainty(
        pcs_list=batch_ids,
        data=onsud_data,
        INPUT_GPK=INPUT_GPK,
        batch_size=log_size,
        batch_label=batch_label,
        log_file=log_file,
         n_monte_carlo=n_monte_carlo,
                    retrofit_config=retrofit_config, 
                        retrofig_model=retrofig_model,
                        scenarios=scenarios,
    )
    logger.info('Batch processing completed successfully')




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

    neb_pcs = pd.read_csv('/Users/gracecolverd/NebulaDataset/notebooks2/neb_dom_pcs.csv')
    
    neb_pcs=neb_pcs['postcode'].tolist() 

    batch_paths = list(set(load_ids_from_file('batch_paths.txt')))
    print(batch_paths)
    logger.info(f"Found {len(batch_paths)} unique batch paths to process")

    for i, batch_path in enumerate(batch_paths, 1):
        logger.info(f"Processing batch {i}/{len(batch_paths)}: {batch_path}")
        label = batch_path.split('/')[-2]
        batch_id = batch_path.split('/')[-1].split('.')[0].split('_')[-1]
        onsud_path = os.path.join(os.path.dirname(batch_path), f'onsud_{batch_id}.csv') 

        postcode_main(batch_path = batch_path, 
                        data_dir = 'intermediate_data',
                        path_to_onsud_file = onsud_path, 
                        path_to_pcshp = PC_SHP_PATH,
                        INPUT_GPK=BUILDING_PATH,
                        retrofit_config=retrofit_config, 
                        retrofig_model=retrofig_model,
                        scenarios=scenarios,
                        region_label=label, 
                        neb_pcs=neb_pcs, 
                        batch_label=batch_id, attr_lab='retrofit_scenario',
                        n_monte_carlo=n_monte_carlo,
                        log_size=log_size
                    
                    )
        logger.info(f"Successfully processed batch for fuel: {batch_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error in main program: {str(e)}", exc_info=True)
        raise