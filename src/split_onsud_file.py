"""
Module: split_onsud_file.py
Description: Processes and splits ONS UPRN Database (ONSUD) files into manageable batches by region.
This module handles the relationship between postcodes and their associated UPRNs (Unique Property Reference Numbers),
organizing them into batch files for efficient processing.

Key Features:
- Loads and validates ONSUD data from regional CSV files e.g. 'EE', 'NW'
- Matches UPRN-postcode mappings with geographic postcode data from shapefiles
- Splits large ONSUD datasets into manageable batches
- Supports both single and double-letter postcode areas

Author: Grace Colverd
Created: 05/2024
Modified: 2024
"""

import os 
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional
from .postcode_utils import check_merge_files, load_onsud_data, find_postcode_for_ONSUD_file


from .logging_config import get_logger
logger = get_logger(__name__)


def split_onsud_and_postcodes(path_to_onsud_file: str, 
                             path_to_pcshp: str, 
                             batch_size: int = 10000) -> None:
    """
    Split ONSUD data and associated postcodes into manageable batches.
    
    Args:
        path_to_onsud_file: Path to the ONSUD CSV file
        path_to_pcshp: Path to the postcode shapefile directory
        batch_size: Number of postcodes per batch (default: 10000)
        
    Output Structure:
        ├── batches/
        │   └── region_label/
        │       ├── batch_0.txt (contains batch_size postcodes)
        │       ├── batch_1.txt
        │       ├── onsud_0.csv (contains UPRNs for batch_0 postcodes)
        │       ├── onsud_1.csv
        │       └── ...
        └── batch_paths.txt
        
    Notes:
        - Resumes processing from last completed batch if interrupted
        - Maintains a log file of processed postcodes
        - Creates separate files for postcodes and their associated UPRNs
    """
    region_label = path_to_onsud_file.split('/')[-1].split('.')[0].split('_')[-1]
    logfile = os.path.join( region_label, 'log_file.csv')
    
    print(f'Processing region: {region_label}')
    
    # Load and prepare data
    onsud_data, _ = load_onsud_data(path_to_onsud_file, path_to_pcshp)
    raw_data = pd.read_csv(path_to_onsud_file, low_memory=False)
    
    # Get list of postcodes to process
    pcs_list = onsud_data.PCDS.unique().tolist()
    
    # Resume from last processed batch if log exists
    if os.path.exists(logfile):
        log = pd.read_csv(logfile)
        done_ids = log['postcode'].unique().tolist()
        pcs_list = [pc for pc in pcs_list if pc not in done_ids]
    
    # Create batch directory
    batch_dir = f'batches/{region_label}/'
    os.makedirs(batch_dir, exist_ok=True)
    logger.info('batch dir: ')
    logger.info(batch_dir)
    logger.info('len pc list ' )
    logger.info(  len(pcs_list))
    
    # Process batches
    for i in range(0, len(pcs_list), batch_size):
        batch = pcs_list[i:i+batch_size]
        batch_num = i // batch_size
        
        # Save postcode batch
        batch_filename = os.path.join(batch_dir, f"batch_{batch_num}.txt")
        with open(batch_filename, 'w') as f:
            f.write('\n'.join(batch))
        
        # Save batch path
        with open('batch_paths.txt', 'a') as f:
            f.write(f"{batch_filename}\n")
        
        # Save associated ONSUD data
        subset_data = raw_data[raw_data['PCDS'].str.strip().isin(batch)].copy()
        subset_data.to_csv(f'{batch_dir}/onsud_{batch_num}.csv', index=False)
    
    print(f'Successfully saved {len(pcs_list) // batch_size + 1} batches to {batch_dir}')

