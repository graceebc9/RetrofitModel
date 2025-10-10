import os 
import pandas as pd
import re 
import geopandas as gpd
from shapely.geometry import box 
import glob 
from typing import Tuple, Optional

from .logging_config import get_logger
logger = get_logger(__name__)



def join_pc_map_three_pc(df, df_col,  pc_map  ):
    """ When joining to postcode files, use all three version of postcodes 
    """
    # merge on any one of three columns in pc_map 
    final_d = [] 
    for col in ['pcd7', 'pcd8', 'pcds']:
        d = df.merge(pc_map , right_on = col, left_on = df_col  )
        final_d.append(d)
    # Concatenate the results
    merged_final = pd.concat(final_d ).drop_duplicates()
    
    if len(df) != len(merged_final):
        print('Warning: some postcodes not matched')
    return merged_final 



def find_data_pc_joint(pc, onsdata, input_gpk, overlap=False):
    """
    Find buildings based on UPRN match to the postcodes and Spatial join 
    input: joint data product from onsud loadaer (pcshp and onsud data) 
    """
    logger.debug(f"Finding data for postcode: {pc}")
    data, pcshp = onsdata 
    pcshp = pcshp[pcshp['POSTCODE']==pc]

    gd = gpd.GeoDataFrame(data[data['PCDS'] == pc].copy(), geometry='geometry')
    if gd.empty:
        logger.warning(f"No data found for postcode {pc}")
        return None 
    
    bbox = box(*gd.total_bounds)
    buildings = gpd.read_file(input_gpk, bbox=bbox)
    uprn_match = buildings[buildings['uprn'].isin(gd['UPRN'])].copy()

    sj_match = buildings.sjoin(pcshp, how='inner', predicate='within')[uprn_match.columns]
    joint_data = pd.concat([uprn_match, sj_match]).drop_duplicates()
    return joint_data 

def check_duplicate_primary_key(df, primary_key_column):
    logger.debug(f"Checking duplicates in column: {primary_key_column}")
    is_duplicate = df[primary_key_column].duplicated().any()
    return is_duplicate

def load_ids_from_file(file_path):
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
    return ids

def load_onsud_data(path_to_onsud_file: str, path_to_pcshp: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load and process ONS UPRN Database (ONSUD) data from a regional file.
    
    Args:
        path_to_onsud_file: Path to the ONSUD CSV file
        path_to_pcshp: Path to the directory containing postcode shapefiles
        
    Returns:
        Tuple containing:
            - Processed ONSUD DataFrame with geographic data
            - Postcode shapefile DataFrame
        Returns None if path_to_onsud_file is None
    """
    if path_to_onsud_file is None:
        logger.warning("No ONSUD file path provided")
        return None
    
    region_label = path_to_onsud_file.split('/')[-1].split('.')[0].split('_')[-1]
    logger.debug(f'Loading ONSUD file for batch: {region_label}')
    
    onsud_df = pd.read_csv(path_to_onsud_file, low_memory=False)
    return find_postcode_for_ONSUD_file(onsud_df, path_to_pcshp)

def find_postcode_for_ONSUD_file(onsud_file: pd.DataFrame, 
                                path_to_pc_shp_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join ONSUD UPRN-to-postcode mapping with geographic postcode data from shapefiles.
    
    Args:
        onsud_file: DataFrame containing ONSUD data
        path_to_pc_shp_folder: Path to folder containing postcode shapefiles
        
    Returns:
        Tuple containing:
            - Merged DataFrame with ONSUD and geographic data
            - Postcode shapefile DataFrame
    """
    logger.debug('Finding postocdes for ONSUD file')
    # Extract leading letters from postcodes and clean data
    onsud_file['leading_letter'] = onsud_file['PCDS'].str.extract(r'^([A-Za-z]{1,2})\d')
    onsud_file = onsud_file[~onsud_file['PCDS'].isna()]
    onsud_file.loc[:, 'PCDS'] = onsud_file['PCDS'].str.strip()
    
    # Load and combine postcode shapefiles
    whole_pc = []
    for pc in onsud_file['leading_letter'].unique():
        pc = pc.lower()
        if len(pc) == 1:
            pc_path = os.path.join(path_to_pc_shp_folder, f'one_letter_pc_code/{pc}/{pc}.shp')
        else:
            pc_path = os.path.join(path_to_pc_shp_folder, f'two_letter_pc_code/{pc}.shp')
        logger.debug(f"Loading shapefile from: {pc_path}")
        pc_shp = gpd.read_file(pc_path)
        whole_pc.append(pc_shp)

    pc_df = pd.concat(whole_pc)
    pc_df['POSTCODE'] = pc_df['POSTCODE'].str.strip()
    
    # Validate postcode coverage
    if len(pc_df.PC_AREA.unique()) != len(onsud_file['leading_letter'].unique()):
        logger.error('Incomplete postcode coverage in shapefile')
        raise ValueError('Incomplete postcode coverage in shapefile')
    
    # Merge and validate data
    check_merge_files(pc_df, onsud_file, 'POSTCODE', 'PCDS')
    merged_data = onsud_file.merge(pc_df, left_on='PCDS', right_on='POSTCODE', how='inner')
    
    missing_count = len(merged_data[merged_data['PC_AREA'].isna()])
    missing_percentage = missing_count / len(merged_data)
    
    if missing_percentage > 0.1:
        logger.error(f'High proportion of missing data: {missing_count} rows ({missing_percentage:.1%})')
        raise ValueError(f'High proportion of missing data: {missing_count} rows ({missing_percentage:.1%})')
    
    logger.info(f"Successfully processed {len(merged_data)} records")
    return merged_data, pc_df

def check_merge_files(df1, df2, col1, col2):
    """
    Validate files and columns before merging.
    """
    # Check if the files are empty
    logger.debug('Checking files before merging')
    if df1.empty or df2.empty:
        logger.error("One or both files are empty")
        if df1.empty:
            logger.error( 'First df empty!!')
        elif df2. empty:
            logger.error('Second df emtpy')
        return False
    
    # Check if the columns to be merged on exist
    if col1 not in df1.columns:
        logger.error(f"Missing merge columns: {col1} for first dataframe  ")
        return False
    elif col2 not in df2.columns:
        logger.error(f"Missing merge columns: {col2} for second dataframe  ")
        return False
    
    # Check columns are same type 
    if df1[col1].dtype != df2[col2].dtype:
        logger.warning(f'Column type mismatch: {col1}({df1[col1].dtype}) != {col2}({df2[col2].dtype})')
    
    return True