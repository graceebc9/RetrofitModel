import os 
import pandas as pd
import re 
import geopandas as gpd
from shapely.geometry import box 
# from src.multi_thread import merge_temp_logs_to_main, generate_batch_list

import glob 

def check_merge_files(df1, df2, col1, col2):
    # Check if the files are empty
    if df1.empty or df2.empty:
        print("Error: One or both files are empty.")
        return False
    
    # Check if the columns to be merged on exist
    if col1 not in df1.columns or col2 not in df2.columns:
        print("Error: One or both columns to be merged on do not exist.")
        return False
    # Check columns are same type 
    if df1[col1].dtype != df2[col2].dtype:
        print('Warning: columns not same type')
    # If one column int, convert toher to int
    
    return True 

# def find_batch_from_pc(pc):
#     print('starting fn')
#     """ find the batch file (txt) based on the postcode"""
#     for file in glob.glob('/Volumes/T9/postcode_data/data/verisk_y2022_V4/atches/*/*txt'):
#         print(file)
#         ids = load_ids_from_file(file)  
#         print('len ids ', len(ids))
#         if pc in ids:
#             return file 
        
def find_batch_from_pc(pc):
    print('starting fn new 2025')
    """ find the batch file (txt) based on the postcode"""
    for file in glob.glob('/Volumes/T9/2024_Data_downloads/Neb_batches/*/*txt'):
        # print(file)
        ids = load_ids_from_file(file)  
        # print('len ids ', len(ids))
        if pc in ids:
            return file 
        
# def load_onsud_from_batch(file):
#     print('file ', file)
#     """ from batxh file name (txt) load the onsud chunk"""
#     reg = file.split('/')[-2]
#     id = file.split('/')[-1].split('.')[0].split('_')[-1]
#     onsud = f'/Volumes/T9/postcode_data/data/batches/{reg}/onsud_{id}.csv'
#     return pd.read_csv(onsud)

def load_onsud_from_batch(file):
    """ from batxh file name (txt) load the onsud chun
    new 2025 fn 
    k"""

    reg = file.split('/')[-2]
    id = file.split('/')[-1].split('.')[0].split('_')[-1]
    onsud = f'/Volumes/T9/2024_Data_downloads/Neb_batches/{reg}/onsud_{id}.csv'
    return pd.read_csv(onsud)


def get_postcode_shapefile(pc, path_to_pc_shp_folder= '/Volumes/T9/Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998/codepoint-poly_5267291'):
    """ find pc shapfile based on the postcode"""
    # Regex pattern to extract 1 or 2 letters at the beginning of the string followed by a digit
    pattern = r'^([A-Za-z]{1,2})\d'
    
    # Using re.search() or re.match() to find the pattern
    match = re.match(pattern, pc)

    # Checking if a match was found and extracting the group
    if match:
        pc_test= match.group(1)  
    pc_test= pc_test.lower()
    if len(pc_test)==1:
        pc_path =os.path.join(path_to_pc_shp_folder,  f'one_letter_pc_code/{pc_test}/{pc_test}.shp'  )
        pc_shp = gpd.read_file(pc_path)    
    else:
        pc_path =os.path.join(path_to_pc_shp_folder,  f'two_letter_pc_code/{pc_test}.shp' ) 
        pc_shp = gpd.read_file(pc_path)  

    return pc_shp[pc_shp['POSTCODE']==pc]


def find_data_pc(pc, data, input_gpk):
    """
    Find buildings based on UPRN match to the postcodes 
    Inputs:
    Pc: postcode
    data: ONSUDE data from loader 
    input_gpk; builder file 
    """
    
    gd = gpd.GeoDataFrame(data[data['PCDS'] == pc].copy(), geometry='geometry')
    
    bbox = box(*gd.total_bounds)
    buildings = gpd.read_file(input_gpk, bbox=bbox)
    uprn_match = buildings[buildings['uprn'].isin(gd['UPRN'])].copy()
    return uprn_match


def find_data_pc_joint(pc, onsdata, input_gpk, overlap=False):
    """
    Find buildings based on UPRN match to the postcodes and Spatial join 
    input: joint data product from onsud loadaer (pcshp and onsud data) 
    """
    
    data, pcshp  = onsdata 
    pcshp = pcshp[pcshp['POSTCODE']==pc]

    gd = gpd.GeoDataFrame(data[data['PCDS'] == pc].copy(), geometry='geometry')
    if gd.empty:
        return None 
    bbox = box(*gd.total_bounds)
    buildings = gpd.read_file(input_gpk, bbox=bbox)
    uprn_match = buildings[buildings['uprn'].isin(gd['UPRN'])].copy()

    sj_match = buildings.sjoin(pcshp, how='inner', predicate='within')[uprn_match.columns]
    joint_data = pd.concat([uprn_match, sj_match ]).drop_duplicates()
    return  joint_data 

def find_data_pc_spatialjoin(pc, input_gpk):
    """
    Find buildings based on spatial join  
    """
    sph = get_postcode_shapefile(pc) 
    
    bbox = box(*sph.total_bounds)
    buildings = gpd.read_file(input_gpk, bbox=bbox)
    uprn_match = buildings.sjoin(sph, how='inner', op='within')
    return uprn_match




def check_duplicate_primary_key(df, primary_key_column):
    # print('checking dupes')
    is_duplicate = df[primary_key_column].duplicated().any()
    return is_duplicate



def load_ids_from_file(file_path):
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
    return ids

def get_onsud_path(onsud_dir, onsud_data  ,label ):

    # DATA_DIR='/Volumes/T9/Data_downloads/ONS_UPRN_database/ONSUD_DEC_2022'
    # date = onsud_dir.split('/')[-1].split('ONSUD_')[-1]
    filepath = os.path.join(onsud_dir, f'Data/ONSUD_{onsud_data}_{label}.csv' ) 
    return filepath

def get_onsud_path_batches(onsud_dir, onsud_data  ,label ):
    filepath = os.path.join(onsud_dir, f'Data/ONSUD_{onsud_data}_{label}.csv' ) 
    return filepath



def load_onsud_data(path_to_onsud_file, path_to_pcshp, ):
    if path_to_onsud_file is None: 
        return None 
    label = path_to_onsud_file.split('/')[-1].split('.')[0].split('_')[-1]
    print(f'Finding data for ONSUD file ', label )
    onsud_df = pd.read_csv(path_to_onsud_file)
    onsud_data, pc_df  = find_postcode_for_ONSUD_file(onsud_file= onsud_df, path_to_pc_shp_folder= path_to_pcshp)

    return onsud_data , pc_df

def get_pcs_to_process(onsud_data, log):
    pcs_list =  onsud_data.PCDS.unique().tolist()
    merge_temp_logs_to_main(log)
    pc_list = generate_batch_list(pcs_list, log , 'postcode')
    return pc_list 
    

def find_postcode_for_ONSUD_file(onsud_file, path_to_pc_shp_folder):
    """ Join ONSUD UPRN TO postcode mapping to postcode geofiles with shapefiles
    onsud file is raw onsud file
    """
    onsud_file['leading_letter'] = onsud_file['PCDS'].str.extract(r'^([A-Za-z]{1,2})\d')
    onsud_file= onsud_file[~onsud_file['PCDS'].isna() ] 
    onsud_file['PCDS'] = onsud_file['PCDS'].str.strip()
    
    whole_pc = [] 
    for pc in onsud_file['leading_letter'].unique():
        pc= pc.lower()
        if len(pc)==1:
            pc_path =os.path.join(path_to_pc_shp_folder,  f'one_letter_pc_code/{pc}/{pc}.shp'  )
            pc_shp = gpd.read_file(pc_path)    
        else:
            pc_path =os.path.join(path_to_pc_shp_folder,  f'two_letter_pc_code/{pc}.shp' ) 
            pc_shp = gpd.read_file(pc_path)    
        whole_pc.append(pc_shp)

    pc_df = pd.concat(whole_pc)
    pc_df['POSTCODE'] = pc_df['POSTCODE'].str.strip() 
      
    if len(pc_df.PC_AREA.unique().tolist()) != len(onsud_file['leading_letter'].unique().tolist()):
        raise ValueError('Not all postcodes are present in the shapefile') 

    check_merge_files(pc_df, onsud_file, 'POSTCODE', 'PCDS') 

    data = onsud_file.merge(pc_df, left_on='PCDS', right_on='POSTCODE', how='inner')
    
    print('Len of missing rows ', len(data[data['PC_AREA'].isna()] ) ) 
    
    if len(data[data['PC_AREA'].isna()] ) > 0.1*len(data):
        raise ValueError('More than 10% of the data is missing')    
    
    return data ,pc_df 