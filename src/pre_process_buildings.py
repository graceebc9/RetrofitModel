"""
Module: pre_process_buildings.py
Description: Pre process the Verisk building data to calculate building metrics.

Key features
 - Update age buckets to be non overlapping
 - bucket height into set batches to enable matching to global averages 
 - fills with nulls for casese where there are missing buildings or missing gas / elec data 
 - update domestic outbuildings incorrectly labelled and those with wrong average floor heights / counts 
 

Author: Grace Colverd
Created: 05/2024
Modified: 2024
"""

import pandas as pd
import numpy as np
import os 
from .postcode_utils import check_merge_files


from .logging_config import get_logger
logger = get_logger(__name__)



# ============================================================
# Constants
# ============================================================

BASEMENT_HEIGHT = 2.4
BASEMENT_PERCENTAGE_OF_PREMISE_AREA = 1
DEFAULT_FLOOR_HEIGHT = 2.3
average_scaling_factor = 0.58
# changed function to allow flexible but these are the defaults
# MAX_THRESHOLD_FLOOR_HEIGHT = 5.3
# MIN_THRESH_FL_HEIGHT = 2.2

# ============================================================
# Data Loading Functions
# ============================================================

# current using not age split floor count 
def load_avg_floor_count():
    """Load the average floor count data from a CSV file."""
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'global_avs' , 'global_average_floor_count_bucket_clean.csv')
    
    return pd.read_csv(csv_path)

def load_scaling_table():
    """Load the GEA (from EPC) scaling table data from a CSV file."""
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'global_avs', 'scaling_factor_gea.csv')
    df = pd.read_csv(csv_path)
    return df


def create_age_buckets(df):
    df['premise_age_bucketed'] = np.where(df['premise_age'].isin(['Pre 1837', '1837-1869', '1870-1918']), 'Pre 1919', df['premise_age'])
    return df 

def get_height_bins():
    height_bins = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 200]
    height_labels = [f"{b}-{height_bins[i+1]}m" for i, b in enumerate(height_bins[:-1])]
    return height_bins, height_labels


def create_height_bucket_cols(df, col):
    """Bucket height into predefined categories."""
    height_bins, height_labels = get_height_bins() 
    # check if col exists in df 
    if col not in df.columns:
        raise Exception(f'Column {col} not found in df')
    df[f'{col}_bucket'] = pd.cut(df[col], bins=height_bins, labels=height_labels, right=False)
    return df


def update_listed_type(df):
    df.loc[:, 'listed_bool'] = df['listed_grade'].apply(lambda x: 1 if x is not None else 0)
    return df 


# ============================================================
# Scaling factrors
# ============================================================

def load_scaling_factor():
    """Load the scaling factor data from a CSV file."""
    current_dir = os.path.dirname(__file__)
    # csv_path = os.path.join(current_dir, 'global_avs', 'scaling_factor.csv')
    csv = '/Users/gracecolverd/NebulaDataset/notebooks/scaling_factor.csv'
    df = pd.read_csv(csv) 
    return df 

 

# ============================================================
# Geometry Fns
# ============================================================


def min_side(polygon):
    
    # Minimum rotated rectangle
    min_rect = polygon.minimum_rotated_rectangle

    # Extract the points of the rectangle to calculate side lengths
    x, y = min_rect.exterior.coords.xy

    # Calculate distances between consecutive points (sides of the rectangle)
    distances = [np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) for i in range(1, len(x))]

    # The least width is the minimum side length of the rectangle
    least_width = min(distances)
    return least_width 


def get_perimeter(polygon):
    # get perimeter of polygon 
    return polygon.length


# ============================================================
# Pre process fns 
# ============================================================


def update_outbuildings(test):
    test.loc[(test['height']==3) & (test['premise_floor_count'] == '2') & (test['uprn_count'] == 0), 'premise_type'] = 'Domestic outbuilding'
    return test




def update_avgfloor_count_outliers(df, MIN_THRESH_FL_HEIGHT, MAX_THRESHOLD_FLOOR_HEIGHT):
    
    df.to_crs('EPSG:27700', inplace=True )
    
    df['min_side'] = df['geometry'].astype(object).apply(min_side)
    # print(df['min_side'])
    df['perimeter_length']=df['geometry'].astype(object).apply(get_perimeter) 
    df['threex_minside'] = [x * 3 for x in df['min_side']]
    
    # Update height validation to include new constraint for heights < 2m with floor count
    df['validated_height'] = np.where(
        (df['height_numeric'] >= df['threex_minside']) | 
        ((df['height_numeric'] < 2) & df['floor_count_numeric'].notna()),
        np.nan,
        df['height_numeric']
    )
    
    df['validated_fc'] = np.where(
        ((df['av_fl_height'] >= MAX_THRESHOLD_FLOOR_HEIGHT) | 
         (df['av_fl_height'] <= MIN_THRESH_FL_HEIGHT)) & 
        (df['height'] < df['threex_minside']),
        np.nan,
        df['floor_count_numeric']
    )
    
    return df


def fill_local_averages(df):
    logger.debug('starting to fill local averages')
    num_builds = len(df )

    num_fc_invalid = df.validated_fc.isna().sum() 
    num_h_invalid = df.validated_height.isna().sum() 

    if num_builds - num_fc_invalid ==0 | len(df)==1 | num_builds - num_h_invalid == 0 : 
        print('Cannot do local fill for FC')
        # height_fla = df[~df['validated_height'].isna()]['validated_height'].mean()
        raise Exception('no local fill ')

    fc_fla= df[~df['validated_fc'].isna()]['validated_fc'].mean()
    
    height_fla = df[~df['validated_height'].isna()]['validated_height'].mean()
    
    df['fc_filled'] = np.where(df['validated_fc'].isna(), fc_fla, df['validated_fc'])
    df['height_filled'] = np.where(df['validated_height'].isna(), height_fla, df['validated_height'] )
    df = create_height_bucket_cols(df, 'height_filled')
    logger.debug('Fill local averages complete')
    return df 

def fill_glob_avs(df, fc = None  ):
    logger.debug('Starting to fill global averages')
    if fc is None:
        fc = load_avg_floor_count() 
    
    fc['height_bucket'] = fc['height_bucket'].astype('object')
    df['height_filled_bucket'] = df['height_filled_bucket'].astype('object')
    
    logger.debug('Starting checks before global average merge')
    check_merge_files(fc, df, 'map_simple_use', 'map_simple_use')
    check_merge_files(fc, df, 'height_bucket', 'height_filled_bucket')
    check_merge_files(fc, df, 'premise_age_bucketed', 'premise_age_bucketed')
    
    df = df.merge(fc, left_on=['map_simple_use', 'premise_age_bucketed',  'height_filled_bucket'], right_on = [ 'map_simple_use', 'premise_age_bucketed', 'height_bucket'], how='left')
    if df.empty:
        raise Exception ('Error merging with global averages')
    df.drop(columns=[ 'height_bucket','total_count'], inplace=True )
    logger.debug('global average fill complete')
    return df 


# def create_heated_vol(df):
#     """
#     calc heated premise are
#     """
#     df['total_fl_area_H'] = df['premise_area'] * df['global_average_floorcount']
#     df['total_fl_area_FC'] = df['premise_area'] * df['floor_count_numeric']
#     df ['total_fl_area_valfc'] = df['premise_area'] * df['fc_filled']
#     return df 

def create_heated_vol(df, scaling_table):
    """
    Calculate heated premise area metrics and create meta columns for analysis.
    Hierarchy: H (global average) → valfc (filled floor count) → FC (raw floor count)
    Also includes average of available values.
    """
    # Calculate base metrics
    df['total_fl_area_H'] = df['premise_area'] * df['global_average_floorcount']
    df['total_fl_area_FC'] = df['premise_area'] * df['floor_count_numeric']
    df['total_fl_area_valfc'] = df['premise_area'] * df['fc_filled']
    df = df.merge(scaling_table[['premise_type', 'premise_age_bucketed', 'scaling']], on=['premise_type', 'premise_age_bucketed'], how='left')
    
    
    df['scaling'] = df['scaling'].fillna(df.scaling.mean())

    # Create meta column with preferred hierarchy
    conditions = [
        df['total_fl_area_H'].notna(),
        df['total_fl_area_valfc'].notna(),
        df['total_fl_area_FC'].notna()
    ]
    choices_value = [
        df['total_fl_area_H'], 
        df['total_fl_area_valfc'], 
        df['total_fl_area_FC']
    ]
    choices_source = ['H', 'valfc', 'FC']
    
    df['total_fl_area_meta'] = np.select(conditions, choices_value, default=np.nan)
    df['total_fl_area_meta_source'] = np.select(conditions, choices_source, default='none')
    
    # Calculate average of available values
    columns = ['total_fl_area_H', 'total_fl_area_valfc', 'total_fl_area_FC']
    df['total_fl_area_avg'] = df[columns].mean(axis=1)
    df['scaling'] = df['scaling'].fillna(average_scaling_factor)
    df['scaled_fl_area'] = df['total_fl_area_meta'] * df['scaling']
    return df

def create_basement_metrics(df):
    basement_conditions = [
        df['basement'].isin(['Basement confirmed', 'Basement likely']),
        ~df['basement'].isin(['Basement confirmed', 'Basement likely'])
    ]
    basement_choices = [1, 0]
    df['base_floor'] = np.select(basement_conditions, basement_choices, default=0)
    df['basement_heated_vol'] = df['base_floor'] *  df['premise_area'] * BASEMENT_HEIGHT * BASEMENT_PERCENTAGE_OF_PREMISE_AREA 
    return df 

def pre_process_buildings(df, fc, scaling_table,  MIN_THRESH_FL_HEIGHT = 2.1, MAX_THRESH_FL_HEIGHT= 5.1):
    """ can only be applied   to a group where you want the local average within the group
    - bcuekts age (turns all pre into pre 1919
    - updat listed into numeric / encoded
    - update outbuilds: for those with heht 3 storey 2 uprn 0 -> outbuildings 
    - update fc outliers: use height / min width ratio and average floor height to nul out invalid heights or fc
    - fill local avs:fill the nulled with local averages 
    - fill glob averages: any local heights, convert to fc with global averages 
    - create heated vol from two diff fcs 
    - create basement metrics using whole premise area for basemenet, whole area heated (can vary this later)

    Ends up with two diff ways of getting floor count, for heated volume :
         either directly (correct fc or local av fc) 
         or through height (either correct height, or local av hegith, converted to global fc for that height)
    """ 

    df = create_age_buckets(df)
    
    df['height_numeric'] = pd.to_numeric(df['height'], errors='coerce')
    df['floor_count_numeric'] = pd.to_numeric(df['premise_floor_count'], errors='coerce')
    df['av_fl_height'] = df['height_numeric'] / df['floor_count_numeric']
    
    df=update_listed_type(df)
    df=update_outbuildings(df)
    df=update_avgfloor_count_outliers(df, MIN_THRESH_FL_HEIGHT, MAX_THRESH_FL_HEIGHT)
    df=fill_local_averages(df)
    df=fill_glob_avs(df, fc)
    df = create_heated_vol(df, scaling_table)
    df = create_basement_metrics(df)
    if df.empty:
        raise Exception('Error empty df ')
    return df 

# ============================================================
# Filter fns
# ============================================================

def produce_clean_building_data(df):
    """Filter and test building data."""
    # print("Filtering non-commercial derelict premises...")
 
    if len(df)==0:
        print('No data to process')
        return None 
    # filtered_df = df[df['premise_use'] != 'Commercial - derelict'].copy()
    fdf =df[df['premise_type']=='Residential'].copy() 
    test_building_metrics(fdf)

    return df

# ============================================================
# Validation and Testing Functions
# ============================================================
def assert_larger(df, col1, col2):
    """Ensure values in col1 are larger than those in col2 where columns are not null."""
    df = df[~df[col1].isna() & ~df[col2].isna()].copy()
    assert (df[col1] >= df[col2]).all(), f"Found rows where {col1} is not larger than {col2}."

def assert_perc(df, col):
    """Ensure values in col are between 0 and 1."""
    assert ((df[col] >= 0) & (df[col] <= 1)).all(), f"Found rows where {col} is not between 0 and 1."

def assert_equal(df, col1, col2):
    """Ensure values in col1 are equal to those in col2."""
    assert (df[col1] == df[col2]).all(), f"For df, found rows where {col1} does not equal {col2}."

def check_nulls_percent(df, col, threshold=0.5):
    """Check if the percentage of nulls in col exceeds a threshold."""
    if df[col].isna().mean() > threshold:
        raise Exception(f'Nulls in {col} are greater than {threshold*100}%.')

  

def test_building_metrics(df):
    """Run various assertions on building metrics."""


    for c in ['total_fl_area_H', 'total_fl_area_FC']:
        check_nulls_percent(df, c, 0)

    test = df[df['validated_height'].isna()].copy() 
    assert_larger(test, 'height', 'height_filled')


def pre_process_building_data(build,   MIN_THRESH_FL_HEIGHT = 2.1, MAX_THRESH_FL_HEIGHT= 5.1):
    # print(MIN_THRESH_FL_HEIGHT,MAX_THRESH_FL_HEIGHT )
    """Calculate and validate building metrics from verisk data."""

    logger.debug(f'Starting to pre process buildingd data, for min threhold floor height: {MIN_THRESH_FL_HEIGHT} and max threshold floor height: {MAX_THRESH_FL_HEIGHT} ')
    
    fc = load_avg_floor_count() 
    logger.debug('Loaded average floor count global averages ')
    
    scaling_table = load_scaling_table()
    processed_df = pre_process_buildings(build, fc , scaling_table, MIN_THRESH_FL_HEIGHT, MAX_THRESH_FL_HEIGHT)
    
    clean_df = produce_clean_building_data(processed_df)
    
    return clean_df