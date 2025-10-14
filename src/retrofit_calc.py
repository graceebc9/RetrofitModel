# At the top of retrofit_calc.py
from .logging_config import get_logger

logger = get_logger(__name__)


import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
# Assuming these modules are available in the package structure
from .RetrofitScenarioGenerator import RetrofitScenarioGenerator 
from .RetrofitModel import RetrofitConfig, BuildingCharacteristics
from .postcode_utils import find_data_pc_joint
from .pre_process_buildings import pre_process_building_data


from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent


def load_energy():
    energy_df = pd.read_csv(f'{root_dir}/src/global_avs/neb_enegry_data.csv')
    return energy_df

def load_eui(eui_path= f'{root_dir}/src/global_avs/neb_eui_table.csv'):
    """
    Load the processed nebula gas EUI and elec EUI variables """
    eui_df = pd.read_csv(eui_path)
    return eui_df 

def get_eui_factor(eui_df, pc, region):
    """
    Get gas and elec EUI factor. first try PC, then region average, then gobal mean (hardcoded)
    """
    pc_res = eui_df[eui_df['postcode']==pc]
    if pc_res.empty:
        try:
            regional_res = eui_df.groupby('region')[['gas_EUI_GIA',  'elec_EUI_GIA']].mean().reset_index()
            elec = regional_res[regional_res['region']==region]['elec_EUI_GIA'].values[0]
            gas = regional_res[regional_res['region']==region]['gas_EUI_GIA'].values[0]
        except: 
            gas =  137.669361
            elec = 38.907537
    else: 
        elec = pc_res['elec_EUI_GIA'].values[0]
        gas = pc_res['gas_EUI_GIA'].values[0]
    return gas, elec 

 
 

def get_conservation_area(uprn_match, conservation_data):
    """
    Tag buildings with conservation areas downloaded from ons 
    """
    uprn_match.to_crs(epsg='4326', inplace=True )
    conservation_data.to_crs(epsg='4326', inplace=True )
    cons_uprn = uprn_match.sjoin(conservation_data, how='left')

    cons_uprn['conservation_area_bool'] = np.where(cons_uprn['NAME'].notna(), True, False )
    cons_uprn['conservation_area_name'] = np.where(cons_uprn['NAME'].notna(), cons_uprn['NAME'], 'None' )
    cols = uprn_match.columns.tolist() + ['conservation_area_bool',  'conservation_area_name']
    return cons_uprn[cols]


def process_postcodes_for_retrofit_with_uncertainty(
    pc,
    onsud_data,
    INPUT_GPK,
    region,
    # retrofit_config,
    retrofig_model,
    energy_column, 
    scenarios,
    conservation_data,
    # n_monte_carlo,
    random_seed=42
):
    """
    Process postcode with full uncertainty analysis.
    
    Returns:
        dict: Results with scenario costs, energy savings, and uncertainty
    """
    def load_gas_deciles(path):
        gas_deciles = pd.read_csv(path)
        return gas_deciles 

    def get_gas_decile_single(pc, decile_df):
        res = decile_df[decile_df['postcode']==pc]
        if res.empty:
            raise Exception('missing gas deciles for pcs')
        else: 
            return res.avg_gas_decile.unique()


    pc = pc.strip()
    logger.debug('Finding UPRN Data')
    uprn_match = find_data_pc_joint(pc, onsud_data, input_gpk=INPUT_GPK)
    uprn_match=get_conservation_area(uprn_match, conservation_data)

    error_dict = {
        'postcode': pc,
        'error': 'No building data found',
        'basic_maintenance_cost': None,
        'comprehensive_fabric_cost': None,
        'deep_retrofit_cost': None,
        'electrification_ready_cost': None,
        'fabric_first_lite_cost': None,
        'total_flat_count': None
    }
    
    if uprn_match is None or uprn_match.empty:
        return error_dict
    logger.debug('Loading EUI' ) 
    energy = load_eui() 
    logger.debug('Pre process buildigns')
    building_data = pre_process_building_data(uprn_match)
    gas_eui, elec_eui = get_eui_factor(pc=pc, eui_df= energy, region = region)
    building_data['total_gas_derived'] =  building_data['total_fl_area_meta'] * gas_eui
    building_data['total_elec_derived'] =  building_data['total_fl_area_meta'] * elec_eui
    building_data['total_energy_dervied'] = building_data['total_gas_derived']  + building_data['total_elec_derived']
    deciles = load_gas_deciles('/Users/gracecolverd/NebulaDataset/notebooks2/neb_unfil_final_gas_deciles.csv')

    gas_decile = get_gas_decile_single(pc, deciles)
    building_data['avg_gas_percentile'] = [gas_decile[0] for x in range(len(building_data)  ) ]
    if building_data is None or building_data.empty:
        return error_dict
    
    # Check for energy data
    if energy_column not in building_data.columns:
        error_dict['error'] = f'Missing energy column: {energy_column}'
        return error_dict
    
    logger.debug('Starting scenario generation')
    scen = RetrofitScenarioGenerator()
    
   
    results = scen.process_dataframe_scenarios( 
    df = building_data,
    region = region,
    model_class = retrofig_model, 
    # typ_config = retrofit_config, 
    scenarios=scenarios,
    random_seed=random_seed
        )
    
    logger.debug('Processing results .. ')
    
    if 'error' in results:
            logger.debug(f'Error found in results for pc {pc}')
            error_dict.update(results)
            return error_dict
    else:
        results['postcode'] = pc 
        return results 
    