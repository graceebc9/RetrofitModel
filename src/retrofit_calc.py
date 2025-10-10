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



def load_eui(eui_path= '/Volumes/T9/2024_Data_downloads/2024_11_nebula_paper_data/2025_revisions/20225_11_final_submission/neb_eui_table.csv'):
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


# flats per beilding 

# def calculate_estimated_flats_per_building(building_footprint_area, typology_col, floor_count):
#     """Calculate estimated number of flats based on building characteristics."""
#     house_typologies = [
#         'Small low terraces', 'Tall terraces 3-4 storeys', 'Large semi detached',
#         'Standard size detached', 'Standard size semi detached',
#         '2 storeys terraces with t rear extension', 'Semi type house in multiples',
#         'Large detached', 'Very large detached', 'Linked and step linked premises',
#         'Domestic outbuilding',
#     ]
    
#     if typology_col in house_typologies or typology_col == 'all_unknown_typology':
#         return 1
    
#     typical_flat_footprints = {
#         'Medium height flats 5-6 storeys': 50,
#         '3-4 storey and smaller flats': 60,
#         'Tall flats 6-15 storeys': 45,
#         'Very tall point block flats': 40,
#         'Planned balanced mixed estates': 65,
#     }
    
#     efficiency_factors = {
#         'Medium height flats 5-6 storeys': 0.75,
#         '3-4 storey and smaller flats': 0.80,
#         'Tall flats 6-15 storeys': 0.70,
#         'Very tall point block flats': 0.65,
#         'Planned balanced mixed estates': 0.80,
#     }
    
#     flat_footprint = typical_flat_footprints.get(typology_col, 50)
#     efficiency = efficiency_factors.get(typology_col, 0.75)
    
#     try:
#         usable_area_per_floor = building_footprint_area * efficiency
#         flats_per_floor = usable_area_per_floor / flat_footprint
#         total_flats = float(floor_count) * float(flats_per_floor)
#         return max(1, round(total_flats))
#     except (TypeError, ZeroDivisionError, ValueError) as e:
#         # E: Replaced magic number -999 with 1 and logged the error
#         logging.error(f"Error calculating flats for typology {typology_col}: {e}. Defaulting to 1.")
#         return 1



def process_postcodes_for_retrofit_with_uncertainty(
    pc,
    onsud_data,
    INPUT_GPK,
    region,
    retrofit_config,
    retrofig_model,
    energy_column, 
    scenarios,
    n_monte_carlo,
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
    
    uprn_match = find_data_pc_joint(pc, onsud_data, input_gpk=INPUT_GPK)
   
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
    
    energy = load_eui() 
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
    scen = RetrofitScenarioGenerator()
    # Calculate with uncertainty
   
    results = scen.process_dataframe_scenarios( 
    df = building_data,
    region = region,
    model_class = retrofig_model, 
    typ_config = retrofit_config, 
    scenarios=scenarios,
    random_seed=random_seed
        )
   
    
    if 'error' in results:
            error_dict.update(results)
            return error_dict
    else:
        results['postcode'] = pc 
        return results 
    

# def calculate_building_costs_df(df, region, col_mapping=None, config=None):
#     """
#     Apply building cost calculations to all rows in a DataFrame.
    
#     Note: Costs returned are deterministic (mean) costs. Full uncertainty analysis 
#     would require sampling costs within the Monte Carlo loop, which is currently 
#     omitted (Issue C).
    
#     Returns:
#         DataFrame with cost columns OR dict with error
#     """
#     try:
#         if df is None or df.empty:
#             return {'error': 'DataFrame is None or empty'}
        
#         if not region:
#             return {'error': 'Region parameter is required'}
        
#         if config is None:
#             config = RetrofitConfig()
        
#         # Mapping definition (as before)
#         default_mapping = {
#             'floor_count': 'total_fl_area_avg',
#             'gross_external_area': 'total_fl_area_avg',
#             'gross_internal_area': 'scaled_fl_area',
#             'footprint_circumference': 'perimeter_length',
#             'flat_count': 'est_num_flats',
#             'building_type': 'premise_type',
#             'age_band': 'premise_age',
#             'building_footprint_area': 'premise_area',
#         }
        
#         if col_mapping is None:
#             col_mapping = default_mapping
#         else:
#             for key, default_val in default_mapping.items():
#                 if key not in col_mapping:
#                     col_mapping[key] = default_val
        
#         # Validate columns exist
#         missing_columns = [
#             f"{field} -> {col_name}" 
#             for field, col_name in col_mapping.items() 
#             if col_name not in df.columns
#         ]
        
#         if missing_columns:
#             return {'error': f'Required columns not found: {", ".join(missing_columns)}'}
        
#         result_df = df.copy()
#         result_df = result_df[result_df[col_mapping['building_type']] != 'Domestic outbuilding']
        
#         # Get list of all possible interventions from the config 
#         all_interventions = config.retrofit_packages['all']
        
#         def calculate_row_costs(row):
           
            
#             # Data validation (as before)
#             required_cols = ['floor_count', 'gross_external_area', 'gross_internal_area',
#                            'footprint_circumference', 'building_type', 'age_band', 'building_footprint_area']
            
#             missing_cols = [col for col in required_cols if col_mapping[col] not in row.index]
#             if missing_cols:
#                 raise ValueError(f'Missing columns: {missing_cols}')
            
#             # Convert and validate
#             floor_count = int(row[col_mapping['floor_count']])
#             gross_external_area = float(row[col_mapping['gross_external_area']])
#             gross_internal_area = float(row[col_mapping['gross_internal_area']])
#             footprint_circumference = float(row[col_mapping['footprint_circumference']])
#             building_footprint_area = float(row[col_mapping['building_footprint_area']])
            
#             # Use max(1) for flat count to ensure valid input to cost calcs
#             raw_flat_count = row.get(col_mapping['flat_count'])
#             flat_count = int(raw_flat_count) if pd.notna(raw_flat_count) and raw_flat_count > 0 else 1
            
#             if any([
#                 pd.isna(floor_count) or floor_count <= 0,
#                 pd.isna(gross_external_area) or gross_external_area <= 0,
#                 pd.isna(gross_internal_area) or gross_internal_area <= 0,
#                 pd.isna(footprint_circumference) or footprint_circumference <= 0,
#                 pd.isna(building_footprint_area) or building_footprint_area <= 0
#             ]):
#                 raise ValueError('Invalid building characteristics')
            
#             building_chars = BuildingCharacteristics(
#                 floor_count=floor_count,
#                 gross_external_area=gross_external_area,
#                 gross_internal_area=gross_internal_area,
#                 footprint_circumference=footprint_circumference,
#                 flat_count=flat_count,
#                 building_footprint_area=building_footprint_area
#             )
            
#             # D: Fixed - Calculate cost for ALL interventions
#             costs = {}
#             typology = row[col_mapping['building_type']]
#             age_band = row[col_mapping['age_band']]

#             for intervention in all_interventions:
#                 try:
#                     # Note: This is the mean/deterministic cost (Issue C)
#                     cost = config.calculate_intervention_cost(
#                         intervention,
#                         building_chars,
#                         typology,
#                         age_band,
#                         region
#                     )
#                     # Skip scenario if cost is NaN or invalid
#                     if pd.isna(cost) or cost <= 0:
#                         logging.debug(f"Skipping scenario {intervention} for building - cost is NaN or invalid")
#                         costs[intervention] = np.nan
#                     else:
#                         costs[intervention] = cost
                    
#                 except ValueError as ve:
#                     # This happens if an intervention is in 'all' but not configured
#                     logging.warning(f"Cost calculation skipped for {intervention}: {ve}")
#                     costs[intervention] = np.nan
            
#             return pd.Series(costs)
        
#         cost_results = result_df.apply(calculate_row_costs, axis=1)
        
#         for col in cost_results.columns:
#             result_df[f'cost_{col}'] = cost_results[col]
        
#         return result_df
        
#     except Exception as e:
#         return {'error': f'Unexpected error: {str(e)}'}

# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Any, Tuple
# import numpy as np

# @dataclass
# class BuildingCharacteristics:
#     """Building physical characteristics for cost calculation."""
#     floor_count: int
#     gross_external_area: float  # sq m
#     gross_internal_area: float  # sq m
#     footprint_circumference: float  # m
#     building_footprint_area: float  # sq m
#     flat_count: Optional[int] = None
    
#     @property
#     def external_wall_area_estimate(self) -> float:
#         """Estimate external wall area from circumference and floor count."""
#         return self.footprint_circumference * self.floor_count * 2.7
    
#     @property
#     def roof_area_estimate(self) -> float:
#         """Estimate roof area from footprint."""
#         return self.building_footprint_area


# @dataclass
# class InterventionScalingRules:
#     """Defines how interventions scale with building characteristics AND their energy savings."""
    
#     # Area-based interventions (cost per sqm) - BASE VALUES for deterministic calculation
#     area_based_interventions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
#         'loft_insulation': {
#             'base_cost_per_sqm': 20,  # Mode value from uncertainty_parameters
#             'min_cost': 500,
#             'max_cost': 2000,
#             'area_type': 'roof'
#         },
#         'cavity_wall_insulation': {
#             'base_cost_per_sqm': 20,
#             'min_cost': 500,
#             'max_cost': 3000,
#             'area_type': 'wall'
#         },
#         'internal_wall_insulation': {
#             'base_cost_per_sqm': 95,
#             'min_cost': 6000,
#             'max_cost': 9000,
#             'area_type': 'wall'
#         },
#         'external_wall_insulation': {
#             'base_cost_per_sqm': 115,
#             'min_cost': 7100,
#             'max_cost': 15000,
#             'area_type': 'wall'
#         },
#         'solar_pv': {
#             'base_cost_per_sqm': 300,
#             'min_cost': 2000,
#             'max_cost': 10000,
#             'area_type': 'roof'
#         },
#         'floor_insulation': {
#             'base_cost_per_sqm': 30,
#             'min_cost': 500,
#             'max_cost': 4000,
#             'area_type': 'floor'
#         },
#         'deep_retrofit_estimate': {
#             'base_cost_per_sqm': 30,
#             'min_cost': 500,
#             'max_cost': 4000,
#             'area_type': 'internal'
#         }
#     })
    
#     # Cost uncertainties for Monte Carlo sampling
#     uncertainty_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
#         'loft_insulation': {
#             'distribution': 'triangular',
#             'min': 10,
#             'mode': 20,   
#             'max': 30,
#             'confidence': 'med',
#             'cap': [500, 2000],
#         },
#         'cavity_wall_insulation': {
#             'distribution': 'triangular',
#             'min': 10,
#             'mode': 20,
#             'max': 30,
#             'confidence': 'med',
#             'cap': [500, 3000],
#         },
#         'internal_wall_insulation': {
#             'distribution': 'triangular',
#             'min': 55,
#             'mode': 95,
#             'max': 140,
#             'confidence': 'med',
#             'cap': [6000, 9000],
#         },
#         'external_wall_insulation': {
#             'distribution': 'triangular',
#             'min': 70,
#             'mode': 115,
#             'max': 160,
#             'confidence': 'med',
#             'cap': [7100, 15000],
#         },
#         'solar_pv': {
#             'distribution': 'triangular',
#             'min': 100,
#             'mode': 300,
#             'max': 500,
#             'cap': [2000, 10000],
#         },
#         'floor_insulation': {
#             'distribution': 'triangular',
#             'min': 10,
#             'mode': 30,
#             'max': 50,
#             'cap': [500, 4000],
#         },
#         'deep_retrofit_estimate': { 
#             'distribution': 'triangular',
#             'min': 10,
#             'mode': 30,
#             'max': 50,
#             'cap': [500, 4000],
#         }
#     })
    
#     # Fixed cost interventions (no area scaling)
#     fixed_cost_interventions: Dict[str, float] = field(default_factory=lambda: {
#         'draught_proofing': 350,
#         'heating_controls': 800,
#         'cylinder_jacket': 150,
#         'boiler_service': 200,
#     })
    
#     # Heat pump cost ranges by typology
#     heat_pump_cost_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
#         # Flats: 7-9k
#         'Very tall point block flats': (7000, 9000),
#         'Tall flats 6-15 storeys': (7000, 9000),
#         'Medium height flats 5-6 storeys': (7000, 9000),
#         '3-4 storey and smaller flats': (7000, 9000),
        
#         # Terraces: 7-9k
#         'Tall terraces 3-4 storeys': (7000, 9000),
#         'Small low terraces': (7000, 9000),
#         '2 storeys terraces with t rear extension': (7000, 9000),
        
#         # 3 bed semi (standard & large semi): 8-11k
#         'Large semi detached': (8000, 11000),
#         'Standard size semi detached': (8000, 11000),
        
#         # Large 4+ bed (detached properties): 10-13k
#         'Very large detached': (10000, 13000),
#         'Large detached': (10000, 13000),
#         'Standard size detached': (10000, 13000),
        
#         # Unknown - using mid-range
#         'all_unknown_typology': (7000, 11000)
#     })

#     # Double glazing cost ranges by typology
#     double_glazing_cost_ranges: Dict[str, Any] = field(default_factory=lambda: {
#         # Large houses (detached): 8-15k
#         'Very large detached': (8000, 15000),
#         'Large detached': (8000, 15000),
#         'Standard size detached': (8000, 15000),
#         'Tall terraces 3-4 storeys': (8000, 15000),
        
#         # Medium houses (semi-detached): 6-9k
#         'Large semi detached': (6000, 9000),
#         'Standard size semi detached': (6000, 9000),
        
#         # Terraced: 4-6k
#         'Small low terraces': (4000, 6000),
#         '2 storeys terraces with t rear extension': (4000, 6000),
        
#         # Flats - these will be scaled by number of flats
#         'Very tall point block flats': 'flat_based',
#         'Tall flats 6-15 storeys': 'flat_based',
#         'Medium height flats 5-6 storeys': 'flat_based',
#         '3-4 storey and smaller flats': 'flat_based',
        
#         # Unknown - using mid-range
#         'all_unknown_typology': (5000, 10000)
#     })

#     # Flat-based double glazing costs configuration
#     flat_based_double_glazing: Dict[str, float] = field(default_factory=lambda: {
#         'base_cost_per_flat': 4000,
#         'individual_flat_cost': 6000,
#         'economies_of_scale_threshold': 10,
#         'min_cost': 4000,
#         'max_cost': 80000
#     })
    
#     # Perimeter typology multipliers
#     perimeter_typology_multipliers: Dict[str, float] = field(default_factory=lambda: {
#         'Standard size semi detached': 0.75,
#         'Large semi detached': 0.75,
#         'Standard size detached': 1.0,
#         'Large detached': 1.0,
#         'Very large detached': 1.0,
#         'Small low terraces': 0.5,
#         'Tall terraces 3-4 storeys': 0.5,
#         'Medium height flats 5-6 storeys': 0.8,
#         '3-4 storey and smaller flats': 0.8,
#         'Tall flats 6-15 storeys': 0.9,
#         'Very tall point block flats': 0.95,
#         'all_unknown_typology': 0.8,
#     })

#     def sample_heat_pump_cost_triangular(self, typology: str, n_samples: int = 1) -> np.ndarray:
#         """
#         Generate Monte Carlo samples for heat pump costs using triangular distribution.
#         Mode is set as midpoint between min and max.
#         """
#         min_cost, max_cost = self.heat_pump_cost_ranges.get(typology, (7000, 11000))
#         mode_cost = (min_cost + max_cost) / 2
        
#         samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
#         return samples
    
#     def sample_double_glazing_cost_triangular(self, typology: str, n_samples: int = 1, 
#                                              num_flats: Optional[int] = None) -> np.ndarray:
#         """
#         Generate Monte Carlo samples for double glazing costs using triangular distribution.
#         Handles both house-based and flat-based pricing.
#         """
#         cost_range = self.double_glazing_cost_ranges.get(typology, (5000, 10000))
        
#         # Handle flat-based calculations
#         if cost_range == 'flat_based':
#             if num_flats is None:
#                 raise ValueError(f"num_flats required for flat typology: {typology}")
            
#             config = self.flat_based_double_glazing
            
#             if num_flats >= config['economies_of_scale_threshold']:
#                 base_cost = num_flats * config['base_cost_per_flat']
#                 min_cost = max(base_cost * 0.9, config['min_cost'])
#                 max_cost = min(base_cost * 1.1, config['max_cost'])
#             else:
#                 base_cost = num_flats * config['individual_flat_cost']
#                 min_cost = max(base_cost * 0.9, config['min_cost'])
#                 max_cost = min(base_cost * 1.1, config['max_cost'])
            
#             mode_cost = base_cost
#             samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
#         else:
#             # Standard house-based pricing
#             min_cost, max_cost = cost_range
#             mode_cost = (min_cost + max_cost) / 2
#             samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
        
#         return samples

#     def sample_intervention_cost(self, 
#                                 intervention: str,
#                                 building_chars: BuildingCharacteristics,
#                                 typology: str,
#                                 age_band: str,
#                                 region: str,
#                                 regional_multiplier: float,
#                                 age_multiplier: float,
#                                 complexity_multiplier: float,
#                                 n_samples: int = 1) -> np.ndarray:
#         """
#         Unified Monte Carlo sampling for all intervention types.
        
#         Parameters:
#         -----------
#         intervention : str
#             Name of the intervention
#         building_chars : BuildingCharacteristics
#             Building physical characteristics
#         typology : str
#             Building typology
#         age_band : str
#             Age band of the building
#         region : str
#             Regional code
#         regional_multiplier : float
#             Pre-calculated regional cost multiplier
#         age_multiplier : float
#             Pre-calculated age band multiplier
#         complexity_multiplier : float
#             Pre-calculated typology complexity multiplier
#         n_samples : int
#             Number of Monte Carlo samples to generate
            
#         Returns:
#         --------
#         np.ndarray : Array of sampled costs
#         """
        
#         # ===== AREA-BASED INTERVENTIONS =====
#         if intervention in self.area_based_interventions:
#             rules = self.area_based_interventions[intervention]
#             uncertainty = self.uncertainty_parameters.get(intervention)
            
#             if uncertainty is None:
#                 raise ValueError(f"No uncertainty parameters defined for {intervention}")
            
#             # Determine which area to use
#             area_type = rules['area_type']
#             if area_type == 'roof':
#                 area = building_chars.roof_area_estimate
#             elif area_type == 'wall':
#                 area = building_chars.external_wall_area_estimate
#             elif area_type == 'floor':
#                 area = building_chars.building_footprint_area
#             elif area_type == 'internal':
#                 area = building_chars.gross_internal_area
#             else:
#                 raise ValueError(f"Unknown area type: {area_type}")
            
#             # Sample cost per sqm from triangular distribution
#             cost_per_sqm_samples = np.random.triangular(
#                 uncertainty['min'],
#                 uncertainty['mode'],
#                 uncertainty['max'],
#                 n_samples
#             )
            
#             # Calculate base costs for all samples
#             base_costs = area * cost_per_sqm_samples
            
#             # Apply multipliers
#             final_costs = base_costs * age_multiplier * complexity_multiplier * regional_multiplier
            
#             # Apply caps (absolute limits)
#             cap_min, cap_max = uncertainty['cap']
#             final_costs = np.clip(final_costs, cap_min, cap_max)
            
#             return final_costs
        
#         # ===== HEAT PUMP (FIXED COST WITH TYPOLOGY VARIATION) =====
#         elif intervention == 'heat_pump_upgrade':
#             samples = self.sample_heat_pump_cost_triangular(typology, n_samples)
#             # Apply multipliers (heat pump costs already account for typology in base ranges)
#             final_costs = samples * age_multiplier * complexity_multiplier * regional_multiplier
#             return final_costs
        
#         # ===== DOUBLE GLAZING / WINDOW UPGRADES (FIXED COST WITH TYPOLOGY VARIATION) =====
#         elif intervention == 'window_upgrades':
#             samples = self.sample_double_glazing_cost_triangular(
#                 typology, 
#                 n_samples, 
#                 building_chars.flat_count
#             )
#             # Apply multipliers
#             final_costs = samples * age_multiplier * complexity_multiplier * regional_multiplier
#             return final_costs
        
#         # ===== FIXED COST INTERVENTIONS =====
#         elif intervention in self.fixed_cost_interventions:
#             base_cost = self.fixed_cost_interventions[intervention]
#             # Apply some uncertainty to fixed costs (±20%)
#             samples = np.random.triangular(
#                 base_cost * 0.8,
#                 base_cost,
#                 base_cost * 1.2,
#                 n_samples
#             )
#             # Apply multipliers
#             final_costs = samples * age_multiplier * complexity_multiplier * regional_multiplier
#             return final_costs
        
#         else:
#             raise ValueError(f"Unknown intervention: {intervention}. Cannot sample cost.")


# @dataclass
# class RetrofitConfig:
#     """Enhanced configuration with Monte Carlo cost sampling."""
    
#     typologies: List[str] = field(default_factory=lambda: [
#         'Medium height flats 5-6 storeys',
#         'Small low terraces',
#         '3-4 storey and smaller flats',
#         'Tall terraces 3-4 storeys',
#         'Large semi detached',
#         'Standard size detached',
#         'Standard size semi detached',
#         '2 storeys terraces with t rear extension',
#         'Semi type house in multiples',
#         'Tall flats 6-15 storeys',
#         'Large detached',
#         'Very tall point block flats',
#         'Very large detached',
#         'Planned balanced mixed estates',
#         'Linked and step linked premises',
#         'Domestic outbuilding',
#         'all_unknown_typology',
#     ])
    
#     age_bands: List[str] = field(default_factory=lambda: [
#         'Pre 1919', '1919-1944', '1945-1959', '1960-1979',
#         '1980-1989', '1990-1999', 'Post 1999'
#     ])
    
#     regional_multipliers: Dict[str, float] = field(default_factory=lambda: {
#         'LN': 1.25, 'SE': 1.15, 'SW': 1.05, 'NW': 0.95, 'NE': 0.85,
#         'YH': 0.90, 'WA': 0.95, 'WM': 0.98, 'EM': 0.95, 'EE': 1.08,
#     })
    
#     valid_regions: List[str] = field(default_factory=lambda: [
#         'LN', 'SE', 'SW', 'NW', 'NE', 'YH', 'WA', 'EM', 'EE', 'WM'
#     ])
    
#     scaling_rules: InterventionScalingRules = field(default_factory=InterventionScalingRules)
    
#     age_band_multipliers: Dict[str, float] = field(default_factory=lambda: {
#         'Post 1999': 0.90, '1990-1999': 0.95, '1980-1989': 1.0,
#         '1960-1979': 1.15, '1945-1959': 1.35, '1919-1944': 1.6, 'Pre 1919': 2.0
#     })
    
#     typology_complexity: Dict[str, float] = field(default_factory=lambda: {
#         'Very tall point block flats': 1.4,
#         'Tall flats 6-15 storeys': 1.2,
#         'Medium height flats 5-6 storeys': 1.1,
#         'Tall terraces 3-4 storeys': 1.1,
#     })
    
#     retrofit_packages: Dict[str, List[str]] = field(default_factory=lambda: {
#         'all': [
#             'cavity_wall_insulation', 'internal_wall_insulation', 'external_wall_insulation',
#             'loft_insulation', 'floor_insulation', 'draught_proofing', 'heating_controls',
#             'cylinder_jacket', 'boiler_service', 'window_upgrades', 'heat_pump_upgrade', 
#             'solar_pv', 'deep_retrofit_estimate'
#         ]
#     })
    
#     def validate_region(self, region: str) -> str:
#         """Validate region code."""
#         if region not in self.valid_regions:
#             raise ValueError(f"Invalid region '{region}'. Valid: {self.valid_regions}")
#         return region
    
#     def get_regional_multiplier(self, region: str) -> float:
#         """Get regional cost multiplier."""
#         return self.regional_multipliers[self.validate_region(region)]
    

#     def sample_intervention_cost_monte_carlo(self,
#                                             intervention: str,
#                                             building_chars: BuildingCharacteristics,
#                                             typology: str,
#                                             age_band: str,
#                                             region: str,
#                                             n_samples: int = 1000) -> np.ndarray:
#         """
#         Sample intervention costs using Monte Carlo simulation.
        
#         Parameters:
#         -----------
#         intervention : str
#             Name of the intervention
#         building_chars : BuildingCharacteristics
#             Building physical characteristics
#         typology : str
#             Building typology
#         age_band : str
#             Age band of the building
#         region : str
#             Regional code
#         n_samples : int
#             Number of Monte Carlo samples (default: 1000)
            
#         Returns:
#         --------
#         np.ndarray : Array of sampled costs
        
#         Example:
#         --------
#         >>> config = RetrofitConfig()
#         >>> building = BuildingCharacteristics(
#         ...     floor_count=3,
#         ...     gross_external_area=200,
#         ...     gross_internal_area=180,
#         ...     footprint_circumference=40,
#         ...     building_footprint_area=100
#         ... )
#         >>> samples = config.sample_intervention_cost_monte_carlo(
#         ...     'loft_insulation',
#         ...     building,
#         ...     'Standard size semi detached',
#         ...     '1960-1979',
#         ...     'LN',
#         ...     n_samples=1000
#         ... )
#         >>> print(f"Mean: £{samples.mean():.2f}")
#         >>> print(f"Median: £{np.median(samples):.2f}")
#         >>> print(f"95% CI: £{np.percentile(samples, 2.5):.2f} - £{np.percentile(samples, 97.5):.2f}")
#         """
#         if typology is None or typology == 'None':
#             return None
        
#         validated_region = self.validate_region(region)
        
#         # Get multipliers
#         age_mult = self.age_band_multipliers.get(age_band, 1.0)
#         complexity_mult = self.typology_complexity.get(typology, 1.0)
#         regional_mult = self.get_regional_multiplier(validated_region)
        
#         # Sample using unified method
#         samples = self.scaling_rules.sample_intervention_cost(
#             intervention=intervention,
#             building_chars=building_chars,
#             typology=typology,
#             age_band=age_band,
#             region=region,
#             regional_multiplier=regional_mult,
#             age_multiplier=age_mult,
#             complexity_multiplier=complexity_mult,
#             n_samples=n_samples
#         )
        
#         return samples
    
#     def sample_retrofit_package_monte_carlo(self,
#                                            interventions: List[str],
#                                            building_chars: BuildingCharacteristics,
#                                            typology: str,
#                                            age_band: str,
#                                            region: str,
#                                            n_samples: int = 1000) -> Dict[str, np.ndarray]:
#         """
#         Sample costs for a package of interventions.
#         Each intervention is sampled independently n_samples times.
        
#         Parameters:
#         -----------
#         interventions : List[str]
#             List of intervention names
#         building_chars : BuildingCharacteristics
#             Building physical characteristics
#         typology : str
#             Building typology
#         age_band : str
#             Age band of the building
#         region : str
#             Regional code
#         n_samples : int
#             Number of Monte Carlo samples per intervention
            
#         Returns:
#         --------
#         Dict[str, np.ndarray] : Dictionary mapping intervention names to their cost samples
        
#         Example:
#         --------
#         >>> results = config.sample_retrofit_package_monte_carlo(
#         ...     ['loft_insulation', 'cavity_wall_insulation', 'heat_pump_upgrade'],
#         ...     building,
#         ...     'Standard size semi detached',
#         ...     '1960-1979',
#         ...     'LN',
#         ...     n_samples=1000
#         ... )
#         >>> total_costs = sum(results.values())  # Element-wise sum across all interventions
#         >>> print(f"Total package mean: £{total_costs.mean():.2f}")
#         """
#         results = {}
        
#         for intervention in interventions:
#             samples = self.sample_intervention_cost_monte_carlo(
#                 intervention=intervention,
#                 building_chars=building_chars,
#                 typology=typology,
#                 age_band=age_band,
#                 region=region,
#                 n_samples=n_samples
#             )
#             results[intervention] = samples
        
#         return results


# def calculate_building_costs_df(df, region, col_mapping=None, config=None, 
#                                 n_samples=10, return_statistics=None):
#     """
#     Apply Monte Carlo building cost calculations to all rows in a DataFrame.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame with building characteristics
#     region : str
#         Regional code (e.g., 'LN', 'SE')
#     col_mapping : dict, optional
#         Mapping of field names to DataFrame columns
#     config : RetrofitConfig, optional
#         Configuration object. If None, creates default RetrofitConfig()
#     n_samples : int, optional
#         Number of Monte Carlo samples per intervention (default: 10)
#     return_statistics : list, optional
#         Statistics to return. Default: ['mean', 'p5', 'p50', 'p95']
#         Options: 'mean', 'median', 'std', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'
    
#     Returns:
#     --------
#     pd.DataFrame or dict
#         DataFrame with cost statistic columns OR dict with error message
        
#     Output columns (for each intervention):
#         - cost_{intervention}_mean
#         - cost_{intervention}_p5
#         - cost_{intervention}_p50
#         - cost_{intervention}_p95
    
#     Example:
#     --------
#     >>> result_df = calculate_building_costs_df(
#     ...     df=buildings_df,
#     ...     region='LN',
#     ...     n_samples=100,
#     ...     return_statistics=['mean', 'p5', 'p95']
#     ... )
#     """
#     import pandas as pd
#     import numpy as np
#     import logging
    
#     try:
#         if df is None or df.empty:
#             return {'error': 'DataFrame is None or empty'}
        
#         if not region:
#             return {'error': 'Region parameter is required'}
        
#         if config is None:
#             config = RetrofitConfig()
        
#         if return_statistics is None:
#             return_statistics = ['mean', 'p5', 'p50', 'p95']
        
#         # Validate statistics requested
#         valid_statistics = ['mean', 'median', 'std', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']
#         invalid_stats = [s for s in return_statistics if s not in valid_statistics]
#         if invalid_stats:
#             return {'error': f'Invalid statistics requested: {invalid_stats}. Valid: {valid_statistics}'}
        
#         # Mapping definition
#         default_mapping = {
#             'floor_count': 'total_fl_area_avg',
#             'gross_external_area': 'total_fl_area_avg',
#             'gross_internal_area': 'scaled_fl_area',
#             'footprint_circumference': 'perimeter_length',
#             'flat_count': 'est_num_flats',
#             'building_type': 'premise_type',
#             'age_band': 'premise_age',
#             'building_footprint_area': 'premise_area',
#         }
        
#         if col_mapping is None:
#             col_mapping = default_mapping
#         else:
#             for key, default_val in default_mapping.items():
#                 if key not in col_mapping:
#                     col_mapping[key] = default_val
        
#         # Validate columns exist
#         missing_columns = [
#             f"{field} -> {col_name}" 
#             for field, col_name in col_mapping.items() 
#             if col_name not in df.columns
#         ]
        
#         if missing_columns:
#             return {'error': f'Required columns not found: {", ".join(missing_columns)}'}
        
#         result_df = df.copy()
#         result_df = result_df[result_df[col_mapping['building_type']] != 'Domestic outbuilding']
        
#         # Get list of all possible interventions from the config 
#         all_interventions = config.retrofit_packages['all']
        
#         def calculate_row_costs(row):
#             """Calculate Monte Carlo cost statistics for all interventions for one building."""
            
#             # Data validation
#             required_cols = ['floor_count', 'gross_external_area', 'gross_internal_area',
#                            'footprint_circumference', 'building_type', 'age_band', 'building_footprint_area']
            
#             missing_cols = [col for col in required_cols if col_mapping[col] not in row.index]
#             if missing_cols:
#                 raise ValueError(f'Missing columns: {missing_cols}')
            
#             # Convert and validate
#             floor_count = int(row[col_mapping['floor_count']])
#             gross_external_area = float(row[col_mapping['gross_external_area']])
#             gross_internal_area = float(row[col_mapping['gross_internal_area']])
#             footprint_circumference = float(row[col_mapping['footprint_circumference']])
#             building_footprint_area = float(row[col_mapping['building_footprint_area']])
            
#             # Use max(1) for flat count to ensure valid input to cost calcs
#             raw_flat_count = row.get(col_mapping['flat_count'])
#             flat_count = int(raw_flat_count) if pd.notna(raw_flat_count) and raw_flat_count > 0 else 1
            
#             if any([
#                 pd.isna(floor_count) or floor_count <= 0,
#                 pd.isna(gross_external_area) or gross_external_area <= 0,
#                 pd.isna(gross_internal_area) or gross_internal_area <= 0,
#                 pd.isna(footprint_circumference) or footprint_circumference <= 0,
#                 pd.isna(building_footprint_area) or building_footprint_area <= 0
#             ]):
#                 raise ValueError('Invalid building characteristics')
            
#             building_chars = BuildingCharacteristics(
#                 floor_count=floor_count,
#                 gross_external_area=gross_external_area,
#                 gross_internal_area=gross_internal_area,
#                 footprint_circumference=footprint_circumference,
#                 flat_count=flat_count,
#                 building_footprint_area=building_footprint_area
#             )
            
#             # Calculate Monte Carlo cost statistics for ALL interventions
#             cost_stats = {}
#             typology = row[col_mapping['building_type']]
#             age_band = row[col_mapping['age_band']]

#             for intervention in all_interventions:
#                 try:
#                     # Get Monte Carlo samples
#                     samples = config.sample_intervention_cost_monte_carlo(
#                         intervention=intervention,
#                         building_chars=building_chars,
#                         typology=typology,
#                         age_band=age_band,
#                         region=region,
#                         n_samples=n_samples
#                     )
                    
#                     # Calculate requested statistics
#                     for stat in return_statistics:
#                         col_name = f'{intervention}_{stat}'
                        
#                         if stat == 'mean':
#                             cost_stats[col_name] = samples.mean()
#                         elif stat == 'median' or stat == 'p50':
#                             cost_stats[col_name] = np.median(samples)
#                         elif stat == 'std':
#                             cost_stats[col_name] = samples.std()
#                         elif stat.startswith('p'):
#                             # Extract percentile number (e.g., 'p95' -> 95)
#                             percentile = int(stat[1:])
#                             cost_stats[col_name] = np.percentile(samples, percentile)
                    
#                 except ValueError as ve:
#                     # This happens if an intervention is not configured or invalid
#                     logging.warning(f"Cost calculation skipped for {intervention}: {ve}")
#                     for stat in return_statistics:
#                         col_name = f'{intervention}_{stat}'
#                         cost_stats[col_name] = np.nan
#                 except Exception as e:
#                     logging.error(f"Unexpected error calculating {intervention}: {e}")
#                     for stat in return_statistics:
#                         col_name = f'{intervention}_{stat}'
#                         cost_stats[col_name] = np.nan
            
#             return pd.Series(cost_stats)
        
#         # Apply cost calculations to all rows
#         cost_results = result_df.apply(calculate_row_costs, axis=1)
        
#         # Add cost columns to result DataFrame
#         for col in cost_results.columns:
#             result_df[f'cost_{col}'] = cost_results[col]
        
#         return result_df
        
#     except Exception as e:
#         return {'error': f'Unexpected error: {str(e)}'}
        

# def calc_retrofit_scenarios_with_uncertainty(
#     pc,
#     proc_building_df_for_pc,
#     region,
#     energy_column='total_gas_derived',
#     n_monte_carlo=100,
#     random_seed=42,
#     apply_performance_gap=True
# ):
#     """
#     Calculate retrofit scenarios with full uncertainty analysis using Monte Carlo simulation.
    
#     Args:
#         pc: postcode 
#         proc_building_df_for_pc: DataFrame with building data for one postcode
#         region: Region code for cost calculations
#         energy_column: Column name containing annual energy consumption (kWh)
#         n_monte_carlo: Number of Monte Carlo iterations (default 10000)
#         random_seed: Random seed for reproducibility
#         apply_performance_gap: Whether to apply performance gap factors
        
#     Returns:
#         dict: Summary results with scenario costs, energy savings, and uncertainty metrics
#               OR dict with 'error' key if failed
#     """
    
#     if proc_building_df_for_pc is None or proc_building_df_for_pc.empty:
#         return {'error': 'Input DataFrame is empty or None'}
    
#     # Check for energy data
#     if energy_column not in proc_building_df_for_pc.columns:
#         return {'error': f'Energy column "{energy_column}" not found in DataFrame'}
    
#     try:
#         # Calculate estimated flats
#         # Note: 'premise_floor_count' is assumed to be the correct column for floor_count
#         proc_building_df_for_pc['est_num_flats'] = proc_building_df_for_pc.apply(
#             lambda row: calculate_estimated_flats_per_building(
#                 building_footprint_area=row['premise_area'],
#                 typology_col=row['premise_type'],
#                 floor_count=row['premise_floor_count']
#             ),
#             axis=1
#         )
        
#         # Calculate building costs
#         df_with_costs = calculate_building_costs_df(proc_building_df_for_pc, region)
        
#         if 'error' in df_with_costs:
#             return df_with_costs
        
#         total_flats_count = df_with_costs['est_num_flats'].sum()
        
#         # Initialize scenario generator (assuming it has the necessary cost_per_kwh param)
#         scenario_generator = RetrofitScenarioGenerator()
        

#         # Calculate scenarios with uncertainty (uses the deterministic costs from df_with_costs)
#         scenario_results = scenario_generator.calculate_scenario_costs_and_savings(
#             df=df_with_costs,
#             energy_column=energy_column,
#             age_column='premise_age',
#             cost_prefix='cost_',
#             n_monte_carlo=n_monte_carlo,
#             use_probabilistic_sampling=True,
#             apply_performance_gap=apply_performance_gap,
#             random_state=random_seed
#         )
        
#         if scenario_results.empty:
#             return {'error': 'Scenario calculation returned empty DataFrame'}
        
#         # A & B: Use the proper aggregation method from RetrofitScenarioGenerator
#         # This correctly calculates total costs, total savings, and proper std aggregation.
#         aggregated = scenario_generator.aggregate_estate_results(
#             scenario_results,
#             energy_column=energy_column
#         )
        
#         # Format results for output
#         results = {
#             'postcode': pc ,
#             'total_flat_count': float(total_flats_count),
#             'n_buildings': int(aggregated['n_buildings'].iloc[0]) if not aggregated.empty else 0,
#             'scenarios': {}
#         }
        
#         for scenario_name, row in aggregated.iterrows():
#             results['scenarios'][scenario_name] = {
#                 'total_cost': float(row['total_investment']),
#                 'mean_savings_kwh': float(row['mean_savings_kwh']),
#                 'median_savings_kwh': float(row['median_savings_kwh']),
#                 'p10_savings_kwh': float(row['p10_savings_kwh']),
#                 'p90_savings_kwh': float(row['p90_savings_kwh']),
#                 'mean_savings_percent': float(row['mean_savings_percent']),
#                 'p10_savings_percent': float(row['p10_savings_percent']),
#                 'p90_savings_percent': float(row['p90_savings_percent']),
#                 'mean_annual_savings_gbp': float(row['mean_annual_savings_gbp']),
#                 'simple_payback_years': float(row['mean_payback_years']),
#                 'conservative_payback_years1': float(row['conservative_payback_years']),
#                 'conservative_payback_years': float(row['mean_payback_years']) * (row['mean_savings_kwh'] / row['p10_savings_kwh']) if row['p10_savings_kwh'] > 0 else np.inf,
#                 'optimistic_payback_years': float(row['mean_payback_years']) * (row['mean_savings_kwh'] / row['p90_savings_kwh']) if row['p90_savings_kwh'] > 0 else np.inf,
#                 'savings_80pct_ci_kwh': float(row['savings_80pct_ci_kwh']),
#                 'std_savings_kwh_agg': float(row.get('std_savings_kwh', np.nan)) # Ensure std is captured
#             }
        
#         return results
        
#     except Exception as e:
#         # Logging the full traceback for unexpected errors is recommended here
#         logging.exception(f"Unexpected error in scenario calculation for region {region}")
#         return {'error': f'Unexpected error: {str(e)}'}


    # # Flatten results for easier consumption
    # flat_results = {
    #     'postcode': results['postcode'],
    #     'total_flat_count': results['total_flat_count'],
    #     'n_buildings': results['n_buildings'],
    #     'error': None
    # }
    
    # # Add scenario costs
    # for scenario_name, scenario_data in results['scenarios'].items():
    #     # Added '_cost' suffix for consistency with the initial error_dict
    #     flat_results[f'{scenario_name}_cost'] = scenario_data['total_cost']
    #     flat_results[f'{scenario_name}_mean_savings_kwh'] = scenario_data['mean_savings_kwh']
    #     flat_results[f'{scenario_name}_p10_savings_kwh'] = scenario_data['p10_savings_kwh']
    #     flat_results[f'{scenario_name}_p90_savings_kwh'] = scenario_data['p90_savings_kwh']
    #     flat_results[f'{scenario_name}_mean_savings_percent'] = scenario_data['mean_savings_percent']
    #     flat_results[f'{scenario_name}_payback_years'] = scenario_data['simple_payback_years']
    #     flat_results[f'{scenario_name}_conservative_payback_years'] = scenario_data['conservative_payback_years']
    #     flat_results[f'{scenario_name}_optimistic_payback_years'] = scenario_data['optimistic_payback_years']
    #     flat_results[f'{scenario_name}_std_savings_kwh_agg'] = scenario_data['std_savings_kwh_agg']
    
    # return flat_results


# # LEGACY FUNCTION - kept for backwards compatibility
# def calc_retrofit_scenarios_for_postcode_vars(proc_building_df_for_pc, region):
#     """
#     Legacy function - calculates costs only (no energy savings).
#     Use calc_retrofit_scenarios_with_uncertainty() for full analysis.
#     """
#     if proc_building_df_for_pc is None or proc_building_df_for_pc.empty:
#         return {'error': 'Input DataFrame is empty or None'}
    
#     try:
#         proc_building_df_for_pc['est_num_flats'] = proc_building_df_for_pc.apply(
#             lambda row: calculate_estimated_flats_per_building(
#                 building_footprint_area=row['premise_area'],
#                 typology_col=row['premise_type'],
#                 floor_count=row['premise_floor_count']
#             ),
#             axis=1
#         )
        
#         # This will return a df with cost_intervention columns
#         df_with_costs = calculate_building_costs_df(proc_building_df_for_pc, region)
        
#         if 'error' in df_with_costs:
#             return df_with_costs
            
#         total_flats_count = df_with_costs['est_num_flats'].sum()

#         scenario_generator = RetrofitScenarioGenerator()
        
#         # This assumes calculate_scenario_costs still exists in RetrofitScenarioGenerator 
#         # to sum the required cost columns for each scenario.
#         scenario_costs_df = scenario_generator.calculate_scenario_costs(
#             df_with_costs,
#             age_column='premise_age',
#             cost_prefix='cost_'
#         )
        
#         if scenario_costs_df.empty:
#             return {'error': "Scenario costs calculation returned empty DataFrame"}
        
#         # Sum the total cost per scenario across the buildings
#         result = scenario_costs_df.groupby(['scenario_name'])['scenario_cost'].sum().reset_index()
        
#         if result.empty:
#             return {'error': "No valid scenarios found"}
        
#         new_row = {'scenario_name': 'total_flat_count', 'scenario_cost': total_flats_count}
#         result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        
#         return result
        
#     except Exception as e:
#         return {'error': f'Unexpected error: {str(e)}'}