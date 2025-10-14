from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np 
from scipy import stats
import logging 

# Import BuildingCharacteristics from cost module
# Add logger at the top of your module
logger = logging.getLogger(__name__)

from .PreProcessRetrofit import vectorized_process_buildings 





@dataclass
class RetrofitScenarioGenerator:
    """
    Generates retrofit scenarios with age-appropriate wall insulation types.
    Includes Monte Carlo uncertainty analysis based on observed performance ranges.
    """

    def process_dataframe_scenarios(self,
                               df: pd.DataFrame,
                              scenarios: list, 
                            #    typ_config,   # RetrofitConfig instance
                               model_class,  # RetrofitCost instance
                               region: str,
                               random_seed: Optional[int] = None,
                               col_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Process scenarios for all buildings in a DataFrame with efficient cost sampling.
        
        Flow:
        1. Determine wall type for building
        2. Generate scenarios (includes sampling existing insulation)
        3. Get list of needed interventions per scenario
        4. Run Monte Carlo on costs ONLY for needed interventions
        5. Run Monte Carlo on energy savings
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with building data
        typ_config: RetrofitConfig object 
        model_class : RetrofitModel for single scenario 
            
        region : str
            Region code (e.g., 'LN', 'SE')
        n_samples : int
            Number of Monte Carlo samples per building
        random_seed : int, optional
            Random seed for reproducibility
        col_mapping : Dict[str, str], optional
            Column name mapping
            
        Required columns in df:
        - age_band
        - current_energy_kwh
        - Building characteristics for cost calculation
        
        Returns:
        --------
        pd.DataFrame : Original DataFrame with added scenario columns
        """
        
        typ_config = model_class.retrofit_config
        # Default column mapping
        default_mapping = {
            'age_band': 'premise_age_bucketed',
            'current_energy_kwh': 'total_gas_derived',
            'floor_count': 'total_fl_area_avg',
            'gross_external_area': 'total_fl_area_avg',
            'gross_internal_area': 'scaled_fl_area',
            'footprint_circumference': 'perimeter_length',
            'flat_count': 'est_num_flats',
            'building_type': 'premise_type',
            'building_footprint_area': 'premise_area',
            'avg_gas_decile': 'avg_gas_decile'
        }
        
        if col_mapping:
            default_mapping.update(col_mapping)
        col_mapping = default_mapping
        
        result_df = df.copy()

        df_typ = vectorized_process_buildings(
                    result_df=result_df,
                    col_mapping=col_mapping , 
                    config=typ_config, 
                    random_seed=random_seed
                )
        
 
        res_sc = [df] 
        
        prob_external = typ_config.existing_intervention_probs['external_wall_occurence']

        for scenario in scenarios:
            
            results = model_class.calculate_building_costs_df_updated(
                df= df_typ, 
                region=region,
                scenario=scenario ,
                prob_external = prob_external , 
                ) 

            logger.debug(f"Type of results: {type(results)}")
            logger.debug(f"Results content: {results}")

            if isinstance(results, dict):
                if 'error' in results:
                    logger.warning(f"Warning: {results['error']}")
                    continue  # Skip this scenario
                logger.debug(f"Results keys: {results.keys()}")
                logger.debug(f"Sample values: {[(k, type(v), v) for k, v in list(results.items())[:3]]}")
                results = pd.DataFrame(results, index=[0])  # Add index=[0] fix here
            else:
                res_sc.append(results)
 
            
        costs_result = pd.concat(res_sc, axis=1)                     
       
        return costs_result 
            
                     



    
     
    # # Base scenario templates
    # base_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
              
    #     'loft_installation': {
    #         'name': 'loft',
    #         'description': 'Low-cost, high-impact maintenance interventions',
    #         'interventions': [
    #             'Loft_single',
    #         ],
    #         'includes_wall_insulation': False,
    #         'installation_approach': 'simultaneous'
    #     },
    #     'wall_installation': {
    #         'name': 'walls',
    #         'description': 'walls',
    #         'interventions': [
    #             'WALL_INSULATION',
    #         ],
    #         'includes_wall_insulation': True,
    #         'installation_approach': 'simultaneous'
    #     },

    #     'window_upgrades': {
    #         'name': 'window_upgrades',
    #         'description': 'window_upgrades - double glazing ',
    #         'interventions': [
    #             'window_upgrades',
    #         ],
    #         'includes_wall_insulation': False,
    #         'installation_approach': 'simultaneous'
    #     },

    #     'floor_insulation': {
    #         'name': 'floor_insulation',
    #         'description': 'floor_insulation',
    #         'interventions': [
    #             'floor_insulation',
    #         ],
    #         'includes_wall_insulation': False,
    #         'installation_approach': 'simultaneous'
    #     },
       
    #     'scenario2': {
    #         'name': 'Scenario 2: envelope only (evans et al)',
    #         'description': 'Focus on insulation with easy wins',
    #         'interventions': [
    #             'WALL_INSULATION',
    #             'loft_insulation',
    #             'double_glazing',
    #         ],
    #         'includes_wall_insulation': True,
    #         'installation_approach': 'simultaneous'
    #     },

    #     'scenario3': {
    #         'name': 'envelope and air source heat pump',
    #         'description': 'Scenario 3 envelope and air source heat pump',
    #         'interventions': [
    #             'WALL_INSULATION',
    #             'loft_insulation',
    #             'double_glazing',
    #             'heat_pump_upgrade',
    #         ],
    #         'includes_wall_insulation': True,
    #         'installation_approach': 'simultaneous'
    #     },

    #     'scenario5': {
    #         'name': 'envelope and air heat pump, solar',
    #         'description': 'scenario 5: envelope, heat pump and solar',
    #         'interventions': [
    #             'WALL_INSULATION',
    #             'loft_insulation',
    #             'double_glazing',
    #             'heat_pump_upgrade',
    #             'solar_pv', 
    #         ],
    #         'includes_wall_insulation': True,
    #         'installation_approach': 'simultaneous'
    #     },
    
    #     'deep_retrofit_grouped': {
    #         'name': 'Deep Retrofit (Net Zero Ready)',
    #         'description': 'Maximum intervention package',
    #         'interventions': [
    #             'deep_retrofit_estimate',
    #         ],
    #         'includes_wall_insulation': True,
    #         'installation_approach': 'simultaneous'
    #     }
    # })
    