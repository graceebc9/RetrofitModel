from .BuildingCharacteristics import BuildingCharacteristics

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional, Tuple, List
import logging 

logger = logging.getLogger(__name__)

from .RetrofitPackages import get_intervention_list 

# --- Configuration (No changes needed here) ---
@dataclass
class InterventionConfig:
    """Unified configuration for a single intervention."""
    area_type: Literal['roof', 'wall', 'floor', 'internal', 'fixed', 'typology_based']
    cost_min: Optional[float] = None
    cost_mode: Optional[float] = None
    cost_max: Optional[float] = None
    cost_by_typology: Optional[Dict[str, Tuple[float, float]]] = None
    cap_min: Optional[float] = None
    cap_max: Optional[float] = None
    distribution: str = 'triangular'
    confidence: str = 'medium'
    notes: str = ''

# Configuration dictionary for individual interventions
INTERVENTION_CONFIGS: Dict[str, InterventionConfig] = {
    'loft_percentile': InterventionConfig(
        area_type='roof', cost_min=10, cost_mode=20, cost_max=30,
        cap_min=500, cap_max=2000
    ),
    'cavity_wall_percentile': InterventionConfig(
        area_type='wall', cost_min=10, cost_mode=20, cost_max=30,
        cap_min=500, cap_max=5000
    ),
    'solid_wall_internal_percentile': InterventionConfig(
        area_type='wall', cost_min=55, cost_mode=95, cost_max=140,
        cap_min=6000, cap_max=9000
    ),
     'solid_wall_external_percentile': InterventionConfig(
        area_type='wall',
        cost_min=70,
        cost_mode=115,
        cost_max=160,
        cap_min=7100,
        cap_max=15000,
    ),

   # Fixed-cost interventions with typology variations
    'heat_pump_percentile': InterventionConfig(
        area_type='typology_based',
        cost_by_typology={
            # Flats: 10-15k
            'Very tall point block flats': (10000, 15000),
            'Tall flats 6-15 storeys': (10000, 15000),
            'Medium height flats 5-6 storeys': (10000, 15000),
            '3-4 storey and smaller flats': (10000, 15000),
            
            # Small terraces: 7-9k
            'Small low terraces': (7000, 9000),
            '2 storeys terraces with t rear extension': (7000, 9000),
            
            # Semi-detached: 8-15k
            'Large semi detached': (8000, 15000),
            'Standard size semi detached': (8000, 15000),
            
            # Larger properties: 12-20k
            'Tall terraces 3-4 storeys': (12000, 20000),
            'Very large detached': (12000, 20000),
            'Large detached': (12000, 20000),
            'Standard size detached': (12000, 20000),
            
            # Unknown - using mid-range
            'all_unknown_typology': (10000, 15000)
        },
            cap_min=6000,
        cap_max=25000,
        confidence='medium',
        notes='Cost varies significantly by building typology and size'
    ),
}


# --- Cost Estimator Class with New Method ---
class CostEstimator:
    """Estimates intervention costs using Monte Carlo simulation."""
    
    def __init__(self, configs: Dict[str, InterventionConfig] = None):
        self.configs = configs or INTERVENTION_CONFIGS

    # --- NEW: Method to combine costs from a list of interventions ---
    def sample_cost_for_package(self,
                                intervention: str,
                                building_chars: BuildingCharacteristics,
                                **kwargs) -> np.ndarray:
        """
        Calculates the combined cost for a list of interventions by summing their individual costs.

        Args:
            interventions (List[str]): A list of intervention names to combine.
            building_chars (BuildingCharacteristics): The building's characteristics.
            **kwargs: Other parameters required by sample_intervention_cost (typology, n_samples, etc.).

        Returns:
            np.ndarray: An array of sampled total costs for the package.
        """ 
        try:
            interventions_list = get_intervention_list(kwargs.get('wall_type'), intervention)
            logger.debug('Intervention list found')
        except:
            logger.debug('Intervention not a join intervention, maually making list')
            interventions_list=[intervention]
        n_samples = kwargs.get('n_samples', 1)
        total_costs = np.zeros(n_samples)
        
        for intervention_name in interventions_list:
            # Call the single intervention method for each item in the list
            component_costs = self.sample_intervention_cost(
                intervention=intervention_name,
                building_chars=building_chars,
                **kwargs
            )
            total_costs += component_costs
        
        return total_costs

    # --- (Helper and single-intervention methods remain the same) ---
    def get_area_for_intervention(self, intervention: str, building_chars: BuildingCharacteristics) -> float:
        config = self.configs.get(intervention)
        if not config:
            raise ValueError(f"Unknown intervention: {intervention}")
        area_type = config.area_type
        if area_type == 'roof': return building_chars.roof_area_estimate
        elif area_type == 'wall': return building_chars.external_wall_area_estimate
        elif area_type in ['fixed', 'typology_based']: return 1.0
        else: raise ValueError(f"Unknown area type: {area_type}")

    def sample_intervention_cost(self,
                                intervention: str,
                                building_chars: BuildingCharacteristics,
                                **kwargs) -> np.ndarray:
        """Samples cost for a SINGLE intervention. This is called by the package method."""
        config = self.configs.get(intervention)
        if not config:
            raise ValueError(f"Unknown intervention: {intervention}")
        
        n_samples = kwargs.get('n_samples', 1)
        
        if config.area_type == 'typology_based':
            # ... (logic for typology-based costs as before) ...
            typology = kwargs.get('typology', 'all_unknown_typology')
            default_range = config.cost_by_typology.get('all_unknown_typology')
            cost_range = config.cost_by_typology.get(typology, default_range)
            min_cost, max_cost = cost_range
            mode_cost = (min_cost + max_cost) / 2
            final_costs = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
        else:
            # ... (logic for area-based/fixed costs as before) ...
            area = self.get_area_for_intervention(intervention, building_chars)
            cost_per_unit = np.random.triangular(config.cost_min, config.cost_mode, config.cost_max, n_samples)
            final_costs = area * cost_per_unit
        
        if config.cap_min is not None and config.cap_max is not None:
            final_costs = np.clip(final_costs, config.cap_min, config.cap_max)
            
        return final_costs

# # --- Example Usage ---
# if __name__ == '__main__':
#     estimator = CostEstimator()

#     sample_building = BuildingCharacteristics(roof_area_estimate=80, external_wall_area_estimate=120)
    
#     # Define simulation parameters
#     sim_params = {
#         'building_chars': sample_building,
#         'typology': 'Semi-detached',
#         'n_samples': 5
#     }

#     # Define the package of interventions you want to combine
#     my_package = ['loft_insulation', 'cavity_wall_insulation']

#     print(f"--- Dynamically combining costs for: {my_package} ---")

#     # Call the new method with your list
#     combined_costs = estimator.sample_cost_for_package(my_package, **sim_params)

#     print(f"Sampled Combined Costs (£): {np.round(combined_costs, 2)}")
#     print(f"Average combined cost: £{combined_costs.mean():,.2f}\n")

#     # You can easily combine a different set of interventions
#     my_second_package = ['solid_wall_internal_insulation', 'heat_pump']
#     print(f"--- Dynamically combining costs for: {my_second_package} ---")
#     second_combo_costs = estimator.sample_cost_for_package(my_second_package, **sim_params)
#     print(f"Sampled Combined Costs (£): {np.round(second_combo_costs, 2)}")
#     print(f"Average combined cost: £{second_combo_costs.mean():,.2f}\n")
    


    # from dataclasses import dataclass, field
# from typing import Dict, Any, Literal, Optional, Tuple
# import numpy as np

# @dataclass
# class InterventionConfig:
#     """Unified configuration for a single intervention"""
#     # Core intervention properties
#     area_type: Literal['roof', 'wall', 'floor', 'internal', 'fixed', 'typology_based']
    
#     # Cost distribution parameters (£/sqm for area-based, £ total for fixed)
#     cost_min: Optional[float] = None
#     cost_mode: Optional[float] = None  # This is the base/deterministic value
#     cost_max: Optional[float] = None
    
#     # Typology-based cost ranges (for interventions like heat pumps)
#     cost_by_typology: Optional[Dict[str, Tuple[float, float]]] = None
    
#     # Absolute cost caps (after multipliers applied)
#     cap_min: Optional[float] = None
#     cap_max: Optional[float] = None
    
#     # Metadata
#     distribution: str = 'triangular'
#     confidence: str = 'medium'
#     notes: str = ''


# # Single unified configuration dictionary
# INTERVENTION_CONFIGS: Dict[str, InterventionConfig] = {
#     # 'loft_insulation': InterventionConfig(
#     #     area_type='roof',
#     #     cost_min=10,
#     #     cost_mode=20,
#     #     cost_max=30,
#     #     cap_min=500,
#     #     cap_max=2000,
#     #     confidence='medium',
#     #     notes='Cost per sqm of roof area'
#     # ),
    
#     'loft_percentile': InterventionConfig(
#         area_type='roof',
#         cost_min=10,
#         cost_mode=20,
#         cost_max=30,
#         cap_min=500,
#         cap_max=2000,
#         confidence='medium',
#         notes='Cost per sqm of roof area'
#     ),
    
#     'cavity_wall_percentile': InterventionConfig(
#         area_type='wall',
#         cost_min=10,
#         cost_mode=20,
#         cost_max=30,
#         cap_min=500,
#         cap_max=5000,
#         confidence='medium',
#         notes='Cost per sqm of external wall area'
#     ),
    
#     'solid_wall_internal_percentile': InterventionConfig(
#         area_type='wall',
#         cost_min=55,
#         cost_mode=95,
#         cost_max=140,
#         cap_min=6000,
#         cap_max=9000,
#         confidence='medium',
#         notes='Cost per sqm of external wall area'
#     ),
    
#     'solid_wall_external_percentile': InterventionConfig(
#         area_type='wall',
#         cost_min=70,
#         cost_mode=115,
#         cost_max=160,
#         cap_min=7100,
#         cap_max=15000,
#         confidence='medium',
#         notes='Cost per sqm of external wall area'
#     ),
    
#     'loft_and_wall_installation': None 
#     ,

#     # Fixed-cost interventions with typology variations
#     'heat_pump_percentile': InterventionConfig(
#         area_type='typology_based',
#         cost_by_typology={
#             # Flats: 10-15k
#             'Very tall point block flats': (10000, 15000),
#             'Tall flats 6-15 storeys': (10000, 15000),
#             'Medium height flats 5-6 storeys': (10000, 15000),
#             '3-4 storey and smaller flats': (10000, 15000),
            
#             # Small terraces: 7-9k
#             'Small low terraces': (7000, 9000),
#             '2 storeys terraces with t rear extension': (7000, 9000),
            
#             # Semi-detached: 8-15k
#             'Large semi detached': (8000, 15000),
#             'Standard size semi detached': (8000, 15000),
            
#             # Larger properties: 12-20k
#             'Tall terraces 3-4 storeys': (12000, 20000),
#             'Very large detached': (12000, 20000),
#             'Large detached': (12000, 20000),
#             'Standard size detached': (12000, 20000),
            
#             # Unknown - using mid-range
#             'all_unknown_typology': (10000, 15000)
#         },
#         cap_min=6000,
#         cap_max=25000,
#         confidence='medium',
#         notes='Cost varies significantly by building typology and size'
#     ),
    
#     'double_glazing': InterventionConfig(
#         area_type='fixed',
#         cost_min=3000,  # Example values - adjust based on your glazing logic
#         cost_mode=5000,
#         cost_max=8000,
#         cap_min=2000,
#         cap_max=15000,
#         confidence='medium',
#         notes='Fixed cost varies by typology and flat count'
#     ),
# }


# class CostEstimator:
#     """Refactored cost estimator with unified config"""
    
#     def __init__(self, configs: Dict[str, InterventionConfig] = None):
#         self.configs = configs or INTERVENTION_CONFIGS
    
#     def get_area_for_intervention(self, 
#                                    intervention: str,
#                                    building_chars) -> float:
#         """Get the relevant area for an intervention"""
#         config = self.configs.get(intervention)
#         if not config:
#             raise ValueError(f"Unknown intervention: {intervention}")
        
#         area_type = config.area_type
        
#         if area_type == 'roof':
#             return building_chars.roof_area_estimate
#         elif area_type == 'wall':
#             return building_chars.external_wall_area_estimate
#         elif area_type == 'floor':
#             return building_chars.building_footprint_area
#         elif area_type == 'internal':
#             return building_chars.gross_internal_area
#         elif area_type == 'fixed':
#             return 1.0  # For fixed costs, area multiplier is 1
#         else:
#             raise ValueError(f"Unknown area type: {area_type}")
    
#     def sample_intervention_cost(self,
#                                 intervention: str,
#                                 building_chars,
#                                 typology: str,
#                                 age_band: str,
#                                 region: str,
#                                 regional_multiplier: float,
#                                 age_multiplier: float,
#                                 complexity_multiplier: float,
#                                 n_samples: int = 1) -> np.ndarray:
#         """
#         Unified Monte Carlo sampling for all intervention types.
        
#         Returns:
#             np.ndarray : Array of sampled costs
#         """
#         config = self.configs.get(intervention)
#         if not config:
#             raise ValueError(f"Unknown intervention: {intervention}")
        
#         # Handle typology-based interventions (e.g., heat pumps)
#         if config.area_type == 'typology_based':
#             if config.cost_by_typology is None:
#                 raise ValueError(f"Typology-based intervention {intervention} missing cost_by_typology")
            
#             # Get the cost range for this typology
#             cost_range = config.cost_by_typology.get(
#                 typology,
#                 config.cost_by_typology.get('all_unknown_typology', (10000, 15000))
#             )
            
#             min_cost, max_cost = cost_range
#             mode_cost = (min_cost + max_cost) / 2  # Use midpoint as mode
            
#             # Sample from triangular distribution
#             cost_samples = np.random.triangular(
#                 min_cost,
#                 mode_cost,
#                 max_cost,
#                 n_samples
#             )
            
#             # Apply multipliers (heat pumps typically use age and regional only)
#             final_costs = cost_samples * age_multiplier * regional_multiplier
            
#             # Apply caps if specified
#             if config.cap_min is not None and config.cap_max is not None:
#                 final_costs = np.clip(final_costs, config.cap_min, config.cap_max)
            
#             return final_costs
        
#         # Handle area-based and fixed-cost interventions
#         area = self.get_area_for_intervention(intervention, building_chars)
        
#         # Sample cost per unit from triangular distribution
#         cost_per_unit_samples = np.random.triangular(
#             config.cost_min,
#             config.cost_mode,
#             config.cost_max,
#             n_samples
#         )
        
#         # Calculate base costs
#         base_costs = area * cost_per_unit_samples
        
#         # Apply multipliers
#         if config.area_type == 'fixed':
#             # Fixed costs might use different multipliers
#             final_costs = base_costs * age_multiplier * regional_multiplier
#         else:
#             # Area-based costs use all multipliers
#             final_costs = base_costs * age_multiplier * complexity_multiplier * regional_multiplier
        
#         # Apply caps (absolute limits)
#         final_costs = np.clip(final_costs, config.cap_min, config.cap_max)
        
#         return final_costs
    