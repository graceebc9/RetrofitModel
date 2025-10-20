from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional, Tuple
import numpy as np

@dataclass
class InterventionConfig:
    """Unified configuration for a single intervention"""
    # Core intervention properties
    area_type: Literal['roof', 'wall', 'floor', 'internal', 'fixed', 'typology_based']
    
    # Cost distribution parameters (£/sqm for area-based, £ total for fixed)
    cost_min: Optional[float] = None
    cost_mode: Optional[float] = None  # This is the base/deterministic value
    cost_max: Optional[float] = None
    
    # Typology-based cost ranges (for interventions like heat pumps)
    cost_by_typology: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Absolute cost caps (after multipliers applied)
    cap_min: Optional[float] = None
    cap_max: Optional[float] = None
    
    # Metadata
    distribution: str = 'triangular'
    confidence: str = 'medium'
    notes: str = ''


# Single unified configuration dictionary
INTERVENTION_CONFIGS: Dict[str, InterventionConfig] = {
    # 'loft_insulation': InterventionConfig(
    #     area_type='roof',
    #     cost_min=10,
    #     cost_mode=20,
    #     cost_max=30,
    #     cap_min=500,
    #     cap_max=2000,
    #     confidence='medium',
    #     notes='Cost per sqm of roof area'
    # ),
    
    'loft_percentile': InterventionConfig(
        area_type='roof',
        cost_min=10,
        cost_mode=20,
        cost_max=30,
        cap_min=500,
        cap_max=2000,
        confidence='medium',
        notes='Cost per sqm of roof area'
    ),
    
    'cavity_wall_percentile': InterventionConfig(
        area_type='wall',
        cost_min=10,
        cost_mode=20,
        cost_max=30,
        cap_min=500,
        cap_max=5000,
        confidence='medium',
        notes='Cost per sqm of external wall area'
    ),
    
    'solid_wall_internal_percentile': InterventionConfig(
        area_type='wall',
        cost_min=55,
        cost_mode=95,
        cost_max=140,
        cap_min=6000,
        cap_max=9000,
        confidence='medium',
        notes='Cost per sqm of external wall area'
    ),
    
    'solid_wall_external_percentile': InterventionConfig(
        area_type='wall',
        cost_min=70,
        cost_mode=115,
        cost_max=160,
        cap_min=7100,
        cap_max=15000,
        confidence='medium',
        notes='Cost per sqm of external wall area'
    ),
    
    'loft_and_wall_installation': None 
    ,

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
    
    'double_glazing': InterventionConfig(
        area_type='fixed',
        cost_min=3000,  # Example values - adjust based on your glazing logic
        cost_mode=5000,
        cost_max=8000,
        cap_min=2000,
        cap_max=15000,
        confidence='medium',
        notes='Fixed cost varies by typology and flat count'
    ),
}


class CostEstimator:
    """Refactored cost estimator with unified config"""
    
    def __init__(self, configs: Dict[str, InterventionConfig] = None):
        self.configs = configs or INTERVENTION_CONFIGS
    
    def get_area_for_intervention(self, 
                                   intervention: str,
                                   building_chars) -> float:
        """Get the relevant area for an intervention"""
        config = self.configs.get(intervention)
        if not config:
            raise ValueError(f"Unknown intervention: {intervention}")
        
        area_type = config.area_type
        
        if area_type == 'roof':
            return building_chars.roof_area_estimate
        elif area_type == 'wall':
            return building_chars.external_wall_area_estimate
        elif area_type == 'floor':
            return building_chars.building_footprint_area
        elif area_type == 'internal':
            return building_chars.gross_internal_area
        elif area_type == 'fixed':
            return 1.0  # For fixed costs, area multiplier is 1
        else:
            raise ValueError(f"Unknown area type: {area_type}")
    
    def sample_intervention_cost(self,
                                intervention: str,
                                building_chars,
                                typology: str,
                                age_band: str,
                                region: str,
                                regional_multiplier: float,
                                age_multiplier: float,
                                complexity_multiplier: float,
                                n_samples: int = 1) -> np.ndarray:
        """
        Unified Monte Carlo sampling for all intervention types.
        
        Returns:
            np.ndarray : Array of sampled costs
        """
        config = self.configs.get(intervention)
        if not config:
            raise ValueError(f"Unknown intervention: {intervention}")
        
        # Handle typology-based interventions (e.g., heat pumps)
        if config.area_type == 'typology_based':
            if config.cost_by_typology is None:
                raise ValueError(f"Typology-based intervention {intervention} missing cost_by_typology")
            
            # Get the cost range for this typology
            cost_range = config.cost_by_typology.get(
                typology,
                config.cost_by_typology.get('all_unknown_typology', (10000, 15000))
            )
            
            min_cost, max_cost = cost_range
            mode_cost = (min_cost + max_cost) / 2  # Use midpoint as mode
            
            # Sample from triangular distribution
            cost_samples = np.random.triangular(
                min_cost,
                mode_cost,
                max_cost,
                n_samples
            )
            
            # Apply multipliers (heat pumps typically use age and regional only)
            final_costs = cost_samples * age_multiplier * regional_multiplier
            
            # Apply caps if specified
            if config.cap_min is not None and config.cap_max is not None:
                final_costs = np.clip(final_costs, config.cap_min, config.cap_max)
            
            return final_costs
        
        # Handle area-based and fixed-cost interventions
        area = self.get_area_for_intervention(intervention, building_chars)
        
        # Sample cost per unit from triangular distribution
        cost_per_unit_samples = np.random.triangular(
            config.cost_min,
            config.cost_mode,
            config.cost_max,
            n_samples
        )
        
        # Calculate base costs
        base_costs = area * cost_per_unit_samples
        
        # Apply multipliers
        if config.area_type == 'fixed':
            # Fixed costs might use different multipliers
            final_costs = base_costs * age_multiplier * regional_multiplier
        else:
            # Area-based costs use all multipliers
            final_costs = base_costs * age_multiplier * complexity_multiplier * regional_multiplier
        
        # Apply caps (absolute limits)
        final_costs = np.clip(final_costs, config.cap_min, config.cap_max)
        
        return final_costs
    
    # def get_deterministic_cost(self,
    #                           intervention: str,
    #                           building_chars,
    #                           typology: str = None,
    #                           regional_multiplier: float = 1.0,
    #                           age_multiplier: float = 1.0,
    #                           complexity_multiplier: float = 1.0) -> float:
    #     """
    #     Get deterministic (mode-based) cost estimate
    #     """
    #     config = self.configs.get(intervention)
    #     if not config:
    #         raise ValueError(f"Unknown intervention: {intervention}")
        
    #     # Handle typology-based interventions
    #     if config.area_type == 'typology_based':
    #         if config.cost_by_typology is None:
    #             raise ValueError(f"Typology-based intervention {intervention} missing cost_by_typology")
            
    #         if typology is None:
    #             raise ValueError(f"Typology required for intervention {intervention}")
            
    #         # Get the cost range for this typology
    #         cost_range = config.cost_by_typology.get(
    #             typology,
    #             config.cost_by_typology.get('all_unknown_typology', (10000, 15000))
    #         )
            
    #         min_cost, max_cost = cost_range
    #         mode_cost = (min_cost + max_cost) / 2  # Use midpoint as mode
            
    #         final_cost = mode_cost * age_multiplier * regional_multiplier
            
    #         # Apply caps if specified
    #         if config.cap_min is not None and config.cap_max is not None:
    #             final_cost = np.clip(final_cost, config.cap_min, config.cap_max)
            
    #         return final_cost
        
    #     # Handle area-based and fixed-cost interventions
    #     area = self.get_area_for_intervention(intervention, building_chars)
    #     base_cost = area * config.cost_mode
        
    #     if config.area_type == 'fixed':
    #         final_cost = base_cost * age_multiplier * regional_multiplier
    #     else:
    #         final_cost = base_cost * age_multiplier * complexity_multiplier * regional_multiplier
        
    #     # Apply caps
    #     return np.clip(final_cost, config.cap_min, config.cap_max)
        
        
        # import pandas as pd
# import numpy as np
# import logging
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Any
# import numpy as np

# from dataclasses import dataclass, field
# from typing import Dict, Any, Literal
# import numpy as np
 
# from .BuildingCharacteristics import BuildingCharacteristics  

# # @dataclass
# # class InterventionScalingRules:
# #     """Defines how interventions scale with building characteristics AND their energy savings."""
     


# @dataclass
# class InterventionConfig:
#     """Unified configuration for a single intervention"""
#     # Core intervention properties
#     area_type: Literal['roof', 'wall', 'floor', 'internal', 'fixed']
    
#     # Cost distribution parameters (£/sqm for area-based, £ total for fixed)
#     cost_min: float
#     cost_mode: float  # This is the base/deterministic value
#     cost_max: float
    
#     # Absolute cost caps (after multipliers applied)
#     cap_min: float
#     cap_max: float
    
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
    
#     # # Fixed-cost interventions (can be added similarly)
#     # 'heat_pump_upgrade': InterventionConfig(
#     #     area_type='fixed',
#     #     cost_min=8000,  # Example values - adjust based on your heat pump logic
#     #     cost_mode=12000,
#     #     cost_max=18000,
#     #     cap_min=6000,
#     #     cap_max=25000,
#     #     confidence='medium',
#     #     notes='Fixed cost varies by typology'
#     # ),
    
#     # 'double_glazing': InterventionConfig(
#     #     area_type='fixed',
#     #     cost_min=3000,  # Example values - adjust based on your glazing logic
#     #     cost_mode=5000,
#     #     cost_max=8000,
#     #     cap_min=2000,
#     #     cap_max=15000,
#     #     confidence='medium',
#     #     notes='Fixed cost varies by typology and flat count'
#     # ),
#     }

    

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
    
#     heat_pump_cost_ranges = {
#         # Flats: 7-9k
#         'Very tall point block flats': (10000, 15000),
#         'Tall flats 6-15 storeys': (10000, 15000),
#         'Medium height flats 5-6 storeys': (10000, 15000),
#         '3-4 storey and smaller flats': (10000, 15000),
        
#         # Terraces: 7-9k
        
#         'Small low terraces': (7000, 9000),
#         '2 storeys terraces with t rear extension': (7000, 9000),
        
#         # 3 bed semi (standard & large semi): 8-11k
#         'Large semi detached': (8000, 15000),
#         'Standard size semi detached': (8000, 15000),
        
#         # Large 4+ bed (detached properties): 10-13k
#         'Tall terraces 3-4 storeys': (12000, 20000),
#         'Very large detached': (12000, 20000),
#         'Large detached': (12000, 20000),
#         'Standard size detached': (12000, 20000),
        
#         # Unknown - using mid-range
#         'all_unknown_typology': (10000, 15000)
# }

#     def sample_heat_pump_cost_triangular(self, typology, n_samples=10):
#         """
#         Generate Monte Carlo samples using triangular distribution
#         (assumes most likely cost is the midpoint)
#         """
#         import numpy as np
        
#         min_cost, max_cost = self.heat_pump_cost_ranges.get(typology, (7000, 11000))
#         mode_cost = (min_cost + max_cost) / 2  # midpoint as most likely
        
#         samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
        
#         return samples
    
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
        
#         # Get relevant area (or 1.0 for fixed costs)
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
#         # Note: You might want to customize which multipliers apply to which interventions
#         if config.area_type == 'fixed':
#             # Fixed costs might use different multipliers
#             final_costs = base_costs * age_multiplier * regional_multiplier
#         else:
#             # Area-based costs use all multipliers
#             final_costs = base_costs * age_multiplier * complexity_multiplier * regional_multiplier
        
#         # Apply caps (absolute limits)
#         final_costs = np.clip(final_costs, config.cap_min, config.cap_max)
        
#         return final_costs
    
#     # def get_deterministic_cost(self,
#     #                           intervention: str,
#     #                           building_chars,
#     #                           regional_multiplier: float = 1.0,
#     #                           age_multiplier: float = 1.0,
#     #                           complexity_multiplier: float = 1.0) -> float:
#     #     """
#     #     Get deterministic (mode-based) cost estimate
#     #     """
#     #     config = self.configs.get(intervention)
#     #     if not config:
#     #         raise ValueError(f"Unknown intervention: {intervention}")
        
#     #     area = self.get_area_for_intervention(intervention, building_chars)
#     #     base_cost = area * config.cost_mode
        
#     #     if config.area_type == 'fixed':
#     #         final_cost = base_cost * age_multiplier * regional_multiplier
#     #     else:
#     #         final_cost = base_cost * age_multiplier * complexity_multiplier * regional_multiplier
        
#     #     # Apply caps
#     #     return np.clip(final_cost, config.cap_min, config.cap_max)


# # # Example usage
# # if __name__ == "__main__":
# #     # Create estimator
# #     estimator = CostEstimator()
    
# #     # Access config easily
# #     loft_config = estimator.configs['loft_insulation']
# #     print(f"Loft insulation mode cost: £{loft_config.cost_mode}/sqm")
# #     print(f"Cost range: £{loft_config.cost_min}-£{loft_config.cost_max}/sqm")
# #     print(f"Total cost cap: £{loft_config.cap_min}-£{loft_config.cap_max}")



#     #  # Area-based interventions (cost per sqm) - BASE VALUES for deterministic calculation
#     # area_based_interventions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
#     #     'loft_percentile': {
#     #         'base_cost_per_sqm': 20,  # Mode value from uncertainty_parameters
#     #         'min_cost': 500,
#     #         'max_cost': 2000,
#     #         'area_type': 'roof'
#     #     },
#     #     'cavity_wall_percentile': {
#     #         'base_cost_per_sqm': 20,
#     #         'min_cost': 500,
#     #         'max_cost': 20000,
#     #         'area_type': 'wall'
#     #     },
#     #     'solid_wall_internal_percentile': {
#     #         'base_cost_per_sqm': 95,
#     #         'min_cost': 6000,
#     #         'max_cost': 20000,
#     #         'area_type': 'wall'
#     #     },
#     #     'solid_wall_external_percentile': {
#     #         'base_cost_per_sqm': 115,
#     #         'min_cost': 7100,
#     #         'max_cost': 25000,
#     #         'area_type': 'wall'
#     #     },
#     # })
    
#     # # Cost uncertainties 
#     # uncertainty_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
#     #     'loft_percentile': {
#     #         'distribution': 'triangular',
#     #         'min': 10,
#     #         'mode': 20,   
#     #         'max': 30,
#     #         'confidence': 'med',
#     #         'cap': [500, 2000],
#     #     },
#     #     'loft_insulation': {
#     #         'distribution': 'triangular',
#     #         'min': 10,
#     #         'mode': 20,   
#     #         'max': 30,
#     #         'confidence': 'med',
#     #         'cap': [500, 2000],
#     #     },
#     #     'cavity_wall_percentile': {
#     #         'distribution': 'triangular',
#     #         'min': 10,
#     #         'mode': 20,
#     #         'max': 30,
#     #         'confidence': 'med',
#     #         'cap': [500, 5000],
#     #     },
     

#     #     'solid_wall_external_percentile': {
#     #         'distribution': 'triangular',
#     #         'min': 70,
#     #         'mode': 115,
#     #         'max': 160,
#     #         'confidence': 'med',
#     #         'cap': [7100, 15000],
#     #     },
#     #     'solid_wall_internal_percentile': {
#     #         'distribution': 'triangular',
#     #         'min': 55,
#     #         'mode': 95,
#     #         'max': 140,
#     #         'confidence': 'med',
#     #         'cap': [6000, 9000],
#     #     },
      
#     # })
    
    
    

 

#     # Double glazing cost ranges for Monte Carlo simulation (in £)
#     double_glazing_cost_ranges: Dict[str, Any] = field(default_factory=lambda:  {
#         # Large houses (detached): 8-15k
#         'Very large detached': (8000, 15000),
#         'Large detached': (8000, 15000),
#         'Standard size detached' : (8000, 15000),
#          'Tall terraces 3-4 storeys': (8000, 15000),
        
#         # Medium houses (semi-detached): 6-9k
#         'Large semi detached': (6000, 9000),
#         'Standard size semi detached': (6000, 9000),
        
#         # Terraced: 4-6k

#         'Small low terraces': (4000, 6000),
#         '2 storeys terraces with t rear extension': (4000, 6000),
        
#         # Flats - these will be scaled by number of flats (see function below)
#         'Very tall point block flats': 'flat_based',
#         'Tall flats 6-15 storeys': 'flat_based',
#         'Medium height flats 5-6 storeys': 'flat_based',
#         '3-4 storey and smaller flats': 'flat_based',
        
#         # Unknown - using mid-range
#         'all_unknown_typology': (5000, 10000)
#     })

#     # Flat-based double glazing costs configuration
#     flat_based_double_glazing: Dict[str, float] = field(default_factory=lambda: {
#         'base_cost_per_flat': 4000,  # Cost per flat in bulk
#         'individual_flat_cost': 6000,  # Cost for single flat
#         'economies_of_scale_threshold': 10,  # Number of flats where bulk pricing kicks in
#         'min_cost': 4000,  # Minimum cost (single flat)
#         'max_cost': 80000  # Maximum reasonable cost for large buildings


#     })

#     def sample_double_glazing_cost_triangular(self, typology, n_samples=1000, num_flats=None):
#         """
#         Generate Monte Carlo samples using triangular distribution
#         (assumes most likely cost is the midpoint)
        
#         FIXED: Properly handles cases where base_cost calculation exceeds max_cost cap,
#         ensuring min_cost <= mode_cost <= max_cost for valid triangular distribution.
#         """
#         import numpy as np
        
#         cost_range = self.double_glazing_cost_ranges.get(typology, (5000, 10000))
        
#         # Handle flat-based calculations
#         if cost_range == 'flat_based':
#             if num_flats is None:
#                 raise ValueError(f"num_flats required for flat typology: {typology}")
            
#             config = self.flat_based_double_glazing
            
#             # Calculate base cost depending on building size
#             if num_flats >= config['economies_of_scale_threshold']:
#                 base_cost = num_flats * config['base_cost_per_flat']
#             else:
#                 base_cost = num_flats * config['individual_flat_cost']
            
#             # Calculate range around base cost (±10%)
#             calculated_min = base_cost * 0.9
#             calculated_max = base_cost * 1.1
#             mode_cost = base_cost
            
#             # Apply absolute caps
#             min_cost = max(calculated_min, config['min_cost'])
#             max_cost = min(calculated_max, config['max_cost'])
            
#             # CRITICAL FIX: Ensure min_cost <= max_cost
#             # This can happen when base_cost is very large and gets capped
#             if min_cost > max_cost:
#                 # Use the cap range itself and set mode to midpoint
#                 min_cost = config['min_cost']
#                 max_cost = config['max_cost']
#                 mode_cost = (min_cost + max_cost) / 2
#             else:
#                 # Clamp mode to be within the valid range [min_cost, max_cost]
#                 mode_cost = np.clip(mode_cost, min_cost, max_cost)
            
#             samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
#         else:
#             # Standard house-based pricing
#             min_cost, max_cost = cost_range
#             mode_cost = (min_cost + max_cost) / 2
#             samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
        
#         return samples


#     # def sample_double_glazing_cost_triangular(self, typology, n_samples=1000, num_flats=None):
#     #     """
#     #     Generate Monte Carlo samples using triangular distribution
#     #     (assumes most likely cost is the midpoint)
#     #     """
#     #     import numpy as np
        
#     #     cost_range = self.double_glazing_cost_ranges.get(typology, (5000, 10000))
        
#     #     # Handle flat-based calculations
#     #     if cost_range == 'flat_based':
#     #         if num_flats is None:
#     #             raise ValueError(f"num_flats required for flat typology: {typology}")
            
#     #         config = self.flat_based_double_glazing
            
#     #         if num_flats >= config['economies_of_scale_threshold']:
#     #             base_cost = num_flats * config['base_cost_per_flat']
#     #             min_cost = max(base_cost * 0.9, config['min_cost'])
#     #             max_cost = min(base_cost * 1.1, config['max_cost'])
#     #         else:
#     #             base_cost = num_flats * config['individual_flat_cost']
#     #             min_cost = max(base_cost * 0.9, config['min_cost'])
#     #             max_cost = min(base_cost * 1.1, config['max_cost'])
            
#     #         mode_cost = base_cost  # Most likely cost is the base calculation
#     #         samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
#     #     else:
#     #         # Standard house-based pricing
#     #         min_cost, max_cost = cost_range
#     #         mode_cost = (min_cost + max_cost) / 2
#     #         samples = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
        
#     #     return samples
 
#     # # Perimeter typology multipliers
#     # perimeter_typology_multipliers: Dict[str, float] = field(default_factory=lambda: {
#     #     'Standard size semi detached': 0.75,
#     #     'Large semi detached': 0.75,
#     #     'Standard size detached': 1.0,
#     #     'Large detached': 1.0,
#     #     'Very large detached': 1.0,
#     #     'Small low terraces': 0.5,
#     #     'Tall terraces 3-4 storeys': 0.5,
#     #     'Medium height flats 5-6 storeys': 0.8,
#     #     '3-4 storey and smaller flats': 0.8,
#     #     'Tall flats 6-15 storeys': 0.9,
#     #     'Very tall point block flats': 0.95,
#     #     'all_unknown_typology': 0.8,
#     # })

#     # def sample_intervention_cost(self, 
#     #                                 intervention: str,
#     #                                 building_chars: BuildingCharacteristics,
#     #                                 typology: str,
#     #                                 age_band: str,
#     #                                 region: str,
#     #                                 regional_multiplier: float,
#     #                                 age_multiplier: float,
#     #                                 complexity_multiplier: float,
#     #                                 n_samples: int = 1) -> np.ndarray:
#     #         """
#     #         Unified Monte Carlo sampling for all intervention types.
            
#     #         Parameters:
#     #         -----------
#     #         intervention : str
#     #             Name of the intervention
#     #         building_chars : BuildingCharacteristics
#     #             Building physical characteristics
#     #         typology : str
#     #             Building typology
#     #         age_band : str
#     #             Age band of the building
#     #         region : str
#     #             Regional code
#     #         regional_multiplier : float
#     #             Pre-calculated regional cost multiplier
#     #         age_multiplier : float
#     #             Pre-calculated age band multiplier
#     #         complexity_multiplier : float
#     #             Pre-calculated typology complexity multiplier
#     #         n_samples : int
#     #             Number of Monte Carlo samples to generate
                
#     #         Returns:
#     #         --------
#     #         np.ndarray : Array of sampled costs
#     #         """
            
#     #         # ===== AREA-BASED INTERVENTIONS =====
#     #         if intervention in self.area_based_interventions:
#     #             rules = self.area_based_interventions[intervention]
#     #             uncertainty = self.uncertainty_parameters.get(intervention)
                
#     #             if uncertainty is None:
#     #                 raise ValueError(f"No uncertainty parameters defined for {intervention}")
                
#     #             # Determine which area to use
#     #             area_type = rules['area_type']
#     #             if area_type == 'roof':
#     #                 area = building_chars.roof_area_estimate
#     #             elif area_type == 'wall':
#     #                 area = building_chars.external_wall_area_estimate
#     #             elif area_type == 'floor':
#     #                 area = building_chars.building_footprint_area
#     #             elif area_type == 'internal':
#     #                 area = building_chars.gross_internal_area
#     #             else:
#     #                 raise ValueError(f"Unknown area type: {area_type}")
                
#     #             # Sample cost per sqm from triangular distribution
#     #             cost_per_sqm_samples = np.random.triangular(
#     #                 uncertainty['min'],
#     #                 uncertainty['mode'],
#     #                 uncertainty['max'],
#     #                 n_samples
#     #             )
                
#     #             # Calculate base costs for all samples
#     #             base_costs = area * cost_per_sqm_samples
                
#     #             # Apply multipliers
#     #             final_costs = base_costs * age_multiplier * complexity_multiplier
                
#     #             # Apply caps (absolute limits)
#     #             cap_min, cap_max = uncertainty['cap']
#     #             final_costs = np.clip(final_costs, cap_min, cap_max)  * regional_multiplier
                
#     #             return final_costs
            
#     #         # ===== HEAT PUMP (FIXED COST WITH TYPOLOGY VARIATION) =====
#     #         elif intervention == 'heat_pump_upgrade':
#     #             samples = self.sample_heat_pump_cost_triangular(typology, n_samples)
#     #             # Apply multipliers (heat pump costs already account for typology in base ranges)
#     #             final_costs = samples * age_multiplier  * regional_multiplier
#     #             return final_costs
            
#     #         # ===== DOUBLE GLAZING / WINDOW UPGRADES (FIXED COST WITH TYPOLOGY VARIATION) =====
#     #         elif intervention == 'double_glazing':
#     #             samples = self.sample_double_glazing_cost_triangular(
#     #                 typology, 
#     #                 n_samples, 
#     #                 building_chars.flat_count
#     #             )
#     #             # Apply multipliers
#     #             final_costs = samples * age_multiplier  * regional_multiplier
#     #             return final_costs
            
     
#     #         else:
#     #             raise ValueError(f"Unknown intervention: {intervention}. Cannot sample cost.")



#  # 'solid_wall_percentile': {
#         #     'base_cost_per_sqm': 100,
#         #     'min_cost': 6000,
#         #     'max_cost': 25000,
#         #     'area_type': 'wall'
#         # },
#         # 'solar_pv': {
#         #     'base_cost_per_sqm': 300,
#         #     'min_cost': 2000,
#         #     'max_cost': 10000,
#         #     'area_type': 'roof'
#         # },
#         # 'floor_insulation': {
#         #     'base_cost_per_sqm': 30,
#         #     'min_cost': 500,
#         #     'max_cost': 4000,
#         #     'area_type': 'floor'
#         # },
#         # 'deep_retrofit_estimate': {
#         #     'base_cost_per_sqm': 300,
#         #     'min_cost': 15000,
#         #     'max_cost': 100000,
#         #     'area_type': 'internal'
#         # }
#           # 'solar_pv': {
#         #     'distribution': 'triangular',
#         #     'min': 100,
#         #     'mode': 300,
#         #     'max': 500,
#         #     'cap': [2000, 10000],
#         # },
#         #    'floor_insulation': {
#         #     'distribution': 'triangular',
#         #     'min': 10,
#         #     'mode': 30,
#         #     'max': 50,
#         #     'cap': [500, 4000],
#         # },

#         # 'deep_retrofit_estimate': { 
#         #     'distribution': 'triangular',
#         #     'min': 800,
#         #     'mode': 985,
#         #     'max': 1100,
#         #     'cap': [30000, 180000],
#         # }

#            # 'internal_wall_insulation': {
#         #     'distribution': 'triangular',
#         #     'min': 55,
#         #     'mode': 95,
#         #     'max': 140,
#         #     'confidence': 'med',
#         #     'cap': [6000, 9000],
#         # },
#         # 'external_wall_insulation': {
#         #     'distribution': 'triangular',
#         #     'min': 70,
#         #     'mode': 115,
#         #     'max': 160,
#         #     'confidence': 'med',
#         #     'cap': [7100, 15000],
#         # },