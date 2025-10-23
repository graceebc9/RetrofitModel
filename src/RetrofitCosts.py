from .BuildingCharacteristics import BuildingCharacteristics
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional, Tuple, List
import logging 

logger = logging.getLogger(__name__)

# Assuming get_intervention_list exists and works as in your original
from .RetrofitPackages import get_intervention_list 
 

@dataclass
class InterventionConfig:
    """
    Unified configuration for a single intervention.
    Cost parameters are now nested under 'epis_scenarios'.
    """
    area_type: Literal['roof', 'wall', 'floor', 'internal', 'fixed', 'typology_based']
    
    # NEW: Scenarios dictionary to hold all cost parameters
    # This allows for epistemic uncertainty (e.g., optimistic/pessimistic)
    epist_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata (remains top-level)
    distribution: str = 'triangular'
    confidence: str = 'medium'
    notes: str = ''

# Configuration dictionary for individual interventions
INTERVENTION_CONFIGS: Dict[str, InterventionConfig] = {
    
    'loft_percentile': InterventionConfig(
        area_type='roof',
        notes='Cost per sqm of roof area',
        epist_scenarios={
            'central': {
                'cost_min': 10, 'cost_mode': 20, 'cost_max': 30,
                'cap_min': 500, 'cap_max': 2000
            },
            'optimistic': {  # 20% cheaper
                'cost_min': 8, 'cost_mode': 16, 'cost_max': 24,
                'cap_min': 400, 'cap_max': 1800
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 13, 'cost_mode': 26, 'cost_max': 39,
                'cap_min': 650, 'cap_max': 2600
            }
        }
    ),
    
    'cavity_wall_percentile': InterventionConfig(
        area_type='wall',
        epist_scenarios={
            'central': {
                'cost_min': 10, 'cost_mode': 20, 'cost_max': 30,
                'cap_min': 500, 'cap_max': 5000
            },
            'optimistic': { # 20% cheaper
                'cost_min': 8, 'cost_mode': 16, 'cost_max': 24,
                'cap_min': 400, 'cap_max': 4000
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 13, 'cost_mode': 26, 'cost_max': 39,
                'cap_min': 650, 'cap_max': 6500
            }
        }
    ),
    
    'solid_wall_internal_percentile': InterventionConfig(
        area_type='wall',
        epist_scenarios={
            'central': {
                'cost_min': 55, 'cost_mode': 95, 'cost_max': 140,
                'cap_min': 6000, 'cap_max': 9000
            },
            'optimistic': { # 20% cheaper
                'cost_min': 44, 'cost_mode': 76, 'cost_max': 112,
                'cap_min': 4800, 'cap_max': 7200
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 72, 'cost_mode': 124, 'cost_max': 182,
                'cap_min': 7800, 'cap_max': 11700
            }
        }
    ),
     'solid_wall_external_percentile': InterventionConfig(
        area_type='wall',
        epist_scenarios={
            'central': {
                'cost_min': 70, 'cost_mode': 115, 'cost_max': 160,
                'cap_min': 7100, 'cap_max': 15000
            },
            'optimistic': { # 20% cheaper
                'cost_min': 56, 'cost_mode': 92, 'cost_max': 128,
                'cap_min': 5680, 'cap_max': 12000
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 91, 'cost_mode': 150, 'cost_max': 208,
                'cap_min': 9230, 'cap_max': 19500
            }
        }
    ),

   'heat_pump_percentile': InterventionConfig(
        area_type='typology_based',
        notes='Cost varies significantly by building typology and size',
        epist_scenarios={
            'central': {
                'cost_by_typology': {
                    'Very tall point block flats': (10000, 15000),
                    'Tall flats 6-15 storeys': (10000, 15000),
                    'Medium height flats 5-6 storeys': (10000, 15000),
                    '3-4 storey and smaller flats': (10000, 15000),
                    'Small low terraces': (7000, 9000),
                    '2 storeys terraces with t rear extension': (7000, 9000),
                    'Large semi detached': (8000, 15000),
                    'Standard size semi detached': (8000, 15000),
                    'Tall terraces 3-4 storeys': (12000, 20000),
                    'Very large detached': (12000, 20000),
                    'Large detached': (12000, 20000),
                    'Standard size detached': (12000, 20000),
                    'all_unknown_typology': (10000, 15000)
                },
                'cap_min': 6000,
                'cap_max': 25000
            },
            'optimistic': { # 30% reduction
                'cost_by_typology': {
                    'Very tall point block flats': (7000, 10500), 
                    'Tall flats 6-15 storeys': (7000, 10500),
                    'Medium height flats 5-6 storeys': (7000, 10500),
                    '3-4 storey and smaller flats': (7000, 10500),
                    'Small low terraces': (4900, 6300),
                    '2 storeys terraces with t rear extension': (4900, 6300),
                    'Large semi detached': (5600, 10500),
                    'Standard size semi detached': (5600, 10500),
                    'Tall terraces 3-4 storeys': (8400, 14000),
                    'Very large detached': (8400, 14000),
                    'Large detached': (8400, 14000),
                    'Standard size detached': (8400, 14000),
                    'all_unknown_typology': (7000, 10500)
                },
                'cap_min': 4500,
                'cap_max': 20000
            },
            'pessimistic': { # 30% increase
                'cost_by_typology': {
                    'Very tall point block flats': (13000, 19500),
                    'Tall flats 6-15 storeys': (13000, 19500),
                    'Medium height flats 5-6 storeys': (13000, 19500),
                    '3-4 storey and smaller flats': (13000, 19500),
                    'Small low terraces': (9100, 11700),
                    '2 storeys terraces with t rear extension': (9100, 11700),
                    'Large semi detached': (10400, 19500),
                    'Standard size semi detached': (10400, 19500),
                    'Tall terraces 3-4 storeys': (15600, 26000),
                    'Very large detached': (15600, 26000),
                    'Large detached': (15600, 26000),
                    'Standard size detached': (15600, 26000),
                    'all_unknown_typology': (13000, 19500)
                },
                'cap_min': 7800, 
                'cap_max': 32500
            }
        }
    ),
}

# --- Cost Estimator Class (Updated for Multipliers) ---
# (This class remains unchanged from the previous answer, 
# as it's already designed to handle scenarios and multipliers)

class CostEstimator:
    """Estimates intervention costs using Monte Carlo simulation."""
    
    def __init__(self, configs: Dict[str, InterventionConfig] = None):
        self.configs = configs or INTERVENTION_CONFIGS

    def sample_cost_for_package(self,
                                intervention: str,
                                building_chars: BuildingCharacteristics,
                                epist_scenario: str = 'central',
                                **kwargs) -> np.ndarray:
        """
        Calculates the combined cost for a list of interventions.
        
        It passes all **kwargs (including n_samples, typology, and
        any epistemic multipliers) down to the single cost sampler.
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
            # Pass all kwargs down
            component_costs = self.sample_intervention_cost(
                intervention=intervention_name,
                building_chars=building_chars,
                epist_scenario=epist_scenario, 
                **kwargs
            )
            total_costs += component_costs
        
        return total_costs

    def get_area_for_intervention(self, intervention: str, building_chars: BuildingCharacteristics) -> float:
        """Helper method to get the correct area for an intervention."""
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
                                epist_scenario: str = 'central',
                                # --- Accepting sampled epistemic factors ---
                                regional_multiplier:float =1.0 ,
                                age_multiplier: float = 1.0,
                                # complexity_multiplier: float = 1.0,
                                **kwargs) -> np.ndarray:
        """
        Samples cost for a SINGLE intervention based on:
        1. The discrete 'scenario' (to get base costs).
        2. Continuous epistemic multipliers (to scale the costs).
        """
        config = self.configs.get(intervention)
        if not config:
            raise ValueError(f"Unknown intervention: {intervention}")
        
        # 1. Get base cost parameters from the discrete 'scenario'
        epist_scenario_params = config.epist_scenario.get(epist_scenario, config.scenarios.get('central'))
        if not epist_scenario_params:
            raise ValueError(f"No 'central' or '{epist_scenario}' epist_scenario found for {intervention}")
        
        n_samples = kwargs.get('n_samples', 1)
        
        # 2. Get base cost samples (Aleatoric "Inner" Loop)
        if config.area_type == 'typology_based':
            typology = kwargs.get('typology', 'all_unknown_typology')
            cost_by_typology = epist_scenario_params['cost_by_typology']
            default_range = cost_by_typology.get('all_unknown_typology')
            cost_range = cost_by_typology.get(typology, default_range)
            
            min_cost, max_cost = cost_range
            mode_cost = (min_cost + max_cost) / 2
            base_costs = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
        else:
            area = self.get_area_for_intervention(intervention, building_chars)
            base_costs = area * np.random.triangular(
                epist_scenario_params['cost_min'], 
                epist_scenario_params['cost_mode'], 
                epist_scenario_params['cost_max'], 
                n_samples
            )
        
        # 3. Apply sampled epistemic multipliers
        # if config.area_type == 'typology_based':
        #     final_costs = base_costs  * age_band_multipliers_uncertainty
        # else:
        final_costs = base_costs  *regional_multiplier * age_multiplier 
        
        # 4. Apply caps (which are also part of the epist_scenario)
        cap_min = epist_scenario_params.get('cap_min')
        cap_max = epist_scenario_params.get('cap_max')
        
        if cap_min is not None and cap_max is not None:
            final_costs = np.clip(final_costs, cap_min, cap_max)
            
        return final_costs

