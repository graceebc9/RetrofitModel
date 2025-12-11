from .BuildingCharacteristics import BuildingCharacteristics
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional, Tuple, List
import logging 
from scipy.stats import norm
import numpy as np
import pandas as pd 

logger = logging.getLogger(__name__)

# Assuming get_intervention_list exists and works as in your original
from .RetrofitPackages import get_intervention_list 
 
from pathlib import Path

# Get the directory where this current file is located
current_dir = Path(__file__).parent

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
                'cap_min': 500, 'cap_max': 8500
            },
            'optimistic': { # 20% cheaper
                'cost_min': 8, 'cost_mode': 16, 'cost_max': 24,
                'cap_min': 400, 'cap_max': 6800
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 13, 'cost_mode': 26, 'cost_max': 39,
                'cap_min': 650, 'cap_max': 11000
            }
        }
    ),
    
    'solid_wall_internal_percentile': InterventionConfig(
        area_type='wall',
        epist_scenarios={
            'central': {
                'cost_min': 55, 'cost_mode': 95, 'cost_max': 140,
                'cap_min': 6000, 'cap_max': 35000
            },
            'optimistic': { # 20% cheaper
                'cost_min': 44, 'cost_mode': 76, 'cost_max': 112,
                'cap_min': 4800, 'cap_max': 28000
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 72, 'cost_mode': 124, 'cost_max': 182,
                'cap_min': 7800, 'cap_max': 45500
            }
        }
    ),
     'solid_wall_external_percentile': InterventionConfig(
        area_type='wall',
        epist_scenarios={
            'central': {
                'cost_min': 70, 'cost_mode': 115, 'cost_max': 160,
                'cap_min': 7100, 'cap_max': 40000
            },
            'optimistic': { # 20% cheaper
                'cost_min': 56, 'cost_mode': 92, 'cost_max': 128,
                'cap_min': 5680, 'cap_max': 32000
            },
            'pessimistic': { # 30% more expensive
                'cost_min': 91, 'cost_mode': 150, 'cost_max': 208,
                'cap_min': 9230, 'cap_max': 52000
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
        
        # Build the path to the CSV file relative to this file
        self.priors_df = pd.read_csv(current_dir / 'global_avs/premise_uprn_priors_LOG_WEIGHTED.csv')

        # 2. Create the lookup maps
        self.default_std_map, self.default_mean_map = self.create_default_maps(self.priors_df)
    
    

    def create_default_maps(self, priors_df):
        """
        Creates maps for the mean (from uprn=1) and a robust 
        std (mean of all valid stds for that type).
        
        Args:
            priors_df (pd.DataFrame): The dataframe from 'premise_uprn_priors.csv'
            
        Returns:
            dict: std_map
            dict: mean_map
        """
        std_map = {}
        mean_map = {}
        
        all_types = priors_df['premise_type'].unique()
        
        # General fallbacks (mean of all good stds)
        general_fallback_std = priors_df['pooled_std'][priors_df['pooled_std'] > 1.0].mean()
        if pd.isna(general_fallback_std): general_fallback_std = 10.0
        
        for p_type in all_types:
            
            # --- Standard Deviation Map (NEW LOGIC) ---
            # 1. Find ALL valid stds for this type
            valid_stds_for_type = priors_df[
                (priors_df['premise_type'] == p_type) & 
                (priors_df['pooled_std'] > 1.0)
            ]['pooled_std']
            
            # 2. Use the mean of those as the default
            if not valid_stds_for_type.empty:
                std_map[p_type] = valid_stds_for_type.mean()
            else:
                # 3. If no valid stds exist for this type, use the global fallback
                std_map[p_type] = general_fallback_std
                    
            # --- Mean Map (Original Logic) ---
            # This logic remains the same, as it's only looking for the uprn=1 mean
            record = priors_df[
                (priors_df['uprn_count'] == 1.0) & 
                (priors_df['premise_type'] == p_type)
            ]
            if not record.empty:
                mean_map[p_type] = record.iloc[0]['weighted_mean']
            else:
                mean_map[p_type] = 0.0 # Fallback
                
        return std_map, mean_map
        
    
    
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

    def get_premise_area(self,  building_chars: BuildingCharacteristics) -> float:
        return building_chars.building_footprint_area
 

    def create_default_std_map(priors_df):
        """
        Creates a map of default standard deviations (from uprn_count=1)
        for each premise_type 
        
        Args:
            priors_df (pd.DataFrame): The dataframe from 'premise_uprn_priors.csv'
            
        Returns:
            dict: {premise_type: default_std}
        """
        std_map = {}
        
        # Get the data for uprn_count = 1
        uprn_1_df = priors_df[priors_df['uprn_count'] == 1.0].copy()
        all_types = priors_df['premise_type'].unique()
        
        # General fallbacks (mean of all good stds)
        general_fallback_std = priors_df['pooled_std'][priors_df['pooled_std'] > 1.0].mean()
        if pd.isna(general_fallback_std): 
            general_fallback_std = 10.0
        
        for p_type in all_types:
            record = uprn_1_df[uprn_1_df['premise_type'] == p_type]
            
            # --- Standard Deviation Map ---
            if not record.empty and record.iloc[0]['pooled_std'] > 1.0:
                std_map[p_type] = record.iloc[0]['pooled_std']
            else:
                # Try to find *any* good std for that type
                fallback_std_series = priors_df[
                    (priors_df['premise_type'] == p_type) & (priors_df['pooled_std'] > 1.0)
                ]['pooled_std']
                if not fallback_std_series.empty:
                    std_map[p_type] = fallback_std_series.iloc[0]
                else:
                    std_map[p_type] = general_fallback_std # Use general fallback
                    
        return std_map

    # --- 2. CORE PROBABILITY ENGIN ---

    def get_uprn_probabilities_refined(self, premise_typology, observed_area, priors_df, default_std_map):
        """
        Calculates posterior probability u:
        - std(2) = std(1)
        - All means are the original weighted_mean
        """
        
        # 1. Filter the model for the given typology
        model_data = priors_df[priors_df['premise_type'] == premise_typology].copy()
        if model_data.empty:
            return pd.DataFrame(columns=['uprn_count', 'posterior_probability']) # Return empty

        # 2. Get the default standard deviation (from uprn_count = 1)
        default_std = default_std_map.get(premise_typology, 10.0)
        
 
        def apply_robust_std(row):
            if row['uprn_count'] == 2:
                # User's rule: ALWAYS use std from uprn=1 for uprn=2
                return default_std 
            elif row['pooled_std'] <= 1.0:
                # Old rule: Use default std for other bad data (0, 3, 4, 12...)
                return default_std
            else:
                # Old rule: Use the original std if it's good (for 0, 1, 3, 4...)
                return row['pooled_std']

        model_data['robust_std'] = model_data.apply(apply_robust_std, axis=1)

        # 4. Calculate Likelihood (using original mean and V2 robust std)
        model_data['likelihood'] = model_data.apply(
            lambda row: norm.pdf(
                observed_area, 
                loc=row['weighted_mean'],  # <-- ALWAYS using original mean
                scale=row['robust_std']
            ),
            axis=1
        )
        
        # 5. Calculate Numerator
        model_data['numerator'] = model_data['likelihood'] * model_data['prior_probability']
        
        # 6. Calculate Denominator
        marginal_likelihood = model_data['numerator'].sum()
        
        # 7. Calculate Posterior Probability
        if marginal_likelihood > 0:
            model_data['posterior_probability'] = model_data['numerator'] / marginal_likelihood
        else:
            model_data['posterior_probability'] = 0.0
            
        # Return a simplified DF, as needed by the cost function
        return model_data[['uprn_count', 'posterior_probability']]


    # def sample_intervention_cost(self,
    #                             intervention: str,
    #                             building_chars: BuildingCharacteristics,
    #                             epist_scenario: str = 'central',
    #                             # --- Accepting sampled epistemic factors ---
    #                             regional_multiplier:float =1.0 ,
    #                             age_multiplier: float = 1.0,
    #                             # complexity_multiplier: float = 1.0,
    #                             **kwargs) -> Tuple[np.ndarray, float]:
    #     """
    #     Samples cost for a SINGLE intervention.

    #     REVISED LOGIC (as per user request):
    #     1.  Always calculate the building-level UPRN scaler based on 'premise_area'
    #         and 'typology', regardless of intervention.
    #     2.  For 'area_type' (wall, roof) costs, the base cost is already
    #         scaled by the building's total area, so the UPRN scaler is NOT applied.
    #     3.  For 'typology_based' (heat pump) costs, the base cost is for a
    #         single unit, so the UPRN scaler IS applied.
    #     """
    #     config = self.configs.get(intervention)
    #     if not config:
    #         raise ValueError(f"Unknown intervention: {intervention}")
        
    #     epist_scenario_params = config.epist_scenarios.get(epist_scenario, config.epist_scenarios.get('central'))
    #     if not epist_scenario_params:
    #         raise ValueError(f"No 'central' or '{epist_scenario}' epist_scenario found for {intervention}")
        
    #     n_samples = kwargs.get('n_samples', 1)
    #     typology = kwargs.get('typology', 'all_unknown_typology')

    #     # --- STEP 1: GET BUILDING-LEVEL UPRN SCALER (FOR ALL INTERVENTIONS) ---
        
    #     # Get the building's actual premise area for the model
    #     building_premise_area = self.get_premise_area()

    #     if building_premise_area is None:
    #         raise Exception('Missing premise area for building ')

    #     logger.debug(f"Running probabilistic scaler for typology '{typology}' with premise_area '{building_premise_area}'")

    #     uprn_probs_df = self.get_uprn_probabilities_refined(
    #         typology,
    #         building_premise_area, 
    #         self.priors_df,
    #         self.default_std_map,
    #     )
        
    #     if uprn_probs_df.empty:
    #         raise Exception('Missing estimated uprn estimation')
    #     else:
    #         expected_uprn_scaler = (uprn_probs_df['posterior_probability'] * uprn_probs_df['uprn_count']).sum()
        
    #     # This is the value we return for logging
    #     logged_uprn_scaler = expected_uprn_scaler
        
    #     # This is the scaler we use for costing (cannot be < 1)
    #     uprn_scaler = max(1.0, expected_uprn_scaler)
        
    #     logger.debug(f"Calculated UPRN scaler: {uprn_scaler:.2f} (Logged: {logged_uprn_scaler:.2f})")

    #     # --- STEP 2: GET BASE COSTS & APPLY SCALER (AS NEEDED) ---

    #     cap_min = epist_scenario_params.get('cap_min')
    #     cap_max = epist_scenario_params.get('cap_max')

    #     if config.area_type == 'typology_based':
    #         # --- BRANCH 1: TYPOLOGY-BASED (e.g., Heat Pump) ---
    #         # Cost is per-unit, so we MUST apply the uprn_scaler.
            
    #         logger.debug(f"Typology-based cost. Applying uprn_scaler: {uprn_scaler:.2f}")
    #         cost_by_typology = epist_scenario_params['cost_by_typology']
    #         default_range = cost_by_typology.get('all_unknown_typology')
    #         cost_range = cost_by_typology.get(typology, default_range)
            
    #         min_cost, max_cost = cost_range
    #         mode_cost = (min_cost + max_cost) / 2
            
    #         # Base cost is for a SINGLE unit
    #         base_costs = np.random.triangular(min_cost, mode_cost, max_cost, n_samples)
            
    #         # Apply all multipliers INCLUDING the UPRN scaler
    #         final_costs = base_costs * regional_multiplier * age_multiplier * uprn_scaler
            
    #         # Scale caps by the UPRN scaler
    #         if cap_min is not None: cap_min = cap_min * uprn_scaler
    #         if cap_max is not None: cap_max = cap_max * uprn_scaler

    #     else:
    #         # --- BRANCH 2: AREA-BASED (e.g., Wall, Roof) ---
    #         # Cost is per-sqm, and area is for the WHOLE building.
    #         # DO NOT apply the uprn_scaler to the cost.
            
    #         logger.debug(f"Area-based cost. UPRN scaler {uprn_scaler:.2f} is NOT applied to cost.")
    #         area = self.get_area_for_intervention(intervention, building_chars)
            
    #         # Base cost is for the ENTIRE building area
    #         base_costs = area * np.random.triangular(
    #             epist_scenario_params['cost_min'], 
    #             epist_scenario_params['cost_mode'], 
    #             epist_scenario_params['cost_max'], 
    #             n_samples
    #         )
            
    #         # Apply multipliers EXCLUDING the UPRN scaler
    #         final_costs = base_costs * regional_multiplier * age_multiplier
            
    #         # Caps are NOT scaled by UPRN count (they are for the whole intervention)
    #         # CORRECTED: All caps are "per-unit" and must be scaled by the uprn_scaler.
    #         if cap_min is not None: 
    #             cap_min = cap_min * uprn_scaler
    #             logger.debug(f"Scaled cap_min to: {cap_min:.2f}")
    #         if cap_max is not None: 
    #             cap_max = cap_max * uprn_scaler
    #             logger.debug(f"Scaled cap_max to: {cap_max:.2f}")

 

    #     # --- STEP 3: APPLY FINAL CAPS ---
    #     if cap_min is not None and cap_max is not None:
    #         final_costs = np.clip(final_costs, cap_min, cap_max)
            
    #     return final_costs, logged_uprn_scaler


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
            logger.warning(f'Unknown intervention: {intervention} ' ) 
            raise ValueError(f"Unknown intervention: {intervention}")
        
        num_flats = getattr(building_chars, 'number_of_flats', 1)

        # --- STEP 0: GET BUILDING-LEVEL UPRN SCALER (FOR ALL INTERVENTIONS) for non flats ---
        if num_flats == 1:
            building_premise_area = self.get_premise_area(building_chars)
            typology =  getattr(building_chars, 'typology',  'Unknown')
            if building_premise_area is None:
                logger.warning('error: Missing premise area for building ' ) 
                raise Exception('Missing premise area for building ')
            logger.debug(f"Running probabilistic scaler for typology {typology} with premise_area '{building_premise_area}'"  ) 
            uprn_probs_df = self.get_uprn_probabilities_refined(
                typology,
                building_premise_area, 
                self.priors_df,
                self.default_std_map,
            )
            if uprn_probs_df.empty:
                logger.warning('Missing estimated uprn estimation ' ) 
                raise Exception('Missing estimated uprn estimation')
            else:
                expected_uprn_scaler = (uprn_probs_df['posterior_probability'] * uprn_probs_df['uprn_count']).sum()
            # This is the value we return for logging
            logged_uprn_scaler = expected_uprn_scaler
            # This is the scaler we use for costing (cannot be < 1)
            uprn_scaler = max(1.0, expected_uprn_scaler)
        
        logger.debug(f"Calculated UPRN scaler: {uprn_scaler:.2f} (Logged: {logged_uprn_scaler:.2f})")

        
        # 1. Get base cost parameters from the discrete 'scenario'
        epist_scenario_params = config.epist_scenarios.get(epist_scenario, config.epist_scenarios.get('central'))
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
            final_costs = base_costs * regional_multiplier * age_multiplier  * uprn_scaler
        else:
            area = self.get_area_for_intervention(intervention, building_chars)
            typology = kwargs.get('typology', 'all_unknown_typology')
            base_costs = area * np.random.triangular(
                epist_scenario_params['cost_min'], 
                epist_scenario_params['cost_mode'], 
                epist_scenario_params['cost_max'], 
                n_samples
            )
            final_costs = base_costs  *regional_multiplier * age_multiplier 
 
        
        # 4. Apply caps (which are also part of the epist_scenario)
        cap_min = epist_scenario_params.get('cap_min')
        cap_max = epist_scenario_params.get('cap_max')
        

        if cap_min is not None and cap_max is not None:
            # NEW: Scale caps for wall insulation in multi-flat buildings
            if config.area_type == 'wall':
                # Check if building has flats
             
                if num_flats > 1:
                    # Scale caps proportionally  
                    flat_scaling_factor = max(1, num_flats * 0.8)  # 0.8 for economies of scale
                    cap_min = cap_min * flat_scaling_factor
                    cap_max = cap_max * flat_scaling_factor
                    logger.debug(f"Scaled caps for {num_flats} flats: min={cap_min:.0f}, max={cap_max:.0f}")
            
            # scale all caps by uprn faqctor for non flat typs 
            if num_flats == 1:
                cap_min = cap_min * uprn_scaler
                cap_max = cap_max * uprn_scaler
                logger.debug(f"Scaled cap_min to: {cap_min:.2f}")
                logger.debug(f"Scaled cap_max to: {cap_max:.2f}")

            final_costs = np.clip(final_costs, cap_min, cap_max)
            
        return final_costs

