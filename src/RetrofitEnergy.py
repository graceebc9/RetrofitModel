import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
 


# # Add logger at the top of your module
# logger = logging.getLogger(__name__)



from .BuildingCharacteristics import BuildingCharacteristics 

@dataclass
class RetrofitEnergy:
    """Enhanced configuration with Monte Carlo cost sampling."""
    
    # Improvement factor for solid wall vs cavity wall (default 10% better)
    solid_wall_internal_improvement_factor: float = 0.1
    solid_wall_external_improvement_factor: float = 0.2
    
    solar_regional_multiplier: Dict[str, float] = field(default_factory=lambda: {
        # high sun regions 
        'SW': 1.20,  # South West            
        'SE': 1.20,            
        'EE': 1.2,
        # mid region 
        'LN': 1.00,  # London
        'EM': 1.00,  # East Midlands
        'WM': 1.00,  # West Midlands
        
        # low sun regions 
        'YH': 0.8,  # Yorkshire and Humber
        'NW': 0.8,  # North West
        'NE': 0.8,  # North East
        'WA': 0.8,  # Wales
    }) 
    
    def __post_init__(self):
        """Generate solid wall configs after initialization."""
        # Generate solid wall configurations from cavity wall
        cavity_config = self.energysaving_uncertainty_parameters['cavity_wall_percentile']
        
        # Create solid internal wall config
        self.energysaving_uncertainty_parameters['solid_wall_internal_percentile'] = \
            self._create_solid_wall_config(cavity_config, self.solid_wall_internal_improvement_factor)
        
        # Create solid external wall config
        self.energysaving_uncertainty_parameters['solid_wall_external_percentile'] = \
            self._create_solid_wall_config(cavity_config, self.solid_wall_external_improvement_factor)
    
    def _create_solid_wall_config(self, cavity_wall_config: Dict, improvement_factor: float) -> Dict:
        """
        Create solid wall config from cavity wall config.
        
        Args:
            cavity_wall_config: The cavity wall configuration dict
            improvement_factor: Factor by which solid wall performs better
                               - Positive means are reduced by this factor
                               - Negative means are increased (more negative) by this factor
        
        Returns:
            Dictionary with solid wall configuration
        """
        solid_wall_config = {
            'distribution': cavity_wall_config['distribution'],
            'gas': {}
        }
        
        for percentile, params in cavity_wall_config['gas'].items():
            mean = params['mean']
            
            # Apply improvement: reduce positive, increase magnitude of negative
            if mean >= 0:
                adjusted_mean = mean * (1 - improvement_factor)
            else:
                adjusted_mean = mean * (1 + improvement_factor)
            
            solid_wall_config['gas'][percentile] = {
                'mean': adjusted_mean,
                'sd': params['sd']  # Keep SD the same
            }
        
        return solid_wall_config
    
    # percentiel ones are from diaz anadaon paper on pecentiles 
    energysaving_uncertainty_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'joint_loft_wall_add': {'distribution': 'None',
                                       },  
        'joint_loft_wall_decay': {'distribution': 'None',
                                       },  
        'joint_heat_ins_add': {'distribution': 'None',
                                       },  
        'joint_heat_ins_decay': {'distribution': 'None',
                                       },  
        'joint_loft_wall_floor_decay': {'distribution': 'None',
                                       },  
        
        'loft_percentile': {
            'distribution': 'normal',
            'gas': {
                0: {'mean': 0.11794812, 'sd': 0.86019675},
                1: {'mean': 0.02333679, 'sd': 0.46902747},
                2: {'mean': -0.02471, 'sd': 0.37714685},
                3: {'mean': -0.0476622, 'sd': 0.31667587},
                4: {'mean': -0.0629936, 'sd': 0.28913223},
                5: {'mean': -0.0781211, 'sd': 0.20514012},
                6: {'mean': -0.0929435, 'sd': 0.20214586},
                7: {'mean': -0.1048867, 'sd': 0.20783682},
                8: {'mean': -0.1113446, 'sd': 0.23797363},
                9: {'mean': -0.110213, 'sd': 0.38473549}
            }
        },
        'cavity_wall_percentile': {
            'distribution': 'normal',
            'gas': {
                0: {'mean': 0.1059182, 'sd': 0.89226159},
                1: {'mean': 0.02026381, 'sd': 0.5013025},
                2: {'mean': -0.023164, 'sd': 0.36765047},
                3: {'mean': -0.0518756, 'sd': 0.32145214},
                4: {'mean': -0.0752905, 'sd': 0.21262479},
                5: {'mean': -0.0975109, 'sd': 0.19887291},
                6: {'mean': -0.1179157, 'sd': 0.19147565},
                7: {'mean': -0.1360034, 'sd': 0.19498504},
                8: {'mean': -0.1537655, 'sd': 0.20694392},
                9: {'mean': -0.1738108, 'sd': 0.37280598}
            }
        },
        # Note: solid_internal_percentile and solid_external_percentile 
        # will be auto-generated in __post_init__

        'heat_pump_percentile': {
            'distribution': 'normal',
            'gas': {
                0: {'mean': -0.70, 'sd': 0.50},
                1: {'mean': -0.80, 'sd': 0.40},
                2: {'mean': -0.85, 'sd': 0.35},
                3: {'mean': -0.90, 'sd': 0.30},
                4: {'mean': -0.92, 'sd': 0.28},
                5: {'mean': -0.95, 'sd': 0.25},
                6: {'mean': -0.96, 'sd': 0.26},
                7: {'mean': -0.97, 'sd': 0.28},
                8: {'mean': -0.98, 'sd': 0.32},
                9: {'mean': -0.99, 'sd': 0.45}
            },
            'electricity': {
                0: {'mean': 0.65, 'sd': 0.45},
                1: {'mean': 0.58, 'sd': 0.35},
                2: {'mean': 0.55, 'sd': 0.30},
                3: {'mean': 0.52, 'sd': 0.27},
                4: {'mean': 0.50, 'sd': 0.25},
                5: {'mean': 0.48, 'sd': 0.22},
                6: {'mean': 0.46, 'sd': 0.24},
                7: {'mean': 0.42, 'sd': 0.26},
                8: {'mean': 0.38, 'sd': 0.30},
                9: {'mean': 0.37, 'sd': 0.42}
            }
        } , 

                
        'solar_pv':{
         'distribution': 'triangular',
            'kwh_per_m': {
                'min': 100,
                'mode': 200,
                'max': 300,
            }
        },
        'loft_insulation': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.039,
                'mode': 0.039,
                'max': 0.17,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'high'
        },
        'cavity_wall_insulation': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.073,
                'mode': 0.082,
                'max': 0.155,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'high'
        },
        # 'internal_wall_insulation': {
        #     'distribution': 'triangular',
        #     'gas': {
        #         'min': 0.132,
        #         'mode': 0.132,
        #         'max': 0.68,
        #     },
        #     'electricity': {
        #         'min': 0.0,
        #         'mode': 0.0,
        #         'max': 0.0,
        #     },
        #     'confidence': 'medium'
        # },
        # 'external_wall_insulation': {
        #     'distribution': 'triangular',
        #     'gas': {
        #         'min': 0.132,
        #         'mode': 0.132,
        #         'max': 0.68,
        #     },
        #     'electricity': {
        #         'min': 0.0,
        #         'mode': 0.0,
        #         'max': 0.0,
        #     },
        #     'confidence': 'medium'
        # },

        'floor_insulation': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.088,
                'mode': 0.169,
                'max': 0.25,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'medium'
        },
        'window_upgrades': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.056,
                'mode': 0.104,
                'max': 0.153,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'medium'
        },
        # 'heat_pump_upgrade': {
        #     'distribution': 'triangular',
        #     'gas': {
        #         'min': 0.90,
        #         'mode': 0.95,
        #         'max': 0.98,
        #     },
        #     'electricity': {
        #         'min': -0.60,  # Negative = increase
        #         'mode': -0.50,
        #         'max': -0.40,
        #     },
        #     'confidence': 'medium'
        # },
 
        'double_glazing': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.056,
                'mode': 0.104,
                'max': 0.153,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'medium'
        },
        'deep_retrofit_estimate': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.55,
                'mode': 0.77,
                'max': 0.99,
            },
            'electricity': {
                'min': 0.20,
                'mode': 0.50,
                'max': 0.70,
            },
            'confidence': 'medium'
        },
    })
 
 
    # def sample_join_savings(self,
    #                             intervention: str,  # Package name like 'loft_and_wall_installation'
    #                             avg_gas_percentile: str,
    #                             wall_type: str, 
    #                             n_samples: int = 1000,
                                
    #                         ) -> Dict[str, np.ndarray]:
    #     """
    #     Sample savings from multiple percentile-based interventions in a package and combine them.
        
    #     Parameters:
    #     -----------
    #     intervention : str
    #         Package name from retrofit_packages (e.g., 'loft_and_wall_installation')
    #     avg_gas_percentile : str
    #         Building's gas percentile (0-9)
    #     wall_type : str
    #         Type of wall intervention
    #     n_samples : int
    #         Number of Monte Carlo samples
    #     method : str
    #         Combination method: 'additive' or 'decay'
    #         - 'additive': simple sum of savings
    #         - 'decay': multiplicative combination using (1-x)(1-y) logic
            
    #     Returns:
    #     --------
    #     Dict with 'gas' key containing combined samples
    #     """
    #     if 'cavity' in wall_type:
    #         wt = 'cavity_wall_percentile' 
    #     elif 'internal' in wall_type:
    #         wt = 'solid_wall_internal_percentile'
    #     elif 'external' in wall_type:
    #         wt = 'solid_wall_external_percentile'
    #     else:
    #         raise Exception('wall-type not as expected: ', wall_type)

    #     joint_intervention_dict = {'joint_loft_wall_add': [wt, 'loft_percentile'], 
    #                             'joint_loft_wall_decay': [wt, 'loft_percentile'], 
    #                                  'joint_heat_ins_add': [wt, 'loft_percentile', 'heat_pump_percentile'] , 
    #                                  'joint_heat_ins_decay': [wt, 'loft_percentile', 'heat_pump_percentile'] , 
    #                                  } 

        
    #     if  'add' in intervention:
    #         method ='additive'
    #     elif 'decay' in intervention:
    #         method ='decay'
    #     else:
    #         raise Exception('Method of joint not defined in intervention name: ', intervention)
    #     # logger.debug(f'Combining method is {method}')
    #     if not str(avg_gas_percentile).isnumeric():
    #         raise ValueError(f'Percentile must be numeric, got: {avg_gas_percentile}')
        
    #     percentile_key = int(avg_gas_percentile)
        
    #     # Get the package containing the list of interventions
    #     if intervention not in joint_intervention_dict.keys():
    #         raise KeyError(f'No package found for: {intervention}')
        
    #     interventions_list = joint_intervention_dict[intervention]
        
    #     # Initialize based on method
    #     if method == 'additive':
    #         combined_samples = np.zeros(n_samples)
    #     elif method == 'decay':
    #         # Start with 1.0 (representing remaining energy usage fraction)
    #         remaining_fraction = np.ones(n_samples)
    #     else:
    #         raise ValueError(f"Method must be 'additive' or 'decay', got: {method}")
        
    #     # Sample from each intervention in the package and combine
    #     for single_intervention in interventions_list:
    #         # Check if intervention exists
    #         if single_intervention not in self.energysaving_uncertainty_parameters:
    #             raise KeyError(f'No data for intervention: {single_intervention}')
            
    #         # Check if percentile exists for this intervention
    #         if percentile_key not in self.energysaving_uncertainty_parameters[single_intervention]['gas']:
    #             raise KeyError(f'No data for percentile {percentile_key} in intervention {single_intervention}')
            
    #         # Get distribution parameters for this intervention at this percentile
    #         dist_params = self.energysaving_uncertainty_parameters[single_intervention]['gas'][percentile_key]
            
    #         # Sample from normal distribution
    #         intervention_samples = np.random.normal(
    #             dist_params['mean'],
    #             dist_params['sd'],
    #             size=n_samples
    #         )
            
    #         # Combine based on method
    #         if method == 'additive':
    #             combined_samples += intervention_samples
    #         elif method == 'decay':
    #             # Multiplicative decay: remaining = remaining * (1 - savings)
    #             remaining_fraction *= (1 - intervention_samples)
        
    #     # Calculate final combined samples
    #     if method == 'decay':
    #         # Convert remaining fraction back to total savings
    #         combined_samples = 1 - remaining_fraction
        
    #     # Clip combined result
    #     combined_samples = np.clip(combined_samples, a_min=-1, a_max=1)
        
    #     return combined_samples
    def sample_join_savings(self,
                            intervention: str,  # Package name like 'loft_and_wall_installation'
                            avg_gas_percentile: str,
                            wall_type: str, 
                            n_samples: int = 1000,
                            
                        ) -> Dict[str, np.ndarray]:
        """
        Sample savings from multiple percentile-based interventions in a package and combine them.
        
        Parameters:
        -----------
        intervention : str
            Package name from retrofit_packages (e.g., 'loft_and_wall_installation')
        avg_gas_percentile : str
            Building's gas percentile (0-9)
        wall_type : str
            Type of wall intervention
        n_samples : int
            Number of Monte Carlo samples
        method : str
            Combination method: 'additive' or 'decay'
            - 'additive': simple sum of savings
            - 'decay': multiplicative combination using (1-x)(1-y) logic
            
        Returns:
        --------
        Dict with 'gas' key containing combined samples. 
        If heat_pump or solar in interventions, also includes 'electricity' key.
        """
        if 'cavity' in wall_type:
            wt = 'cavity_wall_percentile' 
        elif 'internal' in wall_type:
            wt = 'solid_wall_internal_percentile'
        elif 'external' in wall_type:
            wt = 'solid_wall_external_percentile'
        else:
            raise Exception('wall-type not as expected: ', wall_type)

        joint_intervention_dict = {'joint_loft_wall_add': [wt, 'loft_percentile'], 
                                'joint_loft_wall_decay': [wt, 'loft_percentile'], 
                                    'joint_heat_ins_add': [wt, 'loft_percentile', 'heat_pump_percentile'] , 
                                    'joint_heat_ins_decay': [wt, 'loft_percentile', 'heat_pump_percentile'] , 
                                    } 

        
        if 'add' in intervention:
            method = 'additive'
        elif 'decay' in intervention:
            method = 'decay'
        else:
            raise Exception('Method of joint not defined in intervention name: ', intervention)
        
        if not str(avg_gas_percentile).isnumeric():
            raise ValueError(f'Percentile must be numeric, got: {avg_gas_percentile}')
        
        percentile_key = int(avg_gas_percentile)
        
        # Get the package containing the list of interventions
        if intervention not in joint_intervention_dict.keys():
            raise KeyError(f'No package found for: {intervention}')
        
        interventions_list = joint_intervention_dict[intervention]
        
        # Check if we need to process electricity
        include_electricity = any('heat_pump' in interv or 'solar' in interv 
                                for interv in interventions_list)
        
        # Initialize for gas (and electricity if needed) based on method
        if method == 'additive':
            combined_gas = np.zeros(n_samples)
            if include_electricity:
                combined_elec = np.zeros(n_samples)
        elif method == 'decay':
            # Start with 1.0 (representing remaining energy usage fraction)
            remaining_gas = np.ones(n_samples)
            if include_electricity:
                remaining_elec = np.ones(n_samples)
        else:
            raise ValueError(f"Method must be 'additive' or 'decay', got: {method}")
        
        # Sample from each intervention in the package and combine
        for single_intervention in interventions_list:
            # Check if intervention exists
            if single_intervention not in self.energysaving_uncertainty_parameters:
                raise KeyError(f'No data for intervention: {single_intervention}')
            
            # Check if percentile exists for gas
            if percentile_key not in self.energysaving_uncertainty_parameters[single_intervention]['gas']:
                raise KeyError(f'No gas data for percentile {percentile_key} in intervention {single_intervention}')
            
            # Get gas distribution parameters and sample
            gas_params = self.energysaving_uncertainty_parameters[single_intervention]['gas'][percentile_key]
            gas_samples = np.random.normal(
                gas_params['mean'],
                gas_params['sd'],
                size=n_samples
            )
              # CLIP INDIVIDUAL SAMPLES BEFORE COMBINING
            gas_samples = np.clip(gas_samples, a_min=-1, a_max=1)
        
            
            # Sample from electricity if we're including it
            if include_electricity:
                if ('heat_pump' in single_intervention or 'solar' in single_intervention):
                    # This intervention should have electricity data
                    if 'electricity' in self.energysaving_uncertainty_parameters[single_intervention]:
                        if percentile_key in self.energysaving_uncertainty_parameters[single_intervention]['electricity']:
                            elec_params = self.energysaving_uncertainty_parameters[single_intervention]['electricity'][percentile_key]
                            elec_samples = np.random.normal(
                                elec_params['mean'],
                                elec_params['sd'],
                                size=n_samples
                            )
                        else:
                            raise KeyError(f'No electricity data for percentile {percentile_key} in intervention {single_intervention}')
                    else:
                        raise KeyError(f'No electricity data for intervention: {single_intervention}')
                else:
                    # Non-electric intervention (loft, wall) - no electricity impact
                    elec_samples = np.zeros(n_samples)
            
            # Combine based on method
            if method == 'additive':
                combined_gas += gas_samples
                if include_electricity:
                    combined_elec += elec_samples
            elif method == 'decay':
                # Multiplicative decay: remaining = remaining * (1 - savings)
                remaining_gas *= (1 + gas_samples)
                if include_electricity:
                    remaining_elec *= (1 + elec_samples)
        
        # Calculate final combined samples
        if method == 'decay':
            # Convert remaining fraction back to total savings
            combined_gas =-( 1 - remaining_gas)
            if include_electricity:
                combined_elec =-( 1 - remaining_elec)
        
        # Clip combined results
        combined_gas = np.clip(combined_gas, a_min=-1, a_max=1)
        
        result = {'gas': combined_gas}
        
        if include_electricity:
            combined_elec = np.clip(combined_elec, a_min=-1, a_max=1)
            result['electricity'] = combined_elec
        
        return result

    def sample_heat_pump_savings(
        self,
        intervention: str,
        avg_gas_percentile: str,  # or int
        n_samples: int = 1000
    ) -> dict[str, np.ndarray]:
        """Sample savings from normal distributions for gas and electricity.
        
        Returns:
            dict with 'gas' and 'electricity' keys, each containing np.ndarray of samples
        """
        
        if not str(avg_gas_percentile).isnumeric():
            raise ValueError(f'Percentile must be numeric, got: {avg_gas_percentile}')
        
        # Check if intervention exists in data
        if intervention not in self.energysaving_uncertainty_parameters:
            raise KeyError(f'No data for intervention: {intervention}')
        
        # Convert to int for dict lookup
        percentile_key = int(avg_gas_percentile)
        
        # Check if percentile exists for both gas and electricity
        if percentile_key not in self.energysaving_uncertainty_parameters[intervention]['gas']:
            raise KeyError(f'No gas data for percentile: {percentile_key}')
        
        if percentile_key not in self.energysaving_uncertainty_parameters[intervention]['electricity']:
            raise KeyError(f'No electricity data for percentile: {percentile_key}')
        
        # Get distribution parameters
        gas_params = self.energysaving_uncertainty_parameters[intervention]['gas'][percentile_key]
        elec_params = self.energysaving_uncertainty_parameters[intervention]['electricity'][percentile_key]
        
        # Sample from both distributions
        gas_savings = np.random.normal(
            gas_params['mean'],
            gas_params['sd'],
            size=n_samples
        )
        
        elec_savings = np.random.normal(
            elec_params['mean'],
            elec_params['sd'],
            size=n_samples
        )
        
        # Clip values to [-1, 1]
        gas_savings = np.clip(gas_savings, a_min=-1, a_max=1)
        elec_savings = np.clip(elec_savings, a_min=-1, a_max=1)
        
        return {
            'gas': gas_savings,
            'electricity': elec_savings
        }
    
    def sample_intervention_energy_savings_monte_carlo(self,
                                                   intervention: str,
                                                   building_chars: BuildingCharacteristics,
                                                wall_type: str , 
                                                
                                                #    typology: str,
                                                #    age_band: str,
                                                   region: str,
                                                   n_samples: int = 1000,
                                                   roof_scaling: float = 0.8 
                                                   ) -> np.ndarray:
        """
        Sample intervention energy savings using Monte Carlo simulation.
        
        Parameters:
        -----------
        intervention : str
            Name of the intervention
        building_chars : BuildingCharacteristics
            Building physical characteristics
        typology : str
            Building typology
        age_band : str
            Age band of the building
        region : str
            Regional code
        n_samples : int
            Number of Monte Carlo samples (default: 1000)
            
        Returns:
        --------
        np.ndarray : dict of gas and elec with samples 
        
    
        """
        # if typology is None or typology == 'None':
        #     raise ValueError(f"Invalid typology: {typology}")
        avg_gas_percentile = building_chars.avg_gas_percentile
        
        
        # Check if intervention exists in energy savings parameters
        if intervention not in self.energysaving_uncertainty_parameters:
            raise ValueError(f"Intervention '{intervention}' not found in energy savings parameters")
        
        # Get multipliers (if applicable for energy savings)
        # age_mult = self.age_band_multipliers.get(age_band, 1.0)
        # complexity_mult = self.typology_complexity.get(typology, 1.0)
        # regional_mult = self.get_regional_multiplier(region)
        
        # Sample energy savings using the uncertainty parameters
        if intervention == 'solar_pv':
            samples= self.calculate_solar_pv_impact_monte_carlo(
                    region=region,
                    scaled_roof_size=building_chars.solar_roof_area_estimate(roof_scaling), 
                    n_samples=n_samples )
        elif 'joint_' in intervention:
            samples= self.sample_join_savings(
                intervention=intervention, 
                    avg_gas_percentile=avg_gas_percentile,
                    n_samples=n_samples, wall_type = wall_type ,   )
            
        elif 'heat_pump' in intervention: 
            samples = self.sample_heat_pump_savings(
                 intervention=intervention, 
                    avg_gas_percentile=avg_gas_percentile,
                    n_samples=n_samples
            ) 
        elif 'percentile' in intervention:
            samples= self.sample_percentile_savings(
                intervention=intervention, 
                    avg_gas_percentile=avg_gas_percentile,
                    n_samples=n_samples )

        else:
            samples = self._sample_energy_savings(
                intervention=intervention,
                # building_chars=building_chars,
                # typology=typology,
                # age_band=age_band,
                # region=region,
                # regional_multiplier=regional_mult,
                # age_multiplier=age_mult,
                # complexity_multiplier=complexity_mult,
                n_samples=n_samples
            )
        
        return samples
    
    def calculate_solar_pv_impact_monte_carlo(
        self,
        region: str,
        scaled_roof_size: float,
        n_samples: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate solar PV impact using Monte Carlo simulation.
        
        Parameters:
        -----------
        region : str
            Region code: LN, WM, EM, EE, SE, SW, NE, NW, WA, YH
        scaled_roof_size : float
            Roof size in square meters (used to derive system size)
        n_samples : int
            Number of Monte Carlo samples to generate (default: 1000)
            
        Returns:
        --------
        Dict with arrays of sampled values for each metric
        """
        # Define available system sizes and their roof requirements
        roof_to_system = {
            5: 1.0,
            10: 2.0,
            15: 3.0,
            20: 4.0,
            25: 5.0,
            30: 6.0
        }
        
        # Find the closest roof size (snap to nearest option)
        available_roof_sizes = np.array(list(roof_to_system.keys()))
        closest_roof_size = available_roof_sizes[np.argmin(np.abs(available_roof_sizes - scaled_roof_size))]
        # system_size_kwp = roof_to_system[closest_roof_size]
        
        # Get uncertainty parameters
        min_val = self.energysaving_uncertainty_parameters['solar_pv']['kwh_per_m']['min']
        mode_val = self.energysaving_uncertainty_parameters['solar_pv']['kwh_per_m']['mode']
        max_val = self.energysaving_uncertainty_parameters['solar_pv']['kwh_per_m']['max']
        
        # Sample from triangular distribution n_samples times
        base_kwh_per_m = np.random.triangular(min_val, mode_val, max_val, size=n_samples)
        
        # Get regional multiplier (default to 1.0 if not found)
        multiplier = self.regional_multipliers.get(region.upper(), 1.0)
        
        # Apply multiplier to base rate (vectorized)
        adjusted_kwh_per_m = base_kwh_per_m * multiplier
        
        # Calculate annual generation for all samples (vectorized)
        annual_generation_kwh = adjusted_kwh_per_m * closest_roof_size
        
        # Return arrays of samples
        return {
            'annual_generation_kwh': annual_generation_kwh,
            
            'adjusted_kwh_per_m': adjusted_kwh_per_m,
            'regional_multiplier': np.full(n_samples, multiplier),
            'matched_roof_size': np.full(n_samples, closest_roof_size)  # Show which size was matched
    
    
        }
    
   
    def sample_percentile_savings(
        self, 
        intervention: str, 
        avg_gas_percentile: str,  # or int, depending on your data structure
        n_samples: int = 1000
    ) -> np.ndarray:
        """Sample savings from normal distribution for given intervention and percentile."""
        
        if not str(avg_gas_percentile).isnumeric():
            raise ValueError(f'Percentile must be numeric, got: {avg_gas_percentile}')
        
        # Check if intervention/percentile exists in data
        if intervention not in self.energysaving_uncertainty_parameters:
            raise KeyError(f'No data for intervention: {intervention}')
        
        # Convert to int for dict lookup
        percentile_key = int(avg_gas_percentile)
        
        if percentile_key not in self.energysaving_uncertainty_parameters[intervention]['gas']:
            raise KeyError(f'No data for percentile: {percentile_key}')
        
        dist_params = self.energysaving_uncertainty_parameters[intervention]['gas'][percentile_key]
        
        # Sample from normal distribution
        savings = np.random.normal(
            dist_params['mean'], 
            dist_params['sd'], 
            size=n_samples
        )

        savings = np.clip(savings, a_min= -1,  a_max = 1)
        
        return savings
        



    def _sample_from_distribution(self,
                                dist_params: Dict[str, float],
                                dist_type: str,
                                n_samples: int,
                                random_state: np.random.RandomState) -> np.ndarray:
        """
        Helper function to sample from a distribution, handling edge cases.
        
        Parameters:
        -----------
        dist_params : dict
            Dictionary with 'min', 'mode', 'max' keys (or 'min', 'max' for uniform)
        dist_type : str
            Type of distribution ('triangular' or 'uniform')
        n_samples : int
            Number of samples to generate
        random_state : np.random.RandomState
            Random state for reproducibility
            
        Returns:
        --------
        np.ndarray : Array of sampled values
        """
        if dist_type == 'triangular':
            min_val = dist_params['min']
            mode_val = dist_params['mode']
            max_val = dist_params['max']
            
            return random_state.triangular(
                left=min_val,
                mode=mode_val,
                right=max_val,
                size=n_samples
            )
            
        elif dist_type == 'uniform':
            min_val = dist_params['min']
            max_val = dist_params['max']
         
            return random_state.uniform(
                low=min_val,
                high=max_val,
                size=n_samples
            )
        else:
            raise ValueError(f'Unknown distribution type: {dist_type}')
        
    def _sample_energy_savings(self, 
                          intervention: str, 
                          n_samples: int = 1,
                          random_state: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
        """
        Sample energy savings for single intervention from uncertainty distribution.
        
        Parameters:
        -----------
        intervention : str
            Name of the intervention
        n_samples : int
            Number of samples to generate
        random_state : np.random.RandomState, optional
            Random state for reproducibility
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary with 'gas' and 'electricity' keys, 
                            each containing array of savings percentages 
                            (as decimals, e.g., 0.15 for 15%)
        """
        # Check that this is not solar_pv
        if intervention == 'solar_pv':
            raise ValueError(f"This function should not be used for solar_pv intervention")
        
        # Initialize random_state if None
        if random_state is None:
            random_state = np.random.RandomState()
        
        params = self.energysaving_uncertainty_parameters.get(intervention, {})
        
        if not params:
            raise ValueError(f"No uncertainty parameters found for intervention: {intervention}")
        
        dist_type = params.get('distribution', 'uniform')
        
        # Check if gas and electricity parameters exist
        if 'gas' not in params:
            raise ValueError(f"Intervention {intervention} missing 'gas' parameters")
        if 'electricity' not in params:
            raise ValueError(f"Intervention {intervention} missing 'electricity' parameters")
        
        results = {}
        
        # Sample for gas
        gas_params = params['gas']
        results['gas'] = self._sample_from_distribution(
                gas_params, dist_type, n_samples, random_state
            )
        # Handle electricity - skip sampling if all zeros
        elec_params = params['electricity']
        if elec_params['min'] == 0.0 and elec_params['max'] == 0.0:
            # No electricity savings, return zeros without sampling
            results['electricity'] = np.zeros(n_samples)
        else:
            # Sample for electricity
            results['electricity'] = self._sample_from_distribution(
                elec_params, dist_type, n_samples, random_state
            )
        
   
        return results
    