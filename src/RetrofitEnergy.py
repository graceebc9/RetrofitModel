import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
 
from .BuildingCharacteristics import BuildingCharacteristics 

@dataclass
class RetrofitEnergy:
    """Enhanced configuration with with Monte Carlo cost sampling.."""
    solar_regional_multiplier: Dict[str, float] = field(default_factory=lambda: {
            # high sun regions 
            'SW': 1.20,  # South West            
            'SE': 1.20,            
            'EE' :1.2,
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
            
 
    energysaving_uncertainty_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
       
  
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
    # update by 10 of cavity 
    'solid_wall_percentile': {
        'distribution': 'normal',
        'gas': {
             0: {'mean': 0.1159182, 'sd': 0.89226159},
            1: {'mean':    0.02229019, 'sd': 0.5013025},
            2: {'mean': -0.0254804, 'sd': 0.36765047},
            3: {'mean': -0.0570631, 'sd': 0.32145214},
            4: {'mean': -0.0828196, 'sd': 0.21262479},
            5: {'mean': -0.107262, 'sd': 0.19887291},
            6: {'mean': -0.1297073, 'sd': 0.19147565},
            7: {'mean': -0.1496037, 'sd': 0.19498504},
            8: {'mean': -0.1691421, 'sd': 0.20694392},
            9: {'mean': -0.1911919, 'sd': 0.37280598},
        }
    },
    # "Loft_percentile_minmax": {
    #     "distribution": "minmax",
    #     "gas": {
    #         0: {"max": 15.6197738, "min": 7.96985019},      # 0-10th
    #         1: {"max": 4.44876601, "min": 0.2185924},       # 10th-20th
    #         2: {"max": -0.7870848, "min": -4.154916},       # 20th-30th
    #         3: {"max": -3.3577185, "min": -6.1747239},      # 30th-40th
    #         4: {"max": -5.0186254, "min": -7.5800992},      # 40th-50th
    #         5: {"max": -6.5975416, "min": -9.0266738},      # 50th-60th
    #         6: {"max": -8.0975166, "min": -10.491193},      # 60th-70th
    #         7: {"max": -9.2581341, "min": -11.719199},      # 70th-80th
    #         8: {"max": -9.7254937, "min": -12.543418},      # 80th-90th
    #         9: {"max": -9.3560783, "min": -12.686518}       # 90th-100th
    #     }
    # },
    # "Cavity_percentile_minmax": {
    #     "distribution": "minmax",
    #     "gas": {
    #         0: {"max": 14.558245, "min": 6.62539434},       # 0-10th
    #         1: {"max": 4.2762953, "min": -0.2235334},       # 10th-20th
    #         2: {"max": -0.6559457, "min": -3.9768618},      # 20th-30th
    #         3: {"max": -3.744087, "min": -6.6310304},       # 30th-40th
    #         4: {"max": -6.2701725, "min": -8.7879331},      # 40th-50th
    #         5: {"max": -8.5736337, "min": -10.928554},      # 50th-60th
    #         6: {"max": -10.657907, "min": -12.925234},      # 60th-70th
    #         7: {"max": -12.445896, "min": -14.754778},      # 70th-80th
    #         8: {"max": -14.151309, "min": -16.6018},        # 80th-90th
    #         9: {"max": -15.767497, "min": -18.994669}       # 90th-100th
    #     }
    # },
                
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
        'internal_wall_insulation': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.132,
                'mode': 0.132,
                'max': 0.68,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'medium'
        },
        'external_wall_insulation': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.132,
                'mode': 0.132,
                'max': 0.68,
            },
            'electricity': {
                'min': 0.0,
                'mode': 0.0,
                'max': 0.0,
            },
            'confidence': 'medium'
        },
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
        'heat_pump_upgrade': {
            'distribution': 'triangular',
            'gas': {
                'min': 0.90,
                'mode': 0.95,
                'max': 0.98,
            },
            'electricity': {
                'min': -0.60,  # Negative = increase
                'mode': -0.50,
                'max': -0.40,
            },
            'confidence': 'medium'
        },
 
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
 
        
       

    def sample_intervention_energy_savings_monte_carlo(self,
                                                   intervention: str,
                                                   building_chars: BuildingCharacteristics,
                                              
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
    