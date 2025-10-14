from scipy import stats
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .BuildingCharacteristics import BuildingCharacteristics  
from .RetrofitCostsScalingRules import INTERVENTION_CONFIGS, CostEstimator, InterventionConfig
from .RetrofitEnergy import RetrofitEnergy 
from .RetrofitUtils import calculate_estimated_flats_per_building 
from .RetrofitConfig import RetrofitConfig 


# Add logger at the top of your module
logger = logging.getLogger(__name__)

@dataclass
class RetrofitModel:
    """Enhanced configuration with with Monte Carlo cost sampling.."""
    retrofit_config: RetrofitConfig 
    n_samples: int = 100
    solid_wall_internal_improvement_factor: float = 0.1 
    solid_wall_external_improvement_factor: float = 0.2 
    energy_config: Optional[RetrofitEnergy] = None  # Allow custom config or will be auto-created
    cost_estimator: CostEstimator = field(default_factory=CostEstimator)
    custom_intervention_configs: Optional[Dict[str, InterventionConfig]] = None


    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        
        if self.n_samples < 100:
            logger.warning(f"Warning: n_samples={self.n_samples} is low. Consider using 100+ for stable results.")

        # Create RetrofitEnergy config if not provided
        if self.energy_config is None:
            self.energy_config = RetrofitEnergy(
                solid_wall_internal_improvement_factor=self.solid_wall_internal_improvement_factor, 
                solid_wall_external_improvement_factor=self.solid_wall_external_improvement_factor
            )
            logger.info(f"Created RetrofitEnergy config with solid wall improvement factor: internal {self.solid_wall_internal_improvement_factor} and external {self.solid_wall_external_improvement_factor}  ")
        
        # Initialize cost estimator with custom configs if provided
        if self.custom_intervention_configs is not None:
            self.cost_estimator = CostEstimator(self.custom_intervention_configs)
            logger.info(f"Initialized CostEstimator with {len(self.custom_intervention_configs)} custom configs")
        else:
            logger.info(f"Using default CostEstimator with {len(self.cost_estimator.configs)} interventions")


        logger.debug(f"Regional multipliers: {list(self.regional_multipliers.keys())}")
        logger.debug(f"Available scenarios: {list(self.retrofit_packages.keys())}")
        logger.info("RetrofitModel initialized successfully")
   
    typologies: List[str] = field(default_factory=lambda: [
        'Medium height flats 5-6 storeys',
        'Small low terraces',
        '3-4 storey and smaller flats',
        'Tall terraces 3-4 storeys',
        'Large semi detached',
        'Standard size detached',
        'Standard size semi detached',
        '2 storeys terraces with t rear extension',
        'Semi type house in multiples',
        'Tall flats 6-15 storeys',
        'Large detached',
        'Very tall point block flats',
        'Very large detached',
        'Planned balanced mixed estates',
        'Linked and step linked premises',
        'Domestic outbuilding',
        'all_unknown_typology',
    ])
    
    age_bands: List[str] = field(default_factory=lambda: [
        'Pre 1919', '1919-1944', '1945-1959', '1960-1979',
        '1980-1989', '1990-1999', 'Post 1999'
    ])
    
    regional_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'LN': 1.25, 'SE': 1.15, 'SW': 1.05, 'NW': 0.95, 'NE': 0.85,
        'YH': 0.90, 'WA': 0.95, 'WM': 0.98, 'EM': 0.95, 'EE': 1.08,
    })
    
    valid_regions: List[str] = field(default_factory=lambda: [
        'LN', 'SE', 'SW', 'NW', 'NE', 'YH', 'WA', 'EM', 'EE', 'WM'
    ])
    
    # scaling_rules: CostEstimator = field(default_factory=INTERVENTION_CONFIGS)
    

    age_band_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Post 1999': 0.90, '1990-1999': 0.95, '1980-1989': 1.0,
        '1960-1979': 1.15, '1945-1959': 1.35, '1919-1944': 1.6, 'Pre 1919': 2.0
    })
    
    typology_complexity: Dict[str, float] = field(default_factory=lambda: {
        'Very tall point block flats': 1.4,
        'Tall flats 6-15 storeys': 1.2,
        'Medium height flats 5-6 storeys': 1.1,
        'Tall terraces 3-4 storeys': 1.1,
    })
    


    retrofit_packages = {
        
         'loft_installation': {
            'name': 'loft',
            'description': 'using percentiles to get savings',
            'interventions': [
                'loft_percentile',
            ],
            'includes_wall_insulation': False,
            'installation_approach': 'simultaneous'
        },
        'wall_installation': {
            'name': 'wall',
            'description': 'using percentiles to get savings',
            'interventions': [
                'WALL_INSULATION',
            ],
            'includes_wall_insulation': True,
            'installation_approach': 'simultaneous'

        },

        'scenario2': {
            'name': 'Scenario 2: envelope only (evans et al)',
            'description': 'Focus on insulation with easy wins',
            'interventions': [
                'WALL_INSULATION',
                'loft_insulation',
                'double_glazing',
            ],
            'includes_wall_insulation': True,
            'installation_approach': 'simultaneous'
        },

        'scenario3': {
            'name': 'envelope and air source heat pump',
            'description': 'Scenario 3 envelope and air source heat pump',
            'interventions': [
                'WALL_INSULATION',
                'loft_insulation',
                'double_glazing',
                'heat_pump_upgrade',
            ],
            'includes_wall_insulation': True,
            'installation_approach': 'simultaneous'
        },

        'scenario5': {
            'name': 'envelope and air heat pump, solar',
            'description': 'scenario 5: envelope, heat pump and solar',
            'interventions': [
                'WALL_INSULATION',
                'loft_insulation',
                'double_glazing',
                'heat_pump_upgrade',
                'solar_pv', 
            ],
            'includes_wall_insulation': True,
            'installation_approach': 'simultaneous'
        },
    
        'deep_retrofit_grouped': {
            'name': 'Deep Retrofit (Net Zero Ready)',
            'description': 'Maximum intervention package',
            'interventions': [
                'deep_retrofit_estimate',
            ],
            'includes_wall_insulation': True,
            'installation_approach': 'simultaneous'
        }
    }
        
    
    def validate_region(self, region: str) -> str:
        """Validate region code."""
        if region not in self.valid_regions:
            raise ValueError(f"Invalid region '{region}'. Valid: {self.valid_regions}")
        return region
    
    def get_regional_multiplier(self, region: str) -> float:
        """Get regional cost multiplier."""
        return self.regional_multipliers[self.validate_region(region)]



    def _validate_inputs(self, df, region, scenario):
        """Validate input parameters for building cost calculations."""
        if df is None or df.empty:
            return {'error': 'DataFrame is None or empty'}
        
        if not region:
            return {'error': 'Region parameter is required'}
        
        if not scenario:
            return {'error': 'Scenario parameter is required'}
        
        if scenario not in self.retrofit_packages:
            return {'error': f'Scenario "{scenario}" not found in config.retrofit_packages'}
        
        return None  # No errors


    def _validate_statistics(self, return_statistics):
        """Validate and return statistics list."""
        if return_statistics is None:
            return ['mean', 'p5', 'p50', 'p95', 'std' ]
        
        valid_statistics = ['mean', 'median', 'std', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']
        invalid_stats = [s for s in return_statistics if s not in valid_statistics]
        
        if invalid_stats:
            return {'error': f'Invalid statistics requested: {invalid_stats}. Valid: {valid_statistics}'}
        
        return return_statistics


    def _get_scenario_interventions(self, scenario):
        """Extract interventions for a given scenario."""
        selected_scenario = self.retrofit_packages[scenario]
        scenario_interventions = selected_scenario.get('interventions', [])
        
        if not scenario_interventions:
            return {'error': f'No interventions defined for scenario "{scenario}"'}
        
        return scenario_interventions


    def _get_column_mapping(self, col_mapping):
        """Get complete column mapping with defaults."""
        default_mapping = {
            'floor_count': 'total_fl_area_avg',
            'gross_external_area': 'total_fl_area_avg',
            'gross_internal_area': 'scaled_fl_area',
            'footprint_circumference': 'perimeter_length',
            'flat_count': 'est_num_flats',
            'building_type': 'premise_type',
            'age_band': 'premise_age',
            'building_footprint_area': 'premise_area',
            'avg_gas_percentile':'avg_gas_percentile', 
            'inferred_wall_type': 'inferred_wall_type',
            'inferred_insulation_type': 'inferred_insulation_type',

        }
        
        if col_mapping is None:
            return default_mapping
        
        # Merge with defaults
        for key, default_val in default_mapping.items():
            if key not in col_mapping:
                col_mapping[key] = default_val
        
        return col_mapping


    def _prepare_dataframe(self, df, col_mapping):
        """Prepare DataFrame by filtering and calculating estimated flats."""
        result_df = df.copy()
        
        # Filter out domestic outbuildings
        result_df = result_df[result_df[col_mapping['building_type']] != 'Domestic outbuilding']
        
        # Calculate estimated flats
        result_df['est_num_flats'] = result_df.apply(
            lambda row: calculate_estimated_flats_per_building(
                building_footprint_area=row['premise_area'],
                typology_col=row['premise_type'],
                floor_count=row['premise_floor_count']
            ),
            axis=1
        )
        
        return result_df


    def _validate_dataframe_columns(self, df, col_mapping):
        """Validate that all required columns exist in DataFrame."""
        # Check mapped columns
        missing_columns = [
            f"{field} -> {col_name}" 
            for field, col_name in col_mapping.items() 
            if col_name not in df.columns
        ]
        
        if missing_columns:
            return {'error': f'Required columns not found: {", ".join(missing_columns)}'}
        
        # Check required boolean/type columns
        required_new_cols = ['wall_insulated', 'existing_loft_insulation', 
                            'existing_floor_insulation', 'existing_window_upgrades', 
                            'inferred_wall_type']
        missing_new_cols = [col for col in required_new_cols if col not in df.columns]
        
        if missing_new_cols:
            return {'error': f'Required retrofit status columns not found: {", ".join(missing_new_cols)}'}
        
        return None  # No errors


    def get_skip_interventions(self, wall_insulated, existing_loft, existing_floor, existing_windows):
        """
        Create set of interventions to skip based on existing retrofit status.
        
        Parameters:
        -----------
        wall_insulated : bool
            True if wall already insulated
        existing_loft : bool
            True if loft already insulated
        existing_floor : bool
            True if floor already insulated
        existing_windows : bool
            True if windows already upgraded
        
        Returns:
        --------
        set: Interventions that should be skipped (cost = 0)
        """
        skip_interventions = set()
        
        if wall_insulated:
            skip_interventions.add('cavity_wall_insulation')
            skip_interventions.add('external_wall_insulation')
            skip_interventions.add('internal_wall_insulation')
            skip_interventions.add('cavity_wall_percentile')
            skip_interventions.add('solid_wall_percentile')
            skip_interventions.add('solid_wall_internal_percentile')
            skip_interventions.add('solid_wall_external_percentile')

        if existing_loft:
            skip_interventions.add('loft_insulation')
            skip_interventions.add('loft_percentile')
        if existing_floor:
            skip_interventions.add('floor_insulation')
        if existing_windows:
            skip_interventions.add('double_glazing')
        
        return skip_interventions
    


    # def resolve_interventions_for_building(self, scenario_interventions, wall_type, prob_external, percentile ):
    #     """
    #     Resolve scenario interventions, replacing WALL_INSULATION placeholder with specific wall type.
        
    #     Parameters:
    #     -----------
    #     scenario_interventions : list
    #         List of interventions from scenario config (may include 'WALL_INSULATION' placeholder)
    #     wall_type : str
    #         Either 'cavity_wall' or 'solid_wall'
    #     prob_external : float
    #         Probability of selecting external wall insulation for solid walls
        
    #     percentile: bool , if true use percentile interventions 
        
    #     Returns:
    #     --------
    #     tuple: (interventions_list, selected_wall_type)
    #         - interventions_list: List of resolved interventions
    #         - selected_wall_type: Specific wall insulation type selected (or None)
    #     """
    #     interventions_to_calculate = []
    #     selected_wall_insulation = None
        
    #     for intervention in scenario_interventions:
    #         if intervention == 'WALL_INSULATION':
    #             if percentile:
    #                 if wall_type == 'cavity_wall':
    #                     selected_wall_insulation = 'cavity_wall_insulation'
    #                     interventions_to_calculate.append('cavity_wall_percentile')
    #                 else:  # solid_wall
    #                     selected_wall_insulation = 'solid_insulation'
    #                     interventions_to_calculate.append('solid_wall_percentile')
    #             else:
    #                 # Replace with specific wall insulation type based on wall_type
    #                 if wall_type == 'cavity_wall':
    #                     selected_wall_insulation = 'cavity_wall_insulation'
    #                     interventions_to_calculate.append('cavity_wall_insulation')
    #                 else:  # solid_wall
    #                     # Randomly choose internal or external wall insulation
    #                     if np.random.random() < prob_external:
    #                         selected_wall_insulation = 'external_wall_insulation'
    #                     else:
    #                         selected_wall_insulation = 'internal_wall_insulation'
    #                     interventions_to_calculate.append(selected_wall_insulation)
    #         else:
    #             interventions_to_calculate.append(intervention)
        
    #     return interventions_to_calculate, selected_wall_insulation
      
 


    def _calculate_single_statistic(self, samples: np.ndarray, stat: str) -> float:
        """
        Calculate a specific statistic from a single numpy array.
        
        Parameters:
        -----------
        samples : np.ndarray
            Array of Monte Carlo samples
        stat : str
            Statistic to calculate ('mean', 'median', 'p50', 'std', 'p5', 'p95', etc.)
        
        Returns:
        --------
        float: Calculated statistic value
        """
        # Ensure we have a numpy array
        if not isinstance(samples, np.ndarray):
            logging.warning(f"samples is not a numpy array, it's a {type(samples)}")
            try:
                samples = np.array(samples)
                
            except Exception as e:
                logging.error(f"Cannot convert samples to numpy array: {e}")
                raise TypeError(f"Cannot convert samples to numpy array: {e}")
    
 
        # Check if array is empty
        if samples.size == 0:
            logging.error("Cannot calculate statistics on empty array")
            raise ValueError("Cannot calculate statistics on empty array")
        

        
        # Calculate the requested statistic
        try:
            if stat == 'mean':
                result = samples.mean()
            elif stat == 'median' or stat == 'p50':
                result = np.median(samples)
            elif stat == 'std':
                result = samples.std()
            elif stat.startswith('p'):
                percentile = int(stat[1:])
                result = np.percentile(samples, percentile)
            else:
                raise ValueError(f"Unknown statistic: {stat}")
            
            
            return result
            
        except AttributeError as e:
            logging.error(f"AttributeError: {e}")
            logging.error(f"This usually means samples is not a proper numpy array")
            raise
        except Exception as e:
            logging.error(f"Error calculating {stat}: {e}")
            raise


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Combination fns at the samples level of monte carlo 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def _calculate_statistics_costs(self, samples, stat: str) -> dict:
        """
        Calculate a specific statistic from Monte Carlo samples.
        
        Parameters:
        -----------
        samples : np.ndarray or dict or list
            Array of Monte Carlo samples, dict with 'gas' and 'electricity' keys,
            or list of such dicts for multiplicative aggregation
        stat : str
            Statistic to calculate ('mean', 'median', 'p50', 'std', 'p5', 'p95', etc.)
        aggregate_gas_multiplicatively : bool
            If True and samples is a list of dicts, multiply gas samples across interventions
        
        Returns:
        --------
        
        """
        
        result = self._calculate_single_statistic(samples, stat)
        return result
        

    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def _calculate_statistics(self, samples, stat: str) -> dict:
        """
        Calculate a specific statistic from Monte Carlo samples.
        
        Parameters:
        -----------
        samples : np.ndarray or dict or list
            Array of Monte Carlo samples, dict with 'gas' and 'electricity' keys,
            or list of such dicts for multiplicative aggregation
        stat : str
            Statistic to calculate ('mean', 'median', 'p50', 'std', 'p5', 'p95', etc.)
   
        Returns:
        --------
        dict: Dictionary with 'gas' and 'electricity' keys containing calculated statistics
            OR float if input was a simple numpy array (for backwards compatibility)
        """
        
        # Handle list of dicts for multiplicative aggregation
        if isinstance(samples, list):
            logger.debug('Samples are a dict. Combine with addition ')
            if not samples:
                raise ValueError("Empty list provided for aggregation")
            
            # Validate all items are dicts with gas and electricity
            for i, item in enumerate(samples):
                if not isinstance(item, dict):
                    raise ValueError(f"Item {i} in list is not a dict")
                if 'gas' not in item or 'electricity' not in item:
                    raise ValueError(f"Item {i} missing 'gas' or 'electricity' keys")
            
            
            combined_gas = np.ones_like(samples[0]['gas'])
            for item in samples:
                combined_gas *= item['gas']
            
            # Add electricity samples (additive)
            combined_electricity = np.zeros_like(samples[0]['electricity'])
            for item in samples:
                combined_electricity += item['electricity']
            
            # Calculate statistics on combined samples
            results = {
                'gas': self._calculate_single_statistic(combined_gas, stat),
                'electricity': self._calculate_single_statistic(combined_electricity, stat)
            }
            
            return results
        
        # Handle dict input (gas and electricity separate)
        elif isinstance(samples, dict):
            logger.debug('Samples are dict.')
            logging.debug(f"Dict keys: {samples.keys()}")
            # Validate dict structure
            if 'gas' not in samples or 'electricity' not in samples:
                logging.error(f"Dict must have 'gas' and 'electricity' keys. Found: {samples.keys()}")
                raise ValueError(f"Dict must have 'gas' and 'electricity' keys. Found: {samples.keys()}")
            
            # Calculate statistics for each fuel type
            results = {
                'gas': self._calculate_single_statistic(samples['gas'], stat),
                'electricity': self._calculate_single_statistic(samples['electricity'], stat)
            }
            
            return results
        
        # Handle simple numpy array input (backwards compatibility)
        else:
            logger.debug('Samples are not a list or dictionary')
            result = self._calculate_single_statistic(samples, stat)
            return result
    
    def calculate_intervention_energy_savings(
        self, 
        interventions, 
        building_chars,
        region,  
        return_statistics,
        roof_scaling,
    ):
        """
        Calculate Monte Carlo energy savings statistics for a list of interventions.
        Produces flat, consistent column naming similar to cost calculations.
        Example output keys:
            energy_wall_installation_gas_mean
            energy_wall_installation_electricity_p95
        """

        # Store samples by fuel type for each intervention
        energy_stats = {}

        for intervention in interventions:
            if intervention != 'solar_pv':
                try:
                    # Get Monte Carlo samples
                    samples = self.energy_config.sample_intervention_energy_savings_monte_carlo(
                        intervention=intervention,
                        building_chars=building_chars,
                        region=region,
                        n_samples=self.n_samples,
                        roof_scaling=roof_scaling,
                    )

                    # Determine fuel-specific samples
                    gas_samples = None
                    elec_samples = None

                    if isinstance(samples, dict):
                        gas_samples = samples.get('gas', None)
                        elec_samples = samples.get('electricity', None)
                    else:
                        # Some configs may return only one array (assume gas)
                        gas_samples = samples

                    # Compute statistics for gas
                    if gas_samples is not None:
                        for stat in return_statistics:
                            if stat == 'mean':
                                val = np.mean(gas_samples)
                            elif stat == 'median' or stat == 'p50':
                                val = np.median(gas_samples)
                            elif stat == 'p5':
                                val = np.percentile(gas_samples, 5)
                            elif stat == 'p95':
                                val = np.percentile(gas_samples, 95)
                            elif stat == 'std':
                                val = np.std(gas_samples)
                            else:
                                raise ValueError(f"Unsupported statistic: {stat}")

                            col_name = f"energy_{intervention}_gas_{stat}"
                            energy_stats[col_name] = val

                    # Compute statistics for electricity
                    if elec_samples is not None:
                        for stat in return_statistics:
                            if stat == 'mean':
                                val = np.mean(elec_samples)
                            elif stat == 'median' or stat == 'p50':
                                val = np.median(elec_samples)
                            elif stat == 'p5':
                                val = np.percentile(elec_samples, 5)
                            elif stat == 'p95':
                                val = np.percentile(elec_samples, 95)
                            elif stat == 'std':
                                val = np.std(elec_samples)
                            else:
                                raise ValueError(f"Unsupported statistic: {stat}")

                            col_name = f"energy_{intervention}_electricity_{stat}"
                            energy_stats[col_name] = val

                except Exception as e:
                    logger.warning(f"Error processing intervention {intervention}: {e}")
                    # Fill with NaNs for expected stats if failure
                    for fuel in ['gas', 'electricity']:
                        for stat in return_statistics:
                            col_name = f"energy_{intervention}_{fuel}_{stat}"
                            energy_stats[col_name] = np.nan

        return energy_stats

    
    # def calculate_intervention_energy_savings(self, 
    #                                 interventions, 
    #                                 building_chars,
    #                                 region,  
    #                                 return_statistics,
    #                                 roof_scaling, 
    #                                 ):
    #     """
    #     Calculate Monte Carlo energy savings statistics for a list of interventions.
    #     For both gas and electricity: combines samples additively at the sample level, then calculates summary stats.

    #         Example output keys:
    #     energy_wall_installation_gas_mean
    #     energy_wall_installation_electricity_p95


    #     """
    #     # Store samples by fuel type
    #     gas_samples_list = []
    #     electricity_samples_list = []

    #     for intervention in interventions:
    #         if intervention != 'solar_pv':
    #             try:
    #                 # Get Monte Carlo samples for energy savings
    #                 samples = self.energy_config.sample_intervention_energy_savings_monte_carlo(
    #                     intervention=intervention,
    #                     building_chars=building_chars,
    #                     region=region,
    #                     n_samples=self.n_samples,
    #                     roof_scaling=roof_scaling,
                        
    #                 )
                    
    #                 if 'percentile' in intervention:
    #                     # these only return gas samples 
    #                     gas_samples_list.append(samples)
    #                 else:
    #                     # Separate gas and electricity samples
    #                     if 'gas' in samples:
    #                         gas_samples_list.append(samples['gas'])
    #                     elif 'electricity' in samples:
    #                         electricity_samples_list.append(samples['electricity'])
    #                     else:
    #                         logger.info('samples seem strange ')
    #                         logger.info( samples)
                        
    #                         raise Exception('Samples not as expected')
         
    #             except Exception as e:
    #                 # Handle errors appropriately
    #                 logger.warning(f"Error processing intervention {intervention}: {e}")
    #                 continue
        
    #     savings_stats = {}
        
    #     # Process gas samples: additive combination at sample level
    #     if gas_samples_list:
    #         # Add samples element-wise across all interventions
    #         combined_gas_samples = gas_samples_list[0].copy()
    #         for gas_samples in gas_samples_list[1:]:
    #             combined_gas_samples += gas_samples
            
    #         # Calculate summary statistics on combined samples
    #         savings_stats['gas'] = {
    #             'mean': np.mean(combined_gas_samples),
    #             'median': np.median(combined_gas_samples),
    #             'std': np.std(combined_gas_samples),
    #             'p5': np.percentile(combined_gas_samples, 5),
    #             'p95': np.percentile(combined_gas_samples, 95),
    #         }
         
        
    #     # Process electricity samples: additive combination at sample level
    #     if electricity_samples_list:
    #         # Add samples element-wise across all interventions
    #         combined_electricity_samples = electricity_samples_list[0].copy()
    #         for elec_samples in electricity_samples_list[1:]:
    #             combined_electricity_samples += elec_samples
            
    #         savings_stats['electricity'] = {
    #             'mean': np.mean(combined_electricity_samples),
    #             'median': np.median(combined_electricity_samples),
    #             'std': np.std(combined_electricity_samples),
    #             'p5': np.percentile(combined_electricity_samples, 5),
    #             'p95': np.percentile(combined_electricity_samples, 95),
    #         }
        
    #     return savings_stats
     

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Interventiosn costs
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def calculate_intervention_costs(self, interventions, skip_interventions, building_chars, 
                                    typology, age_band, region, 
                                    return_statistics, include_total=True):
        """
        Calculate Monte Carlo cost statistics for a list of interventions.
        -NEW with totals added within monte carlo framewrok. earlier sampling assumes costs are independnat 
        
        Parameters:
        -----------
        interventions : list
            List of intervention names to calculate costs for
        skip_interventions : set
            Set of interventions to skip (already installed)
        building_chars : BuildingCharacteristics
            Building characteristics object
        typology : str
            Building typology
        age_band : str
            Building age band
        region : str
            Regional code
        return_statistics : list
            Statistics to calculate (e.g., ['mean', 'p5', 'p50', 'p95'])
        include_total : bool
            Whether to calculate statistics for total cost across all interventions
        
        Returns:
        --------
        dict: Cost statistics for each intervention (and optionally total)
        """
        cost_stats = {}
        all_samples = [] if include_total else None

 
        
        for intervention in interventions:
            # Check if interv(ention is already installed
            if intervention in skip_interventions:
                # Set cost to 0 for already-installed interventions
                for stat in return_statistics:
                    col_name = f'{intervention}_{stat}'
                    cost_stats[col_name] = 0.0
                
                # Add zeros to total calculation
                if include_total:
                    all_samples.append(np.zeros(self.n_samples))
                continue
            
            try:
              
                # Get Monte Carlo samples
                samples = self.sample_intervention_cost_monte_carlo(
                    intervention=intervention,
                    building_chars=building_chars,
                    typology=typology,
                    age_band=age_band,
                    region=region,
                )
                 
               
                
                # Calculate individual intervention statistics
                for stat in return_statistics:
              
                    col_name = f'{intervention}_{stat}'
                    try:
                        cost_stats[col_name] = self._calculate_single_statistic(samples, stat)
                    except ValueError as stat_error:
                        logging.error(f"Invalid statistic '{stat}' for {intervention}: {stat_error}")
                        cost_stats[col_name] = np.nan
                
                # Store samples for total calculation
                if include_total:
                    all_samples.append(samples)
                
            except ValueError as ve:
                # logging.warning(f"Cost calculation skipped for {intervention}: {ve}")
                for stat in return_statistics:
                    col_name = f'{intervention}_{stat}'
                    cost_stats[col_name] = np.nan
                
                # Add NaN array for total (or skip - your choice)
                if include_total:
                    all_samples.append(np.full(self.n_samples, np.nan))
                    
            except Exception as e:
                logging.error(f"Unexpected error calculating {intervention}: {e}")
                for stat in return_statistics:
                    col_name = f'{intervention}_{stat}'
                    cost_stats[col_name] = np.nan
                
                if include_total:
                    all_samples.append(np.full(self.n_samples, np.nan))
        
        # Calculate total statistics
        if include_total and all_samples:
            # Sum samples element-wise across all interventions
            total_samples = np.sum(all_samples, axis=0)
            
            # Calculate statistics on the total
            for stat in return_statistics:
                col_name = f'total_{stat}'
                try:
                    cost_stats[col_name] = self._calculate_single_statistic(total_samples, stat)
                except ValueError as stat_error:
                    logging.error(f"Invalid statistic '{stat}' for total: {stat_error}")
                    cost_stats[col_name] = np.nan
        
        return cost_stats

    

    
    def sample_intervention_cost_monte_carlo(self,
                                            intervention: str,
                                            building_chars: BuildingCharacteristics,
                                            typology: str,
                                            age_band: str,
                                            region: str,
                                             ) -> np.ndarray:
        """
        Sample intervention costs using Monte Carlo simulation. for single intervention 
        
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
 
            
        Returns:
        --------
        np.ndarray : Array of sampled costs
        
        Example:
        --------
      
        >>> building = BuildingCharacteristics(
        ...     floor_count=3,
        ...     gross_external_area=200,
        ...     gross_internal_area=180,
        ...     footprint_circumference=40,
        ...     building_footprint_area=100
        ... )
        >>> samples = config.sample_intervention_cost_monte_carlo(
        ...     'loft_insulation',
        ...     building,
        ...     'Standard size semi detached',
        ...     '1960-1979',
        ...     'LN',
        ...     n_samples=1000
        ... )
 
        """
        if typology is None or typology == 'None':
            return None
        
        validated_region = self.validate_region(region)
        
        # Get multipliers
        age_mult = self.age_band_multipliers.get(age_band, 1.0)
        complexity_mult = self.typology_complexity.get(typology, 1.0)
        regional_mult = self.get_regional_multiplier(validated_region)
        
        logger.debug(
            f"Sampling {intervention}: region={validated_region}, "
            f"age_mult={age_mult:.2f}, complexity_mult={complexity_mult:.2f}, "
            f"regional_mult={regional_mult:.2f}"
        )

        try:
            samples = self.cost_estimator.sample_intervention_cost(
                intervention=intervention,
                building_chars=building_chars,
                typology=typology,
                age_band=age_band,
                region=region,
                regional_multiplier=regional_mult,
                age_multiplier=age_mult,
                complexity_multiplier=complexity_mult,
                n_samples=self.n_samples
            )
            
            logger.debug(
                f"{intervention} samples: mean=£{samples.mean():,.0f}, "
                f"std=£{samples.std():,.0f}"
            )
            
            return samples
            
        except ValueError as e:
            logger.error(f"Error sampling {intervention}: {e}")
            raise

        # # Sample using unified method
        # samples = self.scaling_rules.sample_intervention_cost(
        #     intervention=intervention,
        #     building_chars=building_chars,
        #     typology=typology,
        #     age_band=age_band,
        #     region=region,
        #     regional_multiplier=regional_mult,
        #     age_multiplier=age_mult,
        #     complexity_multiplier=complexity_mult,
        #     n_samples=self.n_samples
        # )
        
        return samples
     


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # #   Energy Savings      # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 







    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # #   Row Calc      # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



    def calculate_ONLY_row_costs_only(self,
                                    row,
                                        col_mapping, 
                                        scenario_interventions, 
                                        # prob_external, 
                                        region, 
                                        return_statistics):
        """
        Calculate Monte Carlo cost statistics for scenario interventions for one building.
        and now energy 
       
        """
        # Data validation
        required_cols = ['floor_count', 'gross_external_area', 'gross_internal_area', 'inferred_wall_type', 'inferred_insulation_type',
                        'footprint_circumference', 'building_type', 'age_band', 'building_footprint_area', 'avg_gas_percentile']
        
        missing_cols = [col for col in required_cols if col_mapping[col] not in row.index]
        if missing_cols:
            raise ValueError(f'Missing columns: {missing_cols}')
        
        # Convert and validate building characteristics
        floor_count = int(row[col_mapping['floor_count']])
        gross_external_area = float(row[col_mapping['gross_external_area']])
        gross_internal_area = float(row[col_mapping['gross_internal_area']])
        footprint_circumference = float(row[col_mapping['footprint_circumference']])
        building_footprint_area = float(row[col_mapping['building_footprint_area']])
        avg_gas_percentile = int(row[col_mapping['avg_gas_percentile']])
        # Use max(1) for flat count to ensure valid input to cost calcs
        raw_flat_count = row.get(col_mapping['flat_count'])
        flat_count = int(raw_flat_count) if pd.notna(raw_flat_count) and raw_flat_count > 0 else 1
        
        if any([
            pd.isna(floor_count) or floor_count <= 0,
            pd.isna(gross_external_area) or gross_external_area <= 0,
            pd.isna(gross_internal_area) or gross_internal_area <= 0,
            pd.isna(footprint_circumference) or footprint_circumference <= 0,
            pd.isna(building_footprint_area) or building_footprint_area <= 0
        ]):
            raise ValueError('Invalid building characteristics')
        
        building_chars = BuildingCharacteristics(
            floor_count=floor_count,
            gross_external_area=gross_external_area,
            gross_internal_area=gross_internal_area,
            footprint_circumference=footprint_circumference,
            flat_count=flat_count,
            building_footprint_area=building_footprint_area,
            avg_gas_percentile=avg_gas_percentile,
        )
        
        run_percentile= True 

        # 'wall_insulated', 'existing_loft_insulation', 'existing_floor_insulation', 'existing_window_upgrades'
        # Extract retrofit status flags
        wall_insulated = bool(row['wall_insulated'])
        existing_loft = bool(row['existing_loft_insulation'])
        existing_floor = bool(row['existing_floor_insulation'])
        existing_windows = bool(row['existing_window_upgrades'])
        wall_type = str(row['inferred_wall_type']).lower().strip()
        insulation_type = str(row['inferred_insulation_type']).lower().strip()
        
        # Validate wall_type
        if wall_type not in ['cavity_wall', 'solid_wall']:
            raise ValueError(f"Invalid wall_type: '{wall_type}'. Must be 'cavity_wall' or 'solid_wall'")
        if insulation_type not in ['cavity_wall_insulation', 'internal_wall_insulation', 'external_wall_insulation']: 
            raise ValueError("Invalid insulation type ")
        
        # Resolve interventions (replaces WALL_INSULATION placeholder)
        selected_wall_insulation = insulation_type
        interventions_to_calculate= [] 
        
        for intervention in scenario_interventions:
            if intervention=='WALL_INSULATION':
                if selected_wall_insulation =='cavity_wall_insulation': 
                    interventions_to_calculate.append('cavity_wall_percentile')
                elif selected_wall_insulation =='internal_wall_insulation':  # solid_wall
                        interventions_to_calculate.append('solid_wall_internal_percentile')
                elif selected_wall_insulation =='external_wall_insulation':  # solid_wall
                    interventions_to_calculate.append('solid_wall_external_percentile')
            else:
                interventions_to_calculate.append(intervention)


        # interventions_to_calculate, selected_wall_insulation = self.resolve_interventions_for_building(
        #     scenario_interventions, wall_type, prob_external, percentile = run_percentile
        # )
        
        # Determine which interventions to skip
        skip_interventions = self.get_skip_interventions(
            wall_insulated, existing_loft, existing_floor, existing_windows
        )
        
        # Calculate costs for all interventions
        typology = row[col_mapping['building_type']]
        age_band = row[col_mapping['age_band']]
        
        

        cost_stats = self.calculate_intervention_costs(
            interventions=interventions_to_calculate,
            skip_interventions=skip_interventions,
            building_chars=building_chars,
            typology=typology,
            age_band=age_band,
            region=region,
            return_statistics=return_statistics
        )
             
        # Add prefixes to cost and energy statistics
        cost_stats_prefixed = {f'cost_{key}': value for key, value in cost_stats.items()}
        
        
        cost_result = pd.Series(
            cost_stats_prefixed,   
           )
 
        energy_stats = self.calculate_intervention_energy_savings(
            interventions=interventions_to_calculate,
        
            building_chars=building_chars,
            # typology=typology,
            # age_band=age_band,
            region=region,
            return_statistics=return_statistics,
            roof_scaling=self.retrofit_config.existing_intervention_probs['roof_scaling_factor'] , 
            

        )
        
 
         # Add selected wall type to results
        cost_stats['selected_wall_insulation_type'] = selected_wall_insulation
 
   
        
        energy_stats_prefixed = {f'{key}': value for key, value in energy_stats.items()}
        energy_result = pd.Series(energy_stats_prefixed )
        logger.debug('calculate_ONLY_row_costs_only complete')
        return cost_result, energy_result

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # #   Calc all costs and energy fn        # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    def _calculate_and_add_costs(self, 
                                 result_df,
                                  col_mapping,
                                    scenario_interventions, 
                                # prob_external,
                                  region,
                                   scenario,
                                     return_statistics):
        """Calculate costs  and add them to the DataFrame."""
        
        energy_res = result_df.copy() 
        
        
        # Apply cost calculations to all rows, this calc the total at sample time 
        results  = result_df.apply(
            lambda row: self.calculate_ONLY_row_costs_only(row, 
                                                            col_mapping, 
                                                            scenario_interventions, 
                                                            # prob_external,
                                                            region,  
                                                            return_statistics
                                                             ), axis=1
                                    )
        
        # Unpack the results - each element is a tuple of (cost_series, energy_series)
        cost_results = pd.DataFrame([x[0] for x in results])
        energy_results = pd.DataFrame([x[1] for x in results])
        
        self._add_cost_columns(result_df, cost_results, scenario)
        # Add energy columns (individual interventions)
        self._add_individual_energy_columns(energy_res, energy_results, scenario)
        logger.debug('_calculate_and_add_costs Added costs total complete for all rows.')
        return result_df, energy_res



    def _add_cost_columns(self, result_df, cost_results, scenario):
        """Add individual and total cost columns to result DataFrame."""
        # Add individual intervention cost columns
 
        for col in cost_results.columns:
      
            result_df[f'{scenario}_{col}'] = cost_results[col]
        
  

    def _add_individual_energy_columns(self, result_df, energy_results, scenario):
        """Add individual intervention energy columns to result DataFrame."""
 
        for col in energy_results.columns:
     
            result_df[f'{scenario}_{col}'] = energy_results[col]


    def _add_aggregated_energy_columns(self, result_df, energy_results, scenario, return_statistics):
        """Add aggregated gas and electricity energy columns using combined_savings columns."""
        # Add gas energy columns (using combined_savings from multiplicative aggregation)
        self._add_gas_energy_totals(result_df, energy_results, scenario, return_statistics)
        
        # Add electricity energy columns (using combined_savings from additive aggregation)
        self._add_electricity_energy_totals(result_df, energy_results, scenario, return_statistics)


    def _add_gas_energy_totals(self, result_df, energy_results, scenario, return_statistics):
        """Calculate and add total gas energy using combined_savings columns."""
        for stat in return_statistics:
            # Look for the combined_savings column for this statistic
            combined_col = f'combined_savings_{stat}_gas'
            
            if combined_col in energy_results.columns:
                # Use the pre-calculated combined savings (multiplicative aggregation already done)
                result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[combined_col]
            else:
                # Fallback: if no combined column exists (e.g., only 1 intervention or solar only)
                # Look for individual gas columns for this statistic
                gas_stat_cols = [col for col in energy_results.columns 
                            if f'_{stat}_gas' in col.lower() and 'combined' not in col.lower()]
                
                if len(gas_stat_cols) == 1:
                    # Single intervention: just use its value
                    result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[gas_stat_cols[0]]
                elif len(gas_stat_cols) > 1:
                    # Multiple interventions but no combined column: multiply them
                    # (This shouldn't happen with new structure, but kept for safety)
                    result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[gas_stat_cols].prod(axis=1)
                else:
                    # No gas interventions
                    result_df[f'energy_{scenario}_gas_{stat}'] = 1.0


    def _add_electricity_energy_totals(self, result_df, energy_results, scenario, return_statistics):
        """Calculate and add total electricity energy using combined_savings columns."""
        for stat in return_statistics:
            # Look for the combined_savings column for this statistic
            combined_col = f'combined_savings_{stat}_electricity'
            
            if combined_col in energy_results.columns:
                # Use the pre-calculated combined savings (additive aggregation already done)
                result_df[f'energy_{scenario}_electricity_{stat}'] = energy_results[combined_col]
            else:
                # Fallback: if no combined column exists
                # Look for individual electricity columns for this statistic
                elec_stat_cols = [col for col in energy_results.columns 
                                if f'_{stat}_electricity' in col.lower() and 'combined' not in col.lower()]
                
                if len(elec_stat_cols) == 1:
                    # Single intervention: just use its value
                    result_df[f'energy_{scenario}_electricity_{stat}'] = energy_results[elec_stat_cols[0]]
                elif len(elec_stat_cols) > 1:
                    # Multiple interventions but no combined column: add them
                    # (This shouldn't happen with new structure, but kept for safety)
                    result_df[f'energy_{scenario}_electricity_{stat}'] = energy_results[elec_stat_cols].sum(axis=1)
                else:
                    # No electricity interventions
                    result_df[f'energy_{scenario}_electricity_{stat}'] = 0.0


    def _add_default_electricity(self, result_df, energy_results, elec_cols, scenario, return_statistics):
        """Default behavior: Sum electricity additively for other scenarios."""
        for stat in return_statistics:
            elec_intervention_cols = [col for col in elec_cols if col.endswith(f'_{stat}')]
            if elec_intervention_cols:
                result_df[f'energy_{scenario}_elec_{stat}'] = energy_results[elec_intervention_cols].sum(axis=1)

    def _get_cols_scenario_intervention(self, scenario_str, stats = ['mean', 'std', 'p5', 'p50', 'p95']):
        cost_cols= [] 
        energy_cols= [] 
        if scenario_str == 'wall_installation':
            interventions = ['cavity_wall_percentile', 'solid_wall_internal_percentile', 'solid_wall_external_percentile' ]
            elec=False 
        elif scenario_str =='loft_installation':
            interventions = ['loft_percentile']
            elec=False 
        else:
            raise Exception(f'Need to define the interventions  for scenarioi  ({scenario_str}) in RetrofitModel _get_cols_scenario_intervention')
        
        for iint in interventions:
            for s in stats:
                cost_cols.append( f'{scenario_str}_cost_{iint}_{s}' ) 
                energy_cols.append(  f'{scenario_str}_energy_{iint}_gas_{s}' ) 
                if elec:
                    energy_cols.append(  f'{scenario_str}_energy_{iint}_electricity_{s}' ) 
        
        return cost_cols,  energy_cols


    def _ensure_columns_exist(self, df, required_cols):
        """
        Ensure all required columns exist in the DataFrame.
        If any are missing, create them with NaN values.
        """
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # #   Calc all costs and energy fn        # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def calculate_building_costs_df_updated(self,
                                             df, 
                                             region, 
                                             scenario, 
                                            # prob_external, 
                                        col_mapping=None, 
                                        return_statistics=None):
        """
        Apply Monte Carlo building cost calculations to all rows in a DataFrame for a specific scenario.
        
        Main orchestrator function that coordinates validation, preparation, and cost calculations.
        """
        def expand_dict_columns(df):
            df_expanded = df.copy()
            
            for col in df.columns:
                if isinstance(df[col].iloc[0], dict):
                    # Expand dictionary column
                    temp_df = df[col].apply(pd.Series)
                    temp_df.columns = [f"{col}_{subcol}" for subcol in temp_df.columns]
                    df_expanded = pd.concat([df_expanded.drop(columns=[col]), temp_df], axis=1)
            
            return df_expanded
        # Validate inputs
        error = self._validate_inputs(df, region, scenario)
        if error:
            return error
        
        # Validate and get statistics
        return_statistics = self._validate_statistics(return_statistics)
        if isinstance(return_statistics, dict) and 'error' in return_statistics:
            return return_statistics
        
        # Get scenario interventions
        scenario_interventions = self._get_scenario_interventions(scenario)
        if isinstance(scenario_interventions, dict) and 'error' in scenario_interventions:
            return scenario_interventions
        
        # Get column mapping
        col_mapping = self._get_column_mapping(col_mapping)
        
        # Prepare DataFrame
        result_df = self._prepare_dataframe(df, col_mapping)
        # Show first few rows with better formatting
        # logger.debug("DataFrame preview:\n%s", result_df.head().to_string())

        # Validate DataFrame columns
        error = self._validate_dataframe_columns(result_df, col_mapping)
        if error:
            return error
        
        costs_result_df  = result_df.copy() 
        energy_results_df = result_df.copy() 
        dfcols = result_df.columns.tolist()
         
        # Calculate and add costs  
        costs_result_df, energy_results_df = self._calculate_and_add_costs(result_df = costs_result_df, 
                                                                            col_mapping = col_mapping, 
                                                                            scenario_interventions = scenario_interventions, 
                                                                            # prob_external = prob_external,
                                                                            region =region, 
                                                                            scenario = scenario,
                                                                            return_statistics = return_statistics
        )

        # check if wall type solid then no cavity wall cost 

 
        # extra_cols = ['wall_insulated', 'existing_loft_insulation', 'existing_floor_insulation', 'existing_window_upgrades']
        cost_cols, energy_cols = self._get_cols_scenario_intervention(scenario )
  
        
        # ✅ Check for overlap between expected and actual columns
        cost_overlap = set(cost_cols).intersection(costs_result_df.columns)
        energy_overlap = set(energy_cols).intersection(energy_results_df.columns)

        if not cost_overlap:
            logger.warning(f"No overlap found between expected cost columns and DataFrame columns for scenario {scenario}.")
            logger.warning(f"Expected cost cols: {cost_cols}")
            logger.warning(f"Actual cost DF cols: {[x for x in costs_result_df.columns.tolist() if x not in dfcols ] }")

        if not energy_overlap:
            logger.warning(f"No overlap found between expected energy columns and DataFrame columns for scenario {scenario}.")
            logger.warning(f"Expected energy cols: {energy_cols}")
            logger.warning(f"Actual energy DF cols: {[x for x in energy_results_df.columns.tolist() if x not in dfcols ] }")

 

        costs_result_df = self._ensure_columns_exist(costs_result_df, cost_cols)
        energy_results_df = self._ensure_columns_exist(energy_results_df, energy_cols)
        energy_results_df = expand_dict_columns(energy_results_df)

        # if 'geometry' in energy_results_df.columns:
        #     energy_results_df = energy_results_df.drop(columns=['geometry'])
        # Safe concatenation
        data = pd.concat(
            [costs_result_df[ cost_cols ], energy_results_df[energy_cols]],
            axis=1
                )       
        return  data






    # def calculate_intervention_energy_savings(self, 
    #                                       interventions, 
    #                                       building_chars,
    #                                     # typology, 
    #                                     # age_band, 
    #                                     region,  
    #                                     return_statistics,
    #                                     roof_scaling):
    #     """
    #     Calculate Monte Carlo energy savings statistics for a list of interventions.
        
    #     Parameters:
    #     -----------
    #     interventions : list
    #         List of intervention names to calculate energy savings for
        
    #     building_chars : BuildingCharacteristics
    #         Building characteristics object
    #     typology : str
    #         Building typology
    #     age_band : str
    #         Building age band
    #     region : str
    #         Regional code
    
    #     n_samples : int
    #         Number of Monte Carlo samples
    #     return_statistics : list
    #         Statistics to calculate (e.g., ['mean', 'p5', 'p50', 'p95'])
        
    #     Returns:
    #     --------
    #     dict: Energy savings statistics for each intervention
    #     """
    #     savings_stats = {}
        
    #     # Store gas samples for multiplicative combination
    #     gas_samples_list = []

    #     for intervention in interventions:
    #         try:
    #             # Get Monte Carlo samples for energy savings
    #             samples = self.energy_config.sample_intervention_energy_savings_monte_carlo(
    #                 intervention=intervention,
    #                 building_chars=building_chars,
    #                 # typology=typology,
    #                 # age_band=age_band,
    #                 region=region,
    #                 n_samples=self.n_samples,
    #                 roof_scaling = roof_scaling,
    #             )
    #             # need to process solar samples different 
    #             if intervention =='solar_pv':
    #                 # Calculate statistics for different metrics
    #                 for stat in return_statistics:
    #                     # For annual generation and adjusted kwh, calculate the requested statistic
    #                     savings_stats[f'{intervention}_annual_generation_{stat}'] = \
    #                         self._calculate_statistics(samples['annual_generation_kwh'], stat)
                        
    #                     savings_stats[f'{intervention}_adjusted_kwh_per_kwp_{stat}'] = \
    #                         self._calculate_statistics(samples['adjusted_kwh_per_kwp'], stat)
            
    #                     # For matched roof size and regional multiplier, always use mode (they're constant across samples)
    #                     savings_stats[f'{intervention}_matched_roof_size_mode'] = \
    #                         stats.mode(samples['matched_roof_size'], keepdims=False)[0]
                        
    #                     savings_stats[f'{intervention}_regional_multiplier_mode'] = \
    #                         stats.mode(samples['regional_multiplier'], keepdims=False)[0]
        
    #             else:
    #                 # Store gas samples if they exist
    #                 if isinstance(samples, dict) and 'gas' in samples:
    #                     gas_samples_list.append(samples['gas'])
                    
    #                 # Calculate requested statistics
    #                 for stat in return_statistics:
    #                     try:
    #                         result = self._calculate_statistics(samples, stat)
                            
    #                         # Handle dict result (gas and electricity separate)
    #                         if isinstance(result, dict):
    #                             savings_stats[f'{intervention}_savings_{stat}_gas'] = result['gas']
    #                             savings_stats[f'{intervention}_savings_{stat}_electricity'] = result['electricity']
    #                         else:
    #                             # Backwards compatibility: single value result
    #                             savings_stats[f'{intervention}_savings_{stat}'] = result
                                
    #                     except ValueError as stat_error:
    #                         logging.error(f"Invalid statistic '{stat}' for {intervention}: {stat_error}")
    #                         # Set NaN for all fuel types if there's an error
    #                         if isinstance(samples, dict):
    #                             savings_stats[f'{intervention}_savings_{stat}_gas'] = np.nan
    #                             savings_stats[f'{intervention}_savings_{stat}_electricity'] = np.nan
                                
    #                         else:
    #                             savings_stats[f'{intervention}_savings_{stat}'] = np.nan

                    
    #         except ValueError as ve:
    #             logging.warning(f"Energy savings calculation skipped for {intervention}: {ve}")
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_savings_{stat}'
    #                 savings_stats[col_name] = np.nan
    #         except Exception as e:
    #             logging.error(f"Unexpected error calculating energy savings for {intervention}: {e}")
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_savings_{stat}'
    #                 savings_stats[col_name] = np.nan
        
    #     # Calculate combined gas savings multiplicatively
    #     if len(gas_samples_list) > 0:
    #         # Start with array of ones
    #         combined_gas_samples = np.ones_like(gas_samples_list[0])
            
    #         # Multiply all gas samples together
    #         for gas_sample in gas_samples_list:
    #             combined_gas_samples *= gas_sample
            
    #         # Calculate statistics for combined gas savings
    #         for stat in return_statistics:
    #             try:
    #                 combined_result = self._calculate_statistics(combined_gas_samples, stat)
    #                 savings_stats[f'combined_gas_savings_{stat}'] = combined_result
    #             except ValueError as stat_error:
    #                 logging.error(f"Invalid statistic '{stat}' for combined gas: {stat_error}")
    #                 savings_stats[f'combined_gas_savings_{stat}'] = np.nan
        
    #     return savings_stats

    # def calculate_intervention_energy_savings(self, 
    #                                           interventions, 
    #                                           building_chars,
    #                                         # typology, 
    #                                         # age_band, 
    #                                         region,  
    #                                         return_statistics,
    #                                         roof_scaling):
    #     """
    #     Calculate Monte Carlo energy savings statistics for a list of interventions.
        
    #     Parameters:
    #     -----------
    #     interventions : list
    #         List of intervention names to calculate energy savings for
        
    #     building_chars : BuildingCharacteristics
    #         Building characteristics object
    #     typology : str
    #         Building typology
    #     age_band : str
    #         Building age band
    #     region : str
    #         Regional code
      
    #     n_samples : int
    #         Number of Monte Carlo samples
    #     return_statistics : list
    #         Statistics to calculate (e.g., ['mean', 'p5', 'p50', 'p95'])
        
    #     Returns:
    #     --------
    #     dict: Energy savings statistics for each intervention
    #     """
    #     savings_stats = {}
        
    
    #     for intervention in interventions:
    #         try:
    #             # Get Monte Carlo samples for energy savings
    #             samples = self.energy_config.sample_intervention_energy_savings_monte_carlo(
    #                 intervention=intervention,
    #                 building_chars=building_chars,
    #                 # typology=typology,
    #                 # age_band=age_band,
    #                 region=region,
    #                 n_samples=self.n_samples,
    #                 roof_scaling = roof_scaling,
    #             )
    #             # need to process solar samples different 
    #             if intervention =='solar_pv':
    #                 # Calculate statistics for different metrics
    #                 for stat in return_statistics:
    #                     # For annual generation and adjusted kwh, calculate the requested statistic
    #                     savings_stats[f'{intervention}_annual_generation_{stat}'] = \
    #                         self._calculate_statistics(samples['annual_generation_kwh'], stat)
                        
    #                     savings_stats[f'{intervention}_adjusted_kwh_per_kwp_{stat}'] = \
    #                         self._calculate_statistics(samples['adjusted_kwh_per_kwp'], stat)
            
    #                     # For matched roof size and regional multiplier, always use mode (they're constant across samples)
    #                     savings_stats[f'{intervention}_matched_roof_size_mode'] = \
    #                         stats.mode(samples['matched_roof_size'], keepdims=False)[0]
                        
    #                     savings_stats[f'{intervention}_regional_multiplier_mode'] = \
    #                         stats.mode(samples['regional_multiplier'], keepdims=False)[0]
        
    #             else:
    #                 # Calculate requested statistics
    #                 for stat in return_statistics:
    #                     try:
    #                         result = self._calculate_statistics(samples, stat)
                            
    #                         # Handle dict result (gas and electricity separate)
    #                         if isinstance(result, dict):
    #                             savings_stats[f'{intervention}_savings_{stat}_gas'] = result['gas']
    #                             savings_stats[f'{intervention}_savings_{stat}_electricity'] = result['electricity']
    #                         else:
    #                             # Backwards compatibility: single value result
    #                             savings_stats[f'{intervention}_savings_{stat}'] = result
                                
    #                     except ValueError as stat_error:
    #                         logging.error(f"Invalid statistic '{stat}' for {intervention}: {stat_error}")
    #                         # Set NaN for all fuel types if there's an error
    #                         if isinstance(samples, dict):
    #                             savings_stats[f'{intervention}_savings_{stat}_gas'] = np.nan
    #                             savings_stats[f'{intervention}_savings_{stat}_electricity'] = np.nan
                                
    #                         else:
    #                             savings_stats[f'{intervention}_savings_{stat}'] = np.nan
    
                    
    #         except ValueError as ve:
    #             logging.warning(f"Energy savings calculation skipped for {intervention}: {ve}")
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_savings_{stat}'
    #                 savings_stats[col_name] = np.nan
    #         except Exception as e:
    #             logging.error(f"Unexpected error calculating energy savings for {intervention}: {e}")
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_savings_{stat}'
    #                 savings_stats[col_name] = np.nan
        
    #     return savings_stats


    # def _calculate_and_add_costs_and_energy(self, result_df, col_mapping, scenario_interventions, 
    #                             prob_external, region, scenario, return_statistics):
    #     """Calculate costs ad energy and add them to the DataFrame."""
    #     # Apply cost calculations to all rows
        
    #     combined_results = result_df.apply(
    #         lambda row: self.calculate_row_costs(
    #             row, col_mapping, scenario_interventions, prob_external,
    #             region, self.n_samples, return_statistics
    #         ),
    #         axis=1
    #     )
    #     # Show first few rows with better formatting
    #     logger.debug("Costs results preview:\n%s", combined_results.head().to_string())
        
    #     # Split results into cost and energy columns
    #     cost_cols = [col for col in combined_results.columns if 'cost' in col.lower()]
    #     energy_cols = [col for col in combined_results.columns if 'energy' in col.lower()]

    #     cost_results = combined_results[cost_cols]
    #     energy_results = combined_results[energy_cols]
        
    #     # Add individual intervention cost columns
    #     for col in cost_results.columns:
    #         result_df[f'cost_{scenario}_{col}'] = cost_results[col]
        
    #     # Calculate total scenario costs by summing across interventions
    #     for stat in return_statistics:
    #         intervention_cols = [col for col in cost_results.columns if col.endswith(f'_{stat}')]
    #         result_df[f'cost_{scenario}_{stat}'] = cost_results[intervention_cols].sum(axis=1)
        
    #     for col in energy_results.columns:
    #         result_df[f'energy_{scenario}_{col}'] = energy_results[col]

    #     # Calculate total scenario costs by summing across interventions
    #     for stat in return_statistics:
    #         intervention_cols = [col for col in energy_results.columns if col.endswith(f'_{stat}')]
    #         result_df[f'energy_{scenario}_{stat}'] = energy_results[intervention_cols].sum(axis=1)
        
    #     return result_df





    # def _calculate_and_add_costs_and_energy(self, result_df, col_mapping, scenario_interventions, 
    #                                 prob_external, region, scenario, return_statistics):
    #     """Calculate costs and energy and add them to the DataFrame."""
    #     # Apply cost calculations to all rows
    #     combined_results = result_df.apply(
    #         lambda row: self.calculate_row_costs(
    #             row, col_mapping, scenario_interventions, prob_external,
    #             region, self.n_samples, return_statistics
    #         ),
    #         axis=1
    #     )
    #     logger.debug("Costs results preview:\n%s", combined_results.head().to_string())
        
    #     # Split results into cost and energy columns
    #     cost_results, energy_results = self._split_cost_and_energy_results(combined_results)
        
    #     # Add cost columns
    #     self._add_cost_columns(result_df, cost_results, scenario, return_statistics)
        
    #     # Add energy columns (individual interventions)
    #     self._add_individual_energy_columns(result_df, energy_results, scenario)
        
    #     # Add aggregated energy columns (gas and electricity)
    #     self._add_aggregated_energy_columns(result_df, energy_results, scenario, return_statistics)
        
    #     return result_df


    # def _split_cost_and_energy_results(self, combined_results):
    #     """Split combined results into cost and energy DataFrames."""
    #     cost_cols = [col for col in combined_results.columns if 'cost' in col.lower()]
    #     energy_cols = [col for col in combined_results.columns if 'energy' in col.lower()]
        
    #     return combined_results[cost_cols], combined_results[energy_cols]


    # def _add_cost_columns(self, result_df, cost_results, scenario, return_statistics):
    #     """Add individual and total cost columns to result DataFrame."""
    #     # Add individual intervention cost columns
    #     for col in cost_results.columns:
    #         result_df[f'cost_{scenario}_{col}'] = cost_results[col]
        
    #     # Calculate total scenario costs by summing across interventions
    #     for stat in return_statistics:
    #         intervention_cols = [col for col in cost_results.columns if col.endswith(f'_{stat}')]
    #         result_df[f'cost_{scenario}_{stat}'] = cost_results[intervention_cols].sum(axis=1)


    # def _add_individual_energy_columns(self, result_df, energy_results, scenario):
    #     """Add individual intervention energy columns to result DataFrame."""
    #     for col in energy_results.columns:
    #         result_df[f'energy_{scenario}_{col}'] = energy_results[col]


    # def _add_aggregated_energy_columns(self, result_df, energy_results, scenario, return_statistics):
    #     """Add aggregated gas and electricity energy columns based on scenario logic."""
    #     gas_cols = [col for col in energy_results.columns if '_gas' in col.lower()]
    #     elec_cols = [col for col in energy_results.columns if '_elec' in col.lower()]
        
    #     # Add gas energy columns (multiplicative)
    #     self._add_gas_energy_totals(result_df, energy_results, gas_cols, scenario, return_statistics)
        
    #     # Add electricity energy columns (scenario-specific)
    #     self._add_electricity_energy_totals(result_df, energy_results, elec_cols, scenario, return_statistics)

    # def _add_gas_energy_totals(self, result_df, energy_results, gas_cols, scenario, return_statistics):
    #     """Calculate and add total gas energy using multiplicative logic."""
    #     for stat in return_statistics:
    #         # Get all gas columns for this statistic (e.g., all _mean_gas columns)
    #         gas_stat_cols = [col for col in gas_cols if f'_{stat}_gas' in col.lower()]
            
    #         if gas_stat_cols:
    #             # Multiply gas percentages together for matching statistics
    #             result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[gas_stat_cols].prod(axis=1)
    #         else:
    #             result_df[f'energy_{scenario}_gas_{stat}'] = 1.0  # No change if no gas interventions



    # def _add_electricity_energy_totals(self, result_df, energy_results, elec_cols, scenario, return_statistics):
    #     """Calculate and add total electricity energy based on scenario-specific logic."""
    #     if scenario == 2:
    #         self._add_scenario_2_electricity(result_df, return_statistics)
    #     elif scenario == 3:
    #         self._add_scenario_3_electricity(result_df, energy_results, elec_cols, return_statistics)
    #     elif scenario == 5:
    #         self._add_scenario_5_electricity(result_df, energy_results, elec_cols, return_statistics)
    #     else:
    #         self._add_default_electricity(result_df, energy_results, elec_cols, scenario, return_statistics)


    # def _add_scenario_2_electricity(self, result_df, return_statistics):
    #     """Scenario 2: No electricity columns (not relevant)."""
    #     pass  # Don't add any electricity energy columns


    # def _add_scenario_3_electricity(self, result_df, energy_results, elec_cols, return_statistics):
    #     """Scenario 3: Only heat pump affects electricity (additive increase)."""
    #     for stat in return_statistics:
    #         hp_elec_cols = [col for col in elec_cols if 'heat_pump' in col.lower() and col.endswith(f'_{stat}')]
    #         if hp_elec_cols and 'actual_electricity_consumption' in result_df.columns:
    #             # Heat pump column is a percentage, apply it to actual consumption
    #             result_df[f'energy_3_elec_{stat}'] = (
    #                 result_df['actual_electricity_consumption'] * energy_results[hp_elec_cols[0]]
    #             )
    #         else:
    #             result_df[f'energy_3_elec_{stat}'] = 0.0


    # def _add_scenario_5_electricity(self, result_df, energy_results, elec_cols, return_statistics):
    #     """Scenario 5: Heat pump (percentage increase) and solar (kWh offset)."""
    #     for stat in return_statistics:
    #         hp_elec_cols = [col for col in elec_cols if 'heat_pump' in col.lower() and col.endswith(f'_{stat}')]
    #         solar_elec_cols = [col for col in elec_cols if 'solar' in col.lower() and col.endswith(f'_{stat}')]
            
    #         elec_change = 0.0
            
    #         # Add heat pump electricity increase (percentage applied to actual consumption)
    #         if hp_elec_cols and 'actual_electricity_consumption' in result_df.columns:
    #             elec_change += result_df['actual_electricity_consumption'] * energy_results[hp_elec_cols[0]]
            
    #         # Subtract solar electricity offset (already in kWh)
    #         if solar_elec_cols:
    #             elec_change -= energy_results[solar_elec_cols[0]]
            
    #         result_df[f'energy_5_elec_{stat}'] = elec_change




    # def _calculate_statistics(self, samples, stat: str) -> dict:
    #     """
    #     Calculate a specific statistic from Monte Carlo samples.
        
    #     Parameters:
    #     -----------
    #     samples : np.ndarray or dict
    #         Array of Monte Carlo samples, or dict with 'gas' and 'electricity' keys
    #     stat : str
    #         Statistic to calculate ('mean', 'median', 'p50', 'std', 'p5', 'p95', etc.)
        
    #     Returns:
    #     --------
    #     dict: Dictionary with keys 'gas', 'electricity', and 'total' containing calculated statistics
    #         OR float if input was a simple numpy array (for backwards compatibility)
    #     """
 
        
    #     # Handle dict input (gas and electricity separate)
    #     if isinstance(samples, dict):
    #         logging.debug(f"Dict keys: {samples.keys()}")
    #         # Validate dict structure
    #         if 'gas' not in samples or 'electricity' not in samples:
    #             logging.error(f"Dict must have 'gas' and 'electricity' keys. Found: {samples.keys()}")
    #             raise ValueError(f"Dict must have 'gas' and 'electricity' keys. Found: {samples.keys()}")
            
    #         # Calculate statistics for each fuel type
    #         results = {}
            
    #         # Gas statistics
            
    #         results['gas'] = self._calculate_single_statistic(samples['gas'], stat)
            
    #         # Electricity statistics
            
    #         results['electricity'] = self._calculate_single_statistic(samples['electricity'], stat)
            
 
    #         return results
        
    #     # Handle simple numpy array input (backwards compatibility)
    #     else:
            
    #         result = self._calculate_single_statistic(samples, stat)
            
    #         return result


    # def calculate_intervention_costs(self, interventions, skip_interventions, building_chars, 
    #                                 typology, age_band, region , 
    #                                 return_statistics):
    #     """
    #     Calculate Monte Carlo cost statistics for a list of interventions.
        
    #     Parameters:
    #     -----------
    #     interventions : list
    #         List of intervention names to calculate costs for
    #     skip_interventions : set
    #         Set of interventions to skip (already installed)
    #     building_chars : BuildingCharacteristics
    #         Building characteristics object
    #     typology : str
    #         Building typology
    #     age_band : str
    #         Building age band
    #     region : str
    #         Regional code
 
 
    #     return_statistics : list
    #         Statistics to calculate (e.g., ['mean', 'p5', 'p50', 'p95'])
        
    #     Returns:
    #     --------
    #     dict: Cost statistics for each intervention
    #     """
    #     cost_stats = {}
        
    #     for intervention in interventions:
    #         # Check if intervention is already installed
    #         if intervention in skip_interventions:
    #             # Set cost to 0 for already-installed interventions
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_{stat}'
    #                 cost_stats[col_name] = 0.0
    #             continue
            
    #         try:
    #             # Get Monte Carlo samples
    #             samples = self.sample_intervention_cost_monte_carlo(
    #                 intervention=intervention,
    #                 building_chars=building_chars,
    #                 typology=typology,
    #                 age_band=age_band,
    #                 region=region,
    #             )
                
               
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_savings_{stat}'
    #                 try:
    #                     cost_stats[col_name] = self._calculate_statistics(samples, stat)
    #                 except ValueError as stat_error:
    #                     logging.error(f"Invalid statistic '{stat}' for {intervention}: {stat_error}")
    #                     cost_stats[col_name] = np.nan
   
                
    #         except ValueError as ve:
    #             logging.warning(f"Cost calculation skipped for {intervention}: {ve}")
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_{stat}'
    #                 cost_stats[col_name] = np.nan
    #         except Exception as e:
    #             logging.error(f"Unexpected error calculating {intervention}: {e}")
    #             for stat in return_statistics:
    #                 col_name = f'{intervention}_{stat}'
    #                 cost_stats[col_name] = np.nan
        
    #     return cost_stats


    # def calculate_ONLY_row_energy(self, row, col_mapping, scenario_interventions, prob_external, 
    #                     region, n_samples, return_statistics):
    #     """
    #     Calculate Monte Carlo cost statistics for scenario energy for one building.
        
    #     This is the inner function extracted from calculate_building_costs_df_new.
    #     """
    #     # Data validation
    #     required_cols = ['floor_count', 'gross_external_area', 'gross_internal_area',
    #                     'footprint_circumference', 'building_type', 'age_band', 'building_footprint_area']
        
    #     missing_cols = [col for col in required_cols if col_mapping[col] not in row.index]
    #     if missing_cols:
    #         raise ValueError(f'Missing columns: {missing_cols}')
        
    #     # Convert and validate building characteristics
    #     floor_count = int(row[col_mapping['floor_count']])
    #     gross_external_area = float(row[col_mapping['gross_external_area']])
    #     gross_internal_area = float(row[col_mapping['gross_internal_area']])
    #     footprint_circumference = float(row[col_mapping['footprint_circumference']])
    #     building_footprint_area = float(row[col_mapping['building_footprint_area']])
        
    #     # Use max(1) for flat count to ensure valid input to cost calcs
    #     raw_flat_count = row.get(col_mapping['flat_count'])
    #     flat_count = int(raw_flat_count) if pd.notna(raw_flat_count) and raw_flat_count > 0 else 1
        
    #     if any([
    #         pd.isna(floor_count) or floor_count <= 0,
    #         pd.isna(gross_external_area) or gross_external_area <= 0,
    #         pd.isna(gross_internal_area) or gross_internal_area <= 0,
    #         pd.isna(footprint_circumference) or footprint_circumference <= 0,
    #         pd.isna(building_footprint_area) or building_footprint_area <= 0
    #     ]):
    #         raise ValueError('Invalid building characteristics')
        
    #     building_chars = BuildingCharacteristics(
    #         floor_count=floor_count,
    #         gross_external_area=gross_external_area,
    #         gross_internal_area=gross_internal_area,
    #         footprint_circumference=footprint_circumference,
    #         flat_count=flat_count,
    #         building_footprint_area=building_footprint_area,
    #     )
        
    #     # Extract retrofit status flags
    #     wall_insulated = bool(row['wall_insulated'])
    #     existing_loft = bool(row['existing_loft_insulation'])
    #     existing_floor = bool(row['existing_floor_insulation'])
    #     existing_windows = bool(row['existing_window_upgrades'])
    #     wall_type = str(row['wall_type']).lower().strip()
        
    #     # Validate wall_type
    #     if wall_type not in ['cavity_wall', 'solid_wall']:
    #         raise ValueError(f"Invalid wall_type: '{wall_type}'. Must be 'cavity_wall' or 'solid_wall'")
        
    #     # Resolve interventions (replaces WALL_INSULATION placeholder)
    #     interventions_to_calculate, selected_wall_insulation = self.resolve_interventions_for_building(
    #         scenario_interventions, wall_type, prob_external
    #     )
        
    #     # # Determine which interventions to skip
    #     # skip_interventions = self.get_skip_interventions(
    #     #     wall_insulated, existing_loft, existing_floor, existing_windows
    #     # )
        
    #     # Calculate costs for all interventions
    #     typology = row[col_mapping['building_type']]
    #     age_band = row[col_mapping['age_band']]
       

    #     energy_stats = self.calculate_intervention_energy_savings(
    #         interventions=interventions_to_calculate,
        
    #         building_chars=building_chars,
    #         # typology=typology,
    #         # age_band=age_band,
    #         region=region,
    #         return_statistics=return_statistics,
    #         roof_scaling=self.retrofit_config.existing_intervention_probs['roof_scaling_factor'] , 

    #     )
        
        
        
    #     energy_stats_prefixed = {f'energy_{key}': value for key, value in energy_stats.items()}
    #     result = pd.Series(energy_stats_prefixed )

    #     return result
       

    # def _calculate_and_add_energy(self, result_df, col_mapping, scenario_interventions, 
    #                             prob_external, region, scenario, return_statistics):
    #     """Calculate energy and add them to the DataFrame."""
        
        
    #     # Apply cost calculations to all rows
    #     energy_results = result_df.apply(
    #         lambda row: self.calculate_ONLY_row_energy(
    #             row, col_mapping, scenario_interventions, prob_external,
    #             region, self.n_samples, return_statistics
    #         ),
    #         axis=1
    #     )

    #     # Add energy columns (individual interventions)
    #     self._add_individual_energy_columns(result_df, energy_results, scenario)
        
      
    #     return result_df


    # def _calculate_and_add_costs_and_energy(self, result_df, col_mapping, scenario_interventions, 
    #                             prob_external, region, scenario, return_statistics):
    #     """Calculate costs and energy and add them to the DataFrame."""
        
        
    #     # Apply cost calculations to all rows
    #     combined_results = result_df.apply(
    #         lambda row: self.calculate_row_costs(
    #             row, col_mapping, scenario_interventions, prob_external,
    #             region, self.n_samples, return_statistics
    #         ),
    #         axis=1
    #     )
    #     logger.debug("Costs results preview:\n%s", combined_results.head().to_string())
        
    #     # Split results into cost and energy columns
    #     cost_results, energy_results = self._split_cost_and_energy_results(combined_results)
        
    #     # Add cost columns
    #     self._add_cost_columns(result_df, cost_results, scenario, return_statistics)
        
    #     # Add energy columns (individual interventions)
    #     self._add_individual_energy_columns(result_df, energy_results, scenario)
        
    #     # Add aggregated energy columns (gas and electricity) using new combined_savings columns
    #     self._add_aggregated_energy_columns(result_df, energy_results, scenario, return_statistics)
        
    #     return result_df


    # def _split_cost_and_energy_results(self, combined_results):
    #     """Split combined results into cost and energy DataFrames."""
    #     cost_cols = [col for col in combined_results.columns if 'cost' in col.lower()]
    #     energy_cols = [col for col in combined_results.columns if 'energy' in col.lower()]
        
    #     return combined_results[cost_cols], combined_results[energy_cols]
