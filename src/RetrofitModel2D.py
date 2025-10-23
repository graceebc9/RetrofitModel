from scipy import stats
import sys 
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

# Assuming these imports are available in your environment
from .BuildingCharacteristics import BuildingCharacteristics  
from .RetrofitCosts import  CostEstimator, InterventionConfig
from .RetrofitEnergy import RetrofitEnergy 
from .RetrofitUtils import calculate_estimated_flats_per_building 
from .RetrofitConfig import RetrofitConfig 
from .RetrofitPackages import retrofit_packages 

logger = logging.getLogger(__name__)

@dataclass
class RetrofitModel2D:
    """
    2DMC Inner Loop Executor: Simulates retrofits for one fixed Epistemic Scenario
    across N_aleatory (n_samples) runs.
    """
    retrofit_config: RetrofitConfig 
    
    n_samples: int = 100
    
    # 2. EPISTEMIC UNCERTAINTY (Fixed inputs for this run, derived from Outer Loop sampler)
    epistemic_scenario: Dict[str, float] = field(default_factory=dict)
    
    # --- Dependencies ---
    energy_config: Optional[RetrofitEnergy] = None
    cost_estimator: CostEstimator = field(default_factory=CostEstimator)
    custom_intervention_configs: Optional[Dict[str, InterventionConfig]] = None
    

    # --- Epistemic Factor Nominal Defaults (for energy model init) ---
    _SOLID_WALL_INT_NOMINAL = 0.1
    _SOLID_WALL_EXT_NOMINAL = 0.2
    
    # --- Original Class Data Definitions (Fully retained) ---
    typologies: List[str] = field(default_factory=lambda: [
        'Medium height flats 5-6 storeys', 'Small low terraces', '3-4 storey and smaller flats',
        'Tall terraces 3-4 storeys', 'Large semi detached', 'Standard size detached',
        'Standard size semi detached', '2 storeys terraces with t rear extension',
        'Semi type house in multiples', 'Tall flats 6-15 storeys', 'Large detached',
        'Very tall point block flats', 'Very large detached', 'Planned balanced mixed estates',
        'Linked and step linked premises', 'Domestic outbuilding', 'all_unknown_typology',
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
    
    age_band_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Post 1999': 0.90, '1990-1999': 0.95, '1980-1989': 1.0,
        '1960-1979': 1.15, '1945-1959': 1.35, '1919-1944': 1.6, 'Pre 1919': 2.0
    })
    
    # typology_complexity: Dict[str, float] = field(default_factory=lambda: {
    #     'Very tall point block flats': 1.4,
    #     'Tall flats 6-15 storeys': 1.2,
    #     'Medium height flats 5-6 storeys': 1.1,
    #     'Tall terraces 3-4 storeys': 1.1,
    # })

    retrofit_packages= retrofit_packages
     
    decile_risk_scaling: Dict[int, float] = field(default_factory=lambda: {
        0: 1.5, 1: 1.3, 2: 1.0, 3: 1.0, 4: 0.8, 5: 0.7, 6: 0.7, 7: 0.7, 8: 0.8, 9: 1.2
    })

 
    def __post_init__(self):
        """Validate inputs and apply Epistemic factors to internal configs."""
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.n_samples < 100:
            logger.warning(f"Warning: n_samples={self.n_samples} is low. Consider using 100+ for stable results.")

        # 3. PULL EPISTEMIC FACTORS AND APPLY TO RETROFITENERGY
        
        # Factors defining the technical performance of wall measures
        int_factor = self.epistemic_scenario.get('solid_wall_internal_improvement_factor', self._SOLID_WALL_INT_NOMINAL)
        ext_factor = self.epistemic_scenario.get('solid_wall_external_improvement_factor', self._SOLID_WALL_EXT_NOMINAL)
        
        # Create/Update RetrofitEnergy config
        if self.energy_config is None:
            self.energy_config = RetrofitEnergy(
                solid_wall_internal_improvement_factor=int_factor, 
                solid_wall_external_improvement_factor=ext_factor
            )
            logger.debug(f"Created RetrofitEnergy config with Epistemic wall factors: internal {int_factor} and external {ext_factor}")
        else:
            # If provided, ensure it uses the epistemic factors
            self.energy_config.solid_wall_internal_improvement_factor = int_factor
            self.energy_config.solid_wall_external_improvement_factor = ext_factor
            
        # Initialize cost estimator with custom configs if provided (original logic)
        if self.custom_intervention_configs is not None:
            self.cost_estimator = CostEstimator(self.custom_intervention_configs)
        
        # (Original logging from your script)
        logger.debug(f"Regional multipliers: {list(self.regional_multipliers.keys())}")
        logger.debug(f"Available scenarios: {list(self.retrofit_packages.keys())}")
        logger.debug("RetrofitModel (Inner Loop) initialized successfully with fixed Epistemic Scenario.")

    # --- Utility Methods (ORIGINAL CODE) ---

    def validate_region(self, region: str) -> str:
        """Validate region code."""
        if region not in self.valid_regions:
            raise ValueError(f"Invalid region '{region}'. Valid: {self.valid_regions}")
        return region
    
    def get_regional_multiplier(self, region: str) -> float:
        """Get regional cost multiplier."""
        return self.regional_multipliers[self.validate_region(region)]

    def _validate_inputs(self, df, region, scenario):
        # ... (Original validation logic) ...
        if df is None or df.empty:
            return {'error': 'DataFrame is None or empty'}
        if not region:
            return {'error': 'Region parameter is required'}
        if not scenario:
            return {'error': 'Scenario parameter is required'}
        if scenario not in self.retrofit_packages:
            return {'error': f'Scenario "{scenario}" not found in config.retrofit_packages'}
        return None 
    
    def _validate_statistics(self, return_statistics):
        # ... (Original validation logic) ...
        if return_statistics is None:
            return ['mean', 'p5', 'p50', 'p95', 'std' ]
        valid_statistics = ['mean', 'median', 'std', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']
        invalid_stats = [s for s in return_statistics if s not in valid_statistics]
        if invalid_stats:
            return {'error': f'Invalid statistics requested: {invalid_stats}. Valid: {valid_statistics}'}
        return return_statistics

    def _get_scenario_interventions(self, scenario):
        # ... (Original logic) ...
        selected_scenario = self.retrofit_packages[scenario]
        scenario_interventions = selected_scenario.get('interventions', [])
        if not scenario_interventions:
            return {'error': f'No interventions defined for scenario "{scenario}"'}
        return scenario_interventions

    def _get_column_mapping(self, col_mapping):
        # ... (Original logic) ...
        default_mapping = {
            'floor_count': 'total_fl_area_avg', 'gross_external_area': 'total_fl_area_avg',
            'gross_internal_area': 'scaled_fl_area', 'footprint_circumference': 'perimeter_length',
            'flat_count': 'est_num_flats', 'building_type': 'premise_type',
            'age_band': 'premise_age', 'building_footprint_area': 'premise_area',
            'avg_gas_percentile':'avg_gas_percentile', 'inferred_wall_type': 'inferred_wall_type',
            'inferred_insulation_type': 'inferred_insulation_type',
        }
        if col_mapping is None:
            return default_mapping
        for key, default_val in default_mapping.items():
            if key not in col_mapping:
                col_mapping[key] = default_val
        return col_mapping

    def _prepare_dataframe(self, df, col_mapping):
        # ... (Original logic) ...
        result_df = df.copy()
        result_df = result_df[result_df[col_mapping['building_type']] != 'Domestic outbuilding']
        result_df['est_num_flats'] = result_df.apply(
            lambda row: calculate_estimated_flats_per_building(
                building_footprint_area=row['premise_area'],
                typology_col=row['premise_type'],
                floor_count=row['premise_floor_count']
            ), axis=1
        )
        return result_df

    def _validate_dataframe_columns(self, df, col_mapping):
        # ... (Original logic) ...
        missing_columns = [
            f"{field} -> {col_name}" 
            for field, col_name in col_mapping.items() 
            if col_name not in df.columns
        ]
        if missing_columns:
            return {'error': f'Required columns not found: {", ".join(missing_columns)}'}
        required_new_cols = ['wall_insulated', 'existing_loft_insulation', 'existing_floor_insulation', 'existing_window_upgrades', 'inferred_wall_type']
        missing_new_cols = [col for col in required_new_cols if col not in df.columns]
        if missing_new_cols:
            return {'error': f'Required retrofit status columns not found: {", ".join(missing_new_cols)}'}
        return None

    def get_skip_interventions(self, wall_insulated, existing_loft, existing_floor, existing_windows):
        # ... (Original logic) ...
        skip_interventions = set()
        if wall_insulated:
            skip_interventions.add('cavity_wall_insulation'); skip_interventions.add('external_wall_insulation')
            skip_interventions.add('internal_wall_insulation'); skip_interventions.add('cavity_wall_percentile')
            skip_interventions.add('solid_wall_percentile'); skip_interventions.add('solid_wall_internal_percentile')
            skip_interventions.add('solid_wall_external_percentile')
        if existing_loft:
            skip_interventions.add('loft_insulation'); skip_interventions.add('loft_percentile')
        if existing_floor:
            skip_interventions.add('floor_insulation')
        if existing_windows:
            skip_interventions.add('double_glazing')
        return skip_interventions

    def _calculate_single_statistic(self, samples: np.ndarray, stat: str) -> float:
        # ... (Original logic) ...
        if not isinstance(samples, np.ndarray):
            try: samples = np.array(samples)
            except Exception as e: raise TypeError(f"Cannot convert samples to numpy array: {e}")
        if samples.size == 0:
            raise ValueError("Cannot calculate statistics on empty array")
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
        except Exception as e:
            logging.error(f"Error calculating {stat}: {e}")
            raise

    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # CORE MODIFIED ENERGY METHOD: Applying Epistemic Factors
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def calculate_intervention_energy_savings(
        self, 
        interventions, 
        building_chars,
        region,  
        return_statistics,
        roof_scaling,
        wall_type, 
    ):
        """
        Calculate Monte Carlo energy savings statistics, applying Rebound and
        Time-Scale Epistemic factors to the final samples, and P95-Mismatch to the final P95.
        """
        # Epistemic Factors for Final Adjustment (Retrieved from fixed scenario)
        
        beta_TS = self.epistemic_scenario.get('time_scale_bias', 1.0)
        beta_DEC = self.epistemic_scenario.get('decile_misclassification_bias', 0.0)
        
        # 1. Look up the Decile Risk Multiplier
        decile_scale = self.decile_risk_scaling.get(building_chars.avg_gas_percentile, 1.0)
        effective_beta_DEC = beta_DEC * decile_scale

        energy_stats = {}
        all_gas_perc_samples = [] # List of N_aleatory arrays for aggregation
        all_elec_perc_samples = []

        # NOTE on beta_BEH: If your RetrofitEnergy class internally uses the building's
        # characteristic baseline, you must handle the systematic baseline increase
        # (1 + beta_BEH) before calculating savings. Since we don't have RetrofitEnergy code,
        # we'll assume savings samples are absolute MWh saved, and the most robust place 
        # for a systematic baseline shift is in the initial savings calculation 
        # (which is handled by the updated solid wall factors in __post_init__).
        
        for intervention in interventions:
            if intervention != 'solar_pv':
                try:
                    # Get Monte Carlo samples (Aleatory Savings, adjusted by SWI/SWE factors)
                    samples = self.energy_config.sample_intervention_energy_savings_monte_carlo(
                        intervention=intervention,
                        building_chars=building_chars,
                        region=region,
                        n_samples=self.n_samples,
                        roof_scaling=roof_scaling,
                        wall_type=wall_type,
                    )

                    gas_perc_samples = samples.get('gas') if isinstance(samples, dict) else samples
                    elec_perc_samples = samples.get('electricity') if isinstance(samples, dict) else None
                    
                    # 1. APPLY REMAINING EPISTEMIC SCALE FACTOR (Time Scale Mismatch)
                    if gas_perc_samples is not None:
                        # Percentage Reduction Adjusted = Perc_Reduction (Aleatory) * beta_TS
                        gas_perc_samples_adjusted = (gas_perc_samples + effective_beta_DEC)  * beta_TS
                        all_gas_perc_samples.append(gas_perc_samples_adjusted)
                    
                    if elec_perc_samples is not None:
                        logger.debug('We have elec samples here in calculate_intervention_energy_savings ')
                        elec_samples_adjusted = (elec_perc_samples+ effective_beta_DEC) * beta_TS
                        all_elec_perc_samples.append(elec_samples_adjusted)

                        if np.all(elec_perc_samples == 0):
                            logger.warning(f'Electricity samples are all zeros for joint intervention {intervention} '
                                        f'with interventions list: . Expected non-zero values.')
                            raise ValueError(f'Electricity samples are all zeros for joint intervention {intervention}, '
                                        f'but electricity impact was expected from interventions')
                        if np.any(np.isnan(elec_perc_samples)):
                            logger.warning(f'Electricity samples contain NaN values for joint intervention {intervention} '
                                        f'with interventions list: .')
                            raise ValueError(f'Electricity samples contain NaN values for joint intervention {intervention}. '
                                        f'Check the electricity data for interventions: ')

                except Exception as e:
                    logger.warning(f"Error processing intervention {intervention}: {e}")
                    all_gas_perc_samples.append(np.full(self.n_samples, np.nan))
                    all_elec_perc_samples.append(np.full(self.n_samples, np.nan))
        
        # 2. AGGREGATE SAMPLES  
        # CHECK 5: Verify aggregation lists before calculating statistics
        logger.debug(f'Total gas sample arrays collected: {len(all_gas_perc_samples)}')
        logger.debug(f'Total elec sample arrays collected: {len(all_elec_perc_samples)}')
        if all_elec_perc_samples:
            for i, elec_arr in enumerate(all_elec_perc_samples):
                logger.debug(f'Elec array {i}: shape={elec_arr.shape}, mean={np.mean(elec_arr):.4f}, '
                           f'all_zeros={np.all(elec_arr == 0)}, has_nan={np.any(np.isnan(elec_arr))}')
 
        
        # --- Gas Aggregation  ---
        if all_gas_perc_samples:
            # Calculate final statistics  
            for stat in return_statistics:
                val = self._calculate_single_statistic(all_gas_perc_samples, stat)
                col_name = f"gas_{stat}"
                energy_stats[col_name] = val
                        
        # --- Electricity Aggregation  n ---
        if all_elec_perc_samples:
            # Calculate final statistics  
            for stat in return_statistics:
                val = self._calculate_single_statistic(all_elec_perc_samples, stat)
                col_name = f"electricity_{stat}"
                energy_stats[col_name] = val
                logger.debug(f'Electricity {stat}: {val:.4f}')
        else:
            logger.warning('Skipping electricity statistics - no samples collected!')
            
        return energy_stats
 
 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # CORE MODIFIED COST METHOD: Applying Epistemic Cost Factors
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def sample_intervention_cost_monte_carlo(self,
                                            intervention: List,
                                            cost_col_name: str, 
                                            building_chars: BuildingCharacteristics,
                                            typology: str,
                                            wall_insulation_type:str ,
                                            age_band: str,
                                            region: str,
                                             ) -> np.ndarray:
        """
        Sample intervention costs using Monte Carlo simulation, applying Epistemic 
        Cost Multipliers (Regional & Age) to the sampled costs.
        """
        if typology is None or typology == 'None'  :
            return None
        
        validated_region = self.validate_region(region)
        
        # 1. GET NOMINAL MULTIPLIERS (from your fixed class defaults)
        age_mult_nominal = self.age_band_multipliers.get(age_band, 1.0)
        # complexity_mult_nominal = self.typology_complexity.get(typology, 1.0)
        regional_mult_nominal = self.get_regional_multiplier(validated_region)
        
        # 2. GET EPISTEMIC MULTIPLIERS (from the fixed scenario)
        beta_REG = self.epistemic_scenario.get('regional_multipliers_uncertainty', 1.0)
        beta_AGE = self.epistemic_scenario.get('age_band_multipliers_uncertainty', 1.0)
        cost_epist_scenario = self.epistemic_scenario.get('cost_scenario', 1.0) 
        
        # 3. APPLY EPISTEMIC UNCERTAINTY TO NOMINAL MULTIPLIERS
        
        # The true systematic multiplier for this run is the nominal * the sampled error
        final_regional_mult = regional_mult_nominal * beta_REG
        final_age_mult = age_mult_nominal * beta_AGE
        
        logger.debug(
            f"Sampling {cost_col_name}: region={validated_region}, "
            f"final_age_mult={final_age_mult:.2f}, "
            f"final_regional_mult={final_regional_mult:.2f}, "
            # f"complexity_mult={complexity_mult_nominal:.2f}"
            f"Intervention: {intervention}"
            f"Wall Type: {wall_insulation_type}"
            f"Cost scenario : {cost_epist_scenario}"
        )
        
        try:
            samples = self.cost_estimator.sample_cost_for_package(
                intervention=intervention,
                building_chars=building_chars,
                typology=typology,
                wall_type=wall_insulation_type, 
                age_band=age_band,
                region=region,
                cost_col_name=cost_col_name,
                epist_scenario=cost_epist_scenario,
                regional_multiplier=final_regional_mult, # NEW: Use corrected multiplier
                age_multiplier=final_age_mult,           # NEW: Use corrected multiplier
                # complexity_multiplier=complexity_mult_nominal, 
                n_samples=self.n_samples
            )
            
            logger.debug(f"{cost_col_name} samples: mean=£{samples.mean():,.0f}")
            return samples
            
        except ValueError as e:
            logger.error(f"Error sampling {cost_col_name}: {e}")
            raise
    
    def calculate_intervention_costs(self,
                                    intervention, 
                                    cost_col_name, 
                                    building_chars, 
                                    wall_insulation_type,
                                    typology,
                                    age_band,
                                    region, 
                                    return_statistics,
                              
                                    # complexity_multiplier,
                                    include_total=True):
        """
        Calculate Monte Carlo cost statistics for an intervention.
        
        Args:
            intervention: List of intervention names
            cost_col_name: Name for the cost column
            building_chars: Building characteristics dictionary
            wall_insulation_type: Type of wall insulation
            typology: Building typology
            age_band: Age band of the building
            region: Geographic region
            return_statistics: List of statistics to calculate (e.g., ['mean', 'std', 'p10', 'p50', 'p90'])
            include_total: Whether to calculate total costs across interventions
            
        Returns:
            Dictionary of cost statistics
        """
        cost_stats = {}
        all_samples = [] if include_total else None
        
        try:
            logger.debug(f"Attempting cost calculation for: {cost_col_name}")
            logger.debug(f"  intervention={intervention}")
            logger.debug(f"  wall_insulation_type={wall_insulation_type}")
            logger.debug(f"  typology={typology}, age_band={age_band}, region={region}")

            samples = self.sample_intervention_cost_monte_carlo(
                intervention=intervention,
                building_chars=building_chars,
          
                typology=typology,
                age_band=age_band,
                region=region,
                wall_insulation_type=wall_insulation_type, 
                cost_col_name=cost_col_name,
            )
            
            if samples is None:
                logger.debug(f"Skipping {cost_col_name}: missing building characteristics")
                logger.debug(f"  typology={typology}, age_band={age_band}")
                
                # Return empty stats for this intervention
                if return_statistics:
                    for stat in return_statistics:
                        cost_stats[f'{cost_col_name}_{stat}'] = None
                
                return cost_stats
            
            # Calculate requested statistics
            for stat in return_statistics:
                col_name = f'{cost_col_name}_{stat}'
                try:
                    cost_stats[col_name] = self._calculate_single_statistic(samples, stat)
                except ValueError as stat_error:
                    logger.error(f"Invalid statistic '{stat}' for {cost_col_name}: {stat_error}")
                    cost_stats[col_name] = np.nan
            
            if include_total:
                all_samples.append(samples)
                
        except Exception as e:
            logger.error(f"Error calculating {cost_col_name}: {e}")
            for stat in return_statistics: 
                cost_stats[f'{cost_col_name}_{stat}'] = np.nan
            if include_total:
                all_samples.append(np.full(self.n_samples, np.nan))
        
        # Calculate total statistics if requested
        if include_total and all_samples:
            total_samples = np.sum(all_samples, axis=0)
            for stat in return_statistics:
                col_name = f'total_{stat}'
                try:
                    cost_stats[col_name] = self._calculate_single_statistic(total_samples, stat)
                except ValueError as stat_error:
                    logger.error(f"Invalid statistic '{stat}' for total: {stat_error}")
                    cost_stats[col_name] = np.nan
        
        return cost_stats





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
                                        scenario_name, 
                            
                                        region, 
                                        return_statistics  ):
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
        
        # run_percentile= True 

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
 
        
        # # Determine which interventions to skip
        # skip_interventions = self.get_skip_interventions(
        #     wall_insulated, existing_loft, existing_floor, existing_windows
        # )
        
        # Calculate costs for all interventions
        typology = row[col_mapping['building_type']]
        age_band = row[col_mapping['age_band']]
        
        
        # there should be one cost col per scenario which is the total costs 
        cost_stats = self.calculate_intervention_costs(
            intervention=interventions_to_calculate,
            # skip_interventions=skip_interventions,
            building_chars=building_chars,
            wall_insulation_type=selected_wall_insulation, 
            typology=typology,
            age_band=age_band,
            region=region,
       
            return_statistics=return_statistics,
            cost_col_name = scenario_name, 
        )
             
        # Add prefixes to cost and energy statistics
        cost_stats_prefixed = {f'cost_{key}': value for key, value in cost_stats.items()}
        
        
        cost_result = pd.Series(
            cost_stats_prefixed,   
           )
 
        energy_stats = self.calculate_intervention_energy_savings(
            interventions=interventions_to_calculate,
            building_chars=building_chars,
            region=region,
            return_statistics=return_statistics,
            roof_scaling=self.retrofit_config.existing_intervention_probs['roof_scaling_factor'] , 
            wall_type = insulation_type ,
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
                                     return_statistics,
                                    ):
        """Calculate costs  and add them to the DataFrame. method is only for joint sampling """
        
        energy_res = result_df.copy() 
        
        
        # Apply cost calculations to all rows, this calc the total at sample time 
        results  = result_df.apply(
            lambda row: self.calculate_ONLY_row_costs_only(row, 
                                                            col_mapping=col_mapping, 
                                                            scenario_interventions=scenario_interventions, 
                                                            region=region,  
                                                            return_statistics=return_statistics,
                                                            scenario_name=scenario,
                                                            
                                                             ), axis=1
                                    )
        
        # Unpack the results - each element is a tuple of (cost_series, energy_series)
        cost_results = pd.DataFrame([x[0] for x in results])
        energy_results = pd.DataFrame([x[1] for x in results])
        
        # CHECK: Verify energy_results columns are not all NaN
        logger.debug(f'Energy results shape: {energy_results.shape}, columns: {list(energy_results.columns)}')
        
        for col in energy_results.columns:
            if energy_results[col].isna().all():
                logger.error(f'Energy column "{col}" is all NaN in energy_results!')
                raise ValueError(f'Energy column "{col}" contains all NaN values. '
                               f'Check energy calculations for scenario: {scenario}')
            
            # Also check for all zeros (which might indicate a problem)
            if (energy_results[col] == 0).all():
                logger.warning(f'Energy column "{col}" is all zeros in energy_results. '
                             f'This may indicate missing energy savings data.')
            
            # Log summary statistics for debugging
            logger.debug(f'Energy column "{col}": mean={energy_results[col].mean():.4f}, '
                       f'nan_count={energy_results[col].isna().sum()}, '
                       f'zero_count={(energy_results[col] == 0).sum()}')
        
        # CHECK: Also verify cost_results columns
        logger.debug(f'Cost results shape: {cost_results.shape}, columns: {list(cost_results.columns)}')
        
        for col in cost_results.columns:
            if cost_results[col].isna().all():
                logger.error(f'Cost column "{col}" is all NaN in cost_results!')
                raise ValueError(f'Cost column "{col}" contains all NaN values. '
                               f'Check cost calculations for scenario: {scenario}')
            
            logger.debug(f'Cost column "{col}": mean={cost_results[col].mean():.4f}, '
                       f'nan_count={cost_results[col].isna().sum()}')
        
        self._add_cost_columns(result_df, cost_results, scenario)
        # Add energy columns (individual interventions)
        self._add_individual_energy_columns(energy_res, energy_results, scenario)
        logger.debug('_calculate_and_add_costs Added costs total complete for all rows.')
        return result_df, energy_res

    # def _calculate_and_add_costs(self, 
    #                              result_df,
    #                               col_mapping,
    #                                 scenario_interventions, 
    #                             # prob_external,
    #                               region,
    #                                scenario,
    #                                  return_statistics,
    #                                 ):
    #     """Calculate costs  and add them to the DataFrame. method is only for joint sampling """
        
    #     energy_res = result_df.copy() 
        
        
    #     # Apply cost calculations to all rows, this calc the total at sample time 
    #     results  = result_df.apply(
    #         lambda row: self.calculate_ONLY_row_costs_only(row, 
    #                                                         col_mapping=col_mapping, 
    #                                                         scenario_interventions=scenario_interventions, 
    #                                                         region=region,  
    #                                                         return_statistics=return_statistics,
    #                                                         scenario_name=scenario,
                                                            
    #                                                          ), axis=1
    #                                 )
        
    #     # Unpack the results - each element is a tuple of (cost_series, energy_series)
    #     cost_results = pd.DataFrame([x[0] for x in results])
    #     energy_results = pd.DataFrame([x[1] for x in results])
        
    #     self._add_cost_columns(result_df, cost_results, scenario)
    #     # Add energy columns (individual interventions)
    #     self._add_individual_energy_columns(energy_res, energy_results, scenario)
    #     logger.debug('_calculate_and_add_costs Added costs total complete for all rows.')
    #     return result_df, energy_res



    def _add_cost_columns(self, result_df, cost_results, scenario):
        """Add individual and total cost columns to result DataFrame."""
        # Add individual intervention cost columns
 
        for col in cost_results.columns:
      
            result_df[col] = cost_results[col]
        
  

    def _add_individual_energy_columns(self, result_df, energy_results, scenario):
        """Add individual intervention energy columns to result DataFrame."""
        for col in energy_results.columns:
            result_df[f'{scenario}_{col}'] = energy_results[col]


    # def _add_aggregated_energy_columns(self, result_df, energy_results, scenario, return_statistics):
    #     """Add aggregated gas and electricity energy columns using combined_savings columns."""
    #     # Add gas energy columns (using combined_savings from multiplicative aggregation)
    #     self._add_gas_energy_totals(result_df, energy_results, scenario, return_statistics)
        
    #     # Add electricity energy columns (using combined_savings from additive aggregation)
    #     self._add_electricity_energy_totals(result_df, energy_results, scenario, return_statistics)


    # def _add_gas_energy_totals(self, result_df, energy_results, scenario, return_statistics):
    #     """Calculate and add total gas energy using combined_savings columns."""
    #     for stat in return_statistics:
    #         # Look for the combined_savings column for this statistic
    #         combined_col = f'combined_savings_{stat}_gas'
            
    #         if combined_col in energy_results.columns:
    #             # Use the pre-calculated combined savings (multiplicative aggregation already done)
    #             result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[combined_col]
    #         else:
    #             # Fallback: if no combined column exists (e.g., only 1 intervention or solar only)
    #             # Look for individual gas columns for this statistic
    #             gas_stat_cols = [col for col in energy_results.columns 
    #                         if f'_{stat}_gas' in col.lower() and 'combined' not in col.lower()]
                
    #             if len(gas_stat_cols) == 1:
    #                 # Single intervention: just use its value
    #                 result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[gas_stat_cols[0]]
    #             elif len(gas_stat_cols) > 1:
    #                 # Multiple interventions but no combined column: multiply them
    #                 # (This shouldn't happen with new structure, but kept for safety)
    #                 result_df[f'energy_{scenario}_gas_{stat}'] = energy_results[gas_stat_cols].prod(axis=1)
    #             else:
    #                 # No gas interventions
    #                 result_df[f'energy_{scenario}_gas_{stat}'] = 1.0


    # def _add_electricity_energy_totals(self, result_df, energy_results, scenario, return_statistics):
    #     """Calculate and add total electricity energy using combined_savings columns."""
    #     for stat in return_statistics:
    #         # Look for the combined_savings column for this statistic
    #         combined_col = f'combined_savings_{stat}_electricity'
            
    #         if combined_col in energy_results.columns:
    #             # Use the pre-calculated combined savings (additive aggregation already done)
    #             result_df[f'energy_{scenario}_electricity_{stat}'] = energy_results[combined_col]
    #         else:
    #             # Fallback: if no combined column exists
    #             # Look for individual electricity columns for this statistic
    #             elec_stat_cols = [col for col in energy_results.columns 
    #                             if f'_{stat}_electricity' in col.lower() and 'combined' not in col.lower()]
                
    #             if len(elec_stat_cols) == 1:
    #                 # Single intervention: just use its value
    #                 result_df[f'energy_{scenario}_electricity_{stat}'] = energy_results[elec_stat_cols[0]]
    #             elif len(elec_stat_cols) > 1:
    #                 # Multiple interventions but no combined column: add them
    #                 # (This shouldn't happen with new structure, but kept for safety)
    #                 result_df[f'energy_{scenario}_electricity_{stat}'] = energy_results[elec_stat_cols].sum(axis=1)
    #             else:
    #                 # No electricity interventions
    #                 result_df[f'energy_{scenario}_electricity_{stat}'] = 0.0


    # def _add_default_electricity(self, result_df, energy_results, elec_cols, scenario, return_statistics):
    #     """Default behavior: Sum electricity additively for other scenarios."""
    #     for stat in return_statistics:
    #         elec_intervention_cols = [col for col in elec_cols if col.endswith(f'_{stat}')]
    #         if elec_intervention_cols:
    #             result_df[f'energy_{scenario}_elec_{stat}'] = energy_results[elec_intervention_cols].sum(axis=1)
    def _get_cols_scenario_intervention(self, scenario_str, stats=['mean', 'std', 'p5', 'p50', 'p95']):
        cost_cols = [] 
        energy_cols = [] 
        
        if scenario_str == 'wall_installation':
            interventions = ['cavity_wall_percentile', 'solid_wall_internal_percentile', 'solid_wall_external_percentile']
            elec = False 
        elif scenario_str == 'loft_installation':
            interventions = ['loft_percentile']
            elec = False 
        elif scenario_str == 'joint_loft_wall_add':
            interventions = ['joint_loft_wall_add']
            elec = False 
        elif scenario_str == 'joint_loft_wall_decay':
            interventions = ['joint_loft_wall_decay']
            elec = False 
        elif scenario_str == 'heat_pump_only':
            interventions = ['heat_pump_percentile']
            elec = True 
        elif scenario_str == 'join_heat_ins_decay':
            interventions = ['join_heat_ins_decay']
            elec = True 
        elif scenario_str == 'join_heat_ins_add':
            interventions = ['join_heat_ins_add']
            elec = True 
        else:
            raise Exception(f'Need to define the interventions for scenario ({scenario_str}) in RetrofitModel _get_cols_scenario_intervention')
        
        # If only one intervention, don't include intervention name in column names
        single_intervention = len(interventions) == 1
        
        for iint in interventions:
            for s in stats:
                if single_intervention:
                    # Simple column names when there's only one intervention
                    cost_cols.append(f'cost_{scenario_str}_{s}')
                    energy_cols.append(f'{scenario_str}_gas_{s}')
                    if elec:
                        energy_cols.append(f'{scenario_str}_electricity_{s}')  # FIX: Add scenario prefix
                else:
                    # Include intervention name when there are multiple
                    cost_cols.append(f'{iint}_cost_{s}')
                    energy_cols.append(f'{iint}_gas_{s}')
                    if elec:
                        energy_cols.append(f'{iint}_electricity_{s}')
        
        return cost_cols, energy_cols
    
    # def _get_cols_scenario_intervention(self, scenario_str, stats = ['mean', 'std', 'p5', 'p50', 'p95']):
    #     cost_cols= [] 
    #     energy_cols= [] 
    #     if scenario_str == 'wall_installation':
    #         interventions = ['cavity_wall_percentile', 'solid_wall_internal_percentile', 'solid_wall_external_percentile' ]
    #         elec=False 
    #     elif scenario_str =='loft_installation':
    #         interventions = ['loft_percentile']
    #         elec=False 
    #     elif scenario_str =='joint_loft_wall_add':
    #         interventions =['joint_loft_wall_add']
    #         elec=False 
    #     elif scenario_str =='joint_loft_wall_decay':
    #         interventions =['joint_loft_wall_decay']
    #         elec=False 
    #     elif scenario_str =='heat_pump_only':
    #         interventions = ['heat_pump_percentile']
    #         elec =True 
    #     elif scenario_str =='join_heat_ins_decay':
    #         interventions = ['join_heat_ins_decay']
    #         elec =True 
    #     elif scenario_str =='join_heat_ins_add':
    #         interventions = ['join_heat_ins_add']
    #         elec =True 
    #     else:
    #         raise Exception(f'Need to define the interventions  for scenarioi  ({scenario_str}) in RetrofitModel _get_cols_scenario_intervention')
        
    #     for iint in interventions:
    #         for s in stats:
    #             cost_cols.append( f'cost_{scenario_str}_{s}' ) 
    #             energy_cols.append(  f'{scenario_str}_gas_{s}' ) 
    #             if elec:
    #                 energy_cols.append(  f'{scenario_str}_electricity_{s}' ) 
        
    #     return cost_cols,  energy_cols


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

    # def calculate_building_costs_df_updated(self,
    #                                         df, 
    #                                         region, 
    #                                         scenario, 
    #                                         col_mapping=None, 
    #                                         return_statistics=None, 
    #                                       ):
    #     """
    #     Apply Monte Carlo building cost calculations to all rows in a DataFrame for a specific scenario.
        
    #     Main orchestrator function that coordinates validation, preparation, and cost calculations.
    #     """
    #     def expand_dict_columns(df):
    #         df_expanded = df.copy()
            
    #         for col in df.columns:
    #             if isinstance(df[col].iloc[0], dict):
    #                 # Expand dictionary column
    #                 temp_df = df[col].apply(pd.Series)
    #                 temp_df.columns = [f"{col}_{subcol}" for subcol in temp_df.columns]
    #                 df_expanded = pd.concat([df_expanded.drop(columns=[col]), temp_df], axis=1)
            
    #         return df_expanded
    #     # Validate inputs
    #     error = self._validate_inputs(df, region, scenario)
    #     if error:
    #         return error
        
    #     # Validate and get statistics
    #     return_statistics = self._validate_statistics(return_statistics)
    #     if isinstance(return_statistics, dict) and 'error' in return_statistics:
    #         return return_statistics
        
    #     # Get scenario interventions
    #     scenario_interventions = self._get_scenario_interventions(scenario)
    #     if isinstance(scenario_interventions, dict) and 'error' in scenario_interventions:
    #         return scenario_interventions
        
    #     # Get column mapping
    #     col_mapping = self._get_column_mapping(col_mapping)
        
    #     # Prepare DataFrame
    #     result_df = self._prepare_dataframe(df, col_mapping)
    #     base_cols = result_df.columns.tolist() 
    #     # Validate DataFrame columns
    #     error = self._validate_dataframe_columns(result_df, col_mapping)
    #     if error:
    #         return error
        
    #     costs_result_df  = result_df.copy() 
    #     energy_results_df = result_df.copy() 
    #     dfcols = result_df.columns.tolist()
         
    #     # Calculate and add costs  
    #     costs_result_df, energy_results_df = self._calculate_and_add_costs(result_df = costs_result_df, 
    #                                                                         col_mapping = col_mapping, 
    #                                                                         scenario_interventions = scenario_interventions, 
    #                                                                         # prob_external = prob_external,
    #                                                                         region =region, 
    #                                                                         scenario = scenario,
    #                                                                         return_statistics = return_statistics,
                                                                            
    #     )

     
    #     # extra_cols = ['wall_insulated', 'existing_loft_insulation', 'existing_floor_insulation', 'existing_window_upgrades']
    #     cost_cols, energy_cols = self._get_cols_scenario_intervention(scenario )
 
    #     cost_overlap = set(cost_cols).intersection(costs_result_df.columns)
    #     energy_overlap = set(energy_cols).intersection(energy_results_df.columns)

    #     if not cost_overlap:
    #         logger.warning(f"No overlap found between expected cost columns and DataFrame columns for scenario {scenario}.")
    #         logger.warning(f"Expected cost cols: {cost_cols}")
    #         logger.warning(f"Actual cost DF cols: {[x for x in costs_result_df.columns.tolist() if x not in dfcols ] }")

    #     if not energy_overlap:
    #         logger.warning(f"No overlap found between expected energy columns and DataFrame columns for scenario {scenario}.")
    #         logger.warning(f"Expected energy cols: {energy_cols}")
    #         logger.warning(f"Actual energy DF cols: {[x for x in energy_results_df.columns.tolist() if x not in dfcols ] }")

    #     costs_result_df = self._ensure_columns_exist(costs_result_df, cost_cols)
    #     energy_results_df = self._ensure_columns_exist(energy_results_df, energy_cols)
    #     energy_results_df = expand_dict_columns(energy_results_df)
    #     logger.debug(f'base cold: {base_cols}')
    #     c_df =  costs_result_df[ cost_cols ]
    #     e_df = energy_results_df[energy_cols]
    #     c_df = c_df.rename(
    #         columns=lambda c: f"{scenario}_{c}"
    #     )    
    #     e_df = e_df.rename(
    #         columns=lambda c: f"{scenario}_{c}"
    #     )    
    #     data = pd.concat(
    #         [result_df[base_cols], c_df, e_df ],
    #         axis=1
    #             )   
           
    #     return  data


    def calculate_building_costs_df_updated(self,
                                            df, 
                                            region, 
                                            scenario, 
                                            col_mapping=None, 
                                            return_statistics=None, 
                                          ):
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
        base_cols = result_df.columns.tolist() 
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
                                                                            return_statistics = return_statistics,
                                                                            
        )

        # # CHECK 1: Verify costs_result_df after _calculate_and_add_costs
        # logger.debug(f'Costs result DF shape after calculation: {costs_result_df.shape}')
        # new_cost_cols = [col for col in costs_result_df.columns if col not in dfcols]
        # logger.debug(f'New cost columns added: {new_cost_cols}')
        
        # for col in new_cost_cols:
        #     if costs_result_df[col].isna().all():
        #         logger.error(f'Cost column "{col}" is all NaN after calculation for scenario {scenario}!')
        #         raise ValueError(f'Cost column "{col}" contains all NaN values for scenario {scenario}')
        #     logger.debug(f'Cost column "{col}": mean={costs_result_df[col].mean():.4f}, '
        #                f'nan_count={costs_result_df[col].isna().sum()}')
        
        # # CHECK 2: Verify energy_results_df after _calculate_and_add_costs
        # logger.debug(f'Energy result DF shape after calculation: {energy_results_df.shape}')
        # new_energy_cols = [col for col in energy_results_df.columns if col not in dfcols]
        # logger.debug(f'New energy columns added: {new_energy_cols}')
        
        # for col in new_energy_cols:
        #     if energy_results_df[col].isna().all():
        #         logger.error(f'Energy column "{col}" is all NaN after calculation for scenario {scenario}!')
        #         raise ValueError(f'Energy column "{col}" contains all NaN values for scenario {scenario}')
            
        #     # Check for all zeros
        #     if (energy_results_df[col] == 0).all():
        #         logger.warning(f'Energy column "{col}" is all zeros for scenario {scenario}. '
        #                      f'This may indicate missing energy savings data.')
            
        #     # Check if column contains dict values (before expansion)
        #     if isinstance(energy_results_df[col].iloc[0], dict):
        #         logger.debug(f'Energy column "{col}" contains dict values - will be expanded')
        #     else:
        #         logger.debug(f'Energy column "{col}": mean={energy_results_df[col].mean():.4f}, '
        #                    f'nan_count={energy_results_df[col].isna().sum()}, '
        #                    f'zero_count={(energy_results_df[col] == 0).sum()}')
     
        # extra_cols = ['wall_insulated', 'existing_loft_insulation', 'existing_floor_insulation', 'existing_window_upgrades']
        cost_cols, energy_cols = self._get_cols_scenario_intervention(scenario )
 
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
        
        # # CHECK 3: Before expanding dict columns
        # logger.debug(f'Energy columns before dict expansion: {[c for c in energy_results_df.columns if c in energy_cols]}')
        
        energy_results_df = expand_dict_columns(energy_results_df)
        
        # # CHECK 4: After expanding dict columns
        # logger.debug(f'Energy result DF shape after dict expansion: {energy_results_df.shape}')
        # expanded_energy_cols = [col for col in energy_results_df.columns if col not in dfcols]
        # logger.debug(f'Energy columns after dict expansion: {expanded_energy_cols}')
        
        # # Verify expanded columns are not all NaN
        # for col in expanded_energy_cols:
        #     if col in energy_results_df.columns:
        #         if energy_results_df[col].isna().all():
        #             logger.error(f'Expanded energy column "{col}" is all NaN for scenario {scenario}!')
        #             raise ValueError(f'Expanded energy column "{col}" contains all NaN values for scenario {scenario}')
                
        #         # Check for all zeros
        #         if (energy_results_df[col] == 0).all():
        #             logger.warning(f'Expanded energy column "{col}" is all zeros for scenario {scenario}.')
                
        #         logger.debug(f'Expanded energy column "{col}": mean={energy_results_df[col].mean():.4f}, '
        #                    f'nan_count={energy_results_df[col].isna().sum()}, '
        #                    f'zero_count={(energy_results_df[col] == 0).sum()}')
        
        logger.debug(f'base cold: {base_cols}')
        c_df =  costs_result_df[ cost_cols ]
        e_df = energy_results_df[energy_cols]
        
        # CHECK 5: Verify final selection of columns
        logger.debug(f'Final cost columns selected: {c_df.columns.tolist()}')
        logger.debug(f'Final energy columns selected: {e_df.columns.tolist()}')
        
        for col in c_df.columns:
            if c_df[col].isna().all():
                logger.error(f'Final cost column "{col}" is all NaN for scenario {scenario}!')
                raise ValueError(f'Final cost column "{col}" contains all NaN values')
        
        for col in e_df.columns:
            if e_df[col].isna().all():
                logger.error(f'Final energy column "{col}" is all NaN for scenario {scenario}!')
                raise ValueError(f'Final energy column "{col}" contains all NaN values')
            
            if (e_df[col] == 0).all():
                logger.warning(f'Final energy column "{col}" is all zeros for scenario {scenario}.')
        
        c_df = c_df.rename(
            columns=lambda c: f"{scenario}_{c}"
        )    
        e_df = e_df.rename(
            columns=lambda c: f"{scenario}_{c}"
        )    
        
        # CHECK 6: Final verification before concatenation
        logger.debug(f'Final renamed cost columns: {c_df.columns.tolist()}')
        logger.debug(f'Final renamed energy columns: {e_df.columns.tolist()}')
        
        data = pd.concat(
            [result_df[base_cols], c_df, e_df ],
            axis=1
                )   
        
        # CHECK 7: Final output verification
        logger.debug(f'Final concatenated data shape: {data.shape}')
        final_cols = data.columns.tolist()
        logger.debug(f'Total columns in final output: {len(final_cols)}')
        
        # Verify scenario-specific energy columns exist in final output
        scenario_energy_cols = [col for col in final_cols if scenario in col and any(x in col for x in ['gas', 'electricity', 'elec'])]
        logger.debug(f'Scenario {scenario} energy columns in final output: {scenario_energy_cols}')
        
        if not scenario_energy_cols:
            logger.error(f'No energy columns found in final output for scenario {scenario}!')
            raise ValueError(f'No energy columns in final output for scenario {scenario}')
           
        return  data



