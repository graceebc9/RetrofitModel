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
from .RetrofitUtils import calc_est_flats_building 
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
        self.YEARS = 5 
        self.GAS_FACTOR= 0.18      
        self.ELEC_FACTOR = 0.19338  

        if self.n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.n_samples < 100:
            logger.warning(f"Warning: n_samples={self.n_samples} is low. Consider using 100+ for stable results.")

        # 3. PULL EPISTEMIC FACTORS AND APPLY TO RETROFITENERGY
        
        # Factors defining the technical performance of wall measures
        int_factor = self.epistemic_scenario.get('solid_wall_internal_improvement_factor'  )
        ext_factor = self.epistemic_scenario.get('solid_wall_external_improvement_factor')
        
        area_choice_setting = self.epistemic_scenario.get('area_based_choice') 
        
        self.area_col = f'scaled_area_{area_choice_setting}'
        self.gas_col =f'gas_scaled_scaled_area_{area_choice_setting}'
        self.elec_col = f'elec_scaled_scaled_area_{area_choice_setting}'

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
            # --- FIX: Use .get() to check if the scenario exists ---
            selected_scenario = self.retrofit_packages.get(scenario)

            # --- FIX: Check if the scenario itself was found ---
            if selected_scenario is None:
                return {'error': f'Scenario "{scenario}" not found in retrofit_packages.'}

            scenario_interventions = selected_scenario.get('interventions', [])
            if not scenario_interventions:
                return {'error': f'No interventions defined for scenario "{scenario}"'}
            
            return scenario_interventions

 

    def _prepare_dataframe(self, df, col_mapping):
        # ... (Original logic) ...
        result_df = df.copy()
        result_df = result_df[result_df[col_mapping['building_type']] != 'Domestic outbuilding']
        
        fp_mean =  self.epistemic_scenario.get('flat_fp_mean') 
        fp_std =  self.epistemic_scenario.get('flat_fp_std') 
        eff_mean =  self.epistemic_scenario.get('flat_eff_mean') 
        eff_std = self.epistemic_scenario.get('flat_eff_std') 
        
        result_df['est_num_flats'] = result_df.apply(
            lambda row: calc_est_flats_building(
                building_footprint_area=row[col_mapping['footprint_area']],
                typology_col=row[col_mapping['building_type']] ,
                floor_count=row[col_mapping['floor_count']],
                fp_mean=fp_mean, 
                fp_std = fp_std, 
                eff_mean=eff_mean, 
                eff_std=eff_std,

                
                
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

    def _calculate_single_statistic(self, samples: np.ndarray, stat: str, capex: bool = False) -> float:
        
        if not isinstance(samples, np.ndarray):
            try: samples = np.array(samples)
            except Exception as e: raise TypeError(f"Cannot convert samples to numpy array: {e}")
        if samples.size == 0:
            raise ValueError("Cannot calculate statistics on empty array")
        if np.all(np.isnan(samples)):
                return np.nan
        if capex:
            samples = [item for item in samples if item >= 0]
            
        try:    
            if stat == 'mean':
                result = np.nanmean(samples)
                
            elif stat == 'median' or stat == 'p50':
                result = np.nanmedian(samples)
            elif stat == 'std':
                result = np.nanstd(samples)
            elif stat.startswith('p'):
                percentile = int(stat[1:])
                result = np.nanpercentile(samples, percentile)
            else:
                raise ValueError(f"Unknown statistic: {stat}")
            return result
        except Exception as e:
            logging.error(f"Error calculating {stat} (nan-safe): {e}")
            raise

    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # CORE MODIFI-  
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

 
    def bootstrap_ratio(self , capex, carbon, n_boot=1000):
        # Extracts the two columns as numpy arrays for speed
    
        n = len(capex)
        
        ratios = []
        
        # 1. Resample indices n_boot times
        # This creates a matrix of random indices (n_boot x n)
        indices = np.random.randint(0, n, size=(n_boot, n))
        
        # 2. Calculate Means for all resamples at once
        # This is much faster than looping
        boot_mean_capex = capex[indices].mean(axis=1)
        boot_mean_carbon = carbon[indices].mean(axis=1)
        
        # 3. Calculate Ratio (handle potential div by zero if carbon sums to 0)
        # Using a tiny epsilon just in case, though unlikely in bootstrap mean
        boot_ratios = boot_mean_capex / (boot_mean_carbon + 1e-9)
        
        return pd.Series({
            'ratio_mean': boot_ratios.mean(),
            'ratio_p05': np.percentile(boot_ratios, 5),
            'ratio_p95': np.percentile(boot_ratios, 95),
            'ratio_p50': np.percentile(boot_ratios, 50),
            'ratio_std': boot_ratios.std()
        })

 
    def calculate_joint_scenario_statistics(self,
                                            joint_intervention,  # A single string, e.g., 'loft+wall'
                                            building_chars,
                                            wall_insulation_type,
                                            typology,
                                            age_band,
                                            region,
                                            return_statistics,
                                            scenario_name,
                                            total_gas_derived,  # Baseline gas
                                            total_elec_derived): # Baseline elec
        """
        Calculates Monte Carlo statistics for a single, resolved joint intervention
        using a robust hybrid approach for ratio statistics (cost-per-unit).
        
        Assumes that the energy sampling function returns the final, 
        multiplicatively-aggregated percentage savings.
        """
        
        # --- 1. Get Epistemic Factors ---
        beta_TS = self.epistemic_scenario.get('time_scale_bias')
        beta_DEC = self.epistemic_scenario.get('decile_misclassification_bias')
        decile_scale = self.decile_risk_scaling.get(building_chars.avg_gas_percentile)
        effective_beta_DEC = beta_DEC * decile_scale
        
        roof_scaling = self.retrofit_config.existing_intervention_probs['roof_scaling_factor']
        
        stats_results = {}
        
        try:
            # --- 2. Get Cost Samples (Absolute) ---
            total_cost_samples = self.sample_intervention_cost_monte_carlo(
                intervention=[joint_intervention],
                building_chars=building_chars,
                typology=typology,
                age_band=age_band,
                region=region,
                wall_insulation_type=wall_insulation_type,
                cost_col_name=scenario_name,
            )
 

            if total_cost_samples is None:
                logger.warning(f"Cost sampling returned None for {joint_intervention}. Using zeros.")
                total_cost_samples = np.zeros(self.n_samples)

            # --- 3. Get Energy Samples (Percentage) ---
            energy_samples_dict = self.energy_config.sample_intervention_energy_savings_monte_carlo(
                intervention=joint_intervention,
                building_chars=building_chars,
                region=region,
                n_samples=self.n_samples,
                roof_scaling=roof_scaling,
                wall_type=wall_insulation_type,
            )
            logger.debug('energy_samples_dict')
            logger.debug(energy_samples_dict )

            total_gas_perc_samples = energy_samples_dict.get('gas') if isinstance(energy_samples_dict, dict) else energy_samples_dict
            total_elec_perc_samples = energy_samples_dict.get('electricity') if isinstance(energy_samples_dict, dict) else None

            if total_gas_perc_samples is None:
                logger.warning(
                    f"No 'gas' savings returned from energy sampler for intervention '{joint_intervention}'. "
                    f"This will result in 0 gas savings and NaN cost_per_gas_kwh."
                )
                total_gas_perc_samples = np.zeros(self.n_samples)
            if total_elec_perc_samples is None:
                logger.debug(
                    f"No 'electricity' savings returned from energy sampler for intervention '{joint_intervention}'."
                )
                total_elec_perc_samples = np.zeros(self.n_samples)

            # --- 4. Apply Epistemic Adjustments (to total percentages) ---
            final_gas_perc_samples = (total_gas_perc_samples + effective_beta_DEC) * beta_TS
            # technicall elec shoould have a decile mis class bia but cba 
            final_elec_perc_samples = (total_elec_perc_samples) * beta_TS
            
            # --- 5. Convert to Absolute kWh Savings ---
            final_gas_abs_kwh_samples = final_gas_perc_samples * total_gas_derived
            final_elec_abs_kwh_samples = final_elec_perc_samples * total_elec_derived
            # final_total_energy_abs_kwh_samples = final_gas_abs_kwh_samples + final_elec_abs_kwh_samples

            # convert to tons of co2 
            final_gas_abs_ton_co2_samples = (final_gas_abs_kwh_samples * self.YEARS * self.GAS_FACTOR) / 1000  * -1
            final_elec_abs_ton_co2_samples = (final_elec_abs_kwh_samples * self.YEARS * self.ELEC_FACTOR) / 1000  * -1 
            final_total_energy_abs_co2_ton_samples = (final_gas_abs_ton_co2_samples + final_elec_abs_ton_co2_samples )
            
            # chekc if cost or energy cost 2 area ll nan 
            logger.debug(f'nan check cost: {np.isnan(np.sum(total_cost_samples)) }') 
            logger.debug(f'nan check energy : {np.isnan(np.sum(final_total_energy_abs_co2_ton_samples)) }') 
            # capex_per_ton =  total_cost_samples.mean() / final_total_energy_abs_co2_ton_samples.mean() 
            capex_per_ton = self.bootstrap_ratio(capex= total_cost_samples ,  carbon = final_total_energy_abs_co2_ton_samples )
            # ---  6. Define BASE arrays to get stats for ---
            base_sample_arrays = {
                f"cost_{scenario_name}": total_cost_samples,
                f"gas_saving_perc_{scenario_name}": final_gas_perc_samples,
                f"elec_saving_perc_{scenario_name}": final_elec_perc_samples,
                # f"gas_saving_abs_kwh_{scenario_name}": final_gas_abs_kwh_samples,
                # f"elec_saving_abs_kwh_{scenario_name}": final_elec_abs_kwh_samples,
                # f"total_energy_saving_abs_kwh_{scenario_name}": final_total_energy_abs_kwh_samples,
                f"gas_abs_ton_co2_samples_{scenario_name}": final_gas_abs_ton_co2_samples, 
                f"elec_abs_ton_co2_samples_{scenario_name}": final_elec_abs_ton_co2_samples, 
                f"total_energy_abs_co2_ton_samples_{scenario_name}": final_total_energy_abs_co2_ton_samples, 
                # f"capex_per_net_ton_co2_{scenario_name}" : capex_per_ton, 
            }

        except Exception as e:
            logger.error(f"Error in joint sampling for scenario {scenario_name} (intervention: {joint_intervention}): {e}")
            base_sample_arrays = {f"cost_{scenario_name}": np.array([np.nan])} # Dummy

        # --- 7. Calculate Statistics on BASE Sample Arrays ---
        for prefix, samples in base_sample_arrays.items():
            if np.all(pd.isna(samples)):
                logger.debug(f'Warning, all samples in join calc are nan : {prefix}')
                for stat in return_statistics:
                    stats_results[f"{prefix}_{stat}"] = np.nan
            else:
                for stat in return_statistics:
                    col_name = f"{prefix}_{stat}"
                    try:
                        # This assumes _calculate_single_statistic uses nan-safe functions (np.nanmean, etc.)
                        if 'capex' in prefix:
                            # remove negatrive values from stats - in thoery the standard deviaiton should penalise 
                            stats_results[col_name] = self._calculate_single_statistic(samples, stat, capex=True )
                        else:
                            stats_results[col_name] = self._calculate_single_statistic(samples, stat)
                        
                    except ValueError as stat_error:
                        logger.error(f"Invalid statistic '{stat}' for {prefix}: {stat_error}")
                        stats_results[col_name] = np.nan
                            
        # --- 8. [NEW HYBRID LOGIC] Calculate RATIO statistics ---
        stats_results[f'capex_per_net_ton_co2_{scenario_name}_mean'] = capex_per_ton['ratio_mean']
        stats_results[f'capex_per_net_ton_co2_{scenario_name}_std'] = capex_per_ton['ratio_std']
        stats_results[f'capex_per_net_ton_co2_{scenario_name}_p50'] = capex_per_ton['ratio_p50']
        stats_results[f'capex_per_net_ton_co2_{scenario_name}_p95'] = capex_per_ton['ratio_p95']
        stats_results[f'capex_per_net_ton_co2_{scenario_name}_p5'] = capex_per_ton['ratio_p05']

        # --- 8a. Define the "good" savings threshold (successful reduction) ---
        # Savings are negative, so "good" is < -1e-6
        # min_savings_threshold = -1e-6 

        capex_failure_mask = final_total_energy_abs_co2_ton_samples <= 0 
        n_capex_failures = np.sum(capex_failure_mask)
        capex_failure_rate = n_capex_failures / self.n_samples
 
        stats_results[f"capex_saving_failure_rate_{scenario_name}"] = capex_failure_rate
     
        
       
        stats_results['selected_wall_insulation_type'] = wall_insulation_type
        
        
        return stats_results

 
    def calculate_row_statistics(self,
                                row,
                                col_mapping,
                                scenario_interventions,
                                scenario_name,
                                region,
                                return_statistics):
        """
        Calculate Monte Carlo cost, energy, and cost/energy statistics
        for a single building row and a given intervention scenario.
        """
        
        # --- 1. Data Validation ---
        # ADD YOUR BASELINE COLUMNS HERE


        baseline_cols = [self.gas_col, self.elec_col] 
        
        required_cols = ['floor_count', 'gross_external_area', 'gross_internal_area', 'inferred_wall_type', 'inferred_insulation_type',
                        'footprint_circumference', 'building_type', 'age_band', 'building_footprint_area', 'avg_gas_percentile']
        
        # Use col_mapping for all required cols
        mapped_required = [col_mapping[col] for col in required_cols]
        mapped_required.extend(baseline_cols) # Add baselines (assuming they are not in col_mapping)
        
        missing_cols = [col for col in mapped_required if col not in row.index]
        if missing_cols:
            raise ValueError(f'Missing columns: {missing_cols}')
            
        nan_cols = [col for col in mapped_required if pd.isna(row[col])]
        if nan_cols:
            raise ValueError(f'NaN values found in required columns: {nan_cols}')
        
        # --- 2. Extract Building Characteristics ---
        floor_count = int(row[col_mapping['floor_count']])
        gross_external_area = float(row[col_mapping['gross_external_area']])
        gross_internal_area = float(row[col_mapping['gross_internal_area']])
        footprint_circumference = float(row[col_mapping['footprint_circumference']])
        building_footprint_area = float(row[col_mapping['building_footprint_area']])
        avg_gas_percentile = int(row[col_mapping['avg_gas_percentile']])
        typology = row[col_mapping['building_type']]
        age_band = row[col_mapping['age_band']]
        raw_flat_count = row.get(col_mapping['flat_count'])
        flat_count = int(raw_flat_count) if pd.notna(raw_flat_count) and raw_flat_count > 0 else 1
        
        # Extract baseline energy
        total_gas_derived = float(row[self.gas_col])
        total_elec_derived = float(row[self.elec_col])
        
        building_chars = BuildingCharacteristics(
            floor_count=floor_count,
            gross_external_area=gross_external_area,
            gross_internal_area=gross_internal_area,
            footprint_circumference=footprint_circumference,
            flat_count=flat_count,
            building_footprint_area=building_footprint_area,
            avg_gas_percentile=avg_gas_percentile,
            typology=typology,
        )
        
        # --- 3. Resolve Interventions ---
        wall_type = str(row['inferred_wall_type']).lower().strip()
        insulation_type = str(row['inferred_insulation_type']).lower().strip()
        
        selected_wall_insulation = insulation_type
        interventions_to_calculate = []
        
        for intervention in scenario_interventions:
            if intervention == 'WALL_INSULATION':
                if selected_wall_insulation == 'cavity_wall_insulation':
                    interventions_to_calculate.append('cavity_wall_percentile')
                elif selected_wall_insulation == 'internal_wall_insulation':
                    interventions_to_calculate.append('solid_wall_internal_percentile')
                elif selected_wall_insulation == 'external_wall_insulation':
                    interventions_to_calculate.append('solid_wall_external_percentile')
            else:
                interventions_to_calculate.append(intervention)
                
        # --- 4. Validation Check ---
        if len(interventions_to_calculate) != 1:
            raise ValueError(
                f"Scenario '{scenario_name}' resolved to {len(interventions_to_calculate)} "
                f"interventions ({interventions_to_calculate}). "
                "This model is designed for one joint intervention per scenario."
            )
        
        joint_intervention_name = interventions_to_calculate[0]
                
        # --- 5. Call the New Unified Statistics Function ---
        logger.debug(f"Starting joint statistics for scenario: {scenario_name} ({joint_intervention_name})")
        
        combined_stats_dict = self.calculate_joint_scenario_statistics(
            joint_intervention=joint_intervention_name,
            building_chars=building_chars,
            wall_insulation_type=selected_wall_insulation,
            typology=typology,
            age_band=age_band,
            region=region,
            return_statistics=return_statistics,
            scenario_name=scenario_name,
            total_gas_derived=total_gas_derived,
            total_elec_derived=total_elec_derived
        )
        
        logger.debug('Calculation complete. Returning results.')
        logger.debug('combined_stats_dict')
        logger.debug(combined_stats_dict.keys())
        
        # --- 6. Return as pd.Series ---
        combined_result = pd.Series(combined_stats_dict)
        combined_result.index = combined_result.index.astype(str)
        
        return combined_result
 
 
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
            
            # logger.debug(f"{cost_col_name} samples: mean=£{samples.mean():,.0f}")
            return samples
            
        except ValueError as e:
            logger.error(f"Error sampling {cost_col_name}: {e}")
            raise
     

 
 
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
        result_df_reset = result_df.reset_index(drop=False)
        original_index_name = result_df.index.name
        energy_res = result_df.copy() 
        logger.debug(f'Col mapping is " {col_mapping }')
        results  = result_df.apply(
            lambda row: self.calculate_row_statistics(row, 
                                                        col_mapping=col_mapping, 
                                                        scenario_interventions=scenario_interventions, 
                                                        region=region,  
                                                        return_statistics=return_statistics,
                                                        scenario_name=scenario,
                                                            ), axis=1
                                    )
        logger.debug('Row Calc complete')
        logger.debug('Row cols: ')
        logger.debug(results.columns.tolist() )
        
        cost_cols = [col for col in results.columns if col.startswith('cost_') or col.startswith('capex')]
        energy_cols = [col for col in results.columns if not col.startswith('cost_')]
        
        cost_results = results[cost_cols]
        energy_results = results[energy_cols]
    
         
        
        for col in energy_results.columns:
            if energy_results[col].isna().all():
                logger.error(f'Energy column "{col}" is all NaN in energy_results!')
                raise ValueError(f'Energy column "{col}" contains all NaN values. '
                               f'Check energy calculations for scenario: {scenario}')
            
            # Also check for all zeros (which might indicate a problem)
            if (energy_results[col] == 0).all():
                logger.warning(f'Energy column "{col}" is all zeros in energy_results. '
                             f'This may indicate missing energy savings data.')
            
        
        logger.debug('Starting validations on costs results colmns ')
        for col in cost_results.columns:
            if cost_results[col].isna().all():
                
                # [!!! NEW VALIDATION LOGIC !!!]
                # Check if it's a ratio column. NaN is a valid result if savings are 0.
                if "cost_per_" in col:
                    logger.warning(
                        f'Cost-per-unit column "{col}" is all NaN. This is likely '
                        f'because the denominator (energy savings) is 0 for all rows. '
                        f'This is expected for interventions with no savings (e.g., elec for wall_installation).'
                    )
                else:
                    # It's a base cost column (e.g., cost_wall_installation_mean).
                    # If THIS is all NaN, it's a real problem.
                    logger.error(f'Cost column "{col}" is all NaN in cost_results!')
                    raise ValueError(f'Cost column "{col}" contains all NaN values. '
                                   f'Check cost calculations for scenario: {scenario}')
        
        self._add_cost_columns(result_df, cost_results)
        # Add energy columns (individual interventions)
        self._add_individual_energy_columns(energy_res, energy_results)
        logger.debug('About to return results ')
        return result_df, energy_res
 

    def _add_cost_columns(self, result_df, cost_results):
        """Add individual and total cost columns to result DataFrame."""
        # Add individual intervention cost columns
 
        for col in cost_results.columns:
      
            result_df[col] = cost_results[col]
        
  

    def _add_individual_energy_columns(self, result_df, energy_results):
        """Add individual intervention energy columns to result DataFrame."""
        for col in energy_results.columns:
            result_df[col] = energy_results[col]

  

    def _get_cols_scenario_intervention(self, scenario_str, stats=['mean', 'std', 'p5', 'p50', 'p95'], 
                                     metric_types=['gas_abs_ton_co2_samples', 'elec_abs_ton_co2_samples','total_energy_abs_co2_ton_samples', 'gas_saving_perc', 'elec_saving_perc']):
        """
        Get column names for scenario interventions.
        
        Args:
            scenario_str: Scenario name
            stats: List of statistics (mean, std, p5, p50, p95)
            metric_types: List of metric types to include. Options:
                - 'gas_saving_perc', 'elec_saving_perc'
                - 'gas_saving_abs_kwh', 'elec_saving_abs_kwh'
                - 'total_energy_saving_abs_kwh'
        """
        cost_cols = [] 
        energy_cols = [] 
        elec = True 
        if scenario_str == 'wall_installation':
            interventions = ['wall_installation']
           
        elif scenario_str == 'loft_installation':
            interventions = ['loft_percentile']
            
        elif scenario_str == 'joint_loft_wall_add':
            interventions = ['joint_loft_wall_add']
            
        elif scenario_str == 'joint_loft_wall_decay':
            interventions = ['joint_loft_wall_decay']
            
        elif scenario_str == 'heat_pump_only':
            interventions = ['heat_pump_percentile']
            
        elif scenario_str == 'join_heat_ins_decay':
            interventions = ['join_heat_ins_decay']
            
        elif scenario_str == 'join_heat_ins_add':
            interventions = ['join_heat_ins_add']

        elif scenario_str == 'joint_heat_wall_decay':
            interventions = ['joint_heat_wall_decay']
            
        elif scenario_str == 'joint_heat_loft_decay':
            interventions = ['joint_heat_loft_decay']
        

        else:
            raise Exception(f'Need to define the interventions for scenario ({scenario_str}) in RetrofitModel _get_cols_scenario_intervention')
        
        single_intervention = len(interventions) == 1
        
        for iint in interventions:
            # Cost columns
            for s in stats:
                if single_intervention:
                    cost_cols.append(f'cost_{scenario_str}_{s}')
                    # cost_cols.append(f'cost_per_elec_kwh_{scenario_str}_{s}')
                    cost_cols.append(f'capex_per_net_ton_co2_{scenario_str}_{s}')
                    # cost_cols.append(f'cost_per_total_energy_kwh_{scenario_str}_{s}')
                     
                else:
                    cost_cols.append(f'{iint}_cost_{s}')
            
            # Energy columns - new format: {metric_type}_{scenario}_{stat}
            for metric in metric_types:
                for s in stats:
                    if single_intervention:
                        energy_cols.append(f'{metric}_{scenario_str}_{s}')
                    else:
                        energy_cols.append(f'{metric}_{iint}_{s}')
        cost_cols.append(f'capex_saving_failure_rate_{scenario_str}')
        # energy_cols.append(f"elec_saving_failure_rate_{scenario_str}")
        return cost_cols, energy_cols
        
 
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
                                            col_mapping, 
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
         
        try:
            # --- FIX: Wrap this call in a try/except block ---
            result_df = self._prepare_dataframe(df, col_mapping)
        
        except KeyError as e:
            # This catches the 'building_type' error from _prepare_dataframe
            return {'error': f'DataFrame is missing required mapped column: {e}'}
        
        base_cols = result_df.columns.tolist() 
        # Validate DataFrame columns
        error = self._validate_dataframe_columns(result_df, col_mapping)
        if error:
            return error
        
        costs_result_df  = result_df.copy() 
        energy_results_df = result_df.copy() 
        dfcols = result_df.columns.tolist()
        logger.debug(dfcols)
        # Calculate and add costs  
        costs_result_df, energy_results_df = self._calculate_and_add_costs(result_df = costs_result_df, 
                                                                            col_mapping = col_mapping, 
                                                                            scenario_interventions = scenario_interventions, 
                                                                           
                                                                            region =region, 
                                                                            scenario = scenario,
                                                                            return_statistics = return_statistics,
                                                                            
        )
        logger.debug('Results retuend,starting to prcess ')
        logger.debug('costs_result_df')
        logger.debug(costs_result_df.columns.tolist() )
        logger.debug('energy_results_df')
        logger.debug(energy_results_df.columns.tolist() ) 

        # extra_cols = ['wall_insulated', 'existing_loft_insulation', 'existing_floor_insulation', 'existing_window_upgrades']
        cost_cols, energy_cols = self._get_cols_scenario_intervention(scenario )
 

        cost_overlap = set(cost_cols).intersection(costs_result_df.columns)
        energy_overlap = set(energy_cols).intersection(energy_results_df.columns)
        logger.debug('Next validations ??/ ')
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
        
        logger.debug('Final col extractions .. ')
        
        c_df =  costs_result_df[ cost_cols ]
        e_df = energy_results_df[energy_cols]
        
        # CHECK 5: Verify final selection of columns
        # logger.debug(f'Final cost columns selected: {c_df.columns.tolist()}')
        # logger.debug(f'Final energy columns selected: {e_df.columns.tolist()}')
        
        for col in c_df.columns:
            if c_df[col].isna().all():
                
                
                if "cost_per_" in col or 'capex_per' in col:
                    logger.warning(
                        f'Final cost-per-unit column "{col}" is all NaN. This is likely '
                        f'because the denominator (energy savings) is 0 for all rows. '
                        f'Skipping error raise for this ratio column.'
                    )
                else:
                    # Base cost column is NaN. This is a fatal error.
                    logger.error(f'Final cost column "{col}" is all NaN for scenario {scenario}!')
                    raise ValueError(f'Final cost column "{col}" contains all NaN values')
        
            for col in e_df.columns:
                if e_df[col].isna().all():
                    if 'elec' in col:
                        None 
                        # logger.warning(
                        #     # f'Final energy column "{col}" is all NaN for scenario {scenario}. '
                        #     f'This may be expected for electricity columns.'
                        # )
                    else:
                        logger.error(f'Final energy column "{col}" is all NaN for scenario {scenario}!')
                        raise ValueError(f'Final energy column "{col}" contains all NaN values')
                
                elif (e_df[col] == 0).all():
                    # logger.warning(f'Final energy column "{col}" is all zeros for scenario {scenario}.')
                    None 
        c_df = c_df.rename(
            columns=lambda c: f"{scenario}_{c}"
        )    
        e_df = e_df.rename(
            columns=lambda c: f"{scenario}_{c}"
        )    
 
        logger.debug('Final concat: ')
        data = pd.concat(
            [result_df[base_cols], c_df, e_df ],
            axis=1
                )   
        
        # CHECK 7: Final output verification
        # logger.debug(f'Final concatenated data shape: {data.shape}')
        final_cols = data.columns.tolist()
        # logger.debug(f'Total columns in final output: {len(final_cols)}')
        
        # Verify scenario-specific energy columns exist in final output
        scenario_energy_cols = [col for col in final_cols if scenario in col and any(x in col for x in ['gas', 'electricity', 'elec'])]
        # logger.debug(f'Scenario {scenario} energy columns in final output: {scenario_energy_cols}')
        
        if not scenario_energy_cols:
            logger.error(f'No energy columns found in final output for scenario {scenario}!')
            raise ValueError(f'No energy columns in final output for scenario {scenario}')
           
        return  data



