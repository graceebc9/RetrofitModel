from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np 
from scipy import stats 
import logging 

# Import necessary components (Assuming these are correct)
from .PreProcessRetrofit import vectorized_process_buildings 
from .RetrofitModel2D import RetrofitModel2D    

logger = logging.getLogger(__name__)

@dataclass
class RetrofitScenarioGenerator2DMC:
    """
    2DMC Outer Loop Driver: Generates and processes retrofit scenarios across 
    N_epistemic runs, managing systematic uncertainty.
    """
    
    # 1. NEW INPUTS for the Outer Loop
    n_epistemic_runs: int = 50 
    
    # 2. Reference to the sampler function (Placeholder - requires the function we defined)
    epistemic_sampler: Any = None 

    def process_dataframe_scenarios(self,
                                df: pd.DataFrame,
                                scenarios: list, 
                                model_class: Any,  # The RetrofitModel (now Inner Loop) class
                                region: str,
                                random_seed: Optional[int] = None,
                                col_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Process scenarios for all buildings across N_epistemic_runs.
        
        The Outer Loop iterates through Epistemic Scenarios.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with building data
        scenarios: list
            List of scenario names (e.g., ['scenario_A', 'scenario_B'])
        model : Type[RetrofitModel]
            The model to be used 
        region : str
            Region code
        
        Returns:
        --------
        pd.DataFrame : DataFrame containing statistics aggregated across 
                       all N_epistemic_runs for all buildings and scenarios.
        """
        
        if self.epistemic_sampler is None:
            raise ValueError("Epistemic sampler function is required but not provided.")

        typ_config = model_class.retrofit_config
        
        # 1. Define Column Mapping 
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
            'avg_gas_decile': 'avg_gas_decile',
            'cons_bool': 'conservation_area_bool', 
        }
        if col_mapping:
            default_mapping.update(col_mapping)
        col_mapping = default_mapping
        
        result_df = df.copy()
        
        # 2. Initial Vectorized Processing (Only needs to run once)
        # This sets inferred wall types, insulation status, etc., which are NOT epistemic
        logger.debug('Vectorising preproess ... ')
        df_typ = vectorized_process_buildings(
            result_df=result_df,
            col_mapping=col_mapping, 
            config=typ_config, 
            random_seed=random_seed
        )
        
        # 3. GENERATE EPISTEMIC SCENARIOS (Outer Loop Setup)
        logger.debug('Complete. starting to sample the runs .. ')
        # Use a seed for the Outer Loop itself for scenario reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            
        epistemic_scenarios_df = self.epistemic_sampler(self.n_epistemic_runs)
        
        # List to store the results of each Outer Loop run (N_epistemic dataframes)
        all_epistemic_results = []
        logger.debug('Sampled scenarios. starting outer loop. ')
        # 4. START THE OUTER LOOP ITERATION  
        
        logger.info(f"Starting {self.n_epistemic_runs} Epistemic Outer Loop runs...")
        
        for run_idx, scenario_row in epistemic_scenarios_df.iterrows():
            epistemic_scenario = scenario_row.to_dict()
            logger.info(f"--- Running Outer Loop Scenario {run_idx + 1}/{self.n_epistemic_runs} ---")
            
            # 5. INSTANTIATE THE INNER LOOP MODEL (RetrofitModel)
            
            # This creates a NEW RetrofitModel instance for each Outer Loop run.
            # It FIXES the Epistemic factors for all N_aleatory runs within this model instance.
            model_instance = model_class(
                retrofit_config=typ_config,
                n_samples=typ_config.n_samples, # Assuming n_samples is stored in typ_config now
                epistemic_scenario=epistemic_scenario
            )
            
            # 6. RUN ALL SCENARIOS FOR THIS FIXED EPISTEMIC SCENARIO
            
            # Create a base result DataFrame for this run, containing building IDs
            run_results = df_typ.copy()
            
            for scenario in scenarios:
                logger.debug(f"  - Calculating Scenario: {scenario}")
                
                # Call the vectorised calculation function (requires update in RetrofitModel)
                scenario_results = model_instance.calculate_building_costs_df_updated(
                    df=df_typ, 
                    region=region,
                    scenario=scenario,
                )
                
                # Check for errors and merge results 
                if isinstance(scenario_results, dict) and 'error' in scenario_results:
                    logger.warning(f"Scenario {scenario} failed: {scenario_results['error']}")
                    continue
                
    
                # # Merge scenario results into the run_results DataFrame (on index or ID)
                # run_results = pd.merge(
                #     run_results,
                #     scenario_results,
                #     left_index=True, # Assuming your calculate_building_costs_df_updated preserves index
                #     right_index=True,
                #     how='left'
                # )
                    
                # Identify only the NEW scenario-specific output columns 
                # (exclude columns that already exist in df_typ)
                output_columns = [col for col in scenario_results.columns 
                                if col not in df_typ.columns]
                
                # Merge only the new output columns
                run_results = pd.merge(
                    run_results,
                    scenario_results[output_columns],
                    left_index=True,
                    right_index=True,
                    how='left'
                )

            # 7. TAG AND STORE RESULTS
            
            # Add the run index and the Epistemic factors themselves for later analysis
            run_results['epistemic_run_id'] = run_idx
            for k, v in epistemic_scenario.items():
                run_results[f'epistemic__{k}'] = v
            
            all_epistemic_results.append(run_results)
            
        # 8. AGGREGATION (Final Output)
        
        # Concatenate all N_epistemic results into one massive DataFrame
        final_2dmc_output = pd.concat(all_epistemic_results, ignore_index=True)
        
        logger.info(f"2DMC processing complete. Final DataFrame shape: {final_2dmc_output.shape}")

        return final_2dmc_output