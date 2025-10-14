# At the top of retrofit_calc.py
from .logging_config import get_logger

logger = get_logger(__name__)

import geopandas as gpd 
import numpy as np
import pandas as pd
from typing import Dict, Optional

class PreProcessRetrofit:
    """Vectorized version of building intervention sampling"""
    
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : object with insulation_by_age_band, mixed_age_probabilities, 
                 existing_intervention_probs attributes
        """
        self.config = config
    
    def determine_wall_types_vectorized(self, age_bands: pd.Series, random_seed: int) -> pd.Series:
        """
        Vectorized wall type determination for all buildings at once.
        
        Parameters:
        -----------
        age_bands : pd.Series
            Series of age bands for all buildings
        random_seed : int
            Seed for reproducibility
            
        Returns:
        --------
        pd.Series : Wall types ('cavity_wall' or 'solid_wall')
        """
        rng = np.random.RandomState(random_seed)
        n = len(age_bands)
        
        # Initialize result array
        wall_types = np.empty(n, dtype=object)
        
        # Get unique age bands to process efficiently
        for age_band in age_bands.unique():
            mask = age_bands == age_band
            age_data = self.config.insulation_by_age_band.get(age_band, {})
            
            if age_data.get('mixed_scenarios', False):
                # Mixed scenario: sample based on probabilities
                wall_type_probs = self.config.mixed_age_probabilities.get(age_band, {
                    'cavity_wall_insulation': 0.7,
                    'internal_wall_insulation': 0.3
                })
                prob_cavity = wall_type_probs.get('cavity_wall_insulation', 0.7)
                
                # Sample all at once for this age band
                samples = rng.random(mask.sum()) < prob_cavity
                wall_types[mask] = np.where(samples, 'cavity_wall', 'solid_wall')
            else:
                # Non-mixed: deterministic based on primary_wall_type
                primary_wall_type = age_data.get('primary_wall_type', 'cavity_wall')
                wall_type = 'solid_wall' if primary_wall_type == 'solid_wall' else 'cavity_wall'
                wall_types[mask] = wall_type
        
        return pd.Series(wall_types, index=age_bands.index)
    
    def check_walls_already_insulated_vectorized(self, 
                                                  age_bands: pd.Series, 
                                                  wall_types: pd.Series,
                                                  random_seed: int) -> pd.Series:
        """
        Vectorized check for existing wall insulation.
        
        Parameters:
        -----------
        age_bands : pd.Series
            Series of age bands
        wall_types : pd.Series
            Series of wall types
        random_seed : int
            Seed for reproducibility
            
        Returns:
        --------
        pd.Series : Boolean series indicating if walls are already insulated
        """
        rng = np.random.RandomState(random_seed + 1)  # Different seed for independence
        n = len(age_bands)
        
        # Get all probabilities at once
        probs = np.zeros(n)
        
        for idx, (age_band, wall_type) in enumerate(zip(age_bands, wall_types)):
            age_data = self.config.insulation_by_age_band.get(age_band, {})
            existing_prob = age_data.get('existing_insulation_prob', 0.0)
            
            if isinstance(existing_prob, dict):
                insulation_type = 'cavity_wall_insulation' if wall_type == 'cavity_wall' else 'internal_wall_insulation'
                probs[idx] = existing_prob.get(insulation_type, 0.0)
            else:
                probs[idx] = existing_prob
        
        # Sample all at once
        return pd.Series(rng.random(n) < probs, index=age_bands.index)
    
    def sample_existing_interventions_vectorized(self, 
                                                  n_buildings: int,
                                                  random_seed: int) -> pd.DataFrame:
        """
        Vectorized sampling of existing interventions (loft, floor, windows).
        
        Parameters:
        -----------
        n_buildings : int
            Number of buildings
        random_seed : int
            Seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame : Boolean columns for each intervention type
        """
        rng = np.random.RandomState(random_seed + 2)  # Different seed
        
        results = {}
        for intervention, prob in self.config.existing_intervention_probs.items():
            results[intervention] = rng.random(n_buildings) < prob
        
        return pd.DataFrame(results)


def vectorized_process_buildings(result_df: pd.DataFrame, 
                                  col_mapping: Dict[str, str],
                                  config,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    Fully vectorized building processing - replaces the iterrows loop.
    
    Parameters:
    -----------
    result_df : pd.DataFrame
        Input dataframe with building data
    col_mapping : dict
        Column name mappings
    config : object
        Configuration object with probability distributions
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : Enhanced dataframe with all sampled characteristics
    """
    # Create vectorized analyzer
    analyzer = PreProcessRetrofit(config)
    
    # Extract required columns
    age_bands = result_df[col_mapping['age_band']]
    
    # Step 1: Determine wall types (vectorized)
    logger.debug("Determining wall types...")
    result_df['inferred_wall_type'] = analyzer.determine_wall_types_vectorized(age_bands, random_seed)
    
    # Step 2: Check existing wall insulation (vectorized)
    logger.debug("Checking existing wall insulation...")
    result_df['wall_insulated'] = analyzer.check_walls_already_insulated_vectorized(
        age_bands, 
        result_df['inferred_wall_type'],
        random_seed
    )
    
    # Step 3: Sample other existing interventions (vectorized)
    logger.debug("Sampling existing interventions...")
    existing_interventions_df = analyzer.sample_existing_interventions_vectorized(
        len(result_df),
        random_seed
    )
    
    # Add to main dataframe with prefixes for clarity
    for col in existing_interventions_df.columns:
        result_df[f'existing_{col}'] = existing_interventions_df[col].values
    
    logger.debug(f"Processed {len(result_df)} buildings in vectorized manner")
    
    return result_df


# def load_conservation_shapefile(path):
    
#     # path = '/Users/gracecolverd/Downloads/Conservation_Areas_-5503574965118299320/Conservation_Areas.shp'

#     cons = gpd.read_file(path)
#     return cons 