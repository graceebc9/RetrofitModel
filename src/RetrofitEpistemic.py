"""
Module: RetrofitEpistemic.py

Updated epistemic sampling with:
1. Truncated normal for decile_misclassification_bias
2. Optional fixed_factors parameter for sensitivity testing
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, uniform, truncnorm
from pyDOE2 import lhs
from typing import Dict, Any, Optional


# Central/default values for each factor (used in sensitivity testing)
FACTOR_DEFAULTS = {
    'time_scale_bias': 1.0,
    'decile_misclassification_bias': 0.0,
    'solid_wall_internal_improvement_factor': 0.10,
    'solid_wall_external_improvement_factor': 0.20,
    'regional_multipliers_uncertainty': 1.0,
    'age_band_multipliers_uncertainty': 1.0,
    'cost_scenario': 'central',
    'external_wall_probability': 0.5,
    'flat_fp_mean': 55,
    'flat_fp_std': 8,
    'flat_eff_mean': 0.75,
    'flat_eff_std': 0.05,
    'area_based_choice': 'mode',
}


def generate_epistemic_scenarios_lhs(
    N_epistemic_runs: int,
    random_seed: int, 
    fixed_factors: Optional[Dict[str, Any]] = None, 
    
) -> pd.DataFrame:
    """
    Generates N_epistemic_runs scenarios for the Outer Loop using Latin Hypercube Sampling.
    
    Parameters:
    -----------
    N_epistemic_runs : int
        Number of epistemic scenarios to generate
    fixed_factors : dict, optional
        Dictionary of factors to fix at specific values.
        Used for sensitivity testing to isolate individual factor contributions.
        e.g., {'decile_misclassification_bias': 0.0} fixes that factor while others vary.
    
    Returns:
    --------
    pd.DataFrame : DataFrame with N_epistemic_runs rows, one column per factor
    """
    
    N_factors = 13
    
    # Generate the Latin Hypercube Samples (N_epistemic_runs rows, N_factors columns)
    lhs_samples_uniform = lhs(N_factors, samples=N_epistemic_runs, criterion='m', iterations=100,  random_state=random_seed,)
    
    # === Inverse Transform Sampling ===
    
    # Factor 1: Time Scale Bias (beta_TS) - Truncated Normal: loc=1.0, scale=0.05, bounds [0.9, 1.1]
    a_ts, b_ts = (0.9 - 1.0) / 0.05, (1.1 - 1.0) / 0.05 
    ts_samples = truncnorm.ppf(lhs_samples_uniform[:, 0], a=a_ts, b=b_ts, loc=1.0, scale=0.05)

    # Factor 2: Decile Misclassification Bias (beta_DEC) - UPDATED: Truncated Normal
    # loc=0.0, scale=0.02, bounds [-0.05, 0.05] (±2.5σ)
    # Rationale: Postcode-based decile assignment has grouping error, capped at ~0.5 decile effect
    a_dec, b_dec = (-0.05 - 0.0) / 0.02, (0.05 - 0.0) / 0.02
    decile_samples = truncnorm.ppf(lhs_samples_uniform[:, 1], a=a_dec, b=b_dec, loc=0.0, scale=0.02)

    # Factor 3: Solid Wall Internal Improvement (beta_SWI) - Truncated Normal: loc=0.1, scale=0.01, bounds [0.08, 0.12]
    a_swi, b_swi = (0.08 - 0.1) / 0.01, (0.12 - 0.1) / 0.01
    swi_samples = truncnorm.ppf(lhs_samples_uniform[:, 2], a=a_swi, b=b_swi, loc=0.1, scale=0.01)
    
    # Factor 4: Solid Wall External Improvement (beta_SWE) - Truncated Normal: loc=0.2, scale=0.02, bounds [0.15, 0.25]
    a_swe, b_swe = (0.15 - 0.2) / 0.02, (0.25 - 0.2) / 0.02
    swe_samples = truncnorm.ppf(lhs_samples_uniform[:, 3], a=a_swe, b=b_swe, loc=0.2, scale=0.02)

    # Factor 5: Regional Cost Multipliers (beta_REG) - Uniform: Range [0.9, 1.1]
    reg_samples = uniform.ppf(lhs_samples_uniform[:, 4], loc=0.9, scale=0.2) 

    # Factor 6: Age Band Cost Multipliers (beta_AGE) - Uniform: Range [0.92, 1.08]
    age_samples = uniform.ppf(lhs_samples_uniform[:, 5], loc=0.92, scale=0.16)
    
    # Factor 7: Discrete Cost Scenario
    scenario_choices = np.array(['optimistic', 'central', 'pessimistic'])
    cost_scenario_samples_uniform = lhs_samples_uniform[:, 6]
    indices = np.floor(cost_scenario_samples_uniform * 3).astype(int)
    indices = np.clip(indices, 0, 2)
    cost_scenario_samples = scenario_choices[indices]

    # Factor 8: External Wall Occurrence (beta_EWO) - Uniform: Range [0.1, 0.9]
    ewo_samples = uniform.ppf(lhs_samples_uniform[:, 7], loc=0.1, scale=0.8)

    # Factor 9: Mean Flat Footprint (fp_mean) - Truncated Normal: loc=55, scale=5, bounds [40, 70]
    a_fp_m, b_fp_m = (40 - 55) / 5, (70 - 55) / 5
    fp_mean_samples = truncnorm.ppf(lhs_samples_uniform[:, 8], a=a_fp_m, b=b_fp_m, loc=55, scale=5)

    # Factor 10: StdDev of Flat Footprint (fp_std) - Truncated Normal: loc=8, scale=2, bounds [2, 15]
    a_fp_s, b_fp_s = (2 - 8) / 2, (15 - 8) / 2
    fp_std_samples = truncnorm.ppf(lhs_samples_uniform[:, 9], a=a_fp_s, b=b_fp_s, loc=8, scale=2)
    
    # Factor 11: Mean Efficiency (eff_mean) - Truncated Normal: loc=0.75, scale=0.03, bounds [0.5, 0.8]
    a_ef_m, b_ef_m = (0.5 - 0.75) / 0.03, (0.8 - 0.75) / 0.03
    eff_mean_samples = truncnorm.ppf(lhs_samples_uniform[:, 10], a=a_ef_m, b=b_ef_m, loc=0.75, scale=0.03)

    # Factor 12: StdDev of Efficiency (eff_std) - Truncated Normal: loc=0.05, scale=0.02, bounds [0.01, 0.1]
    a_ef_s, b_ef_s = (0.01 - 0.05) / 0.02, (0.1 - 0.05) / 0.02
    eff_std_samples = truncnorm.ppf(lhs_samples_uniform[:, 11], a=a_ef_s, b=b_ef_s, loc=0.05, scale=0.02)

    # Factor 13: Area-Based Choice (Discrete)
    area_choices = np.array(['min', 'max', 'mode'])
    area_choice_samples_uniform = lhs_samples_uniform[:, 12]
    area_indices = np.floor(area_choice_samples_uniform * 3).astype(int)
    area_indices = np.clip(area_indices, 0, 2)
    area_choice_samples = area_choices[area_indices]
    
    # Compile into DataFrame
    epistemic_df = pd.DataFrame({
        'time_scale_bias': ts_samples,
        'decile_misclassification_bias': decile_samples,
        'solid_wall_internal_improvement_factor': swi_samples,
        'solid_wall_external_improvement_factor': swe_samples,
        'regional_multipliers_uncertainty': reg_samples,
        'age_band_multipliers_uncertainty': age_samples,
        'cost_scenario': cost_scenario_samples,
        'external_wall_probability': ewo_samples,
        'flat_fp_mean': fp_mean_samples,
        'flat_fp_std': fp_std_samples,
        'flat_eff_mean': eff_mean_samples,
        'flat_eff_std': eff_std_samples,
        'area_based_choice': area_choice_samples,
    })
    
    # === Apply fixed factors (for sensitivity testing) ===
    if fixed_factors:
        for factor, value in fixed_factors.items():
            if factor in epistemic_df.columns:
                epistemic_df[factor] = value
            else:
                raise ValueError(f"Unknown factor: {factor}. Valid factors: {list(epistemic_df.columns)}")
    
    return epistemic_df