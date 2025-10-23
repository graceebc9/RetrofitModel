import pandas as pd
import numpy as np
from scipy.stats import norm, uniform, truncnorm
from pyDOE2 import lhs # Assuming pyDOE2 is installed: pip install pyDOE2

def generate_epistemic_scenarios_lhs(N_epistemic_runs: int) -> pd.DataFrame:
    """
    Generates N_epistemic_runs scenarios for the Outer Loop using Latin Hypercube Sampling.
    
    The sampling space is 6-dimensional (one for each factor).
    """
    
    # 1. Define the Number of Factors (N=6)
    N_factors = 7
    
    # 2. Generate the Latin Hypercube Samples (N_epistemic_runs rows, N_factors columns)
    # The output is uniformly distributed between 0 and 1.
    lhs_samples_uniform = lhs(N_factors, samples=N_epistemic_runs, criterion='m', iterations=100)
    
    # 3. Inverse Transform Sampling (Map uniform LHS to desired distribution)
    
    # Factor 1: Time Scale Bias (beta_TS) - Truncated Normal: loc=1.0, scale=0.05, bounds [0.9, 1.1]
    a_ts, b_ts = (0.9 - 1.0) / 0.05, (1.1 - 1.0) / 0.05 
    ts_samples = truncnorm.ppf(lhs_samples_uniform[:, 0], a=a_ts, b=b_ts, loc=1.0, scale=0.05)

    # Factor 2: Decile Misclassification Bias (beta_DEC) - Normal: loc=0.0, scale=0.02
    decile_samples = norm.ppf(lhs_samples_uniform[:, 1], loc=0.0, scale=0.02)

    # Factor 3: Solid Wall Internal Improvement (beta_SWI) - Truncated Normal: loc=0.1, scale=0.01, bounds [0.08, 0.12]
    a_swi, b_swi = (0.08 - 0.1) / 0.01, (0.12 - 0.1) / 0.01
    swi_samples = truncnorm.ppf(lhs_samples_uniform[:, 2], a=a_swi, b=b_swi, loc=0.1, scale=0.01)
    
    # Factor 4: Solid Wall External Improvement (beta_SWE) - Truncated Normal: loc=0.2, scale=0.02, bounds [0.15, 0.25]
    a_swe, b_swe = (0.15 - 0.2) / 0.02, (0.25 - 0.2) / 0.02
    swe_samples = truncnorm.ppf(lhs_samples_uniform[:, 3], a=a_swe, b=b_swe, loc=0.2, scale=0.02)

    # Factor 5: Regional Cost Multipliers (beta_REG) - Uniform: Range [0.9, 1.1]
    # uniform.ppf(q, loc, scale) where loc is start and scale is range
    reg_samples = uniform.ppf(lhs_samples_uniform[:, 4], loc=0.9, scale=0.2) 

    # Factor 6: Age Band Cost Multipliers (beta_AGE) - Uniform: Range [0.92, 1.08]
    age_samples = uniform.ppf(lhs_samples_uniform[:, 5], loc=0.92, scale=0.16)
    
    # --- NEW: Factor 7 - Discrete Cost Scenario ---
    
    # We map the [0, 1] sample to 3 bins:
    # [0.0, 0.333...) -> 'optimistic'
    # [0.333..., 0.666...) -> 'central'
    # [0.666..., 1.0] -> 'pessimistic'
    
    scenario_choices = np.array(['optimistic', 'central', 'pessimistic'])
    # Get the 7th column of samples (index 6)
    cost_scenario_samples_uniform = lhs_samples_uniform[:, 6]
    
    # Convert [0, 1] to indices [0, 1, 2]
    # We multiply by 3 (N_choices) and take the floor
    indices = np.floor(cost_scenario_samples_uniform * 3).astype(int)
    
    # Clip to handle the (very rare) edge case of a sample being exactly 1.0
    indices = np.clip(indices, 0, 2)
    
    # Select from the array
    cost_scenario_samples = scenario_choices[indices]

    # 4. Compile into DataFrame
    epistemic_df = pd.DataFrame({
        'time_scale_bias': ts_samples,
        'decile_misclassification_bias': decile_samples,
        'solid_wall_internal_improvement_factor': swi_samples,
        'solid_wall_external_improvement_factor': swe_samples,
        'regional_multipliers_uncertainty': reg_samples,
        'age_band_multipliers_uncertainty': age_samples,
        'cost_scenario': cost_scenario_samples,
    })
    
    return epistemic_df
