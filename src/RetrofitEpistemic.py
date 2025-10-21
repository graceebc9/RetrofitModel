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
    N_factors = 6
    
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
    
    # 4. Compile into DataFrame
    epistemic_df = pd.DataFrame({
        'time_scale_bias': ts_samples,
        'decile_misclassification_bias': decile_samples,
        'solid_wall_internal_improvement_factor': swi_samples,
        'solid_wall_external_improvement_factor': swe_samples,
        'regional_multipliers_uncertainty': reg_samples,
        'age_band_multipliers_uncertainty': age_samples,
    })
    
    return epistemic_df

# import pandas as pd
# import numpy as np
# from scipy.stats import truncnorm, uniform , norm 
# from typing import Dict

# # --- Epistemic Factor Sampler ---

# def generate_epistemic_scenarios(N_epistemic_runs: int) -> pd.DataFrame:
#     """
#     Generates N_epistemic_runs scenarios for the Outer Loop of the 2DMC simulation.
    
#     Factors included:
#     1. Time Scale Bias (beta_TS): Systematic error in savings comparison.
#     2. Decile Misclassification Bias (beta_DEC): Systematic error in initial classification.
#     3. Solid Wall Improvement Factor (Internal) (beta_SWI): Uncertainty in assumed factor (nominal 0.1).
#     4. Solid Wall Improvement Factor (External) (beta_SWE): Uncertainty in assumed factor (nominal 0.2).
#     5. Regional Multipliers Uncertainty (beta_REG): Systematic uncertainty in regional cost factors.
#     6. Age Band Multipliers Uncertainty (beta_AGE): Systematic uncertainty in age-based cost factors.
#     """
    
#     # 1. Define Distribution Parameters (These should be set by expert judgment)
    
#     # Time Scale Bias (beta_TS): Systematic uncertainty around the 1.0 baseline.
#     # Assumes a 5% systematic uncertainty.
#     ts_samples = truncnorm.rvs(a=-0.1, b=0.1, loc=1.0, scale=0.05, size=N_epistemic_runs)

#     # Decile Misclassification Bias (beta_DEC): Systematic uncertainty around a 0.0 shift.
#     # Represents an overall systematic misclassification error of +/- 2%
#     decile_samples = norm.rvs(loc=0.0, scale=0.02, size=N_epistemic_runs)

#     # Solid Wall Internal Improvement (beta_SWI): Uncertainty in the fixed factor (0.1).
#     # Assumes the true factor could range from 0.08 to 0.12 (standard deviation of 0.01).
#     swi_samples = truncnorm.rvs(a=-0.02, b=0.02, loc=0.1, scale=0.01, size=N_epistemic_runs)
    
#     # Solid Wall External Improvement (beta_SWE): Uncertainty in the fixed factor (0.2).
#     # Assumes the true factor could range from 0.15 to 0.25 (standard deviation of 0.02).
#     swe_samples = truncnorm.rvs(a=-0.05, b=0.05, loc=0.2, scale=0.02, size=N_epistemic_runs)

#     # Regional Cost Multipliers (beta_REG): Systematic error in regional cost factors (nominal 1.0).
#     # Assumes the regional multipliers could be systematically +/- 10% off.
#     reg_samples = uniform.rvs(loc=0.9, scale=0.2, size=N_epistemic_runs) # Range: 0.9 to 1.1

#     # Age Band Cost Multipliers (beta_AGE): Systematic error in age band factors (nominal 1.0).
#     # Assumes the age band multipliers could be systematically +/- 8% off.
#     age_samples = uniform.rvs(loc=0.92, scale=0.16, size=N_epistemic_runs) # Range: 0.92 to 1.08
    
#     # 2. Compile into DataFrame
#     epistemic_df = pd.DataFrame({
#         'time_scale_bias': ts_samples,
#         'decile_misclassification_bias': decile_samples,
#         'solid_wall_internal_improvement_factor': swi_samples,
#         'solid_wall_external_improvement_factor': swe_samples,
#         'regional_multipliers_uncertainty': reg_samples,
#         'age_band_multipliers_uncertainty': age_samples,
#     })
    
#     return epistemic_df

# # Example of how to use it:
# # scenarios = generate_epistemic_scenarios(N_epistemic_runs=50)