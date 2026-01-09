import pandas as pd
import numpy as np
from scipy.stats import norm, uniform, truncnorm
from pyDOE2 import lhs # Assuming pyDOE2 is installed: pip install pyDOE2

def generate_epistemic_scenarios_lhs(N_epistemic_runs: int) -> pd.DataFrame:
    """
    Generates N_epistemic_runs scenarios for the Outer Loop using Latin Hypercube Sampling.
    
    The sampling space is 6-dimensional (one for each factor).
    """
    
    N_factors = 13
    
    # 2. Generate the Latin Hypercube Samples (N_epistemic_runs rows, N_factors columns)
    # The output is uniformly distributed between 0 and 1.
    lhs_samples_uniform = lhs(N_factors, samples=N_epistemic_runs, criterion='m', iterations=100)
    
    # 3. Inverse Transform Sampling (Map uniform LHS to desired distribution)
    
    # Factor 1: Time Scale Bias (beta_TS) - Truncated Normal: loc=1.0, scale=0.05, bounds [0.9, 1.1]
    a_ts, b_ts = (0.9 - 1.0) / 0.05, (1.1 - 1.0) / 0.05 
    ts_samples = truncnorm.ppf(lhs_samples_uniform[:, 0], a=a_ts, b=b_ts, loc=1.0, scale=0.05)

    # Factor 2: Decile Misclassification Bias (beta_DEC) - Normal: loc=0.0, scale=0.02
    # decile_samples = norm.ppf(lhs_samples_uniform[:, 1], loc=0.0, scale=0.02)
    # Factor 2: Decile Misclassification Bias (beta_DEC) 
    # Truncated Normal: loc=0.0, scale=0.02, bounds [-0.05, 0.05]
    a_dec, b_dec = (-0.05 - 0.0) / 0.02, (0.05 - 0.0) / 0.02  # ±2.5σ
    decile_samples = truncnorm.ppf(lhs_samples_uniform[:, 1], a=a_dec, b=b_dec, loc=0.0, scale=0.02)

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

     # Factor 8: External Wall Occurrence (beta_EWO) - Uniform: Range [0.3, 0.7]
    # This represents the probability/proportion of external wall retrofits
    # Centered around 0.5 with ±0.2 uncertainty
    ewo_samples = uniform.ppf(lhs_samples_uniform[:, 7], loc=0.1, scale=0.8)

    # --- NEW: 4 Factors (Flat Estimation Model, Default Typology) ---

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

    # --- NEW: Factor 13 - Area-Based Choice (Discrete) ---
    area_choices = np.array(['min', 'max', 'mode'])
    # Get the 13th column of samples (index 12)
    area_choice_samples_uniform = lhs_samples_uniform[:, 12]
    # Convert [0, 1] to indices [0, 1, 2]
    area_indices = np.floor(area_choice_samples_uniform * 3).astype(int)
    # Clip to handle edge case of sample being exactly 1.0
    area_indices = np.clip(area_indices, 0, 2)
    # Select from the array
    area_choice_samples = area_choices[area_indices]
    
    # 4. Compile into DataFrame
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

        'area_based_choice': area_choice_samples,  # NEW
    })
    
    return epistemic_df
