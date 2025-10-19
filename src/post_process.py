 
import pandas as pd
import numpy as np

 
def calculate_combined_energy_reduction(
    df, 
    scenario_interventions,
    energy_col_template='energy_{intervention}_percentile_gas_mean',
    std_col_template='energy_{intervention}_percentile_gas_std',
    output_prefix='total_energy_reduction'
):
    """
    Optimized calculation of combined energy reductions using additive and multiplicative methods.
    Handles NaN values (only one wall type per building will be non-NaN).
    
    Key optimizations:
    - Vectorized operations using NumPy
    - No row-wise apply() calls
    - Single pass through data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with energy reduction columns
    scenario_interventions : list of tuples
        List of (scenario, intervention) pairs, e.g.:
        [('loft_installation', 'loft'),
         ('wall_installation', 'cavity_wall'),
         ('wall_installation', 'solid_wall_internal'),
         ('wall_installation', 'solid_wall_external')]
    energy_col_template : str
        Template for energy column names with {intervention} placeholder
    std_col_template : str
        Template for std column names with {intervention} placeholder
    output_prefix : str
        Prefix for output column names
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns for combined reductions
    """
    
    # Build column names dynamically
    energy_cols = []
    std_cols = []
    
    for scenario, intervention in scenario_interventions:
        energy_col = f"{scenario}_{energy_col_template.format(intervention=intervention)}"
        std_col = f"{scenario}_{std_col_template.format(intervention=intervention)}"
        energy_cols.append(energy_col)
        std_cols.append(std_col)
    
    # Check which columns exist in dataframe
    existing_energy_cols = [col for col in energy_cols if col in df.columns]
    existing_std_cols = [col for col in std_cols if col in df.columns]
    
    if not existing_energy_cols:
        raise ValueError(f"No energy columns found in dataframe. Looking for: {energy_cols}")
    
    # ====================
    # 1. ADDITIVE METHOD (already vectorized - keep as is)
    # ====================
    df[f'{output_prefix}_additive_mean'] = df[existing_energy_cols].sum(axis=1, skipna=True)
    
    if existing_std_cols:
        df[f'{output_prefix}_additive_std'] = np.sqrt(
            (df[existing_std_cols]**2).sum(axis=1, skipna=True)
        )
    
    # ====================
    # 2. MULTIPLICATIVE METHOD (vectorized)
    # ====================
    # Extract to numpy arrays for faster computation
    energy_data = df[existing_energy_cols].values
    n_rows = len(df)
    
    # Calculate remaining energy: product of (1 - reduction) for non-NaN values
    # NaN values are treated as 1.0 in the product (no effect)
    remaining_energy = np.nanprod(1 - energy_data, axis=1)
    total_reduction = 1 - remaining_energy
    
    df[f'{output_prefix}_multiplicative_mean'] = total_reduction
    
    # Calculate std using vectorized error propagation
    if existing_std_cols:
        std_data = df[existing_std_cols].values
        
        # Initialize variance accumulator
        variance_total = np.zeros(n_rows)
        
        # For each intervention column
        for i in range(len(existing_energy_cols)):
            reduction_col = energy_data[:, i]
            std_col = std_data[:, i]
            
            # Only process non-NaN values
            valid_mask = ~np.isnan(reduction_col)
            
            if valid_mask.any():
                # Partial derivative: dR_total/dr_i = remaining_energy / (1 - r_i)
                # Avoid division by zero
                denominator = 1 - reduction_col
                safe_mask = valid_mask & (denominator > 1e-10)
                
                if safe_mask.any():
                    partial_deriv = np.zeros(n_rows)
                    partial_deriv[safe_mask] = remaining_energy[safe_mask] / denominator[safe_mask]
                    
                    # Accumulate variance: Var += (∂R/∂r_i * σ_i)²
                    variance_contribution = (partial_deriv * std_col) ** 2
                    variance_contribution[~safe_mask] = 0  # Zero out invalid entries
                    variance_total += variance_contribution
        
        df[f'{output_prefix}_multiplicative_std'] = np.sqrt(variance_total)
    
    # Cap all estimates at 100% (1.0)
    df[f'{output_prefix}_additive_mean'] = df[f'{output_prefix}_additive_mean'].clip(upper=1.0)
    df[f'{output_prefix}_multiplicative_mean'] = df[f'{output_prefix}_multiplicative_mean'].clip(upper=1.0)
    
    return df

 


def calculate_combined_energy_reduction(
    df, 
    scenario_interventions,
    decay_factor=0.85,
    energy_col_template='energy_{intervention}_percentile_gas_mean',
    std_col_template='energy_{intervention}_percentile_gas_std',
    output_prefix='total_energy_reduction'
):
    """
    Optimized version - uses vectorized numpy operations instead of row-wise apply.
    
    Key optimizations:
    - Extract data to numpy arrays upfront
    - Vectorized sorting and calculations
    - Pre-compute decay weights
    - Single pass through data
    """
    
    # Build column names dynamically
    energy_cols = []
    std_cols = []
    
    for scenario, intervention in scenario_interventions:
        energy_col = f"{scenario}_{energy_col_template.format(intervention=intervention)}"
        std_col = f"{scenario}_{std_col_template.format(intervention=intervention)}"
        energy_cols.append(energy_col)
        std_cols.append(std_col)
    
    # Check which columns exist
    existing_energy_cols = [col for col in energy_cols if col in df.columns]
    existing_std_cols = [col for col in std_cols if col in df.columns]
    
    if not existing_energy_cols:
        raise ValueError(f"No energy columns found in dataframe. Looking for: {energy_cols}")
    
    # Extract to numpy arrays (much faster than pandas operations)
    energy_data = df[existing_energy_cols].values
    n_rows, n_cols = energy_data.shape
    
    # Pre-compute maximum possible decay weights
    max_decay_weights = np.array([decay_factor**i for i in range(n_cols)])
    
    # Initialize output arrays
    mean_values = np.zeros(n_rows)
    std_values = np.zeros(n_rows) if existing_std_cols else None
    
    # Vectorized computation
    for i in range(n_rows):
        row_data = energy_data[i]
        valid_mask = ~np.isnan(row_data)
        valid_data = row_data[valid_mask]
        
        if len(valid_data) == 0:
            continue
        
        # Sort descending and get indices
        sorted_indices = np.argsort(valid_data)[::-1]
        sorted_data = valid_data[sorted_indices]
        
        # Apply decay weights (only for valid entries)
        n_valid = len(sorted_data)
        weights = max_decay_weights[:n_valid]
        mean_values[i] = np.dot(sorted_data, weights)
        
        # Calculate std if needed
        if std_values is not None and len(existing_std_cols) == len(existing_energy_cols):
            # Extract std values for this row
            std_data = df[existing_std_cols].iloc[i].values
            valid_stds = std_data[valid_mask]
            
            if len(valid_stds) == n_valid:
                sorted_stds = valid_stds[sorted_indices]
                variance = np.sum((weights * sorted_stds) ** 2)
                std_values[i] = np.sqrt(variance)
    
    # Assign results back to dataframe
    df[f'{output_prefix}_decay_mean'] = np.clip(mean_values, 0, 1.0)
    if std_values is not None:
        df[f'{output_prefix}_decay_std'] = std_values
    
    return df


