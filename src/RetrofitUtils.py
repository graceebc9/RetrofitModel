import logging 


import logging
import random # Needed for aleatoric sampling

# Add this import at the top of your file
import random 


def calc_est_flats_building(
    building_footprint_area: float, 
    typology_col: str, 
    floor_count: float, 
    fp_mean: float, 
    fp_std: float, 
    eff_mean: float, 
    eff_std: float
) -> int:
    """
    Runs ONE   sample for the number of flats in a building.
    
    This is the INNER LOOP of the 2D Monte Carlo. It takes the epistemic
    parameters (the means and std devs) as arguments.
    """

    
    # --- 1. Handle House Typologies (Deterministic) ---
    house_typologies = [
        'Small low terraces', 'Tall terraces 3-4 storeys', 'Large semi detached',
        'Standard size detached', 'Standard size semi detached', 'Planned balanced mixed estates', 
        '2 storeys terraces with t rear extension', 'Semi type house in multiples',
        'Large detached', 'Very large detached', 'Linked and step linked premises',
        'Domestic outbuilding',
        
    ]
    
    if typology_col in house_typologies or typology_col == 'all_unknown_typology' or typology_col is  None:
        return 1
 
    # --- 2. Run Aleatoric Sample (for Flats) ---
 
    # --- 2. Validate Inputs Before Sampling ---
    if any(param is None for param in [fp_mean, fp_std, eff_mean, eff_std]):
        logging.warning(
            f"Missing epistemic parameters for typology '{typology_col}'. "
            f"fp_mean={fp_mean}, fp_std={fp_std}, eff_mean={eff_mean}, eff_std={eff_std}. "
            f"Defaulting to 1 flat."
        )
        raise Exception (f'Epistemic scenario missing params Missing epistemic parameters for typology {typology_col} ' )

    try:

        # check inputs are present: 
        
        # --- Aleatoric Sampling ---
        # Sample from the distributions defined by the epistemic parameters
        
        # We must clip the samples to avoid nonsensical values
        # e.g., negative footprint or efficiency > 1.0
        
        sampled_footprint = max(20.0, random.normalvariate(fp_mean, fp_std))
        sampled_efficiency = max(0.50, min(0.95, random.normalvariate(eff_mean, eff_std)))
        
        # --- Calculation ---
        usable_area_per_floor = float(building_footprint_area) * sampled_efficiency
        flats_per_floor = usable_area_per_floor / sampled_footprint
        total_flats = float(floor_count) * flats_per_floor
        
        return max(1, round(total_flats))
        
    except (TypeError, ZeroDivisionError, ValueError) as e:
        print(f'building_footprint_area: {building_footprint_area}, floor_count: {floor_count}, typology_col: {typology_col}, sampled_footprint: {sampled_footprint}, sampled_efficiency: {sampled_efficiency} ' )
        # Log the error and return a default
        logging.error(f"Error in aleatoric sample for typology {typology_col}: {e}. Defaulting to 1.")
        return 1


# def calculate_estimated_flats_per_building(building_footprint_area, typology_col, floor_count):
#     """Calculate estimated number of flats based on building characteristics."""
#     house_typologies = [
#         'Small low terraces', 'Tall terraces 3-4 storeys', 'Large semi detached',
#         'Standard size detached', 'Standard size semi detached',
#         '2 storeys terraces with t rear extension', 'Semi type house in multiples',
#         'Large detached', 'Very large detached', 'Linked and step linked premises',
#         'Domestic outbuilding',
#     ]
    
#     if typology_col in house_typologies or typology_col == 'all_unknown_typology':
#         return 1
    
#     typical_flat_footprints = {
#         'Medium height flats 5-6 storeys': 50,
#         '3-4 storey and smaller flats': 60,

#         'Tall flats 6-15 storeys': 45,
#         'Very tall point block flats': 40,
#         'Planned balanced mixed estates': 65,
#     }
    
#     efficiency_factors = {
#         'Medium height flats 5-6 storeys': 0.75,
#         '3-4 storey and smaller flats': 0.80,
#         'Tall flats 6-15 storeys': 0.70,
#         'Very tall point block flats': 0.65,
#         'Planned balanced mixed estates': 0.80,
#     }
    
#     flat_footprint = typical_flat_footprints.get(typology_col, 50)
#     efficiency = efficiency_factors.get(typology_col, 0.75)
    
#     try:
        
#         usable_area_per_floor = building_footprint_area * efficiency
#         flats_per_floor = usable_area_per_floor / flat_footprint
#         total_flats = float(floor_count) * float(flats_per_floor)
#         return max(1, round(total_flats))
#     except (TypeError, ZeroDivisionError, ValueError) as e:
#         # E: Replaced magic number -999 with 1 and logged the error
#         logging.error(f"Error calculating flats for typology {typology_col}: {e}. Defaulting to 1.")
#         return 1


