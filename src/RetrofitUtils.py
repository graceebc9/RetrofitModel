import logging 

def calculate_estimated_flats_per_building(building_footprint_area, typology_col, floor_count):
    """Calculate estimated number of flats based on building characteristics."""
    house_typologies = [
        'Small low terraces', 'Tall terraces 3-4 storeys', 'Large semi detached',
        'Standard size detached', 'Standard size semi detached',
        '2 storeys terraces with t rear extension', 'Semi type house in multiples',
        'Large detached', 'Very large detached', 'Linked and step linked premises',
        'Domestic outbuilding',
    ]
    
    if typology_col in house_typologies or typology_col == 'all_unknown_typology':
        return 1
    
    typical_flat_footprints = {
        'Medium height flats 5-6 storeys': 50,
        '3-4 storey and smaller flats': 60,
        'Tall flats 6-15 storeys': 45,
        'Very tall point block flats': 40,
        'Planned balanced mixed estates': 65,
    }
    
    efficiency_factors = {
        'Medium height flats 5-6 storeys': 0.75,
        '3-4 storey and smaller flats': 0.80,
        'Tall flats 6-15 storeys': 0.70,
        'Very tall point block flats': 0.65,
        'Planned balanced mixed estates': 0.80,
    }
    
    flat_footprint = typical_flat_footprints.get(typology_col, 50)
    efficiency = efficiency_factors.get(typology_col, 0.75)
    
    try:
        
        usable_area_per_floor = building_footprint_area * efficiency
        flats_per_floor = usable_area_per_floor / flat_footprint
        total_flats = float(floor_count) * float(flats_per_floor)
        return max(1, round(total_flats))
    except (TypeError, ZeroDivisionError, ValueError) as e:
        # E: Replaced magic number -999 with 1 and logged the error
        logging.error(f"Error calculating flats for typology {typology_col}: {e}. Defaulting to 1.")
        return 1


