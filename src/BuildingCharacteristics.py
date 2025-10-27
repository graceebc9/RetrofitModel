import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
 


@dataclass
class BuildingCharacteristics:

    """Building physical characteristics for cost calculation."""
    floor_count: int
    gross_external_area: float  # sq m
    gross_internal_area: float  # sq m
    footprint_circumference: float  # m
    building_footprint_area: float  # sq m
    avg_gas_percentile: int
    typology: str  #
    flat_count: Optional[int] = None
    
       
    OPENING_FACTORS = {
        'flat': 0.70,  # More windows in flats
        'house': 0.75,  # Standard houses
        
    }
    
        
    EXTERNAL_WALL_FACTORS = {
        # Detached properties (4 external walls)
        'Standard size detached': 1.0,
        'Large detached': 1.0,
        
        
        # Semi-detached (3 external walls - one shared)
        'Standard size semi detached': 0.75,
        'Large semi detached': 0.75,
        'Semi type house in multiples': 0.75,
        
        # Terraced (2 external walls - two shared)
        'Small low terraces': 0.5,
        'Tall terraces 3-4 storeys': 0.5,
        '2 storeys terraces with t rear extension': 0.5,
        
        # Flats (typically 1-2 external walls, varies by position)
        '3-4 storey and smaller flats': 0.45,  # Average for flats in building
        'Medium height flats 5-6 storeys': 0.40,  # Less external exposure in taller buildings
        
        # Mixed/Complex types
        'Linked and step linked premises': 0.65,  # Between semi and terrace
        'Planned balanced mixed estates': 0.70,  # Conservative estimate for mixed
        
        # Default for missing/unknown
        'unknown': 0.70,  # Conservative middle ground
    }
    
    @property
    def external_wall_area_estimate(self) -> float:
            """Estimate external wall area accounting for typology and openings."""
            
            # Get external wall factor
            factor = self.EXTERNAL_WALL_FACTORS.get(
                self.typology, 
                0.70  # Default conservative estimate
            )
           
            # Determine opening factor based on building type
            if 'flat' in self.typology.lower():
                opening_factor = self.OPENING_FACTORS['flat']
            else:
                opening_factor = self.OPENING_FACTORS['house']
            
            # Calculate
            gross_wall_area = self.footprint_circumference * self.floor_count * 2.7
            return gross_wall_area * factor * opening_factor
    
    @property
    def roof_area_estimate(self) -> float:
        """Estimate roof area from footprint.
     
        """
       
        return np.where( self.building_footprint_area< 30, self.building_footprint_area, 30  ) 
    
    @property
    def solar_roof_area_estimate(self, roof_scaling) -> float:
        """Estimate roof area from footprint.
        input is the footprint area
        """
        scaled_premise_area = self.building_footprint_area * roof_scaling
        return np.where( scaled_premise_area< 30, scaled_premise_area, 30  ) 

