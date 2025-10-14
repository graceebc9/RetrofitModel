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
    flat_count: Optional[int] = None
    
    
    @property
    def external_wall_area_estimate(self) -> float:
        """Estimate external wall area from circumference and floor count."""
        return self.footprint_circumference * self.floor_count * 2.7
    
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

