from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np 
from scipy import stats


@dataclass
class RetrofitConfig:
    """
    Generates retrofit scenarios with age-appropriate wall insulation types.
    Includes Monte Carlo uncertainty analysis based on observed performance ranges.
    """
    energy_cost_per_kwh: float = 0.07  
    
    # 1. ALEATORY UNCERTAINTY (Inner Loop)
    n_samples: int = 100
    
    # Existing intervention probabilities (what % already have these installed) - these are defaults but can be overridden 
    existing_intervention_probs: Dict[str, float] = field(default_factory=lambda: {
        'loft_insulation': 0.67,
        'floor_insulation': 0.10,
        'window_upgrades': 0.10,
        'external_wall_occurence': 0.5,
        'roof_scaling_factor': 0.8
    })

    # Existing insulation probabilities by age band (configurable)
    insulation_probs_by_age_band: Dict[str, float] = field(default_factory=lambda: {
        'Pre 1919': 0.10,
        '1919-1944': 0.10,
        '1945-1959': {'cavity_wall_insulation': 0.50, 'internal_wall_insulation': 0.10},
        '1960-1979': 0.50,
        '1980-1989': 0.70,
        '1990-1999': 0.90,
        'Post 1999': 0.95
    })

    # Wall insulation recommendations by building age
    insulation_by_age_band: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'Pre 1919': {
            'primary_wall_type': 'solid_wall',
            'insulation_intervention': 'internal_wall_insulation',
            'reasoning': 'Pre-1919 buildings are predominantly solid brick/stone construction',
            'alternative': None,
            'confidence': 'high',
            'mixed_scenarios': False
        },
        '1919-1944': {
            'primary_wall_type': 'solid_wall',
            'insulation_intervention': 'internal_wall_insulation', 
            'reasoning': 'Inter-war period mostly solid wall, cavity walls rare until late 1920s',
            'alternative': 'cavity_wall_insulation',
            'confidence': 'high',
            'mixed_scenarios': False
        },
        '1945-1959': {
            'primary_wall_type': 'mixed',
            'insulation_intervention': 'cavity_wall_insulation',
            'reasoning': 'Transition period - cavity walls becoming standard',
            'alternative': 'internal_wall_insulation',
            'confidence': 'medium',
            'mixed_scenarios': True
        },
        '1960-1979': {
            'primary_wall_type': 'cavity_wall',
            'insulation_intervention': 'cavity_wall_insulation',
            'reasoning': 'Cavity walls standard, typically uninsulated',
            'alternative': None,
            'confidence': 'high',
            'mixed_scenarios': False
        },
        '1980-1989': {
            'primary_wall_type': 'cavity_wall',
            'insulation_intervention': 'cavity_wall_insulation',
            'reasoning': 'All cavity wall construction',
            'alternative': None,
            'confidence': 'high',
            'mixed_scenarios': False
        },
        '1990-1999': {
            'primary_wall_type': 'cavity_wall',
            'insulation_intervention': 'cavity_wall_insulation',
            'reasoning': 'Cavity walls with some insulation',
            'alternative': None,
            'confidence': 'high',
            'mixed_scenarios': False
        },
        'Post 1999': {
            'primary_wall_type': 'cavity_wall',
            'insulation_intervention': 'cavity_wall_insulation',
            'reasoning': 'Modern cavity walls',
            'alternative': None,
            'confidence': 'medium',
            'mixed_scenarios': False
        }
    })
    
    # Mixed age probabilities for 1945-1959 period
    mixed_age_probabilities: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        '1945-1959': {
            'cavity_wall_insulation': 0.7,
            'internal_wall_insulation': 0.3
        }
    })