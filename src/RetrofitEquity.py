import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
 
from typing import Dict 


 
EQUITY_WEIGHTS = {
'High Deprivation': 0.4,
'Medium Deprivation': 0.7,
'Medium Deprivation': 1,
} 

def calculate_social_equity_score(selected_df: pd.DataFrame) -> Dict:
    """
    Calculate rigorous social equity metrics for selected projects
    
    Args:
        selected_df: DataFrame with selected projects including 'meta_socio_persona' column
    
    Returns:
        Dictionary with equity metrics
    """
    if len(selected_df) == 0:
        return {
            'vulnerable_investment_pct': 0,
            'equity_concentration': 0,
            'persona_breakdown': {},
            'vulnerable_count': 0,
            'total_count': 0
        }
    
    persona_counts = selected_df['meta_socio_persona'].value_counts()
    total = len(selected_df)
    
    # Calculate % investment in vulnerable groups (deprived + struggling)
    vulnerable_count = persona_counts.get('deprived', 0) + persona_counts.get('struggling', 0)
    vulnerable_pct = (vulnerable_count / total * 100) if total > 0 else 0
    
    # Calculate concentration index (Herfindahl index: 0 = perfect equality, 1 = concentrated)
    proportions = persona_counts / total
    concentration = (proportions ** 2).sum()
    
    # Create persona breakdown
    persona_breakdown = {}
    for persona, count in persona_counts.items():
        count = persona_counts.get(persona, 0)
        pct = (count / total * 100) if total > 0 else 0
        persona_breakdown[persona] = {'count': count, 'pct': pct}
    
    return {
        'vulnerable_investment_pct': vulnerable_pct,
        'equity_concentration': concentration,
        'persona_breakdown': persona_breakdown,
        'vulnerable_count': vulnerable_count,
        'total_count': total
    }


def calculate_scenario_persona_metrics(selected_df: pd.DataFrame, scenario: str) -> Dict:
    """
    Calculate equity metrics for a specific scenario
    
    Args:
        selected_df: DataFrame with selected projects
        scenario: Scenario name
    
    Returns:
        Dictionary with scenario-specific equity metrics
    """
    scenario_df = selected_df[selected_df['scenario'] == scenario]
    
    if len(scenario_df) == 0:
        return None
    
    equity_metrics = calculate_social_equity_score(scenario_df)
    
    # Add cost and CO2 breakdowns by persona
    persona_stats = scenario_df.groupby('meta_socio_persona').agg({
        'cost_of_intervention_mean': 'sum',
        'total_ton_co2_saved': 'sum',
        'upn': 'count'
    }).rename(columns={'upn': 'n_projects'})
    
    return {
        'equity_metrics': equity_metrics,
        'persona_stats': persona_stats
    }

 