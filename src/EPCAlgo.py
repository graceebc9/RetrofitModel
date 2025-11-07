import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
import pandas as pd

import pandas as pd
import logging
from typing import Tuple


allowed_personas = ['lower middle', 'struggling',  'deprived']

def select_epc_algo(df_knapsack: pd.DataFrame, 
                         budget: float, 
                         cost_column: str = 'cost_of_intervention_mean', 
                         efficiency_column: str = 'cost_per_net_ton_co2_kg',
                         logger: logging.Logger = None
                         ) -> Tuple[pd.DataFrame, float]:
    print('Starting EPC ALgo ')
    epc_col = 'CURRENT_ENERGY_RATING'   
    persona_col = 'meta_socio_persona'
    uprn_col = 'upn'  
    list_personas = allowed_personas  
    
    # --- ROBUST FILTERING ---
    # Standardize EPC to ensure matching works (uppercase, stripped)
    df_knapsack[epc_col] = df_knapsack[epc_col].astype(str).str.upper().str.strip()
    
    target_epcs = ['D', 'E', 'F', 'G']
    
    # Filter 1: EPC
    df_epc_filtered = df_knapsack[df_knapsack[epc_col].isin(target_epcs)]
    if df_epc_filtered.empty:
        print('None in epc range')
        raise Exception('None in epc range')
    
    # Filter 2: Personas (ensure input personas are stripped of trailing spaces just in case)
    df_knapsack[persona_col] = df_knapsack[persona_col].astype(str).str.strip()
    df_filtered = df_epc_filtered[df_epc_filtered[persona_col].isin(allowed_personas)].copy()

    if logger:
        logger.info(f"🔍 EPC Algo Start | Budget: £{budget:,.0f}")
        logger.info(f"  Input candidates: {len(df_knapsack)}")
        logger.info(f"  After EPC filter (D-G): {len(df_epc_filtered)}")
        logger.info(f"  After Persona filter: {len(df_filtered)}")
        
        if len(df_knapsack) > 0 and len(df_filtered) == 0:
             logger.warning(f"⚠️ ALL candidates filtered out! Check EPC values (found: {df_knapsack[epc_col].unique()}) and Personas.")

    if df_filtered.empty:
        return pd.DataFrame(), budget

    df_filtered = df_knapsack[
        (df_knapsack[epc_col].isin(['D', 'E', 'F', 'G'])) & 
        (df_knapsack[persona_col].isin(list_personas))
    ]
    
    if logger:
        logger.info(f"\n🔍 Starting random selection algorithm using EPCs:")
        logger.info(f"  Total interventions after filtering: {len(df_filtered):,}")
        logger.info(f"  Unique buildings: {df_filtered[uprn_col].nunique():,}")
        logger.info(f"  Available budget: £{budget:,.0f}")
    
    # Get unique UPRNs and shuffle them randomly
    unique_uprns = df_filtered[uprn_col].unique()
    np.random.shuffle(unique_uprns)
    
    selected_rows = []
    remaining_budget = budget
    total_spent = 0.0
    
    # Iterate through randomly ordered UPRNs
    for uprn in unique_uprns:
        # Get all interventions for this UPRN
        uprn_interventions = df_filtered[df_filtered[uprn_col] == uprn].copy()
        
        # Sort by efficiency (lower is better - less cost per ton CO2)
        # Drop NaN values in efficiency column to avoid issues
        uprn_interventions = uprn_interventions[
            uprn_interventions[efficiency_column].notna()
        ]
        
        if uprn_interventions.empty:
            continue
            
        uprn_interventions = uprn_interventions.sort_values(
            by=efficiency_column, 
            ascending=True
        )
        
        # Select the most cost-efficient intervention
        best_intervention = uprn_interventions.iloc[0]
        intervention_cost = best_intervention[cost_column]
        
        # Check if it fits in remaining budget
        if intervention_cost <= remaining_budget:
            selected_rows.append(best_intervention)
            remaining_budget -= intervention_cost
            total_spent += intervention_cost
            
            if logger and len(selected_rows) % 100 == 0:
                logger.info(f"  Progress: {len(selected_rows)} buildings selected, "
                          f"£{remaining_budget:,.0f} remaining")
        # If doesn't fit, skip this UPRN and continue
    
    # Create selected dataframe
    if selected_rows:
        selected_df = pd.DataFrame(selected_rows)
    else:
        selected_df = pd.DataFrame()
    
    # Log the results
    if logger:
        if not selected_df.empty:
            # This 'total_ton_co2_saved' column name is hardcoded based on
            # the context of the main script (RANK_COL_CO2_SAVED)
            total_co2 = selected_df['total_ton_co2_saved'].sum()
            
            logger.info("\n✅ Selection Complete:")
            logger.info(f"  Buildings covered: {len(selected_df):,}")
            logger.info(f"  Total spent: £{total_spent:,.0f} (Budget: £{budget:,.0f})")
            logger.info(f"  Total CO2 saved: {total_co2:,.2f} tons")
            if total_co2 > 0:
                logger.info(f"  Cost per ton CO2 (Achieved): £{total_spent/total_co2:,.2f}")
        else:
            logger.warning("\n⚠️ No interventions selected (budget may be insufficient for any single project)")
    
    return selected_df, remaining_budget