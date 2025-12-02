import pandas as pd 


vuln_clusters = [0, 6]
middle_clustrs = [4, 7, 8]
upper_cl = [1, 2, 3, 5] 

mapping = {
    **{c: 'high_deprived' for c in vuln_clusters},
    **{c: 'med_deprived' for c in middle_clustrs},
    **{c: 'low_deprived' for c in upper_cl}
}

name_mapping = { 
    0: "Struggling Lone Parents",
    1: "Secure Established Families",
    2: "Solvent Working Households",
    3: "Modest Solo Dwellers",
    4: "At-Risk Singles",
    5: "Senior Citizens",
    6: "Isolated and Deprived",
    7: "Younger Strugglers",
    8: "The Squeezed Middle"
}

# 2. Apply the mapping to your DataFrame
# Assuming your DataFrame is named 'df' and the column is 'Cluster'


def load_personas():
    """Load persona/demographic data."""

    try:
        path = '/home/gb669/rds/hpc-work/energy_map/uk_postcode_clustering/3_single_runs/regional/NE/fuel_poverty_clustering_vars13_kn9/filt_domestic_samples/clusters_res_df.csv'
        personas = pd.read_csv(path)
    except:
        path = '/Volumes/T9/2025_10_RetrofitModel/3-personas/clusters_res_df (1).csv'    
        personas = pd.read_csv(path)
    print('Personas loaded')
    personas['meta_socio_persona'] = personas['cluster'].map(mapping)
    personas['persona_name'] = personas['cluster'].map(name_mapping)
    return personas 