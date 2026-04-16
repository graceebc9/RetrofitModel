import pandas as pd 


high_risk = [0, 2,3 ]
med_risk = [10] 
middle_risk = [5 ,8]
low_risk = [7,9]
v_low_risk = [1, 4, 6, 11] 

mapping = {
    **{c: 'high_risk' for c in high_risk},
    **{c: 'med_risk' for c in med_risk},
    **{c: 'middle_risk' for c in middle_risk},
    **{c: 'low_risk' for c in low_risk},
    **{c: 'v_low_risk' for c in v_low_risk}
}

cluster_names = {
    0: "Diverse Urban Deprivation",
    1: "Affluent Suburbia",
    2: "Vulnerable Working",
    3: "Vulnerable Elderly",
    4: "Affluent Working Families",
    5: "Middle England Elderly",
    6: "Affluent Retirees",
    7: "Thriving Diverse Working Families",
    8: "Middle England Families",
    9: "Off-Grid Thriving",
    10: "Struggling urbanites",
    11: "Highly Affluent Suburbia"
}

cluster_risks = {
    0: "High Risk",
    1: "Very Low Risk",
    2: "High Risk",
    3: "High Risk",
    4: "Very Low Risk",
    5: "Middle Risk",
    6: "Very Low Risk",
    7: "Low Risk",
    8: "Middle Risk",
    9: "Low Risk",
    10: "Medium Risk",
    11: "Very Low Risk"
}


# 2. Apply the mapping to your DataFrame
# Assuming your DataFrame is named 'df' and the column is 'Cluster'


def load_personas():
    """Load persona/demographic data."""

    try:
        path = '/Users/gracecolverd/Downloads/final_results_clustering_for_paoper/fuel_poc_model14_kn12/clusters_res_df.csv'
        path = '/Users/gracecolverd/RetrofitModel/personas/clusters_res_df.csv'
       
        
        personas = pd.read_csv(path)
    except:
        try:
            path = '/rds/user/gb669/hpc-work/energy_map/RetrofitModel/personas/clusters_res_df.csv'
        except:
            print('error cant find personas')
    print('Personas loaded')
    personas['meta_socio_persona'] = personas['cluster'].map(mapping)
    personas['persona_name'] = personas['cluster'].map(cluster_names)
    return personas 