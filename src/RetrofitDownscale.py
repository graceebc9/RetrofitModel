import pandas as pd 


def load_scaled_gas_elec(): 
    try: 
        df = pd.read_csv('/Users/gracecolverd/RetrofitModel/notebook/final_scaled_energy_tables.csv')
    except:
        df = pd.read_csv('/home/gb669/rds/hpc-work/energy_map/RetrofitModel/scaled_energy_data/final_scaled_energy_tables.csv')
    
    return df