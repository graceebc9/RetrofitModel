import pandas as pd 


def load_scaled_gas_elec(): 
    df = pd.read_csv('/Users/gracecolverd/RetrofitModel/notebook/joint_energ_table.csv')
    return df