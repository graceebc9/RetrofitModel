import pandas as pd 


def join_three_pcds(df, df_col,  pc_df  , pcds_cols):
    # merge on any one of three columns in pc_map 
    final_d = [] 
    for col in pcds_cols:
        d = df.merge(pc_df , right_on = col, left_on = df_col  )
        final_d.append(d)
    # Concatenate the results
    merged_final = pd.concat(final_d ).drop_duplicates()
    
    if len(df) != len(merged_final):
        print('Warning: some postcodes not matched')
    return merged_final 
