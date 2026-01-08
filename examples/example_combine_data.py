from metocean_api.ts.ts_mod import combine_data


# Combine several csv files:
df_ts = combine_data(list_files=['tests/data/NORA3_atm_sub_lon1.32_lat53.324_20210101_20210331.csv',
                                           'tests/data/NORA3_wind_sub_lon1.32_lat53.324_20210101_20210131.csv'],
                                output_file='combined_NORA3_lon1.32_lat53.324.csv')

print(df_ts)
