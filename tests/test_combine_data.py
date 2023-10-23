from metocean_api.ts.ts_mod import combine_data

def test_combine_data():
    # Define TimeSeries-object
    df_ts = combine_data(list_files=['tests/data/NORA3_atm_sub_lon1.32_lat53.324_20210101_20210331.csv',
                                     'tests/data/NORA3_wind_sub_lon1.32_lat53.324_20210101_20210131.csv'],
                                output_file=False)
    if df_ts.shape == (2160, 21):
        pass
    else:
        raise ValueError("Shape is not 2160,21")





