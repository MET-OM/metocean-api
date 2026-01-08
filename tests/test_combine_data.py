from metocean_api.ts.ts_mod import combine_data

def test_combine_data():
    df_ts = combine_data(list_files=['tests/data/NORA3_atm_sub_lon1.32_lat53.324_20210101_20210331.csv',
                                     'tests/data/NORA3_wind_sub_lon1.32_lat53.324_20210101_20210131.csv'],
                                output_file=False)
    assert df_ts.shape == (2160, 21)
