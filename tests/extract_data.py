from metocean_api import ts



def test_extract_data():
    # Define TimeSeries-object
    df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2022-12-01', end_time='2022-12-31' , product='NORA3_wind_sub')

    # Import data from thredds.met.no and save it as csv
    df_ts.import_data(save_csv=False)

    print(df_ts.data)


